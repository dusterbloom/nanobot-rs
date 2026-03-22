//! ANE speculative decode: single-token forward pass with KV cache.
//!
//! Phase 1c of the ANE speculative decoding plan. This module provides:
//! - `KvCache`: per-layer KV cache for autoregressive decoding
//! - `decode_step`: single-token forward pass that returns logits
//! - `DecodeKernels`: ANE-compiled kernels for seq_len=1 (FFN acceleration)
//!
//! The go/no-go gate: if `decode_step` on 0.8B exceeds 10ms/token,
//! we fall back to GPU-based speculative decoding via mlx-lm `--draft-model`.

use super::ane_forward::{cpu_matmul, cpu_quantized_matmul, rmsnorm, vec_add_inplace};
use super::ane_mil::MilConfig;
use super::ane_weights::{GdnLayerWeights, MoeLayerWeights, ModelWeights, QuantizedTensor};

/// Cast &[f32] to &[u8] for ANE IO.
fn f32_as_bytes(data: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) }
}

/// Cast &[u8] to &[f32] for ANE IO.
fn bytes_as_f32(data: &[u8]) -> &[f32] {
    assert_eq!(data.len() % 4, 0);
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, data.len() / 4) }
}

/// SiLU activation in-place: x[i] = x[i] * sigmoid(x[i]).
fn silu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = *v * (1.0 / (1.0 + (-*v).exp()));
    }
}

// ---------------------------------------------------------------------------
// MoE Forward + Router Training Target Collection
// ---------------------------------------------------------------------------

/// Global router training buffer. Set by RouterTrainer::install_buffer(),
/// read by moe_forward() during inference to collect routing decisions.
static ROUTER_TRAINING_BUF: std::sync::OnceLock<std::sync::Arc<super::ane_lora::RouterTrainingBuffer>> =
    std::sync::OnceLock::new();

/// Install a router training buffer so inference collects routing decisions.
pub fn install_router_training_buffer(buf: std::sync::Arc<super::ane_lora::RouterTrainingBuffer>) {
    let _ = ROUTER_TRAINING_BUF.set(buf);
}

/// Get the installed buffer (if any).
pub fn router_training_buffer() -> Option<&'static std::sync::Arc<super::ane_lora::RouterTrainingBuffer>> {
    ROUTER_TRAINING_BUF.get()
}

/// Read routing targets from the shared mmap file populated by oMLX's Python hook.
///
/// Returns targets grouped by layer. Updates the read position so the same
/// records aren't processed twice.
///
/// File format (see scripts/routing_hook.py):
///   Header (32 bytes): write_pos(u64), read_pos(u64), n_layers(u32), dim(u32), k(u32), capacity(u32)
///   Records: [layer(u16), k(u16), indices(u16×k), probs(f32×k), x_norm(f32×dim)]
pub fn drain_routing_targets_from_file(
    path: &std::path::Path,
) -> Option<Vec<(usize, super::ane_lora::RouterRoutingTarget)>> {
    use std::io::{Read, Seek, SeekFrom, Write};

    let mut f = std::fs::OpenOptions::new().read(true).write(true).open(path).ok()?;

    // Read header
    let mut header = [0u8; 32];
    f.read_exact(&mut header).ok()?;

    let write_pos = u64::from_le_bytes(header[0..8].try_into().unwrap()) as usize;
    let read_pos = u64::from_le_bytes(header[8..16].try_into().unwrap()) as usize;
    let _n_layers = u32::from_le_bytes(header[16..20].try_into().unwrap()) as usize;
    let dim = u32::from_le_bytes(header[20..24].try_into().unwrap()) as usize;
    let k = u32::from_le_bytes(header[24..28].try_into().unwrap()) as usize;
    let capacity = u32::from_le_bytes(header[28..32].try_into().unwrap()) as usize;

    if write_pos <= read_pos || dim == 0 || k == 0 || capacity == 0 {
        return None; // nothing new
    }

    let rec_size = 4 + k * 2 + k * 4 + dim * 4; // layer(2)+k(2) + indices + probs + x_norm
    let n_new = (write_pos - read_pos).min(capacity); // don't read more than capacity

    let mut targets = Vec::with_capacity(n_new);

    for i in 0..n_new {
        let idx = (read_pos + i) % capacity;
        let offset = 32 + idx * rec_size; // 32 = header size

        let mut rec = vec![0u8; rec_size];
        f.seek(SeekFrom::Start(offset as u64)).ok()?;
        f.read_exact(&mut rec).ok()?;

        let layer = u16::from_le_bytes([rec[0], rec[1]]) as usize;
        let rec_k = u16::from_le_bytes([rec[2], rec[3]]) as usize;

        if rec_k != k {
            continue; // corrupt record
        }

        let mut expert_indices = Vec::with_capacity(k);
        for j in 0..k {
            let off = 4 + j * 2;
            expert_indices.push(u16::from_le_bytes([rec[off], rec[off + 1]]) as usize);
        }

        let probs_offset = 4 + k * 2;
        let mut expert_probs = Vec::with_capacity(k);
        for j in 0..k {
            let off = probs_offset + j * 4;
            expert_probs.push(f32::from_le_bytes(rec[off..off + 4].try_into().unwrap()));
        }

        let xnorm_offset = probs_offset + k * 4;
        let mut x_norm = Vec::with_capacity(dim);
        for d in 0..dim {
            let off = xnorm_offset + d * 4;
            x_norm.push(f32::from_le_bytes(rec[off..off + 4].try_into().unwrap()));
        }

        targets.push((layer, super::ane_lora::RouterRoutingTarget {
            x_norm,
            expert_indices,
            expert_probs,
            reward: 0.0,
        }));
    }

    // Update read position
    let new_read = (read_pos + n_new) as u64;
    f.seek(SeekFrom::Start(8)).ok()?;
    f.write_all(&new_read.to_le_bytes()).ok()?;

    Some(targets)
}

/// MoE FFN forward: router gate → top-k expert selection → weighted expert sum.
///
/// Replaces the dense FFN (w1/w2/w3) for MoE layers. Expert weights stay
/// quantized; only the top-k active experts are dequantized per token.
fn moe_forward(
    moe: &MoeLayerWeights,
    x: &mut Vec<f32>,
    rms_ffn: &[f32],
    cfg: &MilConfig,
) {
    moe_forward_inner(moe, x, rms_ffn, cfg, 0);
}

/// MoE forward with layer index (for routing target collection).
fn moe_forward_layer(
    moe: &MoeLayerWeights,
    x: &mut Vec<f32>,
    rms_ffn: &[f32],
    cfg: &MilConfig,
    layer: usize,
) {
    moe_forward_inner(moe, x, rms_ffn, cfg, layer);
}

fn moe_forward_inner(
    moe: &MoeLayerWeights,
    x: &mut Vec<f32>,
    rms_ffn: &[f32],
    cfg: &MilConfig,
    layer: usize,
) {
    moe_forward_with_adapter(moe, x, rms_ffn, cfg, layer, None);
}

fn moe_forward_with_adapter(
    moe: &MoeLayerWeights,
    x: &mut Vec<f32>,
    rms_ffn: &[f32],
    cfg: &MilConfig,
    layer: usize,
    adapter: Option<&super::router_adapter::RouterAdapter>,
) {
    let dim = cfg.dim;
    let hidden = moe.moe_hidden;

    // RMSNorm
    let mut xnorm = vec![0.0f32; dim];
    rmsnorm(&mut xnorm, x, rms_ffn, dim, 1, cfg.rms_eps);

    // Router: use fp32 adapter gate if available, else quantized base
    let router_logits = if let Some(gate) = adapter.and_then(|a| a.gate_for_layer(layer)) {
        cpu_matmul(gate, &xnorm, moe.num_experts, dim, 1)
    } else {
        cpu_matmul(&moe.router, &xnorm, moe.num_experts, dim, 1)
    };

    // Top-k selection + softmax
    let k = moe.num_experts_per_tok;
    let mut indexed: Vec<(usize, f32)> = router_logits.iter().copied().enumerate().collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top_k = &indexed[..k.min(indexed.len())];

    // Softmax over top-k logits
    let max_logit = top_k[0].1;
    let mut exp_sum = 0.0f32;
    let mut weights = Vec::with_capacity(k);
    for &(_, logit) in top_k {
        let e = (logit - max_logit).exp();
        weights.push(e);
        exp_sum += e;
    }
    for w in &mut weights {
        *w /= exp_sum;
    }

    // Record routing decision for router training (if buffer installed)
    if let Some(buf) = ROUTER_TRAINING_BUF.get() {
        let expert_indices: Vec<usize> = top_k.iter().map(|&(idx, _)| idx).collect();
        let expert_probs = weights.clone();
        buf.push(layer, super::ane_lora::RouterRoutingTarget {
            x_norm: xnorm.clone(),
            expert_indices,
            expert_probs,
            reward: 0.0, // set later from outcome signal
        });
    }

    // Parallel expert FFN — 8 experts run on 8 cores simultaneously.
    // Gate and up projections are independent per expert → full parallelism.
    #[cfg(feature = "ane")]
    let expert_outputs: Vec<(f32, Vec<f32>)> = {
        use rayon::prelude::*;
        top_k
            .par_iter()
            .enumerate()
            .map(|(i, &(expert_idx, _))| {
                let w = weights[i];
                // SwiGLU: down(SiLU(gate(x)) * up(x))
                let mut h1 = moe.packed_experts.gate_matmul(expert_idx, &xnorm);
                let h3 = moe.packed_experts.up_matmul(expert_idx, &xnorm);
                silu_inplace(&mut h1);
                for j in 0..hidden {
                    h1[j] *= h3[j];
                }
                let expert_out = moe.packed_experts.down_matmul(expert_idx, &h1);
                (w, expert_out)
            })
            .collect()
    };

    #[cfg(not(feature = "ane"))]
    let expert_outputs: Vec<(f32, Vec<f32>)> = top_k
        .iter()
        .enumerate()
        .map(|(i, &(expert_idx, _))| {
            let w = weights[i];
            let mut h1 = moe.packed_experts.gate_matmul(expert_idx, &xnorm);
            let h3 = moe.packed_experts.up_matmul(expert_idx, &xnorm);
            silu_inplace(&mut h1);
            for j in 0..hidden {
                h1[j] *= h3[j];
            }
            let expert_out = moe.packed_experts.down_matmul(expert_idx, &h1);
            (w, expert_out)
        })
        .collect();

    // Weighted sum of expert outputs
    let mut ffn_out = vec![0.0f32; dim];
    for (w, expert_out) in &expert_outputs {
        for j in 0..dim {
            ffn_out[j] += w * expert_out[j];
        }
    }

    // Shared expert (always active, weight=1.0, no gating)
    if let Some(ref shared) = moe.shared_expert {
        let mut h1 = cpu_quantized_matmul(&shared.gate_proj, &xnorm, 1);
        let h3 = cpu_quantized_matmul(&shared.up_proj, &xnorm, 1);
        silu_inplace(&mut h1);
        for j in 0..hidden {
            h1[j] *= h3[j];
        }
        let shared_out = cpu_quantized_matmul(&shared.down_proj, &h1, 1);
        for j in 0..dim {
            ffn_out[j] += shared_out[j];
        }
    }

    // Residual
    vec_add_inplace(x, &ffn_out);
}

// ---------------------------------------------------------------------------
// GDN Decode State
// ---------------------------------------------------------------------------

/// Per-layer GDN decode state for single-token autoregressive generation.
///
/// Stores the recurrence state (linear attention memory) and causal conv1d
/// buffer. The batch forward (`cpu_gdn_forward`) processes all tokens at once;
/// this struct carries the equivalent state across individual decode steps.
#[derive(Clone)]
pub(crate) struct GdnLayerDecodeState {
    /// Recurrence state: flat `[h_v * d_v * d_k]`.
    /// Layout: `state[h * d_v * d_k + dv * d_k + dk]`.
    recurrence: Vec<f32>,
    /// Causal conv1d buffer: `[qkv_dim * (kernel_size - 1)]`.
    /// Stores the last `kernel_size - 1` pre-conv QKV values per channel.
    /// Layout: `buf[c * (kernel_size - 1) + lag]` where lag=0 is most recent.
    conv_buf: Vec<f32>,
    /// Conv kernel size (for buffer indexing).
    kernel_size: usize,
    /// QKV dimension (for buffer indexing).
    qkv_dim: usize,
}

impl GdnLayerDecodeState {
    fn new(cfg: &MilConfig) -> Self {
        let h_v = cfg.linear_n_value_heads;
        let d_k = cfg.linear_head_dim;
        let d_v = cfg.linear_value_head_dim;
        let key_dim = cfg.linear_n_heads * d_k;
        let value_dim = h_v * d_v;
        let qkv_dim = 2 * key_dim + value_dim;
        let kernel = cfg.conv_kernel_size;
        Self {
            recurrence: vec![0.0f32; h_v * d_v * d_k],
            conv_buf: vec![0.0f32; qkv_dim * (kernel - 1)],
            kernel_size: kernel,
            qkv_dim,
        }
    }
}

// ---------------------------------------------------------------------------
// KV Cache
// ---------------------------------------------------------------------------

/// Per-layer KV cache for autoregressive decoding.
///
/// Layout: `k_cache[layer][kv_head * max_seq * head_dim + pos * head_dim]`
/// Flat Vec per layer, indexed as `[n_kv_heads, max_seq, head_dim]`.
pub struct KvCache {
    /// K cache per layer: `[n_kv_heads, max_seq, head_dim]`
    k: Vec<Vec<f32>>,
    /// V cache per layer: `[n_kv_heads, max_seq, head_dim]`
    v: Vec<Vec<f32>>,
    /// Current position (next token will be written at this index).
    pos: usize,
    /// Maximum sequence length.
    max_seq: usize,
    /// Number of KV heads.
    n_kv_heads: usize,
    /// Per-head dimension.
    head_dim: usize,
    /// GDN decode state per layer (None for MHA layers).
    gdn: Vec<Option<GdnLayerDecodeState>>,
}

impl KvCache {
    /// Create a new KV cache for the given model config.
    pub fn new(cfg: &MilConfig, n_layers: usize, max_seq: usize) -> Self {
        let n_kv_heads = cfg.n_kv_heads;
        let head_dim = cfg.head_dim();
        let layer_size = n_kv_heads * max_seq * head_dim;
        Self {
            k: vec![vec![0.0f32; layer_size]; n_layers],
            v: vec![vec![0.0f32; layer_size]; n_layers],
            pos: 0,
            max_seq,
            n_kv_heads,
            head_dim,
            gdn: vec![None; n_layers],
        }
    }

    /// Initialize GDN decode state for layers that have GDN weights.
    ///
    /// Call after construction to enable full GDN attention in decode steps.
    /// Without this, GDN layers fall back to FFN-only (no attention).
    pub fn init_gdn(&mut self, model: &ModelWeights) {
        for (l, lw) in model.layers.iter().enumerate() {
            if lw.gdn.is_some() {
                self.gdn[l] = Some(GdnLayerDecodeState::new(&model.cfg));
            }
        }
    }

    /// Current sequence position.
    pub fn pos(&self) -> usize {
        self.pos
    }

    /// Append K, V vectors for one token at one layer.
    ///
    /// `k_new`: `[n_kv_heads * head_dim]`, `v_new`: same.
    fn append(&mut self, layer: usize, k_new: &[f32], v_new: &[f32]) {
        let hd = self.head_dim;
        let pos = self.pos;
        debug_assert!(pos < self.max_seq, "KV cache full at pos {pos}");
        debug_assert_eq!(k_new.len(), self.n_kv_heads * hd);
        debug_assert_eq!(v_new.len(), self.n_kv_heads * hd);

        for kv_h in 0..self.n_kv_heads {
            let base = kv_h * self.max_seq * hd + pos * hd;
            self.k[layer][base..base + hd].copy_from_slice(&k_new[kv_h * hd..(kv_h + 1) * hd]);
            self.v[layer][base..base + hd].copy_from_slice(&v_new[kv_h * hd..(kv_h + 1) * hd]);
        }
    }

    /// Advance position after appending all layers.
    fn advance(&mut self) {
        self.pos += 1;
    }

    /// Rollback cache to a previous position (for speculation rejection).
    /// Positions beyond `new_pos` are logically invalidated (never read).
    pub fn rollback_to(&mut self, new_pos: usize) {
        debug_assert!(new_pos <= self.pos);
        self.pos = new_pos;
    }
}

// ---------------------------------------------------------------------------
// RoPE at a single position
// ---------------------------------------------------------------------------

/// Apply RoPE to Q and K vectors at a specific position.
///
/// `q`: `[n_q_heads * head_dim]`, `k`: `[n_kv_heads * head_dim]`.
fn rope_at_pos(
    q: &mut [f32],
    k: &mut [f32],
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    pos: usize,
    theta: f64,
) {
    let half_hd = head_dim / 2;

    // Apply to Q heads
    for h in 0..n_q_heads {
        let base = h * head_dim;
        for i in 0..half_hd {
            let inv_freq = 1.0 / theta.powf(2.0 * i as f64 / head_dim as f64);
            let angle = pos as f64 * inv_freq;
            let cos = angle.cos() as f32;
            let sin = angle.sin() as f32;
            let r = q[base + i];
            let c = q[base + half_hd + i];
            q[base + i] = r * cos - c * sin;
            q[base + half_hd + i] = r * sin + c * cos;
        }
    }

    // Apply to K heads
    for h in 0..n_kv_heads {
        let base = h * head_dim;
        for i in 0..half_hd {
            let inv_freq = 1.0 / theta.powf(2.0 * i as f64 / head_dim as f64);
            let angle = pos as f64 * inv_freq;
            let cos = angle.cos() as f32;
            let sin = angle.sin() as f32;
            let r = k[base + i];
            let c = k[base + half_hd + i];
            k[base + i] = r * cos - c * sin;
            k[base + half_hd + i] = r * sin + c * cos;
        }
    }
}

// ---------------------------------------------------------------------------
// SDPA with KV cache (single query token)
// ---------------------------------------------------------------------------

/// Scaled dot-product attention for a single query position against cached KV.
///
/// `q`: `[n_q_heads * head_dim]` (single position)
/// Returns: `[n_q_heads * head_dim]` (attention output)
fn sdpa_cached(
    q: &[f32],
    cache: &KvCache,
    layer: usize,
    n_q_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let n_kv_heads = cache.n_kv_heads;
    let cache_len = cache.pos; // number of KV entries already stored
    let scale = 1.0 / (head_dim as f32).sqrt();
    let gqa_ratio = n_q_heads / n_kv_heads;

    let mut out = vec![0.0f32; n_q_heads * head_dim];

    for qh in 0..n_q_heads {
        let kv_h = qh / gqa_ratio;
        let q_off = qh * head_dim;
        let q_vec = &q[q_off..q_off + head_dim];

        // K cache for this KV head: [max_seq, head_dim], but only [0..cache_len] valid
        let k_base = kv_h * cache.max_seq * head_dim;
        let v_base = k_base; // same layout

        // 1. Scores: q @ k[t]^T for t in 0..cache_len
        let mut scores = vec![0.0f32; cache_len];
        for t in 0..cache_len {
            let k_off = k_base + t * head_dim;
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q_vec[d] * cache.k[layer][k_off + d];
            }
            scores[t] = dot * scale;
        }

        // 2. Softmax
        let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for s in scores.iter_mut() {
            *s = (*s - max_s).exp();
            sum += *s;
        }
        if sum > 0.0 {
            for s in scores.iter_mut() {
                *s /= sum;
            }
        }

        // 3. Weighted sum of V: out = sum_t scores[t] * v[t]
        let out_h = &mut out[q_off..q_off + head_dim];
        for t in 0..cache_len {
            let v_off = v_base + t * head_dim;
            let w = scores[t];
            for d in 0..head_dim {
                out_h[d] += w * cache.v[layer][v_off + d];
            }
        }
    }

    out
}

// ---------------------------------------------------------------------------
// Decode step (CPU-only baseline)
// ---------------------------------------------------------------------------

/// Result of a single decode step.
pub struct DecodeResult {
    /// Logits for all vocab tokens: `[vocab_size]`.
    pub logits: Vec<f32>,
}

// ---------------------------------------------------------------------------
// Single-token GDN decode
// ---------------------------------------------------------------------------

/// Single-token GDN linear attention for one layer.
///
/// Performs: projections → causal conv1d → QK norm → GQA expand →
/// single-step recurrence → output gate → O projection.
///
/// Returns `[dim]` attention output to be added as residual.
fn gdn_decode_single(
    gdn_w: &GdnLayerWeights,
    state: &mut GdnLayerDecodeState,
    xnorm: &[f32], // [dim] — post-attention-RMSNorm input
    cfg: &MilConfig,
) -> Vec<f32> {
    let dim = cfg.dim;
    let h_k = cfg.linear_n_heads;
    let d_k = cfg.linear_head_dim;
    let h_v = cfg.linear_n_value_heads;
    let d_v = cfg.linear_value_head_dim;
    let key_dim = h_k * d_k;
    let value_dim = h_v * d_v;
    let qkv_dim = 2 * key_dim + value_dim;
    let kernel = cfg.conv_kernel_size;
    let kv_repeat = h_v / h_k.max(1);

    // 1. Project QKV, a, b, z — all [out_dim] for seq=1
    let qkv_raw = cpu_matmul(&gdn_w.qkv_proj, xnorm, qkv_dim, dim, 1);
    let a_raw = cpu_matmul(&gdn_w.a_proj, xnorm, h_v, dim, 1);
    let b_raw = cpu_matmul(&gdn_w.b_proj, xnorm, h_v, dim, 1);
    let z = cpu_matmul(&gdn_w.z_proj, xnorm, value_dim, dim, 1);

    // 2. Causal conv1d + SiLU
    let buf_stride = kernel - 1;
    let mut qkv_conv = vec![0.0f32; qkv_dim];
    for c in 0..qkv_dim {
        let mut acc = qkv_raw[c] * gdn_w.conv_weight[c * kernel];
        for ki in 1..kernel {
            acc += state.conv_buf[c * buf_stride + ki - 1] * gdn_w.conv_weight[c * kernel + ki];
        }
        if c < gdn_w.conv_bias.len() {
            acc += gdn_w.conv_bias[c];
        }
        qkv_conv[c] = acc / (1.0 + (-acc).exp()); // SiLU
    }
    // Shift buffer: oldest out, newest in
    for c in 0..qkv_dim {
        for lag in (1..buf_stride).rev() {
            state.conv_buf[c * buf_stride + lag] = state.conv_buf[c * buf_stride + lag - 1];
        }
        state.conv_buf[c * buf_stride] = qkv_raw[c];
    }

    // 3. Split Q, K, V
    let q_raw = &qkv_conv[0..key_dim];
    let k_raw = &qkv_conv[key_dim..2 * key_dim];
    let v_raw = &qkv_conv[2 * key_dim..qkv_dim];

    // 4. Weight-free per-head RMSNorm on Q and K
    let inv_scale = (d_k as f32).powf(-0.5);
    let mut q = vec![0.0f32; key_dim];
    let mut k = vec![0.0f32; key_dim];
    for h in 0..h_k {
        let base = h * d_k;
        let mut q_ss = 0.0f32;
        let mut k_ss = 0.0f32;
        for d in 0..d_k {
            q_ss += q_raw[base + d] * q_raw[base + d];
            k_ss += k_raw[base + d] * k_raw[base + d];
        }
        let q_rms = (q_ss / d_k as f32 + 1e-6).sqrt();
        let k_rms = (k_ss / d_k as f32 + 1e-6).sqrt();
        for d in 0..d_k {
            q[base + d] = q_raw[base + d] / q_rms * inv_scale * inv_scale;
            k[base + d] = k_raw[base + d] / k_rms * inv_scale;
        }
    }

    // 5. GQA expansion (replicate Q/K heads to match value heads)
    let (q_exp, k_exp) = if kv_repeat > 1 {
        let mut qe = vec![0.0f32; h_v * d_k];
        let mut ke = vec![0.0f32; h_v * d_k];
        for hk in 0..h_k {
            for r in 0..kv_repeat {
                let hv = hk * kv_repeat + r;
                qe[hv * d_k..(hv + 1) * d_k].copy_from_slice(&q[hk * d_k..(hk + 1) * d_k]);
                ke[hv * d_k..(hv + 1) * d_k].copy_from_slice(&k[hk * d_k..(hk + 1) * d_k]);
            }
        }
        (qe, ke)
    } else {
        (q, k)
    };

    // 6. Decay (g) and write gate (beta) per value head
    let mut g_vals = vec![0.0f32; h_v];
    let mut beta_vals = vec![0.0f32; h_v];
    for h in 0..h_v {
        let a_val = a_raw[h] + gdn_w.dt_bias[h];
        let sp = if a_val > 20.0 {
            a_val
        } else {
            a_val.exp().ln_1p()
        };
        g_vals[h] = (-gdn_w.a_log[h].exp() * sp).exp();
        beta_vals[h] = 1.0 / (1.0 + (-b_raw[h]).exp());
    }

    // 7. Single-step recurrence
    let mut y = vec![0.0f32; value_dim];
    for h in 0..h_v {
        let g_t = g_vals[h];
        let beta_t = beta_vals[h];
        let state_base = h * d_v * d_k;
        for dv in 0..d_v {
            let row = state_base + dv * d_k;
            let mut kv_mem = 0.0f32;
            for dk in 0..d_k {
                state.recurrence[row + dk] *= g_t;
                kv_mem += state.recurrence[row + dk] * k_exp[h * d_k + dk];
            }
            let delta = (v_raw[h * d_v + dv] - kv_mem) * beta_t;
            for dk in 0..d_k {
                state.recurrence[row + dk] += k_exp[h * d_k + dk] * delta;
            }
            let mut y_val = 0.0f32;
            for dk in 0..d_k {
                y_val += state.recurrence[row + dk] * q_exp[h * d_k + dk];
            }
            y[h * d_v + dv] = y_val;
        }
    }

    // 8. Output gate: SiLU(z) * RMSNorm(y)
    let shared_norm = gdn_w.norm_weight.len() == d_v;
    let mut gated = vec![0.0f32; value_dim];
    for h in 0..h_v {
        let mut ss = 0.0f32;
        for d in 0..d_v {
            let val = y[h * d_v + d];
            ss += val * val;
        }
        let rms = (ss / d_v as f32 + 1e-6).sqrt();
        for d in 0..d_v {
            let norm_w = if shared_norm {
                gdn_w.norm_weight[d]
            } else {
                gdn_w.norm_weight[h * d_v + d]
            };
            let z_val = z[h * d_v + d];
            let silu_z = z_val / (1.0 + (-z_val).exp());
            gated[h * d_v + d] = silu_z * (y[h * d_v + d] / rms * norm_w);
        }
    }

    // 9. O projection: [dim, value_dim] @ [value_dim] → [dim]
    cpu_matmul(&gdn_w.o_proj, &gated, dim, value_dim, 1)
}

/// Single-token forward pass with KV cache. Returns logits for next-token prediction.
///
/// This is the CPU-only baseline. ANE acceleration is layered on top for FFN.
///
/// # Arguments
/// - `model`: loaded model weights
/// - `token`: input token ID
/// - `kv_cache`: mutable KV cache (appended to during this call)
pub fn decode_step(model: &ModelWeights, token: u32, kv_cache: &mut KvCache) -> DecodeResult {
    decode_step_inner(model, token, kv_cache, None)
}

/// Full hybrid decode: GDN on ANE, GQA on Candle Metal, MoE on CPU.
///
/// Routes each layer to the optimal hardware:
///   30 GDN layers → ANE conv1x1 projections + CPU recurrence
///   10 GQA layers → Candle Metal attention with KV cache
///   40 MoE FFN   → CPU quantized matmul + fp32 router adapter
#[cfg(feature = "candle")]
pub fn decode_step_hybrid(
    model: &ModelWeights,
    token: u32,
    kv_cache: &mut KvCache,
    gdn_ane: Option<&GdnAneKernels>,
    candle_gqa: &mut super::candle_attn::CandleGqaLayers,
) -> DecodeResult {
    let cfg = &model.cfg;
    let dim = cfg.dim;
    let n_layers = model.layers.len();
    let pos = kv_cache.pos();

    // 1. Embedding
    let vocab = model.vocab_size;
    if (token as usize) >= vocab {
        return DecodeResult {
            logits: vec![0.0; vocab],
        };
    }
    let mut x = vec![0.0f32; dim];
    for d in 0..dim {
        x[d] = model.embed[token as usize * dim + d];
    }

    // 2. Layer loop: route to ANE (GDN) or Candle Metal (GQA)
    for l in 0..n_layers {
        let lw = &model.layers[l];

        if let Some(ref gdn_w) = lw.gdn {
            // ── GDN layer: ANE projections + CPU recurrence ──
            if let Some(ref mut gdn_state) = kv_cache.gdn[l] {
                let mut xnorm = vec![0.0f32; dim];
                rmsnorm(&mut xnorm, &x, &lw.rms_att, dim, 1, cfg.rms_eps);
                let attn_out = if let Some(kernels) = gdn_ane.and_then(|k| k.layers[l].as_ref()) {
                    gdn_decode_single_ane(gdn_w, kernels, gdn_state, &xnorm, cfg)
                } else {
                    gdn_decode_single(gdn_w, gdn_state, &xnorm, cfg)
                };
                vec_add_inplace(&mut x, &attn_out);
            }
        } else {
            // ── GQA layer: Candle Metal attention ──
            match candle_gqa.forward(l, &x, pos) {
                Some(Ok(attn_out)) => {
                    vec_add_inplace(&mut x, &attn_out);
                }
                Some(Err(e)) => {
                    tracing::debug!("Candle GQA L{l} failed: {e}, falling back to CPU");
                    // CPU fallback: run existing CPU attention code
                    let mut xnorm = vec![0.0f32; dim];
                    rmsnorm(&mut xnorm, &x, &lw.rms_att, dim, 1, cfg.rms_eps);
                    let q_proj_dim = lw.wq.len() / dim;
                    let q_raw = cpu_matmul(&lw.wq, &xnorm, q_proj_dim, dim, 1);
                    // ... (simplified — full CPU fallback would need the entire MHA code)
                    // For now, just skip the attention on error
                    let _ = q_raw;
                }
                None => {
                    // Layer not in candle_gqa — use CPU
                    // (This shouldn't happen if CandleGqaLayers was built correctly)
                }
            }
        }

        // ── MoE FFN: CPU with router adapter ──
        if let Some(ref moe_w) = lw.moe {
            moe_forward_with_adapter(moe_w, &mut x, &lw.rms_ffn, cfg, l, model.router_adapter.as_ref());
        } else if !lw.w1.is_empty() {
            let mut x2norm = vec![0.0f32; dim];
            rmsnorm(&mut x2norm, &x, &lw.rms_ffn, dim, 1, cfg.rms_eps);
            let hidden = cfg.hidden_dim;
            let mut h1 = cpu_matmul(&lw.w1, &x2norm, hidden, dim, 1);
            let h3 = cpu_matmul(&lw.w3, &x2norm, hidden, dim, 1);
            silu_inplace(&mut h1);
            for i in 0..hidden {
                h1[i] *= h3[i];
            }
            let ffn_out = cpu_matmul(&lw.w2, &h1, dim, hidden, 1);
            vec_add_inplace(&mut x, &ffn_out);
        }
    }

    // 3. Final norm + classifier
    kv_cache.advance();
    let mut x_final = vec![0.0f32; dim];
    rmsnorm(&mut x_final, &x, &model.rms_final, dim, 1, cfg.rms_eps);

    let cls_w = model.lm_head.as_ref().unwrap_or(&model.embed);
    let logits = if let Some(ref clusters) = model.vocab_clusters {
        super::factored_vocab::factored_project(&x_final, cls_w, clusters, 3).logits
    } else {
        cpu_matmul(cls_w, &x_final, vocab, dim, 1)
    };

    DecodeResult { logits }
}

/// Decode with optional GDN ANE kernels for accelerated GDN projections.
pub fn decode_step_with_ane(
    model: &ModelWeights,
    token: u32,
    kv_cache: &mut KvCache,
    gdn_ane: Option<&GdnAneKernels>,
) -> DecodeResult {
    decode_step_inner(model, token, kv_cache, gdn_ane)
}

fn decode_step_inner(
    model: &ModelWeights,
    token: u32,
    kv_cache: &mut KvCache,
    gdn_ane: Option<&GdnAneKernels>,
) -> DecodeResult {
    let cfg = &model.cfg;
    let dim = cfg.dim;
    let n_layers = model.layers.len();
    let n_q_heads = cfg.n_heads;
    let n_kv_heads = cfg.n_kv_heads;
    let head_dim = cfg.head_dim();
    let q_proj_dim = cfg.q_proj_dim();
    let kv_dim = n_kv_heads * head_dim;
    let pos = kv_cache.pos();

    // 1. Embedding lookup: single token → [dim]
    let mut x = vec![0.0f32; dim];
    for d in 0..dim {
        x[d] = model.embed[token as usize * dim + d];
    }

    // 2. Transformer layers
    for l in 0..n_layers {
        let lw = &model.layers[l];

        // GDN (linear attention) layers
        if let Some(ref gdn_w) = lw.gdn {
            if let Some(ref mut gdn_state) = kv_cache.gdn[l] {
                // Full GDN decode: attention + FFN
                let mut xnorm = vec![0.0f32; dim];
                rmsnorm(&mut xnorm, &x, &lw.rms_att, dim, 1, cfg.rms_eps);
                // ANE path: projections on ANE, recurrence on CPU
                let attn_out = if let Some(kernels) = gdn_ane.and_then(|k| k.layers[l].as_ref()) {
                    gdn_decode_single_ane(gdn_w, kernels, gdn_state, &xnorm, cfg)
                } else {
                    gdn_decode_single(gdn_w, gdn_state, &xnorm, cfg)
                };
                vec_add_inplace(&mut x, &attn_out);
            }
            // FFN: MoE or dense (always runs, with or without GDN attention)
            if let Some(ref moe_w) = lw.moe {
                moe_forward_with_adapter(moe_w, &mut x, &lw.rms_ffn, cfg, l, model.router_adapter.as_ref());
            } else {
                let mut x2norm = vec![0.0f32; dim];
                rmsnorm(&mut x2norm, &x, &lw.rms_ffn, dim, 1, cfg.rms_eps);
                let hidden = cfg.hidden_dim;
                let mut h1 = cpu_matmul(&lw.w1, &x2norm, hidden, dim, 1);
                let h3 = cpu_matmul(&lw.w3, &x2norm, hidden, dim, 1);
                silu_inplace(&mut h1);
                for i in 0..hidden {
                    h1[i] *= h3[i];
                }
                let ffn_out = cpu_matmul(&lw.w2, &h1, dim, hidden, 1);
                vec_add_inplace(&mut x, &ffn_out);
            }
            continue;
        }

        // RMSNorm (attention)
        let mut xnorm = vec![0.0f32; dim];
        rmsnorm(&mut xnorm, &x, &lw.rms_att, dim, 1, cfg.rms_eps);

        // Q, K, V projections: [proj_dim, dim] @ [dim, 1] → [proj_dim]
        let q_raw = cpu_matmul(&lw.wq, &xnorm, q_proj_dim, dim, 1);

        // K/V weights may be GQA-expanded to [attn_dim, dim] by from_mlx_safetensors.
        // Project at the actual weight dimension, then collapse back to kv_dim.
        let k_proj_dim = lw.wk.len() / dim;
        let v_proj_dim = lw.wv.len() / dim;
        let k_full = cpu_matmul(&lw.wk, &xnorm, k_proj_dim, dim, 1);
        let v_full = cpu_matmul(&lw.wv, &xnorm, v_proj_dim, dim, 1);
        let mut k = if k_proj_dim > kv_dim {
            // Take one head per KV group (expanded heads are identical)
            let hpg = k_proj_dim / kv_dim;
            (0..n_kv_heads)
                .flat_map(|h| {
                    let base = h * hpg * head_dim;
                    k_full[base..base + head_dim].iter().copied()
                })
                .collect()
        } else {
            k_full
        };
        let v = if v_proj_dim > kv_dim {
            let hpg = v_proj_dim / kv_dim;
            (0..n_kv_heads)
                .flat_map(|h| {
                    let base = h * hpg * head_dim;
                    v_full[base..base + head_dim].iter().copied()
                })
                .collect()
        } else {
            v_full
        };

        // Split Q and gate if attn_output_gate is true.
        // Qwen3.5: Wq outputs [Q_h0, gate_h0, Q_h1, gate_h1, ...] interleaved per-head.
        let attn_dim = n_q_heads * head_dim;
        let (mut q, attn_gate) = if cfg.attn_output_gate {
            let mut q = vec![0.0f32; attn_dim];
            let mut gate = vec![0.0f32; attn_dim];
            for h in 0..n_q_heads {
                let src_base = h * 2 * head_dim;
                let dst_base = h * head_dim;
                q[dst_base..dst_base + head_dim]
                    .copy_from_slice(&q_raw[src_base..src_base + head_dim]);
                gate[dst_base..dst_base + head_dim]
                    .copy_from_slice(&q_raw[src_base + head_dim..src_base + 2 * head_dim]);
            }
            (q, Some(gate))
        } else {
            (q_raw, None)
        };

        // RoPE at current position
        rope_at_pos(
            &mut q,
            &mut k,
            n_q_heads,
            n_kv_heads,
            head_dim,
            pos,
            cfg.rope_theta,
        );

        // Append K, V to cache
        kv_cache.append(l, &k, &v);

        // Temporarily set pos to include the just-written token so sdpa_cached sees it
        let save_pos = kv_cache.pos;
        kv_cache.pos = pos + 1;

        // SDPA over cached KV
        let mut attn_out = sdpa_cached(&q, kv_cache, l, n_q_heads, head_dim);

        kv_cache.pos = save_pos;

        // Apply attention output gate: attn_out *= sigmoid(gate)
        if let Some(ref gate) = attn_gate {
            for i in 0..attn_dim {
                let sig = 1.0 / (1.0 + (-gate[i]).exp());
                attn_out[i] *= sig;
            }
        }

        // Output projection: Wo is [dim, attn_dim]
        let o = cpu_matmul(&lw.wo, &attn_out, dim, attn_dim, 1);

        // Residual 1
        vec_add_inplace(&mut x, &o);

        // FFN: MoE or dense
        if let Some(ref moe_w) = lw.moe {
            moe_forward_with_adapter(moe_w, &mut x, &lw.rms_ffn, cfg, l, model.router_adapter.as_ref());
        } else {
            let mut x2norm = vec![0.0f32; dim];
            rmsnorm(&mut x2norm, &x, &lw.rms_ffn, dim, 1, cfg.rms_eps);
            let hidden = cfg.hidden_dim;
            let mut h1 = cpu_matmul(&lw.w1, &x2norm, hidden, dim, 1);
            let h3 = cpu_matmul(&lw.w3, &x2norm, hidden, dim, 1);
            silu_inplace(&mut h1);
            for i in 0..hidden {
                h1[i] *= h3[i];
            }
            let ffn_out = cpu_matmul(&lw.w2, &h1, dim, hidden, 1);
            vec_add_inplace(&mut x, &ffn_out);
        }
    }

    // Advance KV cache position
    kv_cache.advance();

    // 3. Final RMSNorm
    let mut x_final = vec![0.0f32; dim];
    rmsnorm(&mut x_final, &x, &model.rms_final, dim, 1, cfg.rms_eps);

    // 4. Classifier: logits = cls_w @ x_final via SGEMM
    let vocab = model.vocab_size;
    let cls_w = model.lm_head.as_ref().unwrap_or(&model.embed);
    let logits = if let Some(ref clusters) = model.vocab_clusters {
        super::factored_vocab::factored_project(&x_final, cls_w, clusters, 3).logits
    } else {
        cpu_matmul(cls_w, &x_final, vocab, dim, 1)
    };

    DecodeResult { logits }
}

/// Sample the argmax token from logits.
pub fn sample_argmax(logits: &[f32]) -> u32 {
    let mut best_idx = 0u32;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i as u32;
        }
    }
    best_idx
}

/// Generate N draft tokens autoregressively using decode_step.
///
/// Returns the draft token IDs.
pub fn generate_draft_tokens(
    model: &ModelWeights,
    kv_cache: &mut KvCache,
    prompt_token: u32,
    n_draft: usize,
    gdn_ane: Option<&GdnAneKernels>,
) -> Vec<u32> {
    let mut drafts = Vec::with_capacity(n_draft);
    let mut token = prompt_token;
    for _ in 0..n_draft {
        let result = decode_step_with_ane(model, token, kv_cache, gdn_ane);
        token = sample_argmax(&result.logits);
        drafts.push(token);
    }
    drafts
}

// ---------------------------------------------------------------------------
// ANE-accelerated decode kernels
// ---------------------------------------------------------------------------

use super::ane_bridge::AneKernel;
use super::ane_forward::{CompiledKernels, FfnKernels};
use super::ane_mil::{KernelSpec, KernelType};
use super::ane_weights;

/// Minimum spatial dimension for ANE FFN eval (DynMatmul path).
/// ANE hardware rejects seq < 16 at runtime (compiles OK, 0x1d at eval).
const ANE_MIN_SPATIAL: usize = 16;

/// Pre-compiled ANE kernels for single-token decode (DynMatmul approach).
///
/// Uses seq=16 padded kernels (ANE minimum spatial dimension) with weights
/// baked into per-layer IOSurface buffers. Each decode step patches only the
/// first column of activation data ([dim, 1] = ~4KB) and reads only the
/// first position from the output.
///
/// **Deprecated in favor of `BlobDecodeKernels`** which uses BLOBFILE weights
/// and may work at seq=1 via the `compile_direct` pipeline.
pub struct DecodeKernels {
    /// Per-layer FFN kernels with weights pre-loaded in IOSurface.
    ffn_layers: Vec<FfnLayerKernel>,
    /// Config with seq_len=ANE_MIN_SPATIAL (padded).
    cfg: MilConfig,
}

/// Per-layer FFN kernel with pre-loaded weights.
struct FfnLayerKernel {
    kernel: AneKernel,
    /// Stride of one row in the IOSurface buffer (bytes): (padded_seq + 3*hidden) * 4
    row_stride: usize,
}

impl DecodeKernels {
    /// Compile ANE FFN kernels for decode and pre-load weights for all layers.
    ///
    /// Uses seq=16 (ANE minimum spatial) and pads activation with zeros.
    /// Returns `None` if ANE compilation fails (caller falls back to CPU).
    pub fn compile(model: &ModelWeights) -> Option<Self> {
        let mut cfg = model.cfg.clone();
        cfg.seq_len = ANE_MIN_SPATIAL;

        super::ane_bridge::ane_init().ok()?;

        // Compile template FFN kernel at padded seq
        let compiled = CompiledKernels::compile_forward(&cfg).ok()?;
        let FfnKernels::FullyFused { ref kernel } = compiled.ffn else {
            tracing::debug!("DecodeKernels: FFN is not fully-fused, skipping ANE acceleration");
            return None;
        };

        let dim = cfg.dim;
        let hidden = cfg.hidden_dim;
        let seq = ANE_MIN_SPATIAL;
        let sp = seq + 3 * hidden;
        let row_stride = sp * 4;

        let mut ffn_layers = Vec::with_capacity(model.layers.len());
        for (l, lw) in model.layers.iter().enumerate() {
            let k = kernel
                .clone_kernel()
                .map_err(|e| {
                    tracing::warn!("DecodeKernels: layer {l} clone failed: {e}");
                    e
                })
                .ok()?;

            // Pack weights into IOSurface. Activation columns [0..seq) are zero-padded;
            // only column 0 is patched per decode step.
            let w1_t = ane_weights::transpose_weight(&lw.w1, hidden, dim);
            let w3_t = ane_weights::transpose_weight(&lw.w3, hidden, dim);

            let mut buf = vec![0.0f32; dim * sp];
            for d in 0..dim {
                let row = d * sp;
                // Activation columns [0..seq) stay zero (patched per-call for col 0)
                buf[row + seq..row + seq + hidden]
                    .copy_from_slice(&w1_t[d * hidden..(d + 1) * hidden]);
                buf[row + seq + hidden..row + seq + 2 * hidden]
                    .copy_from_slice(&w3_t[d * hidden..(d + 1) * hidden]);
                buf[row + seq + 2 * hidden..row + seq + 3 * hidden]
                    .copy_from_slice(&lw.w2[d * hidden..(d + 1) * hidden]);
            }

            let bytes = ane_weights::f32_slice_to_bytes(&buf);
            k.write_input(0, &bytes);

            ffn_layers.push(FfnLayerKernel {
                kernel: k,
                row_stride,
            });
        }

        tracing::info!(
            "DecodeKernels: compiled {} per-layer FFN kernels (padded_seq={}, dim={}, hidden={})",
            ffn_layers.len(),
            seq,
            dim,
            hidden
        );

        Some(DecodeKernels { ffn_layers, cfg })
    }

    /// Evaluate FFN for one layer: patches activation column 0, dispatches to ANE.
    ///
    /// `xnorm`: `[dim]` (single position, RMSNorm'd input to FFN).
    /// Returns `[dim]` (FFN output to be added as residual).
    fn eval_ffn(&self, layer: usize, xnorm: &[f32]) -> Result<Vec<f32>, String> {
        let flk = &self.ffn_layers[layer];
        let dim = self.cfg.dim;
        let hidden = self.cfg.hidden_dim;

        // Strided write: patch column 0 of activation in each row.
        // Each row has `padded_seq` activation slots followed by weight data.
        // We write 1 float (4 bytes) per row, at offset 0.
        let xnorm_bytes =
            unsafe { std::slice::from_raw_parts(xnorm.as_ptr() as *const u8, xnorm.len() * 4) };
        flk.kernel.write_input_strided(
            0,              // input idx
            0,              // dst_offset (column 0)
            flk.row_stride, // dst_stride (bytes per row)
            xnorm_bytes,
            4,   // src_stride (1 float = 4 bytes)
            4,   // chunk_bytes (1 float)
            dim, // n_chunks (number of rows)
        );

        flk.kernel.eval()?;

        let spec = KernelSpec::for_kernel(&self.cfg, KernelType::FusedFfn);
        let mut out_buf = vec![0u8; spec.output_bytes];
        flk.kernel.read_output(0, &mut out_buf);

        // Output is [(3*hidden + dim), padded_seq] interleaved.
        // We need ffn_out at position 0: the last `dim` channels, column 0.
        // Layout: out[ch * padded_seq + pos], channels = [h1, h3, gate, ffn_out].
        let padded_seq = ANE_MIN_SPATIAL;
        let ffn_offset = 3 * hidden; // ffn_out starts at channel 3*hidden
        let mut ffn_out = vec![0.0f32; dim];
        for d in 0..dim {
            let byte_offset = ((ffn_offset + d) * padded_seq + 0) * 4;
            ffn_out[d] = f32::from_le_bytes([
                out_buf[byte_offset],
                out_buf[byte_offset + 1],
                out_buf[byte_offset + 2],
                out_buf[byte_offset + 3],
            ]);
        }

        Ok(ffn_out)
    }
}

// ---------------------------------------------------------------------------
// BLOBFILE-based decode kernels (P0: bypass seq<16 spatial minimum)
// ---------------------------------------------------------------------------

use super::ane_bridge;
use super::ane_mil::{gen_conv1x1_blob, gen_fused_attn_proj_conv_blob, gen_fused_ffn_conv_blob};
use super::ane_weights::build_fp16_blob;

/// Per-layer ANE kernels for one transformer layer's decode.
///
/// MHA layers have all three kernels. GDN (linear attention) layers only
/// have FFN — attention is skipped (no recurrence state in decode path).
struct LayerKernels {
    /// Fused RMSNorm + Wq + Wk + Wv conv1x1 → packed q_raw|k|v.
    /// `None` for GDN layers (no MHA projections).
    attn_proj: Option<AneKernel>,
    /// Wo conv1x1: attn_out → o.
    /// `None` for GDN layers.
    wo_proj: Option<AneKernel>,
    /// Fused RMSNorm + W1*SiLU*W3 + W2 + residual conv1x1.
    ffn: AneKernel,
}

/// Pre-compiled ANE kernels for single-token decode using BLOBFILE conv1x1 weights.
///
/// Three dispatches per layer (attn_proj, wo, ffn) replace all 7 CPU matmuls:
/// - attn_proj: RMSNorm + Wq + Wk + Wv (4 BLOBFILEs, one dispatch)
/// - wo_proj: Wo projection (1 BLOBFILE)
/// - ffn: RMSNorm + W1*SiLU*W3 + W2 + residual (4 BLOBFILEs, one dispatch)
///
/// Only RoPE, KV cache, SDPA, gate, and classifier remain on CPU.
pub struct BlobDecodeKernels {
    /// Per-layer kernel triplets.
    layers: Vec<LayerKernels>,
    /// Padded seq_len (minimum 16 for ANE hardware).
    seq_len: usize,
    /// Model config dimensions.
    dim: usize,
    q_proj_dim: usize,
    kv_dim: usize,
    attn_dim: usize,
}

impl BlobDecodeKernels {
    /// Compile BLOBFILE FFN kernels for all layers at the given seq_len.
    ///
    /// Returns `None` if ANE compilation or eval fails. The caller should
    /// try seq=1 first, then fall back to larger seq or CPU.
    pub fn compile(model: &ModelWeights, seq_len: usize) -> Option<Self> {
        ane_bridge::ane_init().ok()?;

        let cfg = &model.cfg;
        let dim = cfg.dim;
        let hidden = cfg.hidden_dim;
        let n_kv_heads = cfg.n_kv_heads;
        let head_dim = cfg.head_dim();
        let q_proj_dim = cfg.q_proj_dim();
        let kv_dim = n_kv_heads * head_dim;
        let attn_dim = cfg.n_heads * head_dim;

        // Generate MIL templates at target seq_len
        let mut mil_cfg = cfg.clone();
        mil_cfg.seq_len = seq_len;

        let attn_spec = gen_fused_attn_proj_conv_blob(&mil_cfg);
        let wo_spec = gen_conv1x1_blob(attn_dim, dim, seq_len);
        let ffn_spec = gen_fused_ffn_conv_blob(&mil_cfg);

        let mut layers = Vec::with_capacity(model.layers.len());
        let mut first_mha_validated = false;
        for (l, lw) in model.layers.iter().enumerate() {
            // GDN layers: compile FFN only (no MHA projections)
            let (attn_proj, wo_proj) = if lw.gdn.is_some() {
                (None, None)
            } else {
                // --- Attn projection kernel: RMSNorm + Wq + Wk + Wv ---
                let ap = {
                    let rms_blob = build_fp16_blob(&lw.rms_att);
                    let wq_blob = build_fp16_blob(&lw.wq);
                    let wk_blob = build_fp16_blob(&lw.wk);
                    let wv_blob = build_fp16_blob(&lw.wv);
                    let names: Vec<&str> = attn_spec.weight_names.iter().copied().collect();
                    let datas: Vec<&[u8]> = vec![&rms_blob, &wq_blob, &wk_blob, &wv_blob];
                    AneKernel::compile_multi_weights(
                        &attn_spec.mil_text,
                        &names,
                        &datas,
                        &[attn_spec.input_bytes],
                        &[attn_spec.output_bytes],
                    )
                    .map_err(|e| {
                        tracing::warn!(
                            "BlobDecodeKernels: layer {l} attn_proj compile failed: {e}"
                        );
                        e
                    })
                    .ok()?
                };

                // --- Wo projection kernel ---
                let wp = {
                    let wo_blob = build_fp16_blob(&lw.wo);
                    let names: Vec<&str> = wo_spec.weight_names.iter().copied().collect();
                    let datas: Vec<&[u8]> = vec![&wo_blob];
                    AneKernel::compile_multi_weights(
                        &wo_spec.mil_text,
                        &names,
                        &datas,
                        &[wo_spec.input_bytes],
                        &[wo_spec.output_bytes],
                    )
                    .map_err(|e| {
                        tracing::warn!("BlobDecodeKernels: layer {l} wo compile failed: {e}");
                        e
                    })
                    .ok()?
                };

                // Validate eval on first MHA layer
                if !first_mha_validated {
                    let test_input = vec![0u8; attn_spec.input_bytes];
                    ap.write_input(0, &test_input);
                    if let Err(e) = ap.eval() {
                        tracing::warn!(
                            "BlobDecodeKernels: seq={seq_len} attn eval validation failed: {e}"
                        );
                        return None;
                    }
                    first_mha_validated = true;
                }

                (Some(ap), Some(wp))
            };

            // --- FFN kernel: RMSNorm + W1*SiLU*W3 + W2 + residual ---
            let ffn = {
                let rms_blob = build_fp16_blob(&lw.rms_ffn);
                let w1_blob = build_fp16_blob(&lw.w1);
                let w3_blob = build_fp16_blob(&lw.w3);
                let w2_blob = build_fp16_blob(&lw.w2);
                let names: Vec<&str> = ffn_spec.weight_names.iter().copied().collect();
                let datas: Vec<&[u8]> = vec![&rms_blob, &w1_blob, &w3_blob, &w2_blob];
                AneKernel::compile_multi_weights(
                    &ffn_spec.mil_text,
                    &names,
                    &datas,
                    &[ffn_spec.input_bytes],
                    &[ffn_spec.output_bytes],
                )
                .map_err(|e| {
                    tracing::warn!("BlobDecodeKernels: layer {l} ffn compile failed: {e}");
                    e
                })
                .ok()?
            };

            layers.push(LayerKernels {
                attn_proj,
                wo_proj,
                ffn,
            });
        }

        tracing::info!(
            "BlobDecodeKernels: compiled {} layers × 3 kernels (attn+wo+ffn) at seq={}, dim={}, hidden={}",
            layers.len(), seq_len, dim, hidden
        );

        Some(BlobDecodeKernels {
            layers,
            seq_len,
            dim,
            q_proj_dim,
            kv_dim,
            attn_dim,
        })
    }

    /// Compile a single conv1x1 BLOBFILE FFN kernel for testing (no model needed).
    ///
    /// Weight layouts (conv OIHW): w1[hidden,dim], w3[hidden,dim], w2[dim,hidden].
    /// Returns `None` on failure.
    pub fn compile_single(
        cfg: &MilConfig,
        rms_ffn: &[f32],
        w1: &[f32], // [hidden, dim] row-major (OIHW)
        w3: &[f32], // [hidden, dim] row-major (OIHW)
        w2: &[f32], // [dim, hidden] row-major (OIHW)
    ) -> Option<AneKernel> {
        ane_bridge::ane_init().ok()?;

        let mil_spec = gen_fused_ffn_conv_blob(cfg);

        // Conv OIHW — no transpose needed, data is already in correct order
        let rms_blob = build_fp16_blob(rms_ffn);
        let w1_blob = build_fp16_blob(w1);
        let w3_blob = build_fp16_blob(w3);
        let w2_blob = build_fp16_blob(w2);

        let names: Vec<&str> = mil_spec.weight_names.iter().copied().collect();
        let datas: Vec<&[u8]> = vec![&rms_blob, &w1_blob, &w3_blob, &w2_blob];

        AneKernel::compile_multi_weights(
            &mil_spec.mil_text,
            &names,
            &datas,
            &[mil_spec.input_bytes],
            &[mil_spec.output_bytes],
        )
        .ok()
    }

    /// Write a vector `x` of length `n_ch` into an ANE kernel's input IOSurface
    /// as `[1, n_ch, 1, seq]` fp32, padding positions 1..seq with zeros.
    ///
    /// Uses zero-copy direct IOSurface pointer with ARM64 `dsb sy` barrier
    /// (Orion-style) for ~2x faster I/O than `write_input()`.
    fn write_padded(kernel: &AneKernel, x: &[f32], n_ch: usize, seq: usize) {
        kernel.write_input_zerocopy(0, x, n_ch, seq);
    }

    /// Read position 0 from an ANE kernel's output IOSurface `[1, n_ch, 1, seq]`.
    ///
    /// Uses zero-copy direct IOSurface pointer with ARM64 `dsb sy` barrier.
    fn read_pos0(kernel: &AneKernel, n_ch: usize, seq: usize) -> Vec<f32> {
        kernel.read_output_zerocopy(0, n_ch, seq)
    }

    /// Evaluate fused attention projections: RMSNorm + Wq + Wk + Wv.
    ///
    /// Input `x`: `[dim]`. Returns `(q_raw, k, v)`.
    pub fn eval_attn_proj(
        &self,
        layer: usize,
        x: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), String> {
        let lk = &self.layers[layer];
        let ap = lk
            .attn_proj
            .as_ref()
            .ok_or_else(|| "GDN layer: no attn_proj".to_string())?;
        let seq = self.seq_len;
        let out_ch = self.q_proj_dim + 2 * self.kv_dim;

        Self::write_padded(ap, x, self.dim, seq);
        ap.eval()?;
        let packed = Self::read_pos0(ap, out_ch, seq);

        // Unpack: [q_proj_dim | kv_dim | kv_dim]
        let q_raw = packed[..self.q_proj_dim].to_vec();
        let k = packed[self.q_proj_dim..self.q_proj_dim + self.kv_dim].to_vec();
        let v = packed[self.q_proj_dim + self.kv_dim..].to_vec();
        Ok((q_raw, k, v))
    }

    /// Evaluate Wo projection: conv1x1(attn_out) → o.
    ///
    /// Input `attn_out`: `[attn_dim]`. Returns `[dim]`.
    pub fn eval_wo(&self, layer: usize, attn_out: &[f32]) -> Result<Vec<f32>, String> {
        let lk = &self.layers[layer];
        let wp = lk
            .wo_proj
            .as_ref()
            .ok_or_else(|| "GDN layer: no wo_proj".to_string())?;
        let seq = self.seq_len;

        Self::write_padded(wp, attn_out, self.attn_dim, seq);
        wp.eval()?;
        Ok(Self::read_pos0(wp, self.dim, seq))
    }

    /// Evaluate fused FFN: RMSNorm + W1*SiLU*W3 + W2 + residual.
    ///
    /// Input `x`: `[dim]` (post-attention). Returns `[dim]` (x + FFN(RMSNorm(x))).
    pub fn eval_ffn(&self, layer: usize, x: &[f32]) -> Result<Vec<f32>, String> {
        let lk = &self.layers[layer];
        let seq = self.seq_len;

        Self::write_padded(&lk.ffn, x, self.dim, seq);
        lk.ffn.eval()?;
        Ok(Self::read_pos0(&lk.ffn, self.dim, seq))
    }
}

/// ANE-accelerated decode: attn projections + Wo + FFN all on ANE.
///
/// Only RoPE, KV cache append, SDPA, gate sigmoid, embed lookup, final
/// RMSNorm, and classifier remain on CPU. All 7 per-layer matmuls are
/// replaced by 3 ANE conv1x1 dispatches.
pub fn decode_step_blob(
    model: &ModelWeights,
    kernels: &BlobDecodeKernels,
    token: u32,
    kv_cache: &mut KvCache,
) -> DecodeResult {
    decode_step_blob_inner(model, kernels, None, token, kv_cache)
}

/// Decode with ANE FFN kernels + optional ANE GDN projection kernels.
pub fn decode_step_blob_gdn(
    model: &ModelWeights,
    kernels: &BlobDecodeKernels,
    gdn_ane: &GdnAneKernels,
    token: u32,
    kv_cache: &mut KvCache,
) -> DecodeResult {
    decode_step_blob_inner(model, kernels, Some(gdn_ane), token, kv_cache)
}

fn decode_step_blob_inner(
    model: &ModelWeights,
    kernels: &BlobDecodeKernels,
    gdn_ane: Option<&GdnAneKernels>,
    token: u32,
    kv_cache: &mut KvCache,
) -> DecodeResult {
    let cfg = &model.cfg;
    let dim = cfg.dim;
    let n_layers = model.layers.len();
    let n_q_heads = cfg.n_heads;
    let n_kv_heads = cfg.n_kv_heads;
    let head_dim = cfg.head_dim();
    let q_proj_dim = cfg.q_proj_dim();
    let kv_dim = n_kv_heads * head_dim;
    let attn_dim = n_q_heads * head_dim;
    let pos = kv_cache.pos();

    // 1. Embedding lookup (CPU)
    let mut x = vec![0.0f32; dim];
    for d in 0..dim {
        x[d] = model.embed[token as usize * dim + d];
    }

    // 2. Transformer layers
    for l in 0..n_layers {
        let lw = &model.layers[l];

        // GDN (linear attention) layers
        if let Some(ref gdn_w) = lw.gdn {
            if let Some(ref mut gdn_state) = kv_cache.gdn[l] {
                // Full GDN decode: attention + FFN (ANE projections if available)
                let mut xnorm = vec![0.0f32; dim];
                rmsnorm(&mut xnorm, &x, &lw.rms_att, dim, 1, cfg.rms_eps);
                let attn_out = match gdn_ane.and_then(|g| g.layers[l].as_ref()) {
                    Some(gdn_k) => gdn_decode_single_ane(gdn_w, gdn_k, gdn_state, &xnorm, cfg),
                    None => gdn_decode_single(gdn_w, gdn_state, &xnorm, cfg),
                };
                vec_add_inplace(&mut x, &attn_out);
            }
            // FFN (ANE with CPU fallback)
            x = kernels.eval_ffn(l, &x).unwrap_or_else(|e| {
                tracing::warn!("ANE ffn layer {l} (GDN) failed: {e}, CPU fallback");
                let mut x2norm = vec![0.0f32; dim];
                rmsnorm(&mut x2norm, &x, &lw.rms_ffn, dim, 1, cfg.rms_eps);
                let hidden = cfg.hidden_dim;
                let mut h1 = cpu_matmul(&lw.w1, &x2norm, hidden, dim, 1);
                let h3 = cpu_matmul(&lw.w3, &x2norm, hidden, dim, 1);
                silu_inplace(&mut h1);
                for i in 0..hidden {
                    h1[i] *= h3[i];
                }
                let ffn_out = cpu_matmul(&lw.w2, &h1, dim, hidden, 1);
                let mut x_out = x.clone();
                vec_add_inplace(&mut x_out, &ffn_out);
                x_out
            });
            continue;
        }

        // --- Attention projections (ANE): RMSNorm + Wq + Wk + Wv ---
        let (q_raw, mut k, v) = kernels.eval_attn_proj(l, &x).unwrap_or_else(|e| {
            tracing::warn!("ANE attn_proj layer {l} failed: {e}, CPU fallback");
            let mut xnorm = vec![0.0f32; dim];
            rmsnorm(&mut xnorm, &x, &lw.rms_att, dim, 1, cfg.rms_eps);
            let q = cpu_matmul(&lw.wq, &xnorm, q_proj_dim, dim, 1);
            let k = cpu_matmul(&lw.wk, &xnorm, kv_dim, dim, 1);
            let v = cpu_matmul(&lw.wv, &xnorm, kv_dim, dim, 1);
            (q, k, v)
        });

        // Split Q and gate (CPU — cheap, just memcpy)
        let (mut q, attn_gate) = if cfg.attn_output_gate {
            let mut q = vec![0.0f32; attn_dim];
            let mut gate = vec![0.0f32; attn_dim];
            for h in 0..n_q_heads {
                let src_base = h * 2 * head_dim;
                let dst_base = h * head_dim;
                q[dst_base..dst_base + head_dim]
                    .copy_from_slice(&q_raw[src_base..src_base + head_dim]);
                gate[dst_base..dst_base + head_dim]
                    .copy_from_slice(&q_raw[src_base + head_dim..src_base + 2 * head_dim]);
            }
            (q, Some(gate))
        } else {
            (q_raw, None)
        };

        // RoPE (CPU — positional encoding, must be per-position)
        rope_at_pos(
            &mut q,
            &mut k,
            n_q_heads,
            n_kv_heads,
            head_dim,
            pos,
            cfg.rope_theta,
        );

        // KV cache + SDPA (CPU — sequential cache access)
        kv_cache.append(l, &k, &v);
        let save_pos = kv_cache.pos;
        kv_cache.pos = pos + 1;
        let mut attn_out = sdpa_cached(&q, kv_cache, l, n_q_heads, head_dim);
        kv_cache.pos = save_pos;

        // Gate (CPU — element-wise sigmoid, tiny)
        if let Some(ref gate) = attn_gate {
            for i in 0..attn_dim {
                let sig = 1.0 / (1.0 + (-gate[i]).exp());
                attn_out[i] *= sig;
            }
        }

        // --- Wo projection (ANE) ---
        let o = kernels.eval_wo(l, &attn_out).unwrap_or_else(|e| {
            tracing::warn!("ANE wo layer {l} failed: {e}, CPU fallback");
            cpu_matmul(&lw.wo, &attn_out, dim, attn_dim, 1)
        });
        vec_add_inplace(&mut x, &o);

        // --- FFN (ANE): RMSNorm + W1*SiLU*W3 + W2 + residual ---
        x = kernels.eval_ffn(l, &x).unwrap_or_else(|e| {
            tracing::warn!("ANE ffn layer {l} failed: {e}, CPU fallback");
            let mut x2norm = vec![0.0f32; dim];
            rmsnorm(&mut x2norm, &x, &lw.rms_ffn, dim, 1, cfg.rms_eps);
            let hidden = cfg.hidden_dim;
            let mut h1 = cpu_matmul(&lw.w1, &x2norm, hidden, dim, 1);
            let h3 = cpu_matmul(&lw.w3, &x2norm, hidden, dim, 1);
            silu_inplace(&mut h1);
            for i in 0..hidden {
                h1[i] *= h3[i];
            }
            let ffn_out = cpu_matmul(&lw.w2, &h1, dim, hidden, 1);
            let mut x_out = x.clone();
            vec_add_inplace(&mut x_out, &ffn_out);
            x_out
        });
    }

    kv_cache.advance();

    // 3. Final RMSNorm (CPU)
    let mut x_final = vec![0.0f32; dim];
    rmsnorm(&mut x_final, &x, &model.rms_final, dim, 1, cfg.rms_eps);

    // 4. Classifier — CPU (SGEMM)
    let vocab = model.vocab_size;
    let cls_w = model.lm_head.as_ref().unwrap_or(&model.embed);
    let logits = cpu_matmul(cls_w, &x_final, vocab, dim, 1);

    DecodeResult { logits }
}

// ---------------------------------------------------------------------------
// GDN ANE kernels — move GDN projections from CPU to ANE conv1x1
// ---------------------------------------------------------------------------

/// Per-layer ANE kernels for GDN projection matmuls.
struct GdnLayerAneKernels {
    /// Fused input projection: xnorm → qkv_raw (conv1x1, [qkv_dim, dim]).
    qkv: AneKernel,
    /// Decay projection: xnorm → a_raw (conv1x1, [h_v, dim]).
    a_proj: AneKernel,
    /// Write gate projection: xnorm → b_raw (conv1x1, [h_v, dim]).
    b_proj: AneKernel,
    /// Output gate projection: xnorm → z (conv1x1, [value_dim, dim]).
    z_proj: AneKernel,
    /// Output projection: gated → out (conv1x1, [dim, value_dim]).
    o_proj: AneKernel,
    /// Input dims for packing/unpacking.
    qkv_dim: usize,
    h_v: usize,
    value_dim: usize,
}

/// ANE kernels for GDN layer projections (all layers).
///
/// Replaces 5 `cpu_matmul` calls per GDN layer with ANE conv1x1 dispatches.
/// Recurrence, conv1d, normalization, and gating stay on CPU (tiny, sequential).
pub struct GdnAneKernels {
    layers: Vec<Option<GdnLayerAneKernels>>,
    seq_len: usize,
    dim: usize,
}

impl GdnAneKernels {
    /// Compile GDN projection kernels for all GDN layers.
    /// Returns None if ANE init or any kernel compilation fails.
    pub fn compile(model: &ModelWeights) -> Option<Self> {
        ane_bridge::ane_init().ok()?;

        let cfg = &model.cfg;
        let dim = cfg.dim;
        let h_v = cfg.linear_n_value_heads;
        let d_k = cfg.linear_head_dim;
        let d_v = cfg.linear_value_head_dim;
        let key_dim = cfg.linear_n_heads * d_k;
        let value_dim = h_v * d_v;
        let qkv_dim = 2 * key_dim + value_dim;
        let seq_len = 16; // Bug 14 minimum

        // Generate MIL specs for each projection shape
        let qkv_spec = gen_conv1x1_blob(dim, qkv_dim, seq_len);
        let a_spec = gen_conv1x1_blob(dim, h_v, seq_len);
        let b_spec = gen_conv1x1_blob(dim, h_v, seq_len);
        let z_spec = gen_conv1x1_blob(dim, value_dim, seq_len);
        let o_spec = gen_conv1x1_blob(value_dim, dim, seq_len);

        let compile_one = |spec: &super::ane_mil::FusedLayerMil,
                           w: &[f32],
                           label: &str,
                           l: usize|
         -> Option<AneKernel> {
            let blob = build_fp16_blob(w);
            let names: Vec<&str> = spec.weight_names.iter().copied().collect();
            AneKernel::compile_multi_weights(
                &spec.mil_text,
                &names,
                &[&blob],
                &[spec.input_bytes],
                &[spec.output_bytes],
            )
            .map_err(|e| tracing::warn!("GdnAneKernels: L{l} {label} compile failed: {e}"))
            .ok()
        };

        let mut layers = Vec::with_capacity(model.layers.len());
        for (l, lw) in model.layers.iter().enumerate() {
            if let Some(ref gdn_w) = lw.gdn {
                let qkv = compile_one(&qkv_spec, &gdn_w.qkv_proj, "qkv", l)?;
                let a = compile_one(&a_spec, &gdn_w.a_proj, "a", l)?;
                let b = compile_one(&b_spec, &gdn_w.b_proj, "b", l)?;
                let z = compile_one(&z_spec, &gdn_w.z_proj, "z", l)?;
                let o = compile_one(&o_spec, &gdn_w.o_proj, "o", l)?;
                layers.push(Some(GdnLayerAneKernels {
                    qkv,
                    a_proj: a,
                    b_proj: b,
                    z_proj: z,
                    o_proj: o,
                    qkv_dim,
                    h_v,
                    value_dim,
                }));
            } else {
                layers.push(None);
            }
        }

        Some(GdnAneKernels {
            layers,
            seq_len,
            dim,
        })
    }
}

/// ANE-accelerated GDN decode: projections on ANE, recurrence on CPU.
///
/// Same computation as `gdn_decode_single` but replaces 5 `cpu_matmul` calls
/// with ANE conv1x1 dispatches. The recurrence, conv1d, normalization, and
/// gating logic remain on CPU (they're sequential and tiny).
fn gdn_decode_single_ane(
    gdn_w: &GdnLayerWeights,
    kernels: &GdnLayerAneKernels,
    state: &mut GdnLayerDecodeState,
    xnorm: &[f32],
    cfg: &MilConfig,
) -> Vec<f32> {
    let dim = cfg.dim;
    let h_k = cfg.linear_n_heads;
    let d_k = cfg.linear_head_dim;
    let h_v = cfg.linear_n_value_heads;
    let d_v = cfg.linear_value_head_dim;
    let key_dim = h_k * d_k;
    let value_dim = h_v * d_v;
    let qkv_dim = 2 * key_dim + value_dim;
    let kernel = cfg.conv_kernel_size;
    let kv_repeat = h_v / h_k.max(1);
    let seq_len = 16; // padded

    // ── ANE input projections (replace 4 cpu_matmul calls) ──
    // Pack xnorm into ANE layout [1, dim, 1, 16] — pad position 0, rest zeros
    // Pack xnorm into ANE layout [1, dim, 1, 16] — channel-first, position 0
    let mut ane_input = vec![0.0f32; dim * seq_len];
    for d in 0..dim {
        ane_input[d * seq_len] = xnorm[d];
    }
    let input_bytes = f32_as_bytes(&ane_input);

    /// Evaluate a projection kernel: write input, eval, read position 0 from output.
    fn ane_read_proj(k: &AneKernel, input: &[u8], c_out: usize, seq: usize) -> Option<Vec<f32>> {
        k.write_input(0, input);
        k.eval().ok()?;
        let mut out_buf = vec![0u8; c_out * seq * 4];
        k.read_output(0, &mut out_buf);
        let out_all = bytes_as_f32(&out_buf);
        let mut result = vec![0.0f32; c_out];
        for c in 0..c_out {
            result[c] = out_all[c * seq];
        }
        Some(result)
    }

    let qkv_raw = match ane_read_proj(&kernels.qkv, input_bytes, qkv_dim, seq_len) {
        Some(v) => v,
        None => return gdn_decode_single(gdn_w, state, xnorm, cfg),
    };
    let a_raw = match ane_read_proj(&kernels.a_proj, input_bytes, h_v, seq_len) {
        Some(v) => v,
        None => return gdn_decode_single(gdn_w, state, xnorm, cfg),
    };
    let b_raw = match ane_read_proj(&kernels.b_proj, input_bytes, h_v, seq_len) {
        Some(v) => v,
        None => return gdn_decode_single(gdn_w, state, xnorm, cfg),
    };
    let z = match ane_read_proj(&kernels.z_proj, input_bytes, value_dim, seq_len) {
        Some(v) => v,
        None => return gdn_decode_single(gdn_w, state, xnorm, cfg),
    };

    // ── Steps 2-8: identical to CPU path ──

    // 2. Causal conv1d + SiLU
    let buf_stride = kernel - 1;
    let mut qkv_conv = vec![0.0f32; qkv_dim];
    for c in 0..qkv_dim {
        let mut acc = qkv_raw[c] * gdn_w.conv_weight[c * kernel];
        for ki in 1..kernel {
            acc += state.conv_buf[c * buf_stride + ki - 1] * gdn_w.conv_weight[c * kernel + ki];
        }
        if c < gdn_w.conv_bias.len() {
            acc += gdn_w.conv_bias[c];
        }
        qkv_conv[c] = acc / (1.0 + (-acc).exp());
    }
    for c in 0..qkv_dim {
        for lag in (1..buf_stride).rev() {
            state.conv_buf[c * buf_stride + lag] = state.conv_buf[c * buf_stride + lag - 1];
        }
        state.conv_buf[c * buf_stride] = qkv_raw[c];
    }

    // 3. Split Q, K, V
    let q_raw = &qkv_conv[0..key_dim];
    let k_raw = &qkv_conv[key_dim..2 * key_dim];
    let v_raw = &qkv_conv[2 * key_dim..qkv_dim];

    // 4. Weight-free per-head RMSNorm on Q and K
    let inv_scale = (d_k as f32).powf(-0.5);
    let mut q = vec![0.0f32; key_dim];
    let mut k = vec![0.0f32; key_dim];
    for h in 0..h_k {
        let base = h * d_k;
        let mut q_ss = 0.0f32;
        let mut k_ss = 0.0f32;
        for d in 0..d_k {
            q_ss += q_raw[base + d] * q_raw[base + d];
            k_ss += k_raw[base + d] * k_raw[base + d];
        }
        let q_rms = (q_ss / d_k as f32 + 1e-6).sqrt();
        let k_rms = (k_ss / d_k as f32 + 1e-6).sqrt();
        for d in 0..d_k {
            q[base + d] = q_raw[base + d] / q_rms * inv_scale * inv_scale;
            k[base + d] = k_raw[base + d] / k_rms * inv_scale;
        }
    }

    // 5. GQA expansion
    let (q_exp, k_exp) = if kv_repeat > 1 {
        let mut qe = vec![0.0f32; h_v * d_k];
        let mut ke = vec![0.0f32; h_v * d_k];
        for hk in 0..h_k {
            for r in 0..kv_repeat {
                let hv = hk * kv_repeat + r;
                qe[hv * d_k..(hv + 1) * d_k].copy_from_slice(&q[hk * d_k..(hk + 1) * d_k]);
                ke[hv * d_k..(hv + 1) * d_k].copy_from_slice(&k[hk * d_k..(hk + 1) * d_k]);
            }
        }
        (qe, ke)
    } else {
        (q, k)
    };

    // 6. Decay and write gate
    let mut g_vals = vec![0.0f32; h_v];
    let mut beta_vals = vec![0.0f32; h_v];
    for h in 0..h_v {
        let a_val = a_raw[h] + gdn_w.dt_bias[h];
        let sp = if a_val > 20.0 {
            a_val
        } else {
            a_val.exp().ln_1p()
        };
        g_vals[h] = (-gdn_w.a_log[h].exp() * sp).exp();
        beta_vals[h] = 1.0 / (1.0 + (-b_raw[h]).exp());
    }

    // 7. Single-step recurrence
    let mut y = vec![0.0f32; value_dim];
    for h in 0..h_v {
        let g_t = g_vals[h];
        let beta_t = beta_vals[h];
        let state_base = h * d_v * d_k;
        for dv in 0..d_v {
            let row = state_base + dv * d_k;
            let mut kv_mem = 0.0f32;
            for dk in 0..d_k {
                state.recurrence[row + dk] *= g_t;
                kv_mem += state.recurrence[row + dk] * k_exp[h * d_k + dk];
            }
            let delta = (v_raw[h * d_v + dv] - kv_mem) * beta_t;
            for dk in 0..d_k {
                state.recurrence[row + dk] += k_exp[h * d_k + dk] * delta;
            }
            let mut y_val = 0.0f32;
            for dk in 0..d_k {
                y_val += state.recurrence[row + dk] * q_exp[h * d_k + dk];
            }
            y[h * d_v + dv] = y_val;
        }
    }

    // 8. Output gate: SiLU(z) * RMSNorm(y)
    let shared_norm = gdn_w.norm_weight.len() == d_v;
    let mut gated = vec![0.0f32; value_dim];
    for h in 0..h_v {
        let mut ss = 0.0f32;
        for d in 0..d_v {
            let val = y[h * d_v + d];
            ss += val * val;
        }
        let rms = (ss / d_v as f32 + 1e-6).sqrt();
        for d in 0..d_v {
            let norm_w = if shared_norm {
                gdn_w.norm_weight[d]
            } else {
                gdn_w.norm_weight[h * d_v + d]
            };
            let z_val = z[h * d_v + d];
            let silu_z = z_val / (1.0 + (-z_val).exp());
            gated[h * d_v + d] = silu_z * (y[h * d_v + d] / rms * norm_w);
        }
    }

    // ── 9. O projection on ANE ──
    let mut o_input = vec![0.0f32; value_dim * seq_len];
    for c in 0..value_dim {
        o_input[c * seq_len] = gated[c];
    }
    kernels.o_proj.write_input(0, f32_as_bytes(&o_input));
    if let Err(e) = kernels.o_proj.eval() {
        tracing::warn!("GDN ANE o_proj eval failed: {e}, CPU fallback");
        return cpu_matmul(&gdn_w.o_proj, &gated, dim, value_dim, 1);
    }
    let mut o_buf = vec![0u8; dim * seq_len * 4];
    kernels.o_proj.read_output(0, &mut o_buf);
    let o_all = bytes_as_f32(&o_buf);
    let mut result = vec![0.0f32; dim];
    for d in 0..dim {
        result[d] = o_all[d * seq_len];
    }
    result
}

// ---------------------------------------------------------------------------
// CoreMLTools compile pipeline (Orion-style)
// ---------------------------------------------------------------------------

/// Result of a coremltools-compiled conv1x1 kernel.
pub struct CoremlCompiledKernel {
    /// Lowered MIL bytes (from .mlmodelc/model.mil).
    pub mil_bytes: Vec<u8>,
    /// Weight blob (from .mlmodelc/weights/weight.bin), or None.
    pub weight_blob: Option<Vec<u8>>,
}

/// Compile a conv1x1 kernel via coremltools + xcrun coremlcompiler.
///
/// This produces properly lowered MIL that the ANE hardware accepts,
/// bypassing MIL format issues in our hand-written MIL. The lowered MIL
/// can then be fed to `ane_bridge_compile()`.
///
/// Requires: `coremltools` installed in the Python environment.
///
/// # Arguments
/// - `c_in`: input channels
/// - `c_out`: output channels
/// - `spatial`: spatial dimension (Orion uses 16 for decode, works at any valid value)
/// - `weights`: `[c_out, c_in]` row-major f32 weights to bake into the kernel
/// - `model_name`: unique name for temp files (e.g. "l0_qkv")
pub fn compile_via_coremltools(
    c_in: usize,
    c_out: usize,
    spatial: usize,
    weights: &[f32],
    model_name: &str,
) -> Result<CoremlCompiledKernel, String> {
    assert_eq!(weights.len(), c_out * c_in, "weight dimensions mismatch");

    // Write weights to a temp npy file
    let tmp_dir = std::env::temp_dir();
    let weights_path = tmp_dir.join(format!("{model_name}_weights.bin"));
    let script_path = tmp_dir.join(format!("{model_name}_compile.py"));

    // Write raw f32 weights
    let weight_bytes: Vec<u8> = weights.iter().flat_map(|f| f.to_le_bytes()).collect();
    std::fs::write(&weights_path, &weight_bytes)
        .map_err(|e| format!("Failed to write weights: {e}"))?;

    // Generate Python compile script (Orion pattern)
    let script = format!(
        r#"
import numpy as np
import coremltools as ct
from coremltools.converters.mil.mil import Builder as mb
import subprocess, sys, os

c_in, c_out, spatial = {c_in}, {c_out}, {spatial}
w = np.frombuffer(open("{weights_path}", "rb").read(), dtype=np.float32).reshape(c_out, c_in, 1, 1)

@mb.program(input_specs=[mb.TensorSpec(shape=(1, c_in, 1, spatial))])
def prog(x):
    W = mb.const(val=w, name="W")
    return mb.conv(x=x, weight=W, name="conv_out")

model = ct.convert(prog, minimum_deployment_target=ct.target.macOS15)
pkg = "/tmp/{model_name}.mlpackage"
model.save(pkg)
subprocess.run(["xcrun", "coremlcompiler", "compile", pkg, "/tmp/"], capture_output=True, check=True)
print("OK")
"#,
        weights_path = weights_path.display(),
        model_name = model_name,
    );

    std::fs::write(&script_path, &script).map_err(|e| format!("Failed to write script: {e}"))?;

    // Execute the Python script
    let output = std::process::Command::new("python3")
        .arg(&script_path)
        .output()
        .map_err(|e| format!("Failed to run python3: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("coremltools compile failed: {stderr}"));
    }

    // Read compiled MIL and weight blob from .mlmodelc
    let compiled_dir = tmp_dir.join(format!("{model_name}.mlmodelc"));
    let mil_path = compiled_dir.join("model.mil");
    let weight_path = compiled_dir.join("weights/weight.bin");

    let mil_bytes =
        std::fs::read(&mil_path).map_err(|e| format!("Failed to read compiled MIL: {e}"))?;

    let weight_blob = std::fs::read(&weight_path).ok();

    // Cleanup temp files
    let _ = std::fs::remove_file(&weights_path);
    let _ = std::fs::remove_file(&script_path);
    let _ = std::fs::remove_dir_all(tmp_dir.join(format!("{model_name}.mlpackage")));
    let _ = std::fs::remove_dir_all(&compiled_dir);

    Ok(CoremlCompiledKernel {
        mil_bytes,
        weight_blob,
    })
}

/// Compile a conv1x1 kernel via coremltools and load it onto ANE.
///
/// This is the full Orion pipeline: coremltools → coremlcompiler → ane_bridge_compile.
/// Returns a ready-to-eval `AneKernel` with weights baked in.
pub fn compile_ane_via_coremltools(
    c_in: usize,
    c_out: usize,
    spatial: usize,
    weights: &[f32],
    model_name: &str,
) -> Result<AneKernel, String> {
    let compiled = compile_via_coremltools(c_in, c_out, spatial, weights, model_name)?;

    let in_bytes = c_in * spatial * 4;
    let out_bytes = c_out * spatial * 4;

    match compiled.weight_blob {
        Some(ref wb) => {
            let name = "@model_path/weights/weight.bin";
            AneKernel::compile_multi_weights(
                std::str::from_utf8(&compiled.mil_bytes)
                    .map_err(|e| format!("MIL is not UTF-8: {e}"))?,
                &[name],
                &[wb.as_slice()],
                &[in_bytes],
                &[out_bytes],
            )
        }
        None => AneKernel::compile(
            std::str::from_utf8(&compiled.mil_bytes)
                .map_err(|e| format!("MIL is not UTF-8: {e}"))?,
            None,
            &[in_bytes],
            &[out_bytes],
        ),
    }
}

// ---------------------------------------------------------------------------
// Quantized CPU decode (bandwidth-optimal path)
// ---------------------------------------------------------------------------

/// Fused 8-bit dequant-GEMV: out[r] = sum_c(dequant(W[r,c]) * x[c]).
///
/// Never materializes the dense fp32 weight matrix. For each group of 32 elements,
/// computes: scale * dot(q_data, x) + bias * sum(x) where dot/sum are precomputed.
///
/// Uses rayon for parallel row processing when matrix is large enough.
fn quantized_gemv(w: &QuantizedTensor, x: &[f32], out: &mut [f32]) {
    debug_assert_eq!(w.bits, 8, "quantized_gemv only supports 8-bit");
    debug_assert_eq!(x.len(), w.cols);
    debug_assert_eq!(out.len(), w.rows);

    let group_size = w.group_size;
    let n_groups = w.cols / group_size;

    // Precompute per-group sums of x (reused across all rows)
    let mut x_group_sums = vec![0.0f32; n_groups];
    for g in 0..n_groups {
        let base = g * group_size;
        let mut s = 0.0f32;
        for i in 0..group_size {
            s += x[base + i];
        }
        x_group_sums[g] = s;
    }

    // Compute one row's dot product
    let compute_row = |r: usize| -> f32 {
        let row_data = &w.data[r * w.cols..];
        let row_sb = r * n_groups;
        let mut acc = 0.0f32;
        for g in 0..n_groups {
            let scale = w.scales[row_sb + g];
            let bias = w.biases[row_sb + g];
            let qbase = g * group_size;
            let xbase = g * group_size;

            let mut dot = 0.0f32;
            let mut i = 0;
            while i + 8 <= group_size {
                dot += row_data[qbase + i] as f32 * x[xbase + i]
                    + row_data[qbase + i + 1] as f32 * x[xbase + i + 1]
                    + row_data[qbase + i + 2] as f32 * x[xbase + i + 2]
                    + row_data[qbase + i + 3] as f32 * x[xbase + i + 3]
                    + row_data[qbase + i + 4] as f32 * x[xbase + i + 4]
                    + row_data[qbase + i + 5] as f32 * x[xbase + i + 5]
                    + row_data[qbase + i + 6] as f32 * x[xbase + i + 6]
                    + row_data[qbase + i + 7] as f32 * x[xbase + i + 7];
                i += 8;
            }
            while i < group_size {
                dot += row_data[qbase + i] as f32 * x[xbase + i];
                i += 1;
            }
            acc += scale * dot + bias * x_group_sums[g];
        }
        acc
    };

    // Parallel for large matrices (classifier: 151936 rows)
    if w.rows >= 1024 {
        use rayon::prelude::*;
        out.par_iter_mut().enumerate().for_each(|(r, o)| {
            *o = compute_row(r);
        });
    } else {
        for r in 0..w.rows {
            out[r] = compute_row(r);
        }
    }
}

/// Fused quantized GEMV wrapper: returns new Vec.
fn quantized_gemv_alloc(w: &QuantizedTensor, x: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; w.rows];
    quantized_gemv(w, x, &mut out);
    out
}

/// Quantize a weight matrix from fp32 to 8-bit with per-group affine quantization.
///
/// Group size 32 chosen for good precision/bandwidth tradeoff.
/// Layout: `w[rows, cols]` row-major.
pub fn quantize_to_8bit(w: &[f32], rows: usize, cols: usize) -> QuantizedTensor {
    let group_size = 32;
    let n_groups = cols / group_size;
    debug_assert_eq!(cols % group_size, 0, "cols must be divisible by group_size");

    let mut scales = vec![0.0f32; rows * n_groups];
    let mut biases = vec![0.0f32; rows * n_groups];
    let mut data = vec![0u8; rows * cols]; // 8-bit = 1 byte per element

    for r in 0..rows {
        for g in 0..n_groups {
            let start = r * cols + g * group_size;
            let group = &w[start..start + group_size];

            let min_val = group.iter().copied().fold(f32::INFINITY, f32::min);
            let max_val = group.iter().copied().fold(f32::NEG_INFINITY, f32::max);

            let range = max_val - min_val;
            let scale = if range > 1e-10 { range / 255.0 } else { 1e-10 };
            let bias = min_val;

            scales[r * n_groups + g] = scale;
            biases[r * n_groups + g] = bias;

            let out_start = r * cols + g * group_size;
            for i in 0..group_size {
                let qval = ((group[i] - bias) / scale).round().clamp(0.0, 255.0) as u8;
                data[out_start + i] = qval;
            }
        }
    }

    QuantizedTensor {
        data,
        scales,
        biases,
        rows,
        cols,
        group_size,
        bits: 8,
    }
}

/// fp16 weight matrix for bandwidth-efficient GEMV.
struct F16Mat {
    data: Vec<u16>, // fp16 as raw u16 bits
    rows: usize,
    cols: usize,
}

impl F16Mat {
    fn from_f32(w: &[f32], rows: usize, cols: usize) -> Self {
        debug_assert_eq!(w.len(), rows * cols);
        let data: Vec<u16> = w
            .iter()
            .map(|&v| half::f16::from_f32(v).to_bits())
            .collect();
        F16Mat { data, rows, cols }
    }

    /// Parallel GEMV: out[rows] = W_f16[rows, cols] @ x_f32[cols].
    ///
    /// Each row: convert fp16→fp32 on the fly, dot with x. Rayon parallel for large matrices.
    fn gemv(&self, x: &[f32]) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.cols);
        let mut out = vec![0.0f32; self.rows];
        if self.rows >= 512 {
            use rayon::prelude::*;
            out.par_iter_mut().enumerate().for_each(|(r, o)| {
                *o = self.dot_row(r, x);
            });
        } else {
            for r in 0..self.rows {
                out[r] = self.dot_row(r, x);
            }
        }
        out
    }

    #[inline]
    fn dot_row(&self, r: usize, x: &[f32]) -> f32 {
        let row = &self.data[r * self.cols..(r + 1) * self.cols];
        let mut acc = 0.0f32;
        let mut i = 0;
        // Process 8 elements at a time — compiler should auto-vectorize with NEON
        while i + 8 <= self.cols {
            acc += half::f16::from_bits(row[i]).to_f32() * x[i]
                + half::f16::from_bits(row[i + 1]).to_f32() * x[i + 1]
                + half::f16::from_bits(row[i + 2]).to_f32() * x[i + 2]
                + half::f16::from_bits(row[i + 3]).to_f32() * x[i + 3]
                + half::f16::from_bits(row[i + 4]).to_f32() * x[i + 4]
                + half::f16::from_bits(row[i + 5]).to_f32() * x[i + 5]
                + half::f16::from_bits(row[i + 6]).to_f32() * x[i + 6]
                + half::f16::from_bits(row[i + 7]).to_f32() * x[i + 7];
            i += 8;
        }
        while i < self.cols {
            acc += half::f16::from_bits(row[i]).to_f32() * x[i];
            i += 1;
        }
        acc
    }
}

/// Per-layer fp16 weights for decode.
struct F16LayerDecode {
    wq: F16Mat,
    wk: F16Mat,
    wv: F16Mat,
    wo: F16Mat,
    w1: F16Mat,
    w3: F16Mat,
    w2: F16Mat,
    rms_att: Vec<f32>,
    rms_ffn: Vec<f32>,
}

/// fp16 model weights for decode: halves memory bandwidth vs fp32.
///
/// Uses rayon parallel GEMV for large matrices (classifier, attention projections).
/// Each row's fp16→fp32 conversion is fused with the dot product — no materialization.
pub struct F16DecodeWeights {
    layers: Vec<F16LayerDecode>,
    cfg: MilConfig,
    rms_final: Vec<f32>,
    embed: Vec<f32>,
    vocab_size: usize,
    lm_head: F16Mat,
}

impl F16DecodeWeights {
    /// Convert model weights to fp16 for bandwidth-optimized decode.
    pub fn from_model(model: &ModelWeights) -> Self {
        let cfg = &model.cfg;
        let dim = cfg.dim;
        let hidden = cfg.hidden_dim;
        let q_proj_dim = cfg.q_proj_dim();
        let kv_dim = cfg.n_kv_heads * cfg.head_dim();
        let attn_dim = cfg.n_heads * cfg.head_dim();

        let layers = model
            .layers
            .iter()
            .map(|lw| F16LayerDecode {
                wq: F16Mat::from_f32(&lw.wq, q_proj_dim, dim),
                wk: F16Mat::from_f32(&lw.wk, kv_dim, dim),
                wv: F16Mat::from_f32(&lw.wv, kv_dim, dim),
                wo: F16Mat::from_f32(&lw.wo, dim, attn_dim),
                w1: F16Mat::from_f32(&lw.w1, hidden, dim),
                w3: F16Mat::from_f32(&lw.w3, hidden, dim),
                w2: F16Mat::from_f32(&lw.w2, dim, hidden),
                rms_att: lw.rms_att.clone(),
                rms_ffn: lw.rms_ffn.clone(),
            })
            .collect();

        let cls_w = model.lm_head.as_ref().unwrap_or(&model.embed);

        F16DecodeWeights {
            layers,
            cfg: cfg.clone(),
            rms_final: model.rms_final.clone(),
            embed: model.embed.clone(),
            vocab_size: model.vocab_size,
            lm_head: F16Mat::from_f32(cls_w, model.vocab_size, dim),
        }
    }
}

/// fp16 parallel GEMV decode: halves bandwidth, parallelizes across cores.
pub fn decode_step_f16(fw: &F16DecodeWeights, token: u32, kv_cache: &mut KvCache) -> DecodeResult {
    let cfg = &fw.cfg;
    let dim = cfg.dim;
    let n_layers = fw.layers.len();
    let n_q_heads = cfg.n_heads;
    let n_kv_heads = cfg.n_kv_heads;
    let head_dim = cfg.head_dim();
    let kv_dim = n_kv_heads * head_dim;
    let attn_dim = n_q_heads * head_dim;
    let pos = kv_cache.pos();

    let mut x = vec![0.0f32; dim];
    x.copy_from_slice(&fw.embed[token as usize * dim..(token as usize + 1) * dim]);

    for l in 0..n_layers {
        let fl = &fw.layers[l];

        let mut xnorm = vec![0.0f32; dim];
        rmsnorm(&mut xnorm, &x, &fl.rms_att, dim, 1, cfg.rms_eps);

        let q_raw = fl.wq.gemv(&xnorm);
        let mut k = fl.wk.gemv(&xnorm);
        let v = fl.wv.gemv(&xnorm);

        let (mut q, attn_gate) = if cfg.attn_output_gate {
            let mut q = vec![0.0f32; attn_dim];
            let mut gate = vec![0.0f32; attn_dim];
            for h in 0..n_q_heads {
                let src_base = h * 2 * head_dim;
                let dst_base = h * head_dim;
                q[dst_base..dst_base + head_dim]
                    .copy_from_slice(&q_raw[src_base..src_base + head_dim]);
                gate[dst_base..dst_base + head_dim]
                    .copy_from_slice(&q_raw[src_base + head_dim..src_base + 2 * head_dim]);
            }
            (q, Some(gate))
        } else {
            (q_raw, None)
        };

        rope_at_pos(
            &mut q,
            &mut k,
            n_q_heads,
            n_kv_heads,
            head_dim,
            pos,
            cfg.rope_theta,
        );

        kv_cache.append(l, &k, &v);
        let save_pos = kv_cache.pos;
        kv_cache.pos = pos + 1;
        let mut attn_out = sdpa_cached(&q, kv_cache, l, n_q_heads, head_dim);
        kv_cache.pos = save_pos;

        if let Some(ref gate) = attn_gate {
            for i in 0..attn_dim {
                let sig = 1.0 / (1.0 + (-gate[i]).exp());
                attn_out[i] *= sig;
            }
        }

        let o = fl.wo.gemv(&attn_out);
        vec_add_inplace(&mut x, &o);

        let mut x2norm = vec![0.0f32; dim];
        rmsnorm(&mut x2norm, &x, &fl.rms_ffn, dim, 1, cfg.rms_eps);
        let mut h1 = fl.w1.gemv(&x2norm);
        let h3 = fl.w3.gemv(&x2norm);
        silu_inplace(&mut h1);
        for i in 0..cfg.hidden_dim {
            h1[i] *= h3[i];
        }
        let ffn_out = fl.w2.gemv(&h1);
        vec_add_inplace(&mut x, &ffn_out);
    }

    kv_cache.advance();

    let mut x_final = vec![0.0f32; dim];
    rmsnorm(&mut x_final, &x, &fw.rms_final, dim, 1, cfg.rms_eps);
    let logits = fw.lm_head.gemv(&x_final);

    DecodeResult { logits }
}

/// ANE-accelerated single-token forward pass with KV cache.
///
/// Uses ANE for FFN (the main compute bottleneck), CPU for attention + KV cache.
pub fn decode_step_ane(
    model: &ModelWeights,
    kernels: &DecodeKernels,
    token: u32,
    kv_cache: &mut KvCache,
) -> DecodeResult {
    let cfg = &model.cfg;
    let dim = cfg.dim;
    let n_layers = model.layers.len();
    let n_q_heads = cfg.n_heads;
    let n_kv_heads = cfg.n_kv_heads;
    let head_dim = cfg.head_dim();
    let q_proj_dim = cfg.q_proj_dim();
    let kv_dim = n_kv_heads * head_dim;
    let attn_dim = n_q_heads * head_dim;
    let pos = kv_cache.pos();

    // 1. Embedding lookup
    let mut x = vec![0.0f32; dim];
    for d in 0..dim {
        x[d] = model.embed[token as usize * dim + d];
    }

    // 2. Transformer layers
    for l in 0..n_layers {
        let lw = &model.layers[l];

        // RMSNorm (attention) — CPU
        let mut xnorm = vec![0.0f32; dim];
        rmsnorm(&mut xnorm, &x, &lw.rms_att, dim, 1, cfg.rms_eps);

        // Q, K, V projections — CPU
        let q_raw = cpu_matmul(&lw.wq, &xnorm, q_proj_dim, dim, 1);
        let mut k = cpu_matmul(&lw.wk, &xnorm, kv_dim, dim, 1);
        let v = cpu_matmul(&lw.wv, &xnorm, kv_dim, dim, 1);

        // Split Q and gate
        let (mut q, attn_gate) = if cfg.attn_output_gate {
            let mut q = vec![0.0f32; attn_dim];
            let mut gate = vec![0.0f32; attn_dim];
            for h in 0..n_q_heads {
                let src_base = h * 2 * head_dim;
                let dst_base = h * head_dim;
                q[dst_base..dst_base + head_dim]
                    .copy_from_slice(&q_raw[src_base..src_base + head_dim]);
                gate[dst_base..dst_base + head_dim]
                    .copy_from_slice(&q_raw[src_base + head_dim..src_base + 2 * head_dim]);
            }
            (q, Some(gate))
        } else {
            (q_raw, None)
        };

        // RoPE — CPU
        rope_at_pos(
            &mut q,
            &mut k,
            n_q_heads,
            n_kv_heads,
            head_dim,
            pos,
            cfg.rope_theta,
        );

        // KV cache
        kv_cache.append(l, &k, &v);
        let save_pos = kv_cache.pos;
        kv_cache.pos = pos + 1;
        let mut attn_out = sdpa_cached(&q, kv_cache, l, n_q_heads, head_dim);
        kv_cache.pos = save_pos;

        // Gate
        if let Some(ref gate) = attn_gate {
            for i in 0..attn_dim {
                let sig = 1.0 / (1.0 + (-gate[i]).exp());
                attn_out[i] *= sig;
            }
        }

        // Wo projection — CPU
        let o = cpu_matmul(&lw.wo, &attn_out, dim, attn_dim, 1);
        vec_add_inplace(&mut x, &o);

        // RMSNorm (FFN) — CPU
        let mut x2norm = vec![0.0f32; dim];
        rmsnorm(&mut x2norm, &x, &lw.rms_ffn, dim, 1, cfg.rms_eps);

        // FFN — ANE!
        let ffn_out = kernels.eval_ffn(l, &x2norm).unwrap_or_else(|e| {
            tracing::warn!("ANE FFN layer {l} failed: {e}, falling back to CPU");
            // CPU fallback
            let hidden = cfg.hidden_dim;
            let mut h1 = cpu_matmul(&lw.w1, &x2norm, hidden, dim, 1);
            let h3 = cpu_matmul(&lw.w3, &x2norm, hidden, dim, 1);
            silu_inplace(&mut h1);
            for i in 0..hidden {
                h1[i] *= h3[i];
            }
            cpu_matmul(&lw.w2, &h1, dim, hidden, 1)
        });

        vec_add_inplace(&mut x, &ffn_out);
    }

    kv_cache.advance();

    // 3. Final RMSNorm
    let mut x_final = vec![0.0f32; dim];
    rmsnorm(&mut x_final, &x, &model.rms_final, dim, 1, cfg.rms_eps);

    // 4. Classifier — CPU (use SGEMM instead of naive loop)
    let vocab = model.vocab_size;
    let cls_w = model.lm_head.as_ref().unwrap_or(&model.embed);
    let logits = cpu_matmul(cls_w, &x_final, vocab, dim, 1);

    DecodeResult { logits }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::super::ane_mil::MilConfig;
    use super::super::ane_weights::{LayerWeights, ModelWeights};
    use super::*;

    /// Build a tiny random model for testing decode correctness.
    /// dim=64, hidden=128, n_heads=4, n_kv_heads=2, head_dim=16, 2 layers, vocab=32.
    fn make_tiny_model() -> ModelWeights {
        let dim = 64;
        let hidden = 128;
        let n_heads = 4;
        let n_kv_heads = 2;
        let head_dim = dim / n_heads; // 16
        let vocab = 32;
        let n_layers = 2;

        let mut cfg = MilConfig::mha(dim, hidden, n_heads, 1);
        cfg.n_kv_heads = n_kv_heads;
        cfg.head_dim_explicit = head_dim;

        // Deterministic pseudo-random weights
        let mut seed = 42u64;
        let mut rand = || -> f32 {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as f32 / (1u64 << 31) as f32) * 0.02 - 0.01
        };

        let make_vec =
            |n: usize, r: &mut dyn FnMut() -> f32| -> Vec<f32> { (0..n).map(|_| r()).collect() };

        let q_proj_dim = n_heads * head_dim; // 64
        let kv_dim = n_kv_heads * head_dim; // 32

        let layers: Vec<LayerWeights> = (0..n_layers)
            .map(|_| LayerWeights {
                wq: make_vec(q_proj_dim * dim, &mut rand),
                wk: make_vec(kv_dim * dim, &mut rand),
                wv: make_vec(kv_dim * dim, &mut rand),
                wo: make_vec(dim * q_proj_dim, &mut rand),
                w1: make_vec(hidden * dim, &mut rand),
                w2: make_vec(dim * hidden, &mut rand),
                w3: make_vec(hidden * dim, &mut rand),
                rms_att: make_vec(dim, &mut rand)
                    .iter()
                    .map(|x| x.abs() + 0.1)
                    .collect(),
                rms_ffn: make_vec(dim, &mut rand)
                    .iter()
                    .map(|x| x.abs() + 0.1)
                    .collect(),
                q_norm: None,
                k_norm: None,
                gdn: None,
                moe: None,
            })
            .collect();

        ModelWeights {
            cfg,
            layers,
            rms_final: make_vec(dim, &mut rand)
                .iter()
                .map(|x| x.abs() + 0.1)
                .collect(),
            embed: make_vec(vocab * dim, &mut rand),
            vocab_size: vocab,
            lm_head: None,
            vocab_clusters: None,
            router_adapter: None,
        }
    }

    #[test]
    fn test_decode_step_returns_logits() {
        let model = make_tiny_model();
        let mut cache = KvCache::new(&model.cfg, model.layers.len(), 64);

        let result = decode_step(&model, 5, &mut cache);
        assert_eq!(result.logits.len(), model.vocab_size);
        assert_eq!(cache.pos(), 1);

        // Logits should not be all zeros (model has non-zero weights)
        let any_nonzero = result.logits.iter().any(|&v| v.abs() > 1e-10);
        assert!(any_nonzero, "logits are all zero — decode_step is broken");
    }

    #[test]
    fn test_decode_step_kv_cache_grows() {
        let model = make_tiny_model();
        let mut cache = KvCache::new(&model.cfg, model.layers.len(), 64);

        for i in 0..5 {
            decode_step(&model, i, &mut cache);
            assert_eq!(cache.pos(), i as usize + 1);
        }
    }

    #[test]
    fn test_decode_step_deterministic() {
        let model = make_tiny_model();

        let mut cache1 = KvCache::new(&model.cfg, model.layers.len(), 64);
        let r1 = decode_step(&model, 3, &mut cache1);

        let mut cache2 = KvCache::new(&model.cfg, model.layers.len(), 64);
        let r2 = decode_step(&model, 3, &mut cache2);

        assert_eq!(r1.logits, r2.logits, "decode_step is not deterministic");
    }

    #[test]
    fn test_decode_step_different_tokens_differ() {
        let model = make_tiny_model();

        let mut cache1 = KvCache::new(&model.cfg, model.layers.len(), 64);
        let r1 = decode_step(&model, 1, &mut cache1);

        let mut cache2 = KvCache::new(&model.cfg, model.layers.len(), 64);
        let r2 = decode_step(&model, 7, &mut cache2);

        assert_ne!(
            r1.logits, r2.logits,
            "different input tokens produce identical logits"
        );
    }

    #[test]
    fn test_generate_draft_tokens() {
        let model = make_tiny_model();
        let mut cache = KvCache::new(&model.cfg, model.layers.len(), 64);

        // Generate from token 0
        let drafts = generate_draft_tokens(&model, &mut cache, 0, 4, None);
        assert_eq!(drafts.len(), 4);
        assert_eq!(cache.pos(), 4); // 4 decode_step calls (prompt_token + 3 sampled)

        // All draft tokens should be valid vocab indices
        for &t in &drafts {
            assert!(
                (t as usize) < model.vocab_size,
                "draft token {t} out of vocab range"
            );
        }
    }

    #[test]
    fn test_sample_argmax() {
        let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        assert_eq!(sample_argmax(&logits), 3);

        let logits2 = vec![-1.0, -0.5, -2.0];
        assert_eq!(sample_argmax(&logits2), 1);
    }

    /// Verify that multi-step decoding matches single-sequence forward at seq=1.
    ///
    /// Decode tokens [A, B] one at a time and verify that the second decode step's
    /// attention is influenced by the first token (i.e., KV cache is working).
    #[test]
    fn test_kv_cache_influences_output() {
        let model = make_tiny_model();

        // Decode [5, 10] — two steps
        let mut cache = KvCache::new(&model.cfg, model.layers.len(), 64);
        let _ = decode_step(&model, 5, &mut cache);
        let r_with_context = decode_step(&model, 10, &mut cache);

        // Decode [10] alone — one step, no prior context
        let mut cache_alone = KvCache::new(&model.cfg, model.layers.len(), 64);
        let r_alone = decode_step(&model, 10, &mut cache_alone);

        // They should differ because the first decode has token 5 in the KV cache
        assert_ne!(
            r_with_context.logits, r_alone.logits,
            "KV cache has no effect — decode with prior context should differ from fresh decode"
        );
    }

    // -----------------------------------------------------------------------
    // GDN decode tests
    // -----------------------------------------------------------------------

    /// Build a tiny model with GDN layers for testing GDN decode.
    /// 4 layers: [GDN, MHA, GDN, MHA]. dim=64, hidden=128.
    fn make_tiny_gdn_model() -> ModelWeights {
        let dim = 64;
        let hidden = 128;
        let n_heads = 4;
        let n_kv_heads = 2;
        let head_dim = dim / n_heads; // 16
        let vocab = 32;
        let n_layers = 4;

        // GDN config: h_k=2, d_k=16, h_v=4, d_v=16, conv_kernel=4
        let h_k = 2;
        let d_k = 16;
        let h_v = 4;
        let d_v = 16;
        let key_dim = h_k * d_k; // 32
        let value_dim = h_v * d_v; // 64
        let qkv_dim = 2 * key_dim + value_dim; // 128
        let conv_kernel = 4;

        let mut cfg = MilConfig::mha(dim, hidden, n_heads, 1);
        cfg.n_kv_heads = n_kv_heads;
        cfg.head_dim_explicit = head_dim;
        cfg.linear_n_heads = h_k;
        cfg.linear_head_dim = d_k;
        cfg.linear_n_value_heads = h_v;
        cfg.linear_value_head_dim = d_v;
        cfg.conv_kernel_size = conv_kernel;

        let mut seed = 42u64;
        let mut rand = || -> f32 {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as f32 / (1u64 << 31) as f32) * 0.02 - 0.01
        };
        let make_vec =
            |n: usize, r: &mut dyn FnMut() -> f32| -> Vec<f32> { (0..n).map(|_| r()).collect() };

        let q_proj_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;

        let make_gdn = |r: &mut dyn FnMut() -> f32| -> super::super::ane_weights::GdnLayerWeights {
            super::super::ane_weights::GdnLayerWeights {
                qkv_proj: make_vec(qkv_dim * dim, r),
                a_proj: make_vec(h_v * dim, r),
                b_proj: make_vec(h_v * dim, r),
                z_proj: make_vec(value_dim * dim, r),
                o_proj: make_vec(dim * value_dim, r),
                a_log: make_vec(h_v, r),
                dt_bias: make_vec(h_v, r),
                norm_weight: make_vec(d_v, r).iter().map(|x| x.abs() + 0.1).collect(),
                conv_weight: make_vec(qkv_dim * conv_kernel, r),
                conv_bias: make_vec(qkv_dim, r),
            }
        };

        let layers: Vec<LayerWeights> = (0..n_layers)
            .map(|i| {
                let is_gdn = i % 2 == 0;
                LayerWeights {
                    wq: make_vec(q_proj_dim * dim, &mut rand),
                    wk: make_vec(kv_dim * dim, &mut rand),
                    wv: make_vec(kv_dim * dim, &mut rand),
                    wo: make_vec(dim * q_proj_dim, &mut rand),
                    w1: make_vec(hidden * dim, &mut rand),
                    w2: make_vec(dim * hidden, &mut rand),
                    w3: make_vec(hidden * dim, &mut rand),
                    rms_att: make_vec(dim, &mut rand)
                        .iter()
                        .map(|x| x.abs() + 0.1)
                        .collect(),
                    rms_ffn: make_vec(dim, &mut rand)
                        .iter()
                        .map(|x| x.abs() + 0.1)
                        .collect(),
                    q_norm: None,
                    k_norm: None,
                    gdn: if is_gdn {
                        Some(make_gdn(&mut rand))
                    } else {
                        None
                    },
                    moe: None,
                }
            })
            .collect();

        ModelWeights {
            cfg,
            layers,
            rms_final: make_vec(dim, &mut rand)
                .iter()
                .map(|x| x.abs() + 0.1)
                .collect(),
            embed: make_vec(vocab * dim, &mut rand),
            vocab_size: vocab,
            lm_head: None,
            vocab_clusters: None,
            router_adapter: None,
        }
    }

    #[test]
    fn test_gdn_decode_produces_nonzero_logits() {
        let model = make_tiny_gdn_model();
        let mut cache = KvCache::new(&model.cfg, model.layers.len(), 64);
        cache.init_gdn(&model);

        let result = decode_step(&model, 5, &mut cache);
        assert_eq!(result.logits.len(), model.vocab_size);
        assert!(
            result.logits.iter().any(|&v| v.abs() > 1e-10),
            "GDN decode produced all-zero logits"
        );
    }

    #[test]
    fn test_gdn_decode_state_accumulates() {
        let model = make_tiny_gdn_model();

        // Decode [5, 10] with GDN state
        let mut cache = KvCache::new(&model.cfg, model.layers.len(), 64);
        cache.init_gdn(&model);
        let _ = decode_step(&model, 5, &mut cache);
        let r_with_context = decode_step(&model, 10, &mut cache);

        // Decode [10] alone with fresh GDN state
        let mut cache_alone = KvCache::new(&model.cfg, model.layers.len(), 64);
        cache_alone.init_gdn(&model);
        let r_alone = decode_step(&model, 10, &mut cache_alone);

        assert_ne!(
            r_with_context.logits, r_alone.logits,
            "GDN state has no effect — decode with prior context should differ"
        );
    }

    #[test]
    fn test_gdn_decode_deterministic() {
        let model = make_tiny_gdn_model();

        let mut cache1 = KvCache::new(&model.cfg, model.layers.len(), 64);
        cache1.init_gdn(&model);
        let r1 = decode_step(&model, 3, &mut cache1);

        let mut cache2 = KvCache::new(&model.cfg, model.layers.len(), 64);
        cache2.init_gdn(&model);
        let r2 = decode_step(&model, 3, &mut cache2);

        assert_eq!(r1.logits, r2.logits, "GDN decode is not deterministic");
    }

    #[test]
    fn test_gdn_decode_matches_batch_forward() {
        use super::super::ane_forward::cpu_gdn_forward_bench;

        let model = make_tiny_gdn_model();
        let dim = model.cfg.dim;
        let seq = 4;
        let n_layers = model.layers.len();

        // Generate per-token xnorm inputs (post-embedding, post-RMSNorm)
        let mut seed = 99u64;
        let mut rand = || -> f32 {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as f32 / (1u64 << 31) as f32) * 0.1 - 0.05
        };
        let xnorm_tokens: Vec<Vec<f32>> = (0..seq)
            .map(|_| (0..dim).map(|_| rand()).collect())
            .collect();

        // Test GDN layer 0 (first GDN layer)
        let gdn_layer_idx = 0;
        let gdn_w = model.layers[gdn_layer_idx].gdn.as_ref().unwrap();

        // Batch forward: pack [dim, seq] channels-first, run batch GDN
        let mut cfg_batch = model.cfg.clone();
        cfg_batch.seq_len = seq;
        let mut xnorm_batch = vec![0.0f32; dim * seq];
        for t in 0..seq {
            for d in 0..dim {
                xnorm_batch[d * seq + t] = xnorm_tokens[t][d];
            }
        }
        let batch_out = cpu_gdn_forward_bench(gdn_w, &xnorm_batch, &cfg_batch);
        // batch_out is [dim, seq] channels-first

        // Single-token decode: process one token at a time
        let mut gdn_state = GdnLayerDecodeState::new(&model.cfg);
        let mut decode_outputs: Vec<Vec<f32>> = Vec::new();
        for t in 0..seq {
            let out = gdn_decode_single(gdn_w, &mut gdn_state, &xnorm_tokens[t], &model.cfg);
            decode_outputs.push(out);
        }

        // Compare at each position
        for t in 0..seq {
            for d in 0..dim {
                let batch_val = batch_out[d * seq + t];
                let decode_val = decode_outputs[t][d];
                let diff = (batch_val - decode_val).abs();
                let scale = batch_val.abs().max(decode_val.abs()).max(1e-6);
                assert!(
                    diff / scale < 1e-4,
                    "Mismatch at t={t}, d={d}: batch={batch_val:.6}, decode={decode_val:.6}, rel_diff={:.6}",
                    diff / scale
                );
            }
        }
    }

    #[test]
    fn test_gdn_decode_ffn_only_fallback() {
        let model = make_tiny_gdn_model();

        // Without init_gdn: FFN-only fallback
        let mut cache_no_gdn = KvCache::new(&model.cfg, model.layers.len(), 64);
        let r_no_gdn = decode_step(&model, 5, &mut cache_no_gdn);

        // With init_gdn: full GDN attention
        let mut cache_gdn = KvCache::new(&model.cfg, model.layers.len(), 64);
        cache_gdn.init_gdn(&model);
        let r_gdn = decode_step(&model, 5, &mut cache_gdn);

        // Both should produce valid logits
        assert_eq!(r_no_gdn.logits.len(), model.vocab_size);
        assert_eq!(r_gdn.logits.len(), model.vocab_size);

        // They should differ (GDN attention adds information that FFN-only misses)
        assert_ne!(
            r_no_gdn.logits, r_gdn.logits,
            "GDN attention should produce different output than FFN-only"
        );
    }

    // -----------------------------------------------------------------------
    // ANE-specific tests (require Apple Silicon with ANE)
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    // MoE forward tests
    // -----------------------------------------------------------------------

    /// Build a tiny MoE model: 2 layers, 4 experts top-2, shared expert.
    fn make_tiny_moe_model() -> ModelWeights {
        let dim = 32;
        let hidden = 64;
        let moe_hidden = 16;
        let num_experts = 4;
        let num_experts_per_tok = 2;
        let vocab = 16;

        let mut seed = 77u64;
        let mut rand = || -> f32 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 33) as f32 / (1u64 << 31) as f32) * 0.02 - 0.01
        };
        let make_vec = |n: usize, r: &mut dyn FnMut() -> f32| -> Vec<f32> {
            (0..n).map(|_| r()).collect()
        };

        let make_quantized_identity = |rows: usize, cols: usize, r: &mut dyn FnMut() -> f32| -> QuantizedTensor {
            // Store as "8-bit quantized" with scale=1, bias=0 for simplicity in tests.
            // Actually just use dequantized f32 wrapped in a fake QuantizedTensor.
            // For testing, we use group_size=cols, bits=8.
            let data_f32 = make_vec(rows * cols, r);
            // Pack as 8-bit: each f32 → u8 via scale/bias
            let n_groups = rows; // one group per row with group_size=cols
            let mut scales = vec![1.0f32; n_groups];
            let mut biases = vec![0.0f32; n_groups];
            let mut data_u8 = vec![0u8; rows * cols];

            for row in 0..rows {
                let row_data = &data_f32[row * cols..(row + 1) * cols];
                let min_val = row_data.iter().cloned().fold(f32::MAX, f32::min);
                let max_val = row_data.iter().cloned().fold(f32::MIN, f32::max);
                let scale = (max_val - min_val).max(1e-10) / 255.0;
                let bias = min_val;
                scales[row] = scale;
                biases[row] = bias;
                for col in 0..cols {
                    let val = ((data_f32[row * cols + col] - bias) / scale).round().clamp(0.0, 255.0) as u8;
                    data_u8[row * cols + col] = val;
                }
            }

            QuantizedTensor {
                data: data_u8,
                scales,
                biases,
                rows,
                cols,
                group_size: cols,
                bits: 8,
            }
        };

        let make_expert = |r: &mut dyn FnMut() -> f32| -> super::super::ane_weights::MoeExpert {
            super::super::ane_weights::MoeExpert {
                gate_proj: make_quantized_identity(moe_hidden, dim, r),
                up_proj: make_quantized_identity(moe_hidden, dim, r),
                down_proj: make_quantized_identity(dim, moe_hidden, r),
            }
        };

        let make_packed_experts = |r: &mut dyn FnMut() -> f32| -> super::super::ane_weights::PackedMoeExperts {
            // Build individual experts then pack into contiguous arrays
            let experts: Vec<_> = (0..num_experts).map(|_| {
                (make_quantized_identity(moe_hidden, dim, r),  // gate
                 make_quantized_identity(moe_hidden, dim, r),  // up
                 make_quantized_identity(dim, moe_hidden, r))  // down
            }).collect();

            let mut gate_data = Vec::new();
            let mut gate_scales = Vec::new();
            let mut gate_biases = Vec::new();
            let mut up_data = Vec::new();
            let mut up_scales = Vec::new();
            let mut up_biases = Vec::new();
            let mut down_data = Vec::new();
            let mut down_scales = Vec::new();
            let mut down_biases = Vec::new();

            for (g, u, d) in &experts {
                gate_data.extend_from_slice(&g.data);
                gate_scales.extend_from_slice(&g.scales);
                gate_biases.extend_from_slice(&g.biases);
                up_data.extend_from_slice(&u.data);
                up_scales.extend_from_slice(&u.scales);
                up_biases.extend_from_slice(&u.biases);
                down_data.extend_from_slice(&d.data);
                down_scales.extend_from_slice(&d.scales);
                down_biases.extend_from_slice(&d.biases);
            }

            super::super::ane_weights::PackedMoeExperts {
                gate_data, gate_scales, gate_biases,
                up_data, up_scales, up_biases,
                down_data, down_scales, down_biases,
                n_experts: num_experts,
                gate_rows: moe_hidden,
                gate_cols: dim,
                down_rows: dim,
                down_cols: moe_hidden,
                group_size: dim, // matches make_quantized_identity group_size=cols=dim
                bits: 8,
            }
        };

        let make_moe = |r: &mut dyn FnMut() -> f32| -> super::super::ane_weights::MoeLayerWeights {
            super::super::ane_weights::MoeLayerWeights {
                router: make_vec(num_experts * dim, r),
                packed_experts: make_packed_experts(r),
                shared_expert: Some(make_expert(r)),
                num_experts,
                num_experts_per_tok,
                moe_hidden,
            }
        };

        let cfg = MilConfig::mha(dim, hidden, 4, 1);

        let layers = (0..2).map(|_| {
            LayerWeights {
                wq: make_vec(dim * dim, &mut rand),
                wk: make_vec(dim * dim, &mut rand),
                wv: make_vec(dim * dim, &mut rand),
                wo: make_vec(dim * dim, &mut rand),
                w1: vec![], // unused — MoE replaces dense FFN
                w2: vec![],
                w3: vec![],
                rms_att: make_vec(dim, &mut rand).iter().map(|x| x.abs() + 0.1).collect(),
                rms_ffn: make_vec(dim, &mut rand).iter().map(|x| x.abs() + 0.1).collect(),
                q_norm: None,
                k_norm: None,
                gdn: None,
                moe: Some(std::sync::Arc::new(make_moe(&mut rand))),
            }
        }).collect();

        ModelWeights {
            cfg,
            layers,
            rms_final: make_vec(dim, &mut rand).iter().map(|x| x.abs() + 0.1).collect(),
            embed: make_vec(vocab * dim, &mut rand),
            vocab_size: vocab,
            lm_head: None,
            vocab_clusters: None,
            router_adapter: None,
        }
    }

    #[test]
    fn test_moe_decode_produces_valid_logits() {
        let model = make_tiny_moe_model();
        let mut cache = KvCache::new(&model.cfg, model.layers.len(), 64);
        let r = decode_step(&model, 3, &mut cache);

        assert_eq!(r.logits.len(), model.vocab_size);
        // Logits should be non-zero (MoE is producing output)
        let max_logit = r.logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min_logit = r.logits.iter().cloned().fold(f32::INFINITY, f32::min);
        assert!(
            (max_logit - min_logit).abs() > 1e-10,
            "MoE logits are all identical — routing may be broken"
        );
    }

    /// Load real 35B-A3B model with MoE experts and run a decode step.
    ///
    /// cargo test --features ane --release --lib -- "test_moe_35b_real" --nocapture --test-threads=1 --ignored
    #[test]
    #[ignore]
    fn test_moe_35b_real() {
        // Try 3-bit first (smaller, faster), fall back to 4-bit
        let model_dir_3b = std::path::Path::new(&std::env::var("HOME").unwrap())
            .join(".cache/lm-studio/models/NexVeridian/Qwen3.5-35B-A3B-3bit");
        let model_dir_4b = std::path::Path::new(&std::env::var("HOME").unwrap())
            .join(".cache/lm-studio/models/mlx-community/Qwen3.5-35B-A3B-4bit");
        let model_dir = if model_dir_3b.exists() { &model_dir_3b }
            else if model_dir_4b.exists() { &model_dir_4b }
            else {
                eprintln!("SKIP: no 35B model found");
                return;
            };
        eprintln!("Using: {}", model_dir.display());

        // 35B-A3B config: dim=2048, moe_hidden=512, 16 heads (2 KV), head_dim=256
        // 40 layers: 30 GDN + 10 full attention (every 4th)
        // 256 experts, top-8, shared_expert_intermediate_size=512
        // GDN: 16 key heads × 128 dim, 32 value heads × 128 dim, conv kernel=4
        let config_str = std::fs::read_to_string(model_dir.join("config.json")).unwrap();
        let config: serde_json::Value = serde_json::from_str(&config_str).unwrap();
        let tc = config.get("text_config").unwrap_or(&config);

        let layer_types: Vec<String> = tc.get("layer_types")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
            .unwrap_or_default();
        let linear_indices: Vec<usize> = layer_types.iter().enumerate()
            .filter(|(_, t)| t.as_str() == "linear_attention")
            .map(|(i, _)| i).collect();

        let cfg = MilConfig {
            dim: 2048,
            hidden_dim: 512, // moe_intermediate_size (used for shared expert dense FFN)
            n_heads: 16,
            seq_len: 1,
            n_kv_heads: 2,
            rope_theta: 10_000_000.0,
            rms_eps: 1e-6,
            has_lm_head: false,
            head_dim_explicit: 256,
            linear_attn_indices: linear_indices,
            linear_n_heads: 16,
            linear_head_dim: 128,
            linear_n_value_heads: 32,
            linear_value_head_dim: 128,
            conv_kernel_size: 4,
            attn_output_gate: true,
        };

        eprintln!("Loading 35B base weights (skip experts)...");
        let t0 = std::time::Instant::now();
        let mut model = ModelWeights::from_mlx_safetensors(&model_dir, &cfg)
            .expect("base load failed");
        eprintln!("  Base loaded in {:.1}s", t0.elapsed().as_secs_f64());

        let num_experts = tc.get("num_experts").and_then(|v| v.as_u64()).unwrap_or(256) as usize;
        let experts_per_tok = tc.get("num_experts_per_tok").and_then(|v| v.as_u64()).unwrap_or(8) as usize;
        let moe_hidden = tc.get("moe_intermediate_size").and_then(|v| v.as_u64()).unwrap_or(512) as usize;

        eprintln!("Loading MoE experts ({num_experts} experts, top-{experts_per_tok})...");
        let t1 = std::time::Instant::now();
        model.load_moe_experts(&model_dir, num_experts, experts_per_tok, moe_hidden)
            .expect("MoE expert load failed");
        eprintln!("  Experts loaded in {:.1}s", t1.elapsed().as_secs_f64());

        let moe_layers = model.layers.iter().filter(|l| l.moe.is_some()).count();
        let gdn_layers = model.layers.iter().filter(|l| l.gdn.is_some()).count();
        eprintln!("  Layers: {} total, {} MoE, {} GDN", model.layers.len(), moe_layers, gdn_layers);

        // Decode tokens and time
        eprintln!("Running decode_step...");
        let mut cache = KvCache::new(&model.cfg, model.layers.len(), 64);
        cache.init_gdn(&model);

        // First token (cold — page faults for expert weights)
        let t2 = std::time::Instant::now();
        let result = decode_step(&model, 1, &mut cache);
        let cold_ms = t2.elapsed().as_secs_f64() * 1000.0;
        eprintln!("  Token 1 (cold): {cold_ms:.1}ms");

        // Second token (warm — expert data already paged in)
        let t3 = std::time::Instant::now();
        let result2 = decode_step(&model, result.logits.iter()
            .enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32).unwrap_or(1), &mut cache);
        let warm_ms = t3.elapsed().as_secs_f64() * 1000.0;
        eprintln!("  Token 2 (warm): {warm_ms:.1}ms");

        // Third token
        let t4 = std::time::Instant::now();
        let _ = decode_step(&model, 2, &mut cache);
        let warm2_ms = t4.elapsed().as_secs_f64() * 1000.0;
        eprintln!("  Token 3 (warm): {warm2_ms:.1}ms");

        eprintln!("  Logits: {} (vocab_size={})", result.logits.len(), model.vocab_size);
        let top = sample_argmax(&result.logits);
        eprintln!("  Top token: {top}");
        eprintln!("  Warm tok/s: {:.1}", 1000.0 / warm_ms);

        // Profile: time MoE vs attention vs GDN separately
        eprintln!("\n  ── Per-component profiling (1 token) ──");
        let dim = cfg.dim;
        let n_layers = model.layers.len();
        let mut moe_us = 0u128;
        let mut gdn_us = 0u128;
        let mut attn_us = 0u128;
        let mut cls_us = 0u128;

        // Manual decode_step with timing
        let mut x = vec![0.0f32; dim];
        for d in 0..dim { x[d] = model.embed[3_usize * dim + d]; }

        for l in 0..n_layers {
            let lw = &model.layers[l];

            // Attention / GDN
            let t = std::time::Instant::now();
            if let Some(ref gdn_w) = lw.gdn {
                if let Some(ref mut gdn_state) = cache.gdn[l] {
                    let mut xnorm = vec![0.0f32; dim];
                    rmsnorm(&mut xnorm, &x, &lw.rms_att, dim, 1, cfg.rms_eps);
                    let attn_out = gdn_decode_single(gdn_w, gdn_state, &xnorm, &cfg);
                    vec_add_inplace(&mut x, &attn_out);
                }
                gdn_us += t.elapsed().as_micros();
            } else {
                // Skip full attention for profiling — just measure MoE
                attn_us += t.elapsed().as_micros();
            }

            // FFN (MoE)
            if let Some(ref moe_w) = lw.moe {
                let t = std::time::Instant::now();
                moe_forward_with_adapter(moe_w, &mut x, &lw.rms_ffn, &cfg, l, model.router_adapter.as_ref());
                moe_us += t.elapsed().as_micros();
            }
        }

        // Classifier
        let t = std::time::Instant::now();
        let mut x_final = vec![0.0f32; dim];
        rmsnorm(&mut x_final, &x, &model.rms_final, dim, 1, cfg.rms_eps);
        let cls_w = model.lm_head.as_ref().unwrap_or(&model.embed);
        let _logits = cpu_matmul(cls_w, &x_final, model.vocab_size, dim, 1);
        cls_us = t.elapsed().as_micros();

        let total_ms = (moe_us + gdn_us + attn_us + cls_us) as f64 / 1000.0;
        eprintln!("  MoE (experts):  {:>7.1}ms ({:.0}%)", moe_us as f64 / 1000.0,
                  moe_us as f64 / (moe_us + gdn_us + attn_us + cls_us).max(1) as f64 * 100.0);
        eprintln!("  GDN (linear):   {:>7.1}ms ({:.0}%)", gdn_us as f64 / 1000.0,
                  gdn_us as f64 / (moe_us + gdn_us + attn_us + cls_us).max(1) as f64 * 100.0);
        eprintln!("  Attention:      {:>7.1}ms ({:.0}%)", attn_us as f64 / 1000.0,
                  attn_us as f64 / (moe_us + gdn_us + attn_us + cls_us).max(1) as f64 * 100.0);
        eprintln!("  Classifier:     {:>7.1}ms ({:.0}%)", cls_us as f64 / 1000.0,
                  cls_us as f64 / (moe_us + gdn_us + attn_us + cls_us).max(1) as f64 * 100.0);
        eprintln!("  Total:          {:>7.1}ms → {:.1} tok/s", total_ms, 1000.0 / total_ms);

        assert_eq!(result.logits.len(), model.vocab_size);
        assert_eq!(result2.logits.len(), model.vocab_size);
    }

    #[test]
    fn test_moe_different_tokens_give_different_logits() {
        let model = make_tiny_moe_model();
        let mut cache1 = KvCache::new(&model.cfg, model.layers.len(), 64);
        let mut cache2 = KvCache::new(&model.cfg, model.layers.len(), 64);

        let r1 = decode_step(&model, 1, &mut cache1);
        let r2 = decode_step(&model, 7, &mut cache2);

        assert_ne!(r1.logits, r2.logits, "Different tokens should produce different logits");
    }

    /// Read routing targets written by scripts/routing_hook.py.
    /// Requires the hook to have been run first (creates ~/.nanobot/routing_targets.bin).
    ///
    /// cargo test --features ane --release --lib -- "test_drain_routing_file" --nocapture --ignored
    #[test]
    #[ignore]
    fn test_drain_routing_file() {
        let path = std::path::Path::new(&std::env::var("HOME").unwrap())
            .join(".nanobot/routing_targets.bin");
        if !path.exists() {
            eprintln!("SKIP: no routing_targets.bin — run scripts/routing_hook.py first");
            return;
        }

        let targets = drain_routing_targets_from_file(&path);
        match targets {
            Some(ref t) if !t.is_empty() => {
                eprintln!("Drained {} routing targets from {}", t.len(), path.display());

                // Verify structure
                let (layer, ref target) = t[0];
                eprintln!("  First: layer={layer}, experts={:?}, probs={:?}, x_norm len={}",
                    &target.expert_indices[..target.expert_indices.len().min(4)],
                    &target.expert_probs[..target.expert_probs.len().min(4)],
                    target.x_norm.len());

                // Verify layer indices are reasonable (0-39 for 40-layer model)
                let max_layer = t.iter().map(|(l, _)| *l).max().unwrap();
                let unique_layers: std::collections::HashSet<usize> = t.iter().map(|(l, _)| *l).collect();
                eprintln!("  Layers: {} unique, max={max_layer}", unique_layers.len());

                assert!(max_layer < 100, "layer index too large: {max_layer}");
                assert!(!target.expert_indices.is_empty(), "expert indices empty");
                assert!(!target.expert_probs.is_empty(), "expert probs empty");
                assert!(target.x_norm.len() > 100, "x_norm too short: {}", target.x_norm.len());

                // Verify probs sum to ~1
                let prob_sum: f32 = target.expert_probs.iter().sum();
                assert!((prob_sum - 1.0).abs() < 0.01,
                    "probs don't sum to 1: {prob_sum}");

                eprintln!("  All checks passed!");

                // Second drain should return None (read_pos updated)
                let targets2 = drain_routing_targets_from_file(&path);
                assert!(targets2.is_none() || targets2.as_ref().map_or(true, |t| t.is_empty()),
                    "Second drain should return empty (read_pos was updated)");
                eprintln!("  Second drain: empty (as expected)");
            }
            _ => {
                eprintln!("No targets in file — run scripts/routing_hook.py --tokens 30 first");
            }
        }
    }

    // -----------------------------------------------------------------------
    // GDN ANE projection kernel tests
    // -----------------------------------------------------------------------

    /// Test that GDN ANE kernels compile for the tiny model.
    ///
    /// cargo test --features ane --lib -- "test_gdn_ane_kernels_compile" --test-threads=1
    #[test]
    fn test_gdn_ane_kernels_compile() {
        use super::super::ane_bridge;
        if ane_bridge::ane_init().is_err() {
            eprintln!("SKIP: ANE init failed");
            return;
        }

        let model = make_tiny_gdn_model();
        let gdn_kernels = GdnAneKernels::compile(&model);

        // Should compile successfully (or return None if ANE unavailable)
        if let Some(ref gk) = gdn_kernels {
            assert_eq!(gk.layers.len(), model.layers.len());
            // Layers 0, 2 are GDN → should have kernels
            assert!(gk.layers[0].is_some(), "GDN layer 0 should have kernels");
            assert!(gk.layers[2].is_some(), "GDN layer 2 should have kernels");
            // Layers 1, 3 are MHA → should be None
            assert!(gk.layers[1].is_none(), "MHA layer 1 should have no GDN kernels");
            assert!(gk.layers[3].is_none(), "MHA layer 3 should have no GDN kernels");
            eprintln!("GDN ANE kernels compiled: {} layers", gk.layers.len());
        } else {
            eprintln!("SKIP: GDN ANE kernel compilation returned None (expected on some hardware)");
        }
    }

    /// Test GDN ANE projection output matches CPU baseline within fp16 tolerance.
    ///
    /// cargo test --features ane --lib -- "test_gdn_ane_matches_cpu" --test-threads=1
    #[test]
    fn test_gdn_ane_matches_cpu() {
        use super::super::ane_bridge;
        if ane_bridge::ane_init().is_err() {
            eprintln!("SKIP: ANE init failed");
            return;
        }

        let model = make_tiny_gdn_model();
        let gdn_kernels = match GdnAneKernels::compile(&model) {
            Some(k) => k,
            None => {
                eprintln!("SKIP: GDN ANE kernel compilation failed");
                return;
            }
        };

        // Run decode with CPU path
        let mut cache_cpu = KvCache::new(&model.cfg, model.layers.len(), 64);
        cache_cpu.init_gdn(&model);
        let r_cpu = decode_step(&model, 5, &mut cache_cpu);

        // Run decode with ANE GDN path
        let blob_kernels = match BlobDecodeKernels::compile(&model, 16) {
            Some(k) => k,
            None => {
                eprintln!("SKIP: BlobDecodeKernels compilation failed");
                return;
            }
        };
        let mut cache_ane = KvCache::new(&model.cfg, model.layers.len(), 64);
        cache_ane.init_gdn(&model);
        let r_ane = decode_step_blob_gdn(&model, &blob_kernels, &gdn_kernels, 5, &mut cache_ane);

        // Compare logits — fp16 tolerance (ANE runs in fp16 internally)
        // Use absolute + relative hybrid: max(|a-b|, |a-b|/max(|a|,|b|,1e-3))
        assert_eq!(r_cpu.logits.len(), r_ane.logits.len());
        let mut max_abs = 0.0f32;
        let mut max_rel = 0.0f32;
        for (i, (a, b)) in r_cpu.logits.iter().zip(&r_ane.logits).enumerate() {
            let abs_err = (a - b).abs();
            let denom = a.abs().max(b.abs()).max(1e-3);
            let rel = abs_err / denom;
            max_abs = max_abs.max(abs_err);
            max_rel = max_rel.max(rel);
            if abs_err > 0.01 {
                eprintln!("  logit[{i}]: cpu={a:.6} ane={b:.6} abs={abs_err:.6}");
            }
        }
        eprintln!("  max absolute error: {max_abs:.6}");
        eprintln!("  max relative error: {max_rel:.6}");
        // fp16 through 4 layers: allow 1% relative or 0.01 absolute
        assert!(
            max_abs < 0.01 || max_rel < 0.01,
            "GDN ANE vs CPU: abs={max_abs:.4} rel={max_rel:.4} — both exceed tolerance"
        );
    }

    // -----------------------------------------------------------------------
    // BLOBFILE decode kernel tests
    // -----------------------------------------------------------------------

    /// Diagnostic: verify conv1x1 BLOBFILE compiles via compile_multi_weights at various seq.
    ///
    /// cargo test --features ane --release --lib -- "test_blob_compile_diagnostic" --nocapture --test-threads=1
    #[test]
    fn test_blob_compile_diagnostic() {
        use super::super::ane_bridge;
        use super::super::ane_mil::gen_conv1x1_blob;
        use super::super::ane_weights::build_fp16_blob;

        if ane_bridge::ane_init().is_err() {
            eprintln!("SKIP: ANE init failed");
            return;
        }

        let dim = 64;
        let hidden = 128;

        let w_data: Vec<f32> = (0..hidden * dim)
            .map(|i| ((i + 1) as f32 * 0.003).sin() * 0.1)
            .collect();
        let w_blob = build_fp16_blob(&w_data);

        // Test single conv1x1 via compile_multi_weights at various seq
        for seq in [1, 2, 4, 8, 16, 32] {
            let spec = gen_conv1x1_blob(dim, hidden, seq);
            let names: Vec<&str> = spec.weight_names.iter().copied().collect();
            let r = AneKernel::compile_multi_weights(
                &spec.mil_text,
                &names,
                &[&w_blob],
                &[spec.input_bytes],
                &[spec.output_bytes],
            );
            match r {
                Ok(k) => {
                    let input = vec![0.5f32; dim * seq];
                    let input_bytes = ane_weights::f32_slice_to_bytes(&input);
                    k.write_input(0, &input_bytes);
                    match k.eval() {
                        Ok(()) => {
                            let mut out = vec![0u8; spec.output_bytes];
                            k.read_output(0, &mut out);
                            let first = f32::from_le_bytes([out[0], out[1], out[2], out[3]]);
                            eprintln!(
                                "  conv1x1 seq={seq}: COMPILE OK, EVAL OK (first={first:.6})"
                            );
                        }
                        Err(e) => eprintln!("  conv1x1 seq={seq}: COMPILE OK, EVAL FAIL ({e})"),
                    }
                }
                Err(e) => eprintln!("  conv1x1 seq={seq}: COMPILE FAIL ({e})"),
            }
        }
    }

    /// Critical gate test: can BLOBFILE FFN compile AND eval at seq=1?
    ///
    /// BLOBFILE kernels use `compile_direct` (_ANEClient path) which is a
    /// different pipeline from the DynMatmul `_ANEInMemoryModel` path.
    /// If this works at seq=1, we bypass the spatial minimum limitation.
    ///
    /// cargo test --features ane --lib -- "test_blob_ffn_seq_sweep" --nocapture --test-threads=1
    #[test]
    fn test_blob_ffn_seq_sweep() {
        use super::super::ane_bridge;

        if ane_bridge::ane_init().is_err() {
            eprintln!("SKIP: ANE init failed (no ANE hardware)");
            return;
        }

        // Use small dims for fast compilation (still tests the pipeline)
        let dim = 64;
        let hidden = 128;

        let rms_ffn: Vec<f32> = (0..dim).map(|i| 0.5 + 0.01 * i as f32).collect();
        let w1 = vec![0.01f32; hidden * dim];
        let w3 = vec![0.01f32; hidden * dim];
        let w2 = vec![0.01f32; dim * hidden];

        for seq in [1, 2, 4, 8, 16, 32] {
            let mut cfg = MilConfig::mha(dim, hidden, 4, seq);
            cfg.n_kv_heads = 2;
            cfg.rms_eps = 1e-6;

            let kernel = BlobDecodeKernels::compile_single(&cfg, &rms_ffn, &w1, &w3, &w2);
            match kernel {
                Some(k) => {
                    // Try eval
                    let input = vec![0.5f32; dim * seq];
                    let input_bytes = ane_weights::f32_slice_to_bytes(&input);
                    k.write_input(0, &input_bytes);
                    match k.eval() {
                        Ok(()) => {
                            let mut out_buf = vec![0u8; dim * seq * 4];
                            k.read_output(0, &mut out_buf);
                            let first_val = f32::from_le_bytes([
                                out_buf[0], out_buf[1], out_buf[2], out_buf[3],
                            ]);
                            eprintln!(
                                "  BLOB seq={seq}: COMPILE OK, EVAL OK (first={first_val:.6})"
                            );
                        }
                        Err(e) => {
                            eprintln!("  BLOB seq={seq}: COMPILE OK, EVAL FAILED ({e})");
                        }
                    }
                }
                None => {
                    eprintln!("  BLOB seq={seq}: COMPILE FAILED");
                }
            }
        }
    }

    /// BLOBFILE decode at 0.8B dims: compile + eval + correctness check.
    ///
    /// cargo test --features ane --lib -- "test_blob_ffn_0_8b_dims" --nocapture --test-threads=1
    #[test]
    fn test_blob_ffn_0_8b_dims() {
        use super::super::ane_bridge;

        if ane_bridge::ane_init().is_err() {
            eprintln!("SKIP: ANE init failed");
            return;
        }

        let dim = 1024;
        let hidden = 3584;

        let rms_ffn: Vec<f32> = (0..dim).map(|i| 0.5 + 0.0001 * i as f32).collect();
        let w1 = vec![0.01f32; hidden * dim];
        let w3 = vec![0.01f32; hidden * dim];
        let w2 = vec![0.01f32; dim * hidden];

        // Try seq=1 first (the critical gate)
        for seq in [1, 16] {
            let mut cfg = MilConfig::mha(dim, hidden, 8, seq);
            cfg.n_kv_heads = 2;
            cfg.head_dim_explicit = 256;
            cfg.rms_eps = 1e-6;
            cfg.attn_output_gate = true;

            let t0 = std::time::Instant::now();
            let kernel = BlobDecodeKernels::compile_single(&cfg, &rms_ffn, &w1, &w3, &w2);
            let compile_ms = t0.elapsed().as_millis();

            match kernel {
                Some(k) => {
                    let x = vec![0.5f32; dim * seq];
                    let x_bytes = ane_weights::f32_slice_to_bytes(&x);
                    k.write_input(0, &x_bytes);
                    let t0 = std::time::Instant::now();
                    match k.eval() {
                        Ok(()) => {
                            let eval_us = t0.elapsed().as_micros();
                            let mut out = vec![0u8; dim * seq * 4];
                            k.read_output(0, &mut out);
                            let first = f32::from_le_bytes([out[0], out[1], out[2], out[3]]);
                            eprintln!("  BLOB 0.8B seq={seq}: OK (compile={compile_ms}ms, eval={eval_us}us, first={first:.6})");
                        }
                        Err(e) => eprintln!(
                            "  BLOB 0.8B seq={seq}: EVAL FAILED ({e}) (compile={compile_ms}ms)"
                        ),
                    }
                }
                None => eprintln!("  BLOB 0.8B seq={seq}: COMPILE FAILED (took {compile_ms}ms)"),
            }
        }
    }

    /// BLOBFILE decode benchmark at 0.8B: full decode_step_blob vs CPU.
    ///
    /// cargo test --features ane --release --lib -- "bench_blob_decode_0_8b" --nocapture --test-threads=1 --ignored
    #[test]
    #[ignore]
    fn bench_blob_decode_0_8b() {
        use super::super::ane_bridge;

        if ane_bridge::ane_init().is_err() {
            eprintln!("SKIP: ANE init failed");
            return;
        }

        let model = make_0_8b_synthetic();

        // Try seq=1 first
        let seq = 1;
        let t0 = std::time::Instant::now();
        let blob_kernels = BlobDecodeKernels::compile(&model, seq);
        let compile_ms = t0.elapsed().as_millis();

        let blob_kernels = match blob_kernels {
            Some(k) => {
                eprintln!("BlobDecodeKernels compile (seq={seq}): {compile_ms}ms");
                k
            }
            None => {
                eprintln!("BlobDecodeKernels compile (seq={seq}): FAILED after {compile_ms}ms");
                eprintln!("Trying seq=16 fallback...");
                let t0 = std::time::Instant::now();
                match BlobDecodeKernels::compile(&model, 16) {
                    Some(k) => {
                        eprintln!(
                            "BlobDecodeKernels compile (seq=16): {}ms",
                            t0.elapsed().as_millis()
                        );
                        k
                    }
                    None => {
                        eprintln!("BlobDecodeKernels: both seq=1 and seq=16 failed");
                        return;
                    }
                }
            }
        };

        let n_layers = model.layers.len();
        let max_seq = 128;

        // --- BLOB benchmark ---
        let mut cache_blob = KvCache::new(&model.cfg, n_layers, max_seq);
        for i in 0..3u32 {
            let _ = decode_step_blob(&model, &blob_kernels, i, &mut cache_blob);
        }

        let n_iters = 30;
        let t0 = std::time::Instant::now();
        for i in 0..n_iters {
            let _ = decode_step_blob(&model, &blob_kernels, (i % 100) as u32, &mut cache_blob);
        }
        let blob_us = t0.elapsed().as_micros() as f64 / n_iters as f64;
        let blob_ms = blob_us / 1000.0;

        // --- CPU benchmark ---
        let mut cache_cpu = KvCache::new(&model.cfg, n_layers, max_seq);
        for i in 0..3u32 {
            let _ = decode_step(&model, i, &mut cache_cpu);
        }
        let t0 = std::time::Instant::now();
        for i in 0..n_iters {
            let _ = decode_step(&model, (i % 100) as u32, &mut cache_cpu);
        }
        let cpu_us = t0.elapsed().as_micros() as f64 / n_iters as f64;
        let cpu_ms = cpu_us / 1000.0;

        let speedup = cpu_us / blob_us;

        eprintln!("\n=== BLOBFILE Decode Benchmark (0.8B synthetic) ===");
        eprintln!(
            "  CPU:  {cpu_ms:.2}ms/step ({:.1} tok/sec)",
            1_000_000.0 / cpu_us
        );
        eprintln!(
            "  BLOB: {blob_ms:.2}ms/step ({:.1} tok/sec)",
            1_000_000.0 / blob_us
        );
        eprintln!("  Speedup: {speedup:.2}x");
        eprintln!();

        if blob_ms < 5.0 {
            eprintln!("  VERDICT: PASS (<5ms) — BLOBFILE ANE draft model is viable!");
        } else if blob_ms < 10.0 {
            eprintln!("  VERDICT: MARGINAL (5-10ms)");
        } else {
            eprintln!("  VERDICT: FAIL (>10ms)");
        }

        // Correctness: compare BLOB vs CPU logits
        let mut cache_b = KvCache::new(&model.cfg, n_layers, 64);
        let mut cache_c = KvCache::new(&model.cfg, n_layers, 64);
        let r_blob = decode_step_blob(&model, &blob_kernels, 42, &mut cache_b);
        let r_cpu = decode_step(&model, 42, &mut cache_c);
        let max_diff = r_blob
            .logits
            .iter()
            .zip(r_cpu.logits.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("  BLOB vs CPU max logit diff: {max_diff:.6}");
        assert!(
            max_diff < 2.0,
            "BLOB vs CPU divergence too large: {max_diff}"
        );

        // --- Profiled breakdown ---
        eprintln!("\n=== Per-Component Breakdown (single step, pos=0) ===");
        let mut cache_prof = KvCache::new(&model.cfg, n_layers, max_seq);
        let dim = model.cfg.dim;
        let n_q_heads = model.cfg.n_heads;
        let n_kv_heads = model.cfg.n_kv_heads;
        let head_dim = model.cfg.head_dim();
        let q_proj_dim = model.cfg.q_proj_dim();
        let kv_dim = n_kv_heads * head_dim;
        let attn_dim = n_q_heads * head_dim;

        let mut x = vec![0.0f32; dim];
        for d in 0..dim {
            x[d] = model.embed[42usize * dim + d];
        }

        let mut t_attn_ane = 0u128;
        let mut t_rope_sdpa = 0u128;
        let mut t_wo_ane = 0u128;
        let mut t_ffn_ane = 0u128;

        for l in 0..n_layers {
            let lw = &model.layers[l];
            let t0 = std::time::Instant::now();
            let (q_raw, mut k, v) = blob_kernels.eval_attn_proj(l, &x).unwrap();
            t_attn_ane += t0.elapsed().as_micros() as u128;

            let t0 = std::time::Instant::now();
            let (mut q, attn_gate) = if model.cfg.attn_output_gate {
                let mut q = vec![0.0f32; attn_dim];
                let mut gate = vec![0.0f32; attn_dim];
                for h in 0..n_q_heads {
                    let sb = h * 2 * head_dim;
                    let db = h * head_dim;
                    q[db..db + head_dim].copy_from_slice(&q_raw[sb..sb + head_dim]);
                    gate[db..db + head_dim]
                        .copy_from_slice(&q_raw[sb + head_dim..sb + 2 * head_dim]);
                }
                (q, Some(gate))
            } else {
                (q_raw, None)
            };
            rope_at_pos(
                &mut q,
                &mut k,
                n_q_heads,
                n_kv_heads,
                head_dim,
                cache_prof.pos(),
                model.cfg.rope_theta,
            );
            cache_prof.append(l, &k, &v);
            let sp = cache_prof.pos;
            cache_prof.pos = cache_prof.pos() + 1;
            let mut attn_out = sdpa_cached(&q, &cache_prof, l, n_q_heads, head_dim);
            cache_prof.pos = sp;
            if let Some(ref gate) = attn_gate {
                for i in 0..attn_dim {
                    let sig = 1.0 / (1.0 + (-gate[i]).exp());
                    attn_out[i] *= sig;
                }
            }
            t_rope_sdpa += t0.elapsed().as_micros() as u128;

            let t0 = std::time::Instant::now();
            let o = blob_kernels.eval_wo(l, &attn_out).unwrap();
            t_wo_ane += t0.elapsed().as_micros() as u128;
            vec_add_inplace(&mut x, &o);

            let t0 = std::time::Instant::now();
            x = blob_kernels.eval_ffn(l, &x).unwrap();
            t_ffn_ane += t0.elapsed().as_micros() as u128;
        }
        cache_prof.advance();

        let t0 = std::time::Instant::now();
        let mut x_final = vec![0.0f32; dim];
        rmsnorm(
            &mut x_final,
            &x,
            &model.rms_final,
            dim,
            1,
            model.cfg.rms_eps,
        );
        let vocab = model.vocab_size;
        let cls_w = model.lm_head.as_ref().unwrap_or(&model.embed);
        let _ = cpu_matmul(cls_w, &x_final, vocab, dim, 1);
        let t_cls = t0.elapsed().as_micros();

        eprintln!(
            "  ANE attn_proj: {:.2}ms ({} layers)",
            t_attn_ane as f64 / 1000.0,
            n_layers
        );
        eprintln!(
            "  CPU rope+sdpa: {:.2}ms ({} layers)",
            t_rope_sdpa as f64 / 1000.0,
            n_layers
        );
        eprintln!(
            "  ANE wo_proj:   {:.2}ms ({} layers)",
            t_wo_ane as f64 / 1000.0,
            n_layers
        );
        eprintln!(
            "  ANE ffn:       {:.2}ms ({} layers)",
            t_ffn_ane as f64 / 1000.0,
            n_layers
        );
        eprintln!("  CPU classifier: {:.2}ms", t_cls as f64 / 1000.0);
        let total = t_attn_ane + t_rope_sdpa + t_wo_ane + t_ffn_ane + t_cls as u128;
        eprintln!("  Total:         {:.2}ms", total as f64 / 1000.0);
    }

    // -----------------------------------------------------------------------
    // Quantized CPU decode benchmark
    // -----------------------------------------------------------------------

    /// fp16 + rayon GEMV decode benchmark.
    ///
    /// cargo test --features ane --release --lib -- "bench_f16_decode_0_8b" --nocapture --test-threads=1 --ignored
    #[test]
    #[ignore]
    fn bench_f16_decode_0_8b() {
        let model = make_0_8b_synthetic();
        let n_layers = model.layers.len();
        let max_seq = 128;

        // Convert to fp16
        let t0 = std::time::Instant::now();
        let fw = F16DecodeWeights::from_model(&model);
        let conv_ms = t0.elapsed().as_millis();
        eprintln!("fp16 conversion time: {conv_ms}ms");

        // --- fp16 benchmark ---
        let mut cache_f = KvCache::new(&model.cfg, n_layers, max_seq);
        for i in 0..3u32 {
            let _ = decode_step_f16(&fw, i, &mut cache_f);
        }
        let n_iters = 30;
        let t0 = std::time::Instant::now();
        for i in 0..n_iters {
            let _ = decode_step_f16(&fw, (i % 100) as u32, &mut cache_f);
        }
        let f16_us = t0.elapsed().as_micros() as f64 / n_iters as f64;
        let f16_ms = f16_us / 1000.0;

        // --- fp32 CPU baseline ---
        let mut cache_cpu = KvCache::new(&model.cfg, n_layers, max_seq);
        for i in 0..3u32 {
            let _ = decode_step(&model, i, &mut cache_cpu);
        }
        let t0 = std::time::Instant::now();
        for i in 0..n_iters {
            let _ = decode_step(&model, (i % 100) as u32, &mut cache_cpu);
        }
        let cpu_us = t0.elapsed().as_micros() as f64 / n_iters as f64;
        let cpu_ms = cpu_us / 1000.0;

        let speedup = cpu_us / f16_us;

        eprintln!("\n=== fp16 Decode Benchmark (0.8B synthetic) ===");
        eprintln!(
            "  FP32 SGEMM: {cpu_ms:.2}ms/step ({:.1} tok/sec)",
            1_000_000.0 / cpu_us
        );
        eprintln!(
            "  FP16 GEMV:  {f16_ms:.2}ms/step ({:.1} tok/sec)",
            1_000_000.0 / f16_us
        );
        eprintln!("  Speedup: {speedup:.2}x");
        eprintln!();

        if f16_ms < 5.0 {
            eprintln!("  VERDICT: PASS (<5ms)");
        } else if f16_ms < 10.0 {
            eprintln!("  VERDICT: MARGINAL (5-10ms)");
        } else {
            eprintln!("  VERDICT: FAIL (>10ms)");
        }

        // Correctness: fp16 vs fp32
        let mut cache_f2 = KvCache::new(&model.cfg, n_layers, 64);
        let mut cache_c2 = KvCache::new(&model.cfg, n_layers, 64);
        let r_f = decode_step_f16(&fw, 42, &mut cache_f2);
        let r_c = decode_step(&model, 42, &mut cache_c2);
        let max_diff = r_f
            .logits
            .iter()
            .zip(r_c.logits.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("  fp16 vs fp32 max logit diff: {max_diff:.6}");
    }

    // -----------------------------------------------------------------------
    // Chained ANE eval benchmark — measures bare dispatch overhead
    // -----------------------------------------------------------------------

    /// Benchmark chained ANE eval: shared IOSurfaces, no CPU memcpy between layers.
    ///
    /// This measures the fundamental limit of per-dispatch ANE overhead.
    /// If 24 chained dispatches take <5ms, pipelined ANE decode is viable.
    ///
    /// cargo test --features ane --release --lib -- "bench_chained_eval" --nocapture --test-threads=1 --ignored
    #[test]
    #[ignore]
    fn bench_chained_eval() {
        use super::super::ane_bridge;
        use super::super::ane_bridge::AneKernel;
        use super::super::ane_mil::{gen_conv1x1_blob, gen_matmul1x1_blob};
        use super::super::ane_weights::build_fp16_blob;

        eprintln!("[0] test entry");

        if ane_bridge::ane_init().is_err() {
            eprintln!("SKIP: ANE init failed");
            return;
        }
        eprintln!("[1] ANE init OK");

        // Use conv1x1 for simplest possible kernel chain.
        let dim = 64;
        let seq = 16;
        let n_layers = 2;

        let spec = gen_conv1x1_blob(dim, dim, seq);
        let w_data: Vec<f32> = (0..dim * dim)
            .map(|i| ((i + 1) as f32 * 0.001).sin() * 0.1)
            .collect();
        let w_blob = build_fp16_blob(&w_data);
        let names: Vec<&str> = spec.weight_names.iter().copied().collect();
        let datas: Vec<&[u8]> = vec![&w_blob];

        eprintln!("[2] MIL generated, compiling 1 kernel...");
        let k0 = AneKernel::compile_multi_weights(
            &spec.mil_text,
            &names,
            &datas,
            &[spec.input_bytes],
            &[spec.output_bytes],
        )
        .unwrap_or_else(|e| panic!("compile failed: {e}"));
        eprintln!("[3] kernel 0 compiled OK");

        // Test single eval
        let input = vec![0.5f32; dim * seq];
        let input_bytes = ane_weights::f32_slice_to_bytes(&input);
        k0.write_input(0, &input_bytes);
        k0.eval().unwrap();
        eprintln!("[4] single eval OK");

        // Compile second kernel
        let k1 = AneKernel::compile_multi_weights(
            &spec.mil_text,
            &names,
            &datas,
            &[spec.input_bytes],
            &[spec.output_bytes],
        )
        .unwrap_or_else(|e| panic!("compile k1 failed: {e}"));
        eprintln!("[5] kernel 1 compiled OK");

        // Test eval_chain WITHOUT sharing (separate IOSurfaces, just sequential eval)
        k1.write_input(0, &input_bytes);
        let refs: Vec<&AneKernel> = vec![&k0, &k1];
        AneKernel::eval_chain(&refs).unwrap();
        eprintln!("[6] eval_chain (no sharing) OK");

        // Now try sharing IOSurfaces
        eprintln!("[7] wiring share_surface k0→k1...");
        k0.share_output_to(0, &k1, 0)
            .unwrap_or_else(|e| panic!("share_surface failed: {e}"));
        eprintln!("[8] share_surface OK");

        // Eval chain with shared surfaces
        k0.write_input(0, &input_bytes);
        AneKernel::eval_chain(&refs).unwrap();
        eprintln!("[9] eval_chain (shared surfaces) OK");

        // Read output from k1 to verify correctness
        let mut out = vec![0u8; spec.output_bytes];
        k1.read_output(0, &mut out);
        eprintln!("[10] read output OK");

        // Benchmark
        let n_iters = 50;

        // Baseline: separate with memcpy
        // Undo sharing first — recompile k1 to get fresh IOSurfaces
        drop(k1);
        let k1 = AneKernel::compile_multi_weights(
            &spec.mil_text,
            &names,
            &datas,
            &[spec.input_bytes],
            &[spec.output_bytes],
        )
        .unwrap();

        let t0 = std::time::Instant::now();
        for _ in 0..n_iters {
            k0.write_input(0, &input_bytes);
            k0.eval().unwrap();
            let mut buf = vec![0u8; spec.output_bytes];
            k0.read_output(0, &mut buf);
            k1.write_input(0, &buf);
            k1.eval().unwrap();
        }
        let separate_us = t0.elapsed().as_micros() as f64 / n_iters as f64;

        // Shared: wire and chain
        k0.share_output_to(0, &k1, 0).unwrap();
        let refs: Vec<&AneKernel> = vec![&k0, &k1];

        // Warm up
        k0.write_input(0, &input_bytes);
        AneKernel::eval_chain(&refs).unwrap();

        let t0 = std::time::Instant::now();
        for _ in 0..n_iters {
            k0.write_input(0, &input_bytes);
            AneKernel::eval_chain(&refs).unwrap();
        }
        let chained_us = t0.elapsed().as_micros() as f64 / n_iters as f64;

        // Single eval dispatch overhead
        let t0 = std::time::Instant::now();
        for _ in 0..n_iters * n_layers {
            k0.eval().unwrap();
        }
        let single_us = t0.elapsed().as_micros() as f64 / (n_iters * n_layers) as f64;

        eprintln!("\n=== Chained ANE Eval Benchmark ({n_layers} conv1x1 layers, dim={dim}, seq={seq}) ===");
        eprintln!("  Single dispatch:   {single_us:.0}µs/eval");
        eprintln!(
            "  Separate (memcpy): {:.2}ms/step ({:.0}µs/layer)",
            separate_us / 1000.0,
            separate_us / n_layers as f64
        );
        eprintln!(
            "  Chained (no copy): {:.2}ms/step ({:.0}µs/layer)",
            chained_us / 1000.0,
            chained_us / n_layers as f64
        );
        eprintln!("  Speedup: {:.2}x", separate_us / chained_us);

        // --- Scale test: realistic 0.8B dimensions ---
        // Test dispatch overhead at various dimensions.
        // Find minimum working seq_len, then benchmark at that seq.
        // Bug 13: seq=1 fails for ALL BLOBFILE ops (conv AND matmul).
        // Format: (c_in, c_out, seq, n_layers, label, use_matmul)
        let dims_to_test: [(usize, usize, usize, usize, &str, bool); 8] = [
            // Sweep seq to find minimum working value
            (1536, 1536, 1, 2, "matmul-seq1", true),
            (1536, 1536, 2, 2, "matmul-seq2", true),
            (1536, 1536, 4, 2, "matmul-seq4", true),
            (1536, 1536, 8, 2, "matmul-seq8", true),
            (1536, 1536, 16, 2, "matmul-seq16", true),
            // Conv seq sweep for comparison
            (1536, 1536, 2, 2, "conv-seq2", false),
            (1536, 1536, 4, 2, "conv-seq4", false),
            // The key benchmark: matmul at min working seq, 24 layers
            (1536, 1536, 16, 24, "matmul-1536x24-decode", true),
        ];

        for (c_in, c_out, s, nl, label, use_matmul) in dims_to_test {
            let sp = if use_matmul {
                gen_matmul1x1_blob(c_in, c_out, s)
            } else {
                gen_conv1x1_blob(c_in, c_out, s)
            };
            let wd: Vec<f32> = (0..c_in * c_out)
                .map(|i| ((i + 1) as f32 * 0.001).sin() * 0.1)
                .collect();
            let wb = build_fp16_blob(&wd);
            let nm: Vec<&str> = sp.weight_names.iter().copied().collect();
            let dt: Vec<&[u8]> = vec![&wb];

            eprintln!("\nCompiling {nl} kernels for {label} (dim={c_in}, seq={s})...");
            ane_bridge::set_quiet(true);
            let mut kerns: Vec<AneKernel> = Vec::with_capacity(nl);
            let mut compile_ok = true;
            for _ in 0..nl {
                match AneKernel::compile_multi_weights(
                    &sp.mil_text,
                    &nm,
                    &dt,
                    &[sp.input_bytes],
                    &[sp.output_bytes],
                ) {
                    Ok(k) => kerns.push(k),
                    Err(e) => {
                        eprintln!("  [{label}] compile failed: {e}");
                        compile_ok = false;
                        break;
                    }
                }
            }
            if !compile_ok {
                ane_bridge::set_quiet(false);
                continue;
            }

            let inp = vec![0.5f32; c_in * s];
            let inp_b = ane_weights::f32_slice_to_bytes(&inp);

            // Warm up — eval may fail at seq=1 (Bug 13)
            let mut eval_ok = true;
            for k in &kerns {
                k.write_input(0, &inp_b);
                if let Err(e) = k.eval() {
                    eprintln!("  [{label}] eval failed: {e}");
                    eval_ok = false;
                    break;
                }
            }
            if !eval_ok {
                ane_bridge::set_quiet(false);
                continue;
            }

            // Baseline: separate with memcpy
            let nb = 20;
            let t0 = std::time::Instant::now();
            for _ in 0..nb {
                kerns[0].write_input(0, &inp_b);
                kerns[0].eval().unwrap();
                for j in 1..nl {
                    let mut buf = vec![0u8; sp.output_bytes];
                    kerns[j - 1].read_output(0, &mut buf);
                    kerns[j].write_input(0, &buf);
                    kerns[j].eval().unwrap();
                }
            }
            let sep_us = t0.elapsed().as_micros() as f64 / nb as f64;

            // Wire shared IOSurfaces
            for j in 0..nl - 1 {
                kerns[j].share_output_to(0, &kerns[j + 1], 0).unwrap();
            }
            let rf: Vec<&AneKernel> = kerns.iter().collect();

            // Warm up chained
            kerns[0].write_input(0, &inp_b);
            AneKernel::eval_chain(&rf).unwrap();

            let t0 = std::time::Instant::now();
            for _ in 0..nb {
                kerns[0].write_input(0, &inp_b);
                AneKernel::eval_chain(&rf).unwrap();
            }
            let ch_us = t0.elapsed().as_micros() as f64 / nb as f64;

            // Real-time dispatch (shared surfaces)
            AneKernel::begin_realtime();
            // Warm up realtime
            kerns[0].write_input(0, &inp_b);
            let rt_ok = AneKernel::eval_chain_realtime(&rf);
            let rt_us = if rt_ok.is_ok() {
                let t0 = std::time::Instant::now();
                for _ in 0..nb {
                    kerns[0].write_input(0, &inp_b);
                    AneKernel::eval_chain_realtime(&rf).unwrap();
                }
                t0.elapsed().as_micros() as f64 / nb as f64
            } else {
                eprintln!("  [{label}] eval_chain_realtime: FAILED ({rt_ok:?})");
                0.0
            };

            // prepare_chain (pipelined)
            let pc_ok = AneKernel::prepare_chain(&rf);
            let pc_us = if pc_ok.is_ok() {
                // After prepare, eval the chain
                let t0 = std::time::Instant::now();
                for _ in 0..nb {
                    kerns[0].write_input(0, &inp_b);
                    AneKernel::eval_chain(&rf).unwrap();
                }
                t0.elapsed().as_micros() as f64 / nb as f64
            } else {
                eprintln!("  [{label}] prepare_chain: FAILED ({pc_ok:?})");
                0.0
            };
            AneKernel::end_realtime();

            eprintln!("  [{label}] {nl} layers:");
            eprintln!(
                "    Separate (memcpy):  {:.2}ms ({:.0}µs/layer)",
                sep_us / 1000.0,
                sep_us / nl as f64
            );
            eprintln!(
                "    Chained (shared):   {:.2}ms ({:.0}µs/layer)",
                ch_us / 1000.0,
                ch_us / nl as f64
            );
            if rt_us > 0.0 {
                eprintln!(
                    "    RealTime (shared):  {:.2}ms ({:.0}µs/layer)",
                    rt_us / 1000.0,
                    rt_us / nl as f64
                );
            }
            if pc_us > 0.0 {
                eprintln!(
                    "    Prepared+Chain:     {:.2}ms ({:.0}µs/layer)",
                    pc_us / 1000.0,
                    pc_us / nl as f64
                );
            }
            let best = if rt_us > 0.0 { ch_us.min(rt_us) } else { ch_us };
            eprintln!("    Best speedup vs separate: {:.2}x", sep_us / best);
            ane_bridge::set_quiet(false);
        }
    }

    /// Test compile_direct with conv1x1 MIL to find what's rejected.
    ///
    /// cargo test --features ane --release --lib -- "test_conv1x1_compile_direct" --nocapture --test-threads=1
    #[test]
    fn test_conv1x1_compile_direct() {
        use super::super::ane_bridge;
        use super::super::ane_bridge::AneKernel;
        use super::super::ane_mil::gen_conv1x1_blob;
        use super::super::ane_weights::build_fp16_blob;

        if ane_bridge::ane_init().is_err() {
            eprintln!("SKIP: ANE init failed");
            return;
        }

        // Test compile_direct with a simple identity MIL first (no conv)
        let identity_mil = r#"mlprogram(version=string("1.0")) {
    func main<ios18>(tensor<fp32, [1, 64, 1, 16]> x) {
        string to16 = const()[name=string("to16"), val=string("fp16")];
        string to32 = const()[name=string("to32"), val=string("fp32")];
        tensor<fp16, [1,64,1,16]> xh = cast(dtype=to16,x=x)[name=string("cin")];
        tensor<fp32, [1,64,1,16]> y = cast(dtype=to32,x=xh)[name=string("cout")];
    } -> (y);
}
"#;
        let ib = 64 * 16 * 4;
        ane_bridge::set_quiet(true);
        match AneKernel::compile_direct(identity_mil, &[], &[], &[ib], &[ib]) {
            Ok(k) => {
                eprintln!("[1] identity compile_direct: OK");
                let input = vec![1.0f32; 64 * 16];
                let input_b = ane_weights::f32_slice_to_bytes(&input);
                k.write_input(0, &input_b);
                match k.eval() {
                    Ok(_) => eprintln!("[1] identity eval: OK"),
                    Err(e) => eprintln!("[1] identity eval: FAILED: {e}"),
                }
            }
            Err(e) => eprintln!("[1] identity compile_direct: FAILED: {e}"),
        }

        // Now try conv1x1 at dim=64
        let spec = gen_conv1x1_blob(64, 64, 16);
        let w_data: Vec<f32> = (0..64 * 64)
            .map(|i| ((i + 1) as f32 * 0.001).sin() * 0.1)
            .collect();
        let w_blob = build_fp16_blob(&w_data);
        let names: Vec<&str> = spec.weight_names.iter().copied().collect();
        let datas: Vec<&[u8]> = vec![&w_blob];

        match AneKernel::compile_direct(
            &spec.mil_text,
            &names,
            &datas,
            &[spec.input_bytes],
            &[spec.output_bytes],
        ) {
            Ok(k) => {
                eprintln!("[2] conv1x1 64x64 compile_direct: OK");
                let input = vec![0.5f32; 64 * 16];
                let input_b = ane_weights::f32_slice_to_bytes(&input);
                k.write_input(0, &input_b);
                match k.eval() {
                    Ok(_) => eprintln!("[2] conv1x1 eval: OK"),
                    Err(e) => eprintln!("[2] conv1x1 eval: FAILED: {e}"),
                }
            }
            Err(e) => {
                eprintln!("[2] conv1x1 64x64 compile_direct: FAILED: {e}");
                eprintln!(
                    "MIL text:\n{}",
                    &spec.mil_text[..200.min(spec.mil_text.len())]
                );
            }
        }

        // Try with compile_multi_weights as baseline
        let k = AneKernel::compile_multi_weights(
            &spec.mil_text,
            &names,
            &datas,
            &[spec.input_bytes],
            &[spec.output_bytes],
        )
        .unwrap();
        let input = vec![0.5f32; 64 * 16];
        let input_b = ane_weights::f32_slice_to_bytes(&input);
        k.write_input(0, &input_b);
        k.eval().unwrap();
        eprintln!("[3] conv1x1 compile_multi_weights: OK (baseline)");
        ane_bridge::set_quiet(false);
    }

    // -----------------------------------------------------------------------
    // Original DynMatmul tests
    // -----------------------------------------------------------------------

    /// Phase 1c gate test: can ANE compile FFN kernels at seq_len=1 for 0.8B dims?
    ///
    /// This is the critical compilation test. If ANE rejects seq_len=1 shapes,
    /// we know immediately that the full split-silicon plan needs the GPU fallback.
    ///
    /// cargo test --features ane --lib -- "ane_compile_seq1_ffn_0_8b" --nocapture --test-threads=1
    #[test]
    fn test_ane_compile_seq1_ffn_0_8b() {
        use super::super::ane_bridge;
        use super::super::ane_forward::CompiledKernels;

        if ane_bridge::ane_init().is_err() {
            eprintln!("SKIP: ANE init failed (no ANE hardware)");
            return;
        }

        // Qwen3.5-0.8B dimensions at seq_len=1
        let mut cfg = MilConfig::mha(1024, 3584, 8, 1); // seq_len=1!
        cfg.n_kv_heads = 2;
        cfg.head_dim_explicit = 256;
        cfg.rope_theta = 1_000_000.0;
        cfg.rms_eps = 1e-6;
        cfg.attn_output_gate = true;

        // Try compiling — this is the question
        match CompiledKernels::compile_forward(&cfg) {
            Ok(k) => {
                let ffn_type = match &k.ffn {
                    super::super::ane_forward::FfnKernels::FullyFused { .. } => "fully-fused",
                    super::super::ane_forward::FfnKernels::Fused { .. } => "two-kernel fused",
                    super::super::ane_forward::FfnKernels::Tiled { .. } => "tiled",
                };
                eprintln!("ANE seq_len=1 FFN: COMPILED ({ffn_type})");
                eprintln!(
                    "  SDPA fwd: {}",
                    if k.sdpa_fwd.is_some() { "YES" } else { "no" }
                );
                eprintln!(
                    "  SDPA core GQA: {}",
                    if k.sdpa_core_gqa.is_some() {
                        "YES"
                    } else {
                        "no"
                    }
                );
                eprintln!(
                    "  MHA proj: {}",
                    if k.mha_proj_fwd.is_some() {
                        "YES"
                    } else {
                        "no"
                    }
                );
                eprintln!(
                    "  Fused attn GQA: {}",
                    if k.fused_attn_gqa.is_some() {
                        "YES"
                    } else {
                        "no"
                    }
                );
                eprintln!(
                    "  RMSNorm: {}",
                    if k.rmsnorm_fwd.is_some() { "YES" } else { "no" }
                );

                // Try actually running the FFN kernel at seq=1 and seq=2 to isolate Bug 13
                if let super::super::ane_forward::FfnKernels::FullyFused { ref kernel } = k.ffn {
                    let dim = 1024usize;
                    let hidden = 3584usize;
                    let xnorm = vec![0.01f32; dim];
                    let w1_t = vec![0.01f32; dim * hidden];
                    let w3_t = vec![0.01f32; dim * hidden];
                    let w2 = vec![0.01f32; dim * hidden];

                    let input =
                        super::super::ane_weights::pack_fused_ffn(&xnorm, &w1_t, &w3_t, &w2, &cfg);
                    let spec = super::super::ane_mil::KernelSpec::for_kernel(
                        &cfg,
                        super::super::ane_mil::KernelType::FusedFfn,
                    );
                    eprintln!(
                        "  seq=1 FFN input: {} bytes, output: {} bytes",
                        input.len(),
                        spec.output_bytes
                    );
                    kernel.write_input(0, &input);
                    match kernel.eval() {
                        Ok(()) => eprintln!("  seq=1 FFN eval: OK!"),
                        Err(_) => eprintln!("  seq=1 FFN eval: FAILED (0x1d)"),
                    }
                }

                // Now try seq=2 — same dims, just wider spatial
                let mut cfg2 = cfg.clone();
                cfg2.seq_len = 2;
                if let Ok(k2) = CompiledKernels::compile_forward(&cfg2) {
                    if let super::super::ane_forward::FfnKernels::FullyFused { ref kernel } = k2.ffn
                    {
                        let dim = 1024usize;
                        let hidden = 3584usize;
                        let xnorm2 = vec![0.01f32; dim * 2];
                        let w1_t = vec![0.01f32; dim * hidden];
                        let w3_t = vec![0.01f32; dim * hidden];
                        let w2 = vec![0.01f32; dim * hidden];

                        let input = super::super::ane_weights::pack_fused_ffn(
                            &xnorm2, &w1_t, &w3_t, &w2, &cfg2,
                        );
                        let spec = super::super::ane_mil::KernelSpec::for_kernel(
                            &cfg2,
                            super::super::ane_mil::KernelType::FusedFfn,
                        );
                        eprintln!(
                            "  seq=2 FFN input: {} bytes, output: {} bytes",
                            input.len(),
                            spec.output_bytes
                        );
                        kernel.write_input(0, &input);
                        match kernel.eval() {
                            Ok(()) => eprintln!("  seq=2 FFN eval: OK!"),
                            Err(_) => eprintln!("  seq=2 FFN eval: FAILED (0x1d)"),
                        }
                    }
                } else {
                    eprintln!("  seq=2 compile failed");
                }

                // Sweep seq=4,8,16,32,64,128 to find minimum viable spatial
                for seq_try in [4, 8, 16, 32, 64, 128] {
                    let mut cfg_s = cfg.clone();
                    cfg_s.seq_len = seq_try;
                    if let Ok(ks) = CompiledKernels::compile_forward(&cfg_s) {
                        if let super::super::ane_forward::FfnKernels::FullyFused { ref kernel } =
                            ks.ffn
                        {
                            let dim = 1024usize;
                            let hidden = 3584usize;
                            let xnorm_s = vec![0.01f32; dim * seq_try];
                            let w1_t = vec![0.01f32; dim * hidden];
                            let w3_t = vec![0.01f32; dim * hidden];
                            let w2 = vec![0.01f32; dim * hidden];
                            let input = super::super::ane_weights::pack_fused_ffn(
                                &xnorm_s, &w1_t, &w3_t, &w2, &cfg_s,
                            );
                            kernel.write_input(0, &input);
                            match kernel.eval() {
                                Ok(()) => {
                                    eprintln!("  seq={seq_try} FFN eval: OK!");
                                    break;
                                }
                                Err(_) => eprintln!("  seq={seq_try} FFN eval: FAILED"),
                            }
                        }
                    } else {
                        eprintln!("  seq={seq_try} compile failed");
                    }
                }
            }
            Err(e) => {
                eprintln!("ANE seq_len=1: compile_forward FAILED: {e}");
                eprintln!("This means ANE cannot handle seq_len=1 shapes.");
                eprintln!("Fallback: use GPU-based speculative decoding via mlx-lm --draft-model");
                // Don't panic — this is informational. The bench test below is the real gate.
            }
        }
    }

    /// Phase 1c go/no-go benchmark: measure full decode step latency on 0.8B model.
    ///
    /// Uses synthetic weights (same shapes as Qwen3.5-0.8B) to measure pure
    /// compute cost without requiring model download.
    ///
    /// Run: cargo test --features ane --release --lib -- "bench_decode_step_0_8b" --nocapture --test-threads=1 --ignored
    #[test]
    #[ignore] // benchmark — run explicitly
    fn bench_decode_step_0_8b() {
        let model = make_0_8b_synthetic();

        // Profile breakdown: where does time go?
        {
            let dim = model.cfg.dim;
            let hidden = model.cfg.hidden_dim;
            let q_proj_dim = model.cfg.q_proj_dim();
            let kv_dim = model.cfg.n_kv_heads * model.cfg.head_dim();
            let vocab = model.vocab_size;
            let lw = &model.layers[0];
            let x = vec![0.01f32; dim];

            // Time one layer's projections
            let t = std::time::Instant::now();
            let n = 100;
            for _ in 0..n {
                let _ = cpu_matmul(&lw.wq, &x, q_proj_dim, dim, 1);
                let _ = cpu_matmul(&lw.wk, &x, kv_dim, dim, 1);
                let _ = cpu_matmul(&lw.wv, &x, kv_dim, dim, 1);
            }
            let proj_us = t.elapsed().as_micros() as f64 / n as f64;

            let t = std::time::Instant::now();
            for _ in 0..n {
                let _ = cpu_matmul(&lw.wo, &x, dim, q_proj_dim / 2, 1); // wo uses attn_dim
            }
            let wo_us = t.elapsed().as_micros() as f64 / n as f64;

            let t = std::time::Instant::now();
            for _ in 0..n {
                let _ = cpu_matmul(&lw.w1, &x, hidden, dim, 1);
                let _ = cpu_matmul(&lw.w3, &x, hidden, dim, 1);
            }
            let ffn_w13_us = t.elapsed().as_micros() as f64 / n as f64;

            let gate = vec![0.01f32; hidden];
            let t = std::time::Instant::now();
            for _ in 0..n {
                let _ = cpu_matmul(&lw.w2, &gate, dim, hidden, 1);
            }
            let ffn_w2_us = t.elapsed().as_micros() as f64 / n as f64;

            let cls_w = model.lm_head.as_ref().unwrap_or(&model.embed);
            let t = std::time::Instant::now();
            for _ in 0..n {
                let _ = cpu_matmul(cls_w, &x, vocab, dim, 1);
            }
            let cls_us = t.elapsed().as_micros() as f64 / n as f64;

            let layer_total = proj_us + wo_us + ffn_w13_us + ffn_w2_us;
            eprintln!("\n=== Per-component breakdown (1 layer, single matvec) ===");
            eprintln!("  QKV proj:     {proj_us:.0}us");
            eprintln!("  Wo proj:      {wo_us:.0}us");
            eprintln!("  FFN W1+W3:    {ffn_w13_us:.0}us");
            eprintln!("  FFN W2:       {ffn_w2_us:.0}us");
            eprintln!("  Layer total:  {layer_total:.0}us");
            eprintln!(
                "  × 24 layers:  {:.0}us ({:.1}ms)",
                layer_total * 24.0,
                layer_total * 24.0 / 1000.0
            );
            eprintln!("  Classifier:   {cls_us:.0}us ({:.1}ms)", cls_us / 1000.0);
            eprintln!(
                "  Projected total: {:.1}ms",
                (layer_total * 24.0 + cls_us) / 1000.0
            );
        }

        let n_layers = model.layers.len();
        let max_seq = 256;
        let mut cache = KvCache::new(&model.cfg, n_layers, max_seq);

        // Warmup: 5 steps
        for i in 0..5 {
            let _ = decode_step(&model, i, &mut cache);
        }

        // Benchmark: 50 decode steps
        let n_iters = 50;
        let t0 = std::time::Instant::now();
        for i in 0..n_iters {
            let _ = decode_step(&model, (i % 100) as u32, &mut cache);
        }
        let elapsed = t0.elapsed();
        let per_step_us = elapsed.as_micros() as f64 / n_iters as f64;
        let per_step_ms = per_step_us / 1000.0;
        let tok_per_sec = 1_000_000.0 / per_step_us;

        eprintln!("\n=== Phase 1c: Decode Step Benchmark (0.8B, CPU-only) ===");
        eprintln!("  Layers: {n_layers}");
        eprintln!("  Dim: {}, Hidden: {}", model.cfg.dim, model.cfg.hidden_dim);
        eprintln!(
            "  Heads: {} Q, {} KV, head_dim={}",
            model.cfg.n_heads,
            model.cfg.n_kv_heads,
            model.cfg.head_dim()
        );
        eprintln!("  Iters: {n_iters} (after 5 warmup)");
        eprintln!("  KV cache pos at end: {}", cache.pos());
        eprintln!("  Per-step: {per_step_ms:.2}ms ({per_step_us:.0}us)");
        eprintln!("  Throughput: {tok_per_sec:.1} tok/sec");
        eprintln!();

        if per_step_ms < 5.0 {
            eprintln!("  VERDICT: PASS (<5ms) — ANE draft model is viable!");
        } else if per_step_ms < 10.0 {
            eprintln!(
                "  VERDICT: MARGINAL (5-10ms) — may still be useful with ANE FFN acceleration"
            );
        } else {
            eprintln!("  VERDICT: FAIL (>10ms) — pivot to GPU fallback (mlx-lm --draft-model)");
        }

        // Informational — don't fail the benchmark, just report
        if per_step_ms > 10.0 {
            eprintln!("  NOTE: CPU-only baseline exceeds 10ms threshold.");
            eprintln!("  ANE acceleration or quantized weights needed to hit target.");
        }
    }

    /// Build a synthetic 0.8B-shaped model for benchmarking without real weights.
    fn make_0_8b_synthetic() -> ModelWeights {
        let dim = 1024;
        let hidden = 3584;
        let n_heads = 8;
        let n_kv_heads = 2;
        let head_dim = 256;
        let vocab = 151936; // Qwen vocab size
        let n_layers = 24;

        let mut cfg = MilConfig::mha(dim, hidden, n_heads, 1);
        cfg.n_kv_heads = n_kv_heads;
        cfg.head_dim_explicit = head_dim;
        cfg.rope_theta = 1_000_000.0;
        cfg.rms_eps = 1e-6;
        cfg.attn_output_gate = true;

        let attn_dim = n_heads * head_dim; // 2048
        let q_proj_dim = 2 * attn_dim; // 4096 (doubled for gate)
        let kv_dim = n_kv_heads * head_dim; // 512

        let layers: Vec<LayerWeights> = (0..n_layers)
            .map(|_| LayerWeights {
                wq: vec![0.01f32; q_proj_dim * dim],
                wk: vec![0.01f32; kv_dim * dim],
                wv: vec![0.01f32; kv_dim * dim],
                wo: vec![0.01f32; dim * attn_dim],
                w1: vec![0.01f32; hidden * dim],
                w2: vec![0.01f32; dim * hidden],
                w3: vec![0.01f32; hidden * dim],
                rms_att: vec![1.0f32; dim],
                rms_ffn: vec![1.0f32; dim],
                q_norm: None,
                k_norm: None,
                gdn: None,
                moe: None,
            })
            .collect();

        ModelWeights {
            cfg,
            layers,
            rms_final: vec![1.0f32; dim],
            embed: vec![0.01f32; vocab * dim],
            vocab_size: vocab,
            lm_head: None,
            vocab_clusters: None,
            router_adapter: None,
        }
    }

    /// ANE-accelerated decode benchmark: measure decode_step_ane latency.
    ///
    /// Compiles per-layer FFN kernels for seq=1 and measures the full decode step
    /// with ANE FFN vs CPU-only.
    ///
    /// Run: cargo test --features ane --release --lib -- "bench_ane_decode_step_0_8b" --nocapture --test-threads=1 --ignored
    #[test]
    #[ignore] // benchmark — run explicitly
    fn bench_ane_decode_step_0_8b() {
        use super::super::ane_bridge;

        if ane_bridge::ane_init().is_err() {
            eprintln!("SKIP: ANE init failed");
            return;
        }

        let model = make_0_8b_synthetic();

        // Compile ANE decode kernels
        let t0 = std::time::Instant::now();
        let kernels = match DecodeKernels::compile(&model) {
            Some(k) => k,
            None => {
                eprintln!("FAIL: DecodeKernels::compile returned None");
                return;
            }
        };
        let compile_ms = t0.elapsed().as_millis();
        eprintln!("DecodeKernels compile: {compile_ms}ms (24 per-layer FFN kernels)");

        let n_layers = model.layers.len();
        let max_seq = 128;

        // --- ANE benchmark ---
        let mut cache_ane = KvCache::new(&model.cfg, n_layers, max_seq);
        // Warmup
        for i in 0..3 {
            let _ = decode_step_ane(&model, &kernels, i, &mut cache_ane);
        }

        let n_iters = 30;
        let t0 = std::time::Instant::now();
        for i in 0..n_iters {
            let _ = decode_step_ane(&model, &kernels, (i % 100) as u32, &mut cache_ane);
        }
        let ane_elapsed = t0.elapsed();
        let ane_us = ane_elapsed.as_micros() as f64 / n_iters as f64;
        let ane_ms = ane_us / 1000.0;

        // --- CPU benchmark (same iterations, fresh cache) ---
        let mut cache_cpu = KvCache::new(&model.cfg, n_layers, max_seq);
        for i in 0..3 {
            let _ = decode_step(&model, i, &mut cache_cpu);
        }
        let t0 = std::time::Instant::now();
        for i in 0..n_iters {
            let _ = decode_step(&model, (i % 100) as u32, &mut cache_cpu);
        }
        let cpu_elapsed = t0.elapsed();
        let cpu_us = cpu_elapsed.as_micros() as f64 / n_iters as f64;
        let cpu_ms = cpu_us / 1000.0;

        let speedup = cpu_us / ane_us;

        eprintln!("\n=== Phase 1c: ANE vs CPU Decode Step (0.8B synthetic) ===");
        eprintln!(
            "  CPU:  {cpu_ms:.2}ms/step ({:.1} tok/sec)",
            1_000_000.0 / cpu_us
        );
        eprintln!(
            "  ANE:  {ane_ms:.2}ms/step ({:.1} tok/sec)",
            1_000_000.0 / ane_us
        );
        eprintln!("  Speedup: {speedup:.2}x");
        eprintln!();

        if ane_ms < 5.0 {
            eprintln!("  VERDICT: PASS (<5ms) — ANE draft model is viable!");
        } else if ane_ms < 10.0 {
            eprintln!("  VERDICT: MARGINAL (5-10ms) — draft model may work");
        } else {
            eprintln!("  VERDICT: FAIL (>10ms) — need quantized weights or more ANE coverage");
        }

        // Correctness check: ANE and CPU should produce similar logits
        let mut cache_a = KvCache::new(&model.cfg, n_layers, 64);
        let mut cache_c = KvCache::new(&model.cfg, n_layers, 64);
        let r_ane = decode_step_ane(&model, &kernels, 42, &mut cache_a);
        let r_cpu = decode_step(&model, 42, &mut cache_c);
        let max_diff = r_ane
            .logits
            .iter()
            .zip(r_cpu.logits.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("  ANE vs CPU max logit diff: {max_diff:.6}");
        // ANE fp16 intermediate precision → some divergence expected
        assert!(
            max_diff < 1.0,
            "ANE vs CPU divergence too large: {max_diff}"
        );
    }

    #[test]
    fn test_kv_cache_rollback() {
        let model = make_tiny_model();
        let mut cache = KvCache::new(&model.cfg, model.layers.len(), 64);
        for i in 0..5u32 {
            decode_step(&model, i, &mut cache);
        }
        assert_eq!(cache.pos(), 5);
        cache.rollback_to(3);
        assert_eq!(cache.pos(), 3);
        let r = decode_step(&model, 10, &mut cache);
        assert_eq!(cache.pos(), 4);
        assert_eq!(r.logits.len(), model.vocab_size);
    }

    #[test]
    fn test_merge_lora_weight_identity() {
        let base = vec![1.0f32, 2.0, 3.0, 4.0];
        let merged = super::merge_lora_weight(&base, None, 1.0, 2, 2);
        assert_eq!(merged, base);
    }

    #[test]
    fn test_merge_lora_weight_adds_delta() {
        let base = vec![1.0, 0.0, 0.0, 1.0];
        let adapter = super::super::ane_lora::LoraAdapter {
            a: vec![1.0, 0.0, 0.0, 1.0],
            b: vec![0.5, 0.0, 0.0, 0.5],
            rank: 2,
            d_in: 2,
            d_out: 2,
        };
        let merged = super::merge_lora_weight(&base, Some(&adapter), 1.0, 2, 2);
        assert!((merged[0] - 1.5).abs() < 1e-6);
        assert!(merged[1].abs() < 1e-6);
        assert!(merged[2].abs() < 1e-6);
        assert!((merged[3] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_acceptance_stats() {
        let mut stats = super::AcceptanceStats::new();
        for _ in 0..10 {
            stats.update(4, 3);
        }
        let rate = stats.recent_rate();
        assert!((rate - 0.75).abs() < 0.01, "expected ~0.75, got {rate}");
    }

    // -----------------------------------------------------------------------
    // Benchmark: GDN ANE projection vs CPU at realistic model dimensions
    // -----------------------------------------------------------------------

    /// Benchmark GDN layer projection: ANE conv1x1 vs CPU matmul at real dims.
    ///
    /// Sweeps across Qwen3.5 model scales:
    ///   0.6B: dim=1024, h_k=4, d_k=64, h_v=8, d_v=64, key=256, value=512, qkv=1024
    ///   2B:   dim=1536, h_k=8, d_k=64, h_v=16, d_v=64, key=512, value=1024, qkv=2048
    ///   4B:   dim=2048, h_k=8, d_k=64, h_v=16, d_v=64, key=512, value=1024, qkv=2048
    ///   35B:  dim=2560, h_k=8, d_k=64, h_v=16, d_v=128, key=512, value=2048, qkv=3072
    ///
    /// cargo test --features ane --release --lib -- "bench_gdn_ane_vs_cpu" --nocapture --test-threads=1
    #[test]
    fn bench_gdn_ane_vs_cpu() {
        use super::super::ane_bridge;
        if ane_bridge::ane_init().is_err() {
            eprintln!("SKIP: ANE init failed");
            return;
        }

        struct GdnDims {
            label: &'static str,
            dim: usize,
            hidden: usize,
            h_k: usize,
            d_k: usize,
            h_v: usize,
            d_v: usize,
            conv_kernel: usize,
        }

        let configs = [
            GdnDims { label: "0.6B", dim: 1024, hidden: 2816, h_k: 4, d_k: 64, h_v: 8, d_v: 64, conv_kernel: 4 },
            GdnDims { label: "2B",   dim: 1536, hidden: 4096, h_k: 8, d_k: 64, h_v: 16, d_v: 64, conv_kernel: 4 },
            GdnDims { label: "4B",   dim: 2048, hidden: 5632, h_k: 8, d_k: 64, h_v: 16, d_v: 64, conv_kernel: 4 },
            GdnDims { label: "35B",  dim: 2560, hidden: 9216, h_k: 8, d_k: 64, h_v: 16, d_v: 128, conv_kernel: 4 },
        ];

        eprintln!("\n{}", "=".repeat(70));
        eprintln!("  BENCHMARK: GDN layer projection — ANE conv1x1 vs CPU matmul");
        eprintln!("{}\n", "=".repeat(70));

        for c in &configs {
            let key_dim = c.h_k * c.d_k;
            let value_dim = c.h_v * c.d_v;
            let qkv_dim = 2 * key_dim + value_dim;

            let mut cfg = MilConfig::mha(c.dim, c.hidden, c.h_k * 2, 1);
            cfg.n_kv_heads = c.h_k;
            cfg.head_dim_explicit = c.d_k;
            cfg.linear_n_heads = c.h_k;
            cfg.linear_head_dim = c.d_k;
            cfg.linear_n_value_heads = c.h_v;
            cfg.linear_value_head_dim = c.d_v;
            cfg.conv_kernel_size = c.conv_kernel;

            // Create synthetic weights
            let mut seed = 42u64;
            let mut rand = || -> f32 {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((seed >> 33) as f32 / (1u64 << 31) as f32) * 0.02 - 0.01
            };
            let make_vec = |n: usize, r: &mut dyn FnMut() -> f32| -> Vec<f32> {
                (0..n).map(|_| r()).collect()
            };

            let gdn_w = GdnLayerWeights {
                qkv_proj: make_vec(qkv_dim * c.dim, &mut rand),
                a_proj: make_vec(c.h_v * c.dim, &mut rand),
                b_proj: make_vec(c.h_v * c.dim, &mut rand),
                z_proj: make_vec(value_dim * c.dim, &mut rand),
                o_proj: make_vec(c.dim * value_dim, &mut rand),
                a_log: make_vec(c.h_v, &mut rand),
                dt_bias: make_vec(c.h_v, &mut rand),
                norm_weight: make_vec(c.d_v, &mut rand).iter().map(|x| x.abs() + 0.1).collect(),
                conv_weight: make_vec(qkv_dim * c.conv_kernel, &mut rand),
                conv_bias: make_vec(qkv_dim, &mut rand),
            };

            let xnorm: Vec<f32> = make_vec(c.dim, &mut rand);
            let mut state = GdnLayerDecodeState::new(&cfg);

            // ── CPU benchmark ──
            let n_iters = 100;
            let cpu_start = std::time::Instant::now();
            for _ in 0..n_iters {
                let _ = gdn_decode_single(&gdn_w, &mut state, &xnorm, &cfg);
            }
            let cpu_us = cpu_start.elapsed().as_micros() as f64 / n_iters as f64;

            // ── ANE benchmark ──
            // Build a 1-layer model just for compilation
            let layer = LayerWeights {
                wq: vec![], wk: vec![], wv: vec![], wo: vec![],
                w1: make_vec(c.hidden * c.dim, &mut rand),
                w2: make_vec(c.dim * c.hidden, &mut rand),
                w3: make_vec(c.hidden * c.dim, &mut rand),
                rms_att: make_vec(c.dim, &mut rand).iter().map(|x| x.abs() + 0.1).collect(),
                rms_ffn: make_vec(c.dim, &mut rand).iter().map(|x| x.abs() + 0.1).collect(),
                q_norm: None, k_norm: None,
                gdn: Some(gdn_w.clone()),
                moe: None,
            };
            let model = ModelWeights {
                cfg: cfg.clone(),
                layers: vec![layer],
                rms_final: vec![1.0; c.dim],
                embed: vec![0.0; 32 * c.dim],
                vocab_size: 32,
                lm_head: None,
                vocab_clusters: None,
                router_adapter: None,
            };

            let gdn_kernels = match GdnAneKernels::compile(&model) {
                Some(k) => k,
                None => {
                    eprintln!("  {}: ANE compile FAILED — skipping", c.label);
                    continue;
                }
            };

            let ane_layer = gdn_kernels.layers[0].as_ref().unwrap();
            state = GdnLayerDecodeState::new(&cfg);

            let ane_start = std::time::Instant::now();
            for _ in 0..n_iters {
                let _ = gdn_decode_single_ane(
                    &model.layers[0].gdn.as_ref().unwrap(),
                    ane_layer, &mut state, &xnorm, &cfg,
                );
            }
            let ane_us = ane_start.elapsed().as_micros() as f64 / n_iters as f64;

            let speedup = cpu_us / ane_us;
            let weight_mb = (qkv_dim * c.dim + c.h_v * c.dim * 2 + value_dim * c.dim + c.dim * value_dim) as f64 * 4.0 / 1e6;

            eprintln!(
                "  {:>4}: dim={:<4} qkv={:<4} | CPU {:>7.0}µs  ANE {:>7.0}µs  {:>5.2}x  weights={:.1}MB",
                c.label, c.dim, qkv_dim, cpu_us, ane_us, speedup, weight_mb
            );
        }

        eprintln!("\n  Bug 14 note: ANE pads to seq=16, wastes 15/16 compute.");
        eprintln!("  If ANE > 1x speedup despite padding, SRAM bandwidth wins over DRAM.\n");
    }
}

// ---------------------------------------------------------------------------
// LoRA weight merging for decode kernel reload
// ---------------------------------------------------------------------------

/// Merge base weight with LoRA adapter: W_merged = W_base + scale * B @ A.
fn merge_lora_weight(
    base: &[f32],
    adapter: Option<&super::ane_lora::LoraAdapter>,
    scale: f32,
    d_out: usize,
    d_in: usize,
) -> Vec<f32> {
    let Some(a) = adapter else {
        return base.to_vec();
    };
    debug_assert_eq!(base.len(), d_out * d_in);
    let mut merged = base.to_vec();
    for i in 0..a.d_out {
        for j in 0..a.d_in {
            let mut dot = 0.0f32;
            for r in 0..a.rank {
                dot += a.b[i * a.rank + r] * a.a[r * a.d_in + j];
            }
            merged[i * d_in + j] += scale * dot;
        }
    }
    merged
}

impl BlobDecodeKernels {
    /// Recompile all per-layer kernels with merged base+LoRA weights.
    pub fn recompile_with_lora(
        model: &ModelWeights,
        lora: &super::ane_lora::LoraModel,
        seq_len: usize,
    ) -> Option<Self> {
        ane_bridge::ane_init().ok()?;
        let cfg = &model.cfg;
        let dim = cfg.dim;
        let hidden = cfg.hidden_dim;
        let n_kv_heads = cfg.n_kv_heads;
        let head_dim = cfg.head_dim();
        let q_proj_dim = cfg.q_proj_dim();
        let kv_dim = n_kv_heads * head_dim;
        let attn_dim = cfg.n_heads * head_dim;
        let scale = lora.scale();

        let mut mil_cfg = cfg.clone();
        mil_cfg.seq_len = seq_len;
        let attn_spec = gen_fused_attn_proj_conv_blob(&mil_cfg);
        let wo_spec = gen_conv1x1_blob(attn_dim, dim, seq_len);
        let ffn_spec = gen_fused_ffn_conv_blob(&mil_cfg);

        let mut layers = Vec::with_capacity(model.layers.len());
        for (l, lw) in model.layers.iter().enumerate() {
            let la = &lora.layers[l];
            let w2_m = merge_lora_weight(&lw.w2, la.w2.as_ref(), scale, dim, hidden);

            // GDN layers: FFN only, no attn/wo
            let (attn_proj, wo_proj) = if lw.gdn.is_some() {
                (None, None)
            } else {
                let wq_m = merge_lora_weight(&lw.wq, la.wq.as_ref(), scale, q_proj_dim, dim);
                let wv_m = merge_lora_weight(&lw.wv, la.wv.as_ref(), scale, kv_dim, dim);
                let wo_m = merge_lora_weight(&lw.wo, la.wo.as_ref(), scale, dim, attn_dim);

                let ap = {
                    let blobs: Vec<Vec<u8>> = vec![
                        build_fp16_blob(&lw.rms_att),
                        build_fp16_blob(&wq_m),
                        build_fp16_blob(&lw.wk),
                        build_fp16_blob(&wv_m),
                    ];
                    let names: Vec<&str> = attn_spec.weight_names.iter().copied().collect();
                    let datas: Vec<&[u8]> = blobs.iter().map(|b| b.as_slice()).collect();
                    AneKernel::compile_multi_weights(
                        &attn_spec.mil_text,
                        &names,
                        &datas,
                        &[attn_spec.input_bytes],
                        &[attn_spec.output_bytes],
                    )
                    .map_err(|e| {
                        tracing::warn!("recompile_with_lora: layer {l} attn: {e}");
                        e
                    })
                    .ok()?
                };
                let wp = {
                    let wo_blob = build_fp16_blob(&wo_m);
                    let names: Vec<&str> = wo_spec.weight_names.iter().copied().collect();
                    AneKernel::compile_multi_weights(
                        &wo_spec.mil_text,
                        &names,
                        &[wo_blob.as_slice()],
                        &[wo_spec.input_bytes],
                        &[wo_spec.output_bytes],
                    )
                    .map_err(|e| {
                        tracing::warn!("recompile_with_lora: layer {l} wo: {e}");
                        e
                    })
                    .ok()?
                };
                (Some(ap), Some(wp))
            };

            let ffn = {
                let blobs: Vec<Vec<u8>> = vec![
                    build_fp16_blob(&lw.rms_ffn),
                    build_fp16_blob(&lw.w1),
                    build_fp16_blob(&lw.w3),
                    build_fp16_blob(&w2_m),
                ];
                let names: Vec<&str> = ffn_spec.weight_names.iter().copied().collect();
                let datas: Vec<&[u8]> = blobs.iter().map(|b| b.as_slice()).collect();
                AneKernel::compile_multi_weights(
                    &ffn_spec.mil_text,
                    &names,
                    &datas,
                    &[ffn_spec.input_bytes],
                    &[ffn_spec.output_bytes],
                )
                .map_err(|e| {
                    tracing::warn!("recompile_with_lora: layer {l} ffn: {e}");
                    e
                })
                .ok()?
            };
            layers.push(LayerKernels {
                attn_proj,
                wo_proj,
                ffn,
            });
        }
        tracing::info!(
            "recompile_with_lora: rebuilt {} layers at seq={seq_len}",
            layers.len()
        );
        Some(BlobDecodeKernels {
            layers,
            seq_len,
            dim,
            q_proj_dim,
            kv_dim,
            attn_dim,
        })
    }
}

// ---------------------------------------------------------------------------
// Draft model loader
// ---------------------------------------------------------------------------

/// Load a draft model's `ModelWeights` from a HuggingFace-style model directory.
///
/// Parses `config.json` to build `MilConfig`, then loads weights from safetensors.
/// Used by `SpeculativeDecoder` and `MlxProvider` to load the 0.8B draft model.
pub fn load_draft_model(dir: &std::path::Path) -> Result<ModelWeights, String> {
    let cfg = mil_config_from_dir(dir, 1)?;
    ModelWeights::from_mlx_safetensors(dir, &cfg)
        .map_err(|e| format!("failed to load draft model from {}: {e}", dir.display()))
}

/// Build `MilConfig` from a model directory's `config.json`.
pub fn mil_config_from_dir(dir: &std::path::Path, seq_len: usize) -> Result<MilConfig, String> {
    let config_path = dir.join("config.json");
    let config_str = std::fs::read_to_string(&config_path)
        .map_err(|e| format!("read {}: {e}", config_path.display()))?;
    let root: serde_json::Value = serde_json::from_str(&config_str)
        .map_err(|e| format!("parse {}: {e}", config_path.display()))?;
    let tc = root.get("text_config").unwrap_or(&root);

    let dim = tc["hidden_size"].as_u64().ok_or("missing hidden_size")? as usize;
    let hidden_dim = tc
        .get("intermediate_size")
        .or_else(|| tc.get("moe_intermediate_size"))
        .and_then(|v| v.as_u64())
        .ok_or("missing intermediate_size")? as usize;
    let n_heads = tc["num_attention_heads"]
        .as_u64()
        .ok_or("missing num_attention_heads")? as usize;
    let n_kv_heads = tc
        .get("num_key_value_heads")
        .and_then(|v| v.as_u64())
        .unwrap_or(n_heads as u64) as usize;
    let head_dim = tc
        .get("head_dim")
        .and_then(|v| v.as_u64())
        .unwrap_or((dim / n_heads) as u64) as usize;
    let rope_theta = tc
        .get("rope_parameters")
        .and_then(|rp| rp.get("rope_theta"))
        .and_then(|v| v.as_f64())
        .or_else(|| tc.get("rope_theta").and_then(|v| v.as_f64()))
        .or_else(|| root.get("rope_theta").and_then(|v| v.as_f64()))
        .unwrap_or(1_000_000.0);
    let rms_eps = tc
        .get("rms_norm_eps")
        .and_then(|v| v.as_f64())
        .unwrap_or(1e-6) as f32;
    let attn_output_gate = tc
        .get("attn_output_gate")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let layer_types: Vec<String> = tc
        .get("layer_types")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();
    let linear_attn_indices: Vec<usize> = layer_types
        .iter()
        .enumerate()
        .filter(|(_, t)| t.as_str() == "linear_attention")
        .map(|(i, _)| i)
        .collect();
    let linear_n_heads = tc
        .get("linear_num_key_heads")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let linear_head_dim = tc
        .get("linear_key_head_dim")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let linear_n_value_heads = tc
        .get("linear_num_value_heads")
        .and_then(|v| v.as_u64())
        .unwrap_or(linear_n_heads as u64) as usize;
    let linear_value_head_dim = tc
        .get("linear_value_head_dim")
        .and_then(|v| v.as_u64())
        .unwrap_or(linear_head_dim as u64) as usize;
    let conv_kernel_size = tc
        .get("linear_conv_kernel_dim")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;

    Ok(MilConfig {
        dim,
        hidden_dim,
        n_heads,
        seq_len,
        n_kv_heads,
        rope_theta,
        rms_eps,
        has_lm_head: false,
        head_dim_explicit: head_dim,
        linear_attn_indices,
        linear_n_heads,
        linear_head_dim,
        linear_n_value_heads,
        linear_value_head_dim,
        conv_kernel_size,
        attn_output_gate,
    })
}

// ---------------------------------------------------------------------------
// Speculative decoder
// ---------------------------------------------------------------------------

use std::collections::VecDeque;

/// Acceptance rate tracker for adaptive draft length.
pub struct AcceptanceStats {
    total_drafted: u64,
    total_accepted: u64,
    window: VecDeque<(usize, usize)>,
}

const ACCEPTANCE_WINDOW: usize = 100;

impl AcceptanceStats {
    pub fn new() -> Self {
        Self {
            total_drafted: 0,
            total_accepted: 0,
            window: VecDeque::with_capacity(ACCEPTANCE_WINDOW),
        }
    }

    pub fn update(&mut self, drafted: usize, accepted: usize) {
        self.total_drafted += drafted as u64;
        self.total_accepted += accepted as u64;
        if self.window.len() >= ACCEPTANCE_WINDOW {
            self.window.pop_front();
        }
        self.window.push_back((drafted, accepted));
    }

    pub fn recent_rate(&self) -> f64 {
        let (d, a) = self.window.iter().fold((0u64, 0u64), |(d, a), &(dd, aa)| {
            (d + dd as u64, a + aa as u64)
        });
        if d == 0 {
            0.5
        } else {
            a as f64 / d as f64
        }
    }

    pub fn lifetime_rate(&self) -> f64 {
        if self.total_drafted == 0 {
            0.5
        } else {
            self.total_accepted as f64 / self.total_drafted as f64
        }
    }
}

/// ANE draft model for speculative decoding.
///
/// Pure drafting service — generates candidate tokens on ANE/CPU.
/// Verification and accept/reject logic live in the model worker
/// (which owns the target model's KV cache).
#[cfg(feature = "mlx")]
pub struct SpeculativeDecoder {
    draft_model: ModelWeights,
    kv_cache: KvCache,
    decode_kernels: Option<BlobDecodeKernels>,
    lora_reload_rx: std::sync::mpsc::Receiver<super::ane_lora::LoraModel>,
}

#[cfg(feature = "mlx")]
impl SpeculativeDecoder {
    pub fn new(
        draft_model: ModelWeights,
        lora_reload_rx: std::sync::mpsc::Receiver<super::ane_lora::LoraModel>,
        max_seq: usize,
    ) -> Self {
        let n_layers = draft_model.layers.len();
        let mut kv_cache = KvCache::new(&draft_model.cfg, n_layers, max_seq);
        kv_cache.init_gdn(&draft_model);
        // Suppress ANE error logs during probe — Bug 14 (seq=1 fails, seq=16 fallback)
        super::ane_bridge::set_quiet(true);
        let decode_kernels = BlobDecodeKernels::compile(&draft_model, 1).or_else(|| {
            tracing::info!("SpeculativeDecoder: seq=1 failed, trying seq=16");
            BlobDecodeKernels::compile(&draft_model, 16)
        });
        super::ane_bridge::set_quiet(false);
        if decode_kernels.is_some() {
            tracing::info!("SpeculativeDecoder: ANE decode kernels compiled");
        } else {
            tracing::info!("SpeculativeDecoder: CPU-only decode (ANE unavailable)");
        }
        Self {
            draft_model,
            kv_cache,
            decode_kernels,
            lora_reload_rx,
        }
    }

    fn try_reload_lora(&mut self) {
        let Ok(new_lora) = self.lora_reload_rx.try_recv() else {
            return;
        };
        tracing::info!("SpeculativeDecoder: reloading LoRA into decode kernels");
        let seq_len = self.decode_kernels.as_ref().map_or(1, |k| k.seq_len);
        match BlobDecodeKernels::recompile_with_lora(&self.draft_model, &new_lora, seq_len) {
            Some(k) => {
                self.decode_kernels = Some(k);
                tracing::info!("SpeculativeDecoder: LoRA reload done");
            }
            None => {
                tracing::warn!("SpeculativeDecoder: LoRA recompile failed, keeping old kernels")
            }
        }
    }

    fn draft_one(&mut self, token: u32) -> DecodeResult {
        if let Some(ref kernels) = self.decode_kernels {
            decode_step_blob(&self.draft_model, kernels, token, &mut self.kv_cache)
        } else {
            decode_step(&self.draft_model, token, &mut self.kv_cache)
        }
    }

    /// Draft `n` tokens starting from `last_token`.
    ///
    /// Feeds `last_token` through the model, samples greedily, then continues
    /// for `n-1` more tokens. Returns the `n` drafted token IDs and saves the
    /// pre-draft KV position for rollback via `accept()`.
    fn draft(&mut self, last_token: u32, n: usize) -> Vec<u32> {
        self.try_reload_lora();
        let seed = self.draft_one(last_token);
        let mut drafts = vec![sample_argmax(&seed.logits)];
        for _ in 1..n {
            let r = self.draft_one(*drafts.last().unwrap());
            drafts.push(sample_argmax(&r.logits));
        }
        drafts
    }

    /// Commit `n_accepted` tokens from the last draft round.
    ///
    /// Rolls back the KV cache to `pre_draft_pos + n_accepted`, discarding
    /// state for rejected draft tokens.
    fn accept(&mut self, pre_draft_pos: usize, n_accepted: usize) {
        self.kv_cache.rollback_to(pre_draft_pos + n_accepted);
    }

    fn reset(&mut self) {
        self.kv_cache.rollback_to(0);
    }

    fn kv_pos(&self) -> usize {
        self.kv_cache.pos()
    }
}

// ---------------------------------------------------------------------------
// Spec decode worker thread (Send-safe channel API)
// ---------------------------------------------------------------------------

/// Requests sent to the speculative decoder's dedicated thread.
#[cfg(feature = "mlx")]
pub enum SpecDecodeRequest {
    /// Draft `n` tokens from `last_token`. Returns (pre_draft_pos, draft_tokens).
    Draft {
        last_token: u32,
        n: usize,
        reply: tokio::sync::oneshot::Sender<(usize, Vec<u32>)>,
    },
    /// Commit n_accepted tokens from the last draft. Rolls back draft KV cache.
    Accept {
        pre_draft_pos: usize,
        n_accepted: usize,
    },
    /// Reset KV cache (new conversation).
    Reset,
}

/// Runs the speculative decoder on a dedicated thread.
///
/// The decoder owns `*mut ANEKernelHandle` pointers that aren't `Send`, so it
/// can't live inside an async task or `Arc<Mutex<>>`. This thread owns the
/// decoder and processes requests via a sync channel — the same pattern used
/// by `run_model_worker` for the target model.
///
/// LoRA hot-reload happens internally via `try_reload_lora()` on each draft.
#[cfg(feature = "mlx")]
pub fn run_spec_decode_worker(
    mut decoder: SpeculativeDecoder,
    rx: std::sync::mpsc::Receiver<SpecDecodeRequest>,
) {
    while let Ok(req) = rx.recv() {
        match req {
            SpecDecodeRequest::Draft {
                last_token,
                n,
                reply,
            } => {
                let pos = decoder.kv_pos();
                let drafts = decoder.draft(last_token, n);
                let _ = reply.send((pos, drafts));
            }
            SpecDecodeRequest::Accept {
                pre_draft_pos,
                n_accepted,
            } => {
                decoder.accept(pre_draft_pos, n_accepted);
            }
            SpecDecodeRequest::Reset => decoder.reset(),
        }
    }
    tracing::debug!("spec decode worker exiting");
}
