//! Forward inference pipeline for ANE transformer training.
//!
//! CPU ops (RMSNorm, embedding, cross-entropy) plus compiled kernel set
//! and full forward pass: embed → N layers → classifier → loss.

use super::ane_bridge::{self, AneKernel};
use super::ane_mil::{self, KernelSpec, KernelType, MilConfig};
use super::ane_weights::{self, ModelWeights};

// ---------------------------------------------------------------------------
// Token type abstraction
// ---------------------------------------------------------------------------

/// Trait for token ID types (u16 for llama2.c, u32 for large-vocab models like Qwen).
pub trait TokenId: Copy + 'static {
    fn as_usize(self) -> usize;
}
impl TokenId for u16 {
    fn as_usize(self) -> usize { self as usize }
}
impl TokenId for u32 {
    fn as_usize(self) -> usize { self as usize }
}

// ---------------------------------------------------------------------------
// CPU operations
// ---------------------------------------------------------------------------

/// RMSNorm: out[d,s] = x[d,s] * w[d] / sqrt(mean(x[:,s]^2) + eps)
///
/// Data layout: [dim, seq] row-major — each row is one dimension across all seq positions.
/// `x[i*seq + t]` = dimension `i`, position `t`.
pub fn rmsnorm(out: &mut [f32], x: &[f32], w: &[f32], dim: usize, seq: usize, eps: f32) {
    debug_assert_eq!(x.len(), dim * seq);
    debug_assert_eq!(out.len(), dim * seq);
    debug_assert_eq!(w.len(), dim);

    // Compute sum of squares per position: ss[t] = sum_i x[i,t]^2
    let mut ss = vec![0.0f32; seq];
    for i in 0..dim {
        let row = &x[i * seq..(i + 1) * seq];
        for t in 0..seq {
            ss[t] += row[t] * row[t];
        }
    }

    // ss[t] = 1/sqrt(ss[t]/dim + eps)
    let inv_dim = 1.0 / dim as f32;
    for t in 0..seq {
        ss[t] = 1.0 / (ss[t] * inv_dim + eps).sqrt();
    }

    // out[i,t] = x[i,t] * ss[t] * w[i]
    for i in 0..dim {
        let x_row = &x[i * seq..(i + 1) * seq];
        let out_row = &mut out[i * seq..(i + 1) * seq];
        for t in 0..seq {
            out_row[t] = x_row[t] * ss[t] * w[i];
        }
    }
}

/// Embedding lookup: out[d,s] = embed[token[s]*dim + d]
///
/// Output layout: [dim, seq] — channels-first for ANE compatibility.
/// embed layout: [vocab, dim] row-major.
pub fn embed_lookup<T: TokenId>(out: &mut [f32], embed: &[f32], tokens: &[T], dim: usize, seq: usize) {
    debug_assert_eq!(tokens.len(), seq);
    debug_assert_eq!(out.len(), dim * seq);

    for t in 0..seq {
        let tok = tokens[t].as_usize();
        for d in 0..dim {
            out[d * seq + t] = embed[tok * dim + d];
        }
    }
}

/// Cross-entropy loss + gradient.
///
/// logits layout: [vocab, seq] — `logits[v*seq + t]` = logit for vocab v at position t.
/// Returns (mean_loss, dlogits) where dlogits has same layout as logits.
/// Gradient: dlogits[v,t] = (softmax[v] - 1{v==target[t]}) / seq.
pub fn cross_entropy_loss<T: TokenId>(
    logits: &[f32],
    targets: &[T],
    vocab: usize,
    seq: usize,
) -> (f32, Vec<f32>) {
    debug_assert_eq!(logits.len(), vocab * seq);
    debug_assert_eq!(targets.len(), seq);

    let mut dlogits = vec![0.0f32; vocab * seq];
    let mut total_loss = 0.0f32;
    let inv_seq = 1.0 / seq as f32;

    for t in 0..seq {
        // Gather column t: col[v] = logits[v*seq + t]
        let mut col = vec![0.0f32; vocab];
        for v in 0..vocab {
            col[v] = logits[v * seq + t];
        }

        // Numerically stable softmax: subtract max
        let max_v = col.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        for v in 0..vocab {
            col[v] = (col[v] - max_v).exp();
        }
        let sum: f32 = col.iter().sum();
        let inv_sum = 1.0 / sum;
        for v in 0..vocab {
            col[v] *= inv_sum;
        }

        // NLL loss
        let tgt = targets[t].as_usize();
        total_loss -= (col[tgt] + 1e-10).ln();

        // Gradient: softmax - one_hot, divided by seq
        col[tgt] -= 1.0;
        for v in 0..vocab {
            col[v] *= inv_seq;
        }

        // Scatter back
        for v in 0..vocab {
            dlogits[v * seq + t] = col[v];
        }
    }

    (total_loss * inv_seq, dlogits)
}

/// Classifier: logits[V,S] = embed[V,D] @ x[D,S] via naive matmul.
///
/// embed layout: [vocab, dim] row-major.
/// x layout: [dim, seq].
/// logits layout: [vocab, seq].
pub fn classifier_forward(
    logits: &mut [f32],
    embed: &[f32],
    x: &[f32],
    vocab: usize,
    dim: usize,
    seq: usize,
) {
    debug_assert_eq!(embed.len(), vocab * dim);
    debug_assert_eq!(x.len(), dim * seq);
    debug_assert_eq!(logits.len(), vocab * seq);

    // logits[v, t] = sum_d embed[v*dim + d] * x[d*seq + t]
    for v in 0..vocab {
        for t in 0..seq {
            let mut acc = 0.0f32;
            for d in 0..dim {
                acc += embed[v * dim + d] * x[d * seq + t];
            }
            logits[v * seq + t] = acc;
        }
    }
}

/// RoPE backward: un-rotate dQ and dK gradients on CPU.
///
/// Forward applied: rot1 = x1*cos - x2*sin, rot2 = x1*sin + x2*cos
/// Inverse (transpose of rotation matrix):
///   dx1 = drot1*cos + drot2*sin
///   dx2 = -drot1*sin + drot2*cos
///
/// Data layout: [dim, seq] where dim = n_heads * head_dim.
/// cos/sin tables: [seq, half_hd] where half_hd = head_dim / 2.
pub fn rope_backward(
    dq: &mut [f32],
    dk: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    seq: usize,
    theta: f64,
) {
    let half_hd = head_dim / 2;
    let dim = n_heads * head_dim;
    debug_assert_eq!(dq.len(), dim * seq);
    debug_assert_eq!(dk.len(), dim * seq);

    // dq/dk layout: [dim, seq] = [n_heads*head_dim, seq]
    // For head h, dimension d within head: channel = h*head_dim + d
    // Split d into first-half (d < half_hd) and second-half (d >= half_hd)
    for h in 0..n_heads {
        for t in 0..seq {
            for i in 0..half_hd {
                let freq = 1.0 / theta.powf(2.0 * i as f64 / head_dim as f64);
                let angle = t as f64 * freq;
                let cos_v = angle.cos() as f32;
                let sin_v = angle.sin() as f32;

                let ch1 = (h * head_dim + i) * seq + t;
                let ch2 = (h * head_dim + half_hd + i) * seq + t;

                // Un-rotate dQ
                let dq1 = dq[ch1];
                let dq2 = dq[ch2];
                dq[ch1] = dq1 * cos_v + dq2 * sin_v;
                dq[ch2] = -dq1 * sin_v + dq2 * cos_v;

                // Un-rotate dK
                let dk1 = dk[ch1];
                let dk2 = dk[ch2];
                dk[ch1] = dk1 * cos_v + dk2 * sin_v;
                dk[ch2] = -dk1 * sin_v + dk2 * cos_v;
            }
        }
    }
}

/// Per-head QK RMSNorm forward on CPU.
///
/// Applies RMSNorm per-head to Q or K after projection, before RoPE.
/// Data layout: [dim, seq] where dim = n_heads * head_dim.
/// norm_w: [head_dim] — shared across heads.
/// Modifies `x` in-place. Returns pre-norm copy for backward.
pub fn qk_rmsnorm_fwd(
    x: &mut [f32],
    norm_w: &[f32],
    n_heads: usize,
    head_dim: usize,
    seq: usize,
    eps: f32,
) -> Vec<f32> {
    let dim = n_heads * head_dim;
    debug_assert_eq!(x.len(), dim * seq);
    debug_assert_eq!(norm_w.len(), head_dim);

    let pre_norm = x.to_vec();

    for h in 0..n_heads {
        for t in 0..seq {
            // Compute sum of squares for this head at this position
            let mut ss = 0.0f32;
            for d in 0..head_dim {
                let idx = (h * head_dim + d) * seq + t;
                ss += x[idx] * x[idx];
            }
            let rrms = 1.0 / (ss / head_dim as f32 + eps).sqrt();
            for d in 0..head_dim {
                let idx = (h * head_dim + d) * seq + t;
                x[idx] = pre_norm[idx] * rrms * norm_w[d];
            }
        }
    }
    pre_norm
}

/// Per-head QK RMSNorm backward on CPU.
///
/// Given gradient w.r.t. normed output (dx_normed), computes gradient w.r.t.
/// pre-norm input (overwrites dx_normed) and accumulates dnorm_w.
pub fn qk_rmsnorm_bwd(
    dx: &mut [f32],       // in: grad w.r.t. normed, out: grad w.r.t. pre-norm
    dnorm_w: &mut [f32],  // accumulated [head_dim]
    pre_norm: &[f32],     // saved pre-norm values
    norm_w: &[f32],       // [head_dim]
    n_heads: usize,
    head_dim: usize,
    seq: usize,
    eps: f32,
) {
    let dim = n_heads * head_dim;
    debug_assert_eq!(dx.len(), dim * seq);
    debug_assert_eq!(pre_norm.len(), dim * seq);

    let inv_hd = 1.0 / head_dim as f32;

    for h in 0..n_heads {
        for t in 0..seq {
            // Recompute rrms
            let mut ss = 0.0f32;
            for d in 0..head_dim {
                let idx = (h * head_dim + d) * seq + t;
                ss += pre_norm[idx] * pre_norm[idx];
            }
            let rrms = 1.0 / (ss * inv_hd + eps).sqrt();

            // dot = sum_d(dy[d] * x[d] * w[d]) * rrms^2 / hd
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                let idx = (h * head_dim + d) * seq + t;
                dot += dx[idx] * pre_norm[idx] * norm_w[d];
            }
            dot *= rrms * rrms * inv_hd;

            // dx[d] = rrms * (w[d]*dy[d] - x[d]*dot), dnorm_w[d] += dy[d]*x[d]*rrms
            for d in 0..head_dim {
                let idx = (h * head_dim + d) * seq + t;
                let dy_val = dx[idx];
                dx[idx] = rrms * (norm_w[d] * dy_val - pre_norm[idx] * dot);
                dnorm_w[d] += dy_val * pre_norm[idx] * rrms;
            }
        }
    }
}

/// Vector add in-place: dst[i] += src[i] (residual connections).
pub fn vec_add_inplace(dst: &mut [f32], src: &[f32]) {
    debug_assert_eq!(dst.len(), src.len());
    for i in 0..dst.len() {
        dst[i] += src[i];
    }
}

// ---------------------------------------------------------------------------
// Compiled kernel set
// ---------------------------------------------------------------------------

/// Pre-compiled ANE kernels (compile once at init, reuse every step).
pub struct CompiledKernels {
    pub sdpa_fwd: AneKernel,
    pub ffn_w13: AneKernel,
    pub ffn_w2: AneKernel,
    pub mask_blob: Vec<u8>,
    pub rope_cos_blob: Vec<u8>,
    pub rope_sin_blob: Vec<u8>,
}

impl CompiledKernels {
    /// Compile all forward-pass kernels for the given config.
    pub fn compile_forward(cfg: &MilConfig) -> Result<Self, String> {
        ane_bridge::ane_init()?;

        // SDPA forward needs causal mask + RoPE cos/sin as static weights
        let mask_blob = ane_mil::build_causal_mask_blob(cfg.seq_len);
        let (rope_cos_blob, rope_sin_blob) =
            ane_weights::generate_rope_blobs(cfg.seq_len, cfg.head_dim(), cfg.rope_theta);
        let sdpa_spec = KernelSpec::for_kernel(cfg, KernelType::SdpaFwd);
        let sdpa_fwd = AneKernel::compile_multi_weights(
            &sdpa_spec.mil_text,
            &[
                "@model_path/weights/mask.bin",
                "@model_path/weights/rope_cos.bin",
                "@model_path/weights/rope_sin.bin",
            ],
            &[&mask_blob, &rope_cos_blob, &rope_sin_blob],
            &[sdpa_spec.input_bytes],
            &[sdpa_spec.output_bytes],
        )?;

        // FFN W1+W3 (no static weights)
        let w13_spec = KernelSpec::for_kernel(cfg, KernelType::FfnW13);
        let ffn_w13 = AneKernel::compile(
            &w13_spec.mil_text,
            None,
            &[w13_spec.input_bytes],
            &[w13_spec.output_bytes],
        )?;

        // FFN W2 (no static weights)
        let w2_spec = KernelSpec::for_kernel(cfg, KernelType::FfnW2);
        let ffn_w2 = AneKernel::compile(
            &w2_spec.mil_text,
            None,
            &[w2_spec.input_bytes],
            &[w2_spec.output_bytes],
        )?;

        Ok(Self {
            sdpa_fwd,
            ffn_w13,
            ffn_w2,
            mask_blob,
            rope_cos_blob,
            rope_sin_blob,
        })
    }
}

// ---------------------------------------------------------------------------
// Forward pass types
// ---------------------------------------------------------------------------

/// Per-layer activations saved for backward pass.
pub struct LayerActivations {
    pub layer_in: Vec<f32>,  // [dim, seq]
    pub xnorm: Vec<f32>,     // [dim, seq]
    pub q: Vec<f32>,         // [dim, seq] (post-norm, post-rope)
    pub k: Vec<f32>,         // [dim, seq] (post-norm, post-rope)
    pub v: Vec<f32>,         // [dim, seq]
    pub attn_out: Vec<f32>,  // [dim, seq]
    pub o_out: Vec<f32>,     // [dim, seq]
    pub x2: Vec<f32>,        // [dim, seq]
    pub x2norm: Vec<f32>,    // [dim, seq]
    pub h1: Vec<f32>,        // [hidden, seq]
    pub h3: Vec<f32>,        // [hidden, seq]
    pub gate: Vec<f32>,      // [hidden, seq]  (silu(h1)*h3)
    pub ffn_out: Vec<f32>,   // [dim, seq]
    pub q_pre_norm: Option<Vec<f32>>,  // [dim, seq] pre-QK-norm Q (for backward)
    pub k_pre_norm: Option<Vec<f32>>,  // [dim, seq] pre-QK-norm K (for backward)
}

/// Forward pass result.
pub struct ForwardResult {
    pub logits: Vec<f32>,               // [vocab, seq]
    pub loss: f32,
    pub dlogits: Vec<f32>,              // [vocab, seq]
    pub layer_acts: Vec<LayerActivations>,
}

/// Forward pass result with optional LoRA activations.
pub struct ForwardResultWithLora {
    pub base: ForwardResult,
    pub lora_acts: Vec<super::ane_lora::LoraLayerActivations>,
}

/// Run full forward pass: embed → layers → classifier → loss.
///
/// Follows train.m lines 400-506:
/// 1. embed_lookup → x_cur[dim, seq]
/// 2. Per layer: rmsnorm → SDPA(ANE) → residual → rmsnorm → FFN(ANE) → residual
/// 3. Final rmsnorm → classifier → cross-entropy loss
///
/// When `lora` and `lora_kernels` are provided, LoRA deltas are injected:
/// - After Wo (SDPA output): o_out += scale * B_wo @ (A_wo @ attn_out)
/// - After FFN W2: ffn_out += scale * B_w2 @ (A_w2 @ gate)
pub fn forward<T: TokenId>(
    kernels: &CompiledKernels,
    model: &ModelWeights,
    tokens: &[T],
    targets: &[T],
) -> Result<ForwardResult, String> {
    forward_with_lora(kernels, model, None, None, tokens, targets)
        .map(|r| r.base)
}

/// Forward pass with optional LoRA adapters.
pub fn forward_with_lora<T: TokenId>(
    kernels: &CompiledKernels,
    model: &ModelWeights,
    lora: Option<&super::ane_lora::LoraModel>,
    lora_kernels: Option<&super::ane_lora::LoraKernels>,
    tokens: &[T],
    targets: &[T],
) -> Result<ForwardResultWithLora, String> {
    use super::ane_lora::{self, LoraLayerActivations};

    let cfg = &model.cfg;
    let dim = cfg.dim;
    let seq = cfg.seq_len;
    let _hidden = cfg.hidden_dim;
    let n_layers = model.layers.len();
    let lora_scale = lora.map_or(0.0, |l| l.scale());

    // 1. Embedding lookup
    let mut x_cur = vec![0.0f32; dim * seq];
    embed_lookup(&mut x_cur, &model.embed, tokens, dim, seq);

    let mut layer_acts = Vec::with_capacity(n_layers);
    let mut lora_acts_vec = Vec::with_capacity(n_layers);

    // 2. Transformer layers
    for l in 0..n_layers {
        let lw = &model.layers[l];
        let lora_layer = lora.map(|lm| &lm.layers[l]);

        // Save layer input for backward pass
        let layer_in = x_cur.clone();

        // RMSNorm before attention
        let mut xnorm = vec![0.0f32; dim * seq];
        rmsnorm(&mut xnorm, &x_cur, &lw.rms_att, dim, seq, cfg.rms_eps);

        // SDPA forward on ANE
        let sdpa_input = ane_weights::pack_sdpa_fwd(
            &xnorm, &lw.wq, &lw.wk, &lw.wv, &lw.wo, cfg,
        );
        let sdpa_spec = KernelSpec::for_kernel(cfg, KernelType::SdpaFwd);
        kernels.sdpa_fwd.write_input(0, &sdpa_input);
        kernels.sdpa_fwd.eval()?;
        let mut sdpa_out = vec![0u8; sdpa_spec.output_bytes];
        kernels.sdpa_fwd.read_output(0, &mut sdpa_out);
        let [mut o_out, q, k, v, attn_out, _xnorm_pass] =
            ane_weights::unpack_sdpa_fwd(&sdpa_out, cfg);

        let mut lora_layer_acts = LoraLayerActivations::empty();

        // LoRA on Wo: o_out += scale * B_wo @ (A_wo @ attn_out)
        if let (Some(ll), Some(lk)) = (lora_layer, lora_kernels) {
            if let Some(wo_adapter) = ll.wo.as_ref() {
                let (wo_delta, wo_h) = if lora_kernels.is_some() {
                    ane_lora::lora_forward_ane(
                        &lk.attn_a_fwd, &lk.attn_b_fwd,
                        wo_adapter, &attn_out, seq,
                    )?
                } else {
                    wo_adapter.forward_cpu(&attn_out, seq)
                };
                ane_lora::vec_add_scaled(&mut o_out, &wo_delta, lora_scale);
                lora_layer_acts.wo_x = Some(attn_out.clone());
                lora_layer_acts.wo_h = Some(wo_h);
            }
        }

        // Residual: x2 = x_cur + o_out
        let mut x2 = x_cur.clone();
        vec_add_inplace(&mut x2, &o_out);

        // RMSNorm before FFN
        let mut x2norm = vec![0.0f32; dim * seq];
        rmsnorm(&mut x2norm, &x2, &lw.rms_ffn, dim, seq, cfg.rms_eps);

        // FFN W1+W3 on ANE
        let w13_input = ane_weights::pack_ffn_w13(&x2norm, &lw.w1, &lw.w3, cfg);
        let w13_spec = KernelSpec::for_kernel(cfg, KernelType::FfnW13);
        kernels.ffn_w13.write_input(0, &w13_input);
        kernels.ffn_w13.eval()?;
        let mut w13_out = vec![0u8; w13_spec.output_bytes];
        kernels.ffn_w13.read_output(0, &mut w13_out);
        let (h1, h3, gate) = ane_weights::unpack_ffn_w13(&w13_out, cfg);

        // FFN W2 on ANE
        let w2_input = ane_weights::pack_ffn_w2(&gate, &lw.w2, cfg);
        let w2_spec = KernelSpec::for_kernel(cfg, KernelType::FfnW2);
        kernels.ffn_w2.write_input(0, &w2_input);
        kernels.ffn_w2.eval()?;
        let mut w2_out = vec![0u8; w2_spec.output_bytes];
        kernels.ffn_w2.read_output(0, &mut w2_out);
        // FFN W2 output is [1, dim, 1, seq] fp32
        let mut ffn_out: Vec<f32> = w2_out
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // LoRA on W2: ffn_out += scale * B_w2 @ (A_w2 @ gate)
        if let (Some(ll), Some(lk)) = (lora_layer, lora_kernels) {
            if let Some(w2_adapter) = ll.w2.as_ref() {
                let (w2_delta, w2_h) = if lora_kernels.is_some() {
                    ane_lora::lora_forward_ane(
                        &lk.ffn_a_fwd, &lk.ffn_b_fwd,
                        w2_adapter, &gate, seq,
                    )?
                } else {
                    w2_adapter.forward_cpu(&gate, seq)
                };
                ane_lora::vec_add_scaled(&mut ffn_out, &w2_delta, lora_scale);
                lora_layer_acts.w2_x = Some(gate.clone());
                lora_layer_acts.w2_h = Some(w2_h);
            }
        }

        // Residual: x_cur = x2 + ffn_out
        x_cur = x2.clone();
        vec_add_inplace(&mut x_cur, &ffn_out);

        layer_acts.push(LayerActivations {
            layer_in,
            xnorm,
            q,
            k,
            v,
            attn_out,
            o_out,
            x2,
            x2norm,
            h1,
            h3,
            gate,
            ffn_out,
            q_pre_norm: None,
            k_pre_norm: None,
        });
        lora_acts_vec.push(lora_layer_acts);
    }

    // 3. Final RMSNorm
    let mut x_final = vec![0.0f32; dim * seq];
    rmsnorm(&mut x_final, &x_cur, &model.rms_final, dim, seq, cfg.rms_eps);

    // 4. Classifier (use lm_head if untied, else share embed)
    let vocab = model.vocab_size;
    let mut logits = vec![0.0f32; vocab * seq];
    let cls_w = model.lm_head.as_ref().unwrap_or(&model.embed);
    classifier_forward(&mut logits, cls_w, &x_final, vocab, dim, seq);

    // 5. Cross-entropy loss
    let (loss, dlogits) = cross_entropy_loss(&logits, targets, vocab, seq);

    Ok(ForwardResultWithLora {
        base: ForwardResult {
            logits,
            loss,
            dlogits,
            layer_acts,
        },
        lora_acts: lora_acts_vec,
    })
}

// ---------------------------------------------------------------------------
// CPU-only forward pass (no ANE — for large-dim models where dynamic packing
// exceeds IOSurface limits, and for correctness testing)
// ---------------------------------------------------------------------------

/// CPU matmul: out[M,S] = W[M,N] @ x[N,S] (row-major W, channels-first x).
///
/// W: [M, N] row-major — W[m*N + n]
/// x: [N, S] channels-first — x[n*S + s]
/// out: [M, S] — out[m*S + s]
pub fn cpu_matmul(w: &[f32], x: &[f32], m: usize, n: usize, s: usize) -> Vec<f32> {
    debug_assert_eq!(w.len(), m * n);
    debug_assert_eq!(x.len(), n * s);
    let mut out = vec![0.0f32; m * s];
    for mi in 0..m {
        for si in 0..s {
            let mut acc = 0.0f32;
            for ni in 0..n {
                acc += w[mi * n + ni] * x[ni * s + si];
            }
            out[mi * s + si] = acc;
        }
    }
    out
}

/// CPU RoPE rotation (half-convention, channels-first layout).
///
/// q/k: [dim, seq] where dim = n_heads * head_dim.
/// Modifies q and k in-place.
fn cpu_rope(q: &mut [f32], k: &mut [f32], n_heads: usize, head_dim: usize, seq: usize, theta: f64) {
    let half_hd = head_dim / 2;
    for h in 0..n_heads {
        for t in 0..seq {
            for i in 0..half_hd {
                let freq = 1.0 / theta.powf(2.0 * i as f64 / head_dim as f64);
                let angle = t as f64 * freq;
                let cos_v = angle.cos() as f32;
                let sin_v = angle.sin() as f32;

                let ch1 = (h * head_dim + i) * seq + t;
                let ch2 = (h * head_dim + half_hd + i) * seq + t;

                let q1 = q[ch1]; let q2 = q[ch2];
                q[ch1] = q1 * cos_v - q2 * sin_v;
                q[ch2] = q1 * sin_v + q2 * cos_v;

                let k1 = k[ch1]; let k2 = k[ch2];
                k[ch1] = k1 * cos_v - k2 * sin_v;
                k[ch2] = k1 * sin_v + k2 * cos_v;
            }
        }
    }
}

/// CPU scaled dot-product attention with causal mask.
///
/// q, k, v: [dim, seq] where dim = n_heads * head_dim.
/// Returns: attn_out [dim, seq].
fn cpu_sdpa(q: &[f32], k: &[f32], v: &[f32], n_heads: usize, head_dim: usize, seq: usize) -> Vec<f32> {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut out = vec![0.0f32; n_heads * head_dim * seq];

    for h in 0..n_heads {
        // Compute scores[s1, s2] = sum_d q[h,d,s1] * k[h,d,s2] * scale
        let mut scores = vec![0.0f32; seq * seq];
        for s1 in 0..seq {
            for s2 in 0..seq {
                if s2 > s1 {
                    scores[s1 * seq + s2] = f32::NEG_INFINITY; // causal mask
                } else {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        let qi = q[(h * head_dim + d) * seq + s1];
                        let ki = k[(h * head_dim + d) * seq + s2];
                        dot += qi * ki;
                    }
                    scores[s1 * seq + s2] = dot * scale;
                }
            }
        }

        // Softmax per row
        for s1 in 0..seq {
            let row = &mut scores[s1 * seq..(s1 + 1) * seq];
            let max_v = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for v in row.iter_mut() {
                *v = (*v - max_v).exp();
                sum += *v;
            }
            for v in row.iter_mut() {
                *v /= sum;
            }
        }

        // out[h,d,s1] = sum_s2 scores[s1,s2] * v[h,d,s2]
        for d in 0..head_dim {
            for s1 in 0..seq {
                let mut acc = 0.0f32;
                for s2 in 0..seq {
                    acc += scores[s1 * seq + s2] * v[(h * head_dim + d) * seq + s2];
                }
                out[(h * head_dim + d) * seq + s1] = acc;
            }
        }
    }
    out
}

/// CPU SiLU activation: silu(x) = x * sigmoid(x).
fn cpu_silu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = *v / (1.0 + (-*v).exp());
    }
}

/// CPU-only forward pass for large-dim models.
///
/// No ANE kernels needed — all ops run on CPU. Slower but works at any dimension.
/// Supports QK-norm (Qwen3). Supports optional LoRA adapters.
pub fn forward_cpu<T: TokenId>(
    model: &ModelWeights,
    lora: Option<&super::ane_lora::LoraModel>,
    tokens: &[T],
    targets: &[T],
) -> ForwardResultWithLora {
    use super::ane_lora::LoraLayerActivations;

    let cfg = &model.cfg;
    let dim = cfg.dim;
    let hidden = cfg.hidden_dim;
    let seq = cfg.seq_len;
    let n_heads = cfg.n_heads;
    let head_dim = cfg.head_dim();
    let n_layers = model.layers.len();
    let lora_scale = lora.map_or(0.0, |l| l.scale());

    // 1. Embedding
    let mut x_cur = vec![0.0f32; dim * seq];
    embed_lookup(&mut x_cur, &model.embed, tokens, dim, seq);

    let mut layer_acts = Vec::with_capacity(n_layers);
    let mut lora_acts_vec = Vec::with_capacity(n_layers);

    // 2. Transformer layers
    for l in 0..n_layers {
        let lw = &model.layers[l];
        let lora_layer = lora.map(|lm| &lm.layers[l]);

        let layer_in = x_cur.clone();

        // RMSNorm before attention
        let mut xnorm = vec![0.0f32; dim * seq];
        rmsnorm(&mut xnorm, &x_cur, &lw.rms_att, dim, seq, cfg.rms_eps);

        // QKV projections on CPU
        let mut q = cpu_matmul(&lw.wq, &xnorm, dim, dim, seq);
        let mut k = cpu_matmul(&lw.wk, &xnorm, dim, dim, seq);
        let v = cpu_matmul(&lw.wv, &xnorm, dim, dim, seq);

        // QK-norm (Qwen3)
        let q_pre_norm = if let Some(q_norm_w) = &lw.q_norm {
            Some(qk_rmsnorm_fwd(&mut q, q_norm_w, n_heads, head_dim, seq, cfg.rms_eps))
        } else { None };
        let k_pre_norm = if let Some(k_norm_w) = &lw.k_norm {
            Some(qk_rmsnorm_fwd(&mut k, k_norm_w, n_heads, head_dim, seq, cfg.rms_eps))
        } else { None };

        // RoPE
        cpu_rope(&mut q, &mut k, n_heads, head_dim, seq, cfg.rope_theta);

        // Attention
        let attn_out = cpu_sdpa(&q, &k, &v, n_heads, head_dim, seq);

        // Wo projection
        let mut o_out = cpu_matmul(&lw.wo, &attn_out, dim, dim, seq);

        let mut lora_layer_acts = LoraLayerActivations::empty();

        // LoRA on Wo
        if let Some(ll) = lora_layer {
            if let Some(wo_adapter) = ll.wo.as_ref() {
                let (wo_delta, wo_h) = wo_adapter.forward_cpu(&attn_out, seq);
                super::ane_lora::vec_add_scaled(&mut o_out, &wo_delta, lora_scale);
                lora_layer_acts.wo_x = Some(attn_out.clone());
                lora_layer_acts.wo_h = Some(wo_h);
            }
        }

        // Residual
        let mut x2 = x_cur.clone();
        vec_add_inplace(&mut x2, &o_out);

        // RMSNorm before FFN
        let mut x2norm = vec![0.0f32; dim * seq];
        rmsnorm(&mut x2norm, &x2, &lw.rms_ffn, dim, seq, cfg.rms_eps);

        // FFN: gate = silu(W1 @ x2norm) * (W3 @ x2norm)
        let mut h1 = cpu_matmul(&lw.w1, &x2norm, hidden, dim, seq);
        let h3 = cpu_matmul(&lw.w3, &x2norm, hidden, dim, seq);
        cpu_silu_inplace(&mut h1);
        let mut gate = vec![0.0f32; hidden * seq];
        for i in 0..hidden * seq {
            gate[i] = h1[i] * h3[i];
        }

        // FFN W2
        let mut ffn_out = cpu_matmul(&lw.w2, &gate, dim, hidden, seq);

        // LoRA on W2
        if let Some(ll) = lora_layer {
            if let Some(w2_adapter) = ll.w2.as_ref() {
                let (w2_delta, w2_h) = w2_adapter.forward_cpu(&gate, seq);
                super::ane_lora::vec_add_scaled(&mut ffn_out, &w2_delta, lora_scale);
                lora_layer_acts.w2_x = Some(gate.clone());
                lora_layer_acts.w2_h = Some(w2_h);
            }
        }

        // Residual
        x_cur = x2.clone();
        vec_add_inplace(&mut x_cur, &ffn_out);

        layer_acts.push(LayerActivations {
            layer_in, xnorm, q, k, v, attn_out, o_out, x2, x2norm,
            h1, h3, gate, ffn_out, q_pre_norm, k_pre_norm,
        });
        lora_acts_vec.push(lora_layer_acts);
    }

    // 3. Final RMSNorm
    let mut x_final = vec![0.0f32; dim * seq];
    rmsnorm(&mut x_final, &x_cur, &model.rms_final, dim, seq, cfg.rms_eps);

    // 4. Classifier
    let vocab = model.vocab_size;
    let mut logits = vec![0.0f32; vocab * seq];
    let cls_w = model.lm_head.as_ref().unwrap_or(&model.embed);
    classifier_forward(&mut logits, cls_w, &x_final, vocab, dim, seq);

    // 5. Cross-entropy loss
    let (loss, dlogits) = cross_entropy_loss(&logits, targets, vocab, seq);

    ForwardResultWithLora {
        base: ForwardResult { logits, loss, dlogits, layer_acts },
        lora_acts: lora_acts_vec,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Round 1: rmsnorm + embed_lookup
    // -----------------------------------------------------------------------

    #[test]
    fn test_rmsnorm_known_values() {
        // 4x4 input: dim=4, seq=4, all ones
        let dim = 4;
        let seq = 4;
        let x = vec![1.0f32; dim * seq];
        let w = vec![1.0f32; dim];
        let mut out = vec![0.0f32; dim * seq];

        rmsnorm(&mut out, &x, &w, dim, seq, 1e-5);

        // For all-ones: mean(x^2) = 1.0, so rrms = 1/sqrt(1.0 + 1e-5) ≈ 0.99999...
        // out[i,t] = 1.0 * rrms * 1.0 ≈ 1.0
        let expected = 1.0 / (1.0f32 + 1e-5).sqrt();
        for &v in &out {
            assert!((v - expected).abs() < 1e-4, "got {v}, expected {expected}");
        }
    }

    #[test]
    fn test_rmsnorm_varying_weights() {
        let dim = 2;
        let seq = 2;
        // x = [[1, 2], [3, 4]] as [dim=2, seq=2]
        // x[0,0]=1, x[0,1]=2, x[1,0]=3, x[1,1]=4
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let w = vec![2.0, 0.5];
        let mut out = vec![0.0f32; dim * seq];

        rmsnorm(&mut out, &x, &w, dim, seq, 1e-5);

        // Position 0: ss = (1^2 + 3^2)/2 = 5.0, rrms = 1/sqrt(5+1e-5)
        // Position 1: ss = (2^2 + 4^2)/2 = 10.0, rrms = 1/sqrt(10+1e-5)
        let rrms0 = 1.0 / (5.0f32 + 1e-5).sqrt();
        let rrms1 = 1.0 / (10.0f32 + 1e-5).sqrt();

        // out[0,0] = 1*rrms0*2 = 2*rrms0
        assert!((out[0] - 1.0 * rrms0 * 2.0).abs() < 1e-4);
        // out[0,1] = 2*rrms1*2 = 4*rrms1
        assert!((out[1] - 2.0 * rrms1 * 2.0).abs() < 1e-4);
        // out[1,0] = 3*rrms0*0.5 = 1.5*rrms0
        assert!((out[2] - 3.0 * rrms0 * 0.5).abs() < 1e-4);
        // out[1,1] = 4*rrms1*0.5 = 2*rrms1
        assert!((out[3] - 4.0 * rrms1 * 0.5).abs() < 1e-4);
    }

    #[test]
    fn test_embed_lookup_maps_tokens_correctly() {
        let dim = 3;
        let seq = 2;
        let vocab = 4;
        // embed[vocab=4, dim=3] row-major: each row is an embedding vector
        let embed: Vec<f32> = (0..vocab * dim).map(|i| i as f32).collect();
        let tokens = [2u16, 0u16]; // look up rows 2 and 0

        let mut out = vec![0.0f32; dim * seq];
        embed_lookup(&mut out, &embed, &tokens, dim, seq);

        // Token 2 -> embed[6..9] = [6, 7, 8]
        // Token 0 -> embed[0..3] = [0, 1, 2]
        // Output [dim=3, seq=2]:
        //   out[0*2+0]=6, out[0*2+1]=0   (d=0: tok2=6, tok0=0)
        //   out[1*2+0]=7, out[1*2+1]=1   (d=1: tok2=7, tok0=1)
        //   out[2*2+0]=8, out[2*2+1]=2   (d=2: tok2=8, tok0=2)
        assert_eq!(out, vec![6.0, 0.0, 7.0, 1.0, 8.0, 2.0]);
    }

    #[test]
    fn test_embed_lookup_u32_large_token() {
        let dim = 2;
        let seq = 1;
        let vocab = 70000; // > u16::MAX
        let mut embed = vec![0.0f32; vocab * dim];
        // Set embedding for token 66000 (> 65535)
        embed[66000 * dim] = 42.0;
        embed[66000 * dim + 1] = 99.0;

        let tokens = [66000u32];
        let mut out = vec![0.0f32; dim * seq];
        embed_lookup(&mut out, &embed, &tokens, dim, seq);
        assert_eq!(out, vec![42.0, 99.0]);
    }

    // -----------------------------------------------------------------------
    // Round 2: cross_entropy_loss + classifier_forward
    // -----------------------------------------------------------------------

    #[test]
    fn test_cross_entropy_uniform_logits() {
        // Uniform logits → softmax = 1/V → loss = ln(V)
        let vocab = 10;
        let seq = 4;
        let logits = vec![0.0f32; vocab * seq];
        let targets: Vec<u16> = vec![0, 1, 2, 3];

        let (loss, _dlogits) = cross_entropy_loss(&logits, &targets, vocab, seq);

        let expected = (vocab as f32).ln();
        assert!(
            (loss - expected).abs() < 0.01,
            "uniform logits: got loss={loss}, expected ~{expected}"
        );
    }

    #[test]
    fn test_cross_entropy_peaked_logits() {
        // Peaked logits at the correct class → loss near 0
        let vocab = 5;
        let seq = 2;
        let mut logits = vec![0.0f32; vocab * seq];
        let targets = vec![1u16, 3u16];

        // Set high logit for correct targets
        for t in 0..seq {
            let tgt = targets[t] as usize;
            logits[tgt * seq + t] = 20.0; // very confident
        }

        let (loss, _) = cross_entropy_loss(&logits, &targets, vocab, seq);
        assert!(
            loss < 0.01,
            "peaked logits: got loss={loss}, expected near 0"
        );
    }

    #[test]
    fn test_cross_entropy_gradient_sums_near_zero() {
        // Gradient for each position should sum to ~0 (softmax - one_hot sums to 0)
        let vocab = 8;
        let seq = 3;
        let logits: Vec<f32> = (0..vocab * seq).map(|i| (i as f32) * 0.1).collect();
        let targets = vec![2u16, 5u16, 0u16];

        let (_, dlogits) = cross_entropy_loss(&logits, &targets, vocab, seq);

        for t in 0..seq {
            let col_sum: f32 = (0..vocab).map(|v| dlogits[v * seq + t]).sum();
            assert!(
                col_sum.abs() < 1e-5,
                "gradient column {t} sums to {col_sum}, expected ~0"
            );
        }
    }

    #[test]
    fn test_classifier_identity_embed() {
        // If embed = identity and dim=vocab, logits should equal x
        let dim = 3;
        let vocab = 3;
        let seq = 2;

        // Identity embed: embed[v,d] = 1 if v==d else 0
        let mut embed = vec![0.0f32; vocab * dim];
        for i in 0..dim {
            embed[i * dim + i] = 1.0;
        }

        // x = [[1,2],[3,4],[5,6]] as [dim=3, seq=2]
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut logits = vec![0.0f32; vocab * seq];

        classifier_forward(&mut logits, &embed, &x, vocab, dim, seq);

        assert_eq!(logits, x, "identity embed: logits should equal x");
    }

    #[test]
    fn test_classifier_matmul_correctness() {
        // Small known matmul: embed[2,3] @ x[3,1]
        let vocab = 2;
        let dim = 3;
        let seq = 1;
        // embed = [[1,2,3],[4,5,6]]
        let embed = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        // x = [[7],[8],[9]] as [dim=3, seq=1]
        let x = vec![7.0, 8.0, 9.0];
        let mut logits = vec![0.0f32; vocab * seq];

        classifier_forward(&mut logits, &embed, &x, vocab, dim, seq);

        // logits[0] = 1*7 + 2*8 + 3*9 = 7+16+27 = 50
        // logits[1] = 4*7 + 5*8 + 6*9 = 28+40+54 = 122
        assert_eq!(logits, vec![50.0, 122.0]);
    }

    // -----------------------------------------------------------------------
    // Round 3: CompiledKernels::compile_forward (requires ANE hardware)
    // -----------------------------------------------------------------------

    #[test]
    fn test_compiled_kernels_compile() {
        let cfg = MilConfig::mha(64, 128, 4, 16);

        let kernels = CompiledKernels::compile_forward(&cfg);
        assert!(
            kernels.is_ok(),
            "compile_forward failed: {:?}",
            kernels.err()
        );

        let k = kernels.unwrap();
        assert!(!k.mask_blob.is_empty(), "mask blob should not be empty");
    }

    // -----------------------------------------------------------------------
    // Round 4: forward — single layer, identity-like weights
    // -----------------------------------------------------------------------

    #[test]
    fn test_forward_single_layer_smoke() {
        use super::super::ane_weights::LayerWeights;

        let dim = 64;
        let hidden = 128;
        let n_heads = 4;
        let seq = 16;
        let vocab = 32;

        let cfg = MilConfig::mha(dim, hidden, n_heads, seq);

        // Compile kernels
        let kernels = match CompiledKernels::compile_forward(&cfg) {
            Ok(k) => k,
            Err(e) => {
                eprintln!("Skipping forward test (ANE unavailable): {e}");
                return;
            }
        };

        // Build a 1-layer model with small random-ish weights
        let make_small = |n: usize| -> Vec<f32> {
            (0..n).map(|i| ((i as f32 * 0.001).sin()) * 0.01).collect()
        };
        let make_ones = |n: usize| -> Vec<f32> { vec![1.0; n] };

        let layer = LayerWeights {
            wq: make_small(dim * dim),
            wk: make_small(dim * dim),
            wv: make_small(dim * dim),
            wo: make_small(dim * dim),
            w1: make_small(hidden * dim),
            w2: make_small(dim * hidden),
            w3: make_small(hidden * dim),
            rms_att: make_ones(dim),
            rms_ffn: make_ones(dim),
            q_norm: None,
            k_norm: None,
        };

        let model = ModelWeights {
            cfg: cfg.clone(),
            layers: vec![layer],
            rms_final: make_ones(dim),
            embed: make_small(vocab * dim),
            vocab_size: vocab,
            lm_head: None,
        };

        let tokens: Vec<u16> = (0..seq).map(|i| (i % vocab) as u16).collect();
        let targets: Vec<u16> = (0..seq).map(|i| ((i + 1) % vocab) as u16).collect();

        let result = forward(&kernels, &model, &tokens, &targets);
        assert!(result.is_ok(), "forward failed: {:?}", result.err());

        let r = result.unwrap();
        assert!(r.loss.is_finite(), "loss should be finite, got {}", r.loss);
        assert!(r.loss > 0.0, "loss should be positive, got {}", r.loss);
        assert_eq!(r.logits.len(), vocab * seq);
        assert_eq!(r.layer_acts.len(), 1);
        assert_eq!(r.layer_acts[0].layer_in.len(), dim * seq);
        assert_eq!(r.layer_acts[0].h1.len(), hidden * seq);
    }

    // -----------------------------------------------------------------------
    // Round 5: forward — multi-layer
    // -----------------------------------------------------------------------

    #[test]
    fn test_forward_two_layers() {
        use super::super::ane_weights::LayerWeights;

        let dim = 64;
        let hidden = 128;
        let n_heads = 4;
        let seq = 16;
        let vocab = 32;

        let cfg = MilConfig::mha(dim, hidden, n_heads, seq);

        let kernels = match CompiledKernels::compile_forward(&cfg) {
            Ok(k) => k,
            Err(e) => {
                eprintln!("Skipping 2-layer forward test (ANE unavailable): {e}");
                return;
            }
        };

        let make_small = |n: usize, seed: usize| -> Vec<f32> {
            (0..n)
                .map(|i| ((i + seed) as f32 * 0.0013).sin() * 0.01)
                .collect()
        };
        let make_ones = |n: usize| -> Vec<f32> { vec![1.0; n] };

        let make_layer = |seed: usize| LayerWeights {
            wq: make_small(dim * dim, seed),
            wk: make_small(dim * dim, seed + 1000),
            wv: make_small(dim * dim, seed + 2000),
            wo: make_small(dim * dim, seed + 3000),
            w1: make_small(hidden * dim, seed + 4000),
            w2: make_small(dim * hidden, seed + 5000),
            w3: make_small(hidden * dim, seed + 6000),
            rms_att: make_ones(dim),
            rms_ffn: make_ones(dim),
            q_norm: None,
            k_norm: None,
        };

        let model = ModelWeights {
            cfg: cfg.clone(),
            layers: vec![make_layer(0), make_layer(7000)],
            rms_final: make_ones(dim),
            embed: make_small(vocab * dim, 14000),
            vocab_size: vocab,
            lm_head: None,
        };

        let tokens: Vec<u16> = (0..seq).map(|i| (i % vocab) as u16).collect();
        let targets: Vec<u16> = (0..seq).map(|i| ((i + 1) % vocab) as u16).collect();

        let result = forward(&kernels, &model, &tokens, &targets).unwrap();

        assert!(result.loss.is_finite(), "2-layer loss not finite: {}", result.loss);
        assert!(result.loss > 0.0, "2-layer loss should be positive");
        assert_eq!(result.layer_acts.len(), 2);

        // With small random weights, loss should be near ln(vocab) = ln(32) ≈ 3.47
        let ln_vocab = (vocab as f32).ln();
        assert!(
            (result.loss - ln_vocab).abs() < 2.0,
            "2-layer loss={} should be near ln({})={:.2}",
            result.loss,
            vocab,
            ln_vocab
        );
    }

    #[test]
    fn test_vec_add_inplace() {
        let mut dst = vec![1.0, 2.0, 3.0];
        let src = vec![10.0, 20.0, 30.0];
        vec_add_inplace(&mut dst, &src);
        assert_eq!(dst, vec![11.0, 22.0, 33.0]);
    }

    // -----------------------------------------------------------------------
    // Round 8.1: RoPE forward+backward roundtrip
    // -----------------------------------------------------------------------

    #[test]
    fn test_qk_rmsnorm_fwd_matches_manual() {
        let n_heads = 2;
        let head_dim = 4;
        let dim = n_heads * head_dim;
        let seq = 2;
        let eps = 1e-5f32;
        let norm_w = vec![1.0, 2.0, 0.5, 1.5];

        let mut x: Vec<f32> = (0..dim * seq)
            .map(|i| (i as f32 + 1.0) * 0.1)
            .collect();
        let pre = qk_rmsnorm_fwd(&mut x, &norm_w, n_heads, head_dim, seq, eps);

        // Verify manually for head 0, position 0
        let mut ss = 0.0f32;
        for d in 0..head_dim {
            ss += pre[(0 * head_dim + d) * seq + 0].powi(2);
        }
        let rrms = 1.0 / (ss / head_dim as f32 + eps).sqrt();
        for d in 0..head_dim {
            let idx = (0 * head_dim + d) * seq + 0;
            let expected = pre[idx] * rrms * norm_w[d];
            assert!(
                (x[idx] - expected).abs() < 1e-5,
                "qk_rmsnorm h=0 t=0 d={d}: got {}, expected {}",
                x[idx], expected
            );
        }
    }

    #[test]
    fn test_qk_rmsnorm_bwd_numerical() {
        let n_heads = 2;
        let head_dim = 4;
        let dim = n_heads * head_dim;
        let seq = 2;
        let eps_norm = 1e-5f32;
        let fd_eps = 1e-4f32;
        let norm_w = vec![1.0, 2.0, 0.5, 1.5];

        let x_orig: Vec<f32> = (0..dim * seq)
            .map(|i| ((i as f32 * 0.7 + 0.3).sin()) * 2.0)
            .collect();
        let dy: Vec<f32> = (0..dim * seq)
            .map(|i| ((i as f32 * 1.3 + 0.7).cos()) * 0.5)
            .collect();

        // Analytical
        let mut x_fwd = x_orig.clone();
        let pre = qk_rmsnorm_fwd(&mut x_fwd, &norm_w, n_heads, head_dim, seq, eps_norm);
        let mut dx = dy.clone();
        let mut dw = vec![0.0f32; head_dim];
        qk_rmsnorm_bwd(&mut dx, &mut dw, &pre, &norm_w, n_heads, head_dim, seq, eps_norm);

        // Numerical dx
        for idx in 0..dim * seq {
            let mut xp = x_orig.clone();
            let mut xm = x_orig.clone();
            xp[idx] += fd_eps;
            xm[idx] -= fd_eps;
            qk_rmsnorm_fwd(&mut xp, &norm_w, n_heads, head_dim, seq, eps_norm);
            qk_rmsnorm_fwd(&mut xm, &norm_w, n_heads, head_dim, seq, eps_norm);
            let mut num = 0.0f32;
            for j in 0..dim * seq {
                num += dy[j] * (xp[j] - xm[j]) / (2.0 * fd_eps);
            }
            assert!(
                (dx[idx] - num).abs() < 0.01,
                "qk_rmsnorm_bwd dx[{idx}]: analytical={:.6}, numerical={:.6}",
                dx[idx], num
            );
        }
    }

    #[test]
    fn test_rope_backward_roundtrip() {
        // Rotate then un-rotate should be identity
        let n_heads = 4;
        let head_dim = 16;
        let dim = n_heads * head_dim;
        let seq = 8;
        let theta = 10000.0;
        let half_hd = head_dim / 2;

        // Random-ish input
        let orig_q: Vec<f32> = (0..dim * seq)
            .map(|i| ((i as f32 * 0.7 + 0.3).sin()) * 2.0)
            .collect();
        let orig_k = orig_q.clone();

        // Forward rotate on CPU (same logic as MIL kernel)
        let mut q_rot = vec![0.0f32; dim * seq];
        let mut k_rot = vec![0.0f32; dim * seq];
        for h in 0..n_heads {
            for t in 0..seq {
                for i in 0..half_hd {
                    let freq = 1.0 / (theta as f64).powf(2.0 * i as f64 / head_dim as f64);
                    let angle = t as f64 * freq;
                    let cos_v = angle.cos() as f32;
                    let sin_v = angle.sin() as f32;
                    let ch1 = (h * head_dim + i) * seq + t;
                    let ch2 = (h * head_dim + half_hd + i) * seq + t;

                    q_rot[ch1] = orig_q[ch1] * cos_v - orig_q[ch2] * sin_v;
                    q_rot[ch2] = orig_q[ch1] * sin_v + orig_q[ch2] * cos_v;
                    k_rot[ch1] = orig_k[ch1] * cos_v - orig_k[ch2] * sin_v;
                    k_rot[ch2] = orig_k[ch1] * sin_v + orig_k[ch2] * cos_v;
                }
            }
        }

        // Backward un-rotate
        rope_backward(&mut q_rot, &mut k_rot, n_heads, head_dim, seq, theta);

        // Should match original
        for i in 0..dim * seq {
            assert!(
                (q_rot[i] - orig_q[i]).abs() < 1e-5,
                "rope roundtrip q[{i}]: got {}, expected {}",
                q_rot[i], orig_q[i]
            );
            assert!(
                (k_rot[i] - orig_k[i]).abs() < 1e-5,
                "rope roundtrip k[{i}]: got {}, expected {}",
                k_rot[i], orig_k[i]
            );
        }
    }

    // -----------------------------------------------------------------------
    // E2E: Qwen3-1.7B forward smoke test (requires ANE + model weights)
    // -----------------------------------------------------------------------

    #[test]
    fn test_e2e_qwen3_forward_smoke() {
        use super::super::ane_weights::ModelWeights;

        let model_dir = std::path::Path::new(&std::env::var("HOME").unwrap())
            .join(".cache/lm-studio/models/lmstudio-community/Qwen3-1.7B-MLX-8bit");
        if !model_dir.exists() {
            eprintln!("Qwen3-1.7B not found, skipping E2E forward test");
            return;
        }

        // Qwen3-1.7B: dim=2048, hidden=6144, 16 heads, 8 KV heads, head_dim=128
        let seq = 4; // small seq to keep classifier (vocab=151936) tractable on CPU
        let cfg = MilConfig {
            dim: 2048,
            hidden_dim: 6144,
            n_heads: 16,
            seq_len: seq,
            n_kv_heads: 8,
            rope_theta: 1_000_000.0,
            rms_eps: 1e-6,
            has_lm_head: false,
        };

        eprintln!("loading Qwen3-1.7B weights...");
        let t0 = std::time::Instant::now();
        let model = ModelWeights::from_mlx_safetensors(&model_dir, &cfg)
            .expect("from_mlx_safetensors failed");
        eprintln!("loaded in {}ms", t0.elapsed().as_millis());

        let tokens: Vec<u32> = (100..100 + seq as u32).collect();
        let targets: Vec<u32> = (101..101 + seq as u32).collect();

        // CPU-only forward (ANE dynamic packing exceeds IOSurface limits at dim=2048)
        eprintln!("running CPU forward pass (28 layers, vocab={})", model.vocab_size);
        let t0 = std::time::Instant::now();
        let result = forward_cpu(&model, None, &tokens, &targets);
        let fwd_ms = t0.elapsed().as_millis();

        let r = &result.base;
        assert!(r.loss.is_finite(), "loss not finite: {}", r.loss);
        assert!(r.loss > 0.0, "loss should be positive: {}", r.loss);
        assert_eq!(r.layer_acts.len(), 28);
        assert_eq!(r.logits.len(), model.vocab_size * seq);

        // With real trained weights, loss should be well below ln(151936) ~ 11.9
        eprintln!("E2E forward: loss={:.4}, time={}ms", r.loss, fwd_ms);
    }

    // -----------------------------------------------------------------------
    // E2E: Qwen3-1.7B LoRA training test (requires ANE + model weights)
    // -----------------------------------------------------------------------

    #[test]
    fn test_e2e_qwen3_lora_training() {
        use super::super::ane_lora::{LoraConfig, LoraModel, LoraModelAdam};
        use super::super::ane_weights::ModelWeights;

        let model_dir = std::path::Path::new(&std::env::var("HOME").unwrap())
            .join(".cache/lm-studio/models/lmstudio-community/Qwen3-1.7B-MLX-8bit");
        if !model_dir.exists() {
            eprintln!("Qwen3-1.7B not found, skipping E2E LoRA test");
            return;
        }

        let seq = 4;
        let cfg = MilConfig {
            dim: 2048,
            hidden_dim: 6144,
            n_heads: 16,
            seq_len: seq,
            n_kv_heads: 8,
            rope_theta: 1_000_000.0,
            rms_eps: 1e-6,
            has_lm_head: false,
        };

        eprintln!("loading Qwen3-1.7B...");
        let model = ModelWeights::from_mlx_safetensors(&model_dir, &cfg)
            .expect("from_mlx_safetensors failed");

        let lora_cfg = LoraConfig::default(); // rank=32, targets: wq, wv, wo, w2
        let mut lora = LoraModel::new(lora_cfg, model.layers.len(), cfg.dim, cfg.hidden_dim);
        let mut adam = LoraModelAdam::zeros(&lora);

        // Simple training data: predict next token
        let tokens: Vec<u32> = (100..100 + seq as u32).collect();
        let targets: Vec<u32> = (101..101 + seq as u32).collect();

        let n_steps = 5;
        let lr = 5e-4;
        let mut losses = Vec::with_capacity(n_steps);

        eprintln!("training {} steps with LoRA rank=32 (CPU)...", n_steps);
        for step in 0..n_steps {
            let t0 = std::time::Instant::now();

            // CPU forward with LoRA
            let fwd = forward_cpu(&model, Some(&lora), &tokens, &targets);
            let loss = fwd.base.loss;
            losses.push(loss);
            assert!(loss.is_finite(), "step {step}: loss not finite: {loss}");

            // CPU backward for LoRA gradients
            let bwd = super::super::ane_backward::backward_lora_cpu(
                &model, &fwd, &lora, &tokens,
            );

            // Adam update on LoRA params only
            super::super::ane_lora::lora_adam_update(
                &mut lora, &bwd.lora_grads, &mut adam,
                step + 1, lr, 0.9, 0.999, 1e-8, 0.01,
            );

            let step_ms = t0.elapsed().as_millis();
            eprintln!("  step {step}: loss={loss:.4}, time={step_ms}ms");
        }

        // Verify loss decreased
        let first = losses[0];
        let last = losses[n_steps - 1];
        eprintln!("loss trajectory: {:.4} -> {:.4} (delta={:.4})", first, last, last - first);
        assert!(
            last < first,
            "loss should decrease over training: first={first:.4}, last={last:.4}"
        );
    }
}
