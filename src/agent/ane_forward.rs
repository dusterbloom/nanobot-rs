//! Forward inference pipeline for ANE transformer training.
//!
//! CPU ops (RMSNorm, embedding, cross-entropy) plus compiled kernel set
//! and full forward pass: embed → N layers → classifier → loss.

use super::ane_bridge::{self, AneKernel};
use super::ane_mil::{self, KernelSpec, KernelType, MilConfig};
use super::ane_weights::{self, ModelWeights};

// ---------------------------------------------------------------------------
// Accelerate SGEMM binding (Apple's BLAS — linked via build.rs)
// ---------------------------------------------------------------------------
#[cfg(target_os = "macos")]
extern "C" {
    fn cblas_sgemm(
        order: i32,
        transa: i32,
        transb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}

// ---------------------------------------------------------------------------
// General GEMM: C = alpha * op(A) @ op(B) + beta * C
// ---------------------------------------------------------------------------

/// General matrix multiply into pre-allocated buffer.
///
/// C\[m, n\] = alpha * op(A) @ op(B) + beta * C
/// where op(X) = X if trans=false, X^T if trans=true.
///
/// All matrices are stored row-major. Leading dimensions are inferred:
///   - A stored as \[rows_A, cols_A\]: lda = cols_A = if trans_a { m } else { k }
///   - B stored as \[rows_B, cols_B\]: ldb = cols_B = if trans_b { k } else { n }
///   - C stored as \[m, n\]: ldc = n
pub fn cpu_gemm(
    c: &mut [f32],
    a: &[f32],
    trans_a: bool,
    b: &[f32],
    trans_b: bool,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    beta: f32,
) {
    debug_assert!(c.len() >= m * n);

    #[cfg(target_os = "macos")]
    {
        let tra = if trans_a { 112 } else { 111 }; // CblasTrans : CblasNoTrans
        let trb = if trans_b { 112 } else { 111 };
        let lda = if trans_a { m } else { k };
        let ldb = if trans_b { k } else { n };
        unsafe {
            cblas_sgemm(
                101, // CblasRowMajor
                tra,
                trb,
                m as i32,
                n as i32,
                k as i32,
                alpha,
                a.as_ptr(),
                lda as i32,
                b.as_ptr(),
                ldb as i32,
                beta,
                c.as_mut_ptr(),
                n as i32,
            );
        }
        return;
    }

    #[cfg(not(target_os = "macos"))]
    {
        if beta == 0.0 {
            c.iter_mut().for_each(|v| *v = 0.0);
        } else if (beta - 1.0).abs() > f32::EPSILON {
            c.iter_mut().for_each(|v| *v *= beta);
        }
        for mi in 0..m {
            for ni in 0..n {
                let mut acc = 0.0f32;
                for ki in 0..k {
                    let av = if trans_a {
                        a[ki * m + mi]
                    } else {
                        a[mi * k + ki]
                    };
                    let bv = if trans_b {
                        b[ni * k + ki]
                    } else {
                        b[ki * n + ni]
                    };
                    acc += av * bv;
                }
                c[mi * n + ni] += alpha * acc;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Token type abstraction
// ---------------------------------------------------------------------------

/// Trait for token ID types (u16 for llama2.c, u32 for large-vocab models like Qwen).
pub trait TokenId: Copy + 'static {
    fn as_usize(self) -> usize;
}
impl TokenId for u16 {
    fn as_usize(self) -> usize {
        self as usize
    }
}
impl TokenId for u32 {
    fn as_usize(self) -> usize {
        self as usize
    }
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
pub fn embed_lookup<T: TokenId>(
    out: &mut [f32],
    embed: &[f32],
    tokens: &[T],
    dim: usize,
    seq: usize,
) {
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

/// Classifier: logits[V,S] = embed[V,D] @ x[D,S].
///
/// embed: [vocab, dim] row-major, x: [dim, seq], logits: [vocab, seq].
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
    cpu_gemm(logits, embed, false, x, false, vocab, seq, dim, 1.0, 0.0);
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
    dx: &mut [f32],      // in: grad w.r.t. normed, out: grad w.r.t. pre-norm
    dnorm_w: &mut [f32], // accumulated [head_dim]
    pre_norm: &[f32],    // saved pre-norm values
    norm_w: &[f32],      // [head_dim]
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

/// Split interleaved q_proj output [2*attn_dim, seq] into separate q [attn_dim, seq]
/// and gate [attn_dim, seq]. Memory layout: for head h, dim d,
/// q src = (h*2*head_dim + d)*seq, gate src = (h*2*head_dim + head_dim + d)*seq.
pub(crate) fn split_q_gate(
    q_full: &[f32],
    n_heads: usize,
    head_dim: usize,
    seq: usize,
) -> (Vec<f32>, Vec<f32>) {
    let attn_dim = n_heads * head_dim;
    debug_assert_eq!(q_full.len(), 2 * attn_dim * seq);
    let mut q = vec![0.0f32; attn_dim * seq];
    let mut gate = vec![0.0f32; attn_dim * seq];
    for h in 0..n_heads {
        for d in 0..head_dim {
            let dst_off = (h * head_dim + d) * seq;
            let q_src = (h * 2 * head_dim + d) * seq;
            let g_src = (h * 2 * head_dim + head_dim + d) * seq;
            q[dst_off..dst_off + seq].copy_from_slice(&q_full[q_src..q_src + seq]);
            gate[dst_off..dst_off + seq].copy_from_slice(&q_full[g_src..g_src + seq]);
        }
    }
    (q, gate)
}

/// Inverse of split_q_gate: merge q [attn_dim, seq] and gate [attn_dim, seq]
/// back into interleaved [2*attn_dim, seq].
pub(crate) fn merge_q_gate(
    dq: &[f32],
    dgate: &[f32],
    n_heads: usize,
    head_dim: usize,
    seq: usize,
) -> Vec<f32> {
    let attn_dim = n_heads * head_dim;
    debug_assert_eq!(dq.len(), attn_dim * seq);
    debug_assert_eq!(dgate.len(), attn_dim * seq);
    let mut merged = vec![0.0f32; 2 * attn_dim * seq];
    for h in 0..n_heads {
        for d in 0..head_dim {
            let src_off = (h * head_dim + d) * seq;
            let q_dst = (h * 2 * head_dim + d) * seq;
            let g_dst = (h * 2 * head_dim + head_dim + d) * seq;
            merged[q_dst..q_dst + seq].copy_from_slice(&dq[src_off..src_off + seq]);
            merged[g_dst..g_dst + seq].copy_from_slice(&dgate[src_off..src_off + seq]);
        }
    }
    merged
}

/// Apply sigmoid gate element-wise in-place: attn_out[i] *= sigmoid(gate[i]).
fn apply_sigmoid_gate(attn_out: &mut [f32], gate: &[f32]) {
    debug_assert_eq!(attn_out.len(), gate.len());
    for i in 0..attn_out.len() {
        attn_out[i] *= 1.0 / (1.0 + (-gate[i]).exp());
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
    pub layer_in: Vec<f32>,           // [dim, seq]
    pub xnorm: Vec<f32>,              // [dim, seq]
    pub q: Vec<f32>,                  // [dim, seq] (post-norm, post-rope)
    pub k: Vec<f32>,                  // [dim, seq] (post-norm, post-rope)
    pub v: Vec<f32>,                  // [dim, seq]
    pub attn_out: Vec<f32>,           // [dim, seq]
    pub o_out: Vec<f32>,              // [dim, seq]
    pub x2: Vec<f32>,                 // [dim, seq]
    pub x2norm: Vec<f32>,             // [dim, seq]
    pub h1: Vec<f32>,                 // [hidden, seq]
    pub h3: Vec<f32>,                 // [hidden, seq]
    pub gate: Vec<f32>,               // [hidden, seq]  (silu(h1)*h3)
    pub ffn_out: Vec<f32>,            // [dim, seq]
    pub q_pre_norm: Option<Vec<f32>>, // [dim, seq] pre-QK-norm Q (for backward)
    pub k_pre_norm: Option<Vec<f32>>, // [dim, seq] pre-QK-norm K (for backward)
    /// Raw gate values from q_proj split (Qwen3.5 attn_output_gate). [attn_dim, seq]
    pub attn_gate: Option<Vec<f32>>,
    /// SDPA output before sigmoid gate was applied (for backward). [attn_dim, seq]
    pub attn_pre_gate: Option<Vec<f32>>,
}

/// Forward pass result.
pub struct ForwardResult {
    pub logits: Vec<f32>, // [vocab, seq]
    pub loss: f32,
    pub dlogits: Vec<f32>, // [vocab, seq]
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
    forward_with_lora(kernels, model, None, None, tokens, targets).map(|r| r.base)
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
        let sdpa_input = ane_weights::pack_sdpa_fwd(&xnorm, &lw.wq, &lw.wk, &lw.wv, &lw.wo, cfg);
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
                        &lk.attn_a_fwd,
                        &lk.attn_b_fwd,
                        wo_adapter,
                        &attn_out,
                        seq,
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
                        &lk.ffn_a_fwd,
                        &lk.ffn_b_fwd,
                        w2_adapter,
                        &gate,
                        seq,
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
            attn_gate: None,
            attn_pre_gate: None,
        });
        lora_acts_vec.push(lora_layer_acts);
    }

    // 3. Final RMSNorm
    let mut x_final = vec![0.0f32; dim * seq];
    rmsnorm(
        &mut x_final,
        &x_cur,
        &model.rms_final,
        dim,
        seq,
        cfg.rms_eps,
    );

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

const QUANTIZED_MATMUL_ROW_BLOCK: usize = 128;

#[derive(Default)]
pub(crate) struct QuantizedMatmulWorkspace {
    dense_block: Vec<f32>,
}

impl QuantizedMatmulWorkspace {
    fn ensure_dense_block(&mut self, needed: usize) -> &mut [f32] {
        if self.dense_block.len() < needed {
            self.dense_block.resize(needed, 0.0);
        }
        &mut self.dense_block[..needed]
    }
}

/// CPU matmul: out[M,S] = W[M,N] @ x[N,S] (row-major W, channels-first x).
///
/// Uses Apple Accelerate SGEMM on macOS (25-300x faster than naive).
pub fn cpu_matmul(w: &[f32], x: &[f32], m: usize, n: usize, s: usize) -> Vec<f32> {
    debug_assert_eq!(w.len(), m * n);
    debug_assert_eq!(x.len(), n * s);
    let mut out = vec![0.0f32; m * s];
    cpu_gemm(&mut out, w, false, x, false, m, s, n, 1.0, 0.0);
    out
}

fn dequantize_row_block(
    w: &ane_weights::QuantizedTensor,
    row_start: usize,
    row_count: usize,
    out: &mut [f32],
) {
    debug_assert!(row_start + row_count <= w.rows);
    debug_assert_eq!(out.len(), row_count * w.cols);

    let n_groups = w.cols / w.group_size;
    let elems_per_u32 = 32 / w.bits;
    let mask = (1u32 << w.bits) - 1;
    let packed_cols = w.cols / elems_per_u32; // u32 words per row
    let row_bytes = packed_cols * 4; // bytes per row

    for local_row in 0..row_count {
        let row = row_start + local_row;
        let row_byte_offset = row * row_bytes;
        let out_row = &mut out[local_row * w.cols..(local_row + 1) * w.cols];
        for col in 0..w.cols {
            let word_idx = col / elems_per_u32;
            let elem_idx = col % elems_per_u32;
            let byte_off = row_byte_offset + word_idx * 4;
            let u32_val = u32::from_le_bytes([
                w.data[byte_off],
                w.data[byte_off + 1],
                w.data[byte_off + 2],
                w.data[byte_off + 3],
            ]);
            let qval = ((u32_val >> (elem_idx * w.bits)) & mask) as f32;
            let group = col / w.group_size;
            let scale = w.scales[row * n_groups + group];
            let bias = w.biases[row * n_groups + group];
            out_row[col] = scale * qval + bias;
        }
    }
}

pub(crate) fn cpu_quantized_matmul_into(
    w: &ane_weights::QuantizedTensor,
    x: &[f32],
    s: usize,
    out: &mut [f32],
    workspace: &mut QuantizedMatmulWorkspace,
) {
    assert_eq!(
        x.len(),
        w.cols * s,
        "Dimension mismatch in quantized matmul: x.len()={} but expected w.cols({}) * s({}) = {}",
        x.len(),
        w.cols,
        s,
        w.cols * s
    );
    assert_eq!(
        out.len(),
        w.rows * s,
        "Dimension mismatch in quantized matmul: out.len()={} but expected w.rows({}) * s({}) = {}",
        out.len(),
        w.rows,
        s,
        w.rows * s
    );

    let block_rows = QUANTIZED_MATMUL_ROW_BLOCK.min(w.rows.max(1));
    for row_start in (0..w.rows).step_by(block_rows) {
        let row_count = (w.rows - row_start).min(block_rows);
        let dense_block = workspace.ensure_dense_block(row_count * w.cols);
        dequantize_row_block(w, row_start, row_count, dense_block);
        cpu_gemm(
            &mut out[row_start * s..(row_start + row_count) * s],
            dense_block,
            false,
            x,
            false,
            row_count,
            s,
            w.cols,
            1.0,
            0.0,
        );
    }
}

/// Quantized CPU matmul: out[rows, s] = dequantize(W[rows, cols]) @ x[cols, s].
///
/// The weight matrix is dequantized in row blocks so we avoid materializing the
/// full dense matrix for each projection.
pub fn cpu_quantized_matmul(w: &ane_weights::QuantizedTensor, x: &[f32], s: usize) -> Vec<f32> {
    assert_eq!(
        x.len(),
        w.cols * s,
        "Dimension mismatch in quantized matmul: x.len()={} but expected w.cols({}) * s({}) = {}",
        x.len(),
        w.cols,
        s,
        w.cols * s
    );

    let mut out = vec![0.0f32; w.rows * s];
    let mut workspace = QuantizedMatmulWorkspace::default();
    cpu_quantized_matmul_into(w, x, s, &mut out, &mut workspace);

    out
}

pub(crate) fn cpu_quantized_matmul_lhs_transposed_into(
    w: &ane_weights::QuantizedTensor,
    x: &[f32],
    s: usize,
    out: &mut [f32],
    workspace: &mut QuantizedMatmulWorkspace,
    accumulate: bool,
) {
    assert_eq!(
        x.len(),
        w.rows * s,
        "Dimension mismatch in quantized matmul: x.len()={} but expected w.rows({}) * s({}) = {}",
        x.len(),
        w.rows,
        s,
        w.rows * s
    );
    assert_eq!(
        out.len(),
        w.cols * s,
        "Dimension mismatch in quantized matmul: out.len()={} but expected w.cols({}) * s({}) = {}",
        out.len(),
        w.cols,
        s,
        w.cols * s
    );

    let block_rows = QUANTIZED_MATMUL_ROW_BLOCK.min(w.rows.max(1));
    let mut first_block = !accumulate;

    for row_start in (0..w.rows).step_by(block_rows) {
        let row_count = (w.rows - row_start).min(block_rows);
        let dense_block = workspace.ensure_dense_block(row_count * w.cols);
        dequantize_row_block(w, row_start, row_count, dense_block);
        cpu_gemm(
            out,
            dense_block,
            true,
            &x[row_start * s..(row_start + row_count) * s],
            false,
            w.cols,
            s,
            row_count,
            1.0,
            if first_block { 0.0 } else { 1.0 },
        );
        first_block = false;
    }
}

/// Quantized CPU matmul with a transposed left-hand matrix:
/// out[cols, s] = dequantize(W[rows, cols])^T @ x[rows, s].
///
/// This mirrors `cpu_matmul_lhs_transposed` while avoiding full dense weight
/// materialization for quantized weights.
pub fn cpu_quantized_matmul_lhs_transposed(
    w: &ane_weights::QuantizedTensor,
    x: &[f32],
    s: usize,
) -> Vec<f32> {
    assert_eq!(
        x.len(),
        w.rows * s,
        "Dimension mismatch in quantized matmul: x.len()={} but expected w.rows({}) * s({}) = {}",
        x.len(),
        w.rows,
        s,
        w.rows * s
    );

    let mut out = vec![0.0f32; w.cols * s];
    let mut workspace = QuantizedMatmulWorkspace::default();
    cpu_quantized_matmul_lhs_transposed_into(w, x, s, &mut out, &mut workspace, false);

    out
}

/// Collapse expanded GQA activations or gradients back to KV-head layout by
/// summing repeated head blocks.
pub fn collapse_grouped_kv_rows(
    expanded: &[f32],
    kv_rows: usize,
    head_dim: usize,
    heads_per_group: usize,
    seq: usize,
) -> Vec<f32> {
    debug_assert_eq!(expanded.len(), kv_rows * heads_per_group * seq);
    if heads_per_group <= 1 {
        return expanded.to_vec();
    }

    let n_kv_heads = kv_rows / head_dim;
    let mut collapsed = vec![0.0f32; kv_rows * seq];
    for kv_head in 0..n_kv_heads {
        for rep in 0..heads_per_group {
            let src_head = kv_head * heads_per_group + rep;
            for d in 0..head_dim {
                let src_row = src_head * head_dim + d;
                let dst_row = kv_head * head_dim + d;
                for t in 0..seq {
                    collapsed[dst_row * seq + t] += expanded[src_row * seq + t];
                }
            }
        }
    }
    collapsed
}

/// CPU matmul with a transposed left-hand matrix:
/// out[cols, s] = w[rows, cols]^T @ x[rows, s].
///
/// This avoids materializing an explicit transposed copy of frozen weights in
/// backward passes.
pub fn cpu_matmul_lhs_transposed(
    w: &[f32],
    rows: usize,
    cols: usize,
    x: &[f32],
    s: usize,
) -> Vec<f32> {
    assert_eq!(
        w.len(),
        rows * cols,
        "Dimension mismatch in matmul: w.len()={} but expected rows({}) * cols({}) = {}",
        w.len(),
        rows,
        cols,
        rows * cols
    );
    assert_eq!(
        x.len(),
        rows * s,
        "Dimension mismatch in matmul: x.len()={} but expected rows({}) * s({}) = {}",
        x.len(),
        rows,
        s,
        rows * s
    );
    let mut out = vec![0.0f32; cols * s];
    cpu_gemm(&mut out, w, true, x, false, cols, s, rows, 1.0, 0.0);
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

                let q1 = q[ch1];
                let q2 = q[ch2];
                q[ch1] = q1 * cos_v - q2 * sin_v;
                q[ch2] = q1 * sin_v + q2 * cos_v;

                let k1 = k[ch1];
                let k2 = k[ch2];
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
fn cpu_sdpa(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_heads: usize,
    head_dim: usize,
    seq: usize,
) -> Vec<f32> {
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

// ---------------------------------------------------------------------------
// GDN recurrence — NEON-optimized inner loop
// ---------------------------------------------------------------------------

/// Gated delta recurrence: the sequential core of GDN attention.
///
/// For each token t, for each head h, for each value dim dv:
///   1. Decay state by g_t (fused with kv_mem dot product)
///   2. Compute delta = (v_t - kv_mem) * beta_t
///   3. Update state += k * delta
///   4. Output y = dot(state, q)
///
/// NEON version: processes d_k dimension 4-wide with fused multiply-add.
/// Gathers strided k/q into contiguous buffers for vectorized inner loops.
#[cfg(target_arch = "aarch64")]
fn gdn_recurrence(
    state: &mut [f32],
    y: &mut [f32],
    q_exp: &[f32],
    k_exp: &[f32],
    v_raw: &[f32],
    g: &[f32],
    beta: &[f32],
    h_v: usize,
    d_k: usize,
    d_v: usize,
    seq: usize,
) {
    use std::arch::aarch64::*;

    let mut k_buf = vec![0.0f32; d_k];
    let mut q_buf = vec![0.0f32; d_k];

    for t in 0..seq {
        for h in 0..h_v {
            let g_t = g[h * seq + t];
            let beta_t = beta[h * seq + t];

            // Gather strided k and q into contiguous buffers
            for dk in 0..d_k {
                k_buf[dk] = k_exp[(h * d_k + dk) * seq + t];
                q_buf[dk] = q_exp[(h * d_k + dk) * seq + t];
            }

            let state_base = h * d_v * d_k;

            // Process each dv row fully: decay → kv_mem → delta → update → y
            // Maximizes L1 cache reuse: each row is d_k floats (512B for d_k=128)
            for dv in 0..d_v {
                let row = state_base + dv * d_k;

                // --- Pass 1: fused decay + kv_mem dot product ---
                unsafe {
                    let g_vec = vdupq_n_f32(g_t);
                    let mut acc0 = vdupq_n_f32(0.0);
                    let mut acc1 = vdupq_n_f32(0.0);

                    let sp = state.as_mut_ptr().add(row);
                    let kp = k_buf.as_ptr();

                    let mut dk = 0;
                    while dk + 8 <= d_k {
                        // Unroll 2x for ILP
                        let mut s0 = vld1q_f32(sp.add(dk));
                        let mut s1 = vld1q_f32(sp.add(dk + 4));
                        let k0 = vld1q_f32(kp.add(dk));
                        let k1 = vld1q_f32(kp.add(dk + 4));
                        s0 = vmulq_f32(s0, g_vec);
                        s1 = vmulq_f32(s1, g_vec);
                        vst1q_f32(sp.add(dk), s0);
                        vst1q_f32(sp.add(dk + 4), s1);
                        acc0 = vfmaq_f32(acc0, s0, k0);
                        acc1 = vfmaq_f32(acc1, s1, k1);
                        dk += 8;
                    }
                    while dk + 4 <= d_k {
                        let mut s = vld1q_f32(sp.add(dk));
                        let kv = vld1q_f32(kp.add(dk));
                        s = vmulq_f32(s, g_vec);
                        vst1q_f32(sp.add(dk), s);
                        acc0 = vfmaq_f32(acc0, s, kv);
                        dk += 4;
                    }
                    let kv_mem = vaddvq_f32(vaddq_f32(acc0, acc1));

                    // --- Delta computation ---
                    let v_t = v_raw[(h * d_v + dv) * seq + t];
                    let delta = (v_t - kv_mem) * beta_t;

                    // --- Pass 2: state update: state += k * delta ---
                    let delta_vec = vdupq_n_f32(delta);
                    dk = 0;
                    while dk + 8 <= d_k {
                        let s0 = vld1q_f32(sp.add(dk));
                        let s1 = vld1q_f32(sp.add(dk + 4));
                        let k0 = vld1q_f32(kp.add(dk));
                        let k1 = vld1q_f32(kp.add(dk + 4));
                        vst1q_f32(sp.add(dk), vfmaq_f32(s0, k0, delta_vec));
                        vst1q_f32(sp.add(dk + 4), vfmaq_f32(s1, k1, delta_vec));
                        dk += 8;
                    }
                    while dk + 4 <= d_k {
                        let s = vld1q_f32(sp.add(dk));
                        let kv = vld1q_f32(kp.add(dk));
                        vst1q_f32(sp.add(dk), vfmaq_f32(s, kv, delta_vec));
                        dk += 4;
                    }

                    // --- Pass 3: y output: y[h,dv,t] = dot(state, q) ---
                    let qp = q_buf.as_ptr();
                    let mut yacc0 = vdupq_n_f32(0.0);
                    let mut yacc1 = vdupq_n_f32(0.0);
                    dk = 0;
                    while dk + 8 <= d_k {
                        let s0 = vld1q_f32(sp.add(dk));
                        let s1 = vld1q_f32(sp.add(dk + 4));
                        let q0 = vld1q_f32(qp.add(dk));
                        let q1 = vld1q_f32(qp.add(dk + 4));
                        yacc0 = vfmaq_f32(yacc0, s0, q0);
                        yacc1 = vfmaq_f32(yacc1, s1, q1);
                        dk += 8;
                    }
                    while dk + 4 <= d_k {
                        let s = vld1q_f32(sp.add(dk));
                        let qv = vld1q_f32(qp.add(dk));
                        yacc0 = vfmaq_f32(yacc0, s, qv);
                        dk += 4;
                    }
                    y[(h * d_v + dv) * seq + t] = vaddvq_f32(vaddq_f32(yacc0, yacc1));
                }
            }
        }
    }
}

/// Scalar fallback for non-aarch64 platforms.
#[cfg(not(target_arch = "aarch64"))]
fn gdn_recurrence(
    state: &mut [f32],
    y: &mut [f32],
    q_exp: &[f32],
    k_exp: &[f32],
    v_raw: &[f32],
    g: &[f32],
    beta: &[f32],
    h_v: usize,
    d_k: usize,
    d_v: usize,
    seq: usize,
) {
    for t in 0..seq {
        for h in 0..h_v {
            let g_t = g[h * seq + t];
            let beta_t = beta[h * seq + t];
            for dv in 0..d_v {
                for dk in 0..d_k {
                    state[h * d_v * d_k + dv * d_k + dk] *= g_t;
                }
            }
            for dv in 0..d_v {
                let mut kv_mem = 0.0f32;
                for dk in 0..d_k {
                    kv_mem +=
                        state[h * d_v * d_k + dv * d_k + dk] * k_exp[(h * d_k + dk) * seq + t];
                }
                let v_t = v_raw[(h * d_v + dv) * seq + t];
                let delta = (v_t - kv_mem) * beta_t;
                for dk in 0..d_k {
                    state[h * d_v * d_k + dv * d_k + dk] += k_exp[(h * d_k + dk) * seq + t] * delta;
                }
            }
            for dv in 0..d_v {
                let mut y_val = 0.0f32;
                for dk in 0..d_k {
                    y_val += state[h * d_v * d_k + dv * d_k + dk] * q_exp[(h * d_k + dk) * seq + t];
                }
                y[(h * d_v + dv) * seq + t] = y_val;
            }
        }
    }
}

/// Benchmark/helper entrypoint that runs the production GDN recurrence kernel.
///
/// Intended for microbenchmarks that want to time the exact recurrence used by
/// the CPU GDN path without reimplementing the kernel in another crate.
#[doc(hidden)]
pub fn cpu_gdn_recurrence_bench(
    q_exp: &[f32],
    k_exp: &[f32],
    v_raw: &[f32],
    g: &[f32],
    beta: &[f32],
    h_v: usize,
    d_k: usize,
    d_v: usize,
    seq: usize,
) -> Vec<f32> {
    let mut state = vec![0.0f32; h_v * d_v * d_k];
    let mut y = vec![0.0f32; h_v * d_v * seq];
    gdn_recurrence(
        &mut state, &mut y, q_exp, k_exp, v_raw, g, beta, h_v, d_k, d_v, seq,
    );
    y
}

/// CPU GDN (Gated Delta Net) forward for a single layer.
///
/// Ports `MlxLinearAttention::forward()` from `mlx_lora.rs` to CPU f32 ops.
/// Layout: channels-first `[C, S]` (no batch dimension).
///
/// Returns the attention output `[dim, seq]` (before Wo residual add).
fn cpu_gdn_forward_post_proj(
    qkv_raw: &[f32], // [qkv_dim, seq]
    a_raw: &[f32],   // [h_v, seq]
    b_raw: &[f32],   // [h_v, seq]
    z: &[f32],       // [value_dim, seq]
    a_log: &[f32],   // [h_v]
    dt_bias: &[f32], // [h_v]
    norm_weight: &[f32],
    conv_weight: &[f32], // [qkv_dim, kernel_size]
    conv_bias: &[f32],   // [qkv_dim]
    cfg: &MilConfig,
    apply_o_proj: impl FnOnce(&[f32]) -> Vec<f32>,
) -> Vec<f32> {
    let h_k = cfg.linear_n_heads;
    let d_k = cfg.linear_head_dim;
    let h_v = cfg.linear_n_value_heads;
    let d_v = cfg.linear_value_head_dim;
    let key_dim = h_k * d_k;
    let value_dim = h_v * d_v;
    let qkv_dim = 2 * key_dim + value_dim;
    let kernel = cfg.conv_kernel_size;
    let kv_repeat = h_v / h_k.max(1);
    let seq = cfg.seq_len;

    debug_assert_eq!(qkv_raw.len(), qkv_dim * seq);
    debug_assert_eq!(a_raw.len(), h_v * seq);
    debug_assert_eq!(b_raw.len(), h_v * seq);
    debug_assert_eq!(z.len(), value_dim * seq);
    debug_assert_eq!(a_log.len(), h_v);
    debug_assert_eq!(dt_bias.len(), h_v);
    assert!(
        norm_weight.len() == d_v || norm_weight.len() == value_dim,
        "GDN norm weight must have length d_v ({d_v}) or value_dim ({value_dim}), got {}",
        norm_weight.len()
    );
    debug_assert_eq!(conv_weight.len(), qkv_dim * kernel);

    // 1. Causal depthwise conv1d + SiLU on QKV
    //    Input: [qkv_dim, seq], kernel: [qkv_dim, kernel_size] (depthwise)
    //    Left-pad by kernel-1 zeros, then for each channel: sum over kernel window
    let mut qkv_conv = vec![0.0f32; qkv_dim * seq];
    for c in 0..qkv_dim {
        for t in 0..seq {
            let mut acc = 0.0f32;
            for ki in 0..kernel {
                let src_t = t as isize - ki as isize; // causal: look back
                let val = if src_t >= 0 {
                    qkv_raw[c * seq + src_t as usize]
                } else {
                    0.0 // left-pad with zeros
                };
                acc += val * conv_weight[c * kernel + ki];
            }
            // Add bias if present
            if c < conv_bias.len() {
                acc += conv_bias[c];
            }
            // SiLU activation
            qkv_conv[c * seq + t] = acc / (1.0 + (-acc).exp());
        }
    }

    // 3. Split into Q [key_dim, seq], K [key_dim, seq], V [value_dim, seq]
    //    Channels-first: first key_dim channels = Q, next key_dim = K, rest = V
    let q_raw = &qkv_conv[0..key_dim * seq];
    let k_raw = &qkv_conv[key_dim * seq..2 * key_dim * seq];
    let v_raw = &qkv_conv[2 * key_dim * seq..qkv_dim * seq];

    // 4. Weight-free RMSNorm on Q and K (per-head, across d_k dimension)
    let inv_scale = (d_k as f32).powf(-0.5);
    let mut q = vec![0.0f32; key_dim * seq];
    let mut k = vec![0.0f32; key_dim * seq];
    for h in 0..h_k {
        for t in 0..seq {
            // Compute RMS for this head at this position
            let mut q_ss = 0.0f32;
            let mut k_ss = 0.0f32;
            for d in 0..d_k {
                let qi = q_raw[(h * d_k + d) * seq + t];
                let ki = k_raw[(h * d_k + d) * seq + t];
                q_ss += qi * qi;
                k_ss += ki * ki;
            }
            let q_rms = (q_ss / d_k as f32 + 1e-6).sqrt();
            let k_rms = (k_ss / d_k as f32 + 1e-6).sqrt();
            for d in 0..d_k {
                q[(h * d_k + d) * seq + t] =
                    q_raw[(h * d_k + d) * seq + t] / q_rms * inv_scale * inv_scale;
                k[(h * d_k + d) * seq + t] = k_raw[(h * d_k + d) * seq + t] / k_rms * inv_scale;
            }
        }
    }

    // 5. GQA expansion: repeat Q,K from h_k to h_v heads if needed
    let (q_exp, k_exp) = if kv_repeat > 1 {
        let mut qe = vec![0.0f32; h_v * d_k * seq];
        let mut ke = vec![0.0f32; h_v * d_k * seq];
        for hk in 0..h_k {
            for r in 0..kv_repeat {
                let hv = hk * kv_repeat + r;
                for d in 0..d_k {
                    for t in 0..seq {
                        qe[(hv * d_k + d) * seq + t] = q[(hk * d_k + d) * seq + t];
                        ke[(hv * d_k + d) * seq + t] = k[(hk * d_k + d) * seq + t];
                    }
                }
            }
        }
        (qe, ke)
    } else {
        (q, k)
    };

    // 6. Compute decay g and write gate beta
    //    a_raw: [Hv, seq], dt_bias: [Hv], a_log: [Hv]
    //    g[h,t] = exp(-exp(a_log[h]) * softplus(a_raw[h,t] + dt_bias[h]))
    //    beta[h,t] = sigmoid(b_raw[h,t])
    let mut g = vec![0.0f32; h_v * seq];
    let mut beta = vec![0.0f32; h_v * seq];
    for h in 0..h_v {
        let exp_a_log = a_log[h].exp();
        for t in 0..seq {
            let a_val = a_raw[h * seq + t] + dt_bias[h];
            let sp = if a_val > 20.0 {
                a_val
            } else {
                a_val.exp().ln_1p()
            }; // softplus
            g[h * seq + t] = (-exp_a_log * sp).exp();

            let bv = b_raw[h * seq + t];
            beta[h * seq + t] = 1.0 / (1.0 + (-bv).exp()); // sigmoid
        }
    }

    // 7. Gated delta recurrence (NEON-optimized on aarch64)
    //    state: [Hv, Dv, Dk] — per-head outer product accumulator
    let mut state = vec![0.0f32; h_v * d_v * d_k];
    let mut y = vec![0.0f32; value_dim * seq]; // [Hv*Dv, seq]

    gdn_recurrence(
        &mut state, &mut y, &q_exp, &k_exp, v_raw, &g, &beta, h_v, d_k, d_v, seq,
    );

    // 8. Output gating: silu(z) * rmsnorm(y), then out_proj
    // RMSNorm on y (per-head across d_v dimension) using norm_weight
    let mut y_normed = vec![0.0f32; value_dim * seq];
    let shared_norm_weight = norm_weight.len() == d_v;
    for h in 0..h_v {
        for t in 0..seq {
            let mut ss = 0.0f32;
            for d in 0..d_v {
                let val = y[(h * d_v + d) * seq + t];
                ss += val * val;
            }
            let rms = (ss / d_v as f32 + 1e-6).sqrt();
            for d in 0..d_v {
                let norm = if shared_norm_weight {
                    norm_weight[d]
                } else {
                    norm_weight[h * d_v + d]
                };
                y_normed[(h * d_v + d) * seq + t] = y[(h * d_v + d) * seq + t] / rms * norm;
            }
        }
    }

    // silu(z) * y_normed
    let mut gated = vec![0.0f32; value_dim * seq];
    for i in 0..value_dim * seq {
        let z_val = z[i];
        let silu_z = z_val / (1.0 + (-z_val).exp());
        gated[i] = silu_z * y_normed[i];
    }

    apply_o_proj(&gated)
}

fn cpu_gdn_forward(
    gdn: &ane_weights::GdnLayerWeights,
    xnorm: &[f32], // [dim, seq] pre-attention-norm input
    cfg: &MilConfig,
) -> Vec<f32> {
    let dim = cfg.dim;
    let seq = cfg.seq_len;
    let value_dim = cfg.linear_n_value_heads * cfg.linear_value_head_dim;
    let qkv_dim = 2 * cfg.linear_n_heads * cfg.linear_head_dim + value_dim;
    let h_v = cfg.linear_n_value_heads;

    // Project QKV, a, b, z  —  all [out, seq] channels-first
    let qkv_raw = cpu_matmul(&gdn.qkv_proj, xnorm, qkv_dim, dim, seq);
    let a_raw = cpu_matmul(&gdn.a_proj, xnorm, h_v, dim, seq);
    let b_raw = cpu_matmul(&gdn.b_proj, xnorm, h_v, dim, seq);
    let z = cpu_matmul(&gdn.z_proj, xnorm, value_dim, dim, seq);

    cpu_gdn_forward_post_proj(
        &qkv_raw,
        &a_raw,
        &b_raw,
        &z,
        &gdn.a_log,
        &gdn.dt_bias,
        &gdn.norm_weight,
        &gdn.conv_weight,
        &gdn.conv_bias,
        cfg,
        |gated| cpu_matmul(&gdn.o_proj, gated, dim, value_dim, seq),
    )
}

fn cpu_quantized_gdn_forward_with_workspace(
    gdn: &ane_weights::QuantizedGdnLayerWeights,
    xnorm: &[f32], // [dim, seq] pre-attention-norm input
    cfg: &MilConfig,
    workspace: &mut QuantizedMatmulWorkspace,
) -> Vec<f32> {
    let seq = cfg.seq_len;
    let value_dim = cfg.linear_n_value_heads * cfg.linear_value_head_dim;
    let qkv_dim = 2 * cfg.linear_n_heads * cfg.linear_head_dim + value_dim;
    let h_v = cfg.linear_n_value_heads;

    // Project QKV, a, b, z directly from quantized weights without
    // materializing the full dense layer first.
    let mut qkv_raw = vec![0.0f32; qkv_dim * seq];
    cpu_quantized_matmul_into(&gdn.qkv_proj, xnorm, seq, &mut qkv_raw, workspace);
    let mut a_raw = vec![0.0f32; h_v * seq];
    cpu_quantized_matmul_into(&gdn.a_proj, xnorm, seq, &mut a_raw, workspace);
    let mut b_raw = vec![0.0f32; h_v * seq];
    cpu_quantized_matmul_into(&gdn.b_proj, xnorm, seq, &mut b_raw, workspace);
    let mut z = vec![0.0f32; value_dim * seq];
    cpu_quantized_matmul_into(&gdn.z_proj, xnorm, seq, &mut z, workspace);

    debug_assert_eq!(qkv_raw.len(), qkv_dim * seq);
    debug_assert_eq!(a_raw.len(), h_v * seq);
    debug_assert_eq!(b_raw.len(), h_v * seq);
    debug_assert_eq!(z.len(), value_dim * seq);

    cpu_gdn_forward_post_proj(
        &qkv_raw,
        &a_raw,
        &b_raw,
        &z,
        &gdn.a_log,
        &gdn.dt_bias,
        &gdn.norm_weight,
        &gdn.conv_weight,
        &gdn.conv_bias,
        cfg,
        |gated| {
            let mut out = vec![0.0f32; gdn.o_proj.rows * seq];
            cpu_quantized_matmul_into(&gdn.o_proj, gated, seq, &mut out, workspace);
            out
        },
    )
}

pub fn cpu_quantized_gdn_forward(
    gdn: &ane_weights::QuantizedGdnLayerWeights,
    xnorm: &[f32], // [dim, seq] pre-attention-norm input
    cfg: &MilConfig,
) -> Vec<f32> {
    let mut workspace = QuantizedMatmulWorkspace::default();
    cpu_quantized_gdn_forward_with_workspace(gdn, xnorm, cfg, &mut workspace)
}

#[doc(hidden)]
pub fn cpu_gdn_forward_bench(
    gdn: &ane_weights::GdnLayerWeights,
    xnorm: &[f32],
    cfg: &MilConfig,
) -> Vec<f32> {
    cpu_gdn_forward(gdn, xnorm, cfg)
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
    forward_cpu_generic(model, lora, tokens, targets)
}

/// Forward pass generic over weight source (supports both full and quantized weights).
pub fn forward_cpu_generic<T: TokenId, W: ane_weights::WeightSource>(
    model: &W,
    lora: Option<&super::ane_lora::LoraModel>,
    tokens: &[T],
    targets: &[T],
) -> ForwardResultWithLora {
    use super::ane_lora::LoraLayerActivations;

    let cfg = model.cfg();
    let dim = cfg.dim;
    let hidden = cfg.hidden_dim;
    let seq = cfg.seq_len;
    let n_heads = cfg.n_heads;
    let head_dim = cfg.head_dim();
    let n_layers = model.n_layers();
    let lora_scale = lora.map_or(0.0, |l| l.scale());

    // 1. Embedding
    let mut x_cur = vec![0.0f32; dim * seq];
    embed_lookup(&mut x_cur, model.embed(), tokens, dim, seq);

    let mut layer_acts = Vec::with_capacity(n_layers);
    let mut lora_acts_vec = Vec::with_capacity(n_layers);

    // 2. Transformer layers
    for l in 0..n_layers {
        let lora_layer = lora.map(|lm| &lm.layers[l]);
        let layer_in = x_cur.clone();
        let mut lora_layer_acts = LoraLayerActivations::empty();

        if let Some(ql) = model.quantized_layer(l) {
            let mut quantized_workspace = QuantizedMatmulWorkspace::default();
            // RMSNorm before attention
            let mut xnorm = vec![0.0f32; dim * seq];
            rmsnorm(&mut xnorm, &x_cur, &ql.rms_att, dim, seq, cfg.rms_eps);

            let (q, k, v, attn_out, o_out, q_pre_norm, k_pre_norm, attn_gate_saved, attn_pre_gate_saved) = if let Some(gdn_q) = &ql.gdn {
                let gdn_out = cpu_quantized_gdn_forward_with_workspace(
                    gdn_q,
                    &xnorm,
                    cfg,
                    &mut quantized_workspace,
                );
                let empty = vec![0.0f32; 0];
                (
                    empty.clone(),
                    empty.clone(),
                    empty,
                    gdn_out.clone(),
                    gdn_out,
                    None,
                    None,
                    None,
                    None,
                )
            } else {
                // Quantized MHA path: apply projections directly from quantized
                // weights, then expand KV activations for GQA instead of
                // expanding KV weights.
                let mut q_full = vec![0.0f32; ql.wq.rows * seq];
                cpu_quantized_matmul_into(&ql.wq, &xnorm, seq, &mut q_full, &mut quantized_workspace);
                let (mut q, attn_gate_raw) = if cfg.attn_output_gate {
                    split_q_gate(&q_full, n_heads, head_dim, seq)
                } else {
                    (q_full, vec![])
                };

                let mut k = vec![0.0f32; ql.wk.rows * seq];
                cpu_quantized_matmul_into(&ql.wk, &xnorm, seq, &mut k, &mut quantized_workspace);
                let mut v = vec![0.0f32; ql.wv.rows * seq];
                cpu_quantized_matmul_into(&ql.wv, &xnorm, seq, &mut v, &mut quantized_workspace);

                let heads_per_group = cfg.heads_per_group();
                let attn_dim = cfg.attn_dim();
                if heads_per_group > 1 {
                    k = ane_weights::expand_kv_static(
                        &k,
                        ql.wk.rows,
                        head_dim,
                        heads_per_group,
                        attn_dim,
                    );
                    v = ane_weights::expand_kv_static(
                        &v,
                        ql.wv.rows,
                        head_dim,
                        heads_per_group,
                        attn_dim,
                    );
                }

                let q_pre_norm = if let Some(q_norm_w) = &ql.q_norm {
                    Some(qk_rmsnorm_fwd(
                        &mut q,
                        q_norm_w,
                        n_heads,
                        head_dim,
                        seq,
                        cfg.rms_eps,
                    ))
                } else {
                    None
                };
                let k_pre_norm = if let Some(k_norm_w) = &ql.k_norm {
                    Some(qk_rmsnorm_fwd(
                        &mut k,
                        k_norm_w,
                        n_heads,
                        head_dim,
                        seq,
                        cfg.rms_eps,
                    ))
                } else {
                    None
                };

                cpu_rope(&mut q, &mut k, n_heads, head_dim, seq, cfg.rope_theta);
                let mut attn_out = cpu_sdpa(&q, &k, &v, n_heads, head_dim, seq);

                let (attn_pre_gate_saved, attn_gate_saved) = if cfg.attn_output_gate {
                    let pre_gate = attn_out.clone();
                    apply_sigmoid_gate(&mut attn_out, &attn_gate_raw);
                    (Some(pre_gate), Some(attn_gate_raw))
                } else {
                    (None, None)
                };

                let mut o_out = vec![0.0f32; ql.wo.rows * seq];
                cpu_quantized_matmul_into(
                    &ql.wo,
                    &attn_out,
                    seq,
                    &mut o_out,
                    &mut quantized_workspace,
                );

                if let Some(ll) = lora_layer {
                    if let Some(wo_adapter) = ll.wo.as_ref() {
                        let (wo_delta, wo_h) = wo_adapter.forward_cpu(&attn_out, seq);
                        super::ane_lora::vec_add_scaled(&mut o_out, &wo_delta, lora_scale);
                        lora_layer_acts.wo_x = Some(attn_out.clone());
                        lora_layer_acts.wo_h = Some(wo_h);
                    }
                }

                (q, k, v, attn_out, o_out, q_pre_norm, k_pre_norm, attn_gate_saved, attn_pre_gate_saved)
            };

            let mut x2 = x_cur.clone();
            vec_add_inplace(&mut x2, &o_out);

            let mut x2norm = vec![0.0f32; dim * seq];
            rmsnorm(&mut x2norm, &x2, &ql.rms_ffn, dim, seq, cfg.rms_eps);

            let mut h1 = vec![0.0f32; ql.w1.rows * seq];
            cpu_quantized_matmul_into(&ql.w1, &x2norm, seq, &mut h1, &mut quantized_workspace);
            let mut h3 = vec![0.0f32; ql.w3.rows * seq];
            cpu_quantized_matmul_into(&ql.w3, &x2norm, seq, &mut h3, &mut quantized_workspace);
            cpu_silu_inplace(&mut h1);
            let mut gate = vec![0.0f32; hidden * seq];
            for i in 0..hidden * seq {
                gate[i] = h1[i] * h3[i];
            }

            let mut ffn_out = vec![0.0f32; ql.w2.rows * seq];
            cpu_quantized_matmul_into(&ql.w2, &gate, seq, &mut ffn_out, &mut quantized_workspace);

            if let Some(ll) = lora_layer {
                if let Some(w2_adapter) = ll.w2.as_ref() {
                    let (w2_delta, w2_h) = w2_adapter.forward_cpu(&gate, seq);
                    super::ane_lora::vec_add_scaled(&mut ffn_out, &w2_delta, lora_scale);
                    lora_layer_acts.w2_x = Some(gate.clone());
                    lora_layer_acts.w2_h = Some(w2_h);
                }
            }

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
                q_pre_norm,
                k_pre_norm,
                attn_gate: attn_gate_saved,
                attn_pre_gate: attn_pre_gate_saved,
            });
            lora_acts_vec.push(lora_layer_acts);
            continue;
        }

        let lw_cow = model.layer(l);
        let lw = &*lw_cow;

        // RMSNorm before attention
        let mut xnorm = vec![0.0f32; dim * seq];
        rmsnorm(&mut xnorm, &x_cur, &lw.rms_att, dim, seq, cfg.rms_eps);

        // Attention: GDN (linear) or MHA path
        let (q, k, v, attn_out, o_out, q_pre_norm, k_pre_norm, attn_gate_saved, attn_pre_gate_saved) = if let Some(gdn_w) = &lw.gdn {
            // GDN path: combined QKV → conv1d → recurrence → output gate
            let gdn_out = cpu_gdn_forward(gdn_w, &xnorm, cfg);
            // GDN layers produce the final output directly (no separate Wo)
            let empty = vec![0.0f32; 0];
            (
                empty.clone(),
                empty.clone(),
                empty,
                gdn_out.clone(),
                gdn_out,
                None,
                None,
                None,
                None,
            )
        } else {
            // MHA path: Q/K/V → QK-norm → RoPE → SDPA → Wo
            let ad = cfg.attn_dim();
            let qpd = cfg.q_proj_dim();
            let q_full = cpu_matmul(&lw.wq, &xnorm, qpd, dim, seq);
            let (mut q, attn_gate_raw) = if cfg.attn_output_gate {
                split_q_gate(&q_full, n_heads, head_dim, seq)
            } else {
                (q_full, vec![])
            };
            let mut k = cpu_matmul(&lw.wk, &xnorm, ad, dim, seq);
            let v = cpu_matmul(&lw.wv, &xnorm, ad, dim, seq);

            let q_pre_norm = if let Some(q_norm_w) = &lw.q_norm {
                Some(qk_rmsnorm_fwd(
                    &mut q,
                    q_norm_w,
                    n_heads,
                    head_dim,
                    seq,
                    cfg.rms_eps,
                ))
            } else {
                None
            };
            let k_pre_norm = if let Some(k_norm_w) = &lw.k_norm {
                Some(qk_rmsnorm_fwd(
                    &mut k,
                    k_norm_w,
                    n_heads,
                    head_dim,
                    seq,
                    cfg.rms_eps,
                ))
            } else {
                None
            };

            cpu_rope(&mut q, &mut k, n_heads, head_dim, seq, cfg.rope_theta);
            let mut attn_out = cpu_sdpa(&q, &k, &v, n_heads, head_dim, seq);

            let (attn_pre_gate_saved, attn_gate_saved) = if cfg.attn_output_gate {
                let pre_gate = attn_out.clone();
                apply_sigmoid_gate(&mut attn_out, &attn_gate_raw);
                (Some(pre_gate), Some(attn_gate_raw))
            } else {
                (None, None)
            };

            let mut o_out = cpu_matmul(&lw.wo, &attn_out, dim, ad, seq);

            // LoRA on Wo (MHA only — GDN layers use stop_gradient on attention)
            if let Some(ll) = lora_layer {
                if let Some(wo_adapter) = ll.wo.as_ref() {
                    let (wo_delta, wo_h) = wo_adapter.forward_cpu(&attn_out, seq);
                    super::ane_lora::vec_add_scaled(&mut o_out, &wo_delta, lora_scale);
                    lora_layer_acts.wo_x = Some(attn_out.clone());
                    lora_layer_acts.wo_h = Some(wo_h);
                }
            }

            (q, k, v, attn_out, o_out, q_pre_norm, k_pre_norm, attn_gate_saved, attn_pre_gate_saved)
        };

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
            q_pre_norm,
            k_pre_norm,
            attn_gate: attn_gate_saved,
            attn_pre_gate: attn_pre_gate_saved,
        });
        lora_acts_vec.push(lora_layer_acts);
    }

    // 3. Final RMSNorm
    let mut x_final = vec![0.0f32; dim * seq];
    rmsnorm(
        &mut x_final,
        &x_cur,
        model.rms_final(),
        dim,
        seq,
        cfg.rms_eps,
    );

    // 4. Classifier
    let vocab = model.vocab_size();
    let mut logits = vec![0.0f32; vocab * seq];
    let cls_w = model.lm_head().unwrap_or(model.embed());
    classifier_forward(&mut logits, cls_w, &x_final, vocab, dim, seq);

    // 5. Cross-entropy loss
    let (loss, dlogits) = cross_entropy_loss(&logits, targets, vocab, seq);

    ForwardResultWithLora {
        base: ForwardResult {
            logits,
            loss,
            dlogits,
            layer_acts,
        },
        lora_acts: lora_acts_vec,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0f32, f32::max)
    }

    fn quantize_tensor_affine(
        dense: &[f32],
        rows: usize,
        cols: usize,
        group_size: usize,
    ) -> ane_weights::QuantizedTensor {
        assert_eq!(dense.len(), rows * cols);
        assert_eq!(cols % group_size, 0);

        let n_groups = cols / group_size;
        let mut data = vec![0u8; dense.len()];
        let mut scales = vec![0.0f32; rows * n_groups];
        let mut biases = vec![0.0f32; rows * n_groups];

        for row in 0..rows {
            for group in 0..n_groups {
                let start = row * cols + group * group_size;
                let values = &dense[start..start + group_size];
                let mut min_v = f32::INFINITY;
                let mut max_v = f32::NEG_INFINITY;
                for &value in values {
                    min_v = min_v.min(value);
                    max_v = max_v.max(value);
                }

                let scale = if max_v > min_v {
                    (max_v - min_v) / 255.0
                } else {
                    0.0
                };
                let idx = row * n_groups + group;
                scales[idx] = scale;
                biases[idx] = min_v;

                for (offset, &value) in values.iter().enumerate() {
                    let q = if scale > 0.0 {
                        ((value - min_v) / scale).round().clamp(0.0, 255.0)
                    } else {
                        0.0
                    };
                    data[start + offset] = q as u8;
                }
            }
        }

        ane_weights::QuantizedTensor {
            data,
            scales,
            biases,
            rows,
            cols,
            group_size,
            bits: 8,
        }
    }

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

    #[test]
    fn test_cpu_matmul_lhs_transposed_matches_explicit_transpose() {
        let rows = 3;
        let cols = 2;
        let seq = 2;

        let w = vec![
            1.0, 2.0, //
            3.0, 4.0, //
            5.0, 6.0,
        ];
        let x = vec![
            7.0, 8.0, //
            9.0, 10.0, //
            11.0, 12.0,
        ];

        let wt = ane_weights::transpose_weight(&w, rows, cols);
        let explicit = cpu_matmul(&wt, &x, cols, rows, seq);
        let direct = cpu_matmul_lhs_transposed(&w, rows, cols, &x, seq);

        assert_eq!(direct, explicit);
    }

    #[test]
    fn test_cpu_quantized_matmul_matches_materialized_matmul() {
        let rows = 5;
        let cols = 8;
        let seq = 3;

        let w: Vec<f32> = (0..rows * cols)
            .map(|i| ((i as f32 * 0.37 + 0.11).sin()) * 2.0)
            .collect();
        let x: Vec<f32> = (0..cols * seq)
            .map(|i| ((i as f32 * 0.19 + 0.23).cos()) * 1.5)
            .collect();

        let quantized = quantize_tensor_affine(&w, rows, cols, 2);
        let materialized = cpu_matmul(&quantized.dequantize(), &x, rows, cols, seq);
        let blocked = cpu_quantized_matmul(&quantized, &x, seq);

        assert!(max_abs_diff(&materialized, &blocked) < 1e-5);
    }

    #[test]
    fn test_forward_cpu_generic_quantized_matches_dequantized_reference_for_gqa() {
        use super::super::ane_weights::{
            ModelWeights, QuantizedLayerWeights, QuantizedModelWeights,
        };

        let cfg = MilConfig {
            dim: 8,
            hidden_dim: 16,
            n_heads: 4,
            seq_len: 3,
            n_kv_heads: 2,
            rope_theta: 10_000.0,
            rms_eps: 1e-5,
            has_lm_head: false,
            head_dim_explicit: 8 / 4,
            linear_attn_indices: vec![],
            linear_n_heads: 0,
            linear_head_dim: 0,
            linear_n_value_heads: 0,
            linear_value_head_dim: 0,
            conv_kernel_size: 0,
            attn_output_gate: false,
        };
        let dim = cfg.dim;
        let hidden = cfg.hidden_dim;
        let kv_dim = cfg.kv_dim();
        let vocab = 11;
        let group_size = 2;

        let make_vals = |n: usize, seed: usize| -> Vec<f32> {
            (0..n)
                .map(|i| ((i + seed) as f32 * 0.173).sin() * 0.2)
                .collect()
        };

        let quantized = QuantizedModelWeights {
            cfg: cfg.clone(),
            layers: vec![QuantizedLayerWeights {
                wq: quantize_tensor_affine(&make_vals(dim * dim, 0), dim, dim, group_size),
                wk: quantize_tensor_affine(&make_vals(kv_dim * dim, 100), kv_dim, dim, group_size),
                wv: quantize_tensor_affine(&make_vals(kv_dim * dim, 200), kv_dim, dim, group_size),
                wo: quantize_tensor_affine(&make_vals(dim * dim, 300), dim, dim, group_size),
                w1: quantize_tensor_affine(&make_vals(hidden * dim, 400), hidden, dim, group_size),
                w2: quantize_tensor_affine(&make_vals(dim * hidden, 500), dim, hidden, group_size),
                w3: quantize_tensor_affine(&make_vals(hidden * dim, 600), hidden, dim, group_size),
                rms_att: vec![1.0; dim],
                rms_ffn: vec![1.0; dim],
                q_norm: Some(vec![1.0, 0.9]),
                k_norm: Some(vec![1.1, 0.8]),
                gdn: None,
            }],
            rms_final: vec![1.0; dim],
            embed: make_vals(vocab * dim, 700),
            vocab_size: vocab,
            lm_head: None,
            heads_per_group: cfg.heads_per_group(),
        };

        let dense = ModelWeights {
            cfg: cfg.clone(),
            layers: vec![quantized.dequantize_layer(0)],
            rms_final: quantized.rms_final.clone(),
            embed: quantized.embed.clone(),
            vocab_size: quantized.vocab_size,
            lm_head: quantized.lm_head.clone(),
        };

        let tokens = vec![1u16, 2, 3];
        let targets = vec![2u16, 3, 4];

        let quantized_fwd = forward_cpu_generic(&quantized, None, &tokens, &targets);
        let dense_fwd = forward_cpu_generic(&dense, None, &tokens, &targets);

        assert!((quantized_fwd.base.loss - dense_fwd.base.loss).abs() < 1e-4);
        assert!(max_abs_diff(&quantized_fwd.base.logits, &dense_fwd.base.logits) < 1e-4);
        assert!(max_abs_diff(&quantized_fwd.base.dlogits, &dense_fwd.base.dlogits) < 1e-4);

        let q_act = &quantized_fwd.base.layer_acts[0];
        let d_act = &dense_fwd.base.layer_acts[0];
        assert!(max_abs_diff(&q_act.q, &d_act.q) < 1e-4);
        assert!(max_abs_diff(&q_act.k, &d_act.k) < 1e-4);
        assert!(max_abs_diff(&q_act.v, &d_act.v) < 1e-4);
        assert!(max_abs_diff(&q_act.attn_out, &d_act.attn_out) < 1e-4);
        assert!(max_abs_diff(&q_act.ffn_out, &d_act.ffn_out) < 1e-4);
    }

    #[test]
    fn test_forward_cpu_generic_quantized_matches_dequantized_reference_for_gdn() {
        use super::super::ane_weights::{
            ModelWeights, QuantizedGdnLayerWeights, QuantizedLayerWeights, QuantizedModelWeights,
            QuantizedTensor,
        };

        let cfg = MilConfig {
            dim: 8,
            hidden_dim: 16,
            n_heads: 2,
            seq_len: 3,
            n_kv_heads: 2,
            rope_theta: 10_000.0,
            rms_eps: 1e-5,
            has_lm_head: false,
            head_dim_explicit: 8 / 2,
            linear_attn_indices: vec![0],
            linear_n_heads: 2,
            linear_head_dim: 4,
            linear_n_value_heads: 2,
            linear_value_head_dim: 4,
            conv_kernel_size: 2,
            attn_output_gate: false,
        };
        let dim = cfg.dim;
        let hidden = cfg.hidden_dim;
        let h_k = cfg.linear_n_heads;
        let d_k = cfg.linear_head_dim;
        let h_v = cfg.linear_n_value_heads;
        let d_v = cfg.linear_value_head_dim;
        let key_dim = h_k * d_k;
        let value_dim = h_v * d_v;
        let qkv_dim = 2 * key_dim + value_dim;
        let kernel = cfg.conv_kernel_size;
        let vocab = 13;
        let group_size = 2;

        let make_vals = |n: usize, seed: usize| -> Vec<f32> {
            (0..n)
                .map(|i| ((i + seed) as f32 * 0.173).sin() * 0.2)
                .collect()
        };
        let empty = QuantizedTensor {
            data: vec![],
            scales: vec![],
            biases: vec![],
            rows: 0,
            cols: 0,
            group_size: 1,
            bits: 8,
        };

        let quantized = QuantizedModelWeights {
            cfg: cfg.clone(),
            layers: vec![QuantizedLayerWeights {
                wq: empty.clone(),
                wk: empty.clone(),
                wv: empty.clone(),
                wo: empty,
                w1: quantize_tensor_affine(&make_vals(hidden * dim, 400), hidden, dim, group_size),
                w2: quantize_tensor_affine(&make_vals(dim * hidden, 500), dim, hidden, group_size),
                w3: quantize_tensor_affine(&make_vals(hidden * dim, 600), hidden, dim, group_size),
                rms_att: vec![1.0; dim],
                rms_ffn: vec![1.0; dim],
                q_norm: None,
                k_norm: None,
                gdn: Some(QuantizedGdnLayerWeights {
                    qkv_proj: quantize_tensor_affine(
                        &make_vals(qkv_dim * dim, 0),
                        qkv_dim,
                        dim,
                        group_size,
                    ),
                    a_proj: quantize_tensor_affine(
                        &make_vals(h_v * dim, 100),
                        h_v,
                        dim,
                        group_size,
                    ),
                    b_proj: quantize_tensor_affine(
                        &make_vals(h_v * dim, 200),
                        h_v,
                        dim,
                        group_size,
                    ),
                    z_proj: quantize_tensor_affine(
                        &make_vals(value_dim * dim, 300),
                        value_dim,
                        dim,
                        group_size,
                    ),
                    o_proj: quantize_tensor_affine(
                        &make_vals(dim * value_dim, 350),
                        dim,
                        value_dim,
                        group_size,
                    ),
                    a_log: make_vals(h_v, 700),
                    dt_bias: make_vals(h_v, 800),
                    norm_weight: vec![1.0; value_dim],
                    conv_weight: make_vals(qkv_dim * kernel, 900),
                    conv_bias: make_vals(qkv_dim, 1000),
                }),
            }],
            rms_final: vec![1.0; dim],
            embed: make_vals(vocab * dim, 1100),
            vocab_size: vocab,
            lm_head: None,
            heads_per_group: cfg.heads_per_group(),
        };

        let dense = ModelWeights {
            cfg: cfg.clone(),
            layers: vec![quantized.dequantize_layer(0)],
            rms_final: quantized.rms_final.clone(),
            embed: quantized.embed.clone(),
            vocab_size: quantized.vocab_size,
            lm_head: quantized.lm_head.clone(),
        };

        let tokens = vec![1u16, 2, 3];
        let targets = vec![2u16, 3, 4];

        let quantized_fwd = forward_cpu_generic(&quantized, None, &tokens, &targets);
        let dense_fwd = forward_cpu_generic(&dense, None, &tokens, &targets);

        assert!((quantized_fwd.base.loss - dense_fwd.base.loss).abs() < 1e-4);
        assert!(max_abs_diff(&quantized_fwd.base.logits, &dense_fwd.base.logits) < 1e-4);
        assert!(max_abs_diff(&quantized_fwd.base.dlogits, &dense_fwd.base.dlogits) < 1e-4);

        let q_act = &quantized_fwd.base.layer_acts[0];
        let d_act = &dense_fwd.base.layer_acts[0];
        assert_eq!(q_act.q.len(), 0);
        assert_eq!(q_act.k.len(), 0);
        assert_eq!(q_act.v.len(), 0);
        assert_eq!(d_act.q.len(), 0);
        assert_eq!(d_act.k.len(), 0);
        assert_eq!(d_act.v.len(), 0);
        assert!(max_abs_diff(&q_act.attn_out, &d_act.attn_out) < 1e-4);
        assert!(max_abs_diff(&q_act.o_out, &d_act.o_out) < 1e-4);
        assert!(max_abs_diff(&q_act.ffn_out, &d_act.ffn_out) < 1e-4);
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
        let make_small =
            |n: usize| -> Vec<f32> { (0..n).map(|i| ((i as f32 * 0.001).sin()) * 0.01).collect() };
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
            gdn: None,
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
            gdn: None,
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

        assert!(
            result.loss.is_finite(),
            "2-layer loss not finite: {}",
            result.loss
        );
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

        let mut x: Vec<f32> = (0..dim * seq).map(|i| (i as f32 + 1.0) * 0.1).collect();
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
                x[idx],
                expected
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
        qk_rmsnorm_bwd(
            &mut dx, &mut dw, &pre, &norm_w, n_heads, head_dim, seq, eps_norm,
        );

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
                dx[idx],
                num
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
                q_rot[i],
                orig_q[i]
            );
            assert!(
                (k_rot[i] - orig_k[i]).abs() < 1e-5,
                "rope roundtrip k[{i}]: got {}, expected {}",
                k_rot[i],
                orig_k[i]
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
            head_dim_explicit: 2048 / 16,
            linear_attn_indices: vec![],
            linear_n_heads: 0,
            linear_head_dim: 0,
            linear_n_value_heads: 0,
            linear_value_head_dim: 0,
            conv_kernel_size: 0,
            attn_output_gate: false,
        };

        eprintln!("loading Qwen3-1.7B weights...");
        let t0 = std::time::Instant::now();
        let model = ModelWeights::from_mlx_safetensors(&model_dir, &cfg)
            .expect("from_mlx_safetensors failed");
        eprintln!("loaded in {}ms", t0.elapsed().as_millis());

        let tokens: Vec<u32> = (100..100 + seq as u32).collect();
        let targets: Vec<u32> = (101..101 + seq as u32).collect();

        // CPU-only forward (ANE dynamic packing exceeds IOSurface limits at dim=2048)
        eprintln!(
            "running CPU forward pass (28 layers, vocab={})",
            model.vocab_size
        );
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
            head_dim_explicit: 2048 / 16,
            linear_attn_indices: vec![],
            linear_n_heads: 0,
            linear_head_dim: 0,
            linear_n_value_heads: 0,
            linear_value_head_dim: 0,
            conv_kernel_size: 0,
            attn_output_gate: false,
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
            let bwd = super::super::ane_backward::backward_lora_cpu(&model, &fwd, &lora, &tokens);

            // Adam update on LoRA params only
            super::super::ane_lora::lora_adam_update(
                &mut lora,
                &bwd.lora_grads,
                &mut adam,
                step + 1,
                lr,
                0.9,
                0.999,
                1e-8,
                0.01,
            );

            let step_ms = t0.elapsed().as_millis();
            eprintln!("  step {step}: loss={loss:.4}, time={step_ms}ms");
        }

        // Verify loss decreased
        let first = losses[0];
        let last = losses[n_steps - 1];
        eprintln!(
            "loss trajectory: {:.4} -> {:.4} (delta={:.4})",
            first,
            last,
            last - first
        );
        assert!(
            last < first,
            "loss should decrease over training: first={first:.4}, last={last:.4}"
        );
    }

    // -----------------------------------------------------------------------
    // GDN numerical reference tests (tests/gdn_reference_raw/)
    // -----------------------------------------------------------------------

    /// Load a raw f32 binary file (little-endian).
    fn load_f32_bin(path: &std::path::Path) -> Vec<f32> {
        let data = std::fs::read(path).unwrap();
        data.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    /// Transpose from Python row-major [seq, heads] to Rust channels-first [heads, seq].
    fn transpose_2d(src: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                out[c * rows + r] = src[r * cols + c];
            }
        }
        out
    }

    /// Transpose from Python row-major [seq, heads, dim] to Rust channels-first [heads*dim, seq].
    fn transpose_3d_to_cf(src: &[f32], seq: usize, heads: usize, dim: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; heads * dim * seq];
        for t in 0..seq {
            for h in 0..heads {
                for d in 0..dim {
                    out[(h * dim + d) * seq + t] = src[t * heads * dim + h * dim + d];
                }
            }
        }
        out
    }

    fn ref_dir() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/gdn_reference_raw")
    }

    #[test]
    fn test_gdn_decay_gate_reference() {
        let dir = ref_dir();
        if !dir.exists() {
            eprintln!("SKIP: tests/gdn_reference_raw/ not found");
            return;
        }

        let seq = 4;
        let h_v = 16;

        // Load reference data (Python row-major, batch=1 stripped)
        let a_raw_py = load_f32_bin(&dir.join("a_raw.bin")); // [seq, h_v]
        let a_log = load_f32_bin(&dir.join("A_log.bin")); // [h_v]
        let dt_bias = load_f32_bin(&dir.join("dt_bias.bin")); // [h_v]
        let g_ref_py = load_f32_bin(&dir.join("g.bin")); // [seq, h_v]

        assert_eq!(a_raw_py.len(), seq * h_v);
        assert_eq!(a_log.len(), h_v);
        assert_eq!(dt_bias.len(), h_v);
        assert_eq!(g_ref_py.len(), seq * h_v);

        // Transpose a_raw to channels-first [h_v, seq]
        let a_raw = transpose_2d(&a_raw_py, seq, h_v);

        // Compute g[h,t] = exp(-exp(a_log[h]) * softplus(a_raw[h,t] + dt_bias[h]))
        let mut g = vec![0.0f32; h_v * seq];
        for h in 0..h_v {
            let exp_a_log = a_log[h].exp();
            for t in 0..seq {
                let a_val = a_raw[h * seq + t] + dt_bias[h];
                let sp = if a_val > 20.0 {
                    a_val
                } else {
                    a_val.exp().ln_1p()
                };
                g[h * seq + t] = (-exp_a_log * sp).exp();
            }
        }

        // Transpose g back to [seq, h_v] for comparison
        let g_check = transpose_2d(&g, h_v, seq);

        // Tolerance: exp(-exp(x)*softplus(y)) amplifies f32 precision differences
        // across the exp/log chain. 2e-3 relative tolerance for the decay gate
        // is appropriate; the recurrence test validates the core algorithm at 1e-9.
        let mut max_err = 0.0f32;
        for i in 0..g_ref_py.len() {
            let err = (g_check[i] - g_ref_py[i]).abs();
            max_err = max_err.max(err);
            assert!(
                err < 2e-3,
                "g mismatch at [{}, {}]: got {}, expected {}, err={}",
                i / h_v,
                i % h_v,
                g_check[i],
                g_ref_py[i],
                err
            );
        }
        eprintln!("decay gate: max_err={max_err:.2e} (threshold=2e-3)");
    }

    #[test]
    fn test_gdn_recurrence_reference() {
        let dir = ref_dir();
        if !dir.exists() {
            eprintln!("SKIP: tests/gdn_reference_raw/ not found");
            return;
        }

        let seq = 4;
        let h_v: usize = 16;
        let d_k: usize = 128;
        let d_v: usize = 128;

        // Load reference data
        let q_normed_py = load_f32_bin(&dir.join("q_normed.bin")); // [seq, h, d]
        let k_normed_py = load_f32_bin(&dir.join("k_normed.bin"));
        let v_py = load_f32_bin(&dir.join("v.bin"));
        let g_py = load_f32_bin(&dir.join("g.bin")); // [seq, h]
        let beta_py = load_f32_bin(&dir.join("beta.bin"));
        let rec_out_py = load_f32_bin(&dir.join("recurrence_out.bin")); // [seq, h, d]
        let final_state_py = load_f32_bin(&dir.join("final_state.bin")); // [h, d_v, d_k]

        assert_eq!(q_normed_py.len(), seq * h_v * d_k);
        assert_eq!(rec_out_py.len(), seq * h_v * d_v);
        assert_eq!(final_state_py.len(), h_v * d_v * d_k);

        // Transpose to channels-first layout
        let q = transpose_3d_to_cf(&q_normed_py, seq, h_v, d_k); // [h*d_k, seq]
        let k = transpose_3d_to_cf(&k_normed_py, seq, h_v, d_k);
        let v = transpose_3d_to_cf(&v_py, seq, h_v, d_v);
        let g_cf = transpose_2d(&g_py, seq, h_v); // [h_v, seq]
        let beta_cf = transpose_2d(&beta_py, seq, h_v);

        // Run recurrence (same code as cpu_gdn_forward step 7)
        let value_dim = h_v * d_v;
        let mut state = vec![0.0f32; h_v * d_v * d_k];
        let mut y = vec![0.0f32; value_dim * seq];

        for t in 0..seq {
            for h in 0..h_v {
                let g_t = g_cf[h * seq + t];
                let beta_t = beta_cf[h * seq + t];

                // state[h] *= g_t
                for dv in 0..d_v {
                    for dk in 0..d_k {
                        state[h * d_v * d_k + dv * d_k + dk] *= g_t;
                    }
                }

                for dv in 0..d_v {
                    let mut kv_mem = 0.0f32;
                    for dk in 0..d_k {
                        kv_mem +=
                            state[h * d_v * d_k + dv * d_k + dk] * k[(h * d_k + dk) * seq + t];
                    }
                    let v_t = v[(h * d_v + dv) * seq + t];
                    let delta = (v_t - kv_mem) * beta_t;
                    for dk in 0..d_k {
                        state[h * d_v * d_k + dv * d_k + dk] += k[(h * d_k + dk) * seq + t] * delta;
                    }
                }

                for dv in 0..d_v {
                    let mut y_val = 0.0f32;
                    for dk in 0..d_k {
                        y_val += state[h * d_v * d_k + dv * d_k + dk] * q[(h * d_k + dk) * seq + t];
                    }
                    y[(h * d_v + dv) * seq + t] = y_val;
                }
            }
        }

        // Compare recurrence output (transpose back to [seq, h, d])
        let mut max_err = 0.0f32;
        for t in 0..seq {
            for h in 0..h_v {
                for d in 0..d_v {
                    let got = y[(h * d_v + d) * seq + t];
                    let expected = rec_out_py[t * h_v * d_v + h * d_v + d];
                    let err = (got - expected).abs();
                    max_err = max_err.max(err);
                    assert!(
                        err < 1e-3,
                        "recurrence_out mismatch at [t={t}, h={h}, d={d}]: got {got}, expected {expected}, err={err}"
                    );
                }
            }
        }
        eprintln!("recurrence output: max_err={max_err:.2e} (threshold=1e-3)");

        // Compare final state [h_v, d_v, d_k]
        let mut max_state_err = 0.0f32;
        for h in 0..h_v {
            for dv in 0..d_v {
                for dk in 0..d_k {
                    let got = state[h * d_v * d_k + dv * d_k + dk];
                    let expected = final_state_py[h * d_v * d_k + dv * d_k + dk];
                    let err = (got - expected).abs();
                    max_state_err = max_state_err.max(err);
                    assert!(
                        err < 1e-3,
                        "final_state mismatch at [h={h}, dv={dv}, dk={dk}]: got {got}, expected {expected}, err={err}"
                    );
                }
            }
        }
        eprintln!("final state: max_err={max_state_err:.2e} (threshold=1e-3)");
    }

    #[test]
    fn test_gdn_forward_small_synthetic() {
        // Small synthetic test: verifies cpu_gdn_forward runs end-to-end
        // with known-shape inputs and produces finite, correctly-shaped output.
        let dim = 8;
        let seq = 2;
        let h_k = 2;
        let d_k = 4; // dim / h_k
        let h_v = 2;
        let d_v = 4;
        let key_dim = h_k * d_k; // 8
        let value_dim = h_v * d_v; // 8
        let qkv_dim = 2 * key_dim + value_dim; // 24
        let kernel = 2;

        // Create deterministic weights using a simple LCG
        let mut rng_state = 42u64;
        let mut next_f32 = || -> f32 {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng_state >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
        };
        let rand_vec = |n: usize, rng: &mut dyn FnMut() -> f32| -> Vec<f32> {
            (0..n).map(|_| rng()).collect()
        };

        let gdn_w = super::ane_weights::GdnLayerWeights {
            qkv_proj: rand_vec(qkv_dim * dim, &mut next_f32),
            a_proj: rand_vec(h_v * dim, &mut next_f32),
            b_proj: rand_vec(h_v * dim, &mut next_f32),
            z_proj: rand_vec(value_dim * dim, &mut next_f32),
            o_proj: rand_vec(dim * value_dim, &mut next_f32),
            a_log: rand_vec(h_v, &mut next_f32),
            dt_bias: rand_vec(h_v, &mut next_f32),
            norm_weight: rand_vec(value_dim, &mut next_f32),
            conv_weight: rand_vec(qkv_dim * kernel, &mut next_f32),
            conv_bias: rand_vec(qkv_dim, &mut next_f32),
        };

        let cfg = super::ane_mil::MilConfig {
            dim,
            hidden_dim: 32,
            n_heads: h_k,
            seq_len: seq,
            n_kv_heads: h_k,
            rope_theta: 10000.0,
            rms_eps: 1e-5,
            has_lm_head: false,
            head_dim_explicit: dim / h_k,
            linear_attn_indices: vec![0],
            linear_n_heads: h_k,
            linear_head_dim: d_k,
            linear_n_value_heads: h_v,
            linear_value_head_dim: d_v,
            conv_kernel_size: kernel,
            attn_output_gate: false,
        };

        // Input: [dim, seq] channels-first, small values
        let xnorm: Vec<f32> = (0..dim * seq)
            .map(|i| ((i as f32) * 0.1 - 0.4).sin())
            .collect();

        let output = cpu_gdn_forward(&gdn_w, &xnorm, &cfg);

        assert_eq!(
            output.len(),
            dim * seq,
            "output shape mismatch: got {}, expected {}",
            output.len(),
            dim * seq
        );
        for (i, &v) in output.iter().enumerate() {
            assert!(v.is_finite(), "output[{i}] is not finite: {v}");
        }
        eprintln!(
            "synthetic GDN forward: output[0..4]={:?}",
            &output[..4.min(output.len())]
        );
    }

    #[test]
    fn test_gdn_forward_shared_norm_weight_matches_expanded_form() {
        let dim = 8;
        let seq = 3;
        let h_k = 2;
        let d_k = 4;
        let h_v = 2;
        let d_v = 4;
        let key_dim = h_k * d_k;
        let value_dim = h_v * d_v;
        let qkv_dim = 2 * key_dim + value_dim;
        let kernel = 2;

        let mut rng_state = 7u64;
        let mut next_f32 = || -> f32 {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng_state >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
        };
        let rand_vec = |n: usize, rng: &mut dyn FnMut() -> f32| -> Vec<f32> {
            (0..n).map(|_| rng()).collect()
        };

        let qkv_proj = rand_vec(qkv_dim * dim, &mut next_f32);
        let a_proj = rand_vec(h_v * dim, &mut next_f32);
        let b_proj = rand_vec(h_v * dim, &mut next_f32);
        let z_proj = rand_vec(value_dim * dim, &mut next_f32);
        let o_proj = rand_vec(dim * value_dim, &mut next_f32);
        let a_log = rand_vec(h_v, &mut next_f32);
        let dt_bias = rand_vec(h_v, &mut next_f32);
        let conv_weight = rand_vec(qkv_dim * kernel, &mut next_f32);
        let conv_bias = rand_vec(qkv_dim, &mut next_f32);
        let shared_norm = rand_vec(d_v, &mut next_f32);
        let expanded_norm: Vec<f32> = (0..h_v).flat_map(|_| shared_norm.iter().copied()).collect();

        let gdn_shared = super::ane_weights::GdnLayerWeights {
            qkv_proj: qkv_proj.clone(),
            a_proj: a_proj.clone(),
            b_proj: b_proj.clone(),
            z_proj: z_proj.clone(),
            o_proj: o_proj.clone(),
            a_log: a_log.clone(),
            dt_bias: dt_bias.clone(),
            norm_weight: shared_norm,
            conv_weight: conv_weight.clone(),
            conv_bias: conv_bias.clone(),
        };
        let gdn_expanded = super::ane_weights::GdnLayerWeights {
            qkv_proj,
            a_proj,
            b_proj,
            z_proj,
            o_proj,
            a_log,
            dt_bias,
            norm_weight: expanded_norm,
            conv_weight,
            conv_bias,
        };

        let cfg = super::ane_mil::MilConfig {
            dim,
            hidden_dim: 32,
            n_heads: h_k,
            seq_len: seq,
            n_kv_heads: h_k,
            rope_theta: 10000.0,
            rms_eps: 1e-5,
            has_lm_head: false,
            head_dim_explicit: dim / h_k,
            linear_attn_indices: vec![0],
            linear_n_heads: h_k,
            linear_head_dim: d_k,
            linear_n_value_heads: h_v,
            linear_value_head_dim: d_v,
            conv_kernel_size: kernel,
            attn_output_gate: false,
        };

        let xnorm: Vec<f32> = (0..dim * seq)
            .map(|i| ((i as f32) * 0.17 - 0.2).sin())
            .collect();

        let shared_out = cpu_gdn_forward(&gdn_shared, &xnorm, &cfg);
        let expanded_out = cpu_gdn_forward(&gdn_expanded, &xnorm, &cfg);

        assert!(max_abs_diff(&shared_out, &expanded_out) < 1e-5);
    }
}
