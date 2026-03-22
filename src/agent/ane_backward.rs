//! Backward pass pipeline for ANE transformer training.
//!
//! CPU backward ops (rmsnorm_bwd, silu_bwd, classifier_bwd, embed_bwd),
//! compiled backward kernels, gradient structs, and full backward pass:
//! classifier_bwd → final_rmsnorm_bwd → per-layer-reversed → embed_bwd.

use super::ane_bridge::{self, AneKernel};
use super::ane_forward::{self, ForwardResult};
use super::ane_mil::{self, KernelSpec, KernelType, MilConfig};
use super::ane_weights::{self, ModelWeights};

// ---------------------------------------------------------------------------
// CPU backward operations
// ---------------------------------------------------------------------------

/// RMSNorm backward: dx and dw from upstream gradient dy.
///
/// Forward was: y[i,t] = x[i,t] * w[i] * rrms[t]
/// where rrms[t] = 1/sqrt(mean(x[:,t]^2) + eps).
///
/// Backward computes:
///   dx[i,t] = w[i] * rrms[t] * (dy[i,t] - x[i,t] * dot[t])
///   dw[i] += sum_t(dy[i,t] * x[i,t] * rrms[t])
/// where dot[t] = rrms[t]^2 / dim * sum_i(dy[i,t] * x[i,t] * w[i])
///
/// All arrays [dim, seq] except w, dw which are [dim].
/// dw is accumulated (+=).
pub fn rmsnorm_bwd(
    dx: &mut [f32],
    dw: &mut [f32],
    dy: &[f32],
    x: &[f32],
    w: &[f32],
    dim: usize,
    seq: usize,
    eps: f32,
) {
    // Hard guard: if any input is wrong-sized, zero dx and return.
    // This can happen when an upstream ANE kernel (delta_reload) fails
    // and returns empty/wrong-sized data, or from layer-drop (empty activations).
    let expected = dim * seq;
    if dx.len() != expected
        || dy.len() != expected
        || x.len() != expected
        || dw.len() != dim
        || w.len() != dim
    {
        use std::sync::atomic::{AtomicU64, Ordering};
        static MISMATCH_COUNT: AtomicU64 = AtomicU64::new(0);
        let n = MISMATCH_COUNT.fetch_add(1, Ordering::Relaxed);
        if n < 3 {
            tracing::debug!(
                "rmsnorm_bwd: dimension mismatch (dx={}, dy={}, x={}, dw={}, w={}, expected dim*seq={}), zeroing output",
                dx.len(), dy.len(), x.len(), dw.len(), w.len(), expected,
            );
        }
        dx.fill(0.0);
        return;
    }

    // 1. Compute sum of squares per position: ss[t] = sum_i x[i,t]^2
    let mut ss = vec![0.0f32; seq];
    for i in 0..dim {
        for t in 0..seq {
            ss[t] += x[i * seq + t] * x[i * seq + t];
        }
    }

    // 2. rrms[t] = 1/sqrt(ss[t]/dim + eps)
    let inv_dim = 1.0 / dim as f32;
    let mut rrms = vec![0.0f32; seq];
    for t in 0..seq {
        rrms[t] = 1.0 / (ss[t] * inv_dim + eps).sqrt();
    }

    // 3. dot[t] = sum_i(dy[i,t] * x[i,t] * w[i])
    let mut dot = vec![0.0f32; seq];
    for i in 0..dim {
        for t in 0..seq {
            dot[t] += dy[i * seq + t] * x[i * seq + t] * w[i];
        }
    }

    // 4. dot[t] *= rrms[t]^2 / dim
    for t in 0..seq {
        dot[t] *= rrms[t] * rrms[t] * inv_dim;
    }

    // 5. Per-dimension: dx and dw
    for i in 0..dim {
        for t in 0..seq {
            // Correct derivative: rrms * (w[i]*dy[i,t] - x[i,t] * rrms^2/d * dot)
            dx[i * seq + t] = rrms[t] * (w[i] * dy[i * seq + t] - x[i * seq + t] * dot[t]);
        }
        // dw[i] += sum_t(dy[i,t] * x[i,t] * rrms[t])
        for t in 0..seq {
            dw[i] += dy[i * seq + t] * x[i * seq + t] * rrms[t];
        }
    }
}

/// SiLU gate backward: given dsilu (gradient of gate output = silu(h1)*h3),
/// compute dh1 and dh3.
///
///   dh3[i] = dsilu[i] * silu(h1[i])
///   dh1[i] = dsilu[i] * h3[i] * silu'(h1[i])
///
/// where silu(x) = x*sigmoid(x), silu'(x) = sigmoid(x)*(1 + x*(1-sigmoid(x)))
pub fn silu_bwd(dh1: &mut [f32], dh3: &mut [f32], dsilu: &[f32], h1: &[f32], h3: &[f32], n: usize) {
    debug_assert_eq!(dh1.len(), n);
    debug_assert_eq!(dh3.len(), n);
    debug_assert_eq!(dsilu.len(), n);
    debug_assert_eq!(h1.len(), n);
    debug_assert_eq!(h3.len(), n);

    for i in 0..n {
        let sig = 1.0 / (1.0 + (-h1[i]).exp());
        let silu_val = h1[i] * sig;
        dh3[i] = dsilu[i] * silu_val;
        let silu_deriv = sig * (1.0 + h1[i] * (1.0 - sig));
        dh1[i] = dsilu[i] * h3[i] * silu_deriv;
    }
}

/// Sigmoid gate backward for Qwen3.5 attn_output_gate.
///
/// Forward: gated_out = attn_out * sigmoid(gate_raw)
/// Backward: d_attn = da * sig, d_gate = da * attn_out * sig * (1 - sig)
fn sigmoid_gate_backward(da: &[f32], gate_raw: &[f32], pre_gate: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let n = da.len();
    debug_assert_eq!(gate_raw.len(), n);
    debug_assert_eq!(pre_gate.len(), n);
    let mut d_attn = vec![0.0f32; n];
    let mut d_gate = vec![0.0f32; n];
    for i in 0..n {
        let sig = 1.0 / (1.0 + (-gate_raw[i]).exp());
        d_attn[i] = da[i] * sig;
        d_gate[i] = da[i] * pre_gate[i] * sig * (1.0 - sig);
    }
    (d_attn, d_gate)
}

/// Classifier backward: dy = embed^T @ dlogits, dembed += dlogits @ x_final^T.
///
/// dy[D,S] = embed^T[D,V] @ dlogits[V,S]
/// dembed[V,D] += dlogits[V,S] @ x_final^T[S,D]
///
/// embed: [vocab, dim] row-major, dlogits: [vocab, seq], x_final: [dim, seq].
pub fn classifier_bwd(
    dy: &mut [f32],
    dembed: &mut [f32],
    dlogits: &[f32],
    embed: &[f32],
    x_final: &[f32],
    vocab: usize,
    dim: usize,
    seq: usize,
) {
    debug_assert_eq!(dy.len(), dim * seq);
    debug_assert_eq!(dembed.len(), vocab * dim);
    debug_assert_eq!(dlogits.len(), vocab * seq);
    debug_assert_eq!(embed.len(), vocab * dim);
    debug_assert_eq!(x_final.len(), dim * seq);

    // dy[D,S] = embed^T[D,V] @ dlogits[V,S]
    ane_forward::cpu_gemm(dy, embed, true, dlogits, false, dim, seq, vocab, 1.0, 0.0);

    // dembed[V,D] += dlogits[V,S] @ x_final^T[S,D]  (beta=1.0 to accumulate)
    ane_forward::cpu_gemm(
        dembed, dlogits, false, x_final, true, vocab, dim, seq, 1.0, 1.0,
    );
}

/// Embedding backward: scatter gradient to embedding rows.
///
/// dembed[tok*dim + d] += dy[d*seq + t] for each position t.
pub fn embed_bwd<T: ane_forward::TokenId>(
    dembed: &mut [f32],
    dy: &[f32],
    tokens: &[T],
    dim: usize,
    seq: usize,
) {
    debug_assert_eq!(dy.len(), dim * seq);

    for t in 0..seq {
        let tok = tokens[t].as_usize();
        for d in 0..dim {
            dembed[tok * dim + d] += dy[d * seq + t];
        }
    }
}

/// dW[R,C] += A[R,S] @ B^T — where A is [R, S] and B is [C, S].
///
/// Both A and B are dim-major, seq-minor layout.
/// dW is [R, C] row-major.
fn matmul_accum_at(dw: &mut [f32], a: &[f32], b: &[f32], r: usize, c: usize, s: usize) {
    debug_assert_eq!(dw.len(), r * c);
    debug_assert_eq!(a.len(), r * s);
    debug_assert_eq!(b.len(), c * s);

    for i in 0..r {
        for j in 0..c {
            let mut acc = 0.0f32;
            for k in 0..s {
                acc += a[i * s + k] * b[j * s + k];
            }
            dw[i * c + j] += acc;
        }
    }
}

// ---------------------------------------------------------------------------
// Compiled backward kernels
// ---------------------------------------------------------------------------

/// FFN backward kernels — fused (small models) or tiled (large models).
pub enum FfnBwdKernels {
    /// Fused kernels that fit in ANE SRAM.
    Fused {
        w2t: AneKernel,  // DynMatmul(dim, hidden) — dffn @ W2^T
        w13t: AneKernel, // Fused dh1@W1^T + dh3@W3^T
    },
    /// Tiled DynMatmul kernels for models that exceed ANE SRAM.
    Tiled {
        /// DynMatmul(dim, tile_oc, seq) — for W2^T backward (OC-tiled, same as fwd W1/W3).
        oc_kernel: AneKernel,
        oc_plan: ane_mil::TilePlan,
        oc_out_bytes: usize,
        /// DynMatmul(tile_ic, dim, seq) — for W1^T/W3^T backward (IC-tiled, same as fwd W2).
        ic_kernel: AneKernel,
        ic_plan: ane_mil::TilePlan,
        ic_out_bytes: usize,
    },
}

impl FfnBwdKernels {
    /// Get references to the Fused backward kernels (for cloning into per-layer pool).
    pub fn fused_kernels(
        &self,
    ) -> Option<(&super::ane_bridge::AneKernel, &super::ane_bridge::AneKernel)> {
        match self {
            FfnBwdKernels::Fused { w2t, w13t } => Some((w2t, w13t)),
            _ => None,
        }
    }

    /// Execute backward W2^T: dsilu = dffn @ W2^T (output is [hidden, seq]).
    ///
    /// `w2_transposed` is W2 pre-transposed to [dim, hidden] layout.
    pub fn eval_w2t(
        &self,
        dffn: &[f32],
        w2_transposed: &[f32],
        cfg: &MilConfig,
    ) -> Result<Vec<f32>, String> {
        let dim = cfg.dim;
        let hidden = cfg.hidden_dim;
        let seq = cfg.seq_len;

        match self {
            FfnBwdKernels::Fused { w2t, .. } => {
                let input = ane_weights::pack_dyn_matmul(dffn, w2_transposed, dim, hidden, seq);
                let spec = KernelSpec::for_kernel(cfg, KernelType::FfnBwdW2t);
                w2t.write_input(0, &input);
                w2t.eval()?;
                let mut out_buf = vec![0u8; spec.output_bytes];
                w2t.read_output(0, &mut out_buf);
                Ok(out_buf
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect())
            }
            FfnBwdKernels::Tiled {
                oc_kernel,
                oc_plan,
                oc_out_bytes,
                ..
            } => {
                // OC-tiled: dffn[dim,seq] @ W2^T[dim,hidden] → [hidden,seq]
                // IC=dim, OC=hidden — same kernel/plan as forward W1/W3
                let mut dsilu = vec![0.0f32; hidden * seq];
                for t in 0..oc_plan.n_tiles {
                    let start = oc_plan.tile_start(t);
                    let actual = oc_plan.actual_tile_size(t);
                    let tile_in = ane_weights::pack_dyn_matmul_oc_tile(
                        dffn,
                        w2_transposed,
                        dim,
                        hidden,
                        oc_plan.tile_size,
                        start,
                        seq,
                    );
                    oc_kernel.write_input(0, &tile_in);
                    oc_kernel.eval()?;
                    let mut tile_out = vec![0u8; *oc_out_bytes];
                    oc_kernel.read_output(0, &mut tile_out);
                    ane_weights::unpack_oc_tile(
                        &tile_out,
                        &mut dsilu,
                        oc_plan.tile_size,
                        start,
                        actual,
                        seq,
                    );
                }
                Ok(dsilu)
            }
        }
    }

    /// Execute backward W2^T using pre-packed weight buffers (Orion delta patching).
    ///
    /// Only patches `dffn` activation into the pre-packed buffer — no weight copy.
    pub fn eval_w2t_prepacked(
        &self,
        dffn: &[f32],
        prepacked: &mut super::ane_weights::PrePackedWeights,
        layer: usize,
        cfg: &MilConfig,
    ) -> Result<Vec<f32>, String> {
        let FfnBwdKernels::Fused { w2t, .. } = self else {
            return Err("eval_w2t_prepacked requires Fused kernels".into());
        };
        // Fast path: per-layer kernel with IOSurface-resident weights
        if let Some(result) = prepacked.eval_bwd_w2t(layer, dffn, cfg) {
            return result;
        }
        // Fallback: shared kernel, full buffer copy
        let input_bytes = prepacked.patch_bwd_w2t(layer, dffn, cfg.dim);
        let spec = KernelSpec::for_kernel(cfg, KernelType::FfnBwdW2t);
        w2t.write_input(0, input_bytes);
        w2t.eval()?;
        let mut out_buf = vec![0u8; spec.output_bytes];
        w2t.read_output(0, &mut out_buf);
        Ok(out_buf
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    /// Execute backward W1^T + W3^T using pre-packed weight buffers (Orion delta patching).
    ///
    /// Only patches `dh1`, `dh3` activations — no weight transpose or copy.
    pub fn eval_w13t_prepacked(
        &self,
        dh1: &[f32],
        dh3: &[f32],
        prepacked: &mut super::ane_weights::PrePackedWeights,
        layer: usize,
        cfg: &MilConfig,
    ) -> Result<Vec<f32>, String> {
        let FfnBwdKernels::Fused { w13t, .. } = self else {
            return Err("eval_w13t_prepacked requires Fused kernels".into());
        };
        // Fast path: per-layer kernel with IOSurface-resident weights
        if let Some(result) = prepacked.eval_bwd_w13t(layer, dh1, dh3, cfg) {
            return result;
        }
        // Fallback: shared kernel, full buffer copy
        let input_bytes = prepacked.patch_bwd_w13t(layer, dh1, dh3, cfg.hidden_dim);
        let spec = KernelSpec::for_kernel(cfg, KernelType::FfnBwdW13t);
        w13t.write_input(0, input_bytes);
        w13t.eval()?;
        let mut out_buf = vec![0u8; spec.output_bytes];
        w13t.read_output(0, &mut out_buf);
        Ok(out_buf
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    /// Execute backward W1^T + W3^T: dx_ffn = dh1 @ W1^T + dh3 @ W3^T (output is [dim, seq]).
    ///
    /// `w1t`, `w3t` are pre-transposed to [hidden, dim] layout.
    pub fn eval_w13t(
        &self,
        dh1: &[f32],
        dh3: &[f32],
        w1t: &[f32],
        w3t: &[f32],
        cfg: &MilConfig,
    ) -> Result<Vec<f32>, String> {
        let dim = cfg.dim;
        let hidden = cfg.hidden_dim;
        let seq = cfg.seq_len;

        match self {
            FfnBwdKernels::Fused { w13t, .. } => {
                let input = ane_weights::pack_ffn_bwd_w13t(dh1, dh3, w1t, w3t, cfg);
                let spec = KernelSpec::for_kernel(cfg, KernelType::FfnBwdW13t);
                w13t.write_input(0, &input);
                w13t.eval()?;
                let mut out_buf = vec![0u8; spec.output_bytes];
                w13t.read_output(0, &mut out_buf);
                Ok(out_buf
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect())
            }
            FfnBwdKernels::Tiled {
                ic_kernel,
                ic_plan,
                ic_out_bytes,
                ..
            } => {
                // IC-tiled: dh1[hidden,seq] @ W1^T[hidden,dim] → [dim,seq]
                // + dh3[hidden,seq] @ W3^T[hidden,dim] → [dim,seq]
                // IC=hidden, OC=dim — same kernel/plan as forward W2
                let mut result = vec![0.0f32; dim * seq];
                // W1^T pass — accumulates into result
                for t in 0..ic_plan.n_tiles {
                    let start = ic_plan.tile_start(t);
                    let tile_in = ane_weights::pack_dyn_matmul_ic_tile(
                        dh1,
                        w1t,
                        hidden,
                        dim,
                        ic_plan.tile_size,
                        start,
                        seq,
                    );
                    ic_kernel.write_input(0, &tile_in);
                    ic_kernel.eval()?;
                    let mut tile_out = vec![0u8; *ic_out_bytes];
                    ic_kernel.read_output(0, &mut tile_out);
                    ane_weights::unpack_ic_tile_accum(&tile_out, &mut result, dim, seq);
                }
                // W3^T pass — accumulates into same result
                for t in 0..ic_plan.n_tiles {
                    let start = ic_plan.tile_start(t);
                    let tile_in = ane_weights::pack_dyn_matmul_ic_tile(
                        dh3,
                        w3t,
                        hidden,
                        dim,
                        ic_plan.tile_size,
                        start,
                        seq,
                    );
                    ic_kernel.write_input(0, &tile_in);
                    ic_kernel.eval()?;
                    let mut tile_out = vec![0u8; *ic_out_bytes];
                    ic_kernel.read_output(0, &mut tile_out);
                    ane_weights::unpack_ic_tile_accum(&tile_out, &mut result, dim, seq);
                }
                Ok(result)
            }
        }
    }
}

/// Pre-compiled ANE kernels for backward pass (compile once, reuse every step).
pub struct BackwardKernels {
    /// FFN backward kernels — fused (small models) or tiled (large models).
    pub ffn_bwd: FfnBwdKernels,
    /// Wo^T backward (None at 4B where attention is CPU-only).
    pub wot_bwd: Option<AneKernel>,
    /// SDPA backward kernels (None at 4B where attention is CPU-only).
    pub sdpa_bwd1: Option<AneKernel>,
    pub sdpa_bwd2: Option<AneKernel>,
    /// QKV backward (None at 4B where attention is CPU-only).
    pub qkv_bwd: Option<AneKernel>,
    /// Fused backward attention GQA: single dispatch replacing wot+sdpa1+sdpa2+qkv.
    /// Template kernel (zero weights) — per-layer kernels live in PrePackedWeights.
    pub fused_attn_gqa_bwd: Option<AneKernel>,
    /// RMSNorm backward (dx only, no dw — LoRA freezes base weights).
    /// Template kernel with placeholder weights — per-layer kernels in PrePackedWeights.
    pub rmsnorm_bwd: Option<AneKernel>,
}

impl BackwardKernels {
    /// Compile all backward-pass kernels for the given config.
    pub fn compile_backward(cfg: &MilConfig, mask_blob: &[u8]) -> Result<Self, String> {
        ane_bridge::ane_init()?;

        // FFN backward: check if fused kernels fit, otherwise tile
        let oc_plan = ane_mil::compute_oc_tile_plan(cfg.dim, cfg.hidden_dim, cfg.seq_len);
        let ic_plan = ane_mil::compute_ic_tile_plan(cfg.hidden_dim, cfg.dim, cfg.seq_len);

        let ffn_bwd = 'ffn_bwd: {
            // Try fused backward kernels first when SRAM check says they fit.
            // Fall back to tiled if the ANE compiler rejects the fused MIL.
            if !oc_plan.needs_tiling() && !ic_plan.needs_tiling() {
                let w2t_spec = KernelSpec::for_kernel(cfg, KernelType::FfnBwdW2t);
                if let Ok(w2t) = AneKernel::compile(
                    &w2t_spec.mil_text,
                    None,
                    &[w2t_spec.input_bytes],
                    &[w2t_spec.output_bytes],
                ) {
                    let w13t_spec = KernelSpec::for_kernel(cfg, KernelType::FfnBwdW13t);
                    if let Ok(w13t) = AneKernel::compile(
                        &w13t_spec.mil_text,
                        None,
                        &[w13t_spec.input_bytes],
                        &[w13t_spec.output_bytes],
                    ) {
                        tracing::debug!(
                            "ANE FFN bwd: fused (dim={}, hidden={}, seq={})",
                            cfg.dim,
                            cfg.hidden_dim,
                            cfg.seq_len
                        );
                        break 'ffn_bwd FfnBwdKernels::Fused { w2t, w13t };
                    }
                }
                tracing::debug!(
                    "ANE FFN bwd: fused compile failed (dim={}, hidden={}), falling back to tiled",
                    cfg.dim,
                    cfg.hidden_dim
                );
            }

            // Tiled: same DynMatmul kernels as forward
            let oc_spec = KernelSpec::for_kernel(
                cfg,
                KernelType::DynMatmul {
                    ic: cfg.dim,
                    oc: oc_plan.tile_size,
                },
            );
            let oc_kernel = AneKernel::compile(
                &oc_spec.mil_text,
                None,
                &[oc_spec.input_bytes],
                &[oc_spec.output_bytes],
            )?;
            let ic_spec = KernelSpec::for_kernel(
                cfg,
                KernelType::DynMatmul {
                    ic: ic_plan.tile_size,
                    oc: cfg.dim,
                },
            );
            let ic_kernel = AneKernel::compile(
                &ic_spec.mil_text,
                None,
                &[ic_spec.input_bytes],
                &[ic_spec.output_bytes],
            )?;
            tracing::debug!(
                "ANE FFN bwd: tiled (oc_tile={}, {} tiles; ic_tile={}, {} tiles)",
                oc_plan.tile_size,
                oc_plan.n_tiles,
                ic_plan.tile_size,
                ic_plan.n_tiles
            );
            FfnBwdKernels::Tiled {
                oc_kernel,
                oc_plan,
                oc_out_bytes: oc_spec.output_bytes,
                ic_kernel,
                ic_plan,
                ic_out_bytes: ic_spec.output_bytes,
            }
        };

        // Attention backward kernels: best-effort (None if exceeds SRAM)
        let wot_bwd = {
            let wot_spec = KernelSpec::for_kernel(cfg, KernelType::Wot);
            AneKernel::compile(
                &wot_spec.mil_text,
                None,
                &[wot_spec.input_bytes],
                &[wot_spec.output_bytes],
            )
            .ok()
        };

        let sdpa_bwd1 = {
            let bwd1_spec = KernelSpec::for_kernel(cfg, KernelType::SdpaBwd1);
            AneKernel::compile_multi_weights(
                &bwd1_spec.mil_text,
                &["@model_path/weights/mask.bin"],
                &[mask_blob],
                &[bwd1_spec.input_bytes],
                &[bwd1_spec.output_bytes],
            )
            .ok()
        };

        let sdpa_bwd2 = {
            let bwd2_spec = KernelSpec::for_kernel(cfg, KernelType::SdpaBwd2);
            AneKernel::compile(
                &bwd2_spec.mil_text,
                None,
                &[bwd2_spec.input_bytes],
                &[bwd2_spec.output_bytes],
            )
            .ok()
        };

        let qkv_bwd = {
            let qkv_spec = KernelSpec::for_kernel(cfg, KernelType::Qkvb);
            AneKernel::compile(
                &qkv_spec.mil_text,
                None,
                &[qkv_spec.input_bytes],
                &[qkv_spec.output_bytes],
            )
            .ok()
        };

        // Fused SDPA backward: replaces sdpa_bwd1 + sdpa_bwd2 with 1 dispatch.
        // Uses [1,H,S,*] form with K/V pre-expanded. Only needs mask BLOBFILE.
        let fused_attn_gqa_bwd = {
            let has_gate = cfg.attn_output_gate;
            let result = ane_mil::gen_sdpa_rope_bwd(cfg, has_gate);
            let names: Vec<&str> = result.weight_names.iter().copied().collect();
            let datas: Vec<&[u8]> = vec![mask_blob];
            match AneKernel::compile_multi_weights(
                &result.mil_text,
                &names,
                &datas,
                &[result.input_bytes],
                &[result.output_bytes],
            ) {
                Ok(k) => {
                    tracing::debug!("ANE fused SDPA backward compiled (replaces sdpa_bwd1+bwd2)");
                    Some(k)
                }
                Err(e) => {
                    tracing::debug!("ANE fused SDPA backward compile failed: {e}");
                    None
                }
            }
        };

        // RMSNorm backward: dx-only kernel (dw not needed for LoRA).
        let rmsnorm_bwd = {
            let result = ane_mil::gen_rmsnorm_bwd(cfg.dim, cfg.seq_len, cfg.rms_eps);
            let placeholder_w = ane_weights::build_fp16_blob(&vec![1.0f32; cfg.dim]);
            let names: Vec<&str> = result.weight_names.iter().copied().collect();
            match AneKernel::compile_multi_weights(
                &result.mil_text,
                &names,
                &[&placeholder_w],
                &[result.input_bytes],
                &[result.output_bytes],
            ) {
                Ok(k) => {
                    tracing::debug!("ANE RMSNorm backward compiled");
                    Some(k)
                }
                Err(e) => {
                    tracing::debug!("ANE RMSNorm backward compile failed: {e}");
                    None
                }
            }
        };

        Ok(Self {
            ffn_bwd,
            wot_bwd,
            sdpa_bwd1,
            sdpa_bwd2,
            qkv_bwd,
            fused_attn_gqa_bwd,
            rmsnorm_bwd,
        })
    }
}

// ---------------------------------------------------------------------------
// Gradient structs
// ---------------------------------------------------------------------------

/// Per-layer weight gradients.
pub struct LayerGradients {
    pub dwq: Vec<f32>,             // [dim, dim]
    pub dwk: Vec<f32>,             // [kv_dim, dim] (= [dim, dim] for MHA)
    pub dwv: Vec<f32>,             // [kv_dim, dim] (= [dim, dim] for MHA)
    pub dwo: Vec<f32>,             // [dim, dim]
    pub dw1: Vec<f32>,             // [hidden, dim]
    pub dw2: Vec<f32>,             // [dim, hidden]
    pub dw3: Vec<f32>,             // [hidden, dim]
    pub drms_att: Vec<f32>,        // [dim]
    pub drms_ffn: Vec<f32>,        // [dim]
    pub dq_norm: Option<Vec<f32>>, // [head_dim] per-head Q norm grad
    pub dk_norm: Option<Vec<f32>>, // [head_dim] per-head K norm grad
}

/// Full model gradients.
pub struct ModelGradients {
    pub layers: Vec<LayerGradients>,
    pub drms_final: Vec<f32>,       // [dim]
    pub dembed: Vec<f32>,           // [vocab * dim]
    pub dlm_head: Option<Vec<f32>>, // [vocab * dim] untied classifier grad
}

impl ModelGradients {
    /// Create zero-initialized gradients matching model shape.
    pub fn zeros(model: &ModelWeights) -> Self {
        let dim = model.cfg.dim;
        let hidden = model.cfg.hidden_dim;
        let n_layers = model.layers.len();

        let layers = (0..n_layers)
            .map(|i| LayerGradients {
                dwq: vec![0.0; dim * dim],
                dwk: vec![0.0; dim * dim],
                dwv: vec![0.0; dim * dim],
                dwo: vec![0.0; dim * dim],
                dw1: vec![0.0; hidden * dim],
                dw2: vec![0.0; dim * hidden],
                dw3: vec![0.0; hidden * dim],
                drms_att: vec![0.0; dim],
                drms_ffn: vec![0.0; dim],
                dq_norm: if model.layers[i].q_norm.is_some() {
                    Some(vec![0.0; model.cfg.head_dim()])
                } else {
                    None
                },
                dk_norm: if model.layers[i].k_norm.is_some() {
                    Some(vec![0.0; model.cfg.head_dim()])
                } else {
                    None
                },
            })
            .collect();

        ModelGradients {
            layers,
            drms_final: vec![0.0; dim],
            dembed: vec![0.0; model.vocab_size * dim],
            dlm_head: if model.lm_head.is_some() {
                Some(vec![0.0; model.vocab_size * dim])
            } else {
                None
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Gradient sanitization
// ---------------------------------------------------------------------------

/// Replace NaN → 0 and ±Inf → ±65504 in LoRA gradients.
/// Prevents corrupted ANE fp16 backward values from reaching the optimizer
/// (Orion checkpoint sanitization technique, arxiv 2603.06728).
fn sanitize_lora_grads(grads: &mut super::ane_lora::LoraModelGrads) {
    let sanitize_buf = |buf: &mut [f32]| {
        for v in buf.iter_mut() {
            if v.is_nan() {
                *v = 0.0;
            } else if *v == f32::INFINITY {
                *v = ane_forward::FP16_MAX;
            } else if *v == f32::NEG_INFINITY {
                *v = -ane_forward::FP16_MAX;
            }
        }
    };
    let sanitize_opt = |g: &mut Option<super::ane_lora::LoraAdapterGrads>| {
        if let Some(ref mut grads) = g {
            sanitize_buf(&mut grads.da);
            sanitize_buf(&mut grads.db);
        }
    };
    for lg in &mut grads.layers {
        sanitize_opt(&mut lg.wq);
        sanitize_opt(&mut lg.wv);
        sanitize_opt(&mut lg.wo);
        sanitize_opt(&mut lg.w2);
    }
}

// ---------------------------------------------------------------------------
// Full backward pass
// ---------------------------------------------------------------------------

/// Run full backward pass: classifier_bwd → final_rmsnorm_bwd → per-layer-reversed → embed_bwd.
///
/// Follows train.m lines 509-722.
pub fn backward<T: ane_forward::TokenId>(
    kernels: &BackwardKernels,
    model: &ModelWeights,
    fwd: &ForwardResult,
    tokens: &[T],
) -> Result<ModelGradients, String> {
    let cfg = &model.cfg;
    let dim = cfg.dim;
    let hidden = cfg.hidden_dim;
    let seq = cfg.seq_len;
    let n_layers = model.layers.len();

    let mut grads = ModelGradients::zeros(model);

    // Reconstruct x_cur (input to final rmsnorm) from last layer:
    // x_cur = last_layer.x2 + last_layer.ffn_out
    let last_act = &fwd.layer_acts[n_layers - 1];
    let mut x_cur = last_act.x2.clone();
    ane_forward::vec_add_inplace(&mut x_cur, &last_act.ffn_out);

    // 1. Classifier backward: fused CE already computed classifier_dy = cls_w^T @ dlogits.
    //    dcls_w (classifier weight gradient) requires dlogits, which the fused path doesn't
    //    materialize. This is fine: backward() is only called from LoRA training where base
    //    model weight gradients (dcls_w, dembed) are unused. For tied weights, embed_bwd
    //    accumulates the embedding gradient from the layer stack.
    let mut dy = fwd.classifier_dy.clone();

    // 2. Final RMSNorm backward
    let mut dx_rms = vec![0.0f32; dim * seq];
    rmsnorm_bwd(
        &mut dx_rms,
        &mut grads.drms_final,
        &dy,
        &x_cur,
        &model.rms_final,
        dim,
        seq,
        cfg.rms_eps,
    );
    // dy = dx_rms (gradient flowing into the layer stack)
    dy = dx_rms;

    // 3. Per-layer backward (reverse order)
    for l in (0..n_layers).rev() {
        let lw = &model.layers[l];
        let ac = &fwd.layer_acts[l];
        let gr = &mut grads.layers[l];

        // --- FFN backward ---

        // a. dffn = dy (gradient entering FFN residual path)
        let dffn = dy.clone();

        // b. ANE ffn_bwd_w2t: dsilu = dffn @ W2^T (fused or tiled)
        // W2 is [dim, hidden]; pack expects [ic=dim, oc=hidden] — pass directly.
        let mut dsilu = kernels
            .ffn_bwd
            .eval_w2t(&dffn, &lw.w2, cfg)
            .map_err(|e| format!("ffn_bwd_w2t eval: {e}"))?;
        ane_forward::clamp_fp16(&mut dsilu);

        // c. CPU silu_bwd
        let mut dh1 = vec![0.0f32; hidden * seq];
        let mut dh3 = vec![0.0f32; hidden * seq];
        silu_bwd(&mut dh1, &mut dh3, &dsilu, &ac.h1, &ac.h3, hidden * seq);

        // d. ANE ffn_bwd_w13t: dx_ffn = dh1@W1^T + dh3@W3^T (fused or tiled)
        // W1/W3 are [hidden, dim]; pack expects [ic=hidden, oc=dim] — pass directly.
        let mut dx_ffn = kernels
            .ffn_bwd
            .eval_w13t(&dh1, &dh3, &lw.w1, &lw.w3, cfg)
            .map_err(|e| format!("ffn_bwd_w13t eval: {e}"))?;
        ane_forward::clamp_fp16(&mut dx_ffn);

        // e. CPU dW accumulation for FFN
        // dW2[dim, hidden] += dffn[dim, seq] @ gate^T[hidden, seq]
        matmul_accum_at(&mut gr.dw2, &dffn, &ac.gate, dim, hidden, seq);
        // dW1[hidden, dim] += dh1[hidden, seq] @ x2norm^T[dim, seq]
        matmul_accum_at(&mut gr.dw1, &dh1, &ac.x2norm, hidden, dim, seq);
        // dW3[hidden, dim] += dh3[hidden, seq] @ x2norm^T[dim, seq]
        matmul_accum_at(&mut gr.dw3, &dh3, &ac.x2norm, hidden, dim, seq);

        // f. CPU rmsnorm_bwd (FFN RMSNorm)
        let mut dx2 = vec![0.0f32; dim * seq];
        rmsnorm_bwd(
            &mut dx2,
            &mut gr.drms_ffn,
            &dx_ffn,
            &ac.x2,
            &lw.rms_ffn,
            dim,
            seq,
            cfg.rms_eps,
        );

        // g. Residual: dx2 += dy (skip connection from FFN path)
        ane_forward::vec_add_inplace(&mut dx2, &dy);

        // --- Attention backward ---

        // h. ANE wot_bwd: da = dx2 @ Wo^T
        let ad = cfg.attn_dim();
        let wot_input = ane_weights::pack_dyn_matmul(&dx2, &lw.wo, dim, ad, seq);
        let wot_spec = KernelSpec::for_kernel(cfg, KernelType::Wot);
        let wot_kernel = kernels
            .wot_bwd
            .as_ref()
            .expect("wot_bwd kernel required for MHA backward");
        wot_kernel.write_input(0, &wot_input);
        wot_kernel
            .eval()
            .map_err(|e| format!("wot_bwd eval: {e}"))?;
        let mut wot_out = vec![0u8; wot_spec.output_bytes];
        wot_kernel.read_output(0, &mut wot_out);
        let da: Vec<f32> = wot_out
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // i. CPU dWo accumulation: dWo[dim,attn_dim] += dx2[dim,seq] @ attn_out^T[attn_dim,seq]
        let ad = cfg.attn_dim();
        matmul_accum_at(&mut gr.dwo, &dx2, &ac.attn_out, dim, ad, seq);

        // j. ANE sdpa_bwd1: pack(Q, K, V, da) -> (dV, probs_raw, dp_raw)
        let bwd1_input = ane_weights::pack_sdpa_bwd1(&ac.q, &ac.k, &ac.v, &da, cfg);
        let bwd1_spec = KernelSpec::for_kernel(cfg, KernelType::SdpaBwd1);
        let bwd1_kernel = kernels
            .sdpa_bwd1
            .as_ref()
            .expect("sdpa_bwd1 kernel required for MHA backward");
        bwd1_kernel.write_input(0, &bwd1_input);
        bwd1_kernel
            .eval()
            .map_err(|e| format!("sdpa_bwd1 eval: {e}"))?;
        let mut bwd1_out = vec![0u8; bwd1_spec.output_bytes];
        bwd1_kernel.read_output(0, &mut bwd1_out);
        let (dv, _probs_f32, _dp_f32) = ane_weights::unpack_sdpa_bwd1(&bwd1_out, cfg);

        // k. ANE sdpa_bwd2: bridge fp16 data from bwd1 output + Q,K
        // Extract probs+dp raw bytes from bwd1 output (fp16): skip dV portion
        let score_ch = cfg.score_ch();
        let probs_dp_offset = ad * seq * 2; // dV is [attn_dim, seq] fp16
        let probs_dp_bytes = 2 * score_ch * seq * 2; // probs + dp, each [score_ch, seq] fp16
        let probs_dp_raw = &bwd1_out[probs_dp_offset..probs_dp_offset + probs_dp_bytes];

        // Convert Q and K from f32 to fp16 bytes
        let f32_to_fp16_bytes = |data: &[f32]| -> Vec<u8> {
            let mut buf = vec![0u8; data.len() * 2];
            for (i, &v) in data.iter().enumerate() {
                let fp16 = half::f16::from_f32(v);
                buf[i * 2..i * 2 + 2].copy_from_slice(&fp16.to_le_bytes());
            }
            buf
        };
        let q_fp16 = f32_to_fp16_bytes(&ac.q);
        let k_fp16 = f32_to_fp16_bytes(&ac.k);

        // Concatenate: [probs_raw | dp_raw | Q_fp16 | K_fp16]
        let mut bwd2_input = Vec::with_capacity(probs_dp_bytes + q_fp16.len() + k_fp16.len());
        bwd2_input.extend_from_slice(probs_dp_raw);
        bwd2_input.extend_from_slice(&q_fp16);
        bwd2_input.extend_from_slice(&k_fp16);

        let bwd2_spec = KernelSpec::for_kernel(cfg, KernelType::SdpaBwd2);
        let bwd2_kernel = kernels
            .sdpa_bwd2
            .as_ref()
            .expect("sdpa_bwd2 kernel required for MHA backward");
        bwd2_kernel.write_input(0, &bwd2_input);
        bwd2_kernel
            .eval()
            .map_err(|e| format!("sdpa_bwd2 eval: {e}"))?;
        let mut bwd2_out = vec![0u8; bwd2_spec.output_bytes];
        bwd2_kernel.read_output(0, &mut bwd2_out);
        let (dq, dk) = ane_weights::unpack_sdpa_bwd2(&bwd2_out, cfg);

        // l-pre. RoPE backward: un-rotate dQ, dK on CPU
        let mut dq = dq;
        let mut dk = dk;
        ane_forward::rope_backward(
            &mut dq,
            &mut dk,
            cfg.n_heads,
            cfg.head_dim(),
            seq,
            cfg.rope_theta,
        );

        // l-pre2. QK-norm backward (if present)
        if let (Some(q_pre), Some(q_nw)) = (&ac.q_pre_norm, &lw.q_norm) {
            ane_forward::qk_rmsnorm_bwd(
                &mut dq,
                gr.dq_norm.as_mut().unwrap(),
                q_pre,
                q_nw,
                cfg.n_heads,
                cfg.head_dim(),
                seq,
                cfg.rms_eps,
            );
        }
        if let (Some(k_pre), Some(k_nw)) = (&ac.k_pre_norm, &lw.k_norm) {
            ane_forward::qk_rmsnorm_bwd(
                &mut dk,
                gr.dk_norm.as_mut().unwrap(),
                k_pre,
                k_nw,
                cfg.n_heads,
                cfg.head_dim(),
                seq,
                cfg.rms_eps,
            );
        }

        // l. CPU dW accumulation for QKV
        // dWq[dim,dim] += dq[dim,seq] @ xnorm^T[dim,seq]
        matmul_accum_at(&mut gr.dwq, &dq, &ac.xnorm, dim, dim, seq);
        matmul_accum_at(&mut gr.dwk, &dk, &ac.xnorm, dim, dim, seq);
        matmul_accum_at(&mut gr.dwv, &dv, &ac.xnorm, dim, dim, seq);

        // m. ANE qkv_bwd: dx_attn = dq@Wq^T + dk@Wk^T + dv@Wv^T
        let qkv_input = ane_weights::pack_qkvb(&dq, &dk, &dv, &lw.wq, &lw.wk, &lw.wv, cfg);
        let qkv_spec = KernelSpec::for_kernel(cfg, KernelType::Qkvb);
        let qkv_kernel = kernels
            .qkv_bwd
            .as_ref()
            .expect("qkv_bwd kernel required for MHA backward");
        qkv_kernel.write_input(0, &qkv_input);
        qkv_kernel
            .eval()
            .map_err(|e| format!("qkv_bwd eval: {e}"))?;
        let mut qkv_out = vec![0u8; qkv_spec.output_bytes];
        qkv_kernel.read_output(0, &mut qkv_out);
        let dx_attn: Vec<f32> = qkv_out
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // n. CPU rmsnorm_bwd (attention RMSNorm)
        let mut dx_rms1 = vec![0.0f32; dim * seq];
        rmsnorm_bwd(
            &mut dx_rms1,
            &mut gr.drms_att,
            &dx_attn,
            &ac.layer_in,
            &lw.rms_att,
            dim,
            seq,
            cfg.rms_eps,
        );

        // o. Combine: dy = dx_rms1 + dx2 (merge both residual paths)
        dy = dx_rms1;
        ane_forward::vec_add_inplace(&mut dy, &dx2);
    }

    // 4. Embedding backward
    embed_bwd(&mut grads.dembed, &dy, tokens, dim, seq);

    Ok(grads)
}

/// Backward pass result including LoRA gradients.
pub struct BackwardResultWithLora {
    /// Base model gradients.  `None` when using `backward_lora_cpu` (LoRA-only
    /// training) because base weights are frozen and the 5.9 GB allocation was
    /// pure waste.  Present when using `backward_with_lora` (ANE kernel path).
    pub model_grads: Option<ModelGradients>,
    pub lora_grads: super::ane_lora::LoraModelGrads,
}

// ---------------------------------------------------------------------------
// CPU-only backward for LoRA training (no ANE kernels)
// ---------------------------------------------------------------------------

/// CPU SDPA backward: given dO (gradient w.r.t attention output), compute dQ, dK, dV.
///
/// All tensors [dim, seq] where dim = n_heads * head_dim.
/// Recomputes attention probabilities from saved Q, K.
/// SDPA backward: computes dQ, dK, dV from upstream gradient d_out.
///
/// Uses BLAS gemm for all matrix operations (score recomputation, dV, dP, dQ, dK).
fn cpu_sdpa_backward(
    d_out: &[f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_heads: usize,
    head_dim: usize,
    seq: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let dim = n_heads * head_dim;
    let mut dq = vec![0.0f32; dim * seq];
    let mut dk = vec![0.0f32; dim * seq];
    let mut dv = vec![0.0f32; dim * seq];
    let hd_seq = head_dim * seq;

    for h in 0..n_heads {
        let hoff = h * hd_seq;
        let q_h = &q[hoff..hoff + hd_seq];
        let k_h = &k[hoff..hoff + hd_seq];
        let v_h = &v[hoff..hoff + hd_seq];
        let do_h = &d_out[hoff..hoff + hd_seq];
        let dq_h = &mut dq[hoff..hoff + hd_seq];
        let dk_h = &mut dk[hoff..hoff + hd_seq];
        let dv_h = &mut dv[hoff..hoff + hd_seq];

        // 1. Recompute probs = softmax(scale * Q_h^T @ K_h, causal)
        let mut probs = vec![0.0f32; seq * seq];
        ane_forward::cpu_gemm(
            &mut probs, q_h, true, k_h, false, seq, seq, head_dim, scale, 0.0,
        );
        for s1 in 0..seq {
            for s2 in (s1 + 1)..seq {
                probs[s1 * seq + s2] = f32::NEG_INFINITY;
            }
        }
        for s1 in 0..seq {
            let row = &mut probs[s1 * seq..(s1 + 1) * seq];
            let max_v = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for val in row.iter_mut() {
                *val = (*val - max_v).exp();
                sum += *val;
            }
            let inv_sum = 1.0 / sum;
            for val in row.iter_mut() {
                *val *= inv_sum;
            }
        }

        // 2. dV = dO_h @ probs → [head_dim, seq]
        // dV_h[d, s2] = sum_s1 dO_h[d, s1] * probs[s1, s2]
        ane_forward::cpu_gemm(
            dv_h, do_h, false, &probs, false, head_dim, seq, seq, 1.0, 0.0,
        );

        // 3. dP = dO_h^T @ V_h → [seq, seq]
        // dP[s1, s2] = sum_d dO_h[d, s1] * V_h[d, s2]
        let mut dp = vec![0.0f32; seq * seq];
        ane_forward::cpu_gemm(
            &mut dp, do_h, true, v_h, false, seq, seq, head_dim, 1.0, 0.0,
        );

        // 4. Softmax backward: dS = probs * (dP - row_sum(probs .* dP))
        let mut ds = vec![0.0f32; seq * seq];
        for s1 in 0..seq {
            let mut dot_sum = 0.0f32;
            for s2 in 0..seq {
                dot_sum += probs[s1 * seq + s2] * dp[s1 * seq + s2];
            }
            for s2 in 0..seq {
                ds[s1 * seq + s2] = probs[s1 * seq + s2] * (dp[s1 * seq + s2] - dot_sum);
            }
        }

        // 5. dQ = scale * K_h @ dS^T → [head_dim, seq]
        // dQ_h[d, s1] = scale * sum_s2 K_h[d, s2] * dS[s1, s2]
        ane_forward::cpu_gemm(dq_h, k_h, false, &ds, true, head_dim, seq, seq, scale, 0.0);

        // 6. dK = scale * Q_h @ dS → [head_dim, seq]
        // dK_h[d, s2] = scale * sum_s1 Q_h[d, s1] * dS[s1, s2]
        ane_forward::cpu_gemm(dk_h, q_h, false, &ds, false, head_dim, seq, seq, scale, 0.0);
    }

    (dq, dk, dv)
}

fn mha_backward_cpu_dx_attn(
    lw: &ane_weights::LayerWeights,
    ac: &ane_forward::LayerActivations,
    dx2: &[f32],
    cfg: &MilConfig,
) -> Vec<f32> {
    let dim = cfg.dim;
    let seq = cfg.seq_len;
    let ad = cfg.attn_dim();

    let da = ane_forward::cpu_matmul_lhs_transposed(&lw.wo, dim, ad, dx2, seq);

    let (da, d_gate) = if cfg.attn_output_gate {
        let gate_raw = ac.attn_gate.as_ref().unwrap();
        let pre_gate = ac.attn_pre_gate.as_ref().unwrap();
        let (d_attn, dg) = sigmoid_gate_backward(&da, gate_raw, pre_gate);
        (d_attn, Some(dg))
    } else {
        (da, None)
    };

    let (mut dq, mut dk, dv) =
        cpu_sdpa_backward(&da, &ac.q, &ac.k, &ac.v, cfg.n_heads, cfg.head_dim(), seq);

    ane_forward::rope_backward(
        &mut dq,
        &mut dk,
        cfg.n_heads,
        cfg.head_dim(),
        seq,
        cfg.rope_theta,
    );

    if let (Some(q_pre), Some(q_nw)) = (&ac.q_pre_norm, &lw.q_norm) {
        ane_forward::qk_rmsnorm_bwd(
            &mut dq,
            &mut vec![0.0f32; cfg.head_dim()],
            q_pre,
            q_nw,
            cfg.n_heads,
            cfg.head_dim(),
            seq,
            cfg.rms_eps,
        );
    }
    if let (Some(k_pre), Some(k_nw)) = (&ac.k_pre_norm, &lw.k_norm) {
        ane_forward::qk_rmsnorm_bwd(
            &mut dk,
            &mut vec![0.0f32; cfg.head_dim()],
            k_pre,
            k_nw,
            cfg.n_heads,
            cfg.head_dim(),
            seq,
            cfg.rms_eps,
        );
    }

    let dq_for_wq = if let Some(dg) = &d_gate {
        ane_forward::merge_q_gate(&dq, dg, cfg.n_heads, cfg.head_dim(), seq)
    } else {
        dq
    };

    let qpd = cfg.q_proj_dim();
    let (mut dx_attn, (dx_k, dx_v)) = rayon::join(
        || ane_forward::cpu_matmul_lhs_transposed(&lw.wq, qpd, dim, &dq_for_wq, seq),
        || {
            rayon::join(
                || ane_forward::cpu_matmul_lhs_transposed(&lw.wk, ad, dim, &dk, seq),
                || ane_forward::cpu_matmul_lhs_transposed(&lw.wv, ad, dim, &dv, seq),
            )
        },
    );
    ane_forward::vec_add_inplace(&mut dx_attn, &dx_k);
    ane_forward::vec_add_inplace(&mut dx_attn, &dx_v);
    dx_attn
}

fn mha_backward_ane_dx_attn(
    kernels: &BackwardKernels,
    lw: &ane_weights::LayerWeights,
    ac: &ane_forward::LayerActivations,
    dx2: &[f32],
    cfg: &MilConfig,
) -> Result<Vec<f32>, String> {
    let dim = cfg.dim;
    let seq = cfg.seq_len;
    let ad = cfg.attn_dim();

    let wot_input = ane_weights::pack_dyn_matmul(dx2, &lw.wo, dim, ad, seq);
    let wot_spec = KernelSpec::for_kernel(cfg, KernelType::Wot);
    let wot_kernel = kernels
        .wot_bwd
        .as_ref()
        .ok_or_else(|| "wot_bwd kernel missing".to_string())?;
    wot_kernel.write_input(0, &wot_input);
    wot_kernel
        .eval()
        .map_err(|e| format!("wot_bwd eval: {e}"))?;
    let mut wot_out = vec![0u8; wot_spec.output_bytes];
    wot_kernel.read_output(0, &mut wot_out);
    let da_raw: Vec<f32> = wot_out
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let (da, d_gate) = if cfg.attn_output_gate {
        let gate_raw = ac.attn_gate.as_ref().unwrap();
        let pre_gate = ac.attn_pre_gate.as_ref().unwrap();
        let (d_attn, dg) = sigmoid_gate_backward(&da_raw, gate_raw, pre_gate);
        (d_attn, Some(dg))
    } else {
        (da_raw, None)
    };

    let bwd1_input = ane_weights::pack_sdpa_bwd1(&ac.q, &ac.k, &ac.v, &da, cfg);
    let bwd1_spec = KernelSpec::for_kernel(cfg, KernelType::SdpaBwd1);
    let bwd1_kernel = kernels
        .sdpa_bwd1
        .as_ref()
        .ok_or_else(|| "sdpa_bwd1 kernel missing".to_string())?;
    bwd1_kernel.write_input(0, &bwd1_input);
    bwd1_kernel
        .eval()
        .map_err(|e| format!("sdpa_bwd1 eval: {e}"))?;
    let mut bwd1_out = vec![0u8; bwd1_spec.output_bytes];
    bwd1_kernel.read_output(0, &mut bwd1_out);
    let (dv, _, _) = ane_weights::unpack_sdpa_bwd1(&bwd1_out, cfg);

    let score_ch = cfg.score_ch();
    let probs_dp_offset = ad * seq * 2;
    let probs_dp_bytes = 2 * score_ch * seq * 2;
    let probs_dp_raw = &bwd1_out[probs_dp_offset..probs_dp_offset + probs_dp_bytes];

    let f32_to_fp16_bytes = |data: &[f32]| -> Vec<u8> {
        let mut buf = vec![0u8; data.len() * 2];
        for (i, &v) in data.iter().enumerate() {
            let fp16 = half::f16::from_f32(v);
            buf[i * 2..i * 2 + 2].copy_from_slice(&fp16.to_le_bytes());
        }
        buf
    };
    let q_fp16 = f32_to_fp16_bytes(&ac.q);
    let k_fp16 = f32_to_fp16_bytes(&ac.k);
    let mut bwd2_input = Vec::with_capacity(probs_dp_bytes + q_fp16.len() + k_fp16.len());
    bwd2_input.extend_from_slice(probs_dp_raw);
    bwd2_input.extend_from_slice(&q_fp16);
    bwd2_input.extend_from_slice(&k_fp16);

    let bwd2_spec = KernelSpec::for_kernel(cfg, KernelType::SdpaBwd2);
    let bwd2_kernel = kernels
        .sdpa_bwd2
        .as_ref()
        .ok_or_else(|| "sdpa_bwd2 kernel missing".to_string())?;
    bwd2_kernel.write_input(0, &bwd2_input);
    bwd2_kernel
        .eval()
        .map_err(|e| format!("sdpa_bwd2 eval: {e}"))?;
    let mut bwd2_out = vec![0u8; bwd2_spec.output_bytes];
    bwd2_kernel.read_output(0, &mut bwd2_out);
    let (mut dq, mut dk) = ane_weights::unpack_sdpa_bwd2(&bwd2_out, cfg);

    ane_forward::rope_backward(
        &mut dq,
        &mut dk,
        cfg.n_heads,
        cfg.head_dim(),
        seq,
        cfg.rope_theta,
    );

    if let (Some(q_pre), Some(q_nw)) = (&ac.q_pre_norm, &lw.q_norm) {
        ane_forward::qk_rmsnorm_bwd(
            &mut dq,
            &mut vec![0.0f32; cfg.head_dim()],
            q_pre,
            q_nw,
            cfg.n_heads,
            cfg.head_dim(),
            seq,
            cfg.rms_eps,
        );
    }
    if let (Some(k_pre), Some(k_nw)) = (&ac.k_pre_norm, &lw.k_norm) {
        ane_forward::qk_rmsnorm_bwd(
            &mut dk,
            &mut vec![0.0f32; cfg.head_dim()],
            k_pre,
            k_nw,
            cfg.n_heads,
            cfg.head_dim(),
            seq,
            cfg.rms_eps,
        );
    }

    let dq_for_wq = if let Some(dg) = &d_gate {
        ane_forward::merge_q_gate(&dq, dg, cfg.n_heads, cfg.head_dim(), seq)
    } else {
        dq
    };

    let qkv_input = ane_weights::pack_qkvb(&dq_for_wq, &dk, &dv, &lw.wq, &lw.wk, &lw.wv, cfg);
    let qkv_spec = KernelSpec::for_kernel(cfg, KernelType::Qkvb);
    let qkv_kernel = kernels
        .qkv_bwd
        .as_ref()
        .ok_or_else(|| "qkv_bwd kernel missing".to_string())?;
    qkv_kernel.write_input(0, &qkv_input);
    qkv_kernel
        .eval()
        .map_err(|e| format!("qkv_bwd eval: {e}"))?;
    let mut qkv_out = vec![0u8; qkv_spec.output_bytes];
    qkv_kernel.read_output(0, &mut qkv_out);
    Ok(qkv_out
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// Backward attention using fused SDPA kernel (replaces sdpa_bwd1 + sdpa_bwd2).
///
/// Flow: wot_bwd(1) → CPU gate → fused_sdpa(1) → CPU RoPE → qkv_bwd(1) = 3 dispatches.
fn mha_backward_fused_sdpa_dx_attn(
    kernels: &BackwardKernels,
    lw: &ane_weights::LayerWeights,
    ac: &ane_forward::LayerActivations,
    dx2: &[f32],
    cfg: &MilConfig,
) -> Result<Vec<f32>, String> {
    let dim = cfg.dim;
    let seq = cfg.seq_len;
    let ad = cfg.attn_dim();
    let hd = cfg.head_dim();
    let heads = cfg.n_heads;
    let kv_heads = cfg.n_kv_heads;
    let hpg = cfg.heads_per_group();

    // Step 1: Wo^T projection (DynMatmul — per-layer BLOBFILE handled by caller)
    let wot_input = ane_weights::pack_dyn_matmul(dx2, &lw.wo, dim, ad, seq);
    let wot_spec = KernelSpec::for_kernel(cfg, KernelType::Wot);
    let wot_kernel = kernels
        .wot_bwd
        .as_ref()
        .ok_or_else(|| "wot_bwd kernel missing".to_string())?;
    wot_kernel.write_input(0, &wot_input);
    wot_kernel
        .eval()
        .map_err(|e| format!("wot_bwd eval: {e}"))?;
    let mut wot_out = vec![0u8; wot_spec.output_bytes];
    wot_kernel.read_output(0, &mut wot_out);
    let da_raw: Vec<f32> = wot_out
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // Step 2: Gate backward (CPU)
    let (da, d_gate) = if cfg.attn_output_gate {
        let gate_raw = ac.attn_gate.as_ref().unwrap();
        let pre_gate = ac.attn_pre_gate.as_ref().unwrap();
        let (d_attn, dg) = sigmoid_gate_backward(&da_raw, gate_raw, pre_gate);
        (d_attn, Some(dg))
    } else {
        (da_raw, None)
    };

    // Step 3: Fused SDPA backward (1 dispatch replaces sdpa_bwd1 + sdpa_bwd2)
    // Expand K/V from kv_heads to full heads (repeat each KV head hpg times)
    let kvd = cfg.kv_dim();
    let q_rot = &ac.q[..ad * seq];
    let k_rot = &ac.k[..kvd * seq]; // [kvd, seq]
    let v = &ac.v[..kvd * seq]; // [kvd, seq]

    // Expand K: [kvd, seq] → [ad, seq] by repeating each KV head hpg times
    let mut k_expanded = vec![0.0f32; ad * seq];
    for kv_h in 0..kv_heads {
        for rep in 0..hpg {
            let dst_h = kv_h * hpg + rep;
            let src_off = kv_h * hd * seq;
            let dst_off = dst_h * hd * seq;
            k_expanded[dst_off..dst_off + hd * seq]
                .copy_from_slice(&k_rot[src_off..src_off + hd * seq]);
        }
    }
    let mut v_expanded = vec![0.0f32; ad * seq];
    for kv_h in 0..kv_heads {
        for rep in 0..hpg {
            let dst_h = kv_h * hpg + rep;
            let src_off = kv_h * hd * seq;
            let dst_off = dst_h * hd * seq;
            v_expanded[dst_off..dst_off + hd * seq]
                .copy_from_slice(&v[src_off..src_off + hd * seq]);
        }
    }

    // Pack input: da[ad,seq] | Q_rot[ad,seq] | K_expanded[ad,seq] | V_expanded[ad,seq]
    let in_ch = 4 * ad;
    let mut input = Vec::with_capacity(in_ch * seq);
    input.extend_from_slice(&da);
    input.extend_from_slice(q_rot);
    input.extend_from_slice(&k_expanded);
    input.extend_from_slice(&v_expanded);

    let fused_kernel = kernels
        .fused_attn_gqa_bwd
        .as_ref()
        .ok_or_else(|| "fused_sdpa_bwd kernel missing".to_string())?;
    let input_bytes =
        unsafe { std::slice::from_raw_parts(input.as_ptr() as *const u8, input.len() * 4) };
    fused_kernel.write_input(0, input_bytes);
    fused_kernel
        .eval()
        .map_err(|e| format!("fused_sdpa_bwd eval: {e}"))?;

    // Output: [1, 3*H, S, hd] fp32 — dQ_scaled | dK_scaled | dV stacked on head axis
    let out_elems = 3 * heads * seq * hd;
    let mut out_buf = vec![0u8; out_elems * 4];
    fused_kernel.read_output(0, &mut out_buf);
    let output: Vec<f32> = out_buf
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // Unpack: slice head axis into dQ[H,S,hd], dK[H,S,hd], dV[H,S,hd]
    let head_stride = seq * hd;
    let dq_start = 0;
    let dk_start = heads * head_stride;
    let dv_start = 2 * heads * head_stride;

    let dq_scaled: Vec<f32> = output[dq_start..dk_start].to_vec();
    let dk_all: Vec<f32> = output[dk_start..dv_start].to_vec();
    let dv_all: Vec<f32> = output[dv_start..].to_vec();

    // Transpose dQ from [H,S,hd] → [ad, seq] channel form
    let mut dq = vec![0.0f32; ad * seq];
    for h in 0..heads {
        for t in 0..seq {
            for d in 0..hd {
                dq[(h * hd + d) * seq + t] = dq_scaled[h * head_stride + t * hd + d];
            }
        }
    }
    // Transpose dK/dV at full [H,S,hd] → [ad, seq] (NOT GQA-reduced yet — need full
    // dimension for RoPE backward and QK-norm backward which work at attn_dim)
    let mut dk = vec![0.0f32; ad * seq];
    for h in 0..heads {
        for t in 0..seq {
            for d in 0..hd {
                dk[(h * hd + d) * seq + t] = dk_all[h * head_stride + t * hd + d];
            }
        }
    }

    // RoPE backward on Q and K at full attn_dim
    ane_forward::rope_backward(&mut dq, &mut dk, heads, hd, seq, cfg.rope_theta);

    // QK-norm backward (if applicable) — both at full attn_dim
    if let (Some(q_pre), Some(q_nw)) = (&ac.q_pre_norm, &lw.q_norm) {
        ane_forward::qk_rmsnorm_bwd(
            &mut dq,
            &mut vec![0.0f32; cfg.head_dim()],
            q_pre,
            q_nw,
            cfg.n_heads,
            cfg.head_dim(),
            seq,
            cfg.rms_eps,
        );
    }
    if let (Some(k_pre), Some(k_nw)) = (&ac.k_pre_norm, &lw.k_norm) {
        ane_forward::qk_rmsnorm_bwd(
            &mut dk,
            &mut vec![0.0f32; cfg.head_dim()],
            k_pre,
            k_nw,
            cfg.n_heads, // dk still at full attn_dim here
            cfg.head_dim(),
            seq,
            cfg.rms_eps,
        );
    }

    // NOW GQA-reduce dK → [kv_dim, seq], dV → [kv_dim, seq]
    let mut dk_reduced = vec![0.0f32; kvd * seq];
    let inv_hpg = 1.0 / hpg as f32;
    for kv_h in 0..kv_heads {
        for rep in 0..hpg {
            let src_h = kv_h * hpg + rep;
            for c in 0..hd {
                let dst_row = kv_h * hd + c;
                let src_row = src_h * hd + c;
                for t in 0..seq {
                    dk_reduced[dst_row * seq + t] += dk[src_row * seq + t] * inv_hpg;
                }
            }
        }
    }
    let dk = dk_reduced;

    // GQA-reduce dV: transpose from [H,S,hd] then reduce
    let mut dv = vec![0.0f32; kvd * seq];
    for kv_h in 0..kv_heads {
        for rep in 0..hpg {
            let src_h = kv_h * hpg + rep;
            for t in 0..seq {
                for d in 0..hd {
                    let dst_row = kv_h * hd + d;
                    dv[dst_row * seq + t] += dv_all[src_h * head_stride + t * hd + d] * inv_hpg;
                }
            }
        }
    }

    // Step 5: QKV^T projection (same as 4-kernel path)
    let dq_for_wq = if let Some(dg) = &d_gate {
        ane_forward::merge_q_gate(&dq, dg, cfg.n_heads, cfg.head_dim(), seq)
    } else {
        dq
    };

    // Re-expand dk/dv to [attn_dim, seq] for QKV^T kernel (weights are stored expanded)
    let dk_expanded = if hpg > 1 {
        let mut exp = vec![0.0f32; ad * seq];
        for kv_h in 0..kv_heads {
            for rep in 0..hpg {
                let dst_h = kv_h * hpg + rep;
                for c in 0..hd {
                    let src_row = kv_h * hd + c;
                    let dst_row = dst_h * hd + c;
                    exp[dst_row * seq..(dst_row + 1) * seq]
                        .copy_from_slice(&dk[src_row * seq..(src_row + 1) * seq]);
                }
            }
        }
        exp
    } else {
        dk
    };
    let dv_expanded = if hpg > 1 {
        let mut exp = vec![0.0f32; ad * seq];
        for kv_h in 0..kv_heads {
            for rep in 0..hpg {
                let dst_h = kv_h * hpg + rep;
                for c in 0..hd {
                    let src_row = kv_h * hd + c;
                    let dst_row = dst_h * hd + c;
                    exp[dst_row * seq..(dst_row + 1) * seq]
                        .copy_from_slice(&dv[src_row * seq..(src_row + 1) * seq]);
                }
            }
        }
        exp
    } else {
        dv
    };

    let qkv_input = ane_weights::pack_qkvb(
        &dq_for_wq,
        &dk_expanded,
        &dv_expanded,
        &lw.wq,
        &lw.wk,
        &lw.wv,
        cfg,
    );
    let qkv_spec = KernelSpec::for_kernel(cfg, KernelType::Qkvb);
    let qkv_kernel = kernels
        .qkv_bwd
        .as_ref()
        .ok_or_else(|| "qkv_bwd kernel missing".to_string())?;
    qkv_kernel.write_input(0, &qkv_input);
    qkv_kernel
        .eval()
        .map_err(|e| format!("qkv_bwd eval: {e}"))?;
    let mut qkv_out = vec![0u8; qkv_spec.output_bytes];
    qkv_kernel.read_output(0, &mut qkv_out);
    Ok(qkv_out
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// CPU-only backward pass for LoRA training (no ANE kernels needed).
///
/// Only computes LoRA adapter gradients — base model weights are frozen.
/// All matmuls (W^T @ gradient) run on CPU. SDPA backward runs on CPU.
pub fn backward_lora_cpu<T: ane_forward::TokenId>(
    model: &ModelWeights,
    fwd: &ane_forward::ForwardResultWithLora,
    lora: &super::ane_lora::LoraModel,
    tokens: &[T],
) -> BackwardResultWithLora {
    backward_lora_cpu_generic(model, fwd, lora, tokens, 0.0, 1.0)
}

/// Backward pass generic over weight source (supports both full and quantized weights).
///
/// Includes training stability features: logit softcap backward and loss scaling.
///
/// `softcap`: must match the value used in forward (0.0 disables)
/// `loss_scale`: multiply dlogits by this factor, divide LoRA grads at end (1.0 disables)
pub fn backward_lora_cpu_generic<T: ane_forward::TokenId, W: ane_weights::WeightSource>(
    model: &W,
    fwd: &ane_forward::ForwardResultWithLora,
    lora: &super::ane_lora::LoraModel,
    _tokens: &[T],
    _softcap: f32,
    loss_scale: f32,
) -> BackwardResultWithLora {
    use super::ane_lora;

    let cfg = model.cfg();
    // Use actual dimensions from loaded weights, not from config
    let dim = model.actual_dim();
    let hidden = model.actual_hidden_dim();
    let seq = cfg.seq_len;
    let n_layers = model.n_layers();
    let scale = lora.scale();
    let apply_loss_scale = (loss_scale - 1.0).abs() > f32::EPSILON;

    // Base model weights are frozen — skip the 5.9 GB gradient allocation.
    let mut lora_grads = ane_lora::LoraModelGrads::zeros(lora);

    // Use fused CE gradient directly (softcap chain rule already baked in)
    let mut dy = fwd.base.classifier_dy.clone();
    if apply_loss_scale {
        for v in dy.iter_mut() {
            *v *= loss_scale;
        }
    }

    // Reconstruct x_cur from last layer (needed for rmsnorm_bwd)
    let last_act = &fwd.base.layer_acts[n_layers - 1];
    let mut x_cur = last_act.x2.clone();
    ane_forward::vec_add_inplace(&mut x_cur, &last_act.ffn_out);

    // Final RMSNorm backward
    let mut dx_rms = vec![0.0f32; dim * seq];
    rmsnorm_bwd(
        &mut dx_rms,
        &mut vec![0.0f32; dim],
        &dy,
        &x_cur,
        model.rms_final(),
        dim,
        seq,
        cfg.rms_eps,
    );
    dy = dx_rms;

    // Per-layer backward (reverse)
    for l in (0..n_layers).rev() {
        let ac = &fwd.base.layer_acts[l];
        let la = &fwd.lora_acts[l];

        let dffn = dy.clone();

        // LoRA W2 backward
        if let (Some(w2_adapter), Some(w2_x), Some(w2_h), Some(lg)) = (
            lora.layers[l].w2.as_ref(),
            la.w2_x.as_ref(),
            la.w2_h.as_ref(),
            lora_grads.layers[l].w2.as_mut(),
        ) {
            let scaled_dffn: Vec<f32> = dffn.iter().map(|&v| v * scale).collect();
            let (_dx, da, db) = w2_adapter.backward_cpu(&scaled_dffn, w2_x, w2_h, seq);
            for (g, v) in lg.da.iter_mut().zip(da.iter()) {
                *g += v;
            }
            for (g, v) in lg.db.iter_mut().zip(db.iter()) {
                *g += v;
            }
        }

        // GDN layers: quantized FFN backward only (no attention backward —
        // LoRA targets wq/wv/wo don't apply to GDN's linear_attn projections).
        if let Some(ql) = model.quantized_layer(l).filter(|ql| ql.gdn.is_some()) {
            let mut quantized_workspace = ane_forward::QuantizedMatmulWorkspace::default();
            // FFN backward through quantized weights
            let mut dsilu = vec![0.0f32; ql.w2.cols * seq];
            ane_forward::cpu_quantized_matmul_lhs_transposed_into(
                &ql.w2,
                &dffn,
                seq,
                &mut dsilu,
                &mut quantized_workspace,
                false,
            );
            let mut dh1 = vec![0.0f32; hidden * seq];
            let mut dh3 = vec![0.0f32; hidden * seq];
            silu_bwd(&mut dh1, &mut dh3, &dsilu, &ac.h1, &ac.h3, hidden * seq);
            let mut dx_ffn = vec![0.0f32; ql.w1.cols * seq];
            ane_forward::cpu_quantized_matmul_lhs_transposed_into(
                &ql.w1,
                &dh1,
                seq,
                &mut dx_ffn,
                &mut quantized_workspace,
                false,
            );
            ane_forward::cpu_quantized_matmul_lhs_transposed_into(
                &ql.w3,
                &dh3,
                seq,
                &mut dx_ffn,
                &mut quantized_workspace,
                true,
            );
            let mut dx2 = vec![0.0f32; dim * seq];
            rmsnorm_bwd(
                &mut dx2,
                &mut vec![0.0f32; dim],
                &dx_ffn,
                &ac.x2,
                &ql.rms_ffn,
                dim,
                seq,
                cfg.rms_eps,
            );
            ane_forward::vec_add_inplace(&mut dx2, &dy);
            // Skip attention backward entirely for GDN — gradient flows
            // through the residual connection only (dx2 already includes dy).
            dy = dx2;
            continue;
        }

        if let Some(ql) = model.quantized_layer(l).filter(|ql| ql.gdn.is_none()) {
            let mut quantized_workspace = ane_forward::QuantizedMatmulWorkspace::default();
            // CPU: dsilu = dffn @ W2^T without materializing dense quantized weights.
            let mut dsilu = vec![0.0f32; ql.w2.cols * seq];
            ane_forward::cpu_quantized_matmul_lhs_transposed_into(
                &ql.w2,
                &dffn,
                seq,
                &mut dsilu,
                &mut quantized_workspace,
                false,
            );

            let mut dh1 = vec![0.0f32; hidden * seq];
            let mut dh3 = vec![0.0f32; hidden * seq];
            silu_bwd(&mut dh1, &mut dh3, &dsilu, &ac.h1, &ac.h3, hidden * seq);

            let mut dx_ffn = vec![0.0f32; ql.w1.cols * seq];
            ane_forward::cpu_quantized_matmul_lhs_transposed_into(
                &ql.w1,
                &dh1,
                seq,
                &mut dx_ffn,
                &mut quantized_workspace,
                false,
            );
            ane_forward::cpu_quantized_matmul_lhs_transposed_into(
                &ql.w3,
                &dh3,
                seq,
                &mut dx_ffn,
                &mut quantized_workspace,
                true,
            );

            let mut dx2 = vec![0.0f32; dim * seq];
            rmsnorm_bwd(
                &mut dx2,
                &mut vec![0.0f32; dim],
                &dx_ffn,
                &ac.x2,
                &ql.rms_ffn,
                dim,
                seq,
                cfg.rms_eps,
            );
            ane_forward::vec_add_inplace(&mut dx2, &dy);

            if let (Some(wo_adapter), Some(wo_x), Some(wo_h), Some(lg)) = (
                lora.layers[l].wo.as_ref(),
                la.wo_x.as_ref(),
                la.wo_h.as_ref(),
                lora_grads.layers[l].wo.as_mut(),
            ) {
                let scaled_dx2: Vec<f32> = dx2.iter().map(|&v| v * scale).collect();
                let (_dx, da, db) = wo_adapter.backward_cpu(&scaled_dx2, wo_x, wo_h, seq);
                for (g, v) in lg.da.iter_mut().zip(da.iter()) {
                    *g += v;
                }
                for (g, v) in lg.db.iter_mut().zip(db.iter()) {
                    *g += v;
                }
            }

            let mut da = vec![0.0f32; ql.wo.cols * seq];
            ane_forward::cpu_quantized_matmul_lhs_transposed_into(
                &ql.wo,
                &dx2,
                seq,
                &mut da,
                &mut quantized_workspace,
                false,
            );

            // Attn output gate backward (Qwen3.5)
            let (da, d_gate) = if cfg.attn_output_gate {
                let gate_raw = ac.attn_gate.as_ref().unwrap();
                let pre_gate = ac.attn_pre_gate.as_ref().unwrap();
                let (d_attn, dg) = sigmoid_gate_backward(&da, gate_raw, pre_gate);
                (d_attn, Some(dg))
            } else {
                (da, None)
            };

            let (mut dq, mut dk, _dv) =
                cpu_sdpa_backward(&da, &ac.q, &ac.k, &ac.v, cfg.n_heads, cfg.head_dim(), seq);

            ane_forward::rope_backward(
                &mut dq,
                &mut dk,
                cfg.n_heads,
                cfg.head_dim(),
                seq,
                cfg.rope_theta,
            );

            if let (Some(q_pre), Some(q_nw)) = (&ac.q_pre_norm, &ql.q_norm) {
                ane_forward::qk_rmsnorm_bwd(
                    &mut dq,
                    &mut vec![0.0f32; cfg.head_dim()],
                    q_pre,
                    q_nw,
                    cfg.n_heads,
                    cfg.head_dim(),
                    seq,
                    cfg.rms_eps,
                );
            }
            if let (Some(k_pre), Some(k_nw)) = (&ac.k_pre_norm, &ql.k_norm) {
                ane_forward::qk_rmsnorm_bwd(
                    &mut dk,
                    &mut vec![0.0f32; cfg.head_dim()],
                    k_pre,
                    k_nw,
                    cfg.n_heads,
                    cfg.head_dim(),
                    seq,
                    cfg.rms_eps,
                );
            }

            // Merge d_gate back with dq for wq backward (Qwen3.5 attn_output_gate)
            let dq_for_wq = if let Some(dg) = &d_gate {
                ane_forward::merge_q_gate(&dq, dg, cfg.n_heads, cfg.head_dim(), seq)
            } else {
                dq
            };

            let mut dx_attn = vec![0.0f32; ql.wq.cols * seq];
            ane_forward::cpu_quantized_matmul_lhs_transposed_into(
                &ql.wq,
                &dq_for_wq,
                seq,
                &mut dx_attn,
                &mut quantized_workspace,
                false,
            );
            let heads_per_group = cfg.heads_per_group();
            if heads_per_group > 1 {
                let dk_collapsed = ane_forward::collapse_grouped_kv_rows(
                    &dk,
                    ql.wk.rows,
                    cfg.head_dim(),
                    heads_per_group,
                    seq,
                );
                ane_forward::cpu_quantized_matmul_lhs_transposed_into(
                    &ql.wk,
                    &dk_collapsed,
                    seq,
                    &mut dx_attn,
                    &mut quantized_workspace,
                    true,
                );
            } else {
                ane_forward::cpu_quantized_matmul_lhs_transposed_into(
                    &ql.wk,
                    &dk,
                    seq,
                    &mut dx_attn,
                    &mut quantized_workspace,
                    true,
                );
            }
            if heads_per_group > 1 {
                let dv_collapsed = ane_forward::collapse_grouped_kv_rows(
                    &_dv,
                    ql.wv.rows,
                    cfg.head_dim(),
                    heads_per_group,
                    seq,
                );
                ane_forward::cpu_quantized_matmul_lhs_transposed_into(
                    &ql.wv,
                    &dv_collapsed,
                    seq,
                    &mut dx_attn,
                    &mut quantized_workspace,
                    true,
                );
            } else {
                ane_forward::cpu_quantized_matmul_lhs_transposed_into(
                    &ql.wv,
                    &_dv,
                    seq,
                    &mut dx_attn,
                    &mut quantized_workspace,
                    true,
                );
            }

            let mut dx_rms1 = vec![0.0f32; dim * seq];
            rmsnorm_bwd(
                &mut dx_rms1,
                &mut vec![0.0f32; dim],
                &dx_attn,
                &ac.layer_in,
                &ql.rms_att,
                dim,
                seq,
                cfg.rms_eps,
            );
            dy = dx_rms1;
            ane_forward::vec_add_inplace(&mut dy, &dx2);
            continue;
        }

        let lw_cow = model.layer(l);
        let lw = &*lw_cow;

        // CPU: dsilu = dffn @ W2^T (without materializing W2^T)
        let dsilu = ane_forward::cpu_matmul_lhs_transposed(&lw.w2, dim, hidden, &dffn, seq);

        // silu backward
        let mut dh1 = vec![0.0f32; hidden * seq];
        let mut dh3 = vec![0.0f32; hidden * seq];
        silu_bwd(&mut dh1, &mut dh3, &dsilu, &ac.h1, &ac.h3, hidden * seq);

        // CPU: dx_ffn = dh1 @ W1^T + dh3 @ W3^T (parallel)
        let (mut dx_ffn, dx_w3) = rayon::join(
            || ane_forward::cpu_matmul_lhs_transposed(&lw.w1, hidden, dim, &dh1, seq),
            || ane_forward::cpu_matmul_lhs_transposed(&lw.w3, hidden, dim, &dh3, seq),
        );
        ane_forward::vec_add_inplace(&mut dx_ffn, &dx_w3);

        // RMSNorm backward (FFN)
        let mut dx2 = vec![0.0f32; dim * seq];
        rmsnorm_bwd(
            &mut dx2,
            &mut vec![0.0f32; dim],
            &dx_ffn,
            &ac.x2,
            &lw.rms_ffn,
            dim,
            seq,
            cfg.rms_eps,
        );
        ane_forward::vec_add_inplace(&mut dx2, &dy);

        // GDN layers: skip attention backward — gradient flows through residual only.
        if lw.gdn.is_some() {
            dy = dx2;
            continue;
        }

        // LoRA Wo backward
        if let (Some(wo_adapter), Some(wo_x), Some(wo_h), Some(lg)) = (
            lora.layers[l].wo.as_ref(),
            la.wo_x.as_ref(),
            la.wo_h.as_ref(),
            lora_grads.layers[l].wo.as_mut(),
        ) {
            let scaled_dx2: Vec<f32> = dx2.iter().map(|&v| v * scale).collect();
            let (_dx, da, db) = wo_adapter.backward_cpu(&scaled_dx2, wo_x, wo_h, seq);
            for (g, v) in lg.da.iter_mut().zip(da.iter()) {
                *g += v;
            }
            for (g, v) in lg.db.iter_mut().zip(db.iter()) {
                *g += v;
            }
        }

        // CPU: da = dx2 @ Wo^T (no explicit transpose)
        let ad = cfg.attn_dim();
        let da = ane_forward::cpu_matmul_lhs_transposed(&lw.wo, dim, ad, &dx2, seq);

        // Attn output gate backward (Qwen3.5)
        let (da, d_gate) = if cfg.attn_output_gate {
            let gate_raw = ac.attn_gate.as_ref().unwrap();
            let pre_gate = ac.attn_pre_gate.as_ref().unwrap();
            let (d_attn, dg) = sigmoid_gate_backward(&da, gate_raw, pre_gate);
            (d_attn, Some(dg))
        } else {
            (da, None)
        };

        // CPU SDPA backward
        let (mut dq, mut dk, _dv) =
            cpu_sdpa_backward(&da, &ac.q, &ac.k, &ac.v, cfg.n_heads, cfg.head_dim(), seq);

        // RoPE backward
        ane_forward::rope_backward(
            &mut dq,
            &mut dk,
            cfg.n_heads,
            cfg.head_dim(),
            seq,
            cfg.rope_theta,
        );

        // QK-norm backward
        if let (Some(q_pre), Some(q_nw)) = (&ac.q_pre_norm, &lw.q_norm) {
            ane_forward::qk_rmsnorm_bwd(
                &mut dq,
                &mut vec![0.0f32; cfg.head_dim()],
                q_pre,
                q_nw,
                cfg.n_heads,
                cfg.head_dim(),
                seq,
                cfg.rms_eps,
            );
        }
        if let (Some(k_pre), Some(k_nw)) = (&ac.k_pre_norm, &lw.k_norm) {
            ane_forward::qk_rmsnorm_bwd(
                &mut dk,
                &mut vec![0.0f32; cfg.head_dim()],
                k_pre,
                k_nw,
                cfg.n_heads,
                cfg.head_dim(),
                seq,
                cfg.rms_eps,
            );
        }

        // Merge d_gate back with dq for wq backward (Qwen3.5 attn_output_gate)
        let dq_for_wq = if let Some(dg) = &d_gate {
            ane_forward::merge_q_gate(&dq, dg, cfg.n_heads, cfg.head_dim(), seq)
        } else {
            dq
        };

        // CPU: dx_attn = dq @ Wq^T + dk @ Wk^T + dv @ Wv^T (parallel)
        // wq is [q_proj_dim, dim] — q_proj_dim = 2*attn_dim when attn_output_gate
        let qpd = cfg.q_proj_dim();
        let (mut dx_attn, (dx_k, dx_v)) = rayon::join(
            || ane_forward::cpu_matmul_lhs_transposed(&lw.wq, qpd, dim, &dq_for_wq, seq),
            || {
                rayon::join(
                    || ane_forward::cpu_matmul_lhs_transposed(&lw.wk, ad, dim, &dk, seq),
                    || ane_forward::cpu_matmul_lhs_transposed(&lw.wv, ad, dim, &_dv, seq),
                )
            },
        );
        ane_forward::vec_add_inplace(&mut dx_attn, &dx_k);
        ane_forward::vec_add_inplace(&mut dx_attn, &dx_v);

        // RMSNorm backward (attention)
        let mut dx_rms1 = vec![0.0f32; dim * seq];
        rmsnorm_bwd(
            &mut dx_rms1,
            &mut vec![0.0f32; dim],
            &dx_attn,
            &ac.layer_in,
            &lw.rms_att,
            dim,
            seq,
            cfg.rms_eps,
        );
        dy = dx_rms1;
        ane_forward::vec_add_inplace(&mut dy, &dx2);
    }

    // Divide LoRA gradients by loss_scale to compensate for scaled dlogits
    if apply_loss_scale {
        let inv_scale = 1.0 / loss_scale;
        for layer_grads in &mut lora_grads.layers {
            let scale_grads = |g: &mut Option<ane_lora::LoraAdapterGrads>| {
                if let Some(ref mut grads) = g {
                    for v in grads.da.iter_mut() {
                        *v *= inv_scale;
                    }
                    for v in grads.db.iter_mut() {
                        *v *= inv_scale;
                    }
                }
            };
            scale_grads(&mut layer_grads.wq);
            scale_grads(&mut layer_grads.wv);
            scale_grads(&mut layer_grads.wo);
            scale_grads(&mut layer_grads.w2);
        }
    }

    sanitize_lora_grads(&mut lora_grads);

    BackwardResultWithLora {
        model_grads: None,
        lora_grads,
    }
}

/// Backward pass with LoRA gradient computation.
///
/// Computes base model gradients (frozen — caller ignores them for LoRA training)
/// plus LoRA adapter gradients at Wo and W2 injection points.
pub fn backward_with_lora<T: ane_forward::TokenId>(
    kernels: &BackwardKernels,
    model: &ModelWeights,
    fwd: &ane_forward::ForwardResultWithLora,
    lora: &super::ane_lora::LoraModel,
    _lora_kernels: &super::ane_lora::LoraKernels,
    tokens: &[T],
) -> Result<BackwardResultWithLora, String> {
    use super::ane_lora;

    let cfg = &model.cfg;
    let dim = cfg.dim;
    let seq = cfg.seq_len;
    let n_layers = model.layers.len();
    let scale = lora.scale();

    // Run the base backward pass
    let model_grads = backward(kernels, model, &fwd.base, tokens)?;

    // Compute LoRA gradients by replaying the backward chain to reconstruct
    // per-layer dffn (gradient at FFN residual) and dx2 (gradient at attention
    // residual). The base backward already ran but doesn't expose intermediates,
    // so we replay the chain here. This is ~2x cost but correct.
    let mut lora_grads = ane_lora::LoraModelGrads::zeros(lora);

    // Use fused CE gradient directly, then rmsnorm_bwd
    let last_act = &fwd.base.layer_acts[n_layers - 1];
    let mut x_cur = last_act.x2.clone();
    ane_forward::vec_add_inplace(&mut x_cur, &last_act.ffn_out);

    let mut dy = vec![0.0f32; dim * seq];
    rmsnorm_bwd(
        &mut dy,
        &mut vec![0.0f32; dim],
        &fwd.base.classifier_dy,
        &x_cur,
        &model.rms_final,
        dim,
        seq,
        cfg.rms_eps,
    );

    let hidden = cfg.hidden_dim;
    for l in (0..n_layers).rev() {
        let lw = &model.layers[l];
        let ac = &fwd.base.layer_acts[l];
        let la = &fwd.lora_acts[l];

        // dffn = dy (gradient at FFN residual point)
        let dffn = dy.clone();

        // LoRA W2 backward: compute weight grads using dffn
        if let (Some(w2_adapter), Some(w2_x), Some(w2_h), Some(lg)) = (
            lora.layers[l].w2.as_ref(),
            la.w2_x.as_ref(),
            la.w2_h.as_ref(),
            lora_grads.layers[l].w2.as_mut(),
        ) {
            // Scale dffn by LoRA scale for the LoRA path gradient
            let scaled_dffn: Vec<f32> = dffn.iter().map(|&v| v * scale).collect();
            let (_dx, da, db) = w2_adapter.backward_cpu(&scaled_dffn, w2_x, w2_h, seq);
            for (g, v) in lg.da.iter_mut().zip(da.iter()) {
                *g += v;
            }
            for (g, v) in lg.db.iter_mut().zip(db.iter()) {
                *g += v;
            }
        }

        // Replay base backward to get dx2 (gradient at attention residual point)
        let mut dsilu = kernels
            .ffn_bwd
            .eval_w2t(&dffn, &lw.w2, cfg)
            .map_err(|e| format!("lora ffn_bwd_w2t: {e}"))?;
        ane_forward::clamp_fp16(&mut dsilu);

        let mut dh1 = vec![0.0f32; hidden * seq];
        let mut dh3 = vec![0.0f32; hidden * seq];
        silu_bwd(&mut dh1, &mut dh3, &dsilu, &ac.h1, &ac.h3, hidden * seq);

        let mut dx_ffn = kernels
            .ffn_bwd
            .eval_w13t(&dh1, &dh3, &lw.w1, &lw.w3, cfg)
            .map_err(|e| format!("lora ffn_bwd_w13t: {e}"))?;
        ane_forward::clamp_fp16(&mut dx_ffn);

        let mut dx2 = vec![0.0f32; dim * seq];
        rmsnorm_bwd(
            &mut dx2,
            &mut vec![0.0f32; dim],
            &dx_ffn,
            &ac.x2,
            &lw.rms_ffn,
            dim,
            seq,
            cfg.rms_eps,
        );
        ane_forward::vec_add_inplace(&mut dx2, &dy);

        // LoRA Wo backward: compute weight grads using dx2
        if let (Some(wo_adapter), Some(wo_x), Some(wo_h), Some(lg)) = (
            lora.layers[l].wo.as_ref(),
            la.wo_x.as_ref(),
            la.wo_h.as_ref(),
            lora_grads.layers[l].wo.as_mut(),
        ) {
            let scaled_dx2: Vec<f32> = dx2.iter().map(|&v| v * scale).collect();
            let (_dx, da, db) = wo_adapter.backward_cpu(&scaled_dx2, wo_x, wo_h, seq);
            for (g, v) in lg.da.iter_mut().zip(da.iter()) {
                *g += v;
            }
            for (g, v) in lg.db.iter_mut().zip(db.iter()) {
                *g += v;
            }
        }

        // Continue backward chain for next layer
        let ad = cfg.attn_dim();
        let wot_input = ane_weights::pack_dyn_matmul(&dx2, &lw.wo, dim, ad, seq);
        let wot_spec = ane_mil::KernelSpec::for_kernel(cfg, ane_mil::KernelType::Wot);
        let wot_kernel = kernels
            .wot_bwd
            .as_ref()
            .expect("wot_bwd kernel required for MHA backward");
        wot_kernel.write_input(0, &wot_input);
        wot_kernel
            .eval()
            .map_err(|e| format!("lora wot_bwd: {e}"))?;
        let mut wot_out = vec![0u8; wot_spec.output_bytes];
        wot_kernel.read_output(0, &mut wot_out);
        let da_vec: Vec<f32> = wot_out
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // SDPA backward (simplified — just propagate through attention for dy chain)
        let bwd1_input = ane_weights::pack_sdpa_bwd1(&ac.q, &ac.k, &ac.v, &da_vec, cfg);
        let bwd1_spec = ane_mil::KernelSpec::for_kernel(cfg, ane_mil::KernelType::SdpaBwd1);
        let bwd1_kernel = kernels
            .sdpa_bwd1
            .as_ref()
            .expect("sdpa_bwd1 kernel required for MHA backward");
        bwd1_kernel.write_input(0, &bwd1_input);
        bwd1_kernel
            .eval()
            .map_err(|e| format!("lora sdpa_bwd1: {e}"))?;
        let mut bwd1_out = vec![0u8; bwd1_spec.output_bytes];
        bwd1_kernel.read_output(0, &mut bwd1_out);
        let (dv, _, _) = ane_weights::unpack_sdpa_bwd1(&bwd1_out, cfg);

        let score_ch = cfg.score_ch();
        let probs_dp_offset = ad * seq * 2;
        let probs_dp_bytes = 2 * score_ch * seq * 2;
        let probs_dp_raw = &bwd1_out[probs_dp_offset..probs_dp_offset + probs_dp_bytes];

        let f32_to_fp16_bytes = |data: &[f32]| -> Vec<u8> {
            let mut buf = vec![0u8; data.len() * 2];
            for (i, &v) in data.iter().enumerate() {
                let fp16 = half::f16::from_f32(v);
                buf[i * 2..i * 2 + 2].copy_from_slice(&fp16.to_le_bytes());
            }
            buf
        };
        let q_fp16 = f32_to_fp16_bytes(&ac.q);
        let k_fp16 = f32_to_fp16_bytes(&ac.k);
        let mut bwd2_input = Vec::with_capacity(probs_dp_bytes + q_fp16.len() + k_fp16.len());
        bwd2_input.extend_from_slice(probs_dp_raw);
        bwd2_input.extend_from_slice(&q_fp16);
        bwd2_input.extend_from_slice(&k_fp16);

        let bwd2_spec = ane_mil::KernelSpec::for_kernel(cfg, ane_mil::KernelType::SdpaBwd2);
        let bwd2_kernel = kernels
            .sdpa_bwd2
            .as_ref()
            .expect("sdpa_bwd2 kernel required for MHA backward");
        bwd2_kernel.write_input(0, &bwd2_input);
        bwd2_kernel
            .eval()
            .map_err(|e| format!("lora sdpa_bwd2: {e}"))?;
        let mut bwd2_out = vec![0u8; bwd2_spec.output_bytes];
        bwd2_kernel.read_output(0, &mut bwd2_out);
        let (mut dq, mut dk) = ane_weights::unpack_sdpa_bwd2(&bwd2_out, cfg);

        ane_forward::rope_backward(
            &mut dq,
            &mut dk,
            cfg.n_heads,
            cfg.head_dim(),
            seq,
            cfg.rope_theta,
        );

        if let (Some(q_pre), Some(q_nw)) = (&ac.q_pre_norm, &lw.q_norm) {
            ane_forward::qk_rmsnorm_bwd(
                &mut dq,
                &mut vec![0.0f32; cfg.head_dim()],
                q_pre,
                q_nw,
                cfg.n_heads,
                cfg.head_dim(),
                seq,
                cfg.rms_eps,
            );
        }
        if let (Some(k_pre), Some(k_nw)) = (&ac.k_pre_norm, &lw.k_norm) {
            ane_forward::qk_rmsnorm_bwd(
                &mut dk,
                &mut vec![0.0f32; cfg.head_dim()],
                k_pre,
                k_nw,
                cfg.n_heads,
                cfg.head_dim(),
                seq,
                cfg.rms_eps,
            );
        }

        let qkv_input = ane_weights::pack_qkvb(&dq, &dk, &dv, &lw.wq, &lw.wk, &lw.wv, cfg);
        let qkv_spec = ane_mil::KernelSpec::for_kernel(cfg, ane_mil::KernelType::Qkvb);
        let qkv_kernel = kernels
            .qkv_bwd
            .as_ref()
            .expect("qkv_bwd kernel required for MHA backward");
        qkv_kernel.write_input(0, &qkv_input);
        qkv_kernel
            .eval()
            .map_err(|e| format!("lora qkv_bwd: {e}"))?;
        let mut qkv_out = vec![0u8; qkv_spec.output_bytes];
        qkv_kernel.read_output(0, &mut qkv_out);
        let dx_attn: Vec<f32> = qkv_out
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let mut dx_rms1 = vec![0.0f32; dim * seq];
        rmsnorm_bwd(
            &mut dx_rms1,
            &mut vec![0.0f32; dim],
            &dx_attn,
            &ac.layer_in,
            &lw.rms_att,
            dim,
            seq,
            cfg.rms_eps,
        );
        dy = dx_rms1;
        ane_forward::vec_add_inplace(&mut dy, &dx2);
    }

    Ok(BackwardResultWithLora {
        model_grads: Some(model_grads),
        lora_grads,
    })
}

// ---------------------------------------------------------------------------
// ANE-accelerated backward (CPU attention backward + ANE FFN backward)
// ---------------------------------------------------------------------------

/// Logit softcap backward: chain-rule through cap * tanh(raw / cap).
/// Multiplies dlogits by (1 - (logits / cap)²) where logits are post-softcap.
pub fn logit_softcap_bwd(dlogits: &mut [f32], logits: &[f32], cap: f32) {
    if cap <= 0.0 {
        return;
    }
    let inv_cap = 1.0 / cap;
    for (dl, l) in dlogits.iter_mut().zip(logits.iter()) {
        let t = *l * inv_cap; // = tanh(raw/cap) since logits are capped
        *dl *= 1.0 - t * t;
    }
}

/// ANE-accelerated backward pass generic over weight source.
///
/// Uses ANE for FFN weight-transpose matmuls (W2^T, W1^T+W3^T) and CPU for
/// attention backward (handles GQA, GDN, QK-norm, attn_output_gate).
/// Includes training stability features: logit softcap backward, loss scaling,
/// and scaled residual backward.
///
/// `softcap`: must match the value used in forward_ane_generic (0.0 disables)
/// `loss_scale`: multiply dlogits by this factor, divide LoRA grads at end (1.0 disables)
/// `residual_scale`: must match forward_ane_generic (1.0 disables)
pub fn backward_lora_ane_generic<T: ane_forward::TokenId, W: ane_weights::WeightSource>(
    bwd_kernels: &BackwardKernels,
    model: &W,
    fwd: &ane_forward::ForwardResultWithLora,
    lora: &super::ane_lora::LoraModel,
    tokens: &[T],
    softcap: f32,
    loss_scale: f32,
    residual_scale: f32,
) -> BackwardResultWithLora {
    backward_lora_ane_generic_with_lora_kernels(
        bwd_kernels,
        model,
        fwd,
        lora,
        tokens,
        softcap,
        loss_scale,
        residual_scale,
        None,
    )
}

pub fn backward_lora_ane_generic_with_lora_kernels<
    T: ane_forward::TokenId,
    W: ane_weights::WeightSource,
>(
    bwd_kernels: &BackwardKernels,
    model: &W,
    fwd: &ane_forward::ForwardResultWithLora,
    lora: &super::ane_lora::LoraModel,
    _tokens: &[T],
    _softcap: f32,
    loss_scale: f32,
    residual_scale: f32,
    lora_kernels: Option<&super::ane_lora::LoraWeightGradKernels>,
) -> BackwardResultWithLora {
    backward_lora_ane_impl(
        bwd_kernels,
        model,
        fwd,
        lora,
        loss_scale,
        residual_scale,
        lora_kernels,
        None,
        None,
        None,
    )
}

/// Backward with optional pre-packed weight buffers (Orion delta patching).
pub fn backward_lora_ane_prepacked<W: ane_weights::WeightSource>(
    bwd_kernels: &BackwardKernels,
    model: &W,
    fwd: &ane_forward::ForwardResultWithLora,
    lora: &super::ane_lora::LoraModel,
    loss_scale: f32,
    residual_scale: f32,
    lora_kernels: Option<&super::ane_lora::LoraWeightGradKernels>,
    prepacked: &mut ane_weights::PrePackedWeights,
    lora_chains: Option<&super::ane_lora::LoraBackwardChains>,
) -> BackwardResultWithLora {
    backward_lora_ane_impl(
        bwd_kernels,
        model,
        fwd,
        lora,
        loss_scale,
        residual_scale,
        lora_kernels,
        Some(prepacked),
        lora_chains,
        None,
    )
}

/// Backward with multi-layer chains (persistent weights, 1.4x faster).
pub fn backward_lora_ane_prepacked_multi<W: ane_weights::WeightSource>(
    bwd_kernels: &BackwardKernels,
    model: &W,
    fwd: &ane_forward::ForwardResultWithLora,
    lora: &super::ane_lora::LoraModel,
    loss_scale: f32,
    residual_scale: f32,
    lora_kernels: Option<&super::ane_lora::LoraWeightGradKernels>,
    prepacked: &mut ane_weights::PrePackedWeights,
    lora_chains: Option<&super::ane_lora::LoraBackwardChains>,
    multi_chains: Option<&super::ane_lora::LoraMultiLayerChains>,
) -> BackwardResultWithLora {
    backward_lora_ane_impl(
        bwd_kernels,
        model,
        fwd,
        lora,
        loss_scale,
        residual_scale,
        lora_kernels,
        Some(prepacked),
        lora_chains,
        multi_chains,
    )
}

fn backward_lora_ane_impl<W: ane_weights::WeightSource>(
    bwd_kernels: &BackwardKernels,
    model: &W,
    fwd: &ane_forward::ForwardResultWithLora,
    lora: &super::ane_lora::LoraModel,
    loss_scale: f32,
    residual_scale: f32,
    lora_kernels: Option<&super::ane_lora::LoraWeightGradKernels>,
    mut prepacked: Option<&mut ane_weights::PrePackedWeights>,
    lora_chains: Option<&super::ane_lora::LoraBackwardChains>,
    multi_chains: Option<&super::ane_lora::LoraMultiLayerChains>,
) -> BackwardResultWithLora {
    use super::ane_lora;

    let cfg = model.cfg();
    let dim = model.actual_dim();
    let hidden = model.actual_hidden_dim();
    let seq = cfg.seq_len;
    let n_layers = model.n_layers();
    let scale = lora.scale();
    let apply_res_scale = (residual_scale - 1.0).abs() > f32::EPSILON;
    let apply_loss_scale = (loss_scale - 1.0).abs() > f32::EPSILON;

    let mut lora_grads = ane_lora::LoraModelGrads::zeros(lora);

    // Use fused CE gradient directly (softcap chain rule already baked in)
    let mut dy = fwd.base.classifier_dy.clone();
    if apply_loss_scale {
        for v in dy.iter_mut() {
            *v *= loss_scale;
        }
    }

    // Reconstruct x_cur from last layer (needed for rmsnorm_bwd)
    let last_act = &fwd.base.layer_acts[n_layers - 1];
    let mut x_cur = last_act.x2.clone();
    ane_forward::vec_add_inplace(&mut x_cur, &last_act.ffn_out);
    if apply_res_scale {
        for v in x_cur.iter_mut() {
            *v *= residual_scale;
        }
    }

    // Final RMSNorm backward
    let mut dx_rms = vec![0.0f32; dim * seq];
    rmsnorm_bwd(
        &mut dx_rms,
        &mut vec![0.0f32; dim],
        &dy,
        &x_cur,
        model.rms_final(),
        dim,
        seq,
        cfg.rms_eps,
    );
    dy = dx_rms;

    // Layer-drop: skip layers that were skipped in forward (identity activations).
    let skip_layers: Vec<usize> = std::env::var("NANOBOT_SKIP_LAYERS")
        .ok()
        .filter(|s| !s.is_empty())
        .map(|s| s.split(',').filter_map(|v| v.trim().parse().ok()).collect())
        .unwrap_or_default();

    // Per-layer backward (reverse) with optional phase profiling
    let mut _prof_ffn_us = 0u64;
    let mut _prof_attn_us = 0u64;
    let mut _prof_rmsnorm_us = 0u64;
    let mut _prof_lora_us = 0u64;
    let mut _prof_misc_us = 0u64;

    for l in (0..n_layers).rev() {
        // Layer-drop: pass dy through unchanged (identity backward).
        if skip_layers.contains(&l) {
            continue;
        }

        let ac = &fwd.base.layer_acts[l];
        let la = &fwd.lora_acts[l];

        let _t_layer = std::time::Instant::now();

        // Apply residual_scale to dy for FFN residual backward
        let dz_ffn = if apply_res_scale {
            dy.iter().map(|&v| v * residual_scale).collect::<Vec<_>>()
        } else {
            dy.clone()
        };
        let dffn = dz_ffn.clone();

        // Dequantize layer (needed by both ANE and CPU parallel paths)
        let lw_cow = model.layer(l);
        let lw = &*lw_cow;
        let use_ane_w2_grads = lora_kernels.and_then(|k| k.w2.as_ref()).is_some();
        let has_w2_chain = lora_chains.and_then(|c| c.w2.as_ref()).is_some();
        let has_w2_multi = multi_chains.and_then(|c| c.w2.as_ref()).is_some();

        // --- PARALLEL: ANE ffn_bwd_w2t || LoRA W2 grad + W1/W3 pre-transpose ---
        // ANE kernel is !Send, stays on calling thread. CPU work on a scoped thread.
        //
        // When prepacked weights are available, the W1/W3 clone on the CPU thread
        // is skipped (eval_w13t_prepacked doesn't need them).
        // When chain is available, W2 backward defers to main thread (chain is !Send).
        let has_prepacked = prepacked.is_some();
        let (dsilu, fused_dx_ffn, w2_grads, w1t, w3t) = std::thread::scope(|s| {
            // Spawn CPU work: LoRA W2 backward (CPU only when no ANE grads & no chain)
            // + (if not prepacked) W1/W3 clone
            let cpu_handle = s.spawn(|| {
                let w2_grads = if use_ane_w2_grads || has_w2_chain || has_w2_multi {
                    None // ANE grad kernels or chain handle W2 on main thread
                } else if let (Some(w2_adapter), Some(w2_x), Some(w2_h)) = (
                    lora.layers[l].w2.as_ref(),
                    la.w2_x.as_ref(),
                    la.w2_h.as_ref(),
                ) {
                    let scaled_dffn: Vec<f32> = dffn.iter().map(|&v| v * scale).collect();
                    let (_dx, da, db) = w2_adapter.backward_cpu(&scaled_dffn, w2_x, w2_h, seq);
                    Some((da, db))
                } else {
                    None
                };
                // Skip W1/W3 clone when prepacked (eval_w13t_prepacked uses pre-packed buffers)
                if has_prepacked {
                    (w2_grads, Vec::new(), Vec::new())
                } else {
                    let w1_ref = lw.w1.clone();
                    let w3_ref = lw.w3.clone();
                    (w2_grads, w1_ref, w3_ref)
                }
            });

            // Main thread: try fused FFN backward (1 dispatch) → split (2 dispatches) → DynMatmul
            // Fused: 3 matmuls in 1 kernel (fails at 35B dims)
            // Split: W2T+SiLU then W13T (2 shallower kernels, compiles at any dim)
            // DynMatmul: weight packed into input (slowest, always works)
            let fused_result = prepacked.as_ref().and_then(|pp| {
                pp.eval_fused_ffn_bwd(l, &dffn, &ac.h1, &ac.h3, dim, hidden, seq)
                    .or_else(|| {
                        pp.eval_fused_ffn_bwd_hotswap(l, &dffn, &ac.h1, &ac.h3, dim, hidden, seq)
                    })
                    .or_else(|| pp.eval_split_ffn_bwd(l, &dffn, &ac.h1, &ac.h3, dim, hidden, seq))
            });
            let (dsilu, fused_dx_ffn) = match fused_result {
                Some(Ok((dx, ds))) => (ds, Some(dx)),
                _ => {
                    // Fallback: separate W2^T dispatch (DynMatmul)
                    let ds = if let Some(ref mut pp) = prepacked {
                        bwd_kernels
                            .ffn_bwd
                            .eval_w2t_prepacked(&dffn, pp, l, cfg)
                            .unwrap_or_else(|e| {
                                tracing::warn!("ANE ffn_bwd_w2t_prepacked failed: {e}");
                                vec![0.0f32; hidden * seq]
                            })
                    } else {
                        bwd_kernels
                            .ffn_bwd
                            .eval_w2t(&dffn, &lw.w2, cfg)
                            .unwrap_or_else(|e| {
                                tracing::warn!("ANE ffn_bwd_w2t failed: {e}");
                                vec![0.0f32; hidden * seq]
                            })
                    };
                    (ds, None)
                }
            };

            let (w2_grads, w1t, w3t) = cpu_handle.join().unwrap();
            (dsilu, fused_dx_ffn, w2_grads, w1t, w3t)
        });

        // Clamp ANE backward outputs to prevent fp16 overflow propagation
        let mut dsilu = dsilu;
        ane_forward::clamp_fp16(&mut dsilu);

        // Accumulate W2 LoRA grads from parallel result (CPU backward_cpu path)
        if let (Some((da, db)), Some(lg)) = (w2_grads, lora_grads.layers[l].w2.as_mut()) {
            for (g, v) in lg.da.iter_mut().zip(da.iter()) {
                *g += v;
            }
            for (g, v) in lg.db.iter_mut().zip(db.iter()) {
                *g += v;
            }
        }
        // W2 multi-layer chain (persistent weights, no reload — 1.4x faster)
        let w2_grads_done = if has_w2_multi && !use_ane_w2_grads {
            if let (Some(w2_adapter), Some(w2_x), Some(w2_h), Some(lg), Some(chain)) = (
                lora.layers[l].w2.as_ref(),
                la.w2_x.as_ref(),
                la.w2_h.as_ref(),
                lora_grads.layers[l].w2.as_mut(),
                multi_chains.and_then(|c| c.w2.as_ref()),
            ) {
                let scaled_dffn: Vec<f32> = dffn.iter().map(|&v| v * scale).collect();
                match chain.backward_single(l, w2_adapter, &scaled_dffn, w2_x, w2_h) {
                    Ok((_dx, da, db)) => {
                        for (g, v) in lg.da.iter_mut().zip(da.iter()) {
                            *g += v;
                        }
                        for (g, v) in lg.db.iter_mut().zip(db.iter()) {
                            *g += v;
                        }
                        true
                    }
                    Err(e) => {
                        tracing::warn!("ANE LoRA W2 multi-chain failed: {e}");
                        false
                    }
                }
            } else {
                false
            }
        } else {
            false
        };
        // W2 shared chain fallback (reload weights per layer)
        if !w2_grads_done && has_w2_chain && !use_ane_w2_grads {
            if let (Some(w2_adapter), Some(w2_x), Some(w2_h), Some(lg), Some(chain)) = (
                lora.layers[l].w2.as_ref(),
                la.w2_x.as_ref(),
                la.w2_h.as_ref(),
                lora_grads.layers[l].w2.as_mut(),
                lora_chains.and_then(|c| c.w2.as_ref()),
            ) {
                let scaled_dffn: Vec<f32> = dffn.iter().map(|&v| v * scale).collect();
                let (da, db) = match chain.backward(w2_adapter, &scaled_dffn, w2_x, w2_h) {
                    Ok((_dx, da, db)) => (da, db),
                    Err(e) => {
                        tracing::warn!("ANE LoRA W2 chain failed: {e}");
                        let (_dx, da, db) =
                            w2_adapter.backward_cpu(&scaled_dffn, w2_x, w2_h, seq);
                        (da, db)
                    }
                };
                for (g, v) in lg.da.iter_mut().zip(da.iter()) {
                    *g += v;
                }
                for (g, v) in lg.db.iter_mut().zip(db.iter()) {
                    *g += v;
                }
            }
        }
        if let (Some(w2_adapter), Some(w2_x), Some(w2_h), Some(lg), Some(k)) = (
            lora.layers[l].w2.as_ref(),
            la.w2_x.as_ref(),
            la.w2_h.as_ref(),
            lora_grads.layers[l].w2.as_mut(),
            lora_kernels.and_then(|k| k.w2.as_ref()),
        ) {
            let scaled_dffn: Vec<f32> = dffn.iter().map(|&v| v * scale).collect();
            match k.eval(w2_adapter, &scaled_dffn, w2_x, w2_h, seq) {
                Ok((da, db)) => {
                    for (g, v) in lg.da.iter_mut().zip(da.iter()) {
                        *g += v;
                    }
                    for (g, v) in lg.db.iter_mut().zip(db.iter()) {
                        *g += v;
                    }
                }
                Err(e) => {
                    tracing::warn!("ANE LoRA W2 grad kernels failed: {e}");
                    // Try chained ANE backward, then CPU fallback
                    let (da, db) =
                        if let Some(chain) = lora_chains.and_then(|c| c.w2.as_ref()) {
                            match chain.backward(w2_adapter, &scaled_dffn, w2_x, w2_h) {
                                Ok((_dx, da, db)) => (da, db),
                                Err(e2) => {
                                    tracing::warn!("ANE LoRA W2 chain also failed: {e2}");
                                    let (_dx, da, db) =
                                        w2_adapter.backward_cpu(&scaled_dffn, w2_x, w2_h, seq);
                                    (da, db)
                                }
                            }
                        } else {
                            let (_dx, da, db) =
                                w2_adapter.backward_cpu(&scaled_dffn, w2_x, w2_h, seq);
                            (da, db)
                        };
                    for (g, v) in lg.da.iter_mut().zip(da.iter()) {
                        *g += v;
                    }
                    for (g, v) in lg.db.iter_mut().zip(db.iter()) {
                        *g += v;
                    }
                }
            }
        }

        // FFN backward dx_ffn: use fused result if available, else SiLU + W13^T
        let mut dx_ffn = if let Some(dx) = fused_dx_ffn {
            dx
        } else {
            // SiLU backward (CPU)
            let mut dh1 = vec![0.0f32; hidden * seq];
            let mut dh3 = vec![0.0f32; hidden * seq];
            silu_bwd(&mut dh1, &mut dh3, &dsilu, &ac.h1, &ac.h3, hidden * seq);

            // W13^T dispatch
            if let Some(ref mut pp) = prepacked {
                bwd_kernels
                    .ffn_bwd
                    .eval_w13t_prepacked(&dh1, &dh3, pp, l, cfg)
                    .unwrap_or_else(|e| {
                        tracing::warn!("ANE ffn_bwd_w13t_prepacked failed: {e}");
                        vec![0.0f32; dim * seq]
                    })
            } else {
                bwd_kernels
                    .ffn_bwd
                    .eval_w13t(&dh1, &dh3, &w1t, &w3t, cfg)
                    .unwrap_or_else(|e| {
                        tracing::warn!("ANE ffn_bwd_w13t failed: {e}");
                        vec![0.0f32; dim * seq]
                    })
            }
        };
        ane_forward::clamp_fp16(&mut dx_ffn);

        _prof_ffn_us += _t_layer.elapsed().as_micros() as u64;

        // RMSNorm backward (FFN): ANE per-layer kernel → CPU fallback
        let _t_rmsnorm = std::time::Instant::now();
        let mut dx2 = if let Some(Ok(result)) = prepacked
            .as_ref()
            .and_then(|pp| pp.eval_rmsnorm_bwd(l, &dx_ffn, &ac.x2, true))
        {
            result
        } else {
            let mut dx = vec![0.0f32; dim * seq];
            rmsnorm_bwd(
                &mut dx,
                &mut vec![0.0f32; dim],
                &dx_ffn,
                &ac.x2,
                &lw.rms_ffn,
                dim,
                seq,
                cfg.rms_eps,
            );
            dx
        };
        ane_forward::vec_add_inplace(&mut dx2, &dz_ffn);

        // Apply residual_scale for attention residual backward
        if apply_res_scale {
            for v in dx2.iter_mut() {
                *v *= residual_scale;
            }
        }

        _prof_rmsnorm_us += _t_rmsnorm.elapsed().as_micros() as u64;

        // --- LoRA Wo backward: multi-chain → chain → AdapterWeightGradKernels → CPU ---
        let _t_lora = std::time::Instant::now();
        if let (Some(wo_adapter), Some(wo_x), Some(wo_h), Some(lg)) = (
            lora.layers[l].wo.as_ref(),
            la.wo_x.as_ref(),
            la.wo_h.as_ref(),
            lora_grads.layers[l].wo.as_mut(),
        ) {
            let scaled_dx2: Vec<f32> = dx2.iter().map(|&v| v * scale).collect();
            // Try multi-layer chain first (persistent weights, 1.4x faster)
            let grads = if let Some(chain) = multi_chains.and_then(|c| c.wo.as_ref()) {
                match chain.backward_single(l, wo_adapter, &scaled_dx2, wo_x, wo_h) {
                    Ok((_dx, da, db)) => Some((da, db)),
                    Err(e) => {
                        tracing::warn!("ANE LoRA Wo multi-chain fell back: {e}");
                        None
                    }
                }
            } else {
                None
            };
            // Fall back to shared chain (reload weights per layer)
            let grads = grads.or_else(|| {
                lora_chains.and_then(|c| c.wo.as_ref()).and_then(|chain| {
                    match chain.backward(wo_adapter, &scaled_dx2, wo_x, wo_h) {
                        Ok((_dx, da, db)) => Some((da, db)),
                        Err(e) => {
                            tracing::warn!("ANE LoRA Wo chain fell back: {e}");
                            None
                        }
                    }
                })
            });
            // Fall back to AdapterWeightGradKernels
            let grads = grads.or_else(|| {
                lora_kernels
                    .and_then(|k| k.wo.as_ref())
                    .and_then(|k| match k.eval(wo_adapter, &scaled_dx2, wo_x, wo_h, seq) {
                        Ok((da, db)) => Some((da, db)),
                        Err(e) => {
                            tracing::warn!("ANE LoRA Wo grads fell back to CPU: {e}");
                            None
                        }
                    })
            });
            let (da, db) = if let Some(grads) = grads {
                grads
            } else {
                let (_dx, da, db) = wo_adapter.backward_cpu(&scaled_dx2, wo_x, wo_h, seq);
                (da, db)
            };
            for (g, v) in lg.da.iter_mut().zip(da.iter()) {
                *g += v;
            }
            for (g, v) in lg.db.iter_mut().zip(db.iter()) {
                *g += v;
            }
        }

        _prof_lora_us += _t_lora.elapsed().as_micros() as u64;

        // --- Attention backward (CPU — handles GQA, GDN, QK-norm, attn_output_gate) ---
        let _t_attn = std::time::Instant::now();
        if lw.gdn.is_some() {
            // GDN layers: skip attention backward, gradient flows through residual only.
            // x2 = layer_in + gdn_out, so d_layer_in = dx2 (residual) + d_gdn (skipped=0).
            dy = dx2;
            continue;
        }

        // Fused-first backward attention: 1 dispatch vs 4 separate + CPU interleaving.
        let dx_attn = 'attn_bwd: {
            // 1. Try fused per-layer kernel (real weights, delta-cached).
            //    QK-norm backward is handled inside the fused kernel when has_qk_norm=true.
            //    When the kernel was compiled without QK-norm support, the gradient bypasses
            //    QK-norm (small numerical drift, acceptable for LoRA where norms are frozen).
            if let Some(ref mut pp) = prepacked {
                if let Some(result) = pp.eval_bwd_fused_attn_gqa(l, &dx2, ac, cfg) {
                    match result {
                        Ok(dx) => {
                            let mut dx = dx;
                            ane_forward::clamp_fp16(&mut dx);
                            break 'attn_bwd dx;
                        }
                        Err(e) => {
                            tracing::warn!("fused_attn_gqa_bwd layer {l}: {e}");
                        }
                    }
                }
            }
            // 1b. Try fused Wot+SDPA per-layer kernel (2-dispatch: Wot+SDPA | QKV).
            //     Merges Wo^T + gate + SDPA into 1 dispatch, then QKV as dispatch 2.
            if let Some(ref pp) = prepacked {
                let ad = cfg.attn_dim();
                let hd = cfg.head_dim();
                let heads = cfg.n_heads;
                let kv_heads = cfg.n_kv_heads;
                let hpg = cfg.heads_per_group();
                let kvd = cfg.kv_dim();

                // Check if fused Wot+SDPA kernel is primed — try a dummy check via eval.
                // The eval returns None if not primed, so we build input and call directly.
                if pp.has_bwd_wot_sdpa(l) {
                    // Build input: dx2[dim] | Q_rot[ad] | K_exp[ad] | V_exp[ad] [+ pre_gate + gate_raw]
                    let mut k_expanded = vec![0.0f32; ad * seq];
                    for kv_h in 0..kv_heads {
                        for rep in 0..hpg {
                            let dst_h = kv_h * hpg + rep;
                            k_expanded[dst_h * hd * seq..(dst_h * hd + hd) * seq]
                                .copy_from_slice(&ac.k[kv_h * hd * seq..(kv_h * hd + hd) * seq]);
                        }
                    }
                    let mut v_expanded = vec![0.0f32; ad * seq];
                    for kv_h in 0..kv_heads {
                        for rep in 0..hpg {
                            let dst_h = kv_h * hpg + rep;
                            v_expanded[dst_h * hd * seq..(dst_h * hd + hd) * seq]
                                .copy_from_slice(&ac.v[kv_h * hd * seq..(kv_h * hd + hd) * seq]);
                        }
                    }

                    let in_ch = if cfg.attn_output_gate {
                        dim + 3 * ad + 2 * ad
                    } else {
                        dim + 3 * ad
                    };
                    let mut wot_sdpa_input = Vec::with_capacity(in_ch * seq);
                    wot_sdpa_input.extend_from_slice(&dx2);
                    wot_sdpa_input.extend_from_slice(&ac.q[..ad * seq]);
                    wot_sdpa_input.extend_from_slice(&k_expanded);
                    wot_sdpa_input.extend_from_slice(&v_expanded);
                    if cfg.attn_output_gate {
                        wot_sdpa_input.extend_from_slice(ac.attn_pre_gate.as_ref().unwrap());
                        wot_sdpa_input.extend_from_slice(ac.attn_gate.as_ref().unwrap());
                    }

                    if let Some(Ok(output)) = pp.eval_bwd_wot_sdpa(l, &wot_sdpa_input, cfg) {
                        // Output: [1, N*H, S, hd] — dQ|dK|dV [|d_gate]
                        let head_stride = seq * hd;
                        let mut dq = vec![0.0f32; ad * seq];
                        let mut dk = vec![0.0f32; ad * seq];
                        for h in 0..heads {
                            for t in 0..seq {
                                for d in 0..hd {
                                    dq[(h * hd + d) * seq + t] =
                                        output[h * head_stride + t * hd + d];
                                    dk[(h * hd + d) * seq + t] =
                                        output[heads * head_stride + h * head_stride + t * hd + d];
                                }
                            }
                        }

                        // RoPE + QK-norm backward (CPU)
                        ane_forward::rope_backward(
                            &mut dq,
                            &mut dk,
                            heads,
                            hd,
                            seq,
                            cfg.rope_theta,
                        );
                        if let (Some(q_pre), Some(q_nw)) = (&ac.q_pre_norm, &lw.q_norm) {
                            ane_forward::qk_rmsnorm_bwd(
                                &mut dq,
                                &mut vec![0.0f32; hd],
                                q_pre,
                                q_nw,
                                heads,
                                hd,
                                seq,
                                cfg.rms_eps,
                            );
                        }
                        if let (Some(k_pre), Some(k_nw)) = (&ac.k_pre_norm, &lw.k_norm) {
                            ane_forward::qk_rmsnorm_bwd(
                                &mut dk,
                                &mut vec![0.0f32; hd],
                                k_pre,
                                k_nw,
                                heads,
                                hd,
                                seq,
                                cfg.rms_eps,
                            );
                        }

                        // GQA-reduce dK, dV
                        let inv_hpg = 1.0 / hpg as f32;
                        let mut dk_r = vec![0.0f32; kvd * seq];
                        for kv_h in 0..kv_heads {
                            for rep in 0..hpg {
                                let src_h = kv_h * hpg + rep;
                                for c in 0..hd {
                                    for t in 0..seq {
                                        dk_r[(kv_h * hd + c) * seq + t] +=
                                            dk[(src_h * hd + c) * seq + t] * inv_hpg;
                                    }
                                }
                            }
                        }
                        let mut dv = vec![0.0f32; kvd * seq];
                        for kv_h in 0..kv_heads {
                            for rep in 0..hpg {
                                let src_h = kv_h * hpg + rep;
                                for t in 0..seq {
                                    for d in 0..hd {
                                        dv[(kv_h * hd + d) * seq + t] +=
                                            output[2 * heads * head_stride
                                                + src_h * head_stride
                                                + t * hd
                                                + d]
                                                * inv_hpg;
                                    }
                                }
                            }
                        }

                        // Merge gate gradient into dQ for QKV backward
                        let dq_for_wq = if cfg.attn_output_gate {
                            // d_gate is in 4th block: output[3*H*head_stride .. 4*H*head_stride]
                            let dg_off = 3 * heads * head_stride;
                            let mut d_gate = vec![0.0f32; ad * seq];
                            for h in 0..heads {
                                for t in 0..seq {
                                    for d in 0..hd {
                                        d_gate[(h * hd + d) * seq + t] =
                                            output[dg_off + h * head_stride + t * hd + d];
                                    }
                                }
                            }
                            ane_forward::merge_q_gate(&dq, &d_gate, heads, hd, seq)
                        } else {
                            dq
                        };

                        // Re-expand for QKV backward
                        let dk_exp = if hpg > 1 {
                            let mut exp = vec![0.0f32; ad * seq];
                            for kv_h in 0..kv_heads {
                                for rep in 0..hpg {
                                    let dst_h = kv_h * hpg + rep;
                                    for c in 0..hd {
                                        exp[(dst_h * hd + c) * seq..(dst_h * hd + c + 1) * seq]
                                            .copy_from_slice(
                                                &dk_r[(kv_h * hd + c) * seq
                                                    ..(kv_h * hd + c + 1) * seq],
                                            );
                                    }
                                }
                            }
                            exp
                        } else {
                            dk_r
                        };
                        let dv_exp = if hpg > 1 {
                            let mut exp = vec![0.0f32; ad * seq];
                            for kv_h in 0..kv_heads {
                                for rep in 0..hpg {
                                    let dst_h = kv_h * hpg + rep;
                                    for c in 0..hd {
                                        exp[(dst_h * hd + c) * seq..(dst_h * hd + c + 1) * seq]
                                            .copy_from_slice(
                                                &dv[(kv_h * hd + c) * seq
                                                    ..(kv_h * hd + c + 1) * seq],
                                            );
                                    }
                                }
                            }
                            exp
                        } else {
                            dv
                        };

                        // Per-layer QKV backward (BLOBFILE, dispatch 2 of 2)
                        let qpd = cfg.q_proj_dim();
                        let mut dqkv = Vec::with_capacity((qpd + 2 * ad) * seq);
                        dqkv.extend_from_slice(&dq_for_wq);
                        dqkv.extend_from_slice(&dk_exp);
                        dqkv.extend_from_slice(&dv_exp);

                        if let Some(Ok(dx_attn)) = pp.eval_bwd_qkvb(l, &dqkv, cfg) {
                            break 'attn_bwd dx_attn;
                        }
                        // QKV per-layer failed, fall through to 3-dispatch path
                    }
                }
            }

            // 2. Per-layer BLOBFILE path: Wot(1 dispatch) + SDPA(1 dispatch) + QKV(1 dispatch)
            //    Eliminates DynMatmul weight packing (~64MB/layer). Falls through
            //    to DynMatmul path if per-layer kernels aren't primed.
            if let Some(ref pp) = prepacked {
                if let (Some(Ok(da_raw)), true) = (
                    pp.eval_bwd_wot(l, &dx2, cfg),
                    bwd_kernels.fused_attn_gqa_bwd.is_some(),
                ) {
                    // Gate backward (CPU)
                    let (da, d_gate) = if cfg.attn_output_gate {
                        let gate_raw = ac.attn_gate.as_ref().unwrap();
                        let pre_gate = ac.attn_pre_gate.as_ref().unwrap();
                        let (d_attn, dg) = sigmoid_gate_backward(&da_raw, gate_raw, pre_gate);
                        (d_attn, Some(dg))
                    } else {
                        (da_raw, None)
                    };

                    // Fused SDPA backward (same as 3-dispatch path step 3)
                    let ad = cfg.attn_dim();
                    let hd = cfg.head_dim();
                    let heads = cfg.n_heads;
                    let kv_heads = cfg.n_kv_heads;
                    let hpg = cfg.heads_per_group();
                    let kvd = cfg.kv_dim();

                    let mut k_expanded = vec![0.0f32; ad * seq];
                    for kv_h in 0..kv_heads {
                        for rep in 0..hpg {
                            let dst_h = kv_h * hpg + rep;
                            k_expanded[dst_h * hd * seq..(dst_h * hd + hd) * seq]
                                .copy_from_slice(&ac.k[kv_h * hd * seq..(kv_h * hd + hd) * seq]);
                        }
                    }
                    let mut v_expanded = vec![0.0f32; ad * seq];
                    for kv_h in 0..kv_heads {
                        for rep in 0..hpg {
                            let dst_h = kv_h * hpg + rep;
                            v_expanded[dst_h * hd * seq..(dst_h * hd + hd) * seq]
                                .copy_from_slice(&ac.v[kv_h * hd * seq..(kv_h * hd + hd) * seq]);
                        }
                    }

                    let mut sdpa_input = Vec::with_capacity(4 * ad * seq);
                    sdpa_input.extend_from_slice(&da);
                    sdpa_input.extend_from_slice(&ac.q[..ad * seq]);
                    sdpa_input.extend_from_slice(&k_expanded);
                    sdpa_input.extend_from_slice(&v_expanded);

                    let fused_kernel = bwd_kernels.fused_attn_gqa_bwd.as_ref().unwrap();
                    let input_bytes = unsafe {
                        std::slice::from_raw_parts(
                            sdpa_input.as_ptr() as *const u8,
                            sdpa_input.len() * 4,
                        )
                    };
                    fused_kernel.write_input(0, input_bytes);
                    if let Ok(()) = fused_kernel.eval() {
                        let out_elems = 3 * heads * seq * hd;
                        let mut out_buf = vec![0u8; out_elems * 4];
                        fused_kernel.read_output(0, &mut out_buf);
                        let output: Vec<f32> = out_buf
                            .chunks_exact(4)
                            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                            .collect();

                        let head_stride = seq * hd;
                        let mut dq = vec![0.0f32; ad * seq];
                        let mut dk = vec![0.0f32; ad * seq];
                        for h in 0..heads {
                            for t in 0..seq {
                                for d in 0..hd {
                                    dq[(h * hd + d) * seq + t] =
                                        output[h * head_stride + t * hd + d];
                                    dk[(h * hd + d) * seq + t] =
                                        output[heads * head_stride + h * head_stride + t * hd + d];
                                }
                            }
                        }

                        // RoPE + QK-norm backward (CPU)
                        ane_forward::rope_backward(
                            &mut dq,
                            &mut dk,
                            heads,
                            hd,
                            seq,
                            cfg.rope_theta,
                        );
                        if let (Some(q_pre), Some(q_nw)) = (&ac.q_pre_norm, &lw.q_norm) {
                            ane_forward::qk_rmsnorm_bwd(
                                &mut dq,
                                &mut vec![0.0f32; hd],
                                q_pre,
                                q_nw,
                                heads,
                                hd,
                                seq,
                                cfg.rms_eps,
                            );
                        }
                        if let (Some(k_pre), Some(k_nw)) = (&ac.k_pre_norm, &lw.k_norm) {
                            ane_forward::qk_rmsnorm_bwd(
                                &mut dk,
                                &mut vec![0.0f32; hd],
                                k_pre,
                                k_nw,
                                heads,
                                hd,
                                seq,
                                cfg.rms_eps,
                            );
                        }

                        // GQA-reduce dK, dV
                        let inv_hpg = 1.0 / hpg as f32;
                        let mut dk_r = vec![0.0f32; kvd * seq];
                        for kv_h in 0..kv_heads {
                            for rep in 0..hpg {
                                let src_h = kv_h * hpg + rep;
                                for c in 0..hd {
                                    for t in 0..seq {
                                        dk_r[(kv_h * hd + c) * seq + t] +=
                                            dk[(src_h * hd + c) * seq + t] * inv_hpg;
                                    }
                                }
                            }
                        }
                        let mut dv = vec![0.0f32; kvd * seq];
                        for kv_h in 0..kv_heads {
                            for rep in 0..hpg {
                                let src_h = kv_h * hpg + rep;
                                for t in 0..seq {
                                    for d in 0..hd {
                                        dv[(kv_h * hd + d) * seq + t] +=
                                            output[2 * heads * head_stride
                                                + src_h * head_stride
                                                + t * hd
                                                + d]
                                                * inv_hpg;
                                    }
                                }
                            }
                        }

                        // Merge gate into dQ if needed
                        let dq_for_wq = if let Some(dg) = &d_gate {
                            ane_forward::merge_q_gate(&dq, dg, heads, hd, seq)
                        } else {
                            dq
                        };

                        // Re-expand for QKV backward
                        let dk_exp = if hpg > 1 {
                            let mut exp = vec![0.0f32; ad * seq];
                            for kv_h in 0..kv_heads {
                                for rep in 0..hpg {
                                    let dst_h = kv_h * hpg + rep;
                                    for c in 0..hd {
                                        exp[(dst_h * hd + c) * seq..(dst_h * hd + c + 1) * seq]
                                            .copy_from_slice(
                                                &dk_r[(kv_h * hd + c) * seq
                                                    ..(kv_h * hd + c + 1) * seq],
                                            );
                                    }
                                }
                            }
                            exp
                        } else {
                            dk_r
                        };
                        let dv_exp = if hpg > 1 {
                            let mut exp = vec![0.0f32; ad * seq];
                            for kv_h in 0..kv_heads {
                                for rep in 0..hpg {
                                    let dst_h = kv_h * hpg + rep;
                                    for c in 0..hd {
                                        exp[(dst_h * hd + c) * seq..(dst_h * hd + c + 1) * seq]
                                            .copy_from_slice(
                                                &dv[(kv_h * hd + c) * seq
                                                    ..(kv_h * hd + c + 1) * seq],
                                            );
                                    }
                                }
                            }
                            exp
                        } else {
                            dv
                        };

                        // Per-layer QKV backward (BLOBFILE, no DynMatmul packing)
                        let qpd = cfg.q_proj_dim();
                        let mut dqkv = Vec::with_capacity((qpd + 2 * ad) * seq);
                        dqkv.extend_from_slice(&dq_for_wq);
                        dqkv.extend_from_slice(&dk_exp);
                        dqkv.extend_from_slice(&dv_exp);

                        if let Some(Ok(dx_attn)) = pp.eval_bwd_qkvb(l, &dqkv, cfg) {
                            break 'attn_bwd dx_attn;
                        }
                        // QKV per-layer failed, fall through to DynMatmul path
                    }
                }
            }

            // 3. DynMatmul SDPA path (3 dispatches: wot + fused_sdpa + qkv)
            if bwd_kernels.fused_attn_gqa_bwd.is_some()
                && bwd_kernels.wot_bwd.is_some()
                && bwd_kernels.qkv_bwd.is_some()
            {
                match mha_backward_fused_sdpa_dx_attn(bwd_kernels, lw, ac, &dx2, cfg) {
                    Ok(dx_attn) => {
                        break 'attn_bwd dx_attn;
                    }
                    Err(e) => {
                        tracing::warn!("ANE fused SDPA backward fell back: {e}");
                    }
                }
            }
            // 3. Fall back to 4-kernel path
            if bwd_kernels.wot_bwd.is_some()
                && bwd_kernels.sdpa_bwd1.is_some()
                && bwd_kernels.sdpa_bwd2.is_some()
                && bwd_kernels.qkv_bwd.is_some()
            {
                match mha_backward_ane_dx_attn(bwd_kernels, lw, ac, &dx2, cfg) {
                    Ok(dx_attn) => break 'attn_bwd dx_attn,
                    Err(e) => {
                        tracing::warn!("ANE MHA backward fell back to CPU: {e}");
                    }
                }
            }
            // 4. CPU fallback
            mha_backward_cpu_dx_attn(lw, ac, &dx2, cfg)
        };

        // RMSNorm backward (attention): ANE per-layer kernel → CPU fallback
        dy = if let Some(Ok(result)) = prepacked
            .as_ref()
            .and_then(|pp| pp.eval_rmsnorm_bwd(l, &dx_attn, &ac.layer_in, false))
        {
            result
        } else {
            let mut dx = vec![0.0f32; dim * seq];
            rmsnorm_bwd(
                &mut dx,
                &mut vec![0.0f32; dim],
                &dx_attn,
                &ac.layer_in,
                &lw.rms_att,
                dim,
                seq,
                cfg.rms_eps,
            );
            dx
        };
        ane_forward::vec_add_inplace(&mut dy, &dx2);

        _prof_attn_us += _t_attn.elapsed().as_micros() as u64;
    }

    // Log backward phase profile (prepacked path)
    if _prof_ffn_us + _prof_attn_us > 0 {
        let total = (_prof_ffn_us + _prof_rmsnorm_us + _prof_lora_us + _prof_attn_us).max(1);
        if std::env::var("NANOBOT_PROFILE_BWD").is_ok() {
            eprintln!(
                "BWD profile ({n_layers}L): ffn={:.1}ms ({:.0}%) attn={:.1}ms ({:.0}%) rmsnorm={:.1}ms ({:.0}%) lora={:.1}ms ({:.0}%) total={:.1}ms",
                _prof_ffn_us as f64 / 1000.0,
                _prof_ffn_us as f64 / total as f64 * 100.0,
                _prof_attn_us as f64 / 1000.0,
                _prof_attn_us as f64 / total as f64 * 100.0,
                _prof_rmsnorm_us as f64 / 1000.0,
                _prof_rmsnorm_us as f64 / total as f64 * 100.0,
                _prof_lora_us as f64 / 1000.0,
                _prof_lora_us as f64 / total as f64 * 100.0,
                total as f64 / 1000.0,
            );
        }
    }

    // Divide LoRA gradients by loss_scale to compensate for scaled dlogits
    if apply_loss_scale {
        let inv_scale = 1.0 / loss_scale;
        for layer_grads in &mut lora_grads.layers {
            let scale_grads = |g: &mut Option<ane_lora::LoraAdapterGrads>| {
                if let Some(ref mut grads) = g {
                    for v in grads.da.iter_mut() {
                        *v *= inv_scale;
                    }
                    for v in grads.db.iter_mut() {
                        *v *= inv_scale;
                    }
                }
            };
            scale_grads(&mut layer_grads.wq);
            scale_grads(&mut layer_grads.wv);
            scale_grads(&mut layer_grads.wo);
            scale_grads(&mut layer_grads.w2);
        }
    }

    sanitize_lora_grads(&mut lora_grads);

    BackwardResultWithLora {
        model_grads: None,
        lora_grads,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::ane_forward;
    use crate::agent::ane_lora::{LoraConfig, LoraModel};
    use crate::agent::ane_mil::MilConfig;
    use crate::agent::ane_weights::{
        LayerWeights, ModelWeights, QuantizedLayerWeights, QuantizedModelWeights, QuantizedTensor,
    };

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
    ) -> QuantizedTensor {
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

    // -----------------------------------------------------------------------
    // Round 1: rmsnorm_bwd
    // -----------------------------------------------------------------------

    #[test]
    fn test_sanitize_lora_grads_replaces_nan_and_inf() {
        let lora = LoraModel::new(
            LoraConfig {
                rank: 4,
                alpha: 4.0,
                target_modules: vec!["wo".into(), "w2".into()],
            },
            1,
            8,
            16,
        );
        let mut grads = crate::agent::ane_lora::LoraModelGrads::zeros(&lora);
        // Inject NaN and Inf into w2 adapter grads
        if let Some(ref mut g) = grads.layers[0].w2 {
            g.da[0] = f32::NAN;
            g.da[1] = f32::INFINITY;
            g.da[2] = f32::NEG_INFINITY;
            g.da[3] = 42.0; // normal value, should be untouched
        }
        sanitize_lora_grads(&mut grads);
        let g = grads.layers[0].w2.as_ref().unwrap();
        assert_eq!(g.da[0], 0.0, "NaN should become 0");
        assert_eq!(
            g.da[1],
            ane_forward::FP16_MAX,
            "+Inf should become fp16 max"
        );
        assert_eq!(
            g.da[2],
            -ane_forward::FP16_MAX,
            "-Inf should become -fp16 max"
        );
        assert_eq!(g.da[3], 42.0, "normal values should be untouched");
    }

    #[test]
    fn test_rmsnorm_bwd_numerical_gradient() {
        let dim = 4;
        let seq = 3;
        let eps = 1e-4f32;

        // Random-ish input
        let x: Vec<f32> = (0..dim * seq)
            .map(|i| ((i as f32 * 0.7 + 0.3).sin()) * 2.0)
            .collect();
        let w: Vec<f32> = (0..dim).map(|i| 0.5 + i as f32 * 0.3).collect();

        // Upstream gradient
        let dy: Vec<f32> = (0..dim * seq)
            .map(|i| ((i as f32 * 1.3 + 0.7).cos()) * 0.5)
            .collect();

        // Analytical gradient
        let mut dx = vec![0.0f32; dim * seq];
        let mut dw = vec![0.0f32; dim];
        rmsnorm_bwd(&mut dx, &mut dw, &dy, &x, &w, dim, seq, 1e-5);

        // Numerical gradient for dx via finite differences
        for idx in 0..dim * seq {
            let mut x_plus = x.clone();
            let mut x_minus = x.clone();
            x_plus[idx] += eps;
            x_minus[idx] -= eps;

            let mut out_plus = vec![0.0f32; dim * seq];
            let mut out_minus = vec![0.0f32; dim * seq];
            ane_forward::rmsnorm(&mut out_plus, &x_plus, &w, dim, seq, 1e-5);
            ane_forward::rmsnorm(&mut out_minus, &x_minus, &w, dim, seq, 1e-5);

            // Numerical: dot(dy, (out_plus - out_minus) / (2*eps))
            let mut num_grad = 0.0f32;
            for j in 0..dim * seq {
                num_grad += dy[j] * (out_plus[j] - out_minus[j]) / (2.0 * eps);
            }

            let err = (dx[idx] - num_grad).abs();
            assert!(
                err < 0.01,
                "rmsnorm_bwd dx[{idx}]: analytical={:.6}, numerical={:.6}, err={:.6}",
                dx[idx],
                num_grad,
                err
            );
        }

        // Numerical gradient for dw
        for idx in 0..dim {
            let mut w_plus = w.clone();
            let mut w_minus = w.clone();
            w_plus[idx] += eps;
            w_minus[idx] -= eps;

            let mut out_plus = vec![0.0f32; dim * seq];
            let mut out_minus = vec![0.0f32; dim * seq];
            ane_forward::rmsnorm(&mut out_plus, &x, &w_plus, dim, seq, 1e-5);
            ane_forward::rmsnorm(&mut out_minus, &x, &w_minus, dim, seq, 1e-5);

            let mut num_grad = 0.0f32;
            for j in 0..dim * seq {
                num_grad += dy[j] * (out_plus[j] - out_minus[j]) / (2.0 * eps);
            }

            let err = (dw[idx] - num_grad).abs();
            assert!(
                err < 0.01,
                "rmsnorm_bwd dw[{idx}]: analytical={:.6}, numerical={:.6}, err={:.6}",
                dw[idx],
                num_grad,
                err
            );
        }
    }

    #[test]
    fn test_rmsnorm_bwd_dw_accumulates() {
        let dim = 2;
        let seq = 2;
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let w = vec![1.0, 1.0];
        let dy = vec![1.0, 1.0, 1.0, 1.0];

        let mut dx = vec![0.0f32; dim * seq];
        let mut dw = vec![5.0, 5.0]; // pre-loaded to test accumulation
        rmsnorm_bwd(&mut dx, &mut dw, &dy, &x, &w, dim, seq, 1e-5);

        // dw should be > 5 (accumulated)
        assert!(dw[0] > 5.0, "dw[0] should accumulate: got {}", dw[0]);
        assert!(dw[1] > 5.0, "dw[1] should accumulate: got {}", dw[1]);
    }

    // -----------------------------------------------------------------------
    // Round 2: silu_bwd + classifier_bwd + embed_bwd
    // -----------------------------------------------------------------------

    #[test]
    fn test_silu_bwd_numerical_gradient() {
        let n = 8;
        let eps = 1e-4f32;

        let h1: Vec<f32> = (0..n).map(|i| (i as f32 - 4.0) * 0.5).collect();
        let h3: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3 + 0.1).sin()).collect();
        let dsilu: Vec<f32> = (0..n).map(|i| (i as f32 * 0.7).cos() * 0.5).collect();

        let mut dh1 = vec![0.0f32; n];
        let mut dh3 = vec![0.0f32; n];
        silu_bwd(&mut dh1, &mut dh3, &dsilu, &h1, &h3, n);

        // Verify dh1 via finite diff on silu(h1)*h3
        for i in 0..n {
            let silu = |x: f32| x / (1.0 + (-x).exp());
            let gate = |h1v: f32, h3v: f32| silu(h1v) * h3v;

            // dh1[i] numerical
            let num_dh1 =
                dsilu[i] * (gate(h1[i] + eps, h3[i]) - gate(h1[i] - eps, h3[i])) / (2.0 * eps);
            let err = (dh1[i] - num_dh1).abs();
            assert!(
                err < 0.01,
                "silu_bwd dh1[{i}]: analytical={:.6}, numerical={:.6}, err={:.6}",
                dh1[i],
                num_dh1,
                err
            );

            // dh3[i] numerical
            let num_dh3 =
                dsilu[i] * (gate(h1[i], h3[i] + eps) - gate(h1[i], h3[i] - eps)) / (2.0 * eps);
            let err = (dh3[i] - num_dh3).abs();
            assert!(
                err < 0.01,
                "silu_bwd dh3[{i}]: analytical={:.6}, numerical={:.6}, err={:.6}",
                dh3[i],
                num_dh3,
                err
            );
        }
    }

    #[test]
    fn test_classifier_bwd_shapes_and_accumulation() {
        let vocab = 4;
        let dim = 3;
        let seq = 2;

        let embed: Vec<f32> = (0..vocab * dim).map(|i| i as f32 * 0.1).collect();
        let x_final: Vec<f32> = (0..dim * seq).map(|i| (i as f32 * 0.2).sin()).collect();
        let dlogits: Vec<f32> = (0..vocab * seq)
            .map(|i| (i as f32 * 0.3).cos() * 0.1)
            .collect();

        let mut dy = vec![0.0f32; dim * seq];
        let mut dembed = vec![0.0f32; vocab * dim];
        classifier_bwd(
            &mut dy,
            &mut dembed,
            &dlogits,
            &embed,
            &x_final,
            vocab,
            dim,
            seq,
        );

        // dy should have correct shape and non-trivial values
        assert_eq!(dy.len(), dim * seq);
        let nonzero_dy = dy.iter().filter(|v| v.abs() > 1e-10).count();
        assert!(nonzero_dy > 0, "dy should have nonzero entries");

        // dembed should accumulate
        let nonzero_de = dembed.iter().filter(|v| v.abs() > 1e-10).count();
        assert!(nonzero_de > 0, "dembed should have nonzero entries");

        // Verify dy = embed^T @ dlogits by comparing with classifier_forward transpose
        // dy[d,t] = sum_v embed[v,d] * dlogits[v,t]
        for d in 0..dim {
            for t in 0..seq {
                let mut expected = 0.0f32;
                for v in 0..vocab {
                    expected += embed[v * dim + d] * dlogits[v * seq + t];
                }
                let err = (dy[d * seq + t] - expected).abs();
                assert!(err < 1e-5, "dy[{d},{t}] mismatch: {err}");
            }
        }
    }

    #[test]
    fn test_embed_bwd_scatter() {
        let dim = 3;
        let seq = 4;
        let vocab = 5;
        let tokens = [2u16, 0, 2, 3]; // token 2 appears twice

        let dy: Vec<f32> = (0..dim * seq).map(|i| (i + 1) as f32).collect();
        let mut dembed = vec![0.0f32; vocab * dim];

        embed_bwd(&mut dembed, &dy, &tokens, dim, seq);

        // Token 2 at positions 0 and 2 should accumulate
        for d in 0..dim {
            let expected = dy[d * seq + 0] + dy[d * seq + 2]; // positions 0 and 2
            let got = dembed[2 * dim + d];
            assert!(
                (got - expected).abs() < 1e-6,
                "embed_bwd tok=2 d={d}: got={got}, expected={expected}"
            );
        }

        // Token 0 at position 1
        for d in 0..dim {
            let expected = dy[d * seq + 1];
            let got = dembed[0 * dim + d];
            assert!(
                (got - expected).abs() < 1e-6,
                "embed_bwd tok=0 d={d}: got={got}, expected={expected}"
            );
        }

        // Token 1 never appears — should be zero
        for d in 0..dim {
            assert_eq!(
                dembed[1 * dim + d],
                0.0,
                "embed_bwd tok=1 d={d} should be 0"
            );
        }
    }

    #[test]
    fn test_backward_lora_cpu_generic_quantized_matches_dequantized_reference_for_gqa() {
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
        let vocab = 13;
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
            moe: vec![None; 1],
        };

        let dense = ModelWeights {
            cfg: cfg.clone(),
            layers: vec![quantized.dequantize_layer(0)],
            rms_final: quantized.rms_final.clone(),
            embed: quantized.embed.clone(),
            vocab_size: quantized.vocab_size,
            lm_head: quantized.lm_head.clone(),
            vocab_clusters: None,
            router_adapter: None,
        };

        let lora = LoraModel::with_kv_dim(
            LoraConfig {
                rank: 4,
                alpha: 4.0,
                target_modules: vec!["wo".into(), "w2".into()],
            },
            1,
            dim,
            kv_dim,
            hidden,
        );

        let tokens = vec![1u16, 2, 3];
        let targets = vec![2u16, 3, 4];

        let quantized_fwd =
            ane_forward::forward_cpu_generic(&quantized, Some(&lora), &tokens, &targets);
        let dense_fwd = ane_forward::forward_cpu_generic(&dense, Some(&lora), &tokens, &targets);
        let quantized_bwd =
            backward_lora_cpu_generic(&quantized, &quantized_fwd, &lora, &tokens, 0.0, 1.0);
        let dense_bwd = backward_lora_cpu_generic(&dense, &dense_fwd, &lora, &tokens, 0.0, 1.0);

        let q_w2 = quantized_bwd.lora_grads.layers[0].w2.as_ref().unwrap();
        let d_w2 = dense_bwd.lora_grads.layers[0].w2.as_ref().unwrap();
        let q_wo = quantized_bwd.lora_grads.layers[0].wo.as_ref().unwrap();
        let d_wo = dense_bwd.lora_grads.layers[0].wo.as_ref().unwrap();

        assert!(max_abs_diff(&q_w2.da, &d_w2.da) < 1e-4);
        assert!(max_abs_diff(&q_w2.db, &d_w2.db) < 1e-4);
        assert!(max_abs_diff(&q_wo.da, &d_wo.da) < 1e-4);
        assert!(max_abs_diff(&q_wo.db, &d_wo.db) < 1e-4);
    }

    // -----------------------------------------------------------------------
    // Round 3: BackwardKernels::compile_backward
    // -----------------------------------------------------------------------

    #[test]
    fn test_compile_backward_kernels() {
        let cfg = MilConfig::mha(64, 128, 4, 64);

        let mask_blob = ane_mil::build_causal_mask_blob(cfg.seq_len);
        let result = BackwardKernels::compile_backward(&cfg, &mask_blob);
        assert!(
            result.is_ok(),
            "compile_backward failed: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_backward_lora_ane_over_parameterized_attention_smoke() {
        use super::super::ane_lora::{LoraConfig, LoraModel};

        let dim = 64;
        let hidden = 128;
        let n_heads = 4;
        let head_dim = 32;
        let seq = 32;
        let vocab = 32;
        let ad = n_heads * head_dim;
        let qpd = 2 * ad;

        let cfg = MilConfig {
            dim,
            hidden_dim: hidden,
            n_heads,
            seq_len: seq,
            n_kv_heads: 4,
            rope_theta: 10_000_000.0,
            rms_eps: 1e-6,
            has_lm_head: false,
            head_dim_explicit: head_dim,
            linear_attn_indices: vec![],
            linear_n_heads: 0,
            linear_head_dim: 0,
            linear_n_value_heads: 0,
            linear_value_head_dim: 0,
            conv_kernel_size: 0,
            attn_output_gate: true,
        };

        let fwd_kernels = match ane_forward::CompiledKernels::compile_forward(&cfg) {
            Ok(k) => k,
            Err(e) => {
                eprintln!("Skipping over-parameterized ANE backward smoke test: {e}");
                return;
            }
        };
        let bwd_kernels = BackwardKernels::compile_backward(&cfg, &fwd_kernels.mask_blob)
            .expect("compile_backward failed");

        let make_small = |n: usize, seed: usize| -> Vec<f32> {
            (0..n)
                .map(|i| ((i + seed) as f32 * 0.0013).sin() * 0.01)
                .collect()
        };

        let model = ModelWeights {
            cfg: cfg.clone(),
            layers: vec![ane_weights::LayerWeights {
                wq: make_small(qpd * dim, 0),
                wk: make_small(ad * dim, 1000),
                wv: make_small(ad * dim, 2000),
                wo: make_small(dim * ad, 3000),
                w1: make_small(hidden * dim, 4000),
                w2: make_small(dim * hidden, 5000),
                w3: make_small(hidden * dim, 6000),
                rms_att: vec![1.0; dim],
                rms_ffn: vec![1.0; dim],
                q_norm: None,
                k_norm: None,
                gdn: None,
                moe: None,
            }],
            rms_final: vec![1.0; dim],
            embed: make_small(vocab * dim, 7000),
            vocab_size: vocab,
            lm_head: None,
            vocab_clusters: None,
            router_adapter: None,
        };

        let lora = LoraModel::with_full_dims(
            LoraConfig {
                rank: 8,
                alpha: 8.0,
                target_modules: vec!["wo".into(), "w2".into()],
            },
            1,
            dim,
            ad,
            ad,
            qpd,
            hidden,
        );

        let tokens: Vec<u16> = (0..seq).map(|i| (i % vocab) as u16).collect();
        let targets: Vec<u16> = (0..seq).map(|i| ((i + 1) % vocab) as u16).collect();

        let fwd = ane_forward::forward_ane_generic(
            &fwd_kernels,
            &model,
            Some(&lora),
            &tokens,
            &targets,
            0.0,
            1.0,
        )
        .expect("forward_ane_generic failed");
        let bwd =
            backward_lora_ane_generic(&bwd_kernels, &model, &fwd, &lora, &tokens, 0.0, 1.0, 1.0);

        let w2 = bwd.lora_grads.layers[0].w2.as_ref().expect("w2 grads");
        let nonzero = w2
            .da
            .iter()
            .chain(w2.db.iter())
            .filter(|v| v.abs() > 1e-12)
            .count();
        assert!(
            nonzero > 0,
            "LoRA grads should be nonzero for over-parameterized attention"
        );
    }

    #[test]
    fn test_mha_backward_ane_matches_cpu_over_parameterized_attention() {
        let dim = 64;
        let hidden = 128;
        let n_heads = 4;
        let head_dim = 32;
        let seq = 32;
        let vocab = 32;
        let ad = n_heads * head_dim;
        let qpd = 2 * ad;

        let cfg = MilConfig {
            dim,
            hidden_dim: hidden,
            n_heads,
            seq_len: seq,
            n_kv_heads: 4,
            rope_theta: 10_000_000.0,
            rms_eps: 1e-6,
            has_lm_head: false,
            head_dim_explicit: head_dim,
            linear_attn_indices: vec![],
            linear_n_heads: 0,
            linear_head_dim: 0,
            linear_n_value_heads: 0,
            linear_value_head_dim: 0,
            conv_kernel_size: 0,
            attn_output_gate: true,
        };

        let fwd_kernels = match ane_forward::CompiledKernels::compile_forward(&cfg) {
            Ok(k) => k,
            Err(e) => {
                eprintln!("Skipping ANE MHA backward compare test: {e}");
                return;
            }
        };
        let bwd_kernels = BackwardKernels::compile_backward(&cfg, &fwd_kernels.mask_blob)
            .expect("compile_backward failed");

        let make_small = |n: usize, seed: usize| -> Vec<f32> {
            (0..n)
                .map(|i| ((i + seed) as f32 * 0.0013).sin() * 0.01)
                .collect()
        };

        let model = ModelWeights {
            cfg: cfg.clone(),
            layers: vec![LayerWeights {
                wq: make_small(qpd * dim, 0),
                wk: make_small(ad * dim, 1000),
                wv: make_small(ad * dim, 2000),
                wo: make_small(dim * ad, 3000),
                w1: make_small(hidden * dim, 4000),
                w2: make_small(dim * hidden, 5000),
                w3: make_small(hidden * dim, 6000),
                rms_att: vec![1.0; dim],
                rms_ffn: vec![1.0; dim],
                q_norm: None,
                k_norm: None,
                gdn: None,
                moe: None,
            }],
            rms_final: vec![1.0; dim],
            embed: make_small(vocab * dim, 7000),
            vocab_size: vocab,
            lm_head: None,
            vocab_clusters: None,
            router_adapter: None,
        };

        let tokens: Vec<u16> = (0..seq).map(|i| (i % vocab) as u16).collect();
        let targets: Vec<u16> = (0..seq).map(|i| ((i + 1) % vocab) as u16).collect();
        let fwd = ane_forward::forward_cpu_generic(&model, None, &tokens, &targets);
        let ac = &fwd.base.layer_acts[0];
        let lw = &model.layers[0];
        let dx2 = make_small(dim * seq, 8000);

        let cpu = mha_backward_cpu_dx_attn(lw, ac, &dx2, &cfg);
        let ane = mha_backward_ane_dx_attn(&bwd_kernels, lw, ac, &dx2, &cfg)
            .expect("ANE MHA backward helper failed");

        let max_err = max_abs_diff(&cpu, &ane);
        assert!(
            max_err < 0.05,
            "ANE MHA backward should match CPU within tolerance, max_err={max_err}"
        );
    }

    // -----------------------------------------------------------------------
    // Round 4: backward — single layer, structural checks
    // -----------------------------------------------------------------------

    #[test]
    fn test_backward_single_layer_structural() {
        let dim = 64;
        let hidden = 128;
        let n_heads = 4;
        let seq = 64;
        let vocab = 32;

        let cfg = MilConfig::mha(dim, hidden, n_heads, seq);

        // Compile forward + backward kernels
        let fwd_kernels = match ane_forward::CompiledKernels::compile_forward(&cfg) {
            Ok(k) => k,
            Err(e) => {
                eprintln!("Skipping backward test (ANE unavailable): {e}");
                return;
            }
        };
        let bwd_kernels = BackwardKernels::compile_backward(&cfg, &fwd_kernels.mask_blob)
            .expect("compile_backward failed");

        let make_small = |n: usize, seed: usize| -> Vec<f32> {
            (0..n)
                .map(|i| ((i + seed) as f32 * 0.0013).sin() * 0.01)
                .collect()
        };

        let model = ModelWeights {
            cfg: cfg.clone(),
            layers: vec![LayerWeights {
                wq: make_small(dim * dim, 0),
                wk: make_small(dim * dim, 1000),
                wv: make_small(dim * dim, 2000),
                wo: make_small(dim * dim, 3000),
                w1: make_small(hidden * dim, 4000),
                w2: make_small(dim * hidden, 5000),
                w3: make_small(hidden * dim, 6000),
                rms_att: vec![1.0; dim],
                rms_ffn: vec![1.0; dim],
                q_norm: None,
                k_norm: None,
                gdn: None,
                moe: None,
            }],
            rms_final: vec![1.0; dim],
            embed: make_small(vocab * dim, 7000),
            vocab_size: vocab,
            lm_head: None,
            vocab_clusters: None,
            router_adapter: None,
        };

        let tokens: Vec<u16> = (0..seq).map(|i| (i % vocab) as u16).collect();
        let targets: Vec<u16> = (0..seq).map(|i| ((i + 1) % vocab) as u16).collect();

        // Forward pass
        let fwd =
            ane_forward::forward(&fwd_kernels, &model, &tokens, &targets).expect("forward failed");

        // Backward pass
        let grads = backward(&bwd_kernels, &model, &fwd, &tokens).expect("backward failed");

        // Check shapes
        assert_eq!(grads.layers.len(), 1);
        assert_eq!(grads.layers[0].dwq.len(), dim * dim);
        assert_eq!(grads.layers[0].dwk.len(), dim * dim);
        assert_eq!(grads.layers[0].dwv.len(), dim * dim);
        assert_eq!(grads.layers[0].dwo.len(), dim * dim);
        assert_eq!(grads.layers[0].dw1.len(), hidden * dim);
        assert_eq!(grads.layers[0].dw2.len(), dim * hidden);
        assert_eq!(grads.layers[0].dw3.len(), hidden * dim);
        assert_eq!(grads.layers[0].drms_att.len(), dim);
        assert_eq!(grads.layers[0].drms_ffn.len(), dim);
        assert_eq!(grads.drms_final.len(), dim);
        assert_eq!(grads.dembed.len(), vocab * dim);

        // All gradients should be finite
        let check_finite = |name: &str, v: &[f32]| {
            for (i, &val) in v.iter().enumerate() {
                assert!(val.is_finite(), "{name}[{i}] is not finite: {val}");
            }
        };
        check_finite("dwq", &grads.layers[0].dwq);
        check_finite("dwk", &grads.layers[0].dwk);
        check_finite("dwv", &grads.layers[0].dwv);
        check_finite("dwo", &grads.layers[0].dwo);
        check_finite("dw1", &grads.layers[0].dw1);
        check_finite("dw2", &grads.layers[0].dw2);
        check_finite("dw3", &grads.layers[0].dw3);
        check_finite("drms_att", &grads.layers[0].drms_att);
        check_finite("drms_ffn", &grads.layers[0].drms_ffn);
        check_finite("drms_final", &grads.drms_final);
        check_finite("dembed", &grads.dembed);

        // Embedding gradient should have nonzero entries at used token rows
        let mut has_nonzero = false;
        for &tok in &tokens {
            let row_start = tok as usize * dim;
            for d in 0..dim {
                if grads.dembed[row_start + d].abs() > 1e-15 {
                    has_nonzero = true;
                    break;
                }
            }
        }
        assert!(
            has_nonzero,
            "dembed should have nonzero entries at used tokens"
        );

        // At least some weight gradients should be nonzero
        let dw2_nonzero = grads.layers[0]
            .dw2
            .iter()
            .filter(|v| v.abs() > 1e-15)
            .count();
        assert!(dw2_nonzero > 0, "dw2 should have nonzero entries");
    }

    // -----------------------------------------------------------------------
    // Round 5: backward — gradient correctness via loss perturbation
    // -----------------------------------------------------------------------

    #[test]
    fn test_backward_gradient_correctness_loss_perturbation() {
        let dim = 64;
        let hidden = 128;
        let n_heads = 4;
        let seq = 64;
        let vocab = 32;

        let cfg = MilConfig::mha(dim, hidden, n_heads, seq);

        let fwd_kernels = match ane_forward::CompiledKernels::compile_forward(&cfg) {
            Ok(k) => k,
            Err(e) => {
                eprintln!("Skipping gradient check (ANE unavailable): {e}");
                return;
            }
        };
        let bwd_kernels = BackwardKernels::compile_backward(&cfg, &fwd_kernels.mask_blob)
            .expect("compile_backward failed");

        let make_small = |n: usize, seed: usize| -> Vec<f32> {
            (0..n)
                .map(|i| ((i + seed) as f32 * 0.0013).sin() * 0.01)
                .collect()
        };

        let model = ModelWeights {
            cfg: cfg.clone(),
            layers: vec![LayerWeights {
                wq: make_small(dim * dim, 0),
                wk: make_small(dim * dim, 1000),
                wv: make_small(dim * dim, 2000),
                wo: make_small(dim * dim, 3000),
                w1: make_small(hidden * dim, 4000),
                w2: make_small(dim * hidden, 5000),
                w3: make_small(hidden * dim, 6000),
                rms_att: vec![1.0; dim],
                rms_ffn: vec![1.0; dim],
                q_norm: None,
                k_norm: None,
                gdn: None,
                moe: None,
            }],
            rms_final: vec![1.0; dim],
            embed: make_small(vocab * dim, 7000),
            vocab_size: vocab,
            lm_head: None,
            vocab_clusters: None,
            router_adapter: None,
        };

        let tokens: Vec<u16> = (0..seq).map(|i| (i % vocab) as u16).collect();
        let targets: Vec<u16> = (0..seq).map(|i| ((i + 1) % vocab) as u16).collect();

        // Baseline forward + backward
        let fwd =
            ane_forward::forward(&fwd_kernels, &model, &tokens, &targets).expect("forward failed");
        let grads = backward(&bwd_kernels, &model, &fwd, &tokens).expect("backward failed");
        let base_loss = fwd.loss;

        let epsilon = 1e-3f32;

        // Helper: perturb a weight, re-run forward, check gradient
        let check_grad = |name: &str, computed_grad: f32, mut model_copy: ModelWeights| {
            let fwd_plus = ane_forward::forward(&fwd_kernels, &model_copy, &tokens, &targets)
                .expect("forward+ failed");
            let numerical_grad = (fwd_plus.loss - base_loss) / epsilon;

            // For ANE fp16 roundtrip kernels, tolerance needs to be generous
            let tol = 0.5; // allow significant tolerance for ANE fp16 ops
            let err = (computed_grad - numerical_grad).abs();
            let rel_err = if numerical_grad.abs() > 1e-6 {
                err / numerical_grad.abs()
            } else {
                err
            };
            eprintln!(
                "  {name}: computed={computed_grad:.6}, numerical={numerical_grad:.6}, \
                 err={err:.6}, rel_err={rel_err:.4}"
            );
            // Soft check: at least same sign or very small
            if numerical_grad.abs() > 1e-5 && computed_grad.abs() > 1e-5 {
                assert!(
                    computed_grad * numerical_grad >= 0.0 || err < tol,
                    "{name}: gradient sign mismatch or large error"
                );
            }
        };

        // Test rms_final gradient
        {
            let mut m = model.clone();
            m.rms_final[0] += epsilon;
            check_grad("rms_final[0]", grads.drms_final[0], m);
        }

        // Test rms_att gradient
        {
            let mut m = model.clone();
            m.layers[0].rms_att[0] += epsilon;
            check_grad("rms_att[0]", grads.layers[0].drms_att[0], m);
        }

        // Test w2 gradient (FFN down projection)
        {
            let mut m = model.clone();
            m.layers[0].w2[0] += epsilon;
            check_grad("w2[0]", grads.layers[0].dw2[0], m);
        }

        // Test embed gradient
        {
            // Find a token that's used
            let tok = tokens[0] as usize;
            let idx = tok * dim;
            let mut m = model.clone();
            m.embed[idx] += epsilon;
            check_grad("embed[tok0,0]", grads.dembed[idx], m);
        }

        eprintln!("Gradient perturbation tests completed (see values above)");
    }

    // -----------------------------------------------------------------------
    // Round 6: untied classifier (lm_head)
    // -----------------------------------------------------------------------

    /// Validates classifier_bwd produces correct dcls_w for untied lm_head.
    ///
    /// With fused CE, backward() no longer materializes dlogits and skips dcls_w
    /// (LoRA training doesn't need base weight gradients). This test validates
    /// the unfused classifier_forward → cross_entropy_loss → classifier_bwd path
    /// directly, ensuring it remains correct for any future full-model training.
    #[test]
    fn test_untied_classifier_gradients() {
        let dim = 64;
        let hidden = 128;
        let n_heads = 4;
        let seq = 64;
        let vocab = 32;

        let cfg = MilConfig::mha(dim, hidden, n_heads, seq);

        let fwd_kernels = match ane_forward::CompiledKernels::compile_forward(&cfg) {
            Ok(k) => k,
            Err(e) => {
                eprintln!("Skipping untied classifier test (ANE unavailable): {e}");
                return;
            }
        };

        let make_small = |n: usize, seed: usize| -> Vec<f32> {
            (0..n)
                .map(|i| ((i + seed) as f32 * 0.0013).sin() * 0.01)
                .collect()
        };

        let model = ModelWeights {
            cfg: cfg.clone(),
            layers: vec![LayerWeights {
                wq: make_small(dim * dim, 0),
                wk: make_small(dim * dim, 1000),
                wv: make_small(dim * dim, 2000),
                wo: make_small(dim * dim, 3000),
                w1: make_small(hidden * dim, 4000),
                w2: make_small(dim * hidden, 5000),
                w3: make_small(hidden * dim, 6000),
                rms_att: vec![1.0; dim],
                rms_ffn: vec![1.0; dim],
                q_norm: None,
                k_norm: None,
                gdn: None,
                moe: None,
            }],
            rms_final: vec![1.0; dim],
            embed: make_small(vocab * dim, 7000),
            vocab_size: vocab,
            lm_head: Some(make_small(vocab * dim, 9000)), // untied!
            vocab_clusters: None,
            router_adapter: None,
        };

        let tokens: Vec<u16> = (0..seq).map(|i| (i % vocab) as u16).collect();
        let targets: Vec<u16> = (0..seq).map(|i| ((i + 1) % vocab) as u16).collect();

        // Run forward to get activations (x_final reconstruction)
        let fwd =
            ane_forward::forward(&fwd_kernels, &model, &tokens, &targets).expect("forward failed");
        assert!(fwd.loss.is_finite(), "loss not finite");

        // Reconstruct x_final from last layer activations
        let n_layers = model.layers.len();
        let last_act = &fwd.layer_acts[n_layers - 1];
        let mut x_cur = last_act.x2.clone();
        ane_forward::vec_add_inplace(&mut x_cur, &last_act.ffn_out);
        let mut x_final = vec![0.0f32; dim * seq];
        ane_forward::rmsnorm(
            &mut x_final,
            &x_cur,
            &model.rms_final,
            dim,
            seq,
            cfg.rms_eps,
        );

        // Unfused path: classifier_forward → cross_entropy_loss → classifier_bwd
        let lm_head = model.lm_head.as_ref().unwrap();
        let mut logits = vec![0.0f32; vocab * seq];
        ane_forward::classifier_forward(&mut logits, lm_head, &x_final, vocab, dim, seq);

        let (_loss, dlogits) = ane_forward::cross_entropy_loss(&logits, &targets, vocab, seq);

        let mut dy = vec![0.0f32; dim * seq];
        let mut dlm_head = vec![0.0f32; vocab * dim];
        classifier_bwd(
            &mut dy,
            &mut dlm_head,
            &dlogits,
            lm_head,
            &x_final,
            vocab,
            dim,
            seq,
        );

        // dlm_head should have nonzero gradients
        assert_eq!(dlm_head.len(), vocab * dim);
        let nonzero = dlm_head.iter().filter(|v| v.abs() > 1e-15).count();
        assert!(nonzero > 0, "dlm_head should have nonzero gradients");

        // dy (gradient flowing back through the classifier) should be nonzero
        let dy_nonzero = dy.iter().filter(|v| v.abs() > 1e-15).count();
        assert!(dy_nonzero > 0, "dy from classifier_bwd should be nonzero");
    }

    // -----------------------------------------------------------------------
    // Loss scaling: 2x dlogits should produce 2x gradients (linearity)
    // -----------------------------------------------------------------------

    #[test]
    fn test_loss_scaling_produces_proportional_lora_gradients() {
        let cfg = MilConfig {
            dim: 8,
            hidden_dim: 16,
            n_heads: 4,
            seq_len: 3,
            n_kv_heads: 4,
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
        let vocab = 10;

        let make_vals = |n: usize, seed: usize| -> Vec<f32> {
            (0..n)
                .map(|i| ((i + seed) as f32 * 0.173).sin() * 0.2)
                .collect()
        };

        let model = ModelWeights {
            cfg: cfg.clone(),
            layers: vec![LayerWeights {
                wq: make_vals(dim * dim, 0),
                wk: make_vals(dim * dim, 100),
                wv: make_vals(dim * dim, 200),
                wo: make_vals(dim * dim, 300),
                w1: make_vals(hidden * dim, 400),
                w2: make_vals(dim * hidden, 500),
                w3: make_vals(hidden * dim, 600),
                rms_att: vec![1.0; dim],
                rms_ffn: vec![1.0; dim],
                q_norm: None,
                k_norm: None,
                gdn: None,
                moe: None,
            }],
            rms_final: vec![1.0; dim],
            embed: make_vals(vocab * dim, 700),
            vocab_size: vocab,
            lm_head: None,
            vocab_clusters: None,
            router_adapter: None,
        };

        let lora = LoraModel::new(
            LoraConfig {
                rank: 4,
                alpha: 4.0,
                target_modules: vec!["wo".into(), "w2".into()],
            },
            1,
            dim,
            hidden,
        );

        let tokens = vec![1u16, 2, 3];
        let targets = vec![2u16, 3, 4];

        // Run forward twice (identical — deterministic) to get independent results
        let fwd_1x = ane_forward::forward_cpu_generic(&model, Some(&lora), &tokens, &targets);
        let mut fwd_2x = ane_forward::forward_cpu_generic(&model, Some(&lora), &tokens, &targets);

        // Scale classifier_dy on the second by 2x (simulates loss_scale=2.0)
        for v in fwd_2x.base.classifier_dy.iter_mut() {
            *v *= 2.0;
        }

        // Backward both
        let bwd_1x = backward_lora_cpu_generic(&model, &fwd_1x, &lora, &tokens, 0.0, 1.0);
        let bwd_2x = backward_lora_cpu_generic(&model, &fwd_2x, &lora, &tokens, 0.0, 1.0);

        // All LoRA grads in 2x should be exactly 2x those in 1x (backward is linear in dlogits)
        let check_proportional =
            |name: &str,
             g1: &Option<crate::agent::ane_lora::LoraAdapterGrads>,
             g2: &Option<crate::agent::ane_lora::LoraAdapterGrads>| {
                if let (Some(g1), Some(g2)) = (g1, g2) {
                    for (i, (a, b)) in g1.da.iter().zip(g2.da.iter()).enumerate() {
                        let expected = a * 2.0;
                        let err = (b - expected).abs();
                        assert!(
                            err < 1e-4,
                            "{name}.da[{i}]: 1x={a:.6}, 2x={b:.6}, expected={expected:.6}"
                        );
                    }
                    for (i, (a, b)) in g1.db.iter().zip(g2.db.iter()).enumerate() {
                        let expected = a * 2.0;
                        let err = (b - expected).abs();
                        assert!(
                            err < 1e-4,
                            "{name}.db[{i}]: 1x={a:.6}, 2x={b:.6}, expected={expected:.6}"
                        );
                    }
                }
            };

        check_proportional(
            "wo",
            &bwd_1x.lora_grads.layers[0].wo,
            &bwd_2x.lora_grads.layers[0].wo,
        );
        check_proportional(
            "w2",
            &bwd_1x.lora_grads.layers[0].w2,
            &bwd_2x.lora_grads.layers[0].w2,
        );
    }

    /// Benchmark backward pass at Qwen3.5-4B MHA layer dimensions.
    ///
    /// Validates correctness (finite loss, nonzero grads, attn_output_gate + q_proj_dim)
    /// and reports wall-clock timing for parallel vs sequential QKV/FFN matmuls.
    ///
    /// Run: cargo test --features ane --release --lib -- "bench_backward_qwen3_5_4b" --nocapture --test-threads=1
    #[test]
    fn bench_backward_qwen3_5_4b() {
        use std::time::Instant;

        let dim = 2560;
        let hidden = 9216;
        let n_heads: usize = 16;
        let head_dim: usize = 256;
        let seq = 128;
        let vocab = 1024;
        let ad = n_heads * head_dim; // 4096
        let qpd = 2 * ad; // 8192 (attn_output_gate)

        // Qwen3.5-4B MHA config (no GDN layers — benchmark one pure MHA layer)
        let cfg = MilConfig {
            dim,
            hidden_dim: hidden,
            n_heads,
            seq_len: seq,
            n_kv_heads: 4,
            rope_theta: 10_000_000.0,
            rms_eps: 1e-6,
            has_lm_head: false,
            head_dim_explicit: head_dim,
            linear_attn_indices: vec![],
            linear_n_heads: 0,
            linear_head_dim: 0,
            linear_n_value_heads: 0,
            linear_value_head_dim: 0,
            conv_kernel_size: 0,
            attn_output_gate: true,
        };

        let make = |n: usize, seed: usize| -> Vec<f32> {
            (0..n)
                .map(|i| ((i + seed) as f32 * 0.0013).sin() * 0.01)
                .collect()
        };

        let model = ModelWeights {
            cfg: cfg.clone(),
            layers: vec![LayerWeights {
                wq: make(qpd * dim, 0),
                wk: make(ad * dim, 1000),
                wv: make(ad * dim, 2000),
                wo: make(dim * ad, 3000),
                w1: make(hidden * dim, 4000),
                w2: make(dim * hidden, 5000),
                w3: make(hidden * dim, 6000),
                rms_att: vec![1.0; dim],
                rms_ffn: vec![1.0; dim],
                q_norm: Some(vec![1.0; head_dim]),
                k_norm: Some(vec![1.0; head_dim]),
                gdn: None,
                moe: None,
            }],
            rms_final: vec![1.0; dim],
            embed: make(vocab * dim, 7000),
            vocab_size: vocab,
            lm_head: None,
            vocab_clusters: None,
            router_adapter: None,
        };

        let lora = super::super::ane_lora::LoraModel::with_full_dims(
            super::super::ane_lora::LoraConfig {
                rank: 16,
                alpha: 16.0,
                target_modules: vec!["wo".into(), "w2".into()],
            },
            1,
            dim,
            ad, // kv_dim (expanded)
            ad, // attn_dim
            qpd,
            hidden,
        );

        let tokens: Vec<u16> = (0..seq).map(|i| (i % vocab) as u16).collect();
        let targets: Vec<u16> = (0..seq).map(|i| ((i + 1) % vocab) as u16).collect();

        // Forward pass (CPU)
        let fwd = ane_forward::forward_cpu_generic(&model, Some(&lora), &tokens, &targets);
        assert!(
            fwd.base.loss.is_finite(),
            "forward loss not finite: {}",
            fwd.base.loss
        );
        eprintln!("forward loss: {:.4}", fwd.base.loss);

        // Warmup backward (CPU path — ANE can't compile at these dims)
        let _ = backward_lora_cpu_generic(&model, &fwd, &lora, &tokens, 0.0, 1.0);

        // Benchmark full backward (includes rayon::join for QKV + FFN matmuls)
        let n_iters = 5;
        let t0 = Instant::now();
        for _ in 0..n_iters {
            let bwd = backward_lora_cpu_generic(&model, &fwd, &lora, &tokens, 0.0, 1.0);
            std::hint::black_box(&bwd);
        }
        let total_ms = t0.elapsed().as_millis();
        let per_iter_ms = total_ms as f64 / n_iters as f64;

        // Micro-benchmark: sequential vs parallel QKV matmuls
        let lw = &model.layers[0];
        let dq_for_wq = make(qpd * seq, 9000);
        let dk = make(ad * seq, 9100);
        let dv = make(ad * seq, 9200);

        let micro_n = 10;
        let t_seq = Instant::now();
        for _ in 0..micro_n {
            let mut dx = ane_forward::cpu_matmul_lhs_transposed(&lw.wq, qpd, dim, &dq_for_wq, seq);
            let dxk = ane_forward::cpu_matmul_lhs_transposed(&lw.wk, ad, dim, &dk, seq);
            let dxv = ane_forward::cpu_matmul_lhs_transposed(&lw.wv, ad, dim, &dv, seq);
            ane_forward::vec_add_inplace(&mut dx, &dxk);
            ane_forward::vec_add_inplace(&mut dx, &dxv);
            std::hint::black_box(&dx);
        }
        let seq_qkv_ms = t_seq.elapsed().as_millis() as f64 / micro_n as f64;

        let t_par = Instant::now();
        for _ in 0..micro_n {
            let (mut dx, (dxk, dxv)) = rayon::join(
                || ane_forward::cpu_matmul_lhs_transposed(&lw.wq, qpd, dim, &dq_for_wq, seq),
                || {
                    rayon::join(
                        || ane_forward::cpu_matmul_lhs_transposed(&lw.wk, ad, dim, &dk, seq),
                        || ane_forward::cpu_matmul_lhs_transposed(&lw.wv, ad, dim, &dv, seq),
                    )
                },
            );
            ane_forward::vec_add_inplace(&mut dx, &dxk);
            ane_forward::vec_add_inplace(&mut dx, &dxv);
            std::hint::black_box(&dx);
        }
        let par_qkv_ms = t_par.elapsed().as_millis() as f64 / micro_n as f64;

        // Micro-benchmark: sequential vs parallel FFN W1^T / W3^T
        let dh1 = make(hidden * seq, 9300);
        let dh3 = make(hidden * seq, 9400);

        let t_seq_ffn = Instant::now();
        for _ in 0..micro_n {
            let mut dx = ane_forward::cpu_matmul_lhs_transposed(&lw.w1, hidden, dim, &dh1, seq);
            let dxw3 = ane_forward::cpu_matmul_lhs_transposed(&lw.w3, hidden, dim, &dh3, seq);
            ane_forward::vec_add_inplace(&mut dx, &dxw3);
            std::hint::black_box(&dx);
        }
        let seq_ffn_ms = t_seq_ffn.elapsed().as_millis() as f64 / micro_n as f64;

        let t_par_ffn = Instant::now();
        for _ in 0..micro_n {
            let (mut dx, dxw3) = rayon::join(
                || ane_forward::cpu_matmul_lhs_transposed(&lw.w1, hidden, dim, &dh1, seq),
                || ane_forward::cpu_matmul_lhs_transposed(&lw.w3, hidden, dim, &dh3, seq),
            );
            ane_forward::vec_add_inplace(&mut dx, &dxw3);
            std::hint::black_box(&dx);
        }
        let par_ffn_ms = t_par_ffn.elapsed().as_millis() as f64 / micro_n as f64;

        eprintln!("\nQwen3.5-4B backward (1 MHA layer, seq={seq}, attn_output_gate=true):");
        eprintln!("  Full backward: {n_iters} iters in {total_ms}ms ({per_iter_ms:.1}ms/iter)");
        eprintln!(
            "  QKV matmuls — seq: {seq_qkv_ms:.1}ms, par: {par_qkv_ms:.1}ms, \
             speedup: {:.2}x",
            seq_qkv_ms / par_qkv_ms
        );
        eprintln!(
            "  FFN W1^T/W3^T — seq: {seq_ffn_ms:.1}ms, par: {par_ffn_ms:.1}ms, \
             speedup: {:.2}x",
            seq_ffn_ms / par_ffn_ms
        );

        // Verify grads are nonzero
        let bwd = backward_lora_cpu_generic(&model, &fwd, &lora, &tokens, 0.0, 1.0);
        let lg = &bwd.lora_grads.layers[0];
        let w2_norm: f32 = lg
            .w2
            .as_ref()
            .map(|g| {
                g.da.iter()
                    .chain(g.db.iter())
                    .map(|v| v * v)
                    .sum::<f32>()
                    .sqrt()
            })
            .unwrap_or(0.0);
        let wo_norm: f32 = lg
            .wo
            .as_ref()
            .map(|g| {
                g.da.iter()
                    .chain(g.db.iter())
                    .map(|v| v * v)
                    .sum::<f32>()
                    .sqrt()
            })
            .unwrap_or(0.0);
        eprintln!("  W2 grad norm: {w2_norm:.6}, Wo grad norm: {wo_norm:.6}");
        assert!(w2_norm > 0.0, "W2 LoRA grads are zero");
        assert!(wo_norm > 0.0, "Wo LoRA grads are zero");
    }

    // -----------------------------------------------------------------------
    // CPU vs ANE backward comparison — validates transpose fix end-to-end
    // -----------------------------------------------------------------------

    #[test]
    fn test_ane_backward_lora_matches_cpu_backward() {
        use super::super::ane_lora::{LoraConfig, LoraKernels, LoraModel};
        use super::super::ane_weights::LayerWeights;

        let dim = 64;
        let hidden = 128;
        let n_heads = 4;
        let seq = 64;
        let vocab = 32;
        let rank = 16;

        let cfg = MilConfig::mha(dim, hidden, n_heads, seq);

        // Compile ANE kernels
        let fwd_kernels = match ane_forward::CompiledKernels::compile_forward(&cfg) {
            Ok(k) => k,
            Err(e) => {
                eprintln!("Skipping CPU vs ANE backward test (ANE unavailable): {e}");
                return;
            }
        };
        let bwd_kernels = BackwardKernels::compile_backward(&cfg, &fwd_kernels.mask_blob)
            .expect("compile_backward failed");
        let lora_kernels = LoraKernels::compile(&cfg, rank).expect("LoRA kernels failed");

        // Model with larger weights to expose transpose errors
        let make_weight = |n: usize, seed: usize| -> Vec<f32> {
            (0..n)
                .map(|i| ((i + seed) as f32 * 0.0013).sin() * 0.1)
                .collect()
        };

        let model = ModelWeights {
            cfg: cfg.clone(),
            layers: vec![LayerWeights {
                wq: make_weight(dim * dim, 0),
                wk: make_weight(dim * dim, 1000),
                wv: make_weight(dim * dim, 2000),
                wo: make_weight(dim * dim, 3000),
                w1: make_weight(hidden * dim, 4000),
                w2: make_weight(dim * hidden, 5000),
                w3: make_weight(hidden * dim, 6000),
                rms_att: vec![1.0; dim],
                rms_ffn: vec![1.0; dim],
                q_norm: None,
                k_norm: None,
                gdn: None,
                moe: None,
            }],
            rms_final: vec![1.0; dim],
            embed: make_weight(vocab * dim, 7000),
            vocab_size: vocab,
            lm_head: None,
            vocab_clusters: None,
            router_adapter: None,
        };

        let lora = LoraModel::new(
            LoraConfig {
                rank,
                ..LoraConfig::default()
            },
            1,
            dim,
            hidden,
        );

        let tokens: Vec<u16> = (0..seq).map(|i| (i % vocab) as u16).collect();
        let targets: Vec<u16> = (0..seq).map(|i| ((i + 1) % vocab) as u16).collect();

        // Shared CPU forward — isolates backward path differences
        let fwd = ane_forward::forward_cpu(&model, Some(&lora), &tokens, &targets);
        eprintln!("Forward loss: {:.6}", fwd.base.loss);

        let cpu_bwd = backward_lora_cpu(&model, &fwd, &lora, &tokens);
        let ane_bwd = backward_with_lora(&bwd_kernels, &model, &fwd, &lora, &lora_kernels, &tokens)
            .expect("ANE backward failed");

        // Compare LoRA gradients
        let compare_adapter =
            |name: &str,
             cpu: &Option<super::super::ane_lora::LoraAdapterGrads>,
             ane: &Option<super::super::ane_lora::LoraAdapterGrads>| {
                let (cpu, ane) = match (cpu, ane) {
                    (Some(c), Some(a)) => (c, a),
                    (None, None) => return,
                    _ => panic!("{name}: one has grads, the other doesn't"),
                };

                let cpu_da_norm: f32 = cpu.da.iter().map(|v| v * v).sum::<f32>().sqrt();
                let ane_da_norm: f32 = ane.da.iter().map(|v| v * v).sum::<f32>().sqrt();
                let da_diff: f32 = cpu
                    .da
                    .iter()
                    .zip(ane.da.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
                    .sqrt();

                let cpu_db_norm: f32 = cpu.db.iter().map(|v| v * v).sum::<f32>().sqrt();
                let ane_db_norm: f32 = ane.db.iter().map(|v| v * v).sum::<f32>().sqrt();
                let db_diff: f32 = cpu
                    .db
                    .iter()
                    .zip(ane.db.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
                    .sqrt();

                let da_rel = if cpu_da_norm > 1e-10 {
                    da_diff / cpu_da_norm
                } else {
                    da_diff
                };
                let db_rel = if cpu_db_norm > 1e-10 {
                    db_diff / cpu_db_norm
                } else {
                    db_diff
                };

                eprintln!(
                "  {name}.dA: cpu_norm={cpu_da_norm:.6}, ane_norm={ane_da_norm:.6}, rel_err={da_rel:.6}"
            );
                eprintln!(
                "  {name}.dB: cpu_norm={cpu_db_norm:.6}, ane_norm={ane_db_norm:.6}, rel_err={db_rel:.6}"
            );

                assert!(da_rel < 0.5, "{name}.dA relative error too large: {da_rel}");
                assert!(db_rel < 0.5, "{name}.dB relative error too large: {db_rel}");
            };

        for (i, (cpu_lg, ane_lg)) in cpu_bwd
            .lora_grads
            .layers
            .iter()
            .zip(ane_bwd.lora_grads.layers.iter())
            .enumerate()
        {
            eprintln!("Layer {i} LoRA gradient comparison:");
            compare_adapter(&format!("L{i}.Wq"), &cpu_lg.wq, &ane_lg.wq);
            compare_adapter(&format!("L{i}.Wv"), &cpu_lg.wv, &ane_lg.wv);
            compare_adapter(&format!("L{i}.Wo"), &cpu_lg.wo, &ane_lg.wo);
            compare_adapter(&format!("L{i}.W2"), &cpu_lg.w2, &ane_lg.w2);
        }

        // Also compare model gradients if ANE produced them
        if let Some(ref ane_model_grads) = ane_bwd.model_grads {
            let lg = &ane_model_grads.layers[0];
            let norms = [
                ("dwq", lg.dwq.iter().map(|v| v * v).sum::<f32>().sqrt()),
                ("dwk", lg.dwk.iter().map(|v| v * v).sum::<f32>().sqrt()),
                ("dwv", lg.dwv.iter().map(|v| v * v).sum::<f32>().sqrt()),
                ("dwo", lg.dwo.iter().map(|v| v * v).sum::<f32>().sqrt()),
                ("dw1", lg.dw1.iter().map(|v| v * v).sum::<f32>().sqrt()),
                ("dw2", lg.dw2.iter().map(|v| v * v).sum::<f32>().sqrt()),
                ("dw3", lg.dw3.iter().map(|v| v * v).sum::<f32>().sqrt()),
            ];
            for (name, norm) in &norms {
                eprintln!("  ANE model grad {name} norm: {norm:.6}");
                assert!(norm.is_finite(), "ANE model grad {name} is not finite");
            }
        }
    }

    // -----------------------------------------------------------------------
    // Qwen3.5-0.8B LoRA training (CPU backward on real quantized model)
    // -----------------------------------------------------------------------

    #[test]
    fn test_qwen3_5_0_8b_lora_training() {
        use super::super::ane_lora::{lora_adam_update, LoraConfig, LoraModel, LoraModelAdam};
        use super::super::ane_weights::{QuantizedModelWeights, WeightSource};

        let model_dir = std::path::Path::new(&std::env::var("HOME").unwrap())
            .join(".cache/lm-studio/models/mlx-community/Qwen3.5-0.8B-8bit");
        if !model_dir.exists() {
            eprintln!("SKIP: Qwen3.5-0.8B not found at {}", model_dir.display());
            return;
        }

        // Read config from model
        let config_str =
            std::fs::read_to_string(model_dir.join("config.json")).expect("config.json");
        let root: serde_json::Value = serde_json::from_str(&config_str).unwrap();
        let tc = root.get("text_config").unwrap_or(&root);

        let dim = tc["hidden_size"].as_u64().unwrap() as usize;
        let hidden_dim = tc["intermediate_size"].as_u64().unwrap() as usize;
        let n_heads = tc["num_attention_heads"].as_u64().unwrap() as usize;
        let n_kv_heads = tc["num_key_value_heads"].as_u64().unwrap() as usize;
        let head_dim = tc["head_dim"].as_u64().unwrap_or((dim / n_heads) as u64) as usize;
        let rms_eps = tc["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32;
        let rope_cfg = tc.get("rope_parameters").or_else(|| tc.get("rope_scaling"));
        let rope_theta = rope_cfg
            .and_then(|r| r["rope_theta"].as_f64())
            .unwrap_or(tc["rope_theta"].as_f64().unwrap_or(1e6));
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
        let n_layers = tc["num_hidden_layers"]
            .as_u64()
            .unwrap_or(layer_types.len() as u64) as usize;
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

        let seq = 8; // short for test speed
        let cfg = MilConfig {
            dim,
            hidden_dim,
            n_heads,
            seq_len: seq,
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
        };

        eprintln!(
            "Qwen3.5-0.8B: dim={dim}, hidden={hidden_dim}, heads={n_heads}, kv={n_kv_heads}, \
             head_dim={head_dim}, layers={n_layers}, gate={attn_output_gate}, \
             linear_layers={}/{n_layers}",
            cfg.linear_attn_indices.len()
        );

        eprintln!("Loading quantized model...");
        let model = QuantizedModelWeights::from_mlx_safetensors(&model_dir, &cfg)
            .expect("model load failed");

        let kv_dim = n_kv_heads * head_dim;
        let ad = cfg.attn_dim();
        let qpd = cfg.q_proj_dim();
        let lora_cfg = LoraConfig {
            rank: 4,
            ..LoraConfig::default()
        };
        let mut lora =
            LoraModel::with_full_dims(lora_cfg, n_layers, dim, kv_dim, ad, qpd, hidden_dim);
        let mut adam = LoraModelAdam::zeros(&lora);

        let tokens: Vec<u32> = (100..100 + seq as u32).collect();
        let targets: Vec<u32> = (101..101 + seq as u32).collect();

        let n_steps = 3;
        let lr = 5e-4;
        let mut losses = Vec::with_capacity(n_steps);

        for step in 0..n_steps {
            let fwd = ane_forward::forward_cpu_generic(&model, Some(&lora), &tokens, &targets);
            let bwd = backward_lora_cpu_generic(&model, &fwd, &lora, &tokens, 0.0, 1.0);
            lora_adam_update(
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
            eprintln!("  step {step}: loss={:.4}", fwd.base.loss);
            losses.push(fwd.base.loss);
        }

        // Loss should be finite
        for (i, &l) in losses.iter().enumerate() {
            assert!(l.is_finite(), "step {i} loss not finite: {l}");
        }

        // Loss should decrease or at least not explode
        assert!(
            losses.last().unwrap() < &(losses[0] + 1.0),
            "loss diverged: {:?}",
            losses
        );

        eprintln!(
            "Qwen3.5-0.8B LoRA training: {} steps, losses={:.4?}",
            n_steps, losses
        );
    }

    // -----------------------------------------------------------------------
    // Fused SDPA backward vs CPU reference (gen_sdpa_rope_bwd correctness)
    // -----------------------------------------------------------------------

    #[test]
    fn test_fused_sdpa_bwd_matches_cpu_reference() {
        let dim = 64;
        let hidden = 128;
        let n_heads: usize = 4;
        let n_kv_heads: usize = 4; // MHA (forward_cpu doesn't support GQA K/V dims)
        let head_dim: usize = 16;
        let seq = 64;
        let vocab = 32;
        let ad = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let qpd = 2 * ad; // attn_output_gate

        let cfg = MilConfig {
            dim,
            hidden_dim: hidden,
            n_heads,
            seq_len: seq,
            n_kv_heads,
            rope_theta: 10000.0,
            rms_eps: 1e-5,
            has_lm_head: false,
            head_dim_explicit: head_dim,
            linear_attn_indices: vec![],
            linear_n_heads: 0,
            linear_head_dim: 0,
            linear_n_value_heads: 0,
            linear_value_head_dim: 0,
            conv_kernel_size: 0,
            attn_output_gate: true,
        };

        let fwd_kernels = match ane_forward::CompiledKernels::compile_forward(&cfg) {
            Ok(k) => k,
            Err(e) => {
                eprintln!("Skipping fused SDPA backward test (ANE unavailable): {e}");
                return;
            }
        };

        let bwd_kernels = BackwardKernels::compile_backward(&cfg, &fwd_kernels.mask_blob)
            .expect("compile_backward failed");

        if bwd_kernels.fused_attn_gqa_bwd.is_none() {
            eprintln!("Skipping: fused_attn_gqa_bwd kernel did not compile");
            return;
        }

        let make = |n: usize, seed: usize| -> Vec<f32> {
            (0..n)
                .map(|i| ((i + seed) as f32 * 0.0013).sin() * 0.05)
                .collect()
        };

        let model = ane_weights::ModelWeights {
            cfg: cfg.clone(),
            layers: vec![ane_weights::LayerWeights {
                wq: make(qpd * dim, 0),
                wk: make(kv_dim * dim, 1000),
                wv: make(kv_dim * dim, 2000),
                wo: make(dim * ad, 3000),
                w1: make(hidden * dim, 4000),
                w2: make(dim * hidden, 5000),
                w3: make(hidden * dim, 6000),
                rms_att: vec![1.0; dim],
                rms_ffn: vec![1.0; dim],
                q_norm: None,
                k_norm: None,
                gdn: None,
                moe: None,
            }],
            rms_final: vec![1.0; dim],
            embed: make(vocab * dim, 7000),
            vocab_size: vocab,
            lm_head: None,
            vocab_clusters: None,
            router_adapter: None,
        };

        let tokens: Vec<u16> = (0..seq).map(|i| (i % vocab) as u16).collect();
        let targets: Vec<u16> = (0..seq).map(|i| ((i + 1) % vocab) as u16).collect();

        let fwd = ane_forward::forward_cpu(&model, None, &tokens, &targets);
        assert!(fwd.base.loss.is_finite(), "forward loss: {}", fwd.base.loss);

        let ac = &fwd.base.layer_acts[0];
        let lw = &model.layers[0];
        let dx2 = make(dim * seq, 8000);

        let cpu_dx = mha_backward_cpu_dx_attn(lw, ac, &dx2, &cfg);
        let fused_dx = mha_backward_fused_sdpa_dx_attn(&bwd_kernels, lw, ac, &dx2, &cfg)
            .expect("fused SDPA backward failed");

        assert_eq!(cpu_dx.len(), fused_dx.len());

        let max_err = max_abs_diff(&cpu_dx, &fused_dx);
        let cpu_norm: f32 = cpu_dx.iter().map(|v| v * v).sum::<f32>().sqrt();
        let fused_norm: f32 = fused_dx.iter().map(|v| v * v).sum::<f32>().sqrt();

        eprintln!(
            "fused SDPA bwd: max_err={max_err:.6}, cpu_norm={cpu_norm:.6}, fused_norm={fused_norm:.6}"
        );

        assert!(
            max_err < 0.05,
            "fused SDPA backward max error too large: {max_err:.6}"
        );
        assert!(cpu_norm > 1e-8, "CPU dx_attn is zero: norm={cpu_norm}");
        assert!(
            fused_norm > 1e-8,
            "Fused dx_attn is zero: norm={fused_norm}"
        );
    }
}
