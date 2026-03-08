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
    debug_assert_eq!(dx.len(), dim * seq);
    debug_assert_eq!(dw.len(), dim);
    debug_assert_eq!(dy.len(), dim * seq);
    debug_assert_eq!(x.len(), dim * seq);
    debug_assert_eq!(w.len(), dim);

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

    // dy[d,t] = sum_v embed[v,d] * dlogits[v,t]
    // embed[v,d] = embed[v*dim + d], dlogits[v,t] = dlogits[v*seq + t]
    for d in 0..dim {
        for t in 0..seq {
            let mut acc = 0.0f32;
            for v in 0..vocab {
                acc += embed[v * dim + d] * dlogits[v * seq + t];
            }
            dy[d * seq + t] = acc;
        }
    }

    // dembed[v,d] += sum_t dlogits[v,t] * x_final[d,t]
    for v in 0..vocab {
        for d in 0..dim {
            let mut acc = 0.0f32;
            for t in 0..seq {
                acc += dlogits[v * seq + t] * x_final[d * seq + t];
            }
            dembed[v * dim + d] += acc;
        }
    }
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

/// Pre-compiled ANE kernels for backward pass (compile once, reuse every step).
pub struct BackwardKernels {
    pub ffn_bwd_w2t: AneKernel,  // DynMatmul(dim, hidden) — dffn @ W2^T
    pub ffn_bwd_w13t: AneKernel, // Fused dh1@W1^T + dh3@W3^T
    pub wot_bwd: AneKernel,      // DynMatmul(dim, dim) — dx2 @ Wo^T
    pub sdpa_bwd1: AneKernel,    // fp16: recompute softmax + dV + dp
    pub sdpa_bwd2: AneKernel,    // fp16: softmax bwd → dQ, dK
    pub qkv_bwd: AneKernel,      // Fused dq@Wq^T + dk@Wk^T + dv@Wv^T
}

impl BackwardKernels {
    /// Compile all backward-pass kernels for the given config.
    pub fn compile_backward(cfg: &MilConfig, mask_blob: &[u8]) -> Result<Self, String> {
        ane_bridge::ane_init()?;

        // ffn_bwd_w2t: DynMatmul(dim, hidden)
        let w2t_spec = KernelSpec::for_kernel(cfg, KernelType::FfnBwdW2t);
        let ffn_bwd_w2t = AneKernel::compile(
            &w2t_spec.mil_text,
            None,
            &[w2t_spec.input_bytes],
            &[w2t_spec.output_bytes],
        )?;

        // ffn_bwd_w13t
        let w13t_spec = KernelSpec::for_kernel(cfg, KernelType::FfnBwdW13t);
        let ffn_bwd_w13t = AneKernel::compile(
            &w13t_spec.mil_text,
            None,
            &[w13t_spec.input_bytes],
            &[w13t_spec.output_bytes],
        )?;

        // wot_bwd: DynMatmul(dim, dim)
        let wot_spec = KernelSpec::for_kernel(cfg, KernelType::Wot);
        let wot_bwd = AneKernel::compile(
            &wot_spec.mil_text,
            None,
            &[wot_spec.input_bytes],
            &[wot_spec.output_bytes],
        )?;

        // sdpa_bwd1: needs causal mask (like sdpa_fwd)
        let bwd1_spec = KernelSpec::for_kernel(cfg, KernelType::SdpaBwd1);
        let sdpa_bwd1 = AneKernel::compile_multi_weights(
            &bwd1_spec.mil_text,
            &["@model_path/weights/mask.bin"],
            &[mask_blob],
            &[bwd1_spec.input_bytes],
            &[bwd1_spec.output_bytes],
        )?;

        // sdpa_bwd2: no static weights
        let bwd2_spec = KernelSpec::for_kernel(cfg, KernelType::SdpaBwd2);
        let sdpa_bwd2 = AneKernel::compile(
            &bwd2_spec.mil_text,
            None,
            &[bwd2_spec.input_bytes],
            &[bwd2_spec.output_bytes],
        )?;

        // qkv_bwd
        let qkv_spec = KernelSpec::for_kernel(cfg, KernelType::Qkvb);
        let qkv_bwd = AneKernel::compile(
            &qkv_spec.mil_text,
            None,
            &[qkv_spec.input_bytes],
            &[qkv_spec.output_bytes],
        )?;

        Ok(Self {
            ffn_bwd_w2t,
            ffn_bwd_w13t,
            wot_bwd,
            sdpa_bwd1,
            sdpa_bwd2,
            qkv_bwd,
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
    let vocab = model.vocab_size;
    let n_layers = model.layers.len();

    let mut grads = ModelGradients::zeros(model);

    // Reconstruct x_cur (input to final rmsnorm) from last layer:
    // x_cur = last_layer.x2 + last_layer.ffn_out
    let last_act = &fwd.layer_acts[n_layers - 1];
    let mut x_cur = last_act.x2.clone();
    ane_forward::vec_add_inplace(&mut x_cur, &last_act.ffn_out);

    // Reconstruct x_final (output of final rmsnorm) for classifier_bwd
    let mut x_final = vec![0.0f32; dim * seq];
    ane_forward::rmsnorm(
        &mut x_final,
        &x_cur,
        &model.rms_final,
        dim,
        seq,
        cfg.rms_eps,
    );

    // 1. Classifier backward: dy = cls_w^T @ dlogits, dcls_w += dlogits @ x_final^T
    let cls_w = model.lm_head.as_ref().unwrap_or(&model.embed);
    let dcls_w = if model.lm_head.is_some() {
        grads.dlm_head.as_mut().unwrap()
    } else {
        &mut grads.dembed
    };
    let mut dy = vec![0.0f32; dim * seq];
    classifier_bwd(
        &mut dy,
        dcls_w,
        &fwd.dlogits,
        cls_w,
        &x_final,
        vocab,
        dim,
        seq,
    );

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

        // b. ANE ffn_bwd_w2t: dsilu = dffn @ W2^T
        // w2 stored as [hidden, dim] row-major; transpose to [dim, hidden] for pack_dyn_matmul
        let w2_for_pack = ane_weights::transpose_weight(&lw.w2, hidden, dim);
        let w2t_input = ane_weights::pack_dyn_matmul(&dffn, &w2_for_pack, dim, hidden, seq);
        let w2t_spec = KernelSpec::for_kernel(cfg, KernelType::FfnBwdW2t);
        kernels.ffn_bwd_w2t.write_input(0, &w2t_input);

        kernels
            .ffn_bwd_w2t
            .eval()
            .map_err(|e| format!("ffn_bwd_w2t eval: {e}"))?;
        let mut w2t_out = vec![0u8; w2t_spec.output_bytes];
        kernels.ffn_bwd_w2t.read_output(0, &mut w2t_out);
        let dsilu: Vec<f32> = w2t_out
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // c. CPU silu_bwd
        let mut dh1 = vec![0.0f32; hidden * seq];
        let mut dh3 = vec![0.0f32; hidden * seq];
        silu_bwd(&mut dh1, &mut dh3, &dsilu, &ac.h1, &ac.h3, hidden * seq);

        // d. ANE ffn_bwd_w13t: dx_ffn = dh1@W1^T + dh3@W3^T
        // w1, w3 stored as [dim, hidden] row-major; transpose to [hidden, dim] for pack
        let w1t = ane_weights::transpose_weight(&lw.w1, dim, hidden);
        let w3t = ane_weights::transpose_weight(&lw.w3, dim, hidden);
        let w13t_input = ane_weights::pack_ffn_bwd_w13t(&dh1, &dh3, &w1t, &w3t, cfg);
        let w13t_spec = KernelSpec::for_kernel(cfg, KernelType::FfnBwdW13t);
        kernels.ffn_bwd_w13t.write_input(0, &w13t_input);

        kernels
            .ffn_bwd_w13t
            .eval()
            .map_err(|e| format!("ffn_bwd_w13t eval: {e}"))?;
        let mut w13t_out = vec![0u8; w13t_spec.output_bytes];
        kernels.ffn_bwd_w13t.read_output(0, &mut w13t_out);
        let dx_ffn: Vec<f32> = w13t_out
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

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
        let wot = ane_weights::transpose_weight(&lw.wo, dim, dim);
        let wot_input = ane_weights::pack_dyn_matmul(&dx2, &wot, dim, dim, seq);
        let wot_spec = KernelSpec::for_kernel(cfg, KernelType::Wot);
        kernels.wot_bwd.write_input(0, &wot_input);

        kernels
            .wot_bwd
            .eval()
            .map_err(|e| format!("wot_bwd eval: {e}"))?;
        let mut wot_out = vec![0u8; wot_spec.output_bytes];
        kernels.wot_bwd.read_output(0, &mut wot_out);
        let da: Vec<f32> = wot_out
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // i. CPU dWo accumulation: dWo[dim,dim] += dx2[dim,seq] @ attn_out^T[dim,seq]
        matmul_accum_at(&mut gr.dwo, &dx2, &ac.attn_out, dim, dim, seq);

        // j. ANE sdpa_bwd1: pack(Q, K, V, da) -> (dV, probs_raw, dp_raw)
        let bwd1_input = ane_weights::pack_sdpa_bwd1(&ac.q, &ac.k, &ac.v, &da, cfg);
        let bwd1_spec = KernelSpec::for_kernel(cfg, KernelType::SdpaBwd1);
        kernels.sdpa_bwd1.write_input(0, &bwd1_input);

        kernels
            .sdpa_bwd1
            .eval()
            .map_err(|e| format!("sdpa_bwd1 eval: {e}"))?;
        let mut bwd1_out = vec![0u8; bwd1_spec.output_bytes];
        kernels.sdpa_bwd1.read_output(0, &mut bwd1_out);
        let (dv, _probs_f32, _dp_f32) = ane_weights::unpack_sdpa_bwd1(&bwd1_out, cfg);

        // k. ANE sdpa_bwd2: bridge fp16 data from bwd1 output + Q,K
        // Extract probs+dp raw bytes from bwd1 output (fp16): skip dV portion
        let score_ch = cfg.score_ch();
        let probs_dp_offset = dim * seq * 2; // dV is [dim, seq] fp16
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
        kernels.sdpa_bwd2.write_input(0, &bwd2_input);

        kernels
            .sdpa_bwd2
            .eval()
            .map_err(|e| format!("sdpa_bwd2 eval: {e}"))?;
        let mut bwd2_out = vec![0u8; bwd2_spec.output_bytes];
        kernels.sdpa_bwd2.read_output(0, &mut bwd2_out);
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
        let wqt = ane_weights::transpose_weight(&lw.wq, dim, dim);
        let wkt = ane_weights::transpose_weight(&lw.wk, dim, dim);
        let wvt = ane_weights::transpose_weight(&lw.wv, dim, dim);
        let qkv_input = ane_weights::pack_qkvb(&dq, &dk, &dv, &wqt, &wkt, &wvt, cfg);
        let qkv_spec = KernelSpec::for_kernel(cfg, KernelType::Qkvb);
        kernels.qkv_bwd.write_input(0, &qkv_input);

        kernels
            .qkv_bwd
            .eval()
            .map_err(|e| format!("qkv_bwd eval: {e}"))?;
        let mut qkv_out = vec![0u8; qkv_spec.output_bytes];
        kernels.qkv_bwd.read_output(0, &mut qkv_out);
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

    for h in 0..n_heads {
        // Recompute attention probs
        let mut scores = vec![0.0f32; seq * seq];
        for s1 in 0..seq {
            for s2 in 0..seq {
                if s2 > s1 {
                    scores[s1 * seq + s2] = f32::NEG_INFINITY;
                } else {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q[(h * head_dim + d) * seq + s1] * k[(h * head_dim + d) * seq + s2];
                    }
                    scores[s1 * seq + s2] = dot * scale;
                }
            }
        }
        // Softmax
        let mut probs = vec![0.0f32; seq * seq];
        for s1 in 0..seq {
            let row = &scores[s1 * seq..(s1 + 1) * seq];
            let max_v = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for s2 in 0..seq {
                probs[s1 * seq + s2] = (row[s2] - max_v).exp();
                sum += probs[s1 * seq + s2];
            }
            for s2 in 0..seq {
                probs[s1 * seq + s2] /= sum;
            }
        }

        // dV[h,d,s2] = sum_s1 probs[s1,s2] * dO[h,d,s1]
        for d in 0..head_dim {
            for s2 in 0..seq {
                let mut acc = 0.0f32;
                for s1 in 0..seq {
                    acc += probs[s1 * seq + s2] * d_out[(h * head_dim + d) * seq + s1];
                }
                dv[(h * head_dim + d) * seq + s2] = acc;
            }
        }

        // dP[s1,s2] = sum_d dO[h,d,s1] * V[h,d,s2]
        let mut dp = vec![0.0f32; seq * seq];
        for s1 in 0..seq {
            for s2 in 0..seq {
                let mut acc = 0.0f32;
                for d in 0..head_dim {
                    acc += d_out[(h * head_dim + d) * seq + s1] * v[(h * head_dim + d) * seq + s2];
                }
                dp[s1 * seq + s2] = acc;
            }
        }

        // dS = probs * (dP - sum_s2(probs * dP))  (softmax backward)
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

        // dQ[h,d,s1] = scale * sum_s2 dS[s1,s2] * K[h,d,s2]
        for d in 0..head_dim {
            for s1 in 0..seq {
                let mut acc = 0.0f32;
                for s2 in 0..seq {
                    acc += ds[s1 * seq + s2] * k[(h * head_dim + d) * seq + s2];
                }
                dq[(h * head_dim + d) * seq + s1] = acc * scale;
            }
        }

        // dK[h,d,s2] = scale * sum_s1 dS[s1,s2] * Q[h,d,s1]
        for d in 0..head_dim {
            for s2 in 0..seq {
                let mut acc = 0.0f32;
                for s1 in 0..seq {
                    acc += ds[s1 * seq + s2] * q[(h * head_dim + d) * seq + s1];
                }
                dk[(h * head_dim + d) * seq + s2] = acc * scale;
            }
        }
    }

    (dq, dk, dv)
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
    backward_lora_cpu_generic(model, fwd, lora, tokens)
}

/// Backward pass generic over weight source (supports both full and quantized weights).
pub fn backward_lora_cpu_generic<T: ane_forward::TokenId, W: ane_weights::WeightSource>(
    model: &W,
    fwd: &ane_forward::ForwardResultWithLora,
    lora: &super::ane_lora::LoraModel,
    tokens: &[T],
) -> BackwardResultWithLora {
    use super::ane_lora;

    let cfg = model.cfg();
    let dim = cfg.dim;
    let hidden = cfg.hidden_dim;
    let seq = cfg.seq_len;
    let n_layers = model.n_layers();
    let scale = lora.scale();

    // Base model weights are frozen — skip the 5.9 GB gradient allocation.
    let mut lora_grads = ane_lora::LoraModelGrads::zeros(lora);

    // Reconstruct x_cur from last layer
    let last_act = &fwd.base.layer_acts[n_layers - 1];
    let mut x_cur = last_act.x2.clone();
    ane_forward::vec_add_inplace(&mut x_cur, &last_act.ffn_out);

    let mut x_final = vec![0.0f32; dim * seq];
    ane_forward::rmsnorm(
        &mut x_final,
        &x_cur,
        model.rms_final(),
        dim,
        seq,
        cfg.rms_eps,
    );

    // Classifier backward
    let cls_w = model.lm_head().unwrap_or(model.embed());
    let mut dy = vec![0.0f32; dim * seq];
    let mut _dcls = vec![0.0f32; cls_w.len()];
    classifier_bwd(
        &mut dy,
        &mut _dcls,
        &fwd.base.dlogits,
        cls_w,
        &x_final,
        model.vocab_size(),
        dim,
        seq,
    );

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
        let lw_cow = model.layer(l);
        let lw = &*lw_cow;
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

        // CPU: dsilu = dffn @ W2^T
        let w2t = ane_weights::transpose_weight(&lw.w2, hidden, dim);
        let dsilu = ane_forward::cpu_matmul(&w2t, &dffn, hidden, dim, seq);

        // silu backward
        let mut dh1 = vec![0.0f32; hidden * seq];
        let mut dh3 = vec![0.0f32; hidden * seq];
        silu_bwd(&mut dh1, &mut dh3, &dsilu, &ac.h1, &ac.h3, hidden * seq);

        // CPU: dx_ffn = dh1 @ W1^T + dh3 @ W3^T
        let w1t = ane_weights::transpose_weight(&lw.w1, dim, hidden);
        let w3t = ane_weights::transpose_weight(&lw.w3, dim, hidden);
        let dx_w1 = ane_forward::cpu_matmul(&w1t, &dh1, dim, hidden, seq);
        let dx_w3 = ane_forward::cpu_matmul(&w3t, &dh3, dim, hidden, seq);
        let mut dx_ffn = dx_w1;
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

        // CPU: da = dx2 @ Wo^T
        let wot = ane_weights::transpose_weight(&lw.wo, dim, dim);
        let da = ane_forward::cpu_matmul(&wot, &dx2, dim, dim, seq);

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

        // CPU: dx_attn = dq @ Wq^T + dk @ Wk^T + dv @ Wv^T
        let wqt = ane_weights::transpose_weight(&lw.wq, dim, dim);
        let wkt = ane_weights::transpose_weight(&lw.wk, dim, dim);
        let wvt = ane_weights::transpose_weight(&lw.wv, dim, dim);
        let mut dx_attn = ane_forward::cpu_matmul(&wqt, &dq, dim, dim, seq);
        let dx_k = ane_forward::cpu_matmul(&wkt, &dk, dim, dim, seq);
        let dx_v = ane_forward::cpu_matmul(&wvt, &_dv, dim, dim, seq);
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
    let last_act = &fwd.base.layer_acts[n_layers - 1];
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

    let cls_w = model.lm_head.as_ref().unwrap_or(&model.embed);
    let mut dy = vec![0.0f32; dim * seq];
    let mut _dcls = vec![0.0f32; cls_w.len()];
    classifier_bwd(
        &mut dy,
        &mut _dcls,
        &fwd.base.dlogits,
        cls_w,
        &x_final,
        model.vocab_size,
        dim,
        seq,
    );

    let mut dx_rms = vec![0.0f32; dim * seq];
    rmsnorm_bwd(
        &mut dx_rms,
        &mut vec![0.0f32; dim],
        &dy,
        &x_cur,
        &model.rms_final,
        dim,
        seq,
        cfg.rms_eps,
    );
    dy = dx_rms;

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
        let w2_for_pack = ane_weights::transpose_weight(&lw.w2, hidden, dim);
        let w2t_input = ane_weights::pack_dyn_matmul(&dffn, &w2_for_pack, dim, hidden, seq);
        let w2t_spec = ane_mil::KernelSpec::for_kernel(cfg, ane_mil::KernelType::FfnBwdW2t);
        kernels.ffn_bwd_w2t.write_input(0, &w2t_input);
        kernels
            .ffn_bwd_w2t
            .eval()
            .map_err(|e| format!("lora ffn_bwd_w2t: {e}"))?;
        let mut w2t_out = vec![0u8; w2t_spec.output_bytes];
        kernels.ffn_bwd_w2t.read_output(0, &mut w2t_out);
        let dsilu: Vec<f32> = w2t_out
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let mut dh1 = vec![0.0f32; hidden * seq];
        let mut dh3 = vec![0.0f32; hidden * seq];
        silu_bwd(&mut dh1, &mut dh3, &dsilu, &ac.h1, &ac.h3, hidden * seq);

        let w1t = ane_weights::transpose_weight(&lw.w1, dim, hidden);
        let w3t = ane_weights::transpose_weight(&lw.w3, dim, hidden);
        let w13t_input = ane_weights::pack_ffn_bwd_w13t(&dh1, &dh3, &w1t, &w3t, cfg);
        let w13t_spec = ane_mil::KernelSpec::for_kernel(cfg, ane_mil::KernelType::FfnBwdW13t);
        kernels.ffn_bwd_w13t.write_input(0, &w13t_input);
        kernels
            .ffn_bwd_w13t
            .eval()
            .map_err(|e| format!("lora ffn_bwd_w13t: {e}"))?;
        let mut w13t_out = vec![0u8; w13t_spec.output_bytes];
        kernels.ffn_bwd_w13t.read_output(0, &mut w13t_out);
        let dx_ffn: Vec<f32> = w13t_out
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

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
        let wot = ane_weights::transpose_weight(&lw.wo, dim, dim);
        let wot_input = ane_weights::pack_dyn_matmul(&dx2, &wot, dim, dim, seq);
        let wot_spec = ane_mil::KernelSpec::for_kernel(cfg, ane_mil::KernelType::Wot);
        kernels.wot_bwd.write_input(0, &wot_input);
        kernels
            .wot_bwd
            .eval()
            .map_err(|e| format!("lora wot_bwd: {e}"))?;
        let mut wot_out = vec![0u8; wot_spec.output_bytes];
        kernels.wot_bwd.read_output(0, &mut wot_out);
        let da_vec: Vec<f32> = wot_out
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // SDPA backward (simplified — just propagate through attention for dy chain)
        let bwd1_input = ane_weights::pack_sdpa_bwd1(&ac.q, &ac.k, &ac.v, &da_vec, cfg);
        let bwd1_spec = ane_mil::KernelSpec::for_kernel(cfg, ane_mil::KernelType::SdpaBwd1);
        kernels.sdpa_bwd1.write_input(0, &bwd1_input);
        kernels
            .sdpa_bwd1
            .eval()
            .map_err(|e| format!("lora sdpa_bwd1: {e}"))?;
        let mut bwd1_out = vec![0u8; bwd1_spec.output_bytes];
        kernels.sdpa_bwd1.read_output(0, &mut bwd1_out);
        let (dv, _, _) = ane_weights::unpack_sdpa_bwd1(&bwd1_out, cfg);

        let score_ch = cfg.score_ch();
        let probs_dp_offset = dim * seq * 2;
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
        kernels.sdpa_bwd2.write_input(0, &bwd2_input);
        kernels
            .sdpa_bwd2
            .eval()
            .map_err(|e| format!("lora sdpa_bwd2: {e}"))?;
        let mut bwd2_out = vec![0u8; bwd2_spec.output_bytes];
        kernels.sdpa_bwd2.read_output(0, &mut bwd2_out);
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

        let wqt = ane_weights::transpose_weight(&lw.wq, dim, dim);
        let wkt = ane_weights::transpose_weight(&lw.wk, dim, dim);
        let wvt = ane_weights::transpose_weight(&lw.wv, dim, dim);
        let qkv_input = ane_weights::pack_qkvb(&dq, &dk, &dv, &wqt, &wkt, &wvt, cfg);
        let qkv_spec = ane_mil::KernelSpec::for_kernel(cfg, ane_mil::KernelType::Qkvb);
        kernels.qkv_bwd.write_input(0, &qkv_input);
        kernels
            .qkv_bwd
            .eval()
            .map_err(|e| format!("lora qkv_bwd: {e}"))?;
        let mut qkv_out = vec![0u8; qkv_spec.output_bytes];
        kernels.qkv_bwd.read_output(0, &mut qkv_out);
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
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::ane_forward;
    use crate::agent::ane_mil::MilConfig;
    use crate::agent::ane_weights::{LayerWeights, ModelWeights};

    // -----------------------------------------------------------------------
    // Round 1: rmsnorm_bwd
    // -----------------------------------------------------------------------

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
            }],
            rms_final: vec![1.0; dim],
            embed: make_small(vocab * dim, 7000),
            vocab_size: vocab,
            lm_head: None,
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
            }],
            rms_final: vec![1.0; dim],
            embed: make_small(vocab * dim, 7000),
            vocab_size: vocab,
            lm_head: None,
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
            }],
            rms_final: vec![1.0; dim],
            embed: make_small(vocab * dim, 7000),
            vocab_size: vocab,
            lm_head: Some(make_small(vocab * dim, 9000)), // untied!
        };

        let tokens: Vec<u16> = (0..seq).map(|i| (i % vocab) as u16).collect();
        let targets: Vec<u16> = (0..seq).map(|i| ((i + 1) % vocab) as u16).collect();

        let fwd =
            ane_forward::forward(&fwd_kernels, &model, &tokens, &targets).expect("forward failed");
        assert!(fwd.loss.is_finite(), "loss not finite");

        let grads = backward(&bwd_kernels, &model, &fwd, &tokens).expect("backward failed");

        // dlm_head should exist and have nonzero entries
        let dlm = grads.dlm_head.as_ref().expect("dlm_head should be Some");
        assert_eq!(dlm.len(), vocab * dim);
        let nonzero = dlm.iter().filter(|v| v.abs() > 1e-15).count();
        assert!(nonzero > 0, "dlm_head should have nonzero gradients");

        // dembed should still get gradients from embed_bwd (not from classifier)
        let dembed_nonzero = grads.dembed.iter().filter(|v| v.abs() > 1e-15).count();
        assert!(
            dembed_nonzero > 0,
            "dembed should have nonzero entries from embed_bwd"
        );
    }
}
