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
pub fn rmsnorm(out: &mut [f32], x: &[f32], w: &[f32], dim: usize, seq: usize) {
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
    let eps = 1e-5f32;
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
}

impl CompiledKernels {
    /// Compile all forward-pass kernels for the given config.
    pub fn compile_forward(cfg: &MilConfig) -> Result<Self, String> {
        ane_bridge::ane_init()?;

        // SDPA forward needs causal mask as a static weight
        let mask_blob = ane_mil::build_causal_mask_blob(cfg.seq_len);
        let sdpa_spec = KernelSpec::for_kernel(cfg, KernelType::SdpaFwd);
        let sdpa_fwd = AneKernel::compile_multi_weights(
            &sdpa_spec.mil_text,
            &["@model_path/weights/mask.bin"],
            &[&mask_blob],
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
    pub q: Vec<f32>,         // [dim, seq]
    pub k: Vec<f32>,         // [dim, seq]
    pub v: Vec<f32>,         // [dim, seq]
    pub attn_out: Vec<f32>,  // [dim, seq]
    pub o_out: Vec<f32>,     // [dim, seq]
    pub x2: Vec<f32>,        // [dim, seq]
    pub x2norm: Vec<f32>,    // [dim, seq]
    pub h1: Vec<f32>,        // [hidden, seq]
    pub h3: Vec<f32>,        // [hidden, seq]
    pub gate: Vec<f32>,      // [hidden, seq]  (silu(h1)*h3)
    pub ffn_out: Vec<f32>,   // [dim, seq]
}

/// Forward pass result.
pub struct ForwardResult {
    pub logits: Vec<f32>,               // [vocab, seq]
    pub loss: f32,
    pub dlogits: Vec<f32>,              // [vocab, seq]
    pub layer_acts: Vec<LayerActivations>,
}

/// Run full forward pass: embed → layers → classifier → loss.
///
/// Follows train.m lines 400-506:
/// 1. embed_lookup → x_cur[dim, seq]
/// 2. Per layer: rmsnorm → SDPA(ANE) → residual → rmsnorm → FFN(ANE) → residual
/// 3. Final rmsnorm → classifier → cross-entropy loss
pub fn forward<T: TokenId>(
    kernels: &CompiledKernels,
    model: &ModelWeights,
    tokens: &[T],
    targets: &[T],
) -> Result<ForwardResult, String> {
    let cfg = &model.cfg;
    let dim = cfg.dim;
    let seq = cfg.seq_len;
    let _hidden = cfg.hidden_dim;
    let n_layers = model.layers.len();

    // 1. Embedding lookup
    let mut x_cur = vec![0.0f32; dim * seq];
    embed_lookup(&mut x_cur, &model.embed, tokens, dim, seq);

    let mut layer_acts = Vec::with_capacity(n_layers);

    // 2. Transformer layers
    for l in 0..n_layers {
        let lw = &model.layers[l];

        // Save layer input for backward pass
        let layer_in = x_cur.clone();

        // RMSNorm before attention
        let mut xnorm = vec![0.0f32; dim * seq];
        rmsnorm(&mut xnorm, &x_cur, &lw.rms_att, dim, seq);

        // SDPA forward on ANE
        let sdpa_input = ane_weights::pack_sdpa_fwd(
            &xnorm, &lw.wq, &lw.wk, &lw.wv, &lw.wo, cfg,
        );
        let sdpa_spec = KernelSpec::for_kernel(cfg, KernelType::SdpaFwd);
        kernels.sdpa_fwd.write_input(0, &sdpa_input);
        kernels.sdpa_fwd.eval()?;
        let mut sdpa_out = vec![0u8; sdpa_spec.output_bytes];
        kernels.sdpa_fwd.read_output(0, &mut sdpa_out);
        let [o_out, q, k, v, attn_out, _xnorm_pass] =
            ane_weights::unpack_sdpa_fwd(&sdpa_out, cfg);

        // Residual: x2 = x_cur + o_out
        let mut x2 = x_cur.clone();
        vec_add_inplace(&mut x2, &o_out);

        // RMSNorm before FFN
        let mut x2norm = vec![0.0f32; dim * seq];
        rmsnorm(&mut x2norm, &x2, &lw.rms_ffn, dim, seq);

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
        let ffn_out: Vec<f32> = w2_out
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

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
        });
    }

    // 3. Final RMSNorm
    let mut x_final = vec![0.0f32; dim * seq];
    rmsnorm(&mut x_final, &x_cur, &model.rms_final, dim, seq);

    // 4. Classifier
    let vocab = model.vocab_size;
    let mut logits = vec![0.0f32; vocab * seq];
    classifier_forward(&mut logits, &model.embed, &x_final, vocab, dim, seq);

    // 5. Cross-entropy loss
    let (loss, dlogits) = cross_entropy_loss(&logits, targets, vocab, seq);

    Ok(ForwardResult {
        logits,
        loss,
        dlogits,
        layer_acts,
    })
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

        rmsnorm(&mut out, &x, &w, dim, seq);

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

        rmsnorm(&mut out, &x, &w, dim, seq);

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
}
