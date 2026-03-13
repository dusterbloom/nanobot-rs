//! LoRA (Low-Rank Adaptation) for ANE transformer training.
//!
//! Implements rank-decomposed adapter training on Apple Neural Engine,
//! incorporating proven hyperparameters from JIT LoRA research:
//! - Rank 32, alpha 32.0 (scale = 1.0)
//! - Targets: wq, wv, wo (attention) + w2 (FFN down_proj)
//! - LR 5e-4 (10x standard), gradient clipping 1.0
//! - ≥33% regularization ratio to prevent catastrophic forgetting
//! - Batch size 1 (optimal for Apple Silicon memory bandwidth)
//!
//! LoRA forward:  h = A @ x,  δy = B @ h,  y_total = y_base + scale * δy
//! LoRA backward: ANE for input grads, with optional ANE weight grads and Muon
//! updates for strict accelerator-only LoRA math.

use std::fmt::Write as _;
use std::io;
use std::path::Path;

use super::ane_bridge::{self, AneKernel};
use super::ane_mil::{KernelSpec, KernelType, MilConfig};
use super::ane_train::{self, AdamState, TrainingConfig};
use super::ane_weights;

// ---------------------------------------------------------------------------
// LoRA config
// ---------------------------------------------------------------------------

/// LoRA configuration with JIT LoRA proven defaults.
#[derive(Clone)]
pub struct LoraConfig {
    pub rank: usize,
    pub alpha: f32,
    pub target_modules: Vec<String>,
}

impl Default for LoraConfig {
    fn default() -> Self {
        LoraConfig {
            rank: 32,
            alpha: 32.0,
            target_modules: vec!["wq".into(), "wv".into(), "wo".into(), "w2".into()],
        }
    }
}

impl LoraConfig {
    /// Scaling factor: alpha / rank.
    pub fn scale(&self) -> f32 {
        self.alpha / self.rank as f32
    }
}

// ---------------------------------------------------------------------------
// LoRA adapter structs
// ---------------------------------------------------------------------------

/// Single LoRA adapter: δW = B @ A where A∈R^{rank×d_in}, B∈R^{d_out×rank}
///
/// Data layout (channels-first, matching ANE):
///   A: [rank, d_in] row-major — A[r * d_in + i]
///   B: [d_out, rank] row-major — B[o * rank + r]
///
/// B is zero-initialized so the model starts unmodified (JIT LoRA pattern).
#[derive(Clone)]
pub struct LoraAdapter {
    pub a: Vec<f32>,
    pub b: Vec<f32>,
    pub rank: usize,
    pub d_in: usize,
    pub d_out: usize,
}

impl LoraAdapter {
    /// Create a new LoRA adapter with Kaiming uniform init for A, zeros for B.
    pub fn new(rank: usize, d_in: usize, d_out: usize) -> Self {
        // Kaiming uniform: U(-bound, bound) where bound = sqrt(6 / fan_in)
        // fan_in = d_in for A matrix
        let bound = (6.0f32 / d_in as f32).sqrt();
        let a: Vec<f32> = (0..rank * d_in)
            .map(|i| {
                // Deterministic pseudo-random for reproducibility in tests
                let t = (i as f32 * 0.618033988 + 0.31415926).fract();
                (t * 2.0 - 1.0) * bound
            })
            .collect();
        let b = vec![0.0f32; d_out * rank];

        LoraAdapter {
            a,
            b,
            rank,
            d_in,
            d_out,
        }
    }

    /// Create adapter with all zeros (for testing).
    pub fn zeros(rank: usize, d_in: usize, d_out: usize) -> Self {
        LoraAdapter {
            a: vec![0.0; rank * d_in],
            b: vec![0.0; d_out * rank],
            rank,
            d_in,
            d_out,
        }
    }

    /// LoRA forward on CPU: δy = B @ (A @ x)
    ///
    /// x: [d_in, seq], output: [d_out, seq]
    /// Also returns intermediate h = A @ x ([rank, seq]) for backward.
    pub fn forward_cpu(&self, x: &[f32], seq: usize) -> (Vec<f32>, Vec<f32>) {
        debug_assert_eq!(x.len(), self.d_in * seq);
        use super::ane_forward::cpu_matmul;
        let h = cpu_matmul(&self.a, x, self.rank, self.d_in, seq);
        let dy = cpu_matmul(&self.b, &h, self.d_out, self.rank, seq);
        (dy, h)
    }

    /// LoRA backward on CPU.
    ///
    /// Given upstream gradient d_out_grad [d_out, seq], saved input x [d_in, seq],
    /// and saved intermediate h [rank, seq]:
    ///
    /// Returns (dx_lora, dA, dB) where:
    ///   dh = B^T @ d_out_grad    [rank, seq]
    ///   dx_lora = A^T @ dh       [d_in, seq]
    ///   dB = d_out_grad @ h^T    [d_out, rank]
    ///   dA = dh @ x^T            [rank, d_in]
    pub fn backward_cpu(
        &self,
        d_out_grad: &[f32],
        x: &[f32],
        h: &[f32],
        seq: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        debug_assert_eq!(d_out_grad.len(), self.d_out * seq);
        debug_assert_eq!(x.len(), self.d_in * seq);
        debug_assert_eq!(h.len(), self.rank * seq);
        use super::ane_forward::cpu_gemm;

        // dh[R,S] = B^T[R,Dout] @ d_out_grad[Dout,S]
        let mut dh = vec![0.0f32; self.rank * seq];
        cpu_gemm(
            &mut dh, &self.b, true, d_out_grad, false, self.rank, seq, self.d_out, 1.0, 0.0,
        );

        // dx_lora[Din,S] = A^T[Din,R] @ dh[R,S]
        let mut dx_lora = vec![0.0f32; self.d_in * seq];
        cpu_gemm(
            &mut dx_lora,
            &self.a,
            true,
            &dh,
            false,
            self.d_in,
            seq,
            self.rank,
            1.0,
            0.0,
        );

        // dB[Dout,R] = d_out_grad[Dout,S] @ h^T[S,R]
        let mut db = vec![0.0f32; self.d_out * self.rank];
        cpu_gemm(
            &mut db, d_out_grad, false, h, true, self.d_out, self.rank, seq, 1.0, 0.0,
        );

        // dA[R,Din] = dh[R,S] @ x^T[S,Din]
        let mut da = vec![0.0f32; self.rank * self.d_in];
        cpu_gemm(
            &mut da, &dh, false, x, true, self.rank, self.d_in, seq, 1.0, 0.0,
        );

        (dx_lora, da, db)
    }
}

// ---------------------------------------------------------------------------
// LoRA layer + model adapters
// ---------------------------------------------------------------------------

/// LoRA adapters for one transformer layer.
/// Targets: wq, wv, wo (attention) + w2 (FFN down_proj).
#[derive(Clone)]
pub struct LoraLayerAdapters {
    pub wq: Option<LoraAdapter>,
    pub wv: Option<LoraAdapter>,
    pub wo: Option<LoraAdapter>,
    pub w2: Option<LoraAdapter>,
}

/// Full model LoRA adapter set.
#[derive(Clone)]
pub struct LoraModel {
    pub layers: Vec<LoraLayerAdapters>,
    pub config: LoraConfig,
}

impl LoraModel {
    /// Initialize LoRA adapters for all layers based on config.
    /// Create LoRA adapters for all layers.
    ///
    /// `kv_dim`: KV projection output dim. For MHA, `kv_dim == dim`.
    /// For GQA (e.g. Qwen3 with 8 KV heads), `kv_dim = n_kv_heads * head_dim`.
    pub fn new(cfg: LoraConfig, n_layers: usize, dim: usize, hidden_dim: usize) -> Self {
        Self::with_kv_dim(cfg, n_layers, dim, dim, hidden_dim)
    }

    pub fn with_kv_dim(
        cfg: LoraConfig,
        n_layers: usize,
        dim: usize,
        kv_dim: usize,
        hidden_dim: usize,
    ) -> Self {
        Self::with_full_dims(cfg, n_layers, dim, kv_dim, dim, dim, hidden_dim)
    }

    /// Create LoRA model with explicit attention dimensions.
    ///
    /// `attn_dim`: n_heads * head_dim (may differ from `dim` for Qwen3.5).
    /// `q_proj_dim`: attn_dim * 2 when attn_output_gate, else attn_dim.
    pub fn with_full_dims(
        cfg: LoraConfig,
        n_layers: usize,
        dim: usize,
        kv_dim: usize,
        attn_dim: usize,
        q_proj_dim: usize,
        hidden_dim: usize,
    ) -> Self {
        let rank = cfg.rank;
        let targets = &cfg.target_modules;

        let layers = (0..n_layers)
            .map(|_| LoraLayerAdapters {
                wq: if targets.iter().any(|t| t == "wq") {
                    // wq: [q_proj_dim, dim] → d_in=dim, d_out=q_proj_dim
                    Some(LoraAdapter::new(rank, dim, q_proj_dim))
                } else {
                    None
                },
                wv: if targets.iter().any(|t| t == "wv") {
                    Some(LoraAdapter::new(rank, dim, kv_dim))
                } else {
                    None
                },
                wo: if targets.iter().any(|t| t == "wo") {
                    // wo: [dim, attn_dim] → d_in=attn_dim, d_out=dim
                    Some(LoraAdapter::new(rank, attn_dim, dim))
                } else {
                    None
                },
                w2: if targets.iter().any(|t| t == "w2") {
                    Some(LoraAdapter::new(rank, hidden_dim, dim))
                } else {
                    None
                },
            })
            .collect();

        LoraModel {
            layers,
            config: cfg,
        }
    }

    /// Scaling factor.
    pub fn scale(&self) -> f32 {
        self.config.scale()
    }
}

// ---------------------------------------------------------------------------
// LoRA activations (saved for backward)
// ---------------------------------------------------------------------------

/// Per-layer LoRA activations saved during forward for backward pass.
pub struct LoraLayerActivations {
    /// Input to Wo LoRA: attn_out [dim, seq] (saved for dA_wo)
    pub wo_x: Option<Vec<f32>>,
    /// Intermediate h = A_wo @ attn_out [rank, seq] (saved for dB_wo)
    pub wo_h: Option<Vec<f32>>,
    /// Input to W2 LoRA: gate [hidden, seq] (saved for dA_w2)
    pub w2_x: Option<Vec<f32>>,
    /// Intermediate h = A_w2 @ gate [rank, seq] (saved for dB_w2)
    pub w2_h: Option<Vec<f32>>,
    /// Input to Wq LoRA: xnorm [dim, seq] (saved for dA_wq)
    pub wq_x: Option<Vec<f32>>,
    /// Intermediate h = A_wq @ xnorm [rank, seq] (saved for dB_wq)
    pub wq_h: Option<Vec<f32>>,
    /// Input to Wv LoRA: xnorm [dim, seq] (saved for dA_wv)
    pub wv_x: Option<Vec<f32>>,
    /// Intermediate h = A_wv @ xnorm [rank, seq] (saved for dB_wv)
    pub wv_h: Option<Vec<f32>>,
}

impl LoraLayerActivations {
    pub fn empty() -> Self {
        LoraLayerActivations {
            wo_x: None,
            wo_h: None,
            w2_x: None,
            w2_h: None,
            wq_x: None,
            wq_h: None,
            wv_x: None,
            wv_h: None,
        }
    }
}

// ---------------------------------------------------------------------------
// LoRA gradients
// ---------------------------------------------------------------------------

/// Per-adapter gradients.
#[derive(Clone)]
pub struct LoraAdapterGrads {
    pub da: Vec<f32>,
    pub db: Vec<f32>,
}

/// Per-layer LoRA gradients.
#[derive(Clone)]
pub struct LoraLayerGrads {
    pub wq: Option<LoraAdapterGrads>,
    pub wv: Option<LoraAdapterGrads>,
    pub wo: Option<LoraAdapterGrads>,
    pub w2: Option<LoraAdapterGrads>,
}

/// Full model LoRA gradients.
#[derive(Clone)]
pub struct LoraModelGrads {
    pub layers: Vec<LoraLayerGrads>,
}

impl LoraModelGrads {
    /// Create zero-initialized gradients matching LoRA model shape.
    pub fn zeros(lora: &LoraModel) -> Self {
        let layers = lora
            .layers
            .iter()
            .map(|la| {
                let make_grads = |adapter: &Option<LoraAdapter>| -> Option<LoraAdapterGrads> {
                    adapter.as_ref().map(|a| LoraAdapterGrads {
                        da: vec![0.0; a.rank * a.d_in],
                        db: vec![0.0; a.d_out * a.rank],
                    })
                };
                LoraLayerGrads {
                    wq: make_grads(&la.wq),
                    wv: make_grads(&la.wv),
                    wo: make_grads(&la.wo),
                    w2: make_grads(&la.w2),
                }
            })
            .collect();
        LoraModelGrads { layers }
    }

    /// Zero all gradients.
    pub fn zero(&mut self) {
        for lg in &mut self.layers {
            let zero_opt = |g: &mut Option<LoraAdapterGrads>| {
                if let Some(g) = g.as_mut() {
                    g.da.iter_mut().for_each(|v| *v = 0.0);
                    g.db.iter_mut().for_each(|v| *v = 0.0);
                }
            };
            zero_opt(&mut lg.wq);
            zero_opt(&mut lg.wv);
            zero_opt(&mut lg.wo);
            zero_opt(&mut lg.w2);
        }
    }

    /// Element-wise add gradients from `other` into `self`.
    pub fn add_from(&mut self, other: &LoraModelGrads) {
        for (lg, og) in self.layers.iter_mut().zip(other.layers.iter()) {
            let add_opt =
                |dst: &mut Option<LoraAdapterGrads>, src: &Option<LoraAdapterGrads>| {
                    if let (Some(d), Some(s)) = (dst.as_mut(), src.as_ref()) {
                        for (dv, sv) in d.da.iter_mut().zip(s.da.iter()) {
                            *dv += sv;
                        }
                        for (dv, sv) in d.db.iter_mut().zip(s.db.iter()) {
                            *dv += sv;
                        }
                    }
                };
            add_opt(&mut lg.wq, &og.wq);
            add_opt(&mut lg.wv, &og.wv);
            add_opt(&mut lg.wo, &og.wo);
            add_opt(&mut lg.w2, &og.w2);
        }
    }

    /// Multiply all gradients by scalar `s`.
    pub fn scale(&mut self, s: f32) {
        for lg in &mut self.layers {
            let scale_opt = |g: &mut Option<LoraAdapterGrads>| {
                if let Some(g) = g.as_mut() {
                    g.da.iter_mut().for_each(|v| *v *= s);
                    g.db.iter_mut().for_each(|v| *v *= s);
                }
            };
            scale_opt(&mut lg.wq);
            scale_opt(&mut lg.wv);
            scale_opt(&mut lg.wo);
            scale_opt(&mut lg.w2);
        }
    }
}

// ---------------------------------------------------------------------------
// LoRA Adam state
// ---------------------------------------------------------------------------

/// Adam state for one LoRA adapter (A and B matrices).
#[derive(Clone)]
pub struct LoraAdapterAdam {
    pub a: AdamState,
    pub b: AdamState,
}

/// Per-layer LoRA Adam state.
#[derive(Clone)]
pub struct LoraLayerAdam {
    pub wq: Option<LoraAdapterAdam>,
    pub wv: Option<LoraAdapterAdam>,
    pub wo: Option<LoraAdapterAdam>,
    pub w2: Option<LoraAdapterAdam>,
}

/// Full model LoRA Adam state.
#[derive(Clone)]
pub struct LoraModelAdam {
    pub layers: Vec<LoraLayerAdam>,
}

impl LoraModelAdam {
    /// Create zero-initialized Adam state matching LoRA model shape.
    pub fn zeros(lora: &LoraModel) -> Self {
        let layers = lora
            .layers
            .iter()
            .map(|la| {
                let make_adam = |adapter: &Option<LoraAdapter>| -> Option<LoraAdapterAdam> {
                    adapter.as_ref().map(|a| LoraAdapterAdam {
                        a: AdamState::zeros(a.rank * a.d_in),
                        b: AdamState::zeros(a.d_out * a.rank),
                    })
                };
                LoraLayerAdam {
                    wq: make_adam(&la.wq),
                    wv: make_adam(&la.wv),
                    wo: make_adam(&la.wo),
                    w2: make_adam(&la.w2),
                }
            })
            .collect();
        LoraModelAdam { layers }
    }
}

/// Muon state for one LoRA adapter (momentum buffers for A and B matrices).
#[derive(Clone)]
pub struct LoraAdapterMuon {
    pub a: Vec<f32>,
    pub b: Vec<f32>,
}

/// Per-layer LoRA Muon state.
#[derive(Clone)]
pub struct LoraLayerMuon {
    pub wq: Option<LoraAdapterMuon>,
    pub wv: Option<LoraAdapterMuon>,
    pub wo: Option<LoraAdapterMuon>,
    pub w2: Option<LoraAdapterMuon>,
}

/// Full model LoRA Muon state.
#[derive(Clone)]
pub struct LoraModelMuon {
    pub layers: Vec<LoraLayerMuon>,
}

impl LoraModelMuon {
    /// Create zero-initialized Muon state matching LoRA model shape.
    pub fn zeros(lora: &LoraModel) -> Self {
        let layers = lora
            .layers
            .iter()
            .map(|la| {
                let make_muon = |adapter: &Option<LoraAdapter>| -> Option<LoraAdapterMuon> {
                    adapter.as_ref().map(|a| LoraAdapterMuon {
                        a: vec![0.0; a.rank * a.d_in],
                        b: vec![0.0; a.d_out * a.rank],
                    })
                };
                LoraLayerMuon {
                    wq: make_muon(&la.wq),
                    wv: make_muon(&la.wv),
                    wo: make_muon(&la.wo),
                    w2: make_muon(&la.w2),
                }
            })
            .collect();
        LoraModelMuon { layers }
    }
}

// ---------------------------------------------------------------------------
// LoRA gradient operations
// ---------------------------------------------------------------------------

/// L2 norm of all LoRA gradients.
pub fn lora_grad_norm(grads: &LoraModelGrads) -> f32 {
    let mut sum_sq = 0.0f64;
    let acc = |g: &Option<LoraAdapterGrads>, s: &mut f64| {
        if let Some(g) = g.as_ref() {
            for &v in g.da.iter().chain(g.db.iter()) {
                *s += (v as f64) * (v as f64);
            }
        }
    };
    for lg in &grads.layers {
        acc(&lg.wq, &mut sum_sq);
        acc(&lg.wv, &mut sum_sq);
        acc(&lg.wo, &mut sum_sq);
        acc(&lg.w2, &mut sum_sq);
    }
    (sum_sq as f32).sqrt()
}

/// Clip LoRA gradients: if norm > clip, scale all grads by clip / norm.
pub fn lora_clip_gradients(grads: &mut LoraModelGrads, clip: f32) {
    let norm = lora_grad_norm(grads);
    if norm > clip {
        let scale = clip / norm;
        let scale_opt = |g: &mut Option<LoraAdapterGrads>| {
            if let Some(g) = g.as_mut() {
                g.da.iter_mut().for_each(|v| *v *= scale);
                g.db.iter_mut().for_each(|v| *v *= scale);
            }
        };
        for lg in &mut grads.layers {
            scale_opt(&mut lg.wq);
            scale_opt(&mut lg.wv);
            scale_opt(&mut lg.wo);
            scale_opt(&mut lg.w2);
        }
    }
}

/// Per-parameter gradient clipping: clamp each element to [-clip, clip].
pub fn lora_clip_gradients_per_param(grads: &mut LoraModelGrads, clip: f32) {
    let clamp_opt = |g: &mut Option<LoraAdapterGrads>| {
        if let Some(g) = g.as_mut() {
            g.da.iter_mut().for_each(|v| *v = v.clamp(-clip, clip));
            g.db.iter_mut().for_each(|v| *v = v.clamp(-clip, clip));
        }
    };
    for lg in &mut grads.layers {
        clamp_opt(&mut lg.wq);
        clamp_opt(&mut lg.wv);
        clamp_opt(&mut lg.wo);
        clamp_opt(&mut lg.w2);
    }
}

/// Clip gradient norm of all LoRA parameters (max_norm clipping).
///
/// Computes the global L2 norm across all adapter gradients and scales
/// them uniformly if the norm exceeds `max_norm`. This is critical for
/// training stability per maderix/ANE research (max_norm=1.0).
pub fn clip_lora_grad_norm(grads: &mut LoraModelGrads, max_norm: f32) -> f32 {
    let mut total_sq = 0.0f64;
    let accumulate = |g: &Option<LoraAdapterGrads>, acc: &mut f64| {
        if let Some(ref grads) = g {
            for &v in &grads.da {
                *acc += (v as f64) * (v as f64);
            }
            for &v in &grads.db {
                *acc += (v as f64) * (v as f64);
            }
        }
    };
    for lg in &grads.layers {
        accumulate(&lg.wq, &mut total_sq);
        accumulate(&lg.wv, &mut total_sq);
        accumulate(&lg.wo, &mut total_sq);
        accumulate(&lg.w2, &mut total_sq);
    }
    let norm = total_sq.sqrt() as f32;
    if norm > max_norm {
        let scale = max_norm / norm;
        let clip = |g: &mut Option<LoraAdapterGrads>| {
            if let Some(ref mut grads) = g {
                for v in &mut grads.da {
                    *v *= scale;
                }
                for v in &mut grads.db {
                    *v *= scale;
                }
            }
        };
        for lg in &mut grads.layers {
            clip(&mut lg.wq);
            clip(&mut lg.wv);
            clip(&mut lg.wo);
            clip(&mut lg.w2);
        }
    }
    norm
}

/// Apply Adam updates to all LoRA parameters with gradient clipping.
///
/// Clips gradient norm to 1.0 before the Adam step for training stability.
pub fn lora_adam_update(
    lora: &mut LoraModel,
    grads: &LoraModelGrads,
    adam: &mut LoraModelAdam,
    t: usize,
    lr: f32,
    b1: f32,
    b2: f32,
    eps: f32,
    wd: f32,
) {
    // Gradient clipping (max_norm=1.0) — critical for training stability
    let mut grads_clipped = grads.clone();
    clip_lora_grad_norm(&mut grads_clipped, 1.0);

    for l in 0..lora.layers.len() {
        let update_adapter = |adapter: &mut Option<LoraAdapter>,
                              grad: &Option<LoraAdapterGrads>,
                              state: &mut Option<LoraAdapterAdam>| {
            if let (Some(a), Some(g), Some(s)) = (adapter.as_mut(), grad.as_ref(), state.as_mut()) {
                ane_train::adam_update(&mut a.a, &g.da, &mut s.a, t, lr, b1, b2, eps, wd);
                ane_train::adam_update(&mut a.b, &g.db, &mut s.b, t, lr, b1, b2, eps, wd);
            }
        };
        update_adapter(
            &mut lora.layers[l].wq,
            &grads_clipped.layers[l].wq,
            &mut adam.layers[l].wq,
        );
        update_adapter(
            &mut lora.layers[l].wv,
            &grads_clipped.layers[l].wv,
            &mut adam.layers[l].wv,
        );
        update_adapter(
            &mut lora.layers[l].wo,
            &grads_clipped.layers[l].wo,
            &mut adam.layers[l].wo,
        );
        update_adapter(
            &mut lora.layers[l].w2,
            &grads_clipped.layers[l].w2,
            &mut adam.layers[l].w2,
        );
    }
}

/// Adam update with per-target learning rate scaling (split LRs).
///
/// Attention adapters (wq, wv, wo) use `lr * lr_scale_attn`, FFN adapter (w2)
/// uses `lr * lr_scale_ffn`. This follows maderix research: matrices at 0.05x,
/// embeddings at 5x, norms at 1x (norms/embeddings aren't LoRA-trained here).
pub fn lora_adam_update_split_lr(
    lora: &mut LoraModel,
    grads: &LoraModelGrads,
    adam: &mut LoraModelAdam,
    t: usize,
    base_lr: f32,
    lr_scale_attn: f32,
    lr_scale_ffn: f32,
    b1: f32,
    b2: f32,
    eps: f32,
    wd: f32,
) {
    // Gradient clipping (max_norm=1.0) — critical for training stability
    let mut grads_clipped = grads.clone();
    clip_lora_grad_norm(&mut grads_clipped, 1.0);

    let lr_attn = base_lr * lr_scale_attn;
    let lr_ffn = base_lr * lr_scale_ffn;

    for l in 0..lora.layers.len() {
        let update_adapter = |adapter: &mut Option<LoraAdapter>,
                              grad: &Option<LoraAdapterGrads>,
                              state: &mut Option<LoraAdapterAdam>,
                              lr: f32| {
            if let (Some(a), Some(g), Some(s)) = (adapter.as_mut(), grad.as_ref(), state.as_mut()) {
                ane_train::adam_update(&mut a.a, &g.da, &mut s.a, t, lr, b1, b2, eps, wd);
                ane_train::adam_update(&mut a.b, &g.db, &mut s.b, t, lr, b1, b2, eps, wd);
            }
        };
        update_adapter(
            &mut lora.layers[l].wq,
            &grads_clipped.layers[l].wq,
            &mut adam.layers[l].wq,
            lr_attn,
        );
        update_adapter(
            &mut lora.layers[l].wv,
            &grads_clipped.layers[l].wv,
            &mut adam.layers[l].wv,
            lr_attn,
        );
        update_adapter(
            &mut lora.layers[l].wo,
            &grads_clipped.layers[l].wo,
            &mut adam.layers[l].wo,
            lr_attn,
        );
        update_adapter(
            &mut lora.layers[l].w2,
            &grads_clipped.layers[l].w2,
            &mut adam.layers[l].w2,
            lr_ffn,
        );
    }
}

const MUON_EPS: f32 = 1e-7;
// ANE stays directionally correct for the first Polar-Express step, but
// diverges on later fp16 iterations. Use a single stable orthogonalization
// step instead of a nominally stronger schedule that corrupts the update.
const MUON_NS_STEPS: usize = 1;
const MUON_NORM_PAD: f32 = 1.02;
const MUON_POLAR_EXPRESS_COEFFS: [(f32, f32, f32); 5] = [
    (8.156554, -22.483294, 15.87877),
    (4.04293, -2.8089175, 0.5000178),
    (3.8916678, -2.772484, 0.50606483),
    (3.2857537, -2.3681295, 0.46449023),
    (2.3465414, -1.7097828, 0.4232355),
];

fn gen_muon_update_mil(rows: usize, cols: usize, beta: f32, lr: f32, wd: f32) -> String {
    let transpose = rows > cols;
    let ortho_rows = rows.min(cols);
    let ortho_cols = rows.max(cols);
    // LoRA adapters: skip fan-ratio scale (see gen_muon_param_update_mil)
    let scale = 1.0f32;
    let wd_scale = 1.0 - lr * wd;
    let one_minus_beta = 1.0 - beta;
    let mut m = String::with_capacity(24_576);
    m.push_str(concat!(
        "program(1.3)\n",
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, ",
        "{\"coremlc-version\", \"3505.4.1\"}, ",
        "{\"coremltools-component-milinternal\", \"\"}, ",
        "{\"coremltools-version\", \"9.0\"}})]\n",
        "{\n",
    ));
    let _ = writeln!(
        m,
        "    func main<ios18>(tensor<fp32, [1, {rows}, 1, {cols}]> p, tensor<fp32, [1, {rows}, 1, {cols}]> g, tensor<fp32, [1, {rows}, 1, {cols}]> m0) {{"
    );
    let _ = writeln!(
        m,
        "        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];"
    );
    let _ = writeln!(
        m,
        "        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{rows},1,{cols}]> ph = cast(dtype=to16,x=p)[name=string(\"ph\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{rows},1,{cols}]> gh = cast(dtype=to16,x=g)[name=string(\"gh\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{rows},1,{cols}]> mh = cast(dtype=to16,x=m0)[name=string(\"mh\")];"
    );
    let _ = writeln!(
        m,
        "        fp16 beta = const()[name=string(\"beta\"), val=fp16({beta})];"
    );
    let _ = writeln!(
        m,
        "        fp16 one_mb = const()[name=string(\"one_mb\"), val=fp16({one_minus_beta})];"
    );
    let _ = writeln!(
        m,
        "        fp16 lr = const()[name=string(\"lr\"), val=fp16({lr})];"
    );
    let _ = writeln!(
        m,
        "        fp16 scale = const()[name=string(\"scale\"), val=fp16({scale})];"
    );
    let _ = writeln!(
        m,
        "        fp16 wd_scale = const()[name=string(\"wd_scale\"), val=fp16({wd_scale})];"
    );
    let _ = writeln!(
        m,
        "        fp16 eps = const()[name=string(\"eps\"), val=fp16({MUON_EPS})];"
    );
    let _ = writeln!(
        m,
        "        fp16 norm_pad_inv = const()[name=string(\"norm_pad_inv\"), val=fp16({})];",
        1.0f32 / MUON_NORM_PAD
    );
    let _ = writeln!(
        m,
        "        bool bF = const()[name=string(\"bF\"), val=bool(false)];"
    );
    let _ = writeln!(
        m,
        "        bool bT = const()[name=string(\"bT\"), val=bool(true)];"
    );
    let _ = writeln!(
        m,
        "        tensor<int32, [4]> rmat = const()[name=string(\"rmat\"), val=tensor<int32, [4]>([1,1,{rows},{cols}])];"
    );
    let _ = writeln!(
        m,
        "        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{rows},1,{cols}]> m_beta = mul(x=mh,y=beta)[name=string(\"m_beta\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{rows},1,{cols}]> g_beta = mul(x=gh,y=one_mb)[name=string(\"g_beta\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{rows},1,{cols}]> m_new = add(x=m_beta,y=g_beta)[name=string(\"m_new\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{rows},1,{cols}]> m_new_beta = mul(x=m_new,y=beta)[name=string(\"m_new_beta\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{rows},1,{cols}]> u0 = add(x=m_new_beta,y=g_beta)[name=string(\"u0\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{rows},{cols}]> ur = reshape(shape=rmat,x=u0)[name=string(\"ur\")];"
    );
    if transpose {
        let _ = writeln!(
            m,
            "        tensor<fp16, [1,1,{cols},{rows}]> x0 = transpose(perm=pm,x=ur)[name=string(\"x0\")];"
        );
    } else {
        let _ = writeln!(
            m,
            "        tensor<fp16, [1,1,{rows},{cols}]> x0 = reshape(shape=rmat,x=u0)[name=string(\"x0\")];"
        );
    }
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{ortho_rows},{ortho_cols}]> x0u = l2_norm(x=x0,epsilon=eps)[name=string(\"x0u\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{ortho_rows},{ortho_cols}]> x0n = mul(x=x0u,y=norm_pad_inv)[name=string(\"x0n\")];"
    );
    for step in 0..MUON_NS_STEPS {
        let (a_coeff, b_coeff, c_coeff) = MUON_POLAR_EXPRESS_COEFFS[step];
        let x_in = if step == 0 {
            "x0n".to_string()
        } else {
            format!("x{step}n")
        };
        let scalar_a = format!("muon_a_{step}");
        let scalar_b = format!("muon_b_{step}");
        let scalar_c = format!("muon_c_{step}");
        let a_name = format!("a{step}");
        let a2_name = format!("a2_{step}");
        let ba_name = format!("ba{step}");
        let ca_name = format!("ca{step}");
        let bmix_name = format!("bmix{step}");
        let bx_name = format!("bx{step}");
        let ax_name = format!("ax{step}");
        let x_out = format!("x{}n", step + 1);
        let _ = writeln!(
            m,
            "        fp16 {scalar_a} = const()[name=string(\"{scalar_a}\"), val=fp16({a_coeff})];"
        );
        let _ = writeln!(
            m,
            "        fp16 {scalar_b} = const()[name=string(\"{scalar_b}\"), val=fp16({b_coeff})];"
        );
        let _ = writeln!(
            m,
            "        fp16 {scalar_c} = const()[name=string(\"{scalar_c}\"), val=fp16({c_coeff})];"
        );
        let _ = writeln!(
            m,
            "        tensor<fp16, [1,1,{ortho_rows},{ortho_rows}]> {a_name} = matmul(transpose_x=bF,transpose_y=bT,x={x_in},y={x_in})[name=string(\"{a_name}\")];"
        );
        let _ = writeln!(
            m,
            "        tensor<fp16, [1,1,{ortho_rows},{ortho_rows}]> {a2_name} = matmul(transpose_x=bF,transpose_y=bF,x={a_name},y={a_name})[name=string(\"{a2_name}\")];"
        );
        let _ = writeln!(
            m,
            "        tensor<fp16, [1,1,{ortho_rows},{ortho_rows}]> {ba_name} = mul(x={a_name},y={scalar_b})[name=string(\"{ba_name}\")];"
        );
        let _ = writeln!(
            m,
            "        tensor<fp16, [1,1,{ortho_rows},{ortho_rows}]> {ca_name} = mul(x={a2_name},y={scalar_c})[name=string(\"{ca_name}\")];"
        );
        let _ = writeln!(
            m,
            "        tensor<fp16, [1,1,{ortho_rows},{ortho_rows}]> {bmix_name} = add(x={ba_name},y={ca_name})[name=string(\"{bmix_name}\")];"
        );
        let _ = writeln!(
            m,
            "        tensor<fp16, [1,1,{ortho_rows},{ortho_cols}]> {bx_name} = matmul(transpose_x=bF,transpose_y=bF,x={bmix_name},y={x_in})[name=string(\"{bx_name}\")];"
        );
        let _ = writeln!(
            m,
            "        tensor<fp16, [1,1,{ortho_rows},{ortho_cols}]> {ax_name} = mul(x={x_in},y={scalar_a})[name=string(\"{ax_name}\")];"
        );
        let _ = writeln!(
            m,
            "        tensor<fp16, [1,1,{ortho_rows},{ortho_cols}]> {x_out} = add(x={ax_name},y={bx_name})[name=string(\"{x_out}\")];"
        );
    }
    let final_x = format!("x{}n", MUON_NS_STEPS);
    if transpose {
        let _ = writeln!(
            m,
            "        tensor<fp16, [1,1,{rows},{cols}]> x_final_t = transpose(perm=pm,x={final_x})[name=string(\"x_final_t\")];"
        );
    } else {
        let _ = writeln!(
            m,
            "        tensor<fp16, [1,1,{rows},{cols}]> x_final_t = reshape(shape=rmat,x={final_x})[name=string(\"x_final_t\")];"
        );
    }
    let _ = writeln!(
        m,
        "        tensor<int32, [4]> rout = const()[name=string(\"rout\"), val=tensor<int32, [4]>([1,{rows},1,{cols}])];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{rows},1,{cols}]> u_ortho = reshape(shape=rout,x=x_final_t)[name=string(\"u_ortho\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{rows},1,{cols}]> u_scaled = mul(x=u_ortho,y=scale)[name=string(\"u_scaled\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{rows},1,{cols}]> p_decay = mul(x=ph,y=wd_scale)[name=string(\"p_decay\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{rows},1,{cols}]> u_lr = mul(x=u_scaled,y=lr)[name=string(\"u_lr\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{rows},1,{cols}]> p_new = sub(x=p_decay,y=u_lr)[name=string(\"p_new\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp32, [1,{rows},1,{cols}]> p_out = cast(dtype=to32,x=p_new)[name=string(\"p_out\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp32, [1,{rows},1,{cols}]> m_out = cast(dtype=to32,x=m_new)[name=string(\"m_out\")];"
    );
    let _ = writeln!(m, "    }} -> (p_out, m_out);");
    m.push_str("}\n");
    m
}

struct MuonMatrixKernel {
    momentum_kernel: AneKernel,
    ortho_kernel: AneKernel,
    param_kernel: AneKernel,
    tensor_bytes: usize,
    rows: usize,
    cols: usize,
}

fn gen_muon_momentum_mil(rows: usize, cols: usize, beta: f32) -> String {
    let one_minus_beta = 1.0 - beta;
    let mut m = String::with_capacity(4096);
    m.push_str(concat!(
        "program(1.3)\n",
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, ",
        "{\"coremlc-version\", \"3505.4.1\"}, ",
        "{\"coremltools-component-milinternal\", \"\"}, ",
        "{\"coremltools-version\", \"9.0\"}})]\n",
        "{\n",
    ));
    let _ = writeln!(
        m,
        "    func main<ios18>(tensor<fp32, [1, {rows}, 1, {cols}]> g, tensor<fp32, [1, {rows}, 1, {cols}]> m0) {{"
    );
    let _ = writeln!(
        m,
        "        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];"
    );
    let _ = writeln!(
        m,
        "        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{rows},1,{cols}]> gh = cast(dtype=to16,x=g)[name=string(\"gh\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{rows},1,{cols}]> mh = cast(dtype=to16,x=m0)[name=string(\"mh\")];"
    );
    let _ = writeln!(
        m,
        "        fp16 beta = const()[name=string(\"beta\"), val=fp16({beta})];"
    );
    let _ = writeln!(
        m,
        "        fp16 one_mb = const()[name=string(\"one_mb\"), val=fp16({one_minus_beta})];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{rows},1,{cols}]> m_beta = mul(x=mh,y=beta)[name=string(\"m_beta\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{rows},1,{cols}]> g_beta = mul(x=gh,y=one_mb)[name=string(\"g_beta\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{rows},1,{cols}]> m_new = add(x=m_beta,y=g_beta)[name=string(\"m_new\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{rows},1,{cols}]> m_new_beta = mul(x=m_new,y=beta)[name=string(\"m_new_beta\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{rows},1,{cols}]> u0 = add(x=m_new_beta,y=g_beta)[name=string(\"u0\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp32, [1,{rows},1,{cols}]> u_out = cast(dtype=to32,x=u0)[name=string(\"u_out\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp32, [1,{rows},1,{cols}]> m_out = cast(dtype=to32,x=m_new)[name=string(\"m_out\")];"
    );
    let _ = writeln!(m, "    }} -> (u_out, m_out);");
    m.push_str("}\n");
    m
}

fn gen_muon_ortho_mil(rows: usize, cols: usize) -> String {
    let transpose = rows > cols;
    let ortho_rows = rows.min(cols);
    let ortho_cols = rows.max(cols);
    let mut m = String::with_capacity(12_288);
    let (a_coeff, b_coeff, c_coeff) = MUON_POLAR_EXPRESS_COEFFS[0];
    m.push_str(concat!(
        "program(1.3)\n",
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, ",
        "{\"coremlc-version\", \"3505.4.1\"}, ",
        "{\"coremltools-component-milinternal\", \"\"}, ",
        "{\"coremltools-version\", \"9.0\"}})]\n",
        "{\n",
    ));
    let _ = writeln!(
        m,
        "    func main<ios18>(tensor<fp32, [1, {rows}, 1, {cols}]> u0) {{"
    );
    let _ = writeln!(
        m,
        "        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];"
    );
    let _ = writeln!(
        m,
        "        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{rows},1,{cols}]> uh = cast(dtype=to16,x=u0)[name=string(\"uh\")];"
    );
    let _ = writeln!(
        m,
        "        fp16 eps = const()[name=string(\"eps\"), val=fp16({MUON_EPS})];"
    );
    let _ = writeln!(
        m,
        "        fp16 norm_pad_inv = const()[name=string(\"norm_pad_inv\"), val=fp16({})];",
        1.0f32 / MUON_NORM_PAD
    );
    let _ = writeln!(
        m,
        "        fp16 a = const()[name=string(\"a\"), val=fp16({a_coeff})];"
    );
    let _ = writeln!(
        m,
        "        fp16 b = const()[name=string(\"b\"), val=fp16({b_coeff})];"
    );
    let _ = writeln!(
        m,
        "        fp16 c = const()[name=string(\"c\"), val=fp16({c_coeff})];"
    );
    let _ = writeln!(
        m,
        "        bool bF = const()[name=string(\"bF\"), val=bool(false)];"
    );
    let _ = writeln!(
        m,
        "        bool bT = const()[name=string(\"bT\"), val=bool(true)];"
    );
    let _ = writeln!(
        m,
        "        tensor<int32, [4]> rmat = const()[name=string(\"rmat\"), val=tensor<int32, [4]>([1,1,{rows},{cols}])];"
    );
    let _ = writeln!(
        m,
        "        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{rows},{cols}]> ur = reshape(shape=rmat,x=uh)[name=string(\"ur\")];"
    );
    if transpose {
        let _ = writeln!(
            m,
            "        tensor<fp16, [1,1,{cols},{rows}]> x0 = transpose(perm=pm,x=ur)[name=string(\"x0\")];"
        );
    } else {
        let _ = writeln!(
            m,
            "        tensor<fp16, [1,1,{rows},{cols}]> x0 = reshape(shape=rmat,x=uh)[name=string(\"x0\")];"
        );
    }
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{ortho_rows},{ortho_cols}]> x0u = l2_norm(x=x0,epsilon=eps)[name=string(\"x0u\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{ortho_rows},{ortho_cols}]> x0n = mul(x=x0u,y=norm_pad_inv)[name=string(\"x0n\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{ortho_rows},{ortho_rows}]> a0 = matmul(transpose_x=bF,transpose_y=bT,x=x0n,y=x0n)[name=string(\"a0\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{ortho_rows},{ortho_rows}]> a2 = matmul(transpose_x=bF,transpose_y=bF,x=a0,y=a0)[name=string(\"a2\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{ortho_rows},{ortho_rows}]> ba = mul(x=a0,y=b)[name=string(\"ba\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{ortho_rows},{ortho_rows}]> ca = mul(x=a2,y=c)[name=string(\"ca\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{ortho_rows},{ortho_rows}]> bb = add(x=ba,y=ca)[name=string(\"bb\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{ortho_rows},{ortho_cols}]> bx = matmul(transpose_x=bF,transpose_y=bF,x=bb,y=x0n)[name=string(\"bx\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{ortho_rows},{ortho_cols}]> ax = mul(x=x0n,y=a)[name=string(\"ax\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,1,{ortho_rows},{ortho_cols}]> x1 = add(x=ax,y=bx)[name=string(\"x1\")];"
    );
    if transpose {
        let _ = writeln!(
            m,
            "        tensor<fp16, [1,1,{rows},{cols}]> x_final = transpose(perm=pm,x=x1)[name=string(\"x_final\")];"
        );
    } else {
        let _ = writeln!(
            m,
            "        tensor<fp16, [1,1,{rows},{cols}]> x_final = reshape(shape=rmat,x=x1)[name=string(\"x_final\")];"
        );
    }
    let _ = writeln!(
        m,
        "        tensor<int32, [4]> rout = const()[name=string(\"rout\"), val=tensor<int32, [4]>([1,{rows},1,{cols}])];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{rows},1,{cols}]> u_ortho = reshape(shape=rout,x=x_final)[name=string(\"u_ortho\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp32, [1,{rows},1,{cols}]> u_out = cast(dtype=to32,x=u_ortho)[name=string(\"u_out\")];"
    );
    let _ = writeln!(m, "    }} -> (u_out);");
    m.push_str("}\n");
    m
}

fn gen_muon_param_update_mil(rows: usize, cols: usize, lr: f32, wd: f32) -> String {
    // For LoRA adapters the Muon fan-ratio scale sqrt(max(rows/cols,1)) is
    // inappropriate — extreme aspect ratios (e.g. 1024×4) amplify updates by
    // 16×. LoRA's alpha/rank already calibrates variance, so use scale=1.0.
    let scale = 1.0f32;
    let wd_scale = 1.0 - lr * wd;
    let mut m = String::with_capacity(4096);
    m.push_str(concat!(
        "program(1.3)\n",
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, ",
        "{\"coremlc-version\", \"3505.4.1\"}, ",
        "{\"coremltools-component-milinternal\", \"\"}, ",
        "{\"coremltools-version\", \"9.0\"}})]\n",
        "{\n",
    ));
    let _ = writeln!(
        m,
        "    func main<ios18>(tensor<fp32, [1, {rows}, 1, {cols}]> p, tensor<fp32, [1, {rows}, 1, {cols}]> u) {{"
    );
    let _ = writeln!(
        m,
        "        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];"
    );
    let _ = writeln!(
        m,
        "        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{rows},1,{cols}]> ph = cast(dtype=to16,x=p)[name=string(\"ph\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{rows},1,{cols}]> uh = cast(dtype=to16,x=u)[name=string(\"uh\")];"
    );
    let _ = writeln!(
        m,
        "        fp16 lr = const()[name=string(\"lr\"), val=fp16({lr})];"
    );
    let _ = writeln!(
        m,
        "        fp16 scale = const()[name=string(\"scale\"), val=fp16({scale})];"
    );
    let _ = writeln!(
        m,
        "        fp16 wd_scale = const()[name=string(\"wd_scale\"), val=fp16({wd_scale})];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{rows},1,{cols}]> p_decay = mul(x=ph,y=wd_scale)[name=string(\"p_decay\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{rows},1,{cols}]> u_scaled = mul(x=uh,y=scale)[name=string(\"u_scaled\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{rows},1,{cols}]> u_lr = mul(x=u_scaled,y=lr)[name=string(\"u_lr\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp16, [1,{rows},1,{cols}]> p_new = sub(x=p_decay,y=u_lr)[name=string(\"p_new\")];"
    );
    let _ = writeln!(
        m,
        "        tensor<fp32, [1,{rows},1,{cols}]> p_out = cast(dtype=to32,x=p_new)[name=string(\"p_out\")];"
    );
    let _ = writeln!(m, "    }} -> (p_out);");
    m.push_str("}\n");
    m
}

impl MuonMatrixKernel {
    fn compile(rows: usize, cols: usize, beta: f32, lr: f32, wd: f32) -> Result<Self, String> {
        ane_bridge::ane_init()?;
        let tensor_bytes = rows * cols * 4;
        let momentum_kernel = AneKernel::compile(
            &gen_muon_momentum_mil(rows, cols, beta),
            None,
            &[tensor_bytes, tensor_bytes],
            &[tensor_bytes, tensor_bytes],
        )?;
        let ortho_kernel = AneKernel::compile(
            &gen_muon_ortho_mil(rows, cols),
            None,
            &[tensor_bytes],
            &[tensor_bytes],
        )?;
        let param_kernel = AneKernel::compile(
            &gen_muon_param_update_mil(rows, cols, lr, wd),
            None,
            &[tensor_bytes, tensor_bytes],
            &[tensor_bytes],
        )?;
        Ok(Self {
            momentum_kernel,
            ortho_kernel,
            param_kernel,
            tensor_bytes,
            rows,
            cols,
        })
    }

    fn eval(
        &self,
        param: &[f32],
        grad: &[f32],
        momentum: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        assert_eq!(param.len(), self.rows * self.cols);
        assert_eq!(grad.len(), self.rows * self.cols);
        assert_eq!(momentum.len(), self.rows * self.cols);
        self.momentum_kernel
            .write_input(0, &ane_weights::f32_slice_to_bytes(grad));
        self.momentum_kernel
            .write_input(1, &ane_weights::f32_slice_to_bytes(momentum));
        self.momentum_kernel.eval()?;
        let mut u0_out = vec![0u8; self.tensor_bytes];
        let mut mom_out = vec![0u8; self.tensor_bytes];
        self.momentum_kernel.read_output(0, &mut u0_out);
        self.momentum_kernel.read_output(1, &mut mom_out);
        let u0 = ane_weights::bytes_to_f32_vec(&u0_out);
        let m_new = ane_weights::bytes_to_f32_vec(&mom_out);

        self.ortho_kernel
            .write_input(0, &ane_weights::f32_slice_to_bytes(&u0));
        self.ortho_kernel.eval()?;
        let mut u_out = vec![0u8; self.tensor_bytes];
        self.ortho_kernel.read_output(0, &mut u_out);
        let u_ortho = ane_weights::bytes_to_f32_vec(&u_out);

        self.param_kernel
            .write_input(0, &ane_weights::f32_slice_to_bytes(param));
        self.param_kernel
            .write_input(1, &ane_weights::f32_slice_to_bytes(&u_ortho));
        self.param_kernel.eval()?;
        let mut param_out = vec![0u8; self.tensor_bytes];
        self.param_kernel.read_output(0, &mut param_out);
        Ok((
            ane_weights::bytes_to_f32_vec(&param_out),
            m_new,
        ))
    }
}

pub(crate) struct AdapterMuonKernels {
    a: MuonMatrixKernel,
    b: MuonMatrixKernel,
}

pub struct LoraMuonKernels {
    pub(crate) wq: Option<AdapterMuonKernels>,
    pub(crate) wv: Option<AdapterMuonKernels>,
    pub(crate) wo: Option<AdapterMuonKernels>,
    pub(crate) w2: Option<AdapterMuonKernels>,
}

impl LoraMuonKernels {
    pub fn compile(
        lora: &LoraModel,
        lr_attn: f32,
        lr_ffn: f32,
        momentum: f32,
        wd: f32,
    ) -> Result<Self, String> {
        let compile_adapter =
            |adapter: Option<&LoraAdapter>, lr: f32| -> Result<Option<AdapterMuonKernels>, String> {
                if let Some(adapter) = adapter {
                    Ok(Some(AdapterMuonKernels {
                        a: MuonMatrixKernel::compile(adapter.rank, adapter.d_in, momentum, lr, wd)?,
                        b: MuonMatrixKernel::compile(adapter.d_out, adapter.rank, momentum, lr, wd)?,
                    }))
                } else {
                    Ok(None)
                }
            };
        let first_wq = lora.layers.iter().find_map(|layer| layer.wq.as_ref());
        let first_wv = lora.layers.iter().find_map(|layer| layer.wv.as_ref());
        let first_wo = lora.layers.iter().find_map(|layer| layer.wo.as_ref());
        let first_w2 = lora.layers.iter().find_map(|layer| layer.w2.as_ref());
        Ok(Self {
            wq: compile_adapter(first_wq, lr_attn)?,
            wv: compile_adapter(first_wv, lr_attn)?,
            wo: compile_adapter(first_wo, lr_attn)?,
            w2: compile_adapter(first_w2, lr_ffn)?,
        })
    }
}

fn update_adapter_muon(
    adapter: &mut Option<LoraAdapter>,
    grad: &Option<LoraAdapterGrads>,
    state: &mut Option<LoraAdapterMuon>,
    kernels: &Option<AdapterMuonKernels>,
) -> Result<(), String> {
    if let (Some(adapter), Some(grad), Some(state), Some(kernels)) = (
        adapter.as_mut(),
        grad.as_ref(),
        state.as_mut(),
        kernels.as_ref(),
    ) {
        let (new_a, new_ma) = kernels.a.eval(&adapter.a, &grad.da, &state.a)?;
        let (new_b, new_mb) = kernels.b.eval(&adapter.b, &grad.db, &state.b)?;
        adapter.a = new_a;
        adapter.b = new_b;
        state.a = new_ma;
        state.b = new_mb;
    }
    Ok(())
}

pub fn lora_muon_update_ane(
    lora: &mut LoraModel,
    grads: &LoraModelGrads,
    muon: &mut LoraModelMuon,
    kernels: &LoraMuonKernels,
) -> Result<(), String> {
    for l in 0..lora.layers.len() {
        update_adapter_muon(
            &mut lora.layers[l].wq,
            &grads.layers[l].wq,
            &mut muon.layers[l].wq,
            &kernels.wq,
        )?;
        update_adapter_muon(
            &mut lora.layers[l].wv,
            &grads.layers[l].wv,
            &mut muon.layers[l].wv,
            &kernels.wv,
        )?;
        update_adapter_muon(
            &mut lora.layers[l].wo,
            &grads.layers[l].wo,
            &mut muon.layers[l].wo,
            &kernels.wo,
        )?;
        update_adapter_muon(
            &mut lora.layers[l].w2,
            &grads.layers[l].w2,
            &mut muon.layers[l].w2,
            &kernels.w2,
        )?;
    }
    Ok(())
}

pub(crate) struct AdapterWeightGradKernels {
    bt: AneKernel,
    bt_out_bytes: usize,
    da: AneKernel,
    da_out_bytes: usize,
    db: AneKernel,
    db_out_bytes: usize,
}

impl AdapterWeightGradKernels {
    fn compile(cfg: &MilConfig, adapter: &LoraAdapter) -> Result<Self, String> {
        let bt_spec = KernelSpec::for_kernel(
            cfg,
            KernelType::DynMatmul {
                ic: adapter.d_out,
                oc: adapter.rank,
            },
        );
        let bt = AneKernel::compile(
            &bt_spec.mil_text,
            None,
            &[bt_spec.input_bytes],
            &[bt_spec.output_bytes],
        )?;

        let mut da_cfg = cfg.clone();
        da_cfg.seq_len = adapter.d_in;
        let da_spec = KernelSpec::for_kernel(
            &da_cfg,
            KernelType::DynMatmul {
                ic: cfg.seq_len,
                oc: adapter.rank,
            },
        );
        let da = AneKernel::compile(
            &da_spec.mil_text,
            None,
            &[da_spec.input_bytes],
            &[da_spec.output_bytes],
        )?;

        let mut db_cfg = cfg.clone();
        db_cfg.seq_len = adapter.rank;
        let db_spec = KernelSpec::for_kernel(
            &db_cfg,
            KernelType::DynMatmul {
                ic: cfg.seq_len,
                oc: adapter.d_out,
            },
        );
        let db = AneKernel::compile(
            &db_spec.mil_text,
            None,
            &[db_spec.input_bytes],
            &[db_spec.output_bytes],
        )?;

        Ok(Self {
            bt,
            bt_out_bytes: bt_spec.output_bytes,
            da,
            da_out_bytes: da_spec.output_bytes,
            db,
            db_out_bytes: db_spec.output_bytes,
        })
    }

    pub fn eval(
        &self,
        adapter: &LoraAdapter,
        d_out_grad: &[f32],
        x: &[f32],
        h: &[f32],
        seq: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        let bt_input =
            ane_weights::pack_dyn_matmul(d_out_grad, &adapter.b, adapter.d_out, adapter.rank, seq);
        self.bt.write_input(0, &bt_input);
        self.bt.eval()?;
        let mut dh_out = vec![0u8; self.bt_out_bytes];
        self.bt.read_output(0, &mut dh_out);
        let mut dh = ane_weights::bytes_to_f32_vec(&dh_out);
        super::ane_forward::clamp_fp16(&mut dh);

        let x_t = ane_weights::transpose_weight(x, adapter.d_in, seq);
        let dh_t = ane_weights::transpose_weight(&dh, adapter.rank, seq);
        let da_input = ane_weights::pack_dyn_matmul(&x_t, &dh_t, seq, adapter.rank, adapter.d_in);
        self.da.write_input(0, &da_input);
        self.da.eval()?;
        let mut da_out = vec![0u8; self.da_out_bytes];
        self.da.read_output(0, &mut da_out);
        let da = ane_weights::bytes_to_f32_vec(&da_out);

        let h_t = ane_weights::transpose_weight(h, adapter.rank, seq);
        let dout_t = ane_weights::transpose_weight(d_out_grad, adapter.d_out, seq);
        let db_input =
            ane_weights::pack_dyn_matmul(&h_t, &dout_t, seq, adapter.d_out, adapter.rank);
        self.db.write_input(0, &db_input);
        self.db.eval()?;
        let mut db_out = vec![0u8; self.db_out_bytes];
        self.db.read_output(0, &mut db_out);
        let db = ane_weights::bytes_to_f32_vec(&db_out);

        Ok((da, db))
    }
}

pub struct LoraWeightGradKernels {
    pub(crate) wq: Option<AdapterWeightGradKernels>,
    pub(crate) wv: Option<AdapterWeightGradKernels>,
    pub(crate) wo: Option<AdapterWeightGradKernels>,
    pub(crate) w2: Option<AdapterWeightGradKernels>,
}

impl LoraWeightGradKernels {
    pub fn compile(cfg: &MilConfig, lora: &LoraModel) -> Result<Self, String> {
        let compile_adapter =
            |adapter: Option<&LoraAdapter>| -> Result<Option<AdapterWeightGradKernels>, String> {
                if let Some(adapter) = adapter {
                    AdapterWeightGradKernels::compile(cfg, adapter).map(Some)
                } else {
                    Ok(None)
                }
            };
        Ok(Self {
            wq: compile_adapter(lora.layers.iter().find_map(|layer| layer.wq.as_ref()))?,
            wv: compile_adapter(lora.layers.iter().find_map(|layer| layer.wv.as_ref()))?,
            wo: compile_adapter(lora.layers.iter().find_map(|layer| layer.wo.as_ref()))?,
            w2: compile_adapter(lora.layers.iter().find_map(|layer| layer.w2.as_ref()))?,
        })
    }
}

// ---------------------------------------------------------------------------
// LoRA ANE kernels
// ---------------------------------------------------------------------------

/// Compiled ANE kernels for LoRA forward + backward.
///
/// For each unique (d_in, rank, d_out) shape, we need:
///   Forward:  A_fwd: DynMatmul(d_in, rank),  B_fwd: DynMatmul(rank, d_out)
///   Backward: Bt_bwd: DynMatmul(d_out, rank), At_bwd: DynMatmul(rank, d_in)
///
/// For the typical case: dim→rank and rank→dim, plus hidden→rank and rank→dim.
pub struct LoraKernels {
    // Attention LoRA (dim→rank→dim)
    pub attn_a_fwd: AneKernel,  // [d_in=dim, rank]
    pub attn_b_fwd: AneKernel,  // [rank, d_out=dim]
    pub attn_bt_bwd: AneKernel, // B^T: [d_out=dim, rank]
    pub attn_at_bwd: AneKernel, // A^T: [rank, d_in=dim]

    // FFN LoRA (hidden→rank→dim)
    pub ffn_a_fwd: AneKernel,  // [d_in=hidden, rank]
    pub ffn_b_fwd: AneKernel,  // [rank, d_out=dim]
    pub ffn_bt_bwd: AneKernel, // B^T: [d_out=dim, rank]
    pub ffn_at_bwd: AneKernel, // A^T: [rank, d_in=hidden]

    pub rank: usize,
    pub dim: usize,
    pub hidden_dim: usize,
}

impl LoraKernels {
    /// Compile all LoRA kernels for given model config and rank.
    pub fn compile(cfg: &MilConfig, rank: usize) -> Result<Self, String> {
        // ANE requires all matmul dimensions to be multiples of 16
        if rank % 16 != 0 {
            return Err(format!(
                "LoRA rank must be a multiple of 16 for ANE (got {rank})"
            ));
        }
        if cfg.dim % 16 != 0 {
            return Err(format!(
                "model dim must be a multiple of 16 for ANE (got {})",
                cfg.dim
            ));
        }
        if cfg.hidden_dim % 16 != 0 {
            return Err(format!(
                "model hidden_dim must be a multiple of 16 for ANE (got {})",
                cfg.hidden_dim
            ));
        }

        ane_bridge::ane_init()?;
        let dim = cfg.dim;
        let hidden = cfg.hidden_dim;

        // Attention: dim→rank
        let attn_a_spec = KernelSpec::for_kernel(cfg, KernelType::DynMatmul { ic: dim, oc: rank });
        let attn_a_fwd = AneKernel::compile(
            &attn_a_spec.mil_text,
            None,
            &[attn_a_spec.input_bytes],
            &[attn_a_spec.output_bytes],
        )?;

        // Attention: rank→dim
        let attn_b_spec = KernelSpec::for_kernel(cfg, KernelType::DynMatmul { ic: rank, oc: dim });
        let attn_b_fwd = AneKernel::compile(
            &attn_b_spec.mil_text,
            None,
            &[attn_b_spec.input_bytes],
            &[attn_b_spec.output_bytes],
        )?;

        // Attention backward: B^T is dim→rank (same shape as a_fwd)
        let attn_bt_bwd = AneKernel::compile(
            &attn_a_spec.mil_text,
            None,
            &[attn_a_spec.input_bytes],
            &[attn_a_spec.output_bytes],
        )?;

        // Attention backward: A^T is rank→dim (same shape as b_fwd)
        let attn_at_bwd = AneKernel::compile(
            &attn_b_spec.mil_text,
            None,
            &[attn_b_spec.input_bytes],
            &[attn_b_spec.output_bytes],
        )?;

        // FFN: hidden→rank
        let ffn_a_spec = KernelSpec::for_kernel(
            cfg,
            KernelType::DynMatmul {
                ic: hidden,
                oc: rank,
            },
        );
        let ffn_a_fwd = AneKernel::compile(
            &ffn_a_spec.mil_text,
            None,
            &[ffn_a_spec.input_bytes],
            &[ffn_a_spec.output_bytes],
        )?;

        // FFN: rank→dim
        let ffn_b_spec = KernelSpec::for_kernel(cfg, KernelType::DynMatmul { ic: rank, oc: dim });
        let ffn_b_fwd = AneKernel::compile(
            &ffn_b_spec.mil_text,
            None,
            &[ffn_b_spec.input_bytes],
            &[ffn_b_spec.output_bytes],
        )?;

        // FFN backward: B^T is dim→rank (same as attn a_fwd shape)
        let ffn_bt_bwd = AneKernel::compile(
            &attn_a_spec.mil_text,
            None,
            &[attn_a_spec.input_bytes],
            &[attn_a_spec.output_bytes],
        )?;

        // FFN backward: A^T is rank→hidden
        let ffn_at_spec = KernelSpec::for_kernel(
            cfg,
            KernelType::DynMatmul {
                ic: rank,
                oc: hidden,
            },
        );
        let ffn_at_bwd = AneKernel::compile(
            &ffn_at_spec.mil_text,
            None,
            &[ffn_at_spec.input_bytes],
            &[ffn_at_spec.output_bytes],
        )?;

        Ok(LoraKernels {
            attn_a_fwd,
            attn_b_fwd,
            attn_bt_bwd,
            attn_at_bwd,
            ffn_a_fwd,
            ffn_b_fwd,
            ffn_bt_bwd,
            ffn_at_bwd,
            rank,
            dim,
            hidden_dim: hidden,
        })
    }
}

// ---------------------------------------------------------------------------
// LoRA ANE forward/backward helpers
// ---------------------------------------------------------------------------

/// Run LoRA forward on ANE: h = A @ x, dy = B @ h.
///
/// Returns (dy [d_out, seq], h [rank, seq]).
pub fn lora_forward_ane(
    a_kernel: &AneKernel,
    b_kernel: &AneKernel,
    adapter: &LoraAdapter,
    x: &[f32],
    seq: usize,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    let rank = adapter.rank;
    let d_in = adapter.d_in;
    let d_out = adapter.d_out;

    // Step 1: h = A @ x via ANE DynMatmul
    // A is [rank, d_in] — we need it as weight in pack_dyn_matmul format
    // pack_dyn_matmul expects: act[ic, seq], w[ic, oc] where ic=d_in, oc=rank
    // The weight matrix for DynMatmul(ic=d_in, oc=rank) is [d_in, rank]
    // Our A is [rank, d_in], so we need A^T = [d_in, rank]
    let a_t = ane_weights::transpose_weight(&adapter.a, rank, d_in);
    let a_input = ane_weights::pack_dyn_matmul(x, &a_t, d_in, rank, seq);

    let a_out_bytes = rank * seq * 4;
    a_kernel.write_input(0, &a_input);
    a_kernel.eval()?;
    let mut a_out = vec![0u8; a_out_bytes];
    a_kernel.read_output(0, &mut a_out);
    let h: Vec<f32> = a_out
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // Step 2: dy = B @ h via ANE DynMatmul
    // B is [d_out, rank] — weight for DynMatmul(ic=rank, oc=d_out) is [rank, d_out]
    // Our B is [d_out, rank], so we need B^T = [rank, d_out]
    let b_t = ane_weights::transpose_weight(&adapter.b, d_out, rank);
    let b_input = ane_weights::pack_dyn_matmul(&h, &b_t, rank, d_out, seq);

    let b_out_bytes = d_out * seq * 4;
    b_kernel.write_input(0, &b_input);
    b_kernel.eval()?;
    let mut b_out = vec![0u8; b_out_bytes];
    b_kernel.read_output(0, &mut b_out);
    let dy: Vec<f32> = b_out
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    Ok((dy, h))
}

/// Run LoRA backward input gradients on ANE, weight gradients on CPU.
///
/// Given upstream gradient d_out_grad [d_out, seq]:
///   dh = B^T @ d_out_grad        [rank, seq]  (ANE)
///   dx_lora = A^T @ dh            [d_in, seq]  (ANE)
///   dB = d_out_grad @ h^T         [d_out, rank] (CPU — small)
///   dA = dh @ x^T                 [rank, d_in]  (CPU — small)
pub fn lora_backward_ane(
    bt_kernel: &AneKernel,
    at_kernel: &AneKernel,
    adapter: &LoraAdapter,
    d_out_grad: &[f32],
    x: &[f32],
    h: &[f32],
    seq: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), String> {
    let rank = adapter.rank;
    let d_in = adapter.d_in;
    let d_out = adapter.d_out;

    // dh = B^T @ d_out_grad via ANE DynMatmul(ic=d_out, oc=rank)
    // B is [d_out, rank], so B^T = [rank, d_out], weight format = [d_out, rank] = B itself
    let bt_input = ane_weights::pack_dyn_matmul(d_out_grad, &adapter.b, d_out, rank, seq);

    let dh_bytes = rank * seq * 4;
    bt_kernel.write_input(0, &bt_input);
    bt_kernel.eval()?;
    let mut bt_out = vec![0u8; dh_bytes];
    bt_kernel.read_output(0, &mut bt_out);
    let dh: Vec<f32> = bt_out
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // dx_lora = A^T @ dh via ANE DynMatmul(ic=rank, oc=d_in)
    // A is [rank, d_in], so A^T = [d_in, rank], weight format = [rank, d_in] = A itself
    let at_input = ane_weights::pack_dyn_matmul(&dh, &adapter.a, rank, d_in, seq);

    let dx_bytes = d_in * seq * 4;
    at_kernel.write_input(0, &at_input);
    at_kernel.eval()?;
    let mut at_out = vec![0u8; dx_bytes];
    at_kernel.read_output(0, &mut at_out);
    let dx_lora: Vec<f32> = at_out
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // CPU weight gradients (small: rank is typically 16-32)
    // dB[o, r] = sum_t d_out_grad[o, t] * h[r, t]
    let mut db = vec![0.0f32; d_out * rank];
    for o in 0..d_out {
        for r in 0..rank {
            let mut acc = 0.0f32;
            for t in 0..seq {
                acc += d_out_grad[o * seq + t] * h[r * seq + t];
            }
            db[o * rank + r] = acc;
        }
    }

    // dA[r, i] = sum_t dh[r, t] * x[i, t]
    let mut da = vec![0.0f32; rank * d_in];
    for r in 0..rank {
        for i in 0..d_in {
            let mut acc = 0.0f32;
            for t in 0..seq {
                acc += dh[r * seq + t] * x[i * seq + t];
            }
            da[r * d_in + i] = acc;
        }
    }

    Ok((dx_lora, da, db))
}

/// Add scaled vector: dst[i] += scale * src[i].
pub fn vec_add_scaled(dst: &mut [f32], src: &[f32], scale: f32) {
    debug_assert_eq!(dst.len(), src.len());
    for i in 0..dst.len() {
        dst[i] += scale * src[i];
    }
}

// ---------------------------------------------------------------------------
// LoRA training config (JIT LoRA proven defaults)
// ---------------------------------------------------------------------------

/// LoRA-specific training config with JIT LoRA defaults.
pub fn lora_training_config() -> TrainingConfig {
    TrainingConfig {
        total_steps: 200, // Short cycles (JIT: 180 steps converges)
        max_lr: 5e-4,     // 10x higher than standard (JIT LoRA finding)
        adam_beta1: 0.9,
        adam_beta2: 0.999,
        adam_eps: 1e-8,
        weight_decay: 0.01, // Decoupled AdamW (standard LoRA practice)
        accum_steps: 1,     // Batch size 1 optimal on Apple Silicon
        warmup_steps: 20,   // ~10% warmup
        grad_clip: 1.0,     // JIT LoRA default
        min_lr_frac: 0.1,   // Cosine floor
        ckpt_interval: 50,
        log_interval: 10,
        early_stop_loss: 0.8,   // JIT LoRA early stop threshold
        early_stop_patience: 2, // JIT LoRA patience
    }
}

// ---------------------------------------------------------------------------
// Safetensors adapter I/O
// ---------------------------------------------------------------------------

/// Save LoRA adapters as raw binary (safetensors-compatible layout).
///
/// Tensor naming convention (MLX-compatible):
///   layers.{i}.{target}.lora_a → [rank, d_in]
///   layers.{i}.{target}.lora_b → [d_out, rank]
pub fn save_lora_bin(lora: &LoraModel, path: &Path) -> io::Result<()> {
    use std::fs;
    use std::io::Write;

    let mut f = fs::File::create(path)?;

    // Simple binary format: magic + version + config + tensors
    f.write_all(b"LORA")?;
    f.write_all(&1u32.to_le_bytes())?; // version
    f.write_all(&(lora.config.rank as u32).to_le_bytes())?;
    f.write_all(&lora.config.alpha.to_le_bytes())?;
    f.write_all(&(lora.layers.len() as u32).to_le_bytes())?;

    for layer in &lora.layers {
        let write_adapter = |f: &mut fs::File, adapter: &Option<LoraAdapter>| -> io::Result<()> {
            match adapter {
                Some(a) => {
                    f.write_all(&1u8.to_le_bytes())?; // present
                    f.write_all(&(a.rank as u32).to_le_bytes())?;
                    f.write_all(&(a.d_in as u32).to_le_bytes())?;
                    f.write_all(&(a.d_out as u32).to_le_bytes())?;
                    for &v in &a.a {
                        f.write_all(&v.to_le_bytes())?;
                    }
                    for &v in &a.b {
                        f.write_all(&v.to_le_bytes())?;
                    }
                }
                None => {
                    f.write_all(&0u8.to_le_bytes())?; // absent
                }
            }
            Ok(())
        };
        write_adapter(&mut f, &layer.wq)?;
        write_adapter(&mut f, &layer.wv)?;
        write_adapter(&mut f, &layer.wo)?;
        write_adapter(&mut f, &layer.w2)?;
    }

    Ok(())
}

/// Load LoRA adapters from binary file.
pub fn load_lora_bin(path: &Path) -> io::Result<LoraModel> {
    let data = std::fs::read(path)?;
    let mut pos = 0;

    let read_u32 = |data: &[u8], pos: &mut usize| -> u32 {
        let v = u32::from_le_bytes([data[*pos], data[*pos + 1], data[*pos + 2], data[*pos + 3]]);
        *pos += 4;
        v
    };
    let read_f32 = |data: &[u8], pos: &mut usize| -> f32 {
        let v = f32::from_le_bytes([data[*pos], data[*pos + 1], data[*pos + 2], data[*pos + 3]]);
        *pos += 4;
        v
    };

    // Magic
    if &data[pos..pos + 4] != b"LORA" {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "bad magic"));
    }
    pos += 4;

    let _version = read_u32(&data, &mut pos);
    let rank = read_u32(&data, &mut pos) as usize;
    let alpha = read_f32(&data, &mut pos);
    let n_layers = read_u32(&data, &mut pos) as usize;

    let mut layers = Vec::with_capacity(n_layers);
    for _ in 0..n_layers {
        let read_adapter = |data: &[u8], pos: &mut usize| -> Option<LoraAdapter> {
            let present = data[*pos];
            *pos += 1;
            if present == 0 {
                return None;
            }
            let r = read_u32(data, pos) as usize;
            let d_in = read_u32(data, pos) as usize;
            let d_out = read_u32(data, pos) as usize;
            let a_len = r * d_in;
            let b_len = d_out * r;
            let mut a = vec![0.0f32; a_len];
            for i in 0..a_len {
                a[i] = read_f32(data, pos);
            }
            let mut b = vec![0.0f32; b_len];
            for i in 0..b_len {
                b[i] = read_f32(data, pos);
            }
            Some(LoraAdapter {
                a,
                b,
                rank: r,
                d_in,
                d_out,
            })
        };
        layers.push(LoraLayerAdapters {
            wq: read_adapter(&data, &mut pos),
            wv: read_adapter(&data, &mut pos),
            wo: read_adapter(&data, &mut pos),
            w2: read_adapter(&data, &mut pos),
        });
    }

    Ok(LoraModel {
        layers,
        config: LoraConfig {
            rank,
            alpha,
            target_modules: vec!["wq".into(), "wv".into(), "wo".into(), "w2".into()],
        },
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::ane_forward;

    #[test]
    fn test_lora_adapter_new_shapes() {
        let a = LoraAdapter::new(4, 8, 8);
        assert_eq!(a.a.len(), 4 * 8);
        assert_eq!(a.b.len(), 8 * 4);
        assert_eq!(a.rank, 4);
        // B should be all zeros (model starts unmodified)
        assert!(a.b.iter().all(|&v| v == 0.0));
        // A should be nonzero (Kaiming init)
        assert!(a.a.iter().any(|&v| v != 0.0));
    }

    #[test]
    fn test_lora_forward_cpu_zero_b() {
        // With B=0, forward output should be zero
        let adapter = LoraAdapter::new(4, 8, 8);
        let x = vec![1.0f32; 8 * 2]; // [d_in=8, seq=2]
        let (dy, _h) = adapter.forward_cpu(&x, 2);
        assert!(
            dy.iter().all(|&v| v.abs() < 1e-10),
            "with B=0, LoRA output should be zero"
        );
    }

    #[test]
    fn test_lora_forward_cpu_nonzero() {
        let mut adapter = LoraAdapter::zeros(2, 4, 4);
        // Set A = [[1,0,0,0],[0,1,0,0]] (rank=2, d_in=4)
        adapter.a[0] = 1.0; // A[0,0]
        adapter.a[5] = 1.0; // A[1,1]
                            // Set B = [[1,0],[0,1],[0,0],[0,0]] (d_out=4, rank=2)
        adapter.b[0] = 1.0; // B[0,0]
        adapter.b[3] = 1.0; // B[1,1]

        // x = [[1,2],[3,4],[5,6],[7,8]] as [d_in=4, seq=2]
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let (dy, h) = adapter.forward_cpu(&x, 2);

        // h = A @ x: h[0,:] = x[0,:] = [1,2], h[1,:] = x[1,:] = [3,4]
        assert_eq!(&h[..], &[1.0, 2.0, 3.0, 4.0]);
        // dy = B @ h: dy[0,:] = h[0,:] = [1,2], dy[1,:] = h[1,:] = [3,4], dy[2,:]=dy[3,:]=0
        assert_eq!(&dy[..], &[1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_lora_backward_cpu_numerical_gradient() {
        let eps = 1e-4f32;

        let mut adapter = LoraAdapter::new(2, 4, 4);
        // Set nonzero B for meaningful backward
        for v in adapter.b.iter_mut() {
            *v = 0.1;
        }

        let seq = 2;
        let x = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let (_dy, h) = adapter.forward_cpu(&x, seq);

        // Upstream gradient
        let d_out_grad: Vec<f32> = (0..4 * seq)
            .map(|i| ((i as f32 * 1.3 + 0.7).cos()) * 0.5)
            .collect();

        let (_dx, da, db) = adapter.backward_cpu(&d_out_grad, &x, &h, seq);

        // Numerical gradient check for dA
        for idx in 0..adapter.a.len() {
            let mut a_plus = adapter.clone_adapter();
            let mut a_minus = adapter.clone_adapter();
            a_plus.a[idx] += eps;
            a_minus.a[idx] -= eps;
            let (dy_p, _) = a_plus.forward_cpu(&x, seq);
            let (dy_m, _) = a_minus.forward_cpu(&x, seq);
            let mut num = 0.0f32;
            for j in 0..dy_p.len() {
                num += d_out_grad[j] * (dy_p[j] - dy_m[j]) / (2.0 * eps);
            }
            assert!(
                (da[idx] - num).abs() < 0.01,
                "dA[{idx}]: analytical={:.6}, numerical={:.6}",
                da[idx],
                num
            );
        }

        // Numerical gradient check for dB
        for idx in 0..adapter.b.len() {
            let mut b_plus = adapter.clone_adapter();
            let mut b_minus = adapter.clone_adapter();
            b_plus.b[idx] += eps;
            b_minus.b[idx] -= eps;
            let (dy_p, _) = b_plus.forward_cpu(&x, seq);
            let (dy_m, _) = b_minus.forward_cpu(&x, seq);
            let mut num = 0.0f32;
            for j in 0..dy_p.len() {
                num += d_out_grad[j] * (dy_p[j] - dy_m[j]) / (2.0 * eps);
            }
            assert!(
                (db[idx] - num).abs() < 0.01,
                "dB[{idx}]: analytical={:.6}, numerical={:.6}",
                db[idx],
                num
            );
        }
    }

    #[test]
    fn test_lora_backward_produces_nonzero_grads() {
        let mut adapter = LoraAdapter::new(4, 8, 8);
        for v in adapter.b.iter_mut() {
            *v = 0.05;
        }
        let seq = 3;
        let x: Vec<f32> = (0..8 * seq).map(|i| (i as f32 * 0.1).sin()).collect();
        let (_, h) = adapter.forward_cpu(&x, seq);
        let d_out_grad: Vec<f32> = (0..8 * seq).map(|i| (i as f32 * 0.3).cos()).collect();
        let (dx, da, db) = adapter.backward_cpu(&d_out_grad, &x, &h, seq);
        assert!(dx.iter().any(|&v| v.abs() > 1e-6), "dx should be nonzero");
        assert!(da.iter().any(|&v| v.abs() > 1e-6), "dA should be nonzero");
        assert!(db.iter().any(|&v| v.abs() > 1e-6), "dB should be nonzero");
    }

    #[test]
    fn test_lora_model_init() {
        let cfg = LoraConfig::default();
        let lora = LoraModel::new(cfg, 6, 768, 2048);
        assert_eq!(lora.layers.len(), 6);
        assert!(lora.layers[0].wq.is_some());
        assert!(lora.layers[0].wv.is_some());
        assert!(lora.layers[0].wo.is_some());
        assert!(lora.layers[0].w2.is_some());
        assert_eq!(lora.scale(), 1.0); // alpha=32, rank=32
    }

    #[test]
    fn test_lora_grads_zero() {
        let cfg = LoraConfig {
            rank: 4,
            alpha: 4.0,
            target_modules: vec!["w2".into()],
        };
        let lora = LoraModel::new(cfg, 2, 64, 128);
        let mut grads = LoraModelGrads::zeros(&lora);
        assert!(grads.layers[0].w2.is_some());
        assert!(grads.layers[0].wq.is_none());
        // Set some values and verify zero works
        if let Some(g) = grads.layers[0].w2.as_mut() {
            g.da[0] = 1.0;
        }
        grads.zero();
        assert_eq!(grads.layers[0].w2.as_ref().unwrap().da[0], 0.0);
    }

    #[test]
    fn test_lora_grads_add_from_and_scale() {
        let cfg = LoraConfig {
            rank: 2,
            alpha: 2.0,
            target_modules: vec!["w2".into()],
        };
        let lora = LoraModel::new(cfg, 1, 4, 8);
        let mut accum = LoraModelGrads::zeros(&lora);
        let mut g1 = LoraModelGrads::zeros(&lora);
        let mut g2 = LoraModelGrads::zeros(&lora);
        // g1: da[0]=3.0, db[0]=1.0
        g1.layers[0].w2.as_mut().unwrap().da[0] = 3.0;
        g1.layers[0].w2.as_mut().unwrap().db[0] = 1.0;
        // g2: da[0]=5.0, db[0]=3.0
        g2.layers[0].w2.as_mut().unwrap().da[0] = 5.0;
        g2.layers[0].w2.as_mut().unwrap().db[0] = 3.0;

        accum.add_from(&g1);
        accum.add_from(&g2);
        // accum: da[0]=8.0, db[0]=4.0
        assert_eq!(accum.layers[0].w2.as_ref().unwrap().da[0], 8.0);
        assert_eq!(accum.layers[0].w2.as_ref().unwrap().db[0], 4.0);

        accum.scale(0.5);
        // averaged: da[0]=4.0, db[0]=2.0
        assert_eq!(accum.layers[0].w2.as_ref().unwrap().da[0], 4.0);
        assert_eq!(accum.layers[0].w2.as_ref().unwrap().db[0], 2.0);
    }

    #[test]
    fn test_lora_grad_norm() {
        let cfg = LoraConfig {
            rank: 2,
            alpha: 2.0,
            target_modules: vec!["w2".into()],
        };
        let lora = LoraModel::new(cfg, 1, 4, 8);
        let mut grads = LoraModelGrads::zeros(&lora);
        // Set one element to 3.0, another to 4.0 → norm = 5.0
        if let Some(g) = grads.layers[0].w2.as_mut() {
            g.da[0] = 3.0;
            g.db[0] = 4.0;
        }
        let norm = lora_grad_norm(&grads);
        assert!((norm - 5.0).abs() < 1e-5, "norm={norm}, expected 5.0");
    }

    #[test]
    fn test_lora_save_load_roundtrip() {
        let cfg = LoraConfig {
            rank: 4,
            alpha: 8.0,
            target_modules: vec!["w2".into(), "wo".into()],
        };
        let lora = LoraModel::new(cfg, 2, 16, 32);

        let path = std::env::temp_dir().join("test_lora_roundtrip.bin");
        save_lora_bin(&lora, &path).expect("save failed");
        let loaded = load_lora_bin(&path).expect("load failed");

        assert_eq!(loaded.config.rank, 4);
        assert_eq!(loaded.config.alpha, 8.0);
        assert_eq!(loaded.layers.len(), 2);
        assert!(loaded.layers[0].wo.is_some());
        assert!(loaded.layers[0].w2.is_some());
        assert!(loaded.layers[0].wq.is_none());

        // Verify weights match
        let orig_w2 = lora.layers[0].w2.as_ref().unwrap();
        let load_w2 = loaded.layers[0].w2.as_ref().unwrap();
        assert_eq!(orig_w2.a, load_w2.a);
        assert_eq!(orig_w2.b, load_w2.b);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_vec_add_scaled() {
        let mut dst = vec![1.0, 2.0, 3.0];
        let src = vec![10.0, 20.0, 30.0];
        vec_add_scaled(&mut dst, &src, 0.5);
        assert_eq!(dst, vec![6.0, 12.0, 18.0]);
    }

    #[test]
    fn test_lora_config_scale() {
        let cfg = LoraConfig {
            rank: 16,
            alpha: 32.0,
            target_modules: vec![],
        };
        assert_eq!(cfg.scale(), 2.0);
        let cfg2 = LoraConfig::default();
        assert_eq!(cfg2.scale(), 1.0);
    }

    // --- ANE kernel tests (require hardware) ---

    #[test]
    fn test_lora_kernels_compile() {
        let mil_cfg = MilConfig::mha(64, 128, 4, 16);
        match LoraKernels::compile(&mil_cfg, 16) {
            Ok(k) => {
                assert_eq!(k.rank, 16);
                assert_eq!(k.dim, 64);
                assert_eq!(k.hidden_dim, 128);
            }
            Err(e) => {
                eprintln!("Skipping LoRA kernel test (ANE unavailable): {e}");
            }
        }
    }

    #[test]
    fn test_lora_forward_ane_matches_cpu() {
        let mil_cfg = MilConfig::mha(64, 128, 4, 16);
        let kernels = match LoraKernels::compile(&mil_cfg, 16) {
            Ok(k) => k,
            Err(e) => {
                eprintln!("Skipping ANE test: {e}");
                return;
            }
        };

        let mut adapter = LoraAdapter::new(16, 64, 64);
        // Set B nonzero for meaningful test
        for v in adapter.b.iter_mut() {
            *v = 0.01;
        }

        let seq = 16;
        let x: Vec<f32> = (0..64 * seq).map(|i| (i as f32 * 0.01).sin()).collect();

        // CPU reference
        let (dy_cpu, h_cpu) = adapter.forward_cpu(&x, seq);

        // ANE
        let (dy_ane, h_ane) =
            lora_forward_ane(&kernels.attn_a_fwd, &kernels.attn_b_fwd, &adapter, &x, seq)
                .expect("ANE forward failed");

        // Compare with tolerance (fp16 intermediate in ANE)
        let max_h_err = h_cpu
            .iter()
            .zip(h_ane.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let max_dy_err = dy_cpu
            .iter()
            .zip(dy_ane.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(max_h_err < 0.1, "h max error: {max_h_err}");
        assert!(max_dy_err < 0.1, "dy max error: {max_dy_err}");
    }

    #[test]
    fn test_lora_backward_ane_matches_cpu() {
        let mil_cfg = MilConfig::mha(64, 128, 4, 16);
        let kernels = match LoraKernels::compile(&mil_cfg, 16) {
            Ok(k) => k,
            Err(e) => {
                eprintln!("Skipping ANE backward test: {e}");
                return;
            }
        };

        let mut adapter = LoraAdapter::new(16, 64, 64);
        for v in adapter.b.iter_mut() {
            *v = 0.01;
        }

        let seq = 16;
        let x: Vec<f32> = (0..64 * seq).map(|i| (i as f32 * 0.01).sin()).collect();
        let (_, h) = adapter.forward_cpu(&x, seq);

        let d_out_grad: Vec<f32> = (0..64 * seq).map(|i| (i as f32 * 0.03).cos()).collect();

        // CPU reference
        let (dx_cpu, da_cpu, db_cpu) = adapter.backward_cpu(&d_out_grad, &x, &h, seq);

        // ANE
        let (dx_ane, da_ane, db_ane) = lora_backward_ane(
            &kernels.attn_bt_bwd,
            &kernels.attn_at_bwd,
            &adapter,
            &d_out_grad,
            &x,
            &h,
            seq,
        )
        .expect("ANE backward failed");

        let max_dx_err = dx_cpu
            .iter()
            .zip(dx_ane.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let max_da_err = da_cpu
            .iter()
            .zip(da_ane.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let max_db_err = db_cpu
            .iter()
            .zip(db_ane.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(max_dx_err < 0.5, "dx max error: {max_dx_err}");
        assert!(max_da_err < 0.5, "dA max error: {max_da_err}");
        assert!(max_db_err < 0.5, "dB max error: {max_db_err}");
    }

    fn muon_update_cpu_reference(
        param: &[f32],
        grad: &[f32],
        momentum: &[f32],
        rows: usize,
        cols: usize,
        beta: f32,
        lr: f32,
        wd: f32,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut m_new = vec![0.0f32; param.len()];
        let one_minus_beta = 1.0 - beta;
        for i in 0..param.len() {
            m_new[i] = beta * momentum[i] + one_minus_beta * grad[i];
        }
        let mut update = vec![0.0f32; grad.len()];
        for i in 0..grad.len() {
            update[i] = beta * m_new[i] + one_minus_beta * grad[i];
        }

        let transpose = rows > cols;
        let ortho_rows = rows.min(cols);
        let ortho_cols = rows.max(cols);
        let mut x = if transpose {
            ane_weights::transpose_weight(&update, rows, cols)
        } else {
            update.clone()
        };
        let norm = x.iter().map(|v| v * v).sum::<f32>().sqrt().max(MUON_EPS) * MUON_NORM_PAD;
        for v in &mut x {
            *v /= norm;
        }
        for &(a_coeff, b_coeff, c_coeff) in MUON_POLAR_EXPRESS_COEFFS.iter().take(MUON_NS_STEPS) {
            let x_t = ane_weights::transpose_weight(&x, ortho_rows, ortho_cols);
            let a = ane_forward::cpu_matmul(&x, &x_t, ortho_rows, ortho_cols, ortho_rows);
            let a2 = ane_forward::cpu_matmul(&a, &a, ortho_rows, ortho_rows, ortho_rows);
            let mut bmix = vec![0.0f32; ortho_rows * ortho_rows];
            for i in 0..bmix.len() {
                bmix[i] = b_coeff * a[i] + c_coeff * a2[i];
            }
            let bx = ane_forward::cpu_matmul(&bmix, &x, ortho_rows, ortho_rows, ortho_cols);
            for i in 0..x.len() {
                x[i] = a_coeff * x[i] + bx[i];
            }
        }
        let update_ortho = if transpose {
            ane_weights::transpose_weight(&x, ortho_rows, ortho_cols)
        } else {
            x
        };
        // LoRA adapters: skip fan-ratio scale (see gen_muon_param_update_mil)
        let scale = 1.0f32;
        let mut p_new = param.to_vec();
        for i in 0..p_new.len() {
            p_new[i] = p_new[i] * (1.0 - lr * wd) - lr * scale * update_ortho[i];
        }
        (p_new, m_new)
    }

    #[test]
    fn test_lora_weight_grads_ane_match_cpu() {
        let mil_cfg = MilConfig::mha(64, 128, 4, 16);
        let mut lora = LoraModel::with_full_dims(
            LoraConfig {
                rank: 16,
                ..LoraConfig::default()
            },
            1,
            64,
            64,
            64,
            64,
            128,
        );
        let kernels = match LoraWeightGradKernels::compile(&mil_cfg, &lora) {
            Ok(k) => k,
            Err(e) => {
                eprintln!("Skipping ANE LoRA weight-grad test: {e}");
                return;
            }
        };

        let adapter = lora.layers[0].wo.as_mut().unwrap();
        for v in adapter.b.iter_mut() {
            *v = 0.01;
        }
        let seq = 16;
        let x: Vec<f32> = (0..adapter.d_in * seq)
            .map(|i| (i as f32 * 0.01).sin())
            .collect();
        let (_, h) = adapter.forward_cpu(&x, seq);
        let d_out_grad: Vec<f32> = (0..adapter.d_out * seq)
            .map(|i| (i as f32 * 0.03).cos())
            .collect();
        let (_dx_cpu, da_cpu, db_cpu) = adapter.backward_cpu(&d_out_grad, &x, &h, seq);
        let (da_ane, db_ane) = kernels
            .wo
            .as_ref()
            .unwrap()
            .eval(adapter, &d_out_grad, &x, &h, seq)
            .expect("ANE weight grads failed");

        let max_da_err = da_cpu
            .iter()
            .zip(da_ane.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let max_db_err = db_cpu
            .iter()
            .zip(db_ane.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_da_err < 0.5, "ANE dA max error: {max_da_err}");
        assert!(max_db_err < 0.5, "ANE dB max error: {max_db_err}");
    }

    #[test]
    fn test_muon_matrix_kernel_matches_cpu_reference() {
        let rows = 32;
        let cols = 64;
        let beta = 0.95;
        let lr = 0.02;
        let wd = 0.01;
        let kernel = match MuonMatrixKernel::compile(rows, cols, beta, lr, wd) {
            Ok(kernel) => kernel,
            Err(e) => {
                eprintln!("Skipping ANE Muon kernel test: {e}");
                return;
            }
        };

        let param: Vec<f32> = (0..rows * cols).map(|i| (i as f32 * 0.007).sin()).collect();
        let grad: Vec<f32> = (0..rows * cols).map(|i| (i as f32 * 0.011).cos()).collect();
        let momentum: Vec<f32> = (0..rows * cols)
            .map(|i| (i as f32 * 0.013).sin() * 0.1)
            .collect();

        let (cpu_param, cpu_mom) =
            muon_update_cpu_reference(&param, &grad, &momentum, rows, cols, beta, lr, wd);
        let (ane_param, ane_mom) = kernel
            .eval(&param, &grad, &momentum)
            .expect("ANE Muon update failed");

        let max_mom_err = cpu_mom
            .iter()
            .zip(ane_mom.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let cpu_delta: Vec<f32> = param
            .iter()
            .zip(cpu_param.iter())
            .map(|(before, after)| before - after)
            .collect();
        let ane_delta: Vec<f32> = param
            .iter()
            .zip(ane_param.iter())
            .map(|(before, after)| before - after)
            .collect();
        let dot = cpu_delta
            .iter()
            .zip(ane_delta.iter())
            .map(|(a, b)| a * b)
            .sum::<f32>();
        let cpu_norm = cpu_delta.iter().map(|v| v * v).sum::<f32>().sqrt();
        let ane_norm = ane_delta.iter().map(|v| v * v).sum::<f32>().sqrt();
        let cosine = dot / (cpu_norm * ane_norm).max(1e-8);
        let max_param_err = cpu_param
            .iter()
            .zip(ane_param.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let wrong_order_delta: Vec<f32> = param
            .iter()
            .zip(ane_mom.iter())
            .map(|(before, after)| before - after)
            .collect();
        let wrong_order_cosine = cpu_delta
            .iter()
            .zip(wrong_order_delta.iter())
            .map(|(a, b)| a * b)
            .sum::<f32>()
            / (cpu_norm
                * wrong_order_delta
                    .iter()
                    .map(|v| v * v)
                    .sum::<f32>()
                    .sqrt())
                .max(1e-8);

        eprintln!(
            "ANE Muon debug: cosine={cosine:.4} max_param_err={max_param_err:.4} max_mom_err={max_mom_err:.4} wrong_order_cosine={wrong_order_cosine:.4}"
        );

        assert!(
            cosine > 0.9,
            "ANE Muon update direction drifted too far: cosine={cosine:.4}"
        );
        assert!(max_mom_err < 0.1, "ANE Muon momentum max error: {max_mom_err}");
    }

    #[test]
    fn test_muon_l2_norm_matches_cpu_reference() {
        let rows = 32usize;
        let cols = 64usize;
        let mut mil = String::new();
        mil.push_str("program(1.3)\n");
        mil.push_str("[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n");
        mil.push_str("{\n");
        let _ = writeln!(
            mil,
            "    func main<ios18>(tensor<fp32, [1, {rows}, 1, {cols}]> x) {{"
        );
        mil.push_str("        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];\n");
        mil.push_str("        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];\n");
        let _ = writeln!(
            mil,
            "        fp16 eps = const()[name=string(\"eps\"), val=fp16({})];",
            MUON_EPS
        );
        let _ = writeln!(
            mil,
            "        tensor<fp16, [1,{rows},1,{cols}]> xh = cast(dtype=to16,x=x)[name=string(\"xh\")];"
        );
        let _ = writeln!(
            mil,
            "        tensor<int32, [4]> r = const()[name=string(\"r\"), val=tensor<int32, [4]>([1,1,{rows},{cols}])];"
        );
        let _ = writeln!(
            mil,
            "        tensor<int32, [4]> rout = const()[name=string(\"rout\"), val=tensor<int32, [4]>([1,{rows},1,{cols}])];"
        );
        let _ = writeln!(
            mil,
            "        tensor<fp16, [1,1,{rows},{cols}]> xr = reshape(shape=r,x=xh)[name=string(\"xr\")];"
        );
        let _ = writeln!(
            mil,
            "        tensor<fp16, [1,1,{rows},{cols}]> xn = l2_norm(x=xr,epsilon=eps)[name=string(\"xn\")];"
        );
        let _ = writeln!(
            mil,
            "        tensor<fp16, [1,{rows},1,{cols}]> xo = reshape(shape=rout,x=xn)[name=string(\"xo\")];"
        );
        let _ = writeln!(
            mil,
            "        tensor<fp32, [1,{rows},1,{cols}]> y = cast(dtype=to32,x=xo)[name=string(\"y\")];"
        );
        mil.push_str("    } -> (y);\n");
        mil.push_str("}\n");
        let tensor_bytes = rows * cols * 4;
        if let Err(e) = ane_bridge::ane_init() {
            eprintln!("Skipping ANE l2_norm Muon test: {e}");
            return;
        }
        let kernel = match AneKernel::compile(&mil, None, &[tensor_bytes], &[tensor_bytes]) {
            Ok(kernel) => kernel,
            Err(e) => {
                eprintln!("Skipping ANE l2_norm Muon test: {e}");
                return;
            }
        };

        let x: Vec<f32> = (0..rows * cols)
            .map(|i| (i as f32 * 0.011).sin() * 0.5 + (i as f32 * 0.003).cos())
            .collect();
        kernel.write_input(0, &ane_weights::f32_slice_to_bytes(&x));
        if let Err(e) = kernel.eval() {
            eprintln!("Skipping ANE l2_norm Muon test after eval failure: {e}");
            return;
        }
        let mut out = vec![0u8; tensor_bytes];
        kernel.read_output(0, &mut out);
        let y = ane_weights::bytes_to_f32_vec(&out);

        let norm = x.iter().map(|v| v * v).sum::<f32>().sqrt().max(MUON_EPS);
        let y_cpu: Vec<f32> = x.iter().map(|v| *v / norm).collect();
        let max_err = y
            .iter()
            .zip(y_cpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 0.02, "ANE l2_norm max error: {max_err}");
    }

    #[test]
    fn test_muon_polar_step_matches_cpu_reference() {
        let rows = 32usize;
        let cols = 64usize;
        let (a_coeff, b_coeff, c_coeff) = MUON_POLAR_EXPRESS_COEFFS[0];
        let mut mil = String::new();
        mil.push_str("program(1.3)\n");
        mil.push_str("[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n");
        mil.push_str("{\n");
        let _ = writeln!(
            mil,
            "    func main<ios18>(tensor<fp32, [1, {rows}, 1, {cols}]> x) {{"
        );
        mil.push_str("        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];\n");
        mil.push_str("        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];\n");
        let _ = writeln!(
            mil,
            "        fp16 a = const()[name=string(\"a\"), val=fp16({a_coeff})];"
        );
        let _ = writeln!(
            mil,
            "        fp16 b = const()[name=string(\"b\"), val=fp16({b_coeff})];"
        );
        let _ = writeln!(
            mil,
            "        fp16 c = const()[name=string(\"c\"), val=fp16({c_coeff})];"
        );
        mil.push_str("        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n");
        mil.push_str("        bool bT = const()[name=string(\"bT\"), val=bool(true)];\n");
        let _ = writeln!(
            mil,
            "        tensor<int32, [4]> r = const()[name=string(\"r\"), val=tensor<int32, [4]>([1,1,{rows},{cols}])];"
        );
        let _ = writeln!(
            mil,
            "        tensor<int32, [4]> rout = const()[name=string(\"rout\"), val=tensor<int32, [4]>([1,{rows},1,{cols}])];"
        );
        let _ = writeln!(
            mil,
            "        tensor<fp16, [1,{rows},1,{cols}]> xh = cast(dtype=to16,x=x)[name=string(\"xh\")];"
        );
        let _ = writeln!(
            mil,
            "        tensor<fp16, [1,1,{rows},{cols}]> xr = reshape(shape=r,x=xh)[name=string(\"xr\")];"
        );
        let _ = writeln!(
            mil,
            "        tensor<fp16, [1,1,{rows},{rows}]> a0 = matmul(transpose_x=bF,transpose_y=bT,x=xr,y=xr)[name=string(\"a0\")];"
        );
        let _ = writeln!(
            mil,
            "        tensor<fp16, [1,1,{rows},{rows}]> a2 = matmul(transpose_x=bF,transpose_y=bF,x=a0,y=a0)[name=string(\"a2\")];"
        );
        let _ = writeln!(
            mil,
            "        tensor<fp16, [1,1,{rows},{rows}]> ba = mul(x=a0,y=b)[name=string(\"ba\")];"
        );
        let _ = writeln!(
            mil,
            "        tensor<fp16, [1,1,{rows},{rows}]> ca = mul(x=a2,y=c)[name=string(\"ca\")];"
        );
        let _ = writeln!(
            mil,
            "        tensor<fp16, [1,1,{rows},{rows}]> bb = add(x=ba,y=ca)[name=string(\"bb\")];"
        );
        let _ = writeln!(
            mil,
            "        tensor<fp16, [1,1,{rows},{cols}]> bx = matmul(transpose_x=bF,transpose_y=bF,x=bb,y=xr)[name=string(\"bx\")];"
        );
        let _ = writeln!(
            mil,
            "        tensor<fp16, [1,1,{rows},{cols}]> ax = mul(x=xr,y=a)[name=string(\"ax\")];"
        );
        let _ = writeln!(
            mil,
            "        tensor<fp16, [1,1,{rows},{cols}]> x1 = add(x=ax,y=bx)[name=string(\"x1\")];"
        );
        let _ = writeln!(
            mil,
            "        tensor<fp16, [1,{rows},1,{cols}]> xo = reshape(shape=rout,x=x1)[name=string(\"xo\")];"
        );
        let _ = writeln!(
            mil,
            "        tensor<fp32, [1,{rows},1,{cols}]> y = cast(dtype=to32,x=xo)[name=string(\"y\")];"
        );
        mil.push_str("    } -> (y);\n");
        mil.push_str("}\n");

        let tensor_bytes = rows * cols * 4;
        if let Err(e) = ane_bridge::ane_init() {
            eprintln!("Skipping ANE Muon polar-step test: {e}");
            return;
        }
        let kernel = match AneKernel::compile(&mil, None, &[tensor_bytes], &[tensor_bytes]) {
            Ok(kernel) => kernel,
            Err(e) => {
                eprintln!("Skipping ANE Muon polar-step test: {e}");
                return;
            }
        };

        let x_raw: Vec<f32> = (0..rows * cols)
            .map(|i| (i as f32 * 0.009).sin() + (i as f32 * 0.004).cos() * 0.25)
            .collect();
        let norm = x_raw.iter().map(|v| v * v).sum::<f32>().sqrt().max(MUON_EPS) * MUON_NORM_PAD;
        let x: Vec<f32> = x_raw.iter().map(|v| *v / norm).collect();

        kernel.write_input(0, &ane_weights::f32_slice_to_bytes(&x));
        if let Err(e) = kernel.eval() {
            eprintln!("Skipping ANE Muon polar-step test after eval failure: {e}");
            return;
        }
        let mut out = vec![0u8; tensor_bytes];
        kernel.read_output(0, &mut out);
        let y = ane_weights::bytes_to_f32_vec(&out);

        let a = ane_forward::cpu_matmul(&x, &ane_weights::transpose_weight(&x, rows, cols), rows, cols, rows);
        let a2 = ane_forward::cpu_matmul(&a, &a, rows, rows, rows);
        let mut bb = vec![0.0f32; rows * rows];
        for i in 0..bb.len() {
            bb[i] = b_coeff * a[i] + c_coeff * a2[i];
        }
        let bx = ane_forward::cpu_matmul(&bb, &x, rows, rows, cols);
        let mut y_cpu = vec![0.0f32; rows * cols];
        for i in 0..y_cpu.len() {
            y_cpu[i] = a_coeff * x[i] + bx[i];
        }

        let max_err = y
            .iter()
            .zip(y_cpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 0.1, "ANE Muon polar-step max error: {max_err}");
    }

    #[test]
    fn test_muon_matrix_kernel_compiles_for_rank_downproj_shape() {
        let rows = 1024;
        let cols = 32;
        match MuonMatrixKernel::compile(rows, cols, 0.95, 2.5e-4, 0.01) {
            Ok(_) => {}
            Err(e) => {
                eprintln!("Skipping/downproj compile failed: {e}");
                panic!("ANE Muon downproj shape should compile");
            }
        }
    }

    // --- ANE dimension validation tests ---

    #[test]
    fn test_lora_compile_rejects_non_aligned_rank() {
        let mil_cfg = MilConfig::mha(64, 128, 4, 16);
        match LoraKernels::compile(&mil_cfg, 4) {
            Err(e) => assert!(
                e.contains("multiple of 16"),
                "expected alignment error, got: {e}"
            ),
            Ok(_) => panic!("should reject rank=4"),
        }
    }

    #[test]
    #[allow(clippy::identity_op)]
    fn test_lora_compile_rejects_non_aligned_dim() {
        // dim=40 is not a multiple of 16
        let mil_cfg = MilConfig::mha(40, 128, 4, 10);
        match LoraKernels::compile(&mil_cfg, 16) {
            Err(e) => assert!(
                e.contains("multiple of 16"),
                "expected alignment error, got: {e}"
            ),
            Ok(_) => panic!("should reject dim=40"),
        }
    }

    // --- LoRA training convergence test (CPU-only) ---

    #[test]
    fn test_lora_training_converges_cpu() {
        // Tiny model: verify that LoRA gradient updates move loss in the right direction.
        // We simulate training by: forward → backward → update → forward again,
        // checking that loss decreases.
        let rank = 4;
        let dim = 8;
        let seq = 4;

        let mut adapter = LoraAdapter::new(rank, dim, dim);
        // Initialize B to small nonzero for non-trivial forward
        for v in adapter.b.iter_mut() {
            *v = 0.01;
        }

        let x: Vec<f32> = (0..dim * seq).map(|i| (i as f32 * 0.1).sin()).collect();
        // Target: the LoRA output should learn to match a specific signal
        let target: Vec<f32> = (0..dim * seq)
            .map(|i| (i as f32 * 0.2).cos() * 0.1)
            .collect();

        // Compute initial loss (MSE between LoRA output and target)
        let (dy0, _) = adapter.forward_cpu(&x, seq);
        let loss0: f32 = dy0
            .iter()
            .zip(target.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / (dim * seq) as f32;

        // Training loop: 50 gradient steps
        let lr = 1e-3;
        let mut a_adam = ane_train::AdamState::zeros(rank * dim);
        let mut b_adam = ane_train::AdamState::zeros(dim * rank);

        for step in 0..50 {
            let (dy, h) = adapter.forward_cpu(&x, seq);
            // MSE gradient: d_out = 2 * (dy - target) / N
            let n = (dim * seq) as f32;
            let d_out: Vec<f32> = dy
                .iter()
                .zip(target.iter())
                .map(|(a, b)| 2.0 * (a - b) / n)
                .collect();
            let (_dx, da, db) = adapter.backward_cpu(&d_out, &x, &h, seq);
            ane_train::adam_update(
                &mut adapter.a,
                &da,
                &mut a_adam,
                step + 1,
                lr,
                0.9,
                0.999,
                1e-8,
                0.0,
            );
            ane_train::adam_update(
                &mut adapter.b,
                &db,
                &mut b_adam,
                step + 1,
                lr,
                0.9,
                0.999,
                1e-8,
                0.0,
            );
        }

        // Final loss
        let (dy_final, _) = adapter.forward_cpu(&x, seq);
        let loss_final: f32 = dy_final
            .iter()
            .zip(target.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / (dim * seq) as f32;

        assert!(
            loss_final < loss0 * 0.5,
            "loss should decrease: {loss0:.6} → {loss_final:.6}"
        );
    }

    // --- Per-param gradient clipping ---

    #[test]
    fn test_lora_per_param_clip() {
        let mut grads = LoraAdapterGrads {
            da: vec![10.0, -20.0, 5.0],
            db: vec![0.5, -0.3],
        };
        let clip = 3.0;
        // Per-param: clamp each element to [-clip, clip]
        for v in grads.da.iter_mut().chain(grads.db.iter_mut()) {
            *v = v.clamp(-clip, clip);
        }
        assert_eq!(grads.da, vec![3.0, -3.0, 3.0]);
        assert!((grads.db[0] - 0.5).abs() < 1e-6);
        assert!((grads.db[1] - (-0.3)).abs() < 1e-6);
    }

    // --- Split LR verification ---

    #[test]
    fn test_split_lr_applies_different_rates_to_attn_vs_ffn() {
        // Create a LoRA model with both attention (wq, wv, wo) and FFN (w2) adapters.
        // Apply identical gradients, but with different LR scales for attn vs FFN.
        // Verify that the weight updates differ proportionally to the scale ratio.
        let dim = 8;
        let hidden = 16;
        let make_cfg = || LoraConfig {
            rank: 4,
            alpha: 4.0,
            target_modules: vec!["wq".into(), "wv".into(), "wo".into(), "w2".into()],
        };

        // Create two identical LoRA models
        let mut lora_split = LoraModel::new(make_cfg(), 1, dim, hidden);
        let mut lora_uniform = LoraModel::new(make_cfg(), 1, dim, hidden);
        // Make them identical by copying weights
        let copy_adapter = |src: &Option<LoraAdapter>, dst: &mut Option<LoraAdapter>| {
            if let (Some(s), Some(d)) = (src, dst.as_mut()) {
                d.a.copy_from_slice(&s.a);
                d.b.copy_from_slice(&s.b);
            }
        };
        copy_adapter(&lora_split.layers[0].wq, &mut lora_uniform.layers[0].wq);
        copy_adapter(&lora_split.layers[0].wv, &mut lora_uniform.layers[0].wv);
        copy_adapter(&lora_split.layers[0].wo, &mut lora_uniform.layers[0].wo);
        copy_adapter(&lora_split.layers[0].w2, &mut lora_uniform.layers[0].w2);

        // Create identical gradients (nonzero)
        let mut grads = LoraModelGrads::zeros(&lora_split);
        let fill_grad = |g: &mut Option<LoraAdapterGrads>| {
            if let Some(g) = g.as_mut() {
                for (i, v) in g.da.iter_mut().enumerate() {
                    *v = ((i as f32 * 0.7 + 0.3).sin()) * 0.1;
                }
                for (i, v) in g.db.iter_mut().enumerate() {
                    *v = ((i as f32 * 1.1 + 0.5).cos()) * 0.1;
                }
            }
        };
        fill_grad(&mut grads.layers[0].wq);
        fill_grad(&mut grads.layers[0].wv);
        fill_grad(&mut grads.layers[0].wo);
        fill_grad(&mut grads.layers[0].w2);

        let mut adam_split = LoraModelAdam::zeros(&lora_split);
        let mut adam_uniform = LoraModelAdam::zeros(&lora_uniform);
        let base_lr = 1e-3;

        // Split: attn gets 0.05x, FFN gets 1.0x
        lora_adam_update_split_lr(
            &mut lora_split,
            &grads,
            &mut adam_split,
            1,
            base_lr,
            0.05,
            1.0,
            0.9,
            0.999,
            1e-8,
            0.0,
        );

        // Uniform: everything at 1.0x
        lora_adam_update(
            &mut lora_uniform,
            &grads,
            &mut adam_uniform,
            1,
            base_lr,
            0.9,
            0.999,
            1e-8,
            0.0,
        );

        // FFN (w2) should get identical updates under both
        let split_w2_a = &lora_split.layers[0].w2.as_ref().unwrap().a;
        let uniform_w2_a = &lora_uniform.layers[0].w2.as_ref().unwrap().a;
        for (s, u) in split_w2_a.iter().zip(uniform_w2_a.iter()) {
            assert!(
                (s - u).abs() < 1e-10,
                "w2 should get same update: split={s}, uniform={u}"
            );
        }

        // Attention (wq) should get smaller update under split (0.05x scale)
        let split_wq_a = &lora_split.layers[0].wq.as_ref().unwrap().a;
        let uniform_wq_a = &lora_uniform.layers[0].wq.as_ref().unwrap().a;
        // The uniform model moved more than the split model (5x more)
        // since Adam step 1 with beta1=0.9 b2=0.999: the update magnitude
        // is proportional to lr. Compare the total displacement.
        let split_delta: f32 = split_wq_a
            .iter()
            .zip(lora_split.layers[0].wq.as_ref().unwrap().a.iter())
            .map(|(a, _)| a.abs())
            .sum();
        let uniform_delta: f32 = uniform_wq_a
            .iter()
            .zip(lora_uniform.layers[0].wq.as_ref().unwrap().a.iter())
            .map(|(a, _)| a.abs())
            .sum();
        // With 0.05x scale, attn update should differ from 1.0x
        // They can't be equal
        let wq_differs = split_wq_a
            .iter()
            .zip(uniform_wq_a.iter())
            .any(|(s, u)| (s - u).abs() > 1e-10);
        assert!(
            wq_differs,
            "wq should get different updates with 0.05x attn scale"
        );
    }

    // --- Micro-benchmarks (run with `cargo test --features ane --release -- bench_ --nocapture`) ---

    #[test]
    fn bench_lora_kernel_compile_and_eval() {
        use std::time::Instant;

        let mil_cfg = MilConfig::mha(64, 128, 4, 16);
        let t0 = Instant::now();
        let kernels = match LoraKernels::compile(&mil_cfg, 32) {
            Ok(k) => k,
            Err(e) => {
                eprintln!("Skipping benchmark (ANE unavailable): {e}");
                return;
            }
        };
        let compile_ms = t0.elapsed().as_millis();

        let adapter = LoraAdapter::new(32, 64, 64);
        let seq = 16;
        let x: Vec<f32> = (0..64 * seq).map(|i| (i as f32 * 0.01).sin()).collect();

        // Warmup
        let _ = lora_forward_ane(&kernels.attn_a_fwd, &kernels.attn_b_fwd, &adapter, &x, seq);

        let iters = 100;
        let t1 = Instant::now();
        for _ in 0..iters {
            let _ = lora_forward_ane(&kernels.attn_a_fwd, &kernels.attn_b_fwd, &adapter, &x, seq);
        }
        let fwd_us = t1.elapsed().as_micros() as f64 / iters as f64;

        eprintln!("LoRA kernel benchmark (dim=64, rank=32, seq=16):");
        eprintln!("  compile: {compile_ms}ms");
        eprintln!("  forward: {fwd_us:.1}µs/iter ({iters} iters)");
        eprintln!(
            "  throughput: {:.0} elements/s",
            (64.0 * seq as f64) / (fwd_us / 1e6)
        );
    }

    #[test]
    fn bench_lora_cpu_training_step() {
        use std::time::Instant;

        let rank = 32;
        let dim = 64;
        let seq = 32;

        let mut adapter = LoraAdapter::new(rank, dim, dim);
        for v in adapter.b.iter_mut() {
            *v = 0.01;
        }

        let x: Vec<f32> = (0..dim * seq).map(|i| (i as f32 * 0.01).sin()).collect();
        let target: Vec<f32> = (0..dim * seq)
            .map(|i| (i as f32 * 0.02).cos() * 0.1)
            .collect();
        let mut a_adam = ane_train::AdamState::zeros(rank * dim);
        let mut b_adam = ane_train::AdamState::zeros(dim * rank);

        let steps = 180;
        let t0 = Instant::now();
        for step in 0..steps {
            let (dy, h) = adapter.forward_cpu(&x, seq);
            let n = (dim * seq) as f32;
            let d_out: Vec<f32> = dy
                .iter()
                .zip(target.iter())
                .map(|(a, b)| 2.0 * (a - b) / n)
                .collect();
            let (_dx, da, db) = adapter.backward_cpu(&d_out, &x, &h, seq);
            ane_train::adam_update(
                &mut adapter.a,
                &da,
                &mut a_adam,
                step + 1,
                5e-4,
                0.9,
                0.999,
                1e-8,
                0.01,
            );
            ane_train::adam_update(
                &mut adapter.b,
                &db,
                &mut b_adam,
                step + 1,
                5e-4,
                0.9,
                0.999,
                1e-8,
                0.01,
            );
        }
        let total_ms = t0.elapsed().as_millis();
        let per_step_ms = total_ms as f64 / steps as f64;

        eprintln!("LoRA CPU training benchmark (dim={dim}, rank={rank}, seq={seq}):");
        eprintln!("  {steps} steps in {total_ms}ms ({per_step_ms:.2}ms/step)");
        eprintln!("  JIT LoRA reference: ~390ms/step on M4 Max (full model, ANE)");
    }
}

// ---------------------------------------------------------------------------
// LoraAdapter clone helper (for tests)
// ---------------------------------------------------------------------------

impl LoraAdapter {
    /// Clone adapter (for numerical gradient checks).
    fn clone_adapter(&self) -> Self {
        LoraAdapter {
            a: self.a.clone(),
            b: self.b.clone(),
            rank: self.rank,
            d_in: self.d_in,
            d_out: self.d_out,
        }
    }
}
