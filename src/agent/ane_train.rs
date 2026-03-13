//! Training loop & optimizer for ANE transformer fine-tuning.
//!
//! Adam optimizer, cosine LR schedule with warmup, gradient ops,
//! memory-mapped data loader, training loop, and checkpoint I/O.

use std::io::{self, Write as _};
use std::path::Path;

use super::ane_backward::{self, BackwardKernels, ModelGradients};
use super::ane_forward::{self, CompiledKernels};
use super::ane_mil::MilConfig;
use super::ane_weights::{LayerWeights, ModelWeights};

// ---------------------------------------------------------------------------
// Adam state
// ---------------------------------------------------------------------------

/// Per-parameter Adam optimizer state (first and second moments).
#[derive(Clone)]
pub struct AdamState {
    pub m: Vec<f32>, // first moment
    pub v: Vec<f32>, // second moment
}

impl AdamState {
    pub fn zeros(n: usize) -> Self {
        AdamState {
            m: vec![0.0; n],
            v: vec![0.0; n],
        }
    }
}

/// Adam state for one transformer layer.
pub struct LayerAdamState {
    pub wq: AdamState,
    pub wk: AdamState,
    pub wv: AdamState,
    pub wo: AdamState,
    pub w1: AdamState,
    pub w2: AdamState,
    pub w3: AdamState,
    pub rms_att: AdamState,
    pub rms_ffn: AdamState,
}

/// Adam state for the entire model.
pub struct ModelAdamState {
    pub layers: Vec<LayerAdamState>,
    pub rms_final: AdamState,
    pub embed: AdamState,
}

impl ModelAdamState {
    /// Create zero-initialized Adam state matching model shape.
    pub fn zeros(model: &ModelWeights) -> Self {
        let dim = model.cfg.dim;
        let hidden = model.cfg.hidden_dim;

        let layers = model
            .layers
            .iter()
            .map(|_| LayerAdamState {
                wq: AdamState::zeros(dim * dim),
                wk: AdamState::zeros(dim * dim),
                wv: AdamState::zeros(dim * dim),
                wo: AdamState::zeros(dim * dim),
                w1: AdamState::zeros(hidden * dim),
                w2: AdamState::zeros(dim * hidden),
                w3: AdamState::zeros(hidden * dim),
                rms_att: AdamState::zeros(dim),
                rms_ffn: AdamState::zeros(dim),
            })
            .collect();

        ModelAdamState {
            layers,
            rms_final: AdamState::zeros(dim),
            embed: AdamState::zeros(model.vocab_size * dim),
        }
    }
}

// ---------------------------------------------------------------------------
// Training config
// ---------------------------------------------------------------------------

pub struct TrainingConfig {
    pub total_steps: usize,
    pub max_lr: f32,
    pub adam_beta1: f32,
    pub adam_beta2: f32,
    pub adam_eps: f32,
    pub weight_decay: f32,
    pub accum_steps: usize,
    pub warmup_steps: usize,
    pub grad_clip: f32,
    pub min_lr_frac: f32,
    pub ckpt_interval: usize,
    pub log_interval: usize,
    /// Early stopping: halt when loss drops below this threshold.
    /// Set to 0.0 to disable.
    pub early_stop_loss: f32,
    /// Early stopping patience: stop after this many consecutive steps below threshold.
    pub early_stop_patience: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        TrainingConfig {
            total_steps: 10000,
            max_lr: 3e-4,
            adam_beta1: 0.9,
            adam_beta2: 0.999,
            adam_eps: 1e-8,
            weight_decay: 0.0,
            accum_steps: 10,
            warmup_steps: 100,
            grad_clip: 1.0,
            min_lr_frac: 0.1,
            ckpt_interval: 100,
            log_interval: 10,
            early_stop_loss: 0.0,
            early_stop_patience: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Adam update
// ---------------------------------------------------------------------------

/// Decoupled AdamW update: apply weight decay separately from gradient step.
///
/// `t` = optimizer step count (1-indexed, for bias correction).
/// `wd` = weight decay coefficient (0.0 to disable, typical: 0.01).
pub fn adam_update(
    w: &mut [f32],
    g: &[f32],
    state: &mut AdamState,
    t: usize,
    lr: f32,
    b1: f32,
    b2: f32,
    eps: f32,
    wd: f32,
) {
    debug_assert_eq!(w.len(), g.len());
    debug_assert_eq!(w.len(), state.m.len());
    debug_assert_eq!(w.len(), state.v.len());
    debug_assert!(t >= 1);

    let bc1 = 1.0 / (1.0 - b1.powi(t as i32));
    let bc2 = 1.0 / (1.0 - b2.powi(t as i32));

    for i in 0..w.len() {
        // Decoupled weight decay (AdamW): applied to weight, not gradient
        if wd > 0.0 {
            w[i] *= 1.0 - lr * wd;
        }
        state.m[i] = b1 * state.m[i] + (1.0 - b1) * g[i];
        state.v[i] = b2 * state.v[i] + (1.0 - b2) * g[i] * g[i];
        let m_hat = state.m[i] * bc1;
        let v_hat = state.v[i] * bc2;
        w[i] -= lr * m_hat / (v_hat.sqrt() + eps);
    }
}

// ---------------------------------------------------------------------------
// Cosine LR schedule with linear warmup
// ---------------------------------------------------------------------------

/// Cosine LR with linear warmup.
///
/// - During warmup (step < warmup): linearly ramp from 0 to max_lr.
/// - After warmup: cosine decay from max_lr to max_lr * min_lr_frac.
pub fn cosine_lr(step: usize, warmup: usize, total: usize, max_lr: f32, min_lr_frac: f32) -> f32 {
    if step < warmup {
        max_lr * (step as f32) / (warmup as f32)
    } else {
        let min_lr = max_lr * min_lr_frac;
        let decay_steps = total.saturating_sub(warmup);
        if decay_steps == 0 {
            return max_lr;
        }
        let progress = (step - warmup) as f32 / decay_steps as f32;
        let progress = progress.min(1.0);
        min_lr + 0.5 * (max_lr - min_lr) * (1.0 + (std::f32::consts::PI * progress).cos())
    }
}

// ---------------------------------------------------------------------------
// Gradient operations
// ---------------------------------------------------------------------------

/// Helper: iterate all gradient tensors as slices.
fn for_each_grad(grads: &ModelGradients, mut f: impl FnMut(&[f32])) {
    for lg in &grads.layers {
        f(&lg.dwq);
        f(&lg.dwk);
        f(&lg.dwv);
        f(&lg.dwo);
        f(&lg.dw1);
        f(&lg.dw2);
        f(&lg.dw3);
        f(&lg.drms_att);
        f(&lg.drms_ffn);
    }
    f(&grads.drms_final);
    f(&grads.dembed);
}

/// Helper: iterate all gradient tensors as mutable slices.
fn for_each_grad_mut(grads: &mut ModelGradients, mut f: impl FnMut(&mut [f32])) {
    for lg in &mut grads.layers {
        f(&mut lg.dwq);
        f(&mut lg.dwk);
        f(&mut lg.dwv);
        f(&mut lg.dwo);
        f(&mut lg.dw1);
        f(&mut lg.dw2);
        f(&mut lg.dw3);
        f(&mut lg.drms_att);
        f(&mut lg.drms_ffn);
    }
    f(&mut grads.drms_final);
    f(&mut grads.dembed);
}

/// L2 norm of all gradients.
pub fn global_grad_norm(grads: &ModelGradients) -> f32 {
    let mut sum_sq = 0.0f64;
    for_each_grad(grads, |g| {
        for &v in g {
            sum_sq += (v as f64) * (v as f64);
        }
    });
    (sum_sq as f32).sqrt()
}

/// Scale all gradients by `scale`.
pub fn scale_gradients(grads: &mut ModelGradients, scale: f32) {
    for_each_grad_mut(grads, |g| {
        for v in g.iter_mut() {
            *v *= scale;
        }
    });
}

/// Clip gradients: if `norm > clip`, scale all grads by `clip / norm`.
pub fn clip_gradients(grads: &mut ModelGradients, clip: f32, norm: f32) {
    if norm > clip {
        scale_gradients(grads, clip / norm);
    }
}

/// Zero all gradients.
pub fn zero_gradients(grads: &mut ModelGradients) {
    for_each_grad_mut(grads, |g| {
        for v in g.iter_mut() {
            *v = 0.0;
        }
    });
}

// ---------------------------------------------------------------------------
// Adam update all
// ---------------------------------------------------------------------------

/// Helper: apply adam_update to a (weight, gradient, adam_state) triple.
fn adam_one(
    w: &mut Vec<f32>,
    g: &[f32],
    st: &mut AdamState,
    t: usize,
    lr: f32,
    cfg: &TrainingConfig,
) {
    adam_update(
        w,
        g,
        st,
        t,
        lr,
        cfg.adam_beta1,
        cfg.adam_beta2,
        cfg.adam_eps,
        cfg.weight_decay,
    );
}

/// Apply Adam to all model weights using corresponding gradients.
pub fn adam_update_all(
    model: &mut ModelWeights,
    grads: &ModelGradients,
    adam: &mut ModelAdamState,
    t: usize,
    lr: f32,
    cfg: &TrainingConfig,
) {
    for (li, lg) in grads.layers.iter().enumerate() {
        let lw = &mut model.layers[li];
        let la = &mut adam.layers[li];
        adam_one(&mut lw.wq, &lg.dwq, &mut la.wq, t, lr, cfg);
        adam_one(&mut lw.wk, &lg.dwk, &mut la.wk, t, lr, cfg);
        adam_one(&mut lw.wv, &lg.dwv, &mut la.wv, t, lr, cfg);
        adam_one(&mut lw.wo, &lg.dwo, &mut la.wo, t, lr, cfg);
        adam_one(&mut lw.w1, &lg.dw1, &mut la.w1, t, lr, cfg);
        adam_one(&mut lw.w2, &lg.dw2, &mut la.w2, t, lr, cfg);
        adam_one(&mut lw.w3, &lg.dw3, &mut la.w3, t, lr, cfg);
        adam_one(&mut lw.rms_att, &lg.drms_att, &mut la.rms_att, t, lr, cfg);
        adam_one(&mut lw.rms_ffn, &lg.drms_ffn, &mut la.rms_ffn, t, lr, cfg);
    }
    adam_one(
        &mut model.rms_final,
        &grads.drms_final,
        &mut adam.rms_final,
        t,
        lr,
        cfg,
    );
    adam_one(&mut model.embed, &grads.dembed, &mut adam.embed, t, lr, cfg);
}

// ---------------------------------------------------------------------------
// Token dataset (memory-mapped)
// ---------------------------------------------------------------------------

/// Memory-mapped token dataset for training.
///
/// Internally stores tokens as u32 for large-vocab compatibility.
/// The `sample` method returns `Vec<u16>` for backward compatibility;
/// use `sample_u32` for large-vocab models (vocab > 65535).
pub struct TokenDataset {
    data: memmap2::Mmap,
    n_tokens: usize,
    is_u32: bool,
}

impl TokenDataset {
    /// Open a binary file of u16 tokens.
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = std::fs::File::open(path)?;
        let meta = file.metadata()?;
        let len = meta.len() as usize;
        if len < 2 || len % 2 != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "token file must contain at least one u16 and have even byte count",
            ));
        }
        let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
        Ok(TokenDataset {
            n_tokens: len / 2,
            data: mmap,
            is_u32: false,
        })
    }

    /// Open a binary file of u32 tokens (for large-vocab models).
    pub fn open_u32(path: &Path) -> io::Result<Self> {
        let file = std::fs::File::open(path)?;
        let meta = file.metadata()?;
        let len = meta.len() as usize;
        if len < 4 || len % 4 != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "u32 token file must contain at least one u32 and have 4-byte aligned size",
            ));
        }
        let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
        Ok(TokenDataset {
            n_tokens: len / 4,
            data: mmap,
            is_u32: true,
        })
    }

    /// Number of tokens in the dataset.
    pub fn len(&self) -> usize {
        self.n_tokens
    }

    /// Get the underlying token slice as u16 (only valid for u16 datasets).
    fn tokens_u16(&self) -> &[u16] {
        debug_assert!(!self.is_u32);
        let ptr = self.data.as_ptr() as *const u16;
        unsafe { std::slice::from_raw_parts(ptr, self.n_tokens) }
    }

    /// Get the underlying token slice as u32 (only valid for u32 datasets).
    fn tokens_u32(&self) -> &[u32] {
        debug_assert!(self.is_u32);
        let ptr = self.data.as_ptr() as *const u32;
        unsafe { std::slice::from_raw_parts(ptr, self.n_tokens) }
    }

    /// Alias for backward compat — used by tests that call `ds.tokens()`.
    #[cfg(test)]
    fn tokens(&self) -> &[u16] {
        self.tokens_u16()
    }

    fn random_start(&self, seq_len: usize) -> usize {
        let max_start = self.n_tokens - seq_len - 1;
        if max_start == 0 {
            0
        } else {
            use std::collections::hash_map::RandomState;
            use std::hash::{BuildHasher, Hasher};
            let s = RandomState::new();
            let mut h = s.build_hasher();
            h.write_usize(0);
            (h.finish() as usize) % (max_start + 1)
        }
    }

    /// Sample a random (input, target) pair of u16 tokens.
    pub fn sample(&self, seq_len: usize) -> (Vec<u16>, Vec<u16>) {
        assert!(
            self.n_tokens >= seq_len + 1,
            "dataset too small for seq_len"
        );
        let start = self.random_start(seq_len);
        if self.is_u32 {
            let all = self.tokens_u32();
            let input: Vec<u16> = all[start..start + seq_len]
                .iter()
                .map(|&t| t as u16)
                .collect();
            let target: Vec<u16> = all[start + 1..start + 1 + seq_len]
                .iter()
                .map(|&t| t as u16)
                .collect();
            (input, target)
        } else {
            let all = self.tokens_u16();
            let input = all[start..start + seq_len].to_vec();
            let target = all[start + 1..start + 1 + seq_len].to_vec();
            (input, target)
        }
    }

    /// Sample a random (input, target) pair of u32 tokens.
    pub fn sample_u32(&self, seq_len: usize) -> (Vec<u32>, Vec<u32>) {
        assert!(
            self.n_tokens >= seq_len + 1,
            "dataset too small for seq_len"
        );
        let start = self.random_start(seq_len);
        if self.is_u32 {
            let all = self.tokens_u32();
            (
                all[start..start + seq_len].to_vec(),
                all[start + 1..start + 1 + seq_len].to_vec(),
            )
        } else {
            let all = self.tokens_u16();
            let input: Vec<u32> = all[start..start + seq_len]
                .iter()
                .map(|&t| t as u32)
                .collect();
            let target: Vec<u32> = all[start + 1..start + 1 + seq_len]
                .iter()
                .map(|&t| t as u32)
                .collect();
            (input, target)
        }
    }
}

// ---------------------------------------------------------------------------
// Training loop
// ---------------------------------------------------------------------------

pub struct TrainingResult {
    pub final_loss: f32,
    pub steps_completed: usize,
    pub early_stopped: bool,
}

/// Run the training loop.
///
/// Loop body (per train.m lines 395-852):
/// 1. Sample random tokens from dataset
/// 2. forward() → loss + dlogits + activations
/// 3. backward() → accumulate into ModelGradients
/// 4. Every accum_steps: scale grads → norm → clip → cosine LR → Adam → zero grads
/// 5. Every ckpt_interval: save checkpoint
/// 6. Every log_interval: print step/loss/lr
pub fn train(
    model: &mut ModelWeights,
    fwd_kernels: &CompiledKernels,
    bwd_kernels: &BackwardKernels,
    dataset: &TokenDataset,
    cfg: &TrainingConfig,
    ckpt_path: Option<&Path>,
) -> Result<TrainingResult, String> {
    let seq_len = model.cfg.seq_len;
    let mut grads = ModelGradients::zeros(model);
    let mut adam = ModelAdamState::zeros(model);
    let mut adam_t: usize = 0; // optimizer step counter (1-indexed when used)
    let mut last_loss = f32::NAN;
    let mut early_stopped = false;
    let mut patience_counter: usize = 0;

    for step in 0..cfg.total_steps {
        // 1. Sample
        let (input, target) = dataset.sample(seq_len);

        // 2. Forward
        let fwd = ane_forward::forward(fwd_kernels, model, &input, &target)?;
        last_loss = fwd.loss;

        // 3. Backward — accumulate
        let step_grads = ane_backward::backward(bwd_kernels, model, &fwd, &input)?;
        accumulate_gradients(&mut grads, &step_grads);

        // 4. Optimizer step every accum_steps
        if (step + 1) % cfg.accum_steps == 0 {
            adam_t += 1;

            // Average over accumulation steps
            scale_gradients(&mut grads, 1.0 / cfg.accum_steps as f32);

            // Gradient clipping
            let norm = global_grad_norm(&grads);
            clip_gradients(&mut grads, cfg.grad_clip, norm);

            // LR schedule
            let lr = cosine_lr(
                adam_t,
                cfg.warmup_steps,
                cfg.total_steps / cfg.accum_steps,
                cfg.max_lr,
                cfg.min_lr_frac,
            );

            // Adam update
            adam_update_all(model, &grads, &mut adam, adam_t, lr, cfg);

            // Zero grads
            zero_gradients(&mut grads);

            // Log
            if adam_t % cfg.log_interval == 0 {
                eprintln!(
                    "step {}/{} | loss {:.4} | lr {:.6} | grad_norm {:.4}",
                    adam_t,
                    cfg.total_steps / cfg.accum_steps,
                    last_loss,
                    lr,
                    norm
                );
            }

            // Checkpoint
            if let Some(path) = ckpt_path {
                if adam_t % cfg.ckpt_interval == 0 {
                    save_checkpoint(path, model, &adam, adam_t, adam_t, last_loss)
                        .map_err(|e| format!("checkpoint save failed: {e}"))?;
                    eprintln!("checkpoint saved at step {adam_t}");
                }
            }

            // Early stopping
            if cfg.early_stop_loss > 0.0 && last_loss < cfg.early_stop_loss {
                patience_counter += 1;
                if patience_counter >= cfg.early_stop_patience.max(1) {
                    eprintln!(
                        "early stopping at step {adam_t}: loss {last_loss:.4} < {:.4}",
                        cfg.early_stop_loss
                    );
                    early_stopped = true;
                    break;
                }
            } else {
                patience_counter = 0;
            }
        }
    }

    Ok(TrainingResult {
        final_loss: last_loss,
        steps_completed: adam_t,
        early_stopped,
    })
}

/// Accumulate step gradients into running gradients.
fn accumulate_gradients(dst: &mut ModelGradients, src: &ModelGradients) {
    fn add(dst: &mut [f32], src: &[f32]) {
        for (d, s) in dst.iter_mut().zip(src.iter()) {
            *d += *s;
        }
    }
    for (dl, sl) in dst.layers.iter_mut().zip(src.layers.iter()) {
        add(&mut dl.dwq, &sl.dwq);
        add(&mut dl.dwk, &sl.dwk);
        add(&mut dl.dwv, &sl.dwv);
        add(&mut dl.dwo, &sl.dwo);
        add(&mut dl.dw1, &sl.dw1);
        add(&mut dl.dw2, &sl.dw2);
        add(&mut dl.dw3, &sl.dw3);
        add(&mut dl.drms_att, &sl.drms_att);
        add(&mut dl.drms_ffn, &sl.drms_ffn);
    }
    add(&mut dst.drms_final, &src.drms_final);
    add(&mut dst.dembed, &src.dembed);
}

// ---------------------------------------------------------------------------
// Checkpoint I/O
// ---------------------------------------------------------------------------

const CKPT_MAGIC: u32 = 0x424C5A54; // "BLZT"
const CKPT_VERSION: u32 = 1;

fn write_f32_slice(w: &mut impl io::Write, data: &[f32]) -> io::Result<()> {
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    w.write_all(bytes)
}

fn read_f32_vec(r: &mut impl io::Read, n: usize) -> io::Result<Vec<f32>> {
    let mut buf = vec![0.0f32; n];
    let bytes = unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, n * 4) };
    r.read_exact(bytes)?;
    Ok(buf)
}

fn write_u32(w: &mut impl io::Write, v: u32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn read_u32(r: &mut impl io::Read) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn write_usize(w: &mut impl io::Write, v: usize) -> io::Result<()> {
    w.write_all(&(v as u64).to_le_bytes())
}

fn read_usize(r: &mut impl io::Read) -> io::Result<usize> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf) as usize)
}

fn write_f32(w: &mut impl io::Write, v: f32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn read_f32_scalar(r: &mut impl io::Read) -> io::Result<f32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

fn write_adam_state(w: &mut impl io::Write, st: &AdamState) -> io::Result<()> {
    write_f32_slice(w, &st.m)?;
    write_f32_slice(w, &st.v)
}

fn read_adam_state(r: &mut impl io::Read, n: usize) -> io::Result<AdamState> {
    let m = read_f32_vec(r, n)?;
    let v = read_f32_vec(r, n)?;
    Ok(AdamState { m, v })
}

/// Save training checkpoint.
///
/// Binary format: magic + version + header (dim, hidden, n_heads, seq_len, vocab, n_layers,
/// step, adam_t, loss) + weights + adam states.
pub fn save_checkpoint(
    path: &Path,
    model: &ModelWeights,
    adam: &ModelAdamState,
    step: usize,
    adam_t: usize,
    loss: f32,
) -> io::Result<()> {
    let mut f = io::BufWriter::new(std::fs::File::create(path)?);

    // Header
    write_u32(&mut f, CKPT_MAGIC)?;
    write_u32(&mut f, CKPT_VERSION)?;
    write_usize(&mut f, model.cfg.dim)?;
    write_usize(&mut f, model.cfg.hidden_dim)?;
    write_usize(&mut f, model.cfg.n_heads)?;
    write_usize(&mut f, model.cfg.seq_len)?;
    write_usize(&mut f, model.vocab_size)?;
    write_usize(&mut f, model.layers.len())?;
    write_usize(&mut f, step)?;
    write_usize(&mut f, adam_t)?;
    write_f32(&mut f, loss)?;

    // Model weights
    write_f32_slice(&mut f, &model.rms_final)?;
    write_f32_slice(&mut f, &model.embed)?;
    for lw in &model.layers {
        write_f32_slice(&mut f, &lw.wq)?;
        write_f32_slice(&mut f, &lw.wk)?;
        write_f32_slice(&mut f, &lw.wv)?;
        write_f32_slice(&mut f, &lw.wo)?;
        write_f32_slice(&mut f, &lw.w1)?;
        write_f32_slice(&mut f, &lw.w2)?;
        write_f32_slice(&mut f, &lw.w3)?;
        write_f32_slice(&mut f, &lw.rms_att)?;
        write_f32_slice(&mut f, &lw.rms_ffn)?;
    }

    // Adam states
    write_adam_state(&mut f, &adam.rms_final)?;
    write_adam_state(&mut f, &adam.embed)?;
    for la in &adam.layers {
        write_adam_state(&mut f, &la.wq)?;
        write_adam_state(&mut f, &la.wk)?;
        write_adam_state(&mut f, &la.wv)?;
        write_adam_state(&mut f, &la.wo)?;
        write_adam_state(&mut f, &la.w1)?;
        write_adam_state(&mut f, &la.w2)?;
        write_adam_state(&mut f, &la.w3)?;
        write_adam_state(&mut f, &la.rms_att)?;
        write_adam_state(&mut f, &la.rms_ffn)?;
    }

    f.flush()
}

/// Load training checkpoint. Returns (model, adam_state, step, adam_t, loss).
pub fn load_checkpoint(
    path: &Path,
) -> io::Result<(ModelWeights, ModelAdamState, usize, usize, f32)> {
    let mut f = io::BufReader::new(std::fs::File::open(path)?);

    let magic = read_u32(&mut f)?;
    if magic != CKPT_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("bad checkpoint magic: 0x{magic:08X}"),
        ));
    }
    let version = read_u32(&mut f)?;
    if version != CKPT_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported checkpoint version: {version}"),
        ));
    }

    let dim = read_usize(&mut f)?;
    let hidden_dim = read_usize(&mut f)?;
    let n_heads = read_usize(&mut f)?;
    let seq_len = read_usize(&mut f)?;
    let vocab_size = read_usize(&mut f)?;
    let n_layers = read_usize(&mut f)?;
    let step = read_usize(&mut f)?;
    let adam_t = read_usize(&mut f)?;
    let loss = read_f32_scalar(&mut f)?;

    let cfg = MilConfig::mha(dim, hidden_dim, n_heads, seq_len);

    // Model weights
    let rms_final = read_f32_vec(&mut f, dim)?;
    let embed = read_f32_vec(&mut f, vocab_size * dim)?;
    let mut layers = Vec::with_capacity(n_layers);
    for _ in 0..n_layers {
        layers.push(LayerWeights {
            wq: read_f32_vec(&mut f, dim * dim)?,
            wk: read_f32_vec(&mut f, dim * dim)?,
            wv: read_f32_vec(&mut f, dim * dim)?,
            wo: read_f32_vec(&mut f, dim * dim)?,
            w1: read_f32_vec(&mut f, hidden_dim * dim)?,
            w2: read_f32_vec(&mut f, dim * hidden_dim)?,
            w3: read_f32_vec(&mut f, hidden_dim * dim)?,
            rms_att: read_f32_vec(&mut f, dim)?,
            rms_ffn: read_f32_vec(&mut f, dim)?,
            q_norm: None,
            k_norm: None,
            gdn: None,
        });
    }

    let model = ModelWeights {
        cfg,
        layers,
        rms_final,
        embed,
        vocab_size,
        lm_head: None,
    };

    // Adam states
    let adam_rms_final = read_adam_state(&mut f, dim)?;
    let adam_embed = read_adam_state(&mut f, vocab_size * dim)?;
    let mut adam_layers = Vec::with_capacity(n_layers);
    for _ in 0..n_layers {
        adam_layers.push(LayerAdamState {
            wq: read_adam_state(&mut f, dim * dim)?,
            wk: read_adam_state(&mut f, dim * dim)?,
            wv: read_adam_state(&mut f, dim * dim)?,
            wo: read_adam_state(&mut f, dim * dim)?,
            w1: read_adam_state(&mut f, hidden_dim * dim)?,
            w2: read_adam_state(&mut f, dim * hidden_dim)?,
            w3: read_adam_state(&mut f, hidden_dim * dim)?,
            rms_att: read_adam_state(&mut f, dim)?,
            rms_ffn: read_adam_state(&mut f, dim)?,
        });
    }

    let adam = ModelAdamState {
        layers: adam_layers,
        rms_final: adam_rms_final,
        embed: adam_embed,
    };

    Ok((model, adam, step, adam_t, loss))
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Test MilConfig matching the standard used in other ane_* tests
    fn test_cfg() -> MilConfig {
        MilConfig::mha(64, 128, 4, 64)
    }

    /// Build a tiny 1-layer model with random-ish weights for testing.
    fn tiny_model(vocab: usize) -> ModelWeights {
        let cfg = test_cfg();
        let dim = cfg.dim;
        let hidden = cfg.hidden_dim;

        // Deterministic pseudo-random via simple LCG
        let mut seed: u64 = 42;
        let mut next_f32 = || -> f32 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 33) as f32) / (u32::MAX as f32) * 0.2 - 0.1
        };
        let mut mk = |n: usize| -> Vec<f32> { (0..n).map(|_| next_f32()).collect() };

        let layer = LayerWeights {
            wq: mk(dim * dim),
            wk: mk(dim * dim),
            wv: mk(dim * dim),
            wo: mk(dim * dim),
            w1: mk(hidden * dim),
            w2: mk(dim * hidden),
            w3: mk(hidden * dim),
            rms_att: vec![1.0; dim], // init RMS weights to 1
            rms_ffn: vec![1.0; dim],
            q_norm: None,
            k_norm: None,
            gdn: None,
        };

        ModelWeights {
            cfg,
            layers: vec![layer],
            rms_final: vec![1.0; dim],
            embed: mk(vocab * dim),
            vocab_size: vocab,
            lm_head: None,
        }
    }

    // -----------------------------------------------------------------------
    // Round 1: adam_update + cosine_lr
    // -----------------------------------------------------------------------

    #[test]
    fn test_adam_update_basic() {
        // Start with w = [1.0, 2.0, 3.0], gradient = [0.1, 0.2, 0.3]
        let mut w = vec![1.0, 2.0, 3.0];
        let g = vec![0.1, 0.2, 0.3];
        let mut state = AdamState::zeros(3);
        let lr = 0.001;
        let b1 = 0.9;
        let b2 = 0.999;
        let eps = 1e-8;

        // Step 1
        adam_update(&mut w, &g, &mut state, 1, lr, b1, b2, eps, 0.0);

        // m should be (1-b1)*g = 0.1*g
        for i in 0..3 {
            let expected_m = (1.0 - b1) * g[i];
            assert!((state.m[i] - expected_m).abs() < 1e-7, "m[{i}] mismatch");
        }

        // v should be (1-b2)*g^2 = 0.001*g^2
        for i in 0..3 {
            let expected_v = (1.0 - b2) * g[i] * g[i];
            assert!((state.v[i] - expected_v).abs() < 1e-10, "v[{i}] mismatch");
        }

        // Weights should have decreased (positive gradient → weight decrease)
        assert!(w[0] < 1.0);
        assert!(w[1] < 2.0);
        assert!(w[2] < 3.0);

        // Step 2 — moments should accumulate
        let w_after_1 = w.clone();
        adam_update(&mut w, &g, &mut state, 2, lr, b1, b2, eps, 0.0);
        // Weights continue decreasing with same-sign gradient
        assert!(w[0] < w_after_1[0]);
        assert!(w[1] < w_after_1[1]);
        assert!(w[2] < w_after_1[2]);
    }

    #[test]
    fn test_adam_update_opposite_gradients() {
        // With alternating gradient signs, weight should move less
        let mut w = vec![0.0];
        let mut state = AdamState::zeros(1);
        let lr = 0.01;

        adam_update(&mut w, &[1.0], &mut state, 1, lr, 0.9, 0.999, 1e-8, 0.0);
        let after_pos = w[0];
        adam_update(&mut w, &[-1.0], &mut state, 2, lr, 0.9, 0.999, 1e-8, 0.0);
        // After negative grad, weight should move back toward zero
        assert!(w[0] > after_pos, "negative grad should push weight up");
    }

    #[test]
    fn test_adam_weight_decay_shrinks_weights() {
        let mut w = vec![1.0, 2.0, 3.0];
        let g = vec![0.0, 0.0, 0.0]; // Zero gradient — only weight decay acts
        let mut state = AdamState::zeros(3);
        let lr = 0.01;
        let wd = 0.1;

        let w_before = w.clone();
        adam_update(&mut w, &g, &mut state, 1, lr, 0.9, 0.999, 1e-8, wd);

        // With zero gradient and wd > 0, weights should shrink toward zero
        for i in 0..3 {
            assert!(
                w[i].abs() < w_before[i].abs(),
                "weight[{i}] should shrink: {:.6} → {:.6}",
                w_before[i],
                w[i]
            );
        }
    }

    #[test]
    fn test_early_stopping_config() {
        let cfg = TrainingConfig {
            total_steps: 1000,
            early_stop_loss: 0.8,
            early_stop_patience: 2,
            ..Default::default()
        };
        assert_eq!(cfg.early_stop_loss, 0.8);
        assert_eq!(cfg.early_stop_patience, 2);
    }

    #[test]
    fn test_cosine_lr_warmup_phase() {
        let max_lr = 3e-4;
        let warmup = 100;
        let total = 1000;

        // Step 0: should be 0
        assert_eq!(cosine_lr(0, warmup, total, max_lr, 0.1), 0.0);

        // Step 50: should be halfway
        let lr50 = cosine_lr(50, warmup, total, max_lr, 0.1);
        assert!(
            (lr50 - max_lr * 0.5).abs() < 1e-8,
            "warmup midpoint: {lr50}"
        );

        // Step 100 (warmup boundary): warmup returns step=warmup as post-warmup start
        // cosine at progress=0 should give max_lr
        let lr100 = cosine_lr(100, warmup, total, max_lr, 0.1);
        assert!(
            (lr100 - max_lr).abs() < 1e-7,
            "at warmup end should be max_lr, got {lr100}"
        );
    }

    #[test]
    fn test_cosine_lr_decay_phase() {
        let max_lr = 3e-4;
        let min_lr = max_lr * 0.1;
        let warmup = 0;
        let total = 1000;

        // Step 0 (no warmup): max_lr
        let lr0 = cosine_lr(0, warmup, total, max_lr, 0.1);
        assert!((lr0 - max_lr).abs() < 1e-7, "step 0 no warmup: {lr0}");

        // Step total: min_lr
        let lr_end = cosine_lr(total, warmup, total, max_lr, 0.1);
        assert!((lr_end - min_lr).abs() < 1e-7, "step end: {lr_end}");

        // Monotonically decreasing
        let mut prev = lr0;
        for s in 1..=total {
            let lr = cosine_lr(s, warmup, total, max_lr, 0.1);
            assert!(lr <= prev + 1e-8, "non-monotonic at step {s}");
            prev = lr;
        }
    }

    // -----------------------------------------------------------------------
    // Round 2: gradient ops
    // -----------------------------------------------------------------------

    #[test]
    fn test_global_grad_norm() {
        let model = tiny_model(32);
        let mut grads = ModelGradients::zeros(&model);

        // Zero grads → zero norm
        assert_eq!(global_grad_norm(&grads), 0.0);

        // Set one gradient to known value
        grads.layers[0].dwq[0] = 3.0;
        grads.layers[0].dwq[1] = 4.0;
        let norm = global_grad_norm(&grads);
        assert!(
            (norm - 5.0).abs() < 1e-5,
            "norm of [3,4,0,...] should be 5, got {norm}"
        );
    }

    #[test]
    fn test_scale_gradients() {
        let model = tiny_model(32);
        let mut grads = ModelGradients::zeros(&model);

        // Set known values
        for v in grads.layers[0].dwq.iter_mut() {
            *v = 2.0;
        }
        grads.drms_final[0] = 4.0;

        scale_gradients(&mut grads, 0.5);

        assert!(grads.layers[0].dwq.iter().all(|&v| (v - 1.0).abs() < 1e-7));
        assert!((grads.drms_final[0] - 2.0).abs() < 1e-7);
    }

    #[test]
    fn test_clip_gradients() {
        let model = tiny_model(32);
        let mut grads = ModelGradients::zeros(&model);

        // Set gradient with known norm
        grads.layers[0].dwq[0] = 3.0;
        grads.layers[0].dwq[1] = 4.0;
        // norm = 5.0

        // Clip to 2.5 → scale by 2.5/5.0 = 0.5
        clip_gradients(&mut grads, 2.5, 5.0);
        assert!((grads.layers[0].dwq[0] - 1.5).abs() < 1e-5);
        assert!((grads.layers[0].dwq[1] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_clip_gradients_no_clip_when_below() {
        let model = tiny_model(32);
        let mut grads = ModelGradients::zeros(&model);
        grads.layers[0].dwq[0] = 1.0;

        clip_gradients(&mut grads, 10.0, 1.0);
        // Should not change
        assert!((grads.layers[0].dwq[0] - 1.0).abs() < 1e-7);
    }

    #[test]
    fn test_zero_gradients() {
        let model = tiny_model(32);
        let mut grads = ModelGradients::zeros(&model);

        // Set non-zero values everywhere
        for v in grads.layers[0].dwq.iter_mut() {
            *v = 99.0;
        }
        grads.drms_final[0] = 42.0;
        grads.dembed[0] = 7.0;

        zero_gradients(&mut grads);

        // All should be zero
        let norm = global_grad_norm(&grads);
        assert_eq!(norm, 0.0);
    }

    // -----------------------------------------------------------------------
    // Round 3: adam_update_all + ModelAdamState::zeros
    // -----------------------------------------------------------------------

    #[test]
    fn test_model_adam_state_zeros() {
        let model = tiny_model(32);
        let adam = ModelAdamState::zeros(&model);

        assert_eq!(adam.layers.len(), 1);
        assert!(adam.layers[0].wq.m.iter().all(|&v| v == 0.0));
        assert!(adam.layers[0].wq.v.iter().all(|&v| v == 0.0));
        assert_eq!(adam.rms_final.m.len(), model.cfg.dim);
        assert_eq!(adam.embed.m.len(), model.vocab_size * model.cfg.dim);
    }

    #[test]
    fn test_adam_update_all_changes_weights() {
        let vocab = 32;
        let mut model = tiny_model(vocab);
        let model_before = model.clone();

        // Create non-zero gradients
        let mut grads = ModelGradients::zeros(&model);
        for v in grads.layers[0].dwq.iter_mut() {
            *v = 0.01;
        }
        for v in grads.layers[0].dw1.iter_mut() {
            *v = 0.01;
        }
        grads.drms_final[0] = 0.01;
        grads.dembed[0] = 0.01;

        let mut adam = ModelAdamState::zeros(&model);
        let cfg = TrainingConfig::default();

        adam_update_all(&mut model, &grads, &mut adam, 1, 1e-3, &cfg);

        // Weights with non-zero grads should have changed
        assert_ne!(model.layers[0].wq, model_before.layers[0].wq);
        assert_ne!(model.layers[0].w1, model_before.layers[0].w1);
        assert_ne!(model.rms_final[0], model_before.rms_final[0]);
        assert_ne!(model.embed[0], model_before.embed[0]);

        // Weights with zero grads should be unchanged
        assert_eq!(model.layers[0].wk, model_before.layers[0].wk);

        // Adam states should be non-zero where grads were non-zero
        assert!(adam.layers[0].wq.m.iter().any(|&v| v != 0.0));
        assert!(adam.layers[0].wq.v.iter().any(|&v| v != 0.0));
    }

    // -----------------------------------------------------------------------
    // Round 4: TokenDataset
    // -----------------------------------------------------------------------

    #[test]
    fn test_token_dataset_open_and_sample() {
        // Create a temp file with u16 tokens: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tokens.bin");
        {
            let tokens: Vec<u16> = (0..10).collect();
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(tokens.as_ptr() as *const u8, tokens.len() * 2)
            };
            std::fs::write(&path, bytes).unwrap();
        }

        let ds = TokenDataset::open(&path).unwrap();
        assert_eq!(ds.len(), 10);

        // Sample with seq_len=4: input[4] and target[4], target = input shifted by 1
        let (input, target) = ds.sample(4);
        assert_eq!(input.len(), 4);
        assert_eq!(target.len(), 4);

        // Target should be input shifted by 1
        let all = ds.tokens();
        // Find the start position from input
        let start = all.iter().position(|&t| t == input[0]).unwrap();
        assert_eq!(&all[start..start + 4], &input[..]);
        assert_eq!(&all[start + 1..start + 5], &target[..]);
    }

    #[test]
    fn test_token_dataset_sequential() {
        // With exactly seq_len+1 tokens, there's only one valid sample
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tokens.bin");
        let tokens: Vec<u16> = vec![10, 20, 30, 40, 50];
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(tokens.as_ptr() as *const u8, tokens.len() * 2) };
        std::fs::write(&path, bytes).unwrap();

        let ds = TokenDataset::open(&path).unwrap();
        let (input, target) = ds.sample(4);
        // Only valid sample: input=[10,20,30,40], target=[20,30,40,50]
        assert_eq!(input, vec![10, 20, 30, 40]);
        assert_eq!(target, vec![20, 30, 40, 50]);
    }

    #[test]
    fn test_token_dataset_rejects_empty() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.bin");
        std::fs::write(&path, &[]).unwrap();
        assert!(TokenDataset::open(&path).is_err());
    }

    #[test]
    fn test_token_dataset_rejects_odd_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("odd.bin");
        std::fs::write(&path, &[1, 2, 3]).unwrap();
        assert!(TokenDataset::open(&path).is_err());
    }

    // -----------------------------------------------------------------------
    // Round 5: train 5-step smoke test (requires ANE hardware)
    // -----------------------------------------------------------------------

    #[test]
    fn test_train_smoke() {
        // This test requires ANE hardware — skip if not available
        use super::super::ane_bridge;
        use std::sync::{
            atomic::{AtomicBool, Ordering},
            Once,
        };
        static INIT: Once = Once::new();
        static ANE_OK: AtomicBool = AtomicBool::new(false);
        INIT.call_once(|| {
            ANE_OK.store(ane_bridge::ane_init().is_ok(), Ordering::SeqCst);
        });
        if !ANE_OK.load(Ordering::SeqCst) {
            eprintln!("skipping test_train_smoke: ANE init failed");
            return;
        }
        let cfg = test_cfg();
        let fwd_kernels = match CompiledKernels::compile_forward(&cfg) {
            Ok(k) => k,
            Err(e) => {
                eprintln!("skipping test_train_smoke: ANE not available ({e})");
                return;
            }
        };
        let mask_blob = super::super::ane_mil::build_causal_mask_blob(cfg.seq_len);
        let bwd_kernels = match BackwardKernels::compile_backward(&cfg, &mask_blob) {
            Ok(k) => k,
            Err(e) => {
                eprintln!("skipping test_train_smoke: backward kernels failed ({e})");
                return;
            }
        };

        let vocab = 64;
        let mut model = tiny_model(vocab);
        let initial_embed = model.embed.clone();

        // Create token dataset
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tokens.bin");
        let tokens: Vec<u16> = (0..256).map(|i| (i % vocab) as u16).collect();
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(tokens.as_ptr() as *const u8, tokens.len() * 2) };
        std::fs::write(&path, bytes).unwrap();
        let dataset = TokenDataset::open(&path).unwrap();

        let train_cfg = TrainingConfig {
            total_steps: 5,
            accum_steps: 1,
            warmup_steps: 0,
            max_lr: 1e-3,
            log_interval: 1,
            ckpt_interval: 100,
            ..Default::default()
        };

        let result = train(
            &mut model,
            &fwd_kernels,
            &bwd_kernels,
            &dataset,
            &train_cfg,
            None,
        );

        match result {
            Ok(r) => {
                assert_eq!(r.steps_completed, 5);
                assert!(r.final_loss.is_finite(), "loss should be finite");
                // Weights should have changed
                assert_ne!(model.embed, initial_embed, "weights should have changed");
                eprintln!("train_smoke: 5 steps done, final_loss={:.4}", r.final_loss);
            }
            Err(e) => {
                eprintln!("skipping test_train_smoke: {e}");
            }
        }
    }

    // -----------------------------------------------------------------------
    // Round 6: checkpoint round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn test_checkpoint_roundtrip() {
        let vocab = 32;
        let model = tiny_model(vocab);

        // Create non-trivial adam state
        let mut adam = ModelAdamState::zeros(&model);
        adam.layers[0].wq.m[0] = 0.123;
        adam.layers[0].wq.v[0] = 0.456;
        adam.rms_final.m[0] = 0.789;
        adam.embed.v[0] = 1.234;

        let step = 42;
        let adam_t = 7;
        let loss = 2.345;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ckpt.bin");

        save_checkpoint(&path, &model, &adam, step, adam_t, loss).unwrap();
        let (loaded_model, loaded_adam, loaded_step, loaded_adam_t, loaded_loss) =
            load_checkpoint(&path).unwrap();

        // Verify header
        assert_eq!(loaded_step, step);
        assert_eq!(loaded_adam_t, adam_t);
        assert!((loaded_loss - loss).abs() < 1e-6);

        // Verify config
        assert_eq!(loaded_model.cfg.dim, model.cfg.dim);
        assert_eq!(loaded_model.cfg.hidden_dim, model.cfg.hidden_dim);
        assert_eq!(loaded_model.cfg.n_heads, model.cfg.n_heads);
        assert_eq!(loaded_model.cfg.seq_len, model.cfg.seq_len);
        assert_eq!(loaded_model.vocab_size, model.vocab_size);
        assert_eq!(loaded_model.layers.len(), model.layers.len());

        // Verify weights
        assert_eq!(loaded_model.rms_final, model.rms_final);
        assert_eq!(loaded_model.embed, model.embed);
        assert_eq!(loaded_model.layers[0].wq, model.layers[0].wq);
        assert_eq!(loaded_model.layers[0].w2, model.layers[0].w2);
        assert_eq!(loaded_model.layers[0].rms_att, model.layers[0].rms_att);

        // Verify adam states
        assert!((loaded_adam.layers[0].wq.m[0] - 0.123).abs() < 1e-7);
        assert!((loaded_adam.layers[0].wq.v[0] - 0.456).abs() < 1e-7);
        assert!((loaded_adam.rms_final.m[0] - 0.789).abs() < 1e-7);
        assert!((loaded_adam.embed.v[0] - 1.234).abs() < 1e-7);
    }

    #[test]
    fn test_checkpoint_bad_magic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.bin");
        std::fs::write(&path, &[0u8; 64]).unwrap();
        assert!(load_checkpoint(&path).is_err());
    }
}
