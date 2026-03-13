//! ANE↔MLX LoRA weight bridge.
//!
//! Transfers trained LoRA weights from ANE (CPU f32 Vec) to MLX (GPU Array).
//! Both systems use identical shapes: A=[rank, d_in], B=[d_out, rank].
//!
//! Name mapping: ANE wq/wv/wo/w2 → MLX q_proj/v_proj/o_proj/down_proj

use super::ane_lora::{LoraAdapter, LoraModel};
#[cfg(feature = "mlx")]
use super::ane_weights::WeightSource;

/// A single LoRA adapter's weights ready for transfer.
pub struct AdapterDelta {
    pub a: Vec<f32>, // [rank, d_in]
    pub b: Vec<f32>, // [d_out, rank]
    pub rank: usize,
    pub d_in: usize,
    pub d_out: usize,
}

/// Target within a layer that has LoRA.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoraTarget {
    QProj,
    VProj,
    OProj,
    DownProj,
}

impl LoraTarget {
    pub fn ane_name(&self) -> &'static str {
        match self {
            LoraTarget::QProj => "wq",
            LoraTarget::VProj => "wv",
            LoraTarget::OProj => "wo",
            LoraTarget::DownProj => "w2",
        }
    }

    pub fn mlx_name(&self) -> &'static str {
        match self {
            LoraTarget::QProj => "q_proj",
            LoraTarget::VProj => "v_proj",
            LoraTarget::OProj => "o_proj",
            LoraTarget::DownProj => "down_proj",
        }
    }
}

/// Per-layer delta: which target and its A/B weights.
pub struct LayerDelta {
    pub layer_idx: usize,
    pub target: LoraTarget,
    pub delta: AdapterDelta,
}

/// Complete set of LoRA deltas for transfer from ANE → MLX.
pub struct LoraDeltas {
    pub layers: Vec<LayerDelta>,
    pub scale: f32,
}

/// Extract all trained LoRA weights from an ANE LoraModel.
///
/// If `linear_attn_indices` is provided, attention LoRA (wq/wv/wo) is skipped
/// for those layers since GDN layers have stop_gradient on attention output.
pub fn extract_lora_deltas(lora: &LoraModel, linear_attn_indices: Option<&[usize]>) -> LoraDeltas {
    let mut layers = Vec::new();
    let linear = linear_attn_indices.unwrap_or(&[]);

    for (i, layer) in lora.layers.iter().enumerate() {
        let is_linear = linear.contains(&i);

        let targets: &[(LoraTarget, &Option<LoraAdapter>)] = &[
            (LoraTarget::QProj, &layer.wq),
            (LoraTarget::VProj, &layer.wv),
            (LoraTarget::OProj, &layer.wo),
            (LoraTarget::DownProj, &layer.w2),
        ];

        for (target, adapter) in targets {
            // Skip attention LoRA for GDN layers
            if is_linear && *target != LoraTarget::DownProj {
                continue;
            }

            if let Some(a) = adapter {
                layers.push(LayerDelta {
                    layer_idx: i,
                    target: *target,
                    delta: AdapterDelta {
                        a: a.a.clone(),
                        b: a.b.clone(),
                        rank: a.rank,
                        d_in: a.d_in,
                        d_out: a.d_out,
                    },
                });
            }
        }
    }

    LoraDeltas {
        layers,
        scale: lora.scale(),
    }
}

/// Apply LoRA deltas to an MLX model's LoraLinear layers.
///
/// For each delta, finds the corresponding LoraLinear in the MLX model
/// and overwrites its lora_a and lora_b weight arrays.
#[cfg(feature = "mlx")]
pub fn apply_lora_deltas(
    model: &mut super::mlx_lora::MlxLoraModel,
    deltas: &LoraDeltas,
) -> Result<usize, String> {
    use mlx_rs::Array;

    let mut applied = 0;

    for delta in &deltas.layers {
        if delta.layer_idx >= model.layers.len() {
            return Err(format!(
                "layer {} out of range (model has {})",
                delta.layer_idx,
                model.layers.len()
            ));
        }

        let d = &delta.delta;
        let new_a = Array::from_slice(&d.a, &[d.rank as i32, d.d_in as i32]);
        let new_b = Array::from_slice(&d.b, &[d.d_out as i32, d.rank as i32]);

        if model.layers[delta.layer_idx].apply_lora_weights(delta.target.mlx_name(), new_a, new_b) {
            applied += 1;
        }
    }

    Ok(applied)
}

// ---------------------------------------------------------------------------
// ANE training thread
// ---------------------------------------------------------------------------

/// Configuration for the ANE training thread.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AneTrainingOptimizer {
    AdamW,
    AneMuon,
}

#[derive(Debug, Clone)]
pub struct AneTrainingConfig {
    pub model_dir: std::path::PathBuf,
    pub mil_config: super::ane_mil::MilConfig,
    pub epochs: usize,
    pub lr: f32,
    pub linear_attn_indices: Vec<usize>,
    /// KV projection dim for GQA. `n_kv_heads * head_dim`. Equals `dim` for MHA.
    pub kv_dim: usize,
    /// Logit softcapping cap (15.0 recommended, 0.0 disables). Default 15.0.
    pub softcap: f32,
    /// Loss scaling factor (256.0 recommended, 1.0 disables). Default 256.0.
    pub loss_scale: f32,
    /// Attention LoRA LR multiplier (0.05 recommended). Default 0.05.
    pub lr_scale_attn: f32,
    /// FFN LoRA LR multiplier (1.0 recommended). Default 1.0.
    pub lr_scale_ffn: f32,
    /// Residual scaling factor (1.0/sqrt(2*n_layers) recommended, 1.0 disables). Default: 0.0 = auto-compute.
    pub residual_scale: f32,
    /// Optimizer used for LoRA updates.
    pub optimizer: AneTrainingOptimizer,
    /// If true, fail the run instead of silently falling back on ANE compile/eval errors.
    pub strict_ane: bool,
    /// Number of micro-batches to accumulate before one optimizer step (default: 1).
    pub accum_steps: usize,
}

/// Seq-len bucket sizes for ANE kernel compilation. Samples are padded to
/// the nearest bucket. Keeps compilation count low (3 × 10 = 30 kernels).
const BUCKET_SIZES: &[usize] = &[128, 256, 512, 1024];

/// Pre-compiled forward + backward kernels for multiple seq_len buckets.
pub struct BucketKernels {
    pub buckets: Vec<(
        usize,
        super::ane_forward::CompiledKernels,
        super::ane_backward::BackwardKernels,
    )>,
}

impl BucketKernels {
    pub fn empty() -> Self {
        BucketKernels {
            buckets: Vec::new(),
        }
    }

    /// Compile kernel sets for buckets that cover the given sample lengths.
    pub fn compile(
        sample_lens: &[usize],
        base_cfg: &super::ane_mil::MilConfig,
    ) -> Result<Self, String> {
        let mut buckets = Self::empty();
        buckets.ensure(sample_lens, base_cfg)?;
        Ok(buckets)
    }

    /// Compile only buckets that are not already cached.
    pub fn ensure(
        &mut self,
        sample_lens: &[usize],
        base_cfg: &super::ane_mil::MilConfig,
    ) -> Result<usize, String> {
        let mut needed: Vec<usize> = Vec::new();
        for &sl in sample_lens {
            let bucket = BUCKET_SIZES
                .iter()
                .copied()
                .find(|&b| b >= sl)
                .unwrap_or(*BUCKET_SIZES.last().unwrap());
            if !needed.contains(&bucket) && !self.buckets.iter().any(|(bs, _, _)| *bs == bucket) {
                needed.push(bucket);
            }
        }
        needed.sort();

        let mut compiled = 0usize;
        for bucket_seq in needed {
            let mut cfg = base_cfg.clone();
            cfg.seq_len = bucket_seq;
            let fwd = super::ane_forward::CompiledKernels::compile_forward(&cfg)?;
            let bwd = super::ane_backward::BackwardKernels::compile_backward(&cfg, &fwd.mask_blob)?;
            tracing::info!("ANE train: compiled kernels for seq_len={bucket_seq}");
            self.buckets.push((bucket_seq, fwd, bwd));
            compiled += 1;
        }
        self.buckets.sort_by_key(|(bucket_seq, _, _)| *bucket_seq);

        Ok(compiled)
    }

    /// Get the kernel set for a given sequence length (rounds up to nearest bucket).
    pub fn get(
        &self,
        seq_len: usize,
    ) -> &(
        usize,
        super::ane_forward::CompiledKernels,
        super::ane_backward::BackwardKernels,
    ) {
        self.buckets
            .iter()
            .find(|(bs, _, _)| *bs >= seq_len)
            .unwrap_or(self.buckets.last().unwrap())
    }
}

/// Pad a token sequence to `target_len` with zeros.
fn pad_to(tokens: &[u32], target_len: usize) -> Vec<u32> {
    let mut padded = tokens.to_vec();
    padded.resize(target_len, 0);
    padded
}

#[cfg(feature = "mlx")]
struct PreparedTrainingSample {
    tokens_u32: Vec<u32>,
    targets_u32: Vec<u32>,
    tok_pad: Vec<u32>,
    tgt_pad: Vec<u32>,
    bucket_seq: usize,
    effective_loss_scale: f32,
}

#[cfg(feature = "mlx")]
fn prepare_training_samples(
    samples: &[(Vec<i32>, Vec<i32>, f32)],
    bucket_kernels: Option<&BucketKernels>,
    loss_scale: f32,
) -> Vec<PreparedTrainingSample> {
    let mut prepared = Vec::with_capacity(samples.len());
    for (tokens, targets, sample_quality) in samples {
        let tokens_u32: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
        let targets_u32: Vec<u32> = targets.iter().map(|&t| t as u32).collect();
        let quality_scale = sample_quality.max(0.1);
        let effective_loss_scale = loss_scale * quality_scale;

        if let Some(bk) = bucket_kernels {
            let (bucket_seq, _, _) = bk.get(tokens_u32.len());
            prepared.push(PreparedTrainingSample {
                tok_pad: pad_to(&tokens_u32, *bucket_seq),
                tgt_pad: pad_to(&targets_u32, *bucket_seq),
                tokens_u32,
                targets_u32,
                bucket_seq: *bucket_seq,
                effective_loss_scale,
            });
        } else {
            prepared.push(PreparedTrainingSample {
                tok_pad: Vec::new(),
                tgt_pad: Vec::new(),
                bucket_seq: tokens_u32.len(),
                tokens_u32,
                targets_u32,
                effective_loss_scale,
            });
        }
    }
    prepared
}

#[cfg(feature = "mlx")]
fn lora_storage_paths(model_dir: &std::path::Path) -> (std::path::PathBuf, std::path::PathBuf) {
    let lora_dir = dirs::home_dir()
        .unwrap_or_default()
        .join(".nanobot/workspace/lora");
    let model_key = model_dir
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "default".into());
    let lora_path = lora_dir.join(format!("{model_key}.bin"));
    (lora_dir, lora_path)
}

#[cfg(feature = "mlx")]
fn load_or_init_lora(
    cfg: &AneTrainingConfig,
    n_layers: usize,
    lora_path: &std::path::Path,
) -> super::ane_lora::LoraModel {
    use super::ane_lora::{load_lora_bin, LoraConfig, LoraModel};

    let dim = cfg.mil_config.dim;
    let hidden_dim = cfg.mil_config.hidden_dim;

    if lora_path.exists() {
        match load_lora_bin(lora_path) {
            Ok(l) if l.layers.len() == n_layers => {
                tracing::info!("ANE train: restored LoRA from {}", lora_path.display());
                return l;
            }
            _ => {
                tracing::warn!("ANE train: stale LoRA file, reinitializing");
            }
        }
    } else {
        let model_key = cfg
            .model_dir
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "default".into());
        tracing::info!("ANE train: new LoRA for model {model_key}");
    }

    LoraModel::with_full_dims(
        LoraConfig::default(),
        n_layers,
        dim,
        cfg.mil_config.kv_dim(),
        cfg.mil_config.attn_dim(),
        cfg.mil_config.q_proj_dim(),
        hidden_dim,
    )
}

#[cfg(feature = "mlx")]
fn compatible_mil_config(lhs: &super::ane_mil::MilConfig, rhs: &super::ane_mil::MilConfig) -> bool {
    lhs.dim == rhs.dim
        && lhs.hidden_dim == rhs.hidden_dim
        && lhs.n_heads == rhs.n_heads
        && lhs.n_kv_heads == rhs.n_kv_heads
        && lhs.rope_theta.to_bits() == rhs.rope_theta.to_bits()
        && lhs.rms_eps.to_bits() == rhs.rms_eps.to_bits()
        && lhs.has_lm_head == rhs.has_lm_head
        && lhs.head_dim_explicit == rhs.head_dim_explicit
        && lhs.linear_attn_indices == rhs.linear_attn_indices
        && lhs.linear_n_heads == rhs.linear_n_heads
        && lhs.linear_head_dim == rhs.linear_head_dim
        && lhs.linear_n_value_heads == rhs.linear_n_value_heads
        && lhs.linear_value_head_dim == rhs.linear_value_head_dim
        && lhs.conv_kernel_size == rhs.conv_kernel_size
        && lhs.attn_output_gate == rhs.attn_output_gate
}

#[cfg(feature = "mlx")]
fn compatible_training_target(lhs: &AneTrainingConfig, rhs: &AneTrainingConfig) -> bool {
    lhs.model_dir == rhs.model_dir
        && compatible_mil_config(&lhs.mil_config, &rhs.mil_config)
        && lhs.linear_attn_indices == rhs.linear_attn_indices
        && lhs.kv_dim == rhs.kv_dim
}

#[cfg(feature = "mlx")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PersistentAneTrainerStats {
    pub model_loads: usize,
    pub bucket_compiles: usize,
    pub completed_runs: usize,
}

#[cfg(feature = "mlx")]
#[derive(Default)]
struct PersistentAneTrainerStatCounters {
    model_loads: std::sync::atomic::AtomicUsize,
    bucket_compiles: std::sync::atomic::AtomicUsize,
    completed_runs: std::sync::atomic::AtomicUsize,
}

#[cfg(feature = "mlx")]
impl PersistentAneTrainerStatCounters {
    fn snapshot(&self) -> PersistentAneTrainerStats {
        use std::sync::atomic::Ordering;

        PersistentAneTrainerStats {
            model_loads: self.model_loads.load(Ordering::Relaxed),
            bucket_compiles: self.bucket_compiles.load(Ordering::Relaxed),
            completed_runs: self.completed_runs.load(Ordering::Relaxed),
        }
    }
}

#[cfg(feature = "mlx")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MuonKernelSignature {
    lr_attn_bits: u32,
    lr_ffn_bits: u32,
}

#[cfg(feature = "mlx")]
impl MuonKernelSignature {
    fn from_cfg(cfg: &AneTrainingConfig) -> Self {
        Self {
            lr_attn_bits: (cfg.lr * cfg.lr_scale_attn).to_bits(),
            lr_ffn_bits: (cfg.lr * cfg.lr_scale_ffn).to_bits(),
        }
    }
}

#[cfg(feature = "mlx")]
struct AneTrainerSession {
    cfg: AneTrainingConfig,
    model: super::ane_weights::DenseCachedModel,
    lora_dir: std::path::PathBuf,
    lora_path: std::path::PathBuf,
    lora: super::ane_lora::LoraModel,
    adam: super::ane_lora::LoraModelAdam,
    muon: Option<super::ane_lora::LoraModelMuon>,
    muon_kernels: Option<super::ane_lora::LoraMuonKernels>,
    muon_kernel_signature: Option<MuonKernelSignature>,
    bucket_kernels: BucketKernels,
    bucket_lora_grad_kernels: Vec<(usize, super::ane_lora::LoraWeightGradKernels)>,
}

#[cfg(feature = "mlx")]
impl AneTrainerSession {
    fn new(
        cfg: &AneTrainingConfig,
        stats: &PersistentAneTrainerStatCounters,
    ) -> Result<Self, String> {
        use super::ane_lora::LoraModelAdam;
        use super::ane_weights::{DenseCachedModel, QuantizedModelWeights};

        let t0 = std::time::Instant::now();
        let model =
            match QuantizedModelWeights::from_mlx_safetensors(&cfg.model_dir, &cfg.mil_config) {
                Ok(m) => {
                    let q_mb = m.quantized_memory_bytes() as f64 / 1_048_576.0;
                    let cached = DenseCachedModel::auto(m);
                    tracing::info!(
                        "ANE train: loaded model in {}ms (quantized {q_mb:.1} MB, {}/{} layers cached)",
                        t0.elapsed().as_millis(),
                        cached.cached_layer_count(),
                        cached.n_layers(),
                    );
                    cached
                }
                Err(e) => return Err(format!("failed to load weights: {e}")),
            };
        use std::sync::atomic::Ordering;
        stats.model_loads.fetch_add(1, Ordering::Relaxed);

        let (lora_dir, lora_path) = lora_storage_paths(&cfg.model_dir);
        let lora = load_or_init_lora(cfg, model.n_layers(), &lora_path);
        let adam = LoraModelAdam::zeros(&lora);

        Ok(Self {
            cfg: cfg.clone(),
            model,
            lora_dir,
            lora_path,
            lora,
            adam,
            muon: None,
            muon_kernels: None,
            muon_kernel_signature: None,
            bucket_kernels: BucketKernels::empty(),
            bucket_lora_grad_kernels: Vec::new(),
        })
    }

    fn matches_config(&self, cfg: &AneTrainingConfig) -> bool {
        compatible_training_target(&self.cfg, cfg)
    }

    fn ensure_bucket_kernels(
        &mut self,
        sample_lens: &[usize],
        stats: &PersistentAneTrainerStatCounters,
    ) -> Result<(), String> {
        let compiled = self.bucket_kernels.ensure(sample_lens, &self.cfg.mil_config)?;
        if compiled > 0 {
            use std::sync::atomic::Ordering;
            stats.bucket_compiles.fetch_add(compiled, Ordering::Relaxed);
            tracing::info!(
                "ANE train: cached {} bucket(s) total after compiling {compiled} new bucket(s)",
                self.bucket_kernels.buckets.len()
            );
        }
        Ok(())
    }

    fn ensure_muon_grad_kernels(&mut self) -> Result<(), String> {
        use super::ane_lora::LoraWeightGradKernels;

        for (bucket_seq, _, _) in &self.bucket_kernels.buckets {
            if self
                .bucket_lora_grad_kernels
                .iter()
                .any(|(seq_len, _)| *seq_len == *bucket_seq)
            {
                continue;
            }
            let mut grad_cfg = self.cfg.mil_config.clone();
            grad_cfg.seq_len = *bucket_seq;
            let kernels = LoraWeightGradKernels::compile(&grad_cfg, &self.lora)?;
            self.bucket_lora_grad_kernels.push((*bucket_seq, kernels));
        }
        self.bucket_lora_grad_kernels
            .sort_by_key(|(seq_len, _)| *seq_len);
        Ok(())
    }

    fn ensure_optimizer_state(&mut self, cfg: &AneTrainingConfig) -> Result<(), String> {
        use super::ane_lora::{LoraModelAdam, LoraModelMuon, LoraMuonKernels};

        match cfg.optimizer {
            AneTrainingOptimizer::AdamW => {
                self.adam = LoraModelAdam::zeros(&self.lora);
            }
            AneTrainingOptimizer::AneMuon => {
                self.muon = Some(LoraModelMuon::zeros(&self.lora));
            }
        }

        if cfg.optimizer == AneTrainingOptimizer::AneMuon {
            let sig = MuonKernelSignature::from_cfg(cfg);
            if self.muon_kernel_signature != Some(sig) {
                self.muon_kernels = Some(LoraMuonKernels::compile(
                    &self.lora,
                    cfg.lr * cfg.lr_scale_attn,
                    cfg.lr * cfg.lr_scale_ffn,
                    0.95,
                    0.01,
                )?);
                self.muon_kernel_signature = Some(sig);
            }
        }

        Ok(())
    }

    fn save_and_publish(
        &self,
        cfg: &AneTrainingConfig,
        mlx_tx: Option<std::sync::mpsc::SyncSender<super::mlx_server::ModelRequest>>,
    ) -> bool {
        use super::ane_lora::save_lora_bin;

        let saved = if let Err(e) = std::fs::create_dir_all(&self.lora_dir) {
            tracing::warn!("ANE train: failed to create lora dir: {e}");
            false
        } else if let Err(e) = save_lora_bin(&self.lora, &self.lora_path) {
            tracing::warn!("ANE train: failed to save LoRA: {e}");
            false
        } else {
            tracing::info!("ANE train: saved LoRA to {}", self.lora_path.display());
            true
        };

        if saved {
            match crate::agent::mlx_lora::ModelConfig::from_config_json(&cfg.model_dir) {
                Some(model_cfg) => {
                    let adapter_dir = cfg.model_dir.join("adapters");
                    match crate::agent::mlx_lora::export_ane_adapters(
                        &self.lora,
                        &model_cfg,
                        Some(&cfg.linear_attn_indices),
                        &adapter_dir,
                    ) {
                        Ok(n) => {
                            tracing::info!(
                                tensors = n,
                                path = %adapter_dir.display(),
                                "ANE train: exported mlx-lm adapters"
                            );
                        }
                        Err(e) => {
                            tracing::warn!("ANE train: failed to export mlx-lm adapters: {e}");
                        }
                    }
                }
                None => {
                    tracing::warn!(
                        "ANE train: failed to parse config.json for adapter export at {}",
                        cfg.model_dir.display()
                    );
                }
            }
        }

        if let Some(ref tx) = mlx_tx {
            let deltas = extract_lora_deltas(
                &self.lora,
                if cfg.linear_attn_indices.is_empty() {
                    None
                } else {
                    Some(&cfg.linear_attn_indices)
                },
            );
            let n_deltas = deltas.layers.len();
            let _ = tx.send(super::mlx_server::ModelRequest::ApplyLoraDeltas {
                deltas,
                reply: None,
            });
            tracing::info!("ANE train: sent {n_deltas} deltas to MLX worker");
        }

        saved
    }

    fn train(
        &mut self,
        cfg: &AneTrainingConfig,
        samples: &[(Vec<i32>, Vec<i32>, f32)],
        mlx_tx: Option<std::sync::mpsc::SyncSender<super::mlx_server::ModelRequest>>,
        stats: &PersistentAneTrainerStatCounters,
    ) -> bool {
        use super::ane_backward;
        use super::ane_forward;
        use super::ane_lora::{
            lora_adam_update, lora_adam_update_split_lr, lora_muon_update_ane,
        };

        let t0 = std::time::Instant::now();
        let sample_lens: Vec<usize> = samples.iter().map(|(tokens, _, _)| tokens.len()).collect();
        let use_ane = match self.ensure_bucket_kernels(&sample_lens, stats) {
            Ok(()) => !self.bucket_kernels.buckets.is_empty(),
            Err(e) => {
                if cfg.strict_ane || cfg.optimizer == AneTrainingOptimizer::AneMuon {
                    tracing::error!("ANE train: kernel compilation failed in strict mode: {e}");
                    return false;
                }
                tracing::warn!("ANE train: kernel compilation failed ({e}), falling back to CPU");
                false
            }
        };

        if let Err(e) = self.ensure_optimizer_state(cfg) {
            tracing::error!("ANE train: optimizer setup failed: {e}");
            return false;
        }

        if cfg.optimizer == AneTrainingOptimizer::AneMuon {
            if !use_ane {
                tracing::error!("ANE train: Muon requires compiled ANE bucket kernels");
                return false;
            }
            if let Err(e) = self.ensure_muon_grad_kernels() {
                tracing::error!("ANE train: failed to compile Muon LoRA grad kernels: {e}");
                return false;
            }
        }

        let n_layers = self.model.n_layers();
        let residual_scale = if cfg.residual_scale > 0.0 {
            cfg.residual_scale
        } else {
            1.0f32 / (2.0f32 * n_layers as f32).sqrt()
        };
        let use_split_lr = cfg.lr_scale_attn != 1.0 || cfg.lr_scale_ffn != 1.0;

        let accum = cfg.accum_steps.max(1);
        let n_samples = samples.len();
        let steps_per_epoch = (n_samples + accum - 1) / accum;
        let total_opt_steps = steps_per_epoch * cfg.epochs;
        let patience = steps_per_epoch * 2;
        let mut opt_step = 0usize;
        let mut best_loss = f32::INFINITY;
        let mut stale_count = 0usize;

        let bucket_kernels = if use_ane {
            Some(&self.bucket_kernels)
        } else {
            None
        };
        let prepared_samples =
            prepare_training_samples(samples, bucket_kernels, cfg.loss_scale);

        tracing::info!(
            "ANE train: {n_samples} samples, {total_opt_steps} optimizer steps (accum={accum}), lr={}, mode={}, optimizer={:?}",
            cfg.lr,
            if use_ane { "ANE" } else { "CPU" },
            cfg.optimizer,
        );

        let bucket_lora_grad_kernels = &self.bucket_lora_grad_kernels;
        let muon_kernels = self.muon_kernels.as_ref();
        let model = &mut self.model;
        let lora = &mut self.lora;
        let adam = &mut self.adam;
        let muon = &mut self.muon;
        let mut best_lora = lora.clone();
        let mut best_adam = if cfg.optimizer == AneTrainingOptimizer::AdamW {
            Some(adam.clone())
        } else {
            None
        };
        let mut best_muon = if cfg.optimizer == AneTrainingOptimizer::AneMuon {
            muon.clone()
        } else {
            None
        };
        let mut grad_accum = super::ane_lora::LoraModelGrads::zeros(lora);

        let mut total_fwd_us = 0u64;
        let mut total_bwd_us = 0u64;
        let mut total_opt_us = 0u64;
        let mut total_clone_us = 0u64;
        let mut sample_count = 0u64;

        'outer: for _epoch in 0..cfg.epochs {
            for chunk in prepared_samples.chunks(accum) {
                grad_accum.zero();
                let mut chunk_loss = 0.0f32;

                for sample in chunk {
                    let t_fwd = std::time::Instant::now();
                    let (fwd, bwd) = if let Some(bk) = bucket_kernels {
                        let (_, fwd_k, bwd_k) = bk.get(sample.tokens_u32.len());
                        model.cfg_mut().seq_len = sample.bucket_seq;

                        match ane_forward::forward_ane_generic(
                            fwd_k,
                            model,
                            Some(lora),
                            &sample.tok_pad,
                            &sample.tgt_pad,
                            cfg.softcap,
                            residual_scale,
                        ) {
                            Ok(fwd) => {
                                let bwd = if cfg.optimizer == AneTrainingOptimizer::AneMuon {
                                    let grad_kernels = bucket_lora_grad_kernels
                                        .iter()
                                        .find(|(seq_len, _)| *seq_len == sample.bucket_seq)
                                        .map(|(_, kernels)| kernels)
                                        .expect("Muon bucket grad kernels should exist");
                                    ane_backward::backward_lora_ane_generic_with_lora_kernels(
                                        bwd_k,
                                        model,
                                        &fwd,
                                        lora,
                                        &sample.tok_pad,
                                        cfg.softcap,
                                        sample.effective_loss_scale,
                                        residual_scale,
                                        Some(grad_kernels),
                                    )
                                } else {
                                    ane_backward::backward_lora_ane_generic(
                                        bwd_k,
                                        model,
                                        &fwd,
                                        lora,
                                        &sample.tok_pad,
                                        cfg.softcap,
                                        sample.effective_loss_scale,
                                        residual_scale,
                                    )
                                };
                                (fwd, bwd)
                            }
                            Err(e) => {
                                if cfg.strict_ane || cfg.optimizer == AneTrainingOptimizer::AneMuon {
                                    tracing::error!("ANE forward failed in strict mode: {e}");
                                    return false;
                                }
                                tracing::warn!("ANE forward failed ({e}), falling back to CPU");
                                model.cfg_mut().seq_len = sample.tokens_u32.len();
                                let fwd = ane_forward::forward_cpu_generic(
                                    model,
                                    Some(lora),
                                    &sample.tokens_u32,
                                    &sample.targets_u32,
                                );
                                let bwd = ane_backward::backward_lora_cpu_generic(
                                    model,
                                    &fwd,
                                    lora,
                                    &sample.tokens_u32,
                                    cfg.softcap,
                                    sample.effective_loss_scale,
                                );
                                (fwd, bwd)
                            }
                        }
                    } else {
                        model.cfg_mut().seq_len = sample.tokens_u32.len();
                        let fwd = ane_forward::forward_cpu_generic(
                            model,
                            Some(lora),
                            &sample.tokens_u32,
                            &sample.targets_u32,
                        );
                        let bwd = ane_backward::backward_lora_cpu_generic(
                            model,
                            &fwd,
                            lora,
                            &sample.tokens_u32,
                            cfg.softcap,
                            sample.effective_loss_scale,
                        );
                        (fwd, bwd)
                    };

                    let fwd_bwd_us = t_fwd.elapsed().as_micros() as u64;
                    total_fwd_us += fwd_bwd_us;
                    sample_count += 1;

                    let loss = fwd.base.loss;
                    if !loss.is_finite() {
                        tracing::warn!("ANE train: NaN/Inf loss at opt_step {opt_step}, stopping");
                        break 'outer;
                    }
                    chunk_loss += loss;
                    grad_accum.add_from(&bwd.lora_grads);
                }

                let chunk_len = chunk.len();
                if chunk_len > 1 {
                    grad_accum.scale(1.0 / chunk_len as f32);
                }
                chunk_loss /= chunk_len as f32;

                opt_step += 1;
                let t_opt = std::time::Instant::now();
                match cfg.optimizer {
                    AneTrainingOptimizer::AdamW => {
                        if use_split_lr {
                            lora_adam_update_split_lr(
                                lora,
                                &grad_accum,
                                adam,
                                opt_step,
                                cfg.lr,
                                cfg.lr_scale_attn,
                                cfg.lr_scale_ffn,
                                0.9,
                                0.999,
                                1e-8,
                                0.01,
                            );
                        } else {
                            lora_adam_update(
                                lora,
                                &grad_accum,
                                adam,
                                opt_step,
                                cfg.lr,
                                0.9,
                                0.999,
                                1e-8,
                                0.01,
                            );
                        }
                    }
                    AneTrainingOptimizer::AneMuon => {
                        let Some(muon_state) = muon.as_mut() else {
                            tracing::error!("ANE train: missing Muon state");
                            return false;
                        };
                        let Some(muon_kernels) = muon_kernels else {
                            tracing::error!("ANE train: missing Muon kernels");
                            return false;
                        };
                        if let Err(e) =
                            lora_muon_update_ane(lora, &grad_accum, muon_state, muon_kernels)
                        {
                            tracing::error!("ANE train: Muon update failed: {e}");
                            return false;
                        }
                    }
                }

                total_opt_us += t_opt.elapsed().as_micros() as u64;

                let t_clone = std::time::Instant::now();
                if chunk_loss < best_loss {
                    best_loss = chunk_loss;
                    best_lora = lora.clone();
                    match cfg.optimizer {
                        AneTrainingOptimizer::AdamW => {
                            best_adam = Some(adam.clone());
                        }
                        AneTrainingOptimizer::AneMuon => {
                            best_muon = muon.clone();
                        }
                    }
                    stale_count = 0;
                } else {
                    stale_count += 1;
                }
                total_clone_us += t_clone.elapsed().as_micros() as u64;

                if stale_count >= patience {
                    tracing::info!("ANE train: early stop at opt_step {opt_step}, loss={chunk_loss:.4}");
                    break 'outer;
                }

                if opt_step % 5 == 0 || opt_step == total_opt_steps {
                    tracing::debug!("ANE train: step {opt_step}/{total_opt_steps}, loss={chunk_loss:.4}");
                }
            }
        }

        *lora = best_lora;
        match cfg.optimizer {
            AneTrainingOptimizer::AdamW => {
                if let Some(best) = best_adam {
                    *adam = best;
                }
            }
            AneTrainingOptimizer::AneMuon => {
                *muon = best_muon;
            }
        }

        let train_ms = t0.elapsed().as_millis();
        if sample_count > 0 {
            let fwd_bwd_ms = total_fwd_us as f64 / sample_count as f64 / 1000.0;
            let opt_ms = total_opt_us as f64 / opt_step.max(1) as f64 / 1000.0;
            let clone_ms = total_clone_us as f64 / opt_step.max(1) as f64 / 1000.0;
            let fwd_bwd_pct = total_fwd_us as f64 / (total_fwd_us + total_opt_us + total_clone_us).max(1) as f64 * 100.0;
            tracing::info!(
                "ANE train profile: {sample_count} samples, fwd+bwd={fwd_bwd_ms:.1}ms/sample ({fwd_bwd_pct:.0}%), opt={opt_ms:.1}ms/step, clone={clone_ms:.1}ms/step, total={train_ms}ms",
            );
            eprintln!(
                "ANE train profile: {sample_count} samples, fwd+bwd={fwd_bwd_ms:.1}ms/sample ({fwd_bwd_pct:.0}%), opt={opt_ms:.1}ms/step, clone={clone_ms:.1}ms/step, total={train_ms}ms",
            );
        }
        tracing::info!("ANE train: done in {train_ms}ms, best_loss={best_loss:.4}");

        self.save_and_publish(cfg, mlx_tx)
    }
}

#[cfg(feature = "mlx")]
enum PersistentTrainerCommand {
    Train {
        cfg: AneTrainingConfig,
        samples: Vec<(Vec<i32>, Vec<i32>, f32)>,
        mlx_tx: Option<std::sync::mpsc::SyncSender<super::mlx_server::ModelRequest>>,
        reply: std::sync::mpsc::SyncSender<bool>,
    },
}

#[cfg(feature = "mlx")]
fn persistent_trainer_worker(
    rx: std::sync::mpsc::Receiver<PersistentTrainerCommand>,
    stats: std::sync::Arc<PersistentAneTrainerStatCounters>,
) {
    #[cfg(target_os = "macos")]
    unsafe {
        libc::setpriority(libc::PRIO_PROCESS, 0, 10);
    }

    let mut session: Option<AneTrainerSession> = None;

    while let Ok(cmd) = rx.recv() {
        match cmd {
            PersistentTrainerCommand::Train {
                cfg,
                samples,
                mlx_tx,
                reply,
            } => {
                let needs_reload = session
                    .as_ref()
                    .map(|existing| !existing.matches_config(&cfg))
                    .unwrap_or(true);
                if needs_reload {
                    session = match AneTrainerSession::new(&cfg, stats.as_ref()) {
                        Ok(new_session) => Some(new_session),
                        Err(e) => {
                            tracing::error!("ANE train: {e}");
                            let _ = reply.send(false);
                            continue;
                        }
                    };
                }

                let ok = session
                    .as_mut()
                    .map(|existing| existing.train(&cfg, &samples, mlx_tx, stats.as_ref()))
                    .unwrap_or(false);
                if ok {
                    use std::sync::atomic::Ordering;
                    stats.completed_runs.fetch_add(1, Ordering::Relaxed);
                }
                let _ = reply.send(ok);
            }
        }
    }
}

#[cfg(feature = "mlx")]
#[derive(Clone)]
pub struct PersistentAneTrainer {
    tx: std::sync::mpsc::SyncSender<PersistentTrainerCommand>,
    stats: std::sync::Arc<PersistentAneTrainerStatCounters>,
}

#[cfg(feature = "mlx")]
impl Default for PersistentAneTrainer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "mlx")]
impl PersistentAneTrainer {
    pub fn new() -> Self {
        let (tx, rx) = std::sync::mpsc::sync_channel(2);
        let stats = std::sync::Arc::new(PersistentAneTrainerStatCounters::default());
        let worker_stats = stats.clone();
        std::thread::Builder::new()
            .name("ane-lora-train-worker".into())
            .spawn(move || persistent_trainer_worker(rx, worker_stats))
            .expect("failed to spawn persistent ANE training worker");
        Self { tx, stats }
    }

    pub fn spawn_training(
        &self,
        cfg: AneTrainingConfig,
        samples: Vec<(Vec<i32>, Vec<i32>, f32)>,
        mlx_tx: Option<std::sync::mpsc::SyncSender<super::mlx_server::ModelRequest>>,
    ) -> std::thread::JoinHandle<bool> {
        let tx = self.tx.clone();
        std::thread::Builder::new()
            .name("ane-lora-train".into())
            .spawn(move || {
                let (reply_tx, reply_rx) = std::sync::mpsc::sync_channel(1);
                let sent = tx.send(PersistentTrainerCommand::Train {
                    cfg,
                    samples,
                    mlx_tx,
                    reply: reply_tx,
                });
                if sent.is_err() {
                    tracing::error!("ANE train: persistent worker is unavailable");
                    return false;
                }
                reply_rx.recv().unwrap_or(false)
            })
            .expect("failed to spawn ANE training wait thread")
    }

    pub fn stats(&self) -> PersistentAneTrainerStats {
        self.stats.snapshot()
    }
}

/// Spawn a dedicated thread that trains LoRA on CPU/ANE, then sends
/// the resulting deltas to the MLX model worker via `ApplyLoraDeltas`.
///
/// The thread:
/// 1. Loads base model weights from safetensors (CPU, quantized)
/// 2. Initializes or restores LoRA from `~/.nanobot/workspace/lora/{model_key}.bin`
/// 3. Trains for `epochs` steps with AdamW
/// 4. Saves updated LoRA to disk (warm start for next run)
/// 5. Sends `ApplyLoraDeltas` to MLX model worker
///
/// Returns `JoinHandle<bool>` — `true` if training completed and LoRA was saved.
///
/// Each sample is `(tokens, targets, quality_weight)` where quality_weight
/// scales the loss gradient (0.0–1.0). Higher quality samples contribute more.
#[cfg(feature = "mlx")]
pub fn spawn_ane_training(
    cfg: AneTrainingConfig,
    samples: Vec<(Vec<i32>, Vec<i32>, f32)>,
    mlx_tx: Option<std::sync::mpsc::SyncSender<super::mlx_server::ModelRequest>>,
) -> std::thread::JoinHandle<bool> {
    std::thread::Builder::new()
        .name("ane-lora-train".into())
        .spawn(move || {
            let stats = PersistentAneTrainerStatCounters::default();
            let mut session = match AneTrainerSession::new(&cfg, &stats) {
                Ok(session) => session,
                Err(e) => {
                    tracing::error!("ANE train: {e}");
                    return false;
                }
            };
            session.train(&cfg, &samples, mlx_tx, &stats)
        })
        .expect("failed to spawn ANE training thread")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::ane_lora::{LoraAdapter, LoraConfig, LoraLayerAdapters, LoraModel};

    fn make_test_lora(n_layers: usize, dim: usize, hidden: usize) -> LoraModel {
        let rank = 4;
        LoraModel {
            layers: (0..n_layers)
                .map(|_| LoraLayerAdapters {
                    wq: Some(LoraAdapter::new(rank, dim, dim)),
                    wv: Some(LoraAdapter::new(rank, dim, dim)),
                    wo: Some(LoraAdapter::new(rank, dim, dim)),
                    w2: Some(LoraAdapter::new(rank, hidden, dim)),
                })
                .collect(),
            config: LoraConfig {
                rank,
                alpha: 4.0,
                target_modules: vec!["wq".into(), "wv".into(), "wo".into(), "w2".into()],
            },
        }
    }

    #[test]
    fn test_extract_all_layers() {
        let lora = make_test_lora(3, 64, 128);
        let deltas = extract_lora_deltas(&lora, None);

        // 3 layers × 4 targets = 12 deltas
        assert_eq!(deltas.layers.len(), 12);
        assert_eq!(deltas.scale, 1.0); // alpha/rank = 4/4

        // Check shapes
        for d in &deltas.layers {
            match d.target {
                LoraTarget::QProj | LoraTarget::VProj | LoraTarget::OProj => {
                    assert_eq!(d.delta.d_in, 64);
                    assert_eq!(d.delta.d_out, 64);
                    assert_eq!(d.delta.rank, 4);
                    assert_eq!(d.delta.a.len(), 4 * 64);
                    assert_eq!(d.delta.b.len(), 64 * 4);
                }
                LoraTarget::DownProj => {
                    assert_eq!(d.delta.d_in, 128);
                    assert_eq!(d.delta.d_out, 64);
                    assert_eq!(d.delta.rank, 4);
                    assert_eq!(d.delta.a.len(), 4 * 128);
                    assert_eq!(d.delta.b.len(), 64 * 4);
                }
            }
        }
    }

    #[test]
    fn test_extract_skips_gdn_attention_lora() {
        let lora = make_test_lora(4, 64, 128);
        // Layers 0, 1, 2 are GDN (linear_attn), layer 3 is full attention
        let linear_indices = vec![0, 1, 2];
        let deltas = extract_lora_deltas(&lora, Some(&linear_indices));

        // GDN layers: only down_proj (3 layers × 1)
        // Full attn layer: all 4 targets (1 layer × 4)
        // Total: 3 + 4 = 7
        assert_eq!(deltas.layers.len(), 7);

        // Verify no attention LoRA for GDN layers
        for d in &deltas.layers {
            if d.layer_idx < 3 {
                assert_eq!(
                    d.target,
                    LoraTarget::DownProj,
                    "GDN layer {} should only have DownProj, got {:?}",
                    d.layer_idx,
                    d.target
                );
            }
        }
    }

    #[test]
    fn test_weight_data_preserved() {
        let lora = make_test_lora(1, 8, 16);
        let deltas = extract_lora_deltas(&lora, None);

        let q_delta = deltas
            .layers
            .iter()
            .find(|d| d.target == LoraTarget::QProj)
            .unwrap();
        let original = lora.layers[0].wq.as_ref().unwrap();

        assert_eq!(q_delta.delta.a, original.a);
        assert_eq!(q_delta.delta.b, original.b);
    }

    #[test]
    fn test_name_mapping() {
        assert_eq!(LoraTarget::QProj.ane_name(), "wq");
        assert_eq!(LoraTarget::QProj.mlx_name(), "q_proj");
        assert_eq!(LoraTarget::VProj.ane_name(), "wv");
        assert_eq!(LoraTarget::VProj.mlx_name(), "v_proj");
        assert_eq!(LoraTarget::OProj.ane_name(), "wo");
        assert_eq!(LoraTarget::OProj.mlx_name(), "o_proj");
        assert_eq!(LoraTarget::DownProj.ane_name(), "w2");
        assert_eq!(LoraTarget::DownProj.mlx_name(), "down_proj");
    }

    #[cfg(feature = "mlx")]
    #[test]
    fn test_apply_lora_deltas_to_mlx_lora_linear() {
        // Build a LoraLinear, apply known weights via the bridge, verify forward changes.
        use crate::agent::mlx_lora::LoraLinear;
        use mlx_rs::module::Module;
        use mlx_rs::nn::QuantizedLinear;
        use mlx_rs::Array;

        let ql = QuantizedLinear::new(64, 32).unwrap();
        let mut ll = LoraLinear::new(ql, 4, 1.0).unwrap();

        // Forward with zero-init B should equal base
        let x = Array::from_slice(&vec![1.0f32; 64], &[1, 64]);
        let base_out = ll.base.forward(&x).unwrap();
        let before = ll.forward(&x).unwrap();
        let diff_before: f32 = before
            .subtract(&base_out)
            .unwrap()
            .abs()
            .unwrap()
            .sum(None)
            .unwrap()
            .item();
        assert!(
            diff_before < 1e-6,
            "before apply: LoRA should be zero, diff={diff_before}"
        );

        // Set B to non-zero via apply_lora_weights on a mock layer
        // (we test the method directly on LoraLinear instead of through DecoderLayer)
        let ones_a = vec![0.1f32; 4 * 64]; // [rank=4, d_in=64]
        let ones_b = vec![0.1f32; 32 * 4]; // [d_out=32, rank=4]
        *ll.lora_a.weight = Array::from_slice(&ones_a, &[4, 64]);
        *ll.lora_b.weight = Array::from_slice(&ones_b, &[32, 4]);
        ll.set_adapter_active(true);

        let after = ll.forward(&x).unwrap();
        let diff_after: f32 = after
            .subtract(&base_out)
            .unwrap()
            .abs()
            .unwrap()
            .sum(None)
            .unwrap()
            .item();
        assert!(
            diff_after > 0.01,
            "after apply: LoRA should change output, diff={diff_after}"
        );
    }

    #[cfg(feature = "mlx")]
    #[test]
    fn test_prepare_training_samples_bucket_padding_and_quality_floor() {
        use crate::agent::ane_mil::MilConfig;

        if crate::agent::ane_bridge::ane_init().is_err() {
            eprintln!("SKIP: ANE init failed");
            return;
        }

        let cfg = MilConfig {
            dim: 64,
            hidden_dim: 128,
            n_heads: 4,
            seq_len: 64,
            n_kv_heads: 4,
            rope_theta: 10_000.0,
            rms_eps: 1e-6,
            has_lm_head: false,
            head_dim_explicit: 16,
            linear_attn_indices: vec![],
            linear_n_heads: 0,
            linear_head_dim: 0,
            linear_n_value_heads: 0,
            linear_value_head_dim: 0,
            conv_kernel_size: 0,
            attn_output_gate: false,
        };
        let buckets = BucketKernels::compile(&[59], &cfg).expect("bucket compile");
        let samples = vec![(vec![1; 59], vec![2; 59], 0.02)];

        let prepared = prepare_training_samples(&samples, Some(&buckets), 256.0);
        assert_eq!(prepared.len(), 1);
        let sample = &prepared[0];
        assert_eq!(sample.tokens_u32.len(), 59);
        assert_eq!(sample.targets_u32.len(), 59);
        assert_eq!(sample.bucket_seq, 128);
        assert_eq!(sample.tok_pad.len(), 128);
        assert_eq!(sample.tgt_pad.len(), 128);
        assert!(sample.tok_pad[59..].iter().all(|&t| t == 0));
        assert!(sample.tgt_pad[59..].iter().all(|&t| t == 0));
        assert_eq!(sample.effective_loss_scale, 25.6);
    }

    #[cfg(feature = "mlx")]
    #[test]
    fn test_prepare_training_samples_without_buckets_keeps_raw_tokens() {
        let samples = vec![(vec![7, 8, 9], vec![8, 9, 10], 0.5)];
        let prepared = prepare_training_samples(&samples, None, 256.0);
        assert_eq!(prepared.len(), 1);
        let sample = &prepared[0];
        assert_eq!(sample.bucket_seq, 3);
        assert!(sample.tok_pad.is_empty());
        assert!(sample.tgt_pad.is_empty());
        assert_eq!(sample.tokens_u32, vec![7, 8, 9]);
        assert_eq!(sample.targets_u32, vec![8, 9, 10]);
        assert_eq!(sample.effective_loss_scale, 128.0);
    }

    // -----------------------------------------------------------------------
    // E2E integration tests (require Qwen3-1.7B weights on disk + mlx feature)
    // -----------------------------------------------------------------------

    #[cfg(feature = "mlx")]
    fn qwen3_1_7b_dir() -> std::path::PathBuf {
        dirs::home_dir()
            .unwrap()
            .join(".cache/lm-studio/models/lmstudio-community/Qwen3-1.7B-MLX-8bit")
    }

    #[cfg(feature = "mlx")]
    fn skip_if_no_qwen3() -> bool {
        !qwen3_1_7b_dir().join("tokenizer.json").exists()
    }

    /// E2E: ANE trains LoRA, extracts deltas, applies to MLX model,
    /// verifies forward pass output changes.
    #[cfg(feature = "mlx")]
    #[test]
    fn test_e2e_ane_train_apply_to_mlx() {
        if skip_if_no_qwen3() {
            eprintln!("SKIP: Qwen3-1.7B not found");
            return;
        }

        use crate::agent::ane_backward;
        use crate::agent::ane_forward;
        use crate::agent::ane_lora::{
            lora_adam_update, LoraConfig as AneLoraConfig, LoraModelAdam,
        };
        use crate::agent::ane_mil::MilConfig;
        use crate::agent::ane_weights::ModelWeights;
        use crate::agent::mlx_lora::{LoraConfig as MlxLoraConfig, MlxLoraModel, ModelConfig};

        let model_dir = qwen3_1_7b_dir();
        let mil_cfg = MilConfig {
            dim: 2048,
            hidden_dim: 6144,
            n_heads: 16,
            seq_len: 4,
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

        // 1. Load ANE base weights and train LoRA (3 steps)
        eprintln!("loading ANE base model...");
        let ane_model = ModelWeights::from_mlx_safetensors(&model_dir, &mil_cfg)
            .expect("ANE model load failed");
        let n_layers = ane_model.layers.len();

        let ane_lora_cfg = AneLoraConfig::default();
        let mut ane_lora = LoraModel::with_kv_dim(ane_lora_cfg, n_layers, 2048, 1024, 6144);
        let mut adam = LoraModelAdam::zeros(&ane_lora);

        let tokens: Vec<u32> = (100..104).collect();
        let targets: Vec<u32> = (101..105).collect();

        eprintln!("ANE training 3 steps...");
        for step in 0..3 {
            let fwd = ane_forward::forward_cpu(&ane_model, Some(&ane_lora), &tokens, &targets);
            let bwd = ane_backward::backward_lora_cpu(&ane_model, &fwd, &ane_lora, &tokens);
            lora_adam_update(
                &mut ane_lora,
                &bwd.lora_grads,
                &mut adam,
                step + 1,
                5e-4,
                0.9,
                0.999,
                1e-8,
                0.01,
            );
            eprintln!("  step {step}: loss={:.4}", fwd.base.loss);
        }

        // 2. Extract deltas
        let deltas = extract_lora_deltas(&ane_lora, None);
        eprintln!("extracted {} deltas", deltas.layers.len());
        assert_eq!(deltas.layers.len(), n_layers * 4); // 28 layers × 4 targets

        // 3. Load MLX model, capture output before applying deltas
        eprintln!("loading MLX model...");
        let mlx_cfg = ModelConfig::qwen3_1_7b();
        let mlx_lora_cfg = MlxLoraConfig::default();
        let mut mlx_model =
            MlxLoraModel::load(&model_dir, &mlx_cfg, &mlx_lora_cfg).expect("MLX model load failed");

        let input = mlx_rs::Array::from_slice(&[100i32, 101, 102, 103], &[1, 4]);
        let logits_before = mlx_model.forward_logits(&input).expect("forward failed");
        let before_sum: f32 = logits_before.sum(None).unwrap().item();

        // 4. Apply ANE deltas to MLX model
        let applied = apply_lora_deltas(&mut mlx_model, &deltas).expect("apply_lora_deltas failed");
        eprintln!("applied {applied} deltas to MLX model");
        assert_eq!(applied, n_layers * 4);

        // 5. Verify forward output changed
        let logits_after = mlx_model.forward_logits(&input).expect("forward failed");
        let after_sum: f32 = logits_after.sum(None).unwrap().item();

        let diff = (after_sum - before_sum).abs();
        eprintln!("logits sum: before={before_sum:.4}, after={after_sum:.4}, diff={diff:.4}");
        assert!(
            diff > 0.01,
            "MLX output should change after applying ANE deltas, diff={diff}"
        );
    }

    /// E2E: spawn ANE training thread, verify it sends deltas to the model worker.
    /// Uses a mock receiver to capture the ApplyLoraDeltas message.
    #[cfg(feature = "mlx")]
    #[test]
    fn test_e2e_spawn_ane_training_sends_deltas() {
        if skip_if_no_qwen3() {
            eprintln!("SKIP: Qwen3-1.7B not found");
            return;
        }

        use crate::agent::ane_mil::MilConfig;
        use crate::agent::mlx_server::ModelRequest;

        let model_dir = qwen3_1_7b_dir();

        // Tokenize a simple sample
        let tokenizer =
            crate::agent::mlx_lora::MlxTokenizer::load(&model_dir).expect("tokenizer load failed");
        let messages = vec![
            crate::agent::mlx_server::ChatMessage {
                role: "user".into(),
                content: "What is 2+2?".into(),
            },
            crate::agent::mlx_server::ChatMessage {
                role: "assistant".into(),
                content: "4".into(),
            },
        ];
        let (tokens, targets) =
            crate::agent::mlx_server::tokenize_conversation(&tokenizer, &messages)
                .expect("tokenize failed");
        let seq_len = tokens.len();
        eprintln!("tokenized to {seq_len} tokens");

        // Create a channel to capture the ApplyLoraDeltas message
        let (tx, rx) = std::sync::mpsc::sync_channel::<ModelRequest>(4);

        let cfg = AneTrainingConfig {
            model_dir,
            mil_config: MilConfig {
                dim: 2048,
                hidden_dim: 6144,
                n_heads: 16,
                seq_len,
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
            },
            epochs: 1,
            lr: 1e-5,
            linear_attn_indices: vec![],
            kv_dim: 8 * 128, // n_kv_heads=8, head_dim=128
            softcap: 15.0,
            loss_scale: 256.0,
            lr_scale_attn: 0.05,
            lr_scale_ffn: 1.0,
            residual_scale: 0.0,
            optimizer: AneTrainingOptimizer::AdamW,
            strict_ane: false,
            accum_steps: 1,
        };

        eprintln!("spawning ANE training thread...");
        let handle = spawn_ane_training(cfg, vec![(tokens, targets, 1.0)], Some(tx));

        // Wait for the thread to finish
        handle.join().expect("ANE training thread panicked");

        // Check that we received an ApplyLoraDeltas message
        let msg = rx.try_recv().expect("should have received ApplyLoraDeltas");
        match msg {
            ModelRequest::ApplyLoraDeltas { deltas, reply } => {
                eprintln!(
                    "received ApplyLoraDeltas with {} layer deltas",
                    deltas.layers.len()
                );
                assert!(!deltas.layers.is_empty(), "deltas should not be empty");
                assert!(reply.is_none(), "reply should be None (fire-and-forget)");
                // 28 layers × 4 targets = 112
                assert_eq!(deltas.layers.len(), 28 * 4);
            }
            _ => panic!("expected ApplyLoraDeltas, got different ModelRequest"),
        }
    }

    /// Verify no contention: MLX inference completes normally while ANE trains.
    /// Measures that inference latency is not blocked by ANE training.
    #[cfg(feature = "mlx")]
    #[test]
    fn test_e2e_no_contention_ane_mlx() {
        if skip_if_no_qwen3() {
            eprintln!("SKIP: Qwen3-1.7B not found");
            return;
        }

        use crate::agent::ane_mil::MilConfig;
        use crate::agent::mlx_lora::{LoraConfig as MlxLoraConfig, ModelConfig};
        use crate::agent::mlx_server::{ModelRequest, TrainState};
        use std::sync::Arc;

        let model_dir = qwen3_1_7b_dir();

        // Start MLX model worker
        let train_state = Arc::new(TrainState::new());
        let (tx, rx) = std::sync::mpsc::sync_channel::<ModelRequest>(8);

        let dir = model_dir.clone();
        let ts = train_state.clone();
        std::thread::Builder::new()
            .name("mlx-test-worker".into())
            .spawn(move || {
                let cfg = ModelConfig::qwen3_1_7b();
                let lora_cfg = MlxLoraConfig {
                    lr: 1e-5,
                    ..MlxLoraConfig::default()
                };
                crate::agent::mlx_server::run_model_worker(dir, cfg, lora_cfg, ts, rx, None);
            })
            .expect("failed to spawn model worker");

        // Baseline: inference without ANE training
        let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
        tx.send(ModelRequest::Chat {
            prompt: "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n".into(),
            max_tokens: 8,
            temperature: 0.0,
            reply: reply_tx,
        })
        .expect("send failed");
        let t0 = std::time::Instant::now();
        let baseline = reply_rx.blocking_recv().expect("recv failed");
        let baseline_ms = t0.elapsed().as_millis();
        assert!(
            baseline.is_ok(),
            "baseline inference failed: {:?}",
            baseline
        );
        eprintln!("baseline inference: {}ms", baseline_ms);

        // Start ANE training thread (runs concurrently)
        let tokenizer =
            crate::agent::mlx_lora::MlxTokenizer::load(&model_dir).expect("tokenizer load failed");
        let messages = vec![
            crate::agent::mlx_server::ChatMessage {
                role: "user".into(),
                content: "Capital of France?".into(),
            },
            crate::agent::mlx_server::ChatMessage {
                role: "assistant".into(),
                content: "Paris".into(),
            },
        ];
        let (tokens, targets) =
            crate::agent::mlx_server::tokenize_conversation(&tokenizer, &messages)
                .expect("tokenize failed");

        // ANE training sends ApplyLoraDeltas to same tx — won't block inference
        let ane_tx = tx.clone();
        let ane_cfg = AneTrainingConfig {
            model_dir: model_dir.clone(),
            mil_config: MilConfig {
                dim: 2048,
                hidden_dim: 6144,
                n_heads: 16,
                seq_len: 64,
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
            },
            epochs: 2,
            lr: 1e-5,
            linear_attn_indices: vec![],
            kv_dim: 8 * 128,
            softcap: 15.0,
            loss_scale: 256.0,
            lr_scale_attn: 0.05,
            lr_scale_ffn: 1.0,
            residual_scale: 0.0,
            optimizer: AneTrainingOptimizer::AdamW,
            strict_ane: false,
            accum_steps: 1,
        };
        let _ane_handle = spawn_ane_training(ane_cfg, vec![(tokens, targets, 1.0)], Some(ane_tx));

        // Inference DURING ANE training — should not be blocked
        let (reply_tx2, reply_rx2) = tokio::sync::oneshot::channel();
        tx.send(ModelRequest::Chat {
            prompt: "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n".into(),
            max_tokens: 8,
            temperature: 0.0,
            reply: reply_tx2,
        })
        .expect("send failed");
        let t1 = std::time::Instant::now();
        let during = reply_rx2.blocking_recv().expect("recv failed");
        let during_ms = t1.elapsed().as_millis();
        assert!(
            during.is_ok(),
            "inference during ANE training failed: {:?}",
            during
        );
        eprintln!("inference during ANE training: {}ms", during_ms);

        // The inference during training should complete in reasonable time.
        // ANE trains on a separate thread (CPU), MLX infers on GPU — zero contention.
        // Allow 10x baseline as generous margin (accounts for model loading overhead).
        let max_allowed = baseline_ms.max(2000) * 10;
        assert!(
            during_ms < max_allowed,
            "inference during ANE training too slow: {}ms (baseline {}ms, max {}ms)",
            during_ms,
            baseline_ms,
            max_allowed
        );

        eprintln!(
            "no contention verified: baseline={}ms, during_training={}ms",
            baseline_ms, during_ms
        );
    }

    // -----------------------------------------------------------------------
    // oMLX standalone path tests (ANE trains without in-process MLX)
    // -----------------------------------------------------------------------

    /// Qwen3.5 test model: 0.8B is small enough to load in test but has the
    /// same hybrid architecture as the 35B (GDN layers, attn_output_gate, GQA).
    fn qwen3_5_dir() -> std::path::PathBuf {
        dirs::home_dir()
            .unwrap()
            .join(".cache/lm-studio/models/mlx-community/Qwen3.5-0.8B-8bit")
    }

    fn skip_if_no_qwen3_5() -> bool {
        !qwen3_5_dir().join("tokenizer.json").exists()
    }

    fn qwen3_5_alias_dir(
        tag: &str,
    ) -> (tempfile::TempDir, std::path::PathBuf, std::path::PathBuf) {
        use std::os::unix::fs::symlink;

        let tmp = tempfile::TempDir::new().expect("tempdir");
        let unique = tmp.path().file_name().unwrap().to_string_lossy();
        let alias_name = format!(
            "Qwen3.5-0.8B-8bit-{tag}-{}-{unique}",
            std::process::id()
        );
        let alias_dir = tmp.path().join(alias_name);
        symlink(qwen3_5_dir(), &alias_dir).expect("symlink model dir");
        let model_key = alias_dir.file_name().unwrap().to_string_lossy().to_string();
        let lora_path = dirs::home_dir()
            .unwrap()
            .join(".nanobot/workspace/lora")
            .join(format!("{model_key}.bin"));
        (tmp, alias_dir, lora_path)
    }

    fn qwen3_5_overlay_dir(
        tag: &str,
    ) -> (tempfile::TempDir, std::path::PathBuf, std::path::PathBuf) {
        use std::os::unix::fs::symlink;

        let tmp = tempfile::TempDir::new().expect("tempdir");
        let unique = tmp.path().file_name().unwrap().to_string_lossy();
        let overlay_name = format!(
            "Qwen3.5-0.8B-8bit-{tag}-overlay-{}-{unique}",
            std::process::id()
        );
        let overlay_dir = tmp.path().join(overlay_name);
        std::fs::create_dir_all(&overlay_dir).expect("create overlay dir");

        for entry in std::fs::read_dir(qwen3_5_dir()).expect("read model dir") {
            let entry = entry.expect("dir entry");
            let src = entry.path();
            let dst = overlay_dir.join(entry.file_name());
            symlink(&src, &dst).unwrap_or_else(|e| {
                panic!(
                    "symlink {} -> {} failed: {e}",
                    src.display(),
                    dst.display()
                )
            });
        }

        let model_key = overlay_dir.file_name().unwrap().to_string_lossy().to_string();
        let lora_path = dirs::home_dir()
            .unwrap()
            .join(".nanobot/workspace/lora")
            .join(format!("{model_key}.bin"));
        (tmp, overlay_dir, lora_path)
    }

    fn dir_size_bytes(path: &std::path::Path) -> u64 {
        fn walk(path: &std::path::Path) -> u64 {
            let meta = match std::fs::symlink_metadata(path) {
                Ok(m) => m,
                Err(_) => return 0,
            };
            if meta.file_type().is_symlink() {
                return 0;
            }
            if meta.is_file() {
                return meta.len();
            }
            if meta.is_dir() {
                return std::fs::read_dir(path)
                    .ok()
                    .into_iter()
                    .flat_map(|it| it.filter_map(Result::ok))
                    .map(|entry| walk(&entry.path()))
                    .sum();
            }
            0
        }

        walk(path)
    }

    fn current_rss_bytes() -> u64 {
        let pid = std::process::id().to_string();
        let out = std::process::Command::new("ps")
            .args(["-o", "rss=", "-p", &pid])
            .output()
            .expect("ps rss");
        let kb = String::from_utf8_lossy(&out.stdout)
            .trim()
            .parse::<u64>()
            .unwrap_or(0);
        kb * 1024
    }

    fn peak_rss_bytes() -> u64 {
        let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
        let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
        assert_eq!(rc, 0, "getrusage should succeed");
        let usage = unsafe { usage.assume_init() };
        #[cfg(target_os = "macos")]
        {
            usage.ru_maxrss as u64
        }
        #[cfg(not(target_os = "macos"))]
        {
            (usage.ru_maxrss as u64) * 1024
        }
    }

    #[cfg(feature = "ane")]
    fn manual_ane_muon_step_metrics(
        model_dir: &std::path::Path,
        cfg: &AneTrainingConfig,
        tokens: &[u32],
        targets: &[u32],
    ) -> (u128, u128, f32, f32, usize, usize, u64, u64) {
        use crate::agent::ane_backward;
        use crate::agent::ane_forward;
        use crate::agent::ane_lora::{
            lora_muon_update_ane, LoraConfig, LoraModel, LoraModelMuon, LoraMuonKernels,
            LoraWeightGradKernels,
        };
        use crate::agent::ane_weights::{DenseCachedModel, QuantizedModelWeights, WeightSource};

        let sample_len = tokens.len();
        let bucket_kernels =
            BucketKernels::compile(&[sample_len], &cfg.mil_config).expect("bucket compile");
        let (bucket_seq, fwd_k, bwd_k) = bucket_kernels.get(sample_len);

        let mut grad_cfg = cfg.mil_config.clone();
        grad_cfg.seq_len = *bucket_seq;

        let quantized =
            QuantizedModelWeights::from_mlx_safetensors(model_dir, &cfg.mil_config).expect("load");
        let quantized_bytes = quantized.quantized_memory_bytes() as u64;
        let dense_layer_bytes: Vec<u64> = (0..quantized.layers.len())
            .map(|l| quantized.dense_layer_bytes(l) as u64)
            .collect();
        let quantized_mb = quantized_bytes as f64 / 1_048_576.0;
        let mut model = DenseCachedModel::auto(quantized);
        let cached_layers = model.cached_layer_count();
        let dense_cached_bytes: u64 = dense_layer_bytes.iter().take(cached_layers).sum();

        model.cfg_mut().seq_len = *bucket_seq;
        let n_layers = model.n_layers();
        let dim = model.actual_dim();
        let hidden = model.actual_hidden_dim();
        let residual_scale = if cfg.residual_scale > 0.0 {
            cfg.residual_scale
        } else {
            1.0 / (2.0 * n_layers as f32).sqrt()
        };

        let mut lora = LoraModel::with_full_dims(
            LoraConfig::default(),
            n_layers,
            dim,
            cfg.mil_config.kv_dim(),
            cfg.mil_config.attn_dim(),
            cfg.mil_config.q_proj_dim(),
            hidden,
        );
        let mut muon_state = LoraModelMuon::zeros(&lora);
        let grad_kernels =
            LoraWeightGradKernels::compile(&grad_cfg, &lora).expect("grad kernels compile");
        let muon_kernels = LoraMuonKernels::compile(
            &lora,
            cfg.lr * cfg.lr_scale_attn,
            cfg.lr * cfg.lr_scale_ffn,
            0.95,
            0.01,
        )
        .expect("muon kernels compile");

        let tok_pad = pad_to(tokens, *bucket_seq);
        let tgt_pad = pad_to(targets, *bucket_seq);

        let step = |lora: &mut LoraModel, muon_state: &mut LoraModelMuon| -> (u128, f32) {
            let t0 = std::time::Instant::now();
            let fwd = ane_forward::forward_ane_generic(
                fwd_k,
                &model,
                Some(lora),
                &tok_pad,
                &tgt_pad,
                cfg.softcap,
                residual_scale,
            )
            .expect("manual ANE forward should succeed");
            let bwd = ane_backward::backward_lora_ane_generic_with_lora_kernels(
                bwd_k,
                &model,
                &fwd,
                lora,
                &tok_pad,
                cfg.softcap,
                cfg.loss_scale,
                residual_scale,
                Some(&grad_kernels),
            );
            lora_muon_update_ane(lora, &bwd.lora_grads, muon_state, &muon_kernels)
                .expect("manual Muon update should succeed");
            (t0.elapsed().as_millis(), fwd.base.loss)
        };

        let (step1_ms, step1_loss) = step(&mut lora, &mut muon_state);
        let (step2_ms, step2_loss) = step(&mut lora, &mut muon_state);

        eprintln!(
            "ANE in-process baseline: quantized={quantized_mb:.1}MB cached_layers={cached_layers}/{}",
            model.n_layers()
        );

        (
            step1_ms,
            step2_ms,
            step1_loss,
            step2_loss,
            cached_layers,
            model.n_layers(),
            quantized_bytes,
            dense_cached_bytes,
        )
    }

    fn qwen3_5_tokenize_pair(
        model_dir: &std::path::Path,
        user: &str,
        assistant: &str,
    ) -> (Vec<i32>, Vec<i32>) {
        let tokenizer =
            crate::agent::mlx_lora::MlxTokenizer::load(model_dir).expect("tokenizer load");
        let messages = vec![
            crate::agent::mlx_server::ChatMessage {
                role: "user".into(),
                content: user.into(),
            },
            crate::agent::mlx_server::ChatMessage {
                role: "assistant".into(),
                content: assistant.into(),
            },
        ];
        crate::agent::mlx_server::tokenize_conversation(&tokenizer, &messages).expect("tokenize")
    }

    fn qwen3_5_eval_avg_loss(
        model_dir: &std::path::Path,
        lora: Option<&crate::agent::ane_lora::LoraModel>,
        samples: &[(Vec<i32>, Vec<i32>)],
    ) -> f32 {
        use crate::agent::ane_forward;
        use crate::agent::ane_weights::{QuantizedModelWeights, WeightSource};
        use crate::agent::mlx_lora::ModelConfig;

        let mc = ModelConfig::from_config_json(model_dir).expect("model config");
        let mil_cfg = mc.to_mil_config(64);
        let quantized =
            QuantizedModelWeights::from_mlx_safetensors(model_dir, &mil_cfg).expect("load model");
        let mut model = crate::agent::ane_weights::DenseCachedModel::auto(quantized);
        let mut losses = Vec::with_capacity(samples.len());
        for (tokens, targets) in samples {
            let tokens_u32: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
            let targets_u32: Vec<u32> = targets.iter().map(|&t| t as u32).collect();
            model.cfg_mut().seq_len = tokens_u32.len();
            let fwd = ane_forward::forward_cpu_generic(&model, lora, &tokens_u32, &targets_u32);
            losses.push(fwd.base.loss);
        }
        losses.iter().sum::<f32>() / losses.len() as f32
    }

    /// Test 1: `build_ane_training_config` correctly auto-detects Qwen3.5
    /// hybrid architecture: GDN linear attention layers, attn_output_gate,
    /// GQA with head_dim=256. Uses 0.8B as proxy for the 35B MoE variant
    /// (same arch family, different scale).
    #[cfg(feature = "mlx")]
    #[test]
    fn test_build_ane_config_qwen3_5() {
        if skip_if_no_qwen3_5() {
            eprintln!("SKIP: Qwen3.5-0.8B not found");
            return;
        }

        let dir = qwen3_5_dir();
        let cfg = crate::agent::learn_loop::build_ane_training_config(Some(&dir))
            .expect("build_ane_training_config should succeed for Qwen3.5");

        let mil = &cfg.mil_config;
        // Qwen3.5-0.8B: dim=1024, hidden=3584, heads=8, kv_heads=2, head_dim=256
        assert_eq!(mil.dim, 1024, "hidden_size");
        assert_eq!(mil.hidden_dim, 3584, "intermediate_size");
        assert_eq!(mil.n_heads, 8, "num_attention_heads");
        assert_eq!(mil.n_kv_heads, 2, "num_key_value_heads");
        assert_eq!(mil.head_dim_explicit, 256, "head_dim");
        // Qwen3.5 hybrid: mix of linear_attention and full_attention
        assert!(
            !mil.linear_attn_indices.is_empty(),
            "should have linear attention layers"
        );
        assert!(mil.attn_output_gate, "Qwen3.5 uses attn_output_gate");
        // kv_dim = n_kv_heads * head_dim = 2 * 256 = 512
        assert_eq!(cfg.kv_dim, 512, "kv_dim = n_kv_heads * head_dim");
        assert_eq!(
            cfg.linear_attn_indices.len(),
            mil.linear_attn_indices.len(),
            "training config should propagate linear_attn_indices"
        );

        eprintln!(
            "Qwen3.5 config: dim={}, hidden={}, heads={}, kv_heads={}, head_dim={}, \
             linear_layers={}, attn_gate=true",
            mil.dim,
            mil.hidden_dim,
            mil.n_heads,
            mil.n_kv_heads,
            mil.head_dim_explicit,
            mil.linear_attn_indices.len()
        );
    }

    /// Release-only benchmark harness for the current ANE training baseline.
    ///
    /// Prints:
    /// - full model disk footprint
    /// - quantized bytes loaded by the trainer
    /// - LoRA disk footprint
    /// - cold first-run wall time
    /// - repeated-run wall time (same process, no persistent trainer)
    /// - in-process first-step vs second-step timings with preloaded model/kernels
    /// - current/peak RSS deltas
    /// - baseline vs post-train loss delta
    #[cfg(feature = "mlx")]
    #[test]
    #[ignore = "local benchmark; run in release with --features ane,mlx on a machine with Qwen3.5-0.8B"]
    fn bench_qwen3_5_0_8b_ane_training_baseline() {
        if skip_if_no_qwen3_5() {
            eprintln!("SKIP: Qwen3.5-0.8B not found");
            return;
        }

        use crate::agent::ane_lora::load_lora_bin;

        let (_overlay_tmp, overlay_dir, lora_path) = qwen3_5_overlay_dir("ane-bench");
        let prompt = "Answer exactly in one sentence: What is the capital of France?";
        let response = "The capital of France is Paris, and that is the complete answer.";
        let sample = qwen3_5_tokenize_pair(&overlay_dir, prompt, response);
        let eval_samples = vec![sample.clone()];
        let baseline_loss = qwen3_5_eval_avg_loss(&overlay_dir, None, &eval_samples);

        let mut cfg = crate::agent::learn_loop::build_ane_training_config(Some(&overlay_dir))
            .expect("build config");
        cfg.optimizer = AneTrainingOptimizer::AneMuon;
        cfg.strict_ane = true;
        cfg.lr = 2.5e-4;
        cfg.epochs = 20;

        let train_samples = vec![
            (sample.0.clone(), sample.1.clone(), 1.0),
            (sample.0.clone(), sample.1.clone(), 1.0),
            (sample.0.clone(), sample.1.clone(), 1.0),
            (sample.0.clone(), sample.1.clone(), 1.0),
        ];

        let model_disk_bytes = dir_size_bytes(&qwen3_5_dir());
        let rss_before = current_rss_bytes();
        let peak_before = peak_rss_bytes();

        let t0 = std::time::Instant::now();
        let ok = spawn_ane_training(cfg.clone(), train_samples.clone(), None)
            .join()
            .expect("first training thread should not panic");
        let cold_run_ms = t0.elapsed().as_millis();
        assert!(ok, "first training run should succeed");

        assert!(lora_path.exists(), "first training run should save LoRA");
        let lora_bytes = lora_path.metadata().expect("lora metadata").len();
        let adapter_bytes = dir_size_bytes(&overlay_dir.join("adapters"));
        let lora = load_lora_bin(&lora_path).expect("load trained lora");
        let post_loss = qwen3_5_eval_avg_loss(&overlay_dir, Some(&lora), &eval_samples);
        let rss_after_first = current_rss_bytes();
        let peak_after_first = peak_rss_bytes();

        let t1 = std::time::Instant::now();
        let ok2 = spawn_ane_training(cfg.clone(), train_samples, None)
            .join()
            .expect("second training thread should not panic");
        let second_run_ms = t1.elapsed().as_millis();
        assert!(ok2, "second training run should succeed");

        let rss_after_second = current_rss_bytes();
        let peak_after_second = peak_rss_bytes();

        let tokens_u32: Vec<u32> = sample.0.iter().map(|&t| t as u32).collect();
        let targets_u32: Vec<u32> = sample.1.iter().map(|&t| t as u32).collect();
        let (
            step1_ms,
            step2_ms,
            step1_loss,
            step2_loss,
            cached_layers,
            total_layers,
            quantized_bytes,
            dense_cached_bytes,
        ) = manual_ane_muon_step_metrics(&overlay_dir, &cfg, &tokens_u32, &targets_u32);

        let improvement_pct = ((baseline_loss - post_loss) / baseline_loss.max(1e-6)) * 100.0;
        let metrics = serde_json::json!({
            "model_disk_mb": model_disk_bytes as f64 / 1_048_576.0,
            "quantized_loaded_mb": quantized_bytes as f64 / 1_048_576.0,
            "dense_cached_mb_est": dense_cached_bytes as f64 / 1_048_576.0,
            "cached_layers": cached_layers,
            "total_layers": total_layers,
            "lora_disk_mb": lora_bytes as f64 / 1_048_576.0,
            "adapter_disk_mb": adapter_bytes as f64 / 1_048_576.0,
            "cold_run_ms": cold_run_ms,
            "second_run_ms": second_run_ms,
            "preloaded_step1_ms": step1_ms,
            "preloaded_step2_ms": step2_ms,
            "rss_before_mb": rss_before as f64 / 1_048_576.0,
            "rss_after_first_mb": rss_after_first as f64 / 1_048_576.0,
            "rss_after_second_mb": rss_after_second as f64 / 1_048_576.0,
            "peak_rss_delta_first_mb": (peak_after_first.saturating_sub(peak_before)) as f64 / 1_048_576.0,
            "peak_rss_delta_second_mb": (peak_after_second.saturating_sub(peak_before)) as f64 / 1_048_576.0,
            "baseline_loss": baseline_loss,
            "post_loss": post_loss,
            "improvement_pct": improvement_pct,
            "preloaded_step1_loss": step1_loss,
            "preloaded_step2_loss": step2_loss,
        });
        eprintln!("ANE benchmark baseline: {metrics}");

        assert!(
            post_loss < baseline_loss,
            "benchmark training should reduce loss: baseline={baseline_loss:.4}, post={post_loss:.4}"
        );
        assert!(step2_loss.is_finite(), "preloaded second-step loss should be finite");
    }

    /// Regression: the persistent trainer must reuse the loaded 0.8B model
    /// and cached bucket kernels across consecutive runs with the same config.
    #[cfg(feature = "mlx")]
    #[test]
    #[ignore = "local persistent-trainer regression; run with --features ane,mlx on a machine with Qwen3.5-0.8B"]
    fn test_persistent_ane_trainer_reuses_loaded_model() {
        if skip_if_no_qwen3_5() {
            eprintln!("SKIP: Qwen3.5-0.8B not found");
            return;
        }

        let (_alias_tmp, alias_dir, _lora_path) = qwen3_5_alias_dir("persistent-reuse");
        let prompt = "Answer exactly in one sentence: What is the capital of France?";
        let response = "The capital of France is Paris, and that is the complete answer.";
        let sample = qwen3_5_tokenize_pair(&alias_dir, prompt, response);

        let mut cfg = crate::agent::learn_loop::build_ane_training_config(Some(&alias_dir))
            .expect("build config");
        cfg.optimizer = AneTrainingOptimizer::AneMuon;
        cfg.strict_ane = true;
        cfg.lr = 2.5e-4;
        cfg.epochs = 1;

        let trainer = PersistentAneTrainer::new();
        let train_samples = vec![(sample.0.clone(), sample.1.clone(), 1.0)];

        let ok1 = trainer
            .spawn_training(cfg.clone(), train_samples.clone(), None)
            .join()
            .expect("first persistent run should not panic");
        assert!(ok1, "first persistent trainer run should succeed");
        let stats1 = trainer.stats();
        assert_eq!(stats1.model_loads, 1, "first run should load the model once");
        assert!(
            stats1.bucket_compiles >= 1,
            "first run should compile at least one bucket"
        );

        let ok2 = trainer
            .spawn_training(cfg, train_samples, None)
            .join()
            .expect("second persistent run should not panic");
        assert!(ok2, "second persistent trainer run should succeed");
        let stats2 = trainer.stats();
        assert_eq!(
            stats2.model_loads, 1,
            "persistent trainer should reuse the loaded model on the second run"
        );
        assert_eq!(
            stats2.bucket_compiles, stats1.bucket_compiles,
            "persistent trainer should reuse bucket kernels on the second run"
        );
        assert_eq!(
            stats2.completed_runs, 2,
            "persistent trainer should report two completed runs"
        );
    }

    /// Release-only benchmark for the long-lived ANE trainer warm path.
    ///
    /// Prints:
    /// - first vs second persistent-run wall time
    /// - model/bucket reuse counters from the trainer
    /// - baseline vs post-train loss delta
    #[cfg(feature = "mlx")]
    #[test]
    #[ignore = "local benchmark; run in release with --features ane,mlx on a machine with Qwen3.5-0.8B"]
    fn bench_qwen3_5_0_8b_ane_training_persistent() {
        if skip_if_no_qwen3_5() {
            eprintln!("SKIP: Qwen3.5-0.8B not found");
            return;
        }

        use crate::agent::ane_lora::load_lora_bin;

        let (_overlay_tmp, overlay_dir, lora_path) = qwen3_5_overlay_dir("ane-persistent");
        let prompt = "Answer exactly in one sentence: What is the capital of France?";
        let response = "The capital of France is Paris, and that is the complete answer.";
        let sample = qwen3_5_tokenize_pair(&overlay_dir, prompt, response);
        let eval_samples = vec![sample.clone()];
        let baseline_loss = qwen3_5_eval_avg_loss(&overlay_dir, None, &eval_samples);

        let mut cfg = crate::agent::learn_loop::build_ane_training_config(Some(&overlay_dir))
            .expect("build config");
        cfg.optimizer = AneTrainingOptimizer::AneMuon;
        cfg.strict_ane = true;
        cfg.lr = 2.5e-4;
        cfg.epochs = 20;

        let train_samples = vec![
            (sample.0.clone(), sample.1.clone(), 1.0),
            (sample.0.clone(), sample.1.clone(), 1.0),
            (sample.0.clone(), sample.1.clone(), 1.0),
            (sample.0.clone(), sample.1.clone(), 1.0),
        ];

        let trainer = PersistentAneTrainer::new();

        let t0 = std::time::Instant::now();
        let ok1 = trainer
            .spawn_training(cfg.clone(), train_samples.clone(), None)
            .join()
            .expect("first persistent training thread should not panic");
        let first_run_ms = t0.elapsed().as_millis();
        assert!(ok1, "first persistent training run should succeed");

        let t1 = std::time::Instant::now();
        let ok2 = trainer
            .spawn_training(cfg, train_samples, None)
            .join()
            .expect("second persistent training thread should not panic");
        let second_run_ms = t1.elapsed().as_millis();
        assert!(ok2, "second persistent training run should succeed");

        let stats = trainer.stats();
        let lora = load_lora_bin(&lora_path).expect("load trained lora");
        let post_loss = qwen3_5_eval_avg_loss(&overlay_dir, Some(&lora), &eval_samples);
        let improvement_pct = ((baseline_loss - post_loss) / baseline_loss.max(1e-6)) * 100.0;
        let metrics = serde_json::json!({
            "persistent_first_run_ms": first_run_ms,
            "persistent_second_run_ms": second_run_ms,
            "persistent_model_loads": stats.model_loads,
            "persistent_bucket_compiles": stats.bucket_compiles,
            "persistent_completed_runs": stats.completed_runs,
            "baseline_loss": baseline_loss,
            "post_loss": post_loss,
            "improvement_pct": improvement_pct,
        });
        eprintln!("ANE benchmark persistent: {metrics}");

        assert_eq!(stats.model_loads, 1, "persistent trainer should load the model once");
        assert_eq!(
            stats.completed_runs, 2,
            "persistent trainer should report two completed runs"
        );
        assert!(
            second_run_ms < first_run_ms,
            "persistent warm run should be faster than the first run: first={first_run_ms}ms second={second_run_ms}ms"
        );
        assert!(
            post_loss.is_finite(),
            "persistent benchmark should report a finite post-train loss"
        );
    }

    /// Test 1b: BucketKernels compile for Qwen3.5 dims and FFN uses ANE.
    /// SDPA fails (GQA) but is caught by .ok() — training uses ANE for FFN.
    #[cfg(feature = "mlx")]
    #[test]
    fn test_bucket_kernels_compile_qwen3_5() {
        if skip_if_no_qwen3_5() {
            eprintln!("SKIP: Qwen3.5-0.8B not found");
            return;
        }

        let dir = qwen3_5_dir();
        let cfg =
            crate::agent::learn_loop::build_ane_training_config(Some(&dir)).expect("build config");

        // Simulate a single sample of ~20 tokens (fits in 128 bucket)
        let sample_lens = vec![20usize];
        let result = BucketKernels::compile(&sample_lens, &cfg.mil_config);
        assert!(
            result.is_ok(),
            "BucketKernels should compile for Qwen3.5: {:?}",
            result.err()
        );

        let bk = result.unwrap();
        assert!(!bk.buckets.is_empty(), "should have at least 1 bucket");

        let (bucket_seq, fwd, _bwd) = &bk.buckets[0];
        eprintln!("bucket seq={bucket_seq}");
        use crate::agent::ane_forward::FfnKernels;
        eprintln!(
            "  SDPA: {}",
            if fwd.sdpa_fwd.is_some() {
                "ANE"
            } else {
                "CPU (GQA)"
            }
        );
        eprintln!(
            "  FFN:  {}",
            match &fwd.ffn {
                FfnKernels::FullyFused { .. } => "ANE (fully-fused)",
                FfnKernels::Fused { .. } => "ANE (fused)",
                FfnKernels::Tiled { .. } => "ANE (tiled)",
            }
        );

        // The critical assertion: FFN MUST be on ANE
        assert!(
            matches!(
                &fwd.ffn,
                FfnKernels::Fused { .. } | FfnKernels::Tiled { .. }
            ),
            "FFN should compile on ANE"
        );
    }

    /// Test 2: `spawn_ane_training` with `mlx_tx: None` completes and saves
    /// LoRA .bin to disk. This is the oMLX/LM Studio path where there's no
    /// in-process MLX model to hot-swap into.
    #[cfg(feature = "mlx")]
    #[test]
    fn test_ane_standalone_training_no_mlx_tx() {
        if skip_if_no_qwen3_5() {
            eprintln!("SKIP: Qwen3.5-0.8B not found");
            return;
        }

        let dir = qwen3_5_dir();
        let cfg =
            crate::agent::learn_loop::build_ane_training_config(Some(&dir)).expect("build config");

        // Tokenize a sample conversation
        let tokenizer = crate::agent::mlx_lora::MlxTokenizer::load(&dir).expect("tokenizer load");
        let messages = vec![
            crate::agent::mlx_server::ChatMessage {
                role: "user".into(),
                content: "What is the capital of Japan?".into(),
            },
            crate::agent::mlx_server::ChatMessage {
                role: "assistant".into(),
                content: "Tokyo".into(),
            },
        ];
        let (tokens, targets) =
            crate::agent::mlx_server::tokenize_conversation(&tokenizer, &messages)
                .expect("tokenize");
        eprintln!("tokenized to {} tokens", tokens.len());

        // Clean up any existing LoRA for this model so we test fresh creation
        let model_key = dir.file_name().unwrap().to_string_lossy().to_string();
        let lora_dir = dirs::home_dir().unwrap().join(".nanobot/workspace/lora");
        let lora_path = lora_dir.join(format!("{model_key}.bin"));
        let had_existing = lora_path.exists();
        // Don't delete — just check if a new/updated one appears after training

        let modified_before = lora_path.metadata().ok().and_then(|m| m.modified().ok());

        // Spawn training with mlx_tx: None (oMLX standalone path)
        eprintln!("spawning standalone ANE training (no MLX hot-swap)...");
        let handle = spawn_ane_training(cfg, vec![(tokens, targets, 1.0)], None);

        let ok = handle.join().expect("training thread should not panic");
        assert!(ok, "training should complete successfully");

        // Verify LoRA file was saved
        assert!(
            lora_path.exists(),
            "LoRA .bin should exist at {}",
            lora_path.display()
        );
        let modified_after = lora_path.metadata().ok().and_then(|m| m.modified().ok());
        if had_existing {
            assert!(
                modified_after > modified_before,
                "LoRA file should have been updated"
            );
        }
        eprintln!(
            "standalone training complete, LoRA saved to {}",
            lora_path.display()
        );
    }

    /// Offline proof: real Qwen3.5-0.8B ANE Muon training run lowers loss on
    /// its own training sample using the full `spawn_ane_training` path.
    #[cfg(feature = "mlx")]
    #[test]
    fn test_ane_muon_offline_qwen3_5_0_8b_reduces_loss() {
        if skip_if_no_qwen3_5() {
            eprintln!("SKIP: Qwen3.5-0.8B not found");
            return;
        }

        use crate::agent::ane_lora::load_lora_bin;

        let (_alias_tmp, alias_dir, lora_path) = qwen3_5_alias_dir("ane-muon-offline");
        let prompt = "Answer exactly in one sentence: What is the capital of France?";
        let response = "The capital of France is Paris, and that is the complete answer.";
        let sample = qwen3_5_tokenize_pair(&alias_dir, prompt, response);
        let eval_samples = vec![sample.clone()];
        let baseline = qwen3_5_eval_avg_loss(&alias_dir, None, &eval_samples);

        let mut cfg = crate::agent::learn_loop::build_ane_training_config(Some(&alias_dir))
            .expect("build config");
        cfg.optimizer = AneTrainingOptimizer::AneMuon;
        cfg.strict_ane = true;
        cfg.lr = 2.5e-4;
        cfg.epochs = 20;

        let train_samples = vec![
            (sample.0.clone(), sample.1.clone(), 1.0),
            (sample.0.clone(), sample.1.clone(), 1.0),
            (sample.0.clone(), sample.1.clone(), 1.0),
            (sample.0.clone(), sample.1.clone(), 1.0),
        ];

        let handle = spawn_ane_training(cfg, train_samples, None);
        let ok = handle.join().expect("training thread should not panic");
        assert!(ok, "ANE Muon offline training should succeed");
        assert!(
            lora_path.exists(),
            "ANE Muon offline training should save LoRA at {}",
            lora_path.display()
        );

        let lora = load_lora_bin(&lora_path).expect("load trained lora");
        let post = qwen3_5_eval_avg_loss(&alias_dir, Some(&lora), &eval_samples);
        let improvement = (baseline - post) / baseline.max(1e-6);
        eprintln!(
            "ANE Muon offline: baseline_loss={baseline:.4} post_loss={post:.4} improvement={:.2}%",
            improvement * 100.0
        );
        assert!(
            post < baseline,
            "ANE Muon offline loss should decrease: baseline={baseline:.4}, post={post:.4}"
        );
        assert!(
            improvement > 0.0,
            "ANE Muon offline improvement should be positive: got {:.2}%",
            improvement * 100.0
        );
    }

    /// Test 3: The `observe_async` learn loop path fires ANE training when
    /// `ane_model_dir` is set and `mlx_provider` is None (oMLX/LM Studio mode).
    /// Verifies the full wiring: experience recorded → threshold exceeded →
    /// ANE training spawned → completes → experience marked exported.
    #[cfg(feature = "mlx")]
    #[tokio::test]
    async fn test_learn_loop_ane_model_dir_path() {
        if skip_if_no_qwen3_5() {
            eprintln!("SKIP: Qwen3.5-0.8B not found");
            return;
        }

        use crate::agent::agent_core::RuntimeCounters;
        use crate::agent::learn_loop::{DefaultLearnLoop, LearnLoop, TurnOutcome};
        use crate::agent::lora_bridge::ExperienceBuffer;
        use crate::config::schema::PerplexityGateConfig;
        use std::sync::atomic::Ordering;
        use std::sync::Arc;

        let dir = qwen3_5_dir();

        // Use a temp DB for the experience buffer so we don't pollute production
        let tmp = tempfile::NamedTempFile::new().expect("tempfile");
        let eb = ExperienceBuffer::open(tmp.path()).expect("open eb");
        let eb_arc = Arc::new(parking_lot::Mutex::new(eb));

        // No pre-seeding: observe_async records its own experience from the
        // TurnOutcome, so with min_experiences=1 it will trigger training on
        // exactly that 1 experience. After training, unexported should be 0.

        let counters = Arc::new(RuntimeCounters::new(128_000));

        let ll = DefaultLearnLoop {
            calibrator: None,
            experience_buffer: Some(eb_arc.clone()),
            perplexity_gate_config: PerplexityGateConfig {
                enabled: true,
                surprise_threshold: 0.1, // low threshold → easy to trigger
                min_experiences: 1,
                train_epochs: 1,
                mlx_server_url: String::new(),
            },
            #[cfg(feature = "mlx")]
            mlx_provider: None, // No in-process MLX — oMLX mode
            training_counters: Some(counters.clone()),
            ane_model_dir: Some(dir.clone()), // THIS is what we're testing
            #[cfg(all(feature = "ane", feature = "mlx"))]
            ane_trainer: Some(Arc::new(PersistentAneTrainer::new())),
            #[cfg(all(feature = "ane", feature = "mlx"))]
            ane_optimizer_override: None,
            #[cfg(all(feature = "ane", feature = "mlx"))]
            ane_lr_override: None,
            #[cfg(all(feature = "ane", feature = "mlx"))]
            ane_strict_ane: false,
        };

        // Build a TurnOutcome with high surprise content
        let outcome = TurnOutcome {
            user_content: "Explain quantum entanglement in detail with examples".into(),
            final_content: "Quantum entanglement is a phenomenon...".into(),
            reasoning_trace: None,
            turn_messages: vec![],
            model: "local:Qwen3.5-35B".into(),
            session_key: "test-session".into(),
            workspace: std::path::PathBuf::from("/tmp"),
            used_tools: {
                let mut s = std::collections::HashSet::new();
                s.insert("exec_command".into());
                s.insert("web_search".into());
                s
            },
            turn_tool_entries: vec![
                crate::agent::audit::TurnToolEntry {
                    name: "exec_command".into(),
                    id: "call_1".into(),
                    ok: true,
                    duration_ms: 200,
                    result_chars: 500,
                },
                crate::agent::audit::TurnToolEntry {
                    name: "web_search".into(),
                    id: "call_2".into(),
                    ok: true,
                    duration_ms: 300,
                    result_chars: 1000,
                },
            ],
            iterations_used: 3,
            max_iterations: 10,
            turn_count: 1,
            turn_start_elapsed_ms: 2000,
            context_tokens: 5000,
            message_count: 5,
            working_memory_tokens: 100,
            provenance_audit_enabled: false,
            is_local: true,
            cost_usd: 0.0,
            prompt_tokens: 3000,
            completion_tokens: 500,
        };

        // Fire observe_async — should spawn ANE training
        let handle: Option<tokio::task::JoinHandle<()>> = ll.observe_async(outcome);
        assert!(
            handle.is_some(),
            "observe_async should return a JoinHandle (training spawned)"
        );

        // Wait for the async task (which internally waits for the ANE thread)
        handle.unwrap().await.expect("async task should not panic");

        // Verify training ran: training_active should be false (done)
        assert!(
            !counters.training_active.load(Ordering::Relaxed),
            "training_active should be false after completion"
        );
        // training_steps_total should have incremented
        assert!(
            counters.training_steps_total.load(Ordering::Relaxed) >= 1,
            "training_steps_total should be >= 1"
        );

        // Verify experience was marked exported
        {
            let eb = eb_arc.lock();
            let stats = eb.stats().expect("stats");
            assert_eq!(
                stats.unexported, 0,
                "experience should be marked exported after successful training"
            );
        }

        eprintln!("learn_loop ane_model_dir path verified end-to-end");
    }

    /// Online proof: the live learn-loop path records experiences, triggers ANE
    /// Muon training, and produces a saved LoRA that lowers loss.
    #[cfg(feature = "mlx")]
    #[tokio::test]
    async fn test_learn_loop_ane_muon_reduces_loss_online() {
        if skip_if_no_qwen3_5() {
            eprintln!("SKIP: Qwen3.5-0.8B not found");
            return;
        }

        use crate::agent::agent_core::RuntimeCounters;
        use crate::agent::ane_lora::load_lora_bin;
        use crate::agent::learn_loop::{DefaultLearnLoop, LearnLoop, TurnOutcome};
        use crate::agent::lora_bridge::ExperienceBuffer;
        use crate::config::schema::PerplexityGateConfig;
        use std::sync::atomic::Ordering;
        use std::sync::Arc;

        let (_alias_tmp, alias_dir, lora_path) = qwen3_5_alias_dir("ane-muon-online");
        let prompt = "Answer exactly in one sentence: What is the capital of France?";
        let response = "The capital of France is Paris, and that is the complete answer.";
        let sample = qwen3_5_tokenize_pair(&alias_dir, prompt, response);
        let eval_samples = vec![sample.clone()];
        let baseline = qwen3_5_eval_avg_loss(&alias_dir, None, &eval_samples);

        let tmp_db = tempfile::NamedTempFile::new().expect("tempfile");
        let eb = ExperienceBuffer::open(tmp_db.path()).expect("open eb");
        eb.record(prompt, "[]", response, true, 1.0, "local:Qwen3.5-0.8B")
            .expect("seed exp 1");
        eb.record(prompt, "[]", response, true, 1.0, "local:Qwen3.5-0.8B")
            .expect("seed exp 2");
        eb.record(prompt, "[]", response, true, 1.0, "local:Qwen3.5-0.8B")
            .expect("seed exp 3");
        let eb_arc = Arc::new(parking_lot::Mutex::new(eb));
        let counters = Arc::new(RuntimeCounters::new(128_000));

        let ll = DefaultLearnLoop {
            calibrator: None,
            experience_buffer: Some(eb_arc.clone()),
            perplexity_gate_config: PerplexityGateConfig {
                enabled: true,
                surprise_threshold: 0.0,
                min_experiences: 4,
                train_epochs: 20,
                mlx_server_url: String::new(),
            },
            #[cfg(feature = "mlx")]
            mlx_provider: None,
            training_counters: Some(counters.clone()),
            ane_model_dir: Some(alias_dir.clone()),
            #[cfg(all(feature = "ane", feature = "mlx"))]
            ane_trainer: Some(Arc::new(PersistentAneTrainer::new())),
            #[cfg(all(feature = "ane", feature = "mlx"))]
            ane_optimizer_override: Some(AneTrainingOptimizer::AneMuon),
            #[cfg(all(feature = "ane", feature = "mlx"))]
            ane_lr_override: Some(2.5e-4),
            #[cfg(all(feature = "ane", feature = "mlx"))]
            ane_strict_ane: true,
        };

        let outcome = TurnOutcome {
            user_content: prompt.into(),
            final_content: response.into(),
            reasoning_trace: Some("Deliberate exact-answer turn.".into()),
            turn_messages: vec![],
            model: "local:Qwen3.5-0.8B".into(),
            session_key: "ane-muon-online".into(),
            workspace: std::path::PathBuf::from("/tmp"),
            used_tools: {
                let mut tools = std::collections::HashSet::new();
                tools.insert("read_file".into());
                tools
            },
            turn_tool_entries: vec![crate::agent::audit::TurnToolEntry {
                name: "read_file".into(),
                id: "call_1".into(),
                ok: true,
                duration_ms: 12,
                result_chars: 128,
            }],
            iterations_used: 0,
            max_iterations: 1,
            turn_count: 1,
            turn_start_elapsed_ms: 250,
            context_tokens: 256,
            message_count: 2,
            working_memory_tokens: 0,
            provenance_audit_enabled: false,
            is_local: true,
            cost_usd: 0.0,
            prompt_tokens: 64,
            completion_tokens: 16,
        };

        let handle = ll.observe_async(outcome).expect("learn loop should spawn");
        handle.await.expect("learn loop task should not panic");

        assert!(
            !counters.training_active.load(Ordering::Relaxed),
            "training should be inactive after learn loop completes"
        );
        assert!(
            counters.training_steps_total.load(Ordering::Relaxed) >= 1,
            "learn loop should record at least one completed ANE training run"
        );
        assert!(
            lora_path.exists(),
            "learn loop should save LoRA at {}",
            lora_path.display()
        );
        assert_eq!(
            eb_arc.lock().stats().expect("stats").unexported,
            0,
            "all queued experiences should be marked exported after successful training"
        );

        let lora = load_lora_bin(&lora_path).expect("load trained lora");
        let post = qwen3_5_eval_avg_loss(&alias_dir, Some(&lora), &eval_samples);
        let improvement = (baseline - post) / baseline.max(1e-6);
        eprintln!(
            "ANE Muon online: baseline_loss={baseline:.4} post_loss={post:.4} improvement={:.2}%",
            improvement * 100.0
        );
        assert!(
            post < baseline,
            "ANE Muon online loss should decrease: baseline={baseline:.4}, post={post:.4}"
        );
        assert!(
            improvement > 0.0,
            "ANE Muon online improvement should be positive: got {:.2}%",
            improvement * 100.0
        );
    }

    /// Test 4: Before/after eval — train LoRA on a set of samples, then measure
    /// whether perplexity on those samples decreases with the LoRA applied.
    /// This validates the entire pipeline: training produces a LoRA that actually
    /// improves the model's predictions on the training distribution.
    #[cfg(feature = "mlx")]
    #[test]
    fn test_ane_lora_improves_perplexity() {
        if skip_if_no_qwen3_5() {
            eprintln!("SKIP: Qwen3.5-0.8B not found");
            return;
        }

        use crate::agent::ane_forward;
        use crate::agent::ane_lora::{load_lora_bin, save_lora_bin, LoraConfig, LoraModel};
        use crate::agent::ane_weights::{QuantizedModelWeights, WeightSource};
        use crate::agent::mlx_lora::ModelConfig;

        let dir = qwen3_5_dir();
        let mc = ModelConfig::from_config_json(&dir).expect("model config");
        let mil_cfg = mc.to_mil_config(64);

        // Tokenize training samples
        let tokenizer = crate::agent::mlx_lora::MlxTokenizer::load(&dir).expect("tokenizer");
        // Single sample overfit test: if LoRA can't memorize one sample,
        // the gradient computation is wrong.
        let train_convos = vec![(
            "What is the capital of France?",
            "The capital of France is Paris.",
        )];
        let mut samples: Vec<(Vec<i32>, Vec<i32>)> = Vec::new();
        for (user, assistant) in &train_convos {
            let messages = vec![
                crate::agent::mlx_server::ChatMessage {
                    role: "user".into(),
                    content: user.to_string(),
                },
                crate::agent::mlx_server::ChatMessage {
                    role: "assistant".into(),
                    content: assistant.to_string(),
                },
            ];
            if let Ok(pair) = crate::agent::mlx_server::tokenize_conversation(&tokenizer, &messages)
            {
                samples.push(pair);
            }
        }
        assert!(!samples.is_empty(), "should tokenize at least one sample");
        eprintln!("tokenized {} training samples", samples.len());

        // Load quantized model with dense layer cache for speed
        let quantized =
            QuantizedModelWeights::from_mlx_safetensors(&dir, &mil_cfg).expect("load model");
        let mut model = crate::agent::ane_weights::DenseCachedModel::auto(quantized);
        let n_layers = model.n_layers();
        let dim = mil_cfg.dim;
        let hidden = mil_cfg.hidden_dim;

        // 1. Measure BASELINE loss (no LoRA)
        let mut baseline_losses = Vec::new();
        for (tokens, targets) in &samples {
            let tokens_u32: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
            let targets_u32: Vec<u32> = targets.iter().map(|&t| t as u32).collect();
            model.cfg_mut().seq_len = tokens_u32.len();
            let fwd = ane_forward::forward_cpu_generic(&model, None, &tokens_u32, &targets_u32);
            baseline_losses.push(fwd.base.loss);
            eprintln!(
                "  baseline loss (sample {}): {:.4}",
                baseline_losses.len(),
                fwd.base.loss
            );
        }
        let avg_baseline = baseline_losses.iter().sum::<f32>() / baseline_losses.len() as f32;
        eprintln!("average baseline loss: {avg_baseline:.4}");

        // 2. Train LoRA (use spawn_ane_training for realistic end-to-end)
        let cfg = crate::agent::learn_loop::build_ane_training_config(Some(&dir))
            .expect("build ANE config");
        // Use a temporary LoRA path so we don't pollute the workspace
        let tmp_dir = tempfile::TempDir::new().expect("tempdir");
        let lora_path = tmp_dir.path().join("test_eval.bin");

        // Train directly (inline, not via spawn, for control over LoRA path)
        let mut lora = LoraModel::with_full_dims(
            LoraConfig::default(),
            n_layers,
            dim,
            mil_cfg.kv_dim(),
            mil_cfg.attn_dim(),
            mil_cfg.q_proj_dim(),
            hidden,
        );
        let mut adam = crate::agent::ane_lora::LoraModelAdam::zeros(&lora);

        let epochs = 20; // Overfit single sample
        let lr = 1e-4; // Higher LR for single-sample memorization test
        let mut step = 0usize;
        let mut best_loss = f32::INFINITY;
        let mut best_lora = lora.clone();
        for _epoch in 0..epochs {
            for (tokens, targets) in &samples {
                let tokens_u32: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
                let targets_u32: Vec<u32> = targets.iter().map(|&t| t as u32).collect();
                model.cfg_mut().seq_len = tokens_u32.len();

                let fwd = ane_forward::forward_cpu_generic(
                    &model,
                    Some(&lora),
                    &tokens_u32,
                    &targets_u32,
                );
                let bwd = crate::agent::ane_backward::backward_lora_cpu_generic(
                    &model,
                    &fwd,
                    &lora,
                    &tokens_u32,
                    15.0,
                    256.0,
                );
                step += 1;
                crate::agent::ane_lora::lora_adam_update(
                    &mut lora,
                    &bwd.lora_grads,
                    &mut adam,
                    step,
                    lr,
                    0.9,
                    0.999,
                    1e-8,
                    0.01,
                );
                let loss = fwd.base.loss;
                if loss < best_loss {
                    best_loss = loss;
                    best_lora = lora.clone();
                }
                if step % 5 == 0 {
                    eprintln!("  train step {step}, loss={loss:.4} (best={best_loss:.4})");
                }
            }
        }
        lora = best_lora;
        eprintln!("training complete: {step} steps, using best checkpoint (loss={best_loss:.4})");

        // 3. Measure loss WITH LoRA
        let mut lora_losses = Vec::new();
        for (tokens, targets) in &samples {
            let tokens_u32: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
            let targets_u32: Vec<u32> = targets.iter().map(|&t| t as u32).collect();
            model.cfg_mut().seq_len = tokens_u32.len();
            let fwd =
                ane_forward::forward_cpu_generic(&model, Some(&lora), &tokens_u32, &targets_u32);
            lora_losses.push(fwd.base.loss);
            eprintln!(
                "  with-LoRA loss (sample {}): {:.4}",
                lora_losses.len(),
                fwd.base.loss
            );
        }
        let avg_lora = lora_losses.iter().sum::<f32>() / lora_losses.len() as f32;
        eprintln!("average with-LoRA loss: {avg_lora:.4}");

        // 4. Assert improvement
        let improvement = (avg_baseline - avg_lora) / avg_baseline;
        eprintln!(
            "improvement: {:.1}% (baseline={avg_baseline:.4}, with_lora={avg_lora:.4})",
            improvement * 100.0
        );
        assert!(
            avg_lora < avg_baseline,
            "LoRA should reduce loss: baseline={avg_baseline:.4}, with_lora={avg_lora:.4}"
        );
        // Require at least 2% improvement to confirm the LoRA is meaningfully better,
        // not just noise. Single-sample few-shot training produces modest gains.
        assert!(
            improvement > 0.02,
            "LoRA improvement should be >2%: got {:.1}%",
            improvement * 100.0
        );

        // 5. Verify save/load roundtrip preserves the improvement
        save_lora_bin(&lora, &lora_path).expect("save LoRA");
        let loaded_lora = load_lora_bin(&lora_path).expect("load LoRA");
        let mut roundtrip_losses = Vec::new();
        for (tokens, targets) in &samples {
            let tokens_u32: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
            let targets_u32: Vec<u32> = targets.iter().map(|&t| t as u32).collect();
            model.cfg_mut().seq_len = tokens_u32.len();
            let fwd = ane_forward::forward_cpu_generic(
                &model,
                Some(&loaded_lora),
                &tokens_u32,
                &targets_u32,
            );
            roundtrip_losses.push(fwd.base.loss);
        }
        let avg_roundtrip = roundtrip_losses.iter().sum::<f32>() / roundtrip_losses.len() as f32;
        let roundtrip_drift = (avg_roundtrip - avg_lora).abs() / avg_lora;
        eprintln!(
            "roundtrip loss: {avg_roundtrip:.4} (drift: {:.2}%)",
            roundtrip_drift * 100.0
        );
        assert!(
            roundtrip_drift < 0.01,
            "save/load roundtrip should preserve loss within 1%: original={avg_lora:.4}, roundtrip={avg_roundtrip:.4}"
        );

        eprintln!("PASS: LoRA training produces measurable perplexity improvement");
    }

    /// Benchmark: quantized vs dense forward+backward per step.
    /// Not a correctness test — just prints timing comparison.
    #[cfg(feature = "mlx")]
    #[test]
    fn bench_quantized_vs_dense_step() {
        if skip_if_no_qwen3_5() {
            eprintln!("SKIP: Qwen3.5-0.8B not found");
            return;
        }

        use crate::agent::ane_forward;
        use crate::agent::ane_lora::{LoraConfig, LoraModel};
        use crate::agent::ane_weights::{QuantizedModelWeights, WeightSource};
        use crate::agent::mlx_lora::ModelConfig;

        let dir = qwen3_5_dir();
        let mc = ModelConfig::from_config_json(&dir).expect("model config");
        let mil_cfg = mc.to_mil_config(64);

        let tokenizer = crate::agent::mlx_lora::MlxTokenizer::load(&dir).expect("tokenizer");
        let messages = vec![
            crate::agent::mlx_server::ChatMessage {
                role: "user".into(),
                content: "What is the capital of France?".into(),
            },
            crate::agent::mlx_server::ChatMessage {
                role: "assistant".into(),
                content: "The capital of France is Paris.".into(),
            },
        ];
        let (tokens, targets) =
            crate::agent::mlx_server::tokenize_conversation(&tokenizer, &messages)
                .expect("tokenize");
        let tokens_u32: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
        let targets_u32: Vec<u32> = targets.iter().map(|&t| t as u32).collect();

        // Load quantized
        let mut q_model =
            QuantizedModelWeights::from_mlx_safetensors(&dir, &mil_cfg).expect("load");
        let n_layers = q_model.n_layers();
        let dim = mil_cfg.dim;
        let hidden = mil_cfg.hidden_dim;

        let lora = LoraModel::with_full_dims(
            LoraConfig::default(),
            n_layers,
            dim,
            mil_cfg.kv_dim(),
            mil_cfg.attn_dim(),
            mil_cfg.q_proj_dim(),
            hidden,
        );

        // Warmup + time quantized path
        q_model.cfg_mut().seq_len = tokens_u32.len();
        let _ = ane_forward::forward_cpu_generic(&q_model, Some(&lora), &tokens_u32, &targets_u32);

        let iters = 3;
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let fwd =
                ane_forward::forward_cpu_generic(&q_model, Some(&lora), &tokens_u32, &targets_u32);
            let _ = crate::agent::ane_backward::backward_lora_cpu_generic(
                &q_model,
                &fwd,
                &lora,
                &tokens_u32,
                15.0,
                256.0,
            );
        }
        let q_ms = t0.elapsed().as_millis() as f64 / iters as f64;

        // Dense cached model (auto budget)
        let mut d_model = crate::agent::ane_weights::DenseCachedModel::auto(q_model);
        d_model.cfg_mut().seq_len = tokens_u32.len();
        let _ = ane_forward::forward_cpu_generic(&d_model, Some(&lora), &tokens_u32, &targets_u32);

        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let fwd =
                ane_forward::forward_cpu_generic(&d_model, Some(&lora), &tokens_u32, &targets_u32);
            let _ = crate::agent::ane_backward::backward_lora_cpu_generic(
                &d_model,
                &fwd,
                &lora,
                &tokens_u32,
                15.0,
                256.0,
            );
        }
        let d_ms = t0.elapsed().as_millis() as f64 / iters as f64;

        let speedup = q_ms / d_ms;
        eprintln!(
            "quantized: {q_ms:.0}ms/step, cached: {d_ms:.0}ms/step ({}/{} layers), speedup: {speedup:.2}×",
            d_model.cached_layer_count(), d_model.n_layers()
        );
    }

    /// Test: Qwen3.5-35B-A3B training with DenseCachedModel.
    /// Validates the full pipeline on a large MoE model: load → cache → fwd → bwd → LoRA update → loss decreases.
    #[cfg(feature = "mlx")]
    #[test]
    fn test_35b_training_step() {
        let dir: std::path::PathBuf = dirs::home_dir()
            .unwrap()
            .join(".cache/lm-studio/models/mlx-community/Qwen3.5-35B-A3B-4bit");
        if !dir.join("tokenizer.json").exists() {
            eprintln!("SKIP: Qwen3.5-35B-A3B-4bit not found");
            return;
        }

        use crate::agent::ane_forward;
        use crate::agent::ane_lora::{LoraConfig, LoraModel};
        use crate::agent::ane_weights::{DenseCachedModel, QuantizedModelWeights, WeightSource};
        use crate::agent::mlx_lora::ModelConfig;

        let mc = ModelConfig::from_config_json(&dir).expect("model config");
        let mil_cfg = mc.to_mil_config(32); // shorter seq for speed

        // Load + cache
        let t0 = std::time::Instant::now();
        let quantized =
            QuantizedModelWeights::from_mlx_safetensors(&dir, &mil_cfg).expect("load 35B");
        let load_ms = t0.elapsed().as_millis();
        let q_mb = quantized.quantized_memory_bytes() as f64 / 1_048_576.0;
        eprintln!("loaded quantized in {load_ms}ms ({q_mb:.1} MB)");

        let t0 = std::time::Instant::now();
        let mut model = DenseCachedModel::auto(quantized);
        let cache_ms = t0.elapsed().as_millis();
        eprintln!(
            "dense cache: {}/{} layers in {cache_ms}ms",
            model.cached_layer_count(),
            model.n_layers()
        );

        // Tokenize a sample
        let tokenizer = crate::agent::mlx_lora::MlxTokenizer::load(&dir).expect("tokenizer");
        let messages = vec![
            crate::agent::mlx_server::ChatMessage {
                role: "user".into(),
                content: "What is 2+2?".into(),
            },
            crate::agent::mlx_server::ChatMessage {
                role: "assistant".into(),
                content: "4".into(),
            },
        ];
        let (tokens, targets) =
            crate::agent::mlx_server::tokenize_conversation(&tokenizer, &messages)
                .expect("tokenize");
        let tokens_u32: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
        let targets_u32: Vec<u32> = targets.iter().map(|&t| t as u32).collect();
        eprintln!("tokens: {} ids", tokens_u32.len());

        let n_layers = model.n_layers();
        let dim = model.actual_dim();
        let hidden = model.actual_hidden_dim();
        eprintln!(
            "dims: dim={dim}, hidden={hidden}, mil_cfg.dim={}, mil_cfg.hidden_dim={}",
            mil_cfg.dim, mil_cfg.hidden_dim
        );

        // Baseline loss (no LoRA)
        model.cfg_mut().seq_len = tokens_u32.len();
        let baseline_fwd =
            ane_forward::forward_cpu_generic(&model, None, &tokens_u32, &targets_u32);
        let baseline_loss = baseline_fwd.base.loss;
        eprintln!("baseline loss (no LoRA): {baseline_loss:.4}");
        assert!(baseline_loss.is_finite(), "baseline loss should be finite");

        // Train LoRA for a few steps
        let mut lora = LoraModel::with_full_dims(
            LoraConfig::default(),
            n_layers,
            dim,
            mil_cfg.kv_dim(),
            mil_cfg.attn_dim(),
            mil_cfg.q_proj_dim(),
            hidden,
        );
        let mut adam = crate::agent::ane_lora::LoraModelAdam::zeros(&lora);
        let lr = 1e-4;
        let epochs = 10;
        let mut best_loss = f32::INFINITY;
        let mut best_lora = lora.clone();

        // Verify first step produces non-zero dB gradients (dA is zero when B=0, which is correct)
        let fwd = ane_forward::forward_cpu_generic(&model, Some(&lora), &tokens_u32, &targets_u32);
        let bwd = crate::agent::ane_backward::backward_lora_cpu_generic(
            &model,
            &fwd,
            &lora,
            &tokens_u32,
            15.0,
            256.0,
        );
        let has_db_grads = bwd.lora_grads.layers.iter().any(|lg| {
            lg.w2
                .as_ref()
                .map_or(false, |g| g.db.iter().any(|&v| v != 0.0))
        });
        assert!(
            has_db_grads,
            "first step should produce non-zero dB gradients"
        );
        eprintln!("step 0: loss={:.4}, dB grads non-zero ✓", fwd.base.loss);

        // Apply first step
        crate::agent::ane_lora::lora_adam_update(
            &mut lora,
            &bwd.lora_grads,
            &mut adam,
            1,
            lr,
            0.9,
            0.999,
            1e-8,
            0.01,
        );

        // Continue training
        for step in 2..=epochs {
            let fwd =
                ane_forward::forward_cpu_generic(&model, Some(&lora), &tokens_u32, &targets_u32);
            let bwd = crate::agent::ane_backward::backward_lora_cpu_generic(
                &model,
                &fwd,
                &lora,
                &tokens_u32,
                15.0,
                256.0,
            );
            crate::agent::ane_lora::lora_adam_update(
                &mut lora,
                &bwd.lora_grads,
                &mut adam,
                step,
                lr,
                0.9,
                0.999,
                1e-8,
                0.01,
            );
            let loss = fwd.base.loss;
            if loss < best_loss {
                best_loss = loss;
                best_lora = lora.clone();
            }
            if step % 2 == 0 {
                eprintln!("  step {step}, loss={loss:.4} (best={best_loss:.4})");
            }
        }
        lora = best_lora;

        // Verify dA gradients are now non-zero (B has been updated from dB)
        let fwd_after =
            ane_forward::forward_cpu_generic(&model, Some(&lora), &tokens_u32, &targets_u32);
        let bwd_after = crate::agent::ane_backward::backward_lora_cpu_generic(
            &model,
            &fwd_after,
            &lora,
            &tokens_u32,
            15.0,
            256.0,
        );
        let has_da_grads = bwd_after.lora_grads.layers.iter().any(|lg| {
            lg.w2
                .as_ref()
                .map_or(false, |g| g.da.iter().any(|&v| v != 0.0))
        });
        assert!(
            has_da_grads,
            "after training, dA gradients should be non-zero (B is no longer zero)"
        );

        // Evaluate with best LoRA
        let eval_fwd =
            ane_forward::forward_cpu_generic(&model, Some(&lora), &tokens_u32, &targets_u32);
        let trained_loss = eval_fwd.base.loss;
        let improvement = (baseline_loss - trained_loss) / baseline_loss;
        eprintln!(
            "baseline={baseline_loss:.4}, trained={trained_loss:.4}, improvement={:.1}%",
            improvement * 100.0,
        );
        assert!(
            trained_loss < baseline_loss,
            "LoRA training should reduce loss: baseline={baseline_loss:.4}, trained={trained_loss:.4}"
        );

        eprintln!(
            "PASS: 35B training ({epochs} steps, {}/{} cached)",
            model.cached_layer_count(),
            model.n_layers()
        );
    }

    /// Test: one real ANE training step on Qwen3.5-35B-A3B.
    ///
    /// This exercises the actual bucketed ANE path used by `spawn_ane_training`:
    /// compile bucket kernels -> ANE forward -> ANE backward -> Adam update.
    #[cfg(feature = "mlx")]
    #[test]
    #[ignore = "requires local Qwen3.5-35B-A3B checkpoint and runs a real ANE training step"]
    fn test_35b_ane_training_smoke() {
        let dir: std::path::PathBuf = dirs::home_dir()
            .unwrap()
            .join(".cache/lm-studio/models/mlx-community/Qwen3.5-35B-A3B-4bit");
        if !dir.join("tokenizer.json").exists() {
            eprintln!("SKIP: Qwen3.5-35B-A3B-4bit not found");
            return;
        }

        use crate::agent::ane_backward;
        use crate::agent::ane_forward;
        use crate::agent::ane_lora::{
            lora_adam_update, lora_adam_update_split_lr, LoraConfig, LoraModel, LoraModelAdam,
        };
        use crate::agent::ane_weights::{DenseCachedModel, QuantizedModelWeights, WeightSource};

        let train_cfg =
            crate::agent::learn_loop::build_ane_training_config(Some(&dir)).expect("build config");
        let tokenizer = crate::agent::mlx_lora::MlxTokenizer::load(&dir).expect("tokenizer");
        let messages = vec![
            crate::agent::mlx_server::ChatMessage {
                role: "user".into(),
                content: "Explain why addition is commutative, then solve 27 + 15.".into(),
            },
            crate::agent::mlx_server::ChatMessage {
                role: "assistant".into(),
                content: "Addition is commutative because swapping the order of two quantities does not change the total. 27 + 15 = 42.".into(),
            },
        ];
        let (tokens, targets) =
            crate::agent::mlx_server::tokenize_conversation(&tokenizer, &messages)
                .expect("tokenize");
        let tokens_u32: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
        let targets_u32: Vec<u32> = targets.iter().map(|&t| t as u32).collect();
        assert!(!tokens_u32.is_empty(), "tokenization should produce tokens");

        let bucket_kernels = BucketKernels::compile(&[tokens_u32.len()], &train_cfg.mil_config)
            .expect("bucket compile");
        let (bucket_seq, bucket_fwd_k, bucket_bwd_k) = bucket_kernels.get(tokens_u32.len());
        eprintln!(
            "35B ANE smoke: sample_len={} bucket_seq={}",
            tokens_u32.len(),
            bucket_seq
        );
        assert!(bucket_bwd_k.wot_bwd.is_some(), "wot_bwd should compile");
        assert!(bucket_bwd_k.sdpa_bwd1.is_some(), "sdpa_bwd1 should compile");
        assert!(bucket_bwd_k.sdpa_bwd2.is_some(), "sdpa_bwd2 should compile");
        assert!(bucket_bwd_k.qkv_bwd.is_some(), "qkv_bwd should compile");
        let tok_pad = pad_to(&tokens_u32, *bucket_seq);
        let tgt_pad = pad_to(&targets_u32, *bucket_seq);

        let quantized = QuantizedModelWeights::from_mlx_safetensors(&dir, &train_cfg.mil_config)
            .expect("load 35B");
        let mut model = DenseCachedModel::auto(quantized);
        model.cfg_mut().seq_len = *bucket_seq;

        let n_layers = model.n_layers();
        let dim = model.actual_dim();
        let hidden = model.actual_hidden_dim();
        let residual_scale = if train_cfg.residual_scale > 0.0 {
            train_cfg.residual_scale
        } else {
            1.0 / (2.0 * n_layers as f32).sqrt()
        };

        let mut lora = LoraModel::with_full_dims(
            LoraConfig::default(),
            n_layers,
            dim,
            train_cfg.mil_config.kv_dim(),
            train_cfg.mil_config.attn_dim(),
            train_cfg.mil_config.q_proj_dim(),
            hidden,
        );
        let mut adam = LoraModelAdam::zeros(&lora);

        let fwd = ane_forward::forward_ane_generic(
            bucket_fwd_k,
            &model,
            Some(&lora),
            &tok_pad,
            &tgt_pad,
            train_cfg.softcap,
            residual_scale,
        )
        .expect("35B ANE forward should succeed");
        let initial_loss = fwd.base.loss;
        assert!(initial_loss.is_finite(), "initial loss should be finite");

        let bwd = ane_backward::backward_lora_ane_generic(
            bucket_bwd_k,
            &model,
            &fwd,
            &lora,
            &tok_pad,
            train_cfg.softcap,
            train_cfg.loss_scale,
            residual_scale,
        );
        let has_db_grads = bwd.lora_grads.layers.iter().any(|lg| {
            lg.w2
                .as_ref()
                .map_or(false, |g| g.db.iter().any(|&v| v != 0.0))
        });
        assert!(
            has_db_grads,
            "ANE backward should produce non-zero dB gradients"
        );

        let lora_b_norm_before: f32 = lora
            .layers
            .iter()
            .filter_map(|layer| layer.w2.as_ref())
            .flat_map(|adapter| adapter.b.iter())
            .map(|v| v.abs())
            .sum();

        if train_cfg.lr_scale_attn != 1.0 || train_cfg.lr_scale_ffn != 1.0 {
            lora_adam_update_split_lr(
                &mut lora,
                &bwd.lora_grads,
                &mut adam,
                1,
                train_cfg.lr,
                train_cfg.lr_scale_attn,
                train_cfg.lr_scale_ffn,
                0.9,
                0.999,
                1e-8,
                0.01,
            );
        } else {
            lora_adam_update(
                &mut lora,
                &bwd.lora_grads,
                &mut adam,
                1,
                train_cfg.lr,
                0.9,
                0.999,
                1e-8,
                0.01,
            );
        }

        let lora_b_norm_after: f32 = lora
            .layers
            .iter()
            .filter_map(|layer| layer.w2.as_ref())
            .flat_map(|adapter| adapter.b.iter())
            .map(|v| v.abs())
            .sum();
        assert!(
            lora_b_norm_after > lora_b_norm_before,
            "ANE step should update LoRA weights"
        );

        let fwd_after = ane_forward::forward_ane_generic(
            bucket_fwd_k,
            &model,
            Some(&lora),
            &tok_pad,
            &tgt_pad,
            train_cfg.softcap,
            residual_scale,
        )
        .expect("35B ANE post-update forward should succeed");
        let post_loss = fwd_after.base.loss;
        assert!(post_loss.is_finite(), "post-update loss should be finite");
        eprintln!(
            "35B ANE smoke: initial_loss={initial_loss:.4}, post_loss={post_loss:.4}, cached={}/{}",
            model.cached_layer_count(),
            model.n_layers()
        );
    }

    /// Test: adapter export produces correct tensor count for 35B MoE layout.
    ///
    /// 35B: 40 layers, 30 GDN + 10 MHA.
    /// GDN layers: only down_proj → 30 × 2 = 60 tensors
    /// MHA layers: q_proj + v_proj + o_proj + down_proj → 10 × 4 × 2 = 80 tensors
    /// Total: 140
    #[cfg(feature = "mlx")]
    #[test]
    fn test_export_ane_adapters_35b_moe_layout() {
        let dir: std::path::PathBuf = dirs::home_dir()
            .unwrap()
            .join(".cache/lm-studio/models/mlx-community/Qwen3.5-35B-A3B-4bit");
        if !dir.join("config.json").exists() {
            eprintln!("SKIP: Qwen3.5-35B-A3B-4bit not found");
            return;
        }

        let cfg =
            crate::agent::mlx_lora::ModelConfig::from_config_json(&dir).expect("parse config");
        assert_eq!(cfg.n_layers, 40);
        assert_eq!(cfg.linear_attn_indices.len(), 30);
        assert_eq!(cfg.hidden_dim, 512); // MoE intermediate size

        let lora = crate::agent::ane_lora::LoraModel::with_full_dims(
            crate::agent::ane_lora::LoraConfig::default(),
            cfg.n_layers,
            cfg.dim,
            cfg.n_kv_heads * cfg.head_dim,
            cfg.n_heads * cfg.head_dim,
            if cfg.attn_output_gate {
                2 * cfg.n_heads * cfg.head_dim
            } else {
                cfg.n_heads * cfg.head_dim
            },
            cfg.hidden_dim,
        );

        let tmpdir = tempfile::tempdir().unwrap();
        let n = crate::agent::mlx_lora::export_ane_adapters(
            &lora,
            &cfg,
            Some(&cfg.linear_attn_indices),
            tmpdir.path(),
        )
        .expect("35B ANE export failed");

        assert_eq!(
            n, 140,
            "35B MoE: 30 GDN×2 + 10 MHA×8 = 140 tensors"
        );

        let loaded =
            mlx_rs::Array::load_safetensors(&tmpdir.path().join("adapters.safetensors"))
                .expect("load back");
        assert_eq!(loaded.len(), 140);

        // GDN layer 0 should only have down_proj
        assert!(
            loaded
                .keys()
                .any(|k| k.contains(".layers.0.mlp.down_proj.lora_a.weight")),
            "GDN layer 0 should have down_proj"
        );
        assert!(
            !loaded
                .keys()
                .any(|k| k.contains(".layers.0.self_attn.q_proj.lora_a.weight")),
            "GDN layer 0 should NOT have attention LoRA"
        );

        // MHA layer 3 (every 4th is MHA in Qwen3.5) should have attention LoRA
        assert!(
            loaded
                .keys()
                .any(|k| k.contains(".layers.3.self_attn.q_proj.lora_a.weight")),
            "MHA layer 3 should have attention LoRA"
        );

        let config_json: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(tmpdir.path().join("adapter_config.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(config_json["num_layers"], 40);
        assert_eq!(config_json["lora_layers"], 10); // only MHA layers count
        assert_eq!(config_json["rank"], 32);
        eprintln!("PASS: 35B MoE adapter export — 140 tensors, correct layout");
    }

    /// Test: full `spawn_ane_training` pipeline on 35B MoE with mlx_tx=None.
    ///
    /// Exercises the production oMLX path end-to-end:
    /// load weights → dense cache → compile kernels → train → save LoRA →
    /// export adapters.safetensors → verify adapter file on disk.
    #[cfg(feature = "mlx")]
    #[test]
    fn test_35b_spawn_ane_training_standalone() {
        let dir: std::path::PathBuf = dirs::home_dir()
            .unwrap()
            .join(".cache/lm-studio/models/mlx-community/Qwen3.5-35B-A3B-4bit");
        if !dir.join("tokenizer.json").exists() {
            eprintln!("SKIP: Qwen3.5-35B-A3B-4bit not found");
            return;
        }

        let cfg = crate::agent::learn_loop::build_ane_training_config(Some(&dir))
            .expect("build 35B config");

        let tokenizer = crate::agent::mlx_lora::MlxTokenizer::load(&dir).expect("tokenizer");
        let messages = vec![
            crate::agent::mlx_server::ChatMessage {
                role: "user".into(),
                content: "What is 2+2?".into(),
            },
            crate::agent::mlx_server::ChatMessage {
                role: "assistant".into(),
                content: "4".into(),
            },
        ];
        let (tokens, targets) =
            crate::agent::mlx_server::tokenize_conversation(&tokenizer, &messages)
                .expect("tokenize");
        eprintln!("35B standalone: tokenized to {} tokens", tokens.len());

        // Clean adapter dir so we verify fresh export
        let adapter_dir = dir.join("adapters");
        let adapters_file = adapter_dir.join("adapters.safetensors");
        let modified_before = adapters_file.metadata().ok().and_then(|m| m.modified().ok());

        let t0 = std::time::Instant::now();
        let handle = spawn_ane_training(cfg, vec![(tokens, targets, 1.0)], None);
        let ok = handle.join().expect("training thread should not panic");
        let elapsed = t0.elapsed();

        assert!(ok, "35B spawn_ane_training should complete successfully");
        eprintln!("35B standalone: completed in {:.1}s", elapsed.as_secs_f64());

        // Verify LoRA .bin saved
        let model_key = dir.file_name().unwrap().to_string_lossy().to_string();
        let lora_path = dirs::home_dir()
            .unwrap()
            .join(".nanobot/workspace/lora")
            .join(format!("{model_key}.bin"));
        assert!(
            lora_path.exists(),
            "LoRA .bin should exist at {}",
            lora_path.display()
        );

        // Verify adapters.safetensors exported
        assert!(
            adapters_file.exists(),
            "adapters.safetensors should exist at {}",
            adapters_file.display()
        );
        let modified_after = adapters_file.metadata().ok().and_then(|m| m.modified().ok());
        assert!(
            modified_after > modified_before,
            "adapters.safetensors should have been updated"
        );

        // Verify adapter_config.json
        let config_json: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(adapter_dir.join("adapter_config.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(config_json["num_layers"], 40);
        assert_eq!(config_json["lora_layers"], 10);

        eprintln!(
            "PASS: 35B spawn_ane_training standalone — LoRA saved + adapters exported in {:.1}s",
            elapsed.as_secs_f64()
        );
    }

    // -----------------------------------------------------------------------
    // Phase 1: Offline training from session DB
    // -----------------------------------------------------------------------

    /// Extract training-ready conversation windows from sessions.db.
    ///
    /// Opens the DB in read-only mode, finds sessions with tool-call exchanges,
    /// and returns windows of messages suitable for training.
    #[cfg(feature = "mlx")]
    fn extract_training_conversations_from_db(
        db_path: &std::path::Path,
        max_conversations: usize,
        max_window: usize,
    ) -> Vec<Vec<serde_json::Value>> {
        use rusqlite::{params, Connection};
        use serde_json::json;

        let conn = match Connection::open_with_flags(
            db_path,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
        ) {
            Ok(c) => c,
            Err(_) => return Vec::new(),
        };

        // Find sessions that have tool messages (these are the rich ones)
        let session_ids: Vec<String> = {
            let mut stmt = conn
                .prepare(
                    "SELECT DISTINCT session_id FROM messages \
                     WHERE role = 'tool' ORDER BY session_id DESC LIMIT ?1",
                )
                .unwrap();
            stmt.query_map(params![max_conversations * 3], |row| row.get(0))
                .unwrap()
                .flatten()
                .collect()
        };

        let mut conversations = Vec::new();

        for session_id in &session_ids {
            let messages: Vec<serde_json::Value> = {
                let mut stmt = conn
                    .prepare(
                        "SELECT role, content, tool_calls, tool_call_id, tool_name \
                         FROM messages WHERE session_id = ?1 ORDER BY id ASC",
                    )
                    .unwrap();
                stmt.query_map(params![session_id], |row| {
                    let role: String = row.get(0)?;
                    let content: Option<String> = row.get(1)?;
                    let tool_calls: Option<String> = row.get(2)?;
                    let tool_call_id: Option<String> = row.get(3)?;
                    let tool_name: Option<String> = row.get(4)?;

                    let mut msg = json!({
                        "role": role,
                        "content": content.unwrap_or_default(),
                    });
                    if let Some(tc) = tool_calls {
                        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&tc) {
                            msg["tool_calls"] = parsed;
                        }
                    }
                    if let Some(id) = tool_call_id {
                        msg["tool_call_id"] = json!(id);
                    }
                    if let Some(name) = tool_name {
                        msg["name"] = json!(name);
                    }
                    Ok(msg)
                })
                .unwrap()
                .flatten()
                .collect()
            };

            // Filter to trainable roles only
            let trainable = ["user", "assistant", "tool"];
            let filtered: Vec<&serde_json::Value> = messages
                .iter()
                .filter(|m| {
                    let role = m.get("role").and_then(|v| v.as_str()).unwrap_or("");
                    trainable.contains(&role)
                })
                .filter(|m| {
                    // Skip very long messages (would blow up seq_len)
                    let content = m.get("content").and_then(|v| v.as_str()).unwrap_or("");
                    content.len() < 2000
                })
                .collect();

            // Extract conversation windows centered on tool-call exchanges
            let mut i = 0;
            while i < filtered.len() && conversations.len() < max_conversations {
                let role = filtered[i]
                    .get("role")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                if role != "user" {
                    i += 1;
                    continue;
                }

                let mut window: Vec<serde_json::Value> = vec![filtered[i].clone()];
                let mut j = i + 1;
                let mut has_tool = false;

                while j < filtered.len() && window.len() < max_window {
                    let r = filtered[j]
                        .get("role")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    if r == "user" {
                        break;
                    }
                    if r == "tool" {
                        has_tool = true;
                    }
                    window.push(filtered[j].clone());
                    j += 1;
                }

                // Prefer windows with tool exchanges, but accept any user→assistant pair
                if window.len() >= 2 && (has_tool || conversations.is_empty()) {
                    conversations.push(window);
                }

                i = j;
            }

            if conversations.len() >= max_conversations {
                break;
            }
        }

        conversations
    }

    /// Phase 1 test: train on real session data from sessions.db.
    ///
    /// Proves we can:
    /// 1. Read multi-turn conversations with tool calls from the session DB
    /// 2. Tokenize them with the full ChatML template (user/assistant/tool roles)
    /// 3. Run LoRA training and observe loss reduction
    ///
    /// Skips gracefully if sessions.db is missing/empty or no model available.
    #[cfg(feature = "mlx")]
    #[test]
    fn test_offline_training_from_sessions() {
        if skip_if_no_qwen3_5() {
            eprintln!("SKIP: Qwen3.5-0.8B not found");
            return;
        }

        let db_path = dirs::home_dir().unwrap().join(".nanobot/sessions.db");
        if !db_path.exists() {
            eprintln!("SKIP: sessions.db not found");
            return;
        }

        // 1. Extract conversations from session DB
        let conversations = extract_training_conversations_from_db(&db_path, 5, 15);
        if conversations.is_empty() {
            eprintln!("SKIP: no training conversations found in sessions.db");
            return;
        }
        let has_tool = conversations.iter().any(|conv| {
            conv.iter()
                .any(|m| m.get("role").and_then(|v| v.as_str()) == Some("tool"))
        });
        eprintln!(
            "extracted {} conversations (has_tool={})",
            conversations.len(),
            has_tool,
        );
        for (i, conv) in conversations.iter().enumerate() {
            let roles: Vec<&str> = conv
                .iter()
                .map(|m| m.get("role").and_then(|v| v.as_str()).unwrap_or("?"))
                .collect();
            eprintln!("  conv {i}: {} messages, roles={:?}", conv.len(), roles);
        }

        // 2. Tokenize with rich ChatML template
        let dir = qwen3_5_dir();
        let tokenizer = crate::agent::mlx_lora::MlxTokenizer::load(&dir).expect("tokenizer");
        let mut samples: Vec<(Vec<i32>, Vec<i32>)> = Vec::new();
        for conv in &conversations {
            match crate::agent::mlx_server::tokenize_rich_conversation(&tokenizer, conv) {
                Ok(pair) => {
                    // Skip samples that exceed max bucket size
                    if pair.0.len() <= 1024 {
                        samples.push(pair);
                    } else {
                        eprintln!(
                            "  skipping sample with {} tokens (exceeds 1024 limit)",
                            pair.0.len()
                        );
                    }
                }
                Err(e) => eprintln!("  tokenize failed: {e}"),
            }
        }
        assert!(
            !samples.is_empty(),
            "should tokenize at least one conversation from sessions.db"
        );
        eprintln!(
            "tokenized {} samples (token lengths: {:?})",
            samples.len(),
            samples.iter().map(|(t, _)| t.len()).collect::<Vec<_>>(),
        );

        // 3. Load model
        use crate::agent::ane_forward;
        use crate::agent::ane_lora::{LoraConfig, LoraModel, LoraModelAdam};
        use crate::agent::ane_weights::{DenseCachedModel, QuantizedModelWeights, WeightSource};
        use crate::agent::mlx_lora::ModelConfig;

        let mc = ModelConfig::from_config_json(&dir).expect("model config");
        let mil_cfg = mc.to_mil_config(64);

        let quantized =
            QuantizedModelWeights::from_mlx_safetensors(&dir, &mil_cfg).expect("load model");
        let mut model = DenseCachedModel::auto(quantized);
        let n_layers = model.n_layers();
        let dim = mil_cfg.dim;
        let hidden = mil_cfg.hidden_dim;
        eprintln!(
            "model loaded: {} layers, dim={dim}, {}/{} cached",
            n_layers,
            model.cached_layer_count(),
            n_layers,
        );

        // 4. Measure baseline loss
        let mut baseline_losses = Vec::new();
        for (i, (tokens, targets)) in samples.iter().enumerate() {
            let tokens_u32: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
            let targets_u32: Vec<u32> = targets.iter().map(|&t| t as u32).collect();
            model.cfg_mut().seq_len = tokens_u32.len();
            let fwd = ane_forward::forward_cpu_generic(&model, None, &tokens_u32, &targets_u32);
            baseline_losses.push(fwd.base.loss);
            eprintln!(
                "  baseline sample {i}: loss={:.4}, tokens={}",
                fwd.base.loss,
                tokens.len()
            );
        }
        let avg_baseline = baseline_losses.iter().sum::<f32>() / baseline_losses.len() as f32;
        eprintln!("average baseline loss: {avg_baseline:.4}");

        // 5. Train LoRA with per-epoch evaluation
        let mut lora = LoraModel::with_full_dims(
            LoraConfig::default(),
            n_layers,
            dim,
            mil_cfg.kv_dim(),
            mil_cfg.attn_dim(),
            mil_cfg.q_proj_dim(),
            hidden,
        );
        let mut adam = LoraModelAdam::zeros(&lora);

        let epochs = 20;
        let lr = 3e-5; // Lower LR for multi-sample stability (1e-4 diverges)
        let mut step = 0usize;
        let mut best_avg_loss = f32::INFINITY;
        let mut best_lora = lora.clone();

        // Pre-convert samples to u32 for reuse
        let samples_u32: Vec<(Vec<u32>, Vec<u32>)> = samples
            .iter()
            .map(|(t, g)| {
                (
                    t.iter().map(|&x| x as u32).collect(),
                    g.iter().map(|&x| x as u32).collect(),
                )
            })
            .collect();

        for epoch in 0..epochs {
            for (tokens_u32, targets_u32) in &samples_u32 {
                model.cfg_mut().seq_len = tokens_u32.len();

                let fwd =
                    ane_forward::forward_cpu_generic(&model, Some(&lora), tokens_u32, targets_u32);
                let bwd = crate::agent::ane_backward::backward_lora_cpu_generic(
                    &model, &fwd, &lora, tokens_u32, 15.0, 256.0,
                );
                step += 1;
                crate::agent::ane_lora::lora_adam_update(
                    &mut lora,
                    &bwd.lora_grads,
                    &mut adam,
                    step,
                    lr,
                    0.9,
                    0.999,
                    1e-8,
                    0.01,
                );
            }

            // Evaluate on all samples at end of each epoch
            let mut epoch_loss = 0.0f32;
            for (tokens_u32, targets_u32) in &samples_u32 {
                model.cfg_mut().seq_len = tokens_u32.len();
                let fwd =
                    ane_forward::forward_cpu_generic(&model, Some(&lora), tokens_u32, targets_u32);
                epoch_loss += fwd.base.loss;
            }
            let avg_loss = epoch_loss / samples_u32.len() as f32;
            if avg_loss < best_avg_loss {
                best_avg_loss = avg_loss;
                best_lora = lora.clone();
            }
            if (epoch + 1) % 5 == 0 {
                eprintln!(
                    "  epoch {}, avg_loss={avg_loss:.4} (best={best_avg_loss:.4})",
                    epoch + 1,
                );
            }
        }
        lora = best_lora;
        eprintln!("training complete: {step} steps, best_avg_loss={best_avg_loss:.4}");

        // 6. Measure trained loss
        let mut trained_losses = Vec::new();
        for (i, (tokens, targets)) in samples.iter().enumerate() {
            let tokens_u32: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
            let targets_u32: Vec<u32> = targets.iter().map(|&t| t as u32).collect();
            model.cfg_mut().seq_len = tokens_u32.len();
            let fwd =
                ane_forward::forward_cpu_generic(&model, Some(&lora), &tokens_u32, &targets_u32);
            trained_losses.push(fwd.base.loss);
            eprintln!("  trained sample {i}: loss={:.4}", fwd.base.loss);
        }
        let avg_trained = trained_losses.iter().sum::<f32>() / trained_losses.len() as f32;

        // 7. Assert improvement
        let improvement = (avg_baseline - avg_trained) / avg_baseline;
        eprintln!(
            "RESULT: baseline={avg_baseline:.4}, trained={avg_trained:.4}, \
             improvement={:.1}%",
            improvement * 100.0,
        );
        assert!(
            avg_trained < avg_baseline,
            "LoRA should reduce loss on session data: \
             baseline={avg_baseline:.4}, trained={avg_trained:.4}"
        );
        assert!(
            improvement > 0.02,
            "expected >2% improvement on session data, got {:.1}%",
            improvement * 100.0,
        );

        eprintln!(
            "PASS: offline training from sessions.db ({} samples, {step} steps, {:.1}% improvement)",
            samples.len(),
            improvement * 100.0,
        );
    }
}
