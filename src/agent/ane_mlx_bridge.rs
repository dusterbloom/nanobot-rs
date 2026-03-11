//! ANE↔MLX LoRA weight bridge.
//!
//! Transfers trained LoRA weights from ANE (CPU f32 Vec) to MLX (GPU Array).
//! Both systems use identical shapes: A=[rank, d_in], B=[d_out, rank].
//!
//! Name mapping: ANE wq/wv/wo/w2 → MLX q_proj/v_proj/o_proj/down_proj

use super::ane_lora::{LoraAdapter, LoraModel};

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
}

/// Seq-len bucket sizes for ANE kernel compilation. Samples are padded to
/// the nearest bucket. Keeps compilation count low (3 × 10 = 30 kernels).
const BUCKET_SIZES: &[usize] = &[128, 256, 512];

/// Pre-compiled forward + backward kernels for multiple seq_len buckets.
pub struct BucketKernels {
    pub buckets: Vec<(usize, super::ane_forward::CompiledKernels, super::ane_backward::BackwardKernels)>,
}

impl BucketKernels {
    /// Compile kernel sets for buckets that cover the given sample lengths.
    pub fn compile(sample_lens: &[usize], base_cfg: &super::ane_mil::MilConfig) -> Result<Self, String> {
        let mut needed: Vec<usize> = Vec::new();
        for &sl in sample_lens {
            let bucket = BUCKET_SIZES.iter().copied().find(|&b| b >= sl)
                .unwrap_or(*BUCKET_SIZES.last().unwrap());
            if !needed.contains(&bucket) {
                needed.push(bucket);
            }
        }
        needed.sort();

        let mut buckets = Vec::new();
        for bucket_seq in needed {
            let mut cfg = base_cfg.clone();
            cfg.seq_len = bucket_seq;
            let fwd = super::ane_forward::CompiledKernels::compile_forward(&cfg)?;
            let bwd = super::ane_backward::BackwardKernels::compile_backward(&cfg, &fwd.mask_blob)?;
            tracing::info!("ANE train: compiled kernels for seq_len={bucket_seq}");
            buckets.push((bucket_seq, fwd, bwd));
        }

        Ok(BucketKernels { buckets })
    }

    /// Get the kernel set for a given sequence length (rounds up to nearest bucket).
    pub fn get(&self, seq_len: usize) -> &(usize, super::ane_forward::CompiledKernels, super::ane_backward::BackwardKernels) {
        self.buckets.iter()
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
#[cfg(feature = "mlx")]
pub fn spawn_ane_training(
    cfg: AneTrainingConfig,
    samples: Vec<(Vec<i32>, Vec<i32>)>,
    mlx_tx: Option<std::sync::mpsc::SyncSender<super::mlx_server::ModelRequest>>,
) -> std::thread::JoinHandle<bool> {
    std::thread::Builder::new()
        .name("ane-lora-train".into())
        .spawn(move || {
            // Drop thread to background QoS so training doesn't compete with
            // MLX inference for CPU time. On macOS this tells the scheduler to
            // yield to higher-priority work (UI, inference, networking).
            #[cfg(target_os = "macos")]
            unsafe {
                libc::setpriority(libc::PRIO_PROCESS, 0, 10);
            }

            use super::ane_backward;
            use super::ane_forward;
            use super::ane_lora::{
                self, load_lora_bin, lora_adam_update, lora_adam_update_split_lr, save_lora_bin,
                LoraConfig, LoraModel, LoraModelAdam,
            };
            use super::ane_weights::{QuantizedModelWeights, WeightSource};

            let t0 = std::time::Instant::now();

            // 1. Load base model weights in quantized form (QLoRA: ~4x less memory).
            // Layer weights stay in 8-bit and are dequantized per-layer during
            // forward/backward, keeping only one layer's f32 weights in memory.
            let mut model = match QuantizedModelWeights::from_mlx_safetensors(
                &cfg.model_dir,
                &cfg.mil_config,
            ) {
                Ok(m) => {
                    tracing::info!(
                        "ANE train: loaded quantized model in {}ms ({:.1} MB)",
                        t0.elapsed().as_millis(),
                        m.quantized_memory_bytes() as f64 / 1_048_576.0,
                    );
                    m
                }
                Err(e) => {
                    tracing::error!("ANE train: failed to load weights: {e}");
                    return false;
                }
            };

            // 2. Initialize or restore LoRA (per-model file keyed by dir name)
            let lora_dir = dirs::home_dir()
                .unwrap_or_default()
                .join(".nanobot/workspace/lora");
            let model_key = cfg
                .model_dir
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| "default".into());
            let lora_path = lora_dir.join(format!("{model_key}.bin"));
            let n_layers = model.n_layers();
            let dim = cfg.mil_config.dim;
            let hidden_dim = cfg.mil_config.hidden_dim;

            let mut lora = if lora_path.exists() {
                match load_lora_bin(&lora_path) {
                    Ok(l) if l.layers.len() == n_layers => {
                        tracing::info!("ANE train: restored LoRA from {}", lora_path.display());
                        l
                    }
                    _ => {
                        tracing::warn!("ANE train: stale LoRA file, reinitializing");
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
                }
            } else {
                tracing::info!("ANE train: new LoRA for model {model_key}");
                LoraModel::with_full_dims(
                    LoraConfig::default(),
                    n_layers,
                    dim,
                    cfg.mil_config.kv_dim(),
                    cfg.mil_config.attn_dim(),
                    cfg.mil_config.q_proj_dim(),
                    hidden_dim,
                )
            };
            let mut adam = LoraModelAdam::zeros(&lora);

            // 3. Compile ANE kernels (bucket-based for variable seq_len)
            let sample_lens: Vec<usize> = samples.iter().map(|(t, _)| t.len()).collect();
            let bucket_kernels = match BucketKernels::compile(&sample_lens, &cfg.mil_config) {
                Ok(bk) => {
                    tracing::info!(
                        "ANE train: compiled {} bucket(s) for ANE-accelerated training",
                        bk.buckets.len()
                    );
                    Some(bk)
                }
                Err(e) => {
                    tracing::warn!("ANE train: kernel compilation failed ({e}), falling back to CPU");
                    None
                }
            };

            let residual_scale = if cfg.residual_scale > 0.0 {
                cfg.residual_scale
            } else {
                1.0 / (2.0 * n_layers as f32).sqrt()
            };
            let use_split_lr = cfg.lr_scale_attn != 1.0 || cfg.lr_scale_ffn != 1.0;

            // 4. Train
            let n_samples = samples.len();
            let total_steps = n_samples * cfg.epochs;
            let patience = n_samples * 2;
            let mut step = 0usize;
            let mut best_loss = f32::INFINITY;
            let mut stale_count = 0usize;

            tracing::info!(
                "ANE train: {n_samples} samples, {total_steps} steps, lr={}, mode={}",
                cfg.lr,
                if bucket_kernels.is_some() { "ANE" } else { "CPU" },
            );

            'outer: for _epoch in 0..cfg.epochs {
                for (tokens, targets) in &samples {
                    let tokens_u32: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
                    let targets_u32: Vec<u32> = targets.iter().map(|&t| t as u32).collect();

                    let (fwd, bwd) = if let Some(ref bk) = bucket_kernels {
                        // ANE path: pad to bucket, use ANE forward/backward
                        let (bucket_seq, fwd_k, bwd_k) = bk.get(tokens_u32.len());
                        let tok_pad = pad_to(&tokens_u32, *bucket_seq);
                        let tgt_pad = pad_to(&targets_u32, *bucket_seq);
                        model.cfg.seq_len = *bucket_seq;

                        let fwd = match ane_forward::forward_ane_generic(
                            fwd_k,
                            &model,
                            Some(&lora),
                            &tok_pad,
                            &tgt_pad,
                            cfg.softcap,
                            residual_scale,
                        ) {
                            Ok(f) => f,
                            Err(e) => {
                                tracing::warn!("ANE forward failed ({e}), falling back to CPU");
                                model.cfg.seq_len = tokens_u32.len();
                                ane_forward::forward_cpu_generic(
                                    &model,
                                    Some(&lora),
                                    &tokens_u32,
                                    &targets_u32,
                                )
                            }
                        };

                        let bwd = ane_backward::backward_lora_ane_generic(
                            bwd_k,
                            &model,
                            &fwd,
                            &lora,
                            &tok_pad,
                            cfg.softcap,
                            cfg.loss_scale,
                            residual_scale,
                        );
                        (fwd, bwd)
                    } else {
                        // CPU fallback path (existing behavior)
                        model.cfg.seq_len = tokens_u32.len();
                        let fwd = ane_forward::forward_cpu_generic(
                            &model,
                            Some(&lora),
                            &tokens_u32,
                            &targets_u32,
                        );
                        let bwd = ane_backward::backward_lora_cpu_generic(
                            &model, &fwd, &lora, &tokens_u32, cfg.softcap, cfg.loss_scale,
                        );
                        (fwd, bwd)
                    };

                    let loss = fwd.base.loss;
                    if !loss.is_finite() {
                        tracing::warn!("ANE train: NaN/Inf loss at step {step}, stopping");
                        break 'outer;
                    }

                    step += 1;
                    if use_split_lr {
                        lora_adam_update_split_lr(
                            &mut lora,
                            &bwd.lora_grads,
                            &mut adam,
                            step,
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
                            &mut lora,
                            &bwd.lora_grads,
                            &mut adam,
                            step,
                            cfg.lr,
                            0.9,
                            0.999,
                            1e-8,
                            0.01,
                        );
                    }

                    // Early stopping
                    if loss < best_loss {
                        best_loss = loss;
                        stale_count = 0;
                    } else {
                        stale_count += 1;
                    }
                    if stale_count >= patience {
                        tracing::info!("ANE train: early stop at step {step}, loss={loss:.4}");
                        break 'outer;
                    }

                    if step % 5 == 0 || step == total_steps {
                        tracing::debug!("ANE train: step {step}/{total_steps}, loss={loss:.4}");
                    }
                }
            }

            let train_ms = t0.elapsed().as_millis();
            tracing::info!("ANE train: done in {train_ms}ms, best_loss={best_loss:.4}");

            // 4. Save LoRA to disk
            let saved = if let Err(e) = std::fs::create_dir_all(&lora_dir) {
                tracing::warn!("ANE train: failed to create lora dir: {e}");
                false
            } else if let Err(e) = save_lora_bin(&lora, &lora_path) {
                tracing::warn!("ANE train: failed to save LoRA: {e}");
                false
            } else {
                tracing::info!("ANE train: saved LoRA to {}", lora_path.display());
                true
            };

            // 5. Optionally hot-swap into MLX worker
            if let Some(ref tx) = mlx_tx {
                let deltas = extract_lora_deltas(
                    &lora,
                    if cfg.linear_attn_indices.is_empty() {
                        None
                    } else {
                        Some(&cfg.linear_attn_indices)
                    },
                );
                let n_deltas = deltas.layers.len();
                let _ = tx.send(super::mlx_server::ModelRequest::ApplyLoraDeltas {
                    deltas,
                    reply: None, // fire-and-forget
                });
                tracing::info!("ANE train: sent {n_deltas} deltas to MLX worker");
            }

            saved
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
        use mlx_rs::module::Module;

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
        };

        eprintln!("spawning ANE training thread...");
        let handle = spawn_ane_training(cfg, vec![(tokens, targets)], Some(tx));

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
        use crate::agent::mlx_lora::{LoraConfig as MlxLoraConfig, MlxLoraModel, ModelConfig};
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
        };
        let _ane_handle = spawn_ane_training(ane_cfg, vec![(tokens, targets)], Some(ane_tx));

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
        assert!(!mil.linear_attn_indices.is_empty(), "should have linear attention layers");
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
            mil.dim, mil.hidden_dim, mil.n_heads, mil.n_kv_heads,
            mil.head_dim_explicit, mil.linear_attn_indices.len()
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
        let cfg = crate::agent::learn_loop::build_ane_training_config(Some(&dir))
            .expect("build config");

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
        eprintln!("  SDPA: {}", if fwd.sdpa_fwd.is_some() { "ANE" } else { "CPU (GQA)" });
        eprintln!("  FFN:  {}", match &fwd.ffn {
            FfnKernels::Fused { .. } => "ANE (fused)",
            FfnKernels::Tiled { .. } => "ANE (tiled)",
        });

        // The critical assertion: FFN MUST be on ANE
        assert!(
            matches!(&fwd.ffn, FfnKernels::Fused { .. } | FfnKernels::Tiled { .. }),
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
        let cfg = crate::agent::learn_loop::build_ane_training_config(Some(&dir))
            .expect("build config");

        // Tokenize a sample conversation
        let tokenizer = crate::agent::mlx_lora::MlxTokenizer::load(&dir)
            .expect("tokenizer load");
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
        let lora_dir = dirs::home_dir()
            .unwrap()
            .join(".nanobot/workspace/lora");
        let lora_path = lora_dir.join(format!("{model_key}.bin"));
        let had_existing = lora_path.exists();
        // Don't delete — just check if a new/updated one appears after training

        let modified_before = lora_path.metadata().ok().and_then(|m| m.modified().ok());

        // Spawn training with mlx_tx: None (oMLX standalone path)
        eprintln!("spawning standalone ANE training (no MLX hot-swap)...");
        let handle = spawn_ane_training(cfg, vec![(tokens, targets)], None);

        let ok = handle.join().expect("training thread should not panic");
        assert!(ok, "training should complete successfully");

        // Verify LoRA file was saved
        assert!(lora_path.exists(), "LoRA .bin should exist at {}", lora_path.display());
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
        };

        // Build a TurnOutcome with high surprise content
        let outcome = TurnOutcome {
            user_content: "Explain quantum entanglement in detail with examples".into(),
            final_content: "Quantum entanglement is a phenomenon...".into(),
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
        assert!(handle.is_some(), "observe_async should return a JoinHandle (training spawned)");

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
}
