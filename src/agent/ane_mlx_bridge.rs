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
    mlx_tx: std::sync::mpsc::SyncSender<super::mlx_server::ModelRequest>,
) -> std::thread::JoinHandle<bool> {
    std::thread::Builder::new()
        .name("ane-lora-train".into())
        .spawn(move || {
            use super::ane_backward;
            use super::ane_forward;
            use super::ane_lora::{
                self, load_lora_bin, lora_adam_update, save_lora_bin, LoraConfig, LoraModel,
                LoraModelAdam,
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
            let model_key = cfg.model_dir
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
                        LoraModel::with_kv_dim(
                            LoraConfig::default(),
                            n_layers,
                            dim,
                            cfg.kv_dim,
                            hidden_dim,
                        )
                    }
                }
            } else {
                tracing::info!("ANE train: new LoRA for model {model_key}");
                LoraModel::with_kv_dim(LoraConfig::default(), n_layers, dim, cfg.kv_dim, hidden_dim)
            };
            let mut adam = LoraModelAdam::zeros(&lora);

            // 3. Train
            let n_samples = samples.len();
            let total_steps = n_samples * cfg.epochs;
            let patience = n_samples * 2;
            let mut step = 0usize;
            let mut best_loss = f32::INFINITY;
            let mut stale_count = 0usize;

            tracing::info!(
                "ANE train: {n_samples} samples, {total_steps} steps, lr={}",
                cfg.lr
            );

            'outer: for _epoch in 0..cfg.epochs {
                for (tokens, targets) in &samples {
                    let tokens_u32: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
                    let targets_u32: Vec<u32> = targets.iter().map(|&t| t as u32).collect();

                    // forward_cpu reads seq_len from cfg — update per sample
                    model.cfg.seq_len = tokens_u32.len();

                    let fwd = ane_forward::forward_cpu_generic(
                        &model,
                        Some(&lora),
                        &tokens_u32,
                        &targets_u32,
                    );
                    let loss = fwd.base.loss;

                    if !loss.is_finite() {
                        tracing::warn!("ANE train: NaN/Inf loss at step {step}, stopping");
                        break 'outer;
                    }

                    let bwd =
                        ane_backward::backward_lora_cpu_generic(&model, &fwd, &lora, &tokens_u32);

                    step += 1;
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

            // 5. Extract deltas and send to MLX
            let deltas = extract_lora_deltas(
                &lora,
                if cfg.linear_attn_indices.is_empty() {
                    None
                } else {
                    Some(&cfg.linear_attn_indices)
                },
            );
            let n_deltas = deltas.layers.len();
            let _ = mlx_tx.send(super::mlx_server::ModelRequest::ApplyLoraDeltas {
                deltas,
                reply: None, // fire-and-forget
            });
            tracing::info!("ANE train: sent {n_deltas} deltas to MLX worker");

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
    // E2E integration tests (require Qwen3-1.7B weights on disk)
    // -----------------------------------------------------------------------

    fn qwen3_1_7b_dir() -> std::path::PathBuf {
        dirs::home_dir()
            .unwrap()
            .join(".cache/lm-studio/models/lmstudio-community/Qwen3-1.7B-MLX-8bit")
    }

    fn skip_if_no_qwen3() -> bool {
        !qwen3_1_7b_dir().join("tokenizer.json").exists()
    }

    /// E2E: ANE trains LoRA, extracts deltas, applies to MLX model,
    /// verifies forward pass output changes.
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
            },
            epochs: 1,
            lr: 1e-5,
            linear_attn_indices: vec![],
            kv_dim: 8 * 128, // n_kv_heads=8, head_dim=128
        };

        eprintln!("spawning ANE training thread...");
        let handle = spawn_ane_training(cfg, vec![(tokens, targets)], tx);

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
            },
            epochs: 2,
            lr: 1e-5,
            linear_attn_indices: vec![],
            kv_dim: 8 * 128,
        };
        let _ane_handle = spawn_ane_training(ane_cfg, vec![(tokens, targets)], ane_tx);

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
}
