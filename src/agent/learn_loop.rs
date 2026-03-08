//! Unified learning dispatch for turn observations.
//!
//! Every learning/metrics observation flows through the [`LearnLoop`] trait
//! with immediate (sync) and async paths. This replaces scattered direct-write
//! observers in `finalize_response.rs`.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use tracing::{debug, info, warn};

use crate::agent::audit::TurnToolEntry;
use crate::agent::budget_calibrator::BudgetCalibrator;
use crate::agent::lora_bridge::ExperienceBuffer;
use crate::config::schema::PerplexityGateConfig;

/// Flat struct capturing all observer-needed data from a completed turn.
///
/// All fields are owned (no references/lifetimes) so `TurnOutcome` can be
/// moved across thread boundaries into async tasks.
pub(crate) struct TurnOutcome {
    pub user_content: String,
    pub final_content: String,
    pub model: String,
    pub session_key: String,
    pub workspace: PathBuf,
    pub used_tools: HashSet<String>,
    pub turn_tool_entries: Vec<TurnToolEntry>,
    pub iterations_used: u32,
    pub max_iterations: u32,
    pub turn_count: u64,
    pub turn_start_elapsed_ms: u64,
    pub context_tokens: u64,
    pub message_count: usize,
    pub working_memory_tokens: u64,
    pub provenance_audit_enabled: bool,
    pub is_local: bool,
    pub cost_usd: f64,
    pub prompt_tokens: i64,
    pub completion_tokens: i64,
}

/// Single dispatch point for all turn observations.
///
/// Implementations must never panic -- observer failures are logged via
/// `tracing::warn` and never block message delivery.
pub(crate) trait LearnLoop: Send + Sync {
    /// Synchronous observers: audit log, budget calibrator, structured logging.
    fn observe_immediate(&self, outcome: &TurnOutcome);

    /// Async observers: perplexity gate, LoRA training.
    /// Returns `Some(JoinHandle)` when async work is spawned, `None` otherwise.
    fn observe_async(&self, outcome: TurnOutcome) -> Option<tokio::task::JoinHandle<()>>;
}

// ---------------------------------------------------------------------------
// DefaultLearnLoop -- production implementation
// ---------------------------------------------------------------------------

/// Production implementation that dispatches to audit, calibrator, and
/// perplexity gate observers.
pub(crate) struct DefaultLearnLoop {
    pub calibrator: Option<Arc<parking_lot::Mutex<BudgetCalibrator>>>,
    pub experience_buffer: Option<Arc<parking_lot::Mutex<ExperienceBuffer>>>,
    pub perplexity_gate_config: PerplexityGateConfig,
    #[cfg(feature = "mlx")]
    pub mlx_provider: Option<Arc<crate::providers::mlx::MlxProvider>>,
    /// Runtime counters for training status visibility in TUI.
    pub training_counters: Option<Arc<crate::agent::agent_core::RuntimeCounters>>,
}

impl DefaultLearnLoop {
    /// Classify the turn into a task type for budget calibration.
    fn task_type(outcome: &TurnOutcome) -> &'static str {
        if outcome.used_tools.contains("exec_command") {
            "shell"
        } else if outcome.used_tools.contains("web_search") {
            "web_search"
        } else if outcome.used_tools.contains("spawn_agent") {
            "delegate"
        } else if outcome.used_tools.is_empty() {
            "chat"
        } else {
            "tool_use"
        }
    }
}

impl LearnLoop for DefaultLearnLoop {
    fn observe_immediate(&self, outcome: &TurnOutcome) {
        let task_type = Self::task_type(outcome);

        // --- Structured observability log ---
        info!(
            task_type = %task_type,
            iterations = outcome.iterations_used,
            tool_calls = outcome.used_tools.len(),
            prompt_tokens = outcome.prompt_tokens,
            completion_tokens = outcome.completion_tokens,
            cost_usd = format!("{:.6}", outcome.cost_usd),
            duration_ms = outcome.turn_start_elapsed_ms,
            success = !outcome.final_content.is_empty(),
            "turn_completed"
        );

        // --- Audit log ---
        if outcome.provenance_audit_enabled {
            let summary = crate::agent::audit::TurnSummary {
                turn: outcome.turn_count,
                timestamp: chrono::Utc::now().to_rfc3339(),
                context_tokens: outcome.context_tokens as usize,
                message_count: outcome.message_count,
                tools_called: outcome.turn_tool_entries.clone(),
                working_memory_tokens: outcome.working_memory_tokens as usize,
            };
            crate::agent::audit::write_turn_summary(
                &outcome.workspace,
                &outcome.session_key,
                &summary,
            );
        }

        // --- Budget calibrator ---
        if let Some(ref cal_mutex) = self.calibrator {
            let record = crate::agent::budget_calibrator::ExecutionRecord {
                task_type: task_type.to_string(),
                model: outcome.model.clone(),
                iterations_used: outcome.iterations_used,
                max_iterations: outcome.max_iterations,
                success: !outcome.final_content.is_empty(),
                cost_usd: outcome.cost_usd,
                duration_ms: outcome.turn_start_elapsed_ms,
                depth: 0,
                tool_calls: outcome.used_tools.len() as u32,
                created_at: chrono::Utc::now().to_rfc3339(),
            };
            let cal = cal_mutex.lock();
            if let Err(e) = cal.record(&record) {
                warn!("BudgetCalibrator record failed: {}", e);
            }
        }
    }

    fn observe_async(&self, outcome: TurnOutcome) -> Option<tokio::task::JoinHandle<()>> {
        // Only trigger perplexity gate when enabled and content exists.
        // Training fires for all turns (not just tool-use) so the experience
        // buffer captures conversational patterns too.
        if !self.perplexity_gate_config.enabled || outcome.final_content.is_empty() {
            return None;
        }

        // Warn once if the threshold looks like a legacy CE-loss value.
        if self.perplexity_gate_config.surprise_threshold > 1.0 {
            use std::sync::atomic::{AtomicBool, Ordering};
            static WARNED: AtomicBool = AtomicBool::new(false);
            if !WARNED.swap(true, Ordering::Relaxed) {
                warn!(
                    "perplexity_gate: surpriseThreshold={:.1} exceeds heuristic range (0.0-1.0), \
                     clamping to 1.0 — update config to a value like 0.3",
                    self.perplexity_gate_config.surprise_threshold
                );
            }
        }

        let eb_mutex = self.experience_buffer.as_ref()?.clone();
        let pg_config = self.perplexity_gate_config.clone();
        let train_counters = self.training_counters.clone();

        #[cfg(feature = "mlx")]
        let mlx_provider = self.mlx_provider.clone();

        let handle = tokio::spawn(async move {
            let tool_entries_json: Vec<serde_json::Value> = outcome
                .turn_tool_entries
                .iter()
                .map(|e| {
                    serde_json::json!({
                        "name": e.name,
                        "ok": e.ok,
                        "duration_ms": e.duration_ms,
                    })
                })
                .collect();
            let trace_json =
                serde_json::to_string(&tool_entries_json).unwrap_or_else(|_| "[]".into());

            // Heuristic surprise (zero model contention).
            //
            // Model-based perplexity (in-process or HTTP) blocks the model
            // worker, stalling inference for the next user turn. The heuristic
            // is instant and contention-free — training quality comes from the
            // LoRA step itself, not from the gate precision.
            let surprise = crate::agent::lora_bridge::compute_surprise(
                &outcome.user_content,
                &trace_json,
            );

            // Heuristic surprise is 0.0–1.0; clamp the threshold so legacy
            // configs that still carry the old CE-loss default (3.0) don't
            // silently disable training.
            let threshold = (pg_config.surprise_threshold as f64).min(1.0);
            if surprise <= threshold {
                return;
            }

            // Record under lock, then release before any async work.
            let should_train = {
                let eb = eb_mutex.lock();
                match eb.record_with_surprise(
                    &outcome.user_content,
                    &trace_json,
                    &outcome.final_content,
                    true,
                    1.0,
                    &outcome.model,
                    surprise,
                ) {
                    Ok(id) => {
                        debug!(
                            experience_id = id,
                            surprise = format!("{:.3}", surprise),
                            "perplexity_gate: recorded surprising experience"
                        );
                    }
                    Err(e) => {
                        debug!("perplexity_gate: record failed: {}", e);
                    }
                }
                let min_exp = pg_config.min_experiences;
                eb.stats()
                    .map(|s| s.unexported as usize >= min_exp)
                    .unwrap_or(false)
            }; // lock released

            if !should_train {
                return;
            }

            let epochs = pg_config.train_epochs;
            let min_exp = pg_config.min_experiences;

            // Collect experiences under lock.
            let (exps_data, ids) = {
                let eb = eb_mutex.lock();
                let exps = match eb.top_unexported(min_exp) {
                    Ok(e) => e,
                    Err(_) => return,
                };
                if exps.is_empty() {
                    return;
                }
                let data: Vec<(String, String)> = exps
                    .iter()
                    .map(|e| (e.prompt.clone(), e.response.clone()))
                    .collect();
                let ids: Vec<i64> = exps.iter().map(|e| e.id).collect();
                (data, ids)
            };

            // ANE split-silicon training: train on CPU/ANE thread,
            // hot-swap weights into MLX GPU model when done.
            #[cfg(all(feature = "ane", feature = "mlx"))]
            if let Some(ref mlx) = mlx_provider {
                let model_dir = std::path::Path::new(mlx.model_path());
                if let Some(ane_cfg) = build_ane_training_config(Some(model_dir)) {
                    let tokenizer = match crate::agent::mlx_lora::MlxTokenizer::load(
                        std::path::Path::new(&ane_cfg.model_dir),
                    ) {
                        Ok(t) => t,
                        Err(e) => {
                            debug!("perplexity_gate: ANE tokenizer load failed: {e}");
                            // Fall through to HTTP training (no model worker contention)
                            try_http_train(
                                &exps_data,
                                &ids,
                                &eb_mutex,
                                epochs,
                                &pg_config.mlx_server_url,
                                &train_counters,
                            )
                            .await;
                            return;
                        }
                    };
                    let mut samples = Vec::new();
                    for (prompt, response) in &exps_data {
                        let messages = vec![
                            crate::agent::mlx_server::ChatMessage {
                                role: "user".into(),
                                content: prompt.clone(),
                            },
                            crate::agent::mlx_server::ChatMessage {
                                role: "assistant".into(),
                                content: response.clone(),
                            },
                        ];
                        if let Ok(pair) =
                            crate::agent::mlx_server::tokenize_conversation(&tokenizer, &messages)
                        {
                            samples.push(pair);
                        }
                    }
                    if !samples.is_empty() {
                        let mlx_tx = mlx.model_tx();
                        // Signal training start.
                        if let Some(ref tc) = train_counters {
                            tc.training_active.store(true, Ordering::Relaxed);
                            let now_ms = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .map(|d| d.as_millis() as u64)
                                .unwrap_or(0);
                            tc.training_started_ms.store(now_ms, Ordering::Relaxed);
                        }
                        let tc_for_done = train_counters.clone();
                        let handle = crate::agent::ane_mlx_bridge::spawn_ane_training(
                            ane_cfg, samples, mlx_tx,
                        );
                        // Spawn a watcher that clears training_active when the
                        // ANE thread completes (runs on a blocking thread to avoid
                        // tying up the async runtime).
                        tokio::task::spawn_blocking(move || {
                            let _ = handle.join();
                            if let Some(ref tc) = tc_for_done {
                                tc.training_active.store(false, Ordering::Relaxed);
                                tc.training_steps_total.fetch_add(1, Ordering::Relaxed);
                            }
                        });
                        let eb = eb_mutex.lock();
                        let _ = eb.mark_exported(&ids);
                        info!(
                            "perplexity_gate: ANE split-silicon training spawned ({} experiences)",
                            ids.len()
                        );
                        return;
                    }
                }
            }

            // HTTP-only training fallback (short timeout, no model worker contention).
            // In-process MLX training is skipped — it sends ModelRequest::Train to
            // the same worker that handles Chat, blocking inference for the entire
            // training duration. ANE split-silicon (above) is the preferred path.
            try_http_train(
                &exps_data,
                &ids,
                &eb_mutex,
                epochs,
                &pg_config.mlx_server_url,
                &train_counters,
            )
            .await;
        });

        Some(handle)
    }
}

// ---------------------------------------------------------------------------
// Helper functions (moved from finalize_response.rs)
// ---------------------------------------------------------------------------

/// Query the MLX server's `/perplexity` endpoint for real CE-loss surprise.
///
/// Returns `Some(loss)` on success, `None` if the server is unreachable or errors.
/// Uses a short timeout (2s) to avoid blocking the response path.
pub(crate) async fn query_perplexity(
    server_url: &str,
    user_content: &str,
    assistant_content: &str,
) -> Option<f32> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()
        .ok()?;
    let url = format!("{}/perplexity", server_url);
    let body = serde_json::json!({
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ]
    });
    let resp = client.post(&url).json(&body).send().await.ok()?;
    if !resp.status().is_success() {
        return None;
    }
    let json: serde_json::Value = resp.json().await.ok()?;
    json.get("loss").and_then(|v| v.as_f64()).map(|v| v as f32)
}

/// Build ANE training config by auto-detecting model architecture from
/// `config.json`. Falls back to hardcoded Qwen3-1.7B if the active model
/// directory is not provided.
///
/// Returns `None` if no ANE-compatible model is available.
#[cfg(all(feature = "ane", feature = "mlx"))]
fn build_ane_training_config(
    model_dir: Option<&std::path::Path>,
) -> Option<crate::agent::ane_mlx_bridge::AneTrainingConfig> {
    use crate::agent::mlx_lora::ModelConfig;

    // Prefer the active inference model directory when available.
    if let Some(dir) = model_dir {
        if dir.join("config.json").exists() {
            let mc = ModelConfig::from_config_json(dir)?;
            return Some(crate::agent::ane_mlx_bridge::AneTrainingConfig {
                model_dir: dir.to_path_buf(),
                mil_config: mc.to_mil_config(64),
                epochs: 3,
                lr: 1e-5,
                linear_attn_indices: mc.linear_attn_indices.clone(),
                kv_dim: mc.n_kv_heads * mc.head_dim,
            });
        }
    }

    // Fallback: look for a known model in the LM Studio cache.
    let home = dirs::home_dir()?;
    let models_dir = home.join(".cache/lm-studio/models");
    let qwen3_1_7b = models_dir.join("lmstudio-community/Qwen3-1.7B-MLX-8bit");
    if qwen3_1_7b.join("config.json").exists() {
        let mc = ModelConfig::from_config_json(&qwen3_1_7b)?;
        return Some(crate::agent::ane_mlx_bridge::AneTrainingConfig {
            model_dir: qwen3_1_7b,
            mil_config: mc.to_mil_config(64),
            epochs: 3,
            lr: 1e-5,
            linear_attn_indices: mc.linear_attn_indices.clone(),
            kv_dim: mc.n_kv_heads * mc.head_dim,
        });
    }

    None
}

/// HTTP-only training (no model worker contention).
///
/// Sends training data to the mlx-lm server's `/train` endpoint with a short
/// timeout. This path is used when ANE split-silicon training is unavailable
/// and in-process MLX training would block inference.
async fn try_http_train(
    exps_data: &[(String, String)],
    ids: &[i64],
    eb_arc: &Arc<parking_lot::Mutex<ExperienceBuffer>>,
    epochs: usize,
    server_url: &str,
    train_counters: &Option<Arc<crate::agent::agent_core::RuntimeCounters>>,
) {
    // Signal training start.
    if let Some(ref tc) = train_counters {
        tc.training_active.store(true, Ordering::Relaxed);
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        tc.training_started_ms.store(now_ms, Ordering::Relaxed);
    }

    let conversations: Vec<serde_json::Value> = exps_data
        .iter()
        .map(|(p, r)| {
            serde_json::json!([
                {"role": "user", "content": p},
                {"role": "assistant", "content": r}
            ])
        })
        .collect();
    let body = serde_json::json!({"messages": conversations, "epochs": epochs});
    let url = format!("{}/train", server_url);
    let client = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
    {
        Ok(c) => c,
        Err(_) => {
            // Signal training done (failed to build client).
            if let Some(ref tc) = train_counters {
                tc.training_active.store(false, Ordering::Relaxed);
            }
            return;
        }
    };
    match client.post(&url).json(&body).send().await {
        Ok(resp) if resp.status().is_success() => {
            let eb = eb_arc.lock();
            let _ = eb.mark_exported(ids);
            if let Some(ref tc) = train_counters {
                tc.training_steps_total.fetch_add(1, Ordering::Relaxed);
            }
            info!(
                "perplexity_gate: triggered HTTP training with {} experiences",
                ids.len()
            );
        }
        Ok(resp) => debug!("perplexity_gate: /train returned {}", resp.status()),
        Err(e) => debug!("perplexity_gate: HTTP training failed: {e}"),
    }
    // Signal training done.
    if let Some(ref tc) = train_counters {
        tc.training_active.store(false, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test 1: TurnOutcome can be constructed from typical turn data.
    #[test]
    fn test_turn_outcome_construction() {
        let mut tools = HashSet::new();
        tools.insert("exec_command".to_string());

        let outcome = TurnOutcome {
            user_content: "hello".into(),
            final_content: "world".into(),
            model: "gpt-4".into(),
            session_key: "sess-1".into(),
            workspace: PathBuf::from("/tmp"),
            used_tools: tools,
            turn_tool_entries: vec![TurnToolEntry {
                name: "exec_command".into(),
                id: "call_1".into(),
                ok: true,
                duration_ms: 100,
                result_chars: 42,
            }],
            iterations_used: 2,
            max_iterations: 10,
            turn_count: 5,
            turn_start_elapsed_ms: 1500,
            context_tokens: 4000,
            message_count: 10,
            working_memory_tokens: 200,
            provenance_audit_enabled: true,
            is_local: false,
            cost_usd: 0.001,
            prompt_tokens: 3000,
            completion_tokens: 1000,
        };

        assert_eq!(outcome.user_content, "hello");
        assert!(!outcome.used_tools.is_empty());
        assert_eq!(outcome.turn_tool_entries.len(), 1);
        assert_eq!(outcome.model, "gpt-4");
    }

    // Stub struct that records calls for testing dispatch logic.
    struct MockLearnLoop {
        immediate_called: std::sync::atomic::AtomicBool,
        async_called: std::sync::atomic::AtomicBool,
    }

    impl MockLearnLoop {
        fn new() -> Self {
            Self {
                immediate_called: std::sync::atomic::AtomicBool::new(false),
                async_called: std::sync::atomic::AtomicBool::new(false),
            }
        }
    }

    impl LearnLoop for MockLearnLoop {
        fn observe_immediate(&self, _outcome: &TurnOutcome) {
            self.immediate_called
                .store(true, std::sync::atomic::Ordering::Relaxed);
        }
        fn observe_async(&self, _outcome: TurnOutcome) -> Option<tokio::task::JoinHandle<()>> {
            self.async_called
                .store(true, std::sync::atomic::Ordering::Relaxed);
            None
        }
    }

    // Test 2: MockLearnLoop trait impl compiles and can be swapped.
    #[test]
    fn test_mock_learn_loop_trait() {
        let mock = MockLearnLoop::new();
        let outcome = make_test_outcome();
        mock.observe_immediate(&outcome);
        assert!(mock.immediate_called.load(std::sync::atomic::Ordering::Relaxed));
    }

    // Test 3: LearnLoop can be used as trait object (dyn dispatch).
    #[test]
    fn test_learn_loop_as_trait_object() {
        let mock: Arc<dyn LearnLoop> = Arc::new(MockLearnLoop::new());
        let outcome = make_test_outcome();
        mock.observe_immediate(&outcome);
        let result = mock.observe_async(outcome);
        assert!(result.is_none());
    }

    // Test 4: DefaultLearnLoop::observe_immediate does not panic even with
    // no calibrator (None) and audit disabled.
    #[test]
    fn test_default_learn_loop_no_panic() {
        let ll = DefaultLearnLoop {
            calibrator: None,
            experience_buffer: None,
            perplexity_gate_config: PerplexityGateConfig::default(),
            #[cfg(feature = "mlx")]
            mlx_provider: None,
            training_counters: None,
        };
        let outcome = make_test_outcome();
        // Should not panic even with no calibrator and audit disabled.
        ll.observe_immediate(&outcome);
    }

    // Test 4b: DefaultLearnLoop classifies task types correctly.
    #[test]
    fn test_task_type_classification() {
        let mut outcome = make_test_outcome();
        assert_eq!(DefaultLearnLoop::task_type(&outcome), "chat");

        outcome.used_tools.insert("exec_command".into());
        assert_eq!(DefaultLearnLoop::task_type(&outcome), "shell");

        outcome.used_tools.clear();
        outcome.used_tools.insert("web_search".into());
        assert_eq!(DefaultLearnLoop::task_type(&outcome), "web_search");

        outcome.used_tools.clear();
        outcome.used_tools.insert("spawn_agent".into());
        assert_eq!(DefaultLearnLoop::task_type(&outcome), "delegate");

        outcome.used_tools.clear();
        outcome.used_tools.insert("read_file".into());
        assert_eq!(DefaultLearnLoop::task_type(&outcome), "tool_use");
    }

    // Test 5: observe_async returns a JoinHandle that completes without panic.
    #[tokio::test]
    async fn test_observe_async_completes() {
        // Use a mock that spawns actual async work.
        struct AsyncMock;
        impl LearnLoop for AsyncMock {
            fn observe_immediate(&self, _outcome: &TurnOutcome) {}
            fn observe_async(&self, _outcome: TurnOutcome) -> Option<tokio::task::JoinHandle<()>> {
                Some(tokio::spawn(async { /* no-op */ }))
            }
        }

        let mock = AsyncMock;
        let outcome = make_test_outcome();
        let handle = mock.observe_async(outcome);
        assert!(handle.is_some());
        handle.unwrap().await.expect("async observer should not panic");
    }

    fn make_test_outcome() -> TurnOutcome {
        TurnOutcome {
            user_content: "test".into(),
            final_content: "response".into(),
            model: "test-model".into(),
            session_key: "test-session".into(),
            workspace: PathBuf::from("/tmp"),
            used_tools: HashSet::new(),
            turn_tool_entries: vec![],
            iterations_used: 1,
            max_iterations: 10,
            turn_count: 1,
            turn_start_elapsed_ms: 100,
            context_tokens: 1000,
            message_count: 3,
            working_memory_tokens: 0,
            provenance_audit_enabled: false,
            is_local: false,
            cost_usd: 0.0,
            prompt_tokens: 0,
            completion_tokens: 0,
        }
    }
}
