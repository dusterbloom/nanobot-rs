//! Unified learning dispatch for turn observations.
//!
//! Every learning/metrics observation flows through the [`LearnLoop`] trait
//! with immediate (sync) and async paths. This replaces scattered direct-write
//! observers in `finalize_response.rs`.

use std::collections::HashSet;
use std::path::PathBuf;
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
        // Only trigger perplexity gate when enabled, tools were used, and content exists.
        if !self.perplexity_gate_config.enabled
            || outcome.used_tools.is_empty()
            || outcome.final_content.is_empty()
        {
            return None;
        }

        let eb_mutex = self.experience_buffer.as_ref()?.clone();
        let pg_config = self.perplexity_gate_config.clone();

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

            // Try in-process MLX perplexity first, then HTTP, then heuristic.
            let surprise = 'surprise: {
                #[cfg(feature = "mlx")]
                if let Some(ref mlx) = mlx_provider {
                    if let Ok(loss) = mlx
                        .perplexity(&outcome.user_content, &outcome.final_content)
                        .await
                    {
                        break 'surprise loss as f64;
                    }
                }
                match query_perplexity(
                    &pg_config.mlx_server_url,
                    &outcome.user_content,
                    &outcome.final_content,
                )
                .await
                {
                    Some(loss) => loss as f64,
                    None => crate::agent::lora_bridge::compute_surprise(
                        &outcome.user_content,
                        &trace_json,
                    ),
                }
            };

            let threshold = pg_config.surprise_threshold as f64;
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
                if let Some(ane_cfg) = build_ane_training_config() {
                    let tokenizer = match crate::agent::mlx_lora::MlxTokenizer::load(
                        std::path::Path::new(&ane_cfg.model_dir),
                    ) {
                        Ok(t) => t,
                        Err(e) => {
                            debug!("perplexity_gate: ANE tokenizer load failed: {e}");
                            // Fall through to MLX GPU training below
                            #[cfg(feature = "mlx")]
                            {
                                try_mlx_or_http_train(
                                    &mlx_provider,
                                    &exps_data,
                                    &ids,
                                    &eb_mutex,
                                    epochs,
                                    &pg_config.mlx_server_url,
                                )
                                .await;
                            }
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
                        let _handle = crate::agent::ane_mlx_bridge::spawn_ane_training(
                            ane_cfg, samples, mlx_tx,
                        );
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

            // Try in-process MLX GPU training, then HTTP fallback.
            try_mlx_or_http_train(
                #[cfg(feature = "mlx")]
                &mlx_provider,
                #[cfg(not(feature = "mlx"))]
                &None::<()>,
                &exps_data,
                &ids,
                &eb_mutex,
                epochs,
                &pg_config.mlx_server_url,
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

/// Build ANE training config from the local model directory.
///
/// Returns `None` if no ANE-compatible model is available.
#[cfg(all(feature = "ane", feature = "mlx"))]
fn build_ane_training_config() -> Option<crate::agent::ane_mlx_bridge::AneTrainingConfig> {
    use crate::agent::ane_mil::MilConfig;

    let home = dirs::home_dir()?;
    let models_dir = home.join(".cache/lm-studio/models");

    // Try Qwen3-1.7B (standard transformer -- fully ANE-compatible)
    let qwen3_1_7b = models_dir.join("lmstudio-community/Qwen3-1.7B-MLX-8bit");
    if qwen3_1_7b.join("tokenizer.json").exists() {
        return Some(crate::agent::ane_mlx_bridge::AneTrainingConfig {
            model_dir: qwen3_1_7b,
            mil_config: MilConfig {
                dim: 2048,
                hidden_dim: 6144,
                n_heads: 16,
                seq_len: 64,
                n_kv_heads: 8,
                rope_theta: 1_000_000.0,
                rms_eps: 1e-6,
                has_lm_head: false,
            },
            epochs: 3,
            lr: 1e-5,
            linear_attn_indices: vec![], // standard transformer, no GDN
            kv_dim: 8 * 128,            // n_kv_heads=8, head_dim=128
        });
    }

    None
}

/// Try in-process MLX GPU training, then fall back to HTTP /train endpoint.
#[allow(unused_variables)]
async fn try_mlx_or_http_train(
    #[cfg(feature = "mlx")] mlx_provider: &Option<
        Arc<crate::providers::mlx::MlxProvider>,
    >,
    #[cfg(not(feature = "mlx"))] mlx_provider: &Option<()>,
    exps_data: &[(String, String)],
    ids: &[i64],
    eb_arc: &Arc<parking_lot::Mutex<ExperienceBuffer>>,
    epochs: usize,
    server_url: &str,
) {
    #[cfg(feature = "mlx")]
    if let Some(ref mlx) = mlx_provider {
        let convos: Vec<Vec<crate::agent::mlx_server::ChatMessage>> = exps_data
            .iter()
            .map(|(p, r)| {
                vec![
                    crate::agent::mlx_server::ChatMessage {
                        role: "user".into(),
                        content: p.clone(),
                    },
                    crate::agent::mlx_server::ChatMessage {
                        role: "assistant".into(),
                        content: r.clone(),
                    },
                ]
            })
            .collect();
        match mlx.train(convos, epochs).await {
            Ok(()) => {
                let eb = eb_arc.lock();
                let _ = eb.mark_exported(ids);
                info!(
                    "perplexity_gate: in-process training with {} experiences",
                    ids.len()
                );
                return;
            }
            Err(e) => debug!("perplexity_gate: in-process train failed: {e}"),
        }
    }

    // Fall back to HTTP /train.
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
        Err(_) => return,
    };
    match client.post(&url).json(&body).send().await {
        Ok(resp) if resp.status().is_success() => {
            let eb = eb_arc.lock();
            let _ = eb.mark_exported(ids);
            info!(
                "perplexity_gate: triggered training with {} experiences",
                ids.len()
            );
        }
        Ok(resp) => debug!("perplexity_gate: /train returned {}", resp.status()),
        Err(e) => debug!("perplexity_gate: training trigger failed: {e}"),
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
