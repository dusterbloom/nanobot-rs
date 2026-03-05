//! Phase 3 of message processing: finalize and emit the [`OutboundMessage`].
//!
//! Extracted from `agent_loop.rs` to keep that file focused on the iteration
//! state machine. This module contains only the response-finalization logic.

use std::sync::atomic::Ordering;

use chrono::Utc;
use serde_json::json;
use tracing::{debug, info, instrument, warn};

use crate::agent::agent_core::provenance_warning_role;
use crate::agent::agent_loop::{AgentLoopShared, TurnContext};
use crate::agent::token_budget::TokenBudget;
use crate::bus::events::OutboundMessage;

impl AgentLoopShared {
    /// Phase 3: Finalize the response — persist session, build outbound message.
    ///
    /// Consumes the `TurnContext` (by value) since this is the terminal phase.
    /// Stores context stats, writes audit summaries, verifies claims, and
    /// constructs the `OutboundMessage`.
    #[instrument(name = "finalize_response", skip(self, ctx), fields(
        session = %ctx.session_key,
        model = %ctx.core.model,
        iterations = ctx.iterations_used,
        tools_called = ctx.used_tools.len(),
        has_content = !ctx.final_content.is_empty(),
    ))]
    pub(crate) async fn finalize_response(&self, mut ctx: TurnContext) -> Option<OutboundMessage> {
        let counters = &self.core_handle.counters;

        // Store context stats for status bar.
        let final_tokens = TokenBudget::estimate_tokens(&ctx.messages) as u64;
        counters
            .last_context_used
            .store(final_tokens, Ordering::Relaxed);
        counters
            .last_context_max
            .store(ctx.core.token_budget.max_context() as u64, Ordering::Relaxed);
        counters
            .last_message_count
            .store(ctx.messages.len() as u64, Ordering::Relaxed);
        // Store working memory token count.
        let wm_tokens = if ctx.core.memory_enabled {
            let wm_text = ctx.core.working_memory.get_context(&ctx.session_key, usize::MAX);
            TokenBudget::estimate_str_tokens(&wm_text) as u64
        } else {
            0
        };
        counters
            .last_working_memory_tokens
            .store(wm_tokens, Ordering::Relaxed);
        // Store tools called this turn.
        {
            let tools_list: Vec<String> = ctx.used_tools.iter().cloned().collect();
            { let mut guard = counters.last_tools_called.lock();
                *guard = tools_list;
            }
        }

        // Perplexity gate: record experience and optionally trigger training.
        // Must run before audit write which moves turn_tool_entries.
        if self.perplexity_gate_config.enabled
            && !ctx.used_tools.is_empty()
            && !ctx.final_content.is_empty()
        {
            if let Some(ref eb_mutex) = self.experience_buffer {
                let tool_entries: Vec<serde_json::Value> = ctx
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
                    serde_json::to_string(&tool_entries).unwrap_or_else(|_| "[]".into());

                // Try in-process MLX perplexity first, then HTTP, then heuristic.
                let surprise = 'surprise: {
                    #[cfg(feature = "mlx")]
                    if let Some(ref mlx) = self.mlx_provider {
                        if let Ok(loss) = mlx.perplexity(&ctx.user_content, &ctx.final_content).await {
                            break 'surprise loss as f64;
                        }
                    }
                    match query_perplexity(
                        &self.perplexity_gate_config.mlx_server_url,
                        &ctx.user_content,
                        &ctx.final_content,
                    ).await {
                        Some(loss) => loss as f64,
                        None => crate::agent::lora_bridge::compute_surprise(
                            &ctx.user_content, &trace_json,
                        ),
                    }
                };

                let threshold = self.perplexity_gate_config.surprise_threshold as f64;
                if surprise > threshold {
                    // Record under lock, then release before any async work.
                    let should_train = {
                        let eb = eb_mutex.lock();
                        match eb.record_with_surprise(
                            &ctx.user_content,
                            &trace_json,
                            &ctx.final_content,
                            true,
                            1.0,
                            &ctx.core.model,
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
                        let min_exp = self.perplexity_gate_config.min_experiences;
                        eb.stats().map(|s| s.unexported as usize >= min_exp).unwrap_or(false)
                    }; // lock released

                    if should_train {
                        let epochs = self.perplexity_gate_config.train_epochs;
                        let min_exp = self.perplexity_gate_config.min_experiences;
                        let eb_arc = eb_mutex.clone();

                        // Prefer in-process MLX training (no HTTP), fall back to server.
                        #[cfg(feature = "mlx")]
                        let mlx_provider = self.mlx_provider.clone();
                        #[cfg(not(feature = "mlx"))]
                        let mlx_provider: Option<()> = None;

                        let server_url = self.perplexity_gate_config.mlx_server_url.clone();
                        tokio::spawn(async move {
                            // Collect experiences under lock.
                            let (exps_data, ids) = {
                                let eb = eb_arc.lock();
                                let exps = match eb.top_unexported(min_exp) {
                                    Ok(e) => e,
                                    Err(_) => return,
                                };
                                if exps.is_empty() { return; }
                                let data: Vec<(String, String)> = exps.iter()
                                    .map(|e| (e.prompt.clone(), e.response.clone()))
                                    .collect();
                                let ids: Vec<i64> = exps.iter().map(|e| e.id).collect();
                                (data, ids)
                            };

                            // Try in-process MLX training first.
                            #[cfg(feature = "mlx")]
                            if let Some(ref mlx) = mlx_provider {
                                let convos: Vec<Vec<crate::agent::mlx_server::ChatMessage>> =
                                    exps_data.iter().map(|(p, r)| vec![
                                        crate::agent::mlx_server::ChatMessage { role: "user".into(), content: p.clone() },
                                        crate::agent::mlx_server::ChatMessage { role: "assistant".into(), content: r.clone() },
                                    ]).collect();
                                match mlx.train(convos, epochs).await {
                                    Ok(()) => {
                                        let eb = eb_arc.lock();
                                        let _ = eb.mark_exported(&ids);
                                        info!("perplexity_gate: in-process training with {} experiences", ids.len());
                                        return;
                                    }
                                    Err(e) => debug!("perplexity_gate: in-process train failed: {e}"),
                                }
                            }

                            // Fall back to HTTP /train.
                            let conversations: Vec<serde_json::Value> = exps_data.iter().map(|(p, r)| {
                                serde_json::json!([
                                    {"role": "user", "content": p},
                                    {"role": "assistant", "content": r}
                                ])
                            }).collect();
                            let body = serde_json::json!({"messages": conversations, "epochs": epochs});
                            let url = format!("{}/train", server_url);
                            let client = match reqwest::Client::builder()
                                .timeout(std::time::Duration::from_secs(5)).build() {
                                Ok(c) => c, Err(_) => return,
                            };
                            match client.post(&url).json(&body).send().await {
                                Ok(resp) if resp.status().is_success() => {
                                    let eb = eb_arc.lock();
                                    let _ = eb.mark_exported(&ids);
                                    info!("perplexity_gate: triggered training with {} experiences", ids.len());
                                }
                                Ok(resp) => debug!("perplexity_gate: /train returned {}", resp.status()),
                                Err(e) => debug!("perplexity_gate: training trigger failed: {e}"),
                            }
                        });
                    }
                }
            }
        }

        // Write per-turn audit summary.
        if ctx.core.provenance_config.enabled && ctx.core.provenance_config.audit_log {
            let summary = crate::agent::audit::TurnSummary {
                turn: ctx.turn_count,
                timestamp: Utc::now().to_rfc3339(),
                context_tokens: final_tokens as usize,
                message_count: ctx.messages.len(),
                tools_called: ctx.turn_tool_entries,
                working_memory_tokens: wm_tokens as usize,
            };
            crate::agent::audit::write_turn_summary(&ctx.core.workspace, &ctx.session_key, &summary);
        }

        if ctx.final_content.is_empty() && ctx.messages.len() > 2 {
            // Try to surface the last meaningful assistant message as a rescue
            // rather than emitting a generic fallback. This happens when the
            // model exhausted iterations without producing a text-only response
            // (e.g. ended on a tool result with no follow-up assistant turn).
            let last_assistant = ctx.messages.iter().rev()
                .find(|m| m.get("role").and_then(|r| r.as_str()) == Some("assistant"))
                .and_then(|m| m.get("content").and_then(|c| c.as_str()))
                .unwrap_or("");
            if !last_assistant.trim().is_empty() {
                ctx.final_content = format!(
                    "{}\n\n[Note: Tool iteration limit reached. This response may be incomplete.]",
                    last_assistant.trim()
                );
            } else {
                ctx.final_content = "I ran out of tool iterations before producing a final answer. The actions above may be incomplete.".to_string();
            }
        }

        // Phase 3+4: Claim verification and context hygiene.
        if !ctx.final_content.is_empty()
            && ctx.core.provenance_config.enabled
            && ctx.core.provenance_config.verify_claims
        {
            if let Some(ref audit) = ctx.audit {
                let entries = audit.get_entries();
                let (claims, has_fabrication) =
                    crate::agent::provenance::verify_turn_claims(&ctx.final_content, &entries);

                if has_fabrication && ctx.core.provenance_config.strict_mode {
                    let (redacted, redaction_count) =
                        crate::agent::provenance::redact_fabrications(&ctx.final_content, &claims);
                    ctx.final_content = redacted;
                    if redaction_count > 0 {
                        let warning_role = provenance_warning_role(ctx.core.is_local);
                        let warning_content = format!(
                            "NOTICE: {} claim(s) in the previous response could not be \
                             verified against tool outputs and were removed.",
                            redaction_count
                        );
                        ctx.messages.push(json!({
                            "role": warning_role,
                            "content": warning_content
                        }));
                    }
                }
            }
        }

        // Phantom tool call detection: check if LLM claims tool results without calling tools.
        if !ctx.final_content.is_empty() && ctx.core.provenance_config.enabled {
            let tools_list: Vec<String> = ctx.used_tools.iter().cloned().collect();
            if let Some(detection) =
                crate::agent::provenance::detect_phantom_claims(&ctx.final_content, &tools_list)
            {
                warn!(
                    model = %ctx.core.model,
                    patterns = detection.matched_patterns.len(),
                    "phantom_tool_claims_detected: {:?}",
                    detection.matched_patterns
                );

                // Hard block: annotate the response so the user sees the warning.
                if ctx.core.provenance_config.strict_mode {
                    ctx.final_content = crate::agent::provenance::annotate_phantom_response(
                        &ctx.final_content,
                        &detection,
                    );
                }

                // Inject system reminder for the next turn.
                let warning_role = provenance_warning_role(ctx.core.is_local);
                ctx.messages.push(json!({
                    "role": warning_role,
                    "content": detection.system_warning
                }));
            }
        }

        // Ensure the final text response is in the messages array for persistence.
        // Without this, text-only responses (no tool calls) would be lost.
        // Bug 1 fix: if strip_dangling_tool_calls already converted a tool-call
        // assistant message into a plain text assistant message, merging here
        // prevents two consecutive assistant messages from being persisted.
        if !ctx.final_content.is_empty() {
            let last_is_assistant = ctx.messages.last()
                .and_then(|m| m.get("role").and_then(|r| r.as_str()))
                .map(|r| r == "assistant")
                .unwrap_or(false);
            if last_is_assistant {
                if let Some(last) = ctx.messages.last_mut() {
                    let existing = last.get("content").and_then(|c| c.as_str()).unwrap_or("").to_string();
                    last["content"] = json!(format!("{}\n\n{}", existing, ctx.final_content));
                }
            } else {
                ctx.messages.push(json!({"role": "assistant", "content": ctx.final_content.clone()}));
            }
        }

        // Update session history — persist full message array including tool calls.
        // Skip system prompt (index 0) and pre-existing history.
        let new_messages: Vec<serde_json::Value> = if ctx.new_start < ctx.messages.len() {
            ctx.messages[ctx.new_start..].to_vec()
        } else if !ctx.final_content.is_empty() {
            // User message already eagerly persisted (Bug 3 fix, line 274).
            // Only persist assistant response if present.
            vec![json!({"role": "assistant", "content": ctx.final_content.clone()})]
        } else {
            vec![]
        };
        if !new_messages.is_empty() {
            ctx.core.sessions
                .add_messages(&ctx.session_id, &new_messages)
                .await;
        }

        // Auto-complete stale working memory sessions (runs on every message, cheap).
        if ctx.core.memory_enabled {
            for session in ctx.core.working_memory.list_active() {
                if session.session_key != ctx.session_key {
                    let age = Utc::now() - session.updated;
                    if age.num_seconds() > ctx.core.session_complete_after_secs as i64 {
                        ctx.core.working_memory.complete(&session.session_key);
                        debug!("Auto-completed stale session: {}", session.session_key);
                    }
                }
            }

            // Clear current session's working memory if stale (no compaction in N turns).
            {
                let current = ctx.core.working_memory.get_or_create(&ctx.session_key);
                if !current.content.is_empty()
                    && current.last_updated_turn > 0
                    && ctx.turn_count.saturating_sub(current.last_updated_turn) > ctx.core.stale_memory_turn_threshold
                {
                    ctx.core.working_memory.clear(&ctx.session_key);
                    debug!(
                        "Cleared stale working memory for {} (last update turn {}, current turn {})",
                        ctx.session_key, current.last_updated_turn, ctx.turn_count
                    );
                }
            }
        }

        // Record execution stats for budget calibration (append-only, errors silently logged).
        if let Some(ref cal_mutex) = self.calibrator {
            let task_type = if ctx.used_tools.contains("exec_command") {
                "shell"
            } else if ctx.used_tools.contains("web_search") {
                "web_search"
            } else if ctx.used_tools.contains("spawn_agent") {
                "delegate"
            } else if ctx.used_tools.is_empty() {
                "chat"
            } else {
                "tool_use"
            };
            
            // Calculate actual cost from token usage.
            let prompt_tokens = counters.last_actual_prompt_tokens.load(Ordering::Relaxed) as i64;
            let completion_tokens = counters.last_actual_completion_tokens.load(Ordering::Relaxed) as i64;
            let cost_usd = if prompt_tokens > 0 || completion_tokens > 0 {
                let prices = crate::agent::model_prices::ModelPrices::load().await;
                prices.cost_of(&ctx.core.model, prompt_tokens, completion_tokens)
            } else {
                0.0
            };
            
            // Structured log for observability.
            info!(
                task_type = %task_type,
                iterations = ctx.iterations_used,
                tool_calls = ctx.used_tools.len(),
                prompt_tokens = prompt_tokens,
                completion_tokens = completion_tokens,
                cost_usd = format!("{:.6}", cost_usd),
                duration_ms = ctx.turn_start.elapsed().as_millis() as u64,
                success = !ctx.final_content.is_empty(),
                "turn_completed"
            );
            
            let record = crate::agent::budget_calibrator::ExecutionRecord {
                task_type: task_type.to_string(),
                model: ctx.core.model.clone(),
                iterations_used: ctx.iterations_used,
                max_iterations: ctx.core.max_iterations,
                success: !ctx.final_content.is_empty(),
                cost_usd,
                duration_ms: ctx.turn_start.elapsed().as_millis() as u64,
                depth: 0,
                tool_calls: ctx.used_tools.len() as u32,
                created_at: chrono::Utc::now().to_rfc3339(),
            };
            { let cal = cal_mutex.lock();
                if let Err(e) = cal.record(&record) {
                    tracing::debug!("BudgetCalibrator record failed: {}", e);
                }
            }
        }

        ctx.final_content =
            crate::agent::sanitize::sanitize_reasoning_output(&ctx.final_content);

        if ctx.final_content.is_empty() {
            None
        } else {
            let mut outbound = OutboundMessage::new(&ctx.channel, &ctx.chat_id, &ctx.final_content);
            // Propagate voice_message metadata so channels know to reply with voice.
            if ctx.is_voice_message {
                outbound
                    .metadata
                    .insert("voice_message".to_string(), json!(true));
            }
            // Propagate detected_language for TTS voice selection.
            if let Some(ref lang) = ctx.detected_language {
                outbound
                    .metadata
                    .insert("detected_language".to_string(), json!(lang));
            }
            Some(outbound)
        }
    }
}

/// Query the MLX server's `/perplexity` endpoint for real CE-loss surprise.
///
/// Returns `Some(loss)` on success, `None` if the server is unreachable or errors.
/// Uses a short timeout (2s) to avoid blocking the response path.
pub(crate) async fn query_perplexity(server_url: &str, user_content: &str, assistant_content: &str) -> Option<f32> {
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
