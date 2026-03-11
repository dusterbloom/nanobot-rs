//! Phase 3 of message processing: finalize and emit the [`OutboundMessage`].
//!
//! Extracted from `agent_loop.rs` to keep that file focused on the iteration
//! state machine. This module contains only the response-finalization logic.
//!
//! All learning/metrics observations are dispatched through [`LearnLoop`]
//! (see `learn_loop.rs`). Zero direct audit/calibrator/perplexity writes here.

use std::sync::atomic::Ordering;

use chrono::Utc;
use serde_json::json;
use tracing::{debug, instrument, warn};

use crate::agent::agent_core::provenance_warning_role;
use crate::agent::agent_loop::{AgentLoopShared, TurnContext};
use crate::agent::learn_loop::TurnOutcome;
use crate::agent::token_budget::TokenBudget;
use crate::bus::events::OutboundMessage;

impl AgentLoopShared {
    /// Phase 3: Finalize the response -- persist session, build outbound message.
    ///
    /// Consumes the `TurnContext` (by value) since this is the terminal phase.
    /// Stores context stats, dispatches learning observations via LearnLoop,
    /// verifies claims, and constructs the `OutboundMessage`.
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
        counters.last_context_max.store(
            ctx.core.token_budget.max_context() as u64,
            Ordering::Relaxed,
        );
        counters
            .last_message_count
            .store(ctx.messages.len() as u64, Ordering::Relaxed);
        // Store working memory token count.
        let wm_tokens = if ctx.core.memory_enabled {
            let wm_text = ctx
                .core
                .working_memory
                .get_context(&ctx.session_key, usize::MAX);
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
            {
                let mut guard = counters.last_tools_called.lock();
                *guard = tools_list;
            }
        }

        // Pre-compute cost from token counters (async ModelPrices::load).
        let prompt_tokens = counters.last_actual_prompt_tokens.load(Ordering::Relaxed) as i64;
        let completion_tokens = counters
            .last_actual_completion_tokens
            .load(Ordering::Relaxed) as i64;
        let cost_usd = if prompt_tokens > 0 || completion_tokens > 0 {
            let prices = crate::agent::model_prices::ModelPrices::load().await;
            prices.cost_of(&ctx.core.model, prompt_tokens, completion_tokens)
        } else {
            0.0
        };

        // Save turn_tool_entries before post-processing (moved from TurnOutcome construction
        // so observers see final content after rescue/sanitization).
        let turn_tool_entries = std::mem::take(&mut ctx.turn_tool_entries);

        if ctx.final_content.is_empty() && ctx.messages.len() > 2 {
            // Try to surface the last meaningful assistant message as a rescue
            // rather than emitting a generic fallback. This happens when the
            // model exhausted iterations without producing a text-only response
            // (e.g. ended on a tool result with no follow-up assistant turn).
            let last_assistant = ctx
                .messages
                .iter()
                .rev()
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
            let last_is_assistant = ctx
                .messages
                .last()
                .and_then(|m| m.get("role").and_then(|r| r.as_str()))
                .map(|r| r == "assistant")
                .unwrap_or(false);
            if last_is_assistant {
                if let Some(last) = ctx.messages.last_mut() {
                    let existing = last
                        .get("content")
                        .and_then(|c| c.as_str())
                        .unwrap_or("")
                        .to_string();
                    last["content"] = json!(format!("{}\n\n{}", existing, ctx.final_content));
                }
            } else {
                ctx.messages
                    .push(json!({"role": "assistant", "content": ctx.final_content.clone()}));
            }
        }

        // Update session history -- persist full message array including tool calls.
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
            ctx.core
                .sessions
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
                    && ctx.turn_count.saturating_sub(current.last_updated_turn)
                        > ctx.core.stale_memory_turn_threshold
                {
                    ctx.core.working_memory.clear(&ctx.session_key);
                    debug!(
                        "Cleared stale working memory for {} (last update turn {}, current turn {})",
                        ctx.session_key, current.last_updated_turn, ctx.turn_count
                    );
                }
            }
        }

        // Capture reasoning trace BEFORE sanitization strips <think> blocks.
        // Training data benefits from seeing the model's reasoning process.
        let reasoning_trace = if ctx.final_content.contains("<think>") {
            Some(ctx.final_content.clone())
        } else {
            None
        };

        ctx.final_content = crate::agent::sanitize::sanitize_reasoning_output(&ctx.final_content);

        // Construct TurnOutcome AFTER all post-processing (rescue, provenance,
        // sanitization) so observers see the content the user actually receives.
        let outcome = TurnOutcome {
            user_content: ctx.user_content.clone(),
            final_content: ctx.final_content.clone(),
            reasoning_trace,
            turn_messages: new_messages.clone(),
            model: ctx.core.model.clone(),
            session_key: ctx.session_key.clone(),
            workspace: ctx.core.workspace.clone(),
            used_tools: ctx.used_tools.clone(),
            turn_tool_entries,
            iterations_used: ctx.iterations_used,
            max_iterations: ctx.core.max_iterations,
            turn_count: ctx.turn_count,
            turn_start_elapsed_ms: ctx.turn_start.elapsed().as_millis() as u64,
            context_tokens: final_tokens,
            message_count: ctx.messages.len(),
            working_memory_tokens: wm_tokens,
            provenance_audit_enabled: ctx.core.provenance_config.enabled
                && ctx.core.provenance_config.audit_log,
            is_local: ctx.core.is_local,
            cost_usd,
            prompt_tokens,
            completion_tokens,
        };

        // Synchronous observers: audit log, budget calibrator, structured logging.
        self.learn_loop.observe_immediate(&outcome);

        // Async observers: perplexity gate, LoRA training (fire-and-forget).
        self.learn_loop.observe_async(outcome);

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
