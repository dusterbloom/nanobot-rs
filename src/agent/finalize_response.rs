// Error-protocol layer-3 backlog (docs/research/2026-08-06-error-conventions-and-host-bridge.md §3.6):
// the deny regime in Cargo.toml is live; this module still carries pre-existing
// violations of the lints below. Remove this allow as the module migrates onto
// the regime.
// Tracking: docs/error-protocol-backlog.md
#![allow(clippy::as_conversions, clippy::indexing_slicing)]
//! Phase 3 of message processing: finalize and emit the [`OutboundMessage`].
//!
//! Extracted from `agent_loop.rs` to keep that file focused on the iteration
//! state machine. This module contains only the response-finalization logic.

use std::sync::atomic::Ordering;

use serde_json::json;
use tracing::{info, instrument, warn};

use crate::agent::agent_loop::{AgentLoopShared, TurnContext};
use crate::agent::token_budget::TokenBudget;
use crate::bus::events::OutboundMessage;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum IncompleteTurnReason {
    Cancelled,
    IterationLimit,
    Unknown,
}

impl AgentLoopShared {
    /// Phase 3: Finalize the response -- persist session, build outbound message.
    ///
    /// Consumes the `TurnContext` (by value) since this is the terminal phase.
    /// Stores context stats, dispatches learning updates via LearnLoop,
    /// verifies claims, and constructs the `OutboundMessage`.
    #[instrument(name = "finalize_response", skip(self, ctx), fields(
        session = %ctx.session_key,
        model = %ctx.core.model,
        iterations = ctx.iterations_used,
        tool_calls_executed = ctx.turn_tool_entries.len(),
        unique_tools_used = ctx.used_tools.len(),
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
                .get_context(&ctx.session_id, usize::MAX)
                .await
                .unwrap_or_else(|error| {
                    warn!(%error, session_id = %ctx.session_id, "working-memory stats lookup failed");
                    String::new()
                });
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

        let mut rescued_incomplete_assistant = false;
        if ctx.final_content.is_empty() && ctx.messages.len() > 2 {
            let reason = incomplete_turn_reason(&ctx);
            if let Some(rescued) = rescue_incomplete_response(&ctx.messages, reason) {
                ctx.final_content = rescued;
                rescued_incomplete_assistant = true;
            }
        }

        // Phase 3+4: Claim verification and context hygiene.
        if !ctx.final_content.is_empty()
            && !rescued_incomplete_assistant
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
                        // Cache-replay tagged: a warning the model saw live must
                        // replay byte-identical on reload or the warm prompt
                        // prefix diverges. Force `user` role (matches the
                        // response-boundary nudge convention at shared.rs:1193)
                        // — Anthropic's OpenAI-compat layer strips mid-thread
                        // system messages anyway.
                        let warning_content = format!(
                            "NOTICE: {} claim(s) in the previous response could not be \
                             verified against tool outputs and were removed.",
                            redaction_count
                        );
                        ctx.messages
                            .push(crate::agent::markers::scaffold_user(warning_content));
                    }
                }
            }
        }

        // Phantom tool call detection: check if LLM claims tool results without calling tools.
        if !ctx.final_content.is_empty()
            && !rescued_incomplete_assistant
            && ctx.core.provenance_config.enabled
        {
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

                // Inject warning for the next turn. Cache-replay tagged;
                // user role for the same reason as the fabrication warning
                // above.
                ctx.messages.push(crate::agent::markers::scaffold_user(
                    detection.system_warning,
                ));
            }
        }

        // Ensure the final text response is in the messages array for persistence.
        // Without this, text-only responses (no tool calls) would be lost.
        // Bug 1 fix: if strip_dangling_tool_calls already converted a tool-call
        // assistant message into a plain text assistant message, merging here
        // prevents two consecutive assistant messages from being persisted.
        if !ctx.final_content.is_empty() {
            let last_can_absorb_final_text = ctx
                .messages
                .last()
                .map(|m| {
                    let is_assistant = m.get("role").and_then(|r| r.as_str()) == Some("assistant");
                    let has_tool_calls = m
                        .get("tool_calls")
                        .and_then(|v| v.as_array())
                        .map(|calls| !calls.is_empty())
                        .unwrap_or(false);
                    is_assistant && !has_tool_calls
                })
                .unwrap_or(false);
            if last_can_absorb_final_text {
                if let Some(last) = ctx.messages.last_mut() {
                    let existing = last
                        .get("content")
                        .and_then(|c| c.as_str())
                        .unwrap_or("")
                        .to_string();
                    last["content"] = if existing.trim().is_empty() {
                        json!(ctx.final_content.clone())
                    } else {
                        json!(format!("{}\n\n{}", existing, ctx.final_content))
                    };
                }
            } else {
                ctx.messages
                    .push(json!({"role": "assistant", "content": ctx.final_content.clone()}));
            }
        }

        // Update session history -- persist full message array including tool calls.
        // Skip system prompt (index 0) and pre-existing history.
        //
        // Cancellation rescue: when the user cancelled mid-stream, the messages
        // array may contain a partial (truncated) assistant response. If we
        // persist it, the next turn's fingerprint will diverge at that truncated
        // message, forcing a full re-prefill. Strip it so the next call starts
        // from the last complete user message instead.
        if ctx.final_content.is_empty() && ctx.is_cancelled() {
            // Find and remove the partial assistant response (if any).
            // The user message was already eagerly persisted (Bug 3 fix), so
            // we only need to discard the incomplete assistant turn.
            let mut end = ctx.messages.len();
            while end > ctx.new_start {
                if let Some(role) = ctx.messages[end - 1].get("role").and_then(|r| r.as_str()) {
                    if role == "assistant" {
                        end -= 1; // remove the partial assistant
                    } else {
                        break; // hit a user/tool message — stop
                    }
                } else {
                    break;
                }
            }
            // Truncate the messages array to remove the partial response.
            ctx.messages.truncate(end);
        }

        // Final text is the only normal message that has not already been
        // checkpointed in the active tool path. This also retries any earlier
        // SQLite failure without double-inserting successful rows.
        ctx.persist_pending_protocol_messages().await;

        ctx.final_content = crate::agent::sanitize::sanitize_reasoning_output(&ctx.final_content);

        let turn_outcome = if ctx.is_cancelled() {
            "cancelled"
        } else if ctx.final_content.is_empty() {
            "empty"
        } else {
            "finished"
        };
        // The reply is already generated and persisted by this point. A
        // failed turn-finish journal write must not discard it: warn and
        // deliver, letting the session's replay degrade to Incomplete.
        if let Err(error) = ctx
            .core
            .sessions
            .record_turn_finished(
                &ctx.session_id,
                &ctx.request_id,
                ctx.turn_count,
                turn_outcome,
            )
            .await
        {
            warn!(
                %error,
                session = %ctx.session_key,
                "turn_finished_replay_persist_failed; replay degrades to incomplete"
            );
        }

        let cache = counters.session_cache_metrics(&ctx.session_id);
        info!(
            logical_session = %ctx.session_id,
            calls = cache.calls,
            prompt_tokens = cache.prompt_tokens,
            cache_read_tokens = cache.cache_read_tokens,
            cache_creation_tokens = cache.cache_creation_tokens,
            cold_calls = cache.cold_calls,
            efficiency_pct = cache.efficiency_pct(),
            "cache_efficiency"
        );

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

fn incomplete_turn_reason(ctx: &TurnContext) -> IncompleteTurnReason {
    if ctx.is_cancelled() {
        IncompleteTurnReason::Cancelled
    } else if ctx.core.max_iterations > 0 && ctx.iterations_used >= ctx.core.max_iterations {
        IncompleteTurnReason::IterationLimit
    } else {
        IncompleteTurnReason::Unknown
    }
}

fn rescue_incomplete_response(
    messages: &[serde_json::Value],
    reason: IncompleteTurnReason,
) -> Option<String> {
    let note = incomplete_response_note(reason)?;
    let last_assistant = messages
        .iter()
        .rev()
        .find(|m| m.get("role").and_then(|r| r.as_str()) == Some("assistant"))
        .and_then(|m| m.get("content").and_then(|c| c.as_str()))
        .unwrap_or("");

    if !last_assistant.trim().is_empty() {
        Some(format!("{}\n\n{}", last_assistant.trim(), note))
    } else {
        Some(match reason {
            IncompleteTurnReason::IterationLimit => {
                "I ran out of tool iterations before producing a final answer. The actions above may be incomplete.".to_string()
            }
            IncompleteTurnReason::Unknown => {
                "The turn ended before I could produce a final answer. The actions above may be incomplete.".to_string()
            }
            IncompleteTurnReason::Cancelled => return None,
        })
    }
}

fn incomplete_response_note(reason: IncompleteTurnReason) -> Option<&'static str> {
    match reason {
        IncompleteTurnReason::Cancelled => None,
        IncompleteTurnReason::IterationLimit => {
            Some("[Note: Tool iteration limit reached. This response may be incomplete.]")
        }
        IncompleteTurnReason::Unknown => {
            Some("[Note: The turn ended before a final answer was produced. This response may be incomplete.]")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn cancelled_turn_does_not_rescue_or_claim_iteration_limit() {
        let messages = vec![
            json!({"role": "user", "content": "question"}),
            json!({"role": "assistant", "content": "Let me check that.", "tool_calls": []}),
            json!({"role": "tool", "content": "result"}),
        ];

        assert_eq!(
            rescue_incomplete_response(&messages, IncompleteTurnReason::Cancelled),
            None
        );
    }

    #[test]
    fn unknown_incomplete_turn_uses_generic_note() {
        let messages = vec![
            json!({"role": "user", "content": "question"}),
            json!({"role": "assistant", "content": "Let me check that.", "tool_calls": []}),
            json!({"role": "tool", "content": "result"}),
        ];

        let rescued = rescue_incomplete_response(&messages, IncompleteTurnReason::Unknown).unwrap();
        assert!(rescued.contains("Let me check that."));
        assert!(rescued.contains("ended before a final answer"));
        assert!(!rescued.contains("Tool iteration limit reached"));
    }

    #[test]
    fn iteration_limit_note_is_only_for_iteration_limit() {
        let messages = vec![
            json!({"role": "user", "content": "question"}),
            json!({"role": "assistant", "content": "Working."}),
            json!({"role": "tool", "content": "result"}),
        ];

        let rescued =
            rescue_incomplete_response(&messages, IncompleteTurnReason::IterationLimit).unwrap();
        assert!(rescued.contains("Tool iteration limit reached"));
    }
}
