// Error-protocol layer-3 backlog (docs/research/2026-08-06-error-conventions-and-host-bridge.md §3.6):
// the deny regime in Cargo.toml is live; this module still carries pre-existing
// violations of the lints below. Remove this allow as the module migrates onto
// the regime.
// Tracking: docs/error-protocol-backlog.md
#![allow(clippy::as_conversions, clippy::shadow_reuse)]
//! Response classification and handler methods for `step_process_response`.
//!
//! Extracted from `agent_shared.rs` as a `#[path]` submodule.
//!
//! `ResponseKind` is a pure enum produced by `classify_response()` — no IO,
//! fully unit-testable. Each variant maps to a small handler method on
//! `AgentLoopShared` that performs the recovery action.

use std::collections::HashMap;
use std::sync::atomic::Ordering;

use serde_json::{json, Value};
use tracing::{debug, error, info, warn};

use crate::agent::anti_drift;
use crate::agent::protocol::{
    parse_textual_tool_calls, parse_xml_tool_calls, strip_textual_tool_calls, strip_xml_tool_calls,
};
use crate::agent::token_budget::TokenBudget;
use crate::agent::validation;
use crate::errors::ProviderError;
use crate::providers::base::{FinishReason, LLMResponse, ToolCallRequest};
use crate::session::db::{ModelCallPurpose, RecordedProviderRequest, RecordedProviderResponse};
use crate::turn_stream::ControlMarker;

use super::{AgentLoopShared, IterationOutcome, IterationPhase, StepResult, TurnContext};

async fn recorded_auxiliary_chat(
    ctx: &TurnContext,
    purpose: ModelCallPurpose,
    messages: &[Value],
    max_tokens: u32,
    temperature: f64,
) -> anyhow::Result<LLMResponse> {
    let request = RecordedProviderRequest {
        messages: messages.to_vec(),
        tools: None,
        model: ctx.core.model.clone(),
        max_tokens,
        temperature,
        thinking_budget: None,
        top_p: None,
        tool_choice: "auto".to_string(),
        streaming: false,
    };
    let call_id = ctx
        .core
        .sessions
        .record_model_request(
            &ctx.session_id,
            &ctx.request_id,
            ctx.turn_count,
            purpose,
            &request,
        )
        .await
        .map_err(|error| anyhow::anyhow!("auxiliary model request was not recorded: {error}"))?;
    let result = ctx
        .core
        .provider
        .chat(
            messages,
            None,
            Some(&ctx.core.model),
            max_tokens,
            temperature,
            None,
            None,
        )
        .await;
    match result {
        Ok(response) => {
            ctx.core
                .sessions
                .record_model_response(
                    &ctx.session_id,
                    &ctx.request_id,
                    ctx.turn_count,
                    &call_id,
                    &RecordedProviderResponse::from(&response),
                )
                .await
                .map_err(|error| {
                    anyhow::anyhow!("auxiliary model response was not recorded: {error}")
                })?;
            Ok(response)
        }
        Err(error) => {
            if let Err(record_error) = ctx
                .core
                .sessions
                .record_model_failure(
                    &ctx.session_id,
                    &ctx.request_id,
                    ctx.turn_count,
                    &call_id,
                    &error.to_string(),
                )
                .await
            {
                warn!(%record_error, "auxiliary_model_failure_replay_persist_failed");
            }
            Err(error)
        }
    }
}

// ---------------------------------------------------------------------------
// ResponseKind — pure classification of an LLM response
// ---------------------------------------------------------------------------

/// Classified response from the LLM, produced by [`classify_response`].
///
/// Each variant carries only the data needed for its handler — no cloning of
/// the full `LLMResponse` unless the handler genuinely needs it.
#[derive(Debug)]
pub(crate) enum ResponseKind {
    /// Response has native or text-parsed tool calls to execute.
    ToolCalls { tool_calls: Vec<ToolCallRequest> },
    /// Response contains visible text, no tool calls — final answer.
    Text(String),
    /// Validation failed (hallucinated tool call or claimed-but-not-executed).
    /// Carries the error and the raw content for retry-hint injection.
    ValidationError {
        error: validation::ValidationError,
        raw_content: String,
    },
    /// Provider returned an error detail in the response body.
    ProviderError(String),
    /// Local model completed, but the payload is malformed protocol scaffolding
    /// or a degenerate repetition loop. Treat it as failed, not as a user-visible answer.
    PathologicalLocalOutput { reason: &'static str },
    /// Response was truncated (finish_reason=length) with non-empty content.
    Truncated(String),
    /// Response is empty after thinking consumed the entire output budget.
    /// One-shot retry with thinking disabled.
    EmptyAfterThink,
    /// Response is completely empty and we already retried — inject fallback.
    EmptyFinal,
}

// ---------------------------------------------------------------------------
// RetryState — typed counters replacing scattered booleans
// ---------------------------------------------------------------------------

/// Per-turn retry budgets. Each failure mode has a named counter with its own
/// cap, replacing the loose booleans that were scattered across `FlowControl`.
pub(crate) struct RetryState {
    /// Consecutive validation retries (hallucinated tool calls, claimed-but-not-executed).
    /// Capped at `MAX_VALIDATION_RETRIES`. Does NOT consume a main-loop iteration.
    pub(crate) validation: u32,
    /// Auto-continuations for truncated responses (finish_reason=length).
    pub(crate) continuations: u32,
    /// One-shot: already retried an empty response with thinking disabled.
    pub(crate) empty_think_retried: bool,
    /// One-shot: already attempted the rescue pass (forced finalize).
    pub(crate) rescue_attempted: bool,
    /// One-shot: agent-level retry for transient LLM errors (per iteration).
    pub(crate) api_retried: bool,
    /// Server rejected the request with context_length_exceeded and we
    /// trimmed + retried this many times already this turn. Capped (see
    /// MAX_OVERFLOW_RECOVERIES): each attempt must strictly shrink the
    /// context, but a hard cap bounds the prefill cost of pathological
    /// shrink-overflow-shrink cycles.
    pub(crate) overflow_trim_recoveries: u32,
    /// Consecutive lease-renewal rejections (the model emitted a PARTIAL
    /// checkpoint — some labels, missing a field — and was nudged what to
    /// add). Without a cap a small model that keeps emitting partial
    /// checkpoints loops forever: each rejection returns to PreCall, which
    /// advances neither `max_iterations` nor the no-progress counter. After
    /// `MAX_LEASE_RENEWAL_REJECTIONS` the turn finishes with whatever text
    /// the model produced — renewal is a privilege, not a right.
    pub(crate) lease_renewal_rejections: u32,
}

impl RetryState {
    pub(crate) fn new() -> Self {
        Self {
            validation: 0,
            continuations: 0,
            empty_think_retried: false,
            rescue_attempted: false,
            api_retried: false,
            overflow_trim_recoveries: 0,
            lease_renewal_rejections: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// classify_response — pure function, no IO
// ---------------------------------------------------------------------------

/// Classify an LLM response into a [`ResponseKind`].
///
/// This is a pure function: it reads the response and context flags but
/// performs no IO. All recovery actions are deferred to handler methods.
pub(crate) fn classify_response(
    response: &LLMResponse,
    is_local: bool,
    is_textual_replay: bool,
    had_blocked_calls: bool,
    retries: &RetryState,
    thinking_was_on: bool,
) -> ResponseKind {
    // Provider error takes absolute priority (error-protocol doc §2.3).
    if let Err(ProviderError::EmptyStream(detail)) = response.outcome() {
        return ResponseKind::ProviderError(detail);
    }

    let content = response.content.as_deref().unwrap_or("");
    let has_native_tools = response.has_tool_calls();
    let has_visible_text = !content.trim().is_empty();

    // Check for text-embedded tool calls (bracket or XML format) when no
    // native tool_calls exist. We only *detect* here — the actual parsing
    // and stripping happens in the handler so `response` stays immutable.
    // The XML path includes a lenient fallback that recovers TRUNCATED markup
    // (opening tags with no closer), so this returns true for partially-formed
    // tool calls too.
    let has_textual_tools = !has_native_tools
        && has_visible_text
        && (has_bracket_tool_calls(content) || has_xml_tool_calls(content));

    // Pathological discard only when there is NO recoverable tool call: a
    // textual tool call — even truncated and recovered by the lenient parser —
    // is real work, not degenerate output, and must not be thrown away
    // (2026-07-30: truncated `<tool_call>` markup was discarded here ×4,
    // forcing the user to re-roll until the model emitted a native call).
    if is_local && !has_native_tools && !has_textual_tools {
        if let Some(reason) = pathological_local_output_reason(content) {
            return ResponseKind::PathologicalLocalOutput { reason };
        }
    }

    // If there are tool calls (native or textual), validate first.
    if has_native_tools || has_textual_tools {
        // Validation only applies to native tool calls (textual ones are
        // parsed *after* classification, in the handler).
        if has_native_tools {
            let tool_maps = tool_calls_to_maps(&response.tool_calls);
            match validation::validate_response(
                content,
                &tool_maps,
                is_textual_replay,
                had_blocked_calls,
            ) {
                validation::ValidationOutcome::Error(e) => {
                    return ResponseKind::ValidationError {
                        error: e,
                        raw_content: content.to_string(),
                    };
                }
                validation::ValidationOutcome::StripHallucination
                | validation::ValidationOutcome::Ok => {
                    // StripHallucination is handled in the ToolCalls handler.
                }
            }
        }
        return ResponseKind::ToolCalls {
            tool_calls: response.tool_calls.clone(),
        };
    }

    // No tool calls — check for validation errors on pure-text responses.
    let tool_maps: Vec<HashMap<String, Value>> = vec![];
    match validation::validate_response(content, &tool_maps, is_textual_replay, had_blocked_calls) {
        validation::ValidationOutcome::Error(e) => {
            return ResponseKind::ValidationError {
                error: e,
                raw_content: content.to_string(),
            };
        }
        validation::ValidationOutcome::StripHallucination | validation::ValidationOutcome::Ok => {}
    }

    // Empty response handling.
    if !has_visible_text {
        // Local model + truncated + thinking consumed output → rescue or retry.
        if is_local && response.finish_reason == FinishReason::Length && !retries.rescue_attempted {
            return ResponseKind::EmptyAfterThink;
        }
        if thinking_was_on && !retries.empty_think_retried {
            return ResponseKind::EmptyAfterThink;
        }
        return ResponseKind::EmptyFinal;
    }

    // A provider length stop is transport truncation even if a marker-like
    // suffix survived. Continue/retry it before considering finality.
    let is_truncated = response.finish_reason == FinishReason::Length;
    if is_truncated && retries.continuations < 10 {
        // Only classify as Truncated if there's room to continue.
        // (Actual cap is checked in the handler via core.max_continuations.)
        return ResponseKind::Truncated(content.to_string());
    }

    // Plain non-empty text with no tool calls is the final answer.
    ResponseKind::Text(content.to_string())
}

// ---------------------------------------------------------------------------
// Channel helpers (avoids type-inference issues in #[path] submodules)
// ---------------------------------------------------------------------------

fn send_finish_reason(tx: &Option<tokio::sync::mpsc::UnboundedSender<String>>, reason: &str) {
    if let Some(ref tx) = tx {
        let _ = tx.send(ControlMarker::FinishReason(reason.to_string()).encode());
    }
}

/// Stream the provider-reported completion-token count to the REPL renderer.
/// The renderer accumulates these across a turn's LLM calls to show tok/s.
fn send_token_count(tx: &Option<tokio::sync::mpsc::UnboundedSender<String>>, tokens: u64) {
    if let Some(ref tx) = tx {
        let _ = tx.send(ControlMarker::Tokens(tokens).encode());
    }
}

/// Stream the real decode time (ms) for the just-finished LLM call: the call's
/// wall time minus its prefill (ttft). Renderers sum these across the turn to
/// report a true decode tok/s that excludes tool-execution and re-prefill gaps
/// between calls. Skipped when ttft is unknown (non-streaming call) — the
/// renderer then falls back to `wall − first_ttft`.
fn send_decode_time(
    tx: &Option<tokio::sync::mpsc::UnboundedSender<String>>,
    call_start: Option<std::time::Instant>,
    ttft_ms: Option<u64>,
) {
    let (Some(tx), Some(start), Some(ttft)) = (tx, call_start, ttft_ms) else {
        return;
    };
    let decode_ms = (start.elapsed().as_millis() as u64).saturating_sub(ttft);
    if decode_ms > 0 {
        let _ = tx.send(ControlMarker::DecodeMs(decode_ms).encode());
    }
}

/// Stream the provider-reported prompt-token count to the TUI. This lets the
/// UI compute effective prefill throughput even when the local server does not
/// stream prompt_progress chunks.
fn send_prompt_token_count(tx: &Option<tokio::sync::mpsc::UnboundedSender<String>>, tokens: u64) {
    if let Some(ref tx) = tx {
        let _ = tx.send(ControlMarker::PromptTokens(tokens).encode());
    }
}

/// Maximum auto-continuations when the turn's output is being spoken (voice/TTS).
/// A rambling local model would otherwise re-synthesize dozens of chunks.
const VOICE_MAX_CONTINUATIONS: u32 = 4;

/// Normalize whitespace and case for loose repetition comparison.
fn normalize_for_repeat(s: &str) -> String {
    s.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

/// True when an auto-continuation adds nothing new: it is empty, identical to
/// the previous continuation (`prev_norm`), or a sizeable span already present
/// in `accumulated`. Small local models that hit `finish_reason="length"` tend
/// to re-emit the same sentence; without this guard the continue loop runs to
/// the cap, re-synthesizing every repeat to TTS.
fn is_degenerate_continuation(new: &str, prev_norm: &str, accumulated: &str) -> bool {
    let norm = normalize_for_repeat(new);
    if norm.is_empty() || norm == prev_norm {
        return true;
    }
    norm.chars().count() >= 24 && normalize_for_repeat(accumulated).contains(&norm)
}

fn send_delta(tx: &Option<tokio::sync::mpsc::UnboundedSender<String>>, text: &str) {
    if let Some(ref tx) = tx {
        let _ = tx.send(text.to_string());
    }
}

fn send_retract_reply(tx: &Option<tokio::sync::mpsc::UnboundedSender<String>>) {
    if let Some(ref tx) = tx {
        let _ = tx.send(ControlMarker::RetractReply.encode());
    }
}

fn pathological_local_output_reason(content: &str) -> Option<&'static str> {
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return None;
    }

    let lower = trimmed.to_ascii_lowercase();
    if lower.contains("<tool_call") || lower.contains("<tool_code") || lower.contains("</tool_code")
    {
        return Some("malformed_tool_markup");
    }

    let mut prev = '\0';
    let mut run = 0usize;
    for ch in trimmed.chars() {
        if ch == prev {
            run += 1;
        } else {
            prev = ch;
            run = 1;
        }
        if run >= 160 && is_degenerate_repeated_char(ch) {
            return Some("repeated_character_loop");
        }
    }

    None
}

fn is_degenerate_repeated_char(ch: char) -> bool {
    ch.is_ascii_punctuation() || ch == '\u{2588}' || ch == '\u{259e}'
}

fn is_cancelled(token: &Option<tokio_util::sync::CancellationToken>) -> bool {
    token.as_ref().map_or(false, |t| t.is_cancelled())
}

// ---------------------------------------------------------------------------
// Helper predicates (pure)
// ---------------------------------------------------------------------------

fn has_bracket_tool_calls(content: &str) -> bool {
    !parse_textual_tool_calls(content).is_empty()
}

fn has_xml_tool_calls(content: &str) -> bool {
    content.contains("<tool_call>") && !parse_xml_tool_calls(content).is_empty()
}

fn tool_calls_to_maps(tool_calls: &[ToolCallRequest]) -> Vec<HashMap<String, Value>> {
    tool_calls
        .iter()
        .map(|tc| {
            let mut map = HashMap::new();
            map.insert("id".to_string(), Value::String(tc.id.clone()));
            map.insert("name".to_string(), Value::String(tc.name.clone()));
            map.insert(
                "arguments".to_string(),
                Value::Object(
                    tc.arguments
                        .iter()
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect(),
                ),
            );
            map
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Handler methods on AgentLoopShared
// ---------------------------------------------------------------------------

impl AgentLoopShared {
    /// Unified entry point: classify the response and dispatch to the appropriate handler.
    ///
    /// Replaces the old monolithic `step_process_response`.
    pub(crate) async fn step_process_response(
        &self,
        ctx: &mut TurnContext,
        mut response: LLMResponse,
    ) -> StepResult {
        let counters = &self.core_handle.counters;

        // --- Pre-classification mutations ---
        // Extract textual tool calls before classification so the response
        // object is in a consistent state for classify_response().
        self.extract_textual_tool_calls(&mut response);

        // Strip thinking tags leaked by models (Qwen3, MiniCPM, etc.)
        if let Some(ref mut content) = response.content {
            let cleaned = crate::agent::compaction::strip_thinking_tags(content);
            if cleaned.len() != content.len() {
                *content = cleaned;
            }
        }

        // Anti-drift post-completion: collapse babble (before classification).
        // migrated from swappable().is_local — phase 09-03
        if ctx.core.mode().needs_anti_drift()
            && ctx.core.retention.anti_drift.enabled
            && !response.has_tool_calls()
        {
            if let Some(ref mut content) = response.content {
                anti_drift::post_completion_pipeline(
                    content,
                    &ctx.messages,
                    &ctx.core.retention.anti_drift,
                );
            }
        }

        // --- Classify ---
        let thinking_was_on = counters.thinking_budget.load(Ordering::Relaxed) > 0;
        // migrated from swappable().is_local — phase 09-03
        let kind = classify_response(
            &response,
            ctx.core.mode().is_local(),
            ctx.protocol.is_textual_replay(),
            ctx.flow.tool_guard.had_blocked_calls,
            &ctx.flow.retries,
            thinking_was_on,
        );

        // --- Token telemetry (always, regardless of kind) ---
        let defers_metrics = matches!(&kind, ResponseKind::ToolCalls { .. });
        self.emit_token_telemetry(ctx, &response, defers_metrics);
        // Forward the completion-token count to the REPL footer. Sent per LLM
        // call so the renderer can accumulate the turn total and report tok/s.
        let completion_tokens = response
            .usage
            .get("completion_tokens")
            .copied()
            .unwrap_or(-1);
        if completion_tokens > 0 {
            send_token_count(&ctx.text_delta_tx, completion_tokens as u64);
            // Pair the token count with this call's real decode time (call wall
            // time − prefill), so renderers report a true decode tok/s that
            // excludes the tool-execution and re-prefill time between calls.
            send_decode_time(
                &ctx.text_delta_tx,
                ctx.flow.llm_call_start,
                ctx.flow.ttft_ms,
            );
        }

        // --- Dispatch ---
        match kind {
            ResponseKind::ToolCalls { tool_calls: _ } => {
                // Validation may have flagged StripHallucination — clean up.
                if let Some(ref mut content) = response.content {
                    let stripped = validation::strip_hallucinated_text(content);
                    if stripped.len() != content.len() {
                        debug!("Stripping hallucinated tool-call text from response");
                        *content = stripped;
                    }
                }
                StepResult::Next(IterationPhase::Executing { response })
            }

            ResponseKind::Text(content) => {
                // Tool-lease renewal: when the lease was exhausted and the model
                // emitted a structured checkpoint (findings + next + will),
                // renew the lease and continue with tools restored. The
                // checkpoint text is left in the conversation so the user
                // can see what the model committed to.
                //
                // Renewal is the only "synthetic injection" the lease
                // design needs: it confirms to the model that tools are
                // back, then loops back to step_pre_call where tool_defs
                // are recomputed (no longer exhausted). No hidden state.
                if ctx.flow.lease.is_exhausted() && !content.is_empty() {
                    let renewal = ctx.flow.lease.try_renew(&content);
                    if renewal.is_valid() {
                        tracing::info!(
                            session = %ctx.session_key,
                            renewals_used = ctx.flow.lease.renewals_used(),
                            "tool_lease_renewed"
                        );
                        ctx.flow.retries.lease_renewal_rejections = 0;
                        // Nudge the model so it knows tools are available
                        // again. Without this, a small local model may
                        // emit another text answer instead of tool calls
                        // even though tool_defs are back on the next call.
                        ctx.messages
                            .push(crate::agent::markers::scaffold_user(format!(
                                "[Lease renewed — {} more tool calls available. \
                                 Proceed with the plan from your checkpoint.]",
                                ctx.flow.lease.lease_size()
                            )));
                        return StepResult::Next(IterationPhase::PreCall);
                    } else if renewal.was_attempted() && renewal.missing_field() != "out_of_leases"
                    {
                        // Renewal attempted but missing a field. Tell the
                        // model exactly what's missing so the next attempt
                        // can succeed — narrow, deterministic, visible. Bound
                        // it: a model that keeps emitting partial checkpoints
                        // gets only MAX_LEASE_RENEWAL_REJECTIONS nudges, then
                        // the turn finishes with the text it produced. (The
                        // constant lives with the other loop bounds in
                        // agent_loop::shared — see its ordering invariant.)
                        ctx.flow.retries.lease_renewal_rejections =
                            ctx.flow.retries.lease_renewal_rejections.saturating_add(1);
                        if ctx.flow.retries.lease_renewal_rejections
                            > super::shared::MAX_LEASE_RENEWAL_REJECTIONS
                        {
                            tracing::info!(
                                session = %ctx.session_key,
                                rejections = ctx.flow.retries.lease_renewal_rejections,
                                "tool_lease_renewal_rejection_cap_reached — finishing turn with model text"
                            );
                            // Fall through to finish: the model's text is its answer.
                        } else {
                            tracing::info!(
                                session = %ctx.session_key,
                                missing = renewal.missing_field(),
                                rejection = ctx.flow.retries.lease_renewal_rejections,
                                "tool_lease_renewal_rejected"
                            );
                            ctx.messages
                                .push(crate::agent::markers::scaffold_user(format!(
                                    "[Lease renewal rejected — your checkpoint is missing \
                                     '{}'. Either include all of findings:/next:/will: to \
                                     renew, or write your final answer.]",
                                    renewal.missing_field()
                                )));
                            return StepResult::Next(IterationPhase::PreCall);
                        }
                    }
                    // Fall through: either out_of_leases, the renewal-rejection
                    // cap was reached, or the model wrote plain text with no
                    // checkpoint labels (try_renew returned not_attempted). All
                    // mean "finish the turn" — log only a genuine out_of_leases,
                    // not the cap/not_attempted cases it would otherwise mask.
                    if renewal.was_attempted() && renewal.missing_field() == "out_of_leases" {
                        tracing::info!(
                            session = %ctx.session_key,
                            "tool_lease_renewal_out_of_leases"
                        );
                    }
                }
                // Provenance is observe-only here — it never retracts a
                // completed response and never re-prompts.
                //
                // This used to retract the streamed text and loop back to
                // PreCall when the phrase list matched with zero tool calls
                // this turn. That gate is wrong for retrospective questions
                // ("what went wrong last turn?"), which are answered by
                // narrating EARLIER tool calls in past tense and legitimately
                // need no new tools. In session 20260810_081050_8306f8 it
                // discarded a 2637-token answer that took 9m35s to generate,
                // on the substring "I executed", and never persisted it.
                //
                // A phantom now costs a warning header, not the answer:
                // `finalize_response` annotates via `annotate_phantom_response`
                // and the response is delivered. Detection stays; the blast
                // radius is gone.
                if ctx.core.provenance_config.enabled
                    && ctx.used_tools.is_empty()
                    && crate::agent::provenance::detect_phantom_claims(&content, &[]).is_some()
                {
                    counters
                        .phantom_claims_observed
                        .fetch_add(1, Ordering::Relaxed);
                    tracing::info!(
                        session = %ctx.session_key,
                        model = %ctx.core.model,
                        "phantom_tool_claims_observed — response delivered with annotation"
                    );
                }
                if !ctx.flow.content_was_streamed {
                    send_delta(&ctx.text_delta_tx, &content);
                    ctx.flow.content_was_streamed = true;
                }
                send_finish_reason(&ctx.text_delta_tx, response.finish_reason.wire_str());
                StepResult::Done(IterationOutcome::Finished(content))
            }

            ResponseKind::ValidationError { error, raw_content } => {
                self.handle_validation_error(ctx, &error, &raw_content)
            }

            ResponseKind::ProviderError(err_msg) => self.handle_provider_error(ctx, &err_msg).await,

            ResponseKind::PathologicalLocalOutput { reason } => {
                warn!(
                    model = %ctx.core.model,
                    reason,
                    raw_response = ?response.content.as_deref(),
                    "pathological_local_output_discarded"
                );
                if ctx.flow.content_was_streamed {
                    send_retract_reply(&ctx.text_delta_tx);
                    ctx.flow.content_was_streamed = false;
                }
                send_finish_reason(&ctx.text_delta_tx, response.finish_reason.wire_str());
                StepResult::Done(IterationOutcome::Error(
                    "The local model produced malformed or repetitive protocol text, so I discarded that response. Try the request again; if it repeats, restart the local backend.".to_string(),
                ))
            }

            ResponseKind::Truncated(partial) => {
                let full = self.handle_truncated(ctx, &response, partial).await;
                match full.trim() {
                    "" => StepResult::Done(IterationOutcome::Finished(String::new())),
                    _ => {
                        send_finish_reason(&ctx.text_delta_tx, "stop");
                        StepResult::Done(IterationOutcome::Finished(full))
                    }
                }
            }

            ResponseKind::EmptyAfterThink => {
                self.handle_empty_after_think(ctx, &response, counters)
                    .await
            }

            ResponseKind::EmptyFinal => {
                warn!(
                    finish_reason = %response.finish_reason,
                    "empty_llm_response: SLM returned no content and no tool calls, injecting fallback"
                );
                let content =
                    "I couldn't produce a response in this turn. Please try again.".to_string();
                send_finish_reason(&ctx.text_delta_tx, response.finish_reason.wire_str());
                StepResult::Done(IterationOutcome::Finished(content))
            }
        }
    }

    // -----------------------------------------------------------------------
    // Pre-classification: extract textual tool calls
    // -----------------------------------------------------------------------

    fn extract_textual_tool_calls(&self, response: &mut LLMResponse) {
        if response.has_tool_calls() {
            return;
        }
        let content_text = match response.content.as_deref() {
            Some(c) if !c.trim().is_empty() => c,
            _ => return,
        };

        let has_xml_blocks = content_text.contains("<tool_call>");

        let parsed = parse_textual_tool_calls(content_text);
        let parsed = if parsed.is_empty() {
            parse_xml_tool_calls(content_text)
        } else {
            parsed
        };

        // Even when no valid tool calls were parsed, strip empty/malformed
        // <tool_call> blocks from the content.  Leaving them in pollutes the
        // conversation history and confuses the model into repeating the same
        // broken XML format on subsequent iterations.
        if parsed.is_empty() {
            if has_xml_blocks {
                if let Some(ref mut content) = response.content {
                    let cleaned = strip_xml_tool_calls(content);
                    if cleaned.len() != content.len() {
                        debug!(
                            "Stripped empty/malformed <tool_call> blocks from response \
                             (no valid tool calls parsed)"
                        );
                        *content = cleaned;
                    }
                }
            }
            return;
        }

        let is_xml = has_xml_blocks;
        debug!(
            n = parsed.len(),
            format = if is_xml { "xml" } else { "textual" },
            "universal_tool_parse: parsed {} tool call(s) from response text",
            parsed.len()
        );

        let synthesised: Vec<ToolCallRequest> = parsed
            .into_iter()
            .enumerate()
            .map(|(i, ptc)| {
                let args: HashMap<String, Value> = match ptc.args {
                    Value::Object(map) => map.into_iter().collect(),
                    _ => HashMap::new(),
                };
                ToolCallRequest {
                    id: format!("tc_textual_{}", i + 1),
                    name: ptc.tool,
                    arguments: args,
                }
            })
            .collect();

        if let Some(ref mut content) = response.content {
            *content = if is_xml {
                strip_xml_tool_calls(content)
            } else {
                strip_textual_tool_calls(content)
            };
        }
        response.tool_calls = synthesised;
    }

    // -----------------------------------------------------------------------
    // Handler: validation error → inject retry hint
    // -----------------------------------------------------------------------

    fn handle_validation_error(
        &self,
        ctx: &mut TurnContext,
        error: &validation::ValidationError,
        raw_content: &str,
    ) -> StepResult {
        let retry_num = ctx.flow.retries.validation + 1;
        warn!(
            model = %ctx.core.model,
            validation = %format!("{:?}", error),
            retry = retry_num,
            max_retries = validation::MAX_VALIDATION_RETRIES,
            "response_validation_failed"
        );

        // The phantom text is already on the user's screen — a narrated
        // `[exec(command='date')]` reads exactly like an executed call. Retract
        // it so the retry (or the give-up message below) replaces it instead of
        // trailing it. Without this the post-loop emit is suppressed by
        // `content_was_streamed` and the phantom stands as the final answer
        // (higgs + lfm2-2.6b, which never emits tool_calls at all).
        if ctx.flow.content_was_streamed {
            send_retract_reply(&ctx.text_delta_tx);
            ctx.flow.content_was_streamed = false;
        }

        // Both phantom shapes give up after one failed retry: a local model
        // that narrates a call twice narrates it forever, and each further
        // round burns a real iteration for nothing.
        if ctx.flow.retries.validation > 0 {
            warn!(
                model = %ctx.core.model,
                retry = retry_num,
                "response_validation_claimed_tool_intent_repeated"
            );
            return StepResult::Done(IterationOutcome::Error(
                "I could not complete the tool step: the model narrated a tool action instead of emitting a structured tool call, twice in a row. Nothing was executed. If this repeats, the model or endpoint likely does not support native function calling."
                    .to_string(),
            ));
        }

        let hint = validation::generate_retry_prompt(error, retry_num as u8);

        // Keep the fabricated-call text in history so the retry hint below has
        // an antecedent (and the wire keeps alternating roles).
        ctx.messages.push(json!({
            "role": "assistant",
            "content": raw_content
        }));

        // Cache-replay tagged: a validation hint sent live must survive
        // session reload byte-identical, otherwise the warm prompt prefix
        // diverges and Higgs re-prefills the whole context.
        ctx.messages
            .push(crate::agent::markers::scaffold_user(hint));
        debug!(
            "Injected validation retry hint (retry {}/{})",
            retry_num,
            validation::MAX_VALIDATION_RETRIES
        );
        StepResult::Done(IterationOutcome::ValidationRetry)
    }

    // -----------------------------------------------------------------------
    // Handler: provider error
    // -----------------------------------------------------------------------

    async fn handle_provider_error(&self, ctx: &TurnContext, err_msg: &str) -> StepResult {
        error!(model = %ctx.core.model, error = %err_msg, "llm_provider_error");

        // migrated from swappable().is_local — phase 09-03
        if ctx.core.mode().is_local() {
            if let Some(base) = ctx.core.provider.get_api_base() {
                if !crate::server::check_health(base, ctx.core.health_check_timeout_secs).await {
                    error!("Local LLM server is down!");
                    return StepResult::Done(IterationOutcome::Error(
                        "[LLM Error] Local server crashed. Use /restart or /local to recover."
                            .into(),
                    ));
                }
            }
        }

        StepResult::Done(IterationOutcome::Error(format!("[LLM Error] {}", err_msg)))
    }

    // -----------------------------------------------------------------------
    // Handler: truncated response → auto-continue loop
    // -----------------------------------------------------------------------

    async fn handle_truncated(
        &self,
        ctx: &mut TurnContext,
        original_response: &LLMResponse,
        mut accumulated: String,
    ) -> String {
        let counters = &self.core_handle.counters;
        if ctx.core.mode().is_local() {
            info!(
                finish_reason = %original_response.finish_reason,
                "auto_continue_skipped_local: preserving prefix cache instead of issuing hidden Continue prompt"
            );
            return accumulated;
        }

        // Voice/TTS turns cap continuations hard: a rambling local model would
        // otherwise re-synthesize dozens of chunks. `is_voice_message` covers
        // channel voice notes; `suppress_thinking_in_tts` covers REPL voice mode.
        let voice_active =
            ctx.is_voice_message || counters.suppress_thinking_in_tts.load(Ordering::Relaxed);
        let max_cont = if voice_active {
            ctx.core.max_continuations.min(VOICE_MAX_CONTINUATIONS)
        } else {
            ctx.core.max_continuations
        };
        let mut finish_reason = original_response.finish_reason.clone();
        let mut prev_norm = String::new();

        while ctx.flow.retries.continuations < max_cont {
            // Check if still truncated.
            let is_truncated = finish_reason == FinishReason::Length
                || (finish_reason == FinishReason::Stop && super::appears_incomplete(&accumulated));
            if !is_truncated {
                break;
            }

            ctx.flow.retries.continuations += 1;
            if finish_reason == FinishReason::Stop {
                info!("auto_continue: heuristic detected incomplete response despite finish_reason='stop'");
            }
            info!(
                "auto_continue: continuation {}/{} — finish_reason was '{}'",
                ctx.flow.retries.continuations, max_cont, finish_reason
            );

            // Streaming indicator.
            send_delta(&ctx.text_delta_tx, "\x1b[2m [continuing...]\x1b[0m");

            let mut cont_messages = ctx.messages.clone();
            cont_messages.push(json!({
                "role": "assistant",
                "content": &accumulated
            }));
            cont_messages.push(json!({
                "role": "user",
                "content": "Continue."
            }));

            if is_cancelled(&ctx.cancellation_token) {
                break;
            }

            counters.mark_inference_started();
            let cont_result = recorded_auxiliary_chat(
                ctx,
                ModelCallPurpose::Continuation,
                &cont_messages,
                ctx.core.max_tokens,
                ctx.core.temperature,
            )
            .await;
            counters.mark_inference_finished();

            match cont_result {
                Ok(cont_response) => {
                    let continuation = cont_response.content.clone().unwrap_or_default();
                    // Stop if the model is repeating itself rather than adding new content.
                    if is_degenerate_continuation(&continuation, &prev_norm, &accumulated) {
                        info!(
                            "auto_continue: degenerate/repeated continuation at {}/{}, stopping early",
                            ctx.flow.retries.continuations, max_cont
                        );
                        break;
                    }
                    prev_norm = normalize_for_repeat(&continuation);
                    if !continuation.is_empty() {
                        send_delta(&ctx.text_delta_tx, &continuation);
                    }
                    accumulated.push_str(&continuation);
                    finish_reason = cont_response.finish_reason;
                }
                Err(e) => {
                    warn!("auto_continue: continuation call failed: {}", e);
                    break;
                }
            }
        }

        accumulated
    }

    // -----------------------------------------------------------------------
    // Handler: empty response after thinking
    // -----------------------------------------------------------------------

    async fn handle_empty_after_think(
        &self,
        ctx: &mut TurnContext,
        response: &LLMResponse,
        counters: &crate::agent::agent_core::RuntimeCounters,
    ) -> StepResult {
        // Try rescue pass first (forced finalize for local models).
        // migrated from swappable().is_local — phase 09-03
        if ctx.core.mode().is_local()
            && response.finish_reason == FinishReason::Length
            && !ctx.flow.retries.rescue_attempted
        {
            ctx.flow.retries.rescue_attempted = true;
            let rescue_tokens = ctx.core.max_tokens.min(384).max(128);
            let rescue_messages = prepare_rescue_messages(&ctx.messages, &*ctx.protocol);
            counters.mark_inference_started();
            let rescue_result = recorded_auxiliary_chat(
                ctx,
                ModelCallPurpose::EmptyResponseRescue,
                &rescue_messages,
                rescue_tokens,
                0.2,
            )
            .await;
            counters.mark_inference_finished();

            match rescue_result {
                Ok(r) => {
                    let content = r.content.unwrap_or_default();
                    if !content.trim().is_empty() {
                        send_finish_reason(&ctx.text_delta_tx, r.finish_reason.wire_str());
                        return StepResult::Done(IterationOutcome::Finished(content));
                    }
                    // Rescue also empty — fall through to thinking-off retry.
                }
                Err(e) => {
                    warn!("Finalize rescue call failed: {}", e);
                }
            }
        }

        // Retry with thinking temporarily disabled — restore after this iteration.
        if !ctx.flow.retries.empty_think_retried {
            ctx.flow.retries.empty_think_retried = true;
            let saved = counters.thinking_budget.load(Ordering::Relaxed);
            warn!(
                finish_reason = %response.finish_reason,
                saved_budget = saved,
                "empty_llm_response: thinking consumed entire output, retrying with thinking off"
            );
            counters.thinking_budget.store(0, Ordering::Relaxed);
            ctx.flow.restore_thinking_budget = Some(saved);
            return StepResult::Done(IterationOutcome::Continue);
        }

        // Both rescue and thinking-off retry exhausted — fallback.
        warn!(
            finish_reason = %response.finish_reason,
            "empty_llm_response: all recovery attempts exhausted, injecting fallback"
        );
        let content = "I couldn't produce a response in this turn. Please try again.".to_string();
        send_finish_reason(&ctx.text_delta_tx, response.finish_reason.wire_str());
        StepResult::Done(IterationOutcome::Finished(content))
    }

    // -----------------------------------------------------------------------
    // Token telemetry
    // -----------------------------------------------------------------------

    /// Classify a completed LLM call for metrics. A call that yields no
    /// content AND no tool calls is not "ok" — it is the signature of a dead
    /// stream (observed: 600s zero-token Higgs stall recorded as ok), and
    /// hiding it makes provider health invisible in metrics.jsonl.
    pub(super) fn response_status(response: &LLMResponse) -> &'static str {
        let no_content = response
            .content
            .as_deref()
            .map_or(true, |c| c.trim().is_empty());
        if response.outcome().is_err()
            || matches!(
                response.finish_reason,
                FinishReason::Aborted | FinishReason::Cancelled
            )
        {
            "error"
        } else if no_content && response.tool_calls.is_empty() {
            "empty_response"
        } else if Self::raw_pathological_response(response).is_some() {
            "pathological_response"
        } else {
            "ok"
        }
    }

    fn raw_pathological_response(response: &LLMResponse) -> Option<&str> {
        if !response.tool_calls.is_empty() {
            return None;
        }
        response
            .content
            .as_deref()
            .filter(|content| pathological_local_output_reason(content).is_some())
    }

    fn emit_token_telemetry(
        &self,
        ctx: &mut TurnContext,
        response: &LLMResponse,
        defer_until_tool_execution: bool,
    ) {
        let counters = &self.core_handle.counters;
        let estimated_prompt = ctx
            .flow
            .provider_prompt_estimate
            .unwrap_or_else(|| TokenBudget::estimate_tokens(&ctx.messages));
        let actual_prompt = response.usage.get("prompt_tokens").copied().unwrap_or(-1);
        let actual_completion = response
            .usage
            .get("completion_tokens")
            .copied()
            .unwrap_or(-1);
        // Provider prompt-cache telemetry (Anthropic/Zhipu native names). Absent
        // or zero on providers that don't report caching → None, so the JSONL
        // field is omitted and cache hit-rate is measurable per-provider.
        let cache_read_tokens = response
            .usage
            .get("cache_read_input_tokens")
            .copied()
            .filter(|&n| n > 0)
            .map(|n| n as u64);
        let cache_creation_tokens = response
            .usage
            .get("cache_creation_input_tokens")
            .copied()
            .filter(|&n| n > 0)
            .map(|n| n as u64);
        info!(
            "tokens: estimated_prompt={}, actual_prompt={}, actual_completion={}, cache_read={}, cache_creation={}",
            estimated_prompt,
            actual_prompt,
            actual_completion,
            cache_read_tokens.unwrap_or(0),
            cache_creation_tokens.unwrap_or(0),
        );
        // Definitive cache-served diagnostic for local servers. higgs/llama.cpp
        // may report prompt-cache hits under a non-OpenAI key (e.g. a native
        // timings field or prompt_cache_hit_tokens) that
        // extract_usage_numbers doesn't fold into cache_read_input_tokens.
        // Dumping the raw usage map answers "is the prefix cache being served
        // at all" without guessing from a zero cache_read. Cross-reference with
        // the request-side `higgs_session_cache_request` log.
        if ctx.core.model.starts_with("local:") {
            debug!(
                model = %ctx.core.model,
                usage = ?response.usage,
                "local_llm_raw_usage"
            );
        }
        if actual_prompt > 0 {
            counters
                .last_actual_prompt_tokens
                .store(actual_prompt as u64, Ordering::Relaxed);
            send_prompt_token_count(&ctx.text_delta_tx, actual_prompt as u64);
        }
        if actual_completion > 0 {
            counters
                .last_actual_completion_tokens
                .store(actual_completion as u64, Ordering::Relaxed);
        }
        counters
            .last_estimated_prompt_tokens
            .store(estimated_prompt as u64, Ordering::Relaxed);

        let metrics = crate::agent::metrics::RequestMetrics {
            timestamp: chrono::Local::now().to_rfc3339(),
            request_id: ctx.request_id.clone(),
            logical_session: ctx.session_id.clone(),
            cache_route: ctx.higgs_session_route.cache_route().into(),
            role: "main".into(),
            model: ctx.core.model.clone(),
            provider_base: ctx.core.provider.get_api_base().unwrap_or("unknown").into(),
            elapsed_ms: ctx
                .flow
                .llm_call_start
                .map_or(0, |t| t.elapsed().as_millis() as u64),
            ttft_ms: ctx.flow.ttft_ms,
            prompt_tokens: actual_prompt.max(0) as u64,
            completion_tokens: actual_completion.max(0) as u64,
            cache_read_tokens,
            cache_creation_tokens,
            status: Self::response_status(response).into(),
            error_detail: match response.outcome() {
                Err(ProviderError::EmptyStream(detail)) => Some(detail),
                _ => None,
            },
            raw_response: Self::raw_pathological_response(response).map(str::to_owned),
            anti_drift_score: None,
            anti_drift_signals: None,
            tool_calls_requested: response.tool_calls.len() as u32,
            tool_calls_executed: 0,
            validation_result: None,
        };
        counters.record_cache_metrics(
            &metrics.logical_session,
            metrics.prompt_tokens,
            metrics.cache_read_tokens,
            metrics.cache_creation_tokens,
        );
        if defer_until_tool_execution {
            debug_assert!(ctx.flow.pending_request_metrics.is_none());
            ctx.flow.pending_request_metrics = Some(metrics);
        } else {
            crate::agent::metrics::emit(&metrics);
        }
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// Render raw messages through the conversation protocol, then append
/// the rescue prompt. This ensures local servers never receive invalid
/// roles like "tool" or "developer" in the rescue call.
fn prepare_rescue_messages(
    raw_messages: &[Value],
    protocol: &dyn crate::agent::protocol::ConversationProtocol,
) -> Vec<Value> {
    let mut rendered = super::render_via_protocol(protocol, raw_messages);
    rendered.push(json!({
        "role": "user",
        "content": "Return the final answer now. No reasoning. No tool calls. Max 6 lines."
    }));
    rendered
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_response_status_flags_dead_streams() {
        let resp = |content: Option<&str>,
                    tool_calls: Vec<crate::providers::base::ToolCallRequest>,
                    finish_reason: &str| {
            LLMResponse {
                content: content.map(str::to_string),
                tool_calls,
                finish_reason: FinishReason::parse_finish_reason(finish_reason),
                usage: std::collections::HashMap::new(),
            }
        };
        let tc = crate::providers::base::ToolCallRequest {
            id: "tc1".into(),
            name: "list_dir".into(),
            arguments: std::collections::HashMap::new(),
        };

        // The 600s zero-token stall signature: nothing at all.
        assert_eq!(
            AgentLoopShared::response_status(&resp(None, vec![], "stop")),
            "empty_response"
        );
        assert_eq!(
            AgentLoopShared::response_status(&resp(Some("  \n"), vec![], "stop")),
            "empty_response"
        );
        assert_eq!(
            AgentLoopShared::response_status(&resp(
                Some("provider stream ended without content"),
                vec![],
                "error"
            )),
            "error"
        );
        assert_eq!(
            AgentLoopShared::response_status(&resp(None, vec![], "error")),
            "error"
        );
        assert_eq!(
            AgentLoopShared::response_status(&resp(Some("cancelled"), vec![], "cancelled")),
            "error"
        );
        assert_eq!(
            AgentLoopShared::response_status(&resp(Some("aborted"), vec![], "aborted")),
            "error"
        );
        // Legitimate outcomes stay ok.
        assert_eq!(
            AgentLoopShared::response_status(&resp(Some("hi"), vec![], "stop")),
            "ok"
        );
        assert_eq!(
            AgentLoopShared::response_status(&resp(None, vec![tc], "stop")),
            "ok"
        );
        assert_eq!(
            AgentLoopShared::response_status(&resp(
                Some("<tool_call>\n<tool_code>"),
                vec![],
                "stop"
            )),
            "pathological_response"
        );
        assert_eq!(
            AgentLoopShared::response_status(&resp(Some(&"!".repeat(180)), vec![], "stop")),
            "pathological_response"
        );
    }

    #[test]
    fn test_classify_pathological_local_output() {
        let resp = make_response(Some("<tool_call>\n<tool_code>\n!!!!!!!!"), "stop");
        let kind = classify_response(&resp, true, false, false, &default_retries(), false);
        assert!(matches!(
            kind,
            ResponseKind::PathologicalLocalOutput {
                reason: "malformed_tool_markup"
            }
        ));

        let cloud_kind = classify_response(&resp, false, false, false, &default_retries(), false);
        assert!(!matches!(
            cloud_kind,
            ResponseKind::PathologicalLocalOutput { .. }
        ));
    }

    #[test]
    fn test_raw_pathological_response_capture_preserves_provider_payload() {
        let raw = "<tool_call>\n<tool_code>\n!!!!!!!!";
        let resp = make_response(Some(raw), "stop");

        assert_eq!(AgentLoopShared::raw_pathological_response(&resp), Some(raw));
        assert_eq!(
            AgentLoopShared::raw_pathological_response(&make_response(Some("healthy"), "stop")),
            None
        );
    }

    /// Regression for the 2026-07-30 discard loop: a truncated textual tool
    /// call (opening tags, no closer) must classify as a recoverable ToolCalls
    /// response, NOT PathologicalLocalOutput. Before the fix, the `<tool_call`
    /// substring check fired before textual detection and discarded it ×4.
    #[test]
    fn classify_recovers_truncated_tool_call_not_pathological() {
        let raw =
            "<tool_call>\n<function=exec>\n<parameter=command>\ncat ~/.config/higgs/config.toml";
        let resp = make_response(Some(raw), "stop");

        let kind = classify_response(&resp, true, false, false, &default_retries(), false);

        assert!(
            matches!(kind, ResponseKind::ToolCalls { .. }),
            "truncated textual tool call must be recovered as ToolCalls, not discarded; got {kind:?}"
        );
    }

    #[test]
    fn test_is_degenerate_continuation() {
        // Empty / whitespace-only adds nothing.
        assert!(is_degenerate_continuation("   ", "", ""));
        // Identical to the previous continuation (case/space-insensitive).
        assert!(is_degenerate_continuation(
            "I was created by Peppi.",
            &normalize_for_repeat("i was   created by peppi."),
            "earlier text"
        ));
        // A sizeable span already present in accumulated output.
        let acc = "Hello there. I am an AI assistant who helps you.";
        assert!(is_degenerate_continuation(
            "I am an AI assistant who helps you.",
            "something else",
            acc
        ));
        // Genuinely new content continues the loop.
        assert!(!is_degenerate_continuation(
            "And here is a brand new follow-up point.",
            "previous chunk",
            "Hello there."
        ));
        // Short novel fragments are not falsely flagged as repeats-in-accumulated.
        assert!(!is_degenerate_continuation(
            "and so",
            "prev",
            "a long prior body of text"
        ));
    }

    fn make_response(content: Option<&str>, finish_reason: &str) -> LLMResponse {
        LLMResponse {
            content: content.map(|s| s.to_string()),
            tool_calls: vec![],
            finish_reason: FinishReason::parse_finish_reason(finish_reason),
            usage: HashMap::new(),
        }
    }

    fn make_response_with_tools(
        content: Option<&str>,
        tool_names: &[&str],
        finish_reason: &str,
    ) -> LLMResponse {
        let tool_calls = tool_names
            .iter()
            .enumerate()
            .map(|(i, name)| ToolCallRequest {
                id: format!("tc_{}", i),
                name: name.to_string(),
                arguments: HashMap::new(),
            })
            .collect();
        LLMResponse {
            content: content.map(|s| s.to_string()),
            tool_calls,
            finish_reason: FinishReason::parse_finish_reason(finish_reason),
            usage: HashMap::new(),
        }
    }

    fn default_retries() -> RetryState {
        RetryState::new()
    }

    #[test]
    fn plain_text_is_final_answer() {
        let resp = make_response(Some("The answer is 42."), "stop");
        let kind = classify_response(&resp, false, false, false, &default_retries(), false);
        assert!(
            matches!(kind, ResponseKind::Text(ref s) if s == "The answer is 42."),
            "plain non-empty text with no tool calls terminates the turn"
        );
    }

    #[test]
    fn test_classify_tool_calls() {
        let resp = make_response_with_tools(Some("Let me check."), &["read_file"], "stop");
        let kind = classify_response(&resp, false, false, false, &default_retries(), false);
        assert!(matches!(kind, ResponseKind::ToolCalls { .. }));
    }

    #[test]
    fn test_classify_empty_after_think() {
        let resp = make_response(Some(""), "length");
        let kind = classify_response(&resp, true, false, false, &default_retries(), true);
        assert!(matches!(kind, ResponseKind::EmptyAfterThink));
    }

    #[test]
    fn test_classify_empty_final_after_retries() {
        let mut retries = default_retries();
        retries.empty_think_retried = true;
        retries.rescue_attempted = true;
        let resp = make_response(Some(""), "length");
        let kind = classify_response(&resp, true, false, false, &retries, false);
        assert!(matches!(kind, ResponseKind::EmptyFinal));
    }

    #[test]
    fn test_classify_truncated() {
        let resp = make_response(Some("This is a partial response that got cut"), "length");
        let kind = classify_response(&resp, false, false, false, &default_retries(), false);
        assert!(matches!(kind, ResponseKind::Truncated(_)));
    }

    #[test]
    fn test_classify_textual_tool_call_parsed_as_tool_calls() {
        // Text containing [I called: tool(args)] is parsed as a real tool call,
        // not a hallucination. The extraction happens in the handler.
        let resp = make_response(
            Some("[I called: read_file({\"path\":\"/tmp/test\"})]"),
            "stop",
        );
        let kind = classify_response(&resp, false, false, false, &default_retries(), false);
        assert!(matches!(kind, ResponseKind::ToolCalls { .. }));
    }

    #[test]
    fn test_classify_validation_error_hallucinated() {
        // The hallucinated pattern "[Called spawn({})]" without parseable args
        // triggers validation error when no native tool calls exist.
        // Use a pattern that the validation regex catches but the textual parser
        // cannot parse (malformed args).
        let resp = make_response(Some("I did the work. [Called spawn(NOT_JSON)]"), "stop");
        let kind = classify_response(&resp, false, false, false, &default_retries(), false);
        assert!(matches!(
            kind,
            ResponseKind::ValidationError {
                error: validation::ValidationError::HallucinatedToolCall,
                ..
            }
        ));
    }

    /// Tool-intent prose is a final answer, not a validation error — the loop
    /// must deliver it rather than retract and re-issue the turn.
    #[test]
    fn test_classify_tool_intent_prose_as_final_text() {
        let resp = make_response(Some("Let me check that file for you."), "stop");
        let kind = classify_response(&resp, false, false, false, &default_retries(), false);
        assert!(
            matches!(kind, ResponseKind::Text(_)),
            "tool-intent prose must classify as final text, got {kind:?}"
        );
    }

    #[test]
    fn textual_replay_accepts_plain_text_as_final() {
        let resp = make_response(Some("Let me check that file for you."), "stop");
        let kind = classify_response(&resp, false, true, false, &default_retries(), false);
        // Textual replay skips tool-intent validation; plain text is the final answer.
        assert!(matches!(kind, ResponseKind::Text(_)));
    }

    #[test]
    fn test_classify_provider_error() {
        // A dead stream (`finish_reason = "error"`) must classify as
        // ProviderError via `outcome()` with the provider's payload intact.
        let resp = make_response(
            Some("LLM stream ended before the backend produced any response content or tool-call payload."),
            "error",
        );
        let kind = classify_response(&resp, false, false, false, &default_retries(), false);
        assert!(matches!(
            kind,
            ResponseKind::ProviderError(ref msg)
                if msg == "LLM stream ended before the backend produced any response content or tool-call payload."
        ));

        // No payload -> the legacy "Unknown LLM error" fallback, byte-identical.
        let kind = classify_response(
            &make_response(None, "error"),
            false,
            false,
            false,
            &default_retries(),
            false,
        );
        assert!(matches!(
            kind,
            ResponseKind::ProviderError(ref msg) if msg == "Unknown LLM error"
        ));

        // A healthy stop response is NOT a provider error.
        let kind = classify_response(
            &make_response(Some("hi"), "stop"),
            false,
            false,
            false,
            &default_retries(),
            false,
        );
        assert!(!matches!(kind, ResponseKind::ProviderError(_)));
    }

    #[test]
    fn test_classify_text_with_stop_and_complete() {
        let resp = make_response(Some("All done. Here is your answer."), "stop");
        let kind = classify_response(&resp, false, false, false, &default_retries(), false);
        assert!(matches!(kind, ResponseKind::Text(ref s) if s == "All done. Here is your answer."));
    }

    #[test]
    fn test_classify_calling_tool_as_tool_call() {
        // [Calling tool: ...] should be parsed as a real tool call, not plain text.
        let resp = make_response(
            Some(r#"[Calling tool: read_file({"path":"/tmp/test"})]"#),
            "stop",
        );
        let kind = classify_response(&resp, false, false, false, &default_retries(), false);
        assert!(
            matches!(kind, ResponseKind::ToolCalls { .. }),
            "Expected ToolCalls for [Calling tool: ...], got {:?}",
            kind
        );
    }

    #[test]
    fn test_classify_empty_xml_tool_call_as_empty_not_text() {
        // Empty <tool_call></tool_call> should NOT be classified as ToolCalls
        // (no valid function inside) and the XML tags should have been stripped
        // by extract_textual_tool_calls before classification.
        let resp = make_response(Some("<tool_call>\n</tool_call>"), "stop");
        let kind = classify_response(&resp, false, false, false, &default_retries(), false);
        // After stripping, content would be empty -- but classify_response sees
        // the content pre-stripping, so it falls through to text or empty.
        // The key assertion: it must NOT be ResponseKind::ToolCalls.
        assert!(
            !matches!(kind, ResponseKind::ToolCalls { .. }),
            "Empty <tool_call></tool_call> must NOT be classified as tool calls"
        );
    }

    /// Rescue messages must go through protocol rendering so local servers
    /// don't receive invalid roles like "tool" or "developer".
    #[test]
    fn test_prepare_rescue_messages_renders_via_local_protocol() {
        use crate::agent::protocol::LocalProtocol;

        // Raw messages with roles that local templates reject.
        let raw = vec![
            json!({"role": "system", "content": "You are helpful."}),
            json!({"role": "user", "content": "hi"}),
            json!({"role": "assistant", "content": null, "tool_calls": [
                {"id": "tc_1", "type": "function", "function": {"name": "exec", "arguments": "{}"}}
            ]}),
            json!({"role": "tool", "tool_call_id": "tc_1", "name": "exec", "content": "done"}),
        ];

        let protocol = LocalProtocol::native();
        let rendered = super::prepare_rescue_messages(&raw, &protocol);

        // After rendering, the local protocol now emits role:tool for
        // tool results in NativeToolCalls mode (the chat template
        // handles it natively). All roles are valid.
        for msg in &rendered {
            let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("");
            assert!(
                role == "system" || role == "user" || role == "assistant" || role == "tool",
                "rescue messages contain unexpected role '{role}'",
            );
        }
        // Must end with user (the rescue prompt).
        let last_role = rendered.last().unwrap()["role"].as_str().unwrap();
        assert_eq!(last_role, "user", "rescue messages must end with user role");
        // The rescue prompt must be present.
        let last_content = rendered.last().unwrap()["content"].as_str().unwrap();
        assert!(
            last_content.contains("final answer"),
            "rescue prompt missing from rendered messages"
        );
    }

    #[test]
    fn test_tool_calls_to_maps() {
        let tools = vec![ToolCallRequest {
            id: "tc_1".into(),
            name: "read_file".into(),
            arguments: {
                let mut m = HashMap::new();
                m.insert("path".into(), Value::String("/tmp/x".into()));
                m
            },
        }];
        let maps = tool_calls_to_maps(&tools);
        assert_eq!(maps.len(), 1);
        assert_eq!(maps[0]["name"], "read_file");
        assert_eq!(maps[0]["arguments"]["path"], "/tmp/x");
    }
}
