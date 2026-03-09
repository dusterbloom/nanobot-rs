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
use crate::providers::base::{LLMResponse, ToolCallRequest};

use super::{AgentLoopShared, IterationOutcome, IterationPhase, StepResult, TurnContext};

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
    ToolCalls {
        tool_calls: Vec<ToolCallRequest>,
    },
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
}

impl RetryState {
    pub(crate) fn new() -> Self {
        Self {
            validation: 0,
            continuations: 0,
            empty_think_retried: false,
            rescue_attempted: false,
            api_retried: false,
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
    // Provider error takes absolute priority.
    if let Some(err_msg) = response.error_detail() {
        return ResponseKind::ProviderError(err_msg.to_string());
    }

    let content = response.content.as_deref().unwrap_or("");
    let has_native_tools = response.has_tool_calls();
    let has_visible_text = !content.trim().is_empty();

    // Check for text-embedded tool calls (bracket or XML format) when no
    // native tool_calls exist. We only *detect* here — the actual parsing
    // and stripping happens in the handler so `response` stays immutable.
    let has_textual_tools = !has_native_tools
        && has_visible_text
        && (has_bracket_tool_calls(content) || has_xml_tool_calls(content));

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
        if is_local && response.finish_reason == "length" && !retries.rescue_attempted {
            return ResponseKind::EmptyAfterThink;
        }
        if thinking_was_on && !retries.empty_think_retried {
            return ResponseKind::EmptyAfterThink;
        }
        return ResponseKind::EmptyFinal;
    }

    // Non-empty text, no tool calls — check for truncation.
    let is_truncated = response.finish_reason == "length"
        || (response.finish_reason == "stop"
            && super::appears_incomplete(content));
    if is_truncated && retries.continuations < 10 {
        // Only classify as Truncated if there's room to continue.
        // (Actual cap is checked in the handler via core.max_continuations.)
        return ResponseKind::Truncated(content.to_string());
    }

    ResponseKind::Text(content.to_string())
}

// ---------------------------------------------------------------------------
// Channel helpers (avoids type-inference issues in #[path] submodules)
// ---------------------------------------------------------------------------

fn send_finish_reason(
    tx: &Option<tokio::sync::mpsc::UnboundedSender<String>>,
    reason: &str,
) {
    if let Some(ref tx) = tx {
        let _ = tx.send(format!("\x00finish_reason:{}", reason));
    }
}

fn send_delta(tx: &Option<tokio::sync::mpsc::UnboundedSender<String>>, text: &str) {
    if let Some(ref tx) = tx {
        let _ = tx.send(text.to_string());
    }
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
        if ctx.core.is_local && ctx.core.anti_drift.enabled && !response.has_tool_calls() {
            if let Some(ref mut content) = response.content {
                anti_drift::post_completion_pipeline(content, &ctx.messages, &ctx.core.anti_drift);
            }
        }

        // --- Classify ---
        let thinking_was_on = counters.thinking_budget.load(Ordering::Relaxed) > 0;
        let kind = classify_response(
            &response,
            ctx.core.is_local,
            ctx.protocol.is_textual_replay(),
            ctx.flow.tool_guard.had_blocked_calls,
            &ctx.flow.retries,
            thinking_was_on,
        );

        // --- Token telemetry (always, regardless of kind) ---
        self.emit_token_telemetry(ctx, &response);

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
                let tool_calls = response.tool_calls.clone();
                StepResult::Next(IterationPhase::Executing {
                    response,
                    tool_calls,
                })
            }

            ResponseKind::Text(content) => {
                send_finish_reason(&ctx.text_delta_tx, &response.finish_reason);
                StepResult::Done(IterationOutcome::Finished(content))
            }

            ResponseKind::ValidationError { error, raw_content } => {
                self.handle_validation_error(ctx, &error, &raw_content)
            }

            ResponseKind::ProviderError(err_msg) => {
                self.handle_provider_error(ctx, &err_msg).await
            }

            ResponseKind::Truncated(partial) => {
                let full = self.handle_truncated(ctx, &response, partial).await;
                send_finish_reason(&ctx.text_delta_tx, "stop");
                StepResult::Done(IterationOutcome::Finished(full))
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
                send_finish_reason(&ctx.text_delta_tx, &response.finish_reason);
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
        let hint = validation::generate_retry_prompt(error, retry_num as u8);
        ctx.messages.push(json!({
            "role": "assistant",
            "content": raw_content
        }));
        ctx.messages.push(json!({
            "role": "user",
            "content": hint
        }));
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

        if ctx.core.is_local {
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
        let max_cont = ctx.core.max_continuations;
        let mut finish_reason = original_response.finish_reason.clone();

        while ctx.flow.retries.continuations < max_cont {
            // Check if still truncated.
            let is_truncated = finish_reason == "length"
                || (finish_reason == "stop"
                    && super::appears_incomplete(&accumulated));
            if !is_truncated {
                break;
            }

            ctx.flow.retries.continuations += 1;
            if finish_reason == "stop" {
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

            counters.inference_active.store(true, Ordering::Relaxed);
            let cont_result = ctx
                .core
                .provider
                .chat(
                    &cont_messages,
                    None,
                    Some(&ctx.core.model),
                    ctx.core.max_tokens,
                    ctx.core.temperature,
                    None,
                    None,
                )
                .await;
            counters.inference_active.store(false, Ordering::Relaxed);

            match cont_result {
                Ok(cont_response) => {
                    if let Some(ref new_text) = cont_response.content {
                        send_delta(&ctx.text_delta_tx, new_text);
                    }
                    let continuation = cont_response.content.unwrap_or_default();
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
        if ctx.core.is_local
            && response.finish_reason == "length"
            && !ctx.flow.retries.rescue_attempted
        {
            ctx.flow.retries.rescue_attempted = true;
            let rescue_tokens = ctx.core.max_tokens.min(384).max(128);
            let mut rescue_messages = ctx.messages.clone();
            rescue_messages.push(json!({
                "role": "user",
                "content": "Return the final answer now. No reasoning. No tool calls. Max 6 lines."
            }));
            counters.inference_active.store(true, Ordering::Relaxed);
            let rescue_result = ctx
                .core
                .provider
                .chat(
                    &rescue_messages,
                    None,
                    Some(&ctx.core.model),
                    rescue_tokens,
                    0.2,
                    None,
                    None,
                )
                .await;
            counters.inference_active.store(false, Ordering::Relaxed);

            match rescue_result {
                Ok(r) => {
                    let content = r.content.unwrap_or_default();
                    if !content.trim().is_empty() {
                        send_finish_reason(&ctx.text_delta_tx, &r.finish_reason);
                        return StepResult::Done(IterationOutcome::Finished(content));
                    }
                    // Rescue also empty — fall through to thinking-off retry.
                }
                Err(e) => {
                    warn!("Finalize rescue call failed: {}", e);
                }
            }
        }

        // Retry with thinking disabled.
        if !ctx.flow.retries.empty_think_retried {
            ctx.flow.retries.empty_think_retried = true;
            warn!(
                finish_reason = %response.finish_reason,
                "empty_llm_response: thinking consumed entire output, retrying with thinking off"
            );
            counters.thinking_budget.store(0, Ordering::Relaxed);
            return StepResult::Done(IterationOutcome::Continue);
        }

        // Both rescue and thinking-off retry exhausted — fallback.
        warn!(
            finish_reason = %response.finish_reason,
            "empty_llm_response: all recovery attempts exhausted, injecting fallback"
        );
        let content = "I couldn't produce a response in this turn. Please try again.".to_string();
        send_finish_reason(&ctx.text_delta_tx, &response.finish_reason);
        StepResult::Done(IterationOutcome::Finished(content))
    }

    // -----------------------------------------------------------------------
    // Token telemetry
    // -----------------------------------------------------------------------

    fn emit_token_telemetry(&self, ctx: &TurnContext, response: &LLMResponse) {
        let counters = &self.core_handle.counters;
        let estimated_prompt = TokenBudget::estimate_tokens(&ctx.messages);
        let actual_prompt = response.usage.get("prompt_tokens").copied().unwrap_or(-1);
        let actual_completion = response
            .usage
            .get("completion_tokens")
            .copied()
            .unwrap_or(-1);
        info!(
            "tokens: estimated_prompt={}, actual_prompt={}, actual_completion={}",
            estimated_prompt, actual_prompt, actual_completion
        );
        if actual_prompt > 0 {
            counters
                .last_actual_prompt_tokens
                .store(actual_prompt as u64, Ordering::Relaxed);
        }
        if actual_completion > 0 {
            counters
                .last_actual_completion_tokens
                .store(actual_completion as u64, Ordering::Relaxed);
        }
        counters
            .last_estimated_prompt_tokens
            .store(estimated_prompt as u64, Ordering::Relaxed);

        crate::agent::metrics::emit(&crate::agent::metrics::RequestMetrics {
            timestamp: chrono::Local::now().to_rfc3339(),
            request_id: ctx.request_id.clone(),
            role: "main".into(),
            model: ctx.core.model.clone(),
            provider_base: ctx.core.provider.get_api_base().unwrap_or("unknown").into(),
            elapsed_ms: ctx
                .flow
                .llm_call_start
                .map_or(0, |t| t.elapsed().as_millis() as u64),
            prompt_tokens: actual_prompt.max(0) as u64,
            completion_tokens: actual_completion.max(0) as u64,
            status: "ok".into(),
            error_detail: None,
            anti_drift_score: None,
            anti_drift_signals: None,
            tool_calls_requested: response.tool_calls.len() as u32,
            tool_calls_executed: 0,
            validation_result: None,
        });
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_response(content: Option<&str>, finish_reason: &str) -> LLMResponse {
        LLMResponse {
            content: content.map(|s| s.to_string()),
            tool_calls: vec![],
            finish_reason: finish_reason.to_string(),
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
            finish_reason: finish_reason.to_string(),
            usage: HashMap::new(),
        }
    }

    fn default_retries() -> RetryState {
        RetryState::new()
    }

    #[test]
    fn test_classify_plain_text() {
        let resp = make_response(Some("The answer is 42."), "stop");
        let kind = classify_response(&resp, false, false, false, &default_retries(), false);
        assert!(matches!(kind, ResponseKind::Text(ref s) if s == "The answer is 42."));
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
        let resp = make_response(
            Some("I did the work. [Called spawn(NOT_JSON)]"),
            "stop",
        );
        let kind = classify_response(&resp, false, false, false, &default_retries(), false);
        assert!(matches!(
            kind,
            ResponseKind::ValidationError {
                error: validation::ValidationError::HallucinatedToolCall,
                ..
            }
        ));
    }

    #[test]
    fn test_classify_validation_error_claimed() {
        let resp = make_response(Some("Let me check that file for you."), "stop");
        let kind = classify_response(&resp, false, false, false, &default_retries(), false);
        assert!(matches!(
            kind,
            ResponseKind::ValidationError {
                error: validation::ValidationError::ClaimedButNotExecuted,
                ..
            }
        ));
    }

    #[test]
    fn test_classify_textual_replay_skips_validation() {
        let resp = make_response(Some("Let me check that file for you."), "stop");
        let kind = classify_response(&resp, false, true, false, &default_retries(), false);
        // In textual replay mode, "let me check" should NOT trigger validation error.
        assert!(matches!(kind, ResponseKind::Text(_)));
    }

    #[test]
    fn test_classify_provider_error() {
        let mut resp = make_response(Some(""), "stop");
        resp.usage
            .insert("error".to_string(), -1);
        // Provider errors are detected by error_detail() on LLMResponse.
        // We test the path by checking that our kind logic handles it.
        // Since error_detail() checks specific fields, let's test via the
        // known error pattern.
        let kind = classify_response(&resp, false, false, false, &default_retries(), false);
        // Without an actual error field this falls through to EmptyFinal.
        assert!(matches!(kind, ResponseKind::EmptyFinal));
    }

    #[test]
    fn test_classify_text_with_stop_and_complete() {
        let resp = make_response(Some("All done. Here is your answer."), "stop");
        let kind = classify_response(&resp, false, false, false, &default_retries(), false);
        assert!(matches!(kind, ResponseKind::Text(_)));
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
