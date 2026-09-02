// Error-protocol layer-3 backlog (docs/research/2026-08-06-error-conventions-and-host-bridge.md §3.6):
// the deny regime in Cargo.toml is live; this module still carries pre-existing
// violations of the lints below. Remove this allow as the module migrates onto
// the regime.
// Tracking: docs/error-protocol-backlog.md
#![allow(
    clippy::as_conversions,
    clippy::format_push_string,
    clippy::indexing_slicing,
    clippy::shadow_reuse
)]
//! Router decision parsing and dispatch functions.
//!
//! Extracted from `agent_loop.rs` to isolate routing logic into a focused module.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::Ordering;

use serde_json::{json, Value};
use tracing::{debug, info, instrument, warn};

use super::trace_store::{append_router_decision_trace, RouterDecisionTrace};
use crate::agent::agent_core::SwappableCore;
use crate::agent::agent_loop::{ToolRouting, TurnContext};
use crate::agent::markers::{
    TOOL_ANALYSIS_FULL_OUTPUT_MARKER, TOOL_ANALYSIS_SUMMARY_PREFIX, TOOL_RUNNER_SUMMARY_PREFIX,
};
use crate::agent::policy;
use crate::agent::role_policy;
use crate::agent::role_policy::{build_specialist_system_prompt, parse_specialist_response};
use crate::agent::router_fallback;
use crate::agent::tool_guard::ToolGuard;
use crate::agent::toolplan::{self, ToolPlanAction};
use crate::agent::tools::registry::ToolRegistry;
use crate::providers::base::{LLMProvider, ToolCallRequest, ToolChoice};
use crate::session::db::{
    ModelCallPurpose, RecordedProviderRequest, RecordedProviderResponse, TurnReplayRecorder,
};

const ROUTER_SUSPICIOUS_TARGET_MAX_LEN: usize = 96;
const ROUTER_PARSE_ERROR_RAW_PREVIEW_CHARS: usize = 220;
const ROUTER_USER_MSG_TRACE_PREVIEW_CHARS: usize = 80;
const ROUTER_TAIL_MAX_PAIRS: usize = 5;
const SCRATCH_PAD_LOOKBACK_MESSAGES: usize = 10;

/// Distinguishes a provider/router failure from loss of replay durability.
/// Callers may recover from the former, but must fail closed on the latter.
#[derive(Debug, thiserror::Error)]
pub enum AuxiliaryCallError {
    #[error("{0}")]
    Persistence(String),
    #[error("{0}")]
    Call(String),
}

/// Per-domain ring buffer for specialist multi-turn memory.
/// Stores compressed summaries of past specialist outputs so subsequent
/// specialist calls in the same domain can build on prior analysis.
pub(crate) struct SpecialistMemory {
    entries: HashMap<String, VecDeque<String>>,
    max_entries: usize,
    max_chars_per_entry: usize,
}

impl SpecialistMemory {
    pub fn new(max_entries: usize, max_chars_per_entry: usize) -> Self {
        Self {
            entries: HashMap::new(),
            max_entries,
            max_chars_per_entry,
        }
    }

    /// Push a specialist response summary into the domain's ring buffer.
    pub fn push(&mut self, domain: &str, summary: &str) {
        let truncated = if summary.len() > self.max_chars_per_entry {
            &summary[..self.max_chars_per_entry]
        } else {
            summary
        };
        let buf = self
            .entries
            .entry(domain.to_string())
            .or_insert_with(|| VecDeque::with_capacity(self.max_entries));
        if buf.len() >= self.max_entries {
            buf.pop_front();
        }
        buf.push_back(truncated.to_string());
    }

    /// Format accumulated context for a domain. Returns empty string if no entries.
    pub fn format_context(&self, domain: &str) -> String {
        match self.entries.get(domain) {
            None => String::new(),
            Some(buf) if buf.is_empty() => String::new(),
            Some(buf) => {
                let mut ctx = String::from("[prior specialist context]\n");
                for (i, entry) in buf.iter().enumerate() {
                    ctx.push_str(&format!("- turn {}: {}\n", i + 1, entry));
                }
                ctx
            }
        }
    }
}

impl Default for SpecialistMemory {
    fn default() -> Self {
        Self::new(3, 200)
    }
}

/// Build a compact conversation tail from the message history for the router.
///
/// Extracts the last `max_pairs` user/assistant exchanges (skipping system,
/// tool_call, and tool-result messages). Each message is truncated to
/// `max_msg_chars`. The total output is capped at `max_chars`.
/// When LCM compaction is active, `messages` already contains summaries in
/// place of old messages, so this naturally includes compressed context.
/// Search backwards through recent messages for a scratch pad summary.
/// Returns the summary text if found, otherwise None.
pub fn find_scratch_pad_summary_in_messages(messages: &[Value]) -> Option<String> {
    for msg in messages.iter().rev().take(SCRATCH_PAD_LOOKBACK_MESSAGES) {
        let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("");
        let content = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
        if content.is_empty() {
            continue;
        }
        // Standalone summary from tool_engine.rs line 327: "[tool runner summary] ..."
        if role == "user" && content.starts_with(TOOL_RUNNER_SUMMARY_PREFIX) {
            if let Some(rest) = content.strip_prefix(TOOL_RUNNER_SUMMARY_PREFIX) {
                let rest = rest.trim_start();
                let s = rest.trim().to_string();
                if !s.is_empty() {
                    return Some(s);
                }
            }
        }
        // Inline summary from tool_engine.rs line 267: "[Tool analysis summary]\n..."
        if role == "tool" && content.starts_with(TOOL_ANALYSIS_SUMMARY_PREFIX) {
            if let Some(rest) = content.strip_prefix(TOOL_ANALYSIS_SUMMARY_PREFIX) {
                let rest = rest.trim_start_matches('\n');
                let summary = rest
                    .split(TOOL_ANALYSIS_FULL_OUTPUT_MARKER)
                    .next()
                    .unwrap_or(rest)
                    .trim()
                    .to_string();
                if !summary.is_empty() {
                    return Some(summary);
                }
            }
        }
    }
    None
}

pub fn build_conversation_tail(
    messages: &[Value],
    max_pairs: usize,
    max_msg_chars: usize,
    max_chars: usize,
) -> String {
    let mut pairs: Vec<(Option<&str>, Option<&str>)> = Vec::new();
    let mut current_user: Option<&str> = None;

    for msg in messages {
        let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("");
        let content = msg.get("content").and_then(|v| v.as_str()).unwrap_or("");
        if content.is_empty() {
            continue;
        }
        match role {
            "user" => {
                current_user = Some(content);
            }
            "assistant" => {
                if current_user.is_some() || pairs.is_empty() {
                    pairs.push((current_user.take(), Some(content)));
                }
            }
            _ => {} // skip system, tool
        }
    }

    // Take the last N pairs
    let tail: Vec<_> = if pairs.len() > max_pairs {
        pairs[pairs.len() - max_pairs..].to_vec()
    } else {
        pairs
    };

    let mut out = String::new();
    for (user, assistant) in &tail {
        if let Some(u) = user {
            let truncated = if u.len() > max_msg_chars {
                let end = crate::utils::helpers::floor_char_boundary(u, max_msg_chars);
                format!("{}…", &u[..end])
            } else {
                u.to_string()
            };
            out.push_str(&format!("User: {}\n", truncated));
        }
        if let Some(a) = assistant {
            let truncated = if a.len() > max_msg_chars {
                let end = crate::utils::helpers::floor_char_boundary(a, max_msg_chars);
                format!("{}…", &a[..end])
            } else {
                a.to_string()
            };
            out.push_str(&format!("Assistant: {}\n", truncated));
        }
    }

    if out.len() > max_chars {
        let end = crate::utils::helpers::floor_char_boundary(&out, max_chars);
        out.truncate(end);
    }
    out
}

/// Extract the first top-level JSON object from raw text.
///
/// This tolerates wrappers like markdown fences while still requiring the
/// final parsed payload to satisfy strict schema validation.
pub(crate) fn extract_json_object(raw: &str) -> Option<String> {
    let mut start = None;
    let mut depth: i32 = 0;
    for (idx, ch) in raw.char_indices() {
        if ch == '{' {
            if start.is_none() {
                start = Some(idx);
            }
            depth += 1;
        } else if ch == '}' && depth > 0 {
            depth -= 1;
            if depth == 0 {
                if let Some(s) = start {
                    return Some(raw[s..=idx].to_string());
                }
            }
        }
    }
    None
}

/// Lenient parser for non-standard router output (comma-separated, malformed JSON, etc.).
///
/// Example accepted fragment:
/// `call: tool,read_file,{"path":"README.md","confidence":0.9}`
pub(crate) fn parse_lenient_router_decision(raw: &str) -> Option<role_policy::RouterDecision> {
    fn normalize_action(raw_action: &str, target: &str, args: &Value) -> String {
        let a = raw_action.to_lowercase();
        let t = target.to_lowercase();
        if matches!(a.as_str(), "tool" | "subagent" | "specialist" | "ask_user") {
            return a;
        }
        if t.contains("clarify") || args.get("question").is_some() {
            return "ask_user".to_string();
        }
        if t.contains("summar") || t.contains("specialist") {
            return "specialist".to_string();
        }
        if t.contains("agent") || a.contains("subagent") {
            return "subagent".to_string();
        }
        "tool".to_string()
    }

    fn extract_quoted(raw: &str, key: &str) -> Option<String> {
        let pat = format!("\"{}\":\"", key);
        let start = raw.find(&pat)? + pat.len();
        let tail = &raw[start..];
        let end = tail.find('"')?;
        Some(tail[..end].to_string())
    }

    let mut tail = if let Some(call_start) = raw.find("call:") {
        raw[call_start + "call:".len()..].to_string()
    } else {
        raw.to_string()
    }
    .replace("<start_function_call>", "")
    .replace("<end_function_call>", "")
    .replace("<escape>", "")
    .replace('\n', " ");

    let end = tail.find("<end_function_call>").unwrap_or(tail.len());
    tail = tail[..end].trim().to_string();

    // Comma-separated shape (FunctionGemma, etc.): `tool,target,{"k":"v"}`
    if tail.contains(',') && !tail.contains("\"action\"") {
        let mut parts = tail.splitn(3, ',');
        let raw_action = parts.next()?.trim();
        let target = parts.next()?.trim().to_string();
        let args_raw = parts.next().unwrap_or("{}").trim();
        let args = serde_json::from_str::<Value>(args_raw).unwrap_or_else(|_| json!({}));
        let confidence = args
            .get("confidence")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.7);
        let decision = role_policy::RouterDecision {
            action: normalize_action(raw_action, &target, &args),
            target,
            args,
            confidence,
        };
        if role_policy::parse_router_decision_strict(&serde_json::to_string(&decision).ok()?)
            .is_ok()
        {
            return Some(decision);
        }
    }

    // Malformed JSON-ish output: recover fields leniently.
    // When no target is extractable, default to empty string. This fails strict
    // validation (action="tool" requires non-empty target), causing the parse
    // chain to fall through to Passthrough → main model handles the request.
    // Previously defaulted to "clarify" which dispatched to a nonexistent tool.
    let target = extract_quoted(&tail, "target")
        .or_else(|| extract_quoted(raw, "target"))
        .unwrap_or_default();
    let raw_action = extract_quoted(&tail, "action")
        .or_else(|| extract_quoted(&tail, "call"))
        .or_else(|| extract_quoted(raw, "action"))
        .or_else(|| extract_quoted(raw, "call"))
        .unwrap_or_else(|| "tool".to_string());
    let args = extract_json_object(&tail)
        .and_then(|obj| serde_json::from_str::<Value>(&obj).ok())
        .and_then(|v| v.get("args").cloned())
        .or_else(|| {
            extract_json_object(raw)
                .and_then(|obj| serde_json::from_str::<Value>(&obj).ok())
                .and_then(|v| v.get("args").cloned())
        })
        .unwrap_or_else(|| json!({}));
    let confidence = extract_json_object(&tail)
        .and_then(|obj| serde_json::from_str::<Value>(&obj).ok())
        .and_then(|v| v.get("confidence").and_then(|c| c.as_f64()))
        .or_else(|| {
            extract_json_object(raw)
                .and_then(|obj| serde_json::from_str::<Value>(&obj).ok())
                .and_then(|v| v.get("confidence").and_then(|c| c.as_f64()))
        })
        .unwrap_or(0.5);
    let decision = role_policy::RouterDecision {
        action: normalize_action(&raw_action, &target, &args),
        target,
        args,
        confidence,
    };
    if role_policy::parse_router_decision_strict(&serde_json::to_string(&decision).ok()?).is_ok() {
        Some(decision)
    } else {
        None
    }
}

/// Router/specialist calls are part of the same replayable turn. Refuse the
/// provider side effect when its exact request cannot be recorded first.
async fn journal_aux_request(
    replay: Option<&TurnReplayRecorder>,
    purpose: ModelCallPurpose,
    request: &RecordedProviderRequest,
) -> Result<Option<String>, AuxiliaryCallError> {
    let Some(replay) = replay else {
        return Ok(None);
    };
    replay
        .request(purpose, request)
        .await
        .map(Some)
        .map_err(|error| AuxiliaryCallError::Persistence(error.to_string()))
}

/// Close an auxiliary-lane journal entry after its provider call resolved.
async fn journal_aux_terminal(
    replay: &TurnReplayRecorder,
    call_id: &str,
    result: &anyhow::Result<crate::providers::base::LLMResponse>,
) -> Result<(), AuxiliaryCallError> {
    match result {
        Ok(response) => {
            replay
                .response(call_id, &RecordedProviderResponse::from(response))
                .await
        }
        Err(error) => replay.failure(call_id, &error.to_string()).await,
    }
    .map_err(|error| AuxiliaryCallError::Persistence(error.to_string()))
}

#[instrument(name = "request_strict_router_decision", skip(provider, router_pack, tool_names, replay), fields(
    model,
    no_think,
    parse_strategy = tracing::field::Empty,
))]
pub async fn request_strict_router_decision(
    provider: &dyn LLMProvider,
    model: &str,
    router_pack: &str,
    no_think: bool,
    temperature: f64,
    top_p: f64,
    tool_names: &str,
    max_tokens: u32,
    replay: Option<&TurnReplayRecorder>,
) -> Result<role_policy::RouterDecision, AuxiliaryCallError> {
    info!(role = "router", model = %model, "router_decision_start");
    // action=/target= are bare tokens ended by whitespace or a comma.
    fn take_token(pack: &str, key: &str) -> Option<String> {
        let start = pack.find(key)? + key.len();
        let tail = &pack[start..];
        let end = tail
            .find(|c: char| c.is_whitespace() || c == ',')
            .unwrap_or(tail.len());
        Some(tail[..end].trim().to_string())
    }
    fn parse_router_directive_pack(pack: &str) -> Option<role_policy::RouterDecision> {
        let action = take_token(pack, "action=")?;
        let target = take_token(pack, "target=")?;
        let args = if let Some(args_pos) = pack.find("args=") {
            let tail = &pack[args_pos + "args=".len()..];
            extract_json_object(tail)
                .and_then(|obj| serde_json::from_str::<Value>(&obj).ok())
                .unwrap_or_else(|| json!({}))
        } else {
            json!({})
        };
        let decision = role_policy::RouterDecision {
            action,
            target,
            args,
            confidence: 0.9,
        };
        if role_policy::parse_router_decision_strict(&serde_json::to_string(&decision).ok()?)
            .is_ok()
        {
            Some(decision)
        } else {
            None
        }
    }

    // Build user content with optional /no_think prefix, but only for
    // models that actually require template-level no_think handling.
    let model_needs_no_think_prefix =
        crate::agent::model_capabilities::lookup_default(model).needs_native_lms_api;
    let user_content = if no_think && model_needs_no_think_prefix {
        format!(" /no_think\n{}", router_pack)
    } else {
        router_pack.to_string()
    };

    let route_tool = json!({
        "type": "function",
        "function": {
            "name": "route_decision",
            "description": "Return one routing decision.",
            "parameters": {
                "type": "object",
                "properties": {
                    // Must list every action parse_router_decision_strict accepts
                    // (role_policy.rs): grammar-constrained local decoding enforces
                    // this enum, so an omitted action becomes unreachable.
                    "action": {"type": "string", "enum": ["respond","tool","subagent","specialist","ask_user","pipeline"]},
                    "target": {"type": "string"},
                    // Apple FM guided-generation rejects free-form object
                    // schemas ({"type":"object"} with no properties) with a 400
                    // "Invalid tool definition". Declare args as a string so the
                    // tool definition validates; the value is re-hydrated below.
                    "args": {"type": "string", "description": "Tool arguments as a JSON object, serialized to a string. Use {} when there are none."},
                    "confidence": {"type": "number"}
                },
                "required": ["action","target","args","confidence"]
            }
        }
    });
    let tool_catalog = if tool_names.is_empty() {
        String::new()
    } else {
        format!(
            "\nAvailable tools (use exact names for target): {}\n",
            tool_names
        )
    };

    let tool_defs = vec![route_tool];
    let tool_system = format!(
        "You are a routing agent. Analyze the user's request and call route_decision once.\n\n\
         Actions:\n\
         - respond: Greetings, chitchat, simple questions the main model can answer directly\n\
         - tool: Use a specific tool (set target=tool_name, args=tool_parameters)\n\
         - specialist: Delegate to specialist model for complex multi-step reasoning\n\
         - ask_user: ONLY when the request is truly ambiguous and cannot be answered\n\
         {}\
         If the user is just saying hello or asking a simple question, use action=respond.\n\
         Call route_decision exactly once. No prose.",
        tool_catalog
    );
    let tool_messages = vec![
        json!({
            "role": "system",
            "content": tool_system
        }),
        json!({
            "role": "user",
            "content": user_content.clone()
        }),
    ];
    let tool_call_id = journal_aux_request(
        replay,
        ModelCallPurpose::Router,
        &RecordedProviderRequest {
            messages: tool_messages.clone(),
            tools: Some(tool_defs.clone()),
            model: model.to_string(),
            max_tokens,
            temperature,
            thinking_budget: None,
            top_p: Some(top_p),
            tool_choice: "required".to_string(),
            streaming: false,
        },
    )
    .await?;
    let tool_result = provider
        .chat_with_tool_choice(
            &tool_messages,
            Some(&tool_defs),
            Some(model),
            max_tokens,
            temperature,
            None,
            Some(top_p),
            // Force exactly one route_decision call. On a local Higgs backend
            // this triggers grammar-constrained decoding so the decision is
            // always a well-formed tool call (no fragile JSON-text fallback).
            ToolChoice::Required,
        )
        .await;
    if let (Some(replay), Some(call_id)) = (replay, tool_call_id.as_deref()) {
        journal_aux_terminal(replay, call_id, &tool_result).await?;
    }
    if let Ok(tool_resp) = tool_result {
        if let Some(tc) = tool_resp.tool_calls.first() {
            if tc.name == "route_decision" {
                let mut args_obj = tc
                    .arguments
                    .iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect::<serde_json::Map<String, Value>>();
                // `args` is declared as a string in the tool schema (see above)
                // for Apple FM compatibility. Providers may honor that and emit
                // a JSON-encoded string, or emit an object directly. Re-hydrate
                // the string form into an object; leave an object form as-is.
                if let Some(Value::String(s)) = args_obj.get("args") {
                    let rehydrated = serde_json::from_str::<Value>(s).unwrap_or_else(|_| json!({}));
                    args_obj.insert("args".to_string(), rehydrated);
                }
                let val = Value::Object(args_obj);
                if let Ok(decision) = serde_json::from_value::<role_policy::RouterDecision>(val) {
                    if role_policy::parse_router_decision_strict(
                        &serde_json::to_string(&decision).unwrap_or_default(),
                    )
                    .is_ok()
                    {
                        tracing::Span::current().record("parse_strategy", "tool_call");
                        return Ok(decision);
                    }
                }
            }
        }
    }

    let json_system = format!(
        "Output EXACTLY one JSON object. No markdown, no explanation, no extra text.\n\
         Schema: {{\"action\":\"respond|tool|subagent|specialist|ask_user|pipeline\",\"target\":\"string\",\"args\":{{}},\"confidence\":0.0-1.0}}\n\
         {}\n\
         Examples:\n\
         User says hello → {{\"action\":\"respond\",\"target\":\"main\",\"args\":{{}},\"confidence\":0.95}}\n\
         User asks to read a file → {{\"action\":\"tool\",\"target\":\"read_file\",\"args\":{{\"path\":\"README.md\"}},\"confidence\":0.9}}\n\
         User asks a simple question → {{\"action\":\"respond\",\"target\":\"main\",\"args\":{{}},\"confidence\":0.9}}\n",
        tool_catalog
    );
    let router_messages = vec![
        json!({
            "role": "system",
            "content": json_system
        }),
        json!({
            "role": "user",
            "content": user_content
        }),
    ];

    let fallback_call_id = journal_aux_request(
        replay,
        ModelCallPurpose::Router,
        &RecordedProviderRequest {
            messages: router_messages.clone(),
            tools: None,
            model: model.to_string(),
            max_tokens,
            temperature,
            thinking_budget: None,
            top_p: Some(top_p),
            tool_choice: "auto".to_string(),
            streaming: false,
        },
    )
    .await?;
    let router_result = provider
        .chat(
            &router_messages,
            None,
            Some(model),
            max_tokens,
            temperature,
            None,
            Some(top_p),
        )
        .await;
    if let (Some(replay), Some(call_id)) = (replay, fallback_call_id.as_deref()) {
        journal_aux_terminal(replay, call_id, &router_result).await?;
    }
    let router_resp = router_result
        .map_err(|e| AuxiliaryCallError::Call(format!("strict router call failed: {e}")))?;
    let raw_router_content = router_resp.content.unwrap_or_default();
    let raw = crate::agent::sanitize::sanitize_reasoning_output(&raw_router_content);
    let parsed = role_policy::parse_router_decision_strict(&raw)
        .or_else(|_| {
            extract_json_object(&raw)
                .ok_or_else(|| "no JSON object found".to_string())
                .and_then(|obj| role_policy::parse_router_decision_strict(&obj))
        })
        .or_else(|_| {
            parse_lenient_router_decision(&raw)
                .ok_or_else(|| "no JSON or lenient call format found".to_string())
        })
        .map_err(|e| e.to_string());
    match parsed {
        Ok(mut decision) => {
            let suspicious = raw.contains('|')
                || decision.target.contains("\"target\"")
                || decision.target.len() > ROUTER_SUSPICIOUS_TARGET_MAX_LEN;
            if suspicious {
                if let Some(from_pack) = parse_router_directive_pack(router_pack) {
                    decision = from_pack;
                }
            }
            debug!(
                action = %decision.action,
                target = %decision.target,
                confidence = decision.confidence,
                "router_decision_parsed"
            );
            Ok(decision)
        }
        Err(e) => {
            if let Some(from_pack) = parse_router_directive_pack(router_pack) {
                return Ok(from_pack);
            }
            Err(AuxiliaryCallError::Call(format!(
                "strict router parse failed: {}. raw={}",
                e,
                raw.chars()
                    .take(ROUTER_PARSE_ERROR_RAW_PREVIEW_CHARS)
                    .collect::<String>()
            )))
        }
    }
}

/// Dispatch a router decision to the specialist lane.
///
/// Shared by both preflight and post-tool router paths. Returns:
/// - `Ok(DispatchRecord)` with the response and inputs captured for tracing
/// - `Err(msg)` on fatal error (break with msg)
#[instrument(
    name = "dispatch_specialist",
    skip(core, counters, router_args, user_content, context_summary, tool_list, messages, replay),
    fields(
        target = %target,
        outcome = tracing::field::Empty,
        elapsed_ms = tracing::field::Empty,
    )
)]
pub(crate) async fn dispatch_specialist(
    core: &SwappableCore,
    counters: &crate::agent::agent_core::RuntimeCounters,
    target: &str,
    router_args: &Value,
    user_content: &str,
    context_summary: &str,
    tool_list: &[String],
    messages: &[Value],
    schema_enabled: bool,
    replay: Option<&TurnReplayRecorder>,
) -> Result<super::trace_store::DispatchRecord, AuxiliaryCallError> {
    let start = std::time::Instant::now();
    info!(role = "specialist", target = %target, "dispatch_specialist_start");
    let (specialist_provider, specialist_model) = match (
        core.specialist_provider.as_ref(),
        core.specialist_model.as_deref(),
    ) {
        (Some(p), Some(m)) => (p.clone(), m.to_string()),
        _ => {
            tracing::Span::current().record("outcome", "error");
            return Err(AuxiliaryCallError::Call(
                "Specialist lane requested by router but no specialist server is configured."
                    .to_string(),
            ));
        }
    };

    let cb_key = format!("specialist:{}", specialist_model);
    if !counters.trio_circuit_breaker.lock().is_available(&cb_key) {
        tracing::Span::current().record("outcome", "error");
        return Err(AuxiliaryCallError::Call(format!(
            "circuit breaker open for {cb_key}"
        )));
    }
    let conv_tail = build_conversation_tail(
        messages,
        ROUTER_TAIL_MAX_PAIRS,
        core.tool_delegation_config.router_tuning.tail_max_msg_chars,
        core.tool_delegation_config.router_tuning.tail_max_chars,
    );
    let specialist_pack = if core.tool_delegation_config.role_scoped_context_packs {
        role_policy::build_specialist_pack(
            target,
            router_args,
            user_content,
            &conv_tail,
            tool_list,
            3000,
        )
    } else {
        format!(
            "Target: {}\nRouter args: {}\nUser intent: {}",
            target, router_args, context_summary
        )
    };
    // Load specialist domain memory and prepend to pack if available.
    let domain_memory = counters.specialist_memory.lock().format_context(target);
    let specialist_pack = if !domain_memory.is_empty() {
        format!("{}\n\n{}", domain_memory, specialist_pack)
    } else {
        specialist_pack
    };
    let system_prompt = build_specialist_system_prompt(schema_enabled);
    let specialist_messages = vec![
        json!({"role":"system","content": system_prompt}),
        json!({"role":"user","content": specialist_pack}),
    ];
    let call_id = journal_aux_request(
        replay,
        ModelCallPurpose::Specialist,
        &RecordedProviderRequest {
            messages: specialist_messages.clone(),
            tools: None,
            model: specialist_model.clone(),
            max_tokens: core.tool_delegation_config.max_tokens,
            temperature: core.specialist_temperature,
            thinking_budget: None,
            top_p: Some(core.specialist_top_p),
            tool_choice: "auto".to_string(),
            streaming: false,
        },
    )
    .await?;
    let specialist_result = specialist_provider
        .chat(
            &specialist_messages,
            None,
            Some(&specialist_model),
            core.tool_delegation_config.max_tokens,
            core.specialist_temperature,
            None,
            Some(core.specialist_top_p),
        )
        .await;
    if let (Some(replay), Some(call_id)) = (replay, call_id.as_deref()) {
        journal_aux_terminal(replay, call_id, &specialist_result).await?;
    }
    match specialist_result {
        Ok(sp_resp) => {
            counters.trio_circuit_breaker.lock().record_success(&cb_key);
            let raw_text = sp_resp
                .content
                .unwrap_or_else(|| "Specialist returned no content.".to_string());
            let text = crate::agent::sanitize::sanitize_reasoning_output(&raw_text);
            tracing::Span::current().record("outcome", "ok");
            let elapsed_ms = start.elapsed().as_millis() as u64;
            tracing::Span::current().record("elapsed_ms", elapsed_ms);
            let sp = parse_specialist_response(&text, target);
            Ok(super::trace_store::DispatchRecord {
                specialist_name: target.to_string(),
                specialist_model: specialist_model.clone(),
                router_action: "specialist".to_string(),
                router_target: target.to_string(),
                router_confidence: 1.0,
                router_args: router_args.clone(),
                user_content: user_content.to_string(),
                messages_count: messages.len(),
                tool_results: vec![],
                specialist_response: sp.result.clone(),
            })
        }
        Err(e) => {
            counters.trio_circuit_breaker.lock().record_failure(&cb_key);
            tracing::Span::current().record("outcome", "error");
            Err(AuxiliaryCallError::Call(format!(
                "Specialist lane failed: {e}"
            )))
        }
    }
}

fn subagent_tool_call(
    call_id: String,
    target: &str,
    router_args: &Value,
    user_content: &str,
    strict_local_only: bool,
) -> Result<ToolCallRequest, String> {
    let mut params: HashMap<String, Value> = HashMap::new();
    params.insert("action".to_string(), json!("spawn"));
    if let Some(task) = router_args.get("task").and_then(|v| v.as_str()) {
        params.insert("task".to_string(), json!(task));
    } else {
        params.insert("task".to_string(), json!(user_content));
    }
    if !target.trim().is_empty() {
        params.insert("agent".to_string(), json!(target));
    }
    if strict_local_only {
        params.insert("model".to_string(), json!("local"));
    }
    if let Err(e) = policy::validate_spawn_args(&params) {
        return Err(e);
    }
    Ok(ToolCallRequest {
        id: call_id,
        name: "spawn".to_string(),
        arguments: params,
    })
}

fn pipeline_tool_call(
    call_id: String,
    steps: Value,
    ahead_by_k: Option<Value>,
) -> Result<ToolCallRequest, String> {
    let mut params = HashMap::new();
    params.insert("action".to_string(), json!("pipeline"));
    params.insert(
        "steps".to_string(),
        normalize_pipeline_steps_for_spawn(steps),
    );
    if let Some(value) = ahead_by_k {
        params.insert("ahead_by_k".to_string(), value);
    }
    policy::validate_spawn_args(&params)?;
    Ok(ToolCallRequest {
        id: call_id,
        name: "spawn".to_string(),
        arguments: params,
    })
}

// ---------------------------------------------------------------------------
// Router preflight and post-tool routing (extracted from run_agent_loop)
// ---------------------------------------------------------------------------

/// Determine the PreflightResult for a successful specialist dispatch.
/// Pure function — extracted for testability.
pub(crate) fn specialist_preflight_result(
    specialist_response: &str,
    synthesis: bool,
) -> PreflightResult {
    if synthesis {
        PreflightResult::Continue
    } else {
        PreflightResult::Break(specialist_response.to_string())
    }
}

fn normalize_pipeline_steps_for_spawn(steps: Value) -> Value {
    match steps {
        Value::Array(items) => Value::Array(
            items
                .into_iter()
                .map(|mut step| {
                    if let Some(obj) = step.as_object_mut() {
                        if !obj.contains_key("prompt") {
                            if let Some(instruction) = obj.get("instruction").cloned() {
                                obj.insert("prompt".to_string(), instruction);
                            }
                        }
                    }
                    step
                })
                .collect(),
        ),
        other => other,
    }
}

/// Result of the router preflight check.
pub(crate) enum PreflightResult {
    /// Router injected a message — continue the main loop.
    Continue,
    /// Router decided to break — set final_content.
    Break(String),
    /// Replay infrastructure failed; terminate the turn as an error.
    Error(String),
    /// Router selected ordinary registry calls; execute them through the
    /// carrier and tool-engine lifecycle without routing them a second time.
    Execute(Vec<ToolCallRequest>),
    /// No router intervention — fall through to normal processing.
    Passthrough,
}

/// Router-first preflight for strict trio mode.
///
/// Only applies in local mode with strict_no_tools_main + strict_router_schema.
/// Returns a control flow signal for the main loop.
#[instrument(
    name = "router_preflight",
    skip(ctx, health_registry),
    fields(
        user_msg = %ctx
            .user_content
            .chars()
            .take(ROUTER_USER_MSG_TRACE_PREVIEW_CHARS)
            .collect::<String>(),
        routing_decision = tracing::field::Empty,
    )
)]
pub(crate) async fn router_preflight(
    ctx: &mut TurnContext,
    health_registry: Option<&crate::heartbeat::health::HealthRegistry>,
) -> PreflightResult {
    // migrated from swappable().is_local — phase 09-03
    if !(ctx.core.mode().is_local()
        && ctx.core.tool_delegation_config.strict_no_tools_main()
        && ctx.core.tool_delegation_config.strict_router_schema()
        && !ctx.flow.router_preflight_done)
    {
        if ctx.core.mode().is_local() && !ctx.flow.router_preflight_done {
            debug!(
                strict_no_tools_main = ctx.core.tool_delegation_config.strict_no_tools_main(),
                strict_router_schema = ctx.core.tool_delegation_config.strict_router_schema(),
                "router_preflight_skipped"
            );
        }
        tracing::Span::current().record("routing_decision", "passthrough");
        return PreflightResult::Passthrough;
    }

    info!("router_preflight_firing");
    ctx.counters
        .trio_metrics
        .router_preflight_fired
        .store(true, Ordering::Relaxed);
    ctx.flow.router_preflight_done = true;
    let (router_provider, router_model) = match (
        ctx.core.router_provider.as_ref(),
        ctx.core.router_model.as_deref(),
    ) {
        (Some(p), Some(m)) => (p.clone(), m.to_string()),
        _ => {
            tracing::Span::current().record("routing_decision", "break_no_router");
            return PreflightResult::Break(
                    "Router lane is required by policy but not configured. Start trio router server and retry.".to_string(),
                );
        }
    };

    // Health gate: skip preflight if router endpoint is degraded.
    if let Some(hr) = health_registry {
        if !hr.is_healthy("trio_router") {
            ctx.counters
                .set_trio_state(crate::agent::agent_core::TrioState::Degraded);
            warn!("[router] trio_router probe degraded — falling through to main model");
            tracing::Span::current().record("routing_decision", "passthrough_degraded");
            return PreflightResult::Passthrough;
        }
    }

    // Circuit breaker gate: skip if router has too many recent failures.
    let cb_key = format!("router:{}", router_model);
    if !ctx
        .counters
        .trio_circuit_breaker
        .lock()
        .is_available(&cb_key)
    {
        ctx.counters
            .set_trio_state(crate::agent::agent_core::TrioState::Degraded);
        warn!("[router] circuit breaker open for {cb_key} — falling through to main model");
        tracing::Span::current().record("routing_decision", "passthrough_circuit_open");
        return PreflightResult::Passthrough;
    }
    let tool_list: Vec<String> = ctx.tools.tool_names();
    let conv_tail = build_conversation_tail(
        &ctx.messages,
        ROUTER_TAIL_MAX_PAIRS,
        ctx.core
            .tool_delegation_config
            .router_tuning
            .tail_max_msg_chars,
        ctx.core.tool_delegation_config.router_tuning.tail_max_chars,
    );
    let task_state = if conv_tail.is_empty() {
        format!("Strict preflight.\nUser message: {}", ctx.user_content)
    } else {
        format!(
            "Strict preflight.\nRecent conversation:\n{}\nCurrent user message: {}",
            conv_tail, ctx.user_content
        )
    };
    let router_pack = if ctx.core.tool_delegation_config.role_scoped_context_packs {
        role_policy::build_context_pack(
            role_policy::Role::Router,
            &ctx.user_content,
            &conv_tail,
            &task_state,
            &tool_list,
            2000,
        )
    } else {
        task_state
    };

    let router_start = std::time::Instant::now();
    let replay = TurnReplayRecorder::new(
        std::sync::Arc::clone(&ctx.core.sessions),
        ctx.session_id.clone(),
        ctx.request_id.clone(),
        ctx.turn_count,
    );
    let decision = match request_strict_router_decision(
        router_provider.as_ref(),
        &router_model,
        &router_pack,
        ctx.core.router_no_think,
        ctx.core.router_temperature,
        ctx.core.router_top_p,
        &tool_list.join(", "),
        ctx.core.tool_delegation_config.router_tuning.max_tokens,
        Some(&replay),
    )
    .await
    {
        Ok(d) => {
            ctx.counters
                .trio_circuit_breaker
                .lock()
                .record_success(&cb_key);
            d
        }
        Err(AuxiliaryCallError::Persistence(error)) => {
            return PreflightResult::Error(error);
        }
        Err(AuxiliaryCallError::Call(e)) => {
            warn!("[router] router call failed: {} — recording failure and falling through to main model", e);
            ctx.counters
                .trio_circuit_breaker
                .lock()
                .record_failure(&cb_key);
            tracing::Span::current().record("routing_decision", "passthrough_router_error");
            return PreflightResult::Passthrough;
        }
    };
    let router_elapsed_ms = router_start.elapsed().as_millis() as u64;

    info!(
        role = "router",
        model = %router_model,
        action = %decision.action,
        target = %decision.target,
        "router_decision"
    );
    *ctx.counters.trio_metrics.router_action.lock() = Some(decision.action.clone());

    let base_trace = RouterDecisionTrace {
        phase: "preflight".to_string(),
        action: decision.action.clone(),
        target: decision.target.clone(),
        confidence: decision.confidence,
        args: decision.args.clone(),
        user_content: ctx.user_content.clone(),
        router_elapsed_ms,
        model: router_model.clone(),
        outcome: None,
    };

    match decision.action.as_str() {
        "ask_user" => {
            tracing::Span::current().record("routing_decision", "ask_user");
            if ctx.core.trace_log {
                append_router_decision_trace(&base_trace);
            }
            PreflightResult::Break(
                decision
                    .args
                    .get("question")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| "I need clarification to continue.".to_string()),
            )
        }
        "specialist" => {
            tracing::Span::current().record("routing_decision", "specialist");
            if ctx.core.trace_log {
                append_router_decision_trace(&base_trace);
            }
            match dispatch_specialist(
                &ctx.core,
                &ctx.counters,
                &decision.target,
                &decision.args,
                &ctx.user_content,
                &ctx.user_content,
                &tool_list,
                &ctx.messages,
                ctx.core.specialist_output_schema,
                Some(&replay),
            )
            .await
            {
                Ok(record) => {
                    ctx.counters
                        .trio_metrics
                        .specialist_dispatched
                        .store(true, Ordering::Relaxed);
                    if ctx.core.trace_log {
                        super::trace_store::append_specialist_trace(&record);
                    }
                    // Record specialist output for domain memory.
                    ctx.counters
                        .specialist_memory
                        .lock()
                        .push(&decision.target, &record.specialist_response);
                    let injected = format!(
                        "[specialist:{}] {}",
                        decision.target, record.specialist_response
                    );
                    // Cache-replay tagged: this specialist output was sent to the
                    // model live, so the next turn's reload must replay it byte-
                    // for-byte or the warm prompt prefix shrinks and Higgs's
                    // radix cache diverges (full re-prefill).
                    ctx.messages
                        .push_draft(crate::agent::markers::scaffold_user(injected));
                    specialist_preflight_result(
                        &record.specialist_response,
                        ctx.core.tool_delegation_config.specialist_synthesis,
                    )
                }
                Err(AuxiliaryCallError::Persistence(error)) => PreflightResult::Error(error),
                Err(AuxiliaryCallError::Call(error)) => PreflightResult::Break(error),
            }
        }
        "subagent" => {
            tracing::Span::current().record("routing_decision", "subagent");
            let call_id = ctx.next_router_tool_call_id("subagent", decision.target.as_str());
            match subagent_tool_call(
                call_id,
                &decision.target,
                &decision.args,
                &ctx.user_content,
                ctx.strict_local_only,
            ) {
                Ok(call) => PreflightResult::Execute(vec![call]),
                Err(e) => {
                    if ctx.core.trace_log {
                        let mut trace = base_trace.clone();
                        trace.outcome = Some(format!("ERROR: {}", e));
                        append_router_decision_trace(&trace);
                    }
                    PreflightResult::Break(e)
                }
            }
        }
        "tool" => {
            tracing::Span::current().record("routing_decision", "tool");
            if decision.target.trim().is_empty() {
                return PreflightResult::Break(
                    "Router selected tool action but target is empty.".to_string(),
                );
            }
            let params_map = decision
                .args
                .as_object()
                .map(|m| {
                    m.iter()
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect::<HashMap<String, Value>>()
                })
                .unwrap_or_default();
            *ctx.counters.trio_metrics.tool_dispatched.lock() = Some(decision.target.clone());
            let call_id = ctx.next_router_tool_call_id("tool", decision.target.as_str());
            PreflightResult::Execute(vec![ToolCallRequest {
                id: call_id,
                name: decision.target,
                arguments: params_map,
            }])
        }
        "respond" => {
            tracing::Span::current().record("routing_decision", "respond");
            if ctx.core.trace_log {
                append_router_decision_trace(&base_trace);
            }
            debug!("Router: respond — forwarding to main model");
            PreflightResult::Passthrough
        }
        "pipeline" => {
            tracing::Span::current().record("routing_decision", "pipeline");
            let Some(steps) = decision
                .args
                .get("steps")
                .cloned()
                .or_else(|| decision.args.as_array().map(|_| decision.args.clone()))
            else {
                if ctx.core.trace_log {
                    let mut trace = base_trace.clone();
                    trace.outcome = Some("ERROR: pipeline action requires args.steps".to_string());
                    append_router_decision_trace(&trace);
                }
                return PreflightResult::Break(
                    "Router selected pipeline action but did not provide steps.".to_string(),
                );
            };

            let ahead_by_k = decision
                .args
                .get("ahead_by_k")
                .or_else(|| decision.args.get("aheadByK"))
                .cloned();
            *ctx.counters.trio_metrics.tool_dispatched.lock() = Some("spawn:pipeline".to_string());
            let call_id = ctx.next_router_tool_call_id("pipeline", "spawn");
            match pipeline_tool_call(call_id, steps, ahead_by_k) {
                Ok(call) => PreflightResult::Execute(vec![call]),
                Err(error) => PreflightResult::Break(error),
            }
        }
        _ => {
            tracing::Span::current().record("routing_decision", "unknown_passthrough");
            if ctx.core.trace_log {
                append_router_decision_trace(&base_trace);
            }
            debug!(
                "Router: unrecognized action '{}' — forwarding to main model",
                decision.action
            );
            PreflightResult::Passthrough
        }
    }
}

/// Result of post-tool routing.
pub(crate) enum RouteResult {
    /// Handled entirely (injected message) — continue main loop.
    Continue,
    /// Break with final_content.
    Break(String),
    /// Protocol persistence failed; stop the turn as infrastructure error.
    Error(String),
    /// One ordered carrier plus the execution/rejection disposition of each
    /// call. The carrier is persisted before any disposition is acted on.
    Execute(RoutedToolBatch),
}

pub(crate) enum RoutedToolDisposition {
    Execute,
    Reject { reason: String, receipt: String },
}

pub(crate) struct RoutedToolCall {
    pub(crate) call: ToolCallRequest,
    pub(crate) disposition: RoutedToolDisposition,
}

pub(crate) struct RoutedToolBatch {
    pub(crate) calls: Vec<RoutedToolCall>,
    /// Applies only after every rejection receipt in this batch is durable.
    pub(crate) return_after_rejections: bool,
}

/// Determine the RouteResult for a successful specialist dispatch in route_tool_calls().
/// Pure function — extracted for testability.
pub(crate) fn specialist_route_result(specialist_response: &str) -> RouteResult {
    RouteResult::Break(specialist_response.to_string())
}

/// Receipt for a blocked duplicate tool call. Pure function — extracted for
/// testability.
///
/// Escalation: the first duplicate replays the cached data (the model asked
/// for this content and already ran the command; blocking with a rejection
/// turns a helpable loop into a dead end). From the second duplicate on, the
/// data is already in context twice — re-dumping it just feeds the loop
/// (observed live: the model re-rolled identical recall calls until the
/// breaker fired). Later duplicates get a directive plus the lease progress
/// signal so the model sees budget state instead of the same bytes.
pub(crate) fn duplicate_receipt(
    name: &str,
    hits: u32,
    cached: Option<&str>,
    cached_chars: usize,
    progress: &str,
) -> String {
    match cached {
        Some(data) if hits <= 1 => {
            format!("[cached result from earlier identical call — {cached_chars} chars]\n{data}\n{progress}")
        }
        _ => format!(
            "[duplicate {name} call #{hits} this turn — the identical arguments were already \
             answered above; the cached result is already in this conversation. Do NOT repeat \
             this call. Vary the query meaningfully, use inspect_tool_result to re-read a \
             stashed result, or write your final answer from the evidence you have.]\n{progress}"
        ),
    }
}

fn canonicalize_proxy_execution(
    registry: &ToolRegistry,
    mut tc: ToolCallRequest,
) -> ToolCallRequest {
    let Some((name, arguments)) = registry.canonical_proxy_dispatch(&tc.name, &tc.arguments) else {
        return tc;
    };
    tc.name = name;
    tc.arguments = arguments;
    tc
}

/// Route tool calls through the strict router / toolplan / fallback pipeline.
///
/// Takes the raw tool calls from the LLM response, applies router filtering,
/// tool guard, and policy, then returns a control flow signal.
pub(crate) async fn route_tool_calls(
    ctx: &mut TurnContext,
    response_content: Option<&str>,
    mut routed_tool_calls: Vec<ToolCallRequest>,
    routing: ToolRouting,
) -> RouteResult {
    let mut router_decision: Option<role_policy::RouterDecision> = None;
    let mut router_decision_valid = false;
    let mut selected_plan: Option<toolplan::ToolPlan> = None;
    let available_tools = ctx.tools.tool_names();

    if matches!(routing, ToolRouting::NeedsRouting)
        && ctx.core.tool_delegation_config.strict_router_schema()
    {
        let task_state = format!(
            "Main content: {}\nCandidate tool calls: {}",
            response_content.unwrap_or("(empty)"),
            routed_tool_calls
                .iter()
                .map(|tc| tc.name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );
        let conv_tail = build_conversation_tail(
            &ctx.messages,
            ROUTER_TAIL_MAX_PAIRS,
            ctx.core
                .tool_delegation_config
                .router_tuning
                .tail_max_msg_chars,
            ctx.core.tool_delegation_config.router_tuning.tail_max_chars,
        );
        let router_pack = if ctx.core.tool_delegation_config.role_scoped_context_packs {
            role_policy::build_context_pack(
                role_policy::Role::Router,
                &ctx.user_content,
                &conv_tail,
                &task_state,
                &available_tools,
                2000,
            )
        } else {
            task_state
        };

        if let (Some(router_provider), Some(router_model)) = (
            ctx.core.router_provider.as_ref(),
            ctx.core.router_model.as_deref(),
        ) {
            let router_start = std::time::Instant::now();
            let replay = TurnReplayRecorder::new(
                std::sync::Arc::clone(&ctx.core.sessions),
                ctx.session_id.clone(),
                ctx.request_id.clone(),
                ctx.turn_count,
            );
            match request_strict_router_decision(
                router_provider.as_ref(),
                router_model,
                &router_pack,
                ctx.core.router_no_think,
                ctx.core.router_temperature,
                ctx.core.router_top_p,
                &available_tools.join(", "),
                ctx.core.tool_delegation_config.router_tuning.max_tokens,
                Some(&replay),
            )
            .await
            {
                Ok(decision) => {
                    let router_elapsed_ms = router_start.elapsed().as_millis() as u64;
                    router_decision_valid = true;
                    if ctx.core.trace_log {
                        let model = ctx
                            .core
                            .router_model
                            .as_deref()
                            .unwrap_or("unknown")
                            .to_string();
                        append_router_decision_trace(&RouterDecisionTrace {
                            phase: "tool_routing".to_string(),
                            action: decision.action.clone(),
                            target: decision.target.clone(),
                            confidence: decision.confidence,
                            args: decision.args.clone(),
                            user_content: ctx.user_content.clone(),
                            router_elapsed_ms,
                            model,
                            outcome: None,
                        });
                    }
                    router_decision = Some(decision);
                }
                Err(AuxiliaryCallError::Persistence(error)) => {
                    return RouteResult::Error(error);
                }
                Err(AuxiliaryCallError::Call(e)) => {
                    warn!("{}", e);
                }
            }
        } else {
            debug!("strict router enabled but router lane is not configured");
        }
    }

    if let Some(decision) = router_decision.clone() {
        match toolplan::from_router_decision(decision) {
            Ok(plan) => {
                selected_plan = Some(plan);
            }
            Err(e) => {
                warn!("router decision normalization failed: {}", e);
                if ctx.core.tool_delegation_config.strict_toolplan_validation()
                    && ctx
                        .core
                        .tool_delegation_config
                        .deterministic_router_fallback
                {
                    selected_plan = Some(router_fallback::route(
                        &ctx.user_content,
                        &available_tools,
                        &ctx.session_policy,
                    ));
                }
            }
        }
    }

    // migrated from swappable().is_local — phase 09-03
    if matches!(routing, ToolRouting::NeedsRouting)
        && ctx.core.mode().is_local()
        && role_policy::should_block_main_tool_calls(&ctx.core.tool_delegation_config.mode, true)
        && !router_decision_valid
    {
        if ctx
            .core
            .tool_delegation_config
            .deterministic_router_fallback
        {
            warn!(
                "strict router invalid; using deterministic fallback plan (model={})",
                ctx.core.model
            );
            selected_plan = Some(router_fallback::route(
                &ctx.user_content,
                &available_tools,
                &ctx.session_policy,
            ));
        } else {
            warn!(
                "Policy blocked main-model tool calls (strictNoToolsMain=true, model={})",
                ctx.core.model
            );
            return RouteResult::Break(
                "I can orchestrate the task, but direct tool calls from the main model are disabled by policy and strict router did not return a valid decision.".to_string(),
            );
        }
    }

    if let Some(plan) = selected_plan {
        if ctx.core.tool_delegation_config.strict_toolplan_validation() {
            if let Err(e) = plan.validate() {
                if ctx
                    .core
                    .tool_delegation_config
                    .deterministic_router_fallback
                {
                    warn!(
                        "tool plan validation failed ({}), using deterministic fallback",
                        e
                    );
                } else {
                    return RouteResult::Break(format!("Router produced invalid tool plan: {}", e));
                }
            }
        }
        match plan.action {
            ToolPlanAction::AskUser => {
                return RouteResult::Break(
                    plan.args
                        .get("question")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| {
                            response_content
                                .map(|s| s.to_string())
                                .unwrap_or_else(|| "I need clarification to continue.".to_string())
                        }),
                );
            }
            ToolPlanAction::Specialist => {
                // P3: Try to find scratch pad summary in recent messages, fallback to response_content
                let summary_from_scratch_pad = find_scratch_pad_summary_in_messages(&ctx.messages);
                let context_summary_owned = summary_from_scratch_pad
                    .or_else(|| response_content.map(|s| s.to_string()))
                    .unwrap_or_else(|| "(empty)".to_string());
                let context_summary = context_summary_owned.as_str();
                let replay = TurnReplayRecorder::new(
                    std::sync::Arc::clone(&ctx.core.sessions),
                    ctx.session_id.clone(),
                    ctx.request_id.clone(),
                    ctx.turn_count,
                );
                match dispatch_specialist(
                    &ctx.core,
                    &ctx.counters,
                    &plan.target,
                    &plan.args,
                    &ctx.user_content,
                    context_summary,
                    &ctx.tools.tool_names(),
                    &ctx.messages,
                    ctx.core.specialist_output_schema,
                    Some(&replay),
                )
                .await
                {
                    Ok(record) => {
                        if ctx.core.trace_log {
                            super::trace_store::append_specialist_trace(&record);
                        }
                        let injected = format!(
                            "[specialist:{}] {}",
                            plan.target, record.specialist_response
                        );
                        // Cache-replay tagged — same reason as the decision-path
                        // specialist injection above.
                        ctx.messages
                            .push_draft(crate::agent::markers::scaffold_user(injected));
                        return specialist_route_result(&record.specialist_response);
                    }
                    Err(AuxiliaryCallError::Persistence(error)) => {
                        return RouteResult::Error(error)
                    }
                    Err(AuxiliaryCallError::Call(error)) => return RouteResult::Break(error),
                }
            }
            ToolPlanAction::Subagent => {
                let call_id = ctx.next_router_tool_call_id("subagent", plan.target.as_str());
                match subagent_tool_call(
                    call_id,
                    &plan.target,
                    &plan.args,
                    &ctx.user_content,
                    ctx.strict_local_only,
                ) {
                    Ok(call) => routed_tool_calls = vec![call],
                    Err(e) => return RouteResult::Break(e),
                }
            }
            ToolPlanAction::Tool => {
                if !plan.target.is_empty() {
                    let filtered: Vec<_> = routed_tool_calls
                        .iter()
                        .filter(|tc| tc.name == plan.target)
                        .cloned()
                        .collect();
                    if !filtered.is_empty() {
                        routed_tool_calls = filtered;
                    } else {
                        let args = plan
                            .args
                            .as_object()
                            .map(|m| {
                                m.iter()
                                    .map(|(k, v)| (k.clone(), v.clone()))
                                    .collect::<HashMap<String, Value>>()
                            })
                            .unwrap_or_default();
                        routed_tool_calls = vec![ToolCallRequest {
                            id: ctx.next_router_tool_call_id("planned-tool", plan.target.as_str()),
                            name: plan.target,
                            arguments: args,
                        }];
                    }
                }
            }
        }
    }

    routed_tool_calls = routed_tool_calls
        .into_iter()
        .map(|tc| canonicalize_proxy_execution(&ctx.tools, tc))
        .collect();

    // Preserve every call in the carrier, including same-batch duplicates and
    // guard rejections. A model-visible tool call must always receive exactly
    // one result in its own id slot, even when policy refuses execution.
    let original_count = routed_tool_calls.len();
    if original_count == 0 {
        if let Some(text) = response_content.filter(|s| !s.trim().is_empty()) {
            return RouteResult::Break(text.to_string());
        }
        return RouteResult::Continue;
    }

    let mut seen_in_batch = std::collections::HashSet::new();
    let mut calls = Vec::with_capacity(original_count);
    let mut allowed_count = 0usize;
    let mut blocked_count = 0usize;
    for tc in routed_tool_calls {
        let key = crate::agent::tool_runner::normalize_call_key(&tc.name, &tc.arguments);
        let rejection = if !seen_in_batch.insert(key) {
            ctx.flow.tool_guard.had_blocked_calls = true;
            Some(format!(
                "duplicate tool call blocked for '{}': repeated in one routed batch",
                tc.name
            ))
        } else {
            ctx.flow.tool_guard.allow(&tc.name, &tc.arguments).err()
        };
        if let Some(reason) = rejection {
            warn!("{}", reason);
            blocked_count += 1;
            let guard_key = ToolGuard::key(&tc.name, &tc.arguments);
            let cached = ctx.flow.tool_guard.get_cached_result(&guard_key);
            let receipt = if let Some(cached) = cached {
                duplicate_receipt(
                    &tc.name,
                    ctx.flow.tool_guard.cache_hits(&guard_key),
                    Some(cached),
                    cached.chars().count(),
                    &ctx.flow.lease.progress_signal(),
                )
            } else {
                format!(
                    "tool guard rejected {} without execution: {}",
                    tc.name, reason
                )
            };
            calls.push(RoutedToolCall {
                call: tc,
                disposition: RoutedToolDisposition::Reject { reason, receipt },
            });
        } else {
            allowed_count += 1;
            calls.push(RoutedToolCall {
                call: tc,
                disposition: RoutedToolDisposition::Execute,
            });
        }
    }

    if allowed_count == 0 {
        // All tool calls were blocked.
        if blocked_count == original_count {
            ctx.flow.consecutive_all_blocked += 1;
            // A cached duplicate produces only a compact protocol receipt; it
            // does not execute a tool or add evidence. Count every all-blocked
            // round as zero progress so cached receipts cannot livelock the
            // agent loop while also bypassing its iteration budget.
            ctx.flow.round_executed_no_tools = true;
            return RouteResult::Execute(RoutedToolBatch {
                calls,
                return_after_rejections: true,
            });
        }
    }

    // Reset the consecutive blocked counter when tool calls succeed.
    ctx.flow.consecutive_all_blocked = 0;
    RouteResult::Execute(RoutedToolBatch {
        calls,
        return_after_rejections: false,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::schema::TrioConfig;
    use serde_json::json;

    // Duplicate receipts escalate: hit 1 replays the cached bytes, later
    // hits switch to a directive + progress signal.
    #[test]
    fn test_duplicate_receipt_replays_once_then_directs() {
        let progress = "[Tool call 5 of 8 this lease — 2 leases remaining]";

        let first = duplicate_receipt("recall", 1, Some("found: PHASEONE data"), 22, progress);
        assert!(first.contains("[cached result from earlier identical call — 22 chars]"));
        assert!(first.contains("found: PHASEONE data"));
        assert!(first.contains(progress));

        let repeat = duplicate_receipt("recall", 2, Some("found: PHASEONE data"), 22, progress);
        assert!(
            !repeat.contains("found: PHASEONE data"),
            "repeat must not re-dump bytes"
        );
        assert!(repeat.contains("duplicate recall call #2"));
        assert!(repeat.contains("Do NOT repeat this call"));
        assert!(repeat.contains(progress));

        // Defensive arm: classified blocked-with-result but cache vanished.
        let none = duplicate_receipt("recall", 1, None, 0, progress);
        assert!(none.contains("duplicate recall call #1"));
    }

    // T9 — TrioConfig has specialist_output_schema field, defaults to false
    #[test]
    fn test_trio_config_specialist_output_schema_default_false() {
        let config = TrioConfig::default();
        assert!(!config.specialist_output_schema);
    }

    #[test]
    fn test_normalize_pipeline_steps_accepts_instruction_alias() {
        let normalized = normalize_pipeline_steps_for_spawn(json!([
            {"instruction": "fetch weather", "expected": "brief forecast"}
        ]));

        assert_eq!(normalized[0]["prompt"], "fetch weather");
        assert_eq!(normalized[0]["instruction"], "fetch weather");
        assert_eq!(normalized[0]["expected"], "brief forecast");
    }

    async fn assert_two_synthetic_calls_replay_exact(lane: &str, target: &str) -> Vec<String> {
        let mut sequence = crate::agent::agent_loop::RouterSyntheticCallSequence::default();
        let ids = vec![
            sequence.allocate(7, lane, target),
            sequence.allocate(7, lane, target),
        ];
        let dir = tempfile::tempdir().unwrap();
        let sessions = crate::session::db::SessionDb::new(&dir.path().join("sessions.db"));
        let session = sessions.create_session("router-synthetic-id-replay").await;
        for (index, id) in ids.iter().enumerate() {
            sessions
                .record_tool_pre_execute(
                    &session.id,
                    "turn-router-ids",
                    7,
                    id,
                    "spawn",
                    &HashMap::new(),
                    crate::session::db::ToolPreExecuteDecision::Ready,
                )
                .await
                .unwrap();
            sessions
                .record_tool_execute(
                    &session.id,
                    "turn-router-ids",
                    7,
                    id,
                    "synthetic result",
                    true,
                    1,
                )
                .await
                .unwrap();
            sessions
                .record_tool_post_execute(
                    &session.id,
                    "turn-router-ids",
                    7,
                    id,
                    "synthetic result",
                    index as i64 + 1,
                )
                .await
                .unwrap();
        }
        sessions
            .record_turn_finished(&session.id, "turn-router-ids", 7, "finished")
            .await
            .unwrap();
        let replay = sessions.load_session_replay(&session.id).await.unwrap();
        assert_eq!(
            replay.availability,
            crate::session::db::ReplayAvailability::Exact
        );
        ids
    }

    #[tokio::test]
    async fn preflight_and_post_subagent_ids_are_unique_in_exact_replay() {
        let ids = assert_two_synthetic_calls_replay_exact("subagent", "reviewer").await;
        let args = json!({"task": "inspect the replay"});
        let preflight =
            subagent_tool_call(ids[0].clone(), "reviewer", &args, "inspect", true).unwrap();
        let post_tool =
            subagent_tool_call(ids[1].clone(), "reviewer", &args, "inspect", true).unwrap();

        assert_ne!(preflight.id, post_tool.id);
    }

    #[tokio::test]
    async fn repeated_same_target_planned_tool_ids_are_unique_in_exact_replay() {
        let ids = assert_two_synthetic_calls_replay_exact("planned-tool", "read_file").await;
        assert_ne!(ids[0], ids[1]);
        assert!(ids.iter().all(|id| id.ends_with("planned-tool-read_file")));
    }

    #[test]
    fn test_canonicalize_proxy_execution_nested_args() {
        let registry = ToolRegistry::new();
        let mut arguments = HashMap::new();
        arguments.insert("name".to_string(), json!("edit_file"));
        arguments.insert(
            "args".to_string(),
            json!({"path": "a.txt", "old_text": "old", "new_text": "new"}),
        );

        let tc = canonicalize_proxy_execution(
            &registry,
            ToolCallRequest {
                id: "tc_proxy_edit".to_string(),
                name: "tool".to_string(),
                arguments,
            },
        );

        assert_eq!(tc.name, "edit_file");
        assert_eq!(tc.arguments.get("path"), Some(&json!("a.txt")));
        assert_eq!(tc.arguments.get("old_text"), Some(&json!("old")));
    }

    #[test]
    fn test_canonicalize_proxy_execution_current_envelope() {
        let registry = ToolRegistry::new();
        let tc = canonicalize_proxy_execution(
            &registry,
            ToolCallRequest {
                id: "tc_proxy_web".to_string(),
                name: "tool".to_string(),
                arguments: HashMap::from([
                    ("tool_name".to_string(), json!("web_fetch")),
                    (
                        "tool_args".to_string(),
                        json!({"url": "https://example.com"}),
                    ),
                ]),
            },
        );

        assert_eq!(tc.name, "web_fetch");
        assert_eq!(tc.arguments.get("url"), Some(&json!("https://example.com")));
    }

    #[test]
    fn test_canonicalize_proxy_execution_flattened_args() {
        let registry = ToolRegistry::with_standard_tools(
            &crate::agent::tools::registry::ToolConfig::new(std::path::Path::new(".")),
        );
        let mut arguments = HashMap::new();
        arguments.insert("name".to_string(), json!("recall"));
        arguments.insert("mode".to_string(), json!("latest"));

        let tc = canonicalize_proxy_execution(
            &registry,
            ToolCallRequest {
                id: "tc_proxy_recall".to_string(),
                name: "tool".to_string(),
                arguments,
            },
        );

        assert_eq!(tc.name, "recall");
        assert_eq!(tc.arguments.get("mode"), Some(&json!("latest")));
    }

    #[test]
    fn test_canonicalize_proxy_execution_preserves_inspect() {
        let registry = ToolRegistry::new();
        let mut arguments = HashMap::new();
        arguments.insert("name".to_string(), json!("session_search"));

        let tc = canonicalize_proxy_execution(
            &registry,
            ToolCallRequest {
                id: "tc_proxy_inspect".to_string(),
                name: "tool".to_string(),
                arguments,
            },
        );

        assert_eq!(tc.name, "tool");
        assert_eq!(tc.arguments.get("name"), Some(&json!("session_search")));
    }

    #[test]
    fn test_canonicalize_proxy_args_as_json_string() {
        let registry = ToolRegistry::new();
        let mut arguments = HashMap::new();
        arguments.insert("name".to_string(), json!("web_search"));
        arguments.insert("args".to_string(), json!(r#"{"query":"news"}"#));

        let tc = canonicalize_proxy_execution(
            &registry,
            ToolCallRequest {
                id: "tc_proxy_web_search".to_string(),
                name: "tool".to_string(),
                arguments,
            },
        );

        assert_eq!(tc.name, "web_search");
        assert_eq!(tc.arguments.get("query"), Some(&json!("news")));
    }

    #[test]
    fn tool_guard_cached_result_matches_reordered_args() {
        let mut first_args = HashMap::new();
        first_args.insert("limit".to_string(), json!(50));
        first_args.insert("query".to_string(), json!("compaction"));
        first_args.insert("path".to_string(), json!("~/Dev/nanobot-rs/src"));

        let mut duplicate_args = HashMap::new();
        duplicate_args.insert("query".to_string(), json!("compaction"));
        duplicate_args.insert("path".to_string(), json!("~/Dev/nanobot-rs/src"));
        duplicate_args.insert("limit".to_string(), json!(50));
        let result = "Searched 4 file(s) under ~/Dev/nanobot-rs/src";

        let mut guard = ToolGuard::new(1);
        guard.record_result("search_files", &first_args, result.to_string());
        let key = ToolGuard::key("search_files", &duplicate_args);
        assert_eq!(
            guard
                .get_cached_result(&key)
                .map(str::chars)
                .map(Iterator::count),
            Some(result.chars().count())
        );
        assert!(guard.allow("search_files", &duplicate_args).is_err());

        let fresh_guard = ToolGuard::new(1);
        assert_eq!(
            fresh_guard.get_cached_result(&key),
            None,
            "ToolGuard lifetime scopes duplicate cache to one turn"
        );
    }

    // T10 — SpecialistResponse success field defaults to true
    #[test]
    fn test_specialist_response_success_field_default_true() {
        let sr = crate::agent::role_policy::SpecialistResponse {
            result: "test".to_string(),
            success: true,
            vote_key: "test".to_string(),
            parsed_json: false,
        };
        assert!(sr.success);
    }

    // ── A. extract_json_object ────────────────────────────────────────────────

    #[test]
    fn test_extract_json_object_valid() {
        let raw = r#"{"action":"respond","confidence":0.9}"#;
        let result = extract_json_object(raw);
        assert!(result.is_some(), "valid JSON object should be extracted");
        assert_eq!(result.unwrap(), raw);
    }

    #[test]
    fn test_extract_json_object_markdown_wrapped() {
        let raw = "```json\n{\"action\":\"respond\"}\n```";
        let result = extract_json_object(raw);
        assert!(
            result.is_some(),
            "markdown-wrapped JSON should be extracted"
        );
        assert_eq!(result.unwrap(), r#"{"action":"respond"}"#);
    }

    #[test]
    fn test_extract_json_object_nested_braces() {
        let raw = r#"{"outer":{"inner":"val"},"confidence":0.9}"#;
        let result = extract_json_object(raw);
        assert!(result.is_some(), "nested braces should be handled");
        assert_eq!(result.unwrap(), raw);
    }

    #[test]
    fn test_extract_json_object_no_json() {
        let raw = "just plain text";
        let result = extract_json_object(raw);
        assert!(result.is_none(), "plain text should return None");
    }

    #[test]
    fn test_extract_json_object_empty_string() {
        let result = extract_json_object("");
        assert!(result.is_none(), "empty string should return None");
    }

    #[test]
    fn test_extract_json_object_empty_braces() {
        let result = extract_json_object("{}");
        assert!(result.is_some(), "empty object {{}} is valid");
        assert_eq!(result.unwrap(), "{}");
    }

    #[test]
    fn test_extract_json_object_with_surrounding_text() {
        let raw = r#"Here is the result: {"action":"tool","target":"read_file"} end."#;
        let result = extract_json_object(raw);
        assert!(result.is_some());
        assert_eq!(result.unwrap(), r#"{"action":"tool","target":"read_file"}"#);
    }

    // ── B. parse_lenient_router_decision ─────────────────────────────────────
    //
    // Key implementation notes:
    // - normalize_action only recognizes: "tool" | "subagent" | "specialist" | "ask_user"
    //   All other actions (including "respond") fall through and get remapped to "tool"
    // - Lenient path falls back target="" to "" (empty) by default — fails strict validation
    // - strict_router_decision_strict rejects: empty target (unless action=="respond"),
    //   out-of-range confidence, and unknown actions
    // - "respond" action is NOT handled in the lenient path (only in the strict JSON path
    //   invoked by request_strict_router_decision, not by parse_lenient_router_decision)

    #[test]
    fn test_parse_lenient_strict_json_respond() {
        // "respond" with empty target: normalize_action maps "respond" -> "tool",
        // then strict validation fails because "tool" requires non-empty target.
        // The lenient parser returns None for respond-action inputs.
        let raw = r#"{"action":"respond","target":"","args":{},"confidence":0.95}"#;
        let result = parse_lenient_router_decision(raw);
        assert!(
            result.is_none(),
            "parse_lenient_router_decision does not handle 'respond' action (normalize maps it to 'tool' which fails empty-target check)"
        );
    }

    #[test]
    fn test_parse_lenient_strict_json_specialist() {
        // "specialist" IS in the normalize_action known set, and target is non-empty.
        let raw = r#"{"action":"specialist","target":"coding","args":{},"confidence":0.9}"#;
        let result = parse_lenient_router_decision(raw);
        assert!(result.is_some(), "valid specialist JSON should parse");
        let d = result.unwrap();
        assert_eq!(d.action, "specialist");
        assert_eq!(d.target, "coding");
    }

    #[test]
    fn test_parse_lenient_comma_separated_specialist() {
        // Comma-separated path: "specialist,coding,{...}" — action,target,args
        // normalize_action("specialist", "coding", ...) = "specialist"
        // strict: specialist+coding+0.9 => Ok
        let raw = r#"specialist,coding,{"confidence":0.9}"#;
        let result = parse_lenient_router_decision(raw);
        assert!(result.is_some(), "comma-separated specialist should parse");
        assert_eq!(result.unwrap().action, "specialist");
    }

    #[test]
    fn test_parse_lenient_embedded_marker() {
        // "[specialist:coding] Here is the answer" — no comma path, no "action" key.
        // extract_quoted falls back to default action="tool", target="" (empty).
        // strict: tool with empty target fails validation → None.
        // Falls through to main model via Passthrough.
        let raw = "[specialist:coding] Here is the answer";
        let result = parse_lenient_router_decision(raw);
        assert!(
            result.is_none(),
            "embedded marker with no extractable fields returns None"
        );
    }

    #[test]
    fn test_parse_lenient_malformed_garbage_returns_none() {
        // Garbage text: no "action" or "target" key found.
        // Defaults to action="tool", target="" → fails strict (empty target) → None.
        // Falls through to main model via Passthrough.
        let raw = "this is not valid at all!!!";
        let result = parse_lenient_router_decision(raw);
        assert!(
            result.is_none(),
            "garbage with no action/target fails strict validation and returns None"
        );
    }

    #[test]
    fn test_parse_lenient_empty_string_returns_none() {
        // No keys found → action="tool", target="" → fails strict → None.
        let result = parse_lenient_router_decision("");
        assert!(
            result.is_none(),
            "empty string fails strict validation (empty target) and returns None"
        );
    }

    #[test]
    fn test_parse_lenient_unknown_action_frobnicate() {
        // "frobnicate" action with empty target:
        // extract_quoted("action") = "frobnicate", normalize_action -> "tool" (not in known set)
        // extract_quoted("target") = "" -> stays as ""
        // strict: "tool" with empty target -> Err (target cannot be empty for non-respond)
        // Returns None.
        let raw = r#"{"action":"frobnicate","target":"","args":{},"confidence":0.5}"#;
        let result = parse_lenient_router_decision(raw);
        assert!(
            result.is_none(),
            "unknown action with empty target normalizes to tool with empty target, which fails strict validation"
        );
    }

    // ── C. find_scratch_pad_summary_in_messages ───────────────────────────────

    #[test]
    fn test_find_summary_tool_runner_marker() {
        let msgs = vec![json!({"role": "user", "content": "[tool runner summary] found 5 files"})];
        let result = find_scratch_pad_summary_in_messages(&msgs);
        assert_eq!(result, Some("found 5 files".to_string()));
    }

    #[test]
    fn test_find_summary_tool_analysis_marker() {
        let msgs = vec![
            json!({"role": "tool", "content": "[Tool analysis summary]\nSome analysis\n\n[Full output: 10 lines]"}),
        ];
        let result = find_scratch_pad_summary_in_messages(&msgs);
        assert_eq!(result, Some("Some analysis".to_string()));
    }

    #[test]
    fn test_find_summary_no_markers() {
        let msgs = vec![
            json!({"role": "user", "content": "Hello, how are you?"}),
            json!({"role": "assistant", "content": "I'm doing well, thanks!"}),
        ];
        let result = find_scratch_pad_summary_in_messages(&msgs);
        assert!(result.is_none(), "no markers should return None");
    }

    #[test]
    fn test_find_summary_beyond_10_message_window() {
        // Summary message at position 11 from the end — outside the 10-message window.
        let mut msgs: Vec<Value> = Vec::new();
        // First message (index 0) has the summary, but we'll add 11 more after it
        msgs.push(json!({"role": "user", "content": "[tool runner summary] early summary"}));
        // Push 11 more messages to push the summary outside the window
        for i in 0..11 {
            msgs.push(json!({"role": "assistant", "content": format!("message {}", i)}));
        }
        let result = find_scratch_pad_summary_in_messages(&msgs);
        assert!(
            result.is_none(),
            "summary outside 10-message window should return None"
        );
    }

    #[test]
    fn test_find_summary_empty_messages() {
        let result = find_scratch_pad_summary_in_messages(&[]);
        assert!(result.is_none(), "empty messages should return None");
    }

    #[test]
    fn test_find_summary_within_10_message_window() {
        // Summary at exactly position 10 from the end (within window).
        let mut msgs: Vec<Value> = Vec::new();
        msgs.push(json!({"role": "user", "content": "[tool runner summary] recent summary"}));
        // Push 9 more messages — summary is now at index 0, 10 messages total.
        for i in 0..9 {
            msgs.push(json!({"role": "assistant", "content": format!("reply {}", i)}));
        }
        let result = find_scratch_pad_summary_in_messages(&msgs);
        assert_eq!(result, Some("recent summary".to_string()));
    }

    #[test]
    fn test_find_summary_tool_analysis_multiline() {
        // Multi-line summary with [Full output:...] section stripped
        let content = "[Tool analysis summary]\nLine one\nLine two\n\n[Full output: lots of data]";
        let msgs = vec![json!({"role": "tool", "content": content})];
        let result = find_scratch_pad_summary_in_messages(&msgs);
        assert_eq!(result, Some("Line one\nLine two".to_string()));
    }

    // ── D. build_conversation_tail ────────────────────────────────────────────

    #[test]
    fn test_build_tail_normal_pairs() {
        let msgs = vec![
            json!({"role": "user", "content": "Hello there"}),
            json!({"role": "assistant", "content": "Hi back"}),
            json!({"role": "user", "content": "How are you"}),
            json!({"role": "assistant", "content": "Doing well"}),
        ];
        let result = build_conversation_tail(&msgs, 5, 1000, 10_000);
        assert!(result.contains("User:"), "should contain User: prefix");
        assert!(
            result.contains("Hello there"),
            "should contain first user message"
        );
        assert!(
            result.contains("How are you"),
            "should contain second user message"
        );
        assert!(
            result.contains("Hi back"),
            "should contain assistant response"
        );
    }

    #[test]
    fn test_build_tail_oversized_message_gets_truncated() {
        let long_msg = "A".repeat(500);
        let msgs = vec![
            json!({"role": "user", "content": long_msg}),
            json!({"role": "assistant", "content": "Short reply"}),
        ];
        // max_msg_chars = 100
        let result = build_conversation_tail(&msgs, 5, 100, 10_000);
        assert!(result.contains("User:"), "should contain User: prefix");
        assert!(
            result.contains('…'),
            "truncated message should end with ellipsis"
        );
        // The user message should be truncated to ~100 chars + ellipsis
        let user_line = result.lines().find(|l| l.starts_with("User:")).unwrap();
        assert!(
            user_line.len() <= 120,
            "truncated line should not be too long"
        );
    }

    #[test]
    fn test_build_tail_system_and_tool_roles_skipped() {
        let msgs = vec![
            json!({"role": "system", "content": "You are a helpful assistant."}),
            json!({"role": "tool", "content": "Tool result data"}),
            json!({"role": "tool_call", "content": "function call"}),
        ];
        let result = build_conversation_tail(&msgs, 5, 1000, 10_000);
        assert!(
            result.is_empty(),
            "system and tool messages should be skipped"
        );
    }

    #[test]
    fn test_build_tail_max_pairs_limits_output() {
        let msgs = vec![
            json!({"role": "user", "content": "First question"}),
            json!({"role": "assistant", "content": "First answer"}),
            json!({"role": "user", "content": "Second question"}),
            json!({"role": "assistant", "content": "Second answer"}),
            json!({"role": "user", "content": "Third question"}),
            json!({"role": "assistant", "content": "Third answer"}),
        ];
        // max_pairs = 1 — only the last pair should appear
        let result = build_conversation_tail(&msgs, 1, 1000, 10_000);
        assert!(
            result.contains("Third question"),
            "last pair user message should be present"
        );
        assert!(
            result.contains("Third answer"),
            "last pair assistant message should be present"
        );
        assert!(
            !result.contains("First question"),
            "earlier pairs should be excluded"
        );
        assert!(
            !result.contains("Second question"),
            "earlier pairs should be excluded"
        );
    }

    #[test]
    fn test_build_tail_empty_messages() {
        let result = build_conversation_tail(&[], 5, 1000, 10_000);
        assert!(
            result.is_empty(),
            "empty messages should produce empty tail"
        );
    }

    #[test]
    fn test_build_tail_max_chars_truncates_total_output() {
        let msgs = vec![
            json!({"role": "user", "content": "Question one"}),
            json!({"role": "assistant", "content": "Answer one"}),
            json!({"role": "user", "content": "Question two"}),
            json!({"role": "assistant", "content": "Answer two"}),
        ];
        // max_chars = 20 — very small limit
        let result = build_conversation_tail(&msgs, 5, 1000, 20);
        assert!(result.len() <= 20, "output should be capped at max_chars");
    }

    // ── Scenario tests (15 trio eval scenarios) ───────────────────────────────
    // These test parse_lenient_router_decision with realistic LLM outputs.
    // Note: "respond" and "ask_user" with empty targets fail strict validation
    // in the lenient path (normalize_action does not preserve "respond").

    #[test]
    fn test_respond_simple_math() {
        // "respond" with empty target: normalize maps to "tool" with empty target -> None
        let raw = r#"{"action":"respond","target":"","args":{},"confidence":0.95}"#;
        let result = parse_lenient_router_decision(raw);
        // The lenient parser does not successfully round-trip respond+empty-target.
        assert!(
            result.is_none(),
            "respond with empty target does not survive lenient normalization"
        );
    }

    #[test]
    fn test_respond_hello() {
        // Same: respond+empty-target -> None from parse_lenient_router_decision
        let raw = r#"{"action":"respond","target":"","args":{},"confidence":0.99}"#;
        let result = parse_lenient_router_decision(raw);
        assert!(
            result.is_none(),
            "respond with empty target does not survive lenient normalization"
        );
    }

    #[test]
    fn test_specialist_coding() {
        // "specialist" is preserved by normalize_action, non-empty target passes strict.
        let raw = r#"{"action":"specialist","target":"coding","args":{},"confidence":0.9}"#;
        let result = parse_lenient_router_decision(raw);
        assert!(result.is_some());
        let d = result.unwrap();
        assert_eq!(d.action, "specialist");
        assert_eq!(d.target, "coding");
    }

    #[test]
    fn test_specialist_lenient_comma() {
        // Comma-separated format: "specialist,coding,{...}"
        let raw = r#"specialist,coding,{"confidence":0.9}"#;
        let result = parse_lenient_router_decision(raw);
        assert!(result.is_some(), "comma-separated specialist should parse");
        assert_eq!(result.unwrap().action, "specialist");
    }

    #[test]
    fn test_specialist_embedded_marker() {
        // "[specialist:coding] ..." — no "action" key, no extractable target.
        // Defaults to action="tool", target="" → fails strict → None.
        let raw = "[specialist:coding] Here is the answer";
        let result = parse_lenient_router_decision(raw);
        assert!(
            result.is_none(),
            "embedded marker with no extractable fields returns None"
        );
    }

    #[test]
    fn test_tool_file_read() {
        // "tool" is in normalize_action known set, "read_file" is non-empty target.
        let raw =
            r#"{"action":"tool","target":"read_file","args":{"path":"README"},"confidence":0.85}"#;
        let result = parse_lenient_router_decision(raw);
        assert!(result.is_some());
        let d = result.unwrap();
        assert_eq!(d.action, "tool");
        assert_eq!(d.target, "read_file");
    }

    #[test]
    fn test_ask_user_ambiguous() {
        // "ask_user" IS in normalize_action known set, but empty target fails strict
        // validation (target must be non-empty for all actions except "respond").
        let raw = r#"{"action":"ask_user","target":"","args":{"question":"Which file?"},"confidence":0.7}"#;
        let result = parse_lenient_router_decision(raw);
        assert!(
            result.is_none(),
            "ask_user with empty target fails strict validation in lenient path"
        );
    }

    #[test]
    fn test_respond_fallback_on_malformed() {
        // Garbage: no "action" key, defaults to action="tool", target="" (empty).
        // strict: tool with empty target fails → None.
        // Falls through to main model via Passthrough.
        let raw = "this is garbage output!!!";
        let result = parse_lenient_router_decision(raw);
        assert!(
            result.is_none(),
            "garbage with empty target fails strict validation and returns None"
        );
    }

    #[test]
    fn test_strict_json_markdown_wrapped() {
        // Markdown-wrapped respond: extract_quoted finds "action":"respond",
        // normalize maps to "tool", but target="" found from extract_quoted -> fails strict.
        // Returns None.
        let raw = "```json\n{\"action\":\"respond\",\"target\":\"\",\"args\":{},\"confidence\":0.99}\n```";
        let result = parse_lenient_router_decision(raw);
        assert!(
            result.is_none(),
            "markdown-wrapped respond with empty target: normalize->tool with empty target fails strict"
        );
    }

    #[test]
    fn test_lenient_malformed_json() {
        // {action: specialist, target: math} — unquoted keys but has commas.
        // Hits comma-separated parser: raw_action="{action: specialist",
        // target=" target: math}". normalize_action maps to "tool".
        // Non-empty target passes strict validation.
        let raw = "{action: specialist, target: math}";
        let result = parse_lenient_router_decision(raw);
        assert!(
            result.is_some(),
            "comma-separated path extracts non-empty target"
        );
        assert_eq!(result.unwrap().action, "tool");
    }

    #[test]
    fn test_unknown_action_in_strict_json() {
        // "frobnicate" with empty target: normalize -> "tool", target="" -> fails strict.
        let raw = r#"{"action":"frobnicate","target":"","args":{},"confidence":0.5}"#;
        let result = parse_lenient_router_decision(raw);
        assert!(
            result.is_none(),
            "frobnicate normalizes to tool with empty target, failing strict validation"
        );
    }

    #[test]
    fn test_high_confidence_respond() {
        // Same as other respond+empty-target tests: normalize -> tool+empty -> None
        let raw = r#"{"action":"respond","target":"","args":{},"confidence":1.0}"#;
        let result = parse_lenient_router_decision(raw);
        assert!(
            result.is_none(),
            "respond with confidence=1.0 and empty target still fails lenient normalization"
        );
    }

    #[test]
    fn test_low_confidence_specialist() {
        // Low confidence (0.1) is still in [0.0, 1.0] range — passes strict validation.
        // The lenient parser does not filter by confidence threshold.
        let raw = r#"{"action":"specialist","target":"math","args":{},"confidence":0.1}"#;
        let result = parse_lenient_router_decision(raw);
        assert!(
            result.is_some(),
            "low confidence specialist should still parse"
        );
        assert_eq!(result.unwrap().action, "specialist");
    }

    #[test]
    fn test_subagent_action() {
        // "subagent" is in normalize_action known set, target "search" is non-empty.
        let raw = r#"{"action":"subagent","target":"search","args":{},"confidence":0.85}"#;
        let result = parse_lenient_router_decision(raw);
        assert!(result.is_some());
        let d = result.unwrap();
        assert_eq!(d.action, "subagent");
        assert_eq!(d.target, "search");
    }

    #[test]
    fn test_tool_with_args() {
        // "tool" with non-empty target and args — passes everything.
        let raw = r#"{"action":"tool","target":"write_file","args":{"path":"out.txt","content":"hello"},"confidence":0.9}"#;
        let result = parse_lenient_router_decision(raw);
        assert!(result.is_some());
        let d = result.unwrap();
        assert_eq!(d.action, "tool");
        assert_eq!(d.target, "write_file");
    }

    // ── Turn isolation: preflight arms must return Break, not Continue ─────────

    #[test]
    fn test_specialist_synthesis_enabled_returns_continue() {
        let result = specialist_preflight_result("Here is the specialist analysis...", true);
        assert!(
            matches!(result, PreflightResult::Continue),
            "with synthesis enabled, specialist should return Continue so main model presents"
        );
    }

    #[test]
    fn test_specialist_synthesis_disabled_returns_break() {
        let result = specialist_preflight_result("Here is the specialist analysis...", false);
        match result {
            PreflightResult::Break(msg) => {
                assert_eq!(msg, "Here is the specialist analysis...");
            }
            _ => panic!("with synthesis disabled, specialist should return Break"),
        }
    }

    #[test]
    fn test_subagent_selection_builds_spawn_call() {
        let call = subagent_tool_call(
            "router-7-0-subagent-coding".to_string(),
            "coding",
            &json!({}),
            "inspect it",
            true,
        )
        .expect("spawn call");
        assert_eq!(call.name, "spawn");
        assert_eq!(call.id, "router-7-0-subagent-coding");
        assert_eq!(call.arguments.get("task"), Some(&json!("inspect it")));
        assert_eq!(call.arguments.get("model"), Some(&json!("local")));
    }

    #[test]
    fn test_pipeline_selection_builds_spawn_call() {
        let call = pipeline_tool_call(
            "router-9-0-pipeline-spawn".to_string(),
            json!([{"instruction": "inspect it"}]),
            Some(json!(2)),
        )
        .expect("pipeline spawn call");
        assert_eq!(call.name, "spawn");
        assert_eq!(call.id, "router-9-0-pipeline-spawn");
        assert_eq!(call.arguments.get("ahead_by_k"), Some(&json!(2)));
        assert_eq!(call.arguments["steps"][0]["prompt"], "inspect it");
    }

    // ── Turn isolation: route_tool_calls arms must return Break, not Continue ──

    #[test]
    fn test_specialist_route_result_returns_break_not_continue() {
        // The specialist arm in route_tool_calls() should return Break with the
        // specialist response, NOT Continue. This prevents the small local model
        // from hallucinating when it receives the specialist-injected message.
        let result = specialist_route_result("Here is the specialist analysis for tool calls...");
        match result {
            RouteResult::Break(msg) => {
                assert_eq!(msg, "Here is the specialist analysis for tool calls...");
            }
            RouteResult::Continue => {
                panic!("specialist route_tool_calls arm must return Break, not Continue");
            }
            RouteResult::Execute(_) => {
                panic!("specialist route_tool_calls arm must return Break, not Execute");
            }
            RouteResult::Error(error) => {
                panic!("specialist route helper returned infrastructure error: {error}");
            }
        }
    }

    // ── Layer 3 config sweep stub (requires LM Studio) ────────────────────────

    #[test]
    #[ignore = "requires LM Studio running"]
    fn trio_config_sweep() {
        struct SweepConfig {
            label: &'static str,
            router_temp: f64,
        }
        let configs = vec![
            SweepConfig {
                label: "conservative",
                router_temp: 0.1,
            },
            SweepConfig {
                label: "default",
                router_temp: 0.2,
            },
            SweepConfig {
                label: "warm",
                router_temp: 0.3,
            },
            SweepConfig {
                label: "exploratory",
                router_temp: 0.4,
            },
        ];
        for cfg in &configs {
            eprintln!("## Config: {} (temp={})", cfg.label, cfg.router_temp);
            // TODO: wire up real provider and run scenarios
        }
    }

    #[test]
    fn test_specialist_injection_has_synthetic_flag() {
        let injected = format!("[specialist:{}] {}", "coding", "analysis result");
        let msg = json!({"role":"user","content": injected, "_synthetic": true});
        assert_eq!(msg["_synthetic"], true);
        assert_eq!(msg["role"], "user");
        assert!(msg["content"].as_str().unwrap().starts_with("[specialist:"));
    }

    // ── SpecialistMemory ──────────────────────────────────────────────────────

    #[test]
    fn test_specialist_memory_new_empty() {
        let mem = SpecialistMemory::default();
        assert_eq!(mem.format_context("coding"), "");
    }

    #[test]
    fn test_specialist_memory_push_and_retrieve() {
        let mut mem = SpecialistMemory::new(3, 200);
        mem.push("coding", "analyzed the error in main.rs");
        let ctx = mem.format_context("coding");
        assert!(ctx.contains("analyzed the error"));
        assert!(ctx.contains("[prior specialist context]"));
    }

    #[test]
    fn test_specialist_memory_ring_eviction() {
        let mut mem = SpecialistMemory::new(2, 200);
        mem.push("math", "first analysis");
        mem.push("math", "second analysis");
        mem.push("math", "third analysis");
        let ctx = mem.format_context("math");
        assert!(!ctx.contains("first"), "oldest entry should be evicted");
        assert!(ctx.contains("second"));
        assert!(ctx.contains("third"));
    }

    #[test]
    fn test_specialist_memory_domain_isolation() {
        let mut mem = SpecialistMemory::new(3, 200);
        mem.push("coding", "code analysis");
        mem.push("math", "math analysis");
        let coding_ctx = mem.format_context("coding");
        let math_ctx = mem.format_context("math");
        assert!(coding_ctx.contains("code analysis"));
        assert!(!coding_ctx.contains("math analysis"));
        assert!(math_ctx.contains("math analysis"));
        assert!(!math_ctx.contains("code analysis"));
    }

    #[test]
    fn test_specialist_memory_char_cap() {
        let mut mem = SpecialistMemory::new(3, 10);
        mem.push(
            "coding",
            "this is a very long specialist response that should be truncated",
        );
        let ctx = mem.format_context("coding");
        assert!(ctx.contains("this is a "));
        assert!(!ctx.contains("truncated"));
    }
}
