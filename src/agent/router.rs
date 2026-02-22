//! Router decision parsing and dispatch functions.
//!
//! Extracted from `agent_loop.rs` to isolate routing logic into a focused module.

use std::collections::HashMap;
use std::sync::atomic::Ordering;

use serde_json::{json, Value};
use tracing::{debug, info, instrument, warn};

use crate::agent::agent_core::SwappableCore;
use crate::agent::agent_loop::TurnContext;
use crate::agent::context::ContextBuilder;
use crate::agent::policy;
use crate::agent::role_policy;
use crate::agent::router_fallback;
use crate::agent::tool_guard::ToolGuard;
use crate::agent::toolplan::{self, ToolPlanAction};
use crate::agent::tools::registry::ToolRegistry;
use crate::providers::base::{LLMProvider, ToolCallRequest};

/// Build a compact conversation tail from the message history for the router.
///
/// Extracts the last `max_pairs` user/assistant exchanges (skipping system,
/// tool_call, and tool-result messages). Each message is truncated to
/// `max_msg_chars`. The total output is capped at `max_chars`.
/// When LCM compaction is active, `messages` already contains summaries in
/// place of old messages, so this naturally includes compressed context.
/// Search backwards through recent messages for a scratch pad summary.
/// Returns the summary text if found, otherwise None.
pub(crate) fn find_scratch_pad_summary_in_messages(messages: &[Value]) -> Option<String> {
    for msg in messages.iter().rev().take(10) {
        let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("");
        let content = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
        if content.is_empty() {
            continue;
        }
        // Standalone summary from tool_engine.rs line 327: "[tool runner summary] ..."
        if role == "user" && content.starts_with("[tool runner summary] ") {
            if let Some(rest) = content.strip_prefix("[tool runner summary] ") {
                let s = rest.trim().to_string();
                if !s.is_empty() {
                    return Some(s);
                }
            }
        }
        // Inline summary from tool_engine.rs line 267: "[Tool analysis summary]\n..."
        if role == "tool" && content.starts_with("[Tool analysis summary]\n") {
            if let Some(rest) = content.strip_prefix("[Tool analysis summary]\n") {
                let summary = rest
                    .split("\n\n[Full output:")
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

pub(crate) fn build_conversation_tail(messages: &[Value], max_pairs: usize, max_msg_chars: usize, max_chars: usize) -> String {
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
    let target = extract_quoted(&tail, "target")
        .or_else(|| extract_quoted(raw, "target"))
        .unwrap_or_else(|| "clarify".to_string());
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

#[instrument(name = "request_strict_router_decision", skip(provider, router_pack, tool_names), fields(
    model,
    no_think,
    parse_strategy = tracing::field::Empty,
))]
pub(crate) async fn request_strict_router_decision(
    provider: &dyn LLMProvider,
    model: &str,
    router_pack: &str,
    no_think: bool,
    temperature: f64,
    top_p: f64,
    tool_names: &str,
) -> Result<role_policy::RouterDecision, String> {
    info!(role = "router", model = %model, "router_decision_start");
    fn parse_router_directive_pack(pack: &str) -> Option<role_policy::RouterDecision> {
        let action = {
            let pat = "action=";
            let start = pack.find(pat)? + pat.len();
            let tail = &pack[start..];
            let end = tail
                .find(|c: char| c.is_whitespace() || c == ',')
                .unwrap_or(tail.len());
            tail[..end].trim().to_string()
        };
        let target = {
            let pat = "target=";
            let start = pack.find(pat)? + pat.len();
            let tail = &pack[start..];
            let end = tail
                .find(|c: char| c.is_whitespace() || c == ',')
                .unwrap_or(tail.len());
            tail[..end].trim().to_string()
        };
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

    // Build user content with optional /no_think prefix for Nemotron-Orchestrator-8B
    let user_content = if no_think {
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
                    "action": {"type": "string", "enum": ["respond","tool","specialist","ask_user"]},
                    "target": {"type": "string"},
                    "args": {"type": "object"},
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
    if let Ok(tool_resp) = provider
        .chat(&tool_messages, Some(&tool_defs), Some(model), 256, temperature, None, Some(top_p))
        .await
    {
        if let Some(tc) = tool_resp.tool_calls.first() {
            if tc.name == "route_decision" {
                let args_obj = tc
                    .arguments
                    .iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect::<serde_json::Map<String, Value>>();
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
         Schema: {{\"action\":\"respond|tool|specialist|ask_user\",\"target\":\"string\",\"args\":{{}},\"confidence\":0.0-1.0}}\n\
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

    let router_resp = provider
        .chat(&router_messages, None, Some(model), 256, temperature, None, Some(top_p))
        .await
        .map_err(|e| format!("strict router call failed: {}", e))?;
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
                || decision.target.len() > 96;
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
            Err(format!(
                "strict router parse failed: {}. raw={}",
                e,
                raw.chars().take(220).collect::<String>()
            ))
        }
    }
}

/// Dispatch a router decision to the specialist lane.
///
/// Shared by both preflight and post-tool router paths. Returns:
/// - `Ok(text)` to inject as a user message and continue
/// - `Err(msg)` on fatal error (break with msg)
#[instrument(
    name = "dispatch_specialist",
    skip(core, counters, router_args, user_content, context_summary, tool_list, messages),
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
) -> Result<String, String> {
    let start = std::time::Instant::now();
    info!(role = "specialist", target = %target, "dispatch_specialist_start");
    let (specialist_provider, specialist_model) = match (
        core.specialist_provider.as_ref(),
        core.specialist_model.as_deref(),
    ) {
        (Some(p), Some(m)) => (p.clone(), m.to_string()),
        _ => {
            tracing::Span::current().record("outcome", "error");
            return Err(
                "Specialist lane requested by router but no specialist server is configured."
                    .to_string(),
            );
        }
    };

    let cb_key = format!("specialist:{}", specialist_model);
    if !counters.trio_circuit_breaker.lock().unwrap().is_available(&cb_key) {
        tracing::Span::current().record("outcome", "error");
        return Err(format!("circuit breaker open for {}", cb_key));
    }
    let specialist_state = format!(
        "Target: {}\nRouter args: {}\nUser intent: {}",
        target, router_args, context_summary
    );
    let conv_tail = build_conversation_tail(messages, 5, 200, 800);
    let specialist_pack = if core.tool_delegation_config.role_scoped_context_packs {
        role_policy::build_context_pack(
            role_policy::Role::Specialist,
            user_content,
            &conv_tail,
            &specialist_state,
            tool_list,
            3000,
        )
    } else {
        specialist_state
    };
    let specialist_messages = vec![
        json!({"role":"system","content":"ROLE=SPECIALIST\nReturn concise actionable output only. No markdown unless requested."}),
        json!({"role":"user","content": specialist_pack}),
    ];
    match specialist_provider
        .chat(
            &specialist_messages,
            None,
            Some(&specialist_model),
            core.tool_delegation_config.max_tokens,
            0.3,
            None,
            None,
        )
        .await
    {
        Ok(sp_resp) => {
            counters.trio_circuit_breaker.lock().unwrap().record_success(&cb_key);
            let raw_text = sp_resp
                .content
                .unwrap_or_else(|| "Specialist returned no content.".to_string());
            let text = crate::agent::sanitize::sanitize_reasoning_output(&raw_text);
            tracing::Span::current().record("outcome", "ok");
            tracing::Span::current().record("elapsed_ms", start.elapsed().as_millis() as u64);
            Ok(format!("[specialist:{}] {}", target, text))
        }
        Err(e) => {
            counters.trio_circuit_breaker.lock().unwrap().record_failure(&cb_key);
            tracing::Span::current().record("outcome", "error");
            Err(format!("Specialist lane failed: {}", e))
        }
    }
}

/// Dispatch a router decision to spawn a subagent.
///
/// Returns the formatted result string to inject as a user message.
pub(crate) async fn dispatch_subagent(
    tools: &ToolRegistry,
    target: &str,
    router_args: &Value,
    user_content: &str,
    strict_local_only: bool,
    tool_guard: &mut ToolGuard,
) -> Result<String, String> {
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
    if let Err(e) = tool_guard.allow("spawn", &params) {
        warn!("{}", e);
        return Ok(format!("[tool-guard] {}", e));
    }
    let spawn_result = tools.execute("spawn", params).await;
    Ok(format!("[router:subagent] {}", spawn_result.data))
}

// ---------------------------------------------------------------------------
// Router preflight and post-tool routing (extracted from run_agent_loop)
// ---------------------------------------------------------------------------

/// Result of the router preflight check.
pub(crate) enum PreflightResult {
    /// Router injected a message — continue the main loop.
    Continue,
    /// Router decided to break — set final_content.
    Break(String),
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
        user_msg = %ctx.user_content.chars().take(80).collect::<String>(),
        routing_decision = tracing::field::Empty,
    )
)]
pub(crate) async fn router_preflight(
    ctx: &mut TurnContext,
    health_registry: Option<&crate::heartbeat::health::HealthRegistry>,
) -> PreflightResult {
    if !(ctx.core.is_local
        && ctx.core.tool_delegation_config.strict_no_tools_main
        && ctx.core.tool_delegation_config.strict_router_schema
        && !ctx.flow.router_preflight_done)
    {
        if ctx.core.is_local && !ctx.flow.router_preflight_done {
            debug!(
                strict_no_tools_main = ctx.core.tool_delegation_config.strict_no_tools_main,
                strict_router_schema = ctx.core.tool_delegation_config.strict_router_schema,
                "router_preflight_skipped"
            );
        }
        tracing::Span::current().record("routing_decision", "passthrough");
        return PreflightResult::Passthrough;
    }

    info!("router_preflight_firing");
    ctx.counters.trio_metrics.router_preflight_fired.store(true, Ordering::Relaxed);
    ctx.flow.router_preflight_done = true;
    let (router_provider, router_model) =
        match (ctx.core.router_provider.as_ref(), ctx.core.router_model.as_deref()) {
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
            ctx.counters.set_trio_state(crate::agent::agent_core::TrioState::Degraded);
            warn!("[router] trio_router probe degraded — falling through to main model");
            tracing::Span::current().record("routing_decision", "passthrough_degraded");
            return PreflightResult::Passthrough;
        }
    }

    // Circuit breaker gate: skip if router has too many recent failures.
    let cb_key = format!("router:{}", router_model);
    if !ctx.counters.trio_circuit_breaker.lock().unwrap().is_available(&cb_key) {
        ctx.counters.set_trio_state(crate::agent::agent_core::TrioState::Degraded);
        warn!("[router] circuit breaker open for {cb_key} — falling through to main model");
        tracing::Span::current().record("routing_decision", "passthrough_circuit_open");
        return PreflightResult::Passthrough;
    }
    let tool_list = if ctx.core.is_local {
        ctx.tools
            .get_local_definitions(&ctx.messages, &ctx.used_tools)
            .iter()
            .filter_map(|d| {
                d.pointer("/function/name")
                    .and_then(|v| v.as_str())
                    .map(String::from)
            })
            .collect()
    } else {
        ctx.tools.tool_names()
    };
    let conv_tail = build_conversation_tail(&ctx.messages, 5, 200, 800);
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

    let decision = match request_strict_router_decision(
        router_provider.as_ref(),
        &router_model,
        &router_pack,
        ctx.core.router_no_think,
        ctx.core.router_temperature,
        ctx.core.router_top_p,
        &tool_list.join(", "),
    )
    .await
    {
        Ok(d) => {
            ctx.counters.trio_circuit_breaker.lock().unwrap().record_success(&cb_key);
            d
        }
        Err(e) => {
            warn!("[router] router call failed: {} — recording failure and falling through to main model", e);
            ctx.counters.trio_circuit_breaker.lock().unwrap().record_failure(&cb_key);
            tracing::Span::current().record("routing_decision", "passthrough_router_error");
            return PreflightResult::Passthrough;
        }
    };

    info!(
        role = "router",
        model = %router_model,
        action = %decision.action,
        target = %decision.target,
        "router_decision"
    );
    *ctx.counters.trio_metrics.router_action.lock().unwrap() = Some(decision.action.clone());

    match decision.action.as_str() {
        "ask_user" => {
            tracing::Span::current().record("routing_decision", "ask_user");
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
            match dispatch_specialist(
                &ctx.core,
                &ctx.counters,
                &decision.target,
                &decision.args,
                &ctx.user_content,
                &ctx.user_content,
                &tool_list,
                &ctx.messages,
            )
            .await
            {
                Ok(text) => {
                    ctx.counters.trio_metrics.specialist_dispatched.store(true, Ordering::Relaxed);
                    ctx.messages
                        .push(json!({"role":"user","content": text}));
                    PreflightResult::Continue
                }
                Err(e) => PreflightResult::Break(e),
            }
        }
        "subagent" => {
            tracing::Span::current().record("routing_decision", "subagent");
            match dispatch_subagent(
                &ctx.tools,
                &decision.target,
                &decision.args,
                &ctx.user_content,
                ctx.strict_local_only,
                &mut ctx.flow.tool_guard,
            )
            .await
            {
                Ok(text) => {
                    ctx.messages
                        .push(json!({"role":"user","content": text}));
                    PreflightResult::Continue
                }
                Err(e) => PreflightResult::Break(e),
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
            if let Err(e) = ctx.flow.tool_guard.allow(&decision.target, &params_map) {
                warn!("{}", e);
                ctx.messages.push(json!({
                    "role":"user",
                    "content": format!("[tool-guard] {}", e),
                }));
                return PreflightResult::Continue;
            }
            let tr = ctx.tools.execute(&decision.target, params_map).await;
            ctx.messages.push(json!({
                "role":"user",
                "content": format!("[router:tool:{}] {}", decision.target, tr.data),
            }));
            *ctx.counters.trio_metrics.tool_dispatched.lock().unwrap() = Some(decision.target.clone());
            ctx.used_tools.insert(decision.target);
            PreflightResult::Continue
        }
        "respond" => {
            tracing::Span::current().record("routing_decision", "respond");
            debug!("Router: respond — forwarding to main model");
            PreflightResult::Passthrough
        }
        _ => {
            tracing::Span::current().record("routing_decision", "unknown_passthrough");
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
    /// Filtered tool calls ready for execution.
    Execute(Vec<ToolCallRequest>),
}

/// Route tool calls through the strict router / toolplan / fallback pipeline.
///
/// Takes the raw tool calls from the LLM response, applies router filtering,
/// tool guard, and policy, then returns a control flow signal.
pub(crate) async fn route_tool_calls(
    ctx: &mut TurnContext,
    response_content: Option<&str>,
    mut routed_tool_calls: Vec<ToolCallRequest>,
) -> RouteResult {
    let mut router_decision: Option<role_policy::RouterDecision> = None;
    let mut router_decision_valid = false;
    let mut selected_plan: Option<toolplan::ToolPlan> = None;
    let available_tools = ctx.tools.tool_names();

    if ctx.core.tool_delegation_config.strict_router_schema {
        let task_state = format!(
            "Main content: {}\nCandidate tool calls: {}",
            response_content.unwrap_or("(empty)"),
            routed_tool_calls
                .iter()
                .map(|tc| tc.name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );
        let conv_tail = build_conversation_tail(&ctx.messages, 5, 200, 800);
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

        if let (Some(router_provider), Some(router_model)) =
            (ctx.core.router_provider.as_ref(), ctx.core.router_model.as_deref())
        {
            match request_strict_router_decision(
                router_provider.as_ref(),
                router_model,
                &router_pack,
                ctx.core.router_no_think,
                ctx.core.router_temperature,
                ctx.core.router_top_p,
                &available_tools.join(", "),
            )
            .await
            {
                Ok(decision) => {
                    router_decision_valid = true;
                    router_decision = Some(decision);
                }
                Err(e) => {
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
                if ctx.core.tool_delegation_config.strict_toolplan_validation
                    && ctx.core.tool_delegation_config.deterministic_router_fallback
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

    if ctx.core.is_local
        && role_policy::should_block_main_tool_calls(
            ctx.core.tool_delegation_config.strict_no_tools_main,
            true,
        )
        && !router_decision_valid
    {
        if ctx.core.tool_delegation_config.deterministic_router_fallback {
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
        if ctx.core.tool_delegation_config.strict_toolplan_validation {
            if let Err(e) = plan.validate() {
                if ctx.core.tool_delegation_config.deterministic_router_fallback {
                    warn!(
                        "tool plan validation failed ({}), using deterministic fallback",
                        e
                    );
                } else {
                    return RouteResult::Break(format!(
                        "Router produced invalid tool plan: {}",
                        e
                    ));
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
                match dispatch_specialist(
                    &ctx.core,
                    &ctx.counters,
                    &plan.target,
                    &plan.args,
                    &ctx.user_content,
                    context_summary,
                    &ctx.tools.tool_names(),
                    &ctx.messages,
                )
                .await
                {
                    Ok(text) => {
                        ctx.messages
                            .push(json!({"role":"user","content": text}));
                        return RouteResult::Continue;
                    }
                    Err(e) => return RouteResult::Break(e),
                }
            }
            ToolPlanAction::Subagent => {
                match dispatch_subagent(
                    &ctx.tools,
                    &plan.target,
                    &plan.args,
                    &ctx.user_content,
                    ctx.strict_local_only,
                    &mut ctx.flow.tool_guard,
                )
                .await
                {
                    Ok(text) => {
                        ctx.messages
                            .push(json!({"role":"user","content": text}));
                        return RouteResult::Continue;
                    }
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
                            id: format!("planned-{}-{}", ctx.turn_count, plan.target),
                            name: plan.target,
                            arguments: args,
                        }];
                    }
                }
            }
        }
    }

    // Tool guard filtering: split calls into allowed, blocked-with-cache, blocked-without-cache.
    let original_count = routed_tool_calls.len();
    let mut allowed_calls: Vec<ToolCallRequest> = Vec::new();
    let mut blocked_with_result: Vec<(ToolCallRequest, String)> = Vec::new();
    let mut blocked_no_result = 0usize;

    for tc in routed_tool_calls {
        match ctx.flow.tool_guard.allow(&tc.name, &tc.arguments) {
            Ok(()) => allowed_calls.push(tc),
            Err(e) => {
                warn!("{}", e);
                let key = ToolGuard::key(&tc.name, &tc.arguments);
                if let Some(cached) = ctx.flow.tool_guard.get_cached_result(&key) {
                    blocked_with_result.push((tc, cached.to_string()));
                } else {
                    blocked_no_result += 1;
                }
            }
        }
    }

    let total_blocked = blocked_with_result.len() + blocked_no_result;

    // Replay cached results for blocked calls that have them.
    if !blocked_with_result.is_empty() {
        let tc_json: Vec<Value> = blocked_with_result
            .iter()
            .map(|(tc, _)| tc.to_openai_json())
            .collect();
        ContextBuilder::add_assistant_message(
            &mut ctx.messages,
            response_content,
            Some(&tc_json),
        );
        for (tc, cached_result) in &blocked_with_result {
            ContextBuilder::add_tool_result(
                &mut ctx.messages,
                &tc.id,
                &tc.name,
                cached_result,
            );
        }
    }

    if allowed_calls.is_empty() {
        // All tool calls were blocked.
        if total_blocked > 0 && total_blocked == original_count {
            if !blocked_with_result.is_empty() && blocked_no_result == 0 {
                // All blocked calls had cached results — replay succeeded.
                ctx.flow.consecutive_all_blocked = 0;
                return RouteResult::Continue;
            }
            ctx.flow.consecutive_all_blocked += 1;
            // Circuit breaker: after 2 consecutive all-blocked rounds, force a
            // text response. The LLM is stuck in a loop requesting the same tools.
            if ctx.flow.consecutive_all_blocked >= 2 {
                warn!(
                    rounds = ctx.flow.consecutive_all_blocked,
                    "tool_loop_circuit_breaker: model stuck requesting blocked tools, forcing response"
                );
                return RouteResult::Break(
                    "I've been trying to use the same tools repeatedly. Let me answer with what I have so far."
                        .to_string(),
                );
            }
        }
        if let Some(text) = response_content.filter(|s| !s.trim().is_empty()) {
            return RouteResult::Break(text.to_string());
        }
        return RouteResult::Continue;
    }

    // Reset the consecutive blocked counter when tool calls succeed.
    ctx.flow.consecutive_all_blocked = 0;
    RouteResult::Execute(allowed_calls)
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

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
        assert!(result.is_some(), "markdown-wrapped JSON should be extracted");
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
    // - Lenient path falls back target="" to "clarify" by default (via unwrap_or_else)
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
        // extract_quoted falls back to default "tool". extract_quoted("target") = None -> "clarify".
        // normalize_action("tool","clarify",{}) = "tool".
        // strict: tool+clarify+0.5 => Ok.
        // So this returns Some(action="tool", target="clarify").
        let raw = "[specialist:coding] Here is the answer";
        let result = parse_lenient_router_decision(raw);
        assert!(result.is_some(), "embedded marker falls through to lenient fallback: Some");
        let d = result.unwrap();
        assert_eq!(d.action, "tool", "lenient fallback normalizes to 'tool'");
        assert_eq!(d.target, "clarify", "lenient fallback target defaults to 'clarify'");
    }

    #[test]
    fn test_parse_lenient_malformed_garbage_returns_some_tool() {
        // Garbage text: no "action" key found, defaults to action="tool", target="clarify".
        // strict: tool+clarify+0.5 => Ok. Returns Some(action="tool").
        let raw = "this is not valid at all!!!";
        let result = parse_lenient_router_decision(raw);
        assert!(
            result.is_some(),
            "garbage with no action/target falls back to tool+clarify which passes strict"
        );
        let d = result.unwrap();
        assert_eq!(d.action, "tool");
        assert_eq!(d.target, "clarify");
    }

    #[test]
    fn test_parse_lenient_empty_string_returns_some_tool() {
        // Same fallback: no keys found -> action="tool", target="clarify".
        let result = parse_lenient_router_decision("");
        assert!(
            result.is_some(),
            "empty string falls back to tool+clarify which passes strict validation"
        );
        let d = result.unwrap();
        assert_eq!(d.action, "tool");
        assert_eq!(d.target, "clarify");
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
        let msgs = vec![
            json!({"role": "user", "content": "[tool runner summary] found 5 files"}),
        ];
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
        assert!(result.is_none(), "summary outside 10-message window should return None");
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
        assert!(result.contains("Hello there"), "should contain first user message");
        assert!(result.contains("How are you"), "should contain second user message");
        assert!(result.contains("Hi back"), "should contain assistant response");
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
        assert!(result.contains('…'), "truncated message should end with ellipsis");
        // The user message should be truncated to ~100 chars + ellipsis
        let user_line = result.lines().find(|l| l.starts_with("User:")).unwrap();
        assert!(user_line.len() <= 120, "truncated line should not be too long");
    }

    #[test]
    fn test_build_tail_system_and_tool_roles_skipped() {
        let msgs = vec![
            json!({"role": "system", "content": "You are a helpful assistant."}),
            json!({"role": "tool", "content": "Tool result data"}),
            json!({"role": "tool_call", "content": "function call"}),
        ];
        let result = build_conversation_tail(&msgs, 5, 1000, 10_000);
        assert!(result.is_empty(), "system and tool messages should be skipped");
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
        assert!(result.contains("Third question"), "last pair user message should be present");
        assert!(result.contains("Third answer"), "last pair assistant message should be present");
        assert!(!result.contains("First question"), "earlier pairs should be excluded");
        assert!(!result.contains("Second question"), "earlier pairs should be excluded");
    }

    #[test]
    fn test_build_tail_empty_messages() {
        let result = build_conversation_tail(&[], 5, 1000, 10_000);
        assert!(result.is_empty(), "empty messages should produce empty tail");
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
        // "[specialist:coding] ..." — no "action" key, extract_quoted defaults to "tool".
        // target defaults to "clarify". Returns Some(action="tool", target="clarify").
        let raw = "[specialist:coding] Here is the answer";
        let result = parse_lenient_router_decision(raw);
        assert!(result.is_some(), "embedded marker falls back to tool+clarify");
        let d = result.unwrap();
        assert_eq!(d.action, "tool");
    }

    #[test]
    fn test_tool_file_read() {
        // "tool" is in normalize_action known set, "read_file" is non-empty target.
        let raw = r#"{"action":"tool","target":"read_file","args":{"path":"README"},"confidence":0.85}"#;
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
        // Garbage: no "action" key, defaults to action="tool", target="clarify".
        // strict: tool+clarify+0.5 => Ok. Returns Some(action="tool").
        let raw = "this is garbage output!!!";
        let result = parse_lenient_router_decision(raw);
        assert!(
            result.is_some(),
            "garbage falls back to tool+clarify which passes strict validation"
        );
        assert_eq!(result.unwrap().action, "tool");
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
        // {action: specialist, target: math} — unquoted keys.
        // extract_quoted won't find "action" (no quotes) -> default "tool".
        // extract_quoted won't find "target" -> default "clarify".
        // Returns Some(action="tool", target="clarify").
        let raw = "{action: specialist, target: math}";
        let result = parse_lenient_router_decision(raw);
        assert!(result.is_some(), "unquoted-key JSON defaults to tool+clarify");
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
        assert!(result.is_some(), "low confidence specialist should still parse");
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

    // ── Layer 3 config sweep stub (requires LM Studio) ────────────────────────

    #[test]
    #[ignore = "requires LM Studio running"]
    fn trio_config_sweep() {
        struct SweepConfig {
            label: &'static str,
            router_temp: f64,
        }
        let configs = vec![
            SweepConfig { label: "conservative", router_temp: 0.1 },
            SweepConfig { label: "default", router_temp: 0.2 },
            SweepConfig { label: "warm", router_temp: 0.3 },
            SweepConfig { label: "exploratory", router_temp: 0.4 },
        ];
        for cfg in &configs {
            eprintln!("## Config: {} (temp={})", cfg.label, cfg.router_temp);
            // TODO: wire up real provider and run scenarios
        }
    }
}

