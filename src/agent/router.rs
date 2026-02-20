//! Router decision parsing and dispatch functions.
//!
//! Extracted from `agent_loop.rs` to isolate routing logic into a focused module.

use std::collections::HashMap;

use serde_json::{json, Value};
use tracing::{debug, warn};

use crate::agent::agent_core::SwappableCore;
use crate::agent::agent_loop::TurnContext;
use crate::agent::policy;
use crate::agent::role_policy;
use crate::agent::router_fallback;
use crate::agent::tool_guard::ToolGuard;
use crate::agent::toolplan::{self, ToolPlanAction};
use crate::agent::tools::registry::ToolRegistry;
use crate::providers::base::{LLMProvider, ToolCallRequest};

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

pub(crate) async fn request_strict_router_decision(
    provider: &dyn LLMProvider,
    model: &str,
    router_pack: &str,
    no_think: bool,
    temperature: f64,
) -> Result<role_policy::RouterDecision, String> {
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
    let tool_defs = vec![route_tool];
    let tool_messages = vec![
        json!({
            "role": "system",
            "content": concat!(
                "You are a routing agent. Analyze the user's request and call route_decision once.\n\n",
                "Actions:\n",
                "- respond: Greetings, chitchat, simple questions the main model can answer directly\n",
                "- tool: Use a specific tool (set target=tool_name, args=tool_parameters)\n",
                "- specialist: Delegate to specialist model for complex multi-step reasoning\n",
                "- ask_user: ONLY when the request is truly ambiguous and cannot be answered\n\n",
                "If the user is just saying hello or asking a simple question, use action=respond.\n",
                "Call route_decision exactly once. No prose.",
            )
        }),
        json!({
            "role": "user",
            "content": user_content.clone()
        }),
    ];
    if let Ok(tool_resp) = provider
        .chat(&tool_messages, Some(&tool_defs), Some(model), 256, temperature, None)
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
                        return Ok(decision);
                    }
                }
            }
        }
    }

    let router_messages = vec![
        json!({
            "role": "system",
            "content": concat!(
                "Output EXACTLY one JSON object. No markdown, no explanation, no extra text.\n",
                "Schema: {\"action\":\"respond|tool|specialist|ask_user\",\"target\":\"string\",\"args\":{},\"confidence\":0.0-1.0}\n\n",
                "Examples:\n",
                "User says hello → {\"action\":\"respond\",\"target\":\"main\",\"args\":{},\"confidence\":0.95}\n",
                "User asks to read a file → {\"action\":\"tool\",\"target\":\"read_file\",\"args\":{\"path\":\"README.md\"},\"confidence\":0.9}\n",
                "User asks a simple question → {\"action\":\"respond\",\"target\":\"main\",\"args\":{},\"confidence\":0.9}\n",
            )
        }),
        json!({
            "role": "user",
            "content": user_content
        }),
    ];

    let router_resp = provider
        .chat(&router_messages, None, Some(model), 256, temperature, None)
        .await
        .map_err(|e| format!("strict router call failed: {}", e))?;
    let raw = router_resp.content.unwrap_or_default();
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
pub(crate) async fn dispatch_specialist(
    core: &SwappableCore,
    target: &str,
    router_args: &Value,
    user_content: &str,
    context_summary: &str,
    tool_list: &[String],
) -> Result<String, String> {
    let (specialist_provider, specialist_model) = match (
        core.specialist_provider.as_ref(),
        core.specialist_model.as_deref(),
    ) {
        (Some(p), Some(m)) => (p.clone(), m.to_string()),
        _ => {
            return Err(
                "Specialist lane requested by router but no specialist server is configured."
                    .to_string(),
            );
        }
    };
    let specialist_state = format!(
        "Target: {}\nRouter args: {}\nUser intent: {}",
        target, router_args, context_summary
    );
    let specialist_pack = if core.tool_delegation_config.role_scoped_context_packs {
        role_policy::build_context_pack(
            role_policy::Role::Specialist,
            user_content,
            "(live turn; summary omitted)",
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
        )
        .await
    {
        Ok(sp_resp) => {
            let text = sp_resp
                .content
                .unwrap_or_else(|| "Specialist returned no content.".to_string());
            Ok(format!("[specialist:{}] {}", target, text))
        }
        Err(e) => Err(format!("Specialist lane failed: {}", e)),
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
pub(crate) async fn router_preflight(ctx: &mut TurnContext) -> PreflightResult {
    if !(ctx.core.is_local
        && ctx.core.tool_delegation_config.strict_no_tools_main
        && ctx.core.tool_delegation_config.strict_router_schema
        && !ctx.router_preflight_done)
    {
        return PreflightResult::Passthrough;
    }

    ctx.router_preflight_done = true;
    let (router_provider, router_model) =
        match (ctx.core.router_provider.as_ref(), ctx.core.router_model.as_deref()) {
            (Some(p), Some(m)) => (p.clone(), m.to_string()),
            _ => {
                return PreflightResult::Break(
                    "Router lane is required by policy but not configured. Start trio router server and retry.".to_string(),
                );
            }
        };
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
    let task_state = format!("Strict preflight. User message: {}", ctx.user_content);
    let router_pack = if ctx.core.tool_delegation_config.role_scoped_context_packs {
        role_policy::build_context_pack(
            role_policy::Role::Router,
            &ctx.user_content,
            "(live turn; summary omitted)",
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
    )
    .await
    {
        Ok(d) => d,
        Err(e) => {
            return PreflightResult::Break(format!("Router policy failed: {}.", e));
        }
    };

    debug!(
        "Router decision: action={}, target={}, args={}",
        decision.action,
        decision.target,
        serde_json::to_string(&decision.args).unwrap_or_default()
    );

    match decision.action.as_str() {
        "ask_user" => PreflightResult::Break(
            decision
                .args
                .get("question")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .unwrap_or_else(|| "I need clarification to continue.".to_string()),
        ),
        "specialist" => {
            match dispatch_specialist(
                &ctx.core,
                &decision.target,
                &decision.args,
                &ctx.user_content,
                &ctx.user_content,
                &tool_list,
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
        "subagent" => {
            match dispatch_subagent(
                &ctx.tools,
                &decision.target,
                &decision.args,
                &ctx.user_content,
                ctx.strict_local_only,
                &mut ctx.tool_guard,
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
            if let Err(e) = ctx.tool_guard.allow(&decision.target, &params_map) {
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
            ctx.used_tools.insert(decision.target);
            PreflightResult::Continue
        }
        "respond" => {
            debug!("Router: respond — forwarding to main model");
            PreflightResult::Passthrough
        }
        _ => {
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
        let router_pack = if ctx.core.tool_delegation_config.role_scoped_context_packs {
            role_policy::build_context_pack(
                role_policy::Role::Router,
                &ctx.user_content,
                "(live turn; summary omitted)",
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
                let context_summary = response_content.unwrap_or("(empty)");
                match dispatch_specialist(
                    &ctx.core,
                    &plan.target,
                    &plan.args,
                    &ctx.user_content,
                    context_summary,
                    &ctx.tools.tool_names(),
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
                    &mut ctx.tool_guard,
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

    // Tool guard filtering.
    let mut blocked_calls = 0usize;
    routed_tool_calls.retain(|tc| match ctx.tool_guard.allow(&tc.name, &tc.arguments) {
        Ok(()) => true,
        Err(e) => {
            blocked_calls += 1;
            warn!("{}", e);
            false
        }
    });
    if blocked_calls > 0 {
        ctx.messages.push(json!({
            "role":"user",
            "content": format!(
                "[tool-guard] blocked {} duplicate tool call(s). Continue without re-running identical calls.",
                blocked_calls
            ),
        }));
    }
    if routed_tool_calls.is_empty() {
        if let Some(text) = response_content.filter(|s| !s.trim().is_empty()) {
            return RouteResult::Break(text.to_string());
        }
        return RouteResult::Continue;
    }

    RouteResult::Execute(routed_tool_calls)
}
