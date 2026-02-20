#![allow(dead_code)]
//! Main agent loop that consumes inbound messages and produces responses.
//!
//! Ported from Python `agent/loop.py`.
//!
//! The agent loop uses a fan-out pattern for concurrent message processing:
//! messages from different sessions run in parallel (up to `max_concurrent_chats`),
//! while messages within the same session are serialized to preserve ordering.

use std::collections::HashMap;
use std::path::PathBuf;

use chrono::Utc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use serde_json::{json, Value};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use tokio::sync::{Mutex, Semaphore};
use tracing::{debug, error, info, warn};

use crate::agent::agent_profiles;
use crate::agent::audit::{AuditLog, ToolEvent};
use crate::agent::compaction::ContextCompactor;
use crate::agent::context::ContextBuilder;
use crate::agent::learning::LearningStore;
use crate::agent::pipeline;
use crate::agent::policy;
use crate::agent::reflector::Reflector;
use crate::agent::role_policy;
use crate::agent::router_fallback;
use crate::agent::subagent::SubagentManager;
use crate::agent::system_state::{self, AhaPriority, AhaSignal, SystemState};
use crate::agent::thread_repair;
use crate::agent::token_budget::TokenBudget;
use crate::agent::tool_guard::ToolGuard;
use crate::agent::toolplan::{self, ToolPlanAction};
use crate::agent::tool_runner::{self, Budget, ToolRunnerConfig};
use crate::agent::tools::registry::{ToolConfig, ToolRegistry};
use crate::agent::tools::{
    CancelCallback, CheckCallback, CheckInboxTool, CronScheduleTool, ListCallback, LoopCallback,
    MessageTool, PipelineCallback, SendCallback, SendEmailTool, SpawnCallback, SpawnTool,
    SpawnToolLite, WaitCallback,
};
use crate::agent::working_memory::WorkingMemoryStore;
use crate::bus::events::{InboundMessage, OutboundMessage};
use crate::config::schema::{
    EmailConfig, MemoryConfig, ProprioceptionConfig, ProvenanceConfig, ToolDelegationConfig,
    TrioConfig,
};
use crate::cron::service::CronService;
use crate::providers::base::{LLMProvider, StreamChunk, ToolCallRequest};
use crate::providers::openai_compat::OpenAICompatProvider;
use crate::session::manager::SessionManager;

use crate::agent::context_hygiene;
use crate::agent::validation;

// ---------------------------------------------------------------------------
// Shared core (identical across all agents, swappable on /local toggle)
// ---------------------------------------------------------------------------

/// Fields that change on `/local` and `/model` — behind `Arc<RwLock<Arc<>>>`.
///
/// When the user toggles `/local` or `/model`, a new `SwappableCore` is built
/// and swapped into the handle so every agent sees the change.
pub struct SwappableCore {
    pub provider: Arc<dyn LLMProvider>,
    pub workspace: PathBuf,
    pub model: String,
    pub max_iterations: u32,
    pub max_tokens: u32,
    pub temperature: f64,
    pub context: ContextBuilder,
    pub sessions: SessionManager,
    pub token_budget: TokenBudget,
    pub compactor: ContextCompactor,
    pub learning: LearningStore,
    pub working_memory: WorkingMemoryStore,
    pub working_memory_budget: usize,
    pub brave_api_key: Option<String>,
    pub exec_timeout: u64,
    pub restrict_to_workspace: bool,
    pub memory_enabled: bool,
    pub memory_provider: Arc<dyn LLMProvider>,
    pub memory_model: String,
    pub reflection_threshold: usize,
    pub is_local: bool,
    pub main_no_think: bool,
    pub tool_runner_provider: Option<Arc<dyn LLMProvider>>,
    pub tool_runner_model: Option<String>,
    pub router_provider: Option<Arc<dyn LLMProvider>>,
    pub router_model: Option<String>,
    pub router_no_think: bool,
    pub router_temperature: f64,
    #[allow(dead_code)]
    pub router_top_p: f64,
    pub specialist_provider: Option<Arc<dyn LLMProvider>>,
    pub specialist_model: Option<String>,
    pub tool_delegation_config: ToolDelegationConfig,
    pub provenance_config: ProvenanceConfig,
    pub max_tool_result_chars: usize,
    pub session_complete_after_secs: u64,
    pub max_message_age_turns: usize,
    pub max_history_turns: usize,
}

/// Atomic counters that survive core swaps — never behind `RwLock`.
///
/// These counters persist across `/local` and `/model` hot-swaps because
/// they live outside the swappable core. Previously they were inside
/// `SharedCore` and silently reset to zero on every swap.
pub struct RuntimeCounters {
    pub learning_turn_counter: AtomicU64,
    pub last_context_used: AtomicU64,
    pub last_context_max: AtomicU64,
    pub last_message_count: AtomicU64,
    pub last_working_memory_tokens: AtomicU64,
    pub last_tools_called: std::sync::Mutex<Vec<String>>,
    /// Tracks whether the delegation provider is alive. Set to `false` when
    /// the delegation LLM returns a hard error or times out, causing subsequent
    /// calls to fall through to inline execution. Reset to `true` on core
    /// rebuild (`rebuild_core`) and `/restart` command.
    pub delegation_healthy: AtomicBool,
    /// Counts tool calls since delegation was marked unhealthy. Used to
    /// periodically re-probe: every 10 inline calls, try delegation once
    /// more in case the server recovered.
    pub delegation_retry_counter: AtomicU64,
    /// Extended thinking budget in tokens. 0 = disabled, >0 = enabled with that budget.
    /// Toggled by `/think` or `/t`. `/think 16000` sets a specific budget.
    pub thinking_budget: AtomicU32,
    /// Remaining turns with boosted max_tokens (set by `/long`). Counts down to 0.
    pub long_mode_turns: AtomicU32,
    /// Last actual prompt tokens from LLM provider (for telemetry).
    pub last_actual_prompt_tokens: AtomicU64,
    /// Last actual completion tokens from LLM provider (for telemetry).
    pub last_actual_completion_tokens: AtomicU64,
    /// Last estimated prompt tokens (our estimate, for comparison).
    pub last_estimated_prompt_tokens: AtomicU64,
    /// When true, ThinkingDelta tokens are NOT sent to delta_tx (suppressed
    /// from TTS). Auto-set when voice mode is active; toggled by `/nothink`.
    pub suppress_thinking_in_tts: AtomicBool,
    /// Set to true while an LLM call is in flight. The health watchdog reads
    /// this to skip health checks during inference (avoiding false "unhealthy"
    /// restarts when the server is busy processing a large prompt).
    /// Wrapped in Arc so the watchdog can hold a cheap clone without needing
    /// the full RuntimeCounters.
    pub inference_active: Arc<AtomicBool>,
}

impl RuntimeCounters {
    pub fn new(max_context_tokens: usize) -> Self {
        Self {
            learning_turn_counter: AtomicU64::new(0),
            last_context_used: AtomicU64::new(0),
            last_context_max: AtomicU64::new(max_context_tokens as u64),
            last_message_count: AtomicU64::new(0),
            last_working_memory_tokens: AtomicU64::new(0),
            last_tools_called: std::sync::Mutex::new(Vec::new()),
            delegation_healthy: AtomicBool::new(true),
            delegation_retry_counter: AtomicU64::new(0),
            thinking_budget: AtomicU32::new(0),
            long_mode_turns: AtomicU32::new(0),
            last_actual_prompt_tokens: AtomicU64::new(0),
            last_actual_completion_tokens: AtomicU64::new(0),
            last_estimated_prompt_tokens: AtomicU64::new(0),
            suppress_thinking_in_tts: AtomicBool::new(false),
            inference_active: Arc::new(AtomicBool::new(false)),
        }
    }
}

/// Combined handle: cheap to clone (two pointer bumps).
///
/// `core` is swapped on `/local` and `/model`. `counters` persists forever.
#[derive(Clone)]
pub struct AgentHandle {
    core: Arc<std::sync::RwLock<Arc<SwappableCore>>>,
    pub counters: Arc<RuntimeCounters>,
}

impl AgentHandle {
    /// Create a new handle from a swappable core and runtime counters.
    pub fn new(core: SwappableCore, counters: Arc<RuntimeCounters>) -> Self {
        Self {
            core: Arc::new(std::sync::RwLock::new(Arc::new(core))),
            counters,
        }
    }

    /// Snapshot the current swappable core (cheap Arc clone under brief read lock).
    pub fn swappable(&self) -> Arc<SwappableCore> {
        self.core.read().unwrap().clone()
    }

    /// Replace the swappable core (write lock). Counters are untouched.
    pub fn swap_core(&self, new_core: SwappableCore) {
        *self.core.write().unwrap() = Arc::new(new_core);
    }
}

// Backward-compatibility alias during migration.
pub type SharedCoreHandle = AgentHandle;

/// Local chat templates often reject mid-conversation `system` messages.
/// In local mode, provenance reminders must be emitted as `user` role.
fn provenance_warning_role(is_local: bool) -> &'static str {
    if is_local {
        "user"
    } else {
        "system"
    }
}

fn is_small_local_model(model: &str) -> bool {
    let m = model.to_ascii_lowercase();
    m.contains("nanbeige")
        || m.contains("functiongemma")
        || m.contains("ministral-3")
        || m.contains("qwen3-1.7b")
        || m.contains("3b")
        || m.contains("1.7b")
}

/// Extract the first top-level JSON object from raw text.
///
/// This tolerates wrappers like markdown fences while still requiring the
/// final parsed payload to satisfy strict schema validation.
fn extract_json_object(raw: &str) -> Option<String> {
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
fn parse_lenient_router_decision(raw: &str) -> Option<role_policy::RouterDecision> {
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

async fn request_strict_router_decision(
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
/// - `Ok(Some(text))` to inject as a user message and continue
/// - `Ok(None)` to break with an error (set in `final_content`)
/// - `Err(msg)` on fatal error (break with msg)
async fn dispatch_specialist(
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
async fn dispatch_subagent(
    tools: &ToolRegistry,
    target: &str,
    router_args: &Value,
    user_content: &str,
    strict_local_only: bool,
    tool_guard: &mut crate::agent::tool_guard::ToolGuard,
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

/// Named-field input for [`build_swappable_core`].
///
/// Replaces 18 positional parameters with a single struct so callers
/// are immune to parameter-ordering bugs.
pub struct SwappableCoreConfig {
    pub provider: Arc<dyn LLMProvider>,
    pub workspace: PathBuf,
    pub model: String,
    pub max_iterations: u32,
    pub max_tokens: u32,
    pub temperature: f64,
    pub max_context_tokens: usize,
    pub brave_api_key: Option<String>,
    pub exec_timeout: u64,
    pub restrict_to_workspace: bool,
    pub memory_config: MemoryConfig,
    pub is_local: bool,
    pub compaction_provider: Option<Arc<dyn LLMProvider>>,
    pub tool_delegation: ToolDelegationConfig,
    pub provenance: ProvenanceConfig,
    pub max_tool_result_chars: usize,
    pub delegation_provider: Option<Arc<dyn LLMProvider>>,
    pub specialist_provider: Option<Arc<dyn LLMProvider>>,
    pub trio_config: TrioConfig,
}

/// Build a `SwappableCore` from the given config.
///
/// Called once at startup and again for every `/local` or `/model` toggle.
/// Resolves provider selection, memory config, tool delegation, and router setup.
pub fn build_swappable_core(cfg: SwappableCoreConfig) -> SwappableCore {
    let SwappableCoreConfig {
        provider,
        workspace,
        model,
        max_iterations,
        max_tokens,
        temperature,
        max_context_tokens,
        brave_api_key,
        exec_timeout,
        restrict_to_workspace,
        memory_config,
        is_local,
        compaction_provider,
        tool_delegation,
        provenance,
        max_tool_result_chars,
        delegation_provider,
        specialist_provider,
        trio_config,
    } = cfg;
    let router_provider = delegation_provider.clone();
    let mut context = if is_local {
        ContextBuilder::new_lite(&workspace)
    } else {
        ContextBuilder::new(&workspace)
    };
    // Scale prompt budgets proportionally to the model's context window.
    // Without this, a 1M-context model gets the same tiny fixed caps as a 16K model.
    if is_local {
        context.set_lite_mode(max_context_tokens);
    } else {
        context.scale_budgets(max_context_tokens);
    }
    context.model_name = model.clone();
    // Inject provenance verification rules when enabled.
    if provenance.enabled && provenance.system_prompt_rules {
        context.provenance_enabled = true;
    }
    // RLM lazy skills: skills loaded as summaries, fetched on demand.
    context.lazy_skills = memory_config.lazy_skills;
    // Wire subagent profiles into the system prompt so the model knows
    // what agents exist and when to delegate instead of doing everything itself.
    let profiles = agent_profiles::load_profiles(&workspace);
    context.agent_profiles = agent_profiles::profiles_summary(&profiles);
    let sessions = SessionManager::new(&workspace);

    // When local, use dedicated compaction provider if available, else main provider.
    let (memory_provider, memory_model): (Arc<dyn LLMProvider>, String) = if is_local {
        let m = if memory_config.model.is_empty() {
            model.clone()
        } else {
            memory_config.model.clone()
        };
        if let Some(cp) = compaction_provider {
            (cp, m)
        } else {
            (provider.clone(), m)
        }
    } else if let Some(ref mem_provider_cfg) = memory_config.provider {
        let p: Arc<dyn LLMProvider> = Arc::new(OpenAICompatProvider::new(
            &mem_provider_cfg.api_key,
            mem_provider_cfg
                .api_base
                .as_deref()
                .or(Some("http://localhost:8080/v1")),
            None,
        ));
        let m = if memory_config.model.is_empty() {
            model.clone()
        } else {
            memory_config.model.clone()
        };
        (p, m)
    } else {
        let m = if memory_config.model.is_empty() {
            model.clone()
        } else {
            memory_config.model.clone()
        };
        (provider.clone(), m)
    };

    let token_budget = TokenBudget::new(max_context_tokens, max_tokens as usize);
    let compactor = ContextCompactor::new(
        memory_provider.clone(),
        memory_model.clone(),
        max_context_tokens,
    )
    .with_thresholds(
        memory_config.compaction_threshold_percent,
        memory_config.compaction_threshold_tokens,
    );
    let learning = LearningStore::new(&workspace);
    let working_memory = WorkingMemoryStore::new(&workspace);

    // Build tool runner provider if delegation is enabled.
    let (tool_runner_provider, tool_runner_model) = if tool_delegation.enabled {
        let is_auto_local = delegation_provider.is_some();
        let tr_provider: Arc<dyn LLMProvider> = if let Some(dp) = delegation_provider {
            dp // Auto-spawned local delegation server
        } else if let Some(ref tr_cfg) = tool_delegation.provider {
            Arc::new(OpenAICompatProvider::new(
                &tr_cfg.api_key,
                tr_cfg
                    .api_base
                    .as_deref()
                    .or(Some("http://localhost:8080/v1")),
                None,
            ))
        } else {
            provider.clone() // Fallback to main
        };
        // Pick the delegation model. When config is empty, fall back to the
        // delegation provider's own default (e.g. local server's model) rather
        // than the main model — the main model may be a cloud name like
        // "anthropic/claude-opus-4-5" that the local server doesn't understand.
        let tr_model = if !tool_delegation.model.is_empty() {
            tool_delegation.model.clone()
        } else if is_auto_local || model.starts_with("claude-max") || model.contains('/') {
            // Auto-spawned local delegation, or cloud model name — use provider default.
            tr_provider.get_default_model().to_string()
        } else {
            model.clone()
        };
        (Some(tr_provider), Some(tr_model))
    } else {
        (None, None)
    };

    let specialist_model = specialist_provider
        .as_ref()
        .map(|provider| provider.get_default_model().to_string());
    let router_model = router_provider
        .as_ref()
        .map(|provider| provider.get_default_model().to_string());

    SwappableCore {
        provider,
        workspace,
        model,
        max_iterations,
        max_tokens,
        temperature,
        context,
        sessions,
        token_budget,
        compactor,
        learning,
        working_memory,
        // Scale working memory like other budgets. If the user left it at
        // the default (600), apply proportional scaling; otherwise respect their override.
        working_memory_budget: if memory_config.working_memory_budget == 600 {
            (max_context_tokens * 15 / 1000).clamp(300, 15_000) // 1.5%
        } else {
            memory_config.working_memory_budget
        },
        brave_api_key,
        exec_timeout,
        restrict_to_workspace,
        memory_enabled: memory_config.enabled,
        memory_provider,
        memory_model,
        reflection_threshold: memory_config.reflection_threshold,
        is_local,
        main_no_think: trio_config.main_no_think,
        tool_runner_provider,
        tool_runner_model,
        router_provider,
        router_model,
        router_no_think: trio_config.router_no_think,
        router_temperature: trio_config.router_temperature,
        router_top_p: trio_config.router_top_p,
        specialist_provider,
        specialist_model,
        tool_delegation_config: tool_delegation,
        provenance_config: provenance,
        max_tool_result_chars,
        session_complete_after_secs: memory_config.session_complete_after_secs,
        max_message_age_turns: memory_config.max_message_age_turns,
        max_history_turns: memory_config.max_history_turns,
    }
}

// ---------------------------------------------------------------------------
// History limit scaling
// ---------------------------------------------------------------------------

/// Scale history message count with context window size.
///
/// Small models (16K) can't afford 100 messages of history — that alone
/// can eat 40%+ of the context. Scale linearly: ~20 msgs at 16K, ~100 at
/// 128K, clamped to [6, 100].
fn history_limit(max_context_tokens: usize) -> usize {
    // Real-world average is ~150 tokens per message (user queries + assistant
    // responses). Reserve at most 30% of context for history.
    let max_history_tokens = max_context_tokens * 3 / 10;
    let limit = max_history_tokens / 150;
    limit.clamp(6, 100)
}

// ---------------------------------------------------------------------------
// Background compaction helpers
// ---------------------------------------------------------------------------

/// Pending compaction result ready to be swapped into the conversation.
struct PendingCompaction {
    result: crate::agent::compaction::CompactionResult,
    watermark: usize, // messages.len() when compaction was spawned
}

/// Swap compacted messages into the live conversation, preserving
/// messages added after the compaction snapshot was taken.
fn apply_compaction_result(messages: &mut Vec<Value>, pending: PendingCompaction) {
    let new_messages: Vec<Value> = if pending.watermark < messages.len() {
        messages[pending.watermark..].to_vec()
    } else {
        vec![]
    };
    let mut swapped = Vec::with_capacity(1 + pending.result.messages.len() + new_messages.len());
    swapped.push(messages[0].clone()); // fresh system msg
    if pending.result.messages.len() > 1 {
        swapped.extend_from_slice(&pending.result.messages[1..]); // skip stale system msg
    }
    swapped.extend(new_messages);
    *messages = swapped;
}

// ---------------------------------------------------------------------------
// Per-instance state (different per agent)
// ---------------------------------------------------------------------------

/// Per-instance state that differs between the REPL agent and gateway agents.
struct AgentLoopShared {
    core_handle: SharedCoreHandle,
    subagents: Arc<SubagentManager>,
    bus_outbound_tx: UnboundedSender<OutboundMessage>,
    #[allow(dead_code)]
    bus_inbound_tx: UnboundedSender<InboundMessage>,
    cron_service: Option<Arc<CronService>>,
    email_config: Option<EmailConfig>,
    repl_display_tx: Option<UnboundedSender<String>>,
    /// Cached memory bulletin for system prompt injection (zero-cost reads).
    bulletin_cache: Arc<arc_swap::ArcSwap<String>>,
    /// Shared system state for ensemble proprioception.
    system_state: Arc<arc_swap::ArcSwap<SystemState>>,
    /// Proprioception config (feature toggles).
    proprioception_config: ProprioceptionConfig,
    /// Receiver for priority signals from subagents (aha channel).
    aha_rx: Arc<Mutex<tokio::sync::mpsc::UnboundedReceiver<AhaSignal>>>,
    /// Sender for priority signals (given to subagent manager).
    aha_tx: tokio::sync::mpsc::UnboundedSender<AhaSignal>,
    /// Sticky per-session policy flags (e.g. local_only).
    session_policies: Arc<Mutex<HashMap<String, policy::SessionPolicy>>>,
}

/// Per-message state that flows through the three processing phases.
///
/// Owns all per-turn mutable state that previously lived as local variables
/// inside `process_message`. No lifetimes needed — values are cloned from the
/// inbound message where required.
struct TurnContext {
    // --- Config (set during prepare, immutable after) ---
    core: Arc<SwappableCore>,
    session_key: String,
    session_policy: policy::SessionPolicy,
    strict_local_only: bool,
    turn_count: u64,
    streaming: bool,
    audit: Option<AuditLog>,
    tools: ToolRegistry,
    user_content: String,
    channel: String,
    chat_id: String,
    is_voice_message: bool,
    detected_language: Option<String>,

    // --- Channels (moved into context) ---
    text_delta_tx: Option<tokio::sync::mpsc::UnboundedSender<String>>,
    tool_event_tx: Option<tokio::sync::mpsc::UnboundedSender<ToolEvent>>,
    cancellation_token: Option<tokio_util::sync::CancellationToken>,
    priority_rx: Option<tokio::sync::mpsc::UnboundedReceiver<String>>,

    // --- Conversation state ---
    messages: Vec<Value>,
    new_start: usize,

    // --- Tracking ---
    used_tools: std::collections::HashSet<String>,
    final_content: String,
    turn_tool_entries: Vec<crate::agent::audit::TurnToolEntry>,

    // --- Budget/compaction ---
    compaction_slot: Arc<tokio::sync::Mutex<Option<PendingCompaction>>>,
    compaction_in_flight: Arc<AtomicBool>,
    content_gate: crate::agent::context_gate::ContentGate,

    // --- Flow control ---
    force_response: bool,
    router_preflight_done: bool,
    tool_guard: ToolGuard,
    turns_since_compaction: u32,
    forced_finalize_attempted: bool,
    content_was_streamed: bool,
}

impl AgentLoopShared {
    /// Build a fresh [`ToolRegistry`] with context-sensitive tools (message,
    /// spawn, cron) pre-configured for a specific channel/chat_id.
    ///
    /// Takes a snapshot of `SwappableCore` so the registry is consistent for the
    /// entire message processing.
    async fn build_tools(
        &self,
        core: &SwappableCore,
        channel: &str,
        chat_id: &str,
    ) -> ToolRegistry {
        // Standard stateless tools via unified ToolConfig.
        let tool_config = ToolConfig {
            workspace: core.workspace.clone(),
            exec_timeout: core.exec_timeout,
            restrict_to_workspace: core.restrict_to_workspace,
            max_tool_result_chars: core.max_tool_result_chars,
            brave_api_key: core.brave_api_key.clone(),
            ..ToolConfig::new(&core.workspace)
        };
        let mut tools = ToolRegistry::with_standard_tools(&tool_config);

        // Message tool - context baked in.
        let outbound_tx_clone = self.bus_outbound_tx.clone();
        let send_cb: SendCallback = Arc::new(move |msg: OutboundMessage| {
            let tx = outbound_tx_clone.clone();
            Box::pin(async move {
                tx.send(msg)
                    .map_err(|e| anyhow::anyhow!("Failed to send outbound message: {}", e))
            })
        });
        let message_tool = Arc::new(MessageTool::new(Some(send_cb), channel, chat_id));
        tools.register(Box::new(MessageToolProxy(message_tool)));

        // Spawn tool - context baked in.
        let subagents_ref = self.subagents.clone();
        let session_policies_ref = self.session_policies.clone();
        let spawn_cb: SpawnCallback =
            Arc::new(move |task, label, agent, model, ch, cid, working_dir| {
                let mgr = subagents_ref.clone();
                let policies = session_policies_ref.clone();
                Box::pin(async move {
                    let key = format!("{}:{}", ch, cid);
                    let policy = {
                        let map = policies.lock().await;
                        map.get(&key).cloned().unwrap_or_default()
                    };
                    let effective_model = policy::enforce_subagent_model(&policy, model);
                    mgr.spawn(task, label, agent, effective_model, ch, cid, working_dir)
                        .await
                })
            });
        let subagents_ref2 = self.subagents.clone();
        let list_workspace = core.workspace.clone();
        let list_cb: ListCallback = Arc::new(move || {
            let mgr = subagents_ref2.clone();
            let ws = list_workspace.clone();
            Box::pin(async move {
                let running = mgr.list_running().await;
                let mut out = String::new();

                // Running subagents
                if running.is_empty() {
                    out.push_str("No subagents currently running.\n");
                } else {
                    out.push_str(&format!("{} subagent(s) running:\n", running.len()));
                    for info in &running {
                        let elapsed = info.started_at.elapsed().as_secs();
                        out.push_str(&format!(
                            "  • {} (id: {}) — running for {}s\n",
                            info.label, info.task_id, elapsed
                        ));
                    }
                }

                // Recently completed (from events.jsonl)
                let recent = SubagentManager::read_recent_completed(&ws, 10);
                if !recent.is_empty() {
                    out.push_str(&format!("\nRecently completed ({}):\n", recent.len()));
                    for entry in &recent {
                        out.push_str(entry);
                        out.push('\n');
                    }
                }

                out
            })
        });
        let subagents_ref3 = self.subagents.clone();
        let cancel_cb: CancelCallback = Arc::new(move |task_id: String| {
            let mgr = subagents_ref3.clone();
            Box::pin(async move {
                if mgr.cancel(&task_id).await {
                    format!("Subagent '{}' cancelled.", task_id)
                } else {
                    format!("No running subagent found matching '{}'.", task_id)
                }
            })
        });
        let subagents_ref4 = self.subagents.clone();
        let wait_cb: WaitCallback = Arc::new(move |task_id: String, timeout_secs: u64| {
            let mgr = subagents_ref4.clone();
            Box::pin(async move {
                let timeout = std::time::Duration::from_secs(timeout_secs);
                mgr.wait_for(&task_id, timeout).await
            })
        });
        let check_workspace = core.workspace.clone();
        let check_cb: CheckCallback = Arc::new(move |task_id: String| {
            let ws = check_workspace.clone();
            Box::pin(async move {
                match SubagentManager::read_event_result(&ws, &task_id) {
                    Some(result) => result,
                    None => format!("No completed result found for task_id '{}'.", task_id),
                }
            })
        });
        // Pipeline callback: parse steps JSON, build PipelineConfig, run pipeline.
        // Uses the delegation provider/model (cheap) for pipeline LLM calls.
        let pipeline_provider = core
            .tool_runner_provider
            .clone()
            .unwrap_or_else(|| core.provider.clone());
        let pipeline_model = core
            .tool_runner_model
            .clone()
            .unwrap_or_else(|| core.model.clone());
        let pipeline_workspace = core.workspace.clone();
        let pipeline_cb: PipelineCallback =
            Arc::new(move |steps_json: String, ahead_by_k: usize| {
                let provider = pipeline_provider.clone();
                let model = pipeline_model.clone();
                let workspace = pipeline_workspace.clone();
                Box::pin(async move {
                    // Parse steps from JSON.
                    let steps: Vec<serde_json::Value> = match serde_json::from_str(&steps_json) {
                        Ok(s) => s,
                        Err(e) => return format!("Error parsing pipeline steps: {}", e),
                    };
                    let pipeline_steps: Vec<pipeline::PipelineStep> = steps
                        .iter()
                        .enumerate()
                        .map(|(i, s)| pipeline::PipelineStep {
                            index: i,
                            prompt: s["prompt"].as_str().unwrap_or("").to_string(),
                            expected: s["expected"].as_str().map(|s| s.to_string()),
                            tools: s["tools"].as_array().map(|arr| {
                                arr.iter()
                                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                                    .collect()
                            }),
                            max_iterations: s["max_iterations"].as_u64().map(|n| n as u32),
                        })
                        .collect();
                    if pipeline_steps.is_empty() {
                        return "Error: no valid pipeline steps provided.".to_string();
                    }
                    let config = pipeline::PipelineConfig {
                        pipeline_id: format!(
                            "pipe-{}",
                            chrono::Utc::now().timestamp_millis() % 100_000_000
                        ),
                        steps: pipeline_steps,
                        ahead_by_k,
                        max_voters: if ahead_by_k > 0 {
                            ahead_by_k * 2 + 1
                        } else {
                            1
                        },
                        model: model.clone(),
                    };
                    let result =
                        pipeline::run_pipeline(&config, provider.as_ref(), &workspace).await;
                    // Format result for the agent.
                    let mut output = format!(
                        "Pipeline '{}' completed: {}/{} steps\n",
                        result.pipeline_id, result.steps_completed, result.steps_total
                    );
                    for sr in &result.results {
                        let correct_str = match sr.correct {
                            Some(true) => " ✓",
                            Some(false) => " ✗",
                            None => "",
                        };
                        output.push_str(&format!(
                            "  Step {}: {}{} ({}ms, {} voters)\n",
                            sr.index,
                            sr.answer.chars().take(200).collect::<String>(),
                            correct_str,
                            sr.duration_ms,
                            sr.voters_used
                        ));
                    }
                    output.push_str(&format!("Total time: {}ms", result.total_duration_ms));
                    output
                })
            });

        // Loop callback: run an autonomous refinement loop via SubagentManager.
        let subagents_ref5 = self.subagents.clone();
        let loop_cb: LoopCallback = Arc::new(
            move |task: String,
                  max_rounds: u32,
                  tools_filter: Option<Vec<String>>,
                  stop_condition: Option<String>,
                  model: Option<String>,
                  working_dir: Option<String>| {
                let mgr = subagents_ref5.clone();
                Box::pin(async move {
                    mgr.run_loop(
                        task,
                        max_rounds,
                        tools_filter,
                        stop_condition,
                        model,
                        working_dir,
                    )
                    .await
                })
            },
        );

        let spawn_tool = Arc::new(SpawnTool::new());
        // Set callbacks and context before registering so they're ready for use.
        spawn_tool.set_callback(spawn_cb).await;
        spawn_tool.set_list_callback(list_cb).await;
        spawn_tool.set_cancel_callback(cancel_cb).await;
        spawn_tool.set_wait_callback(wait_cb).await;
        spawn_tool.set_check_callback(check_cb).await;
        spawn_tool.set_pipeline_callback(pipeline_cb).await;
        spawn_tool.set_loop_callback(loop_cb).await;
        spawn_tool.set_context(channel, chat_id).await;
        // Local models get the lite schema (~200 tokens) instead of the full
        // schema (~1,100 tokens) which would consume 55% of a 4K context.
        if core.is_local {
            tools.register(Box::new(SpawnToolLite(spawn_tool)));
        } else {
            tools.register(Box::new(SpawnToolProxy(spawn_tool)));
        }

        // Cron tool (optional) - context baked in.
        if let Some(ref svc) = self.cron_service {
            let ct = Arc::new(CronScheduleTool::new(svc.clone()));
            ct.set_context(channel, chat_id).await;
            tools.register(Box::new(CronToolProxy(ct)));
        }

        // Email tools (optional) - available when email is configured.
        if let Some(ref email_cfg) = self.email_config {
            tools.register(Box::new(CheckInboxTool::new(email_cfg.clone())));
            tools.register(Box::new(SendEmailTool::new(email_cfg.clone())));
        }

        tools
    }

    /// Process an inbound message through the agent loop.
    ///
    /// When `text_delta_tx` is `Some`, text deltas are streamed to the sender
    /// as they arrive (used by CLI/voice). When `None`, a blocking LLM call
    /// is used (gateway mode).
    ///
    /// This method takes `&self` and is safe to call from multiple concurrent
    /// tasks. Per-message tool instances eliminate shared-context races.
    async fn process_message(
        &self,
        msg: &InboundMessage,
        text_delta_tx: Option<tokio::sync::mpsc::UnboundedSender<String>>,
        tool_event_tx: Option<tokio::sync::mpsc::UnboundedSender<ToolEvent>>,
        cancellation_token: Option<tokio_util::sync::CancellationToken>,
        priority_rx: Option<tokio::sync::mpsc::UnboundedReceiver<String>>,
    ) -> Option<OutboundMessage> {
        let mut ctx = self.prepare_context(msg, text_delta_tx, tool_event_tx, cancellation_token, priority_rx).await;
        self.run_agent_loop(&mut ctx).await;
        self.finalize_response(ctx).await
    }

    /// Phase 1: Build the [`TurnContext`] from the inbound message.
    ///
    /// Snapshots the swappable core, extracts session info, builds tools,
    /// loads history, constructs the message array, and initialises all
    /// per-turn tracking state.
    async fn prepare_context(
        &self,
        msg: &InboundMessage,
        text_delta_tx: Option<tokio::sync::mpsc::UnboundedSender<String>>,
        tool_event_tx: Option<tokio::sync::mpsc::UnboundedSender<ToolEvent>>,
        cancellation_token: Option<tokio_util::sync::CancellationToken>,
        priority_rx: Option<tokio::sync::mpsc::UnboundedReceiver<String>>,
    ) -> TurnContext {
        let streaming = text_delta_tx.is_some();

        // Snapshot core — instant Arc clone under brief read lock.
        let core = self.core_handle.swappable();
        let counters = &self.core_handle.counters;
        let turn_count = counters
            .learning_turn_counter
            .fetch_add(1, Ordering::Relaxed)
            + 1;
        if turn_count % 50 == 0 {
            core.learning.prune();
        }

        let session_key = msg
            .metadata
            .get("session_key")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("{}:{}", msg.channel, msg.chat_id));

        let session_policy = {
            let mut map = self.session_policies.lock().await;
            let entry = map.entry(session_key.clone()).or_default();
            if core.tool_delegation_config.strict_local_only {
                entry.local_only = true;
            }
            policy::update_from_user_text(entry, &msg.content);
            entry.clone()
        };
        let strict_local_only =
            core.tool_delegation_config.strict_local_only || session_policy.local_only;

        debug!(
            "Processing message{} from {} on {}: {}",
            if streaming { " (streaming)" } else { "" },
            msg.sender_id,
            msg.channel,
            &msg.content[..msg.content.len().min(80)]
        );

        // Create audit log if provenance is enabled.
        let audit = if core.provenance_config.enabled && core.provenance_config.audit_log {
            Some(AuditLog::new(&core.workspace, &session_key))
        } else {
            None
        };

        // Build per-message tools with context baked in.
        let tools = self.build_tools(&core, &msg.channel, &msg.chat_id).await;

        // Get session history. Track count so we know where new messages start.
        let history = core
            .sessions
            .get_history(
                &session_key,
                history_limit(core.token_budget.max_context()),
                core.max_history_turns,
            )
            .await;
        // Track where new (unsaved) messages start. Updated after compaction
        // swaps to avoid re-persisting already-saved messages.
        let new_start = 1 + history.len();

        // Extract media paths.
        let media_paths: Vec<String> = msg
            .metadata
            .get("media")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();

        // Build messages.
        let is_voice_message = msg
            .metadata
            .get("voice_message")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let detected_language = msg
            .metadata
            .get("detected_language")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let user_content = if core.main_no_think {
            format!(" /no_think\n{}", msg.content)
        } else {
            msg.content.clone()
        };
        let mut messages = core.context.build_messages(
            &history,
            &user_content,
            None,
            if media_paths.is_empty() {
                None
            } else {
                Some(&media_paths)
            },
            Some(&msg.channel),
            Some(&msg.chat_id),
            is_voice_message,
            detected_language.as_deref(),
        );

        // Inject per-session working memory into the system message.
        if core.memory_enabled {
            let mut wm = core
                .working_memory
                .get_context(&session_key, core.working_memory_budget);
            // Append learning context (tool patterns) if available.
            let learning_ctx = core.learning.get_learning_context();
            if !learning_ctx.is_empty() {
                wm.push_str("\n\n## Tool Patterns\n\n");
                wm.push_str(&learning_ctx);
            }
            if !wm.is_empty() {
                if let Some(system_content) = messages
                    .first()
                    .and_then(|m| m["content"].as_str())
                    .map(|s| s.to_string())
                {
                    let enriched = format!(
                        "{}\n\n---\n\n# Working Memory (Current Session)\n\n{}",
                        system_content, wm
                    );
                    messages[0]["content"] = Value::String(enriched);
                }
            }
        }

        // Auto-inject background task status into system prompt so the agent
        // is naturally aware of running/completed subagents without explicit tool calls.
        {
            let running = self.subagents.list_running().await;
            let recent =
                crate::agent::subagent::SubagentManager::read_recent_completed(&core.workspace, 5);
            let status = crate::agent::subagent::format_status_block(&running, &recent);
            if !status.is_empty() {
                if let Some(sys) = messages
                    .first()
                    .and_then(|m| m["content"].as_str())
                    .map(|s| s.to_string())
                {
                    messages[0]["content"] = Value::String(format!("{}{}", sys, status));
                }
            }
        }

        // Inject memory bulletin if available (zero-cost Arc load).
        {
            let bulletin = self.bulletin_cache.load_full();
            if !bulletin.is_empty() {
                if let Some(sys) = messages
                    .first()
                    .and_then(|m| m["content"].as_str())
                    .map(|s| s.to_string())
                {
                    messages[0]["content"] =
                        Value::String(format!("{}\n\n## Memory Briefing\n\n{}", sys, &*bulletin));
                }
            }
        }

        // Tag the current user message (last in the array) with turn number
        // for age-based eviction in trim_to_fit.
        if let Some(last) = messages.last_mut() {
            last["_turn"] = json!(turn_count);
        }

        // Background compaction state.
        let compaction_slot: Arc<tokio::sync::Mutex<Option<PendingCompaction>>> =
            Arc::new(tokio::sync::Mutex::new(None));
        let compaction_in_flight = Arc::new(AtomicBool::new(false));

        // Context gate: budget-aware content sizing for this turn.
        let cache_dir = crate::utils::helpers::get_data_path()
            .join("cache")
            .join("tool_outputs");
        let mut content_gate = crate::agent::context_gate::ContentGate::new(
            core.token_budget.max_context(),
            0.20,
            cache_dir,
        );
        // Pre-consume the tokens already used by system prompt + history.
        let initial_tokens = TokenBudget::estimate_tokens(&messages);
        content_gate.budget.consume(initial_tokens);

        let tool_guard = ToolGuard::new(core.tool_delegation_config.max_same_tool_call_per_turn);

        TurnContext {
            core,
            session_key,
            session_policy,
            strict_local_only,
            turn_count,
            streaming,
            audit,
            tools,
            user_content,
            channel: msg.channel.clone(),
            chat_id: msg.chat_id.clone(),
            is_voice_message,
            detected_language,
            text_delta_tx,
            tool_event_tx,
            cancellation_token,
            priority_rx,
            messages,
            new_start,
            used_tools: std::collections::HashSet::new(),
            final_content: String::new(),
            turn_tool_entries: Vec::new(),
            compaction_slot,
            compaction_in_flight,
            content_gate,
            force_response: false,
            router_preflight_done: false,
            tool_guard,
            turns_since_compaction: 0,
            forced_finalize_attempted: false,
            content_was_streamed: false,
        }
    }

    /// Phase 2: Run the main agent loop (LLM calls + tool execution).
    ///
    /// Iterates up to `max_iterations`, calling the LLM, handling tool calls
    /// or router decisions, and accumulating results in `ctx`.
    async fn run_agent_loop(&self, ctx: &mut TurnContext) {
        let counters = &self.core_handle.counters;

        for iteration in 0..ctx.core.max_iterations {
            debug!(
                "Agent iteration{} {}/{}",
                if ctx.streaming { " (streaming)" } else { "" },
                iteration + 1,
                ctx.core.max_iterations
            );

            // --- Context Hygiene: clean up conversation history ---
            context_hygiene::hygiene_pipeline(&mut ctx.messages);

            // --- Proprioception: update SystemState ---
            if self.proprioception_config.enabled {
                let tools_list: Vec<String> = if let Ok(guard) = counters.last_tools_called.lock() {
                    guard.clone()
                } else {
                    Vec::new()
                };
                let tool_refs: Vec<&str> = tools_list.iter().map(|s| s.as_str()).collect();
                let phase = system_state::infer_phase(&tool_refs);
                let active_subs = self.subagents.list_running().await.len().min(255) as u8;
                let state = SystemState::snapshot(
                    phase,
                    counters.last_context_used.load(Ordering::Relaxed),
                    counters.last_context_max.load(Ordering::Relaxed),
                    ctx.turn_count,
                    ctx.messages.len() as u64,
                    ctx.turns_since_compaction,
                    counters.delegation_healthy.load(Ordering::Relaxed),
                    0,    // recent_tool_failures — not tracked yet
                    true, // last_tool_ok
                    active_subs,
                    0, // pending_aha_signals filled below
                );
                self.system_state.store(Arc::new(state));
            }

            // --- Aha Channel: poll priority signals from subagents ---
            if self.proprioception_config.enabled && self.proprioception_config.aha_channel {
                if let Ok(mut rx) = self.aha_rx.try_lock() {
                    while let Ok(signal) = rx.try_recv() {
                        match signal.priority {
                            AhaPriority::Critical => {
                                ctx.messages.push(json!({
                                    "role": "user",
                                    "content": format!(
                                        "[ALERT from subagent {}] {}",
                                        signal.agent_id, signal.message
                                    )
                                }));
                            }
                            AhaPriority::High => {
                                ctx.messages.push(json!({
                                    "role": "user",
                                    "content": format!(
                                        "[Signal from subagent {}] {}",
                                        signal.agent_id, signal.message
                                    )
                                }));
                            }
                            AhaPriority::Normal => {
                                // Normal signals are informational — logged only.
                                debug!(
                                    "Aha signal (normal) from {}: {}",
                                    signal.agent_id, signal.message
                                );
                            }
                        }
                    }
                }
            }

            // --- Heartbeat: inject grounding message ---
            if self.proprioception_config.enabled {
                let state = self.system_state.load_full();
                if system_state::should_ground(
                    iteration,
                    self.proprioception_config.grounding_interval,
                    state.context_pressure,
                ) {
                    let grounding = system_state::format_grounding(&state);
                    ctx.messages.push(json!({
                        "role": "user",
                        "content": grounding
                    }));
                }
            }

            ctx.turns_since_compaction += 1;

            // Check if background compaction finished — swap in compacted messages.
            if let Ok(mut guard) = ctx.compaction_slot.try_lock() {
                if let Some(pending) = guard.take() {
                    debug!(
                        "Compaction swap: {} msgs -> {} compacted + {} new",
                        pending.watermark,
                        pending.result.messages.len(),
                        ctx.messages.len().saturating_sub(pending.watermark)
                    );
                    apply_compaction_result(&mut ctx.messages, pending);
                    // After compaction, all messages in the array are "new" from
                    // the perspective of persistence (the session file was rebuilt).
                    ctx.new_start = ctx.messages.len();
                    ctx.turns_since_compaction = 0;
                }
            }

            // Response boundary: suppress exec/write_file tools to force text output.
            let boundary_active = ctx.force_response
                && ctx.core.provenance_config.enabled
                && ctx.core.provenance_config.response_boundary;
            if boundary_active {
                // Use "user" role, not "system". The Anthropic OpenAI-compat
                // endpoint strips mid-conversation system messages, which would
                // leave the conversation ending with an assistant message and
                // trigger a "does not support assistant message prefill" error.
                let remaining = ctx.core.max_iterations.saturating_sub(iteration as u32 + 1);
                let budget_note = if remaining <= 5 {
                    format!(
                        " [Budget: {}/{} iterations remaining — wrap up soon]",
                        remaining, ctx.core.max_iterations
                    )
                } else {
                    String::new()
                };
                ctx.messages.push(json!({
                    "role": "user",
                    "content": format!(
                        "[system] You just executed a tool that modifies files or runs commands. \
                         Report the result to the user before making additional tool calls.{budget_note}"
                    )
                }));
                ctx.force_response = false;
            }

            // Filter tool definitions to relevant tools.
            // Local models get a minimal set to conserve context tokens.
            let current_phase = self.system_state.load_full().task_phase;
            let mut tool_defs = if ctx.core.is_local {
                ctx.tools.get_local_definitions(&ctx.messages, &ctx.used_tools)
            } else if self.proprioception_config.enabled
                && self.proprioception_config.dynamic_tool_scoping
            {
                ctx.tools.get_scoped_definitions(&current_phase, &ctx.messages, &ctx.used_tools)
            } else {
                ctx.tools.get_relevant_definitions(&ctx.messages, &ctx.used_tools)
            };
            if ctx.core.is_local && ctx.core.tool_delegation_config.strict_no_tools_main {
                // Hard separation (local trio only): main model is conversation/orchestration only.
                // Cloud providers handle tools natively and must never have them stripped.
                tool_defs.clear();
            }
            if boundary_active {
                tool_defs.retain(|def| {
                    let name = def
                        .pointer("/function/name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    name != "exec" && name != "write_file"
                });
            }
            let tool_defs_opt: Option<&[Value]> = if tool_defs.is_empty() {
                None
            } else {
                Some(&tool_defs)
            };

            // Trim messages to fit context budget.
            let tool_def_tokens =
                TokenBudget::estimate_tool_def_tokens(tool_defs_opt.unwrap_or(&[]));
            ctx.messages = ctx.core.token_budget.trim_to_fit_with_age(
                &ctx.messages,
                tool_def_tokens,
                ctx.turn_count,
                ctx.core.max_message_age_turns,
            );

            // Spawn background compaction when threshold exceeded.
            if !ctx.compaction_in_flight.load(Ordering::Relaxed)
                && ctx.core
                    .compactor
                    .needs_compaction(&ctx.messages, &ctx.core.token_budget, tool_def_tokens)
            {
                let slot = ctx.compaction_slot.clone();
                let in_flight = ctx.compaction_in_flight.clone();
                let bg_messages = ctx.messages.clone();
                let bg_core = ctx.core.clone();
                let bg_session_key = ctx.session_key.clone();
                let watermark = ctx.messages.len();
                in_flight.store(true, Ordering::SeqCst);

                let bg_proprio = self.proprioception_config.clone();
                tokio::spawn(async move {
                    let result = if bg_proprio.enabled && bg_proprio.gradient_memory {
                        bg_core
                            .compactor
                            .compact_gradient(
                                &bg_messages,
                                &bg_core.token_budget,
                                0,
                                bg_proprio.raw_window,
                                bg_proprio.light_window,
                            )
                            .await
                    } else if bg_proprio.enabled && bg_proprio.audience_aware_compaction {
                        let reader =
                            crate::agent::compaction::ReaderProfile::from_model(&bg_core.model);
                        bg_core
                            .compactor
                            .compact_for_reader(&bg_messages, &bg_core.token_budget, 0, &reader)
                            .await
                    } else {
                        bg_core
                            .compactor
                            .compact(&bg_messages, &bg_core.token_budget, 0)
                            .await
                    };
                    if bg_core.memory_enabled {
                        if let Some(ref summary) = result.observation {
                            bg_core
                                .working_memory
                                .update_from_compaction(&bg_session_key, summary);
                        }
                    }
                    if result.messages.len() < bg_messages.len() {
                        *slot.lock().await = Some(PendingCompaction { result, watermark });
                    }
                    in_flight.store(false, Ordering::SeqCst);
                });
            }

            // Repair protocol violations before calling the LLM.
            if ctx.core.is_local {
                thread_repair::repair_for_local(&mut ctx.messages);
            } else {
                thread_repair::repair_messages(&mut ctx.messages);
            }

            // Pre-flight context size check: emergency trim if we're about to
            // exceed the model's context window. The 95% threshold leaves room
            // for the response tokens.
            let estimated = TokenBudget::estimate_tokens(&ctx.messages);
            let max_ctx = ctx.core.token_budget.max_context();
            if max_ctx > 0 && estimated > (max_ctx as f64 * 0.95) as usize {
                warn!(
                    "Pre-flight check: {}tok > 95% of {}tok — emergency trim",
                    estimated, max_ctx
                );
                // tool_def_tokens=0 is conservative (trims more aggressively).
                ctx.messages = ctx.core.token_budget.trim_to_fit(&ctx.messages, 0);
                // Re-run repair after trim in case truncation broke protocol.
                if ctx.core.is_local {
                    thread_repair::repair_for_local(&mut ctx.messages);
                } else {
                    thread_repair::repair_messages(&mut ctx.messages);
                }
            }

            // Router-first preflight for strict trio mode.
            // Only applies in local mode — trio routing is a local-model orchestration
            // pattern and must not block cloud providers (which handle tools natively).
            if ctx.core.is_local
                && ctx.core.tool_delegation_config.strict_no_tools_main
                && ctx.core.tool_delegation_config.strict_router_schema
                && !ctx.router_preflight_done
            {
                ctx.router_preflight_done = true;
                let (router_provider, router_model) =
                    match (ctx.core.router_provider.as_ref(), ctx.core.router_model.as_deref()) {
                        (Some(p), Some(m)) => (p.clone(), m.to_string()),
                        _ => {
                            ctx.final_content = "Router lane is required by policy but not configured. Start trio router server and retry.".to_string();
                            break;
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
                        ctx.final_content = format!("Router policy failed: {}.", e);
                        break;
                    }
                };

                debug!(
                    "Router decision: action={}, target={}, args={}",
                    decision.action,
                    decision.target,
                    serde_json::to_string(&decision.args).unwrap_or_default()
                );

                match decision.action.as_str() {
                    "ask_user" => {
                        ctx.final_content = decision
                            .args
                            .get("question")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string())
                            .unwrap_or_else(|| "I need clarification to continue.".to_string());
                        break;
                    }
                    "specialist" => {
                        match dispatch_specialist(
                            &ctx.core, &decision.target, &decision.args,
                            &ctx.user_content, &ctx.user_content, &tool_list,
                        ).await {
                            Ok(text) => {
                                ctx.messages.push(json!({"role":"user","content": text}));
                                continue;
                            }
                            Err(e) => { ctx.final_content = e; break; }
                        }
                    }
                    "subagent" => {
                        match dispatch_subagent(
                            &ctx.tools, &decision.target, &decision.args,
                            &ctx.user_content, ctx.strict_local_only, &mut ctx.tool_guard,
                        ).await {
                            Ok(text) => {
                                ctx.messages.push(json!({"role":"user","content": text}));
                                continue;
                            }
                            Err(e) => { ctx.final_content = e; break; }
                        }
                    }
                    "tool" => {
                        if decision.target.trim().is_empty() {
                            ctx.final_content = "Router selected tool action but target is empty.".to_string();
                            break;
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
                            continue;
                        }
                        let tr = ctx.tools.execute(&decision.target, params_map).await;
                        ctx.messages.push(json!({
                            "role":"user",
                            "content": format!("[router:tool:{}] {}", decision.target, tr.data),
                        }));
                        ctx.used_tools.insert(decision.target);
                        continue;
                    }
                    "respond" => {
                        // Router says main model can handle this directly — fall through.
                        debug!("Router: respond — forwarding to main model");
                    }
                    _ => {
                        debug!("Router: unrecognized action '{}' — forwarding to main model", decision.action);
                    }
                }
            }

            // Adaptive max_tokens: size the response budget to the task.
            let effective_max_tokens = {
                let base = ctx.core.max_tokens;
                // Check for /long override (temporary boost).
                let long_override = counters.long_mode_turns.load(Ordering::Relaxed);
                if long_override > 0 {
                    counters.long_mode_turns.fetch_sub(1, Ordering::Relaxed);
                    base.max(8192)
                } else {
                    // Detect long-form triggers in the user message.
                    let user_text = ctx.messages
                        .last()
                        .and_then(|m| m.get("content"))
                        .and_then(|c| c.as_str())
                        .unwrap_or("");
                    let lower = user_text.to_lowercase();
                    let is_long_form = lower.contains("explain in detail")
                        || lower.contains("write a ")
                        || lower.contains("create a script")
                        || lower.contains("write code")
                        || lower.contains("implement ")
                        || lower.contains("full example")
                        || lower.starts_with("write ")
                        || user_text.len() > 500;
                    // Count recent tool calls: if tool-heavy, use smaller budget.
                    let recent_tool_calls = ctx.messages
                        .iter()
                        .rev()
                        .take(6)
                        .filter(|m| m.get("role").and_then(|r| r.as_str()) == Some("tool"))
                        .count();
                    if is_long_form {
                        base.max(4096)
                    } else if recent_tool_calls > 3 {
                        base.min(1024).max(512)
                    } else {
                        base
                    }
                }
            };

            // Call LLM — streaming or blocking depending on mode.
            let thinking_budget = {
                let stored = counters.thinking_budget.load(Ordering::Relaxed);
                if stored > 0 {
                    // Small local models can burn the whole completion budget in reasoning.
                    // Hard-cap explicit thinking to keep them action-oriented.
                    if ctx.core.is_local && is_small_local_model(&ctx.core.model) {
                        Some(stored.min(256))
                    } else {
                        Some(stored)
                    }
                } else {
                    None
                }
            };
            // Signal watchdog: LLM inference is active — skip health checks.
            counters.inference_active.store(true, Ordering::Relaxed);

            let mut response = if let Some(ref delta_tx) = ctx.text_delta_tx {
                // Streaming path: forward text deltas as they arrive.
                let mut stream = match ctx.core
                    .provider
                    .chat_stream(
                        &ctx.messages,
                        tool_defs_opt,
                        Some(&ctx.core.model),
                        effective_max_tokens,
                        ctx.core.temperature,
                        thinking_budget,
                    )
                    .await
                {
                    Ok(s) => s,
                    Err(e) => {
                        counters.inference_active.store(false, Ordering::Relaxed);
                        error!("LLM streaming call failed: {}", e);
                        ctx.final_content = format!("I encountered an error: {}", e);
                        break;
                    }
                };

                let mut streamed_response = None;
                let mut in_thinking = false;
                let suppress_thinking = counters.suppress_thinking_in_tts.load(Ordering::Relaxed);
                loop {
                    tokio::select! {
                        biased;
                        _ = async {
                            if let Some(ref token) = ctx.cancellation_token {
                                token.cancelled().await;
                            } else {
                                std::future::pending::<()>().await;
                            }
                        } => {
                            // Cancelled — drop stream to signal provider task.
                            drop(stream);
                            break;
                        }
                        chunk = stream.rx.recv() => {
                            match chunk {
                                Some(StreamChunk::ThinkingDelta(delta)) => {
                                    if suppress_thinking {
                                        // Skip thinking tokens entirely (voice mode / /nothink)
                                        continue;
                                    }
                                    // Render thinking tokens as dimmed text
                                    if !in_thinking {
                                        in_thinking = true;
                                        let _ = delta_tx.send("\x1b[90m\u{1f9e0} \x1b[2m".to_string());
                                    }
                                    let _ = delta_tx.send(delta);
                                }
                                Some(StreamChunk::TextDelta(delta)) => {
                                    if in_thinking {
                                        in_thinking = false;
                                        let _ = delta_tx.send("\x1b[0m\n\n".to_string());
                                    }
                                    ctx.content_was_streamed = true;
                                    let _ = delta_tx.send(delta);
                                }
                                Some(StreamChunk::Done(resp)) => {
                                    if in_thinking {
                                        let _ = delta_tx.send("\x1b[0m\n\n".to_string());
                                    }
                                    streamed_response = Some(resp);
                                    break;
                                }
                                None => break,
                            }
                        }
                    }
                }

                match streamed_response {
                    Some(r) => r,
                    None => {
                        counters.inference_active.store(false, Ordering::Relaxed);
                        // Stream ended without Done — either cancelled or genuine error.
                        if ctx.cancellation_token
                            .as_ref()
                            .map_or(false, |t| t.is_cancelled())
                        {
                            // Cancelled mid-stream — exit cleanly.
                            break;
                        }
                        error!("LLM stream ended without Done");
                        ctx.final_content = "I encountered a streaming error.".to_string();
                        break;
                    }
                }
            } else {
                // Blocking path: single request/response.
                match ctx.core
                    .provider
                    .chat(
                        &ctx.messages,
                        tool_defs_opt,
                        Some(&ctx.core.model),
                        effective_max_tokens,
                        ctx.core.temperature,
                        thinking_budget,
                    )
                    .await
                {
                    Ok(r) => r,
                    Err(e) => {
                        counters.inference_active.store(false, Ordering::Relaxed);
                        error!("LLM call failed: {}", e);
                        ctx.final_content = format!("I encountered an error: {}", e);
                        break;
                    }
                }
            };

            // Inference complete — allow watchdog health checks again.
            counters.inference_active.store(false, Ordering::Relaxed);

            // --- Response Validation: detect hallucinated tool calls ---
            let content_str = response.content.as_deref().unwrap_or("");
            let tool_calls_as_maps: Vec<HashMap<String, Value>> = response
                .tool_calls
                .iter()
                .map(|tc| {
                    let mut map = HashMap::new();
                    map.insert("id".to_string(), Value::String(tc.id.clone()));
                    map.insert("name".to_string(), Value::String(tc.name.clone()));
                    map.insert(
                        "arguments".to_string(),
                        Value::Object(tc.arguments.iter().map(|(k, v)| (k.clone(), v.clone())).collect()),
                    );
                    map
                })
                .collect();

            if let Err(validation_err) = validation::validate_response(content_str, &tool_calls_as_maps) {
                warn!("Response validation failed: {:?}", validation_err);
                if !response.has_tool_calls() {
                    let hint = validation::generate_retry_prompt(&validation_err, 1);
                    ctx.messages.push(json!({
                        "role": "assistant",
                        "content": content_str
                    }));
                    ctx.messages.push(json!({
                        "role": "user",
                        "content": hint
                    }));
                    debug!("Injected validation retry hint");
                    continue;
                }
            }

            // Rescue pass: if local model consumed completion on reasoning and produced no
            // visible answer, force one concise no-thinking completion once.
            let empty_visible = response
                .content
                .as_ref()
                .map(|s| s.trim().is_empty())
                .unwrap_or(true);
            if ctx.core.is_local
                && !response.has_tool_calls()
                && empty_visible
                && response.finish_reason == "length"
                && !ctx.forced_finalize_attempted
            {
                ctx.forced_finalize_attempted = true;
                let rescue_tokens = effective_max_tokens.min(384).max(128);
                let mut rescue_messages = ctx.messages.clone();
                rescue_messages.push(json!({
                    "role": "user",
                    "content": "Return the final answer now. No reasoning. No tool calls. Max 6 lines."
                }));
                counters.inference_active.store(true, Ordering::Relaxed);
                match ctx.core
                    .provider
                    .chat(
                        &rescue_messages,
                        None,
                        Some(&ctx.core.model),
                        rescue_tokens,
                        0.2,
                        None,
                    )
                    .await
                {
                    Ok(r) => {
                        response = r;
                    }
                    Err(e) => {
                        warn!("Finalize rescue call failed: {}", e);
                    }
                }
                counters.inference_active.store(false, Ordering::Relaxed);
            }

            // Check for LLM provider errors before processing the response.
            // openai_compat.rs wraps errors as Ok(LLMResponse { finish_reason: "error", ... })
            // so they slip through as normal responses unless explicitly caught.
            if response.finish_reason == "error" {
                let err_msg = response.content.as_deref().unwrap_or("Unknown LLM error");
                error!("LLM provider error: {}", err_msg);

                // In local mode, check if the server is still alive.
                if ctx.core.is_local {
                    if let Some(base) = ctx.core.provider.get_api_base() {
                        if !crate::server::check_health(base).await {
                            error!("Local LLM server is down!");
                            ctx.final_content = "[LLM Error] Local server crashed. Use /restart or /local to recover.".into();
                            break;
                        }
                    }
                }

                ctx.final_content = format!("[LLM Error] {}", err_msg);
                break;
            }

            // Token telemetry: log actual vs estimated usage.
            {
                let estimated_prompt = TokenBudget::estimate_tokens(&ctx.messages);
                let actual_prompt = response.usage.get("prompt_tokens").copied().unwrap_or(-1);
                let actual_completion = response
                    .usage
                    .get("completion_tokens")
                    .copied()
                    .unwrap_or(-1);
                info!(
                    "tokens: estimated_prompt={}, actual_prompt={}, actual_completion={}, max_tokens={}",
                    estimated_prompt, actual_prompt, actual_completion, effective_max_tokens
                );
                // Store actual tokens for /status display.
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
            }

            if response.has_tool_calls() {
                let mut routed_tool_calls = response.tool_calls.clone();
                let mut router_decision: Option<role_policy::RouterDecision> = None;
                let mut router_decision_valid = false;
                let mut selected_plan: Option<toolplan::ToolPlan> = None;
                let available_tools = ctx.tools.tool_names();

                if ctx.core.tool_delegation_config.strict_router_schema {
                    let task_state = format!(
                        "Main content: {}\nCandidate tool calls: {}",
                        response.content.as_deref().unwrap_or("(empty)"),
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
                        ctx.final_content = "I can orchestrate the task, but direct tool calls from the main model are disabled by policy and strict router did not return a valid decision.".to_string();
                        break;
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
                                ctx.final_content = format!("Router produced invalid tool plan: {}", e);
                                break;
                            }
                        }
                    }
                    match plan.action {
                        ToolPlanAction::AskUser => {
                            ctx.final_content = plan
                                .args
                                .get("question")
                                .and_then(|v| v.as_str())
                                .map(|s| s.to_string())
                                .unwrap_or_else(|| {
                                    response
                                        .content
                                        .clone()
                                        .unwrap_or_else(|| "I need clarification to continue.".to_string())
                                });
                            break;
                        }
                        ToolPlanAction::Specialist => {
                            let context_summary = response.content.as_deref().unwrap_or("(empty)");
                            match dispatch_specialist(
                                &ctx.core, &plan.target, &plan.args,
                                &ctx.user_content, context_summary, &ctx.tools.tool_names(),
                            ).await {
                                Ok(text) => {
                                    ctx.messages.push(json!({"role":"user","content": text}));
                                    continue;
                                }
                                Err(e) => { ctx.final_content = e; break; }
                            }
                        }
                        ToolPlanAction::Subagent => {
                            match dispatch_subagent(
                                &ctx.tools, &plan.target, &plan.args,
                                &ctx.user_content, ctx.strict_local_only, &mut ctx.tool_guard,
                            ).await {
                                Ok(text) => {
                                    ctx.messages.push(json!({"role":"user","content": text}));
                                    continue;
                                }
                                Err(e) => { ctx.final_content = e; break; }
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
                    if let Some(text) = response.content.as_ref().filter(|s| !s.trim().is_empty()) {
                        ctx.final_content = text.clone();
                        break;
                    }
                    continue;
                }

                // Context pressure check: if high, log a warning. The correct
                // response is compaction, NOT spawning the main model as its
                // own tool runner (which doubles cost for no benefit).
                let context_tokens = TokenBudget::estimate_tokens(&ctx.messages);
                let max_tokens = ctx.core.token_budget.max_context();
                let pressure = if max_tokens > 0 {
                    context_tokens as f64 / max_tokens as f64
                } else {
                    0.0
                };
                if pressure > 0.7 && !ctx.core.tool_delegation_config.enabled {
                    debug!(
                        "Context pressure {:.0}% but delegation disabled — consider enabling delegation or compaction",
                        pressure * 100.0,
                    );
                }

                // Check if we should delegate to the tool runner.
                // Skip delegation if the provider was previously marked dead.
                let mut delegation_alive = counters.delegation_healthy.load(Ordering::Relaxed);
                // Periodically re-probe: every 10 inline calls, try delegation
                // once in case the server recovered (e.g. user restarted it).
                if !delegation_alive && ctx.core.tool_delegation_config.enabled {
                    let retries = counters
                        .delegation_retry_counter
                        .fetch_add(1, Ordering::Relaxed);
                    if retries > 0 && retries % 10 == 0 {
                        info!(
                            "Re-probing delegation provider (attempt {} since failure)",
                            retries
                        );
                        delegation_alive = true; // try this one time
                    } else {
                        debug!("Delegation provider unhealthy — inline execution ({}/10 until re-probe)", retries % 10);
                    }
                }
                let should_delegate = ctx.core.tool_delegation_config.enabled && delegation_alive;
                // Resolve provider+model from explicit config.
                let delegation_provider = ctx.core.tool_runner_provider.clone();
                let delegation_model = ctx.core.tool_runner_model.clone();

                if should_delegate {
                    if let (Some(ref tr_provider), Some(ref tr_model)) =
                        (&delegation_provider, &delegation_model)
                    {
                        debug!(
                            "Delegating {} tool calls to tool runner (model: {})",
                            routed_tool_calls.len(),
                            tr_model
                        );

                        // Local delegation providers require user-last messages
                        // for tool→generate flow.
                        // Check delegation provider's locality, not the main model's.
                        let needs_user_cont = ctx.core.is_local || delegation_provider.is_some();

                        // Detect [VERBATIM] marker: the main model is asking for
                        // raw tool output instead of a delegation summary.
                        let verbatim = response
                            .content
                            .as_ref()
                            .map(|c| c.contains("[VERBATIM]"))
                            .unwrap_or(false);

                        // The delegation model has a small context (8K tokens).
                        // Cap tool results to ~750 tokens (~3000 chars) so the system
                        // prompt, tool call messages, and response all fit comfortably.
                        // Use the main model's limit only if it's already smaller.
                        let delegation_result_limit = ctx.core.max_tool_result_chars.min(3000);

                        let runner_config = ToolRunnerConfig {
                            provider: tr_provider.clone(),
                            model: tr_model.clone(),
                            max_iterations: ctx.core.tool_delegation_config.max_iterations,
                            max_tokens: ctx.core.tool_delegation_config.max_tokens,
                            needs_user_continuation: needs_user_cont,
                            max_tool_result_chars: delegation_result_limit,
                            short_circuit_chars: 200,
                            depth: 0,
                            cancellation_token: ctx.cancellation_token.clone(),
                            verbatim,
                            budget: {
                                let cost_budget = ctx.core.tool_delegation_config.cost_budget;
                                if cost_budget > 0.0 {
                                    let prices =
                                        crate::agent::model_prices::ModelPrices::load().await;
                                    Some(Budget::root_with_cost(
                                        ctx.core.tool_delegation_config.max_iterations,
                                        2,
                                        cost_budget,
                                        std::sync::Arc::new(prices),
                                    ))
                                } else {
                                    Some(Budget::root(
                                        ctx.core.tool_delegation_config.max_iterations,
                                        2,
                                    ))
                                }
                            },
                        };

                        // Emit tool call start events for delegated calls.
                        if let Some(ref tx) = ctx.tool_event_tx {
                            for tc in &routed_tool_calls {
                                let preview: String = serde_json::to_string(&tc.arguments)
                                    .unwrap_or_default()
                                    .chars()
                                    .take(80)
                                    .collect();
                                let _ = tx.send(ToolEvent::CallStart {
                                    tool_name: tc.name.clone(),
                                    tool_call_id: tc.id.clone(),
                                    arguments_preview: preview,
                                });
                            }
                        }

                        // Build task description for the delegation model.
                        // Use the main model's content (alongside tool calls) as
                        // instructions — it naturally contains intent, constraints,
                        // and expected format. Fall back to the user message if
                        // the main model didn't produce text.
                        let tool_names: Vec<&str> = routed_tool_calls
                            .iter()
                            .map(|tc| tc.name.as_str())
                            .collect();
                        let instructions = response
                            .content
                            .as_deref()
                            .filter(|c| !c.trim().is_empty())
                            .map(|c| c.chars().take(400).collect::<String>())
                            .unwrap_or_else(|| ctx.user_content.chars().take(300).collect::<String>());
                        let task_desc = if ctx.core.tool_delegation_config.role_scoped_context_packs {
                            let task_state = format!(
                                "Tool lane execution\nPlanned tools: {}",
                                tool_names.join(", ")
                            );
                            role_policy::build_context_pack(
                                role_policy::Role::Main,
                                &instructions,
                                "(live turn; summary omitted)",
                                &task_state,
                                &ctx.tools.tool_names(),
                                2500,
                            )
                        } else {
                            format!(
                                "Instructions: {}\nTools to execute: {}",
                                instructions,
                                tool_names.join(", ")
                            )
                        };

                        let delegation_start = std::time::Instant::now();
                        let run_result = tool_runner::run_tool_loop(
                            &runner_config,
                            &routed_tool_calls,
                            &ctx.tools,
                            &task_desc,
                        )
                        .await;
                        let delegation_elapsed_ms = delegation_start.elapsed().as_millis() as u64;

                        // Only mark unhealthy on actual provider/tool-runner errors.
                        // Slow-but-successful runs are quality/perf issues, not
                        // liveness failures. Marking unhealthy on latency alone
                        // causes false negatives in long-running delegated tasks.
                        let is_hard_failure = run_result.error.is_some();
                        if is_hard_failure && !run_result.tool_results.is_empty() {
                            let reason = format!(
                                "delegation model errored: {}",
                                run_result.error.as_deref().unwrap_or("unknown error")
                            );
                            let results_preview: String = run_result
                                .tool_results
                                .first()
                                .map(|(_, name, data)| {
                                    format!(
                                        "[{}]: {}",
                                        name,
                                        data.chars().take(200).collect::<String>()
                                    )
                                })
                                .unwrap_or_default();
                            warn!(
                                "Delegation failed — {}. model={}, iterations={}, results={}, preview={}. \
                                 Marking unhealthy. Restart servers or toggle /local to recover.",
                                reason,
                                tr_model,
                                run_result.iterations_used,
                                run_result.tool_results.len(),
                                results_preview,
                            );
                            counters.delegation_healthy.store(false, Ordering::Relaxed);
                        } else if delegation_elapsed_ms > 30_000 {
                            debug!(
                                "Delegation run was slow ({} ms) but succeeded — keeping provider healthy",
                                delegation_elapsed_ms,
                            );
                        } else if run_result.summary.is_none()
                            && !run_result.tool_results.is_empty()
                        {
                            debug!(
                                "Delegation returned no summary (model={}, iters={}), using results inline",
                                tr_model, run_result.iterations_used,
                            );
                        } else if !counters.delegation_healthy.load(Ordering::Relaxed) {
                            // Re-probe succeeded — server recovered!
                            info!("Delegation provider recovered — re-enabling delegation");
                            counters.delegation_healthy.store(true, Ordering::Relaxed);
                            counters
                                .delegation_retry_counter
                                .store(0, Ordering::Relaxed);
                        }

                        debug!(
                            "Tool runner completed: {} results in {} iterations",
                            run_result.tool_results.len(),
                            run_result.iterations_used
                        );

                        // Build the assistant message with original tool_calls.
                        let tc_json: Vec<Value> = routed_tool_calls
                            .iter()
                            .map(|tc| tc.to_openai_json())
                            .collect();
                        ContextBuilder::add_assistant_message(
                            &mut ctx.messages,
                            response.content.as_deref(),
                            Some(&tc_json),
                        );

                        // Add tool results from the runner to the main context.
                        // ContentGate handles budget-aware sizing.
                        let preview_max = ctx.core.tool_delegation_config.max_result_preview_chars;

                        for tc in &routed_tool_calls {
                            // Find the matching result from the runner.
                            let full_data = run_result
                                .tool_results
                                .iter()
                                .find(|(id, _, _)| id == &tc.id)
                                .map(|(_, _, data)| data.as_str())
                                .unwrap_or("(no result)");

                            let injected = ctx.content_gate.admit_simple(full_data).into_text();

                            if ctx.core.provenance_config.enabled {
                                ContextBuilder::add_tool_result_immutable(
                                    &mut ctx.messages,
                                    &tc.id,
                                    &tc.name,
                                    &injected,
                                );
                            } else {
                                ContextBuilder::add_tool_result(
                                    &mut ctx.messages,
                                    &tc.id,
                                    &tc.name,
                                    &injected,
                                );
                            }
                            ctx.used_tools.insert(tc.name.clone());
                        }

                        // Inject the runner's summary so the main LLM knows what
                        // the tools found without needing full output.
                        let has_extra = run_result.tool_results.len() > routed_tool_calls.len();
                        if run_result.summary.is_some() || has_extra {
                            let summary_text = if has_extra {
                                let extra = tool_runner::format_results_for_context(
                                    &run_result,
                                    preview_max,
                                    None, // TODO: wire ContentGate here
                                );
                                format!(
                                    "[Tool runner executed {} additional calls]\n{}",
                                    run_result.tool_results.len() - routed_tool_calls.len(),
                                    extra
                                )
                            } else {
                                run_result.summary.clone().unwrap_or_default()
                            };
                            if !summary_text.is_empty() {
                                // Inject as user role, not assistant. This is
                                // injected context about what the tool runner did,
                                // not an actual LLM response. Using assistant role
                                // causes "assistant message prefill" errors when
                                // the Anthropic API sees the conversation ending
                                // with an assistant message.
                                let prefix = if verbatim {
                                    "[tool runner output]"
                                } else {
                                    "[tool runner summary]"
                                };
                                ctx.messages.push(json!({
                                    "role": "user",
                                    "content": format!("{} {}", prefix, summary_text)
                                }));
                            }
                        }

                        // Record learning + audit for all tool results.
                        let executor = format!("tool_runner:{}", tr_model);
                        let n_results = run_result.tool_results.len().max(1) as u64;
                        for (tool_call_id, tool_name, data) in &run_result.tool_results {
                            let ok = !data.starts_with("Error:");
                            // Distribute total delegation time across tool results.
                            let per_tool_ms = delegation_elapsed_ms / n_results;

                            // Emit tool call end event.
                            if let Some(ref tx) = ctx.tool_event_tx {
                                let _ = tx.send(ToolEvent::CallEnd {
                                    tool_name: tool_name.clone(),
                                    tool_call_id: tool_call_id.clone(),
                                    result_data: data.clone(),
                                    ok,
                                    duration_ms: per_tool_ms,
                                });
                            }

                            // Record in audit log.
                            if let Some(ref audit) = ctx.audit {
                                let _ = audit.record(
                                    tool_name,
                                    tool_call_id,
                                    &json!({}), // args not available from runner results
                                    data,
                                    ok,
                                    per_tool_ms,
                                    &executor,
                                );
                            }

                            ctx.core.learning.record_extended(
                                tool_name,
                                ok,
                                &data.chars().take(100).collect::<String>(),
                                if ok { None } else { Some(data) },
                                Some(tr_model),
                                Some(tr_model),
                                None,
                            );
                            ctx.used_tools.insert(tool_name.clone());
                        }

                        // Set response boundary flag if any delegated tool was exec/write_file.
                        for (_, tool_name, _) in &run_result.tool_results {
                            if tool_name == "exec" || tool_name == "write_file" {
                                ctx.force_response = true;
                                break;
                            }
                        }

                        // Local models via --jinja require strict user/assistant alternation.
                        // Tool results are folded into user messages by
                        // repair_for_strict_alternation() at the top of the loop.
                        // Do NOT add extra user continuation — it would create
                        // consecutive user messages.

                        // Continue the main loop — the main LLM will see the results.
                        continue;
                    }
                }

                // Inline path (default, unchanged): execute tools directly.
                let tc_json: Vec<Value> = routed_tool_calls
                    .iter()
                    .map(|tc| tc.to_openai_json())
                    .collect();

                ContextBuilder::add_assistant_message(
                    &mut ctx.messages,
                    response.content.as_deref(),
                    Some(&tc_json),
                );

                // Execute each tool call.
                for tc in &routed_tool_calls {
                    debug!("Executing tool: {} (id: {})", tc.name, tc.id);

                    // Emit tool call start event.
                    if let Some(ref tx) = ctx.tool_event_tx {
                        let preview: String = serde_json::to_string(&tc.arguments)
                            .unwrap_or_default()
                            .chars()
                            .take(80)
                            .collect();
                        let _ = tx.send(ToolEvent::CallStart {
                            tool_name: tc.name.clone(),
                            tool_call_id: tc.id.clone(),
                            arguments_preview: preview,
                        });
                    }

                    let start = std::time::Instant::now();

                    // Spawn heartbeat that emits Progress every 2s for tools that
                    // don't emit their own (everything except exec).
                    let heartbeat = if let Some(ref tx) = ctx.tool_event_tx {
                        let hb_tx = tx.clone();
                        let hb_name = tc.name.clone();
                        let hb_id = tc.id.clone();
                        let hb_start = start;
                        Some(tokio::spawn(async move {
                            let mut interval = tokio::time::interval(Duration::from_secs(2));
                            interval.tick().await; // skip first immediate tick
                            loop {
                                interval.tick().await;
                                let _ = hb_tx.send(ToolEvent::Progress {
                                    tool_name: hb_name.clone(),
                                    tool_call_id: hb_id.clone(),
                                    elapsed_ms: hb_start.elapsed().as_millis() as u64,
                                    output_preview: None,
                                });
                            }
                        }))
                    } else {
                        None
                    };

                    let result = if let Some(ref tx) = ctx.tool_event_tx {
                        use crate::agent::tools::base::ToolExecutionContext;
                        let exec_ctx = ToolExecutionContext {
                            event_tx: tx.clone(),
                            cancellation_token: ctx.cancellation_token
                                .as_ref()
                                .map(|t| t.child_token())
                                .unwrap_or_else(tokio_util::sync::CancellationToken::new),
                            tool_call_id: tc.id.clone(),
                        };
                        ctx.tools
                            .execute_with_context(&tc.name, tc.arguments.clone(), &exec_ctx)
                            .await
                    } else {
                        ctx.tools.execute(&tc.name, tc.arguments.clone()).await
                    };

                    // Stop heartbeat when tool finishes.
                    if let Some(hb) = heartbeat {
                        hb.abort();
                    }
                    let duration_ms = start.elapsed().as_millis() as u64;
                    debug!(
                        "Tool {} result ({}B, ok={}, {}ms)",
                        tc.name,
                        result.data.len(),
                        result.ok,
                        duration_ms
                    );
                    // Gate tool result through context budget.
                    let data = ctx.content_gate.admit_simple(&result.data).into_text();
                    if ctx.core.provenance_config.enabled {
                        ContextBuilder::add_tool_result_immutable(
                            &mut ctx.messages,
                            &tc.id,
                            &tc.name,
                            &data,
                        );
                    } else {
                        ContextBuilder::add_tool_result(&mut ctx.messages, &tc.id, &tc.name, &data);
                    }

                    // Emit tool call end event.
                    if let Some(ref tx) = ctx.tool_event_tx {
                        let _ = tx.send(ToolEvent::CallEnd {
                            tool_name: tc.name.clone(),
                            tool_call_id: tc.id.clone(),
                            result_data: result.data.clone(),
                            ok: result.ok,
                            duration_ms,
                        });
                    }

                    // Record in audit log.
                    if let Some(ref audit) = ctx.audit {
                        let args_value = serde_json::to_value(&tc.arguments).unwrap_or(json!({}));
                        let _ = audit.record(
                            &tc.name,
                            &tc.id,
                            &args_value,
                            &result.data,
                            result.ok,
                            duration_ms,
                            "inline",
                        );
                    }

                    // Track used tools.
                    ctx.used_tools.insert(tc.name.clone());

                    // Collect for turn audit summary.
                    ctx.turn_tool_entries.push(crate::agent::audit::TurnToolEntry {
                        name: tc.name.clone(),
                        id: tc.id.clone(),
                        ok: result.ok,
                        duration_ms,
                        result_chars: result.data.len(),
                    });

                    // Record tool outcome for learning.
                    let context_str: String = tc
                        .arguments
                        .values()
                        .filter_map(|v| v.as_str())
                        .next()
                        .unwrap_or_default()
                        .chars()
                        .take(100)
                        .collect();
                    ctx.core.learning.record(
                        &tc.name,
                        result.ok,
                        &context_str,
                        result.error.as_deref(),
                    );

                    // Set response boundary flag for exec/write_file.
                    if tc.name == "exec" || tc.name == "write_file" {
                        ctx.force_response = true;
                    }
                }

                // Local models via --jinja require strict user/assistant alternation.
                // Tool results are folded into user messages by
                // repair_for_strict_alternation() at the top of the loop.
                // Do NOT add extra user continuation — it would create
                // consecutive user messages.

                // Check for priority user messages injected mid-task.
                if let Some(ref mut rx) = ctx.priority_rx {
                    if let Ok(priority_msg) = rx.try_recv() {
                        ctx.messages.push(json!({
                            "role": "user",
                            "content": format!("[PRIORITY USER MESSAGE]: {}", priority_msg)
                        }));
                        // Continue to next LLM call — let the model see and adjust.
                    }
                }

                // Check cancellation between tool call iterations.
                if ctx.cancellation_token
                    .as_ref()
                    .map_or(false, |t| t.is_cancelled())
                {
                    break;
                }
            } else {
                // No tool calls -- the agent is done.
                ctx.final_content = response.content.unwrap_or_default();
                if ctx.final_content.trim().is_empty() {
                    ctx.final_content = "I couldn't produce a final answer in this turn. Please retry with /thinking off."
                        .to_string();
                }
                break;
            }
        }

        // If the loop exited via a non-streaming path (e.g. router preflight
        // decision, error, ask_user) the final_content was set directly without
        // any text deltas being sent through the streaming channel.  Emit it
        // now so the REPL's incremental renderer actually displays something.
        // Skip if content was already streamed via TextDelta chunks to avoid
        // duplication.
        if !ctx.final_content.is_empty() && !ctx.content_was_streamed {
            if let Some(ref tx) = ctx.text_delta_tx {
                let _ = tx.send(ctx.final_content.clone());
            }
        }
    }

    /// Phase 3: Finalize the response — persist session, build outbound message.
    ///
    /// Consumes the `TurnContext` (by value) since this is the terminal phase.
    /// Stores context stats, writes audit summaries, verifies claims, and
    /// constructs the `OutboundMessage`.
    async fn finalize_response(&self, mut ctx: TurnContext) -> Option<OutboundMessage> {
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
            if let Ok(mut guard) = counters.last_tools_called.lock() {
                *guard = tools_list;
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
            ctx.final_content = "I ran out of tool iterations before producing a final answer. The actions above may be incomplete.".to_string();
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
                    "Phantom tool claims detected ({} patterns): {:?}",
                    detection.matched_patterns.len(),
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
        if !ctx.final_content.is_empty() {
            ctx.messages.push(json!({"role": "assistant", "content": ctx.final_content.clone()}));
        }

        // Update session history — persist full message array including tool calls.
        // Skip system prompt (index 0) and pre-existing history.
        let new_messages: Vec<Value> = if ctx.new_start < ctx.messages.len() {
            ctx.messages[ctx.new_start..].to_vec()
        } else {
            // Fallback: save at least user + assistant text.
            let mut fallback = vec![json!({"role": "user", "content": ctx.user_content.clone()})];
            if !ctx.final_content.is_empty() {
                fallback.push(json!({"role": "assistant", "content": ctx.final_content.clone()}));
            }
            fallback
        };
        if !new_messages.is_empty() {
            ctx.core.sessions
                .add_messages_raw(&ctx.session_key, &new_messages)
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
        }

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

// ---------------------------------------------------------------------------
// AgentLoop (owns the receiver + orchestrates concurrency)
// ---------------------------------------------------------------------------

/// The core agent loop.
///
/// Consumes [`InboundMessage`]s from the bus, runs the LLM + tool loop, and
/// publishes [`OutboundMessage`]s when the agent produces a response.
///
/// In gateway mode, messages for different sessions run concurrently (up to
/// `max_concurrent_chats`), while messages within the same session are
/// serialized to preserve conversation ordering.
pub struct AgentLoop {
    shared: Arc<AgentLoopShared>,
    bus_inbound_rx: UnboundedReceiver<InboundMessage>,
    running: Arc<AtomicBool>,
    max_concurrent_chats: usize,
    reflection_spawned: AtomicBool,
}

impl AgentLoop {
    /// Create a new `AgentLoop`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        core_handle: SharedCoreHandle,
        bus_inbound_rx: UnboundedReceiver<InboundMessage>,
        bus_outbound_tx: UnboundedSender<OutboundMessage>,
        bus_inbound_tx: UnboundedSender<InboundMessage>,
        cron_service: Option<Arc<CronService>>,
        max_concurrent_chats: usize,
        email_config: Option<EmailConfig>,
        repl_display_tx: Option<UnboundedSender<String>>,
        providers_config: Option<crate::config::schema::ProvidersConfig>,
        proprioception_config: ProprioceptionConfig,
    ) -> Self {
        // Read core to initialize the subagent manager.
        let core = core_handle.swappable();
        let mut subagent_mgr = SubagentManager::new(
            core.provider.clone(),
            core.workspace.clone(),
            bus_inbound_tx.clone(),
            core.model.clone(),
            core.brave_api_key.clone(),
            core.exec_timeout,
            core.restrict_to_workspace,
            core.is_local,
            core.max_tool_result_chars,
        );
        if let Some(pc) = providers_config {
            subagent_mgr = subagent_mgr.with_providers_config(pc);
        }
        // Wire up the cheap default model for subagents from config.
        subagent_mgr = subagent_mgr.with_default_subagent_model(
            core.tool_delegation_config.default_subagent_model.clone(),
        );
        if let Some(ref dtx) = repl_display_tx {
            subagent_mgr = subagent_mgr.with_display_tx(dtx.clone());
        }
        if core.is_local {
            subagent_mgr = subagent_mgr.with_local_context_limit(core.token_budget.max_context());
        }

        // Create aha channel before subagent manager so we can pass the sender.
        let (aha_tx, aha_rx) = tokio::sync::mpsc::unbounded_channel();
        if proprioception_config.aha_channel {
            subagent_mgr = subagent_mgr.with_aha_tx(aha_tx.clone());
        }

        let subagents = Arc::new(subagent_mgr);

        // Load persisted bulletin from disk (warm start).
        let bulletin_cache = {
            let core = core_handle.swappable();
            let cache = crate::agent::bulletin::BulletinCache::new();
            if let Some(persisted) =
                crate::agent::bulletin::load_persisted_bulletin(&core.workspace)
            {
                cache.update(persisted);
            }
            cache.handle()
        };

        let system_state = Arc::new(arc_swap::ArcSwap::from_pointee(SystemState::default()));

        let shared = Arc::new(AgentLoopShared {
            core_handle,
            subagents,
            bus_outbound_tx,
            bus_inbound_tx,
            cron_service,
            email_config,
            repl_display_tx,
            bulletin_cache,
            system_state,
            proprioception_config,
            aha_rx: Arc::new(Mutex::new(aha_rx)),
            aha_tx,
            session_policies: Arc::new(Mutex::new(HashMap::new())),
        });

        Self {
            shared,
            bus_inbound_rx,
            running: Arc::new(AtomicBool::new(false)),
            max_concurrent_chats,
            reflection_spawned: AtomicBool::new(false),
        }
    }

    /// Spawn a periodic bulletin refresh task (compaction model, when idle).
    fn spawn_bulletin_refresh(shared: &Arc<AgentLoopShared>, running: &Arc<AtomicBool>) {
        let core = shared.core_handle.swappable();
        if !core.memory_enabled {
            return;
        }
        let provider = core.memory_provider.clone();
        let model = core.memory_model.clone();
        let workspace = core.workspace.clone();
        let cache = shared.bulletin_cache.clone();
        let running = running.clone();

        tokio::spawn(async move {
            // Initial delay: let the system settle before first bulletin.
            tokio::time::sleep(Duration::from_secs(5 * 60)).await;

            while running.load(Ordering::Relaxed) {
                debug!("Bulletin: refreshing...");
                if let Err(e) = crate::agent::bulletin::refresh_bulletin(
                    provider.as_ref(),
                    &model,
                    &workspace,
                    &cache,
                )
                .await
                {
                    warn!("Bulletin refresh failed: {}", e);
                }
                // Sleep until next refresh.
                tokio::time::sleep(Duration::from_secs(
                    crate::agent::bulletin::DEFAULT_BULLETIN_INTERVAL_S,
                ))
                .await;
            }
        });
        info!(
            "Bulletin refresh task spawned (every {}min)",
            crate::agent::bulletin::DEFAULT_BULLETIN_INTERVAL_S / 60
        );
    }

    /// Spawn a background reflection task if observations exceed threshold.
    fn spawn_background_reflection(shared: &Arc<AgentLoopShared>) {
        let core = shared.core_handle.swappable();
        if !core.memory_enabled {
            return;
        }
        let reflector = Reflector::new(
            core.memory_provider.clone(),
            core.memory_model.clone(),
            &core.workspace,
            core.reflection_threshold,
        );
        if reflector.should_reflect() {
            tokio::spawn(async move {
                info!("Background: reflecting on accumulated observations...");
                if let Err(e) = reflector.reflect().await {
                    warn!("Background reflection failed: {}", e);
                } else {
                    info!("Background reflection complete — MEMORY.md updated");
                }
            });
        }
    }

    /// Run the main agent loop until stopped.
    ///
    /// Messages for different sessions are processed concurrently (up to
    /// `max_concurrent_chats`). Messages within the same session are serialized.
    pub async fn run(&mut self) {
        self.running.store(true, Ordering::SeqCst);
        info!(
            "Agent loop started (max_concurrent_chats={})",
            self.max_concurrent_chats
        );

        // Spawn background reflection if observations have accumulated.
        Self::spawn_background_reflection(&self.shared);

        // Spawn periodic bulletin refresh (compaction model synthesizes briefing).
        Self::spawn_bulletin_refresh(&self.shared, &self.running);

        let semaphore = Arc::new(Semaphore::new(self.max_concurrent_chats));
        // Per-session locks to serialize messages within the same conversation.
        let session_locks: Arc<Mutex<HashMap<String, Arc<Mutex<()>>>>> =
            Arc::new(Mutex::new(HashMap::new()));

        while self.running.load(Ordering::SeqCst) {
            let msg = match tokio::time::timeout(Duration::from_secs(1), self.bus_inbound_rx.recv())
                .await
            {
                Ok(Some(msg)) => msg,
                Ok(None) => {
                    info!("Inbound channel closed, stopping agent loop");
                    break;
                }
                Err(_) => continue, // timeout - loop and check running flag
            };

            // Coalesce rapid messages from the same session (Telegram, WhatsApp).
            // Waits up to 400ms for follow-up messages before processing.
            let msg = if crate::bus::events::should_coalesce(&msg.channel) {
                let session = msg.session_key();
                let mut batch = vec![msg];
                let deadline = tokio::time::Instant::now() + Duration::from_millis(400);
                loop {
                    match tokio::time::timeout_at(deadline, self.bus_inbound_rx.recv()).await {
                        Ok(Some(next)) if next.session_key() == session => {
                            batch.push(next);
                        }
                        Ok(Some(other)) => {
                            // Different session — coalesce what we have, push other back.
                            // Can't push back into mpsc, so process inline as separate spawn.
                            let other_key = other.session_key();
                            let other_lock = {
                                let mut locks = session_locks.lock().await;
                                locks
                                    .entry(other_key)
                                    .or_insert_with(|| Arc::new(Mutex::new(())))
                                    .clone()
                            };
                            let other_shared = self.shared.clone();
                            let other_outbound_tx = self.shared.bus_outbound_tx.clone();
                            let _other_display_tx = self.shared.repl_display_tx.clone();
                            let other_sem = semaphore.clone();
                            tokio::spawn(async move {
                                if let Ok(permit) = other_sem.acquire_owned().await {
                                    let _guard = other_lock.lock().await;
                                    if let Some(resp) = other_shared
                                        .process_message(&other, None, None, None, None)
                                        .await
                                    {
                                        let _ = other_outbound_tx.send(resp);
                                    }
                                    drop(permit);
                                }
                            });
                            break;
                        }
                        _ => break, // timeout or channel closed
                    }
                }
                if batch.len() > 1 {
                    debug!("Coalesced {} messages for session", batch.len());
                }
                crate::bus::events::coalesce_messages(batch)
            } else {
                msg
            };

            // System messages (subagent announces) are handled inline (fast).
            let is_system = msg
                .metadata
                .get("is_system")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

            if is_system {
                debug!(
                    "Processing system message: {}",
                    &msg.content[..msg.content.len().min(80)]
                );
                let outbound = OutboundMessage::new(&msg.channel, &msg.chat_id, &msg.content);
                if let Err(e) = self.shared.bus_outbound_tx.send(outbound) {
                    error!("Failed to publish outbound message: {}", e);
                }
                continue;
            }

            // Acquire a concurrency permit.
            let permit = match semaphore.clone().acquire_owned().await {
                Ok(p) => p,
                Err(_) => {
                    error!("Semaphore closed unexpectedly");
                    break;
                }
            };

            // Get or create the per-session lock.
            let session_key = msg.session_key();
            let session_lock = {
                let mut locks = session_locks.lock().await;
                locks
                    .entry(session_key)
                    .or_insert_with(|| Arc::new(Mutex::new(())))
                    .clone()
            };

            let shared = self.shared.clone();
            let outbound_tx = self.shared.bus_outbound_tx.clone();
            let display_tx = self.shared.repl_display_tx.clone();

            tokio::spawn(async move {
                // Serialize within the same session.
                let _session_guard = session_lock.lock().await;

                // Notify REPL about inbound channel message.
                if let Some(ref dtx) = display_tx {
                    let preview = if msg.content.len() > 120 {
                        let end = crate::utils::helpers::floor_char_boundary(&msg.content, 120);
                        format!("{}...", &msg.content[..end])
                    } else {
                        msg.content.clone()
                    };
                    let _ = dtx.send(format!(
                        "\x1b[2m[{}]\x1b[0m \x1b[36m{}\x1b[0m: {}",
                        msg.channel, msg.sender_id, preview
                    ));
                }

                let response = shared.process_message(&msg, None, None, None, None).await;

                if let Some(outbound) = response {
                    // Notify REPL about outbound response.
                    if let Some(ref dtx) = display_tx {
                        let preview = if outbound.content.len() > 120 {
                            let end =
                                crate::utils::helpers::floor_char_boundary(&outbound.content, 120);
                            format!("{}...", &outbound.content[..end])
                        } else {
                            outbound.content.clone()
                        };
                        let _ = dtx.send(format!(
                            "\x1b[2m[{}]\x1b[0m \x1b[33mbot\x1b[0m: {}",
                            outbound.channel, preview
                        ));
                    }
                    if let Err(e) = outbound_tx.send(outbound) {
                        error!("Failed to publish outbound message: {}", e);
                    }
                }

                drop(permit); // release concurrency slot
            });
        }

        info!("Agent loop stopped");
    }

    /// Return a handle to the subagent manager.
    pub fn subagent_manager(&self) -> Arc<SubagentManager> {
        self.shared.subagents.clone()
    }

    /// Signal the agent loop to stop.
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
    }

    /// Process a message directly (for CLI / cron usage) without going through
    /// the bus.
    pub async fn process_direct(
        &self,
        content: &str,
        session_key: &str,
        channel: &str,
        chat_id: &str,
    ) -> String {
        self.process_direct_with_lang(content, session_key, channel, chat_id, None)
            .await
    }

    /// Like `process_direct` but allows passing a detected language code
    /// (e.g. "it", "es") so the LLM responds in that language.
    pub async fn process_direct_with_lang(
        &self,
        content: &str,
        session_key: &str,
        channel: &str,
        chat_id: &str,
        detected_language: Option<&str>,
    ) -> String {
        // Spawn background reflection once per session (on first message).
        if !self.reflection_spawned.swap(true, Ordering::SeqCst) {
            Self::spawn_background_reflection(&self.shared);
        }

        let mut msg = InboundMessage::new(channel, "user", chat_id, content);
        msg.metadata
            .insert("session_key".to_string(), json!(session_key));
        if let Some(lang) = detected_language {
            msg.metadata
                .insert("detected_language".to_string(), json!(lang));
        }

        match self
            .shared
            .process_message(&msg, None, None, None, None)
            .await
        {
            Some(response) => response.content,
            None => String::new(),
        }
    }

    /// Like `process_direct_with_lang` but streams text deltas to `text_delta_tx`
    /// as they arrive from the LLM. Returns the full response text.
    pub async fn process_direct_streaming(
        &self,
        content: &str,
        session_key: &str,
        channel: &str,
        chat_id: &str,
        detected_language: Option<&str>,
        text_delta_tx: tokio::sync::mpsc::UnboundedSender<String>,
        tool_event_tx: Option<tokio::sync::mpsc::UnboundedSender<ToolEvent>>,
        cancellation_token: Option<tokio_util::sync::CancellationToken>,
        priority_rx: Option<tokio::sync::mpsc::UnboundedReceiver<String>>,
    ) -> String {
        if !self.reflection_spawned.swap(true, Ordering::SeqCst) {
            Self::spawn_background_reflection(&self.shared);
        }

        let mut msg = InboundMessage::new(channel, "user", chat_id, content);
        msg.metadata
            .insert("session_key".to_string(), json!(session_key));
        if let Some(lang) = detected_language {
            msg.metadata
                .insert("detected_language".to_string(), json!(lang));
        }

        match self
            .shared
            .process_message(
                &msg,
                Some(text_delta_tx),
                tool_event_tx,
                cancellation_token,
                priority_rx,
            )
            .await
        {
            Some(response) => response.content,
            None => String::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tool proxy wrappers
// ---------------------------------------------------------------------------
//
// Because `Arc<MessageTool>` etc. don't implement `Tool` directly (the trait
// requires owned `Box<dyn Tool>`), we create thin proxy wrappers that
// implement `Tool` by delegating to the inner `Arc`.

/// Proxy that wraps `Arc<MessageTool>` to satisfy `Tool`.
struct MessageToolProxy(Arc<MessageTool>);

#[async_trait::async_trait]
impl crate::agent::tools::Tool for MessageToolProxy {
    fn name(&self) -> &str {
        self.0.name()
    }
    fn description(&self) -> &str {
        self.0.description()
    }
    fn parameters(&self) -> Value {
        self.0.parameters()
    }
    async fn execute(&self, params: HashMap<String, Value>) -> String {
        self.0.execute(params).await
    }
}

/// Proxy that wraps `Arc<SpawnTool>` to satisfy `Tool`.
struct SpawnToolProxy(Arc<SpawnTool>);

#[async_trait::async_trait]
impl crate::agent::tools::Tool for SpawnToolProxy {
    fn name(&self) -> &str {
        self.0.name()
    }
    fn description(&self) -> &str {
        self.0.description()
    }
    fn parameters(&self) -> Value {
        self.0.parameters()
    }
    async fn execute(&self, params: HashMap<String, Value>) -> String {
        self.0.execute(params).await
    }
}

/// Proxy that wraps `Arc<CronScheduleTool>` to satisfy `Tool`.
struct CronToolProxy(Arc<CronScheduleTool>);

#[async_trait::async_trait]
impl crate::agent::tools::Tool for CronToolProxy {
    fn name(&self) -> &str {
        self.0.name()
    }
    fn description(&self) -> &str {
        self.0.description()
    }
    fn parameters(&self) -> Value {
        self.0.parameters()
    }
    async fn execute(&self, params: HashMap<String, Value>) -> String {
        self.0.execute(params).await
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::schema::ProviderConfig;
    use crate::providers::openai_compat::OpenAICompatProvider;
    use async_trait::async_trait;

    /// Minimal mock LLM provider for wiring tests.
    struct MockLLM {
        name: String,
    }

    impl MockLLM {
        fn named(name: &str) -> Arc<dyn LLMProvider> {
            Arc::new(Self {
                name: name.to_string(),
            })
        }
    }

    #[async_trait]
    impl LLMProvider for MockLLM {
        async fn chat(
            &self,
            _messages: &[Value],
            _tools: Option<&[Value]>,
            _model: Option<&str>,
            _max_tokens: u32,
            _temperature: f64,
            _thinking_budget: Option<u32>,
        ) -> anyhow::Result<crate::providers::base::LLMResponse> {
            Ok(crate::providers::base::LLMResponse {
                content: Some("mock".to_string()),
                tool_calls: vec![],
                finish_reason: "stop".to_string(),
                usage: std::collections::HashMap::new(),
            })
        }

        fn get_default_model(&self) -> &str {
            &self.name
        }
    }

    struct StaticResponseLLM {
        name: String,
        body: String,
    }

    impl StaticResponseLLM {
        fn new(name: &str, body: &str) -> Self {
            Self {
                name: name.to_string(),
                body: body.to_string(),
            }
        }
    }

    #[async_trait]
    impl LLMProvider for StaticResponseLLM {
        async fn chat(
            &self,
            _messages: &[Value],
            _tools: Option<&[Value]>,
            _model: Option<&str>,
            _max_tokens: u32,
            _temperature: f64,
            _thinking_budget: Option<u32>,
        ) -> anyhow::Result<crate::providers::base::LLMResponse> {
            Ok(crate::providers::base::LLMResponse {
                content: Some(self.body.clone()),
                tool_calls: vec![],
                finish_reason: "stop".to_string(),
                usage: std::collections::HashMap::new(),
            })
        }

        fn get_default_model(&self) -> &str {
            &self.name
        }
    }

    /// Helper to build a SwappableCore with minimal config for wiring tests.
    fn build_test_core(
        delegation_enabled: bool,
        delegation_provider: Option<Arc<dyn LLMProvider>>,
        config_provider: Option<ProviderConfig>,
    ) -> SwappableCore {
        let workspace = tempfile::tempdir().unwrap().into_path();
        let main = MockLLM::named("main-provider");
        let td = ToolDelegationConfig {
            enabled: delegation_enabled,
            model: "delegation-model".to_string(),
            provider: config_provider,
            auto_local: true,
            ..Default::default()
        };
        build_swappable_core(SwappableCoreConfig {
            provider: main,
            workspace,
            model: "main-model".to_string(),
            max_iterations: 10,
            max_tokens: 4096,
            temperature: 0.7,
            max_context_tokens: 16384,
            brave_api_key: None,
            exec_timeout: 30,
            restrict_to_workspace: false,
            memory_config: MemoryConfig::default(),
            is_local: false,
            compaction_provider: None,
            tool_delegation: td,
            provenance: ProvenanceConfig::default(),
            max_tool_result_chars: 2000,
            delegation_provider,
            specialist_provider: None,
            trio_config: TrioConfig::default(),
        })
    }

    #[test]
    fn test_provenance_warning_role_local_safe() {
        assert_eq!(provenance_warning_role(true), "user");
        assert_eq!(provenance_warning_role(false), "system");
    }

    #[test]
    fn test_extract_json_object_from_markdown_fence() {
        let raw = "```json\n{\"action\":\"tool\",\"target\":\"exec\",\"args\":{},\"confidence\":0.9}\n```";
        let obj = extract_json_object(raw).expect("json object");
        assert!(obj.starts_with('{'));
        assert!(obj.ends_with('}'));
        assert!(obj.contains("\"action\":\"tool\""));
    }

    #[test]
    fn test_extract_json_object_none_when_missing() {
        assert!(extract_json_object("no json here").is_none());
    }

    #[tokio::test]
    async fn test_request_strict_router_decision_action_matrix() {
        let cases = vec![
            (
                r#"{"action":"tool","target":"read_file","args":{"path":"README.md"},"confidence":0.9}"#,
                "tool",
            ),
            (
                r#"{"action":"subagent","target":"builder","args":{"task":"x"},"confidence":0.8}"#,
                "subagent",
            ),
            (
                r#"{"action":"specialist","target":"summarizer","args":{"style":"tight"},"confidence":0.7}"#,
                "specialist",
            ),
            (
                r#"{"action":"ask_user","target":"clarify","args":{"question":"Need path?"},"confidence":0.6}"#,
                "ask_user",
            ),
        ];

        for (raw, expected_action) in cases {
            let llm = StaticResponseLLM::new("router", raw);
            let decision = request_strict_router_decision(
                &llm,
                "router",
                "route this action with strict schema",
                false,
                0.6,
            )
            .await
            .expect("valid strict router decision");
            assert_eq!(decision.action, expected_action);
        }
    }

    /// Real-provider trio probe.
    ///
    /// Runs against live OpenAI-compatible endpoints (e.g. LM Studio):
    /// - main: `NANOBOT_REAL_MAIN_BASE` (default: http://127.0.0.1:8080/v1)
    /// - router: `NANOBOT_REAL_ROUTER_BASE` (default: http://127.0.0.1:8094/v1)
    /// - specialist: `NANOBOT_REAL_SPECIALIST_BASE` (default: http://127.0.0.1:8095/v1)
    ///
    /// Optional model overrides:
    /// - `NANOBOT_REAL_MAIN_MODEL`
    /// - `NANOBOT_REAL_ROUTER_MODEL`
    /// - `NANOBOT_REAL_SPECIALIST_MODEL`
    #[tokio::test]
    #[ignore = "requires running local providers on main/router/specialist ports"]
    async fn test_real_providers_trio_probe() {
        let main_base = std::env::var("NANOBOT_REAL_MAIN_BASE")
            .unwrap_or_else(|_| "http://127.0.0.1:8080/v1".to_string());
        let router_base = std::env::var("NANOBOT_REAL_ROUTER_BASE")
            .unwrap_or_else(|_| "http://127.0.0.1:8094/v1".to_string());
        let specialist_base = std::env::var("NANOBOT_REAL_SPECIALIST_BASE")
            .unwrap_or_else(|_| "http://127.0.0.1:8095/v1".to_string());
        let main_model = std::env::var("NANOBOT_REAL_MAIN_MODEL")
            .unwrap_or_else(|_| "local-model".to_string());
        let router_model = std::env::var("NANOBOT_REAL_ROUTER_MODEL")
            .unwrap_or_else(|_| "local-delegation".to_string());
        let specialist_model = std::env::var("NANOBOT_REAL_SPECIALIST_MODEL")
            .unwrap_or_else(|_| "local-specialist".to_string());

        let main = OpenAICompatProvider::new("local", Some(&main_base), Some(&main_model));
        let router = OpenAICompatProvider::new("local", Some(&router_base), Some(&router_model));
        let specialist =
            OpenAICompatProvider::new("local", Some(&specialist_base), Some(&specialist_model));

        let mut failures: Vec<String> = Vec::new();

        // Router: force each action in a constrained prompt and verify strict parsing.
        let router_cases = vec![
            ("tool", "Return action=tool target=read_file args={\"path\":\"README.md\"}."),
            (
                "subagent",
                "Return action=subagent target=builder args={\"task\":\"diagnose issue\"}.",
            ),
            (
                "specialist",
                "Return action=specialist target=summarizer args={\"objective\":\"compress\"}.",
            ),
            (
                "ask_user",
                "Return action=ask_user target=clarify args={\"question\":\"Which file?\"}.",
            ),
        ];
        for (expected_action, directive) in router_cases {
            let pack = format!("{}\nFollow schema strictly.", directive);
            match request_strict_router_decision(&router, &router_model, &pack, false, 0.6).await {
                Ok(d) => {
                    if d.action != expected_action {
                        failures.push(format!(
                            "router action mismatch: expected={}, got={} target={}",
                            expected_action, d.action, d.target
                        ));
                    }
                }
                Err(e) => failures.push(format!("router {} failed: {}", expected_action, e)),
            }
        }

        // Specialist must produce non-empty response (with warmup retries).
        let specialist_messages = vec![
            json!({"role":"system","content":"ROLE=SPECIALIST\nReturn concise output."}),
            json!({"role":"user","content":"Summarize: tool call failed because server was down and port conflicted."}),
        ];
        let mut specialist_ok = false;
        for _ in 0..10 {
            match specialist
                .chat(
                    &specialist_messages,
                    None,
                    Some(&specialist_model),
                    256,
                    0.2,
                    None,
                )
                .await
            {
                Ok(resp) => {
                    let text = resp.content.unwrap_or_default();
                    if !text.trim().is_empty() {
                        specialist_ok = true;
                        break;
                    }
                }
                Err(e) => {
                    let msg = e.to_string();
                    if msg.to_lowercase().contains("loading model")
                        || msg.to_lowercase().contains("503")
                    {
                        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                        continue;
                    }
                    failures.push(format!("specialist call failed: {}", msg));
                    break;
                }
            }
            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
        }
        if !specialist_ok {
            failures.push("specialist did not become ready / returned empty output".to_string());
        }

        // Main provider smoke: should answer plain text with no tools when none offered.
        let main_messages = vec![json!({"role":"user","content":"Reply with exactly: main-ok"})];
        match main
            .chat(&main_messages, None, Some(&main_model), 64, 0.0, None)
            .await
        {
            Ok(resp) => {
                if resp.has_tool_calls() {
                    failures.push("main returned tool calls unexpectedly".to_string());
                }
                let text = resp.content.unwrap_or_default();
                if !text.to_lowercase().contains("main-ok") {
                    failures.push(format!("main output mismatch: {}", text));
                }
            }
            Err(e) => failures.push(format!("main call failed: {}", e)),
        }

        if !failures.is_empty() {
            panic!(
                "real trio probe failed (main={}, router={}, specialist={}):\n{}",
                main_base,
                router_base,
                specialist_base,
                failures.join("\n")
            );
        }
    }

    // -- Delegation provider wiring tests --

    #[test]
    fn test_delegation_disabled_no_runner_provider() {
        let core = build_test_core(false, None, None);
        assert!(
            core.tool_runner_provider.is_none(),
            "When delegation is disabled, tool_runner_provider should be None"
        );
        assert!(core.tool_runner_model.is_none());
    }

    #[test]
    fn test_delegation_enabled_with_auto_provider() {
        // When an auto-spawned delegation provider is passed, it should be used
        let dp = MockLLM::named("auto-delegation");
        let core = build_test_core(true, Some(dp), None);

        assert!(core.tool_runner_provider.is_some());
        let provider = core.tool_runner_provider.as_ref().unwrap();
        assert_eq!(
            provider.get_default_model(),
            "auto-delegation",
            "Should use the auto-spawned delegation provider"
        );
        assert_eq!(core.tool_runner_model.as_deref(), Some("delegation-model"));
    }

    #[test]
    fn test_delegation_auto_provider_takes_priority_over_config() {
        // Auto-spawned provider should take priority over config provider
        let dp = MockLLM::named("auto-delegation");
        let config_provider = ProviderConfig {
            api_key: "key".to_string(),
            api_base: Some("http://localhost:9999/v1".to_string()),
        };
        let core = build_test_core(true, Some(dp), Some(config_provider));

        let provider = core.tool_runner_provider.as_ref().unwrap();
        assert_eq!(
            provider.get_default_model(),
            "auto-delegation",
            "Auto-spawned provider should beat config provider"
        );
    }

    #[test]
    fn test_delegation_config_provider_used_when_no_auto() {
        // When no auto provider, but config has one, it should create OpenAICompatProvider
        let config_provider = ProviderConfig {
            api_key: "key".to_string(),
            api_base: Some("http://localhost:9999/v1".to_string()),
        };
        let core = build_test_core(true, None, Some(config_provider));

        assert!(
            core.tool_runner_provider.is_some(),
            "Should have a provider from config"
        );
    }

    #[test]
    fn test_delegation_falls_back_to_main_provider() {
        // When delegation enabled but no auto provider and no config provider,
        // should fall back to main
        let core = build_test_core(true, None, None);

        assert!(core.tool_runner_provider.is_some());
        let provider = core.tool_runner_provider.as_ref().unwrap();
        assert_eq!(
            provider.get_default_model(),
            "main-provider",
            "Should fall back to main provider"
        );
    }

    #[test]
    fn test_delegation_model_uses_config_model() {
        let core = build_test_core(true, None, None);
        assert_eq!(
            core.tool_runner_model.as_deref(),
            Some("delegation-model"),
            "Should use the model from ToolDelegationConfig"
        );
    }

    #[test]
    fn test_delegation_model_falls_back_to_main_when_empty() {
        let workspace = tempfile::tempdir().unwrap().into_path();
        let main = MockLLM::named("main-provider");
        let td = ToolDelegationConfig {
            enabled: true,
            model: String::new(), // Empty → fall back to main model
            auto_local: true,
            ..Default::default()
        };
        let core = build_swappable_core(SwappableCoreConfig {
            provider: main,
            workspace,
            model: "main-model".to_string(),
            max_iterations: 10,
            max_tokens: 4096,
            temperature: 0.7,
            max_context_tokens: 16384,
            brave_api_key: None,
            exec_timeout: 30,
            restrict_to_workspace: false,
            memory_config: MemoryConfig::default(),
            is_local: false,
            compaction_provider: None,
            tool_delegation: td,
            provenance: ProvenanceConfig::default(),
            max_tool_result_chars: 2000,
            delegation_provider: None,
            specialist_provider: None,
            trio_config: TrioConfig::default(),
        });
        assert_eq!(
            core.tool_runner_model.as_deref(),
            Some("main-model"),
            "Empty delegation model should fall back to main model"
        );
    }

    #[test]
    fn test_delegation_disabled_ignores_passed_provider() {
        // Even if a delegation_provider is passed, it should be ignored
        // when delegation is disabled.
        let dp = MockLLM::named("auto-delegation");
        let core = build_test_core(false, Some(dp), None);

        assert!(
            core.tool_runner_provider.is_none(),
            "Delegation disabled should ignore passed provider"
        );
        assert!(core.tool_runner_model.is_none());
    }

    #[test]
    fn test_delegation_with_is_local_true() {
        // Verify wiring works when is_local=true (uses lite context builder)
        let workspace = tempfile::tempdir().unwrap().into_path();
        let main = MockLLM::named("local-main");
        let dp = MockLLM::named("local-delegation");
        let td = ToolDelegationConfig {
            enabled: true,
            model: "delegation-model".to_string(),
            auto_local: true,
            ..Default::default()
        };
        let core = build_swappable_core(SwappableCoreConfig {
            provider: main,
            workspace,
            model: "local-model".to_string(),
            max_iterations: 10,
            max_tokens: 4096,
            temperature: 0.7,
            max_context_tokens: 16384,
            brave_api_key: None,
            exec_timeout: 30,
            restrict_to_workspace: false,
            memory_config: MemoryConfig::default(),
            is_local: true,
            compaction_provider: None,
            tool_delegation: td,
            provenance: ProvenanceConfig::default(),
            max_tool_result_chars: 2000,
            delegation_provider: Some(dp),
            specialist_provider: None,
            trio_config: TrioConfig::default(),
        });

        assert!(core.is_local);
        assert!(core.tool_runner_provider.is_some());
        assert_eq!(
            core.tool_runner_provider
                .as_ref()
                .unwrap()
                .get_default_model(),
            "local-delegation",
            "Local mode should still use the delegation provider"
        );
    }

    #[test]
    fn test_delegation_with_compaction_and_delegation_providers() {
        // Both compaction and delegation providers set — should not interfere
        let workspace = tempfile::tempdir().unwrap().into_path();
        let main = MockLLM::named("main");
        let compaction = MockLLM::named("compaction");
        let delegation = MockLLM::named("delegation");
        let td = ToolDelegationConfig {
            enabled: true,
            model: "deleg-model".to_string(),
            auto_local: true,
            ..Default::default()
        };
        let core = build_swappable_core(SwappableCoreConfig {
            provider: main,
            workspace,
            model: "main-model".to_string(),
            max_iterations: 10,
            max_tokens: 4096,
            temperature: 0.7,
            max_context_tokens: 16384,
            brave_api_key: None,
            exec_timeout: 30,
            restrict_to_workspace: false,
            memory_config: MemoryConfig::default(),
            is_local: true,
            compaction_provider: Some(compaction),
            tool_delegation: td,
            provenance: ProvenanceConfig::default(),
            max_tool_result_chars: 2000,
            delegation_provider: Some(delegation),
            specialist_provider: None,
            trio_config: TrioConfig::default(),
        });

        // Compaction provider goes to memory_provider, delegation to tool_runner
        assert_eq!(
            core.memory_provider.get_default_model(),
            "compaction",
            "Memory should use compaction provider"
        );
        assert_eq!(
            core.tool_runner_provider
                .as_ref()
                .unwrap()
                .get_default_model(),
            "delegation",
            "Tool runner should use delegation provider"
        );
    }
}
