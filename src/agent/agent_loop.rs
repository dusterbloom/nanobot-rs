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
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use serde_json::{json, Value};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use tokio::sync::{Mutex, Semaphore};
use tracing::{debug, error, info, warn};

use crate::agent::audit::{AuditLog, ToolEvent};
use crate::agent::compaction::ContextCompactor;
use crate::agent::context::ContextBuilder;
use crate::agent::learning::LearningStore;
use crate::agent::reflector::Reflector;
use crate::agent::working_memory::WorkingMemoryStore;
use crate::agent::subagent::SubagentManager;
use crate::agent::thread_repair;
use crate::agent::token_budget::TokenBudget;
use crate::agent::tools::{
    CheckInboxTool, CronScheduleTool, EditFileTool, ExecTool, ListDirTool, MessageTool,
    ReadFileTool, ReadSkillTool, RecallTool, SendCallback, SendEmailTool, SpawnCallback, SpawnTool, ToolRegistry,
    WebFetchTool, WebSearchTool, WriteFileTool,
};
use crate::bus::events::{InboundMessage, OutboundMessage};
use crate::agent::tool_runner::{self, ToolRunnerConfig};
use crate::config::schema::{EmailConfig, MemoryConfig, ProvenanceConfig, ToolDelegationConfig};
use crate::cron::service::CronService;
use crate::providers::base::{LLMProvider, StreamChunk};
use crate::providers::openai_compat::OpenAICompatProvider;
use crate::session::manager::SessionManager;

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
    pub tool_runner_provider: Option<Arc<dyn LLMProvider>>,
    pub tool_runner_model: Option<String>,
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
    /// the delegation LLM returns an error, causing subsequent calls to fall
    /// through to inline execution. Reset to `true` when the core is rebuilt
    /// (e.g. `/local` toggle restarts servers).
    pub delegation_healthy: AtomicBool,
    /// Counts tool calls since delegation was marked unhealthy. Used to
    /// periodically re-probe: every 10 inline calls, try delegation once
    /// more in case the server recovered.
    pub delegation_retry_counter: AtomicU64,
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

/// Build a `SwappableCore` from the given parameters.
///
/// When `is_local` is true, the compactor and memory operations use a dedicated
/// `compaction_provider` if supplied (e.g. a CPU-only Qwen3-0.6B server), or
/// fall back to the main (local) provider.
pub fn build_swappable_core(
    provider: Arc<dyn LLMProvider>,
    workspace: PathBuf,
    model: String,
    max_iterations: u32,
    max_tokens: u32,
    temperature: f64,
    max_context_tokens: usize,
    brave_api_key: Option<String>,
    exec_timeout: u64,
    restrict_to_workspace: bool,
    memory_config: MemoryConfig,
    is_local: bool,
    compaction_provider: Option<Arc<dyn LLMProvider>>,
    tool_delegation: ToolDelegationConfig,
    provenance: ProvenanceConfig,
    max_tool_result_chars: usize,
    delegation_provider: Option<Arc<dyn LLMProvider>>,
) -> SwappableCore {
    let mut context = if is_local {
        ContextBuilder::new_lite(&workspace)
    } else {
        ContextBuilder::new(&workspace)
    };
    context.model_name = model.clone();
    // Inject provenance verification rules when enabled.
    if provenance.enabled && provenance.system_prompt_rules {
        context.provenance_enabled = true;
    }
    // RLM lazy skills: skills loaded as summaries, fetched on demand.
    context.lazy_skills = memory_config.lazy_skills;
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
    let compactor = ContextCompactor::new(memory_provider.clone(), memory_model.clone(), max_context_tokens)
        .with_thresholds(memory_config.compaction_threshold_percent, memory_config.compaction_threshold_tokens);
    let learning = LearningStore::new(&workspace);
    let working_memory = WorkingMemoryStore::new(&workspace);

    // Build tool runner provider if delegation is enabled.
    let (tool_runner_provider, tool_runner_model) = if tool_delegation.enabled {
        let tr_model = if tool_delegation.model.is_empty() {
            model.clone()
        } else {
            tool_delegation.model.clone()
        };
        let tr_provider: Arc<dyn LLMProvider> = if let Some(dp) = delegation_provider {
            dp // Auto-spawned local delegation server
        } else if let Some(ref tr_cfg) = tool_delegation.provider {
            Arc::new(OpenAICompatProvider::new(
                &tr_cfg.api_key,
                tr_cfg.api_base.as_deref().or(Some("http://localhost:8080/v1")),
                None,
            ))
        } else {
            provider.clone() // Fallback to main
        };
        (Some(tr_provider), Some(tr_model))
    } else {
        (None, None)
    };

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
        working_memory_budget: memory_config.working_memory_budget,
        brave_api_key,
        exec_timeout,
        restrict_to_workspace,
        memory_enabled: memory_config.enabled,
        memory_provider,
        memory_model,
        reflection_threshold: memory_config.reflection_threshold,
        is_local,
        tool_runner_provider,
        tool_runner_model,
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
    bus_inbound_tx: UnboundedSender<InboundMessage>,
    cron_service: Option<Arc<CronService>>,
    email_config: Option<EmailConfig>,
    repl_display_tx: Option<UnboundedSender<String>>,
}

impl AgentLoopShared {
    /// Build a fresh [`ToolRegistry`] with context-sensitive tools (message,
    /// spawn, cron) pre-configured for a specific channel/chat_id.
    ///
    /// Takes a snapshot of `SwappableCore` so the registry is consistent for the
    /// entire message processing.
    async fn build_tools(&self, core: &SwappableCore, channel: &str, chat_id: &str) -> ToolRegistry {
        let mut tools = ToolRegistry::new();

        // File system tools (stateless).
        tools.register(Box::new(ReadFileTool));
        tools.register(Box::new(WriteFileTool));
        tools.register(Box::new(EditFileTool));
        tools.register(Box::new(ListDirTool));

        // Shell (stateless config).
        tools.register(Box::new(ExecTool::new(
            core.exec_timeout,
            Some(core.workspace.to_string_lossy().to_string()),
            None,
            None,
            core.restrict_to_workspace,
            core.max_tool_result_chars,
        )));

        // Web (stateless config).
        tools.register(Box::new(WebSearchTool::new(core.brave_api_key.clone(), 5)));
        tools.register(Box::new(WebFetchTool::new(50_000)));

        // Memory recall (stateless config).
        tools.register(Box::new(RecallTool::new(&core.workspace)));

        // Skill reader — on-demand skill loading (RLM lazy mode).
        tools.register(Box::new(ReadSkillTool::new(&core.workspace)));

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
        let spawn_cb: SpawnCallback = Arc::new(move |task, label, ch, cid| {
            let mgr = subagents_ref.clone();
            Box::pin(async move { mgr.spawn(task, label, ch, cid).await })
        });
        let spawn_tool = Arc::new(SpawnTool::new());
        // Set callback and context before registering so they're ready for use.
        spawn_tool.set_callback(spawn_cb).await;
        spawn_tool.set_context(channel, chat_id).await;
        tools.register(Box::new(SpawnToolProxy(spawn_tool)));

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
    ) -> Option<OutboundMessage> {
        let streaming = text_delta_tx.is_some();

        // Snapshot core — instant Arc clone under brief read lock.
        let core = self.core_handle.swappable();
        let counters = &self.core_handle.counters;
        let turn_count = counters.learning_turn_counter.fetch_add(1, Ordering::Relaxed) + 1;
        if turn_count % 50 == 0 {
            core.learning.prune();
        }

        let session_key = msg
            .metadata
            .get("session_key")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("{}:{}", msg.channel, msg.chat_id));

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
        let history = core.sessions.get_history(
                &session_key,
                history_limit(core.token_budget.max_context()),
                core.max_history_turns,
            ).await;
        // Track where new (unsaved) messages start. Updated after compaction
        // swaps to avoid re-persisting already-saved messages.
        let mut new_start = 1 + history.len();

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
        let mut messages = core.context.build_messages(
            &history,
            &msg.content,
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
            let mut wm = core.working_memory.get_context(&session_key, core.working_memory_budget);
            // Append learning context (tool patterns) if available.
            let learning_ctx = core.learning.get_learning_context();
            if !learning_ctx.is_empty() {
                wm.push_str("\n\n## Tool Patterns\n\n");
                wm.push_str(&learning_ctx);
            }
            if !wm.is_empty() {
                if let Some(system_content) = messages.first().and_then(|m| m["content"].as_str()).map(|s| s.to_string()) {
                    let enriched = format!(
                        "{}\n\n---\n\n# Working Memory (Current Session)\n\n{}",
                        system_content, wm
                    );
                    messages[0]["content"] = Value::String(enriched);
                }
            }
        }

        // Tag the current user message (last in the array) with turn number
        // for age-based eviction in trim_to_fit.
        if let Some(last) = messages.last_mut() {
            last["_turn"] = json!(turn_count);
        }

        // Track which tools have been used for smart tool selection.
        let mut used_tools: std::collections::HashSet<String> = std::collections::HashSet::new();

        let mut final_content = String::new();

        // Collect tool call details for turn audit summary.
        let mut turn_tool_entries: Vec<crate::agent::audit::TurnToolEntry> = Vec::new();

        // Background compaction state.
        let compaction_slot: Arc<tokio::sync::Mutex<Option<PendingCompaction>>> =
            Arc::new(tokio::sync::Mutex::new(None));
        let compaction_in_flight = Arc::new(AtomicBool::new(false));

        // Response boundary: after exec/write_file, force a text response.
        let mut force_response = false;

        // Agent loop: call LLM, handle tool calls, repeat.
        for iteration in 0..core.max_iterations {
            debug!("Agent iteration{} {}/{}", if streaming { " (streaming)" } else { "" }, iteration + 1, core.max_iterations);

            // Check if background compaction finished — swap in compacted messages.
            if let Ok(mut guard) = compaction_slot.try_lock() {
                if let Some(pending) = guard.take() {
                    debug!(
                        "Compaction swap: {} msgs -> {} compacted + {} new",
                        pending.watermark,
                        pending.result.messages.len(),
                        messages.len().saturating_sub(pending.watermark)
                    );
                    apply_compaction_result(&mut messages, pending);
                    // After compaction, all messages in the array are "new" from
                    // the perspective of persistence (the session file was rebuilt).
                    new_start = messages.len();
                }
            }

            // Response boundary: suppress exec/write_file tools to force text output.
            let boundary_active = force_response
                && core.provenance_config.enabled
                && core.provenance_config.response_boundary;
            if boundary_active {
                // Use "user" role, not "system". The Anthropic OpenAI-compat
                // endpoint strips mid-conversation system messages, which would
                // leave the conversation ending with an assistant message and
                // trigger a "does not support assistant message prefill" error.
                messages.push(json!({
                    "role": "user",
                    "content": "[system] You just executed a tool that modifies files or runs commands. \
                                Report the result to the user before making additional tool calls."
                }));
                force_response = false;
            }

            // Filter tool definitions to relevant tools.
            let mut tool_defs = tools.get_relevant_definitions(&messages, &used_tools);
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
            let tool_def_tokens = TokenBudget::estimate_tool_def_tokens(tool_defs_opt.unwrap_or(&[]));
            messages = core.token_budget.trim_to_fit_with_age(
                &messages, tool_def_tokens, turn_count, core.max_message_age_turns,
            );

            // Spawn background compaction when threshold exceeded.
            if !compaction_in_flight.load(Ordering::Relaxed)
                && core.compactor.needs_compaction(&messages, &core.token_budget, tool_def_tokens)
            {
                let slot = compaction_slot.clone();
                let in_flight = compaction_in_flight.clone();
                let bg_messages = messages.clone();
                let bg_core = core.clone();
                let bg_session_key = session_key.clone();
                let bg_channel = msg.channel.clone();
                let watermark = messages.len();
                in_flight.store(true, Ordering::SeqCst);

                tokio::spawn(async move {
                    let result = bg_core
                        .compactor
                        .compact(&bg_messages, &bg_core.token_budget, 0)
                        .await;
                    if bg_core.memory_enabled {
                        if let Some(ref summary) = result.observation {
                            bg_core.working_memory.update_from_compaction(&bg_session_key, summary);
                        }
                    }
                    if result.messages.len() < bg_messages.len() {
                        *slot.lock().await = Some(PendingCompaction { result, watermark });
                    }
                    in_flight.store(false, Ordering::SeqCst);
                });
            }

            // Repair any protocol violations before calling the LLM.
            thread_repair::repair_messages(&mut messages);

            // Call LLM — streaming or blocking depending on mode.
            let response = if let Some(ref delta_tx) = text_delta_tx {
                // Streaming path: forward text deltas as they arrive.
                let mut stream = match core
                    .provider
                    .chat_stream(
                        &messages,
                        tool_defs_opt,
                        Some(&core.model),
                        core.max_tokens,
                        core.temperature,
                    )
                    .await
                {
                    Ok(s) => s,
                    Err(e) => {
                        error!("LLM streaming call failed: {}", e);
                        final_content = format!("I encountered an error: {}", e);
                        break;
                    }
                };

                let mut streamed_response = None;
                while let Some(chunk) = stream.rx.recv().await {
                    match chunk {
                        StreamChunk::TextDelta(delta) => {
                            let _ = delta_tx.send(delta);
                        }
                        StreamChunk::Done(resp) => {
                            streamed_response = Some(resp);
                        }
                    }
                }

                match streamed_response {
                    Some(r) => r,
                    None => {
                        error!("LLM stream ended without Done");
                        final_content = "I encountered a streaming error.".to_string();
                        break;
                    }
                }
            } else {
                // Blocking path: single request/response.
                match core
                    .provider
                    .chat(
                        &messages,
                        tool_defs_opt,
                        Some(&core.model),
                        core.max_tokens,
                        core.temperature,
                    )
                    .await
                {
                    Ok(r) => r,
                    Err(e) => {
                        error!("LLM call failed: {}", e);
                        final_content = format!("I encountered an error: {}", e);
                        break;
                    }
                }
            };

            if response.has_tool_calls() {
                // Auto-escalation: when context pressure is high, delegate
                // even if tool_delegation isn't explicitly enabled. This
                // uses the main provider with slim results to save context.
                let context_tokens = TokenBudget::estimate_tokens(&messages);
                let max_tokens = core.token_budget.max_context();
                let pressure = if max_tokens > 0 {
                    context_tokens as f64 / max_tokens as f64
                } else {
                    0.0
                };
                let auto_escalate = !core.tool_delegation_config.enabled
                    && pressure > 0.7
                    && core.tool_runner_provider.is_none();

                if auto_escalate {
                    info!(
                        "Context pressure {:.0}% — auto-delegating {} tool calls",
                        pressure * 100.0,
                        response.tool_calls.len()
                    );
                }

                // Check if we should delegate to the tool runner.
                // Skip delegation if the provider was previously marked dead.
                let mut delegation_alive = counters.delegation_healthy.load(Ordering::Relaxed);
                // Periodically re-probe: every 10 inline calls, try delegation
                // once in case the server recovered (e.g. user restarted it).
                if !delegation_alive && core.tool_delegation_config.enabled {
                    let retries = counters.delegation_retry_counter.fetch_add(1, Ordering::Relaxed);
                    if retries > 0 && retries % 10 == 0 {
                        info!("Re-probing delegation provider (attempt {} since failure)", retries);
                        delegation_alive = true; // try this one time
                    } else {
                        debug!("Delegation provider unhealthy — inline execution ({}/10 until re-probe)", retries % 10);
                    }
                }
                let should_delegate = (core.tool_delegation_config.enabled || auto_escalate)
                    && delegation_alive;
                // Resolve provider+model: explicit config, or fall back to main provider.
                let delegation_provider = core.tool_runner_provider.clone()
                    .or_else(|| if auto_escalate { Some(core.provider.clone()) } else { None });
                let delegation_model = core.tool_runner_model.clone()
                    .or_else(|| if auto_escalate { Some(core.model.clone()) } else { None });

                if should_delegate {
                    if let (Some(ref tr_provider), Some(ref tr_model)) =
                        (&delegation_provider, &delegation_model)
                    {
                        debug!(
                            "Delegating {} tool calls to tool runner (model: {})",
                            response.tool_calls.len(),
                            tr_model
                        );

                        // Auto-escalation uses the main provider which may be
                        // a local llama-server that requires user-last messages.
                        // Explicit delegation providers (Ministral) handle
                        // tool→generate natively and don't need it.
                        let needs_user_cont = auto_escalate && core.is_local;

                        // The delegation model typically has a small context (4K tokens).
                        // Cap tool results to ~1500 tokens (~6000 chars) so the system
                        // prompt, tool call messages, and response all fit comfortably.
                        // Use the main model's limit only if it's already smaller.
                        let delegation_result_limit = core.max_tool_result_chars
                            .min(6000);

                        let runner_config = ToolRunnerConfig {
                            provider: tr_provider.clone(),
                            model: tr_model.clone(),
                            max_iterations: core.tool_delegation_config.max_iterations,
                            max_tokens: core.tool_delegation_config.max_tokens,
                            needs_user_continuation: needs_user_cont,
                            max_tool_result_chars: delegation_result_limit,
                            short_circuit_chars: 200,
                            depth: 0,
                            cancellation_token: cancellation_token.clone(),
                        };

                        // Emit tool call start events for delegated calls.
                        if let Some(ref tx) = tool_event_tx {
                            for tc in &response.tool_calls {
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
                        let tool_names: Vec<&str> = response.tool_calls
                            .iter()
                            .map(|tc| tc.name.as_str())
                            .collect();
                        let instructions = response.content.as_deref()
                            .filter(|c| !c.trim().is_empty())
                            .map(|c| c.chars().take(400).collect::<String>())
                            .unwrap_or_else(|| {
                                msg.content.chars().take(300).collect::<String>()
                            });
                        let task_desc = format!(
                            "Instructions: {}\nTools to execute: {}",
                            instructions,
                            tool_names.join(", ")
                        );

                        let run_result = tool_runner::run_tool_loop(
                            &runner_config,
                            &response.tool_calls,
                            &tools,
                            &task_desc,
                        )
                        .await;

                        // If the runner returned no summary, the delegation
                        // LLM likely failed (server down). Mark it unhealthy
                        // so future calls go inline without wasting time.
                        if run_result.summary.is_none() && !run_result.tool_results.is_empty() {
                            warn!(
                                "Delegation model returned no summary — marking provider unhealthy. \
                                 Tool results will flow to main model directly. \
                                 Restart servers or toggle /local to recover."
                            );
                            counters.delegation_healthy.store(false, Ordering::Relaxed);
                        } else if !counters.delegation_healthy.load(Ordering::Relaxed) {
                            // Re-probe succeeded — server recovered!
                            info!("Delegation provider recovered — re-enabling delegation");
                            counters.delegation_healthy.store(true, Ordering::Relaxed);
                            counters.delegation_retry_counter.store(0, Ordering::Relaxed);
                        }

                        debug!(
                            "Tool runner completed: {} results in {} iterations",
                            run_result.tool_results.len(),
                            run_result.iterations_used
                        );

                        // Build the assistant message with original tool_calls.
                        let tc_json: Vec<Value> = response
                            .tool_calls
                            .iter()
                            .map(|tc| {
                                json!({
                                    "id": tc.id,
                                    "type": "function",
                                    "function": {
                                        "name": tc.name,
                                        "arguments": serde_json::to_string(&tc.arguments)
                                            .unwrap_or_else(|_| "{}".to_string()),
                                    }
                                })
                            })
                            .collect();
                        ContextBuilder::add_assistant_message(
                            &mut messages,
                            response.content.as_deref(),
                            Some(&tc_json),
                        );

                        // Add tool results from the runner to the main context.
                        // In slim mode, truncate results to save context budget —
                        // the runner's summary carries the meaning.
                        let slim = core.tool_delegation_config.enabled
                            && core.tool_delegation_config.slim_results;
                        let preview_max = core.tool_delegation_config.max_result_preview_chars;

                        for tc in &response.tool_calls {
                            // Find the matching result from the runner.
                            let full_data = run_result
                                .tool_results
                                .iter()
                                .find(|(id, _, _)| id == &tc.id)
                                .map(|(_, _, data)| data.as_str())
                                .unwrap_or("(no result)");

                            let injected = if slim && full_data.len() > preview_max {
                                format!(
                                    "{}… ({} chars total)",
                                    &full_data[..preview_max],
                                    full_data.len()
                                )
                            } else {
                                full_data.to_string()
                            };

                            if core.provenance_config.enabled {
                                ContextBuilder::add_tool_result_immutable(
                                    &mut messages,
                                    &tc.id,
                                    &tc.name,
                                    &injected,
                                );
                            } else {
                                ContextBuilder::add_tool_result(
                                    &mut messages,
                                    &tc.id,
                                    &tc.name,
                                    &injected,
                                );
                            }
                            used_tools.insert(tc.name.clone());
                        }

                        // Inject the runner's summary so the main LLM knows what
                        // the tools found without needing full output.
                        let has_extra =
                            run_result.tool_results.len() > response.tool_calls.len();
                        if run_result.summary.is_some() || has_extra {
                            let summary_text = if has_extra {
                                let extra = tool_runner::format_results_for_context(
                                    &run_result,
                                    preview_max,
                                );
                                format!(
                                    "[Tool runner executed {} additional calls]\n{}",
                                    run_result.tool_results.len()
                                        - response.tool_calls.len(),
                                    extra
                                )
                            } else {
                                run_result
                                    .summary
                                    .clone()
                                    .unwrap_or_default()
                            };
                            if !summary_text.is_empty() {
                                // Inject as user role, not assistant. This is
                                // injected context about what the tool runner did,
                                // not an actual LLM response. Using assistant role
                                // causes "assistant message prefill" errors when
                                // the Anthropic API sees the conversation ending
                                // with an assistant message.
                                messages.push(json!({
                                    "role": "user",
                                    "content": format!("[tool runner summary] {}", summary_text)
                                }));
                            }
                        }

                        // Record learning + audit for all tool results.
                        let executor = format!("tool_runner:{}", tr_model);
                        for (tool_call_id, tool_name, data) in &run_result.tool_results {
                            let ok = !data.starts_with("Error:");

                            // Emit tool call end event.
                            if let Some(ref tx) = tool_event_tx {
                                let _ = tx.send(ToolEvent::CallEnd {
                                    tool_name: tool_name.clone(),
                                    tool_call_id: tool_call_id.clone(),
                                    result_data: data.clone(),
                                    ok,
                                    duration_ms: 0,
                                });
                            }

                            // Record in audit log.
                            if let Some(ref audit) = audit {
                                let _ = audit.record(
                                    tool_name,
                                    tool_call_id,
                                    &json!({}), // args not available from runner results
                                    data,
                                    ok,
                                    0, // duration not tracked per-tool in runner
                                    &executor,
                                );
                            }

                            core.learning.record_extended(
                                tool_name,
                                ok,
                                &data.chars().take(100).collect::<String>(),
                                if ok { None } else { Some(data) },
                                Some(tr_model),
                                Some(tr_model),
                                None,
                            );
                            used_tools.insert(tool_name.clone());
                        }

                        // Set response boundary flag if any delegated tool was exec/write_file.
                        for (_, tool_name, _) in &run_result.tool_results {
                            if tool_name == "exec" || tool_name == "write_file" {
                                force_response = true;
                                break;
                            }
                        }

                        // Mistral/Ministral templates handle tool→generate
                        // natively. Do NOT add user continuation messages —
                        // they break the template's role alternation check.

                        // Continue the main loop — the main LLM will see the results.
                        continue;
                    }
                }

                // Inline path (default, unchanged): execute tools directly.
                let tc_json: Vec<Value> = response
                    .tool_calls
                    .iter()
                    .map(|tc| {
                        json!({
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": serde_json::to_string(&tc.arguments)
                                    .unwrap_or_else(|_| "{}".to_string()),
                            }
                        })
                    })
                    .collect();

                ContextBuilder::add_assistant_message(
                    &mut messages,
                    response.content.as_deref(),
                    Some(&tc_json),
                );

                // Execute each tool call.
                for tc in &response.tool_calls {
                    debug!("Executing tool: {} (id: {})", tc.name, tc.id);

                    // Emit tool call start event.
                    if let Some(ref tx) = tool_event_tx {
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
                    let result = if let Some(ref tx) = tool_event_tx {
                        use crate::agent::tools::base::ToolExecutionContext;
                        let ctx = ToolExecutionContext {
                            event_tx: tx.clone(),
                            cancellation_token: cancellation_token.as_ref()
                                .map(|t| t.child_token())
                                .unwrap_or_else(tokio_util::sync::CancellationToken::new),
                            tool_call_id: tc.id.clone(),
                        };
                        tools.execute_with_context(&tc.name, tc.arguments.clone(), &ctx).await
                    } else {
                        tools.execute(&tc.name, tc.arguments.clone()).await
                    };
                    let duration_ms = start.elapsed().as_millis() as u64;
                    debug!(
                        "Tool {} result ({}B, ok={}, {}ms)",
                        tc.name,
                        result.data.len(),
                        result.ok,
                        duration_ms
                    );
                    // Cap tool result size to avoid blowing context (UTF-8 safe).
                    let data = if result.data.len() > core.max_tool_result_chars {
                        let safe_end = result.data.char_indices()
                            .take_while(|(i, _)| *i < core.max_tool_result_chars)
                            .last()
                            .map(|(i, c)| i + c.len_utf8())
                            .unwrap_or(0);
                        format!(
                            "{}\n\n[truncated: {} total chars, showing first {}]",
                            &result.data[..safe_end], result.data.len(), safe_end
                        )
                    } else {
                        result.data.clone()
                    };
                    if core.provenance_config.enabled {
                        ContextBuilder::add_tool_result_immutable(&mut messages, &tc.id, &tc.name, &data);
                    } else {
                        ContextBuilder::add_tool_result(&mut messages, &tc.id, &tc.name, &data);
                    }

                    // Emit tool call end event.
                    if let Some(ref tx) = tool_event_tx {
                        let _ = tx.send(ToolEvent::CallEnd {
                            tool_name: tc.name.clone(),
                            tool_call_id: tc.id.clone(),
                            result_data: result.data.clone(),
                            ok: result.ok,
                            duration_ms,
                        });
                    }

                    // Record in audit log.
                    if let Some(ref audit) = audit {
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
                    used_tools.insert(tc.name.clone());

                    // Collect for turn audit summary.
                    turn_tool_entries.push(crate::agent::audit::TurnToolEntry {
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
                    core.learning.record(
                        &tc.name,
                        result.ok,
                        &context_str,
                        result.error.as_deref(),
                    );

                    // Set response boundary flag for exec/write_file.
                    if tc.name == "exec" || tc.name == "write_file" {
                        force_response = true;
                    }
                }

                // Mistral/Ministral templates handle tool→generate
                // natively. Do NOT add user continuation messages —
                // they break the template's role alternation check.

                // Check cancellation between tool call iterations.
                if cancellation_token.as_ref().map_or(false, |t| t.is_cancelled()) {
                    break;
                }
            } else {
                // No tool calls -- the agent is done.
                final_content = response.content.unwrap_or_default();
                break;
            }
        }

        // Store context stats for status bar.
        let final_tokens = TokenBudget::estimate_tokens(&messages) as u64;
        counters.last_context_used
            .store(final_tokens, Ordering::Relaxed);
        counters.last_context_max
            .store(core.token_budget.max_context() as u64, Ordering::Relaxed);
        counters.last_message_count
            .store(messages.len() as u64, Ordering::Relaxed);
        // Store working memory token count.
        let wm_tokens = if core.memory_enabled {
            let wm_text = core.working_memory.get_context(&session_key, usize::MAX);
            TokenBudget::estimate_str_tokens(&wm_text) as u64
        } else {
            0
        };
        counters.last_working_memory_tokens.store(wm_tokens, Ordering::Relaxed);
        // Store tools called this turn.
        {
            let tools_list: Vec<String> = used_tools.iter().cloned().collect();
            if let Ok(mut guard) = counters.last_tools_called.lock() {
                *guard = tools_list;
            }
        }

        // Write per-turn audit summary.
        if core.provenance_config.enabled && core.provenance_config.audit_log {
            let summary = crate::agent::audit::TurnSummary {
                turn: turn_count,
                timestamp: Utc::now().to_rfc3339(),
                context_tokens: final_tokens as usize,
                message_count: messages.len(),
                tools_called: turn_tool_entries,
                working_memory_tokens: wm_tokens as usize,
            };
            crate::agent::audit::write_turn_summary(&core.workspace, &session_key, &summary);
        }

        if final_content.is_empty() && messages.len() > 2 {
            final_content = "I ran out of tool iterations before producing a final answer. The actions above may be incomplete.".to_string();
        }

        // Phase 3+4: Claim verification and context hygiene.
        if !final_content.is_empty()
            && core.provenance_config.enabled
            && core.provenance_config.verify_claims
        {
            if let Some(ref audit) = audit {
                let entries = audit.get_entries();
                let (claims, has_fabrication) =
                    crate::agent::provenance::verify_turn_claims(&final_content, &entries);

                if has_fabrication && core.provenance_config.strict_mode {
                    let (redacted, redaction_count) =
                        crate::agent::provenance::redact_fabrications(&final_content, &claims);
                    final_content = redacted;
                    if redaction_count > 0 {
                        messages.push(json!({
                            "role": "system",
                            "content": format!(
                                "NOTICE: {} claim(s) in the previous response could not be \
                                 verified against tool outputs and were removed.",
                                redaction_count
                            )
                        }));
                    }
                }
            }
        }

        // Phantom tool call detection: check if LLM claims tool results without calling tools.
        if !final_content.is_empty() && core.provenance_config.enabled {
            let tools_list: Vec<String> = used_tools.iter().cloned().collect();
            if let Some(detection) = crate::agent::provenance::detect_phantom_claims(&final_content, &tools_list) {
                warn!(
                    "Phantom tool claims detected ({} patterns): {:?}",
                    detection.matched_patterns.len(),
                    detection.matched_patterns
                );

                // Hard block: annotate the response so the user sees the warning.
                if core.provenance_config.strict_mode {
                    final_content = crate::agent::provenance::annotate_phantom_response(
                        &final_content, &detection,
                    );
                }

                // Inject system reminder for the next turn.
                messages.push(json!({
                    "role": "system",
                    "content": detection.system_warning
                }));
            }
        }

        // Ensure the final text response is in the messages array for persistence.
        // Without this, text-only responses (no tool calls) would be lost.
        if !final_content.is_empty() {
            messages.push(json!({"role": "assistant", "content": final_content.clone()}));
        }

        // Update session history — persist full message array including tool calls.
        // Skip system prompt (index 0) and pre-existing history.
        let new_messages: Vec<Value> = if new_start < messages.len() {
            messages[new_start..].to_vec()
        } else {
            // Fallback: save at least user + assistant text.
            let mut fallback = vec![json!({"role": "user", "content": msg.content.clone()})];
            if !final_content.is_empty() {
                fallback.push(json!({"role": "assistant", "content": final_content.clone()}));
            }
            fallback
        };
        if !new_messages.is_empty() {
            core.sessions.add_messages_raw(&session_key, &new_messages).await;
        }

        // Auto-complete stale working memory sessions (runs on every message, cheap).
        if core.memory_enabled {
            for session in core.working_memory.list_active() {
                if session.session_key != session_key {
                    let age = Utc::now() - session.updated;
                    if age.num_seconds() > core.session_complete_after_secs as i64 {
                        core.working_memory.complete(&session.session_key);
                        debug!("Auto-completed stale session: {}", session.session_key);
                    }
                }
            }
        }

        if final_content.is_empty() {
            None
        } else {
            let mut outbound = OutboundMessage::new(&msg.channel, &msg.chat_id, &final_content);
            // Propagate voice_message metadata so channels know to reply with voice.
            if msg
                .metadata
                .get("voice_message")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
            {
                outbound
                    .metadata
                    .insert("voice_message".to_string(), json!(true));
            }
            // Propagate detected_language for TTS voice selection.
            if let Some(lang) = msg.metadata.get("detected_language") {
                outbound
                    .metadata
                    .insert("detected_language".to_string(), lang.clone());
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
        );
        if let Some(ref dtx) = repl_display_tx {
            subagent_mgr = subagent_mgr.with_display_tx(dtx.clone());
        }
        let subagents = Arc::new(subagent_mgr);

        let shared = Arc::new(AgentLoopShared {
            core_handle,
            subagents,
            bus_outbound_tx,
            bus_inbound_tx,
            cron_service,
            email_config,
            repl_display_tx,
        });

        Self {
            shared,
            bus_inbound_rx,
            running: Arc::new(AtomicBool::new(false)),
            max_concurrent_chats,
            reflection_spawned: AtomicBool::new(false),
        }
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
                        format!("{}...", &msg.content[..120])
                    } else {
                        msg.content.clone()
                    };
                    let _ = dtx.send(format!(
                        "\x1b[2m[{}]\x1b[0m \x1b[36m{}\x1b[0m: {}",
                        msg.channel, msg.sender_id, preview
                    ));
                }

                let response = shared.process_message(&msg, None, None, None).await;

                if let Some(outbound) = response {
                    // Notify REPL about outbound response.
                    if let Some(ref dtx) = display_tx {
                        let preview = if outbound.content.len() > 120 {
                            format!("{}...", &outbound.content[..120])
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

        match self.shared.process_message(&msg, None, None, None).await {
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
            .process_message(&msg, Some(text_delta_tx), tool_event_tx, cancellation_token)
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
    use async_trait::async_trait;

    /// Minimal mock LLM provider for wiring tests.
    struct MockLLM {
        name: String,
    }

    impl MockLLM {
        fn named(name: &str) -> Arc<dyn LLMProvider> {
            Arc::new(Self { name: name.to_string() })
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
        build_swappable_core(
            main,
            workspace,
            "main-model".to_string(),
            10,
            4096,
            0.7,
            16384,
            None,
            30,
            false,
            MemoryConfig::default(),
            false,
            None,
            td,
            ProvenanceConfig::default(),
            2000,
            delegation_provider,
        )
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
        let core = build_swappable_core(
            main,
            workspace,
            "main-model".to_string(),
            10,
            4096,
            0.7,
            16384,
            None,
            30,
            false,
            MemoryConfig::default(),
            false,
            None,
            td,
            ProvenanceConfig::default(),
            2000,
            None,
        );
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
        let core = build_swappable_core(
            main,
            workspace,
            "local-model".to_string(),
            10,
            4096,
            0.7,
            16384,
            None,
            30,
            false,
            MemoryConfig::default(),
            true, // is_local = true
            None,
            td,
            ProvenanceConfig::default(),
            2000,
            Some(dp),
        );

        assert!(core.is_local);
        assert!(core.tool_runner_provider.is_some());
        assert_eq!(
            core.tool_runner_provider.as_ref().unwrap().get_default_model(),
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
        let core = build_swappable_core(
            main,
            workspace,
            "main-model".to_string(),
            10,
            4096,
            0.7,
            16384,
            None,
            30,
            false,
            MemoryConfig::default(),
            true,
            Some(compaction),
            td,
            ProvenanceConfig::default(),
            2000,
            Some(delegation),
        );

        // Compaction provider goes to memory_provider, delegation to tool_runner
        assert_eq!(
            core.memory_provider.get_default_model(),
            "compaction",
            "Memory should use compaction provider"
        );
        assert_eq!(
            core.tool_runner_provider.as_ref().unwrap().get_default_model(),
            "delegation",
            "Tool runner should use delegation provider"
        );
    }
}
