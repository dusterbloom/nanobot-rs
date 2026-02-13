//! Main agent loop that consumes inbound messages and produces responses.
//!
//! Ported from Python `agent/loop.py`.
//!
//! The agent loop uses a fan-out pattern for concurrent message processing:
//! messages from different sessions run in parallel (up to `max_concurrent_chats`),
//! while messages within the same session are serialized to preserve ordering.

use std::collections::HashMap;
use std::path::PathBuf;
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
    ReadFileTool, RecallTool, SendCallback, SendEmailTool, SpawnCallback, SpawnTool, ToolRegistry,
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

/// Core state shared identically across all agent instances.
///
/// When the user toggles `/local` or `/model`, a new `SharedCore` is built
/// and swapped into the handle so every agent sees the change.
pub struct SharedCore {
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
    pub learning_turn_counter: AtomicU64,
    pub last_context_used: AtomicU64,
    pub last_context_max: AtomicU64,
    pub is_local: bool,
    pub tool_runner_provider: Option<Arc<dyn LLMProvider>>,
    pub tool_runner_model: Option<String>,
    pub tool_delegation_config: ToolDelegationConfig,
    pub provenance_config: ProvenanceConfig,
}

/// Handle for hot-swapping the shared core.
///
/// Readers clone the inner `Arc<SharedCore>` under a brief read lock.
/// Writers (only `/local` toggle) take the write lock to replace the inner Arc.
pub type SharedCoreHandle = Arc<std::sync::RwLock<Arc<SharedCore>>>;

/// Build a `SharedCore` from the given parameters.
///
/// When `is_local` is true, the compactor and memory operations use a dedicated
/// `compaction_provider` if supplied (e.g. a CPU-only Qwen3-0.6B server), or
/// fall back to the main (local) provider.
pub fn build_shared_core(
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
) -> SharedCore {
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
    let compactor = ContextCompactor::new(memory_provider.clone(), memory_model.clone(), max_context_tokens);
    let learning = LearningStore::new(&workspace);
    let working_memory = WorkingMemoryStore::new(&workspace);

    // Build tool runner provider if delegation is enabled.
    let (tool_runner_provider, tool_runner_model) = if tool_delegation.enabled {
        let tr_model = if tool_delegation.model.is_empty() {
            model.clone()
        } else {
            tool_delegation.model.clone()
        };
        let tr_provider: Arc<dyn LLMProvider> =
            if let Some(ref tr_cfg) = tool_delegation.provider {
                Arc::new(OpenAICompatProvider::new(
                    &tr_cfg.api_key,
                    tr_cfg.api_base.as_deref().or(Some("http://localhost:8080/v1")),
                    None,
                ))
            } else {
                provider.clone()
            };
        (Some(tr_provider), Some(tr_model))
    } else {
        (None, None)
    };

    SharedCore {
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
        learning_turn_counter: AtomicU64::new(0),
        last_context_used: AtomicU64::new(0),
        last_context_max: AtomicU64::new(max_context_tokens as u64),
        is_local,
        tool_runner_provider,
        tool_runner_model,
        tool_delegation_config: tool_delegation,
        provenance_config: provenance,
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
    /// Takes a snapshot of `SharedCore` so the registry is consistent for the
    /// entire message processing.
    async fn build_tools(&self, core: &SharedCore, channel: &str, chat_id: &str) -> ToolRegistry {
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
        )));

        // Web (stateless config).
        tools.register(Box::new(WebSearchTool::new(core.brave_api_key.clone(), 5)));
        tools.register(Box::new(WebFetchTool::new(50_000)));

        // Memory recall (stateless config).
        tools.register(Box::new(RecallTool::new(&core.workspace)));

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
    ) -> Option<OutboundMessage> {
        let streaming = text_delta_tx.is_some();

        // Snapshot core — instant Arc clone under brief read lock.
        let core = self.core_handle.read().unwrap().clone();
        let turn_count = core.learning_turn_counter.fetch_add(1, Ordering::Relaxed) + 1;
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

        // Get session history.
        let history = core.sessions.get_history(
                &session_key,
                history_limit(core.token_budget.max_context()),
            ).await;

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
            let wm = core.working_memory.get_context(&session_key, core.working_memory_budget);
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

        // Track which tools have been used for smart tool selection.
        let mut used_tools: std::collections::HashSet<String> = std::collections::HashSet::new();

        let mut final_content = String::new();

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
                }
            }

            // Response boundary: suppress exec/write_file tools to force text output.
            let boundary_active = force_response
                && core.provenance_config.enabled
                && core.provenance_config.response_boundary;
            if boundary_active {
                messages.push(json!({
                    "role": "system",
                    "content": "You just executed a tool that modifies files or runs commands. \
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
            let tool_def_tokens = core
                .token_budget
                .estimate_tool_def_tokens_with_fallback(tool_defs_opt.unwrap_or(&[]));
            messages = core.token_budget.trim_to_fit(&messages, tool_def_tokens);

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
                // Check if we should delegate to the tool runner.
                if core.tool_delegation_config.enabled {
                    if let (Some(ref tr_provider), Some(ref tr_model)) =
                        (&core.tool_runner_provider, &core.tool_runner_model)
                    {
                        debug!(
                            "Delegating {} tool calls to tool runner (model: {})",
                            response.tool_calls.len(),
                            tr_model
                        );

                        let runner_config = ToolRunnerConfig {
                            provider: tr_provider.clone(),
                            model: tr_model.clone(),
                            max_iterations: core.tool_delegation_config.max_iterations,
                            max_tokens: core.tool_delegation_config.max_tokens,
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

                        let run_result = tool_runner::run_tool_loop(
                            &runner_config,
                            &response.tool_calls,
                            &tools,
                            &msg.content,
                        )
                        .await;

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
                        // We need to add results for the original tool calls first.
                        for tc in &response.tool_calls {
                            // Find the matching result from the runner.
                            let result_data = run_result
                                .tool_results
                                .iter()
                                .find(|(id, _, _)| id == &tc.id)
                                .map(|(_, _, data)| data.as_str())
                                .unwrap_or("(no result)");
                            if core.provenance_config.enabled {
                                ContextBuilder::add_tool_result_immutable(
                                    &mut messages,
                                    &tc.id,
                                    &tc.name,
                                    result_data,
                                );
                            } else {
                                ContextBuilder::add_tool_result(
                                    &mut messages,
                                    &tc.id,
                                    &tc.name,
                                    result_data,
                                );
                            }
                            used_tools.insert(tc.name.clone());
                        }

                        // If the runner executed additional tool calls beyond the
                        // initial ones, inject a summary as a system-like message.
                        if run_result.tool_results.len() > response.tool_calls.len() {
                            let extra_summary =
                                tool_runner::format_results_for_context(&run_result);
                            ContextBuilder::add_assistant_message(
                                &mut messages,
                                Some(&format!(
                                    "[Tool runner executed {} additional tool calls]\n{}",
                                    run_result.tool_results.len() - response.tool_calls.len(),
                                    extra_summary
                                )),
                                None,
                            );
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
                    let result = tools.execute(&tc.name, tc.arguments.clone()).await;
                    let duration_ms = start.elapsed().as_millis() as u64;
                    debug!(
                        "Tool {} result ({}B, ok={}, {}ms)",
                        tc.name,
                        result.data.len(),
                        result.ok,
                        duration_ms
                    );
                    if core.provenance_config.enabled {
                        ContextBuilder::add_tool_result_immutable(&mut messages, &tc.id, &tc.name, &result.data);
                    } else {
                        ContextBuilder::add_tool_result(&mut messages, &tc.id, &tc.name, &result.data);
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
            } else {
                // No tool calls -- the agent is done.
                final_content = response.content.unwrap_or_default();
                break;
            }
        }

        // Store context stats for status bar.
        let final_tokens = core.token_budget.estimate_tokens_with_fallback(&messages) as u64;
        core.last_context_used
            .store(final_tokens, Ordering::Relaxed);
        core.last_context_max
            .store(core.token_budget.max_context() as u64, Ordering::Relaxed);

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

        // Update session history.
        if final_content.is_empty() {
            core.sessions
                .add_message_and_save(&session_key, "user", &msg.content)
                .await;
        } else {
            core.sessions
                .add_messages_and_save(
                    &session_key,
                    &[("user", &msg.content), ("assistant", &final_content)],
                )
                .await;
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
        let core = core_handle.read().unwrap().clone();
        let subagents = Arc::new(SubagentManager::new(
            core.provider.clone(),
            core.workspace.clone(),
            bus_inbound_tx.clone(),
            core.model.clone(),
            core.brave_api_key.clone(),
            core.exec_timeout,
            core.restrict_to_workspace,
        ));

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
        let core = shared.core_handle.read().unwrap().clone();
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

                let response = shared.process_message(&msg, None, None).await;

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

        match self.shared.process_message(&msg, None, None).await {
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
            .process_message(&msg, Some(text_delta_tx), tool_event_tx)
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
