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

use crate::agent::compaction::ContextCompactor;
use crate::agent::environment::EnvironmentScanner;
use crate::agent::context::ContextBuilder;
use crate::agent::learning::LearningStore;
use crate::agent::observer::ObservationStore;
use crate::agent::reflector::Reflector;
use crate::agent::semantic::SemanticIndex;
use crate::agent::subagent::SubagentManager;
use crate::agent::thread_repair;
use crate::agent::token_budget::TokenBudget;
use crate::agent::scratchpad::SharedScratchpad;
use crate::agent::taskboard::TaskBoard;
use crate::agent::tools::{
    CheckInboxTool, CronScheduleTool, EditFileTool, ExecTool, ListDirTool, MessageTool,
    ReadFileTool, ScratchpadTool, SendCallback, SendEmailTool, SkillManagerTool, SpawnCallback,
    SpawnTool, TaskBoardTool, ToolRegistry, WebFetchTool, WebSearchTool, WriteFileTool,
};
use crate::bus::events::{InboundMessage, OutboundMessage};
use crate::config::schema::{EmailConfig, MemoryConfig};
use crate::cron::service::CronService;
use crate::providers::base::{LLMProvider, StreamChunk};
use crate::providers::openai_compat::OpenAICompatProvider;
use crate::session::manager::SessionManager;
use crate::store::conversation::ConversationStore;

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
    pub conversation_store: Arc<ConversationStore>,
    pub semantic_index: Arc<SemanticIndex>,
    pub semantic_budget: usize,
    pub token_budget: TokenBudget,
    pub compactor: ContextCompactor,
    pub learning: LearningStore,
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
) -> SharedCore {
    let mut context = ContextBuilder::new(&workspace);
    context.model_name = model.clone();
    context.observation_budget = memory_config.observation_budget;
    context.semantic_budget = memory_config.semantic_budget;

    // Environment scanning (cached, fast on subsequent starts).
    let env_scanner = EnvironmentScanner::new(&workspace);
    let env_caps = env_scanner.get_capabilities();
    context.environment_summary = Some(EnvironmentScanner::format_for_context(&env_caps));

    // Time awareness: find the most recent session file modification time.
    context.last_interaction = detect_last_interaction(&workspace);

    // Git change detection: check for workspace changes since last session.
    context.git_changes = detect_git_changes(&workspace);

    // SLM budget mode: reduce all context budgets for local models.
    if is_local {
        context.bootstrap_budget = 1000;
        context.long_term_memory_budget = 500;
        context.today_notes_budget = 0;
        context.observation_budget = 0;
        context.learning_budget = 0;
        context.semantic_budget = 500;
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
    let compactor = ContextCompactor::new(memory_provider.clone(), memory_model.clone());
    let learning = LearningStore::new(&workspace);

    // SQLite conversation store (parallel write path alongside JSONL sessions).
    let store_db_path = crate::config::loader::get_data_dir().join("store.db");
    let conversation_store = match ConversationStore::new(&store_db_path) {
        Ok(s) => Arc::new(s),
        Err(e) => {
            warn!("Failed to open conversation store at {:?}: {}. Using in-memory fallback.", store_db_path, e);
            Arc::new(ConversationStore::new(std::path::Path::new(":memory:")).expect("in-memory SQLite must work"))
        }
    };

    // Semantic retrieval index (BM25 over turns + memory files).
    let semantic_index = Arc::new(SemanticIndex::new(&workspace, conversation_store.clone()));
    context.semantic_index = Some(semantic_index.clone());

    SharedCore {
        provider,
        workspace,
        model,
        max_iterations,
        max_tokens,
        temperature,
        context,
        sessions,
        conversation_store,
        semantic_index,
        semantic_budget: memory_config.semantic_budget,
        token_budget,
        compactor,
        learning,
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
    }
}

/// Detect the most recent interaction time by scanning session files.
fn detect_last_interaction(workspace: &std::path::Path) -> Option<chrono::DateTime<chrono::Local>> {
    let sessions_dir = workspace.join("sessions");
    if !sessions_dir.is_dir() {
        return None;
    }
    let mut latest: Option<std::time::SystemTime> = None;
    if let Ok(entries) = std::fs::read_dir(&sessions_dir) {
        for entry in entries.flatten() {
            if let Ok(meta) = entry.metadata() {
                if let Ok(mtime) = meta.modified() {
                    latest = Some(match latest {
                        Some(prev) if mtime > prev => mtime,
                        Some(prev) => prev,
                        None => mtime,
                    });
                }
            }
        }
    }
    latest.map(|t| chrono::DateTime::<chrono::Local>::from(t))
}

/// Detect git changes in the workspace since last session.
fn detect_git_changes(workspace: &std::path::Path) -> Option<String> {
    // Only run if the workspace is inside a git repo.
    let status_output = std::process::Command::new("git")
        .args(["status", "--porcelain"])
        .current_dir(workspace)
        .output()
        .ok()?;

    if !status_output.status.success() {
        return None; // Not a git repo or git not available.
    }

    let status = String::from_utf8_lossy(&status_output.stdout);
    let changed_files: Vec<&str> = status.lines().collect();

    let log_output = std::process::Command::new("git")
        .args(["log", "--oneline", "-5", "--since=12h"])
        .current_dir(workspace)
        .output()
        .ok()?;

    let log = String::from_utf8_lossy(&log_output.stdout);
    let recent_commits: Vec<&str> = log.lines().collect();

    if changed_files.is_empty() && recent_commits.is_empty() {
        return None;
    }

    let mut parts = Vec::new();
    if !changed_files.is_empty() {
        parts.push(format!(
            "{} file(s) with uncommitted changes",
            changed_files.len()
        ));
    }
    if !recent_commits.is_empty() {
        parts.push(format!(
            "Recent commits (last 12h):\n{}",
            recent_commits.join("\n")
        ));
    }
    Some(parts.join("\n"))
}

// ---------------------------------------------------------------------------
// Per-instance state (different per agent)
// ---------------------------------------------------------------------------

/// Per-instance state that differs between the REPL agent and gateway agents.
struct AgentLoopShared {
    core_handle: SharedCoreHandle,
    subagents: Arc<SubagentManager>,
    task_board: Arc<TaskBoard>,
    scratchpad: Arc<SharedScratchpad>,
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
        tools.condensed = core.is_local;

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

        // Skill manager (dynamic skill creation).
        tools.register(Box::new(SkillManagerTool::new(core.workspace.clone())));

        // Task board and scratchpad (multi-agent coordination).
        tools.register(Box::new(TaskBoardTool::new(self.task_board.clone())));
        tools.register(Box::new(ScratchpadTool::new(self.scratchpad.clone())));

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
        let spawn_cb: SpawnCallback = Arc::new(move |task, label, ch, cid, task_id| {
            let mgr = subagents_ref.clone();
            Box::pin(async move { mgr.spawn(task, label, ch, cid, task_id).await })
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

    /// Process a regular inbound message through the agent loop.
    ///
    /// This method takes `&self` and is safe to call from multiple concurrent
    /// tasks. Per-message tool instances eliminate shared-context races.
    async fn process_message(&self, msg: &InboundMessage) -> Option<OutboundMessage> {
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
            "Processing message from {} on {}: {}",
            msg.sender_id,
            msg.channel,
            &msg.content[..msg.content.len().min(80)]
        );

        // Build per-message tools with context baked in.
        let tools = self.build_tools(&core, &msg.channel, &msg.chat_id).await;

        // Get session history.
        let history = core.sessions.get_history(&session_key, 100).await;

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

        // Inject task board summary into system prompt.
        {
            let board_summary = self.task_board.summary();
            if !board_summary.is_empty() {
                if let Some(sys_msg) = messages.first_mut() {
                    if let Some(content) = sys_msg.get("content").and_then(|c| c.as_str()).map(|s| s.to_string()) {
                        sys_msg["content"] = json!(format!("{}\n\n---\n\n# Task Board\n\n{}", content, board_summary));
                    }
                }
            }
        }

        // Track which tools have been used for smart tool selection.
        let mut used_tools: std::collections::HashSet<String> = std::collections::HashSet::new();

        // Pre-loop compaction: summarize old messages if over budget.
        // When a dedicated compaction server is available (local mode with Qwen3-0.6B),
        // this works great. When it's not, the ContextCompactor's Stage 1 failure is
        // handled gracefully (falls back to trim_to_fit in the loop below).
        {
            let initial_tool_defs = tools.get_relevant_definitions(&messages, &used_tools);
            let initial_tool_tokens = core
                .token_budget
                .estimate_tool_def_tokens_with_fallback(&initial_tool_defs);
            let compaction_result = core
                .compactor
                .compact(&messages, &core.token_budget, initial_tool_tokens)
                .await;
            messages = compaction_result.messages;

            // Persist observation in background — never blocks user chat.
            if core.memory_enabled {
                if let Some(summary) = compaction_result.observation {
                    let workspace = core.workspace.clone();
                    let obs_session_key = session_key.clone();
                    let obs_channel = msg.channel.clone();
                    let sem_index = core.semantic_index.clone();
                    tokio::spawn(async move {
                        let observer = ObservationStore::new(&workspace);
                        match observer.save(&summary, &obs_session_key, Some(&obs_channel)) {
                            Ok(path) => {
                                sem_index.add_observation(&path, &summary);
                            }
                            Err(e) => {
                                tracing::warn!("Failed to save observation: {}", e);
                            }
                        }
                    });
                }
            }
        }

        let mut final_content = String::new();

        // Agent loop: call LLM, handle tool calls, repeat.
        for iteration in 0..core.max_iterations {
            debug!("Agent iteration {}/{}", iteration + 1, core.max_iterations);

            // Filter tool definitions to relevant tools.
            let tool_defs = tools.get_relevant_definitions(&messages, &used_tools);
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

            // Repair any protocol violations before calling the LLM.
            thread_repair::repair_messages(&mut messages);

            let response = match core
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
            };

            if response.has_tool_calls() {
                // Build tool_calls JSON for the assistant message.
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
                    let result = tools.execute(&tc.name, tc.arguments.clone()).await;
                    debug!(
                        "Tool {} result ({}B, ok={})",
                        tc.name,
                        result.data.len(),
                        result.ok
                    );
                    ContextBuilder::add_tool_result(&mut messages, &tc.id, &tc.name, &result.data);

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

        // Write turns to SQLite conversation store (parallel path) + index.
        {
            let store = core.conversation_store.clone();
            let sem_index = core.semantic_index.clone();
            let channel = msg.channel.clone();
            let chat_id = msg.chat_id.clone();
            let user_content = msg.content.clone();
            let assistant_content = final_content.clone();
            let model_name = core.model.clone();
            let lang = detected_language.clone();
            tokio::task::spawn_blocking(move || {
                let sid = store.ensure_session(&channel, &chat_id);
                store.add_turn(&sid, "user", Some(&user_content), None, None, None, None, None, Some(&channel), lang.as_deref(), is_voice_message, None);
                if !assistant_content.is_empty() {
                    let turn_id = store.add_turn(&sid, "assistant", Some(&assistant_content), None, None, Some(&model_name), None, None, Some(&channel), lang.as_deref(), false, None);
                    sem_index.add_turn(&turn_id, &assistant_content);
                }
            });
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

    /// Process a message with streaming: text deltas are forwarded to
    /// `text_delta_tx` as they arrive from the LLM. The full response text
    /// is still returned for session history. Optionally emits `TuiEvent`s
    /// for tool execution visibility.
    async fn process_message_streaming(
        &self,
        msg: &InboundMessage,
        text_delta_tx: tokio::sync::mpsc::UnboundedSender<String>,
        tui_event_tx: Option<tokio::sync::mpsc::UnboundedSender<crate::tui::events::TuiEvent>>,
    ) -> Option<OutboundMessage> {
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
            "Processing message (streaming) from {} on {}: {}",
            msg.sender_id,
            msg.channel,
            &msg.content[..msg.content.len().min(80)]
        );

        let tools = self.build_tools(&core, &msg.channel, &msg.chat_id).await;

        let history = core.sessions.get_history(&session_key, 100).await;

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

        // Inject task board summary into system prompt.
        {
            let board_summary = self.task_board.summary();
            if !board_summary.is_empty() {
                if let Some(sys_msg) = messages.first_mut() {
                    if let Some(content) = sys_msg.get("content").and_then(|c| c.as_str()).map(|s| s.to_string()) {
                        sys_msg["content"] = json!(format!("{}\n\n---\n\n# Task Board\n\n{}", content, board_summary));
                    }
                }
            }
        }

        let mut used_tools: std::collections::HashSet<String> = std::collections::HashSet::new();

        // Pre-loop compaction
        {
            let initial_tool_defs = tools.get_relevant_definitions(&messages, &used_tools);
            let initial_tool_tokens = core
                .token_budget
                .estimate_tool_def_tokens_with_fallback(&initial_tool_defs);
            let compaction_result = core
                .compactor
                .compact(&messages, &core.token_budget, initial_tool_tokens)
                .await;
            messages = compaction_result.messages;

            if core.memory_enabled {
                if let Some(summary) = compaction_result.observation {
                    let workspace = core.workspace.clone();
                    let obs_session_key = session_key.clone();
                    let obs_channel = msg.channel.clone();
                    let sem_index = core.semantic_index.clone();
                    tokio::spawn(async move {
                        let observer = ObservationStore::new(&workspace);
                        match observer.save(&summary, &obs_session_key, Some(&obs_channel)) {
                            Ok(path) => {
                                sem_index.add_observation(&path, &summary);
                            }
                            Err(e) => {
                                tracing::warn!("Failed to save observation: {}", e);
                            }
                        }
                    });
                }
            }
        }

        let mut final_content = String::new();

        for iteration in 0..core.max_iterations {
            debug!("Agent iteration (streaming) {}/{}", iteration + 1, core.max_iterations);

            let tool_defs = tools.get_relevant_definitions(&messages, &used_tools);
            let tool_defs_opt: Option<&[Value]> = if tool_defs.is_empty() {
                None
            } else {
                Some(&tool_defs)
            };

            let tool_def_tokens = core
                .token_budget
                .estimate_tool_def_tokens_with_fallback(tool_defs_opt.unwrap_or(&[]));
            messages = core.token_budget.trim_to_fit(&messages, tool_def_tokens);
            thread_repair::repair_messages(&mut messages);

            // Emit thinking event at start of each LLM call
            if let Some(ref tx) = tui_event_tx {
                let _ = tx.send(crate::tui::events::TuiEvent::Thinking);
            }

            // Use streaming API
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

            // Consume stream chunks
            let mut response = None;
            while let Some(chunk) = stream.rx.recv().await {
                match chunk {
                    StreamChunk::TextDelta(delta) => {
                        // Forward text deltas for TTS pipeline and TUI
                        if let Some(ref tx) = tui_event_tx {
                            let _ = tx.send(crate::tui::events::TuiEvent::TextDelta(delta.clone()));
                        }
                        let _ = text_delta_tx.send(delta);
                    }
                    StreamChunk::Done(resp) => {
                        response = Some(resp);
                    }
                }
            }

            let response = match response {
                Some(r) => r,
                None => {
                    error!("LLM stream ended without Done");
                    final_content = "I encountered a streaming error.".to_string();
                    break;
                }
            };

            if response.has_tool_calls() {
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

                for tc in &response.tool_calls {
                    debug!("Executing tool: {} (id: {})", tc.name, tc.id);
                    if let Some(ref tx) = tui_event_tx {
                        let _ = tx.send(crate::tui::events::TuiEvent::ToolStart {
                            name: tc.name.clone(),
                            id: tc.id.clone(),
                        });
                    }
                    let tool_start = std::time::Instant::now();
                    let result = tools.execute(&tc.name, tc.arguments.clone()).await;
                    let tool_duration = tool_start.elapsed();
                    debug!(
                        "Tool {} result ({}B, ok={})",
                        tc.name,
                        result.data.len(),
                        result.ok
                    );
                    if let Some(ref tx) = tui_event_tx {
                        let _ = tx.send(crate::tui::events::TuiEvent::ToolComplete {
                            name: tc.name.clone(),
                            id: tc.id.clone(),
                            success: result.ok,
                            duration: tool_duration,
                        });
                    }
                    ContextBuilder::add_tool_result(&mut messages, &tc.id, &tc.name, &result.data);
                    used_tools.insert(tc.name.clone());

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
                }
            } else {
                final_content = response.content.unwrap_or_default();
                break;
            }
        }

        // Store context stats
        let final_tokens = core.token_budget.estimate_tokens_with_fallback(&messages) as u64;
        core.last_context_used
            .store(final_tokens, Ordering::Relaxed);
        core.last_context_max
            .store(core.token_budget.max_context() as u64, Ordering::Relaxed);

        if final_content.is_empty() && messages.len() > 2 {
            final_content = "I ran out of tool iterations before producing a final answer. The actions above may be incomplete.".to_string();
        }

        // Update session history
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

        // Write turns to SQLite conversation store (parallel path) + index.
        {
            let store = core.conversation_store.clone();
            let sem_index = core.semantic_index.clone();
            let channel = msg.channel.clone();
            let chat_id = msg.chat_id.clone();
            let user_content = msg.content.clone();
            let assistant_content = final_content.clone();
            let model_name = core.model.clone();
            let lang = detected_language.clone();
            tokio::task::spawn_blocking(move || {
                let sid = store.ensure_session(&channel, &chat_id);
                store.add_turn(&sid, "user", Some(&user_content), None, None, None, None, None, Some(&channel), lang.as_deref(), is_voice_message, None);
                if !assistant_content.is_empty() {
                    let turn_id = store.add_turn(&sid, "assistant", Some(&assistant_content), None, None, Some(&model_name), None, None, Some(&channel), lang.as_deref(), false, None);
                    sem_index.add_turn(&turn_id, &assistant_content);
                }
            });
        }

        // Signal completion to TUI
        if let Some(ref tx) = tui_event_tx {
            let _ = tx.send(crate::tui::events::TuiEvent::Done);
        }

        if final_content.is_empty() {
            None
        } else {
            let mut outbound = OutboundMessage::new(&msg.channel, &msg.chat_id, &final_content);
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
        let task_board = Arc::new(TaskBoard::new(&core.workspace));
        let scratchpad = Arc::new(SharedScratchpad::new(&core.workspace));

        let subagents = Arc::new(SubagentManager::new(
            core.provider.clone(),
            core.workspace.clone(),
            bus_inbound_tx.clone(),
            core.model.clone(),
            core.brave_api_key.clone(),
            core.exec_timeout,
            core.restrict_to_workspace,
            task_board.clone(),
            scratchpad.clone(),
        ));

        let shared = Arc::new(AgentLoopShared {
            core_handle,
            subagents,
            task_board,
            scratchpad,
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
            let sem_index = core.semantic_index.clone();
            tokio::spawn(async move {
                info!("Background: reflecting on accumulated observations...");
                if let Err(e) = reflector.reflect().await {
                    warn!("Background reflection failed: {}", e);
                } else {
                    info!("Background reflection complete — MEMORY.md updated");
                    sem_index.mark_dirty();
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

                let response = shared.process_message(&msg).await;

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

        // End all active sessions in the conversation store.
        let core = self.shared.core_handle.read().unwrap().clone();
        core.conversation_store.end_all_active_sessions();
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

        match self.shared.process_message(&msg).await {
            Some(response) => response.content,
            None => String::new(),
        }
    }

    /// Like `process_direct_with_lang` but streams text deltas to `text_delta_tx`
    /// as they arrive from the LLM. Returns the full response text. Optionally
    /// accepts a `tui_event_tx` for rich TUI event emission.
    pub async fn process_direct_streaming(
        &self,
        content: &str,
        session_key: &str,
        channel: &str,
        chat_id: &str,
        detected_language: Option<&str>,
        text_delta_tx: tokio::sync::mpsc::UnboundedSender<String>,
        tui_event_tx: Option<tokio::sync::mpsc::UnboundedSender<crate::tui::events::TuiEvent>>,
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
            .process_message_streaming(&msg, text_delta_tx, tui_event_tx)
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
