//! Main agent loop that consumes inbound messages and produces responses.
//!
//! Ported from Python `agent/loop.py`.
//!
//! The agent loop uses a fan-out pattern for concurrent message processing:
//! messages from different sessions run in parallel (up to `max_concurrent_chats`),
//! while messages within the same session are serialized to preserve ordering.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use serde_json::{json, Value};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use tokio::sync::{Mutex, Semaphore};
use tracing::{debug, error, info};

use crate::agent::compaction::ContextCompactor;
use crate::agent::context::ContextBuilder;
use crate::agent::learning::LearningStore;
use crate::agent::subagent::SubagentManager;
use crate::agent::thread_repair;
use crate::agent::token_budget::TokenBudget;
use crate::agent::tools::{
    CronScheduleTool, ExecTool, ListDirTool, MessageTool, ReadFileTool, SendCallback,
    SpawnCallback, SpawnTool, ToolRegistry, WebFetchTool, WebSearchTool, WriteFileTool,
    EditFileTool,
};
use crate::bus::events::{InboundMessage, OutboundMessage};
use crate::cron::service::CronService;
use crate::providers::base::LLMProvider;
use crate::session::manager::SessionManager;

// ---------------------------------------------------------------------------
// Shared state (Arc-wrapped for concurrent access)
// ---------------------------------------------------------------------------

/// Shared state that is accessed by concurrent message-processing tasks.
///
/// All fields are either `Send + Sync` naturally or wrapped in synchronization
/// primitives. The `sessions` field uses internal locking; all other fields
/// are read-only after construction.
struct AgentLoopShared {
    provider: Arc<dyn LLMProvider>,
    workspace: PathBuf,
    model: String,
    max_iterations: u32,
    max_tokens: u32,
    temperature: f64,
    context: ContextBuilder,
    sessions: SessionManager,
    subagents: Arc<SubagentManager>,
    token_budget: TokenBudget,
    compactor: ContextCompactor,
    learning: LearningStore,
    bus_outbound_tx: UnboundedSender<OutboundMessage>,
    bus_inbound_tx: UnboundedSender<InboundMessage>,
    // Parameters for building per-message tool registries.
    brave_api_key: Option<String>,
    exec_timeout: u64,
    restrict_to_workspace: bool,
    cron_service: Option<Arc<CronService>>,
}

impl AgentLoopShared {
    /// Build a fresh [`ToolRegistry`] with context-sensitive tools (message,
    /// spawn, cron) pre-configured for a specific channel/chat_id.
    ///
    /// This eliminates the shared-context race condition: each concurrent
    /// message-processing task gets its own tools with the correct context.
    fn build_tools(&self, channel: &str, chat_id: &str) -> ToolRegistry {
        let mut tools = ToolRegistry::new();

        // File system tools (stateless).
        tools.register(Box::new(ReadFileTool));
        tools.register(Box::new(WriteFileTool));
        tools.register(Box::new(EditFileTool));
        tools.register(Box::new(ListDirTool));

        // Shell (stateless config).
        tools.register(Box::new(ExecTool::new(
            self.exec_timeout,
            Some(self.workspace.to_string_lossy().to_string()),
            None,
            None,
            self.restrict_to_workspace,
        )));

        // Web (stateless config).
        tools.register(Box::new(WebSearchTool::new(self.brave_api_key.clone(), 5)));
        tools.register(Box::new(WebFetchTool::new(50_000)));

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
        // Set callback and context synchronously (tool is fresh, no contention).
        {
            let st = spawn_tool.clone();
            let cb = spawn_cb;
            let ch = channel.to_string();
            let cid = chat_id.to_string();
            tokio::spawn(async move {
                st.set_callback(cb).await;
                st.set_context(&ch, &cid).await;
            });
        }
        tools.register(Box::new(SpawnToolProxy(spawn_tool)));

        // Cron tool (optional) - context baked in.
        if let Some(ref svc) = self.cron_service {
            let ct = Arc::new(CronScheduleTool::new(svc.clone()));
            let ct2 = ct.clone();
            let ch = channel.to_string();
            let cid = chat_id.to_string();
            tokio::spawn(async move {
                ct2.set_context(&ch, &cid).await;
            });
            tools.register(Box::new(CronToolProxy(ct)));
        }

        tools
    }

    /// Process a regular inbound message through the agent loop.
    ///
    /// This method takes `&self` and is safe to call from multiple concurrent
    /// tasks. Per-message tool instances eliminate shared-context races.
    async fn process_message(&self, msg: &InboundMessage) -> Option<OutboundMessage> {
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
        let tools = self.build_tools(&msg.channel, &msg.chat_id);

        // Get session history.
        let history = self.sessions.get_history(&session_key, 100).await;

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
        let mut messages = self.context.build_messages(
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
        );

        // Track which tools have been used for smart tool selection.
        let mut used_tools: std::collections::HashSet<String> = std::collections::HashSet::new();

        // Pre-loop compaction: summarize old messages if over budget.
        let initial_tool_defs = tools.get_relevant_definitions(&messages, &used_tools);
        let initial_tool_tokens = TokenBudget::estimate_tool_def_tokens(&initial_tool_defs);
        messages = self
            .compactor
            .compact(&messages, &self.token_budget, initial_tool_tokens)
            .await;

        let mut final_content = String::new();

        // Agent loop: call LLM, handle tool calls, repeat.
        for iteration in 0..self.max_iterations {
            debug!("Agent iteration {}/{}", iteration + 1, self.max_iterations);

            // Filter tool definitions to relevant tools.
            let tool_defs = tools.get_relevant_definitions(&messages, &used_tools);
            let tool_defs_opt: Option<&[Value]> = if tool_defs.is_empty() {
                None
            } else {
                Some(&tool_defs)
            };

            // Trim messages to fit context budget.
            let tool_def_tokens =
                TokenBudget::estimate_tool_def_tokens(tool_defs_opt.unwrap_or(&[]));
            messages = self.token_budget.trim_to_fit(&messages, tool_def_tokens);

            // Repair any protocol violations before calling the LLM.
            thread_repair::repair_messages(&mut messages);

            let response = match self
                .provider
                .chat(
                    &messages,
                    tool_defs_opt,
                    Some(&self.model),
                    self.max_tokens,
                    self.temperature,
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
                    debug!("Tool {} result ({}B)", tc.name, result.len());
                    ContextBuilder::add_tool_result(
                        &mut messages,
                        &tc.id,
                        &tc.name,
                        &result,
                    );

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
                    let succeeded = !result.starts_with("Error:");
                    self.learning.record(
                        &tc.name,
                        succeeded,
                        &context_str,
                        if succeeded { None } else { Some(&result) },
                    );
                }
            } else {
                // No tool calls -- the agent is done.
                final_content = response.content.unwrap_or_default();
                break;
            }
        }

        if final_content.is_empty() && messages.len() > 2 {
            final_content = "I completed the requested actions.".to_string();
        }

        // Update session history.
        if final_content.is_empty() {
            self.sessions
                .add_message_and_save(&session_key, "user", &msg.content)
                .await;
        } else {
            self.sessions
                .add_messages_and_save(
                    &session_key,
                    &[("user", &msg.content), ("assistant", &final_content)],
                )
                .await;
        }

        if final_content.is_empty() {
            None
        } else {
            Some(OutboundMessage::new(
                &msg.channel,
                &msg.chat_id,
                &final_content,
            ))
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
}

impl AgentLoop {
    /// Create a new `AgentLoop`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        bus_inbound_rx: UnboundedReceiver<InboundMessage>,
        bus_outbound_tx: UnboundedSender<OutboundMessage>,
        bus_inbound_tx: UnboundedSender<InboundMessage>,
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
        cron_service: Option<Arc<CronService>>,
        max_concurrent_chats: usize,
    ) -> Self {
        let mut context = ContextBuilder::new(&workspace);
        context.model_name = model.clone();
        let sessions = SessionManager::new(&workspace);

        // Create the subagent manager.
        let subagents = Arc::new(SubagentManager::new(
            provider.clone(),
            workspace.clone(),
            bus_inbound_tx.clone(),
            model.clone(),
            brave_api_key.clone(),
            exec_timeout,
            restrict_to_workspace,
        ));

        let token_budget = TokenBudget::new(max_context_tokens, max_tokens as usize);
        let compactor = ContextCompactor::new(provider.clone(), model.clone());
        let learning = LearningStore::new(&workspace);

        let shared = Arc::new(AgentLoopShared {
            provider,
            workspace,
            model,
            max_iterations,
            max_tokens,
            temperature,
            context,
            sessions,
            subagents,
            token_budget,
            compactor,
            learning,
            bus_outbound_tx,
            bus_inbound_tx,
            brave_api_key,
            exec_timeout,
            restrict_to_workspace,
            cron_service,
        });

        Self {
            shared,
            bus_inbound_rx,
            running: Arc::new(AtomicBool::new(false)),
            max_concurrent_chats,
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

        let semaphore = Arc::new(Semaphore::new(self.max_concurrent_chats));
        // Per-session locks to serialize messages within the same conversation.
        let session_locks: Arc<Mutex<HashMap<String, Arc<Mutex<()>>>>> =
            Arc::new(Mutex::new(HashMap::new()));

        while self.running.load(Ordering::SeqCst) {
            let msg = match tokio::time::timeout(
                Duration::from_secs(1),
                self.bus_inbound_rx.recv(),
            )
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

            tokio::spawn(async move {
                // Serialize within the same session.
                let _session_guard = session_lock.lock().await;

                let response = shared.process_message(&msg).await;

                if let Some(outbound) = response {
                    if let Err(e) = outbound_tx.send(outbound) {
                        error!("Failed to publish outbound message: {}", e);
                    }
                }

                drop(permit); // release concurrency slot
            });
        }

        info!("Agent loop stopped");
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
        let mut msg = InboundMessage::new(channel, "user", chat_id, content);
        msg.metadata
            .insert("session_key".to_string(), json!(session_key));

        match self.shared.process_message(&msg).await {
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
