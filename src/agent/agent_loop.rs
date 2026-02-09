//! Main agent loop that consumes inbound messages and produces responses.
//!
//! Ported from Python `agent/loop.py`.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use serde_json::{json, Value};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use tracing::{debug, error, info, warn};

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

/// The core agent loop.
///
/// Consumes [`InboundMessage`]s from the bus, runs the LLM + tool loop, and
/// publishes [`OutboundMessage`]s when the agent produces a response.
pub struct AgentLoop {
    bus_inbound_rx: UnboundedReceiver<InboundMessage>,
    bus_outbound_tx: UnboundedSender<OutboundMessage>,
    bus_inbound_tx: UnboundedSender<InboundMessage>,
    provider: Arc<dyn LLMProvider>,
    workspace: PathBuf,
    model: String,
    max_iterations: u32,
    max_tokens: u32,
    temperature: f64,
    context: ContextBuilder,
    sessions: SessionManager,
    tools: ToolRegistry,
    subagents: Arc<SubagentManager>,
    /// Shared references to tools that need per-message context updates.
    message_tool: Arc<MessageTool>,
    spawn_tool: Arc<SpawnTool>,
    cron_tool: Option<Arc<CronScheduleTool>>,
    running: Arc<AtomicBool>,
    token_budget: TokenBudget,
    learning: LearningStore,
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

        // ---------------------------------------------------------------
        // Build tool registry
        // ---------------------------------------------------------------
        let mut tools = ToolRegistry::new();

        // File system tools.
        tools.register(Box::new(ReadFileTool));
        tools.register(Box::new(WriteFileTool));
        tools.register(Box::new(EditFileTool));
        tools.register(Box::new(ListDirTool));

        // Shell.
        tools.register(Box::new(ExecTool::new(
            exec_timeout,
            Some(workspace.to_string_lossy().to_string()),
            None,
            None,
            restrict_to_workspace,
        )));

        // Web.
        tools.register(Box::new(WebSearchTool::new(brave_api_key.clone(), 5)));
        tools.register(Box::new(WebFetchTool::new(50_000)));

        // Message tool.
        let outbound_tx_clone = bus_outbound_tx.clone();
        let send_cb: SendCallback = Arc::new(move |msg: OutboundMessage| {
            let tx = outbound_tx_clone.clone();
            Box::pin(async move {
                tx.send(msg)
                    .map_err(|e| anyhow::anyhow!("Failed to send outbound message: {}", e))
            })
        });
        let message_tool = Arc::new(MessageTool::new(Some(send_cb), "", ""));
        tools.register(Box::new(MessageToolProxy(message_tool.clone())));

        // Spawn tool.
        let subagents_ref = subagents.clone();
        let spawn_cb: SpawnCallback = Arc::new(move |task, label, channel, chat_id| {
            let mgr = subagents_ref.clone();
            Box::pin(async move { mgr.spawn(task, label, channel, chat_id).await })
        });
        let spawn_tool = Arc::new(SpawnTool::new());
        {
            let spawn_tool_init = spawn_tool.clone();
            let cb = spawn_cb;
            tokio::spawn(async move {
                spawn_tool_init.set_callback(cb).await;
            });
        }
        tools.register(Box::new(SpawnToolProxy(spawn_tool.clone())));

        // Cron tool (optional).
        let cron_tool = cron_service.map(|svc| {
            let ct = Arc::new(CronScheduleTool::new(svc));
            tools.register(Box::new(CronToolProxy(ct.clone())));
            ct
        });

        let token_budget = TokenBudget::new(max_context_tokens, max_tokens as usize);
        let learning = LearningStore::new(&workspace);

        Self {
            bus_inbound_rx,
            bus_outbound_tx,
            bus_inbound_tx,
            provider,
            workspace,
            model,
            max_iterations,
            max_tokens,
            temperature,
            context,
            sessions,
            tools,
            subagents,
            message_tool,
            spawn_tool,
            cron_tool,
            running: Arc::new(AtomicBool::new(false)),
            token_budget,
            learning,
        }
    }

    /// Run the main agent loop until stopped.
    pub async fn run(&mut self) {
        self.running.store(true, Ordering::SeqCst);
        info!("Agent loop started");

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

            // System messages (subagent announces) are handled differently.
            let is_system = msg
                .metadata
                .get("is_system")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

            let response = if is_system {
                self._process_system_message(&msg).await
            } else {
                self._process_message(&msg).await
            };

            if let Some(outbound) = response {
                if let Err(e) = self.bus_outbound_tx.send(outbound) {
                    error!("Failed to publish outbound message: {}", e);
                }
            }
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
        &mut self,
        content: &str,
        session_key: &str,
        channel: &str,
        chat_id: &str,
    ) -> String {
        let mut msg = InboundMessage::new(channel, "user", chat_id, content);
        msg.metadata
            .insert("session_key".to_string(), json!(session_key));

        match self._process_message(&msg).await {
            Some(response) => response.content,
            None => String::new(),
        }
    }

    // ------------------------------------------------------------------
    // Core processing
    // ------------------------------------------------------------------

    /// Process a regular inbound message through the agent loop.
    async fn _process_message(&mut self, msg: &InboundMessage) -> Option<OutboundMessage> {
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

        // Update tool contexts.
        self.message_tool
            .set_context(&msg.channel, &msg.chat_id)
            .await;
        self.spawn_tool
            .set_context(&msg.channel, &msg.chat_id)
            .await;
        if let Some(ref ct) = self.cron_tool {
            ct.set_context(&msg.channel, &msg.chat_id).await;
        }

        // Get or create session.
        let session = self.sessions.get_or_create(&session_key);
        let history = session.get_history(100);

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
        let media_ref: Vec<String> = media_paths;

        // Build messages.
        let mut messages = self.context.build_messages(
            &history,
            &msg.content,
            None,
            if media_ref.is_empty() {
                None
            } else {
                Some(&media_ref)
            },
            Some(&msg.channel),
            Some(&msg.chat_id),
        );

        // Track which tools have been used for smart tool selection.
        let mut used_tools: std::collections::HashSet<String> = std::collections::HashSet::new();

        let mut final_content = String::new();

        // Agent loop: call LLM, handle tool calls, repeat.
        for iteration in 0..self.max_iterations {
            debug!("Agent iteration {}/{}", iteration + 1, self.max_iterations);

            // Phase 3: Filter tool definitions to relevant tools.
            let tool_defs = self
                .tools
                .get_relevant_definitions(&messages, &used_tools);
            let tool_defs_opt: Option<&[Value]> = if tool_defs.is_empty() {
                None
            } else {
                Some(&tool_defs)
            };

            // Phase 1: Trim messages to fit context budget.
            let tool_def_tokens = TokenBudget::estimate_tool_def_tokens(
                tool_defs_opt.unwrap_or(&[]),
            );
            messages = self.token_budget.trim_to_fit(&messages, tool_def_tokens);

            // Phase 2: Repair any protocol violations before calling the LLM.
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
                    let result = self.tools.execute(&tc.name, tc.arguments.clone()).await;
                    debug!(
                        "Tool {} result ({}B)",
                        tc.name,
                        result.len()
                    );
                    ContextBuilder::add_tool_result(
                        &mut messages,
                        &tc.id,
                        &tc.name,
                        &result,
                    );

                    // Phase 3: Track used tools.
                    used_tools.insert(tc.name.clone());

                    // Phase 4: Record tool outcome for learning.
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
        {
            let session = self.sessions.get_or_create(&session_key);
            session.add_message("user", &msg.content);
            if !final_content.is_empty() {
                session.add_message("assistant", &final_content);
            }
        }
        // Mutable borrow dropped; now save from cache.
        self.sessions.save_cached(&session_key);

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

    /// Handle system messages (e.g. subagent completion announcements).
    async fn _process_system_message(
        &mut self,
        msg: &InboundMessage,
    ) -> Option<OutboundMessage> {
        debug!("Processing system message: {}", &msg.content[..msg.content.len().min(80)]);

        // Forward the announcement as an outbound message so the user sees it.
        Some(OutboundMessage::new(
            &msg.channel,
            &msg.chat_id,
            &msg.content,
        ))
    }
}

// ---------------------------------------------------------------------------
// Tool proxy wrappers
// ---------------------------------------------------------------------------
//
// Because `Arc<MessageTool>` etc. don't implement `Tool` directly (the trait
// requires owned `Box<dyn Tool>`), we create thin proxy wrappers that
// implement `Tool` by delegating to the inner `Arc`.

use std::collections::HashMap;

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
