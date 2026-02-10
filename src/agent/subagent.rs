//! Subagent manager for background task execution.
//!
//! Spawns independent agent loops that can read/write files, execute commands,
//! and search the web, then announce results back to the main agent.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use serde_json::{json, Value};
use tokio::sync::mpsc::UnboundedSender;
use tokio::sync::Mutex;
use tracing::{debug, error, info};
use uuid::Uuid;

use crate::agent::context::ContextBuilder;
use crate::agent::tools::{
    ExecTool, ListDirTool, ReadFileTool, ToolRegistry, WebFetchTool, WebSearchTool, WriteFileTool,
};
use crate::bus::events::InboundMessage;
use crate::providers::base::LLMProvider;

/// Maximum iterations for a subagent run.
const MAX_SUBAGENT_ITERATIONS: u32 = 15;

/// Manages background subagent tasks.
pub struct SubagentManager {
    provider: Arc<dyn LLMProvider>,
    workspace: PathBuf,
    bus_tx: UnboundedSender<InboundMessage>,
    model: String,
    brave_api_key: Option<String>,
    exec_timeout: u64,
    restrict_to_workspace: bool,
    running_tasks: Arc<Mutex<HashMap<String, tokio::task::JoinHandle<()>>>>,
}

impl SubagentManager {
    /// Create a new subagent manager.
    pub fn new(
        provider: Arc<dyn LLMProvider>,
        workspace: PathBuf,
        bus_tx: UnboundedSender<InboundMessage>,
        model: String,
        brave_api_key: Option<String>,
        exec_timeout: u64,
        restrict_to_workspace: bool,
    ) -> Self {
        Self {
            provider,
            workspace,
            bus_tx,
            model,
            brave_api_key,
            exec_timeout,
            restrict_to_workspace,
            running_tasks: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Spawn a background subagent task.
    ///
    /// Returns a status message with the task ID.
    pub async fn spawn(
        &self,
        task: String,
        label: Option<String>,
        origin_channel: String,
        origin_chat_id: String,
    ) -> String {
        let task_id = Uuid::new_v4().to_string()[..8].to_string();
        let display_label = label
            .clone()
            .unwrap_or_else(|| task.chars().take(40).collect());

        info!(
            "Spawning subagent {} for: {}",
            task_id, display_label
        );

        let provider = self.provider.clone();
        let workspace = self.workspace.clone();
        let bus_tx = self.bus_tx.clone();
        let model = self.model.clone();
        let brave_api_key = self.brave_api_key.clone();
        let exec_timeout = self.exec_timeout;
        let restrict_to_workspace = self.restrict_to_workspace;
        let running_tasks = self.running_tasks.clone();
        let tid = task_id.clone();
        let lbl = display_label.clone();
        let tsk = task.clone();

        let handle = tokio::spawn(async move {
            let result = Self::_run_subagent(
                &tid,
                &tsk,
                &lbl,
                provider.as_ref(),
                &workspace,
                &model,
                brave_api_key.as_deref(),
                exec_timeout,
                restrict_to_workspace,
            )
            .await;

            let (result_text, status) = match result {
                Ok(text) => (text, "completed"),
                Err(e) => (format!("Error: {}", e), "failed"),
            };

            Self::_announce_result(
                &bus_tx,
                &tid,
                &lbl,
                &tsk,
                &result_text,
                &origin_channel,
                &origin_chat_id,
                status,
            );

            // Remove self from running tasks.
            let mut tasks = running_tasks.lock().await;
            tasks.remove(&tid);
        });

        // Track the task.
        {
            let mut tasks = self.running_tasks.lock().await;
            tasks.insert(task_id.clone(), handle);
        }

        format!(
            "Subagent '{}' spawned (id: {}). It will announce results when done.",
            display_label, task_id
        )
    }

    /// Get the count of currently running subagent tasks.
    pub async fn get_running_count(&self) -> usize {
        let tasks = self.running_tasks.lock().await;
        tasks.len()
    }

    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    /// Run the subagent agent loop.
    async fn _run_subagent(
        task_id: &str,
        task: &str,
        label: &str,
        provider: &dyn LLMProvider,
        workspace: &PathBuf,
        model: &str,
        brave_api_key: Option<&str>,
        exec_timeout: u64,
        restrict_to_workspace: bool,
    ) -> anyhow::Result<String> {
        debug!("Subagent {} starting: {}", task_id, label);

        // Build a tool registry with basic tools (no message, no spawn).
        let mut tools = ToolRegistry::new();
        tools.register(Box::new(ReadFileTool));
        tools.register(Box::new(WriteFileTool));
        tools.register(Box::new(ListDirTool));
        tools.register(Box::new(ExecTool::new(
            exec_timeout,
            Some(workspace.to_string_lossy().to_string()),
            None,
            None,
            restrict_to_workspace,
        )));
        tools.register(Box::new(WebSearchTool::new(
            brave_api_key.map(|s| s.to_string()),
            5,
        )));
        tools.register(Box::new(WebFetchTool::new(50_000)));

        // Build the subagent system prompt.
        let system_prompt = Self::_build_subagent_prompt(task, workspace);

        let mut messages: Vec<Value> = vec![
            json!({"role": "system", "content": system_prompt}),
            json!({"role": "user", "content": task}),
        ];

        let tool_defs = tools.get_definitions();
        let tool_defs_opt: Option<&[Value]> = if tool_defs.is_empty() {
            None
        } else {
            Some(&tool_defs)
        };

        let mut final_content = String::new();

        for iteration in 0..MAX_SUBAGENT_ITERATIONS {
            debug!(
                "Subagent {} iteration {}/{}",
                task_id,
                iteration + 1,
                MAX_SUBAGENT_ITERATIONS
            );

            let response = provider
                .chat(&messages, tool_defs_opt, Some(model), 4096, 0.7)
                .await?;

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
                    debug!("Subagent {} calling tool: {}", task_id, tc.name);
                    let result = tools.execute(&tc.name, tc.arguments.clone()).await;
                    ContextBuilder::add_tool_result(
                        &mut messages,
                        &tc.id,
                        &tc.name,
                        &result,
                    );
                }
            } else {
                // No tool calls -- the subagent is done.
                final_content = response.content.unwrap_or_default();
                break;
            }
        }

        if final_content.is_empty() {
            final_content = "Subagent completed but produced no final text.".to_string();
        }

        Ok(final_content)
    }

    /// Announce the subagent result to the bus as an InboundMessage.
    fn _announce_result(
        bus_tx: &UnboundedSender<InboundMessage>,
        task_id: &str,
        label: &str,
        task: &str,
        result: &str,
        origin_channel: &str,
        origin_chat_id: &str,
        status: &str,
    ) {
        let announcement = format!(
            "[Subagent {} ({})] Status: {}\nTask: {}\n\nResult:\n{}",
            label, task_id, status, task, result
        );

        let mut msg = InboundMessage::new(
            origin_channel,
            "subagent",
            origin_chat_id,
            &announcement,
        );
        msg.metadata
            .insert("subagent_task_id".to_string(), json!(task_id));
        msg.metadata
            .insert("subagent_status".to_string(), json!(status));
        msg.metadata.insert("is_system".to_string(), json!(true));

        let _ = bus_tx.send(msg);
    }

    /// Build the system prompt for a subagent.
    fn _build_subagent_prompt(task: &str, workspace: &PathBuf) -> String {
        let workspace_str = workspace.to_string_lossy();
        format!(
            r#"You are a subagent of nanobot, a helpful AI assistant.

You have been spawned to complete a specific task. Focus on this task and complete it efficiently.

## Workspace
Your workspace is at: {workspace_str}

## Task
{task}

## Instructions
- Focus only on the assigned task.
- Use tools to accomplish the task (read files, write files, execute commands, search web).
- When done, provide a clear summary of what you accomplished.
- Do not try to communicate with users directly - your result will be announced by the main agent.
- Be thorough but efficient. Do not perform unnecessary actions."#
        )
    }
}
