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

/// Truncate text for display: max `max_lines` lines or `max_chars` characters.
fn truncate_for_display(data: &str, max_lines: usize, max_chars: usize) -> String {
    let mut out = String::new();
    let mut lines = 0usize;
    let mut chars = 0usize;
    for line in data.lines() {
        if lines >= max_lines || chars >= max_chars {
            out.push_str("...[truncated]");
            break;
        }
        if !out.is_empty() {
            out.push('\n');
            chars += 1;
        }
        let remaining = max_chars.saturating_sub(chars);
        if line.len() > remaining {
            let partial: String = line.chars().take(remaining).collect();
            out.push_str(&partial);
            out.push_str("...[truncated]");
            break;
        }
        out.push_str(line);
        chars += line.len();
        lines += 1;
    }
    out
}

/// Info about a running subagent task (cheaply cloneable).
#[derive(Clone)]
pub struct SubagentInfo {
    pub task_id: String,
    pub label: String,
    pub started_at: std::time::Instant,
}

/// Manages background subagent tasks.
pub struct SubagentManager {
    provider: Arc<dyn LLMProvider>,
    workspace: PathBuf,
    bus_tx: UnboundedSender<InboundMessage>,
    model: String,
    brave_api_key: Option<String>,
    exec_timeout: u64,
    restrict_to_workspace: bool,
    is_local: bool,
    /// Direct display channel for CLI/REPL mode. In gateway mode the bus
    /// delivers results to channels, but in CLI mode nobody reads the bus
    /// so we send directly to the terminal.
    display_tx: Option<UnboundedSender<String>>,
    running_tasks: Arc<Mutex<HashMap<String, (SubagentInfo, tokio::task::JoinHandle<()>)>>>,
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
        is_local: bool,
    ) -> Self {
        Self {
            provider,
            workspace,
            bus_tx,
            model,
            brave_api_key,
            exec_timeout,
            restrict_to_workspace,
            is_local,
            display_tx: None,
            running_tasks: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Set the display channel for direct CLI/REPL result delivery.
    pub fn with_display_tx(mut self, tx: UnboundedSender<String>) -> Self {
        self.display_tx = Some(tx);
        self
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

        info!("Spawning subagent {} for: {}", task_id, display_label);

        let provider = self.provider.clone();
        let workspace = self.workspace.clone();
        let bus_tx = self.bus_tx.clone();
        let model = self.model.clone();
        let brave_api_key = self.brave_api_key.clone();
        let exec_timeout = self.exec_timeout;
        let restrict_to_workspace = self.restrict_to_workspace;
        let is_local = self.is_local;
        let display_tx = self.display_tx.clone();
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
                is_local,
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

            // In CLI mode, send directly to the terminal since the bus
            // isn't consumed by process_direct().
            if let Some(ref dtx) = display_tx {
                let status_color = if status == "completed" { "\x1b[32m" } else { "\x1b[31m" };
                let header = format!(
                    "\n  {status_color}\u{25cf}\x1b[0m \x1b[1mSubagent: {}\x1b[0m  \x1b[2m({})\x1b[0m  {status_color}{}\x1b[0m",
                    lbl, tid, status
                );
                // Strip markdown formatting for terminal display
                let clean_result = result_text.replace("**", "").replace("__", "");
                let truncated = truncate_for_display(&clean_result, 20, 2000);
                let mut block = header;
                block.push('\n');
                block.push_str("    \x1b[2m\u{250c}\u{2500} result \u{2500}\x1b[0m\n");
                for line in truncated.lines() {
                    block.push_str(&format!("    \x1b[2m\u{2502}\x1b[0m {}\n", line));
                }
                block.push_str("    \x1b[2m\u{2514}\u{2500}\x1b[0m\n");
                let _ = dtx.send(block);
            }

            // Remove self from running tasks.
            let mut tasks = running_tasks.lock().await;
            tasks.remove(&tid);
        });

        // Track the task.
        {
            let info = SubagentInfo {
                task_id: task_id.clone(),
                label: display_label.clone(),
                started_at: std::time::Instant::now(),
            };
            let mut tasks = self.running_tasks.lock().await;
            tasks.insert(task_id.clone(), (info, handle));
        }

        format!(
            "Subagent '{}' spawned (id: {}). It will announce results when done.",
            display_label, task_id
        )
    }

    /// Get the count of currently running subagent tasks.
    pub async fn get_running_count(&self) -> usize {
        let mut tasks = self.running_tasks.lock().await;
        tasks.retain(|_, (_, h)| !h.is_finished());
        tasks.len()
    }

    /// List all running subagent tasks.
    pub async fn list_running(&self) -> Vec<SubagentInfo> {
        let mut tasks = self.running_tasks.lock().await;
        tasks.retain(|_, (_, h)| !h.is_finished());
        tasks.values().map(|(info, _)| info.clone()).collect()
    }

    /// Cancel a running subagent by task ID (or prefix match).
    pub async fn cancel(&self, task_id: &str) -> bool {
        let mut tasks = self.running_tasks.lock().await;
        let key = tasks.keys().find(|k| k.starts_with(task_id)).cloned();
        if let Some(k) = key {
            if let Some((_, handle)) = tasks.remove(&k) {
                handle.abort();
                return true;
            }
        }
        false
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
        is_local: bool,
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
            30000,
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
                    ContextBuilder::add_tool_result(&mut messages, &tc.id, &tc.name, &result.data);
                }

                // Local models (llama-server) require conversations to end
                // with a user message. Mistral/Ministral handle tool→generate
                // natively and break if a user message is injected here.
                if is_local {
                    messages.push(json!({
                        "role": "user",
                        "content": "Based on the tool results above, continue with your task. Call more tools if needed, or provide your final answer."
                    }));
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

        let mut msg =
            InboundMessage::new(origin_channel, "subagent", origin_chat_id, &announcement);
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

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use crate::providers::base::{LLMResponse, ToolCallRequest};

    /// Mock provider that captures messages and returns a tool call on first
    /// call, then a text-only response on second call.
    struct SubagentCapturingProvider {
        captured: tokio::sync::Mutex<Vec<Vec<Value>>>,
    }

    impl SubagentCapturingProvider {
        fn new() -> Self {
            Self {
                captured: tokio::sync::Mutex::new(Vec::new()),
            }
        }
    }

    #[async_trait]
    impl LLMProvider for SubagentCapturingProvider {
        async fn chat(
            &self,
            messages: &[Value],
            _tools: Option<&[Value]>,
            _model: Option<&str>,
            _max_tokens: u32,
            _temperature: f64,
        ) -> anyhow::Result<LLMResponse> {
            let mut captured = self.captured.lock().await;
            let call_num = captured.len();
            captured.push(messages.to_vec());

            if call_num == 0 {
                // First call: return a tool call (list_dir)
                Ok(LLMResponse {
                    content: None,
                    tool_calls: vec![ToolCallRequest {
                        id: "tc_1".to_string(),
                        name: "list_dir".to_string(),
                        arguments: {
                            let mut m = HashMap::new();
                            m.insert("path".to_string(), json!("."));
                            m
                        },
                    }],
                    finish_reason: "tool_calls".to_string(),
                    usage: HashMap::new(),
                })
            } else {
                // Second call: done
                Ok(LLMResponse {
                    content: Some("Task complete.".to_string()),
                    tool_calls: vec![],
                    finish_reason: "stop".to_string(),
                    usage: HashMap::new(),
                })
            }
        }

        fn get_default_model(&self) -> &str {
            "subagent-mock"
        }
    }

    #[tokio::test]
    async fn test_subagent_adds_user_continuation_after_tool_results() {
        let provider = Arc::new(SubagentCapturingProvider::new());
        let workspace = tempfile::tempdir().unwrap().into_path();

        let result = SubagentManager::_run_subagent(
            "test-id",
            "List the current directory",
            "test-label",
            provider.as_ref(),
            &workspace,
            "mock-model",
            None,
            5,
            false,
            false, // is_local
        )
        .await
        .unwrap();

        assert_eq!(result, "Task complete.");

        let captured = provider.captured.lock().await;
        assert_eq!(captured.len(), 2, "Should have made 2 LLM calls");

        // Second call's messages should end with tool results (NOT user
        // continuation). Mistral/Ministral templates handle tool→generate
        // natively and adding a user message breaks role alternation.
        let second_call_msgs = &captured[1];
        let last_msg = second_call_msgs.last().unwrap();
        assert_eq!(
            last_msg["role"].as_str(),
            Some("tool"),
            "Last message before second LLM call should be role:tool, got: {}",
            last_msg
        );

        let roles: Vec<&str> = second_call_msgs
            .iter()
            .filter_map(|m| m["role"].as_str())
            .collect();
        // Expected: system, user, assistant, tool
        assert_eq!(roles.last(), Some(&"tool"));
    }

    #[tokio::test]
    async fn test_subagent_no_tool_calls_returns_immediately() {
        /// Provider that never returns tool calls
        struct ImmediateProvider;

        #[async_trait]
        impl LLMProvider for ImmediateProvider {
            async fn chat(
                &self,
                _messages: &[Value],
                _tools: Option<&[Value]>,
                _model: Option<&str>,
                _max_tokens: u32,
                _temperature: f64,
            ) -> anyhow::Result<LLMResponse> {
                Ok(LLMResponse {
                    content: Some("Immediate answer.".to_string()),
                    tool_calls: vec![],
                    finish_reason: "stop".to_string(),
                    usage: HashMap::new(),
                })
            }
            fn get_default_model(&self) -> &str { "immediate" }
        }

        let workspace = tempfile::tempdir().unwrap().into_path();
        let result = SubagentManager::_run_subagent(
            "test-id",
            "Simple question",
            "test",
            &ImmediateProvider,
            &workspace,
            "mock",
            None,
            5,
            false,
            false, // is_local
        )
        .await
        .unwrap();

        assert_eq!(result, "Immediate answer.");
    }
}
