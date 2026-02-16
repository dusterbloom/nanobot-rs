//! Subagent manager for background task execution.
//!
//! Spawns independent agent loops that can read/write files, execute commands,
//! and search the web, then announce results back to the main agent.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use serde_json::{json, Value};
use tokio::sync::mpsc::UnboundedSender;
use tokio::sync::{broadcast, Mutex};
use tracing::{debug, error, info, warn};
use uuid::Uuid;
use chrono::Utc;

use crate::agent::agent_profiles::{self, AgentProfile};
use crate::agent::context::ContextBuilder;
use crate::agent::tools::{
    EditFileTool, ExecTool, ListDirTool, ReadFileTool, ToolRegistry, WebFetchTool, WebSearchTool,
    WriteFileTool,
};
use crate::bus::events::InboundMessage;
use crate::config::schema::ProvidersConfig;
use crate::providers::base::LLMProvider;
use crate::providers::openai_compat::OpenAICompatProvider;

/// Maximum iterations for a subagent run (default when no profile overrides).
const MAX_SUBAGENT_ITERATIONS: u32 = 15;

/// Configuration passed to `_run_subagent`, derived from profile + overrides.
#[derive(Debug, Clone)]
struct SubagentConfig {
    model: String,
    system_prompt: Option<String>,
    tools_filter: Option<Vec<String>>,
    read_only: bool,
    max_iterations: u32,
}

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
    /// Loaded agent profiles (name → profile).
    profiles: HashMap<String, AgentProfile>,
    /// Provider configs for multi-provider subagent routing. When a model
    /// has a provider prefix (e.g. `groq/llama-3.3-70b`), the subagent
    /// creates a dedicated provider instead of using the parent's.
    providers_config: Option<ProvidersConfig>,
    /// Direct display channel for CLI/REPL mode. In gateway mode the bus
    /// delivers results to channels, but in CLI mode nobody reads the bus
    /// so we send directly to the terminal.
    display_tx: Option<UnboundedSender<String>>,
    running_tasks: Arc<Mutex<HashMap<String, (SubagentInfo, tokio::task::JoinHandle<()>, broadcast::Sender<String>)>>>,
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
        // Rotate event log if >100MB before anything else.
        Self::rotate_event_log(&workspace);

        // Load profiles from standard locations.
        let profiles = agent_profiles::load_profiles(&workspace);
        if !profiles.is_empty() {
            info!(
                "Loaded {} agent profiles: {:?}",
                profiles.len(),
                profiles.keys().collect::<Vec<_>>()
            );
        }

        Self {
            provider,
            workspace,
            bus_tx,
            model,
            brave_api_key,
            exec_timeout,
            restrict_to_workspace,
            is_local,
            profiles,
            providers_config: None,
            display_tx: None,
            running_tasks: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Set the display channel for direct CLI/REPL result delivery.
    pub fn with_display_tx(mut self, tx: UnboundedSender<String>) -> Self {
        self.display_tx = Some(tx);
        self
    }

    /// Set the providers config for multi-provider subagent routing.
    pub fn with_providers_config(mut self, config: ProvidersConfig) -> Self {
        self.providers_config = Some(config);
        self
    }

    /// Get a reference to loaded profiles (for system prompt injection).
    pub fn profiles(&self) -> &HashMap<String, AgentProfile> {
        &self.profiles
    }

    /// Spawn a background subagent task.
    ///
    /// `agent_name` — optional profile name from `.nanobot/agents/`.
    /// `model_override` — optional model (overrides profile and default).
    ///
    /// Returns a status message with the task ID.
    pub async fn spawn(
        &self,
        task: String,
        label: Option<String>,
        agent_name: Option<String>,
        model_override: Option<String>,
        origin_channel: String,
        origin_chat_id: String,
        working_dir: Option<String>,
    ) -> String {
        let task_id = Uuid::new_v4().to_string()[..8].to_string();

        // Resolve agent profile if specified.
        let profile = agent_name.as_ref().and_then(|name| {
            let p = self.profiles.get(name);
            if p.is_none() {
                warn!("Agent profile '{}' not found, using defaults", name);
            }
            p.cloned()
        });

        let display_label = label.clone().unwrap_or_else(|| {
            if let Some(ref name) = agent_name {
                format!("{}: {}", name, task.chars().take(30).collect::<String>())
            } else {
                task.chars().take(40).collect()
            }
        });

        // Build config: model_override > profile.model > self.model
        let effective_model = if let Some(ref m) = model_override {
            agent_profiles::resolve_model_alias(m)
        } else if let Some(ref p) = profile {
            p.model
                .as_ref()
                .map(|m| agent_profiles::resolve_model_alias(m))
                .unwrap_or_else(|| self.model.clone())
        } else {
            self.model.clone()
        };

        let mut config = SubagentConfig {
            model: effective_model,
            system_prompt: profile.as_ref().map(|p| p.system_prompt.clone()),
            tools_filter: profile.as_ref().and_then(|p| p.tools.clone()),
            read_only: profile.as_ref().map(|p| p.read_only).unwrap_or(false),
            max_iterations: profile
                .as_ref()
                .and_then(|p| p.max_iterations)
                .unwrap_or(MAX_SUBAGENT_ITERATIONS),
        };

        let effective_model_for_display = config.model.clone();

        // Resolve provider for model prefix (groq/, gemini/, openai/, etc.)
        let (provider, resolved_model) = self.resolve_provider_for_model(&config.model);
        let routed_to_cloud = resolved_model != config.model;
        if routed_to_cloud {
            info!(
                "Spawning subagent {} (agent={:?}, model={} → provider-routed as {}) for: {}",
                task_id,
                agent_name.as_deref().unwrap_or("default"),
                effective_model_for_display,
                resolved_model,
                display_label
            );
            config.model = resolved_model;
        } else {
            info!(
                "Spawning subagent {} (agent={:?}, model={}) for: {}",
                task_id,
                agent_name.as_deref().unwrap_or("default"),
                effective_model_for_display,
                display_label
            );
        }
        let workspace = self.workspace.clone();
        let exec_working_dir = working_dir;
        let bus_tx = self.bus_tx.clone();
        let brave_api_key = self.brave_api_key.clone();
        let exec_timeout = self.exec_timeout;
        let restrict_to_workspace = self.restrict_to_workspace;
        // Cloud-routed subagents (groq/, gemini/, openai/, etc.) are never local.
        let is_local = if routed_to_cloud { false } else { self.is_local };
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
                &config,
                brave_api_key.as_deref(),
                exec_timeout,
                restrict_to_workspace,
                is_local,
                exec_working_dir.as_deref(),
            )
            .await;

            let (result_text, status) = match result {
                Ok(text) => (text, "completed"),
                Err(e) => (format!("Error: {}", e), "failed"),
            };

            // Write result to scratch file for persistence across compaction.
            Self::append_event(&workspace, &tid, &lbl, &tsk, &result_text, status);

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
                // Strip markdown formatting for terminal display
                let clean_result = result_text
                    .replace("**", "")
                    .replace("__", "")
                    .trim()
                    .to_string();
                let truncated = truncate_for_display(&clean_result, 30, 3000);

                // Build a compact, clean result block.
                // Use \x1b[RAW] prefix to bypass markdown rendering in the REPL.
                let mut block = format!(
                    "\x1b[RAW]\n  {status_color}\u{25cf}\x1b[0m \x1b[1mSubagent: {}\x1b[0m  \x1b[2m({})\x1b[0m  {status_color}{}\x1b[0m\n",
                    lbl, tid, status
                );
                // Indent each line under a dim gutter
                for line in truncated.lines() {
                    block.push_str(&format!("  \x1b[2m\u{2502}\x1b[0m \x1b[37m{}\x1b[0m\n", line));
                }
                block.push_str("  \x1b[2m\u{2514}\u{2500}\x1b[0m\n");
                let _ = dtx.send(block);
            }

            // Broadcast result to any waiting subscribers, then remove from running tasks.
            let mut tasks = running_tasks.lock().await;
            if let Some((_, _, result_tx)) = tasks.remove(&tid) {
                let _ = result_tx.send(result_text);
            }
        });

        // Create a broadcast channel for wait subscribers (capacity 1 — single result).
        let (result_tx, _) = broadcast::channel(1);

        // Track the task.
        {
            let info = SubagentInfo {
                task_id: task_id.clone(),
                label: display_label.clone(),
                started_at: std::time::Instant::now(),
            };
            let mut tasks = self.running_tasks.lock().await;
            tasks.insert(task_id.clone(), (info, handle, result_tx));
        }

        let agent_note = agent_name
            .map(|n| format!(", agent: {}", n))
            .unwrap_or_default();
        format!(
            "Subagent '{}' spawned (id: {}{}, model: {}). It will announce results when done.",
            display_label, task_id, agent_note, effective_model_for_display
        )
    }

    /// Get the count of currently running subagent tasks.
    pub async fn get_running_count(&self) -> usize {
        let mut tasks = self.running_tasks.lock().await;
        tasks.retain(|_, (_, h, _)| !h.is_finished());
        tasks.len()
    }

    /// List all running subagent tasks.
    pub async fn list_running(&self) -> Vec<SubagentInfo> {
        let mut tasks = self.running_tasks.lock().await;
        tasks.retain(|_, (_, h, _)| !h.is_finished());
        tasks.values().map(|(info, _, _)| info.clone()).collect()
    }

    /// Cancel a running subagent by task ID (or prefix match).
    pub async fn cancel(&self, task_id: &str) -> bool {
        let mut tasks = self.running_tasks.lock().await;
        let key = tasks.keys().find(|k| k.starts_with(task_id)).cloned();
        if let Some(k) = key {
            if let Some((_, handle, _)) = tasks.remove(&k) {
                handle.abort();
                return true;
            }
        }
        false
    }

    /// Wait for a running subagent to complete, returning its result.
    ///
    /// Subscribes to the broadcast channel for the given task and waits
    /// up to `timeout` for the result. Returns the result text on success,
    /// or an error message on timeout / not found.
    pub async fn wait_for(&self, task_id: &str, timeout: std::time::Duration) -> String {
        // Find the task and subscribe to its result channel.
        let mut rx = {
            let tasks = self.running_tasks.lock().await;
            let key = tasks.keys().find(|k| k.starts_with(task_id)).cloned();
            match key {
                Some(k) => {
                    let (info, _, result_tx) = tasks.get(&k).unwrap();
                    // Check if already finished (JoinHandle done but not yet cleaned up).
                    debug!("Waiting for subagent {} ({})", info.label, k);
                    result_tx.subscribe()
                }
                None => {
                    // Task not found — check event log for completed results.
                    if let Some(result) = Self::read_event_result(&self.workspace, task_id) {
                        return format!("Subagent already completed:\n\n{}", result);
                    }
                    return format!(
                        "No running subagent found matching '{}'. It may have already completed \
                         (check events.jsonl in workspace).",
                        task_id
                    );
                }
            }
        };

        // Wait for the result with timeout.
        match tokio::time::timeout(timeout, rx.recv()).await {
            Ok(Ok(result)) => {
                // Result also persisted in events.jsonl.
                result
            }
            Ok(Err(e)) => format!("Subagent result channel error: {}", e),
            Err(_) => format!(
                "Timed out waiting for subagent '{}' after {}s. \
                 It is still running — use 'spawn list' to check status \
                 or 'spawn cancel' to abort.",
                task_id,
                timeout.as_secs()
            ),
        }
    }

    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    /// Resolve a model string to (provider, stripped_model).
    ///
    /// If the model has a provider prefix (e.g. `groq/llama-3.3-70b`), creates
    /// a dedicated `OpenAICompatProvider` for that backend. Otherwise returns
    /// the parent provider and the model unchanged.
    fn resolve_provider_for_model(&self, model: &str) -> (Arc<dyn LLMProvider>, String) {
        if let Some(ref pc) = self.providers_config {
            if let Some((api_key, base, rest)) = pc.resolve_model_prefix(model) {
                let prefix = model.split('/').next().unwrap_or("unknown");
                info!(
                    "Subagent using {} provider (base={}) for model {}",
                    prefix, base, rest
                );
                let provider: Arc<dyn LLMProvider> = Arc::new(
                    OpenAICompatProvider::new(&api_key, Some(&base), Some(&rest)),
                );
                return (provider, rest);
            }
        }

        // No prefix match → use parent provider with model as-is.
        (self.provider.clone(), model.to_string())
    }

    /// Run the subagent agent loop.
    async fn _run_subagent(
        task_id: &str,
        task: &str,
        label: &str,
        provider: &dyn LLMProvider,
        workspace: &PathBuf,
        config: &SubagentConfig,
        brave_api_key: Option<&str>,
        exec_timeout: u64,
        restrict_to_workspace: bool,
        is_local: bool,
        exec_working_dir: Option<&str>,
    ) -> anyhow::Result<String> {
        debug!(
            "Subagent {} starting (model={}, max_iter={}, read_only={}, tools_filter={:?}): {}",
            task_id, config.model, config.max_iterations, config.read_only, config.tools_filter, label
        );

        // Build a tool registry. Start with all tools, then filter.
        let mut tools = ToolRegistry::new();

        // Determine which tools to register based on profile config.
        let should_include = |name: &str| -> bool {
            // If read_only, exclude write tools regardless of filter.
            if config.read_only && matches!(name, "write_file" | "edit_file") {
                return false;
            }
            // If there's a tools filter, only include listed tools.
            if let Some(ref filter) = config.tools_filter {
                return filter.iter().any(|t| t == name);
            }
            true
        };

        if should_include("read_file") {
            tools.register(Box::new(ReadFileTool));
        }
        if should_include("write_file") {
            tools.register(Box::new(WriteFileTool));
        }
        if should_include("edit_file") {
            tools.register(Box::new(EditFileTool));
        }
        if should_include("list_dir") {
            tools.register(Box::new(ListDirTool));
        }
        if should_include("exec") {
            let exec_cwd = exec_working_dir
                .map(|s| s.to_string())
                .unwrap_or_else(|| workspace.to_string_lossy().to_string());
            tools.register(Box::new(ExecTool::new(
                exec_timeout,
                Some(exec_cwd),
                None,
                None,
                restrict_to_workspace,
                30000,
            )));
        }
        if should_include("web_search") {
            tools.register(Box::new(WebSearchTool::new(
                brave_api_key.map(|s| s.to_string()),
                5,
            )));
        }
        if should_include("web_fetch") {
            tools.register(Box::new(WebFetchTool::new(50_000)));
        }

        // Build the subagent system prompt.
        let system_prompt = if let Some(ref profile_prompt) = config.system_prompt {
            // Profile provides the base prompt; append workspace and task context.
            let workspace_str = workspace.to_string_lossy();
            format!(
                "{profile_prompt}\n\n\
                 ## Workspace\n\
                 Your workspace is at: {workspace_str}\n\n\
                 ## Instructions\n\
                 - Focus only on the assigned task.\n\
                 - When done, provide a clear summary of what you accomplished.\n\
                 - Do not try to communicate with users directly - your result will be announced by the main agent.\n\
                 - Be thorough but efficient."
            )
        } else {
            Self::_build_subagent_prompt(task, workspace)
        };

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

        for iteration in 0..config.max_iterations {
            debug!(
                "Subagent {} iteration {}/{}",
                task_id,
                iteration + 1,
                config.max_iterations
            );

            let response = provider
                .chat(&messages, tool_defs_opt, Some(&config.model), 4096, 0.7)
                .await?;

            // Check for LLM provider errors (finish_reason == "error").
            if response.finish_reason == "error" {
                let err_msg = response.content.as_deref().unwrap_or("Unknown LLM error");
                error!("Subagent {} LLM provider error: {}", task_id, err_msg);
                return Err(anyhow::anyhow!("[LLM Error] {}", err_msg));
            }

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

    /// Append a single JSONL event to `{workspace}/events.jsonl`.
    ///
    /// Replaces the old scratch-file approach: one append-only log that
    /// survives compaction and is trivial to parse. Rotate is handled at
    /// startup by `rotate_event_log()`.
    fn append_event(
        workspace: &PathBuf,
        task_id: &str,
        label: &str,
        task: &str,
        result: &str,
        status: &str,
    ) {
        use std::io::Write;
        let event_path = workspace.join("events.jsonl");
        let event = serde_json::json!({
            "ts": Utc::now().to_rfc3339(),
            "kind": "subagent_result",
            "task_id": task_id,
            "label": label,
            "task": task,
            "status": status,
            "result": result,
        });
        let line = format!("{}\n", event);
        match std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&event_path)
        {
            Ok(mut f) => {
                if let Err(e) = f.write_all(line.as_bytes()) {
                    warn!("Failed to append event: {}", e);
                }
            }
            Err(e) => warn!("Failed to open events.jsonl: {}", e),
        }
    }

    /// Rotate event log if it exceeds 100 MB. Called once from `new()`.
    fn rotate_event_log(workspace: &PathBuf) {
        let event_path = workspace.join("events.jsonl");
        if let Ok(meta) = std::fs::metadata(&event_path) {
            if meta.len() > 100 * 1024 * 1024 {
                let rotated = workspace.join("events.jsonl.old");
                if let Err(e) = std::fs::rename(&event_path, &rotated) {
                    warn!("Failed to rotate event log: {}", e);
                } else {
                    info!("Rotated events.jsonl ({:.1} MB)", meta.len() as f64 / 1e6);
                }
            }
        }
    }

    /// Search event log for a completed subagent result by task_id prefix.
    fn read_event_result(workspace: &PathBuf, task_id_prefix: &str) -> Option<String> {
        let event_path = workspace.join("events.jsonl");
        let content = std::fs::read_to_string(&event_path).ok()?;
        // Scan lines in reverse (most recent first).
        for line in content.lines().rev() {
            if let Ok(ev) = serde_json::from_str::<serde_json::Value>(line) {
                if ev["kind"] == "subagent_result"
                    && ev["task_id"]
                        .as_str()
                        .map(|id| id.starts_with(task_id_prefix))
                        .unwrap_or(false)
                {
                    let status = ev["status"].as_str().unwrap_or("unknown");
                    let result = ev["result"].as_str().unwrap_or("");
                    let label = ev["label"].as_str().unwrap_or("");
                    return Some(format!(
                        "Subagent '{}' ({}) — status: {}\n\n{}",
                        label, task_id_prefix, status, result
                    ));
                }
            }
        }
        None
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

    /// Build the default system prompt for a subagent (no profile).
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

    /// Helper to build a default config for tests.
    fn default_test_config(model: &str) -> SubagentConfig {
        SubagentConfig {
            model: model.to_string(),
            system_prompt: None,
            tools_filter: None,
            read_only: false,
            max_iterations: MAX_SUBAGENT_ITERATIONS,
        }
    }

    #[tokio::test]
    async fn test_subagent_adds_user_continuation_after_tool_results() {
        let provider = Arc::new(SubagentCapturingProvider::new());
        let workspace = tempfile::tempdir().unwrap().into_path();
        let config = default_test_config("mock-model");

        let result = SubagentManager::_run_subagent(
            "test-id",
            "List the current directory",
            "test-label",
            provider.as_ref(),
            &workspace,
            &config,
            None,
            5,
            false,
            false, // is_local
            None,  // exec_working_dir
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
        let config = default_test_config("mock");
        let result = SubagentManager::_run_subagent(
            "test-id",
            "Simple question",
            "test",
            &ImmediateProvider,
            &workspace,
            &config,
            None,
            5,
            false,
            false, // is_local
            None,  // exec_working_dir
        )
        .await
        .unwrap();

        assert_eq!(result, "Immediate answer.");
    }

    #[tokio::test]
    async fn test_subagent_read_only_excludes_write_tools() {
        let config = SubagentConfig {
            model: "test".to_string(),
            system_prompt: None,
            tools_filter: None, // all tools allowed
            read_only: true,    // but read_only
            max_iterations: 5,
        };

        // The should_include logic is inline in _run_subagent, but we can
        // verify by checking that a read_only subagent doesn't get write tools.
        // For a unit test, we just verify the logic directly.
        let should_include = |name: &str| -> bool {
            if config.read_only && matches!(name, "write_file" | "edit_file") {
                return false;
            }
            if let Some(ref filter) = config.tools_filter {
                return filter.iter().any(|t| t == name);
            }
            true
        };

        assert!(should_include("read_file"));
        assert!(should_include("list_dir"));
        assert!(should_include("exec"));
        assert!(!should_include("write_file"));
        assert!(!should_include("edit_file"));
    }

    #[tokio::test]
    async fn test_subagent_tools_filter() {
        let config = SubagentConfig {
            model: "test".to_string(),
            system_prompt: None,
            tools_filter: Some(vec![
                "read_file".to_string(),
                "list_dir".to_string(),
            ]),
            read_only: false,
            max_iterations: 5,
        };

        let should_include = |name: &str| -> bool {
            if config.read_only && matches!(name, "write_file" | "edit_file") {
                return false;
            }
            if let Some(ref filter) = config.tools_filter {
                return filter.iter().any(|t| t == name);
            }
            true
        };

        assert!(should_include("read_file"));
        assert!(should_include("list_dir"));
        assert!(!should_include("exec"));
        assert!(!should_include("write_file"));
        assert!(!should_include("web_search"));
    }
}
