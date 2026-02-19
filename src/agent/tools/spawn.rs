//! Spawn tool for creating background subagents.
//!
//! Supports named agent profiles and model overrides for context-efficient
//! delegation. Also supports listing running subagents and cancelling them.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::Mutex;

use super::base::Tool;

/// Type alias for the spawn callback.
///
/// Arguments: (task, label, agent_name, model_override, origin_channel, origin_chat_id, working_dir) -> result string.
pub type SpawnCallback = Arc<
    dyn Fn(
            String,
            Option<String>,
            Option<String>,
            Option<String>,
            String,
            String,
            Option<String>,
        ) -> Pin<Box<dyn Future<Output = String> + Send>>
        + Send
        + Sync,
>;

/// Type alias for the list callback. Returns formatted list of running subagents.
pub type ListCallback = Arc<dyn Fn() -> Pin<Box<dyn Future<Output = String> + Send>> + Send + Sync>;

/// Type alias for the cancel callback. Takes task_id prefix, returns success message.
pub type CancelCallback =
    Arc<dyn Fn(String) -> Pin<Box<dyn Future<Output = String> + Send>> + Send + Sync>;

/// Type alias for the wait callback. Takes task_id prefix + timeout secs, returns result.
pub type WaitCallback =
    Arc<dyn Fn(String, u64) -> Pin<Box<dyn Future<Output = String> + Send>> + Send + Sync>;

/// Type alias for the check callback. Takes task_id prefix, returns result without blocking.
pub type CheckCallback =
    Arc<dyn Fn(String) -> Pin<Box<dyn Future<Output = String> + Send>> + Send + Sync>;

/// Type alias for the pipeline callback. Takes (steps_json, ahead_by_k) -> result string.
pub type PipelineCallback =
    Arc<dyn Fn(String, usize) -> Pin<Box<dyn Future<Output = String> + Send>> + Send + Sync>;

/// Type alias for the loop callback.
/// Takes (task, max_rounds, tools, stop_condition, model, working_dir) -> result string.
pub type LoopCallback = Arc<
    dyn Fn(
            String,
            u32,
            Option<Vec<String>>,
            Option<String>,
            Option<String>,
            Option<String>,
        ) -> Pin<Box<dyn Future<Output = String> + Send>>
        + Send
        + Sync,
>;

/// Tool to spawn a subagent for background task execution.
///
/// The subagent runs asynchronously and announces its result back
/// to the main agent when complete. Supports named agent profiles
/// for specialized behavior and model overrides for cost control.
///
/// Also supports `action: "list"` to check running subagents and
/// `action: "cancel"` to abort a stuck subagent.
pub struct SpawnTool {
    spawn_callback: Arc<Mutex<Option<SpawnCallback>>>,
    list_callback: Arc<Mutex<Option<ListCallback>>>,
    cancel_callback: Arc<Mutex<Option<CancelCallback>>>,
    wait_callback: Arc<Mutex<Option<WaitCallback>>>,
    check_callback: Arc<Mutex<Option<CheckCallback>>>,
    pipeline_callback: Arc<Mutex<Option<PipelineCallback>>>,
    loop_callback: Arc<Mutex<Option<LoopCallback>>>,
    origin_channel: Arc<Mutex<String>>,
    origin_chat_id: Arc<Mutex<String>>,
}

impl SpawnTool {
    /// Create a new spawn tool.
    pub fn new() -> Self {
        Self {
            spawn_callback: Arc::new(Mutex::new(None)),
            list_callback: Arc::new(Mutex::new(None)),
            cancel_callback: Arc::new(Mutex::new(None)),
            wait_callback: Arc::new(Mutex::new(None)),
            check_callback: Arc::new(Mutex::new(None)),
            pipeline_callback: Arc::new(Mutex::new(None)),
            loop_callback: Arc::new(Mutex::new(None)),
            origin_channel: Arc::new(Mutex::new("cli".to_string())),
            origin_chat_id: Arc::new(Mutex::new("direct".to_string())),
        }
    }

    /// Set the origin context for subagent announcements.
    pub async fn set_context(&self, channel: &str, chat_id: &str) {
        *self.origin_channel.lock().await = channel.to_string();
        *self.origin_chat_id.lock().await = chat_id.to_string();
    }

    /// Set the spawn callback.
    pub async fn set_callback(&self, callback: SpawnCallback) {
        *self.spawn_callback.lock().await = Some(callback);
    }

    /// Set the list callback for checking running subagents.
    pub async fn set_list_callback(&self, callback: ListCallback) {
        *self.list_callback.lock().await = Some(callback);
    }

    /// Set the cancel callback for aborting subagents.
    pub async fn set_cancel_callback(&self, callback: CancelCallback) {
        *self.cancel_callback.lock().await = Some(callback);
    }

    /// Set the wait callback for blocking until a subagent completes.
    pub async fn set_wait_callback(&self, callback: WaitCallback) {
        *self.wait_callback.lock().await = Some(callback);
    }

    /// Set the check callback for non-blocking result lookup.
    pub async fn set_check_callback(&self, callback: CheckCallback) {
        *self.check_callback.lock().await = Some(callback);
    }

    /// Set the pipeline callback for running multi-step pipelines.
    pub async fn set_pipeline_callback(&self, callback: PipelineCallback) {
        *self.pipeline_callback.lock().await = Some(callback);
    }

    /// Set the loop callback for autonomous refinement loops.
    pub async fn set_loop_callback(&self, callback: LoopCallback) {
        *self.loop_callback.lock().await = Some(callback);
    }
}

impl Default for SpawnTool {
    fn default() -> Self {
        Self::new()
    }
}

/// Lightweight spawn tool for local models with limited context.
///
/// Same callbacks and execution logic as SpawnTool, but with a minimal
/// schema (~200 tokens vs ~1,100). Drops pipeline/loop actions that
/// require cloud-level reasoning.
pub struct SpawnToolLite(pub Arc<SpawnTool>);

#[async_trait]
impl Tool for SpawnToolLite {
    fn name(&self) -> &str {
        "spawn"
    }

    fn description(&self) -> &str {
        "Run a background task, list tasks, check results, wait, or cancel."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["spawn", "list", "check", "wait", "cancel"]
                },
                "task": {
                    "type": "string",
                    "description": "Task description (for spawn)"
                },
                "task_id": {
                    "type": "string",
                    "description": "Task ID (for check/wait/cancel)"
                }
            }
        })
    }

    async fn execute(&self, params: HashMap<String, serde_json::Value>) -> String {
        self.0.execute(params).await
    }
}

#[async_trait]
impl Tool for SpawnTool {
    fn name(&self) -> &str {
        "spawn"
    }

    fn description(&self) -> &str {
        "Spawn a subagent to handle a task in the background, list running and recently completed subagents, \
         check a specific result, wait for one to finish, cancel one, \
         run a multi-step pipeline, or run an autonomous refinement loop. \
         Use action='spawn' (default) to start a new subagent. \
         Use action='list' to see running subagents AND recently completed ones. \
         Use action='check' with task_id to retrieve a completed subagent's result (non-blocking). \
         Use action='wait' with task_id to block until a subagent finishes and get its result. \
         Use action='cancel' with task_id to abort a stuck subagent. \
         Use action='pipeline' with steps array for multi-step execution with optional tool use per step. \
         Use action='loop' with task and max_rounds for autonomous iterative refinement with tools. \
         Use 'agent' to pick a specialized profile (explore, reviewer, builder, researcher) \
         and 'model' to control cost (e.g. 'haiku' for cheap/fast tasks)."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Action to perform: 'spawn' (default, requires 'task'), 'check'/'wait'/'cancel' (require 'task_id'), 'pipeline' (requires 'steps'), 'loop' (requires 'task'), or 'list'",
                    "enum": ["spawn", "list", "check", "wait", "cancel", "pipeline", "loop"]
                },
                "steps": {
                    "type": "array",
                    "description": "Pipeline steps array (for action='pipeline'). Each step: {prompt, expected?, tools?, max_iterations?}",
                    "items": {
                        "type": "object",
                        "properties": {
                            "prompt": { "type": "string", "description": "The prompt/task for this step" },
                            "expected": { "type": "string", "description": "Expected answer for verification" },
                            "tools": {
                                "type": "array",
                                "items": { "type": "string" },
                                "description": "Tool names this step can use (e.g. ['exec', 'read_file']). Omit for text-only steps."
                            },
                            "max_iterations": {
                                "type": "integer",
                                "description": "Max tool iterations for this step (default: 5). Only used when tools are specified."
                            }
                        }
                    }
                },
                "ahead_by_k": {
                    "type": "integer",
                    "description": "MAKER voting margin (for action='pipeline'). 0 = no voting. Default: 0"
                },
                "task": {
                    "type": "string",
                    "description": "The task for the subagent to complete (required for action='spawn')"
                },
                "task_id": {
                    "type": "string",
                    "description": "Task ID or prefix to wait for or cancel (required for action='wait' and action='cancel')"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds for action='wait' (default: 120)"
                },
                "label": {
                    "type": "string",
                    "description": "Optional short label for the task (for display)"
                },
                "agent": {
                    "type": "string",
                    "description": "Agent profile name (e.g. 'explore', 'reviewer', 'builder', 'researcher'). Omit for general-purpose."
                },
                "model": {
                    "type": "string",
                    "description": "Model override. Use 'haiku' for fast/cheap, 'sonnet' for balanced, 'opus' for complex reasoning, 'local' for local model. Use provider prefix for external models: 'groq/llama-3.3-70b-versatile', 'gemini/gemini-2.0-flash', 'openai/gpt-4o'. Omit to use profile default or parent model."
                },
                "working_dir": {
                    "type": "string",
                    "description": "Working directory for the subagent's exec tool. Defaults to the workspace directory."
                },
                "max_rounds": {
                    "type": "integer",
                    "description": "Maximum rounds for action='loop' (default: 5). Each round runs a full agent iteration."
                },
                "tools": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Tool names for action='loop' (e.g. ['read_file', 'exec', 'write_file'])"
                },
                "stop_condition": {
                    "type": "string",
                    "description": "Stop condition text for action='loop'. Loop stops when output contains this text or 'DONE'."
                }
            },
            "required": []
        })
    }

    async fn execute(&self, params: HashMap<String, serde_json::Value>) -> String {
        let action = params
            .get("action")
            .and_then(|v| v.as_str())
            .unwrap_or("spawn");

        match action {
            "list" => {
                let cb_guard = self.list_callback.lock().await;
                match cb_guard.as_ref() {
                    Some(cb) => {
                        let cb = cb.clone();
                        drop(cb_guard);
                        cb().await
                    }
                    None => "Error: List callback not configured".to_string(),
                }
            }
            "check" => {
                let task_id = match params.get("task_id").and_then(|v| v.as_str()) {
                    Some(id) => id.to_string(),
                    None => return "Error: 'task_id' parameter is required for check".to_string(),
                };
                let cb_guard = self.check_callback.lock().await;
                match cb_guard.as_ref() {
                    Some(cb) => {
                        let cb = cb.clone();
                        drop(cb_guard);
                        cb(task_id).await
                    }
                    None => "Error: Check callback not configured".to_string(),
                }
            }
            "cancel" => {
                let task_id = match params.get("task_id").and_then(|v| v.as_str()) {
                    Some(id) => id.to_string(),
                    None => return "Error: 'task_id' parameter is required for cancel".to_string(),
                };
                let cb_guard = self.cancel_callback.lock().await;
                match cb_guard.as_ref() {
                    Some(cb) => {
                        let cb = cb.clone();
                        drop(cb_guard);
                        cb(task_id).await
                    }
                    None => "Error: Cancel callback not configured".to_string(),
                }
            }
            "wait" => {
                let task_id = match params.get("task_id").and_then(|v| v.as_str()) {
                    Some(id) => id.to_string(),
                    None => return "Error: 'task_id' parameter is required for wait".to_string(),
                };
                let timeout = params
                    .get("timeout")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(120);
                let cb_guard = self.wait_callback.lock().await;
                match cb_guard.as_ref() {
                    Some(cb) => {
                        let cb = cb.clone();
                        drop(cb_guard);
                        cb(task_id, timeout).await
                    }
                    None => "Error: Wait callback not configured".to_string(),
                }
            }
            "pipeline" => {
                let steps_json = match params.get("steps") {
                    Some(v) => serde_json::to_string(v).unwrap_or_else(|_| "[]".to_string()),
                    None => return "Error: 'steps' parameter is required for pipeline".to_string(),
                };
                let ahead_by_k = params
                    .get("ahead_by_k")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as usize;

                let cb_guard = self.pipeline_callback.lock().await;
                match cb_guard.as_ref() {
                    Some(cb) => {
                        let cb = cb.clone();
                        drop(cb_guard);
                        cb(steps_json, ahead_by_k).await
                    }
                    None => "Error: Pipeline callback not configured".to_string(),
                }
            }
            "loop" => {
                let task = match params.get("task").and_then(|v| v.as_str()) {
                    Some(t) => t.to_string(),
                    None => return "Error: 'task' parameter is required for loop".to_string(),
                };
                let max_rounds = params
                    .get("max_rounds")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(5) as u32;
                let tools_filter: Option<Vec<String>> =
                    params.get("tools").and_then(|v| v.as_array()).map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str().map(|s| s.to_string()))
                            .collect()
                    });
                let stop_condition = params
                    .get("stop_condition")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                let model = params
                    .get("model")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                let working_dir = params
                    .get("working_dir")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());

                let cb_guard = self.loop_callback.lock().await;
                match cb_guard.as_ref() {
                    Some(cb) => {
                        let cb = cb.clone();
                        drop(cb_guard);
                        cb(
                            task,
                            max_rounds,
                            tools_filter,
                            stop_condition,
                            model,
                            working_dir,
                        )
                        .await
                    }
                    None => "Error: Loop callback not configured".to_string(),
                }
            }
            "spawn" | _ => {
                let task = match params.get("task").and_then(|v| v.as_str()) {
                    Some(t) => t.to_string(),
                    None => return "Error: 'task' parameter is required".to_string(),
                };

                let label = params
                    .get("label")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());

                let agent = params
                    .get("agent")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());

                let model = params
                    .get("model")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());

                let working_dir = params
                    .get("working_dir")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());

                let channel = self.origin_channel.lock().await.clone();
                let chat_id = self.origin_chat_id.lock().await.clone();

                let callback_guard = self.spawn_callback.lock().await;
                let callback = match callback_guard.as_ref() {
                    Some(cb) => cb.clone(),
                    None => return "Error: Spawn callback not configured".to_string(),
                };
                // Drop the lock before awaiting.
                drop(callback_guard);

                callback(task, label, agent, model, channel, chat_id, working_dir).await
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spawn_tool_name() {
        let tool = SpawnTool::new();
        assert_eq!(tool.name(), "spawn");
    }

    #[test]
    fn test_spawn_tool_default() {
        let tool = SpawnTool::default();
        assert_eq!(tool.name(), "spawn");
    }

    #[test]
    fn test_spawn_tool_description() {
        let tool = SpawnTool::new();
        let desc = tool.description();
        assert!(!desc.is_empty());
        assert!(desc.contains("subagent") || desc.contains("background"));
        assert!(desc.contains("list"));
        assert!(desc.contains("cancel"));
    }

    #[test]
    fn test_spawn_tool_parameters() {
        let tool = SpawnTool::new();
        let params = tool.parameters();
        assert_eq!(params["type"], "object");
        assert!(params["properties"]["task"].is_object());
        assert!(params["properties"]["action"].is_object());
        assert!(params["properties"]["task_id"].is_object());
        assert!(params["properties"]["agent"].is_object());
        assert!(params["properties"]["model"].is_object());
        // No required params (task only needed for spawn, not list/cancel)
        let required = params["required"].as_array().unwrap();
        assert!(required.is_empty());
        // No oneOf — Anthropic rejects it. Requirements are in action description.
        assert!(params.get("oneOf").is_none());
    }

    #[tokio::test]
    async fn test_execute_without_callback() {
        let tool = SpawnTool::new();
        let mut params = HashMap::new();
        params.insert(
            "task".to_string(),
            serde_json::Value::String("do something".to_string()),
        );
        let result = tool.execute(params).await;
        assert!(result.contains("Spawn callback not configured"));
    }

    #[tokio::test]
    async fn test_execute_missing_task() {
        let tool = SpawnTool::new();
        let params = HashMap::new();
        let result = tool.execute(params).await;
        assert!(result.contains("'task' parameter is required"));
    }

    #[tokio::test]
    async fn test_list_without_callback() {
        let tool = SpawnTool::new();
        let mut params = HashMap::new();
        params.insert(
            "action".to_string(),
            serde_json::Value::String("list".to_string()),
        );
        let result = tool.execute(params).await;
        assert!(result.contains("List callback not configured"));
    }

    #[tokio::test]
    async fn test_cancel_without_task_id() {
        let tool = SpawnTool::new();
        let mut params = HashMap::new();
        params.insert(
            "action".to_string(),
            serde_json::Value::String("cancel".to_string()),
        );
        let result = tool.execute(params).await;
        assert!(result.contains("'task_id' parameter is required"));
    }

    #[tokio::test]
    async fn test_cancel_without_callback() {
        let tool = SpawnTool::new();
        let mut params = HashMap::new();
        params.insert(
            "action".to_string(),
            serde_json::Value::String("cancel".to_string()),
        );
        params.insert(
            "task_id".to_string(),
            serde_json::Value::String("abc123".to_string()),
        );
        let result = tool.execute(params).await;
        assert!(result.contains("Cancel callback not configured"));
    }

    #[tokio::test]
    async fn test_list_with_mock_callback() {
        let tool = SpawnTool::new();
        let list_cb: ListCallback =
            Arc::new(|| Box::pin(async { "No subagents currently running.".to_string() }));
        tool.set_list_callback(list_cb).await;

        let mut params = HashMap::new();
        params.insert(
            "action".to_string(),
            serde_json::Value::String("list".to_string()),
        );
        let result = tool.execute(params).await;
        assert!(result.contains("No subagents"));
    }

    #[tokio::test]
    async fn test_cancel_with_mock_callback() {
        let tool = SpawnTool::new();
        let cancel_cb: CancelCallback =
            Arc::new(|id: String| Box::pin(async move { format!("Cancelled {}", id) }));
        tool.set_cancel_callback(cancel_cb).await;

        let mut params = HashMap::new();
        params.insert(
            "action".to_string(),
            serde_json::Value::String("cancel".to_string()),
        );
        params.insert(
            "task_id".to_string(),
            serde_json::Value::String("abc123".to_string()),
        );
        let result = tool.execute(params).await;
        assert!(result.contains("Cancelled abc123"));
    }

    #[tokio::test]
    async fn test_set_context() {
        let tool = SpawnTool::new();
        tool.set_context("discord", "guild_123").await;

        let mut params = HashMap::new();
        params.insert(
            "task".to_string(),
            serde_json::Value::String("test task".to_string()),
        );
        let result = tool.execute(params).await;
        assert!(result.contains("Spawn callback not configured"));
    }

    #[tokio::test]
    async fn test_execute_with_mock_callback() {
        let tool = SpawnTool::new();

        let callback: SpawnCallback = Arc::new(
            |task: String,
             label: Option<String>,
             agent: Option<String>,
             model: Option<String>,
             channel: String,
             chat_id: String,
             working_dir: Option<String>| {
                Box::pin(async move {
                    format!(
                        "spawned: task={}, label={}, agent={}, model={}, channel={}, chat_id={}, working_dir={}",
                        task,
                        label.unwrap_or_else(|| "none".to_string()),
                        agent.unwrap_or_else(|| "none".to_string()),
                        model.unwrap_or_else(|| "none".to_string()),
                        channel,
                        chat_id,
                        working_dir.unwrap_or_else(|| "none".to_string()),
                    )
                })
            },
        );
        tool.set_callback(callback).await;
        tool.set_context("telegram", "42").await;

        let mut params = HashMap::new();
        params.insert(
            "task".to_string(),
            serde_json::Value::String("analyze data".to_string()),
        );
        params.insert(
            "label".to_string(),
            serde_json::Value::String("data-analysis".to_string()),
        );
        params.insert(
            "agent".to_string(),
            serde_json::Value::String("explore".to_string()),
        );
        params.insert(
            "model".to_string(),
            serde_json::Value::String("haiku".to_string()),
        );
        let result = tool.execute(params).await;
        assert!(result.contains("spawned:"));
        assert!(result.contains("task=analyze data"));
        assert!(result.contains("label=data-analysis"));
        assert!(result.contains("agent=explore"));
        assert!(result.contains("model=haiku"));
        assert!(result.contains("channel=telegram"));
        assert!(result.contains("chat_id=42"));
    }

    #[tokio::test]
    async fn test_execute_with_callback_no_optional_params() {
        let tool = SpawnTool::new();

        let callback: SpawnCallback = Arc::new(
            |task: String,
             label: Option<String>,
             agent: Option<String>,
             model: Option<String>,
             _channel: String,
             _chat_id: String,
             _working_dir: Option<String>| {
                Box::pin(async move {
                    format!(
                        "task={}, has_label={}, has_agent={}, has_model={}",
                        task,
                        label.is_some(),
                        agent.is_some(),
                        model.is_some(),
                    )
                })
            },
        );
        tool.set_callback(callback).await;

        let mut params = HashMap::new();
        params.insert(
            "task".to_string(),
            serde_json::Value::String("simple task".to_string()),
        );
        let result = tool.execute(params).await;
        assert!(result.contains("task=simple task"));
        assert!(result.contains("has_label=false"));
        assert!(result.contains("has_agent=false"));
        assert!(result.contains("has_model=false"));
    }

    #[tokio::test]
    async fn test_execute_with_working_dir() {
        let tool = SpawnTool::new();

        let callback: SpawnCallback = Arc::new(
            |_task: String,
             _label: Option<String>,
             _agent: Option<String>,
             _model: Option<String>,
             _channel: String,
             _chat_id: String,
             working_dir: Option<String>| {
                Box::pin(async move {
                    format!(
                        "working_dir={}",
                        working_dir.unwrap_or_else(|| "none".to_string()),
                    )
                })
            },
        );
        tool.set_callback(callback).await;

        // With working_dir
        let mut params = HashMap::new();
        params.insert("task".to_string(), serde_json::json!("build project"));
        params.insert("working_dir".to_string(), serde_json::json!("/tmp/project"));
        let result = tool.execute(params).await;
        assert!(result.contains("working_dir=/tmp/project"));

        // Without working_dir
        let mut params = HashMap::new();
        params.insert("task".to_string(), serde_json::json!("build project"));
        let result = tool.execute(params).await;
        assert!(result.contains("working_dir=none"));
    }
}
