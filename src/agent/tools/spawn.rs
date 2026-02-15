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
/// Arguments: (task, label, agent_name, model_override, origin_channel, origin_chat_id) -> result string.
pub type SpawnCallback = Arc<
    dyn Fn(
            String,
            Option<String>,
            Option<String>,
            Option<String>,
            String,
            String,
        ) -> Pin<Box<dyn Future<Output = String> + Send>>
        + Send
        + Sync,
>;

/// Type alias for the list callback. Returns formatted list of running subagents.
pub type ListCallback = Arc<
    dyn Fn() -> Pin<Box<dyn Future<Output = String> + Send>> + Send + Sync,
>;

/// Type alias for the cancel callback. Takes task_id prefix, returns success message.
pub type CancelCallback = Arc<
    dyn Fn(String) -> Pin<Box<dyn Future<Output = String> + Send>> + Send + Sync,
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
}

impl Default for SpawnTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for SpawnTool {
    fn name(&self) -> &str {
        "spawn"
    }

    fn description(&self) -> &str {
        "Spawn a subagent to handle a task in the background, list running subagents, or cancel one. \
         Use action='spawn' (default) to start a new subagent. \
         Use action='list' to check status of running subagents. \
         Use action='cancel' with task_id to abort a stuck subagent. \
         Use 'agent' to pick a specialized profile (explore, reviewer, builder, researcher) \
         and 'model' to control cost (e.g. 'haiku' for cheap/fast tasks)."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Action to perform: 'spawn' (default), 'list', or 'cancel'",
                    "enum": ["spawn", "list", "cancel"]
                },
                "task": {
                    "type": "string",
                    "description": "The task for the subagent to complete (required for action='spawn')"
                },
                "task_id": {
                    "type": "string",
                    "description": "Task ID or prefix to cancel (required for action='cancel')"
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

                let channel = self.origin_channel.lock().await.clone();
                let chat_id = self.origin_chat_id.lock().await.clone();

                let callback_guard = self.spawn_callback.lock().await;
                let callback = match callback_guard.as_ref() {
                    Some(cb) => cb.clone(),
                    None => return "Error: Spawn callback not configured".to_string(),
                };
                // Drop the lock before awaiting.
                drop(callback_guard);

                callback(task, label, agent, model, channel, chat_id).await
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
        let list_cb: ListCallback = Arc::new(|| {
            Box::pin(async { "No subagents currently running.".to_string() })
        });
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
        let cancel_cb: CancelCallback = Arc::new(|id: String| {
            Box::pin(async move { format!("Cancelled {}", id) })
        });
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
             chat_id: String| {
                Box::pin(async move {
                    format!(
                        "spawned: task={}, label={}, agent={}, model={}, channel={}, chat_id={}",
                        task,
                        label.unwrap_or_else(|| "none".to_string()),
                        agent.unwrap_or_else(|| "none".to_string()),
                        model.unwrap_or_else(|| "none".to_string()),
                        channel,
                        chat_id,
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
             _chat_id: String| {
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
}
