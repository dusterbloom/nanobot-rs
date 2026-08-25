//! Spawn tool for creating background subagents.
//!
//! Supports named agent profiles and model overrides for context-efficient
//! delegation. Also supports listing running subagents and cancelling them.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;

use super::base::{PermissionLevel, Tool, ToolContext, ToolOutput, ToolResult};
use crate::agent::host_bridge::{
    CancelRequest, CheckRequest, HostBridge, HostRequest, ListSubagentsRequest, LoopRequest,
    PipelineRequest, SpawnRequest, WaitRequest,
};
use crate::errors::ToolError;

/// Tool to spawn a subagent for background task execution.
///
/// The subagent runs asynchronously and announces its result back
/// to the main agent when complete. Supports named agent profiles
/// for specialized behavior and model overrides for cost control.
///
/// Also supports `action: "list"` to check running subagents and
/// `action: "cancel"` to abort a stuck subagent.
///
/// Depends only on [`HostBridge`] (research §3.3 DIP): the production host
/// (`AgentHost` in `tool_wiring`) and test mocks are interchangeable. Origin
/// channel/chat are baked into the host at the registry boundary — the tool
/// never carries them (doc §3.2/§3.3).
pub struct SpawnTool {
    host: Arc<dyn HostBridge>,
}

impl SpawnTool {
    /// Create a new spawn tool wired to a typed host bridge.
    pub fn new(host: Arc<dyn HostBridge>) -> Self {
        Self { host }
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

    fn permission(&self) -> PermissionLevel {
        PermissionLevel::System
    }

    fn description(&self) -> &str {
        "Run a background task, list tasks, check results, wait, or cancel.\n\
         spawn: start task (needs 'task'). list: show all tasks. check: get result or running status (needs 'task_id').\n\
         wait: block until done (needs 'task_id'). cancel: abort task (needs 'task_id').\n\
         Example: {\"action\": \"spawn\", \"task\": \"search for all TODO comments\"}"
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["spawn", "list", "check", "wait", "cancel"],
                    "description": "spawn=start new task, list=show tasks, check=get completed result or running status, wait=block until done, cancel=abort"
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

/// Typed spawn action — the parse output of the tool (research §3.3 SRP:
/// arg extraction + action routing lives here; invocation lives in the host).
///
/// The wire mapping into [`HostRequest`] is one-to-one (see
/// [`From<SpawnAction> for HostRequest`]); origin channel/chat are filled
/// with empty placeholders and baked in by the host.
#[derive(Debug, Clone, PartialEq)]
pub enum SpawnAction {
    /// `spawn` (the default action).
    Spawn {
        task: String,
        label: Option<String>,
        /// `profile` wins over `agent` when both are given (legacy
        /// resolution); unknown-name errors are the host's, not the tool's.
        profile: Option<String>,
        model: Option<String>,
        working_dir: Option<String>,
    },
    /// `action: "list"` — no payload.
    List,
    /// `action: "check"` — non-blocking result lookup by task-id prefix.
    Check { task_id: String },
    /// `action: "cancel"` — abort a running subagent by task-id prefix.
    Cancel { task_id: String },
    /// `action: "wait"` — block until completion (default timeout 120s).
    Wait { task_id: String, timeout_secs: u64 },
    /// `action: "pipeline"` — multi-step pipeline (MAKER voting).
    Pipeline {
        steps: Vec<serde_json::Value>,
        ahead_by_k: usize,
    },
    /// `action: "loop"` — autonomous refinement loop (default 5 rounds).
    Loop {
        task: String,
        max_rounds: u32,
        tools: Option<Vec<String>>,
        stop_condition: Option<String>,
        model: Option<String>,
        working_dir: Option<String>,
    },
}

/// Parse outcome: a typed action, a typed error, or — one legacy quirk — a
/// success text. The pipeline `steps` parse reproduces the legacy callback's
/// two-step (serialize → re-parse), which reported a malformed (non-array)
/// `steps` value as a *success* carrying `"Error parsing pipeline steps:
/// {serde}"` — no `"Error: "` prefix. That string is model-visible, so it
/// survives unchanged.
pub(crate) enum ParseError {
    Tool(ToolError),
    LegacyText(String),
}

impl From<ToolError> for ParseError {
    fn from(e: ToolError) -> Self {
        ParseError::Tool(e)
    }
}

impl SpawnAction {
    /// Parse tool params into a typed action. Every error string is the
    /// exact legacy one, so the model sees byte-identical output.
    pub(crate) fn parse(params: &HashMap<String, serde_json::Value>) -> Result<Self, ParseError> {
        let action = params
            .get("action")
            .and_then(|v| v.as_str())
            .unwrap_or("spawn");
        match action {
            "list" => Ok(SpawnAction::List),
            "check" => Ok(SpawnAction::Check {
                task_id: required_str(params, "task_id", "check")?,
            }),
            "cancel" => Ok(SpawnAction::Cancel {
                task_id: required_str(params, "task_id", "cancel")?,
            }),
            "wait" => Ok(SpawnAction::Wait {
                task_id: required_str(params, "task_id", "wait")?,
                timeout_secs: params
                    .get("timeout")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(120),
            }),
            "pipeline" => {
                let steps_value = params.get("steps").ok_or_else(|| ToolError::Execution {
                    message: "'steps' parameter is required for pipeline".to_string(),
                })?;
                // Legacy two-step: serialize the raw value, then re-parse it
                // as a sequence. A non-array value (string/map) failed the
                // re-parse in the old callback and came back as the success
                // text "Error parsing pipeline steps: {serde error}" — no
                // "Error:" prefix. Reproduced verbatim.
                let steps_json =
                    serde_json::to_string(steps_value).unwrap_or_else(|_| "[]".to_string());
                let steps: Vec<serde_json::Value> = match serde_json::from_str(&steps_json) {
                    Ok(s) => s,
                    Err(e) => {
                        return Err(ParseError::LegacyText(format!(
                            "Error parsing pipeline steps: {e}"
                        )))
                    }
                };
                let ahead_by_k = params
                    .get("ahead_by_k")
                    .and_then(|v| v.as_u64())
                    .map(|v| usize::try_from(v).unwrap_or(usize::MAX))
                    .unwrap_or(0);
                Ok(SpawnAction::Pipeline { steps, ahead_by_k })
            }
            "loop" => Ok(SpawnAction::Loop {
                task: required_str(params, "task", "loop")?,
                max_rounds: params
                    .get("max_rounds")
                    .and_then(|v| v.as_u64())
                    .map(|v| u32::try_from(v).unwrap_or(u32::MAX))
                    .unwrap_or(5),
                tools: params.get("tools").and_then(|v| v.as_array()).map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect()
                }),
                stop_condition: params
                    .get("stop_condition")
                    .and_then(|v| v.as_str())
                    .map(str::to_string),
                model: params
                    .get("model")
                    .and_then(|v| v.as_str())
                    .map(str::to_string),
                working_dir: params
                    .get("working_dir")
                    .and_then(|v| v.as_str())
                    .map(str::to_string),
            }),
            "spawn" | _ => Ok(SpawnAction::Spawn {
                task: required_str(params, "task", "spawn")?,
                label: params
                    .get("label")
                    .and_then(|v| v.as_str())
                    .map(str::to_string),
                // 'profile' and 'agent' are synonyms; 'profile' wins when both
                // are given (legacy resolution, unchanged).
                profile: params
                    .get("profile")
                    .or_else(|| params.get("agent"))
                    .and_then(|v| v.as_str())
                    .map(str::to_string),
                model: params
                    .get("model")
                    .and_then(|v| v.as_str())
                    .map(str::to_string),
                working_dir: params
                    .get("working_dir")
                    .and_then(|v| v.as_str())
                    .map(str::to_string),
            }),
        }
    }
}

/// Legacy-identical required-param error. The default `spawn` action reads
/// `'task' parameter is required` (no suffix); the routed actions read
/// `'task_id' parameter is required for check` etc. — byte-stable.
fn required_str(
    params: &HashMap<String, serde_json::Value>,
    key: &str,
    action: &str,
) -> Result<String, ToolError> {
    params
        .get(key)
        .and_then(|v| v.as_str())
        .map(str::to_string)
        .ok_or_else(|| {
            let message = if action == "spawn" {
                format!("'{key}' parameter is required")
            } else {
                format!("'{key}' parameter is required for {action}")
            };
            ToolError::Execution { message }
        })
}

impl From<SpawnAction> for HostRequest {
    fn from(action: SpawnAction) -> Self {
        match action {
            SpawnAction::Spawn {
                task,
                label,
                profile,
                model,
                working_dir,
            } => {
                HostRequest::Spawn(SpawnRequest {
                    task,
                    label,
                    profile,
                    model,
                    working_dir,
                    // Origin channel/chat are baked into the host at the
                    // registry boundary (doc §3.2/§3.3); the tool never
                    // carries them, and they never travel on the wire.
                    channel: String::new(),
                    chat_id: String::new(),
                })
            }
            SpawnAction::List => HostRequest::ListSubagents(ListSubagentsRequest {}),
            SpawnAction::Check { task_id } => HostRequest::CheckSubagent(CheckRequest { task_id }),
            SpawnAction::Cancel { task_id } => {
                HostRequest::CancelSubagent(CancelRequest { task_id })
            }
            SpawnAction::Wait {
                task_id,
                timeout_secs,
            } => HostRequest::WaitSubagent(WaitRequest {
                task_id,
                timeout_secs,
            }),
            SpawnAction::Pipeline { steps, ahead_by_k } => {
                HostRequest::RunPipeline(PipelineRequest { steps, ahead_by_k })
            }
            SpawnAction::Loop {
                task,
                max_rounds,
                tools,
                stop_condition,
                model,
                working_dir,
            } => HostRequest::RunLoop(LoopRequest {
                task,
                max_rounds,
                tools,
                stop_condition,
                model,
                working_dir,
            }),
        }
    }
}

#[async_trait]
impl Tool for SpawnTool {
    fn name(&self) -> &str {
        "spawn"
    }

    fn permission(&self) -> PermissionLevel {
        PermissionLevel::System
    }

    fn description(&self) -> &str {
        "Run background subagents: spawn a task, list running/completed ones, check/wait/cancel by \
         task_id, or run a multi-step pipeline or refinement loop (see 'action'). \
         Use 'agent' to pick a specialized profile and 'model' to control cost."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "'spawn' (default) and 'loop' require 'task'; 'check'/'wait'/'cancel' require 'task_id'; 'pipeline' requires 'steps'",
                    "enum": ["spawn", "list", "check", "wait", "cancel", "pipeline", "loop"]
                },
                "steps": {
                    "type": "array",
                    "description": "Pipeline steps (pipeline)",
                    "items": {
                        "type": "object",
                        "properties": {
                            "prompt": { "type": "string", "description": "Step prompt" },
                            "expected": { "type": "string", "description": "Expected answer for verification" },
                            "tools": {
                                "type": "array",
                                "items": { "type": "string" },
                                "description": "Allowed tool names; omit for text-only"
                            },
                            "max_iterations": {
                                "type": "integer",
                                "description": "Max tool iterations (default: 5)"
                            }
                        }
                    }
                },
                "ahead_by_k": {
                    "type": "integer",
                    "description": "MAKER voting margin (pipeline). 0 = no voting (default)"
                },
                "task": {
                    "type": "string",
                    "description": "Task description (spawn/loop)"
                },
                "task_id": {
                    "type": "string",
                    "description": "Task ID or prefix (check/wait/cancel)"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Wait timeout in seconds (default: 120)"
                },
                "label": {
                    "type": "string",
                    "description": "Short display label"
                },
                "agent": {
                    "type": "string",
                    "description": "Profile: 'explore', 'reviewer', 'builder', 'researcher'. Omit for general-purpose."
                },
                "profile": {
                    "type": "string",
                    "description": "Named agent profile — available profiles are listed in your system prompt under 'Subagent Profiles'. Explicit 'model'/'tools' params override the profile's values. Unknown names return an error listing valid profiles."
                },
                "model": {
                    "type": "string",
                    "description": "Model override: 'haiku'/'sonnet'/'opus'/'local', or provider-prefixed (e.g. 'groq/llama-3.3-70b-versatile'). Omit for profile/parent default."
                },
                "working_dir": {
                    "type": "string",
                    "description": "Subagent exec working directory (default: workspace)"
                },
                "max_rounds": {
                    "type": "integer",
                    "description": "Max loop rounds (default: 5)"
                },
                "tools": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Allowed tool names (loop)"
                },
                "stop_condition": {
                    "type": "string",
                    "description": "Loop stops when output contains this text or 'DONE'"
                }
            },
            "required": []
        })
    }

    /// Typed entry point: parses the params into a typed [`SpawnAction`]
    /// (every validation error string byte-identical to the legacy surface)
    /// and drives the request through the host bridge — one call, no locks,
    /// no `Option<callback>` slots (research §3.4). Origin channel/chat are
    /// baked into the host; the tool sends empty placeholders that never
    /// travel on the wire (`#[serde(skip)]`). The trait's default String
    /// `execute` renders this byte-for-byte.
    async fn execute_typed(
        &self,
        params: HashMap<String, serde_json::Value>,
        _ctx: &ToolContext,
    ) -> ToolResult {
        let action = match SpawnAction::parse(&params) {
            Ok(action) => action,
            Err(ParseError::Tool(e)) => return Err(e),
            // Legacy quirk preserved exactly: a malformed (non-array)
            // pipeline `steps` value made the old callback return
            // "Error parsing pipeline steps: …" as a *success* — no "Error:"
            // prefix. Emit verbatim.
            Err(ParseError::LegacyText(text)) => return Ok(ToolOutput { text }),
        };
        let reply = self.host.call(action.into()).await?;
        let text = reply
            .get("text")
            .and_then(|v| v.as_str())
            .map(str::to_string)
            .ok_or_else(|| ToolError::Execution {
                message: "host reply missing 'text'".to_string(),
            })?;
        Ok(ToolOutput { text })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::host_bridge::{
        CancelReply, CheckReply, HostDispatcher, ListReply, LoopHost, LoopReply, MessageHost,
        PipelineHost, PipelineReply, SendMessageReply, SendMessageRequest, SpawnHost, SpawnReply,
        WaitReply,
    };
    use serde_json::json;

    /// Test host that echoes every request back into the model-visible text,
    /// so tests can assert exactly what the tool forwarded to the bridge.
    struct EchoHost;

    #[async_trait]
    impl SpawnHost for EchoHost {
        async fn spawn(&self, req: SpawnRequest) -> Result<SpawnReply, ToolError> {
            Ok(SpawnReply {
                task_id: "abc12345".to_string(),
                text: format!(
                    "spawned: task={}, label={}, profile={}, model={}, channel={}, chat_id={}, working_dir={}",
                    req.task,
                    req.label.unwrap_or_else(|| "none".to_string()),
                    req.profile.unwrap_or_else(|| "none".to_string()),
                    req.model.unwrap_or_else(|| "none".to_string()),
                    req.channel,
                    req.chat_id,
                    req.working_dir.unwrap_or_else(|| "none".to_string()),
                ),
            })
        }
        async fn list_subagents(&self) -> Result<ListReply, ToolError> {
            Ok(ListReply {
                text: "No subagents currently running.".to_string(),
            })
        }
        async fn cancel(&self, req: CancelRequest) -> Result<CancelReply, ToolError> {
            Ok(CancelReply {
                text: format!("Cancelled {}", req.task_id),
            })
        }
        async fn wait(&self, req: WaitRequest) -> Result<WaitReply, ToolError> {
            Ok(WaitReply {
                text: format!("waited {} {}s", req.task_id, req.timeout_secs),
            })
        }
        async fn check(&self, req: CheckRequest) -> Result<CheckReply, ToolError> {
            Ok(CheckReply {
                text: format!("checking {}", req.task_id),
            })
        }
    }

    #[async_trait]
    impl PipelineHost for EchoHost {
        async fn run_pipeline(&self, req: PipelineRequest) -> Result<PipelineReply, ToolError> {
            let steps = serde_json::to_string(&req.steps).unwrap_or_default();
            Ok(PipelineReply {
                text: format!("pipeline {steps} ahead {}", req.ahead_by_k),
            })
        }
    }

    #[async_trait]
    impl LoopHost for EchoHost {
        async fn run_loop(&self, req: LoopRequest) -> Result<LoopReply, ToolError> {
            Ok(LoopReply {
                text: format!("loop {} r{}", req.task, req.max_rounds),
            })
        }
    }

    #[async_trait]
    impl MessageHost for EchoHost {
        async fn send(&self, req: SendMessageRequest) -> Result<SendMessageReply, ToolError> {
            Ok(SendMessageReply {
                text: format!("sent {}:{}", req.channel, req.chat_id),
            })
        }
    }

    /// Test host whose capability methods all fail with a typed error, used
    /// to prove host errors propagate through the bridge unchanged.
    struct FailingHost;

    #[async_trait]
    impl SpawnHost for FailingHost {
        async fn spawn(&self, _req: SpawnRequest) -> Result<SpawnReply, ToolError> {
            Err(ToolError::Execution {
                message: "spawn exploded".to_string(),
            })
        }
        async fn list_subagents(&self) -> Result<ListReply, ToolError> {
            Err(ToolError::Execution {
                message: "list exploded".to_string(),
            })
        }
        async fn cancel(&self, _req: CancelRequest) -> Result<CancelReply, ToolError> {
            Err(ToolError::Execution {
                message: "cancel exploded".to_string(),
            })
        }
        async fn wait(&self, _req: WaitRequest) -> Result<WaitReply, ToolError> {
            Err(ToolError::Execution {
                message: "wait exploded".to_string(),
            })
        }
        async fn check(&self, _req: CheckRequest) -> Result<CheckReply, ToolError> {
            Err(ToolError::Execution {
                message: "check exploded".to_string(),
            })
        }
    }

    #[async_trait]
    impl PipelineHost for FailingHost {
        async fn run_pipeline(&self, _req: PipelineRequest) -> Result<PipelineReply, ToolError> {
            Err(ToolError::Execution {
                message: "pipeline exploded".to_string(),
            })
        }
    }

    #[async_trait]
    impl LoopHost for FailingHost {
        async fn run_loop(&self, _req: LoopRequest) -> Result<LoopReply, ToolError> {
            Err(ToolError::Execution {
                message: "loop exploded".to_string(),
            })
        }
    }

    #[async_trait]
    impl MessageHost for FailingHost {
        async fn send(&self, _req: SendMessageRequest) -> Result<SendMessageReply, ToolError> {
            Err(ToolError::Execution {
                message: "send exploded".to_string(),
            })
        }
    }

    /// The production composition: a dispatcher over a single mock host.
    fn test_host(host: impl HostBridge + 'static) -> Arc<dyn HostBridge> {
        let host = Arc::new(host);
        Arc::new(HostDispatcher::new(
            host.clone(),
            host.clone(),
            host.clone(),
            host.clone(),
        ))
    }

    #[test]
    fn test_spawn_tool_name() {
        let tool = SpawnTool::new(test_host(EchoHost));
        assert_eq!(tool.name(), "spawn");
    }

    #[test]
    fn test_spawn_tool_description() {
        let tool = SpawnTool::new(test_host(EchoHost));
        let desc = tool.description();
        assert!(!desc.is_empty());
        assert!(desc.contains("subagent") || desc.contains("background"));
        assert!(desc.contains("list"));
        assert!(desc.contains("cancel"));
    }

    #[test]
    fn test_spawn_tool_parameters() {
        let tool = SpawnTool::new(test_host(EchoHost));
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

    #[test]
    fn test_spawn_tool_parameters_contains_profile() {
        let tool = SpawnTool::new(test_host(EchoHost));
        let params = tool.parameters();
        assert!(params["properties"]["profile"].is_object());
        let desc = params["properties"]["profile"]["description"]
            .as_str()
            .unwrap();
        // The model must learn where profiles are listed and what overrides them.
        assert!(desc.to_lowercase().contains("system prompt"));
        assert!(desc.contains("model"));
    }

    #[tokio::test]
    async fn test_profile_param_routes_to_profile_slot() {
        let tool = SpawnTool::new(test_host(EchoHost));
        let mut params = HashMap::new();
        params.insert("task".to_string(), json!("investigate"));
        params.insert("profile".to_string(), json!("researcher"));
        let result = tool.execute(params).await;
        assert!(result.contains("profile=researcher"));
        // 'agent' is a synonym; 'profile' wins when both are given.
        let mut params = HashMap::new();
        params.insert("task".to_string(), json!("investigate"));
        params.insert("profile".to_string(), json!("researcher"));
        params.insert("agent".to_string(), json!("explore"));
        let result = tool.execute(params).await;
        assert!(result.contains("profile=researcher"));
    }

    #[tokio::test]
    async fn test_execute_missing_task() {
        let tool = SpawnTool::new(test_host(EchoHost));
        let params = HashMap::new();
        let result = tool.execute(params).await;
        assert_eq!(result, "Error: 'task' parameter is required");
    }

    #[tokio::test]
    async fn test_cancel_without_task_id() {
        let tool = SpawnTool::new(test_host(EchoHost));
        let mut params = HashMap::new();
        params.insert("action".to_string(), json!("cancel"));
        let result = tool.execute(params).await;
        assert_eq!(result, "Error: 'task_id' parameter is required for cancel");
    }

    #[tokio::test]
    async fn test_wait_without_task_id() {
        let tool = SpawnTool::new(test_host(EchoHost));
        let mut params = HashMap::new();
        params.insert("action".to_string(), json!("wait"));
        let result = tool.execute(params).await;
        assert_eq!(result, "Error: 'task_id' parameter is required for wait");
    }

    #[tokio::test]
    async fn test_pipeline_without_steps() {
        let tool = SpawnTool::new(test_host(EchoHost));
        let mut params = HashMap::new();
        params.insert("action".to_string(), json!("pipeline"));
        let result = tool.execute(params).await;
        assert_eq!(result, "Error: 'steps' parameter is required for pipeline");
    }

    #[tokio::test]
    async fn test_loop_without_task() {
        let tool = SpawnTool::new(test_host(EchoHost));
        let mut params = HashMap::new();
        params.insert("action".to_string(), json!("loop"));
        let result = tool.execute(params).await;
        assert_eq!(result, "Error: 'task' parameter is required for loop");
    }

    #[tokio::test]
    async fn test_host_errors_propagate_through_the_bridge() {
        let tool = SpawnTool::new(test_host(FailingHost));
        let mut params = HashMap::new();
        params.insert("task".to_string(), json!("t"));
        let result = tool.execute(params).await;
        assert_eq!(result, "Error: spawn exploded");

        let mut params = HashMap::new();
        params.insert("action".to_string(), json!("list"));
        let result = tool.execute(params).await;
        assert_eq!(result, "Error: list exploded");
    }

    #[tokio::test]
    async fn test_list_forwards_to_host() {
        let tool = SpawnTool::new(test_host(EchoHost));
        let mut params = HashMap::new();
        params.insert("action".to_string(), json!("list"));
        let result = tool.execute(params).await;
        assert_eq!(result, "No subagents currently running.");
    }

    #[tokio::test]
    async fn test_cancel_forwards_to_host() {
        let tool = SpawnTool::new(test_host(EchoHost));
        let mut params = HashMap::new();
        params.insert("action".to_string(), json!("cancel"));
        params.insert("task_id".to_string(), json!("abc123"));
        let result = tool.execute(params).await;
        assert_eq!(result, "Cancelled abc123");
    }

    #[tokio::test]
    async fn test_check_forwards_to_host() {
        let tool = SpawnTool::new(test_host(EchoHost));
        let mut params = HashMap::new();
        params.insert("action".to_string(), json!("check"));
        params.insert("task_id".to_string(), json!("abc123"));
        let result = tool.execute(params).await;
        assert_eq!(result, "checking abc123");
    }

    #[tokio::test]
    async fn test_execute_forwards_full_params() {
        let tool = SpawnTool::new(test_host(EchoHost));
        let mut params = HashMap::new();
        params.insert("task".to_string(), json!("analyze data"));
        params.insert("label".to_string(), json!("data-analysis"));
        params.insert("profile".to_string(), json!("explore"));
        params.insert("model".to_string(), json!("haiku"));
        params.insert("working_dir".to_string(), json!("/tmp/project"));
        let result = tool.execute(params).await;
        assert!(result.contains("task=analyze data"));
        assert!(result.contains("label=data-analysis"));
        assert!(result.contains("profile=explore"));
        assert!(result.contains("model=haiku"));
        assert!(result.contains("working_dir=/tmp/project"));
        // Origin channel/chat are baked into the host — the tool never
        // carries them (empty placeholders on the request).
        assert!(result.contains("channel=, chat_id="));
    }

    #[tokio::test]
    async fn test_execute_without_optional_params() {
        let tool = SpawnTool::new(test_host(EchoHost));
        let mut params = HashMap::new();
        params.insert("task".to_string(), json!("simple task"));
        let result = tool.execute(params).await;
        assert!(result.contains("task=simple task"));
        assert!(result.contains("label=none"));
        assert!(result.contains("profile=none"));
        assert!(result.contains("model=none"));
    }

    #[tokio::test]
    async fn test_pipeline_forwards_steps_and_ahead_by_k() {
        let tool = SpawnTool::new(test_host(EchoHost));
        let mut params = HashMap::new();
        params.insert("action".to_string(), json!("pipeline"));
        params.insert("steps".to_string(), json!([{"prompt": "one"}]));
        params.insert("ahead_by_k".to_string(), json!(2));
        let result = tool.execute(params).await;
        assert_eq!(result, r#"pipeline [{"prompt":"one"}] ahead 2"#);
    }

    #[tokio::test]
    async fn test_loop_forwards_task_and_rounds() {
        let tool = SpawnTool::new(test_host(EchoHost));
        let mut params = HashMap::new();
        params.insert("action".to_string(), json!("loop"));
        params.insert("task".to_string(), json!("refine"));
        params.insert("max_rounds".to_string(), json!(3));
        let result = tool.execute(params).await;
        assert_eq!(result, "loop refine r3");
    }

    #[tokio::test]
    async fn test_wait_forwards_timeout() {
        let tool = SpawnTool::new(test_host(EchoHost));
        let mut params = HashMap::new();
        params.insert("action".to_string(), json!("wait"));
        params.insert("task_id".to_string(), json!("abc123"));
        let result = tool.execute(params).await;
        // No timeout param → legacy default of 120s.
        assert_eq!(result, "waited abc123 120s");

        let mut params = HashMap::new();
        params.insert("action".to_string(), json!("wait"));
        params.insert("task_id".to_string(), json!("abc123"));
        params.insert("timeout".to_string(), json!(10));
        let result = tool.execute(params).await;
        assert_eq!(result, "waited abc123 10s");
    }

    #[tokio::test]
    async fn test_malformed_pipeline_steps_reports_legacy_text() {
        // Legacy quirk: a non-array `steps` value came back from the old
        // callback as a *success* carrying "Error parsing pipeline steps: …"
        // (no "Error:" prefix). The bridge must reproduce it exactly.
        let tool = SpawnTool::new(test_host(EchoHost));
        let mut params = HashMap::new();
        params.insert("action".to_string(), json!("pipeline"));
        params.insert("steps".to_string(), json!("not-an-array"));
        let result = tool.execute(params).await;
        assert_eq!(
            result,
            "Error parsing pipeline steps: invalid type: string \"not-an-array\", expected a sequence at line 1 column 14"
        );
    }

    #[test]
    fn spawn_action_wire_round_trip() {
        // The typed action maps onto the exact wire request.
        let action = SpawnAction::Spawn {
            task: "t".to_string(),
            label: None,
            profile: Some("explore".to_string()),
            model: None,
            working_dir: None,
        };
        let wire = serde_json::to_string(&HostRequest::from(action)).unwrap();
        assert!(wire.contains(r#""type":"spawn""#));
        assert!(wire.contains(r#""task":"t""#));
        assert!(wire.contains(r#""profile":"explore""#));
        // Origin never travels on the wire.
        assert!(!wire.contains("channel"));
        assert!(!wire.contains("chat_id"));
    }
}
