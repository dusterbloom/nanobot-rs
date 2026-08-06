// Error-protocol layer-3 backlog (docs/research/2026-08-06-error-conventions-and-host-bridge.md §3.6):
// the deny regime in Cargo.toml is live; this module still carries pre-existing
// violations of the lints below. Remove this allow as the module migrates onto
// the regime.
// Tracking: docs/error-protocol-backlog.md
#![allow(
    clippy::as_conversions,
    clippy::format_push_string,
    clippy::shadow_reuse,
    clippy::shadow_unrelated,
)]
//! Tool registry construction and wiring.
//!
//! Extracted from `agent_loop.rs` to isolate the callback-heavy tool setup.

#![allow(clippy::disallowed_types)] // anyhow is the app convention — the ban targets tool boundaries (error protocol §2.5)
use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::mpsc::UnboundedSender;

use crate::agent::agent_core::SwappableCore;
use crate::agent::agent_loop::AgentLoopShared;
use crate::agent::host_bridge::{
    CancelReply, CancelRequest, CheckReply, CheckRequest, HostBridge, HostDispatcher, ListReply,
    LoopHost, LoopReply, LoopRequest, MessageHost, PipelineHost, PipelineReply, PipelineRequest,
    SendMessageReply, SendMessageRequest, SpawnHost, SpawnReply, SpawnRequest, WaitReply,
    WaitRequest,
};
use crate::agent::pipeline;
use crate::agent::policy;
use crate::agent::subagent::SubagentManager;
use crate::agent::tools::registry::{ToolConfig, ToolRegistry};
use crate::agent::tools::{
    CheckInboxTool, CronScheduleTool, MessageTool, SendEmailTool, SpawnTool, SpawnToolLite,
    TodoTool,
};
use crate::bus::events::OutboundMessage;
use crate::errors::ToolError;
use crate::providers::base::LLMProvider;

/// Generic proxy that delegates the [`Tool`] trait to an `Arc<T>`.
pub(crate) struct ArcToolProxy<T: crate::agent::tools::Tool>(pub(crate) Arc<T>);

#[async_trait::async_trait]
impl<T: crate::agent::tools::Tool> crate::agent::tools::Tool for ArcToolProxy<T> {
    fn name(&self) -> &str {
        self.0.name()
    }
    fn description(&self) -> &str {
        self.0.description()
    }
    fn parameters(&self) -> Value {
        self.0.parameters()
    }
    fn concurrency(&self) -> crate::agent::tools::base::ToolConcurrency {
        self.0.concurrency()
    }
    async fn execute(&self, params: HashMap<String, Value>) -> String {
        self.0.execute(params).await
    }
}

fn should_register_message_tool(channel: &str) -> bool {
    !matches!(channel, "cli" | "voice" | "tui" | "repl")
}

/// The production typed host (research §3.3 DIP): one struct implementing the
/// four capability traits, injected as `Arc<dyn HostBridge>` at the registry
/// boundary. Each method is the byte-identical port of the legacy closure
/// body that used to live in `build_tools` — the logic is now a named,
/// testable type instead of 8 anonymous closures, and tools depend on the
/// trait, never on this struct or `SubagentManager`.
pub(crate) struct AgentHost {
    subagents: Arc<SubagentManager>,
    session_policies: Arc<tokio::sync::Mutex<HashMap<String, policy::SessionPolicy>>>,
    pipeline_provider: Arc<dyn LLMProvider>,
    pipeline_model: String,
    workspace: PathBuf,
    outbound: UnboundedSender<OutboundMessage>,
    channel: String,
    chat_id: String,
}

#[async_trait]
impl SpawnHost for AgentHost {
    async fn spawn(&self, req: SpawnRequest) -> Result<SpawnReply, ToolError> {
        // Origin channel/chat are baked into the host at the registry boundary
        // (research §3.2/§3.3): the request's `channel`/`chat_id` travel only
        // on a cross-process wire, never in-process.
        let key = format!("{}:{}", self.channel, self.chat_id);
        let policy = {
            let map = self.session_policies.lock().await;
            map.get(&key).cloned().unwrap_or_default()
        };
        let effective_model = policy::enforce_subagent_model(&policy, req.model);
        let out = self
            .subagents
            .spawn(
                req.task,
                req.label,
                req.profile,
                effective_model,
                self.channel.clone(),
                self.chat_id.clone(),
                req.working_dir,
            )
            .await;
        let text = crate::agent::host_bridge::into_model_text(out)?;
        let task_id = crate::agent::host_bridge::spawn_task_id(&text);
        Ok(SpawnReply { task_id, text })
    }

    async fn list_subagents(&self) -> Result<ListReply, ToolError> {
        let running = self.subagents.list_running().await;
        let mut out = String::new();

        // Running subagents
        if running.is_empty() {
            out.push_str("No subagents currently running.\n");
        } else {
            let _ = writeln!(out, "{} subagent(s) running:", running.len());
            for info in &running {
                let elapsed = info.started_at.elapsed().as_secs();
                let _ = writeln!(
                    out,
                    "  • {} (id: {}) — running for {}s",
                    info.label, info.task_id, elapsed
                );
            }
        }

        // Recently completed (from events.jsonl)
        let recent = SubagentManager::read_recent_completed(&self.workspace, 10);
        if !recent.is_empty() {
            let _ = writeln!(out, "\nRecently completed ({}):", recent.len());
            for entry in &recent {
                out.push_str(entry);
                out.push('\n');
            }
        }

        Ok(ListReply { text: out })
    }

    async fn cancel(&self, req: CancelRequest) -> Result<CancelReply, ToolError> {
        if self.subagents.cancel(&req.task_id).await {
            Ok(CancelReply {
                text: format!("Subagent '{}' cancelled.", req.task_id),
            })
        } else {
            Ok(CancelReply {
                text: format!("No running subagent found matching '{}'.", req.task_id),
            })
        }
    }

    async fn wait(&self, req: WaitRequest) -> Result<WaitReply, ToolError> {
        let timeout = std::time::Duration::from_secs(req.timeout_secs);
        let text = crate::agent::host_bridge::into_model_text(
            self.subagents.wait_for(&req.task_id, timeout).await,
        )?;
        Ok(WaitReply { text })
    }

    async fn check(&self, req: CheckRequest) -> Result<CheckReply, ToolError> {
        match SubagentManager::read_event_result(&self.workspace, &req.task_id) {
            Some(result) => Ok(CheckReply {
                text: crate::agent::host_bridge::into_model_text(result)?,
            }),
            None => {
                let running = self.subagents.list_running().await;
                if let Some(info) = running
                    .iter()
                    .find(|info| info.task_id.starts_with(&req.task_id))
                {
                    let elapsed = info.started_at.elapsed().as_secs();
                    Ok(CheckReply {
                        text: format!(
                            "Subagent '{}' ({}) is still running after {}s. Use action='wait' to block for completion, action='list' for all tasks, or action='cancel' to abort.",
                            info.label, info.task_id, elapsed
                        ),
                    })
                } else {
                    Ok(CheckReply {
                        text: format!(
                            "No running or completed result found for task_id '{}'.",
                            req.task_id
                        ),
                    })
                }
            }
        }
    }
}

#[async_trait]
impl PipelineHost for AgentHost {
    async fn run_pipeline(&self, req: PipelineRequest) -> Result<PipelineReply, ToolError> {
        let pipeline_steps: Vec<pipeline::PipelineStep> = req
            .steps
            .iter()
            .enumerate()
            .map(|(i, s)| pipeline::PipelineStep {
                index: i,
                prompt: s["prompt"].as_str().unwrap_or("").to_string(),
                expected: s["expected"].as_str().map(|s| s.to_string()),
                tools: s["tools"].as_array().map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect()
                }),
                max_iterations: s["max_iterations"].as_u64().map(|n| n as u32),
            })
            .collect();
        if pipeline_steps.is_empty() {
            // Byte-identical to the legacy pipeline callback's error string
            // ("Error: no valid pipeline steps provided.").
            return Err(ToolError::Execution {
                message: "no valid pipeline steps provided.".to_string(),
            });
        }
        let config = pipeline::PipelineConfig {
            pipeline_id: format!(
                "pipe-{}",
                chrono::Utc::now().timestamp_millis() % 100_000_000
            ),
            steps: pipeline_steps,
            ahead_by_k: req.ahead_by_k,
            max_voters: if req.ahead_by_k > 0 {
                req.ahead_by_k * 2 + 1
            } else {
                1
            },
            model: self.pipeline_model.clone(),
        };
        let result =
            pipeline::run_pipeline(&config, self.pipeline_provider.as_ref(), &self.workspace).await;
        // Format result for the agent (byte-identical to the legacy callback).
        let mut output = format!(
            "Pipeline '{}' completed: {}/{} steps\n",
            result.pipeline_id, result.steps_completed, result.steps_total
        );
        for sr in &result.results {
            let correct_str = match sr.correct {
                Some(true) => " ✓",
                Some(false) => " ✗",
                None => "",
            };
            let _ = writeln!(
                output,
                "  Step {}: {}{} ({}ms, {} voters)",
                sr.index,
                sr.answer.chars().take(200).collect::<String>(),
                correct_str,
                sr.duration_ms,
                sr.voters_used
            );
        }
        let _ = write!(output, "Total time: {}ms", result.total_duration_ms);
        Ok(PipelineReply { text: output })
    }
}

#[async_trait]
impl LoopHost for AgentHost {
    async fn run_loop(&self, req: LoopRequest) -> Result<LoopReply, ToolError> {
        let text = crate::agent::host_bridge::into_model_text(
            self.subagents
                .run_loop(
                    req.task,
                    req.max_rounds,
                    req.tools,
                    req.stop_condition,
                    req.model,
                    req.working_dir,
                )
                .await,
        )?;
        Ok(LoopReply { text })
    }
}

#[async_trait]
impl MessageHost for AgentHost {
    async fn send(&self, req: SendMessageRequest) -> Result<SendMessageReply, ToolError> {
        let msg = OutboundMessage::new(&req.channel, &req.chat_id, &req.content);
        self.outbound
            .send(msg)
            .map_err(|e| ToolError::Execution {
                // Byte-identical to the legacy send closure: it wrapped the
                // bus error in anyhow as "Failed to send outbound message: {e}"
                // and the old MessageTool prefixed "Error sending message: ".
                // The merged host must reproduce both prefixes so the model
                // sees the same string after the message swap.
                message: format!("Error sending message: Failed to send outbound message: {e}"),
            })?;
        Ok(SendMessageReply {
            text: format!("Message sent to {}:{}", req.channel, req.chat_id),
        })
    }
}

impl AgentLoopShared {
    /// Build a fresh [`ToolRegistry`] with context-sensitive tools (message,
    /// spawn, cron) pre-configured for a specific channel/chat_id.
    ///
    /// Takes a snapshot of `SwappableCore` so the registry is consistent for the
    /// entire message processing.
    ///
    /// Returns `(ToolRegistry, SharedEngine)` so the caller can wire the engine
    /// into [`TurnContext`] for agent-loop-level plan guidance and backtracking.
    pub(crate) async fn build_tools(
        &self,
        core: &SwappableCore,
        channel: &str,
        chat_id: &str,
    ) -> (
        ToolRegistry,
        crate::agent::tools::reasoning_tools::SharedEngine,
    ) {
        // Standard stateless tools via unified ToolConfig.
        let db_path = dirs::home_dir()
            .unwrap_or_default()
            .join(".nanobot")
            .join("sessions.db");
        let tool_config = ToolConfig {
            workspace: core.workspace.clone(),
            exec_timeout: core.exec_timeout,
            restrict_to_workspace: core.restrict_to_workspace,
            max_tool_result_chars: core.max_tool_result_chars,
            brave_api_key: core.brave_api_key.clone(),
            search_provider: core.search_provider.clone(),
            searxng_url: core.searxng_url.clone(),
            crw_url: core.crw_url.clone(),
            search_max_results: core.search_max_results,
            exec_working_dir: std::env::current_dir()
                .ok()
                .map(|p| p.to_string_lossy().to_string()),
            db_path: Some(db_path),
            health_registry: self.health_registry.clone(),
            code_execution: core.code_execution.clone(),
            #[cfg(feature = "python-kernel")]
            python_kernel: core.python_kernel.clone(),
            ..ToolConfig::new(&core.workspace)
        };
        let mut tools = ToolRegistry::with_standard_tools(&tool_config);

        // Typed host bridge (research §3.3 DIP): build the production host once
        // and inject it at the registry boundary. Each AgentHost method is the
        // byte-identical port of the legacy closure body that used to live
        // here (doc §3.7 Step 3); the closures are gone — the logic is a
        // named, testable type, and tools depend on the trait, never on this
        // struct or `SubagentManager`. The dispatcher is the single OCP choke
        // point, so production and a future cross-process transport share one
        // dispatch path.
        let agent_host = Arc::new(AgentHost {
            subagents: self.subagents.clone(),
            session_policies: self.session_policies.clone(),
            pipeline_provider: core
                .tool_runner_provider
                .clone()
                .unwrap_or_else(|| core.provider.clone()),
            pipeline_model: core
                .tool_runner_model
                .clone()
                .unwrap_or_else(|| core.model.clone()),
            workspace: core.workspace.clone(),
            outbound: self.bus_outbound_tx.clone(),
            channel: channel.to_string(),
            chat_id: chat_id.to_string(),
        });
        let host: Arc<dyn HostBridge> = Arc::new(HostDispatcher::new(
            agent_host.clone(),
            agent_host.clone(),
            agent_host.clone(),
            agent_host.clone(),
        ));

        // Direct UI channels already reply through `finalize_response`; exposing
        // `message` there gives models a second, failure-prone way to answer.
        if should_register_message_tool(channel) {
            // The send closure is gone (doc §3.7 Step 3): the message path is
            // the typed MessageHost trait on AgentHost, injected here.
            let message_tool = Arc::new(MessageTool::new(host.clone(), channel, chat_id));
            tools.register(Box::new(ArcToolProxy(message_tool)));
        }

        // Spawn tool - origin channel/chat are baked into the host (doc
        // §3.2/§3.3); the tool no longer carries context or callback slots.
        let spawn_tool = Arc::new(SpawnTool::new(host.clone()));
        // Local models get the lite schema (~200 tokens) instead of the full
        // schema (~1,100 tokens) which would consume 55% of a 4K context.
        // migrated from swappable().is_local — phase 09-03
        if core.mode().is_local() {
            tools.register(Box::new(SpawnToolLite(spawn_tool)));
        } else {
            tools.register(Box::new(ArcToolProxy(spawn_tool)));
        }

        tools = tools.with_host(Some(host));

        // Cron tool (optional) - context baked in.
        if let Some(ref svc) = self.cron_service {
            let ct = Arc::new(CronScheduleTool::new(svc.clone()));
            ct.set_context(channel, chat_id).await;
            tools.register(Box::new(ArcToolProxy(ct)));
        }

        // Todo scratchpad — workspace-scoped working memory.
        tools.register(Box::new(TodoTool::new(&core.workspace)));

        // Email tools (optional) - available when email is configured.
        if let Some(ref email_cfg) = self.email_config {
            tools.register(Box::new(CheckInboxTool::new(email_cfg.clone())));
            tools.register(Box::new(SendEmailTool::new(email_cfg.clone())));
        }

        // Reasoning tools — checkpoint, backtrack, plan (share the engine).
        // Only registered when reasoning is enabled in config.
        use crate::agent::reasoning::ReasoningEngine;
        use crate::agent::tools::reasoning_tools::{
            BacktrackTool, CheckpointTool, PlanTool, SharedEngine,
        };
        let reasoning_engine: SharedEngine =
            Arc::new(parking_lot::Mutex::new(ReasoningEngine::new()));
        {
            let mut eng = reasoning_engine.lock();
            eng.set_max_checkpoints(core.reasoning_config.max_checkpoints);
        }
        if core.reasoning_config.enabled {
            let checkpoint_tool = Arc::new(CheckpointTool::new(reasoning_engine.clone()));
            let backtrack_tool = Arc::new(BacktrackTool::new(reasoning_engine.clone()));
            let plan_tool = Arc::new(PlanTool::new(reasoning_engine.clone()));
            tools.register(Box::new(ArcToolProxy(checkpoint_tool)));
            tools.register(Box::new(ArcToolProxy(backtrack_tool)));
            tools.register(Box::new(ArcToolProxy(plan_tool)));
        }

        (tools, reasoning_engine)
    }
}

#[cfg(test)]
mod tests {
    use super::should_register_message_tool;

    #[test]
    fn direct_channels_do_not_register_message_tool() {
        for channel in ["cli", "voice", "tui", "repl"] {
            assert!(
                !should_register_message_tool(channel),
                "{channel} should use the final response path, not the message tool"
            );
        }
    }

    #[test]
    fn gateway_channels_register_message_tool() {
        for channel in ["telegram", "whatsapp", "email"] {
            assert!(
                should_register_message_tool(channel),
                "{channel} should allow explicit out-of-band sends"
            );
        }
    }
}

/// Unit tests for the production typed host. `AgentHost` is a plain struct,
/// so the capability methods are testable without the agent loop — exactly
/// the LSP/DIP payoff of porting the closure bodies into named methods.
#[cfg(test)]
mod agent_host_tests {
    use super::*;
    use crate::agent::host_bridge::{
        CancelRequest, CheckRequest, HostReply, HostRequest, ListSubagentsRequest,
        SendMessageRequest, SpawnRequest, WaitRequest,
    };
    use crate::bus::events::InboundMessage;
    use crate::providers::base::{LLMProvider, LLMResponse};

    /// Provider that never completes a call — enough for `SubagentManager` to
    /// construct and for `spawn` to start a task that fails harmlessly.
    struct NeverCompletesProvider;

    #[async_trait]
    impl LLMProvider for NeverCompletesProvider {
        fn get_default_model(&self) -> &str {
            "mock-model"
        }

        async fn chat(
            &self,
            _messages: &[serde_json::Value],
            _tools: Option<&[serde_json::Value]>,
            _model: Option<&str>,
            _max_tokens: u32,
            _temperature: f64,
            _thinking_budget: Option<u32>,
            _top_p: Option<f64>,
        ) -> anyhow::Result<LLMResponse> {
            anyhow::bail!("mock provider never completes calls");
        }
    }

    fn make_host(workspace: &std::path::Path) -> AgentHost {
        let (bus_tx, _bus_rx) = tokio::sync::mpsc::unbounded_channel::<InboundMessage>();
        let (outbound, _out_rx) = tokio::sync::mpsc::unbounded_channel::<OutboundMessage>();
        let subagents = Arc::new(SubagentManager::new(
            Arc::new(NeverCompletesProvider),
            workspace.to_path_buf(),
            bus_tx,
            "mock-model".to_string(),
            None,
            0,
            false,
            false,
            30_000,
        ));
        AgentHost {
            subagents,
            session_policies: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
            pipeline_provider: Arc::new(NeverCompletesProvider),
            pipeline_model: "mock-model".to_string(),
            workspace: workspace.to_path_buf(),
            outbound,
            channel: "telegram".to_string(),
            chat_id: "42".to_string(),
        }
    }

    #[tokio::test]
    async fn send_puts_message_on_the_bus_and_returns_confirmation() {
        let dir = tempfile::tempdir().unwrap();
        let host = make_host(dir.path());
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<OutboundMessage>();
        // Rebind outbound to a channel we can observe.
        let mut host = host;
        host.outbound = tx;

        let reply = host
            .send(SendMessageRequest {
                channel: "telegram".to_string(),
                chat_id: "42".to_string(),
                content: "hello".to_string(),
            })
            .await
            .unwrap();
        assert_eq!(reply.text, "Message sent to telegram:42");
        let msg = rx.try_recv().unwrap();
        assert_eq!(msg.channel, "telegram");
        assert_eq!(msg.chat_id, "42");
        assert_eq!(msg.content, "hello");
    }

    #[tokio::test]
    async fn cancel_unknown_task_reports_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let host = make_host(dir.path());
        let reply = host
            .cancel(CancelRequest {
                task_id: "nope".to_string(),
            })
            .await
            .unwrap();
        assert_eq!(reply.text, "No running subagent found matching 'nope'.");
    }

    #[tokio::test]
    async fn list_empty_workspace_reports_no_running_subagents() {
        let dir = tempfile::tempdir().unwrap();
        let host = make_host(dir.path());
        let reply = host.list_subagents().await.unwrap();
        assert!(reply.text.contains("No subagents currently running."));
    }

    #[tokio::test]
    async fn check_unknown_task_reports_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let host = make_host(dir.path());
        let reply = host
            .check(CheckRequest {
                task_id: "nope".to_string(),
            })
            .await
            .unwrap();
        assert!(
            reply
                .text
                .contains("No running or completed result found for task_id 'nope'.")
        );
    }

    #[tokio::test]
    async fn wait_unknown_task_reports_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let host = make_host(dir.path());
        let reply = host
            .wait(WaitRequest {
                task_id: "nope".to_string(),
                timeout_secs: 1,
            })
            .await
            .unwrap();
        assert!(reply.text.contains("No running subagent found matching 'nope'."));
    }

    #[tokio::test]
    async fn spawn_returns_byte_stable_confirmation_with_task_id() {
        let dir = tempfile::tempdir().unwrap();
        let host = make_host(dir.path());
        let reply = host
            .spawn(SpawnRequest {
                task: "do a thing".to_string(),
                label: Some("explore".to_string()),
                profile: None,
                model: None,
                working_dir: None,
                channel: "telegram".to_string(),
                chat_id: "42".to_string(),
            })
            .await
            .unwrap();
        // The exact wire format the legacy closure produced.
        assert!(reply.text.starts_with("Subagent 'explore' spawned (id: "));
        assert!(reply.text.contains(", model: mock-model). It will announce results when done."));
        assert_eq!(reply.task_id.len(), 8);
        assert!(reply.text.contains(&format!("(id: {}", reply.task_id)));
    }

    /// End-to-end dispatcher over AgentHost — the production composition.
    #[tokio::test]
    async fn dispatcher_over_agent_host_routes_requests() {
        let dir = tempfile::tempdir().unwrap();
        let agent = Arc::new(make_host(dir.path()));
        let dispatcher =
            HostDispatcher::new(agent.clone(), agent.clone(), agent.clone(), agent.clone());

        let reply = dispatcher
            .dispatch(HostRequest::CancelSubagent(CancelRequest {
                task_id: "nope".to_string(),
            }))
            .await;
        let HostReply::Ok { data } = reply else {
            panic!("expected ok");
        };
        assert_eq!(data["text"], "No running subagent found matching 'nope'.");

        let reply = dispatcher
            .dispatch(HostRequest::ListSubagents(ListSubagentsRequest {}))
            .await;
        let HostReply::Ok { data } = reply else {
            panic!("expected ok");
        };
        assert!(data["text"].as_str().unwrap().contains("No subagents currently running."));
    }
}
