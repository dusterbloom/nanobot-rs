//! Tool registry construction and wiring.
//!
//! Extracted from `agent_loop.rs` to isolate the callback-heavy tool setup.

use std::collections::HashMap;
use std::sync::Arc;

use serde_json::Value;

use crate::agent::agent_core::SwappableCore;
use crate::agent::agent_loop::AgentLoopShared;
use crate::agent::pipeline;
use crate::agent::policy;
use crate::agent::subagent::SubagentManager;
use crate::agent::tools::registry::{ToolConfig, ToolRegistry};
use crate::agent::tools::{
    CancelCallback, CheckCallback, CheckInboxTool, CronScheduleTool, ListCallback, LoopCallback,
    MessageTool, PipelineCallback, SendCallback, SendEmailTool, SpawnCallback, SpawnTool,
    SpawnToolLite, WaitCallback,
};
use crate::bus::events::OutboundMessage;

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
    async fn execute(&self, params: HashMap<String, Value>) -> String {
        self.0.execute(params).await
    }
}

impl AgentLoopShared {
    /// Build a fresh [`ToolRegistry`] with context-sensitive tools (message,
    /// spawn, cron) pre-configured for a specific channel/chat_id.
    ///
    /// Takes a snapshot of `SwappableCore` so the registry is consistent for the
    /// entire message processing.
    pub(crate) async fn build_tools(
        &self,
        core: &SwappableCore,
        channel: &str,
        chat_id: &str,
    ) -> ToolRegistry {
        // Standard stateless tools via unified ToolConfig.
        let tool_config = ToolConfig {
            workspace: core.workspace.clone(),
            exec_timeout: core.exec_timeout,
            restrict_to_workspace: core.restrict_to_workspace,
            max_tool_result_chars: core.max_tool_result_chars,
            brave_api_key: core.brave_api_key.clone(),
            ..ToolConfig::new(&core.workspace)
        };
        let mut tools = ToolRegistry::with_standard_tools(&tool_config);

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
        tools.register(Box::new(ArcToolProxy(message_tool)));

        // Spawn tool - context baked in.
        let subagents_ref = self.subagents.clone();
        let session_policies_ref = self.session_policies.clone();
        let spawn_cb: SpawnCallback =
            Arc::new(move |task, label, agent, model, ch, cid, working_dir| {
                let mgr = subagents_ref.clone();
                let policies = session_policies_ref.clone();
                Box::pin(async move {
                    let key = format!("{}:{}", ch, cid);
                    let policy = {
                        let map = policies.lock().await;
                        map.get(&key).cloned().unwrap_or_default()
                    };
                    let effective_model = policy::enforce_subagent_model(&policy, model);
                    mgr.spawn(task, label, agent, effective_model, ch, cid, working_dir)
                        .await
                })
            });
        let subagents_ref2 = self.subagents.clone();
        let list_workspace = core.workspace.clone();
        let list_cb: ListCallback = Arc::new(move || {
            let mgr = subagents_ref2.clone();
            let ws = list_workspace.clone();
            Box::pin(async move {
                let running = mgr.list_running().await;
                let mut out = String::new();

                // Running subagents
                if running.is_empty() {
                    out.push_str("No subagents currently running.\n");
                } else {
                    out.push_str(&format!("{} subagent(s) running:\n", running.len()));
                    for info in &running {
                        let elapsed = info.started_at.elapsed().as_secs();
                        out.push_str(&format!(
                            "  • {} (id: {}) — running for {}s\n",
                            info.label, info.task_id, elapsed
                        ));
                    }
                }

                // Recently completed (from events.jsonl)
                let recent = SubagentManager::read_recent_completed(&ws, 10);
                if !recent.is_empty() {
                    out.push_str(&format!("\nRecently completed ({}):\n", recent.len()));
                    for entry in &recent {
                        out.push_str(entry);
                        out.push('\n');
                    }
                }

                out
            })
        });
        let subagents_ref3 = self.subagents.clone();
        let cancel_cb: CancelCallback = Arc::new(move |task_id: String| {
            let mgr = subagents_ref3.clone();
            Box::pin(async move {
                if mgr.cancel(&task_id).await {
                    format!("Subagent '{}' cancelled.", task_id)
                } else {
                    format!("No running subagent found matching '{}'.", task_id)
                }
            })
        });
        let subagents_ref4 = self.subagents.clone();
        let wait_cb: WaitCallback = Arc::new(move |task_id: String, timeout_secs: u64| {
            let mgr = subagents_ref4.clone();
            Box::pin(async move {
                let timeout = std::time::Duration::from_secs(timeout_secs);
                mgr.wait_for(&task_id, timeout).await
            })
        });
        let check_workspace = core.workspace.clone();
        let check_cb: CheckCallback = Arc::new(move |task_id: String| {
            let ws = check_workspace.clone();
            Box::pin(async move {
                match SubagentManager::read_event_result(&ws, &task_id) {
                    Some(result) => result,
                    None => format!("No completed result found for task_id '{}'.", task_id),
                }
            })
        });
        // Pipeline callback: parse steps JSON, build PipelineConfig, run pipeline.
        // Uses the delegation provider/model (cheap) for pipeline LLM calls.
        let pipeline_provider = core
            .tool_runner_provider
            .clone()
            .unwrap_or_else(|| core.provider.clone());
        let pipeline_model = core
            .tool_runner_model
            .clone()
            .unwrap_or_else(|| core.model.clone());
        let pipeline_workspace = core.workspace.clone();
        let pipeline_cb: PipelineCallback =
            Arc::new(move |steps_json: String, ahead_by_k: usize| {
                let provider = pipeline_provider.clone();
                let model = pipeline_model.clone();
                let workspace = pipeline_workspace.clone();
                Box::pin(async move {
                    // Parse steps from JSON.
                    let steps: Vec<serde_json::Value> = match serde_json::from_str(&steps_json) {
                        Ok(s) => s,
                        Err(e) => return format!("Error parsing pipeline steps: {}", e),
                    };
                    let pipeline_steps: Vec<pipeline::PipelineStep> = steps
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
                        return "Error: no valid pipeline steps provided.".to_string();
                    }
                    let config = pipeline::PipelineConfig {
                        pipeline_id: format!(
                            "pipe-{}",
                            chrono::Utc::now().timestamp_millis() % 100_000_000
                        ),
                        steps: pipeline_steps,
                        ahead_by_k,
                        max_voters: if ahead_by_k > 0 {
                            ahead_by_k * 2 + 1
                        } else {
                            1
                        },
                        model: model.clone(),
                    };
                    let result =
                        pipeline::run_pipeline(&config, provider.as_ref(), &workspace).await;
                    // Format result for the agent.
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
                        output.push_str(&format!(
                            "  Step {}: {}{} ({}ms, {} voters)\n",
                            sr.index,
                            sr.answer.chars().take(200).collect::<String>(),
                            correct_str,
                            sr.duration_ms,
                            sr.voters_used
                        ));
                    }
                    output.push_str(&format!("Total time: {}ms", result.total_duration_ms));
                    output
                })
            });

        // Loop callback: run an autonomous refinement loop via SubagentManager.
        let subagents_ref5 = self.subagents.clone();
        let loop_cb: LoopCallback = Arc::new(
            move |task: String,
                  max_rounds: u32,
                  tools_filter: Option<Vec<String>>,
                  stop_condition: Option<String>,
                  model: Option<String>,
                  working_dir: Option<String>| {
                let mgr = subagents_ref5.clone();
                Box::pin(async move {
                    mgr.run_loop(
                        task,
                        max_rounds,
                        tools_filter,
                        stop_condition,
                        model,
                        working_dir,
                    )
                    .await
                })
            },
        );

        let spawn_tool = Arc::new(SpawnTool::new());
        // Set callbacks and context before registering so they're ready for use.
        spawn_tool.set_callback(spawn_cb).await;
        spawn_tool.set_list_callback(list_cb).await;
        spawn_tool.set_cancel_callback(cancel_cb).await;
        spawn_tool.set_wait_callback(wait_cb).await;
        spawn_tool.set_check_callback(check_cb).await;
        spawn_tool.set_pipeline_callback(pipeline_cb).await;
        spawn_tool.set_loop_callback(loop_cb).await;
        spawn_tool.set_context(channel, chat_id).await;
        // Local models get the lite schema (~200 tokens) instead of the full
        // schema (~1,100 tokens) which would consume 55% of a 4K context.
        if core.is_local {
            tools.register(Box::new(SpawnToolLite(spawn_tool)));
        } else {
            tools.register(Box::new(ArcToolProxy(spawn_tool)));
        }

        // Cron tool (optional) - context baked in.
        if let Some(ref svc) = self.cron_service {
            let ct = Arc::new(CronScheduleTool::new(svc.clone()));
            ct.set_context(channel, chat_id).await;
            tools.register(Box::new(ArcToolProxy(ct)));
        }

        // Email tools (optional) - available when email is configured.
        if let Some(ref email_cfg) = self.email_config {
            tools.register(Box::new(CheckInboxTool::new(email_cfg.clone())));
            tools.register(Box::new(SendEmailTool::new(email_cfg.clone())));
        }

        tools
    }
}
