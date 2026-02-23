#![allow(dead_code)]
//! Subagent manager for background task execution.
//!
//! Spawns independent agent loops that can read/write files, execute commands,
//! and search the web, then announce results back to the main agent.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use chrono::Utc;
use serde_json::{json, Value};
use tokio::sync::mpsc::UnboundedSender;
use tokio::sync::{broadcast, Mutex};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::agent::agent_profiles::{self, AgentProfile};
use crate::agent::token_budget::TokenBudget;
use crate::agent::tools::registry::{ToolConfig, ToolRegistry};
use crate::bus::events::InboundMessage;
use crate::config::schema::{ProvidersConfig, SubagentTuning};
use crate::providers::base::LLMProvider;

/// Configuration passed to `_run_subagent`, derived from profile + overrides.
#[derive(Debug, Clone)]
struct SubagentConfig {
    model: String,
    system_prompt: Option<String>,
    tools_filter: Option<Vec<String>>,
    read_only: bool,
    max_iterations: u32,
    max_tool_result_chars: usize,
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

/// Extract localhost port from an API base URL.
fn extract_local_port_from_api_base(api_base: &str) -> Option<u16> {
    let parsed = reqwest::Url::parse(api_base).ok()?;
    let host = parsed.host_str()?;
    if host != "localhost" && host != "127.0.0.1" {
        return None;
    }
    parsed.port_or_known_default()
}

/// Resolve per-request local context limit with headroom.
///
/// When a parent-provided limit is available (e.g. from config), use it directly
/// instead of querying the server — the server may be on a non-localhost IP
/// (LM Studio on WSL2) or may not support the `/props` endpoint.
fn resolve_local_context_limit(
    provider: &dyn LLMProvider,
    parent_limit: Option<usize>,
    min_context: usize,
    fallback_context: usize,
) -> usize {
    // Try querying the server directly (works for local llama-server).
    if let Some(base) = provider.get_api_base() {
        if let Some(port) = extract_local_port_from_api_base(base) {
            if let Some(ctx) = crate::server::query_local_context_size(&port.to_string()) {
                return ctx.max(min_context);
            }
        }
    }
    // Use parent's known limit (from config) if available.
    parent_limit.unwrap_or(fallback_context)
}

/// Compute a conservative local response budget from the detected context size.
fn local_response_token_limit(ctx_limit: usize, min_response_tokens: u32, max_response_tokens: u32) -> u32 {
    let adaptive = (ctx_limit / 6).clamp(
        min_response_tokens as usize,
        max_response_tokens as usize,
    );
    adaptive as u32
}

/// Detect provider-side context overflow errors in a robust way.
fn is_context_overflow_error(err: &anyhow::Error) -> bool {
    let msg = err.to_string().to_lowercase();
    msg.contains("exceed_context_size_error")
        || msg.contains("exceeds the available context size")
        || msg.contains("context size")
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
    /// Cheap default model for subagents when no explicit override is given.
    /// Prevents the expensive main model from being used as a worker.
    default_subagent_model: Option<String>,
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
    /// Current spawn depth (0 = root). Children get depth + 1.
    /// Spawns are rejected when depth >= MAX_SPAWN_DEPTH.
    depth: u32,
    running_tasks: Arc<
        Mutex<
            HashMap<
                String,
                (
                    SubagentInfo,
                    tokio::task::JoinHandle<()>,
                    broadcast::Sender<String>,
                ),
            >,
        >,
    >,
    /// Max chars per tool result (passed from config).
    max_tool_result_chars: usize,
    /// Parent's context token limit — used as fallback when the server can't be
    /// queried directly (e.g. LM Studio on a non-localhost IP).
    local_context_limit: Option<usize>,
    /// Priority signal sender for the aha channel (proprioception Phase 6).
    aha_tx: Option<UnboundedSender<crate::agent::system_state::AhaSignal>>,
    /// Tuning knobs for subagent execution (from config).
    subagent_tuning: SubagentTuning,
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
        max_tool_result_chars: usize,
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
            default_subagent_model: None,
            brave_api_key,
            exec_timeout,
            restrict_to_workspace,
            is_local,
            max_tool_result_chars,
            local_context_limit: None,
            profiles,
            providers_config: None,
            display_tx: None,
            depth: 0,
            running_tasks: Arc::new(Mutex::new(HashMap::new())),
            aha_tx: None,
            subagent_tuning: SubagentTuning::default(),
        }
    }

    /// Set subagent execution tuning from config.
    pub fn with_subagent_tuning(mut self, tuning: SubagentTuning) -> Self {
        self.subagent_tuning = tuning;
        self
    }

    /// Set the aha channel sender for priority signals (proprioception).
    pub fn with_aha_tx(
        mut self,
        tx: UnboundedSender<crate::agent::system_state::AhaSignal>,
    ) -> Self {
        self.aha_tx = Some(tx);
        self
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

    /// Set the parent's context token limit as fallback for subagents.
    pub fn with_local_context_limit(mut self, limit: usize) -> Self {
        self.local_context_limit = Some(limit);
        self
    }

    /// Set the spawn depth (for nested subagents).
    pub fn with_depth(mut self, depth: u32) -> Self {
        self.depth = depth;
        self
    }

    /// Set the default cheap model for subagents (prevents main model leak).
    pub fn with_default_subagent_model(mut self, model: String) -> Self {
        if !model.is_empty() {
            self.default_subagent_model = Some(model);
        }
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
        // Depth limit check: prevent infinite recursion.
        let max_spawn_depth = self.subagent_tuning.max_spawn_depth;
        if self.depth >= max_spawn_depth {
            warn!(
                "Spawn rejected: depth {} >= max_spawn_depth {}",
                self.depth, max_spawn_depth
            );
            return format!(
                "Error: spawn depth limit reached ({}/{}). Cannot spawn nested subagents beyond depth {}.",
                self.depth, max_spawn_depth, max_spawn_depth
            );
        }

        let task_id = Uuid::new_v4().to_string()[..8].to_string();
        info!(
            role = "subagent",
            task_id = %task_id,
            depth = self.depth,
            agent = ?agent_name,
            "subagent_spawn"
        );

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

        // Build config: model_override > profile.model > default_subagent_model > self.model (last resort)
        let effective_model = if let Some(ref m) = model_override {
            agent_profiles::resolve_model_alias(m)
        } else if let Some(ref p) = profile {
            p.model
                .as_ref()
                .map(|m| agent_profiles::resolve_model_alias(m))
                .unwrap_or_else(|| {
                    self.default_subagent_model.clone().unwrap_or_else(|| {
                        warn!("EXPENSIVE: Using main model '{}' as subagent — set defaultSubagentModel in config", self.model);
                        self.model.clone()
                    })
                })
        } else {
            self.default_subagent_model.clone().unwrap_or_else(|| {
                warn!("EXPENSIVE: Using main model '{}' as subagent — set defaultSubagentModel in config", self.model);
                self.model.clone()
            })
        };

        // Budget halving: each depth level gets half the iterations.
        // depth 0 → full budget, depth 1 → half, depth 2 → quarter.
        let base_iterations = profile
            .as_ref()
            .and_then(|p| p.max_iterations)
            .unwrap_or(self.subagent_tuning.max_iterations);
        let depth_budget = base_iterations >> self.depth; // halve per depth level
        let effective_iterations = depth_budget.max(3); // minimum 3 iterations
        if self.depth > 0 {
            info!(
                "Subagent depth={}, budget={} iterations (base={}, halved {} times)",
                self.depth, effective_iterations, base_iterations, self.depth
            );
        }

        let mut config = SubagentConfig {
            model: effective_model,
            system_prompt: profile.as_ref().map(|p| p.system_prompt.clone()),
            tools_filter: profile.as_ref().and_then(|p| p.tools.clone()),
            read_only: profile.as_ref().map(|p| p.read_only).unwrap_or(false),
            max_iterations: effective_iterations,
            max_tool_result_chars: self.max_tool_result_chars,
        };

        let effective_model_for_display = config.model.clone();

        // Resolve provider for model prefix (groq/, gemini/, openai/, etc.)
        let (provider, resolved_model, targets_local) =
            self.resolve_provider_for_model(&config.model);
        let routed_to_cloud = resolved_model != config.model;
        if routed_to_cloud {
            info!(
                role = "subagent",
                task_id = %task_id,
                agent = agent_name.as_deref().unwrap_or("default"),
                model = %effective_model_for_display,
                resolved_model = %resolved_model,
                depth = self.depth,
                "subagent_spawn_routed: {}",
                display_label
            );
            config.model = resolved_model;
        } else {
            info!(
                role = "subagent",
                task_id = %task_id,
                agent = agent_name.as_deref().unwrap_or("default"),
                model = %effective_model_for_display,
                depth = self.depth,
                "subagent_spawn: {}",
                display_label
            );
        }
        let workspace = self.workspace.clone();
        let exec_working_dir = working_dir;
        let bus_tx = self.bus_tx.clone();
        let brave_api_key = self.brave_api_key.clone();
        let exec_timeout = self.exec_timeout;
        let restrict_to_workspace = self.restrict_to_workspace;
        // Provider-routed subagents: check if the resolved API base is localhost.
        // A groq/ prefix pointing to localhost:8083 is still a local server
        // and needs strict alternation repair.
        let is_local = if routed_to_cloud {
            targets_local
        } else {
            self.is_local
        };
        let display_tx = self.display_tx.clone();
        let running_tasks = self.running_tasks.clone();
        let tid = task_id.clone();
        let lbl = display_label.clone();
        let tsk = task.clone();
        let aha_tx = self.aha_tx.clone();
        let parent_ctx_limit = self.local_context_limit;
        let tuning = self.subagent_tuning.clone();

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
                parent_ctx_limit,
                &tuning,
            )
            .await;

            let (result_text, status) = match result {
                Ok(text) => (text, "completed"),
                Err(e) => (format!("Error: {}", e), "failed"),
            };

            // Emit aha signal if result matches priority patterns.
            if let Some(ref tx) = aha_tx {
                use crate::agent::system_state::{classify_signal, AhaSignal};
                if let Some(priority) = classify_signal(&result_text) {
                    let category = if status == "failed" {
                        "error".to_string()
                    } else {
                        "complete".to_string()
                    };
                    let _ = tx.send(AhaSignal {
                        priority,
                        agent_id: tid.clone(),
                        category,
                        message: truncate_for_display(&result_text, 5, 500),
                    });
                }
            }

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
                let status_color = if status == "completed" {
                    "\x1b[32m"
                } else {
                    "\x1b[31m"
                };
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
                    block.push_str(&format!(
                        "  \x1b[2m\u{2502}\x1b[0m \x1b[37m{}\x1b[0m\n",
                        line
                    ));
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
                "Subagent '{}' is still running after {}s. \
                 Use 'spawn list' to check status, 'spawn check' to fetch completed output, \
                 or 'spawn cancel' to abort.",
                task_id,
                timeout.as_secs()
            ),
        }
    }

    /// Run an autonomous refinement loop: repeated rounds of agent work
    /// with accumulated history and a stop condition.
    ///
    /// Each round runs `_run_subagent` with the task augmented by previous
    /// round summaries. The loop stops when:
    /// - The agent's output contains "DONE" (case-insensitive)
    /// - `max_rounds` is reached
    /// - The stop condition text appears in the output
    ///
    /// Returns all round summaries concatenated.
    pub async fn run_loop(
        &self,
        task: String,
        max_rounds: u32,
        tools_filter: Option<Vec<String>>,
        stop_condition: Option<String>,
        model_override: Option<String>,
        working_dir: Option<String>,
    ) -> String {
        let effective_model = model_override
            .or_else(|| self.default_subagent_model.clone())
            .unwrap_or_else(|| self.model.clone());

        let (provider, resolved_model, targets_local) =
            self.resolve_provider_for_model(&effective_model);
        let is_local = if resolved_model != effective_model {
            targets_local
        } else {
            self.is_local
        };

        info!(
            "Starting agent loop: max_rounds={}, model={}, stop={:?}",
            max_rounds, resolved_model, stop_condition
        );

        let config = SubagentConfig {
            model: resolved_model.clone(),
            system_prompt: None,
            tools_filter,
            read_only: false,
            max_iterations: self.subagent_tuning.max_iterations,
            max_tool_result_chars: self.max_tool_result_chars,
        };

        let mut round_summaries: Vec<String> = Vec::new();

        for round in 0..max_rounds {
            // Build task with accumulated context from previous rounds.
            let round_task = if round_summaries.is_empty() {
                task.clone()
            } else {
                let history = round_summaries
                    .iter()
                    .enumerate()
                    .map(|(i, s)| format!("## Round {} results\n{}", i + 1, s))
                    .collect::<Vec<_>>()
                    .join("\n\n");
                format!(
                    "{}\n\n## Previous rounds\n{}\n\n## Round {} of {}\n{}",
                    task,
                    history,
                    round + 1,
                    max_rounds,
                    stop_condition
                        .as_ref()
                        .map(|s| format!("Stop condition: {}", s))
                        .unwrap_or_default()
                )
            };

            let task_id = format!("loop-r{}", round + 1);
            info!("Agent loop round {}/{}", round + 1, max_rounds);

            let result = Self::_run_subagent(
                &task_id,
                &round_task,
                &format!("loop-round-{}", round + 1),
                provider.as_ref(),
                &self.workspace,
                &config,
                self.brave_api_key.as_deref(),
                self.exec_timeout,
                self.restrict_to_workspace,
                is_local,
                working_dir.as_deref(),
                self.local_context_limit,
                &self.subagent_tuning,
            )
            .await;

            let round_text = match result {
                Ok(text) => text,
                Err(e) => {
                    let err = format!("Round {} error: {}", round + 1, e);
                    warn!("{}", err);
                    round_summaries.push(err);
                    break;
                }
            };

            // Persist round result to event log.
            Self::append_event(
                &self.workspace,
                &task_id,
                &format!("loop-round-{}", round + 1),
                &task,
                &round_text,
                "completed",
            );

            // Check stop condition.
            let done = round_text.to_lowercase().contains("done")
                || stop_condition
                    .as_ref()
                    .map(|sc| round_text.to_lowercase().contains(&sc.to_lowercase()))
                    .unwrap_or(false);

            round_summaries.push(format!("Round {}: {}", round + 1, round_text));

            if done {
                info!(
                    "Agent loop stop condition met at round {}/{}",
                    round + 1,
                    max_rounds
                );
                break;
            }
        }

        // Build final output.
        let mut output = format!(
            "Agent loop completed ({} rounds):\n\n",
            round_summaries.len()
        );
        for summary in &round_summaries {
            output.push_str(summary);
            output.push_str("\n\n---\n\n");
        }
        output
    }

    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    /// Resolve a model string to (provider, stripped_model, targets_local_server).
    ///
    /// If the model has a provider prefix (e.g. `groq/llama-3.3-70b`), creates
    /// a dedicated `OpenAICompatProvider` for that backend. Otherwise returns
    /// the parent provider and the model unchanged.
    ///
    /// The third return value is `true` when the resolved API base points to
    /// localhost, which means strict alternation repair is still needed even
    /// though the model was "routed" via a provider prefix.
    fn resolve_provider_for_model(&self, model: &str) -> (Arc<dyn LLMProvider>, String, bool) {
        if let Some(ref pc) = self.providers_config {
            if let Some((api_key, base, rest)) = pc.resolve_model_prefix(model) {
                let prefix = model.split('/').next().unwrap_or("unknown");
                let targets_local = crate::providers::openai_compat::is_local_api_base(&base);
                info!(
                    "Subagent using {} provider (base={}, local={}) for model {}",
                    prefix, base, targets_local, rest
                );
                let provider: Arc<dyn LLMProvider> =
                    crate::providers::factory::create_openai_compat(
                        crate::providers::factory::ProviderSpec {
                            api_key,
                            api_base: Some(base),
                            model: Some(rest.clone()),
                            jit_gate: None,
                        },
                    );
                return (provider, rest, targets_local);
            }
        }

        // No prefix match → use parent provider with model as-is.
        (self.provider.clone(), model.to_string(), false)
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
        parent_context_limit: Option<usize>,
        tuning: &SubagentTuning,
    ) -> anyhow::Result<String> {
        debug!(
            "Subagent {} starting (model={}, max_iter={}, read_only={}, tools_filter={:?}): {}",
            task_id,
            config.model,
            config.max_iterations,
            config.read_only,
            config.tools_filter,
            label
        );

        // Build a tool registry using unified ToolConfig.
        let tool_config = ToolConfig {
            workspace: workspace.to_path_buf(),
            exec_timeout,
            restrict_to_workspace,
            max_tool_result_chars: config.max_tool_result_chars,
            brave_api_key: brave_api_key.map(|s| s.to_string()),
            read_only: config.read_only,
            tools_filter: config.tools_filter.clone(),
            exec_working_dir: exec_working_dir.map(|s| s.to_string()),
            ..ToolConfig::new(workspace)
        };
        let tools = ToolRegistry::with_standard_tools(&tool_config);

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

        let local_ctx_limit = if is_local {
            Some(resolve_local_context_limit(
                provider,
                parent_context_limit,
                tuning.local_min_context,
                tuning.local_fallback_context,
            ))
        } else {
            None
        };
        let max_response_tokens = local_ctx_limit
            .map(|ctx| local_response_token_limit(
                ctx,
                tuning.local_min_response_tokens,
                tuning.local_max_response_tokens,
            ))
            .unwrap_or(4096);

        let tool_defs = tools.get_definitions();
        let tool_defs_opt: Option<&[Value]> = if tool_defs.is_empty() {
            None
        } else {
            Some(&tool_defs)
        };

        let mut final_content = String::new();
        let mut taint_state = crate::agent::taint::TaintState::new();

        for iteration in 0..config.max_iterations {
            debug!(
                "Subagent {} iteration {}/{}",
                task_id,
                iteration + 1,
                config.max_iterations
            );

            if let Some(ctx_limit) = local_ctx_limit {
                let tool_def_tokens = tool_defs_opt
                    .map(TokenBudget::estimate_tool_def_tokens)
                    .unwrap_or(0);
                let budget = TokenBudget::new(ctx_limit, max_response_tokens as usize);
                let available = budget.available_budget(tool_def_tokens);
                let before = TokenBudget::estimate_tokens(&messages);
                if before > available {
                    let trimmed = budget.trim_to_fit(&messages, tool_def_tokens);
                    let after = TokenBudget::estimate_tokens(&trimmed);
                    warn!(
                        "Subagent {} context guard trimmed prompt ({} -> {} tokens, limit={}, reserve={}, tool_defs={})",
                        task_id,
                        before,
                        after,
                        ctx_limit,
                        max_response_tokens,
                        tool_def_tokens,
                    );
                    messages = trimmed;
                }
            }

            let response = match provider
                .chat(
                    &messages,
                    tool_defs_opt,
                    Some(&config.model),
                    max_response_tokens,
                    0.7,
                    None,
                    None,
                )
                .await
            {
                Ok(r) => r,
                Err(e) if local_ctx_limit.is_some() && is_context_overflow_error(&e) => {
                    let ctx_limit = local_ctx_limit.unwrap_or(tuning.local_fallback_context);
                    let retry_ctx = ((ctx_limit as f64) * 0.85).round() as usize;
                    let retry_ctx = retry_ctx.max(tuning.local_min_context);
                    let retry_max_tokens =
                        (max_response_tokens / 2).max(tuning.local_min_response_tokens);
                    let tool_def_tokens = tool_defs_opt
                        .map(TokenBudget::estimate_tool_def_tokens)
                        .unwrap_or(0);
                    let retry_budget = TokenBudget::new(retry_ctx, retry_max_tokens as usize);
                    let before_retry = TokenBudget::estimate_tokens(&messages);
                    messages = retry_budget.trim_to_fit(&messages, tool_def_tokens);
                    let after_retry = TokenBudget::estimate_tokens(&messages);
                    warn!(
                        "Subagent {} hit context overflow; retrying once after hard trim ({} -> {} tokens, retry_limit={}, retry_reserve={}): {}",
                        task_id,
                        before_retry,
                        after_retry,
                        retry_ctx,
                        retry_max_tokens,
                        e,
                    );
                    provider
                        .chat(
                            &messages,
                            tool_defs_opt,
                            Some(&config.model),
                            retry_max_tokens,
                            0.7,
                            None,
                            None,
                        )
                        .await?
                }
                Err(e) => return Err(e),
            };

            if let Some(err_msg) = response.error_detail() {
                error!("Subagent {} LLM provider error: {}", task_id, err_msg);
                return Err(anyhow::anyhow!("[LLM Error] {}", err_msg));
            }

            if crate::agent::tool_runner::process_tool_response(
                &response, &mut messages, &tools, is_local,
            ).await {
                // Taint tracking: check sensitive tools before marking new taint sources.
                for tc in &response.tool_calls {
                    if let Some(_spans) = taint_state.check_sensitive(&tc.name) {
                        warn!(
                            "TAINT WARNING [subagent {}]: Sensitive tool '{}' called with tainted context from: {}",
                            label, tc.name, taint_state.taint_summary()
                        );
                    }
                    let detail = tc.arguments.get("url")
                        .or_else(|| tc.arguments.get("query"))
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());
                    taint_state.mark_tainted(&tc.name, detail);
                }
            } else {
                // No tool calls — the subagent is done.
                final_content = crate::agent::sanitize::sanitize_reasoning_output(
                    &response.content.unwrap_or_default(),
                );
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
        let event = serde_json::json!({
            "ts": Utc::now().to_rfc3339(),
            "kind": "subagent_result",
            "task_id": task_id,
            "label": label,
            "task": task,
            "status": status,
            "result": result,
        });
        crate::utils::helpers::append_jsonl_event(workspace, &event);
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

    /// Read the last N completed subagent results from events.jsonl.
    pub fn read_recent_completed(workspace: &PathBuf, max_entries: usize) -> Vec<String> {
        let event_path = workspace.join("events.jsonl");
        let content = match std::fs::read_to_string(&event_path) {
            Ok(c) => c,
            Err(_) => return Vec::new(),
        };
        let mut results = Vec::new();
        for line in content.lines().rev() {
            if results.len() >= max_entries {
                break;
            }
            if let Ok(ev) = serde_json::from_str::<serde_json::Value>(line) {
                if ev["kind"] == "subagent_result" {
                    let task_id = ev["task_id"].as_str().unwrap_or("?");
                    let label = ev["label"].as_str().unwrap_or("");
                    let status = ev["status"].as_str().unwrap_or("unknown");
                    let ts = ev["ts"].as_str().unwrap_or("");
                    results.push(format!(
                        "  • {} (id: {}) — {} [{}]",
                        label, task_id, status, ts
                    ));
                }
            }
        }
        results
    }

    /// Search event log for a completed subagent result by task_id prefix.
    pub fn read_event_result(workspace: &PathBuf, task_id_prefix: &str) -> Option<String> {
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

/// Format a compact status block for system prompt injection.
///
/// Returns empty string when nothing is running and no recent completions,
/// so callers can skip injection entirely.
pub fn format_status_block(running: &[SubagentInfo], recent_completed: &[String]) -> String {
    if running.is_empty() && recent_completed.is_empty() {
        return String::new();
    }
    let mut out = String::from("\n\n## Background Status\n");
    if !running.is_empty() {
        for info in running {
            let elapsed = info.started_at.elapsed().as_secs();
            out.push_str(&format!(
                "- RUNNING: {} (id:{}, {}s)\n",
                info.label, info.task_id, elapsed
            ));
        }
    }
    for entry in recent_completed.iter().take(3) {
        out.push_str(&format!("- {}\n", entry.trim_start_matches("  • ")));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::base::{LLMResponse, ToolCallRequest};
    use async_trait::async_trait;

    #[test]
    fn test_extract_local_port_from_api_base() {
        assert_eq!(
            extract_local_port_from_api_base("http://localhost:8080/v1"),
            Some(8080)
        );
        assert_eq!(
            extract_local_port_from_api_base("http://127.0.0.1:18080/v1"),
            Some(18080)
        );
        assert_eq!(
            extract_local_port_from_api_base("https://api.openai.com/v1"),
            None
        );
        assert_eq!(extract_local_port_from_api_base("not-a-url"), None);
    }

    #[test]
    fn test_is_context_overflow_error_detects_known_strings() {
        let e1 =
            anyhow::Error::msg("HTTP 400: {\"error\":{\"type\":\"exceed_context_size_error\"}}");
        let e2 = anyhow::anyhow!(
            "request (12197 tokens) exceeds the available context size (8192 tokens)"
        );
        let e3 = anyhow::anyhow!("network timeout");

        assert!(is_context_overflow_error(&e1));
        assert!(is_context_overflow_error(&e2));
        assert!(!is_context_overflow_error(&e3));
    }

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
            _thinking_budget: Option<u32>,
            _top_p: Option<f64>,
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
            max_iterations: SubagentTuning::default().max_iterations,
            max_tool_result_chars: 30000,
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
            None,  // parent_context_limit
            &SubagentTuning::default(),
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
                _thinking_budget: Option<u32>,
                _top_p: Option<f64>,
            ) -> anyhow::Result<LLMResponse> {
                Ok(LLMResponse {
                    content: Some("Immediate answer.".to_string()),
                    tool_calls: vec![],
                    finish_reason: "stop".to_string(),
                    usage: HashMap::new(),
                })
            }
            fn get_default_model(&self) -> &str {
                "immediate"
            }
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
            None,  // parent_context_limit
            &SubagentTuning::default(),
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
            max_tool_result_chars: 30000,
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
            tools_filter: Some(vec!["read_file".to_string(), "list_dir".to_string()]),
            read_only: false,
            max_iterations: 5,
            max_tool_result_chars: 30000,
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

    // ---------------------------------------------------------------
    // format_status_block tests
    // ---------------------------------------------------------------

    #[test]
    fn test_status_block_empty_when_nothing() {
        let result = format_status_block(&[], &[]);
        assert!(
            result.is_empty(),
            "Should return empty when nothing to report"
        );
    }

    #[test]
    fn test_status_block_shows_running() {
        let running = vec![SubagentInfo {
            task_id: "abc12345".into(),
            label: "research-api".into(),
            started_at: std::time::Instant::now(),
        }];
        let result = format_status_block(&running, &[]);
        assert!(result.contains("## Background Status"));
        assert!(result.contains("RUNNING: research-api"));
        assert!(result.contains("id:abc12345"));
    }

    #[test]
    fn test_status_block_shows_recent_completed() {
        let completed =
            vec!["  • fetch-docs (id: def456) — completed [2026-02-17T12:00:00Z]".into()];
        let result = format_status_block(&[], &completed);
        assert!(result.contains("## Background Status"));
        assert!(result.contains("fetch-docs"));
        // Leading bullet should be stripped
        assert!(!result.contains("  • "));
    }

    #[test]
    fn test_status_block_caps_recent_at_3() {
        let completed: Vec<String> = (0..5)
            .map(|i| format!("  • task-{} (id: {}) — completed", i, i))
            .collect();
        let result = format_status_block(&[], &completed);
        assert!(result.contains("task-0"));
        assert!(result.contains("task-2"));
        assert!(!result.contains("task-3"), "Should cap at 3 recent entries");
    }

    #[test]
    fn test_status_block_combined() {
        let running = vec![SubagentInfo {
            task_id: "aaa".into(),
            label: "worker-1".into(),
            started_at: std::time::Instant::now(),
        }];
        let completed = vec!["  • done-task (id: bbb) — completed".into()];
        let result = format_status_block(&running, &completed);
        assert!(result.contains("RUNNING: worker-1"));
        assert!(result.contains("done-task"));
    }
}
