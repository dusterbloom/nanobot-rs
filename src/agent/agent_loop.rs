#![allow(dead_code)]
//! Main agent loop that consumes inbound messages and produces responses.
//!
//! Ported from Python `agent/loop.py`.
//!
//! The agent loop uses a fan-out pattern for concurrent message processing:
//! messages from different sessions run in parallel (up to `max_concurrent_chats`),
//! while messages within the same session are serialized to preserve ordering.

use std::collections::HashMap;

use chrono::Utc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use serde_json::{json, Value};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use tokio::sync::{Mutex, Semaphore};
use tracing::{debug, error, info, instrument, warn};

use crate::agent::audit::{AuditLog, ToolEvent};
use crate::agent::anti_drift;
use crate::agent::context_hygiene;
use crate::agent::policy;
use crate::agent::protocol::{CloudProtocol, ConversationProtocol, LocalProtocol};
use crate::agent::reflector::Reflector;
use crate::agent::subagent::SubagentManager;
use crate::agent::system_state::{self, AhaPriority, AhaSignal, SystemState};
use crate::agent::turn::turn_from_legacy;
use crate::agent::compaction::ContextCompactor;
use crate::agent::token_budget::TokenBudget;
use crate::agent::tool_guard::ToolGuard;
use crate::agent::tools::registry::ToolRegistry;
use crate::agent::validation;
use crate::bus::events::{InboundMessage, OutboundMessage};
use crate::agent::lcm::{CompactionAction, LcmConfig, LcmEngine};
use crate::config::schema::{EmailConfig, LcmSchemaConfig, ProprioceptionConfig};
use crate::cron::service::CronService;
use crate::providers::base::{LLMResponse, StreamChunk, ToolCallRequest};

// ---------------------------------------------------------------------------
// Core types re-exported from agent_core module
// ---------------------------------------------------------------------------
use crate::agent::agent_core::{
    append_to_system_prompt, apply_compaction_result, history_limit,
    provenance_warning_role, PendingCompaction,
};
pub use crate::agent::agent_core::{
    build_swappable_core, AgentHandle, RuntimeCounters, SharedCoreHandle, SwappableCore,
    SwappableCoreConfig,
};


// ---------------------------------------------------------------------------
// Per-instance state (different per agent)
// ---------------------------------------------------------------------------

/// Per-instance state that differs between the REPL agent and gateway agents.
pub(crate) struct AgentLoopShared {
    pub(crate) core_handle: SharedCoreHandle,
    pub(crate) subagents: Arc<SubagentManager>,
    pub(crate) bus_outbound_tx: UnboundedSender<OutboundMessage>,
    #[allow(dead_code)]
    pub(crate) bus_inbound_tx: UnboundedSender<InboundMessage>,
    pub(crate) cron_service: Option<Arc<CronService>>,
    pub(crate) email_config: Option<EmailConfig>,
    pub(crate) repl_display_tx: Option<UnboundedSender<String>>,
    /// Cached memory bulletin for system prompt injection (zero-cost reads).
    pub(crate) bulletin_cache: Arc<arc_swap::ArcSwap<String>>,
    /// Shared system state for ensemble proprioception.
    pub(crate) system_state: Arc<arc_swap::ArcSwap<SystemState>>,
    /// Proprioception config (feature toggles).
    pub(crate) proprioception_config: ProprioceptionConfig,
    /// Receiver for priority signals from subagents (aha channel).
    pub(crate) aha_rx: Arc<Mutex<tokio::sync::mpsc::UnboundedReceiver<AhaSignal>>>,
    /// Sender for priority signals (given to subagent manager).
    pub(crate) aha_tx: tokio::sync::mpsc::UnboundedSender<AhaSignal>,
    /// Sticky per-session policy flags (e.g. local_only).
    pub(crate) session_policies: Arc<Mutex<HashMap<String, policy::SessionPolicy>>>,
    /// Per-session LCM engines for lossless context management.
    pub(crate) lcm_engines: Arc<Mutex<HashMap<String, Arc<tokio::sync::Mutex<LcmEngine>>>>>,
    /// LCM configuration.
    pub(crate) lcm_config: LcmSchemaConfig,
    /// Dedicated LCM compactor (when `lcm.compaction_endpoint` is configured).
    pub(crate) lcm_compactor: Option<Arc<ContextCompactor>>,
    /// Health probe registry — used to gate LCM compaction when endpoint is degraded.
    pub(crate) health_registry: Option<Arc<crate::heartbeat::health::HealthRegistry>>,
    /// Budget calibrator for recording execution stats (append-only SQLite).
    pub(crate) calibrator: Option<std::sync::Mutex<crate::agent::budget_calibrator::BudgetCalibrator>>,
}

/// Per-message state that flows through the three processing phases.
///
/// Owns all per-turn mutable state that previously lived as local variables
/// inside `process_message`. No lifetimes needed — values are cloned from the
/// inbound message where required.
pub(crate) struct TurnContext {
    // --- Config (set during prepare, immutable after) ---
    pub(crate) core: Arc<SwappableCore>,
    pub(crate) request_id: String,
    pub(crate) session_key: String,
    pub(crate) session_policy: policy::SessionPolicy,
    pub(crate) strict_local_only: bool,
    pub(crate) turn_count: u64,
    pub(crate) streaming: bool,
    pub(crate) audit: Option<AuditLog>,
    pub(crate) tools: ToolRegistry,
    pub(crate) user_content: String,
    pub(crate) channel: String,
    pub(crate) chat_id: String,
    pub(crate) is_voice_message: bool,
    pub(crate) detected_language: Option<String>,

    // --- Channels (moved into context) ---
    pub(crate) text_delta_tx: Option<tokio::sync::mpsc::UnboundedSender<String>>,
    pub(crate) tool_event_tx: Option<tokio::sync::mpsc::UnboundedSender<ToolEvent>>,
    pub(crate) cancellation_token: Option<tokio_util::sync::CancellationToken>,
    pub(crate) priority_rx: Option<tokio::sync::mpsc::UnboundedReceiver<String>>,

    // --- Conversation state ---
    pub(crate) messages: Vec<Value>,
    pub(crate) new_start: usize,
    /// Protocol-rendered wire format, computed in `step_pre_call` and used
    /// exclusively for LLM provider calls. `messages` remains the raw
    /// accumulator (with metadata tags) for trimming and session persistence.
    pub(crate) rendered_messages: Vec<Value>,
    /// Protocol selected for this turn based on `core.is_local`.
    pub(crate) protocol: Arc<dyn ConversationProtocol>,

    // --- Tracking ---
    pub(crate) used_tools: std::collections::HashSet<String>,
    pub(crate) final_content: String,
    pub(crate) turn_tool_entries: Vec<crate::agent::audit::TurnToolEntry>,
    /// Number of LLM iterations consumed in this agent turn (for calibration).
    pub(crate) iterations_used: u32,
    /// Wall-clock start of this agent turn (for duration measurement).
    pub(crate) turn_start: std::time::Instant,

    // --- Budget/compaction ---
    pub(crate) compaction: CompactionHandle,
    pub(crate) content_gate: crate::agent::context_gate::ContentGate,

    // --- Observability ---
    pub(crate) counters: Arc<RuntimeCounters>,

    // --- Flow control ---
    pub(crate) flow: FlowControl,

    // --- Health ---
    pub(crate) health_registry: Option<Arc<crate::heartbeat::health::HealthRegistry>>,

    // --- Security ---
    /// Tracks taint introduced by web tools; used to warn before sensitive tool calls.
    pub(crate) taint_state: crate::agent::taint::TaintState,
}

/// Per-turn flow control flags.
///
/// These are orthogonal booleans (not a linear state machine):
/// - `force_response`: set by exec/write_file tools, cleared after boundary injection
/// - `router_preflight_done`: one-shot, set after router runs
/// - `forced_finalize_attempted`: one-shot, set after rescue pass for empty responses
/// - `content_was_streamed`: one-shot, set when TextDelta chunks are sent
/// - `iterations_since_compaction`: counter, reset when compaction swaps in
/// - `tool_guard`: per-turn tool call policy enforcement
pub(crate) struct FlowControl {
    pub(crate) force_response: bool,
    pub(crate) router_preflight_done: bool,
    pub(crate) tool_guard: ToolGuard,
    pub(crate) iterations_since_compaction: u32,
    pub(crate) forced_finalize_attempted: bool,
    pub(crate) content_was_streamed: bool,
    /// Consecutive rounds where ALL tool calls were blocked by the guard.
    /// When this reaches the threshold, the loop forces a text response.
    pub(crate) consecutive_all_blocked: u32,
    /// When the LLM call started — set in step_call_llm, read in step_process_response.
    pub(crate) llm_call_start: Option<std::time::Instant>,
}

/// Shared handles for background compaction coordination.
pub(crate) struct CompactionHandle {
    pub(crate) slot: Arc<tokio::sync::Mutex<Option<PendingCompaction>>>,
    pub(crate) in_flight: Arc<AtomicBool>,
}

// ---------------------------------------------------------------------------
// Iteration state machine
// ---------------------------------------------------------------------------

/// The phase within a single agent loop iteration.
///
/// Each variant carries only the data needed for that phase.
/// Transitions are driven by the return value of each step method.
enum IterationPhase {
    /// Pre-LLM housekeeping: context hygiene, proprioception, aha channel,
    /// heartbeat injection, compaction check.
    Preparing,
    /// Response boundary injection, tool definition filtering, message
    /// trimming, compaction spawn, protocol repair, pre-flight check,
    /// router preflight, adaptive max_tokens.
    PreCall,
    /// Call LLM (streaming or blocking).
    Calling { tool_defs: Vec<Value>, max_tokens: u32 },
    /// Validate response, rescue pass, error check, token telemetry.
    Processing { response: LLMResponse },
    /// Route and execute tool calls (delegated or inline).
    Executing { response: LLMResponse, tool_calls: Vec<ToolCallRequest> },
}

/// Outcome of a single iteration, returned to the outer loop.
enum IterationOutcome {
    /// Continue to next iteration.
    Continue,
    /// Agent produced final content — use as response.
    Finished(String),
    /// Error occurred — use as final content.
    Error(String),
}

/// What a step function produces: either the next phase or a terminal outcome.
enum StepResult {
    /// Transition to the next phase within this iteration.
    Next(IterationPhase),
    /// Iteration is done — report outcome to the outer loop.
    Done(IterationOutcome),
}

impl AgentLoopShared {

    /// Process an inbound message through the agent loop.
    ///
    /// When `text_delta_tx` is `Some`, text deltas are streamed to the sender
    /// as they arrive (used by CLI/voice). When `None`, a blocking LLM call
    /// is used (gateway mode).
    ///
    /// This method takes `&self` and is safe to call from multiple concurrent
    /// tasks. Per-message tool instances eliminate shared-context races.
    async fn process_message(
        &self,
        msg: &InboundMessage,
        text_delta_tx: Option<tokio::sync::mpsc::UnboundedSender<String>>,
        tool_event_tx: Option<tokio::sync::mpsc::UnboundedSender<ToolEvent>>,
        cancellation_token: Option<tokio_util::sync::CancellationToken>,
        priority_rx: Option<tokio::sync::mpsc::UnboundedReceiver<String>>,
    ) -> Option<OutboundMessage> {
        let request_id = uuid::Uuid::new_v4().to_string()[..8].to_string();
        let core = self.core_handle.swappable();
        info!(
            request_id = %request_id,
            role = "main",
            model = %core.model,
            channel = %msg.channel,
            "request_start"
        );
        drop(core);

        let mut ctx = self.prepare_context(msg, text_delta_tx, tool_event_tx, cancellation_token, priority_rx).await;
        self.run_agent_loop(&mut ctx).await;
        self.finalize_response(ctx).await
    }

    /// Phase 1: Build the [`TurnContext`] from the inbound message.
    ///
    /// Snapshots the swappable core, extracts session info, builds tools,
    /// loads history, constructs the message array, and initialises all
    /// per-turn tracking state.
    async fn prepare_context(
        &self,
        msg: &InboundMessage,
        text_delta_tx: Option<tokio::sync::mpsc::UnboundedSender<String>>,
        tool_event_tx: Option<tokio::sync::mpsc::UnboundedSender<ToolEvent>>,
        cancellation_token: Option<tokio_util::sync::CancellationToken>,
        priority_rx: Option<tokio::sync::mpsc::UnboundedReceiver<String>>,
    ) -> TurnContext {
        let streaming = text_delta_tx.is_some();

        // Snapshot core — instant Arc clone under brief read lock.
        let core = self.core_handle.swappable();
        let counters = &self.core_handle.counters;
        let turn_count = counters
            .learning_turn_counter
            .fetch_add(1, Ordering::Relaxed)
            + 1;
        if turn_count % 50 == 0 {
            core.learning.prune();
        }

        let session_key = msg
            .metadata
            .get("session_key")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("{}:{}", msg.channel, msg.chat_id));

        let session_policy = {
            let mut map = self.session_policies.lock().await;
            let entry = map.entry(session_key.clone()).or_default();
            if core.tool_delegation_config.strict_local_only {
                entry.local_only = true;
            }
            policy::update_from_user_text(entry, &msg.content);
            entry.clone()
        };
        let strict_local_only =
            core.tool_delegation_config.strict_local_only || session_policy.local_only;

        debug!(
            "Processing message{} from {} on {}: {}",
            if streaming { " (streaming)" } else { "" },
            msg.sender_id,
            msg.channel,
            &msg.content[..msg.content.len().min(80)]
        );

        // Create audit log if provenance is enabled.
        let audit = if core.provenance_config.enabled && core.provenance_config.audit_log {
            Some(AuditLog::new(&core.workspace, &session_key))
        } else {
            None
        };

        // Build per-message tools with context baked in.
        let mut tools = self.build_tools(&core, &msg.channel, &msg.chat_id).await;

        // Register lcm_expand tool when LCM is enabled.
        if self.lcm_config.enabled {
            let lcm_engine = {
                let mut engines = self.lcm_engines.lock().await;
                engines
                    .entry(session_key.clone())
                    .or_insert_with(|| {
                        let config = LcmConfig::from(&self.lcm_config);
                        Arc::new(tokio::sync::Mutex::new(LcmEngine::new(config)))
                    })
                    .clone()
            };
            use crate::agent::lcm::LcmExpandTool;
            tools.register(Box::new(LcmExpandTool::new(lcm_engine)));
        }

        // Get session history. Track count so we know where new messages start.
        let history = core
            .sessions
            .get_history(
                &session_key,
                history_limit(core.token_budget.max_context()),
                core.max_history_turns,
            )
            .await;
        // Track where new (unsaved) messages start. Updated after compaction
        // swaps to avoid re-persisting already-saved messages.
        let new_start = 1 + history.len();

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

        // Build messages.
        let is_voice_message = msg
            .metadata
            .get("voice_message")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let detected_language = msg
            .metadata
            .get("detected_language")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        // /no_think is handled at the provider level (system prompt via native
        // LMS API for Nemotron) — never inject it into user content where the
        // model treats it as literal text and leaks it into tool arguments.
        let user_content = msg.content.clone();
        let mut messages = core.context.build_messages(
            &history,
            &user_content,
            None,
            if media_paths.is_empty() {
                None
            } else {
                Some(&media_paths)
            },
            Some(&msg.channel),
            Some(&msg.chat_id),
            is_voice_message,
            detected_language.as_deref(),
        );

        // Inject per-session working memory into the system message.
        if core.memory_enabled {
            let mut wm = core
                .working_memory
                .get_context(&session_key, core.working_memory_budget);
            // Append learning context (tool patterns) if available.
            let learning_ctx = core.learning.get_learning_context();
            if !learning_ctx.is_empty() {
                wm.push_str("\n\n## Tool Patterns\n\n");
                wm.push_str(&learning_ctx);
            }
            if !wm.is_empty() {
                append_to_system_prompt(
                    &mut messages,
                    &format!("\n\n---\n\n# Working Memory (Current Session)\n\n{}", wm),
                );
            }
        }

        // Inject recent daily notes for local mode continuity.
        if core.is_local && core.memory_enabled {
            let memory_store = crate::agent::memory::MemoryStore::new(&core.workspace);
            let notes = memory_store.read_recent_daily_notes(3);
            if !notes.is_empty() {
                append_to_system_prompt(
                    &mut messages,
                    &format!("\n\n## Recent Notes\n\n{}", notes),
                );
            }
        }

        // Auto-inject background task status into system prompt so the agent
        // is naturally aware of running/completed subagents without explicit tool calls.
        {
            let running = self.subagents.list_running().await;
            let recent =
                crate::agent::subagent::SubagentManager::read_recent_completed(&core.workspace, 5);
            let status = crate::agent::subagent::format_status_block(&running, &recent);
            if !status.is_empty() {
                append_to_system_prompt(&mut messages, &status);
            }
        }

        // Inject memory bulletin if available (zero-cost Arc load).
        {
            let bulletin = self.bulletin_cache.load_full();
            if !bulletin.is_empty() {
                append_to_system_prompt(
                    &mut messages,
                    &format!("\n\n## Memory Briefing\n\n{}", &*bulletin),
                );
            }
        }

        // Tag the current user message (last in the array) with turn number
        // for age-based eviction in trim_to_fit.
        if let Some(last) = messages.last_mut() {
            last["_turn"] = json!(turn_count);
        }

        // Background compaction state.
        let compaction_slot: Arc<tokio::sync::Mutex<Option<PendingCompaction>>> =
            Arc::new(tokio::sync::Mutex::new(None));
        let compaction_in_flight = Arc::new(AtomicBool::new(false));

        // Context gate: budget-aware content sizing for this turn.
        let cache_dir = crate::utils::helpers::get_data_path()
            .join("cache")
            .join("tool_outputs");
        let mut content_gate = crate::agent::context_gate::ContentGate::new(
            core.token_budget.max_context(),
            0.20,
            cache_dir,
        );
        // Pre-consume the tokens already used by system prompt + history.
        let initial_tokens = TokenBudget::estimate_tokens(&messages);
        content_gate.budget.consume(initial_tokens);

        let tool_guard = ToolGuard::new(core.tool_delegation_config.max_same_tool_call_per_turn);

        let request_id = uuid::Uuid::new_v4().to_string()[..8].to_string();
        info!(
            request_id = %request_id,
            model = %core.model,
            session = %session_key,
            turn = turn_count,
            "request_started"
        );

        // Select conversation protocol based on whether we're talking to a local model.
        // Protocol correctness is enforced at render time — no repair needed.
        let protocol: Arc<dyn ConversationProtocol> = if core.is_local {
            Arc::new(LocalProtocol)
        } else {
            Arc::new(CloudProtocol)
        };

        TurnContext {
            core,
            request_id,
            session_key,
            session_policy,
            strict_local_only,
            turn_count,
            streaming,
            audit,
            tools,
            user_content,
            channel: msg.channel.clone(),
            chat_id: msg.chat_id.clone(),
            is_voice_message,
            detected_language,
            text_delta_tx,
            tool_event_tx,
            cancellation_token,
            priority_rx,
            messages,
            new_start,
            rendered_messages: Vec::new(),
            protocol,
            used_tools: std::collections::HashSet::new(),
            final_content: String::new(),
            turn_tool_entries: Vec::new(),
            iterations_used: 0,
            turn_start: std::time::Instant::now(),
            compaction: CompactionHandle {
                slot: compaction_slot,
                in_flight: compaction_in_flight,
            },
            content_gate,
            counters: self.core_handle.counters.clone(),
            flow: FlowControl {
                force_response: false,
                router_preflight_done: false,
                tool_guard,
                iterations_since_compaction: 0,
                forced_finalize_attempted: false,
                content_was_streamed: false,
                consecutive_all_blocked: 0,
                llm_call_start: None,
            },
            health_registry: self.health_registry.clone(),
            taint_state: crate::agent::taint::TaintState::new(),
        }
    }

    /// Phase 2: Run the main agent loop (LLM calls + tool execution).
    ///
    /// Thin loop driver: delegates each iteration to [`run_iteration`] which
    /// drives the inner state machine through `IterationPhase` steps.
    #[instrument(name = "agent_loop", skip(self, ctx), fields(
        session = %ctx.session_key,
        mode = if ctx.core.is_local && ctx.core.tool_delegation_config.strict_no_tools_main { "trio" } else { "inline" },
        model = %ctx.core.model,
        streaming = ctx.streaming,
    ))]
    async fn run_agent_loop(&self, ctx: &mut TurnContext) {
        for iteration in 0..ctx.core.max_iterations {
            debug!(
                "Agent iteration{} {}/{}",
                if ctx.streaming { " (streaming)" } else { "" },
                iteration + 1,
                ctx.core.max_iterations
            );

            ctx.iterations_used = iteration + 1;
            let outcome = self.run_iteration(ctx, iteration).await;
            match outcome {
                IterationOutcome::Continue => continue,
                IterationOutcome::Finished(content) => {
                    ctx.final_content = content;
                    break;
                }
                IterationOutcome::Error(msg) => {
                    ctx.final_content = msg;
                    break;
                }
            }
        }

        // If the loop exited via a non-streaming path (e.g. router preflight
        // decision, error, ask_user) the final_content was set directly without
        // any text deltas being sent through the streaming channel.  Emit it
        // now so the REPL's incremental renderer actually displays something.
        // Skip if content was already streamed via TextDelta chunks to avoid
        // duplication.
        if !ctx.final_content.is_empty() && !ctx.flow.content_was_streamed {
            if let Some(ref tx) = ctx.text_delta_tx {
                let _ = tx.send(ctx.final_content.clone());
            }
        }
    }

    /// Drive a single iteration through the phase state machine.
    async fn run_iteration(&self, ctx: &mut TurnContext, iteration: u32) -> IterationOutcome {
        let mut phase = IterationPhase::Preparing;
        loop {
            match match phase {
                IterationPhase::Preparing =>
                    self.step_prepare(ctx, iteration).await,
                IterationPhase::PreCall =>
                    self.step_pre_call(ctx, iteration).await,
                IterationPhase::Calling { tool_defs, max_tokens } =>
                    self.step_call_llm(ctx, tool_defs, max_tokens).await,
                IterationPhase::Processing { response } =>
                    self.step_process_response(ctx, response).await,
                IterationPhase::Executing { response, tool_calls } =>
                    self.step_execute_tools(ctx, response, tool_calls).await,
            } {
                StepResult::Next(next_phase) => phase = next_phase,
                StepResult::Done(outcome) => return outcome,
            }
        }
    }

    // -----------------------------------------------------------------------
    // Step 1: Preparing — pre-LLM housekeeping
    // -----------------------------------------------------------------------

    /// Context hygiene, proprioception, aha channel, heartbeat,
    /// compaction-check, iteration counter.
    #[instrument(name = "step_prepare", skip(self, ctx), fields(iteration))]
    async fn step_prepare(&self, ctx: &mut TurnContext, iteration: u32) -> StepResult {
        let counters = &self.core_handle.counters;

        // --- Context Hygiene: clean up conversation history ---
        context_hygiene::hygiene_pipeline(&mut ctx.messages, ctx.core.hygiene_keep_last_messages);

        // --- Anti-Drift: quality-based cleanup for local models ---
        if ctx.core.is_local && ctx.core.anti_drift.enabled {
            anti_drift::pre_completion_pipeline(&mut ctx.messages, iteration, &ctx.core.anti_drift);
        }

        // --- Proprioception: update SystemState ---
        if self.proprioception_config.enabled {
            let tools_list: Vec<String> = if let Ok(guard) = counters.last_tools_called.lock() {
                guard.clone()
            } else {
                Vec::new()
            };
            let tool_refs: Vec<&str> = tools_list.iter().map(|s| s.as_str()).collect();
            let phase = system_state::infer_phase(&tool_refs);
            let active_subs = self.subagents.list_running().await.len().min(255) as u8;
            let state = SystemState::snapshot(
                phase,
                counters.last_context_used.load(Ordering::Relaxed),
                counters.last_context_max.load(Ordering::Relaxed),
                ctx.turn_count,
                ctx.messages.len() as u64,
                ctx.flow.iterations_since_compaction,
                counters.delegation_healthy.load(Ordering::Relaxed),
                0,    // recent_tool_failures — not tracked yet
                true, // last_tool_ok
                active_subs,
                0, // pending_aha_signals filled below
            );
            self.system_state.store(Arc::new(state));
        }

        // --- Aha Channel: poll priority signals from subagents ---
        if self.proprioception_config.enabled && self.proprioception_config.aha_channel {
            if let Ok(mut rx) = self.aha_rx.try_lock() {
                while let Ok(signal) = rx.try_recv() {
                    match signal.priority {
                        AhaPriority::Critical => {
                            ctx.messages.push(json!({
                                "role": "user",
                                "content": format!(
                                    "[ALERT from subagent {}] {}",
                                    signal.agent_id, signal.message
                                )
                            }));
                        }
                        AhaPriority::High => {
                            ctx.messages.push(json!({
                                "role": "user",
                                "content": format!(
                                    "[Signal from subagent {}] {}",
                                    signal.agent_id, signal.message
                                )
                            }));
                        }
                        AhaPriority::Normal => {
                            // Normal signals are informational — logged only.
                            debug!(
                                "Aha signal (normal) from {}: {}",
                                signal.agent_id, signal.message
                            );
                        }
                    }
                }
            }
        }

        // --- Heartbeat: inject grounding message ---
        if self.proprioception_config.enabled {
            let state = self.system_state.load_full();
            if system_state::should_ground(
                iteration,
                self.proprioception_config.grounding_interval,
                state.context_pressure,
            ) {
                let grounding = system_state::format_grounding(&state);
                ctx.messages.push(json!({
                    "role": "user",
                    "content": grounding
                }));
            }
        }

        ctx.flow.iterations_since_compaction += 1;

        // Check if background compaction finished — swap in compacted messages.
        if let Ok(mut guard) = ctx.compaction.slot.try_lock() {
            if let Some(pending) = guard.take() {
                debug!(
                    "Compaction swap: {} msgs -> {} compacted + {} new",
                    pending.watermark,
                    pending.result.messages.len(),
                    ctx.messages.len().saturating_sub(pending.watermark)
                );
                apply_compaction_result(&mut ctx.messages, pending);
                // After compaction, all messages in the array are "new" from
                // the perspective of persistence (the session file was rebuilt).
                ctx.new_start = ctx.messages.len();
                ctx.flow.iterations_since_compaction = 0;
            }
        }

        StepResult::Next(IterationPhase::PreCall)
    }

    // -----------------------------------------------------------------------
    // Step 2: PreCall — build tool defs, trim, compaction, repair, preflight
    // -----------------------------------------------------------------------

    /// Response boundary injection, tool definition filtering, message
    /// trimming, background compaction spawn, protocol repair, pre-flight
    /// context size check, router preflight, adaptive max_tokens.
    #[instrument(name = "step_pre_call", skip(self, ctx), fields(
        iteration,
        trio_mode = ctx.core.is_local && ctx.core.tool_delegation_config.strict_no_tools_main,
        boundary_active = ctx.flow.force_response,
        msg_count = ctx.messages.len(),
    ))]
    async fn step_pre_call(&self, ctx: &mut TurnContext, iteration: u32) -> StepResult {
        let counters = &self.core_handle.counters;

        // Response boundary: suppress exec/write_file tools to force text output.
        let boundary_active = ctx.flow.force_response
            && ctx.core.provenance_config.enabled
            && ctx.core.provenance_config.response_boundary;
        if boundary_active {
            // Use "user" role, not "system". The Anthropic OpenAI-compat
            // endpoint strips mid-conversation system messages, which would
            // leave the conversation ending with an assistant message and
            // trigger a "does not support assistant message prefill" error.
            let remaining = ctx.core.max_iterations.saturating_sub(iteration as u32 + 1);
            let budget_note = if remaining <= 5 {
                format!(
                    " [Budget: {}/{} iterations remaining — wrap up soon]",
                    remaining, ctx.core.max_iterations
                )
            } else {
                String::new()
            };
            ctx.messages.push(json!({
                "role": "user",
                "content": format!(
                    "[system] You just executed a tool that modifies files or runs commands. \
                     Report the result to the user before making additional tool calls.{budget_note}"
                )
            }));
            ctx.flow.force_response = false;
        }

        // Filter tool definitions to relevant tools.
        // Local models get a minimal set to conserve context tokens.
        let current_phase = self.system_state.load_full().task_phase;
        let mut tool_defs = if ctx.core.is_local {
            ctx.tools.get_local_definitions(&ctx.messages, &ctx.used_tools)
        } else if self.proprioception_config.enabled
            && self.proprioception_config.dynamic_tool_scoping
        {
            ctx.tools.get_scoped_definitions(&current_phase, &ctx.messages, &ctx.used_tools)
        } else {
            ctx.tools.get_relevant_definitions(&ctx.messages, &ctx.used_tools)
        };
        // Save tool_defs before potential stripping so we can restore them if
        // the router preflight returns Passthrough (router said "respond") — in
        // that case the main model must have tools as fallback.
        let saved_tool_defs = tool_defs.clone();
        if ctx.core.is_local && ctx.core.tool_delegation_config.strict_no_tools_main {
            // Hard separation (local trio only): main model is conversation/orchestration only.
            // Cloud providers handle tools natively and must never have them stripped.
            // BUT: if trio routing is degraded, keep tools so main model can still act.
            let router_probe_healthy = self.health_registry
                .as_ref()
                .map_or(false, |reg| reg.is_healthy("trio_router"));
            // Use the same key format as router.rs: "router:{model}".
            // Fallback to "trio_router" only when no router model is configured
            // (in which case trio won't run anyway).
            let cb_key = ctx.core.router_model
                .as_deref()
                .map_or_else(|| "trio_router".to_string(), |m| format!("router:{}", m));
            let cb_available = ctx.counters.trio_circuit_breaker.lock().unwrap()
                .is_available(&cb_key);
            if should_strip_tools_for_trio(
                ctx.core.is_local,
                ctx.core.tool_delegation_config.strict_no_tools_main,
                router_probe_healthy,
                cb_available,
            ) {
                ctx.counters.set_trio_state(crate::agent::agent_core::TrioState::Active);
                tool_defs.clear();
                // Tell the main model it's in orchestration mode (tools stripped).
                append_to_system_prompt(&mut ctx.messages, concat!(
                    "\n\n## Orchestration Mode (Active)\n",
                    "A trio routing system handles tool execution on your behalf.\n",
                    "- You do NOT have direct tool access in this mode.\n",
                    "- If a tool result appears as `[router:tool:X]` or `[specialist:X]`, ",
                    "incorporate that result into your response.\n",
                    "- If you need additional tool actions, describe them clearly ",
                    "(e.g., \"I need to read src/main.rs\") and the next turn will route it.\n",
                    "- Focus on reasoning, planning, and conversation.\n",
                ));
            } else {
                ctx.counters.set_trio_state(crate::agent::agent_core::TrioState::Degraded);
                debug!("trio degraded — keeping tools for main model fallback");
            }
        }
        if boundary_active {
            tool_defs.retain(|def| {
                let name = def
                    .pointer("/function/name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                name != "exec" && name != "write_file"
            });
        }
        let tool_defs_opt: Option<&[Value]> = if tool_defs.is_empty() {
            None
        } else {
            Some(&tool_defs)
        };

        // Trim messages to fit context budget.
        let tool_def_tokens =
            TokenBudget::estimate_tool_def_tokens(tool_defs_opt.unwrap_or(&[]));
        ctx.messages = ctx.core.token_budget.trim_to_fit_with_age(
            &ctx.messages,
            tool_def_tokens,
            ctx.turn_count,
            ctx.core.max_message_age_turns,
        );

        // Spawn background compaction when threshold exceeded.
        // When LCM is enabled, use the LCM engine's control loop instead.
        if self.lcm_config.enabled {
            // LCM path: get or create per-session engine, check thresholds.
            // On first creation, check for existing summaries and rebuild DAG if present.
            let lcm_engine = {
                let mut engines = self.lcm_engines.lock().await;
                if !engines.contains_key(&ctx.session_key) {
                    let config = LcmConfig::from(&self.lcm_config);
                    
                    // Check if session has existing summary turns.
                    let all_msgs = ctx.core.sessions.get_all_messages(&ctx.session_key).await;
                    let turns: Vec<crate::agent::turn::Turn> = all_msgs
                        .iter()
                        .filter_map(|v| crate::agent::turn::turn_from_legacy(v))
                        .collect();
                    let has_summaries = turns.iter().any(|t| t.is_summary());
                    
                    let engine = if has_summaries {
                        // Rebuild from persisted summaries.
                        debug!(
                            session = %ctx.session_key,
                            summary_count = turns.iter().filter(|t| t.is_summary()).count(),
                            "LCM: rebuilding engine from persisted summaries"
                        );
                        LcmEngine::rebuild_from_turns(
                            &turns,
                            config,
                            ctx.protocol.as_ref(),
                            "", // system prompt not needed for rebuild
                        )
                    } else {
                        // Fresh engine - will ingest messages below.
                        LcmEngine::new(config)
                    };
                    
                    engines.insert(ctx.session_key.clone(), Arc::new(tokio::sync::Mutex::new(engine)));
                }
                engines.get(&ctx.session_key).cloned().unwrap()
            };

            // Feed messages into the LCM engine's store (idempotent by index).
            {
                let mut engine = lcm_engine.lock().await;
                let store_len = engine.store_len();
                for msg in ctx.messages.iter().skip(store_len) {
                    engine.ingest(msg.clone());
                }
            }

            // Check thresholds and spawn compaction if needed.
            // Pre-flight: skip LCM compaction if endpoint is degraded.
            let lcm_healthy = self.health_registry
                .as_ref()
                .map_or(true, |reg| reg.is_healthy("lcm_compaction"));
            if !lcm_healthy {
                debug!("LCM compaction skipped: endpoint degraded");
            }
            if lcm_healthy && !ctx.compaction.in_flight.load(Ordering::Relaxed) {
                let action = {
                    let engine = lcm_engine.lock().await;
                    engine.check_thresholds(&ctx.core.token_budget, tool_def_tokens)
                };

                match action {
                    CompactionAction::Async | CompactionAction::Blocking => {
                        tracing::info!(
                            compaction_type = if action == CompactionAction::Async { "lcm_async" } else { "lcm_blocking" },
                            msg_count = ctx.messages.len(),
                            "lcm_compaction_triggered"
                        );
                        let slot = ctx.compaction.slot.clone();
                        let in_flight = ctx.compaction.in_flight.clone();
                        let bg_messages = ctx.messages.clone();
                        let bg_core = ctx.core.clone();
                        let bg_session_key = ctx.session_key.clone();
                        let bg_lcm = lcm_engine.clone();
                        let bg_lcm_compactor = self.lcm_compactor.clone();
                        let watermark = ctx.messages.len();
                        let bg_turn_count = ctx.turn_count;
                        in_flight.store(true, Ordering::SeqCst);

                        if action == CompactionAction::Async {
                            // Mark async pending so we don't re-trigger.
                            let mut engine = lcm_engine.lock().await;
                            engine.request_async_compaction();
                        }

                        tokio::spawn(async move {
                            let timeout_result = tokio::time::timeout(
                                Duration::from_secs(30),
                                async {
                                    // Use dedicated LCM compactor if configured,
                                    // otherwise fall back to the core memory compactor.
                                    let compactor: &ContextCompactor = bg_lcm_compactor
                                        .as_deref()
                                        .unwrap_or(&bg_core.compactor);
                                    let summary_turn = {
                                        let mut engine = bg_lcm.lock().await;
                                        engine
                                            .compact(compactor, &bg_core.token_budget, 0)
                                            .await
                                    };

                                    // Extract text from Turn::Summary for working memory and result.
                                    let observation: Option<String> = summary_turn.as_ref().and_then(|t| {
                                        if let crate::agent::turn::Turn::Summary { text, .. } = t {
                                            Some(text.clone())
                                        } else {
                                            None
                                        }
                                    });

                                    // Persist Turn::Summary to session JSONL for lossless restart.
                                    if let Some(ref turn) = summary_turn {
                                        if let Some(summary_json) = turn.summary_to_json() {
                                            debug!(
                                                session = %bg_session_key,
                                                "LCM: persisting summary turn to session"
                                            );
                                            bg_core.sessions.add_message_raw(&bg_session_key, &summary_json).await;
                                        }
                                    }

                                    // Update working memory with compaction observation.
                                    if bg_core.memory_enabled {
                                        if let Some(ref summary_text) = observation {
                                            bg_core
                                                .working_memory
                                                .update_from_compaction(&bg_session_key, summary_text, bg_turn_count);
                                        }
                                    }

                                    // Build CompactionResult from LCM's active context.
                                    let compacted_messages = {
                                        let engine = bg_lcm.lock().await;
                                        engine.active_context()
                                    };

                                    if compacted_messages.len() < bg_messages.len() {
                                        let result = crate::agent::compaction::CompactionResult {
                                            messages: compacted_messages,
                                            observation,
                                        };
                                        *slot.lock().await =
                                            Some(PendingCompaction { result, watermark });
                                    }
                                },
                            )
                            .await;
                            if timeout_result.is_err() {
                                warn!("LCM compaction timed out after 30s, resetting in_flight");
                            }
                            in_flight.store(false, Ordering::SeqCst);
                        });
                    }
                    CompactionAction::None => {}
                }
            }
        } else if !ctx.compaction.in_flight.load(Ordering::Relaxed)
            && ctx.core
                .compactor
                .needs_compaction(&ctx.messages, &ctx.core.token_budget, tool_def_tokens)
        {
            tracing::info!(
                compaction_type = "core_async",
                msg_count = ctx.messages.len(),
                "core_compaction_triggered"
            );
            let slot = ctx.compaction.slot.clone();
            let in_flight = ctx.compaction.in_flight.clone();
            let bg_messages = ctx.messages.clone();
            let bg_core = ctx.core.clone();
            let bg_session_key = ctx.session_key.clone();
            let watermark = ctx.messages.len();
            let bg_turn_count = ctx.turn_count;
            in_flight.store(true, Ordering::SeqCst);

            let bg_proprio = self.proprioception_config.clone();
            tokio::spawn(async move {
                let timeout_result = tokio::time::timeout(
                    Duration::from_secs(30),
                    async {
                        let result = if bg_proprio.enabled && bg_proprio.gradient_memory {
                            bg_core
                                .compactor
                                .compact_gradient(
                                    &bg_messages,
                                    &bg_core.token_budget,
                                    0,
                                    bg_proprio.raw_window,
                                    bg_proprio.light_window,
                                )
                                .await
                        } else if bg_proprio.enabled && bg_proprio.audience_aware_compaction {
                            let reader =
                                crate::agent::compaction::ReaderProfile::from_model(&bg_core.model);
                            bg_core
                                .compactor
                                .compact_for_reader(&bg_messages, &bg_core.token_budget, 0, &reader)
                                .await
                        } else {
                            bg_core
                                .compactor
                                .compact(&bg_messages, &bg_core.token_budget, 0)
                                .await
                        };
                        if bg_core.memory_enabled {
                            if let Some(ref summary) = result.observation {
                                bg_core
                                    .working_memory
                                    .update_from_compaction(&bg_session_key, summary, bg_turn_count);
                            }
                        }
                        if result.messages.len() < bg_messages.len() {
                            *slot.lock().await = Some(PendingCompaction { result, watermark });
                        }
                    },
                )
                .await;
                if timeout_result.is_err() {
                    warn!("Core compaction timed out after 30s, resetting in_flight");
                }
                in_flight.store(false, Ordering::SeqCst);
            });
        }

        // Render protocol-correct wire format for the LLM call.
        // `ctx.messages` retains raw format (with metadata) for trimming/LCM.
        // `ctx.rendered_messages` is what gets sent to the provider.
        ctx.rendered_messages = render_via_protocol(&*ctx.protocol, &ctx.messages);

        // Pre-flight context size check: emergency trim if we're about to
        // exceed the model's context window. The 95% threshold leaves room
        // for the response tokens.
        let estimated = TokenBudget::estimate_tokens(&ctx.rendered_messages);
        let max_ctx = ctx.core.token_budget.max_context();
        if max_ctx > 0 && estimated > (max_ctx as f64 * 0.95) as usize {
            warn!(
                estimated_tokens = estimated,
                max_context = max_ctx,
                model = %ctx.core.model,
                "context_overflow_emergency_trim"
            );
            // tool_def_tokens=0 is conservative (trims more aggressively).
            ctx.messages = ctx.core.token_budget.trim_to_fit(&ctx.messages, 0);
            // Re-render after trim to rebuild protocol-correct wire format.
            ctx.rendered_messages = render_via_protocol(&*ctx.protocol, &ctx.messages);
        }

        // Router-first preflight for strict trio mode.
        match crate::agent::router::router_preflight(ctx, self.health_registry.as_deref()).await {
            crate::agent::router::PreflightResult::Continue => {
                return StepResult::Done(IterationOutcome::Continue);
            }
            crate::agent::router::PreflightResult::Break(msg) => {
                return StepResult::Done(IterationOutcome::Finished(msg));
            }
            crate::agent::router::PreflightResult::Passthrough => {
                // Router decided not to handle this request — restore tools so
                // the main model can still call them directly as a fallback.
                // Without this, tool_defs was cleared in the trio stripping block
                // above and the main model would answer "I cannot directly do X"
                // instead of calling list_dir, exec, etc.
                if tool_defs.is_empty() && !saved_tool_defs.is_empty() {
                    debug!("router_preflight=Passthrough — restoring tool_defs for main model fallback");
                    tool_defs = saved_tool_defs;
                }
            }
            crate::agent::router::PreflightResult::Pipeline(_steps_json) => {
                info!("[trio] pipeline action received");
                // Message already injected by router_preflight.
                // Continue to main model — full pipeline execution TBD.
            }
        }

        // Adaptive max_tokens: size the response budget to the task.
        let effective_max_tokens = {
            let base = ctx.core.max_tokens;
            // Check for /long override (temporary boost).
            let had_long = counters.long_mode_turns
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
                    if v > 0 { Some(v - 1) } else { None }
                })
                .is_ok();
            if had_long {
                base.max(8192)
            } else {
                // Detect long-form triggers in the user message.
                let user_text = ctx.messages
                    .last()
                    .and_then(|m| m.get("content"))
                    .and_then(|c| c.as_str())
                    .unwrap_or("");
                let lower = user_text.to_lowercase();
                let is_long_form = lower.contains("explain in detail")
                    || lower.contains("write a ")
                    || lower.contains("create a script")
                    || lower.contains("write code")
                    || lower.contains("implement ")
                    || lower.contains("full example")
                    || lower.starts_with("write ")
                    || user_text.len() > 500;
                // Count recent tool calls: if tool-heavy, use smaller budget.
                let recent_tool_calls = ctx.messages
                    .iter()
                    .rev()
                    .take(6)
                    .filter(|m| m.get("role").and_then(|r| r.as_str()) == Some("tool"))
                    .count();
                if is_long_form {
                    base.max(4096)
                } else if recent_tool_calls > 3 {
                    base.min(1024).max(512)
                } else {
                    base
                }
            }
        };

        StepResult::Next(IterationPhase::Calling {
            tool_defs,
            max_tokens: effective_max_tokens,
        })
    }

    // -----------------------------------------------------------------------
    // Step 3: Calling — invoke the LLM (streaming or blocking)
    // -----------------------------------------------------------------------

    /// Thinking budget calculation, inference_active flag, streaming path
    /// (with cancellation support) or blocking path.
    #[instrument(name = "step_call_llm", skip(self, ctx, tool_defs), fields(
        model = %ctx.core.model,
        streaming = ctx.streaming,
        max_tokens,
        n_tool_defs = tool_defs.len(),
    ))]
    async fn step_call_llm(
        &self,
        ctx: &mut TurnContext,
        tool_defs: Vec<Value>,
        max_tokens: u32,
    ) -> StepResult {
        let counters = &self.core_handle.counters;
        let tool_defs_opt: Option<&[Value]> = if tool_defs.is_empty() {
            None
        } else {
            Some(&tool_defs)
        };

        let thinking_budget = {
            let stored = counters.thinking_budget.load(Ordering::Relaxed);
            // Reasoning params are user-controlled via /think — any model can receive them.
            // The provider layer omits params entirely when budget is None, so non-thinking
            // models get a clean request with no unknown fields.
            if stored > 0 {
                // Small local models can burn the whole completion budget in reasoning.
                // Hard-cap explicit thinking to keep them action-oriented.
                if ctx.core.is_local && ctx.core.model_capabilities.size_class == crate::agent::model_capabilities::ModelSizeClass::Small {
                    Some(stored.min(256))
                } else {
                    Some(stored)
                }
            } else {
                None
            }
        };
        // Signal watchdog: LLM inference is active — skip health checks.
        counters.inference_active.store(true, Ordering::Relaxed);
        ctx.flow.llm_call_start = Some(std::time::Instant::now());

        // Use the protocol-rendered wire format for the provider call.
        // `ctx.rendered_messages` was computed by `render_via_protocol()` in step_pre_call.
        let messages_for_llm = if ctx.rendered_messages.is_empty() {
            // Fallback: render now if step_pre_call was bypassed (should not happen in practice).
            render_via_protocol(&*ctx.protocol, &ctx.messages)
        } else {
            ctx.rendered_messages.clone()
        };

        let response = if let Some(ref delta_tx) = ctx.text_delta_tx {
            // Streaming path: forward text deltas as they arrive.
            let mut stream = match ctx.core
                .provider
                .chat_stream(
                    &messages_for_llm,
                    tool_defs_opt,
                    Some(&ctx.core.model),
                    max_tokens,
                    ctx.core.temperature,
                    thinking_budget,
                    None,
                )
                .await
            {
                Ok(s) => s,
                Err(e) => {
                    counters.inference_active.store(false, Ordering::Relaxed);
                    error!(model = %ctx.core.model, error = %e, "llm_stream_call_failed");
                    return StepResult::Done(IterationOutcome::Error(
                        format!("I encountered an error: {}", e),
                    ));
                }
            };

            let mut streamed_response = None;
            let mut in_thinking = false;
            let suppress_thinking = counters.suppress_thinking_in_tts.load(Ordering::Relaxed);
            loop {
                tokio::select! {
                    biased;
                    _ = async {
                        if let Some(ref token) = ctx.cancellation_token {
                            token.cancelled().await;
                        } else {
                            std::future::pending::<()>().await;
                        }
                    } => {
                        // Cancelled — drop stream to signal provider task.
                        drop(stream);
                        break;
                    }
                    chunk = stream.rx.recv() => {
                        match chunk {
                            Some(StreamChunk::ThinkingDelta(delta)) => {
                                if suppress_thinking {
                                    // Skip thinking tokens entirely (voice mode / /nothink)
                                    continue;
                                }
                                // Render thinking tokens as dimmed text
                                if !in_thinking {
                                    in_thinking = true;
                                    let _ = delta_tx.send("\x1b[90m\u{1f9e0} \x1b[2m".to_string());
                                }
                                let _ = delta_tx.send(delta);
                            }
                            Some(StreamChunk::TextDelta(delta)) => {
                                if in_thinking {
                                    in_thinking = false;
                                    let _ = delta_tx.send("\x1b[0m\n\n".to_string());
                                }
                                ctx.flow.content_was_streamed = true;
                                let _ = delta_tx.send(delta);
                            }
                            Some(StreamChunk::Done(resp)) => {
                                if in_thinking {
                                    let _ = delta_tx.send("\x1b[0m\n\n".to_string());
                                }
                                streamed_response = Some(resp);
                                break;
                            }
                            None => break,
                        }
                    }
                }
            }

            match streamed_response {
                Some(r) => r,
                None => {
                    counters.inference_active.store(false, Ordering::Relaxed);
                    // Stream ended without Done — either cancelled or genuine error.
                    if ctx.cancellation_token
                        .as_ref()
                        .map_or(false, |t| t.is_cancelled())
                    {
                        // Cancelled mid-stream — exit cleanly.
                        return StepResult::Done(IterationOutcome::Finished(String::new()));
                    }
                    error!("LLM stream ended without Done");
                    return StepResult::Done(IterationOutcome::Error(
                        "I encountered a streaming error.".to_string(),
                    ));
                }
            }
        } else {
            // Blocking path: single request/response.
            match ctx.core
                .provider
                .chat(
                    &messages_for_llm,
                    tool_defs_opt,
                    Some(&ctx.core.model),
                    max_tokens,
                    ctx.core.temperature,
                    thinking_budget,
                    None,
                )
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    counters.inference_active.store(false, Ordering::Relaxed);
                    error!(model = %ctx.core.model, error = %e, "llm_call_failed");
                    return StepResult::Done(IterationOutcome::Error(
                        format!("I encountered an error: {}", e),
                    ));
                }
            }
        };

        // Inference complete — allow watchdog health checks again.
        counters.inference_active.store(false, Ordering::Relaxed);

        StepResult::Next(IterationPhase::Processing { response })
    }

    // -----------------------------------------------------------------------
    // Step 4: Processing — validate response, rescue pass, error check, telemetry
    // -----------------------------------------------------------------------

    /// Response validation (hallucinated tool calls), rescue pass for empty
    /// local model responses, provider error check, token telemetry.
    #[instrument(name = "step_process_response", skip(self, ctx, response), fields(
        has_tool_calls = response.has_tool_calls(),
        finish_reason = %response.finish_reason,
        n_tool_calls = response.tool_calls.len(),
    ))]
    async fn step_process_response(
        &self,
        ctx: &mut TurnContext,
        mut response: LLMResponse,
    ) -> StepResult {
        let counters = &self.core_handle.counters;

        // --- Response Validation: detect hallucinated tool calls ---
        let content_str = response.content.as_deref().unwrap_or("");
        let tool_calls_as_maps: Vec<HashMap<String, Value>> = response
            .tool_calls
            .iter()
            .map(|tc| {
                let mut map = HashMap::new();
                map.insert("id".to_string(), Value::String(tc.id.clone()));
                map.insert("name".to_string(), Value::String(tc.name.clone()));
                map.insert(
                    "arguments".to_string(),
                    Value::Object(tc.arguments.iter().map(|(k, v)| (k.clone(), v.clone())).collect()),
                );
                map
            })
            .collect();

        if let Err(validation_err) = validation::validate_response(content_str, &tool_calls_as_maps) {
            warn!(
                model = %ctx.core.model,
                validation = %format!("{:?}", validation_err),
                "response_validation_failed"
            );
            if !response.has_tool_calls() {
                let hint = validation::generate_retry_prompt(&validation_err, 1);
                ctx.messages.push(json!({
                    "role": "assistant",
                    "content": content_str
                }));
                ctx.messages.push(json!({
                    "role": "user",
                    "content": hint
                }));
                debug!("Injected validation retry hint");
                return StepResult::Done(IterationOutcome::Continue);
            }
        }

        // --- Strip thinking tags leaked by small models (Qwen3, etc.) ---
        if ctx.core.is_local {
            if let Some(ref mut content) = response.content {
                let cleaned = crate::agent::compaction::strip_thinking_tags(content);
                if cleaned.len() != content.len() {
                    *content = cleaned;
                }
            }
        }

        // Rescue pass: if local model consumed completion on reasoning and produced no
        // visible answer, force one concise no-thinking completion once.
        let empty_visible = response
            .content
            .as_ref()
            .map(|s| s.trim().is_empty())
            .unwrap_or(true);
        if ctx.core.is_local
            && !response.has_tool_calls()
            && empty_visible
            && response.finish_reason == "length"
            && !ctx.flow.forced_finalize_attempted
        {
            ctx.flow.forced_finalize_attempted = true;
            // Use the same max_tokens cap the original code used (from effective_max_tokens
            // passed through the response, but we only need rescue_tokens here).
            let rescue_tokens = ctx.core.max_tokens.min(384).max(128);
            let mut rescue_messages = ctx.messages.clone();
            rescue_messages.push(json!({
                "role": "user",
                "content": "Return the final answer now. No reasoning. No tool calls. Max 6 lines."
            }));
            counters.inference_active.store(true, Ordering::Relaxed);
            match ctx.core
                .provider
                .chat(
                    &rescue_messages,
                    None,
                    Some(&ctx.core.model),
                    rescue_tokens,
                    0.2,
                    None,
                    None,
                )
                .await
            {
                Ok(r) => {
                    response = r;
                }
                Err(e) => {
                    warn!("Finalize rescue call failed: {}", e);
                }
            }
            counters.inference_active.store(false, Ordering::Relaxed);
        }

        // --- Anti-Drift post-completion: collapse babble ---
        if ctx.core.is_local && ctx.core.anti_drift.enabled && !response.has_tool_calls() {
            if let Some(ref mut content) = response.content {
                anti_drift::post_completion_pipeline(content, &ctx.messages, &ctx.core.anti_drift);
            }
        }

        // Check for LLM provider errors before processing the response.
        if let Some(err_msg) = response.error_detail() {
            error!(model = %ctx.core.model, error = %err_msg, "llm_provider_error");

            // In local mode, check if the server is still alive.
            if ctx.core.is_local {
                if let Some(base) = ctx.core.provider.get_api_base() {
                    if !crate::server::check_health(base).await {
                        error!("Local LLM server is down!");
                        return StepResult::Done(IterationOutcome::Error(
                            "[LLM Error] Local server crashed. Use /restart or /local to recover.".into(),
                        ));
                    }
                }
            }

            return StepResult::Done(IterationOutcome::Error(
                format!("[LLM Error] {}", err_msg),
            ));
        }

        // Token telemetry: log actual vs estimated usage.
        {
            let estimated_prompt = TokenBudget::estimate_tokens(&ctx.messages);
            let actual_prompt = response.usage.get("prompt_tokens").copied().unwrap_or(-1);
            let actual_completion = response
                .usage
                .get("completion_tokens")
                .copied()
                .unwrap_or(-1);
            // Note: max_tokens is not available here (it was consumed by the Calling phase).
            // We log what we can — the actual usage is the important part.
            info!(
                "tokens: estimated_prompt={}, actual_prompt={}, actual_completion={}",
                estimated_prompt, actual_prompt, actual_completion
            );
            // Store actual tokens for /status display.
            if actual_prompt > 0 {
                counters
                    .last_actual_prompt_tokens
                    .store(actual_prompt as u64, Ordering::Relaxed);
            }
            if actual_completion > 0 {
                counters
                    .last_actual_completion_tokens
                    .store(actual_completion as u64, Ordering::Relaxed);
            }
            counters
                .last_estimated_prompt_tokens
                .store(estimated_prompt as u64, Ordering::Relaxed);

            // Emit per-call metrics to ~/.nanobot/metrics.jsonl.
            crate::agent::metrics::emit(&crate::agent::metrics::RequestMetrics {
                timestamp: chrono::Local::now().to_rfc3339(),
                request_id: ctx.request_id.clone(),
                role: "main".into(),
                model: ctx.core.model.clone(),
                provider_base: ctx.core.provider.get_api_base().unwrap_or("unknown").into(),
                elapsed_ms: ctx.flow.llm_call_start.map_or(0, |t| t.elapsed().as_millis() as u64),
                prompt_tokens: actual_prompt.max(0) as u64,
                completion_tokens: actual_completion.max(0) as u64,
                status: "ok".into(),
                error_detail: None,
                anti_drift_score: None,
                anti_drift_signals: None,
                tool_calls_requested: response.tool_calls.len() as u32,
                tool_calls_executed: 0, // updated after execution
                validation_result: None,
            });
        }

        // Branch: tool calls → Executing, no tool calls → finished.
        if response.has_tool_calls() {
            let tool_calls = response.tool_calls.clone();
            StepResult::Next(IterationPhase::Executing { response, tool_calls })
        } else {
            let mut content = response.content.unwrap_or_default();
            if content.trim().is_empty() {
                warn!(
                    finish_reason = %response.finish_reason,
                    "empty_llm_response: SLM returned no content and no tool calls, injecting fallback"
                );
                content = "I couldn't produce a final answer in this turn. Please retry with /thinking off."
                    .to_string();
            }
            StepResult::Done(IterationOutcome::Finished(content))
        }
    }

    // -----------------------------------------------------------------------
    // Step 5: Executing — route and execute tool calls
    // -----------------------------------------------------------------------

    /// Route tool calls through the router, check context pressure,
    /// delegation decision + execute, inline fallback, priority message
    /// check, cancellation check.
    #[instrument(name = "step_execute_tools", skip(self, ctx, response, _tool_calls), fields(
        delegation_enabled = ctx.core.tool_delegation_config.enabled,
        n_tool_calls = response.tool_calls.len(),
    ))]
    async fn step_execute_tools(
        &self,
        ctx: &mut TurnContext,
        response: LLMResponse,
        _tool_calls: Vec<ToolCallRequest>,
    ) -> StepResult {
        let counters = &self.core_handle.counters;

        let routed_tool_calls = match crate::agent::router::route_tool_calls(
            ctx,
            response.content.as_deref(),
            response.tool_calls.clone(),
        )
        .await
        {
            crate::agent::router::RouteResult::Continue => {
                return StepResult::Done(IterationOutcome::Continue);
            }
            crate::agent::router::RouteResult::Break(msg) => {
                return StepResult::Done(IterationOutcome::Finished(msg));
            }
            crate::agent::router::RouteResult::Execute(calls) => calls,
        };

        // Deduplicate identical tool calls within the same batch.
        // Local models sometimes emit the same call multiple times in a single response.
        let routed_tool_calls = {
            let mut seen = std::collections::HashSet::new();
            let before = routed_tool_calls.len();
            let deduped: Vec<_> = routed_tool_calls
                .into_iter()
                .filter(|tc| {
                    let key = crate::agent::tool_runner::normalize_call_key(&tc.name, &tc.arguments);
                    seen.insert(key)
                })
                .collect();
            if deduped.len() < before {
                tracing::warn!(
                    before,
                    after = deduped.len(),
                    "Deduplicated identical tool calls in batch"
                );
            }
            deduped
        };

        // Inject working_dir into exec tool calls when missing.
        // Local models often omit working_dir, causing commands to run in
        // the wrong directory. Default to the process's current directory.
        let routed_tool_calls: Vec<_> = routed_tool_calls
            .into_iter()
            .map(|mut tc| {
                if tc.name == "exec" && !tc.arguments.contains_key("working_dir") {
                    if let Ok(cwd) = std::env::current_dir() {
                        tc.arguments.insert(
                            "working_dir".to_string(),
                            serde_json::Value::String(cwd.to_string_lossy().to_string()),
                        );
                    }
                }
                tc
            })
            .collect();

        // Context pressure check: if high, log a warning. The correct
        // response is compaction, NOT spawning the main model as its
        // own tool runner (which doubles cost for no benefit).
        let context_tokens = TokenBudget::estimate_tokens(&ctx.messages);
        let max_tokens = ctx.core.token_budget.max_context();
        let pressure = if max_tokens > 0 {
            context_tokens as f64 / max_tokens as f64
        } else {
            0.0
        };
        if pressure > 0.7 && !ctx.core.tool_delegation_config.enabled {
            debug!(
                "Context pressure {:.0}% but delegation disabled — consider enabling delegation or compaction",
                pressure * 100.0,
            );
        }

        // Check if we should delegate to the tool runner.
        // Skip delegation if the provider was previously marked dead.
        let mut delegation_alive = counters.delegation_healthy.load(Ordering::Relaxed);
        // Periodically re-probe: every 10 inline calls, try delegation
        // once in case the server recovered (e.g. user restarted it).
        if !delegation_alive && ctx.core.tool_delegation_config.enabled {
            let retries = counters
                .delegation_retry_counter
                .fetch_add(1, Ordering::Relaxed);
            if retries > 0 && retries % 10 == 0 {
                info!(
                    "Re-probing delegation provider (attempt {} since failure)",
                    retries
                );
                delegation_alive = true; // try this one time
            } else {
                debug!("Delegation provider unhealthy — inline execution ({}/10 until re-probe)", retries % 10);
            }
        }
        let should_delegate = ctx.core.tool_delegation_config.enabled && delegation_alive;
        // Resolve provider+model from explicit config.
        let delegation_provider = ctx.core.tool_runner_provider.clone();
        let delegation_model = ctx.core.tool_runner_model.clone();

        if should_delegate {
            if crate::agent::tool_engine::execute_tools_delegated(
                ctx,
                counters,
                &routed_tool_calls,
                &response,
                &delegation_provider,
                &delegation_model,
            )
            .await
            {
                // Delegation handled execution — continue the main loop.
                return StepResult::Done(IterationOutcome::Continue);
            }
        }

        // Inline path (default, unchanged): execute tools directly.
        crate::agent::tool_engine::execute_tools_inline(
            ctx,
            &routed_tool_calls,
            &response,
        )
        .await;

        // Local models via --jinja require strict user/assistant alternation.
        // Tool results are folded into user messages by
        // repair_for_strict_alternation() at the top of the loop.
        // Do NOT add extra user continuation — it would create
        // consecutive user messages.

        // Check for priority user messages injected mid-task.
        if let Some(ref mut rx) = ctx.priority_rx {
            if let Ok(priority_msg) = rx.try_recv() {
                ctx.messages.push(json!({
                    "role": "user",
                    "content": format!("[PRIORITY USER MESSAGE]: {}", priority_msg)
                }));
                // Continue to next LLM call — let the model see and adjust.
            }
        }

        // Check cancellation between tool call iterations.
        if ctx.cancellation_token
            .as_ref()
            .map_or(false, |t| t.is_cancelled())
        {
            return StepResult::Done(IterationOutcome::Finished(String::new()));
        }

        StepResult::Done(IterationOutcome::Continue)
    }

    /// Phase 3: Finalize the response — persist session, build outbound message.
    ///
    /// Consumes the `TurnContext` (by value) since this is the terminal phase.
    /// Stores context stats, writes audit summaries, verifies claims, and
    /// constructs the `OutboundMessage`.
    async fn finalize_response(&self, mut ctx: TurnContext) -> Option<OutboundMessage> {
        let counters = &self.core_handle.counters;

        // Store context stats for status bar.
        let final_tokens = TokenBudget::estimate_tokens(&ctx.messages) as u64;
        counters
            .last_context_used
            .store(final_tokens, Ordering::Relaxed);
        counters
            .last_context_max
            .store(ctx.core.token_budget.max_context() as u64, Ordering::Relaxed);
        counters
            .last_message_count
            .store(ctx.messages.len() as u64, Ordering::Relaxed);
        // Store working memory token count.
        let wm_tokens = if ctx.core.memory_enabled {
            let wm_text = ctx.core.working_memory.get_context(&ctx.session_key, usize::MAX);
            TokenBudget::estimate_str_tokens(&wm_text) as u64
        } else {
            0
        };
        counters
            .last_working_memory_tokens
            .store(wm_tokens, Ordering::Relaxed);
        // Store tools called this turn.
        {
            let tools_list: Vec<String> = ctx.used_tools.iter().cloned().collect();
            if let Ok(mut guard) = counters.last_tools_called.lock() {
                *guard = tools_list;
            }
        }

        // Write per-turn audit summary.
        if ctx.core.provenance_config.enabled && ctx.core.provenance_config.audit_log {
            let summary = crate::agent::audit::TurnSummary {
                turn: ctx.turn_count,
                timestamp: Utc::now().to_rfc3339(),
                context_tokens: final_tokens as usize,
                message_count: ctx.messages.len(),
                tools_called: ctx.turn_tool_entries,
                working_memory_tokens: wm_tokens as usize,
            };
            crate::agent::audit::write_turn_summary(&ctx.core.workspace, &ctx.session_key, &summary);
        }

        if ctx.final_content.is_empty() && ctx.messages.len() > 2 {
            ctx.final_content = "I ran out of tool iterations before producing a final answer. The actions above may be incomplete.".to_string();
        }

        // Phase 3+4: Claim verification and context hygiene.
        if !ctx.final_content.is_empty()
            && ctx.core.provenance_config.enabled
            && ctx.core.provenance_config.verify_claims
        {
            if let Some(ref audit) = ctx.audit {
                let entries = audit.get_entries();
                let (claims, has_fabrication) =
                    crate::agent::provenance::verify_turn_claims(&ctx.final_content, &entries);

                if has_fabrication && ctx.core.provenance_config.strict_mode {
                    let (redacted, redaction_count) =
                        crate::agent::provenance::redact_fabrications(&ctx.final_content, &claims);
                    ctx.final_content = redacted;
                    if redaction_count > 0 {
                        let warning_role = provenance_warning_role(ctx.core.is_local);
                        let warning_content = format!(
                            "NOTICE: {} claim(s) in the previous response could not be \
                             verified against tool outputs and were removed.",
                            redaction_count
                        );
                        ctx.messages.push(json!({
                            "role": warning_role,
                            "content": warning_content
                        }));
                    }
                }
            }
        }

        // Phantom tool call detection: check if LLM claims tool results without calling tools.
        if !ctx.final_content.is_empty() && ctx.core.provenance_config.enabled {
            let tools_list: Vec<String> = ctx.used_tools.iter().cloned().collect();
            if let Some(detection) =
                crate::agent::provenance::detect_phantom_claims(&ctx.final_content, &tools_list)
            {
                warn!(
                    model = %ctx.core.model,
                    patterns = detection.matched_patterns.len(),
                    "phantom_tool_claims_detected: {:?}",
                    detection.matched_patterns
                );

                // Hard block: annotate the response so the user sees the warning.
                if ctx.core.provenance_config.strict_mode {
                    ctx.final_content = crate::agent::provenance::annotate_phantom_response(
                        &ctx.final_content,
                        &detection,
                    );
                }

                // Inject system reminder for the next turn.
                let warning_role = provenance_warning_role(ctx.core.is_local);
                ctx.messages.push(json!({
                    "role": warning_role,
                    "content": detection.system_warning
                }));
            }
        }

        // Ensure the final text response is in the messages array for persistence.
        // Without this, text-only responses (no tool calls) would be lost.
        if !ctx.final_content.is_empty() {
            ctx.messages.push(json!({"role": "assistant", "content": ctx.final_content.clone()}));
        }

        // Update session history — persist full message array including tool calls.
        // Skip system prompt (index 0) and pre-existing history.
        let new_messages: Vec<Value> = if ctx.new_start < ctx.messages.len() {
            ctx.messages[ctx.new_start..].to_vec()
        } else {
            // Fallback: save at least user + assistant text.
            let mut fallback = vec![json!({"role": "user", "content": ctx.user_content.clone()})];
            if !ctx.final_content.is_empty() {
                fallback.push(json!({"role": "assistant", "content": ctx.final_content.clone()}));
            }
            fallback
        };
        if !new_messages.is_empty() {
            ctx.core.sessions
                .add_messages_raw(&ctx.session_key, &new_messages)
                .await;
        }

        // Auto-complete stale working memory sessions (runs on every message, cheap).
        if ctx.core.memory_enabled {
            for session in ctx.core.working_memory.list_active() {
                if session.session_key != ctx.session_key {
                    let age = Utc::now() - session.updated;
                    if age.num_seconds() > ctx.core.session_complete_after_secs as i64 {
                        ctx.core.working_memory.complete(&session.session_key);
                        debug!("Auto-completed stale session: {}", session.session_key);
                    }
                }
            }

            // Clear current session's working memory if stale (no compaction in N turns).
            {
                let current = ctx.core.working_memory.get_or_create(&ctx.session_key);
                if !current.content.is_empty()
                    && current.last_updated_turn > 0
                    && ctx.turn_count.saturating_sub(current.last_updated_turn) > ctx.core.stale_memory_turn_threshold
                {
                    ctx.core.working_memory.clear(&ctx.session_key);
                    debug!(
                        "Cleared stale working memory for {} (last update turn {}, current turn {})",
                        ctx.session_key, current.last_updated_turn, ctx.turn_count
                    );
                }
            }
        }

        // Record execution stats for budget calibration (append-only, errors silently logged).
        if let Some(ref cal_mutex) = self.calibrator {
            let task_type = if ctx.used_tools.contains("exec_command") {
                "shell"
            } else if ctx.used_tools.contains("web_search") {
                "web_search"
            } else if ctx.used_tools.contains("spawn_agent") {
                "delegate"
            } else if ctx.used_tools.is_empty() {
                "chat"
            } else {
                "tool_use"
            };
            let record = crate::agent::budget_calibrator::ExecutionRecord {
                task_type: task_type.to_string(),
                model: ctx.core.model.clone(),
                iterations_used: ctx.iterations_used,
                max_iterations: ctx.core.max_iterations,
                success: !ctx.final_content.is_empty(),
                cost_usd: 0.0, // TODO: wire actual cost tracking
                duration_ms: ctx.turn_start.elapsed().as_millis() as u64,
                depth: 0,
                tool_calls: ctx.used_tools.len() as u32,
                created_at: chrono::Utc::now().to_rfc3339(),
            };
            if let Ok(cal) = cal_mutex.lock() {
                if let Err(e) = cal.record(&record) {
                    tracing::debug!("BudgetCalibrator record failed: {}", e);
                }
            }
        }

        ctx.final_content =
            crate::agent::sanitize::sanitize_reasoning_output(&ctx.final_content);

        if ctx.final_content.is_empty() {
            None
        } else {
            let mut outbound = OutboundMessage::new(&ctx.channel, &ctx.chat_id, &ctx.final_content);
            // Propagate voice_message metadata so channels know to reply with voice.
            if ctx.is_voice_message {
                outbound
                    .metadata
                    .insert("voice_message".to_string(), json!(true));
            }
            // Propagate detected_language for TTS voice selection.
            if let Some(ref lang) = ctx.detected_language {
                outbound
                    .metadata
                    .insert("detected_language".to_string(), json!(lang));
            }
            Some(outbound)
        }
    }
}

// ---------------------------------------------------------------------------
// Pure helpers (no IO — fully unit-testable)
// ---------------------------------------------------------------------------

/// Convert raw wire-format messages to canonical `Turn` sequence, then render
/// via the given protocol to produce a clean wire format for the LLM call.
///
/// - Position 0 is expected to be `role:system`; it is extracted and passed as
///   the `system` argument to `protocol.render()`.
/// - Any `_turn` / `_synthetic` metadata tags on raw messages are not forwarded
///   to the wire output (they are internal-only fields used for trimming).
fn render_via_protocol(protocol: &dyn ConversationProtocol, messages: &[Value]) -> Vec<Value> {
    // Extract system prompt from the leading system message (if present).
    let system = messages
        .first()
        .filter(|m| m.get("role").and_then(|r| r.as_str()) == Some("system"))
        .and_then(|m| m.get("content").and_then(|c| c.as_str()))
        .unwrap_or("")
        .to_string();

    let non_system_start = if messages
        .first()
        .map(|m| m.get("role").and_then(|r| r.as_str()) == Some("system"))
        .unwrap_or(false)
    {
        1
    } else {
        0
    };

    let turns: Vec<_> = messages[non_system_start..]
        .iter()
        .filter_map(|m| turn_from_legacy(m))
        .collect();

    protocol.render(&system, &turns)
}

/// Decide whether trio routing is healthy enough to strip tools from the main model.
/// Pure function: takes health status as booleans, returns true if tools should be stripped.
#[instrument(name = "should_strip_tools_for_trio", fields(
    is_local,
    strict_no_tools_main,
    router_probe_healthy,
    circuit_breaker_available,
))]
fn should_strip_tools_for_trio(
    is_local: bool,
    strict_no_tools_main: bool,
    router_probe_healthy: bool,
    circuit_breaker_available: bool,
) -> bool {
    let result = is_local && strict_no_tools_main && router_probe_healthy && circuit_breaker_available;
    tracing::debug!(strip_tools = result, "trio_strip_decision");
    result
}

// ---------------------------------------------------------------------------
// AgentLoop (owns the receiver + orchestrates concurrency)
// ---------------------------------------------------------------------------

/// The core agent loop.
///
/// Consumes [`InboundMessage`]s from the bus, runs the LLM + tool loop, and
/// publishes [`OutboundMessage`]s when the agent produces a response.
///
/// In gateway mode, messages for different sessions run concurrently (up to
/// `max_concurrent_chats`), while messages within the same session are
/// serialized to preserve conversation ordering.
pub struct AgentLoop {
    shared: Arc<AgentLoopShared>,
    bus_inbound_rx: UnboundedReceiver<InboundMessage>,
    running: Arc<AtomicBool>,
    max_concurrent_chats: usize,
    reflection_spawned: AtomicBool,
}

impl AgentLoop {
    /// Create a new `AgentLoop`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        core_handle: SharedCoreHandle,
        bus_inbound_rx: UnboundedReceiver<InboundMessage>,
        bus_outbound_tx: UnboundedSender<OutboundMessage>,
        bus_inbound_tx: UnboundedSender<InboundMessage>,
        cron_service: Option<Arc<CronService>>,
        max_concurrent_chats: usize,
        email_config: Option<EmailConfig>,
        repl_display_tx: Option<UnboundedSender<String>>,
        providers_config: Option<crate::config::schema::ProvidersConfig>,
        proprioception_config: ProprioceptionConfig,
        lcm_config: LcmSchemaConfig,
        health_registry: Option<Arc<crate::heartbeat::health::HealthRegistry>>,
    ) -> Self {
        // Read core to initialize the subagent manager.
        let core = core_handle.swappable();
        let mut subagent_mgr = SubagentManager::new(
            core.provider.clone(),
            core.workspace.clone(),
            bus_inbound_tx.clone(),
            core.model.clone(),
            core.brave_api_key.clone(),
            core.exec_timeout,
            core.restrict_to_workspace,
            core.is_local,
            core.max_tool_result_chars,
        );
        if let Some(pc) = providers_config {
            subagent_mgr = subagent_mgr.with_providers_config(pc);
        }
        // Wire up the cheap default model for subagents from config.
        subagent_mgr = subagent_mgr.with_default_subagent_model(
            core.tool_delegation_config.default_subagent_model.clone(),
        );
        // Wire up subagent tuning from config.
        subagent_mgr = subagent_mgr.with_subagent_tuning(
            core.tool_delegation_config.subagent.clone(),
        );
        if let Some(ref dtx) = repl_display_tx {
            subagent_mgr = subagent_mgr.with_display_tx(dtx.clone());
        }
        if core.is_local {
            subagent_mgr = subagent_mgr.with_local_context_limit(core.token_budget.max_context());
        }

        // Create aha channel before subagent manager so we can pass the sender.
        let (aha_tx, aha_rx) = tokio::sync::mpsc::unbounded_channel();
        if proprioception_config.aha_channel {
            subagent_mgr = subagent_mgr.with_aha_tx(aha_tx.clone());
        }

        let subagents = Arc::new(subagent_mgr);

        // Load persisted bulletin from disk (warm start).
        let bulletin_cache = {
            let core = core_handle.swappable();
            let cache = crate::agent::bulletin::BulletinCache::new();
            if let Some(persisted) =
                crate::agent::bulletin::load_persisted_bulletin(&core.workspace)
            {
                cache.update(persisted);
            }
            cache.handle()
        };

        let system_state = Arc::new(arc_swap::ArcSwap::from_pointee(SystemState::default()));

        // Build dedicated LCM compactor when compaction_endpoint is configured.
        let lcm_compactor = lcm_config.compaction_endpoint.as_ref().map(|ep| {
            let provider: Arc<dyn crate::providers::base::LLMProvider> =
                crate::providers::factory::create_openai_compat(
                    crate::providers::factory::ProviderSpec {
                        api_key: "lcm-compactor".to_string(),
                        api_base: Some(ep.url.clone()),
                        model: Some(ep.model.clone()),
                        jit_gate: None,
                    },
                );
            Arc::new(ContextCompactor::new(
                provider,
                ep.model.clone(),
                lcm_config.compaction_context_size,
            ))
        });

        let shared = Arc::new(AgentLoopShared {
            core_handle,
            subagents,
            bus_outbound_tx,
            bus_inbound_tx,
            cron_service,
            email_config,
            repl_display_tx,
            bulletin_cache,
            system_state,
            proprioception_config,
            aha_rx: Arc::new(Mutex::new(aha_rx)),
            aha_tx,
            session_policies: Arc::new(Mutex::new(HashMap::new())),
            lcm_engines: Arc::new(Mutex::new(HashMap::new())),
            lcm_config,
            lcm_compactor,
            health_registry,
            calibrator: match crate::agent::budget_calibrator::BudgetCalibrator::open_default() {
                Ok(c) => Some(std::sync::Mutex::new(c)),
                Err(e) => {
                    tracing::warn!("BudgetCalibrator init failed, recording disabled: {}", e);
                    None
                }
            },
        });

        Self {
            shared,
            bus_inbound_rx,
            running: Arc::new(AtomicBool::new(false)),
            max_concurrent_chats,
            reflection_spawned: AtomicBool::new(false),
        }
    }

    /// Spawn a periodic bulletin refresh task (compaction model, when idle).
    fn spawn_bulletin_refresh(shared: &Arc<AgentLoopShared>, running: &Arc<AtomicBool>) {
        let core = shared.core_handle.swappable();
        if !core.memory_enabled {
            return;
        }
        let provider = core.memory_provider.clone();
        let model = core.memory_model.clone();
        let workspace = core.workspace.clone();
        let cache = shared.bulletin_cache.clone();
        let running = running.clone();

        tokio::spawn(async move {
            // Initial delay: let the system settle before first bulletin.
            tokio::time::sleep(Duration::from_secs(5 * 60)).await;

            while running.load(Ordering::Relaxed) {
                debug!("Bulletin: refreshing...");
                if let Err(e) = crate::agent::bulletin::refresh_bulletin(
                    provider.as_ref(),
                    &model,
                    &workspace,
                    &cache,
                )
                .await
                {
                    warn!("Bulletin refresh failed: {}", e);
                }
                // Sleep until next refresh.
                tokio::time::sleep(Duration::from_secs(
                    crate::agent::bulletin::DEFAULT_BULLETIN_INTERVAL_S,
                ))
                .await;
            }
        });
        info!(
            "Bulletin refresh task spawned (every {}min)",
            crate::agent::bulletin::DEFAULT_BULLETIN_INTERVAL_S / 60
        );
    }

    /// Spawn a background reflection task if observations exceed threshold.
    fn spawn_background_reflection(shared: &Arc<AgentLoopShared>) {
        let core = shared.core_handle.swappable();
        if !core.memory_enabled {
            return;
        }
        let reflector = Reflector::new(
            core.memory_provider.clone(),
            core.memory_model.clone(),
            &core.workspace,
            core.reflection_threshold,
        );
        if reflector.should_reflect() {
            tokio::spawn(async move {
                info!("Background: reflecting on accumulated observations...");
                if let Err(e) = reflector.reflect().await {
                    warn!("Background reflection failed: {}", e);
                } else {
                    info!("Background reflection complete — MEMORY.md updated");
                }
            });
        }
    }

    /// Run the main agent loop until stopped.
    ///
    /// Messages for different sessions are processed concurrently (up to
    /// `max_concurrent_chats`). Messages within the same session are serialized.
    pub async fn run(&mut self) {
        self.running.store(true, Ordering::SeqCst);
        info!(
            "Agent loop started (max_concurrent_chats={})",
            self.max_concurrent_chats
        );

        // Spawn background reflection if observations have accumulated.
        Self::spawn_background_reflection(&self.shared);

        // Spawn periodic bulletin refresh (compaction model synthesizes briefing).
        Self::spawn_bulletin_refresh(&self.shared, &self.running);

        let semaphore = Arc::new(Semaphore::new(self.max_concurrent_chats));
        // Per-session locks to serialize messages within the same conversation.
        let session_locks: Arc<Mutex<HashMap<String, Arc<Mutex<()>>>>> =
            Arc::new(Mutex::new(HashMap::new()));

        while self.running.load(Ordering::SeqCst) {
            let msg = match tokio::time::timeout(Duration::from_secs(1), self.bus_inbound_rx.recv())
                .await
            {
                Ok(Some(msg)) => msg,
                Ok(None) => {
                    info!("Inbound channel closed, stopping agent loop");
                    break;
                }
                Err(_) => continue, // timeout - loop and check running flag
            };

            // Coalesce rapid messages from the same session (Telegram, WhatsApp).
            // Waits up to 400ms for follow-up messages before processing.
            let msg = if crate::bus::events::should_coalesce(&msg.channel)
                && !msg.content.trim_start().starts_with('/')
            {
                let session = msg.session_key();
                let mut batch = vec![msg];
                let deadline = tokio::time::Instant::now() + Duration::from_millis(400);
                loop {
                    match tokio::time::timeout_at(deadline, self.bus_inbound_rx.recv()).await {
                        Ok(Some(next)) if next.session_key() == session => {
                            batch.push(next);
                        }
                        Ok(Some(other)) => {
                            // Different session — coalesce what we have, push other back.
                            // Can't push back into mpsc, so process inline as separate spawn.
                            let other_key = other.session_key();
                            let other_lock = {
                                let mut locks = session_locks.lock().await;
                                locks
                                    .entry(other_key)
                                    .or_insert_with(|| Arc::new(Mutex::new(())))
                                    .clone()
                            };
                            let other_shared = self.shared.clone();
                            let other_outbound_tx = self.shared.bus_outbound_tx.clone();
                            let _other_display_tx = self.shared.repl_display_tx.clone();
                            let other_sem = semaphore.clone();
                            tokio::spawn(async move {
                                if let Ok(permit) = other_sem.acquire_owned().await {
                                    let _guard = other_lock.lock().await;
                                    if let Some(resp) = other_shared
                                        .process_message(&other, None, None, None, None)
                                        .await
                                    {
                                        let _ = other_outbound_tx.send(resp);
                                    }
                                    drop(permit);
                                }
                            });
                            break;
                        }
                        _ => break, // timeout or channel closed
                    }
                }
                if batch.len() > 1 {
                    debug!("Coalesced {} messages for session", batch.len());
                }
                crate::bus::events::coalesce_messages(batch)
            } else {
                msg
            };

            // System messages (subagent announces) are handled inline (fast).
            let is_system = msg
                .metadata
                .get("is_system")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

            if is_system {
                debug!(
                    "Processing system message: {}",
                    &msg.content[..msg.content.len().min(80)]
                );
                let outbound = OutboundMessage::new(&msg.channel, &msg.chat_id, &msg.content);
                if let Err(e) = self.shared.bus_outbound_tx.send(outbound) {
                    error!("Failed to publish outbound message: {}", e);
                }
                continue;
            }

            // Gateway slash command interception — handle before LLM processing.
            if msg.content.trim().starts_with('/') {
                if let Some(response_text) = crate::agent::gateway_commands::dispatch(&self.shared, &msg).await {
                    let outbound = crate::bus::events::OutboundMessage::new(&msg.channel, &msg.chat_id, &response_text);
                    if let Err(e) = self.shared.bus_outbound_tx.send(outbound) {
                        tracing::error!("Failed to send command response: {}", e);
                    }
                    continue;
                }
            }

            // Acquire a concurrency permit.
            let permit = match semaphore.clone().acquire_owned().await {
                Ok(p) => p,
                Err(_) => {
                    error!("Semaphore closed unexpectedly");
                    break;
                }
            };

            // Get or create the per-session lock.
            let session_key = msg.session_key();
            let session_lock = {
                let mut locks = session_locks.lock().await;
                locks
                    .entry(session_key)
                    .or_insert_with(|| Arc::new(Mutex::new(())))
                    .clone()
            };

            let shared = self.shared.clone();
            let outbound_tx = self.shared.bus_outbound_tx.clone();
            let display_tx = self.shared.repl_display_tx.clone();

            tokio::spawn(async move {
                // Serialize within the same session.
                let _session_guard = session_lock.lock().await;

                // Notify REPL about inbound channel message.
                if let Some(ref dtx) = display_tx {
                    let preview = if msg.content.len() > 120 {
                        let end = crate::utils::helpers::floor_char_boundary(&msg.content, 120);
                        format!("{}...", &msg.content[..end])
                    } else {
                        msg.content.clone()
                    };
                    let _ = dtx.send(format!(
                        "\x1b[2m[{}]\x1b[0m \x1b[36m{}\x1b[0m: {}",
                        msg.channel, msg.sender_id, preview
                    ));
                }

                let response = shared.process_message(&msg, None, None, None, None).await;

                if let Some(outbound) = response {
                    // Notify REPL about outbound response.
                    if let Some(ref dtx) = display_tx {
                        let preview = if outbound.content.len() > 120 {
                            let end =
                                crate::utils::helpers::floor_char_boundary(&outbound.content, 120);
                            format!("{}...", &outbound.content[..end])
                        } else {
                            outbound.content.clone()
                        };
                        let _ = dtx.send(format!(
                            "\x1b[2m[{}]\x1b[0m \x1b[33mbot\x1b[0m: {}",
                            outbound.channel, preview
                        ));
                    }
                    if let Err(e) = outbound_tx.send(outbound) {
                        error!("Failed to publish outbound message: {}", e);
                    }
                }

                drop(permit); // release concurrency slot
            });
        }

        info!("Agent loop stopped");
    }

    /// Return a handle to the subagent manager.
    pub fn subagent_manager(&self) -> Arc<SubagentManager> {
        self.shared.subagents.clone()
    }

    /// Signal the agent loop to stop.
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
    }

    /// Process a message directly (for CLI / cron usage) without going through
    /// the bus.
    pub async fn process_direct(
        &self,
        content: &str,
        session_key: &str,
        channel: &str,
        chat_id: &str,
    ) -> String {
        self.process_direct_with_lang(content, session_key, channel, chat_id, None)
            .await
    }

    /// Like `process_direct` but allows passing a detected language code
    /// (e.g. "it", "es") so the LLM responds in that language.
    pub async fn process_direct_with_lang(
        &self,
        content: &str,
        session_key: &str,
        channel: &str,
        chat_id: &str,
        detected_language: Option<&str>,
    ) -> String {
        // Spawn background reflection once per session (on first message).
        if !self.reflection_spawned.swap(true, Ordering::SeqCst) {
            Self::spawn_background_reflection(&self.shared);
        }

        let mut msg = InboundMessage::new(channel, "user", chat_id, content);
        msg.metadata
            .insert("session_key".to_string(), json!(session_key));
        if let Some(lang) = detected_language {
            msg.metadata
                .insert("detected_language".to_string(), json!(lang));
        }

        match self
            .shared
            .process_message(&msg, None, None, None, None)
            .await
        {
            Some(response) => response.content,
            None => String::new(),
        }
    }

    /// Like `process_direct_with_lang` but streams text deltas to `text_delta_tx`
    /// as they arrive from the LLM. Returns the full response text.
    pub async fn process_direct_streaming(
        &self,
        content: &str,
        session_key: &str,
        channel: &str,
        chat_id: &str,
        detected_language: Option<&str>,
        text_delta_tx: tokio::sync::mpsc::UnboundedSender<String>,
        tool_event_tx: Option<tokio::sync::mpsc::UnboundedSender<ToolEvent>>,
        cancellation_token: Option<tokio_util::sync::CancellationToken>,
        priority_rx: Option<tokio::sync::mpsc::UnboundedReceiver<String>>,
    ) -> String {
        if !self.reflection_spawned.swap(true, Ordering::SeqCst) {
            Self::spawn_background_reflection(&self.shared);
        }

        let mut msg = InboundMessage::new(channel, "user", chat_id, content);
        msg.metadata
            .insert("session_key".to_string(), json!(session_key));
        if let Some(lang) = detected_language {
            msg.metadata
                .insert("detected_language".to_string(), json!(lang));
        }

        match self
            .shared
            .process_message(
                &msg,
                Some(text_delta_tx),
                tool_event_tx,
                cancellation_token,
                priority_rx,
            )
            .await
        {
            Some(response) => response.content,
            None => String::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tool proxy wrappers
// ---------------------------------------------------------------------------
//
// Because `Arc<MessageTool>` etc. don't implement `Tool` directly (the trait
// requires owned `Box<dyn Tool>`), we create thin proxy wrappers that
// implement `Tool` by delegating to the inner `Arc`.


// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use backon::BackoffBuilder;
    use crate::agent::router::{extract_json_object, parse_lenient_router_decision, request_strict_router_decision};
    use crate::config::schema::{MemoryConfig, ProvenanceConfig, ProviderConfig, ToolDelegationConfig, TrioConfig};
    use crate::providers::base::LLMProvider;
    use crate::providers::openai_compat::OpenAICompatProvider;
    use async_trait::async_trait;

    /// Minimal mock LLM provider for wiring tests.
    struct MockLLM {
        name: String,
    }

    impl MockLLM {
        fn named(name: &str) -> Arc<dyn LLMProvider> {
            Arc::new(Self {
                name: name.to_string(),
            })
        }
    }

    #[async_trait]
    impl LLMProvider for MockLLM {
        async fn chat(
            &self,
            _messages: &[Value],
            _tools: Option<&[Value]>,
            _model: Option<&str>,
            _max_tokens: u32,
            _temperature: f64,
            _thinking_budget: Option<u32>,
            _top_p: Option<f64>,
        ) -> anyhow::Result<crate::providers::base::LLMResponse> {
            Ok(crate::providers::base::LLMResponse {
                content: Some("mock".to_string()),
                tool_calls: vec![],
                finish_reason: "stop".to_string(),
                usage: std::collections::HashMap::new(),
            })
        }

        fn get_default_model(&self) -> &str {
            &self.name
        }
    }

    struct StaticResponseLLM {
        name: String,
        body: String,
    }

    impl StaticResponseLLM {
        fn new(name: &str, body: &str) -> Self {
            Self {
                name: name.to_string(),
                body: body.to_string(),
            }
        }
    }

    #[async_trait]
    impl LLMProvider for StaticResponseLLM {
        async fn chat(
            &self,
            _messages: &[Value],
            _tools: Option<&[Value]>,
            _model: Option<&str>,
            _max_tokens: u32,
            _temperature: f64,
            _thinking_budget: Option<u32>,
            _top_p: Option<f64>,
        ) -> anyhow::Result<crate::providers::base::LLMResponse> {
            Ok(crate::providers::base::LLMResponse {
                content: Some(self.body.clone()),
                tool_calls: vec![],
                finish_reason: "stop".to_string(),
                usage: std::collections::HashMap::new(),
            })
        }

        fn get_default_model(&self) -> &str {
            &self.name
        }
    }

    /// Helper to build a SwappableCore with minimal config for wiring tests.
    fn build_test_core(
        delegation_enabled: bool,
        delegation_provider: Option<Arc<dyn LLMProvider>>,
        config_provider: Option<ProviderConfig>,
    ) -> SwappableCore {
        let workspace = tempfile::tempdir().unwrap().into_path();
        let main = MockLLM::named("main-provider");
        let td = ToolDelegationConfig {
            enabled: delegation_enabled,
            model: "delegation-model".to_string(),
            provider: config_provider,
            auto_local: true,
            ..Default::default()
        };
        build_swappable_core(SwappableCoreConfig {
            provider: main,
            workspace,
            model: "main-model".to_string(),
            max_iterations: 10,
            max_tokens: 4096,
            temperature: 0.7,
            max_context_tokens: 16384,
            brave_api_key: None,
            exec_timeout: 30,
            restrict_to_workspace: false,
            memory_config: MemoryConfig::default(),
            is_local: false,
            compaction_provider: None,
            tool_delegation: td,
            provenance: ProvenanceConfig::default(),
            max_tool_result_chars: 2000,
            delegation_provider,
            specialist_provider: None,
            trio_config: TrioConfig::default(),
            model_capabilities_overrides: std::collections::HashMap::new(),
        })
    }

    #[test]
    fn test_provenance_warning_role_local_safe() {
        assert_eq!(provenance_warning_role(true), "user");
        assert_eq!(provenance_warning_role(false), "system");
    }

    #[test]
    fn test_extract_json_object_from_markdown_fence() {
        let raw = "```json\n{\"action\":\"tool\",\"target\":\"exec\",\"args\":{},\"confidence\":0.9}\n```";
        let obj = extract_json_object(raw).expect("json object");
        assert!(obj.starts_with('{'));
        assert!(obj.ends_with('}'));
        assert!(obj.contains("\"action\":\"tool\""));
    }

    #[test]
    fn test_extract_json_object_none_when_missing() {
        assert!(extract_json_object("no json here").is_none());
    }

    #[tokio::test]
    async fn test_request_strict_router_decision_action_matrix() {
        let cases = vec![
            (
                r#"{"action":"tool","target":"read_file","args":{"path":"README.md"},"confidence":0.9}"#,
                "tool",
            ),
            (
                r#"{"action":"subagent","target":"builder","args":{"task":"x"},"confidence":0.8}"#,
                "subagent",
            ),
            (
                r#"{"action":"specialist","target":"summarizer","args":{"style":"tight"},"confidence":0.7}"#,
                "specialist",
            ),
            (
                r#"{"action":"ask_user","target":"clarify","args":{"question":"Need path?"},"confidence":0.6}"#,
                "ask_user",
            ),
        ];

        for (raw, expected_action) in cases {
            let llm = StaticResponseLLM::new("router", raw);
            let decision = request_strict_router_decision(
                &llm,
                "router",
                "route this action with strict schema",
                false,
                0.6,
                1.0,
                "",
            )
            .await
            .expect("valid strict router decision");
            assert_eq!(decision.action, expected_action);
        }
    }

    /// Real-provider trio probe.
    ///
    /// Runs against live OpenAI-compatible endpoints (e.g. LM Studio):
    /// - main: `NANOBOT_REAL_MAIN_BASE` (default: http://127.0.0.1:8080/v1)
    /// - router: `NANOBOT_REAL_ROUTER_BASE` (default: http://127.0.0.1:8094/v1)
    /// - specialist: `NANOBOT_REAL_SPECIALIST_BASE` (default: http://127.0.0.1:8095/v1)
    ///
    /// Optional model overrides:
    /// - `NANOBOT_REAL_MAIN_MODEL`
    /// - `NANOBOT_REAL_ROUTER_MODEL`
    /// - `NANOBOT_REAL_SPECIALIST_MODEL`
    #[tokio::test]
    #[ignore = "requires running local providers on main/router/specialist ports"]
    async fn test_real_providers_trio_probe() {
        let main_base = std::env::var("NANOBOT_REAL_MAIN_BASE")
            .unwrap_or_else(|_| "http://127.0.0.1:8080/v1".to_string());
        let router_base = std::env::var("NANOBOT_REAL_ROUTER_BASE")
            .unwrap_or_else(|_| "http://127.0.0.1:8094/v1".to_string());
        let specialist_base = std::env::var("NANOBOT_REAL_SPECIALIST_BASE")
            .unwrap_or_else(|_| "http://127.0.0.1:8095/v1".to_string());
        let main_model = std::env::var("NANOBOT_REAL_MAIN_MODEL")
            .unwrap_or_else(|_| "local-model".to_string());
        let router_model = std::env::var("NANOBOT_REAL_ROUTER_MODEL")
            .unwrap_or_else(|_| "local-delegation".to_string());
        let specialist_model = std::env::var("NANOBOT_REAL_SPECIALIST_MODEL")
            .unwrap_or_else(|_| "local-specialist".to_string());

        let main = OpenAICompatProvider::new("local", Some(&main_base), Some(&main_model));
        let router = OpenAICompatProvider::new("local", Some(&router_base), Some(&router_model));
        let specialist =
            OpenAICompatProvider::new("local", Some(&specialist_base), Some(&specialist_model));

        let mut failures: Vec<String> = Vec::new();

        // Router: force each action in a constrained prompt and verify strict parsing.
        let router_cases = vec![
            ("tool", "Return action=tool target=read_file args={\"path\":\"README.md\"}."),
            (
                "subagent",
                "Return action=subagent target=builder args={\"task\":\"diagnose issue\"}.",
            ),
            (
                "specialist",
                "Return action=specialist target=summarizer args={\"objective\":\"compress\"}.",
            ),
            (
                "ask_user",
                "Return action=ask_user target=clarify args={\"question\":\"Which file?\"}.",
            ),
        ];
        for (expected_action, directive) in router_cases {
            let pack = format!("{}\nFollow schema strictly.", directive);
            match request_strict_router_decision(&router, &router_model, &pack, false, 0.6, 1.0, "").await {
                Ok(d) => {
                    if d.action != expected_action {
                        failures.push(format!(
                            "router action mismatch: expected={}, got={} target={}",
                            expected_action, d.action, d.target
                        ));
                    }
                }
                Err(e) => failures.push(format!("router {} failed: {}", expected_action, e)),
            }
        }

        // Specialist must produce non-empty response (with warmup retries).
        let specialist_messages = vec![
            json!({"role":"system","content":"ROLE=SPECIALIST\nReturn concise output."}),
            json!({"role":"user","content":"Summarize: tool call failed because server was down and port conflicted."}),
        ];
        let mut specialist_ok = false;
        let mut warmup_backoff = backon::ConstantBuilder::default()
            .with_delay(Duration::from_secs(2))
            .with_max_times(10)
            .build();
        loop {
            match specialist
                .chat(
                    &specialist_messages,
                    None,
                    Some(&specialist_model),
                    256,
                    0.2,
                    None,
                    None,
                )
                .await
            {
                Ok(resp) => {
                    let text = resp.content.unwrap_or_default();
                    if !text.trim().is_empty() {
                        specialist_ok = true;
                        break;
                    }
                }
                Err(e) => {
                    let msg = e.to_string();
                    let lower = msg.to_lowercase();
                    if !lower.contains("loading model") && !lower.contains("503") {
                        failures.push(format!("specialist call failed: {}", msg));
                        break;
                    }
                }
            }
            match warmup_backoff.next() {
                Some(delay) => tokio::time::sleep(delay).await,
                None => break,
            }
        }
        if !specialist_ok {
            failures.push("specialist did not become ready / returned empty output".to_string());
        }

        // Main provider smoke: should answer plain text with no tools when none offered.
        let main_messages = vec![json!({"role":"user","content":"Reply with exactly: main-ok"})];
        match main
            .chat(&main_messages, None, Some(&main_model), 64, 0.0, None, None)
            .await
        {
            Ok(resp) => {
                if resp.has_tool_calls() {
                    failures.push("main returned tool calls unexpectedly".to_string());
                }
                let text = resp.content.unwrap_or_default();
                if !text.to_lowercase().contains("main-ok") {
                    failures.push(format!("main output mismatch: {}", text));
                }
            }
            Err(e) => failures.push(format!("main call failed: {}", e)),
        }

        if !failures.is_empty() {
            panic!(
                "real trio probe failed (main={}, router={}, specialist={}):\n{}",
                main_base,
                router_base,
                specialist_base,
                failures.join("\n")
            );
        }
    }

    // -- Delegation provider wiring tests --

    #[test]
    fn test_delegation_disabled_no_runner_provider() {
        let core = build_test_core(false, None, None);
        assert!(
            core.tool_runner_provider.is_none(),
            "When delegation is disabled, tool_runner_provider should be None"
        );
        assert!(core.tool_runner_model.is_none());
    }

    #[test]
    fn test_delegation_enabled_with_auto_provider() {
        // When an auto-spawned delegation provider is passed, it should be used
        let dp = MockLLM::named("auto-delegation");
        let core = build_test_core(true, Some(dp), None);

        assert!(core.tool_runner_provider.is_some());
        let provider = core.tool_runner_provider.as_ref().unwrap();
        assert_eq!(
            provider.get_default_model(),
            "auto-delegation",
            "Should use the auto-spawned delegation provider"
        );
        assert_eq!(core.tool_runner_model.as_deref(), Some("delegation-model"));
    }

    #[test]
    fn test_delegation_auto_provider_takes_priority_over_config() {
        // Auto-spawned provider should take priority over config provider
        let dp = MockLLM::named("auto-delegation");
        let config_provider = ProviderConfig {
            api_key: "key".to_string(),
            api_base: Some("http://localhost:9999/v1".to_string()),
        };
        let core = build_test_core(true, Some(dp), Some(config_provider));

        let provider = core.tool_runner_provider.as_ref().unwrap();
        assert_eq!(
            provider.get_default_model(),
            "auto-delegation",
            "Auto-spawned provider should beat config provider"
        );
    }

    #[test]
    fn test_delegation_config_provider_used_when_no_auto() {
        // When no auto provider, but config has one, it should create OpenAICompatProvider
        let config_provider = ProviderConfig {
            api_key: "key".to_string(),
            api_base: Some("http://localhost:9999/v1".to_string()),
        };
        let core = build_test_core(true, None, Some(config_provider));

        assert!(
            core.tool_runner_provider.is_some(),
            "Should have a provider from config"
        );
    }

    #[test]
    fn test_delegation_falls_back_to_main_provider() {
        // When delegation enabled but no auto provider and no config provider,
        // should fall back to main
        let core = build_test_core(true, None, None);

        assert!(core.tool_runner_provider.is_some());
        let provider = core.tool_runner_provider.as_ref().unwrap();
        assert_eq!(
            provider.get_default_model(),
            "main-provider",
            "Should fall back to main provider"
        );
    }

    #[test]
    fn test_delegation_model_uses_config_model() {
        let core = build_test_core(true, None, None);
        assert_eq!(
            core.tool_runner_model.as_deref(),
            Some("delegation-model"),
            "Should use the model from ToolDelegationConfig"
        );
    }

    #[test]
    fn test_delegation_model_falls_back_to_main_when_empty() {
        let workspace = tempfile::tempdir().unwrap().into_path();
        let main = MockLLM::named("main-provider");
        let td = ToolDelegationConfig {
            enabled: true,
            model: String::new(), // Empty → fall back to main model
            auto_local: true,
            ..Default::default()
        };
        let core = build_swappable_core(SwappableCoreConfig {
            provider: main,
            workspace,
            model: "main-model".to_string(),
            max_iterations: 10,
            max_tokens: 4096,
            temperature: 0.7,
            max_context_tokens: 16384,
            brave_api_key: None,
            exec_timeout: 30,
            restrict_to_workspace: false,
            memory_config: MemoryConfig::default(),
            is_local: false,
            compaction_provider: None,
            tool_delegation: td,
            provenance: ProvenanceConfig::default(),
            max_tool_result_chars: 2000,
            delegation_provider: None,
            specialist_provider: None,
            trio_config: TrioConfig::default(),
            model_capabilities_overrides: std::collections::HashMap::new(),
        });
        assert_eq!(
            core.tool_runner_model.as_deref(),
            Some("main-model"),
            "Empty delegation model should fall back to main model"
        );
    }

    #[test]
    fn test_delegation_disabled_ignores_passed_provider() {
        // Even if a delegation_provider is passed, it should be ignored
        // when delegation is disabled.
        let dp = MockLLM::named("auto-delegation");
        let core = build_test_core(false, Some(dp), None);

        assert!(
            core.tool_runner_provider.is_none(),
            "Delegation disabled should ignore passed provider"
        );
        assert!(core.tool_runner_model.is_none());
    }

    #[test]
    fn test_delegation_with_is_local_true() {
        // Verify wiring works when is_local=true (uses lite context builder)
        let workspace = tempfile::tempdir().unwrap().into_path();
        let main = MockLLM::named("local-main");
        let dp = MockLLM::named("local-delegation");
        let td = ToolDelegationConfig {
            enabled: true,
            model: "delegation-model".to_string(),
            auto_local: true,
            ..Default::default()
        };
        let core = build_swappable_core(SwappableCoreConfig {
            provider: main,
            workspace,
            model: "local-model".to_string(),
            max_iterations: 10,
            max_tokens: 4096,
            temperature: 0.7,
            max_context_tokens: 16384,
            brave_api_key: None,
            exec_timeout: 30,
            restrict_to_workspace: false,
            memory_config: MemoryConfig::default(),
            is_local: true,
            compaction_provider: None,
            tool_delegation: td,
            provenance: ProvenanceConfig::default(),
            max_tool_result_chars: 2000,
            delegation_provider: Some(dp),
            specialist_provider: None,
            trio_config: TrioConfig::default(),
            model_capabilities_overrides: std::collections::HashMap::new(),
        });

        assert!(core.is_local);
        assert!(core.tool_runner_provider.is_some());
        assert_eq!(
            core.tool_runner_provider
                .as_ref()
                .unwrap()
                .get_default_model(),
            "local-delegation",
            "Local mode should still use the delegation provider"
        );
    }

    #[test]
    fn test_delegation_with_compaction_and_delegation_providers() {
        // Both compaction and delegation providers set — should not interfere
        let workspace = tempfile::tempdir().unwrap().into_path();
        let main = MockLLM::named("main");
        let compaction = MockLLM::named("compaction");
        let delegation = MockLLM::named("delegation");
        let td = ToolDelegationConfig {
            enabled: true,
            model: "deleg-model".to_string(),
            auto_local: true,
            ..Default::default()
        };
        let core = build_swappable_core(SwappableCoreConfig {
            provider: main,
            workspace,
            model: "main-model".to_string(),
            max_iterations: 10,
            max_tokens: 4096,
            temperature: 0.7,
            max_context_tokens: 16384,
            brave_api_key: None,
            exec_timeout: 30,
            restrict_to_workspace: false,
            memory_config: MemoryConfig::default(),
            is_local: true,
            compaction_provider: Some(compaction),
            tool_delegation: td,
            provenance: ProvenanceConfig::default(),
            max_tool_result_chars: 2000,
            delegation_provider: Some(delegation),
            specialist_provider: None,
            trio_config: TrioConfig::default(),
            model_capabilities_overrides: std::collections::HashMap::new(),
        });

        // Compaction provider goes to memory_provider, delegation to tool_runner
        assert_eq!(
            core.memory_provider.get_default_model(),
            "compaction",
            "Memory should use compaction provider"
        );
        assert_eq!(
            core.tool_runner_provider
                .as_ref()
                .unwrap()
                .get_default_model(),
            "delegation",
            "Tool runner should use delegation provider"
        );
    }

    // -----------------------------------------------------------------------
    // E2E: Full agent loop with LCM enabled against real local LLM.
    //
    // This test requires LM Studio (or compatible) running. Set env vars:
    //   NANOBOT_LCM_TEST_BASE  — API base (default: http://127.0.0.1:1234/v1)
    //   NANOBOT_LCM_TEST_MODEL — Model name (default: local-model)
    //
    // Run with: cargo test test_real_lcm_e2e -- --ignored --nocapture
    // -----------------------------------------------------------------------

    #[tokio::test]
    #[ignore = "requires running local LLM on NANOBOT_LCM_TEST_BASE"]
    async fn test_real_lcm_e2e_compact_and_expand() {
        use crate::config::schema::LcmSchemaConfig;

        let api_base = std::env::var("NANOBOT_LCM_TEST_BASE")
            .unwrap_or_else(|_| "http://127.0.0.1:1234/v1".to_string());
        let model_name = std::env::var("NANOBOT_LCM_TEST_MODEL")
            .unwrap_or_else(|_| "local-model".to_string());

        eprintln!("LCM E2E: using {} model={}", api_base, model_name);

        // Real provider pointing at local LLM.
        let provider: Arc<dyn LLMProvider> = Arc::new(
            OpenAICompatProvider::new("local", Some(&api_base), Some(&model_name)),
        );

        // Warm up: verify the model is responding.
        let warmup = provider
            .chat(
                &[json!({"role": "user", "content": "Reply with exactly: ok"})],
                None,
                Some(&model_name),
                32,
                0.0,
                None,
                None,
            )
            .await;
        match warmup {
            Ok(r) => eprintln!("LCM E2E warmup: {}", r.content.as_deref().unwrap_or("(empty)")),
            Err(e) => panic!("LCM E2E: model not responding at {}: {}", api_base, e),
        }

        let workspace = tempfile::tempdir().unwrap().keep();

        // Build core with small context window + LCM thresholds that trigger fast.
        let core = build_swappable_core(SwappableCoreConfig {
            provider: provider.clone(),
            workspace: workspace.clone(),
            model: model_name.clone(),
            max_iterations: 3,
            max_tokens: 512,
            temperature: 0.3,
            max_context_tokens: 2048, // Tiny so compaction triggers quickly.
            brave_api_key: None,
            exec_timeout: 30,
            restrict_to_workspace: false,
            memory_config: MemoryConfig::default(),
            is_local: true,
            compaction_provider: Some(provider.clone()),
            tool_delegation: ToolDelegationConfig::default(),
            provenance: ProvenanceConfig::default(),
            max_tool_result_chars: 2000,
            delegation_provider: None,
            specialist_provider: None,
            trio_config: TrioConfig::default(),
            model_capabilities_overrides: std::collections::HashMap::new(),
        });
        let counters = Arc::new(crate::agent::agent_core::RuntimeCounters::new(2048));
        let core_handle = AgentHandle::new(core, counters);

        let (inbound_tx, inbound_rx) = tokio::sync::mpsc::unbounded_channel::<InboundMessage>();
        let (outbound_tx, _outbound_rx) = tokio::sync::mpsc::unbounded_channel::<OutboundMessage>();

        let lcm_config = LcmSchemaConfig {
            enabled: true,
            tau_soft: 0.3,  // Trigger early.
            tau_hard: 0.6,
            deterministic_target: 128,
            ..Default::default()
        };

        let agent_loop = AgentLoop::new(
            core_handle,
            inbound_rx,
            outbound_tx,
            inbound_tx,
            None, // no cron
            1,
            None, // no email
            None, // no repl display
            None, // no providers config
            ProprioceptionConfig::default(),
            lcm_config,
            None, // no health registry
        );

        let session_key = "lcm-e2e-test";
        let mut responses = Vec::new();

        // Send 12 verbose messages to fill the tiny 2K context.
        let prompts = [
            "Explain Rust ownership rules in detail with examples of move semantics. Be thorough and give at least 3 examples.",
            "Now explain borrowing and the difference between mutable and immutable references with code examples.",
            "Describe lifetime annotations and why they are needed. Give concrete examples with structs and functions.",
            "What are the rules for lifetime elision? When can you omit lifetime annotations? List all three rules.",
            "Explain smart pointers: Box, Rc, Arc, and when to use each one. Give a real-world use case for each.",
            "What is interior mutability? Explain Cell, RefCell, and Mutex with examples of each.",
            "Describe async/await in Rust. How do Futures work under the hood? Explain the state machine transformation.",
            "Explain trait objects vs generics. When would you use dynamic dispatch vs static dispatch?",
            "What are the differences between String and &str? When should you use each one in function signatures?",
            "Explain the Drop trait and how Rust's destructors work. What is the order of dropping?",
            "Describe the Pin and Unpin traits. Why are they needed for async Rust and self-referential structs?",
            "Explain how pattern matching works in Rust. Cover match, if let, while let, and destructuring.",
        ];

        for (i, prompt) in prompts.iter().enumerate() {
            eprintln!("LCM E2E: sending message {}/{}...", i + 1, prompts.len());
            let resp = agent_loop
                .process_direct(prompt, session_key, "test", "lcm-e2e")
                .await;
            eprintln!(
                "LCM E2E: response {} ({} chars): {}",
                i + 1,
                resp.len(),
                &resp[..resp.len().min(80)]
            );
            assert!(
                !resp.is_empty(),
                "Message {} should get a non-empty response",
                i + 1
            );
            responses.push(resp);
        }

        // Check LCM engine state.
        let engines = agent_loop.shared.lcm_engines.lock().await;
        let engine_arc = engines
            .get(session_key)
            .expect("LCM engine should exist for session");
        let engine = engine_arc.lock().await;

        eprintln!(
            "LCM E2E results: store={} active={} dag_nodes={}",
            engine.store_len(),
            engine.active_len(),
            engine.dag_ref().len()
        );

        // Invariant 1: store has messages from the conversation.
        // Note: with is_local + small context, trim_to_fit_with_age runs before
        // LCM ingestion, so the store only contains messages that survived trimming.
        // The session JSONL (on-disk) is the true immutable store; the in-memory
        // LCM store tracks what entered the active context window.
        assert!(
            engine.store_len() >= 5,
            "Store should have at least 5 messages (system + some turns), got {}",
            engine.store_len()
        );

        // Invariant 2: active context should be shorter than store (compaction happened).
        // With tau_soft=0.3 and 4K context, compaction should trigger early.
        assert!(
            engine.active_len() < engine.store_len(),
            "Active ({}) should be shorter than store ({}) — compaction should have triggered",
            engine.active_len(),
            engine.store_len()
        );

        // Invariant 3: DAG should have at least one summary node.
        assert!(
            engine.dag_ref().len() >= 1,
            "DAG should have at least 1 summary node, got {}",
            engine.dag_ref().len()
        );

        // Invariant 4: every summary node's source IDs resolve to real messages.
        for i in 0..engine.dag_ref().len() {
            let node = engine.dag_ref().get(i).unwrap();
            let expanded = engine.expand(&node.source_ids);
            assert_eq!(
                expanded.len(),
                node.source_ids.len(),
                "Summary node {} has {} source IDs but only {} resolve",
                i,
                node.source_ids.len(),
                expanded.len()
            );
            eprintln!(
                "  DAG node {}: level={} sources={:?} tokens={}",
                i, node.level, node.source_ids, node.tokens
            );
        }

        // Invariant 5: active context contains at least one Summary entry.
        let summary_count = engine
            .active_entries()
            .iter()
            .filter(|e| matches!(e, crate::agent::lcm::ContextEntry::Summary { .. }))
            .count();
        assert!(
            summary_count >= 1,
            "Active context should have at least 1 summary entry, got {}",
            summary_count
        );

        // Invariant 6: lossless expand — all store IDs are retrievable.
        let all_ids: Vec<usize> = (0..engine.store_len()).collect();
        let expanded = engine.expand(&all_ids);
        assert_eq!(
            expanded.len(),
            engine.store_len(),
            "All {} store messages should be retrievable via expand",
            engine.store_len()
        );
        for (id, msg) in &expanded {
            let content = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
            assert!(
                !content.is_empty(),
                "Expanded message {} should have content",
                id
            );
        }

        eprintln!("LCM E2E: ALL INVARIANTS PASSED");
        eprintln!(
            "  Messages: {} stored, {} active, {} summary nodes",
            engine.store_len(),
            engine.active_len(),
            engine.dag_ref().len()
        );

        // Cleanup.
        drop(engine);
        drop(engines);
        let _ = std::fs::remove_dir_all(&workspace);
    }

    #[tokio::test]
    async fn test_compaction_timeout_resets_in_flight() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;
        use std::time::Duration;

        let in_flight = Arc::new(AtomicBool::new(false));

        // Simulate: set in_flight before spawning compaction
        in_flight.store(true, Ordering::SeqCst);
        assert!(in_flight.load(Ordering::SeqCst));

        let flag = in_flight.clone();
        let handle = tokio::spawn(async move {
            let timeout_result = tokio::time::timeout(
                Duration::from_millis(100), // Short timeout for test
                async {
                    // Simulate a hanging compaction endpoint
                    tokio::time::sleep(Duration::from_secs(60)).await;
                },
            )
            .await;
            assert!(timeout_result.is_err(), "should have timed out");
            flag.store(false, Ordering::SeqCst); // Must always execute
        });

        // Wait for the spawned task to complete
        handle.await.unwrap();

        // The critical assertion: in_flight must be reset even after timeout
        assert!(
            !in_flight.load(Ordering::SeqCst),
            "in_flight must reset to false after timeout"
        );
    }

    // -----------------------------------------------------------------------
    // Trio E2E test harness
    //
    // All tests require a single LM Studio endpoint serving three models.
    // Configure via env vars:
    //   NANOBOT_TRIO_BASE            — API base (default: http://192.168.1.22:1234/v1)
    //   NANOBOT_TRIO_MAIN_MODEL      — Main model name
    //   NANOBOT_TRIO_ROUTER_MODEL    — Router model name
    //   NANOBOT_TRIO_SPECIALIST_MODEL — Specialist model name
    //
    // Run with: cargo test test_trio_e2e -- --ignored --nocapture
    // -----------------------------------------------------------------------

    /// Read trio E2E env vars (single shared endpoint).
    fn trio_e2e_env() -> (String, String, String, String) {
        let base = std::env::var("NANOBOT_TRIO_BASE")
            .unwrap_or_else(|_| "http://192.168.1.22:1234/v1".to_string());
        let main_model = std::env::var("NANOBOT_TRIO_MAIN_MODEL")
            .unwrap_or_else(|_| "gemma-3n-e4b-it".to_string());
        let router_model = std::env::var("NANOBOT_TRIO_ROUTER_MODEL")
            .unwrap_or_else(|_| "nvidia_orchestrator-8b".to_string());
        let specialist_model = std::env::var("NANOBOT_TRIO_SPECIALIST_MODEL")
            .unwrap_or_else(|_| "qwen3-1.7b".to_string());
        (base, main_model, router_model, specialist_model)
    }

    /// Build an AgentLoop wired for trio E2E testing.
    ///
    /// All three providers share one LM Studio endpoint, differentiated by model name.
    /// A shared JitGate serialises requests to prevent concurrent model-loading crashes.
    fn build_trio_e2e_harness(
        base_url: &str,
        main_model: &str,
        router_model: &str,
        specialist_model: &str,
    ) -> (AgentLoop, std::path::PathBuf) {
        use crate::providers::factory;
        use crate::providers::jit_gate::JitGate;
        use crate::config::schema::LcmSchemaConfig;

        let jit_gate = std::sync::Arc::new(JitGate::new());

        let main_provider: Arc<dyn LLMProvider> = factory::create_openai_compat(
            factory::ProviderSpec::local(base_url, Some(main_model))
                .with_jit_gate_opt(Some(jit_gate.clone())),
        );
        let router_provider: Arc<dyn LLMProvider> = factory::create_openai_compat(
            factory::ProviderSpec::local(base_url, Some(router_model))
                .with_jit_gate_opt(Some(jit_gate.clone())),
        );
        let specialist_provider: Arc<dyn LLMProvider> = factory::create_openai_compat(
            factory::ProviderSpec::local(base_url, Some(specialist_model))
                .with_jit_gate_opt(Some(jit_gate.clone())),
        );

        let workspace = tempfile::tempdir().unwrap().into_path();

        let mut td = ToolDelegationConfig {
            mode: crate::config::schema::DelegationMode::Trio,
            ..Default::default()
        };
        td.apply_mode();

        let trio_config = TrioConfig {
            enabled: true,
            router_model: router_model.to_string(),
            specialist_model: specialist_model.to_string(),
            ..Default::default()
        };

        let core = build_swappable_core(SwappableCoreConfig {
            provider: main_provider,
            workspace: workspace.clone(),
            model: main_model.to_string(),
            max_iterations: 5,
            max_tokens: 512,
            temperature: 0.3,
            max_context_tokens: 4096,
            brave_api_key: None,
            exec_timeout: 30,
            restrict_to_workspace: true,
            memory_config: MemoryConfig::default(),
            is_local: true,
            compaction_provider: None,
            tool_delegation: td,
            provenance: ProvenanceConfig::default(),
            max_tool_result_chars: 2000,
            delegation_provider: Some(router_provider),
            specialist_provider: Some(specialist_provider),
            trio_config,
            model_capabilities_overrides: std::collections::HashMap::new(),
        });

        let counters = Arc::new(crate::agent::agent_core::RuntimeCounters::new(4096));
        let core_handle = AgentHandle::new(core, counters);

        let (inbound_tx, inbound_rx) = tokio::sync::mpsc::unbounded_channel::<InboundMessage>();
        let (outbound_tx, _outbound_rx) = tokio::sync::mpsc::unbounded_channel::<OutboundMessage>();

        let agent_loop = AgentLoop::new(
            core_handle,
            inbound_rx,
            outbound_tx,
            inbound_tx,
            None,
            1,
            None,
            None,
            None,
            ProprioceptionConfig::default(),
            LcmSchemaConfig::default(),
            None,
        );

        (agent_loop, workspace)
    }

    /// Warmup a provider with backon retries (models may need JIT loading time).
    async fn warmup_trio_provider(
        provider: &dyn LLMProvider,
        model: &str,
        role: &str,
    ) {
        use backon::ConstantBuilder;

        let messages = vec![serde_json::json!({"role": "user", "content": "Reply with: ok"})];
        let mut backoff = ConstantBuilder::default()
            .with_delay(Duration::from_secs(2))
            .with_max_times(10)
            .build();
        loop {
            match provider.chat(&messages, None, Some(model), 32, 0.0, None, None).await {
                Ok(resp) => {
                    let text = resp.content.unwrap_or_default();
                    if !text.trim().is_empty() {
                        eprintln!("  {} warmup OK: {}", role, &text[..text.len().min(40)]);
                        return;
                    }
                }
                Err(e) => {
                    let msg = e.to_string().to_lowercase();
                    if !msg.contains("loading") && !msg.contains("503") {
                        panic!("{} warmup failed (non-retryable): {}", role, e);
                    }
                }
            }
            match backoff.next() {
                Some(delay) => {
                    eprintln!("  {} warming up, retrying in {:?}...", role, delay);
                    tokio::time::sleep(delay).await;
                }
                None => panic!("{} did not become ready after retries", role),
            }
        }
    }

    #[tokio::test]
    #[ignore = "requires LM Studio at NANOBOT_TRIO_BASE"]
    async fn test_trio_e2e_preflight() {
        let (base, main_model, router_model, specialist_model) = trio_e2e_env();
        eprintln!("trio E2E preflight: base={}", base);

        // 1. Verify LM Studio /models endpoint is reachable
        let models_url = format!("{}/models", base.trim_end_matches("/v1").trim_end_matches('/'));
        // Try the /v1/models path first (standard OpenAI-compat)
        let models_url_v1 = format!("{}/models", base.trim_end_matches('/'));
        let client = reqwest::Client::new();
        let models_resp = client
            .get(&models_url_v1)
            .header("Authorization", "Bearer local")
            .timeout(Duration::from_secs(10))
            .send()
            .await;

        match &models_resp {
            Ok(resp) if resp.status().is_success() => {
                eprintln!("  /models endpoint OK (status {})", resp.status());
            }
            Ok(resp) => {
                panic!(
                    "preflight FAILED: /models returned HTTP {} — is LM Studio running at {}?",
                    resp.status(),
                    base
                );
            }
            Err(e) => {
                panic!(
                    "preflight FAILED: cannot reach {} — {}\nStart LM Studio or set NANOBOT_TRIO_BASE.",
                    models_url_v1, e
                );
            }
        }

        // 2. Parse model list and check availability
        let body: serde_json::Value = models_resp
            .unwrap()
            .json()
            .await
            .expect("preflight: /models response is not valid JSON");

        let model_ids: Vec<String> = body
            .get("data")
            .and_then(|d| d.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|m| m.get("id").and_then(|id| id.as_str()).map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        eprintln!("  available models: {:?}", model_ids);

        // Note: LM Studio with JIT loading may not list all models upfront.
        // We log availability but don't fail — the warmup step below is the real gate.
        for (name, role) in [
            (&main_model, "main"),
            (&router_model, "router"),
            (&specialist_model, "specialist"),
        ] {
            if model_ids.iter().any(|id| id.contains(name.as_str())) {
                eprintln!("  {} model '{}' found in /models", role, name);
            } else {
                eprintln!("  {} model '{}' NOT listed (may JIT-load on demand)", role, name);
            }
        }

        // 3. Build harness and warmup all 3 providers (the real gate)
        let (agent_loop, workspace) =
            build_trio_e2e_harness(&base, &main_model, &router_model, &specialist_model);

        let core = agent_loop.shared.core_handle.swappable();
        warmup_trio_provider(&*core.provider, &main_model, "main").await;
        warmup_trio_provider(
            core.router_provider.as_ref().unwrap().as_ref(),
            &router_model,
            "router",
        )
        .await;
        warmup_trio_provider(
            core.specialist_provider.as_ref().unwrap().as_ref(),
            &specialist_model,
            "specialist",
        )
        .await;

        eprintln!("trio E2E preflight: ALL OK — infrastructure ready");
        let _ = std::fs::remove_dir_all(&workspace);
    }

    #[tokio::test]
    #[ignore = "requires LM Studio at NANOBOT_TRIO_BASE"]
    async fn test_trio_e2e_respond() {
        let (base, main_model, router_model, specialist_model) = trio_e2e_env();
        eprintln!("trio E2E respond: base={}", base);

        let (agent_loop, workspace) = build_trio_e2e_harness(&base, &main_model, &router_model, &specialist_model);

        // Warmup all 3 models
        let core = agent_loop.shared.core_handle.swappable();
        warmup_trio_provider(&*core.provider, &main_model, "main").await;
        warmup_trio_provider(core.router_provider.as_ref().unwrap().as_ref(), &router_model, "router").await;
        warmup_trio_provider(core.specialist_provider.as_ref().unwrap().as_ref(), &specialist_model, "specialist").await;

        let resp = tokio::time::timeout(
            Duration::from_secs(180),
            agent_loop.process_direct("Hello, what is 2 + 2?", "trio-e2e-respond", "test", "trio-e2e"),
        )
        .await
        .expect("test timed out");

        eprintln!("trio E2E respond: response ({} chars): {}", resp.len(), &resp[..resp.len().min(200)]);
        assert!(!resp.is_empty(), "response should be non-empty");

        let _ = std::fs::remove_dir_all(&workspace);
    }

    #[tokio::test]
    #[ignore = "requires LM Studio at NANOBOT_TRIO_BASE"]
    async fn test_trio_e2e_tool_dispatch() {
        let (base, main_model, router_model, specialist_model) = trio_e2e_env();
        eprintln!("trio E2E tool dispatch: base={}", base);

        let (agent_loop, workspace) = build_trio_e2e_harness(&base, &main_model, &router_model, &specialist_model);

        // Write a known file to workspace
        std::fs::write(workspace.join("README.md"), "Nanobot is a lightweight AI assistant framework written in Rust.").unwrap();

        let core = agent_loop.shared.core_handle.swappable();
        warmup_trio_provider(&*core.provider, &main_model, "main").await;
        warmup_trio_provider(core.router_provider.as_ref().unwrap().as_ref(), &router_model, "router").await;
        warmup_trio_provider(core.specialist_provider.as_ref().unwrap().as_ref(), &specialist_model, "specialist").await;

        let resp = tokio::time::timeout(
            Duration::from_secs(180),
            agent_loop.process_direct(
                "Read the file README.md and tell me what it says",
                "trio-e2e-tool",
                "test",
                "trio-e2e",
            ),
        )
        .await
        .expect("test timed out");

        eprintln!("trio E2E tool dispatch: response ({} chars): {}", resp.len(), &resp[..resp.len().min(200)]);
        assert!(!resp.is_empty(), "response should be non-empty");

        // Check TrioMetrics
        let metrics = &agent_loop.shared.core_handle.counters.trio_metrics;
        eprintln!(
            "  metrics: preflight={} action={:?} specialist={} tool={:?}",
            metrics.router_preflight_fired.load(std::sync::atomic::Ordering::Relaxed),
            metrics.router_action.lock().unwrap(),
            metrics.specialist_dispatched.load(std::sync::atomic::Ordering::Relaxed),
            metrics.tool_dispatched.lock().unwrap(),
        );

        let _ = std::fs::remove_dir_all(&workspace);
    }

    #[tokio::test]
    #[ignore = "requires LM Studio at NANOBOT_TRIO_BASE"]
    async fn test_trio_e2e_specialist_dispatch() {
        let (base, main_model, router_model, specialist_model) = trio_e2e_env();
        eprintln!("trio E2E specialist: base={}", base);

        let (agent_loop, workspace) = build_trio_e2e_harness(&base, &main_model, &router_model, &specialist_model);

        let core = agent_loop.shared.core_handle.swappable();
        warmup_trio_provider(&*core.provider, &main_model, "main").await;
        warmup_trio_provider(core.router_provider.as_ref().unwrap().as_ref(), &router_model, "router").await;
        warmup_trio_provider(core.specialist_provider.as_ref().unwrap().as_ref(), &specialist_model, "specialist").await;

        let resp = tokio::time::timeout(
            Duration::from_secs(180),
            agent_loop.process_direct(
                "Provide a detailed technical analysis of REST vs GraphQL",
                "trio-e2e-specialist",
                "test",
                "trio-e2e",
            ),
        )
        .await
        .expect("test timed out");

        eprintln!("trio E2E specialist: response ({} chars): {}", resp.len(), &resp[..resp.len().min(200)]);
        assert!(!resp.is_empty(), "response should be non-empty");
        assert!(resp.len() > 50, "specialist response should be substantive (>50 chars)");

        let _ = std::fs::remove_dir_all(&workspace);
    }

    #[tokio::test]
    #[ignore = "requires LM Studio at NANOBOT_TRIO_BASE"]
    async fn test_trio_e2e_ask_user() {
        let (base, main_model, router_model, specialist_model) = trio_e2e_env();
        eprintln!("trio E2E ask_user: base={}", base);

        let (agent_loop, workspace) = build_trio_e2e_harness(&base, &main_model, &router_model, &specialist_model);

        let core = agent_loop.shared.core_handle.swappable();
        warmup_trio_provider(&*core.provider, &main_model, "main").await;
        warmup_trio_provider(core.router_provider.as_ref().unwrap().as_ref(), &router_model, "router").await;
        warmup_trio_provider(core.specialist_provider.as_ref().unwrap().as_ref(), &specialist_model, "specialist").await;

        let resp = tokio::time::timeout(
            Duration::from_secs(180),
            agent_loop.process_direct(
                "Do that thing with the file",
                "trio-e2e-ask",
                "test",
                "trio-e2e",
            ),
        )
        .await
        .expect("test timed out");

        eprintln!("trio E2E ask_user: response ({} chars): {}", resp.len(), &resp[..resp.len().min(200)]);
        assert!(!resp.is_empty(), "response should be non-empty");

        let _ = std::fs::remove_dir_all(&workspace);
    }

    #[tokio::test]
    #[ignore = "requires LM Studio at NANOBOT_TRIO_BASE"]
    async fn test_trio_e2e_router_unreachable() {
        let (base, main_model, _router_model, specialist_model) = trio_e2e_env();
        eprintln!("trio E2E router unreachable: base={}", base);

        // Router on dead port, main + specialist on real endpoint
        let (agent_loop, workspace) = build_trio_e2e_harness(
            &base,
            &main_model,
            &"unreachable-router-model".to_string(), // model doesn't matter since we override the provider
            &specialist_model,
        );

        // Actually, the harness uses shared base for all providers.
        // For unreachable router, we need a custom build with bad router URL.
        // Let's build it manually.
        drop(agent_loop);
        let _ = std::fs::remove_dir_all(&workspace);

        use crate::providers::factory;
        use crate::providers::jit_gate::JitGate;
        use crate::config::schema::{DelegationMode, LcmSchemaConfig};

        let jit_gate = std::sync::Arc::new(JitGate::new());
        let main_provider: Arc<dyn LLMProvider> = factory::create_openai_compat(
            factory::ProviderSpec::local(&base, Some(&main_model))
                .with_jit_gate_opt(Some(jit_gate.clone())),
        );
        // Router points to dead port
        let router_provider: Arc<dyn LLMProvider> = Arc::new(
            OpenAICompatProvider::new("local", Some("http://127.0.0.1:19999/v1"), Some("dead-router")),
        );
        let specialist_provider: Arc<dyn LLMProvider> = factory::create_openai_compat(
            factory::ProviderSpec::local(&base, Some(&specialist_model))
                .with_jit_gate_opt(Some(jit_gate.clone())),
        );

        let workspace = tempfile::tempdir().unwrap().into_path();
        let mut td = ToolDelegationConfig {
            mode: DelegationMode::Trio,
            ..Default::default()
        };
        td.apply_mode();

        let trio_config = TrioConfig {
            enabled: true,
            router_model: "dead-router".to_string(),
            specialist_model: specialist_model.to_string(),
            ..Default::default()
        };

        let core = build_swappable_core(SwappableCoreConfig {
            provider: main_provider,
            workspace: workspace.clone(),
            model: main_model.to_string(),
            max_iterations: 5,
            max_tokens: 512,
            temperature: 0.3,
            max_context_tokens: 4096,
            brave_api_key: None,
            exec_timeout: 30,
            restrict_to_workspace: true,
            memory_config: MemoryConfig::default(),
            is_local: true,
            compaction_provider: None,
            tool_delegation: td,
            provenance: ProvenanceConfig::default(),
            max_tool_result_chars: 2000,
            delegation_provider: Some(router_provider),
            specialist_provider: Some(specialist_provider),
            trio_config,
            model_capabilities_overrides: std::collections::HashMap::new(),
        });
        let counters = Arc::new(crate::agent::agent_core::RuntimeCounters::new(4096));
        let core_handle = AgentHandle::new(core, counters);

        let (inbound_tx, inbound_rx) = tokio::sync::mpsc::unbounded_channel::<InboundMessage>();
        let (outbound_tx, _outbound_rx) = tokio::sync::mpsc::unbounded_channel::<OutboundMessage>();

        let agent_loop = AgentLoop::new(
            core_handle,
            inbound_rx,
            outbound_tx,
            inbound_tx,
            None,
            1,
            None,
            None,
            None,
            ProprioceptionConfig::default(),
            LcmSchemaConfig::default(),
            None,
        );

        // Only warmup main (router is intentionally dead)
        let core = agent_loop.shared.core_handle.swappable();
        warmup_trio_provider(&*core.provider, &main_model, "main").await;

        let resp = tokio::time::timeout(
            Duration::from_secs(60),
            agent_loop.process_direct("Hello", "trio-e2e-router-dead", "test", "trio-e2e"),
        )
        .await
        .expect("test timed out");

        eprintln!("trio E2E router unreachable: response ({} chars): {}", resp.len(), &resp[..resp.len().min(200)]);
        assert!(!resp.is_empty(), "should get error response, not panic");

        let _ = std::fs::remove_dir_all(&workspace);
    }

    #[tokio::test]
    #[ignore = "requires LM Studio at NANOBOT_TRIO_BASE"]
    async fn test_trio_e2e_specialist_unreachable() {
        let (base, main_model, router_model, _specialist_model) = trio_e2e_env();
        eprintln!("trio E2E specialist unreachable: base={}", base);

        use crate::providers::factory;
        use crate::providers::jit_gate::JitGate;
        use crate::config::schema::{DelegationMode, LcmSchemaConfig};

        let jit_gate = std::sync::Arc::new(JitGate::new());
        let main_provider: Arc<dyn LLMProvider> = factory::create_openai_compat(
            factory::ProviderSpec::local(&base, Some(&main_model))
                .with_jit_gate_opt(Some(jit_gate.clone())),
        );
        let router_provider: Arc<dyn LLMProvider> = factory::create_openai_compat(
            factory::ProviderSpec::local(&base, Some(&router_model))
                .with_jit_gate_opt(Some(jit_gate.clone())),
        );
        // Specialist points to dead port
        let specialist_provider: Arc<dyn LLMProvider> = Arc::new(
            OpenAICompatProvider::new("local", Some("http://127.0.0.1:19999/v1"), Some("dead-specialist")),
        );

        let workspace = tempfile::tempdir().unwrap().into_path();
        let mut td = ToolDelegationConfig {
            mode: DelegationMode::Trio,
            ..Default::default()
        };
        td.apply_mode();

        let trio_config = TrioConfig {
            enabled: true,
            router_model: router_model.to_string(),
            specialist_model: "dead-specialist".to_string(),
            ..Default::default()
        };

        let core = build_swappable_core(SwappableCoreConfig {
            provider: main_provider,
            workspace: workspace.clone(),
            model: main_model.to_string(),
            max_iterations: 5,
            max_tokens: 512,
            temperature: 0.3,
            max_context_tokens: 4096,
            brave_api_key: None,
            exec_timeout: 30,
            restrict_to_workspace: true,
            memory_config: MemoryConfig::default(),
            is_local: true,
            compaction_provider: None,
            tool_delegation: td,
            provenance: ProvenanceConfig::default(),
            max_tool_result_chars: 2000,
            delegation_provider: Some(router_provider),
            specialist_provider: Some(specialist_provider),
            trio_config,
            model_capabilities_overrides: std::collections::HashMap::new(),
        });
        let counters = Arc::new(crate::agent::agent_core::RuntimeCounters::new(4096));
        let core_handle = AgentHandle::new(core, counters);

        let (inbound_tx, inbound_rx) = tokio::sync::mpsc::unbounded_channel::<InboundMessage>();
        let (outbound_tx, _outbound_rx) = tokio::sync::mpsc::unbounded_channel::<OutboundMessage>();

        let agent_loop = AgentLoop::new(
            core_handle,
            inbound_rx,
            outbound_tx,
            inbound_tx,
            None,
            1,
            None,
            None,
            None,
            ProprioceptionConfig::default(),
            LcmSchemaConfig::default(),
            None,
        );

        let core = agent_loop.shared.core_handle.swappable();
        warmup_trio_provider(&*core.provider, &main_model, "main").await;
        warmup_trio_provider(core.router_provider.as_ref().unwrap().as_ref(), &router_model, "router").await;

        let resp = tokio::time::timeout(
            Duration::from_secs(180),
            agent_loop.process_direct(
                "Provide a detailed technical analysis of REST vs GraphQL",
                "trio-e2e-specialist-dead",
                "test",
                "trio-e2e",
            ),
        )
        .await
        .expect("test timed out");

        eprintln!("trio E2E specialist unreachable: response ({} chars): {}", resp.len(), &resp[..resp.len().min(200)]);
        assert!(!resp.is_empty(), "should get response despite dead specialist");

        let _ = std::fs::remove_dir_all(&workspace);
    }

    #[tokio::test]
    #[ignore = "requires LM Studio at NANOBOT_TRIO_BASE"]
    async fn test_trio_e2e_multi_turn() {
        let (base, main_model, router_model, specialist_model) = trio_e2e_env();
        eprintln!("trio E2E multi-turn: base={}", base);

        let (agent_loop, workspace) = build_trio_e2e_harness(&base, &main_model, &router_model, &specialist_model);

        // Write test file
        std::fs::write(workspace.join("README.md"), "Nanobot is a lightweight AI assistant.").unwrap();

        let core = agent_loop.shared.core_handle.swappable();
        warmup_trio_provider(&*core.provider, &main_model, "main").await;
        warmup_trio_provider(core.router_provider.as_ref().unwrap().as_ref(), &router_model, "router").await;
        warmup_trio_provider(core.specialist_provider.as_ref().unwrap().as_ref(), &specialist_model, "specialist").await;

        let session_key = "trio-e2e-multi";

        // Turn 1: simple greeting (respond path)
        let resp1 = tokio::time::timeout(
            Duration::from_secs(180),
            agent_loop.process_direct("Hello", session_key, "test", "trio-e2e"),
        )
        .await
        .expect("turn 1 timed out");
        eprintln!("turn 1 ({} chars): {}", resp1.len(), &resp1[..resp1.len().min(100)]);
        assert!(!resp1.is_empty(), "turn 1 should be non-empty");

        // Turn 2: tool path
        let resp2 = tokio::time::timeout(
            Duration::from_secs(180),
            agent_loop.process_direct("Read README.md", session_key, "test", "trio-e2e"),
        )
        .await
        .expect("turn 2 timed out");
        eprintln!("turn 2 ({} chars): {}", resp2.len(), &resp2[..resp2.len().min(100)]);
        assert!(!resp2.is_empty(), "turn 2 should be non-empty");

        // Turn 3: follow-up (tests session state persistence)
        let resp3 = tokio::time::timeout(
            Duration::from_secs(180),
            agent_loop.process_direct("Summarize what you found", session_key, "test", "trio-e2e"),
        )
        .await
        .expect("turn 3 timed out");
        eprintln!("turn 3 ({} chars): {}", resp3.len(), &resp3[..resp3.len().min(100)]);
        assert!(!resp3.is_empty(), "turn 3 should be non-empty");

        let _ = std::fs::remove_dir_all(&workspace);
    }

    // -----------------------------------------------------------------------
    // should_strip_tools_for_trio — pure function tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_should_strip_tools_all_healthy() {
        assert!(should_strip_tools_for_trio(true, true, true, true));
    }

    #[test]
    fn test_should_strip_tools_not_local() {
        // Cloud mode: never strip tools via this path.
        assert!(!should_strip_tools_for_trio(false, true, true, true));
    }

    #[test]
    fn test_should_strip_tools_no_strict_mode() {
        // strict_no_tools_main is false: don't strip.
        assert!(!should_strip_tools_for_trio(true, false, true, true));
    }

    #[test]
    fn test_should_strip_tools_router_unhealthy() {
        // Router probe degraded: keep tools for fallback.
        assert!(!should_strip_tools_for_trio(true, true, false, true));
    }

    #[test]
    fn test_should_strip_tools_circuit_breaker_open() {
        // Circuit breaker tripped: keep tools for fallback.
        assert!(!should_strip_tools_for_trio(true, true, true, false));
    }

    #[test]
    fn test_should_strip_tools_both_degraded() {
        // Both degraded: definitely keep tools.
        assert!(!should_strip_tools_for_trio(true, true, false, false));
    }

    // -----------------------------------------------------------------------
    // Offline trio E2E tests (no network required — all providers are mocks)
    // -----------------------------------------------------------------------

    /// A mock LLM provider that returns responses from a pre-loaded queue.
    ///
    /// Each call pops the next response. When the queue is empty it returns a
    /// sentinel error string so tests can detect over-calling.
    struct SequenceProvider {
        name: String,
        responses: std::sync::Mutex<std::collections::VecDeque<String>>,
        call_count: std::sync::atomic::AtomicU32,
    }

    impl SequenceProvider {
        fn new(name: &str, responses: Vec<&str>) -> Self {
            Self {
                name: name.to_string(),
                responses: std::sync::Mutex::new(
                    responses.into_iter().map(|s| s.to_string()).collect(),
                ),
                call_count: std::sync::atomic::AtomicU32::new(0),
            }
        }

        fn call_count(&self) -> u32 {
            self.call_count.load(std::sync::atomic::Ordering::Relaxed)
        }
    }

    #[async_trait]
    impl LLMProvider for SequenceProvider {
        async fn chat(
            &self,
            _messages: &[Value],
            _tools: Option<&[Value]>,
            _model: Option<&str>,
            _max_tokens: u32,
            _temperature: f64,
            _thinking_budget: Option<u32>,
            _top_p: Option<f64>,
        ) -> anyhow::Result<crate::providers::base::LLMResponse> {
            self.call_count
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let response = {
                let mut deque = self.responses.lock().unwrap();
                if deque.is_empty() {
                    "ERROR: no responses left in SequenceProvider".to_string()
                } else {
                    deque.pop_front().unwrap()
                }
            };
            Ok(crate::providers::base::LLMResponse {
                content: Some(response),
                tool_calls: vec![],
                finish_reason: "stop".to_string(),
                usage: std::collections::HashMap::new(),
            })
        }

        fn get_default_model(&self) -> &str {
            &self.name
        }
    }

    /// Build an offline trio harness from pre-built mock providers.
    ///
    /// Mirrors `build_trio_e2e_harness` but accepts providers directly rather
    /// than constructing real HTTP clients. No background probes are wired.
    fn build_trio_offline_harness(
        main: Arc<dyn LLMProvider>,
        router: Arc<dyn LLMProvider>,
        specialist: Arc<dyn LLMProvider>,
    ) -> (AgentLoop, std::path::PathBuf) {
        use crate::config::schema::LcmSchemaConfig;

        let workspace = tempfile::tempdir().unwrap().into_path();

        let mut td = ToolDelegationConfig {
            mode: crate::config::schema::DelegationMode::Trio,
            ..Default::default()
        };
        td.apply_mode(); // sets strict_no_tools_main = true, strict_router_schema = true

        let router_model = router.get_default_model().to_string();
        let specialist_model = specialist.get_default_model().to_string();

        let trio_config = TrioConfig {
            enabled: true,
            router_model: router_model.clone(),
            specialist_model: specialist_model.clone(),
            ..Default::default()
        };

        let core = build_swappable_core(SwappableCoreConfig {
            provider: main,
            workspace: workspace.clone(),
            model: "offline-main".to_string(),
            max_iterations: 5,
            max_tokens: 512,
            temperature: 0.3,
            max_context_tokens: 4096,
            brave_api_key: None,
            exec_timeout: 30,
            restrict_to_workspace: true,
            memory_config: MemoryConfig::default(),
            is_local: true,
            compaction_provider: None,
            tool_delegation: td,
            provenance: ProvenanceConfig::default(),
            max_tool_result_chars: 2000,
            delegation_provider: Some(router),
            specialist_provider: Some(specialist),
            trio_config,
            model_capabilities_overrides: std::collections::HashMap::new(),
        });

        let counters = Arc::new(crate::agent::agent_core::RuntimeCounters::new(4096));
        let core_handle = AgentHandle::new(core, counters);

        let (inbound_tx, inbound_rx) =
            tokio::sync::mpsc::unbounded_channel::<InboundMessage>();
        let (outbound_tx, _outbound_rx) =
            tokio::sync::mpsc::unbounded_channel::<OutboundMessage>();

        let agent_loop = AgentLoop::new(
            core_handle,
            inbound_rx,
            outbound_tx,
            inbound_tx,
            None,
            1,
            None,
            None,
            None,
            ProprioceptionConfig::default(),
            LcmSchemaConfig::default(),
            None, // no health_registry — offline tests manage their own
        );

        (agent_loop, workspace)
    }

    // -----------------------------------------------------------------------
    // Test 1: router decides "respond" — specialist is never called
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_trio_offline_e2e_respond() {
        let router_resp = r#"{"action":"respond","target":"main","args":{},"confidence":0.9}"#;
        let main_resp = "Four.";

        let router: Arc<dyn LLMProvider> = Arc::new(SequenceProvider::new(
            "offline-router",
            vec![router_resp, router_resp, router_resp],
        ));
        let main: Arc<dyn LLMProvider> = Arc::new(StaticResponseLLM::new("offline-main", main_resp));
        let specialist: Arc<dyn LLMProvider> = Arc::new(StaticResponseLLM::new(
            "offline-specialist",
            "specialist unused",
        ));

        let (agent_loop, workspace) =
            build_trio_offline_harness(main, router, specialist);

        let resp = agent_loop
            .process_direct("What is 2+2?", "trio-offline-respond", "test", "offline")
            .await;

        eprintln!(
            "test_trio_offline_e2e_respond: response ({} chars): {}",
            resp.len(),
            &resp[..resp.len().min(200)]
        );

        let counters = &agent_loop.shared.core_handle.counters;
        let metrics = &counters.trio_metrics;

        assert!(
            metrics.router_preflight_fired.load(std::sync::atomic::Ordering::Relaxed),
            "router preflight should have fired"
        );
        assert_eq!(
            metrics.router_action.lock().unwrap().as_deref(),
            Some("respond"),
            "router_action should be 'respond'"
        );
        assert!(
            !metrics.specialist_dispatched.load(std::sync::atomic::Ordering::Relaxed),
            "specialist should NOT have been dispatched for a 'respond' decision"
        );
        assert!(!resp.is_empty(), "response should be non-empty");

        let _ = std::fs::remove_dir_all(&workspace);
    }

    // -----------------------------------------------------------------------
    // Test 2: router decides "specialist" — specialist is called
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_trio_offline_e2e_specialist_dispatch() {
        let router_resp = r#"{"action":"specialist","target":"coding","args":{"task":"explain loops"},"confidence":0.85}"#;

        let router: Arc<dyn LLMProvider> = Arc::new(SequenceProvider::new(
            "offline-router",
            vec![router_resp, router_resp, router_resp],
        ));
        let main: Arc<dyn LLMProvider> =
            Arc::new(StaticResponseLLM::new("offline-main", "delegating"));
        let specialist: Arc<dyn LLMProvider> = Arc::new(StaticResponseLLM::new(
            "offline-specialist",
            "Here is the specialist answer.",
        ));

        let (agent_loop, workspace) =
            build_trio_offline_harness(main, router, specialist);

        let resp = agent_loop
            .process_direct(
                "Explain for loops",
                "trio-offline-specialist",
                "test",
                "offline",
            )
            .await;

        eprintln!(
            "test_trio_offline_e2e_specialist_dispatch: response ({} chars): {}",
            resp.len(),
            &resp[..resp.len().min(200)]
        );

        let metrics = &agent_loop.shared.core_handle.counters.trio_metrics;

        assert_eq!(
            metrics.router_action.lock().unwrap().as_deref(),
            Some("specialist"),
            "router_action should be 'specialist'"
        );
        assert!(
            metrics.specialist_dispatched.load(std::sync::atomic::Ordering::Relaxed),
            "specialist should have been dispatched"
        );
        assert!(!resp.is_empty(), "response should be non-empty");

        let _ = std::fs::remove_dir_all(&workspace);
    }

    // -----------------------------------------------------------------------
    // Test 3: circuit breaker cascade
    //
    // The router returns non-JSON 3+ times. Each failure is recorded under
    // the key "router:{model}" (as router.rs does). However, agent_loop.rs
    // checks availability under "trio_router" — so the CB check at the
    // should_strip_tools_for_trio call site never sees the tripped breaker.
    //
    // This test documents that discrepancy explicitly.
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_trio_offline_e2e_circuit_breaker_cascade() {
        // All 4 router calls return non-JSON to trip the circuit breaker.
        let router: Arc<dyn LLMProvider> = Arc::new(SequenceProvider::new(
            "offline-router",
            vec![
                "this is not json at all !!!",
                "this is not json at all !!!",
                "this is not json at all !!!",
                "this is not json at all !!!",
            ],
        ));
        let main: Arc<dyn LLMProvider> = Arc::new(StaticResponseLLM::new(
            "offline-main",
            "main fallback response",
        ));
        let specialist: Arc<dyn LLMProvider> =
            Arc::new(StaticResponseLLM::new("offline-specialist", "specialist unused"));

        let (agent_loop, workspace) =
            build_trio_offline_harness(main, router, specialist);

        // Send 4 messages — each failure increments the CB counter.
        // After 3 failures (default threshold) the CB is tripped.
        // The 4th call will be via Passthrough (router returns early) because
        // the CB key "router:offline-router" is open. Main answers directly.
        for i in 0..4u32 {
            let resp = agent_loop
                .process_direct(
                    &format!("message {}", i),
                    "trio-offline-cb",
                    "test",
                    "offline",
                )
                .await;
            eprintln!(
                "  cascade msg {}: ({} chars) {}",
                i,
                resp.len(),
                &resp[..resp.len().min(80)]
            );
        }

        let counters = &agent_loop.shared.core_handle.counters;

        // After repeated failures the trio state should be Degraded.
        let state = counters.get_trio_state();
        eprintln!("trio_state after cascade: {:?}", state);
        assert_eq!(
            state,
            crate::agent::agent_core::TrioState::Degraded,
            "trio_state should be Degraded after repeated router failures"
        );

        // Verify CB key alignment after the fix.
        //
        // The offline harness returns mock responses that fail strict AND lenient
        // parsing (lenient no longer defaults to phantom "clarify" target — it
        // returns None when no target can be extracted). Each parse failure records
        // a CB failure, so after 4 turns the CB should be tripped.
        //
        // The shared CB key format ("router:{model}") ensures that the
        // tool-stripping guard in step_pre_call and the routing skip in
        // router_preflight observe the same state.
        let cb_correct_key_available = counters
            .trio_circuit_breaker
            .lock()
            .unwrap()
            .is_available("router:offline-router");
        eprintln!(
            "CB 'router:offline-router' available after 4 turns: {}",
            cb_correct_key_available
        );
        // Parse failures are now correctly recorded — CB should be tripped.
        assert!(
            !cb_correct_key_available,
            "CB 'router:offline-router' should be tripped: parse failures are now recorded"
        );
        // The legacy key "trio_router" is also untouched.
        let cb_legacy_key_available = counters
            .trio_circuit_breaker
            .lock()
            .unwrap()
            .is_available("trio_router");
        assert!(
            cb_legacy_key_available,
            "CB 'trio_router' should be untouched — agent_loop now uses 'router:{{model}}' key"
        );

        let _ = std::fs::remove_dir_all(&workspace);
    }

    // -----------------------------------------------------------------------
    // Test 4: health gate — degraded router probe bypasses preflight
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_trio_offline_e2e_health_gate() {
        use crate::heartbeat::health::{HealthProbe, HealthRegistry, ProbeResult};
        use crate::config::schema::LcmSchemaConfig;

        // A mock probe that always returns unhealthy (simulates router being down).
        struct AlwaysUnhealthyProbe;

        #[async_trait]
        impl HealthProbe for AlwaysUnhealthyProbe {
            fn name(&self) -> &str {
                "trio_router"
            }

            fn interval_secs(&self) -> u64 {
                0 // always due
            }

            async fn check(&self) -> ProbeResult {
                ProbeResult {
                    healthy: false,
                    latency_ms: 0,
                    detail: Some("simulated failure".to_string()),
                }
            }
        }

        // Build a registry and degrade the trio_router probe.
        let mut health_registry = HealthRegistry::new();
        health_registry.register(Box::new(AlwaysUnhealthyProbe));
        // Run 3 times to reach DEGRADED_THRESHOLD = 3.
        for _ in 0..3 {
            health_registry.run_due_probes().await;
        }
        assert!(
            !health_registry.is_healthy("trio_router"),
            "trio_router should be degraded after 3 failures"
        );
        let health_registry = Arc::new(health_registry);

        // The router SequenceProvider would fail the test if called (empty queue).
        // We keep a typed Arc so we can read call_count() after the run.
        let router_seq = Arc::new(SequenceProvider::new(
            "offline-router",
            vec![], // empty — calling this would return the sentinel error
        ));
        let router: Arc<dyn LLMProvider> = router_seq.clone();
        let main: Arc<dyn LLMProvider> =
            Arc::new(StaticResponseLLM::new("offline-main", "main answer"));
        let specialist: Arc<dyn LLMProvider> =
            Arc::new(StaticResponseLLM::new("offline-specialist", "specialist unused"));

        // Build harness manually so we can wire in the health registry.
        let workspace = tempfile::tempdir().unwrap().into_path();
        let mut td = ToolDelegationConfig {
            mode: crate::config::schema::DelegationMode::Trio,
            ..Default::default()
        };
        td.apply_mode();

        let router_model = router.get_default_model().to_string();
        let specialist_model = specialist.get_default_model().to_string();
        let trio_config = TrioConfig {
            enabled: true,
            router_model: router_model.clone(),
            specialist_model: specialist_model.clone(),
            ..Default::default()
        };

        let core = build_swappable_core(SwappableCoreConfig {
            provider: main,
            workspace: workspace.clone(),
            model: "offline-main".to_string(),
            max_iterations: 5,
            max_tokens: 512,
            temperature: 0.3,
            max_context_tokens: 4096,
            brave_api_key: None,
            exec_timeout: 30,
            restrict_to_workspace: true,
            memory_config: MemoryConfig::default(),
            is_local: true,
            compaction_provider: None,
            tool_delegation: td,
            provenance: ProvenanceConfig::default(),
            max_tool_result_chars: 2000,
            delegation_provider: Some(router.clone()),
            specialist_provider: Some(specialist),
            trio_config,
            model_capabilities_overrides: std::collections::HashMap::new(),
        });

        let counters = Arc::new(crate::agent::agent_core::RuntimeCounters::new(4096));
        let core_handle = AgentHandle::new(core, counters);

        let (inbound_tx, inbound_rx) =
            tokio::sync::mpsc::unbounded_channel::<InboundMessage>();
        let (outbound_tx, _outbound_rx) =
            tokio::sync::mpsc::unbounded_channel::<OutboundMessage>();

        let agent_loop = AgentLoop::new(
            core_handle,
            inbound_rx,
            outbound_tx,
            inbound_tx,
            None,
            1,
            None,
            None,
            None,
            ProprioceptionConfig::default(),
            LcmSchemaConfig::default(),
            Some(health_registry), // health registry is wired in here
        );

        let resp = agent_loop
            .process_direct(
                "Hello",
                "trio-offline-health-gate",
                "test",
                "offline",
            )
            .await;

        eprintln!(
            "test_trio_offline_e2e_health_gate: response ({} chars): {}",
            resp.len(),
            &resp[..resp.len().min(200)]
        );

        // When the health gate fires, router_preflight returns Passthrough and sets Degraded.
        let state = agent_loop
            .shared
            .core_handle
            .counters
            .get_trio_state();
        eprintln!("trio_state after health gate: {:?}", state);
        assert_eq!(
            state,
            crate::agent::agent_core::TrioState::Degraded,
            "trio_state should be Degraded when health gate fires"
        );

        // Response must come from main (non-empty).
        assert!(!resp.is_empty(), "response should come from main, not be empty");

        // router_preflight_fired should be true (we entered preflight but returned Passthrough).
        let metrics = &agent_loop.shared.core_handle.counters.trio_metrics;
        assert!(
            metrics.router_preflight_fired.load(std::sync::atomic::Ordering::Relaxed),
            "router_preflight_fired should be true (preflight was entered)"
        );

        // Specialist must not have been dispatched.
        assert!(
            !metrics.specialist_dispatched.load(std::sync::atomic::Ordering::Relaxed),
            "specialist should not be dispatched when health gate is active"
        );

        // Router's chat() should never have been called — health gate fired before it.
        assert_eq!(
            router_seq.call_count(),
            0,
            "router provider's chat() call count should be 0 (health gate bypassed it)"
        );

        let _ = std::fs::remove_dir_all(&workspace);
    }

    // -----------------------------------------------------------------------
    // Test 5: lenient parse fallback
    //
    // Router returns FunctionGemma comma-separated format:
    //   "specialist,coding,{}"
    // `parse_lenient_router_decision` handles this format.
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_trio_offline_e2e_parse_fallback_lenient() {
        // Lenient format: "action,target,{args}" — no JSON wrapper.
        // This exercises the comma-separated branch in parse_lenient_router_decision.
        let router_resp = "specialist,coding,{}";

        let router: Arc<dyn LLMProvider> = Arc::new(SequenceProvider::new(
            "offline-router",
            vec![router_resp, router_resp, router_resp],
        ));
        let main: Arc<dyn LLMProvider> =
            Arc::new(StaticResponseLLM::new("offline-main", "delegating"));
        let specialist: Arc<dyn LLMProvider> = Arc::new(StaticResponseLLM::new(
            "offline-specialist",
            "lenient parse worked",
        ));

        // Verify that parse_lenient_router_decision handles this format before
        // wiring it into the full agent loop.
        let lenient_decision = parse_lenient_router_decision(router_resp);
        assert!(
            lenient_decision.is_some(),
            "parse_lenient_router_decision should accept 'specialist,coding,{{}}'"
        );
        let lenient_decision = lenient_decision.unwrap();
        assert_eq!(
            lenient_decision.action, "specialist",
            "lenient decision action should be 'specialist'"
        );

        let (agent_loop, workspace) =
            build_trio_offline_harness(main, router, specialist);

        let resp = agent_loop
            .process_direct(
                "Explain something complex",
                "trio-offline-lenient",
                "test",
                "offline",
            )
            .await;

        eprintln!(
            "test_trio_offline_e2e_parse_fallback_lenient: response ({} chars): {}",
            resp.len(),
            &resp[..resp.len().min(200)]
        );

        let metrics = &agent_loop.shared.core_handle.counters.trio_metrics;

        assert_eq!(
            metrics.router_action.lock().unwrap().as_deref(),
            Some("specialist"),
            "router_action should be 'specialist' after lenient parse"
        );
        assert!(
            metrics.specialist_dispatched.load(std::sync::atomic::Ordering::Relaxed),
            "specialist should have been dispatched after lenient parse"
        );
        assert!(!resp.is_empty(), "response should be non-empty");

        let _ = std::fs::remove_dir_all(&workspace);
    }
}
