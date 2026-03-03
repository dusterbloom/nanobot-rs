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
use crate::agent::reasoning::{BranchAttempt, ReasoningEngine, ReasoningMode, StepStatus};
use crate::agent::tools::reasoning_tools::SharedEngine;
use crate::errors::is_retryable_provider_error;
use crate::agent::anti_drift;
use crate::agent::context_hygiene;
use crate::agent::policy;
use crate::agent::protocol::{
    parse_textual_tool_calls, strip_textual_tool_calls, CloudProtocol, ConversationProtocol,
    LocalProtocol,
};
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
use crate::config::schema::{AdaptiveTokenConfig, EmailConfig, LcmSchemaConfig, ProprioceptionConfig};
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
    /// Cluster router for distributed inference (feature-gated).
    #[cfg(feature = "cluster")]
    pub(crate) cluster_router: Option<Arc<crate::cluster::router::ClusterRouter>>,
    /// Knowledge store for proactive grounding retrieval.
    pub(crate) knowledge_store: Option<Arc<std::sync::Mutex<crate::agent::knowledge_store::KnowledgeStore>>>,
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
    pub(crate) session_id: String,
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
    /// Bug 5 fix: after a compaction swap, ctx.messages shrinks but the LCM
    /// engine's store_len still reflects the pre-compaction count. This field
    /// overrides the skip offset used in step_pre_call's ingestion loop so
    /// that messages are re-ingested from the correct position after a swap.
    pub(crate) lcm_synced_to: Option<usize>,

    // --- Observability ---
    pub(crate) counters: Arc<RuntimeCounters>,

    // --- Flow control ---
    pub(crate) flow: FlowControl,

    // --- Health ---
    pub(crate) health_registry: Option<Arc<crate::heartbeat::health::HealthRegistry>>,

    // --- Security ---
    /// Tracks taint introduced by web tools; used to warn before sensitive tool calls.
    pub(crate) taint_state: crate::agent::taint::TaintState,

    // --- Reasoning ---
    /// Shared reasoning engine for plan-guided execution and backtracking.
    pub(crate) reasoning: SharedEngine,
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
    /// Whether the agent-level retry for transient LLM errors has already been used
    /// in this turn. Limits retries to one per iteration to avoid infinite loops.
    pub(crate) agent_retry_attempted: bool,
    /// Number of auto-continuations used this turn for truncated responses.
    pub(crate) continuations_used: u32,
    /// Consecutive validation retries for the current main-loop iteration.
    /// Reset to 0 on each successful (non-validation) iteration. Capped at
    /// `MAX_VALIDATION_RETRIES` before the retry is treated as a normal
    /// continuation (i.e. consumes an iteration slot).
    pub(crate) validation_retries: u32,
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
    /// Validation failed and a retry hint was injected. Does NOT consume a
    /// main-loop iteration slot — the outer loop re-runs the same iteration.
    ValidationRetry,
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

        // Bug 3 fix: eagerly persist the user message before the LLM call so
        // it is not lost if the agent crashes mid-turn. Bump new_start so
        // finalize_response does not double-persist it.
        if ctx.new_start < ctx.messages.len() {
            let user_msg = ctx.messages[ctx.new_start].clone();
            ctx.core.sessions
                .add_message(&ctx.session_id, &user_msg)
                .await;
            ctx.new_start += 1;
        }

        self.run_agent_loop(&mut ctx).await;
        self.finalize_response(ctx).await
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
        // Auto-decompose: detect numbered steps in user message and build a plan.
        // This helps small models that can't call the plan tool themselves.
        if ctx.core.reasoning_config.enabled && ctx.core.reasoning_config.auto_decompose {
            if let Ok(engine) = ctx.reasoning.lock() {
                // Only auto-decompose if no plan exists yet (Linear mode).
                if *engine.mode() == ReasoningMode::Linear {
                    drop(engine); // Release lock before re-acquiring mutably
                    if let Some(steps) = crate::agent::reasoning::parse_numbered_steps(&ctx.user_content) {
                        let step_budget = ctx.core.reasoning_config.step_budget;
                        let new_engine = ReasoningEngine::from_goals(&steps, step_budget);
                        if let Ok(mut engine) = ctx.reasoning.lock() {
                            *engine = new_engine;
                            info!(
                                steps = steps.len(),
                                first = %steps[0],
                                "auto_decompose: parsed numbered steps from user message"
                            );
                        }
                    }
                }
            }
        }

        // `iteration` counts only "real" (non-validation-retry) iterations so
        // that format-correction retries don't eat into the main budget.
        let mut iteration: u32 = 0;
        // Nudge the model to wrap up before it hits the hard iteration cap.
        // Trigger at 80% of the budget (ceiling), sent only once.
        let nudge_at = ((ctx.core.max_iterations as f64) * 0.8).ceil() as u32;
        let mut nudge_sent = false;
        while iteration < ctx.core.max_iterations {
            // Early exit if cancelled (e.g. user pressed Esc/Enter in REPL).
            if ctx.cancellation_token.as_ref().map_or(false, |t| t.is_cancelled()) {
                debug!("agent loop: cancelled before iteration {}", iteration);
                break;
            }

            // Nudge the model when approaching the iteration budget.
            if iteration == nudge_at && !nudge_sent {
                nudge_sent = true;
                let remaining = ctx.core.max_iterations - iteration;
                let nudge_msg = format!(
                    "[System notice] You have {} iteration(s) remaining. Produce your final answer now.",
                    remaining
                );
                ctx.messages.push(serde_json::json!({
                    "role": "user",
                    "content": nudge_msg
                }));
                info!(
                    "iteration_nudge: injected wrap-up nudge at iteration {}/{}",
                    iteration, ctx.core.max_iterations
                );
            }

            // Plan-guided: inject current step instruction into conversation.
            {
                if let Ok(engine) = ctx.reasoning.lock() {
                    if let Some(instruction) = engine.step_instruction() {
                        ctx.messages.push(json!({
                            "role": "user",
                            "content": format!("[Current objective] {}", instruction),
                            "_synthetic": true,
                        }));
                    }
                }
            }

            debug!(
                "Agent iteration{} {}/{} (validation_retries={})",
                if ctx.streaming { " (streaming)" } else { "" },
                iteration + 1,
                ctx.core.max_iterations,
                ctx.flow.validation_retries
            );

            // Sync messages to reasoning engine so CheckpointTool can capture them.
            if let Ok(mut engine) = ctx.reasoning.lock() {
                engine.sync_messages(&ctx.messages);
            }

            ctx.iterations_used = iteration + 1;
            let outcome = self.run_iteration(ctx, iteration).await;

            // Check for pending backtrack (set by BacktrackTool during tool execution).
            {
                if let Ok(mut engine) = ctx.reasoning.lock() {
                    if let Some(restored) = engine.take_pending_restore() {
                        ctx.messages = restored;
                        iteration += 1;
                        ctx.flow.validation_retries = 0;
                        continue;
                    }
                }
            }

            match outcome {
                IterationOutcome::ValidationRetry => {
                    // A validation error injected a corrective hint. Only count
                    // this against the validation budget, not the main iteration
                    // budget, so format corrections don't exhaust real work slots.
                    ctx.flow.validation_retries += 1;
                    if ctx.flow.validation_retries >= validation::MAX_VALIDATION_RETRIES as u32 {
                        // Exhausted validation retries — treat as a normal
                        // iteration so the loop can make forward progress.
                        warn!(
                            "validation retries exhausted ({}/{}), counting as real iteration",
                            ctx.flow.validation_retries,
                            validation::MAX_VALIDATION_RETRIES,
                        );
                        ctx.flow.validation_retries = 0;
                        iteration += 1;
                    } else {
                        debug!(
                            "validation retry {}/{} — not counting against main budget",
                            ctx.flow.validation_retries,
                            validation::MAX_VALIDATION_RETRIES,
                        );
                        // Do NOT increment `iteration` — re-run the same slot.
                    }
                    continue;
                }
                IterationOutcome::Continue => {
                    // Successful (non-validation) iteration — reset retry counter
                    // and advance to the next main-budget slot.
                    ctx.flow.validation_retries = 0;
                    iteration += 1;
                    // Consume step budget if plan-guided.
                    if let Ok(mut engine) = ctx.reasoning.lock() {
                        engine.consume_iteration();
                        if *engine.mode() != ReasoningMode::Linear
                            && engine.step_budget_remaining() == 0
                        {
                            engine.mark_current_failed("iteration budget exhausted");
                            if let Some(cp) = engine.pop_checkpoint() {
                                drop(engine);
                                ctx.messages = cp.messages;
                                continue;
                            }
                        }
                    }
                    continue;
                }
                IterationOutcome::Finished(content) => {
                    ctx.flow.validation_retries = 0;
                    iteration += 1;
                    // In plan-guided mode, advance to next step.
                    let should_continue = {
                        if let Ok(mut engine) = ctx.reasoning.lock() {
                            if *engine.mode() != ReasoningMode::Linear {
                                engine.mark_current_completed(Some(content.clone()));
                                engine.advance();
                                !engine.is_complete()
                            } else {
                                false
                            }
                        } else {
                            false
                        }
                    };
                    if should_continue {
                        // More plan steps to execute — don't break.
                        continue;
                    }
                    ctx.final_content = content;
                    break;
                }
                IterationOutcome::Error(msg) => {
                    ctx.flow.validation_retries = 0;
                    iteration += 1;
                    // Try backtracking before giving up.
                    let should_backtrack = {
                        if let Ok(mut engine) = ctx.reasoning.lock() {
                            if *engine.mode() != ReasoningMode::Linear {
                                engine.mark_current_failed(&msg);
                                if engine.find_alternative().is_some() {
                                    if let Some(cp) = engine.pop_checkpoint() {
                                        engine.record_branch(BranchAttempt {
                                            step_id: 0,
                                            approach: "previous".into(),
                                            outcome: StepStatus::Failed(msg.clone()),
                                            iterations_consumed: iteration,
                                        });
                                        Some(cp.messages)
                                    } else {
                                        None
                                    }
                                } else {
                                    None
                                }
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    };
                    if let Some(restored) = should_backtrack {
                        ctx.messages = restored;
                        continue;
                    }
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
                // Bug 5 fix: after compaction ctx.messages shrinks but the LCM
                // engine's store_len reflects the old count. Override the
                // skip offset so step_pre_call ingests from index 0 (i.e. all
                // messages in the new shorter array) instead of skipping past
                // the end.
                ctx.lcm_synced_to = Some(0);
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
                "content": format!("[system] Acknowledged.{budget_note}")
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
            // Bug 5 fix: after a compaction swap ctx.messages shrinks but
            // store_len still reflects the pre-compaction count, causing the
            // loop to skip everything. Use lcm_synced_to (set to 0 after swap)
            // as the skip offset when it is present.
            {
                let mut engine = lcm_engine.lock().await;
                let store_len = engine.store_len();
                let skip = ctx.lcm_synced_to.unwrap_or(store_len);
                // After using the override, reset it so future iterations use
                // the normal store_len path.
                ctx.lcm_synced_to = None;
                for msg in ctx.messages.iter().skip(skip) {
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
                        let bg_session_id = ctx.session_id.clone();
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
                                            bg_core.sessions.add_message(&bg_session_id, &summary_json).await;
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

        // Proactive grounding: inject relevant knowledge before LLM call.
        if self.proprioception_config.proactive_retrieval && iteration == 0 {
            if let Some(user_text) = last_user_message(&ctx.messages) {
                if !user_text.is_empty() {
                    let intent = crate::agent::proactive::extract_intent(&user_text);
                    if intent.confidence >= 0.2 {
                        let budget = (ctx.core.token_budget.max_context() / 20).min(500);
                        let learning_context = ctx.core.learning.get_learning_context();
                        let ks_guard = self.knowledge_store.as_ref().map(|ks| ks.lock().unwrap());
                        let ks_ref = ks_guard.as_deref();
                        let payload = crate::agent::proactive::retrieve_grounding(
                            &intent, ks_ref, &learning_context, budget,
                        );
                        if let Some(text) = crate::agent::proactive::format_grounding_message(&payload) {
                            debug!(
                                category = ?intent.category,
                                confidence = intent.confidence,
                                snippets = payload.knowledge_snippets.len(),
                                estimated_tokens = payload.estimated_tokens,
                                "proactive_grounding_injected"
                            );
                            ctx.messages.push(serde_json::json!({
                                "role": if ctx.core.is_local { "user" } else { "system" },
                                "content": text,
                                "_synthetic": true,
                            }));
                        }
                    }
                }
            }
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
            let user_text = ctx.messages
                .last()
                .and_then(|m| m.get("content"))
                .and_then(|c| c.as_str())
                .unwrap_or("");
            // Count recent tool calls: if tool-heavy, use smaller budget.
            let recent_tool_calls = ctx.messages
                .iter()
                .rev()
                .take(6)
                .filter(|m| m.get("role").and_then(|r| r.as_str()) == Some("tool"))
                .count();
            let thinking_budget = {
                let stored = counters.thinking_budget.load(Ordering::Relaxed);
                if stored > 0 {
                    Some(stored)
                } else {
                    None
                }
            };
            adaptive_max_tokens(
                base,
                had_long,
                user_text,
                recent_tool_calls,
                ctx.core.is_local,
                thinking_budget,
                &ctx.core.adaptive_tokens,
            )
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
                    Some(stored.min(ctx.core.adaptive_tokens.local_thinking_small_model_cap))
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
                    if !ctx.flow.agent_retry_attempted && is_retryable_provider_error(&e) {
                        ctx.flow.agent_retry_attempted = true;
                        warn!(model = %ctx.core.model, error = %e, "llm_stream_call_failed_retrying");
                        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                        return StepResult::Done(IterationOutcome::Continue);
                    }
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
                        debug!("streaming cancelled by user");
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
                    if !ctx.flow.agent_retry_attempted && is_retryable_provider_error(&e) {
                        ctx.flow.agent_retry_attempted = true;
                        warn!(model = %ctx.core.model, error = %e, "llm_call_failed_retrying");
                        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                        return StepResult::Done(IterationOutcome::Continue);
                    }
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

        // --- Universal textual tool-call extraction ---
        // If the response has no native tool_calls but contains [I called: ...]
        // patterns, parse them regardless of protocol mode. This prevents
        // the validation step from flagging them as hallucinated tool calls.
        if !response.has_tool_calls()
            && response
                .content
                .as_ref()
                .map(|c| !c.trim().is_empty())
                .unwrap_or(false)
        {
            let content_text = response.content.as_deref().unwrap_or("");
            let parsed = parse_textual_tool_calls(content_text);
            if !parsed.is_empty() {
                debug!(
                    n = parsed.len(),
                    "universal_textual_parse: parsed {} tool call(s) from response text",
                    parsed.len()
                );
                let synthesised: Vec<crate::providers::base::ToolCallRequest> = parsed
                    .into_iter()
                    .enumerate()
                    .map(|(i, ptc)| {
                        let args: HashMap<String, Value> = match ptc.args {
                            Value::Object(map) => map.into_iter().collect(),
                            _ => HashMap::new(),
                        };
                        crate::providers::base::ToolCallRequest {
                            id: format!("tc_textual_{}", i + 1),
                            name: ptc.tool,
                            arguments: args,
                        }
                    })
                    .collect();
                if let Some(ref mut content) = response.content {
                    *content = strip_textual_tool_calls(content);
                }
                response.tool_calls = synthesised;
            }
        }

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

        match validation::validate_response(content_str, &tool_calls_as_maps, ctx.protocol.is_textual_replay()) {
            validation::ValidationOutcome::Error(validation_err) => {
                let retry_num = ctx.flow.validation_retries + 1;
                warn!(
                    model = %ctx.core.model,
                    validation = %format!("{:?}", validation_err),
                    retry = retry_num,
                    max_retries = validation::MAX_VALIDATION_RETRIES,
                    "response_validation_failed"
                );
                let hint = validation::generate_retry_prompt(
                    &validation_err,
                    retry_num as u8,
                );
                ctx.messages.push(json!({
                    "role": "assistant",
                    "content": content_str
                }));
                ctx.messages.push(json!({
                    "role": "user",
                    "content": hint
                }));
                debug!("Injected validation retry hint (retry {}/{})", retry_num, validation::MAX_VALIDATION_RETRIES);
                return StepResult::Done(IterationOutcome::ValidationRetry);
            }
            validation::ValidationOutcome::StripHallucination => {
                debug!("Stripping hallucinated tool-call text from response");
                if let Some(ref mut content) = response.content {
                    *content = validation::strip_hallucinated_text(content);
                }
            }
            validation::ValidationOutcome::Ok => {}
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

        // --- Auto-continue: stitch truncated non-empty responses ---
        let max_cont = ctx.core.max_continuations;
        while !response.has_tool_calls()
            && response
                .content
                .as_ref()
                .map(|s| !s.trim().is_empty())
                .unwrap_or(false)
            && (response.finish_reason == "length"
                || (response.finish_reason == "stop"
                    && response.content.as_ref().map(|s| appears_incomplete(s)).unwrap_or(false)))
            && ctx.flow.continuations_used < max_cont
        {
            ctx.flow.continuations_used += 1;
            if response.finish_reason == "stop" {
                info!("auto_continue: heuristic detected incomplete response despite finish_reason='stop'");
            }
            info!(
                "auto_continue: continuation {}/{} — finish_reason was '{}'",
                ctx.flow.continuations_used, max_cont, response.finish_reason
            );

            // Streaming indicator
            if let Some(ref tx) = ctx.text_delta_tx {
                let _ = tx.send("\x1b[2m [continuing...]\x1b[0m".to_string());
            }

            // Build continuation messages: original context + partial as assistant + "Continue."
            let mut cont_messages = ctx.messages.clone();
            cont_messages.push(json!({
                "role": "assistant",
                "content": response.content.as_deref().unwrap_or("")
            }));
            cont_messages.push(json!({
                "role": "user",
                "content": "Continue."
            }));

            // Check cancellation before LLM call
            if ctx.cancellation_token.as_ref().map_or(false, |t| t.is_cancelled()) {
                break;
            }

            counters.inference_active.store(true, Ordering::Relaxed);
            let cont_result = ctx.core
                .provider
                .chat(
                    &cont_messages,
                    None,           // no tools during continuation
                    Some(&ctx.core.model),
                    ctx.core.max_tokens,
                    ctx.core.temperature,
                    None,           // no thinking budget
                    None,
                )
                .await;
            counters.inference_active.store(false, Ordering::Relaxed);

            match cont_result {
                Ok(cont_response) => {
                    // Stream continuation content
                    if let Some(ref new_text) = cont_response.content {
                        if let Some(ref tx) = ctx.text_delta_tx {
                            let _ = tx.send(new_text.clone());
                        }
                    }

                    // Stitch content
                    let original = response.content.take().unwrap_or_default();
                    let continuation = cont_response.content.unwrap_or_default();
                    response.content = Some(format!("{}{}", original, continuation));

                    // Update finish_reason (enables next iteration if still "length")
                    response.finish_reason = cont_response.finish_reason;

                    // Merge usage
                    for (key, val) in cont_response.usage {
                        *response.usage.entry(key).or_insert(0) += val;
                    }
                }
                Err(e) => {
                    warn!("auto_continue: continuation call failed: {}", e);
                    break;
                }
            }
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
                    if !crate::server::check_health(base, ctx.core.health_check_timeout_secs).await {
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
            // Send finish_reason metadata to the REPL renderer before closing the channel.
            if let Some(ref tx) = ctx.text_delta_tx {
                let _ = tx.send(format!("\x00finish_reason:{}", response.finish_reason));
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

        // Auto-checkpoint before risky tools (exec, write_file) when enabled.
        if ctx.core.reasoning_config.auto_checkpoint_before_exec {
            let should_checkpoint = routed_tool_calls.iter().any(|tc| {
                tc.name == "exec" || tc.name == "write_file"
            });
            if should_checkpoint {
                if let Ok(mut engine) = ctx.reasoning.lock() {
                    if *engine.mode() != crate::agent::reasoning::ReasoningMode::Linear {
                        engine.sync_messages(&ctx.messages);
                        engine.save_checkpoint("pre_exec", &ctx.messages, ctx.iterations_used);
                    }
                }
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
fn last_user_message(messages: &[serde_json::Value]) -> Option<String> {
    messages.iter().rev()
        .find(|m| m.get("role").and_then(|r| r.as_str()) == Some("user"))
        .and_then(|m| m.get("content").and_then(|c| c.as_str()))
        .map(|s| s.to_string())
}

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

const ADAPTIVE_TOOL_HEAVY_WINDOW_THRESHOLD: usize = 3;

fn adaptive_max_tokens(
    base: u32,
    had_long: bool,
    user_text: &str,
    recent_tool_calls: usize,
    is_local: bool,
    thinking_budget: Option<u32>,
    cfg: &AdaptiveTokenConfig,
) -> u32 {
    let mut effective = if had_long {
        base.max(cfg.adaptive_long_mode_min_tokens)
    } else {
        let lower = user_text.to_lowercase();
        let is_long_form = lower.contains("explain in detail")
            || lower.contains("write a ")
            || lower.contains("create a script")
            || lower.contains("write code")
            || lower.contains("implement ")
            || lower.contains("full example")
            || lower.starts_with("write ")
            || user_text.len() > cfg.adaptive_long_form_trigger_chars as usize;

        if is_long_form {
            base.max(cfg.adaptive_long_form_min_tokens)
        } else if recent_tool_calls > ADAPTIVE_TOOL_HEAVY_WINDOW_THRESHOLD {
            base.min(cfg.adaptive_tool_heavy_max_tokens)
                .max(cfg.adaptive_tool_heavy_min_tokens)
        } else {
            base
        }
    };

    if is_local && thinking_budget.is_some() {
        effective = effective
            .saturating_sub(cfg.local_thinking_reserve_tokens)
            .max(cfg.local_thinking_min_completion_tokens);
    }

    effective
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
                        retry: crate::config::schema::RetryConfig::default(),
                        timeout_secs: 120,
                        lms_native_probe_secs: 2,
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
            #[cfg(feature = "cluster")]
            cluster_router: None,
            knowledge_store: crate::agent::knowledge_store::KnowledgeStore::open_default().ok()
                .map(|ks| Arc::new(std::sync::Mutex::new(ks))),
        });

        Self {
            shared,
            bus_inbound_rx,
            running: Arc::new(AtomicBool::new(false)),
            max_concurrent_chats,
            reflection_spawned: AtomicBool::new(false),
        }
    }

    /// Set the cluster router for distributed inference routing.
    ///
    /// Must be called before `run()` or `process_direct()` to take effect.
    #[cfg(feature = "cluster")]
    pub fn set_cluster_router(&mut self, router: Arc<crate::cluster::router::ClusterRouter>) {
        // SAFETY: we hold &mut self so no concurrent access exists yet.
        let shared = Arc::get_mut(&mut self.shared)
            .expect("set_cluster_router called after shared Arc was cloned");
        shared.cluster_router = Some(router.clone());
        // Also pass the router down to the subagent manager.
        let subagents = Arc::get_mut(&mut shared.subagents)
            .expect("set_cluster_router: subagents Arc already shared");
        subagents.cluster_router = Some(router);
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

                // For Telegram: set up streaming with typing indicator + progressive edits.
                let (stream_tx, stream_is_telegram) = if msg.channel == "telegram" {
                    let bot_token = msg
                        .metadata
                        .get("bot_token")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let chat_id_str = msg.chat_id.clone();
                    if !bot_token.is_empty() {
                        let chat_id_num: i64 = chat_id_str.parse().unwrap_or(0);
                        let (delta_tx, mut delta_rx) =
                            tokio::sync::mpsc::unbounded_channel::<String>();
                        let stream_client = reqwest::Client::new();
                        let stream_token = bot_token.clone();
                        tokio::spawn(async move {
                            crate::channels::telegram::tg_send_typing_action(
                                &stream_client,
                                &stream_token,
                                chat_id_num,
                            )
                            .await;
                            let msg_id = crate::channels::telegram::tg_send_placeholder(
                                &stream_client,
                                &stream_token,
                                chat_id_num,
                            )
                            .await;
                            let Some(message_id) = msg_id else {
                                while delta_rx.recv().await.is_some() {}
                                return;
                            };
                            let mut accumulated = String::new();
                            let mut dirty = false;
                            let mut interval =
                                tokio::time::interval(std::time::Duration::from_millis(500));
                            interval.set_missed_tick_behavior(
                                tokio::time::MissedTickBehavior::Skip,
                            );
                            loop {
                                tokio::select! {
                                    delta = delta_rx.recv() => {
                                        match delta {
                                            Some(chunk) => {
                                                accumulated.push_str(&chunk);
                                                dirty = true;
                                            }
                                            None => {
                                                if dirty && !accumulated.is_empty() {
                                                    crate::channels::telegram::tg_edit_message(
                                                        &stream_client,
                                                        &stream_token,
                                                        chat_id_num,
                                                        message_id,
                                                        &accumulated,
                                                    )
                                                    .await;
                                                }
                                                break;
                                            }
                                        }
                                    }
                                    _ = interval.tick() => {
                                        if dirty && !accumulated.is_empty() {
                                            crate::channels::telegram::tg_edit_message(
                                                &stream_client,
                                                &stream_token,
                                                chat_id_num,
                                                message_id,
                                                &accumulated,
                                            )
                                            .await;
                                            dirty = false;
                                        }
                                    }
                                }
                            }
                        });
                        (Some(delta_tx), true)
                    } else {
                        (None, false)
                    }
                } else {
                    (None, false)
                };

                let response =
                    shared.process_message(&msg, stream_tx, None, None, None).await;

                let outbound = match response {
                    Some(mut outbound) => {
                        if stream_is_telegram {
                            outbound.metadata.insert(
                                "streaming_handled".to_string(),
                                serde_json::json!(true),
                            );
                        }
                        outbound
                    }
                    None => {
                        error!(
                            channel = %msg.channel,
                            chat_id = %msg.chat_id,
                            "process_message returned None; sending error feedback to user"
                        );
                        crate::bus::events::OutboundMessage::new(
                            &msg.channel,
                            &msg.chat_id,
                            "[nanobot] Sorry, I encountered an error processing your message. Please try again.",
                        )
                    }
                };

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

                drop(permit); // release concurrency slot
            });
        }

        info!("Agent loop stopped");
    }

    /// Return a handle to the subagent manager.
    pub fn subagent_manager(&self) -> Arc<SubagentManager> {
        self.shared.subagents.clone()
    }

    /// Clear the LCM engine for a session (e.g. on /clear command).
    ///
    /// This resets the summary DAG and active context so stale summaries
    /// don't pollute fresh conversations after /clear.
    pub async fn clear_lcm_engine(&self, session_key: &str) {
        let mut engines = self.shared.lcm_engines.lock().await;
        if engines.remove(session_key).is_some() {
            debug!(session = %session_key, "LCM engine cleared");
        }
    }

    /// Clear the bulletin cache (e.g. on /clear command).
    pub fn clear_bulletin_cache(&self) {
        self.shared.bulletin_cache.store(Arc::new(String::new()));
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
// Heuristic helpers
// ============================================================================

/// Detect responses that appear truncated despite finish_reason being "stop".
///
/// This catches cases where the model stops at special characters (e.g., backtick)
/// due to tokenizer/stop-token issues in local model servers.
pub(crate) fn appears_incomplete(content: &str) -> bool {
    let trimmed = content.trim_end();
    if trimmed.is_empty() {
        return false;
    }

    // Ends mid-sentence (no terminal punctuation, not a code block fence).
    // Strip trailing emoji (non-ASCII symbols like 🦀 😄 🔁) and any surrounding
    // whitespace before checking the "real" last character — an emoji after a
    // period must not trigger continuation.
    let stripped = trimmed.trim_end_matches(|c: char| !c.is_alphanumeric() && !c.is_ascii()).trim_end();
    let text_for_check = if stripped.is_empty() { trimmed } else { stripped };
    let last_char = text_for_check.chars().last().unwrap();
    let ends_mid_sentence = !matches!(last_char, '.' | '!' | '?' | ':' | '"' | '\'' | ')' | ']' | '}' | '`')
        && !trimmed.ends_with("```");

    // Has unclosed backtick (odd number of backticks on the last line).
    // Exclude code fences (lines that are purely backticks, e.g. "```") —
    // those are block delimiters, not inline code markers.
    let last_line = trimmed.lines().last().unwrap_or("");
    let last_line_trimmed = last_line.trim();
    let is_code_fence = !last_line_trimmed.is_empty()
        && last_line_trimmed.chars().all(|c| c == '`')
        && last_line_trimmed.len() >= 3;
    let backtick_count = last_line.chars().filter(|&c| c == '`').count();
    let unclosed_backtick = !is_code_fence && backtick_count % 2 != 0;

    // Has unclosed parenthesis/bracket on the last line
    let unclosed_paren = last_line.chars().filter(|&c| c == '(').count()
        > last_line.chars().filter(|&c| c == ')').count();

    unclosed_backtick || (ends_mid_sentence && trimmed.len() > 20) || unclosed_paren
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[path = "agent_loop_tests.rs"]
mod tests;
