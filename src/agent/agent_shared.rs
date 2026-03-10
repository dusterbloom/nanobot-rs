//! `AgentLoopShared` struct, supporting types, and the `impl AgentLoopShared` block
//! containing the main agent loop step methods.
//!
//! Extracted from `agent_loop.rs` as a `#[path]` submodule.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use serde_json::{json, Value};
use tokio::sync::mpsc::UnboundedSender;
use tokio::sync::Mutex;
use tracing::{debug, error, info, instrument, warn};

use crate::agent::anti_drift;
use crate::agent::audit::{AuditLog, ToolEvent};
use crate::agent::compaction::ContextCompactor;
use crate::agent::context_hygiene;
use crate::agent::lcm::{CompactionAction, LcmConfig, LcmEngine};
use crate::agent::policy;
use crate::agent::protocol::{ConversationProtocol, XmlToolCallFilter};
use crate::agent::reasoning::{BranchAttempt, ReasoningEngine, ReasoningMode, StepStatus};
use crate::agent::subagent::SubagentManager;
use crate::agent::system_state::{self, AhaPriority, AhaSignal, SystemState};
use crate::agent::token_budget::TokenBudget;
use crate::agent::tool_gate::ToolGate;
use crate::agent::tool_guard::ToolGuard;
use crate::agent::tools::reasoning_tools::SharedEngine;
use crate::agent::tools::registry::ToolRegistry;
use crate::agent::validation;
use crate::bus::events::OutboundMessage;
use crate::config::schema::{EmailConfig, LcmSchemaConfig, ProprioceptionConfig};
use crate::cron::service::CronService;
use crate::errors::is_retryable_provider_error;
use crate::providers::base::{LLMResponse, StreamChunk, ToolCallRequest};

use crate::agent::agent_core::{
    append_to_system_prompt, apply_compaction_result, PendingCompaction, RuntimeCounters,
    SharedCoreHandle, SwappableCore,
};

use super::{
    adaptive_max_tokens, appears_incomplete, last_user_message, render_via_protocol,
    should_strip_tools_for_trio,
};

#[path = "agent_response.rs"]
pub(crate) mod agent_response;
pub(crate) use agent_response::RetryState;

// ---------------------------------------------------------------------------
// Per-instance state (different per agent)
// ---------------------------------------------------------------------------

/// Per-instance state that differs between the REPL agent and gateway agents.
pub(crate) struct AgentLoopShared {
    pub(crate) core_handle: SharedCoreHandle,
    pub(crate) subagents: Arc<SubagentManager>,
    pub(crate) bus_outbound_tx: UnboundedSender<OutboundMessage>,
    #[allow(dead_code)]
    pub(crate) bus_inbound_tx: UnboundedSender<crate::bus::events::InboundMessage>,
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
    pub(crate) calibrator:
        Option<Arc<parking_lot::Mutex<crate::agent::budget_calibrator::BudgetCalibrator>>>,
    /// Unified learning dispatch for turn observations.
    pub(crate) learn_loop: Arc<dyn crate::agent::learn_loop::LearnLoop>,
    /// Cluster router for distributed inference (feature-gated).
    #[cfg(feature = "cluster")]
    pub(crate) cluster_router: Option<Arc<crate::cluster::router::ClusterRouter>>,
    /// Knowledge store for proactive grounding retrieval.
    pub(crate) knowledge_store:
        Option<Arc<parking_lot::Mutex<crate::agent::knowledge_store::KnowledgeStore>>>,
    /// Experience buffer for perplexity gate (online learning).
    pub(crate) experience_buffer:
        Option<Arc<parking_lot::Mutex<crate::agent::lora_bridge::ExperienceBuffer>>>,
    /// Perplexity gate configuration.
    pub(crate) perplexity_gate_config: crate::config::schema::PerplexityGateConfig,
    /// In-process MLX provider for direct perplexity scoring + training (no HTTP).
    #[cfg(feature = "mlx")]
    pub(crate) mlx_provider: Option<std::sync::Arc<crate::providers::mlx::MlxProvider>>,
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
/// - `content_was_streamed`: one-shot, set when TextDelta chunks are sent
/// - `iterations_since_compaction`: counter, reset when compaction swaps in
/// - `tool_guard`: per-turn tool call policy enforcement
/// - `retries`: typed per-failure counters (validation, continuation, rescue, etc.)
pub(crate) struct FlowControl {
    pub(crate) force_response: bool,
    pub(crate) router_preflight_done: bool,
    pub(crate) tool_guard: ToolGuard,
    pub(crate) iterations_since_compaction: u32,
    pub(crate) content_was_streamed: bool,
    /// Consecutive rounds where ALL tool calls were blocked by the guard.
    /// When this reaches the threshold, the loop forces a text response.
    pub(crate) consecutive_all_blocked: u32,
    /// When the LLM call started — set in step_call_llm, read in step_process_response.
    pub(crate) llm_call_start: Option<std::time::Instant>,
    /// Typed retry counters — each failure mode has a named field with its own cap.
    pub(crate) retries: RetryState,
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
    Calling {
        tool_defs: Vec<Value>,
        max_tokens: u32,
    },
    /// Validate response, rescue pass, error check, token telemetry.
    Processing { response: LLMResponse },
    /// Route and execute tool calls (delegated or inline).
    Executing {
        response: LLMResponse,
        tool_calls: Vec<ToolCallRequest>,
    },
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
    pub(super) async fn process_message(
        &self,
        msg: &crate::bus::events::InboundMessage,
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

        let mut ctx = self
            .prepare_context(
                msg,
                text_delta_tx,
                tool_event_tx,
                cancellation_token,
                priority_rx,
            )
            .await;

        // Bug 3 fix: eagerly persist the user message before the LLM call so
        // it is not lost if the agent crashes mid-turn. Bump new_start so
        // finalize_response does not double-persist it.
        if ctx.new_start < ctx.messages.len() {
            let user_msg = ctx.messages[ctx.new_start].clone();
            ctx.core
                .sessions
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
            {
                let engine = ctx.reasoning.lock();
                // Only auto-decompose if no plan exists yet (Linear mode).
                if *engine.mode() == ReasoningMode::Linear {
                    drop(engine); // Release lock before re-acquiring mutably
                    if let Some(steps) =
                        crate::agent::reasoning::parse_numbered_steps(&ctx.user_content)
                    {
                        let step_budget = ctx.core.reasoning_config.step_budget;
                        let new_engine = ReasoningEngine::from_goals(&steps, step_budget);
                        {
                            let mut engine = ctx.reasoning.lock();
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
        let mut consecutive_empty = 0u32; // text-only responses with no tool calls
        const MAX_CONSECUTIVE_EMPTY: u32 = 3;
        while iteration < ctx.core.max_iterations {
            // Early exit if cancelled (e.g. user pressed Esc/Enter in REPL).
            if ctx
                .cancellation_token
                .as_ref()
                .map_or(false, |t| t.is_cancelled())
            {
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
                {
                    let engine = ctx.reasoning.lock();
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
                ctx.flow.retries.validation
            );

            // Sync messages to reasoning engine so CheckpointTool can capture them.
            {
                let mut engine = ctx.reasoning.lock();
                engine.sync_messages(&ctx.messages);
            }

            ctx.iterations_used = iteration + 1;
            let outcome = self.run_iteration(ctx, iteration).await;

            // Check for pending backtrack (set by BacktrackTool during tool execution).
            {
                {
                    let mut engine = ctx.reasoning.lock();
                    if let Some(restored) = engine.take_pending_restore() {
                        ctx.messages = restored;
                        iteration += 1;
                        ctx.flow.retries.validation = 0;
                        continue;
                    }
                }
            }

            match outcome {
                IterationOutcome::ValidationRetry => {
                    // A validation error injected a corrective hint. Only count
                    // this against the validation budget, not the main iteration
                    // budget, so format corrections don't exhaust real work slots.
                    consecutive_empty += 1;
                    if consecutive_empty >= MAX_CONSECUTIVE_EMPTY {
                        warn!(
                            "loop_breaker: {} consecutive non-tool iterations, forcing stop",
                            consecutive_empty
                        );
                        ctx.messages.push(json!({
                            "role": "user",
                            "content": format!(
                                "[System] Loop detected: you produced {} consecutive responses without executing any tool calls. \
                                 Your output may contain leaked thinking (<think> blocks) or text descriptions of actions instead of actual tool calls. \
                                 Stop describing what you want to do — either use a tool call or give your final answer as plain text.",
                                consecutive_empty
                            )
                        }));
                        // Promote to a real iteration to make progress
                        ctx.flow.retries.validation = 0;
                        iteration += 1;
                        continue;
                    }
                    ctx.flow.retries.validation += 1;
                    if ctx.flow.retries.validation >= validation::MAX_VALIDATION_RETRIES as u32 {
                        // Exhausted validation retries — treat as a normal
                        // iteration so the loop can make forward progress.
                        warn!(
                            "validation retries exhausted ({}/{}), counting as real iteration",
                            ctx.flow.retries.validation,
                            validation::MAX_VALIDATION_RETRIES,
                        );
                        ctx.flow.retries.validation = 0;
                        iteration += 1;
                    } else {
                        debug!(
                            "validation retry {}/{} — not counting against main budget",
                            ctx.flow.retries.validation,
                            validation::MAX_VALIDATION_RETRIES,
                        );
                        // Do NOT increment `iteration` — re-run the same slot.
                    }
                    continue;
                }
                IterationOutcome::Continue => {
                    // Successful tool execution — reset both counters.
                    consecutive_empty = 0;
                    ctx.flow.retries.validation = 0;
                    iteration += 1;
                    // Consume step budget if plan-guided.
                    {
                        let mut engine = ctx.reasoning.lock();
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
                    consecutive_empty = 0;
                    ctx.flow.retries.validation = 0;
                    iteration += 1;
                    // In plan-guided mode, advance to next step.
                    let should_continue = {
                        let mut engine = ctx.reasoning.lock();
                        if *engine.mode() != ReasoningMode::Linear {
                            engine.mark_current_completed(Some(content.clone()));
                            engine.advance();
                            !engine.is_complete()
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
                    ctx.flow.retries.validation = 0;
                    iteration += 1;
                    // Try backtracking before giving up.
                    let should_backtrack = {
                        let mut engine = ctx.reasoning.lock();
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
                IterationPhase::Preparing => self.step_prepare(ctx, iteration).await,
                IterationPhase::PreCall => self.step_pre_call(ctx, iteration).await,
                IterationPhase::Calling {
                    tool_defs,
                    max_tokens,
                } => self.step_call_llm(ctx, tool_defs, max_tokens).await,
                IterationPhase::Processing { response } => {
                    self.step_process_response(ctx, response).await
                }
                IterationPhase::Executing {
                    response,
                    tool_calls,
                } => self.step_execute_tools(ctx, response, tool_calls).await,
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
            let tools_list: Vec<String> = {
                let guard = counters.last_tools_called.lock();
                guard.clone()
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
            ctx.tools
                .get_local_definitions(&ctx.messages, &ctx.used_tools)
        } else if self.proprioception_config.enabled
            && self.proprioception_config.dynamic_tool_scoping
        {
            ctx.tools
                .get_scoped_definitions(&current_phase, &ctx.messages, &ctx.used_tools)
        } else {
            ctx.tools
                .get_relevant_definitions(&ctx.messages, &ctx.used_tools)
        };
        // Save tool_defs before potential stripping so we can restore them if
        // the router preflight returns Passthrough (router said "respond") — in
        // that case the main model must have tools as fallback.
        let saved_tool_defs = tool_defs.clone();
        if ctx.core.is_local && ctx.core.tool_delegation_config.strict_no_tools_main {
            // Hard separation (local trio only): main model is conversation/orchestration only.
            // Cloud providers handle tools natively and must never have them stripped.
            // BUT: if trio routing is degraded, keep tools so main model can still act.
            let router_probe_healthy = self
                .health_registry
                .as_ref()
                .map_or(false, |reg| reg.is_healthy("trio_router"));
            // Use the same key format as router.rs: "router:{model}".
            // Fallback to "trio_router" only when no router model is configured
            // (in which case trio won't run anyway).
            let cb_key = ctx
                .core
                .router_model
                .as_deref()
                .map_or_else(|| "trio_router".to_string(), |m| format!("router:{}", m));
            let cb_available = ctx
                .counters
                .trio_circuit_breaker
                .lock()
                .is_available(&cb_key);
            if should_strip_tools_for_trio(
                ctx.core.is_local,
                ctx.core.tool_delegation_config.strict_no_tools_main,
                router_probe_healthy,
                cb_available,
            ) {
                ctx.counters
                    .set_trio_state(crate::agent::agent_core::TrioState::Active);
                tool_defs.clear();
                // Tell the main model it's in orchestration mode (tools stripped).
                append_to_system_prompt(
                    &mut ctx.messages,
                    concat!(
                        "\n\n## Orchestration Mode (Active)\n",
                        "A trio routing system handles tool execution on your behalf.\n",
                        "- You do NOT have direct tool access in this mode.\n",
                        "- If a tool result appears as `[router:tool:X]` or `[specialist:X]`, ",
                        "incorporate that result into your response.\n",
                        "- If you need additional tool actions, describe them clearly ",
                        "(e.g., \"I need to read src/main.rs\") and the next turn will route it.\n",
                        "- Focus on reasoning, planning, and conversation.\n",
                    ),
                );
            } else {
                ctx.counters
                    .set_trio_state(crate::agent::agent_core::TrioState::Degraded);
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
        // Apply ToolGate size-class filtering (main agent loop only;
        // subagents handle their own tools_filter in SubagentManager).
        // Lane policy can override the effective size class (e.g. Answer
        // lane forces Small/tiny tier regardless of actual model size).
        let effective_size = ctx
            .core
            .lane
            .policy()
            .tools
            .effective_size_class(ctx.core.model_capabilities.size_class);
        if let Some(allowed) = ToolGate::filter(effective_size, None) {
            let allowed_set: std::collections::HashSet<&str> =
                allowed.iter().map(|s| s.as_str()).collect();
            tool_defs.retain(|def| {
                def.pointer("/function/name")
                    .and_then(|v| v.as_str())
                    .map_or(false, |name| allowed_set.contains(name))
            });
        }
        let tool_defs_opt: Option<&[Value]> = if tool_defs.is_empty() {
            None
        } else {
            Some(&tool_defs)
        };

        // Trim messages to fit context budget.
        let tool_def_tokens = TokenBudget::estimate_tool_def_tokens(tool_defs_opt.unwrap_or(&[]));
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
            // On first creation, restore DAG from DB summary_nodes table,
            // falling back to legacy Turn::Summary entries in messages.
            let lcm_engine = {
                let mut engines = self.lcm_engines.lock().await;
                if !engines.contains_key(&ctx.session_key) {
                    let config = LcmConfig::from(&self.lcm_config);

                    // Try DB-persisted DAG first (preferred).
                    let db_nodes = ctx.core.sessions.load_summary_nodes(&ctx.session_id).await;

                    let engine = if !db_nodes.is_empty() {
                        let all_msgs = ctx.core.sessions.get_all_messages(&ctx.session_id).await;
                        debug!(
                            session = %ctx.session_key,
                            node_count = db_nodes.len(),
                            "LCM: rebuilding engine from DB summary nodes"
                        );
                        LcmEngine::rebuild_from_db_nodes(&all_msgs, &db_nodes, config)
                    } else {
                        // Fallback: check for legacy Turn::Summary entries.
                        let all_msgs = ctx.core.sessions.get_all_messages(&ctx.session_id).await;
                        let turns: Vec<crate::agent::turn::Turn> = all_msgs
                            .iter()
                            .filter_map(|v| crate::agent::turn::turn_from_legacy(v))
                            .collect();
                        let has_summaries = turns.iter().any(|t| t.is_summary());

                        if has_summaries {
                            debug!(
                                session = %ctx.session_key,
                                summary_count = turns.iter().filter(|t| t.is_summary()).count(),
                                "LCM: rebuilding engine from legacy Turn::Summary entries"
                            );
                            LcmEngine::rebuild_from_turns(&turns, config, ctx.protocol.as_ref(), "")
                        } else {
                            LcmEngine::new(config)
                        }
                    };

                    engines.insert(
                        ctx.session_key.clone(),
                        Arc::new(tokio::sync::Mutex::new(engine)),
                    );
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
            let lcm_healthy = self
                .health_registry
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
                            compaction_type = if action == CompactionAction::Async {
                                "lcm_async"
                            } else {
                                "lcm_blocking"
                            },
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
                            let timeout_result =
                                tokio::time::timeout(Duration::from_secs(90), async {
                                    // Use dedicated LCM compactor if configured,
                                    // otherwise fall back to the core memory compactor.
                                    let compactor: &ContextCompactor =
                                        bg_lcm_compactor.as_deref().unwrap_or(&bg_core.compactor);
                                    let summary_turn = {
                                        let mut engine = bg_lcm.lock().await;
                                        engine.compact(compactor, &bg_core.token_budget, 0).await
                                    };

                                    // Extract text from Turn::Summary for working memory and result.
                                    let observation: Option<String> =
                                        summary_turn.as_ref().and_then(|t| {
                                            if let crate::agent::turn::Turn::Summary {
                                                text, ..
                                            } = t
                                            {
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
                                            bg_core
                                                .sessions
                                                .add_message(&bg_session_id, &summary_json)
                                                .await;
                                        }

                                        // Persist summary node to SQLite for DAG restoration.
                                        if let crate::agent::turn::Turn::Summary {
                                            text: ref s_text,
                                            ref source_ids,
                                            level: s_level,
                                        } = turn
                                        {
                                            let engine = bg_lcm.lock().await;
                                            // The node ID is dag.len() - 1 (just created).
                                            let node_id =
                                                engine.dag().len().saturating_sub(1);
                                            let tokens = crate::agent::token_budget::TokenBudget::estimate_str_tokens(s_text);
                                            bg_core
                                                .sessions
                                                .save_summary_node(
                                                    &bg_session_id,
                                                    node_id,
                                                    source_ids,
                                                    &[],
                                                    s_text,
                                                    tokens,
                                                    *s_level,
                                                )
                                                .await;
                                        }
                                    }

                                    // Update working memory with compaction observation.
                                    if bg_core.memory_enabled {
                                        if let Some(ref summary_text) = observation {
                                            bg_core.working_memory.update_from_compaction(
                                                &bg_session_key,
                                                summary_text,
                                                bg_turn_count,
                                            );
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
                                })
                                .await;
                            if timeout_result.is_err() {
                                warn!("LCM compaction timed out after 90s, resetting in_flight");
                            }
                            in_flight.store(false, Ordering::SeqCst);
                        });
                    }
                    CompactionAction::None => {}
                }
            }

            // Auto-expand relevant summaries before the LLM call.
            // This is the key innovation: the system decides when to expand,
            // not the model. Uses keyword overlap (no LLM needed).
            {
                let mut engine = lcm_engine.lock().await;
                if !engine.dag().is_empty() {
                    let expanded = engine.auto_expand(&ctx.core.token_budget, tool_def_tokens);
                    if expanded {
                        // Replace ctx.messages with the auto-expanded context.
                        ctx.messages = engine.active_context();
                        debug!("LCM auto_expand: replaced context with expanded messages");
                    }
                }
            }
        } else if !ctx.compaction.in_flight.load(Ordering::Relaxed)
            && ctx.core.compactor.needs_compaction(
                &ctx.messages,
                &ctx.core.token_budget,
                tool_def_tokens,
            )
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
                let timeout_result = tokio::time::timeout(Duration::from_secs(90), async {
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
                            bg_core.working_memory.update_from_compaction(
                                &bg_session_key,
                                summary,
                                bg_turn_count,
                            );
                        }
                    }
                    if result.messages.len() < bg_messages.len() {
                        *slot.lock().await = Some(PendingCompaction { result, watermark });
                    }
                })
                .await;
                if timeout_result.is_err() {
                    warn!("Core compaction timed out after 90s, resetting in_flight");
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
                        let ks_guard = self.knowledge_store.as_ref().map(|ks| ks.lock());
                        let ks_ref = ks_guard.as_deref();
                        let payload = crate::agent::proactive::retrieve_grounding(
                            &intent,
                            ks_ref,
                            &learning_context,
                            budget,
                        );
                        if let Some(text) =
                            crate::agent::proactive::format_grounding_message(&payload)
                        {
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
            let had_long = counters
                .long_mode_turns
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
                    if v > 0 {
                        Some(v - 1)
                    } else {
                        None
                    }
                })
                .is_ok();
            let user_text = ctx
                .messages
                .last()
                .and_then(|m| m.get("content"))
                .and_then(|c| c.as_str())
                .unwrap_or("");
            // Count recent tool calls: if tool-heavy, use smaller budget.
            let recent_tool_calls = ctx
                .messages
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
                if ctx.core.is_local
                    && ctx.core.model_capabilities.size_class
                        == crate::agent::model_capabilities::ModelSizeClass::Small
                {
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
            let mut stream = match ctx
                .core
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
                    if !ctx.flow.retries.api_retried && is_retryable_provider_error(&e) {
                        ctx.flow.retries.api_retried = true;
                        warn!(model = %ctx.core.model, error = %e, "llm_stream_call_failed_retrying");
                        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                        return StepResult::Done(IterationOutcome::Continue);
                    }
                    counters.inference_active.store(false, Ordering::Relaxed);
                    error!(model = %ctx.core.model, error = %e, "llm_stream_call_failed");
                    return StepResult::Done(IterationOutcome::Error(format!(
                        "I encountered an error: {}",
                        e
                    )));
                }
            };

            let mut streamed_response = None;
            let mut in_thinking = false;
            let suppress_thinking_tts = counters.suppress_thinking_in_tts.load(Ordering::Relaxed);
            let thinking_enabled = counters.thinking_budget.load(Ordering::Relaxed) > 0;
            let mut xml_filter = XmlToolCallFilter::new();
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
                                if !thinking_enabled || suppress_thinking_tts {
                                    // /t off → hide from display; /nothink → hide from TTS
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
                                // Filter out <tool_call>...</tool_call> XML
                                // blocks so they don't render in the terminal.
                                let filtered = xml_filter.filter(&delta);
                                if !filtered.is_empty() {
                                    let _ = delta_tx.send(filtered);
                                }
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
                    if ctx
                        .cancellation_token
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
            match ctx
                .core
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
                    if !ctx.flow.retries.api_retried && is_retryable_provider_error(&e) {
                        ctx.flow.retries.api_retried = true;
                        warn!(model = %ctx.core.model, error = %e, "llm_call_failed_retrying");
                        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                        return StepResult::Done(IterationOutcome::Continue);
                    }
                    counters.inference_active.store(false, Ordering::Relaxed);
                    error!(model = %ctx.core.model, error = %e, "llm_call_failed");
                    return StepResult::Done(IterationOutcome::Error(format!(
                        "I encountered an error: {}",
                        e
                    )));
                }
            }
        };

        // Inference complete — allow watchdog health checks again.
        counters.inference_active.store(false, Ordering::Relaxed);

        StepResult::Next(IterationPhase::Processing { response })
    }

    // -----------------------------------------------------------------------
    // Step 4: Processing — delegated to agent_response.rs
    // -----------------------------------------------------------------------
    // `step_process_response` is now implemented in `agent_response.rs` via
    // the `#[path]` submodule. It classifies the response into a
    // `ResponseKind` and dispatches to typed handler methods.
    //
    // See: agent_response::AgentLoopShared::step_process_response()

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
                    let key =
                        crate::agent::tool_runner::normalize_call_key(&tc.name, &tc.arguments);
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
                debug!(
                    "Delegation provider unhealthy — inline execution ({}/10 until re-probe)",
                    retries % 10
                );
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
            let should_checkpoint = routed_tool_calls
                .iter()
                .any(|tc| tc.name == "exec" || tc.name == "write_file");
            if should_checkpoint {
                {
                    let mut engine = ctx.reasoning.lock();
                    if *engine.mode() != crate::agent::reasoning::ReasoningMode::Linear {
                        engine.sync_messages(&ctx.messages);
                        engine.save_checkpoint("pre_exec", &ctx.messages, ctx.iterations_used);
                    }
                }
            }
        }

        // Inline path (default, unchanged): execute tools directly.
        crate::agent::tool_engine::execute_tools_inline(ctx, &routed_tool_calls, &response).await;

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
        if ctx
            .cancellation_token
            .as_ref()
            .map_or(false, |t| t.is_cancelled())
        {
            return StepResult::Done(IterationOutcome::Finished(String::new()));
        }

        StepResult::Done(IterationOutcome::Continue)
    }
}
