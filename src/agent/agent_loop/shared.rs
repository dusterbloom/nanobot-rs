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
use crate::agent::prefix_guard;
use crate::agent::protocol::{ConversationProtocol, XmlToolCallFilter};
use crate::agent::reasoning::{ReasoningEngine, ReasoningMode};
use crate::agent::runtime_mode::RuntimeMode;
use crate::agent::subagent::SubagentManager;
use crate::agent::system_state::{self, AhaPriority, AhaSignal, SystemState};
use crate::agent::token_budget::TokenBudget;
use crate::agent::tool_guard::ToolGuard;
use crate::agent::tools::reasoning_tools::SharedEngine;
use crate::agent::tools::registry::ToolRegistry;
use crate::agent::validation;
use crate::bus::events::OutboundMessage;
use crate::config::schema::{EmailConfig, LcmSchemaConfig, ProprioceptionConfig};
use crate::cron::service::CronService;
use crate::errors::is_retryable_provider_error;
use crate::providers::base::{LLMResponse, StreamChunk, ToolChoice};

use crate::agent::agent_core::{
    append_to_system_prompt, apply_compaction_result, PendingCompaction, RuntimeCounters,
    SharedCoreHandle, SwappableCore,
};

use super::{
    adaptive_max_tokens, last_user_message, render_via_protocol, should_strip_tools_for_trio,
};

// `response` is a sibling module declared in `mod.rs`. RetryState is re-exported
// from there; we just need a local alias for the field type below.
use super::response::RetryState;

fn send_cache_reset_marker(tx: &Option<tokio::sync::mpsc::UnboundedSender<String>>, reason: &str) {
    if let Some(tx) = tx {
        let _ = tx.send(format!("\x00cache:reset:{reason}"));
    }
}

fn clear_prompt_cache_state(ctx: &TurnContext) -> bool {
    let had_fingerprint = ctx
        .counters
        .prompt_fingerprints
        .lock()
        .remove(&ctx.session_key)
        .is_some();
    let had_watermark = ctx
        .counters
        .prompt_cache_watermark
        .lock()
        .remove(&ctx.session_key)
        .is_some();
    had_fingerprint || had_watermark
}

fn send_retract_reply_marker(tx: &Option<tokio::sync::mpsc::UnboundedSender<String>>) {
    if let Some(tx) = tx {
        let _ = tx.send("\x00retract_reply".to_string());
    }
}

fn stable_higgs_session_id(session_id: &str, epoch: u64) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in session_id.bytes().chain(epoch.to_le_bytes()) {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn attach_higgs_session_marker(messages: &mut [Value], session_id: u64) {
    if let Some(first) = messages.first_mut().and_then(Value::as_object_mut) {
        first.insert(
            crate::providers::openai_compat::NANOBOT_HIGGS_SESSION_ID_FIELD.to_string(),
            json!(session_id),
        );
    }
}

fn proactive_grounding_preserves_prefix_cache(is_local: bool, local_tail_opt_in: bool) -> bool {
    !is_local || local_tail_opt_in
}

// ---------------------------------------------------------------------------
// Per-instance state (different per agent)
// ---------------------------------------------------------------------------

/// Per-instance state that differs between the REPL agent and gateway agents.
pub(crate) struct AgentLoopShared {
    pub(crate) core_handle: SharedCoreHandle,
    pub(crate) subagents: Arc<SubagentManager>,
    pub(crate) bus_outbound_tx: UnboundedSender<OutboundMessage>,
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
    /// Runtime toggle for LCM (initialised from `lcm_config.is_enabled()`).
    /// Allows `/lcm` to flip without rebuilding the agent loop.
    pub(crate) lcm_enabled: AtomicBool,
    /// Dedicated LCM compactor (when `lcm.compaction_endpoint` is configured).
    pub(crate) lcm_compactor: Option<Arc<ContextCompactor>>,
    /// Health probe registry — used to gate LCM compaction when endpoint is degraded.
    pub(crate) health_registry: Option<Arc<crate::heartbeat::health::HealthRegistry>>,
    /// Cluster router for distributed inference (feature-gated).
    #[cfg(feature = "cluster")]
    pub(crate) cluster_router: Option<Arc<crate::cluster::router::ClusterRouter>>,
    /// Knowledge store for proactive grounding retrieval.
    pub(crate) knowledge_store:
        Option<Arc<parking_lot::Mutex<crate::agent::knowledge_store::KnowledgeStore>>>,
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

impl TurnContext {
    /// Check whether this turn has been cancelled (e.g. user pressed Esc in REPL).
    pub(crate) fn is_cancelled(&self) -> bool {
        self.cancellation_token
            .as_ref()
            .map_or(false, |t| t.is_cancelled())
    }

    /// Restore thinking budget if a previous iteration temporarily disabled it.
    pub(crate) fn restore_thinking_budget(&mut self) {
        if let Some(saved) = self.flow.restore_thinking_budget.take() {
            self.counters
                .thinking_budget
                .store(saved, Ordering::Relaxed);
        }
    }
}

/// Response-boundary lifecycle. After a side-effect tool (exec/write_file)
/// runs, the next LLM call is nudged to report results as text.
///
/// Enforcement happens at execution time (side-effect calls are rejected
/// with an error result), NOT by stripping tools from the schema: the tool
/// array renders at the head of the prompt, and any change there invalidates
/// server-side prefix caches — a full re-prefill of a 14k-token local
/// context costs ~60s at ~250 tok/s prefill speed.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub(crate) enum ResponseBoundary {
    /// No boundary in effect.
    #[default]
    Off,
    /// A side-effect tool just ran; arm the boundary on the next call.
    Pending,
    /// This call was nudged to respond; side-effect tool calls are rejected
    /// unless the response also carries a text report (compliance is
    /// behavioral — see `tool_engine::boundary_blocks_side_effects`).
    Armed,
}

/// Advance the response-boundary state machine at the start of an LLM call.
/// Returns the new state plus whether the wrap-up nudge should be injected.
///
/// One-shot by construction: `Armed` never carries into the next call, so a
/// model that insists on calling exec gets exactly one rejection nudge and
/// may proceed on the call after — no livelock.
fn advance_response_boundary(
    state: ResponseBoundary,
    cfg_enabled: bool,
) -> (ResponseBoundary, bool) {
    match state {
        ResponseBoundary::Pending if cfg_enabled => (ResponseBoundary::Armed, true),
        ResponseBoundary::Pending | ResponseBoundary::Armed => (ResponseBoundary::Off, false),
        ResponseBoundary::Off => (ResponseBoundary::Off, false),
    }
}

/// Per-turn flow control flags.
///
/// These are orthogonal fields (not a linear state machine):
/// - `boundary`: response-boundary lifecycle, set by exec/write_file tools
/// - `router_preflight_done`: one-shot, set after router runs
/// - `content_was_streamed`: one-shot, set when TextDelta chunks are sent
/// - `iterations_since_compaction`: counter, reset when compaction swaps in
/// - `tool_guard`: per-turn tool call policy enforcement
/// - `retries`: typed per-failure counters (validation, continuation, rescue, etc.)
pub(crate) struct FlowControl {
    pub(crate) boundary: ResponseBoundary,
    pub(crate) router_preflight_done: bool,
    pub(crate) tool_guard: ToolGuard,
    pub(crate) iterations_since_compaction: u32,
    pub(crate) content_was_streamed: bool,
    /// Consecutive rounds where ALL tool calls were blocked by the guard.
    /// When this reaches the threshold, the loop forces a text response.
    pub(crate) consecutive_all_blocked: u32,
    /// Set during tool execution when a round executed ZERO tools — every call
    /// was boundary-rejected or duplicate-blocked. Such a round made no progress,
    /// so the main loop does not spend a real iteration on it. Reset at the start
    /// of each iteration. Bounded by construction: the response boundary is
    /// one-shot (can't re-reject without an intervening successful round) and the
    /// duplicate circuit breaker forces a text response after 2 all-blocked rounds.
    pub(crate) round_executed_no_tools: bool,
    /// When the LLM call started — set in step_call_llm, read in step_process_response.
    pub(crate) llm_call_start: Option<std::time::Instant>,
    /// Time to first token (ms) for the current LLM call: elapsed from
    /// `llm_call_start` to the first streamed chunk. This is the prefill cost —
    /// the metric that dominates TTFT. Reset per call; `None` until first token
    /// (or for non-streaming calls).
    pub(crate) ttft_ms: Option<u64>,
    /// Typed retry counters — each failure mode has a named field with its own cap.
    pub(crate) retries: RetryState,
    /// Saved thinking budget to restore after a thinking-off retry iteration.
    pub(crate) restore_thinking_budget: Option<u32>,
}

impl FlowControl {
    /// Record time-to-first-token for the current call on the first streamed
    /// chunk (prefill done). Idempotent within a call — only the first chunk
    /// sets it; later chunks are no-ops.
    pub(crate) fn mark_first_token(&mut self) {
        if self.ttft_ms.is_none() {
            self.ttft_ms = self.llm_call_start.map(|t| t.elapsed().as_millis() as u64);
        }
    }
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
pub(crate) enum IterationPhase {
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
    Executing { response: LLMResponse },
}

/// Outcome of a single iteration, returned to the outer loop.
pub(crate) enum IterationOutcome {
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
pub(crate) enum StepResult {
    /// Transition to the next phase within this iteration.
    Next(IterationPhase),
    /// Iteration is done — report outcome to the outer loop.
    Done(IterationOutcome),
}

/// Decide whether a response warrants a Tier-2 forced-tool recovery: re-issuing
/// the turn with `tool_choice=required` so a local Higgs backend grammar-forces
/// a valid tool call instead of looping on a corrective hint.
///
/// Fires only when the model produced no real tool call, on the first
/// validation slot, in local mode, with tools available, and when the content
/// reads as a botched/claimed tool call. TextualReplay suppresses normal
/// validation, so it gets a narrower explicit check for plain named-tool
/// narration or invented tool-call markup.
fn should_attempt_forced_recovery(
    has_tool_calls: bool,
    is_local: bool,
    validation_retries: u32,
    tools_present: bool,
    content: Option<&str>,
    is_textual_replay: bool,
    had_blocked_calls: bool,
) -> bool {
    if has_tool_calls || !is_local || validation_retries != 0 || !tools_present {
        return false;
    }
    let Some(content) = content else {
        return false;
    };
    let outcome = validation::validate_response(content, &[], is_textual_replay, had_blocked_calls);
    if matches!(
        outcome,
        validation::ValidationOutcome::Error(
            validation::ValidationError::ClaimedButNotExecuted
                | validation::ValidationError::HallucinatedToolCall
        )
    ) {
        return true;
    }

    // Textual replay suppresses normal validation because bracketed tool
    // history is legitimate there, but plain named-tool narration and invented
    // tool-call envelopes are still botched calls.
    is_textual_replay
        && (validation::has_claimed_tool_intent(content)
            || validation::has_xml_hallucinated_tool_call(content)
            || validation::has_raw_json_hallucinated_tool_call(content))
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
        mode = if ctx.core.mode().is_local() && ctx.core.tool_delegation_config.strict_no_tools_main { "trio" } else { "inline" },
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
            if ctx.is_cancelled() {
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
                ctx.messages
                    .push(crate::agent::markers::scaffold_user(nudge_msg));
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
            // Per-round progress flag; tool execution sets it true if every call
            // was blocked/rejected (see the Continue arm below).
            ctx.flow.round_executed_no_tools = false;
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
                        ctx.messages.push(crate::agent::markers::scaffold_user(format!(
                            "[System] Loop detected: you produced {} consecutive responses without executing any tool calls. \
                             Your output may contain leaked thinking (<think> blocks) or text descriptions of actions instead of actual tool calls. \
                             Stop describing what you want to do — either use a tool call or give your final answer as plain text.",
                            consecutive_empty
                        )));
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
                    // Do NOT restore thinking budget here. When restore_thinking_budget
                    // is Some, the previous iteration temporarily disabled thinking for
                    // an empty-response retry. Restoring before the retry runs defeats
                    // the purpose. Restoration happens in the Finished arm instead.

                    ctx.flow.retries.validation = 0;

                    // A round where every tool call was blocked/rejected ran no
                    // tool and made no progress — don't spend a real iteration on
                    // it (so a behavioral-boundary rejection or a duplicate block
                    // can't silently eat the budget). Bounded: the boundary is
                    // one-shot and the duplicate circuit breaker caps it at 2.
                    if ctx.flow.round_executed_no_tools {
                        debug!(
                            "iteration not counted: round executed no tools (all blocked/rejected)"
                        );
                        continue;
                    }

                    // Successful tool execution — reset the empty counter.
                    consecutive_empty = 0;
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
                    ctx.restore_thinking_budget();
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
                    {
                        let mut engine = ctx.reasoning.lock();
                        if *engine.mode() != ReasoningMode::Linear {
                            engine.mark_current_failed(&msg);
                        }
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
                IterationPhase::Executing { response } => {
                    self.step_execute_tools(ctx, response).await
                }
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

        // Freeze the already-sent prefix (warm in the server's KV cache) so the
        // cleanup passes below rewrite only the uncached tail. Without this,
        // hygiene/anti-drift rewrite the middle of the array every iteration,
        // moving the first-divergent token earlier and forcing a full
        // re-prefill of the whole context (~65s for 19k tokens). The watermark
        // is re-anchored on every send in step_call; 0 = cold / post-trim /
        // post-compaction, i.e. unrestricted cleanup while a re-prefill is
        // already sunk cost. See `agent::prefix_guard`.
        let frozen_prefix = counters
            .prompt_cache_watermark
            .lock()
            .get(&ctx.session_key)
            .copied()
            .unwrap_or(0);

        // --- Context Hygiene: clean up conversation history ---
        prefix_guard::with_frozen_prefix(&mut ctx.messages, frozen_prefix, |m| {
            context_hygiene::hygiene_pipeline(m, ctx.core.hygiene_keep_last_messages);
        });

        // --- Anti-Drift: quality-based cleanup for local models ---
        if ctx.core.mode().needs_anti_drift() && ctx.core.anti_drift.enabled {
            prefix_guard::with_frozen_prefix(&mut ctx.messages, frozen_prefix, |m| {
                anti_drift::pre_completion_pipeline(m, iteration, &ctx.core.anti_drift);
            });
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
                ctx.messages
                    .push(crate::agent::markers::scaffold_user(grounding));
            }
        }

        ctx.flow.iterations_since_compaction += 1;

        // Install finished compaction only at cold/checkpoint boundaries. LCM may
        // summarize in the background, but replacing already-sent prompt bytes
        // while the local prefix cache is warm causes the long re-prefill stalls
        // this cache watermark is meant to prevent.
        self.install_pending_compaction(ctx, false).await;

        StepResult::Next(IterationPhase::PreCall)
    }

    // -----------------------------------------------------------------------
    // Step 2: PreCall — build tool defs, trim, compaction, repair, preflight
    // -----------------------------------------------------------------------

    /// Pre-LLM-call orchestrator: delegates to [`select_tool_definitions`],
    /// [`manage_compaction`], and [`compute_adaptive_max_tokens`], with
    /// inline steps for response boundary, trimming, grounding, rendering,
    /// emergency trim, and router preflight.
    #[instrument(name = "step_pre_call", skip(self, ctx), fields(
        iteration,
        trio_mode = ctx.core.mode().is_local() && ctx.core.tool_delegation_config.strict_no_tools_main,
        boundary = ?ctx.flow.boundary,
        msg_count = ctx.messages.len(),
    ))]
    async fn step_pre_call(&self, ctx: &mut TurnContext, iteration: u32) -> StepResult {
        // Response boundary: after exec/write_file, nudge the model to report
        // results as text. Tool definitions stay byte-stable across calls —
        // enforcement happens at execution time (execute_tools_inline rejects
        // side-effect calls while Armed) so the prompt head never changes and
        // server-side prefix caches survive the boundary.
        let boundary_cfg =
            ctx.core.provenance_config.enabled && ctx.core.provenance_config.response_boundary;
        let (new_boundary, inject_nudge) =
            advance_response_boundary(ctx.flow.boundary, boundary_cfg);
        ctx.flow.boundary = new_boundary;
        if inject_nudge {
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
            // Behavioral boundary: this nudge fires only when the model ran a
            // side-effect tool without reporting. Tell it what to do (report
            // first) instead of a bare acknowledgement. Marked `_synthetic` so
            // it is not persisted as a real turn and does not break the prefix
            // cache on the next reload.
            ctx.messages
                .push(crate::agent::markers::scaffold_user(format!(
                    "[system] Report what the previous tool results showed before \
                 running more tools.{budget_note}"
                )));
        }

        // Select and filter tool definitions for this turn.
        let (mut tool_defs, saved_tool_defs) = self.select_tool_definitions(ctx);
        let tool_defs_opt: Option<&[Value]> = if tool_defs.is_empty() {
            None
        } else {
            Some(&tool_defs)
        };

        // Trim messages to fit context budget.
        let tool_def_tokens = TokenBudget::estimate_tool_def_tokens(tool_defs_opt.unwrap_or(&[]));
        let frozen_prefix = ctx
            .counters
            .prompt_cache_watermark
            .lock()
            .get(&ctx.session_key)
            .copied()
            .unwrap_or(0);
        let (trimmed_messages, prefix_preserved) = ctx
            .core
            .token_budget
            .trim_to_fit_with_age_preserving_prefix(
                &ctx.messages,
                tool_def_tokens,
                ctx.turn_count,
                ctx.core.max_message_age_turns,
                frozen_prefix,
            );
        if !prefix_preserved && frozen_prefix > 0 {
            clear_prompt_cache_state(ctx);
            send_cache_reset_marker(&ctx.text_delta_tx, "trim");
            warn!(
                session = %ctx.session_key,
                frozen_prefix,
                before_messages = ctx.messages.len(),
                after_messages = trimmed_messages.len(),
                "prompt_cache_watermark_invalidated_by_token_trim"
            );
        }
        ctx.messages = trimmed_messages;
        if !prefix_preserved {
            self.install_pending_compaction(ctx, true).await;
        }

        // Spawn background compaction when threshold exceeded.
        self.manage_compaction(ctx, tool_def_tokens).await;

        // Proactive grounding: inject relevant knowledge before LLM call.
        //
        // Local models receive grounding as a synthetic `user` turn. That is
        // useful, but cache-hostile: LocalProtocol merges consecutive user
        // turns, while synthetic turns are stripped from the next replay, so
        // turn N+1 diverges at turn N's user message. Keep the local fast path
        // append-only unless the existing local-tail opt-in explicitly chooses
        // per-turn retrieval over prefix-cache reuse.
        let local_retrieval_opt_in = proactive_grounding_preserves_prefix_cache(
            ctx.core.mode().is_local(),
            std::env::var("NANOBOT_LOCAL_TAIL").is_ok(),
        );
        if self.proprioception_config.proactive_retrieval
            && local_retrieval_opt_in
            && iteration == 0
        {
            if let Some(user_text) = last_user_message(&ctx.messages) {
                if !user_text.is_empty() {
                    let intent = crate::agent::proactive::extract_intent(&user_text);
                    if intent.confidence >= 0.2 {
                        let budget = (ctx.core.token_budget.max_context() / 20).min(500);
                        // learnings removed; proactive grounding no longer uses tool-pattern hints
                        let learning_context = String::new();
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
                                "role": ctx.core.mode().grounding_role(),
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
            let frozen_prefix = ctx
                .counters
                .prompt_cache_watermark
                .lock()
                .get(&ctx.session_key)
                .copied()
                .unwrap_or(0);
            // tool_def_tokens=0 is conservative (trims more aggressively).
            let (trimmed_messages, prefix_preserved) = ctx
                .core
                .token_budget
                .trim_to_fit_with_age_preserving_prefix(&ctx.messages, 0, 0, 0, frozen_prefix);
            if !prefix_preserved && frozen_prefix > 0 {
                clear_prompt_cache_state(ctx);
                send_cache_reset_marker(&ctx.text_delta_tx, "emergency_trim");
                warn!(
                    session = %ctx.session_key,
                    frozen_prefix,
                    before_messages = ctx.messages.len(),
                    after_messages = trimmed_messages.len(),
                    "prompt_cache_watermark_invalidated_by_emergency_trim"
                );
            }
            ctx.messages = trimmed_messages;
            if !prefix_preserved {
                self.install_pending_compaction(ctx, true).await;
            }
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
        }

        // Adaptive max_tokens: size the response budget to the task.
        let effective_max_tokens = self.compute_adaptive_max_tokens(ctx);

        StepResult::Next(IterationPhase::Calling {
            tool_defs,
            max_tokens: effective_max_tokens,
        })
    }

    /// Select and filter tool definitions for this turn.
    ///
    /// Returns `(active_defs, saved_defs)` where `saved_defs` preserves the
    /// pre-trio-stripping state for router passthrough fallback.
    fn select_tool_definitions(&self, ctx: &mut TurnContext) -> (Vec<Value>, Vec<Value>) {
        // Filter tool definitions to relevant tools.
        // Local models get a minimal set to conserve context tokens.
        let current_phase = self.system_state.load_full().task_phase;
        let mut tool_defs = match ctx.core.mode() {
            RuntimeMode::Local { .. } => match ctx.core.local_tool_mode {
                crate::config::schema::LocalToolMode::Proxy => ctx.tools.get_proxy_definition(),
                crate::config::schema::LocalToolMode::Slim => ctx.tools.get_slim_definitions(),
                crate::config::schema::LocalToolMode::Full => ctx.tools.get_local_definitions(),
            },
            RuntimeMode::Cloud => {
                if self.proprioception_config.enabled
                    && self.proprioception_config.dynamic_tool_scoping
                {
                    ctx.tools
                        .get_scoped_definitions(&current_phase, &ctx.messages, &ctx.used_tools)
                } else {
                    // Cloud models have 100K+ context — give them all registered
                    // tools instead of keyword-gated subsets that hide capabilities.
                    ctx.tools.get_definitions()
                }
            }
        };
        // Tool-averse models (no tool-calling training, e.g. VibeThinker):
        // the native `tools` parameter confuses or errors their chat
        // templates, and nothing else would teach them the textual syntax the
        // response parser expects. Move the tool catalog into the system
        // prompt as a textual-protocol lesson and send no `tools` at all.
        if ctx.protocol.is_textual_replay()
            && !ctx.core.model_capabilities.tool_calling
            && !tool_defs.is_empty()
        {
            let already_taught = ctx
                .messages
                .first()
                .and_then(|m| m["content"].as_str())
                .is_some_and(|s| s.contains(crate::agent::protocol::TEXTUAL_TOOLS_MARKER));
            if !already_taught {
                append_to_system_prompt(
                    &mut ctx.messages,
                    &crate::agent::protocol::textual_tools_block(&tool_defs),
                );
            }
            tool_defs.clear();
        }
        // Save tool_defs before potential stripping so we can restore them if
        // the router preflight returns Passthrough (router said "respond") — in
        // that case the main model must have tools as fallback.
        let saved_tool_defs = tool_defs.clone();
        if ctx.core.mode().is_local() && ctx.core.tool_delegation_config.strict_no_tools_main {
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
                ctx.core.mode().is_local(),
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
        // NOTE: the response boundary deliberately does NOT filter tool_defs.
        // Schema changes invalidate server-side prefix caches (full re-prefill);
        // side-effect calls are rejected at execution time instead.
        //
        // Tool gating runs for cloud models only. Local models already get
        // condensed tool descriptions (~350 tokens for 12 tools, <1.1% of 32K
        // context) and real availability is enforced by `is_available()`, so
        // filtering would only remove useful tools.
        if matches!(ctx.core.mode(), RuntimeMode::Cloud) {
            if let Some(allowed) = ctx
                .core
                .lane
                .policy()
                .tools
                .allowed_tools(ctx.core.model_capabilities.size_class)
            {
                let allowed_set: std::collections::HashSet<&str> =
                    allowed.iter().map(|s| s.as_str()).collect();
                tool_defs.retain(|def| {
                    def.pointer("/function/name")
                        .and_then(|v| v.as_str())
                        .map_or(false, |name| allowed_set.contains(name))
                });
            }
        }

        (tool_defs, saved_tool_defs)
    }

    /// Install a finished compaction result only when it cannot invalidate a
    /// warm cached prefix, or when the caller already made an explicit
    /// checkpoint/reset. Otherwise leave the result pending as sidecar LCM state.
    async fn install_pending_compaction(
        &self,
        ctx: &mut TurnContext,
        allow_checkpoint: bool,
    ) -> bool {
        let Ok(mut guard) = ctx.compaction.slot.try_lock() else {
            return false;
        };
        let Some(pending) = guard.take() else {
            return false;
        };

        let frozen_prefix = ctx
            .counters
            .prompt_cache_watermark
            .lock()
            .get(&ctx.session_key)
            .copied()
            .unwrap_or(0);
        let rewrites_prompt = pending.result.messages.len() < pending.watermark;
        if rewrites_prompt && frozen_prefix > 0 && !allow_checkpoint {
            debug!(
                session = %ctx.session_key,
                frozen_prefix,
                compacted_messages = pending.result.messages.len(),
                watermark = pending.watermark,
                "lcm_compaction_deferred_for_prompt_cache"
            );
            *guard = Some(pending);
            return false;
        }

        if rewrites_prompt && clear_prompt_cache_state(ctx) {
            send_cache_reset_marker(&ctx.text_delta_tx, "lcm_checkpoint");
            warn!(
                session = %ctx.session_key,
                frozen_prefix,
                compacted_messages = pending.result.messages.len(),
                watermark = pending.watermark,
                "prompt_cache_watermark_invalidated_by_lcm_checkpoint"
            );
        }

        debug!(
            "Compaction swap: {} msgs -> {} compacted + {} new",
            pending.watermark,
            pending.result.messages.len(),
            ctx.messages.len().saturating_sub(pending.watermark)
        );
        // Record stats for `/lcm stats`: tokens of the replaced prefix vs its
        // compacted form (estimates, same estimator as budget accounting).
        let prefix_end = pending.watermark.min(ctx.messages.len());
        ctx.counters.record_compaction(
            TokenBudget::estimate_tokens(&ctx.messages[..prefix_end]) as u64,
            TokenBudget::estimate_tokens(&pending.result.messages) as u64,
        );
        apply_compaction_result(&mut ctx.messages, pending);
        // After compaction, all messages in the array are "new" from the
        // perspective of persistence (the session file was rebuilt).
        ctx.new_start = ctx.messages.len();
        ctx.flow.iterations_since_compaction = 0;
        // Bug 5 fix: after compaction ctx.messages shrinks but the LCM engine's
        // store_len reflects the old count. Override the skip offset so
        // step_pre_call ingests from index 0 instead of skipping past the end.
        ctx.lcm_synced_to = Some(0);
        true
    }

    /// Spawn background compaction when threshold exceeded.
    ///
    /// When LCM is enabled, uses the LCM engine's control loop (with DAG,
    /// summary persistence, and auto-expand). Otherwise falls back to core
    /// compaction (gradient/audience-aware/simple).
    async fn manage_compaction(&self, ctx: &mut TurnContext, tool_def_tokens: usize) {
        if self.lcm_enabled.load(Ordering::Relaxed) {
            // LCM path: get or create per-session engine, check thresholds.
            //
            // The engine starts EMPTY and ingests only the filtered messages
            // from ctx.messages (assembled by prepare_context/filter_history).
            // Previous versions loaded ALL session messages from the DB here,
            // which could be 1000+ messages / 150K+ tokens — immediately
            // triggering a massive compaction that made 75+ sequential LLM
            // calls to the 0.8B summarizer while competing for GPU time with
            // the main model. The session DB is the durable store; the LCM
            // engine only needs the current context window.
            let lcm_engine = {
                let mut engines = self.lcm_engines.lock().await;
                if !engines.contains_key(&ctx.session_key) {
                    let config = LcmConfig::from(&self.lcm_config);
                    let engine = LcmEngine::new(config);
                    engines.insert(
                        ctx.session_key.clone(),
                        Arc::new(tokio::sync::Mutex::new(engine)),
                    );
                }
                engines.get(&ctx.session_key).cloned().unwrap()
            };

            // Feed messages into the LCM engine's store.
            {
                let mut engine = lcm_engine.lock().await;
                if ctx.lcm_synced_to == Some(0) {
                    // Post-compaction: ctx.messages was rebuilt from the engine's
                    // active_context(). Re-ingesting via ingest() would APPEND
                    // duplicates. Reset the engine from the compacted messages.
                    engine.reset_from_messages(&ctx.messages);
                    ctx.lcm_synced_to = None;
                } else {
                    // Normal path: append only new messages (idempotent by index).
                    let store_len = engine.store_len();
                    for msg in ctx.messages.iter().skip(store_len) {
                        engine.ingest(msg.clone());
                    }
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
            let has_pending_compaction = ctx
                .compaction
                .slot
                .try_lock()
                .map(|guard| guard.is_some())
                .unwrap_or(true);
            if has_pending_compaction {
                debug!("LCM compaction skipped: pending checkpoint not installed yet");
            }
            if lcm_healthy
                && !has_pending_compaction
                && !ctx.compaction.in_flight.load(Ordering::Relaxed)
            {
                let (action, conv_tokens, available, hard_limit, soft_limit) = {
                    let engine = lcm_engine.lock().await;
                    let action = engine.check_thresholds(&ctx.core.token_budget, tool_def_tokens);
                    let available = ctx.core.token_budget.available_budget(tool_def_tokens);
                    let conv = engine.conversation_tokens();
                    let hard = (available as f64 * engine.tau_hard()) as usize;
                    let soft = (available as f64 * engine.tau_soft()) as usize;
                    (action, conv, available, hard, soft)
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
                            conv_tokens = conv_tokens,
                            available = available,
                            hard_limit = hard_limit,
                            soft_limit = soft_limit,
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

                        // Lazily start auxiliary server before compaction uses its endpoint.
                        // If aux fails, clear the dedicated compactor so we fall back to
                        // the main provider's compactor (bg_core.compactor).

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
                let frozen_prefix = ctx
                    .counters
                    .prompt_cache_watermark
                    .lock()
                    .get(&ctx.session_key)
                    .copied()
                    .unwrap_or(0);
                let mut engine = lcm_engine.lock().await;
                if !engine.dag().is_empty() && frozen_prefix == 0 {
                    let expanded = engine.auto_expand(&ctx.core.token_budget, tool_def_tokens);
                    if expanded {
                        // Replace ctx.messages with the auto-expanded context.
                        ctx.messages = engine.active_context();
                        debug!("LCM auto_expand: replaced context with expanded messages");
                    }
                } else if !engine.dag().is_empty() {
                    debug!(
                        session = %ctx.session_key,
                        frozen_prefix,
                        "LCM auto_expand skipped: prompt cache warm"
                    );
                }
            }
        } else {
            let has_pending_compaction = ctx
                .compaction
                .slot
                .try_lock()
                .map(|guard| guard.is_some())
                .unwrap_or(true);
            if has_pending_compaction {
                debug!("Core compaction skipped: pending checkpoint not installed yet");
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
        }
    }

    /// Compute effective max_tokens for this LLM call.
    ///
    /// Takes into account: `/long` override, user message complexity,
    /// recent tool call density, and thinking budget.
    fn compute_adaptive_max_tokens(&self, ctx: &TurnContext) -> u32 {
        let counters = &self.core_handle.counters;
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
            ctx.core.mode().is_local(),
            thinking_budget,
            &ctx.core.adaptive_tokens,
        )
    }

    // -----------------------------------------------------------------------
    // Step 3: Calling — invoke the LLM (streaming or blocking)
    // -----------------------------------------------------------------------

    /// Handle an LLM provider error: retry once if retryable, otherwise return error.
    async fn handle_llm_error(
        e: anyhow::Error,
        ctx: &mut TurnContext,
        counters: &RuntimeCounters,
        label: &str,
    ) -> StepResult {
        if !ctx.flow.retries.api_retried && is_retryable_provider_error(&e) {
            ctx.flow.retries.api_retried = true;
            warn!(model = %ctx.core.model, error = %e, "{label}_retrying");
            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
            return StepResult::Done(IterationOutcome::Continue);
        }
        counters.mark_inference_finished();
        error!(model = %ctx.core.model, error = %e, "{label}_failed");
        StepResult::Done(IterationOutcome::Error(format!(
            "I encountered an error: {}",
            e
        )))
    }

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
                // The cap value is config-driven (adaptive_tokens.local_thinking_small_model_cap),
                // so we match on mode + size_class rather than using the hardcoded
                // mode.thinking_cap_policy() constant.
                match ctx.core.mode() {
                    RuntimeMode::Local { caps }
                        if caps.size_class
                            == crate::agent::model_capabilities::ModelSizeClass::Small =>
                    {
                        Some(stored.min(ctx.core.adaptive_tokens.local_thinking_small_model_cap))
                    }
                    _ => Some(stored),
                }
            } else {
                None
            }
        };
        // Signal watchdog: LLM inference is active — skip health checks.
        counters.mark_inference_started();
        ctx.flow.llm_call_start = Some(std::time::Instant::now());
        ctx.flow.ttft_ms = None; // reset per call; set on this call's first token

        // Use the protocol-rendered wire format for the provider call.
        // `ctx.rendered_messages` was computed by `render_via_protocol()` in step_pre_call.
        let mut messages_for_llm = if ctx.rendered_messages.is_empty() {
            // Fallback: render now if step_pre_call was bypassed (should not happen in practice).
            render_via_protocol(&*ctx.protocol, &ctx.messages)
        } else {
            ctx.rendered_messages.clone()
        };

        // Prefix-divergence diagnostic: a prompt that is not an append-only
        // extension of this session's previous call forces the server to
        // re-prefill everything past the divergence point (~60s for a 14k
        // local context). Make every such miss a one-line diagnosis.
        use crate::agent::prompt_fingerprint::{self, PromptDelta};
        let prompt_fp = prompt_fingerprint::fingerprint(&messages_for_llm);
        let prompt_msg_count = messages_for_llm.len();
        let tool_def_tokens = TokenBudget::estimate_tool_def_tokens(tool_defs_opt.unwrap_or(&[]));
        let prompt_total_estimate =
            TokenBudget::estimate_tokens(&messages_for_llm).saturating_add(tool_def_tokens);
        {
            let store = counters.prompt_fingerprints.lock();
            let prompt_delta = prompt_fingerprint::compare(store.get(&ctx.session_key), &prompt_fp);
            let prefill_estimate = match prompt_delta {
                PromptDelta::First => prompt_total_estimate,
                PromptDelta::AppendOnly { added_msgs } => {
                    let tail_start = prompt_msg_count.saturating_sub(added_msgs);
                    TokenBudget::estimate_tokens(&messages_for_llm[tail_start..])
                }
                PromptDelta::Diverged {
                    first_divergent_msg,
                    ..
                } => {
                    let tail_start = first_divergent_msg.min(prompt_msg_count);
                    TokenBudget::estimate_tokens(&messages_for_llm[tail_start..])
                }
            };
            let cache_marker = match prompt_delta {
                PromptDelta::Diverged {
                    first_divergent_msg,
                    prev_msgs,
                    new_msgs,
                } => {
                    // WARN (not info): the default subscriber filter is `warn`,
                    // and a prefix divergence costs ~60s/turn on local — this must
                    // be visible in the log without RUST_LOG=info. Self-suppresses
                    // once the cause is fixed (the AppendOnly branch stays debug).
                    //
                    // `role_snippet` names the mutated message so the cache-busting
                    // field is identifiable without a separate dump: e.g. a tool
                    // result being re-compacted, or a provenance notice shifting.
                    let role_snippet = messages_for_llm
                        .get(first_divergent_msg)
                        .map(|m| {
                            let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("?");
                            let content = m.get("content").and_then(|c| c.as_str()).unwrap_or("");
                            let snippet: String = content.chars().take(70).collect();
                            format!("[{role}] {snippet}")
                        })
                        .unwrap_or_else(|| "(out of range)".to_string());
                    tracing::warn!(
                        session = %ctx.session_key,
                        at_msg = first_divergent_msg,
                        prev_msgs,
                        new_msgs,
                        prefill_estimate,
                        %role_snippet,
                        "prompt_prefix_diverged — server re-prefills past this point"
                    );
                    format!(
                        "\x00cache:diverged:{}:{}:{}",
                        first_divergent_msg, prev_msgs, new_msgs,
                    )
                }
                PromptDelta::AppendOnly { added_msgs } => {
                    debug!(
                        session = %ctx.session_key,
                        added_msgs,
                        prefill_estimate,
                        "prompt_append_only"
                    );
                    format!("\x00cache:append:{}:{}", added_msgs, prompt_msg_count)
                }
                PromptDelta::First => format!("\x00cache:first:{}", prompt_msg_count),
            };
            if let Some(ref delta_tx) = ctx.text_delta_tx {
                let _ = delta_tx.send(cache_marker);
                if prefill_estimate > 0 {
                    let _ = delta_tx.send(format!("\x00prefill_estimate:{prefill_estimate}"));
                }
            }
        }

        // Tool-block divergence diagnostic. The message fingerprint above is
        // blind to tool schemas by design, yet chat templates render the tool
        // block at the prompt head — so a tool block that changes between turns
        // busts the prefix cache invisibly (server re-prefills everything). This
        // catches that case: WARN when the serialized tool array hash changes.
        {
            let tool_count = tool_defs_opt.map_or(0, |t| t.len());
            let new_tool_hash = prompt_fingerprint::hash_tools(tool_defs_opt.unwrap_or(&[]));
            let mut tool_store = counters.prompt_tool_hashes.lock();
            if let Some(prev) = tool_store.get(&ctx.session_key) {
                if *prev != new_tool_hash {
                    tracing::warn!(
                        session = %ctx.session_key,
                        tool_count,
                        prev_hash = prev,
                        new_hash = new_tool_hash,
                        "tool_block_changed — chat template re-renders tool head, busting prefix cache"
                    );
                }
            }
            tool_store.insert(ctx.session_key.to_string(), new_tool_hash);
        }

        if ctx.core.mode().is_local() && ctx.core.provider.get_api_base().is_some() {
            let provider_session_id = stable_higgs_session_id(
                &ctx.session_id,
                counters.session_prompt_epoch(&ctx.session_key),
            );
            attach_higgs_session_marker(&mut messages_for_llm, provider_session_id);
        }

        let response = if let Some(ref delta_tx) = ctx.text_delta_tx {
            // Streaming path: forward text deltas to the REPL/voice renderer as
            // they arrive so the answer streams live — tokens appear as they are
            // generated and feed sentence-by-sentence into streaming TTS. Brief
            // pre-tool prose ("let me check…") streams as visible progress; tool
            // parameters and output never reach this channel (they are emitted as
            // ToolEvents and rendered separately).
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
                    return Self::handle_llm_error(e, ctx, counters, "llm_stream_call").await;
                }
            };

            let mut streamed_response = None;
            let mut in_thinking = false;
            let suppress_thinking_display =
                counters.suppress_thinking_display.load(Ordering::Relaxed);
            let thinking_enabled = counters.thinking_budget.load(Ordering::Relaxed) > 0;
            let hidden_reasoning_enabled =
                crate::agent::model_capabilities::prefers_hidden_reasoning(&ctx.core.model);
            let display_thinking =
                !suppress_thinking_display && (thinking_enabled || hidden_reasoning_enabled);
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
                                // First token (even a hidden thinking token) marks end of prefill.
                                ctx.flow.mark_first_token();
                                if !display_thinking {
                                    continue;
                                }
                                // Render thinking tokens as dimmed text
                                if !in_thinking {
                                    in_thinking = true;
                                    let _ = delta_tx.send("\x1b[90m\x1b[2m".to_string());
                                }
                                let _ = delta_tx.send(delta);
                            }
                            Some(StreamChunk::TextDelta(delta)) => {
                                ctx.flow.mark_first_token();
                                if in_thinking {
                                    in_thinking = false;
                                    let _ = delta_tx.send("\x1b[0m\n\n".to_string());
                                }
                                // Filter out <tool_call>...</tool_call> XML
                                // blocks so they don't render in the terminal.
                                let filtered = xml_filter.filter(&delta);
                                if !filtered.is_empty() {
                                    ctx.flow.content_was_streamed = true;
                                    let _ = delta_tx.send(filtered);
                                }
                            }
                            Some(StreamChunk::ToolCallDelta) => {
                                // Pure tool-call responses have no text/thinking
                                // deltas — the first tool-call fragment marks
                                // end of prefill for TTFT.
                                ctx.flow.mark_first_token();
                            }
                            Some(StreamChunk::PrefillProgress { processed, total }) => {
                                // Prefill still running — not a token, so no
                                // mark_first_token. Forward to the REPL spinner
                                // as a control marker.
                                let _ =
                                    delta_tx.send(format!("\x00prefill:{}/{}", processed, total));
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
                    counters.mark_inference_finished();
                    // Stream ended without Done — either cancelled or genuine error.
                    if ctx.is_cancelled() {
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
                    return Self::handle_llm_error(e, ctx, counters, "llm_call").await;
                }
            }
        };

        // Inference complete — allow watchdog health checks again.
        counters.mark_inference_finished();

        // Only successful provider calls prove that the server accepted this
        // prompt. Failed/cancelled calls must not seed the local cache model, or
        // the next long-context turn may preserve a prefix that never warmed.
        counters
            .prompt_fingerprints
            .lock()
            .insert(ctx.session_key.clone(), prompt_fp);
        counters
            .prompt_cache_watermark
            .lock()
            .insert(ctx.session_key.clone(), ctx.messages.len());

        // Tier-2 forced-tool recovery: if a local model botched a tool call
        // (intent prose / hallucinated syntax / empty block) instead of emitting
        // one, re-issue once with tool_choice=required so the Higgs backend
        // grammar-constrains a valid call — replacing the old hint-and-loop.
        let response = self
            .maybe_recover_botched_tool_call(
                ctx,
                response,
                &messages_for_llm,
                tool_defs_opt,
                max_tokens,
            )
            .await;

        StepResult::Next(IterationPhase::Processing { response })
    }

    /// One-shot forced-tool recovery for local backends. See call site above.
    ///
    /// Fires only on the first validation slot (`retries.validation == 0`), in
    /// local mode, with tools present, when the response has no real tool calls
    /// but its content reads as a botched/claimed tool call. Returns the
    /// recovered (constrained) response on success, else the original unchanged
    /// so the normal validation-retry path still applies.
    async fn maybe_recover_botched_tool_call(
        &self,
        ctx: &TurnContext,
        response: LLMResponse,
        messages_for_llm: &[Value],
        tool_defs_opt: Option<&[Value]>,
        max_tokens: u32,
    ) -> LLMResponse {
        if !should_attempt_forced_recovery(
            response.has_tool_calls(),
            ctx.core.mode().is_local(),
            ctx.flow.retries.validation,
            tool_defs_opt.is_some_and(|t| !t.is_empty()),
            response.content.as_deref(),
            ctx.protocol.is_textual_replay(),
            ctx.flow.tool_guard.had_blocked_calls,
        ) {
            return response;
        }

        info!(
            model = %ctx.core.model,
            "forced_tool_recovery: botched tool intent — re-issuing with tool_choice=required"
        );
        match ctx
            .core
            .provider
            .chat_with_tool_choice(
                messages_for_llm,
                tool_defs_opt,
                Some(&ctx.core.model),
                max_tokens,
                ctx.core.temperature,
                None, // forced from-token-0 constraint runs thinking-off
                None,
                ToolChoice::Required,
            )
            .await
        {
            Ok(recovered) if recovered.has_tool_calls() => {
                info!("forced_tool_recovery: recovered a constrained tool call");
                if ctx.flow.content_was_streamed {
                    send_retract_reply_marker(&ctx.text_delta_tx);
                }
                recovered
            }
            // No tool call (e.g. constraint disabled server-side) or error:
            // fall back to the original response and the hint-retry path.
            Ok(_) | Err(_) => response,
        }
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
    #[instrument(name = "step_execute_tools", skip(self, ctx, response), fields(
        delegation_enabled = ctx.core.tool_delegation_config.enabled,
        n_tool_calls = response.tool_calls.len(),
    ))]
    async fn step_execute_tools(&self, ctx: &mut TurnContext, response: LLMResponse) -> StepResult {
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

        // Lazily start auxiliary server if delegation targets a local endpoint.

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
        // A boundary-armed call must not delegate batches containing
        // side-effect tools — the inline path below rejects them in-protocol.
        let boundary_blocks_batch = ctx.flow.boundary == ResponseBoundary::Armed
            && routed_tool_calls
                .iter()
                .any(|tc| crate::agent::tool_engine::is_side_effect_tool(&tc.name));
        // Resolve provider+model from explicit config.
        let delegation_provider = ctx.core.tool_runner_provider.clone();
        let delegation_model = ctx.core.tool_runner_model.clone();
        // Same-model local delegation is pure prefix-cache poison. The delegation
        // sub-loop runs many distinct prompts on the SAME local server+model as
        // the main agent; those calls evict the main conversation's KV/radix
        // prefix, forcing a full re-prefill (~60-90s at large context, measured)
        // every tool round. It also yields ZERO token-cost benefit (same model).
        // When delegation would reuse the main local model, run tools inline so
        // the main prefix stays warm. A genuinely separate delegation model
        // (different name/server) is unaffected.
        let delegation_reuses_main_model =
            crate::agent::tool_engine::delegation_reuses_main_local_model(
                ctx.core.mode().is_local(),
                &ctx.core.model,
                delegation_model.as_deref(),
            );
        if delegation_reuses_main_model {
            debug!(
                model = %ctx.core.model,
                "delegation skipped: same local model as main — inline keeps the prefix cache warm"
            );
        }
        let should_delegate = ctx.core.tool_delegation_config.enabled
            && delegation_alive
            && !boundary_blocks_batch
            && !delegation_reuses_main_model;

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
                .any(|tc| crate::agent::tool_engine::is_side_effect_tool(&tc.name));
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
        if ctx.is_cancelled() {
            return StepResult::Done(IterationOutcome::Finished(String::new()));
        }

        StepResult::Done(IterationOutcome::Continue)
    }
}

// ============================================================================
// Wave 0 coverage net — pins current `is_local` branch outputs.
//
// Phase 09 plan:
//   .planning/phases/09-runtime-mode-spine/00-wave-0-coverage-PLAN.md
//
// These tests capture the output of every `ctx.core.is_local` branch site
// listed in 09-RESEARCH.md §1 "Critical Branch Points" so that the Wave 1–3
// migration to `RuntimeMode` is auditable: if any wave silently changes a
// derived value, one of these tests fails.
//
// Branch sites covered here (file:line from 09-RESEARCH.md):
//   :331   — trio mode tracing tag (`is_local && strict_no_tools_main`)   → `is_trio_mode_active`
//   :625   — anti-drift gate (`is_local && anti_drift.enabled`)           → `anti_drift_enabled_for_turn`
//   :671-673 (same as :625; historical dup in RESEARCH)                    → `anti_drift_enabled_for_turn`
//   :743   — pre-call tracing (`is_local && strict_no_tools_main`)        → `is_trio_mode_active`
//   :820   — grounding role ternary                                       → `grounding_role`
//   :866-868 (same as :820; documented row in RESEARCH)                    → `grounding_role`
//   :900   — `select_tool_definitions` local branch                       → covered via TODO + pins through `local_tool_mode`
//   :920   — trio-strip outer gate (`is_local && strict_no_tools_main`)   → `is_trio_mode_active`
//   :942-951 — `should_strip_tools_for_trio` free fn                      → pinned in `agent_heuristics::tests`
//   :983   — ToolGate cloud gate (`!is_local`)                            → `tool_gate_enabled_for_turn`
//   :1029-1036 (same decision as :983; RESEARCH row)                       → `tool_gate_enabled_for_turn`
//   :1351  — `adaptive_max_tokens(is_local, ...)`                         → pinned in `agent_heuristics::tests`
//   :1411  — thinking-cap small-model guard                               → `thinking_cap_applied`
//   :1457-1460 (same decision as :1411; RESEARCH row)                      → `thinking_cap_applied`
//
// Strategy: the deep branches inside async `impl AgentLoopShared` methods
// require a full `TurnContext` with counters, subagents, health registry,
// system_state snapshots, and `ToolRegistry`. Building those is out of scope
// for a coverage-only wave (would constitute a production refactor). Per
// PLAN.md action step 4, such branches are pinned here via small helper
// functions that mirror the decision expression verbatim; Wave 1 will replace
// the helpers with `RuntimeMode` method calls and re-run this same assertion
// set as the invariant net.
// ============================================================================
#[cfg(test)]
mod tests {
    use super::{
        advance_response_boundary, proactive_grounding_preserves_prefix_cache, ResponseBoundary,
    };

    /// The response boundary is one-shot: Pending → Armed (with nudge) → Off.
    /// Schema never changes; a model that insists on exec gets exactly one
    /// rejection and may proceed on the following call — no livelock.
    #[test]
    fn test_response_boundary_lifecycle() {
        use ResponseBoundary::{Armed, Off, Pending};
        // Pending + feature on → Armed, inject the wrap-up nudge.
        assert_eq!(advance_response_boundary(Pending, true), (Armed, true));
        // Armed never carries into the next call (one rejection max).
        assert_eq!(advance_response_boundary(Armed, true), (Off, false));
        // Feature off: Pending is dropped silently.
        assert_eq!(advance_response_boundary(Pending, false), (Off, false));
        // Off is stable regardless of config.
        assert_eq!(advance_response_boundary(Off, true), (Off, false));
        assert_eq!(advance_response_boundary(Off, false), (Off, false));
    }

    // Pure decision helpers — one per `is_local` read site. Each mirrors the
    // exact expression used at the cited line. Keeping them here (test-only)
    // avoids adding production code while still giving us assertion targets.

    /// Mirrors `agent_shared.rs:331, :743, :920` —
    /// `ctx.core.is_local && ctx.core.tool_delegation_config.strict_no_tools_main`.
    fn is_trio_mode_active(is_local: bool, strict_no_tools_main: bool) -> bool {
        is_local && strict_no_tools_main
    }

    /// Mirrors `agent_shared.rs:625, :671-673` —
    /// `ctx.core.is_local && ctx.core.anti_drift.enabled`.
    fn anti_drift_enabled_for_turn(is_local: bool, anti_drift_cfg_enabled: bool) -> bool {
        is_local && anti_drift_cfg_enabled
    }

    /// Mirrors `agent_shared.rs:820, :866-868` —
    /// `if ctx.core.is_local { "user" } else { "system" }`.
    fn grounding_role(is_local: bool) -> &'static str {
        if is_local {
            "user"
        } else {
            "system"
        }
    }

    /// Mirrors `agent_shared.rs:983, :1029-1036` — ToolGate size-class filter
    /// runs **only** for cloud models (`!is_local`).
    fn tool_gate_enabled_for_turn(is_local: bool) -> bool {
        !is_local
    }

    /// Mirrors `agent_shared.rs:1411, :1457-1460` — thinking-cap hard-limit is
    /// applied iff `is_local && size_class == Small`.
    fn thinking_cap_applied(
        is_local: bool,
        size_class: crate::agent::model_capabilities::ModelSizeClass,
    ) -> bool {
        is_local && size_class == crate::agent::model_capabilities::ModelSizeClass::Small
    }

    // ---- Tests ---------------------------------------------------------

    #[test]
    fn test_local_proactive_grounding_keeps_cache_fast_path() {
        assert!(
            proactive_grounding_preserves_prefix_cache(false, false),
            "cloud grounding is not governed by the local prefix-cache tradeoff"
        );
        assert!(
            !proactive_grounding_preserves_prefix_cache(true, false),
            "local default skips synthetic per-turn grounding to avoid msg-N cache resets"
        );
        assert!(
            proactive_grounding_preserves_prefix_cache(true, true),
            "NANOBOT_LOCAL_TAIL remains the explicit local relevance-over-cache opt-in"
        );
    }

    #[test]
    fn test_is_local_trio_mode_gate() {
        // Pins agent_shared.rs:331, :743, :920 — trio mode is ACTIVE only when
        // both `is_local` and `strict_no_tools_main` are true.
        assert!(is_trio_mode_active(true, true), "local + strict → trio ON");
        assert!(
            !is_trio_mode_active(true, false),
            "local without strict → trio OFF"
        );
        assert!(
            !is_trio_mode_active(false, true),
            "cloud never trios even with strict"
        );
        assert!(
            !is_trio_mode_active(false, false),
            "cloud + not-strict → trio OFF"
        );
    }

    #[test]
    fn test_is_local_anti_drift_gate() {
        // Pins agent_shared.rs:625, :671-673 — anti-drift pre-completion
        // pipeline runs only when `is_local` AND the anti_drift config is
        // enabled. Cloud models never see anti-drift.
        assert!(
            anti_drift_enabled_for_turn(true, true),
            "local + cfg.enabled → anti-drift runs"
        );
        assert!(
            !anti_drift_enabled_for_turn(true, false),
            "local + cfg.disabled → anti-drift skipped"
        );
        assert!(
            !anti_drift_enabled_for_turn(false, true),
            "cloud never runs anti-drift even if cfg.enabled"
        );
        assert!(
            !anti_drift_enabled_for_turn(false, false),
            "cloud + cfg.disabled → anti-drift skipped"
        );
    }

    #[test]
    fn test_is_local_grounding_role_ternary() {
        // Pins agent_shared.rs:820, :866-868 — proactive-grounding injection
        // uses role="user" on local (chat templates reject mid-conversation
        // system messages) and role="system" on cloud.
        assert_eq!(grounding_role(true), "user", "local grounding → user role");
        assert_eq!(
            grounding_role(false),
            "system",
            "cloud grounding → system role"
        );
    }

    #[test]
    fn test_is_local_tool_gate_cloud_only() {
        // Pins agent_shared.rs:983, :1029-1036 — ToolGate (phase-aware tool
        // scoping) runs for cloud only. Local models already get condensed
        // defs (<1.1% context) so ToolGate would hurt more than help.
        assert!(tool_gate_enabled_for_turn(false), "cloud → ToolGate runs");
        assert!(
            !tool_gate_enabled_for_turn(true),
            "local → ToolGate skipped"
        );
    }

    #[test]
    fn test_is_local_thinking_cap_small_model_guard() {
        // Pins agent_shared.rs:1411, :1457-1460 — thinking budget is
        // hard-capped only for local + small model. Local + medium/large and
        // any cloud size class pass through uncapped.
        use crate::agent::model_capabilities::ModelSizeClass;
        assert!(
            thinking_cap_applied(true, ModelSizeClass::Small),
            "local small → cap applied"
        );
        assert!(
            !thinking_cap_applied(true, ModelSizeClass::Medium),
            "local medium → no cap"
        );
        assert!(
            !thinking_cap_applied(true, ModelSizeClass::Large),
            "local large → no cap"
        );
        assert!(
            !thinking_cap_applied(false, ModelSizeClass::Small),
            "cloud small → no cap"
        );
        assert!(
            !thinking_cap_applied(false, ModelSizeClass::Medium),
            "cloud medium → no cap"
        );
        assert!(
            !thinking_cap_applied(false, ModelSizeClass::Large),
            "cloud large → no cap"
        );
    }

    #[test]
    fn test_is_local_tool_def_mode_dispatch_local() {
        // Pins agent_shared.rs:900 — when `is_local=true`, tool-def selection
        // dispatches on `local_tool_mode` (Proxy | Slim | Full). When
        // `is_local=false`, the cloud path takes over (tested by absence: the
        // local dispatch simply doesn't run).
        //
        // We pin the *shape* of the LocalToolMode enum here so that any
        // addition/removal of a variant during Wave 1 forces a compile error
        // in this test — exhaustive-match is a Nyquist filter.
        use crate::config::schema::LocalToolMode;
        let mode = LocalToolMode::default();
        match mode {
            LocalToolMode::Proxy | LocalToolMode::Slim | LocalToolMode::Full => {
                // Exhaustive match ensures any new variant trips the compiler,
                // forcing Wave 1 to update the RuntimeMode::Local { tool_mode }
                // constructor mapping.
            }
        }
        // Sanity: default is Slim (individual tool schemas, condensed descs).
        // Wave 1 `RuntimeMode::Local { tool_mode }` constructor must carry this
        // same default through unchanged.
        assert!(
            matches!(LocalToolMode::default(), LocalToolMode::Slim),
            "LocalToolMode::default() must stay Slim — Wave 1 RuntimeMode::Local tool_mode default pins on this"
        );
    }

    // NOTE (TODO phase-09-w1): Four of the eleven `is_local` reads in
    // agent_shared.rs live inside async methods on `AgentLoopShared` that
    // require a fully-wired `TurnContext` (counters, subagents, health
    // registry, ToolRegistry, system_state). Building that harness for a
    // read-only pin would exceed Wave 0's zero-production-change scope.
    // The decision expressions at those sites are structurally identical to
    // the helpers above, which ARE exercised:
    //   :942-951 → `should_strip_tools_for_trio` free fn (pinned in
    //              agent_heuristics::tests::test_should_strip_tools_for_trio_is_local_gate)
    //   :1351   → `adaptive_max_tokens(is_local, …)` free fn (pinned in
    //              agent_heuristics::tests::test_adaptive_max_tokens_is_local_budget)
    // Wave 1 will replace every `ctx.core.is_local` site with a
    // `ctx.core.mode()` method call; the invariant suite in Wave 0's
    // runtime_mode.rs will then act as the deep-path regression net.
}

#[cfg(test)]
mod forced_recovery_tests {
    use super::should_attempt_forced_recovery;

    // Real trigger strings (mirror src/agent/validation.rs tests).
    const CLAIMED: &str = "Let me check that file for you."; // ClaimedButNotExecuted
    const HALLUCINATED: &str = "I'll read it.\n[Called read_file({\"path\":\"/x\"})]"; // HallucinatedToolCall
    const CLEAN: &str = "The answer is 42."; // Ok — a genuine final answer

    #[test]
    fn fires_on_claimed_tool_intent() {
        assert!(should_attempt_forced_recovery(
            false,
            true,
            0,
            true,
            Some(CLAIMED),
            false,
            false
        ));
    }

    #[test]
    fn fires_on_hallucinated_call() {
        assert!(should_attempt_forced_recovery(
            false,
            true,
            0,
            true,
            Some(HALLUCINATED),
            false,
            false
        ));
    }

    #[test]
    fn skips_when_real_tool_calls_present() {
        assert!(!should_attempt_forced_recovery(
            true,
            true,
            0,
            true,
            Some(CLAIMED),
            false,
            false
        ));
    }

    #[test]
    fn skips_on_cloud_backend() {
        assert!(!should_attempt_forced_recovery(
            false,
            false,
            0,
            true,
            Some(CLAIMED),
            false,
            false
        ));
    }

    #[test]
    fn skips_after_first_validation_slot() {
        // One-shot: only the first slot (retries == 0) recovers.
        assert!(!should_attempt_forced_recovery(
            false,
            true,
            1,
            true,
            Some(CLAIMED),
            false,
            false
        ));
    }

    #[test]
    fn skips_when_no_tools_available() {
        assert!(!should_attempt_forced_recovery(
            false,
            true,
            0,
            false,
            Some(CLAIMED),
            false,
            false
        ));
    }

    #[test]
    fn skips_on_genuine_final_answer() {
        // Critical false-positive guard: a real answer must never be hijacked.
        assert!(!should_attempt_forced_recovery(
            false,
            true,
            0,
            true,
            Some(CLEAN),
            false,
            false
        ));
    }

    #[test]
    fn fires_on_named_tool_intent_in_textual_replay_mode() {
        // TextualReplay legitimately replays bracket tool history, but plain
        // future-action prose is still a botched tool call.
        assert!(should_attempt_forced_recovery(
            false,
            true,
            0,
            true,
            Some("I can use the `web_fetch` tool to get the content of that URL."),
            true,
            false
        ));
    }

    #[test]
    fn fires_on_xml_tool_call_hallucination_in_textual_replay_mode() {
        // TextualReplay allows our bracket replay format, not arbitrary fake
        // XML/function-call envelopes emitted as visible answer text.
        assert!(should_attempt_forced_recovery(
            false,
            true,
            0,
            true,
            Some(
                r#"<xml>
  <bigtag name="web_search">
    <arguments>
      <jsonobject>
        <parameters>
          <string>latest news</string>
        </parameters>
      </jsonobject>
    </arguments>
  </bigtag>
</xml>"#
            ),
            true,
            false
        ));
    }

    #[test]
    fn fires_on_raw_json_tool_call_hallucination_in_textual_replay_mode() {
        assert!(should_attempt_forced_recovery(
            false,
            true,
            0,
            true,
            Some(
                r#"{ "name": "exec", "parameters": { "command": "git clone https://github.com/dusterbloom/skybloom", "timeout": 60, "working_dir": "/home/your_user/Dev/nanobot-rs" } }"#,
            ),
            true,
            false
        ));
    }

    #[test]
    fn skips_bracket_history_in_textual_replay_mode() {
        assert!(!should_attempt_forced_recovery(
            false,
            true,
            0,
            true,
            Some("[I called: recall({\"query\":\"peppi\"})]"),
            true,
            false
        ));
    }
}
