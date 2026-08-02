//! `AgentLoopShared` struct, supporting types, and the `impl AgentLoopShared` block
//! containing the main agent loop step methods.
//!
//! Extracted from `agent_loop.rs` as a `#[path]` submodule.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use serde_json::{json, Value};
use tokio::sync::mpsc::UnboundedSender;
use tokio::sync::Mutex;
use tracing::{debug, error, info, instrument, warn};

use crate::agent::agent_loop::heuristics::{
    adaptive_max_tokens_for_artifact_action, evaluate_repeated_tool_round, RepeatBreakerAction,
};
use crate::agent::audit::{AuditLog, ToolEvent};
use crate::agent::context::ContextBuilder;
use crate::agent::lcm::{CompactionAction, CompactionFailureMode, LcmConfig, LcmEngine};
use crate::agent::lease::{BatchAdmission, Lease};
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
    append_to_system_prompt, apply_compaction_result, stable_higgs_session_id, RuntimeCounters,
    SharedCoreHandle, SwappableCore,
};

use super::{last_user_message, render_via_protocol, should_strip_tools_for_trio};

// `response` is a sibling module declared in `mod.rs`. RetryState is re-exported
// from there; we just need a local alias for the field type below.
use super::response::RetryState;
use crate::turn_stream::{BackendActivity, CacheResetReason, CacheStatus, ControlMarker};

use super::budget::{
    advertised_tool_names, attach_higgs_session_marker, clear_prompt_cache_state,
    conversation_token_count, divergent_message_digest, invalidate_prompt_cache_for_rewrite,
    proactive_grounding_preserves_prefix_cache, send_cache_reset_marker, send_compaction_marker,
    send_retract_reply_marker, should_allow_checkpoint, should_inject_heartbeat_grounding,
};
use super::compaction::execute_lcm_compaction;
use super::local_stream::{
    emit_stream_abort_metrics, local_artifact_action_for_turn, local_no_stream_headers_error,
    local_no_stream_progress_error, local_stream_no_progress_timeout, BackendActivityHeartbeat,
    LocalStreamProgress,
};

// `CompactionHandle` moved to the `compaction` submodule; re-exported here so
// the `agent_loop::CompactionHandle` path used by `prepare_context.rs` keeps
// working.
pub(crate) use super::compaction::CompactionHandle;

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
    /// Per-session previous-session continuity note. Computed once on the
    /// first turn of a fresh session (None for resumed sessions) and replayed
    /// byte-identically on every later turn so the system-prompt prefix stays
    /// stable within a session.
    pub(crate) continuity_notes: Arc<Mutex<HashMap<String, Option<String>>>>,
    /// Per-session LCM engines for lossless context management.
    pub(crate) lcm_engines: Arc<Mutex<HashMap<String, Arc<tokio::sync::Mutex<LcmEngine>>>>>,
    /// Per-concrete-session checkpoint state shared across agent turns.
    pub(crate) compaction_handles: Arc<Mutex<HashMap<String, CompactionHandle>>>,
    /// LCM configuration.
    pub(crate) lcm_config: LcmSchemaConfig,
    /// Health probes for foreground/router/specialist providers.
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
    /// Direct tool names advertised in the current provider request.
    ///
    /// The registry intentionally contains more tools than the model sees:
    /// uncommon tools stay reachable through the `tool` proxy. This snapshot
    /// lets execution reject stale direct calls without changing the prompt
    /// head and invalidating local prefix caches.
    pub(crate) advertised_tool_names: Option<HashSet<String>>,

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

    /// Persist every newly appended real protocol message and tag it with its
    /// SQLite row id. Tool-call carriers are flushed before execution and tool
    /// results immediately after injection, so a crash cannot leave a side
    /// effect without the conversation bytes that caused and described it.
    /// The row ids also make same-turn messages visible to the LCM ingester.
    pub(crate) async fn persist_pending_protocol_messages(&mut self) -> bool {
        // Do not rely solely on `new_start`: token trimming can remove old
        // history and shift every index during the turn. Persist by durable-id
        // presence instead; DB-loaded history already carries `_db_id`.
        let mut pending_indices = Vec::new();
        let mut pending_messages = Vec::new();
        for index in 0..self.messages.len() {
            let role = self.messages[index]
                .get("role")
                .and_then(|value| value.as_str());
            // Cache-replay scaffolds (boundary nudges, checkpoint notices, …)
            // were sent to the model, so they are part of the warm prompt
            // prefix. They MUST be persisted or the next turn's reloaded
            // history shrinks and diverges the prefix (`prompt_prefix_diverged`
            // — the observed 38→32 mid-session wire shrink). filter_history
            // replays them on reload without counting them as turns.
            if self.messages[index].get("_db_id").is_some()
                || (crate::agent::markers::is_synthetic(&self.messages[index])
                    && !crate::agent::markers::is_cache_replay(&self.messages[index]))
                // LCM summary blocks are engine-owned VIEWS of already-persisted
                // rows (they carry no _db_id by design). Persisting one creates
                // a fresh raw user row every turn, which later compactions
                // re-summarize — recursive summary-of-summary pollution.
                || self.messages[index].get("_lcm_summary").is_some()
                || matches!(role, Some("system" | "developer" | "summary"))
            {
                continue;
            }

            pending_indices.push(index);
            pending_messages.push(self.messages[index].clone());
        }

        if pending_messages.is_empty() {
            self.new_start = self.messages.len();
            return true;
        }

        let row_ids = match self
            .core
            .sessions
            .add_messages_checked(&self.session_id, &pending_messages)
            .await
        {
            Ok(row_ids) => row_ids,
            Err(error) => {
                warn!(
                    session = %self.session_key,
                    pending = pending_messages.len(),
                    %error,
                    "active_turn_protocol_group_persist_failed"
                );
                return false;
            }
        };
        for (index, row_id) in pending_indices.into_iter().zip(row_ids) {
            self.messages[index]["_db_id"] = json!(row_id);
        }
        self.new_start = self.messages.len();
        true
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

// ---------------------------------------------------------------------------
// Loop convergence bounds
// ---------------------------------------------------------------------------
//
// These constants govern independent loop termination paths. They are tuning
// constants, not user config (see AGENTS.md "no configurability that wasn't
// requested"). Lease exhaustion terminates at the first rejected batch and
// therefore does not feed the generic no-progress breaker.
pub(crate) const NO_PROGRESS_HARD_STOP: u32 = 4;
pub(crate) const MAX_LEASE_RENEWAL_REJECTIONS: u32 = 2;
pub(crate) const LEASE_OVER_BUDGET_FINAL: &str =
    "I stopped this turn because the model requested another tool batch after exhausting its tool lease. Ask me to continue if you want a fresh lease.";

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
    /// Consecutive rounds that executed zero tools (for ANY reason: lease
    /// exhausted, coarse-family cap, boundary reject, duplicate block). Such
    /// rounds are "not counted" against `max_iterations` so they cannot silently
    /// eat the model's budget — but without a hard cap a model that keeps
    /// emitting blocked tool calls (e.g. lease exhausted but still emitting
    /// tool_calls after the strip) would spin forever. After
    /// `NO_PROGRESS_HARD_STOP` such rounds the loop forces a final answer
    /// regardless. This is the universal termination invariant the loop relies
    /// on; the per-reason counters above feed into it but do not replace it.
    pub(crate) consecutive_no_progress_rounds: u32,
    /// Set during tool execution when a round executed ZERO tools — every call
    /// was boundary-rejected or duplicate-blocked. Such a round made no progress,
    /// so the main loop does not spend a real iteration on it. Reset at the start
    /// of each iteration. Bounded by construction: the response boundary is
    /// one-shot (can't re-reject without an intervening successful round) and cached
    /// duplicate receipts force a text response immediately.
    pub(crate) round_executed_no_tools: bool,
    /// Per-turn tool lease. Caps the total tool calls per turn at
    /// `lease_size * (1 + max_renewals)`. A complete assistant batch is
    /// admitted or rejected before execution; rejection ends the turn without
    /// changing the advertised tool schema. See
    /// `docs/superpowers/specs/2026-07-27-tool-leases-design.md`.
    pub(crate) lease: Lease,
    /// Append-only provider-facing detail allowance shared by every new tool
    /// result in this turn. Persisted messages are never revisited to reclaim it.
    pub(crate) tool_preview_chars_remaining: usize,
    /// When the LLM call started — set in step_call_llm, read in step_process_response.
    pub(crate) llm_call_start: Option<std::time::Instant>,
    /// Time to first token (ms) for the current LLM call: elapsed from
    /// `llm_call_start` to the first streamed chunk. This is the prefill cost —
    /// the metric that dominates TTFT. Reset per call; `None` until first token
    /// (or for non-streaming calls).
    pub(crate) ttft_ms: Option<u64>,
    /// Prompt estimate for the exact provider request, including tool schemas.
    /// Compared with provider-reported `prompt_tokens` in turn telemetry.
    pub(crate) provider_prompt_estimate: Option<usize>,
    /// Typed retry counters — each failure mode has a named field with its own cap.
    pub(crate) retries: RetryState,
    /// Saved thinking budget to restore after a thinking-off retry iteration.
    pub(crate) restore_thinking_budget: Option<u32>,
    /// Exact provider-request identity across tool rounds. If a completed tool
    /// round produces the same request bytes, calling the model would only
    /// repeat stale work; force a context checkpoint first.
    pub(crate) provider_request: ProviderRequestState,
    pub(crate) tool_rounds_completed: u32,
    /// Metrics for a provider response that requested tools. Emission is
    /// deferred until routing/execution knows the truthful executed count.
    pub(crate) pending_request_metrics: Option<crate::agent::metrics::RequestMetrics>,
    /// Normalized keys of the tool calls dispatched in the most recent
    /// executed round (empty if the round ran no tools). Set in
    /// `step_execute_tools`; read by the repeated-tool-call loop breaker.
    pub(crate) last_round_keys: Vec<String>,
    /// Normalized keys of the previously executed round, for repeat detection.
    pub(crate) prev_round_keys: Vec<String>,
    /// Consecutive executed rounds that dispatched the identical tool calls as
    /// the previous round. Drives the repeated-tool-call loop breaker.
    pub(crate) consecutive_repeat_rounds: u32,
    /// True once we've already nudged about a repeating tool call; the next
    /// repeat forces a stop instead of nudging again.
    pub(crate) repeat_nudged: bool,
    /// Infrastructure error surfaced by the tool engine when the
    /// "handles-not-bodies" invariant cannot be honored — i.e. the immutable
    /// tool-result stash rejected a write (`Conflict` / `Failed`). When set
    /// after `step_execute_tools`, the loop finalizes the turn with this error
    /// so the user sees the abort reason and the body is NEVER shown raw.
    /// See `abort_turn_on_stash_failure` in `tool_engine.rs`.
    pub(crate) infra_error: Option<String>,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ProviderRequestAdmission {
    Proceed,
    ForceCheckpoint,
}

#[derive(Default)]
pub(crate) struct ProviderRequestState {
    last_hash: Option<u64>,
    last_tool_round: u32,
}

impl ProviderRequestState {
    fn admit(&mut self, request_hash: u64, tool_round: u32) -> ProviderRequestAdmission {
        if self.last_hash == Some(request_hash) && tool_round > self.last_tool_round {
            return ProviderRequestAdmission::ForceCheckpoint;
        }
        self.last_hash = Some(request_hash);
        self.last_tool_round = tool_round;
        ProviderRequestAdmission::Proceed
    }
}

impl TurnContext {
    pub(crate) fn emit_pending_request_metrics(&mut self, tool_calls_executed: u32) {
        if let Some(mut metrics) = self.flow.pending_request_metrics.take() {
            metrics.tool_calls_executed = tool_calls_executed;
            crate::agent::metrics::emit(&metrics);
        }
    }
}

#[cfg(test)]
mod lcm_checkpoint_tests {
    use crate::agent::agent_loop::compaction::LcmCompactionMutation;
    use crate::agent::compaction::ContextCompactor;
    use crate::agent::lcm::{CompactionFailureMode, LcmConfig, LcmEngine};
    use crate::agent::token_budget::TokenBudget;
    use crate::providers::base::{LLMProvider, LLMResponse};
    use serde_json::json;
    use std::sync::Arc;

    /// Mock LLM that returns a short summary — just enough to exercise the
    /// checkpoint-rollback-on-drop machinery below, which doesn't care which
    /// escalation level produced the summary.
    /// Reflects real input content back as short bullets instead of a fixed
    /// canned string, so it survives the compaction fidelity gate's
    /// topic-anchor / protected-literal checks (`compaction.rs`) for
    /// whatever conversation this test feeds it.
    struct StubSummarizer;

    #[async_trait::async_trait]
    impl LLMProvider for StubSummarizer {
        async fn chat(
            &self,
            messages: &[serde_json::Value],
            _tools: Option<&[serde_json::Value]>,
            _model: Option<&str>,
            _max_tokens: u32,
            _temperature: f64,
            _thinking_budget: Option<u32>,
            _top_p: Option<f64>,
        ) -> anyhow::Result<LLMResponse> {
            let request = messages
                .iter()
                .rev()
                .find_map(|m| m.get("content").and_then(|c| c.as_str()))
                .unwrap_or("");
            let heads: Vec<String> = request
                .lines()
                .filter(|line| {
                    ["user: ", "assistant: ", "tool: "]
                        .iter()
                        .any(|role| line.starts_with(role))
                })
                .map(|line| {
                    line.split_whitespace()
                        .take(3)
                        .collect::<Vec<_>>()
                        .join(" ")
                })
                .collect();
            let chunk_size = heads.len().div_ceil(15).max(1);
            let bullets: Vec<String> = heads
                .chunks(chunk_size)
                .map(|group| format!("- {}", group.join("; ")))
                .collect();
            Ok(LLMResponse {
                content: Some(bullets.join("\n")),
                tool_calls: vec![],
                finish_reason: "stop".to_string(),
                usage: std::collections::HashMap::new(),
            })
        }

        fn get_default_model(&self) -> &str {
            "stub-summarizer"
        }
    }

    #[tokio::test]
    async fn failed_checkpoint_persistence_restores_tentative_engine_state() {
        let mut engine = LcmEngine::new(LcmConfig {
            tau_soft: 0.05,
            tau_hard: 0.8,
            deterministic_target: 64,
        });
        for id in 1..=12 {
            let role = if id % 2 == 0 { "assistant" } else { "user" };
            engine.ingest(json!({
                "role": role,
                "content": format!("checkpoint-{id} {}", "x".repeat(400)),
                "_db_id": id,
            }));
        }
        engine.request_async_compaction();
        let active_before = engine.active_context();
        let dag_before = engine.dag().len();
        let store_before = engine.store_len();

        let compactor = ContextCompactor::new(
            Arc::new(StubSummarizer) as Arc<dyn LLMProvider>,
            "mock".to_string(),
            4096,
        );
        {
            let mut mutation = LcmCompactionMutation::new(&mut engine);
            let compacted = mutation
                .engine_mut()
                .compact(
                    Some(&compactor),
                    &TokenBudget::new(4096, 512),
                    0,
                    CompactionFailureMode::Deterministic,
                )
                .await;
            assert!(compacted.is_some());
            assert_ne!(mutation.engine().active_context(), active_before);
            // Simulate SQLite failure/cancellation by dropping without commit.
        }

        assert_eq!(engine.active_context(), active_before);
        assert_eq!(engine.dag().len(), dag_before);
        assert_eq!(engine.store_len(), store_before);
        assert_eq!(
            engine.check_thresholds(&TokenBudget::new(4096, 512), 0),
            crate::agent::lcm::CompactionAction::Async,
            "rollback must clear the in-flight marker so compaction can retry"
        );
    }
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
        let mut ctx = self
            .prepare_context(
                msg,
                text_delta_tx,
                tool_event_tx,
                cancellation_token,
                priority_rx,
            )
            .await;

        // `prepare_context` owns request-id creation. Reuse that exact id for
        // the outer lifecycle log, provider metrics, and downstream spans so a
        // single turn never appears as two unrelated requests.
        info!(
            request_id = %ctx.request_id,
            role = "main",
            model = %ctx.core.model,
            channel = %msg.channel,
            "request_start"
        );

        // Make the inbound user turn durable before the first provider call.
        ctx.persist_pending_protocol_messages().await;

        self.run_agent_loop(&mut ctx).await;
        self.finalize_response(ctx).await
    }

    /// Phase 2: Run the main agent loop (LLM calls + tool execution).
    ///
    /// Thin loop driver: delegates each iteration to [`run_iteration`] which
    /// drives the inner state machine through `IterationPhase` steps.
    #[instrument(name = "agent_loop", skip(self, ctx), fields(
        session = %ctx.session_key,
        mode = if ctx.core.mode().is_local() && ctx.core.tool_delegation_config.strict_no_tools_main() { "trio" } else { "inline" },
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
        // Executed rounds that dispatch the identical tool calls (name + args) as
        // the previous round. The model is firing tools without consuming their
        // results; two in a row means it never "sees" the output that came back.
        const MAX_CONSECUTIVE_REPEAT_ROUNDS: u32 = 2;
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
                        // Cache-replay tagged: a step instruction sent live must
                        // replay byte-identical on reload or the warm prefix
                        // diverges and Higgs re-prefills.
                        ctx.messages
                            .push(crate::agent::markers::scaffold_user(format!(
                                "[Current objective] {}",
                                instruction
                            )));
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
                    // can't silently eat the budget). But a model can keep emitting
                    // blocked tool calls indefinitely (e.g. lease exhausted, tools
                    // stripped, still emitting tool_calls). Bound the TOTAL
                    // consecutive no-progress rounds: after the hard stop the loop
                    // forces a final answer regardless of the reason. This is the
                    // universal convergence invariant — without it the loop spins.
                    if ctx.flow.round_executed_no_tools {
                        ctx.flow.consecutive_no_progress_rounds = ctx
                            .flow
                            .consecutive_no_progress_rounds
                            .saturating_add(1);
                        if ctx.flow.consecutive_no_progress_rounds >= NO_PROGRESS_HARD_STOP {
                            warn!(
                                rounds = ctx.flow.consecutive_no_progress_rounds,
                                "no_progress_hard_stop: repeated zero-tool rounds, forcing final answer"
                            );
                            ctx.final_content =
                                "I was looping on blocked tool requests without making progress, \
                                 so I stopped. Rephrase the request or restart the turn."
                                    .to_string();
                            break;
                        }
                        debug!(
                            rounds = ctx.flow.consecutive_no_progress_rounds,
                            "iteration not counted: round executed no tools (all blocked/rejected)"
                        );
                        continue;
                    }

                    // Successful tool execution — reset the empty counter and the
                    // no-progress streak.
                    consecutive_empty = 0;
                    ctx.flow.consecutive_no_progress_rounds = 0;

                    // Repeated successful tool-call breaker: the model fires
                    // tools without consuming their results when it dispatches
                    // the identical calls two rounds in a row. Nudge once that
                    // the results are already in context; stop if it repeats
                    // again after the nudge.
                    let (action, new_rounds, new_nudged) = evaluate_repeated_tool_round(
                        &ctx.flow.last_round_keys,
                        &ctx.flow.prev_round_keys,
                        ctx.flow.consecutive_repeat_rounds,
                        ctx.flow.repeat_nudged,
                        MAX_CONSECUTIVE_REPEAT_ROUNDS,
                    );
                    ctx.flow.consecutive_repeat_rounds = new_rounds;
                    ctx.flow.repeat_nudged = new_nudged;
                    ctx.flow.prev_round_keys = ctx.flow.last_round_keys.clone();

                    match action {
                        RepeatBreakerAction::Stop => {
                            warn!(
                                "tool_loop_breaker: repeated identical tool calls after nudge, forcing stop"
                            );
                            ctx.messages.push(crate::agent::markers::scaffold_user(
                                "[System] You called the same tool(s) with the same arguments again even though the results are already in your context above. Stopping to avoid an infinite loop — use the results you already have, or give your final answer.".to_string(),
                            ));
                        }
                        RepeatBreakerAction::Nudge => {
                            warn!(
                                "tool_loop_breaker: {} consecutive identical tool rounds, injecting result-available nudge",
                                new_rounds + 1
                            );
                            ctx.messages.push(crate::agent::markers::scaffold_user(
                                "[System] Your tool results are already in the conversation above — you just called the same tool(s) with the same arguments again without using them. The output is present; stop re-calling and either act on the results or give your final answer.".to_string(),
                            ));
                        }
                        RepeatBreakerAction::Continue => {}
                    }

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
                    // A finished turn breaks any repeating-tool-call streak.
                    ctx.flow.consecutive_repeat_rounds = 0;
                    ctx.flow.repeat_nudged = false;
                    ctx.flow.prev_round_keys.clear();
                    ctx.flow.last_round_keys.clear();
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
        let cache_watermark = counters
            .prompt_cache_watermark
            .lock()
            .get(&ctx.session_key)
            .copied()
            .unwrap_or(0);
        // Persisted-row floor: rows with a `_db_id` are replayed VERBATIM from
        // SQLite on the next turn's reload, so an in-memory-only rewrite of
        // them (anti-drift collapse, hygiene dedup/drop) is reverted at the
        // turn boundary — the next reload no longer matches the previous wire
        // and the server re-prefills the whole context. This floor holds even
        // when the cache watermark was cleared by a sanctioned reset (stall
        // checkpoint, trim): the reset pays for ONE re-prefill, not for a
        // second one when the reload reverts the rewrite. Cleanup passes may
        // only touch the unpersisted scratch tail.
        let persisted_floor = ctx
            .messages
            .iter()
            .rposition(|m| m.get("_db_id").is_some())
            .map_or(0, |i| i + 1);
        let frozen_prefix = cache_watermark.max(persisted_floor);

        // --- Retention shaping: context hygiene, then (local-only, gated)
        // anti-drift quality cleanup. Single owner: `agent::retention`.
        let run_anti_drift =
            ctx.core.mode().needs_anti_drift() && ctx.core.retention.anti_drift.enabled;
        prefix_guard::with_frozen_prefix(&mut ctx.messages, frozen_prefix, |m| {
            ctx.core
                .retention
                .apply_shaping(m, iteration, run_anti_drift);
        });

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
                            ctx.messages
                                .push(crate::agent::markers::scaffold_user(format!(
                                    "[ALERT from subagent {}] {}",
                                    signal.agent_id, signal.message
                                )));
                        }
                        AhaPriority::High => {
                            ctx.messages
                                .push(crate::agent::markers::scaffold_user(format!(
                                    "[Signal from subagent {}] {}",
                                    signal.agent_id, signal.message
                                )));
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
            if should_inject_heartbeat_grounding(
                iteration,
                self.proprioception_config.grounding_interval,
                state.context_pressure,
                ctx.core.mode().is_local(),
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
        // this cache watermark is meant to prevent — UNLESS pressure has crossed
        // `tau_hard`, where deferring any longer guarantees hitting max tokens.
        let state = self.system_state.load_full();
        let allow_checkpoint =
            should_allow_checkpoint(state.context_pressure, self.lcm_config.tau_hard);
        self.install_pending_compaction(ctx, allow_checkpoint).await;

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
        trio_mode = ctx.core.mode().is_local() && ctx.core.tool_delegation_config.strict_no_tools_main(),
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
                 running more tools. If you created or changed an artifact, do not \
                 claim completion until you validate it with an appropriate tool \
                 and fix any errors.{budget_note}"
                )));
        }

        // Select and filter tool definitions for this turn. Lease enforcement
        // never mutates this array: Higgs retains the tool schema as part of
        // the session prefix, so admission happens atomically at execution.
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
        let (trimmed_messages, trim_disposition) = ctx.core.retention.apply_budget(
            &ctx.core.token_budget,
            &ctx.messages,
            tool_def_tokens,
            crate::agent::retention::BudgetMode::Normal {
                turn_count: ctx.turn_count,
            },
            frozen_prefix,
        );
        let prefix_preserved = matches!(
            trim_disposition,
            crate::agent::token_budget::PrefixTrimDisposition::Preserved
        );
        if !prefix_preserved && frozen_prefix > 0 {
            let rotated = invalidate_prompt_cache_for_rewrite(ctx, CacheResetReason::Trim);
            warn!(
                session = %ctx.session_key,
                frozen_prefix,
                before_messages = ctx.messages.len(),
                after_messages = trimmed_messages.len(),
                rotated,
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
        // append-only. Cloud models retain proactive retrieval.
        let local_retrieval_opt_in =
            proactive_grounding_preserves_prefix_cache(ctx.core.mode().is_local());
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
            // Emergency mode is conservative (ignores age, trims more aggressively).
            let (trimmed_messages, trim_disposition) = ctx.core.retention.apply_budget(
                &ctx.core.token_budget,
                &ctx.messages,
                0,
                crate::agent::retention::BudgetMode::Emergency,
                frozen_prefix,
            );
            let prefix_preserved = matches!(
                trim_disposition,
                crate::agent::token_budget::PrefixTrimDisposition::Preserved
            );
            if !prefix_preserved && frozen_prefix > 0 {
                let rotated =
                    invalidate_prompt_cache_for_rewrite(ctx, CacheResetReason::EmergencyTrim);
                warn!(
                    session = %ctx.session_key,
                    frozen_prefix,
                    before_messages = ctx.messages.len(),
                    after_messages = trimmed_messages.len(),
                    rotated,
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

        // Router-first preflight for strict trio mode. The router can only
        // strip tools when trio is active, so the passthrough-restore below
        // is only meaningful then. When trio is off (the common local
        // single-model setup) the preflight is a pure passthrough and the
        // restore must NOT run — `tool_defs` is empty here only because the
        // lease forced a text-only call, and restoring would undo lease
        // enforcement (the 20260729_155613_2b65f0 failure). Gating the
        // whole block makes trio-off absent from the hot path instead of
        // present-but-inert.
        let trio_active = ctx.core.mode().is_local()
            && ctx.core.tool_delegation_config.strict_no_tools_main();
        if trio_active {
            match crate::agent::router::router_preflight(ctx, self.health_registry.as_deref()).await
            {
                crate::agent::router::PreflightResult::Continue => {
                    return StepResult::Done(IterationOutcome::Continue);
                }
                crate::agent::router::PreflightResult::Break(msg) => {
                    return StepResult::Done(IterationOutcome::Finished(msg));
                }
                crate::agent::router::PreflightResult::Passthrough => {
                    // Restore tool definitions removed by strict trio routing
                    // when the router elects to fall back to the main model.
                    if tool_defs.is_empty() && !saved_tool_defs.is_empty() {
                        debug!("router_preflight=Passthrough — restoring tool_defs for main model fallback");
                        tool_defs = saved_tool_defs;
                    }
                }
            }
        }

        ctx.advertised_tool_names = Some(advertised_tool_names(&tool_defs));

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
        // One protocol for local and cloud: hot tools have native schemas and
        // the proxy exposes the long tail. Bonsai otherwise mixes the proxy
        // envelope with native calls (for example exec(args={command: ...})).
        // The larger schema prefix is stable and retained by local backends;
        // paying it once is cheaper than repeated malformed generations.
        let mut tool_defs = ctx.tools.get_core_plus_proxy_definitions();
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
        if ctx.core.mode().is_local() && ctx.core.tool_delegation_config.strict_no_tools_main() {
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
                ctx.core.tool_delegation_config.strict_no_tools_main(),
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
    /// checkpoint/reset. Otherwise leave the background result pending.
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

        if !pending.matches_snapshot_prefix(&ctx.messages) {
            warn!(
                session = %ctx.session_key,
                snapshot_messages = pending.watermark(),
                live_messages = ctx.messages.len(),
                "stale_lcm_checkpoint_discarded"
            );
            return false;
        }

        let frozen_prefix = ctx
            .counters
            .prompt_cache_watermark
            .lock()
            .get(&ctx.session_key)
            .copied()
            .unwrap_or(0);
        let rewrites_prompt = pending.result.messages != pending.snapshot;
        if rewrites_prompt && frozen_prefix > 0 && !allow_checkpoint {
            debug!(
                session = %ctx.session_key,
                frozen_prefix,
                compacted_messages = pending.result.messages.len(),
                watermark = pending.watermark(),
                "lcm_compaction_deferred_for_prompt_cache"
            );
            *guard = Some(pending);
            return false;
        }
        // `pending` was moved out of the slot; the guard is only needed for the
        // deferral put-back above. Drop it so the rewrite below can take
        // `&mut ctx` (invalidate_prompt_cache_for_rewrite) without borrowing it.
        drop(guard);

        // Always clear the prompt fingerprint/watermark before
        // `apply_compaction_result` rewrites the wire. The old gate
        // (`if rewrites_prompt`) compared the compaction RESULT to the
        // SNAPSHOT — but `apply_compaction_result` rewrites the LIVE
        // wire against the snapshot, which can diverge from the result
        // comparison when the wire grew between trigger and install.
        // The result: compaction installs without firing
        // `prompt_cache_watermark_invalidated_by_lcm_checkpoint`, and
        // the next iteration's fingerprint check sees the rewritten
        // prefix as an unsanctioned `Diverged` (cache reset, ~60s
        // re-prefill on Higgs). Clearing unconditionally is safe —
        // worst case the fingerprint was already cleared and this is a
        // no-op; the next call recomputes it as `First` instead of
        // `AppendOnly` (a one-time cheap re-prefill, not an unsanctioned
        // divergence).
        let rotated = invalidate_prompt_cache_for_rewrite(ctx, CacheResetReason::LcmCheckpoint);
        if rotated {
            warn!(
                session = %ctx.session_key,
                frozen_prefix,
                compacted_messages = pending.result.messages.len(),
                watermark = pending.watermark(),
                "prompt_cache_watermark_invalidated_by_lcm_checkpoint"
            );
        } else {
            // Even without Higgs session rotation, the local fingerprint
            // clear happened. Log at INFO so this path is visible in the
            // daemon log without WARN filtering.
            info!(
                session = %ctx.session_key,
                compacted_messages = pending.result.messages.len(),
                watermark = pending.watermark(),
                "prompt_cache_cleared_for_lcm_checkpoint_no_rotation"
            );
        }

        debug!(
            "Compaction swap: {} msgs -> {} compacted + {} new",
            pending.watermark(),
            pending.result.messages.len(),
            ctx.messages.len().saturating_sub(pending.watermark())
        );
        // Record stats for `/lcm stats`: tokens of the replaced prefix vs its
        // compacted form (estimates, same estimator as budget accounting).
        ctx.counters.record_compaction(
            TokenBudget::estimate_tokens(&pending.snapshot) as u64,
            TokenBudget::estimate_tokens(&pending.result.messages) as u64,
        );
        if !apply_compaction_result(&mut ctx.messages, pending) {
            return false;
        }
        // Compaction rewrites only the in-memory active window. Raw protocol
        // messages remain durable in SQLite and are identified by `_db_id`.
        ctx.new_start = ctx.messages.len();
        ctx.flow.iterations_since_compaction = 0;
        // No ingest bookkeeping needed after the swap: the engine's store is
        // keyed by `_db_id` and ingest is an idempotent upsert, so re-offering
        // already-stored messages is a no-op.
        true
    }

    /// Spawn background compaction when threshold exceeded.
    ///
    /// Uses the LCM engine's control loop with its persisted DAG and
    /// deterministic hard-pressure fallback.
    async fn manage_compaction(&self, ctx: &mut TurnContext, tool_def_tokens: usize) {
        {
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
                if !engines.contains_key(&ctx.session_id) {
                    let config = LcmConfig::from(&self.lcm_config);
                    let engine = LcmEngine::new(config);
                    engines.insert(
                        ctx.session_id.clone(),
                        Arc::new(tokio::sync::Mutex::new(engine)),
                    );
                }
                engines.get(&ctx.session_id).cloned().unwrap()
            };

            // Feed messages into the LCM engine's append-only store.
            //
            // Ingest is an idempotent upsert keyed by `_db_id` (the SQLite
            // rowid): messages already in the store are skipped, and messages
            // not yet persisted this turn carry no `_db_id` and are skipped
            // too — they are always inside the protect window and get
            // ingested next turn, once get_history supplies their rowid.
            // Synthetic scaffolds and summary placeholders are not originals
            // and never enter the store.
            {
                let mut engine = lcm_engine.lock().await;
                for msg in &ctx.messages {
                    if crate::agent::markers::is_synthetic(msg)
                        || msg.get("role").and_then(|r| r.as_str()) == Some("summary")
                    {
                        continue;
                    }
                    let _ = engine.ingest(msg.clone());
                }
            }

            let budget_core = ctx.core.clone();
            let budget = &budget_core.token_budget;
            let (mut action, conv_tokens, available, hard_limit, soft_limit) = {
                let engine = lcm_engine.lock().await;
                let available = budget.available_budget(tool_def_tokens);
                (
                    engine.check_thresholds_with_available(available),
                    engine.conversation_tokens(),
                    available,
                    (available as f64 * engine.tau_hard()) as usize,
                    (available as f64 * engine.tau_soft()) as usize,
                )
            };

            // A soft job may finish after the raw foreground context has
            // crossed the hard threshold. At that point foreground inference
            // waits for it and installs its checkpoint before doing anything
            // else. Each model request has a sliding inactivity deadline, so
            // the whole multi-request compaction must not have a wall clock
            // timeout that can cancel otherwise healthy progress.
            let mut raw_hard = conversation_token_count(&ctx.messages) > hard_limit;
            if raw_hard && ctx.compaction.in_flight.load(Ordering::Acquire) {
                let wait_for_soft_job = async {
                    while ctx.compaction.in_flight.load(Ordering::Acquire) {
                        tokio::time::sleep(Duration::from_millis(10)).await;
                    }
                };
                wait_for_soft_job.await;
            }
            if raw_hard {
                self.install_pending_compaction(ctx, true).await;
                raw_hard = conversation_token_count(&ctx.messages) > hard_limit;
                action = {
                    let engine = lcm_engine.lock().await;
                    engine.check_thresholds_with_available(available)
                };
            }

            let has_pending = ctx.compaction.slot.lock().await.is_some();
            let in_flight = ctx.compaction.in_flight.load(Ordering::Acquire);
            let must_block = raw_hard || action == CompactionAction::Blocking;

            if must_block && !in_flight {
                tracing::info!(
                    compaction_type = "lcm_blocking",
                    msg_count = ctx.messages.len(),
                    conv_tokens,
                    available,
                    hard_limit,
                    soft_limit,
                    "lcm_compaction_triggered"
                );
                ctx.compaction.in_flight.store(true, Ordering::Release);
                let core = ctx.core.clone();
                let session_id = ctx.session_id.clone();
                let messages = ctx.messages.clone();
                // SQLite message_count is a durable, concrete-session sequence.
                // The process-global learning counter remains telemetry only and
                // must not order one session's working-memory checkpoints.
                let session_turn = core
                    .sessions
                    .get_session(&session_id)
                    .await
                    .map_or(0, |session| session.message_count as u64);
                // Signal the TUI before the await: this compaction will block
                // the turn for ~30-90s (spawn compactor + summarize + install).
                // Without this marker the user sees a silent freeze.
                send_compaction_marker(
                    &ctx.text_delta_tx,
                    crate::turn_stream::CompactionStatus::Started {
                        messages: ctx.messages.len() as u32,
                    },
                );
                let pending = execute_lcm_compaction(
                    core,
                    session_id,
                    lcm_engine.clone(),
                    messages,
                    session_turn,
                    CompactionFailureMode::Deterministic,
                )
                .await;
                if let Some(pending) = pending {
                    *ctx.compaction.slot.lock().await = Some(pending);
                }
                ctx.compaction.in_flight.store(false, Ordering::Release);
                self.install_pending_compaction(ctx, true).await;
                // Clear the compaction indicator. The next event (prefill,
                // cache reset, etc.) replaces the Activity row, but in case
                // the turn finishes here we don't want a stuck "compacting".
                send_compaction_marker(
                    &ctx.text_delta_tx,
                    crate::turn_stream::CompactionStatus::Finished,
                );
            } else if action == CompactionAction::Async && !has_pending && !in_flight {
                tracing::info!(
                    compaction_type = "lcm_async",
                    msg_count = ctx.messages.len(),
                    conv_tokens,
                    available,
                    hard_limit,
                    soft_limit,
                    "lcm_compaction_triggered"
                );
                {
                    let mut engine = lcm_engine.lock().await;
                    engine.request_async_compaction();
                }
                let slot = ctx.compaction.slot.clone();
                let in_flight = ctx.compaction.in_flight.clone();
                let session_turn = ctx
                    .core
                    .sessions
                    .get_session(&ctx.session_id)
                    .await
                    .map_or(0, |session| session.message_count as u64);
                let task = execute_lcm_compaction(
                    ctx.core.clone(),
                    ctx.session_id.clone(),
                    lcm_engine.clone(),
                    ctx.messages.clone(),
                    session_turn,
                    CompactionFailureMode::PreserveContext,
                );
                in_flight.store(true, Ordering::Release);
                tokio::spawn(async move {
                    if let Some(pending) = task.await {
                        *slot.lock().await = Some(pending);
                    }
                    in_flight.store(false, Ordering::Release);
                });
            } else if has_pending {
                debug!("LCM compaction deferred: checkpoint is waiting for a safe install");
            }

            // Auto-expand relevant summaries before the LLM call. The system,
            // not the model, decides when to surface older detail. Expansions are
            // APPENDED to the tail (after any frozen cache prefix), so this runs
            // even on warm sessions without invalidating the prompt cache.
            {
                let mut engine = lcm_engine.lock().await;
                if !engine.dag().is_empty() {
                    let expand_t0 = std::time::Instant::now();
                    // Stamp the current turn so freshly-created summaries
                    // become eligible for auto_expand only after
                    // FRESH_SUMMARY_COOLDOWN_TURNS. Without this, a summary
                    // created by compaction at turn N can be reinjected at
                    // turn N+1, undoing the compaction (live failure
                    // 2026-07-27 12:13:06 saw +12463 tokens reinjected 24s
                    // after a successful 12463→1398 compaction).
                    let current_turn = ctx
                        .core
                        .sessions
                        .get_session(&ctx.session_id)
                        .await
                        .map_or(0, |session| session.message_count as u64);
                    engine.set_current_turn(current_turn);
                    // wire_tokens = actual rendered prompt size. Counting the
                    // wire (not the engine's internal active) is what stops
                    // reinjection from pushing the prompt past the active
                    // model's τ_hard context threshold.
                    let wire_tokens = TokenBudget::estimate_tokens(&ctx.rendered_messages);
                    let appended =
                        engine.auto_expand(&ctx.core.token_budget, tool_def_tokens, wire_tokens);
                    tracing::info!(
                        target: "turn_timing",
                        auto_expand_ms = expand_t0.elapsed().as_millis() as u64,
                        summaries = engine.dag().len(),
                        "auto_expand_timing"
                    );
                    if !appended.is_empty() {
                        debug!(
                            session = %ctx.session_key,
                            count = appended.len(),
                            "LCM auto_expand: appended expanded originals to tail"
                        );
                        ctx.messages.extend(appended);
                    }
                }
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
        // Size the response from the actual user request, not from the latest
        // raw message. After a tool call the latest raw message is often a tool
        // result folded through the local protocol, which can wrongly trigger
        // long-form budgets on every post-tool continuation.
        let user_text = ctx.user_content.as_str();
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
        let local_artifact_action = local_artifact_action_for_turn(ctx);
        adaptive_max_tokens_for_artifact_action(
            base,
            had_long,
            user_text,
            recent_tool_calls,
            ctx.core.mode().is_local(),
            local_artifact_action,
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
        let diag_t0 = std::time::Instant::now();
        let prompt_fp = prompt_fingerprint::fingerprint(&messages_for_llm);
        let prompt_msg_count = messages_for_llm.len();
        let tool_def_tokens = TokenBudget::estimate_tool_def_tokens(tool_defs_opt.unwrap_or(&[]));
        let prompt_total_estimate =
            TokenBudget::estimate_tokens(&messages_for_llm).saturating_add(tool_def_tokens);
        ctx.flow.provider_prompt_estimate = Some(prompt_total_estimate);
        let prompt_delta = {
            let store = counters.prompt_fingerprints.lock();
            prompt_fingerprint::compare(store.get(&ctx.session_key), &prompt_fp)
        };
        {
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
                    // After the trim/compaction rotation fixes, every sanctioned
                    // rewrite clears the prompt fingerprint and returns
                    // `PromptDelta::First`; reaching this `Diverged` branch means
                    // the divergence is UNSANCTIONED — a message whose rendered
                    // bytes changed across turns under the same higgs session id
                    // (the `token_mismatch` class). `divergent_message_digest`
                    // names the structural kind so the root cause is obvious.
                    let digest = messages_for_llm
                        .get(first_divergent_msg)
                        .map(divergent_message_digest)
                        .unwrap_or_else(|| "(out of range)".to_string());
                    // Diagnostic: dump the full rendered content of the
                    // divergent message and its neighbors so the exact
                    // byte change is visible in the log.
                    let store = counters.prompt_fingerprints.lock();
                    let prev_fp = store.get(&ctx.session_key);
                    if let Some(prev_fp) = prev_fp {
                        let prev_hash = prev_fp.msg_hash_at(first_divergent_msg);
                        let new_hash = messages_for_llm
                            .get(first_divergent_msg)
                            .map(prompt_fingerprint::hash_value);
                        tracing::warn!(
                            session = %ctx.session_key,
                            at_msg = first_divergent_msg,
                            prev_msgs,
                            new_msgs,
                            prev_hash = ?prev_hash,
                            new_hash = ?new_hash,
                            "divergence_hash_comparison"
                        );
                    }
                    let dump_msg = |idx: usize, label: &str| {
                        if let Some(m) = messages_for_llm.get(idx) {
                            let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("?");
                            let content = m
                                .get("content")
                                .and_then(|c| c.as_str())
                                .unwrap_or("(non-string)");
                            tracing::warn!(
                                session = %ctx.session_key,
                                idx, label, role,
                                content_len = content.len(),
                                content_preview = %content.chars().take(200).collect::<String>(),
                                "divergence_context_dump"
                            );
                        }
                    };
                    dump_msg(first_divergent_msg.saturating_sub(1), "prev_msg");
                    dump_msg(first_divergent_msg, "divergent_msg");
                    dump_msg(first_divergent_msg + 1, "next_msg");
                    // Coarse class for the TUI footer. Extracted from the
                    // divergent message's tags so the user sees
                    // `cache reset · lcm summary @ msg N` instead of the
                    // generic `cache reset · msg N`. Static strings only.
                    let divergent_msg = messages_for_llm.get(first_divergent_msg);
                    let class: &'static str = divergent_msg
                        .map(|m| {
                            if m.get("_lcm_summary")
                                .and_then(|v| v.as_bool())
                                .unwrap_or(false)
                            {
                                "lcm summary"
                            } else if m.get("tool_call_id").is_some() {
                                "tool result"
                            } else if m
                                .get("_synthetic")
                                .and_then(|v| v.as_bool())
                                .unwrap_or(false)
                            {
                                "synthetic"
                            } else if m
                                .get("_cache_replay")
                                .and_then(|v| v.as_bool())
                                .unwrap_or(false)
                            {
                                "cache-replay"
                            } else {
                                match m.get("role").and_then(|r| r.as_str()).unwrap_or("message") {
                                    "assistant" => "assistant",
                                    "user" => "user",
                                    "system" => "system",
                                    _ => "message",
                                }
                            }
                        })
                        .unwrap_or("unknown");
                    tracing::warn!(
                        session = %ctx.session_key,
                        at_msg = first_divergent_msg,
                        prev_msgs,
                        new_msgs,
                        prefill_estimate,
                        class = %class,
                        digest = %digest,
                        "prompt_prefix_diverged — unsanctioned token_mismatch class; server re-prefills past this point"
                    );
                    ControlMarker::CacheStatus(CacheStatus::Diverged {
                        at: first_divergent_msg,
                        prev: prev_msgs,
                        messages: new_msgs,
                        class,
                    })
                    .encode()
                }
                PromptDelta::AppendOnly { added_msgs } => {
                    debug!(
                        session = %ctx.session_key,
                        added_msgs,
                        prefill_estimate,
                        "prompt_append_only"
                    );
                    ControlMarker::CacheStatus(CacheStatus::AppendOnly {
                        added: added_msgs,
                        messages: prompt_msg_count,
                    })
                    .encode()
                }
                PromptDelta::First => ControlMarker::CacheStatus(CacheStatus::First {
                    messages: prompt_msg_count,
                })
                .encode(),
            };
            if let Some(ref delta_tx) = ctx.text_delta_tx {
                let _ = delta_tx.send(cache_marker);
                if prefill_estimate > 0 {
                    let _ = delta_tx
                        .send(ControlMarker::PrefillEstimate(prefill_estimate as u64).encode());
                }
            }
        }
        let unexpected_higgs_divergence = matches!(prompt_delta, PromptDelta::Diverged { .. })
            && ctx.core.mode().is_local()
            && ctx.core.provider.supports_higgs_session_cache();
        if unexpected_higgs_divergence {
            invalidate_prompt_cache_for_rewrite(ctx, CacheResetReason::UnexpectedReplayDivergence);
        }
        // Tool schemas are rendered at the prompt head but deliberately absent
        // from the message fingerprint. A real topology change therefore needs
        // a new Higgs retained-session epoch before deriving the request marker;
        // otherwise Higgs receives changed prefix bytes under the old session.
        let tool_count = tool_defs_opt.map_or(0, |tools| tools.len());
        let new_tool_hash = prompt_fingerprint::hash_tools(tool_defs_opt.unwrap_or(&[]));
        let previous_tool_hash = counters
            .prompt_tool_hashes
            .lock()
            .insert(ctx.session_key.to_string(), new_tool_hash);
        if let Some(previous_tool_hash) =
            previous_tool_hash.filter(|old| *old != new_tool_hash)
        {
            // Invalidation clears every prompt fingerprint, including the tool
            // hash. Do not hold its lock across this call, and reinstall the
            // new topology afterward as the baseline for the next request.
            let rotated = if ctx.core.mode().is_local()
                && ctx.core.provider.supports_higgs_session_cache()
            {
                invalidate_prompt_cache_for_rewrite(ctx, CacheResetReason::ToolTopology)
            } else {
                false
            };
            counters
                .prompt_tool_hashes
                .lock()
                .insert(ctx.session_key.to_string(), new_tool_hash);
            tracing::warn!(
                session = %ctx.session_key,
                tool_count,
                prev_hash = previous_tool_hash,
                new_hash = new_tool_hash,
                rotated,
                "tool_block_changed — rotated retained session before changed prompt head"
            );
        }
        tracing::info!(
            target: "turn_timing",
            prefix_diag_ms = diag_t0.elapsed().as_millis() as u64,
            msg_count = prompt_msg_count,
            "prefix_diag_timing"
        );

        let mut pending_higgs_drop = Vec::new();
        let mut higgs_session_marker = None;
        if ctx.core.mode().is_local() && ctx.core.provider.get_api_base().is_some() {
            let provider_session_id = stable_higgs_session_id(
                &ctx.session_id,
                counters.session_prompt_epoch(&ctx.session_key),
            );
            higgs_session_marker = Some(provider_session_id);
            counters.record_higgs_session_id(&ctx.session_key, provider_session_id);
            pending_higgs_drop = counters.pending_higgs_session_drop_ids(&ctx.session_key);
            attach_higgs_session_marker(&mut messages_for_llm, provider_session_id, &[]);
        }

        let request_hash = crate::agent::prompt_fingerprint::hash_provider_request(
            &messages_for_llm,
            tool_defs_opt.unwrap_or(&[]),
        );
        if ctx
            .flow
            .provider_request
            .admit(request_hash, ctx.flow.tool_rounds_completed)
            == ProviderRequestAdmission::ForceCheckpoint
        {
            clear_prompt_cache_state(ctx);
            send_cache_reset_marker(&ctx.text_delta_tx, CacheResetReason::StalledProviderRequest);
            ctx.messages
                .push(crate::agent::markers::scaffold_user(
                    "[system] Context checkpoint: the latest tool round did not change the provider request. Re-read the newest tool result and continue from it."
                ));
            ctx.rendered_messages.clear();
            warn!(
                session = %ctx.session_key,
                request_hash,
                tool_round = ctx.flow.tool_rounds_completed,
                "provider_request_stalled_after_tool_progress"
            );
            return StepResult::Done(IterationOutcome::Continue);
        }

        // Signal watchdog only after request admission: a rejected no-progress
        // call never marks inference active or starts latency telemetry.
        counters.mark_inference_started();
        ctx.flow.llm_call_start = Some(std::time::Instant::now());
        ctx.flow.ttft_ms = None;

        if let Some(provider_session_id) = higgs_session_marker {
            let drop_session_ids = pending_higgs_drop.iter().copied().collect::<Vec<_>>();
            attach_higgs_session_marker(
                &mut messages_for_llm,
                provider_session_id,
                &drop_session_ids,
            );
        }

        let no_progress_timeout = local_stream_no_progress_timeout(ctx);
        let response = if let Some(ref delta_tx) = ctx.text_delta_tx {
            // Streaming path: forward text deltas to the REPL/voice renderer as
            // they arrive so the answer streams live — tokens appear as they are
            // generated and feed sentence-by-sentence into streaming TTS. Brief
            // pre-tool prose ("let me check…") streams as visible progress; tool
            // parameters and output never reach this channel (they are emitted as
            // ToolEvents and rendered separately).
            let backend_activity = BackendActivityHeartbeat::start(
                Some(delta_tx.clone()),
                no_progress_timeout.is_some(),
            );
            let stream_call = ctx.core.provider.chat_stream(
                &messages_for_llm,
                tool_defs_opt,
                Some(&ctx.core.model),
                max_tokens,
                ctx.core.temperature,
                thinking_budget,
                None,
            );
            let mut stream = if let Some(timeout) = no_progress_timeout {
                match tokio::time::timeout(timeout, stream_call).await {
                    Ok(Ok(s)) => s,
                    Ok(Err(e)) => {
                        return Self::handle_llm_error(e, ctx, counters, "llm_stream_call").await;
                    }
                    Err(_) => {
                        counters.mark_inference_finished();
                        let detail = local_no_stream_headers_error(timeout);
                        error!(
                            model = %ctx.core.model,
                            timeout_secs = timeout.as_secs(),
                            "llm_stream_headers_timeout"
                        );
                        emit_stream_abort_metrics(ctx, &detail);
                        return StepResult::Done(IterationOutcome::Error(detail));
                    }
                }
            } else {
                match stream_call.await {
                    Ok(s) => s,
                    Err(e) => {
                        return Self::handle_llm_error(e, ctx, counters, "llm_stream_call").await;
                    }
                }
            };
            if let Some(activity) = backend_activity.as_ref() {
                activity.mark_progress(BackendActivity::Prefill);
            }

            let mut streamed_response = None;
            let mut stream_progress = LocalStreamProgress::new();
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
                    _ = async {
                        if let Some(timeout) = no_progress_timeout {
                            tokio::time::sleep(timeout).await;
                        } else {
                            std::future::pending::<()>().await;
                        }
                    } => {
                        counters.mark_inference_finished();
                        let timeout = no_progress_timeout.expect("timeout branch is disabled when None");
                        let detail = local_no_stream_progress_error(timeout);
                        error!(
                            model = %ctx.core.model,
                            timeout_secs = timeout.as_secs(),
                            "llm_stream_no_progress_timeout"
                        );
                        drop(stream);
                        emit_stream_abort_metrics(ctx, &detail);
                        return StepResult::Done(IterationOutcome::Error(detail));
                    }
                    chunk = stream.rx.recv() => {
                        match chunk {
                            Some(StreamChunk::TransportProgress) => {
                                // Receiving this branch restarts the per-loop
                                // no-progress sleep. It proves transport
                                // liveness only: do not mark TTFT, decoding, or
                                // artifact tool-payload progress.
                            }
                            Some(StreamChunk::ThinkingDelta(delta)) => {
                                if let Some(activity) = backend_activity.as_ref() {
                                    let phase = stream_progress.on_text_or_thinking_delta();
                                    activity.mark_progress(phase);
                                }
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
                                if let Some(activity) = backend_activity.as_ref() {
                                    let phase = stream_progress.on_text_or_thinking_delta();
                                    activity.mark_progress(phase);
                                }
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
                                let phase = stream_progress.on_tool_call_delta();
                                if let Some(activity) = backend_activity.as_ref() {
                                    activity.mark_progress(phase);
                                }
                                // Pure tool-call responses have no text/thinking
                                // deltas — the first tool-call fragment marks
                                // end of prefill for TTFT.
                                ctx.flow.mark_first_token();
                            }
                            Some(StreamChunk::PrefillProgress { processed, total }) => {
                                // Prefill still running — not a token, so no
                                // mark_first_token. Forward to the REPL spinner
                                // as a control marker. Send it before the
                                // backend phase update so completed-prefill
                                // progress cannot overwrite the post-prefill
                                // tool-payload status in the TUI.
                                let _ =
                                    delta_tx.send(
                                        ControlMarker::PrefillProgress { processed, total }
                                            .encode(),
                                    );
                                if let Some(activity) = backend_activity.as_ref() {
                                    let phase =
                                        stream_progress.on_prefill_progress(processed, total);
                                    activity.mark_progress(phase);
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
                    counters.mark_inference_finished();
                    // Stream ended without Done — either cancelled or genuine error.
                    if ctx.is_cancelled() {
                        // Cancelled mid-stream — exit cleanly.
                        emit_stream_abort_metrics(
                            ctx,
                            "The stream was cancelled before the backend returned a final response.",
                        );
                        return StepResult::Done(IterationOutcome::Finished(String::new()));
                    }
                    error!("LLM stream ended without Done");
                    emit_stream_abort_metrics(
                        ctx,
                        "The LLM stream ended without a final response.",
                    );
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

        if !pending_higgs_drop.is_empty() {
            counters.clear_pending_higgs_session_drop_ids(&ctx.session_key, &pending_higgs_drop);
        }

        // Inference complete — allow watchdog health checks again.
        counters.mark_inference_finished();

        // Only valid completed provider calls seed Nanobot's local prompt-cache
        // model. A 600s zero-token stall can arrive as `finish_reason=stop`,
        // so share the metrics classifier instead of keying only on
        // `finish_reason=error`.
        if Self::response_status(&response) == "ok" {
            counters
                .prompt_fingerprints
                .lock()
                .insert(ctx.session_key.clone(), prompt_fp);
            counters
                .prompt_cache_watermark
                .lock()
                .insert(ctx.session_key.clone(), ctx.messages.len());
        }

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
        // Reset the per-round dispatched-key record; set again only when tools
        // actually execute so a no-tool round can't leave a stale key behind.
        ctx.flow.last_round_keys.clear();

        let routed_tool_calls = match crate::agent::router::route_tool_calls(
            ctx,
            response.content.as_deref(),
            response.tool_calls.clone(),
        )
        .await
        {
            crate::agent::router::RouteResult::Continue => {
                ctx.emit_pending_request_metrics(0);
                ctx.flow.tool_rounds_completed = ctx.flow.tool_rounds_completed.saturating_add(1);
                return StepResult::Done(IterationOutcome::Continue);
            }
            crate::agent::router::RouteResult::Break(msg) => {
                ctx.emit_pending_request_metrics(0);
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

        // The assistant emitted one protocol batch, so the lease makes one
        // atomic decision before any member can execute. Partial execution
        // would leave the model believing the whole batch ran and makes retry
        // semantics unsafe for side-effect tools.
        let batch_count = u32::try_from(routed_tool_calls.len()).unwrap_or(u32::MAX);
        if let BatchAdmission::Rejected { remaining } =
            ctx.flow.lease.admit_batch(batch_count)
        {
            let tool_calls: Vec<Value> = routed_tool_calls
                .iter()
                .map(|tool_call| tool_call.to_openai_json())
                .collect();
            ContextBuilder::add_assistant_message(
                &mut ctx.messages,
                response.content.as_deref(),
                Some(&tool_calls),
            );
            for tool_call in &routed_tool_calls {
                tracing::info!(
                    session = %ctx.session_key,
                    tool = %tool_call.name,
                    batch_count,
                    remaining,
                    "tool_lease_rejected_batch"
                );
                let receipt = format!(
                    "lease exhausted: {} was not executed — this batch requested {} calls with {} remaining. Write a renewal checkpoint before requesting another tool in a new turn.",
                    tool_call.name, batch_count, remaining
                );
                ContextBuilder::add_tool_result_immutable_with_status(
                    &mut ctx.messages,
                    &tool_call.id,
                    &tool_call.name,
                    &receipt,
                    false,
                );
            }
            if !ctx.persist_pending_protocol_messages().await {
                ctx.emit_pending_request_metrics(0);
                return StepResult::Done(IterationOutcome::Error(
                    "[Session Error] Rejected tool batch could not be recorded atomically; no tools were executed."
                        .to_string(),
                ));
            }
            ctx.emit_pending_request_metrics(0);
            return StepResult::Done(IterationOutcome::Finished(
                LEASE_OVER_BUDGET_FINAL.to_string(),
            ));
        }
        // Snapshot the dispatched tool-call keys for the repeated-call breaker.
        let dispatched_keys: Vec<String> = routed_tool_calls
            .iter()
            .map(|tc| crate::agent::tool_runner::normalize_call_key(&tc.name, &tc.arguments))
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
                .any(|tc| crate::agent::tool_engine::requires_result_report(&tc.name));
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

        let tool_entries_before = ctx.turn_tool_entries.len();
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
                // Stash invariant violation (Hole 1): the immutable store
                // rejected a write. Fail the turn with the infra error — never
                // re-run a side-effect tool, never show a raw body.
                if let Some(e) = ctx.flow.infra_error.take() {
                    return StepResult::Done(IterationOutcome::Error(e));
                }
                // Delegation handled execution — continue the main loop.
                ctx.emit_pending_request_metrics(routed_tool_calls.len() as u32);
                ctx.flow.tool_rounds_completed = ctx.flow.tool_rounds_completed.saturating_add(1);
                ctx.flow.last_round_keys = dispatched_keys.clone();
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
        // Stash invariant violation (Hole 1): the immutable store rejected a
        // write during inline execution. Fail the turn with the infra error —
        // never re-run a side-effect tool, never show a raw body.
        if let Some(e) = ctx.flow.infra_error.take() {
            return StepResult::Done(IterationOutcome::Error(e));
        }
        let executed = ctx
            .turn_tool_entries
            .len()
            .saturating_sub(tool_entries_before) as u32;
        ctx.emit_pending_request_metrics(executed);
        ctx.flow.tool_rounds_completed = ctx.flow.tool_rounds_completed.saturating_add(1);

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

        ctx.flow.last_round_keys = dispatched_keys.clone();
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
//   :900   — `select_tool_definitions`                                    → Lean production surface
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
        advance_response_boundary, attach_higgs_session_marker, divergent_message_digest,
        proactive_grounding_preserves_prefix_cache, ResponseBoundary,
    };
    use crate::agent::agent_core::{stable_higgs_session_id, RuntimeCounters};
    use crate::config::schema::CircuitBreakerConfig;
    use serde_json::json;

    /// The divergence diagnostic must name the divergent message's STRUCTURAL
    /// kind, not just its role, so the cache-busting class is identifiable from
    /// the WARN line alone (this is how we pinpoint the higgs `token_mismatch`
    /// cause without guessing). Every sanctioned rewrite clears the fingerprint
    /// → `PromptDelta::First`, so a `Diverged` is by definition unsanctioned.
    #[test]
    fn test_divergent_message_digest_classifies_structural_kind() {
        // Plain persisted user turn.
        let plain = json!({"role": "user", "content": "what is 2+2", "_db_id": 7});
        assert_eq!(
            divergent_message_digest(&plain),
            "[user] (persisted) what is 2+2"
        );

        // Transient synthetic user (the prime suspect: sent but not replayed).
        let synthetic = json!({"role": "user", "content": "grounding nudge", "_synthetic": true});
        assert_eq!(
            divergent_message_digest(&synthetic),
            "[user] (synthetic) grounding nudge"
        );

        // Cache-replay scaffold (persisted synthetic).
        let scaffold = json!({
            "role": "user",
            "content": "[system] checkpoint",
            "_synthetic": true,
            "_cache_replay": true,
            "_db_id": 9,
        });
        assert_eq!(
            divergent_message_digest(&scaffold),
            "[user] (synthetic,cache-replay,persisted) [system] checkpoint"
        );

        // Tool result (rendered role may differ from raw — a render-instability
        // suspect).
        let tool_result = json!({
            "role": "tool",
            "tool_call_id": "tc_1",
            "name": "read_file",
            "content": "file body",
        });
        assert_eq!(
            divergent_message_digest(&tool_result),
            "[tool] (tool-result) file body"
        );

        // LCM summary (engine-owned view, renders as user; re-summarization
        // suspect).
        let summary = json!({"role": "user", "content": "prior turns...", "_lcm_summary": true});
        assert_eq!(
            divergent_message_digest(&summary),
            "[user] (lcm-summary) prior turns..."
        );

        // Long content is truncated to a readable snippet.
        let long = json!({"role": "assistant", "content": "x".repeat(120)});
        let digest = divergent_message_digest(&long);
        assert!(
            digest.starts_with("[assistant] "),
            "no kind tag for a plain assistant: {digest}"
        );
        assert!(
            digest.matches('x').count() <= 70,
            "snippet must be truncated: {digest}"
        );
    }

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

    #[test]
    fn test_higgs_marker_sends_drop_id_queued_before_session_rollover() {
        let counters = RuntimeCounters::new_with_config(16384, &CircuitBreakerConfig::default());
        let session_key = "cli:test";
        let original_drop_id = stable_higgs_session_id("sqlite-session-before-clear", 0);
        let current_session_id = stable_higgs_session_id("sqlite-session-after-clear", 1);

        counters.record_higgs_session_id(session_key, original_drop_id);
        counters.reset_session_prompt_state(session_key);
        counters.record_higgs_session_id(session_key, current_session_id);

        let mut messages = vec![json!({"role": "system", "content": "system"})];
        attach_higgs_session_marker(
            &mut messages,
            current_session_id,
            &counters.pending_higgs_session_drop_ids(session_key),
        );

        assert_eq!(
            messages[0][crate::providers::openai_compat::NANOBOT_HIGGS_SESSION_ID_FIELD],
            json!(current_session_id)
        );
        assert_eq!(
            messages[0][crate::providers::openai_compat::NANOBOT_HIGGS_DROP_SESSION_ID_FIELD],
            json!(original_drop_id)
        );
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
            proactive_grounding_preserves_prefix_cache(false),
            "cloud grounding is not governed by the local prefix-cache tradeoff"
        );
        assert!(
            !proactive_grounding_preserves_prefix_cache(true),
            "local default skips synthetic per-turn grounding to avoid msg-N cache resets"
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

#[cfg(test)]
mod cache_pressure_tests {
    use super::{
        should_allow_checkpoint, should_inject_heartbeat_grounding, ProviderRequestAdmission,
        ProviderRequestState,
    };

    #[test]
    fn identical_provider_request_after_tool_progress_forces_checkpoint() {
        let mut state = ProviderRequestState::default();
        assert_eq!(state.admit(42, 0), ProviderRequestAdmission::Proceed);
        assert_eq!(
            state.admit(42, 0),
            ProviderRequestAdmission::Proceed,
            "an ordinary retry with no intervening tool round remains valid"
        );
        assert_eq!(
            state.admit(42, 1),
            ProviderRequestAdmission::ForceCheckpoint,
            "new tool progress must change the exact provider request"
        );
        assert_eq!(state.admit(43, 1), ProviderRequestAdmission::Proceed);
    }

    // -- should_inject_heartbeat_grounding --------------------------------

    #[test]
    fn heartbeat_grounding_skipped_on_local() {
        assert!(!should_inject_heartbeat_grounding(8, 8, 0.5, true));
    }

    #[test]
    fn heartbeat_grounding_allowed_on_cloud() {
        assert!(should_inject_heartbeat_grounding(8, 8, 0.5, false));
    }

    #[test]
    fn heartbeat_grounding_respects_cadence() {
        assert!(!should_inject_heartbeat_grounding(4, 8, 0.5, false));
    }

    #[test]
    fn heartbeat_grounding_respects_pressure_ceiling() {
        // Above 0.85 pressure, should_ground already returns false.
        assert!(!should_inject_heartbeat_grounding(8, 8, 0.9, false));
    }

    // -- should_allow_checkpoint ------------------------------------------

    #[test]
    fn checkpoint_deferred_below_tau_hard() {
        assert!(!should_allow_checkpoint(0.50, 0.85));
        assert!(!should_allow_checkpoint(0.84, 0.85));
    }

    #[test]
    fn checkpoint_forced_at_tau_hard_boundary() {
        // Exactly tau_hard: accept the re-prefill to break the grow-forever
        // deadlock where warm-cache deferral starves compaction entirely.
        assert!(should_allow_checkpoint(0.85, 0.85));
    }

    #[test]
    fn checkpoint_forced_above_tau_hard() {
        assert!(should_allow_checkpoint(0.90, 0.85));
        assert!(should_allow_checkpoint(1.00, 0.85));
    }

    #[test]
    fn checkpoint_is_not_forced_before_model_context_pressure() {
        assert!(!should_allow_checkpoint(0.40, 0.85));
    }
}
