// Error-protocol layer-3 backlog (docs/research/2026-08-06-error-conventions-and-host-bridge.md §3.6):
// the deny regime in Cargo.toml is live; this module still carries pre-existing
// violations of the lints below. Remove this allow as the module migrates onto
// the regime.
// Tracking: docs/error-protocol-backlog.md
#![allow(
    clippy::as_conversions,
    clippy::indexing_slicing,
    clippy::shadow_reuse,
    clippy::shadow_unrelated
)]
//! `AgentLoopShared` struct, supporting types, and the `impl AgentLoopShared` block
//! containing the main agent loop step methods.
//!
//! Extracted from `agent_loop.rs` as a `#[path]` submodule.

#![allow(clippy::disallowed_types)] // anyhow is the app convention — the ban targets tool boundaries (error protocol §2.5)
use std::collections::{HashMap, HashSet};
use std::sync::atomic::Ordering;
use std::sync::Arc;

use serde_json::{json, Value};
use tokio::sync::mpsc::UnboundedSender;
use tokio::sync::Mutex;
use tracing::{debug, error, info, instrument, warn};

use crate::agent::agent_loop::heuristics::{
    adaptive_max_tokens_for_artifact_action, evaluate_repeated_tool_round, RepeatBreakerAction,
};
use crate::agent::anti_drift;
use crate::agent::audit::{AuditLog, ToolEvent};
use crate::agent::context::ContextBuilder;
use crate::agent::lcm::{
    AutoExpansionCandidate, CompactionAction, CompactionFailureMode, LcmConfig, LcmEngine,
};
use crate::agent::lease::Lease;
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
use crate::errors::{
    classify_retained_session_error, is_retryable_provider_error, RetainedSessionErrorKind,
};
use crate::providers::base::{LLMResponse, StreamChunk, ToolChoice};
use crate::session::db::{ModelCallPurpose, RecordedProviderRequest, RecordedProviderResponse};

use crate::agent::agent_core::{
    append_to_system_prompt, apply_compaction_result, ExpansionCheckpoint, PendingCompaction,
    RuntimeCounters, SessionRetirement, SharedCoreHandle, SwappableCore, ToolPresentationMode,
};

use super::{last_user_message, render_via_protocol, should_strip_tools_for_trio};

// `response` is a sibling module declared in `mod.rs`. RetryState is re-exported
// from there; we just need a local alias for the field type below.
use super::response::RetryState;
use crate::turn_stream::{BackendActivity, CacheResetReason, CacheStatus, ControlMarker};

use super::budget::{
    advertised_tool_names, attach_higgs_session_control, clear_prompt_cache_state,
    conversation_token_count, divergent_message_digest, invalidate_prompt_cache_for_rewrite,
    proactive_grounding_preserves_prefix_cache, send_cache_reset_marker, send_compaction_marker,
    send_retract_reply_marker, should_allow_checkpoint, should_inject_heartbeat_grounding,
    strip_higgs_session_lease_control,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ProviderRequestRetryPolicy {
    Standard,
    OneShotLease,
}

enum ForcedToolRecoveryOutcome {
    Response(LLMResponse),
    ProviderError {
        original: LLMResponse,
        error: anyhow::Error,
    },
}

impl ProviderRequestRetryPolicy {
    fn allows_retry(self) -> bool {
        matches!(self, Self::Standard)
    }
}

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
    /// Idle-window agency config + inbound tracker (gateway-only feature;
    /// default-disabled). See `agent::idle`.
    pub(crate) idle: crate::agent::idle::IdleRuntime,
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
    /// Set during pre-call accounting and consumed only after foreground work ends.
    pub(crate) soft_compaction_requested: bool,
    /// Planned LCM nodes carried by the current provider request. They remain
    /// eligible until that request completes successfully.
    pub(crate) staged_auto_expansion: Option<AppliedAutoExpansion>,
    /// Higgs cache route selected once for this turn and retained across every
    /// tool iteration until terminal cleanup.
    pub(crate) higgs_session_route: HiggsSessionRoute,
    /// Synchronous fail-safe for cancellation or task abort after retained
    /// preflight. Normal terminal/fallback paths disarm it after explicit
    /// checkpoint retirement.
    pub(crate) retained_route_cleanup: RetainedRouteCleanupGuard,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AutoExpansionMaterializationKind {
    ExactCheckpoint,
    FlattenedFallback,
}

struct AutoExpansionMaterialization {
    messages: Vec<Value>,
    kind: AutoExpansionMaterializationKind,
}

pub(crate) struct AppliedAutoExpansion {
    logical_messages: Vec<Value>,
    rendered_messages: Vec<Value>,
    node_ids: Vec<usize>,
    exact_count: usize,
    estimated_added_tokens: usize,
    retained_plan: Option<RetainedExpansionPlan>,
}

#[derive(Clone)]
struct RetainedExpansionPlan {
    checkpoint: ExpansionCheckpoint,
    compacted_prefix: Vec<Value>,
    fallback_prefix: Option<Vec<Value>>,
    exact_prefix_len: usize,
}

impl AppliedAutoExpansion {
    fn retained_plan(&self) -> Option<RetainedExpansionPlan> {
        self.retained_plan.clone()
    }

    fn use_flattened_fallback(&mut self, protocol: &dyn ConversationProtocol) -> bool {
        let Some(plan) = self.retained_plan.take() else {
            return false;
        };
        let Some(fallback_prefix) = plan.fallback_prefix else {
            return false;
        };
        self.logical_messages = fallback_prefix;
        self.rendered_messages = render_via_protocol(protocol, &self.logical_messages);
        self.exact_count = 0;
        true
    }
}

struct CommittedAutoExpansion(AppliedAutoExpansion);

#[derive(Clone, Default)]
pub(crate) struct PromptCacheSnapshot {
    fingerprint: Option<crate::agent::prompt_fingerprint::PromptFingerprint>,
    watermark: Option<usize>,
    route_identity: (Option<u64>, u64, Option<u64>),
}

impl PromptCacheSnapshot {
    fn capture(counters: &RuntimeCounters, session_key: &str) -> Self {
        let _transition = counters.lock_prompt_cache_transition();
        let route_identity = counters.prompt_cache_route_identity(session_key);
        let fingerprints = counters.prompt_fingerprints.lock();
        let watermarks = counters.prompt_cache_watermark.lock();
        Self {
            fingerprint: fingerprints.get(session_key).cloned(),
            watermark: watermarks.get(session_key).copied(),
            route_identity,
        }
    }

    fn capture_and_clear(counters: &RuntimeCounters, session_key: &str) -> Self {
        let _transition = counters.lock_prompt_cache_transition();
        let route_identity = counters.prompt_cache_route_identity(session_key);
        let mut fingerprints = counters.prompt_fingerprints.lock();
        let mut watermarks = counters.prompt_cache_watermark.lock();
        Self {
            fingerprint: fingerprints.remove(session_key),
            watermark: watermarks.remove(session_key),
            route_identity,
        }
    }

    fn restore(self, counters: &RuntimeCounters, session_key: &str) {
        self.restore_inner(counters, session_key, || {});
    }

    fn restore_inner<F>(self, counters: &RuntimeCounters, session_key: &str, observe_identity: F)
    where
        F: FnOnce(),
    {
        let _transition = counters.lock_prompt_cache_transition();
        if counters.prompt_cache_route_identity(session_key) != self.route_identity {
            return;
        }
        observe_identity();
        let mut fingerprints = counters.prompt_fingerprints.lock();
        let mut watermarks = counters.prompt_cache_watermark.lock();
        match self.fingerprint {
            Some(fingerprint) => {
                fingerprints.insert(session_key.to_string(), fingerprint);
            }
            None => {
                fingerprints.remove(session_key);
            }
        }
        match self.watermark {
            Some(watermark) => {
                watermarks.insert(session_key.to_string(), watermark);
            }
            None => {
                watermarks.remove(session_key);
            }
        }
    }

    #[cfg(test)]
    fn restore_observed<F>(self, counters: &RuntimeCounters, session_key: &str, observe_identity: F)
    where
        F: FnOnce(),
    {
        self.restore_inner(counters, session_key, observe_identity);
    }
}

#[derive(Default)]
pub(crate) struct RetainedRouteCleanupGuard {
    armed: Option<(
        Arc<RuntimeCounters>,
        String,
        ExpansionCheckpoint,
        PromptCacheSnapshot,
    )>,
}

impl RetainedRouteCleanupGuard {
    fn arm(
        &mut self,
        counters: Arc<RuntimeCounters>,
        session_key: String,
        checkpoint: ExpansionCheckpoint,
        active_prompt_cache: PromptCacheSnapshot,
    ) {
        self.armed = Some((counters, session_key, checkpoint, active_prompt_cache));
    }

    fn disarm(&mut self) {
        self.armed = None;
    }
}

impl Drop for RetainedRouteCleanupGuard {
    fn drop(&mut self) {
        let Some((counters, session_key, checkpoint, active_prompt_cache)) = self.armed.take()
        else {
            return;
        };
        active_prompt_cache.restore(&counters, &session_key);
        counters.discard_expansion_checkpoint(
            &session_key,
            checkpoint.old_higgs_session_id,
            checkpoint.summary_node_id,
        );
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ActiveCompactedRoute {
    Active,
    Fallback,
    OverflowSummary,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RetainedExpansionFailure {
    Unavailable,
    ContextOverflow,
}

/// One cache route for the whole foreground turn. The retained variant keeps
/// both active fallbacks so a later tool-iteration failure can replace the
/// exact prefix without disturbing assistant/tool messages appended after it.
pub(crate) enum HiggsSessionRoute {
    ActiveCompacted {
        route: ActiveCompactedRoute,
    },
    RetainedExpansion {
        checkpoint: ExpansionCheckpoint,
        compacted_prefix: Vec<Value>,
        fallback_prefix: Option<Vec<Value>>,
        exact_prefix_len: usize,
        expansion_published: bool,
        active_prompt_cache: PromptCacheSnapshot,
    },
}

impl Default for HiggsSessionRoute {
    fn default() -> Self {
        Self::ActiveCompacted {
            route: ActiveCompactedRoute::Active,
        }
    }
}

impl HiggsSessionRoute {
    fn retained_expansion(
        checkpoint: ExpansionCheckpoint,
        compacted_prefix: Vec<Value>,
        fallback_prefix: Option<Vec<Value>>,
        exact_prefix_len: usize,
        active_prompt_cache: PromptCacheSnapshot,
    ) -> Self {
        Self::RetainedExpansion {
            checkpoint,
            compacted_prefix,
            fallback_prefix,
            exact_prefix_len,
            expansion_published: false,
            active_prompt_cache,
        }
    }

    pub(super) fn cache_route(&self) -> &'static str {
        match self {
            Self::ActiveCompacted {
                route: ActiveCompactedRoute::Active,
            } => "active",
            Self::ActiveCompacted {
                route: ActiveCompactedRoute::Fallback | ActiveCompactedRoute::OverflowSummary,
            } => "fallback",
            Self::RetainedExpansion { .. } => "retained_expansion",
        }
    }

    fn mark_expansion_published(&mut self) {
        if let Self::RetainedExpansion {
            expansion_published,
            ..
        } = self
        {
            *expansion_published = true;
        }
    }

    fn permits_auto_expansion(&self) -> bool {
        matches!(
            self,
            Self::ActiveCompacted {
                route: ActiveCompactedRoute::Active | ActiveCompactedRoute::Fallback
            }
        )
    }

    fn retained_checkpoint(&self) -> Option<&ExpansionCheckpoint> {
        match self {
            Self::RetainedExpansion { checkpoint, .. } => Some(checkpoint),
            Self::ActiveCompacted { .. } => None,
        }
    }

    fn take_retained_checkpoint(&mut self) -> Option<(ExpansionCheckpoint, PromptCacheSnapshot)> {
        let retained = std::mem::take(self);
        match retained {
            Self::RetainedExpansion {
                checkpoint,
                active_prompt_cache,
                ..
            } => Some((checkpoint, active_prompt_cache)),
            active => {
                *self = active;
                None
            }
        }
    }

    fn fallback_from_retained(
        &mut self,
        failure: RetainedExpansionFailure,
        logical_messages: &mut Vec<Value>,
        rendered_messages: &mut Vec<Value>,
        protocol: &dyn ConversationProtocol,
    ) -> Option<(ExpansionCheckpoint, PromptCacheSnapshot)> {
        let retained = std::mem::take(self);
        let Self::RetainedExpansion {
            checkpoint,
            compacted_prefix,
            fallback_prefix,
            exact_prefix_len,
            expansion_published,
            active_prompt_cache,
        } = retained
        else {
            return None;
        };
        let use_flattened =
            matches!(failure, RetainedExpansionFailure::Unavailable) && fallback_prefix.is_some();
        *self = Self::ActiveCompacted {
            route: if use_flattened {
                ActiveCompactedRoute::Fallback
            } else {
                ActiveCompactedRoute::OverflowSummary
            },
        };
        if expansion_published {
            let tail = logical_messages
                .get(exact_prefix_len..)
                .unwrap_or_default()
                .to_vec();
            *logical_messages = if use_flattened {
                fallback_prefix.unwrap_or(compacted_prefix)
            } else {
                compacted_prefix
            };
            logical_messages.extend(tail);
            *rendered_messages = render_via_protocol(protocol, logical_messages);
        }
        Some((checkpoint, active_prompt_cache))
    }
}

impl CommittedAutoExpansion {
    /// Publish the prompt and its cache identity as one synchronous operation.
    /// The only constructor first commits every planned LCM node, so callers
    /// cannot expose reconstructed messages while their summaries stay eligible.
    fn publish(
        self,
        logical_messages: &mut Vec<Value>,
        rendered_messages: &mut Vec<Value>,
        counters: &RuntimeCounters,
        session_key: &str,
        prompt_fingerprint: crate::agent::prompt_fingerprint::PromptFingerprint,
    ) {
        *logical_messages = self.0.logical_messages;
        *rendered_messages = self.0.rendered_messages;
        let _transition = counters.lock_prompt_cache_transition();
        counters
            .prompt_fingerprints
            .lock()
            .insert(session_key.to_string(), prompt_fingerprint);
        counters
            .prompt_cache_watermark
            .lock()
            .insert(session_key.to_string(), logical_messages.len());
    }
}

/// Resolve and lock the per-session engine before mutating either eligibility
/// or the turn prompt. Once the awaited lock is acquired, the batch commit and
/// returned publication capability are produced in the same poll; cancellation
/// before that point therefore leaves both engine and prompt state untouched.
async fn commit_staged_auto_expansion(
    staged: AppliedAutoExpansion,
    lcm_engines: &Arc<Mutex<HashMap<String, Arc<tokio::sync::Mutex<LcmEngine>>>>>,
    session_id: &str,
) -> Option<CommittedAutoExpansion> {
    let engine = { lcm_engines.lock().await.get(session_id).cloned() }?;
    let mut engine = engine.lock().await;
    engine
        .commit_auto_expansions(&staged.node_ids)
        .then_some(CommittedAutoExpansion(staged))
}

/// Materialize a planned candidate without touching LCM eligibility state.
/// Exact reuse requires one matching summary and complete, ordered durable IDs;
/// every uncertain shape preserves the compacted prompt and appends the current
/// bounded flattened representation instead.
fn materialize_auto_expansion(
    messages: &[Value],
    candidate: &AutoExpansionCandidate,
    checkpoint: Option<&ExpansionCheckpoint>,
) -> AutoExpansionMaterialization {
    let exact_span = checkpoint.and_then(|checkpoint| {
        if !checkpoint.lease_confirmed || checkpoint.summary_node_id != candidate.node_id {
            return None;
        }
        let replaced_source_ids = checkpoint
            .replaced_span
            .iter()
            .map(|message| {
                message
                    .get("_db_id")
                    .and_then(Value::as_u64)
                    .and_then(|id| usize::try_from(id).ok())
            })
            .collect::<Option<Vec<_>>>()?;
        (replaced_source_ids == candidate.source_ids).then_some(&checkpoint.replaced_span)
    });
    if let Some(replaced_span) = exact_span {
        let mut positions = messages.iter().enumerate().filter_map(|(index, message)| {
            (message == &candidate.summary_message).then_some(index)
        });
        if let Some(summary_index) = positions.next() {
            if positions.next().is_none() {
                let mut exact =
                    Vec::with_capacity(messages.len().saturating_sub(1) + replaced_span.len());
                exact.extend_from_slice(&messages[..summary_index]);
                exact.extend(replaced_span.iter().cloned());
                exact.extend_from_slice(&messages[summary_index + 1..]);
                return AutoExpansionMaterialization {
                    messages: exact,
                    kind: AutoExpansionMaterializationKind::ExactCheckpoint,
                };
            }
        }
    }

    let mut flattened = messages.to_vec();
    flattened.push(candidate.flattened_fallback.clone());
    AutoExpansionMaterialization {
        messages: flattened,
        kind: AutoExpansionMaterializationKind::FlattenedFallback,
    }
}

/// Build and protocol-render the complete candidate prompt before publishing
/// any mutation to the turn. Returning `None` leaves the compacted logical
/// prompt untouched, which prevents an oversized expansion from triggering the
/// ordinary emergency-trim/session-rotation path.
fn apply_auto_expansion_candidates(
    protocol: &dyn ConversationProtocol,
    messages: &[Value],
    candidates: &[AutoExpansionCandidate],
    checkpoint: Option<&ExpansionCheckpoint>,
    tool_def_tokens: usize,
    prompt_token_limit: usize,
) -> Option<AppliedAutoExpansion> {
    if candidates.is_empty() {
        return None;
    }
    let mut logical_messages = messages.to_vec();
    let mut exact_count = 0;
    for candidate in candidates {
        let materialized = materialize_auto_expansion(&logical_messages, candidate, checkpoint);
        if materialized.kind == AutoExpansionMaterializationKind::ExactCheckpoint {
            exact_count += 1;
        }
        logical_messages = materialized.messages;
    }
    let rendered_messages = render_via_protocol(protocol, &logical_messages);
    let estimated_tokens =
        TokenBudget::estimate_tokens(&rendered_messages).saturating_add(tool_def_tokens);
    if estimated_tokens > prompt_token_limit {
        return None;
    }
    let retained_plan = if candidates.len() == 1 && exact_count == 1 {
        checkpoint.cloned().map(|checkpoint| {
            let fallback_prefix =
                materialize_auto_expansion(messages, &candidates[0], None).messages;
            let fallback_rendered = render_via_protocol(protocol, &fallback_prefix);
            let fallback_tokens =
                TokenBudget::estimate_tokens(&fallback_rendered).saturating_add(tool_def_tokens);
            RetainedExpansionPlan {
                checkpoint,
                compacted_prefix: messages.to_vec(),
                fallback_prefix: (fallback_tokens <= prompt_token_limit).then_some(fallback_prefix),
                exact_prefix_len: logical_messages.len(),
            }
        })
    } else {
        None
    };
    Some(AppliedAutoExpansion {
        logical_messages,
        rendered_messages,
        node_ids: candidates
            .iter()
            .map(|candidate| candidate.node_id)
            .collect(),
        exact_count,
        estimated_added_tokens: candidates
            .iter()
            .map(|candidate| candidate.estimated_tokens)
            .fold(0usize, usize::saturating_add),
        retained_plan,
    })
}

struct SoftCompactionRequest {
    core: Arc<SwappableCore>,
    session_id: String,
    compaction: CompactionHandle,
    turn_cancellation: Option<tokio_util::sync::CancellationToken>,
    prompt_prefix: Vec<Value>,
}

impl SoftCompactionRequest {
    fn take(ctx: &mut TurnContext) -> Option<Self> {
        if !std::mem::take(&mut ctx.soft_compaction_requested) || ctx.is_cancelled() {
            return None;
        }
        let prompt_prefix = ctx
            .messages
            .iter()
            .take_while(|message| {
                matches!(
                    message.get("role").and_then(Value::as_str),
                    Some("system" | "developer")
                )
            })
            .cloned()
            .collect();
        Some(Self {
            core: ctx.core.clone(),
            session_id: ctx.session_id.clone(),
            compaction: ctx.compaction.clone(),
            turn_cancellation: ctx.cancellation_token.clone(),
            prompt_prefix,
        })
    }

    fn is_cancelled(&self) -> bool {
        self.turn_cancellation
            .as_ref()
            .is_some_and(tokio_util::sync::CancellationToken::is_cancelled)
    }
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
    pub(crate) async fn persist_pending_protocol_messages(&mut self) {
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
            return;
        }

        // One checked transaction — replay must observe the whole protocol
        // group (carrier + receipts) or none of it. A per-message loop that
        // breaks mid-group silently truncates protocol history (see main
        // a05fc81).
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
                // Leave the messages untagged (no `_db_id`) so the next
                // persist retries the whole group; do not advance new_start.
                return;
            }
        };
        for (index, row_id) in pending_indices.into_iter().zip(row_ids) {
            self.messages[index]["_db_id"] = json!(row_id);
        }
        self.new_start = self.messages.len();
    }
}

/// Signal generation cancellation from either owner without dropping the
/// execution future. Once publication is claimed, execution ignores the signal
/// and this waiter continues through durable publication and pending handoff.
async fn await_compaction_with_cancellation(
    execution: impl std::future::Future<Output = Option<PendingCompaction>>,
    cancellation: tokio_util::sync::CancellationToken,
    job_cancellation: tokio_util::sync::CancellationToken,
    turn_cancellation: Option<tokio_util::sync::CancellationToken>,
) -> Option<PendingCompaction> {
    let turn_cancelled = async move {
        match turn_cancellation {
            Some(token) => token.cancelled().await,
            None => std::future::pending().await,
        }
    };
    tokio::pin!(execution);
    tokio::select! {
        biased;
        () = job_cancellation.cancelled() => {
            cancellation.cancel();
            execution.await
        }
        () = turn_cancelled => {
            cancellation.cancel();
            execution.await
        }
        result = &mut execution => result,
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
// Lease exhaustion is a paired tool-result receipt. The universal no-progress
// stop converges repeated blocked calls without mutating the frozen tool schema
// that participates in the server-side prompt prefix.
pub(crate) const NO_PROGRESS_HARD_STOP: u32 = 4;
pub(crate) const MAX_LEASE_RENEWAL_REJECTIONS: u32 = 2;

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
    /// `lease_size * (1 + max_renewals)`; after exhaustion, paired rejection
    /// receipts preserve provider protocol and the no-progress breaker bounds
    /// a model that ignores them without changing the prompt prefix.
    pub(crate) lease: Lease,
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
    use crate::providers::base::{FinishReason, LLMProvider, LLMResponse};
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
                finish_reason: FinishReason::Stop,
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
        validation::ValidationOutcome::Error(validation::ValidationError::HallucinatedToolCall)
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
    pub(crate) async fn compaction_handle_for_session(&self, session_id: &str) -> CompactionHandle {
        let mut handles = self.compaction_handles.lock().await;
        handles
            .entry(session_id.to_string())
            .or_insert_with(|| CompactionHandle::for_session(session_id))
            .clone()
    }

    pub(crate) async fn remove_compaction_handle_if_owned(
        &self,
        session_id: &str,
        candidate: &CompactionHandle,
    ) {
        let mut handles = self.compaction_handles.lock().await;
        if handles
            .get(session_id)
            .is_some_and(|current| current.same_owner(candidate))
        {
            handles.remove(session_id);
        }
    }

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
        let soft_compaction = SoftCompactionRequest::take(&mut ctx);
        let response = self.finalize_response(ctx).await;
        if let Some(request) = soft_compaction {
            self.spawn_requested_soft_compaction(request).await;
        }
        response
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
                        ctx.flow.consecutive_no_progress_rounds =
                            ctx.flow.consecutive_no_progress_rounds.saturating_add(1);
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

        if let Some((checkpoint, active_prompt_cache)) =
            ctx.higgs_session_route.take_retained_checkpoint()
        {
            active_prompt_cache.restore(&ctx.counters, &ctx.session_key);
            ctx.counters.discard_expansion_checkpoint(
                &ctx.session_key,
                checkpoint.old_higgs_session_id,
                checkpoint.summary_node_id,
            );
            ctx.retained_route_cleanup.disarm();
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
        // Compute tool activity on the FULL array before the frozen-prefix
        // split — `recent_assistant_used_tools` inside `inject_format_anchor`
        // would otherwise only see the tail (post-watermark messages) and miss
        // recent assistant tool calls trapped in the cached prefix.
        let tools_active = anti_drift::recent_assistant_used_tools(&ctx.messages);
        prefix_guard::with_frozen_prefix(&mut ctx.messages, frozen_prefix, |m| {
            ctx.core
                .retention
                .apply_shaping(m, iteration, run_anti_drift, tools_active);
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

        // Select and filter tool definitions for this turn. Tool-lease
        // enforcement happens at execution time: rejections remain paired
        // protocol messages and never change the schema Higgs caches.
        let (mut tool_defs, saved_tool_defs, mut tool_presentation_mode) =
            self.select_tool_definitions(ctx);
        // Reuse only a catalog previously installed from a final provider
        // array. First-use candidates are not installed until router policy
        // below has had its chance to restore native definitions.
        if let Some(frozen) = ctx
            .counters
            .frozen_tool_definitions(&ctx.session_key, tool_presentation_mode)
        {
            tool_defs = frozen;
        }
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

        // Account for compaction pressure; soft work is only requested here.
        self.manage_compaction(ctx, tool_def_tokens).await;
        if ctx.is_cancelled() {
            return StepResult::Done(IterationOutcome::Finished(String::new()));
        }

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
        // single-model setup), the preflight is a pure passthrough, so gating
        // the whole block makes trio-off absent from the hot path.
        let trio_active =
            ctx.core.mode().is_local() && ctx.core.tool_delegation_config.strict_no_tools_main();
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
                    if tool_defs.is_empty() && !saved_tool_defs.is_empty() {
                        debug!("router_preflight=Passthrough — restoring tool_defs for main model fallback");
                        tool_defs = saved_tool_defs;
                        tool_presentation_mode = ToolPresentationMode::Native;
                    }
                }
            }
        }

        tool_defs = self.freeze_final_tool_catalog(ctx, tool_presentation_mode, tool_defs);

        ctx.advertised_tool_names = Some(advertised_tool_names(&tool_defs));

        // Adaptive max_tokens: size the response budget to the task.
        let effective_max_tokens = self.compute_adaptive_max_tokens(ctx);
        self.apply_planned_auto_expansion(ctx, &tool_defs, effective_max_tokens)
            .await;

        StepResult::Next(IterationPhase::Calling {
            tool_defs,
            max_tokens: effective_max_tokens,
        })
    }

    /// Select and filter tool definitions for this turn.
    ///
    /// Returns `(active_defs, saved_defs, mode)` where `saved_defs` preserves
    /// the native post-policy state for router passthrough fallback.
    fn select_tool_definitions(
        &self,
        ctx: &mut TurnContext,
    ) -> (Vec<Value>, Vec<Value>, ToolPresentationMode) {
        // One protocol for local and cloud: hot tools have native schemas and
        // the proxy exposes the long tail. Bonsai otherwise mixes the proxy
        // envelope with native calls (for example exec(args={command: ...})).
        // The larger schema prefix is stable and retained by local backends;
        // paying it once is cheaper than repeated malformed generations.
        let mut tool_defs = ctx.tools.get_core_plus_proxy_definitions();
        let mut mode = ToolPresentationMode::Native;
        // Tool-averse models (no tool-calling training, e.g. VibeThinker):
        // the native `tools` parameter confuses or errors their chat
        // templates, and nothing else would teach them the textual syntax the
        // response parser expects. Move the tool catalog into the system
        // prompt as a textual-protocol lesson and send no `tools` at all.
        if ctx.protocol.is_textual_replay() && !ctx.core.model_capabilities.tool_calling {
            mode = ToolPresentationMode::Textual;
            let already_taught = ctx
                .messages
                .first()
                .and_then(|m| m["content"].as_str())
                .is_some_and(|s| s.contains(crate::agent::protocol::TEXTUAL_TOOLS_MARKER));
            if !already_taught && !tool_defs.is_empty() {
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
        let saved_tool_defs = ctx
            .counters
            .frozen_tool_definitions(&ctx.session_key, ToolPresentationMode::Native)
            .unwrap_or_else(|| tool_defs.clone());
        if mode == ToolPresentationMode::Native
            && ctx.core.mode().is_local()
            && ctx.core.tool_delegation_config.strict_no_tools_main()
        {
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
                mode = ToolPresentationMode::Trio;
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

        (tool_defs, saved_tool_defs, mode)
    }

    /// Freeze the exact array that will be passed to the provider. The catalog
    /// lock is released before a mode-change rotation, then reacquired only to
    /// install the new final array; no reset runs while holding it.
    fn freeze_final_tool_catalog(
        &self,
        ctx: &mut TurnContext,
        mode: ToolPresentationMode,
        candidate: Vec<Value>,
    ) -> Vec<Value> {
        if let Some(frozen) = ctx.counters.frozen_tool_definitions(&ctx.session_key, mode) {
            return frozen;
        }

        if ctx
            .counters
            .tool_presentation_mode_changed(&ctx.session_key, mode)
        {
            let previous_mode = ctx.counters.tool_presentation_mode(&ctx.session_key);
            let rotated =
                invalidate_prompt_cache_for_rewrite(ctx, CacheResetReason::ToolBlockChange);
            warn!(
                session = %ctx.session_key,
                ?previous_mode,
                ?mode,
                rotated,
                "tool_block_changed — rotating before installing final provider catalog"
            );
        }
        ctx.counters
            .install_tool_catalog(&ctx.session_key, mode, candidate.clone());
        candidate
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
        let higgs_capable =
            ctx.core.mode().is_local() && ctx.core.provider.supports_higgs_session_cache();
        let rotated = if higgs_capable {
            let frozen_tool_hash = ctx
                .counters
                .prompt_tool_hashes
                .lock()
                .get(&ctx.session_key)
                .copied()
                .unwrap_or(0);
            let checkpoint_context = ctx.counters.expansion_checkpoint_context(
                &ctx.session_key,
                &ctx.core.model,
                frozen_tool_hash,
                RuntimeCounters::now_epoch_ms().saturating_add(300_000),
            );
            let retirement = checkpoint_context
                .and_then(|context| pending.expansion_retirement(context))
                .unwrap_or(SessionRetirement::Drop);
            ctx.counters
                .retire_higgs_session(&ctx.session_key, retirement);
            ctx.counters
                .note_cache_reset(&ctx.session_key, CacheResetReason::LcmCheckpoint.as_wire());
            send_cache_reset_marker(&ctx.text_delta_tx, CacheResetReason::LcmCheckpoint);
            true
        } else {
            invalidate_prompt_cache_for_rewrite(ctx, CacheResetReason::LcmCheckpoint)
        };
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

    /// Start requested soft work only after the complete foreground loop.
    async fn spawn_requested_soft_compaction(&self, request: SoftCompactionRequest) {
        if request.is_cancelled() {
            return;
        }
        let Some(admission) = request.compaction.admit().await else {
            return;
        };
        if request.compaction.has_pending().await || request.compaction.has_job().await {
            return;
        }
        let max_messages =
            crate::agent::agent_core::history_limit_lcm(request.core.token_budget.max_context());
        let history = request
            .core
            .sessions
            .get_history(
                &request.session_id,
                max_messages,
                request.core.max_history_turns,
            )
            .await;
        if request.is_cancelled() {
            return;
        }
        let Some(lcm_engine) = self
            .lcm_engines
            .lock()
            .await
            .get(&request.session_id)
            .cloned()
        else {
            return;
        };
        {
            let mut engine = lcm_engine.lock().await;
            for message in &history {
                engine.ingest(message.clone());
            }
        }
        if request.is_cancelled() {
            return;
        }
        let session_turn = request
            .core
            .sessions
            .get_session(&request.session_id)
            .await
            .map_or(0, |session| session.message_count as u64);
        if request.is_cancelled() {
            return;
        }

        let core = request.core;
        let session_id = request.session_id;
        let mut messages = request.prompt_prefix;
        messages.extend(history);
        let turn_cancellation = request.turn_cancellation;
        let _ = request
            .compaction
            .try_start_admitted(&admission, move |job_cancellation, publication| {
                let cancellation = tokio_util::sync::CancellationToken::new();
                let execution = execute_lcm_compaction(
                    core,
                    session_id,
                    lcm_engine,
                    messages,
                    session_turn,
                    CompactionFailureMode::PreserveContext,
                    cancellation.clone(),
                    publication,
                );
                await_compaction_with_cancellation(
                    execution,
                    cancellation,
                    job_cancellation,
                    turn_cancellation,
                )
            })
            .await;
    }

    /// Apply hard pressure now or record soft pressure for post-loop execution.
    async fn manage_compaction(&self, ctx: &mut TurnContext, tool_def_tokens: usize) {
        let Some(admission) = ctx.compaction.admit().await else {
            return;
        };
        ctx.compaction.cancel_and_reap().await;
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
                engines
                    .entry(ctx.session_id.clone())
                    .or_insert_with(|| {
                        let config = LcmConfig::from(&self.lcm_config);
                        Arc::new(tokio::sync::Mutex::new(LcmEngine::new(config)))
                    })
                    .clone()
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

            let mut raw_hard = conversation_token_count(&ctx.messages) > hard_limit;
            if raw_hard {
                self.install_pending_compaction(ctx, true).await;
                raw_hard = conversation_token_count(&ctx.messages) > hard_limit;
                action = {
                    let engine = lcm_engine.lock().await;
                    engine.check_thresholds_with_available(available)
                };
            }

            let has_pending = ctx.compaction.has_pending().await;
            let must_block = raw_hard || action == CompactionAction::Blocking;
            if must_block {
                ctx.soft_compaction_requested = false;
            }

            let blocking_started = if must_block {
                tracing::info!(
                    compaction_type = "lcm_blocking",
                    msg_count = ctx.messages.len(),
                    conv_tokens,
                    available,
                    hard_limit,
                    soft_limit,
                    "lcm_compaction_triggered"
                );
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
                let compaction_engine = lcm_engine.clone();
                let turn_cancellation = ctx.cancellation_token.clone();
                let started = ctx
                    .compaction
                    .try_start_admitted(&admission, move |job_cancellation, publication| {
                        let cancellation = tokio_util::sync::CancellationToken::new();
                        let execution = execute_lcm_compaction(
                            core,
                            session_id,
                            compaction_engine,
                            messages,
                            session_turn,
                            CompactionFailureMode::Deterministic,
                            cancellation.clone(),
                            publication,
                        );
                        await_compaction_with_cancellation(
                            execution,
                            cancellation,
                            job_cancellation,
                            turn_cancellation,
                        )
                    })
                    .await;
                Some(started)
            } else if action == CompactionAction::Async && !has_pending {
                tracing::info!(
                    compaction_type = "lcm_async",
                    msg_count = ctx.messages.len(),
                    conv_tokens,
                    available,
                    hard_limit,
                    soft_limit,
                    "lcm_compaction_triggered"
                );
                ctx.soft_compaction_requested = true;
                None
            } else if has_pending {
                debug!("LCM compaction deferred: checkpoint is waiting for a safe install");
                None
            } else {
                None
            };

            // A blocking caller waits for completion, but never owns the job
            // future. Dropping the caller therefore cannot cancel publication
            // or the pending-slot handoff. Release admission first so clear can
            // close the session and cancel generation while this waiter sleeps.
            drop(admission);
            if let Some(started) = blocking_started {
                if started {
                    ctx.compaction.wait_for_completion().await;
                }
                self.install_pending_compaction(ctx, true).await;
                // Clear the compaction indicator. The next event (prefill,
                // cache reset, etc.) replaces the Activity row, but in case
                // the turn finishes here we don't want a stuck "compacting".
                send_compaction_marker(
                    &ctx.text_delta_tx,
                    crate::turn_stream::CompactionStatus::Finished,
                );
            }

            if ctx.is_cancelled() {
                return;
            }
        }
    }

    /// Select and materialize expansion candidates for one provider attempt.
    /// The LCM engine remains unchanged until `step_call_llm` commits the node
    /// IDs after a successful response.
    async fn apply_planned_auto_expansion(
        &self,
        ctx: &mut TurnContext,
        tool_defs: &[Value],
        max_tokens: u32,
    ) {
        if !ctx.higgs_session_route.permits_auto_expansion() {
            return;
        }
        ctx.staged_auto_expansion = None;
        let Some(lcm_engine) = self.lcm_engines.lock().await.get(&ctx.session_id).cloned() else {
            return;
        };
        let current_turn = ctx
            .core
            .sessions
            .get_session(&ctx.session_id)
            .await
            .map_or(0, |session| session.message_count as u64);
        let tool_def_tokens = TokenBudget::estimate_tool_def_tokens(tool_defs);
        let wire_tokens = TokenBudget::estimate_tokens(&ctx.rendered_messages);
        let expand_t0 = std::time::Instant::now();
        let (candidates, prompt_token_limit, summary_count) = {
            let mut engine = lcm_engine.lock().await;
            if engine.dag().is_empty() {
                return;
            }
            engine.set_current_turn(current_turn);
            let available = ctx.core.token_budget.available_budget(tool_def_tokens);
            let hard_message_limit = (available as f64 * engine.tau_hard()) as usize;
            let provider_prompt_limit = ctx
                .core
                .token_budget
                .max_context()
                .saturating_sub(max_tokens as usize);
            (
                engine.plan_auto_expansion(&ctx.core.token_budget, tool_def_tokens, wire_tokens),
                hard_message_limit
                    .saturating_add(tool_def_tokens)
                    .min(provider_prompt_limit),
                engine.dag().len(),
            )
        };
        tracing::info!(
            target: "turn_timing",
            auto_expand_ms = expand_t0.elapsed().as_millis() as u64,
            summaries = summary_count,
            "auto_expand_timing"
        );
        if candidates.is_empty() {
            return;
        }

        let frozen_tool_hash = crate::agent::prompt_fingerprint::hash_tools(tool_defs);
        let checkpoint = ctx.core.provider.supports_higgs_session_cache().then(|| {
            ctx.counters.confirmed_expansion_checkpoint(
                &ctx.session_key,
                &ctx.core.model,
                frozen_tool_hash,
                RuntimeCounters::now_epoch_ms(),
            )
        });
        let checkpoint = checkpoint.flatten();
        let Some(applied) = apply_auto_expansion_candidates(
            &*ctx.protocol,
            &ctx.messages,
            &candidates,
            checkpoint.as_ref(),
            tool_def_tokens,
            prompt_token_limit,
        ) else {
            debug!(
                session = %ctx.session_key,
                candidates = candidates.len(),
                prompt_token_limit,
                "LCM auto-expand: reconstructed prompt is oversized; preserving summary"
            );
            return;
        };

        debug!(
            session = %ctx.session_key,
            count = applied.node_ids.len(),
            exact_count = applied.exact_count,
            estimated_added_tokens = applied.estimated_added_tokens,
            "LCM auto-expand: planned expansion for provider response"
        );
        ctx.staged_auto_expansion = Some(applied);
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

    async fn persist_model_failure(ctx: &TurnContext, call_id: &str, detail: &str) {
        if let Err(record_error) = ctx
            .core
            .sessions
            .record_model_failure(
                &ctx.session_id,
                &ctx.request_id,
                ctx.turn_count,
                call_id,
                detail,
            )
            .await
        {
            error!(
                session = %ctx.session_key,
                call_id,
                error = %record_error,
                "model_failure_replay_persist_failed"
            );
        }
    }

    /// Handle an LLM provider error: retry once if retryable, otherwise return error.
    async fn handle_llm_error(
        e: anyhow::Error,
        ctx: &mut TurnContext,
        counters: &RuntimeCounters,
        label: &str,
        retry_policy: ProviderRequestRetryPolicy,
    ) -> StepResult {
        if retry_policy.allows_retry()
            && !ctx.flow.retries.api_retried
            && is_retryable_provider_error(&e)
        {
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

    fn discard_selected_expansion_checkpoint(
        ctx: &mut TurnContext,
        checkpoint: &ExpansionCheckpoint,
        failure: RetainedExpansionFailure,
    ) {
        let used_flattened = matches!(failure, RetainedExpansionFailure::Unavailable)
            && ctx
                .staged_auto_expansion
                .as_mut()
                .is_some_and(|staged| staged.use_flattened_fallback(&*ctx.protocol));
        if !used_flattened {
            ctx.staged_auto_expansion = None;
        }
        ctx.higgs_session_route = HiggsSessionRoute::ActiveCompacted {
            route: if used_flattened {
                ActiveCompactedRoute::Fallback
            } else {
                ActiveCompactedRoute::OverflowSummary
            },
        };
        ctx.counters.discard_expansion_checkpoint(
            &ctx.session_key,
            checkpoint.old_higgs_session_id,
            checkpoint.summary_node_id,
        );
    }

    fn fallback_retained_expansion_route(ctx: &mut TurnContext, failure: RetainedExpansionFailure) {
        let checkpoint = ctx.higgs_session_route.fallback_from_retained(
            failure,
            &mut ctx.messages,
            &mut ctx.rendered_messages,
            &*ctx.protocol,
        );
        if let Some((checkpoint, active_prompt_cache)) = checkpoint {
            match failure {
                RetainedExpansionFailure::Unavailable => {
                    let used_flattened = ctx
                        .staged_auto_expansion
                        .as_mut()
                        .is_some_and(|staged| staged.use_flattened_fallback(&*ctx.protocol));
                    if !used_flattened {
                        ctx.staged_auto_expansion = None;
                    }
                }
                RetainedExpansionFailure::ContextOverflow => {
                    ctx.staged_auto_expansion = None;
                }
            }
            active_prompt_cache.restore(&ctx.counters, &ctx.session_key);
            ctx.counters.discard_expansion_checkpoint(
                &ctx.session_key,
                checkpoint.old_higgs_session_id,
                checkpoint.summary_node_id,
            );
            ctx.retained_route_cleanup.disarm();
        }
    }

    fn handle_retained_route_error(ctx: &mut TurnContext, error: &anyhow::Error) -> bool {
        if ctx.higgs_session_route.retained_checkpoint().is_none() {
            return false;
        }
        let Some(failure) = classify_retained_session_error(error).map(|kind| match kind {
            RetainedSessionErrorKind::Unavailable => RetainedExpansionFailure::Unavailable,
            RetainedSessionErrorKind::ContextOverflow => RetainedExpansionFailure::ContextOverflow,
        }) else {
            return false;
        };
        warn!(
            session = %ctx.session_key,
            error = %error,
            "retained_expansion_foreground_failed_falling_back"
        );
        Self::fallback_retained_expansion_route(ctx, failure);
        true
    }

    /// Validate the retained server state without generating tokens. This uses
    /// the same provider object and request construction as foreground calls;
    /// only the retained route control and zero completion budget differ.
    async fn prepare_retained_expansion_route(
        &self,
        ctx: &mut TurnContext,
        tool_defs_opt: Option<&[Value]>,
        max_tokens: u32,
    ) {
        if !ctx.core.provider.supports_higgs_session_cache()
            || !matches!(
                ctx.higgs_session_route,
                HiggsSessionRoute::ActiveCompacted { .. }
            )
        {
            return;
        }
        let Some(plan) = ctx
            .staged_auto_expansion
            .as_ref()
            .and_then(AppliedAutoExpansion::retained_plan)
        else {
            return;
        };
        let max_prompt_tokens = ctx
            .core
            .token_budget
            .max_context()
            .saturating_sub(max_tokens as usize)
            .min(u32::MAX as usize) as u32;
        let frozen_tool_hash =
            crate::agent::prompt_fingerprint::hash_tools(tool_defs_opt.unwrap_or(&[]));
        let Some(reservation) = ctx.counters.reserve_retained_expansion_request(
            &ctx.session_key,
            &plan.checkpoint,
            &ctx.core.model,
            frozen_tool_hash,
            max_prompt_tokens,
            RuntimeCounters::now_epoch_ms(),
        ) else {
            Self::discard_selected_expansion_checkpoint(
                ctx,
                &plan.checkpoint,
                RetainedExpansionFailure::Unavailable,
            );
            return;
        };
        let sent_drop_ids = reservation.drop_ids().to_vec();
        let mut messages = ctx
            .staged_auto_expansion
            .as_ref()
            .map(|staged| staged.rendered_messages.clone())
            .unwrap_or_else(|| render_via_protocol(&*ctx.protocol, &ctx.messages));
        attach_higgs_session_control(&mut messages, reservation.control());
        let recorded_request = RecordedProviderRequest {
            messages: messages.clone(),
            tools: tool_defs_opt.map(<[Value]>::to_vec),
            model: ctx.core.model.clone(),
            max_tokens: 0,
            temperature: ctx.core.temperature,
            thinking_budget: None,
            top_p: None,
            tool_choice: "auto".to_string(),
            streaming: false,
        };
        let call_id = match ctx
            .core
            .sessions
            .record_model_request(
                &ctx.session_id,
                &ctx.request_id,
                ctx.turn_count,
                ModelCallPurpose::RetainedExpansionPreflight,
                &recorded_request,
            )
            .await
        {
            Ok(call_id) => call_id,
            Err(error) => {
                warn!(
                    session = %ctx.session_key,
                    %error,
                    "retained_expansion_preflight_replay_persist_failed"
                );
                drop(reservation);
                Self::discard_selected_expansion_checkpoint(
                    ctx,
                    &plan.checkpoint,
                    RetainedExpansionFailure::Unavailable,
                );
                return;
            }
        };
        let result = ctx
            .core
            .provider
            .chat(
                &messages,
                tool_defs_opt,
                Some(&ctx.core.model),
                0,
                ctx.core.temperature,
                None,
                None,
            )
            .await;
        let failure = match result {
            Ok(response) => {
                if let Err(error) = ctx
                    .core
                    .sessions
                    .record_model_response(
                        &ctx.session_id,
                        &ctx.request_id,
                        ctx.turn_count,
                        &call_id,
                        &RecordedProviderResponse::from(&response),
                    )
                    .await
                {
                    Some(anyhow::anyhow!(
                        "retained preflight response replay persistence failed: {error}"
                    ))
                } else {
                    response.outcome().err().map(anyhow::Error::new)
                }
            }
            Err(error) => {
                Self::persist_model_failure(ctx, &call_id, &error.to_string()).await;
                Some(error)
            }
        };
        drop(reservation);
        if let Some(error) = failure {
            let failure = classify_retained_session_error(&error).map_or(
                RetainedExpansionFailure::Unavailable,
                |kind| match kind {
                    RetainedSessionErrorKind::Unavailable => RetainedExpansionFailure::Unavailable,
                    RetainedSessionErrorKind::ContextOverflow => {
                        RetainedExpansionFailure::ContextOverflow
                    }
                },
            );
            warn!(
                session = %ctx.session_key,
                error = %error,
                "retained_expansion_preflight_failed"
            );
            Self::discard_selected_expansion_checkpoint(ctx, &plan.checkpoint, failure);
            return;
        }
        if !sent_drop_ids.is_empty() {
            ctx.counters
                .clear_pending_higgs_session_drop_ids(&ctx.session_key, &sent_drop_ids);
        }
        // The retained request targets a different physical Higgs session.
        // Snapshot and clear the active route's cache identity together so the
        // first retained foreground call cannot be diagnosed against the
        // compacted session's fingerprint. Later retained tool iterations then
        // compare against the retained fingerprint installed by step_call_llm.
        let active_prompt_cache =
            PromptCacheSnapshot::capture_and_clear(&ctx.counters, &ctx.session_key);
        ctx.higgs_session_route = HiggsSessionRoute::retained_expansion(
            plan.checkpoint.clone(),
            plan.compacted_prefix,
            plan.fallback_prefix,
            plan.exact_prefix_len,
            active_prompt_cache.clone(),
        );
        ctx.retained_route_cleanup.arm(
            Arc::clone(&ctx.counters),
            ctx.session_key.clone(),
            plan.checkpoint,
            active_prompt_cache,
        );
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
        self.prepare_retained_expansion_route(ctx, tool_defs_opt, max_tokens)
            .await;
        // Expansion materialization is request-local until this call succeeds;
        // retries continue from the unchanged compacted logical conversation.
        let mut messages_for_llm = if let Some(staged) = &ctx.staged_auto_expansion {
            staged.rendered_messages.clone()
        } else if ctx.rendered_messages.is_empty() {
            // Fallback: render now if step_pre_call was bypassed (should not happen in practice).
            render_via_protocol(&*ctx.protocol, &ctx.messages)
        } else {
            ctx.rendered_messages.clone()
        };

        let mut pending_higgs_drop = Vec::new();
        let mut higgs_control = None;
        let mut higgs_request_reservation = None;
        if ctx.core.provider.supports_higgs_session_cache() {
            let frozen_tool_hash =
                crate::agent::prompt_fingerprint::hash_tools(tool_defs_opt.unwrap_or(&[]));
            let max_prompt_tokens = ctx
                .core
                .token_budget
                .max_context()
                .saturating_sub(max_tokens as usize)
                .min(u32::MAX as usize) as u32;
            let retained_checkpoint = ctx.higgs_session_route.retained_checkpoint().cloned();
            let reservation = retained_checkpoint.as_ref().and_then(|checkpoint| {
                counters.reserve_retained_expansion_request(
                    &ctx.session_key,
                    checkpoint,
                    &ctx.core.model,
                    frozen_tool_hash,
                    max_prompt_tokens,
                    RuntimeCounters::now_epoch_ms(),
                )
            });
            let reservation = if let Some(reservation) = reservation {
                reservation
            } else {
                if retained_checkpoint.is_some() {
                    Self::fallback_retained_expansion_route(
                        ctx,
                        RetainedExpansionFailure::Unavailable,
                    );
                    messages_for_llm = if let Some(staged) = &ctx.staged_auto_expansion {
                        staged.rendered_messages.clone()
                    } else {
                        ctx.rendered_messages.clone()
                    };
                }
                counters.reserve_higgs_session_request(
                    &ctx.session_key,
                    &ctx.session_id,
                    &ctx.core.model,
                    frozen_tool_hash,
                    max_prompt_tokens,
                    RuntimeCounters::now_epoch_ms(),
                )
            };
            pending_higgs_drop = reservation.drop_ids().to_vec();
            higgs_control = Some(reservation.control().clone());
            higgs_request_reservation = Some(reservation);
        }

        // Prefix-divergence diagnostic: a prompt that is not an append-only
        // extension of this session's previous call forces the server to
        // re-prefill everything past the divergence point (~60s for a 14k
        // local context). Make every such miss a one-line diagnosis.
        use crate::agent::prompt_fingerprint::{self, PromptDelta};
        let diag_t0 = std::time::Instant::now();
        let prompt_fp = prompt_fingerprint::fingerprint(&messages_for_llm);
        // One checked invariant covering the five sites that write the prompt
        // HEAD: prepare_context's continuity note and stable-prompt assignment,
        // agent_core::append_to_system_prompt, context.rs's developer-message
        // rewrite, and openai_compat's developer→system fold. Each was
        // previously held only by a doc comment. Runs on the RENDERED array
        // immediately before the call, so it sees what the server sees no
        // matter which layer did the writing.
        //
        // The sixth site, `context.rs::insert_tail_before_user`, inserts before
        // the LAST message rather than at the head; it is covered by the
        // append-only fingerprint comparison just below, which now spans turn
        // boundaries.
        prefix_guard::assert_stable_head(
            &ctx.session_key,
            &messages_for_llm,
            &counters.prompt_head_hashes,
        );
        let prompt_msg_count = messages_for_llm.len();
        let tool_def_tokens = TokenBudget::estimate_tool_def_tokens(tool_defs_opt.unwrap_or(&[]));
        let prompt_total_estimate =
            TokenBudget::estimate_tokens(&messages_for_llm).saturating_add(tool_def_tokens);
        ctx.flow.provider_prompt_estimate = Some(prompt_total_estimate);
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
                    counters.cache_diverged.fetch_add(1, Ordering::Relaxed);
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
                PromptDelta::First => {
                    // `First` means "no fingerprint to compare against". That
                    // is turn one — OR a deliberate mid-session clear, which
                    // costs exactly the same full re-prefill but used to be
                    // reported as a fresh start and therefore never appeared
                    // in the log at all. `take_cache_reset` tells the two
                    // apart so a sanctioned reset is priced, not hidden.
                    match counters.take_cache_reset(&ctx.session_key) {
                        Some(reason) => {
                            counters
                                .cache_sanctioned_resets
                                .fetch_add(1, Ordering::Relaxed);
                            tracing::warn!(
                                session = %ctx.session_key,
                                reason,
                                messages = prompt_msg_count,
                                prefill_estimate,
                                "prompt_cache_sanctioned_reset — prefix dropped on purpose; server re-prefills the whole context"
                            );
                        }
                        None => debug!(
                            session = %ctx.session_key,
                            messages = prompt_msg_count,
                            prefill_estimate,
                            "prompt_cache_cold_start"
                        ),
                    }
                    ControlMarker::CacheStatus(CacheStatus::First {
                        messages: prompt_msg_count,
                    })
                    .encode()
                }
            };
            if let Some(ref delta_tx) = ctx.text_delta_tx {
                let _ = delta_tx.send(cache_marker);
                if prefill_estimate > 0 {
                    let _ = delta_tx
                        .send(ControlMarker::PrefillEstimate(prefill_estimate as u64).encode());
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
            let _transition = counters.lock_prompt_cache_transition();
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
        tracing::info!(
            target: "turn_timing",
            prefix_diag_ms = diag_t0.elapsed().as_millis() as u64,
            msg_count = prompt_msg_count,
            "prefix_diag_timing"
        );

        let retry_policy = if higgs_control
            .as_ref()
            .is_some_and(|control| control.session_lease.is_some())
        {
            ProviderRequestRetryPolicy::OneShotLease
        } else {
            ProviderRequestRetryPolicy::Standard
        };

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

        if let Some(control) = &higgs_control {
            attach_higgs_session_control(&mut messages_for_llm, control);
        }

        // The provider boundary is the exact model-visible contract: protocol
        // rendering, tool presentation, retained-session controls, and sampling
        // settings are final here. Make it durable before the call so neither a
        // workspace change nor a crash can force replay to regenerate bytes.
        let recorded_request = RecordedProviderRequest {
            messages: messages_for_llm.clone(),
            tools: tool_defs_opt.map(<[Value]>::to_vec),
            model: ctx.core.model.clone(),
            max_tokens,
            temperature: ctx.core.temperature,
            thinking_budget,
            top_p: None,
            tool_choice: "auto".to_string(),
            streaming: ctx.text_delta_tx.is_some(),
        };
        let model_call_id = match ctx
            .core
            .sessions
            .record_model_request(
                &ctx.session_id,
                &ctx.request_id,
                ctx.turn_count,
                ModelCallPurpose::Main,
                &recorded_request,
            )
            .await
        {
            Ok(call_id) => call_id,
            Err(record_error) => {
                counters.mark_inference_finished();
                error!(
                    session = %ctx.session_key,
                    error = %record_error,
                    "model_request_replay_persist_failed"
                );
                return StepResult::Done(IterationOutcome::Error(
                    "I could not durably record the model request, so I stopped before sending it."
                        .to_string(),
                ));
            }
        };

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
                        Self::persist_model_failure(ctx, &model_call_id, &e.to_string()).await;
                        if Self::handle_retained_route_error(ctx, &e) {
                            counters.mark_inference_finished();
                            return StepResult::Done(IterationOutcome::Continue);
                        }
                        return Self::handle_llm_error(
                            e,
                            ctx,
                            counters,
                            "llm_stream_call",
                            retry_policy,
                        )
                        .await;
                    }
                    Err(_) => {
                        counters.mark_inference_finished();
                        let detail = local_no_stream_headers_error(timeout);
                        Self::persist_model_failure(ctx, &model_call_id, &detail).await;
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
                        Self::persist_model_failure(ctx, &model_call_id, &e.to_string()).await;
                        if Self::handle_retained_route_error(ctx, &e) {
                            counters.mark_inference_finished();
                            return StepResult::Done(IterationOutcome::Continue);
                        }
                        return Self::handle_llm_error(
                            e,
                            ctx,
                            counters,
                            "llm_stream_call",
                            retry_policy,
                        )
                        .await;
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
                        // The None branch above sleeps forever, so this arm is only
                        // reachable when Some; a zero timeout is a safe fallback.
                        let timeout = match no_progress_timeout {
                            Some(t) => t,
                            None => std::time::Duration::ZERO,
                        };
                        let detail = local_no_stream_progress_error(timeout);
                        Self::persist_model_failure(ctx, &model_call_id, &detail).await;
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
                        Self::persist_model_failure(
                            ctx,
                            &model_call_id,
                            "stream cancelled before terminal response",
                        )
                        .await;
                        emit_stream_abort_metrics(
                            ctx,
                            "The stream was cancelled before the backend returned a final response.",
                        );
                        return StepResult::Done(IterationOutcome::Finished(String::new()));
                    }
                    error!("LLM stream ended without Done");
                    Self::persist_model_failure(
                        ctx,
                        &model_call_id,
                        "stream ended without terminal response",
                    )
                    .await;
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
                    Self::persist_model_failure(ctx, &model_call_id, &e.to_string()).await;
                    if Self::handle_retained_route_error(ctx, &e) {
                        counters.mark_inference_finished();
                        return StepResult::Done(IterationOutcome::Continue);
                    }
                    return Self::handle_llm_error(e, ctx, counters, "llm_call", retry_policy)
                        .await;
                }
            }
        };

        if let Err(record_error) = ctx
            .core
            .sessions
            .record_model_response(
                &ctx.session_id,
                &ctx.request_id,
                ctx.turn_count,
                &model_call_id,
                &RecordedProviderResponse::from(&response),
            )
            .await
        {
            counters.mark_inference_finished();
            error!(
                session = %ctx.session_key,
                call_id = %model_call_id,
                error = %record_error,
                "model_response_replay_persist_failed"
            );
            return StepResult::Done(IterationOutcome::Error(
                "I received a model response but could not record it durably, so I stopped before acting on it."
                    .to_string(),
            ));
        }

        if !pending_higgs_drop.is_empty() {
            counters.clear_pending_higgs_session_drop_ids(&ctx.session_key, &pending_higgs_drop);
        }

        // Inference complete — allow watchdog health checks again.
        counters.mark_inference_finished();

        if let Err(error) = response.outcome().map_err(anyhow::Error::new) {
            if Self::handle_retained_route_error(ctx, &error) {
                return StepResult::Done(IterationOutcome::Continue);
            }
        }

        // A lease is a one-shot operation. Resolve the response that actually
        // carried it before any forced-tool recovery can issue a second call.
        if let Some(reservation) = higgs_request_reservation.as_mut() {
            reservation.resolve_lease(response.usage.get("higgs_session_lease_active").copied());
        }
        let mut recovery_messages = messages_for_llm.clone();
        strip_higgs_session_lease_control(&mut recovery_messages);

        // Tier-2 forced-tool recovery: if a local model botched a tool call
        // (intent prose / hallucinated syntax / empty block) instead of emitting
        // one, re-issue once with tool_choice=required so the Higgs backend
        // grammar-constrains a valid call — replacing the old hint-and-loop.
        let recovery = self
            .maybe_recover_botched_tool_call(
                ctx,
                response,
                &recovery_messages,
                tool_defs_opt,
                max_tokens,
            )
            .await;
        let response = match recovery {
            ForcedToolRecoveryOutcome::Response(response) => response,
            ForcedToolRecoveryOutcome::ProviderError { original, error } => {
                if Self::handle_retained_route_error(ctx, &error) {
                    if ctx.flow.content_was_streamed {
                        send_retract_reply_marker(&ctx.text_delta_tx);
                    }
                    return StepResult::Done(IterationOutcome::Continue);
                }
                warn!(
                    model = %ctx.core.model,
                    error = %error,
                    "forced_tool_recovery_provider_error_using_original_response"
                );
                original
            }
        };

        let response_ok = Self::response_status(&response) == "ok";
        if response_ok {
            if let Some(staged) = ctx.staged_auto_expansion.take() {
                if let Some(committed) =
                    commit_staged_auto_expansion(staged, &self.lcm_engines, &ctx.session_id).await
                {
                    committed.publish(
                        &mut ctx.messages,
                        &mut ctx.rendered_messages,
                        counters,
                        &ctx.session_key,
                        prompt_fp,
                    );
                    ctx.higgs_session_route.mark_expansion_published();
                } else {
                    Self::fallback_retained_expansion_route(
                        ctx,
                        RetainedExpansionFailure::Unavailable,
                    );
                }
            } else {
                let _transition = counters.lock_prompt_cache_transition();
                counters
                    .prompt_fingerprints
                    .lock()
                    .insert(ctx.session_key.clone(), prompt_fp);
                counters
                    .prompt_cache_watermark
                    .lock()
                    .insert(ctx.session_key.clone(), ctx.messages.len());
            }
        } else {
            ctx.staged_auto_expansion = None;
            Self::fallback_retained_expansion_route(ctx, RetainedExpansionFailure::Unavailable);
        }

        drop(higgs_request_reservation);
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
    ) -> ForcedToolRecoveryOutcome {
        if !should_attempt_forced_recovery(
            response.has_tool_calls(),
            ctx.core.mode().is_local(),
            ctx.flow.retries.validation,
            tool_defs_opt.is_some_and(|t| !t.is_empty()),
            response.content.as_deref(),
            ctx.protocol.is_textual_replay(),
            ctx.flow.tool_guard.had_blocked_calls,
        ) {
            return ForcedToolRecoveryOutcome::Response(response);
        }

        info!(
            model = %ctx.core.model,
            "forced_tool_recovery: botched tool intent — re-issuing with tool_choice=required"
        );
        let request = RecordedProviderRequest {
            messages: messages_for_llm.to_vec(),
            tools: tool_defs_opt.map(<[Value]>::to_vec),
            model: ctx.core.model.clone(),
            max_tokens,
            temperature: ctx.core.temperature,
            thinking_budget: None,
            top_p: None,
            tool_choice: "required".to_string(),
            streaming: false,
        };
        let call_id = match ctx
            .core
            .sessions
            .record_model_request(
                &ctx.session_id,
                &ctx.request_id,
                ctx.turn_count,
                ModelCallPurpose::ForcedToolRecovery,
                &request,
            )
            .await
        {
            Ok(call_id) => call_id,
            Err(error) => {
                return ForcedToolRecoveryOutcome::ProviderError {
                    original: response,
                    error: anyhow::anyhow!(
                        "forced-tool recovery was not sent because replay persistence failed: {error}"
                    ),
                };
            }
        };
        let recovered = ctx
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
            .await;
        match recovered {
            Ok(recovered) => {
                if let Err(error) = ctx
                    .core
                    .sessions
                    .record_model_response(
                        &ctx.session_id,
                        &ctx.request_id,
                        ctx.turn_count,
                        &call_id,
                        &RecordedProviderResponse::from(&recovered),
                    )
                    .await
                {
                    return ForcedToolRecoveryOutcome::ProviderError {
                        original: response,
                        error: anyhow::anyhow!(
                            "forced-tool recovery response could not be recorded: {error}"
                        ),
                    };
                }
                if !recovered.has_tool_calls() {
                    return ForcedToolRecoveryOutcome::Response(response);
                }
                info!("forced_tool_recovery: recovered a constrained tool call");
                if ctx.flow.content_was_streamed {
                    send_retract_reply_marker(&ctx.text_delta_tx);
                }
                ForcedToolRecoveryOutcome::Response(recovered)
            }
            Err(error) => {
                Self::persist_model_failure(ctx, &call_id, &error.to_string()).await;
                ForcedToolRecoveryOutcome::ProviderError {
                    original: response,
                    error,
                }
            }
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

        // One carrier owns every lease disposition below. Persist it before
        // either execution path can add a result, so rejected proxy calls are
        // protocol-valid and cache-replayable just like executed calls.
        crate::agent::tool_engine::journal_tool_call_carrier(ctx, &routed_tool_calls, &response)
            .await;

        // Tool-lease enforcement. Each call is recorded against the per-turn
        // lease; calls that exceed the lease budget are NOT executed — they
        // get a rejection receipt in their tool_call_id slot (preserves the
        // wire contract) and the model sees a clear renewal prompt.
        //
        // Cache-safe by design: tool_defs are never modified at execution
        // time, so the tool-block hash stays byte-stable across rounds and
        // the prefix cache hits.
        //
        // There is no longer a consecutive-same-family cap (retired 2026-07-30
        // — it over-fired on legitimate exploration and busted the cache).
        // Identical-call loops are bounded by `ToolGuard`'s per-key counter.
        let mut allowed_calls: Vec<_> = Vec::with_capacity(routed_tool_calls.len());
        let mut blocked_calls: Vec<(String, String, &'static str)> = Vec::new();
        for tc in routed_tool_calls {
            // `lease.record_tool_call` returns `lease_exhausted` when the
            // per-lease budget is gone. We do NOT pre-check `is_exhausted`
            // separately — record_tool_call is the single source of truth.
            let result = ctx.flow.lease.record_tool_call();
            if result.allowed {
                allowed_calls.push(tc);
            } else if crate::agent::tool_engine::is_read_only_tool(&tc.name)
                && ctx.flow.lease.auto_renew_for_read_only()
            {
                // Read-only tools (read_file, list_dir, etc.) auto-renew
                // without a checkpoint — they can't loop destructively and
                // the rejection+renewal dance wastes 3 round-trips on
                // legitimate multi-file exploration.
                //
                // Re-record after renewal: auto_renew_for_read_only resets
                // iterations_used to 0 but the call that triggered the
                // renewal is not yet counted against the new lease.
                // Without this the first call after every auto-renewal is
                // free, granting lease_size+1 per renewal.
                let _ = ctx.flow.lease.record_tool_call();
                tracing::info!(
                    session = %ctx.session_key,
                    tool = %tc.name,
                    "tool_lease_auto_renewed_read_only"
                );
                allowed_calls.push(tc);
            } else {
                let reason = result.reason.unwrap_or("lease_blocked");
                if let Err(error) = ctx
                    .core
                    .sessions
                    .record_tool_pre_execute(
                        &ctx.session_id,
                        &ctx.request_id,
                        ctx.turn_count,
                        &tc.id,
                        &tc.name,
                        &tc.arguments,
                        crate::session::db::ToolPreExecuteDecision::Rejected {
                            reason: format!("lease:{reason}"),
                        },
                    )
                    .await
                {
                    return StepResult::Done(IterationOutcome::Error(format!(
                        "tool {} was rejected but its pre-execution decision could not be recorded: {error}",
                        tc.id
                    )));
                }
                let name = tc.name.clone();
                let id = tc.id.clone();
                blocked_calls.push((name, id, reason));
            }
        }
        // Inject rejection receipts for blocked calls. Each receipt
        // carries the tool_call_id the model emitted, so the wire's
        // assistant-tool_calls → tool-results pairing stays intact.
        for (name, id, reason) in &blocked_calls {
            tracing::info!(
                session = %ctx.session_key,
                tool = %name,
                reason,
                "tool_lease_blocked_call"
            );
            let msg = format!(
                "lease exhausted: {name} was not executed — your per-turn \
                 tool budget is used up. Write a renewal checkpoint \
                 (findings:/next:/will:) to continue with more tools, or \
                 write your final answer."
            );
            ContextBuilder::add_tool_result_immutable_with_status(
                &mut ctx.messages,
                id,
                name,
                &msg,
                false,
            );
        }
        if !blocked_calls.is_empty() {
            ctx.persist_pending_protocol_messages().await;
        }
        let routed_tool_calls = allowed_calls;
        if routed_tool_calls.is_empty() && !blocked_calls.is_empty() {
            // Every call this round was blocked by the lease — flag it
            // so the loop machinery doesn't count this as a real
            // iteration (matches the response_boundary pattern).
            ctx.flow.round_executed_no_tools = true;
            ctx.flow.tool_guard.had_blocked_calls = true;
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
        advance_response_boundary, apply_auto_expansion_candidates, commit_staged_auto_expansion,
        divergent_message_digest,
        materialize_auto_expansion, proactive_grounding_preserves_prefix_cache,
        AppliedAutoExpansion, AutoExpansionMaterializationKind, HiggsSessionRoute,
        PromptCacheSnapshot, ResponseBoundary, RetainedExpansionFailure,
    };
    use crate::agent::agent_core::{
        stable_higgs_session_id, ExpansionCheckpoint, RuntimeCounters, SessionRetirement,
        ToolPresentationMode,
    };
    use crate::agent::agent_loop::budget::attach_higgs_session_marker;
    use crate::agent::lcm::AutoExpansionCandidate;
    use crate::agent::protocol::CloudProtocol;
    use crate::agent::token_budget::TokenBudget;
    use crate::config::schema::CircuitBreakerConfig;
    use serde_json::{json, Value};
    use std::sync::Arc;

    #[test]
    fn exact_auto_expansion_replaces_one_summary_in_place_with_original_tool_sequence() {
        let summary = json!({
            "role": "user",
            "content": "summary placeholder",
            "_lcm_summary": true,
        });
        let replaced_span = vec![
            json!({
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{\"path\":\"a.rs\"}"},
                }],
                "_db_id": 11,
            }),
            json!({
                "role": "tool",
                "tool_call_id": "call-1",
                "name": "read_file",
                "content": "exact bytes",
                "_db_id": 12,
            }),
            json!({
                "role": "assistant",
                "content": "The file contains exact bytes.",
                "_db_id": 13,
            }),
        ];
        let logical = vec![
            json!({"role": "system", "content": "system"}),
            json!({"role": "user", "content": "before", "_db_id": 10}),
            summary.clone(),
            json!({"role": "user", "content": "after", "_db_id": 14}),
        ];
        let candidate = AutoExpansionCandidate {
            node_id: 7,
            source_ids: vec![11, 12, 13],
            estimated_tokens: 42,
            flattened_fallback: json!({
                "role": "user",
                "content": "flattened fallback",
                "_synthetic": true,
            }),
            summary_message: summary,
        };
        let checkpoint = ExpansionCheckpoint {
            old_higgs_session_id: 91,
            summary_node_id: 7,
            replaced_span: replaced_span.clone(),
            frozen_tool_hash: 1,
            model: "bonsai".to_string(),
            presentation_mode: ToolPresentationMode::Native,
            catalog_generation: 1,
            expires_at_ms: 900_000,
            lease_confirmed: true,
        };

        let materialized = materialize_auto_expansion(&logical, &candidate, Some(&checkpoint));

        assert_eq!(
            materialized.kind,
            AutoExpansionMaterializationKind::ExactCheckpoint
        );
        assert_eq!(
            materialized.messages,
            vec![
                json!({"role": "system", "content": "system"}),
                json!({"role": "user", "content": "before", "_db_id": 10}),
                replaced_span[0].clone(),
                replaced_span[1].clone(),
                replaced_span[2].clone(),
                json!({"role": "user", "content": "after", "_db_id": 14}),
            ],
            "exact expansion must preserve the original assistant/tool ordering at the summary index"
        );
    }

    #[test]
    fn unconfirmed_mismatched_or_incomplete_checkpoint_uses_flattened_fallback() {
        let summary = json!({
            "role": "user",
            "content": "summary placeholder",
            "_lcm_summary": true,
        });
        let logical = vec![
            json!({"role": "system", "content": "system"}),
            summary.clone(),
            json!({"role": "user", "content": "latest"}),
        ];
        let candidate = AutoExpansionCandidate {
            node_id: 7,
            source_ids: vec![11, 12],
            estimated_tokens: 20,
            flattened_fallback: json!({
                "role": "user",
                "content": "flattened fallback",
                "_synthetic": true,
            }),
            summary_message: summary,
        };
        let base = ExpansionCheckpoint {
            old_higgs_session_id: 91,
            summary_node_id: 7,
            replaced_span: vec![
                json!({"role": "user", "content": "raw one", "_db_id": 11}),
                json!({"role": "assistant", "content": "raw two", "_db_id": 12}),
            ],
            frozen_tool_hash: 1,
            model: "bonsai".to_string(),
            presentation_mode: ToolPresentationMode::Native,
            catalog_generation: 1,
            expires_at_ms: 900_000,
            lease_confirmed: true,
        };
        let mut unconfirmed = base.clone();
        unconfirmed.lease_confirmed = false;
        let mut node_mismatch = base.clone();
        node_mismatch.summary_node_id = 8;
        let mut incomplete = base.clone();
        incomplete.replaced_span.pop();

        for (case, checkpoint) in [
            ("absent", None),
            ("unconfirmed", Some(&unconfirmed)),
            ("node mismatch", Some(&node_mismatch)),
            ("incomplete raw coverage", Some(&incomplete)),
        ] {
            let materialized = materialize_auto_expansion(&logical, &candidate, checkpoint);
            assert_eq!(
                materialized.kind,
                AutoExpansionMaterializationKind::FlattenedFallback,
                "case {case}"
            );
            assert_eq!(
                materialized.messages[..logical.len()],
                logical,
                "case {case}"
            );
            assert_eq!(
                materialized.messages.last(),
                Some(&candidate.flattened_fallback),
                "case {case}"
            );
        }
    }

    #[test]
    fn auto_expansion_application_renders_the_whole_prompt_and_rejects_overflow() {
        let summary = json!({
            "role": "user",
            "content": "summary placeholder",
            "_lcm_summary": true,
        });
        let logical = vec![
            json!({"role": "system", "content": "system"}),
            json!({"role": "user", "content": "before", "_db_id": 10}),
            summary.clone(),
            json!({"role": "user", "content": "after", "_db_id": 13}),
        ];
        let candidate = AutoExpansionCandidate {
            node_id: 7,
            source_ids: vec![11, 12],
            estimated_tokens: 20,
            flattened_fallback: json!({
                "role": "user",
                "content": "flattened fallback",
                "_synthetic": true,
            }),
            summary_message: summary,
        };
        let checkpoint = ExpansionCheckpoint {
            old_higgs_session_id: 91,
            summary_node_id: 7,
            replaced_span: vec![
                json!({"role": "assistant", "content": "exact answer", "_db_id": 11}),
                json!({"role": "user", "content": "exact follow-up", "_db_id": 12}),
            ],
            frozen_tool_hash: 1,
            model: "bonsai".to_string(),
            presentation_mode: ToolPresentationMode::Native,
            catalog_generation: 1,
            expires_at_ms: 900_000,
            lease_confirmed: true,
        };

        let applied = apply_auto_expansion_candidates(
            &CloudProtocol,
            &logical,
            &[candidate.clone()],
            Some(&checkpoint),
            0,
            10_000,
        )
        .expect("the exact reconstructed prompt fits");
        assert_eq!(applied.node_ids, vec![7]);
        assert_eq!(
            applied.rendered_messages,
            vec![
                json!({"role": "system", "content": "system"}),
                json!({"role": "user", "content": "before"}),
                json!({"role": "assistant", "content": "exact answer"}),
                json!({"role": "user", "content": "exact follow-up"}),
                json!({"role": "user", "content": "after"}),
            ],
            "the entire reconstructed logical array must pass through CloudProtocol"
        );

        assert!(apply_auto_expansion_candidates(
            &CloudProtocol,
            &logical,
            &[candidate],
            Some(&checkpoint),
            0,
            1,
        )
        .is_none());
        assert_eq!(
            logical,
            vec![
                json!({"role": "system", "content": "system"}),
                json!({"role": "user", "content": "before", "_db_id": 10}),
                json!({"role": "user", "content": "summary placeholder", "_lcm_summary": true}),
                json!({"role": "user", "content": "after", "_db_id": 13}),
            ],
            "oversize planning must not rewrite the compacted logical prompt"
        );
    }

    #[test]
    fn retained_route_fallback_preserves_post_expansion_tool_tail() {
        let checkpoint = ExpansionCheckpoint {
            old_higgs_session_id: 91,
            summary_node_id: 7,
            replaced_span: vec![json!({"role": "user", "content": "exact", "_db_id": 11})],
            frozen_tool_hash: 1,
            model: "bonsai".to_string(),
            presentation_mode: ToolPresentationMode::Native,
            catalog_generation: 1,
            expires_at_ms: 900_000,
            lease_confirmed: true,
        };
        let compacted_prefix = vec![
            json!({"role": "system", "content": "system"}),
            json!({"role": "user", "content": "summary"}),
        ];
        let fallback_prefix = vec![
            compacted_prefix[0].clone(),
            compacted_prefix[1].clone(),
            json!({"role": "user", "content": "flattened fallback", "_synthetic": true}),
        ];
        let exact_prefix = vec![
            compacted_prefix[0].clone(),
            json!({"role": "user", "content": "exact", "_db_id": 11}),
        ];
        let tail = vec![
            json!({"role": "assistant", "content": null, "tool_calls": [{"id": "tc", "type": "function", "function": {"name": "read", "arguments": "{}"}}]}),
            json!({"role": "tool", "tool_call_id": "tc", "content": "result"}),
        ];

        for (failure, expected_prefix) in [
            (RetainedExpansionFailure::Unavailable, &fallback_prefix),
            (RetainedExpansionFailure::ContextOverflow, &compacted_prefix),
        ] {
            let mut route = HiggsSessionRoute::retained_expansion(
                checkpoint.clone(),
                compacted_prefix.clone(),
                Some(fallback_prefix.clone()),
                exact_prefix.len(),
                PromptCacheSnapshot::default(),
            );
            route.mark_expansion_published();
            let mut logical = exact_prefix.clone();
            logical.extend(tail.clone());
            let mut rendered = Vec::new();

            let retired = route
                .fallback_from_retained(failure, &mut logical, &mut rendered, &CloudProtocol)
                .expect("retained route must return its checkpoint for deletion");

            assert_eq!(retired.0, checkpoint);
            assert_eq!(route.cache_route(), "fallback");
            assert_eq!(
                &logical[..expected_prefix.len()],
                expected_prefix.as_slice()
            );
            assert_eq!(&logical[expected_prefix.len()..], tail.as_slice());
            assert_eq!(
                rendered,
                crate::agent::agent_loop::render_via_protocol(&CloudProtocol, &logical)
            );
        }
    }

    #[test]
    fn exact_fit_with_oversized_flattened_variant_keeps_summary_only_fallback() {
        let summary = json!({
            "role": "user",
            "content": "summary placeholder",
            "_lcm_summary": true,
        });
        let compacted = vec![
            json!({"role": "system", "content": "system"}),
            summary.clone(),
            json!({"role": "user", "content": "latest"}),
        ];
        let candidate = AutoExpansionCandidate {
            node_id: 7,
            source_ids: vec![11],
            estimated_tokens: 1,
            flattened_fallback: json!({
                "role": "user",
                "content": "oversized flattened detail ".repeat(4_000),
                "_synthetic": true,
            }),
            summary_message: summary,
        };
        let checkpoint = ExpansionCheckpoint {
            old_higgs_session_id: 91,
            summary_node_id: 7,
            replaced_span: vec![json!({
                "role": "user",
                "content": "tiny exact detail",
                "_db_id": 11,
            })],
            frozen_tool_hash: 1,
            model: "bonsai".to_string(),
            presentation_mode: ToolPresentationMode::Native,
            catalog_generation: 1,
            expires_at_ms: 900_000,
            lease_confirmed: true,
        };
        let exact = materialize_auto_expansion(&compacted, &candidate, Some(&checkpoint));
        let exact_limit = TokenBudget::estimate_tokens(
            &crate::agent::agent_loop::render_via_protocol(&CloudProtocol, &exact.messages),
        );
        let mut applied = apply_auto_expansion_candidates(
            &CloudProtocol,
            &compacted,
            &[candidate],
            Some(&checkpoint),
            0,
            exact_limit,
        )
        .expect("the exact route fits its authoritative prompt limit");
        let plan = applied
            .retained_plan()
            .expect("exact route must be retained");
        assert!(
            plan.fallback_prefix.is_none(),
            "the independently oversized flattened variant must not be staged"
        );
        assert!(!applied.use_flattened_fallback(&CloudProtocol));
        assert_eq!(applied.logical_messages, exact.messages);
    }

    #[test]
    fn leaving_retained_route_restores_active_fingerprint_and_watermark_together() {
        let counters = RuntimeCounters::new_with_config(32_768, &CircuitBreakerConfig::default());
        let session = "cli:retained-cache-snapshot";
        let active = crate::agent::prompt_fingerprint::fingerprint(&[
            json!({"role": "system", "content": "system"}),
            json!({"role": "user", "content": "summary"}),
        ]);
        counters
            .prompt_fingerprints
            .lock()
            .insert(session.to_string(), active.clone());
        counters
            .prompt_cache_watermark
            .lock()
            .insert(session.to_string(), 2);
        let snapshot = PromptCacheSnapshot::capture(&counters, session);

        counters.prompt_fingerprints.lock().insert(
            session.to_string(),
            crate::agent::prompt_fingerprint::fingerprint(&[
                json!({"role": "system", "content": "system"}),
                json!({"role": "user", "content": "exact"}),
            ]),
        );
        counters
            .prompt_cache_watermark
            .lock()
            .insert(session.to_string(), 99);
        snapshot.restore(&counters, session);

        assert_eq!(
            counters.prompt_fingerprints.lock().get(session),
            Some(&active)
        );
        assert_eq!(
            counters.prompt_cache_watermark.lock().get(session),
            Some(&2)
        );
    }

    #[test]
    fn reset_before_retained_fallback_does_not_restore_stale_active_cache_state() {
        let counters = RuntimeCounters::new_with_config(32_768, &CircuitBreakerConfig::default());
        let session = "cli:retained-cache-reset";
        let durable = "sqlite:retained-cache-reset";
        counters.install_tool_catalog(session, ToolPresentationMode::Native, Vec::new());
        let old_active = counters.activate_higgs_session_id(session, durable);
        let fingerprint = crate::agent::prompt_fingerprint::fingerprint(&[
            json!({"role": "system", "content": "system"}),
            json!({"role": "user", "content": "summary"}),
        ]);
        counters
            .prompt_fingerprints
            .lock()
            .insert(session.to_string(), fingerprint);
        counters
            .prompt_cache_watermark
            .lock()
            .insert(session.to_string(), 2);
        let snapshot = PromptCacheSnapshot::capture_and_clear(&counters, session);

        counters.reset_session_prompt_state(session);
        counters.install_tool_catalog(session, ToolPresentationMode::Native, Vec::new());
        let fresh_active = counters.activate_higgs_session_id(session, durable);
        assert_ne!(fresh_active, old_active);
        snapshot.restore(&counters, session);

        assert_eq!(counters.prompt_fingerprints.lock().get(session), None);
        assert_eq!(counters.prompt_cache_watermark.lock().get(session), None);
    }

    #[test]
    fn concurrent_reset_after_snapshot_identity_observation_wins_restore_transaction() {
        let counters = Arc::new(RuntimeCounters::new_with_config(
            32_768,
            &CircuitBreakerConfig::default(),
        ));
        let session = "cli:retained-cache-restore-race";
        let durable = "sqlite:retained-cache-restore-race";
        counters.install_tool_catalog(session, ToolPresentationMode::Native, Vec::new());
        counters.activate_higgs_session_id(session, durable);
        let stale_fingerprint = crate::agent::prompt_fingerprint::fingerprint(&[
            json!({"role": "system", "content": "stale system"}),
            json!({"role": "user", "content": "stale summary"}),
        ]);
        counters
            .prompt_fingerprints
            .lock()
            .insert(session.to_string(), stale_fingerprint);
        counters
            .prompt_cache_watermark
            .lock()
            .insert(session.to_string(), 2);
        let snapshot = PromptCacheSnapshot::capture_and_clear(&counters, session);
        let identity_observed = Arc::new(std::sync::Barrier::new(2));
        let release_restore = Arc::new(std::sync::Barrier::new(2));
        let restore_thread = {
            let counters = Arc::clone(&counters);
            let identity_observed = Arc::clone(&identity_observed);
            let release_restore = Arc::clone(&release_restore);
            std::thread::spawn(move || {
                snapshot.restore_observed(&counters, session, || {
                    identity_observed.wait();
                    release_restore.wait();
                });
            })
        };
        identity_observed.wait();

        let reset_started = Arc::new(std::sync::Barrier::new(2));
        let (reset_done_tx, reset_done_rx) = std::sync::mpsc::channel();
        let reset_thread = {
            let counters = Arc::clone(&counters);
            let reset_started = Arc::clone(&reset_started);
            std::thread::spawn(move || {
                reset_started.wait();
                counters.reset_session_prompt_state(session);
                counters.install_tool_catalog(session, ToolPresentationMode::Native, Vec::new());
                counters.activate_higgs_session_id(session, durable);
                reset_done_tx.send(()).unwrap();
            })
        };
        reset_started.wait();
        // Without a shared transition lock the reset completes inside the
        // observed check→write gap. With the fixed transaction it blocks here
        // until restore is released, then clears the just-restored state.
        let _ = reset_done_rx.recv_timeout(std::time::Duration::from_millis(100));
        release_restore.wait();
        restore_thread.join().unwrap();
        reset_thread.join().unwrap();

        assert_eq!(counters.prompt_fingerprints.lock().get(session), None);
        assert_eq!(counters.prompt_cache_watermark.lock().get(session), None);

        let fresh_fingerprint = crate::agent::prompt_fingerprint::fingerprint(&[
            json!({"role": "system", "content": "fresh system"}),
            json!({"role": "user", "content": "fresh prompt"}),
        ]);
        counters
            .prompt_fingerprints
            .lock()
            .insert(session.to_string(), fresh_fingerprint.clone());
        counters
            .prompt_cache_watermark
            .lock()
            .insert(session.to_string(), 2);
        let fresh_snapshot = PromptCacheSnapshot::capture_and_clear(&counters, session);
        fresh_snapshot.restore(&counters, session);
        assert_eq!(
            counters.prompt_fingerprints.lock().get(session),
            Some(&fresh_fingerprint)
        );
        assert_eq!(
            counters.prompt_cache_watermark.lock().get(session),
            Some(&2)
        );
    }

    #[test]
    fn concurrent_colliding_reservation_cannot_publish_prior_route_cache_state() {
        let counters = Arc::new(RuntimeCounters::new_with_config(
            32_768,
            &CircuitBreakerConfig::default(),
        ));
        let session = "cli:retained-cache-reservation-race";
        let prior_durable = "sqlite:retained-cache-prior";
        let changed_durable = "sqlite:retained-cache-changed";
        let model = "bonsai";
        let definitions = Vec::new();
        let tool_hash = crate::agent::prompt_fingerprint::hash_tools(&definitions);
        counters.install_tool_catalog(session, ToolPresentationMode::Native, definitions);

        // Retirement advances the epoch to one. Retain exactly the ID the
        // changed durable identity would derive at that epoch, forcing request
        // activation to advance the epoch again while selecting its active ID.
        let colliding_id = stable_higgs_session_id(changed_durable, 1);
        assert!(counters.record_higgs_session_id(session, colliding_id));
        counters.retire_higgs_session(
            session,
            SessionRetirement::LeaseForExpansion {
                summary_node_id: 73,
                replaced_span: vec![json!({"role": "user", "content": "exact raw"})],
                checkpoint_context: counters
                    .expansion_checkpoint_context(session, model, tool_hash, 900_000)
                    .unwrap(),
            },
        );
        let prior_active = counters.activate_higgs_session_id(session, prior_durable);
        assert_ne!(prior_active, colliding_id);

        let stale_fingerprint = crate::agent::prompt_fingerprint::fingerprint(&[
            json!({"role": "system", "content": "prior system"}),
            json!({"role": "user", "content": "prior summary"}),
        ]);
        counters
            .prompt_fingerprints
            .lock()
            .insert(session.to_string(), stale_fingerprint);
        counters
            .prompt_cache_watermark
            .lock()
            .insert(session.to_string(), 2);
        let snapshot = PromptCacheSnapshot::capture_and_clear(&counters, session);

        let identity_observed = Arc::new(std::sync::Barrier::new(2));
        let release_restore = Arc::new(std::sync::Barrier::new(2));
        let restore_thread = {
            let counters = Arc::clone(&counters);
            let identity_observed = Arc::clone(&identity_observed);
            let release_restore = Arc::clone(&release_restore);
            std::thread::spawn(move || {
                snapshot.restore_observed(&counters, session, || {
                    identity_observed.wait();
                    release_restore.wait();
                });
            })
        };
        identity_observed.wait();

        let reservation_started = Arc::new(std::sync::Barrier::new(2));
        let (reservation_done_tx, reservation_done_rx) = std::sync::mpsc::channel();
        let reservation_thread = {
            let counters = Arc::clone(&counters);
            let reservation_started = Arc::clone(&reservation_started);
            std::thread::spawn(move || {
                reservation_started.wait();
                let reservation = counters.reserve_higgs_session_request(
                    session,
                    changed_durable,
                    model,
                    tool_hash,
                    31_744,
                    600_000,
                );
                reservation_done_tx.send(reservation.active_id()).unwrap();
            })
        };
        reservation_started.wait();
        // The unfixed reservation can complete in this check→write gap. The
        // fixed path waits for the transition, then invalidates the old route's
        // restored cache maps as part of installing the changed active ID.
        let completed_before_release = reservation_done_rx
            .recv_timeout(std::time::Duration::from_millis(100))
            .ok();
        release_restore.wait();
        restore_thread.join().unwrap();
        let active_id =
            completed_before_release.unwrap_or_else(|| reservation_done_rx.recv().unwrap());
        reservation_thread.join().unwrap();

        assert_eq!(active_id, stable_higgs_session_id(changed_durable, 2));
        assert_eq!(counters.session_prompt_epoch(session), 2);
        assert_eq!(counters.active_higgs_session_id(session), Some(active_id));
        assert_eq!(counters.prompt_fingerprints.lock().get(session), None);
        assert_eq!(counters.prompt_cache_watermark.lock().get(session), None);
    }

    #[test]
    fn failed_empty_and_retried_expansion_attempts_remain_request_local() {
        let summary = json!({
            "role": "user",
            "content": "summary placeholder",
            "_lcm_summary": true,
        });
        let compacted = vec![
            json!({"role": "system", "content": "system"}),
            summary.clone(),
            json!({"role": "user", "content": "latest"}),
        ];
        let compacted_wire =
            crate::agent::agent_loop::render_via_protocol(&CloudProtocol, &compacted);
        let candidate = AutoExpansionCandidate {
            node_id: 7,
            source_ids: vec![11],
            estimated_tokens: 10,
            flattened_fallback: json!({
                "role": "user",
                "content": "flattened fallback",
                "_synthetic": true,
            }),
            summary_message: summary,
        };

        for terminal_path in ["provider error", "empty response", "retry"] {
            let logical = compacted.clone();
            let rendered = compacted_wire.clone();
            let staged = apply_auto_expansion_candidates(
                &CloudProtocol,
                &logical,
                &[candidate.clone()],
                None,
                0,
                10_000,
            )
            .unwrap();
            assert!(staged.rendered_messages.iter().any(|message| {
                message
                    .get("content")
                    .and_then(Value::as_str)
                    .is_some_and(|content| content.contains("flattened fallback"))
            }));

            drop(staged);

            assert_eq!(logical, compacted, "case {terminal_path}");
            assert_eq!(rendered, compacted_wire, "case {terminal_path}");
        }
    }

    fn expansion_engine(node_ids: &[usize]) -> crate::agent::lcm::LcmEngine {
        let raw_messages = node_ids
            .iter()
            .map(|node_id| {
                json!({
                    "role": "user",
                    "content": format!("source for node {node_id}"),
                    "_db_id": node_id + 100,
                })
            })
            .collect::<Vec<_>>();
        let nodes = node_ids
            .iter()
            .map(|node_id| {
                (
                    *node_id,
                    vec![node_id + 100],
                    Vec::new(),
                    format!("summary node {node_id}"),
                    4,
                    1,
                    crate::agent::lcm::SummaryManifest::default(),
                    "db_id".to_string(),
                )
            })
            .collect::<Vec<_>>();
        crate::agent::lcm::LcmEngine::rebuild_from_db_nodes(
            &raw_messages,
            &nodes,
            crate::agent::lcm::LcmConfig::default(),
        )
    }

    fn staged_expansion(node_ids: Vec<usize>) -> AppliedAutoExpansion {
        AppliedAutoExpansion {
            logical_messages: vec![json!({"role": "user", "content": "expanded logical"})],
            rendered_messages: vec![json!({"role": "user", "content": "expanded wire"})],
            node_ids,
            exact_count: 0,
            estimated_added_tokens: 4,
            retained_plan: None,
        }
    }

    #[tokio::test]
    async fn cancellation_before_expansion_engine_lock_preserves_all_public_state() {
        let session_id = "cancelled-expansion";
        let session_key = "cli:cancelled-expansion";
        let engine = std::sync::Arc::new(tokio::sync::Mutex::new(expansion_engine(&[7])));
        let engines =
            std::sync::Arc::new(tokio::sync::Mutex::new(std::collections::HashMap::from([
                (session_id.to_string(), std::sync::Arc::clone(&engine)),
            ])));
        let engine_guard = engine.lock().await;
        let counters = RuntimeCounters::new_with_config(16_384, &CircuitBreakerConfig::default());
        let compacted = vec![json!({"role": "user", "content": "compacted"})];
        let compacted_wire = compacted.clone();
        let old_fingerprint = crate::agent::prompt_fingerprint::fingerprint(&compacted_wire);
        counters
            .prompt_fingerprints
            .lock()
            .insert(session_key.to_string(), old_fingerprint.clone());
        counters
            .prompt_cache_watermark
            .lock()
            .insert(session_key.to_string(), compacted.len());

        let mut commit = Box::pin(commit_staged_auto_expansion(
            staged_expansion(vec![7]),
            &engines,
            session_id,
        ));
        std::future::poll_fn(|cx| match std::future::Future::poll(commit.as_mut(), cx) {
            std::task::Poll::Pending => std::task::Poll::Ready(()),
            std::task::Poll::Ready(_) => {
                panic!("commit completed while the per-session engine lock was held")
            }
        })
        .await;
        drop(commit);

        assert_eq!(
            compacted,
            vec![json!({"role": "user", "content": "compacted"})]
        );
        assert_eq!(compacted_wire, compacted);
        assert_eq!(
            counters.prompt_fingerprints.lock().get(session_key),
            Some(&old_fingerprint)
        );
        assert_eq!(
            counters
                .prompt_cache_watermark
                .lock()
                .get(session_key)
                .copied(),
            Some(1)
        );

        drop(engine_guard);
        assert!(engine.lock().await.commit_auto_expansion(7));
    }

    #[tokio::test]
    async fn missing_or_false_expansion_commit_keeps_compacted_prompt_and_cache() {
        let session_key = "cli:failed-expansion-commit";
        let counters = RuntimeCounters::new_with_config(16_384, &CircuitBreakerConfig::default());
        let logical = vec![json!({"role": "user", "content": "compacted"})];
        let rendered = logical.clone();
        let old_fingerprint = crate::agent::prompt_fingerprint::fingerprint(&rendered);
        counters
            .prompt_fingerprints
            .lock()
            .insert(session_key.to_string(), old_fingerprint.clone());
        counters
            .prompt_cache_watermark
            .lock()
            .insert(session_key.to_string(), 1);

        let empty_engines =
            std::sync::Arc::new(tokio::sync::Mutex::new(std::collections::HashMap::new()));
        assert!(
            commit_staged_auto_expansion(staged_expansion(vec![7]), &empty_engines, "missing",)
                .await
                .is_none()
        );

        let mut engine = expansion_engine(&[7, 8]);
        assert!(engine.commit_auto_expansion(7));
        let engine = std::sync::Arc::new(tokio::sync::Mutex::new(engine));
        let engines =
            std::sync::Arc::new(tokio::sync::Mutex::new(std::collections::HashMap::from([
                ("present".to_string(), std::sync::Arc::clone(&engine)),
            ])));
        assert!(
            commit_staged_auto_expansion(staged_expansion(vec![8, 7]), &engines, "present",)
                .await
                .is_none()
        );

        assert_eq!(
            logical,
            vec![json!({"role": "user", "content": "compacted"})]
        );
        assert_eq!(rendered, logical);
        assert_eq!(
            counters.prompt_fingerprints.lock().get(session_key),
            Some(&old_fingerprint)
        );
        assert_eq!(
            counters
                .prompt_cache_watermark
                .lock()
                .get(session_key)
                .copied(),
            Some(1)
        );
        assert!(
            engine.lock().await.commit_auto_expansion(8),
            "a failed batch must not partially consume another node"
        );
    }

    #[tokio::test]
    async fn successful_expansion_commit_publishes_prompt_and_cache_once() {
        let session_id = "successful-expansion";
        let session_key = "cli:successful-expansion";
        let engine = std::sync::Arc::new(tokio::sync::Mutex::new(expansion_engine(&[7])));
        let engines =
            std::sync::Arc::new(tokio::sync::Mutex::new(std::collections::HashMap::from([
                (session_id.to_string(), std::sync::Arc::clone(&engine)),
            ])));
        let counters = RuntimeCounters::new_with_config(16_384, &CircuitBreakerConfig::default());
        let mut logical = vec![json!({"role": "user", "content": "compacted"})];
        let mut rendered = logical.clone();
        let compacted_fingerprint = crate::agent::prompt_fingerprint::fingerprint(&rendered);
        counters
            .prompt_fingerprints
            .lock()
            .insert(session_key.to_string(), compacted_fingerprint.clone());
        counters
            .prompt_cache_watermark
            .lock()
            .insert(session_key.to_string(), logical.len());
        let expanded_fingerprint = crate::agent::prompt_fingerprint::fingerprint(&[
            json!({"role": "user", "content": "expanded wire"}),
        ]);

        let committed =
            commit_staged_auto_expansion(staged_expansion(vec![7]), &engines, session_id)
                .await
                .expect("known eligible node commits");
        assert_eq!(
            logical,
            vec![json!({"role": "user", "content": "compacted"})]
        );
        assert_eq!(rendered, logical);
        assert_eq!(
            counters.prompt_fingerprints.lock().get(session_key),
            Some(&compacted_fingerprint),
            "the awaitable commit phase must not publish cache state"
        );

        committed.publish(
            &mut logical,
            &mut rendered,
            &counters,
            session_key,
            expanded_fingerprint.clone(),
        );

        assert_eq!(
            logical,
            vec![json!({"role": "user", "content": "expanded logical"})]
        );
        assert_eq!(
            rendered,
            vec![json!({"role": "user", "content": "expanded wire"})]
        );
        assert_eq!(
            counters.prompt_fingerprints.lock().get(session_key),
            Some(&expanded_fingerprint)
        );
        assert_eq!(
            counters
                .prompt_cache_watermark
                .lock()
                .get(session_key)
                .copied(),
            Some(1)
        );
        assert!(!engine.lock().await.commit_auto_expansion(7));
    }

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

    #[test]
    fn unsent_higgs_request_id_is_not_repurposed_as_expansion_checkpoint() {
        let counters = std::sync::Arc::new(RuntimeCounters::new_with_config(
            16_384,
            &CircuitBreakerConfig::default(),
        ));
        let session_key = "cli:reserved-request";
        let definitions = vec![json!({"type": "function", "name": "read"})];
        counters.install_tool_catalog(
            session_key,
            ToolPresentationMode::Native,
            definitions.clone(),
        );
        let tool_hash = crate::agent::prompt_fingerprint::hash_tools(&definitions);
        let reservation = counters.reserve_higgs_session_request(
            session_key,
            "sqlite:reserved-request",
            "bonsai",
            tool_hash,
            15_000,
            600_000,
        );
        let emitted_active_id = reservation.active_id();

        let retiring = std::sync::Arc::clone(&counters);
        std::thread::spawn(move || {
            retiring.retire_higgs_session(
                session_key,
                SessionRetirement::LeaseForExpansion {
                    summary_node_id: 41,
                    replaced_span: vec![json!({"role": "user", "content": "exact raw"})],
                    checkpoint_context: retiring
                        .expansion_checkpoint_context(session_key, "bonsai", tool_hash, 900_000)
                        .unwrap(),
                },
            );
        })
        .join()
        .unwrap();

        let mut messages = vec![json!({"role": "system", "content": "system"})];
        attach_higgs_session_marker(&mut messages, emitted_active_id, reservation.drop_ids());
        let wire_active_id = messages[0]
            [crate::providers::openai_compat::NANOBOT_HIGGS_SESSION_ID_FIELD]
            .as_u64()
            .unwrap();

        assert_eq!(wire_active_id, emitted_active_id);
        assert_ne!(
            counters
                .expansion_checkpoint(session_key)
                .map(|checkpoint| checkpoint.old_higgs_session_id),
            Some(wire_active_id),
            "an ID reserved for an unsent request became the expansion checkpoint"
        );
        assert!(
            !reservation.drop_ids().contains(&wire_active_id),
            "the same request must not emit its active ID as a drop"
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
    use super::{should_attempt_forced_recovery, ProviderRequestRetryPolicy};

    // Real trigger strings (mirror src/agent/validation.rs tests).
    const CLAIMED: &str = "Let me check that file for you."; // prose only — NOT an error
    const HALLUCINATED: &str = "I'll read it.\n[Called read_file({\"path\":\"/x\"})]"; // HallucinatedToolCall
    const CLEAN: &str = "The answer is 42."; // Ok — a genuine final answer

    #[test]
    fn one_shot_lease_request_disables_outer_retry() {
        assert!(!ProviderRequestRetryPolicy::OneShotLease.allows_retry());
        assert!(ProviderRequestRetryPolicy::Standard.allows_retry());
    }

    /// Forced recovery re-issues the turn with `tool_choice=required`, which
    /// throws away whatever the model just wrote. Prose that merely sounds
    /// like tool intent is indistinguishable from a finished answer, so it
    /// must NOT trigger that. Only fabricated call syntax does.
    #[test]
    fn does_not_fire_on_tool_intent_prose() {
        assert!(!should_attempt_forced_recovery(
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
