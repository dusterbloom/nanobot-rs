// Error-protocol layer-3 backlog (docs/research/2026-08-06-error-conventions-and-host-bridge.md §3.6):
// the deny regime in Cargo.toml is live; this module still carries pre-existing
// violations of the lints below. Remove this allow as the module migrates onto
// the regime.
// Tracking: docs/error-protocol-backlog.md
#![allow(
    clippy::as_conversions,
    clippy::indexing_slicing,
    clippy::shadow_unrelated
)]
//! Core types shared across the agent system.
//!
//! Extracted from `agent_loop.rs` to reduce file size and improve modularity.
//! Contains: SwappableCore, RuntimeCounters, AgentHandle, build helpers, and
//! compaction utilities.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicU8, Ordering};
use std::sync::Arc;

use serde_json::Value;
use tracing::debug;

use crate::agent::agent_profiles;
use crate::agent::circuit_breaker::CircuitBreaker;
use crate::agent::compaction::ContextCompactor;
use crate::agent::context::ContextBuilder;
use crate::agent::lane::Lane;
use crate::agent::runtime_mode::RuntimeMode;
use crate::agent::token_budget::TokenBudget;
use crate::agent::working_memory::WorkingMemoryStore;
use crate::config::schema::{
    AdaptiveTokenConfig, CircuitBreakerConfig, MemoryConfig, ProvenanceConfig,
    ToolDelegationConfig, TrioConfig,
};
use crate::providers::base::LLMProvider;
use crate::session::db::SessionDb;

const LOCAL_ARTIFACT_INTENT_TTL_TURNS: u64 = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LocalArtifactIntentState {
    is_rich: bool,
    expires_after_turn: u64,
}

// ---------------------------------------------------------------------------
// Shared core (identical across all agents, swappable on /local toggle)
// ---------------------------------------------------------------------------

/// Fields that change on `/local` and `/model` — behind `Arc<RwLock<Arc<>>>`.
///
/// When the user toggles `/local` or `/model`, a new `SwappableCore` is built
/// and swapped into the handle so every agent sees the change.
pub struct SwappableCore {
    pub provider: Arc<dyn LLMProvider>,
    pub workspace: PathBuf,
    pub model: String,
    pub max_iterations: u32,
    pub max_continuations: u32,
    pub max_tokens: u32,
    pub temperature: f64,
    pub context: ContextBuilder,
    pub sessions: Arc<SessionDb>,
    pub token_budget: TokenBudget,
    pub compactor: ContextCompactor,
    pub working_memory: WorkingMemoryStore,
    pub working_memory_budget: usize,
    pub brave_api_key: Option<String>,
    pub search_provider: String,
    pub searxng_url: String,
    /// Base URL of a local crw-server for web_fetch; empty = disabled.
    pub crw_url: String,
    pub search_max_results: u32,
    pub exec_timeout: u64,
    pub restrict_to_workspace: bool,
    pub memory_enabled: bool,
    /// Provider/model used only for durable-memory reflection. LCM compaction
    /// always follows `provider` + `model`, so context policy cannot drift
    /// onto a second model with a different context window.
    pub memory_provider: Arc<dyn LLMProvider>,
    pub memory_model: String,
    pub reflection_threshold: usize,
    /// Typed runtime descriptor. Single source of truth for "is this a local
    /// backend?" via `mode().is_local()`. The legacy `is_local: bool` field was
    /// removed in R6 — it duplicated information already carried by this enum.
    pub mode: RuntimeMode,
    pub lane: Lane,
    pub tool_runner_provider: Option<Arc<dyn LLMProvider>>,
    pub tool_runner_model: Option<String>,
    pub router_provider: Option<Arc<dyn LLMProvider>>,
    pub router_model: Option<String>,
    pub router_no_think: bool,
    pub router_temperature: f64,
    pub router_top_p: f64,
    pub specialist_provider: Option<Arc<dyn LLMProvider>>,
    pub specialist_model: Option<String>,
    pub specialist_temperature: f64,
    pub specialist_top_p: f64,
    pub tool_delegation_config: ToolDelegationConfig,
    pub provenance_config: ProvenanceConfig,
    pub max_tool_result_chars: usize,
    pub session_complete_after_secs: u64,
    pub max_history_turns: usize,
    pub model_capabilities: crate::agent::model_capabilities::ModelCapabilities,
    /// Single owner of hygiene/anti-drift/budget-trim retention knobs. See
    /// `agent::retention` — replaces the formerly separate `anti_drift`,
    /// `max_message_age_turns`, and `hygiene_keep_last_messages` fields.
    pub retention: crate::agent::retention::RetentionPolicy,
    /// When true, specialist is instructed to return strict JSON and the response
    /// is parsed as `SpecialistResponse`. Sourced from `TrioConfig::specialist_output_schema`.
    pub specialist_output_schema: bool,
    pub trace_log: bool,
    pub reasoning_config: crate::config::schema::ReasoningConfig,
    /// Code execution tool config.
    pub code_execution: crate::config::schema::CodeExecutionConfig,
    /// Python kernel tool config (feature: python-kernel).
    #[allow(dead_code)]
    pub python_kernel: crate::config::schema::PythonKernelConfig,
    /// Interval in seconds between tool-heartbeat progress ticks (default: 2).
    pub tool_heartbeat_secs: u64,
    /// Timeout in seconds for a single health-check HTTP request (default: 2).
    pub health_check_timeout_secs: u64,
    /// Adaptive token budget tuning (formerly hardcoded constants in agent_loop.rs).
    pub adaptive_tokens: AdaptiveTokenConfig,
}

impl SwappableCore {
    /// Typed runtime descriptor for this core.
    pub fn mode(&self) -> &RuntimeMode {
        &self.mode
    }
}

/// Current trio routing state — transitions logged once, not per-check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TrioState {
    /// Trio routing fully operational.
    Active = 0,
    /// Trio degraded — some components unhealthy, falling back.
    Degraded = 1,
    /// Trio disabled — running as standalone single model.
    Standalone = 2,
}

/// How the final post-policy tool catalog is presented to the main model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ToolPresentationMode {
    Native,
    Textual,
    Trio,
    ForcedText,
}

#[derive(Clone, Debug)]
struct FrozenToolCatalog {
    mode: ToolPresentationMode,
    definitions: Vec<serde_json::Value>,
    generation: u64,
}

#[derive(Clone, Copy)]
enum PromptResetScope {
    LogicalSession,
    PromptRewrite,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct HiggsSessionLease {
    pub(crate) session_id: u64,
    pub(crate) ttl_seconds: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ExpansionCheckpoint {
    pub(crate) old_higgs_session_id: u64,
    pub(crate) summary_node_id: usize,
    pub(crate) replaced_span: Vec<Value>,
    pub(crate) frozen_tool_hash: u64,
    pub(crate) model: String,
    pub(crate) presentation_mode: ToolPresentationMode,
    pub(crate) catalog_generation: u64,
    pub(crate) expires_at_ms: u64,
    pub(crate) lease_confirmed: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ExpansionCheckpointContext {
    model: String,
    presentation_mode: ToolPresentationMode,
    catalog_generation: u64,
    frozen_tool_hash: u64,
    expires_at_ms: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum SessionRetirement {
    Drop,
    LeaseForExpansion {
        summary_node_id: usize,
        replaced_span: Vec<Value>,
        checkpoint_context: ExpansionCheckpointContext,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum HiggsSessionReusePolicy {
    BestEffort,
    RequireContinuation,
}

impl HiggsSessionReusePolicy {
    pub(crate) fn as_wire(self) -> &'static str {
        match self {
            Self::BestEffort => "best_effort",
            Self::RequireContinuation => "require_continuation",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct HiggsSessionControl {
    pub(crate) active_id: u64,
    pub(crate) drop_ids: Vec<u64>,
    pub(crate) session_lease: Option<HiggsSessionLease>,
    pub(crate) reuse_policy: HiggsSessionReusePolicy,
    pub(crate) max_prompt_tokens: u32,
}

#[derive(Default)]
struct HiggsSessionState {
    epoch: u64,
    active_id: Option<u64>,
    pending_drop_ids: Vec<u64>,
    pending_lease: Option<HiggsSessionLease>,
    claimed_lease_id: Option<u64>,
    expansion_checkpoint: Option<ExpansionCheckpoint>,
    in_flight_active_ids: std::collections::HashMap<u64, usize>,
}

pub(crate) struct HiggsSessionRequestReservation {
    counters: Arc<RuntimeCounters>,
    session_key: String,
    control: HiggsSessionControl,
    lease_finalized: bool,
}

impl HiggsSessionRequestReservation {
    #[cfg(test)]
    pub(crate) fn active_id(&self) -> u64 {
        self.control.active_id
    }

    pub(crate) fn drop_ids(&self) -> &[u64] {
        &self.control.drop_ids
    }

    pub(crate) fn control(&self) -> &HiggsSessionControl {
        &self.control
    }

    pub(crate) fn resolve_lease(&mut self, lease_active: Option<i64>) {
        let Some(lease) = self.control.session_lease else {
            return;
        };
        if self.lease_finalized {
            return;
        }
        self.counters.resolve_higgs_session_lease(
            &self.session_key,
            lease.session_id,
            lease_active,
        );
        self.lease_finalized = true;
    }
}

impl Drop for HiggsSessionRequestReservation {
    fn drop(&mut self) {
        self.resolve_lease(None);
        self.counters.release_higgs_session_request(
            &self.session_key,
            self.control.active_id,
            self.control.session_lease.map(|lease| lease.session_id),
        );
    }
}

/// Observability counters for trio routing, populated by router.rs.
pub struct TrioMetrics {
    pub router_preflight_fired: AtomicBool,
    pub router_action: parking_lot::Mutex<Option<String>>,
    pub specialist_dispatched: AtomicBool,
    pub tool_dispatched: parking_lot::Mutex<Option<String>>,
}

impl Default for TrioMetrics {
    fn default() -> Self {
        Self {
            router_preflight_fired: AtomicBool::new(false),
            router_action: parking_lot::Mutex::new(None),
            specialist_dispatched: AtomicBool::new(false),
            tool_dispatched: parking_lot::Mutex::new(None),
        }
    }
}

/// Atomic counters that survive core swaps — never behind `RwLock`.
///
/// These counters persist across `/local` and `/model` hot-swaps because
/// they live outside the swappable core. Previously they were inside
/// `SharedCore` and silently reset to zero on every swap.
pub struct RuntimeCounters {
    pub learning_turn_counter: AtomicU64,
    pub last_context_used: AtomicU64,
    pub last_context_max: AtomicU64,
    pub last_message_count: AtomicU64,
    pub last_working_memory_tokens: AtomicU64,
    pub last_tools_called: parking_lot::Mutex<Vec<String>>,
    /// Tracks whether the delegation provider is alive. Set to `false` when
    /// the delegation LLM returns a hard error or times out, causing subsequent
    /// calls to fall through to inline execution. Reset to `true` on core
    /// rebuild (`rebuild_core`) and `/restart` command.
    pub delegation_healthy: AtomicBool,
    /// Counts tool calls since delegation was marked unhealthy. Used to
    /// periodically re-probe: every 10 inline calls, try delegation once
    /// more in case the server recovered.
    pub delegation_retry_counter: AtomicU64,
    /// Extended thinking budget in tokens. 0 = disabled, >0 = enabled with that budget.
    /// Toggled by `/think` or `/t`. `/think 16000` sets a specific budget.
    pub thinking_budget: AtomicU32,
    /// Remaining turns with boosted max_tokens (set by `/long`). Counts down to 0.
    pub long_mode_turns: AtomicU32,
    /// Last actual prompt tokens from LLM provider (for telemetry).
    pub last_actual_prompt_tokens: AtomicU64,
    /// Last actual completion tokens from LLM provider (for telemetry).
    pub last_actual_completion_tokens: AtomicU64,
    /// Last estimated prompt tokens (our estimate, for comparison).
    pub last_estimated_prompt_tokens: AtomicU64,
    /// Cumulative provider cache accounting keyed by durable logical session
    /// id. Retained Higgs session rotations must not split these totals.
    cache_metrics: parking_lot::Mutex<
        std::collections::HashMap<String, crate::agent::metrics::SessionCacheMetrics>,
    >,
    /// When true, ThinkingDelta tokens are not sent to delta_tx for visual
    /// rendering. Toggled by `/nothink` and config-level no-think mode.
    pub suppress_thinking_display: AtomicBool,
    /// When true, voice/TTS paths should not speak ThinkingDelta tokens.
    /// Auto-set while voice mode is active.
    pub suppress_thinking_in_tts: AtomicBool,
    /// Set to true while an LLM call is in flight. The health watchdog reads
    /// this to skip health checks during inference (avoiding false "unhealthy"
    /// restarts when the server is busy processing a large prompt).
    /// Wrapped in Arc so the watchdog can hold a cheap clone without needing
    /// the full RuntimeCounters.
    pub inference_active: Arc<AtomicBool>,
    /// Timestamp (epoch ms) when the most recent inference finished.
    pub last_inference_finished_ms: AtomicU64,
    /// Trio routing observability.
    pub trio_metrics: TrioMetrics,
    /// Circuit breaker for trio routing providers.
    pub trio_circuit_breaker: parking_lot::Mutex<CircuitBreaker>,
    /// Current trio routing state for observability.
    pub trio_state: AtomicU8,
    /// Per-domain ring buffer memory for specialist multi-turn context.
    pub specialist_memory: parking_lot::Mutex<crate::agent::router::SpecialistMemory>,
    /// Per-session prompt fingerprints for the prefix-divergence diagnostic
    /// (~8 bytes per message per session). See `agent::prompt_fingerprint`.
    pub prompt_fingerprints: parking_lot::Mutex<
        std::collections::HashMap<String, crate::agent::prompt_fingerprint::PromptFingerprint>,
    >,
    /// Per-session hash of the rendered `messages[0]` (the system prompt).
    ///
    /// Six places mutate the prompt head or insert ahead of the tail, each
    /// previously guarded only by a doc comment ("callers must pass the SAME
    /// note on every turn"). Chat templates render the head first, so a single
    /// changed byte there re-prefills the entire context. This turns those six
    /// comment-contracts into one checked invariant — see
    /// `agent::prefix_guard::assert_stable_head`.
    pub prompt_head_hashes: parking_lot::Mutex<std::collections::HashMap<String, u64>>,
    /// Per-session hash of the tool-definition array sent to the provider.
    /// The message fingerprint deliberately excludes tool schemas, so this
    /// catches the case where messages are append-only but the rendered token
    /// stream still diverges because the tool block (rendered at the prompt
    /// head by chat templates) changed — busting the prefix cache invisibly.
    pub prompt_tool_hashes: parking_lot::Mutex<std::collections::HashMap<String, u64>>,
    /// Final post-policy tool arrays, frozen per session and presentation mode.
    /// Execution-time registry availability remains authoritative.
    session_tool_catalogs: parking_lot::Mutex<std::collections::HashMap<String, FrozenToolCatalog>>,
    /// Per-session prefix-cache watermark: the number of leading messages
    /// already sent (hence warm on the inference server). Mid-turn cleanup is
    /// frozen below this index so the rendered prompt stays an append-only
    /// extension of the last send. Re-anchored on every send. See
    /// `agent::prefix_guard`.
    pub prompt_cache_watermark: parking_lot::Mutex<std::collections::HashMap<String, usize>>,
    /// Serializes prompt-cache identity changes with fingerprint/watermark
    /// capture, publication, restoration, and invalidation. Lock order starts
    /// here, then catalog → Higgs state → cache maps; no async work is allowed
    /// while held.
    prompt_cache_transition: parking_lot::Mutex<()>,
    /// Retained-session transitions are one transaction: epoch, active id,
    /// expansion lease/checkpoint, and pending drops must never be observed in
    /// partially updated combinations by concurrent requests or resets.
    higgs_sessions: parking_lot::Mutex<std::collections::HashMap<String, HiggsSessionState>>,
    /// Per-session local artifact intent for short follow-up edit turns
    /// ("make it red", "also add a score") after an explicit local artifact
    /// request. Bounded by turn count, not persisted.
    local_artifact_intent:
        parking_lot::Mutex<std::collections::HashMap<String, LocalArtifactIntentState>>,
    /// Number of installed compactions this session (see `record_compaction`).
    pub lcm_compaction_count: AtomicU64,
    /// Cumulative estimated tokens of compacted prefixes before compaction.
    pub lcm_tokens_before: AtomicU64,
    /// Cumulative estimated tokens of the same prefixes after compaction.
    pub lcm_tokens_after: AtomicU64,
    /// Epoch ms of the most recently installed compaction (0 = never).
    pub lcm_last_compaction_ms: AtomicU64,
    /// Responses that matched the phantom phrase list with zero tool calls.
    /// Observe-only: the response is still delivered (annotated). Tracks the
    /// detector's false-positive pressure without letting it discard work.
    pub phantom_claims_observed: AtomicU64,
    /// Prompt-prefix divergences the loop did NOT sanction — a message whose
    /// rendered bytes changed across turns. Always a full server re-prefill.
    pub cache_diverged: AtomicU64,
    /// Prompt-prefix resets the loop DID sanction (trim, compaction, history
    /// reload). Also a full re-prefill — counted so a "sanctioned" reset can
    /// never again be silently free. See `agent::prompt_fingerprint`.
    pub cache_sanctioned_resets: AtomicU64,
    /// Why this session's prompt fingerprint was last cleared, pending
    /// attribution at the next provider call.
    ///
    /// Clearing the fingerprint makes the next comparison report `First`,
    /// which reads identically to a genuine cold start — so a sanctioned
    /// rewrite used to cost a full re-prefill and leave no trace at all. In
    /// session 20260810_081050_8306f8 that hid 124.54s of prefill. Recording
    /// the reason here lets the next call log the reset instead of silently
    /// treating it as turn one.
    pending_cache_reset: parking_lot::Mutex<std::collections::HashMap<String, &'static str>>,
}

impl RuntimeCounters {
    pub fn new_with_config(max_context_tokens: usize, cb_config: &CircuitBreakerConfig) -> Self {
        Self {
            learning_turn_counter: AtomicU64::new(0),
            last_context_used: AtomicU64::new(0),
            last_context_max: AtomicU64::new(max_context_tokens as u64),
            last_message_count: AtomicU64::new(0),
            last_working_memory_tokens: AtomicU64::new(0),
            last_tools_called: parking_lot::Mutex::new(Vec::new()),
            delegation_healthy: AtomicBool::new(true),
            delegation_retry_counter: AtomicU64::new(0),
            thinking_budget: AtomicU32::new(0),
            long_mode_turns: AtomicU32::new(0),
            last_actual_prompt_tokens: AtomicU64::new(0),
            last_actual_completion_tokens: AtomicU64::new(0),
            last_estimated_prompt_tokens: AtomicU64::new(0),
            cache_metrics: parking_lot::Mutex::new(std::collections::HashMap::new()),
            suppress_thinking_display: AtomicBool::new(false),
            suppress_thinking_in_tts: AtomicBool::new(false),
            inference_active: Arc::new(AtomicBool::new(false)),
            last_inference_finished_ms: AtomicU64::new(0),
            trio_metrics: TrioMetrics::default(),
            trio_circuit_breaker: parking_lot::Mutex::new(CircuitBreaker::new(cb_config)),
            trio_state: AtomicU8::new(TrioState::Standalone as u8),
            specialist_memory: parking_lot::Mutex::new(
                crate::agent::router::SpecialistMemory::default(),
            ),
            prompt_fingerprints: parking_lot::Mutex::new(std::collections::HashMap::new()),
            prompt_head_hashes: parking_lot::Mutex::new(std::collections::HashMap::new()),
            prompt_tool_hashes: parking_lot::Mutex::new(std::collections::HashMap::new()),
            session_tool_catalogs: parking_lot::Mutex::new(std::collections::HashMap::new()),
            prompt_cache_watermark: parking_lot::Mutex::new(std::collections::HashMap::new()),
            prompt_cache_transition: parking_lot::Mutex::new(()),
            higgs_sessions: parking_lot::Mutex::new(std::collections::HashMap::new()),
            local_artifact_intent: parking_lot::Mutex::new(std::collections::HashMap::new()),
            lcm_compaction_count: AtomicU64::new(0),
            lcm_tokens_before: AtomicU64::new(0),
            lcm_tokens_after: AtomicU64::new(0),
            lcm_last_compaction_ms: AtomicU64::new(0),
            phantom_claims_observed: AtomicU64::new(0),
            cache_diverged: AtomicU64::new(0),
            cache_sanctioned_resets: AtomicU64::new(0),
            pending_cache_reset: parking_lot::Mutex::new(std::collections::HashMap::new()),
        }
    }

    /// Record why this session's prompt prefix was invalidated. Consumed by
    /// the next provider call via [`Self::take_cache_reset`].
    pub fn note_cache_reset(&self, session_key: &str, reason: &'static str) {
        self.pending_cache_reset
            .lock()
            .insert(session_key.to_string(), reason);
    }

    /// Take the pending reset reason, if the prefix was deliberately dropped
    /// since the last provider call. `None` means a genuine cold start.
    pub fn take_cache_reset(&self, session_key: &str) -> Option<&'static str> {
        self.pending_cache_reset.lock().remove(session_key)
    }

    pub fn record_cache_metrics(
        &self,
        logical_session: &str,
        prompt_tokens: u64,
        cache_read_tokens: Option<u64>,
        cache_creation_tokens: Option<u64>,
    ) {
        self.cache_metrics
            .lock()
            .entry(logical_session.to_string())
            .or_default()
            .record(prompt_tokens, cache_read_tokens, cache_creation_tokens);
    }

    pub fn session_cache_metrics(
        &self,
        logical_session: &str,
    ) -> crate::agent::metrics::SessionCacheMetrics {
        self.cache_metrics
            .lock()
            .get(logical_session)
            .copied()
            .unwrap_or_default()
    }

    pub(crate) fn frozen_tool_definitions(
        &self,
        session_key: &str,
        mode: ToolPresentationMode,
    ) -> Option<Vec<serde_json::Value>> {
        self.session_tool_catalogs
            .lock()
            .get(session_key)
            .filter(|catalog| catalog.mode == mode)
            .map(|catalog| catalog.definitions.clone())
    }

    pub(crate) fn tool_presentation_mode_changed(
        &self,
        session_key: &str,
        mode: ToolPresentationMode,
    ) -> bool {
        self.session_tool_catalogs
            .lock()
            .get(session_key)
            .is_some_and(|catalog| catalog.mode != mode)
    }

    pub(crate) fn tool_presentation_mode(&self, session_key: &str) -> Option<ToolPresentationMode> {
        self.session_tool_catalogs
            .lock()
            .get(session_key)
            .map(|catalog| catalog.mode)
    }

    pub(crate) fn install_tool_catalog(
        &self,
        session_key: &str,
        mode: ToolPresentationMode,
        definitions: Vec<serde_json::Value>,
    ) {
        let _transition = self.prompt_cache_transition.lock();
        let mut catalogs = self.session_tool_catalogs.lock();
        let generation = catalogs
            .get(session_key)
            .map_or(1, |catalog| catalog.generation.saturating_add(1));
        catalogs.insert(
            session_key.to_string(),
            FrozenToolCatalog {
                mode,
                definitions,
                generation,
            },
        );
        let mut sessions = self.higgs_sessions.lock();
        let Some(state) = sessions.get_mut(session_key) else {
            return;
        };
        if let Some(checkpoint) = state.expansion_checkpoint.take() {
            Self::queue_higgs_session_drop(state, checkpoint.old_higgs_session_id);
        }
        if let Some(lease) = state.pending_lease.take() {
            Self::queue_higgs_session_drop(state, lease.session_id);
        }
    }

    pub(crate) fn expansion_checkpoint_context(
        &self,
        session_key: &str,
        model: &str,
        frozen_tool_hash: u64,
        expires_at_ms: u64,
    ) -> Option<ExpansionCheckpointContext> {
        self.session_tool_catalogs
            .lock()
            .get(session_key)
            .map(|catalog| ExpansionCheckpointContext {
                model: model.to_string(),
                presentation_mode: catalog.mode,
                catalog_generation: catalog.generation,
                frozen_tool_hash,
                expires_at_ms,
            })
    }

    /// Identity of the physical compacted prompt-cache route. Cache snapshots
    /// captured for a retained turn may be restored only while all three
    /// components still match; reset, rotation, and catalog replacement each
    /// invalidate at least one component.
    pub(crate) fn prompt_cache_route_identity(
        &self,
        session_key: &str,
    ) -> (Option<u64>, u64, Option<u64>) {
        let catalogs = self.session_tool_catalogs.lock();
        let catalog_generation = catalogs.get(session_key).map(|catalog| catalog.generation);
        let sessions = self.higgs_sessions.lock();
        let state = sessions.get(session_key);
        (
            state.and_then(|state| state.active_id),
            state.map_or(0, |state| state.epoch),
            catalog_generation,
        )
    }

    /// Return an exact-reconstruction checkpoint only when the lease and every
    /// prompt-identity component still match the request being planned.
    /// Selection is read-only; request reservation remains the invalidation
    /// boundary that queues stale retained IDs for deletion.
    pub(crate) fn confirmed_expansion_checkpoint(
        &self,
        session_key: &str,
        model: &str,
        frozen_tool_hash: u64,
        now_ms: u64,
    ) -> Option<ExpansionCheckpoint> {
        self.confirmed_expansion_checkpoint_inner(
            session_key,
            model,
            frozen_tool_hash,
            now_ms,
            || {},
        )
    }

    fn confirmed_expansion_checkpoint_inner<F>(
        &self,
        session_key: &str,
        model: &str,
        frozen_tool_hash: u64,
        now_ms: u64,
        observe_catalog_snapshot: F,
    ) -> Option<ExpansionCheckpoint>
    where
        F: FnOnce(),
    {
        let catalogs = self.session_tool_catalogs.lock();
        let catalog_identity = catalogs
            .get(session_key)
            .map(|catalog| (catalog.mode, catalog.generation));
        observe_catalog_snapshot();
        let checkpoint = self
            .higgs_sessions
            .lock()
            .get(session_key)
            .and_then(|state| state.expansion_checkpoint.as_ref())
            .filter(|checkpoint| {
                checkpoint.lease_confirmed
                    && checkpoint.expires_at_ms > now_ms
                    && checkpoint.model == model
                    && checkpoint.frozen_tool_hash == frozen_tool_hash
                    && catalog_identity
                        == Some((checkpoint.presentation_mode, checkpoint.catalog_generation))
            })
            .cloned();
        drop(catalogs);
        checkpoint
    }

    #[cfg(test)]
    fn confirmed_expansion_checkpoint_observed<F>(
        &self,
        session_key: &str,
        model: &str,
        frozen_tool_hash: u64,
        now_ms: u64,
        observe_catalog_snapshot: F,
    ) -> Option<ExpansionCheckpoint>
    where
        F: FnOnce(),
    {
        self.confirmed_expansion_checkpoint_inner(
            session_key,
            model,
            frozen_tool_hash,
            now_ms,
            observe_catalog_snapshot,
        )
    }

    pub(crate) fn clear_tool_catalog(&self, session_key: &str) -> bool {
        let _transition = self.prompt_cache_transition.lock();
        self.session_tool_catalogs
            .lock()
            .remove(session_key)
            .is_some()
    }

    fn rotate_prompt_session(&self, session_key: &str, scope: PromptResetScope) -> u64 {
        if matches!(scope, PromptResetScope::LogicalSession) {
            self.clear_local_artifact_intent(session_key);
            self.clear_tool_catalog(session_key);
            self.note_cache_reset(session_key, "session_reset");
        }
        self.retire_higgs_session(session_key, SessionRetirement::Drop)
    }

    /// Reset all prompt-cache bookkeeping for a session and advance its prompt epoch.
    ///
    /// The epoch is rendered into the next prompt as a tiny stable marker. This
    /// forces local resident servers to treat post-clear/post-switch prompts as
    /// a fresh prefix even when the user starts with identical text like `hi`.
    pub fn reset_session_prompt_state(&self, session_key: &str) -> u64 {
        self.rotate_prompt_session(session_key, PromptResetScope::LogicalSession)
    }

    #[cfg(test)]
    pub fn session_prompt_epoch(&self, session_key: &str) -> u64 {
        self.higgs_sessions
            .lock()
            .get(session_key)
            .map(|state| state.epoch)
            .unwrap_or(0)
    }

    pub(crate) fn lock_prompt_cache_transition(&self) -> parking_lot::MutexGuard<'_, ()> {
        self.prompt_cache_transition.lock()
    }

    /// Clear only the local prompt-cache bookkeeping (fingerprint + watermark)
    /// for a session, without touching the retained higgs session id or epoch.
    /// Used when a rewrite happens on a backend that does not support retained
    /// sessions, or as the local-clear half of [`invalidate_prompt_cache`].
    pub fn clear_local_prompt_cache(&self, session_key: &str) -> bool {
        let _transition = self.prompt_cache_transition.lock();
        let had_fingerprint = self
            .prompt_fingerprints
            .lock()
            .remove(session_key)
            .is_some();
        let had_watermark = self
            .prompt_cache_watermark
            .lock()
            .remove(session_key)
            .is_some();
        had_fingerprint || had_watermark
    }

    /// Consolidated cache invalidation for a sanctioned prompt rewrite (trim,
    /// compaction). When `rotate` is set (the provider supports the higgs
    /// retained-session protocol), rotates the retained session — epoch bump +
    /// queued drop + full bookkeeping clear — so the server cold-starts the
    /// rewritten prompt. Otherwise clears only the local fingerprint/watermark.
    /// Returns `true` iff the session was rotated.
    pub fn invalidate_prompt_cache(&self, session_key: &str, rotate: bool) -> bool {
        if rotate {
            self.rotate_prompt_session(session_key, PromptResetScope::PromptRewrite);
            true
        } else {
            self.clear_local_prompt_cache(session_key);
            false
        }
    }

    #[cfg(test)]
    pub fn record_higgs_session_id(&self, session_key: &str, session_id: u64) -> bool {
        let mut sessions = self.higgs_sessions.lock();
        let state = sessions.entry(session_key.to_string()).or_default();
        if state
            .expansion_checkpoint
            .as_ref()
            .is_some_and(|checkpoint| checkpoint.old_higgs_session_id == session_id)
        {
            return false;
        }
        state.active_id = Some(session_id);
        true
    }

    fn activate_higgs_session_state(
        state: &mut HiggsSessionState,
        durable_session_id: &str,
    ) -> u64 {
        let mut active_id = stable_higgs_session_id(durable_session_id, state.epoch);
        while state
            .expansion_checkpoint
            .as_ref()
            .is_some_and(|checkpoint| checkpoint.old_higgs_session_id == active_id)
        {
            state.epoch = state
                .epoch
                .checked_add(1)
                .expect("Higgs prompt epoch exhausted");
            active_id = stable_higgs_session_id(durable_session_id, state.epoch);
        }
        state.active_id = Some(active_id);
        active_id
    }

    pub(crate) fn reserve_higgs_session_request(
        self: &Arc<Self>,
        session_key: &str,
        durable_session_id: &str,
        model: &str,
        frozen_tool_hash: u64,
        max_prompt_tokens: u32,
        now_ms: u64,
    ) -> HiggsSessionRequestReservation {
        let _transition = self.prompt_cache_transition.lock();
        let catalogs = self.session_tool_catalogs.lock();
        let catalog_identity = catalogs
            .get(session_key)
            .map(|catalog| (catalog.mode, catalog.generation));
        let mut sessions = self.higgs_sessions.lock();
        let state = sessions.entry(session_key.to_string()).or_default();
        let checkpoint_is_valid = state
            .expansion_checkpoint
            .as_ref()
            .is_none_or(|checkpoint| {
                checkpoint.expires_at_ms > now_ms
                    && checkpoint.model == model
                    && checkpoint.frozen_tool_hash == frozen_tool_hash
                    && catalog_identity
                        == Some((checkpoint.presentation_mode, checkpoint.catalog_generation))
            });
        if !checkpoint_is_valid {
            if let Some(checkpoint) = state.expansion_checkpoint.take() {
                Self::queue_higgs_session_drop(state, checkpoint.old_higgs_session_id);
            }
            if let Some(lease) = state.pending_lease.take() {
                Self::queue_higgs_session_drop(state, lease.session_id);
            }
        }
        let prior_route_identity = (state.active_id, state.epoch);
        let active_id = Self::activate_higgs_session_state(state, durable_session_id);
        if prior_route_identity != (Some(active_id), state.epoch) {
            self.prompt_fingerprints.lock().remove(session_key);
            self.prompt_cache_watermark.lock().remove(session_key);
        }
        *state.in_flight_active_ids.entry(active_id).or_default() += 1;
        let session_lease = if state.claimed_lease_id.is_none() {
            state.pending_lease.take().inspect(|lease| {
                state.claimed_lease_id = Some(lease.session_id);
                *state
                    .in_flight_active_ids
                    .entry(lease.session_id)
                    .or_default() += 1;
            })
        } else {
            None
        };
        let drop_ids = state
            .pending_drop_ids
            .iter()
            .copied()
            .filter(|drop_id| !state.in_flight_active_ids.contains_key(drop_id))
            .collect();
        HiggsSessionRequestReservation {
            counters: Arc::clone(self),
            session_key: session_key.to_string(),
            control: HiggsSessionControl {
                active_id,
                drop_ids,
                session_lease,
                reuse_policy: HiggsSessionReusePolicy::BestEffort,
                max_prompt_tokens,
            },
            lease_finalized: false,
        }
    }

    /// Pin a confirmed retained checkpoint for one request without changing
    /// the compacted session's active ID. Catalog and checkpoint identity are
    /// validated under the established catalog→Higgs lock order.
    pub(crate) fn reserve_retained_expansion_request(
        self: &Arc<Self>,
        session_key: &str,
        checkpoint: &ExpansionCheckpoint,
        model: &str,
        frozen_tool_hash: u64,
        max_prompt_tokens: u32,
        now_ms: u64,
    ) -> Option<HiggsSessionRequestReservation> {
        let catalogs = self.session_tool_catalogs.lock();
        let catalog_identity = catalogs
            .get(session_key)
            .map(|catalog| (catalog.mode, catalog.generation));
        let mut sessions = self.higgs_sessions.lock();
        let state = sessions.get_mut(session_key)?;
        let current_matches = state.expansion_checkpoint.as_ref().is_some_and(|current| {
            current == checkpoint
                && current.lease_confirmed
                && current.expires_at_ms > now_ms
                && current.model == model
                && current.frozen_tool_hash == frozen_tool_hash
                && catalog_identity == Some((current.presentation_mode, current.catalog_generation))
        });
        if !current_matches {
            if state
                .expansion_checkpoint
                .as_ref()
                .is_some_and(|current| current == checkpoint)
            {
                state.expansion_checkpoint = None;
                Self::queue_higgs_session_drop(state, checkpoint.old_higgs_session_id);
            }
            return None;
        }

        let retained_id = checkpoint.old_higgs_session_id;
        *state.in_flight_active_ids.entry(retained_id).or_default() += 1;
        let drop_ids = state
            .pending_drop_ids
            .iter()
            .copied()
            .filter(|drop_id| !state.in_flight_active_ids.contains_key(drop_id))
            .collect();
        drop(catalogs);
        Some(HiggsSessionRequestReservation {
            counters: Arc::clone(self),
            session_key: session_key.to_string(),
            control: HiggsSessionControl {
                active_id: retained_id,
                drop_ids,
                session_lease: None,
                reuse_policy: HiggsSessionReusePolicy::RequireContinuation,
                max_prompt_tokens,
            },
            lease_finalized: false,
        })
    }

    /// Retire only the checkpoint selected by this turn. A newer replacement
    /// remains intact; the compacted active ID is never rotated or overwritten.
    pub(crate) fn discard_expansion_checkpoint(
        &self,
        session_key: &str,
        old_higgs_session_id: u64,
        summary_node_id: usize,
    ) -> bool {
        let mut sessions = self.higgs_sessions.lock();
        let Some(state) = sessions.get_mut(session_key) else {
            return false;
        };
        let matches = state
            .expansion_checkpoint
            .as_ref()
            .is_some_and(|checkpoint| {
                checkpoint.old_higgs_session_id == old_higgs_session_id
                    && checkpoint.summary_node_id == summary_node_id
            });
        if !matches {
            return false;
        }
        state.expansion_checkpoint = None;
        Self::queue_higgs_session_drop(state, old_higgs_session_id);
        true
    }

    pub(crate) fn resolve_higgs_session_lease(
        &self,
        session_key: &str,
        leased_session_id: u64,
        lease_active: Option<i64>,
    ) {
        let mut sessions = self.higgs_sessions.lock();
        let Some(state) = sessions.get_mut(session_key) else {
            return;
        };
        if state.claimed_lease_id != Some(leased_session_id) {
            return;
        }
        state.claimed_lease_id = None;
        let checkpoint_matches = state
            .expansion_checkpoint
            .as_ref()
            .is_some_and(|checkpoint| checkpoint.old_higgs_session_id == leased_session_id);
        if !checkpoint_matches {
            Self::queue_higgs_session_drop(state, leased_session_id);
            return;
        }
        if lease_active == Some(1) {
            if let Some(checkpoint) = state.expansion_checkpoint.as_mut() {
                checkpoint.lease_confirmed = true;
            }
        } else {
            state.expansion_checkpoint = None;
            Self::queue_higgs_session_drop(state, leased_session_id);
        }
    }

    fn release_higgs_session_request(
        &self,
        session_key: &str,
        active_id: u64,
        leased_session_id: Option<u64>,
    ) {
        let mut sessions = self.higgs_sessions.lock();
        let Some(state) = sessions.get_mut(session_key) else {
            return;
        };
        for session_id in [Some(active_id), leased_session_id].into_iter().flatten() {
            let Some(in_flight) = state.in_flight_active_ids.get_mut(&session_id) else {
                continue;
            };
            *in_flight -= 1;
            if *in_flight == 0 {
                state.in_flight_active_ids.remove(&session_id);
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn activate_higgs_session_id(
        &self,
        session_key: &str,
        durable_session_id: &str,
    ) -> u64 {
        let mut sessions = self.higgs_sessions.lock();
        let state = sessions.entry(session_key.to_string()).or_default();
        Self::activate_higgs_session_state(state, durable_session_id)
    }

    fn queue_higgs_session_drop(state: &mut HiggsSessionState, session_id: u64) {
        if !state.pending_drop_ids.contains(&session_id) {
            state.pending_drop_ids.push(session_id);
        }
    }

    pub(crate) fn retire_higgs_session(
        &self,
        session_key: &str,
        retirement: SessionRetirement,
    ) -> u64 {
        self.retire_higgs_session_inner(session_key, retirement, || {})
    }

    fn retire_higgs_session_inner<F>(
        &self,
        session_key: &str,
        retirement: SessionRetirement,
        observe_transaction: F,
    ) -> u64
    where
        F: FnOnce(),
    {
        let _transition = self.prompt_cache_transition.lock();
        let mut sessions = self.higgs_sessions.lock();
        self.prompt_fingerprints.lock().remove(session_key);
        self.prompt_tool_hashes.lock().remove(session_key);
        self.prompt_cache_watermark.lock().remove(session_key);
        let state = sessions.entry(session_key.to_string()).or_default();
        let prior_checkpoint = state.expansion_checkpoint.take();
        let prior_lease = state.pending_lease.take();
        if let Some(checkpoint) = prior_checkpoint {
            Self::queue_higgs_session_drop(state, checkpoint.old_higgs_session_id);
        }
        if let Some(lease) = prior_lease {
            Self::queue_higgs_session_drop(state, lease.session_id);
        }

        let active_id = state.active_id.take();
        let active_is_in_flight =
            active_id.is_some_and(|active_id| state.in_flight_active_ids.contains_key(&active_id));
        observe_transaction();
        match (retirement, active_id) {
            (SessionRetirement::Drop, Some(active_id)) => {
                Self::queue_higgs_session_drop(state, active_id);
            }
            (
                SessionRetirement::LeaseForExpansion {
                    summary_node_id,
                    replaced_span,
                    checkpoint_context,
                },
                Some(active_id),
            ) if !active_is_in_flight => {
                let lease = HiggsSessionLease {
                    session_id: active_id,
                    ttl_seconds: 300,
                };
                state.pending_lease = Some(lease);
                state.expansion_checkpoint = Some(ExpansionCheckpoint {
                    old_higgs_session_id: active_id,
                    summary_node_id,
                    replaced_span,
                    frozen_tool_hash: checkpoint_context.frozen_tool_hash,
                    model: checkpoint_context.model,
                    presentation_mode: checkpoint_context.presentation_mode,
                    catalog_generation: checkpoint_context.catalog_generation,
                    expires_at_ms: checkpoint_context.expires_at_ms,
                    lease_confirmed: false,
                });
            }
            (SessionRetirement::LeaseForExpansion { .. }, Some(active_id)) => {
                // A selected request may already carry this ID on its immutable
                // wire payload. Retiring it as an expansion checkpoint before
                // that request completes would make one ID both active and
                // retained. Fall back to a one-shot drop on the next request.
                Self::queue_higgs_session_drop(state, active_id);
            }
            (SessionRetirement::Drop | SessionRetirement::LeaseForExpansion { .. }, None) => {}
        }

        state.epoch = state.epoch.saturating_add(1);
        state.epoch
    }

    #[cfg(test)]
    fn retire_higgs_session_observed<F>(
        &self,
        session_key: &str,
        retirement: SessionRetirement,
        observe_transaction: F,
    ) -> u64
    where
        F: FnOnce(),
    {
        self.retire_higgs_session_inner(session_key, retirement, observe_transaction)
    }

    #[cfg(test)]
    pub(crate) fn expansion_checkpoint(&self, session_key: &str) -> Option<ExpansionCheckpoint> {
        self.higgs_sessions
            .lock()
            .get(session_key)
            .and_then(|state| state.expansion_checkpoint.clone())
    }

    #[cfg(test)]
    pub(crate) fn pending_higgs_session_lease(
        &self,
        session_key: &str,
    ) -> Option<HiggsSessionLease> {
        self.higgs_sessions
            .lock()
            .get(session_key)
            .and_then(|state| state.pending_lease)
    }

    #[cfg(test)]
    pub(crate) fn active_higgs_session_id(&self, session_key: &str) -> Option<u64> {
        self.higgs_sessions
            .lock()
            .get(session_key)
            .and_then(|state| state.active_id)
    }

    #[cfg(test)]
    pub fn pending_higgs_session_drop_ids(&self, session_key: &str) -> Vec<u64> {
        self.higgs_sessions
            .lock()
            .get(session_key)
            .map(|state| state.pending_drop_ids.clone())
            .unwrap_or_default()
    }

    pub fn clear_pending_higgs_session_drop_id(&self, session_key: &str, drop_id: u64) -> bool {
        let mut sessions = self.higgs_sessions.lock();
        let Some(state) = sessions.get_mut(session_key) else {
            return false;
        };
        let Some(pos) = state
            .pending_drop_ids
            .iter()
            .position(|candidate| *candidate == drop_id)
        else {
            return false;
        };
        state.pending_drop_ids.remove(pos);
        true
    }

    pub fn clear_pending_higgs_session_drop_ids(
        &self,
        session_key: &str,
        drop_ids_to_clear: &[u64],
    ) -> usize {
        drop_ids_to_clear
            .iter()
            .filter(|drop_id| self.clear_pending_higgs_session_drop_id(session_key, **drop_id))
            .count()
    }

    pub(crate) fn record_local_artifact_intent(
        &self,
        session_key: &str,
        turn_count: u64,
        is_rich: bool,
    ) {
        self.local_artifact_intent.lock().insert(
            session_key.to_string(),
            LocalArtifactIntentState {
                is_rich,
                expires_after_turn: turn_count.saturating_add(LOCAL_ARTIFACT_INTENT_TTL_TURNS),
            },
        );
    }

    pub(crate) fn local_artifact_intent_is_rich(
        &self,
        session_key: &str,
        turn_count: u64,
    ) -> Option<bool> {
        let mut intents = self.local_artifact_intent.lock();
        let state = intents.get(session_key).copied()?;
        if turn_count > state.expires_after_turn {
            intents.remove(session_key);
            return None;
        }
        Some(state.is_rich)
    }

    pub(crate) fn clear_local_artifact_intent(&self, session_key: &str) -> bool {
        self.local_artifact_intent
            .lock()
            .remove(session_key)
            .is_some()
    }
}

pub(crate) fn stable_higgs_session_id(session_id: &str, epoch: u64) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in session_id.bytes().chain(epoch.to_le_bytes()) {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

impl RuntimeCounters {
    pub(crate) fn now_epoch_ms() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0)
    }

    /// Record an installed compaction for `/lcm stats`: bump the count,
    /// accumulate before/after token estimates, and stamp the time.
    pub fn record_compaction(&self, tokens_before: u64, tokens_after: u64) {
        self.lcm_compaction_count.fetch_add(1, Ordering::Relaxed);
        self.lcm_tokens_before
            .fetch_add(tokens_before, Ordering::Relaxed);
        self.lcm_tokens_after
            .fetch_add(tokens_after, Ordering::Relaxed);
        self.lcm_last_compaction_ms
            .store(Self::now_epoch_ms(), Ordering::Relaxed);
    }

    pub fn mark_inference_started(&self) {
        self.inference_active.store(true, Ordering::Relaxed);
    }

    pub fn mark_inference_finished(&self) {
        self.inference_active.store(false, Ordering::Relaxed);
        self.last_inference_finished_ms
            .store(Self::now_epoch_ms(), Ordering::Relaxed);
    }

    /// Update trio state, logging only on transitions.
    pub fn set_trio_state(&self, new_state: TrioState) {
        let old = self
            .trio_state
            .swap(new_state as u8, std::sync::atomic::Ordering::Relaxed);
        if old != new_state as u8 {
            match new_state {
                TrioState::Active => tracing::info!("trio_state_transition: -> Active"),
                TrioState::Degraded => tracing::warn!("trio_state_transition: -> Degraded"),
                TrioState::Standalone => tracing::warn!("trio_state_transition: -> Standalone"),
            }
        }
    }

    pub fn get_trio_state(&self) -> TrioState {
        match self.trio_state.load(std::sync::atomic::Ordering::Relaxed) {
            0 => TrioState::Active,
            1 => TrioState::Degraded,
            _ => TrioState::Standalone,
        }
    }
}

/// Combined handle: cheap to clone (two pointer bumps).
///
/// `core` is swapped on `/local` and `/model`. `counters` persists forever.
#[derive(Clone)]
pub struct AgentHandle {
    core: Arc<parking_lot::RwLock<Arc<SwappableCore>>>,
    pub counters: Arc<RuntimeCounters>,
}

impl AgentHandle {
    /// Create a new handle from a swappable core and runtime counters.
    pub fn new(core: SwappableCore, counters: Arc<RuntimeCounters>) -> Self {
        Self {
            core: Arc::new(parking_lot::RwLock::new(Arc::new(core))),
            counters,
        }
    }

    /// Snapshot the current swappable core (cheap Arc clone under brief read lock).
    pub fn swappable(&self) -> Arc<SwappableCore> {
        self.core.read().clone()
    }

    /// Replace the swappable core (write lock). Counters are untouched.
    pub fn swap_core(&self, new_core: SwappableCore) {
        *self.core.write() = Arc::new(new_core);
    }
}

// Backward-compatibility alias during migration.
pub type SharedCoreHandle = AgentHandle;

// ---------------------------------------------------------------------------
// SwappableCore construction
// ---------------------------------------------------------------------------

/// Named-field input for [`build_swappable_core`].
///
/// Replaces 18 positional parameters with a single struct so callers
/// are immune to parameter-ordering bugs.
pub struct SwappableCoreConfig {
    pub provider: Arc<dyn LLMProvider>,
    pub workspace: PathBuf,
    pub model: String,
    pub max_iterations: u32,
    pub max_continuations: u32,
    pub max_tokens: u32,
    pub temperature: f64,
    pub max_context_tokens: usize,
    pub brave_api_key: Option<String>,
    pub search_provider: String,
    pub searxng_url: String,
    /// Base URL of a local crw-server for web_fetch; empty = disabled.
    pub crw_url: String,
    pub search_max_results: u32,
    pub exec_timeout: u64,
    pub restrict_to_workspace: bool,
    pub memory_config: MemoryConfig,
    pub is_local: bool,
    pub lane: Lane,
    pub tool_delegation: ToolDelegationConfig,
    pub provenance: ProvenanceConfig,
    pub max_tool_result_chars: usize,
    pub delegation_provider: Option<Arc<dyn LLMProvider>>,
    pub specialist_provider: Option<Arc<dyn LLMProvider>>,
    pub trio_config: TrioConfig,
    pub model_capabilities_overrides: std::collections::HashMap<
        String,
        crate::agent::model_capabilities::ModelCapabilitiesOverride,
    >,
    pub reasoning_config: crate::config::schema::ReasoningConfig,
    /// Interval in seconds between tool-heartbeat progress ticks (default: 2).
    pub tool_heartbeat_secs: u64,
    /// Timeout in seconds for a single health-check HTTP request (default: 2).
    pub health_check_timeout_secs: u64,
    /// Adaptive token budget tuning (formerly hardcoded constants in agent_loop.rs).
    pub adaptive_tokens: AdaptiveTokenConfig,
    /// Optional override for the sessions SQLite DB path. Production passes
    /// `None` to use the default `~/.nanobot/sessions.db`; test harnesses pass
    /// `Some(workspace.join("sessions.db"))` so parallel tests don't contend
    /// on the user's real session DB.
    pub sessions_db_path: Option<PathBuf>,
    /// Code execution (Python RPC) tool settings.
    pub code_execution: crate::config::schema::CodeExecutionConfig,
    /// Python kernel tool (PyO3, feature: python-kernel).
    pub python_kernel: crate::config::schema::PythonKernelConfig,
}

/// Build a `SwappableCore` from the given config.
///
/// Called once at startup and again for every `/local` or `/model` toggle.
/// Resolves provider selection, memory config, tool delegation, and router setup.
#[allow(deprecated)] // reads lazy_skills for backward-compat wire-through
pub fn build_swappable_core(cfg: SwappableCoreConfig) -> SwappableCore {
    let SwappableCoreConfig {
        provider,
        workspace,
        model,
        max_iterations,
        max_continuations,
        max_tokens,
        temperature,
        max_context_tokens,
        brave_api_key,
        search_provider,
        searxng_url,
        crw_url,
        search_max_results,
        exec_timeout,
        restrict_to_workspace,
        memory_config,
        is_local,
        lane,
        tool_delegation,
        provenance,
        max_tool_result_chars,
        delegation_provider,
        specialist_provider,
        trio_config,
        model_capabilities_overrides,
        reasoning_config,
        tool_heartbeat_secs,
        health_check_timeout_secs,
        adaptive_tokens,
        sessions_db_path,
        code_execution,
        python_kernel,
    } = cfg;
    let model_capabilities =
        crate::agent::model_capabilities::lookup(&model, &model_capabilities_overrides);
    // Construct the typed runtime descriptor *once*, from the same inputs that
    // decide `is_local`. Parallel-rollout invariant (Wave 2): `is_local` and
    // `mode` must agree. Wave 3 migrates downstream readers; Wave 4 removes
    // the bool. See .planning/phases/09-runtime-mode-spine/09-CONTEXT.md.
    let mode = if is_local {
        RuntimeMode::from_caps(Some(Arc::new(model_capabilities.clone())))
    } else {
        RuntimeMode::from_caps(None)
    };
    debug_assert_eq!(
        matches!(mode, RuntimeMode::Local { .. }),
        is_local,
        "is_local and RuntimeMode must agree during parallel rollout"
    );
    let router_provider = delegation_provider.clone();
    // Branch 1 (Wave 2): context constructor selection is driven by RuntimeMode.
    let mut context = match mode {
        RuntimeMode::Local { .. } => ContextBuilder::new_lite(&workspace),
        RuntimeMode::Cloud => ContextBuilder::new(&workspace),
    };
    // Branch 2 (Wave 2): scale prompt budgets proportionally to the context window.
    // Local uses the lite clamps; cloud uses the full scaling curve.
    match mode {
        RuntimeMode::Local { .. } => context.set_lite_mode(max_context_tokens),
        RuntimeMode::Cloud => context.scale_budgets(max_context_tokens),
    }
    context.model_name = model.clone();
    // Keep prompt assembly behind the typed runtime descriptor, so local/cloud
    // behavior has one source of truth while the legacy bool is phased out.
    context.local_prompt_mode = mode.is_local();
    // Inject provenance verification rules when enabled.
    if provenance.enabled && provenance.system_prompt_rules {
        context.provenance_enabled = true;
    }
    // RLM lazy skills: skills loaded as summaries, fetched on demand.
    context.lazy_skills = memory_config.lazy_skills;
    // 3-tier skill disclosure: compact (default) | xml | eager.
    context.skill_disclosure = memory_config.skill_disclosure.clone();
    // Wire subagent profiles into the system prompt so the model knows
    // what agents exist and when to delegate instead of doing everything itself.
    let profiles = agent_profiles::load_profiles(&workspace);
    context.agent_profiles = agent_profiles::profiles_summary(&profiles);
    let db_path = sessions_db_path.unwrap_or_else(|| {
        dirs::home_dir()
            .unwrap_or_default()
            .join(".nanobot")
            .join("sessions.db")
    });
    let sessions = Arc::new(SessionDb::new(&db_path));

    // Branch 3 (Wave 2): memory-provider resolution is extracted into a named
    // helper dispatched via `match mode`. See `resolve_memory_provider` below.
    let (memory_provider, memory_model) = resolve_memory_provider(
        &mode,
        &memory_config,
        &model,
        &provider,
        specialist_provider.as_ref(),
    );

    // Branch 4 (Wave 2): response-reserve cap is derived from the runtime mode.
    // Cloud: passthrough of `max_tokens`. Local: clamp to 25% of the context
    // window so conversation + tool defs still fit.
    let effective_reserve = mode.reserve_cap(max_tokens as usize, max_context_tokens);
    let token_budget = TokenBudget::new(max_context_tokens, effective_reserve);
    let compactor = ContextCompactor::new(provider.clone(), model.clone(), max_context_tokens);
    debug!(
        model = %model,
        memory_model = %memory_model,
        max_context_tokens,
        "agent_core: main-model compactor initialized"
    );
    let working_memory = WorkingMemoryStore::new(sessions.clone());

    // Build tool runner provider if delegation is enabled.
    let (tool_runner_provider, tool_runner_model) = if tool_delegation.enabled {
        let is_auto_local = delegation_provider.is_some();
        let tr_provider: Arc<dyn LLMProvider> = if let Some(dp) = delegation_provider {
            dp // Auto-spawned local delegation server
        } else if let Some(ref tr_cfg) = tool_delegation.provider {
            let model_hint = if !tool_delegation.model.is_empty() {
                Some(tool_delegation.model.as_str())
            } else {
                None
            };
            let default_base = match mode {
                RuntimeMode::Local { .. } => provider.get_api_base(),
                RuntimeMode::Cloud => None,
            };
            crate::providers::factory::from_provider_config_for_model_with_default_base(
                tr_cfg,
                model_hint,
                default_base,
            )
        } else {
            provider.clone() // Fallback to main
        };
        // Pick the delegation model. When config is empty, fall back to the
        // delegation provider's own default (e.g. local server's model) rather
        // than the main model — the main model may be a cloud name like
        // "anthropic/claude-opus-4-5" that the local server doesn't understand.
        let tr_model = if !tool_delegation.model.is_empty() {
            tool_delegation.model.clone()
        } else if is_auto_local || model.starts_with("claude-max") || model.contains('/') {
            // Auto-spawned local delegation, or cloud model name — use provider default.
            tr_provider.get_default_model().to_string()
        } else {
            model.clone()
        };
        (Some(tr_provider), Some(tr_model))
    } else {
        (None, None)
    };

    let specialist_model = specialist_provider
        .as_ref()
        .map(|provider| provider.get_default_model().to_string());
    let router_model = router_provider
        .as_ref()
        .map(|provider| provider.get_default_model().to_string());

    SwappableCore {
        provider,
        workspace,
        model,
        max_iterations,
        max_continuations,
        max_tokens,
        temperature,
        context,
        sessions,
        token_budget,
        compactor,
        working_memory,
        // Scale working memory like other budgets. If the user left it at
        // the default (600), apply proportional scaling; otherwise respect their override.
        working_memory_budget: if memory_config.working_memory_budget == 600 {
            (max_context_tokens * 15 / 1000).clamp(300, 15_000) // 1.5%
        } else {
            memory_config.working_memory_budget
        },
        brave_api_key,
        search_provider,
        searxng_url,
        crw_url,
        search_max_results,
        exec_timeout,
        restrict_to_workspace,
        code_execution,
        python_kernel,
        memory_enabled: memory_config.enabled,
        memory_provider,
        memory_model,
        reflection_threshold: memory_config.reflection_threshold,
        mode,
        lane,
        tool_runner_provider,
        tool_runner_model,
        router_provider,
        router_model,
        router_no_think: trio_config.router_no_think,
        router_temperature: trio_config.router_temperature,
        router_top_p: trio_config.router_top_p,
        specialist_provider,
        specialist_model,
        specialist_temperature: trio_config.specialist_temperature,
        specialist_top_p: trio_config.specialist_top_p,
        tool_delegation_config: tool_delegation,
        provenance_config: provenance,
        max_tool_result_chars,
        session_complete_after_secs: memory_config.session_complete_after_secs,
        max_history_turns: memory_config.max_history_turns,
        model_capabilities,
        retention: crate::agent::retention::RetentionPolicy::from_config(
            &memory_config,
            &trio_config.anti_drift,
        ),
        specialist_output_schema: trio_config.specialist_output_schema,
        trace_log: trio_config.trace_log,
        reasoning_config,
        tool_heartbeat_secs,
        health_check_timeout_secs,
        adaptive_tokens,
    }
}

// ---------------------------------------------------------------------------
// Memory-provider resolution (Wave 2 extraction — G4 SPLIT)
// ---------------------------------------------------------------------------

/// Resolve the memory provider + model for a freshly-built `SwappableCore`.
///
/// Extracted from `build_swappable_core` in Wave 2 (09-02). Dispatch is
/// driven by [`RuntimeMode`] via exhaustive `match` (G5 BRANCH → TYPE).
///
/// Priority:
///  1. Explicit `memory.model` / `memory.provider`.
///  2. Cloud default: "haiku" (cheap, fast summarisation) when the main
///     provider is Anthropic native or OpenRouter; otherwise the main model.
///  3. Local default: trio specialist (if available) → main provider.
///
/// This selection is reflection-only. LCM is constructed directly from the
/// foreground provider/model and therefore cannot acquire a second context
/// ceiling or endpoint.
fn resolve_memory_provider(
    mode: &RuntimeMode,
    memory_config: &MemoryConfig,
    model: &str,
    provider: &Arc<dyn LLMProvider>,
    specialist_provider: Option<&Arc<dyn LLMProvider>>,
) -> (Arc<dyn LLMProvider>, String) {
    match mode {
        RuntimeMode::Local { .. } => {
            let mem_model = if !memory_config.model.is_empty() {
                memory_config.model.clone()
            } else if memory_config.provider.is_some() {
                model.to_string()
            } else if let Some(sp) = specialist_provider {
                sp.get_default_model().to_string()
            } else {
                model.to_string()
            };
            let mem_provider: Arc<dyn LLMProvider> =
                if let Some(ref mem_provider_cfg) = memory_config.provider {
                    crate::providers::factory::from_provider_config_for_model_with_default_base(
                        mem_provider_cfg,
                        Some(&mem_model),
                        provider.get_api_base(),
                    )
                } else if memory_config.model.is_empty() {
                    specialist_provider
                        .cloned()
                        .unwrap_or_else(|| provider.clone())
                } else {
                    provider.clone()
                };
            (mem_provider, mem_model)
        }
        RuntimeMode::Cloud => {
            let mem_model = if !memory_config.model.is_empty() {
                memory_config.model.clone()
            } else if provider.get_api_base().is_none()
                || provider
                    .get_api_base()
                    .map_or(false, |b| b.contains("openrouter"))
            {
                // Anthropic native or OpenRouter — use haiku for cheap memory ops.
                "haiku".to_string()
            } else {
                model.to_string()
            };
            let mem_provider: Arc<dyn LLMProvider> =
                if let Some(ref mem_provider_cfg) = memory_config.provider {
                    crate::providers::factory::from_provider_config_for_model(
                        mem_provider_cfg,
                        Some(&mem_model),
                    )
                } else {
                    provider.clone()
                };
            (mem_provider, mem_model)
        }
    }
}

// ---------------------------------------------------------------------------
// History limit scaling
// ---------------------------------------------------------------------------

/// History message limit before LCM's overflow backstop trims raw history.
///
/// The trim ceiling (limit · 150 tokens, enforced by filter_history Stage 6)
/// MUST sit comfortably ABOVE LCM's soft compaction threshold
/// (~tau_soft · (C − reserve − tool_defs) ≈ 0.375–0.5·C), otherwise trimming
/// caps conversation tokens just below the trigger and compaction never
/// fires. That is exactly what happened with the 0.4·C / 600-message limits:
/// at C=200K the ceiling was 600·150 = 90K tokens vs a ~97K soft limit.
///
/// LCM therefore allows ~0.7·C of history and raises the upper clamp so it
/// doesn't bite at large contexts (0.7·200000/150 ≈ 933 > 600). Long-session
/// hygiene is compaction's job here; trim is only the overflow backstop.
pub(crate) fn history_limit_lcm(max_context_tokens: usize) -> usize {
    let max_history_tokens = max_context_tokens * 7 / 10;
    let limit = max_history_tokens / 150;
    limit.clamp(6, 2000)
}

// ---------------------------------------------------------------------------
// Background compaction helpers
// ---------------------------------------------------------------------------

/// Pending compaction result ready to be swapped into the conversation.
pub(crate) struct PendingCompaction {
    pub result: crate::agent::compaction::CompactionResult,
    /// Exact live array that LCM compacted. Its leading system/developer prefix
    /// is non-durable and may change; the durable conversation bytes may not.
    pub snapshot: Vec<Value>,
    /// DAG node created by this exact compaction transaction.
    pub summary_node_id: usize,
}

fn prompt_prefix_len(messages: &[Value]) -> usize {
    messages
        .iter()
        .take_while(|message| {
            matches!(
                message.get("role").and_then(Value::as_str),
                Some("system" | "developer")
            )
        })
        .count()
}

impl PendingCompaction {
    pub(crate) fn watermark(&self) -> usize {
        self.snapshot.len()
    }

    fn live_conversation_watermark(&self, messages: &[Value]) -> Option<usize> {
        let snapshot_prefix_len = prompt_prefix_len(&self.snapshot);
        let live_prefix_len = prompt_prefix_len(messages);
        let snapshot_conversation = &self.snapshot[snapshot_prefix_len..];
        messages[live_prefix_len..]
            .starts_with(snapshot_conversation)
            .then_some(live_prefix_len + snapshot_conversation.len())
    }

    pub(crate) fn matches_snapshot_prefix(&self, messages: &[Value]) -> bool {
        self.live_conversation_watermark(messages).is_some()
    }

    /// Capture the single contiguous raw span replaced by the created summary.
    /// A merge over an older summary cannot prove full raw coverage and must
    /// use the normal compacted fallback rather than retained expansion.
    pub(crate) fn expansion_retirement(
        &self,
        checkpoint_context: ExpansionCheckpointContext,
    ) -> Option<SessionRetirement> {
        let prefix_len = self
            .snapshot
            .iter()
            .zip(&self.result.messages)
            .take_while(|(before, after)| before == after)
            .count();
        let suffix_len = self.snapshot[prefix_len..]
            .iter()
            .rev()
            .zip(self.result.messages[prefix_len..].iter().rev())
            .take_while(|(before, after)| before == after)
            .count();
        let snapshot_end = self.snapshot.len().saturating_sub(suffix_len);
        let result_end = self.result.messages.len().saturating_sub(suffix_len);
        let replaced_span = self.snapshot.get(prefix_len..snapshot_end)?;
        let replacement = self.result.messages.get(prefix_len..result_end)?;
        let replaces_with_one_summary = replacement.len() == 1
            && replacement[0]
                .get("_lcm_summary")
                .and_then(Value::as_bool)
                .unwrap_or(false);
        let has_full_raw_span = !replaced_span.is_empty()
            && replaced_span
                .iter()
                .all(|message| message.get("_db_id").is_some());
        (replaces_with_one_summary && has_full_raw_span).then(|| {
            SessionRetirement::LeaseForExpansion {
                summary_node_id: self.summary_node_id,
                replaced_span: replaced_span.to_vec(),
                checkpoint_context,
            }
        })
    }
}

/// Swap compacted messages into the live conversation, preserving
/// messages added after the compaction snapshot was taken.
pub(crate) fn apply_compaction_result(
    messages: &mut Vec<Value>,
    pending: PendingCompaction,
) -> bool {
    let Some(live_conversation_watermark) = pending.live_conversation_watermark(messages) else {
        return false;
    };

    let new_messages = messages[live_conversation_watermark..].to_vec();
    // The result carries the complete snapshot prompt prefix. Preserve the
    // current copy of that non-durable prefix, then append every compacted LCM
    // conversation entry. In particular, result[0] is not assumed to be a
    // system message: LCM's active context itself deliberately has no system.
    let live_prefix_len = prompt_prefix_len(messages);
    let result_prefix_len = prompt_prefix_len(&pending.result.messages);
    let mut swapped = Vec::with_capacity(
        live_prefix_len
            + pending
                .result
                .messages
                .len()
                .saturating_sub(result_prefix_len)
            + new_messages.len(),
    );
    swapped.extend_from_slice(&messages[..live_prefix_len]);
    swapped.extend_from_slice(&pending.result.messages[result_prefix_len..]);
    swapped.extend(new_messages);
    *messages = swapped;
    true
}

/// Append a suffix to the first (system) message's content.
///
/// Sole remaining caller: agent_shared.rs trio orchestration (Phase 5 scope).
/// All prepare_context.rs calls have been replaced with typed SectionEntry
/// values that flow through the PromptAssembler pipeline.
pub(crate) fn append_to_system_prompt(messages: &mut [Value], suffix: &str) {
    if let Some(sys) = messages
        .first()
        .and_then(|m| m["content"].as_str())
        .map(|s| s.to_string())
    {
        messages[0]["content"] = Value::String(format!("{}{}", sys, suffix));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::schema::CircuitBreakerConfig;

    fn checkpoint_context(frozen_tool_hash: u64, expires_at_ms: u64) -> ExpansionCheckpointContext {
        ExpansionCheckpointContext {
            model: "bonsai".to_string(),
            presentation_mode: ToolPresentationMode::Native,
            catalog_generation: 1,
            frozen_tool_hash,
            expires_at_ms,
        }
    }

    #[test]
    fn pending_compaction_recovers_one_exact_raw_replacement_span() {
        let system = serde_json::json!({"role": "system", "content": "system"});
        let first = serde_json::json!({"role": "user", "content": "first", "_db_id": 1});
        let second = serde_json::json!({"role": "assistant", "content": "second", "_db_id": 2});
        let tail = serde_json::json!({"role": "user", "content": "tail", "_db_id": 3});
        let summary = serde_json::json!({
            "role": "user",
            "content": "summary",
            "_lcm_summary": true
        });
        let pending = PendingCompaction {
            result: crate::agent::compaction::CompactionResult {
                messages: vec![system.clone(), summary, tail.clone()],
            },
            snapshot: vec![system, first.clone(), second.clone(), tail],
            summary_node_id: 44,
        };

        let Some(SessionRetirement::LeaseForExpansion {
            summary_node_id,
            replaced_span,
            checkpoint_context,
        }) = pending.expansion_retirement(checkpoint_context(0xabc, 900_000))
        else {
            panic!("one contiguous raw replacement must be recoverable");
        };
        assert_eq!(summary_node_id, 44);
        assert_eq!(replaced_span, vec![first, second]);
        assert_eq!(checkpoint_context.frozen_tool_hash, 0xabc);
        assert_eq!(checkpoint_context.expires_at_ms, 900_000);
    }

    #[test]
    fn merged_summary_without_full_raw_span_is_not_checkpointed() {
        let old_summary = serde_json::json!({
            "role": "user",
            "content": "older summary",
            "_lcm_summary": true
        });
        let raw = serde_json::json!({"role": "user", "content": "new raw", "_db_id": 9});
        let new_summary = serde_json::json!({
            "role": "user",
            "content": "merged summary",
            "_lcm_summary": true
        });
        let pending = PendingCompaction {
            result: crate::agent::compaction::CompactionResult {
                messages: vec![new_summary],
            },
            snapshot: vec![old_summary, raw],
            summary_node_id: 45,
        };

        assert_eq!(
            pending.expansion_retirement(checkpoint_context(7, 900_000)),
            None
        );
    }

    #[test]
    fn compaction_leases_old_id_and_reset_drops_every_retained_id() {
        let counters = RuntimeCounters::new_with_config(32_768, &CircuitBreakerConfig::default());
        let session = "cli:checkpoint";
        let old_id = stable_higgs_session_id(session, 0);
        assert!(counters.record_higgs_session_id(session, old_id));

        let next_epoch = counters.retire_higgs_session(
            session,
            SessionRetirement::LeaseForExpansion {
                summary_node_id: 44,
                replaced_span: vec![serde_json::json!({"role": "user", "content": "exact"})],
                checkpoint_context: checkpoint_context(7, 900_000),
            },
        );
        let checkpoint = counters
            .expansion_checkpoint(session)
            .expect("compaction must register a checkpoint");
        assert_eq!(checkpoint.old_higgs_session_id, old_id);
        assert_eq!(checkpoint.summary_node_id, 44);
        assert_eq!(checkpoint.frozen_tool_hash, 7);
        assert_eq!(checkpoint.expires_at_ms, 900_000);
        assert!(!checkpoint.lease_confirmed);
        assert_eq!(
            counters.pending_higgs_session_lease(session),
            Some(HiggsSessionLease {
                session_id: old_id,
                ttl_seconds: 300,
            })
        );
        assert!(counters.pending_higgs_session_drop_ids(session).is_empty());

        let fresh_id = stable_higgs_session_id(session, next_epoch);
        assert_ne!(fresh_id, old_id);
        assert!(counters.record_higgs_session_id(session, fresh_id));
        assert!(
            !counters.record_higgs_session_id(session, old_id),
            "the retained expansion id must never become the compacted active id"
        );

        counters.reset_session_prompt_state(session);
        assert_eq!(counters.expansion_checkpoint(session), None);
        assert_eq!(counters.pending_higgs_session_lease(session), None);
        let mut drops = counters.pending_higgs_session_drop_ids(session);
        drops.sort_unstable();
        let mut expected = vec![old_id, fresh_id];
        expected.sort_unstable();
        assert_eq!(drops, expected);
    }

    #[test]
    fn newer_compaction_replaces_and_drops_prior_checkpoint() {
        let counters = RuntimeCounters::new_with_config(32_768, &CircuitBreakerConfig::default());
        let session = "cli:replacement";
        assert!(counters.record_higgs_session_id(session, 10));
        counters.retire_higgs_session(
            session,
            SessionRetirement::LeaseForExpansion {
                summary_node_id: 1,
                replaced_span: vec![serde_json::json!({"role": "user", "content": "first"})],
                checkpoint_context: checkpoint_context(100, 1_000),
            },
        );
        assert!(counters.record_higgs_session_id(session, 20));

        counters.retire_higgs_session(
            session,
            SessionRetirement::LeaseForExpansion {
                summary_node_id: 2,
                replaced_span: vec![serde_json::json!({"role": "user", "content": "second"})],
                checkpoint_context: checkpoint_context(200, 2_000),
            },
        );

        assert_eq!(counters.pending_higgs_session_drop_ids(session), vec![10]);
        assert_eq!(
            counters.pending_higgs_session_lease(session),
            Some(HiggsSessionLease {
                session_id: 20,
                ttl_seconds: 300,
            })
        );
        let checkpoint = counters.expansion_checkpoint(session).unwrap();
        assert_eq!(checkpoint.old_higgs_session_id, 20);
        assert_eq!(checkpoint.summary_node_id, 2);
        assert_eq!(counters.active_higgs_session_id(session), None);
    }

    #[test]
    fn exact_expansion_checkpoint_requires_confirmation_and_current_identity() {
        let counters = Arc::new(RuntimeCounters::new_with_config(
            32_768,
            &CircuitBreakerConfig::default(),
        ));
        let session = "cli:exact-expansion-identity";
        let durable = "sqlite:exact-expansion-identity";
        let definitions = vec![serde_json::json!({"type": "function", "name": "read"})];
        let tool_hash = crate::agent::prompt_fingerprint::hash_tools(&definitions);
        counters.install_tool_catalog(session, ToolPresentationMode::Native, definitions.clone());
        let old_id = stable_higgs_session_id(durable, 0);
        assert!(counters.record_higgs_session_id(session, old_id));
        counters.retire_higgs_session(
            session,
            SessionRetirement::LeaseForExpansion {
                summary_node_id: 7,
                replaced_span: vec![
                    serde_json::json!({"role": "user", "content": "raw", "_db_id": 11}),
                ],
                checkpoint_context: counters
                    .expansion_checkpoint_context(session, "bonsai", tool_hash, 900_000)
                    .unwrap(),
            },
        );

        assert_eq!(
            counters.confirmed_expansion_checkpoint(session, "bonsai", tool_hash, 600_000),
            None,
            "an unacknowledged lease must not authorize exact reconstruction"
        );
        let mut lease_attempt = counters
            .reserve_higgs_session_request(session, durable, "bonsai", tool_hash, 31_744, 600_000);
        lease_attempt.resolve_lease(Some(1));
        drop(lease_attempt);

        assert_eq!(
            counters
                .confirmed_expansion_checkpoint(session, "bonsai", tool_hash, 600_001)
                .map(|checkpoint| checkpoint.summary_node_id),
            Some(7)
        );
        for (case, model, hash, now_ms) in [
            ("model", "other", tool_hash, 600_001),
            ("tool hash", "bonsai", tool_hash.wrapping_add(1), 600_001),
            ("expiry", "bonsai", tool_hash, 900_000),
        ] {
            assert_eq!(
                counters.confirmed_expansion_checkpoint(session, model, hash, now_ms),
                None,
                "case {case}"
            );
        }

        counters
            .session_tool_catalogs
            .lock()
            .get_mut(session)
            .unwrap()
            .mode = ToolPresentationMode::Textual;
        assert_eq!(
            counters.confirmed_expansion_checkpoint(session, "bonsai", tool_hash, 600_001),
            None,
            "presentation mismatch"
        );
        let mut catalogs = counters.session_tool_catalogs.lock();
        let catalog = catalogs.get_mut(session).unwrap();
        catalog.mode = ToolPresentationMode::Native;
        catalog.generation += 1;
        drop(catalogs);
        assert_eq!(
            counters.confirmed_expansion_checkpoint(session, "bonsai", tool_hash, 600_001),
            None,
            "catalog generation mismatch"
        );
    }

    #[test]
    fn catalog_replacement_cannot_cross_a_confirmed_checkpoint_snapshot() {
        let counters = Arc::new(RuntimeCounters::new_with_config(
            32_768,
            &CircuitBreakerConfig::default(),
        ));
        let session = "cli:checkpoint-catalog-race";
        let durable = "sqlite:checkpoint-catalog-race";
        let definitions = vec![serde_json::json!({"type": "function", "name": "read"})];
        let tool_hash = crate::agent::prompt_fingerprint::hash_tools(&definitions);
        counters.install_tool_catalog(session, ToolPresentationMode::Native, definitions);
        let old_id = stable_higgs_session_id(durable, 0);
        assert!(counters.record_higgs_session_id(session, old_id));
        counters.retire_higgs_session(
            session,
            SessionRetirement::LeaseForExpansion {
                summary_node_id: 7,
                replaced_span: vec![
                    serde_json::json!({"role": "user", "content": "raw", "_db_id": 11}),
                ],
                checkpoint_context: counters
                    .expansion_checkpoint_context(session, "bonsai", tool_hash, 900_000)
                    .unwrap(),
            },
        );
        let mut lease_attempt = counters
            .reserve_higgs_session_request(session, durable, "bonsai", tool_hash, 31_744, 600_000);
        lease_attempt.resolve_lease(Some(1));
        drop(lease_attempt);

        let snapshot_entered = Arc::new(std::sync::Barrier::new(2));
        let release_snapshot = Arc::new(std::sync::Barrier::new(2));
        let reading = Arc::clone(&counters);
        let entered = Arc::clone(&snapshot_entered);
        let release = Arc::clone(&release_snapshot);
        let reader = std::thread::spawn(move || {
            reading.confirmed_expansion_checkpoint_observed(
                session,
                "bonsai",
                tool_hash,
                600_001,
                || {
                    entered.wait();
                    release.wait();
                },
            )
        });
        snapshot_entered.wait();

        let replacing = Arc::clone(&counters);
        let (replaced_tx, replaced_rx) = std::sync::mpsc::channel();
        let replacer = std::thread::spawn(move || {
            replacing.install_tool_catalog(
                session,
                ToolPresentationMode::Native,
                vec![serde_json::json!({"type": "function", "name": "write"})],
            );
            replaced_tx.send(()).unwrap();
        });
        assert!(
            replaced_rx
                .recv_timeout(std::time::Duration::from_millis(100))
                .is_err(),
            "catalog replacement crossed a partially read checkpoint identity"
        );

        release_snapshot.wait();
        assert_eq!(reader.join().unwrap().unwrap().summary_node_id, 7);
        replaced_rx
            .recv_timeout(std::time::Duration::from_secs(1))
            .expect("catalog replacement did not resume after checkpoint snapshot");
        replacer.join().unwrap();
        assert_eq!(
            counters.confirmed_expansion_checkpoint(session, "bonsai", tool_hash, 600_001),
            None,
            "a checkpoint from the prior catalog generation survived replacement"
        );
    }

    #[test]
    fn drop_waits_for_the_entire_expansion_retirement_transaction() {
        let counters = Arc::new(RuntimeCounters::new_with_config(
            32_768,
            &CircuitBreakerConfig::default(),
        ));
        let session = "cli:atomic-drop";
        assert!(counters.record_higgs_session_id(session, 10));

        let retirement_entered = Arc::new(std::sync::Barrier::new(2));
        let release_retirement = Arc::new(std::sync::Barrier::new(2));
        let retiring = Arc::clone(&counters);
        let entered = Arc::clone(&retirement_entered);
        let release = Arc::clone(&release_retirement);
        let lease_thread = std::thread::spawn(move || {
            retiring.retire_higgs_session_observed(
                session,
                SessionRetirement::LeaseForExpansion {
                    summary_node_id: 1,
                    replaced_span: vec![serde_json::json!({"role": "user", "content": "raw"})],
                    checkpoint_context: checkpoint_context(100, 1_000),
                },
                || {
                    entered.wait();
                    release.wait();
                },
            )
        });
        retirement_entered.wait();

        let dropping = Arc::clone(&counters);
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        let drop_thread = std::thread::spawn(move || {
            dropping.retire_higgs_session(session, SessionRetirement::Drop);
            done_tx.send(()).unwrap();
        });
        assert!(
            done_rx
                .recv_timeout(std::time::Duration::from_millis(100))
                .is_err(),
            "drop crossed a partially installed expansion retirement"
        );

        release_retirement.wait();
        lease_thread.join().unwrap();
        done_rx
            .recv_timeout(std::time::Duration::from_secs(1))
            .expect("drop did not resume after retirement released the state lock");
        drop_thread.join().unwrap();
        assert_eq!(counters.expansion_checkpoint(session), None);
        assert_eq!(counters.pending_higgs_session_lease(session), None);
        assert_eq!(counters.pending_higgs_session_drop_ids(session), vec![10]);
    }

    #[test]
    fn recording_the_expansion_id_waits_for_and_is_rejected_after_retirement() {
        let counters = Arc::new(RuntimeCounters::new_with_config(
            32_768,
            &CircuitBreakerConfig::default(),
        ));
        let session = "cli:atomic-record";
        assert!(counters.record_higgs_session_id(session, 10));

        let retirement_entered = Arc::new(std::sync::Barrier::new(2));
        let release_retirement = Arc::new(std::sync::Barrier::new(2));
        let retiring = Arc::clone(&counters);
        let entered = Arc::clone(&retirement_entered);
        let release = Arc::clone(&release_retirement);
        let lease_thread = std::thread::spawn(move || {
            retiring.retire_higgs_session_observed(
                session,
                SessionRetirement::LeaseForExpansion {
                    summary_node_id: 1,
                    replaced_span: vec![serde_json::json!({"role": "user", "content": "raw"})],
                    checkpoint_context: checkpoint_context(100, 1_000),
                },
                || {
                    entered.wait();
                    release.wait();
                },
            )
        });
        retirement_entered.wait();

        let recording = Arc::clone(&counters);
        let (result_tx, result_rx) = std::sync::mpsc::channel();
        let record_thread = std::thread::spawn(move || {
            result_tx
                .send(recording.record_higgs_session_id(session, 10))
                .unwrap();
        });
        assert!(
            result_rx
                .recv_timeout(std::time::Duration::from_millis(100))
                .is_err(),
            "record crossed a partially installed expansion retirement"
        );

        release_retirement.wait();
        lease_thread.join().unwrap();
        assert!(!result_rx
            .recv_timeout(std::time::Duration::from_secs(1))
            .expect("record did not resume after retirement released the state lock"));
        record_thread.join().unwrap();
        assert_eq!(counters.active_higgs_session_id(session), None);
        assert_eq!(
            counters
                .expansion_checkpoint(session)
                .unwrap()
                .old_higgs_session_id,
            10
        );
    }

    #[test]
    fn activating_after_retirement_skips_a_colliding_expansion_id_atomically() {
        let counters = RuntimeCounters::new_with_config(32_768, &CircuitBreakerConfig::default());
        let session_key = "cli:atomic-activate";
        let durable_session_id = "sqlite:atomic-activate";
        let colliding_id = stable_higgs_session_id(durable_session_id, 1);
        assert!(counters.record_higgs_session_id(session_key, colliding_id));
        counters.retire_higgs_session(
            session_key,
            SessionRetirement::LeaseForExpansion {
                summary_node_id: 1,
                replaced_span: vec![serde_json::json!({"role": "user", "content": "raw"})],
                checkpoint_context: checkpoint_context(100, 1_000),
            },
        );

        let active_id = counters.activate_higgs_session_id(session_key, durable_session_id);

        assert_ne!(active_id, colliding_id);
        assert_eq!(active_id, stable_higgs_session_id(durable_session_id, 2));
        assert_eq!(
            counters.active_higgs_session_id(session_key),
            Some(active_id)
        );
        assert_eq!(counters.session_prompt_epoch(session_key), 2);
    }

    #[test]
    fn fallback_drop_waits_for_the_final_reservation_of_that_id() {
        let counters = Arc::new(RuntimeCounters::new_with_config(
            32_768,
            &CircuitBreakerConfig::default(),
        ));
        let session_key = "cli:deferred-in-flight-drop";
        let durable_session_id = "sqlite:deferred-in-flight-drop";
        let first_x = counters.reserve_higgs_session_request(
            session_key,
            durable_session_id,
            "bonsai",
            0,
            0,
            0,
        );
        let second_x = counters.reserve_higgs_session_request(
            session_key,
            durable_session_id,
            "bonsai",
            0,
            0,
            0,
        );
        let x = first_x.active_id();
        assert_eq!(second_x.active_id(), x);

        counters.retire_higgs_session(
            session_key,
            SessionRetirement::LeaseForExpansion {
                summary_node_id: 1,
                replaced_span: vec![serde_json::json!({"role": "user", "content": "raw"})],
                checkpoint_context: checkpoint_context(100, 1_000),
            },
        );

        let y_while_both_x_live = counters.reserve_higgs_session_request(
            session_key,
            durable_session_id,
            "bonsai",
            0,
            0,
            0,
        );
        assert_ne!(y_while_both_x_live.active_id(), x);
        assert!(!y_while_both_x_live.drop_ids().contains(&x));

        drop(first_x);
        let y_while_one_x_live = counters.reserve_higgs_session_request(
            session_key,
            durable_session_id,
            "bonsai",
            0,
            0,
            0,
        );
        assert!(!y_while_one_x_live.drop_ids().contains(&x));

        drop(second_x);
        let next_request = counters.reserve_higgs_session_request(
            session_key,
            durable_session_id,
            "bonsai",
            0,
            0,
            0,
        );
        assert_eq!(
            next_request
                .drop_ids()
                .iter()
                .filter(|drop_id| **drop_id == x)
                .count(),
            1,
            "the retired ID must become eligible exactly once after its final reservation drops"
        );
    }

    #[test]
    fn trim_retirement_drops_active_and_checkpoint_ids_without_leasing() {
        let counters = RuntimeCounters::new_with_config(32_768, &CircuitBreakerConfig::default());
        let session = "cli:trim-retirement";
        assert!(counters.record_higgs_session_id(session, 10));
        counters.retire_higgs_session(
            session,
            SessionRetirement::LeaseForExpansion {
                summary_node_id: 1,
                replaced_span: vec![serde_json::json!({"role": "user", "content": "raw"})],
                checkpoint_context: checkpoint_context(100, 1_000),
            },
        );
        assert!(counters.record_higgs_session_id(session, 20));

        assert!(counters.invalidate_prompt_cache(session, true));

        assert_eq!(counters.expansion_checkpoint(session), None);
        assert_eq!(counters.pending_higgs_session_lease(session), None);
        assert_eq!(
            counters.pending_higgs_session_drop_ids(session),
            vec![10, 20]
        );
    }

    #[test]
    fn eligible_checkpoint_lease_is_confirmed_only_by_numeric_one() {
        let counters = Arc::new(RuntimeCounters::new_with_config(
            32_768,
            &CircuitBreakerConfig::default(),
        ));
        let session = "cli:lease-confirmation";
        let durable = "sqlite:lease-confirmation";
        let definitions = vec![serde_json::json!({"type": "function", "name": "read"})];
        counters.install_tool_catalog(session, ToolPresentationMode::Native, definitions.clone());
        let tool_hash = crate::agent::prompt_fingerprint::hash_tools(&definitions);
        let checkpoint_context = counters
            .expansion_checkpoint_context(session, "bonsai", tool_hash, 900_000)
            .expect("installed tool catalog must produce checkpoint identity");
        let old_id = stable_higgs_session_id(durable, 0);
        assert!(counters.record_higgs_session_id(session, old_id));
        counters.retire_higgs_session(
            session,
            SessionRetirement::LeaseForExpansion {
                summary_node_id: 44,
                replaced_span: vec![serde_json::json!({"role": "user", "content": "raw"})],
                checkpoint_context,
            },
        );

        let mut request = counters
            .reserve_higgs_session_request(session, durable, "bonsai", tool_hash, 31_744, 600_000);
        assert_ne!(request.control().active_id, old_id);
        assert_eq!(
            request.control().session_lease,
            Some(HiggsSessionLease {
                session_id: old_id,
                ttl_seconds: 300,
            })
        );
        assert_eq!(
            request.control().reuse_policy,
            HiggsSessionReusePolicy::BestEffort
        );
        assert_eq!(request.control().max_prompt_tokens, 31_744);
        request.resolve_lease(Some(1));

        let checkpoint = counters.expansion_checkpoint(session).unwrap();
        assert!(checkpoint.lease_confirmed);
        assert_eq!(counters.pending_higgs_session_lease(session), None);
        assert!(!counters
            .pending_higgs_session_drop_ids(session)
            .contains(&old_id));
    }

    #[test]
    fn retained_expansion_reservation_pins_old_id_without_replacing_active() {
        let counters = Arc::new(RuntimeCounters::new_with_config(
            32_768,
            &CircuitBreakerConfig::default(),
        ));
        let session = "cli:retained-expansion-route";
        let durable = "sqlite:retained-expansion-route";
        let definitions = vec![serde_json::json!({"type": "function", "name": "read"})];
        counters.install_tool_catalog(session, ToolPresentationMode::Native, definitions.clone());
        let tool_hash = crate::agent::prompt_fingerprint::hash_tools(&definitions);
        let old_id = stable_higgs_session_id(durable, 0);
        assert!(counters.record_higgs_session_id(session, old_id));
        counters.retire_higgs_session(
            session,
            SessionRetirement::LeaseForExpansion {
                summary_node_id: 44,
                replaced_span: vec![
                    serde_json::json!({"role": "user", "content": "exact", "_db_id": 7}),
                ],
                checkpoint_context: counters
                    .expansion_checkpoint_context(session, "bonsai", tool_hash, 900_000)
                    .unwrap(),
            },
        );
        let mut lease = counters
            .reserve_higgs_session_request(session, durable, "bonsai", tool_hash, 31_744, 600_000);
        let fresh_id = lease.control().active_id;
        assert_ne!(fresh_id, old_id);
        lease.resolve_lease(Some(1));
        drop(lease);
        let checkpoint = counters.expansion_checkpoint(session).unwrap();

        let retained = counters
            .reserve_retained_expansion_request(
                session,
                &checkpoint,
                "bonsai",
                tool_hash,
                31_744,
                600_001,
            )
            .expect("confirmed current checkpoint must reserve its old ID");

        assert_eq!(retained.control().active_id, old_id);
        assert_eq!(retained.control().session_lease, None);
        assert_eq!(
            retained.control().reuse_policy,
            HiggsSessionReusePolicy::RequireContinuation
        );
        assert_eq!(
            retained.control().reuse_policy.as_wire(),
            "require_continuation"
        );
        assert_eq!(retained.control().max_prompt_tokens, 31_744);
        assert_eq!(counters.active_higgs_session_id(session), Some(fresh_id));
        drop(retained);

        assert!(counters.discard_expansion_checkpoint(session, old_id, 44));
        assert_eq!(counters.expansion_checkpoint(session), None);
        assert_eq!(
            counters.pending_higgs_session_drop_ids(session),
            vec![old_id]
        );
        assert_eq!(counters.active_higgs_session_id(session), Some(fresh_id));
    }

    #[test]
    fn every_retained_reservation_revalidates_current_model_tool_hash_and_catalog() {
        for mismatch in ["model", "tool_hash", "catalog"] {
            let counters = Arc::new(RuntimeCounters::new_with_config(
                32_768,
                &CircuitBreakerConfig::default(),
            ));
            let session = format!("cli:retained-identity-{mismatch}");
            let durable = format!("sqlite:retained-identity-{mismatch}");
            let definitions = vec![serde_json::json!({"type": "function", "name": "read"})];
            counters.install_tool_catalog(
                &session,
                ToolPresentationMode::Native,
                definitions.clone(),
            );
            let tool_hash = crate::agent::prompt_fingerprint::hash_tools(&definitions);
            let old_id = stable_higgs_session_id(&durable, 0);
            assert!(counters.record_higgs_session_id(&session, old_id));
            counters.retire_higgs_session(
                &session,
                SessionRetirement::LeaseForExpansion {
                    summary_node_id: 44,
                    replaced_span: vec![serde_json::json!({
                        "role": "user",
                        "content": "exact",
                        "_db_id": 7,
                    })],
                    checkpoint_context: counters
                        .expansion_checkpoint_context(&session, "bonsai", tool_hash, 900_000)
                        .unwrap(),
                },
            );
            let mut lease = counters.reserve_higgs_session_request(
                &session, &durable, "bonsai", tool_hash, 31_744, 600_000,
            );
            let fresh_id = lease.control().active_id;
            lease.resolve_lease(Some(1));
            drop(lease);
            let checkpoint = counters.expansion_checkpoint(&session).unwrap();

            if mismatch == "catalog" {
                counters.install_tool_catalog(
                    &session,
                    ToolPresentationMode::Native,
                    definitions.clone(),
                );
            }
            let request_model = if mismatch == "model" {
                "different-model"
            } else {
                "bonsai"
            };
            let request_tool_hash = if mismatch == "tool_hash" {
                tool_hash.wrapping_add(1)
            } else {
                tool_hash
            };
            assert!(counters
                .reserve_retained_expansion_request(
                    &session,
                    &checkpoint,
                    request_model,
                    request_tool_hash,
                    31_744,
                    600_001,
                )
                .is_none());
            assert_eq!(counters.active_higgs_session_id(&session), Some(fresh_id));
            assert_eq!(counters.expansion_checkpoint(&session), None);
            assert_eq!(
                counters.pending_higgs_session_drop_ids(&session),
                vec![old_id]
            );
        }
    }

    #[test]
    fn missing_or_zero_lease_ack_discards_checkpoint_and_queues_old_id_once() {
        for (suffix, acknowledgement) in [("missing", None), ("zero", Some(0))] {
            let counters = Arc::new(RuntimeCounters::new_with_config(
                32_768,
                &CircuitBreakerConfig::default(),
            ));
            let session = format!("cli:lease-{suffix}-ack");
            let durable = format!("sqlite:lease-{suffix}-ack");
            let definitions = vec![serde_json::json!({"type": "function", "name": "read"})];
            counters.install_tool_catalog(
                &session,
                ToolPresentationMode::Native,
                definitions.clone(),
            );
            let tool_hash = crate::agent::prompt_fingerprint::hash_tools(&definitions);
            let checkpoint_context = counters
                .expansion_checkpoint_context(&session, "bonsai", tool_hash, 900_000)
                .unwrap();
            let old_id = stable_higgs_session_id(&durable, 0);
            assert!(counters.record_higgs_session_id(&session, old_id));
            counters.retire_higgs_session(
                &session,
                SessionRetirement::LeaseForExpansion {
                    summary_node_id: 44,
                    replaced_span: vec![serde_json::json!({"role": "user", "content": "raw"})],
                    checkpoint_context,
                },
            );
            let mut request = counters.reserve_higgs_session_request(
                &session, &durable, "bonsai", tool_hash, 31_744, 600_000,
            );
            assert_eq!(request.control().session_lease.unwrap().session_id, old_id);

            request.resolve_lease(acknowledgement);

            assert_eq!(counters.expansion_checkpoint(&session), None);
            assert_eq!(counters.pending_higgs_session_lease(&session), None);
            assert_eq!(
                counters.pending_higgs_session_drop_ids(&session),
                vec![old_id]
            );
        }
    }

    #[test]
    fn checkpoint_identity_mismatch_or_expiry_discards_and_drops_old_id() {
        for (suffix, request_model, request_hash, now_ms) in [
            ("expired", "bonsai", 77, 900_000),
            ("model", "other-model", 77, 600_000),
            ("tool-hash", "bonsai", 78, 600_000),
        ] {
            let counters = Arc::new(RuntimeCounters::new_with_config(
                32_768,
                &CircuitBreakerConfig::default(),
            ));
            let session = format!("cli:checkpoint-{suffix}");
            let durable = format!("sqlite:checkpoint-{suffix}");
            counters.install_tool_catalog(
                &session,
                ToolPresentationMode::Native,
                vec![serde_json::json!({"type": "function", "name": "read"})],
            );
            let old_id = stable_higgs_session_id(&durable, 0);
            assert!(counters.record_higgs_session_id(&session, old_id));
            let checkpoint_context = counters
                .expansion_checkpoint_context(&session, "bonsai", 77, 900_000)
                .unwrap();
            counters.retire_higgs_session(
                &session,
                SessionRetirement::LeaseForExpansion {
                    summary_node_id: 44,
                    replaced_span: vec![serde_json::json!({"role": "user", "content": "raw"})],
                    checkpoint_context,
                },
            );

            let request = counters.reserve_higgs_session_request(
                &session,
                &durable,
                request_model,
                request_hash,
                31_744,
                now_ms,
            );

            assert_eq!(request.control().session_lease, None, "case {suffix}");
            assert_eq!(
                counters.expansion_checkpoint(&session),
                None,
                "case {suffix}"
            );
            assert_eq!(
                counters.pending_higgs_session_drop_ids(&session),
                vec![old_id],
                "case {suffix}"
            );
        }
    }

    #[test]
    fn presentation_or_catalog_generation_change_discards_checkpoint() {
        for (suffix, replacement_mode) in [
            ("presentation", ToolPresentationMode::Textual),
            ("generation", ToolPresentationMode::Native),
        ] {
            let counters =
                RuntimeCounters::new_with_config(32_768, &CircuitBreakerConfig::default());
            let session = format!("cli:catalog-{suffix}");
            counters.install_tool_catalog(
                &session,
                ToolPresentationMode::Native,
                vec![serde_json::json!({"type": "function", "name": "read"})],
            );
            assert!(counters.record_higgs_session_id(&session, 41));
            let checkpoint_context = counters
                .expansion_checkpoint_context(&session, "bonsai", 77, 900_000)
                .unwrap();
            counters.retire_higgs_session(
                &session,
                SessionRetirement::LeaseForExpansion {
                    summary_node_id: 44,
                    replaced_span: vec![serde_json::json!({"role": "user", "content": "raw"})],
                    checkpoint_context,
                },
            );

            counters.install_tool_catalog(
                &session,
                replacement_mode,
                vec![serde_json::json!({"type": "function", "name": "write"})],
            );

            assert_eq!(
                counters.expansion_checkpoint(&session),
                None,
                "case {suffix}"
            );
            assert_eq!(
                counters.pending_higgs_session_lease(&session),
                None,
                "case {suffix}"
            );
            assert_eq!(counters.pending_higgs_session_drop_ids(&session), vec![41]);
        }
    }

    #[test]
    fn pending_lease_is_claimed_by_exactly_one_concurrent_reservation() {
        let counters = Arc::new(RuntimeCounters::new_with_config(
            32_768,
            &CircuitBreakerConfig::default(),
        ));
        let session = "cli:single-lease-claim";
        let durable = "sqlite:single-lease-claim";
        let definitions = vec![serde_json::json!({"type": "function", "name": "read"})];
        counters.install_tool_catalog(session, ToolPresentationMode::Native, definitions.clone());
        let tool_hash = crate::agent::prompt_fingerprint::hash_tools(&definitions);
        let old_id = stable_higgs_session_id(durable, 0);
        assert!(counters.record_higgs_session_id(session, old_id));
        counters.retire_higgs_session(
            session,
            SessionRetirement::LeaseForExpansion {
                summary_node_id: 44,
                replaced_span: vec![serde_json::json!({"role": "user", "content": "raw"})],
                checkpoint_context: counters
                    .expansion_checkpoint_context(session, "bonsai", tool_hash, 900_000)
                    .unwrap(),
            },
        );

        let first = counters
            .reserve_higgs_session_request(session, durable, "bonsai", tool_hash, 31_744, 600_000);
        let second = counters
            .reserve_higgs_session_request(session, durable, "bonsai", tool_hash, 31_744, 600_000);

        assert_eq!(first.control().session_lease.unwrap().session_id, old_id);
        assert_eq!(second.control().session_lease, None);
    }

    #[test]
    fn invalidated_claimed_lease_is_not_dropped_until_attempt_releases() {
        let counters = Arc::new(RuntimeCounters::new_with_config(
            32_768,
            &CircuitBreakerConfig::default(),
        ));
        let session = "cli:in-flight-lease-drop";
        let durable = "sqlite:in-flight-lease-drop";
        let definitions = vec![serde_json::json!({"type": "function", "name": "read"})];
        counters.install_tool_catalog(session, ToolPresentationMode::Native, definitions.clone());
        let tool_hash = crate::agent::prompt_fingerprint::hash_tools(&definitions);
        let old_id = stable_higgs_session_id(durable, 0);
        assert!(counters.record_higgs_session_id(session, old_id));
        counters.retire_higgs_session(
            session,
            SessionRetirement::LeaseForExpansion {
                summary_node_id: 44,
                replaced_span: vec![serde_json::json!({"role": "user", "content": "raw"})],
                checkpoint_context: counters
                    .expansion_checkpoint_context(session, "bonsai", tool_hash, 900_000)
                    .unwrap(),
            },
        );
        let lease_attempt = counters
            .reserve_higgs_session_request(session, durable, "bonsai", tool_hash, 31_744, 600_000);
        assert_eq!(
            lease_attempt.control().session_lease.unwrap().session_id,
            old_id
        );

        counters.install_tool_catalog(
            session,
            ToolPresentationMode::Native,
            vec![serde_json::json!({"type": "function", "name": "write"})],
        );
        let while_lease_live = counters
            .reserve_higgs_session_request(session, durable, "bonsai", tool_hash, 31_744, 600_001);
        assert!(!while_lease_live.drop_ids().contains(&old_id));

        drop(lease_attempt);
        let after_release = counters
            .reserve_higgs_session_request(session, durable, "bonsai", tool_hash, 31_744, 600_002);
        assert_eq!(
            after_release
                .drop_ids()
                .iter()
                .filter(|candidate| **candidate == old_id)
                .count(),
            1
        );
    }

    #[test]
    fn unresolved_lease_terminal_paths_discard_checkpoint() {
        for terminal_path in [
            "blocking error",
            "stream error",
            "cancellation",
            "timeout",
            "stream without Done",
        ] {
            let counters = Arc::new(RuntimeCounters::new_with_config(
                32_768,
                &CircuitBreakerConfig::default(),
            ));
            let session = format!("cli:lease-attempt-drop:{terminal_path}");
            let durable = format!("sqlite:lease-attempt-drop:{terminal_path}");
            let definitions = vec![serde_json::json!({"type": "function", "name": "read"})];
            counters.install_tool_catalog(
                &session,
                ToolPresentationMode::Native,
                definitions.clone(),
            );
            let tool_hash = crate::agent::prompt_fingerprint::hash_tools(&definitions);
            let old_id = stable_higgs_session_id(&durable, 0);
            assert!(counters.record_higgs_session_id(&session, old_id));
            counters.retire_higgs_session(
                &session,
                SessionRetirement::LeaseForExpansion {
                    summary_node_id: 44,
                    replaced_span: vec![serde_json::json!({"role": "user", "content": "raw"})],
                    checkpoint_context: counters
                        .expansion_checkpoint_context(&session, "bonsai", tool_hash, 900_000)
                        .unwrap(),
                },
            );
            let attempt = counters.reserve_higgs_session_request(
                &session, &durable, "bonsai", tool_hash, 31_744, 600_000,
            );
            assert!(
                attempt.control().session_lease.is_some(),
                "{terminal_path} must claim the lease"
            );

            // Every provider exit owns the reservation until the exit boundary.
            // Dropping it is the shared fail-closed finalizer for all five paths.
            drop(attempt);

            assert_eq!(
                counters.expansion_checkpoint(&session),
                None,
                "{terminal_path} must discard the unconfirmed checkpoint"
            );
            assert_eq!(
                counters.pending_higgs_session_lease(&session),
                None,
                "{terminal_path} must not leave the lease retransmittable"
            );
            assert_eq!(
                counters.pending_higgs_session_drop_ids(&session),
                vec![old_id],
                "{terminal_path} must queue the old session for deletion"
            );
        }
    }

    #[test]
    fn test_history_limits_lcm_ceiling_clears_soft_threshold() {
        // Invariant: with LCM enabled, the filter_history token ceiling
        // (limit * 150) must exceed LCM's soft compaction threshold —
        // tau_soft(0.5) * (C - ~25% worst-case reserve/tool overhead) —
        // by at least a ~2-turn (600 token) margin. Otherwise trimming caps
        // conversation tokens below the trigger and compaction never fires.
        for c in [3200usize, 8192, 32768, 131072, 200000] {
            let ceiling = history_limit_lcm(c) * 150;
            let soft_threshold = (c - c / 4) / 2; // 0.5 * (C - C/4)
            assert!(
                ceiling > soft_threshold + 600,
                "C={c}: ceiling {ceiling} must exceed soft threshold {soft_threshold} + 600"
            );
        }

        // LCM mode keeps the lower clamp and scales past the old 600 cap.
        assert_eq!(history_limit_lcm(1000), 6);
        assert_eq!(history_limit_lcm(200000), 933);
    }

    #[test]
    fn compaction_swap_preserves_prompt_prefix_summary_and_appended_suffix() {
        let snapshot = vec![
            serde_json::json!({"role": "system", "content": "system"}),
            serde_json::json!({"role": "developer", "content": "old developer"}),
            serde_json::json!({"role": "user", "content": "old", "_db_id": 1}),
        ];
        let mut live = vec![
            serde_json::json!({"role": "system", "content": "current system"}),
            serde_json::json!({"role": "developer", "content": "current working memory"}),
            snapshot[2].clone(),
        ];
        live.push(serde_json::json!({
            "role": "tool",
            "content": "new result",
            "tool_call_id": "call-1",
            "_db_id": 2
        }));
        let summary = serde_json::json!({
            "role": "user",
            "content": "summary",
            "_lcm_summary": true
        });
        let pending = PendingCompaction {
            result: crate::agent::compaction::CompactionResult {
                messages: vec![snapshot[0].clone(), snapshot[1].clone(), summary.clone()],
            },
            snapshot,
            summary_node_id: 0,
        };

        assert!(apply_compaction_result(&mut live, pending));
        assert_eq!(live[0]["content"], "current system");
        assert_eq!(live[1]["content"], "current working memory");
        assert_eq!(live[2], summary);
        assert_eq!(live[3]["content"], "new result");
    }

    #[test]
    fn compaction_swap_accepts_an_added_developer_prefix() {
        let snapshot = vec![
            serde_json::json!({"role": "system", "content": "system"}),
            serde_json::json!({"role": "user", "content": "old", "_db_id": 1}),
        ];
        let summary = serde_json::json!({
            "role": "user",
            "content": "summary",
            "_lcm_summary": true
        });
        let mut live = vec![
            serde_json::json!({"role": "system", "content": "current system"}),
            serde_json::json!({"role": "developer", "content": "new working memory"}),
            snapshot[1].clone(),
            serde_json::json!({"role": "assistant", "content": "new tail", "_db_id": 2}),
        ];
        let pending = PendingCompaction {
            result: crate::agent::compaction::CompactionResult {
                messages: vec![snapshot[0].clone(), summary.clone()],
            },
            snapshot,
            summary_node_id: 0,
        };

        assert!(apply_compaction_result(&mut live, pending));
        assert_eq!(live[0]["content"], "current system");
        assert_eq!(live[1]["content"], "new working memory");
        assert_eq!(live[2], summary);
        assert_eq!(live[3]["content"], "new tail");
    }

    #[test]
    fn compaction_swap_rejects_a_rewritten_snapshot_without_mutation() {
        let snapshot = vec![
            serde_json::json!({"role": "system", "content": "system"}),
            serde_json::json!({"role": "user", "content": "old", "_db_id": 1}),
        ];
        let mut live = vec![
            serde_json::json!({"role": "system", "content": "current system"}),
            serde_json::json!({"role": "developer", "content": "current working memory"}),
            serde_json::json!({"role": "user", "content": "rewritten", "_db_id": 2}),
        ];
        let before = live.clone();
        let pending = PendingCompaction {
            result: crate::agent::compaction::CompactionResult {
                messages: vec![
                    snapshot[0].clone(),
                    serde_json::json!({"role": "user", "content": "summary", "_lcm_summary": true}),
                ],
            },
            snapshot,
            summary_node_id: 0,
        };

        assert!(!apply_compaction_result(&mut live, pending));
        assert_eq!(live, before);
    }

    /// A sanctioned reset must be attributable exactly once, per session.
    ///
    /// Once, because the reason is consumed by the next provider call — if it
    /// lingered, every later call in the session would re-report the same
    /// re-prefill and the ledger would overcount. Per session, because one
    /// session's `/clear` must not be blamed on another's next turn.
    #[test]
    fn cache_reset_reason_is_taken_once_per_session() {
        let counters = RuntimeCounters::new_with_config(16384, &CircuitBreakerConfig::default());

        // Never reset → nothing to attribute; this is a genuine cold start.
        assert_eq!(counters.take_cache_reset("a"), None);

        counters.note_cache_reset("a", "trim");
        assert_eq!(counters.take_cache_reset("a"), Some("trim"));
        assert_eq!(
            counters.take_cache_reset("a"),
            None,
            "a reset must not be reported twice"
        );

        // Sessions are independent.
        counters.note_cache_reset("a", "history_reload");
        assert_eq!(counters.take_cache_reset("b"), None);
        assert_eq!(counters.take_cache_reset("a"), Some("history_reload"));

        // The real reset paths record a reason, not just clear state.
        counters.reset_session_prompt_state("c");
        assert_eq!(counters.take_cache_reset("c"), Some("session_reset"));
    }

    #[test]
    fn test_trio_state_default_is_standalone() {
        let counters = RuntimeCounters::new_with_config(16384, &CircuitBreakerConfig::default());
        assert_eq!(counters.get_trio_state(), TrioState::Standalone);
    }

    #[test]
    fn cache_metrics_are_aggregated_by_logical_session() {
        let counters = RuntimeCounters::new_with_config(16384, &CircuitBreakerConfig::default());

        counters.record_cache_metrics("logical-a", 100, Some(75), Some(25));
        counters.record_cache_metrics("logical-a", 50, Some(0), Some(50));
        counters.record_cache_metrics("logical-b", 20, Some(20), Some(0));

        let a = counters.session_cache_metrics("logical-a");
        assert_eq!(a.calls, 2);
        assert_eq!(a.prompt_tokens, 150);
        assert_eq!(a.cache_read_tokens, 75);
        assert_eq!(a.cache_creation_tokens, 75);
        assert_eq!(a.cold_calls, 1);
        assert_eq!(a.efficiency_pct(), 50.0);

        let b = counters.session_cache_metrics("logical-b");
        assert_eq!(b.calls, 1);
        assert_eq!(b.cache_read_tokens, 20);
    }

    #[test]
    fn frozen_tool_catalog_reuses_final_defs_until_presentation_mode_changes() {
        let counters = RuntimeCounters::new_with_config(16384, &CircuitBreakerConfig::default());
        let session = "cli:catalog";
        let native = vec![serde_json::json!({"function": {"name": "read_file"}})];
        let availability_flip = vec![serde_json::json!({"function": {"name": "exec"}})];

        assert_eq!(
            counters.frozen_tool_definitions(session, ToolPresentationMode::Native),
            None
        );
        counters.install_tool_catalog(session, ToolPresentationMode::Native, native.clone());
        assert_eq!(
            counters.frozen_tool_definitions(session, ToolPresentationMode::Native),
            Some(native.clone()),
            "same-mode availability changes must reuse the final frozen array"
        );
        assert_ne!(native, availability_flip);

        let mut previous_mode = ToolPresentationMode::Native;
        for mode in [
            ToolPresentationMode::Textual,
            ToolPresentationMode::Trio,
            ToolPresentationMode::ForcedText,
        ] {
            assert!(counters.tool_presentation_mode_changed(session, mode));
            assert_eq!(
                counters.tool_presentation_mode(session),
                Some(previous_mode),
                "detecting a transition must not install before rotation"
            );
            counters.install_tool_catalog(session, mode, Vec::new());
            assert_eq!(
                counters.frozen_tool_definitions(session, mode),
                Some(Vec::new())
            );
            previous_mode = mode;
        }
    }

    #[test]
    fn compaction_rotation_preserves_catalog_but_logical_reset_clears_it() {
        let counters = RuntimeCounters::new_with_config(16384, &CircuitBreakerConfig::default());
        let session = "cli:catalog-reset";
        let defs = vec![serde_json::json!({"function": {"name": "read_file"}})];
        counters.install_tool_catalog(session, ToolPresentationMode::Native, defs.clone());

        counters.invalidate_prompt_cache(session, true);
        assert_eq!(
            counters.frozen_tool_definitions(session, ToolPresentationMode::Native),
            Some(defs),
            "LCM/trim prompt rotation must preserve the frozen catalog"
        );

        counters.reset_session_prompt_state(session);
        assert_eq!(
            counters.frozen_tool_definitions(session, ToolPresentationMode::Native),
            None,
            "logical reset/model change must clear the frozen catalog"
        );
    }

    #[test]
    fn test_trio_state_transitions() {
        let counters = RuntimeCounters::new_with_config(16384, &CircuitBreakerConfig::default());

        counters.set_trio_state(TrioState::Active);
        assert_eq!(counters.get_trio_state(), TrioState::Active);

        counters.set_trio_state(TrioState::Degraded);
        assert_eq!(counters.get_trio_state(), TrioState::Degraded);

        counters.set_trio_state(TrioState::Standalone);
        assert_eq!(counters.get_trio_state(), TrioState::Standalone);
    }

    #[test]
    fn test_trio_state_no_log_on_same_state() {
        // Setting the same state twice should not log (swap returns same value).
        let counters = RuntimeCounters::new_with_config(16384, &CircuitBreakerConfig::default());

        counters.set_trio_state(TrioState::Active);
        // Second call with same state — no log, no panic.
        counters.set_trio_state(TrioState::Active);
        assert_eq!(counters.get_trio_state(), TrioState::Active);
    }

    /// Epoch rotation is the SOLE cache-invalidation mechanism after a reset:
    /// it folds into `stable_higgs_session_id`, giving the server a brand-new
    /// session id (cold start). System-message content is no longer mutated,
    /// so a compaction/trim no longer re-prefills message 0.
    #[test]
    fn test_reset_rotates_higgs_session_id_purely_via_epoch() {
        let s = "cli:test";
        assert_ne!(stable_higgs_session_id(s, 0), stable_higgs_session_id(s, 1));
        assert_ne!(stable_higgs_session_id(s, 1), stable_higgs_session_id(s, 2));
        // Session identity still dominates epoch: different sessions never
        // collide even at the same epoch.
        assert_ne!(
            stable_higgs_session_id(s, 1),
            stable_higgs_session_id("cli:other", 1)
        );
    }

    /// Consolidated prompt-cache invalidation for a sanctioned rewrite (trim,
    /// compaction). On a higgs-capable backend the rewrite must ROTATE the
    /// retained session (epoch bump + queued drop + cleared fingerprint), so the
    /// server cold-starts the shrunken prompt instead of rejecting it as
    /// "not_growing" and re-prefilling under the stale session id. On a
    /// non-higgs backend it clears only the local fingerprint/watermark.
    /// Both branches are exercised here — this is the logic the trim and
    /// compaction paths share via `invalidate_prompt_cache_for_rewrite`.
    #[test]
    fn test_invalidate_prompt_cache_rotates_when_higgs_capable_clears_otherwise() {
        let counters = RuntimeCounters::new_with_config(16384, &CircuitBreakerConfig::default());
        let s = "cli:trim";

        // Warm state: an active higgs session, a stored fingerprint, a watermark.
        counters.record_higgs_session_id(s, 100);
        let fp = crate::agent::prompt_fingerprint::fingerprint(&[serde_json::json!({
            "role": "user",
            "content": "hi",
        })]);
        counters
            .prompt_fingerprints
            .lock()
            .insert(s.to_string(), fp);
        counters
            .prompt_cache_watermark
            .lock()
            .insert(s.to_string(), 7);

        // Higgs-capable rewrite → rotate: epoch bumps, old id is queued for
        // drop, and the warm fingerprint is cleared (forces a fresh prefix).
        let rotated = counters.invalidate_prompt_cache(s, true);
        assert!(rotated, "higgs-capable rewrite must rotate the session");
        assert_eq!(counters.session_prompt_epoch(s), 1);
        assert_eq!(counters.pending_higgs_session_drop_ids(s), vec![100]);
        assert!(
            !counters.prompt_fingerprints.lock().contains_key(s),
            "rotation must clear the stale prefix fingerprint"
        );
        // Clear the queued drop so the non-rotating branch starts clean.
        assert!(counters.clear_pending_higgs_session_drop_id(s, 100));

        // Non-higgs rewrite → clear local bookkeeping only: NO epoch bump, NO
        // drop, the active session id stays live.
        counters.record_higgs_session_id(s, 200);
        counters
            .prompt_cache_watermark
            .lock()
            .insert(s.to_string(), 9);
        let rotated = counters.invalidate_prompt_cache(s, false);
        assert!(!rotated, "non-higgs rewrite must not rotate");
        assert_eq!(counters.session_prompt_epoch(s), 1, "epoch unchanged");
        assert!(
            counters.pending_higgs_session_drop_ids(s).is_empty(),
            "no drop queued for a non-rotating clear"
        );
        assert_eq!(
            counters.active_higgs_session_id(s),
            Some(200),
            "active session id must survive a non-rotating clear"
        );
        assert!(
            !counters.prompt_cache_watermark.lock().contains_key(s),
            "local watermark must be cleared"
        );
    }

    #[test]
    fn test_reset_session_prompt_state_clears_cache_and_bumps_epoch() {
        let counters = RuntimeCounters::new_with_config(16384, &CircuitBreakerConfig::default());
        let session = "cli:test";
        let fp = crate::agent::prompt_fingerprint::fingerprint(&[serde_json::json!({
            "role": "user",
            "content": "hi",
        })]);

        counters
            .prompt_fingerprints
            .lock()
            .insert(session.to_string(), fp);
        counters
            .prompt_cache_watermark
            .lock()
            .insert(session.to_string(), 7);
        counters.record_local_artifact_intent(session, 10, true);

        counters.record_higgs_session_id(session, 10);
        assert_eq!(counters.reset_session_prompt_state(session), 1);
        assert!(!counters.prompt_fingerprints.lock().contains_key(session));
        assert!(!counters.prompt_cache_watermark.lock().contains_key(session));
        assert_eq!(counters.local_artifact_intent_is_rich(session, 10), None);
        assert_eq!(counters.session_prompt_epoch(session), 1);
        assert_eq!(counters.pending_higgs_session_drop_ids(session), vec![10]);
        assert!(counters.clear_pending_higgs_session_drop_id(session, 10));
        assert!(counters.pending_higgs_session_drop_ids(session).is_empty());

        counters.record_higgs_session_id(session, 11);
        assert_eq!(counters.reset_session_prompt_state(session), 2);
        assert_eq!(counters.session_prompt_epoch(session), 2);
        assert_eq!(counters.pending_higgs_session_drop_ids(session), vec![11]);
    }

    #[test]
    fn test_reset_session_prompt_state_queues_multiple_higgs_drops() {
        let counters = RuntimeCounters::new_with_config(16384, &CircuitBreakerConfig::default());
        let session = "cli:test";

        for (epoch, drop_id) in [10, 11, 12].into_iter().enumerate() {
            counters.record_higgs_session_id(session, drop_id);
            assert_eq!(
                counters.reset_session_prompt_state(session),
                epoch as u64 + 1
            );
        }
        assert_eq!(
            counters.pending_higgs_session_drop_ids(session),
            vec![10, 11, 12]
        );

        assert_eq!(
            counters.clear_pending_higgs_session_drop_ids(session, &[10, 12]),
            2
        );
        assert_eq!(counters.pending_higgs_session_drop_ids(session), vec![11]);
        assert!(!counters.clear_pending_higgs_session_drop_id(session, 10));
        assert!(counters.clear_pending_higgs_session_drop_id(session, 11));
        assert!(counters.pending_higgs_session_drop_ids(session).is_empty());
    }

    #[test]
    fn test_pending_higgs_drop_keeps_id_from_before_session_rollover() {
        let counters = RuntimeCounters::new_with_config(16384, &CircuitBreakerConfig::default());
        let session_key = "cli:test";
        let original_session_id = "sqlite-session-before-clear";
        let rolled_over_session_id = "sqlite-session-after-clear";

        let original_drop_id = stable_higgs_session_id(original_session_id, 0);
        counters.record_higgs_session_id(session_key, original_drop_id);
        counters.reset_session_prompt_state(session_key);
        counters.record_higgs_session_id(
            session_key,
            stable_higgs_session_id(rolled_over_session_id, 1),
        );

        assert_eq!(
            counters.pending_higgs_session_drop_ids(session_key),
            vec![original_drop_id]
        );
    }

    #[test]
    fn test_local_artifact_intent_is_bounded_by_turn_count() {
        let counters = RuntimeCounters::new_with_config(16384, &CircuitBreakerConfig::default());
        let session = "cli:test";

        counters.record_local_artifact_intent(session, 10, true);
        assert_eq!(
            counters.local_artifact_intent_is_rich(session, 10),
            Some(true)
        );
        assert_eq!(
            counters.local_artifact_intent_is_rich(session, 14),
            Some(true)
        );
        assert_eq!(counters.local_artifact_intent_is_rich(session, 15), None);

        counters.record_local_artifact_intent(session, 20, false);
        assert_eq!(
            counters.local_artifact_intent_is_rich(session, 21),
            Some(false)
        );
        assert!(counters.clear_local_artifact_intent(session));
        assert_eq!(counters.local_artifact_intent_is_rich(session, 21), None);
    }
}
