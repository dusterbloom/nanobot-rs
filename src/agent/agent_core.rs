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
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

use serde_json::Value;
use tracing::debug;

use crate::agent::agent_profiles;
use crate::agent::compaction::ContextCompactor;
use crate::agent::context::ContextBuilder;
use crate::agent::lane::Lane;
use crate::agent::prompt_fingerprint::PromptFingerprint;
use crate::agent::runtime_mode::RuntimeMode;
use crate::agent::token_budget::TokenBudget;
use crate::agent::working_memory::WorkingMemoryStore;
use crate::config::schema::{
    AdaptiveTokenConfig, MemoryConfig, ProvenanceConfig, ToolDelegationConfig, TrioConfig,
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
    /// Word cap enforced on the long-term MEMORY.md file at reflection time.
    pub memory_file_max_words: usize,
    /// Typed runtime descriptor. Single source of truth for "is this a local
    /// backend?" via `mode().is_local()`. The legacy `is_local: bool` field was
    /// removed in R6 — it duplicated information already carried by this enum.
    pub mode: RuntimeMode,
    pub lane: Lane,
    pub tool_delegation_config: ToolDelegationConfig,
    pub provenance_config: ProvenanceConfig,
    pub max_tool_result_chars: usize,
    pub session_complete_after_secs: u64,
    pub max_history_turns: usize,
    pub model_capabilities: crate::agent::model_capabilities::ModelCapabilities,
    /// Single owner of hygiene/anti-drift retention knobs. See
    /// `agent::retention` — replaces the formerly separate `anti_drift` and
    /// `hygiene_keep_last_messages` fields.
    pub retention: crate::agent::retention::RetentionPolicy,
    pub reasoning_config: crate::config::schema::ReasoningConfig,
    /// Code execution tool config.
    pub code_execution: crate::config::schema::CodeExecutionConfig,
    /// Python kernel tool config (feature: python-kernel).
    pub python_kernel: crate::config::schema::PythonKernelConfig,
    /// Cua driver (local desktop computer-use) tool settings.
    pub cua: crate::config::schema::CuaToolConfig,
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

/// How the final post-policy tool catalog is presented to the main model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ToolPresentationMode {
    Native,
    Textual,
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

/// The only prompt bytes that survive an LCM rewrite. The provider can reuse
/// this exact prefix even though the retained session id is rotated.
#[derive(Clone, Debug)]
pub(crate) struct StablePromptAnchor {
    pub(crate) fingerprint: PromptFingerprint,
    pub(crate) watermark: usize,
    pub(crate) tool_hash: u64,
}

#[derive(Clone, Debug)]
enum PromptCacheBookkeeping {
    Clear,
    Preserve(StablePromptAnchor),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum HiggsSessionReusePolicy {
    Seed,
    RequireContinuation,
}

impl HiggsSessionReusePolicy {
    pub(crate) fn as_wire(self) -> &'static str {
        match self {
            Self::Seed => "seed",
            Self::RequireContinuation => "require_continuation",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct HiggsSessionControl {
    pub(crate) active_id: u64,
    pub(crate) drop_ids: Vec<u64>,
    pub(crate) reuse_policy: HiggsSessionReusePolicy,
    pub(crate) max_prompt_tokens: u32,
}

#[derive(Default)]
struct HiggsSessionState {
    epoch: u64,
    active_id: Option<u64>,
    /// Active ID whose exact bytes were acknowledged by Higgs. A different
    /// or absent ID is explicitly unseeded and may not require continuation.
    published_active_id: Option<u64>,
    pending_drop_ids: Vec<u64>,
    in_flight_active_ids: std::collections::HashMap<u64, usize>,
    /// One-shot exception for the intentional LCM session rotation: the next
    /// request may reuse the stable prefix anchor instead of looking cold.
    preserved_prefix_epoch: Option<u64>,
}

pub(crate) struct HiggsSessionRequestReservation {
    counters: Arc<RuntimeCounters>,
    session_key: String,
    control: HiggsSessionControl,
}

impl HiggsSessionRequestReservation {
    #[cfg(test)]
    pub(crate) fn active_id(&self) -> u64 {
        self.control.active_id
    }

    pub(crate) fn publish_exact(&self) {
        let mut sessions = self.counters.higgs_sessions.lock();
        if let Some(state) = sessions.get_mut(&self.session_key) {
            if state.active_id == Some(self.control.active_id) {
                state.published_active_id = Some(self.control.active_id);
            }
        }
    }

    pub(crate) fn drop_ids(&self) -> &[u64] {
        &self.control.drop_ids
    }

    pub(crate) fn control(&self) -> &HiggsSessionControl {
        &self.control
    }
}

impl Drop for HiggsSessionRequestReservation {
    fn drop(&mut self) {
        self.counters
            .release_higgs_session_request(&self.session_key, self.control.active_id);
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
    /// Retained-session transitions are one transaction: epoch, active id and
    /// pending drops must never be observed in partially updated combinations
    /// by concurrent requests or resets.
    higgs_sessions: parking_lot::Mutex<std::collections::HashMap<String, HiggsSessionState>>,
    /// Eager-drop flusher, wired by the embedding loop with the active
    /// provider: fires a standalone higgs session-drop right after a
    /// rotation queues one, so the retired session's resident KV frees
    /// before the next prompt prefills instead of riding that request.
    higgs_drop_flusher: parking_lot::Mutex<Option<std::sync::Arc<dyn Fn(u64) + Send + Sync>>>,
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
    pub fn new(max_context_tokens: usize) -> Self {
        Self {
            learning_turn_counter: AtomicU64::new(0),
            last_context_used: AtomicU64::new(0),
            last_context_max: AtomicU64::new(max_context_tokens as u64),
            last_message_count: AtomicU64::new(0),
            last_working_memory_tokens: AtomicU64::new(0),
            last_tools_called: parking_lot::Mutex::new(Vec::new()),
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
            prompt_fingerprints: parking_lot::Mutex::new(std::collections::HashMap::new()),
            prompt_head_hashes: parking_lot::Mutex::new(std::collections::HashMap::new()),
            prompt_tool_hashes: parking_lot::Mutex::new(std::collections::HashMap::new()),
            session_tool_catalogs: parking_lot::Mutex::new(std::collections::HashMap::new()),
            prompt_cache_watermark: parking_lot::Mutex::new(std::collections::HashMap::new()),
            prompt_cache_transition: parking_lot::Mutex::new(()),
            higgs_sessions: parking_lot::Mutex::new(std::collections::HashMap::new()),
            higgs_drop_flusher: parking_lot::Mutex::new(None),
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
            // A new logical session may legitimately start with a new head.
            self.prompt_head_hashes.lock().remove(session_key);
            self.note_cache_reset(session_key, "session_reset");
        }
        self.retire_higgs_session(session_key)
    }

    /// Reset all prompt-cache bookkeeping for a session and advance its prompt epoch.
    ///
    /// The epoch is rendered into the next prompt as a tiny stable marker. This
    /// forces local resident servers to treat post-clear/post-switch prompts as
    /// a fresh prefix even when the user starts with identical text like `hi`.
    pub fn reset_session_prompt_state(&self, session_key: &str) -> u64 {
        self.rotate_prompt_session(session_key, PromptResetScope::LogicalSession)
    }

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

    /// Publish the unchanged prefix after a non-Higgs LCM rewrite. The next
    /// request compares its new summary/tail against this anchor and charges
    /// only the changed suffix as prefill.
    pub(crate) fn reanchor_local_prompt_cache(
        &self,
        session_key: &str,
        anchor: StablePromptAnchor,
    ) {
        let _transition = self.prompt_cache_transition.lock();
        self.prompt_fingerprints
            .lock()
            .insert(session_key.to_string(), anchor.fingerprint);
        self.prompt_tool_hashes
            .lock()
            .insert(session_key.to_string(), anchor.tool_hash);
        self.prompt_cache_watermark
            .lock()
            .insert(session_key.to_string(), anchor.watermark);
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
    pub fn record_higgs_session_id(&self, session_key: &str, session_id: u64) {
        let mut sessions = self.higgs_sessions.lock();
        let state = sessions.entry(session_key.to_string()).or_default();
        state.active_id = Some(session_id);
    }

    fn activate_higgs_session_state(
        state: &mut HiggsSessionState,
        durable_session_id: &str,
    ) -> u64 {
        let active_id = stable_higgs_session_id(durable_session_id, state.epoch);
        state.active_id = Some(active_id);
        active_id
    }

    /// Restore the conservative server-publication hint after a process restart.
    ///
    /// A resumed durable transcript has already completed at least one assistant
    /// turn. Trying the deterministic retained ID as an exact continuation is
    /// safe: if Higgs restarted or evicted it, the typed unavailable path rotates
    /// and seeds once. Never overwrite live in-process state, especially after a
    /// deliberate compaction rotation.
    pub(crate) fn restore_higgs_publication_hint(
        &self,
        session_key: &str,
        durable_session_id: &str,
    ) -> bool {
        let _transition = self.prompt_cache_transition.lock();
        let mut sessions = self.higgs_sessions.lock();
        if sessions.contains_key(session_key) {
            return false;
        }
        let active_id = stable_higgs_session_id(durable_session_id, 0);
        sessions.insert(
            session_key.to_string(),
            HiggsSessionState {
                active_id: Some(active_id),
                published_active_id: Some(active_id),
                ..HiggsSessionState::default()
            },
        );
        true
    }

    pub(crate) fn reserve_higgs_session_request(
        self: &Arc<Self>,
        session_key: &str,
        durable_session_id: &str,
        max_prompt_tokens: u32,
    ) -> HiggsSessionRequestReservation {
        let _transition = self.prompt_cache_transition.lock();
        let mut sessions = self.higgs_sessions.lock();
        let state = sessions.entry(session_key.to_string()).or_default();
        let prior_route_identity = (state.active_id, state.epoch);
        let active_id = Self::activate_higgs_session_state(state, durable_session_id);
        let preserve_prefix = state.preserved_prefix_epoch == Some(state.epoch);
        if prior_route_identity != (Some(active_id), state.epoch) && !preserve_prefix {
            self.prompt_fingerprints.lock().remove(session_key);
            self.prompt_cache_watermark.lock().remove(session_key);
        }
        state.preserved_prefix_epoch = None;
        *state.in_flight_active_ids.entry(active_id).or_default() += 1;
        let drop_ids = state
            .pending_drop_ids
            .iter()
            .copied()
            .filter(|drop_id| !state.in_flight_active_ids.contains_key(drop_id))
            .collect();
        let reuse_policy = if state.published_active_id == Some(active_id) {
            HiggsSessionReusePolicy::RequireContinuation
        } else {
            HiggsSessionReusePolicy::Seed
        };
        HiggsSessionRequestReservation {
            counters: Arc::clone(self),
            session_key: session_key.to_string(),
            control: HiggsSessionControl {
                active_id,
                drop_ids,
                reuse_policy,
                max_prompt_tokens,
            },
        }
    }

    fn release_higgs_session_request(&self, session_key: &str, active_id: u64) {
        let mut sessions = self.higgs_sessions.lock();
        let Some(state) = sessions.get_mut(session_key) else {
            return;
        };
        let Some(in_flight) = state.in_flight_active_ids.get_mut(&active_id) else {
            return;
        };
        *in_flight -= 1;
        if *in_flight == 0 {
            state.in_flight_active_ids.remove(&active_id);
        }
    }

    fn queue_higgs_session_drop(state: &mut HiggsSessionState, session_id: u64) {
        if !state.pending_drop_ids.contains(&session_id) {
            state.pending_drop_ids.push(session_id);
        }
    }

    /// Wire the eager higgs session-drop flusher. Called once by the
    /// embedding loop with the live provider; fires fire-and-forget drops
    /// on rotation. No-op until set (plain CLI/tests).
    pub(crate) fn set_higgs_drop_flusher(
        &self,
        flusher: std::sync::Arc<dyn Fn(u64) + Send + Sync>,
    ) {
        *self.higgs_drop_flusher.lock() = Some(flusher);
    }

    pub(crate) fn retire_higgs_session(&self, session_key: &str) -> u64 {
        self.retire_higgs_session_inner(session_key, PromptCacheBookkeeping::Clear)
    }

    /// Rotate the retained session after an LCM fold while keeping the exact
    /// stable-prefix anchor for the next request. This is a route transition,
    /// not a prompt-cache reset: the changed suffix is new, the prefix is not.
    pub(crate) fn retire_higgs_session_preserving_prefix(
        &self,
        session_key: &str,
        anchor: StablePromptAnchor,
    ) -> u64 {
        self.retire_higgs_session_inner(session_key, PromptCacheBookkeeping::Preserve(anchor))
    }

    fn retire_higgs_session_inner(
        &self,
        session_key: &str,
        bookkeeping: PromptCacheBookkeeping,
    ) -> u64 {
        let _transition = self.prompt_cache_transition.lock();
        let mut sessions = self.higgs_sessions.lock();
        match &bookkeeping {
            PromptCacheBookkeeping::Clear => {
                self.prompt_fingerprints.lock().remove(session_key);
                self.prompt_tool_hashes.lock().remove(session_key);
                self.prompt_cache_watermark.lock().remove(session_key);
            }
            PromptCacheBookkeeping::Preserve(anchor) => {
                self.prompt_fingerprints
                    .lock()
                    .insert(session_key.to_string(), anchor.fingerprint.clone());
                self.prompt_tool_hashes
                    .lock()
                    .insert(session_key.to_string(), anchor.tool_hash);
                self.prompt_cache_watermark
                    .lock()
                    .insert(session_key.to_string(), anchor.watermark);
            }
        }
        let state = sessions.entry(session_key.to_string()).or_default();
        if let Some(active_id) = state.active_id.take() {
            Self::queue_higgs_session_drop(state, active_id);
            // Eager reclaim: POST the drop to higgs now instead of waiting for
            // the next chat request, so the retired session's resident KV
            // frees before the next (smaller) prompt prefills. Best-effort —
            // the piggybacked drop fields remain the durable fallback, and a
            // session still in flight reports dropped=false and is covered by
            // that fallback.
            let flush = self.higgs_drop_flusher.lock().clone();
            if let Some(flush) = flush {
                flush(active_id);
            }
        }

        state.epoch = state.epoch.saturating_add(1);
        state.preserved_prefix_epoch = match bookkeeping {
            PromptCacheBookkeeping::Clear => None,
            PromptCacheBookkeeping::Preserve(_) => Some(state.epoch),
        };
        state.epoch
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
}

/// Combined handle: cheap to clone (two pointer bumps).
///
/// `core` is swapped on `/local` and `/model`. `counters` persists forever.
#[derive(Clone)]
pub struct AgentHandle {
    core: Arc<parking_lot::RwLock<Arc<SwappableCore>>>,
    pub counters: Arc<RuntimeCounters>,
    /// Live Higgs capacity snapshot; persists across core swaps exactly like
    /// `counters`, so a model switch invalidates it without losing the
    /// runtime itself.
    pub capacity: Arc<crate::agent::capacity::CapacityRuntime>,
}

impl AgentHandle {
    /// Create a new handle from a swappable core and runtime counters.
    pub fn new(core: SwappableCore, counters: Arc<RuntimeCounters>) -> Self {
        Self {
            core: Arc::new(parking_lot::RwLock::new(Arc::new(core))),
            counters,
            capacity: crate::agent::capacity::CapacityRuntime::shared(),
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
    /// Cua driver (local desktop computer-use) tool settings.
    pub cua: crate::config::schema::CuaToolConfig,
}

/// Build a `SwappableCore` from the given config.
///
/// Called once at startup and again for every `/local` or `/model` toggle.
/// Resolves provider selection and memory config.
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
        trio_config,
        model_capabilities_overrides,
        reasoning_config,
        tool_heartbeat_secs,
        health_check_timeout_secs,
        adaptive_tokens,
        sessions_db_path,
        code_execution,
        python_kernel,
        cua,
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
    let (memory_provider, memory_model) =
        resolve_memory_provider(&mode, &memory_config, &model, &provider);

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
        cua,
        memory_enabled: memory_config.enabled,
        memory_provider,
        memory_model,
        reflection_threshold: memory_config.reflection_threshold,
        memory_file_max_words: memory_config.memory_file_max_words,
        mode,
        lane,
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
///  3. Local default: the main provider.
///
/// This selection is reflection-only. LCM is constructed directly from the
/// foreground provider/model and therefore cannot acquire a second context
/// ceiling or endpoint.
fn resolve_memory_provider(
    mode: &RuntimeMode,
    memory_config: &MemoryConfig,
    model: &str,
    provider: &Arc<dyn LLMProvider>,
) -> (Arc<dyn LLMProvider>, String) {
    match mode {
        RuntimeMode::Local { .. } => {
            let mem_model = if memory_config.model.is_empty() {
                model.to_string()
            } else {
                memory_config.model.clone()
            };
            let mem_provider: Arc<dyn LLMProvider> =
                if let Some(ref mem_provider_cfg) = memory_config.provider {
                    crate::providers::factory::from_provider_config_for_model_with_default_base(
                        mem_provider_cfg,
                        Some(&mem_model),
                        provider.get_api_base(),
                    )
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
// Background compaction helpers
// ---------------------------------------------------------------------------

/// Pending compaction result ready to be swapped into the conversation.
pub(crate) struct PendingCompaction {
    pub result: crate::agent::compaction::CompactionResult,
    /// Exact live array that LCM compacted. Its leading system/developer prefix
    /// is non-durable and may change; the durable conversation bytes may not.
    pub snapshot: Vec<Value>,
}

pub(crate) fn prompt_prefix_len(messages: &[Value]) -> usize {
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
}

/// Swap compacted messages into the live conversation, preserving
/// messages added after the compaction snapshot was taken. Returns the
/// swapped log, or `None` when the live conversation no longer matches the
/// compaction snapshot's prefix.
pub(crate) fn apply_compaction_result(
    messages: &[Value],
    pending: PendingCompaction,
) -> Option<Vec<Value>> {
    let live_conversation_watermark = pending.live_conversation_watermark(messages)?;

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
    Some(swapped)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fallback_drop_waits_for_the_final_reservation_of_that_id() {
        let counters = Arc::new(RuntimeCounters::new(32_768));
        let session_key = "cli:deferred-in-flight-drop";
        let durable_session_id = "sqlite:deferred-in-flight-drop";
        let first_x = counters.reserve_higgs_session_request(session_key, durable_session_id, 0);
        let second_x = counters.reserve_higgs_session_request(session_key, durable_session_id, 0);
        let x = first_x.active_id();
        assert_eq!(second_x.active_id(), x);

        counters.retire_higgs_session(session_key);

        let y_while_both_x_live =
            counters.reserve_higgs_session_request(session_key, durable_session_id, 0);
        assert_ne!(y_while_both_x_live.active_id(), x);
        assert!(!y_while_both_x_live.drop_ids().contains(&x));

        drop(first_x);
        let y_while_one_x_live =
            counters.reserve_higgs_session_request(session_key, durable_session_id, 0);
        assert!(!y_while_one_x_live.drop_ids().contains(&x));

        drop(second_x);
        let next_request =
            counters.reserve_higgs_session_request(session_key, durable_session_id, 0);
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
    fn fresh_session_seeds_once_then_requires_exact_continuation() {
        let counters = Arc::new(RuntimeCounters::new(32_768));
        let first = counters.reserve_higgs_session_request("cli:seed", "sqlite:seed", 24_576);
        assert_eq!(first.control().reuse_policy, HiggsSessionReusePolicy::Seed);
        first.publish_exact();
        drop(first);

        let second = counters.reserve_higgs_session_request("cli:seed", "sqlite:seed", 24_576);
        assert_eq!(
            second.control().reuse_policy,
            HiggsSessionReusePolicy::RequireContinuation
        );
    }

    #[test]
    fn resumed_process_restores_required_continuation_hint_once() {
        let counters = Arc::new(RuntimeCounters::new(32_768));
        assert!(counters.restore_higgs_publication_hint("cli:resume", "sqlite:resume",));

        let resumed = counters.reserve_higgs_session_request("cli:resume", "sqlite:resume", 24_576);
        assert_eq!(
            resumed.control().reuse_policy,
            HiggsSessionReusePolicy::RequireContinuation
        );

        counters.invalidate_prompt_cache("cli:resume", true);
        assert!(!counters.restore_higgs_publication_hint("cli:resume", "sqlite:resume",));
        let rotated = counters.reserve_higgs_session_request("cli:resume", "sqlite:resume", 24_576);
        assert_eq!(
            rotated.control().reuse_policy,
            HiggsSessionReusePolicy::Seed
        );
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
        };

        let swapped = apply_compaction_result(&live, pending).expect("swap must apply");
        assert_eq!(swapped[0]["content"], "current system");
        assert_eq!(swapped[1]["content"], "current working memory");
        assert_eq!(swapped[2], summary);
        assert_eq!(swapped[3]["content"], "new result");
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
        let live = vec![
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
        };

        let swapped = apply_compaction_result(&live, pending).expect("swap must apply");
        assert_eq!(swapped[0]["content"], "current system");
        assert_eq!(swapped[1]["content"], "new working memory");
        assert_eq!(swapped[2], summary);
        assert_eq!(swapped[3]["content"], "new tail");
    }

    #[test]
    fn compaction_swap_rejects_a_rewritten_snapshot_without_mutation() {
        let snapshot = vec![
            serde_json::json!({"role": "system", "content": "system"}),
            serde_json::json!({"role": "user", "content": "old", "_db_id": 1}),
        ];
        let live = vec![
            serde_json::json!({"role": "system", "content": "current system"}),
            serde_json::json!({"role": "developer", "content": "current working memory"}),
            serde_json::json!({"role": "user", "content": "rewritten", "_db_id": 2}),
        ];
        let pending = PendingCompaction {
            result: crate::agent::compaction::CompactionResult {
                messages: vec![
                    snapshot[0].clone(),
                    serde_json::json!({"role": "user", "content": "summary", "_lcm_summary": true}),
                ],
            },
            snapshot,
        };

        // With the input taken by shared reference, rejection cannot mutate.
        assert!(apply_compaction_result(&live, pending).is_none());
    }

    /// A sanctioned reset must be attributable exactly once, per session.
    ///
    /// Once, because the reason is consumed by the next provider call — if it
    /// lingered, every later call in the session would re-report the same
    /// re-prefill and the ledger would overcount. Per session, because one
    /// session's `/clear` must not be blamed on another's next turn.
    #[test]
    fn cache_reset_reason_is_taken_once_per_session() {
        let counters = RuntimeCounters::new(16384);

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
    fn cache_metrics_are_aggregated_by_logical_session() {
        let counters = RuntimeCounters::new(16384);

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
        let counters = RuntimeCounters::new(16384);
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
        for mode in [ToolPresentationMode::Textual, ToolPresentationMode::Native] {
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
        let counters = RuntimeCounters::new(16384);
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
    fn lcm_rotation_keeps_prefix_anchor_without_sanctioned_reset() {
        let counters = Arc::new(RuntimeCounters::new(16_384));
        let session = "cli:lcm-prefix-anchor";
        let durable = "sqlite:lcm-prefix-anchor";
        let fingerprint = crate::agent::prompt_fingerprint::fingerprint(&[
            serde_json::json!({"role": "system", "content": "stable"}),
        ]);
        let anchor = StablePromptAnchor {
            fingerprint: fingerprint.clone(),
            watermark: 1,
            tool_hash: 77,
        };

        counters.record_higgs_session_id(session, 10);
        counters
            .prompt_fingerprints
            .lock()
            .insert(session.to_string(), fingerprint);
        counters
            .prompt_cache_watermark
            .lock()
            .insert(session.to_string(), 9);
        counters
            .prompt_tool_hashes
            .lock()
            .insert(session.to_string(), 77);

        let epoch = counters.retire_higgs_session_preserving_prefix(session, anchor.clone());
        assert_eq!(epoch, 1);
        assert_eq!(counters.take_cache_reset(session), None);
        assert_eq!(
            counters.prompt_fingerprints.lock().get(session),
            Some(&anchor.fingerprint)
        );
        assert_eq!(
            counters.prompt_cache_watermark.lock().get(session),
            Some(&1)
        );
        assert_eq!(counters.prompt_tool_hashes.lock().get(session), Some(&77));

        let request = counters.reserve_higgs_session_request(session, durable, 16_000);
        assert_ne!(request.control().active_id, 10);
        assert_eq!(
            counters.prompt_fingerprints.lock().get(session),
            Some(&anchor.fingerprint),
            "intentional route rotation must not turn the next request cold"
        );
        assert_eq!(
            counters.prompt_cache_watermark.lock().get(session),
            Some(&1)
        );
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
        let counters = RuntimeCounters::new(16384);
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
        let counters = RuntimeCounters::new(16384);
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
        counters
            .prompt_head_hashes
            .lock()
            .insert(session.to_string(), 42);
        counters.record_local_artifact_intent(session, 10, true);

        counters.record_higgs_session_id(session, 10);
        assert_eq!(counters.reset_session_prompt_state(session), 1);
        assert!(!counters.prompt_fingerprints.lock().contains_key(session));
        assert!(!counters.prompt_cache_watermark.lock().contains_key(session));
        assert!(!counters.prompt_head_hashes.lock().contains_key(session));
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
        let counters = RuntimeCounters::new(16384);
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
        let counters = RuntimeCounters::new(16384);
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
        let counters = RuntimeCounters::new(16384);
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

#[cfg(test)]
mod higgs_drop_flusher_tests {
    use super::*;

    #[test]
    fn drop_retirement_fires_the_eager_flush_hook() {
        let counters = RuntimeCounters::new(32_768);
        let flushed = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let flushed_clone = std::sync::Arc::clone(&flushed);
        counters.set_higgs_drop_flusher(std::sync::Arc::new(move |session_id| {
            flushed_clone.lock().unwrap().push(session_id);
        }));

        // A live retained session (as the wire-lease flow would have left it).
        counters.record_higgs_session_id("flush-hook", 777);

        let epoch = counters.retire_higgs_session("flush-hook");
        assert!(epoch >= 1);
        assert_eq!(
            flushed.lock().unwrap().as_slice(),
            &[777],
            "the rotation must eagerly flush the retired session id"
        );
    }

    #[test]
    fn drop_retirement_without_flusher_is_a_clean_noop() {
        let counters = RuntimeCounters::new(32_768);
        counters.record_higgs_session_id("no-flush", 555);
        let epoch = counters.retire_higgs_session("no-flush");
        assert!(epoch >= 1, "rotation proceeds with no flusher wired");
    }
}
