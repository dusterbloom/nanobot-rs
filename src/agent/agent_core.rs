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
    /// Per-session hash of the tool-definition array sent to the provider.
    /// The message fingerprint deliberately excludes tool schemas, so this
    /// catches the case where messages are append-only but the rendered token
    /// stream still diverges because the tool block (rendered at the prompt
    /// head by chat templates) changed — busting the prefix cache invisibly.
    pub prompt_tool_hashes: parking_lot::Mutex<std::collections::HashMap<String, u64>>,
    /// Per-session prefix-cache watermark: the number of leading messages
    /// already sent (hence warm on the inference server). Mid-turn cleanup is
    /// frozen below this index so the rendered prompt stays an append-only
    /// extension of the last send. Re-anchored on every send. See
    /// `agent::prefix_guard`.
    pub prompt_cache_watermark: parking_lot::Mutex<std::collections::HashMap<String, usize>>,
    /// Per-session prompt epoch. Bumped by `/clear` and model switches so a
    /// resident local server cannot keep continuing from a stale KV cache when
    /// the next user prompt happens to be a prefix of the old conversation.
    pub prompt_session_epoch: parking_lot::Mutex<std::collections::HashMap<String, u64>>,
    /// Higgs retained-session id currently associated with each channel
    /// session. Keeping the concrete id here lets a reset queue it before the
    /// durable SQLite session id rolls over.
    active_higgs_session_ids: parking_lot::Mutex<std::collections::HashMap<String, u64>>,
    /// One-shot concrete Higgs retained-session ids to drop after `/clear` or
    /// model switches. These must not be re-derived from a later SQLite
    /// session id because clears can roll that id over before the next request.
    pub pending_higgs_session_drop_ids:
        parking_lot::Mutex<std::collections::HashMap<String, Vec<u64>>>,
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
            prompt_tool_hashes: parking_lot::Mutex::new(std::collections::HashMap::new()),
            prompt_cache_watermark: parking_lot::Mutex::new(std::collections::HashMap::new()),
            prompt_session_epoch: parking_lot::Mutex::new(std::collections::HashMap::new()),
            active_higgs_session_ids: parking_lot::Mutex::new(std::collections::HashMap::new()),
            pending_higgs_session_drop_ids: parking_lot::Mutex::new(
                std::collections::HashMap::new(),
            ),
            local_artifact_intent: parking_lot::Mutex::new(std::collections::HashMap::new()),
            lcm_compaction_count: AtomicU64::new(0),
            lcm_tokens_before: AtomicU64::new(0),
            lcm_tokens_after: AtomicU64::new(0),
            lcm_last_compaction_ms: AtomicU64::new(0),
        }
    }

    /// Reset all prompt-cache bookkeeping for a session and advance its prompt epoch.
    ///
    /// The epoch is rendered into the next prompt as a tiny stable marker. This
    /// forces local resident servers to treat post-clear/post-switch prompts as
    /// a fresh prefix even when the user starts with identical text like `hi`.
    pub fn reset_session_prompt_state(&self, session_key: &str) -> u64 {
        self.prompt_fingerprints.lock().remove(session_key);
        self.prompt_tool_hashes.lock().remove(session_key);
        self.prompt_cache_watermark.lock().remove(session_key);
        self.clear_local_artifact_intent(session_key);

        let mut epochs = self.prompt_session_epoch.lock();
        if let Some(drop_id) = self.active_higgs_session_ids.lock().remove(session_key) {
            let mut pending_drops = self.pending_higgs_session_drop_ids.lock();
            let drops = pending_drops.entry(session_key.to_string()).or_default();
            if !drops.contains(&drop_id) {
                drops.push(drop_id);
            }
        }
        let next = epochs
            .get(session_key)
            .copied()
            .unwrap_or(0)
            .saturating_add(1);
        epochs.insert(session_key.to_string(), next);
        next
    }

    pub fn session_prompt_epoch(&self, session_key: &str) -> u64 {
        self.prompt_session_epoch
            .lock()
            .get(session_key)
            .copied()
            .unwrap_or(0)
    }

    /// Clear only the local prompt-cache bookkeeping (fingerprint + watermark)
    /// for a session, without touching the retained higgs session id or epoch.
    /// Used when a rewrite happens on a backend that does not support retained
    /// sessions, or as the local-clear half of [`invalidate_prompt_cache`].
    pub fn clear_local_prompt_cache(&self, session_key: &str) -> bool {
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
            self.reset_session_prompt_state(session_key);
            true
        } else {
            self.clear_local_prompt_cache(session_key);
            false
        }
    }

    pub fn record_higgs_session_id(&self, session_key: &str, session_id: u64) {
        self.active_higgs_session_ids
            .lock()
            .insert(session_key.to_string(), session_id);
    }

    pub fn pending_higgs_session_drop_ids(&self, session_key: &str) -> Vec<u64> {
        self.pending_higgs_session_drop_ids
            .lock()
            .get(session_key)
            .cloned()
            .unwrap_or_default()
    }

    pub fn clear_pending_higgs_session_drop_id(&self, session_key: &str, drop_id: u64) -> bool {
        let mut drops = self.pending_higgs_session_drop_ids.lock();
        let Some(drop_ids) = drops.get_mut(session_key) else {
            return false;
        };
        let Some(pos) = drop_ids.iter().position(|candidate| *candidate == drop_id) else {
            return false;
        };
        drop_ids.remove(pos);
        if drop_ids.is_empty() {
            drops.remove(session_key);
        }
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
    /// Exact live array that LCM compacted. Installation is allowed only when
    /// the current array still starts with these bytes; that makes appended
    /// tool traffic safe without guessing from a stale numeric watermark.
    pub snapshot: Vec<Value>,
}

impl PendingCompaction {
    pub(crate) fn watermark(&self) -> usize {
        self.snapshot.len()
    }

    pub(crate) fn matches_snapshot_prefix(&self, messages: &[Value]) -> bool {
        messages.starts_with(&self.snapshot)
    }
}

/// Swap compacted messages into the live conversation, preserving
/// messages added after the compaction snapshot was taken.
pub(crate) fn apply_compaction_result(
    messages: &mut Vec<Value>,
    pending: PendingCompaction,
) -> bool {
    if !pending.matches_snapshot_prefix(messages) {
        return false;
    }

    let new_messages = messages[pending.watermark()..].to_vec();
    // The result carries the complete snapshot prompt prefix. Preserve the
    // current copy of that non-durable prefix, then append every compacted LCM
    // conversation entry. In particular, result[0] is not assumed to be a
    // system message: LCM's active context itself deliberately has no system.
    let prompt_prefix_len = |values: &[Value]| {
        values
            .iter()
            .take_while(|message| {
                matches!(
                    message.get("role").and_then(Value::as_str),
                    Some("system" | "developer")
                )
            })
            .count()
    };
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
            serde_json::json!({"role": "developer", "content": "developer"}),
            serde_json::json!({"role": "user", "content": "old", "_db_id": 1}),
        ];
        let mut live = snapshot.clone();
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

        assert!(apply_compaction_result(&mut live, pending));
        assert_eq!(live[0]["role"], "system");
        assert_eq!(live[1]["role"], "developer");
        assert_eq!(live[2], summary);
        assert_eq!(live[3]["content"], "new result");
    }

    #[test]
    fn compaction_swap_rejects_a_rewritten_snapshot_without_mutation() {
        let snapshot = vec![
            serde_json::json!({"role": "system", "content": "system"}),
            serde_json::json!({"role": "user", "content": "old", "_db_id": 1}),
        ];
        let mut live = vec![
            snapshot[0].clone(),
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
        };

        assert!(!apply_compaction_result(&mut live, pending));
        assert_eq!(live, before);
    }

    #[test]
    fn test_trio_state_default_is_standalone() {
        let counters = RuntimeCounters::new_with_config(16384, &CircuitBreakerConfig::default());
        assert_eq!(counters.get_trio_state(), TrioState::Standalone);
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
        assert!(
            counters.active_higgs_session_ids.lock().contains_key(s),
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
