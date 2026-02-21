//! Core types shared across the agent system.
//!
//! Extracted from `agent_loop.rs` to reduce file size and improve modularity.
//! Contains: SwappableCore, RuntimeCounters, AgentHandle, build helpers, and
//! compaction utilities.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64};
use std::sync::Arc;

use serde_json::Value;

use crate::agent::agent_profiles;
use crate::agent::compaction::ContextCompactor;
use crate::agent::context::ContextBuilder;
use crate::agent::learning::LearningStore;
use crate::agent::token_budget::TokenBudget;
use crate::agent::working_memory::WorkingMemoryStore;
use crate::config::schema::{AntiDriftConfig, MemoryConfig, ProvenanceConfig, ToolDelegationConfig, TrioConfig};
use crate::providers::base::LLMProvider;
use crate::session::manager::SessionManager;

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
    pub max_tokens: u32,
    pub temperature: f64,
    pub context: ContextBuilder,
    pub sessions: SessionManager,
    pub token_budget: TokenBudget,
    pub compactor: ContextCompactor,
    pub learning: LearningStore,
    pub working_memory: WorkingMemoryStore,
    pub working_memory_budget: usize,
    pub brave_api_key: Option<String>,
    pub exec_timeout: u64,
    pub restrict_to_workspace: bool,
    pub memory_enabled: bool,
    pub memory_provider: Arc<dyn LLMProvider>,
    pub memory_model: String,
    pub reflection_threshold: usize,
    pub is_local: bool,
    pub anti_drift: AntiDriftConfig,
    pub main_no_think: bool,
    pub tool_runner_provider: Option<Arc<dyn LLMProvider>>,
    pub tool_runner_model: Option<String>,
    pub router_provider: Option<Arc<dyn LLMProvider>>,
    pub router_model: Option<String>,
    pub router_no_think: bool,
    pub router_temperature: f64,
    #[allow(dead_code)]
    pub router_top_p: f64,
    pub specialist_provider: Option<Arc<dyn LLMProvider>>,
    pub specialist_model: Option<String>,
    pub tool_delegation_config: ToolDelegationConfig,
    pub provenance_config: ProvenanceConfig,
    pub max_tool_result_chars: usize,
    pub session_complete_after_secs: u64,
    pub max_message_age_turns: usize,
    pub max_history_turns: usize,
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
    pub last_tools_called: std::sync::Mutex<Vec<String>>,
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
    /// When true, ThinkingDelta tokens are NOT sent to delta_tx (suppressed
    /// from TTS). Auto-set when voice mode is active; toggled by `/nothink`.
    pub suppress_thinking_in_tts: AtomicBool,
    /// Set to true while an LLM call is in flight. The health watchdog reads
    /// this to skip health checks during inference (avoiding false "unhealthy"
    /// restarts when the server is busy processing a large prompt).
    /// Wrapped in Arc so the watchdog can hold a cheap clone without needing
    /// the full RuntimeCounters.
    pub inference_active: Arc<AtomicBool>,
}

impl RuntimeCounters {
    pub fn new(max_context_tokens: usize) -> Self {
        Self {
            learning_turn_counter: AtomicU64::new(0),
            last_context_used: AtomicU64::new(0),
            last_context_max: AtomicU64::new(max_context_tokens as u64),
            last_message_count: AtomicU64::new(0),
            last_working_memory_tokens: AtomicU64::new(0),
            last_tools_called: std::sync::Mutex::new(Vec::new()),
            delegation_healthy: AtomicBool::new(true),
            delegation_retry_counter: AtomicU64::new(0),
            thinking_budget: AtomicU32::new(0),
            long_mode_turns: AtomicU32::new(0),
            last_actual_prompt_tokens: AtomicU64::new(0),
            last_actual_completion_tokens: AtomicU64::new(0),
            last_estimated_prompt_tokens: AtomicU64::new(0),
            suppress_thinking_in_tts: AtomicBool::new(false),
            inference_active: Arc::new(AtomicBool::new(false)),
        }
    }
}

/// Combined handle: cheap to clone (two pointer bumps).
///
/// `core` is swapped on `/local` and `/model`. `counters` persists forever.
#[derive(Clone)]
pub struct AgentHandle {
    core: Arc<std::sync::RwLock<Arc<SwappableCore>>>,
    pub counters: Arc<RuntimeCounters>,
}

impl AgentHandle {
    /// Create a new handle from a swappable core and runtime counters.
    pub fn new(core: SwappableCore, counters: Arc<RuntimeCounters>) -> Self {
        Self {
            core: Arc::new(std::sync::RwLock::new(Arc::new(core))),
            counters,
        }
    }

    /// Snapshot the current swappable core (cheap Arc clone under brief read lock).
    pub fn swappable(&self) -> Arc<SwappableCore> {
        self.core.read().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Replace the swappable core (write lock). Counters are untouched.
    pub fn swap_core(&self, new_core: SwappableCore) {
        *self.core.write().unwrap_or_else(|e| e.into_inner()) = Arc::new(new_core);
    }
}

// Backward-compatibility alias during migration.
pub type SharedCoreHandle = AgentHandle;

/// Local chat templates often reject mid-conversation `system` messages.
/// In local mode, provenance reminders must be emitted as `user` role.
pub(crate) fn provenance_warning_role(is_local: bool) -> &'static str {
    if is_local {
        "user"
    } else {
        "system"
    }
}

pub(crate) fn is_small_local_model(model: &str) -> bool {
    let m = model.to_ascii_lowercase();
    m.contains("nanbeige")
        || m.contains("functiongemma")
        || m.contains("ministral-3")
        || m.contains("qwen3-1.7b")
        || m.contains("3b")
        || m.contains("1.7b")
}

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
    pub max_tokens: u32,
    pub temperature: f64,
    pub max_context_tokens: usize,
    pub brave_api_key: Option<String>,
    pub exec_timeout: u64,
    pub restrict_to_workspace: bool,
    pub memory_config: MemoryConfig,
    pub is_local: bool,
    pub compaction_provider: Option<Arc<dyn LLMProvider>>,
    pub tool_delegation: ToolDelegationConfig,
    pub provenance: ProvenanceConfig,
    pub max_tool_result_chars: usize,
    pub delegation_provider: Option<Arc<dyn LLMProvider>>,
    pub specialist_provider: Option<Arc<dyn LLMProvider>>,
    pub trio_config: TrioConfig,
}

/// Build a `SwappableCore` from the given config.
///
/// Called once at startup and again for every `/local` or `/model` toggle.
/// Resolves provider selection, memory config, tool delegation, and router setup.
pub fn build_swappable_core(cfg: SwappableCoreConfig) -> SwappableCore {
    let SwappableCoreConfig {
        provider,
        workspace,
        model,
        max_iterations,
        max_tokens,
        temperature,
        max_context_tokens,
        brave_api_key,
        exec_timeout,
        restrict_to_workspace,
        memory_config,
        is_local,
        compaction_provider,
        tool_delegation,
        provenance,
        max_tool_result_chars,
        delegation_provider,
        specialist_provider,
        trio_config,
    } = cfg;
    let router_provider = delegation_provider.clone();
    let mut context = if is_local {
        ContextBuilder::new_lite(&workspace)
    } else {
        ContextBuilder::new(&workspace)
    };
    // Scale prompt budgets proportionally to the model's context window.
    // Without this, a 1M-context model gets the same tiny fixed caps as a 16K model.
    if is_local {
        context.set_lite_mode(max_context_tokens);
    } else {
        context.scale_budgets(max_context_tokens);
    }
    context.model_name = model.clone();
    // Inject provenance verification rules when enabled.
    if provenance.enabled && provenance.system_prompt_rules {
        context.provenance_enabled = true;
    }
    // RLM lazy skills: skills loaded as summaries, fetched on demand.
    context.lazy_skills = memory_config.lazy_skills;
    // Wire subagent profiles into the system prompt so the model knows
    // what agents exist and when to delegate instead of doing everything itself.
    let profiles = agent_profiles::load_profiles(&workspace);
    context.agent_profiles = agent_profiles::profiles_summary(&profiles);
    let sessions = SessionManager::new(&workspace);

    // Resolve memory provider + model.
    //
    // Priority:
    //   1. Explicit `memory.model` / `memory.provider` from config.json
    //   2. Cloud default: "haiku" (cheap, fast summarisation)
    //   3. Local default: specialist provider from trio (if available),
    //      then dedicated compaction provider, then main provider.
    let (memory_provider, memory_model): (Arc<dyn LLMProvider>, String) = if is_local {
        // --- Local mode ---
        let mem_model = if !memory_config.model.is_empty() {
            memory_config.model.clone()
        } else if let Some(ref sp) = specialist_provider {
            // Trio specialist is ideal for summarisation tasks.
            sp.get_default_model().to_string()
        } else {
            model.clone()
        };
        let mem_provider: Arc<dyn LLMProvider> = if let Some(ref mem_provider_cfg) =
            memory_config.provider
        {
            crate::providers::factory::from_provider_config(mem_provider_cfg)
        } else if let Some(ref sp) = specialist_provider {
            // Reuse trio specialist provider when no explicit memory provider.
            sp.clone()
        } else if let Some(cp) = compaction_provider {
            cp
        } else {
            provider.clone()
        };
        (mem_provider, mem_model)
    } else {
        // --- Cloud mode ---
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
            model.clone()
        };
        let mem_provider: Arc<dyn LLMProvider> =
            if let Some(ref mem_provider_cfg) = memory_config.provider {
                crate::providers::factory::from_provider_config(mem_provider_cfg)
            } else {
                provider.clone()
            };
        (mem_provider, mem_model)
    };

    let token_budget = TokenBudget::new(max_context_tokens, max_tokens as usize);
    let compaction_ctx_size = if memory_config.compaction_model_context_size > 0 {
        memory_config.compaction_model_context_size
    } else {
        max_context_tokens
    };
    let compactor = ContextCompactor::new(
        memory_provider.clone(),
        memory_model.clone(),
        compaction_ctx_size,
    )
    .with_thresholds(
        memory_config.compaction_threshold_percent,
        memory_config.compaction_threshold_tokens,
    );
    let learning = LearningStore::new(&workspace);
    let working_memory = WorkingMemoryStore::new(&workspace);

    // Build tool runner provider if delegation is enabled.
    let (tool_runner_provider, tool_runner_model) = if tool_delegation.enabled {
        let is_auto_local = delegation_provider.is_some();
        let tr_provider: Arc<dyn LLMProvider> = if let Some(dp) = delegation_provider {
            dp // Auto-spawned local delegation server
        } else if let Some(ref tr_cfg) = tool_delegation.provider {
            crate::providers::factory::from_provider_config(tr_cfg)
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
        max_tokens,
        temperature,
        context,
        sessions,
        token_budget,
        compactor,
        learning,
        working_memory,
        // Scale working memory like other budgets. If the user left it at
        // the default (600), apply proportional scaling; otherwise respect their override.
        working_memory_budget: if memory_config.working_memory_budget == 600 {
            (max_context_tokens * 15 / 1000).clamp(300, 15_000) // 1.5%
        } else {
            memory_config.working_memory_budget
        },
        brave_api_key,
        exec_timeout,
        restrict_to_workspace,
        memory_enabled: memory_config.enabled,
        memory_provider,
        memory_model,
        reflection_threshold: memory_config.reflection_threshold,
        is_local,
        anti_drift: trio_config.anti_drift.clone(),
        main_no_think: trio_config.main_no_think,
        tool_runner_provider,
        tool_runner_model,
        router_provider,
        router_model,
        router_no_think: trio_config.router_no_think,
        router_temperature: trio_config.router_temperature,
        router_top_p: trio_config.router_top_p,
        specialist_provider,
        specialist_model,
        tool_delegation_config: tool_delegation,
        provenance_config: provenance,
        max_tool_result_chars,
        session_complete_after_secs: memory_config.session_complete_after_secs,
        max_message_age_turns: memory_config.max_message_age_turns,
        max_history_turns: memory_config.max_history_turns,
    }
}

// ---------------------------------------------------------------------------
// History limit scaling
// ---------------------------------------------------------------------------

/// Scale history message count with context window size.
///
/// Small models (16K) can't afford 100 messages of history — that alone
/// can eat 40%+ of the context. Scale linearly: ~20 msgs at 16K, ~100 at
/// 128K, clamped to [6, 100].
pub(crate) fn history_limit(max_context_tokens: usize) -> usize {
    // Real-world average is ~150 tokens per message (user queries + assistant
    // responses). Reserve at most 30% of context for history.
    let max_history_tokens = max_context_tokens * 3 / 10;
    let limit = max_history_tokens / 150;
    limit.clamp(6, 100)
}

// ---------------------------------------------------------------------------
// Background compaction helpers
// ---------------------------------------------------------------------------

/// Pending compaction result ready to be swapped into the conversation.
pub(crate) struct PendingCompaction {
    pub result: crate::agent::compaction::CompactionResult,
    pub watermark: usize, // messages.len() when compaction was spawned
}

/// Swap compacted messages into the live conversation, preserving
/// messages added after the compaction snapshot was taken.
pub(crate) fn apply_compaction_result(messages: &mut Vec<Value>, pending: PendingCompaction) {
    let new_messages: Vec<Value> = if pending.watermark < messages.len() {
        messages[pending.watermark..].to_vec()
    } else {
        vec![]
    };
    let mut swapped = Vec::with_capacity(1 + pending.result.messages.len() + new_messages.len());
    swapped.push(messages[0].clone()); // fresh system msg
    if pending.result.messages.len() > 1 {
        swapped.extend_from_slice(&pending.result.messages[1..]); // skip stale system msg
    }
    swapped.extend(new_messages);
    *messages = swapped;
}

/// Append a suffix to the first (system) message's content.
pub(crate) fn append_to_system_prompt(messages: &mut [Value], suffix: &str) {
    if let Some(sys) = messages
        .first()
        .and_then(|m| m["content"].as_str())
        .map(|s| s.to_string())
    {
        messages[0]["content"] = Value::String(format!("{}{}", sys, suffix));
    }
}
