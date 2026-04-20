//! Typed replacement for `SwappableCore::is_local`.
//!
//! Derivation methods on this enum centralize the 33 branch points cataloged
//! in `.planning/phases/09-runtime-mode-spine/09-RESEARCH.md` §1. This module
//! owns runtime policy; callers pass explicit inputs (`max_context`,
//! `max_tokens`, `configured_iterations`) and the enum dispatches via `match`.
//!
//! **Parallel rollout (Wave 1):** this type is introduced alongside the
//! existing `is_local: bool` field on `SwappableCore`. No production callsite
//! is migrated yet; Wave 2 migrates derivations, Wave 3 removes the field.
//!
//! The two-variant shape is locked by 09-CONTEXT.md: `Cloud` is a unit
//! variant, `Local` wraps `Arc<ModelCapabilities>`. Cluster / remote
//! LM-Studio / private-IP OpenAI-compat endpoints all route through `Local`
//! — the capabilities carry the differentiation.

use std::sync::Arc;

use crate::agent::model_capabilities::{ModelCapabilities, ModelSizeClass};
use crate::agent::protocol::LocalReplayMode;
use crate::config::schema::LocalToolMode;

/// Runtime descriptor. Every `is_local` branch in the codebase is expected
/// to collapse into one method call on this enum during Waves 2–3.
#[derive(Debug, Clone)]
pub enum RuntimeMode {
    /// Cloud / remote managed API (Anthropic, OpenAI, OpenRouter, ...).
    Cloud,
    /// Any locally-reachable backend (Higgs sidecar, LM Studio, vLLM,
    /// cluster peer). Capabilities carry the finer-grained differentiation.
    Local {
        /// Cheap handle to the resolved model capabilities. Shared via `Arc`
        /// so `RuntimeMode` stays `Clone` without deep-copying the struct.
        caps: Arc<ModelCapabilities>,
    },
}

/// Policy for capping the model's `thinking` budget.
///
/// This is a new local enum (not an existing type) because the current
/// codebase expresses the policy as a scattered `if size_class == Small` +
/// magic-number clamp; isolating it here satisfies G1 (BOOL → ENUM) and
/// G5 (BRANCH → TYPE) from `CLAUDE.md`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkingCapPolicy {
    /// No hard cap — the configured budget is used verbatim.
    Uncapped,
    /// Cap at `u32` tokens (used by small local models whose scratchpad
    /// throws away anything past a short budget).
    Hard(u32),
}

/// Budget shares (as integer percentages) for the four system-prompt
/// sub-budgets. Matches the ratios in `context.rs::scale_budgets` /
/// `context.rs::set_lite_mode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BudgetShares {
    pub history_pct: u32,
    pub memory_pct: u32,
    pub working_memory_pct: u32,
    pub output_pct: u32,
}

/// Small constants live next to the one method that uses them (G4).
const LOCAL_CONTEXT_CAP_LITE: usize = 800;
const LOCAL_MAX_ITERATIONS_CLAMP: u32 = 15;
const CLOUD_MAX_ITERATIONS_SCALE_DIVISOR: usize = 4000;
const CLOUD_MAX_ITERATIONS_CEILING: u32 = 50;
const SMALL_LOCAL_THINKING_BUDGET: u32 = 2048;

impl RuntimeMode {
    // ------------------------------------------------------------------
    // Constructors
    // ------------------------------------------------------------------

    /// Resolve a `RuntimeMode` from an optional capabilities handle.
    ///
    /// `Some(caps)` → `Local { caps }` (any local backend is selected).
    /// `None`       → `Cloud` (no local backend configured).
    ///
    /// Wave 2 will add a `from_config(...)` constructor that threads the
    /// `Config` + resolved `ModelCapabilities` through in one call. For
    /// Wave 1 this minimal form is enough to drive the invariant tests.
    pub fn from_caps(caps: Option<Arc<ModelCapabilities>>) -> Self {
        match caps {
            Some(caps) => Self::Local { caps },
            None => Self::Cloud,
        }
    }

    /// `true` iff this is a `Local` variant. Provided as a convenience for
    /// readers that still want a bool — Wave 3 removes the remaining
    /// call sites by switching them to `match`.
    pub fn is_local(&self) -> bool {
        matches!(self, Self::Local { .. })
    }

    // ------------------------------------------------------------------
    // Derivation methods (the critical-branch table)
    // ------------------------------------------------------------------

    /// Return the effective bootstrap / working-memory context cap.
    ///
    /// - Cloud: `max` (uncapped — cloud models have 100K+ context).
    /// - Local: a fixed `LOCAL_CONTEXT_CAP_LITE` (the "lite" budget that
    ///   `ContextBuilder::set_lite_mode` clamps bootstrap+memory to).
    pub fn context_cap(&self, max: usize) -> usize {
        match self {
            Self::Cloud => max,
            Self::Local { .. } => LOCAL_CONTEXT_CAP_LITE,
        }
    }

    /// Return the effective response-reserve budget.
    ///
    /// - Cloud: `max_tokens` (no clamp).
    /// - Local: `min(max_tokens, max_ctx / 4)` — the 25 % clamp lifted from
    ///   `agent_core.rs::effective_reserve` preserves some conversation
    ///   space even when the user configured a huge `max_tokens`.
    pub fn reserve_cap(&self, max_tokens: usize, max_ctx: usize) -> usize {
        match self {
            Self::Cloud => max_tokens,
            Self::Local { .. } => max_tokens.min(max_ctx / 4),
        }
    }

    /// Return the effective max tool-iterations value.
    ///
    /// - Cloud: scale `configured` up to a context-dependent ceiling.
    /// - Local: clamp `configured` at `LOCAL_MAX_ITERATIONS_CLAMP`.
    ///
    /// Mirrors `cli::core_builder::effective_max_iterations`.
    pub fn max_iterations(&self, configured: u32, max_ctx: usize) -> u32 {
        match self {
            Self::Cloud => {
                let context_scaled = (max_ctx / CLOUD_MAX_ITERATIONS_SCALE_DIVISOR)
                    .min(CLOUD_MAX_ITERATIONS_CEILING as usize) as u32;
                configured.max(context_scaled)
            }
            Self::Local { .. } => configured.min(LOCAL_MAX_ITERATIONS_CLAMP),
        }
    }

    /// Return the budget-share ratios for the four system-prompt
    /// sub-budgets.
    ///
    /// The `_max_ctx` argument is accepted (but unused) so callers have a
    /// single signature as both variants start to take context-dependent
    /// parameters in later waves.
    pub fn budget_strategy(&self, _max_ctx: usize) -> BudgetShares {
        match self {
            Self::Cloud => BudgetShares {
                history_pct: 2,
                memory_pct: 1,
                working_memory_pct: 4,
                output_pct: 2,
            },
            Self::Local { .. } => BudgetShares {
                history_pct: 2,
                memory_pct: 1,
                working_memory_pct: 2,
                output_pct: 1,
            },
        }
    }

    /// Return the resolved local replay protocol, if any.
    ///
    /// Returns `None` for `Cloud` (cloud APIs use native `tool_calls`, the
    /// `LocalReplayMode` concept does not apply). Returns `Some(mode)` for
    /// `Local`, following the `env > caps > default` priority documented in
    /// 09-CONTEXT.md "Protocol selection".
    pub fn protocol(&self) -> Option<LocalReplayMode> {
        match self {
            Self::Cloud => None,
            Self::Local { caps } => Some(resolve_protocol_from_env_or_caps(caps)),
        }
    }

    /// Return the tool-schema presentation mode.
    ///
    /// Reuses the existing `LocalToolMode` enum from `config::schema` so no
    /// call site has to translate between types during the parallel rollout.
    /// Cloud always gets `Full`; Local's choice is driven by
    /// `caps.tool_calling`.
    pub fn tool_def_mode(&self) -> LocalToolMode {
        match self {
            Self::Cloud => LocalToolMode::Full,
            Self::Local { caps } => {
                if caps.tool_calling {
                    LocalToolMode::Slim
                } else {
                    LocalToolMode::Proxy
                }
            }
        }
    }

    /// `true` iff the anti-drift nudge should be enabled for this runtime.
    ///
    /// Mirrors `agent_shared.rs::ctx.core.is_local && ctx.core.anti_drift.enabled`
    /// — the `is_local` half of that gate is what this method replaces.
    pub fn needs_anti_drift(&self) -> bool {
        match self {
            Self::Cloud => false,
            Self::Local { .. } => true,
        }
    }

    /// Role used for grounding / instruction messages.
    ///
    /// Cloud: `"system"`. Local: `"user"` (local models ignore mid-thread
    /// `system` turns, so the grounding is smuggled through `user`).
    pub fn grounding_role(&self) -> &'static str {
        match self {
            Self::Cloud => "system",
            Self::Local { .. } => "user",
        }
    }

    /// Return the thinking-cap policy for the current runtime.
    ///
    /// - Cloud: `Uncapped`.
    /// - Local + `size_class == Small`: `Hard(SMALL_LOCAL_THINKING_BUDGET)`.
    /// - Local + Medium/Large: `Uncapped`.
    pub fn thinking_cap_policy(&self) -> ThinkingCapPolicy {
        match self {
            Self::Cloud => ThinkingCapPolicy::Uncapped,
            Self::Local { caps } => match caps.size_class {
                ModelSizeClass::Small => ThinkingCapPolicy::Hard(SMALL_LOCAL_THINKING_BUDGET),
                ModelSizeClass::Medium | ModelSizeClass::Large => ThinkingCapPolicy::Uncapped,
            },
        }
    }

    /// `true` iff this runtime needs the local-protocol message
    /// massaging (no mid-thread `system`, no trailing `assistant`, etc.).
    ///
    /// Convenience wrapper over `self.protocol().is_some()`. Kept as a
    /// named method so Wave 3's grep discipline is easy: every
    /// `needs_local_protocol()` call site becomes a `match` on the
    /// dispatching method when the field is finally deleted.
    pub fn needs_local_protocol(&self) -> bool {
        self.protocol().is_some()
    }
}

/// Resolve the `LocalReplayMode` for a `Local` runtime following the
/// documented `env > caps > default` priority. Extracted as a free
/// function so `protocol()` stays under the G4 (40-line) ceiling.
fn resolve_protocol_from_env_or_caps(caps: &ModelCapabilities) -> LocalReplayMode {
    if let Some(mode) = env_protocol_override() {
        return mode;
    }
    if caps.tool_calling {
        LocalReplayMode::NativeToolCalls
    } else {
        LocalReplayMode::TextualReplay
    }
}

/// Read `NANOBOT_LOCAL_PROTOCOL_MODE` and map accepted values to
/// `LocalReplayMode`. Mirrors `protocol.rs::LocalReplayMode::from_env` so
/// the two paths stay in lock-step during the parallel rollout.
fn env_protocol_override() -> Option<LocalReplayMode> {
    let raw = std::env::var("NANOBOT_LOCAL_PROTOCOL_MODE").ok()?;
    let value = raw.trim().to_ascii_lowercase();
    match value.as_str() {
        "native" | "tool_calls" | "native_tool_calls" => Some(LocalReplayMode::NativeToolCalls),
        "text" | "textual" | "textual_replay" => Some(LocalReplayMode::TextualReplay),
        _ => None,
    }
}

// ----------------------------------------------------------------------
// Tests — pinned fixtures for the invariants Waves 2/3 must preserve.
// ----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::model_capabilities::ReaderTier;

    // -- Fixture builders -------------------------------------------------

    mod fixtures {
        use super::*;

        /// Canonical Cloud fixture (no caps — `from_caps(None)`).
        pub fn cloud_mode() -> RuntimeMode {
            RuntimeMode::from_caps(None)
        }

        /// Higgs-like large local model: 32K ctx, tool-calling, size=large.
        pub fn higgs_caps() -> Arc<ModelCapabilities> {
            Arc::new(ModelCapabilities {
                size_class: ModelSizeClass::Large,
                tool_calling: true,
                thinking: true,
                needs_native_lms_api: false,
                strict_alternation: false,
                max_reliable_output: 16_384,
                scratch_pad_rounds: 10,
                reader_tier: ReaderTier::Advanced,
                parser: None,
            })
        }

        /// Small local model: no tool calling, size=small.
        pub fn small_caps() -> Arc<ModelCapabilities> {
            Arc::new(ModelCapabilities {
                size_class: ModelSizeClass::Small,
                tool_calling: false,
                thinking: false,
                needs_native_lms_api: false,
                strict_alternation: true,
                max_reliable_output: 512,
                scratch_pad_rounds: 3,
                reader_tier: ReaderTier::Minimal,
                parser: None,
            })
        }

        pub fn higgs_mode() -> RuntimeMode {
            RuntimeMode::from_caps(Some(higgs_caps()))
        }

        pub fn small_mode() -> RuntimeMode {
            RuntimeMode::from_caps(Some(small_caps()))
        }
    }

    // -- Constructor + variant-resolution tests --------------------------

    /// VALIDATION map: `resolves_cloud_variant` — `from_caps(None) -> Cloud`.
    #[test]
    fn resolves_cloud_variant() {
        let mode = RuntimeMode::from_caps(None);
        assert!(matches!(mode, RuntimeMode::Cloud));
        assert!(!mode.is_local());
    }

    /// VALIDATION map: `resolves_higgs_variant` — `from_caps(Some(caps)) -> Local { caps }`.
    #[test]
    fn resolves_higgs_variant() {
        let caps = fixtures::higgs_caps();
        let before = Arc::strong_count(&caps);
        let mode = RuntimeMode::from_caps(Some(caps.clone()));
        assert!(mode.is_local());
        match &mode {
            RuntimeMode::Local { caps: c } => {
                // Same Arc (not a deep clone).
                assert!(Arc::ptr_eq(c, &caps));
            }
            _ => panic!("expected Local variant"),
        }
        // The constructor took ownership of a clone, so strong_count grew by 1.
        assert!(Arc::strong_count(&caps) > before);
    }

    #[test]
    fn resolves_small_local_variant() {
        let mode = RuntimeMode::from_caps(Some(fixtures::small_caps()));
        assert!(mode.is_local());
    }

    // -- context_cap ------------------------------------------------------

    #[test]
    fn context_cap_cloud_uncapped() {
        assert_eq!(fixtures::cloud_mode().context_cap(200_000), 200_000);
    }

    #[test]
    fn context_cap_local_clamped_to_lite() {
        assert_eq!(fixtures::higgs_mode().context_cap(32_768), 800);
        assert_eq!(fixtures::small_mode().context_cap(8_192), 800);
    }

    // -- reserve_cap ------------------------------------------------------

    #[test]
    fn reserve_cap_cloud_passthrough() {
        assert_eq!(fixtures::cloud_mode().reserve_cap(8_192, 200_000), 8_192);
    }

    #[test]
    fn reserve_cap_local_higgs_at_25_pct_boundary() {
        // 0.25 * 32_768 == 8_192; min(8_192, 8_192) == 8_192.
        assert_eq!(fixtures::higgs_mode().reserve_cap(8_192, 32_768), 8_192);
    }

    #[test]
    fn reserve_cap_local_small_clamped() {
        // 0.25 * 8_192 == 2_048; min(8_192, 2_048) == 2_048.
        assert_eq!(fixtures::small_mode().reserve_cap(8_192, 8_192), 2_048);
    }

    // -- max_iterations ---------------------------------------------------

    #[test]
    fn max_iterations_cloud_scales_up() {
        // Cloud: configured=50, ctx=200_000 → at least 50 (scaled ceiling is 50).
        assert!(fixtures::cloud_mode().max_iterations(50, 200_000) >= 50);
        // Cloud: small configured + huge ctx should scale up.
        assert_eq!(fixtures::cloud_mode().max_iterations(5, 200_000), 50);
    }

    #[test]
    fn max_iterations_local_clamped_to_15() {
        assert_eq!(fixtures::higgs_mode().max_iterations(50, 200_000), 15);
        assert_eq!(fixtures::small_mode().max_iterations(50, 8_192), 15);
        // Local configured below the clamp stays as-is.
        assert_eq!(fixtures::higgs_mode().max_iterations(10, 32_768), 10);
    }

    // -- budget_strategy --------------------------------------------------

    #[test]
    fn budget_strategy_cloud_ratios() {
        let shares = fixtures::cloud_mode().budget_strategy(200_000);
        assert_eq!(shares.history_pct, 2);
        assert_eq!(shares.memory_pct, 1);
        assert_eq!(shares.working_memory_pct, 4);
        assert_eq!(shares.output_pct, 2);
    }

    #[test]
    fn budget_strategy_local_ratios() {
        let shares = fixtures::higgs_mode().budget_strategy(32_768);
        assert_eq!(shares.history_pct, 2);
        assert_eq!(shares.memory_pct, 1);
        assert_eq!(shares.working_memory_pct, 2);
        assert_eq!(shares.output_pct, 1);
        // Small local has identical ratios.
        assert_eq!(
            fixtures::small_mode().budget_strategy(8_192),
            fixtures::higgs_mode().budget_strategy(32_768)
        );
    }

    // -- protocol (caps-derived path, no env override) --------------------

    /// Global mutex that serialises every test which touches
    /// `NANOBOT_LOCAL_PROTOCOL_MODE`. `cargo test` runs tests in parallel
    /// by default, so any test that reads or mutates the env var MUST hold
    /// this guard — otherwise the env-override test races with the
    /// caps-derived tests and sees whichever state the scheduler left
    /// behind. The lock is re-entrant-safe via `Mutex::lock()` returning a
    /// guard that is dropped when each test returns.
    fn env_mutex() -> &'static std::sync::Mutex<()> {
        static ENV_MUTEX: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
        ENV_MUTEX.get_or_init(|| std::sync::Mutex::new(()))
    }

    /// Acquire the env lock and remove the override variable — so callers
    /// that need "caps-only" resolution observe a deterministic state.
    /// Returned guard keeps the lock held for the duration of the test.
    fn lock_env_cleared() -> std::sync::MutexGuard<'static, ()> {
        // Poison-tolerant: a panic in another env test should not prevent
        // this one from running. We recover the inner guard either way.
        let guard = env_mutex().lock().unwrap_or_else(|p| p.into_inner());
        std::env::remove_var("NANOBOT_LOCAL_PROTOCOL_MODE");
        guard
    }

    #[test]
    fn protocol_cloud_is_none() {
        let _g = lock_env_cleared();
        assert!(fixtures::cloud_mode().protocol().is_none());
    }

    #[test]
    fn protocol_derives_from_caps_native_when_tool_calling() {
        let _g = lock_env_cleared();
        assert_eq!(
            fixtures::higgs_mode().protocol(),
            Some(LocalReplayMode::NativeToolCalls)
        );
    }

    #[test]
    fn protocol_derives_from_caps_textual_when_no_tool_calling() {
        let _g = lock_env_cleared();
        assert_eq!(
            fixtures::small_mode().protocol(),
            Some(LocalReplayMode::TextualReplay)
        );
    }

    // -- protocol env override --------------------------------------------

    /// Env override wins over caps. Because env state is process-global,
    /// we save / restore the previous value around the mutation and
    /// assert both directions (textual override on a native-capable model
    /// and native override on a textual-defaulted model) in a single test.
    #[test]
    fn protocol_env_override_beats_caps() {
        let _guard = env_mutex().lock().unwrap_or_else(|p| p.into_inner());
        let key = "NANOBOT_LOCAL_PROTOCOL_MODE";
        let saved = std::env::var(key).ok();

        // Case 1: caps say native → env forces textual.
        std::env::set_var(key, "textual");
        assert_eq!(
            fixtures::higgs_mode().protocol(),
            Some(LocalReplayMode::TextualReplay)
        );

        // Case 2: caps say textual → env forces native.
        std::env::set_var(key, "native");
        assert_eq!(
            fixtures::small_mode().protocol(),
            Some(LocalReplayMode::NativeToolCalls)
        );

        // Case 3: Cloud ignores the env var entirely.
        assert!(fixtures::cloud_mode().protocol().is_none());

        // Restore.
        match saved {
            Some(v) => std::env::set_var(key, v),
            None => std::env::remove_var(key),
        }
    }

    // -- tool_def_mode ---------------------------------------------------

    #[test]
    fn tool_def_mode_cloud_is_full() {
        assert_eq!(fixtures::cloud_mode().tool_def_mode(), LocalToolMode::Full);
    }

    #[test]
    fn tool_def_mode_local_slim_when_tool_calling() {
        assert_eq!(fixtures::higgs_mode().tool_def_mode(), LocalToolMode::Slim);
    }

    #[test]
    fn tool_def_mode_local_proxy_when_no_tool_calling() {
        assert_eq!(fixtures::small_mode().tool_def_mode(), LocalToolMode::Proxy);
    }

    // -- needs_anti_drift ------------------------------------------------

    #[test]
    fn needs_anti_drift_cloud_false() {
        assert!(!fixtures::cloud_mode().needs_anti_drift());
    }

    #[test]
    fn needs_anti_drift_local_true_for_all_sizes() {
        assert!(fixtures::higgs_mode().needs_anti_drift());
        assert!(fixtures::small_mode().needs_anti_drift());
    }

    // -- grounding_role --------------------------------------------------

    #[test]
    fn grounding_role_cloud_is_system() {
        assert_eq!(fixtures::cloud_mode().grounding_role(), "system");
    }

    #[test]
    fn grounding_role_local_is_user() {
        assert_eq!(fixtures::higgs_mode().grounding_role(), "user");
        assert_eq!(fixtures::small_mode().grounding_role(), "user");
    }

    // -- thinking_cap_policy --------------------------------------------

    #[test]
    fn thinking_cap_cloud_uncapped() {
        assert_eq!(
            fixtures::cloud_mode().thinking_cap_policy(),
            ThinkingCapPolicy::Uncapped
        );
    }

    #[test]
    fn thinking_cap_local_large_uncapped() {
        assert_eq!(
            fixtures::higgs_mode().thinking_cap_policy(),
            ThinkingCapPolicy::Uncapped
        );
    }

    #[test]
    fn thinking_cap_local_small_is_hard() {
        match fixtures::small_mode().thinking_cap_policy() {
            ThinkingCapPolicy::Hard(budget) => assert_eq!(budget, SMALL_LOCAL_THINKING_BUDGET),
            other => panic!("expected Hard(..), got {:?}", other),
        }
    }

    // -- needs_local_protocol convenience --------------------------------

    #[test]
    fn needs_local_protocol_matches_protocol_some() {
        let _g = lock_env_cleared();
        assert!(!fixtures::cloud_mode().needs_local_protocol());
        assert!(fixtures::higgs_mode().needs_local_protocol());
        assert!(fixtures::small_mode().needs_local_protocol());
    }
}
