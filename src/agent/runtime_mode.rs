// Error-protocol layer-3 backlog (docs/research/2026-08-06-error-conventions-and-host-bridge.md §3.6):
// the deny regime in Cargo.toml is live; this module still carries pre-existing
// violations of the lints below. Remove this allow as the module migrates onto
// the regime.
#![allow(
    clippy::shadow_reuse,
)]
//! Typed replacement for `SwappableCore::is_local`.
//!
//! `RuntimeMode` distinguishes a `Cloud` managed API from any locally-reachable
//! backend (Higgs sidecar, LM Studio, vLLM, cluster peer), carrying the resolved
//! model capabilities for the local case. Live derivations dispatch via `match`
//! on the variant rather than scattering `if is_local` branches across the code.
//!
//! The two-variant shape is locked by 09-CONTEXT.md: `Cloud` is a unit variant,
//! `Local` wraps `Arc<ModelCapabilities>` — cluster / remote LM-Studio /
//! private-IP OpenAI-compat endpoints all route through `Local`, the
//! capabilities carry the differentiation.

use std::sync::Arc;

use crate::agent::model_capabilities::ModelCapabilities;

/// Runtime descriptor. Every `is_local` branch collapses into one method call
/// on this enum, dispatched via `match`.
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

impl RuntimeMode {
    /// Resolve a `RuntimeMode` from an optional capabilities handle.
    ///
    /// `Some(caps)` → `Local { caps }` (any local backend is selected).
    /// `None`       → `Cloud` (no local backend configured).
    pub fn from_caps(caps: Option<Arc<ModelCapabilities>>) -> Self {
        match caps {
            Some(caps) => Self::Local { caps },
            None => Self::Cloud,
        }
    }

    /// `true` iff this is a `Local` variant.
    pub fn is_local(&self) -> bool {
        matches!(self, Self::Local { .. })
    }

    /// Return the effective response-reserve budget.
    ///
    /// - Cloud: `max_tokens` (no clamp).
    /// - Local: `min(max_tokens, max_ctx / 4)` — the 25 % clamp preserves some
    ///   conversation space even when the user configured a huge `max_tokens`.
    pub fn reserve_cap(&self, max_tokens: usize, max_ctx: usize) -> usize {
        match self {
            Self::Cloud => max_tokens,
            Self::Local { .. } => max_tokens.min(max_ctx / 4),
        }
    }

    /// `true` iff the anti-drift nudge should be enabled for this runtime.
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
}

// ----------------------------------------------------------------------
// Tests
// ----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::model_capabilities::{ModelSizeClass, ReaderTier};

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
                vision: false,
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
                vision: false,
            })
        }

        pub fn higgs_mode() -> RuntimeMode {
            RuntimeMode::from_caps(Some(higgs_caps()))
        }

        pub fn small_mode() -> RuntimeMode {
            RuntimeMode::from_caps(Some(small_caps()))
        }
    }

    // -- Constructor + variant resolution --------------------------------

    #[test]
    fn resolves_cloud_variant() {
        let mode = RuntimeMode::from_caps(None);
        assert!(matches!(mode, RuntimeMode::Cloud));
        assert!(!mode.is_local());
    }

    #[test]
    fn resolves_higgs_variant() {
        let caps = fixtures::higgs_caps();
        let before = Arc::strong_count(&caps);
        let mode = RuntimeMode::from_caps(Some(caps.clone()));
        assert!(mode.is_local());
        match &mode {
            RuntimeMode::Local { caps: c } => assert!(Arc::ptr_eq(c, &caps)),
            _ => panic!("expected Local variant"),
        }
        assert!(Arc::strong_count(&caps) > before);
    }

    #[test]
    fn resolves_small_local_variant() {
        let mode = RuntimeMode::from_caps(Some(fixtures::small_caps()));
        assert!(mode.is_local());
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

    // -- needs_anti_drift -------------------------------------------------

    #[test]
    fn needs_anti_drift_cloud_false() {
        assert!(!fixtures::cloud_mode().needs_anti_drift());
    }

    #[test]
    fn needs_anti_drift_local_true_for_all_sizes() {
        assert!(fixtures::higgs_mode().needs_anti_drift());
        assert!(fixtures::small_mode().needs_anti_drift());
    }

    // -- grounding_role ---------------------------------------------------

    #[test]
    fn grounding_role_cloud_is_system() {
        assert_eq!(fixtures::cloud_mode().grounding_role(), "system");
    }

    #[test]
    fn grounding_role_local_is_user() {
        assert_eq!(fixtures::higgs_mode().grounding_role(), "user");
        assert_eq!(fixtures::small_mode().grounding_role(), "user");
    }
}
