//! Model capability detection: replaces scattered model-name string matching
//! with a centralized registry. Built-in patterns cover known models; config
//! overrides let users customize for new/custom models.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Broad size class for model-dependent tuning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelSizeClass {
    Small,
    Medium,
    Large,
}

/// Reader capability tier for compaction (mirrors old ReaderCapability).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReaderTier {
    Minimal,
    Standard,
    Advanced,
}

/// Capabilities of a model, looked up once and stored on SwappableCore.
#[derive(Debug, Clone)]
pub struct ModelCapabilities {
    pub size_class: ModelSizeClass,
    pub tool_calling: bool,
    pub thinking: bool,
    pub needs_native_lms_api: bool,
    pub strict_alternation: bool,
    pub max_reliable_output: usize,
    pub scratch_pad_rounds: usize,
    pub reader_tier: ReaderTier,
}

/// Partial override from config.json `modelCapabilities` section.
/// Only specified fields override the built-in defaults.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct ModelCapabilitiesOverride {
    pub size_class: Option<ModelSizeClass>,
    pub tool_calling: Option<bool>,
    pub thinking: Option<bool>,
    pub needs_native_lms_api: Option<bool>,
    pub strict_alternation: Option<bool>,
    pub max_reliable_output: Option<usize>,
    pub scratch_pad_rounds: Option<usize>,
    pub reader_tier: Option<ReaderTier>,
}

/// Build `ModelCapabilities` from a model name string.
///
/// Applies built-in pattern matching first, then overlays any config overrides.
/// Patterns replicate the exact logic from the old scattered functions:
/// - `agent_core::is_small_local_model()`
/// - `compaction::ReaderProfile::from_model()`
/// - `tool_runner::scratch_pad_round_budget()`
/// - `openai_compat::needs_native_lms_api()`
pub fn lookup(
    model: &str,
    overrides: &HashMap<String, ModelCapabilitiesOverride>,
) -> ModelCapabilities {
    let lower = model.to_ascii_lowercase();

    // Start with built-in pattern matching
    let mut caps = builtin_capabilities(&lower);

    // Apply overrides: check each override key as a substring match
    for (pattern, ovr) in overrides {
        if lower.contains(&pattern.to_ascii_lowercase()) {
            apply_override(&mut caps, ovr);
            break; // first match wins
        }
    }

    caps
}

fn builtin_capabilities(lower: &str) -> ModelCapabilities {
    // Specific model patterns (most specific first)
    if lower.contains("nanbeige") {
        return ModelCapabilities {
            size_class: ModelSizeClass::Small,
            tool_calling: true,
            thinking: false,
            needs_native_lms_api: false,
            strict_alternation: true,
            max_reliable_output: 512,
            scratch_pad_rounds: 3,
            reader_tier: ReaderTier::Minimal,
        };
    }
    if lower.contains("functiongemma") {
        return ModelCapabilities {
            size_class: ModelSizeClass::Small,
            tool_calling: true,
            thinking: false,
            needs_native_lms_api: false,
            strict_alternation: true,
            max_reliable_output: 512,
            scratch_pad_rounds: 2,
            reader_tier: ReaderTier::Minimal,
        };
    }
    if lower.contains("ministral-3") {
        return ModelCapabilities {
            size_class: ModelSizeClass::Small,
            tool_calling: true,
            thinking: false,
            needs_native_lms_api: false,
            strict_alternation: true,
            max_reliable_output: 1024,
            scratch_pad_rounds: 4,
            reader_tier: ReaderTier::Minimal,
        };
    }
    if lower.contains("qwen3-1.7b") {
        return ModelCapabilities {
            size_class: ModelSizeClass::Small,
            tool_calling: true,
            thinking: true,
            needs_native_lms_api: false,
            strict_alternation: true,
            max_reliable_output: 1024,
            scratch_pad_rounds: 4,
            reader_tier: ReaderTier::Minimal,
        };
    }
    // Nemotron / orchestrator models (Medium, thinking enabled, native LMS API)
    if lower.contains("nemotron") || lower.contains("orchestrator") {
        return ModelCapabilities {
            size_class: ModelSizeClass::Medium,
            tool_calling: true,
            thinking: true,
            needs_native_lms_api: true,
            strict_alternation: false,
            max_reliable_output: 4096,
            scratch_pad_rounds: 10,
            reader_tier: ReaderTier::Standard,
        };
    }
    // Cloud / large models
    if lower.contains("claude")
        || lower.contains("gpt-4")
        || lower.contains("opus")
        || lower.contains("sonnet")
        || lower.contains("gemini-2")
        || lower.contains("gemini-1.5")
        || lower.contains("gemini")
    {
        return ModelCapabilities {
            size_class: ModelSizeClass::Large,
            tool_calling: true,
            thinking: true,
            needs_native_lms_api: false,
            strict_alternation: false,
            max_reliable_output: 16384,
            scratch_pad_rounds: 10,
            reader_tier: ReaderTier::Advanced,
        };
    }
    // Generic small model patterns (catch-all for size indicators)
    if lower.contains("3b")
        || lower.contains("1.7b")
        || lower.contains("0.5b")
        || lower.contains("1b")
        || lower.contains("ministral")
    {
        return ModelCapabilities {
            size_class: ModelSizeClass::Small,
            tool_calling: true,
            thinking: false,
            needs_native_lms_api: false,
            strict_alternation: true,
            max_reliable_output: 1024,
            scratch_pad_rounds: 4,
            reader_tier: ReaderTier::Minimal,
        };
    }
    // Unknown default: Medium, conservative
    ModelCapabilities {
        size_class: ModelSizeClass::Medium,
        tool_calling: true,
        thinking: false,
        needs_native_lms_api: false,
        strict_alternation: false,
        max_reliable_output: 4096,
        scratch_pad_rounds: 10,
        reader_tier: ReaderTier::Standard,
    }
}

fn apply_override(caps: &mut ModelCapabilities, ovr: &ModelCapabilitiesOverride) {
    if let Some(v) = ovr.size_class {
        caps.size_class = v;
    }
    if let Some(v) = ovr.tool_calling {
        caps.tool_calling = v;
    }
    if let Some(v) = ovr.thinking {
        caps.thinking = v;
    }
    if let Some(v) = ovr.needs_native_lms_api {
        caps.needs_native_lms_api = v;
    }
    if let Some(v) = ovr.strict_alternation {
        caps.strict_alternation = v;
    }
    if let Some(v) = ovr.max_reliable_output {
        caps.max_reliable_output = v;
    }
    if let Some(v) = ovr.scratch_pad_rounds {
        caps.scratch_pad_rounds = v;
    }
    if let Some(v) = ovr.reader_tier {
        caps.reader_tier = v;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_overrides() -> HashMap<String, ModelCapabilitiesOverride> {
        HashMap::new()
    }

    #[test]
    fn test_nanbeige() {
        let caps = lookup("nanbeige-2-8b", &empty_overrides());
        assert_eq!(caps.size_class, ModelSizeClass::Small);
        assert_eq!(caps.scratch_pad_rounds, 3);
        assert_eq!(caps.reader_tier, ReaderTier::Minimal);
        assert!(!caps.needs_native_lms_api);
        assert!(caps.strict_alternation);
    }

    #[test]
    fn test_functiongemma() {
        let caps = lookup("functiongemma-2b", &empty_overrides());
        assert_eq!(caps.size_class, ModelSizeClass::Small);
        assert_eq!(caps.scratch_pad_rounds, 2);
    }

    #[test]
    fn test_ministral_3() {
        let caps = lookup("ministral-3b-instruct", &empty_overrides());
        assert_eq!(caps.size_class, ModelSizeClass::Small);
        assert_eq!(caps.scratch_pad_rounds, 4);
        assert_eq!(caps.reader_tier, ReaderTier::Minimal);
    }

    #[test]
    fn test_qwen3_thinking() {
        let caps = lookup("qwen3-1.7b-q4_k_m", &empty_overrides());
        assert_eq!(caps.size_class, ModelSizeClass::Small);
        assert!(caps.thinking);
        assert_eq!(caps.scratch_pad_rounds, 4);
    }

    #[test]
    fn test_nemotron() {
        let caps = lookup("nvidia/nemotron-mini-4b", &empty_overrides());
        assert_eq!(caps.size_class, ModelSizeClass::Medium);
        assert!(caps.needs_native_lms_api);
        assert!(caps.thinking);
        assert!(!caps.strict_alternation);
    }

    #[test]
    fn test_orchestrator() {
        let caps = lookup("orchestrator-7b", &empty_overrides());
        assert!(caps.needs_native_lms_api);
    }

    #[test]
    fn test_claude() {
        let caps = lookup("anthropic/claude-opus-4-5", &empty_overrides());
        assert_eq!(caps.size_class, ModelSizeClass::Large);
        assert_eq!(caps.reader_tier, ReaderTier::Advanced);
        assert_eq!(caps.scratch_pad_rounds, 10);
    }

    #[test]
    fn test_gpt4() {
        let caps = lookup("gpt-4-turbo", &empty_overrides());
        assert_eq!(caps.size_class, ModelSizeClass::Large);
        assert_eq!(caps.reader_tier, ReaderTier::Advanced);
    }

    #[test]
    fn test_gemini() {
        let caps = lookup("gemini-2-flash", &empty_overrides());
        assert_eq!(caps.size_class, ModelSizeClass::Large);
        assert_eq!(caps.reader_tier, ReaderTier::Advanced);
    }

    #[test]
    fn test_generic_3b() {
        let caps = lookup("some-random-3b-model", &empty_overrides());
        assert_eq!(caps.size_class, ModelSizeClass::Small);
        assert_eq!(caps.reader_tier, ReaderTier::Minimal);
    }

    #[test]
    fn test_unknown_default() {
        let caps = lookup("my-custom-model", &empty_overrides());
        assert_eq!(caps.size_class, ModelSizeClass::Medium);
        assert_eq!(caps.reader_tier, ReaderTier::Standard);
        assert_eq!(caps.scratch_pad_rounds, 10);
        assert!(!caps.needs_native_lms_api);
    }

    #[test]
    fn test_config_override() {
        let mut overrides = HashMap::new();
        overrides.insert(
            "my-custom".to_string(),
            ModelCapabilitiesOverride {
                size_class: Some(ModelSizeClass::Small),
                max_reliable_output: Some(512),
                scratch_pad_rounds: Some(3),
                ..Default::default()
            },
        );
        let caps = lookup("my-custom-3b", &overrides);
        assert_eq!(caps.size_class, ModelSizeClass::Small);
        assert_eq!(caps.max_reliable_output, 512);
        assert_eq!(caps.scratch_pad_rounds, 3);
    }

    #[test]
    fn test_default_assertions() {
        // Verify every field of the unknown default matches expected values
        let caps = lookup("unknown-model-xyz", &empty_overrides());
        assert_eq!(caps.size_class, ModelSizeClass::Medium);
        assert!(caps.tool_calling);
        assert!(!caps.thinking);
        assert!(!caps.needs_native_lms_api);
        assert!(!caps.strict_alternation);
        assert_eq!(caps.max_reliable_output, 4096);
        assert_eq!(caps.scratch_pad_rounds, 10);
        assert_eq!(caps.reader_tier, ReaderTier::Standard);
    }
}
