//! Centralized provider construction.
//!
//! All LLM provider instances should be created through this module's
//! functions rather than calling `OpenAICompatProvider::new()` directly.
//! This centralises URL resolution, JIT gate wiring, and localhost fallback
//! logic in one place.

use std::sync::Arc;

use crate::config::schema::ProviderConfig;
use crate::providers::anthropic::AnthropicProvider;
use crate::providers::base::LLMProvider;
use crate::providers::jit_gate::JitGate;
use crate::providers::openai_compat::OpenAICompatProvider;

/// Descriptor for creating a provider instance.
pub struct ProviderSpec {
    /// API key (or "local" for local servers).
    pub api_key: String,
    /// API base URL. None = auto-detect from key prefix.
    pub api_base: Option<String>,
    /// Model identifier. None = provider default.
    pub model: Option<String>,
    /// Optional JIT gate for serialised access to shared servers.
    pub jit_gate: Option<Arc<JitGate>>,
}

impl ProviderSpec {
    /// Create a spec for a local server.
    pub fn local(base_url: &str, model: Option<&str>) -> Self {
        ProviderSpec {
            api_key: "local".to_string(),
            api_base: Some(base_url.to_string()),
            model: model.map(String::from),
            jit_gate: None,
        }
    }

    /// Create from a `ProviderConfig` (config.json provider section) with a
    /// default base URL fallback.
    pub fn from_config(cfg: &ProviderConfig, default_base: Option<&str>) -> Self {
        ProviderSpec {
            api_key: cfg.api_key.clone(),
            api_base: cfg
                .api_base
                .clone()
                .or_else(|| default_base.map(String::from)),
            model: None,
            jit_gate: None,
        }
    }

    /// Conditionally attach a JIT gate.
    pub fn with_jit_gate_opt(mut self, gate: Option<Arc<JitGate>>) -> Self {
        self.jit_gate = gate;
        self
    }
}

/// Create an OpenAI-compatible provider from a spec.
pub fn create_openai_compat(spec: ProviderSpec) -> Arc<dyn LLMProvider> {
    let mut prov = OpenAICompatProvider::new(
        &spec.api_key,
        spec.api_base.as_deref(),
        spec.model.as_deref(),
    );
    if let Some(gate) = spec.jit_gate {
        prov = prov.with_jit_gate(gate);
    }
    Arc::new(prov)
}

/// Create an Anthropic native provider (for OAuth / direct API).
pub fn create_anthropic(token: &str, model: Option<&str>) -> Arc<dyn LLMProvider> {
    Arc::new(AnthropicProvider::new(token, model))
}

/// Determine whether an api_base URL points to a local server.
fn is_local_base(base: &str) -> bool {
    base.contains("localhost") || base.contains("127.0.0.1")
}

/// Check whether a model name belongs to the Claude family.
fn is_claude_model(model: &str) -> bool {
    model.starts_with("claude")
}

/// Create a provider from a `ProviderConfig` with `localhost:8080` fallback.
///
/// Routing rules (applied in order):
/// 1. `api_base` contains `localhost` / `127.0.0.1` → OpenAICompat (local server).
/// 2. `api_key` starts with `sk-ant-` AND model is Claude (or unspecified) → AnthropicProvider.
/// 3. Otherwise → OpenAICompat (safe fallback for all other cloud providers).
///
/// The model hint prevents misrouting non-Claude models (e.g. ministral) to
/// the Anthropic Messages API, which would 404.
pub fn from_provider_config(cfg: &ProviderConfig) -> Arc<dyn LLMProvider> {
    from_provider_config_for_model(cfg, None)
}

/// Model-aware variant of [`from_provider_config`].
///
/// When the target `model` is known at the call site, pass it here so the
/// routing logic can avoid sending non-Claude models to the Anthropic API.
pub fn from_provider_config_for_model(
    cfg: &ProviderConfig,
    model: Option<&str>,
) -> Arc<dyn LLMProvider> {
    // Rule 1: explicit local base URL → always OpenAICompat.
    if let Some(ref base) = cfg.api_base {
        if is_local_base(base) {
            return create_openai_compat(ProviderSpec::from_config(cfg, None));
        }
    }

    // Rule 2: Anthropic API key + Claude model (or no model specified) → AnthropicProvider.
    if cfg.api_key.starts_with("sk-ant-") {
        let use_anthropic = match model {
            Some(m) => is_claude_model(m),
            None => true, // No model hint → assume Claude (backward compat).
        };
        if use_anthropic {
            return create_anthropic(&cfg.api_key, model);
        }
        // Non-Claude model with Anthropic key → fall through to OpenAICompat.
        // The key won't authenticate against localhost, but the caller should
        // have configured an api_base for non-Claude models.
    }

    // Rule 3: default – OpenAICompat with localhost:8080 fallback.
    create_openai_compat(ProviderSpec::from_config(cfg, Some("http://localhost:8080/v1")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_spec_local() {
        let spec = ProviderSpec::local("http://localhost:1234/v1", Some("my-model"));
        assert_eq!(spec.api_key, "local");
        assert_eq!(spec.api_base.as_deref(), Some("http://localhost:1234/v1"));
        assert_eq!(spec.model.as_deref(), Some("my-model"));
        assert!(spec.jit_gate.is_none());
    }

    #[test]
    fn test_provider_spec_from_config() {
        let cfg = ProviderConfig {
            api_key: "sk-test".to_string(),
            api_base: None,
        };
        let spec = ProviderSpec::from_config(&cfg, Some("http://localhost:8080/v1"));
        assert_eq!(spec.api_key, "sk-test");
        assert_eq!(
            spec.api_base.as_deref(),
            Some("http://localhost:8080/v1")
        );
        assert!(spec.model.is_none());
    }

    #[test]
    fn test_provider_spec_from_config_with_custom_base() {
        let cfg = ProviderConfig {
            api_key: "sk-test".to_string(),
            api_base: Some("https://custom.api.com/v1".to_string()),
        };
        let spec = ProviderSpec::from_config(&cfg, Some("http://localhost:8080/v1"));
        // Custom base takes precedence over default.
        assert_eq!(
            spec.api_base.as_deref(),
            Some("https://custom.api.com/v1")
        );
    }

    #[test]
    fn test_with_jit_gate_opt_none() {
        let spec = ProviderSpec::local("http://localhost:1234/v1", None)
            .with_jit_gate_opt(None);
        assert!(spec.jit_gate.is_none());
    }

    #[test]
    fn test_with_jit_gate_opt_some() {
        let gate = Arc::new(JitGate::new());
        let spec = ProviderSpec::local("http://localhost:1234/v1", None)
            .with_jit_gate_opt(Some(gate));
        assert!(spec.jit_gate.is_some());
    }

    #[test]
    fn test_create_openai_compat() {
        let spec = ProviderSpec {
            api_key: "test-key".to_string(),
            api_base: Some("https://api.example.com/v1".to_string()),
            model: Some("gpt-4".to_string()),
            jit_gate: None,
        };
        let provider = create_openai_compat(spec);
        assert_eq!(provider.get_default_model(), "gpt-4");
    }

    #[test]
    fn test_create_anthropic() {
        let provider = create_anthropic("test-token", Some("claude-sonnet-4-20250514"));
        assert_eq!(provider.get_default_model(), "claude-sonnet-4-20250514");
    }

    #[test]
    fn test_from_provider_config_unknown_key_uses_openai_compat() {
        let cfg = ProviderConfig {
            api_key: "sk-test".to_string(),
            api_base: None,
        };
        let provider = from_provider_config(&cfg);
        // Should use the localhost:8080 fallback.
        assert!(provider.get_api_base().is_some());
        assert_eq!(
            provider.get_api_base().as_deref(),
            Some("http://localhost:8080/v1")
        );
    }

    #[test]
    fn test_from_provider_config_local_base_uses_openai_compat() {
        let cfg = ProviderConfig {
            api_key: "local".to_string(),
            api_base: Some("http://localhost:11434/v1".to_string()),
        };
        let provider = from_provider_config(&cfg);
        assert_eq!(
            provider.get_api_base().as_deref(),
            Some("http://localhost:11434/v1")
        );
    }

    #[test]
    fn test_from_provider_config_127_base_uses_openai_compat() {
        let cfg = ProviderConfig {
            api_key: "local".to_string(),
            api_base: Some("http://127.0.0.1:8080/v1".to_string()),
        };
        let provider = from_provider_config(&cfg);
        assert_eq!(
            provider.get_api_base().as_deref(),
            Some("http://127.0.0.1:8080/v1")
        );
    }

    #[test]
    fn test_from_provider_config_anthropic_key_uses_anthropic_provider() {
        let cfg = ProviderConfig {
            api_key: "sk-ant-api03-abc123".to_string(),
            api_base: None,
        };
        let provider = from_provider_config(&cfg);
        // AnthropicProvider reports None for get_api_base() or the Anthropic base.
        // The key check: it should NOT be pointing to localhost.
        let base = provider.get_api_base();
        if let Some(b) = base {
            assert!(!is_local_base(&b), "Anthropic key should not route to local server");
        }
    }

    #[test]
    fn test_from_provider_config_anthropic_oauth_key_uses_anthropic_provider() {
        let cfg = ProviderConfig {
            api_key: "sk-ant-oat01-abc123".to_string(),
            api_base: None,
        };
        let provider = from_provider_config(&cfg);
        let base = provider.get_api_base();
        if let Some(b) = base {
            assert!(!is_local_base(&b), "OAuth key should not route to local server");
        }
    }

    #[test]
    fn test_is_local_base_localhost() {
        assert!(is_local_base("http://localhost:8080/v1"));
        assert!(is_local_base("http://localhost/v1"));
    }

    #[test]
    fn test_is_local_base_127() {
        assert!(is_local_base("http://127.0.0.1:11434/v1"));
    }

    #[test]
    fn test_is_local_base_remote() {
        assert!(!is_local_base("https://api.anthropic.com/v1"));
        assert!(!is_local_base("https://openrouter.ai/api/v1"));
        assert!(!is_local_base("https://api.openai.com/v1"));
    }

    // --- Model-aware routing tests (from_provider_config_for_model) ---

    #[test]
    fn test_anthropic_key_with_claude_model_uses_anthropic() {
        let cfg = ProviderConfig {
            api_key: "sk-ant-api03-abc123".to_string(),
            api_base: None,
        };
        let provider = from_provider_config_for_model(&cfg, Some("claude-3-haiku-20240307"));
        // AnthropicProvider has no api_base or returns the Anthropic URL.
        let base = provider.get_api_base();
        if let Some(b) = base {
            assert!(
                b.contains("anthropic"),
                "Claude model + Anthropic key should route to Anthropic, got: {}",
                b
            );
        }
        // Also: the default model should be set.
        assert_eq!(provider.get_default_model(), "claude-3-haiku-20240307");
    }

    #[test]
    fn test_anthropic_key_with_non_claude_model_uses_openai_compat() {
        // THIS IS THE BUG: sk-ant-* + ministral was being sent to Anthropic → 404
        let cfg = ProviderConfig {
            api_key: "sk-ant-api03-abc123".to_string(),
            api_base: None,
        };
        let provider = from_provider_config_for_model(&cfg, Some("ministral-3-8b-instruct-2512"));
        // Should NOT be Anthropic — should be OpenAICompat.
        let base = provider.get_api_base();
        assert!(
            base.is_some(),
            "Non-Claude model should use OpenAICompat (which has an api_base)"
        );
        let b = base.unwrap();
        assert!(
            !b.contains("anthropic"),
            "ministral should NOT be routed to Anthropic, got: {}",
            b
        );
    }

    #[test]
    fn test_anthropic_key_with_gemma_model_uses_openai_compat() {
        let cfg = ProviderConfig {
            api_key: "sk-ant-api03-abc123".to_string(),
            api_base: None,
        };
        let provider = from_provider_config_for_model(&cfg, Some("gemma-2-9b-it"));
        let base = provider.get_api_base();
        assert!(base.is_some(), "gemma should use OpenAICompat");
    }

    #[test]
    fn test_anthropic_key_no_model_hint_uses_anthropic() {
        // Backward compat: no model → assume Claude.
        let cfg = ProviderConfig {
            api_key: "sk-ant-api03-abc123".to_string(),
            api_base: None,
        };
        let provider = from_provider_config_for_model(&cfg, None);
        let base = provider.get_api_base();
        // Should be Anthropic (None base or anthropic URL).
        if let Some(b) = &base {
            assert!(!is_local_base(b), "No model hint + Anthropic key should not go to localhost");
        }
    }

    #[test]
    fn test_local_base_overrides_model_hint() {
        // Local base always wins, even with Claude model.
        let cfg = ProviderConfig {
            api_key: "sk-ant-api03-abc123".to_string(),
            api_base: Some("http://localhost:1234/v1".to_string()),
        };
        let provider = from_provider_config_for_model(&cfg, Some("claude-3-haiku-20240307"));
        assert_eq!(
            provider.get_api_base().as_deref(),
            Some("http://localhost:1234/v1"),
            "Local base should override Anthropic key even for Claude model"
        );
    }

    #[test]
    fn test_non_anthropic_key_with_any_model_uses_openai_compat() {
        let cfg = ProviderConfig {
            api_key: "sk-or-test123".to_string(),
            api_base: None,
        };
        let provider = from_provider_config_for_model(&cfg, Some("ministral-3-8b-instruct-2512"));
        let base = provider.get_api_base();
        assert_eq!(
            base.as_deref(),
            Some("http://localhost:8080/v1"),
            "Non-Anthropic key should always use OpenAICompat"
        );
    }

    #[test]
    fn test_is_claude_model() {
        assert!(is_claude_model("claude-3-haiku-20240307"));
        assert!(is_claude_model("claude-sonnet-4-20250514"));
        assert!(is_claude_model("claude-opus-4-20250514"));
        assert!(!is_claude_model("ministral-3-8b-instruct-2512"));
        assert!(!is_claude_model("gemma-2-9b-it"));
        assert!(!is_claude_model("llama-3.3-70b"));
        assert!(!is_claude_model("gpt-4o"));
    }
}
