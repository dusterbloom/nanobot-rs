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

/// Create a provider from a `ProviderConfig` with `localhost:8080` fallback.
///
/// This is the common pattern for memory / delegation provider overrides
/// in `build_swappable_core()`.
pub fn from_provider_config(cfg: &ProviderConfig) -> Arc<dyn LLMProvider> {
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
    fn test_from_provider_config() {
        let cfg = ProviderConfig {
            api_key: "sk-test".to_string(),
            api_base: None,
        };
        let provider = from_provider_config(&cfg);
        // Should use the localhost:8080 fallback and the default model.
        assert!(provider.get_api_base().is_some());
    }
}
