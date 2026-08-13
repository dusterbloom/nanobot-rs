// Error-protocol layer-3 backlog (docs/research/2026-08-06-error-conventions-and-host-bridge.md §3.6):
// the deny regime in Cargo.toml is live; this module still carries pre-existing
// violations of the lints below. Remove this allow as the module migrates onto
// the regime.
// Tracking: docs/error-protocol-backlog.md
#![allow(clippy::shadow_reuse)]
//! Centralized provider construction.
//!
//! All LLM provider instances should be created through this module's
//! functions rather than calling `OpenAICompatProvider::new()` directly.
//! This centralises URL resolution, JIT gate wiring, and localhost fallback
//! logic in one place.

use std::sync::Arc;

use async_trait::async_trait;

use crate::config::schema::{ProviderConfig, RetryConfig};
use crate::providers::anthropic::AnthropicProvider;
use crate::providers::base::{LLMProvider, LLMResponse};
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
    /// Retry backoff settings (default: provider 1-30s, JIT 2-8s).
    pub retry: RetryConfig,
    /// HTTP request timeout in seconds (default: 120).
    pub timeout_secs: u64,
    /// Timeout for LMS native probe requests in seconds (default: 2).
    pub lms_native_probe_secs: u64,
    /// Whether grammar-constrained tool calls are enabled for local backends
    /// (default: true). The config escape hatch sets this false.
    pub constrained_tool_calls: bool,
    /// Whether to send Higgs' `session_id` extension for cache-resident turns.
    pub higgs_session_cache: bool,
    /// OpenAI-compatible repetition penalty.
    pub repetition_penalty: Option<f64>,
    /// OpenAI-compatible frequency penalty.
    pub frequency_penalty: Option<f64>,
    /// OpenAI-compatible presence penalty.
    pub presence_penalty: Option<f64>,
}

impl ProviderSpec {
    /// Create a spec for a local server.
    pub fn local(base_url: &str, model: Option<&str>) -> Self {
        Self::local_with_key(base_url, model, "local")
    }

    /// Create a spec for a local server with a custom API key.
    pub fn local_with_key(base_url: &str, model: Option<&str>, api_key: &str) -> Self {
        ProviderSpec {
            api_key: api_key.to_string(),
            api_base: Some(base_url.to_string()),
            model: model.map(String::from),
            jit_gate: None,
            retry: RetryConfig::default(),
            timeout_secs: 120,
            lms_native_probe_secs: 2,
            constrained_tool_calls: true,
            higgs_session_cache: false,
            repetition_penalty: None,
            frequency_penalty: None,
            presence_penalty: None,
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
            retry: RetryConfig::default(),
            timeout_secs: 120,
            lms_native_probe_secs: 2,
            constrained_tool_calls: true,
            higgs_session_cache: false,
            repetition_penalty: None,
            frequency_penalty: None,
            presence_penalty: None,
        }
    }

    /// Conditionally attach a JIT gate.
    pub fn with_jit_gate_opt(mut self, gate: Option<Arc<JitGate>>) -> Self {
        self.jit_gate = gate;
        self
    }

    /// Override HTTP timeout and LMS native probe timeout from `TimeoutsConfig`.
    pub fn with_timeout_config(mut self, timeouts: &crate::config::schema::TimeoutsConfig) -> Self {
        self.timeout_secs = timeouts.provider_http_secs;
        self.lms_native_probe_secs = timeouts.lms_native_probe_secs;
        self
    }

    /// Override retry backoff settings from `RetryConfig`.
    pub fn with_retry(mut self, retry: RetryConfig) -> Self {
        self.retry = retry;
        self
    }

    /// Enable/disable grammar-constrained tool calls for local backends.
    pub fn with_constrained_tool_calls(mut self, enabled: bool) -> Self {
        self.constrained_tool_calls = enabled;
        self
    }

    /// Enable Higgs' cache-resident continuation extension for this provider.
    pub fn with_higgs_session_cache(mut self, enabled: bool) -> Self {
        self.higgs_session_cache = enabled;
        self
    }

    /// Attach OpenAI-compatible sampling penalties to this provider spec.
    pub fn with_sampling_penalties(
        mut self,
        repetition_penalty: Option<f64>,
        frequency_penalty: Option<f64>,
        presence_penalty: Option<f64>,
    ) -> Self {
        self.repetition_penalty = repetition_penalty;
        self.frequency_penalty = frequency_penalty;
        self.presence_penalty = presence_penalty;
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
    if spec.timeout_secs != 120 {
        prov = prov.with_timeout(spec.timeout_secs);
    }
    if spec.lms_native_probe_secs != 2 {
        prov = prov.with_lms_native_probe_secs(spec.lms_native_probe_secs);
    }
    if let Some(gate) = spec.jit_gate {
        prov = prov.with_jit_gate(gate);
    }
    prov = prov.with_sampling_penalties(
        spec.repetition_penalty,
        spec.frequency_penalty,
        spec.presence_penalty,
    );
    prov = prov.with_retry_config(
        spec.retry.provider_min_secs,
        spec.retry.provider_max_secs,
        spec.retry.jit_min_secs,
        spec.retry.jit_max_secs,
    );
    prov = prov.with_constrained_tool_calls(spec.constrained_tool_calls);
    prov = prov.with_higgs_session_cache(spec.higgs_session_cache);
    Arc::new(prov)
}

/// Create an Anthropic native provider (for OAuth / direct API).
pub fn create_anthropic(token: &str, model: Option<&str>) -> Arc<dyn LLMProvider> {
    Arc::new(AnthropicProvider::new(token, model))
}

struct MisconfiguredProvider {
    model: String,
    message: String,
}

#[async_trait]
impl LLMProvider for MisconfiguredProvider {
    async fn chat(
        &self,
        _messages: &[serde_json::Value],
        _tools: Option<&[serde_json::Value]>,
        _model: Option<&str>,
        _max_tokens: u32,
        _temperature: f64,
        _thinking_budget: Option<u32>,
        _top_p: Option<f64>,
    ) -> anyhow::Result<LLMResponse> {
        Err(anyhow::anyhow!(self.message.clone()))
    }

    fn get_default_model(&self) -> &str {
        &self.model
    }
}

/// Determine whether an api_base URL points to a local server.
fn is_local_base(base: &str) -> bool {
    base.contains("localhost") || base.contains("127.0.0.1")
}

/// Check whether a model name belongs to the Claude family.
pub fn is_claude_model(model: &str) -> bool {
    model.starts_with("claude")
}

fn openai_compat_from_config(
    cfg: &ProviderConfig,
    model: Option<&str>,
    default_base: Option<&str>,
) -> Arc<dyn LLMProvider> {
    let mut spec = ProviderSpec::from_config(cfg, default_base);
    spec.model = model.map(String::from);
    create_openai_compat(spec)
}

/// Create a provider from a `ProviderConfig`.
///
/// This form has no implicit local fallback. Local callers that want an
/// auxiliary provider to inherit the current server must use
/// [`from_provider_config_for_model_with_default_base`].
pub fn from_provider_config_for_model(
    cfg: &ProviderConfig,
    model: Option<&str>,
) -> Arc<dyn LLMProvider> {
    from_provider_config_for_model_with_default_base(cfg, model, None)
}

/// Create a provider from a `ProviderConfig`, optionally inheriting a caller
/// supplied base URL.
///
/// Routing rules (applied in order):
/// 1. `api_base` contains `localhost` / `127.0.0.1` → OpenAICompat (local server).
/// 2. `api_key` starts with `sk-ant-` AND model is Claude (or unspecified) → AnthropicProvider.
/// 3. Otherwise → OpenAICompat, using only explicit config or `default_base`.
///
/// When the target `model` is known at the call site, pass it here so the
/// routing logic can avoid sending non-Claude models to the Anthropic API.
pub fn from_provider_config_for_model_with_default_base(
    cfg: &ProviderConfig,
    model: Option<&str>,
    default_base: Option<&str>,
) -> Arc<dyn LLMProvider> {
    // Rule 1: explicit local base URL → always OpenAICompat.
    if let Some(ref base) = cfg.api_base {
        if is_local_base(base) {
            return openai_compat_from_config(cfg, model, None);
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
        if cfg.api_base.is_none() && default_base.is_none() {
            let model = model.unwrap_or("non-Claude model").to_string();
            return Arc::new(MisconfiguredProvider {
                model: model.clone(),
                message: format!(
                    "Provider config has an Anthropic API key for {model}, but no apiBase/default_base. \
                     Set apiBase for that model or use a Claude model."
                ),
            });
        }
    }

    openai_compat_from_config(cfg, model, default_base)
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
        assert_eq!(spec.api_base.as_deref(), Some("http://localhost:8080/v1"));
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
        assert_eq!(spec.api_base.as_deref(), Some("https://custom.api.com/v1"));
    }

    #[test]
    fn test_with_jit_gate_opt_none() {
        let spec = ProviderSpec::local("http://localhost:1234/v1", None).with_jit_gate_opt(None);
        assert!(spec.jit_gate.is_none());
    }

    #[test]
    fn test_with_jit_gate_opt_some() {
        let gate = Arc::new(JitGate::new());
        let spec =
            ProviderSpec::local("http://localhost:1234/v1", None).with_jit_gate_opt(Some(gate));
        assert!(spec.jit_gate.is_some());
    }

    #[test]
    fn test_create_openai_compat() {
        let spec = ProviderSpec {
            api_key: "test-key".to_string(),
            api_base: Some("https://api.example.com/v1".to_string()),
            model: Some("gpt-4".to_string()),
            jit_gate: None,
            retry: RetryConfig::default(),
            timeout_secs: 120,
            lms_native_probe_secs: 2,
            constrained_tool_calls: true,
            higgs_session_cache: false,
            repetition_penalty: None,
            frequency_penalty: None,
            presence_penalty: None,
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
        let provider = from_provider_config_for_model(&cfg, None);
        assert!(provider.get_api_base().is_some());
        assert!(
            provider
                .get_api_base()
                .is_some_and(|base| base.contains("openrouter")),
            "unconfigured provider should not invent localhost: {:?}",
            provider.get_api_base()
        );
    }

    #[test]
    fn test_provider_config_default_base_inherits_local_endpoint() {
        let cfg = ProviderConfig {
            api_key: "local".to_string(),
            api_base: None,
        };
        let provider = from_provider_config_for_model_with_default_base(
            &cfg,
            Some("local:qwen36-35b"),
            Some("http://127.0.0.1:8000/v1"),
        );
        assert_eq!(
            provider.get_api_base().as_deref(),
            Some("http://127.0.0.1:8000/v1")
        );
        assert_eq!(provider.get_default_model(), "local:qwen36-35b");
    }

    #[test]
    fn test_from_provider_config_local_base_uses_openai_compat() {
        let cfg = ProviderConfig {
            api_key: "local".to_string(),
            api_base: Some("http://localhost:11434/v1".to_string()),
        };
        let provider = from_provider_config_for_model(&cfg, None);
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
        let provider = from_provider_config_for_model(&cfg, None);
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
        let provider = from_provider_config_for_model(&cfg, None);
        // AnthropicProvider reports None for get_api_base() or the Anthropic base.
        // The key check: it should NOT be pointing to localhost.
        let base = provider.get_api_base();
        if let Some(b) = base {
            assert!(
                !is_local_base(&b),
                "Anthropic key should not route to local server"
            );
        }
    }

    #[test]
    fn test_from_provider_config_anthropic_oauth_key_uses_anthropic_provider() {
        let cfg = ProviderConfig {
            api_key: "sk-ant-oat01-abc123".to_string(),
            api_base: None,
        };
        let provider = from_provider_config_for_model(&cfg, None);
        let base = provider.get_api_base();
        if let Some(b) = base {
            assert!(
                !is_local_base(&b),
                "OAuth key should not route to local server"
            );
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
        let base = provider.get_api_base();
        assert!(
            base.is_none(),
            "Non-Claude Anthropic config without apiBase should fail before HTTP, not invent a base: {base:?}"
        );
        assert_eq!(provider.get_default_model(), "ministral-3-8b-instruct-2512");
    }

    #[test]
    fn test_anthropic_key_with_gemma_model_uses_openai_compat() {
        let cfg = ProviderConfig {
            api_key: "sk-ant-api03-abc123".to_string(),
            api_base: None,
        };
        let provider = from_provider_config_for_model(&cfg, Some("gemma-2-9b-it"));
        let base = provider.get_api_base();
        assert!(
            base.is_none(),
            "gemma with Anthropic key and no apiBase should fail before HTTP"
        );
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
            assert!(
                !is_local_base(b),
                "No model hint + Anthropic key should not go to localhost"
            );
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
            Some("https://openrouter.ai/api/v1"),
            "OpenRouter key should use OpenRouter, not a local fallback"
        );
        assert_eq!(provider.get_default_model(), "ministral-3-8b-instruct-2512");
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
