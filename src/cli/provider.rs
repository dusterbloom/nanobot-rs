//! Cloud provider creation, OAuth loading, and API key validation.

// Interactive/app boundary (error-protocol layer 3 backlog): printing IS the
// product here (REPL/TUI/CLI), and the thin glue code keeps pragmatic
// unwraps on always-set state (rl, runtime, static regexes). The deny regime
// in Cargo.toml stays live for the core; this module lands on the regime
// when its backlog is migrated.
#![allow(
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable,
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::shadow_reuse,
    clippy::shadow_unrelated,
    clippy::shadow_same,
    clippy::format_push_string,
    clippy::string_add
)]
use std::sync::Arc;

use crate::config::schema::Config;
use crate::providers::base::LLMProvider;
use crate::providers::factory;

pub(crate) fn create_provider(config: &Config) -> Arc<dyn LLMProvider> {
    use tracing::info;
    let model = &config.agents.defaults.model;
    let (repetition_penalty, frequency_penalty, presence_penalty) = (
        config.agents.defaults.repetition_penalty,
        config.agents.defaults.frequency_penalty,
        config.agents.defaults.presence_penalty,
    );

    // Try provider prefix resolution (e.g. "zhipu-coding/glm-5", "groq/llama-3.3-70b")
    if let Some((api_key, api_base, stripped_model)) = config.resolve_provider_for_model(model) {
        info!(
            "create_provider: prefix resolved model={} -> base={}, stripped={}",
            model, api_base, stripped_model
        );
        return factory::create_openai_compat(factory::ProviderSpec {
            api_key,
            api_base: Some(api_base),
            model: Some(stripped_model),
            jit_gate: None,
            retry: config.retry.clone(),
            timeout_secs: config.timeouts.provider_http_secs,
            lms_native_probe_secs: config.timeouts.lms_native_probe_secs,
            constrained_tool_calls: config.agents.defaults.constrained_tool_calls,
            higgs_session_cache: false,
            repetition_penalty,
            frequency_penalty,
            presence_penalty,
        });
    }

    // Model has a known provider prefix but that provider's key is empty.
    // Strip the prefix so the fallback provider (whichever has a key) gets
    // a clean model name instead of "anthropic/claude-opus-4-5".
    let model =
        if let Some(stripped) = crate::config::schema::ProvidersConfig::strip_known_prefix(model) {
            info!(
                "create_provider: prefix provider has no key, stripped '{}' -> '{}'",
                model, stripped
            );
            stripped.to_string()
        } else {
            model.clone()
        };

    let api_key = config.get_api_key().unwrap_or_default();

    let api_base = config.get_api_base();
    info!(
        "create_provider: using OpenAICompatProvider (model={}, base={:?})",
        model, api_base
    );
    factory::create_openai_compat(factory::ProviderSpec {
        api_key,
        api_base,
        model: Some(model),
        jit_gate: None,
        retry: config.retry.clone(),
        timeout_secs: config.timeouts.provider_http_secs,
        lms_native_probe_secs: config.timeouts.lms_native_probe_secs,
        constrained_tool_calls: config.agents.defaults.constrained_tool_calls,
        higgs_session_cache: false,
        repetition_penalty,
        frequency_penalty,
        presence_penalty,
    })
}

/// Check that an LLM API key is configured, exit with error if not.
///
/// Allows through if OAuth credentials exist at `~/.claude/.credentials.json`
/// (Claude Max auto-detection), if a provider-prefix model is configured,
/// or if local mode is active (localApiBase set).
pub(crate) fn check_api_key(config: &Config) {
    let model = &config.agents.defaults.model;
    let has_prefix = config.resolve_provider_for_model(model).is_some();
    let local_env = std::env::var("NANOBOT_LOCAL")
        .map(|v| v == "1" || v.to_lowercase() == "true")
        .unwrap_or(false);
    let has_local = local_env || !config.agents.defaults.local_api_base.is_empty();
    if config.get_api_key().is_none() && !has_prefix && !has_local && !model.starts_with("bedrock/")
    {
        eprintln!("Error: No API key configured.");
        eprintln!(
            "Set one in ~/.nanobot/config.json, e.g. providers.openai.apiKey or providers.openrouter.apiKey"
        );
        std::process::exit(1);
    }
}
