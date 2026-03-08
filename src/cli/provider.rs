//! Cloud provider creation, OAuth loading, and API key validation.

use std::sync::Arc;

use crate::config::schema::Config;
use crate::providers::base::LLMProvider;
use crate::providers::factory;
use crate::providers::oauth::OAuthTokenManager;

/// Load a direct AnthropicProvider using OAuth tokens from Claude CLI.
///
/// Reads the access token from `~/.claude/.credentials.json`, refreshes if
/// needed, and returns an `AnthropicProvider` with OAuth identity headers
/// (same approach as OpenClaw -- direct API, no proxy, no CLI subprocess).
fn load_oauth_provider(sub_model: &str) -> anyhow::Result<Arc<dyn LLMProvider>> {
    use tracing::info;

    let mut mgr = OAuthTokenManager::load()?;

    // Refresh synchronously at startup if needed (tokens last ~8h, so
    // refreshing once at provider creation is sufficient for most sessions).
    let rt = tokio::runtime::Handle::try_current();
    let token = if let Ok(handle) = rt {
        tokio::task::block_in_place(|| handle.block_on(mgr.access_token()))?
    } else {
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(mgr.access_token())?
    };

    info!(
        "create_provider: AnthropicProvider (OAuth direct, model={}, token={}...)",
        sub_model,
        &token[..token.len().min(20)]
    );

    Ok(factory::create_anthropic(&token, Some(sub_model)))
}

pub(crate) fn create_provider(config: &Config) -> Arc<dyn LLMProvider> {
    use tracing::info;
    let model = &config.agents.defaults.model;

    // "claude-max" or "claude-max/opus" -> OAuth token from Claude CLI credentials.
    if model.starts_with("claude-max") {
        let sub_model = model
            .strip_prefix("claude-max/")
            .or_else(|| model.strip_prefix("claude-max"))
            .filter(|s| !s.is_empty())
            .unwrap_or("claude-opus-4-6");
        match load_oauth_provider(sub_model) {
            Ok(provider) => return provider,
            Err(e) => {
                eprintln!("Error: Failed to load Claude Max OAuth credentials: {}", e);
                eprintln!("Make sure you're logged into Claude CLI: claude login");
                std::process::exit(1);
            }
        }
    }

    // Claude model + no Anthropic API key + OAuth credentials exist -> use OAuth.
    // This catches cases where other providers (OpenAI, Groq, etc.) have keys set
    // but the user wants to use their Claude Max subscription for Claude models.
    if is_claude_model(model)
        && config.providers.anthropic.api_key.is_empty()
        && has_oauth_credentials()
    {
        info!(
            "create_provider: Claude model '{}' with no Anthropic API key, using OAuth",
            model
        );
        match load_oauth_provider(model) {
            Ok(provider) => return provider,
            Err(e) => {
                info!(
                    "OAuth fallback failed ({}), continuing with key-based provider",
                    e
                );
            }
        }
    }

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
        });
    }

    let api_key = config.get_api_key().unwrap_or_default();

    // No API key configured at all -> try OAuth as last resort.
    if api_key.is_empty() && has_oauth_credentials() {
        info!(
            "create_provider: no API key configured, falling back to OAuth for model={}",
            model
        );
        match load_oauth_provider(model) {
            Ok(provider) => return provider,
            Err(e) => {
                info!("OAuth fallback failed ({}), continuing with empty key", e);
            }
        }
    }

    let api_base = config.get_api_base();
    info!(
        "create_provider: using OpenAICompatProvider (model={}, base={:?})",
        model, api_base
    );
    factory::create_openai_compat(factory::ProviderSpec {
        api_key,
        api_base,
        model: Some(model.to_string()),
        jit_gate: None,
        retry: config.retry.clone(),
        timeout_secs: config.timeouts.provider_http_secs,
        lms_native_probe_secs: config.timeouts.lms_native_probe_secs,
    })
}

/// Check if a model name refers to a Claude/Anthropic model.
///
/// Handles both bare names (`claude-opus-4-5`) and prefixed names
/// (`anthropic/claude-opus-4-5`).
pub(super) fn is_claude_model(model: &str) -> bool {
    let m = model.to_lowercase();
    // Strip provider prefix if present (e.g. "anthropic/claude-opus-4-5" -> "claude-opus-4-5")
    let base = m.rsplit('/').next().unwrap_or(&m);
    base.starts_with("claude")
        || base.starts_with("opus")
        || base.starts_with("sonnet")
        || base.starts_with("haiku")
}

/// Check if Claude CLI OAuth credentials exist on disk.
pub(super) fn has_oauth_credentials() -> bool {
    dirs::home_dir()
        .map(|h| h.join(".claude").join(".credentials.json").exists())
        .unwrap_or(false)
}

/// Check that an LLM API key is configured, exit with error if not.
///
/// Allows through if OAuth credentials exist at `~/.claude/.credentials.json`
/// (Claude Max auto-detection).
pub(crate) fn check_api_key(config: &Config) {
    let model = &config.agents.defaults.model;
    if config.get_api_key().is_none()
        && !model.starts_with("bedrock/")
        && !model.starts_with("claude-max")
        && !has_oauth_credentials()
    {
        eprintln!("Error: No API key configured.");
        eprintln!("Set one in ~/.nanobot/config.json under providers.openrouter.apiKey");
        eprintln!("Or authenticate with Claude CLI: claude login");
        std::process::exit(1);
    }
}
