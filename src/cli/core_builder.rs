//! Core handle construction, agent loop creation, and local provider wiring.

use std::sync::atomic::Ordering;
use std::sync::Arc;

use tokio::sync::mpsc;

use super::*;
use crate::agent::agent_core::{
    build_swappable_core, AgentHandle, RuntimeCounters, SwappableCoreConfig,
};
use crate::agent::agent_loop::{AgentLoop, SharedCoreHandle};
use crate::agent::lane::Lane;
use crate::bus::events::{InboundMessage, OutboundMessage};
use crate::config::schema::{is_higgs_backend, AdaptiveTokenConfig, Config};
use crate::cron::service::CronService;
use crate::providers::base::LLMProvider;
use crate::providers::factory;
use crate::providers::jit_gate::JitGate;

/// MLX provider handle, kept alive alongside the agent loop so the model worker
/// thread persists and can be shared with the perplexity gate.

/// Return the appropriate context window size for a cloud model.
///
/// Scale max tool iterations with context size.
///
/// Cloud models with large context windows (128K+) can use many more tool
/// iterations without risking context exhaustion. Local models with tight
/// context (16K-32K) need a lower cap to leave room for the actual
/// conversation.
///
/// Returns the configured value unchanged when it's already appropriate,
/// or scales it up/down based on available context.
pub(crate) fn effective_max_iterations(
    configured: u32,
    max_context_tokens: usize,
    is_local: bool,
) -> u32 {
    // Scale the iteration budget with the context window for both local and
    // cloud: each tool round trip costs ~500-1500 tokens, so even a generous
    // cap is a small fraction of a large window. `configured` is the floor;
    // context scaling only ever raises it.
    //
    // Local models decode slower and pay more for context growth, so they get a
    // larger divisor and a lower ceiling than cloud — but no longer a flat
    // `min(15)` that cut off long tool chains while the context sat near-empty
    // (the failure this fixes: a local turn capped at 15 iterations with context
    // at 1% used).
    //   cloud: ~32 at 128K, ~40 at 160K, 50 at 200K+
    //   local: ~25 at 128K, 30 at 150K+
    let (divisor, ceiling) = if is_local { (5000, 30) } else { (4000, 50) };
    let context_scaled = (max_context_tokens / divisor).min(ceiling) as u32;
    configured.max(context_scaled)
}

/// Models with known large context windows get their full capacity;
/// everything else uses the config default (128K).
pub(super) fn model_context_size(model: &str, config_default: usize) -> usize {
    let m = model.to_lowercase();
    if m.contains("opus") || m.contains("sonnet") || m.contains("claude") {
        // Claude 4.x family: 1M token context
        config_default.max(1_000_000)
    } else if m.contains("gemini") {
        config_default.max(1_000_000)
    } else if m.contains("qwen3.6") || m.contains("qwen36") || m.contains("qwen3.5") {
        // Qwen3.5/3.6 (incl. A3B MoE) ship 256K native context. Cap at 128K to
        // bound local decode-slowdown at high fill while leaving ample history
        // room — its hybrid arch only grows KV on full-attention layers, so long
        // context is cheap here. Smaller local models (e.g. Bonsai 32K/64K) keep
        // the conservative config default and are not overshot.
        config_default.max(131_072)
    } else {
        config_default
    }
}

const APPLE_FM_CONTEXT_TOKENS: usize = 4_096;

fn is_apple_fm_model(model: &str) -> bool {
    matches!(model.trim().to_ascii_lowercase().as_str(), "system" | "pcc")
}

fn resolved_local_context_tokens(
    model_id: &str,
    detected_context_tokens: Option<usize>,
    fallback_context_tokens: usize,
) -> usize {
    if is_apple_fm_model(model_id) {
        detected_context_tokens
            .unwrap_or(APPLE_FM_CONTEXT_TOKENS)
            .min(APPLE_FM_CONTEXT_TOKENS)
    } else {
        detected_context_tokens.unwrap_or(fallback_context_tokens)
    }
}

/// Strip GGUF quantisation suffix and extension from a model filename to get
/// the bare model identifier that LM Studio / remote servers recognise.
///
/// Examples:
///   "nanbeige4.1-3b-q8_0.gguf"          -> "nanbeige4.1-3b"
///   "Qwen3-8B-Q4_K_M.gguf"              -> "Qwen3-8B"
///   "ministral-3-8b-instruct-2512.gguf"  -> "ministral-3-8b-instruct-2512"
///   "nanbeige4.1-3b"                     -> "nanbeige4.1-3b" (no-op)
pub(crate) fn strip_gguf_suffix(name: &str) -> &str {
    let name = name.strip_suffix(".gguf").unwrap_or(name);
    // Match common quant patterns: -q8_0, -Q4_K_M, -IQ2_XS, -f16, -f32, etc.
    // Pattern: last segment starting with `-[qQfFiI]` followed by digits/underscores/letters.
    // Minimum 3 chars to avoid stripping model variant suffixes like `-i1` (imatrix).
    if let Some(idx) = name.rfind('-') {
        let suffix = &name[idx + 1..];
        let first = suffix.as_bytes().first().copied().unwrap_or(0);
        if matches!(first, b'q' | b'Q' | b'f' | b'F' | b'i' | b'I')
            && suffix.len() >= 3
            && suffix.as_bytes()[1].is_ascii_digit()
        {
            return &name[..idx];
        }
    }
    name
}

/// Resolve the API base URL for a local role.
///
/// When `localApiBase` is set in config, ALL local providers share that URL
/// (LM Studio JIT loading differentiates by model name, not by port).
/// Otherwise falls back to `http://localhost:{port}/v1`.
pub(super) fn local_base_url(config: &Config, fallback_port: &str) -> String {
    let custom = &config.agents.defaults.local_api_base;
    if !custom.is_empty() {
        let trimmed = custom.trim_end_matches('/');
        if trimmed.ends_with("/v1") {
            trimmed.to_string()
        } else {
            format!("{trimmed}/v1")
        }
    } else {
        format!("http://localhost:{}/v1", fallback_port)
    }
}

/// Resolved local providers for all roles (main, compaction, delegation, specialist).
pub(super) struct LocalProviders {
    pub main: Arc<dyn LLMProvider>,
    /// Real model identity used for capabilities, prompt policy, UI, and snapshots.
    pub semantic_model_id: String,
    pub compaction: Option<Arc<dyn LLMProvider>>,
    pub delegation: Option<Arc<dyn LLMProvider>>,
    pub specialist: Option<Arc<dyn LLMProvider>>,
    pub max_context_tokens: usize,
}

fn shared_local_role_model<'a>(configured_role_model: &'a str, main_model_id: &'a str) -> &'a str {
    if configured_role_model.is_empty() {
        main_model_id
    } else {
        configured_role_model
    }
}

fn local_transport_model_id(config: &Config, local_model_name: Option<&str>) -> String {
    let configured = local_model_name
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .or_else(|| {
            let configured = config.agents.defaults.lms_main_model.trim();
            (!configured.is_empty()).then_some(configured)
        })
        .or_else(|| {
            let configured = config.agents.defaults.local_model.trim();
            (!configured.is_empty()).then_some(configured)
        })
        .unwrap_or("local-model");

    strip_gguf_suffix(configured).to_string()
}

fn local_semantic_model_id(config: &Config, transport_model_id: &str) -> String {
    if is_higgs_backend(&config.agents.defaults.local_backend) && transport_model_id == "active" {
        let configured = config.agents.defaults.local_model.trim();
        if !configured.is_empty() {
            return strip_gguf_suffix(configured).to_string();
        }
        if let Some(model_dir) = config.agents.defaults.mlx_model_dir.as_deref() {
            if let Some(name) = std::path::Path::new(model_dir)
                .file_name()
                .and_then(|name| name.to_str())
                .filter(|name| !name.is_empty())
            {
                return name.to_string();
            }
        }
    }
    transport_model_id.to_string()
}

/// Build providers for all local roles from config + endpoint resolution.
///
/// Endpoint priority per trio role:
///   1. `trio.router_endpoint` / `trio.specialist_endpoint` (explicit URL+model)
///   2. `localApiBase` + `trio.router_model` / `trio.specialist_model` (shared JIT server)
///   3. Separate port fallback (delegation_port / specialist_port)
///   4. None (disabled)
pub(super) fn make_local_providers(
    config: &Config,
    local_port: &str,
    local_model_name: Option<&str>,
    compaction_port: Option<&str>,
    delegation_port: Option<&str>,
    specialist_port: Option<&str>,
) -> LocalProviders {
    let has_custom_base = !config.agents.defaults.local_api_base.is_empty();
    let base_url = local_base_url(config, local_port);

    // Resolve main model name.
    // Always strip GGUF suffix -- config may hold a .gguf filename even when
    // using LM Studio, which expects clean identifiers.
    let model_id = local_transport_model_id(config, local_model_name);
    let semantic_model_id = local_semantic_model_id(config, &model_id);

    // Create JIT gate only for JIT-loading servers (LM Studio). Resident
    // servers (oMLX, Higgs) keep models loaded in memory; serialising their
    // requests behind a single-permit gate only stalls main/compaction/memory
    // ops without preventing any eviction — and serialised cold reloads are
    // exactly the 89–122s prefill outliers we saw.
    // Skip when lms CLI pre-loads models (skip_jit_gate = true).
    let is_jit_server = !is_higgs_backend(&config.agents.defaults.local_backend);
    let jit_gate: Option<Arc<JitGate>> =
        if has_custom_base && is_jit_server && !config.agents.defaults.skip_jit_gate {
            Some(Arc::new(JitGate::new()))
        } else {
            None
        };

    let api_key = &config.agents.defaults.local_api_key;
    let constrained = config.agents.defaults.constrained_tool_calls;

    let main: Arc<dyn LLMProvider> = factory::create_openai_compat(
        factory::ProviderSpec::local_with_key(&base_url, Some(&model_id), api_key)
            .with_jit_gate_opt(jit_gate.clone())
            .with_timeout_config(&config.timeouts)
            .with_retry(config.retry.clone())
            .with_higgs_session_cache(is_higgs_backend(&config.agents.defaults.local_backend)),
    );

    // Auto-detect context size from the active server; fall back to config default.
    // The cluster path (custom base, possibly remote) needs a URL-aware probe so
    // peers exposing /props (llama-server) get their real n_ctx instead of the
    // 32k schema default.
    let detected_context_tokens = if has_custom_base {
        crate::server::query_context_size_from_url(&base_url)
    } else {
        crate::server::query_local_context_size(local_port)
    };
    let max_context_tokens = resolved_local_context_tokens(
        &model_id,
        detected_context_tokens,
        config.agents.defaults.local_max_context_tokens,
    );

    // Compaction provider (separate port only).
    let compaction: Option<Arc<dyn LLMProvider>> =
        compaction_port.map(|p| -> Arc<dyn LLMProvider> {
            factory::create_openai_compat(
                factory::ProviderSpec::local_with_key(&local_base_url(config, p), None, api_key)
                    .with_jit_gate_opt(jit_gate.clone())
                    .with_timeout_config(&config.timeouts)
                    .with_retry(config.retry.clone()),
            )
        });

    // Helper: create a provider for a trio role with endpoint resolution.
    let make_role_provider = |role_name: &str,
                              endpoint: &Option<crate::config::schema::ModelEndpoint>,
                              trio_model: &str,
                              fallback_port: Option<&str>|
     -> Option<Arc<dyn LLMProvider>> {
        // Priority 1: explicit endpoint (url + model)
        if let Some(ep) = endpoint {
            // Use JIT gate if endpoint URL matches the shared base (same server).
            let gate = jit_gate.as_ref().filter(|_| ep.url == base_url).cloned();
            return Some(factory::create_openai_compat(factory::ProviderSpec {
                api_key: api_key.to_string(),
                api_base: Some(ep.url.clone()),
                model: Some(ep.model.clone()),
                jit_gate: gate,
                retry: config.retry.clone(),
                timeout_secs: config.timeouts.provider_http_secs,
                lms_native_probe_secs: config.timeouts.lms_native_probe_secs,
                constrained_tool_calls: constrained,
                higgs_session_cache: false,
            }));
        }

        // Priority 2: shared JIT server (localApiBase set) + trio model name
        if has_custom_base {
            let model = shared_local_role_model(trio_model, &model_id);
            if trio_model.is_empty() {
                tracing::warn!(
                    role = role_name,
                    model = %model,
                    "No local role model configured; reusing main local model"
                );
            }
            return Some(factory::create_openai_compat(
                factory::ProviderSpec::local_with_key(&base_url, Some(model), api_key)
                    .with_jit_gate_opt(jit_gate.clone())
                    .with_timeout_config(&config.timeouts)
                    .with_retry(config.retry.clone())
                    .with_constrained_tool_calls(constrained),
            ));
        }

        // Priority 3: separate port fallback
        fallback_port.map(|p| -> Arc<dyn LLMProvider> {
            factory::create_openai_compat(
                factory::ProviderSpec::local_with_key(
                    &local_base_url(config, p),
                    Some(role_name),
                    api_key,
                )
                .with_timeout_config(&config.timeouts)
                .with_retry(config.retry.clone())
                .with_constrained_tool_calls(constrained),
            )
        })
    };

    let delegation = if config.tool_delegation.enabled || config.trio.enabled {
        make_role_provider(
            "local-delegation",
            &config.trio.router_endpoint,
            &config.trio.router_model,
            delegation_port,
        )
    } else {
        None
    };

    let specialist = if config.trio.enabled {
        make_role_provider(
            "local-specialist",
            &config.trio.specialist_endpoint,
            &config.trio.specialist_model,
            specialist_port,
        )
    } else {
        None
    };

    LocalProviders {
        main,
        semantic_model_id,
        compaction,
        delegation,
        specialist,
        max_context_tokens,
    }
}

/// Build a `SwappableCoreConfig` from shared config + per-call overrides.
///
/// Centralises the 25-field struct construction that was previously copy-pasted
/// across `build_core_handle` and `rebuild_core`.
fn core_config_from(
    config: &Config,
    provider: Arc<dyn LLMProvider>,
    model: String,
    max_context_tokens: usize,
    is_local: bool,
    compaction: Option<Arc<dyn LLMProvider>>,
    delegation: Option<Arc<dyn LLMProvider>>,
    specialist: Option<Arc<dyn LLMProvider>>,
) -> SwappableCoreConfig {
    let lane = config
        .agents
        .default_lane
        .as_deref()
        .and_then(|s| s.parse::<Lane>().ok())
        .unwrap_or_default();
    let brave_key = if config.tools.web.search.api_key.is_empty() {
        None
    } else {
        Some(config.tools.web.search.api_key.clone())
    };
    let max_iters = effective_max_iterations(
        config.agents.defaults.max_tool_iterations,
        max_context_tokens,
        is_local,
    );
    SwappableCoreConfig {
        provider,
        workspace: config.workspace_path(),
        model,
        max_iterations: max_iters,
        max_continuations: config.agents.defaults.max_continuations,
        max_tokens: config.agents.defaults.max_tokens,
        temperature: config.agents.defaults.temperature,
        max_context_tokens,
        brave_api_key: brave_key,
        search_provider: config.tools.web.search.provider.clone(),
        searxng_url: config.tools.web.search.searxng_url.clone(),
        crw_url: config.tools.web.fetch.crw_url.clone(),
        search_max_results: config.tools.web.search.max_results,
        exec_timeout: config.tools.exec_.timeout,
        restrict_to_workspace: config.tools.exec_.restrict_to_workspace,
        memory_config: config.memory.clone(),
        is_local,
        local_tool_mode: config.tools.local_tool_mode.clone(),
        lane,
        compaction_provider: compaction,
        tool_delegation: config.tool_delegation.clone(),
        provenance: config.provenance.clone(),
        max_tool_result_chars: config.agents.defaults.max_tool_result_chars,
        delegation_provider: delegation,
        specialist_provider: specialist,
        trio_config: config.trio.clone(),
        model_capabilities_overrides: config.model_capabilities.clone(),
        reasoning_config: config.reasoning.clone(),
        tool_heartbeat_secs: config.monitoring.tool_heartbeat_secs,
        health_check_timeout_secs: config.monitoring.health_check_timeout_secs,
        adaptive_tokens: AdaptiveTokenConfig::from_defaults(&config.agents.defaults),
        sessions_db_path: None,
    }
}

/// Resolve MLX context size from config.

pub(crate) fn build_core_handle(
    config: &Config,
    local_port: &str,
    local_model_name: Option<&str>,
    compaction_port: Option<&str>,
    delegation_port: Option<&str>,
    specialist_port: Option<&str>,
    is_local: bool,
) -> SharedCoreHandle {
    let (provider, model, max_context_tokens, cp, dp, sp) = if is_local {
        let lp = make_local_providers(
            config,
            local_port,
            local_model_name,
            compaction_port,
            delegation_port,
            specialist_port,
        );
        let model = format!("local:{}", lp.semantic_model_id);
        // Size context per-model on the local path too (not just cloud): a
        // capable long-context model (e.g. Qwen3.6, 256K native) gets its real
        // budget, while smaller local models (Bonsai 32K/64K) keep the
        // conservative server-probed/default value and are never overshot.
        let ctx = model_context_size(&lp.semantic_model_id, lp.max_context_tokens);
        (
            lp.main,
            model,
            ctx,
            lp.compaction,
            lp.delegation,
            lp.specialist,
        )
    } else {
        let provider = create_provider(config);
        let model = config.agents.defaults.model.clone();
        let ctx = model_context_size(&model, config.agents.defaults.max_context_tokens);
        (provider, model, ctx, None, None, None)
    };

    let core = build_swappable_core(core_config_from(
        config,
        provider,
        model,
        max_context_tokens,
        is_local,
        cp,
        dp,
        sp,
    ));
    let counters =
        RuntimeCounters::new_with_config(max_context_tokens, &config.trio.circuit_breaker);
    // When main_no_think is enabled, suppress thinking display from the start
    // so the user doesn't need to run /nothink manually each session.
    if config.trio.main_no_think {
        counters
            .suppress_thinking_display
            .store(true, Ordering::Relaxed);
    }
    // Attach lazy auxiliary server (spawns on first delegation/compaction/memory use).
    AgentHandle::new(core, Arc::new(counters))
}

/// Rebuild the shared core for `/local` toggle or `/model` swap.
///
/// All agents sharing this handle see the new provider on their next message.
pub(crate) fn rebuild_core(
    handle: &SharedCoreHandle,
    config: &Config,
    local_port: &str,
    local_model_name: Option<&str>,
    compaction_port: Option<&str>,
    delegation_port: Option<&str>,
    specialist_port: Option<&str>,
    is_local: bool,
) {
    let (provider, model, max_context_tokens, cp, dp, sp) = if is_local {
        let lp = make_local_providers(
            config,
            local_port,
            local_model_name,
            compaction_port,
            delegation_port,
            specialist_port,
        );
        let model = format!("local:{}", lp.semantic_model_id);
        // Size context per-model on the local path too (not just cloud): a
        // capable long-context model (e.g. Qwen3.6, 256K native) gets its real
        // budget, while smaller local models (Bonsai 32K/64K) keep the
        // conservative server-probed/default value and are never overshot.
        let ctx = model_context_size(&lp.semantic_model_id, lp.max_context_tokens);
        (
            lp.main,
            model,
            ctx,
            lp.compaction,
            lp.delegation,
            lp.specialist,
        )
    } else {
        let provider = create_provider(config);
        let model = config.agents.defaults.model.clone();
        let ctx = model_context_size(&model, config.agents.defaults.max_context_tokens);
        (provider, model, ctx, None, None, None)
    };

    let new_core = build_swappable_core(core_config_from(
        config,
        provider,
        model,
        max_context_tokens,
        is_local,
        cp,
        dp,
        sp,
    ));
    // Swap only the core; counters survive.
    handle.swap_core(new_core);
    // Update max context since the new model may have a different size.
    handle
        .counters
        .last_context_max
        .store(max_context_tokens as u64, Ordering::Relaxed);
    // Reset delegation health -- new core may have a fresh delegation server.
    handle
        .counters
        .delegation_healthy
        .store(true, Ordering::Relaxed);
    handle
        .counters
        .delegation_retry_counter
        .store(0, Ordering::Relaxed);
}

/// Create an agent loop with per-instance channels, using the shared core handle.
pub(crate) fn create_agent_loop(
    core_handle: SharedCoreHandle,
    config: &Config,
    cron_service: Option<Arc<CronService>>,
    email_config: Option<crate::config::schema::EmailConfig>,
    repl_display_tx: Option<mpsc::UnboundedSender<String>>,
    health_registry: Option<Arc<crate::heartbeat::health::HealthRegistry>>,
) -> AgentLoop {
    let (inbound_tx, inbound_rx) = mpsc::unbounded_channel::<InboundMessage>();
    let (outbound_tx, _outbound_rx) = mpsc::unbounded_channel::<OutboundMessage>();

    let mut lcm_config = config.lcm.clone();
    // Inject the local API key so the LCM compactor can authenticate with oMLX.
    lcm_config.api_key = config.agents.defaults.local_api_key.clone();
    // migrated from swappable().is_local — phase 09-03
    // LCM auto-enable: inline match (single site) — keep the dispatch local
    // to the one call rather than adding a RuntimeMode::lcm_auto() method.
    if core_handle.swappable().mode().is_local() && !lcm_config.is_enabled() {
        tracing::info!("Auto-enabling LCM for local mode");
        lcm_config.enabled = Some(true);
    }

    let agent_loop = AgentLoop::new(
        core_handle,
        inbound_rx,
        outbound_tx,
        inbound_tx,
        cron_service,
        config.agents.defaults.max_concurrent_chats,
        email_config,
        repl_display_tx,
        Some(config.providers.clone()),
        config.proprioception.clone(),
        lcm_config,
        health_registry,
    );

    agent_loop
}

/// Set up cluster discovery for REPL path. Returns the ClusterState so callers
/// can store it for /cluster command access.
///
/// Must be called after `create_agent_loop` -- attaches a `ClusterRouter` to the
/// existing agent loop and starts the background discovery task.
#[cfg(feature = "cluster")]
pub(crate) fn setup_cluster_for_repl(
    agent_loop: &mut AgentLoop,
    config: &Config,
) -> Option<Arc<crate::cluster::state::ClusterState>> {
    if !config.cluster.enabled {
        return None;
    }
    let cluster_state = crate::cluster::state::ClusterState::new();
    let discovery = crate::cluster::discovery::ClusterDiscovery::new(
        config.cluster.clone(),
        cluster_state.clone(),
        config.agents.defaults.local_api_key.clone(),
    );
    let _discovery_handle = discovery.run();
    tracing::info!("cluster_discovery_started");
    let router = Arc::new(crate::cluster::router::ClusterRouter::new(
        cluster_state.clone(),
        config.cluster.clone(),
    ));
    agent_loop.set_cluster_router(router);
    Some(Arc::new(cluster_state))
}

#[cfg(test)]
mod matching_tests {
    use super::*;

    #[test]
    fn test_shared_local_role_model_reuses_main_when_unconfigured() {
        assert_eq!(
            shared_local_role_model("", "Qwen3.6-35B-A3B-4bit"),
            "Qwen3.6-35B-A3B-4bit"
        );
    }

    #[test]
    fn test_shared_local_role_model_uses_configured_role_model() {
        assert_eq!(
            shared_local_role_model("Qwen3.5-0.8B-8bit", "Qwen3.6-35B-A3B-4bit"),
            "Qwen3.5-0.8B-8bit"
        );
    }

    #[test]
    fn test_resolved_local_context_tokens_caps_apple_fm() {
        assert_eq!(resolved_local_context_tokens("system", None, 32_768), 4_096);
        assert_eq!(
            resolved_local_context_tokens("pcc", Some(32_768), 32_768),
            4_096
        );
    }

    #[test]
    fn test_resolved_local_context_tokens_keeps_detected_non_apple_ctx() {
        assert_eq!(
            resolved_local_context_tokens("qwen36-35b", Some(131_072), 32_768),
            131_072
        );
        assert_eq!(
            resolved_local_context_tokens("qwen36-35b", None, 32_768),
            32_768
        );
    }
}
