//! Core handle construction, agent loop creation, and local provider wiring.

use std::sync::atomic::Ordering;
use std::sync::Arc;

use tokio::sync::mpsc;

use super::*;
use crate::agent::agent_loop::{
    build_swappable_core, AgentHandle, AgentLoop, RuntimeCounters, SharedCoreHandle,
    SwappableCoreConfig,
};
use crate::bus::events::{InboundMessage, OutboundMessage};
use crate::config::schema::{AdaptiveTokenConfig, Config};
use crate::cron::service::CronService;
use crate::providers::base::LLMProvider;
use crate::providers::factory;
use crate::providers::jit_gate::JitGate;

/// MLX provider handle, kept alive alongside the agent loop so the model worker
/// thread persists and can be shared with the perplexity gate.
#[cfg(feature = "mlx")]
pub(crate) struct MlxHandle {
    pub provider: Arc<crate::providers::mlx::MlxProvider>,
}

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
    if is_local {
        // Local models: cap at 15 to preserve limited context.
        configured.min(15)
    } else {
        // Cloud models: scale up with context. Each tool iteration uses
        // ~500-1500 tokens on average, so even 50 iterations at 1M context
        // is only ~5% of the budget.
        // ~25 at 128K, ~40 at 500K, ~50 at 1M+
        let context_scaled = (max_context_tokens / 4000).min(50) as u32;
        configured.max(context_scaled)
    }
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
    } else {
        config_default
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
        custom.clone()
    } else {
        format!("http://localhost:{}/v1", fallback_port)
    }
}

/// Resolved local providers for all roles (main, compaction, delegation, specialist).
pub(super) struct LocalProviders {
    pub main: Arc<dyn LLMProvider>,
    pub model_id: String,
    pub compaction: Option<Arc<dyn LLMProvider>>,
    pub delegation: Option<Arc<dyn LLMProvider>>,
    pub specialist: Option<Arc<dyn LLMProvider>>,
    pub max_context_tokens: usize,
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
    let model_id = strip_gguf_suffix(local_model_name.unwrap_or("local-model")).to_string();

    // Create JIT gate for JIT-loading servers (e.g. LM Studio).
    // All providers sharing the same JIT endpoint get the same gate so requests
    // are serialised -- prevents concurrent model switches that crash the server.
    // Skip when lms CLI pre-loads models (skip_jit_gate = true).
    let jit_gate: Option<Arc<JitGate>> = if has_custom_base && !config.agents.defaults.skip_jit_gate
    {
        Some(Arc::new(JitGate::new()))
    } else {
        None
    };

    let main: Arc<dyn LLMProvider> = factory::create_openai_compat(
        factory::ProviderSpec::local(&base_url, Some(&model_id))
            .with_jit_gate_opt(jit_gate.clone())
            .with_timeout_config(&config.timeouts)
            .with_retry(config.retry.clone()),
    );

    // Auto-detect context size from local server; fall back to config default.
    let max_context_tokens = if !has_custom_base {
        crate::server::query_local_context_size(local_port)
            .unwrap_or(config.agents.defaults.local_max_context_tokens)
    } else {
        config.agents.defaults.local_max_context_tokens
    };

    // Compaction provider (separate port only).
    let compaction: Option<Arc<dyn LLMProvider>> =
        compaction_port.map(|p| -> Arc<dyn LLMProvider> {
            factory::create_openai_compat(
                factory::ProviderSpec::local(&local_base_url(config, p), None)
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
                api_key: role_name.to_string(),
                api_base: Some(ep.url.clone()),
                model: Some(ep.model.clone()),
                jit_gate: gate,
                retry: config.retry.clone(),
                timeout_secs: config.timeouts.provider_http_secs,
                lms_native_probe_secs: config.timeouts.lms_native_probe_secs,
            }));
        }

        // Priority 2: shared JIT server (localApiBase set) + trio model name
        if has_custom_base {
            let model = if !trio_model.is_empty() {
                trio_model
            } else {
                role_name
            };
            return Some(factory::create_openai_compat(
                factory::ProviderSpec::local(&base_url, Some(model))
                    .with_jit_gate_opt(jit_gate.clone())
                    .with_timeout_config(&config.timeouts)
                    .with_retry(config.retry.clone()),
            ));
        }

        // Priority 3: separate port fallback
        fallback_port.map(|p| -> Arc<dyn LLMProvider> {
            factory::create_openai_compat(
                factory::ProviderSpec::local(&local_base_url(config, p), Some(role_name))
                    .with_timeout_config(&config.timeouts)
                    .with_retry(config.retry.clone()),
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
        model_id,
        compaction,
        delegation,
        specialist,
        max_context_tokens,
    }
}

/// Resolve the MLX model directory from config or default location.
///
/// When `mlxModelDir` is `"auto"`, scans `~/.cache/lm-studio/models/` for
/// directories containing `tokenizer.json` + `*.safetensors` (no `.gguf`).
#[cfg(feature = "mlx")]
pub(crate) fn resolve_mlx_model_dir(config: &Config) -> std::path::PathBuf {
    if let Some(ref dir) = config.agents.defaults.mlx_model_dir {
        if dir == "auto" {
            if let Some(found) = auto_detect_mlx_model_dir() {
                return found;
            }
            tracing::warn!("mlxModelDir=auto but no MLX model found, using default");
        } else {
            return std::path::PathBuf::from(dir);
        }
    }
    dirs::home_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join(".cache/lm-studio/models/mlx-community/Qwen3.5-2B-MLX-8bit")
}

/// Scan `~/.cache/lm-studio/models/` for an MLX model directory.
///
/// An MLX dir has `tokenizer.json` + at least one `.safetensors` file and no `.gguf`.
#[cfg(feature = "mlx")]
pub(crate) fn auto_detect_mlx_model_dir() -> Option<std::path::PathBuf> {
    let base = dirs::home_dir()?.join(".cache/lm-studio/models");
    if !base.is_dir() {
        return None;
    }
    find_mlx_dir_recursive(&base)
}

/// Recursively find a directory that looks like an MLX model.
#[cfg(feature = "mlx")]
fn find_mlx_dir_recursive(dir: &std::path::Path) -> Option<std::path::PathBuf> {
    let entries = std::fs::read_dir(dir).ok()?;
    let mut subdirs = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            // Check if this dir is an MLX model dir
            if is_mlx_model_dir(&path) {
                return Some(path);
            }
            subdirs.push(path);
        }
    }
    // Recurse into subdirs
    for sub in subdirs {
        if let Some(found) = find_mlx_dir_recursive(&sub) {
            return Some(found);
        }
    }
    None
}

/// Re-export from mlx_lm — single canonical check for MLX model directories.
#[cfg(feature = "mlx")]
use crate::agent::mlx_lm::is_mlx_model_dir;

/// Infer the MLX preset name from a model directory path.
///
/// Matches known model names in the directory path.
#[cfg(feature = "mlx")]
pub(crate) fn preset_from_model_dir(dir: &std::path::Path) -> &'static str {
    let s = dir.to_string_lossy();
    let s_lower = s.to_lowercase();
    if s_lower.contains("qwen3.5") {
        "qwen3.5-2b"
    } else if s_lower.contains("qwen3-8b") || s_lower.contains("qwen3-8_b") {
        "qwen3-8b"
    } else if s_lower.contains("qwen3-4b") || s_lower.contains("qwen3-4_b") {
        "qwen3-4b"
    } else if s_lower.contains("qwen3-1.7b") || s_lower.contains("qwen3-1_7b") {
        "qwen3-1.7b"
    } else if s_lower.contains("qwen3-0.6b") || s_lower.contains("qwen3-0_6b") {
        "qwen3-1.7b" // closest architecture (standard Qwen3 transformer)
    } else {
        // Non-Qwen or unknown model — in-process loading will fail gracefully,
        // mlx-lm server handles inference for any model.
        "unknown"
    }
}

/// Start the in-process MLX provider. Returns the handle and an Arc provider
/// for use as the main LLM provider.
#[cfg(feature = "mlx")]
pub(crate) fn start_mlx_provider(config: &Config) -> anyhow::Result<MlxHandle> {
    use crate::agent::mlx_lora::{LoraConfig, ModelConfig};

    let model_dir = resolve_mlx_model_dir(config);
    // Always auto-detect preset from model dir name to avoid stale config mismatches.
    let effective_preset = preset_from_model_dir(&model_dir).to_string();
    let model_config = match effective_preset.as_str() {
        "qwen3-1.7b" => ModelConfig::qwen3_1_7b(),
        "qwen3-4b" => ModelConfig::qwen3_4b(),
        "qwen3-8b" => ModelConfig::qwen3_8b(),
        "qwen3.5-2b" => ModelConfig::qwen3_5_2b(),
        _ => {
            // Unknown model — use Qwen3-1.7B as a placeholder; in-process load
            // will fail gracefully and mlx-lm server handles inference.
            ModelConfig::qwen3_1_7b()
        }
    };
    let lora_config = LoraConfig {
        lr: 1e-5, // mlx_lm default; 5e-4 causes NaN on real sequences
        ..LoraConfig::default()
    };

    tracing::info!(
        model_dir = %model_dir.display(),
        preset = %config.agents.defaults.mlx_preset,
        "starting in-process MLX provider"
    );

    let mlx_lm_url = config.agents.defaults.mlx_lm_url.clone();
    let provider = crate::providers::mlx::MlxProvider::start_with_mlx_lm(
        model_dir, model_config, lora_config, mlx_lm_url,
    )?;
    Ok(MlxHandle {
        provider: Arc::new(provider),
    })
}

/// Build a `SwappableCoreConfig` from shared config + per-call overrides.
///
/// Centralises the 25-field struct construction that was previously copy-pasted
/// across `build_core_handle`, `build_core_handle_mlx`, `rebuild_core`, and
/// `rebuild_core_mlx`.
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
        search_max_results: config.tools.web.search.max_results,
        exec_timeout: config.tools.exec_.timeout,
        restrict_to_workspace: config.tools.exec_.restrict_to_workspace,
        memory_config: config.memory.clone(),
        is_local,
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
    }
}

/// Resolve MLX context cap: full config value when mlx-lm server handles
/// inference, capped to 4K for in-process (GDN recurrence is slow).
#[cfg(feature = "mlx")]
fn mlx_max_context(config: &Config) -> usize {
    if config.agents.defaults.mlx_lm_url.is_some() {
        config.agents.defaults.local_max_context_tokens
    } else {
        config.agents.defaults.local_max_context_tokens.min(4096)
    }
}

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
        let model = format!("local:{}", lp.model_id);
        (
            lp.main,
            model,
            lp.max_context_tokens,
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
        config, provider, model, max_context_tokens, is_local, cp, dp, sp,
    ));
    let counters = Arc::new(RuntimeCounters::new_with_config(
        max_context_tokens,
        &config.trio.circuit_breaker,
    ));
    // When main_no_think is enabled, also suppress thinking display from the start
    // so the user doesn't need to run /nothink manually each session.
    if config.trio.main_no_think {
        counters
            .suppress_thinking_in_tts
            .store(true, Ordering::Relaxed);
    }
    AgentHandle::new(core, counters)
}

/// Build a core handle using the in-process MLX provider as the main LLM.
///
/// The MLX provider runs inference on Apple Silicon GPU via the same model
/// that serves perplexity scoring and LoRA training. Context is limited to
/// 32K tokens (local model default).
///
/// `is_local` is set to `false` because MLX speaks proper tool_calls and
/// does not need the local protocol quirks (user-last, no prefill).
#[cfg(feature = "mlx")]
pub(crate) fn build_core_handle_mlx(
    config: &Config,
    mlx: &MlxHandle,
) -> SharedCoreHandle {
    let provider: Arc<dyn LLMProvider> = mlx.provider.clone();
    let model = format!("mlx:{}", provider.get_default_model());
    let max_context_tokens = mlx_max_context(config);

    let core = build_swappable_core(core_config_from(
        config, provider, model, max_context_tokens, true, None, None, None,
    ));
    let counters = Arc::new(RuntimeCounters::new_with_config(
        max_context_tokens,
        &config.trio.circuit_breaker,
    ));
    AgentHandle::new(core, counters)
}

/// Rebuild the shared core for MLX mode (e.g. `/ctx` or `/model` changes).
///
/// Like `rebuild_core` but uses the MLX provider instead of LM Studio.
#[cfg(feature = "mlx")]
pub(crate) fn rebuild_core_mlx(
    handle: &SharedCoreHandle,
    config: &Config,
    mlx: &MlxHandle,
) {
    let provider: Arc<dyn LLMProvider> = mlx.provider.clone();
    let model = format!("mlx:{}", provider.get_default_model());
    let max_context_tokens = mlx_max_context(config);

    let new_core = build_swappable_core(core_config_from(
        config, provider, model, max_context_tokens, true, None, None, None,
    ));
    handle.swap_core(new_core);
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
        let model = format!("local:{}", lp.model_id);
        (
            lp.main,
            model,
            lp.max_context_tokens,
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
        config, provider, model, max_context_tokens, is_local, cp, dp, sp,
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
    create_agent_loop_inner(core_handle, config, cron_service, email_config, repl_display_tx, health_registry, None)
}

/// Create an agent loop wired to an in-process MLX provider.
///
/// The MLX provider is set as the perplexity + training backend on the agent
/// loop, and the perplexity gate is auto-enabled.
#[cfg(feature = "mlx")]
pub(crate) fn create_agent_loop_mlx(
    core_handle: SharedCoreHandle,
    config: &Config,
    cron_service: Option<Arc<CronService>>,
    email_config: Option<crate::config::schema::EmailConfig>,
    repl_display_tx: Option<mpsc::UnboundedSender<String>>,
    health_registry: Option<Arc<crate::heartbeat::health::HealthRegistry>>,
    mlx: &MlxHandle,
) -> AgentLoop {
    create_agent_loop_inner(core_handle, config, cron_service, email_config, repl_display_tx, health_registry, Some(mlx))
}

fn create_agent_loop_inner(
    core_handle: SharedCoreHandle,
    config: &Config,
    cron_service: Option<Arc<CronService>>,
    email_config: Option<crate::config::schema::EmailConfig>,
    repl_display_tx: Option<mpsc::UnboundedSender<String>>,
    health_registry: Option<Arc<crate::heartbeat::health::HealthRegistry>>,
    #[cfg(feature = "mlx")] mlx: Option<&MlxHandle>,
    #[cfg(not(feature = "mlx"))] _mlx: Option<&()>,
) -> AgentLoop {
    let (inbound_tx, inbound_rx) = mpsc::unbounded_channel::<InboundMessage>();
    let (outbound_tx, _outbound_rx) = mpsc::unbounded_channel::<OutboundMessage>();

    let mut lcm_config = config.lcm.clone();
    if core_handle.swappable().is_local && !lcm_config.enabled {
        tracing::info!("Auto-enabling LCM for local mode");
        lcm_config.enabled = true;
    }

    let mut agent_loop = AgentLoop::new(
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

    // Wire MLX provider for in-process perplexity + training.
    #[cfg(feature = "mlx")]
    if let Some(mlx) = mlx {
        agent_loop.set_mlx_provider(mlx.provider.clone());
        // Auto-enable perplexity gate when using MLX engine.
        let mut gate_config = config.perplexity_gate.clone();
        gate_config.enabled = true;
        agent_loop.set_perplexity_gate(gate_config);
        tracing::info!("MLX provider wired: perplexity gate auto-enabled");
    } else if config.perplexity_gate.enabled {
        agent_loop.set_perplexity_gate(config.perplexity_gate.clone());
    }

    #[cfg(not(feature = "mlx"))]
    if config.perplexity_gate.enabled {
        agent_loop.set_perplexity_gate(config.perplexity_gate.clone());
    }

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
