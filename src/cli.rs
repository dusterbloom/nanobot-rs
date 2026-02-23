//! CLI subcommand handlers for nanobot.
//!
//! This module contains all command implementations that were previously in main.rs.
//! Functions are extracted here to keep main.rs focused on argument parsing and routing.

use std::io::{self, Write};
use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use tokio::sync::mpsc;

use crate::agent::agent_loop::{
    build_swappable_core, AgentHandle, AgentLoop, RuntimeCounters, SharedCoreHandle,
    SwappableCoreConfig,
};
use crate::agent::tuning::{score_feasible_profiles, select_optimal_from_input, OptimizationInput};
use crate::bus::events::{InboundMessage, OutboundMessage};
use crate::channels::manager::ChannelManager;
use crate::config::loader::{get_config_path, get_data_dir, load_config, save_config};
use crate::config::schema::Config;
use crate::cron::service::CronService;
use crate::cron::types::CronSchedule;
use crate::heartbeat::service::{
    HeartbeatService, DEFAULT_HEARTBEAT_INTERVAL_S, DEFAULT_MAINTENANCE_COMMANDS,
};
use crate::providers::base::LLMProvider;
use crate::providers::jit_gate::JitGate;
use crate::providers::oauth::OAuthTokenManager;
use crate::providers::factory;
use crate::utils::helpers::get_workspace_path;

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    #[ignore] // Requires network access to Telegram API
    fn test_validate_telegram_token_valid() {
        assert!(validate_telegram_token(
            "123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11"
        ));
    }

    #[test]
    fn test_validate_telegram_token_invalid() {
        assert!(!validate_telegram_token("invalid"));
        assert!(!validate_telegram_token("123456"));
        assert!(!validate_telegram_token(""));
    }

    #[test]
    fn test_run_tune_from_path_selects_best_profile() {
        let dir = tempdir().unwrap();
        let input_path = dir.path().join("bench.json");

        // Create a sample benchmark JSON with multiple measurements
        let json_data = r#"{
            "measurements": [
                {
                    "profile": {
                        "id": "small",
                        "model": "Qwen2.5-1.5B-Instruct",
                        "ctx_size": 8192,
                        "max_tokens": 768,
                        "temperature": 0.3
                    },
                    "sample": {
                        "ttft_ms": 120.0,
                        "output_toks_per_sec": 45.2,
                        "quality_score": 0.75,
                        "tool_success_rate": 0.90,
                        "context_overflow_rate": 0.0
                    }
                },
                {
                    "profile": {
                        "id": "medium",
                        "model": "Qwen2.5-7B-Instruct",
                        "ctx_size": 32768,
                        "max_tokens": 768,
                        "temperature": 0.3
                    },
                    "sample": {
                        "ttft_ms": 250.0,
                        "output_toks_per_sec": 28.5,
                        "quality_score": 0.85,
                        "tool_success_rate": 0.95,
                        "context_overflow_rate": 0.0
                    }
                }
            ]
        }"#;

        std::fs::write(&input_path, json_data).unwrap();

        let result = run_tune_from_path(&input_path, false);
        assert!(result.is_ok(), "Error: {:?}", result);
        let output = result.unwrap();
        assert!(output.contains("Best local profile"));
    }

    #[test]
    fn test_run_tune_from_path_json_output() {
        let dir = tempdir().unwrap();
        let input_path = dir.path().join("bench.json");

        let json_data = r#"{
            "measurements": [
                {
                    "profile": {
                        "id": "test",
                        "model": "Qwen2.5-1.5B-Instruct",
                        "ctx_size": 8192,
                        "max_tokens": 768,
                        "temperature": 0.3
                    },
                    "sample": {
                        "ttft_ms": 120.0,
                        "output_toks_per_sec": 45.2,
                        "quality_score": 0.80,
                        "tool_success_rate": 0.92,
                        "context_overflow_rate": 0.0
                    }
                }
            ]
        }"#;

        std::fs::write(&input_path, json_data).unwrap();

        let result = run_tune_from_path(&input_path, true);
        assert!(result.is_ok());
        let output = result.unwrap();
        // Should be valid JSON
        assert!(serde_json::from_str::<serde_json::Value>(&output).is_ok());
    }

    #[test]
    fn test_create_workspace_templates() {
        let dir = tempdir().unwrap();
        let workspace = dir.path();

        create_workspace_templates(workspace);

        // Check that expected files were created
        assert!(workspace.join("AGENTS.md").exists());
        assert!(workspace.join("SOUL.md").exists());
        assert!(workspace.join("USER.md").exists());
        assert!(workspace.join("memory").join("MEMORY.md").exists());

        // Verify content of one file
        let agents_content = std::fs::read_to_string(workspace.join("AGENTS.md")).unwrap();
        assert!(agents_content.contains("Agent Instructions"));
    }

    // -- effective_max_iterations tests --

    #[test]
    fn test_effective_max_iterations_cloud_1m() {
        // Cloud model with 1M context should get ~50 iterations
        let result = effective_max_iterations(20, 1_000_000, false);
        assert_eq!(
            result, 50,
            "1M context cloud model should get 50 iterations"
        );
    }

    #[test]
    fn test_effective_max_iterations_cloud_128k() {
        // Cloud model with 128K context
        let result = effective_max_iterations(20, 128_000, false);
        assert_eq!(
            result, 32,
            "128K context cloud model should get 32 iterations"
        );
    }

    #[test]
    fn test_effective_max_iterations_cloud_never_below_configured() {
        // Even small cloud context should never go below configured value
        let result = effective_max_iterations(20, 32_000, false);
        assert!(result >= 20, "Cloud should never go below configured value");
    }

    #[test]
    fn test_effective_max_iterations_local_capped() {
        // Local models capped at 15 regardless of configured value
        let result = effective_max_iterations(20, 32_000, true);
        assert_eq!(result, 15, "Local should cap at 15");
    }

    #[test]
    fn test_effective_max_iterations_local_respects_low_config() {
        // If configured value is already low, don't raise it for local
        let result = effective_max_iterations(10, 32_000, true);
        assert_eq!(result, 10, "Local should not raise configured value");
    }

    #[test]
    fn test_effective_max_iterations_cloud_user_override_high() {
        // If user sets high value, cloud should honor it
        let result = effective_max_iterations(100, 128_000, false);
        assert_eq!(result, 100, "Cloud should honor user's high setting");
    }

    #[test]
    fn test_build_core_handle_local_forces_local_provider_even_with_cloud_keys() {
        let mut cfg = Config::default();
        cfg.agents.defaults.model = "anthropic/claude-opus-4-5".to_string();
        cfg.providers.openrouter.api_key = "sk-or-cloud-key".to_string();
        cfg.providers.openrouter.api_base = Some("https://openrouter.ai/api/v1".to_string());

        let handle = build_core_handle(
            &cfg,
            "18080",
            Some("Qwen3-8B-Q4_K_M.gguf"),
            None,
            None,
            None,
            true,
        );
        let core = handle.swappable();

        assert!(
            core.is_local,
            "local mode should force local provider wiring"
        );
        assert_eq!(core.model, "local:Qwen3-8B");
        assert_eq!(
            core.provider.get_api_base(),
            Some("http://localhost:18080/v1"),
            "main provider should point to local server in local mode"
        );
        assert_eq!(
            core.memory_provider.get_api_base(),
            Some("http://localhost:18080/v1"),
            "memory provider should also stay local in local mode"
        );
    }

    #[test]
    fn test_make_eval_provider_local_uses_local_endpoint() {
        let provider = make_eval_provider(true, 18081);
        assert_eq!(provider.get_default_model(), "local-model");
        assert_eq!(provider.get_api_base(), Some("http://localhost:18081/v1"));
    }

    #[test]
    fn test_eval_model_name_local_is_port_scoped() {
        assert_eq!(eval_model_name(true, 18082), "local:18082");
    }

    #[test]
    fn test_strip_gguf_suffix() {
        assert_eq!(strip_gguf_suffix("nanbeige4.1-3b-q8_0.gguf"), "nanbeige4.1-3b");
        assert_eq!(strip_gguf_suffix("Qwen3-8B-Q4_K_M.gguf"), "Qwen3-8B");
        assert_eq!(strip_gguf_suffix("ministral-3-8b-instruct-2512.gguf"), "ministral-3-8b-instruct-2512");
        assert_eq!(strip_gguf_suffix("nanbeige4.1-3b"), "nanbeige4.1-3b");
        assert_eq!(strip_gguf_suffix("model-f16.gguf"), "model");
        assert_eq!(strip_gguf_suffix("local-model"), "local-model");
        // Model variant suffixes like -i1 (imatrix) must NOT be stripped.
        assert_eq!(
            strip_gguf_suffix("huihui-nvidia-nemotron-nano-9b-v2-abliterated-i1"),
            "huihui-nvidia-nemotron-nano-9b-v2-abliterated-i1"
        );
        // Real quant suffixes with >= 3 chars still get stripped.
        assert_eq!(strip_gguf_suffix("model-Q4_K_M"), "model");
        assert_eq!(strip_gguf_suffix("model-q8_0"), "model");
    }

}

// ============================================================================
// Voice Library Setup
// ============================================================================

fn setup_voice_libs() {
    use std::fs;

    // Find sherpa-rs cache directory
    let cache_base = dirs::home_dir()
        .map(|h| h.join("Library/Caches/sherpa-rs/aarch64-apple-darwin"))
        .filter(|p| p.exists());

    let Some(cache_base) = cache_base else {
        return;
    };

    // Find the sherpa-onnx lib directory
    // Structure: ~/Library/Caches/sherpa-rs/aarch64-apple-darwin/{hash}/sherpa-onnx-{version}/lib/
    let mut lib_dir: Option<std::path::PathBuf> = None;

    if let Ok(entries) = std::fs::read_dir(&cache_base) {
        for entry in entries.flatten() {
            let hash_dir = entry.path();
            if !hash_dir.is_dir() {
                continue;
            }
            // Check if it's a 40-char hash directory
            if let Some(name) = hash_dir.file_name().and_then(|n| n.to_str()) {
                if name.len() >= 40 {
                    // Look for sherpa-onnx-* inside
                    if let Ok(sub_entries) = std::fs::read_dir(&hash_dir) {
                        for sub_entry in sub_entries.flatten() {
                            let onnx_dir = sub_entry.path();
                            if onnx_dir.is_dir() {
                                if let Some(onnx_name) = onnx_dir.file_name().and_then(|n| n.to_str()) {
                                    if onnx_name.starts_with("sherpa-onnx-") {
                                        lib_dir = Some(onnx_dir.join("lib"));
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if lib_dir.is_some() {
                break;
            }
        }
    }

    let Some(lib_dir) = lib_dir.filter(|p| p.exists()) else {
        return;
    };

    // Destination: ~/.local/lib/
    let dest_dir = dirs::home_dir()
        .map(|h| h.join(".local/lib"))
        .expect("Failed to get home directory");

    // Create destination if needed
    if !dest_dir.exists() {
        if let Err(e) = fs::create_dir_all(&dest_dir) {
            eprintln!("  Warning: Could not create ~/.local/lib: {}", e);
            return;
        }
    }

    // Libraries to copy
    let libs = ["libonnxruntime.1.17.1.dylib", "libsherpa-onnx-c-api.dylib"];

    for lib_name in &libs {
        let src = lib_dir.join(lib_name);
        let dst = dest_dir.join(lib_name);

        if src.exists() {
            // Copy if destination doesn't exist or source is different size
            let src_meta = fs::metadata(&src).ok();
            let dst_meta = fs::metadata(&dst).ok();
            let needs_copy = !dst.exists()
                || src_meta
                    .zip(dst_meta)
                    .map(|(s, d)| s.len() != d.len())
                    .unwrap_or(true);

            if needs_copy {
                match fs::copy(&src, &dst) {
                    Ok(_) => println!("  Installed voice library: {}", lib_name),
                    Err(e) => {
                        eprintln!("  Warning: Could not copy {}: {}", lib_name, e);
                    }
                }
            }
        }
    }

    // Also create a symlink for libonnxruntime.dylib if needed
    let src_onnx = lib_dir.join("libonnxruntime.dylib");
    let dst_onnx = dest_dir.join("libonnxruntime.dylib");
    if src_onnx.exists() && !dst_onnx.exists() {
        if let Err(e) = std::os::unix::fs::symlink(&src_onnx, &dst_onnx) {
            // If symlink fails, try copying
            if fs::copy(&src_onnx, &dst_onnx).is_ok() {
                println!("  Installed voice library: libonnxruntime.dylib");
            }
        }
    }
}

// ============================================================================
// Onboard
// ============================================================================

pub(crate) fn cmd_onboard() {
    let config_path = get_config_path();

    if config_path.exists() {
        println!("Config already exists at {}", config_path.display());
        print!("Overwrite? [y/N] ");
        io::stdout().flush().ok();
        let mut input = String::new();
        io::stdin().read_line(&mut input).ok();
        if !input.trim().eq_ignore_ascii_case("y") {
            return;
        }
    }

    let config = Config::default();
    save_config(&config, None);
    println!("  Created config at {}", config_path.display());

    let workspace = get_workspace_path(None);
    println!("  Created workspace at {}", workspace.display());

    create_workspace_templates(&workspace);

    setup_voice_libs();

    println!("\n{} nanobot is ready!", crate::LOGO);
    println!("\nNext steps:");
    println!("  1. Add your API key to ~/.nanobot/config.json");
    println!("     Get one at: https://openrouter.ai/keys");
    println!("  2. Chat: nanobot agent -m \"Hello!\"");
}

pub(crate) fn create_workspace_templates(workspace: &Path) {
    let templates: Vec<(&str, &str)> = vec![
        ("AGENTS.md", "# Agent Instructions\n\nYou are a helpful AI assistant. Be concise, accurate, and friendly.\n\n## Guidelines\n\n- Always explain what you're doing before taking actions\n- Ask for clarification when the request is ambiguous\n- Use tools to help accomplish tasks\n- Remember important information in your memory files\n"),
        ("SOUL.md", "# Soul\n\nI am nanobot, a lightweight AI assistant.\n\n## Personality\n\n- Helpful and friendly\n- Concise and to the point\n- Curious and eager to learn\n\n## Values\n\n- Accuracy over speed\n- User privacy and safety\n- Transparency in actions\n"),
        ("USER.md", "# User\n\nInformation about the user goes here.\n\n## Preferences\n\n- Communication style: (casual/formal)\n- Timezone: (your timezone)\n- Language: (your preferred language)\n"),
        ("TOOLS.md", "# Tool Usage\n\nGuidelines for using tools effectively.\n\n## Principles\n\n- Use one tool at a time and verify results before proceeding\n- Prefer read operations before write operations\n- Always confirm destructive actions with the user\n- Check tool output for errors before continuing\n"),
        ("IDENTITY.md", "# Identity\n\nYou are nanobot, a personal AI assistant.\n\n## Core Traits\n\n- You run locally on the user's hardware\n- You have access to the filesystem, shell, and web\n- You maintain memory across sessions via memory files\n- You can spawn subagents for parallel work\n"),
    ];

    for (filename, content) in &templates {
        let file_path = workspace.join(filename);
        if !file_path.exists() {
            std::fs::write(&file_path, content).ok();
            println!("  Created {}", filename);
        }
    }

    let memory_dir = workspace.join("memory");
    std::fs::create_dir_all(&memory_dir).ok();
    let memory_file = memory_dir.join("MEMORY.md");
    if !memory_file.exists() {
        std::fs::write(
            &memory_file,
            "# Long-term Memory\n\nThis file stores important information that should persist across sessions.\n",
        )
        .ok();
        println!("  Created memory/MEMORY.md");
    }
}

// ============================================================================
// Core Handle & Agent Loop
// ============================================================================

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
fn effective_max_iterations(configured: u32, max_context_tokens: usize, is_local: bool) -> u32 {
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
fn model_context_size(model: &str, config_default: usize) -> usize {
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
///   "nanbeige4.1-3b-q8_0.gguf"          → "nanbeige4.1-3b"
///   "Qwen3-8B-Q4_K_M.gguf"              → "Qwen3-8B"
///   "ministral-3-8b-instruct-2512.gguf"  → "ministral-3-8b-instruct-2512"
///   "nanbeige4.1-3b"                     → "nanbeige4.1-3b" (no-op)
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
fn local_base_url(config: &Config, fallback_port: &str) -> String {
    let custom = &config.agents.defaults.local_api_base;
    if !custom.is_empty() {
        custom.clone()
    } else {
        format!("http://localhost:{}/v1", fallback_port)
    }
}

/// Resolved local providers for all roles (main, compaction, delegation, specialist).
struct LocalProviders {
    main: Arc<dyn LLMProvider>,
    model_id: String,
    compaction: Option<Arc<dyn LLMProvider>>,
    delegation: Option<Arc<dyn LLMProvider>>,
    specialist: Option<Arc<dyn LLMProvider>>,
    max_context_tokens: usize,
}

/// Build providers for all local roles from config + endpoint resolution.
///
/// Endpoint priority per trio role:
///   1. `trio.router_endpoint` / `trio.specialist_endpoint` (explicit URL+model)
///   2. `localApiBase` + `trio.router_model` / `trio.specialist_model` (shared JIT server)
///   3. Separate port fallback (delegation_port / specialist_port)
///   4. None (disabled)
fn make_local_providers(
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
    // Always strip GGUF suffix — config may hold a .gguf filename even when
    // using LM Studio, which expects clean identifiers.
    let model_id = strip_gguf_suffix(local_model_name.unwrap_or("local-model")).to_string();

    // Create JIT gate for JIT-loading servers (e.g. LM Studio).
    // All providers sharing the same JIT endpoint get the same gate so requests
    // are serialised — prevents concurrent model switches that crash the server.
    // Skip when lms CLI pre-loads models (skip_jit_gate = true).
    let jit_gate: Option<Arc<JitGate>> = if has_custom_base && !config.agents.defaults.skip_jit_gate {
        Some(Arc::new(JitGate::new()))
    } else {
        None
    };

    let main: Arc<dyn LLMProvider> = factory::create_openai_compat(
        factory::ProviderSpec::local(&base_url, Some(&model_id))
            .with_jit_gate_opt(jit_gate.clone()),
    );

    // Auto-detect context size from local server; fall back to config default.
    let max_context_tokens = if !has_custom_base {
        crate::server::query_local_context_size(local_port)
            .unwrap_or(config.agents.defaults.local_max_context_tokens)
    } else {
        config.agents.defaults.local_max_context_tokens
    };

    // Compaction provider (separate port only).
    let compaction: Option<Arc<dyn LLMProvider>> = compaction_port.map(|p| -> Arc<dyn LLMProvider> {
        factory::create_openai_compat(
            factory::ProviderSpec::local(&local_base_url(config, p), None)
                .with_jit_gate_opt(jit_gate.clone()),
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
            return Some(factory::create_openai_compat(
                factory::ProviderSpec {
                    api_key: role_name.to_string(),
                    api_base: Some(ep.url.clone()),
                    model: Some(ep.model.clone()),
                    jit_gate: gate,
                },
            ));
        }

        // Priority 2: shared JIT server (localApiBase set) + trio model name
        if has_custom_base {
            let model = if !trio_model.is_empty() { trio_model } else { role_name };
            return Some(factory::create_openai_compat(
                factory::ProviderSpec::local(&base_url, Some(model))
                    .with_jit_gate_opt(jit_gate.clone()),
            ));
        }

        // Priority 3: separate port fallback
        fallback_port.map(|p| -> Arc<dyn LLMProvider> {
            factory::create_openai_compat(
                factory::ProviderSpec::local(&local_base_url(config, p), Some(role_name)),
            )
        })
    };

    let delegation = make_role_provider(
        "local-delegation",
        &config.trio.router_endpoint,
        &config.trio.router_model,
        delegation_port,
    );

    let specialist = make_role_provider(
        "local-specialist",
        &config.trio.specialist_endpoint,
        &config.trio.specialist_model,
        specialist_port,
    );

    LocalProviders {
        main,
        model_id,
        compaction,
        delegation,
        specialist,
        max_context_tokens,
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
        let lp = make_local_providers(config, local_port, local_model_name, compaction_port, delegation_port, specialist_port);
        let model = format!("local:{}", lp.model_id);
        (lp.main, model, lp.max_context_tokens, lp.compaction, lp.delegation, lp.specialist)
    } else {
        let provider = create_provider(config);
        let model = config.agents.defaults.model.clone();
        let ctx = model_context_size(&model, config.agents.defaults.max_context_tokens);
        (provider, model, ctx, None, None, None)
    };

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

    let core = build_swappable_core(SwappableCoreConfig {
        provider,
        workspace: config.workspace_path(),
        model,
        max_iterations: max_iters,
        max_tokens: config.agents.defaults.max_tokens,
        temperature: config.agents.defaults.temperature,
        max_context_tokens,
        brave_api_key: brave_key,
        exec_timeout: config.tools.exec_.timeout,
        restrict_to_workspace: config.tools.exec_.restrict_to_workspace,
        memory_config: config.memory.clone(),
        is_local,
        compaction_provider: cp,
        tool_delegation: config.tool_delegation.clone(),
        provenance: config.provenance.clone(),
        max_tool_result_chars: config.agents.defaults.max_tool_result_chars,
        delegation_provider: dp,
        specialist_provider: sp,
        trio_config: config.trio.clone(),
        model_capabilities_overrides: config.model_capabilities.clone(),
    });
    let counters = Arc::new(RuntimeCounters::new_with_config(max_context_tokens, &config.trio.circuit_breaker));
    // When main_no_think is enabled, also suppress thinking display from the start
    // so the user doesn't need to run /nothink manually each session.
    if config.trio.main_no_think {
        counters.suppress_thinking_in_tts.store(true, Ordering::Relaxed);
    }
    AgentHandle::new(core, counters)
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
        let lp = make_local_providers(config, local_port, local_model_name, compaction_port, delegation_port, specialist_port);
        let model = format!("local:{}", lp.model_id);
        (lp.main, model, lp.max_context_tokens, lp.compaction, lp.delegation, lp.specialist)
    } else {
        let provider = create_provider(config);
        let model = config.agents.defaults.model.clone();
        let ctx = model_context_size(&model, config.agents.defaults.max_context_tokens);
        (provider, model, ctx, None, None, None)
    };

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

    let new_core = build_swappable_core(SwappableCoreConfig {
        provider,
        workspace: config.workspace_path(),
        model,
        max_iterations: max_iters,
        max_tokens: config.agents.defaults.max_tokens,
        temperature: config.agents.defaults.temperature,
        max_context_tokens,
        brave_api_key: brave_key,
        exec_timeout: config.tools.exec_.timeout,
        restrict_to_workspace: config.tools.exec_.restrict_to_workspace,
        memory_config: config.memory.clone(),
        is_local,
        compaction_provider: cp,
        tool_delegation: config.tool_delegation.clone(),
        provenance: config.provenance.clone(),
        max_tool_result_chars: config.agents.defaults.max_tool_result_chars,
        delegation_provider: dp,
        specialist_provider: sp,
        trio_config: config.trio.clone(),
        model_capabilities_overrides: config.model_capabilities.clone(),
    });
    // Swap only the core; counters survive.
    handle.swap_core(new_core);
    // Update max context since the new model may have a different size.
    handle
        .counters
        .last_context_max
        .store(max_context_tokens as u64, Ordering::Relaxed);
    // Reset delegation health — new core may have a fresh delegation server.
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
    if core_handle.swappable().is_local && !lcm_config.enabled {
        tracing::info!("Auto-enabling LCM for local mode");
        lcm_config.enabled = true;
    }

    AgentLoop::new(
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
    )
}

// ============================================================================
// Gateway
// ============================================================================

pub(crate) fn cmd_gateway(port: u16, verbose: bool) {
    if verbose {
        eprintln!("Verbose mode enabled");
    }

    println!(
        "{} Starting nanobot gateway on port {}...",
        crate::LOGO,
        port
    );

    let mut config = load_config(None);
    check_api_key(&config);

    // Trio auto-detection (same logic as REPL startup)
    if crate::repl::should_auto_activate_trio(
        !config.agents.defaults.local_api_base.is_empty(),
        &config.trio.router_model,
        &config.trio.specialist_model,
        config.trio.router_endpoint.is_some(),
        config.trio.specialist_endpoint.is_some(),
        &config.tool_delegation.mode,
    ) {
        crate::repl::trio_enable(&mut config);
        tracing::info!(
            router_model = %config.trio.router_model,
            specialist_model = %config.trio.specialist_model,
            "trio_auto_activated_gateway"
        );
    }

    let is_local = !config.agents.defaults.local_api_base.is_empty();
    let core_handle = build_core_handle(&config, "8080", None, None, None, None, is_local);
    let runtime = tokio::runtime::Runtime::new().expect("Failed to create tokio runtime");
    runtime.block_on(run_gateway_async(config, core_handle, None, None));
}

/// Shared async gateway: creates channels, provider, cron, agent loop, and runs until stopped.
///
/// If `stop_signal` is `Some`, watches the flag for shutdown (used when spawned from REPL).
/// If `None`, watches for Ctrl+C (used for standalone CLI commands).
pub(crate) async fn run_gateway_async(
    config: Config,
    core_handle: SharedCoreHandle,
    stop_signal: Option<Arc<std::sync::atomic::AtomicBool>>,
    repl_display_tx: Option<mpsc::UnboundedSender<String>>,
) {
    use std::time::Duration;
    use tracing::info;

    let (inbound_tx, inbound_rx) = mpsc::unbounded_channel::<InboundMessage>();
    let (outbound_tx, outbound_rx) = mpsc::unbounded_channel::<OutboundMessage>();

    let cron_store_path = get_data_dir().join("cron").join("jobs.json");
    let mut cron_service = CronService::new(cron_store_path);
    cron_service.start().await;
    let cron_status = cron_service.status();
    let cron_arc = Arc::new(cron_service);

    let health_registry = Arc::new(crate::heartbeat::health::build_registry(&config));

    let mut lcm_config = config.lcm.clone();
    if core_handle.swappable().is_local && !lcm_config.enabled {
        tracing::info!("Auto-enabling LCM for local mode");
        lcm_config.enabled = true;
    }

    let mut agent_loop = AgentLoop::new(
        core_handle,
        inbound_rx,
        outbound_tx,
        inbound_tx.clone(),
        Some(cron_arc),
        config.agents.defaults.max_concurrent_chats,
        None, // gateway agent uses bus for email, not tools
        repl_display_tx,
        Some(config.providers.clone()),
        config.proprioception.clone(),
        lcm_config,
        Some(health_registry.clone()),
    );

    // Initialize voice pipeline for channels (when voice feature is enabled).
    #[cfg(feature = "voice")]
    let voice_pipeline: Option<Arc<crate::voice_pipeline::VoicePipeline>> = {
        use tracing::warn;
        match crate::voice_pipeline::VoicePipeline::new().await {
            Ok(vp) => {
                info!("Voice pipeline initialized for channels");
                Some(Arc::new(vp))
            }
            Err(e) => {
                warn!(
                    "Voice pipeline init failed (voice messages will not be transcribed): {}",
                    e
                );
                None
            }
        }
    };

    let channel_manager = ChannelManager::new(
        &config,
        inbound_tx,
        outbound_rx,
        #[cfg(feature = "voice")]
        voice_pipeline,
    );

    let quiet = stop_signal.is_some();

    let enabled = channel_manager.enabled_channels();
    if !quiet {
        if !enabled.is_empty() {
            println!("  Channels enabled: {}", enabled.join(", "));
        } else {
            println!("  Warning: No channels enabled");
        }

        {
            let job_count = cron_status
                .get("jobs")
                .and_then(|v| v.as_i64())
                .unwrap_or(0);
            if job_count > 0 {
                println!("  Cron: {} scheduled jobs", job_count);
            }
        }

        println!(
            "  Heartbeat: every {}s ({} maintenance commands)",
            DEFAULT_HEARTBEAT_INTERVAL_S,
            DEFAULT_MAINTENANCE_COMMANDS.len()
        );
    }

    // Heartbeat: maintenance commands + optional agent tasks.
    let maintenance_cmds: Vec<String> = DEFAULT_MAINTENANCE_COMMANDS
        .iter()
        .map(|s| s.to_string())
        .collect();
    let heartbeat = HeartbeatService::new(
        config.workspace_path(),
        None, // No LLM callback yet — maintenance only
        maintenance_cmds,
        DEFAULT_HEARTBEAT_INTERVAL_S,
        true,
        Some(health_registry.clone()),
    );
    heartbeat.start().await;

    // start_all() spawns channels as background tasks and returns immediately,
    // so call it before the select rather than racing it.
    channel_manager.start_all().await;

    tokio::select! {
        _ = agent_loop.run() => {
            info!("Agent loop ended");
        }
        _ = async {
            match &stop_signal {
                Some(flag) => {
                    while !flag.load(Ordering::Relaxed) {
                        tokio::time::sleep(Duration::from_millis(200)).await;
                    }
                }
                None => { tokio::signal::ctrl_c().await.ok(); }
            }
        } => {
            if stop_signal.is_none() {
                println!("\nShutting down...");
            }
        }
    }

    agent_loop.stop();
    heartbeat.stop().await;
    channel_manager.stop_all().await;
}

// ============================================================================
// Quick-start channel commands
// ============================================================================

pub(crate) fn cmd_whatsapp() {
    println!("{} Starting WhatsApp...\n", crate::LOGO);

    let mut config = load_config(None);
    check_api_key(&config);

    config.channels.whatsapp.enabled = true;
    config.channels.telegram.enabled = false;
    config.channels.feishu.enabled = false;
    config.channels.email.enabled = false;

    println!("  Scan the QR code when it appears");
    println!("  Press Ctrl+C to stop\n");

    let core_handle = build_core_handle(&config, "8080", None, None, None, None, false);
    let runtime = tokio::runtime::Runtime::new().expect("Failed to create tokio runtime");
    runtime.block_on(run_gateway_async(config, core_handle, None, None));
}

pub(crate) fn cmd_telegram(token_arg: Option<String>) {
    println!("{} Starting Telegram...\n", crate::LOGO);

    let mut config = load_config(None);
    check_api_key(&config);

    let saved_token = &config.channels.telegram.token;
    let token = match token_arg {
        Some(t) => t,
        None if !saved_token.is_empty() => {
            println!("  Using saved bot token");
            saved_token.clone()
        }
        None => {
            println!("  No Telegram bot token found.");
            println!("  Get one from @BotFather on Telegram.\n");
            print!("  Enter bot token: ");
            io::stdout().flush().ok();
            let mut input = String::new();
            io::stdin().read_line(&mut input).ok();
            let t = input.trim().to_string();
            if t.is_empty() {
                eprintln!("Error: No token provided.");
                std::process::exit(1);
            }

            print!("  Validating token... ");
            io::stdout().flush().ok();
            if validate_telegram_token(&t) {
                println!("valid!\n");
            } else {
                println!("invalid!");
                eprintln!("Error: Token validation failed. Check the token and try again.");
                std::process::exit(1);
            }

            print!("  Save token to config for next time? [Y/n] ");
            io::stdout().flush().ok();
            let mut answer = String::new();
            io::stdin().read_line(&mut answer).ok();
            if !answer.trim().eq_ignore_ascii_case("n") {
                let mut save_cfg = load_config(None);
                save_cfg.channels.telegram.token = t.clone();
                save_config(&save_cfg, None);
                println!("  Token saved to ~/.nanobot/config.json\n");
            }

            t
        }
    };

    config.channels.telegram.token = token;
    config.channels.telegram.enabled = true;
    config.channels.whatsapp.enabled = false;
    config.channels.feishu.enabled = false;
    config.channels.email.enabled = false;

    println!("  Press Ctrl+C to stop\n");

    let core_handle = build_core_handle(&config, "8080", None, None, None, None, false);
    let runtime = tokio::runtime::Runtime::new().expect("Failed to create tokio runtime");
    runtime.block_on(run_gateway_async(config, core_handle, None, None));
}

pub(crate) fn cmd_email(
    imap_host_arg: Option<String>,
    smtp_host_arg: Option<String>,
    username_arg: Option<String>,
    password_arg: Option<String>,
) {
    println!("{} Starting Email...\n", crate::LOGO);

    let mut config = load_config(None);
    check_api_key(&config);

    let email_cfg = &config.channels.email;

    // Resolve each field: CLI arg > saved config > interactive prompt.
    let imap_host = imap_host_arg
        .or_else(|| {
            if !email_cfg.imap_host.is_empty() {
                println!("  Using saved IMAP host: {}", email_cfg.imap_host);
                Some(email_cfg.imap_host.clone())
            } else {
                None
            }
        })
        .unwrap_or_else(|| {
            print!("  IMAP host (e.g. imap.gmail.com): ");
            io::stdout().flush().ok();
            let mut input = String::new();
            io::stdin().read_line(&mut input).ok();
            input.trim().to_string()
        });

    let smtp_host = smtp_host_arg
        .or_else(|| {
            if !email_cfg.smtp_host.is_empty() {
                println!("  Using saved SMTP host: {}", email_cfg.smtp_host);
                Some(email_cfg.smtp_host.clone())
            } else {
                None
            }
        })
        .unwrap_or_else(|| {
            print!("  SMTP host (e.g. smtp.gmail.com): ");
            io::stdout().flush().ok();
            let mut input = String::new();
            io::stdin().read_line(&mut input).ok();
            input.trim().to_string()
        });

    let username = username_arg
        .or_else(|| {
            if !email_cfg.username.is_empty() {
                println!("  Using saved username: {}", email_cfg.username);
                Some(email_cfg.username.clone())
            } else {
                None
            }
        })
        .unwrap_or_else(|| {
            print!("  Email address: ");
            io::stdout().flush().ok();
            let mut input = String::new();
            io::stdin().read_line(&mut input).ok();
            input.trim().to_string()
        });

    let password = password_arg
        .or_else(|| {
            if !email_cfg.password.is_empty() {
                println!("  Using saved password");
                Some(email_cfg.password.clone())
            } else {
                None
            }
        })
        .unwrap_or_else(|| {
            print!("  Password (or app password): ");
            io::stdout().flush().ok();
            let mut input = String::new();
            io::stdin().read_line(&mut input).ok();
            input.trim().to_string()
        });

    if imap_host.is_empty() || smtp_host.is_empty() || username.is_empty() || password.is_empty() {
        eprintln!("Error: All email fields are required.");
        std::process::exit(1);
    }

    // Ask to save.
    let needs_save = email_cfg.imap_host.is_empty();
    if needs_save {
        print!("  Save email settings to config for next time? [Y/n] ");
        io::stdout().flush().ok();
        let mut answer = String::new();
        io::stdin().read_line(&mut answer).ok();
        if !answer.trim().eq_ignore_ascii_case("n") {
            let mut save_cfg = load_config(None);
            save_cfg.channels.email.imap_host = imap_host.clone();
            save_cfg.channels.email.smtp_host = smtp_host.clone();
            save_cfg.channels.email.username = username.clone();
            save_cfg.channels.email.password = password.clone();
            save_config(&save_cfg, None);
            println!("  Settings saved to ~/.nanobot/config.json\n");
        }
    }

    config.channels.email.imap_host = imap_host;
    config.channels.email.smtp_host = smtp_host;
    config.channels.email.username = username;
    config.channels.email.password = password;
    config.channels.email.enabled = true;
    config.channels.whatsapp.enabled = false;
    config.channels.telegram.enabled = false;
    config.channels.feishu.enabled = false;

    println!("  Press Ctrl+C to stop\n");

    let core_handle = build_core_handle(&config, "8080", None, None, None, None, false);
    let runtime = tokio::runtime::Runtime::new().expect("Failed to create tokio runtime");
    runtime.block_on(run_gateway_async(config, core_handle, None, None));
}

/// Validate a Telegram bot token by calling the getMe API.
pub(crate) fn validate_telegram_token(token: &str) -> bool {
    let url = format!("https://api.telegram.org/bot{}/getMe", token);
    reqwest::blocking::get(&url)
        .ok()
        .and_then(|r| r.json::<serde_json::Value>().ok())
        .and_then(|d| d.get("ok")?.as_bool())
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

/// Check if a model name refers to a Claude/Anthropic model.
fn is_claude_model(model: &str) -> bool {
    let m = model.to_lowercase();
    m.starts_with("claude")
        || m.starts_with("opus")
        || m.starts_with("sonnet")
        || m.starts_with("haiku")
}

/// Check if Claude CLI OAuth credentials exist on disk.
fn has_oauth_credentials() -> bool {
    dirs::home_dir()
        .map(|h| h.join(".claude").join(".credentials.json").exists())
        .unwrap_or(false)
}

// ============================================================================
// Status
// ============================================================================

pub(crate) fn cmd_status() {
    let config_path = get_config_path();
    let config = load_config(None);
    let workspace = config.workspace_path();

    println!("{} nanobot Status\n", crate::LOGO);
    println!(
        "Config: {} [{}]",
        config_path.display(),
        if config_path.exists() {
            "ok"
        } else {
            "missing"
        }
    );
    println!(
        "Workspace: {} [{}]",
        workspace.display(),
        if workspace.exists() { "ok" } else { "missing" }
    );

    if config_path.exists() {
        println!("Model: {}", config.agents.defaults.model);
        println!(
            "OpenRouter API: {}",
            if config.providers.openrouter.api_key.is_empty() {
                "not set"
            } else {
                "configured"
            }
        );
        println!(
            "Anthropic API: {}",
            if config.providers.anthropic.api_key.is_empty() {
                "not set"
            } else {
                "configured"
            }
        );
        println!(
            "OpenAI API: {}",
            if config.providers.openai.api_key.is_empty() {
                "not set"
            } else {
                "configured"
            }
        );
        println!(
            "Gemini API: {}",
            if config.providers.gemini.api_key.is_empty() {
                "not set"
            } else {
                "configured"
            }
        );
        let vllm_status = if let Some(ref base) = config.providers.vllm.api_base {
            format!("configured ({})", base)
        } else {
            "not set".to_string()
        };
        println!("vLLM/Local: {}", vllm_status);
    }

    // Check local LLM status
    println!("\nLocal LLM Servers:");
    for (name, port) in [("main", 8080), ("fast", 8081), ("coder", 8082)] {
        let url = format!("http://localhost:{}/health", port);
        let status = match reqwest::blocking::get(&url) {
            Ok(resp) if resp.status().is_success() => "running",
            _ => "stopped",
        };
        println!("  {} (port {}): {}", name, port, status);
    }
}

pub(crate) fn cmd_tune(input_path: String, json: bool) {
    let path = std::path::PathBuf::from(input_path);
    match run_tune_from_path(&path, json) {
        Ok(output) => println!("{}", output),
        Err(e) => {
            eprintln!("Tune failed: {}", e);
            std::process::exit(1);
        }
    }
}

pub(crate) fn run_tune_from_path(path: &Path, as_json: bool) -> Result<String, String> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| format!("failed reading benchmark file '{}': {}", path.display(), e))?;

    let input: OptimizationInput = serde_json::from_str(&raw)
        .map_err(|e| format!("failed parsing benchmark JSON '{}': {}", path.display(), e))?;

    if input.measurements.is_empty() {
        return Err("benchmark file has no measurements".to_string());
    }

    let constraints = input.resolved_constraints();
    let weights = input.resolved_weights();
    let scored = score_feasible_profiles(&input.measurements, constraints, weights);
    let best = select_optimal_from_input(&input).ok_or_else(|| {
        format!(
            "no feasible profile found (min_quality={}, min_tool_success={}, max_ttft_ms={}, max_overflow={})",
            constraints.min_quality_score,
            constraints.min_tool_success_rate,
            constraints.max_ttft_ms,
            constraints.max_context_overflow_rate
        )
    })?;

    if as_json {
        return serde_json::to_string_pretty(&best)
            .map_err(|e| format!("failed serializing result JSON: {}", e));
    }

    let mut out = String::new();
    out.push_str("Best local profile\n");
    out.push_str(&format!("  id: {}\n", best.profile.id));
    out.push_str(&format!("  model: {}\n", best.profile.model));
    out.push_str(&format!("  ctx_size: {}\n", best.profile.ctx_size));
    out.push_str(&format!("  max_tokens: {}\n", best.profile.max_tokens));
    out.push_str(&format!("  temperature: {:.3}\n", best.profile.temperature));
    out.push_str(&format!("  score: {:.4}\n", best.score));
    out.push_str(&format!(
        "  metrics: quality={:.3}, tool_success={:.3}, ttft_ms={:.0}, toks_per_sec={:.1}, overflow={:.3}\n",
        best.sample.quality_score,
        best.sample.tool_success_rate,
        best.sample.ttft_ms,
        best.sample.output_toks_per_sec,
        best.sample.context_overflow_rate
    ));

    if scored.len() > 1 {
        out.push_str("Top alternatives\n");
        for candidate in scored.iter().take(3).skip(1) {
            out.push_str(&format!(
                "  - {} (score {:.4}, q {:.3}, ttft {:.0} ms, {:.1} tok/s)\n",
                candidate.profile.id,
                candidate.score,
                candidate.sample.quality_score,
                candidate.sample.ttft_ms,
                candidate.sample.output_toks_per_sec
            ));
        }
    }

    Ok(out)
}

// ============================================================================
// Channels
// ============================================================================

pub(crate) fn cmd_channels_status() {
    let config = load_config(None);
    println!("Channel Status\n");
    println!(
        "  WhatsApp: {} ({})",
        if config.channels.whatsapp.enabled {
            "enabled"
        } else {
            "disabled"
        },
        config.channels.whatsapp.effective_bridge_url()
    );
    let tg_info = if config.channels.telegram.token.is_empty() {
        "not configured".to_string()
    } else {
        let t = &config.channels.telegram.token;
        format!("token: {}...", &t[..t.len().min(10)])
    };
    println!(
        "  Telegram: {} ({})",
        if config.channels.telegram.enabled {
            "enabled"
        } else {
            "disabled"
        },
        tg_info
    );
    println!(
        "  Feishu: {}",
        if config.channels.feishu.enabled {
            "enabled"
        } else {
            "disabled"
        }
    );
    let email_info = if config.channels.email.imap_host.is_empty() {
        "not configured".to_string()
    } else {
        format!(
            "imap: {}, smtp: {}",
            config.channels.email.imap_host, config.channels.email.smtp_host
        )
    };
    println!(
        "  Email: {} ({})",
        if config.channels.email.enabled {
            "enabled"
        } else {
            "disabled"
        },
        email_info
    );
}

// ============================================================================
// Cron
// ============================================================================

pub(crate) fn cmd_cron_list(include_all: bool) {
    let store_path = get_data_dir().join("cron").join("jobs.json");
    let service = CronService::new(store_path);
    let jobs = service.list_jobs(include_all);

    if jobs.is_empty() {
        println!("No scheduled jobs.");
        return;
    }

    println!("Scheduled Jobs\n");
    println!(
        "{:<10} {:<20} {:<15} {:<10} {}",
        "ID", "Name", "Schedule", "Status", "Next Run"
    );
    println!("{}", "-".repeat(70));

    for job in &jobs {
        let sched = match job.schedule.kind.as_str() {
            "every" => format!("every {}s", job.schedule.every_ms.unwrap_or(0) / 1000),
            "cron" => job.schedule.expr.clone().unwrap_or_default(),
            _ => "one-time".to_string(),
        };
        let status = if job.enabled { "enabled" } else { "disabled" };
        let next_run = job
            .state
            .next_run_at_ms
            .map(|ms| {
                chrono::DateTime::from_timestamp(ms / 1000, 0)
                    .map(|dt| dt.format("%Y-%m-%d %H:%M").to_string())
                    .unwrap_or_default()
            })
            .unwrap_or_default();
        println!(
            "{:<10} {:<20} {:<15} {:<10} {}",
            job.id, job.name, sched, status, next_run
        );
    }
}

pub(crate) fn cmd_cron_add(
    name: String,
    message: String,
    every: Option<u64>,
    cron_expr: Option<String>,
    deliver: bool,
    to: Option<String>,
    channel: Option<String>,
) {
    let schedule = if let Some(secs) = every {
        CronSchedule {
            kind: "every".to_string(),
            every_ms: Some((secs * 1000) as i64),
            ..Default::default()
        }
    } else if let Some(expr) = cron_expr {
        CronSchedule {
            kind: "cron".to_string(),
            expr: Some(expr),
            ..Default::default()
        }
    } else {
        eprintln!("Error: Must specify --every or --cron");
        std::process::exit(1);
    };

    let store_path = get_data_dir().join("cron").join("jobs.json");
    let mut service = CronService::new(store_path);
    let job = service.add_job(
        &name,
        schedule,
        &message,
        deliver,
        channel.as_deref(),
        to.as_deref(),
        false,
    );
    println!("  Added job '{}' ({})", job.name, job.id);
}

pub(crate) fn cmd_cron_remove(job_id: String) {
    let store_path = get_data_dir().join("cron").join("jobs.json");
    let mut service = CronService::new(store_path);
    if service.remove_job(&job_id) {
        println!("  Removed job {}", job_id);
    } else {
        eprintln!("Job {} not found", job_id);
    }
}

pub(crate) fn cmd_cron_enable(job_id: String, disable: bool) {
    let store_path = get_data_dir().join("cron").join("jobs.json");
    let mut service = CronService::new(store_path);
    if let Some(job) = service.enable_job(&job_id, !disable) {
        let status = if disable { "disabled" } else { "enabled" };
        println!("  Job '{}' {}", job.name, status);
    } else {
        eprintln!("Job {} not found", job_id);
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// Load a direct AnthropicProvider using OAuth tokens from Claude CLI.
///
/// Reads the access token from `~/.claude/.credentials.json`, refreshes if
/// needed, and returns an `AnthropicProvider` with OAuth identity headers
/// (same approach as OpenClaw — direct API, no proxy, no CLI subprocess).
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

    // "claude-max" or "claude-max/opus" → OAuth token from Claude CLI credentials.
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

    // Claude model + no Anthropic API key + OAuth credentials exist → use OAuth.
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
            "create_provider: prefix resolved model={} → base={}, stripped={}",
            model, api_base, stripped_model
        );
        return factory::create_openai_compat(factory::ProviderSpec {
            api_key,
            api_base: Some(api_base),
            model: Some(stripped_model),
            jit_gate: None,
        });
    }

    let api_key = config.get_api_key().unwrap_or_default();

    // No API key configured at all → try OAuth as last resort.
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
    })
}

// ============================================================================
// Knowledge store commands
// ============================================================================

/// Ingest files into the knowledge store for search.
pub(crate) fn cmd_ingest(
    files: Vec<String>,
    name: Option<String>,
    chunk_size: usize,
    overlap: usize,
) {
    use crate::agent::knowledge_store::KnowledgeStore;

    let store = match KnowledgeStore::open_default() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error opening knowledge store: {}", e);
            std::process::exit(1);
        }
    };

    for file_path in &files {
        let path = std::path::Path::new(file_path);
        if !path.exists() {
            eprintln!("File not found: {}", file_path);
            continue;
        }

        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("Error reading {}: {}", file_path, e);
                continue;
            }
        };

        let source_name = name.clone().unwrap_or_else(|| {
            path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or(file_path)
                .to_string()
        });

        match store.ingest(&source_name, Some(file_path), &content, chunk_size, overlap) {
            Ok(result) => {
                println!(
                    "Ingested '{}': {} chunks, {} chars",
                    source_name, result.chunks_created, result.total_chars
                );
            }
            Err(e) => {
                eprintln!("Error ingesting {}: {}", file_path, e);
            }
        }
    }

    // Show summary
    match store.stats() {
        Ok(stats) => {
            println!(
                "\nKnowledge store: {} sources, {} chunks, {} total chars",
                stats.total_sources, stats.total_chunks, stats.total_chars
            );
        }
        Err(e) => eprintln!("Error getting stats: {}", e),
    }
}

/// Search the knowledge store.
pub(crate) fn cmd_search(query: String, limit: usize) {
    use crate::agent::knowledge_store::KnowledgeStore;

    let store = match KnowledgeStore::open_default() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error opening knowledge store: {}", e);
            std::process::exit(1);
        }
    };

    match store.search(&query, limit) {
        Ok(hits) => {
            if hits.is_empty() {
                println!("No results for '{}'", query);
                return;
            }
            println!("Results for '{}':\n", query);
            for (i, hit) in hits.iter().enumerate() {
                println!(
                    "{}. [{}] chunk #{}\n   {}\n",
                    i + 1,
                    hit.source_name,
                    hit.chunk_idx,
                    hit.snippet.replace('\n', "\n   ")
                );
            }
        }
        Err(e) => {
            eprintln!("Search error: {}", e);
            std::process::exit(1);
        }
    }
}

// ============================================================================
// Evaluation helpers
// ============================================================================

/// Build an LLM provider for eval: local LM Studio or cloud API.
fn make_eval_provider(local: bool, port: u16) -> Arc<dyn LLMProvider> {
    if local {
        factory::create_openai_compat(factory::ProviderSpec::local(
            &format!("http://localhost:{}/v1", port),
            Some("local-model"),
        ))
    } else {
        let config = load_config(None);
        check_api_key(&config);
        create_provider(&config)
    }
}

/// Detect the model name for result labelling.
fn eval_model_name(local: bool, port: u16) -> String {
    if local {
        format!("local:{}", port)
    } else {
        let config = load_config(None);
        config.agents.defaults.model.clone()
    }
}

/// Wrap an LLM provider into the `Fn(String) -> Future<Result<String, String>>`
/// closure that runner.rs expects.
fn make_llm_caller(
    provider: Arc<dyn LLMProvider>,
) -> impl Fn(String) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<String, String>>>>
{
    move |prompt: String| {
        let p = provider.clone();
        Box::pin(async move {
            let messages = vec![serde_json::json!({"role": "user", "content": prompt})];
            p.chat(&messages, None, None, 512, 0.3, None, None)
                .await
                .map(|r| r.content.unwrap_or_default())
                .map_err(|e| e.to_string())
        })
    }
}

// ============================================================================
// Evaluation benchmarks
// ============================================================================

pub(crate) fn cmd_eval_hanoi(
    disks: u8,
    calibrate: bool,
    samples: usize,
    solve: bool,
    catts: bool,
    k: usize,
    local: bool,
    port: u16,
) {
    use crate::agent::eval::hanoi;
    use crate::agent::eval::results;
    use crate::agent::eval::runner;
    use crate::agent::step_voter::estimate_voters_needed;

    if calibrate {
        let model_name = eval_model_name(local, port);
        println!(
            "{} Hanoi Calibration: {} disks, {} samples (model: {})\n",
            crate::LOGO,
            disks,
            samples,
            model_name
        );

        let total_steps = (1usize << disks as usize) - 1;
        println!("  Total optimal steps: {}", total_steps);
        println!("  Sampling up to {} steps for calibration...\n", samples);

        let provider = make_eval_provider(local, port);
        let llm_call = make_llm_caller(provider);

        let cal_config = runner::HanoiCalibrationConfig {
            num_disks: disks,
            num_samples: samples,
            target_reliability: 0.999,
        };

        let runtime = tokio::runtime::Runtime::new().expect("Failed to create tokio runtime");
        let mut cal = runtime.block_on(runner::calibrate_hanoi(&cal_config, llm_call));
        cal.model = model_name.clone();

        println!("  Results:");
        println!("  ─────────────────────────");
        println!(
            "    Accuracy:       {:.1}% ({}/{})",
            cal.accuracy * 100.0,
            (cal.accuracy * cal.num_samples as f64).round() as usize,
            cal.num_samples
        );
        println!("    Red Flag Rate:  {:.1}%", cal.red_flag_rate * 100.0);
        println!("    Median Latency: {:.0}ms", cal.median_latency_ms);

        // Voters needed
        match estimate_voters_needed(cal.accuracy, 0.999, 15) {
            Some(v) => println!("    Voters Needed:  {} (for 99.9% reliability)", v),
            None => println!("    Voters Needed:  impossible at this accuracy"),
        }

        // Save result
        let eval_result = runner::calibration_to_eval_result(&cal, disks, 0.999, &model_name);
        let dir = results::default_results_dir();
        match results::save_result(&eval_result, &dir) {
            Ok(path) => println!("\n  Saved to {}", path.display()),
            Err(e) => eprintln!("\n  Failed to save: {}", e),
        }
    } else if solve {
        println!(
            "{} Hanoi Solve: {} disks, k={}{}\n",
            crate::LOGO,
            disks,
            k,
            if catts { " (CATTS enabled)" } else { "" }
        );

        let solution = hanoi::optimal_solution(disks);
        println!("  Optimal solution: {} steps", solution.len());
        println!("  (Full MAKER voting solve not yet implemented.)");
    } else {
        println!("Usage: nanobot eval hanoi --calibrate|--solve [options]");
        println!("  --calibrate --samples N    Measure model accuracy on N sampled steps");
        println!("  --solve --catts -k K       Full solve with MAKER voting");
    }
}

pub(crate) fn cmd_eval_haystack(
    facts: usize,
    length: usize,
    aggregate: bool,
    local: bool,
    port: u16,
) {
    use crate::agent::eval::haystack;
    use crate::agent::eval::results;
    use crate::agent::eval::runner;
    use crate::agent::knowledge_store::KnowledgeStore;

    println!(
        "{} Aggregation Haystack: {} facts, {} chars{}\n",
        crate::LOGO,
        facts,
        length,
        if aggregate {
            " + aggregation"
        } else {
            " (retrieval only)"
        }
    );

    // Generate synthetic data
    let fact_list = haystack::generate_facts(facts, 42);
    let document = haystack::assemble_document(&fact_list, length, 42);

    println!("  Generated {} facts", fact_list.len());
    println!("  Document length: {} chars", document.len());

    // Ingest into a temporary knowledge store
    let tmp = std::env::temp_dir().join(format!("nanobot_eval_haystack_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).ok();
    let db_path = tmp.join("eval_haystack.db");
    let store = KnowledgeStore::open(&db_path).unwrap_or_else(|e| {
        eprintln!("Failed to open knowledge store: {}", e);
        std::process::exit(1);
    });

    let ingest_result = store
        .ingest("eval_haystack", None, &document, 4096, 256)
        .unwrap_or_else(|e| {
            eprintln!("Failed to ingest document: {}", e);
            std::process::exit(1);
        });

    println!("  Ingested: {} chunks\n", ingest_result.chunks_created);

    // Tier 1: Pure FTS5 retrieval benchmark
    let metrics = haystack::evaluate_retrieval(&fact_list, |query| {
        store
            .search(query, 10)
            .unwrap_or_default()
            .into_iter()
            .map(|h| h.snippet)
            .collect()
    });

    println!("  Tier 1: FTS5 Retrieval");
    println!("  ─────────────────────────");
    println!("    Precision: {:.3}", metrics.precision);
    println!("    Recall:    {:.3}", metrics.recall);
    println!("    MRR:       {:.3}", metrics.mrr);
    println!(
        "    Found:     {}/{}\n",
        metrics.facts_found, metrics.facts_total
    );

    // Save tier 1 result
    let model_name = eval_model_name(local, port);
    let retrieval_result = results::EvalResult {
        benchmark: results::BenchmarkType::Haystack,
        started_at: results::now_timestamp(),
        completed_at: results::now_timestamp(),
        model: "fts5".to_string(),
        data: results::BenchmarkData::HaystackRetrieval {
            num_facts: facts,
            total_length: length,
            precision: metrics.precision,
            recall: metrics.recall,
            mrr: metrics.mrr,
        },
        metadata: Default::default(),
    };
    let dir = results::default_results_dir();
    match results::save_result(&retrieval_result, &dir) {
        Ok(path) => println!("  Saved retrieval result to {}\n", path.display()),
        Err(e) => eprintln!("  Failed to save: {}\n", e),
    }

    if aggregate {
        let tasks = haystack::generate_aggregation_tasks(&fact_list);
        println!("  Tier 2: Aggregation Tasks ({})", model_name);
        println!("  ─────────────────────────");
        println!(
            "  Running {} aggregation tasks against LLM...\n",
            tasks.len()
        );

        let provider = make_eval_provider(local, port);
        let llm_call = make_llm_caller(provider);

        let runtime = tokio::runtime::Runtime::new().expect("Failed to create tokio runtime");
        let agg_results = runtime.block_on(runner::run_haystack_aggregation(&tasks, llm_call));

        let correct = agg_results.iter().filter(|r| r.correct).count();
        let total = agg_results.len();
        let accuracy = if total > 0 {
            correct as f64 / total as f64
        } else {
            0.0
        };
        let mean_search = agg_results
            .iter()
            .map(|r| r.search_calls as f64)
            .sum::<f64>()
            / total.max(1) as f64;

        for (i, res) in agg_results.iter().enumerate() {
            let status = if res.correct { "PASS" } else { "FAIL" };
            let prompt = haystack::build_aggregation_prompt(&res.task);
            println!(
                "    [{}] Task {}: {}",
                status,
                i + 1,
                prompt.lines().next().unwrap_or("")
            );
        }

        println!(
            "\n    Accuracy: {}/{} ({:.1}%)",
            correct,
            total,
            accuracy * 100.0
        );

        // Save tier 2 result
        let agg_eval_result = results::EvalResult {
            benchmark: results::BenchmarkType::Haystack,
            started_at: results::now_timestamp(),
            completed_at: results::now_timestamp(),
            model: model_name,
            data: results::BenchmarkData::HaystackAggregation {
                num_facts: facts,
                total_length: length,
                tasks_correct: correct,
                tasks_total: total,
                accuracy,
                mean_search_calls: mean_search,
            },
            metadata: Default::default(),
        };
        match results::save_result(&agg_eval_result, &dir) {
            Ok(path) => println!("  Saved aggregation result to {}", path.display()),
            Err(e) => eprintln!("  Failed to save: {}", e),
        }
    }
}

pub(crate) fn cmd_eval_learn(family: String, tasks: usize, depth: usize, local: bool, port: u16) {
    use crate::agent::eval::learning;
    use crate::agent::eval::results;
    use crate::agent::eval::runner;

    let task_family = match family.as_str() {
        "arithmetic" => learning::TaskFamily::ArithmeticChain { depth },
        "fact-retrieval" => learning::TaskFamily::FactRetrieval { num_facts: depth },
        "tool-chain" => learning::TaskFamily::ToolChain { num_steps: depth },
        _ => {
            eprintln!(
                "Unknown task family: '{}'. Use: arithmetic, fact-retrieval, tool-chain",
                family
            );
            std::process::exit(1);
        }
    };

    let model_name = eval_model_name(local, port);
    println!(
        "{} Learning Curve: {} tasks, family={:?} (model: {})\n",
        crate::LOGO,
        tasks,
        task_family,
        model_name
    );

    let curriculum = learning::generate_curriculum(&task_family, tasks, 42);
    println!("  Generated {} tasks", curriculum.len());
    println!("  Running against LLM...\n");

    let provider = make_eval_provider(local, port);
    let llm_call = make_llm_caller(provider);

    let runtime = tokio::runtime::Runtime::new().expect("Failed to create tokio runtime");
    let executions = runtime.block_on(runner::run_learning_eval(&curriculum, llm_call));

    // Compute metrics
    let curve = learning::compute_learning_curve(&task_family, &executions, 5.min(tasks));
    let correct = executions.iter().filter(|e| e.success).count();
    let total = executions.len();
    let final_accuracy = if total > 0 {
        correct as f64 / total as f64
    } else {
        0.0
    };

    // Print per-task results (compact)
    for exec in &executions {
        let status = if exec.success { "PASS" } else { "FAIL" };
        let task = &curriculum[exec.task_index];
        println!(
            "    [{}] Task {} (d{}): got \"{}\" expected \"{}\" ({:.0}ms)",
            status,
            exec.task_index,
            task.difficulty,
            exec.agent_answer
                .lines()
                .next()
                .unwrap_or("")
                .chars()
                .take(40)
                .collect::<String>(),
            task.expected_answer,
            exec.duration_ms
        );
    }

    println!("\n  Results:");
    println!("  ─────────────────────────");
    println!(
        "    Accuracy:         {}/{} ({:.1}%)",
        correct,
        total,
        final_accuracy * 100.0
    );
    println!("    Forward Transfer: {:.3}", curve.forward_transfer);
    println!("    Surprise Rate:    {:.1}%", curve.surprise_rate * 100.0);

    // Save result
    let eval_result = results::EvalResult {
        benchmark: results::BenchmarkType::Learning,
        started_at: results::now_timestamp(),
        completed_at: results::now_timestamp(),
        model: model_name,
        data: results::BenchmarkData::Learning {
            family,
            total_tasks: total,
            completed: correct,
            final_accuracy,
            forward_transfer: curve.forward_transfer,
            surprise_rate: curve.surprise_rate,
        },
        metadata: Default::default(),
    };
    let dir = results::default_results_dir();
    match results::save_result(&eval_result, &dir) {
        Ok(path) => println!("\n  Saved to {}", path.display()),
        Err(e) => eprintln!("\n  Failed to save: {}", e),
    }
}

pub(crate) fn cmd_eval_sprint(corpus_size: usize, questions: usize, local: bool, port: u16) {
    use crate::agent::eval::results;
    use crate::agent::eval::runner;
    use crate::agent::eval::sprint;

    let model_name = eval_model_name(local, port);
    let config = sprint::SprintConfig {
        corpus_size,
        num_questions: questions,
        ..Default::default()
    };

    println!(
        "{} Research Sprint: {} chars corpus, {} questions (model: {})\n",
        crate::LOGO,
        corpus_size,
        questions,
        model_name
    );

    let (domains, _document) = sprint::generate_corpus(&config);
    let all_facts = sprint::all_facts(&domains);
    let question_list = sprint::generate_questions(&domains, questions, config.seed);

    println!("  Domains: {}", domains.len());
    println!("  Total facts: {}", all_facts.len());
    println!("  Questions: {}", question_list.len());
    println!("  Running against LLM...\n");

    let provider = make_eval_provider(local, port);
    let llm_call = make_llm_caller(provider);

    let runtime = tokio::runtime::Runtime::new().expect("Failed to create tokio runtime");
    let (scorecard, executions) = runtime.block_on(runner::run_sprint(&config, llm_call));

    // Print per-question results
    for exec in &executions {
        let q = &question_list[exec.index];
        let status = if exec.correct { "PASS" } else { "FAIL" };
        println!(
            "    [{}] Q{} [{}]: got \"{}\"",
            status,
            exec.index + 1,
            q.difficulty.label(),
            exec.agent_answer
                .lines()
                .next()
                .unwrap_or("")
                .chars()
                .take(50)
                .collect::<String>()
        );
    }

    println!();
    print!("{}", sprint::format_scorecard(&scorecard));

    // Save result
    let catts_trend: Vec<f64> = executions
        .iter()
        .map(|e| if e.catts_accepted_pilot { 1.0 } else { 0.0 })
        .collect();
    let time_per_q: Vec<f64> = executions.iter().map(|e| e.duration_ms as f64).collect();

    let eval_result = results::EvalResult {
        benchmark: results::BenchmarkType::Sprint,
        started_at: results::now_timestamp(),
        completed_at: results::now_timestamp(),
        model: model_name,
        data: results::BenchmarkData::Sprint {
            corpus_size,
            questions_total: scorecard.questions_total,
            questions_correct: scorecard.questions_correct,
            accuracy: scorecard.accuracy,
            compound_score: scorecard.compound_score,
            catts_savings_trend: catts_trend,
            time_per_question: time_per_q,
        },
        metadata: Default::default(),
    };
    let dir = results::default_results_dir();
    match results::save_result(&eval_result, &dir) {
        Ok(path) => println!("\n  Saved to {}", path.display()),
        Err(e) => eprintln!("\n  Failed to save: {}", e),
    }
}

pub(crate) fn cmd_eval_report() {
    use crate::agent::eval::results;

    let dir = results::default_results_dir();
    println!("{} Evaluation Results\n", crate::LOGO);
    println!("  Results directory: {}\n", dir.display());

    match results::load_all_results(&dir) {
        Ok(results_list) if results_list.is_empty() => {
            println!("  No saved results found.");
            println!("  Run a benchmark first: nanobot eval hanoi --calibrate");
        }
        Ok(results_list) => {
            println!("{}", results::format_summary(&results_list));
        }
        Err(e) => {
            println!("  No results found ({})", e);
        }
    }
}

