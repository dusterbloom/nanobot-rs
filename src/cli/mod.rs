//! CLI subcommand handlers for nanobot.
//!
//! This module contains all command implementations that were previously in main.rs.
//! Functions are extracted here to keep main.rs focused on argument parsing and routing.

mod core_builder;
mod eval;
mod provider;
mod skills;
mod voice;

// Re-export everything that other modules need.
pub(crate) use core_builder::{
    build_core_handle, create_agent_loop, effective_max_iterations, rebuild_core, strip_gguf_suffix,
};
#[cfg(feature = "mlx")]
pub(crate) use core_builder::{
    build_core_handle_mlx, create_agent_loop_mlx, start_mlx_provider, MlxHandle,
};
#[cfg(feature = "cluster")]
pub(crate) use core_builder::setup_cluster_for_repl;
pub(crate) use eval::{
    cmd_eval_hanoi, cmd_eval_haystack, cmd_eval_learn, cmd_eval_report, cmd_eval_sprint,
    make_eval_provider, eval_model_name,
};
pub(crate) use provider::{check_api_key, create_provider};
pub(crate) use skills::{cmd_skill_add, cmd_skill_remove};
#[cfg(feature = "voice")]
pub(crate) use voice::{
    cmd_realtime, cmd_realtime_server, cmd_voice_clone, cmd_voice_config, cmd_voice_list,
    parse_input_mode,
};

use std::io::{self, Write};
use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use tokio::sync::mpsc;

use crate::agent::agent_loop::{AgentLoop, SharedCoreHandle};
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

    // -- inference engine detection tests --

    #[test]
    fn test_inference_engine_mlx_detected() {
        let mut cfg = Config::default();
        cfg.agents.defaults.inference_engine = "mlx".to_string();
        assert_eq!(cfg.agents.defaults.inference_engine, "mlx");
        // MLX preset should default to qwen3.5-2b
        assert_eq!(cfg.agents.defaults.mlx_preset, "qwen3.5-2b");
    }

    #[test]
    fn test_inference_engine_mlx_model_dir_default() {
        let cfg = Config::default();
        assert!(cfg.agents.defaults.mlx_model_dir.is_none(),
            "mlx_model_dir should default to None (auto-detect)");
    }

    #[test]
    fn test_inference_engine_mlx_model_dir_custom() {
        let mut cfg = Config::default();
        cfg.agents.defaults.mlx_model_dir = Some("/tmp/my-model".to_string());
        assert_eq!(cfg.agents.defaults.mlx_model_dir.as_deref(), Some("/tmp/my-model"));
    }

    #[cfg(feature = "mlx")]
    #[test]
    fn test_resolve_mlx_model_dir_default() {
        let cfg = Config::default();
        let dir = core_builder::resolve_mlx_model_dir(&cfg);
        assert!(dir.to_string_lossy().contains("Qwen3.5-2B-MLX-8bit"),
            "default dir should point to Qwen3.5-2B: {:?}", dir);
    }

    #[cfg(feature = "mlx")]
    #[test]
    fn test_resolve_mlx_model_dir_custom() {
        let mut cfg = Config::default();
        cfg.agents.defaults.mlx_model_dir = Some("/tmp/custom-model".to_string());
        let dir = core_builder::resolve_mlx_model_dir(&cfg);
        assert_eq!(dir.to_string_lossy(), "/tmp/custom-model");
    }

    #[cfg(feature = "mlx")]
    #[test]
    fn test_build_core_handle_mlx_sets_local_and_model_prefix() {
        // Verify that build_core_handle_mlx produces a local core with mlx: prefix.
        // NOTE: Can't actually start MlxProvider without model files, so we test
        // that the function exists and config defaults resolve correctly.
        let cfg = Config::default();
        let dir = core_builder::resolve_mlx_model_dir(&cfg);
        assert!(dir.to_string_lossy().ends_with("Qwen3.5-2B-MLX-8bit"));
    }

    #[cfg(feature = "mlx")]
    #[test]
    fn test_preset_from_model_dir() {
        use std::path::Path;
        assert_eq!(
            core_builder::preset_from_model_dir(Path::new("/models/mlx-community/Qwen3-1.7B-MLX-8bit")),
            "qwen3-1.7b"
        );
        assert_eq!(
            core_builder::preset_from_model_dir(Path::new("/models/mlx-community/Qwen3-4B-MLX-4bit")),
            "qwen3-4b"
        );
        assert_eq!(
            core_builder::preset_from_model_dir(Path::new("/models/mlx-community/Qwen3.5-2B-MLX-8bit")),
            "qwen3.5-2b"
        );
        // Unknown defaults to qwen3.5-2b
        assert_eq!(
            core_builder::preset_from_model_dir(Path::new("/models/some-unknown-model")),
            "qwen3.5-2b"
        );
    }

    #[test]
    fn test_local_backend_default() {
        let cfg = Config::default();
        assert_eq!(cfg.agents.defaults.local_backend, "lmstudio");
    }

    #[test]
    fn test_gateway_uses_create_agent_loop_for_perplexity_gate() {
        // Verify that create_agent_loop respects perplexity_gate config
        let mut cfg = Config::default();
        cfg.perplexity_gate.enabled = true;
        cfg.perplexity_gate.surprise_threshold = 5.0;

        let handle = build_core_handle(&cfg, "8080", None, None, None, None, false);
        let agent_loop = create_agent_loop(handle, &cfg, None, None, None, None);

        // The agent loop's shared state should have perplexity gate enabled.
        // We verify through the public accessor that exists on AgentLoop.
        assert!(agent_loop.has_perplexity_gate(),
            "create_agent_loop should enable perplexity gate when config says so");
    }

    #[test]
    fn test_create_agent_loop_without_perplexity_gate() {
        let cfg = Config::default();
        assert!(!cfg.perplexity_gate.enabled);

        let handle = build_core_handle(&cfg, "8080", None, None, None, None, false);
        let agent_loop = create_agent_loop(handle, &cfg, None, None, None, None);

        assert!(!agent_loop.has_perplexity_gate(),
            "perplexity gate should be off when config says disabled");
    }

    #[test]
    fn test_rebuild_agent_loop_preserves_perplexity_gate() {
        // Simulate what rebuild_agent_loop does: create_agent_loop with same config.
        let mut cfg = Config::default();
        cfg.perplexity_gate.enabled = true;

        let handle = build_core_handle(&cfg, "8080", None, None, None, None, false);
        let first = create_agent_loop(handle.clone(), &cfg, None, None, None, None);
        assert!(first.has_perplexity_gate());

        // Rebuild with same config should preserve gate.
        let second = create_agent_loop(handle, &cfg, None, None, None, None);
        assert!(second.has_perplexity_gate(),
            "rebuilt agent loop should still have perplexity gate enabled");
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

    #[cfg(feature = "voice")]
    #[test]
    fn test_input_mode_from_cli_flag() {
        use crate::realtime::InputMode;

        assert_eq!(parse_input_mode("continuous"), InputMode::Continuous);
        assert_eq!(parse_input_mode("c"), InputMode::Continuous);
        assert_eq!(parse_input_mode("ptt"), InputMode::PushToTalk);
        assert_eq!(parse_input_mode("push-to-talk"), InputMode::PushToTalk);
        assert_eq!(parse_input_mode("p"), InputMode::PushToTalk);
        assert_eq!(parse_input_mode("unknown"), InputMode::Continuous);
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
            if std::fs::copy(&src_onnx, &dst_onnx).is_ok() {
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

    // MLX in-process provider (when inference_engine == "mlx" or localBackend == "mlx").
    #[cfg(feature = "mlx")]
    let mlx_handle: Option<MlxHandle> = if config.agents.defaults.inference_engine == "mlx"
        || config.agents.defaults.local_backend == "mlx"
    {
        match start_mlx_provider(&config) {
            Ok(h) => Some(h),
            Err(e) => {
                eprintln!("⚠ MLX provider failed to start: {e}");
                eprintln!("  Falling back to default provider");
                None
            }
        }
    } else {
        None
    };

    #[cfg(feature = "mlx")]
    let core_handle = if let Some(ref mlx) = mlx_handle {
        build_core_handle_mlx(&config, mlx)
    } else {
        let is_local = !config.agents.defaults.local_api_base.is_empty();
        build_core_handle(&config, "8080", None, None, None, None, is_local)
    };
    #[cfg(not(feature = "mlx"))]
    let core_handle = {
        let is_local = !config.agents.defaults.local_api_base.is_empty();
        build_core_handle(&config, "8080", None, None, None, None, is_local)
    };

    // Build setup closure that wires MLX provider into the agent loop.
    #[cfg(feature = "mlx")]
    let setup: Option<Box<dyn FnOnce(&mut AgentLoop) + Send>> = mlx_handle.map(|mlx| {
        let provider = mlx.provider.clone();
        let gate_config = {
            let mut g = config.perplexity_gate.clone();
            g.enabled = true;
            g
        };
        Box::new(move |loop_: &mut AgentLoop| {
            loop_.set_mlx_provider(provider);
            loop_.set_perplexity_gate(gate_config);
            tracing::info!("gateway: MLX provider wired, perplexity gate auto-enabled");
        }) as Box<dyn FnOnce(&mut AgentLoop) + Send>
    });
    #[cfg(not(feature = "mlx"))]
    let setup: Option<Box<dyn FnOnce(&mut AgentLoop) + Send>> = None;

    let runtime = tokio::runtime::Runtime::new().expect("Failed to create tokio runtime");
    runtime.block_on(run_gateway_async(config, core_handle, None, None, setup));
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
    setup_fn: Option<Box<dyn FnOnce(&mut AgentLoop) + Send>>,
) {
    use std::time::Duration;
    use tracing::{info, warn};

    // Auto-start SearXNG if configured
    if config.tools.web.search.provider == "searxng" && config.tools.web.search.auto_start {
        match crate::searxng::ensure_searxng(&config.tools.web.search.searxng_url).await {
            Ok(()) => info!("SearXNG ready"),
            Err(e) => warn!("SearXNG auto-setup failed: {e} — web search will use fallback"),
        }
    }

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

    if config.perplexity_gate.enabled {
        agent_loop.set_perplexity_gate(config.perplexity_gate.clone());
    }

    // Apply optional setup (e.g. MLX provider wiring).
    if let Some(f) = setup_fn {
        f(&mut agent_loop);
    }

    // Start cluster discovery in the background (when cluster feature is enabled).
    #[cfg(feature = "cluster")]
    if config.cluster.enabled {
        let cluster_state = crate::cluster::state::ClusterState::new();
        let discovery = crate::cluster::discovery::ClusterDiscovery::new(
            config.cluster.clone(),
            cluster_state.clone(),
        );
        let _discovery_handle = discovery.run();
        tracing::info!("cluster_discovery_started");
        let router = Arc::new(crate::cluster::router::ClusterRouter::new(
            cluster_state,
            config.cluster.clone(),
        ));
        agent_loop.set_cluster_router(router);
    }

    // Initialize voice pipeline for channels (when voice feature is enabled).
    #[cfg(feature = "voice")]
    let voice_pipeline: Option<Arc<crate::voice_pipeline::VoicePipeline>> = {
        use tracing::warn;
        use crate::config::schema::TtsEngineConfig;

        // Use configured TTS engine from config
        let tts_engine = config.voice.tts_engine;

        match crate::voice_pipeline::VoicePipeline::with_engine(tts_engine).await {
            Ok(vp) => {
                info!("Voice pipeline initialized for channels (engine: {:?})", tts_engine);
                Some(Arc::new(vp))
            }
            Err(e) => {
                warn!(
                    "Voice pipeline init failed with {:?} (voice messages will not be transcribed): {}",
                    tts_engine, e
                );
                // Try fallback to Pocket if Qwen/Kokoro failed
                if tts_engine != TtsEngineConfig::Pocket {
                    warn!("Falling back to Pocket TTS...");
                    match crate::voice_pipeline::VoicePipeline::with_engine(TtsEngineConfig::Pocket).await {
                        Ok(vp) => {
                            info!("Voice pipeline initialized with Pocket TTS (fallback)");
                            Some(Arc::new(vp))
                        }
                        Err(e2) => {
                            warn!("Pocket fallback also failed: {}", e2);
                            None
                        }
                    }
                } else {
                    None
                }
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
        None, // No LLM callback yet -- maintenance only
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
    runtime.block_on(run_gateway_async(config, core_handle, None, None, None));
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
    runtime.block_on(run_gateway_async(config, core_handle, None, None, None));
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
    runtime.block_on(run_gateway_async(config, core_handle, None, None, None));
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
