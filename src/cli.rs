//! CLI subcommand handlers for nanobot.
//!
//! This module contains all command implementations that were previously in main.rs.
//! Functions are extracted here to keep main.rs focused on argument parsing and routing.

use std::io::{self, Write};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use tokio::sync::mpsc;

use crate::agent::agent_loop::{build_swappable_core, AgentHandle, AgentLoop, RuntimeCounters, SharedCoreHandle};
use crate::agent::tuning::{score_feasible_profiles, select_optimal_from_input, OptimizationInput};
use crate::bus::events::{InboundMessage, OutboundMessage};
use crate::channels::manager::ChannelManager;
use crate::config::loader::{get_config_path, get_data_dir, load_config, save_config};
use crate::config::schema::Config;
use crate::cron::service::CronService;
use crate::cron::types::CronSchedule;
use crate::providers::base::LLMProvider;
use crate::providers::openai_compat::OpenAICompatProvider;
use crate::utils::helpers::get_workspace_path;

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use tempfile::tempdir;

    #[test]
    #[ignore] // Requires network access to Telegram API
    fn test_validate_telegram_token_valid() {
        assert!(validate_telegram_token("123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11"));
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
        assert_eq!(result, 50, "1M context cloud model should get 50 iterations");
    }

    #[test]
    fn test_effective_max_iterations_cloud_128k() {
        // Cloud model with 128K context
        let result = effective_max_iterations(20, 128_000, false);
        assert_eq!(result, 32, "128K context cloud model should get 32 iterations");
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

pub(crate) fn build_core_handle(
    config: &Config,
    local_port: &str,
    local_model_name: Option<&str>,
    compaction_port: Option<&str>,
    delegation_port: Option<&str>,
) -> SharedCoreHandle {
    let is_local = crate::LOCAL_MODE.load(Ordering::SeqCst);
    let provider: Arc<dyn LLMProvider> = if is_local {
        Arc::new(OpenAICompatProvider::new(
            "local",
            Some(&format!("http://localhost:{}/v1", local_port)),
            Some("local-model"),
        ))
    } else {
        create_provider(config)
    };

    let model = if is_local {
        format!("local:{}", local_model_name.unwrap_or("local-model"))
    } else {
        config.agents.defaults.model.clone()
    };

    let brave_key = if config.tools.web.search.api_key.is_empty() {
        None
    } else {
        Some(config.tools.web.search.api_key.clone())
    };

    // Auto-detect context size from local server; fall back to config/model default.
    let max_context_tokens = if is_local {
        crate::server::query_local_context_size(local_port).unwrap_or(config.agents.defaults.max_context_tokens)
    } else {
        model_context_size(&model, config.agents.defaults.max_context_tokens)
    };

    let cp: Option<Arc<dyn LLMProvider>> = if is_local {
        compaction_port.map(|p| -> Arc<dyn LLMProvider> {
            Arc::new(OpenAICompatProvider::new(
                "local-compaction",
                Some(&format!("http://localhost:{}/v1", p)),
                None,
            ))
        })
    } else {
        None
    };

    let dp: Option<Arc<dyn LLMProvider>> = delegation_port.map(|p| -> Arc<dyn LLMProvider> {
        Arc::new(OpenAICompatProvider::new(
            "local-delegation",
            Some(&format!("http://localhost:{}/v1", p)),
            None,
        ))
    });

    let max_iters = effective_max_iterations(
        config.agents.defaults.max_tool_iterations,
        max_context_tokens,
        is_local,
    );

    let core = build_swappable_core(
        provider,
        config.workspace_path(),
        model,
        max_iters,
        config.agents.defaults.max_tokens,
        config.agents.defaults.temperature,
        max_context_tokens,
        brave_key,
        config.tools.exec_.timeout,
        config.tools.exec_.restrict_to_workspace,
        config.memory.clone(),
        is_local,
        cp,
        config.tool_delegation.clone(),
        config.provenance.clone(),
        config.agents.defaults.max_tool_result_chars,
        dp,
    );
    let counters = Arc::new(RuntimeCounters::new(max_context_tokens));
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
) {
    let is_local = crate::LOCAL_MODE.load(Ordering::SeqCst);
    let provider: Arc<dyn LLMProvider> = if is_local {
        Arc::new(OpenAICompatProvider::new(
            "local",
            Some(&format!("http://localhost:{}/v1", local_port)),
            Some("local-model"),
        ))
    } else {
        create_provider(config)
    };

    let model = if is_local {
        format!("local:{}", local_model_name.unwrap_or("local-model"))
    } else {
        config.agents.defaults.model.clone()
    };

    let brave_key = if config.tools.web.search.api_key.is_empty() {
        None
    } else {
        Some(config.tools.web.search.api_key.clone())
    };

    // Auto-detect context size from local server; fall back to config/model default.
    let max_context_tokens = if is_local {
        crate::server::query_local_context_size(local_port).unwrap_or(config.agents.defaults.max_context_tokens)
    } else {
        model_context_size(&model, config.agents.defaults.max_context_tokens)
    };

    let cp: Option<Arc<dyn LLMProvider>> = if is_local {
        compaction_port.map(|p| -> Arc<dyn LLMProvider> {
            Arc::new(OpenAICompatProvider::new(
                "local-compaction",
                Some(&format!("http://localhost:{}/v1", p)),
                None,
            ))
        })
    } else {
        None
    };

    let dp: Option<Arc<dyn LLMProvider>> = delegation_port.map(|p| -> Arc<dyn LLMProvider> {
        Arc::new(OpenAICompatProvider::new(
            "local-delegation",
            Some(&format!("http://localhost:{}/v1", p)),
            None,
        ))
    });

    let max_iters = effective_max_iterations(
        config.agents.defaults.max_tool_iterations,
        max_context_tokens,
        is_local,
    );

    let new_core = build_swappable_core(
        provider,
        config.workspace_path(),
        model,
        max_iters,
        config.agents.defaults.max_tokens,
        config.agents.defaults.temperature,
        max_context_tokens,
        brave_key,
        config.tools.exec_.timeout,
        config.tools.exec_.restrict_to_workspace,
        config.memory.clone(),
        is_local,
        cp,
        config.tool_delegation.clone(),
        config.provenance.clone(),
        config.agents.defaults.max_tool_result_chars,
        dp,
    );
    // Swap only the core; counters survive.
    handle.swap_core(new_core);
    // Update max context since the new model may have a different size.
    handle.counters.last_context_max.store(max_context_tokens as u64, Ordering::Relaxed);
}

/// Create an agent loop with per-instance channels, using the shared core handle.
pub(crate) fn create_agent_loop(
    core_handle: SharedCoreHandle,
    config: &Config,
    cron_service: Option<Arc<CronService>>,
    email_config: Option<crate::config::schema::EmailConfig>,
    repl_display_tx: Option<mpsc::UnboundedSender<String>>,
) -> AgentLoop {
    let (inbound_tx, inbound_rx) = mpsc::unbounded_channel::<InboundMessage>();
    let (outbound_tx, _outbound_rx) = mpsc::unbounded_channel::<OutboundMessage>();

    AgentLoop::new(
        core_handle,
        inbound_rx,
        outbound_tx,
        inbound_tx,
        cron_service,
        config.agents.defaults.max_concurrent_chats,
        email_config,
        repl_display_tx,
    )
}

// ============================================================================
// Gateway
// ============================================================================

pub(crate) fn cmd_gateway(port: u16, verbose: bool) {
    if verbose {
        eprintln!("Verbose mode enabled");
    }

    println!("{} Starting nanobot gateway on port {}...", crate::LOGO, port);

    let config = load_config(None);
    check_api_key(&config);

    let core_handle = build_core_handle(&config, "8080", None, None, None);
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

    let mut agent_loop = AgentLoop::new(
        core_handle,
        inbound_rx,
        outbound_tx,
        inbound_tx.clone(),
        Some(cron_arc),
        config.agents.defaults.max_concurrent_chats,
        None, // gateway agent uses bus for email, not tools
        repl_display_tx,
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

        println!("  Heartbeat: every 30m");
    }

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

    let core_handle = build_core_handle(&config, "8080", None, None, None);
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

    let core_handle = build_core_handle(&config, "8080", None, None, None);
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

    let core_handle = build_core_handle(&config, "8080", None, None, None);
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
pub(crate) fn check_api_key(config: &Config) {
    let model = &config.agents.defaults.model;
    if config.get_api_key().is_none() && !model.starts_with("bedrock/") {
        eprintln!("Error: No API key configured.");
        eprintln!("Set one in ~/.nanobot/config.json under providers.openrouter.apiKey");
        std::process::exit(1);
    }
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

pub(crate) fn create_provider(config: &Config) -> Arc<dyn LLMProvider> {
    let api_key = config.get_api_key().unwrap_or_default();
    let api_base = config.get_api_base();
    let model = &config.agents.defaults.model;
    Arc::new(OpenAICompatProvider::new(
        &api_key,
        api_base.as_deref(),
        Some(model.as_str()),
    ))
}
