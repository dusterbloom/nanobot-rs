//! nanoclaw - A lightweight personal AI assistant framework in Rust.
//! Based on nanobot by HKUDS (https://github.com/HKUDS/nanobot).
//!
//! Local LLM support: Use Ctrl+L or /local to toggle between cloud and local mode.

mod agent;
mod bus;
mod channels;
mod config;
mod cron;
mod heartbeat;
mod providers;
mod session;
mod utils;

use std::io::{self, BufRead, Write as _};
use std::net::TcpListener;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use clap::{Parser, Subcommand};
use tokio::sync::mpsc;
use tracing::info;

use crate::bus::events::{InboundMessage, OutboundMessage};
use crate::config::loader::{get_config_path, get_data_dir, load_config, save_config};
use crate::config::schema::Config;
use crate::agent::agent_loop::AgentLoop;
use crate::channels::manager::ChannelManager;
use crate::cron::service::CronService;
use crate::cron::types::CronSchedule;
use crate::providers::base::LLMProvider;
use crate::providers::openai_compat::OpenAICompatProvider;
use crate::utils::helpers::get_workspace_path;

const VERSION: &str = "0.1.0";
const LOGO: &str = "\u{1F408}"; // cat emoji
const LOCAL_LOGO: &str = "\u{1F3E0}"; // house emoji for local mode

// Global flag for local mode
static LOCAL_MODE: AtomicBool = AtomicBool::new(false);

#[derive(Parser)]
#[command(name = "nanoclaw", about = "nanoclaw - Personal AI Assistant", version = VERSION)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Initialize nanoclaw configuration and workspace.
    Onboard,
    /// Interact with the agent directly.
    Agent {
        /// Message to send to the agent.
        #[arg(short, long)]
        message: Option<String>,
        /// Session ID.
        #[arg(short, long, default_value = "cli:default")]
        session: String,
        /// Use local LLM instead of cloud API.
        #[arg(short, long)]
        local: bool,
    },
    /// Start the nanoclaw gateway (channels + agent loop).
    Gateway {
        /// Gateway port.
        #[arg(short, long, default_value_t = 18790)]
        port: u16,
        /// Verbose logging.
        #[arg(short, long)]
        verbose: bool,
    },
    /// Show nanoclaw status.
    Status,
    /// Manage channels.
    Channels {
        #[command(subcommand)]
        action: ChannelsAction,
    },
    /// Manage scheduled tasks.
    Cron {
        #[command(subcommand)]
        action: CronAction,
    },
}

#[derive(Subcommand)]
enum ChannelsAction {
    /// Show channel status.
    Status,
}

#[derive(Subcommand)]
enum CronAction {
    /// List scheduled jobs.
    List {
        /// Include disabled jobs.
        #[arg(short, long)]
        all: bool,
    },
    /// Add a scheduled job.
    Add {
        /// Job name.
        #[arg(short, long)]
        name: String,
        /// Message for agent.
        #[arg(short, long)]
        message: String,
        /// Run every N seconds.
        #[arg(short, long)]
        every: Option<u64>,
        /// Cron expression.
        #[arg(short, long)]
        cron: Option<String>,
        /// Deliver response to channel.
        #[arg(short, long)]
        deliver: bool,
        /// Recipient for delivery.
        #[arg(long)]
        to: Option<String>,
        /// Channel for delivery.
        #[arg(long)]
        channel: Option<String>,
    },
    /// Remove a scheduled job.
    Remove {
        /// Job ID to remove.
        job_id: String,
    },
    /// Enable or disable a job.
    Enable {
        /// Job ID.
        job_id: String,
        /// Disable instead of enable.
        #[arg(long)]
        disable: bool,
    },
}

fn main() {
    let cli = Cli::parse();

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    match cli.command {
        Commands::Onboard => cmd_onboard(),
        Commands::Agent { message, session, local } => cmd_agent(message, session, local),
        Commands::Gateway { port, verbose } => cmd_gateway(port, verbose),
        Commands::Status => cmd_status(),
        Commands::Channels { action } => match action {
            ChannelsAction::Status => cmd_channels_status(),
        },
        Commands::Cron { action } => match action {
            CronAction::List { all } => cmd_cron_list(all),
            CronAction::Add {
                name, message, every, cron, deliver, to, channel,
            } => cmd_cron_add(name, message, every, cron, deliver, to, channel),
            CronAction::Remove { job_id } => cmd_cron_remove(job_id),
            CronAction::Enable { job_id, disable } => cmd_cron_enable(job_id, disable),
        },
    }
}

// ============================================================================
// Onboard
// ============================================================================

fn cmd_onboard() {
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

    println!("\n{} nanoclaw is ready!", LOGO);
    println!("\nNext steps:");
    println!("  1. Add your API key to ~/.nanoclaw/config.json");
    println!("     Get one at: https://openrouter.ai/keys");
    println!("  2. Chat: nanoclaw agent -m \"Hello!\"");
}

fn create_workspace_templates(workspace: &std::path::Path) {
    let templates: Vec<(&str, &str)> = vec![
        ("AGENTS.md", "# Agent Instructions\n\nYou are a helpful AI assistant. Be concise, accurate, and friendly.\n\n## Guidelines\n\n- Always explain what you're doing before taking actions\n- Ask for clarification when the request is ambiguous\n- Use tools to help accomplish tasks\n- Remember important information in your memory files\n"),
        ("SOUL.md", "# Soul\n\nI am nanoclaw, a lightweight AI assistant.\n\n## Personality\n\n- Helpful and friendly\n- Concise and to the point\n- Curious and eager to learn\n\n## Values\n\n- Accuracy over speed\n- User privacy and safety\n- Transparency in actions\n"),
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
// Agent
// ============================================================================

fn cmd_agent(message: Option<String>, session_id: String, local_flag: bool) {
    let config = load_config(None);
    
    // Check environment variable for local mode
    let local_env = std::env::var("NANOCLAW_LOCAL")
        .map(|v| v == "1" || v.to_lowercase() == "true")
        .unwrap_or(false);
    
    // Set initial local mode from flag or environment
    if local_flag || local_env {
        LOCAL_MODE.store(true, Ordering::SeqCst);
    }
    
    let mut local_port = std::env::var("NANOCLAW_LOCAL_PORT")
        .unwrap_or_else(|_| "8080".to_string());
    
    // Check if we can proceed
    let is_local = LOCAL_MODE.load(Ordering::SeqCst);
    if !is_local {
        let api_key = config.get_api_key();
        let model = &config.agents.defaults.model;
        if api_key.is_none() && !model.starts_with("bedrock/") {
            eprintln!("Error: No API key configured.");
            eprintln!("Set one in ~/.nanoclaw/config.json under providers.openrouter.apiKey");
            eprintln!("Or use --local flag to use a local LLM server.");
            std::process::exit(1);
        }
    }

    let runtime = tokio::runtime::Runtime::new().expect("Failed to create tokio runtime");

    runtime.block_on(async {
        // Create initial agent loop
        let (mut agent_loop, config) = create_agent_loop(&config, &local_port).await;

        if let Some(msg) = message {
            let logo = if LOCAL_MODE.load(Ordering::SeqCst) { LOCAL_LOGO } else { LOGO };
            let response = agent_loop
                .process_direct(&msg, &session_id, "cli", "direct")
                .await;
            println!("\n{} {}", logo, response);
        } else {
            print_mode_banner(&local_port);
            println!("Type /local to toggle local/cloud mode, Ctrl+C to exit\n");

            let mut llama_process: Option<std::process::Child> = None;

            loop {
                let is_local = LOCAL_MODE.load(Ordering::SeqCst);
                let prompt = if is_local {
                    format!("{} You: ", LOCAL_LOGO)
                } else {
                    "You: ".to_string()
                };
                print!("{}", prompt);
                io::stdout().flush().ok();
                
                let mut input = String::new();
                match io::stdin().read_line(&mut input) {
                    Ok(0) | Err(_) => break,
                    _ => {}
                }
                
                let input = input.trim();
                if input.is_empty() {
                    continue;
                }
                
                // Handle mode toggle commands
                if input == "/local" || input == "/l" {
                    let currently_local = LOCAL_MODE.load(Ordering::SeqCst);

                    if !currently_local {
                        // Toggle ON: spawn llama.cpp server
                        let port = find_available_port(8080);
                        println!("\nStarting llama.cpp server on port {}...", port);

                        match spawn_llama_server(port) {
                            Ok(child) => {
                                llama_process = Some(child);
                                if wait_for_server_ready(port, 30).await {
                                    local_port = port.to_string();
                                    LOCAL_MODE.store(true, Ordering::SeqCst);
                                    let (new_loop, _) = create_agent_loop(&config, &local_port).await;
                                    agent_loop = new_loop;
                                    print_mode_banner(&local_port);
                                } else {
                                    eprintln!("\n\u{26a0}\u{fe0f}  Server failed to start within 30 seconds");
                                    if let Some(ref mut child) = llama_process {
                                        child.kill().ok();
                                        child.wait().ok();
                                    }
                                    llama_process = None;
                                }
                            }
                            Err(e) => {
                                eprintln!("\n\u{26a0}\u{fe0f}  Failed to start llama.cpp server: {}", e);
                            }
                        }
                    } else {
                        // Toggle OFF: kill server and switch to cloud
                        if let Some(ref mut child) = llama_process {
                            println!("\nStopping llama.cpp server...");
                            child.kill().ok();
                            child.wait().ok();
                        }
                        llama_process = None;
                        LOCAL_MODE.store(false, Ordering::SeqCst);
                        let (new_loop, _) = create_agent_loop(&config, &local_port).await;
                        agent_loop = new_loop;
                        print_mode_banner(&local_port);
                    }

                    continue;
                }
                
                // Handle help command
                if input == "/help" || input == "/h" || input == "/?" {
                    println!("\nCommands:");
                    println!("  /local, /l  - Toggle between local and cloud mode");
                    println!("  /status     - Show current mode and model info");
                    println!("  /help, /h   - Show this help");
                    println!("  Ctrl+C      - Exit\n");
                    continue;
                }
                
                // Handle status command
                if input == "/status" || input == "/s" {
                    print_mode_banner(&local_port);
                    continue;
                }
                
                let response = agent_loop
                    .process_direct(input, &session_id, "cli", "direct")
                    .await;
                
                let logo = if LOCAL_MODE.load(Ordering::SeqCst) { LOCAL_LOGO } else { LOGO };
                println!("\n{} {}\n", logo, response);
            }
            // Cleanup: kill llama.cpp server if still running
            if let Some(ref mut child) = llama_process {
                println!("Stopping llama.cpp server...");
                child.kill().ok();
                child.wait().ok();
            }

            println!("Goodbye!");
        }
    });
}

/// Create an agent loop with the appropriate provider based on local mode.
async fn create_agent_loop(config: &Config, local_port: &str) -> (AgentLoop, Config) {
    let (inbound_tx, inbound_rx) = mpsc::unbounded_channel::<InboundMessage>();
    let (outbound_tx, _outbound_rx) = mpsc::unbounded_channel::<OutboundMessage>();

    let is_local = LOCAL_MODE.load(Ordering::SeqCst);
    let provider: Arc<dyn LLMProvider> = if is_local {
        // Create local provider pointing to llama.cpp server
        Arc::new(OpenAICompatProvider::new(
            "local", // API key not needed for local
            Some(&format!("http://localhost:{}/v1", local_port)),
            Some("local-model"),
        ))
    } else {
        create_provider(config)
    };
    
    let model = if is_local {
        "local-model".to_string()
    } else {
        config.agents.defaults.model.clone()
    };

    let brave_key = if config.tools.web.search.api_key.is_empty() {
        None
    } else {
        Some(config.tools.web.search.api_key.clone())
    };

    let cron_store_path = get_data_dir().join("cron").join("jobs.json");
    let cron_service = Arc::new(CronService::new(cron_store_path));

    let agent_loop = AgentLoop::new(
        inbound_rx,
        outbound_tx,
        inbound_tx,
        provider,
        config.workspace_path(),
        model,
        config.agents.defaults.max_tool_iterations,
        brave_key,
        config.tools.exec_.timeout,
        config.tools.exec_.restrict_to_workspace,
        Some(cron_service),
    );

    (agent_loop, config.clone())
}

/// Print the current mode banner.
fn print_mode_banner(local_port: &str) {
    let is_local = LOCAL_MODE.load(Ordering::SeqCst);
    if is_local {
        println!("\n{} LOCAL MODE - Using llama.cpp server on port {}", LOCAL_LOGO, local_port);
        // Try to get model info
        let props_url = format!("http://localhost:{}/props", local_port);
        if let Ok(resp) = reqwest::blocking::get(&props_url) {
            if let Ok(json) = resp.json::<serde_json::Value>() {
                if let Some(model) = json.get("default_generation_settings")
                    .and_then(|s| s.get("model"))
                    .and_then(|m| m.as_str()) 
                {
                    let model_name = std::path::Path::new(model)
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or(model);
                    println!("   Model: {}", model_name);
                }
            }
        }
    } else {
        let config = load_config(None);
        println!("\n{} CLOUD MODE - Using {} via API", LOGO, config.agents.defaults.model);
    }
    println!();
}

// ============================================================================
// Gateway
// ============================================================================

fn cmd_gateway(port: u16, verbose: bool) {
    if verbose {
        eprintln!("Verbose mode enabled");
    }

    println!("{} Starting nanoclaw gateway on port {}...", LOGO, port);

    let config = load_config(None);
    let api_key = config.get_api_key();
    let model = config.agents.defaults.model.clone();

    if api_key.is_none() && !model.starts_with("bedrock/") {
        eprintln!("Error: No API key configured.");
        std::process::exit(1);
    }

    let runtime = tokio::runtime::Runtime::new().expect("Failed to create tokio runtime");

    runtime.block_on(async {
        let (inbound_tx, inbound_rx) = mpsc::unbounded_channel::<InboundMessage>();
        let (outbound_tx, outbound_rx) = mpsc::unbounded_channel::<OutboundMessage>();

        let provider = create_provider(&config);
        let brave_key = if config.tools.web.search.api_key.is_empty() {
            None
        } else {
            Some(config.tools.web.search.api_key.clone())
        };

        let cron_store_path = get_data_dir().join("cron").join("jobs.json");
        let mut cron_service = CronService::new(cron_store_path);
        cron_service.start().await;
        let cron_status = cron_service.status();
        let cron_arc = Arc::new(cron_service);

        let mut agent_loop = AgentLoop::new(
            inbound_rx,
            outbound_tx,
            inbound_tx.clone(),
            provider,
            config.workspace_path(),
            model,
            config.agents.defaults.max_tool_iterations,
            brave_key,
            config.tools.exec_.timeout,
            config.tools.exec_.restrict_to_workspace,
            Some(cron_arc),
        );

        let channel_manager = ChannelManager::new(&config, inbound_tx, outbound_rx);

        let enabled = channel_manager.enabled_channels();
        if !enabled.is_empty() {
            println!("  Channels enabled: {}", enabled.join(", "));
        } else {
            println!("  Warning: No channels enabled");
        }

        {
            let job_count = cron_status.get("jobs").and_then(|v| v.as_i64()).unwrap_or(0);
            if job_count > 0 {
                println!("  Cron: {} scheduled jobs", job_count);
            }
        }

        println!("  Heartbeat: every 30m");

        tokio::select! {
            _ = agent_loop.run() => {
                info!("Agent loop ended");
            }
            _ = channel_manager.start_all() => {
                info!("Channel manager ended");
            }
            _ = tokio::signal::ctrl_c() => {
                println!("\nShutting down...");
            }
        }

        agent_loop.stop();
        channel_manager.stop_all().await;
    });
}

// ============================================================================
// Status
// ============================================================================

fn cmd_status() {
    let config_path = get_config_path();
    let config = load_config(None);
    let workspace = config.workspace_path();

    println!("{} nanoclaw Status\n", LOGO);
    println!(
        "Config: {} [{}]",
        config_path.display(),
        if config_path.exists() { "ok" } else { "missing" }
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
            if config.providers.openrouter.api_key.is_empty() { "not set" } else { "configured" }
        );
        println!(
            "Anthropic API: {}",
            if config.providers.anthropic.api_key.is_empty() { "not set" } else { "configured" }
        );
        println!(
            "OpenAI API: {}",
            if config.providers.openai.api_key.is_empty() { "not set" } else { "configured" }
        );
        println!(
            "Gemini API: {}",
            if config.providers.gemini.api_key.is_empty() { "not set" } else { "configured" }
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
            Ok(resp) if resp.status().is_success() => "🟢 running",
            _ => "⚪ stopped",
        };
        println!("  {} (port {}): {}", name, port, status);
    }
}

// ============================================================================
// Channels
// ============================================================================

fn cmd_channels_status() {
    let config = load_config(None);
    println!("Channel Status\n");
    println!(
        "  WhatsApp: {} ({})",
        if config.channels.whatsapp.enabled { "enabled" } else { "disabled" },
        config.channels.whatsapp.bridge_url
    );
    let tg_info = if config.channels.telegram.token.is_empty() {
        "not configured".to_string()
    } else {
        let t = &config.channels.telegram.token;
        format!("token: {}...", &t[..t.len().min(10)])
    };
    println!(
        "  Telegram: {} ({})",
        if config.channels.telegram.enabled { "enabled" } else { "disabled" },
        tg_info
    );
    println!(
        "  Feishu: {}",
        if config.channels.feishu.enabled { "enabled" } else { "disabled" }
    );
}

// ============================================================================
// Cron
// ============================================================================

fn cmd_cron_list(include_all: bool) {
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

fn cmd_cron_add(
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

fn cmd_cron_remove(job_id: String) {
    let store_path = get_data_dir().join("cron").join("jobs.json");
    let mut service = CronService::new(store_path);
    if service.remove_job(&job_id) {
        println!("  Removed job {}", job_id);
    } else {
        eprintln!("Job {} not found", job_id);
    }
}

fn cmd_cron_enable(job_id: String, disable: bool) {
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

fn find_available_port(start: u16) -> u16 {
    for port in start..=start.saturating_add(99) {
        if TcpListener::bind(("127.0.0.1", port)).is_ok() {
            return port;
        }
    }
    start // fallback
}

fn spawn_llama_server(port: u16) -> Result<std::process::Child, String> {
    let home = dirs::home_dir().ok_or("Could not determine home directory")?;
    let server_path = home.join("llama.cpp/build/bin/llama-server");
    let model_path = home.join("models/NVIDIA-Nemotron-Nano-8B-v1-Q4_K_M.gguf");

    if !server_path.exists() {
        return Err(format!("llama-server not found at {}", server_path.display()));
    }
    if !model_path.exists() {
        return Err(format!("Model not found at {}", model_path.display()));
    }

    std::process::Command::new(&server_path)
        .arg("--model")
        .arg(&model_path)
        .arg("--port")
        .arg(port.to_string())
        .arg("--ctx-size")
        .arg("16384")
        .arg("--n-gpu-layers")
        .arg("99")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .map_err(|e| format!("Failed to spawn llama-server: {}", e))
}

async fn wait_for_server_ready(port: u16, timeout_secs: u64) -> bool {
    let client = reqwest::Client::new();
    let url = format!("http://localhost:{}/health", port);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(timeout_secs);

    while std::time::Instant::now() < deadline {
        if let Ok(resp) = client.get(&url).send().await {
            if resp.status().is_success() {
                return true;
            }
        }
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }
    false
}

fn create_provider(config: &Config) -> Arc<dyn LLMProvider> {
    let api_key = config.get_api_key().unwrap_or_default();
    let api_base = config.get_api_base();
    let model = &config.agents.defaults.model;
    Arc::new(OpenAICompatProvider::new(
        &api_key,
        api_base.as_deref(),
        Some(model.as_str()),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    // ======================================================================
    // find_available_port
    // ======================================================================

    #[test]
    fn test_find_port_returns_bindable_port() {
        let port = find_available_port(49152);
        // The returned port should actually be free
        let listener = TcpListener::bind(("127.0.0.1", port));
        assert!(listener.is_ok(), "Port {} should be bindable", port);
    }

    #[test]
    fn test_find_port_within_range() {
        let start = 40000;
        let port = find_available_port(start);
        assert!(port >= start, "Port {} < start {}", port, start);
        assert!(port < start + 100, "Port {} >= start + 100", port);
    }

    #[test]
    fn test_find_port_skips_occupied() {
        // Occupy a port, then ask for one starting there
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let occupied = listener.local_addr().unwrap().port();

        let found = find_available_port(occupied);
        assert_ne!(found, occupied, "Should skip the occupied port");
        // Keep listener alive so the port stays occupied
        drop(listener);
    }

    #[test]
    fn test_find_port_skips_consecutive_occupied() {
        let base: u16 = 48500;
        let l1 = TcpListener::bind(("127.0.0.1", base));
        let l2 = TcpListener::bind(("127.0.0.1", base + 1));

        if let (Ok(_l1), Ok(_l2)) = (l1, l2) {
            let found = find_available_port(base);
            assert!(found >= base + 2, "Should skip both occupied ports, got {}", found);
        }
        // If we can't bind both (already in use), test is inconclusive — that's fine
    }

    #[test]
    fn test_find_port_high_start_no_overflow() {
        // Edge case: start near u16::MAX should not panic from overflow
        let port = find_available_port(65500);
        assert!(port >= 65500);
    }

    // ======================================================================
    // spawn_llama_server
    // ======================================================================

    #[test]
    fn test_spawn_server_errors_when_binary_missing() {
        let home = dirs::home_dir().unwrap();
        let server_path = home.join("llama.cpp/build/bin/llama-server");

        if !server_path.exists() {
            let result = spawn_llama_server(19876);
            assert!(result.is_err());
            assert!(
                result.unwrap_err().contains("llama-server not found"),
                "Should report missing binary"
            );
        }
    }

    #[test]
    fn test_spawn_server_errors_when_model_missing() {
        let home = dirs::home_dir().unwrap();
        let server_path = home.join("llama.cpp/build/bin/llama-server");
        let model_path = home.join("models/NVIDIA-Nemotron-Nano-8B-v1-Q4_K_M.gguf");

        if server_path.exists() && !model_path.exists() {
            let result = spawn_llama_server(19877);
            assert!(result.is_err());
            assert!(
                result.unwrap_err().contains("Model not found"),
                "Should report missing model"
            );
        }
    }

    // ======================================================================
    // wait_for_server_ready
    // ======================================================================

    #[tokio::test]
    async fn test_wait_timeout_when_no_server() {
        let result = wait_for_server_ready(19999, 1).await;
        assert!(!result, "Should return false when no server running");
    }

    #[tokio::test]
    async fn test_wait_zero_timeout_returns_false() {
        let result = wait_for_server_ready(19998, 0).await;
        assert!(!result, "Should return false immediately with zero timeout");
    }

    #[tokio::test]
    async fn test_wait_finds_healthy_server() {
        // Spin up a mock HTTP server that returns 200 on any request
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();

        tokio::spawn(async move {
            loop {
                if let Ok((mut stream, _)) = listener.accept().await {
                    use tokio::io::{AsyncReadExt, AsyncWriteExt};
                    let mut buf = [0u8; 1024];
                    let _ = stream.read(&mut buf).await;
                    let resp = "HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Length: 2\r\n\r\nok";
                    stream.write_all(resp.as_bytes()).await.ok();
                }
            }
        });

        let result = wait_for_server_ready(port, 5).await;
        assert!(result, "Should detect the healthy server");
    }

    #[tokio::test]
    async fn test_wait_retries_on_503_then_succeeds() {
        // Mock server returns 503 twice, then 200
        let request_count = Arc::new(AtomicUsize::new(0));
        let count_clone = request_count.clone();

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();

        tokio::spawn(async move {
            loop {
                if let Ok((mut stream, _)) = listener.accept().await {
                    use tokio::io::{AsyncReadExt, AsyncWriteExt};
                    let mut buf = [0u8; 1024];
                    let _ = stream.read(&mut buf).await;

                    let n = count_clone.fetch_add(1, Ordering::SeqCst);
                    let resp = if n < 2 {
                        "HTTP/1.1 503 Service Unavailable\r\nConnection: close\r\nContent-Length: 7\r\n\r\nloading"
                    } else {
                        "HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Length: 2\r\n\r\nok"
                    };
                    stream.write_all(resp.as_bytes()).await.ok();
                }
            }
        });

        let result = wait_for_server_ready(port, 10).await;
        assert!(result, "Should succeed after retries");
        assert!(
            request_count.load(Ordering::SeqCst) >= 3,
            "Should have retried at least 3 times"
        );
    }
}
