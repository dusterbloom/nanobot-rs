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
#[cfg(feature = "voice")]
mod voice;

use std::io::{self, BufRead, Write as _};
use std::net::TcpListener;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use clap::{Parser, Subcommand};
use rustyline::error::ReadlineError;
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
#[cfg(feature = "voice")]
const VOICE_LOGO: &str = "\u{1F3A4}"; // microphone emoji for voice mode

/// Build a styled termimad skin for rendering LLM markdown responses.
fn make_skin() -> termimad::MadSkin {
    use termimad::crossterm::style::Color;
    let mut skin = termimad::MadSkin::default_dark();
    skin.headers[0].set_fg(Color::Cyan);
    skin.headers[1].set_fg(Color::Cyan);
    skin.bold.set_fg(Color::White);
    skin.italic.set_fg(Color::Magenta);
    skin.inline_code.set_fg(Color::Green);
    skin.code_block.set_fg(Color::Green);
    skin
}

/// ANSI escape helpers for colored terminal output.
mod tui {
    pub const RESET: &str = "\x1b[0m";
    pub const BOLD: &str = "\x1b[1m";
    pub const DIM: &str = "\x1b[2m";
    pub const CYAN: &str = "\x1b[36m";
    pub const GREEN: &str = "\x1b[32m";
    pub const YELLOW: &str = "\x1b[33m";
    pub const MAGENTA: &str = "\x1b[35m";
    pub const WHITE: &str = "\x1b[97m";
    pub const CLEAR_SCREEN: &str = "\x1b[2J\x1b[H";
    pub const HIDE_CURSOR: &str = "\x1b[?25l";
    pub const SHOW_CURSOR: &str = "\x1b[?25h";

    /// Print the nanobot demoscene-style ASCII logo.
    pub fn print_logo() {
        println!("  {BOLD}{CYAN} _____             _       _   {RESET}");
        println!("  {BOLD}{WHITE}|   | |___ ___ ___| |_ ___| |_ {RESET}");
        println!("  {BOLD}{WHITE}| | | | .'|   | . | . | . |  _|{RESET}");
        println!("  {BOLD}{CYAN}|_|___|__,|_|_|___|___|___|_|  {RESET}");
    }

    /// Animated loading sequence.
    pub fn loading_animation(message: &str) {
        use std::io::Write;
        let frames = ["   ", ".  ", ".. ", "..."];
        print!("{HIDE_CURSOR}");
        for i in 0..8 {
            print!("\r  {DIM}{}{}{RESET}  ", message, frames[i % frames.len()]);
            std::io::stdout().flush().ok();
            std::thread::sleep(std::time::Duration::from_millis(150));
        }
        print!("\r{}\r", " ".repeat(60)); // clear the line
        print!("{SHOW_CURSOR}");
        std::io::stdout().flush().ok();
    }
}

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
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn,nanoclaw=info,ort=off,supertonic=off")),
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
            let skin = make_skin();
            let response = agent_loop
                .process_direct(&msg, &session_id, "cli", "direct")
                .await;
            println!();
            skin.print_text(&response);
        } else {
            print_startup_splash(&local_port);

            let mut llama_process: Option<std::process::Child> = None;
            let default_model = dirs::home_dir().unwrap().join("models").join(DEFAULT_LOCAL_MODEL);
            let mut current_model_path: std::path::PathBuf = default_model;
            #[cfg(feature = "voice")]
            let mut voice_session: Option<voice::VoiceSession> = None;

            // Readline editor with history
            let history_path = get_data_dir().join("history.txt");
            let mut rl = rustyline::DefaultEditor::new()
                .expect("Failed to create line editor");
            let _ = rl.load_history(&history_path);

            // Markdown skin for rendering LLM responses
            let skin = make_skin();

            loop {
                let is_local = LOCAL_MODE.load(Ordering::SeqCst);
                #[cfg(feature = "voice")]
                let voice_on = voice_session.is_some();
                #[cfg(not(feature = "voice"))]
                let voice_on = false;

                let prompt = if voice_on {
                    #[cfg(feature = "voice")]
                    { format!("{} {}{}You:{} ", VOICE_LOGO, tui::BOLD, tui::MAGENTA, tui::RESET) }
                    #[cfg(not(feature = "voice"))]
                    { format!("{}{}You:{} ", tui::BOLD, tui::GREEN, tui::RESET) }
                } else if is_local {
                    format!("{} {}{}You:{} ", LOCAL_LOGO, tui::BOLD, tui::YELLOW, tui::RESET)
                } else {
                    format!("{}{}You:{} ", tui::BOLD, tui::GREEN, tui::RESET)
                };

                // === GET INPUT ===
                let input_text: String;
                let mut do_record = false;

                #[cfg(feature = "voice")]
                if voice_on {
                    print!("{}", prompt);
                    io::stdout().flush().ok();
                    match voice_read_input() {
                        VoiceAction::Record => {
                            do_record = true;
                            input_text = String::new();
                        }
                        VoiceAction::Text(t) => {
                            input_text = t;
                        }
                        VoiceAction::Exit => break,
                    }
                } else {
                    match rl.readline(&prompt) {
                        Ok(line) => {
                            let _ = rl.add_history_entry(&line);
                            input_text = line;
                        }
                        Err(ReadlineError::Interrupted | ReadlineError::Eof) => break,
                        Err(_) => break,
                    }
                }

                #[cfg(not(feature = "voice"))]
                {
                    match rl.readline(&prompt) {
                        Ok(line) => {
                            let _ = rl.add_history_entry(&line);
                            input_text = line;
                        }
                        Err(ReadlineError::Interrupted | ReadlineError::Eof) => break,
                        Err(_) => break,
                    }
                }

                // === VOICE RECORDING ===
                #[cfg(feature = "voice")]
                if do_record {
                    if let Some(ref mut vs) = voice_session {
                        vs.stop_playback();
                        let mut keep_recording = true;
                        while keep_recording {
                            keep_recording = false;
                            match vs.record_and_transcribe() {
                                Ok(Some(text)) => {
                                    println!();
                                    let response = agent_loop
                                        .process_direct(&text, &session_id, "voice", "direct")
                                        .await;
                                    println!();
                                    skin.print_text(&response);
                                    println!();
                                    let tts_text = strip_markdown_for_tts(&response);
                                    if !tts_text.is_empty() {
                                        if speak_interruptible(vs, &tts_text) {
                                            // User interrupted TTS — loop back to record
                                            keep_recording = true;
                                        }
                                    }
                                }
                                Ok(None) => println!("(no speech detected)\n"),
                                Err(e) => eprintln!("Recording error: {}\n", e),
                            }
                        }
                        drain_stdin();
                    }
                    continue;
                }

                // === TEXT INPUT ===
                let input = input_text.trim();
                if input.is_empty() { continue; }

                // Handle mode toggle commands
                if input == "/local" || input == "/l" {
                    let currently_local = LOCAL_MODE.load(Ordering::SeqCst);

                    if !currently_local {
                        // Toggle ON: check if a llama.cpp server is already running
                        let mut found_port: Option<u16> = None;
                        for port in 8080..=8089 {
                            let url = format!("http://localhost:{}/health", port);
                            if let Ok(resp) = reqwest::blocking::get(&url) {
                                if resp.status().is_success() {
                                    found_port = Some(port);
                                    break;
                                }
                            }
                        }

                        if let Some(port) = found_port {
                            // Reuse existing server
                            println!("\n  {}{}Reusing{} llama.cpp server on port {}", tui::BOLD, tui::YELLOW, tui::RESET, port);
                            local_port = port.to_string();
                            LOCAL_MODE.store(true, Ordering::SeqCst);
                            let (new_loop, _) = create_agent_loop(&config, &local_port).await;
                            agent_loop = new_loop;
                            print_mode_banner(&local_port);
                        } else {
                            // Kill any orphaned servers from previous runs
                            kill_stale_llama_servers();
                            let port = find_available_port(8080);
                            println!("\n  {}{}Starting{} llama.cpp server on port {}...", tui::BOLD, tui::YELLOW, tui::RESET, port);

                            match spawn_llama_server(port, &current_model_path) {
                                Ok(child) => {
                                    llama_process = Some(child);
                                    if wait_for_server_ready(port, 120, &mut llama_process).await {
                                        local_port = port.to_string();
                                        LOCAL_MODE.store(true, Ordering::SeqCst);
                                        let (new_loop, _) = create_agent_loop(&config, &local_port).await;
                                        agent_loop = new_loop;
                                        print_mode_banner(&local_port);
                                    } else {
                                        println!("  {}{}Server failed to start{}", tui::BOLD, tui::YELLOW, tui::RESET);
                                        if let Some(ref mut child) = llama_process {
                                            child.kill().ok();
                                            child.wait().ok();
                                        }
                                        llama_process = None;
                                        println!("  {}Remaining in cloud mode{}\n", tui::DIM, tui::RESET);
                                    }
                                }
                                Err(e) => {
                                    println!("\n  {}{}Failed to start server:{} {}", tui::BOLD, tui::YELLOW, tui::RESET, e);
                                    println!("  {}Remaining in cloud mode{}\n", tui::DIM, tui::RESET);
                                }
                            }
                        }
                    } else {
                        // Toggle OFF: kill server and switch to cloud
                        if let Some(ref mut child) = llama_process {
                            println!("\n  {}Stopping llama.cpp server...{}", tui::DIM, tui::RESET);
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

                // Handle model selection
                if input == "/model" || input == "/m" {
                    let models = list_local_models();
                    if models.is_empty() {
                        println!("\nNo .gguf models found in ~/models/\n");
                        continue;
                    }

                    println!("\nAvailable models:");
                    for (i, path) in models.iter().enumerate() {
                        let name = path.file_name().unwrap().to_string_lossy();
                        let size_mb = std::fs::metadata(path)
                            .map(|m| m.len() / 1_048_576)
                            .unwrap_or(0);
                        let marker = if *path == current_model_path { " (active)" } else { "" };
                        println!("  [{}] {} ({} MB){}", i + 1, name, size_mb, marker);
                    }
                    let model_prompt = format!("Select model [1-{}] or Enter to cancel: ", models.len());
                    let choice = match rl.readline(&model_prompt) {
                        Ok(line) => line,
                        Err(_) => { continue; }
                    };
                    let choice = choice.trim();
                    if choice.is_empty() {
                        continue;
                    }
                    let idx: usize = match choice.parse::<usize>() {
                        Ok(n) if n >= 1 && n <= models.len() => n - 1,
                        _ => {
                            println!("Invalid selection.\n");
                            continue;
                        }
                    };

                    let selected = &models[idx];
                    current_model_path = selected.clone();
                    let name = selected.file_name().unwrap().to_string_lossy();
                    println!("\nSelected: {}", name);

                    // If local mode is active, restart the server with the new model
                    if LOCAL_MODE.load(Ordering::SeqCst) {
                        // Kill existing server we spawned + any orphans
                        if let Some(ref mut child) = llama_process {
                            println!("  {}Stopping current server...{}", tui::DIM, tui::RESET);
                            child.kill().ok();
                            child.wait().ok();
                        }
                        llama_process = None;
                        kill_stale_llama_servers();

                        let port = find_available_port(8080);
                        println!("  {}{}Starting{} llama.cpp server on port {}...", tui::BOLD, tui::YELLOW, tui::RESET, port);

                        match spawn_llama_server(port, &current_model_path) {
                            Ok(child) => {
                                llama_process = Some(child);
                                if wait_for_server_ready(port, 120, &mut llama_process).await {
                                    local_port = port.to_string();
                                    let (new_loop, _) = create_agent_loop(&config, &local_port).await;
                                    agent_loop = new_loop;
                                    print_mode_banner(&local_port);
                                } else {
                                    println!("  {}{}Server failed to start{}", tui::BOLD, tui::YELLOW, tui::RESET);
                                    if let Some(ref mut child) = llama_process {
                                        child.kill().ok();
                                        child.wait().ok();
                                    }
                                    llama_process = None;
                                }
                            }
                            Err(e) => {
                                println!("\n  {}{}Failed to start server:{} {}", tui::BOLD, tui::YELLOW, tui::RESET, e);
                            }
                        }
                    } else {
                        println!("Model will be used next time you toggle /local on.\n");
                    }

                    continue;
                }

                // Handle voice toggle
                #[cfg(feature = "voice")]
                if input == "/voice" || input == "/v" {
                    if voice_session.is_some() {
                        if let Some(ref mut vs) = voice_session {
                            vs.stop_playback();
                        }
                        voice_session = None;
                        println!("\nVoice mode OFF\n");
                    } else {
                        match voice::VoiceSession::new().await {
                            Ok(vs) => {
                                voice_session = Some(vs);
                                println!("\nVoice mode ON. Ctrl+Space or Enter to speak, type for text.\n");
                            }
                            Err(e) => eprintln!("\nFailed to start voice mode: {}\n", e),
                        }
                    }
                    continue;
                }

                // Handle help command
                if input == "/help" || input == "/h" || input == "/?" {
                    println!("\nCommands:");
                    println!("  /local, /l  - Toggle between local and cloud mode");
                    println!("  /model, /m  - Select local model from ~/models/");
                    println!("  /voice, /v  - Toggle voice mode (Ctrl+Space or Enter to speak)");
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

                // Process message
                let channel = if voice_on { "voice" } else { "cli" };
                let response = agent_loop
                    .process_direct(input, &session_id, channel, "direct")
                    .await;

                println!();
                skin.print_text(&response);
                println!();

                #[cfg(feature = "voice")]
                if let Some(ref mut vs) = voice_session {
                    let tts_text = strip_markdown_for_tts(&response);
                    if !tts_text.is_empty() {
                        speak_interruptible(vs, &tts_text);
                    }
                }
            }
            // Cleanup: kill llama.cpp server if still running
            // Save readline history
            let _ = rl.save_history(&history_path);

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
        config.agents.defaults.max_tokens,
        config.agents.defaults.temperature,
        config.agents.defaults.max_context_tokens,
        brave_key,
        config.tools.exec_.timeout,
        config.tools.exec_.restrict_to_workspace,
        Some(cron_service),
    );

    (agent_loop, config.clone())
}

/// Print the current mode banner (compact, for mode switches mid-session).
fn print_mode_banner(local_port: &str) {
    use tui::*;
    let is_local = LOCAL_MODE.load(Ordering::SeqCst);
    println!();
    if is_local {
        println!("  {BOLD}{YELLOW}LOCAL MODE{RESET} {DIM}llama.cpp on port {local_port}{RESET}");
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
                    println!("  {DIM}Model: {RESET}{GREEN}{model_name}{RESET}");
                }
            }
        }
    } else {
        let config = load_config(None);
        println!("  {BOLD}{CYAN}CLOUD MODE{RESET} {DIM}{}{RESET}", config.agents.defaults.model);
    }
    println!();
}

/// Full startup splash: clear screen, ASCII logo, mode info, hints.
fn print_startup_splash(local_port: &str) {
    use tui::*;
    // Clear the terminal for a fresh start.
    print!("{CLEAR_SCREEN}");
    std::io::Write::flush(&mut std::io::stdout()).ok();

    tui::print_logo();

    let is_local = LOCAL_MODE.load(Ordering::SeqCst);
    if is_local {
        println!("  {BOLD}{YELLOW}LOCAL{RESET} {DIM}llama.cpp :{local_port}{RESET}");
    } else {
        let config = load_config(None);
        println!("  {BOLD}{CYAN}CLOUD{RESET} {DIM}{}{RESET}", config.agents.defaults.model);
    }
    println!("  {DIM}v{VERSION}  |  /local  /model  /voice  Ctrl+C quit{RESET}");
    println!();

    // Brief loading animation
    tui::loading_animation("Initializing agent");
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
            config.agents.defaults.max_tokens,
            config.agents.defaults.temperature,
            config.agents.defaults.max_context_tokens,
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

fn list_local_models() -> Vec<std::path::PathBuf> {
    let home = match dirs::home_dir() {
        Some(h) => h,
        None => return vec![],
    };
    let models_dir = home.join("models");
    let mut models: Vec<std::path::PathBuf> = std::fs::read_dir(&models_dir)
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("gguf"))
        .collect();
    models.sort_by(|a, b| {
        a.file_name().cmp(&b.file_name())
    });
    models
}

const DEFAULT_LOCAL_MODEL: &str = "NVIDIA-Nemotron-Nano-9B-v2-Q4_K_M.gguf";

fn kill_stale_llama_servers() {
    // Kill any orphaned llama-server processes from previous runs
    let _ = std::process::Command::new("pkill")
        .args(["-f", "llama-server"])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
    // Brief pause to let ports be released
    std::thread::sleep(std::time::Duration::from_millis(300));
}

fn spawn_llama_server(port: u16, model_path: &std::path::Path) -> Result<std::process::Child, String> {
    let home = dirs::home_dir().ok_or("Could not determine home directory")?;
    let server_path = home.join("llama.cpp/build/bin/llama-server");

    if !server_path.exists() {
        return Err(format!("llama-server not found at {}", server_path.display()));
    }
    if !model_path.exists() {
        return Err(format!("Model not found at {}", model_path.display()));
    }

    std::process::Command::new(&server_path)
        .arg("--model")
        .arg(model_path)
        .arg("--port")
        .arg(port.to_string())
        .arg("--ctx-size")
        .arg("16384")
        .arg("--n-gpu-layers")
        .arg("99")
        .arg("--flash-attn")
        .arg("on")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to spawn llama-server: {}", e))
}

async fn wait_for_server_ready(port: u16, timeout_secs: u64, llama_process: &mut Option<std::process::Child>) -> bool {
    use std::io::Write;

    // Drain stderr in a background thread so the pipe buffer doesn't block the server.
    let stderr_lines: Arc<std::sync::Mutex<Vec<String>>> = Arc::new(std::sync::Mutex::new(Vec::new()));
    if let Some(ref mut child) = llama_process {
        if let Some(stderr) = child.stderr.take() {
            let lines = stderr_lines.clone();
            std::thread::spawn(move || {
                use std::io::BufRead;
                let reader = std::io::BufReader::new(stderr);
                for line in reader.lines() {
                    if let Ok(l) = line {
                        lines.lock().unwrap().push(l);
                    }
                }
            });
        }
    }

    let client = reqwest::Client::new();
    let url = format!("http://localhost:{}/health", port);
    let start = std::time::Instant::now();
    let deadline = start + std::time::Duration::from_secs(timeout_secs);
    let bar_width = 24usize;

    print!("{}", tui::HIDE_CURSOR);
    std::io::stdout().flush().ok();

    while std::time::Instant::now() < deadline {
        // Check if server process crashed
        if let Some(ref mut child) = llama_process {
            if let Ok(Some(_)) = child.try_wait() {
                // Clear the bar line, show error
                print!("\r{}{}{}  ", tui::SHOW_CURSOR, tui::RESET, " ".repeat(bar_width + 30));
                print!("\r  {}Server exited unexpectedly{}\n", tui::YELLOW, tui::RESET);
                // Show last few stderr lines as hint
                let lines = stderr_lines.lock().unwrap();
                if let Some(last) = lines.last() {
                    println!("  {}{}{}", tui::DIM, last, tui::RESET);
                }
                std::io::stdout().flush().ok();
                return false;
            }
        }

        // Draw progress bar
        let elapsed = start.elapsed().as_secs_f64();
        let frac = (elapsed / timeout_secs as f64).min(1.0);
        let filled = (frac * bar_width as f64) as usize;
        let empty = bar_width - filled;
        print!(
            "\r  {}Loading model [{}{}{}{}{}] {:.0}s{}",
            tui::DIM, tui::RESET, tui::CYAN,
            "\u{2588}".repeat(filled),   // █
            "\u{2591}".repeat(empty),    // ░
            tui::DIM, elapsed, tui::RESET,
        );
        std::io::stdout().flush().ok();

        if let Ok(resp) = client.get(&url).send().await {
            if resp.status().is_success() {
                // Fill bar to 100% briefly
                print!(
                    "\r  {}Loading model [{}{}{}] done{}",
                    tui::DIM, tui::RESET, tui::CYAN,
                    "\u{2588}".repeat(bar_width), tui::RESET,
                );
                std::io::stdout().flush().ok();
                std::thread::sleep(std::time::Duration::from_millis(200));
                print!("\r{}{}\r", tui::SHOW_CURSOR, " ".repeat(bar_width + 30));
                std::io::stdout().flush().ok();
                return true;
            }
        }
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }
    print!("\r{}{}\r", tui::SHOW_CURSOR, " ".repeat(bar_width + 30));
    std::io::stdout().flush().ok();
    false
}

/// Strip markdown formatting, code blocks, emojis, and special characters
/// so that TTS receives only clean natural language text.
#[cfg(feature = "voice")]
fn strip_markdown_for_tts(text: &str) -> String {
    let mut out = String::new();
    let mut in_code_block = false;

    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("```") {
            in_code_block = !in_code_block;
            continue;
        }
        if in_code_block { continue; }

        let line = trimmed.trim_start_matches('#').trim();
        if line.is_empty() { continue; }

        for c in line.chars() {
            match c {
                'a'..='z' | 'A'..='Z' | '0'..='9' => out.push(c),
                ' ' | '.' | ',' | '!' | '?' | ';' | ':' | '\'' | '"' | '-' | '(' | ')' => out.push(c),
                '*' | '_' | '`' | '~' | '[' | ']' | '|' | '#' => {} // strip markdown syntax
                _ if c.is_alphabetic() => out.push(c), // keep non-English letters
                _ => {} // strip emojis, arrows, etc.
            }
        }
        out.push(' ');
    }

    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Flush any buffered terminal input (e.g. extra Enter keypresses during recording).
#[cfg(feature = "voice")]
fn drain_stdin() {
    #[cfg(unix)]
    {
        use std::os::unix::io::AsRawFd;
        let fd = std::io::stdin().as_raw_fd();
        unsafe { libc::tcflush(fd, libc::TCIFLUSH); }
    }
}

/// Speak with TTS while watching for user interrupt (Enter or Ctrl+Space).
/// Returns true if the user interrupted (wants to speak next).
#[cfg(feature = "voice")]
fn speak_interruptible(vs: &mut voice::VoiceSession, text: &str) -> bool {
    use crossterm::event::{self, Event, KeyCode, KeyModifiers};
    use crossterm::terminal;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    vs.clear_cancel();
    let cancel = vs.cancel_flag();
    let done = Arc::new(AtomicBool::new(false));
    let done2 = done.clone();

    // Spawn thread to watch for keypress during TTS
    let watcher = std::thread::spawn(move || {
        terminal::enable_raw_mode().ok();
        let mut interrupted = false;
        while !done2.load(Ordering::Relaxed) {
            if event::poll(std::time::Duration::from_millis(100)).unwrap_or(false) {
                if let Ok(Event::Key(key)) = event::read() {
                    let is_interrupt = key.code == KeyCode::Enter
                        || (key.code == KeyCode::Char(' ')
                            && key.modifiers.contains(KeyModifiers::CONTROL));
                    if is_interrupt {
                        cancel.store(true, Ordering::Relaxed);
                        interrupted = true;
                        break;
                    }
                }
            }
        }
        terminal::disable_raw_mode().ok();
        interrupted
    });

    if let Err(e) = vs.speak(text) {
        eprintln!("TTS error: {}", e);
    }

    // Signal watcher to stop and collect result
    done.store(true, Ordering::Relaxed);
    let interrupted = watcher.join().unwrap_or(false);

    if interrupted {
        vs.stop_playback();
    }

    interrupted
}

#[cfg(feature = "voice")]
enum VoiceAction {
    Record,
    Text(String),
    Exit,
}

/// Read input in voice mode using crossterm raw terminal.
/// Ctrl+Space or Enter (empty) → Record, typed text + Enter → Text, Ctrl+C → Exit.
#[cfg(feature = "voice")]
fn voice_read_input() -> VoiceAction {
    use crossterm::event::{self, Event, KeyCode, KeyModifiers};
    use crossterm::terminal;

    if terminal::enable_raw_mode().is_err() {
        // Fallback: just use regular read_line
        let mut line = String::new();
        return match io::stdin().read_line(&mut line) {
            Ok(0) | Err(_) => VoiceAction::Exit,
            _ => {
                let trimmed = line.trim().to_string();
                if trimmed.is_empty() { VoiceAction::Record } else { VoiceAction::Text(trimmed) }
            }
        };
    }

    let mut buffer = String::new();

    let result = loop {
        match event::read() {
            Ok(Event::Key(key)) => {
                // Ctrl+Space → record
                if (key.code == KeyCode::Char(' ') && key.modifiers.contains(KeyModifiers::CONTROL))
                    || (key.code == KeyCode::Char('\0'))
                {
                    print!("\r\n");
                    io::stdout().flush().ok();
                    break VoiceAction::Record;
                }
                // Ctrl+C → exit
                if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
                    print!("\r\n");
                    io::stdout().flush().ok();
                    break VoiceAction::Exit;
                }
                // Enter
                if key.code == KeyCode::Enter {
                    print!("\r\n");
                    io::stdout().flush().ok();
                    if buffer.is_empty() {
                        break VoiceAction::Record;
                    }
                    break VoiceAction::Text(buffer);
                }
                // Backspace
                if key.code == KeyCode::Backspace {
                    if buffer.pop().is_some() {
                        print!("\x08 \x08");
                        io::stdout().flush().ok();
                    }
                    continue;
                }
                // Regular character (no ctrl/alt modifier)
                if let KeyCode::Char(c) = key.code {
                    if !key.modifiers.contains(KeyModifiers::CONTROL)
                        && !key.modifiers.contains(KeyModifiers::ALT)
                    {
                        buffer.push(c);
                        print!("{}", c);
                        io::stdout().flush().ok();
                    }
                }
            }
            Ok(_) => {} // ignore mouse, resize, etc.
            Err(_) => break VoiceAction::Exit,
        }
    };

    terminal::disable_raw_mode().ok();
    result
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
            let fake_model = home.join("models/nonexistent.gguf");
            let result = spawn_llama_server(19876, &fake_model);
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
        let model_path = home.join("models/nonexistent-test-model.gguf");

        if server_path.exists() {
            let result = spawn_llama_server(19877, &model_path);
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
        let mut proc = None;
        let result = wait_for_server_ready(19999, 1, &mut proc).await;
        assert!(!result, "Should return false when no server running");
    }

    #[tokio::test]
    async fn test_wait_zero_timeout_returns_false() {
        let mut proc = None;
        let result = wait_for_server_ready(19998, 0, &mut proc).await;
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

        let mut proc = None;
        let result = wait_for_server_ready(port, 5, &mut proc).await;
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

        let mut proc = None;
        let result = wait_for_server_ready(port, 10, &mut proc).await;
        assert!(result, "Should succeed after retries");
        assert!(
            request_count.load(Ordering::SeqCst) >= 3,
            "Should have retried at least 3 times"
        );
    }
}
