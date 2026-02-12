//! nanobot - A lightweight personal AI assistant framework in Rust.
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
mod syntax;
mod utils;
#[cfg(feature = "voice")]
mod voice;
#[cfg(feature = "voice")]
mod voice_pipeline;

use std::io::{self, BufRead, Write as _};
use std::net::TcpListener;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use clap::{Parser, Subcommand};
use rustyline::error::ReadlineError;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use crate::agent::agent_loop::{build_shared_core, AgentLoop, SharedCoreHandle};
use crate::agent::tuning::{
    score_feasible_profiles, select_optimal_from_input, OptimizationInput,
};
use crate::bus::events::{InboundMessage, OutboundMessage};
use crate::channels::manager::ChannelManager;
use crate::config::loader::{get_config_path, get_data_dir, load_config, save_config};
use crate::config::schema::Config;
use crate::cron::service::CronService;
use crate::cron::types::CronSchedule;
use crate::providers::base::LLMProvider;
use crate::providers::openai_compat::OpenAICompatProvider;
use crate::utils::helpers::get_workspace_path;

const VERSION: &str = "0.1.0";
const LOGO: &str = "*";

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

/// Render markdown response to the terminal.
///
/// Uses termimad for structural markdown (headers, bold, lists, etc.).
fn render_markdown(text: &str, skin: &termimad::MadSkin) {
    skin.print_text(text);
}

/// ANSI escape helpers for colored terminal output.
mod tui {
    pub const RESET: &str = "\x1b[0m";
    pub const BOLD: &str = "\x1b[1m";
    pub const DIM: &str = "\x1b[2m";
    pub const CYAN: &str = "\x1b[36m";
    pub const GREEN: &str = "\x1b[32m";
    pub const YELLOW: &str = "\x1b[33m";
    pub const RED: &str = "\x1b[31m";
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
#[command(name = "nanobot", about = "nanobot - Personal AI Assistant", version = VERSION)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Initialize nanobot configuration and workspace.
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
        /// Language hint for voice TTS engine (e.g. "en" uses faster Supertonic).
        #[arg(long)]
        lang: Option<String>,
    },
    /// Start the nanobot gateway (channels + agent loop).
    Gateway {
        /// Gateway port.
        #[arg(short, long, default_value_t = 18790)]
        port: u16,
        /// Verbose logging.
        #[arg(short, long)]
        verbose: bool,
    },
    /// Show nanobot status.
    Status,
    /// Select the best local profile from benchmark results.
    Tune {
        /// Path to benchmark JSON input file.
        #[arg(short, long)]
        input: String,
        /// Print the selected profile as JSON.
        #[arg(long)]
        json: bool,
    },
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
    /// Quick-start WhatsApp channel (zero config).
    #[command(name = "whatsapp", alias = "wa")]
    WhatsApp,
    /// Quick-start Telegram channel.
    Telegram {
        /// Bot token (prompted interactively if not provided).
        #[arg(short, long)]
        token: Option<String>,
    },
    /// Quick-start Email channel.
    Email {
        /// IMAP host (prompted interactively if not provided).
        #[arg(long)]
        imap_host: Option<String>,
        /// SMTP host (prompted interactively if not provided).
        #[arg(long)]
        smtp_host: Option<String>,
        /// Email account username/address.
        #[arg(short, long)]
        username: Option<String>,
        /// Email account password or app password.
        #[arg(short, long)]
        password: Option<String>,
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
    // Safety net: kill orphaned llama-server on panic so it doesn't hold VRAM
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = std::process::Command::new("pkill")
            .args(["-f", "llama-server"])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();
        default_hook(info);
    }));

    let cli = Cli::parse();

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                tracing_subscriber::EnvFilter::new("warn,ort=off,supertonic=off")
            }),
        )
        .init();

    match cli.command {
        Commands::Onboard => cmd_onboard(),
        Commands::Agent {
            message,
            session,
            local,
            lang,
        } => cmd_agent(message, session, local, lang),
        Commands::Gateway { port, verbose } => cmd_gateway(port, verbose),
        Commands::Status => cmd_status(),
        Commands::Tune { input, json } => cmd_tune(input, json),
        Commands::Channels { action } => match action {
            ChannelsAction::Status => cmd_channels_status(),
        },
        Commands::Cron { action } => match action {
            CronAction::List { all } => cmd_cron_list(all),
            CronAction::Add {
                name,
                message,
                every,
                cron,
                deliver,
                to,
                channel,
            } => cmd_cron_add(name, message, every, cron, deliver, to, channel),
            CronAction::Remove { job_id } => cmd_cron_remove(job_id),
            CronAction::Enable { job_id, disable } => cmd_cron_enable(job_id, disable),
        },
        Commands::WhatsApp => cmd_whatsapp(),
        Commands::Telegram { token } => cmd_telegram(token),
        Commands::Email {
            imap_host,
            smtp_host,
            username,
            password,
        } => cmd_email(imap_host, smtp_host, username, password),
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

    println!("\n{} nanobot is ready!", LOGO);
    println!("\nNext steps:");
    println!("  1. Add your API key to ~/.nanobot/config.json");
    println!("     Get one at: https://openrouter.ai/keys");
    println!("  2. Chat: nanobot agent -m \"Hello!\"");
}

fn create_workspace_templates(workspace: &std::path::Path) {
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
// Agent
// ============================================================================

fn cmd_agent(message: Option<String>, session_id: String, local_flag: bool, lang: Option<String>) {
    let config = load_config(None);

    // Check environment variable for local mode
    let local_env = std::env::var("NANOBOT_LOCAL")
        .map(|v| v == "1" || v.to_lowercase() == "true")
        .unwrap_or(false);

    // Set initial local mode from flag or environment
    if local_flag || local_env {
        LOCAL_MODE.store(true, Ordering::SeqCst);
    }

    let mut local_port = std::env::var("NANOBOT_LOCAL_PORT").unwrap_or_else(|_| "8080".to_string());

    // Check if we can proceed
    let is_local = LOCAL_MODE.load(Ordering::SeqCst);
    if !is_local {
        let api_key = config.get_api_key();
        let model = &config.agents.defaults.model;
        if api_key.is_none() && !model.starts_with("bedrock/") {
            eprintln!("Error: No API key configured.");
            eprintln!("Set one in ~/.nanobot/config.json under providers.openrouter.apiKey");
            eprintln!("Or use --local flag to use a local LLM server.");
            std::process::exit(1);
        }
    }

    let runtime = tokio::runtime::Runtime::new().expect("Failed to create tokio runtime");

    runtime.block_on(async {
        // Create shared core and initial agent loop.
        let core_handle = build_core_handle(&config, &local_port, Some(DEFAULT_LOCAL_MODEL), None);
        let cron_store_path = get_data_dir().join("cron").join("jobs.json");
        let cron_service = Arc::new(CronService::new(cron_store_path));

        // Provide email config to the REPL agent when credentials are configured.
        let email_config = {
            let ec = &config.channels.email;
            if !ec.imap_host.is_empty() && !ec.username.is_empty() && !ec.password.is_empty() {
                Some(ec.clone())
            } else {
                None
            }
        };

        let mut agent_loop = create_agent_loop(
            core_handle.clone(), &config, Some(cron_service.clone()), email_config.clone(), None,
        );

        if let Some(msg) = message {
            let (delta_tx, mut delta_rx) =
                tokio::sync::mpsc::unbounded_channel::<String>();
            let print_task = tokio::spawn(async move {
                use std::io::Write as _;
                while let Some(delta) = delta_rx.recv().await {
                    print!("{}", delta);
                    std::io::stdout().flush().ok();
                }
                println!();
            });
            println!();
            let _response = agent_loop
                .process_direct_streaming(&msg, &session_id, "cli", "direct", None, delta_tx)
                .await;
            let _ = print_task.await;
        } else {
            print_startup_splash(&local_port);

            let mut llama_process: Option<std::process::Child> = None;
            let mut compaction_process: Option<std::process::Child> = None;
            let mut compaction_port: Option<String> = None;
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

            // Background channel state
            struct ActiveChannel {
                name: String,
                stop: Arc<AtomicBool>,
                handle: tokio::task::JoinHandle<()>,
            }
            let mut active_channels: Vec<ActiveChannel> = vec![];
            // Channel for background gateways to send display lines to the REPL.
            let (display_tx, mut display_rx) = mpsc::unbounded_channel::<String>();

            loop {
                // Drain any pending display messages from background channels.
                while let Ok(line) = display_rx.try_recv() {
                    println!("\r{}", line);
                }
                let is_local = LOCAL_MODE.load(Ordering::SeqCst);
                #[cfg(feature = "voice")]
                let voice_on = voice_session.is_some();
                #[cfg(not(feature = "voice"))]
                let voice_on = false;

                let prompt = if voice_on {
                    format!("{}{}~>{} ", tui::BOLD, tui::MAGENTA, tui::RESET)
                } else if is_local {
                    format!("{}{}L>{} ", tui::BOLD, tui::YELLOW, tui::RESET)
                } else {
                    format!("{}{}>{} ", tui::BOLD, tui::GREEN, tui::RESET)
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

                // === VOICE RECORDING (streaming pipeline) ===
                #[cfg(feature = "voice")]
                if do_record {
                    if let Some(ref mut vs) = voice_session {
                        vs.stop_playback();
                        let mut keep_recording = true;
                        while keep_recording {
                            keep_recording = false;
                            match vs.record_and_transcribe() {
                                Ok(Some((text, lang))) => {
                                    // Start streaming TTS pipeline
                                    vs.clear_cancel();
                                    let cancel = vs.cancel_flag();

                                    // Display channel: synthesis thread → terminal
                                    // Text appears when TTS finishes each sentence (synced with audio)
                                    let (display_tx, mut display_rx) =
                                        tokio::sync::mpsc::unbounded_channel::<String>();

                                    match vs.start_streaming_speak(&lang, Some(display_tx)) {
                                        Ok((sentence_tx, tts_handle)) => {
                                            // Delta channel: LLM → accumulator (silent, feeds TTS only)
                                            let (delta_tx, mut delta_rx) =
                                                tokio::sync::mpsc::unbounded_channel::<String>();

                                            let acc_sentence_tx = sentence_tx.clone();
                                            let accumulator_task = tokio::spawn(async move {
                                                let mut acc = voice::SentenceAccumulator::new(acc_sentence_tx);
                                                while let Some(delta) = delta_rx.recv().await {
                                                    acc.push(&delta);
                                                }
                                                acc.flush();
                                            });

                                            // Display task: print sentences as TTS synthesizes them
                                            let display_task = tokio::spawn(async move {
                                                use std::io::Write as _;
                                                let mut first = true;
                                                while let Some(sentence) = display_rx.recv().await {
                                                    if first {
                                                        first = false;
                                                    } else {
                                                        print!(" ");
                                                    }
                                                    print!("{}", sentence);
                                                    std::io::stdout().flush().ok();
                                                }
                                                println!();
                                            });

                                            // Stream LLM response (deltas go to accumulator silently)
                                            let _response = agent_loop
                                                .process_direct_streaming(
                                                    &text,
                                                    &session_id,
                                                    "voice",
                                                    "direct",
                                                    Some(&lang),
                                                    delta_tx,
                                                )
                                                .await;

                                            // Wait for accumulator to flush remaining sentences
                                            let _ = accumulator_task.await;

                                            // Interrupt watcher: runs while TTS plays remaining audio
                                            let done = Arc::new(AtomicBool::new(false));
                                            let done2 = done.clone();
                                            let cancel2 = cancel.clone();
                                            let watcher = std::thread::spawn(move || {
                                                use crossterm::event::{self, Event, KeyCode, KeyModifiers};
                                                use crossterm::terminal;
                                                terminal::enable_raw_mode().ok();
                                                let mut interrupted = false;
                                                while !done2.load(Ordering::Relaxed) {
                                                    if event::poll(std::time::Duration::from_millis(100)).unwrap_or(false) {
                                                        if let Ok(Event::Key(key)) = event::read() {
                                                            let is_interrupt = key.code == KeyCode::Enter
                                                                || (key.code == KeyCode::Char(' ')
                                                                    && key.modifiers.contains(KeyModifiers::CONTROL));
                                                            if is_interrupt {
                                                                cancel2.store(true, Ordering::Relaxed);
                                                                interrupted = true;
                                                                break;
                                                            }
                                                        }
                                                    }
                                                }
                                                terminal::disable_raw_mode().ok();
                                                interrupted
                                            });

                                            // Wait for TTS + playback + display to finish
                                            let _ = tts_handle.join();
                                            done.store(true, Ordering::Relaxed);
                                            let interrupted = watcher.join().unwrap_or(false);
                                            let _ = display_task.await;

                                            {
                                                let sa_count = agent_loop.subagent_manager().get_running_count().await;
                                                active_channels.retain(|ch| !ch.handle.is_finished());
                                                let ch_names: Vec<&str> = active_channels.iter().map(|c| match c.name.as_str() {
                                                    "whatsapp" => "wa", "telegram" => "tg", other => other,
                                                }).collect();
                                                print_status_bar(&core_handle, &ch_names, sa_count);
                                            }

                                            if interrupted {
                                                keep_recording = true;
                                            }
                                        }
                                        Err(e) => {
                                            eprintln!("Streaming TTS failed ({}), falling back", e);
                                            let response = agent_loop
                                                .process_direct_with_lang(&text, &session_id, "voice", "direct", Some(&lang))
                                                .await;
                                            println!();
                                            render_markdown(&response, &skin);
                                            println!();
                                            let tts_text = strip_markdown_for_tts(&response);
                                            if !tts_text.is_empty() {
                                                if speak_interruptible(vs, &tts_text, "en") {
                                                    keep_recording = true;
                                                }
                                            }
                                        }
                                    }
                                }
                                Ok(None) => println!("\x1b[2m(no speech detected)\x1b[0m"),
                                Err(e) => eprintln!("\x1b[31m{}\x1b[0m", e),
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
                                    // Reuse only if this server is configured with
                                    // one slot (`n_parallel=1`), otherwise per-request
                                    // context can be much smaller than advertised.
                                    let props_url = format!("http://localhost:{}/props", port);
                                    let n_parallel = reqwest::blocking::get(&props_url)
                                        .ok()
                                        .and_then(|r| r.json::<serde_json::Value>().ok())
                                        .and_then(|json| {
                                            json.get("default_generation_settings")
                                                .and_then(|s| s.get("n_parallel"))
                                                .and_then(|n| n.as_u64())
                                                .or_else(|| {
                                                    json.get("n_parallel").and_then(|n| n.as_u64())
                                                })
                                        })
                                        .unwrap_or(1);
                                    if n_parallel <= 1 {
                                        found_port = Some(port);
                                        break;
                                    }
                                }
                            }
                        }

                        if let Some(port) = found_port {
                            // Reuse existing server
                            println!("\n  {}{}Reusing{} llama.cpp server on port {}", tui::BOLD, tui::YELLOW, tui::RESET, port);
                            local_port = port.to_string();
                            LOCAL_MODE.store(true, Ordering::SeqCst);
                            let main_ctx = compute_optimal_context_size(&current_model_path);
                            start_compaction_if_available(&mut compaction_process, &mut compaction_port, main_ctx).await;
                            rebuild_core(&core_handle, &config, &local_port, current_model_path.file_name().and_then(|n| n.to_str()), compaction_port.as_deref());
                            agent_loop = create_agent_loop(core_handle.clone(), &config, Some(cron_service.clone()), email_config.clone(), None);
                            print_mode_banner(&local_port);
                        } else {
                            // Kill any orphaned servers from previous runs
                            kill_stale_llama_servers();
                            let port = find_available_port(8080);
                            let ctx_size = compute_optimal_context_size(&current_model_path);
                            println!("\n  {}{}Starting{} llama.cpp server on port {} (ctx: {}K)...", tui::BOLD, tui::YELLOW, tui::RESET, port, ctx_size / 1024);

                            match spawn_llama_server(port, &current_model_path, ctx_size) {
                                Ok(child) => {
                                    llama_process = Some(child);
                                    if wait_for_server_ready(port, 120, &mut llama_process).await {
                                        local_port = port.to_string();
                                        LOCAL_MODE.store(true, Ordering::SeqCst);
                                        start_compaction_if_available(&mut compaction_process, &mut compaction_port, ctx_size).await;
                                        rebuild_core(&core_handle, &config, &local_port, current_model_path.file_name().and_then(|n| n.to_str()), compaction_port.as_deref());
                                        agent_loop = create_agent_loop(core_handle.clone(), &config, Some(cron_service.clone()), email_config.clone(), None);
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
                        } else {
                            // If we were reusing an external server, make sure we
                            // don't keep carrying stale settings across toggles.
                            kill_stale_llama_servers();
                        }
                        llama_process = None;
                        stop_compaction_server(&mut compaction_process, &mut compaction_port);
                        LOCAL_MODE.store(false, Ordering::SeqCst);
                        rebuild_core(&core_handle, &config, &local_port, current_model_path.file_name().and_then(|n| n.to_str()), compaction_port.as_deref());
                        agent_loop = create_agent_loop(core_handle.clone(), &config, Some(cron_service.clone()), email_config.clone(), None);
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
                    let previous_model_path = current_model_path.clone();
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
                        let ctx_size = compute_optimal_context_size(&current_model_path);
                        println!("  {}{}Starting{} llama.cpp server on port {} (ctx: {}K)...", tui::BOLD, tui::YELLOW, tui::RESET, port, ctx_size / 1024);

                        match spawn_llama_server(port, &current_model_path, ctx_size) {
                            Ok(child) => {
                                llama_process = Some(child);
                                if wait_for_server_ready(port, 120, &mut llama_process).await {
                                    local_port = port.to_string();
                                    rebuild_core(&core_handle, &config, &local_port, current_model_path.file_name().and_then(|n| n.to_str()), compaction_port.as_deref());
                                    agent_loop = create_agent_loop(core_handle.clone(), &config, Some(cron_service.clone()), email_config.clone(), None);
                                    print_mode_banner(&local_port);
                                } else {
                                    println!("  {}{}Server failed to start with new model{}", tui::BOLD, tui::YELLOW, tui::RESET);
                                    if let Some(ref mut child) = llama_process {
                                        child.kill().ok();
                                        child.wait().ok();
                                    }
                                    llama_process = None;

                                    // Restart the previous working model
                                    current_model_path = previous_model_path.clone();
                                    let prev_name = current_model_path.file_name().unwrap().to_string_lossy();
                                    println!("  {}Restarting previous model: {}{}", tui::DIM, prev_name, tui::RESET);
                                    let port = find_available_port(8080);
                                    let ctx_size = compute_optimal_context_size(&current_model_path);
                                    match spawn_llama_server(port, &current_model_path, ctx_size) {
                                        Ok(child) => {
                                            llama_process = Some(child);
                                            if wait_for_server_ready(port, 120, &mut llama_process).await {
                                                local_port = port.to_string();
                                                rebuild_core(&core_handle, &config, &local_port, current_model_path.file_name().and_then(|n| n.to_str()), compaction_port.as_deref());
                                                agent_loop = create_agent_loop(core_handle.clone(), &config, Some(cron_service.clone()), email_config.clone(), None);
                                                println!("  {}Restored: {}{}\n", tui::DIM, prev_name, tui::RESET);
                                            } else {
                                                println!("  {}{}Previous model also failed — switching to cloud{}", tui::BOLD, tui::YELLOW, tui::RESET);
                                                LOCAL_MODE.store(false, Ordering::SeqCst);
                                                rebuild_core(&core_handle, &config, &local_port, current_model_path.file_name().and_then(|n| n.to_str()), compaction_port.as_deref());
                                                agent_loop = create_agent_loop(core_handle.clone(), &config, Some(cron_service.clone()), email_config.clone(), None);
                                            }
                                        }
                                        Err(_) => {
                                            println!("  {}{}Previous model also failed — switching to cloud{}", tui::BOLD, tui::YELLOW, tui::RESET);
                                            LOCAL_MODE.store(false, Ordering::SeqCst);
                                            rebuild_core(&core_handle, &config, &local_port, current_model_path.file_name().and_then(|n| n.to_str()), compaction_port.as_deref());
                                            agent_loop = create_agent_loop(core_handle.clone(), &config, Some(cron_service.clone()), email_config.clone(), None);
                                        }
                                    }
                                }
                            }
                            Err(e) => {
                                println!("\n  {}{}Failed to start server:{} {}", tui::BOLD, tui::YELLOW, tui::RESET, e);

                                // Restart the previous working model
                                current_model_path = previous_model_path.clone();
                                let prev_name = current_model_path.file_name().unwrap().to_string_lossy();
                                println!("  {}Restarting previous model: {}{}", tui::DIM, prev_name, tui::RESET);
                                let port = find_available_port(8080);
                                let ctx_size = compute_optimal_context_size(&current_model_path);
                                match spawn_llama_server(port, &current_model_path, ctx_size) {
                                    Ok(child) => {
                                        llama_process = Some(child);
                                        if wait_for_server_ready(port, 120, &mut llama_process).await {
                                            local_port = port.to_string();
                                            rebuild_core(&core_handle, &config, &local_port, current_model_path.file_name().and_then(|n| n.to_str()), compaction_port.as_deref());
                                            agent_loop = create_agent_loop(core_handle.clone(), &config, Some(cron_service.clone()), email_config.clone(), None);
                                            println!("  {}Restored: {}{}\n", tui::DIM, prev_name, tui::RESET);
                                        } else {
                                            println!("  {}{}Previous model also failed — switching to cloud{}", tui::BOLD, tui::YELLOW, tui::RESET);
                                            LOCAL_MODE.store(false, Ordering::SeqCst);
                                            rebuild_core(&core_handle, &config, &local_port, current_model_path.file_name().and_then(|n| n.to_str()), compaction_port.as_deref());
                                            agent_loop = create_agent_loop(core_handle.clone(), &config, Some(cron_service.clone()), email_config.clone(), None);
                                        }
                                    }
                                    Err(_) => {
                                        println!("  {}{}Previous model also failed — switching to cloud{}", tui::BOLD, tui::YELLOW, tui::RESET);
                                        LOCAL_MODE.store(false, Ordering::SeqCst);
                                        rebuild_core(&core_handle, &config, &local_port, current_model_path.file_name().and_then(|n| n.to_str()), compaction_port.as_deref());
                                        agent_loop = create_agent_loop(core_handle.clone(), &config, Some(cron_service.clone()), email_config.clone(), None);
                                    }
                                }
                            }
                        }
                    } else {
                        println!("Model will be used next time you toggle /local on.\n");
                    }

                    continue;
                }

                // Handle context size change
                if input == "/ctx" || input.starts_with("/ctx ") {
                    if !LOCAL_MODE.load(Ordering::SeqCst) {
                        println!("\n  {}Not in local mode — use /local first{}\n", tui::DIM, tui::RESET);
                        continue;
                    }

                    let arg = input.strip_prefix("/ctx").unwrap().trim();
                    let new_ctx: usize = if arg.is_empty() {
                        // No argument → re-auto-detect
                        let auto = compute_optimal_context_size(&current_model_path);
                        println!("\n  Auto-detected: {}K", auto / 1024);
                        auto
                    } else {
                        // Parse: accept "32768", "32K", "32k"
                        let s = arg.to_lowercase();
                        let parsed = if let Some(prefix) = s.strip_suffix('k') {
                            prefix.parse::<usize>().map(|n| n * 1024)
                        } else {
                            s.parse::<usize>()
                        };
                        match parsed {
                            Ok(n) if n >= 2048 => n,
                            Ok(_) => {
                                println!("\n  Minimum context size is 2048 (2K)\n");
                                continue;
                            }
                            Err(_) => {
                                println!("\n  Usage: /ctx [size]  e.g. /ctx 32K or /ctx 32768\n");
                                continue;
                            }
                        }
                    };

                    // Round down to nearest 1024
                    let new_ctx = (new_ctx / 1024) * 1024;

                    // Restart server with new context size
                    if let Some(ref mut child) = llama_process {
                        println!("  {}Stopping current server...{}", tui::DIM, tui::RESET);
                        child.kill().ok();
                        child.wait().ok();
                    }
                    llama_process = None;
                    kill_stale_llama_servers();

                    let port = find_available_port(8080);
                    println!("  {}{}Restarting{} llama.cpp on port {} (ctx: {}K)...", tui::BOLD, tui::YELLOW, tui::RESET, port, new_ctx / 1024);

                    match spawn_llama_server(port, &current_model_path, new_ctx) {
                        Ok(child) => {
                            llama_process = Some(child);
                            if wait_for_server_ready(port, 120, &mut llama_process).await {
                                local_port = port.to_string();
                                rebuild_core(&core_handle, &config, &local_port, current_model_path.file_name().and_then(|n| n.to_str()), compaction_port.as_deref());
                                agent_loop = create_agent_loop(core_handle.clone(), &config, Some(cron_service.clone()), email_config.clone(), None);
                                print_mode_banner(&local_port);
                            } else {
                                println!("  {}{}Server failed to start with new context size{}", tui::BOLD, tui::YELLOW, tui::RESET);
                                if let Some(ref mut child) = llama_process {
                                    child.kill().ok();
                                    child.wait().ok();
                                }
                                llama_process = None;

                                // Restart with auto-detected size as fallback
                                let port = find_available_port(8080);
                                let fallback_ctx = compute_optimal_context_size(&current_model_path);
                                println!("  {}Falling back to auto-detected context ({}K)...{}", tui::DIM, fallback_ctx / 1024, tui::RESET);
                                match spawn_llama_server(port, &current_model_path, fallback_ctx) {
                                    Ok(child) => {
                                        llama_process = Some(child);
                                        if wait_for_server_ready(port, 120, &mut llama_process).await {
                                            local_port = port.to_string();
                                            rebuild_core(&core_handle, &config, &local_port, current_model_path.file_name().and_then(|n| n.to_str()), compaction_port.as_deref());
                                            agent_loop = create_agent_loop(core_handle.clone(), &config, Some(cron_service.clone()), email_config.clone(), None);
                                            print_mode_banner(&local_port);
                                        } else {
                                            println!("  {}{}Fallback also failed — switching to cloud{}", tui::BOLD, tui::YELLOW, tui::RESET);
                                            LOCAL_MODE.store(false, Ordering::SeqCst);
                                            rebuild_core(&core_handle, &config, &local_port, current_model_path.file_name().and_then(|n| n.to_str()), compaction_port.as_deref());
                                            agent_loop = create_agent_loop(core_handle.clone(), &config, Some(cron_service.clone()), email_config.clone(), None);
                                        }
                                    }
                                    Err(_) => {
                                        println!("  {}{}Fallback also failed — switching to cloud{}", tui::BOLD, tui::YELLOW, tui::RESET);
                                        LOCAL_MODE.store(false, Ordering::SeqCst);
                                        rebuild_core(&core_handle, &config, &local_port, current_model_path.file_name().and_then(|n| n.to_str()), compaction_port.as_deref());
                                        agent_loop = create_agent_loop(core_handle.clone(), &config, Some(cron_service.clone()), email_config.clone(), None);
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            println!("\n  {}{}Failed to start server:{} {}", tui::BOLD, tui::YELLOW, tui::RESET, e);
                            println!("  {}Remaining in cloud mode{}\n", tui::DIM, tui::RESET);
                            LOCAL_MODE.store(false, Ordering::SeqCst);
                            rebuild_core(&core_handle, &config, &local_port, current_model_path.file_name().and_then(|n| n.to_str()), compaction_port.as_deref());
                            agent_loop = create_agent_loop(core_handle.clone(), &config, Some(cron_service.clone()), email_config.clone(), None);
                        }
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
                        match voice::VoiceSession::with_lang(lang.as_deref()).await {
                            Ok(vs) => {
                                voice_session = Some(vs);
                                println!("\nVoice mode ON. Ctrl+Space or Enter to speak, type for text.\n");
                            }
                            Err(e) => eprintln!("\nFailed to start voice mode: {}\n", e),
                        }
                    }
                    continue;
                }

                // Handle WhatsApp quick-start from REPL
                if input == "/whatsapp" || input == "/wa" {
                    active_channels.retain(|ch| !ch.handle.is_finished());
                    if active_channels.iter().any(|ch| ch.name == "whatsapp") {
                        println!("\n  WhatsApp is already running. Use /stop to stop channels.\n");
                        continue;
                    }
                    let mut gw_config = load_config(None);
                    check_api_key(&gw_config);
                    gw_config.channels.whatsapp.enabled = true;
                    gw_config.channels.telegram.enabled = false;
                    gw_config.channels.feishu.enabled = false;
                    gw_config.channels.email.enabled = false;
                    let stop = Arc::new(AtomicBool::new(false));
                    let stop2 = stop.clone();
                    let dtx = display_tx.clone();
                    let ch = core_handle.clone();
                    println!("\n  Scan the QR code when it appears");
                    let handle = tokio::spawn(async move {
                        run_gateway_async(gw_config, ch, Some(stop2), Some(dtx)).await;
                    });
                    active_channels.push(ActiveChannel {
                        name: "whatsapp".to_string(), stop, handle,
                    });
                    println!("  WhatsApp running in background. Continue chatting.\n");
                    continue;
                }

                // Handle Telegram quick-start from REPL
                if input == "/telegram" || input == "/tg" {
                    active_channels.retain(|ch| !ch.handle.is_finished());
                    if active_channels.iter().any(|ch| ch.name == "telegram") {
                        println!("\n  Telegram is already running. Use /stop to stop channels.\n");
                        continue;
                    }
                    println!();
                    let mut gw_config = load_config(None);
                    check_api_key(&gw_config);
                    let saved_token = &gw_config.channels.telegram.token;
                    let token = if !saved_token.is_empty() {
                        println!("  Using saved bot token");
                        saved_token.clone()
                    } else {
                        println!("  No Telegram bot token found.");
                        println!("  Get one from @BotFather on Telegram.\n");
                        let tok_prompt = "  Enter bot token: ";
                        let t = match rl.readline(tok_prompt) {
                            Ok(line) => line.trim().to_string(),
                            Err(_) => { continue; }
                        };
                        if t.is_empty() {
                            println!("  Cancelled.\n");
                            continue;
                        }
                        print!("  Validating token... ");
                        io::stdout().flush().ok();
                        if validate_telegram_token(&t) {
                            println!("valid!\n");
                        } else {
                            println!("invalid!");
                            println!("  Check the token and try again.\n");
                            continue;
                        }
                        let save_prompt = "  Save token to config for next time? [Y/n] ";
                        if let Ok(answer) = rl.readline(save_prompt) {
                            if !answer.trim().eq_ignore_ascii_case("n") {
                                let mut save_cfg = load_config(None);
                                save_cfg.channels.telegram.token = t.clone();
                                save_config(&save_cfg, None);
                                println!("  Token saved to ~/.nanobot/config.json\n");
                            }
                        }
                        t
                    };
                    gw_config.channels.telegram.token = token;
                    gw_config.channels.telegram.enabled = true;
                    gw_config.channels.whatsapp.enabled = false;
                    gw_config.channels.feishu.enabled = false;
                    gw_config.channels.email.enabled = false;
                    let stop = Arc::new(AtomicBool::new(false));
                    let stop2 = stop.clone();
                    let dtx = display_tx.clone();
                    let ch = core_handle.clone();
                    let handle = tokio::spawn(async move {
                        run_gateway_async(gw_config, ch, Some(stop2), Some(dtx)).await;
                    });
                    active_channels.push(ActiveChannel {
                        name: "telegram".to_string(), stop, handle,
                    });
                    println!("  Telegram running in background. Continue chatting.\n");
                    continue;
                }

                // Handle Email quick-start from REPL
                if input == "/email" {
                    active_channels.retain(|ch| !ch.handle.is_finished());
                    if active_channels.iter().any(|ch| ch.name == "email") {
                        println!("\n  Email is already running. Use /stop to stop channels.\n");
                        continue;
                    }
                    println!();
                    let mut gw_config = load_config(None);
                    check_api_key(&gw_config);
                    let email_cfg = &gw_config.channels.email;
                    if email_cfg.imap_host.is_empty() || email_cfg.username.is_empty() || email_cfg.password.is_empty() {
                        println!("  Email not configured. Run `nanobot email` first or add settings to config.json.\n");
                        continue;
                    }
                    println!("  Starting Email channel...");
                    println!("  Polling {}", email_cfg.imap_host);
                    gw_config.channels.email.enabled = true;
                    gw_config.channels.whatsapp.enabled = false;
                    gw_config.channels.telegram.enabled = false;
                    gw_config.channels.feishu.enabled = false;
                    let stop = Arc::new(AtomicBool::new(false));
                    let stop2 = stop.clone();
                    let dtx = display_tx.clone();
                    let ch = core_handle.clone();
                    let handle = tokio::spawn(async move {
                        run_gateway_async(gw_config, ch, Some(stop2), Some(dtx)).await;
                    });
                    active_channels.push(ActiveChannel {
                        name: "email".to_string(), stop, handle,
                    });
                    println!("  Email running in background. Continue chatting.\n");
                    continue;
                }

                // Handle stop command — stop all background channels
                if input == "/stop" {
                    active_channels.retain(|ch| !ch.handle.is_finished());
                    if active_channels.is_empty() {
                        println!("\n  No channels running.\n");
                    } else {
                        let names: Vec<String> = active_channels.iter().map(|c| c.name.clone()).collect();
                        println!("\n  Stopping: {}", names.join(", "));
                        for ch in &active_channels {
                            ch.stop.store(true, Ordering::Relaxed);
                        }
                        tokio::time::sleep(Duration::from_secs(1)).await;
                        for ch in &active_channels {
                            ch.handle.abort();
                        }
                        active_channels.clear();
                        println!("  All channels stopped.\n");
                    }
                    continue;
                }

                // Handle help command
                if input == "/paste" || input == "/p" {
                    println!("  {}Paste mode: type or paste text, then enter --- on its own line to send{}", tui::DIM, tui::RESET);
                    let mut lines: Vec<String> = Vec::new();
                    let stdin = io::stdin();
                    for line in stdin.lock().lines() {
                        match line {
                            Ok(l) if l.trim() == "---" => break,
                            Ok(l) => lines.push(l),
                            Err(_) => break,
                        }
                    }
                    let pasted = lines.join("\n").trim().to_string();
                    if pasted.is_empty() {
                        continue;
                    }
                    let _ = rl.add_history_entry(&pasted);
                    let channel = if voice_on { "voice" } else { "cli" };
                    let (delta_tx, mut delta_rx) =
                        tokio::sync::mpsc::unbounded_channel::<String>();
                    let print_task = tokio::spawn(async move {
                        use std::io::Write as _;
                        while let Some(delta) = delta_rx.recv().await {
                            print!("{}", delta);
                            std::io::stdout().flush().ok();
                        }
                        println!();
                    });
                    println!();
                    let response = agent_loop
                        .process_direct_streaming(
                            &pasted, &session_id, channel, "direct", None, delta_tx,
                        )
                        .await;
                    let _ = print_task.await;
                    if !response.is_empty() {
                        use std::io::Write as _;
                        let lines = response.chars().filter(|&c| c == '\n').count() + 2;
                        print!("\x1b[{}A\x1b[J", lines);
                        std::io::stdout().flush().ok();
                        let skin = make_skin();
                        render_markdown(&response, &skin);
                    }
                    println!();
                    {
                        let sa_count = agent_loop.subagent_manager().get_running_count().await;
                        active_channels.retain(|ch| !ch.handle.is_finished());
                        let ch_names: Vec<&str> = active_channels.iter().map(|c| match c.name.as_str() {
                            "whatsapp" => "wa", "telegram" => "tg", other => other,
                        }).collect();
                        print_status_bar(&core_handle, &ch_names, sa_count);
                    }
                    continue;
                }

                if input == "/help" || input == "/h" || input == "/?" {
                    println!("\nCommands:");
                    println!("  /local, /l      - Toggle between local and cloud mode");
                    println!("  /model, /m      - Select local model from ~/models/");
                    println!("  /ctx [size]     - Set context size (e.g. /ctx 32K) or auto-detect");
                    println!("  /voice, /v      - Toggle voice mode (Ctrl+Space or Enter to speak)");
                    println!("  /whatsapp, /wa  - Start WhatsApp channel (runs alongside chat)");
                    println!("  /telegram, /tg  - Start Telegram channel (runs alongside chat)");
                    println!("  /email          - Start Email channel (runs alongside chat)");
                    println!("  /paste, /p      - Paste mode: multiline input until --- on its own line");
                    println!("  /stop           - Stop all running channels");
                    println!("  /agents, /a     - List running background agents");
                    println!("  /kill <id>      - Cancel a background agent");
                    println!("  /status, /s     - Show current mode, model, and channel info");
                    println!("  /help, /h       - Show this help");
                    println!("  Ctrl+C          - Exit\n");
                    continue;
                }

                // Handle /agents command — list running subagents
                if input == "/agents" || input == "/a" {
                    let agents = agent_loop.subagent_manager().list_running().await;
                    if agents.is_empty() {
                        println!("\n  No agents running.\n");
                    } else {
                        println!("\n  Running agents:\n");
                        println!("  {:<10} {:<26} {}", "ID", "LABEL", "ELAPSED");
                        for a in &agents {
                            let elapsed = a.started_at.elapsed();
                            let mins = elapsed.as_secs() / 60;
                            let secs = elapsed.as_secs() % 60;
                            println!("  {:<10} {:<26} {}m {:02}s", a.task_id, a.label, mins, secs);
                        }
                        println!(
                            "\n  {} agent{} running. /kill <id> to cancel.\n",
                            agents.len(),
                            if agents.len() > 1 { "s" } else { "" }
                        );
                    }
                    continue;
                }

                // Handle /kill command — cancel a subagent
                if input.starts_with("/kill ") {
                    let id = input[6..].trim();
                    if id.is_empty() {
                        println!("\n  Usage: /kill <id>\n");
                    } else if agent_loop.subagent_manager().cancel(id).await {
                        println!("\n  Cancelled agent {}.\n", id);
                    } else {
                        println!("\n  No running agent matching '{}'.\n", id);
                    }
                    continue;
                }

                // Handle status command
                if input == "/status" || input == "/s" {
                    let core = core_handle.read().unwrap().clone();
                    let is_local = LOCAL_MODE.load(Ordering::SeqCst);
                    let model_name = &core.model;
                    let mode_label = if is_local { "local" } else { "cloud" };

                    println!();
                    println!("  {}MODE{}      {} ({})", tui::BOLD, tui::RESET, mode_label, model_name);

                    let used = core.last_context_used.load(Ordering::Relaxed) as usize;
                    let max = core.last_context_max.load(Ordering::Relaxed) as usize;
                    let pct = if max > 0 { (used * 100) / max } else { 0 };
                    let ctx_color = match pct {
                        0..=49 => tui::GREEN,
                        50..=79 => tui::YELLOW,
                        _ => tui::RED,
                    };
                    println!(
                        "  {}CONTEXT{}   {:>6} / {:>6} tokens ({}{}{}%{})",
                        tui::BOLD, tui::RESET,
                        format_thousands(used), format_thousands(max),
                        ctx_color, tui::BOLD, pct, tui::RESET
                    );

                    let obs_count = {
                        let obs = crate::agent::observer::ObservationStore::new(&core.workspace);
                        obs.count()
                    };
                    println!(
                        "  {}MEMORY{}    {} ({} observations)",
                        tui::BOLD, tui::RESET,
                        if core.memory_enabled { "enabled" } else { "disabled" },
                        obs_count
                    );

                    let agent_count = agent_loop.subagent_manager().get_running_count().await;
                    println!("  {}AGENTS{}    {} running", tui::BOLD, tui::RESET, agent_count);

                    active_channels.retain(|ch| !ch.handle.is_finished());
                    if !active_channels.is_empty() {
                        let ch_names: Vec<&str> = active_channels.iter().map(|c| match c.name.as_str() {
                            "whatsapp" => "wa",
                            "telegram" => "tg",
                            other => other,
                        }).collect();
                        println!("  {}CHANNELS{}  {}", tui::BOLD, tui::RESET, ch_names.join(" "));
                    }

                    let turn = core.learning_turn_counter.load(Ordering::Relaxed);
                    println!("  {}TURN{}      {}", tui::BOLD, tui::RESET, turn);

                    if is_local {
                        if let Some(ref cp) = compaction_port {
                            println!("  {}COMPACT{}   on port {} (CPU)", tui::BOLD, tui::RESET, cp);
                        }
                    }

                    println!();
                    continue;
                }

                // Process message (streaming)
                let channel = if voice_on { "voice" } else { "cli" };

                let (delta_tx, mut delta_rx) =
                    tokio::sync::mpsc::unbounded_channel::<String>();

                let print_task = tokio::spawn(async move {
                    use std::io::Write as _;
                    while let Some(delta) = delta_rx.recv().await {
                        print!("{}", delta);
                        std::io::stdout().flush().ok();
                    }
                    println!();
                });

                println!();
                let response = agent_loop
                    .process_direct_streaming(
                        input,
                        &session_id,
                        channel,
                        "direct",
                        None,
                        delta_tx,
                    )
                    .await;
                let _ = print_task.await;
                // Erase raw streamed text, re-render with markdown formatting.
                if !response.is_empty() {
                    use std::io::Write as _;
                    let lines = response.chars().filter(|&c| c == '\n').count() + 2; // +1 trailing newline, +1 blank
                    print!("\x1b[{}A\x1b[J", lines);
                    std::io::stdout().flush().ok();
                    let skin = make_skin();
                    render_markdown(&response, &skin);
                }
                println!();
                {
                    let sa_count = agent_loop.subagent_manager().get_running_count().await;
                    active_channels.retain(|ch| !ch.handle.is_finished());
                    let ch_names: Vec<&str> = active_channels.iter().map(|c| match c.name.as_str() {
                        "whatsapp" => "wa", "telegram" => "tg", other => other,
                    }).collect();
                    print_status_bar(&core_handle, &ch_names, sa_count);
                }

                #[cfg(feature = "voice")]
                if let Some(ref mut vs) = voice_session {
                    let tts_text = strip_markdown_for_tts(&response);
                    if !tts_text.is_empty() {
                        speak_interruptible(vs, &tts_text, "en");
                    }
                }
            }
            // Stop any active background channels
            for ch in &active_channels {
                ch.stop.store(true, Ordering::Relaxed);
            }
            if !active_channels.is_empty() {
                tokio::time::sleep(Duration::from_millis(500)).await;
                for ch in &active_channels {
                    ch.handle.abort();
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
            if let Some(ref mut child) = compaction_process {
                child.kill().ok();
                child.wait().ok();
            }

            println!("Goodbye!");
        }
    });
}

/// Build a `SharedCoreHandle` from config — called once at startup.
fn build_core_handle(
    config: &Config,
    local_port: &str,
    local_model_name: Option<&str>,
    compaction_port: Option<&str>,
) -> SharedCoreHandle {
    let is_local = LOCAL_MODE.load(Ordering::SeqCst);
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

    // Auto-detect context size from local server; fall back to config default.
    let max_context_tokens = if is_local {
        query_local_context_size(local_port).unwrap_or(config.agents.defaults.max_context_tokens)
    } else {
        config.agents.defaults.max_context_tokens
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

    let core = build_shared_core(
        provider,
        config.workspace_path(),
        model,
        config.agents.defaults.max_tool_iterations,
        config.agents.defaults.max_tokens,
        config.agents.defaults.temperature,
        max_context_tokens,
        brave_key,
        config.tools.exec_.timeout,
        config.tools.exec_.restrict_to_workspace,
        config.memory.clone(),
        is_local,
        cp,
    );
    Arc::new(std::sync::RwLock::new(Arc::new(core)))
}

/// Rebuild the shared core for `/local` toggle or `/model` swap.
///
/// All agents sharing this handle see the new provider on their next message.
fn rebuild_core(
    handle: &SharedCoreHandle,
    config: &Config,
    local_port: &str,
    local_model_name: Option<&str>,
    compaction_port: Option<&str>,
) {
    let is_local = LOCAL_MODE.load(Ordering::SeqCst);
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

    // Auto-detect context size from local server; fall back to config default.
    let max_context_tokens = if is_local {
        query_local_context_size(local_port).unwrap_or(config.agents.defaults.max_context_tokens)
    } else {
        config.agents.defaults.max_context_tokens
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

    let new_core = build_shared_core(
        provider,
        config.workspace_path(),
        model,
        config.agents.defaults.max_tool_iterations,
        config.agents.defaults.max_tokens,
        config.agents.defaults.temperature,
        max_context_tokens,
        brave_key,
        config.tools.exec_.timeout,
        config.tools.exec_.restrict_to_workspace,
        config.memory.clone(),
        is_local,
        cp,
    );
    *handle.write().unwrap() = Arc::new(new_core);
}

/// Create an agent loop with per-instance channels, using the shared core handle.
fn create_agent_loop(
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

/// Query the local llama.cpp server for its actual context size (`n_ctx`).
///
/// Returns the server's context window with 5% headroom subtracted (to account
/// for token-estimation drift). Falls back to `None` if the server is
/// unreachable or the response is unexpected.
fn query_local_context_size(port: &str) -> Option<usize> {
    let url = format!("http://localhost:{}/props", port);
    let props = reqwest::blocking::get(&url)
        .ok()?
        .json::<serde_json::Value>()
        .ok()?;
    let n_ctx = props
        .get("default_generation_settings")
        .and_then(|v| v.get("n_ctx"))
        .and_then(|v| v.as_u64())
        .or_else(|| props.get("n_ctx").and_then(|v| v.as_u64()))? as usize;
    let n_parallel = props
        .get("default_generation_settings")
        .and_then(|v| v.get("n_parallel"))
        .and_then(|v| v.as_u64())
        .or_else(|| props.get("n_parallel").and_then(|v| v.as_u64()))
        .unwrap_or(1)
        .max(1) as usize;
    let per_request_ctx = (n_ctx / n_parallel).max(1);
    // Apply 5% headroom — our char/4 estimator can overshoot slightly.
    Some((per_request_ctx as f64 * 0.95) as usize)
}

/// Format a number with thousands separators (e.g. 12430 -> "12,430").
fn format_thousands(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}

/// Print a compact status bar after each agent response.
fn print_status_bar(core_handle: &SharedCoreHandle, channel_names: &[&str], subagent_count: usize) {
    use std::sync::atomic::Ordering;
    let core = core_handle.read().unwrap().clone();
    let used = core.last_context_used.load(Ordering::Relaxed) as usize;
    let max = core.last_context_max.load(Ordering::Relaxed) as usize;
    let turn = core.learning_turn_counter.load(Ordering::Relaxed);

    let pct = if max > 0 { (used * 100) / max } else { 0 };
    let ctx_color = match pct {
        0..=49 => tui::GREEN,
        50..=79 => tui::YELLOW,
        _ => tui::RED,
    };

    let mut parts: Vec<String> = Vec::new();
    parts.push(format!(
        "ctx {}{}{}%{}",
        ctx_color,
        tui::BOLD,
        pct,
        tui::RESET
    ));

    if !channel_names.is_empty() {
        parts.push(format!(
            "{}{}{}",
            tui::CYAN,
            channel_names.join(" "),
            tui::RESET
        ));
    }

    if subagent_count > 0 {
        parts.push(format!(
            "{} agent{}",
            subagent_count,
            if subagent_count > 1 { "s" } else { "" }
        ));
    }

    parts.push(format!("t:{}", turn));

    println!("  {}{}{}", tui::DIM, parts.join(" | "), tui::RESET);
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
                if let Some(model) = json
                    .get("default_generation_settings")
                    .and_then(|s| s.get("model"))
                    .and_then(|m| m.as_str())
                {
                    let model_name = std::path::Path::new(model)
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or(model);
                    println!("  {DIM}Model: {RESET}{GREEN}{model_name}{RESET}");
                }
                if let Some(n_ctx) = json
                    .get("default_generation_settings")
                    .and_then(|s| s.get("n_ctx"))
                    .and_then(|n| n.as_u64())
                {
                    let n_parallel = json
                        .get("default_generation_settings")
                        .and_then(|s| s.get("n_parallel"))
                        .and_then(|n| n.as_u64())
                        .or_else(|| json.get("n_parallel").and_then(|n| n.as_u64()))
                        .unwrap_or(1)
                        .max(1);
                    let per_request = (n_ctx / n_parallel).max(1);
                    if n_parallel > 1 {
                        println!(
                            "  {DIM}Context: {RESET}{GREEN}{}K{RESET}{DIM} ({}K total / parallel {}){RESET}",
                            per_request / 1024,
                            n_ctx / 1024,
                            n_parallel
                        );
                    } else {
                        println!("  {DIM}Context: {RESET}{GREEN}{}K{RESET}", n_ctx / 1024);
                    }
                }
            }
        }
    } else {
        let config = load_config(None);
        println!(
            "  {BOLD}{CYAN}CLOUD MODE{RESET} {DIM}{}{RESET}",
            config.agents.defaults.model
        );
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
        println!(
            "  {BOLD}{CYAN}CLOUD{RESET} {DIM}{}{RESET}",
            config.agents.defaults.model
        );
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

    println!("{} Starting nanobot gateway on port {}...", LOGO, port);

    let config = load_config(None);
    check_api_key(&config);

    let core_handle = build_core_handle(&config, "8080", None, None);
    let runtime = tokio::runtime::Runtime::new().expect("Failed to create tokio runtime");
    runtime.block_on(run_gateway_async(config, core_handle, None, None));
}

/// Shared async gateway: creates channels, provider, cron, agent loop, and runs until stopped.
///
/// If `stop_signal` is `Some`, watches the flag for shutdown (used when spawned from REPL).
/// If `None`, watches for Ctrl+C (used for standalone CLI commands).
async fn run_gateway_async(
    config: Config,
    core_handle: SharedCoreHandle,
    stop_signal: Option<Arc<AtomicBool>>,
    repl_display_tx: Option<mpsc::UnboundedSender<String>>,
) {
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
    let voice_pipeline: Option<Arc<voice_pipeline::VoicePipeline>> = {
        match voice_pipeline::VoicePipeline::new().await {
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

fn cmd_whatsapp() {
    println!("{} Starting WhatsApp...\n", LOGO);

    let mut config = load_config(None);
    check_api_key(&config);

    config.channels.whatsapp.enabled = true;
    config.channels.telegram.enabled = false;
    config.channels.feishu.enabled = false;
    config.channels.email.enabled = false;

    println!("  Scan the QR code when it appears");
    println!("  Press Ctrl+C to stop\n");

    let core_handle = build_core_handle(&config, "8080", None, None);
    let runtime = tokio::runtime::Runtime::new().expect("Failed to create tokio runtime");
    runtime.block_on(run_gateway_async(config, core_handle, None, None));
}

fn cmd_telegram(token_arg: Option<String>) {
    println!("{} Starting Telegram...\n", LOGO);

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

    let core_handle = build_core_handle(&config, "8080", None, None);
    let runtime = tokio::runtime::Runtime::new().expect("Failed to create tokio runtime");
    runtime.block_on(run_gateway_async(config, core_handle, None, None));
}

fn cmd_email(
    imap_host_arg: Option<String>,
    smtp_host_arg: Option<String>,
    username_arg: Option<String>,
    password_arg: Option<String>,
) {
    println!("{} Starting Email...\n", LOGO);

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

    let core_handle = build_core_handle(&config, "8080", None, None);
    let runtime = tokio::runtime::Runtime::new().expect("Failed to create tokio runtime");
    runtime.block_on(run_gateway_async(config, core_handle, None, None));
}

/// Validate a Telegram bot token by calling the getMe API.
fn validate_telegram_token(token: &str) -> bool {
    let url = format!("https://api.telegram.org/bot{}/getMe", token);
    reqwest::blocking::get(&url)
        .ok()
        .and_then(|r| r.json::<serde_json::Value>().ok())
        .and_then(|d| d.get("ok")?.as_bool())
        .unwrap_or(false)
}

/// Check that an LLM API key is configured, exit with error if not.
fn check_api_key(config: &Config) {
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

fn cmd_status() {
    let config_path = get_config_path();
    let config = load_config(None);
    let workspace = config.workspace_path();

    println!("{} nanobot Status\n", LOGO);
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
            Ok(resp) if resp.status().is_success() => "🟢 running",
            _ => "⚪ stopped",
        };
        println!("  {} (port {}): {}", name, port, status);
    }
}

fn cmd_tune(input_path: String, json: bool) {
    let path = std::path::PathBuf::from(input_path);
    match run_tune_from_path(&path, json) {
        Ok(output) => println!("{}", output),
        Err(e) => {
            eprintln!("Tune failed: {}", e);
            std::process::exit(1);
        }
    }
}

fn run_tune_from_path(path: &std::path::Path, as_json: bool) -> Result<String, String> {
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

fn cmd_channels_status() {
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
    models.sort_by(|a, b| a.file_name().cmp(&b.file_name()));
    models
}

const DEFAULT_LOCAL_MODEL: &str = "NVIDIA-Nemotron-Nano-9B-v2-Q4_K_M.gguf";

const COMPACTION_MODEL_URL: &str =
    "https://huggingface.co/MaziyarPanahi/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B.Q4_K_M.gguf";
const COMPACTION_MODEL_FILENAME: &str = "Qwen3-0.6B.Q4_K_M.gguf";

/// Ensure the dedicated compaction model is available locally.
///
/// Downloads Qwen3-0.6B Q4_K_M (~500MB) to `~/.nanobot/models/` if not already
/// present. Returns `None` on failure (graceful degradation — compaction just
/// gets skipped and the system falls back to `trim_to_fit`).
fn ensure_compaction_model() -> Option<std::path::PathBuf> {
    let models_dir = dirs::home_dir()?.join(".nanobot").join("models");
    std::fs::create_dir_all(&models_dir).ok()?;

    let model_path = models_dir.join(COMPACTION_MODEL_FILENAME);
    if model_path.exists() {
        return Some(model_path);
    }

    println!(
        "  {}{}Downloading{} compaction model (Qwen3-0.6B, ~500MB)...",
        tui::BOLD,
        tui::YELLOW,
        tui::RESET
    );

    let tmp_path = models_dir.join(format!("{}.downloading", COMPACTION_MODEL_FILENAME));
    let result = (|| -> Result<(), Box<dyn std::error::Error>> {
        let mut resp = reqwest::blocking::get(COMPACTION_MODEL_URL)?;
        if !resp.status().is_success() {
            return Err(format!("HTTP {}", resp.status()).into());
        }
        let mut file = std::fs::File::create(&tmp_path)?;
        resp.copy_to(&mut file)?;
        std::fs::rename(&tmp_path, &model_path)?;
        Ok(())
    })();

    match result {
        Ok(()) => {
            println!(
                "  {}{}Done{} — saved to {}",
                tui::BOLD,
                tui::GREEN,
                tui::RESET,
                model_path.display()
            );
            Some(model_path)
        }
        Err(e) => {
            println!(
                "  {}{}Download failed:{} {} (compaction will use trim_to_fit fallback)",
                tui::BOLD,
                tui::YELLOW,
                tui::RESET,
                e
            );
            // Clean up partial download
            let _ = std::fs::remove_file(&tmp_path);
            None
        }
    }
}

/// Start the dedicated compaction server if the model is available.
///
/// Downloads the model on first run, spawns a GPU-accelerated llama-server on
/// port 8090+ with context matching the main model, and stores the process
/// handle and port. Gracefully degrades if anything fails.
async fn start_compaction_if_available(
    compaction_process: &mut Option<std::process::Child>,
    compaction_port: &mut Option<String>,
    main_ctx_size: usize,
) {
    // Already running?
    if compaction_process.is_some() {
        return;
    }

    let model_path = match ensure_compaction_model() {
        Some(p) => p,
        None => return,
    };

    let port = find_available_port(8090);
    println!(
        "  {}{}Starting{} compaction server on port {} (ctx: {}K, GPU)...",
        tui::BOLD,
        tui::YELLOW,
        tui::RESET,
        port,
        main_ctx_size / 1024,
    );

    match spawn_compaction_server(port, &model_path, main_ctx_size) {
        Ok(child) => {
            *compaction_process = Some(child);
            if wait_for_server_ready(port, 15, compaction_process).await {
                *compaction_port = Some(port.to_string());
                println!(
                    "  {}{}Compaction server ready{} (Qwen3-0.6B on GPU)",
                    tui::BOLD,
                    tui::GREEN,
                    tui::RESET
                );
            } else {
                println!(
                    "  {}{}Compaction server failed to start{} (using trim_to_fit fallback)",
                    tui::BOLD,
                    tui::YELLOW,
                    tui::RESET
                );
                if let Some(ref mut child) = compaction_process {
                    child.kill().ok();
                    child.wait().ok();
                }
                *compaction_process = None;
            }
        }
        Err(e) => {
            println!(
                "  {}{}Compaction server failed:{} {} (using trim_to_fit fallback)",
                tui::BOLD,
                tui::YELLOW,
                tui::RESET,
                e
            );
        }
    }
}

/// Stop the compaction server and clear state.
fn stop_compaction_server(
    compaction_process: &mut Option<std::process::Child>,
    compaction_port: &mut Option<String>,
) {
    if let Some(ref mut child) = compaction_process {
        child.kill().ok();
        child.wait().ok();
    }
    *compaction_process = None;
    *compaction_port = None;
}

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

// ---------------------------------------------------------------------------
// Auto-size context window: GGUF metadata + system resources
// ---------------------------------------------------------------------------

struct GgufModelInfo {
    n_layers: u32,
    n_kv_heads: u32,
    n_heads: u32,
    embedding_dim: u32,
    context_length: u32,
}

/// Parse architecture-specific metadata from a GGUF file header.
fn parse_gguf_metadata(path: &std::path::Path) -> Option<GgufModelInfo> {
    use std::io::{Read, Seek, SeekFrom};

    let mut f = std::fs::File::open(path).ok()?;
    let mut buf4 = [0u8; 4];
    let mut buf8 = [0u8; 8];

    // Magic "GGUF"
    f.read_exact(&mut buf4).ok()?;
    if &buf4 != b"GGUF" {
        return None;
    }

    // Version (u32 LE) — we support v2 and v3
    f.read_exact(&mut buf4).ok()?;
    let version = u32::from_le_bytes(buf4);
    if version < 2 {
        return None;
    }

    // tensor_count (u64), kv_count (u64)
    f.read_exact(&mut buf8).ok()?;
    let _tensor_count = u64::from_le_bytes(buf8);
    f.read_exact(&mut buf8).ok()?;
    let kv_count = u64::from_le_bytes(buf8);

    fn gguf_read_string(f: &mut std::fs::File) -> Option<String> {
        let mut b8 = [0u8; 8];
        f.read_exact(&mut b8).ok()?;
        let len = u64::from_le_bytes(b8) as usize;
        if len > 256 {
            f.seek(SeekFrom::Current(len as i64)).ok()?;
            return Some(String::new());
        }
        let mut s = vec![0u8; len];
        f.read_exact(&mut s).ok()?;
        String::from_utf8(s).ok()
    }

    fn gguf_skip_value(f: &mut std::fs::File, vtype: u32) -> Option<()> {
        match vtype {
            0 | 1 | 7 => {
                let mut b = [0u8; 1];
                f.read_exact(&mut b).ok()?;
            }
            2 | 3 => {
                let mut b = [0u8; 2];
                f.read_exact(&mut b).ok()?;
            }
            4 | 5 | 6 => {
                let mut b = [0u8; 4];
                f.read_exact(&mut b).ok()?;
            }
            8 => {
                gguf_read_string(f)?;
            }
            9 => {
                let mut tb = [0u8; 4];
                f.read_exact(&mut tb).ok()?;
                let elem_type = u32::from_le_bytes(tb);
                let mut cb = [0u8; 8];
                f.read_exact(&mut cb).ok()?;
                let count = u64::from_le_bytes(cb);
                for _ in 0..count {
                    gguf_skip_value(f, elem_type)?;
                }
            }
            10 | 11 | 12 => {
                let mut b = [0u8; 8];
                f.read_exact(&mut b).ok()?;
            }
            _ => return None,
        }
        Some(())
    }

    let mut arch = String::new();
    let mut n_layers: Option<u32> = None;
    let mut n_kv_heads: Option<u32> = None;
    let mut n_heads: Option<u32> = None;
    let mut embedding_dim: Option<u32> = None;
    let mut context_length: Option<u32> = None;

    for _ in 0..kv_count {
        let key = match gguf_read_string(&mut f) {
            Some(k) => k,
            None => return None,
        };

        // Read value type
        f.read_exact(&mut buf4).ok()?;
        let vtype = u32::from_le_bytes(buf4);

        if key == "general.architecture" && vtype == 8 {
            arch = gguf_read_string(&mut f)?;
            continue;
        }

        // Check for u32 metadata fields (type 4 = u32, type 5 = i32)
        if (vtype == 4 || vtype == 5) && !arch.is_empty() {
            let mut vb = [0u8; 4];
            f.read_exact(&mut vb).ok()?;
            let val = u32::from_le_bytes(vb);
            if key == format!("{}.block_count", arch) {
                n_layers = Some(val);
            } else if key == format!("{}.attention.head_count_kv", arch) {
                n_kv_heads = Some(val);
            } else if key == format!("{}.attention.head_count", arch) {
                n_heads = Some(val);
            } else if key == format!("{}.embedding_length", arch) {
                embedding_dim = Some(val);
            } else if key == format!("{}.context_length", arch) {
                context_length = Some(val);
            }
            continue;
        }

        // Skip values we don't need
        gguf_skip_value(&mut f, vtype)?;
    }

    Some(GgufModelInfo {
        n_layers: n_layers?,
        n_kv_heads: n_kv_heads?,
        n_heads: n_heads?,
        embedding_dim: embedding_dim?,
        context_length: context_length?,
    })
}

/// Detect available VRAM (via nvidia-smi) and RAM (via /proc/meminfo).
/// Returns (vram_bytes, ram_bytes).
fn detect_available_memory() -> (Option<u64>, u64) {
    // Try VRAM via nvidia-smi
    let vram = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.free", "--format=csv,noheader,nounits"])
        .output()
        .ok()
        .and_then(|out| {
            if !out.status.success() {
                return None;
            }
            let s = String::from_utf8_lossy(&out.stdout);
            s.trim().lines().next()?.trim().parse::<u64>().ok()
        })
        .map(|mib| mib * 1024 * 1024);

    // RAM via /proc/meminfo
    let ram = std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|contents| {
            for line in contents.lines() {
                if line.starts_with("MemAvailable:") {
                    let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
                    return Some(kb * 1024);
                }
            }
            None
        })
        .unwrap_or(8 * 1024 * 1024 * 1024); // 8 GB fallback

    (vram, ram)
}

/// Practical context cap based on model file size (proxy for parameter count).
///
/// Small models become unresponsive with very large contexts — attention is O(n²)
/// and they lack the capacity to utilize long contexts effectively.
fn practical_context_cap(model_file_size_bytes: u64) -> usize {
    let gb = model_file_size_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    if gb < 2.0 {
        8192
    }
    // tiny (~1-3B heavy quant)
    else if gb < 4.0 {
        16384
    }
    // small (~3-7B)
    else if gb < 8.0 {
        32768
    }
    // medium (~7-14B)
    else if gb < 16.0 {
        65536
    }
    // large (~14-30B)
    else {
        usize::MAX
    } // xlarge (30B+) — no cap
}

/// Compute optimal --ctx-size for a GGUF model given available system resources.
fn compute_optimal_context_size(model_path: &std::path::Path) -> usize {
    const OVERHEAD: u64 = 512 * 1024 * 1024; // 512 MB
    const FALLBACK_CTX: usize = 16384;

    let model_file_size = std::fs::metadata(model_path).map(|m| m.len()).unwrap_or(0);

    let gguf = match parse_gguf_metadata(model_path) {
        Some(info) => info,
        None => {
            // GGUF parse failed — use practical cap based on file size, or fallback
            let cap = practical_context_cap(model_file_size).min(FALLBACK_CTX);
            debug!(
                "GGUF parse failed for {}, using {}K context (file size: {:.1}GB)",
                model_path.display(),
                cap / 1024,
                model_file_size as f64 / 1e9
            );
            return cap;
        }
    };

    let head_dim = gguf.embedding_dim / gguf.n_heads;
    // KV cache per token (FP16): 2 (K+V) × layers × kv_heads × head_dim × 2 bytes
    let kv_per_token = 2u64 * gguf.n_layers as u64 * gguf.n_kv_heads as u64 * head_dim as u64 * 2;

    let (vram, ram) = detect_available_memory();

    let available_for_kv = if let Some(vram_bytes) = vram {
        // GPU mode: VRAM must hold model weights + KV cache
        vram_bytes
            .saturating_sub(model_file_size)
            .saturating_sub(OVERHEAD)
    } else {
        // CPU mode: weights are mmap'd, RAM mainly for KV cache
        ram.saturating_sub(OVERHEAD)
    };

    if kv_per_token == 0 {
        let cap = practical_context_cap(model_file_size).min(FALLBACK_CTX);
        debug!("KV per token is 0, using {}K context", cap / 1024);
        return cap;
    }

    let max_ctx_from_memory = (available_for_kv / kv_per_token) as usize;
    let cap = practical_context_cap(model_file_size);
    // Clamp: at least 4096, at most min(memory allows, GGUF native, practical cap)
    let ctx = max_ctx_from_memory
        .max(4096)
        .min(gguf.context_length as usize)
        .min(cap);
    // Round down to nearest 1024
    let ctx = (ctx / 1024) * 1024;

    let mem_source = if vram.is_some() { "VRAM" } else { "RAM" };
    debug!(
        "Auto-sized context: {} tokens ({}K) — kv/tok={}B, available {}={:.1}GB, model={:.1}GB, practical_cap={}K",
        ctx, ctx / 1024, kv_per_token,
        mem_source, available_for_kv as f64 / 1e9,
        model_file_size as f64 / 1e9,
        cap / 1024,
    );

    ctx
}

/// Spawn a llama-server for context compaction (summarization).
///
/// Uses GPU acceleration and matches the main model's context size so
/// large conversations can be summarized in a single LLM call.
fn spawn_compaction_server(
    port: u16,
    model_path: &std::path::Path,
    ctx_size: usize,
) -> Result<std::process::Child, String> {
    let home = dirs::home_dir().ok_or("Could not determine home directory")?;
    let server_path = home.join("llama.cpp/build/bin/llama-server");

    if !server_path.exists() {
        return Err(format!(
            "llama-server not found at {}",
            server_path.display()
        ));
    }
    if !model_path.exists() {
        return Err(format!(
            "Compaction model not found at {}",
            model_path.display()
        ));
    }

    std::process::Command::new(&server_path)
        .arg("--model")
        .arg(model_path)
        .arg("--port")
        .arg(port.to_string())
        .arg("--ctx-size")
        .arg(ctx_size.to_string())
        .arg("--parallel")
        .arg("1")
        .arg("--n-gpu-layers")
        .arg("99")
        .arg("--flash-attn")
        .arg("on")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to spawn compaction server: {}", e))
}

fn spawn_llama_server(
    port: u16,
    model_path: &std::path::Path,
    ctx_size: usize,
) -> Result<std::process::Child, String> {
    let home = dirs::home_dir().ok_or("Could not determine home directory")?;
    let server_path = home.join("llama.cpp/build/bin/llama-server");

    if !server_path.exists() {
        return Err(format!(
            "llama-server not found at {}",
            server_path.display()
        ));
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
        .arg(ctx_size.to_string())
        .arg("--parallel")
        .arg("1")
        .arg("--n-gpu-layers")
        .arg("99")
        .arg("--flash-attn")
        .arg("on")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to spawn llama-server: {}", e))
}

async fn wait_for_server_ready(
    port: u16,
    timeout_secs: u64,
    llama_process: &mut Option<std::process::Child>,
) -> bool {
    use std::io::Write;

    // Drain stderr in a background thread so the pipe buffer doesn't block the server.
    let stderr_lines: Arc<std::sync::Mutex<Vec<String>>> =
        Arc::new(std::sync::Mutex::new(Vec::new()));
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
                print!(
                    "\r{}{}{}  ",
                    tui::SHOW_CURSOR,
                    tui::RESET,
                    " ".repeat(bar_width + 30)
                );
                print!(
                    "\r  {}Server exited unexpectedly{}\n",
                    tui::YELLOW,
                    tui::RESET
                );
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
            tui::DIM,
            tui::RESET,
            tui::CYAN,
            "\u{2588}".repeat(filled), // █
            "\u{2591}".repeat(empty),  // ░
            tui::DIM,
            elapsed,
            tui::RESET,
        );
        std::io::stdout().flush().ok();

        if let Ok(resp) = client.get(&url).send().await {
            if resp.status().is_success() {
                // Fill bar to 100% briefly
                print!(
                    "\r  {}Loading model [{}{}{}] done{}",
                    tui::DIM,
                    tui::RESET,
                    tui::CYAN,
                    "\u{2588}".repeat(bar_width),
                    tui::RESET,
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
pub(crate) fn strip_markdown_for_tts(text: &str) -> String {
    let mut out = String::new();
    let mut in_code_block = false;

    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("```") {
            in_code_block = !in_code_block;
            continue;
        }
        if in_code_block {
            continue;
        }

        let line = trimmed.trim_start_matches('#').trim();
        if line.is_empty() {
            continue;
        }

        for c in line.chars() {
            match c {
                'a'..='z' | 'A'..='Z' | '0'..='9' => out.push(c),
                ' ' | '.' | ',' | '!' | '?' | ';' | ':' | '\'' | '"' | '-' | '(' | ')' => {
                    out.push(c)
                }
                '*' | '_' | '`' | '~' | '[' | ']' | '|' | '#' => {} // strip markdown syntax
                _ if c.is_alphabetic() => out.push(c),              // keep non-English letters
                _ => {}                                             // strip emojis, arrows, etc.
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
        unsafe {
            libc::tcflush(fd, libc::TCIFLUSH);
        }
    }
}

/// Speak with TTS while watching for user interrupt (Enter or Ctrl+Space).
/// Returns true if the user interrupted (wants to speak next).
#[cfg(feature = "voice")]
fn speak_interruptible(vs: &mut voice::VoiceSession, text: &str, lang: &str) -> bool {
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

    if let Err(e) = vs.speak(text, lang) {
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
                if trimmed.is_empty() {
                    VoiceAction::Record
                } else {
                    VoiceAction::Text(trimmed)
                }
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
    use std::fs;
    use std::sync::atomic::AtomicUsize;
    use tempfile::tempdir;

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
            assert!(
                found >= base + 2,
                "Should skip both occupied ports, got {}",
                found
            );
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
            let result = spawn_llama_server(19876, &fake_model, 8192);
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
            let result = spawn_llama_server(19877, &model_path, 8192);
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
                    let resp =
                        "HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Length: 2\r\n\r\nok";
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

    #[test]
    fn test_cli_parses_tune_command() {
        let cli = Cli::try_parse_from(["nanobot", "tune", "--input", "bench.json"]).unwrap();
        match cli.command {
            Commands::Tune { input, json } => {
                assert_eq!(input, "bench.json");
                assert!(!json);
            }
            other => panic!("unexpected parsed command: {:?}", std::mem::discriminant(&other)),
        }
    }

    #[test]
    fn test_run_tune_from_path_selects_best_profile() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("bench.json");
        let payload = r#"{
  "measurements": [
    {
      "profile": {
        "id": "fast",
        "model": "slm-a",
        "ctx_size": 16384,
        "max_tokens": 768,
        "temperature": 0.3
      },
      "sample": {
        "ttft_ms": 650.0,
        "output_toks_per_sec": 95.0,
        "quality_score": 0.81,
        "tool_success_rate": 0.95,
        "context_overflow_rate": 0.0
      }
    },
    {
      "profile": {
        "id": "slow",
        "model": "slm-b",
        "ctx_size": 16384,
        "max_tokens": 768,
        "temperature": 0.3
      },
      "sample": {
        "ttft_ms": 1300.0,
        "output_toks_per_sec": 40.0,
        "quality_score": 0.82,
        "tool_success_rate": 0.95,
        "context_overflow_rate": 0.0
      }
    }
  ]
}"#;
        fs::write(&path, payload).unwrap();

        let output = run_tune_from_path(&path, false).expect("expected tuned profile output");
        assert!(output.contains("fast"), "output: {}", output);
    }

    #[test]
    fn test_run_tune_from_path_json_output() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("bench.json");
        let payload = r#"{
  "measurements": [
    {
      "profile": {
        "id": "balanced",
        "model": "slm-c",
        "ctx_size": 16384,
        "max_tokens": 768,
        "temperature": 0.3
      },
      "sample": {
        "ttft_ms": 800.0,
        "output_toks_per_sec": 70.0,
        "quality_score": 0.86,
        "tool_success_rate": 0.97,
        "context_overflow_rate": 0.0
      }
    }
  ]
}"#;
        fs::write(&path, payload).unwrap();

        let output = run_tune_from_path(&path, true).expect("expected JSON output");
        let parsed: serde_json::Value = serde_json::from_str(&output).unwrap();
        assert_eq!(parsed["profile"]["id"].as_str(), Some("balanced"));
    }
}
