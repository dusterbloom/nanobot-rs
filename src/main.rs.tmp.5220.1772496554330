//! nanobot - A lightweight personal AI assistant framework in Rust.
//! Based on nanobot by HKUDS (https://github.com/HKUDS/nanobot).
//!
//! Local LLM support: Use Ctrl+L or /local to toggle between cloud and local mode.

mod agent;
mod bus;
mod channels;
#[cfg(feature = "cluster")]
mod cluster;
mod cli;
mod config;
mod cron;
mod errors;
mod heartbeat;
mod lms;
#[cfg(feature = "voice")]
mod realtime;
mod providers;
mod repl;
mod server;
mod session;
mod sessions_cmd;
mod syntax;
mod tui;
mod utils;
#[cfg(feature = "voice")]
mod voice;
#[cfg(feature = "voice")]
mod voice_pipeline;

use std::io::IsTerminal;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

use clap::{Parser, Subcommand};

pub(crate) const VERSION: &str = "0.1.0";
pub(crate) const LOGO: &str = "*";

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
        /// Language hint for voice TTS engine (e.g. "en" uses faster Pocket).
        #[arg(long)]
        lang: Option<String>,
        /// Resume the most recent session for the session key.
        #[arg(short = 'c', long = "continue")]
        continue_session: bool,
        /// Resume a specific session by ID.
        #[arg(short = 'r', long)]
        resume: Option<String>,
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
    /// Ingest documents into the knowledge store for search.
    Ingest {
        /// File path(s) to ingest.
        #[arg(required = true)]
        files: Vec<String>,
        /// Custom source name (defaults to filename).
        #[arg(short, long)]
        name: Option<String>,
        /// Chunk size in characters. Default: 4096.
        #[arg(long, default_value_t = 4096)]
        chunk_size: usize,
        /// Overlap between chunks in characters. Default: 256.
        #[arg(long, default_value_t = 256)]
        overlap: usize,
    },
    /// Search the knowledge store.
    Search {
        /// Search query.
        query: String,
        /// Maximum results. Default: 5.
        #[arg(short, long, default_value_t = 5)]
        limit: usize,
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
    /// Manage sessions and log files.
    Sessions {
        #[command(subcommand)]
        action: SessionsAction,
    },
    /// Run evaluation benchmarks.
    Eval {
        #[command(subcommand)]
        action: EvalAction,
    },
    /// Start realtime voice session with LLM agent.
    #[cfg(feature = "voice")]
    Realtime {
        /// TTS engine: pocket, kokoro, qwen, qwenLarge, qwenOnnx, qwenOnnxInt8.
        #[arg(long, default_value = "pocket")]
        engine: String,
        /// Voice name for Qwen engines (e.g., ryan, serena).
        #[arg(long, default_value = "ryan")]
        voice: String,
        /// Session ID.
        #[arg(short, long, default_value = "realtime:default")]
        session: String,
        /// Use local LLM instead of cloud API.
        #[arg(short, long)]
        local: bool,
    },
    /// Start WebSocket server for OpenAI-compatible realtime API.
    #[cfg(feature = "voice")]
    #[command(name = "realtime-server")]
    RealtimeServer {
        /// Port to listen on.
        #[arg(short, long, default_value_t = 8080)]
        port: u16,
        /// TTS engine: pocket, kokoro, qwen, qwenLarge.
        #[arg(long, default_value = "pocket")]
        engine: String,
        /// Voice name for TTS.
        #[arg(long, default_value = "ryan")]
        voice: String,
        /// Host to bind to.
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
    },
    /// Manage voice profiles and voice cloning.
    #[cfg(feature = "voice")]
    Voice {
        #[command(subcommand)]
        action: VoiceAction,
    },
}

#[cfg(feature = "voice")]
#[derive(Subcommand)]
enum VoiceAction {
    /// List available voices for the specified TTS engine.
    List {
        /// TTS engine: pocket, kokoro, qwen, qwenLarge.
        #[arg(long, default_value = "qwen")]
        engine: String,
    },
    /// Clone a voice from a reference audio file (requires QwenLarge).
    Clone {
        /// Name for the cloned voice profile.
        name: String,
        /// Path to reference audio file (.wav).
        audio: String,
        /// Optional transcript of the reference audio.
        #[arg(short, long)]
        transcript: Option<String>,
    },
    /// Show voice configuration help.
    Config,
}

#[derive(Subcommand)]
enum EvalAction {
    /// Run Towers of Hanoi benchmark (MAKER replication).
    Hanoi {
        /// Number of disks. Default: 5.
        #[arg(short, long, default_value_t = 5)]
        disks: u8,
        /// Run calibration (measure model accuracy on sampled steps).
        #[arg(long)]
        calibrate: bool,
        /// Number of calibration samples. Default: 100.
        #[arg(long, default_value_t = 100)]
        samples: usize,
        /// Run full solve with MAKER voting.
        #[arg(long)]
        solve: bool,
        /// Enable CATTS confidence gating.
        #[arg(long)]
        catts: bool,
        /// Ahead-by-k margin for voting. Default: 1.
        #[arg(short, long, default_value_t = 1)]
        k: usize,
        /// Use local LLM.
        #[arg(short, long)]
        local: bool,
        /// Local server port. Default: 8080.
        #[arg(long, default_value_t = 8080)]
        port: u16,
    },
    /// Run Aggregation Haystack benchmark (Oolong-inspired).
    Haystack {
        /// Number of synthetic facts. Default: 50.
        #[arg(long, default_value_t = 50)]
        facts: usize,
        /// Total document length in characters. Default: 100000.
        #[arg(long, default_value_t = 100_000)]
        length: usize,
        /// Run Tier 2: aggregation tasks with LLM.
        #[arg(long)]
        aggregate: bool,
        /// Use local LLM.
        #[arg(short, long)]
        local: bool,
        /// Local server port. Default: 8080.
        #[arg(long, default_value_t = 8080)]
        port: u16,
    },
    /// Run Learning Curve benchmark (SWE-Bench-CL inspired).
    Learn {
        /// Task family: arithmetic, fact-retrieval, tool-chain.
        #[arg(long, default_value = "arithmetic")]
        family: String,
        /// Number of tasks in the curriculum. Default: 50.
        #[arg(long, default_value_t = 50)]
        tasks: usize,
        /// Depth/complexity parameter. Default: 3.
        #[arg(long, default_value_t = 3)]
        depth: usize,
        /// Use local LLM.
        #[arg(short, long)]
        local: bool,
        /// Local server port. Default: 8080.
        #[arg(long, default_value_t = 8080)]
        port: u16,
    },
    /// Run Research Sprint compound benchmark.
    Sprint {
        /// Corpus size in characters. Default: 500000.
        #[arg(long, default_value_t = 500_000)]
        corpus_size: usize,
        /// Number of questions. Default: 20.
        #[arg(long, default_value_t = 20)]
        questions: usize,
        /// Use local LLM.
        #[arg(short, long)]
        local: bool,
        /// Local server port. Default: 8080.
        #[arg(long, default_value_t = 8080)]
        port: u16,
    },
    /// Show saved evaluation results.
    Report,
}

#[derive(Subcommand)]
enum SessionsAction {
    /// List all sessions with date, size, and message count.
    List,
    /// Resume an existing session's REPL.
    Resume {
        /// Session ID to resume (from `sessions list`).
        id: String,
        /// Use local LLM instead of cloud API.
        #[arg(short, long)]
        local: bool,
    },
    /// Start a fresh named session.
    New {
        /// Optional human-readable label (default: cli:<uuid8>).
        #[arg(long)]
        name: Option<String>,
        /// Use local LLM instead of cloud API.
        #[arg(short, long)]
        local: bool,
    },
    /// Export a session to stdout (markdown or JSONL).
    Export {
        /// Session key to export.
        key: String,
        /// Output format: "md" (default) or "jsonl".
        #[arg(long, default_value = "md")]
        format: String,
    },
    /// Purge session and log files older than the given duration.
    Purge {
        /// Age threshold (e.g. "7d", "24h", "30d").
        #[arg(long)]
        older_than: String,
    },
    /// Gzip session files older than 24 hours.
    Archive,
    /// Wipe all sessions, logs, and metrics.
    Nuke {
        /// Skip confirmation prompt.
        #[arg(long)]
        force: bool,
    },
    /// Index orphaned JSONL sessions into searchable SESSION_*.md files.
    Index,
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
    // Safety net: restore terminal state on panic
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        tui::force_exit_raw_mode();
        print!("\x1b[r"); // reset scroll region
        print!("\x1b[?25h"); // show cursor
        let _ = std::io::Write::flush(&mut std::io::stdout());
        default_hook(info);
    }));

    let cli = Cli::parse();

    // Detect interactive REPL mode: `nanobot agent` with no -m message and a TTY.
    let is_interactive_repl = matches!(&cli.command, Commands::Agent { message: None, .. })
        && std::io::stdout().is_terminal();

    // Always suppress noisy crates regardless of RUST_LOG setting.
    // When RUST_LOG is set (e.g. "debug"), append mandatory filters so html5ever
    // and other spammy crates don't flood the log file.
    let noisy_crate_filters = ",html5ever=error,ort=off,pocket_tts=off,hyper=warn,reqwest=warn,rustyline=warn";
    let env_filter = match tracing_subscriber::EnvFilter::try_from_default_env() {
        Ok(_) => {
            // RUST_LOG is set — append our mandatory suppressions
            let combined = format!("{}{}", std::env::var("RUST_LOG").unwrap_or_default(), noisy_crate_filters);
            tracing_subscriber::EnvFilter::new(combined)
        }
        Err(_) => {
            tracing_subscriber::EnvFilter::new(format!("warn{}", noisy_crate_filters))
        }
    };

    // Chrome tracing: build layer + guard (feature-gated).
    // Guard must live until program exit to flush the trace file.
    // Don't drop guard in async context — it can cause tokio panic on /local toggle.
    #[cfg(feature = "trace-chrome")]
    let mut _chrome_guard_opt: Option<tracing_chrome::FlushGuard> = None;

    if is_interactive_repl {
        // Redirect tracing to a daily-rotated log file to prevent WARN logs from
        // interleaving with streaming output on stderr.  Rolling appender produces
        // files like `nanobot.log.2026-02-20` and keeps the current day open.
        let log_dir = dirs::home_dir()
            .unwrap_or_default()
            .join(".nanobot")
            .join("logs");
        let _ = std::fs::create_dir_all(&log_dir);
        let file_appender = tracing_appender::rolling::daily(&log_dir, "nanobot.log");

        let fmt_layer = tracing_subscriber::fmt::layer()
            .with_writer(file_appender)
            .json()
            .with_span_events(tracing_subscriber::fmt::format::FmtSpan::CLOSE)
            .with_ansi(false);

        #[cfg(not(feature = "trace-chrome"))]
        {
            tracing_subscriber::registry()
                .with(env_filter)
                .with(fmt_layer)
                .try_init()
                .ok();
        }

        #[cfg(feature = "trace-chrome")]
        {
            let trace_dir = dirs::home_dir()
                .unwrap_or_default()
                .join(".nanobot")
                .join("traces");
            let _ = std::fs::create_dir_all(&trace_dir);
            let trace_path = trace_dir.join(format!(
                "nanobot-{}.json",
                chrono::Local::now().format("%Y%m%d-%H%M%S")
            ));
            eprintln!("[trace] Writing chrome trace to {}", trace_path.display());
            let (chrome_layer, guard) = tracing_chrome::ChromeLayerBuilder::new()
                .file(trace_path)
                .include_args(true)
                .build();
            _chrome_guard_opt = Some(guard);

            tracing_subscriber::registry()
                .with(env_filter)
                .with(fmt_layer)
                .with(chrome_layer)
                .try_init()
                .ok();
        }
    } else {
        let fmt_layer = tracing_subscriber::fmt::layer();

        #[cfg(not(feature = "trace-chrome"))]
        {
            tracing_subscriber::registry()
                .with(env_filter)
                .with(fmt_layer)
                .try_init()
                .ok();
        }

        #[cfg(feature = "trace-chrome")]
        {
            let trace_dir = dirs::home_dir()
                .unwrap_or_default()
                .join(".nanobot")
                .join("traces");
            let _ = std::fs::create_dir_all(&trace_dir);
            let trace_path = trace_dir.join(format!(
                "nanobot-{}.json",
                chrono::Local::now().format("%Y%m%d-%H%M%S")
            ));
            eprintln!("[trace] Writing chrome trace to {}", trace_path.display());
            let (chrome_layer, guard) = tracing_chrome::ChromeLayerBuilder::new()
                .file(trace_path)
                .include_args(true)
                .build();
            _chrome_guard_opt = Some(guard);

            tracing_subscriber::registry()
                .with(env_filter)
                .with(fmt_layer)
                .with(chrome_layer)
                .try_init()
                .ok();
        }
    }

    tracing::info!(
        version = env!("CARGO_PKG_VERSION"),
        command = %format!("{:?}", std::mem::discriminant(&cli.command)),
        "nanobot started"
    );

    match cli.command {
        Commands::Onboard => cli::cmd_onboard(),
        Commands::Agent {
            message,
            session,
            local,
            lang,
            continue_session: _continue_session,
            resume: _resume,
        } => repl::cmd_agent(message, session, local, lang),
        Commands::Gateway { port, verbose } => cli::cmd_gateway(port, verbose),
        Commands::Status => cli::cmd_status(),
        Commands::Tune { input, json } => cli::cmd_tune(input, json),
        Commands::Channels { action } => match action {
            ChannelsAction::Status => cli::cmd_channels_status(),
        },
        Commands::Cron { action } => match action {
            CronAction::List { all } => cli::cmd_cron_list(all),
            CronAction::Add {
                name,
                message,
                every,
                cron,
                deliver,
                to,
                channel,
            } => cli::cmd_cron_add(name, message, every, cron, deliver, to, channel),
            CronAction::Remove { job_id } => cli::cmd_cron_remove(job_id),
            CronAction::Enable { job_id, disable } => cli::cmd_cron_enable(job_id, disable),
        },
        Commands::Ingest {
            files,
            name,
            chunk_size,
            overlap,
        } => cli::cmd_ingest(files, name, chunk_size, overlap),
        Commands::Search { query, limit } => cli::cmd_search(query, limit),
        Commands::WhatsApp => cli::cmd_whatsapp(),
        Commands::Telegram { token } => cli::cmd_telegram(token),
        Commands::Email {
            imap_host,
            smtp_host,
            username,
            password,
        } => cli::cmd_email(imap_host, smtp_host, username, password),
        Commands::Eval { action } => match action {
            EvalAction::Hanoi {
                disks,
                calibrate,
                samples,
                solve,
                catts,
                k,
                local,
                port,
            } => cli::cmd_eval_hanoi(disks, calibrate, samples, solve, catts, k, local, port),
            EvalAction::Haystack {
                facts,
                length,
                aggregate,
                local,
                port,
            } => cli::cmd_eval_haystack(facts, length, aggregate, local, port),
            EvalAction::Learn {
                family,
                tasks,
                depth,
                local,
                port,
            } => cli::cmd_eval_learn(family, tasks, depth, local, port),
            EvalAction::Sprint {
                corpus_size,
                questions,
                local,
                port,
            } => cli::cmd_eval_sprint(corpus_size, questions, local, port),
            EvalAction::Report => cli::cmd_eval_report(),
        },
        Commands::Sessions { action } => match action {
            SessionsAction::List => sessions_cmd::cmd_sessions_list(),
            SessionsAction::Resume { id, local } => repl::cmd_agent(None, id, local, None),
            SessionsAction::New { name, local } => {
                let key = sessions_cmd::make_session_key(name.as_deref());
                repl::cmd_agent(None, key, local, None)
            }
            SessionsAction::Export { key, format } => sessions_cmd::cmd_sessions_export(&key, &format),
            SessionsAction::Purge { older_than } => sessions_cmd::cmd_sessions_purge(&older_than),
            SessionsAction::Archive => sessions_cmd::cmd_sessions_archive(),
            SessionsAction::Nuke { force } => sessions_cmd::cmd_sessions_nuke(force),
            SessionsAction::Index => {
                let sessions_dir = dirs::home_dir().unwrap().join(".nanobot/sessions");
                let workspace = crate::utils::helpers::get_workspace_path(None);
                let memory_sessions_dir = workspace.join("memory").join("sessions");
                let (indexed, skipped, errors) =
                    agent::session_indexer::index_sessions(&sessions_dir, &memory_sessions_dir);
                println!(
                    "Indexed {} sessions ({} skipped, {} errors)",
                    indexed, skipped, errors
                );
            }
        },
        #[cfg(feature = "voice")]
        Commands::Realtime {
            engine,
            voice,
            session,
            local,
        } => cli::cmd_realtime(engine, voice, session, local),
        #[cfg(feature = "voice")]
        Commands::RealtimeServer {
            port,
            engine,
            voice,
            host,
        } => cli::cmd_realtime_server(port, engine, voice, host),
        #[cfg(feature = "voice")]
        Commands::Voice { action } => match action {
            VoiceAction::List { engine } => cli::cmd_voice_list(engine),
            VoiceAction::Clone { name, audio, transcript } => {
                cli::cmd_voice_clone(name, audio, transcript)
            }
            VoiceAction::Config => cli::cmd_voice_config(),
        },
    }
}

// ============================================================================
// Tests
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
/// Downloads the model on first run, spawns a CPU-only llama-server on port 8090+,
/// and stores the process handle and port. Gracefully degrades if anything fails.
async fn start_compaction_if_available(
    compaction_process: &mut Option<std::process::Child>,
    compaction_port: &mut Option<String>,
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
        "  {}{}Starting{} compaction server on port {} (CPU-only)...",
        tui::BOLD,
        tui::YELLOW,
        tui::RESET,
        port
    );

    match spawn_compaction_server(port, &model_path) {
        Ok(child) => {
            *compaction_process = Some(child);
            if wait_for_server_ready(port, 15, compaction_process).await {
                *compaction_port = Some(port.to_string());
                println!(
                    "  {}{}Compaction server ready{} (Qwen3-0.6B on CPU)",
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

/// Spawn a CPU-only llama-server for context compaction (summarization).
///
/// Uses `--n-gpu-layers 0` so it never competes with the main model for VRAM.
/// Fixed 4K context — summarization doesn't need more.
fn spawn_compaction_server(
    port: u16,
    model_path: &std::path::Path,
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
        .arg("16384")
        .arg("--parallel")
        .arg("1")
        .arg("--n-gpu-layers")
        .arg("0")
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

>>>>>>> Stashed changes
#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn test_cli_parses_tune_command() {
        let cli = Cli::try_parse_from(["nanobot", "tune", "--input", "bench.json"]).unwrap();
        match cli.command {
            Commands::Tune { input, json } => {
                assert_eq!(input, "bench.json");
                assert!(!json);
            }
            other => panic!(
                "unexpected parsed command: {:?}",
                std::mem::discriminant(&other)
            ),
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

        let output = cli::run_tune_from_path(&path, false).expect("expected tuned profile output");
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

        let output = cli::run_tune_from_path(&path, true).expect("expected JSON output");
        let parsed: serde_json::Value = serde_json::from_str(&output).unwrap();
        assert_eq!(parsed["profile"]["id"].as_str(), Some("balanced"));
    }

    #[test]
    fn test_cli_parses_sessions_resume() {
        let cli = Cli::try_parse_from(["nanobot", "sessions", "resume", "20260302_143022_a7f2b1"]).unwrap();
        match cli.command {
            Commands::Sessions { action } => match action {
                SessionsAction::Resume { id, local } => {
                    assert_eq!(id, "20260302_143022_a7f2b1");
                    assert!(!local);
                }
                other => panic!("unexpected action: {:?}", std::mem::discriminant(&other)),
            },
            other => panic!("unexpected command: {:?}", std::mem::discriminant(&other)),
        }
    }

    #[test]
    fn test_cli_parses_sessions_new_with_name() {
        let cli = Cli::try_parse_from(["nanobot", "sessions", "new", "--name", "lab"]).unwrap();
        match cli.command {
            Commands::Sessions { action } => match action {
                SessionsAction::New { name, local } => {
                    assert_eq!(name, Some("lab".to_string()));
                    assert!(!local);
                }
                other => panic!("unexpected action: {:?}", std::mem::discriminant(&other)),
            },
            other => panic!("unexpected command: {:?}", std::mem::discriminant(&other)),
        }
    }

    #[test]
    fn test_cli_parses_sessions_export() {
        let cli =
            Cli::try_parse_from(["nanobot", "sessions", "export", "cli:x", "--format", "jsonl"])
                .unwrap();
        match cli.command {
            Commands::Sessions { action } => match action {
                SessionsAction::Export { key, format } => {
                    assert_eq!(key, "cli:x");
                    assert_eq!(format, "jsonl");
                }
                other => panic!("unexpected action: {:?}", std::mem::discriminant(&other)),
            },
            other => panic!("unexpected command: {:?}", std::mem::discriminant(&other)),
        }
    }

    #[test]
    fn test_cli_parses_sessions_nuke() {
        let cli = Cli::try_parse_from(["nanobot", "sessions", "nuke", "--force"]).unwrap();
        match cli.command {
            Commands::Sessions { action } => match action {
                SessionsAction::Nuke { force } => {
                    assert!(force);
                }
                other => panic!("unexpected action: {:?}", std::mem::discriminant(&other)),
            },
            other => panic!("unexpected command: {:?}", std::mem::discriminant(&other)),
        }
    }
}
