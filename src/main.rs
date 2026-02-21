//! nanobot - A lightweight personal AI assistant framework in Rust.
//! Based on nanobot by HKUDS (https://github.com/HKUDS/nanobot).
//!
//! Local LLM support: Use Ctrl+L or /local to toggle between cloud and local mode.

mod agent;
mod bus;
mod channels;
mod cli;
mod config;
mod cron;
mod errors;
mod heartbeat;
mod lms;
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
        /// Session key to resume (from `sessions list`).
        key: String,
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
    #[cfg(feature = "trace-chrome")]
    let _chrome_guard: tracing_chrome::FlushGuard;

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
            _chrome_guard = guard;

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
            _chrome_guard = guard;

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
            SessionsAction::Resume { key, local } => repl::cmd_agent(None, key, local, None),
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
    }
}

// ============================================================================
// Tests
// ============================================================================

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
        let cli = Cli::try_parse_from(["nanobot", "sessions", "resume", "cli:test"]).unwrap();
        match cli.command {
            Commands::Sessions { action } => match action {
                SessionsAction::Resume { key, local } => {
                    assert_eq!(key, "cli:test");
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
