//! nanobot - A lightweight personal AI assistant framework in Rust.
//! Based on nanobot by HKUDS (https://github.com/HKUDS/nanobot).
//!
//! Local LLM support: Use Ctrl+L or /local to toggle between cloud and local mode.

// Interactive/app boundary (error-protocol layer 3 backlog): printing IS the
// product here (REPL/TUI/CLI), and the thin glue code keeps pragmatic
// unwraps on always-set state (rl, runtime, static regexes). The deny regime
// in Cargo.toml stays live for the core; this module lands on the regime
// when its backlog is migrated.
#![allow(
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable,
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::shadow_reuse,
    clippy::shadow_unrelated,
    clippy::shadow_same,
    clippy::format_push_string,
    clippy::string_add
)]
use std::io::IsTerminal;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

use clap::{Parser, Subcommand};

use crate::VERSION;
use crate::{agent, cli, repl, sessions_cmd, tui};

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
        /// Language hint for voice TTS.
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
    /// Manage skills.
    Skills {
        #[command(subcommand)]
        action: SkillsAction,
    },
    /// Manage voice engines and configuration.
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
        /// TTS engine: supertonic or say.
        #[arg(long, default_value = "supertonic")]
        engine: String,
    },
    /// Show voice configuration help.
    Config,
}

/// Commands that hand the terminal to the interactive REPL/TUI. Tracing must
/// go to the log file for these, never to stderr.
fn launches_interactive_ui(cmd: &Commands) -> bool {
    matches!(
        cmd,
        Commands::Agent { message: None, .. }
            | Commands::Sessions {
                action: SessionsAction::Resume { .. } | SessionsAction::New { .. },
            }
    )
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
    /// Import one legacy JSONL transcript, or scan the legacy sessions directory.
    ImportJsonl {
        /// Legacy JSONL file. Omit to scan ~/.nanobot/sessions/*.jsonl.
        path: Option<std::path::PathBuf>,
    },
    /// Delete one concrete SQLite session transactionally.
    Delete {
        /// Session ID to delete (from `sessions list`).
        id: String,
        /// Skip confirmation prompt.
        #[arg(long)]
        force: bool,
    },
    /// Purge session and log files older than the given duration.
    Purge {
        /// Age threshold (e.g. "7d", "24h", "30d").
        #[arg(long)]
        older_than: String,
    },
    /// Show SQLite session, legacy import, log, and metrics disk usage.
    Archive,
    /// Wipe all sessions, logs, and metrics.
    Nuke {
        /// Skip confirmation prompt.
        #[arg(long)]
        force: bool,
    },
}

#[derive(Subcommand)]
enum ChannelsAction {
    /// Show channel status.
    Status,
}

#[derive(Subcommand)]
enum SkillsAction {
    /// Validate all discoverable skills and report issues.
    Validate,
    /// List all skills with their source and description.
    List,
    /// Install skills from a GitHub repository.
    ///
    /// Examples:
    ///   nanobot skills add vercel-labs/agent-skills
    ///   nanobot skills add vercel-labs/agent-skills@vercel-react-best-practices
    Add {
        /// Source in format owner/repo or owner/repo@skill-name
        source: String,
    },
    /// Remove an installed skill by name.
    Remove {
        /// Skill name to remove
        name: String,
    },
    /// Search the skills.sh registry for installable skills.
    Find {
        /// Search query
        query: Vec<String>,
    },
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
        /// Message for agent (required unless --reflect).
        #[arg(short, long)]
        message: Option<String>,
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
        /// Schedule a memory reflection instead of an agent message.
        #[arg(long)]
        reflect: bool,
        /// Schedule dream consolidation instead of an agent message.
        #[arg(long, conflicts_with = "reflect")]
        dream: bool,
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

pub fn run() {
    // Clean up any stale child processes from previous crashed runs.
    agent::pid_file::cleanup_stale_pids();

    // Safety net: restore terminal state on panic and kill child processes
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        agent::pid_file::cleanup_stale_pids();
        agent::pid_file::release_agent_singleton();
        // Best-effort: kill any background ffplay from webradio skill.
        let _ = std::process::Command::new("pkill")
            .args(["-f", "ffplay.*-nodisp"])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();
        tui::force_exit_raw_mode();
        print!("\x1b[r"); // reset scroll region
        print!("\x1b[?25h"); // show cursor
        let _ = std::io::Write::flush(&mut std::io::stdout());
        default_hook(info);
    }));

    let cli = Cli::parse();

    // Detect interactive REPL/TUI mode and a TTY. Every command that runs the
    // interactive UI must be listed here: with the terminal owned by the TUI,
    // a stderr tracing layer splatters WARN lines across the transcript and
    // input box (the `sessions resume` log-corruption bug).
    let is_interactive_repl =
        launches_interactive_ui(&cli.command) && std::io::stdout().is_terminal();

    // Always suppress noisy crates regardless of RUST_LOG setting.
    // When RUST_LOG is set (e.g. "debug"), append mandatory filters so html5ever
    // and other spammy crates don't flood the log file.
    //
    // `mdns_sd=off`: the cluster-discovery mDNS daemon logs every DNS packet
    // (`read_others`, `read_header`, `query question: DnsQuestion`, `received N
    // bytes from IP`) at DEBUG — these are never actionable for nanobot and
    // single-handedly produced a 5 GB daily log under RUST_LOG=debug.
    let noisy_crate_filters =
        ",html5ever=error,ort=off,hyper=warn,reqwest=warn,rustyline=warn,mdns_sd=off";
    let env_filter = match tracing_subscriber::EnvFilter::try_from_default_env() {
        Ok(_) => {
            // RUST_LOG is set — append our mandatory suppressions
            let combined = format!(
                "{}{}",
                std::env::var("RUST_LOG").unwrap_or_default(),
                noisy_crate_filters
            );
            tracing_subscriber::EnvFilter::new(combined)
        }
        Err(_) => tracing_subscriber::EnvFilter::new(format!("warn{}", noisy_crate_filters)),
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
            continue_session,
            resume,
        } => repl::cmd_agent(message, session, local, lang, resume, continue_session),
        Commands::Gateway { port, verbose } => cli::cmd_gateway(port, verbose),
        Commands::Status => cli::cmd_status(),
        Commands::Tune { input, json } => cli::cmd_tune(input, json),
        Commands::Channels { action } => match action {
            ChannelsAction::Status => cli::cmd_channels_status(),
        },
        Commands::Skills { action } => {
            let workspace = crate::utils::helpers::get_workspace_path(None);
            let loader = agent::skills::SkillsLoader::new(&workspace, None);
            match action {
                SkillsAction::List => {
                    let skills = loader.list_skills(false);
                    if skills.is_empty() {
                        println!("No skills found.");
                    } else {
                        println!("{} skill(s) found:\n", skills.len());
                        for s in &skills {
                            let desc = loader
                                .get_skill_metadata(&s.name)
                                .and_then(|m| m.get("description").cloned())
                                .unwrap_or_else(|| "(no description)".to_string());
                            let version = loader
                                .get_skill_metadata(&s.name)
                                .and_then(|m| m.get("version").cloned())
                                .map(|v| format!(" v{}", v))
                                .unwrap_or_default();
                            println!("  [{}]{} — {} ({})", s.source, version, s.name, desc);
                        }
                    }
                }
                SkillsAction::Validate => {
                    let results = loader.validate_all();
                    if results.is_empty() {
                        println!("No skills found.");
                        return;
                    }
                    let mut any_issue = false;
                    for r in &results {
                        if r.errors.is_empty() && r.warnings.is_empty() {
                            println!("  OK   {}", r.name);
                        } else {
                            any_issue = true;
                            if !r.errors.is_empty() {
                                println!("  FAIL {}", r.name);
                                for e in &r.errors {
                                    println!("       ERROR: {}", e);
                                }
                            }
                            for w in &r.warnings {
                                println!("       WARN:  {}", w);
                            }
                        }
                    }
                    let total = results.len();
                    let ok = results.iter().filter(|r| r.is_valid()).count();
                    println!("\n{}/{} skill(s) valid.", ok, total);
                    if any_issue {
                        std::process::exit(1);
                    }
                }
                SkillsAction::Add { source } => {
                    let rt =
                        tokio::runtime::Runtime::new().expect("Failed to create tokio runtime");
                    match rt.block_on(cli::cmd_skill_add(&workspace, &source)) {
                        Ok(installed) => {
                            for name in &installed {
                                println!("  Installed: {}", name);
                            }
                            println!("\n{} skill(s) installed.", installed.len());
                        }
                        Err(e) => {
                            eprintln!("Error: {}", e);
                            std::process::exit(1);
                        }
                    }
                }
                SkillsAction::Remove { name } => match cli::cmd_skill_remove(&workspace, &name) {
                    Ok(()) => println!("Removed skill: {}", name),
                    Err(e) => {
                        eprintln!("Error: {}", e);
                        std::process::exit(1);
                    }
                },
                SkillsAction::Find { query } => {
                    let query = query.join(" ");
                    let rt =
                        tokio::runtime::Runtime::new().expect("Failed to create tokio runtime");
                    match rt.block_on(cli::cmd_skill_search(&query)) {
                        Ok(hits) if hits.is_empty() => {
                            println!("No skills found for \"{}\".", query);
                        }
                        Ok(hits) => {
                            for h in hits.iter().take(10) {
                                println!("  {:>7} installs  {}@{}", h.installs, h.source, h.skill);
                            }
                            println!("\nInstall with: nanobot skills add <source>@<skill>");
                        }
                        Err(e) => {
                            eprintln!("Error: {}", e);
                            std::process::exit(1);
                        }
                    }
                }
            }
        }
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
                reflect,
                dream,
            } => {
                // CLI flag → enum at the boundary (G1): payload kind, not a bool.
                let kind = if reflect {
                    crate::cron::types::PayloadKind::Reflect
                } else if dream {
                    crate::cron::types::PayloadKind::Dream
                } else {
                    crate::cron::types::PayloadKind::AgentTurn
                };
                cli::cmd_cron_add(name, message, every, cron, deliver, to, channel, kind)
            }
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
        Commands::Sessions { action } => match action {
            SessionsAction::List => tokio::runtime::Runtime::new()
                .expect("Failed to create tokio runtime")
                .block_on(sessions_cmd::cmd_sessions_list()),
            SessionsAction::Resume { id, local } => repl::cmd_agent(
                None,
                "cli:default".to_string(),
                local,
                None,
                Some(id),
                false,
            ),
            SessionsAction::New { name, local } => {
                let key = sessions_cmd::make_session_key(name.as_deref());
                repl::cmd_agent(None, key, local, None, None, false)
            }
            SessionsAction::Export { key, format } => tokio::runtime::Runtime::new()
                .expect("Failed to create tokio runtime")
                .block_on(sessions_cmd::cmd_sessions_export(&key, &format)),
            SessionsAction::ImportJsonl { path } => tokio::runtime::Runtime::new()
                .expect("Failed to create tokio runtime")
                .block_on(sessions_cmd::cmd_sessions_import(path.as_deref())),
            SessionsAction::Delete { id, force } => tokio::runtime::Runtime::new()
                .expect("Failed to create tokio runtime")
                .block_on(sessions_cmd::cmd_sessions_delete(&id, force)),
            SessionsAction::Purge { older_than } => tokio::runtime::Runtime::new()
                .expect("Failed to create tokio runtime")
                .block_on(sessions_cmd::cmd_sessions_purge(&older_than)),
            SessionsAction::Archive => tokio::runtime::Runtime::new()
                .expect("Failed to create tokio runtime")
                .block_on(sessions_cmd::cmd_sessions_archive()),
            SessionsAction::Nuke { force } => tokio::runtime::Runtime::new()
                .expect("Failed to create tokio runtime")
                .block_on(sessions_cmd::cmd_sessions_nuke(force)),
        },
        #[cfg(feature = "voice")]
        Commands::Voice { action } => match action {
            VoiceAction::List { engine } => cli::cmd_voice_list(engine),
            VoiceAction::Config => cli::cmd_voice_config(),
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
    fn test_launches_interactive_ui_covers_all_repl_entrypoints() {
        // Interactive: the TUI owns the terminal, tracing must go to file.
        let parse = |args: &[&str]| Cli::try_parse_from(args).unwrap().command;
        assert!(launches_interactive_ui(&parse(&["nanobot", "agent"])));
        assert!(launches_interactive_ui(&parse(&[
            "nanobot",
            "sessions",
            "resume",
            "20260704_072151_f06df0"
        ])));
        assert!(launches_interactive_ui(&parse(&[
            "nanobot", "sessions", "new"
        ])));

        // Non-interactive: stderr logging is fine (and wanted).
        assert!(!launches_interactive_ui(&parse(&[
            "nanobot", "agent", "-m", "hi"
        ])));
        assert!(!launches_interactive_ui(&parse(&["nanobot", "status"])));
        assert!(!launches_interactive_ui(&parse(&[
            "nanobot", "sessions", "list"
        ])));
    }

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
        let cli = Cli::try_parse_from(["nanobot", "sessions", "resume", "20260302_143022_a7f2b1"])
            .unwrap();
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
        let cli = Cli::try_parse_from([
            "nanobot", "sessions", "export", "cli:x", "--format", "jsonl",
        ])
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
    fn test_cli_parses_sessions_import_jsonl_with_path() {
        let cli = Cli::try_parse_from([
            "nanobot",
            "sessions",
            "import-jsonl",
            "/tmp/legacy-session.jsonl",
        ])
        .unwrap();
        match cli.command {
            Commands::Sessions { action } => match action {
                SessionsAction::ImportJsonl { path } => {
                    assert_eq!(
                        path,
                        Some(std::path::PathBuf::from("/tmp/legacy-session.jsonl"))
                    );
                }
                other => panic!("unexpected action: {:?}", std::mem::discriminant(&other)),
            },
            other => panic!("unexpected command: {:?}", std::mem::discriminant(&other)),
        }
    }

    #[test]
    fn test_cli_parses_sessions_import_jsonl_without_path() {
        let cli = Cli::try_parse_from(["nanobot", "sessions", "import-jsonl"]).unwrap();
        match cli.command {
            Commands::Sessions { action } => match action {
                SessionsAction::ImportJsonl { path } => assert!(path.is_none()),
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

    #[test]
    fn test_cli_parses_sessions_delete() {
        let cli = Cli::try_parse_from([
            "nanobot",
            "sessions",
            "delete",
            "20260302_143022_a7f2b1",
            "--force",
        ])
        .unwrap();
        match cli.command {
            Commands::Sessions { action } => match action {
                SessionsAction::Delete { id, force } => {
                    assert_eq!(id, "20260302_143022_a7f2b1");
                    assert!(force);
                }
                other => panic!("unexpected action: {:?}", std::mem::discriminant(&other)),
            },
            other => panic!("unexpected command: {:?}", std::mem::discriminant(&other)),
        }
    }
}
