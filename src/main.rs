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
mod heartbeat;
mod providers;
mod repl;
mod server;
mod session;
mod syntax;
mod tui;
mod utils;
#[cfg(feature = "voice")]
mod voice;
#[cfg(feature = "voice")]
mod voice_pipeline;

use std::sync::atomic::AtomicBool;

use clap::{Parser, Subcommand};

pub(crate) const VERSION: &str = "0.1.0";
pub(crate) const LOGO: &str = "*";

// Global flag for local mode
pub(crate) static LOCAL_MODE: AtomicBool = AtomicBool::new(false);

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
                tracing_subscriber::EnvFilter::new("warn,ort=off,pocket_tts=off,html5ever=error")
            }),
        )
        .init();

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
        Commands::WhatsApp => cli::cmd_whatsapp(),
        Commands::Telegram { token } => cli::cmd_telegram(token),
        Commands::Email {
            imap_host,
            smtp_host,
            username,
            password,
        } => cli::cmd_email(imap_host, smtp_host, username, password),
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
}
