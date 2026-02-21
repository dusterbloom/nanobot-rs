#![allow(dead_code)]
//! Command dispatch and handlers for the REPL.
//!
//! Contains `ReplContext`, `normalize_alias()`, `dispatch()`, and all
//! `cmd_xxx()` command handlers.

use std::io::{self, Write as _};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use rustyline::error::ReadlineError;
use rustyline::ExternalPrinter as _;
use tokio::sync::mpsc;

use crate::agent::agent_loop::{AgentLoop, SharedCoreHandle};
use crate::agent::audit::AuditLog;
use crate::agent::provenance::{ClaimStatus, ClaimVerifier};
use crate::cli;
use crate::config::loader::{load_config, save_config};
use crate::config::schema::{Config, EmailConfig};
use crate::cron::service::CronService;
use crate::server;
use crate::tui;

// ============================================================================
// ReplContext — all mutable state for the REPL command handlers
// ============================================================================

/// Mutable state for the REPL main loop and slash-command handlers.
///
/// Single flat struct (not sub-structs) because commands like `/local` need
/// mutable access to srv, agent_loop, core_handle, and config simultaneously.
pub(crate) struct ReplContext {
    pub config: Config,
    pub core_handle: SharedCoreHandle,
    pub agent_loop: AgentLoop,
    pub session_id: String,
    pub lang: Option<String>,
    pub srv: super::ServerState,
    pub current_model_path: PathBuf,
    pub active_channels: Vec<super::ActiveChannel>,
    pub display_tx: mpsc::UnboundedSender<String>,
    pub display_rx: mpsc::UnboundedReceiver<String>,
    pub cron_service: Arc<CronService>,
    pub email_config: Option<EmailConfig>,
    pub rl: Option<rustyline::DefaultEditor>,
    /// Health watchdog task handle — aborted on mode switch, restarted on `/local`.
    pub watchdog_handle: Option<tokio::task::JoinHandle<()>>,
    /// Sender for auto-restart requests (passed to watchdog).
    pub restart_tx: mpsc::UnboundedSender<crate::server::RestartRequest>,
    /// Receiver for auto-restart requests from the health watchdog.
    pub restart_rx: mpsc::UnboundedReceiver<crate::server::RestartRequest>,
    /// Health probe registry for endpoint liveness (shown in /status).
    pub health_registry: Option<Arc<crate::heartbeat::health::HealthRegistry>>,
    #[cfg(feature = "voice")]
    pub voice_session: Option<crate::voice::VoiceSession>,
}

// ============================================================================
// DRY helpers — replace 3–10x copy-paste patterns
// ============================================================================

impl ReplContext {
    /// Rebuild the agent loop after a server or config change.
    ///
    /// Replaces the 8x copy-pasted `agent_loop = cli::create_agent_loop(...)` pattern.
    pub fn rebuild_agent_loop(&mut self) {
        self.agent_loop = cli::create_agent_loop(
            self.core_handle.clone(),
            &self.config,
            Some(self.cron_service.clone()),
            self.email_config.clone(),
            Some(self.display_tx.clone()),
            self.health_registry.clone(),
        );
    }

    /// (Re)start the health watchdog for all active local servers.
    ///
    /// Aborts any previous watchdog task, collects current server ports, and
    /// spawns a fresh watchdog with auto-repair. Called on REPL init and on `/local` toggle-on.
    /// No-op when using a remote local server (LM Studio) — nothing to watch.
    pub fn restart_watchdog(&mut self) {
        if let Some(handle) = self.watchdog_handle.take() {
            handle.abort();
        }
        // LMS-managed server: nothing to watch locally.
        if !self.config.agents.defaults.local_api_base.is_empty() {
            return;
        }
        let ports = vec![("main".to_string(), self.srv.local_port.clone())];
        self.watchdog_handle = Some(crate::server::start_health_watchdog_with_autorepair(
            ports,
            self.display_tx.clone(),
            self.restart_tx.clone(),
            Arc::clone(&self.core_handle.counters.inference_active),
        ));
    }

    /// Stop the health watchdog (e.g. when switching to cloud mode).
    pub fn stop_watchdog(&mut self) {
        if let Some(handle) = self.watchdog_handle.take() {
            handle.abort();
        }
    }

    /// Check for and handle any pending auto-restart requests from the watchdog.
    ///
    /// Returns true if a restart was performed.
    pub async fn handle_restart_requests(&mut self) -> bool {
        // When using a remote local server (e.g. LM Studio), there are no
        // local server processes to restart — drain and ignore any stale requests.
        if !self.config.agents.defaults.local_api_base.is_empty() {
            while self.restart_rx.try_recv().is_ok() {}
            return false;
        }
        let mut restarted = false;
        while let Ok(req) = self.restart_rx.try_recv() {
            if req.role == "main" {
                let _ = self.display_tx.send(format!(
                    "\x1b[RAW]\n  \x1b[33m\u{25cf}\x1b[0m Auto-restarting main server...\n"
                ));
                self.cmd_restart().await;
                
                // Wait for server to be ready
                for i in 0..10 {
                    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                    if server::check_chat_health(&self.srv.local_port).await {
                        let _ = self.display_tx.send(format!(
                            "\x1b[RAW]\n  \x1b[32m\u{25cf}\x1b[0m Main server \x1b[32mready\x1b[0m\n"
                        ));
                        break;
                    }
                    if i == 9 {
                        let _ = self.display_tx.send(format!(
                            "\x1b[RAW]\n  \x1b[31m\u{25cf}\x1b[0m Main server failed to start\n"
                        ));
                    }
                }
                restarted = true;
            }
        }
        restarted
    }

    /// Drain pending display messages from background channels/subagents.
    ///
    /// Replaces the 3x copy-pasted `while let Ok(line) = display_rx.try_recv()` pattern.
    pub fn drain_display(&mut self) {
        while let Ok(line) = self.display_rx.try_recv() {
            if line.starts_with("\x1b[RAW]") {
                print!("\r{}", &line[6..]);
            } else {
                print!("\r{}", crate::syntax::render_response(&line));
            }
        }
    }

    /// Async readline that drains display messages while waiting for input.
    ///
    /// Uses rustyline's `ExternalPrinter` to safely output subagent results
    /// while `readline()` blocks on a background thread.  This way results
    /// appear immediately instead of waiting until the next user input.
    pub async fn readline_async(&mut self, prompt: &str) -> Result<String, ReadlineError> {
        // Take the editor out of the Option — no placeholder DefaultEditor created,
        // so no second SIGWINCH handler is registered.
        let mut rl = self.rl.take().expect("editor already borrowed");
        let printer_result = rl.create_external_printer();

        let prompt_owned = prompt.to_string();
        let handle = tokio::task::spawn_blocking(move || {
            let result = rl.readline(&prompt_owned);
            (rl, result)
        });
        tokio::pin!(handle);

        let result = if let Ok(mut printer) = printer_result {
            loop {
                tokio::select! {
                    res = &mut handle => {
                        let (rl_back, readline_res) = res.expect("readline task panicked");
                        self.rl = Some(rl_back);
                        break readline_res;
                    }
                    Some(line) = self.display_rx.recv() => {
                        let rendered = if line.starts_with("\x1b[RAW]") {
                            line[6..].to_string()
                        } else {
                            crate::syntax::render_response(&line)
                        };
                        let _ = printer.print(rendered);
                    }
                }
            }
        } else {
            // No external printer available — just wait for readline.
            let (rl_back, readline_res) = handle.await.expect("readline task panicked");
            self.rl = Some(rl_back);
            readline_res
        };

        result
    }

    /// Print the status bar after a response completes.
    ///
    /// Replaces the 3x copy-pasted block that gets subagent count, retains
    /// finished channels, collects channel names, and calls `tui::print_status_bar`.
    pub async fn print_status_bar(&mut self) {
        let sa_count = self.agent_loop.subagent_manager().get_running_count().await;
        self.active_channels.retain(|ch| !ch.handle.is_finished());
        let ch_names: Vec<&str> = self
            .active_channels
            .iter()
            .map(|c| super::short_channel_name(&c.name))
            .collect();
        tui::print_status_bar(&self.core_handle, &ch_names, sa_count);
    }

    /// Rebuild the shared core from current ServerState and then rebuild the agent loop.
    ///
    /// Combines `apply_server_change` + `rebuild_agent_loop` into one call.
    /// Reads `is_local` from the current core to pass through, unless the caller
    /// is about to change modes — in which case use `apply_and_rebuild_with`.
    pub fn apply_and_rebuild(&mut self) {
        let is_local = self.core_handle.swappable().is_local;
        self.apply_and_rebuild_with(is_local);
    }

    /// Like `apply_and_rebuild` but with an explicit `is_local` override.
    /// Use when toggling between local and cloud mode.
    pub fn apply_and_rebuild_with(&mut self, is_local: bool) {
        super::apply_server_change(
            &self.srv,
            &self.current_model_path,
            &self.core_handle,
            &self.config,
            is_local,
        );
        self.rebuild_agent_loop();
    }
}

// ============================================================================
// Alias resolution
// ============================================================================

/// Normalize command aliases to their canonical form.
pub(crate) fn normalize_alias(cmd: &str) -> &str {
    match cmd {
        "/l" => "/local",
        "/m" => "/model",
        "/t" | "/thinking" => "/think",
        "/nt" => "/nothink",
        "/v" => "/voice",
        "/ss" => "/sessions",
        "/wa" => "/whatsapp",
        "/tg" => "/telegram",
        "/p" | "/prov" => "/provenance",
        "/h" | "/?" => "/help",
        "/a" => "/agents",
        "/s" => "/status",
        "/rd" => "/restart",
        "/ctx-info" => "/context",
        other => other,
    }
}

// ============================================================================
// Dispatch — routes slash commands to handlers
// ============================================================================

impl ReplContext {
    /// Dispatch a slash command. Returns `true` if the input was a recognized command.
    pub async fn dispatch(&mut self, input: &str) -> bool {
        let (cmd, arg) = input
            .split_once(' ')
            .map(|(c, a)| (c, a.trim()))
            .unwrap_or((input, ""));
        let cmd = normalize_alias(cmd);
        match cmd {
            "/help" => {
                super::print_help();
            }
            "/think" => {
                self.cmd_think(arg);
            }
            "/nothink" => {
                self.cmd_nothink();
            }
            "/status" => {
                self.cmd_status().await;
            }
            "/context" => {
                self.cmd_context();
            }
            "/memory" => {
                self.cmd_memory();
            }
            "/agents" => {
                self.cmd_agents().await;
            }
            "/audit" => {
                self.cmd_audit();
            }
            "/verify" => {
                self.cmd_verify().await;
            }
            "/kill" => {
                self.cmd_kill(arg).await;
            }
            "/stop" => {
                self.cmd_stop().await;
            }

            "/replay" => {
                self.cmd_replay(arg).await;
            }
            "/long" => {
                self.cmd_long(arg);
            }
            "/provenance" => {
                self.cmd_provenance();
            }
            "/restart" => {
                self.cmd_restart().await;
            }
            "/ctx" => {
                self.cmd_ctx(arg).await;
            }
            "/model" => {
                self.cmd_model().await;
            }
            "/local" => {
                self.cmd_local().await;
            }
            "/whatsapp" => {
                self.cmd_whatsapp();
            }
            "/telegram" => {
                self.cmd_telegram();
            }
            "/email" => {
                self.cmd_email();
            }
            "/trio" => {
                self.cmd_trio(arg).await;
            }
            "/sessions" => {
                self.cmd_sessions(arg);
            }
            #[cfg(feature = "voice")]
            "/voice" => {
                self.cmd_voice().await;
            }
            _ => {
                return false;
            }
        }
        true
    }
}

// ============================================================================
// Read-only commands (Phase 2)
// ============================================================================

impl ReplContext {
    /// /status — show current mode, model, and channel info.
    async fn cmd_status(&mut self) {
        let core = self.core_handle.swappable();
        let counters = &self.core_handle.counters;
        let is_local = core.is_local;
        let model_name = &core.model;
        let mode_label = if is_local { "local" } else { "cloud" };
        let lane_label = if is_local {
            if self.config.trio.enabled {
                "trio"
            } else {
                "legacy"
            }
        } else {
            "cloud"
        };

        println!();
        println!(
            "  {}MODE{}      {} ({}, {})",
            tui::BOLD,
            tui::RESET,
            mode_label,
            lane_label,
            model_name
        );

        let thinking = counters.thinking_budget.load(Ordering::Relaxed);
        if thinking > 0 {
            println!(
                "  {}THINKING{}  {}\u{1f9e0}{} enabled (budget: {} tokens)",
                tui::BOLD,
                tui::RESET,
                tui::GREY,
                tui::RESET,
                thinking
            );
        }

        let used = counters.last_context_used.load(Ordering::Relaxed) as usize;
        let max = counters.last_context_max.load(Ordering::Relaxed) as usize;
        let pct = if max > 0 { (used * 100) / max } else { 0 };
        let ctx_color = match pct {
            0..=49 => tui::GREEN,
            50..=79 => tui::YELLOW,
            _ => tui::RED,
        };
        println!(
            "  {}CONTEXT{}   {:>6} / {:>6} tokens ({}{}{}%{})",
            tui::BOLD,
            tui::RESET,
            tui::format_thousands(used),
            tui::format_thousands(max),
            ctx_color,
            tui::BOLD,
            pct,
            tui::RESET
        );

        let obs_count = {
            let obs = crate::agent::observer::ObservationStore::new(&core.workspace);
            obs.count()
        };
        println!(
            "  {}MEMORY{}    {} ({} observations)",
            tui::BOLD,
            tui::RESET,
            if core.memory_enabled {
                "enabled"
            } else {
                "disabled"
            },
            obs_count
        );

        let agent_count = self.agent_loop.subagent_manager().get_running_count().await;
        println!(
            "  {}AGENTS{}    {} running",
            tui::BOLD,
            tui::RESET,
            agent_count
        );

        self.active_channels.retain(|ch| !ch.handle.is_finished());
        if !self.active_channels.is_empty() {
            let ch_names: Vec<&str> = self
                .active_channels
                .iter()
                .map(|c| super::short_channel_name(&c.name))
                .collect();
            println!(
                "  {}CHANNELS{}  {}",
                tui::BOLD,
                tui::RESET,
                ch_names.join(" ")
            );
        }

        let turn = counters.learning_turn_counter.load(Ordering::Relaxed);
        println!("  {}TURN{}      {}", tui::BOLD, tui::RESET, turn);

        // Token telemetry.
        let est_prompt = counters
            .last_estimated_prompt_tokens
            .load(Ordering::Relaxed);
        let act_prompt = counters.last_actual_prompt_tokens.load(Ordering::Relaxed);
        let act_completion = counters
            .last_actual_completion_tokens
            .load(Ordering::Relaxed);
        if act_prompt > 0 || est_prompt > 0 {
            let drift = if act_prompt > 0 && est_prompt > 0 {
                let diff = (act_prompt as f64 - est_prompt as f64) / act_prompt as f64 * 100.0;
                format!(" (drift: {:+.0}%)", diff)
            } else {
                String::new()
            };
            println!(
                "  {}TOKENS{}    est:{} actual:{}+{}{}",
                tui::BOLD,
                tui::RESET,
                tui::format_thousands(est_prompt as usize),
                tui::format_thousands(act_prompt as usize),
                tui::format_thousands(act_completion as usize),
                drift,
            );
        }

        let long_remaining = counters.long_mode_turns.load(Ordering::Relaxed);
        if long_remaining > 0 {
            println!(
                "  {}LONG{}      {} turns remaining",
                tui::BOLD,
                tui::RESET,
                long_remaining
            );
        }

        // Server health display (local mode or when delegation server is active).
        {
            let mut servers: Vec<String> = Vec::new();
            let remote_base = &self.config.agents.defaults.local_api_base;
            let has_remote_local = !remote_base.is_empty();

            if is_local {
                if has_remote_local {
                    // Remote local server (e.g. LM Studio): check the remote URL.
                    let health = crate::server::check_health(remote_base).await;
                    let (color, label) = if health {
                        (tui::GREEN, "healthy")
                    } else {
                        (tui::RED, "DOWN")
                    };
                    servers.push(format!(
                        "LM Studio ({}{}{}{})",
                        color, tui::BOLD, label, tui::RESET
                    ));
                } else {
                    let main_health = crate::server::check_health(&format!(
                        "http://localhost:{}/v1",
                        self.srv.local_port
                    ))
                    .await;
                    let (color, label) = if main_health {
                        (tui::GREEN, "healthy")
                    } else {
                        (tui::RED, "DOWN")
                    };
                    servers.push(format!(
                        "main:{} ({}{}{}{})",
                        self.srv.local_port,
                        color,
                        tui::BOLD,
                        label,
                        tui::RESET
                    ));
                }
            }


            if !servers.is_empty() {
                println!(
                    "  {}SERVERS{}   {}",
                    tui::BOLD,
                    tui::RESET,
                    servers.join("  ")
                );
            }
        }

        // Health probes
        if let Some(ref registry) = self.health_registry {
            let states = registry.all_states();
            if !states.is_empty() {
                let probe_labels: Vec<String> = states.iter().map(|s| {
                    use crate::heartbeat::health::ProbeStatus;
                    let (indicator, label) = match s.status {
                        ProbeStatus::Healthy => {
                            let ms = s.last_result.as_ref()
                                .map(|r| format!(" ({}ms)", r.latency_ms))
                                .unwrap_or_default();
                            (format!("{}{}●{}", tui::GREEN, tui::BOLD, tui::RESET), format!("{}{}", s.name, ms))
                        }
                        ProbeStatus::Degraded => {
                            let ago = s.last_healthy.map(|t| {
                                let secs = t.elapsed().as_secs();
                                if secs < 60 { format!(" ({}s ago)", secs) }
                                else { format!(" ({}m ago)", secs / 60) }
                            }).unwrap_or_default();
                            (format!("{}{}●{}", tui::RED, tui::BOLD, tui::RESET), format!("{}: DOWN{}", s.name, ago))
                        }
                        ProbeStatus::Unknown => {
                            (format!("{}{}●{}", tui::YELLOW, tui::BOLD, tui::RESET), format!("{}: pending", s.name))
                        }
                    };
                    format!("{} {}", indicator, label)
                }).collect();
                println!(
                    "  {}HEALTH{}    {}",
                    tui::BOLD, tui::RESET,
                    probe_labels.join("  ")
                );
            }
        }

        // TRIO section — only shown when trio mode is active.
        if self.config.trio.enabled {
            let router_health = if let Some(ref hr) = self.health_registry {
                if hr.is_healthy("trio_router") { "healthy" } else { "degraded" }
            } else {
                "n/a"
            };
            let specialist_health = if let Some(ref hr) = self.health_registry {
                if hr.is_healthy("trio_specialist") { "healthy" } else { "degraded" }
            } else {
                "n/a"
            };
            let last_action = counters
                .trio_metrics
                .router_action
                .lock()
                .unwrap()
                .clone()
                .unwrap_or_else(|| "none".to_string());
            let preflight = counters
                .trio_metrics
                .router_preflight_fired
                .load(Ordering::Relaxed);
            let specialist_dispatched = counters
                .trio_metrics
                .specialist_dispatched
                .load(Ordering::Relaxed);
            println!(
                "  {}TRIO{}      router={} specialist={} last_action={} preflight={} dispatched={}",
                tui::BOLD,
                tui::RESET,
                router_health,
                specialist_health,
                last_action,
                preflight,
                specialist_dispatched,
            );
        }

        println!();
    }

    /// /context — show context breakdown (tokens, messages, memory).
    fn cmd_context(&self) {
        let core = self.core_handle.swappable();
        let counters = &self.core_handle.counters;
        let used = counters.last_context_used.load(Ordering::Relaxed) as usize;
        let max = counters.last_context_max.load(Ordering::Relaxed) as usize;
        let msg_count = counters.last_message_count.load(Ordering::Relaxed) as usize;
        let wm_tokens = counters.last_working_memory_tokens.load(Ordering::Relaxed) as usize;
        let turn = counters.learning_turn_counter.load(Ordering::Relaxed);
        let pct = if max > 0 {
            (used as f64 / max as f64) * 100.0
        } else {
            0.0
        };

        // Estimate system prompt size from an empty message build.
        let system_prompt = core.context.build_messages(
            &[],
            "",
            None,
            None,
            Some("cli"),
            Some("repl"),
            false,
            None,
        );
        let system_tokens = if let Some(sys) = system_prompt.first() {
            crate::agent::token_budget::TokenBudget::estimate_message_tokens_pub(sys)
        } else {
            0
        };

        println!();
        println!("  {}Context Breakdown{}", tui::BOLD, tui::RESET);
        // System prompt component breakdown.
        let identity_tokens = {
            let identity = core.context.build_system_prompt(None, None);
            crate::agent::token_budget::TokenBudget::estimate_str_tokens(&identity)
        };
        let bootstrap_budget = core.context.bootstrap_budget;
        let ltm_budget = core.context.long_term_memory_budget;

        println!(
            "  {}System prompt:    {} {:>6} tokens{}",
            tui::DIM,
            tui::RESET,
            tui::format_thousands(system_tokens),
            tui::RESET
        );
        println!(
            "  {}  identity:       {} {:>6}{}",
            tui::DIM,
            tui::RESET,
            tui::format_thousands(identity_tokens),
            tui::RESET
        );
        println!(
            "  {}  bootstrap cap:  {} {:>6}{}",
            tui::DIM,
            tui::RESET,
            tui::format_thousands(bootstrap_budget),
            tui::RESET
        );
        println!(
            "  {}  memory cap:     {} {:>6}{}",
            tui::DIM,
            tui::RESET,
            tui::format_thousands(ltm_budget),
            tui::RESET
        );
        println!(
            "  {}History:          {} {:>6} messages{}",
            tui::DIM,
            tui::RESET,
            msg_count,
            tui::RESET
        );
        println!(
            "  {}Working memory:   {} {:>6} tokens{}",
            tui::DIM,
            tui::RESET,
            tui::format_thousands(wm_tokens),
            tui::RESET
        );
        println!(
            "  {}Turn:             {} {:>6}{}",
            tui::DIM,
            tui::RESET,
            turn,
            tui::RESET
        );

        // Token accuracy comparison.
        let counters = &self.core_handle.counters;
        let est = counters
            .last_estimated_prompt_tokens
            .load(Ordering::Relaxed);
        let act = counters.last_actual_prompt_tokens.load(Ordering::Relaxed);
        if act > 0 && est > 0 {
            let drift_pct = (act as f64 - est as f64) / act as f64 * 100.0;
            println!(
                "  {}Estimation drift: {} {:>+5.1}%{}",
                tui::DIM,
                tui::RESET,
                drift_pct,
                tui::RESET
            );
        }

        println!("  {}─────────────────────────────{}", tui::DIM, tui::RESET);
        println!(
            "  {}Total:            {} {:>6} / {} tokens ({:.1}%){}",
            tui::DIM,
            tui::RESET,
            tui::format_thousands(used),
            tui::format_thousands(max),
            pct,
            tui::RESET
        );
        println!();
    }

    /// /memory — show working memory for current session.
    fn cmd_memory(&self) {
        let core = self.core_handle.swappable();
        if !core.memory_enabled {
            println!("\n  Memory system is disabled.\n");
        } else {
            let wm = core
                .working_memory
                .get_context(&self.session_id, usize::MAX);
            if wm.is_empty() {
                println!("\n  No working memory for this session.\n");
            } else {
                println!(
                    "\n  {}Working Memory (session: {}){}\n",
                    tui::BOLD,
                    self.session_id,
                    tui::RESET
                );
                for line in wm.lines() {
                    println!("  {}{}{}", tui::DIM, line, tui::RESET);
                }
                let tokens = crate::agent::token_budget::TokenBudget::estimate_str_tokens(&wm);
                println!(
                    "\n  {}({} tokens){}\n",
                    tui::DIM,
                    tui::format_thousands(tokens),
                    tui::RESET
                );
            }
            // Also show learning context if available.
            let learning = core.learning.get_learning_context();
            if !learning.is_empty() {
                println!("  {}Tool Patterns{}\n", tui::BOLD, tui::RESET);
                for line in learning.lines() {
                    println!("  {}{}{}", tui::DIM, line, tui::RESET);
                }
                println!();
            }
        }
    }

    /// /agents — list running subagents.
    async fn cmd_agents(&self) {
        let agents = self.agent_loop.subagent_manager().list_running().await;
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
    }

    /// /audit — display audit log for current session.
    fn cmd_audit(&self) {
        let core = self.core_handle.swappable();
        if !core.provenance_config.enabled {
            println!("\n  Provenance is not enabled. Set provenance.enabled = true in config.\n");
        } else {
            let audit = AuditLog::new(&core.workspace, &self.session_id);
            let entries = audit.get_entries();
            if entries.is_empty() {
                println!("\n  No audit entries for this session.\n");
            } else {
                println!("\n  Audit log ({} entries):\n", entries.len());
                println!(
                    "  {:<4} {:<14} {:<12} {:<6} {:<8} {}",
                    "SEQ", "TOOL", "EXECUTOR", "OK", "MS", "RESULT (preview)"
                );
                for e in &entries {
                    let preview: String = e.result_data.chars().take(40).collect();
                    let preview = preview.replace('\n', " ");
                    println!(
                        "  {:<4} {:<14} {:<12} {:<6} {:<8} {}",
                        e.seq,
                        &e.tool_name[..e.tool_name.len().min(14)],
                        &e.executor[..e.executor.len().min(12)],
                        if e.result_ok { "yes" } else { "NO" },
                        e.duration_ms,
                        preview,
                    );
                }
                match audit.verify_chain() {
                    Ok(n) => println!(
                        "\n  \x1b[32m\u{2713}\x1b[0m Hash chain valid ({} entries)",
                        n
                    ),
                    Err(e) => println!("\n  \x1b[31m\u{2717}\x1b[0m Hash chain BROKEN: {}", e),
                }
                println!();
            }
        }
    }

    /// /verify — re-run claim verification on last response.
    async fn cmd_verify(&self) {
        let core = self.core_handle.swappable();
        if !core.provenance_config.enabled {
            println!("\n  Provenance is not enabled. Set provenance.enabled = true in config.\n");
        } else {
            let audit = AuditLog::new(&core.workspace, &self.session_id);
            let entries = audit.get_entries();
            if entries.is_empty() {
                println!("\n  No audit entries to verify against.\n");
            } else {
                // Get last assistant response from session history.
                let history = core.sessions.get_history(&self.session_id, 10, 0).await;
                let last_response = history
                    .iter()
                    .rev()
                    .find(|m| m.get("role").and_then(|r| r.as_str()) == Some("assistant"))
                    .and_then(|m| m.get("content").and_then(|c| c.as_str()));
                match last_response {
                    Some(text) => {
                        let verifier = ClaimVerifier::new(&entries);
                        let claims = verifier.verify(text);
                        if claims.is_empty() {
                            println!("\n  No verifiable claims found in last response.\n");
                        } else {
                            println!("\n  Claim verification ({} claims):\n", claims.len());
                            for c in &claims {
                                let (marker, color) = match c.status {
                                    ClaimStatus::Observed => ("\u{2713}", "\x1b[32m"),
                                    ClaimStatus::Derived => ("~", "\x1b[34m"),
                                    ClaimStatus::Claimed => ("\u{26a0}", "\x1b[33m"),
                                    ClaimStatus::Recalled => ("\u{25c7}", "\x1b[2m"),
                                };
                                let preview: String = c.text.chars().take(60).collect();
                                println!(
                                    "  {}{}\x1b[0m [{}] {}",
                                    color, marker, c.claim_type, preview
                                );
                            }
                            let summary = verifier.unverified_summary(&claims);
                            if !summary.is_empty() {
                                println!("\n  \x1b[33m{}\x1b[0m", summary);
                            }
                            println!();
                        }
                    }
                    None => println!("\n  No assistant response found in session history.\n"),
                }
            }
        }
    }
}

// ============================================================================
// Simple mutation commands (Phase 3)
// ============================================================================

impl ReplContext {
    /// /think, /t — toggle extended thinking / reasoning mode.
    /// /think <budget> — enable with specific token budget (e.g. /think 16000).
    fn cmd_think(&self, arg: &str) {
        let counters = &self.core_handle.counters;
        let core = self.core_handle.swappable();
        let default_budget = (core.max_tokens / 2).clamp(1024, 32000);

        if !arg.is_empty() {
            let mode = arg.to_ascii_lowercase();
            match mode.as_str() {
                "on" | "enable" | "enabled" | "true" => {
                    counters
                        .thinking_budget
                        .store(default_budget, Ordering::Relaxed);
                    println!(
                        "\n  \x1b[90m\u{1f9e0}\x1b[0m Thinking \x1b[32menabled\x1b[0m — budget: {} tokens\n",
                        default_budget
                    );
                    return;
                }
                "off" | "disable" | "disabled" | "false" => {
                    counters.thinking_budget.store(0, Ordering::Relaxed);
                    println!("\n  Thinking \x1b[33mdisabled\x1b[0m\n");
                    return;
                }
                _ => {}
            }

            // Parse explicit numeric budget
            match arg.parse::<u32>() {
                Ok(budget) if budget == 0 => {
                    counters.thinking_budget.store(0, Ordering::Relaxed);
                    println!("\n  Thinking \x1b[33mdisabled\x1b[0m\n");
                }
                Ok(budget) => {
                    let clamped = budget.clamp(1024, 128000);
                    counters.thinking_budget.store(clamped, Ordering::Relaxed);
                    println!("\n  \x1b[90m\u{1f9e0}\x1b[0m Thinking \x1b[32menabled\x1b[0m — budget: {} tokens\n", clamped);
                }
                Err(_) => {
                    println!(
                        "\n  Usage: /think [on|off|budget]\n  Examples: /think, /thinking off, /think 16000, /think 0\n"
                    );
                }
            }
        } else {
            // Toggle: off → default budget, on → off
            let was_on = counters.thinking_budget.load(Ordering::Relaxed) > 0;
            if was_on {
                counters.thinking_budget.store(0, Ordering::Relaxed);
                println!("\n  Thinking \x1b[33mdisabled\x1b[0m\n");
            } else {
                counters
                    .thinking_budget
                    .store(default_budget, Ordering::Relaxed);
                println!("\n  \x1b[90m\u{1f9e0}\x1b[0m Thinking \x1b[32menabled\x1b[0m — budget: {} tokens\n", default_budget);
            }
        }
    }

    /// /nothink, /nt — suppress thinking tokens from output (and TTS).
    /// Sets thinking budget to 0 and enables suppress_thinking_in_tts.
    fn cmd_nothink(&self) {
        let counters = &self.core_handle.counters;
        let was_suppressed = counters.suppress_thinking_in_tts.load(Ordering::Relaxed);
        if was_suppressed {
            // Toggle off — re-enable thinking display (but thinking budget stays 0)
            counters
                .suppress_thinking_in_tts
                .store(false, Ordering::Relaxed);
            println!(
                "\n  Thinking display \x1b[32mrestored\x1b[0m (use /think to re-enable thinking)\n"
            );
        } else {
            counters.thinking_budget.store(0, Ordering::Relaxed);
            counters
                .suppress_thinking_in_tts
                .store(true, Ordering::Relaxed);
            println!("\n  Thinking \x1b[33msuppressed\x1b[0m — no thinking tokens sent to output or TTS\n");
        }
    }

    /// /long [N] — boost max_tokens to 8192 for the next N turns (default 3).
    /// /long 0 resets to normal adaptive mode.
    fn cmd_long(&self, arg: &str) {
        let counters = &self.core_handle.counters;
        if !arg.is_empty() {
            match arg.parse::<u32>() {
                Ok(0) => {
                    counters
                        .long_mode_turns
                        .store(0, std::sync::atomic::Ordering::Relaxed);
                    println!("\n  Long mode \x1b[33mdisabled\x1b[0m — back to adaptive.\n");
                }
                Ok(n) => {
                    let clamped = n.min(20);
                    counters
                        .long_mode_turns
                        .store(clamped, std::sync::atomic::Ordering::Relaxed);
                    println!(
                        "\n  Long mode \x1b[32menabled\x1b[0m for {} turn{} (max_tokens=8192).\n",
                        clamped,
                        if clamped > 1 { "s" } else { "" }
                    );
                }
                Err(_) => {
                    println!("\n  Usage: /long [turns]  (default: 3, 0 to disable)\n");
                }
            }
        } else {
            counters
                .long_mode_turns
                .store(3, std::sync::atomic::Ordering::Relaxed);
            println!("\n  Long mode \x1b[32menabled\x1b[0m for 3 turns (max_tokens=8192).\n");
        }
    }

    /// /kill <id> — cancel a background subagent.
    async fn cmd_kill(&self, arg: &str) {
        let id = arg.trim();
        if id.is_empty() {
            println!("\n  Usage: /kill <id>\n");
        } else if self.agent_loop.subagent_manager().cancel(id).await {
            println!("\n  Cancelled agent {}.\n", id);
        } else {
            println!("\n  No running agent matching '{}'.\n", id);
        }
    }

    /// /stop — stop all running background channels.
    async fn cmd_stop(&mut self) {
        self.active_channels.retain(|ch| !ch.handle.is_finished());
        if self.active_channels.is_empty() {
            println!("\n  No channels running.\n");
        } else {
            let names: Vec<String> = self
                .active_channels
                .iter()
                .map(|c| c.name.clone())
                .collect();
            println!("\n  Stopping: {}", names.join(", "));
            for ch in &self.active_channels {
                ch.stop.store(true, Ordering::Relaxed);
            }
            tokio::time::sleep(Duration::from_secs(1)).await;
            for ch in &self.active_channels {
                ch.handle.abort();
            }
            self.active_channels.clear();
            println!("  All channels stopped.\n");
        }
    }

    /// /sessions — session management (list, export, purge, archive, index).
    fn cmd_sessions(&self, arg: &str) {
        let (sub, rest) = arg
            .split_once(' ')
            .map(|(s, r)| (s.trim(), r.trim()))
            .unwrap_or((if arg.is_empty() { "list" } else { arg }, ""));

        match sub {
            "list" => {
                crate::sessions_cmd::cmd_sessions_list();
            }
            "export" => {
                if rest.is_empty() {
                    eprintln!("Usage: /sessions export <session-key> [format]");
                    return;
                }
                let (key, fmt) = rest
                    .split_once(' ')
                    .map(|(k, f)| (k.trim(), f.trim()))
                    .unwrap_or((rest, "md"));
                crate::sessions_cmd::cmd_sessions_export(key, fmt);
            }
            "purge" => {
                if rest.is_empty() {
                    eprintln!("Usage: /sessions purge <duration> (e.g. 7d, 24h)");
                    return;
                }
                crate::sessions_cmd::cmd_sessions_purge(rest);
            }
            "archive" => {
                crate::sessions_cmd::cmd_sessions_archive();
            }
            "index" => {
                let sessions_dir = dirs::home_dir().unwrap().join(".nanobot/sessions");
                let core = self.core_handle.swappable();
                let memory_sessions_dir = core.workspace.join("memory").join("sessions");
                let (indexed, skipped, errors) =
                    crate::agent::session_indexer::index_sessions(&sessions_dir, &memory_sessions_dir);
                println!(
                    "Indexed {} sessions ({} skipped, {} errors)",
                    indexed, skipped, errors
                );
            }
            _ => {
                eprintln!("Unknown subcommand '{}'. Available: list, export, purge, archive, index", sub);
            }
        }
    }

    /// /replay — show session message history.
    async fn cmd_replay(&self, arg: &str) {
        let core = self.core_handle.swappable();
        let history = core.sessions.get_history(&self.session_id, 200, 0).await;

        if history.is_empty() {
            println!("\n  No messages in session history.\n");
        } else if arg == "full" {
            // Show full content of all messages.
            println!(
                "\n  {}Session replay ({} messages):{}\n",
                tui::BOLD,
                history.len(),
                tui::RESET
            );
            for (i, msg) in history.iter().enumerate() {
                let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("?");
                let content = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
                let has_tc = msg.get("tool_calls").is_some();
                let tc_id = msg.get("tool_call_id").and_then(|v| v.as_str());
                println!(
                    "  {}[{}]{} {} {}",
                    tui::DIM,
                    i,
                    tui::RESET,
                    role,
                    if has_tc {
                        "[+tool_calls]"
                    } else if tc_id.is_some() {
                        &format!("[tc:{}]", tc_id.unwrap())
                    } else {
                        ""
                    }
                );
                if !content.is_empty() {
                    let preview: String = content.chars().take(200).collect();
                    for line in preview.lines() {
                        println!("    {}{}{}", tui::DIM, line, tui::RESET);
                    }
                    if content.len() > 200 {
                        println!(
                            "    {}...({} total chars){}",
                            tui::DIM,
                            content.len(),
                            tui::RESET
                        );
                    }
                }
            }
            println!();
        } else if let Ok(idx) = arg.parse::<usize>() {
            // Show specific message.
            if idx >= history.len() {
                println!(
                    "\n  Message {} out of range (0..{}).\n",
                    idx,
                    history.len() - 1
                );
            } else {
                let msg = &history[idx];
                println!("\n  {}Message [{}]:{}\n", tui::BOLD, idx, tui::RESET);
                let pretty = serde_json::to_string_pretty(msg).unwrap_or_default();
                for line in pretty.lines() {
                    println!("  {}", line);
                }
                println!();
            }
        } else {
            // Summary mode (default).
            println!(
                "\n  {}Session replay ({} messages):{}\n",
                tui::BOLD,
                history.len(),
                tui::RESET
            );
            for (i, msg) in history.iter().enumerate() {
                let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("?");
                let content = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
                let tokens = crate::agent::token_budget::TokenBudget::estimate_str_tokens(content);
                let has_tc = msg.get("tool_calls").is_some();
                let name = msg.get("name").and_then(|n| n.as_str());
                let extra = if has_tc {
                    " [+tool_calls]"
                } else if let Some(n) = name {
                    &format!(" [{}]", n)
                } else {
                    ""
                };
                let preview: String = content.chars().take(60).collect();
                let preview = preview.replace('\n', " ");
                println!(
                    "  {}[{:>3}]{} {:<10} ({:>5} tok){} {}",
                    tui::DIM,
                    i,
                    tui::RESET,
                    role,
                    tokens,
                    extra,
                    preview
                );
            }
            println!(
                "\n  {}Usage: /replay full | /replay <N>{}\n",
                tui::DIM,
                tui::RESET
            );
        }
    }
}

// ============================================================================
// Server lifecycle commands (Phase 4)
// ============================================================================

impl ReplContext {
    /// /provenance — toggle provenance display on/off.
    fn cmd_provenance(&mut self) {
        let was_enabled = {
            let core = self.core_handle.swappable();
            core.provenance_config.enabled
        };
        // Toggle by rebuilding core with modified config.
        let mut toggled_config = self.config.clone();
        toggled_config.provenance.enabled = !was_enabled;
        let model_name = if !self.config.agents.defaults.lms_main_model.is_empty() {
            Some(self.config.agents.defaults.lms_main_model.as_str())
        } else {
            self.current_model_path.file_name().and_then(|n| n.to_str())
        };
        cli::rebuild_core(
            &self.core_handle,
            &toggled_config,
            &self.srv.local_port,
            model_name,
            None,
            None,
            None,
            self.core_handle.swappable().is_local,
        );
        self.agent_loop = cli::create_agent_loop(
            self.core_handle.clone(),
            &toggled_config,
            Some(self.cron_service.clone()),
            self.email_config.clone(),
            Some(self.display_tx.clone()),
            self.health_registry.clone(),
        );
        if !was_enabled {
            println!(
                "\n  Provenance \x1b[32menabled\x1b[0m (tool calls visible, audit logging on)\n"
            );
        } else {
            println!("\n  Provenance \x1b[33mdisabled\x1b[0m\n");
        }
    }

    /// /restart — restart local servers and reload models.
    pub async fn cmd_restart(&mut self) {
        if self.srv.lms_managed {
            if let Some(ref bin) = self.srv.lms_binary.clone() {
                let lms_port = self.config.agents.defaults.lms_port;

                // Restart LMS server
                print!("  Restarting LM Studio server... ");
                io::stdout().flush().ok();
                crate::lms::server_stop(bin).ok();
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                match crate::lms::server_start(bin, lms_port).await {
                    Ok(()) => println!("{}OK{}", tui::GREEN, tui::RESET),
                    Err(e) => {
                        println!("{}FAILED: {}{}", tui::RED, e, tui::RESET);
                        return;
                    }
                }

                // Reload main model
                let main_model = if !self.config.agents.defaults.lms_main_model.is_empty() {
                    self.config.agents.defaults.lms_main_model.clone()
                } else {
                    self.config.agents.defaults.local_model.clone()
                };
                let main_ctx = Some(self.config.agents.defaults.local_max_context_tokens);
                print!("  Loading {}... ", main_model);
                io::stdout().flush().ok();
                match crate::lms::load_model(lms_port, &main_model, main_ctx).await {
                    Ok(()) => println!("{}OK{}", tui::GREEN, tui::RESET),
                    Err(e) => println!("{}FAILED: {}{}", tui::RED, e, tui::RESET),
                }

                // Reload trio models if enabled
                if self.config.trio.enabled {
                    if !self.config.trio.router_model.is_empty() {
                        print!("  Loading {}... ", self.config.trio.router_model);
                        io::stdout().flush().ok();
                        match crate::lms::load_model(lms_port, &self.config.trio.router_model, Some(self.config.trio.router_ctx_tokens)).await {
                            Ok(()) => println!("{}OK{}", tui::GREEN, tui::RESET),
                            Err(e) => println!("{}FAILED: {}{}", tui::RED, e, tui::RESET),
                        }
                    }
                    if !self.config.trio.specialist_model.is_empty() {
                        print!("  Loading {}... ", self.config.trio.specialist_model);
                        io::stdout().flush().ok();
                        match crate::lms::load_model(lms_port, &self.config.trio.specialist_model, Some(self.config.trio.specialist_ctx_tokens)).await {
                            Ok(()) => println!("{}OK{}", tui::GREEN, tui::RESET),
                            Err(e) => println!("{}FAILED: {}{}", tui::RED, e, tui::RESET),
                        }
                    }
                }

                // Update URL in case host/port changed
                let lms_host = crate::lms::api_host();
                self.config.agents.defaults.local_api_base =
                    format!("http://{}:{}/v1", lms_host, lms_port);
            }
        }

        self.apply_and_rebuild();
        println!(
            "  {}{}Rebuilt{} agent core.",
            tui::BOLD,
            tui::GREEN,
            tui::RESET
        );
    }

    /// /ctx [size] — show or set context size for the main model.
    async fn cmd_ctx(&mut self, arg: &str) {
        if !self.core_handle.swappable().is_local {
            println!(
                "\n  {}Not in local mode — use /local first{}\n",
                tui::DIM,
                tui::RESET
            );
            return;
        }

        let new_ctx: usize = match super::parse_ctx_arg(arg) {
            Ok(Some(n)) => n,
            Ok(None) => {
                let auto = server::compute_optimal_context_size(&self.current_model_path);
                let current = self.config.agents.defaults.local_max_context_tokens;
                println!("\n  Current: {}K", current / 1024);
                println!("  Auto-detected optimal: {}K", auto / 1024);
                if self.config.trio.enabled {
                    let budget = self.compute_current_vram_budget();
                    let total_gb = budget.total_vram_bytes as f64 / 1e9;
                    let limit_gb = budget.effective_limit_bytes as f64 / 1e9;
                    let status = if budget.fits { "OK" } else { "OVER" };
                    println!("  VRAM: {:.1} / {:.1} GB [{}]", total_gb, limit_gb, status);
                }
                println!("\n  Usage: /ctx <size>  e.g. /ctx 32K or /ctx 32768\n");
                return;
            }
            Err(msg) => {
                println!("\n  {}\n", msg);
                println!("  Usage: /ctx <size>  e.g. /ctx 32K or /ctx 32768\n");
                return;
            }
        };

        // Apply context change
        self.config.agents.defaults.local_max_context_tokens = new_ctx;

        // Persist to disk
        let mut disk_cfg = load_config(None);
        disk_cfg.agents.defaults.local_max_context_tokens = new_ctx;
        save_config(&disk_cfg, None);

        // Reload model in LMS with new context
        if self.srv.lms_managed {
            let lms_port = self.config.agents.defaults.lms_port;
            let model_name = if !self.config.agents.defaults.lms_main_model.is_empty() {
                self.config.agents.defaults.lms_main_model.clone()
            } else {
                self.config.agents.defaults.local_model.clone()
            };
            if !model_name.is_empty() {
                print!("  Reloading {} with {}K context... ", model_name, new_ctx / 1024);
                io::stdout().flush().ok();
                match crate::lms::load_model(lms_port, &model_name, Some(new_ctx)).await {
                    Ok(()) => println!("{}OK{}", tui::GREEN, tui::RESET),
                    Err(e) => println!("{}FAILED: {}{}", tui::RED, e, tui::RESET),
                }
            }
        }

        // Warn if VRAM budget exceeded
        if self.config.trio.enabled {
            let budget = self.compute_current_vram_budget();
            if !budget.fits {
                println!(
                    "  \x1b[33mWarning:\x1b[0m Total VRAM ({:.1} GB) exceeds limit ({:.1} GB).",
                    budget.total_vram_bytes as f64 / 1e9,
                    budget.effective_limit_bytes as f64 / 1e9,
                );
                println!("  Reduce context sizes or switch to smaller models.");
            }
        }

        self.apply_and_rebuild();
        println!(
            "\n  Context size set to {}{}K{}.\n",
            tui::BOLD,
            new_ctx / 1024,
            tui::RESET,
        );
    }

    /// /model — select local model from ~/models/.
    async fn cmd_model(&mut self) {
        if !self.core_handle.swappable().is_local {
            println!("\n  /model is only available in local mode. Use /local to switch.\n");
            return;
        }

        // LMS-managed: list from LM Studio's downloaded models via HTTP API
        if self.srv.lms_managed {
            let lms_port = self.config.agents.defaults.lms_port;
            let available = crate::lms::list_available(lms_port).await;
            // Filter out embedding models (they contain "embedding" in the name)
            let llm_models: Vec<&String> = available
                .iter()
                .filter(|m| !m.to_lowercase().contains("embedding"))
                .collect();
            if llm_models.is_empty() {
                println!("\nNo models found in LM Studio.\n");
                return;
            }

            let loaded = crate::lms::list_loaded(lms_port).await;
            let current_model = if !self.config.agents.defaults.lms_main_model.is_empty() {
                &self.config.agents.defaults.lms_main_model
            } else {
                &self.config.agents.defaults.local_model
            };

            println!("\nLM Studio models:");
            for (i, name) in llm_models.iter().enumerate() {
                let is_loaded = loaded.iter().any(|l| l.contains(name.as_str()) || name.contains(l.as_str()));
                let is_active = crate::lms::is_model_available(std::slice::from_ref(*name), current_model);
                let marker = if is_active {
                    " (active)"
                } else if is_loaded {
                    " (loaded)"
                } else {
                    ""
                };
                println!("  [{}] {}{}", i + 1, name, marker);
            }

            let model_prompt = format!("Select model [1-{}] or Enter to cancel: ", llm_models.len());
            let choice = match self.rl.as_mut().unwrap().readline(&model_prompt) {
                Ok(line) => line,
                Err(_) => return,
            };
            let choice = choice.trim();
            if choice.is_empty() {
                return;
            }
            let idx: usize = match choice.parse::<usize>() {
                Ok(n) if n >= 1 && n <= llm_models.len() => n - 1,
                _ => {
                    println!("Invalid selection.\n");
                    return;
                }
            };

            let selected = llm_models[idx].clone();
            println!("\nSelected: {}", selected);

            // Unload the previous main model to free VRAM
            let prev_model = &self.config.agents.defaults.lms_main_model;
            if !prev_model.is_empty() && prev_model != &selected {
                print!("  Unloading {}... ", prev_model);
                io::stdout().flush().ok();
                match crate::lms::unload_model(lms_port, prev_model).await {
                    Ok(()) => println!("{}OK{}", tui::GREEN, tui::RESET),
                    Err(e) => println!("{}warn: {}{}", tui::YELLOW, e, tui::RESET),
                }
            }

            // Load the model in LMS with context length
            let ctx = Some(self.config.agents.defaults.local_max_context_tokens);
            print!("  Loading {}... ", selected);
            io::stdout().flush().ok();
            match crate::lms::load_model(lms_port, &selected, ctx).await {
                Ok(()) => println!("{}OK{}", tui::GREEN, tui::RESET),
                Err(e) => println!("{}FAILED: {}{}", tui::RED, e, tui::RESET),
            }

            // Persist selection and update current_model_path so rebuild_core
            // sends the correct model identifier to LM Studio.
            self.config.agents.defaults.local_model = selected.clone();
            self.config.agents.defaults.lms_main_model = selected.clone();
            self.current_model_path = PathBuf::from(&selected);
            let mut disk_cfg = load_config(None);
            disk_cfg.agents.defaults.local_model = selected.clone();
            disk_cfg.agents.defaults.lms_main_model = selected;
            save_config(&disk_cfg, None);

            self.apply_and_rebuild();

            // Warn if VRAM budget exceeded after model change
            if self.config.trio.enabled {
                let budget = self.compute_current_vram_budget();
                if !budget.fits {
                    println!(
                        "  \x1b[33mWarning:\x1b[0m VRAM usage ({:.1} GB) exceeds limit ({:.1} GB).",
                        budget.total_vram_bytes as f64 / 1e9,
                        budget.effective_limit_bytes as f64 / 1e9,
                    );
                    println!("  Use /trio budget for details.");
                }
            }

            println!("  {}Model switched.{}\n", tui::DIM, tui::RESET);
            return;
        }

        // Fallback: scan ~/models/*.gguf (llama-server or no engine)
        let models = server::list_local_models();
        if models.is_empty() {
            println!("\nNo .gguf models found in ~/models/\n");
            return;
        }

        println!("\nAvailable models:");
        for (i, path) in models.iter().enumerate() {
            let name = path.file_name().unwrap().to_string_lossy();
            let size_mb = std::fs::metadata(path)
                .map(|m| m.len() / 1_048_576)
                .unwrap_or(0);
            let marker = if *path == self.current_model_path {
                " (active)"
            } else {
                ""
            };
            println!("  [{}] {} ({} MB){}", i + 1, name, size_mb, marker);
        }
        let model_prompt = format!("Select model [1-{}] or Enter to cancel: ", models.len());
        let choice = match self.rl.as_mut().unwrap().readline(&model_prompt) {
            Ok(line) => line,
            Err(_) => {
                return;
            }
        };
        let choice = choice.trim();
        if choice.is_empty() {
            return;
        }
        let idx: usize = match choice.parse::<usize>() {
            Ok(n) if n >= 1 && n <= models.len() => n - 1,
            _ => {
                println!("Invalid selection.\n");
                return;
            }
        };

        let selected = &models[idx];
        self.current_model_path = selected.clone();
        let name = selected.file_name().unwrap().to_string_lossy();
        println!("\nSelected: {}", name);

        // Persist the selection to config (load fresh to avoid clobbering
        // fields changed externally, e.g. localApiBase).
        self.config.agents.defaults.local_model = name.to_string();
        let mut disk_cfg = load_config(None);
        disk_cfg.agents.defaults.local_model = name.to_string();
        save_config(&disk_cfg, None);

        // If local mode is active, apply the new model.
        if self.core_handle.swappable().is_local {
            self.apply_and_rebuild();

            // Warn if VRAM budget exceeded after model change
            if self.config.trio.enabled {
                let budget = self.compute_current_vram_budget();
                if !budget.fits {
                    println!(
                        "  \x1b[33mWarning:\x1b[0m VRAM usage ({:.1} GB) exceeds limit ({:.1} GB).",
                        budget.total_vram_bytes as f64 / 1e9,
                        budget.effective_limit_bytes as f64 / 1e9,
                    );
                    println!("  Use /trio budget for details.");
                }
            }

            println!(
                "  {}Model switched — server will load on next request.{}",
                tui::DIM, tui::RESET
            );
        } else {
            println!("Model will be used next time you toggle /local on.\n");
        }
    }

    /// /trio — manage trio mode (router + specialist helpers).
    ///
    /// Subcommands:
    ///   /trio                      — toggle trio on/off
    ///   /trio status               — show current trio config
    ///   /trio budget               — show VRAM budget breakdown
    ///   /trio router               — pick router model from LM Studio
    ///   /trio specialist           — pick specialist model from LM Studio
    ///   /trio router temp 0.3      — set router temperature
    ///   /trio specialist ctx 8K    — set specialist context size
    ///   /trio router nothink       — toggle router no_think
    ///   /trio main nothink         — toggle main no_think
    ///   /trio cap 12               — set VRAM cap (GB)
    async fn cmd_trio(&mut self, arg: &str) {
        let parts: Vec<&str> = arg.split_whitespace().collect();
        match parts.as_slice() {
            ["status" | "s"] => self.cmd_trio_status().await,
            ["router" | "r"] => self.cmd_trio_pick_model("router").await,
            ["specialist" | "spec"] => self.cmd_trio_pick_model("specialist").await,
            ["budget" | "b"] => self.cmd_trio_budget().await,

            // Parameter subcommands
            ["router" | "r", "temp" | "temperature", val] =>
                self.cmd_trio_set_param("router", "temperature", val),
            ["specialist" | "spec", "temp" | "temperature", val] =>
                self.cmd_trio_set_param("specialist", "temperature", val),
            ["router" | "r", "ctx" | "context", val] =>
                self.cmd_trio_set_param("router", "ctx", val),
            ["specialist" | "spec", "ctx" | "context", val] =>
                self.cmd_trio_set_param("specialist", "ctx", val),
            ["router" | "r", "nothink" | "no_think"] =>
                self.cmd_trio_set_param("router", "no_think", "toggle"),
            ["main", "nothink" | "no_think"] =>
                self.cmd_trio_set_param("main", "no_think", "toggle"),
            ["cap" | "vram", val] =>
                self.cmd_trio_set_param("trio", "vram_cap", val),

            [] => self.cmd_trio_toggle().await,
            _ => {
                println!("\n  Usage: /trio [subcommand]");
                println!("    /trio                      Toggle trio on/off");
                println!("    /trio status               Show current trio config");
                println!("    /trio budget               Show VRAM budget breakdown");
                println!("    /trio router               Pick router model");
                println!("    /trio specialist            Pick specialist model");
                println!("    /trio router temp 0.3       Set router temperature");
                println!("    /trio specialist temp 0.7   Set specialist temperature");
                println!("    /trio router ctx 4K         Set router context");
                println!("    /trio specialist ctx 8K     Set specialist context");
                println!("    /trio router nothink        Toggle router no_think");
                println!("    /trio main nothink          Toggle main no_think");
                println!("    /trio cap 12                Set VRAM cap (GB)\n");
            }
        }
    }

    /// Set a trio parameter (temperature, context, no_think, vram_cap).
    fn cmd_trio_set_param(&mut self, role: &str, param: &str, value: &str) {
        match apply_trio_param(&mut self.config, role, param, value) {
            Ok(desc) => {
                self.persist_trio_config();
                self.apply_and_rebuild();
                println!("\n  Set {}.\n", desc);
            }
            Err(msg) => {
                println!("\n  Error: {}\n", msg);
            }
        }
    }

    /// Show VRAM budget breakdown.
    async fn cmd_trio_budget(&self) {
        if !self.core_handle.swappable().is_local {
            println!(
                "\n  {}Not in local mode — use /local first{}\n",
                tui::DIM,
                tui::RESET
            );
            return;
        }
        let budget = self.compute_current_vram_budget();
        println!("{}", format_vram_budget(&budget));
    }

    /// Toggle trio mode on/off.
    async fn cmd_trio_toggle(&mut self) {
        let was_enabled = self.config.trio.enabled;

        if was_enabled {
            trio_disable(&mut self.config);

            // Unload router + specialist from LMS (keep main loaded)
            if self.srv.lms_managed {
                let lms_port = self.config.agents.defaults.lms_port;
                if !self.config.trio.router_model.is_empty() {
                    let _ = crate::lms::unload_model(lms_port, &self.config.trio.router_model).await;
                }
                if !self.config.trio.specialist_model.is_empty() {
                    let _ = crate::lms::unload_model(lms_port, &self.config.trio.specialist_model).await;
                }
            }

            self.persist_trio_config();
            self.apply_and_rebuild();
            println!(
                "\n  Trio \x1b[33mdisabled\x1b[0m — single model (inline) mode.\n"
            );
        } else {
            let needs_warning = trio_enable(&mut self.config);
            if needs_warning {
                println!("\n  \x1b[33mWarning:\x1b[0m Router or specialist model not configured.");
                println!("  Use /trio router and /trio specialist to pick models first.");
                println!("  Or set them in config.json under \"trio\".\n");
            }

            // Auto-compute optimal context sizes to fit VRAM budget
            if self.core_handle.swappable().is_local {
                let budget = self.compute_current_vram_budget();
                if budget.fits {
                    // Apply computed context sizes
                    self.config.agents.defaults.local_max_context_tokens = budget.main_ctx;
                    if budget.router_ctx > 0 {
                        self.config.trio.router_ctx_tokens = budget.router_ctx;
                    }
                    if budget.specialist_ctx > 0 {
                        self.config.trio.specialist_ctx_tokens = budget.specialist_ctx;
                    }
                    let total_gb = budget.total_vram_bytes as f64 / 1e9;
                    let limit_gb = budget.effective_limit_bytes as f64 / 1e9;
                    println!(
                        "  Auto-computed contexts: main={}K router={}K specialist={}K ({:.1}/{:.1} GB)",
                        budget.main_ctx / 1024,
                        budget.router_ctx / 1024,
                        budget.specialist_ctx / 1024,
                        total_gb,
                        limit_gb,
                    );
                } else {
                    println!(
                        "  \x1b[33mWarning:\x1b[0m Models may exceed VRAM ({:.1}/{:.1} GB).",
                        budget.total_vram_bytes as f64 / 1e9,
                        budget.effective_limit_bytes as f64 / 1e9,
                    );
                    println!("  Use /trio budget for details, /trio cap to adjust.");
                }
            }

            // Load router + specialist on LMS if available
            if self.srv.lms_managed {
                let lms_port = self.config.agents.defaults.lms_port;
                if !self.config.trio.router_model.is_empty() {
                    print!(
                        "  Loading {}... ",
                        self.config.trio.router_model
                    );
                    io::stdout().flush().ok();
                    match crate::lms::load_model(lms_port, &self.config.trio.router_model, Some(self.config.trio.router_ctx_tokens)).await {
                        Ok(()) => println!("{}OK{}", tui::GREEN, tui::RESET),
                        Err(e) => println!("{}FAILED: {}{}", tui::RED, e, tui::RESET),
                    }
                }
                if !self.config.trio.specialist_model.is_empty() {
                    print!(
                        "  Loading {}... ",
                        self.config.trio.specialist_model
                    );
                    io::stdout().flush().ok();
                    match crate::lms::load_model(lms_port, &self.config.trio.specialist_model, Some(self.config.trio.specialist_ctx_tokens)).await
                    {
                        Ok(()) => println!("{}OK{}", tui::GREEN, tui::RESET),
                        Err(e) => println!("{}FAILED: {}{}", tui::RED, e, tui::RESET),
                    }
                }
            }

            self.persist_trio_config();
            self.apply_and_rebuild();
            println!(
                "\n  Trio \x1b[32menabled\x1b[0m — main + router + specialist.\n"
            );
        }
    }

    /// Show current trio configuration.
    async fn cmd_trio_status(&self) {
        let trio = &self.config.trio;
        let td = &self.config.tool_delegation;
        let enabled_label = if trio.enabled {
            format!("{}enabled{}", tui::GREEN, tui::RESET)
        } else {
            format!("{}disabled{}", tui::YELLOW, tui::RESET)
        };

        println!();
        println!(
            "  {}TRIO{}       {}",
            tui::BOLD, tui::RESET, enabled_label
        );
        println!(
            "  {}MODE{}       {:?}",
            tui::BOLD, tui::RESET, td.mode
        );

        // Main model
        let main = if !self.config.agents.defaults.lms_main_model.is_empty() {
            self.config.agents.defaults.lms_main_model.clone()
        } else if !self.config.agents.defaults.local_model.is_empty() {
            self.config.agents.defaults.local_model.clone()
        } else {
            "(default)".to_string()
        };
        println!(
            "  {}MAIN{}       {}{}{}",
            tui::BOLD, tui::RESET, tui::DIM, main, tui::RESET
        );

        // Router
        let router = if trio.router_model.is_empty() {
            "\x1b[33m(not set)\x1b[0m".to_string()
        } else {
            format!("{}{}{}", tui::DIM, trio.router_model, tui::RESET)
        };
        println!(
            "  {}ROUTER{}     {}",
            tui::BOLD, tui::RESET, router
        );

        // Specialist
        let specialist = if trio.specialist_model.is_empty() {
            "\x1b[33m(not set)\x1b[0m".to_string()
        } else {
            format!("{}{}{}", tui::DIM, trio.specialist_model, tui::RESET)
        };
        println!(
            "  {}SPECIALIST{} {}",
            tui::BOLD, tui::RESET, specialist
        );

        // Context sizes
        println!(
            "  {}CTX{}        main={}K  router={}K  specialist={}K",
            tui::BOLD,
            tui::RESET,
            self.config.agents.defaults.local_max_context_tokens / 1024,
            trio.router_ctx_tokens / 1024,
            trio.specialist_ctx_tokens / 1024,
        );

        // Loaded models (if LMS managed)
        if self.srv.lms_managed {
            let lms_port = self.config.agents.defaults.lms_port;
            let loaded = crate::lms::list_loaded(lms_port).await;
            if !loaded.is_empty() {
                println!(
                    "  {}LOADED{}     {}{}{}",
                    tui::BOLD,
                    tui::RESET,
                    tui::DIM,
                    loaded.join(", "),
                    tui::RESET
                );
            }
        }

        // VRAM budget summary (local mode only)
        if self.core_handle.swappable().is_local {
            let budget = self.compute_current_vram_budget();
            let total_gb = budget.total_vram_bytes as f64 / 1e9;
            let limit_gb = budget.effective_limit_bytes as f64 / 1e9;
            let status = if budget.fits {
                format!("{}OK{}", tui::GREEN, tui::RESET)
            } else {
                format!("\x1b[31mOVER\x1b[0m")
            };
            println!(
                "  {}VRAM{}       {:.1} / {:.1} GB  [{}]",
                tui::BOLD, tui::RESET, total_gb, limit_gb, status,
            );
        }

        println!();
    }

    /// Pick a model for a trio role (router or specialist) from LM Studio's available models.
    async fn cmd_trio_pick_model(&mut self, role: &str) {
        // Get available models from LMS
        let models: Vec<String> = if self.srv.lms_managed {
            let lms_port = self.config.agents.defaults.lms_port;
            crate::lms::list_available(lms_port)
                .await
                .into_iter()
                .filter(|m| {
                    // Filter out embedding models
                    !m.to_lowercase().contains("embedding")
                })
                .collect()
        } else {
            Vec::new()
        };

        if models.is_empty() {
            println!("\n  No LLM models found in LM Studio.");
            println!(
                "  Set manually in config.json: trio.{}Model\n",
                role
            );
            // Allow manual entry
            let manual_prompt = format!("  Enter {} model ID (or Enter to cancel): ", role);
            let input = match self.rl.as_mut().unwrap().readline(&manual_prompt) {
                Ok(line) => line.trim().to_string(),
                Err(_) => return,
            };
            if input.is_empty() {
                return;
            }
            self.set_trio_role_model(role, &input).await;
            return;
        }

        let current = match role {
            "router" => &self.config.trio.router_model,
            "specialist" => &self.config.trio.specialist_model,
            _ => unreachable!(),
        };

        println!("\n  Available models for {}:", role);
        for (i, model) in models.iter().enumerate() {
            let marker = if crate::lms::is_model_available(std::slice::from_ref(model), current) {
                " (active)"
            } else {
                ""
            };
            println!("  [{}] {}{}", i + 1, model, marker);
        }

        let pick_prompt = format!(
            "  Select {} model [1-{}] or Enter to cancel: ",
            role,
            models.len()
        );
        let choice = match self.rl.as_mut().unwrap().readline(&pick_prompt) {
            Ok(line) => line,
            Err(_) => return,
        };
        let choice = choice.trim();
        if choice.is_empty() {
            return;
        }
        let idx: usize = match choice.parse::<usize>() {
            Ok(n) if n >= 1 && n <= models.len() => n - 1,
            _ => {
                println!("  Invalid selection.\n");
                return;
            }
        };

        let selected = &models[idx];
        self.set_trio_role_model(role, selected).await;
    }

    /// Apply a model selection to a trio role and persist.
    async fn set_trio_role_model(&mut self, role: &str, model: &str) {
        set_trio_role_model_pure(&mut self.config, role, model);

        // Load the model in LMS if trio is active
        if self.config.trio.enabled && self.srv.lms_managed {
            let lms_port = self.config.agents.defaults.lms_port;
            let ctx = match role {
                "router" => Some(self.config.trio.router_ctx_tokens),
                "specialist" => Some(self.config.trio.specialist_ctx_tokens),
                _ => Some(self.config.agents.defaults.local_max_context_tokens),
            };
            print!("  Loading {}... ", model);
            io::stdout().flush().ok();
            match crate::lms::load_model(lms_port, model, ctx).await {
                Ok(()) => println!("{}OK{}", tui::GREEN, tui::RESET),
                Err(e) => println!("{}FAILED: {}{}", tui::RED, e, tui::RESET),
            }
        }

        self.persist_trio_config();
        self.apply_and_rebuild();
        println!(
            "\n  {} {} set to {}{}{}.",
            role.chars().next().unwrap().to_uppercase().collect::<String>()
                + &role[1..],
            tui::BOLD,
            tui::RESET,
            model,
            ""
        );

        // Warn if VRAM budget exceeded after model change
        if self.config.trio.enabled && self.core_handle.swappable().is_local {
            let budget = self.compute_current_vram_budget();
            if !budget.fits {
                println!(
                    "  \x1b[33mWarning:\x1b[0m VRAM usage ({:.1} GB) exceeds limit ({:.1} GB).",
                    budget.total_vram_bytes as f64 / 1e9,
                    budget.effective_limit_bytes as f64 / 1e9,
                );
                println!("  Use /trio budget for details.");
            }
        }
        println!();
    }

    /// Persist trio + tool_delegation config to disk.
    fn persist_trio_config(&self) {
        let mut disk_cfg = load_config(None);
        persist_trio_fields(&self.config, &mut disk_cfg);
        save_config(&disk_cfg, None);
    }

    fn persist_local_config(&self) {
        let mut disk_cfg = load_config(None);
        // Persist local mode settings
        disk_cfg.agents.defaults.local_api_base = self.config.agents.defaults.local_api_base.clone();
        disk_cfg.agents.defaults.skip_jit_gate = self.config.agents.defaults.skip_jit_gate;
        disk_cfg.agents.defaults.lms_port = self.config.agents.defaults.lms_port;
        disk_cfg.agents.defaults.lms_main_model = self.config.agents.defaults.lms_main_model.clone();
        save_config(&disk_cfg, None);
    }

    /// /local — toggle between local and cloud mode.
    async fn cmd_local(&mut self) {
        let currently_local = self.core_handle.swappable().is_local;

        if !currently_local {
            // Try to start LM Studio if no engine is active.
            if self.config.agents.defaults.local_api_base.is_empty()
                && !self.srv.lms_managed
                && self.srv.engine == super::InferenceEngine::None
            {
                let preference = &self.config.agents.defaults.inference_engine;
                if let Some((super::InferenceEngine::Lms, bin)) = super::resolve_inference_engine(preference) {
                    let lms_port = self.config.agents.defaults.lms_port;
                    println!(
                        "\n  {}{}LM Studio{} detected, starting server on port {}...",
                        tui::BOLD, tui::YELLOW, tui::RESET, lms_port
                    );
                    match crate::lms::server_start(&bin, lms_port).await {
                        Ok(()) => {
                            let main_model = if !self.config.agents.defaults.lms_main_model.is_empty() {
                                self.config.agents.defaults.lms_main_model.clone()
                            } else {
                                let mn = self.current_model_path
                                    .file_name()
                                    .and_then(|n| n.to_str())
                                    .unwrap_or(server::DEFAULT_LOCAL_MODEL);
                                cli::strip_gguf_suffix(mn).to_string()
                            };
                            let main_ctx = Some(self.config.agents.defaults.local_max_context_tokens);
                            print!("  Loading {}... ", main_model);
                            io::stdout().flush().ok();
                            match crate::lms::load_model(lms_port, &main_model, main_ctx).await {
                                Ok(()) => println!("{}OK{}", tui::GREEN, tui::RESET),
                                Err(e) => println!("{}FAILED: {}{}", tui::RED, e, tui::RESET),
                            }
                            if self.config.trio.enabled {
                                if !self.config.trio.router_model.is_empty() {
                                    print!("  Loading {}... ", self.config.trio.router_model);
                                    io::stdout().flush().ok();
                                    match crate::lms::load_model(lms_port, &self.config.trio.router_model, Some(self.config.trio.router_ctx_tokens)).await {
                                        Ok(()) => println!("{}OK{}", tui::GREEN, tui::RESET),
                                        Err(e) => println!("{}FAILED: {}{}", tui::RED, e, tui::RESET),
                                    }
                                }
                                if !self.config.trio.specialist_model.is_empty() {
                                    print!("  Loading {}... ", self.config.trio.specialist_model);
                                    io::stdout().flush().ok();
                                    match crate::lms::load_model(lms_port, &self.config.trio.specialist_model, Some(self.config.trio.specialist_ctx_tokens)).await {
                                        Ok(()) => println!("{}OK{}", tui::GREEN, tui::RESET),
                                        Err(e) => println!("{}FAILED: {}{}", tui::RED, e, tui::RESET),
                                    }
                                }
                            }
                            self.srv.lms_managed = true;
                            self.srv.lms_binary = Some(bin);
                            self.srv.engine = super::InferenceEngine::Lms;
                            self.srv.local_port = lms_port.to_string();
                            self.config.agents.defaults.local_api_base =
                                format!("http://{}:{}/v1", crate::lms::api_host(), lms_port);
                            self.config.agents.defaults.skip_jit_gate = true;
                        }
                        Err(e) => {
                            println!(
                                "  {}{}lms server start failed:{} {}",
                                tui::BOLD, tui::YELLOW, tui::RESET, e
                            );
                            println!("  {}Remaining in cloud mode{}\n", tui::DIM, tui::RESET);
                            return;
                        }
                    }
                } else {
                    println!(
                        "\n  {}{}No local inference engine found.{} Install LM Studio (lms CLI).",
                        tui::BOLD, tui::YELLOW, tui::RESET
                    );
                    return;
                }
            }

            // When trio strict mode is on but router model is unavailable,
            // disable strict flags so the single model can handle tools directly.
            if self.config.tool_delegation.strict_no_tools_main
                && self.config.tool_delegation.strict_router_schema
            {
                let router_available = if self.srv.lms_managed {
                    let lms_port = self.config.agents.defaults.lms_port;
                    let available = crate::lms::list_available(lms_port).await;
                    crate::lms::is_model_available(&available, &self.config.trio.router_model)
                } else {
                    !self.config.trio.router_model.is_empty()
                };
                if !router_available {
                    self.config.tool_delegation.strict_no_tools_main = false;
                    self.config.tool_delegation.strict_router_schema = false;
                }
            }

            // Flip to local mode and rebuild.
            self.persist_local_config();
            self.apply_and_rebuild_with(true);
            tui::print_mode_banner(&self.srv.local_port, true);
        } else {
            // Toggle OFF
            if self.srv.lms_managed {
                let lms_port = self.config.agents.defaults.lms_port;
                crate::lms::unload_all(lms_port).await.ok();
                self.srv.lms_managed = false;
                self.srv.lms_binary = None;
                self.config.agents.defaults.local_api_base.clear();
                self.config.agents.defaults.skip_jit_gate = false;
            }
            self.srv.engine = super::InferenceEngine::None;
            self.config.agents.defaults.skip_jit_gate = false;
            self.stop_watchdog();
            self.persist_local_config();
            self.apply_and_rebuild_with(false);
            tui::print_mode_banner(&self.srv.local_port, false);
        }
    }
}

// ============================================================================
// Channel + voice commands (Phase 5)
// ============================================================================

impl ReplContext {
    /// /whatsapp — start WhatsApp channel in background.
    fn cmd_whatsapp(&mut self) {
        self.active_channels.retain(|ch| !ch.handle.is_finished());
        if self.active_channels.iter().any(|ch| ch.name == "whatsapp") {
            println!("\n  WhatsApp is already running. Use /stop to stop channels.\n");
            return;
        }
        let mut gw_config = load_config(None);
        cli::check_api_key(&gw_config);
        gw_config.channels.whatsapp.enabled = true;
        gw_config.channels.telegram.enabled = false;
        gw_config.channels.feishu.enabled = false;
        gw_config.channels.email.enabled = false;
        let stop = Arc::new(AtomicBool::new(false));
        let stop2 = stop.clone();
        let dtx = self.display_tx.clone();
        let ch = self.core_handle.clone();
        println!("\n  Scan the QR code when it appears");
        let handle = tokio::spawn(async move {
            cli::run_gateway_async(gw_config, ch, Some(stop2), Some(dtx)).await;
        });
        self.active_channels.push(super::ActiveChannel {
            name: "whatsapp".to_string(),
            stop,
            handle,
        });
        println!("  WhatsApp running in background. Continue chatting.\n");
    }

    /// /telegram — start Telegram channel in background.
    fn cmd_telegram(&mut self) {
        self.active_channels.retain(|ch| !ch.handle.is_finished());
        if self.active_channels.iter().any(|ch| ch.name == "telegram") {
            println!("\n  Telegram is already running. Use /stop to stop channels.\n");
            return;
        }
        println!();
        let mut gw_config = load_config(None);
        cli::check_api_key(&gw_config);
        let saved_token = &gw_config.channels.telegram.token;
        let token = if !saved_token.is_empty() {
            println!("  Using saved bot token");
            saved_token.clone()
        } else {
            println!("  No Telegram bot token found.");
            println!("  Get one from @BotFather on Telegram.\n");
            let tok_prompt = "  Enter bot token: ";
            let t = match self.rl.as_mut().unwrap().readline(tok_prompt) {
                Ok(line) => line.trim().to_string(),
                Err(_) => {
                    return;
                }
            };
            if t.is_empty() {
                println!("  Cancelled.\n");
                return;
            }
            print!("  Validating token... ");
            io::stdout().flush().ok();
            if cli::validate_telegram_token(&t) {
                println!("valid!\n");
            } else {
                println!("invalid!");
                println!("  Check the token and try again.\n");
                return;
            }
            let save_prompt = "  Save token to config for next time? [Y/n] ";
            if let Ok(answer) = self.rl.as_mut().unwrap().readline(save_prompt) {
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
        let dtx = self.display_tx.clone();
        let ch = self.core_handle.clone();
        let handle = tokio::spawn(async move {
            cli::run_gateway_async(gw_config, ch, Some(stop2), Some(dtx)).await;
        });
        self.active_channels.push(super::ActiveChannel {
            name: "telegram".to_string(),
            stop,
            handle,
        });
        println!("  Telegram running in background. Continue chatting.\n");
    }

    /// /email — start Email channel in background.
    fn cmd_email(&mut self) {
        self.active_channels.retain(|ch| !ch.handle.is_finished());
        if self.active_channels.iter().any(|ch| ch.name == "email") {
            println!("\n  Email is already running. Use /stop to stop channels.\n");
            return;
        }
        println!();
        let mut gw_config = load_config(None);
        cli::check_api_key(&gw_config);
        let email_cfg = &gw_config.channels.email;
        if email_cfg.imap_host.is_empty()
            || email_cfg.username.is_empty()
            || email_cfg.password.is_empty()
        {
            println!("  Email not configured. Run `nanobot email` first or add settings to config.json.\n");
            return;
        }
        println!("  Starting Email channel...");
        println!("  Polling {}", email_cfg.imap_host);
        gw_config.channels.email.enabled = true;
        gw_config.channels.whatsapp.enabled = false;
        gw_config.channels.telegram.enabled = false;
        gw_config.channels.feishu.enabled = false;
        let stop = Arc::new(AtomicBool::new(false));
        let stop2 = stop.clone();
        let dtx = self.display_tx.clone();
        let ch = self.core_handle.clone();
        let handle = tokio::spawn(async move {
            cli::run_gateway_async(gw_config, ch, Some(stop2), Some(dtx)).await;
        });
        self.active_channels.push(super::ActiveChannel {
            name: "email".to_string(),
            stop,
            handle,
        });
        println!("  Email running in background. Continue chatting.\n");
    }

    /// /voice — toggle voice mode.
    #[cfg(feature = "voice")]
    async fn cmd_voice(&mut self) {
        if self.voice_session.is_some() {
            if let Some(ref mut vs) = self.voice_session {
                vs.stop_playback();
            }
            self.voice_session = None;
            // Restore thinking display when voice mode turns off.
            self.core_handle
                .counters
                .suppress_thinking_in_tts
                .store(false, Ordering::Relaxed);
            println!("\nVoice mode OFF\n");
        } else {
            match crate::voice::VoiceSession::with_lang(self.lang.as_deref()).await {
                Ok(vs) => {
                    self.voice_session = Some(vs);
                    // Auto-suppress thinking tokens from TTS.
                    self.core_handle
                        .counters
                        .suppress_thinking_in_tts
                        .store(true, Ordering::Relaxed);
                    println!(
                        "\nVoice mode ON. Ctrl+Space or Enter to speak/interrupt, type for text.\n"
                    );
                }
                Err(e) => eprintln!("\nFailed to start voice mode: {}\n", e),
            }
        }
    }
}

// ============================================================================
// Pure trio config helpers (testable without ReplContext)
// ============================================================================

/// Enable trio mode on a Config. Returns `true` if router/specialist are missing (needs warning).
pub(crate) fn trio_enable(cfg: &mut crate::config::schema::Config) -> bool {
    use crate::config::schema::DelegationMode;

    let needs_warning =
        cfg.trio.router_model.is_empty() || cfg.trio.specialist_model.is_empty();
    cfg.trio.enabled = true;
    cfg.tool_delegation.mode = DelegationMode::Trio;
    cfg.tool_delegation.apply_mode();
    needs_warning
}

/// Auto-detect router and specialist from available LM Studio models.
///
/// Scans model names for known patterns. Only picks models that aren't
/// the main model. Returns (router, specialist) — either may be None.
pub(crate) fn pick_trio_models(
    available: &[String],
    main_model: &str,
) -> (Option<String>, Option<String>) {
    let main_lower = main_model.to_lowercase();

    // Fuzzy "is this the main model?" — matches the same way is_model_available does:
    // exact, or either side is a substring of the other (handles org prefixes,
    // resolved-vs-config name mismatches, etc.)
    let is_main = |candidate_lower: &str| -> bool {
        if main_lower.is_empty() {
            return false;
        }
        candidate_lower == main_lower
            || candidate_lower.contains(&main_lower)
            || main_lower.contains(candidate_lower)
    };

    // Router detection (first match wins)
    let router_patterns: &[&str] = &["orchestrator", "router"];
    let router = router_patterns.iter().find_map(|pat| {
        available.iter().find(|m| {
            let low = m.to_lowercase();
            low.contains(pat) && !is_main(&low)
        })
    }).cloned();

    // Specialist detection (first match wins, excludes main + router)
    let router_lower = router.as_ref().map(|r| r.to_lowercase());
    let specialist_patterns: &[&str] = &[
        "function-calling",
        "functiongemma",
        "instruct",
        "ministral",
    ];
    let specialist = specialist_patterns.iter().find_map(|pat| {
        available.iter().find(|m| {
            let low = m.to_lowercase();
            low.contains(pat)
                && !is_main(&low)
                && router_lower.as_deref() != Some(low.as_str())
        })
    }).cloned();

    (router, specialist)
}

/// Whether trio mode should be auto-activated at REPL startup.
///
/// Returns `true` when all conditions hold:
/// - Running in local mode (`is_local`)
/// - Both router and specialist models are configured (non-empty)
/// - Not already in `DelegationMode::Trio`
pub(crate) fn should_auto_activate_trio(
    is_local: bool,
    router_model: &str,
    specialist_model: &str,
    current_mode: &crate::config::schema::DelegationMode,
) -> bool {
    use crate::config::schema::DelegationMode;

    is_local
        && !router_model.is_empty()
        && !specialist_model.is_empty()
        && *current_mode != DelegationMode::Trio
}

/// Disable trio mode on a Config, switching to inline (single model).
pub(crate) fn trio_disable(cfg: &mut crate::config::schema::Config) {
    use crate::config::schema::DelegationMode;

    cfg.trio.enabled = false;
    cfg.tool_delegation.mode = DelegationMode::Inline;
    cfg.tool_delegation.apply_mode();
}

/// Set a trio role model (router or specialist) on a Config.
pub(crate) fn set_trio_role_model_pure(
    cfg: &mut crate::config::schema::Config,
    role: &str,
    model: &str,
) {
    match role {
        "router" => cfg.trio.router_model = model.to_string(),
        "specialist" => cfg.trio.specialist_model = model.to_string(),
        _ => {}
    }
}

/// Copy trio-related fields from `live` config to `disk` config for persistence.
/// Does NOT touch non-trio fields (model, api keys, etc).
pub(crate) fn persist_trio_fields(
    live: &crate::config::schema::Config,
    disk: &mut crate::config::schema::Config,
) {
    disk.trio = live.trio.clone();
    disk.tool_delegation.mode = live.tool_delegation.mode;
    disk.tool_delegation.strict_no_tools_main = live.tool_delegation.strict_no_tools_main;
    disk.tool_delegation.strict_router_schema = live.tool_delegation.strict_router_schema;
    disk.tool_delegation.role_scoped_context_packs =
        live.tool_delegation.role_scoped_context_packs;
    disk.agents.defaults.local_max_context_tokens =
        live.agents.defaults.local_max_context_tokens;
}

// ============================================================================
// Utility
// ============================================================================

impl ReplContext {
    /// Whether voice mode is currently active.
    pub fn voice_on(&self) -> bool {
        #[cfg(feature = "voice")]
        {
            self.voice_session.is_some()
        }
        #[cfg(not(feature = "voice"))]
        {
            false
        }
    }

    /// Compute the current VRAM budget from live config and model paths.
    fn compute_current_vram_budget(&self) -> crate::server::VramBudgetResult {
        let (vram, ram) = server::detect_available_memory();
        let available = vram.unwrap_or(ram);

        let main_profile = if self.srv.lms_managed {
            let model_name = if !self.config.agents.defaults.lms_main_model.is_empty() {
                &self.config.agents.defaults.lms_main_model
            } else {
                &self.config.agents.defaults.local_model
            };
            server::estimate_model_profile_from_name(model_name)
        } else {
            server::resolve_model_profile_from_path(
                &self.current_model_path.display().to_string(),
                &self.current_model_path,
            )
        };

        let router_profile = if self.config.trio.enabled
            && !self.config.trio.router_model.is_empty()
        {
            Some(server::estimate_model_profile_from_name(
                &self.config.trio.router_model,
            ))
        } else {
            None
        };

        let specialist_profile = if self.config.trio.enabled
            && !self.config.trio.specialist_model.is_empty()
        {
            Some(server::estimate_model_profile_from_name(
                &self.config.trio.specialist_model,
            ))
        } else {
            None
        };

        let input = crate::server::VramBudgetInput {
            available_memory: available,
            vram_cap: (self.config.trio.vram_cap_gb * 1e9) as u64,
            overhead_per_model: 512 * 1_000_000,
            main: main_profile,
            router: router_profile,
            specialist: specialist_profile,
        };

        server::compute_vram_budget(&input)
    }
}

// ============================================================================
// VRAM Budget Display (pure, testable)
// ============================================================================

/// Format a VRAM budget result for display. Pure function.
pub(crate) fn format_vram_budget(result: &crate::server::VramBudgetResult) -> String {
    use std::fmt::Write;
    let mut out = String::new();

    let cap_gb = result.effective_limit_bytes as f64 / 1e9;
    let _ = writeln!(out);
    let _ = writeln!(out, "  \x1b[1mVRAM BUDGET\x1b[0m  (limit: {:.1} GB)", cap_gb);
    let _ = writeln!(out, "  {}", "\u{2500}".repeat(52));

    for b in &result.breakdown {
        let role_upper = b.role.to_uppercase();
        let weights_gb = b.weights_bytes as f64 / 1e9;
        let kv_gb = b.kv_cache_bytes as f64 / 1e9;
        let total_gb = weights_gb + kv_gb + b.overhead_bytes as f64 / 1e9;
        let ctx_k = b.context_tokens / 1024;
        let _ = writeln!(
            out,
            "  {:<11} {:<20} {:.1} GB + {:.1} GB KV ({}K ctx) = {:.1} GB",
            role_upper, b.name, weights_gb, kv_gb, ctx_k, total_gb,
        );
    }

    let overhead_count = result.breakdown.len();
    let overhead_per = result.breakdown.first().map_or(0, |b| b.overhead_bytes);
    let overhead_total = overhead_count as f64 * overhead_per as f64 / 1e9;
    let _ = writeln!(
        out,
        "  {:<11} {} x {:.0} MB                                     = {:.1} GB",
        "OVERHEAD",
        overhead_count,
        overhead_per as f64 / 1e6,
        overhead_total,
    );

    let _ = writeln!(out, "  {}", "\u{2500}".repeat(52));
    let total_gb = result.total_vram_bytes as f64 / 1e9;
    let status = if result.fits {
        format!("\x1b[32mOK\x1b[0m")
    } else {
        format!("\x1b[31mOVER\x1b[0m")
    };
    let _ = writeln!(
        out,
        "  {:<11} {:.1} GB / {:.1} GB  {}",
        "TOTAL", total_gb, cap_gb, status,
    );
    let _ = writeln!(out);

    out
}

// ============================================================================
// Trio Parameter Helpers (pure, testable)
// ============================================================================

/// Apply a trio parameter change to a Config. Returns Ok(description) or Err(message).
pub(crate) fn apply_trio_param(
    config: &mut crate::config::schema::Config,
    role: &str,
    param: &str,
    value: &str,
) -> Result<String, String> {
    match (role, param) {
        ("router", "temperature") => {
            let v: f64 = value.parse().map_err(|_| format!("Invalid number: {}", value))?;
            config.trio.router_temperature = v.clamp(0.0, 2.0);
            Ok(format!("router temperature = {:.1}", config.trio.router_temperature))
        }
        ("specialist", "temperature") => {
            let v: f64 = value.parse().map_err(|_| format!("Invalid number: {}", value))?;
            config.trio.specialist_temperature = v.clamp(0.0, 2.0);
            Ok(format!("specialist temperature = {:.1}", config.trio.specialist_temperature))
        }
        ("router", "ctx") => {
            let ctx = super::parse_ctx_arg(value).map_err(|e| e.to_string())?
                .ok_or_else(|| "Missing context size".to_string())?;
            config.trio.router_ctx_tokens = ctx;
            Ok(format!("router ctx = {}K", ctx / 1024))
        }
        ("specialist", "ctx") => {
            let ctx = super::parse_ctx_arg(value).map_err(|e| e.to_string())?
                .ok_or_else(|| "Missing context size".to_string())?;
            config.trio.specialist_ctx_tokens = ctx;
            Ok(format!("specialist ctx = {}K", ctx / 1024))
        }
        ("router", "no_think") => {
            config.trio.router_no_think = !config.trio.router_no_think;
            Ok(format!("router no_think = {}", config.trio.router_no_think))
        }
        ("main", "no_think") => {
            config.trio.main_no_think = !config.trio.main_no_think;
            Ok(format!("main no_think = {}", config.trio.main_no_think))
        }
        ("trio", "vram_cap") => {
            let gb: f64 = value.parse().map_err(|_| format!("Invalid number: {}", value))?;
            if gb < 1.0 || gb > 256.0 {
                return Err("VRAM cap must be between 1 and 256 GB".to_string());
            }
            config.trio.vram_cap_gb = gb;
            Ok(format!("vram cap = {:.1} GB", gb))
        }
        _ => Err(format!("Unknown parameter: {}.{}", role, param)),
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_alias_all_aliases() {
        assert_eq!(normalize_alias("/l"), "/local");
        assert_eq!(normalize_alias("/m"), "/model");
        assert_eq!(normalize_alias("/t"), "/think");
        assert_eq!(normalize_alias("/thinking"), "/think");
        assert_eq!(normalize_alias("/nt"), "/nothink");
        assert_eq!(normalize_alias("/v"), "/voice");
        assert_eq!(normalize_alias("/wa"), "/whatsapp");
        assert_eq!(normalize_alias("/tg"), "/telegram");
        assert_eq!(normalize_alias("/prov"), "/provenance");
        assert_eq!(normalize_alias("/h"), "/help");
        assert_eq!(normalize_alias("/?"), "/help");
        assert_eq!(normalize_alias("/a"), "/agents");
        assert_eq!(normalize_alias("/s"), "/status");
        assert_eq!(normalize_alias("/rd"), "/restart");
        assert_eq!(normalize_alias("/ctx-info"), "/context");
        assert_eq!(normalize_alias("/ss"), "/sessions");
    }

    #[test]
    fn test_normalize_alias_passthrough() {
        assert_eq!(normalize_alias("/status"), "/status");
        assert_eq!(normalize_alias("/help"), "/help");
        assert_eq!(normalize_alias("/local"), "/local");
        assert_eq!(normalize_alias("/unknown"), "/unknown");
        assert_eq!(normalize_alias("hello"), "hello");
    }

    #[test]
    fn test_command_arg_parsing() {
        // Verify split_once behavior used in dispatch
        let input = "/ctx 32K";
        let (cmd, arg) = input
            .split_once(' ')
            .map(|(c, a)| (c, a.trim()))
            .unwrap_or((input, ""));
        assert_eq!(cmd, "/ctx");
        assert_eq!(arg, "32K");

        // No arg
        let input2 = "/status";
        let (cmd2, arg2) = input2
            .split_once(' ')
            .map(|(c, a)| (c, a.trim()))
            .unwrap_or((input2, ""));
        assert_eq!(cmd2, "/status");
        assert_eq!(arg2, "");
    }

    #[test]
    fn test_trio_passthrough_not_aliased() {
        assert_eq!(normalize_alias("/trio"), "/trio");
    }

    #[test]
    fn test_trio_subcommand_arg_parsing() {
        let input = "/trio status";
        let (cmd, arg) = input
            .split_once(' ')
            .map(|(c, a)| (c, a.trim()))
            .unwrap_or((input, ""));
        assert_eq!(cmd, "/trio");
        assert_eq!(arg, "status");

        let input2 = "/trio router";
        let (_, arg2) = input2
            .split_once(' ')
            .map(|(c, a)| (c, a.trim()))
            .unwrap_or((input2, ""));
        assert_eq!(arg2, "router");

        // No subcommand = toggle
        let input3 = "/trio";
        let (_, arg3) = input3
            .split_once(' ')
            .map(|(c, a)| (c, a.trim()))
            .unwrap_or((input3, ""));
        assert_eq!(arg3, "");
    }

    // ---- trio toggle: pure config state transitions ----

    #[test]
    fn test_trio_enable_sets_all_strict_flags() {
        use crate::config::schema::{Config, DelegationMode};

        let mut cfg = Config::default();
        trio_enable(&mut cfg);

        assert!(cfg.trio.enabled);
        assert_eq!(cfg.tool_delegation.mode, DelegationMode::Trio);
        assert!(cfg.tool_delegation.enabled, "delegation must stay on for trio");
        assert!(cfg.tool_delegation.strict_no_tools_main);
        assert!(cfg.tool_delegation.strict_router_schema);
        assert!(cfg.tool_delegation.role_scoped_context_packs);
    }

    #[test]
    fn test_trio_disable_clears_strict_flags() {
        use crate::config::schema::{Config, DelegationMode};

        let mut cfg = Config::default();
        trio_enable(&mut cfg);
        trio_disable(&mut cfg);

        assert!(!cfg.trio.enabled);
        assert_eq!(cfg.tool_delegation.mode, DelegationMode::Inline);
        assert!(!cfg.tool_delegation.enabled, "inline disables delegation");
        assert!(!cfg.tool_delegation.strict_no_tools_main);
        assert!(!cfg.tool_delegation.strict_router_schema);
        assert!(!cfg.tool_delegation.role_scoped_context_packs);
    }

    #[test]
    fn test_trio_double_enable_is_idempotent() {
        use crate::config::schema::{Config, DelegationMode};

        let mut cfg = Config::default();
        trio_enable(&mut cfg);
        trio_enable(&mut cfg);

        assert!(cfg.trio.enabled);
        assert_eq!(cfg.tool_delegation.mode, DelegationMode::Trio);
        assert!(cfg.tool_delegation.strict_no_tools_main);
    }

    #[test]
    fn test_trio_double_disable_is_idempotent() {
        use crate::config::schema::{Config, DelegationMode};

        let mut cfg = Config::default();
        trio_disable(&mut cfg);

        assert!(!cfg.trio.enabled);
        assert_eq!(cfg.tool_delegation.mode, DelegationMode::Inline);
        assert!(!cfg.tool_delegation.strict_no_tools_main);
    }

    #[test]
    fn test_trio_enable_preserves_model_names() {
        use crate::config::schema::Config;

        let mut cfg = Config::default();
        cfg.trio.router_model = "qwen3-1.7b".to_string();
        cfg.trio.specialist_model = "ministral-3-8b".to_string();

        trio_enable(&mut cfg);

        assert_eq!(cfg.trio.router_model, "qwen3-1.7b");
        assert_eq!(cfg.trio.specialist_model, "ministral-3-8b");
    }

    #[test]
    fn test_trio_disable_preserves_model_names() {
        use crate::config::schema::Config;

        let mut cfg = Config::default();
        cfg.trio.router_model = "qwen3-1.7b".to_string();
        cfg.trio.specialist_model = "ministral-3-8b".to_string();
        trio_enable(&mut cfg);
        trio_disable(&mut cfg);

        // Models stay configured even when trio is off
        assert_eq!(cfg.trio.router_model, "qwen3-1.7b");
        assert_eq!(cfg.trio.specialist_model, "ministral-3-8b");
    }

    #[test]
    fn test_trio_enable_with_empty_models_still_enables() {
        use crate::config::schema::Config;

        let mut cfg = Config::default();
        assert!(cfg.trio.router_model.is_empty());
        assert!(cfg.trio.specialist_model.is_empty());

        let needs_warning = trio_enable(&mut cfg);
        assert!(cfg.trio.enabled, "should enable even with empty models");
        assert!(needs_warning, "should warn about missing models");
    }

    #[test]
    fn test_trio_enable_with_both_models_no_warning() {
        use crate::config::schema::Config;

        let mut cfg = Config::default();
        cfg.trio.router_model = "qwen3-1.7b".to_string();
        cfg.trio.specialist_model = "ministral-3-8b".to_string();

        let needs_warning = trio_enable(&mut cfg);
        assert!(!needs_warning);
    }

    // ---- trio role assignment ----

    #[test]
    fn test_set_trio_router_model() {
        use crate::config::schema::Config;

        let mut cfg = Config::default();
        set_trio_role_model_pure(&mut cfg, "router", "qwen3-1.7b");

        assert_eq!(cfg.trio.router_model, "qwen3-1.7b");
        assert!(cfg.trio.specialist_model.is_empty(), "specialist untouched");
    }

    #[test]
    fn test_set_trio_specialist_model() {
        use crate::config::schema::Config;

        let mut cfg = Config::default();
        set_trio_role_model_pure(&mut cfg, "specialist", "ministral-3-8b");

        assert_eq!(cfg.trio.specialist_model, "ministral-3-8b");
        assert!(cfg.trio.router_model.is_empty(), "router untouched");
    }

    #[test]
    fn test_set_trio_role_overwrites_previous() {
        use crate::config::schema::Config;

        let mut cfg = Config::default();
        set_trio_role_model_pure(&mut cfg, "router", "old-model");
        set_trio_role_model_pure(&mut cfg, "router", "new-model");

        assert_eq!(cfg.trio.router_model, "new-model");
    }

    // ---- persist_trio_fields: what gets copied to the disk config ----

    #[test]
    fn test_persist_trio_fields_copies_all_required() {
        use crate::config::schema::{Config, DelegationMode};

        let mut live = Config::default();
        live.trio.enabled = true;
        live.trio.router_model = "qwen3-1.7b".to_string();
        live.trio.specialist_model = "ministral-3-8b".to_string();
        live.tool_delegation.mode = DelegationMode::Trio;
        live.tool_delegation.strict_no_tools_main = true;
        live.tool_delegation.strict_router_schema = true;
        live.tool_delegation.role_scoped_context_packs = true;

        let mut disk = Config::default();
        persist_trio_fields(&live, &mut disk);

        assert!(disk.trio.enabled);
        assert_eq!(disk.trio.router_model, "qwen3-1.7b");
        assert_eq!(disk.trio.specialist_model, "ministral-3-8b");
        assert_eq!(disk.tool_delegation.mode, DelegationMode::Trio);
        assert!(disk.tool_delegation.strict_no_tools_main);
        assert!(disk.tool_delegation.strict_router_schema);
        assert!(disk.tool_delegation.role_scoped_context_packs);
    }

    #[test]
    fn test_persist_trio_fields_does_not_clobber_other_config() {
        use crate::config::schema::{Config, DelegationMode};

        let live = Config::default();
        let mut disk = Config::default();
        disk.agents.defaults.model = "custom-cloud-model".to_string();
        disk.providers.openrouter.api_key = "my-key".to_string();

        persist_trio_fields(&live, &mut disk);

        assert_eq!(disk.agents.defaults.model, "custom-cloud-model", "should not clobber model");
        assert_eq!(disk.providers.openrouter.api_key, "my-key", "should not clobber api key");
    }

    // ---- roundtrip: enable → disable → enable preserves full config ----

    #[test]
    fn test_trio_roundtrip_enable_disable_enable() {
        use crate::config::schema::{Config, DelegationMode};

        let mut cfg = Config::default();
        cfg.trio.router_model = "qwen3-1.7b".to_string();
        cfg.trio.specialist_model = "ministral-3-8b".to_string();

        trio_enable(&mut cfg);
        assert!(cfg.trio.enabled);
        assert!(cfg.tool_delegation.strict_no_tools_main);

        trio_disable(&mut cfg);
        assert!(!cfg.trio.enabled);
        assert!(!cfg.tool_delegation.strict_no_tools_main);

        trio_enable(&mut cfg);
        assert!(cfg.trio.enabled);
        assert!(cfg.tool_delegation.strict_no_tools_main);
        assert_eq!(cfg.trio.router_model, "qwen3-1.7b", "models survive roundtrip");
    }

    // ---- apply_trio_param tests ----

    #[test]
    fn test_apply_trio_param_router_temp() {
        use crate::config::schema::Config;
        let mut config = Config::default();
        let result = apply_trio_param(&mut config, "router", "temperature", "0.3");
        assert!(result.is_ok());
        assert!((config.trio.router_temperature - 0.3).abs() < f64::EPSILON);
    }

    #[test]
    fn test_apply_trio_param_specialist_temp() {
        use crate::config::schema::Config;
        let mut config = Config::default();
        let result = apply_trio_param(&mut config, "specialist", "temperature", "0.5");
        assert!(result.is_ok());
        assert!((config.trio.specialist_temperature - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_apply_trio_param_invalid_temp() {
        use crate::config::schema::Config;
        let mut config = Config::default();
        let result = apply_trio_param(&mut config, "router", "temperature", "abc");
        assert!(result.is_err());
    }

    #[test]
    fn test_apply_trio_param_temp_clamps() {
        use crate::config::schema::Config;
        let mut config = Config::default();
        let result = apply_trio_param(&mut config, "router", "temperature", "5.0");
        assert!(result.is_ok());
        assert!((config.trio.router_temperature - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_apply_trio_param_router_ctx() {
        use crate::config::schema::Config;
        let mut config = Config::default();
        let result = apply_trio_param(&mut config, "router", "ctx", "8K");
        assert!(result.is_ok());
        assert_eq!(config.trio.router_ctx_tokens, 8192);
    }

    #[test]
    fn test_apply_trio_param_specialist_ctx() {
        use crate::config::schema::Config;
        let mut config = Config::default();
        let result = apply_trio_param(&mut config, "specialist", "ctx", "16384");
        assert!(result.is_ok());
        assert_eq!(config.trio.specialist_ctx_tokens, 16384);
    }

    #[test]
    fn test_apply_trio_param_vram_cap() {
        use crate::config::schema::Config;
        let mut config = Config::default();
        let result = apply_trio_param(&mut config, "trio", "vram_cap", "12");
        assert!(result.is_ok());
        assert!((config.trio.vram_cap_gb - 12.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_apply_trio_param_vram_cap_out_of_range() {
        use crate::config::schema::Config;
        let mut config = Config::default();
        let result = apply_trio_param(&mut config, "trio", "vram_cap", "0.1");
        assert!(result.is_err());
    }

    #[test]
    fn test_apply_trio_param_router_nothink_toggle() {
        use crate::config::schema::Config;
        let mut config = Config::default();
        assert!(config.trio.router_no_think); // default true
        let result = apply_trio_param(&mut config, "router", "no_think", "toggle");
        assert!(result.is_ok());
        assert!(!config.trio.router_no_think);
        let _ = apply_trio_param(&mut config, "router", "no_think", "toggle");
        assert!(config.trio.router_no_think);
    }

    #[test]
    fn test_apply_trio_param_main_nothink_toggle() {
        use crate::config::schema::Config;
        let mut config = Config::default();
        assert!(config.trio.main_no_think);
        let result = apply_trio_param(&mut config, "main", "no_think", "toggle");
        assert!(result.is_ok());
        assert!(!config.trio.main_no_think);
    }

    #[test]
    fn test_apply_trio_param_unknown() {
        use crate::config::schema::Config;
        let mut config = Config::default();
        let result = apply_trio_param(&mut config, "router", "unknown_param", "x");
        assert!(result.is_err());
    }

    // ---- format_vram_budget tests ----

    #[test]
    fn test_format_vram_budget_contains_role_names() {
        let result = crate::server::VramBudgetResult {
            main_ctx: 16384,
            router_ctx: 4096,
            specialist_ctx: 8192,
            total_vram_bytes: 12_000_000_000,
            effective_limit_bytes: 16_000_000_000,
            fits: true,
            breakdown: vec![
                crate::server::VramModelBreakdown {
                    role: "main".to_string(),
                    name: "test-main".to_string(),
                    weights_bytes: 5_000_000_000,
                    kv_cache_bytes: 1_000_000_000,
                    context_tokens: 16384,
                    overhead_bytes: 512_000_000,
                },
                crate::server::VramModelBreakdown {
                    role: "router".to_string(),
                    name: "test-router".to_string(),
                    weights_bytes: 3_000_000_000,
                    kv_cache_bytes: 200_000_000,
                    context_tokens: 4096,
                    overhead_bytes: 512_000_000,
                },
                crate::server::VramModelBreakdown {
                    role: "specialist".to_string(),
                    name: "test-specialist".to_string(),
                    weights_bytes: 2_000_000_000,
                    kv_cache_bytes: 500_000_000,
                    context_tokens: 8192,
                    overhead_bytes: 512_000_000,
                },
            ],
        };
        let output = format_vram_budget(&result);
        assert!(output.contains("MAIN"), "Should contain MAIN role");
        assert!(output.contains("ROUTER"), "Should contain ROUTER role");
        assert!(output.contains("SPECIALIST"), "Should contain SPECIALIST role");
        assert!(output.contains("OK"), "Should show OK when fits=true");
        assert!(output.contains("TOTAL"), "Should contain TOTAL line");
    }

    #[test]
    fn test_format_vram_budget_over_shows_over() {
        let result = crate::server::VramBudgetResult {
            main_ctx: 4096,
            router_ctx: 2048,
            specialist_ctx: 4096,
            total_vram_bytes: 20_000_000_000,
            effective_limit_bytes: 16_000_000_000,
            fits: false,
            breakdown: vec![
                crate::server::VramModelBreakdown {
                    role: "main".to_string(),
                    name: "big-model".to_string(),
                    weights_bytes: 10_000_000_000,
                    kv_cache_bytes: 500_000_000,
                    context_tokens: 4096,
                    overhead_bytes: 512_000_000,
                },
            ],
        };
        let output = format_vram_budget(&result);
        assert!(output.contains("OVER"), "Should show OVER when fits=false");
    }

    // ---- should_auto_activate_trio ----

    #[test]
    fn test_should_auto_activate_trio_happy_path() {
        use crate::config::schema::DelegationMode;
        assert!(should_auto_activate_trio(
            true,
            "qwen3-1.7b",
            "ministral-3-8b",
            &DelegationMode::Delegated,
        ));
    }

    #[test]
    fn test_should_auto_activate_trio_not_local() {
        use crate::config::schema::DelegationMode;
        assert!(!should_auto_activate_trio(
            false,
            "qwen3-1.7b",
            "ministral-3-8b",
            &DelegationMode::Delegated,
        ));
    }

    #[test]
    fn test_should_auto_activate_trio_empty_router() {
        use crate::config::schema::DelegationMode;
        assert!(!should_auto_activate_trio(
            true,
            "",
            "ministral-3-8b",
            &DelegationMode::Delegated,
        ));
    }

    #[test]
    fn test_should_auto_activate_trio_empty_specialist() {
        use crate::config::schema::DelegationMode;
        assert!(!should_auto_activate_trio(
            true,
            "qwen3-1.7b",
            "",
            &DelegationMode::Delegated,
        ));
    }

    #[test]
    fn test_should_auto_activate_trio_already_trio() {
        use crate::config::schema::DelegationMode;
        assert!(!should_auto_activate_trio(
            true,
            "qwen3-1.7b",
            "ministral-3-8b",
            &DelegationMode::Trio,
        ));
    }

    #[test]
    fn test_should_auto_activate_trio_both_empty() {
        use crate::config::schema::DelegationMode;
        assert!(!should_auto_activate_trio(
            true,
            "",
            "",
            &DelegationMode::Delegated,
        ));
    }

    // ---- pick_trio_models ----

    #[test]
    fn test_pick_trio_models_orchestrator_and_instruct() {
        let available = vec![
            "Nemotron-Orchestrator-8B".to_string(),
            "Ministral-3-3B".to_string(),
            "Qwen3-14B".to_string(),
        ];
        let (router, specialist) = pick_trio_models(&available, "Qwen3-14B");
        assert_eq!(router.as_deref(), Some("Nemotron-Orchestrator-8B"));
        assert_eq!(specialist.as_deref(), Some("Ministral-3-3B"));
    }

    #[test]
    fn test_pick_trio_models_skips_main() {
        let available = vec![
            "Nemotron-Orchestrator-8B".to_string(),
        ];
        // Main model matches the only orchestrator — no router picked
        let (router, specialist) = pick_trio_models(&available, "Nemotron-Orchestrator-8B");
        assert!(router.is_none());
        assert!(specialist.is_none());
    }

    #[test]
    fn test_pick_trio_models_router_only() {
        let available = vec![
            "Nemotron-Orchestrator-8B".to_string(),
            "Qwen3-14B".to_string(),
        ];
        let (router, specialist) = pick_trio_models(&available, "Qwen3-14B");
        assert_eq!(router.as_deref(), Some("Nemotron-Orchestrator-8B"));
        assert!(specialist.is_none());
    }

    #[test]
    fn test_pick_trio_models_specialist_only() {
        let available = vec![
            "Qwen3-14B".to_string(),
            "Hermes-3-Llama-instruct-8B".to_string(),
        ];
        let (router, specialist) = pick_trio_models(&available, "Qwen3-14B");
        assert!(router.is_none());
        assert_eq!(specialist.as_deref(), Some("Hermes-3-Llama-instruct-8B"));
    }

    #[test]
    fn test_pick_trio_models_empty_list() {
        let (router, specialist) = pick_trio_models(&[], "Qwen3-14B");
        assert!(router.is_none());
        assert!(specialist.is_none());
    }

    #[test]
    fn test_pick_trio_models_function_calling_preferred() {
        let available = vec![
            "Qwen3-14B".to_string(),
            "Hermes-function-calling-3B".to_string(),
            "Llama-instruct-8B".to_string(),
        ];
        let (_, specialist) = pick_trio_models(&available, "Qwen3-14B");
        assert_eq!(specialist.as_deref(), Some("Hermes-function-calling-3B"));
    }

    #[test]
    fn test_pick_trio_models_case_insensitive() {
        let available = vec![
            "nemotron-ORCHESTRATOR-8b".to_string(),
            "MINISTRAL-3-3B".to_string(),
            "Qwen3-14B".to_string(),
        ];
        let (router, specialist) = pick_trio_models(&available, "Qwen3-14B");
        assert_eq!(router.as_deref(), Some("nemotron-ORCHESTRATOR-8b"));
        assert_eq!(specialist.as_deref(), Some("MINISTRAL-3-3B"));
    }

    #[test]
    fn test_pick_trio_models_no_duplicate_assignment() {
        // A model matching both router and specialist patterns should only be assigned once
        let available = vec![
            "orchestrator-instruct-8B".to_string(),
            "Qwen3-14B".to_string(),
        ];
        let (router, specialist) = pick_trio_models(&available, "Qwen3-14B");
        assert_eq!(router.as_deref(), Some("orchestrator-instruct-8B"));
        // specialist must not reuse the router
        assert!(specialist.is_none());
    }

    // ---- e2e: pick → fill config → should_auto_activate ----

    #[test]
    fn test_pick_trio_e2e_auto_activates() {
        use crate::config::schema::{Config, DelegationMode};

        let available = vec![
            "Qwen3-14B".to_string(),
            "Nemotron-Orchestrator-8B".to_string(),
            "Ministral-3-3B".to_string(),
        ];
        let main_model = "Qwen3-14B";

        // Step 1: pick
        let (auto_router, auto_specialist) = pick_trio_models(&available, main_model);

        // Step 2: fill empty config slots (mirrors mod.rs wiring)
        let mut config = Config::default();
        assert!(config.trio.router_model.is_empty());
        assert!(config.trio.specialist_model.is_empty());

        if config.trio.router_model.is_empty() {
            if let Some(r) = auto_router {
                config.trio.router_model = r;
            }
        }
        if config.trio.specialist_model.is_empty() {
            if let Some(s) = auto_specialist {
                config.trio.specialist_model = s;
            }
        }

        assert_eq!(config.trio.router_model, "Nemotron-Orchestrator-8B");
        assert_eq!(config.trio.specialist_model, "Ministral-3-3B");

        // Step 3: should_auto_activate now returns true
        assert!(should_auto_activate_trio(
            true,
            &config.trio.router_model,
            &config.trio.specialist_model,
            &DelegationMode::Delegated,
        ));

        // Step 4: trio_enable activates without warning
        let needs_warning = trio_enable(&mut config);
        assert!(!needs_warning);
        assert!(config.trio.enabled);
        assert_eq!(config.tool_delegation.mode, DelegationMode::Trio);
    }

    #[test]
    fn test_pick_trio_e2e_explicit_config_not_overridden() {
        use crate::config::schema::Config;

        let available = vec![
            "Qwen3-14B".to_string(),
            "Nemotron-Orchestrator-8B".to_string(),
            "Ministral-3-3B".to_string(),
        ];

        let mut config = Config::default();
        config.trio.router_model = "my-custom-router".to_string();
        config.trio.specialist_model = "my-custom-specialist".to_string();

        let (auto_router, auto_specialist) = pick_trio_models(&available, "Qwen3-14B");
        // Only fill empty slots
        if config.trio.router_model.is_empty() {
            if let Some(r) = auto_router {
                config.trio.router_model = r;
            }
        }
        if config.trio.specialist_model.is_empty() {
            if let Some(s) = auto_specialist {
                config.trio.specialist_model = s;
            }
        }

        // Explicit config preserved
        assert_eq!(config.trio.router_model, "my-custom-router");
        assert_eq!(config.trio.specialist_model, "my-custom-specialist");
    }

    // ---- fuzzy main-model exclusion (real LM Studio keys have org prefixes) ----

    #[test]
    fn test_pick_trio_models_org_prefix_skips_main() {
        // LMS returns "nvidia/nvidia-nemotron-nano-12b-v2-vl" but main_model
        // was resolved to "nvidia-nemotron-nano-12b-v2-vl" (without org prefix).
        let available = vec![
            "nvidia/nvidia-nemotron-nano-12b-v2-vl".to_string(),
            "nvidia/nemotron-orchestrator-8b".to_string(),
        ];
        let (router, specialist) = pick_trio_models(&available, "nvidia-nemotron-nano-12b-v2-vl");
        assert_eq!(router.as_deref(), Some("nvidia/nemotron-orchestrator-8b"));
        // Main model must NOT be picked as specialist even with org prefix mismatch
        assert!(specialist.is_none());
    }

    #[test]
    fn test_pick_trio_models_main_unresolved_hint() {
        // main_model is the original GGUF hint that didn't resolve (returned as-is).
        // available has the real LMS key. Substring match should still exclude it.
        let available = vec![
            "nvidia/nvidia-nemotron-nano-12b-v2-vl".to_string(),
            "nvidia/nemotron-orchestrator-8b".to_string(),
            "bartowski/ministral-3-3b".to_string(),
        ];
        let (router, specialist) = pick_trio_models(&available, "NVIDIA-Nemotron-Nano-12B-v2");
        assert_eq!(router.as_deref(), Some("nvidia/nemotron-orchestrator-8b"));
        // "nvidia-nemotron-nano-12b-v2" substring-matches available[0], so skip it
        assert_eq!(specialist.as_deref(), Some("bartowski/ministral-3-3b"));
    }

    #[test]
    fn test_pick_trio_models_real_lms_model_list() {
        // Mimics the user's actual LM Studio model list from error output
        let available = vec![
            "nemotron-orchestrator-8b-claude-4.5-opus-distill".to_string(),
            "nvidia-nemotron-nano-12b-v2-vl".to_string(),
            "nvidia_orchestrator-8b".to_string(),
        ];
        let (router, specialist) = pick_trio_models(&available, "nvidia-nemotron-nano-12b-v2-vl");
        // Should pick an orchestrator (not the main model)
        assert_eq!(
            router.as_deref(),
            Some("nemotron-orchestrator-8b-claude-4.5-opus-distill")
        );
        // No specialist patterns match any available model
        assert!(specialist.is_none());
    }
}
