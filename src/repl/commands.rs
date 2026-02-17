//! Command dispatch and handlers for the REPL.
//!
//! Contains `ReplContext`, `normalize_alias()`, `dispatch()`, and all
//! `cmd_xxx()` command handlers.

use std::io::{self, BufRead, IsTerminal, Write as _};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use rustyline::error::ReadlineError;
use rustyline::ExternalPrinter as _;
use tokio::sync::mpsc;

use crate::agent::agent_loop::{AgentLoop, SharedCoreHandle};
use crate::agent::audit::{AuditLog, ToolEvent};
use crate::agent::provenance::{ClaimVerifier, ClaimStatus};
use crate::cli;
use crate::config::loader::{get_data_dir, load_config, save_config};
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
    pub rl: rustyline::DefaultEditor,
    /// Health watchdog task handle — aborted on mode switch, restarted on `/local`.
    pub watchdog_handle: Option<tokio::task::JoinHandle<()>>,
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
        );
    }

    /// (Re)start the health watchdog for all active local servers.
    ///
    /// Aborts any previous watchdog task, collects current server ports, and
    /// spawns a fresh watchdog. Called on REPL init and on `/local` toggle-on.
    pub fn restart_watchdog(&mut self) {
        if let Some(handle) = self.watchdog_handle.take() {
            handle.abort();
        }
        let mut ports = Vec::new();
        ports.push(("main".to_string(), self.srv.local_port.clone()));
        if let Some(ref cp) = self.srv.compaction_port {
            ports.push(("compaction".to_string(), cp.clone()));
        }
        if let Some(ref dp) = self.srv.delegation_port {
            ports.push(("delegation".to_string(), dp.clone()));
        }
        self.watchdog_handle = Some(
            crate::server::start_health_watchdog(ports, self.display_tx.clone()),
        );
    }

    /// Stop the health watchdog (e.g. when switching to cloud mode).
    pub fn stop_watchdog(&mut self) {
        if let Some(handle) = self.watchdog_handle.take() {
            handle.abort();
        }
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
        let printer_result = self.rl.create_external_printer();

        // Move the editor to a blocking thread so we can select! on display_rx.
        let mut rl = std::mem::replace(
            &mut self.rl,
            rustyline::DefaultEditor::new().expect("DefaultEditor::new"),
        );
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
                        self.rl = rl_back;
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
            self.rl = rl_back;
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
        let ch_names: Vec<&str> = self.active_channels.iter()
            .map(|c| super::short_channel_name(&c.name))
            .collect();
        tui::print_status_bar(&self.core_handle, &ch_names, sa_count);
    }

    /// Rebuild the shared core from current ServerState and then rebuild the agent loop.
    ///
    /// Combines `apply_server_change` + `rebuild_agent_loop` into one call.
    pub fn apply_and_rebuild(&mut self) {
        super::apply_server_change(
            &self.srv,
            &self.current_model_path,
            &self.core_handle,
            &self.config,
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
        "/t" => "/think",
        "/v" => "/voice",
        "/wa" => "/whatsapp",
        "/tg" => "/telegram",
        "/p" => "/provenance",
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
        let (cmd, arg) = input.split_once(' ')
            .map(|(c, a)| (c, a.trim()))
            .unwrap_or((input, ""));
        let cmd = normalize_alias(cmd);
        match cmd {
            "/help" => { super::print_help(); }
            "/think" => { self.cmd_think(arg); }
            "/status" => { self.cmd_status().await; }
            "/context" => { self.cmd_context(); }
            "/memory" => { self.cmd_memory(); }
            "/agents" => { self.cmd_agents().await; }
            "/audit" => { self.cmd_audit(); }
            "/verify" => { self.cmd_verify().await; }
            "/kill" => { self.cmd_kill(arg).await; }
            "/stop" => { self.cmd_stop().await; }

            "/replay" => { self.cmd_replay(arg).await; }
            "/provenance" => { self.cmd_provenance(); }
            "/restart" => { self.cmd_restart().await; }
            "/ctx" => { self.cmd_ctx(arg).await; }
            "/model" => { self.cmd_model().await; }
            "/local" => { self.cmd_local().await; }
            "/whatsapp" => { self.cmd_whatsapp(); }
            "/telegram" => { self.cmd_telegram(); }
            "/email" => { self.cmd_email(); }
            #[cfg(feature = "voice")]
            "/voice" => { self.cmd_voice().await; }
            _ => { return false; }
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
        let is_local = crate::LOCAL_MODE.load(Ordering::SeqCst);
        let model_name = &core.model;
        let mode_label = if is_local { "local" } else { "cloud" };

        println!();
        println!("  {}MODE{}      {} ({})", tui::BOLD, tui::RESET, mode_label, model_name);

        let thinking = counters.thinking_budget.load(Ordering::Relaxed);
        if thinking > 0 {
            println!("  {}THINKING{}  {}\u{1f9e0}{} enabled (budget: {} tokens)", tui::BOLD, tui::RESET, tui::GREY, tui::RESET, thinking);
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
            tui::BOLD, tui::RESET,
            tui::format_thousands(used), tui::format_thousands(max),
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

        let agent_count = self.agent_loop.subagent_manager().get_running_count().await;
        println!("  {}AGENTS{}    {} running", tui::BOLD, tui::RESET, agent_count);

        self.active_channels.retain(|ch| !ch.handle.is_finished());
        if !self.active_channels.is_empty() {
            let ch_names: Vec<&str> = self.active_channels.iter()
                .map(|c| super::short_channel_name(&c.name))
                .collect();
            println!("  {}CHANNELS{}  {}", tui::BOLD, tui::RESET, ch_names.join(" "));
        }

        let turn = counters.learning_turn_counter.load(Ordering::Relaxed);
        println!("  {}TURN{}      {}", tui::BOLD, tui::RESET, turn);

        // Server health display (local mode or when delegation server is active).
        {
            let mut servers: Vec<String> = Vec::new();

            if is_local {
                let main_health = crate::server::check_health(
                    &format!("http://localhost:{}/v1", self.srv.local_port)
                ).await;
                let (color, label) = if main_health {
                    (tui::GREEN, "healthy")
                } else {
                    (tui::RED, "DOWN")
                };
                servers.push(format!(
                    "main:{} ({}{}{}{})",
                    self.srv.local_port, color, tui::BOLD, label, tui::RESET
                ));
            }

            if let Some(ref cp) = self.srv.compaction_port {
                let health = crate::server::check_health(
                    &format!("http://localhost:{}/v1", cp)
                ).await;
                let (color, label) = if health {
                    (tui::GREEN, "healthy")
                } else {
                    (tui::RED, "DOWN")
                };
                servers.push(format!(
                    "compact:{} ({}{}{}{})",
                    cp, color, tui::BOLD, label, tui::RESET
                ));
            }

            if let Some(ref dp) = self.srv.delegation_port {
                let deleg_healthy = counters.delegation_healthy.load(Ordering::Relaxed);
                let (color, label) = if deleg_healthy {
                    (tui::GREEN, "healthy")
                } else {
                    (tui::RED, "DOWN")
                };
                servers.push(format!(
                    "deleg:{} ({}{}{}{})",
                    dp, color, tui::BOLD, label, tui::RESET
                ));
            }

            if !servers.is_empty() {
                println!(
                    "  {}SERVERS{}   {}",
                    tui::BOLD, tui::RESET,
                    servers.join("  ")
                );
            }
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
        let pct = if max > 0 { (used as f64 / max as f64) * 100.0 } else { 0.0 };

        // Estimate system prompt size from an empty message build.
        let system_prompt = core.context.build_messages(
            &[], "", None, None, Some("cli"), Some("repl"), false, None,
        );
        let system_tokens = if let Some(sys) = system_prompt.first() {
            crate::agent::token_budget::TokenBudget::estimate_message_tokens_pub(sys)
        } else {
            0
        };

        println!();
        println!("  {}Context Breakdown{}", tui::BOLD, tui::RESET);
        println!("  {}System prompt:    {} {:>6} tokens{}", tui::DIM, tui::RESET, tui::format_thousands(system_tokens), tui::RESET);
        println!("  {}History:          {} {:>6} messages{}", tui::DIM, tui::RESET, msg_count, tui::RESET);
        println!("  {}Working memory:   {} {:>6} tokens{}", tui::DIM, tui::RESET, tui::format_thousands(wm_tokens), tui::RESET);
        println!("  {}Turn:             {} {:>6}{}", tui::DIM, tui::RESET, turn, tui::RESET);
        println!("  {}─────────────────────────────{}", tui::DIM, tui::RESET);
        println!(
            "  {}Total:            {} {:>6} / {} tokens ({:.1}%){}",
            tui::DIM, tui::RESET,
            tui::format_thousands(used), tui::format_thousands(max),
            pct, tui::RESET
        );
        println!();
    }

    /// /memory — show working memory for current session.
    fn cmd_memory(&self) {
        let core = self.core_handle.swappable();
        if !core.memory_enabled {
            println!("\n  Memory system is disabled.\n");
        } else {
            let wm = core.working_memory.get_context(&self.session_id, usize::MAX);
            if wm.is_empty() {
                println!("\n  No working memory for this session.\n");
            } else {
                println!("\n  {}Working Memory (session: {}){}\n", tui::BOLD, self.session_id, tui::RESET);
                for line in wm.lines() {
                    println!("  {}{}{}", tui::DIM, line, tui::RESET);
                }
                let tokens = crate::agent::token_budget::TokenBudget::estimate_str_tokens(&wm);
                println!("\n  {}({} tokens){}\n", tui::DIM, tui::format_thousands(tokens), tui::RESET);
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
                println!("  {:<4} {:<14} {:<12} {:<6} {:<8} {}", "SEQ", "TOOL", "EXECUTOR", "OK", "MS", "RESULT (preview)");
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
                    Ok(n) => println!("\n  \x1b[32m\u{2713}\x1b[0m Hash chain valid ({} entries)", n),
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
                let last_response = history.iter().rev()
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
                                    ClaimStatus::Derived  => ("~", "\x1b[34m"),
                                    ClaimStatus::Claimed  => ("\u{26a0}", "\x1b[33m"),
                                    ClaimStatus::Recalled => ("\u{25c7}", "\x1b[2m"),
                                };
                                let preview: String = c.text.chars().take(60).collect();
                                println!("  {}{}\x1b[0m [{}] {}", color, marker, c.claim_type, preview);
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

        if !arg.is_empty() {
            // Parse explicit budget
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
                    println!("\n  Usage: /think [budget]\n  Examples: /think, /think 16000, /think 0\n");
                }
            }
        } else {
            // Toggle: off → default budget, on → off
            let was_on = counters.thinking_budget.load(Ordering::Relaxed) > 0;
            if was_on {
                counters.thinking_budget.store(0, Ordering::Relaxed);
                println!("\n  Thinking \x1b[33mdisabled\x1b[0m\n");
            } else {
                // Default budget: half of max_tokens, clamped to [1024, 32000]
                let default_budget = (core.max_tokens / 2).clamp(1024, 32000);
                counters.thinking_budget.store(default_budget, Ordering::Relaxed);
                println!("\n  \x1b[90m\u{1f9e0}\x1b[0m Thinking \x1b[32menabled\x1b[0m — budget: {} tokens\n", default_budget);
            }
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
            let names: Vec<String> = self.active_channels.iter().map(|c| c.name.clone()).collect();
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

    /// /replay — show session message history.
    async fn cmd_replay(&self, arg: &str) {
        let core = self.core_handle.swappable();
        let history = core.sessions.get_history(&self.session_id, 200, 0).await;

        if history.is_empty() {
            println!("\n  No messages in session history.\n");
        } else if arg == "full" {
            // Show full content of all messages.
            println!("\n  {}Session replay ({} messages):{}\n", tui::BOLD, history.len(), tui::RESET);
            for (i, msg) in history.iter().enumerate() {
                let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("?");
                let content = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
                let has_tc = msg.get("tool_calls").is_some();
                let tc_id = msg.get("tool_call_id").and_then(|v| v.as_str());
                println!("  {}[{}]{} {} {}", tui::DIM, i, tui::RESET, role,
                    if has_tc { "[+tool_calls]" } else if tc_id.is_some() { &format!("[tc:{}]", tc_id.unwrap()) } else { "" });
                if !content.is_empty() {
                    let preview: String = content.chars().take(200).collect();
                    for line in preview.lines() {
                        println!("    {}{}{}", tui::DIM, line, tui::RESET);
                    }
                    if content.len() > 200 {
                        println!("    {}...({} total chars){}", tui::DIM, content.len(), tui::RESET);
                    }
                }
            }
            println!();
        } else if let Ok(idx) = arg.parse::<usize>() {
            // Show specific message.
            if idx >= history.len() {
                println!("\n  Message {} out of range (0..{}).\n", idx, history.len() - 1);
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
            println!("\n  {}Session replay ({} messages):{}\n", tui::BOLD, history.len(), tui::RESET);
            for (i, msg) in history.iter().enumerate() {
                let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("?");
                let content = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
                let tokens = crate::agent::token_budget::TokenBudget::estimate_str_tokens(content);
                let has_tc = msg.get("tool_calls").is_some();
                let name = msg.get("name").and_then(|n| n.as_str());
                let extra = if has_tc { " [+tool_calls]" }
                    else if let Some(n) = name { &format!(" [{}]", n) }
                    else { "" };
                let preview: String = content.chars().take(60).collect();
                let preview = preview.replace('\n', " ");
                println!(
                    "  {}[{:>3}]{} {:<10} ({:>5} tok){} {}",
                    tui::DIM, i, tui::RESET,
                    role, tokens, extra, preview
                );
            }
            println!("\n  {}Usage: /replay full | /replay <N>{}\n", tui::DIM, tui::RESET);
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
        cli::rebuild_core(
            &self.core_handle,
            &toggled_config,
            &self.srv.local_port,
            self.current_model_path.file_name().and_then(|n| n.to_str()),
            self.srv.compaction_port.as_deref(),
            self.srv.delegation_port.as_deref(),
        );
        self.agent_loop = cli::create_agent_loop(
            self.core_handle.clone(), &toggled_config, Some(self.cron_service.clone()), self.email_config.clone(), Some(self.display_tx.clone()),
        );
        if !was_enabled {
            println!("\n  Provenance \x1b[32menabled\x1b[0m (tool calls visible, audit logging on)\n");
        } else {
            println!("\n  Provenance \x1b[33mdisabled\x1b[0m\n");
        }
    }

    /// /restart — restart delegation server.
    async fn cmd_restart(&mut self) {
        let core = self.core_handle.swappable();
        if !core.tool_delegation_config.enabled {
            println!("  Tool delegation is not enabled.");
            return;
        }

        // Stop existing delegation server if any.
        server::stop_delegation_server(
            &mut self.srv.delegation_process,
            &mut self.srv.delegation_port,
        );

        // Try to start a fresh one.
        server::start_delegation_if_available(
            &mut self.srv.delegation_process,
            &mut self.srv.delegation_port,
        ).await;

        if self.srv.delegation_port.is_some() {
            // Reset health flag and rebuild core with new provider.
            self.core_handle.counters.delegation_healthy.store(true, Ordering::Relaxed);
            self.core_handle.counters.delegation_retry_counter.store(0, Ordering::Relaxed);
            cli::rebuild_core(
                &self.core_handle, &self.config, &self.srv.local_port,
                self.current_model_path.file_name().and_then(|n| n.to_str()),
                self.srv.compaction_port.as_deref(),
                self.srv.delegation_port.as_deref(),
            );
            self.rebuild_agent_loop();
        } else {
            println!("  No suitable delegation model found in ~/models/");
        }
    }

    /// /ctx [size] — change context size.
    async fn cmd_ctx(&mut self, arg: &str) {
        if !crate::LOCAL_MODE.load(Ordering::SeqCst) {
            println!("\n  {}Not in local mode — use /local first{}\n", tui::DIM, tui::RESET);
            return;
        }

        let new_ctx: usize = match super::parse_ctx_arg(arg) {
            Ok(Some(n)) => n,
            Ok(None) => {
                // No argument → re-auto-detect
                let auto = server::compute_optimal_context_size(&self.current_model_path);
                println!("\n  Auto-detected: {}K", auto / 1024);
                auto
            }
            Err(msg) => {
                println!("\n  {}\n", msg);
                println!("  Usage: /ctx [size]  e.g. /ctx 32K or /ctx 32768\n");
                return;
            }
        };

        // Restart server with new context size
        self.srv.kill_current();
        let fallback_ctx = server::compute_optimal_context_size(&self.current_model_path);
        println!("  {}{}Restarting{} llama.cpp (ctx: {}K)...", tui::BOLD, tui::YELLOW, tui::RESET, new_ctx / 1024);
        match super::start_with_fallback(&mut self.srv, &self.current_model_path, new_ctx, Some((&self.current_model_path, fallback_ctx))).await {
            super::StartOutcome::Started | super::StartOutcome::Fallback => {
                self.apply_and_rebuild();
                tui::print_mode_banner(&self.srv.local_port);
            }
            super::StartOutcome::CloudFallback => {
                self.apply_and_rebuild();
            }
        }
    }

    /// /model — select local model from ~/models/.
    async fn cmd_model(&mut self) {
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
            let marker = if *path == self.current_model_path { " (active)" } else { "" };
            println!("  [{}] {} ({} MB){}", i + 1, name, size_mb, marker);
        }
        let model_prompt = format!("Select model [1-{}] or Enter to cancel: ", models.len());
        let choice = match self.rl.readline(&model_prompt) {
            Ok(line) => line,
            Err(_) => { return; }
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
        let previous_model_path = self.current_model_path.clone();
        self.current_model_path = selected.clone();
        let name = selected.file_name().unwrap().to_string_lossy();
        println!("\nSelected: {}", name);

        // If local mode is active, restart the server with the new model
        if crate::LOCAL_MODE.load(Ordering::SeqCst) {
            self.srv.kill_current();
            let ctx_size = server::compute_optimal_context_size(&self.current_model_path);
            println!("  {}{}Starting{} llama.cpp (ctx: {}K)...", tui::BOLD, tui::YELLOW, tui::RESET, ctx_size / 1024);
            let fallback_ctx = server::compute_optimal_context_size(&previous_model_path);
            match super::start_with_fallback(&mut self.srv, &self.current_model_path, ctx_size, Some((&previous_model_path, fallback_ctx))).await {
                super::StartOutcome::Started => {
                    self.apply_and_rebuild();
                    tui::print_mode_banner(&self.srv.local_port);
                }
                super::StartOutcome::Fallback => {
                    self.current_model_path = previous_model_path.clone();
                    self.apply_and_rebuild();
                    println!("  {}Restored previous model{}", tui::DIM, tui::RESET);
                }
                super::StartOutcome::CloudFallback => {
                    self.apply_and_rebuild();
                }
            }
        } else {
            println!("Model will be used next time you toggle /local on.\n");
        }
    }

    /// /local — toggle between local and cloud mode.
    async fn cmd_local(&mut self) {
        let currently_local = crate::LOCAL_MODE.load(Ordering::SeqCst);

        if !currently_local {
            // Toggle ON: check if a llama.cpp server is already running
            let mut found_port: Option<u16> = None;
            for port in 8080..=8089 {
                let url = format!("http://localhost:{}/health", port);
                if let Ok(resp) = reqwest::blocking::get(&url) {
                    if resp.status().is_success() {
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
                self.srv.local_port = port.to_string();
                crate::LOCAL_MODE.store(true, Ordering::SeqCst);
                let main_ctx = server::compute_optimal_context_size(&self.current_model_path);
                server::start_compaction_if_available(&mut self.srv.compaction_process, &mut self.srv.compaction_port, main_ctx).await;
                if self.config.tool_delegation.enabled && self.config.tool_delegation.auto_local && self.config.tool_delegation.provider.is_none() {
                    server::start_delegation_if_available(&mut self.srv.delegation_process, &mut self.srv.delegation_port).await;
                }
                cli::rebuild_core(&self.core_handle, &self.config, &self.srv.local_port, self.current_model_path.file_name().and_then(|n| n.to_str()), self.srv.compaction_port.as_deref(), self.srv.delegation_port.as_deref());
                self.rebuild_agent_loop();
                self.restart_watchdog();
                tui::print_mode_banner(&self.srv.local_port);
            } else {
                // Kill any orphaned servers from previous runs
                self.srv.kill_current();
                let ctx_size = server::compute_optimal_context_size(&self.current_model_path);
                println!("\n  {}{}Starting{} llama.cpp server (ctx: {}K)...", tui::BOLD, tui::YELLOW, tui::RESET, ctx_size / 1024);

                match super::try_start_server(&mut self.srv, &self.current_model_path, ctx_size).await {
                    Ok(_) => {
                        crate::LOCAL_MODE.store(true, Ordering::SeqCst);
                        server::start_compaction_if_available(&mut self.srv.compaction_process, &mut self.srv.compaction_port, ctx_size).await;
                        if self.config.tool_delegation.enabled && self.config.tool_delegation.auto_local && self.config.tool_delegation.provider.is_none() {
                            server::start_delegation_if_available(&mut self.srv.delegation_process, &mut self.srv.delegation_port).await;
                        }
                        self.apply_and_rebuild();
                        self.restart_watchdog();
                        tui::print_mode_banner(&self.srv.local_port);
                    }
                    Err(e) => {
                        println!("  {}{}Failed: {}{}", tui::BOLD, tui::YELLOW, e, tui::RESET);
                        println!("  {}Remaining in cloud mode{}\n", tui::DIM, tui::RESET);
                    }
                }
            }
        } else {
            // Toggle OFF: kill main and compaction servers, but keep delegation alive
            if let Some(ref mut child) = self.srv.llama_process {
                child.kill().ok();
                child.wait().ok();
            }
            self.srv.llama_process = None;
            server::stop_compaction_server(&mut self.srv.compaction_process, &mut self.srv.compaction_port);
            self.stop_watchdog();
            crate::LOCAL_MODE.store(false, Ordering::SeqCst);

            // Re-spawn delegation if it wasn't already running
            if self.srv.delegation_port.is_none()
                && self.config.tool_delegation.enabled
                && self.config.tool_delegation.auto_local
                && self.config.tool_delegation.provider.is_none()
            {
                server::start_delegation_if_available(
                    &mut self.srv.delegation_process,
                    &mut self.srv.delegation_port,
                ).await;
            }

            self.apply_and_rebuild();
            tui::print_mode_banner(&self.srv.local_port);
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
            name: "whatsapp".to_string(), stop, handle,
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
            let t = match self.rl.readline(tok_prompt) {
                Ok(line) => line.trim().to_string(),
                Err(_) => { return; }
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
            if let Ok(answer) = self.rl.readline(save_prompt) {
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
            name: "telegram".to_string(), stop, handle,
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
        if email_cfg.imap_host.is_empty() || email_cfg.username.is_empty() || email_cfg.password.is_empty() {
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
            name: "email".to_string(), stop, handle,
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
            println!("\nVoice mode OFF\n");
        } else {
            match crate::voice::VoiceSession::with_lang(self.lang.as_deref()).await {
                Ok(vs) => {
                    self.voice_session = Some(vs);
                    println!("\nVoice mode ON. Ctrl+Space or Enter to speak/interrupt, type for text.\n");
                }
                Err(e) => eprintln!("\nFailed to start voice mode: {}\n", e),
            }
        }
    }
}

// ============================================================================
// Utility
// ============================================================================

impl ReplContext {
    /// Whether voice mode is currently active.
    pub fn voice_on(&self) -> bool {
        #[cfg(feature = "voice")]
        { self.voice_session.is_some() }
        #[cfg(not(feature = "voice"))]
        { false }
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
        let (cmd, arg) = input.split_once(' ')
            .map(|(c, a)| (c, a.trim()))
            .unwrap_or((input, ""));
        assert_eq!(cmd, "/ctx");
        assert_eq!(arg, "32K");

        // No arg
        let input2 = "/status";
        let (cmd2, arg2) = input2.split_once(' ')
            .map(|(c, a)| (c, a.trim()))
            .unwrap_or((input2, ""));
        assert_eq!(cmd2, "/status");
        assert_eq!(arg2, "");
    }
}
