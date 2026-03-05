//! Simple mutation REPL commands: /think, /nothink, /long, /kill, /stop, /sessions, /replay.

use std::sync::atomic::Ordering;
use std::time::Duration;

use super::*;

impl ReplContext {
    /// /think, /t — toggle extended thinking / reasoning mode.
    /// /think <budget> — enable with specific token budget (e.g. /think 16000).
    pub(super) fn cmd_think(&self, arg: &str) {
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
    pub(super) fn cmd_nothink(&self) {
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
    pub(super) fn cmd_long(&self, arg: &str) {
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
    pub(super) async fn cmd_kill(&self, arg: &str) {
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
    pub(super) async fn cmd_stop(&mut self) {
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
    pub(super) fn cmd_sessions(&self, arg: &str) {
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
    pub(super) async fn cmd_replay(&self, arg: &str) {
        let core = self.core_handle.swappable();
        let session_meta = core.sessions.get_or_resume(&self.session_id).await;
        let history = core.sessions.get_history(&session_meta.id, 200, 0).await;

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
                        "[+tool_calls]".to_string()
                    } else if tc_id.is_some() {
                        format!("[tc:{}]", tc_id.unwrap())
                    } else {
                        String::new()
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
