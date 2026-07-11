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
                    counters
                        .suppress_thinking_display
                        .store(false, Ordering::Relaxed);
                    println!(
                        "\n  Thinking \x1b[32menabled\x1b[0m — budget: {} tokens\n",
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
                    counters
                        .suppress_thinking_display
                        .store(false, Ordering::Relaxed);
                    println!(
                        "\n  Thinking \x1b[32menabled\x1b[0m — budget: {} tokens\n",
                        clamped
                    );
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
                counters
                    .suppress_thinking_display
                    .store(false, Ordering::Relaxed);
                println!(
                    "\n  Thinking \x1b[32menabled\x1b[0m — budget: {} tokens\n",
                    default_budget
                );
            }
        }
    }

    /// /nothink, /nt — suppress thinking tokens from output (and TTS).
    /// Sets thinking budget to 0 and enables the display/TTS suppressors.
    pub(super) fn cmd_nothink(&self) {
        let counters = &self.core_handle.counters;
        let was_suppressed = counters.suppress_thinking_display.load(Ordering::Relaxed);
        if was_suppressed {
            // Toggle off — re-enable thinking display (but thinking budget stays 0)
            counters
                .suppress_thinking_display
                .store(false, Ordering::Relaxed);
            counters
                .suppress_thinking_in_tts
                .store(false, Ordering::Relaxed);
            println!(
                "\n  Thinking display \x1b[32mrestored\x1b[0m (use /think to re-enable thinking)\n"
            );
        } else {
            counters.thinking_budget.store(0, Ordering::Relaxed);
            counters
                .suppress_thinking_display
                .store(true, Ordering::Relaxed);
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

    /// /lcm — toggle Lossless Context Management on/off.
    /// /lcm stats — show compaction statistics without toggling.
    pub(super) fn cmd_lcm(&self, arg: &str) {
        match LcmAction::parse(arg) {
            LcmAction::Toggle => {
                let enabled = self.agent_loop.toggle_lcm();
                if enabled {
                    println!("\n  LCM \x1b[32menabled\x1b[0m — compaction active.\n");
                } else {
                    println!("\n  LCM \x1b[33mdisabled\x1b[0m.\n");
                }
            }
            LcmAction::Stats => {
                println!("{}", format_lcm_stats(&self.gather_lcm_stats()));
            }
            LcmAction::Usage => {
                println!("\n  Usage: /lcm [stats]\n");
            }
        }
    }

    /// Snapshot the LCM counters for display.
    fn gather_lcm_stats(&self) -> LcmStats {
        use crate::agent::agent_core::RuntimeCounters;

        let counters = &self.core_handle.counters;
        let last_ms = counters.lcm_last_compaction_ms.load(Ordering::Relaxed);
        let last_secs_ago = (last_ms > 0)
            .then(|| RuntimeCounters::now_epoch_ms().saturating_sub(last_ms) / 1000);
        LcmStats {
            enabled: self.agent_loop.lcm_enabled(),
            compactions: counters.lcm_compaction_count.load(Ordering::Relaxed),
            tokens_before: counters.lcm_tokens_before.load(Ordering::Relaxed),
            tokens_after: counters.lcm_tokens_after.load(Ordering::Relaxed),
            last_secs_ago,
        }
    }

    /// /learn — trigger a reflection pass now: distill accumulated working
    /// sessions and observations into MEMORY.md (and the knowledge graph)
    /// without waiting for the token threshold or session exit.
    pub(super) async fn cmd_learn(&self) {
        let core = self.core_handle.swappable();
        if !core.memory_enabled {
            println!("\n  Memory system is disabled.\n");
            return;
        }
        // threshold=0: reflect whenever there is any accumulated content.
        let reflector = crate::agent::reflector::Reflector::new(
            core.memory_provider.clone(),
            core.memory_model.clone(),
            &core.workspace,
            0,
        );
        if !reflector.should_reflect() {
            println!("\n  Nothing to learn yet — no working sessions or observations accumulated.\n");
            return;
        }
        println!("\n  Reflecting on accumulated sessions...");
        // On-demand Bonsai: the compaction sidecar is down between compactions,
        // so bring it up for this memory op and release it after. (No-op when
        // always-on or no sidecar.)
        let sidecar = crate::higgs::CompactionSidecarSpec::from_config(&self.config);
        if let Some(spec) = sidecar.as_ref() {
            if let Err(e) = spec.ensure_up().await {
                println!("  (compaction sidecar unavailable: {})", e);
            }
        }
        let result = reflector.reflect().await;
        if let Some(spec) = sidecar.as_ref() {
            spec.release().await;
        }
        match result {
            Ok(()) => println!("  Learned. MEMORY.md updated.\n"),
            Err(e) => println!("  Reflection failed: {}\n", e),
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
    pub(super) async fn cmd_sessions(&self, arg: &str) {
        let (sub, rest) = arg
            .split_once(' ')
            .map(|(s, r)| (s.trim(), r.trim()))
            .unwrap_or((if arg.is_empty() { "list" } else { arg }, ""));

        match sub {
            "list" => {
                self.cmd_sessions_list().await;
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
                self.cmd_sessions_export(key, fmt).await;
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
                let (indexed, skipped, errors) = crate::agent::session_indexer::index_sessions(
                    &sessions_dir,
                    &memory_sessions_dir,
                );
                println!(
                    "Indexed {} sessions ({} skipped, {} errors)",
                    indexed, skipped, errors
                );
            }
            _ => {
                eprintln!(
                    "Unknown subcommand '{}'. Available: list, export, purge, archive, index",
                    sub
                );
            }
        }
    }

    async fn cmd_sessions_list(&self) {
        let core = self.core_handle.swappable();
        let sessions = core.sessions.list_sessions(None, 100).await;

        if sessions.is_empty() {
            println!("No sessions found.");
            return;
        }

        println!("{:<40} {:<30} {:>6}", "SESSION KEY", "UPDATED", "MSGS");
        println!("{}", "-".repeat(80));

        for s in &sessions {
            let updated = s.updated_at.format("%Y-%m-%d %H:%M:%S UTC").to_string();
            println!(
                "{:<40} {:<30} {:>6}",
                crate::utils::helpers::truncate_string(&s.session_key, 38),
                crate::utils::helpers::truncate_string(&updated, 28),
                s.message_count,
            );
        }
        println!("\n{} session(s) total.", sessions.len());
    }

    async fn cmd_sessions_export(&self, key: &str, format: &str) {
        let core = self.core_handle.swappable();
        let session_id = if let Some(meta) = core.sessions.get_latest_session(key).await {
            meta.id
        } else if let Some(meta) = core.sessions.get_session(key).await {
            meta.id
        } else {
            eprintln!("Session '{}' not found.", key);
            eprintln!("Use `nanobot sessions list` to see available sessions.");
            return;
        };

        let messages = core.sessions.get_all_messages(&session_id).await;
        if format == "jsonl" {
            for msg in &messages {
                println!("{}", serde_json::to_string(msg).unwrap_or_default());
            }
            return;
        }

        println!("# Session: {}\n", key);
        for parsed in &messages {
            let role = parsed
                .get("role")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            let timestamp = parsed
                .get("timestamp")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let time_display = if timestamp.len() >= 19 {
                &timestamp[11..19]
            } else {
                timestamp
            };

            match role {
                "user" => {
                    let text = parsed.get("content").and_then(|v| v.as_str()).unwrap_or("");
                    println!("## User ({})\n\n{}\n", time_display, text);
                }
                "assistant" => {
                    let text = parsed.get("content").and_then(|v| v.as_str()).unwrap_or("");
                    if !text.is_empty() {
                        println!("## Assistant ({})\n\n{}\n", time_display, text);
                    }
                }
                "tool" => {
                    let tool_name = parsed
                        .get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("tool");
                    let result = parsed.get("content").and_then(|v| v.as_str()).unwrap_or("");
                    let abbreviated = crate::utils::helpers::truncate_string(result, 200);
                    println!(
                        "## Tool: {} ({})\n\n{}\n",
                        tool_name, time_display, abbreviated
                    );
                }
                _ => {
                    let text = parsed.get("content").and_then(|v| v.as_str()).unwrap_or("");
                    if !text.is_empty() {
                        println!("## {} ({})\n\n{}\n", role, time_display, text);
                    }
                }
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

/// What `/lcm <arg>` should do.
#[derive(Debug, PartialEq)]
enum LcmAction {
    /// No argument — flip LCM on/off (existing behavior).
    Toggle,
    /// `stats` — display counters, never toggles.
    Stats,
    /// Anything else — show usage, never toggles.
    Usage,
}

impl LcmAction {
    fn parse(arg: &str) -> Self {
        match arg.trim() {
            "" => Self::Toggle,
            "stats" => Self::Stats,
            _ => Self::Usage,
        }
    }
}

/// Snapshot of session LCM compaction statistics for display.
struct LcmStats {
    enabled: bool,
    compactions: u64,
    tokens_before: u64,
    tokens_after: u64,
    last_secs_ago: Option<u64>,
}

/// Render LCM stats for the terminal. Pure — unit-tested below.
fn format_lcm_stats(stats: &LcmStats) -> String {
    let state = if stats.enabled {
        "\x1b[32menabled\x1b[0m"
    } else {
        "\x1b[33mdisabled\x1b[0m"
    };
    let mut out = format!("\n  LCM {}\n", state);

    if stats.compactions == 0 {
        out.push_str("  No compactions this session.\n");
        return out;
    }

    let saved = stats.tokens_before.saturating_sub(stats.tokens_after);
    out.push_str(&format!(
        "  Compactions this session: {}\n  Tokens: {} → {} (saved {})\n",
        stats.compactions, stats.tokens_before, stats.tokens_after, saved
    ));
    if let Some(secs) = stats.last_secs_ago {
        out.push_str(&format!("  Last compaction: {}s ago\n", secs));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lcm_action_parse() {
        assert_eq!(LcmAction::parse(""), LcmAction::Toggle);
        assert_eq!(LcmAction::parse("stats"), LcmAction::Stats);
        assert_eq!(LcmAction::parse(" stats "), LcmAction::Stats);
        // Unknown args must not toggle — they fall through to usage.
        assert_eq!(LcmAction::parse("bogus"), LcmAction::Usage);
    }

    #[test]
    fn test_format_lcm_stats_zero_compactions() {
        let s = format_lcm_stats(&LcmStats {
            enabled: false,
            compactions: 0,
            tokens_before: 0,
            tokens_after: 0,
            last_secs_ago: None,
        });
        assert!(s.contains("disabled"));
        assert!(s.to_lowercase().contains("no compactions"));
        assert!(!s.to_lowercase().contains("last compaction"));
    }

    #[test]
    fn test_format_lcm_stats_with_compactions() {
        let s = format_lcm_stats(&LcmStats {
            enabled: true,
            compactions: 3,
            tokens_before: 45000,
            tokens_after: 12000,
            last_secs_ago: Some(42),
        });
        assert!(s.contains("enabled"));
        assert!(s.contains('3'));
        assert!(s.contains("45000"));
        assert!(s.contains("12000"));
        assert!(s.contains("33000"), "should show tokens saved: {}", s);
        assert!(s.contains("42s"));
    }
}
