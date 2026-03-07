//! Read-only REPL commands: /status, /context, /memory, /clear, /agents, /audit, /verify.

use std::sync::atomic::Ordering;

use crate::agent::context::PromptBlockKind;

use super::*;

impl ReplContext {
    /// /status — show current mode, model, and channel info.
    pub(super) async fn cmd_status(&mut self) {
        let core = self.core_handle.swappable();
        let counters = &self.core_handle.counters;
        let is_local = core.is_local;
        let is_mlx = core.model.starts_with("mlx:");
        let model_name = &core.model;
        let mode_label = if is_mlx {
            "mlx"
        } else if is_local {
            "local"
        } else {
            "cloud"
        };
        let lane_label = if is_mlx {
            "in-process"
        } else if is_local {
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
            "  {}MODE{}      {} ({}, {}, {})",
            tui::BOLD,
            tui::RESET,
            mode_label,
            lane_label,
            core.lane,
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
                .map(|c| super::super::short_channel_name(&c.name))
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
                    let health = crate::server::check_health(
                        remote_base,
                        self.config.monitoring.health_check_timeout_secs,
                    )
                    .await;
                    let (color, label) = if health {
                        (tui::GREEN, "healthy")
                    } else {
                        (tui::RED, "DOWN")
                    };
                    servers.push(format!(
                        "LM Studio ({}{}{}{})",
                        color,
                        tui::BOLD,
                        label,
                        tui::RESET
                    ));
                } else {
                    let main_health = crate::server::check_health(
                        &format!("http://localhost:{}/v1", self.srv.local_port),
                        self.config.monitoring.health_check_timeout_secs,
                    )
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
                let probe_labels: Vec<String> = states
                    .iter()
                    .map(|s| {
                        use crate::heartbeat::health::ProbeStatus;
                        let (indicator, label) = match s.status {
                            ProbeStatus::Healthy => {
                                let ms = s
                                    .last_result
                                    .as_ref()
                                    .map(|r| format!(" ({}ms)", r.latency_ms))
                                    .unwrap_or_default();
                                (
                                    format!("{}{}●{}", tui::GREEN, tui::BOLD, tui::RESET),
                                    format!("{}{}", s.name, ms),
                                )
                            }
                            ProbeStatus::Degraded => {
                                let ago = s
                                    .last_healthy
                                    .map(|t| {
                                        let secs = t.elapsed().as_secs();
                                        if secs < 60 {
                                            format!(" ({}s ago)", secs)
                                        } else {
                                            format!(" ({}m ago)", secs / 60)
                                        }
                                    })
                                    .unwrap_or_default();
                                (
                                    format!("{}{}●{}", tui::RED, tui::BOLD, tui::RESET),
                                    format!("{}: DOWN{}", s.name, ago),
                                )
                            }
                            ProbeStatus::Unknown => (
                                format!("{}{}●{}", tui::YELLOW, tui::BOLD, tui::RESET),
                                format!("{}: pending", s.name),
                            ),
                        };
                        format!("{} {}", indicator, label)
                    })
                    .collect();
                println!(
                    "  {}HEALTH{}    {}",
                    tui::BOLD,
                    tui::RESET,
                    probe_labels.join("  ")
                );
            }
        }

        // TRIO section — only shown when trio mode is active.
        if self.config.trio.enabled {
            let router_health = if let Some(ref hr) = self.health_registry {
                if hr.is_healthy("trio_router") {
                    "healthy"
                } else {
                    "degraded"
                }
            } else {
                "n/a"
            };
            let specialist_health = if let Some(ref hr) = self.health_registry {
                if hr.is_healthy("trio_specialist") {
                    "healthy"
                } else {
                    "degraded"
                }
            } else {
                "n/a"
            };
            let last_action = counters
                .trio_metrics
                .router_action
                .lock()
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
    pub(super) async fn cmd_context(&self) {
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

        println!();
        println!("  {}Context Breakdown{}", tui::BOLD, tui::RESET);
        let block_kind = |kind: PromptBlockKind| match kind {
            PromptBlockKind::Prefix => "prefix",
            PromptBlockKind::Static => "static",
            PromptBlockKind::Runtime => "runtime",
        };

        if core.context.local_prompt_mode {
            let runtime_blocks = self
                .agent_loop
                .local_prompt_runtime_blocks(&self.session_id)
                .await;
            let report = core.context.describe_local_system_prompt(
                None,
                Some("cli"),
                Some("repl"),
                false,
                None,
                &runtime_blocks,
            );
            let wm_in_prompt = report
                .blocks
                .iter()
                .find(|b| b.title == "Working Memory" && b.included)
                .map(|b| b.tokens)
                .unwrap_or(0);

            println!(
                "  {}Prompt mode:      {} local lean{}",
                tui::DIM,
                tui::RESET,
                tui::RESET
            );
            match report.cap_tokens {
                Some(cap) => println!(
                    "  {}System prompt:    {} {:>6} tokens (cap {}){}",
                    tui::DIM,
                    tui::RESET,
                    tui::format_thousands(report.total_tokens),
                    tui::format_thousands(cap),
                    tui::RESET
                ),
                None => println!(
                    "  {}System prompt:    {} {:>6} tokens{}",
                    tui::DIM,
                    tui::RESET,
                    tui::format_thousands(report.total_tokens),
                    tui::RESET
                ),
            }
            println!("  {}  included:{}", tui::DIM, tui::RESET,);
            for block in report.blocks.iter().filter(|b| b.included) {
                println!(
                    "  {}    {:<7} {:<18}{} {:>6}{}",
                    tui::DIM,
                    block_kind(block.kind),
                    block.title,
                    tui::RESET,
                    tui::format_thousands(block.tokens),
                    tui::RESET
                );
            }
            let dropped: Vec<_> = report.blocks.iter().filter(|b| !b.included).collect();
            if !dropped.is_empty() {
                println!("  {}  dropped by cap:{}", tui::DIM, tui::RESET);
                for block in dropped {
                    println!(
                        "  {}    {:<7} {:<18}{} {:>6}{}",
                        tui::DIM,
                        block_kind(block.kind),
                        block.title,
                        tui::RESET,
                        tui::format_thousands(block.tokens),
                        tui::RESET
                    );
                }
            }
            println!(
                "  {}Working memory:   {} {:>6} stored / {:>6} in prompt{}",
                tui::DIM,
                tui::RESET,
                tui::format_thousands(wm_tokens),
                tui::format_thousands(wm_in_prompt),
                tui::RESET
            );
        } else {
            let messages = core.context.build_messages(
                &[],
                "",
                None,
                None,
                Some("cli"),
                Some("repl"),
                false,
                None,
            );
            let system_tokens = messages
                .iter()
                .find(|m| m["role"] == "system")
                .map(crate::agent::token_budget::TokenBudget::estimate_message_tokens_pub)
                .unwrap_or(0);
            let developer_tokens = messages
                .iter()
                .find(|m| m["role"] == "developer")
                .map(crate::agent::token_budget::TokenBudget::estimate_message_tokens_pub)
                .unwrap_or(0);
            let identity_tokens = crate::agent::token_budget::TokenBudget::estimate_str_tokens(
                &core.context.build_identity_prompt(),
            );

            println!(
                "  {}Prompt mode:      {} cloud split{}",
                tui::DIM,
                tui::RESET,
                tui::RESET
            );
            println!(
                "  {}System role:      {} {:>6} tokens{}",
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
            if developer_tokens > 0 {
                println!(
                    "  {}Developer role:   {} {:>6} tokens{}",
                    tui::DIM,
                    tui::RESET,
                    tui::format_thousands(developer_tokens),
                    tui::RESET
                );
            }
            println!(
                "  {}  bootstrap cap:  {} {:>6}{}",
                tui::DIM,
                tui::RESET,
                tui::format_thousands(core.context.bootstrap_budget),
                tui::RESET
            );
            println!(
                "  {}  memory cap:     {} {:>6}{}",
                tui::DIM,
                tui::RESET,
                tui::format_thousands(core.context.long_term_memory_budget),
                tui::RESET
            );
            println!(
                "  {}Working memory:   {} {:>6} tokens{}",
                tui::DIM,
                tui::RESET,
                tui::format_thousands(wm_tokens),
                tui::RESET
            );
        }

        println!(
            "  {}History:          {} {:>6} messages{}",
            tui::DIM,
            tui::RESET,
            msg_count,
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
        let completion = counters
            .last_actual_completion_tokens
            .load(Ordering::Relaxed);
        if est > 0 {
            println!(
                "  {}Last prompt est:  {} {:>6} tokens{}",
                tui::DIM,
                tui::RESET,
                tui::format_thousands(est as usize),
                tui::RESET
            );
        }
        if act > 0 {
            println!(
                "  {}Provider usage:   {} {:>6} prompt / {:>6} completion{}",
                tui::DIM,
                tui::RESET,
                tui::format_thousands(act as usize),
                tui::format_thousands(completion as usize),
                tui::RESET
            );
        }
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
    pub(super) fn cmd_memory(&self) {
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

    /// /clear — clear working memory and conversation history for the current session.
    pub(super) async fn cmd_clear(&self) {
        let core = self.core_handle.swappable();
        if !core.memory_enabled {
            println!("\n  Memory system is disabled.\n");
            return;
        }
        let had_content = !core
            .working_memory
            .get_context(&self.session_id, 1)
            .is_empty();
        core.working_memory.clear(&self.session_id);
        let session_meta = core.sessions.get_or_resume(&self.session_id).await;
        core.sessions.clear_history(&session_meta.id).await;
        self.agent_loop.clear_lcm_engine(&self.session_id).await;
        self.agent_loop.clear_bulletin_cache();
        if had_content {
            println!(
                "\n  Working memory and conversation cleared for session: {}\n",
                self.session_id
            );
        } else {
            println!("\n  Conversation history cleared.\n");
        }
    }

    /// /agents — list running subagents.
    pub(super) async fn cmd_agents(&self) {
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
    pub(super) fn cmd_audit(&self) {
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
    pub(super) async fn cmd_verify(&self) {
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
                let session_meta = core.sessions.get_or_resume(&self.session_id).await;
                let history = core.sessions.get_history(&session_meta.id, 10, 0).await;
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
