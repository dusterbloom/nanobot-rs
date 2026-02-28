#![allow(dead_code)]
//! REPL loop and interactive command dispatch for `nanobot agent`.
//!
//! Contains the main agent REPL, slash-command handlers, voice recording
//! pipeline, and background channel management.

mod commands;
mod incremental;

pub(crate) use commands::{should_auto_activate_trio, trio_enable};

use std::io::{self, IsTerminal, Write as _};
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use rustyline::error::ReadlineError;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use crate::agent::agent_loop::SharedCoreHandle;
use crate::agent::audit::{AuditLog, ToolEvent};
use crate::agent::provenance::{ClaimStatus, ClaimVerifier};
use crate::cli;
use crate::config::loader::{get_data_dir, load_config, save_config};
use crate::config::schema::Config;
use crate::cron::service::CronService;
use crate::heartbeat::service::{
    HeartbeatService, DEFAULT_HEARTBEAT_INTERVAL_S, DEFAULT_MAINTENANCE_COMMANDS,
};
use crate::server;
use crate::syntax;
use crate::tui;

// ============================================================================
// Streaming TTS type (feature-gated)
// ============================================================================

#[cfg(feature = "voice")]
type TtsSentenceSender = Option<std::sync::mpsc::Sender<crate::voice::TtsCommand>>;
#[cfg(not(feature = "voice"))]
type TtsSentenceSender = Option<()>;

// ============================================================================
// Helpers (testable, pure-ish)
// ============================================================================

/// Truncate tool output for verbatim display: max `max_lines` lines or `max_chars` characters.
fn truncate_output(data: &str, max_lines: usize, max_chars: usize) -> String {
    let mut out = String::new();
    let mut lines = 0usize;
    let mut chars = 0usize;
    for line in data.lines() {
        if lines >= max_lines || chars >= max_chars {
            out.push_str("...[truncated]");
            break;
        }
        if !out.is_empty() {
            out.push('\n');
            chars += 1;
        }
        let remaining = max_chars.saturating_sub(chars);
        if line.len() > remaining {
            let partial: String = line.chars().take(remaining).collect();
            out.push_str(&partial);
            out.push_str("...[truncated]");
            break;
        }
        out.push_str(line);
        chars += line.len();
        lines += 1;
    }
    out
}

/// Extract a short context label from tool result data for display.
///
/// For `read_file`, extracts the file path from the `# /path/to/file (lines ...)`
/// header. For `exec`, extracts the command if short enough. Returns empty
/// string if no useful context can be extracted.
fn extract_tool_context(tool_name: &str, result_data: &str) -> String {
    match tool_name {
        "read_file" => {
            // Result starts with "# /path/to/file.rs (lines N-M of T)"
            if let Some(line) = result_data.lines().next() {
                if let Some(rest) = line.strip_prefix("# ") {
                    // Extract just the filename, not the full path
                    if let Some(paren) = rest.find(" (") {
                        let path = &rest[..paren];
                        // Show just filename or last 2 components
                        let short: String = path
                            .rsplit('/')
                            .take(2)
                            .collect::<Vec<_>>()
                            .into_iter()
                            .rev()
                            .collect::<Vec<_>>()
                            .join("/");
                        return short;
                    }
                }
            }
            // Fallback: check for error messages
            if result_data.starts_with("Error:") {
                let preview: String = result_data.chars().take(60).collect();
                return preview;
            }
            String::new()
        }
        "exec" => {
            // For exec, result_data is the output, not the command.
            // We don't have the command here, so skip.
            String::new()
        }
        "edit_file" => {
            if result_data.contains("Successfully edited") {
                if let Some(path) = result_data.strip_prefix("Successfully edited ") {
                    let path = path.trim();
                    let short: String = path
                        .rsplit('/')
                        .take(2)
                        .collect::<Vec<_>>()
                        .into_iter()
                        .rev()
                        .collect::<Vec<_>>()
                        .join("/");
                    return short;
                }
            }
            String::new()
        }
        _ => String::new(),
    }
}

async fn prewarm_remote_lms_models(config: &Config, main_model: &str) {
    let base = config.agents.defaults.local_api_base.trim();
    if base.is_empty() {
        return;
    }
    let native = base.trim_end_matches('/').trim_end_matches("/v1");
    let url = format!("{}/api/v1/models/load", native);

    let mut models: Vec<(String, Option<usize>)> = Vec::new();
    if !main_model.trim().is_empty() {
        models.push((
            main_model.trim().to_string(),
            Some(config.agents.defaults.local_max_context_tokens),
        ));
    }

    let role_models_enabled = config.trio.enabled
        || commands::should_auto_activate_trio(
            true,
            &config.trio.router_model,
            &config.trio.specialist_model,
            config.trio.router_endpoint.is_some(),
            config.trio.specialist_endpoint.is_some(),
            &config.tool_delegation.mode,
        );

    if role_models_enabled {
        if !config.trio.router_model.trim().is_empty() {
            models.push((
                config.trio.router_model.trim().to_string(),
                Some(config.trio.router_ctx_tokens),
            ));
        }
        if !config.trio.specialist_model.trim().is_empty() {
            models.push((
                config.trio.specialist_model.trim().to_string(),
                Some(config.trio.specialist_ctx_tokens),
            ));
        }
    }

    if config.lcm.enabled {
        if let Some(ref ep) = config.lcm.compaction_endpoint {
            if !ep.model.trim().is_empty() {
                models.push((ep.model.trim().to_string(), Some(config.lcm.compaction_context_size)));
            }
        }
    }

    let mut seen = BTreeSet::new();
    let client = reqwest::Client::new();
    for (model, ctx) in models {
        if !seen.insert(model.clone()) {
            continue;
        }
        let mut body = serde_json::json!({ "model": model });
        if let Some(c) = ctx {
            body["context_length"] = serde_json::json!(c);
        }
        match client.post(&url).json(&body).send().await {
            Ok(resp) if resp.status().is_success() => {
                info!(model = %body["model"].as_str().unwrap_or(""), "remote_lms_prewarm_ok");
            }
            Ok(resp) => {
                let status = resp.status();
                let text = resp.text().await.unwrap_or_default();
                warn!(model = %body["model"].as_str().unwrap_or(""), %status, body = %text, "remote_lms_prewarm_failed");
            }
            Err(e) => {
                warn!(model = %body["model"].as_str().unwrap_or(""), error = %e, "remote_lms_prewarm_error");
            }
        }
    }
}

/// Parse a `/ctx` argument into a byte count.
///
/// Accepts:
/// - `""` (empty) → None (means auto-detect)
/// - `"32768"` → Some(32768)
/// - `"32K"` or `"32k"` → Some(32768)
/// - Values < 2048 → Err
/// - Non-numeric → Err
pub(crate) fn parse_ctx_arg(arg: &str) -> Result<Option<usize>, &'static str> {
    let s = arg.trim();
    if s.is_empty() {
        return Ok(None);
    }
    let lower = s.to_lowercase();
    let n = if let Some(prefix) = lower.strip_suffix('k') {
        prefix
            .parse::<usize>()
            .map(|n| n * 1024)
            .map_err(|_| "invalid number")?
    } else {
        lower.parse::<usize>().map_err(|_| "invalid number")?
    };
    if n < 2048 {
        return Err("minimum context size is 2048");
    }
    // Round down to nearest 1024
    Ok(Some((n / 1024) * 1024))
}

/// Shorten channel names for status display.
pub(crate) fn short_channel_name(name: &str) -> &str {
    match name {
        "whatsapp" => "wa",
        "telegram" => "tg",
        other => other,
    }
}

/// Build the REPL prompt string based on current mode.
pub(crate) fn build_prompt(is_local: bool, voice_on: bool, thinking_on: bool) -> String {
    let think_prefix = if thinking_on {
        "\x1b[2m\u{1f9e0}\x1b[0m"
    } else {
        ""
    };
    if voice_on {
        format!(
            "{}{}{}~>{} ",
            think_prefix,
            crate::tui::BOLD,
            crate::tui::MAGENTA,
            crate::tui::RESET
        )
    } else if is_local {
        format!(
            "{}{}{}L>{} ",
            think_prefix,
            crate::tui::BOLD,
            crate::tui::YELLOW,
            crate::tui::RESET
        )
    } else {
        format!(
            "{}{}{}>{} ",
            think_prefix,
            crate::tui::BOLD,
            crate::tui::GREEN,
            crate::tui::RESET
        )
    }
}

/// Print the /help text.
pub(crate) fn print_help() {
    println!("\nCommands:");
    println!("  /local, /l      - Toggle between local and cloud mode");
    println!("  /model, /m [q]  - Pick model from all sources (LMS, cluster, ~/models/)");
    println!("  /trio           - Toggle trio mode (router + specialist)");
    println!("  /trio budget    - Show VRAM budget breakdown");
    println!("  /trio cap <GB>  - Set VRAM cap (e.g. /trio cap 12)");
    println!("  /ctx [size]     - Set context size (e.g. /ctx 32K) or auto-detect");
    println!("  /think, /t      - Toggle extended thinking (/thinking on|off|N)");
    println!("  /voice, /v      - Toggle voice mode (Ctrl+Space or Enter to speak)");
    println!("  /whatsapp, /wa  - Start WhatsApp channel (runs alongside chat)");
    println!("  /telegram, /tg  - Start Telegram channel (runs alongside chat)");
    println!("  /email          - Start Email channel (runs alongside chat)");

    println!("  /stop           - Stop all running channels");
    println!("  /agents, /a     - List running background agents");
    println!("  /kill <id>      - Cancel a background agent");
    println!("  /status, /s     - Show current mode, model, and channel info");
    println!("  /context        - Show context breakdown (tokens, messages, memory)");
    println!("  /memory         - Show working memory for current session");
    println!("  /clear, /c      - Clear working memory for current session");
    println!("  /replay         - Show session message history (/replay full | /replay N)");
    println!("  /restart, /rd   - Restart local servers (or delegation in cloud mode)");
    println!("  /sessions, /ss  - Session management (list, export, purge, archive, index)");
    println!("  /audit          - Show audit log for current session");
    println!("  /verify         - Re-verify claims in last response");
    println!("  /provenance     - Toggle provenance display on/off");
    println!("  /cluster, /cl   - Show cluster peers, models, and routing status");
    println!("  /help, /h       - Show this help");
    println!("  Ctrl+C          - Exit\n");
}

// ============================================================================
// Input Watcher (Full-Duplex REPL)
// ============================================================================

/// Spawn a key watcher thread that runs during agent streaming/tool execution.
///
/// Handles:
/// - **Enter**: cancel via `cancel_token` + set `enter_interrupted` flag.
///   In voice mode this signals "start recording"; in text mode it's a fast cancel.
/// - **ESC+ESC** (within 500ms): instant cancel via `cancel_token`
/// - **Ctrl+C**: backup cancel via `cancel_token`
/// - **Backtick (`)**: temporarily exits raw mode, reads an injection line
///   from stdin, sends it through `inject_tx`, re-enters raw mode
///
/// The thread exits when `done` is set to `true`.
///
/// Modeled on `tui::spawn_interrupt_watcher()` (voice mode pattern).
pub(crate) fn spawn_input_watcher(
    cancel_token: tokio_util::sync::CancellationToken,
    inject_tx: tokio::sync::mpsc::UnboundedSender<String>,
    done: Arc<AtomicBool>,
    enter_interrupted: Arc<AtomicBool>,
) -> std::thread::JoinHandle<()> {
    use termimad::crossterm::event::{self, Event, KeyCode, KeyModifiers};
    

    std::thread::spawn(move || {
        let owned = tui::enter_raw_mode();
        let mut last_esc: Option<Instant> = None;

        while !done.load(Ordering::Relaxed) {
            if event::poll(Duration::from_millis(100)).unwrap_or(false) {
                if let Ok(Event::Key(key)) = event::read() {
                    // Enter → cancel + signal "user wants to input/record"
                    if key.code == KeyCode::Enter {
                        enter_interrupted.store(true, Ordering::Relaxed);
                        cancel_token.cancel();
                        break;
                    }

                    // Ctrl+C → cancel
                    if key.code == KeyCode::Char('c')
                        && key.modifiers.contains(KeyModifiers::CONTROL)
                    {
                        cancel_token.cancel();
                        break;
                    }

                    // ESC double-tap → cancel
                    if key.code == KeyCode::Esc {
                        if let Some(prev) = last_esc {
                            if prev.elapsed() < Duration::from_millis(2000) {
                                cancel_token.cancel();
                                break;
                            }
                        }
                        last_esc = Some(Instant::now());
                        continue;
                    }

                    // Backtick → inject prompt
                    if key.code == KeyCode::Char('`')
                        && !key.modifiers.contains(KeyModifiers::CONTROL)
                        && !key.modifiers.contains(KeyModifiers::ALT)
                    {
                        // Exit raw mode so the user gets normal line editing.
                        tui::exit_raw_mode(owned);
                        print!("\n\x1b[33minject>\x1b[0m ");
                        io::stdout().flush().ok();

                        let mut line = String::new();
                        if io::stdin().read_line(&mut line).is_ok() {
                            let trimmed = line.trim().to_string();
                            if !trimmed.is_empty() {
                                let _ = inject_tx.send(trimmed);
                            }
                        }

                        // Re-enter raw mode for continued watching.
                        // Note: we don't update `owned` here because we already own the mode.
                        tui::enter_raw_mode();
                        continue;
                    }

                    // Any other key clears the ESC state.
                    last_esc = None;
                }
            }
        }

        tui::exit_raw_mode(owned);
    })
}

/// Stream an LLM response with live delta printing, then erase and re-render with syntax highlighting.
///
/// This replaces the 3x copy-pasted pattern:
///   create delta channel → spawn print task → stream → await → erase raw → re-render
///
/// When provenance is enabled, tool call events are displayed during streaming
/// and claim verification is applied to the final render.
///
/// Returns the full response text.
pub(crate) async fn stream_and_render(
    agent_loop: &mut crate::agent::agent_loop::AgentLoop,
    input: &str,
    session_id: &str,
    channel: &str,
    lang: Option<&str>,
    core_handle: &SharedCoreHandle,
) -> String {
    stream_and_render_inner(
        agent_loop,
        input,
        session_id,
        channel,
        lang,
        core_handle,
        false,
        None,
    )
    .await
    .0
}

/// Like `stream_and_render` but skips the user text erase-and-reprint.
/// Use when the caller has already rendered the user turn (e.g. voice recording).
///
/// Returns `(response_text, enter_interrupted)`. When `enter_interrupted` is true,
/// the user pressed Enter to cancel — the voice loop should skip TTS and start recording.
#[cfg(feature = "voice")]
pub(crate) async fn stream_and_render_voice(
    agent_loop: &mut crate::agent::agent_loop::AgentLoop,
    input: &str,
    session_id: &str,
    channel: &str,
    lang: Option<&str>,
    core_handle: &SharedCoreHandle,
    tts_sentence_tx: Option<std::sync::mpsc::Sender<crate::voice::TtsCommand>>,
) -> (String, bool) {
    stream_and_render_inner(
        agent_loop,
        input,
        session_id,
        channel,
        lang,
        core_handle,
        true,
        tts_sentence_tx,
    )
    .await
}

async fn stream_and_render_inner(
    agent_loop: &mut crate::agent::agent_loop::AgentLoop,
    input: &str,
    session_id: &str,
    channel: &str,
    lang: Option<&str>,
    core_handle: &SharedCoreHandle,
    user_already_rendered: bool,
    tts_tx: TtsSentenceSender,
) -> (String, bool) {
    // Erase raw readline output and reprint user text in grey box (skip if caller already rendered).
    if !user_already_rendered && std::io::stdout().is_terminal() {
        use std::io::Write as _;
        let prompt_and_input = format!("> {}", input);
        let raw_lines = tui::terminal_rows(&prompt_and_input, 0);
        print!("\x1b[{}A", raw_lines);
        for _ in 0..raw_lines {
            print!("\x1b[2K\r\n");
        }
        print!("\x1b[{}A", raw_lines);
        std::io::stdout().flush().ok();
        print!("{}", syntax::render_turn(input, syntax::TurnRole::User));
    }

    // Check provenance config for tool event display and claim verification.
    let (show_tool_calls, verify_claims, strict_mode, workspace) = {
        let core = core_handle.swappable();
        (
            core.provenance_config.enabled && core.provenance_config.show_tool_calls,
            core.provenance_config.enabled && core.provenance_config.verify_claims,
            core.provenance_config.strict_mode,
            core.workspace.clone(),
        )
    };

    let (delta_tx, mut delta_rx) = tokio::sync::mpsc::unbounded_channel::<String>();

    // Create tool event channel when provenance is enabled.
    let (tool_rx_opt, tool_event_tx) = if show_tool_calls {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<ToolEvent>();
        (Some(rx), Some(tx))
    } else {
        (None, None)
    };

    // Spawn unified print task: incremental renderer for text deltas,
    // tool events interleaved via clear_partial/restore_partial.
    let has_tool_rx = tool_rx_opt.is_some();
    let mut tool_rx_opt = tool_rx_opt;
    let print_task = tokio::spawn(async move {
        use std::io::Write as _;
        #[cfg(feature = "voice")]
        let mut tts_acc = tts_tx.map(|tx| crate::voice::SentenceAccumulator::new_streaming(tx));
        #[cfg(not(feature = "voice"))]
        let _ = tts_tx;

        let mut renderer = incremental::IncrementalRenderer::new();
        let mut full_text = String::new();
        let mut tool_lines = 0usize;
        let mut collected: Vec<String> = Vec::new();
        let mut delta_done = false;
        let mut tool_done = !has_tool_rx;
        // Track previous CallEnd to coalesce repeated status-check calls.
        let mut prev_call_end: Option<(String, usize)> = None;

        loop {
            if delta_done && tool_done {
                break;
            }
            tokio::select! {
                biased;
                delta = delta_rx.recv(), if !delta_done => {
                    match delta {
                        Some(d) => {
                            full_text.push_str(&d);
                            renderer.push(&d);
                            #[cfg(feature = "voice")]
                            if let Some(ref mut acc) = tts_acc {
                                acc.push(&d);
                            }
                        }
                        None => {
                            delta_done = true;
                            renderer.finish();
                            #[cfg(feature = "voice")]
                            if let Some(acc) = tts_acc.take() {
                                acc.flush();
                            }
                        },
                    }
                }
                event = async {
                    match tool_rx_opt {
                        Some(ref mut rx) => rx.recv().await,
                        None => std::future::pending().await,
                    }
                }, if !tool_done => {
                    match event {
                        Some(ToolEvent::CallStart { ref tool_name, ref arguments_preview, .. }) => {
                            renderer.flush_pending();
                            renderer.clear_partial();
                            renderer.emit_marker();
                            let line = format!(
                                "\x1b[36m  \u{25b6} {}({})\x1b[0m",
                                tool_name, arguments_preview
                            );
                            print!("\r{}\x1b[K", line);
                            std::io::stdout().flush().ok();
                            renderer.restore_partial();
                        }
                        Some(ToolEvent::Progress { ref tool_name, elapsed_ms, ref output_preview, .. }) => {
                            renderer.flush_pending();
                            renderer.clear_partial();
                            renderer.emit_marker();
                            let preview_str = output_preview.as_deref().unwrap_or("");
                            let line = format!(
                                "\x1b[36m  \u{25b6} {}\x1b[0m  \x1b[2m{}s{}\x1b[0m",
                                tool_name,
                                elapsed_ms / 1000,
                                if preview_str.is_empty() {
                                    String::new()
                                } else {
                                    format!(" {}", preview_str)
                                }
                            );
                            print!("\r\x1b[K{}", line);
                            std::io::stdout().flush().ok();
                            renderer.restore_partial();
                        }
                        Some(ToolEvent::CallEnd { ref tool_name, ok, duration_ms, ref result_data, .. }) => {
                            renderer.flush_pending();
                            renderer.clear_partial();
                            renderer.emit_marker();
                            // Coalesce repeated CallEnd for the same tool.
                            if let Some((ref prev_name, prev_lines)) = prev_call_end {
                                if prev_name == tool_name && prev_lines > 0 {
                                    print!("\x1b[{}A", prev_lines);
                                    for _ in 0..prev_lines { print!("\x1b[2K\r\n"); }
                                    print!("\x1b[{}A", prev_lines);
                                    std::io::stdout().flush().ok();
                                    tool_lines = tool_lines.saturating_sub(prev_lines);
                                    let keep = collected.len().saturating_sub(prev_lines);
                                    collected.truncate(keep);
                                }
                            }

                            let marker = if ok { "\x1b[32m\u{2713}\x1b[0m" } else { "\x1b[31m\u{2717}\x1b[0m" };
                            // Extract a short context label from result data.
                            let context = extract_tool_context(tool_name, result_data);
                            let status_line = if context.is_empty() {
                                format!(
                                    "\x1b[36m  \u{25b6} {}\x1b[0m  {} \x1b[2m{}ms\x1b[0m",
                                    tool_name, marker, duration_ms
                                )
                            } else {
                                format!(
                                    "\x1b[36m  \u{25b6} {}\x1b[0m \x1b[2m{}\x1b[0m  {} \x1b[2m{}ms\x1b[0m",
                                    tool_name, context, marker, duration_ms
                                )
                            };
                            println!("\r\x1b[K{}", status_line);
                            let mut this_box_lines = 1usize;
                            collected.push(status_line);

                            if ok && !result_data.is_empty() {
                                let truncated = truncate_output(result_data, 40, 2000);
                                if !truncated.is_empty() {
                                    let header = "    \x1b[2m\u{250c}\u{2500} output \u{2500}\x1b[0m";
                                    println!("\r{}", header);
                                    collected.push(header.to_string());
                                    this_box_lines += 1;
                                    for line in truncated.lines() {
                                        let formatted = format!("    \x1b[2m\u{2502}\x1b[0m {}", line);
                                        println!("\r{}", formatted);
                                        collected.push(formatted);
                                        this_box_lines += 1;
                                    }
                                    let footer = "    \x1b[2m\u{2514}\u{2500}\x1b[0m";
                                    println!("\r{}", footer);
                                    collected.push(footer.to_string());
                                    this_box_lines += 1;
                                }
                            } else if !ok && !result_data.is_empty() {
                                let preview: String = result_data.chars().take(80).collect();
                                let err_line = format!("    \x1b[2m\x1b[31m{}\x1b[0m", preview);
                                println!("\r{}", err_line);
                                collected.push(err_line);
                                this_box_lines += 1;
                            }
                            tool_lines += this_box_lines;
                            prev_call_end = Some((tool_name.clone(), this_box_lines));
                            std::io::stdout().flush().ok();
                            renderer.restore_partial();
                        }
                        None => tool_done = true,
                    }
                }
                _ = tokio::time::sleep(Duration::from_secs(1)), if !delta_done => {
                    renderer.tick();
                }
            }
        }
        // Use \r\n — raw mode (input watcher) makes \n LF-only.
        print!("\r\n");
        std::io::stdout().flush().ok();
        (tool_lines, collected)
    });

    println!();

    // Full-duplex input watcher: handles Enter (cancel + record), ESC+ESC (cancel),
    // Ctrl+C (cancel), and backtick (priority injection) during streaming/tool execution.
    let cancel_token = tokio_util::sync::CancellationToken::new();
    let (inject_tx, inject_rx) = tokio::sync::mpsc::unbounded_channel();
    let watcher_done = Arc::new(AtomicBool::new(false));
    let enter_interrupted = Arc::new(AtomicBool::new(false));

    let watcher = spawn_input_watcher(
        cancel_token.clone(),
        inject_tx,
        watcher_done.clone(),
        enter_interrupted.clone(),
    );

    let response = agent_loop
        .process_direct_streaming(
            input,
            session_id,
            channel,
            "direct",
            lang,
            delta_tx,
            tool_event_tx,
            Some(cancel_token.clone()),
            Some(inject_rx),
        )
        .await;

    // Signal watcher thread to stop and wait for it.
    watcher_done.store(true, Ordering::Relaxed);
    watcher.join().ok();
    // Defensive: ensure raw mode is off even if watcher thread panicked.
    tui::force_exit_raw_mode();
    // Flush any leftover keystrokes (e.g. rapid Esc presses) so they don't
    // leak into rustyline as partial ANSI escape sequences, which would hang.
    tui::drain_stdin();

    let cancelled = cancel_token.is_cancelled();
    let (_tool_lines, _tool_event_lines) = print_task.await.unwrap_or((0, Vec::new()));

    // Response was already rendered incrementally by IncrementalRenderer.
    // Only clean up trailing blank line and optionally append provenance footer.
    if !response.is_empty() && std::io::stdout().is_terminal() {
        use std::io::Write as _;
        // Erase the trailing \r\n from the print task (1 line).
        print!("\r\x1b[1A\x1b[2K");
        std::io::stdout().flush().ok();

        // Show redaction warning if strict mode removed fabricated claims.
        let redaction_count = response.matches("[unverified claim removed]").count();
        if redaction_count > 0 {
            println!(
                "\x1b[33m\x1b[1m  \u{26a0} {} claim(s) could not be verified against tool outputs and were redacted.\x1b[0m\n",
                redaction_count
            );
        }

        // Provenance: append claim summary footer (no full re-render — text
        // was already printed incrementally by IncrementalRenderer).
        if verify_claims {
            let audit = AuditLog::new(&workspace, session_id);
            let entries = audit.get_entries();
            let verifier = ClaimVerifier::new(&entries);
            let annotated = verifier.verify(&response);
            let claims: Vec<(usize, usize, u8, String)> = annotated
                .iter()
                .map(|c| {
                    let status = match c.status {
                        ClaimStatus::Observed => 0u8,
                        ClaimStatus::Derived => 1,
                        ClaimStatus::Claimed => 2,
                        ClaimStatus::Recalled => 3,
                    };
                    (c.span.0, c.span.1, status, c.text.clone())
                })
                .collect();
            print!("{}", syntax::render_provenance_footer(&claims, strict_mode));
        }
    }

    let was_enter = enter_interrupted.load(Ordering::Relaxed);
    if cancelled {
        if was_enter {
            // Enter-interrupt: user wants to take over. Brief marker.
            println!("\n  \x1b[2mInterrupted.\x1b[0m");
        } else {
            println!("\n  \x1b[33mCancelled.\x1b[0m");
        }
    }

    (response, was_enter)
}

// ============================================================================
// Server Lifecycle (DRY: replaces 6x copy-pasted spawn+wait+rebuild patterns)
// ============================================================================

/// Which inference engine backend is managing the local server.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum InferenceEngine {
    /// No local engine active.
    None,
    /// LM Studio via `lms` CLI (daemon mode).
    Lms,
}

pub(crate) struct ServerState {
    pub local_port: String,
    /// True when LM Studio's `lms` CLI manages the server lifecycle.
    pub lms_managed: bool,
    /// Path to the `lms` binary (set when lms_managed is true).
    pub lms_binary: Option<std::path::PathBuf>,
    /// Which inference engine is currently active.
    pub engine: InferenceEngine,
}

impl ServerState {
    pub fn new(port: String) -> Self {
        Self {
            local_port: port,
            lms_managed: false,
            lms_binary: None,
            engine: InferenceEngine::None,
        }
    }

    /// Unload models from the current LMS-managed server.
    pub async fn kill_current(&mut self, lms_port: u16) {
        if self.lms_managed {
            crate::lms::unload_all(lms_port).await.ok();
        }
        self.engine = InferenceEngine::None;
    }

    /// Full shutdown: stop LM Studio server.
    pub fn shutdown(&mut self) {
        if self.lms_managed {
            if let Some(ref bin) = self.lms_binary {
                println!("Stopping LM Studio server...");
                crate::lms::server_stop(bin).ok();
            }
            self.lms_managed = false;
        }
        self.engine = InferenceEngine::None;
    }
}

/// Resolve which inference engine to use based on config preference.
///
/// Returns `(engine_kind, binary_path)` for the first available engine.
/// Currently only LM Studio is supported.
pub(crate) fn resolve_inference_engine(
    _preference: &str,
) -> Option<(InferenceEngine, std::path::PathBuf)> {
    crate::lms::find_lms_binary().map(|b| (InferenceEngine::Lms, b))
}

/// Rebuild the agent core and agent loop after a server change.
///
/// Call this after a mode switch or config update.
pub(crate) fn apply_server_change(
    state: &ServerState,
    model_path: &std::path::Path,
    core_handle: &SharedCoreHandle,
    config: &Config,
    is_local: bool,
) {
    // Prefer lms_main_model which preserves namespace prefixes like "qwen/qwen3-vl-8b".
    // PathBuf::file_name() strips parent components, breaking namespaced model identifiers.
    let model_name = if !config.agents.defaults.lms_main_model.is_empty() {
        Some(config.agents.defaults.lms_main_model.as_str())
    } else {
        model_path.file_name().and_then(|n| n.to_str())
    };
    cli::rebuild_core(
        core_handle,
        config,
        &state.local_port,
        model_name,
        None,
        None,
        None,
        is_local,
    );
}

// ============================================================================
// cmd_agent - Main REPL entry point
// ============================================================================

// Background channel state
pub(crate) struct ActiveChannel {
    pub name: String,
    pub stop: Arc<AtomicBool>,
    pub handle: tokio::task::JoinHandle<()>,
}

pub(crate) fn cmd_agent(
    message: Option<String>,
    session_id: String,
    local_flag: bool,
    lang: Option<String>,
) {
    let mut config = load_config(None);

    // Resolve voice language: CLI --lang > config voice.language > None (auto)
    let lang = lang.or_else(|| config.voice.language.clone());

    // Check environment variable for local mode
    let local_env = std::env::var("NANOBOT_LOCAL")
        .map(|v| v == "1" || v.to_lowercase() == "true")
        .unwrap_or(false);

    // Set initial local mode from flag, environment, or config (localApiBase).
    let has_remote_local = !config.agents.defaults.local_api_base.is_empty();
    let is_local = local_flag || local_env || has_remote_local;

    let local_port = std::env::var("NANOBOT_LOCAL_PORT").unwrap_or_else(|_| "8080".to_string());
    if !is_local {
        let api_key = config.get_api_key();
        let model = &config.agents.defaults.model;
        let has_prefix = config.resolve_provider_for_model(model).is_some();
        let has_oauth = dirs::home_dir()
            .map(|h| h.join(".claude").join(".credentials.json").exists())
            .unwrap_or(false);
        if api_key.is_none()
            && !has_prefix
            && !model.starts_with("bedrock/")
            && !model.starts_with("claude-max")
            && !has_oauth
        {
            eprintln!("Error: No API key configured.");
            eprintln!("Set one in ~/.nanobot/config.json under providers.openrouter.apiKey");
            eprintln!("Or authenticate with Claude CLI: claude login");
            eprintln!("Or use --local flag to use a local LLM server.");
            std::process::exit(1);
        }
    }

    let runtime = tokio::runtime::Runtime::new().expect("Failed to create tokio runtime");

    runtime.block_on(async {
        // Create shared core and initial agent loop.
        // Use persisted local model name if available, else hardcoded default.
        let mut local_model_name: String = if config.agents.defaults.local_model.is_empty() {
            server::DEFAULT_LOCAL_MODEL.to_string()
        } else {
            config.agents.defaults.local_model.clone()
        };

        // In local mode with single-message (-m), just start main server.
        // Trio is for interactive sessions - single messages use inline tools.
        // When localApiBase is set, skip all local server spawning — use remote server.
        let mut trio_state: Option<ServerState> = None;
        if !has_remote_local && is_local && message.is_some() {
            // Single-message local mode: start LMS if available.
            let mut srv = ServerState::new(local_port.clone());
            let preference = &config.agents.defaults.inference_engine;
            if let Some((InferenceEngine::Lms, bin)) = resolve_inference_engine(preference) {
                let lms_port = config.agents.defaults.lms_port;
                match crate::lms::server_start(&bin, lms_port).await {
                    Ok(()) => {
                        let available = crate::lms::list_available(lms_port).await;
                        let main_model = if !config.agents.defaults.lms_main_model.is_empty() {
                            config.agents.defaults.lms_main_model.clone()
                        } else {
                            let hint = cli::strip_gguf_suffix(&local_model_name);
                            crate::lms::resolve_model_name(&available, hint)
                        };
                        let main_ctx = Some(config.agents.defaults.local_max_context_tokens);
                        if let Err(e) = crate::lms::load_model(lms_port, &main_model, main_ctx).await {
                            eprintln!("Warning: lms load failed: {}", e);
                        } else {
                            local_model_name = main_model;
                            srv.lms_managed = true;
                            srv.lms_binary = Some(bin);
                            srv.local_port = lms_port.to_string();
                            let lms_host = crate::lms::api_host();
                            config.agents.defaults.local_api_base =
                                format!("http://{}:{}/v1", lms_host, lms_port);
                            config.agents.defaults.skip_jit_gate = true;
                        }

                        // Trio model loading for single-message mode.
                        let (auto_router, auto_specialist) =
                            commands::pick_trio_models(&available, &local_model_name);
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
                        if config.trio.enabled
                            || commands::should_auto_activate_trio(
                                is_local,
                                &config.trio.router_model,
                                &config.trio.specialist_model,
                                config.trio.router_endpoint.is_some(),
                                config.trio.specialist_endpoint.is_some(),
                                &config.tool_delegation.mode,
                            )
                        {
                            // Load router model if configured.
                            if !config.trio.router_model.is_empty() {
                                let _ = crate::lms::load_model(
                                    lms_port,
                                    &config.trio.router_model,
                                    Some(config.trio.router_ctx_tokens),
                                )
                                .await;
                            }
                            // Load specialist model if configured.
                            if !config.trio.specialist_model.is_empty() {
                                let _ = crate::lms::load_model(
                                    lms_port,
                                    &config.trio.specialist_model,
                                    Some(config.trio.specialist_ctx_tokens),
                                )
                                .await;
                            }
                            commands::trio_enable(&mut config);
                        }
                    }
                    Err(e) => {
                        eprintln!("Warning: lms server start failed: {}", e);
                    }
                }
            } else {
                eprintln!("Error: No local inference engine found. Install LM Studio (lms CLI).");
                std::process::exit(1);
            }
            trio_state = Some(srv);
        }

        // Interactive REPL: detect LMS and set config BEFORE building core,
        // so the initial core handle and SubagentManager get the right URL.
        let mut srv = ServerState::new(local_port.clone());
        let mut config = config; // shadow to allow mutation
        let is_interactive = message.is_none();
        if is_interactive && is_local && !has_remote_local {
            tui::register_resize_handler();
            tui::print_startup_splash(&local_port, is_local);

            let preference = &config.agents.defaults.inference_engine;
            if let Some((InferenceEngine::Lms, bin)) = resolve_inference_engine(preference) {
                let lms_port = config.agents.defaults.lms_port;
                println!(
                    "  {}{}LM Studio{} detected, starting server on port {}...",
                    tui::BOLD, tui::YELLOW, tui::RESET, lms_port
                );

                match crate::lms::server_start(&bin, lms_port).await {
                    Ok(()) => {
                        let available = crate::lms::list_available(lms_port).await;
                        let main_model = if !config.agents.defaults.lms_main_model.is_empty() {
                            config.agents.defaults.lms_main_model.clone()
                        } else {
                            let hint = cli::strip_gguf_suffix(&local_model_name);
                            crate::lms::resolve_model_name(&available, hint)
                        };
                        let main_ctx = Some(config.agents.defaults.local_max_context_tokens);
                        print!("  Loading {}... ", main_model);
                        io::stdout().flush().ok();
                        match crate::lms::load_model(lms_port, &main_model, main_ctx).await {
                            Ok(()) => println!("{}OK{}", tui::GREEN, tui::RESET),
                            Err(e) => println!("{}FAILED: {}{}", tui::RED, e, tui::RESET),
                        }

                        // Auto-detect trio roles from available models if not configured.
                        let (auto_router, auto_specialist) =
                            commands::pick_trio_models(&available, &main_model);
                        if config.trio.router_model.is_empty() {
                            if let Some(r) = auto_router {
                                config.trio.router_model = r;
                                info!(router = %config.trio.router_model, "trio_router_auto_detected");
                            }
                        }
                        if config.trio.specialist_model.is_empty() {
                            if let Some(s) = auto_specialist {
                                config.trio.specialist_model = s;
                                info!(specialist = %config.trio.specialist_model, "trio_specialist_auto_detected");
                            }
                        }

                        if config.trio.enabled {
                            if !config.trio.router_model.is_empty() {
                                print!("  Loading {}... ", config.trio.router_model);
                                io::stdout().flush().ok();
                                match crate::lms::load_model(lms_port, &config.trio.router_model, Some(config.trio.router_ctx_tokens)).await {
                                    Ok(()) => println!("{}OK{}", tui::GREEN, tui::RESET),
                                    Err(e) => println!("{}FAILED: {}{}", tui::RED, e, tui::RESET),
                                }
                            }
                            if !config.trio.specialist_model.is_empty() {
                                print!("  Loading {}... ", config.trio.specialist_model);
                                io::stdout().flush().ok();
                                match crate::lms::load_model(lms_port, &config.trio.specialist_model, Some(config.trio.specialist_ctx_tokens)).await {
                                    Ok(()) => println!("{}OK{}", tui::GREEN, tui::RESET),
                                    Err(e) => println!("{}FAILED: {}{}", tui::RED, e, tui::RESET),
                                }
                            }
                        }

                        // Load LCM compaction model when configured.
                        if config.lcm.enabled {
                            if let Some(ref ep) = config.lcm.compaction_endpoint {
                                print!("  Loading {} (LCM compactor)... ", ep.model);
                                io::stdout().flush().ok();
                                match crate::lms::load_model(lms_port, &ep.model, Some(config.lcm.compaction_context_size)).await {
                                    Ok(()) => println!("{}OK{}", tui::GREEN, tui::RESET),
                                    Err(e) => println!("{}FAILED: {}{}", tui::RED, e, tui::RESET),
                                }
                            }
                        }

                        local_model_name = main_model;
                        srv.lms_managed = true;
                        srv.lms_binary = Some(bin);
                        srv.local_port = lms_port.to_string();
                        let lms_host = crate::lms::api_host();
                        config.agents.defaults.local_api_base =
                            format!("http://{}:{}/v1", lms_host, lms_port);
                        config.agents.defaults.skip_jit_gate = true;
                    }
                    Err(e) => {
                        println!(
                            "  {}{}lms server start failed:{} {}",
                            tui::BOLD, tui::YELLOW, tui::RESET, e
                        );
                    }
                }
            }
        }

        // Recompute has_remote_local after potential lms setup
        let mut has_remote_local = !config.agents.defaults.local_api_base.is_empty();

        // --- Offline cluster peer fallback ---
        // If the saved endpoint is a remote peer we didn't start, probe it.
        // On failure: try starting local LMS, or clear the dead endpoint.
        if is_local && has_remote_local && !srv.lms_managed {
            let peer_port = config
                .agents
                .defaults
                .local_api_base
                .split(':')
                .last()
                .and_then(|p| p.split('/').next())
                .and_then(|p| p.parse::<u16>().ok())
                .unwrap_or(18080);

            let probe = crate::lms::list_available(peer_port).await;
            if probe.is_empty() {
                let dead_url = config.agents.defaults.local_api_base.clone();
                println!(
                    "  {}{}Saved endpoint {} is unreachable.{}",
                    tui::BOLD,
                    tui::YELLOW,
                    dead_url,
                    tui::RESET,
                );

                let preference = &config.agents.defaults.inference_engine;
                let mut fell_back = false;
                if let Some((InferenceEngine::Lms, bin)) = resolve_inference_engine(preference) {
                    let lms_port = config.agents.defaults.lms_port;
                    println!(
                        "  Attempting local LM Studio fallback on port {}...",
                        lms_port
                    );
                    if let Ok(()) = crate::lms::server_start(&bin, lms_port).await {
                        let available = crate::lms::list_available(lms_port).await;
                        if !available.is_empty() {
                            let model = if !config.agents.defaults.lms_main_model.is_empty() {
                                config.agents.defaults.lms_main_model.clone()
                            } else {
                                let hint = cli::strip_gguf_suffix(&local_model_name);
                                crate::lms::resolve_model_name(&available, hint)
                            };
                            let ctx =
                                Some(config.agents.defaults.local_max_context_tokens);
                            print!("  Loading {}... ", model);
                            io::stdout().flush().ok();
                            match crate::lms::load_model(lms_port, &model, ctx).await {
                                Ok(()) => {
                                    println!("{}OK{}", tui::GREEN, tui::RESET);
                                    local_model_name = model;
                                    srv.lms_managed = true;
                                    srv.lms_binary = Some(bin);
                                    srv.local_port = lms_port.to_string();
                                    let lms_host = crate::lms::api_host();
                                    config.agents.defaults.local_api_base =
                                        format!("http://{}:{}/v1", lms_host, lms_port);
                                    config.agents.defaults.skip_jit_gate = true;
                                    fell_back = true;
                                    println!(
                                        "  {}Fell back to local LM Studio.{}",
                                        tui::GREEN, tui::RESET
                                    );
                                }
                                Err(e) => {
                                    println!("{}FAILED: {}{}", tui::RED, e, tui::RESET);
                                }
                            }
                        }
                    }
                }

                if !fell_back {
                    // Clear the dead endpoint so the user isn't stuck
                    config.agents.defaults.local_api_base.clear();
                    let mut disk_cfg = load_config(None);
                    disk_cfg.agents.defaults.local_api_base.clear();
                    save_config(&disk_cfg, None);
                    has_remote_local = false;
                    println!(
                        "  Cleared dead endpoint. Use {}/m{} to pick a model when a peer comes online.",
                        tui::BOLD, tui::RESET,
                    );
                }
            }
        }

        // Remote LM Studio base: proactively prewarm main/router/specialist models
        // to avoid first-turn latency spikes from JIT loading.
        if is_local && has_remote_local && !srv.lms_managed {
            prewarm_remote_lms_models(&config, &local_model_name).await;
        }

        // Auto-activate trio mode for local sessions when both router and
        // specialist models are configured.  The downgrade block below will
        // revert strict flags if the router turns out to be unreachable.
        if commands::should_auto_activate_trio(
            is_local,
            &config.trio.router_model,
            &config.trio.specialist_model,
            config.trio.router_endpoint.is_some(),
            config.trio.specialist_endpoint.is_some(),
            &config.tool_delegation.mode,
        ) {
            commands::trio_enable(&mut config);
            info!(
                delegation_mode = ?config.tool_delegation.mode,
                router_model = %config.trio.router_model,
                specialist_model = %config.trio.specialist_model,
                "trio_auto_activated"
            );
        }

        // When no trio router is available, disable strict mode so the single model
        // can handle tools directly. Must happen BEFORE build_core_handle so the core
        // gets the updated tool_delegation_config.
        if config.tool_delegation.strict_no_tools_main
            && config.tool_delegation.strict_router_schema
        {
            let router_available = if srv.lms_managed || has_remote_local {
                // For both managed (started by nanobot) and remote LM Studio,
                // verify the model is actually loaded via list_available()
                let lms_port = if srv.lms_managed {
                    config.agents.defaults.lms_port
                } else {
                    // Extract port from local_api_base (e.g. "http://localhost:18080/v1")
                    config.agents.defaults.local_api_base
                        .split(':')
                        .last()
                        .and_then(|p| p.split('/').next())
                        .and_then(|p| p.parse::<u16>().ok())
                        .unwrap_or(18080)
                };
                let available = crate::lms::list_available(lms_port).await;
                crate::lms::is_model_available(&available, &config.trio.router_model)
            } else {
                false
            };

            if !router_available {
                info!("trio_downgrade: router not available, clearing strict flags");
                config.tool_delegation.strict_no_tools_main = false;
                config.tool_delegation.strict_router_schema = false;
            }
        }

        info!(
            delegation_mode = ?config.tool_delegation.mode,
            strict_no_tools_main = config.tool_delegation.strict_no_tools_main,
            strict_router_schema = config.tool_delegation.strict_router_schema,
            is_local,
            "delegation_config_at_core_build"
        );

        let core_handle = cli::build_core_handle(
            &config,
            &srv.local_port,
            Some(&local_model_name),
            None,
            None,
            None,
            is_local,
        );
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

        // Channel for subagents/background gateways to send display lines to the REPL.
        let (display_tx, display_rx) = mpsc::unbounded_channel::<String>();

        let health_registry = std::sync::Arc::new(crate::heartbeat::health::build_registry(&config));

        let mut agent_loop = cli::create_agent_loop(
            core_handle.clone(),
            &config,
            Some(cron_service.clone()),
            email_config.clone(),
            Some(display_tx.clone()),
            Some(health_registry.clone()),
        );

        // Set up cluster discovery for the REPL path (feature-gated).
        // Returns the ClusterState so /cluster commands can query it.
        #[cfg(feature = "cluster")]
        let cluster_state = cli::setup_cluster_for_repl(&mut agent_loop, &config);

        if let Some(msg) = message {
            // Single-message mode: process and exit.
            // Keep trio servers alive during processing (they're dropped at end of scope).
            let _servers = &trio_state;
            let mut agent_loop = agent_loop;
            stream_and_render(
                &mut agent_loop,
                &msg,
                &session_id,
                "cli",
                None,
                &core_handle,
            )
            .await;
            index_sessions_background();
        } else {
            // Interactive REPL mode.
            // Splash and LMS detection already happened above (before core build).
            if !is_local || has_remote_local {
                tui::register_resize_handler();
                tui::print_startup_splash(&local_port, is_local);
            }

            // Load persisted local model preference, falling back to hardcoded default.
            let default_model = {
                let models_dir = dirs::home_dir().unwrap().join("models");
                let saved = &config.agents.defaults.local_model;
                if !saved.is_empty() {
                    let saved_path = models_dir.join(saved);
                    if saved_path.exists() {
                        saved_path
                    } else {
                        models_dir.join(server::DEFAULT_LOCAL_MODEL)
                    }
                } else {
                    models_dir.join(server::DEFAULT_LOCAL_MODEL)
                }
            };

            // Readline editor with history
            let history_path = get_data_dir().join("history.txt");
            let mut rl = rustyline::DefaultEditor::new().expect("Failed to create line editor");

            // Alt+Enter inserts a newline (multi-line editing) instead of submitting.
            rl.bind_sequence(
                rustyline::KeyEvent(rustyline::KeyCode::Enter, rustyline::Modifiers::ALT),
                rustyline::Cmd::Newline,
            );

            // Build ReplContext — all mutable REPL state in one struct.
            let (restart_tx, restart_rx) = tokio::sync::mpsc::unbounded_channel();
            let mut ctx = commands::ReplContext {
                config,
                core_handle,
                agent_loop,
                session_id,
                lang,
                srv,
                current_model_path: default_model,
                active_channels: vec![],
                display_tx,
                display_rx,
                cron_service,
                email_config,
                rl: Some(rl),
                watchdog_handle: None,
                restart_tx: restart_tx.clone(),
                restart_rx,
                health_registry: Some(health_registry),
                #[cfg(feature = "voice")]
                voice_session: None,
                #[cfg(feature = "cluster")]
                cluster_state,
            };

            let _ = ctx.rl.as_mut().unwrap().load_history(&history_path);

            // JIT warmup: pre-load models on the remote JIT server (e.g. LM Studio).
            // This forces each model to load sequentially before any real requests,
            // avoiding concurrent model-switch crashes and cold-start latency.
            // Fires for any JIT server (localApiBase set), not just trio mode.
            if has_remote_local && !ctx.srv.lms_managed {
                use crate::providers::jit_gate::warmup_jit_models;

                let base = &ctx.config.agents.defaults.local_api_base;
                let mut models_to_warm: Vec<&str> = Vec::new();

                // Main model — always warm so first message is fast.
                let main_model_ref = if ctx.config.agents.defaults.local_model.is_empty() {
                    server::DEFAULT_LOCAL_MODEL
                } else {
                    &ctx.config.agents.defaults.local_model
                };
                let main_id = cli::strip_gguf_suffix(main_model_ref);
                models_to_warm.push(main_id);

                // Trio models (router + specialist) when enabled.
                if ctx.config.trio.enabled {
                    // Router: prefer explicit endpoint model, fall back to trio config.
                    if let Some(ref ep) = ctx.config.trio.router_endpoint {
                        models_to_warm.push(&ep.model);
                    } else if !ctx.config.trio.router_model.is_empty() {
                        models_to_warm.push(&ctx.config.trio.router_model);
                    }
                    // Specialist: prefer explicit endpoint model, fall back to trio config.
                    if let Some(ref ep) = ctx.config.trio.specialist_endpoint {
                        models_to_warm.push(&ep.model);
                    } else if !ctx.config.trio.specialist_model.is_empty() {
                        models_to_warm.push(&ctx.config.trio.specialist_model);
                    }
                }

                // LCM compaction model when configured.
                if let Some(ref ep) = ctx.config.lcm.compaction_endpoint {
                    if ctx.config.lcm.enabled {
                        models_to_warm.push(&ep.model);
                    }
                }

                print!(
                    "  {}Warming up {} model(s)...{} ",
                    tui::DIM,
                    models_to_warm.len(),
                    tui::RESET
                );
                io::stdout().flush().ok();

                warmup_jit_models(base, "local", &models_to_warm).await;
                println!("{}done{}", tui::DIM, tui::RESET);
            }

            // Start heartbeat: maintenance commands run on every tick (no LLM).
            let maintenance_cmds: Vec<String> = DEFAULT_MAINTENANCE_COMMANDS
                .iter()
                .map(|s| s.to_string())
                .collect();
            let heartbeat = HeartbeatService::new(
                ctx.config.workspace_path(),
                None, // No LLM callback — maintenance only
                maintenance_cmds,
                DEFAULT_HEARTBEAT_INTERVAL_S,
                true,
                ctx.health_registry.clone(),
            );
            heartbeat.start().await;

            // Start health watchdog for local servers (if any are running).
            // Skip when using a remote local server (e.g. LM Studio) — there is
            // no local server to monitor and the watchdog would spam the remote.
            if is_local && !has_remote_local {
                ctx.restart_watchdog();
            }

            // First bar render pushes content up to make room; all subsequent
            // renders refresh in place so we never get a duplicate bar.
            let mut bar_needs_push = true;

            loop {
                // Drain any pending display messages from background channels.
                ctx.drain_display();

                // Handle auto-restart requests from watchdog.
                ctx.handle_restart_requests().await;

                let is_local = ctx.core_handle.swappable().is_local;
                let voice_on = ctx.voice_on();
                let thinking_on = ctx
                    .core_handle
                    .counters
                    .thinking_budget
                    .load(Ordering::SeqCst)
                    > 0;
                let prompt = build_prompt(is_local, voice_on, thinking_on);

                // Render Claude Code-style input bar below the prompt line.
                let sa_count = ctx.agent_loop.subagent_manager().get_running_count().await;
                ctx.active_channels.retain(|ch| !ch.handle.is_finished());
                let ch_names: Vec<&str> = ctx
                    .active_channels
                    .iter()
                    .map(|c| short_channel_name(&c.name))
                    .collect();
                tui::render_input_bar(&ctx.core_handle, &ch_names, sa_count, bar_needs_push);
                bar_needs_push = false;

                // === GET INPUT ===
                let input_text: String;
                #[allow(unused_mut)]
                let mut do_record = false;

                #[cfg(feature = "voice")]
                if voice_on {
                    print!("{}", prompt);
                    io::stdout().flush().ok();
                    match tui::voice_read_input() {
                        tui::VoiceAction::Record => {
                            do_record = true;
                            input_text = String::new();
                        }
                        tui::VoiceAction::Text(t) => {
                            input_text = t;
                        }
                        tui::VoiceAction::Exit => break,
                    }
                } else {
                    match ctx.readline_async(&prompt).await {
                        Ok(line) => {
                            let _ = ctx.rl.as_mut().unwrap().add_history_entry(&line);
                            input_text = line;
                        }
                        Err(ReadlineError::Interrupted | ReadlineError::Eof) => break,
                        Err(_) => break,
                    }
                }

                #[cfg(not(feature = "voice"))]
                {
                    match ctx.readline_async(&prompt).await {
                        Ok(line) => {
                            let _ = ctx.rl.as_mut().unwrap().add_history_entry(&line);
                            input_text = line;
                        }
                        Err(ReadlineError::Interrupted | ReadlineError::Eof) => break,
                        Err(_) => break,
                    }
                }

                // Clear the input line and the prompt line above it (not the bar).
                // The scroll region stays intact so the bar remains pinned at bottom.
                // In voice mode, voice_read_input() prints \r\n leaving the ~> prompt
                // on the line above — move up and clear both lines.
                if do_record {
                    print!("\x1b[A\x1b[2K\x1b[2K\r"); // up, clear prompt line, clear current line
                } else {
                    print!("\x1b[2K\r"); // clear current line
                }
                io::stdout().flush().ok();

                // === VOICE RECORDING ===
                // Uses the same stream_and_render pipeline as text mode for
                // identical UI quality (syntax highlighting, provenance, status bar).
                // TTS plays after rendering completes.
                #[cfg(feature = "voice")]
                if do_record {
                    let mut keep_recording = true;
                    // Stop any ongoing playback.
                    if let Some(ref mut vs) = ctx.voice_session {
                        vs.stop_playback();
                    }
                    while keep_recording {
                        keep_recording = false;

                        // Phase 1: Record and transcribe (borrows vs briefly).
                        let transcription = ctx
                            .voice_session
                            .as_mut()
                            .and_then(|vs| vs.record_and_transcribe().transpose())
                            .transpose();

                        match transcription {
                            Ok(Some((text, detected_lang))) => {
                                let tts_lang_owned = ctx.lang.clone().unwrap_or(detected_lang);

                                // Render user text with purple ● marker.
                                print!(
                                    "{}",
                                    syntax::render_turn(&text, syntax::TurnRole::VoiceUser)
                                );

                                // Start streaming TTS pipeline BEFORE LLM call.
                                let tts_parts = ctx.voice_session.as_mut().and_then(|vs| {
                                    vs.clear_cancel();
                                    vs.start_streaming_speak(&tts_lang_owned, None).ok()
                                });
                                let (sentence_tx, join_handle) = match tts_parts {
                                    Some((tx, jh)) => (Some(tx), Some(jh)),
                                    None => (None, None),
                                };

                                // Phase 2: LLM call with parallel TTS feeding.
                                let (_response, enter_pressed) = stream_and_render_voice(
                                    &mut ctx.agent_loop,
                                    &text,
                                    &ctx.session_id,
                                    "voice",
                                    Some(&tts_lang_owned),
                                    &ctx.core_handle,
                                    sentence_tx,
                                )
                                .await;

                                ctx.drain_display();
                                println!();
                                ctx.print_status_bar().await;
                                println!(); // breathing room before next input bar

                                // Phase 3: Wait for TTS playback to finish.
                                if enter_pressed {
                                    // Enter-interrupt: skip TTS, start recording immediately.
                                    if let Some(ref mut vs) = ctx.voice_session {
                                        vs.stop_playback();
                                    }
                                    keep_recording = true;
                                } else if let Some(jh) = join_handle {
                                    let cancel = ctx
                                        .voice_session
                                        .as_ref()
                                        .map(|vs| vs.cancel_flag())
                                        .unwrap_or_else(|| Arc::new(AtomicBool::new(false)));
                                    let done = Arc::new(AtomicBool::new(false));
                                    let done2 = done.clone();
                                    let watcher =
                                        tui::spawn_interrupt_watcher(cancel.clone(), done2);
                                    let _ = jh.join(); // blocks until all audio played
                                    done.store(true, Ordering::Relaxed);
                                    let interrupted = watcher.join().unwrap_or(false);
                                    if interrupted {
                                        if let Some(ref mut vs) = ctx.voice_session {
                                            vs.stop_playback();
                                        }
                                        keep_recording = true;
                                    }
                                }
                            }
                            Ok(None) => println!("\x1b[2m(no speech detected)\x1b[0m"),
                            Err(e) => eprintln!("\x1b[31m{}\x1b[0m", e),
                        }
                    }
                    tui::drain_stdin();
                    continue;
                }

                // === TEXT INPUT ===
                let input = input_text.trim();
                if input.is_empty() {
                    continue;
                }

                // Dispatch slash commands.
                if input.starts_with('/') && ctx.dispatch(input).await {
                    continue;
                }

                // Process message (streaming)
                let channel = if voice_on { "voice" } else { "cli" };
                #[allow(unused_variables)]
                let response = stream_and_render(
                    &mut ctx.agent_loop,
                    input,
                    &ctx.session_id,
                    channel,
                    None,
                    &ctx.core_handle,
                )
                .await;
                ctx.drain_display();
                println!();

                // Refresh the input bar in place (no scroll push — just update content).
                let ch_names: Vec<&str> = ctx
                    .active_channels
                    .iter()
                    .map(|c| c.name.as_str())
                    .collect();
                let sa_count = ctx.agent_loop.subagent_manager().get_running_count().await;
                tui::render_input_bar(&ctx.core_handle, &ch_names, sa_count, false);

                #[cfg(feature = "voice")]
                if let Some(ref mut vs) = ctx.voice_session {
                    let tts_text = tui::strip_markdown_for_tts(&response);
                    if !tts_text.is_empty() {
                        tui::speak_interruptible(vs, &tts_text, "en");
                    }
                }
            }

            // Stop any active background channels
            for ch in &ctx.active_channels {
                ch.stop.store(true, Ordering::Relaxed);
            }
            if !ctx.active_channels.is_empty() {
                tokio::time::sleep(Duration::from_millis(500)).await;
                for ch in &ctx.active_channels {
                    ch.handle.abort();
                }
            }

            // Shutdown voice session first — leaks native TTS engines to
            // avoid C++ destructor segfault on exit.
            #[cfg(feature = "voice")]
            if let Some(vs) = ctx.voice_session.take() {
                vs.shutdown();
            }

            // Reset terminal: clear the pinned input bar and restore full scroll region
            // so the shell prompt returns to a clean state.
            {
                use std::io::Write as _;
                print!("\x1b[r"); // reset scroll region to full terminal
                let h = tui::terminal_height();
                print!("\x1b[{};1H", h); // move to last row
                print!("\x1b[J"); // clear from cursor to end of screen
                std::io::stdout().flush().ok();
            }

            // Cleanup: stop heartbeat, save readline history, kill servers
            heartbeat.stop().await;
            let _ = ctx.rl.as_mut().unwrap().save_history(&history_path);

            // Unload trio models so LM Studio returns to just the main model.
            // Run when trio is enabled and we have any LMS connection (managed or
            // user-started via localApiBase) — not only when nanobot started the server.
            if ctx.config.trio.enabled && (ctx.srv.lms_managed || has_remote_local) {
                let lms_port = if ctx.srv.lms_managed {
                    ctx.config.agents.defaults.lms_port
                } else {
                    // Extract port from localApiBase (e.g. "http://localhost:18080/v1")
                    ctx.config.agents.defaults.local_api_base
                        .split(':')
                        .last()
                        .and_then(|p| p.split('/').next())
                        .and_then(|p| p.parse::<u16>().ok())
                        .unwrap_or(ctx.config.agents.defaults.lms_port)
                };
                if !ctx.config.trio.router_model.is_empty() {
                    let _ = crate::lms::unload_model(lms_port, &ctx.config.trio.router_model).await;
                }
                if !ctx.config.trio.specialist_model.is_empty() {
                    let _ = crate::lms::unload_model(lms_port, &ctx.config.trio.specialist_model).await;
                }
            }

            ctx.srv.shutdown();

            // Re-index qmd sessions collection so the latest conversation is
            // immediately searchable via recall in the next session.
            index_sessions_background();

            println!("Goodbye!");
        }
    });
}

// ============================================================================
// Post-session indexing
// ============================================================================

/// Re-index the `qmd sessions` collection so the latest conversation is
/// searchable via the recall tool in the next session. Fire-and-forget:
/// errors are logged but never block shutdown.
fn index_sessions_background() {
    use std::process::Command;
    match Command::new("qmd").args(["update"]).output() {
        Ok(out) if out.status.success() => {
            debug!("qmd update completed (sessions re-indexed)");
        }
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            warn!("qmd update exited with {}: {}", out.status, stderr.trim());
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            // qmd not installed — silently skip.
        }
        Err(e) => {
            warn!("qmd update failed: {}", e);
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- parse_ctx_arg ---

    #[test]
    fn test_parse_ctx_arg_empty_returns_none() {
        assert_eq!(parse_ctx_arg("").unwrap(), None);
        assert_eq!(parse_ctx_arg("  ").unwrap(), None);
    }

    #[test]
    fn test_parse_ctx_arg_numeric() {
        assert_eq!(parse_ctx_arg("32768").unwrap(), Some(32768));
        assert_eq!(parse_ctx_arg("4096").unwrap(), Some(4096));
        assert_eq!(parse_ctx_arg("2048").unwrap(), Some(2048));
    }

    #[test]
    fn test_parse_ctx_arg_k_suffix() {
        assert_eq!(parse_ctx_arg("32K").unwrap(), Some(32768));
        assert_eq!(parse_ctx_arg("32k").unwrap(), Some(32768));
        assert_eq!(parse_ctx_arg("4K").unwrap(), Some(4096));
        assert_eq!(parse_ctx_arg("128k").unwrap(), Some(131072));
    }

    #[test]
    fn test_parse_ctx_arg_rounds_down() {
        // 5000 → rounds to 4096 (4 * 1024)
        assert_eq!(parse_ctx_arg("5000").unwrap(), Some(4096));
        // 33000 → rounds to 32768 (32 * 1024)
        assert_eq!(parse_ctx_arg("33000").unwrap(), Some(32768));
    }

    #[test]
    fn test_parse_ctx_arg_too_small() {
        assert!(parse_ctx_arg("1024").is_err());
        assert!(parse_ctx_arg("1K").is_err());
        assert!(parse_ctx_arg("100").is_err());
    }

    #[test]
    fn test_parse_ctx_arg_invalid() {
        assert!(parse_ctx_arg("abc").is_err());
        assert!(parse_ctx_arg("32M").is_err());
        assert!(parse_ctx_arg("--").is_err());
    }

    // --- short_channel_name ---

    #[test]
    fn test_short_channel_name() {
        assert_eq!(short_channel_name("whatsapp"), "wa");
        assert_eq!(short_channel_name("telegram"), "tg");
        assert_eq!(short_channel_name("email"), "email");
        assert_eq!(short_channel_name("other"), "other");
    }

    // --- build_prompt ---

    #[test]
    fn test_build_prompt_cloud() {
        let p = build_prompt(false, false, false);
        assert!(p.contains(">"));
        // Cloud prompt uses GREEN
        assert!(p.contains(crate::tui::GREEN));
    }

    #[test]
    fn test_build_prompt_local() {
        let p = build_prompt(true, false, false);
        assert!(p.contains("L>"));
        assert!(p.contains(crate::tui::YELLOW));
    }

    #[test]
    fn test_build_prompt_voice() {
        let p = build_prompt(false, true, false);
        assert!(p.contains("~>"));
        assert!(p.contains(crate::tui::MAGENTA));
    }

    #[test]
    fn test_build_prompt_thinking() {
        let p = build_prompt(false, false, true);
        assert!(p.contains("\u{1f9e0}"));
        assert!(p.contains(">"));
    }

    // --- ServerState ---

    #[test]
    fn test_server_state_new() {
        let state = ServerState::new("8080".to_string());
        assert_eq!(state.local_port, "8080");
        assert!(!state.lms_managed);
        assert!(state.lms_binary.is_none());
        assert_eq!(state.engine, InferenceEngine::None);
    }

    #[tokio::test]
    async fn test_server_state_kill_current_when_empty() {
        // Should not panic when there's no process to kill
        let mut state = ServerState::new("8080".to_string());
        state.kill_current(1234).await;
        assert_eq!(state.engine, InferenceEngine::None);
    }

    #[test]
    fn test_server_state_shutdown_when_empty() {
        // Should not panic when there's nothing to shut down
        let mut state = ServerState::new("8080".to_string());
        state.shutdown();
        assert!(!state.lms_managed);
        assert_eq!(state.engine, InferenceEngine::None);
    }

    // --- truncate_output ---

    #[test]
    fn test_truncate_output_short() {
        let result = truncate_output("hello\nworld", 40, 2000);
        assert_eq!(result, "hello\nworld");
    }

    #[test]
    fn test_truncate_output_max_lines() {
        let data = (0..50)
            .map(|i| format!("line {}", i))
            .collect::<Vec<_>>()
            .join("\n");
        let result = truncate_output(&data, 5, 10000);
        assert!(result.lines().count() <= 6); // 5 lines + truncated marker
        assert!(result.contains("...[truncated]"));
    }

    #[test]
    fn test_truncate_output_max_chars() {
        let data = "x".repeat(5000);
        let result = truncate_output(&data, 100, 100);
        assert!(result.len() < 200);
        assert!(result.contains("...[truncated]"));
    }

    #[test]
    fn test_truncate_output_empty() {
        let result = truncate_output("", 40, 2000);
        assert_eq!(result, "");
    }

}
