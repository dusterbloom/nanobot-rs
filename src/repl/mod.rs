//! REPL loop and interactive command dispatch for `nanobot agent`.
//!
//! Contains the main agent REPL, slash-command handlers, voice recording
//! pipeline, and background channel management.

mod commands;

use std::io::{self, IsTerminal, Write as _};
use std::net::TcpListener;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use rustyline::error::ReadlineError;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use crate::agent::agent_loop::{AgentLoop, SharedCoreHandle};
use crate::agent::audit::{AuditLog, ToolEvent};
use crate::agent::provenance::{ClaimVerifier, ClaimStatus};
use crate::cli;
use crate::config::loader::{get_data_dir, load_config, save_config};
use crate::config::schema::Config;
use crate::cron::service::CronService;
use crate::heartbeat::service::{HeartbeatService, DEFAULT_HEARTBEAT_INTERVAL_S, DEFAULT_MAINTENANCE_COMMANDS};
use crate::providers::base::LLMProvider;
use crate::providers::openai_compat::OpenAICompatProvider;
use crate::server;
use crate::syntax;
use crate::tui;
use crate::utils::helpers::get_workspace_path;

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
        prefix.parse::<usize>().map(|n| n * 1024).map_err(|_| "invalid number")?
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
pub(crate) fn build_prompt(is_local: bool, voice_on: bool) -> String {
    if voice_on {
        format!("{}{}~>{} ", crate::tui::BOLD, crate::tui::MAGENTA, crate::tui::RESET)
    } else if is_local {
        format!("{}{}L>{} ", crate::tui::BOLD, crate::tui::YELLOW, crate::tui::RESET)
    } else {
        format!("{}{}>{} ", crate::tui::BOLD, crate::tui::GREEN, crate::tui::RESET)
    }
}

/// Print the /help text.
pub(crate) fn print_help() {
    println!("\nCommands:");
    println!("  /local, /l      - Toggle between local and cloud mode");
    println!("  /model, /m      - Select local model from ~/models/");
    println!("  /ctx [size]     - Set context size (e.g. /ctx 32K) or auto-detect");
    println!("  /voice, /v      - Toggle voice mode (Ctrl+Space or Enter to speak)");
    println!("  /whatsapp, /wa  - Start WhatsApp channel (runs alongside chat)");
    println!("  /telegram, /tg  - Start Telegram channel (runs alongside chat)");
    println!("  /email          - Start Email channel (runs alongside chat)");
    println!("  /paste, /p      - Paste mode: multiline input until --- on its own line");
    println!("  /stop           - Stop all running channels");
    println!("  /agents, /a     - List running background agents");
    println!("  /kill <id>      - Cancel a background agent");
    println!("  /status, /s     - Show current mode, model, and channel info");
    println!("  /context        - Show context breakdown (tokens, messages, memory)");
    println!("  /memory         - Show working memory for current session");
    println!("  /replay         - Show session message history (/replay full | /replay N)");
    println!("  /restart, /rd   - Restart delegation server");
    println!("  /audit          - Show audit log for current session");
    println!("  /verify         - Re-verify claims in last response");
    println!("  /provenance     - Toggle provenance display on/off");
    println!("  /help, /h       - Show this help");
    println!("  Ctrl+C          - Exit\n");
}

// ============================================================================
// Input Watcher (Full-Duplex REPL)
// ============================================================================

/// Spawn a key watcher thread that runs during agent streaming/tool execution.
///
/// Handles:
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
) -> std::thread::JoinHandle<()> {
    use termimad::crossterm::event::{self, Event, KeyCode, KeyModifiers};
    use termimad::crossterm::terminal;

    std::thread::spawn(move || {
        terminal::enable_raw_mode().ok();
        let mut last_esc: Option<Instant> = None;

        while !done.load(Ordering::Relaxed) {
            if event::poll(Duration::from_millis(100)).unwrap_or(false) {
                if let Ok(Event::Key(key)) = event::read() {
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
                            if prev.elapsed() < Duration::from_millis(500) {
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
                        terminal::disable_raw_mode().ok();
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
                        terminal::enable_raw_mode().ok();
                        continue;
                    }

                    // Any other key clears the ESC state.
                    last_esc = None;
                }
            }
        }

        terminal::disable_raw_mode().ok();
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
    stream_and_render_inner(agent_loop, input, session_id, channel, lang, core_handle, false, None).await
}

/// Like `stream_and_render` but skips the user text erase-and-reprint.
/// Use when the caller has already rendered the user turn (e.g. voice recording).
#[cfg(feature = "voice")]
pub(crate) async fn stream_and_render_voice(
    agent_loop: &mut crate::agent::agent_loop::AgentLoop,
    input: &str,
    session_id: &str,
    channel: &str,
    lang: Option<&str>,
    core_handle: &SharedCoreHandle,
    tts_sentence_tx: Option<std::sync::mpsc::Sender<crate::voice::TtsCommand>>,
) -> String {
    stream_and_render_inner(agent_loop, input, session_id, channel, lang, core_handle, true, tts_sentence_tx).await
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
) -> String {
    // Erase raw readline output and reprint user text in grey box (skip if caller already rendered).
    if !user_already_rendered && std::io::stdout().is_terminal() {
        use std::io::Write as _;
        let prompt_and_input = format!("> {}", input);
        let raw_lines = tui::terminal_rows(&prompt_and_input, 0);
        print!("\x1b[{}A\x1b[J", raw_lines);
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

    // Spawn combined print task handling both text deltas and tool events.
    // Returns (tool_line_count, collected_tool_lines) so we can replay tool events after re-render.
    let print_task = if let Some(mut tool_rx) = tool_rx_opt {
        tokio::spawn(async move {
            use std::io::Write as _;
            #[cfg(feature = "voice")]
            let mut tts_acc = tts_tx.map(|tx| crate::voice::SentenceAccumulator::new_streaming(tx));
            #[cfg(not(feature = "voice"))]
            let _ = tts_tx;
            let mut tool_lines = 0usize;
            let mut collected: Vec<String> = Vec::new();
            let mut delta_done = false;
            let mut tool_done = false;
            // Track previous CallEnd to coalesce repeated status-check calls
            // (e.g. spawn list polling). When the same tool fires consecutive
            // CallEnd events, erase the previous box and overwrite in-place.
            let mut prev_call_end: Option<(String, usize)> = None; // (tool_name, lines_printed)
            loop {
                if delta_done && tool_done {
                    break;
                }
                tokio::select! {
                    biased;
                    delta = delta_rx.recv(), if !delta_done => {
                        match delta {
                            Some(d) => {
                                print!("{}", d);
                                std::io::stdout().flush().ok();
                                #[cfg(feature = "voice")]
                                if let Some(ref mut acc) = tts_acc {
                                    acc.push(&d);
                                }
                            }
                            None => {
                                delta_done = true;
                                #[cfg(feature = "voice")]
                                if let Some(acc) = tts_acc.take() {
                                    acc.flush();
                                }
                            },
                        }
                    }
                    event = tool_rx.recv(), if !tool_done => {
                        match event {
                            Some(ToolEvent::CallStart { ref tool_name, ref arguments_preview, .. }) => {
                                let line = format!(
                                    "\x1b[36m  \u{25b6} {}({})\x1b[0m",
                                    tool_name, arguments_preview
                                );
                                print!("\r{}\x1b[K", line);
                                std::io::stdout().flush().ok();
                                // CallStart line gets overwritten by CallEnd, don't collect.
                            }
                            Some(ToolEvent::Progress { ref tool_name, elapsed_ms, ref output_preview, .. }) => {
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
                            }
                            Some(ToolEvent::CallEnd { ref tool_name, ok, duration_ms, ref result_data, .. }) => {
                                // Coalesce repeated CallEnd for the same tool: erase
                                // the previous box and overwrite it so e.g. 129 spawn-list
                                // polls render as a single updating status instead of spam.
                                if let Some((ref prev_name, prev_lines)) = prev_call_end {
                                    if prev_name == tool_name && prev_lines > 0 {
                                        // Move cursor up over the previous box and clear.
                                        print!("\x1b[{}A\x1b[J", prev_lines);
                                        std::io::stdout().flush().ok();
                                        tool_lines = tool_lines.saturating_sub(prev_lines);
                                        // Remove the collected lines from the previous box.
                                        let keep = collected.len().saturating_sub(prev_lines);
                                        collected.truncate(keep);
                                    }
                                }

                                let marker = if ok { "\x1b[32m\u{2713}\x1b[0m" } else { "\x1b[31m\u{2717}\x1b[0m" };
                                let status_line = format!(
                                    "\x1b[36m  \u{25b6} {}\x1b[0m  {} \x1b[2m{}ms\x1b[0m",
                                    tool_name, marker, duration_ms
                                );
                                println!("\r\x1b[K{}", status_line);
                                let mut this_box_lines = 1usize;
                                collected.push(status_line);

                                if ok && !result_data.is_empty() {
                                    let truncated = truncate_output(result_data, 40, 2000);
                                    if !truncated.is_empty() {
                                        println!("    \x1b[2m\u{250c}\u{2500} output \u{2500}\x1b[0m");
                                        collected.push("    \x1b[2m\u{250c}\u{2500} output \u{2500}\x1b[0m".to_string());
                                        this_box_lines += 1;
                                        for line in truncated.lines() {
                                            let formatted = format!("    \x1b[2m\u{2502}\x1b[0m {}", line);
                                            println!("{}", formatted);
                                            collected.push(formatted);
                                            this_box_lines += 1;
                                        }
                                        println!("    \x1b[2m\u{2514}\u{2500}\x1b[0m");
                                        collected.push("    \x1b[2m\u{2514}\u{2500}\x1b[0m".to_string());
                                        this_box_lines += 1;
                                    }
                                } else if !ok && !result_data.is_empty() {
                                    let preview: String = result_data.chars().take(80).collect();
                                    let err_line = format!("    \x1b[2m\x1b[31m{}\x1b[0m", preview);
                                    println!("{}", err_line);
                                    collected.push(err_line);
                                    this_box_lines += 1;
                                }
                                tool_lines += this_box_lines;
                                prev_call_end = Some((tool_name.clone(), this_box_lines));
                                std::io::stdout().flush().ok();
                            }
                            None => tool_done = true,
                        }
                    }
                }
            }
            println!();
            (tool_lines, collected)
        })
    } else {
        tokio::spawn(async move {
            use std::io::Write as _;
            #[cfg(feature = "voice")]
            let mut tts_acc = tts_tx.map(|tx| crate::voice::SentenceAccumulator::new_streaming(tx));
            #[cfg(not(feature = "voice"))]
            let _ = tts_tx;
            while let Some(delta) = delta_rx.recv().await {
                print!("{}", delta);
                std::io::stdout().flush().ok();
                #[cfg(feature = "voice")]
                if let Some(ref mut acc) = tts_acc {
                    acc.push(&delta);
                }
            }
            #[cfg(feature = "voice")]
            if let Some(acc) = tts_acc.take() {
                acc.flush();
            }
            println!();
            (0usize, Vec::<String>::new())
        })
    };

    println!();

    // Full-duplex input watcher: handles ESC+ESC (cancel), Ctrl+C (cancel),
    // and backtick (priority message injection) during streaming/tool execution.
    let cancel_token = tokio_util::sync::CancellationToken::new();
    let (inject_tx, inject_rx) = tokio::sync::mpsc::unbounded_channel();
    let watcher_done = Arc::new(AtomicBool::new(false));

    let watcher = spawn_input_watcher(
        cancel_token.clone(),
        inject_tx,
        watcher_done.clone(),
    );

    let response = agent_loop
        .process_direct_streaming(
            input, session_id, channel, "direct", lang,
            delta_tx, tool_event_tx,
            Some(cancel_token.clone()),
            Some(inject_rx),
        )
        .await;

    // Signal watcher thread to stop and wait for it.
    watcher_done.store(true, Ordering::Relaxed);
    watcher.join().ok();

    let cancelled = cancel_token.is_cancelled();
    let (tool_lines, tool_event_lines) = print_task.await.unwrap_or((0, Vec::new()));

    // Erase raw streamed text + tool event lines, re-render with formatting + И marker
    if !response.is_empty() && std::io::stdout().is_terminal() {
        use std::io::Write as _;
        let text_lines = tui::terminal_rows(&response, 1);
        let total_lines = text_lines + tool_lines;
        print!("\x1b[{}A\x1b[J", total_lines);
        std::io::stdout().flush().ok();

        // Re-render tool events first so the user can see what the LLM actually did.
        if !tool_event_lines.is_empty() {
            for line in &tool_event_lines {
                println!("{}", line);
            }
            println!();
        }

        // Show redaction warning if strict mode removed fabricated claims.
        let redaction_count = response.matches("[unverified claim removed]").count();
        if redaction_count > 0 {
            println!(
                "\x1b[33m\x1b[1m  \u{26a0} {} claim(s) could not be verified against tool outputs and were redacted.\x1b[0m\n",
                redaction_count
            );
        }

        // Re-render with provenance claim verification if enabled.
        if verify_claims {
            let audit = AuditLog::new(&workspace, session_id);
            let entries = audit.get_entries();
            let verifier = ClaimVerifier::new(&entries);
            let annotated = verifier.verify(&response);
            let claims: Vec<(usize, usize, u8, String)> = annotated.iter().map(|c| {
                let status = match c.status {
                    ClaimStatus::Observed => 0u8,
                    ClaimStatus::Derived => 1,
                    ClaimStatus::Claimed => 2,
                    ClaimStatus::Recalled => 3,
                };
                (c.span.0, c.span.1, status, c.text.clone())
            }).collect();
            print!("{}", syntax::render_turn_with_provenance(
                &response, syntax::TurnRole::Assistant, &claims, strict_mode,
            ));
        } else {
            print!("{}", syntax::render_turn(&response, syntax::TurnRole::Assistant));
        }
    }

    if cancelled {
        println!("\n  \x1b[33mCancelled.\x1b[0m");
    }

    response
}

// ============================================================================
// Server Lifecycle (DRY: replaces 6x copy-pasted spawn+wait+rebuild patterns)
// ============================================================================

/// Mutable server lifecycle state for the REPL.
pub(crate) struct ServerState {
    pub llama_process: Option<std::process::Child>,
    pub compaction_process: Option<std::process::Child>,
    pub compaction_port: Option<String>,
    pub delegation_process: Option<std::process::Child>,
    pub delegation_port: Option<String>,
    pub local_port: String,
}

impl ServerState {
    pub fn new(port: String) -> Self {
        Self {
            llama_process: None,
            compaction_process: None,
            compaction_port: None,
            delegation_process: None,
            delegation_port: None,
            local_port: port,
        }
    }

    /// Kill the current llama process (if we own one) and any orphaned servers.
    pub fn kill_current(&mut self) {
        if let Some(ref mut child) = self.llama_process {
            let pid = child.id();
            child.kill().ok();
            child.wait().ok();
            server::unrecord_server_pid(pid);
        }
        self.llama_process = None;
        server::kill_tracked_servers();
    }

    /// Full shutdown: kill llama + compaction + delegation servers.
    pub fn shutdown(&mut self) {
        if let Some(ref mut child) = self.llama_process {
            println!("Stopping llama.cpp server...");
            let pid = child.id();
            child.kill().ok();
            child.wait().ok();
            server::unrecord_server_pid(pid);
        }
        self.llama_process = None;
        server::stop_compaction_server(&mut self.compaction_process, &mut self.compaction_port);
        server::stop_delegation_server(&mut self.delegation_process, &mut self.delegation_port);
    }
}

/// Try to start a llama server with the given model and context size.
///
/// On success: updates `state.local_port` and `state.llama_process`, returns `Ok(port_string)`.
/// On failure: cleans up and returns `Err(message)`.
pub(crate) async fn try_start_server(
    state: &mut ServerState,
    model_path: &std::path::Path,
    ctx_size: usize,
) -> Result<String, String> {
    let port = server::find_available_port(8080);
    match server::spawn_llama_server(port, model_path, ctx_size) {
        Ok(child) => {
            server::record_server_pid("main", child.id());
            state.llama_process = Some(child);
            if server::wait_for_server_ready(port, 120, &mut state.llama_process).await {
                let port_str = port.to_string();
                state.local_port = port_str.clone();
                Ok(port_str)
            } else {
                // Server started but didn't become healthy
                if let Some(ref mut child) = state.llama_process {
                    child.kill().ok();
                    child.wait().ok();
                }
                state.llama_process = None;
                Err("server failed to become ready".to_string())
            }
        }
        Err(e) => Err(format!("failed to spawn server: {}", e)),
    }
}

/// Rebuild the agent core and agent loop after a server change.
///
/// Call this after a successful `try_start_server` or mode switch.
pub(crate) fn apply_server_change(
    state: &ServerState,
    model_path: &std::path::Path,
    core_handle: &SharedCoreHandle,
    config: &Config,
) {
    cli::rebuild_core(
        core_handle,
        config,
        &state.local_port,
        model_path.file_name().and_then(|n| n.to_str()),
        state.compaction_port.as_deref(),
        state.delegation_port.as_deref(),
    );
}

/// Outcome of a server start attempt with fallback.
pub(crate) enum StartOutcome {
    /// Primary model started successfully.
    Started,
    /// Primary failed, fallback succeeded (model path was restored).
    Fallback,
    /// Both primary and fallback failed — switched to cloud mode.
    CloudFallback,
}

/// Start server with the primary model, falling back to `fallback_model` if it fails,
/// and finally switching to cloud mode if both fail.
///
/// This replaces the 4x copy-pasted restart-with-fallback pattern.
pub(crate) async fn start_with_fallback(
    state: &mut ServerState,
    primary_model: &std::path::Path,
    primary_ctx: usize,
    fallback: Option<(&std::path::Path, usize)>,
) -> StartOutcome {
    // Try primary
    match try_start_server(state, primary_model, primary_ctx).await {
        Ok(_) => return StartOutcome::Started,
        Err(e) => {
            println!(
                "  {}{}Server failed:{} {}",
                tui::BOLD, tui::YELLOW, tui::RESET, e
            );
        }
    }

    // Try fallback model/ctx if provided
    if let Some((fb_model, fb_ctx)) = fallback {
        let fb_name = fb_model
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");
        println!(
            "  {}Falling back to: {} ({}K)...{}",
            tui::DIM,
            fb_name,
            fb_ctx / 1024,
            tui::RESET
        );
        match try_start_server(state, fb_model, fb_ctx).await {
            Ok(_) => return StartOutcome::Fallback,
            Err(e) => {
                println!(
                    "  {}{}Fallback also failed:{} {}",
                    tui::BOLD, tui::YELLOW, tui::RESET, e
                );
            }
        }
    }

    // Both failed — switch to cloud
    println!(
        "  {}{}Switching to cloud mode{}",
        tui::BOLD, tui::YELLOW, tui::RESET
    );
    crate::LOCAL_MODE.store(false, Ordering::SeqCst);
    StartOutcome::CloudFallback
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

pub(crate) fn cmd_agent(message: Option<String>, session_id: String, local_flag: bool, lang: Option<String>) {
    let config = load_config(None);

    // Resolve voice language: CLI --lang > config voice.language > None (auto)
    let lang = lang.or_else(|| config.voice.language.clone());

    // Check environment variable for local mode
    let local_env = std::env::var("NANOBOT_LOCAL")
        .map(|v| v == "1" || v.to_lowercase() == "true")
        .unwrap_or(false);

    // Set initial local mode from flag or environment
    if local_flag || local_env {
        crate::LOCAL_MODE.store(true, Ordering::SeqCst);
    }

    let local_port = std::env::var("NANOBOT_LOCAL_PORT").unwrap_or_else(|_| "8080".to_string());

    // Check if we can proceed
    let is_local = crate::LOCAL_MODE.load(Ordering::SeqCst);
    if !is_local {
        let api_key = config.get_api_key();
        let model = &config.agents.defaults.model;
        let has_prefix = config.resolve_provider_for_model(model).is_some();
        let has_oauth = dirs::home_dir()
            .map(|h| h.join(".claude").join(".credentials.json").exists())
            .unwrap_or(false);
        if api_key.is_none() && !has_prefix && !model.starts_with("bedrock/") && !model.starts_with("claude-max") && !has_oauth {
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
        let core_handle = cli::build_core_handle(&config, &local_port, Some(server::DEFAULT_LOCAL_MODEL), None, None);
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

        let agent_loop = cli::create_agent_loop(
            core_handle.clone(), &config, Some(cron_service.clone()), email_config.clone(), Some(display_tx.clone()),
        );

        if let Some(msg) = message {
            // Single-message mode: process and exit.
            let mut agent_loop = agent_loop;
            stream_and_render(&mut agent_loop, &msg, &session_id, "cli", None, &core_handle).await;
            index_sessions_background();
        } else {
            // Interactive REPL mode.
            tui::print_startup_splash(&local_port);

            let mut srv = ServerState::new(local_port.clone());
            let default_model = dirs::home_dir().unwrap().join("models").join(server::DEFAULT_LOCAL_MODEL);

            // Auto-spawn delegation server for cloud mode
            if !is_local
                && config.tool_delegation.enabled
                && config.tool_delegation.auto_local
                && config.tool_delegation.provider.is_none()
            {
                server::start_delegation_if_available(
                    &mut srv.delegation_process,
                    &mut srv.delegation_port,
                ).await;
                if srv.delegation_port.is_some() {
                    cli::rebuild_core(
                        &core_handle, &config, &local_port,
                        Some(server::DEFAULT_LOCAL_MODEL), None,
                        srv.delegation_port.as_deref(),
                    );
                }
            }

            // Readline editor with history
            let history_path = get_data_dir().join("history.txt");
            let rl = rustyline::DefaultEditor::new()
                .expect("Failed to create line editor");

            // Build ReplContext — all mutable REPL state in one struct.
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
                rl,
                #[cfg(feature = "voice")]
                voice_session: None,
            };

            let _ = ctx.rl.load_history(&history_path);

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
            );
            heartbeat.start().await;

            // Start health watchdog for local servers (if any are running).
            if is_local {
                let mut ports = Vec::new();
                ports.push(("main".to_string(), ctx.srv.local_port.clone()));
                if let Some(ref cp) = ctx.srv.compaction_port {
                    ports.push(("compaction".to_string(), cp.clone()));
                }
                if let Some(ref dp) = ctx.srv.delegation_port {
                    ports.push(("delegation".to_string(), dp.clone()));
                }
                let _watchdog = server::start_health_watchdog(ports, ctx.display_tx.clone());
            }

            loop {
                // Drain any pending display messages from background channels.
                ctx.drain_display();

                let is_local = crate::LOCAL_MODE.load(Ordering::SeqCst);
                let voice_on = ctx.voice_on();
                let prompt = build_prompt(is_local, voice_on);

                // === GET INPUT ===
                let input_text: String;
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
                            let _ = ctx.rl.add_history_entry(&line);
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
                            let _ = ctx.rl.add_history_entry(&line);
                            input_text = line;
                        }
                        Err(ReadlineError::Interrupted | ReadlineError::Eof) => break,
                        Err(_) => break,
                    }
                }

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
                        let transcription = ctx.voice_session.as_mut()
                            .and_then(|vs| vs.record_and_transcribe().transpose())
                            .transpose();

                        match transcription {
                            Ok(Some((text, detected_lang))) => {
                                let tts_lang_owned = ctx.lang.clone().unwrap_or(detected_lang);

                                // Render user text with purple ● marker.
                                print!("{}", syntax::render_turn(&text, syntax::TurnRole::VoiceUser));

                                // Start streaming TTS pipeline BEFORE LLM call.
                                let tts_parts = ctx.voice_session.as_mut()
                                    .and_then(|vs| {
                                        vs.clear_cancel();
                                        vs.start_streaming_speak(&tts_lang_owned, None).ok()
                                    });
                                let (sentence_tx, join_handle) = match tts_parts {
                                    Some((tx, jh)) => (Some(tx), Some(jh)),
                                    None => (None, None),
                                };

                                // Phase 2: LLM call with parallel TTS feeding.
                                let _response = stream_and_render_voice(
                                    &mut ctx.agent_loop, &text, &ctx.session_id,
                                    "voice", Some(&tts_lang_owned), &ctx.core_handle,
                                    sentence_tx,
                                ).await;

                                ctx.drain_display();
                                println!();
                                ctx.print_status_bar().await;

                                // Phase 3: Wait for TTS playback to finish.
                                if let Some(jh) = join_handle {
                                    let cancel = ctx.voice_session.as_ref()
                                        .map(|vs| vs.cancel_flag())
                                        .unwrap_or_else(|| Arc::new(AtomicBool::new(false)));
                                    let done = Arc::new(AtomicBool::new(false));
                                    let done2 = done.clone();
                                    let watcher = tui::spawn_interrupt_watcher(cancel.clone(), done2);
                                    let _ = jh.join();  // blocks until all audio played
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
                if input.is_empty() { continue; }

                // Dispatch slash commands.
                if input.starts_with('/') && ctx.dispatch(input).await {
                    continue;
                }

                // Process message (streaming)
                let channel = if voice_on { "voice" } else { "cli" };
                let response = stream_and_render(&mut ctx.agent_loop, input, &ctx.session_id, channel, None, &ctx.core_handle).await;
                ctx.drain_display();
                println!();
                ctx.print_status_bar().await;

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

            // Cleanup: stop heartbeat, save readline history, kill servers
            heartbeat.stop().await;
            let _ = ctx.rl.save_history(&history_path);
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
        let p = build_prompt(false, false);
        assert!(p.contains(">"));
        // Cloud prompt uses GREEN
        assert!(p.contains(crate::tui::GREEN));
    }

    #[test]
    fn test_build_prompt_local() {
        let p = build_prompt(true, false);
        assert!(p.contains("L>"));
        assert!(p.contains(crate::tui::YELLOW));
    }

    #[test]
    fn test_build_prompt_voice() {
        let p = build_prompt(false, true);
        assert!(p.contains("~>"));
        assert!(p.contains(crate::tui::MAGENTA));
    }

    // --- ServerState ---

    #[test]
    fn test_server_state_new() {
        let state = ServerState::new("8080".to_string());
        assert!(state.llama_process.is_none());
        assert!(state.compaction_process.is_none());
        assert!(state.compaction_port.is_none());
        assert!(state.delegation_process.is_none());
        assert!(state.delegation_port.is_none());
        assert_eq!(state.local_port, "8080");
    }

    #[test]
    fn test_server_state_kill_current_when_empty() {
        // Should not panic when there's no process to kill
        let mut state = ServerState::new("8080".to_string());
        state.kill_current();
        assert!(state.llama_process.is_none());
    }

    #[test]
    fn test_server_state_shutdown_when_empty() {
        // Should not panic when there's nothing to shut down
        let mut state = ServerState::new("8080".to_string());
        state.shutdown();
        assert!(state.llama_process.is_none());
        assert!(state.compaction_process.is_none());
        assert!(state.compaction_port.is_none());
        assert!(state.delegation_process.is_none());
        assert!(state.delegation_port.is_none());
    }

    #[test]
    fn test_server_state_kill_terminates_child() {
        // Spawn a real but harmless process, then verify kill_current terminates it
        let child = std::process::Command::new("sleep")
            .arg("60")
            .spawn()
            .expect("failed to spawn sleep");
        let pid = child.id();
        let mut state = ServerState::new("8080".to_string());
        state.llama_process = Some(child);

        state.kill_current();
        assert!(state.llama_process.is_none());

        // Verify the process is gone (kill(pid, 0) should fail)
        let status = unsafe { libc::kill(pid as i32, 0) };
        assert_ne!(status, 0, "process should be dead after kill_current");
    }

    // --- truncate_output ---

    #[test]
    fn test_truncate_output_short() {
        let result = truncate_output("hello\nworld", 40, 2000);
        assert_eq!(result, "hello\nworld");
    }

    #[test]
    fn test_truncate_output_max_lines() {
        let data = (0..50).map(|i| format!("line {}", i)).collect::<Vec<_>>().join("\n");
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

    // --- StartOutcome ---

    #[test]
    fn test_start_outcome_variants() {
        // Just verify the enum variants exist and can be matched
        let outcomes = vec![
            StartOutcome::Started,
            StartOutcome::Fallback,
            StartOutcome::CloudFallback,
        ];
        for outcome in outcomes {
            match outcome {
                StartOutcome::Started => {}
                StartOutcome::Fallback => {}
                StartOutcome::CloudFallback => {}
            }
        }
    }
}
