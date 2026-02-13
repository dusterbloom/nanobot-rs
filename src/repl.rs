//! REPL loop and interactive command dispatch for `nanobot agent`.
//!
//! Contains the main agent REPL, slash-command handlers, voice recording
//! pipeline, and background channel management.

use std::io::{self, BufRead, IsTerminal, Write as _};
use std::net::TcpListener;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use rustyline::error::ReadlineError;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use crate::agent::agent_loop::{AgentLoop, SharedCoreHandle};
use crate::cli;
use crate::config::loader::{get_data_dir, load_config, save_config};
use crate::config::schema::Config;
use crate::cron::service::CronService;
use crate::providers::base::LLMProvider;
use crate::providers::openai_compat::OpenAICompatProvider;
use crate::server;
use crate::syntax;
use crate::tui;
use crate::utils::helpers::get_workspace_path;
#[cfg(feature = "voice")]
use crate::voice;
#[cfg(feature = "voice")]
use crate::voice_pipeline;

// ============================================================================
// Helpers (testable, pure-ish)
// ============================================================================

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
    println!("  /help, /h       - Show this help");
    println!("  Ctrl+C          - Exit\n");
}

/// Stream an LLM response with live delta printing, then erase and re-render with syntax highlighting.
///
/// This replaces the 3x copy-pasted pattern:
///   create delta channel → spawn print task → stream → await → erase raw → re-render
///
/// Returns the full response text.
pub(crate) async fn stream_and_render(
    agent_loop: &mut crate::agent::agent_loop::AgentLoop,
    input: &str,
    session_id: &str,
    channel: &str,
    lang: Option<&str>,
) -> String {
    let (delta_tx, mut delta_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    let print_task = tokio::spawn(async move {
        use std::io::Write as _;
        while let Some(delta) = delta_rx.recv().await {
            print!("{}", delta);
            std::io::stdout().flush().ok();
        }
        println!();
    });
    println!();
    let response = agent_loop
        .process_direct_streaming(input, session_id, channel, "direct", lang, delta_tx)
        .await;
    let _ = print_task.await;

    // Erase raw streamed text, re-render with formatting
    if !response.is_empty() && std::io::stdout().is_terminal() {
        use std::io::Write as _;
        let lines = tui::terminal_rows(&response, 1);
        print!("\x1b[{}A\x1b[J", lines);
        std::io::stdout().flush().ok();
        println!("{}{}И{}", tui::BOLD, tui::WHITE, tui::RESET);
        print!("{}", syntax::render_response(&response));
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
    pub local_port: String,
}

impl ServerState {
    pub fn new(port: String) -> Self {
        Self {
            llama_process: None,
            compaction_process: None,
            compaction_port: None,
            local_port: port,
        }
    }

    /// Kill the current llama process (if we own one) and any orphaned servers.
    pub fn kill_current(&mut self) {
        if let Some(ref mut child) = self.llama_process {
            child.kill().ok();
            child.wait().ok();
        }
        self.llama_process = None;
        server::kill_stale_llama_servers();
    }

    /// Full shutdown: kill llama + compaction servers.
    pub fn shutdown(&mut self) {
        if let Some(ref mut child) = self.llama_process {
            println!("Stopping llama.cpp server...");
            child.kill().ok();
            child.wait().ok();
        }
        self.llama_process = None;
        server::stop_compaction_server(&mut self.compaction_process, &mut self.compaction_port);
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
struct ActiveChannel {
    name: String,
    stop: Arc<AtomicBool>,
    handle: tokio::task::JoinHandle<()>,
}

pub(crate) fn cmd_agent(message: Option<String>, session_id: String, local_flag: bool, lang: Option<String>) {
    let config = load_config(None);

    // Check environment variable for local mode
    let local_env = std::env::var("NANOBOT_LOCAL")
        .map(|v| v == "1" || v.to_lowercase() == "true")
        .unwrap_or(false);

    // Set initial local mode from flag or environment
    if local_flag || local_env {
        crate::LOCAL_MODE.store(true, Ordering::SeqCst);
    }

    let mut local_port = std::env::var("NANOBOT_LOCAL_PORT").unwrap_or_else(|_| "8080".to_string());

    // Check if we can proceed
    let is_local = crate::LOCAL_MODE.load(Ordering::SeqCst);
    if !is_local {
        let api_key = config.get_api_key();
        let model = &config.agents.defaults.model;
        if api_key.is_none() && !model.starts_with("bedrock/") {
            eprintln!("Error: No API key configured.");
            eprintln!("Set one in ~/.nanobot/config.json under providers.openrouter.apiKey");
            eprintln!("Or use --local flag to use a local LLM server.");
            std::process::exit(1);
        }
    }

    let runtime = tokio::runtime::Runtime::new().expect("Failed to create tokio runtime");

    runtime.block_on(async {
        // Create shared core and initial agent loop.
        let core_handle = cli::build_core_handle(&config, &local_port, Some(server::DEFAULT_LOCAL_MODEL), None);
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

        let mut agent_loop = cli::create_agent_loop(
            core_handle.clone(), &config, Some(cron_service.clone()), email_config.clone(), None,
        );

        if let Some(msg) = message {
            stream_and_render(&mut agent_loop, &msg, &session_id, "cli", None).await;
        } else {
            tui::print_startup_splash(&local_port);

            let mut srv = ServerState::new(local_port.clone());
            let default_model = dirs::home_dir().unwrap().join("models").join(server::DEFAULT_LOCAL_MODEL);
            let mut current_model_path: std::path::PathBuf = default_model;
            #[cfg(feature = "voice")]
            let mut voice_session: Option<voice::VoiceSession> = None;

            // Readline editor with history
            let history_path = get_data_dir().join("history.txt");
            let mut rl = rustyline::DefaultEditor::new()
                .expect("Failed to create line editor");
            let _ = rl.load_history(&history_path);

            let mut active_channels: Vec<ActiveChannel> = vec![];
            // Channel for background gateways to send display lines to the REPL.
            let (display_tx, mut display_rx) = mpsc::unbounded_channel::<String>();

            loop {
                // Drain any pending display messages from background channels.
                while let Ok(line) = display_rx.try_recv() {
                    println!("\r{}", line);
                }
                let is_local = crate::LOCAL_MODE.load(Ordering::SeqCst);
                #[cfg(feature = "voice")]
                let voice_on = voice_session.is_some();
                #[cfg(not(feature = "voice"))]
                let voice_on = false;

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
                    match rl.readline(&prompt) {
                        Ok(line) => {
                            let _ = rl.add_history_entry(&line);
                            input_text = line;
                        }
                        Err(ReadlineError::Interrupted | ReadlineError::Eof) => break,
                        Err(_) => break,
                    }
                }

                #[cfg(not(feature = "voice"))]
                {
                    match rl.readline(&prompt) {
                        Ok(line) => {
                            let _ = rl.add_history_entry(&line);
                            input_text = line;
                        }
                        Err(ReadlineError::Interrupted | ReadlineError::Eof) => break,
                        Err(_) => break,
                    }
                }

                // === VOICE RECORDING (streaming pipeline) ===
                #[cfg(feature = "voice")]
                if do_record {
                    if let Some(ref mut vs) = voice_session {
                        vs.stop_playback();
                        let mut keep_recording = true;
                        while keep_recording {
                            keep_recording = false;
                            match vs.record_and_transcribe() {
                                Ok(Some((text, lang))) => {
                                    // Start streaming TTS pipeline
                                    vs.clear_cancel();
                                    let cancel = vs.cancel_flag();

                                    // Display channel: synthesis thread → terminal
                                    // Text appears when TTS finishes each sentence (synced with audio)
                                    let (display_tx, mut display_rx) =
                                        tokio::sync::mpsc::unbounded_channel::<String>();

                                    match vs.start_streaming_speak(&lang, Some(display_tx)) {
                                        Ok((sentence_tx, tts_handle)) => {
                                            // Delta channel: LLM → accumulator (silent, feeds TTS only)
                                            let (delta_tx, mut delta_rx) =
                                                tokio::sync::mpsc::unbounded_channel::<String>();

                                            let acc_sentence_tx = sentence_tx.clone();
                                            let accumulator_task = tokio::spawn(async move {
                                                let mut acc = voice::SentenceAccumulator::new(acc_sentence_tx);
                                                while let Some(delta) = delta_rx.recv().await {
                                                    acc.push(&delta);
                                                }
                                                acc.flush();
                                            });

                                            // Display task: print sentences as TTS synthesizes them
                                            // Returns total chars printed so we can erase the right number of rows.
                                            let display_task = tokio::spawn(async move {
                                                use std::io::Write as _;
                                                let mut total_chars: usize = 0;
                                                let mut first = true;
                                                while let Some(sentence) = display_rx.recv().await {
                                                    if first {
                                                        first = false;
                                                    } else {
                                                        print!(" ");
                                                        total_chars += 1;
                                                    }
                                                    print!("{}", sentence);
                                                    total_chars += sentence.len();
                                                    std::io::stdout().flush().ok();
                                                }
                                                println!();
                                                total_chars
                                            });

                                            // Interrupt watcher: runs during LLM streaming AND TTS playback
                                            let done = Arc::new(AtomicBool::new(false));
                                            let watcher = tui::spawn_interrupt_watcher(cancel.clone(), done.clone());

                                            // Stream LLM response (deltas go to accumulator silently)
                                            let response = agent_loop
                                                .process_direct_streaming(
                                                    &text,
                                                    &session_id,
                                                    "voice",
                                                    "direct",
                                                    Some(&lang),
                                                    delta_tx,
                                                )
                                                .await;

                                            // Wait for accumulator to flush remaining sentences
                                            let _ = accumulator_task.await;

                                            // Wait for TTS + playback + display to finish
                                            // (if cancelled, TTS breaks out of loop quickly)
                                            let _ = tts_handle.join();
                                            done.store(true, Ordering::Relaxed);
                                            let interrupted = watcher.join().unwrap_or(false);
                                            let display_chars = display_task.await.unwrap_or(0);

                                            // Erase plain streamed text, re-render with formatting.
                                            // Voice display is one long wrapped line, so use terminal
                                            // width to compute how many rows to erase.
                                            if !response.is_empty() && display_chars > 0 {
                                                use std::io::Write as _;
                                                let term_width = crossterm::terminal::size()
                                                    .map(|(w, _)| w as usize)
                                                    .unwrap_or(80);
                                                let rows = (display_chars / term_width) + 2;
                                                print!("\x1b[{}A\x1b[J", rows);
                                                std::io::stdout().flush().ok();
                                                print!("{}", syntax::render_response(&response));
                                                println!();
                                            }

                                            {
                                                let sa_count = agent_loop.subagent_manager().get_running_count().await;
                                                active_channels.retain(|ch| !ch.handle.is_finished());
                                                let ch_names: Vec<&str> = active_channels.iter().map(|c| short_channel_name(&c.name)).collect();
                                                tui::print_status_bar(&core_handle, &ch_names, sa_count);
                                            }

                                            if interrupted {
                                                keep_recording = true;
                                            }
                                        }
                                        Err(e) => {
                                            eprintln!("Streaming TTS failed ({}), falling back", e);
                                            let response = agent_loop
                                                .process_direct_with_lang(&text, &session_id, "voice", "direct", Some(&lang))
                                                .await;
                                            println!();
                                            print!("{}", syntax::render_response(&response));
                                            println!();
                                            let tts_text = tui::strip_markdown_for_tts(&response);
                                            if !tts_text.is_empty() {
                                                if tui::speak_interruptible(vs, &tts_text, "en") {
                                                    keep_recording = true;
                                                }
                                            }
                                        }
                                    }
                                }
                                Ok(None) => println!("\x1b[2m(no speech detected)\x1b[0m"),
                                Err(e) => eprintln!("\x1b[31m{}\x1b[0m", e),
                            }
                        }
                        tui::drain_stdin();
                    }
                    continue;
                }

                // === TEXT INPUT ===
                let input = input_text.trim();
                if input.is_empty() { continue; }

                // Handle mode toggle commands
                if input == "/local" || input == "/l" {
                    let currently_local = crate::LOCAL_MODE.load(Ordering::SeqCst);

                    if !currently_local {
                        // Toggle ON: check if a llama.cpp server is already running
                        let mut found_port: Option<u16> = None;
                        for port in 8080..=8089 {
                            let url = format!("http://localhost:{}/health", port);
                            if let Ok(resp) = reqwest::blocking::get(&url) {
                                if resp.status().is_success() {
                                    // Reuse only if this server is configured with
                                    // one slot (`n_parallel=1`), otherwise per-request
                                    // context can be much smaller than advertised.
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
                            srv.local_port = port.to_string();
                            crate::LOCAL_MODE.store(true, Ordering::SeqCst);
                            let main_ctx = server::compute_optimal_context_size(&current_model_path);
                            server::start_compaction_if_available(&mut srv.compaction_process, &mut srv.compaction_port, main_ctx).await;
                            cli::rebuild_core(&core_handle, &config, &srv.local_port, current_model_path.file_name().and_then(|n| n.to_str()), srv.compaction_port.as_deref());
                            agent_loop = cli::create_agent_loop(core_handle.clone(), &config, Some(cron_service.clone()), email_config.clone(), None);
                            tui::print_mode_banner(&srv.local_port);
                        } else {
                            // Kill any orphaned servers from previous runs
                            srv.kill_current();
                            let ctx_size = server::compute_optimal_context_size(&current_model_path);
                            println!("\n  {}{}Starting{} llama.cpp server (ctx: {}K)...", tui::BOLD, tui::YELLOW, tui::RESET, ctx_size / 1024);

                            match try_start_server(&mut srv, &current_model_path, ctx_size).await {
                                Ok(_) => {
                                    crate::LOCAL_MODE.store(true, Ordering::SeqCst);
                                    server::start_compaction_if_available(&mut srv.compaction_process, &mut srv.compaction_port, ctx_size).await;
                                    apply_server_change(&srv, &current_model_path, &core_handle, &config);
                                    agent_loop = cli::create_agent_loop(core_handle.clone(), &config, Some(cron_service.clone()), email_config.clone(), None);
                                    tui::print_mode_banner(&srv.local_port);
                                }
                                Err(e) => {
                                    println!("  {}{}Failed: {}{}", tui::BOLD, tui::YELLOW, e, tui::RESET);
                                    println!("  {}Remaining in cloud mode{}\n", tui::DIM, tui::RESET);
                                }
                            }
                        }
                    } else {
                        // Toggle OFF: kill server and switch to cloud
                        srv.shutdown();
                        crate::LOCAL_MODE.store(false, Ordering::SeqCst);
                        apply_server_change(&srv, &current_model_path, &core_handle, &config);
                        agent_loop = cli::create_agent_loop(core_handle.clone(), &config, Some(cron_service.clone()), email_config.clone(), None);
                        tui::print_mode_banner(&srv.local_port);
                    }

                    continue;
                }

                // Handle model selection
                if input == "/model" || input == "/m" {
                    let models = server::list_local_models();
                    if models.is_empty() {
                        println!("\nNo .gguf models found in ~/models/\n");
                        continue;
                    }

                    println!("\nAvailable models:");
                    for (i, path) in models.iter().enumerate() {
                        let name = path.file_name().unwrap().to_string_lossy();
                        let size_mb = std::fs::metadata(path)
                            .map(|m| m.len() / 1_048_576)
                            .unwrap_or(0);
                        let marker = if *path == current_model_path { " (active)" } else { "" };
                        println!("  [{}] {} ({} MB){}", i + 1, name, size_mb, marker);
                    }
                    let model_prompt = format!("Select model [1-{}] or Enter to cancel: ", models.len());
                    let choice = match rl.readline(&model_prompt) {
                        Ok(line) => line,
                        Err(_) => { continue; }
                    };
                    let choice = choice.trim();
                    if choice.is_empty() {
                        continue;
                    }
                    let idx: usize = match choice.parse::<usize>() {
                        Ok(n) if n >= 1 && n <= models.len() => n - 1,
                        _ => {
                            println!("Invalid selection.\n");
                            continue;
                        }
                    };

                    let selected = &models[idx];
                    let previous_model_path = current_model_path.clone();
                    current_model_path = selected.clone();
                    let name = selected.file_name().unwrap().to_string_lossy();
                    println!("\nSelected: {}", name);

                    // If local mode is active, restart the server with the new model
                    if crate::LOCAL_MODE.load(Ordering::SeqCst) {
                        srv.kill_current();
                        let ctx_size = server::compute_optimal_context_size(&current_model_path);
                        println!("  {}{}Starting{} llama.cpp (ctx: {}K)...", tui::BOLD, tui::YELLOW, tui::RESET, ctx_size / 1024);
                        let fallback_ctx = server::compute_optimal_context_size(&previous_model_path);
                        match start_with_fallback(&mut srv, &current_model_path, ctx_size, Some((&previous_model_path, fallback_ctx))).await {
                            StartOutcome::Started => {
                                apply_server_change(&srv, &current_model_path, &core_handle, &config);
                                agent_loop = cli::create_agent_loop(core_handle.clone(), &config, Some(cron_service.clone()), email_config.clone(), None);
                                tui::print_mode_banner(&srv.local_port);
                            }
                            StartOutcome::Fallback => {
                                current_model_path = previous_model_path.clone();
                                apply_server_change(&srv, &current_model_path, &core_handle, &config);
                                agent_loop = cli::create_agent_loop(core_handle.clone(), &config, Some(cron_service.clone()), email_config.clone(), None);
                                println!("  {}Restored previous model{}", tui::DIM, tui::RESET);
                            }
                            StartOutcome::CloudFallback => {
                                apply_server_change(&srv, &current_model_path, &core_handle, &config);
                                agent_loop = cli::create_agent_loop(core_handle.clone(), &config, Some(cron_service.clone()), email_config.clone(), None);
                            }
                        }
                    } else {
                        println!("Model will be used next time you toggle /local on.\n");
                    }

                    continue;
                }

                // Handle context size change
                if input == "/ctx" || input.starts_with("/ctx ") {
                    if !crate::LOCAL_MODE.load(Ordering::SeqCst) {
                        println!("\n  {}Not in local mode — use /local first{}\n", tui::DIM, tui::RESET);
                        continue;
                    }

                    let arg = input.strip_prefix("/ctx").unwrap().trim();
                    let new_ctx: usize = match parse_ctx_arg(arg) {
                        Ok(Some(n)) => n,
                        Ok(None) => {
                            // No argument → re-auto-detect
                            let auto = server::compute_optimal_context_size(&current_model_path);
                            println!("\n  Auto-detected: {}K", auto / 1024);
                            auto
                        }
                        Err(msg) => {
                            println!("\n  {}\n", msg);
                            println!("  Usage: /ctx [size]  e.g. /ctx 32K or /ctx 32768\n");
                            continue;
                        }
                    };

                    // Restart server with new context size
                    srv.kill_current();
                    let fallback_ctx = server::compute_optimal_context_size(&current_model_path);
                    println!("  {}{}Restarting{} llama.cpp (ctx: {}K)...", tui::BOLD, tui::YELLOW, tui::RESET, new_ctx / 1024);
                    match start_with_fallback(&mut srv, &current_model_path, new_ctx, Some((&current_model_path, fallback_ctx))).await {
                        StartOutcome::Started | StartOutcome::Fallback => {
                            apply_server_change(&srv, &current_model_path, &core_handle, &config);
                            agent_loop = cli::create_agent_loop(core_handle.clone(), &config, Some(cron_service.clone()), email_config.clone(), None);
                            tui::print_mode_banner(&srv.local_port);
                        }
                        StartOutcome::CloudFallback => {
                            apply_server_change(&srv, &current_model_path, &core_handle, &config);
                            agent_loop = cli::create_agent_loop(core_handle.clone(), &config, Some(cron_service.clone()), email_config.clone(), None);
                        }
                    }

                    continue;
                }

                // Handle voice toggle
                #[cfg(feature = "voice")]
                if input == "/voice" || input == "/v" {
                    if voice_session.is_some() {
                        if let Some(ref mut vs) = voice_session {
                            vs.stop_playback();
                        }
                        voice_session = None;
                        println!("\nVoice mode OFF\n");
                    } else {
                        match voice::VoiceSession::with_lang(lang.as_deref()).await {
                            Ok(vs) => {
                                voice_session = Some(vs);
                                println!("\nVoice mode ON. Ctrl+Space or Enter to speak, type for text.\n");
                            }
                            Err(e) => eprintln!("\nFailed to start voice mode: {}\n", e),
                        }
                    }
                    continue;
                }

                // Handle WhatsApp quick-start from REPL
                if input == "/whatsapp" || input == "/wa" {
                    active_channels.retain(|ch| !ch.handle.is_finished());
                    if active_channels.iter().any(|ch| ch.name == "whatsapp") {
                        println!("\n  WhatsApp is already running. Use /stop to stop channels.\n");
                        continue;
                    }
                    let mut gw_config = load_config(None);
                    cli::check_api_key(&gw_config);
                    gw_config.channels.whatsapp.enabled = true;
                    gw_config.channels.telegram.enabled = false;
                    gw_config.channels.feishu.enabled = false;
                    gw_config.channels.email.enabled = false;
                    let stop = Arc::new(AtomicBool::new(false));
                    let stop2 = stop.clone();
                    let dtx = display_tx.clone();
                    let ch = core_handle.clone();
                    println!("\n  Scan the QR code when it appears");
                    let handle = tokio::spawn(async move {
                        cli::run_gateway_async(gw_config, ch, Some(stop2), Some(dtx)).await;
                    });
                    active_channels.push(ActiveChannel {
                        name: "whatsapp".to_string(), stop, handle,
                    });
                    println!("  WhatsApp running in background. Continue chatting.\n");
                    continue;
                }

                // Handle Telegram quick-start from REPL
                if input == "/telegram" || input == "/tg" {
                    active_channels.retain(|ch| !ch.handle.is_finished());
                    if active_channels.iter().any(|ch| ch.name == "telegram") {
                        println!("\n  Telegram is already running. Use /stop to stop channels.\n");
                        continue;
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
                        let t = match rl.readline(tok_prompt) {
                            Ok(line) => line.trim().to_string(),
                            Err(_) => { continue; }
                        };
                        if t.is_empty() {
                            println!("  Cancelled.\n");
                            continue;
                        }
                        print!("  Validating token... ");
                        io::stdout().flush().ok();
                        if cli::validate_telegram_token(&t) {
                            println!("valid!\n");
                        } else {
                            println!("invalid!");
                            println!("  Check the token and try again.\n");
                            continue;
                        }
                        let save_prompt = "  Save token to config for next time? [Y/n] ";
                        if let Ok(answer) = rl.readline(save_prompt) {
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
                    let dtx = display_tx.clone();
                    let ch = core_handle.clone();
                    let handle = tokio::spawn(async move {
                        cli::run_gateway_async(gw_config, ch, Some(stop2), Some(dtx)).await;
                    });
                    active_channels.push(ActiveChannel {
                        name: "telegram".to_string(), stop, handle,
                    });
                    println!("  Telegram running in background. Continue chatting.\n");
                    continue;
                }

                // Handle Email quick-start from REPL
                if input == "/email" {
                    active_channels.retain(|ch| !ch.handle.is_finished());
                    if active_channels.iter().any(|ch| ch.name == "email") {
                        println!("\n  Email is already running. Use /stop to stop channels.\n");
                        continue;
                    }
                    println!();
                    let mut gw_config = load_config(None);
                    cli::check_api_key(&gw_config);
                    let email_cfg = &gw_config.channels.email;
                    if email_cfg.imap_host.is_empty() || email_cfg.username.is_empty() || email_cfg.password.is_empty() {
                        println!("  Email not configured. Run `nanobot email` first or add settings to config.json.\n");
                        continue;
                    }
                    println!("  Starting Email channel...");
                    println!("  Polling {}", email_cfg.imap_host);
                    gw_config.channels.email.enabled = true;
                    gw_config.channels.whatsapp.enabled = false;
                    gw_config.channels.telegram.enabled = false;
                    gw_config.channels.feishu.enabled = false;
                    let stop = Arc::new(AtomicBool::new(false));
                    let stop2 = stop.clone();
                    let dtx = display_tx.clone();
                    let ch = core_handle.clone();
                    let handle = tokio::spawn(async move {
                        cli::run_gateway_async(gw_config, ch, Some(stop2), Some(dtx)).await;
                    });
                    active_channels.push(ActiveChannel {
                        name: "email".to_string(), stop, handle,
                    });
                    println!("  Email running in background. Continue chatting.\n");
                    continue;
                }

                // Handle stop command — stop all background channels
                if input == "/stop" {
                    active_channels.retain(|ch| !ch.handle.is_finished());
                    if active_channels.is_empty() {
                        println!("\n  No channels running.\n");
                    } else {
                        let names: Vec<String> = active_channels.iter().map(|c| c.name.clone()).collect();
                        println!("\n  Stopping: {}", names.join(", "));
                        for ch in &active_channels {
                            ch.stop.store(true, Ordering::Relaxed);
                        }
                        tokio::time::sleep(Duration::from_secs(1)).await;
                        for ch in &active_channels {
                            ch.handle.abort();
                        }
                        active_channels.clear();
                        println!("  All channels stopped.\n");
                    }
                    continue;
                }

                // Handle help command
                if input == "/paste" || input == "/p" {
                    println!("  {}Paste mode: type or paste text, then enter --- on its own line to send{}", tui::DIM, tui::RESET);
                    let mut lines: Vec<String> = Vec::new();
                    let stdin = io::stdin();
                    for line in stdin.lock().lines() {
                        match line {
                            Ok(l) if l.trim() == "---" => break,
                            Ok(l) => lines.push(l),
                            Err(_) => break,
                        }
                    }
                    let pasted = lines.join("\n").trim().to_string();
                    if pasted.is_empty() {
                        continue;
                    }
                    let _ = rl.add_history_entry(&pasted);
                    let channel = if voice_on { "voice" } else { "cli" };
                    stream_and_render(&mut agent_loop, &pasted, &session_id, channel, None).await;
                    println!();
                    {
                        let sa_count = agent_loop.subagent_manager().get_running_count().await;
                        active_channels.retain(|ch| !ch.handle.is_finished());
                        let ch_names: Vec<&str> = active_channels.iter().map(|c| short_channel_name(&c.name)).collect();
                        tui::print_status_bar(&core_handle, &ch_names, sa_count);
                    }
                    continue;
                }

                if input == "/help" || input == "/h" || input == "/?" {
                    print_help();
                    continue;
                }

                // Handle /agents command — list running subagents
                if input == "/agents" || input == "/a" {
                    let agents = agent_loop.subagent_manager().list_running().await;
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
                    continue;
                }

                // Handle /kill command — cancel a subagent
                if input.starts_with("/kill ") {
                    let id = input[6..].trim();
                    if id.is_empty() {
                        println!("\n  Usage: /kill <id>\n");
                    } else if agent_loop.subagent_manager().cancel(id).await {
                        println!("\n  Cancelled agent {}.\n", id);
                    } else {
                        println!("\n  No running agent matching '{}'.\n", id);
                    }
                    continue;
                }

                // Handle status command
                if input == "/status" || input == "/s" {
                    let core = core_handle.read().unwrap().clone();
                    let is_local = crate::LOCAL_MODE.load(Ordering::SeqCst);
                    let model_name = &core.model;
                    let mode_label = if is_local { "local" } else { "cloud" };

                    println!();
                    println!("  {}MODE{}      {} ({})", tui::BOLD, tui::RESET, mode_label, model_name);

                    let used = core.last_context_used.load(Ordering::Relaxed) as usize;
                    let max = core.last_context_max.load(Ordering::Relaxed) as usize;
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

                    let agent_count = agent_loop.subagent_manager().get_running_count().await;
                    println!("  {}AGENTS{}    {} running", tui::BOLD, tui::RESET, agent_count);

                    active_channels.retain(|ch| !ch.handle.is_finished());
                    if !active_channels.is_empty() {
                        let ch_names: Vec<&str> = active_channels.iter().map(|c| short_channel_name(&c.name)).collect();
                        println!("  {}CHANNELS{}  {}", tui::BOLD, tui::RESET, ch_names.join(" "));
                    }

                    let turn = core.learning_turn_counter.load(Ordering::Relaxed);
                    println!("  {}TURN{}      {}", tui::BOLD, tui::RESET, turn);

                    if is_local {
                        if let Some(ref cp) = srv.compaction_port {
                            println!("  {}COMPACT{}   on port {} (CPU)", tui::BOLD, tui::RESET, cp);
                        }
                    }

                    println!();
                    continue;
                }

                // Process message (streaming)
                let channel = if voice_on { "voice" } else { "cli" };
                let response = stream_and_render(&mut agent_loop, input, &session_id, channel, None).await;
                println!();
                {
                    let sa_count = agent_loop.subagent_manager().get_running_count().await;
                    active_channels.retain(|ch| !ch.handle.is_finished());
                    let ch_names: Vec<&str> = active_channels.iter().map(|c| short_channel_name(&c.name)).collect();
                    tui::print_status_bar(&core_handle, &ch_names, sa_count);
                }

                #[cfg(feature = "voice")]
                if let Some(ref mut vs) = voice_session {
                    let tts_text = tui::strip_markdown_for_tts(&response);
                    if !tts_text.is_empty() {
                        tui::speak_interruptible(vs, &tts_text, "en");
                    }
                }
            }
            // Stop any active background channels
            for ch in &active_channels {
                ch.stop.store(true, Ordering::Relaxed);
            }
            if !active_channels.is_empty() {
                tokio::time::sleep(Duration::from_millis(500)).await;
                for ch in &active_channels {
                    ch.handle.abort();
                }
            }

            // Cleanup: kill llama.cpp server if still running
            // Save readline history
            let _ = rl.save_history(&history_path);

            srv.shutdown();

            println!("Goodbye!");
        }
    });
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
