//! Full-screen ratatui UI for `nanobot agent`, opt-in via `NANOBOT_TUI=1`.
//!
//! This is the in-progress replacement for the classic pinned-bar REPL. It runs
//! the same agent streaming surface (`process_direct_streaming`) but renders
//! into a re-renderable transcript that reflows on resize and supports a real
//! multi-line input box (`ratatui-textarea`). The classic renderer remains the
//! default until this reaches parity.
//!
//! Architecture: one async task owns the terminal and the [`App`] state. A
//! dedicated OS thread reads crossterm events (blocking) and forwards them over
//! an mpsc channel, so the event loop can `select!` across input, the agent's
//! text-delta and tool-event channels, and an animation tick — without ever
//! holding a blocking read across an `.await`.
//!
//! Classic slash commands (`/local`, `/think`, `/status`, …) that
//! print ANSI or open interactive pickers can't run inside the alt-screen, so
//! they use a suspend/resume bridge: the UI leaves the alt-screen, pauses the
//! input reader (freeing stdin), runs the real `ReplContext::dispatch`, then
//! re-enters and redraws.

mod app;
mod render;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use ratatui::crossterm::cursor::{Hide, MoveTo, Show};
use ratatui::crossterm::event::{
    self, DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture,
    Event,
};
use ratatui::crossterm::execute;
use ratatui::crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, Clear, ClearType, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::DefaultTerminal;
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};
use tokio::time::MissedTickBehavior;
use tokio_util::sync::CancellationToken;

use crate::agent::agent_loop::{AgentLoop, SharedCoreHandle};
use crate::agent::audit::ToolEvent;
use crate::repl::commands::{unique_direct_model_match, ModelEntry, ReplContext};
use app::{draw_outro, Action, App, Footer, StreamingAction, SubmittedTurn};

/// Immutable per-session handles a single turn needs to drive the agent.
struct Session<'a> {
    agent: &'a AgentLoop,
    core: &'a SharedCoreHandle,
    session_id: &'a str,
    lang: Option<&'a str>,
}

/// Result of one streaming assistant turn.
struct TurnOutcome {
    #[cfg(feature = "voice")]
    reply: String,
    queued_turn: Option<SubmittedTurn>,
}

/// Restores terminal modes on drop, including on panic. `ratatui::init`'s panic
/// hook restores the alt-screen/raw-mode, but not bracketed paste or the reader
/// thread — this guard closes those gaps. Idempotent with the normal cleanup.
struct TerminalGuard {
    stop: Arc<AtomicBool>,
    restored: bool,
}

impl TerminalGuard {
    fn restore(&mut self) {
        if self.restored {
            return;
        }
        self.stop.store(true, Ordering::Relaxed);
        let _ = execute!(
            std::io::stdout(),
            DisableBracketedPaste,
            DisableMouseCapture
        );
        ratatui::restore();
        self.restored = true;
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        self.restore();
    }
}

/// Whether the full-screen ratatui UI is requested (`NANOBOT_TUI=1`/`true`).
pub(crate) fn enabled() -> bool {
    std::env::var("NANOBOT_TUI")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// Run the full-screen UI for one interactive session. Restores the terminal
/// before returning, even on panic (via [`TerminalGuard`]).
pub(crate) async fn run(ctx: &mut ReplContext) -> std::io::Result<()> {
    let mut terminal = ratatui::init();
    let stop = Arc::new(AtomicBool::new(false));
    let paused = Arc::new(AtomicBool::new(false));
    let (ev_tx, mut ev_rx) = unbounded_channel::<Event>();
    let reader = spawn_event_reader(ev_tx, stop.clone(), paused.clone());
    let mut guard = TerminalGuard {
        stop: stop.clone(),
        restored: false,
    };
    let _ = execute!(std::io::stdout(), EnableBracketedPaste, EnableMouseCapture);

    let mut app = App::new();
    let result = event_loop(&mut terminal, &mut app, ctx, &mut ev_rx, &paused).await;

    // Native farewell frame before the terminal is restored.
    if result.is_ok() {
        let _ = terminal.draw(draw_outro);
        std::thread::sleep(Duration::from_millis(600));
    }
    stop.store(true, Ordering::Relaxed);
    let _ = reader.join();
    guard.restore();
    if result.is_ok() {
        clear_normal_screen();
    }
    result
    // `guard` drops here (and on panic): DisableBracketedPaste + ratatui::restore().
}

fn clear_normal_screen() {
    let _ = execute!(std::io::stdout(), Clear(ClearType::All), MoveTo(0, 0), Show);
}

fn model_accepts_images(model: &str) -> bool {
    let model = model.to_ascii_lowercase();
    [
        "vision",
        "vl",
        "llava",
        "pixtral",
        "gpt-4o",
        "o3",
        "o4",
        "gemini",
        "claude-3",
        "qwen-vl",
        "qwen2.5-vl",
        "qwen3-vl",
    ]
    .iter()
    .any(|marker| model.contains(marker))
}

/// Snapshot the quiet footer state (cwd / model / context usage) from the core.
fn footer_snapshot(core: &SharedCoreHandle) -> Footer {
    let used = core.counters.last_context_used.load(Ordering::Relaxed) as usize;
    let max = core.counters.last_context_max.load(Ordering::Relaxed) as usize;
    let model = core.swappable().model.clone();
    let cwd = std::env::current_dir()
        .ok()
        .map(|p| home_relative(&p))
        .unwrap_or_else(|| "?".into());
    Footer {
        cwd,
        model,
        ctx_used: used,
        ctx_max: max,
    }
}

/// Abbreviate a path under `$HOME` to `~/…`. Matches on a path-segment boundary
/// so a sibling like `/home/userfoo` is not mistaken for being under `/home/user`.
fn home_relative(p: &std::path::Path) -> String {
    let s = p.display().to_string();
    if let Ok(home) = std::env::var("HOME") {
        if !home.is_empty() {
            if s == home {
                return "~".to_string();
            }
            if let Some(rest) = s.strip_prefix(&format!("{home}/")) {
                return format!("~/{rest}");
            }
        }
    }
    s
}

/// Blocking crossterm event reader. Polls so it can observe `stop`/`paused`
/// between events; while paused it leaves stdin alone for the classic bridge.
fn spawn_event_reader(
    tx: UnboundedSender<Event>,
    stop: Arc<AtomicBool>,
    paused: Arc<AtomicBool>,
) -> JoinHandle<()> {
    std::thread::spawn(move || {
        while !stop.load(Ordering::Relaxed) {
            if paused.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(20));
                continue;
            }
            match event::poll(Duration::from_millis(100)) {
                Ok(true) => match event::read() {
                    Ok(ev) => {
                        if tx.send(ev).is_err() {
                            break;
                        }
                    }
                    Err(_) => break,
                },
                Ok(false) => {}
                Err(_) => break,
            }
        }
    })
}

/// Idle loop: redraw, wait for the next event, dispatch. Plain messages stream
/// via `run_turn`; classic slash commands go through the suspend/resume bridge.
async fn event_loop(
    terminal: &mut DefaultTerminal,
    app: &mut App,
    ctx: &mut ReplContext,
    ev_rx: &mut UnboundedReceiver<Event>,
    paused: &Arc<AtomicBool>,
) -> std::io::Result<()> {
    'ui: loop {
        let footer = footer_snapshot(&ctx.core_handle);
        terminal.draw(|f| app.draw(f, &footer))?;
        let Some(ev) = ev_rx.recv().await else { break };
        match app.on_idle_event(ev) {
            Action::Quit => break,
            Action::Continue => {}
            Action::Submit(turn) => {
                let mut queued = Some(turn);
                while let Some(mut turn) = queued.take() {
                    if turn.media.is_empty() {
                        if let Some(rest) = turn.text.strip_prefix('/') {
                            let text = turn.text.clone();
                            if slash_command(terminal, app, ctx, &text, rest, ev_rx, paused).await?
                            {
                                break 'ui; // /quit
                            }
                            continue;
                        }
                    }
                    let session = Session {
                        agent: &ctx.agent_loop,
                        core: &ctx.core_handle,
                        session_id: &ctx.session_id,
                        lang: ctx.lang.as_deref(),
                    };
                    if !turn.media.is_empty() {
                        let model = session.core.swappable().model.clone();
                        if !model_accepts_images(&model) {
                            let suffix = if turn.media.len() == 1 { "" } else { "s" };
                            app.push_note(format!(
                                "{} image{} not sent — {model} does not advertise image input",
                                turn.media.len(),
                                suffix
                            ));
                            turn.media.clear();
                        }
                    }
                    queued = run_turn(terminal, app, &session, &turn, ev_rx)
                        .await?
                        .queued_turn;
                }
            }
            Action::Record => {
                #[cfg(feature = "voice")]
                voice_cycle(terminal, app, ctx, ev_rx, paused).await?;
            }
            Action::PickModel(entry) => apply_model_selection(terminal, app, ctx, entry).await?,
        }
    }
    Ok(())
}

async fn apply_model_selection(
    terminal: &mut DefaultTerminal,
    app: &mut App,
    ctx: &mut ReplContext,
    entry: ModelEntry,
) -> std::io::Result<()> {
    let id = entry.id.clone();
    app.push_note(format!("switching to {id} ..."));
    let footer = footer_snapshot(&ctx.core_handle);
    terminal.draw(|f| app.draw(f, &footer))?;
    match ctx.apply_model_entry(entry).await {
        Ok(report) => {
            for note in report.notes {
                app.push_note(note);
            }
            app.push_note(format!("model switched to {}", report.model_id));
        }
        Err(e) => app.push_note(format!("model switch failed: {e}")),
    }
    Ok(())
}

/// Route a slash command. Returns `Ok(true)` when the UI should exit (`/quit`).
/// UI-local commands are handled inline; everything else goes to the classic
/// dispatcher via the suspend/resume bridge.
async fn slash_command(
    terminal: &mut DefaultTerminal,
    app: &mut App,
    ctx: &mut ReplContext,
    full: &str,
    rest: &str,
    ev_rx: &mut UnboundedReceiver<Event>,
    paused: &Arc<AtomicBool>,
) -> std::io::Result<bool> {
    match rest.split_whitespace().next().unwrap_or("") {
        "quit" | "exit" | "q" => return Ok(true),
        "help" | "?" => app.set_help(true),
        "clear" => {
            ctx.clear_session_state().await;
            app.clear_transcript();
        }
        "mode" => {
            let arg = rest.strip_prefix("mode").map(str::trim).unwrap_or("");
            if arg.is_empty() {
                app.cycle_mode();
            } else if !app.set_mode(arg) {
                app.push_note(format!("unknown mode '{arg}' — use calm | inspect | deep"));
            }
        }
        "model" | "m" => {
            if ctx.model_picker_available() {
                let filter = model_command_direct_arg(rest);
                let mut entries = ctx.collect_all_models().await;
                if let Some(filter) = filter {
                    let filter_lower = filter.trim().to_lowercase();
                    entries.retain(|e| e.id.to_lowercase().contains(&filter_lower));
                    if entries.is_empty() {
                        app.push_note(format!("no models matching \"{filter}\""));
                    } else {
                        let refs: Vec<&ModelEntry> = entries.iter().collect();
                        if let Some(selected) =
                            unique_direct_model_match(&refs, &filter_lower).cloned()
                        {
                            apply_model_selection(terminal, app, ctx, selected).await?;
                        } else {
                            app.open_model_picker(entries);
                        }
                    }
                } else if entries.is_empty() {
                    app.push_note("no models found".into());
                } else {
                    app.open_model_picker(entries);
                }
            } else {
                app.push_note("/model is only available in local mode — use /local".into());
            }
        }
        "voice" | "v" => {
            #[cfg(feature = "voice")]
            {
                // Toggle natively — no alt-screen switch (the bridge leaked the
                // classic startup screen). Header reflects the new state.
                let on = ctx.toggle_voice().await;
                app.set_voice(on);
                app.push_note(if on {
                    "voice on — press Enter on an empty line to speak".to_string()
                } else {
                    "voice off".to_string()
                });
            }
            #[cfg(not(feature = "voice"))]
            app.push_note(
                "voice needs a build with the feature — rebuild with: cargo run --release --features voice,cluster -- agent".into(),
            );
        }
        _ => run_classic_command(terminal, app, ctx, full, ev_rx, paused).await?,
    }
    Ok(false)
}

fn model_command_direct_arg(rest: &str) -> Option<&str> {
    let trimmed = rest.trim();
    let command = trimmed.split_whitespace().next()?;
    if !matches!(command, "model" | "m") {
        return None;
    }
    let arg = trimmed[command.len()..].trim();
    (!arg.is_empty()).then_some(arg)
}

/// Leave the alt-screen and pause the reader so stdin is free for a classic
/// command's picker or for voice recording (both read stdin themselves).
fn suspend_ui(paused: &Arc<AtomicBool>) -> std::io::Result<()> {
    paused.store(true, Ordering::Relaxed);
    std::thread::sleep(Duration::from_millis(120)); // let an in-flight poll finish
    disable_raw_mode()?;
    execute!(
        std::io::stdout(),
        LeaveAlternateScreen,
        DisableBracketedPaste,
        DisableMouseCapture,
        Show
    )?;
    clear_normal_screen();
    Ok(())
}

/// Re-enter the alt-screen, resume the reader, drop stray input, repaint.
fn resume_ui(
    terminal: &mut DefaultTerminal,
    ev_rx: &mut UnboundedReceiver<Event>,
    paused: &Arc<AtomicBool>,
) -> std::io::Result<()> {
    enable_raw_mode()?;
    execute!(
        std::io::stdout(),
        EnterAlternateScreen,
        EnableBracketedPaste,
        EnableMouseCapture,
        Hide
    )?;
    terminal.clear()?;
    paused.store(false, Ordering::Relaxed);
    while ev_rx.try_recv().is_ok() {}
    Ok(())
}

/// Suspend the alt-screen, run the classic `dispatch` (which may print ANSI or
/// open an interactive picker), then resume and redraw.
async fn run_classic_command(
    terminal: &mut DefaultTerminal,
    app: &mut App,
    ctx: &mut ReplContext,
    input: &str,
    ev_rx: &mut UnboundedReceiver<Event>,
    paused: &Arc<AtomicBool>,
) -> std::io::Result<()> {
    use std::io::Write as _;

    suspend_ui(paused)?;
    println!();
    let handled = ctx.dispatch(input).await;
    if handled {
        // Wait for ANY key (Esc included). A line-read here only returned on
        // Enter, so pressing Esc left the user wedged (→ Ctrl+C to escape).
        let _ = enable_raw_mode();
        print!("\r\n  \x1b[2mpress any key to return to TRENTADUE\x1b[0m ");
        let _ = std::io::stdout().flush();
        let _ = event::read();
        clear_normal_screen();
    }
    resume_ui(terminal, ev_rx, paused)?;

    if !handled {
        app.push_note(format!("unknown command: {input}"));
    }
    Ok(())
}

/// Voice turn: suspend (so the recorder owns stdin), record + transcribe, resume,
/// run the agent turn, then speak the reply. Recording is blocking, hence the
/// suspend — `record_and_transcribe` reads keys for its own stop control.
#[cfg(feature = "voice")]
async fn voice_cycle(
    terminal: &mut DefaultTerminal,
    app: &mut App,
    ctx: &mut ReplContext,
    ev_rx: &mut UnboundedReceiver<Event>,
    paused: &Arc<AtomicBool>,
) -> std::io::Result<()> {
    loop {
        // --- record (inline, no screen switch) ---
        app.set_recording(true);
        let footer = footer_snapshot(&ctx.core_handle);
        terminal.draw(|f| app.draw(f, &footer))?;
        paused.store(true, Ordering::Relaxed);
        std::thread::sleep(Duration::from_millis(120));
        let captured = ctx
            .voice_session
            .as_mut()
            .map(|vs| vs.record_and_transcribe());
        let _ = enable_raw_mode(); // recorder toggles its own raw mode
        paused.store(false, Ordering::Relaxed);
        while ev_rx.try_recv().is_ok() {}
        app.set_recording(false);
        terminal.clear()?;

        let (text, lang) = match captured {
            Some(Ok(Some(t))) => t,
            Some(Ok(None)) => {
                app.push_note("(no speech detected)".into());
                return Ok(());
            }
            Some(Err(e)) => {
                app.push_note(format!("voice error: {e}"));
                return Ok(());
            }
            None => return Ok(()),
        };

        // --- reply ---
        let reply = {
            let session = Session {
                agent: &ctx.agent_loop,
                core: &ctx.core_handle,
                session_id: &ctx.session_id,
                lang: ctx.lang.as_deref(),
            };
            let outcome = run_turn(
                terminal,
                app,
                &session,
                &SubmittedTurn {
                    text: text.clone(),
                    media: Vec::new(),
                },
                ev_rx,
            )
            .await?;
            outcome.reply
        };
        if reply.trim().is_empty() {
            return Ok(());
        }

        // --- speak, interruptible: Enter / Ctrl+Space drops TTS instantly ---
        app.set_speaking(true);
        let footer = footer_snapshot(&ctx.core_handle);
        terminal.draw(|f| app.draw(f, &footer))?;
        let interrupted = {
            let Some(vs) = ctx.voice_session.as_mut() else {
                return Ok(());
            };
            vs.clear_cancel();
            let cancel = vs.cancel_flag();
            paused.store(true, Ordering::Relaxed); // free stdin for the watcher
            std::thread::sleep(Duration::from_millis(60));
            let done = Arc::new(AtomicBool::new(false));
            let watcher = crate::tui::spawn_interrupt_watcher(cancel, done.clone());
            let _ = vs.speak(&reply, &lang); // blocks; cancel flag stops it early
            done.store(true, Ordering::Relaxed);
            let i = watcher.join().unwrap_or(false);
            let _ = enable_raw_mode();
            paused.store(false, Ordering::Relaxed);
            i
        };
        while ev_rx.try_recv().is_ok() {}
        app.set_speaking(false);

        // Barge-in: if the user interrupted, record again immediately.
        if !interrupted {
            return Ok(());
        }
    }
}

/// Drive a single agent turn, streaming text and tool events into the
/// transcript and redrawing on every update until the response completes.
async fn run_turn(
    terminal: &mut DefaultTerminal,
    app: &mut App,
    session: &Session<'_>,
    turn: &SubmittedTurn,
    ev_rx: &mut UnboundedReceiver<Event>,
) -> std::io::Result<TurnOutcome> {
    app.begin_turn(&turn.display_text());

    let (delta_tx, mut delta_rx) = unbounded_channel::<String>();
    let (tool_tx, mut tool_rx) = unbounded_channel::<ToolEvent>();
    let cancel = CancellationToken::new();

    let fut = session.agent.process_direct_streaming(
        &turn.text,
        session.session_id,
        "cli",
        "direct",
        session.lang,
        delta_tx,
        Some(tool_tx),
        Some(cancel.clone()),
        None,
        (!turn.media.is_empty()).then_some(turn.media.as_slice()),
    );
    tokio::pin!(fut);

    let mut tick = tokio::time::interval(Duration::from_millis(80));
    tick.set_missed_tick_behavior(MissedTickBehavior::Skip);

    let mut queued_turn = None;
    let mut cancel_requested = false;
    let response = loop {
        let footer = footer_snapshot(session.core);
        terminal.draw(|f| app.draw(f, &footer))?;
        tokio::select! {
            biased;
            Some(ev) = ev_rx.recv() => {
                match app.on_streaming_event(ev) {
                    StreamingAction::Continue => {}
                    StreamingAction::Cancel => {
                        cancel_requested = true;
                        cancel.cancel();
                    }
                    StreamingAction::CancelAndSubmit(turn) => {
                        queued_turn = Some(turn);
                        cancel_requested = true;
                        cancel.cancel();
                    }
                }
            }
            resp = &mut fut => break resp,
            Some(d) = delta_rx.recv(), if !cancel_requested => app.on_delta(&d),
            Some(e) = tool_rx.recv(), if !cancel_requested => {
                drain_pending_deltas(app, &mut delta_rx);
                app.on_tool_event(e);
            }
            _ = tick.tick() => app.tick(0.08),
        }
    };

    // Drain anything buffered after the agent returned (deltas can land just
    // before the future resolves under `biased`).
    if cancel_requested {
        while delta_rx.try_recv().is_ok() {}
        while tool_rx.try_recv().is_ok() {}
    } else {
        drain_pending_deltas(app, &mut delta_rx);
        while let Ok(e) = tool_rx.try_recv() {
            drain_pending_deltas(app, &mut delta_rx);
            app.on_tool_event(e);
        }
        drain_pending_deltas(app, &mut delta_rx);
    }
    #[cfg(feature = "voice")]
    let reply = if cancel_requested {
        String::new()
    } else {
        response.clone()
    };
    let response = if cancel_requested {
        String::new()
    } else {
        response
    };
    app.finish_turn(response);
    let footer = footer_snapshot(session.core);
    terminal.draw(|f| app.draw(f, &footer))?;
    Ok(TurnOutcome {
        #[cfg(feature = "voice")]
        reply,
        queued_turn,
    })
}

fn drain_pending_deltas(app: &mut App, delta_rx: &mut UnboundedReceiver<String>) {
    while let Ok(d) = delta_rx.try_recv() {
        app.on_delta(&d);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    fn buffer_text(buf: &ratatui::buffer::Buffer) -> String {
        let area = *buf.area();
        let mut s = String::new();
        for y in 0..area.height {
            for x in 0..area.width {
                if let Some(c) = buf.cell((x, y)) {
                    s.push_str(c.symbol());
                }
            }
            s.push('\n');
        }
        s
    }

    #[test]
    fn draining_deltas_before_tool_events_preserves_setup_text_order() {
        let mut app = App::new();
        app.begin_turn("news");

        let (tx, mut delta_rx) = unbounded_channel::<String>();
        tx.send("Let me check first.".into()).unwrap();
        drain_pending_deltas(&mut app, &mut delta_rx);
        app.on_tool_event(ToolEvent::CallStart {
            tool_name: "exec".into(),
            tool_call_id: "c1".into(),
            arguments_preview: "news.py".into(),
        });

        let footer = Footer {
            cwd: "~/Dev/nanobot-rs".into(),
            model: "local:test".into(),
            ctx_used: 1,
            ctx_max: 10,
        };
        let mut term = Terminal::new(TestBackend::new(100, 12)).unwrap();
        term.draw(|f| app.draw(f, &footer)).unwrap();
        let text = buffer_text(term.backend().buffer());
        let reply = text.find("Let me check first.").expect("reply rendered");
        let tool = text.find("exec").expect("tool rendered");
        assert!(
            reply < tool,
            "setup text should render before tool:\n{text}"
        );
    }

    #[test]
    fn model_image_gate_recognizes_vision_models_only() {
        assert!(model_accepts_images("gpt-4o-mini"));
        assert!(model_accepts_images("qwen2.5-vl-7b"));
        assert!(!model_accepts_images("local:qwen36-35b"));
    }

    #[test]
    fn model_command_with_argument_extracts_filter() {
        assert_eq!(
            model_command_direct_arg("model VibeThinker-3B-mlx-8Bit"),
            Some("VibeThinker-3B-mlx-8Bit")
        );
        assert_eq!(model_command_direct_arg("model"), None);
        assert_eq!(model_command_direct_arg("m"), None);
    }
}
