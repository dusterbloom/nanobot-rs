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
//! Classic slash commands (`/model`, `/local`, `/think`, `/status`, …) that
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

use ratatui::crossterm::cursor::{Hide, Show};
use ratatui::crossterm::event::{self, DisableBracketedPaste, EnableBracketedPaste, Event};
use ratatui::crossterm::execute;
use ratatui::crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::DefaultTerminal;
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};
use tokio::time::MissedTickBehavior;
use tokio_util::sync::CancellationToken;

use crate::agent::agent_loop::{AgentLoop, SharedCoreHandle};
use crate::agent::audit::ToolEvent;
use crate::repl::commands::ReplContext;
use app::{Action, App, Footer};

/// Immutable per-session handles a single turn needs to drive the agent.
struct Session<'a> {
    agent: &'a AgentLoop,
    core: &'a SharedCoreHandle,
    session_id: &'a str,
    lang: Option<&'a str>,
}

/// Restores terminal modes on drop, including on panic. `ratatui::init`'s panic
/// hook restores the alt-screen/raw-mode, but not bracketed paste or the reader
/// thread — this guard closes those gaps. Idempotent with the normal cleanup.
struct TerminalGuard {
    stop: Arc<AtomicBool>,
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        let _ = execute!(std::io::stdout(), DisableBracketedPaste);
        ratatui::restore();
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
    let _guard = TerminalGuard { stop: stop.clone() };
    let _ = execute!(std::io::stdout(), EnableBracketedPaste);

    let mut app = App::new();
    let result = event_loop(&mut terminal, &mut app, ctx, &mut ev_rx, &paused).await;

    stop.store(true, Ordering::Relaxed);
    let _ = reader.join();
    result
    // `_guard` drops here (and on panic): DisableBracketedPaste + ratatui::restore().
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
    loop {
        let footer = footer_snapshot(&ctx.core_handle);
        terminal.draw(|f| app.draw(f, &footer))?;
        let Some(ev) = ev_rx.recv().await else { break };
        match app.on_idle_event(ev) {
            Action::Quit => break,
            Action::Continue => {}
            Action::Submit(text) => {
                if let Some(rest) = text.strip_prefix('/') {
                    if slash_command(terminal, app, ctx, &text, rest, ev_rx, paused).await? {
                        break; // /quit
                    }
                    continue;
                }
                let session = Session {
                    agent: &ctx.agent_loop,
                    core: &ctx.core_handle,
                    session_id: &ctx.session_id,
                    lang: ctx.lang.as_deref(),
                };
                run_turn(terminal, app, &session, &text, ev_rx).await?;
            }
            Action::Record => {
                #[cfg(feature = "voice")]
                voice_cycle(terminal, app, ctx, ev_rx, paused).await?;
            }
        }
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
        "clear" => app.clear_transcript(),
        "voice" | "v" => {
            // Toggle voice via the classic dispatcher, then reflect the new
            // state so the UI knows Enter-on-empty should record.
            run_classic_command(terminal, app, ctx, full, ev_rx, paused).await?;
            app.set_voice(ctx.voice_on());
        }
        _ => run_classic_command(terminal, app, ctx, full, ev_rx, paused).await?,
    }
    Ok(false)
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
        Show
    )?;
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
        print!("\n\x1b[2m— press Enter to return to TRENTADUE —\x1b[0m ");
        let _ = std::io::stdout().flush();
        let mut buf = String::new();
        let _ = std::io::stdin().read_line(&mut buf);
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
    suspend_ui(paused)?;
    let captured = ctx
        .voice_session
        .as_mut()
        .map(|vs| vs.record_and_transcribe());
    resume_ui(terminal, ev_rx, paused)?;

    match captured {
        Some(Ok(Some((text, lang)))) => {
            let session = Session {
                agent: &ctx.agent_loop,
                core: &ctx.core_handle,
                session_id: &ctx.session_id,
                lang: ctx.lang.as_deref(),
            };
            let reply = run_turn(terminal, app, &session, &text, ev_rx).await?;
            if !reply.trim().is_empty() {
                if let Some(vs) = ctx.voice_session.as_mut() {
                    let _ = vs.speak(&reply, &lang);
                }
            }
        }
        Some(Ok(None)) => app.push_note("(no speech detected)".into()),
        Some(Err(e)) => app.push_note(format!("voice error: {e}")),
        None => {}
    }
    Ok(())
}

/// Drive a single agent turn, streaming text and tool events into the
/// transcript and redrawing on every update until the response completes.
async fn run_turn(
    terminal: &mut DefaultTerminal,
    app: &mut App,
    session: &Session<'_>,
    input: &str,
    ev_rx: &mut UnboundedReceiver<Event>,
) -> std::io::Result<String> {
    app.begin_turn(input);

    let (delta_tx, mut delta_rx) = unbounded_channel::<String>();
    let (tool_tx, mut tool_rx) = unbounded_channel::<ToolEvent>();
    let cancel = CancellationToken::new();

    let fut = session.agent.process_direct_streaming(
        input,
        session.session_id,
        "cli",
        "direct",
        session.lang,
        delta_tx,
        Some(tool_tx),
        Some(cancel.clone()),
        None,
    );
    tokio::pin!(fut);

    let mut tick = tokio::time::interval(Duration::from_millis(80));
    tick.set_missed_tick_behavior(MissedTickBehavior::Skip);

    let response = loop {
        let footer = footer_snapshot(session.core);
        terminal.draw(|f| app.draw(f, &footer))?;
        tokio::select! {
            biased;
            resp = &mut fut => break resp,
            Some(d) = delta_rx.recv() => app.on_delta(&d),
            Some(e) = tool_rx.recv() => app.on_tool_event(e),
            Some(ev) = ev_rx.recv() => {
                if app.on_streaming_event(ev) {
                    cancel.cancel();
                }
            }
            _ = tick.tick() => app.tick(0.08),
        }
    };

    // Drain anything buffered after the agent returned (deltas can land just
    // before the future resolves under `biased`).
    while let Ok(d) = delta_rx.try_recv() {
        app.on_delta(&d);
    }
    while let Ok(e) = tool_rx.try_recv() {
        app.on_tool_event(e);
    }
    let reply = response.clone();
    app.finish_turn(response);
    let footer = footer_snapshot(session.core);
    terminal.draw(|f| app.draw(f, &footer))?;
    Ok(reply)
}
