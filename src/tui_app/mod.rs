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

mod app;
mod render;

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use ratatui::crossterm::event::{self, DisableBracketedPaste, EnableBracketedPaste, Event};
use ratatui::crossterm::execute;
use ratatui::DefaultTerminal;
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};
use tokio::time::MissedTickBehavior;
use tokio_util::sync::CancellationToken;

use crate::agent::agent_loop::{AgentLoop, SharedCoreHandle};
use crate::agent::audit::ToolEvent;
use app::{Action, App, Footer};

/// Immutable per-session handles the UI loop needs to drive the agent.
struct Session<'a> {
    agent: &'a AgentLoop,
    core: &'a SharedCoreHandle,
    session_id: &'a str,
    lang: Option<&'a str>,
}

/// Restores terminal modes on drop, including on panic. `ratatui::init`'s panic
/// hook restores the alt-screen/raw-mode, but not bracketed paste or the reader
/// thread — this guard closes those gaps. Idempotent with the normal-path
/// cleanup (disabling paste / restoring twice is harmless).
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
pub(crate) async fn run(
    agent: &AgentLoop,
    core_handle: &SharedCoreHandle,
    session_id: &str,
    lang: Option<&str>,
) -> std::io::Result<()> {
    let mut terminal = ratatui::init();
    let stop = Arc::new(AtomicBool::new(false));
    let (ev_tx, mut ev_rx) = unbounded_channel::<Event>();
    let reader = spawn_event_reader(ev_tx, stop.clone());
    let _guard = TerminalGuard { stop: stop.clone() };
    let _ = execute!(std::io::stdout(), EnableBracketedPaste);

    let session = Session {
        agent,
        core: core_handle,
        session_id,
        lang,
    };
    let mut app = App::new();
    let result = event_loop(&mut terminal, &mut app, &session, &mut ev_rx).await;

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
fn home_relative(p: &Path) -> String {
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

/// Blocking crossterm event reader. Polls so it can observe `stop` between
/// events and exit promptly when the UI shuts down.
fn spawn_event_reader(tx: UnboundedSender<Event>, stop: Arc<AtomicBool>) -> JoinHandle<()> {
    std::thread::spawn(move || {
        while !stop.load(Ordering::Relaxed) {
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

/// Idle loop: redraw, wait for the next event, dispatch. A submitted message
/// hands off to `run_turn` for the duration of the agent's response.
async fn event_loop(
    terminal: &mut DefaultTerminal,
    app: &mut App,
    session: &Session<'_>,
    ev_rx: &mut UnboundedReceiver<Event>,
) -> std::io::Result<()> {
    loop {
        let footer = footer_snapshot(session.core);
        terminal.draw(|f| app.draw(f, &footer))?;
        let Some(ev) = ev_rx.recv().await else { break };
        match app.on_idle_event(ev) {
            Action::Quit => break,
            Action::Continue => {}
            Action::Submit(text) => {
                if let Some(cmd) = text.strip_prefix('/') {
                    match cmd {
                        "quit" | "exit" | "q" => break,
                        "help" | "?" => app.set_help(true),
                        "clear" => app.clear_transcript(),
                        _ => app.push_note(format!(
                            "/{cmd} isn't in the TUI yet — type /help. For /model, /voice, /local, /think, /status use the classic REPL (run without NANOBOT_TUI)."
                        )),
                    }
                    continue;
                }
                run_turn(terminal, app, session, &text, ev_rx).await?;
            }
        }
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
) -> std::io::Result<()> {
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
    app.finish_turn(response);
    let footer = footer_snapshot(session.core);
    terminal.draw(|f| app.draw(f, &footer))?;
    Ok(())
}
