//! Full-screen ratatui UI for `nanobot agent` — the default on real terminals.
//!
//! It runs the same agent streaming surface (`process_direct_streaming`) as the
//! classic pinned-bar REPL but renders into a re-renderable transcript that
//! reflows on resize and supports a real multi-line input box
//! (`ratatui-textarea`). `NANOBOT_TUI=0` opts back into the classic REPL;
//! `NANOBOT_TUI=1` forces the TUI even when the tty probe fails.
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
use crate::config::loader::{load_config, save_config};
use crate::repl::commands::{unique_direct_model_match, ModelEntry, ReplContext};
use crate::turn_stream::{Completion, TurnEvent, TurnStream};
use app::{
    draw_outro, Action, App, BackgroundJob, Footer, SessionPick, SessionRow, StreamingAction,
    SubmittedTurn,
};

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

/// Whether the full-screen ratatui UI should run. Default ON when both stdin
/// and stdout are real terminals; `NANOBOT_TUI=0`/`false` opts back into the
/// classic REPL, `NANOBOT_TUI=1`/`true` forces the TUI regardless of the probe.
pub(crate) fn enabled() -> bool {
    use std::io::IsTerminal;
    enabled_from(
        std::env::var("NANOBOT_TUI").ok().as_deref(),
        std::io::stdin().is_terminal() && std::io::stdout().is_terminal(),
    )
}

fn enabled_from(var: Option<&str>, is_tty: bool) -> bool {
    match var {
        Some(v) if v == "0" || v.eq_ignore_ascii_case("false") => false,
        Some(v) if v == "1" || v.eq_ignore_ascii_case("true") => true,
        _ => is_tty,
    }
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
    app.set_theme_index(ctx.config.agents.defaults.theme_index);

    // Paint a clean first frame (header + welcome + input) BEFORE the session
    // load below. Entering the alternate screen does not reliably erase what a
    // previous TUI run left in that buffer on every terminal, and the session
    // load can take a while (snapshot model restore may switch models) — without
    // this, stale content (e.g. an old jobs overlay + displaced header) stays
    // visible until the event loop's first draw.
    terminal.clear()?;
    let footer = footer_snapshot(&ctx.core_handle);
    terminal.draw(|f| app.draw(f, &footer))?;
    tracing::info!("tui_first_frame_drawn");

    load_current_session_state(&mut app, ctx).await;
    let result = event_loop(&mut terminal, &mut app, ctx, &mut ev_rx, &paused).await;
    save_current_snapshot(&app, ctx).await;

    // Persist the chosen color scheme (Ctrl+P) so it survives restarts.
    let mut disk = load_config(None);
    if disk.agents.defaults.theme_index != app.theme_index() {
        disk.agents.defaults.theme_index = app.theme_index();
        save_config(&disk, None);
    }

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
    // Blink tick: at idle the loop otherwise only redraws on input, so the
    // cursor would sit static. A slow interval wakes the loop to flip the blink.
    let mut blink = tokio::time::interval(Duration::from_millis(530));
    blink.set_missed_tick_behavior(MissedTickBehavior::Skip);
    'ui: loop {
        refresh_background_jobs(app, &ctx.agent_loop).await;
        let footer = footer_snapshot(&ctx.core_handle);
        terminal.draw(|f| app.draw(f, &footer))?;
        let ev = tokio::select! {
            ev = ev_rx.recv() => match ev {
                Some(e) => e,
                None => break,
            },
            _ = blink.tick() => {
                app.toggle_cursor();
                continue;
            }
        };
        match app.on_idle_event(ev) {
            Action::Quit => break,
            Action::Continue => {}
            Action::Submit(turn) => {
                let mut queued = Some(turn);
                while let Some(mut turn) = queued.take() {
                    if turn.media.is_empty() {
                        if let Some(rest) = as_slash_command(&turn.text) {
                            let text = turn.text.clone();
                            if slash_command(terminal, app, ctx, &text, rest, ev_rx, paused).await?
                            {
                                break 'ui; // /quit
                            }
                            save_current_snapshot(app, ctx).await;
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
                        let core = session.core.swappable();
                        if !core.model_capabilities.vision {
                            let suffix = if turn.media.len() == 1 { "" } else { "s" };
                            app.push_note(format!(
                                "{} image{} not sent — {} does not advertise image input \
                                 (set modelCapabilities.<name>.vision = true to override)",
                                turn.media.len(),
                                suffix,
                                core.model
                            ));
                            turn.media.clear();
                        }
                    }
                    queued = run_turn(terminal, app, &session, &turn, "cli", ev_rx)
                        .await?
                        .queued_turn;
                    save_current_snapshot(app, ctx).await;
                }
            }
            Action::Record => {
                #[cfg(feature = "voice")]
                {
                    voice_cycle(terminal, app, ctx, ev_rx, paused).await?;
                    save_current_snapshot(app, ctx).await;
                }
            }
            Action::PickModel(entry) => {
                apply_model_selection(terminal, app, ctx, entry).await?;
                save_current_snapshot(app, ctx).await;
            }
            Action::SessionSearch(query) => {
                let rows = load_session_rows(ctx, &query).await;
                app.set_session_rows(rows);
            }
            Action::PreviewSession(pick) => {
                let preview = preview_session(ctx, &pick).await;
                app.set_session_preview(&pick.session_id, preview);
            }
            Action::ResumeSession(pick) => {
                resume_session(app, ctx, pick).await;
                save_current_snapshot(app, ctx).await;
            }
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

/// Classify input that starts with `/`: a genuine slash command (`/model`,
/// `/help`) versus an absolute file path (`/var/folders/…png`) that drag-and-drop
/// pasted. Returns the command body (text after the `/`) only for real commands,
/// so a path is never misrouted to the dispatcher as "unknown command".
fn as_slash_command(text: &str) -> Option<&str> {
    let rest = text.strip_prefix('/')?;
    let first = rest.split_whitespace().next().unwrap_or("");
    // A command name is a short word with no path separators; a file path has
    // more slashes and a dotted extension.
    let looks_like_command = !first.is_empty()
        && first
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '?'));
    looks_like_command.then_some(rest)
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
    app.record_command(full);
    match rest.split_whitespace().next().unwrap_or("") {
        "quit" | "exit" | "q" => return Ok(true),
        "help" | "?" => app.set_help(true),
        "clear" => {
            ctx.clear_session_state().await;
            app.clear_transcript();
        }
        "jobs" => app.toggle_jobs(),
        "sessions" | "ss" => {
            let arg = sessions_native_arg(rest);
            match arg {
                SessionsRoute::Native(query) => {
                    let rows = load_session_rows(ctx, query).await;
                    app.open_session_picker(rows, query.to_string());
                }
                SessionsRoute::Classic => {
                    run_classic_command(terminal, app, ctx, full, ev_rx, paused).await?;
                }
            }
        }
        "mode" => {
            let arg = rest.strip_prefix("mode").map(str::trim).unwrap_or("");
            app.apply_mode_command(arg);
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

enum SessionsRoute<'a> {
    Native(&'a str),
    Classic,
}

fn sessions_native_arg(rest: &str) -> SessionsRoute<'_> {
    let trimmed = rest.trim();
    let command = trimmed.split_whitespace().next().unwrap_or("");
    if !matches!(command, "sessions" | "ss") {
        return SessionsRoute::Native("");
    }
    let arg = trimmed[command.len()..].trim();
    let first = arg.split_whitespace().next().unwrap_or("");
    match first {
        "export" | "purge" | "archive" => SessionsRoute::Classic,
        "list" => SessionsRoute::Native(""),
        _ => SessionsRoute::Native(arg),
    }
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

async fn load_current_session_state(app: &mut App, ctx: &mut ReplContext) {
    let core = ctx.core_handle.swappable();
    let meta = core.sessions.get_or_resume(&ctx.session_id).await;
    let history = core.sessions.get_history(&meta.id, 80, 20).await;
    app.load_transcript_from_history(&history);
    if let Some(snapshot) = core
        .sessions
        .load_snapshot(&ctx.session_id)
        .await
        .filter(|snapshot| snapshot.session_id == meta.id)
    {
        let snapshot_model = snapshot.model.clone();
        app.apply_snapshot(&snapshot);
        restore_snapshot_model(app, ctx, &snapshot_model).await;
    }
}

async fn save_current_snapshot(app: &App, ctx: &ReplContext) {
    let core = ctx.core_handle.swappable();
    let meta = core.sessions.get_or_resume(&ctx.session_id).await;
    let cwd = std::env::current_dir()
        .ok()
        .map(|p| p.display().to_string())
        .unwrap_or_default();
    let snapshot = app.snapshot(&ctx.session_id, &meta.id, cwd, core.model.clone());
    core.sessions.save_snapshot(&snapshot).await;
}

async fn restore_snapshot_model(app: &mut App, ctx: &mut ReplContext, model: &str) {
    let model = model.trim();
    if model.is_empty() || ctx.core_handle.swappable().model == model {
        return;
    }
    if !ctx.model_picker_available() {
        app.push_note(format!(
            "snapshot model {model} not restored: picker unavailable"
        ));
        return;
    }

    let entries = ctx.collect_all_models().await;
    let Some(entry) = snapshot_model_entry(&entries, model) else {
        app.push_note(format!("snapshot model {model} not found"));
        return;
    };
    let id = entry.id.clone();
    match ctx.apply_model_entry(entry).await {
        Ok(report) => {
            for note in report.notes {
                app.push_note(note);
            }
            app.push_note(format!("snapshot model restored to {id}"));
        }
        Err(e) => app.push_note(format!("snapshot model restore failed for {id}: {e}")),
    }
}

fn snapshot_model_entry(entries: &[ModelEntry], model: &str) -> Option<ModelEntry> {
    let query = model.trim();
    if query.is_empty() {
        return None;
    }
    if let Some(entry) = entries.iter().find(|entry| entry.id == query) {
        return Some(entry.clone());
    }
    let refs: Vec<&ModelEntry> = entries.iter().collect();
    unique_direct_model_match(&refs, query).cloned()
}

async fn load_session_rows(ctx: &ReplContext, query: &str) -> Vec<SessionRow> {
    let core = ctx.core_handle.swappable();
    let query = query.trim();
    if query.is_empty() {
        return core
            .sessions
            .list_sessions(None, 50)
            .await
            .into_iter()
            .map(|meta| SessionRow {
                session_id: meta.id,
                session_key: meta.session_key,
                updated_at: format_session_time(meta.updated_at),
                message_count: meta.message_count,
                snippet: String::new(),
                preview: None,
            })
            .collect();
    }

    let mut rows = Vec::new();
    // First, search by message content using prefix matching.
    for result in core.sessions.search_messages_prefix(query, 80).await {
        if rows
            .iter()
            .any(|row: &SessionRow| row.session_id == result.session_id)
        {
            continue;
        }
        rows.push(SessionRow {
            session_id: result.session_id,
            session_key: result.session_key,
            updated_at: result.timestamp,
            message_count: 0,
            snippet: clean_search_snippet(&result.snippet, &result.content),
            preview: None,
        });
        if rows.len() >= 50 {
            return rows;
        }
    }

    // Also include sessions whose session_key contains the query as a case-insensitive substring.
    let query_lower = query.to_lowercase();
    for meta in core.sessions.list_sessions(None, 50).await {
        if !meta.session_key.to_lowercase().contains(&query_lower) {
            continue;
        }
        if rows
            .iter()
            .any(|row: &SessionRow| row.session_id == meta.id)
        {
            continue;
        }
        rows.push(SessionRow {
            session_id: meta.id,
            session_key: meta.session_key,
            updated_at: format_session_time(meta.updated_at),
            message_count: meta.message_count,
            snippet: String::new(),
            preview: None,
        });
        if rows.len() >= 50 {
            return rows;
        }
    }
    rows
}

async fn preview_session(ctx: &ReplContext, pick: &SessionPick) -> String {
    let core = ctx.core_handle.swappable();
    let history = core.sessions.get_history(&pick.session_id, 24, 8).await;
    if history.is_empty() {
        return "No recent transcript rows.".to_string();
    }
    history
        .iter()
        .filter_map(format_history_line)
        .collect::<Vec<_>>()
        .join("\n")
}

async fn resume_session(app: &mut App, ctx: &mut ReplContext, pick: SessionPick) {
    save_current_snapshot(app, ctx).await;
    let core = ctx.core_handle.swappable();
    let resumed = match core.sessions.resume_session(&pick.session_id).await {
        Ok(Some(meta)) => meta,
        Ok(None) => {
            app.push_note(format!("session {} no longer exists", pick.session_id));
            return;
        }
        Err(error) => {
            app.push_note(format!("failed to resume {}: {error}", pick.session_id));
            return;
        }
    };
    ctx.session_id = resumed.session_key.clone();
    ctx.core_handle
        .counters
        .reset_session_prompt_state(&ctx.session_id);
    load_current_session_state(app, ctx).await;
    app.push_note(format!("resumed {}", resumed.id));
}

async fn refresh_background_jobs(app: &mut App, agent: &AgentLoop) {
    let jobs = agent
        .subagent_manager()
        .list_running()
        .await
        .into_iter()
        .map(|info| BackgroundJob {
            id: info.task_id,
            label: info.label,
            kind: "agent".to_string(),
            elapsed_ms: info.started_at.elapsed().as_millis() as u64,
        })
        .collect();
    app.set_background_jobs(jobs);
}

fn format_session_time(time: chrono::DateTime<chrono::Utc>) -> String {
    time.format("%Y-%m-%d %H:%M").to_string()
}

fn clean_search_snippet(snippet: &str, fallback: &str) -> String {
    let text = if snippet.trim().is_empty() {
        fallback
    } else {
        snippet
    };
    text.replace(">>>", "")
        .replace("<<<", "")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn format_history_line(msg: &serde_json::Value) -> Option<String> {
    let role = msg.get("role").and_then(serde_json::Value::as_str)?;
    let content = msg_content(msg)?;
    let content = content.split_whitespace().collect::<Vec<_>>().join(" ");
    if content.is_empty() {
        return None;
    }
    Some(format!("{role}: {}", shorten(&content, 140)))
}

fn msg_content(msg: &serde_json::Value) -> Option<String> {
    let content = msg.get("content")?;
    if let Some(text) = content.as_str() {
        return Some(text.to_string());
    }
    if let Some(parts) = content.as_array() {
        let text = parts
            .iter()
            .filter_map(|part| {
                part.as_str().map(ToOwned::to_owned).or_else(|| {
                    part.get("text")
                        .and_then(serde_json::Value::as_str)
                        .map(ToOwned::to_owned)
                })
            })
            .collect::<Vec<_>>()
            .join("\n");
        return (!text.is_empty()).then_some(text);
    }
    None
}

fn shorten(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        text.to_string()
    } else {
        format!(
            "{}...",
            text.chars()
                .take(max_chars.saturating_sub(3))
                .collect::<String>()
        )
    }
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
                "voice",
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
    channel: &str,
    ev_rx: &mut UnboundedReceiver<Event>,
) -> std::io::Result<TurnOutcome> {
    app.begin_turn(&turn.display_text());

    let (delta_tx, delta_rx) = unbounded_channel::<String>();
    let (tool_tx, tool_rx) = unbounded_channel::<ToolEvent>();
    let cancel = CancellationToken::new();

    // Spawn the agent on its own task so this render loop keeps drawing while a
    // CPU-heavy turn runs — otherwise synchronous stretches inside the turn
    // (prompt build, JSON encode, parsing) would starve the `tick` branch and
    // freeze the spinner/elapsed clock. We select on the JoinHandle instead.
    let handle = session.agent.spawn_direct_streaming(
        turn.text.clone(),
        session.session_id.to_string(),
        channel.to_string(),
        "direct".to_string(),
        session.lang.map(|s| s.to_string()),
        delta_tx,
        Some(tool_tx),
        Some(cancel.clone()),
        (!turn.media.is_empty()).then(|| turn.media.clone()),
    );

    // The shared engine owns the turn lifecycle: biased delta-first ordering,
    // deltas-before-tool-rows, post-completion drains, and cancel-discards-
    // everything (a cancelled turn finishes with an empty response).
    let mut stream = TurnStream::new(
        delta_rx,
        Some(tool_rx),
        Completion::AgentHandle(handle),
        Some(cancel),
    );

    let mut tick = tokio::time::interval(Duration::from_millis(80));
    tick.set_missed_tick_behavior(MissedTickBehavior::Skip);

    let mut queued_turn = None;
    let response = loop {
        refresh_background_jobs(app, session.agent).await;
        let footer = footer_snapshot(session.core);
        terminal.draw(|f| app.draw(f, &footer))?;
        tokio::select! {
            biased;
            Some(ev) = ev_rx.recv() => {
                match app.on_streaming_event(ev) {
                    StreamingAction::Continue => {}
                    StreamingAction::Cancel => stream.cancel(),
                    StreamingAction::CancelAndSubmit(turn) => {
                        queued_turn = Some(turn);
                        stream.cancel();
                    }
                }
            }
            event = stream.next() => match event {
                TurnEvent::Delta(d) => app.on_delta(&d),
                TurnEvent::Tool(e) => app.on_tool_event(e),
                TurnEvent::Finished(response) => break response,
            },
            _ = tick.tick() => app.tick(0.08),
        }
    };

    #[cfg(feature = "voice")]
    let reply = response.clone();
    app.finish_turn(response);
    let footer = footer_snapshot(session.core);
    terminal.draw(|f| app.draw(f, &footer))?;
    Ok(TurnOutcome {
        #[cfg(feature = "voice")]
        reply,
        queued_turn,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    #[test]
    fn tui_defaults_on_for_ttys_and_env_overrides_both_ways() {
        // Unset env: the tty probe decides.
        assert!(enabled_from(None, true));
        assert!(!enabled_from(None, false));
        // Explicit opt-out wins over a tty; opt-in wins over no tty.
        assert!(!enabled_from(Some("0"), true));
        assert!(!enabled_from(Some("false"), true));
        assert!(enabled_from(Some("1"), false));
        assert!(enabled_from(Some("TRUE"), false));
        // Unrecognized values fall back to the probe.
        assert!(enabled_from(Some("yes"), true));
        assert!(!enabled_from(Some("yes"), false));
    }

    /// The delta-flush rule `run_turn` used to enforce inline and now gets
    /// structurally from [`TurnStream`]; kept here so the render-order pin
    /// below stays a pure `App` test.
    fn drain_pending_deltas(app: &mut App, delta_rx: &mut UnboundedReceiver<String>) {
        while let Ok(d) = delta_rx.try_recv() {
            app.on_delta(&d);
        }
    }

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
    fn first_frame_of_fresh_app_is_clean() {
        // First paint must be: header, welcome hint, input box — never the
        // jobs overlay (regression: stale alt-screen content / overlay text
        // visible at startup without ever toggling /jobs).
        let mut app = App::new();
        let footer = Footer {
            cwd: "~".into(),
            model: "local:test".into(),
            ctx_used: 0,
            ctx_max: 0,
        };
        let mut term = Terminal::new(TestBackend::new(100, 30)).unwrap();
        term.draw(|f| app.draw(f, &footer)).unwrap();
        let text = buffer_text(term.backend().buffer());
        assert!(
            !text.contains("no running jobs"),
            "jobs overlay leaked into first frame:\n{text}"
        );
        assert!(text.contains("TRENTADUE"), "brand header missing:\n{text}");
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

    #[test]
    fn sessions_router_does_not_restore_removed_index_subcommand() {
        assert!(matches!(
            sessions_native_arg("sessions export cli:test"),
            SessionsRoute::Classic
        ));
        assert!(matches!(
            sessions_native_arg("sessions purge 7d"),
            SessionsRoute::Classic
        ));
        assert!(matches!(
            sessions_native_arg("sessions archive"),
            SessionsRoute::Classic
        ));
        assert!(matches!(
            sessions_native_arg("sessions index"),
            SessionsRoute::Native("index")
        ));
    }

    #[test]
    fn snapshot_model_entry_prefers_exact_model_id() {
        let entries = vec![
            ModelEntry::test_local("alpha-plus"),
            ModelEntry::test_local("alpha"),
        ];

        let selected = snapshot_model_entry(&entries, "alpha").unwrap();
        assert_eq!(selected.id, "alpha");
        assert!(snapshot_model_entry(&entries, "missing").is_none());
    }

    #[test]
    fn slash_command_classifier_rejects_paths() {
        // Genuine commands resolve to their body.
        assert_eq!(as_slash_command("/model gpt-4"), Some("model gpt-4"));
        assert_eq!(as_slash_command("/help"), Some("help"));
        assert_eq!(as_slash_command("/quit"), Some("quit"));
        assert_eq!(as_slash_command("/?"), Some("?"));
        // Absolute paths (drag-and-drop) are NOT commands.
        assert_eq!(
            as_slash_command("/var/folders/x/Screenshot.png see this"),
            None
        );
        assert_eq!(as_slash_command("/Users/me/pic.png"), None);
        // Plain text isn't a command either.
        assert_eq!(as_slash_command("hello there"), None);
    }
}
