//! Transcript model, input box and drawing for the full-screen ratatui UI.
//!
//! The transcript is a flat, chronological list of [`Cell`]s. Streaming text
//! appends to the trailing `Reply` cell (or opens a new one after a tool runs),
//! so the user sees text / tool / text interleaved exactly as it happened —
//! matching the classic incremental renderer, but on a re-renderable model that
//! reflows on resize and can be scrolled.
//!
//! Layout follows the "calm" design: a small header status line, the transcript,
//! a stable input box, and a quiet stateful footer (cwd / model / mode / ctx).
//! Per-turn metadata (`▸ elapsed · ttft · tokens`) is captured live and pinned
//! beneath each turn. The async event loop lives in `mod.rs`; this file is pure
//! state + rendering with no terminal I/O, which keeps it unit-testable.

use std::time::Instant;

use ratatui::crossterm::event::{Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::layout::{Constraint, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, BorderType, Borders, Padding, Paragraph};
use ratatui::Frame;
use ratatui_textarea::{TextArea, WrapMode};
use unicode_width::UnicodeWidthChar;

use super::render;
use crate::agent::audit::ToolEvent;
use crate::repl::{parse_control_marker, ControlMarker};

const BRAND: &str = "\u{259e}"; // ▞  the nanobot wordmark glyph
const DOT: &str = "\u{2022}"; //   •  status / user marker
const RUN: &str = "\u{25b6}"; //   ▶
const OK: &str = "\u{2713}"; //   ✓
const ERR: &str = "\u{2717}"; //  ✗
const TURN: &str = "\u{21b3}"; // ↳
const META: &str = "\u{25b8}"; // ▸
const RULE: &str = "\u{2500}"; // ─

/// Quiet, stateful footer data, snapshotted from the agent core each frame.
pub(crate) struct Footer {
    pub cwd: String,
    pub model: String,
    pub ctx_used: usize,
    pub ctx_max: usize,
}

/// How a submitted idle event should steer the outer loop.
pub(crate) enum Action {
    /// Keep looping (redraw on the next iteration).
    Continue,
    /// User asked to leave the app.
    Quit,
    /// User submitted a message to send to the agent.
    Submit(String),
}

/// Status of a single tool invocation.
enum ToolState {
    Running,
    Ok,
    Err,
}

/// One chronological entry in the transcript.
enum Cell {
    /// A user turn (verbatim, may be multi-line).
    User(String),
    /// Assistant text. Streamed deltas append here until a tool interrupts.
    Reply(String),
    /// A tool call, updated in place when its `CallEnd` arrives.
    Tool {
        id: String,
        name: String,
        args: String,
        state: ToolState,
        summary: Option<String>,
        ms: u64,
    },
    /// Per-turn metadata pinned beneath the turn's output.
    Meta {
        elapsed_s: f32,
        ttft_s: Option<f32>,
        tokens: u64,
    },
    /// A local system note (e.g. an unsupported command).
    Note(String),
}

/// All UI state for the ratatui app.
pub(crate) struct App {
    transcript: Vec<Cell>,
    input: TextArea<'static>,
    /// True while the agent is producing a turn.
    streaming: bool,
    /// True between turn start and the first text/tool output (prefill phase).
    awaiting_first: bool,
    /// Whether any text delta arrived this turn (drives the non-streaming fallback).
    got_text: bool,
    /// Whether this turn produced any visible output (gates the metadata line).
    turn_produced: bool,
    /// Wall-clock start of the current turn (for elapsed + ttft).
    turn_start: Option<Instant>,
    /// Time to first output token/tool, seconds.
    ttft: Option<f32>,
    /// Completion tokens accumulated this turn (summed across tool-loop responses).
    turn_tokens: u64,
    /// Server-reported prefill progress `(processed, total)`, when available.
    prefill: Option<(u64, u64)>,
    /// Rows scrolled up from the bottom; `0` sticks to the latest output.
    scroll_from_bottom: usize,
    /// Max scrollable rows from the last draw, used to clamp `scroll_from_bottom`
    /// so paging back down always reaches the bottom.
    max_scroll: usize,
}

impl App {
    pub(crate) fn new() -> Self {
        Self {
            transcript: Vec::new(),
            input: configure_input(),
            streaming: false,
            awaiting_first: false,
            got_text: false,
            turn_produced: false,
            turn_start: None,
            ttft: None,
            turn_tokens: 0,
            prefill: None,
            scroll_from_bottom: 0,
            max_scroll: 0,
        }
    }

    // --- turn lifecycle -----------------------------------------------------

    pub(crate) fn begin_turn(&mut self, input: &str) {
        self.transcript.push(Cell::User(input.to_string()));
        self.streaming = true;
        self.awaiting_first = true;
        self.got_text = false;
        self.turn_produced = false;
        self.turn_start = Some(Instant::now());
        self.ttft = None;
        self.turn_tokens = 0;
        self.prefill = None;
        self.scroll_from_bottom = 0; // jump to bottom for the new turn
    }

    /// Consume one text delta. Control markers (prefill/tokens/finish) are
    /// interpreted, never rendered; everything else is assistant text.
    pub(crate) fn on_delta(&mut self, d: &str) {
        match parse_control_marker(d) {
            Some(ControlMarker::PrefillProgress { processed, total }) => {
                if self.awaiting_first {
                    self.prefill = Some((processed, total));
                }
            }
            Some(ControlMarker::Tokens(n)) => self.turn_tokens += n,
            Some(ControlMarker::FinishReason(_)) => {}
            None => self.push_text(d),
        }
    }

    fn push_text(&mut self, d: &str) {
        self.mark_first_output();
        self.got_text = true;
        if let Some(Cell::Reply(t)) = self.transcript.last_mut() {
            t.push_str(d);
        } else {
            self.transcript.push(Cell::Reply(d.to_string()));
        }
    }

    /// Record the first observable output of a turn (ends prefill, stamps ttft).
    fn mark_first_output(&mut self) {
        self.turn_produced = true;
        self.awaiting_first = false;
        self.prefill = None;
        if self.ttft.is_none() {
            self.ttft = Some(self.elapsed_s());
        }
    }

    pub(crate) fn on_tool_event(&mut self, ev: ToolEvent) {
        match ev {
            ToolEvent::CallStart {
                tool_name,
                tool_call_id,
                arguments_preview,
            } => {
                self.mark_first_output();
                self.transcript.push(Cell::Tool {
                    id: tool_call_id,
                    name: tool_name,
                    args: arguments_preview,
                    state: ToolState::Running,
                    summary: None,
                    ms: 0,
                });
            }
            ToolEvent::CallEnd {
                tool_name,
                tool_call_id,
                result_data,
                ok,
                duration_ms,
            } => self.complete_tool(&tool_call_id, &tool_name, &result_data, ok, duration_ms),
            ToolEvent::Progress { .. } => {}
        }
    }

    fn complete_tool(&mut self, id: &str, name: &str, data: &str, ok: bool, ms: u64) {
        let summary = summarize_output(data, ok);
        let state = if ok { ToolState::Ok } else { ToolState::Err };
        let idx = self
            .transcript
            .iter()
            .rposition(|c| matches!(c, Cell::Tool { id: cid, .. } if cid.as_str() == id));
        match idx {
            Some(i) => {
                if let Cell::Tool {
                    state: s,
                    summary: sum,
                    ms: m,
                    ..
                } = &mut self.transcript[i]
                {
                    *s = state;
                    *sum = summary;
                    *m = ms;
                }
            }
            None => self.transcript.push(Cell::Tool {
                id: id.to_string(),
                name: name.to_string(),
                args: String::new(),
                state,
                summary,
                ms,
            }),
        }
    }

    /// Finalize the turn. `resp` is the agent's full return string; it backfills
    /// a `Reply` when the provider didn't stream any deltas, and pins a metadata
    /// line beneath the turn's output.
    pub(crate) fn finish_turn(&mut self, resp: String) {
        self.streaming = false;
        self.awaiting_first = false;
        self.prefill = None;
        if !self.got_text && !resp.trim().is_empty() {
            // Non-streaming providers return the whole string at once; stamp ttft
            // so the metadata line still shows all three fields.
            if self.ttft.is_none() {
                self.ttft = Some(self.elapsed_s());
            }
            self.transcript.push(Cell::Reply(resp));
            self.turn_produced = true;
        }
        if let Some(Cell::Reply(t)) = self.transcript.last() {
            if t.trim().is_empty() {
                self.transcript.pop();
            }
        }
        if self.turn_produced {
            self.transcript.push(Cell::Meta {
                elapsed_s: self.elapsed_s(),
                ttft_s: self.ttft,
                tokens: self.turn_tokens,
            });
        }
    }

    pub(crate) fn push_note(&mut self, note: String) {
        self.transcript.push(Cell::Note(note));
    }

    /// Tick exists to drive redraws for the streaming animation; elapsed time is
    /// read from the wall clock, so this is intentionally a no-op.
    pub(crate) fn tick(&mut self, _dt: f32) {}

    fn elapsed_s(&self) -> f32 {
        self.turn_start
            .map(|t| t.elapsed().as_secs_f32())
            .unwrap_or(0.0)
    }

    // --- input handling -----------------------------------------------------

    /// Handle an event while idle (no turn in flight).
    pub(crate) fn on_idle_event(&mut self, ev: Event) -> Action {
        match ev {
            Event::Key(k) if is_press(&k) => self.on_idle_key(k),
            Event::Paste(s) => {
                self.input.insert_str(&s);
                Action::Continue
            }
            _ => Action::Continue,
        }
    }

    fn on_idle_key(&mut self, k: KeyEvent) -> Action {
        let ctrl = k.modifiers.contains(KeyModifiers::CONTROL);
        match k.code {
            KeyCode::Char('c' | 'd') if ctrl => Action::Quit,
            KeyCode::Enter if k.modifiers.is_empty() => self.submit(),
            KeyCode::PageUp => {
                self.scroll_up(10);
                Action::Continue
            }
            KeyCode::PageDown => {
                self.scroll_down(10);
                Action::Continue
            }
            _ => {
                self.input.input(k);
                Action::Continue
            }
        }
    }

    fn submit(&mut self) -> Action {
        let text = self.input.lines().join("\n");
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return Action::Continue;
        }
        let owned = trimmed.to_string();
        self.input = configure_input();
        Action::Submit(owned)
    }

    /// Handle an event while a turn is streaming. Returns `true` to cancel.
    pub(crate) fn on_streaming_event(&mut self, ev: Event) -> bool {
        let Event::Key(k) = ev else { return false };
        if !is_press(&k) {
            return false;
        }
        let ctrl = k.modifiers.contains(KeyModifiers::CONTROL);
        match k.code {
            KeyCode::Esc => true,
            KeyCode::Char('c') if ctrl => true,
            KeyCode::PageUp => {
                self.scroll_up(10);
                false
            }
            KeyCode::PageDown => {
                self.scroll_down(10);
                false
            }
            _ => false,
        }
    }

    fn scroll_up(&mut self, n: usize) {
        // Clamp to the last-known scrollable height so we never overshoot the
        // top — otherwise paging back down has to "unwind" the overshoot first.
        self.scroll_from_bottom = (self.scroll_from_bottom + n).min(self.max_scroll);
    }

    fn scroll_down(&mut self, n: usize) {
        self.scroll_from_bottom = self.scroll_from_bottom.saturating_sub(n);
    }

    // --- drawing ------------------------------------------------------------

    pub(crate) fn draw(&mut self, f: &mut Frame, footer: &Footer) {
        let area = f.area();
        let input_h = self.input_height();
        let chunks = Layout::vertical([
            Constraint::Length(2), // header: status + rule
            Constraint::Min(1),    // transcript
            Constraint::Length(input_h),
            Constraint::Length(1), // footer
        ])
        .split(area);

        let header = chunks[0];
        f.render_widget(
            Paragraph::new(self.header_lines(header.width as usize)),
            header,
        );

        let transcript = chunks[1];
        let width = transcript.width.max(1) as usize;
        let height = transcript.height as usize;
        let rows = self.transcript_rows(width);
        let max_off = rows.len().saturating_sub(height);
        self.max_scroll = max_off;
        self.scroll_from_bottom = self.scroll_from_bottom.min(max_off);
        let off = max_off - self.scroll_from_bottom;
        let visible: Vec<Line> = rows.into_iter().skip(off).take(height).collect();
        f.render_widget(Paragraph::new(Text::from(visible)), transcript);

        f.render_widget(&self.input, chunks[2]);
        f.render_widget(Paragraph::new(footer_line(footer)), chunks[3]);
    }

    fn input_height(&self) -> u16 {
        let n = self.input.lines().len().max(1) as u16;
        n.min(8) + 2 // +2 for the rounded border
    }

    fn header_lines(&self, width: usize) -> Text<'static> {
        // The brand mark itself is the activity indicator: solid when idle, a
        // single cell sweeping while streaming.
        let active = self
            .streaming
            .then(|| ((self.elapsed_s() * 6.0) as usize) % 3);
        let mut status = vec![Span::raw("  ")];
        status.extend(brand_mark(active));
        status.push(Span::styled(" TRENTADUE", style(Color::White, true)));
        status.push(Span::styled(" \u{b7} 32", dim_color(Color::Cyan)));
        status.push(Span::styled("  \u{b7}  ", dim()));
        if self.streaming {
            let tail = match self.prefill {
                Some((p, t)) if t > 0 => {
                    format!("{}% \u{b7} {:.1}s", p * 100 / t, self.elapsed_s())
                }
                _ => format!("{:.1}s", self.elapsed_s()),
            };
            status.push(Span::styled(tail, dim()));
        } else {
            status.push(Span::styled("ready", style(Color::Green, false)));
        }
        let rule = Span::styled(RULE.repeat(width), dim());
        Text::from(vec![Line::from(status), Line::from(rule)])
    }

    fn transcript_rows(&self, width: usize) -> Vec<Line<'static>> {
        if self.transcript.is_empty() {
            return vec![
                Line::default(),
                Line::from(Span::styled(
                    "  Ask nanobot-rs anything.  Enter to send · Alt+Enter newline · Ctrl+C quit",
                    dim(),
                )),
            ];
        }
        let mut rows: Vec<Line> = Vec::new();
        for cell in &self.transcript {
            // Blank line before each user turn separates conversation turns.
            if matches!(cell, Cell::User(_)) && !rows.is_empty() {
                rows.push(Line::default());
            }
            for segs in cell_lines(cell) {
                wrap_segments(&segs, width, &mut rows);
            }
            // Breathing room between the user turn and the assistant reply.
            if matches!(cell, Cell::User(_)) {
                rows.push(Line::default());
            }
        }
        rows
    }
}

// ---------------------------------------------------------------------------
// Free functions (rendering + wrapping)
// ---------------------------------------------------------------------------

fn configure_input() -> TextArea<'static> {
    let mut ta = TextArea::default();
    ta.set_wrap_mode(WrapMode::WordOrGlyph);
    ta.set_block(
        Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(dim())
            .padding(Padding::horizontal(1)),
    );
    ta.set_placeholder_text("Ask nanobot-rs anything...");
    ta.set_cursor_line_style(Style::default());
    ta
}

fn is_press(k: &KeyEvent) -> bool {
    k.kind != KeyEventKind::Release
}

/// Truncate to `max` display chars with an ellipsis, so a long tool-arg preview
/// can't dominate the transcript row before wrapping.
fn clip(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let head: String = s.chars().take(max.saturating_sub(1)).collect();
    format!("{head}\u{2026}")
}

fn style(color: Color, bold: bool) -> Style {
    let s = Style::default().fg(color);
    if bold {
        s.add_modifier(Modifier::BOLD)
    } else {
        s
    }
}

fn dim() -> Style {
    Style::default().add_modifier(Modifier::DIM)
}

fn dim_color(color: Color) -> Style {
    Style::default().fg(color).add_modifier(Modifier::DIM)
}

/// Quiet footer: cwd · model · mode · ctx usage.
fn footer_line(footer: &Footer) -> Line<'static> {
    let mut spans = vec![
        Span::styled("  cwd ", dim()),
        Span::styled(footer.cwd.clone(), style(Color::Green, false)),
        Span::styled("   model ", dim()),
        Span::styled(footer.model.clone(), style(Color::Green, false)),
        Span::styled("   mode ", dim()),
        Span::styled(
            " calm ",
            Style::default().fg(Color::Green).bg(Color::DarkGray),
        ),
        Span::styled("   ctx ", dim()),
    ];
    if footer.ctx_max > 0 {
        let pct = footer.ctx_used * 100 / footer.ctx_max;
        let ctx_color = match pct {
            0..=49 => Color::Green,
            50..=79 => Color::Yellow,
            _ => Color::Red,
        };
        spans.push(Span::styled(
            crate::tui::format_tokens(footer.ctx_used),
            style(ctx_color, false),
        ));
        spans.push(Span::styled("/", dim()));
        spans.push(Span::styled(crate::tui::format_tokens(footer.ctx_max), dim()));
    } else {
        spans.push(Span::styled("\u{2014}", dim())); // —
    }
    Line::from(spans)
}

/// The TRENTADUE brand mark `▞▞▞`. With `active` set, that cell is lit and the
/// others dimmed (a sweep during streaming); otherwise all three are lit.
fn brand_mark(active: Option<usize>) -> Vec<Span<'static>> {
    (0..3)
        .map(|i| {
            let lit = active.map_or(true, |a| a == i);
            let st = if lit {
                style(Color::Cyan, true)
            } else {
                dim_color(Color::Cyan)
            };
            Span::styled(BRAND, st)
        })
        .collect()
}

/// Render one cell into logical lines, each a list of `(text, style)` segments.
fn cell_lines(cell: &Cell) -> Vec<Vec<(String, Style)>> {
    match cell {
        Cell::User(text) => text
            .split('\n')
            .enumerate()
            .map(|(i, l)| {
                let head = if i == 0 {
                    (format!("  {DOT} "), style(Color::Green, true))
                } else {
                    ("    ".to_string(), Style::default())
                };
                vec![head, (l.to_string(), style(Color::White, true))]
            })
            .collect(),
        Cell::Reply(text) => render::markdown(text)
            .into_iter()
            .enumerate()
            .map(|(i, mut segs)| {
                let head = if i == 0 {
                    (format!("  {BRAND}{BRAND}{BRAND} "), style(Color::Cyan, true))
                } else {
                    ("      ".to_string(), Style::default())
                };
                let mut line = vec![head];
                line.append(&mut segs);
                line
            })
            .collect(),
        Cell::Tool {
            name,
            args,
            state,
            summary,
            ms,
            ..
        } => {
            let mut head = vec![
                (format!("  {RUN} "), style(Color::Cyan, false)),
                (format!("{name}({})", clip(args, 48)), style(Color::Cyan, false)),
            ];
            match state {
                ToolState::Running => head.push(("  …".to_string(), dim())),
                ToolState::Ok => head.push((format!("  {OK} {ms}ms"), style(Color::Green, false))),
                ToolState::Err => head.push((format!("  {ERR} {ms}ms"), style(Color::Red, false))),
            }
            let mut out = vec![head];
            if let Some(s) = summary {
                out.push(vec![(format!("    {TURN} "), dim()), (s.clone(), dim())]);
            }
            out
        }
        Cell::Meta {
            elapsed_s,
            ttft_s,
            tokens,
        } => {
            let mut s = format!("  {META} {:.1}s", elapsed_s);
            if let Some(t) = ttft_s {
                s.push_str(&format!("  ttft {:.2}s", t));
            }
            if *tokens > 0 {
                s.push_str(&format!("  {tokens} tok"));
            }
            vec![vec![(s, dim())]]
        }
        Cell::Note(text) => vec![vec![
            ("  ".to_string(), Style::default()),
            (text.clone(), style(Color::Yellow, false)),
        ]],
    }
}

/// Soft-wrap one logical line (given as styled segments) to `width` display
/// columns, pushing one or more rows into `out`. Always pushes at least one row
/// so a blank logical line stays visible.
fn wrap_segments(segs: &[(String, Style)], width: usize, out: &mut Vec<Line<'static>>) {
    let width = width.max(1);
    let mut row: Vec<(char, Style)> = Vec::new();
    let mut w = 0usize;
    for (text, st) in segs {
        for c in text.chars() {
            let cw = UnicodeWidthChar::width(c).unwrap_or(0);
            if w + cw > width && !row.is_empty() {
                out.push(row_to_line(&row));
                row.clear();
                w = 0;
            }
            row.push((c, *st));
            w += cw;
        }
    }
    out.push(row_to_line(&row));
}

/// Collapse `(char, style)` cells into a `Line`, coalescing runs of equal style.
fn row_to_line(row: &[(char, Style)]) -> Line<'static> {
    let mut spans: Vec<Span<'static>> = Vec::new();
    let mut buf = String::new();
    let mut cur: Option<Style> = None;
    for &(c, st) in row {
        match cur {
            Some(s) if s == st => buf.push(c),
            _ => {
                if let Some(s) = cur {
                    spans.push(Span::styled(std::mem::take(&mut buf), s));
                }
                buf.push(c);
                cur = Some(st);
            }
        }
    }
    if let Some(s) = cur {
        spans.push(Span::styled(buf, s));
    }
    Line::from(spans)
}

/// One-line summary of a tool result, collapsed like the classic renderer.
fn summarize_output(data: &str, ok: bool) -> Option<String> {
    let first = data.lines().find(|l| !l.trim().is_empty())?.trim();
    if !ok {
        return Some(first.chars().take(80).collect());
    }
    let n = data.lines().filter(|l| !l.trim().is_empty()).count();
    let preview: String = first.chars().take(72).collect();
    let more = if preview.chars().count() < first.chars().count() {
        "…"
    } else {
        ""
    };
    Some(format!(
        "{preview}{more}  · {n} line{}",
        if n == 1 { "" } else { "s" }
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_footer() -> Footer {
        Footer {
            cwd: "~/Dev/nanobot-rs".into(),
            model: "bonsai-8b-mlx".into(),
            ctx_used: 2200,
            ctx_max: 65500,
        }
    }

    fn kinds(app: &App) -> Vec<&'static str> {
        app.transcript
            .iter()
            .map(|c| match c {
                Cell::User(_) => "user",
                Cell::Reply(_) => "reply",
                Cell::Tool { .. } => "tool",
                Cell::Meta { .. } => "meta",
                Cell::Note(_) => "note",
            })
            .collect()
    }

    fn reply_text(app: &App) -> String {
        app.transcript
            .iter()
            .filter_map(|c| match c {
                Cell::Reply(t) => Some(t.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("|")
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
    fn wrap_splits_on_display_width_and_preserves_text() {
        let mut out = Vec::new();
        wrap_segments(&[("abcdef".to_string(), Style::default())], 3, &mut out);
        let rows: Vec<String> = out
            .iter()
            .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect())
            .collect();
        assert_eq!(rows, vec!["abc".to_string(), "def".to_string()]);
    }

    #[test]
    fn wrap_emits_one_row_for_short_line() {
        let mut out = Vec::new();
        wrap_segments(&[("hi".to_string(), Style::default())], 80, &mut out);
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn wrap_guards_zero_width() {
        let mut out = Vec::new();
        wrap_segments(&[("ab".to_string(), Style::default())], 0, &mut out);
        assert!(out.len() >= 2);
    }

    #[test]
    fn streaming_deltas_interleave_with_tools_chronologically() {
        let mut app = App::new();
        app.begin_turn("hello");
        app.on_delta("before ");
        app.on_delta("tool");
        app.on_tool_event(ToolEvent::CallStart {
            tool_name: "read_file".into(),
            tool_call_id: "c1".into(),
            arguments_preview: "path=x".into(),
        });
        app.on_delta("after tool");
        assert_eq!(kinds(&app), vec!["user", "reply", "tool", "reply"]);
        assert_eq!(reply_text(&app), "before tool|after tool");
    }

    #[test]
    fn tool_callend_updates_matching_cell_in_place() {
        let mut app = App::new();
        app.begin_turn("go");
        app.on_tool_event(ToolEvent::CallStart {
            tool_name: "exec".into(),
            tool_call_id: "c1".into(),
            arguments_preview: "ls".into(),
        });
        app.on_tool_event(ToolEvent::CallEnd {
            tool_name: "exec".into(),
            tool_call_id: "c1".into(),
            result_data: "a\nb\nc".into(),
            ok: true,
            duration_ms: 12,
        });
        let tools: Vec<_> = app
            .transcript
            .iter()
            .filter(|c| matches!(c, Cell::Tool { .. }))
            .collect();
        assert_eq!(tools.len(), 1);
        match tools[0] {
            Cell::Tool {
                state: ToolState::Ok,
                summary: Some(s),
                ms,
                ..
            } => {
                assert_eq!(*ms, 12);
                assert!(s.contains("3 lines"), "summary was: {s}");
            }
            _ => panic!("tool not completed"),
        }
    }

    #[test]
    fn non_streaming_response_backfills_reply_and_meta() {
        let mut app = App::new();
        app.begin_turn("q");
        app.finish_turn("the answer".into());
        assert_eq!(kinds(&app), vec!["user", "reply", "meta"]);
        assert_eq!(reply_text(&app), "the answer");
    }

    #[test]
    fn empty_turn_drops_reply_and_meta() {
        let mut app = App::new();
        app.begin_turn("q");
        app.finish_turn(String::new());
        assert_eq!(kinds(&app), vec!["user"]);
    }

    #[test]
    fn metadata_captures_tokens_and_marks_produced() {
        let mut app = App::new();
        app.begin_turn("q");
        app.on_delta("\u{0}tokens:40");
        app.on_delta("hi");
        app.on_delta("\u{0}tokens:16");
        app.finish_turn(String::new());
        match app.transcript.last() {
            Some(Cell::Meta { tokens, ttft_s, .. }) => {
                assert_eq!(*tokens, 56);
                assert!(ttft_s.is_some(), "ttft should be stamped at first text");
            }
            other => panic!("expected meta cell, got {:?}", other.map(|_| ())),
        }
    }

    #[test]
    fn control_markers_never_become_visible_text() {
        let mut app = App::new();
        app.begin_turn("q");
        app.on_delta("\u{0}prefill:5/10");
        assert_eq!(app.prefill, Some((5, 10)));
        assert_eq!(kinds(&app), vec!["user"]);
        app.on_delta("real");
        assert_eq!(reply_text(&app), "real");
        assert_eq!(app.prefill, None);
    }

    #[test]
    fn draw_renders_header_turn_meta_and_footer() {
        use ratatui::backend::TestBackend;
        use ratatui::Terminal;

        let mut app = App::new();
        app.begin_turn("hello world");
        app.on_delta("hi there");
        app.on_delta("\u{0}tokens:78");
        app.finish_turn(String::new());

        // 100 cols so the quiet footer isn't truncated (it correctly truncates
        // on narrow terminals, which a 60-col backend would hide).
        let mut term = Terminal::new(TestBackend::new(100, 16)).unwrap();
        term.draw(|f| app.draw(f, &test_footer())).unwrap();
        let text = buffer_text(term.backend().buffer());

        assert!(text.contains("TRENTADUE"), "brand header missing:\n{text}");
        assert!(text.contains("hello world"), "user turn missing:\n{text}");
        assert!(text.contains("hi there"), "reply missing:\n{text}");
        assert!(text.contains("78 tok"), "metadata tokens missing:\n{text}");
        assert!(text.contains("bonsai-8b-mlx"), "footer model missing:\n{text}");
        assert!(text.contains("ctx"), "footer ctx missing:\n{text}");
    }

    #[test]
    fn empty_transcript_shows_welcome_hint() {
        use ratatui::backend::TestBackend;
        use ratatui::Terminal;

        let mut app = App::new();
        let mut term = Terminal::new(TestBackend::new(80, 12)).unwrap();
        term.draw(|f| app.draw(f, &test_footer())).unwrap();
        let text = buffer_text(term.backend().buffer());
        assert!(text.contains("Ask nanobot-rs anything"), "welcome hint missing");
    }
}
