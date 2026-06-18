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

use std::path::{Path, PathBuf};
use std::time::Instant;

use once_cell::sync::Lazy;
use ratatui::crossterm::event::{
    Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseEventKind,
};
use ratatui::layout::{Alignment, Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, BorderType, Borders, Clear, Padding, Paragraph};
use ratatui::Frame;
use ratatui_textarea::{TextArea, WrapMode};
use regex::Regex;
use serde_json::Value;
use unicode_width::UnicodeWidthChar;

use super::render;
use crate::agent::audit::ToolEvent;
use crate::repl::commands::ModelEntry;
use crate::repl::{parse_control_marker, CacheResetReason, CacheStatus, ControlMarker};

const BRAND: &str = "\u{259e}"; // ▞  the nanobot wordmark glyph
const DOT: &str = "\u{2022}"; //   •  status / user marker
const RUN: &str = "\u{25b6}"; //   ▶
const OK: &str = "\u{2713}"; //   ✓
const ERR: &str = "\u{2717}"; //  ✗
const TURN: &str = "\u{21b3}"; // ↳
const META: &str = "\u{25b8}"; // ▸
const RULE: &str = "\u{2500}"; // ─
const TOOL_BRIDGE_TEXT: &str = "Sure, on it ...";
const ACCENT: Color = Color::Rgb(0x28, 0xB3, 0xC1);
const OK_COLOR: Color = Color::Rgb(0x39, 0xB8, 0x45);
const WARN_COLOR: Color = Color::Rgb(0xC8, 0xA1, 0x3A);
const ERR_COLOR: Color = Color::Rgb(0xE0, 0x66, 0x66);

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
    Submit(SubmittedTurn),
    /// Voice mode: start a record → transcribe → reply → speak cycle.
    Record,
    /// User picked a model in the native picker; apply it by id.
    PickModel(String),
}

/// What a key/mouse/paste event means while the assistant is streaming.
pub(crate) enum StreamingAction {
    Continue,
    Cancel,
    CancelAndSubmit(SubmittedTurn),
}

/// User input ready for one agent turn.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SubmittedTurn {
    pub text: String,
    pub media: Vec<String>,
}

impl SubmittedTurn {
    pub(crate) fn display_text(&self) -> String {
        if self.media.is_empty() {
            return self.text.clone();
        }
        let mut out = self.text.clone();
        for path in &self.media {
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str("[image: ");
            out.push_str(
                Path::new(path)
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or(path),
            );
            out.push(']');
        }
        out
    }
}

/// One selectable row in the model picker.
struct PickRow {
    id: String,
    label: String,
    active: bool,
}

/// Native in-TUI model picker (replaces the classic screen-switching list).
struct ModelPicker {
    rows: Vec<PickRow>,
    selected: usize,
}

/// Disclosure level — how much tool output and metadata the transcript shows.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Mode {
    /// Minimal: collapsed tool summaries, compact metadata.
    Calm,
    /// Tool output previews + full metadata.
    Inspect,
    /// Full (capped) tool output + full metadata.
    Deep,
}

impl Mode {
    fn next(self) -> Self {
        match self {
            Mode::Calm => Mode::Inspect,
            Mode::Inspect => Mode::Deep,
            Mode::Deep => Mode::Calm,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Mode::Calm => "calm",
            Mode::Inspect => "inspect",
            Mode::Deep => "deep",
        }
    }

    fn parse(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().as_str() {
            "calm" => Some(Mode::Calm),
            "inspect" => Some(Mode::Inspect),
            "deep" => Some(Mode::Deep),
            _ => None,
        }
    }

    /// Tool-output lines to show: `None` = summary only (calm), else a cap.
    fn tool_output_lines(self) -> Option<usize> {
        match self {
            Mode::Calm => None,
            Mode::Inspect => Some(6),
            Mode::Deep => Some(usize::MAX),
        }
    }

    fn verbose_meta(self) -> bool {
        !matches!(self, Mode::Calm)
    }
}

/// Reverse-search (Ctrl+R) state.
struct Search {
    query: String,
    /// History index of the current match, if any.
    idx: Option<usize>,
}

/// Status of a single tool invocation.
enum ToolState {
    Running,
    Ok,
    Err,
}

/// Live turn-local status shown exactly where the answer will start.
enum ActivityPhase {
    Prefill,
    Thinking,
    Decoding,
}

/// One chronological entry in the transcript.
enum Cell {
    /// A user turn (verbatim, may be multi-line).
    User(String),
    /// A live status line under the user turn while the assistant is warming up.
    Activity {
        phase: ActivityPhase,
        prefill: Option<(u64, u64)>,
        prefill_estimate: Option<u64>,
        prefill_tps: Option<f32>,
        prefill_tps_estimated: bool,
        cache: Option<CacheStatus>,
    },
    /// Assistant text. Streamed deltas append here until a tool interrupts.
    Reply(String),
    /// Model reasoning (`<think>` / `reasoning_content`), rendered as a muted
    /// block above the reply when the provider exposes hidden thoughts.
    Thinking(String),
    /// A tool call, updated in place when its `CallEnd` arrives.
    Tool {
        id: String,
        name: String,
        args: String,
        state: ToolState,
        summary: Option<String>,
        /// Full (capped) tool output, shown in inspect/deep mode.
        output: String,
        /// Last progress preview while running.
        preview: Option<String>,
        ms: u64,
    },
    /// Per-turn metadata pinned beneath the turn's output.
    Meta {
        elapsed_s: f32,
        ttft_s: Option<f32>,
        prefill_tps: Option<f32>,
        cache: Option<CacheStatus>,
        tokens: u64,
        /// Generation throughput (tokens / decode time), when computable.
        tps: Option<f32>,
        /// True when `tokens` was estimated (provider didn't report a count).
        estimated: bool,
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
    /// Prompt tokens reported by the provider for the first call in this turn.
    /// This is paired with `ttft`, which is also the first-call latency.
    turn_prompt_tokens: u64,
    /// Estimated uncached prompt work for the first call in this turn.
    turn_prefill_estimate: u64,
    /// Cache status for the turn's first LLM call, unless a later call diverges.
    turn_cache: Option<CacheStatus>,
    /// Server-reported prefill progress `(processed, total)`, when available.
    prefill: Option<(u64, u64)>,
    /// Estimated uncached prompt work for the currently visible prefill row.
    prefill_estimate: Option<u64>,
    /// Effective prefill throughput computed from progress markers.
    prefill_tps: Option<f32>,
    /// Last observed prefill throughput this turn, preserved after the activity row clears.
    turn_prefill_tps: Option<f32>,
    /// Last known effective prefill throughput across turns, used for ETA when
    /// the server does not stream native progress.
    prefill_rate_hint: Option<f32>,
    /// First time we entered prefill for this turn.
    prefill_started: Option<Instant>,
    /// Last prefill progress sample `(processed_tokens, timestamp)`.
    prefill_last: Option<(u64, Instant)>,
    /// True while legacy ANSI-delimited thinking deltas are being received.
    in_thinking_stream: bool,
    /// Rows scrolled up from the bottom; `0` sticks to the latest output.
    scroll_from_bottom: usize,
    /// Max scrollable rows from the last draw, used to clamp `scroll_from_bottom`
    /// so paging back down always reaches the bottom.
    max_scroll: usize,
    /// Whether the help overlay is showing.
    show_help: bool,
    /// Whether voice mode is active (Enter on an empty line records).
    voice: bool,
    /// Whether a voice recording is in progress (frozen UI, shown in header).
    recording: bool,
    /// Whether the assistant reply is being spoken (Enter barges in).
    speaking: bool,
    /// Assistant text accumulated this turn, for token estimation when the
    /// provider doesn't report a completion-token count.
    turn_text: String,
    /// Submitted-prompt history (most recent last).
    history: Vec<String>,
    /// Browse position in `history` (None = editing a fresh draft).
    hist_pos: Option<usize>,
    /// Draft saved while browsing history or reverse-searching.
    draft: String,
    /// Active reverse-search (Ctrl+R), if any.
    search: Option<Search>,
    /// Native model picker overlay, if open.
    picker: Option<ModelPicker>,
    /// Disclosure level (calm / inspect / deep).
    mode: Mode,
    /// View-only toggle for model reasoning blocks. Thinking remains enabled
    /// and stored; this only decides whether `Cell::Thinking` rows render.
    show_thinking: bool,
    /// Local image paths attached to the next submitted turn.
    attachments: Vec<String>,
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
            turn_prompt_tokens: 0,
            turn_prefill_estimate: 0,
            turn_cache: None,
            prefill: None,
            prefill_estimate: None,
            prefill_tps: None,
            turn_prefill_tps: None,
            prefill_rate_hint: None,
            prefill_started: None,
            prefill_last: None,
            in_thinking_stream: false,
            scroll_from_bottom: 0,
            max_scroll: 0,
            show_help: false,
            voice: false,
            recording: false,
            speaking: false,
            turn_text: String::new(),
            history: Vec::new(),
            hist_pos: None,
            draft: String::new(),
            search: None,
            picker: None,
            mode: Mode::Calm,
            show_thinking: true,
            attachments: Vec::new(),
        }
    }

    /// Cycle calm → inspect → deep → calm (`/mode` with no argument).
    pub(crate) fn cycle_mode(&mut self) {
        self.mode = self.mode.next();
    }

    /// Set the mode by name; returns false if the name is unknown.
    pub(crate) fn set_mode(&mut self, name: &str) -> bool {
        match Mode::parse(name) {
            Some(m) => {
                self.mode = m;
                true
            }
            None => false,
        }
    }

    pub(crate) fn mode_label(&self) -> &'static str {
        self.mode.label()
    }

    fn toggle_thinking_display(&mut self) {
        self.show_thinking = !self.show_thinking;
    }

    /// Open the native model picker, selecting the active model if present.
    pub(crate) fn open_model_picker(&mut self, entries: Vec<ModelEntry>) {
        let rows: Vec<PickRow> = entries
            .iter()
            .map(|e| PickRow {
                id: e.id.clone(),
                label: format!(
                    "{}   {}{}",
                    e.id,
                    e.source_tag(),
                    if e.is_loaded { " · loaded" } else { "" }
                ),
                active: e.is_active,
            })
            .collect();
        let selected = rows.iter().position(|r| r.active).unwrap_or(0);
        self.picker = Some(ModelPicker { rows, selected });
    }

    fn on_picker_key(&mut self, k: KeyEvent) -> Action {
        let ctrl = k.modifiers.contains(KeyModifiers::CONTROL);
        let Some(p) = self.picker.as_mut() else {
            return Action::Continue;
        };
        match k.code {
            KeyCode::Char('d') if ctrl => Action::Quit,
            KeyCode::Up => {
                p.selected = p.selected.saturating_sub(1);
                Action::Continue
            }
            KeyCode::Down => {
                if p.selected + 1 < p.rows.len() {
                    p.selected += 1;
                }
                Action::Continue
            }
            KeyCode::Enter => {
                let id = p.rows.get(p.selected).map(|r| r.id.clone());
                self.picker = None;
                match id {
                    Some(id) => Action::PickModel(id),
                    None => Action::Continue,
                }
            }
            KeyCode::Esc => {
                self.picker = None;
                Action::Continue
            }
            _ => Action::Continue,
        }
    }

    /// Toggle the "recording" indicator (drawn in the header during a voice turn).
    pub(crate) fn set_recording(&mut self, on: bool) {
        self.recording = on;
    }

    /// Toggle the "speaking" indicator (TTS playing; Enter barges in).
    pub(crate) fn set_speaking(&mut self, on: bool) {
        self.speaking = on;
    }

    /// Open or close the help overlay.
    pub(crate) fn set_help(&mut self, on: bool) {
        self.show_help = on;
    }

    /// Reflect whether voice mode is active (set after a `/voice` toggle).
    pub(crate) fn set_voice(&mut self, on: bool) {
        self.voice = on;
    }

    /// Drop all transcript history (`/clear`).
    pub(crate) fn clear_transcript(&mut self) {
        self.transcript.clear();
        self.scroll_from_bottom = 0;
    }

    // --- turn lifecycle -----------------------------------------------------

    pub(crate) fn begin_turn(&mut self, input: &str) {
        self.transcript.push(Cell::User(input.to_string()));
        self.transcript.push(Cell::Activity {
            phase: ActivityPhase::Prefill,
            prefill: None,
            prefill_estimate: None,
            prefill_tps: None,
            prefill_tps_estimated: false,
            cache: None,
        });
        self.streaming = true;
        self.awaiting_first = true;
        self.got_text = false;
        self.turn_produced = false;
        self.turn_start = Some(Instant::now());
        self.ttft = None;
        self.turn_tokens = 0;
        self.turn_prompt_tokens = 0;
        self.turn_prefill_estimate = 0;
        self.turn_cache = None;
        self.turn_text.clear();
        self.prefill = None;
        self.prefill_estimate = None;
        self.prefill_tps = None;
        self.turn_prefill_tps = None;
        let now = Instant::now();
        self.prefill_started = Some(now);
        self.prefill_last = None;
        self.in_thinking_stream = false;
        self.scroll_from_bottom = 0; // jump to bottom for the new turn
    }

    /// Consume one text delta. Control markers (prefill/tokens/finish) are
    /// interpreted, never rendered; everything else is assistant text.
    pub(crate) fn on_delta(&mut self, d: &str) {
        if self.on_thinking_style_delta(d) {
            return;
        }
        match parse_control_marker(d) {
            Some(ControlMarker::PrefillProgress { processed, total }) => {
                if self.streaming {
                    if self.prefill_started.is_none() {
                        self.prefill_started = Some(Instant::now());
                        self.prefill_last = None;
                    }
                    self.prefill = Some((processed, total));
                    self.record_prefill_progress(processed);
                    self.upsert_activity(ActivityPhase::Prefill, self.prefill);
                }
            }
            Some(ControlMarker::CacheStatus(status)) => self.upsert_activity_cache(status),
            Some(ControlMarker::Tokens(n)) => self.turn_tokens += n,
            Some(ControlMarker::PromptTokens(n)) => {
                if self.turn_prompt_tokens == 0 {
                    self.turn_prompt_tokens = n;
                }
            }
            Some(ControlMarker::PrefillEstimate(n)) => {
                if self.streaming {
                    self.prefill_estimate = (n > 0).then_some(n);
                    if self.awaiting_first && self.turn_prefill_estimate == 0 {
                        self.turn_prefill_estimate = n;
                    }
                    self.upsert_activity(ActivityPhase::Prefill, self.prefill);
                }
            }
            Some(ControlMarker::RetractReply) => self.retract_streamed_reply(),
            Some(ControlMarker::FinishReason(_)) => {}
            None => {
                // Between the thinking ANSI markers, deltas are reasoning text;
                // accumulate them into a muted Thinking block instead of the reply.
                if self.in_thinking_stream {
                    self.push_thinking(d);
                } else {
                    self.push_text(d);
                }
            }
        }
    }

    fn push_text(&mut self, d: &str) {
        let appending_reply = matches!(self.transcript.last(), Some(Cell::Reply(_)));
        let d = if appending_reply {
            d
        } else {
            trim_leading_reply_gap(d)
        };
        if d.is_empty() {
            return;
        }
        self.mark_first_output();
        self.remove_trailing_activity();
        self.got_text = true;
        self.turn_text.push_str(d);
        if let Some(Cell::Reply(t)) = self.transcript.last_mut() {
            t.push_str(d);
        } else {
            self.transcript.push(Cell::Reply(d.to_string()));
        }
    }

    /// Append reasoning text to the current muted Thinking block (created above
    /// the reply). Not counted as answer text (`got_text`/`turn_text`).
    fn push_thinking(&mut self, d: &str) {
        self.mark_first_output();
        self.remove_trailing_activity();
        let new_block = !matches!(self.transcript.last(), Some(Cell::Thinking(_)));
        let d = clean_thinking_delta(d, new_block);
        if d.is_empty() {
            return;
        }
        if let Some(Cell::Thinking(t)) = self.transcript.last_mut() {
            t.push_str(&d);
        } else {
            self.transcript.push(Cell::Thinking(d));
        }
    }

    fn on_thinking_style_delta(&mut self, d: &str) -> bool {
        match d {
            "\x1b[90m\x1b[2m" => {
                self.in_thinking_stream = true;
                self.mark_first_activity();
                self.upsert_activity(ActivityPhase::Thinking, None);
                true
            }
            "\x1b[0m\n\n" if self.in_thinking_stream => {
                self.in_thinking_stream = false;
                self.upsert_activity(ActivityPhase::Decoding, None);
                true
            }
            _ => false,
        }
    }

    /// Record the first observable output of a turn (ends prefill, stamps ttft).
    fn mark_first_output(&mut self) {
        self.turn_produced = true;
        self.mark_first_activity();
    }

    fn mark_first_activity(&mut self) {
        self.awaiting_first = false;
        self.prefill = None;
        self.prefill_estimate = None;
        self.prefill_tps = None;
        self.prefill_started = None;
        self.prefill_last = None;
        if self.ttft.is_none() {
            self.ttft = Some(self.elapsed_s());
        }
    }

    fn record_prefill_progress(&mut self, processed: u64) {
        let now = Instant::now();
        let rate = self
            .prefill_last
            .and_then(|(prev_processed, prev_at)| {
                let delta_tokens = processed.saturating_sub(prev_processed);
                let delta_secs = now.duration_since(prev_at).as_secs_f32();
                (delta_tokens > 0 && delta_secs >= 0.05).then_some(delta_tokens as f32 / delta_secs)
            })
            .or_else(|| {
                let elapsed = now.duration_since(self.prefill_started?).as_secs_f32();
                (processed > 0 && elapsed >= 0.05).then_some(processed as f32 / elapsed)
            });
        if let Some(rate) = rate {
            self.prefill_tps = Some(rate);
            self.turn_prefill_tps = Some(rate);
        }
        self.prefill_last = Some((processed, now));
    }

    fn activity_prefill_rate(&self) -> (Option<f32>, bool) {
        if let Some(rate) = self.prefill_tps {
            return (Some(rate), false);
        }
        if self.prefill.is_none() && self.prefill_estimate.is_some() {
            return (self.prefill_rate_hint, self.prefill_rate_hint.is_some());
        }
        (None, false)
    }

    fn upsert_activity(&mut self, phase: ActivityPhase, prefill: Option<(u64, u64)>) {
        let (rate, rate_estimated) = self.activity_prefill_rate();
        if let Some(Cell::Activity {
            phase: p,
            prefill: pf,
            prefill_estimate,
            prefill_tps,
            prefill_tps_estimated,
            ..
        }) = self.transcript.last_mut()
        {
            *p = phase;
            *pf = prefill;
            *prefill_estimate = self.prefill_estimate;
            *prefill_tps = rate;
            *prefill_tps_estimated = rate_estimated;
        } else {
            self.transcript.push(Cell::Activity {
                phase,
                prefill,
                prefill_estimate: self.prefill_estimate,
                prefill_tps: rate,
                prefill_tps_estimated: rate_estimated,
                cache: None,
            });
        }
    }

    fn upsert_activity_cache(&mut self, cache: CacheStatus) {
        self.remember_cache_status(cache);
        if !self.streaming {
            return;
        }
        let already_prefilling = matches!(
            self.transcript.last(),
            Some(Cell::Activity {
                phase: ActivityPhase::Prefill,
                ..
            })
        );
        if !already_prefilling {
            self.prefill = None;
            self.prefill_estimate = None;
            self.prefill_tps = None;
            self.prefill_started = Some(Instant::now());
            self.prefill_last = None;
            self.upsert_activity(ActivityPhase::Prefill, None);
        }
        if let Some(Cell::Activity { cache: c, .. }) = self.transcript.last_mut() {
            if !should_replace_cache_status(*c, cache) {
                return;
            }
            *c = Some(cache);
        } else {
            self.transcript.push(Cell::Activity {
                phase: ActivityPhase::Prefill,
                prefill: self.prefill,
                prefill_estimate: self.prefill_estimate,
                prefill_tps: self.activity_prefill_rate().0,
                prefill_tps_estimated: self.activity_prefill_rate().1,
                cache: Some(cache),
            });
        }
    }

    fn remember_cache_status(&mut self, cache: CacheStatus) {
        if should_replace_cache_status(self.turn_cache, cache) {
            self.turn_cache = Some(cache);
        }
    }

    fn remove_trailing_activity(&mut self) {
        if matches!(self.transcript.last(), Some(Cell::Activity { .. })) {
            self.transcript.pop();
        }
    }

    fn retract_streamed_reply(&mut self) {
        self.remove_trailing_activity();
        let Some(turn_start) = self
            .transcript
            .iter()
            .rposition(|cell| matches!(cell, Cell::User(_)))
        else {
            return;
        };
        if let Some(rel_reply) = self.transcript[turn_start + 1..]
            .iter()
            .rposition(|cell| matches!(cell, Cell::Reply(_)))
        {
            self.transcript.remove(turn_start + 1 + rel_reply);
        }
        self.turn_text.clear();
        self.turn_tokens = 0;
        self.got_text = false;
        self.in_thinking_stream = false;
        self.turn_produced = self.transcript[turn_start + 1..]
            .iter()
            .any(|cell| matches!(cell, Cell::Thinking(_) | Cell::Tool { .. }));
    }

    pub(crate) fn on_tool_event(&mut self, ev: ToolEvent) {
        match ev {
            ToolEvent::CallStart {
                tool_name,
                tool_call_id,
                arguments_preview,
            } => {
                let had_activity = matches!(self.transcript.last(), Some(Cell::Activity { .. }));
                let keep_tool_bridge = had_activity && !self.turn_has_reply_or_tool();
                self.mark_first_output();
                if keep_tool_bridge {
                    self.upsert_activity(ActivityPhase::Decoding, None);
                } else if had_activity {
                    self.remove_trailing_activity();
                }
                self.transcript.push(Cell::Tool {
                    id: tool_call_id,
                    name: tool_name,
                    args: arguments_preview,
                    state: ToolState::Running,
                    summary: None,
                    output: String::new(),
                    preview: None,
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
            ToolEvent::Progress {
                tool_call_id,
                elapsed_ms,
                output_preview,
                ..
            } => self.update_tool_progress(&tool_call_id, elapsed_ms, output_preview),
        }
    }

    fn turn_has_reply_or_tool(&self) -> bool {
        self.transcript
            .iter()
            .rev()
            .take_while(|cell| !matches!(cell, Cell::User(_)))
            .any(|cell| matches!(cell, Cell::Reply(_) | Cell::Tool { .. }))
    }

    fn update_tool_progress(&mut self, id: &str, elapsed_ms: u64, preview: Option<String>) {
        if let Some(Cell::Tool { ms, preview: p, .. }) = self
            .transcript
            .iter_mut()
            .rfind(|c| matches!(c, Cell::Tool { id: cid, .. } if cid.as_str() == id))
        {
            *ms = elapsed_ms;
            *p = preview;
        }
    }

    fn complete_tool(&mut self, id: &str, name: &str, data: &str, ok: bool, ms: u64) {
        let summary = summarize_output(data, ok);
        let output: String = data.chars().take(4000).collect(); // cap: never dump huge
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
                    output: out,
                    preview: p,
                    ms: m,
                    ..
                } = &mut self.transcript[i]
                {
                    *s = state;
                    *sum = summary;
                    *out = output;
                    *p = None;
                    *m = ms;
                }
            }
            None => self.transcript.push(Cell::Tool {
                id: id.to_string(),
                name: name.to_string(),
                args: String::new(),
                state,
                summary,
                output,
                preview: None,
                ms,
            }),
        }
        if self.streaming {
            self.upsert_activity(ActivityPhase::Decoding, None);
        }
    }

    /// Finalize the turn. `resp` is the agent's full return string; it backfills
    /// a `Reply` when the provider didn't stream any deltas, and pins a metadata
    /// line beneath the turn's output.
    pub(crate) fn finish_turn(&mut self, resp: String) {
        self.streaming = false;
        self.awaiting_first = false;
        self.prefill = None;
        self.prefill_estimate = None;
        self.in_thinking_stream = false;
        if !self.turn_produced && !resp.trim().is_empty() {
            // Non-streaming providers return the whole string at once; stamp ttft
            // so the metadata line still shows all three fields.
            if self.ttft.is_none() {
                self.ttft = Some(self.elapsed_s());
            }
            self.remove_trailing_activity();
            self.turn_text = resp.clone();
            self.transcript.push(Cell::Reply(resp));
            self.turn_produced = true;
        }
        self.remove_trailing_activity();
        if let Some(Cell::Reply(t)) = self.transcript.last() {
            if t.trim().is_empty() {
                self.transcript.pop();
            }
        }
        if self.turn_produced {
            // Prefer the provider-reported count; otherwise estimate from the
            // reply text so throughput shows for models that omit usage.
            let reported = self.turn_tokens;
            let tokens = if reported > 0 {
                reported
            } else {
                estimate_tokens(&self.turn_text)
            };
            let elapsed = self.elapsed_s();
            let gen_time = match self.ttft {
                Some(t) => (elapsed - t).max(0.05),
                None => elapsed.max(0.05),
            };
            let tps = (tokens > 0).then(|| tokens as f32 / gen_time);
            let prefer_estimated_work = matches!(
                self.turn_cache,
                Some(CacheStatus::AppendOnly { .. })
                    | Some(CacheStatus::Diverged { .. })
                    | Some(CacheStatus::Reset { .. })
            );
            let prefill_work_tokens = if prefer_estimated_work && self.turn_prefill_estimate > 0 {
                self.turn_prefill_estimate
            } else if self.turn_prompt_tokens > 0 {
                self.turn_prompt_tokens
            } else {
                self.turn_prefill_estimate
            };
            let estimated_prefill_tps = self
                .ttft
                .filter(|t| *t > 0.05)
                .and_then(|t| (prefill_work_tokens > 0).then(|| prefill_work_tokens as f32 / t));
            let prefill_tps = self.turn_prefill_tps.or(estimated_prefill_tps);
            if let Some(rate) = prefill_tps.filter(|r| r.is_finite() && *r > 0.0) {
                self.prefill_rate_hint = Some(match self.prefill_rate_hint {
                    Some(prev) if prev.is_finite() && prev > 0.0 => prev * 0.7 + rate * 0.3,
                    _ => rate,
                });
            }
            self.transcript.push(Cell::Meta {
                elapsed_s: elapsed,
                ttft_s: self.ttft,
                prefill_tps,
                cache: self.turn_cache,
                tokens,
                tps,
                estimated: reported == 0,
            });
        }
    }

    pub(crate) fn push_note(&mut self, note: String) {
        self.transcript.push(Cell::Note(note));
    }

    /// Tick exists to drive redraws for live activity rows; elapsed time is read
    /// from the wall clock, so this is intentionally a no-op.
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
            Event::Key(k) if is_press(&k) => {
                // Help overlay is modal: any key dismisses it (Ctrl+D still quits).
                if self.show_help {
                    let ctrl = k.modifiers.contains(KeyModifiers::CONTROL);
                    if ctrl && k.code == KeyCode::Char('d') {
                        return Action::Quit;
                    }
                    self.show_help = false;
                    return Action::Continue;
                }
                if self.picker.is_some() {
                    return self.on_picker_key(k);
                }
                self.on_idle_key(k)
            }
            Event::Paste(s) => {
                self.on_paste(&s);
                Action::Continue
            }
            Event::Mouse(m) => {
                self.on_mouse(m.kind);
                Action::Continue
            }
            _ => Action::Continue,
        }
    }

    fn on_idle_key(&mut self, k: KeyEvent) -> Action {
        let ctrl = k.modifiers.contains(KeyModifiers::CONTROL);
        if self.search.is_some() {
            return self.on_search_key(k, ctrl);
        }
        match k.code {
            // Ctrl+D is the explicit quit (EOF). Ctrl+C is NOT a kill switch:
            // it clears a typed line, otherwise does nothing.
            KeyCode::Char('d') if ctrl => Action::Quit,
            KeyCode::Char('c') if ctrl => {
                self.clear_input();
                Action::Continue
            }
            KeyCode::Char('r') if ctrl => {
                self.start_search();
                Action::Continue
            }
            KeyCode::Char('t') if ctrl => {
                self.toggle_thinking_display();
                Action::Continue
            }
            // Esc goes back: clears the input line (and closes help/cancels a
            // turn in their handlers).
            KeyCode::Esc => {
                self.clear_input();
                Action::Continue
            }
            // `?` opens help only when the input is empty, so it can still be typed.
            KeyCode::Char('?') if !ctrl && self.input_is_empty() => {
                self.show_help = true;
                Action::Continue
            }
            // Up/Down recall prompt history at the input's edges; otherwise they
            // move the cursor within multi-line input.
            KeyCode::Up => {
                if self.input.cursor().0 == 0 {
                    self.history_prev();
                } else {
                    self.input.input(k);
                }
                Action::Continue
            }
            KeyCode::Down => {
                let last = self.input.lines().len().saturating_sub(1);
                if self.input.cursor().0 >= last {
                    self.history_next();
                } else {
                    self.input.input(k);
                }
                Action::Continue
            }
            KeyCode::Enter if k.modifiers.is_empty() => {
                if self.voice && self.input_is_empty() {
                    Action::Record
                } else {
                    self.submit()
                }
            }
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

    fn on_paste(&mut self, text: &str) {
        let (cleaned, media) = extract_image_attachments(text);
        if !media.is_empty() && cleaned.trim().is_empty() {
            self.add_attachments(media);
        } else {
            self.input.insert_str(text);
        }
    }

    fn submit(&mut self) -> Action {
        let text = self.input.lines().join("\n");
        let (cleaned, mut media) = extract_image_attachments(&text);
        media.extend(std::mem::take(&mut self.attachments));
        dedupe(&mut media);
        let trimmed = cleaned.trim();
        if trimmed.is_empty() && media.is_empty() {
            return Action::Continue;
        }
        let owned = if trimmed.is_empty() {
            "Please look at the attached image.".to_string()
        } else {
            trimmed.to_string()
        };
        if self.history.last() != Some(&owned) {
            self.history.push(owned.clone());
        }
        self.hist_pos = None;
        self.input = configure_input();
        Action::Submit(SubmittedTurn { text: owned, media })
    }

    fn input_is_empty(&self) -> bool {
        self.input.lines().iter().all(|l| l.is_empty())
    }

    fn clear_input(&mut self) {
        self.input = configure_input();
        self.attachments.clear();
        self.hist_pos = None;
    }

    fn load_input(&mut self, text: &str) {
        self.input = configure_input();
        self.input.insert_str(text);
    }

    // --- prompt history (Up/Down) ------------------------------------------

    fn history_prev(&mut self) {
        if self.history.is_empty() {
            return;
        }
        let idx = match self.hist_pos {
            None => {
                self.draft = self.input.lines().join("\n");
                self.history.len() - 1
            }
            Some(0) => 0,
            Some(i) => i - 1,
        };
        self.hist_pos = Some(idx);
        self.load_input(&self.history[idx].clone());
    }

    fn history_next(&mut self) {
        match self.hist_pos {
            Some(i) if i + 1 < self.history.len() => {
                self.hist_pos = Some(i + 1);
                self.load_input(&self.history[i + 1].clone());
            }
            Some(_) => {
                self.hist_pos = None;
                self.load_input(&self.draft.clone());
            }
            None => {}
        }
    }

    // --- reverse search (Ctrl+R) -------------------------------------------

    fn start_search(&mut self) {
        self.draft = self.input.lines().join("\n");
        self.search = Some(Search {
            query: String::new(),
            idx: None,
        });
    }

    fn on_search_key(&mut self, k: KeyEvent, ctrl: bool) -> Action {
        match k.code {
            KeyCode::Esc => self.cancel_search(),
            KeyCode::Enter => self.search = None, // accept: keep the matched line
            KeyCode::Char('c') if ctrl => self.cancel_search(),
            KeyCode::Char('r') if ctrl => self.search_older(),
            KeyCode::Backspace => {
                if let Some(s) = self.search.as_mut() {
                    s.query.pop();
                }
                self.refresh_search();
            }
            KeyCode::Char(c) if !ctrl => {
                if let Some(s) = self.search.as_mut() {
                    s.query.push(c);
                }
                self.refresh_search();
            }
            _ => {}
        }
        Action::Continue
    }

    fn refresh_search(&mut self) {
        let query = self
            .search
            .as_ref()
            .map(|s| s.query.clone())
            .unwrap_or_default();
        let idx = self.find_match(&query, None);
        if let Some(s) = self.search.as_mut() {
            s.idx = idx;
        }
        match idx {
            Some(i) => self.load_input(&self.history[i].clone()),
            None => self.load_input(""),
        }
    }

    fn search_older(&mut self) {
        let Some(s) = self.search.as_ref() else {
            return;
        };
        let (query, before) = (s.query.clone(), s.idx.unwrap_or(self.history.len()));
        if let Some(i) = self.find_match(&query, Some(before)) {
            if let Some(s) = self.search.as_mut() {
                s.idx = Some(i);
            }
            self.load_input(&self.history[i].clone());
        }
    }

    /// Most recent history index `< before` whose entry contains `query`.
    fn find_match(&self, query: &str, before: Option<usize>) -> Option<usize> {
        if query.is_empty() {
            return None;
        }
        let end = before.unwrap_or(self.history.len()).min(self.history.len());
        self.history[..end].iter().rposition(|h| h.contains(query))
    }

    fn cancel_search(&mut self) {
        self.search = None;
        self.load_input(&self.draft.clone());
    }

    /// Handle an event while a turn is streaming.
    pub(crate) fn on_streaming_event(&mut self, ev: Event) -> StreamingAction {
        match ev {
            Event::Key(k) if is_press(&k) => {
                if self.show_help {
                    self.show_help = false;
                    return StreamingAction::Continue;
                }
                let ctrl = k.modifiers.contains(KeyModifiers::CONTROL);
                match k.code {
                    KeyCode::Esc => {
                        self.clear_input();
                        StreamingAction::Cancel
                    }
                    KeyCode::Char('c') if ctrl => {
                        self.clear_input();
                        StreamingAction::Cancel
                    }
                    KeyCode::Char('t') if ctrl => {
                        self.toggle_thinking_display();
                        StreamingAction::Continue
                    }
                    KeyCode::Enter if k.modifiers.is_empty() => match self.submit() {
                        Action::Submit(turn) => StreamingAction::CancelAndSubmit(turn),
                        _ => StreamingAction::Cancel,
                    },
                    KeyCode::PageUp => {
                        self.scroll_up(10);
                        StreamingAction::Continue
                    }
                    KeyCode::PageDown => {
                        self.scroll_down(10);
                        StreamingAction::Continue
                    }
                    KeyCode::Up => {
                        if self.input.cursor().0 == 0 {
                            self.history_prev();
                        } else {
                            self.input.input(k);
                        }
                        StreamingAction::Continue
                    }
                    KeyCode::Down => {
                        let last = self.input.lines().len().saturating_sub(1);
                        if self.input.cursor().0 >= last {
                            self.history_next();
                        } else {
                            self.input.input(k);
                        }
                        StreamingAction::Continue
                    }
                    _ => {
                        self.input.input(k);
                        StreamingAction::Continue
                    }
                }
            }
            Event::Paste(s) => {
                self.on_paste(&s);
                StreamingAction::Continue
            }
            Event::Mouse(m) => {
                self.on_mouse(m.kind);
                StreamingAction::Continue
            }
            _ => StreamingAction::Continue,
        }
    }

    fn on_mouse(&mut self, kind: MouseEventKind) {
        match kind {
            MouseEventKind::ScrollUp => self.scroll_up(3),
            MouseEventKind::ScrollDown => self.scroll_down(3),
            _ => {}
        }
    }

    fn add_attachments(&mut self, paths: Vec<String>) {
        for path in paths {
            if !self.attachments.contains(&path) {
                self.attachments.push(path);
            }
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
        if self.transcript.is_empty() {
            render_intro(f, transcript);
        } else {
            let width = transcript.width.max(1) as usize;
            let height = transcript.height as usize;
            let rows = self.transcript_rows(width);
            let max_off = rows.len().saturating_sub(height);
            self.max_scroll = max_off;
            self.scroll_from_bottom = self.scroll_from_bottom.min(max_off);
            let visible: Vec<Line> = if rows.len() < height && self.scroll_from_bottom == 0 {
                let mut visible = vec![Line::default(); height - rows.len()];
                visible.extend(rows);
                visible
            } else {
                let off = max_off - self.scroll_from_bottom;
                rows.into_iter().skip(off).take(height).collect()
            };
            f.render_widget(Paragraph::new(Text::from(visible)), transcript);
        }

        let title = self.input_title();
        self.input.set_block(input_block(title));
        f.render_widget(&self.input, chunks[2]);
        f.render_widget(
            Paragraph::new(footer_line(footer, self.mode, self.show_thinking)),
            chunks[3],
        );

        if self.show_help {
            render_help(f, area);
        }
        if let Some(p) = &self.picker {
            render_picker(f, area, p);
        }
    }

    fn input_height(&self) -> u16 {
        let n = self.input.lines().len().max(1) as u16;
        n.min(8) + 2 // +2 for the rounded border
    }

    fn input_title(&self) -> Option<String> {
        if let Some(s) = &self.search {
            return Some(format!(" reverse-search \u{2039}{}\u{203a} ", s.query));
        }
        if self.streaming {
            return Some(if self.input_is_empty() && self.attachments.is_empty() {
                " Type and press Enter or ESC to interrupt ".to_string()
            } else {
                " Enter interrupts and sends · Esc cancels ".to_string()
            });
        }
        if self.attachments.is_empty() {
            None
        } else {
            let suffix = if self.attachments.len() == 1 { "" } else { "s" };
            Some(format!(" {} image{} ", self.attachments.len(), suffix))
        }
    }

    fn header_lines(&self, width: usize) -> Text<'static> {
        let mut status = vec![Span::raw("  ")];
        status.extend(brand_mark());
        status.push(Span::styled(" TRENTADUE", style(Color::White, true)));
        status.push(Span::styled(" \u{b7} 32", dim_color(ACCENT)));
        status.push(Span::styled("  \u{b7}  ", dim()));
        if self.recording {
            status.push(Span::styled(
                format!("{DOT} recording"),
                style(ERR_COLOR, true),
            ));
            status.push(Span::styled("  \u{b7}  Enter/Esc to stop", dim()));
        } else if self.speaking {
            status.push(Span::styled(format!("{DOT} speaking"), style(ACCENT, true)));
            status.push(Span::styled("  \u{b7}  Enter to interrupt", dim()));
        } else if self.streaming {
            status.push(Span::styled("working", dim_color(ACCENT)));
            status.push(Span::styled("  \u{b7}  type to interrupt", dim()));
        } else if self.voice {
            status.push(Span::styled("voice", style(OK_COLOR, false)));
            status.push(Span::styled("  ·  Enter to speak", dim()));
        } else {
            status.push(Span::styled("ready", style(OK_COLOR, false)));
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
        let mut prev_was_reply = false;
        let mut prev_was_tool = false;
        let mut assistant_mark_used = false;
        for cell in &self.transcript {
            if matches!(cell, Cell::Thinking(_)) && !self.show_thinking {
                continue;
            }
            // Blank line before each user turn separates conversation turns.
            if matches!(cell, Cell::User(_)) && !rows.is_empty() {
                rows.push(Line::default());
            }
            if matches!(cell, Cell::User(_)) {
                assistant_mark_used = false;
            }
            if matches!(cell, Cell::Tool { .. }) && prev_was_reply {
                rows.push(Line::default());
            }
            if matches!(cell, Cell::Reply(_)) && prev_was_tool {
                rows.push(Line::default());
            }
            let consumes_assistant_mark = cell_consumes_assistant_mark(cell);
            let show_reply_mark =
                (matches!(cell, Cell::Reply(_)) || consumes_assistant_mark) && !assistant_mark_used;
            for segs in
                cell_lines_with_reply_mark(cell, self.mode, self.elapsed_s(), show_reply_mark)
            {
                wrap_segments(&segs, width, &mut rows);
            }
            if matches!(cell, Cell::Reply(_)) || (consumes_assistant_mark && show_reply_mark) {
                assistant_mark_used = true;
            }
            // Breathing room between the user turn and the assistant reply.
            if matches!(cell, Cell::User(_)) {
                rows.push(Line::default());
            }
            prev_was_reply = matches!(cell, Cell::Reply(_));
            prev_was_tool = matches!(cell, Cell::Tool { .. });
        }
        if self.streaming && self.scroll_from_bottom == 0 {
            rows.push(Line::default());
        }
        rows
    }
}

// ---------------------------------------------------------------------------
// Free functions (rendering + wrapping)
// ---------------------------------------------------------------------------

static MD_IMAGE_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"!\[[^\]]*\]\(([^)\n]+)\)").expect("valid image regex"));

fn extract_image_attachments(text: &str) -> (String, Vec<String>) {
    let mut media = Vec::new();
    let without_markdown = MD_IMAGE_RE
        .replace_all(text, |caps: &regex::Captures<'_>| {
            let original = caps.get(0).map(|m| m.as_str()).unwrap_or("");
            let target = caps.get(1).map(|m| m.as_str()).unwrap_or("");
            match normalize_image_path(target) {
                Some(path) => {
                    media.push(path);
                    String::new()
                }
                None => original.to_string(),
            }
        })
        .to_string();

    let mut cleaned_lines = Vec::new();
    for line in without_markdown.lines() {
        let mut kept = Vec::new();
        for word in line.split_whitespace() {
            match normalize_image_path(word) {
                Some(path) => media.push(path),
                None => kept.push(word),
            }
        }
        cleaned_lines.push(kept.join(" "));
    }
    dedupe(&mut media);
    (cleaned_lines.join("\n"), media)
}

fn normalize_image_path(raw: &str) -> Option<String> {
    let trimmed = raw
        .trim()
        .trim_matches(|c| matches!(c, '<' | '>' | '"' | '\''))
        .trim_end_matches(|c| matches!(c, ',' | ';' | '.'));
    let pathish = trimmed.strip_prefix("file://").unwrap_or(trimmed);
    if !has_image_extension(pathish) {
        return None;
    }
    let expanded = expand_home(pathish);
    let path = Path::new(&expanded);
    if !path.is_file() {
        return None;
    }
    let canonical = std::fs::canonicalize(path).unwrap_or_else(|_| PathBuf::from(path));
    Some(canonical.to_string_lossy().to_string())
}

fn expand_home(path: &str) -> String {
    if path == "~" {
        return std::env::var("HOME").unwrap_or_else(|_| path.to_string());
    }
    if let Some(rest) = path.strip_prefix("~/") {
        if let Ok(home) = std::env::var("HOME") {
            return format!("{home}/{rest}");
        }
    }
    path.to_string()
}

fn has_image_extension(path: &str) -> bool {
    let lower = path.to_ascii_lowercase();
    matches!(
        lower.rsplit('.').next(),
        Some("png" | "jpg" | "jpeg" | "gif" | "webp" | "bmp" | "tif" | "tiff" | "svg")
    )
}

fn dedupe(items: &mut Vec<String>) {
    let mut out = Vec::with_capacity(items.len());
    for item in items.drain(..) {
        if !out.contains(&item) {
            out.push(item);
        }
    }
    *items = out;
}

fn input_block(title: Option<String>) -> Block<'static> {
    let mut b = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(dim())
        .padding(Padding::horizontal(1));
    if let Some(t) = title {
        b = b.title(Span::styled(t, style(ACCENT, false)));
    }
    b
}

fn configure_input() -> TextArea<'static> {
    let mut ta = TextArea::default();
    ta.set_wrap_mode(WrapMode::WordOrGlyph);
    ta.set_block(input_block(None));
    ta.set_placeholder_text("Ask nanobot-rs anything");
    ta.set_cursor_line_style(Style::default());
    ta
}

fn is_press(k: &KeyEvent) -> bool {
    k.kind != KeyEventKind::Release
}

/// Keep tool hints intact and let the transcript wrapper do the line breaking.
fn clip(s: &str, _max: usize) -> String {
    s.to_string()
}

fn activity_marker(elapsed_s: f32) -> String {
    match ((elapsed_s * 6.0) as usize) % 4 {
        0 => format!("{BRAND} "),
        1 => format!("{BRAND}{BRAND}"),
        2 => format!(" {BRAND}"),
        _ => format!("{BRAND}{BRAND}"),
    }
}

fn cell_consumes_assistant_mark(cell: &Cell) -> bool {
    matches!(
        cell,
        Cell::Activity {
            phase: ActivityPhase::Decoding,
            ..
        }
    )
}

fn cache_status_label(status: CacheStatus) -> (String, Color) {
    match status {
        CacheStatus::First { messages } => {
            (format!("cache priming · {} msg", messages), Color::DarkGray)
        }
        CacheStatus::AppendOnly { added, .. } => {
            if added == 0 {
                ("cache warm".to_string(), OK_COLOR)
            } else {
                (format!("cache warm · +{} msg", added), OK_COLOR)
            }
        }
        CacheStatus::Diverged { at, .. } => (format!("cache reset · msg {}", at), WARN_COLOR),
        CacheStatus::Reset { reason } => match reason {
            CacheResetReason::Trim => ("cache reset · trim".to_string(), WARN_COLOR),
            CacheResetReason::EmergencyTrim => {
                ("cache reset · emergency trim".to_string(), WARN_COLOR)
            }
        },
    }
}

fn should_replace_cache_status(existing: Option<CacheStatus>, incoming: CacheStatus) -> bool {
    let Some(existing) = existing else {
        return true;
    };
    cache_status_rank(incoming) >= cache_status_rank(existing)
}

fn cache_status_rank(status: CacheStatus) -> u8 {
    match status {
        CacheStatus::First { .. } | CacheStatus::AppendOnly { .. } => 1,
        CacheStatus::Diverged { .. } => 2,
        CacheStatus::Reset { .. } => 3,
    }
}

fn format_prefill_rate(tps: f32) -> String {
    if tps >= 1000.0 {
        format!("{:.1}K tok/s", tps / 1000.0)
    } else {
        format!("{:.0} tok/s", tps)
    }
}

fn compact_tool_label(name: &str, args: &str) -> String {
    match tool_arg_hint(args) {
        Some(hint) => format!("{name} {hint}"),
        None => name.to_string(),
    }
}

fn tool_arg_hint(args: &str) -> Option<String> {
    let value: Value = serde_json::from_str(args).ok()?;
    let obj = value.as_object()?;
    for key in ["name", "query", "url", "command", "path"] {
        let Some(raw) = obj.get(key).and_then(Value::as_str).map(str::trim) else {
            continue;
        };
        if raw.is_empty() {
            continue;
        }
        return Some(match key {
            "url" => compact_url(raw),
            "command" => clip(raw, 30),
            "path" => clip(raw.rsplit('/').next().unwrap_or(raw), 30),
            _ => clip(raw, 34),
        });
    }
    None
}

fn compact_url(url: &str) -> String {
    let without_scheme = url
        .strip_prefix("https://")
        .or_else(|| url.strip_prefix("http://"))
        .unwrap_or(url);
    let host = without_scheme.split('/').next().unwrap_or(without_scheme);
    if host.is_empty() {
        clip(url, 34)
    } else {
        clip(host, 34)
    }
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

fn thinking_text_style() -> Style {
    Style::default().fg(Color::Gray)
}

fn clean_thinking_delta(delta: &str, first_chunk: bool) -> String {
    let text = if first_chunk {
        strip_initial_thinking_marker(delta)
    } else {
        delta
    };
    strip_ascii_case_insensitive(text, "</think>")
}

fn trim_leading_reply_gap(mut text: &str) -> &str {
    loop {
        let Some(line_end) = text.find(['\n', '\r']) else {
            return text;
        };
        let (line, rest) = text.split_at(line_end);
        if !line.chars().all(|c| c == ' ' || c == '\t') {
            return text;
        }
        if let Some(rest) = rest.strip_prefix("\r\n") {
            text = rest;
        } else if let Some(rest) = rest.strip_prefix('\n') {
            text = rest;
        } else if let Some(rest) = rest.strip_prefix('\r') {
            text = rest;
        } else {
            return text;
        }
    }
}

fn strip_initial_thinking_marker(text: &str) -> &str {
    let mut text = text.trim_start();
    if let Some(rest) = strip_ascii_prefix_case_insensitive(text, "thinking") {
        let rest = rest.trim_start();
        if strip_ascii_prefix_case_insensitive(rest, "<think>").is_some() {
            text = rest;
        }
    }
    if let Some(rest) = strip_ascii_prefix_case_insensitive(text, "<think>") {
        text = rest.trim_start();
    }
    text
}

fn strip_ascii_prefix_case_insensitive<'a>(text: &'a str, prefix: &str) -> Option<&'a str> {
    let head = text.get(..prefix.len())?;
    head.eq_ignore_ascii_case(prefix)
        .then_some(&text[prefix.len()..])
}

fn strip_ascii_case_insensitive(text: &str, needle: &str) -> String {
    if needle.is_empty() {
        return text.to_string();
    }

    let Some(idx) = text
        .as_bytes()
        .windows(needle.len())
        .position(|window| window.eq_ignore_ascii_case(needle.as_bytes()))
    else {
        return text.to_string();
    };

    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    let mut next_idx = Some(idx);
    while let Some(idx) = next_idx {
        out.push_str(&rest[..idx]);
        rest = &rest[idx + needle.len()..];
        next_idx = rest
            .as_bytes()
            .windows(needle.len())
            .position(|window| window.eq_ignore_ascii_case(needle.as_bytes()));
    }
    out.push_str(rest);
    out
}

/// Quiet footer: cwd · model · mode · ctx usage.
fn footer_line(footer: &Footer, mode: Mode, show_thinking: bool) -> Line<'static> {
    let mut spans = vec![
        Span::styled("  cwd ", dim()),
        Span::styled(footer.cwd.clone(), style(OK_COLOR, false)),
        Span::styled("   model ", dim()),
        Span::styled(footer.model.clone(), style(OK_COLOR, false)),
        Span::styled("   mode ", dim()),
    ];
    for m in [Mode::Calm, Mode::Inspect, Mode::Deep] {
        let st = if m == mode {
            Style::default().fg(Color::Black).bg(OK_COLOR)
        } else {
            dim()
        };
        spans.push(Span::styled(format!(" {} ", m.label()), st));
    }
    if !show_thinking {
        spans.push(Span::styled("   think ", dim()));
        spans.push(Span::styled("hidden", style(WARN_COLOR, false)));
    }
    spans.push(Span::styled("   ctx ", dim()));
    if footer.ctx_max > 0 {
        let pct = footer.ctx_used * 100 / footer.ctx_max;
        let ctx_color = match pct {
            0..=49 => OK_COLOR,
            50..=79 => WARN_COLOR,
            _ => ERR_COLOR,
        };
        spans.push(Span::styled(
            crate::tui::format_tokens(footer.ctx_used),
            style(ctx_color, false),
        ));
        spans.push(Span::styled("/", dim()));
        spans.push(Span::styled(
            crate::tui::format_tokens(footer.ctx_max),
            dim(),
        ));
    } else {
        spans.push(Span::styled("\u{2014}", dim())); // —
    }
    Line::from(spans)
}

/// Centered branded welcome shown while the transcript is empty.
fn render_intro(f: &mut Frame, area: Rect) {
    let lines = vec![
        Line::from(Span::styled(
            format!("{BRAND}{BRAND}{BRAND}  TRENTADUE"),
            style(ACCENT, true),
        )),
        Line::from(Span::styled("a calm, local agent  \u{b7}  32", dim())),
        Line::default(),
        Line::from(Span::styled(
            "Enter send   \u{b7}   /help keys   \u{b7}   /model switch   \u{b7}   Ctrl+D quit",
            dim(),
        )),
    ];
    let h = lines.len() as u16;
    let top = area.height.saturating_sub(h) / 2;
    let rect = Rect {
        x: area.x,
        y: area.y + top,
        width: area.width,
        height: h.min(area.height),
    };
    f.render_widget(
        Paragraph::new(Text::from(lines)).alignment(Alignment::Center),
        rect,
    );
}

/// Full-frame farewell drawn just before the terminal is restored on quit.
pub(crate) fn draw_outro(f: &mut Frame) {
    let area = f.area();
    let lines = vec![
        Line::from(Span::styled(
            format!("{BRAND}{BRAND}{BRAND}  TRENTADUE"),
            style(ACCENT, true),
        )),
        Line::from(Span::styled("session ended  \u{b7}  see you soon", dim())),
    ];
    let h = lines.len() as u16;
    let top = area.height.saturating_sub(h) / 2;
    let rect = Rect {
        x: area.x,
        y: area.y + top,
        width: area.width,
        height: h.min(area.height),
    };
    f.render_widget(Clear, area);
    f.render_widget(
        Paragraph::new(Text::from(lines)).alignment(Alignment::Center),
        rect,
    );
}

/// Draw the centered help overlay over the current frame.
fn render_help(f: &mut Frame, area: Rect) {
    let lines = help_lines();
    let w = 56.min(area.width.saturating_sub(4));
    let h = (lines.len() as u16 + 2).min(area.height.saturating_sub(2));
    let popup = centered_rect(w, h, area);
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(style(ACCENT, false))
        .title(Span::styled(" help ", style(ACCENT, true)))
        .padding(Padding::horizontal(1));
    f.render_widget(Clear, popup);
    f.render_widget(Paragraph::new(Text::from(lines)).block(block), popup);
}

/// Draw the native model picker as a centered, scrollable, selectable list.
fn render_picker(f: &mut Frame, area: Rect, p: &ModelPicker) {
    let w = 72.min(area.width.saturating_sub(4));
    let inner_max = (area.height.saturating_sub(6) as usize).max(1);
    let visible = p.rows.len().clamp(1, inner_max);
    let popup = centered_rect(w, visible as u16 + 2, area);

    // Window so the selected row stays in view.
    let start = p.selected.saturating_sub(visible.saturating_sub(1));
    let mut lines: Vec<Line> = Vec::new();
    for (i, row) in p.rows.iter().enumerate().skip(start).take(visible) {
        let selected = i == p.selected;
        let prefix = if selected { "\u{203a} " } else { "  " };
        let dot = if row.active { "\u{25cf} " } else { "  " };
        let row_style = if selected {
            Style::default().fg(Color::Black).bg(OK_COLOR)
        } else if row.active {
            style(OK_COLOR, false)
        } else {
            Style::default()
        };
        lines.push(Line::from(Span::styled(
            format!("{prefix}{dot}{}", row.label),
            row_style,
        )));
    }
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(style(ACCENT, false))
        .title(Span::styled(
            " select model  ·  \u{2191}\u{2193} Enter Esc ",
            style(ACCENT, true),
        ))
        .padding(Padding::horizontal(1));
    f.render_widget(Clear, popup);
    f.render_widget(Paragraph::new(Text::from(lines)).block(block), popup);
}

fn centered_rect(w: u16, h: u16, area: Rect) -> Rect {
    Rect {
        x: area.x + area.width.saturating_sub(w) / 2,
        y: area.y + area.height.saturating_sub(h) / 2,
        width: w.min(area.width),
        height: h.min(area.height),
    }
}

/// Content of the help overlay: keys, TUI commands, and where model/voice live.
fn help_lines() -> Vec<Line<'static>> {
    let key = |k: &'static str, d: &'static str| {
        Line::from(vec![
            Span::styled(format!("  {k:<12}"), style(OK_COLOR, false)),
            Span::styled(d, Style::default()),
        ])
    };
    let head = |t: &'static str| Line::from(Span::styled(t, style(ACCENT, true)));
    vec![
        head("keys"),
        key("Enter", "send message"),
        key("Alt+Enter", "newline"),
        key("Up / Down", "prompt history"),
        key("Ctrl+R", "reverse-search history"),
        key("Ctrl+T", "show / hide thinking"),
        key("PgUp/PgDn", "scroll transcript"),
        key("Esc", "clear input / cancel turn / close"),
        key("Ctrl+C", "cancel reply / clear input"),
        key("Ctrl+D", "quit"),
        Line::default(),
        head("commands"),
        key("/help  ?", "this overlay"),
        key("/clear", "clear the transcript"),
        key("/mode", "calm / inspect / deep (cycle or name)"),
        key("/model", "switch model (opens picker)"),
        key("/local", "toggle local / cloud"),
        key("/think", "toggle thinking"),
        key("/status", "show status"),
        key("/quit", "exit"),
        Line::default(),
        Line::from(Span::styled(
            "  /model, /local, /think, /status briefly drop to a",
            dim(),
        )),
        Line::from(Span::styled(
            "  classic view, then return. /voice not in the TUI yet.",
            dim(),
        )),
    ]
}

/// The TRENTADUE brand mark `▞▞▞`, always stable in the header.
fn brand_mark() -> Vec<Span<'static>> {
    (0..3)
        .map(|_| Span::styled(BRAND, style(ACCENT, true)))
        .collect()
}

/// Render one cell into logical lines, each a list of `(text, style)` segments.
#[cfg(test)]
fn cell_lines(cell: &Cell, mode: Mode, elapsed_s: f32) -> Vec<Vec<(String, Style)>> {
    cell_lines_with_reply_mark(cell, mode, elapsed_s, true)
}

fn cell_lines_with_reply_mark(
    cell: &Cell,
    mode: Mode,
    elapsed_s: f32,
    show_reply_mark: bool,
) -> Vec<Vec<(String, Style)>> {
    match cell {
        Cell::User(text) => text
            .split('\n')
            .enumerate()
            .map(|(i, l)| {
                let head = if i == 0 {
                    (format!("  {DOT} "), style(OK_COLOR, true))
                } else {
                    ("    ".to_string(), Style::default())
                };
                vec![head, (l.to_string(), style(Color::White, true))]
            })
            .collect(),
        Cell::Activity {
            phase,
            prefill,
            prefill_estimate,
            prefill_tps,
            prefill_tps_estimated,
            cache,
        } => {
            let mut segs = vec![
                ("  ".to_string(), Style::default()),
                (activity_marker(elapsed_s), dim_color(ACCENT)),
                (" ".to_string(), Style::default()),
            ];
            match phase {
                ActivityPhase::Prefill => {
                    segs.push(("prefill".to_string(), dim_color(ACCENT)));
                    if let Some(cache) = cache {
                        let (label, color) = cache_status_label(*cache);
                        segs.push((" · ".to_string(), dim()));
                        segs.push((label, dim_color(color)));
                    }
                    if let Some((processed, total)) = prefill {
                        if *total > 0 {
                            segs.push((
                                format!(
                                    " {}% · {}/{}",
                                    processed * 100 / total,
                                    crate::tui::format_tokens(*processed as usize),
                                    crate::tui::format_tokens(*total as usize)
                                ),
                                dim(),
                            ));
                        }
                    } else if let Some(tokens) = prefill_estimate {
                        if *tokens > 0 {
                            let token_label = crate::tui::format_tokens(*tokens as usize);
                            if *prefill_tps_estimated {
                                if let Some(tps) = prefill_tps {
                                    let pct = ((elapsed_s * *tps * 100.0) / *tokens as f32)
                                        .floor()
                                        .clamp(0.0, 99.0)
                                        as u64;
                                    segs.push((format!(" est {pct}% · ~{token_label}"), dim()));
                                } else {
                                    segs.push((format!(" ~{token_label}"), dim()));
                                }
                            } else {
                                segs.push((format!(" ~{token_label}"), dim()));
                            }
                        }
                    }
                    if let Some(tps) = prefill_tps {
                        if *tps > 0.0 {
                            segs.push((" · ".to_string(), dim()));
                            let label = if *prefill_tps_estimated {
                                format!("~{}", format_prefill_rate(*tps))
                            } else {
                                format_prefill_rate(*tps)
                            };
                            segs.push((label, dim_color(ACCENT)));
                        }
                    }
                    segs.push((format!(" · {:.1}s", elapsed_s), dim()));
                }
                ActivityPhase::Thinking => {
                    segs.push(("thinking".to_string(), dim_color(WARN_COLOR)));
                    segs.push((format!(" · {:.1}s", elapsed_s), dim()));
                }
                ActivityPhase::Decoding => {
                    if show_reply_mark {
                        return vec![vec![
                            (format!("  {BRAND}{BRAND}   "), style(ACCENT, true)),
                            (TOOL_BRIDGE_TEXT.to_string(), Style::default()),
                        ]];
                    }
                    segs.push(("decoding".to_string(), dim_color(ACCENT)));
                    segs.push((format!(" · {:.1}s", elapsed_s), dim()));
                }
            }
            vec![segs]
        }
        Cell::Reply(text) => render::markdown(text)
            .into_iter()
            .enumerate()
            .map(|(i, mut segs)| {
                let head = if i == 0 && show_reply_mark {
                    (format!("  {BRAND}{BRAND} "), style(ACCENT, true))
                } else {
                    ("     ".to_string(), Style::default())
                };
                let mut line = vec![head];
                line.append(&mut segs);
                line
            })
            .collect(),
        Cell::Thinking(text) => text
            .split('\n')
            .map(|l| {
                vec![
                    ("     ".to_string(), Style::default()),
                    (l.to_string(), thinking_text_style()),
                ]
            })
            .collect(),
        Cell::Tool {
            name,
            args,
            state,
            summary,
            output,
            preview,
            ms,
            ..
        } => {
            let label = if matches!(mode, Mode::Calm) {
                compact_tool_label(name, args)
            } else {
                format!("{name}({})", clip(args, 48))
            };
            let mut head = vec![
                (format!("       {RUN} "), style(ACCENT, false)),
                (label, style(ACCENT, false)),
            ];
            match state {
                ToolState::Running => {
                    let elapsed = if *ms > 0 {
                        format!("  running · {:.1}s", *ms as f32 / 1000.0)
                    } else {
                        "  running".to_string()
                    };
                    head.push((elapsed, style(WARN_COLOR, false)));
                }
                ToolState::Ok => head.push((format!("  {OK} {ms}ms"), style(OK_COLOR, false))),
                ToolState::Err => head.push((format!("  {ERR} {ms}ms"), style(ERR_COLOR, false))),
            }
            let mut out = vec![head];
            let summary_line = |out: &mut Vec<Vec<(String, Style)>>| {
                if let Some(s) = summary {
                    out.push(vec![
                        (format!("           {TURN} "), dim()),
                        (s.clone(), dim()),
                    ]);
                }
            };
            match mode.tool_output_lines() {
                None => summary_line(&mut out), // calm: one-line summary
                Some(max) => {
                    let lines: Vec<&str> =
                        output.lines().filter(|l| !l.trim().is_empty()).collect();
                    if lines.is_empty() {
                        summary_line(&mut out);
                    } else {
                        let shown = lines.len().min(max);
                        for l in lines.iter().take(shown) {
                            out.push(vec![(format!("           {l}"), dim())]);
                        }
                        if lines.len() > shown {
                            out.push(vec![(
                                format!("           + {} more lines", lines.len() - shown),
                                dim(),
                            )]);
                        }
                    }
                }
            }
            if matches!(state, ToolState::Running) {
                if let Some(p) = preview {
                    if !p.trim().is_empty() {
                        out.push(vec![
                            (format!("           {TURN} "), dim()),
                            (compact_output_line(p), dim()),
                        ]);
                    }
                }
            }
            out
        }
        Cell::Meta {
            elapsed_s,
            ttft_s,
            prefill_tps,
            cache,
            tokens,
            tps,
            estimated,
        } => {
            let mut segs = vec![(format!("  {META} {:.1}s", elapsed_s), dim())];
            if let Some(cache) = cache {
                let (label, color) = cache_status_label(*cache);
                segs.push(("  ".to_string(), dim()));
                segs.push((label, dim_color(color)));
            }
            if mode.verbose_meta() {
                if let Some(t) = ttft_s {
                    segs.push((format!("  ttft {:.2}s", t), dim()));
                }
            }
            let show_prefill = mode.verbose_meta() || ttft_s.map(|t| t >= 1.0).unwrap_or(false);
            if show_prefill {
                if let Some(rate) = prefill_tps {
                    segs.push((format!("  prefill {}", format_prefill_rate(*rate)), dim()));
                }
            }
            if *tokens > 0 {
                let mark = if *estimated { "~" } else { "" };
                segs.push((format!("  {mark}{tokens} tok"), dim()));
                if let Some(tps) = tps {
                    segs.push((format!("  {mark}{tps:.0} tok/s"), dim()));
                }
            }
            vec![segs]
        }
        Cell::Note(text) => vec![vec![
            ("  ".to_string(), Style::default()),
            (text.clone(), style(WARN_COLOR, false)),
        ]],
    }
}

/// Soft-wrap one logical line (given as styled segments) to `width` display
/// columns, pushing one or more rows into `out`. Always pushes at least one row
/// so a blank logical line stays visible.
fn wrap_segments(segs: &[(String, Style)], width: usize, out: &mut Vec<Line<'static>>) {
    let width = width.max(1);
    let indent = continuation_indent(segs, width);
    let mut row: Vec<(char, Style)> = Vec::new();
    let mut w = 0usize;
    for (text, st) in segs {
        for c in text.chars() {
            let cw = UnicodeWidthChar::width(c).unwrap_or(0);
            if w + cw > width && !row.is_empty() {
                out.push(row_to_line(&row));
                row.clear();
                w = push_indent(&mut row, indent);
            }
            row.push((c, *st));
            w += cw;
        }
    }
    out.push(row_to_line(&row));
}

fn continuation_indent(segs: &[(String, Style)], width: usize) -> usize {
    let indent = if segs.len() > 1 {
        let first = &segs[0].0;
        if first.starts_with(' ') && first.chars().any(|c| !c.is_whitespace()) {
            text_width(first)
        } else {
            leading_space_width(segs)
        }
    } else {
        leading_space_width(segs)
    };
    indent.min(width.saturating_sub(1))
}

fn push_indent(row: &mut Vec<(char, Style)>, indent: usize) -> usize {
    row.extend((0..indent).map(|_| (' ', Style::default())));
    indent
}

fn leading_space_width(segs: &[(String, Style)]) -> usize {
    let mut width = 0;
    for (text, _) in segs {
        for c in text.chars() {
            if c == ' ' {
                width += 1;
            } else {
                return width;
            }
        }
    }
    width
}

fn text_width(s: &str) -> usize {
    s.chars()
        .map(|c| UnicodeWidthChar::width(c).unwrap_or(0))
        .sum()
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

/// Rough token estimate (~4 chars/token), used only when the provider omits a
/// usage count so throughput can still be shown (marked `~` in the UI).
fn estimate_tokens(text: &str) -> u64 {
    (text.chars().count() as f32 / 4.0).ceil() as u64
}

/// One-line summary of a tool result, collapsed like the classic renderer.
fn summarize_output(data: &str, ok: bool) -> Option<String> {
    let first = data.lines().find(|l| !l.trim().is_empty())?.trim();
    if !ok {
        return Some(compact_output_line(first));
    }
    if let Some(summary) = summarize_json_output(first) {
        return Some(summary);
    }
    let n = data.lines().filter(|l| !l.trim().is_empty()).count();
    let first = compact_output_line(first);
    Some(format!(
        "{first}  · {n} line{}",
        if n == 1 { "" } else { "s" }
    ))
}

fn summarize_json_output(first: &str) -> Option<String> {
    let value: Value = serde_json::from_str(first).ok()?;
    let obj = value.as_object()?;
    let mut parts = Vec::new();
    if let Some(status) = obj.get("status").and_then(Value::as_i64) {
        parts.push(format!("HTTP {status}"));
    }
    if let Some(extractor) = obj.get("extractor").and_then(Value::as_str) {
        if !extractor.trim().is_empty() {
            parts.push(extractor.trim().to_string());
        }
    }
    if let Some(url) = obj
        .get("finalUrl")
        .or_else(|| obj.get("url"))
        .and_then(Value::as_str)
    {
        let compact = compact_url(url);
        if !compact.is_empty() {
            parts.push(compact);
        }
    }
    if let Some(len) = obj.get("length").and_then(Value::as_u64) {
        parts.push(format!("{} chars", crate::tui::format_tokens(len as usize)));
    } else if let Some(text) = obj.get("text").and_then(Value::as_str) {
        parts.push(format!(
            "{} chars",
            crate::tui::format_tokens(text.chars().count())
        ));
    }
    if parts.is_empty() {
        None
    } else {
        Some(parts.join(" · "))
    }
}

fn compact_output_line(line: &str) -> String {
    let line = line.trim();
    if line.len() <= 120 {
        return line.to_string();
    }
    format!("{}...", line.chars().take(117).collect::<String>())
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
                Cell::Activity { .. } => "activity",
                Cell::Reply(_) => "reply",
                Cell::Thinking(_) => "thinking",
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

    fn flatten_text(text: Text<'static>) -> String {
        text.lines
            .iter()
            .flat_map(|l| l.spans.iter())
            .map(|s| s.content.as_ref())
            .collect()
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
    fn wrap_preserves_visual_gutter_on_continuation_rows() {
        let mut out = Vec::new();
        wrap_segments(
            &[
                (format!("  {BRAND}{BRAND} "), style(ACCENT, true)),
                ("abcdef".to_string(), Style::default()),
            ],
            8,
            &mut out,
        );
        let rows: Vec<String> = out
            .iter()
            .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect())
            .collect();
        assert_eq!(rows, vec!["  ▞▞ abc".to_string(), "     def".to_string()]);
    }

    #[test]
    fn streaming_deltas_interleave_with_tools_chronologically() {
        let mut app = App::new();
        app.begin_turn("hello");
        assert_eq!(kinds(&app), vec!["user", "activity"]);
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
    fn activity_line_tracks_prefill_under_user_then_yields_to_answer() {
        let mut app = App::new();
        app.begin_turn("q");
        app.on_delta("\u{0}prefill:5/10");
        assert_eq!(kinds(&app), vec!["user", "activity"]);
        match app.transcript.get(1) {
            Some(Cell::Activity {
                phase: ActivityPhase::Prefill,
                prefill: Some((5, 10)),
                ..
            }) => {}
            _ => panic!("expected prefill activity under user turn"),
        }

        app.on_delta("answer");
        assert_eq!(kinds(&app), vec!["user", "reply"]);
        assert_eq!(reply_text(&app), "answer");
    }

    #[test]
    fn activity_line_reports_effective_prefill_rate() {
        let mut app = App::new();
        app.begin_turn("q");
        app.prefill_started = Some(Instant::now() - std::time::Duration::from_secs(1));
        app.on_delta("\u{0}prefill:1200/2400");

        match app.transcript.get(1) {
            Some(Cell::Activity {
                prefill_tps: Some(tps),
                ..
            }) => assert!(*tps > 900.0, "expected visible prefill rate, got {tps}"),
            _ => panic!("expected prefill activity with throughput"),
        }
        let rendered = flatten_text(Text::from(app.transcript_rows(100)));
        assert!(
            rendered.contains("tok/s"),
            "prefill throughput should render in activity row: {rendered}"
        );
    }

    #[test]
    fn activity_line_estimates_prefill_when_server_has_no_progress() {
        let mut app = App::new();
        app.prefill_rate_hint = Some(250.0);
        app.begin_turn("q");
        app.turn_start = Some(Instant::now() - std::time::Duration::from_secs(2));
        app.on_delta("\u{0}prefill_estimate:1000");

        match app.transcript.get(1) {
            Some(Cell::Activity {
                prefill_estimate: Some(1000),
                prefill_tps: Some(tps),
                prefill_tps_estimated: true,
                ..
            }) => assert!(
                (*tps - 250.0).abs() < 1.0,
                "expected hinted prefill rate, got {tps}"
            ),
            _ => panic!("expected estimated prefill activity"),
        }

        let rendered = flatten_text(Text::from(app.transcript_rows(100)));
        assert!(
            rendered.contains("est 50%"),
            "missing estimated percent: {rendered}"
        );
        assert!(
            rendered.contains("~250 tok/s"),
            "estimated rate should be marked approximate: {rendered}"
        );
    }

    #[test]
    fn thinking_deltas_render_as_separate_block_not_dropped() {
        let mut app = App::new();
        app.begin_turn("q");
        app.on_delta("\x1b[90m\x1b[2m");
        // The thinking ANSI markers are control-only (never rendered verbatim);
        // the reasoning text between them lands in a separate Thinking block.
        app.on_delta("private thought");
        assert_eq!(kinds(&app), vec!["user", "thinking"]);
        match app.transcript.get(1) {
            Some(Cell::Thinking(t)) => assert_eq!(t, "private thought"),
            _ => panic!("expected thinking cell"),
        }
        app.on_delta("\x1b[0m\n\n");
        app.on_delta("visible");
        // Reasoning stays in its own cell; the answer is a separate reply.
        assert_eq!(kinds(&app), vec!["user", "thinking", "reply"]);
        assert_eq!(reply_text(&app), "visible");
    }

    #[test]
    fn thinking_block_strips_raw_think_markers_and_label() {
        let mut app = App::new();
        app.begin_turn("q");
        app.on_delta("\x1b[90m\x1b[2m");
        app.on_delta("thinking <think>The user says hi");
        app.on_delta("</think>");
        app.on_delta("\x1b[0m\n\n");

        match app.transcript.get(1) {
            Some(Cell::Thinking(t)) => assert_eq!(t, "The user says hi"),
            _ => panic!("expected thinking cell"),
        }

        let rendered = flatten_text(Text::from(app.transcript_rows(100)));
        assert!(rendered.contains("The user says hi"));
        assert!(!rendered.contains("thinking <think>"));
        assert!(!rendered.contains("<think>"));
        assert!(!rendered.contains("</think>"));
    }

    #[test]
    fn thinking_block_renders_readable_grey() {
        use ratatui::style::{Color, Modifier};
        let mut app = App::new();
        app.begin_turn("q");
        app.on_delta("\x1b[90m\x1b[2m");
        app.on_delta("my secret reasoning");
        app.on_delta("\x1b[0m\n\n");
        app.on_delta("the answer");

        let rows = app.transcript_rows(100);
        let spans: Vec<_> = rows.iter().flat_map(|l| l.spans.iter()).collect();

        // The reasoning is muted but readable, not the old dim dark grey.
        let thinking = spans
            .iter()
            .find(|s| s.content.contains("my secret reasoning"))
            .expect("thinking text rendered");
        assert_eq!(
            thinking.style.fg,
            Some(Color::Gray),
            "thinking is readable grey"
        );
        assert!(
            !thinking.style.add_modifier.contains(Modifier::DIM),
            "thinking should not be dimmed"
        );
        assert!(
            spans.iter().all(|s| s.content != "  thinking "),
            "thinking label is hidden"
        );

        // The answer is not greyed out.
        let answer = spans
            .iter()
            .find(|s| s.content.contains("the answer"))
            .expect("answer rendered");
        assert_ne!(answer.style.fg, Some(Color::Gray), "answer is not grey");
    }

    #[test]
    fn ctrl_t_hides_and_shows_thinking_without_dropping_it() {
        let mut app = App::new();
        app.begin_turn("q");
        app.on_delta("\x1b[90m\x1b[2m");
        app.on_delta("private thought");
        app.on_delta("\x1b[0m\n\n");
        app.on_delta("visible answer");

        assert!(flatten_text(Text::from(app.transcript_rows(100))).contains("private thought"));

        let action = app.on_idle_event(Event::Key(KeyEvent::new(
            KeyCode::Char('t'),
            KeyModifiers::CONTROL,
        )));
        assert!(matches!(action, Action::Continue));
        assert_eq!(kinds(&app), vec!["user", "thinking", "reply"]);

        let hidden = flatten_text(Text::from(app.transcript_rows(100)));
        assert!(!hidden.contains("private thought"));
        assert!(hidden.contains("visible answer"));

        let action = app.on_idle_event(Event::Key(KeyEvent::new(
            KeyCode::Char('t'),
            KeyModifiers::CONTROL,
        )));
        assert!(matches!(action, Action::Continue));
        let shown = flatten_text(Text::from(app.transcript_rows(100)));
        assert!(shown.contains("private thought"));
    }

    #[test]
    fn ctrl_t_toggles_thinking_while_streaming() {
        let mut app = App::new();
        app.begin_turn("q");
        app.on_delta("\x1b[90m\x1b[2m");
        app.on_delta("live thought");

        let action = app.on_streaming_event(Event::Key(KeyEvent::new(
            KeyCode::Char('t'),
            KeyModifiers::CONTROL,
        )));
        assert!(matches!(action, StreamingAction::Continue));
        assert!(!flatten_text(Text::from(app.transcript_rows(100))).contains("live thought"));

        let action = app.on_streaming_event(Event::Key(KeyEvent::new(
            KeyCode::Char('t'),
            KeyModifiers::CONTROL,
        )));
        assert!(matches!(action, StreamingAction::Continue));
        assert!(flatten_text(Text::from(app.transcript_rows(100))).contains("live thought"));
    }

    #[test]
    fn reply_after_thinking_trims_leading_blank_gap() {
        let mut app = App::new();
        app.begin_turn("q");
        app.on_delta("\x1b[90m\x1b[2m");
        app.on_delta("private thought");
        app.on_delta("\x1b[0m\n\n");
        app.on_delta("\n\nMy name is nanobot.");

        assert_eq!(reply_text(&app), "My name is nanobot.");

        let rows: Vec<String> = app
            .transcript_rows(120)
            .iter()
            .map(|line| line.spans.iter().map(|s| s.content.as_ref()).collect())
            .collect();
        let rendered = rows.join("\n");
        assert!(
            rows.iter()
                .any(|line| line.starts_with("  ▞▞ My name is nanobot.")),
            "answer should share the brand marker row:\n{rendered}"
        );
        assert!(
            !rows
                .iter()
                .any(|line| line.trim() == format!("{BRAND}{BRAND}")),
            "brand marker must not render on an empty gap row:\n{rendered}"
        );
    }

    #[test]
    fn reply_after_thinking_ignores_split_blank_gap_delta() {
        let mut app = App::new();
        app.begin_turn("q");
        app.on_delta("\x1b[90m\x1b[2m");
        app.on_delta("private thought");
        app.on_delta("\x1b[0m\n\n");
        app.on_delta("\n\n");
        app.on_delta("My name is nanobot.");

        assert_eq!(reply_text(&app), "My name is nanobot.");
        assert_eq!(kinds(&app), vec!["user", "thinking", "reply"]);
    }

    #[test]
    fn thinking_only_stream_does_not_backfill_reasoning_as_reply() {
        let mut app = App::new();
        app.begin_turn("q");
        app.on_delta("\x1b[90m\x1b[2m");
        app.on_delta("private thought");
        app.on_delta("\x1b[0m\n\n");

        app.finish_turn("private thought".to_string());

        assert_eq!(kinds(&app), vec!["user", "thinking", "meta"]);
        assert_eq!(
            reply_text(&app),
            "",
            "reasoning fallback content must not be duplicated as visible reply text"
        );
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
    fn tool_first_turn_keeps_activity_before_tool() {
        let mut app = App::new();
        app.begin_turn("search");
        app.on_delta("\u{0}prefill_estimate:400");
        app.on_tool_event(ToolEvent::CallStart {
            tool_name: "web_search".into(),
            tool_call_id: "c1".into(),
            arguments_preview: "latest news".into(),
        });

        assert_eq!(kinds(&app), vec!["user", "activity", "tool"]);
        assert!(matches!(
            app.transcript.get(1),
            Some(Cell::Activity {
                phase: ActivityPhase::Decoding,
                ..
            })
        ));
    }

    #[test]
    fn tool_after_reply_drops_stale_decoding_activity() {
        let mut app = App::new();
        app.begin_turn("inspect");
        app.on_delta("Let me look.");
        app.on_tool_event(ToolEvent::CallStart {
            tool_name: "list_dir".into(),
            tool_call_id: "c1".into(),
            arguments_preview: "src".into(),
        });
        app.on_tool_event(ToolEvent::CallEnd {
            tool_name: "list_dir".into(),
            tool_call_id: "c1".into(),
            result_data: "providers\nrepl".into(),
            ok: true,
            duration_ms: 1,
        });
        assert_eq!(kinds(&app), vec!["user", "reply", "tool", "activity"]);

        app.on_tool_event(ToolEvent::CallStart {
            tool_name: "read_file".into(),
            tool_call_id: "c2".into(),
            arguments_preview: "src/providers/mod.rs".into(),
        });

        assert_eq!(kinds(&app), vec!["user", "reply", "tool", "tool"]);
        let rendered = flatten_text(Text::from(app.transcript_rows(100)));
        assert!(
            !rendered.contains("decoding"),
            "stale decoding placeholder leaked before tool:\n{rendered}"
        );
    }

    #[test]
    fn forced_tool_recovery_retracts_streamed_prose_before_tool() {
        let mut app = App::new();
        app.begin_turn("fetch this");
        app.on_delta("I can use the `web_fetch` tool to get that URL.");

        app.on_delta("\u{0}retract_reply");

        assert_eq!(kinds(&app), vec!["user"]);
        assert_eq!(reply_text(&app), "");

        app.on_tool_event(ToolEvent::CallStart {
            tool_name: "web_fetch".into(),
            tool_call_id: "c1".into(),
            arguments_preview: "https://example.com".into(),
        });

        assert_eq!(kinds(&app), vec!["user", "tool"]);
        let rendered = flatten_text(Text::from(app.transcript_rows(100)));
        assert!(
            !rendered.contains("I can use the `web_fetch` tool"),
            "recovered tool call should remove streamed tool narration:\n{rendered}"
        );
        assert!(
            rendered.contains("web_fetch"),
            "tool call should remain visible after retraction:\n{rendered}"
        );
    }

    #[test]
    fn second_tool_call_drops_decoding_but_keeps_first_tool_bridge() {
        let mut app = App::new();
        app.begin_turn("search");
        app.on_delta("\u{0}prefill_estimate:400");
        app.on_tool_event(ToolEvent::CallStart {
            tool_name: "web_search".into(),
            tool_call_id: "c1".into(),
            arguments_preview: "nanobot".into(),
        });
        app.on_tool_event(ToolEvent::CallEnd {
            tool_name: "web_search".into(),
            tool_call_id: "c1".into(),
            result_data: "one\ntwo".into(),
            ok: true,
            duration_ms: 1,
        });
        assert_eq!(kinds(&app), vec!["user", "activity", "tool", "activity"]);

        app.on_tool_event(ToolEvent::CallStart {
            tool_name: "web_fetch".into(),
            tool_call_id: "c2".into(),
            arguments_preview: "https://example.com".into(),
        });

        assert_eq!(kinds(&app), vec!["user", "activity", "tool", "tool"]);
        let rendered = flatten_text(Text::from(app.transcript_rows(100)));
        assert!(
            rendered.contains(TOOL_BRIDGE_TEXT),
            "first tool bridge disappeared:\n{rendered}"
        );
        assert_eq!(
            rendered.matches("decoding").count(),
            0,
            "stale decoding placeholder leaked before second tool:\n{rendered}"
        );
    }

    #[test]
    fn tool_first_turn_renders_bridge_before_tool_and_reply() {
        let mut app = App::new();
        app.begin_turn("search");
        app.on_delta("\u{0}prefill_estimate:400");
        app.on_tool_event(ToolEvent::CallStart {
            tool_name: "web_search".into(),
            tool_call_id: "c1".into(),
            arguments_preview: r#"{"query":"history of computing from abacus to quantum"}"#.into(),
        });
        app.on_tool_event(ToolEvent::CallEnd {
            tool_name: "web_search".into(),
            tool_call_id: "c1".into(),
            result_data: "Results for: history of computing from abacus to quantum\none\ntwo"
                .into(),
            ok: true,
            duration_ms: 1037,
        });
        app.on_delta("Wow, here's the history.");

        let rows: Vec<String> = app
            .transcript_rows(120)
            .into_iter()
            .map(|line| {
                line.spans
                    .iter()
                    .map(|span| span.content.as_ref())
                    .collect()
            })
            .collect();
        let bridge_idx = rows
            .iter()
            .position(|line| line.contains(TOOL_BRIDGE_TEXT))
            .expect("bridge rendered");
        let tool_idx = rows
            .iter()
            .position(|line| line.contains("web_search history of computing"))
            .expect("tool rendered");
        let reply_idx = rows
            .iter()
            .position(|line| line.contains("Wow, here's the history."))
            .expect("reply rendered");

        assert!(
            bridge_idx < tool_idx && tool_idx < reply_idx,
            "expected bridge, then tool, then reply:\n{}",
            rows.join("\n")
        );
        assert_eq!(
            rows.iter().filter(|line| line.contains("▞▞")).count(),
            1,
            "tool-first assistant turn should use one marker:\n{}",
            rows.join("\n")
        );
    }

    #[test]
    fn decoding_continuation_after_tool_does_not_reuse_reply_mark() {
        let mut app = App::new();
        app.begin_turn("search");
        app.on_tool_event(ToolEvent::CallStart {
            tool_name: "web_search".into(),
            tool_call_id: "c1".into(),
            arguments_preview: r#"{"query":"history of computing"}"#.into(),
        });
        app.on_tool_event(ToolEvent::CallEnd {
            tool_name: "web_search".into(),
            tool_call_id: "c1".into(),
            result_data: "Results for: history of computing".into(),
            ok: true,
            duration_ms: 10,
        });

        let rows: Vec<String> = app
            .transcript_rows(120)
            .into_iter()
            .map(|line| {
                line.spans
                    .iter()
                    .map(|span| span.content.as_ref())
                    .collect()
            })
            .collect();
        assert_eq!(
            rows.iter().filter(|line| line.contains("▞▞")).count(),
            1,
            "continuation activity should not claim a second marker:\n{}",
            rows.join("\n")
        );
    }

    #[test]
    fn tool_progress_updates_running_row_preview() {
        let mut app = App::new();
        app.begin_turn("go");
        app.on_tool_event(ToolEvent::CallStart {
            tool_name: "exec".into(),
            tool_call_id: "c1".into(),
            arguments_preview: "cargo test".into(),
        });
        app.on_tool_event(ToolEvent::Progress {
            tool_name: "exec".into(),
            tool_call_id: "c1".into(),
            elapsed_ms: 2100,
            output_preview: Some("compiling nanobot-rs".into()),
        });
        let tool = app
            .transcript
            .iter()
            .find(|c| matches!(c, Cell::Tool { .. }))
            .unwrap();
        let flat: String = cell_lines(tool, Mode::Calm, 2.1)
            .iter()
            .flatten()
            .map(|(t, _)| t.clone())
            .collect();
        assert!(flat.contains("running"));
        assert!(flat.contains("2.1s"));
        assert!(flat.contains("compiling nanobot-rs"));
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
        app.on_delta("\u{0}cache:append:2:9");
        app.on_delta("\u{0}prefill:5/10");
        assert_eq!(app.prefill, Some((5, 10)));
        assert_eq!(kinds(&app), vec!["user", "activity"]);
        match app.transcript.get(1) {
            Some(Cell::Activity {
                cache:
                    Some(CacheStatus::AppendOnly {
                        added: 2,
                        messages: 9,
                    }),
                ..
            }) => {}
            _ => panic!("expected cache status on activity row"),
        }
        let rendered = flatten_text(Text::from(app.transcript_rows(80)));
        assert!(
            rendered.contains("cache warm"),
            "cache status should render in activity row: {rendered}"
        );
        app.on_delta("real");
        assert_eq!(reply_text(&app), "real");
        assert_eq!(app.prefill, None);
    }

    #[test]
    fn cache_status_survives_activity_row_into_meta() {
        let mut app = App::new();
        app.begin_turn("q");
        app.on_delta("\u{0}cache:append:2:9");
        app.on_delta("real");
        app.finish_turn(String::new());

        match app.transcript.last() {
            Some(Cell::Meta {
                cache:
                    Some(CacheStatus::AppendOnly {
                        added: 2,
                        messages: 9,
                    }),
                ..
            }) => {}
            _ => panic!("expected cache status in meta"),
        }
        let rendered: String = cell_lines(app.transcript.last().unwrap(), Mode::Calm, 1.0)
            .iter()
            .flatten()
            .map(|(t, _)| t.clone())
            .collect();
        assert!(
            rendered.contains("cache warm"),
            "cache verdict should remain visible after turn: {rendered}"
        );
    }

    #[test]
    fn explicit_trim_reset_outranks_following_divergence_marker() {
        let mut app = App::new();
        app.begin_turn("q");
        app.on_delta("\u{0}cache:reset:trim");
        app.on_delta("\u{0}cache:diverged:1:5:4:0");
        app.on_delta("real");
        app.finish_turn(String::new());

        match app.transcript.last() {
            Some(Cell::Meta {
                cache:
                    Some(CacheStatus::Reset {
                        reason: CacheResetReason::Trim,
                    }),
                ..
            }) => {}
            _ => panic!("expected explicit trim reset in meta"),
        }
        let rendered: String = cell_lines(app.transcript.last().unwrap(), Mode::Calm, 1.0)
            .iter()
            .flatten()
            .map(|(t, _)| t.clone())
            .collect();
        assert!(
            rendered.contains("cache reset · trim"),
            "trim reset reason should stay visible: {rendered}"
        );
        assert!(
            !rendered.contains("cache reset · msg"),
            "generic divergence should not replace explicit trim cause: {rendered}"
        );
    }

    #[test]
    fn tool_loop_cache_marker_reenters_prefill_activity() {
        let mut app = App::new();
        app.begin_turn("q");
        app.on_delta("checking");
        app.on_tool_event(ToolEvent::CallStart {
            tool_name: "exec".into(),
            tool_call_id: "c1".into(),
            arguments_preview: "cmd=true".into(),
        });
        app.on_tool_event(ToolEvent::CallEnd {
            tool_name: "exec".into(),
            tool_call_id: "c1".into(),
            result_data: "ok".into(),
            ok: true,
            duration_ms: 12,
        });

        assert!(matches!(
            app.transcript.last(),
            Some(Cell::Activity {
                phase: ActivityPhase::Decoding,
                ..
            })
        ));

        app.on_delta("\u{0}cache:append:2:5");
        app.on_delta("\u{0}prefill:50/100");

        match app.transcript.last() {
            Some(Cell::Activity {
                phase: ActivityPhase::Prefill,
                prefill: Some((50, 100)),
                cache: Some(CacheStatus::AppendOnly { added: 2, .. }),
                ..
            }) => {}
            _ => panic!("expected second LLM call to show prefill after tool"),
        }
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
        assert!(
            text.contains("bonsai-8b-mlx"),
            "footer model missing:\n{text}"
        );
        assert!(text.contains("ctx"), "footer ctx missing:\n{text}");
    }

    #[test]
    fn short_transcript_bottom_anchors_near_input() {
        use ratatui::backend::TestBackend;
        use ratatui::Terminal;

        let mut app = App::new();
        app.begin_turn("hello");
        app.on_delta("short reply");
        let mut term = Terminal::new(TestBackend::new(80, 16)).unwrap();
        term.draw(|f| app.draw(f, &test_footer())).unwrap();
        let text = buffer_text(term.backend().buffer());
        let first_content = text
            .lines()
            .position(|l| l.contains("hello"))
            .expect("user turn rendered");
        assert!(
            first_content > 4,
            "short transcripts should use blank space above, not below:\n{text}"
        );
    }

    #[test]
    fn streaming_transcript_reserves_bottom_spacer() {
        let mut app = App::new();
        app.begin_turn("hello");
        app.on_delta("\u{0}prefill_estimate:100");
        let rows = app.transcript_rows(80);
        assert!(
            rows.last().map(|l| l.spans.is_empty()).unwrap_or(false),
            "streaming transcript should leave a blank row above input"
        );
    }

    #[test]
    fn streaming_header_is_stable_and_timerless() {
        let mut app = App::new();
        app.begin_turn("hello");
        app.on_delta("\u{0}prefill:5/10");

        let first = flatten_text(app.header_lines(80));
        app.tick(1.0);
        let second = flatten_text(app.header_lines(80));

        assert_eq!(first, second, "streaming header must not animate");
        assert!(
            first.contains("working"),
            "streaming state missing: {first}"
        );
        assert!(!first.contains("5/10"), "prefill belongs in the turn row");
        assert!(!first.contains("0."), "timer belongs in the turn row");
    }

    #[test]
    fn assistant_reply_uses_two_glyph_prefix() {
        let cell = Cell::Reply("hello".into());
        let flat: String = cell_lines(&cell, Mode::Calm, 0.0)
            .iter()
            .flatten()
            .map(|(t, _)| t.clone())
            .collect();
        assert!(
            flat.starts_with("  ▞▞ hello"),
            "reply marker too large: {flat}"
        );
        assert!(
            !flat.starts_with("  ▞▞▞"),
            "reply marker should be two glyphs"
        );
    }

    #[test]
    fn assistant_turn_uses_one_brand_across_tool_continuation() {
        let mut app = App::new();
        app.begin_turn("inspect");
        app.on_delta("Let me look.");
        app.on_tool_event(ToolEvent::CallStart {
            tool_name: "read_file".into(),
            tool_call_id: "c1".into(),
            arguments_preview: r#"{"path":"render.rs"}"#.into(),
        });
        app.on_tool_event(ToolEvent::CallEnd {
            tool_name: "read_file".into(),
            tool_call_id: "c1".into(),
            result_data: "one\ntwo".into(),
            ok: true,
            duration_ms: 1,
        });
        app.on_delta("Now I know.");

        let rows: Vec<String> = app
            .transcript_rows(120)
            .iter()
            .map(|line| line.spans.iter().map(|s| s.content.as_ref()).collect())
            .collect();
        let joined = rows.join("\n");
        assert_eq!(
            joined.matches("▞▞").count(),
            1,
            "assistant turn should have one brand mark:\n{joined}"
        );
        let tool_idx = rows
            .iter()
            .position(|line| line.contains("read_file render.rs"))
            .expect("tool rendered");
        let continuation_idx = rows
            .iter()
            .position(|line| line.contains("Now I know."))
            .expect("continuation rendered");
        assert!(
            continuation_idx > tool_idx + 1,
            "continuation should have a blank row after tool:\n{joined}"
        );
        assert!(
            rows[continuation_idx].starts_with("     Now I know."),
            "continuation should be indented without a new brand mark:\n{}",
            rows[continuation_idx]
        );
    }

    #[test]
    fn calm_tool_rows_use_compact_argument_hints() {
        let cell = Cell::Tool {
            id: "c1".into(),
            name: "web_fetch".into(),
            args: r#"{"url":"https://hacker-news.firebaseio.com/v0/topstories.json"}"#.into(),
            state: ToolState::Ok,
            summary: Some("1 line".into()),
            output: String::new(),
            preview: None,
            ms: 517,
        };
        let flat: String = cell_lines(&cell, Mode::Calm, 0.0)
            .iter()
            .flatten()
            .map(|(t, _)| t.clone())
            .collect();
        assert!(flat.contains("web_fetch hacker-news.firebaseio.com"));
        assert!(!flat.contains(r#"{"url""#), "calm mode leaked JSON: {flat}");
    }

    #[test]
    fn calm_tool_summary_collapses_json_output() {
        let mut app = App::new();
        app.begin_turn("fetch");
        app.on_tool_event(ToolEvent::CallStart {
            tool_name: "web_fetch".into(),
            tool_call_id: "c1".into(),
            arguments_preview: r#"{"url":"https://www.bbc.com/news/world"}"#.into(),
        });
        app.on_tool_event(ToolEvent::CallEnd {
            tool_name: "web_fetch".into(),
            tool_call_id: "c1".into(),
            result_data: r##"{"extractor":"readability","finalUrl":"https://www.bbc.com/news/world","length":3000,"status":200,"text":"# World | Latest News"}"##.into(),
            ok: true,
            duration_ms: 528,
        });
        let tool = app
            .transcript
            .iter()
            .find(|c| matches!(c, Cell::Tool { .. }))
            .unwrap();
        let flat: String = cell_lines(tool, Mode::Calm, 0.0)
            .iter()
            .flatten()
            .map(|(t, _)| t.clone())
            .collect();

        assert!(flat.contains("HTTP 200"), "summary missing status: {flat}");
        assert!(
            flat.contains("readability") && flat.contains("www.bbc.com"),
            "summary missing fetch identity: {flat}"
        );
        assert!(
            !flat.contains(r#""text":"#),
            "calm mode leaked raw fetch JSON: {flat}"
        );
    }

    #[test]
    fn tool_hints_preserve_long_text_without_ellipsis() {
        let command = "cd ~/Dev/nanobot-rs && git log --format=%H%x20%s --decorate --stat";
        let cell = Cell::Tool {
            id: "c1".into(),
            name: "exec".into(),
            args: format!(r#"{{"command":"{command}"}}"#),
            state: ToolState::Ok,
            summary: Some("done".into()),
            output: String::new(),
            preview: None,
            ms: 30,
        };
        let flat: String = cell_lines(&cell, Mode::Calm, 0.0)
            .iter()
            .flatten()
            .map(|(t, _)| t.clone())
            .collect();

        assert!(flat.contains(command), "tool command was hidden: {flat}");
        assert!(
            !flat.contains('\u{2026}'),
            "tool hint should not ellipsize: {flat}"
        );
    }

    #[test]
    fn tool_blocks_are_indented_after_assistant_text() {
        use ratatui::backend::TestBackend;
        use ratatui::Terminal;

        let mut app = App::new();
        app.begin_turn("inspect");
        app.on_delta("Let me look.");
        app.on_tool_event(ToolEvent::CallStart {
            tool_name: "read_file".into(),
            tool_call_id: "c1".into(),
            arguments_preview: r#"{"path":"render.rs"}"#.into(),
        });
        app.on_tool_event(ToolEvent::CallEnd {
            tool_name: "read_file".into(),
            tool_call_id: "c1".into(),
            result_data: "# file\none\ntwo".into(),
            ok: true,
            duration_ms: 1,
        });

        let mut term = Terminal::new(TestBackend::new(90, 16)).unwrap();
        term.draw(|f| app.draw(f, &test_footer())).unwrap();
        let lines: Vec<String> = buffer_text(term.backend().buffer())
            .lines()
            .map(|line| line.to_string())
            .collect();
        let reply_idx = lines
            .iter()
            .position(|line| line.contains("Let me look."))
            .expect("reply rendered");
        let tool_idx = lines
            .iter()
            .position(|line| line.contains("read_file render.rs"))
            .expect("tool rendered");
        assert!(
            tool_idx > reply_idx + 1,
            "tool should have a blank row after reply:\n{}",
            lines.join("\n")
        );
        assert!(
            lines[tool_idx].starts_with("       ▶"),
            "tool should be indented:\n{}",
            lines[tool_idx]
        );
    }

    #[test]
    fn mouse_wheel_scrolls_transcript_in_idle_and_streaming() {
        use ratatui::backend::TestBackend;
        use ratatui::crossterm::event::{MouseEvent, MouseEventKind};
        use ratatui::Terminal;

        let mut app = App::new();
        for i in 0..30 {
            app.transcript.push(Cell::User(format!("turn {i}")));
        }
        let mut term = Terminal::new(TestBackend::new(80, 12)).unwrap();
        term.draw(|f| app.draw(f, &test_footer())).unwrap();
        assert!(app.max_scroll > 0, "transcript should be scrollable");

        let wheel_up = Event::Mouse(MouseEvent {
            kind: MouseEventKind::ScrollUp,
            column: 0,
            row: 0,
            modifiers: KeyModifiers::NONE,
        });
        let _ = app.on_idle_event(wheel_up.clone());
        assert!(app.scroll_from_bottom > 0, "idle wheel should scroll up");
        let before = app.scroll_from_bottom;
        assert!(matches!(
            app.on_streaming_event(wheel_up),
            StreamingAction::Continue
        ));
        assert!(
            app.scroll_from_bottom > before,
            "streaming wheel should keep scrolling"
        );
    }

    #[test]
    fn streaming_input_accepts_typeahead() {
        let mut app = App::new();
        app.begin_turn("long task");

        let action = app.on_streaming_event(Event::Key(KeyEvent::new(
            KeyCode::Char('n'),
            KeyModifiers::NONE,
        )));

        assert!(matches!(action, StreamingAction::Continue));
        assert_eq!(input_text(&app), "n");
    }

    #[test]
    fn streaming_enter_with_text_cancels_and_submits_draft() {
        let mut app = App::new();
        app.begin_turn("long task");
        app.input.insert_str("actually do this");

        match app.on_streaming_event(Event::Key(KeyEvent::new(
            KeyCode::Enter,
            KeyModifiers::NONE,
        ))) {
            StreamingAction::CancelAndSubmit(turn) => {
                assert_eq!(turn.text, "actually do this");
                assert!(turn.media.is_empty());
            }
            _ => panic!("typed Enter should interrupt and submit draft"),
        }
        assert_eq!(input_text(&app), "");
    }

    #[test]
    fn streaming_enter_empty_cancels_without_submit() {
        let mut app = App::new();
        app.begin_turn("long task");

        assert!(matches!(
            app.on_streaming_event(Event::Key(KeyEvent::new(
                KeyCode::Enter,
                KeyModifiers::NONE,
            ))),
            StreamingAction::Cancel
        ));
    }

    #[test]
    fn streaming_escape_cancels_and_clears_draft() {
        let mut app = App::new();
        app.begin_turn("long task");
        app.input.insert_str("draft");

        assert!(matches!(
            app.on_streaming_event(Event::Key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))),
            StreamingAction::Cancel
        ));
        assert_eq!(input_text(&app), "");
    }

    #[test]
    fn submit_extracts_local_image_references() {
        let dir = tempfile::tempdir().unwrap();
        let image = dir.path().join("shot.png");
        std::fs::write(&image, b"png").unwrap();

        let mut app = App::new();
        app.input
            .insert_str(&format!("please inspect ![screen]({})", image.display()));
        match app.submit() {
            Action::Submit(turn) => {
                assert_eq!(turn.text, "please inspect");
                let expected = std::fs::canonicalize(&image).unwrap();
                assert_eq!(turn.media, vec![expected.to_string_lossy().to_string()]);
                assert!(turn.display_text().contains("[image: shot.png]"));
            }
            _ => panic!("expected submit"),
        }
    }

    #[test]
    fn empty_transcript_shows_welcome_hint() {
        use ratatui::backend::TestBackend;
        use ratatui::Terminal;

        let mut app = App::new();
        let mut term = Terminal::new(TestBackend::new(80, 12)).unwrap();
        term.draw(|f| app.draw(f, &test_footer())).unwrap();
        let text = buffer_text(term.backend().buffer());
        assert!(text.contains("TRENTADUE"), "intro brand missing");
        assert!(text.contains("/help"), "intro hints missing");
    }

    #[test]
    fn streaming_input_border_keeps_interrupt_hint() {
        use ratatui::backend::TestBackend;
        use ratatui::Terminal;

        let mut app = App::new();
        app.begin_turn("long task");
        let mut term = Terminal::new(TestBackend::new(100, 12)).unwrap();
        term.draw(|f| app.draw(f, &test_footer())).unwrap();

        let text = buffer_text(term.backend().buffer());
        assert!(
            text.contains("Type and press Enter or ESC to interrupt"),
            "streaming input border hint missing:\n{text}"
        );
    }

    #[test]
    fn scroll_up_clamps_and_returns_to_bottom() {
        use ratatui::backend::TestBackend;
        use ratatui::Terminal;

        let mut app = App::new();
        for i in 0..40 {
            app.begin_turn(&format!("q{i}"));
            app.on_delta(&format!("reply {i}"));
            app.finish_turn(String::new());
        }
        let mut term = Terminal::new(TestBackend::new(40, 12)).unwrap();
        term.draw(|f| app.draw(f, &test_footer())).unwrap();
        assert!(app.max_scroll > 0, "tall transcript should be scrollable");

        // Scroll far past the top: must clamp, never overshoot.
        for _ in 0..100 {
            app.scroll_up(10);
        }
        assert_eq!(app.scroll_from_bottom, app.max_scroll, "clamped at the top");

        // Paging back down reaches the bottom (the reported bug).
        app.scroll_down(app.max_scroll);
        assert_eq!(app.scroll_from_bottom, 0, "returns to bottom");
    }

    #[test]
    fn help_overlay_lists_keys_commands_and_model_path() {
        use ratatui::backend::TestBackend;
        use ratatui::Terminal;

        let mut app = App::new();
        app.set_help(true);
        let mut term = Terminal::new(TestBackend::new(70, 24)).unwrap();
        term.draw(|f| app.draw(f, &test_footer())).unwrap();
        let text = buffer_text(term.backend().buffer());
        assert!(text.contains("keys"), "help keys missing:\n{text}");
        assert!(text.contains("/clear"), "help commands missing:\n{text}");
        assert!(text.contains("/model"), "model command missing:\n{text}");
    }

    #[test]
    fn clear_transcript_empties_history() {
        let mut app = App::new();
        app.begin_turn("hi");
        app.on_delta("yo");
        app.finish_turn(String::new());
        assert!(!app.transcript.is_empty());
        app.clear_transcript();
        assert!(app.transcript.is_empty());
    }

    #[test]
    fn meta_estimates_tokens_and_throughput_when_unreported() {
        let mut app = App::new();
        app.begin_turn("q");
        app.on_delta("hello world this is a reply"); // no token marker sent
        app.finish_turn(String::new());
        match app.transcript.last() {
            Some(Cell::Meta {
                tokens,
                tps,
                estimated,
                ..
            }) => {
                assert!(*estimated, "must be flagged estimated");
                assert!(*tokens > 0, "estimated token count > 0");
                assert!(tps.is_some(), "throughput computed from estimate");
            }
            other => panic!("expected meta, got {:?}", other.map(|_| ())),
        }
    }

    #[test]
    fn meta_uses_reported_tokens_when_present() {
        let mut app = App::new();
        app.begin_turn("q");
        app.on_delta("hi");
        app.on_delta("\u{0}tokens:123");
        app.finish_turn(String::new());
        match app.transcript.last() {
            Some(Cell::Meta {
                tokens, estimated, ..
            }) => {
                assert_eq!(*tokens, 123);
                assert!(!*estimated, "reported tokens are exact, not estimated");
            }
            _ => panic!("expected meta"),
        }
    }

    #[test]
    fn meta_reports_effective_prefill_rate_from_prompt_tokens() {
        let mut app = App::new();
        app.begin_turn("q");
        app.on_delta("\u{0}prompt_tokens:1200");
        app.ttft = Some(1.0);
        app.on_delta("hi");
        app.finish_turn(String::new());

        match app.transcript.last() {
            Some(Cell::Meta {
                prefill_tps: Some(rate),
                ..
            }) => assert!(
                *rate > 1000.0,
                "expected effective prefill rate, got {rate}"
            ),
            _ => panic!("expected meta with prefill throughput"),
        }

        let meta = app.transcript.last().unwrap();
        let rendered: String = cell_lines(meta, Mode::Inspect, 1.0)
            .iter()
            .flatten()
            .map(|(t, _)| t.clone())
            .collect();
        assert!(rendered.contains("prefill"), "missing label: {rendered}");
        assert!(rendered.contains("tok/s"), "missing rate: {rendered}");
    }

    #[test]
    fn meta_learns_prefill_rate_from_estimated_work() {
        let mut app = App::new();
        app.begin_turn("q");
        app.on_delta("\u{0}prefill_estimate:1000");
        app.ttft = Some(2.0);
        app.on_delta("hi");
        app.finish_turn(String::new());

        match app.transcript.last() {
            Some(Cell::Meta {
                prefill_tps: Some(rate),
                ..
            }) => assert!(
                (*rate - 500.0).abs() < 1.0,
                "expected estimated prefill throughput, got {rate}"
            ),
            _ => panic!("expected meta with estimated prefill throughput"),
        }
        assert!(
            app.prefill_rate_hint
                .is_some_and(|rate| (rate - 500.0).abs() < 1.0),
            "estimated throughput should seed the next turn's live progress"
        );
    }

    #[test]
    fn meta_preserves_observed_prefill_rate_without_usage() {
        let mut app = App::new();
        app.begin_turn("q");
        app.prefill_started = Some(Instant::now() - std::time::Duration::from_secs(1));
        app.on_delta("\u{0}prefill:1200/2400");
        app.on_delta("hi");
        app.finish_turn(String::new());

        match app.transcript.last() {
            Some(Cell::Meta {
                prefill_tps: Some(rate),
                ..
            }) => assert!(*rate > 900.0, "expected observed prefill rate, got {rate}"),
            _ => panic!("expected meta with observed prefill throughput"),
        }
    }

    fn input_text(app: &App) -> String {
        app.input.lines().join("\n")
    }

    #[test]
    fn up_down_recall_prompt_history() {
        let mut app = App::new();
        app.history.push("first".into());
        app.history.push("second".into());

        app.history_prev();
        assert_eq!(input_text(&app), "second");
        app.history_prev();
        assert_eq!(input_text(&app), "first");
        app.history_prev(); // clamps at oldest
        assert_eq!(input_text(&app), "first");
        app.history_next();
        assert_eq!(input_text(&app), "second");
        app.history_next(); // past newest → back to (empty) draft
        assert_eq!(input_text(&app), "");
    }

    #[test]
    fn reverse_search_finds_and_steps_back_through_matches() {
        let mut app = App::new();
        app.history.push("cargo build".into());
        app.history.push("git commit".into());
        app.history.push("cargo test".into());

        app.start_search();
        if let Some(s) = app.search.as_mut() {
            s.query.push_str("car");
        }
        app.refresh_search();
        assert_eq!(input_text(&app), "cargo test", "newest match");
        app.search_older();
        assert_eq!(input_text(&app), "cargo build", "older match");

        app.cancel_search();
        assert!(app.search.is_none());
        assert_eq!(input_text(&app), "", "draft restored on cancel");
    }

    fn pick_rows() -> Vec<PickRow> {
        vec![
            PickRow {
                id: "alpha".into(),
                label: "alpha   oMLX".into(),
                active: true,
            },
            PickRow {
                id: "beta".into(),
                label: "beta   LM Studio".into(),
                active: false,
            },
            PickRow {
                id: "gamma".into(),
                label: "gamma   file".into(),
                active: false,
            },
        ]
    }

    #[test]
    fn model_picker_navigates_and_selects() {
        let mut app = App::new();
        app.picker = Some(ModelPicker {
            rows: pick_rows(),
            selected: 0,
        });
        app.on_picker_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
        app.on_picker_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
        match app.on_picker_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)) {
            Action::PickModel(id) => assert_eq!(id, "gamma"),
            _ => panic!("expected PickModel"),
        }
        assert!(app.picker.is_none(), "picker closes after selection");
    }

    #[test]
    fn model_picker_esc_cancels() {
        let mut app = App::new();
        app.picker = Some(ModelPicker {
            rows: pick_rows(),
            selected: 1,
        });
        let act = app.on_picker_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
        assert!(matches!(act, Action::Continue));
        assert!(app.picker.is_none());
    }

    #[test]
    fn model_picker_renders_models() {
        use ratatui::backend::TestBackend;
        use ratatui::Terminal;

        let mut app = App::new();
        app.picker = Some(ModelPicker {
            rows: pick_rows(),
            selected: 1,
        });
        let mut term = Terminal::new(TestBackend::new(80, 16)).unwrap();
        term.draw(|f| app.draw(f, &test_footer())).unwrap();
        let text = buffer_text(term.backend().buffer());
        assert!(text.contains("alpha"), "models listed:\n{text}");
        assert!(text.contains("gamma"));
        assert!(text.contains("select model"));
    }

    #[test]
    fn mode_cycles_and_sets_by_name() {
        let mut app = App::new();
        assert_eq!(app.mode_label(), "calm");
        app.cycle_mode();
        assert_eq!(app.mode_label(), "inspect");
        app.cycle_mode();
        assert_eq!(app.mode_label(), "deep");
        app.cycle_mode();
        assert_eq!(app.mode_label(), "calm");
        assert!(app.set_mode("Deep"));
        assert_eq!(app.mode_label(), "deep");
        assert!(!app.set_mode("bogus"));
        assert_eq!(app.mode_label(), "deep");
    }

    #[test]
    fn mode_controls_tool_output_visibility() {
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
            result_data: "line one\nline two\nline three".into(),
            ok: true,
            duration_ms: 5,
        });
        let tool = app
            .transcript
            .iter()
            .find(|c| matches!(c, Cell::Tool { .. }))
            .unwrap();
        let flat = |mode: Mode| -> String {
            cell_lines(tool, mode, 0.0)
                .iter()
                .flatten()
                .map(|(t, _)| t.clone())
                .collect()
        };
        assert!(
            !flat(Mode::Calm).contains("line two"),
            "calm hides raw output"
        );
        assert!(
            flat(Mode::Deep).contains("line two"),
            "deep shows raw output"
        );
    }
}
