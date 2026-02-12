/// Streaming markdown display for real-time LLM output.
///
/// Accumulates text deltas, re-renders through termimad on each push,
/// and uses cursor movement to overwrite only changed lines — giving
/// live-updating markdown with minimal flicker.
///
/// Also handles rich TUI events (tool indicators, thinking spinner).

use std::io::Write;
use std::time::Instant;

use super::ansi::{self, BOLD, CYAN, DIM, GREEN, RED, RESET};
use super::events::TuiEvent;
use super::markdown::make_skin;

/// Minimum interval between re-renders (~30 fps).
const RENDER_INTERVAL_MS: u128 = 33;

/// Streams markdown to the terminal, re-rendering incrementally as deltas arrive.
/// Also displays tool execution indicators inline.
pub struct StreamDisplay {
    /// Whether we've printed the turn header yet.
    header_printed: bool,
    /// Whether we've received any text content in the current render block.
    has_content: bool,
    /// Accumulated raw text from all deltas so far.
    buffer: String,
    /// Byte offset into `buffer` for the current render block. Text before this
    /// offset has been finalized (printed and locked — we won't cursor-up over it).
    /// This advances when tool lines break the rendered block.
    buffer_offset: usize,
    /// Previously rendered lines in the current block (for differential updates).
    rendered_lines: Vec<String>,
    /// termimad skin for markdown rendering.
    skin: termimad::MadSkin,
    /// Available width for markdown rendering (terminal width minus margins).
    width: usize,
    /// Last time we performed a re-render.
    last_render: Instant,
    /// Whether there are unrendered deltas (dirty flag).
    dirty: bool,
    /// Currently displayed tool/thinking indicator line (via print!, no newline).
    tool_line_active: bool,
}

impl StreamDisplay {
    pub fn new() -> Self {
        let width = ansi::terminal_width().saturating_sub(4);
        Self {
            header_printed: false,
            has_content: false,
            buffer: String::new(),
            buffer_offset: 0,
            rendered_lines: Vec::new(),
            skin: make_skin(),
            width: width.max(20),
            last_render: Instant::now(),
            dirty: false,
            tool_line_active: false,
        }
    }

    /// Print the turn header if not already printed.
    fn ensure_header(&mut self) {
        if !self.header_printed {
            self.header_printed = true;
            println!();
            let term_width = ansi::terminal_width().saturating_sub(2);
            let prefix_len = 4;
            let label = "nanobot";
            let label_len = label.len() + 2;
            let rule_len = term_width.saturating_sub(prefix_len + label_len);
            println!(
                "  {DIM}───{RESET} {GREEN}{BOLD}{label}{RESET} {DIM}{}{RESET}",
                "─".repeat(rule_len),
            );
        }
    }

    /// Clear any active tool/thinking indicator line (the one printed with
    /// `print!` — no trailing newline).
    fn clear_tool_line(&mut self) {
        if self.tool_line_active {
            print!("\x1b[2K\r");
            let _ = std::io::stdout().flush();
            self.tool_line_active = false;
        }
    }

    /// Finalize the current render block. Any previously rendered markdown lines
    /// are considered "locked" and won't be cursor-up'd over again.
    fn finalize_block(&mut self) {
        if self.dirty {
            self.render();
        }
        // Advance offset so next render only processes new text
        self.buffer_offset = self.buffer.len();
        self.rendered_lines.clear();
        self.has_content = false;
    }

    /// Handle a TUI event.
    pub fn handle_event(&mut self, event: TuiEvent) {
        match event {
            TuiEvent::TextDelta(delta) => self.push(&delta),
            TuiEvent::ToolStart { name, .. } => {
                self.ensure_header();
                // Finalize any in-progress rendered content before the tool line
                if self.has_content {
                    self.finalize_block();
                }
                self.clear_tool_line();
                print!("  {DIM}⚙ {name}…{RESET}");
                let _ = std::io::stdout().flush();
                self.tool_line_active = true;
            }
            TuiEvent::ToolComplete {
                name,
                success,
                duration,
                ..
            } => {
                self.clear_tool_line();
                let status = if success {
                    format!("{GREEN}✓{RESET}")
                } else {
                    format!("{RED}✗{RESET}")
                };
                let ms = duration.as_millis();
                let time_str = if ms < 1000 {
                    format!("{ms}ms")
                } else {
                    format!("{:.1}s", duration.as_secs_f64())
                };
                println!("  {DIM}{status} {name} {CYAN}{time_str}{RESET}");
                let _ = std::io::stdout().flush();
            }
            TuiEvent::Thinking => {
                self.ensure_header();
                if !self.has_content {
                    self.clear_tool_line();
                    print!("  {DIM}⠋ thinking…{RESET}");
                    let _ = std::io::stdout().flush();
                    self.tool_line_active = true;
                }
            }
            TuiEvent::Done => {
                self.clear_tool_line();
                // Final render of any pending content
                if self.dirty {
                    self.render();
                }
            }
        }
    }

    /// Push a text delta to the display. Accumulates text and throttled
    /// re-renders through termimad for live markdown.
    pub fn push(&mut self, delta: &str) {
        if delta.is_empty() {
            return;
        }

        self.ensure_header();

        // Clear any tool/thinking indicator before first text in this block
        if !self.has_content {
            self.clear_tool_line();
            self.has_content = true;
        }

        self.buffer.push_str(delta);
        self.dirty = true;

        // Throttle re-renders to ~30fps
        let elapsed = self.last_render.elapsed().as_millis();
        if elapsed >= RENDER_INTERVAL_MS {
            self.render();
        }
    }

    /// Re-render the current block (buffer from `buffer_offset` onward) through
    /// termimad and diff against previously rendered lines. Uses cursor-up +
    /// clear-line to overwrite only changed lines.
    fn render(&mut self) {
        self.dirty = false;
        self.last_render = Instant::now();

        // Only render text from the current block
        let block_text = &self.buffer[self.buffer_offset..];
        if block_text.is_empty() {
            return;
        }

        // Render markdown through termimad
        let rendered = format!("{}", self.skin.text(block_text, Some(self.width)));
        let new_lines: Vec<String> = rendered.lines().map(|l| l.trim_end().to_string()).collect();

        let old_count = self.rendered_lines.len();
        let new_count = new_lines.len();

        if old_count == 0 {
            // First render of this block: just print everything
            for line in &new_lines {
                println!("  {}", line);
            }
        } else {
            // Move cursor up to the start of the previously rendered block
            if old_count > 0 {
                print!("\x1b[{}A", old_count);
            }

            // Re-print all lines (overwriting old content)
            for (i, line) in new_lines.iter().enumerate() {
                if i < old_count {
                    // Overwrite existing line
                    print!("\x1b[2K\r  {}\n", line);
                } else {
                    // New line beyond previous content
                    print!("  {}\n", line);
                }
            }

            // If new content has fewer lines than old, clear leftover lines
            for _ in new_count..old_count {
                print!("\x1b[2K\n");
            }
            // Move back up if we cleared extra lines
            if new_count < old_count {
                print!("\x1b[{}A", old_count - new_count);
            }
        }

        let _ = std::io::stdout().flush();
        self.rendered_lines = new_lines;
    }

    /// Finalize the stream display. Performs a final render of any pending
    /// content. Returns `true` if any content was displayed (header was printed).
    pub fn finish(&mut self) -> bool {
        if !self.header_printed {
            return false;
        }
        self.clear_tool_line();
        // Final render to catch any throttled content
        if self.dirty {
            self.render();
        }
        let _ = std::io::stdout().flush();
        true
    }
}
