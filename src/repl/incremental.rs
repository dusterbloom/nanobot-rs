#![allow(dead_code)]
//! Incremental line renderer for streaming LLM responses.
//!
//! Replaces the old progress-line display with live formatted text.
//! Lines are rendered immediately as they complete (on `\n`), with
//! inline markdown formatting for prose and syntect highlighting
//! for code blocks.

use std::io::Write as _;
use std::time::Instant;

use syntect::easy::HighlightLines;
use syntect::util::as_24_bit_terminal_escaped;

use crate::syntax::{SYNTAX_SET, THEME_SET};

/// Streaming state machine for classifying incoming lines.
enum StreamState {
    /// Normal prose — render with inline markdown.
    Prose,
    /// Inside a fenced code block — highlight with syntect.
    CodeBlock {
        lang: String,
        highlighter: HighlightLines<'static>,
    },
    /// Thinking deltas (dimmed). Transition in when we see `\x1b[90m`,
    /// transition out on `\x1b[0m\n\n`.
    Thinking { lines_printed: usize },
}

/// Incremental line renderer that prints formatted text as it streams.
///
/// Feed deltas via [`push()`], which buffers until `\n` and then prints
/// each completed line with appropriate formatting. Call [`finish()`]
/// when the stream ends to flush any remaining partial line.
pub struct IncrementalRenderer {
    state: StreamState,
    line_buffer: String,
    total_words: usize,
    start_time: Instant,
    first_line: bool,
    /// Track whether we have a visible partial line on the current terminal row.
    has_partial: bool,
    /// Number of terminal rows the current partial occupies (for multi-row erase).
    partial_rows: usize,
}

// SAFETY: IncrementalRenderer is only ever created, used, and dropped within
// a single tokio task. The non-Send parts (HighlightLines from syntect/onig
// containing raw pointers) are never shared between threads.
unsafe impl Send for IncrementalRenderer {}

impl IncrementalRenderer {
    pub fn new() -> Self {
        Self {
            state: StreamState::Prose,
            line_buffer: String::new(),
            total_words: 0,
            start_time: Instant::now(),
            first_line: true,
            has_partial: false,
            partial_rows: 0,
        }
    }

    /// Feed a text delta. May print zero or more completed lines to stdout.
    pub fn push(&mut self, delta: &str) {
        // Detect thinking start: agent_loop sends "\x1b[90m🧠 \x1b[2m"
        if delta.starts_with("\x1b[90m") {
            if !matches!(self.state, StreamState::Thinking { .. }) {
                self.clear_partial_internal();
                self.state = StreamState::Thinking { lines_printed: 0 };
            }
            // Print the thinking prefix dim
            print!("\r\x1b[K{}", delta);
            std::io::stdout().flush().ok();
            self.has_partial = true;
            return;
        }

        // Detect thinking end: agent_loop sends "\x1b[0m\n\n" before normal text.
        if matches!(self.state, StreamState::Thinking { .. }) {
            if let Some(reset_pos) = delta.find("\x1b[0m") {
                let before_reset = &delta[..reset_pos];
                let lines_printed = match &mut self.state {
                    StreamState::Thinking { lines_printed } => lines_printed,
                    _ => unreachable!(),
                };

                for ch in before_reset.chars() {
                    if ch == '\n' {
                        println!("\r");
                        *lines_printed += 1;
                        self.has_partial = false;
                    } else {
                        print!("{}", ch);
                        self.has_partial = true;
                    }
                }

                // Exit dim mode and preserve the blank separator lines.
                print!("\x1b[0m");
                let after_reset = &delta[reset_pos + "\x1b[0m".len()..];
                let mut split_at = 0usize;
                for (idx, ch) in after_reset.char_indices() {
                    if ch == '\n' || ch == '\r' {
                        split_at = idx + ch.len_utf8();
                    } else {
                        break;
                    }
                }
                if split_at > 0 {
                    print!("{}", &after_reset[..split_at]);
                }
                std::io::stdout().flush().ok();

                self.has_partial = false;
                self.state = StreamState::Prose;

                // Process any content after the reset/newline suffix.
                let remaining = &after_reset[split_at..];
                if !remaining.is_empty() {
                    self.push(remaining);
                }
                return;
            }
            // Still in thinking — print dim
            let lines_printed = match &mut self.state {
                StreamState::Thinking { lines_printed } => lines_printed,
                _ => unreachable!(),
            };
            for ch in delta.chars() {
                if ch == '\n' {
                    println!("\r");
                    *lines_printed += 1;
                    self.has_partial = false;
                } else {
                    print!("{}", ch);
                    self.has_partial = true;
                }
            }
            std::io::stdout().flush().ok();
            return;
        }

        // Normal prose/code handling: buffer and flush on newlines
        self.line_buffer.push_str(delta);
        self.flush_lines();
    }

    /// Flush all completed lines from the line buffer.
    fn flush_lines(&mut self) {
        while let Some(nl_pos) = self.line_buffer.find('\n') {
            let line = self.line_buffer[..nl_pos].to_string();
            self.line_buffer = self.line_buffer[nl_pos + 1..].to_string();

            // Clear any partial line display (may span multiple wrapped rows)
            if self.has_partial {
                self.erase_partial_rows();
                self.has_partial = false;
                self.partial_rows = 0;
            }

            self.render_line(&line);
        }

        // Show partial line (content after last \n) as overwritable.
        // Erase previous partial first — it may span multiple wrapped rows.
        if !self.line_buffer.is_empty() {
            if self.has_partial && self.partial_rows > 0 {
                self.erase_n_rows(self.partial_rows);
            }
            let partial = if matches!(self.state, StreamState::CodeBlock { .. }) {
                format!("  \x1b[2m{}\x1b[0m", &self.line_buffer)
            } else {
                format!("\x1b[2m{}\x1b[0m", &self.line_buffer)
            };
            print!("\r\x1b[K{}", partial);
            std::io::stdout().flush().ok();
            self.has_partial = true;
            self.partial_rows = self.compute_partial_rows();
        }
    }

    /// Render a single completed line based on current state.
    fn render_line(&mut self, line: &str) {
        // Count words for stats
        self.total_words += line.split_whitespace().count();

        // Check for code fence transitions
        let trimmed = line.trim_start();
        if trimmed.starts_with("```") {
            match &self.state {
                StreamState::CodeBlock { .. } => {
                    // Closing fence: print footer rule, transition to Prose
                    // Determine rule width (match the header's width)
                    println!("\r\x1b[38;5;240m{}\x1b[0m", "─".repeat(40));
                    self.state = StreamState::Prose;
                    return;
                }
                _ => {
                    // Opening fence: extract language, create highlighter
                    let lang = trimmed.strip_prefix("```").unwrap_or("").trim().to_string();
                    let lang_display = if lang.is_empty() { "code" } else { &lang };

                    // Print header rule
                    let rule_width: usize = 40;
                    let label_len = lang_display.len() + 2;
                    let remaining = rule_width.saturating_sub(label_len);
                    let left = 3.min(remaining);
                    let right = remaining.saturating_sub(left);
                    println!(
                        "\r\x1b[38;5;240m{} {} {}\x1b[0m",
                        "─".repeat(left),
                        lang_display,
                        "─".repeat(right),
                    );

                    // Create highlighter
                    let syntax = SYNTAX_SET
                        .find_syntax_by_token(&lang)
                        .or_else(|| SYNTAX_SET.find_syntax_by_extension(&lang))
                        .unwrap_or_else(|| SYNTAX_SET.find_syntax_plain_text());
                    let theme = &THEME_SET.themes["base16-ocean.dark"];
                    let highlighter = HighlightLines::new(syntax, theme);

                    self.state = StreamState::CodeBlock { lang, highlighter };
                    return;
                }
            }
        }

        // Render based on current state
        match &mut self.state {
            StreamState::Prose => {
                // Skip empty lines before the И marker
                if self.first_line && line.trim().is_empty() {
                    return;
                }
                let rendered = render_inline_markdown(line);
                if self.first_line {
                    // И marker on first prose line
                    println!("\r\x1b[1m\x1b[97mИ\x1b[0m {}", rendered);
                    self.first_line = false;
                } else {
                    println!("\r  {}", rendered);
                }
            }
            StreamState::CodeBlock {
                ref mut highlighter,
                ..
            } => {
                // Syntax-highlight the line
                match highlighter.highlight_line(line, &SYNTAX_SET) {
                    Ok(ranges) => {
                        let escaped = as_24_bit_terminal_escaped(&ranges[..], false);
                        println!("\r  {}\x1b[0m", escaped);
                    }
                    Err(_) => {
                        println!("\r  \x1b[32m{}\x1b[0m", line);
                    }
                }
            }
            StreamState::Thinking { .. } => {
                // Shouldn't reach here — thinking is handled in push()
                println!("\r\x1b[2m{}\x1b[0m", line);
            }
        }
        std::io::stdout().flush().ok();
    }

    /// Stream ended. Flush remaining partial line, print stats footer.
    pub fn finish(&mut self) {
        // Flush any remaining partial line
        if !self.line_buffer.is_empty() {
            if self.has_partial {
                self.erase_partial_rows();
                self.has_partial = false;
                self.partial_rows = 0;
            }
            let remaining = std::mem::take(&mut self.line_buffer);
            self.render_line(&remaining);
        }

        // Close any open code block
        if matches!(self.state, StreamState::CodeBlock { .. }) {
            println!("\r\x1b[38;5;240m{}\x1b[0m", "─".repeat(40));
            self.state = StreamState::Prose;
        }

        // Print stats footer
        let elapsed = self.start_time.elapsed().as_secs_f32();
        let rate = if elapsed > 0.5 {
            format!("  {:.1}w/s", self.total_words as f32 / elapsed)
        } else {
            String::new()
        };
        println!(
            "\r\x1b[2m⧗ {:.1}s  {}w{}\x1b[0m",
            elapsed, self.total_words, rate
        );
        std::io::stdout().flush().ok();
    }

    /// Whether the renderer hasn't printed any text yet (still on first line).
    pub fn is_first_line(&self) -> bool {
        self.first_line
    }

    /// Emit the И marker now (used when tool events arrive before any text).
    /// After this, subsequent text will render without the marker.
    pub fn emit_marker(&mut self) {
        if self.first_line {
            println!("\r\x1b[1m\x1b[97mИ\x1b[0m");
            std::io::stdout().flush().ok();
            self.first_line = false;
        }
    }

    /// Flush any buffered text as permanent output before tool events.
    ///
    /// Unlike `clear_partial()` which just erases the display, this renders
    /// the buffered text as a completed line so it appears *before* tool
    /// status lines. Call this before tool event rendering to preserve
    /// narrative order (introduction text before tool output).
    ///
    /// Returns `true` if text was flushed (caller may want to add spacing).
    pub fn flush_pending(&mut self) -> bool {
        if !self.line_buffer.is_empty() {
            // Erase the dimmed partial display first
            if self.has_partial {
                self.erase_partial_rows();
                self.has_partial = false;
                self.partial_rows = 0;
            }
            // Render the buffered text as a permanent line
            let remaining = std::mem::take(&mut self.line_buffer);
            self.render_line(&remaining);
            println!();
            std::io::stdout().flush().ok();
            true
        } else {
            false
        }
    }

    /// Clear the current partial line display (before tool event).
    pub fn clear_partial(&self) {
        if self.has_partial {
            self.erase_partial_rows();
        }
    }

    /// Restore partial line display (after tool event).
    pub fn restore_partial(&self) {
        if self.has_partial && !self.line_buffer.is_empty() {
            let partial = if matches!(self.state, StreamState::CodeBlock { .. }) {
                format!("  \x1b[2m{}\x1b[0m", &self.line_buffer)
            } else {
                format!("\x1b[2m{}\x1b[0m", &self.line_buffer)
            };
            print!("\r\x1b[K{}", partial);
            std::io::stdout().flush().ok();
        }
    }

    /// Periodic tick — show stall indicator if no data arriving.
    pub fn tick(&self) {
        // Nothing to do if no partial line — the last printed line is permanent
        // Could add a spinner here in the future
    }

    /// Compute how many terminal rows the current partial line occupies.
    fn compute_partial_rows(&self) -> usize {
        let width = crate::tui::terminal_width();
        let display_len = unicode_width::UnicodeWidthStr::width(self.line_buffer.as_str())
            + if matches!(self.state, StreamState::CodeBlock { .. }) {
                2
            } else {
                0
            };
        if display_len == 0 {
            1
        } else {
            (display_len + width - 1) / width
        }
    }

    /// Erase `n` terminal rows (current row + n-1 rows above).
    fn erase_n_rows(&self, n: usize) {
        // Erase current (bottom) row first, then move up and erase remaining.
        print!("\x1b[2K");
        for _ in 1..n {
            print!("\x1b[1A\x1b[2K");
        }
        print!("\r");
        std::io::stdout().flush().ok();
    }

    /// Erase all terminal rows occupied by the current partial line.
    fn erase_partial_rows(&self) {
        let rows = if self.partial_rows > 0 {
            self.partial_rows
        } else {
            self.compute_partial_rows()
        };
        self.erase_n_rows(rows);
    }

    /// Internal helper to clear partial without borrow issues.
    fn clear_partial_internal(&mut self) {
        if self.has_partial {
            self.erase_partial_rows();
            self.has_partial = false;
            self.partial_rows = 0;
        }
    }
}

/// Render a single prose line with inline markdown formatting.
///
/// Handles: headings, bold, italic, inline code, bullets, numbered lists, links.
/// This is intentionally simpler than termimad — we don't need paragraph reflow
/// for line-by-line streaming.
fn render_inline_markdown(line: &str) -> String {
    // Empty line
    if line.trim().is_empty() {
        return String::new();
    }

    // Headings: ^#{1,6} text
    if let Some(rest) = try_strip_heading(line) {
        return rest;
    }

    // Bullet lists: ^[\s]*[-*] text
    if let Some(rest) = try_strip_bullet(line) {
        return rest;
    }

    // Numbered lists: ^[\s]*\d+\. text
    if let Some(rest) = try_strip_numbered(line) {
        return rest;
    }

    // Inline formatting on the remaining text
    format_inline(line)
}

fn try_strip_heading(line: &str) -> Option<String> {
    let trimmed = line.trim_start();
    if !trimmed.starts_with('#') {
        return None;
    }
    let level = trimmed.chars().take_while(|c| *c == '#').count();
    if level > 6 {
        return None;
    }
    let rest = trimmed[level..].strip_prefix(' ')?;
    // Headings: cyan + bold for h1-h2, cyan for h3+
    let formatted = format_inline(rest);
    if level <= 2 {
        Some(format!("\x1b[1m\x1b[36m{}\x1b[0m", formatted))
    } else {
        Some(format!("\x1b[36m{}\x1b[0m", formatted))
    }
}

fn try_strip_bullet(line: &str) -> Option<String> {
    let indent = line.len() - line.trim_start().len();
    let trimmed = line.trim_start();
    if (trimmed.starts_with("- ") || trimmed.starts_with("* ")) && trimmed.len() > 2 {
        let rest = &trimmed[2..];
        let indent_str = " ".repeat(indent);
        Some(format!("{}• {}", indent_str, format_inline(rest)))
    } else {
        None
    }
}

fn try_strip_numbered(line: &str) -> Option<String> {
    let indent = line.len() - line.trim_start().len();
    let trimmed = line.trim_start();
    let num_end = trimmed.find(". ")?;
    let num_part = &trimmed[..num_end];
    if num_part.chars().all(|c| c.is_ascii_digit()) && !num_part.is_empty() {
        let rest = &trimmed[num_end + 2..];
        let indent_str = " ".repeat(indent);
        Some(format!(
            "{}{}. {}",
            indent_str,
            num_part,
            format_inline(rest)
        ))
    } else {
        None
    }
}

/// Apply inline formatting: bold, italic, inline code, links.
fn format_inline(text: &str) -> String {
    let mut result = String::with_capacity(text.len() + 32);
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        // Bold: **text**
        if i + 1 < len && chars[i] == '*' && chars[i + 1] == '*' {
            if let Some(end) = find_closing(&chars, i + 2, &['*', '*']) {
                let inner: String = chars[i + 2..end].iter().collect();
                result.push_str(&format!("\x1b[1m{}\x1b[0m", inner));
                i = end + 2;
                continue;
            }
        }

        // Italic: *text* (but not **)
        if chars[i] == '*' && (i + 1 >= len || chars[i + 1] != '*') {
            if let Some(end) = find_closing_single(&chars, i + 1, '*') {
                let inner: String = chars[i + 1..end].iter().collect();
                result.push_str(&format!("\x1b[3m{}\x1b[0m", inner));
                i = end + 1;
                continue;
            }
        }

        // Italic: _text_
        if chars[i] == '_' && (i == 0 || chars[i - 1].is_whitespace()) {
            if let Some(end) = find_closing_single(&chars, i + 1, '_') {
                let inner: String = chars[i + 1..end].iter().collect();
                result.push_str(&format!("\x1b[3m{}\x1b[0m", inner));
                i = end + 1;
                continue;
            }
        }

        // Inline code: `text`
        if chars[i] == '`' {
            if let Some(end) = find_closing_single(&chars, i + 1, '`') {
                let inner: String = chars[i + 1..end].iter().collect();
                result.push_str(&format!("\x1b[32m{}\x1b[0m", inner));
                i = end + 1;
                continue;
            }
        }

        // Links: [text](url)
        if chars[i] == '[' {
            if let Some((text_end, url_start, url_end)) = find_link(&chars, i) {
                let link_text: String = chars[i + 1..text_end].iter().collect();
                let link_url: String = chars[url_start..url_end].iter().collect();
                result.push_str(&format!("{} \x1b[2m{}\x1b[0m", link_text, link_url));
                i = url_end + 1;
                continue;
            }
        }

        result.push(chars[i]);
        i += 1;
    }

    result
}

/// Find closing double-char delimiter (e.g., ** for bold).
fn find_closing(chars: &[char], start: usize, delim: &[char; 2]) -> Option<usize> {
    let len = chars.len();
    let mut j = start;
    while j + 1 < len {
        if chars[j] == delim[0] && chars[j + 1] == delim[1] {
            return Some(j);
        }
        j += 1;
    }
    None
}

/// Find closing single-char delimiter (e.g., * for italic, ` for code).
fn find_closing_single(chars: &[char], start: usize, delim: char) -> Option<usize> {
    for j in start..chars.len() {
        if chars[j] == delim {
            return Some(j);
        }
    }
    None
}

/// Find a markdown link: [text](url). Returns (text_end, url_start, url_end).
fn find_link(chars: &[char], start: usize) -> Option<(usize, usize, usize)> {
    // Find closing ]
    let mut j = start + 1;
    while j < chars.len() && chars[j] != ']' {
        j += 1;
    }
    if j >= chars.len() {
        return None;
    }
    let text_end = j;
    // Expect (
    if j + 1 >= chars.len() || chars[j + 1] != '(' {
        return None;
    }
    let url_start = j + 2;
    // Find closing )
    let mut k = url_start;
    while k < chars.len() && chars[k] != ')' {
        k += 1;
    }
    if k >= chars.len() {
        return None;
    }
    Some((text_end, url_start, k))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Strip ANSI escape sequences for content assertions.
    fn strip_ansi(s: &str) -> String {
        let mut out = String::new();
        let mut chars = s.chars().peekable();
        while let Some(c) = chars.next() {
            if c == '\x1b' {
                while let Some(&next) = chars.peek() {
                    chars.next();
                    if next.is_ascii_alphabetic() {
                        break;
                    }
                }
            } else {
                out.push(c);
            }
        }
        out
    }

    #[test]
    fn test_render_heading() {
        let result = render_inline_markdown("# Hello World");
        let plain = strip_ansi(&result);
        assert!(plain.contains("Hello World"));
        // Should have cyan color
        assert!(result.contains("\x1b[36m"));
    }

    #[test]
    fn test_render_h3() {
        let result = render_inline_markdown("### Sub heading");
        let plain = strip_ansi(&result);
        assert!(plain.contains("Sub heading"));
        assert!(result.contains("\x1b[36m"));
        // h3+ should not be bold
        assert!(!result.contains("\x1b[1m"));
    }

    #[test]
    fn test_render_bullet() {
        let result = render_inline_markdown("- item one");
        let plain = strip_ansi(&result);
        assert!(plain.contains("•"));
        assert!(plain.contains("item one"));
    }

    #[test]
    fn test_render_numbered_list() {
        let result = render_inline_markdown("1. first item");
        let plain = strip_ansi(&result);
        assert!(plain.contains("1."));
        assert!(plain.contains("first item"));
    }

    #[test]
    fn test_render_bold() {
        let result = render_inline_markdown("This is **bold** text");
        assert!(result.contains("\x1b[1m"));
        let plain = strip_ansi(&result);
        assert!(plain.contains("bold"));
        assert!(plain.contains("This is"));
    }

    #[test]
    fn test_render_italic() {
        let result = render_inline_markdown("This is *italic* text");
        assert!(result.contains("\x1b[3m"));
        let plain = strip_ansi(&result);
        assert!(plain.contains("italic"));
    }

    #[test]
    fn test_render_inline_code() {
        let result = render_inline_markdown("Use `foo()` here");
        assert!(result.contains("\x1b[32m"));
        let plain = strip_ansi(&result);
        assert!(plain.contains("foo()"));
    }

    #[test]
    fn test_render_link() {
        let result = render_inline_markdown("See [docs](https://example.com) for info");
        let plain = strip_ansi(&result);
        assert!(plain.contains("docs"));
        assert!(plain.contains("https://example.com"));
    }

    #[test]
    fn test_render_empty_line() {
        let result = render_inline_markdown("");
        assert!(result.is_empty());
    }

    #[test]
    fn test_render_plain_text() {
        let result = render_inline_markdown("Just plain text");
        let plain = strip_ansi(&result);
        assert_eq!(plain, "Just plain text");
    }

    #[test]
    fn test_format_inline_mixed() {
        let result = format_inline("Hello **bold** and `code` end");
        let plain = strip_ansi(&result);
        assert_eq!(plain, "Hello bold and code end");
    }

    #[test]
    fn test_indented_bullet() {
        let result = render_inline_markdown("  - nested item");
        let plain = strip_ansi(&result);
        assert!(plain.contains("•"));
        assert!(plain.starts_with("  "));
    }
}
