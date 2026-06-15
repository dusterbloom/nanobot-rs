//! Markdown → ratatui styled segments for assistant replies.
//!
//! Produces `Vec<Vec<(String, Style)>>` (logical lines of styled segments) that
//! feed the App's existing soft-wrap pipeline. Covers the markdown an assistant
//! actually emits: headings, **bold**, *italic*, `inline code`, bullet/numbered
//! lists, and fenced code blocks (syntax-highlighted via syntect). This mirrors
//! the classic `incremental.rs` renderer's logic, but emits `ratatui` styles
//! instead of ANSI escapes.

use once_cell::sync::Lazy;
use ratatui::style::{Color, Modifier, Style};
use syntect::easy::HighlightLines;
use syntect::highlighting::{Theme, ThemeSet};
use syntect::parsing::SyntaxSet;

/// A styled text run within a logical line.
pub(super) type Seg = (String, Style);

static SYNTAXES: Lazy<SyntaxSet> = Lazy::new(SyntaxSet::load_defaults_newlines);
static THEME: Lazy<Theme> =
    Lazy::new(|| ThemeSet::load_defaults().themes["base16-ocean.dark"].clone());

/// Inline code foreground (soft amber) — distinct without a background fill,
/// which keeps soft-wrapping clean.
const CODE_FG: Color = Color::Rgb(0xD7, 0xBA, 0x7D);

/// Render an assistant message into logical lines of styled segments.
pub(super) fn markdown(text: &str) -> Vec<Vec<Seg>> {
    let mut out: Vec<Vec<Seg>> = Vec::new();
    let mut highlighter: Option<HighlightLines<'static>> = None;

    for raw in text.split('\n') {
        let trimmed = raw.trim_start();
        if trimmed.starts_with("```") {
            highlighter = toggle_fence(highlighter, trimmed);
            continue;
        }
        match highlighter.as_mut() {
            Some(h) => out.push(highlight_code(raw, h)),
            None => out.push(inline_line(raw)),
        }
    }
    out
}

/// Open or close a fenced code block, returning the live highlighter (if open).
fn toggle_fence(current: Option<HighlightLines<'static>>, fence: &str) -> Option<HighlightLines<'static>> {
    if current.is_some() {
        return None; // closing fence
    }
    let lang = fence.trim_start_matches('`').trim();
    let syntax = SYNTAXES
        .find_syntax_by_token(lang)
        .unwrap_or_else(|| SYNTAXES.find_syntax_plain_text());
    Some(HighlightLines::new(syntax, &THEME))
}

/// Syntax-highlight one code line into segments, with a 2-space gutter.
fn highlight_code(line: &str, h: &mut HighlightLines<'static>) -> Vec<Seg> {
    let mut segs = vec![("  ".to_string(), Style::default())];
    match h.highlight_line(line, &SYNTAXES) {
        Ok(ranges) => {
            for (st, txt) in ranges {
                let c = st.foreground;
                segs.push((txt.to_string(), Style::default().fg(Color::Rgb(c.r, c.g, c.b))));
            }
        }
        Err(_) => segs.push((line.to_string(), Style::default())),
    }
    segs
}

/// Render a non-code line: heading / bullet / numbered / inline-formatted prose.
fn inline_line(line: &str) -> Vec<Seg> {
    if line.trim().is_empty() {
        return vec![(String::new(), Style::default())];
    }
    let trimmed = line.trim_start();
    let indent = line.len() - trimmed.len();

    if let Some(seg) = heading(trimmed) {
        return seg;
    }
    if let Some((marker, rest)) = list_marker(trimmed, indent) {
        let mut segs = vec![(marker, Style::default().fg(Color::Cyan))];
        segs.extend(inline_spans(rest));
        return segs;
    }
    inline_spans(line)
}

fn heading(trimmed: &str) -> Option<Vec<Seg>> {
    if !trimmed.starts_with('#') {
        return None;
    }
    let level = trimmed.chars().take_while(|&c| c == '#').count();
    if level > 6 {
        return None;
    }
    let rest = trimmed[level..].strip_prefix(' ')?;
    let style = if level <= 2 {
        Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::Cyan)
    };
    Some(vec![(rest.to_string(), style)])
}

/// Detect a bullet (`- `/`* `) or numbered (`N. `) list item, returning the
/// normalized marker (with original indent) and the remaining text.
fn list_marker(trimmed: &str, indent: usize) -> Option<(String, &str)> {
    let pad = " ".repeat(indent);
    if (trimmed.starts_with("- ") || trimmed.starts_with("* ")) && trimmed.len() > 2 {
        return Some((format!("{pad}\u{2022} "), &trimmed[2..]));
    }
    let end = trimmed.find(". ")?;
    let num = &trimmed[..end];
    if !num.is_empty() && num.chars().all(|c| c.is_ascii_digit()) {
        return Some((format!("{pad}{num}. "), &trimmed[end + 2..]));
    }
    None
}

/// Parse inline `**bold**`, `*italic*`, and `` `code` `` into styled segments.
fn inline_spans(text: &str) -> Vec<Seg> {
    let chars: Vec<char> = text.chars().collect();
    let mut segs: Vec<Seg> = Vec::new();
    let mut plain = String::new();
    let mut i = 0;

    while i < chars.len() {
        if let Some((inner, next, style)) = match_span(&chars, i) {
            if !plain.is_empty() {
                segs.push((std::mem::take(&mut plain), Style::default()));
            }
            segs.push((inner, style));
            i = next;
            continue;
        }
        plain.push(chars[i]);
        i += 1;
    }
    if !plain.is_empty() {
        segs.push((plain, Style::default()));
    }
    if segs.is_empty() {
        segs.push((String::new(), Style::default()));
    }
    segs
}

/// Try to match an inline span starting at `i`. Returns `(text, next_index, style)`.
fn match_span(chars: &[char], i: usize) -> Option<(String, usize, Style)> {
    // Inline code: `code`
    if chars[i] == '`' {
        let end = find(chars, i + 1, |c| c == '`')?;
        let inner: String = chars[i + 1..end].iter().collect();
        return Some((inner, end + 1, Style::default().fg(CODE_FG)));
    }
    // Bold: **text**
    if chars[i] == '*' && chars.get(i + 1) == Some(&'*') {
        let end = find_pair(chars, i + 2)?;
        let inner: String = chars[i + 2..end].iter().collect();
        return Some((inner, end + 2, Style::default().add_modifier(Modifier::BOLD)));
    }
    // Italic: *text* (single star, not bold)
    if chars[i] == '*' && chars.get(i + 1) != Some(&'*') {
        let end = find(chars, i + 1, |c| c == '*')?;
        let inner: String = chars[i + 1..end].iter().collect();
        return Some((inner, end + 1, Style::default().add_modifier(Modifier::ITALIC)));
    }
    None
}

fn find(chars: &[char], from: usize, pred: impl Fn(char) -> bool) -> Option<usize> {
    (from..chars.len()).find(|&j| pred(chars[j]))
}

fn find_pair(chars: &[char], from: usize) -> Option<usize> {
    (from..chars.len().saturating_sub(1)).find(|&j| chars[j] == '*' && chars[j + 1] == '*')
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat(segs: &[Seg]) -> String {
        segs.iter().map(|(t, _)| t.as_str()).collect()
    }

    #[test]
    fn bold_and_code_split_into_styled_segments() {
        let line = inline_spans("a **b** `c` d");
        // text is preserved (markers stripped)
        assert_eq!(flat(&line), "a b c d");
        // bold + code produce distinct styled runs
        assert!(line
            .iter()
            .any(|(t, s)| t == "b" && s.add_modifier.contains(Modifier::BOLD)));
        assert!(line.iter().any(|(t, s)| t == "c" && s.fg == Some(CODE_FG)));
    }

    #[test]
    fn heading_is_cyan_and_strips_hashes() {
        let lines = markdown("## Title");
        assert_eq!(flat(&lines[0]), "Title");
        assert_eq!(lines[0][0].1.fg, Some(Color::Cyan));
    }

    #[test]
    fn bullets_get_a_dot_marker() {
        let lines = markdown("- one\n- two");
        assert!(flat(&lines[0]).starts_with("\u{2022} "));
        assert!(flat(&lines[0]).contains("one"));
    }

    #[test]
    fn fenced_code_is_highlighted_and_fences_hidden() {
        let lines = markdown("text\n```rust\nlet x = 1;\n```\nafter");
        let joined: Vec<String> = lines.iter().map(|l| flat(l)).collect();
        // fence lines are consumed, code content survives with a gutter
        assert!(joined.iter().any(|l| l.contains("let x = 1;")));
        assert!(!joined.iter().any(|l| l.contains("```")));
        assert!(joined.iter().any(|l| l == "text"));
        assert!(joined.iter().any(|l| l == "after"));
    }

    #[test]
    fn blank_lines_are_preserved() {
        let lines = markdown("a\n\nb");
        assert_eq!(lines.len(), 3);
        assert_eq!(flat(&lines[1]), "");
    }
}
