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
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

/// A styled text run within a logical line.
pub(super) type Seg = (String, Style);

static SYNTAXES: Lazy<SyntaxSet> = Lazy::new(SyntaxSet::load_defaults_newlines);
static THEME: Lazy<Theme> =
    Lazy::new(|| ThemeSet::load_defaults().themes["base16-ocean.dark"].clone());

/// Inline code foreground (soft amber) — distinct without a background fill,
/// which keeps soft-wrapping clean.
const CODE_FG: Color = Color::Rgb(0xD7, 0xBA, 0x7D);
const MARKDOWN_ACCENT: Color = Color::Rgb(0x28, 0xB3, 0xC1);
// Wrap oversized table cells here so the transcript never has to hide them.
const TABLE_CELL_WIDTH_CAP: usize = 64;

/// Render an assistant message into logical lines of styled segments.
pub(super) fn markdown(text: &str) -> Vec<Vec<Seg>> {
    let mut out: Vec<Vec<Seg>> = Vec::new();
    let mut highlighter: Option<HighlightLines<'static>> = None;
    let lines: Vec<&str> = text.split('\n').collect();
    let mut i = 0;

    while i < lines.len() {
        let raw = lines[i];
        let trimmed = raw.trim_start();
        if trimmed.starts_with("```") {
            highlighter = toggle_fence(highlighter, trimmed);
            i += 1;
            continue;
        }
        match highlighter.as_mut() {
            Some(h) => out.push(highlight_code(raw, h)),
            None => {
                if i + 1 < lines.len() && is_table_row(raw) && is_table_separator(lines[i + 1]) {
                    let start = i;
                    i += 2; // header + separator
                    while i < lines.len() && is_table_row(lines[i]) {
                        i += 1;
                    }
                    out.extend(table_lines(&lines[start..i]));
                    continue;
                }
                out.push(inline_line(raw));
            }
        }
        i += 1;
    }
    out
}

/// Open or close a fenced code block, returning the live highlighter (if open).
fn toggle_fence(
    current: Option<HighlightLines<'static>>,
    fence: &str,
) -> Option<HighlightLines<'static>> {
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
                segs.push((
                    txt.to_string(),
                    Style::default().fg(Color::Rgb(c.r, c.g, c.b)),
                ));
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

    if is_horizontal_rule(trimmed) {
        return vec![("\u{2500}".repeat(48), Style::default().fg(Color::DarkGray))];
    }
    if let Some(seg) = heading(trimmed) {
        return seg;
    }
    if let Some((marker, rest)) = list_marker(trimmed, indent) {
        let mut segs = vec![(marker, Style::default().fg(MARKDOWN_ACCENT))];
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
        Style::default()
            .fg(MARKDOWN_ACCENT)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(MARKDOWN_ACCENT)
    };
    Some(vec![(rest.to_string(), style)])
}

fn is_horizontal_rule(trimmed: &str) -> bool {
    let marks: Vec<char> = trimmed.chars().filter(|c| !c.is_whitespace()).collect();
    marks.len() >= 3
        && matches!(marks.first(), Some('*' | '-' | '_'))
        && marks.iter().all(|c| Some(c) == marks.first())
}

fn is_table_row(line: &str) -> bool {
    let trimmed = line.trim();
    trimmed.contains('|') && !trimmed.is_empty()
}

fn is_table_separator(line: &str) -> bool {
    let trimmed = line.trim();
    trimmed.contains("---") && trimmed.chars().all(|c| matches!(c, '|' | '-' | ':' | ' '))
}

fn table_lines(lines: &[&str]) -> Vec<Vec<Seg>> {
    let rows: Vec<Vec<String>> = lines.iter().map(|line| table_cells(line)).collect();
    if rows.is_empty() {
        return Vec::new();
    }
    let cols = rows.iter().map(Vec::len).max().unwrap_or(0);
    let mut widths = vec![0usize; cols];
    for row in rows.iter().filter(|row| !is_separator_cells(row)) {
        for (i, cell) in row.iter().enumerate() {
            widths[i] =
                widths[i].max(display_width(&table_cell_text(cell)).min(TABLE_CELL_WIDTH_CAP));
        }
    }
    let mut rendered = Vec::new();
    for (row_idx, row) in rows.iter().enumerate() {
        if is_separator_cells(row) {
            rendered.push(table_rule(&widths));
            continue;
        }
        rendered.extend(table_row(row, &widths, row_idx == 0));
    }
    rendered
}

fn table_cells(line: &str) -> Vec<String> {
    line.trim()
        .trim_matches('|')
        .split('|')
        .map(|cell| cell.trim().to_string())
        .collect()
}

fn is_separator_cells(row: &[String]) -> bool {
    !row.is_empty()
        && row.iter().all(|cell| {
            let trimmed = cell.trim();
            trimmed.contains("---") && trimmed.chars().all(|c| matches!(c, '-' | ':' | ' '))
        })
}

fn table_rule(widths: &[usize]) -> Vec<Seg> {
    let mut line = String::from("  ");
    for (i, width) in widths.iter().enumerate() {
        if i > 0 {
            line.push_str("  ");
        }
        line.push_str(&"─".repeat((*width).max(3)));
    }
    vec![(line, Style::default().fg(Color::DarkGray))]
}

fn table_row(row: &[String], widths: &[usize], header: bool) -> Vec<Vec<Seg>> {
    let wrapped: Vec<Vec<String>> = widths
        .iter()
        .enumerate()
        .map(|(i, width)| {
            let cell = row.get(i).map(String::as_str).unwrap_or("");
            wrap_table_cell(&table_cell_text(cell), *width)
        })
        .collect();
    let height = wrapped.iter().map(Vec::len).max().unwrap_or(1);
    let mut lines = Vec::with_capacity(height);

    for line_idx in 0..height {
        let mut segs = vec![("  ".to_string(), Style::default())];
        for (i, width) in widths.iter().enumerate() {
            if i > 0 {
                segs.push(("  ".to_string(), Style::default().fg(Color::DarkGray)));
            }
            let cell = wrapped
                .get(i)
                .and_then(|lines| lines.get(line_idx))
                .map(String::as_str)
                .unwrap_or("");
            let text = if i + 1 == widths.len() {
                cell.to_string()
            } else {
                pad_to_width(cell, *width)
            };
            let style = if header {
                Style::default()
                    .fg(MARKDOWN_ACCENT)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };
            segs.push((text, style));
        }
        lines.push(segs);
    }
    lines
}

fn wrap_table_cell(cell: &str, width: usize) -> Vec<String> {
    let width = width.max(1);
    let mut lines = Vec::new();
    let mut current = String::new();
    let mut current_width = 0usize;

    for word in cell.split_whitespace() {
        let word_width = display_width(word);
        if word_width <= width {
            if current.is_empty() {
                current.push_str(word);
                current_width = word_width;
            } else if current_width + 1 + word_width <= width {
                current.push(' ');
                current.push_str(word);
                current_width += 1 + word_width;
            } else {
                lines.push(std::mem::take(&mut current));
                current.push_str(word);
                current_width = word_width;
            }
            continue;
        }

        if !current.is_empty() {
            lines.push(std::mem::take(&mut current));
        }
        let chunks = split_wide_word(word, width);
        let last = chunks.len().saturating_sub(1);
        for (idx, chunk) in chunks.into_iter().enumerate() {
            if idx == last {
                current_width = display_width(&chunk);
                current = chunk;
            } else {
                lines.push(chunk);
            }
        }
    }

    if !current.is_empty() {
        lines.push(current);
    }
    if lines.is_empty() {
        lines.push(String::new());
    }
    lines
}

fn table_cell_text(cell: &str) -> String {
    inline_spans(cell)
        .iter()
        .map(|(text, _)| text.as_str())
        .collect()
}

fn split_wide_word(word: &str, width: usize) -> Vec<String> {
    let width = width.max(1);
    let mut chunks = Vec::new();
    let mut current = String::new();
    let mut current_width = 0usize;

    for c in word.chars() {
        let char_width = UnicodeWidthChar::width(c).unwrap_or(0);
        if current_width + char_width > width && !current.is_empty() {
            chunks.push(std::mem::take(&mut current));
            current_width = 0;
        }
        current.push(c);
        current_width += char_width;
    }

    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

fn pad_to_width(text: &str, width: usize) -> String {
    let pad = width.saturating_sub(display_width(text));
    let mut padded = text.to_string();
    padded.push_str(&" ".repeat(pad));
    padded
}

fn display_width(text: &str) -> usize {
    UnicodeWidthStr::width(text)
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
        if let Some((mut spans, next)) = match_span(&chars, i) {
            if !plain.is_empty() {
                segs.push((std::mem::take(&mut plain), Style::default()));
            }
            segs.append(&mut spans);
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
fn match_span(chars: &[char], i: usize) -> Option<(Vec<Seg>, usize)> {
    if chars[i] == '!' && chars.get(i + 1) == Some(&'[') {
        let (alt, url, next) = bracket_link(chars, i + 1)?;
        let label = if alt.trim().is_empty() {
            "image".to_string()
        } else {
            format!("image: {alt}")
        };
        return Some((
            vec![
                (
                    label,
                    Style::default()
                        .fg(MARKDOWN_ACCENT)
                        .add_modifier(Modifier::BOLD),
                ),
                (format!(" ({url})"), Style::default().fg(Color::DarkGray)),
            ],
            next,
        ));
    }
    if chars[i] == '[' {
        let (label, url, next) = bracket_link(chars, i)?;
        return Some((
            vec![
                (
                    label,
                    Style::default()
                        .fg(MARKDOWN_ACCENT)
                        .add_modifier(Modifier::UNDERLINED),
                ),
                (format!(" ({url})"), Style::default().fg(Color::DarkGray)),
            ],
            next,
        ));
    }
    // Inline code: `code`
    if chars[i] == '`' {
        let end = find(chars, i + 1, |c| c == '`')?;
        let inner: String = chars[i + 1..end].iter().collect();
        return Some((vec![(inner, Style::default().fg(CODE_FG))], end + 1));
    }
    // Bold italic: ***text***
    if chars[i] == '*' && chars.get(i + 1) == Some(&'*') && chars.get(i + 2) == Some(&'*') {
        let end = find_triple(chars, i + 3)?;
        let inner: String = chars[i + 3..end].iter().collect();
        return Some((
            vec![(
                inner,
                Style::default().add_modifier(Modifier::BOLD | Modifier::ITALIC),
            )],
            end + 3,
        ));
    }
    // Bold: **text**
    if chars[i] == '*' && chars.get(i + 1) == Some(&'*') {
        let end = find_pair(chars, i + 2)?;
        let inner: String = chars[i + 2..end].iter().collect();
        return Some((
            vec![(inner, Style::default().add_modifier(Modifier::BOLD))],
            end + 2,
        ));
    }
    // Italic: *text* (single star, not bold)
    if chars[i] == '*' && chars.get(i + 1) != Some(&'*') {
        let end = find(chars, i + 1, |c| c == '*')?;
        let inner: String = chars[i + 1..end].iter().collect();
        return Some((
            vec![(inner, Style::default().add_modifier(Modifier::ITALIC))],
            end + 1,
        ));
    }
    None
}

fn bracket_link(chars: &[char], open: usize) -> Option<(String, String, usize)> {
    if chars.get(open) != Some(&'[') {
        return None;
    }
    let close = find(chars, open + 1, |c| c == ']')?;
    if chars.get(close + 1) != Some(&'(') {
        return None;
    }
    let end = find(chars, close + 2, |c| c == ')')?;
    let label: String = chars[open + 1..close].iter().collect();
    let url: String = chars[close + 2..end].iter().collect();
    Some((label, url, end + 1))
}

fn find(chars: &[char], from: usize, pred: impl Fn(char) -> bool) -> Option<usize> {
    (from..chars.len()).find(|&j| pred(chars[j]))
}

fn find_pair(chars: &[char], from: usize) -> Option<usize> {
    (from..chars.len().saturating_sub(1)).find(|&j| chars[j] == '*' && chars[j + 1] == '*')
}

fn find_triple(chars: &[char], from: usize) -> Option<usize> {
    (from..chars.len().saturating_sub(2))
        .find(|&j| chars[j] == '*' && chars[j + 1] == '*' && chars[j + 2] == '*')
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
    fn bold_italic_triple_stars_strip_markers() {
        let line = inline_spans("a ***b*** c");
        assert_eq!(flat(&line), "a b c");
        assert!(line.iter().any(|(t, s)| {
            t == "b"
                && s.add_modifier.contains(Modifier::BOLD)
                && s.add_modifier.contains(Modifier::ITALIC)
        }));
    }

    #[test]
    fn heading_is_cyan_and_strips_hashes() {
        let lines = markdown("## Title");
        assert_eq!(flat(&lines[0]), "Title");
        assert_eq!(lines[0][0].1.fg, Some(MARKDOWN_ACCENT));
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

    #[test]
    fn horizontal_rule_markers_render_as_rule() {
        let lines = markdown("before\n***\nafter");
        let rendered: Vec<String> = lines.iter().map(|line| flat(line)).collect();
        assert_eq!(rendered[0], "before");
        assert!(rendered[1].contains("\u{2500}\u{2500}\u{2500}"));
        assert!(!rendered[1].contains("***"));
        assert_eq!(rendered[2], "after");
    }

    #[test]
    fn links_and_images_render_as_terminal_spans() {
        let lines = markdown("see [docs](https://example.com) and ![chart](plot.png)");
        let flat = flat(&lines[0]);
        assert!(flat.contains("docs (https://example.com)"));
        assert!(flat.contains("image: chart (plot.png)"));
        assert!(!flat.contains("![chart]"));
    }

    #[test]
    fn markdown_tables_render_as_aligned_rows() {
        let lines = markdown("| Name | Score |\n| --- | ---: |\n| Ada | 10 |\n| Bob | 9 |");
        let rendered: Vec<String> = lines.iter().map(|line| flat(line)).collect();
        assert!(rendered[0].contains("Name"));
        assert!(rendered[0].contains("Score"));
        assert!(rendered[1].contains("───"));
        assert!(rendered[2].contains("Ada"));
        assert!(rendered[2].contains("10"));
        assert!(!rendered.iter().any(|line| line.contains("---:")));
    }

    #[test]
    fn markdown_tables_strip_inline_markers_in_cells() {
        let lines = markdown("| Commit | Scope |\n| --- | --- |\n| `65e6ef9` | **skills** |");
        let rendered: Vec<String> = lines.iter().map(|line| flat(line)).collect();
        let joined = rendered.join("\n");

        assert!(joined.contains("65e6ef9"));
        assert!(joined.contains("skills"));
        assert!(!joined.contains('`'));
        assert!(!joined.contains("**"));
    }

    #[test]
    fn markdown_tables_wrap_long_cells_without_clipping() {
        let long =
            "The logic for picking skills was refactored so the reply can explain the full change";
        let input = format!("| Commit | Summary |\n| --- | --- |\n| 65e6ef9 | {long} |");
        let lines = markdown(&input);
        let rendered: Vec<String> = lines.iter().map(|line| flat(line)).collect();
        let joined = rendered.join("\n");

        assert!(joined.contains("The logic for picking skills was refactored"));
        assert!(joined.contains("reply can"));
        assert!(joined.contains("explain the full change"));
        assert!(!joined.contains('\u{2026}'));
        assert!(
            rendered.len() > 3,
            "long table cell should wrap into visible continuation rows: {joined}"
        );
    }
}
