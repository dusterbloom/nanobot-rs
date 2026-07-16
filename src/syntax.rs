//! Unified text rendering pipeline for LLM responses.
//!
//! Combines termimad prose formatting (headers, bold, lists, links) with
//! syntect code block highlighting. One function `render_response()` handles
//! all display paths.

use once_cell::sync::Lazy;
use syntect::easy::HighlightLines;
use syntect::highlighting::ThemeSet;
use syntect::parsing::SyntaxSet;
use syntect::util::as_24_bit_terminal_escaped;
use termimad::crossterm::style::Color;
use termimad::MadSkin;

pub(crate) static SYNTAX_SET: Lazy<SyntaxSet> = Lazy::new(|| SyntaxSet::load_defaults_newlines());
pub(crate) static THEME_SET: Lazy<ThemeSet> = Lazy::new(ThemeSet::load_defaults);

/// Shared termimad skin for prose rendering.
pub(crate) static SKIN: Lazy<MadSkin> = Lazy::new(|| {
    let mut skin = MadSkin::default_dark();
    skin.headers[0].set_fg(Color::Cyan);
    skin.headers[1].set_fg(Color::Cyan);
    skin.bold.set_fg(Color::White);
    skin.italic.set_fg(Color::Magenta);
    skin.inline_code.set_fg(Color::Green);
    skin.code_block.set_fg(Color::Green);
    skin
});

/// Which side of the conversation a turn belongs to.
///
/// Assistant turns are rendered incrementally (see `render_response` and the
/// repl streaming path), so there is no assistant variant here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnRole {
    User,
    /// Voice mode user — same as User but with a purple ● marker.
    VoiceUser,
}

/// Render a user conversation turn with a role marker.
///
/// The text is raw (not markdown-rendered — the prompt may contain literal
/// markup). Empty input returns an empty string.
pub fn render_turn(text: &str, role: TurnRole) -> String {
    if text.is_empty() {
        return String::new();
    }
    let mut output = String::new();
    // Minimal user turn: a single coloured dot marker, no background box
    // (the grey box on past text reads as clutter).
    let marker_color = if role == TurnRole::VoiceUser {
        "\x1b[35m" // purple for voice
    } else {
        "\x1b[32m" // green for text
    };
    for (i, line) in text.lines().enumerate() {
        if i == 0 {
            output.push_str(&format!("{marker_color}●\x1b[0m {line}\n"));
        } else {
            output.push_str(&format!("  {line}\n"));
        }
    }
    // Blank line before the assistant reply.
    output.push('\n');
    output
}

/// Render only the provenance annotation footer (no full response re-render).
///
/// Use this when the response was already rendered incrementally and we only
/// need to append claim verification summary.
pub fn render_provenance_footer(claims: &[(usize, usize, u8, String)], strict: bool) -> String {
    let mut output = String::new();

    // If there are claimed items in strict mode, append a summary.
    if strict {
        let claimed: Vec<&(usize, usize, u8, String)> =
            claims.iter().filter(|c| c.2 == 2).collect();
        if !claimed.is_empty() {
            output.push_str(&format!(
                "\n\x1b[33m\x1b[1m⚠ {} unverified claim(s):\x1b[0m\n",
                claimed.len()
            ));
            for (_, _, _, ref claim_text) in &claimed {
                let preview: String = claim_text.chars().take(60).collect();
                output.push_str(&format!("  \x1b[33m⚠\x1b[0m {}\n", preview));
            }
        }
    }

    // Append per-claim status markers as a footer block.
    if !claims.is_empty() {
        let observed = claims.iter().filter(|c| c.2 == 0).count();
        let derived = claims.iter().filter(|c| c.2 == 1).count();
        let claimed_count = claims.iter().filter(|c| c.2 == 2).count();
        let recalled = claims.iter().filter(|c| c.2 == 3).count();

        let mut parts = Vec::new();
        if observed > 0 {
            parts.push(format!("\x1b[32m✓{}\x1b[0m", observed));
        }
        if derived > 0 {
            parts.push(format!("\x1b[34m~{}\x1b[0m", derived));
        }
        if claimed_count > 0 {
            parts.push(format!("\x1b[33m⚠{}\x1b[0m", claimed_count));
        }
        if recalled > 0 {
            parts.push(format!("\x1b[2m◇{}\x1b[0m", recalled));
        }
        output.push_str(&format!("\x1b[2mprovenance: {}\x1b[0m\n", parts.join(" ")));
    }

    output
}

/// Render an LLM response with termimad prose formatting and syntect code highlighting.
///
/// Splits input into prose and fenced code segments:
/// - Prose → termimad `term_text()` (headers, bold, lists, links)
/// - Code  → syntect highlighting with box borders
///
/// Returns a ready-to-print `String` (includes ANSI escapes).
pub fn render_response(text: &str) -> String {
    let mut output = String::new();
    let mut in_code_block = false;
    let mut code_lang = String::new();
    let mut code_content = String::new();
    let mut prose_buf = String::new();

    for line in text.lines() {
        if line.trim_start().starts_with("```") {
            if in_code_block {
                // End of code block — flush code through syntect
                output.push_str(&render_code_block(&code_content, &code_lang));
                code_content.clear();
                code_lang.clear();
                in_code_block = false;
            } else {
                // Start of code block — flush accumulated prose through termimad
                if !prose_buf.is_empty() {
                    output.push_str(&SKIN.term_text(&prose_buf).to_string());
                    prose_buf.clear();
                }
                in_code_block = true;
                code_lang = line
                    .trim_start()
                    .strip_prefix("```")
                    .unwrap_or("")
                    .trim()
                    .to_string();
            }
        } else if in_code_block {
            code_content.push_str(line);
            code_content.push('\n');
        } else {
            prose_buf.push_str(line);
            prose_buf.push('\n');
        }
    }

    // Flush remaining buffers
    if in_code_block && !code_content.is_empty() {
        output.push_str(&render_code_block(&code_content, &code_lang));
    }
    if !prose_buf.is_empty() {
        output.push_str(&SKIN.term_text(&prose_buf).to_string());
    }

    output
}

/// Render a single code block with syntax highlighting.
///
/// No side borders — code lines are clean for copy-paste.
/// Top/bottom rules sized to content width.
pub(crate) fn render_code_block(code: &str, lang: &str) -> String {
    let mut output = String::new();

    // Determine rule width from content (minimum 40, capped at 80)
    let max_line = code.lines().map(|l| l.len()).max().unwrap_or(0);
    let lang_display = if lang.is_empty() { "code" } else { lang };
    // label takes " lang " = lang.len() + 2 chars inside the rule
    let rule_width = (max_line + 2).clamp(40, 80);

    // Header: ─── lang ──────────
    let label_len = lang_display.len() + 2; // space + name + space
    let remaining = rule_width.saturating_sub(label_len);
    let left = 3.min(remaining);
    let right = remaining.saturating_sub(left);
    output.push_str(&format!(
        "\x1b[38;5;240m{} {} {}\x1b[0m\n",
        "─".repeat(left),
        lang_display,
        "─".repeat(right),
    ));

    // Try to get syntax for the language
    let syntax = SYNTAX_SET
        .find_syntax_by_token(lang)
        .or_else(|| SYNTAX_SET.find_syntax_by_extension(lang))
        .unwrap_or_else(|| SYNTAX_SET.find_syntax_plain_text());

    let theme = &THEME_SET.themes["base16-ocean.dark"];
    let mut highlighter = HighlightLines::new(syntax, theme);

    // Highlight each line — no side borders, just indented by 2 spaces
    for line in code.lines() {
        output.push_str("  ");
        match highlighter.highlight_line(line, &SYNTAX_SET) {
            Ok(ranges) => {
                let escaped = as_24_bit_terminal_escaped(&ranges[..], false);
                output.push_str(&escaped);
                output.push_str("\x1b[0m\n");
            }
            Err(_) => {
                output.push_str(&format!("\x1b[32m{}\x1b[0m\n", line));
            }
        }
    }

    // Footer: ──────────────────
    output.push_str(&format!(
        "\x1b[38;5;240m{}\x1b[0m\n",
        "─".repeat(rule_width)
    ));

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Strip ANSI escape sequences so we can assert on plain text content.
    fn strip_ansi(s: &str) -> String {
        let mut out = String::new();
        let mut chars = s.chars().peekable();
        while let Some(c) = chars.next() {
            if c == '\x1b' {
                // Skip until we hit a letter (end of escape sequence)
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
    fn test_render_simple_code_block() {
        let input = "```rust\nfn main() {}\n```";
        let output = render_response(input);
        assert!(output.contains("rust"));
        let plain = strip_ansi(&output);
        assert!(plain.contains("fn main"));
    }

    #[test]
    fn test_render_no_language() {
        let input = "```\nsome code\n```";
        let output = render_response(input);
        let plain = strip_ansi(&output);
        assert!(plain.contains("code"));
        assert!(plain.contains("some code"));
    }

    #[test]
    fn test_preserve_non_code() {
        let input = "Hello world\n\nSome text";
        let output = render_response(input);
        let plain = strip_ansi(&output);
        assert!(plain.contains("Hello world"));
        assert!(plain.contains("Some text"));
    }

    #[test]
    fn test_prose_only() {
        let input = "# Heading\n\nSome **bold** text and *italic* words.";
        let output = render_response(input);
        let plain = strip_ansi(&output);
        assert!(plain.contains("Heading"));
        assert!(plain.contains("bold"));
        assert!(plain.contains("italic"));
    }

    #[test]
    fn test_code_only() {
        let input = "```python\nprint('hello')\n```";
        let output = render_response(input);
        let plain = strip_ansi(&output);
        assert!(plain.contains("python"));
        assert!(plain.contains("print"));
        // Should have horizontal rule separators
        assert!(plain.contains("─"));
    }

    #[test]
    fn test_mixed_prose_and_code() {
        let input = "Here is an example:\n\n```rust\nlet x = 42;\n```\n\nThat was the code.";
        let output = render_response(input);
        let plain = strip_ansi(&output);
        assert!(plain.contains("example"));
        assert!(plain.contains("let x"));
        assert!(plain.contains("That was the code"));
        assert!(plain.contains("─"));
    }

    #[test]
    fn test_unclosed_code_block() {
        let input = "Some text\n```rust\nfn foo() {}";
        let output = render_response(input);
        let plain = strip_ansi(&output);
        assert!(plain.contains("fn foo"));
    }

    #[test]
    fn test_empty_input() {
        let output = render_response("");
        assert!(output.is_empty());
    }

    // --- render_turn tests ---

    #[test]
    fn test_render_turn_user_no_marker() {
        let output = render_turn("my question", TurnRole::User);
        let plain = strip_ansi(&output);
        assert!(!plain.contains('▞'), "user turn must NOT have ▞▞▞ marker");
    }

    #[test]
    fn test_render_turn_user_preserves_raw_text() {
        let output = render_turn("Hello **bold** text", TurnRole::User);
        let plain = strip_ansi(&output);
        // User text is raw (grey box), not markdown-rendered.
        assert!(
            plain.contains("**bold**"),
            "user text should be raw, not markdown-processed"
        );
    }

    #[test]
    fn test_render_turn_user_preserves_code_text() {
        let output = render_turn("Look:\n\n```rust\nlet x = 1;\n```", TurnRole::User);
        let plain = strip_ansi(&output);
        assert!(plain.contains("let x = 1;"));
        assert!(
            plain.contains("```rust"),
            "code fences should be preserved raw"
        );
    }

    #[test]
    fn test_render_turn_empty_returns_empty() {
        assert!(render_turn("", TurnRole::User).is_empty());
    }
}
