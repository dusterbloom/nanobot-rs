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

static SYNTAX_SET: Lazy<SyntaxSet> = Lazy::new(|| SyntaxSet::load_defaults_newlines());
static THEME_SET: Lazy<ThemeSet> = Lazy::new(ThemeSet::load_defaults);

/// Shared termimad skin for prose rendering.
static SKIN: Lazy<MadSkin> = Lazy::new(|| {
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnRole {
    User,
    Assistant,
}

/// Render a complete conversation turn with role marker and markdown formatting.
///
/// - **Assistant** turns get a bold white `И` marker line before the content.
/// - **User** turns are rendered through the same markdown pipeline (no marker).
/// - Empty input returns an empty string.
pub fn render_turn(text: &str, role: TurnRole) -> String {
    if text.is_empty() {
        return String::new();
    }
    let mut output = String::new();
    match role {
        TurnRole::User => {
            // Dark grey background per line (raw text, no markdown pipeline —
            // render_response resets would kill the bg color).
            for line in text.lines() {
                output.push_str(&format!("\x1b[48;5;236m {} \x1b[0m\n", line));
            }
            // Extra blank line after user text before assistant reply.
            output.push('\n');
        }
        TurnRole::Assistant => {
            // И marker on the same line as the start of the reply.
            // trim_start() strips the leading newline termimad injects.
            let rendered = render_response(text);
            let trimmed = rendered.trim_start();
            output.push_str(&format!("\x1b[1m\x1b[97mИ\x1b[0m {}", trimmed));
        }
    }
    output
}

/// Render a turn with provenance claim annotations.
///
/// Appends verification markers inline:
/// - ✓ (green) = Observed (verified by audit log)
/// - ~ (blue) = Derived (partial match)
/// - ⚠ (yellow) = Claimed (no matching tool call)
/// - ◇ (dim) = Recalled (from memory, not current tools)
///
/// If `strict` is true and there are Claimed items, appends a summary block.
///
/// Claims format: `(start, end, status, claim_text)` where status is:
/// - 0 = Observed
/// - 1 = Derived
/// - 2 = Claimed
/// - 3 = Recalled
pub fn render_turn_with_provenance(
    text: &str,
    role: TurnRole,
    claims: &[(usize, usize, u8, String)],
    strict: bool,
) -> String {
    if text.is_empty() {
        return String::new();
    }

    // Start with the normal rendered turn.
    let mut output = render_turn(text, role);

    // If there are claimed items in strict mode, append a summary.
    if strict {
        let claimed: Vec<&(usize, usize, u8, String)> = claims.iter().filter(|c| c.2 == 2).collect();
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
fn render_code_block(code: &str, lang: &str) -> String {
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
    output.push_str(&format!("\x1b[38;5;240m{}\x1b[0m\n", "─".repeat(rule_width)));

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
    fn test_render_turn_assistant_has_marker() {
        let output = render_turn("Hello world", TurnRole::Assistant);
        let plain = strip_ansi(&output);
        assert!(plain.contains('И'), "assistant turn must start with И marker");
        assert!(plain.contains("Hello world"));
    }

    #[test]
    fn test_render_turn_assistant_marker_before_content() {
        let output = render_turn("Some response", TurnRole::Assistant);
        let plain = strip_ansi(&output);
        let marker_pos = plain.find('И').expect("И must be present");
        let content_pos = plain.find("Some response").expect("content must be present");
        assert!(marker_pos < content_pos, "И must appear before content");
    }

    #[test]
    fn test_render_turn_user_no_marker() {
        let output = render_turn("my question", TurnRole::User);
        let plain = strip_ansi(&output);
        assert!(!plain.contains('И'), "user turn must NOT have И marker");
    }

    #[test]
    fn test_render_turn_user_preserves_raw_text() {
        let output = render_turn("Hello **bold** text", TurnRole::User);
        let plain = strip_ansi(&output);
        // User text is raw (grey box), not markdown-rendered.
        assert!(plain.contains("**bold**"), "user text should be raw, not markdown-processed");
    }

    #[test]
    fn test_render_turn_user_preserves_code_text() {
        let output = render_turn("Look:\n\n```rust\nlet x = 1;\n```", TurnRole::User);
        let plain = strip_ansi(&output);
        assert!(plain.contains("let x = 1;"));
        assert!(plain.contains("```rust"), "code fences should be preserved raw");
    }

    #[test]
    fn test_render_turn_assistant_renders_code_block() {
        let output = render_turn("Here:\n\n```python\nprint(1)\n```", TurnRole::Assistant);
        let plain = strip_ansi(&output);
        assert!(plain.contains('И'));
        assert!(plain.contains("print"));
        assert!(plain.contains("─"));
    }

    #[test]
    fn test_render_turn_empty_returns_empty() {
        assert!(render_turn("", TurnRole::Assistant).is_empty());
        assert!(render_turn("", TurnRole::User).is_empty());
    }

    // --- render_turn_with_provenance tests ---

    #[test]
    fn test_render_turn_with_provenance_empty() {
        let output = render_turn_with_provenance("", TurnRole::Assistant, &[], false);
        assert!(output.is_empty());
    }

    #[test]
    fn test_render_turn_with_provenance_no_claims() {
        let output = render_turn_with_provenance("Hello world", TurnRole::Assistant, &[], false);
        let plain = strip_ansi(&output);
        assert!(plain.contains("Hello world"));
        assert!(!plain.contains("provenance:"));
    }

    #[test]
    fn test_render_turn_with_provenance_with_claims() {
        let claims = vec![
            (0, 5, 0u8, "read file".to_string()),  // Observed
            (10, 20, 2u8, "deleted stuff".to_string()),  // Claimed
        ];
        let output = render_turn_with_provenance("Hello world with claims", TurnRole::Assistant, &claims, false);
        let plain = strip_ansi(&output);
        assert!(plain.contains("provenance:"));
    }

    #[test]
    fn test_render_turn_with_provenance_strict_mode() {
        let claims = vec![
            (0, 5, 2u8, "I deleted the files".to_string()),  // Claimed
        ];
        let output = render_turn_with_provenance("Test", TurnRole::Assistant, &claims, true);
        let plain = strip_ansi(&output);
        assert!(plain.contains("unverified"));
    }
}
