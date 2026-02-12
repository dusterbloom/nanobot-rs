/// Markdown rendering utilities for terminal output.

use super::ansi::terminal_width;

/// Build a styled termimad skin for rendering LLM markdown responses.
pub fn make_skin() -> termimad::MadSkin {
    use termimad::crossterm::style::Color;
    let mut skin = termimad::MadSkin::default_dark();
    skin.headers[0].set_fg(Color::White);
    skin.headers[1].set_fg(Color::White);
    skin.bold.set_fg(Color::White);
    skin.italic.set_fg(Color::AnsiValue(248)); // soft gray
    skin.inline_code.set_fg(Color::Cyan);
    skin.code_block.set_fg(Color::Cyan);
    skin
}

/// Render markdown through the skin with a 2-space left margin.
pub fn print_markdown(skin: &termimad::MadSkin, md: &str) {
    let width = terminal_width().saturating_sub(4);
    let rendered = format!("{}", skin.text(md, Some(width)));
    for line in rendered.lines() {
        println!("  {}", line.trim_end());
    }
}
