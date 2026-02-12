/// Component trait and built-in widget components for the TUI.

use super::ansi::*;

/// A renderable component that produces lines of terminal output.
pub trait Component {
    /// Render the component into lines of text (with ANSI escapes).
    /// `width` is the available character width (excluding margins).
    fn render(&self, width: usize) -> Vec<String>;
}

/// Turn header separator: ─── Role ─────────────────
pub struct TurnHeader<'a> {
    pub role: &'a str,
    pub is_user: bool,
}

impl<'a> Component for TurnHeader<'a> {
    fn render(&self, width: usize) -> Vec<String> {
        let prefix_len = 4; // "─── "
        let label_len = self.role.len() + 2;
        let rule_len = width.saturating_sub(prefix_len + label_len);
        let color = if self.is_user { WHITE } else { GREEN };
        vec![format!(
            "{DIM}───{RESET} {color}{BOLD}{}{RESET} {DIM}{}{RESET}",
            self.role,
            "─".repeat(rule_len),
        )]
    }
}

/// Inline tool execution indicator.
pub struct ToolIndicator<'a> {
    pub name: &'a str,
    pub running: bool,
}

impl<'a> Component for ToolIndicator<'a> {
    fn render(&self, _width: usize) -> Vec<String> {
        if self.running {
            vec![format!("{DIM}⚙ {}{RESET}", self.name)]
        } else {
            vec![format!("{DIM}✓ {}{RESET}", self.name)]
        }
    }
}

/// Braille-style thinking spinner (single line).
pub struct Spinner<'a> {
    pub message: &'a str,
    pub frame: usize,
}

impl<'a> Component for Spinner<'a> {
    fn render(&self, _width: usize) -> Vec<String> {
        const FRAMES: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
        let f = FRAMES[self.frame % FRAMES.len()];
        vec![format!("{DIM}{f} {}{RESET}", self.message)]
    }
}
