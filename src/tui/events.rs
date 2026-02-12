/// Rich TUI events emitted by the agent loop for display purposes.

use std::time::Duration;

/// Events consumed by the TUI display task to show real-time agent activity.
#[derive(Debug, Clone)]
pub enum TuiEvent {
    /// A text delta from the LLM response stream.
    TextDelta(String),
    /// A tool execution has started.
    ToolStart {
        name: String,
        id: String,
    },
    /// A tool execution has completed.
    ToolComplete {
        name: String,
        id: String,
        success: bool,
        duration: Duration,
    },
    /// The LLM is "thinking" (stream started but no text yet).
    Thinking,
    /// The response is fully complete.
    Done,
}
