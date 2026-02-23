//! Slash command dispatch for gateway mode (Telegram/WhatsApp/Feishu).
//!
//! All output is plain text — no ANSI escape codes, because channels render
//! them as garbled characters.

use std::sync::atomic::Ordering;

use crate::agent::agent_loop::AgentLoopShared;
use crate::bus::events::InboundMessage;

/// Try to handle a gateway slash command. Returns `Some(response)` if handled,
/// `None` if the message should be forwarded to the LLM.
pub(crate) async fn dispatch(
    shared: &AgentLoopShared,
    msg: &InboundMessage,
) -> Option<String> {
    let content = msg.content.trim();
    if !content.starts_with('/') {
        return None;
    }
    // Strip Telegram bot mention suffix: "/status@my_bot" -> "/status"
    let (raw_cmd, arg) = content
        .split_once(' ')
        .map(|(c, a)| (c, a.trim()))
        .unwrap_or((content, ""));
    let cmd = raw_cmd.split('@').next().unwrap_or(raw_cmd);

    match cmd {
        "/start" => Some(cmd_start()),
        "/help" => Some(cmd_help()),
        "/status" => Some(cmd_status(shared).await),
        "/clear" => Some(cmd_clear(shared, &msg.session_key()).await),
        "/agents" => Some(cmd_agents(shared).await),
        "/kill" => Some(cmd_kill(shared, arg).await),
        "/think" => Some(cmd_think(shared, arg)),
        "/long" => Some(cmd_long(shared, arg)),
        "/context" => Some(cmd_context(shared)),
        "/memory" => Some(cmd_memory(shared, &msg.session_key())),
        _ => None, // Unknown — forward to LLM
    }
}

// ---------------------------------------------------------------------------
// Command implementations
// ---------------------------------------------------------------------------

fn cmd_start() -> String {
    "Hello! I'm your nanobot assistant. Send me a message or type /help for commands.".to_string()
}

fn cmd_help() -> String {
    [
        "Available commands:",
        "  /help    - Show this help",
        "  /status  - Current mode, model, context usage",
        "  /clear   - Clear working memory and history",
        "  /agents  - List running subagents",
        "  /kill <id> - Cancel a running subagent",
        "  /think [on|off|N] - Toggle extended thinking",
        "  /long [N] - Set long-response mode",
        "  /context - Context token breakdown",
        "  /memory  - Show working memory contents",
    ]
    .join("\n")
}

async fn cmd_status(shared: &AgentLoopShared) -> String {
    let core = shared.core_handle.swappable();
    let counters = &shared.core_handle.counters;
    let model = &core.model;
    let thinking = counters.thinking_budget.load(Ordering::Relaxed);
    let long_turns = counters.long_mode_turns.load(Ordering::Relaxed);
    let ctx_used = counters.last_context_used.load(Ordering::Relaxed);
    let ctx_max = counters.last_context_max.load(Ordering::Relaxed);
    let msg_count = counters.last_message_count.load(Ordering::Relaxed);
    let running = shared.subagents.get_running_count().await;
    let trio_state = counters.get_trio_state();

    let mut lines = vec![
        format!("Model: {}", model),
        format!("Trio: {:?}", trio_state),
        format!("Context: {}/{} tokens", ctx_used, ctx_max),
        format!("Messages in context: {}", msg_count),
    ];
    if thinking > 0 {
        lines.push(format!("Thinking: {} tokens", thinking));
    }
    if long_turns > 0 {
        lines.push(format!("Long mode: {} turns remaining", long_turns));
    }
    if running > 0 {
        lines.push(format!("Running subagents: {}", running));
    }
    lines.join("\n")
}

async fn cmd_clear(shared: &AgentLoopShared, session_key: &str) -> String {
    let core = shared.core_handle.swappable();
    core.working_memory.clear(session_key);
    core.sessions.clear_history(session_key).await;
    "Working memory and history cleared.".to_string()
}

async fn cmd_agents(shared: &AgentLoopShared) -> String {
    let running = shared.subagents.list_running().await;
    if running.is_empty() {
        return "No subagents running.".to_string();
    }
    let mut lines = vec![format!("{} subagent(s) running:", running.len())];
    for info in &running {
        lines.push(format!("  [{}] {}", info.task_id, info.label));
    }
    lines.join("\n")
}

async fn cmd_kill(shared: &AgentLoopShared, arg: &str) -> String {
    if arg.is_empty() {
        return "Usage: /kill <task-id>".to_string();
    }
    if shared.subagents.cancel(arg).await {
        format!("Cancelled subagent: {}", arg)
    } else {
        format!("No running subagent matching: {}", arg)
    }
}

fn cmd_think(shared: &AgentLoopShared, arg: &str) -> String {
    let counters = &shared.core_handle.counters;
    match arg {
        "" | "on" => {
            let budget = 10000u32;
            counters.thinking_budget.store(budget, Ordering::Relaxed);
            format!("Thinking enabled: {} tokens", budget)
        }
        "off" => {
            counters.thinking_budget.store(0, Ordering::Relaxed);
            "Thinking disabled.".to_string()
        }
        n => match n.parse::<u32>() {
            Ok(budget) => {
                counters.thinking_budget.store(budget, Ordering::Relaxed);
                format!("Thinking budget set to {} tokens", budget)
            }
            Err(_) => "Usage: /think [on|off|N]".to_string(),
        },
    }
}

fn cmd_long(shared: &AgentLoopShared, arg: &str) -> String {
    let counters = &shared.core_handle.counters;
    let turns: u32 = if arg.is_empty() {
        3
    } else {
        match arg.parse() {
            Ok(n) => n,
            Err(_) => return "Usage: /long [N]".to_string(),
        }
    };
    counters.long_mode_turns.store(turns, Ordering::Relaxed);
    if turns == 0 {
        "Long mode disabled.".to_string()
    } else {
        format!("Long mode enabled for {} turns.", turns)
    }
}

fn cmd_context(shared: &AgentLoopShared) -> String {
    let counters = &shared.core_handle.counters;
    let used = counters.last_context_used.load(Ordering::Relaxed);
    let max = counters.last_context_max.load(Ordering::Relaxed);
    let msgs = counters.last_message_count.load(Ordering::Relaxed);
    let wm = counters.last_working_memory_tokens.load(Ordering::Relaxed);
    let prompt = counters.last_actual_prompt_tokens.load(Ordering::Relaxed);
    let completion = counters.last_actual_completion_tokens.load(Ordering::Relaxed);

    let pct = if max > 0 {
        (used as f64 / max as f64 * 100.0) as u64
    } else {
        0
    };
    [
        format!("Context: {}/{} tokens ({}%)", used, max, pct),
        format!("Messages: {}", msgs),
        format!("Working memory: {} tokens", wm),
        format!("Last prompt: {} tokens", prompt),
        format!("Last completion: {} tokens", completion),
    ]
    .join("\n")
}

fn cmd_memory(shared: &AgentLoopShared, session_key: &str) -> String {
    let core = shared.core_handle.swappable();
    let content = core.working_memory.get_context(session_key, 4000);
    if content.is_empty() {
        "Working memory is empty.".to_string()
    } else {
        format!("Working memory:\n{}", content)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cmd_start() {
        let result = cmd_start();
        assert!(result.contains("Hello"));
    }

    #[test]
    fn test_cmd_help() {
        let help = cmd_help();
        assert!(help.contains("/help"));
        assert!(help.contains("/status"));
        assert!(help.contains("/clear"));
        assert!(help.contains("/think"));
        // No ANSI escapes
        assert!(!help.contains("\x1b["));
    }

    #[test]
    fn test_no_ansi_in_start() {
        assert!(!cmd_start().contains("\x1b["));
    }

    #[test]
    fn test_dispatch_parsing() {
        // Test that the Telegram bot mention stripping logic works
        let raw = "/status@mybot";
        let cmd = raw.split('@').next().unwrap_or(raw);
        assert_eq!(cmd, "/status");
    }

    #[test]
    fn test_dispatch_with_args() {
        let content = "/think 16000";
        let (raw_cmd, arg) = content
            .split_once(' ')
            .map(|(c, a)| (c, a.trim()))
            .unwrap_or((content, ""));
        let cmd = raw_cmd.split('@').next().unwrap_or(raw_cmd);
        assert_eq!(cmd, "/think");
        assert_eq!(arg, "16000");
    }

    #[test]
    fn test_dispatch_no_args() {
        let content = "/help";
        let (raw_cmd, arg) = content
            .split_once(' ')
            .map(|(c, a)| (c, a.trim()))
            .unwrap_or((content, ""));
        let cmd = raw_cmd.split('@').next().unwrap_or(raw_cmd);
        assert_eq!(cmd, "/help");
        assert_eq!(arg, "");
    }

    #[test]
    fn test_non_slash_returns_none_logic() {
        // Verify the dispatch guard logic
        let content = "hello world";
        assert!(!content.starts_with('/'));
    }

    #[test]
    fn test_unknown_command_not_matched() {
        // /unknown should not match any arm in dispatch
        let cmd = "/unknown";
        let matched = matches!(
            cmd,
            "/start"
                | "/help"
                | "/status"
                | "/clear"
                | "/agents"
                | "/kill"
                | "/think"
                | "/long"
                | "/context"
                | "/memory"
        );
        assert!(!matched, "/unknown should not be handled by dispatch");
    }
}
