//! Pure helper functions for the agent loop (no IO — fully unit-testable).
//!
//! Extracted from `agent_loop.rs` as a `#[path]` submodule.

use serde_json::Value;
use tracing::instrument;

use crate::agent::protocol::ConversationProtocol;
use crate::agent::turn::turn_from_legacy;
use crate::config::schema::AdaptiveTokenConfig;

// ---------------------------------------------------------------------------
// Pure helpers (no IO — fully unit-testable)
// ---------------------------------------------------------------------------

pub(super) fn last_user_message(messages: &[serde_json::Value]) -> Option<String> {
    messages
        .iter()
        .rev()
        .find(|m| m.get("role").and_then(|r| r.as_str()) == Some("user"))
        .and_then(|m| m.get("content").and_then(|c| c.as_str()))
        .map(|s| s.to_string())
}

/// Convert raw wire-format messages to canonical `Turn` sequence, then render
/// via the given protocol to produce a clean wire format for the LLM call.
///
/// - Position 0 is expected to be `role:system`; it is extracted and passed as
///   the `system` argument to `protocol.render()`.
/// - Any `_turn` / `_synthetic` metadata tags on raw messages are not forwarded
///   to the wire output (they are internal-only fields used for trimming).
pub(super) fn render_via_protocol(
    protocol: &dyn ConversationProtocol,
    messages: &[Value],
) -> Vec<Value> {
    // Extract system prompt from the leading system message (if present).
    let system = messages
        .first()
        .filter(|m| m.get("role").and_then(|r| r.as_str()) == Some("system"))
        .and_then(|m| m.get("content").and_then(|c| c.as_str()))
        .unwrap_or("")
        .to_string();

    let non_system_start = if messages
        .first()
        .map(|m| m.get("role").and_then(|r| r.as_str()) == Some("system"))
        .unwrap_or(false)
    {
        1
    } else {
        0
    };

    let turns: Vec<_> = messages[non_system_start..]
        .iter()
        .filter_map(|m| turn_from_legacy(m))
        .collect();

    protocol.render(&system, &turns)
}

/// Decide whether trio routing is healthy enough to strip tools from the main model.
/// Pure function: takes health status as booleans, returns true if tools should be stripped.
#[instrument(
    name = "should_strip_tools_for_trio",
    fields(
        is_local,
        strict_no_tools_main,
        router_probe_healthy,
        circuit_breaker_available,
    )
)]
pub(super) fn should_strip_tools_for_trio(
    is_local: bool,
    strict_no_tools_main: bool,
    router_probe_healthy: bool,
    circuit_breaker_available: bool,
) -> bool {
    let result =
        is_local && strict_no_tools_main && router_probe_healthy && circuit_breaker_available;
    tracing::debug!(strip_tools = result, "trio_strip_decision");
    result
}

const ADAPTIVE_TOOL_HEAVY_WINDOW_THRESHOLD: usize = 3;

pub(super) fn adaptive_max_tokens(
    base: u32,
    had_long: bool,
    user_text: &str,
    recent_tool_calls: usize,
    is_local: bool,
    thinking_budget: Option<u32>,
    cfg: &AdaptiveTokenConfig,
) -> u32 {
    let mut effective = if had_long {
        base.max(cfg.adaptive_long_mode_min_tokens)
    } else {
        let lower = user_text.to_lowercase();
        let is_long_form = lower.contains("explain in detail")
            || lower.contains("write a ")
            || lower.contains("create a script")
            || lower.contains("write code")
            || lower.contains("implement ")
            || lower.contains("full example")
            || lower.starts_with("write ")
            || user_text.len() > cfg.adaptive_long_form_trigger_chars as usize;

        if is_long_form {
            base.max(cfg.adaptive_long_form_min_tokens)
        } else if recent_tool_calls > ADAPTIVE_TOOL_HEAVY_WINDOW_THRESHOLD {
            base.min(cfg.adaptive_tool_heavy_max_tokens)
                .max(cfg.adaptive_tool_heavy_min_tokens)
        } else {
            base
        }
    };

    if is_local {
        if let Some(budget) = thinking_budget {
            // Reasoning models burn thinking tokens INSIDE the max_tokens budget.
            // Add the thinking budget on top so the model has room for both
            // thinking AND completion output, capped at 32K to stay within
            // typical local model context limits.
            effective = (effective + budget).min(32768);
        }
    }

    effective
}

/// Proactive recall: search the knowledge store for context relevant to the user's message.
/// Returns a formatted string of relevant snippets, or None if nothing useful was found.
/// Silently returns None on any error (knowledge store missing, etc.).
#[allow(dead_code)]
pub(super) fn proactive_recall(user_message: &str) -> Option<String> {
    // Skip very short messages (greetings, single words).
    if user_message.len() < 15 {
        return None;
    }

    let store = crate::agent::knowledge_store::KnowledgeStore::open_default().ok()?;
    let hits = store.search(user_message, 3).ok()?;

    if hits.is_empty() {
        return None;
    }

    let mut output = String::new();
    for hit in &hits {
        // Truncate long snippets.
        let snippet: String = if hit.snippet.len() > 300 {
            hit.snippet.chars().take(300).collect::<String>() + "..."
        } else {
            hit.snippet.clone()
        };
        output.push_str(&format!(
            "**{}** (chunk {}): {}\n",
            hit.source_name, hit.chunk_idx, snippet
        ));
    }

    Some(output.trim_end().to_string())
}

// ============================================================================
// Heuristic helpers
// ============================================================================

/// Detect responses that appear truncated despite finish_reason being "stop".
///
/// This catches cases where the model stops at special characters (e.g., backtick)
/// due to tokenizer/stop-token issues in local model servers.
pub(crate) fn appears_incomplete(content: &str) -> bool {
    let trimmed = content.trim_end();
    if trimmed.is_empty() {
        return false;
    }

    // Ends mid-sentence (no terminal punctuation, not a code block fence).
    // Strip trailing emoji (non-ASCII symbols like rust crab, smiley, etc.) and any surrounding
    // whitespace before checking the "real" last character — an emoji after a
    // period must not trigger continuation.
    let stripped = trimmed
        .trim_end_matches(|c: char| !c.is_alphanumeric() && !c.is_ascii())
        .trim_end();
    let text_for_check = if stripped.is_empty() {
        trimmed
    } else {
        stripped
    };
    let last_char = text_for_check.chars().last().unwrap();
    let ends_mid_sentence = !matches!(
        last_char,
        '.' | '!' | '?' | ':' | '"' | '\'' | ')' | ']' | '}' | '`'
    ) && !trimmed.ends_with("```");

    // Has unclosed backtick (odd number of backticks on the last line).
    // Exclude code fences (lines that are purely backticks, e.g. "```") —
    // those are block delimiters, not inline code markers.
    let last_line = trimmed.lines().last().unwrap_or("");
    let last_line_trimmed = last_line.trim();
    let is_code_fence = !last_line_trimmed.is_empty()
        && last_line_trimmed.chars().all(|c| c == '`')
        && last_line_trimmed.len() >= 3;
    let backtick_count = last_line.chars().filter(|&c| c == '`').count();
    let unclosed_backtick = !is_code_fence && backtick_count % 2 != 0;

    // Has unclosed parenthesis/bracket on the last line
    let unclosed_paren = last_line.chars().filter(|&c| c == '(').count()
        > last_line.chars().filter(|&c| c == ')').count();

    unclosed_backtick || (ends_mid_sentence && trimmed.len() > 20) || unclosed_paren
}
