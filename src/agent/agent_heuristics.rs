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

// ============================================================================
// Wave 0 coverage net — pins current `is_local` branch outputs.
//
// Phase 09 plan:
//   .planning/phases/09-runtime-mode-spine/00-wave-0-coverage-PLAN.md
//
// These tests capture the output of every `is_local` branch read in this file
// (agent_heuristics.rs:73-83 and :119-127) so the Wave 1 → Wave 3 refactor to
// `RuntimeMode` can be audited via `cargo test --lib`. The two helpers in this
// module (`should_strip_tools_for_trio`, `adaptive_max_tokens`) are called
// from `agent_shared.rs:942-951` and `agent_shared.rs:1351` respectively, so
// these tests ALSO anchor two of the `agent_shared.rs` branch sites that
// couldn't be unit-tested in place.
// ============================================================================
#[cfg(test)]
mod tests {
    use super::{adaptive_max_tokens, should_strip_tools_for_trio};
    use crate::config::schema::AdaptiveTokenConfig;

    // -----------------------------------------------------------------------
    // should_strip_tools_for_trio — pins agent_heuristics.rs:73-83
    // (trio-strip AND-chain: is_local, strict_no_tools_main, router health,
    //  circuit breaker availability). Also covers agent_shared.rs:942-951.
    // -----------------------------------------------------------------------

    #[test]
    fn test_should_strip_tools_for_trio_is_local_gate() {
        // pins agent_heuristics.rs:73-83 — trio strip AND-chain
        // Cloud (is_local=false) must NEVER strip tools regardless of the
        // other three permissions (cloud providers handle tools natively).
        assert!(
            !should_strip_tools_for_trio(false, true, true, true),
            "cloud must never strip tools even when every downstream permission is granted"
        );
        assert!(
            !should_strip_tools_for_trio(false, false, false, false),
            "cloud with no permissions → still never strip"
        );

        // Local with all permissions → strip.
        assert!(
            should_strip_tools_for_trio(true, true, true, true),
            "local + strict + healthy + cb-available → strip"
        );

        // Local + any permission missing → do NOT strip (AND-chain).
        assert!(
            !should_strip_tools_for_trio(true, false, true, true),
            "local without strict → do not strip"
        );
        assert!(
            !should_strip_tools_for_trio(true, true, false, true),
            "local with degraded router probe → do not strip"
        );
        assert!(
            !should_strip_tools_for_trio(true, true, true, false),
            "local with tripped circuit breaker → do not strip"
        );
    }

    #[test]
    fn test_should_strip_tools_for_trio_truth_table() {
        // pins agent_heuristics.rs:80 — boolean AND of four flags
        // Exhaustive 16-row truth table: the output is `true` iff all four
        // inputs are `true`.
        for mask in 0u8..16 {
            let is_local = mask & 0b1000 != 0;
            let strict = mask & 0b0100 != 0;
            let healthy = mask & 0b0010 != 0;
            let cb = mask & 0b0001 != 0;
            let expected = is_local && strict && healthy && cb;
            let got = should_strip_tools_for_trio(is_local, strict, healthy, cb);
            assert_eq!(
                got, expected,
                "mask={:04b} — is_local={} strict={} healthy={} cb={} → expected {}",
                mask, is_local, strict, healthy, cb, expected
            );
        }
    }

    // -----------------------------------------------------------------------
    // adaptive_max_tokens — pins agent_heuristics.rs:92, :119-127
    // (thinking budget is ADDED on top of max_tokens for local models;
    //  excluded for cloud). Also covers agent_shared.rs:1351.
    // -----------------------------------------------------------------------

    #[test]
    fn test_adaptive_max_tokens_is_local_budget() {
        // pins agent_heuristics.rs:119-127 — thinking-budget addition is
        // local-only. Base tokens: 2000. Thinking budget: 1000.
        let cfg = AdaptiveTokenConfig::default();
        let base = 2000u32;
        let thinking = Some(1000u32);

        // Local with thinking budget → base + budget (capped at 32_768).
        let local_total = adaptive_max_tokens(base, false, "short msg", 0, true, thinking, &cfg);
        assert_eq!(
            local_total, 3000,
            "local + thinking=1000 on top of base=2000 → 3000"
        );

        // Cloud with the same inputs → no addition (thinking is included
        // inside max_tokens on cloud).
        let cloud_total = adaptive_max_tokens(base, false, "short msg", 0, false, thinking, &cfg);
        assert_eq!(
            cloud_total, base,
            "cloud does not add thinking budget on top of max_tokens"
        );

        // Local with thinking = None → no addition.
        let local_no_think = adaptive_max_tokens(base, false, "short msg", 0, true, None, &cfg);
        assert_eq!(
            local_no_think, base,
            "local + no thinking budget → base unchanged"
        );
    }

    #[test]
    fn test_adaptive_max_tokens_local_thinking_clamps_to_32k() {
        // pins agent_heuristics.rs:125 — .min(32_768) cap on the local path
        // keeps us within typical local-model context limits even when base +
        // thinking would otherwise exceed it.
        let cfg = AdaptiveTokenConfig::default();
        let got = adaptive_max_tokens(40_000, false, "x", 0, true, Some(10_000), &cfg);
        assert_eq!(got, 32_768, "local thinking total must clamp to 32_768");

        // Cloud path ignores thinking addition entirely, so no clamp needed;
        // pin that the cloud branch does not re-clamp to 32K.
        let cloud_got = adaptive_max_tokens(40_000, false, "x", 0, false, Some(10_000), &cfg);
        assert_eq!(
            cloud_got, 40_000,
            "cloud path leaves base untouched — no 32K clamp"
        );
    }

    #[test]
    fn test_adaptive_max_tokens_is_local_orthogonal_to_longform() {
        // pins agent_heuristics.rs:92 — `is_local` is an independent axis
        // from the long-form / tool-heavy / long-mode branches above. A
        // long-form bump followed by local-thinking addition must compose
        // correctly (bump first, then +thinking on local).
        let cfg = AdaptiveTokenConfig::default();
        let long_form_trigger = "explain in detail how this works";
        let base = 1000u32;
        let thinking = Some(500u32);

        // Cloud long-form: bumped to adaptive_long_form_min_tokens (4096 default).
        let cloud_long =
            adaptive_max_tokens(base, false, long_form_trigger, 0, false, thinking, &cfg);
        assert_eq!(
            cloud_long, cfg.adaptive_long_form_min_tokens,
            "cloud long-form bump → adaptive_long_form_min_tokens; no thinking add"
        );

        // Local long-form: bump first, then add thinking budget on top.
        let local_long =
            adaptive_max_tokens(base, false, long_form_trigger, 0, true, thinking, &cfg);
        assert_eq!(
            local_long,
            cfg.adaptive_long_form_min_tokens + 500,
            "local long-form: bump then +thinking"
        );
    }
}
