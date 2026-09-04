// Error-protocol layer-3 backlog (docs/research/2026-08-06-error-conventions-and-host-bridge.md §3.6):
// the deny regime in Cargo.toml is live; this module still carries pre-existing
// violations of the lints below. Remove this allow as the module migrates onto
// the regime.
// Tracking: docs/error-protocol-backlog.md
#![allow(
    clippy::as_conversions,
    clippy::format_push_string,
    clippy::indexing_slicing,
    clippy::string_add
)]
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

/// Outcome of the repeated successful tool-call breaker for one executed round.
pub(crate) enum RepeatBreakerAction {
    /// Not a repeat (or first occurrence) — no action.
    Continue,
    /// Identical round repeated past threshold for the first time: nudge the
    /// model that tool results are already in context.
    Nudge,
    /// Repeated again after a nudge: stop the loop.
    Stop,
}

/// Pure decision for the repeated successful tool-call breaker.
///
/// `last`/`prev` are the normalized tool-call keys of the current and previous
/// executed rounds (empty if no tools ran). `prior_rounds` is the consecutive
/// repeat count carried from the previous round; `prior_nudged` whether we have
/// already nudged. Returns `(action, new_rounds, new_nudged)`.
///
/// The model "fires" tools without consuming their results when it dispatches
/// the exact same calls (name + args) two rounds in a row. We nudge once; if it
/// repeats again after the nudge we stop, rather than burning the whole budget.
pub(crate) fn evaluate_repeated_tool_round(
    last: &[String],
    prev: &[String],
    prior_rounds: u32,
    prior_nudged: bool,
    max: u32,
) -> (RepeatBreakerAction, u32, bool) {
    let is_repeat = !last.is_empty() && last == prev;
    let rounds = if is_repeat { prior_rounds + 1 } else { 0 };
    let nudged = if is_repeat { prior_nudged } else { false };
    if rounds >= max && nudged {
        (RepeatBreakerAction::Stop, rounds, nudged)
    } else if rounds >= max {
        // Nudge, and keep the counter one below threshold so a single further
        // repeat forces Stop.
        (RepeatBreakerAction::Nudge, max - 1, true)
    } else {
        (RepeatBreakerAction::Continue, rounds, nudged)
    }
}

const ADAPTIVE_TOOL_HEAVY_WINDOW_THRESHOLD: usize = 3;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum LocalArtifactAction {
    Simple,
    Rich,
}

fn strip_request_lead_in(lower: &str) -> &str {
    [
        "please ",
        "can you ",
        "could you ",
        "would you ",
        "i want you to ",
        "i need you to ",
        "i would like you to ",
        "i'd like you to ",
        "i want to ",
        "let's ",
        "let us ",
    ]
    .iter()
    .find_map(|prefix| lower.strip_prefix(prefix))
    .unwrap_or(lower)
}

pub(super) fn local_artifact_action(user_text: &str) -> Option<LocalArtifactAction> {
    let lower = user_text.trim().to_ascii_lowercase();
    let request = strip_request_lead_in(&lower);
    let action = [
        "build ",
        "create ",
        "edit ",
        "generate ",
        "implement ",
        "make ",
        "modify ",
        "save ",
        "update ",
        "write ",
    ]
    .iter()
    .any(|action| request.starts_with(action));
    let artifact = request.contains("file")
        || request.contains("folder")
        || request.contains("html")
        || request.contains(".html")
        || request.contains("javascript")
        || request.contains(" js")
        || request.contains(".js")
        || request.contains("css")
        || request.contains(".css")
        || request.contains("script")
        || request.contains("page")
        || request.contains("game")
        || request.contains("app")
        || request.contains("playable")
        || request.contains("code");
    if !action || !artifact {
        return None;
    }

    let rich_artifact = request.contains("html")
        || request.contains(".html")
        || request.contains("javascript")
        || request.contains(".js")
        || request.contains(".css")
        || request.contains("css")
        || request.contains("game")
        || request.contains("app")
        || request.contains("playable");

    Some(if rich_artifact {
        LocalArtifactAction::Rich
    } else {
        LocalArtifactAction::Simple
    })
}

fn strip_followup_lead_in(lower: &str) -> &str {
    [
        "also ", "and ", "now ", "then ", "next ", "instead ", "but ", "ok ", "okay ",
    ]
    .iter()
    .find_map(|prefix| lower.strip_prefix(prefix))
    .unwrap_or(lower)
}

fn contains_artifact_reference(request: &str) -> bool {
    [
        " it",
        " that",
        " this",
        "the file",
        "the folder",
        "the page",
        "the app",
        "the game",
        "the html",
        "the script",
        "button",
        "layout",
        "style",
        "color",
        "font",
        "screen",
        "counter",
        "score",
        "canvas",
        "animation",
        "responsive",
        "mobile",
        "header",
        "footer",
        "label",
        "menu",
        "toolbar",
        "panel",
        "modal",
        "background",
    ]
    .iter()
    .any(|needle| request.contains(needle))
}

fn looks_like_artifact_followup(user_text: &str) -> bool {
    let lower = user_text.trim().to_ascii_lowercase();
    let request = strip_followup_lead_in(strip_request_lead_in(strip_followup_lead_in(&lower)));
    if request.starts_with("continue")
        || request.starts_with("keep going")
        || request.starts_with("finish")
        || request.starts_with("complete")
        || request.contains("get it done")
    {
        return true;
    }
    let action = [
        "add ", "adjust ", "align ", "center ", "change ", "delete ", "edit ", "fix ", "make ",
        "modify ", "move ", "put ", "remove ", "rename ", "replace ", "save ", "set ", "style ",
        "tweak ", "update ", "use ",
    ]
    .iter()
    .any(|action| request.starts_with(action));
    if !action {
        return false;
    }

    contains_artifact_reference(request)
        || request.starts_with("add ")
        || request.starts_with("fix ")
        || request.starts_with("remove ")
        || request.starts_with("save ")
        || request.starts_with("update ")
}

pub(super) fn local_artifact_action_with_sticky(
    user_text: &str,
    sticky_action: Option<LocalArtifactAction>,
) -> Option<LocalArtifactAction> {
    local_artifact_action(user_text)
        .or_else(|| sticky_action.filter(|_| looks_like_artifact_followup(user_text)))
}

fn adaptive_fallback_max_tokens(
    base: u32,
    user_text: &str,
    recent_tool_calls: usize,
    cfg: &AdaptiveTokenConfig,
) -> u32 {
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
}

pub(super) fn adaptive_max_tokens(
    base: u32,
    had_long: bool,
    user_text: &str,
    recent_tool_calls: usize,
    is_local: bool,
    thinking_budget: Option<u32>,
    cfg: &AdaptiveTokenConfig,
) -> u32 {
    let artifact_action = is_local.then(|| local_artifact_action(user_text)).flatten();
    adaptive_max_tokens_for_artifact_action(
        base,
        had_long,
        user_text,
        recent_tool_calls,
        is_local,
        artifact_action,
        thinking_budget,
        cfg,
    )
}

pub(super) fn adaptive_max_tokens_for_artifact_action(
    base: u32,
    had_long: bool,
    user_text: &str,
    recent_tool_calls: usize,
    is_local: bool,
    local_artifact_action: Option<LocalArtifactAction>,
    thinking_budget: Option<u32>,
    cfg: &AdaptiveTokenConfig,
) -> u32 {
    let mut effective = if had_long {
        base.max(cfg.adaptive_long_mode_min_tokens)
    } else if is_local {
        match local_artifact_action {
            Some(LocalArtifactAction::Rich) => {
                // Browser-playable artifacts often need a large `write_file`
                // payload; the post-tool cap is too small and causes a plain
                // text dump instead of a completed file action.
                base.max(cfg.adaptive_long_mode_min_tokens)
            }
            Some(LocalArtifactAction::Simple) => {
                // Local models need enough decode room to finish structured
                // action payloads. Keep ordinary post-tool reporting
                // latency-focused below.
                base.max(cfg.adaptive_long_form_min_tokens)
            }
            None if recent_tool_calls > 0 => {
                // Local post-tool passes are latency-sensitive unless the
                // user's follow-up asks us to create or update an artifact.
                base.min(cfg.adaptive_tool_heavy_max_tokens)
                    .max(cfg.adaptive_tool_heavy_min_tokens)
            }
            None => adaptive_fallback_max_tokens(base, user_text, recent_tool_calls, cfg),
        }
    } else {
        adaptive_fallback_max_tokens(base, user_text, recent_tool_calls, cfg)
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
    let last_char = text_for_check.chars().last().unwrap_or('\0');
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
    use super::{
        adaptive_max_tokens, adaptive_max_tokens_for_artifact_action, evaluate_repeated_tool_round,
        local_artifact_action_with_sticky, should_strip_tools_for_trio, LocalArtifactAction,
        RepeatBreakerAction,
    };
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

    #[test]
    fn test_adaptive_max_tokens_local_post_tool_stays_tool_heavy() {
        let cfg = AdaptiveTokenConfig::default();
        let base = 1024u32;
        let long_form_trigger = "implement a complete multi-step edit after reading the file";

        let local_post_tool =
            adaptive_max_tokens(base, false, long_form_trigger, 1, true, None, &cfg);
        assert_eq!(
            local_post_tool, cfg.adaptive_long_form_min_tokens,
            "the artifact request keeps elevated headroom after a tool call"
        );

        let local_long_mode =
            adaptive_max_tokens(base, true, long_form_trigger, 1, true, None, &cfg);
        assert_eq!(
            local_long_mode, cfg.adaptive_long_mode_min_tokens,
            "/long still overrides the local post-tool cap"
        );
    }

    #[test]
    fn test_adaptive_max_tokens_escalates_local_file_and_tool_action_turns() {
        let cfg = AdaptiveTokenConfig::default();
        let base = 1024u32;

        let file_generation = adaptive_max_tokens(
            base,
            false,
            "Create a file named report.md",
            0,
            true,
            None,
            &cfg,
        );
        assert_eq!(
            file_generation, cfg.adaptive_long_form_min_tokens,
            "local file generation needs enough room to emit a complete tool call"
        );

        let post_tool =
            adaptive_max_tokens(base, false, "Summarize the result", 1, true, None, &cfg);
        assert_eq!(
            post_tool, base,
            "ordinary local post-tool reporting remains latency-focused"
        );

        let ordinary_local = adaptive_max_tokens(base, false, "What is Rust?", 0, true, None, &cfg);
        assert_eq!(ordinary_local, base);

        let cloud_file = adaptive_max_tokens(
            base,
            false,
            "Create a file named report.md",
            0,
            false,
            None,
            &cfg,
        );
        assert_eq!(cloud_file, base, "cloud long-form sizing remains unchanged");
    }

    #[test]
    fn test_adaptive_max_tokens_local_html_game_request_uses_long_mode_headroom() {
        let cfg = AdaptiveTokenConfig::default();
        let base = 2048u32;

        let tetris_request = adaptive_max_tokens(
            base,
            false,
            "I want you to create a colorfun and fun tetris game in a single HTML file at `~/Dev/tetris`",
            2,
            true,
            None,
            &cfg,
        );
        assert_eq!(
            tetris_request, cfg.adaptive_long_mode_min_tokens,
            "local HTML/game artifact creation needs enough room to complete a write_file call"
        );
    }

    #[test]
    fn test_local_artifact_action_with_sticky_recognizes_followups() {
        let sticky = Some(LocalArtifactAction::Rich);

        assert_eq!(
            local_artifact_action_with_sticky("make it harder", sticky),
            Some(LocalArtifactAction::Rich)
        );
        assert_eq!(
            local_artifact_action_with_sticky("also add a score counter", sticky),
            Some(LocalArtifactAction::Rich)
        );
        assert_eq!(
            local_artifact_action_with_sticky("continue please", sticky),
            Some(LocalArtifactAction::Rich)
        );
        assert_eq!(
            local_artifact_action_with_sticky("can you get it done", sticky),
            Some(LocalArtifactAction::Rich)
        );
        assert_eq!(
            local_artifact_action_with_sticky("update the file", Some(LocalArtifactAction::Simple)),
            Some(LocalArtifactAction::Simple)
        );
    }

    #[test]
    fn test_local_artifact_action_with_sticky_stays_bounded_to_followup_edits() {
        let sticky = Some(LocalArtifactAction::Rich);

        assert_eq!(
            local_artifact_action_with_sticky("make it harder", None),
            None
        );
        assert_eq!(
            local_artifact_action_with_sticky("what time is it?", sticky),
            None
        );
        assert_eq!(
            local_artifact_action_with_sticky("make me a sandwich", sticky),
            None
        );
    }

    #[test]
    fn test_adaptive_max_tokens_uses_sticky_rich_artifact_action() {
        let cfg = AdaptiveTokenConfig::default();
        let base = 1024u32;

        let out = adaptive_max_tokens_for_artifact_action(
            base,
            false,
            "make it harder",
            1,
            true,
            local_artifact_action_with_sticky("make it harder", Some(LocalArtifactAction::Rich)),
            None,
            &cfg,
        );
        assert_eq!(out, cfg.adaptive_long_mode_min_tokens);
    }

    // -----------------------------------------------------------------------
    // evaluate_repeated_tool_round — pins the repeated-tool-call breaker
    // decision (shared.rs:1135-1186). Two identical successful rounds in a row
    // nudge; a third after the nudge stops.
    // -----------------------------------------------------------------------

    #[test]
    fn test_repeat_breaker_first_identical_round_is_continue() {
        // R1 dispatches [recall:X]; prev is empty → not a repeat.
        let (action, rounds, nudged) =
            evaluate_repeated_tool_round(&["recall:X".to_string()], &[], 0, false, 2);
        assert!(matches!(action, RepeatBreakerAction::Continue));
        assert_eq!(rounds, 0);
        assert!(!nudged);
    }

    #[test]
    fn test_repeat_breaker_two_identical_rounds_nudges() {
        // R2 dispatches [recall:X] again, identical to prev → count hits max → Nudge.
        let (action, rounds, nudged) = evaluate_repeated_tool_round(
            &["recall:X".to_string()],
            &["recall:X".to_string()],
            1,
            false,
            2,
        );
        assert!(matches!(action, RepeatBreakerAction::Nudge));
        // Counter held one below threshold so the next repeat forces Stop.
        assert_eq!(rounds, 1);
        assert!(nudged);
    }

    #[test]
    fn test_repeat_breaker_third_identical_round_after_nudge_stops() {
        // R3 identical again, already nudged → Stop.
        let (action, _rounds, nudged) = evaluate_repeated_tool_round(
            &["recall:X".to_string()],
            &["recall:X".to_string()],
            1,
            true,
            2,
        );
        assert!(matches!(action, RepeatBreakerAction::Stop));
        assert!(nudged);
    }

    #[test]
    fn test_repeat_breaker_different_round_resets() {
        // A round with different tools breaks the streak.
        let (action, rounds, nudged) = evaluate_repeated_tool_round(
            &["recall:Y".to_string()],
            &["recall:X".to_string()],
            1,
            true,
            2,
        );
        assert!(matches!(action, RepeatBreakerAction::Continue));
        assert_eq!(rounds, 0);
        assert!(!nudged, "streak must reset when the round changes");
    }

    #[test]
    fn test_repeat_breaker_no_tools_round_is_not_a_repeat() {
        // A round that executed no tools must not trigger the breaker.
        let (action, rounds, nudged) =
            evaluate_repeated_tool_round(&[], &["recall:X".to_string()], 1, false, 2);
        assert!(matches!(action, RepeatBreakerAction::Continue));
        assert_eq!(rounds, 0);
        assert!(!nudged);
    }
}
