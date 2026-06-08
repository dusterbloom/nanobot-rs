//! Pure filtering functions for session message history.
//!
//! Extracted from the JSONL `SessionManager` for reuse by the SQLite
//! `SessionDb`. All functions are pure — no I/O.

use serde_json::Value;
use tracing::warn;

/// Estimate tokens for a single JSON message (cheap heuristic: chars / 4).
fn estimate_msg_tokens(m: &Value) -> usize {
    let content_len = m
        .get("content")
        .and_then(|v| v.as_str())
        .map(|s| s.len())
        .unwrap_or(0);
    let tc_len = m
        .get("tool_calls")
        .map(|v| v.to_string().len())
        .unwrap_or(0);
    // ~4 chars per token is a conservative estimate (tiktoken cl100k_base).
    (content_len + tc_len + 20) / 4 // +20 for role/JSON overhead
}

/// A real conversational user turn: a `role: "user"` message that is NOT an
/// injected synthetic scaffolding nudge (grounding, format-anchor, response
/// boundary, iteration notice, etc.). Only these count as turns and serve as
/// history-drop boundaries — counting synthetic nudges as turns would head-drop
/// real history on every reload and diverge the prompt prefix, defeating the
/// server-side prefix cache.
fn is_real_user_turn(msg: &Value) -> bool {
    msg.get("role").and_then(|r| r.as_str()) == Some("user")
        && !msg
            .get("_synthetic")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
}

/// Advance an index past leading `role: "tool"` messages whose parent
/// `assistant+tool_calls` is outside the window. Sending a lone tool result
/// to the LLM is a protocol error.
fn skip_leading_orphan_tools(messages: &[Value], start: usize) -> usize {
    let mut i = start;
    while i < messages.len() {
        if messages[i].get("role").and_then(|r| r.as_str()) == Some("tool") {
            i += 1;
        } else {
            break;
        }
    }
    i
}

/// Filter messages: respect clear markers, skip orphaned tool results,
/// filter synthetics, apply turn limit, token budget, and map to wire format.
///
/// This is the primary entry point — it applies all filtering stages
/// in sequence:
///
/// 1. `max_messages` window — take the last N messages
/// 2. Clear markers — only show messages after the last `role: "clear"` marker
/// 3. Orphaned tool results — skip leading `role: "tool"` messages at the
///    window boundary when their parent assistant+tool_calls is outside the window
/// 4. Turn limit — keep only the last `max_turns` user-assistant pairs
/// 5. Per-message filter/map — strip synthetics, clear markers, summaries;
///    copy role, content, tool_calls, tool_call_id, name, _turn to wire format
/// 6. Token budget — drop oldest messages until total tokens ≤ budget
///    (prevents context bombs when sessions accumulate large tool results)
pub fn filter_history(messages: &[Value], max_messages: usize, max_turns: usize) -> Vec<Value> {
    // Stage 1: max_messages window — start index into `messages`.
    // max_messages=0 means "no limit".
    let start = if max_messages > 0 && messages.len() > max_messages {
        messages.len() - max_messages
    } else {
        0
    };

    // Stage 2: respect logical session clears. Only show messages after the
    // most recent clear marker. Markers are preserved on disk for an
    // append-only audit trail but must not appear in the runtime wire history.
    let clear_start = messages
        .iter()
        .rposition(|m| m.get("role").and_then(|r| r.as_str()) == Some("clear"))
        .map(|i| i + 1)
        .unwrap_or(0);

    // Stage 3: advance past orphaned tool results at the window boundary.
    let mut safe_start = skip_leading_orphan_tools(messages, start.max(clear_start));

    // Stage 4: turn-based limit. Scan backward from the end counting REAL user
    // messages (not synthetic nudges) as turn boundaries. If more than
    // `max_turns` real turns are seen, advance safe_start to the oldest kept
    // real-turn start. Because the boundary is always a real user-turn start,
    // dropping happens at whole-turn granularity: the kept prefix only changes
    // when an entire oldest turn ages out, not on every reload.
    if max_turns > 0 {
        let mut turns_seen = 0;
        let mut turn_start = safe_start;
        for i in (safe_start..messages.len()).rev() {
            if is_real_user_turn(&messages[i]) {
                turns_seen += 1;
                if turns_seen > max_turns {
                    break;
                }
                turn_start = i;
            }
        }
        safe_start = safe_start.max(turn_start);
    }

    // Stage 5: filter and map each surviving message to wire format.
    let mapped: Vec<Value> = messages[safe_start..]
        .iter()
        .filter(|m| {
            // Skip synthetic router/specialist injections — ephemeral to the
            // turn they were created in.
            !m.get("_synthetic").and_then(|v| v.as_bool()).unwrap_or(false)
                // Skip clear markers; they must not appear in the wire history.
                && m.get("role").and_then(|v| v.as_str()) != Some("clear")
                // Skip internal LCM summary entries — not valid wire format.
                && m.get("role").and_then(|r| r.as_str()) != Some("summary")
        })
        .map(|m| {
            let role = m.get("role").and_then(|v| v.as_str()).unwrap_or("user");
            let mut msg = serde_json::json!({
                "role": role,
                "content": m.get("content").and_then(|v| v.as_str()).unwrap_or(""),
            });
            // Preserve tool_calls on assistant messages.
            if let Some(tc) = m.get("tool_calls") {
                msg["tool_calls"] = tc.clone();
            }
            // Preserve tool_call_id on tool result messages.
            if let Some(id) = m.get("tool_call_id") {
                msg["tool_call_id"] = id.clone();
            }
            // Preserve name on tool result messages.
            if let Some(name) = m.get("name") {
                msg["name"] = name.clone();
            }
            // Preserve _turn field (used by age-based eviction).
            if let Some(turn) = m.get("_turn") {
                msg["_turn"] = turn.clone();
            }
            msg
        })
        .collect();

    // Stage 6: Token budget — prevent context bombs from sessions that
    // accumulated large tool results. Walk backward from the end, keeping
    // messages until the cumulative token count exceeds the budget.
    // Budget is derived from max_messages: since history_limit() already
    // calculates "30% of context / 150 tokens per message", we use
    // max_messages * 150 as the token ceiling. This gives history at most
    // 30% of the context window in tokens, not just in message count.
    let token_budget = max_messages.saturating_mul(150);
    if token_budget == 0 {
        // max_messages=0 means "no limit" in Stage 1; honour that here too.
        return mapped;
    }
    let total_tokens: usize = mapped.iter().map(|m| estimate_msg_tokens(m)).sum();
    if total_tokens <= token_budget {
        return mapped;
    }

    // Over budget. Drop from the front (oldest), skip orphaned tool results.
    let mut cumulative = 0;
    let mut keep_from = mapped.len();
    for i in (0..mapped.len()).rev() {
        let msg_tokens = estimate_msg_tokens(&mapped[i]);
        if cumulative + msg_tokens > token_budget {
            break;
        }
        cumulative += msg_tokens;
        keep_from = i;
    }
    let keep_from = skip_leading_orphan_tools(&mapped, keep_from);
    // Snap the cut forward to the next real user-turn boundary so we drop whole
    // oldest turns rather than slicing mid-turn. This keeps the kept-history
    // prefix stable across reloads (the cut only advances a full turn at a time)
    // and avoids starting the wire history on an assistant/tool message.
    let keep_from = mapped[keep_from..]
        .iter()
        .position(is_real_user_turn)
        .map(|off| keep_from + off)
        .unwrap_or(keep_from);
    let dropped = mapped.len() - (mapped.len() - keep_from);
    if dropped > 0 {
        warn!(
            dropped_messages = dropped,
            budget_tokens = token_budget,
            total_tokens = total_tokens,
            kept_messages = mapped.len() - keep_from,
            "token_budget_trim: dropped oldest messages from session history"
        );
    }
    mapped[keep_from..].to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    fn user(content: &str) -> Value {
        json!({"role": "user", "content": content})
    }

    fn assistant(content: &str) -> Value {
        json!({"role": "assistant", "content": content})
    }

    fn tool_call_assistant(id: &str) -> Value {
        json!({
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": id, "type": "function", "function": {"name": "exec", "arguments": "{}"}}]
        })
    }

    fn tool_result(id: &str) -> Value {
        json!({"role": "tool", "tool_call_id": id, "name": "exec", "content": "ok"})
    }

    fn clear() -> Value {
        json!({"role": "clear", "timestamp": "2026-01-01T00:00:00Z"})
    }

    fn synthetic(content: &str) -> Value {
        json!({"role": "user", "content": content, "_synthetic": true})
    }

    fn summary(content: &str) -> Value {
        json!({"role": "summary", "content": content})
    }

    fn role_of(m: &Value) -> &str {
        m.get("role").and_then(|r| r.as_str()).unwrap_or("")
    }

    // ------------------------------------------------------------------
    // Basic round-trip
    // ------------------------------------------------------------------

    #[test]
    fn test_empty_input_returns_empty() {
        let result = filter_history(&[], 100, 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_basic_round_trip() {
        let messages = vec![user("hello"), assistant("hi"), user("how are you?")];
        let result = filter_history(&messages, 100, 0);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0]["content"], "hello");
        assert_eq!(result[1]["content"], "hi");
        assert_eq!(result[2]["content"], "how are you?");
    }

    // ------------------------------------------------------------------
    // max_messages windowing
    // ------------------------------------------------------------------

    #[test]
    fn test_max_messages_windowing() {
        let messages = vec![
            user("q1"),
            assistant("a1"),
            user("q2"),
            assistant("a2"),
            user("q3"),
            assistant("a3"),
        ];
        // Window of 2 returns only the last 2 messages.
        let result = filter_history(&messages, 2, 0);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0]["content"], "q3");
        assert_eq!(result[1]["content"], "a3");
    }

    #[test]
    fn test_max_messages_larger_than_slice_returns_all() {
        let messages = vec![user("a"), assistant("b")];
        let result = filter_history(&messages, 1000, 0);
        assert_eq!(result.len(), 2);
    }

    // ------------------------------------------------------------------
    // Clear marker
    // ------------------------------------------------------------------

    #[test]
    fn test_clear_marker_respected() {
        let messages = vec![
            user("old question"),
            assistant("old answer"),
            clear(),
            user("new question"),
            assistant("new answer"),
        ];
        let result = filter_history(&messages, 100, 0);
        assert_eq!(
            result.len(),
            2,
            "only messages after clear should be returned"
        );
        assert_eq!(result[0]["content"], "new question");
        assert_eq!(result[1]["content"], "new answer");
        assert!(
            result.iter().all(|m| role_of(m) != "clear"),
            "clear marker must not appear in output"
        );
    }

    #[test]
    fn test_most_recent_clear_marker_used() {
        // Two clear markers — only the last one matters.
        let messages = vec![
            user("very old"),
            clear(),
            user("old"),
            assistant("old answer"),
            clear(),
            user("fresh"),
            assistant("fresh answer"),
        ];
        let result = filter_history(&messages, 100, 0);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0]["content"], "fresh");
        assert_eq!(result[1]["content"], "fresh answer");
    }

    #[test]
    fn test_clear_marker_at_end_returns_empty() {
        let messages = vec![user("q"), assistant("a"), clear()];
        let result = filter_history(&messages, 100, 0);
        assert!(
            result.is_empty(),
            "nothing after clear marker should yield empty history"
        );
    }

    // ------------------------------------------------------------------
    // Orphaned tool results
    // ------------------------------------------------------------------

    #[test]
    fn test_orphaned_tool_results_skipped_at_boundary() {
        // user → assistant+tc → tool → assistant → user → assistant
        // Window of 4 starts at index 2 (the tool result) — that is orphaned.
        let messages = vec![
            user("q1"),
            tool_call_assistant("tc_1"),
            tool_result("tc_1"),
            assistant("Done"),
            user("q2"),
            assistant("answer"),
        ];
        let result = filter_history(&messages, 4, 0);
        assert!(
            result.iter().all(|m| role_of(m) != "tool"),
            "orphaned tool result at window boundary must be skipped"
        );
        // Remaining: assistant("Done"), user("q2"), assistant("answer") = 3
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_non_orphaned_tool_results_preserved() {
        // assistant+tc → tool → user → assistant — all 4 fit; tool is NOT orphaned.
        let messages = vec![
            tool_call_assistant("tc_1"),
            tool_result("tc_1"),
            user("thanks"),
            assistant("you're welcome"),
        ];
        let result = filter_history(&messages, 100, 0);
        assert_eq!(result.len(), 4);
        assert!(
            result.iter().any(|m| role_of(m) == "tool"),
            "complete tool group must be preserved"
        );
    }

    #[test]
    fn test_multiple_consecutive_orphaned_tool_results_all_skipped() {
        // Two orphaned tool results at the window boundary.
        let messages = vec![
            user("q1"),
            tool_call_assistant("tc_1"),
            tool_result("tc_1"),
            tool_result("tc_2"), // also orphaned (no matching assistant in window)
            user("q2"),
            assistant("a2"),
        ];
        // Window starts at index 2 (tc_1 tool result).
        let result = filter_history(&messages, 4, 0);
        assert!(
            result.iter().all(|m| role_of(m) != "tool"),
            "all orphaned tool results at boundary must be skipped"
        );
    }

    // ------------------------------------------------------------------
    // Synthetic message filtering
    // ------------------------------------------------------------------

    #[test]
    fn test_synthetic_messages_filtered() {
        let messages = vec![
            user("hello"),
            assistant("hi"),
            synthetic("[specialist:coding] injected context"),
        ];
        let result = filter_history(&messages, 100, 0);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0]["content"], "hello");
        assert_eq!(result[1]["content"], "hi");
    }

    #[test]
    fn test_multiple_synthetics_all_filtered() {
        let messages = vec![
            user("real question"),
            synthetic("[specialist:coding] long analysis..."),
            synthetic("[router:tool:web_fetch] <html>...</html>"),
            assistant("real answer"),
            user("follow up"),
            assistant("follow up answer"),
        ];
        let result = filter_history(&messages, 100, 0);
        assert_eq!(result.len(), 4);
        assert_eq!(result[0]["content"], "real question");
        assert_eq!(result[1]["content"], "real answer");
        assert_eq!(result[2]["content"], "follow up");
        assert_eq!(result[3]["content"], "follow up answer");
    }

    // ------------------------------------------------------------------
    // Summary filtering
    // ------------------------------------------------------------------

    #[test]
    fn test_summary_messages_filtered() {
        let messages = vec![
            user("question"),
            summary("This is an internal LCM summary."),
            assistant("answer"),
        ];
        let result = filter_history(&messages, 100, 0);
        assert!(
            result.iter().all(|m| role_of(m) != "summary"),
            "role:summary entries must be filtered"
        );
        assert_eq!(result.len(), 2, "only user + assistant should remain");
    }

    // ------------------------------------------------------------------
    // Turn limit
    // ------------------------------------------------------------------

    #[test]
    fn test_turn_limit_applied() {
        // 6 turns: user→assistant × 6 = 12 messages
        let mut messages = Vec::new();
        for i in 0..6u32 {
            messages.push(json!({"role": "user", "content": format!("question {}", i)}));
            messages.push(json!({"role": "assistant", "content": format!("answer {}", i)}));
        }

        // max_turns=3 → last 3 user-assistant pairs = 6 messages
        let result = filter_history(&messages, 100, 3);
        assert_eq!(result.len(), 6);
        assert_eq!(result[0]["content"], "question 3");
        assert_eq!(result[5]["content"], "answer 5");
    }

    #[test]
    fn test_turn_limit_zero_means_no_limit() {
        let mut messages = Vec::new();
        for i in 0..6u32 {
            messages.push(json!({"role": "user", "content": format!("q{}", i)}));
            messages.push(json!({"role": "assistant", "content": format!("a{}", i)}));
        }
        let result = filter_history(&messages, 100, 0);
        assert_eq!(result.len(), 12);
    }

    #[test]
    fn test_turn_limit_one() {
        let mut messages = Vec::new();
        for i in 0..6u32 {
            messages.push(json!({"role": "user", "content": format!("q{}", i)}));
            messages.push(json!({"role": "assistant", "content": format!("a{}", i)}));
        }
        let result = filter_history(&messages, 100, 1);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0]["content"], "q5");
        assert_eq!(result[1]["content"], "a5");
    }

    #[test]
    fn test_synthetic_nudges_dont_count_as_turns() {
        // 3 real turns, each padded with synthetic scaffolding nudges injected
        // during the turn's tool loop. Even with max_turns=3, all 3 REAL turns
        // must survive — synthetic nudges are not turns. Before the fix, the 9
        // nudges (role=user) pushed the turn count past the limit and head-
        // dropped real history on every reload, diverging the prompt prefix.
        let mut messages = Vec::new();
        for i in 0..3u32 {
            messages.push(user(&format!("real question {i}")));
            messages.push(assistant(&format!("answer {i}")));
            messages.push(synthetic("[grounding] Turn x. Context: 3% used."));
            messages.push(synthetic("[format-anchor] reminder"));
            messages.push(synthetic("[system] report first"));
        }
        let result = filter_history(&messages, 100, 3);

        let real_users: Vec<&str> = result
            .iter()
            .filter(|m| role_of(m) == "user")
            .filter_map(|m| m["content"].as_str())
            .collect();
        assert_eq!(
            real_users,
            vec!["real question 0", "real question 1", "real question 2"],
            "all real turns must survive; synthetic nudges must not be counted as turns"
        );
        // Synthetic nudge content must never reach the wire history.
        assert!(
            result
                .iter()
                .all(|m| !m["content"].as_str().unwrap_or("").starts_with('[')),
            "no synthetic scaffolding nudge should survive to the wire"
        );
    }

    #[test]
    fn test_prefix_byte_stable_across_reloads() {
        // The prefix-cache invariant: turn N+1's reloaded wire history must be an
        // APPEND-ONLY extension of turn N's — reload N's output is a byte-
        // identical prefix of N+1's, so the server only re-prefills the appended
        // tail instead of cold-prefilling from the divergence point.
        let turn_n = vec![
            user("q0"),
            assistant("a0"),
            synthetic("[grounding] g"),
            synthetic("[format-anchor] f"),
            user("q1"),
            assistant("a1"),
        ];
        let mut turn_n1 = turn_n.clone();
        turn_n1.extend(vec![
            synthetic("[system] report first"),
            user("q2"),
            assistant("a2"),
        ]);

        let out_n = filter_history(&turn_n, 100, 10);
        let out_n1 = filter_history(&turn_n1, 100, 10);

        assert!(out_n1.len() >= out_n.len());
        for (i, m) in out_n.iter().enumerate() {
            assert_eq!(
                m, &out_n1[i],
                "prefix diverged at index {i}: reload is not append-only"
            );
        }
    }

    #[test]
    fn test_token_budget_drops_at_real_user_turn_boundary() {
        // When over the Stage-6 token budget, the kept head must begin at a real
        // user turn — we drop whole oldest turns rather than slicing mid-turn and
        // leaving the history starting on an assistant message. Here the natural
        // budget cut lands on `a1` (the response to a large user turn); the snap
        // must advance it forward to the next real user turn (`q2`).
        let big_user = "x".repeat(8000); // ~2000 tokens, dwarfs the budget
        let messages = vec![
            user("q0"),
            assistant("a0"),
            json!({"role": "user", "content": big_user}),
            assistant("a1"),
            user("q2"),
            assistant("a2"),
        ];
        // max_messages=6 == len (no Stage-1 window drop); Stage-6 budget = 900.
        let result = filter_history(&messages, 6, 0);

        assert!(!result.is_empty());
        assert_eq!(
            role_of(&result[0]),
            "user",
            "kept history must start at a real user turn, not mid-turn"
        );
        assert_eq!(result[0]["content"], "q2");
    }

    // ------------------------------------------------------------------
    // Wire format field preservation
    // ------------------------------------------------------------------

    #[test]
    fn test_tool_calls_preserved_on_assistant() {
        let messages = vec![
            user("read a file"),
            json!({
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "tc_1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]
            }),
            json!({"role": "tool", "tool_call_id": "tc_1", "name": "read_file", "content": "data"}),
            assistant("done"),
        ];
        let result = filter_history(&messages, 100, 0);
        assert_eq!(result.len(), 4);
        assert!(
            result[1].get("tool_calls").is_some(),
            "tool_calls must be preserved"
        );
    }

    #[test]
    fn test_tool_call_id_and_name_preserved_on_tool_result() {
        let messages = vec![
            user("do it"),
            tool_call_assistant("tc_42"),
            json!({
                "role": "tool",
                "tool_call_id": "tc_42",
                "name": "exec",
                "content": "result"
            }),
            assistant("done"),
        ];
        let result = filter_history(&messages, 100, 0);
        assert_eq!(result.len(), 4);
        assert_eq!(
            result[2].get("tool_call_id").and_then(|v| v.as_str()),
            Some("tc_42")
        );
        assert_eq!(result[2].get("name").and_then(|v| v.as_str()), Some("exec"));
    }

    #[test]
    fn test_turn_field_preserved() {
        let messages = vec![
            json!({"role": "user", "content": "hello", "_turn": 1}),
            json!({"role": "assistant", "content": "hi", "_turn": 1}),
        ];
        let result = filter_history(&messages, 100, 0);
        assert_eq!(result.len(), 2);
        assert_eq!(
            result[0].get("_turn").and_then(|v| v.as_u64()),
            Some(1),
            "_turn must be preserved on user message"
        );
        assert_eq!(
            result[1].get("_turn").and_then(|v| v.as_u64()),
            Some(1),
            "_turn must be preserved on assistant message"
        );
    }

    #[test]
    fn test_extra_fields_not_leaked_to_wire_format() {
        // Fields like timestamp and metadata keys should NOT appear in the output.
        let messages = vec![json!({
            "role": "user",
            "content": "hello",
            "timestamp": "2026-01-01T00:00:00Z",
            "extra_internal_field": "should_not_appear",
        })];
        let result = filter_history(&messages, 100, 0);
        assert_eq!(result.len(), 1);
        // Only role, content (and optionally _turn/tool_calls etc.) should be present.
        assert!(
            result[0].get("timestamp").is_none(),
            "timestamp must not leak to wire format"
        );
        assert!(
            result[0].get("extra_internal_field").is_none(),
            "internal fields must not leak"
        );
    }

    // ------------------------------------------------------------------
    // Interaction between stages
    // ------------------------------------------------------------------

    #[test]
    fn test_clear_marker_and_max_messages_interact_correctly() {
        // clear takes priority: even if max_messages window would reach before clear,
        // clear_start wins via the max() call.
        let messages = vec![
            user("before_clear"), // index 0
            clear(),              // index 1
            user("after_clear"),  // index 2
            assistant("answer"),  // index 3
        ];
        // max_messages=4 (all), but clear at index 1 means safe_start=2
        let result = filter_history(&messages, 4, 0);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0]["content"], "after_clear");
    }

    #[test]
    fn test_turn_limit_and_clear_interact_correctly() {
        // Turns are counted within the post-clear window, not from before it.
        let messages = vec![
            user("old q"),
            assistant("old a"),
            clear(),
            user("q1"),
            assistant("a1"),
            user("q2"),
            assistant("a2"),
        ];
        // max_turns=1 applied to the post-clear slice → last 1 turn = q2/a2
        let result = filter_history(&messages, 100, 1);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0]["content"], "q2");
        assert_eq!(result[1]["content"], "a2");
    }

    // ------------------------------------------------------------------
    // Stage 6: Token budget
    // ------------------------------------------------------------------

    #[test]
    fn test_token_budget_drops_old_messages_when_over_budget() {
        // max_messages=6 → token budget = 6 * 150 = 900 tokens.
        // Each "x".repeat(2000) message is ~500 tokens (2020 chars / 4).
        // 3 such messages = ~1500 tokens > 900 budget.
        let big = "x".repeat(2000);
        let messages = vec![
            user(&big),          // ~500 tokens (oldest, should be dropped)
            assistant(&big),     // ~500 tokens (should be dropped)
            user("recent"),      // ~6 tokens (kept)
            assistant("answer"), // ~7 tokens (kept)
        ];
        let result = filter_history(&messages, 6, 0);
        // The two big messages (~1000 tokens) exceed the budget of 900.
        // Walking backward: "answer" (7) + "recent" (6) = 13, well under 900.
        // Adding assistant(&big) would be 13 + 500 = 513, still under.
        // Adding user(&big) would be 513 + 500 = 1013 > 900 → stop.
        // So we keep 3 messages.
        assert!(
            result.len() <= 3,
            "expected at most 3 messages, got {}",
            result.len()
        );
        // The last message must be the "answer"
        assert_eq!(result.last().unwrap()["content"], "answer");
    }

    #[test]
    fn test_token_budget_zero_max_messages_preserves_all() {
        // max_messages=0 means "no limit" in Stage 1. Stage 6 must honour
        // that contract and not drop everything (H1 regression guard).
        let messages = vec![
            user("hello"),
            assistant("world"),
            user("follow up"),
            assistant("reply"),
        ];
        let result = filter_history(&messages, 0, 0);
        assert_eq!(result.len(), 4, "max_messages=0 must preserve all history");
    }

    #[test]
    fn test_token_budget_preserves_small_history() {
        // Small messages well under budget → all preserved.
        let messages = vec![
            user("hi"),
            assistant("hello"),
            user("how are you"),
            assistant("fine"),
        ];
        let result = filter_history(&messages, 100, 0);
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_token_budget_skips_orphaned_tool_results_at_boundary() {
        // When token budget drops messages, the new boundary might land on
        // a tool result whose parent assistant+tool_calls was dropped.
        // The orphan-skip must advance past it.
        //
        // Layout: [assistant+tc (big), tool_result (big), user, assistant]
        // max_messages=2 → budget = 300 tokens.
        // The big messages (~500 tokens each) blow the budget; backward walk
        // keeps user + assistant (~13 tokens), then can't fit tool_result.
        // keep_from lands at index 2 (user) — no orphan at boundary.
        //
        // To force the orphan case: budget must fit the tool_result but NOT
        // its parent. Use max_messages=5 → budget=750. Backward walk keeps
        // assistant(6) + user(6) + tool_result(~505) = 517 < 750, then tries
        // assistant+tc (~45) = 562 < 750 — fits. So we need the tool result
        // to be the tipping point.
        //
        // Simplest: make the assistant+tool_calls message large so it busts
        // the budget, leaving tool_result as first kept message (orphaned).
        let big_args = "y".repeat(3000); // ~750 tokens in tool_calls JSON
        let messages = vec![
            json!({
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "exec", "arguments": &big_args}}]
            }),
            json!({"role": "tool", "tool_call_id": "c1", "name": "exec", "content": "ok"}),
            user("question"),
            assistant("answer"),
        ];
        // max_messages=4 → budget = 600 tokens.
        // assistant+tc is ~770 tokens (big_args alone). Total > 600.
        // Backward walk: answer(7) + question(7) + tool(8) = 22 < 600.
        // Adding assistant+tc (770) = 792 > 600 → stop. keep_from = 1 (tool).
        // Orphan skip advances past the tool result → keep_from = 2.
        let result = filter_history(&messages, 4, 0);
        assert!(
            result.iter().all(|m| role_of(m) != "tool"),
            "orphaned tool result at token-budget boundary must be skipped"
        );
        assert_eq!(result.len(), 2, "only user + assistant should remain");
        assert_eq!(result[0]["content"], "question");
        assert_eq!(result[1]["content"], "answer");
    }
}
