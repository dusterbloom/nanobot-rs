//! Pure filtering functions for session message history.
//!
//! Extracted from the JSONL `SessionManager` for reuse by the SQLite
//! `SessionDb`. All functions are pure — no I/O.

use serde_json::Value;

/// Filter messages: respect clear markers, skip orphaned tool results,
/// filter synthetics, apply turn limit, and map to wire format.
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
pub fn filter_history(messages: &[Value], max_messages: usize, max_turns: usize) -> Vec<Value> {
    // Stage 1: max_messages window — start index into `messages`.
    let start = if messages.len() > max_messages {
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
    // A tool result is orphaned when its matching assistant+tool_calls message
    // was before the window start and got dropped. Sending a lone tool result
    // to the LLM is a protocol error, so we skip it.
    let mut safe_start = start.max(clear_start);
    while safe_start < messages.len() {
        let role = messages[safe_start]
            .get("role")
            .and_then(|r| r.as_str())
            .unwrap_or("");
        if role == "tool" {
            safe_start += 1;
        } else {
            break;
        }
    }

    // Stage 4: turn-based limit. Scan backward from the end counting user
    // messages as turn boundaries. If more than `max_turns` user messages are
    // seen, advance safe_start to drop the oldest ones.
    if max_turns > 0 {
        let mut turns_seen = 0;
        let mut turn_start = safe_start;
        for i in (safe_start..messages.len()).rev() {
            if messages[i].get("role").and_then(|r| r.as_str()) == Some("user") {
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
    messages[safe_start..]
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
        .collect()
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
        assert_eq!(result.len(), 2, "only messages after clear should be returned");
        assert_eq!(result[0]["content"], "new question");
        assert_eq!(result[1]["content"], "new answer");
        assert!(result.iter().all(|m| role_of(m) != "clear"), "clear marker must not appear in output");
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
        assert!(result.is_empty(), "nothing after clear marker should yield empty history");
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
        assert!(result.iter().any(|m| role_of(m) == "tool"), "complete tool group must be preserved");
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
        assert!(result[1].get("tool_calls").is_some(), "tool_calls must be preserved");
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
        assert_eq!(
            result[2].get("name").and_then(|v| v.as_str()),
            Some("exec")
        );
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
        assert!(result[0].get("timestamp").is_none(), "timestamp must not leak to wire format");
        assert!(result[0].get("extra_internal_field").is_none(), "internal fields must not leak");
    }

    // ------------------------------------------------------------------
    // Interaction between stages
    // ------------------------------------------------------------------

    #[test]
    fn test_clear_marker_and_max_messages_interact_correctly() {
        // clear takes priority: even if max_messages window would reach before clear,
        // clear_start wins via the max() call.
        let messages = vec![
            user("before_clear"),   // index 0
            clear(),               // index 1
            user("after_clear"),   // index 2
            assistant("answer"),   // index 3
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
}
