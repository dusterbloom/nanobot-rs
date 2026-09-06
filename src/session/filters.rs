// Error-protocol layer-3 backlog (docs/research/2026-08-06-error-conventions-and-host-bridge.md §3.6):
// the deny regime in Cargo.toml is live; this module still carries pre-existing
// violations of the lints below. Remove this allow as the module migrates onto
// the regime.
// Tracking: docs/error-protocol-backlog.md
#![allow(clippy::indexing_slicing, clippy::shadow_reuse)]
//! Pure filtering functions for session message history.
//!
//! Extracted from the JSONL `SessionManager` for reuse by the SQLite
//! `SessionDb`. All functions are pure — no I/O.

use serde_json::Value;
use tracing::warn;

use crate::agent::context_hygiene::cap_tool_result_for_replay;
#[cfg(test)]
use crate::agent::context_hygiene::TOOL_RESULT_REPLAY_MAX_BYTES;

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

fn is_non_replayable_synthetic(msg: &Value) -> bool {
    msg.get("_synthetic")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
        && !is_cache_replay_synthetic(msg)
}

fn is_cache_replay_synthetic(msg: &Value) -> bool {
    if msg
        .get("_cache_replay")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
    {
        return true;
    }

    // Legacy scaffold rows created before `_cache_replay` existed only have
    // `synthetic=1` in SQLite. Preserve the known turn-scaffolding prefixes
    // because they were sent to the model and are therefore part of the warm
    // prompt prefix.
    if msg.get("role").and_then(|r| r.as_str()) != Some("user") {
        return false;
    }
    let content = msg.get("content").and_then(|v| v.as_str()).unwrap_or("");
    content.starts_with("[format-anchor]")
        || content.starts_with("[grounding]")
        || content.starts_with("[System notice]")
        || content.starts_with("[System] Loop detected:")
        || content.starts_with("[system] Report what the previous tool results showed before")
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
/// filter non-replayable synthetics, apply turn limit, and map to wire format.
///
/// This is the primary entry point — it applies all filtering stages
/// in sequence:
///
/// 1. `max_messages` window — take the last N messages
/// 2. Clear markers — only show messages after the last `role: "clear"` marker
/// 3. Orphaned tool results — skip leading `role: "tool"` messages at the
///    window boundary when their parent assistant+tool_calls is outside the window
/// 4. Turn limit — keep only the last `max_turns` user-assistant pairs
/// 5. Per-message filter/map — strip non-replayable synthetics, clear markers, summaries;
///    copy role, content, tool_calls, tool_call_id, name, _turn to wire format
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
        // Real-turn start indices within the current window, oldest first.
        let turn_starts: Vec<usize> = (safe_start..messages.len())
            .filter(|&i| is_real_user_turn(&messages[i]))
            .collect();
        let n = turn_starts.len();
        if n > max_turns {
            // Hysteresis: advance the drop boundary in whole batches of `batch`
            // turns, not one turn per reload. Dropping the single oldest turn on
            // every reload shifts the kept-history HEAD each turn, diverging the
            // prompt prefix and forcing the inference server to re-prefill the
            // entire context (~50s at 15k tokens on a local MLX backend). With
            // batched drops the head stays byte-stable for `batch` reloads at a
            // time, so the prefix cache stays warm between drops. `batch` of 1
            // (i.e. max_turns < 2) preserves the original drop-one-each behavior.
            let batch = (max_turns / 2).max(1);
            let dropped = ((n - max_turns) / batch) * batch;
            if dropped > 0 {
                safe_start = safe_start.max(turn_starts[dropped]);
            }
        }
    }

    // Stage 5: filter and map each surviving message to wire format.
    let mapped: Vec<Value> = messages[safe_start..]
        .iter()
        .enumerate()
        .filter(|(_, m)| {
            // Skip synthetic router/specialist injections. Cache-replay
            // scaffolds were already sent to the model, so dropping them on
            // reload would mutate the warm prompt prefix.
            !is_non_replayable_synthetic(m)
                // Skip clear markers; they must not appear in the wire history.
                && m.get("role").and_then(|v| v.as_str()) != Some("clear")
                // Skip internal LCM summary entries — not valid wire format.
                && m.get("role").and_then(|r| r.as_str()) != Some("summary")
        })
        .map(|(_offset, m)| {
            let role = m.get("role").and_then(|v| v.as_str()).unwrap_or("user");
            // Tool results are the bulkiest, lowest-value-once-stale part of
            // history (web_fetch / skill dumps). Cap their body to a generous,
            // FIXED size so one large dump can't crowd conversation out of the
            // token budget. The cap is applied identically on every reload (not
            // age-based), so it never shifts the prompt prefix — no extra
            // re-prefill, unlike dropping or sliding truncation.
            //
            // `recall_tool_result` bodies get NO special treatment. They used
            // to be swapped for a short "[recalled earlier…]" receipt here, on
            // the reasoning that a one-shot recall shouldn't balloon the
            // prompt forever. That swap is a read-time byte change: the turn
            // that recalled saw ~10 KB, the next turn replayed ~120 bytes, and
            // the inference server — which matches on content — lost its KV
            // prefix and re-prefilled everything. Measured in session
            // 20260810_081050_8306f8: 8 tool messages 24907 → 15060 bytes,
            // higgs `boundary_splice_failed` → 124.54s of cold prefill on a
            // 9267-token prompt. The comment that justified it claimed "the
            // Higgs radix cache is unaffected"; the log says otherwise.
            // A recalled body is now just a tool result, capped like the rest.
            let raw = m.get("content").and_then(|v| v.as_str()).unwrap_or("");
            let content = if role == "tool"
                && crate::agent::tool_engine::is_stable_tool_result_representation(raw)
            {
                // Handles and explicit retrieval excerpts are rendered once at
                // ingestion: pass through byte-identical on reload so the
                // prefix cache never sees a drift.
                raw.to_string()
            } else if role == "tool" {
                cap_tool_body(raw)
            } else {
                raw.to_string()
            };
            let mut msg = serde_json::json!({
                "role": role,
                "content": content,
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
            // Preserve _db_id (stable SQLite rowid — the LCM engine's
            // MessageId). Internal-only, like _turn: the protocol render
            // drops it before the wire call.
            if let Some(db_id) = m.get("_db_id") {
                msg["_db_id"] = db_id.clone();
            }
            // Execution status is internal replay metadata. Preserve it so
            // canonical Turn reconstruction never has to infer truth from
            // display text; protocol rendering strips it before provider I/O.
            if let Some(ok) = m.get("ok") {
                msg["ok"] = ok.clone();
            }
            msg
        })
        .collect();

    // Loading history must not invent a token ceiling from a message count.
    // The agent's effective TokenBudget and LCM own context fitting; dropping
    // turns here hides them from compaction and silently destroys recall even
    // while the actual provider prompt fits. Keep row/turn limits above only.
    mapped
}

/// Cap an oversized tool-result body to the shared replay limit. Deterministic
/// in its input, so the same stored tool result always renders identically.
fn cap_tool_body(content: &str) -> String {
    cap_tool_result_for_replay(content)
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

    fn cache_replay_synthetic(content: &str) -> Value {
        json!({
            "role": "user",
            "content": content,
            "_synthetic": true,
            "_cache_replay": true
        })
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

    #[test]
    fn test_cache_replay_synthetic_preserved_but_not_counted_as_turn() {
        let turn_n = vec![
            user("q0"),
            assistant("a0"),
            cache_replay_synthetic("[system] Report what the previous tool results showed."),
        ];
        let mut turn_n1 = turn_n.clone();
        turn_n1.extend([user("continue please"), assistant("continuing")]);

        let out_n = filter_history(&turn_n, 100, 2);
        let out_n1 = filter_history(&turn_n1, 100, 2);

        assert_eq!(
            out_n.iter().filter(|m| role_of(m) == "user").count(),
            2,
            "cache-replay scaffold must survive reload"
        );
        assert_eq!(
            out_n1[0..out_n.len()],
            out_n[..],
            "next reload must preserve the scaffold-containing sent prefix"
        );
        assert!(
            out_n1
                .iter()
                .all(|m| m.get("_synthetic").is_none() && m.get("_cache_replay").is_none()),
            "synthetic metadata must not leak into wire history"
        );
    }

    #[test]
    fn test_legacy_scaffold_synthetic_preserved_without_cache_replay_marker() {
        let turn_n = vec![
            user("q0"),
            assistant("a0"),
            synthetic(
                "[format-anchor] Reminder: use tool calls for actions, not text descriptions.",
            ),
        ];
        let mut turn_n1 = turn_n.clone();
        turn_n1.extend([user("continue please"), assistant("continuing")]);

        let out_n = filter_history(&turn_n, 100, 2);
        let out_n1 = filter_history(&turn_n1, 100, 2);

        assert!(
            out_n.iter().any(|m| {
                m["content"]
                    .as_str()
                    .unwrap_or("")
                    .starts_with("[format-anchor]")
            }),
            "legacy sent scaffold rows must survive reload"
        );
        assert_eq!(
            out_n1[0..out_n.len()],
            out_n[..],
            "legacy scaffold replay must keep reloads append-only"
        );
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
            messages.push(synthetic(
                "[system] Report what the previous tool results showed before running more tools.",
            ));
        }
        let result = filter_history(&messages, 100, 3);

        let real_users: Vec<&str> = result
            .iter()
            .filter(|m| role_of(m) == "user")
            .filter(|m| !m["content"].as_str().unwrap_or("").starts_with('['))
            .filter_map(|m| m["content"].as_str())
            .collect();
        assert_eq!(
            real_users,
            vec!["real question 0", "real question 1", "real question 2"],
            "all real turns must survive; synthetic nudges must not be counted as turns"
        );
        let scaffold_count = result
            .iter()
            .filter(|m| m["content"].as_str().unwrap_or("").starts_with('['))
            .count();
        assert_eq!(
            scaffold_count, 9,
            "sent scaffolding nudges must be replayed for cache stability"
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
    fn test_turn_limit_hysteresis_keeps_prefix_stable_past_limit() {
        // Past the turn limit, the kept-history HEAD must not shift on every
        // reload. The original code dropped the single oldest turn each reload,
        // so once a session passed `max_turns` every new turn re-based the head
        // and the inference server re-prefilled the whole context (~50s at 15k
        // tokens). With hysteresis (batch = max_turns/2), the drop boundary
        // advances only every `batch` turns, so consecutive reloads stay
        // append-only between drops and the prefix cache stays warm.
        let max_turns = 10;
        // 20 real turns added one at a time; capture the reload after each.
        // max_messages = 0 isolates the Stage-4 turn limit (no Stage-1 window,
        // no Stage-6 token budget).
        let mut messages = Vec::new();
        let mut reloads = Vec::new();
        for t in 0..20 {
            messages.push(user(&format!("q{t}")));
            messages.push(assistant(&format!("a{t}")));
            reloads.push(filter_history(&messages, 0, max_turns));
        }

        // A reload that is NOT an append-only extension of the previous one is a
        // head shift = a full re-prefill on the live server.
        let head_shifts = reloads
            .windows(2)
            .filter(|w| {
                let (prev, cur) = (&w[0], &w[1]);
                let append_only =
                    prev.len() <= cur.len() && prev.iter().zip(cur.iter()).all(|(a, b)| a == b);
                !append_only
            })
            .count();

        // Turns 11..=20 exceed the limit. Without hysteresis that is ~10 head
        // shifts (one per reload). With batch=5 it must be at most 2.
        assert!(
            head_shifts <= 2,
            "expected batched drops (<=2 head shifts over 20 turns), got {head_shifts} \
             — the kept-history head is shifting on (nearly) every reload, busting the cache"
        );
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
    // History budget ownership
    // ------------------------------------------------------------------

    #[test]
    fn history_reload_keeps_long_turns_for_the_context_budget_owner() {
        let mut messages = Vec::new();
        for batch in 1..=8 {
            messages.push(user(&format!("BATCH_{batch} {}", "archive ".repeat(2400))));
            messages.push(assistant("OK"));
        }
        // A 49K context loads 229 messages. That count is not a 34350-token
        // ceiling: the agent must see every turn before fitting/compacting it.
        let result = filter_history(&messages, 229, 600);
        assert_eq!(result.len(), messages.len());
        assert!(result[0]["content"]
            .as_str()
            .unwrap()
            .starts_with("BATCH_1 "));
    }

    #[test]
    fn test_token_budget_zero_max_messages_preserves_all() {
        // max_messages=0 means no row limit.
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
        let big_args = "y".repeat(3000);
        let messages = vec![
            json!({
                "role": "assistant", "content": "",
                "tool_calls": [{"id": "c1", "type": "function", "function": {
                    "name": "exec", "arguments": &big_args
                }}]
            }),
            json!({"role": "tool", "tool_call_id": "c1", "name": "exec", "content": "ok"}),
            user("question"),
            assistant("answer"),
        ];
        // Size alone must not split a valid tool pair. If the explicit row
        // window excludes its parent, the orphan still must be removed.
        let complete = filter_history(&messages, 4, 0);
        assert_eq!(complete.len(), 4);
        assert_eq!(complete[1]["tool_call_id"], "c1");
        let result = filter_history(&messages, 3, 0);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0]["content"], "question");
        assert_eq!(result[1]["content"], "answer");
    }

    #[test]
    fn test_oversized_tool_body_capped_deterministically() {
        // A large tool result is capped to a fixed size with a marker, and the
        // render is deterministic — the same stored result yields identical wire
        // bytes on every reload, so it never shifts the prompt prefix (no extra
        // re-prefill). Small tool bodies and non-tool content pass through.
        let big = "x".repeat(TOOL_RESULT_REPLAY_MAX_BYTES + 5000);
        let messages = vec![
            user("fetch something"),
            tool_call_assistant("t1"),
            json!({"role": "tool", "tool_call_id": "t1", "name": "exec", "content": big}),
            assistant("done"),
            user("and a small one"),
            tool_call_assistant("t2"),
            json!({"role": "tool", "tool_call_id": "t2", "name": "exec", "content": "short result"}),
            assistant("ok"),
        ];
        // Stages 1/4/6 disabled (0,0) to isolate the Stage-5 tool-body cap.
        let a = filter_history(&messages, 0, 0);
        let b = filter_history(&messages, 0, 0);
        assert_eq!(a, b, "render must be deterministic for prefix stability");

        let big_tool = a
            .iter()
            .find(|m| role_of(m) == "tool" && m["tool_call_id"] == "t1")
            .unwrap();
        let body = big_tool["content"].as_str().unwrap();
        assert!(
            body.len() <= TOOL_RESULT_REPLAY_MAX_BYTES + 40,
            "oversized tool body must be capped"
        );
        assert!(
            body.ends_with("[tool output truncated]"),
            "truncation marker appended"
        );

        // Small tool body and non-tool content are untouched.
        let small_tool = a
            .iter()
            .find(|m| role_of(m) == "tool" && m["tool_call_id"] == "t2")
            .unwrap();
        assert_eq!(
            small_tool["content"], "short result",
            "small tool unchanged"
        );
        assert_eq!(
            a[0]["content"], "fetch something",
            "user content not capped"
        );
    }

    /// Reload must be byte-stable: `filter_history` is applied to the SAME
    /// stored rows on every turn, so running it twice must produce identical
    /// content. A `recall_tool_result` body used to shrink to a short receipt
    /// here, which meant turn N sent ~10 KB and turn N+1 sent ~120 bytes —
    /// the inference server matches on content, so it dropped its KV prefix
    /// and re-prefilled (124.54s measured, session 20260810_081050_8306f8).
    ///
    /// Asserted for a recalled body and a plain oversized body together: the
    /// invariant is "reload is a pure function of stored bytes", not a fact
    /// about one tool name.
    #[test]
    fn reload_is_byte_stable_for_tool_bodies() {
        let big = "x".repeat(TOOL_RESULT_REPLAY_MAX_BYTES + 5000);
        let messages = vec![
            user("recall it"),
            json!({
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "recall_1",
                        "type": "function",
                        "function": {
                            "name": "recall_tool_result",
                            "arguments": "{\"tool_call_id\":\"original_tool_call\"}"
                        }
                    },
                    {
                        "id": "fetch_1",
                        "type": "function",
                        "function": { "name": "web_fetch", "arguments": "{}" }
                    }
                ]
            }),
            json!({
                "role": "tool",
                "tool_call_id": "recall_1",
                "name": "recall_tool_result",
                "content": big.clone(),
            }),
            json!({
                "role": "tool",
                "tool_call_id": "fetch_1",
                "name": "web_fetch",
                "content": big,
            }),
        ];

        let first = filter_history(&messages, 0, 0);
        let second = filter_history(&messages, 0, 0);
        assert_eq!(first, second, "reload must be deterministic");

        let body = |out: &[Value], id: &str| -> String {
            out.iter()
                .find(|m| role_of(m) == "tool" && m["tool_call_id"] == id)
                .and_then(|m| m["content"].as_str())
                .unwrap_or_default()
                .to_string()
        };

        // A recalled body is capped exactly like any other tool body — no
        // tool-name-specific shrink that the live wire never applied.
        assert_eq!(
            body(&first, "recall_1"),
            body(&first, "fetch_1"),
            "recall_tool_result must not be treated differently from web_fetch"
        );
        assert!(
            !body(&first, "recall_1").contains("recalled earlier"),
            "the one-shot receipt must be gone"
        );
    }

    #[test]
    fn test_failed_recall_tool_result_replays_exact_error() {
        let error =
            "No stored output for tool_call_id='missing' in this session. Re-run the original tool.";
        let messages = vec![
            user("recall it"),
            json!({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "recall_1",
                    "type": "function",
                    "function": {
                        "name": "recall_tool_result",
                        "arguments": "{\"tool_call_id\":\"missing\"}"
                    }
                }]
            }),
            json!({
                "role": "tool",
                "tool_call_id": "recall_1",
                "name": "recall_tool_result",
                "content": error,
            }),
        ];

        let result = filter_history(&messages, 0, 0);
        let tool = result
            .iter()
            .find(|m| role_of(m) == "tool" && m["tool_call_id"] == "recall_1")
            .unwrap();
        assert_eq!(tool["content"], error);
        assert!(!tool["content"].as_str().unwrap().contains("shown raw once"));
    }

    #[test]
    fn test_wrapped_failed_recall_tool_result_replays_exact_error() {
        let error = "[VERBATIM TOOL OUTPUT — do not paraphrase]\nError: No stored output for tool_call_id='missing' in this session. Re-run the original tool.\n[END TOOL OUTPUT]";
        let messages = vec![
            user("recall it"),
            json!({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "recall_1",
                    "type": "function",
                    "function": {
                        "name": "recall_tool_result",
                        "arguments": "{\"tool_call_id\":\"missing\"}"
                    }
                }]
            }),
            json!({
                "role": "tool",
                "tool_call_id": "recall_1",
                "name": "recall_tool_result",
                "content": error,
            }),
        ];

        let result = filter_history(&messages, 0, 0);
        let tool = result
            .iter()
            .find(|m| role_of(m) == "tool" && m["tool_call_id"] == "recall_1")
            .unwrap();
        assert_eq!(tool["content"], error);
        assert!(!tool["content"].as_str().unwrap().contains("shown raw once"));
    }
}
