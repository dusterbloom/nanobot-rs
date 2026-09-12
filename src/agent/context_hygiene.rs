// Error-protocol layer-3 backlog (docs/research/2026-08-06-error-conventions-and-host-bridge.md §3.6):
// the deny regime in Cargo.toml is live; this module still carries pre-existing
// violations of the lints below. Remove this allow as the module migrates onto
// the regime.
// Tracking: docs/error-protocol-backlog.md
#![allow(clippy::indexing_slicing)]
#![allow(dead_code)]
//! Context hygiene pipeline for cleaning up conversation history.
//!
//! Vendored and adapted from stakpak-agent-core:
//! https://github.com/stakpak/agent/blob/main/libs/agent-core/src/context.rs
//!
//! Removes duplicate tool results, orphaned results, dangling tool calls,
//! and merges consecutive same-role messages to prevent context drift.
//!
//! ## Pipeline Order
//! 1. `dedup_tool_results` - Keep only last result per tool_call_id
//! 2. `merge_consecutive_same_role` - Merge consecutive same-role messages
//! 3. `drop_old_tool_results` - Keep last N tool results (whole multi-call batches)
//! 4. `truncate_old_assistant_messages` - Compress old assistant content
//! 5. `strip_dangling_tool_calls` - Strip individual tool calls lacking immediate results
//! 6. `remove_orphaned_tool_results` - Remove results without matching calls

use std::collections::{HashMap, HashSet};

use serde_json::{json, Value};
use tracing::debug;

use super::is_synthetic_injection;
use crate::agent::compaction::tool_output_digest;

const TRUNCATED_ASSISTANT_PLACEHOLDER: &str = "[assistant message truncated]";

/// Policy for shrinking an oversized tool-result body.
///
/// Each variant reproduces one pre-existing call site byte-for-byte:
/// - `ByteCap`: the session-reload cap (`src/session/filters.rs`).
///   Deterministic in its input, so the same stored tool result always
///   renders identically — keeping the prompt prefix byte-stable.
/// - `Digest`: the token-budget pressure digest (`src/agent/token_budget.rs`),
///   replacing bodies longer than `preview_len` with a compact
///   `TOOL_OUTPUT_DIGEST` marker.
#[derive(Clone, Copy, Debug)]
pub enum ToolBodyPolicy {
    /// Cap the body to at most this many bytes (backed off to a UTF-8 char
    ///   boundary), appending the `...[tool output truncated]` marker.
    ByteCap(usize),
    /// Replace bodies longer than `preview_len` bytes with a digest marker
    /// (sha256 prefix + original length + single-line preview).
    Digest { preview_len: usize },
}

/// Max bytes retained for a tool-result body in replayable prompt history.
///
/// The same cap must apply before a tool result enters live `ctx.messages` and
/// when session history is loaded from SQLite. Otherwise a same-turn prompt can
/// warm the server cache with bytes that the next turn will never replay.
///
/// Sized above `config::schema::DEFAULT_MAX_TOOL_RESULT_CHARS` (10_000) so a
/// result that fits the user-visible char cap (near-ASCII) also fits the byte
/// cap and stays inline instead of degrading to a stashed handle. A static cap
/// can't track per-agent char caps exactly (UTF-8 multibyte), but 12_000 keeps
/// the 8-10KB dead band — where a sub-cap result became an unresolvable handle
/// (session 20260804_204406_c16eb0) — inside the inline contract.
pub(crate) const TOOL_RESULT_REPLAY_MAX_BYTES: usize = 12_000;

/// Hybrid exposure threshold (prototype): ordinary tool results up to this
/// size are injected INLINE (deterministic bytes → cache-stable, and too
/// small to pressure context); larger results become handles. Kills the
/// inspect-tax for the common small-result case (kiss/tool-surface bench:
/// pi-style read/explore needed 13-17 inspect_tool_result calls per task
/// because every result, however small, was a handle). Both live ingestion
/// and get_history replay must apply the SAME threshold, or reloads rewrite
/// inline bodies into handles and bust the retained KV prefix.
pub(crate) const INLINE_TOOL_RESULT_MAX_BYTES: usize = 4_096;

pub(crate) fn cap_tool_result_for_replay(content: &str) -> String {
    shrink_tool_body(
        content,
        ToolBodyPolicy::ByteCap(TOOL_RESULT_REPLAY_MAX_BYTES),
    )
    .unwrap_or_else(|| content.to_string())
}

/// Tool-result status is part of the protocol contract, not display text.
/// Keep this shared so provenance wrappers, boundary rejections, legacy replay,
/// and raw live results all agree on whether a tool actually succeeded.
pub(crate) fn tool_result_status_text(content: &str) -> &str {
    content
        .trim_start()
        .strip_prefix("[VERBATIM TOOL OUTPUT — do not paraphrase]")
        .unwrap_or_else(|| content.trim_start())
        .trim_start()
}

pub(crate) fn tool_result_ok(content: &str) -> bool {
    let normalized = tool_result_status_text(content).trim();
    !(normalized.starts_with("Error:")
        || normalized.starts_with("response boundary:")
        || normalized.starts_with("No stored output for tool_call_id=")
        || normalized == "(no result)")
}

/// Shrink a tool-result body according to `policy`.
///
/// Returns `Some(replacement)` when the body was shrunk, `None` when it is
/// small enough to pass through unchanged. Which messages to shrink (all on
/// reload, older-than-N, under budget pressure) stays the caller's business —
/// this unifies only the body-shrinking string logic.
pub fn shrink_tool_body(content: &str, policy: ToolBodyPolicy) -> Option<String> {
    match policy {
        ToolBodyPolicy::ByteCap(max_bytes) => {
            if content.len() <= max_bytes {
                return None;
            }
            let mut end = max_bytes;
            while end > 0 && !content.is_char_boundary(end) {
                end -= 1;
            }
            Some(format!("{}\n...[tool output truncated]", &content[..end]))
        }
        ToolBodyPolicy::Digest { preview_len } => {
            if content.len() <= preview_len {
                return None;
            }
            Some(tool_output_digest(content, preview_len))
        }
    }
}

pub fn hygiene_pipeline(messages: &mut Vec<Value>, keep_last_messages: usize) {
    if messages.is_empty() {
        return;
    }

    let before = messages.len();

    dedup_tool_results(messages);
    merge_consecutive_same_role(messages);
    drop_old_tool_results(messages, keep_last_messages);
    truncate_old_assistant_messages(messages, keep_last_messages);
    strip_dangling_tool_calls(messages);
    remove_orphaned_tool_results(messages);

    let removed = before.saturating_sub(messages.len());
    if removed > 0 {
        debug!(
            "Context hygiene: {} → {} messages (removed {})",
            before,
            messages.len(),
            removed
        );
    }
}

fn dedup_tool_results(messages: &mut Vec<Value>) {
    let mut last_positions: HashMap<String, usize> = HashMap::new();

    for (idx, message) in messages.iter().enumerate() {
        if let Some(tool_call_id) = get_tool_result_id(message) {
            last_positions.insert(tool_call_id, idx);
        }
    }

    let mut to_keep = vec![true; messages.len()];
    for (idx, message) in messages.iter().enumerate() {
        if let Some(tool_call_id) = get_tool_result_id(message) {
            if let Some(&last_idx) = last_positions.get(&tool_call_id) {
                if idx != last_idx {
                    to_keep[idx] = false;
                }
            }
        }
    }

    let mut write_idx = 0;
    for (read_idx, keep) in to_keep.into_iter().enumerate() {
        if keep {
            if write_idx != read_idx {
                messages[write_idx] = messages[read_idx].clone();
            }
            write_idx += 1;
        }
    }
    messages.truncate(write_idx);
}

fn merge_consecutive_same_role(messages: &mut Vec<Value>) {
    if messages.len() <= 1 {
        return;
    }

    let mut merged: Vec<Value> = Vec::with_capacity(messages.len());

    for message in messages.drain(..) {
        if let Some(prev) = merged.last_mut() {
            // Never merge tool-role messages (OpenAI protocol).
            // Never merge synthetic router/specialist injections — they must
            // remain as distinct messages so the model can see both the user's
            // original prompt and the injected tool result separately.
            let same_role = get_role(prev) == get_role(&message) && get_role(&message) != "tool";
            let either_synthetic = is_synthetic_injection(prev) || is_synthetic_injection(&message);

            if same_role && !either_synthetic {
                let prev_content = get_content(prev);
                let msg_content = get_content(&message);
                let combined = format!("{}\n{}", prev_content, msg_content);
                let turn_tag = prev.get("_turn").cloned();
                *prev = json!({
                    "role": get_role(&message),
                    "content": combined
                });
                if let Some(turn) = turn_tag {
                    prev["_turn"] = turn;
                }
                if let Some(tc) = message.get("tool_calls") {
                    prev["tool_calls"] = tc.clone();
                }
                continue;
            }
        }
        merged.push(message);
    }

    *messages = merged;
}

/// Drop tool-result messages, keeping at least the last `keep_last_n`.
///
/// (Formerly `truncate_old_tool_results` — renamed: it removes messages, it
/// does not shrink bodies. Body shrinking lives in [`shrink_tool_body`].)
///
/// Keeps/drops *whole multi-call batches*: a single assistant turn's tool
/// results are never split by the keep boundary. If the boundary would land
/// inside a multi-call batch, the whole straddling batch is kept (round up),
/// so the next pipeline pass always sees a consistent sibling set — never a
/// partial one that [`strip_dangling_tool_calls`] could mistake for a
/// corrupt log and wipe.
pub fn drop_old_tool_results(messages: &mut Vec<Value>, keep_last_n: usize) {
    if keep_last_n == usize::MAX || messages.len() <= keep_last_n {
        return;
    }

    // Group tool results into batches: each maximal run of consecutive
    // tool-role messages is one batch (an assistant turn's results arrive
    // contiguously, and `merge_consecutive_same_role` never merges tool
    // messages). Keeping/dropping whole batches guarantees a multi-call
    // assistant is never left with a partial result set at the keep boundary.
    let mut batches: Vec<Vec<(usize, String)>> = Vec::new();
    for (idx, message) in messages.iter().enumerate() {
        if get_role(message) != "tool" {
            continue;
        }
        let Some(result_id) = get_tool_result_id(message) else {
            continue;
        };
        let continues_run = batches
            .last()
            .and_then(|b| b.last())
            .is_some_and(|(prev_idx, _)| *prev_idx + 1 == idx);
        if !continues_run {
            batches.push(Vec::new());
        }
        if let Some(batch_slot) = batches.last_mut() {
            batch_slot.push((idx, result_id));
        }
    }

    let total_results: usize = batches.iter().map(Vec::len).sum();
    if total_results <= keep_last_n {
        return;
    }

    // Walk batches newest-first, keeping whole batches until the cumulative
    // kept count reaches `keep_last_n`. The straddling batch (the one that
    // crosses the count threshold) is kept entirely — round up, never split.
    let mut kept_count: usize = 0;
    let mut keep_from_batch: usize = batches.len();
    for (rev_i, batch) in batches.iter().enumerate().rev() {
        kept_count = kept_count.saturating_add(batch.len());
        if kept_count >= keep_last_n {
            keep_from_batch = rev_i;
            break;
        }
    }

    let keep_ids: HashSet<String> = batches[keep_from_batch..]
        .iter()
        .flat_map(|kept_batch| kept_batch.iter().map(|(_, id)| id))
        .cloned()
        .collect();

    messages.retain(|m| match get_tool_result_id(m) {
        Some(id) => keep_ids.contains(&id),
        None => true,
    });
}

fn truncate_old_assistant_messages(messages: &mut Vec<Value>, keep_last_n: usize) {
    if keep_last_n == usize::MAX {
        return;
    }

    let assistant_indices: Vec<usize> = messages
        .iter()
        .enumerate()
        .filter_map(|(idx, m)| {
            if get_role(m) == "assistant" {
                Some(idx)
            } else {
                None
            }
        })
        .collect();

    if assistant_indices.len() <= keep_last_n {
        return;
    }

    let keep_start = assistant_indices.len().saturating_sub(keep_last_n);
    let keep_indices: HashSet<usize> = assistant_indices.into_iter().skip(keep_start).collect();

    for (idx, message) in messages.iter_mut().enumerate() {
        if get_role(message) != "assistant" || keep_indices.contains(&idx) {
            continue;
        }

        let has_tool_calls = message
            .get("tool_calls")
            .and_then(|tc| tc.as_array())
            .map(|a| !a.is_empty())
            .unwrap_or(false);

        if has_tool_calls {
            if let Some(content) = message.get("content").and_then(|c| c.as_str()) {
                if !content.is_empty() {
                    tracing::warn!(
                        target: "cache_hygiene",
                        role = "assistant",
                        has_tool_calls = true,
                        content_len = content.len(),
                        "hygiene_truncated_assistant_with_tool_calls — bytes mutate mid-session, busts prompt prefix cache"
                    );
                    message["content"] = Value::String(TRUNCATED_ASSISTANT_PLACEHOLDER.to_string());
                }
            }
        } else {
            let content = message
                .get("content")
                .and_then(|c| c.as_str())
                .unwrap_or("");
            if !content.is_empty() {
                tracing::warn!(
                    target: "cache_hygiene",
                    role = "assistant",
                    has_tool_calls = false,
                    content_len = content.len(),
                    "hygiene_truncated_assistant_text — bytes mutate mid-session, busts prompt prefix cache"
                );
                message["content"] = Value::String(TRUNCATED_ASSISTANT_PLACEHOLDER.to_string());
            }
        }
    }
}

pub fn strip_dangling_tool_calls(messages: &mut Vec<Value>) {
    for idx in 0..messages.len() {
        let tool_call_ids: Vec<String> = get_tool_call_ids(&messages[idx]);
        if tool_call_ids.is_empty() {
            continue;
        }

        let mut all_results: HashSet<String> = HashSet::new();
        let mut next_idx = idx + 1;
        while next_idx < messages.len() && get_role(&messages[next_idx]) == "tool" {
            if let Some(id) = get_tool_result_id(&messages[next_idx]) {
                all_results.insert(id);
            }
            next_idx += 1;
        }

        // A tool call is "dangling" only when its OWN matching result is
        // absent from the immediate trailing tool-run. Strip just those calls
        // and keep the ones whose results survived the prior keep-window
        // pass. The previous all-or-nothing check (`iter().all(...)`) wiped
        // the WHOLE batch whenever any sibling was missing — which destroyed
        // retention-driven partial sets produced by `drop_old_tool_results`
        // at the keep boundary, so the last N tool results the caller asked
        // to keep could be lost along with the assistant carrier.
        let dangling_ids: HashSet<String> = tool_call_ids
            .iter()
            .filter(|id| !all_results.contains(id.as_str()))
            .cloned()
            .collect();

        if dangling_ids.is_empty() {
            continue;
        }

        if let Some(tool_calls) = messages[idx]
            .get("tool_calls")
            .and_then(|t| t.as_array())
            .cloned()
        {
            let retained_calls: Vec<Value> = tool_calls
                .into_iter()
                .filter(|call| match call.get("id").and_then(|v| v.as_str()) {
                    Some(call_id) => !dangling_ids.contains(call_id),
                    None => true,
                })
                .collect();
            messages[idx]["tool_calls"] = Value::Array(retained_calls);
        }
    }

    messages.retain(|m| {
        let role = get_role(m);
        if role == "assistant" {
            let content = get_content(m);
            let has_tool_calls = m
                .get("tool_calls")
                .and_then(|tc| tc.as_array())
                .map(|a| !a.is_empty())
                .unwrap_or(false);
            !content.is_empty() || has_tool_calls
        } else {
            true
        }
    });
}

pub fn remove_orphaned_tool_results(messages: &mut Vec<Value>) {
    let mut seen_tool_calls: HashSet<String> = HashSet::new();

    for message in messages.iter_mut() {
        let ids = get_tool_call_ids(message);
        seen_tool_calls.extend(ids);

        if get_role(message) == "tool" {
            if let Some(id) = get_tool_result_id(message) {
                if !seen_tool_calls.contains(&id) {
                    message["content"] = Value::String(String::new());
                }
            }
        }
    }

    messages.retain(|m| {
        if get_role(m) == "tool" {
            !get_content(m).is_empty()
        } else {
            true
        }
    });
}

fn get_role(message: &Value) -> &str {
    message.get("role").and_then(|r| r.as_str()).unwrap_or("")
}

fn get_content(message: &Value) -> String {
    message
        .get("content")
        .and_then(|c| c.as_str())
        .unwrap_or("")
        .to_string()
}

fn get_tool_result_id(message: &Value) -> Option<String> {
    if get_role(message) == "tool" {
        message
            .get("tool_call_id")
            .and_then(|id| id.as_str())
            .map(|s| s.to_string())
    } else {
        None
    }
}

fn get_tool_call_ids(message: &Value) -> Vec<String> {
    message
        .get("tool_calls")
        .and_then(|tc| tc.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|tc| {
                    tc.get("id")
                        .and_then(|id| id.as_str())
                        .map(|s| s.to_string())
                })
                .collect()
        })
        .unwrap_or_default()
}

fn get_tool_result_ids_set(message: &Value) -> HashSet<String> {
    if get_role(message) == "tool" {
        get_tool_result_id(message).into_iter().collect()
    } else {
        HashSet::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Live-shape repro: a plain alternating user/assistant history (with
    /// `_db_id` tags as supplied by get_history) must pass hygiene intact —
    /// bisecting a wire bug where 28 ctx messages reached the provider as 3.
    #[test]
    fn test_hygiene_preserves_plain_alternating_history() {
        let mut messages = vec![json!({"role": "system", "content": "sys"})];
        messages.push(json!({"role": "developer", "content": "protocol"}));
        for i in 0..12 {
            messages.push(json!({"role": "user", "content": format!("q{i}"), "_db_id": 2*i+1}));
            messages
                .push(json!({"role": "assistant", "content": format!("a{i}"), "_db_id": 2*i+2}));
        }
        messages.push(json!({"role": "user", "content": "current question"}));
        let before = messages.len(); // 27

        hygiene_pipeline(&mut messages, 20);

        assert_eq!(
            messages.len(),
            before,
            "hygiene must not delete plain user/assistant turns; survivors: {:?}",
            messages
                .iter()
                .map(|m| m.get("role").and_then(|r| r.as_str()).unwrap_or("?"))
                .collect::<Vec<_>>()
        );
    }

    fn assistant_with_tool_call(id: &str) -> Value {
        json!({
            "role": "assistant",
            "content": "Let me check that.",
            "tool_calls": [{
                "id": id,
                "name": "read_file",
                "arguments": {"path": "/tmp/test"}
            }]
        })
    }

    fn assistant_plain(content: &str) -> Value {
        json!({
            "role": "assistant",
            "content": content
        })
    }

    fn tool_result(id: &str, content: &str) -> Value {
        json!({
            "role": "tool",
            "tool_call_id": id,
            "content": content
        })
    }

    fn user_message(content: &str) -> Value {
        json!({
            "role": "user",
            "content": content
        })
    }

    /// Branches of the unified body-shrink primitive not pinned elsewhere:
    /// pass-through (`None`) for both policies, and the UTF-8 char-boundary
    /// backoff in `ByteCap`. The shrink outputs themselves are pinned by the
    /// filters.rs and token_budget.rs suites (byte-stability contracts).
    #[test]
    fn test_shrink_tool_body_pass_through_and_char_boundary() {
        // Small bodies pass through unchanged under both policies.
        assert_eq!(shrink_tool_body("ok", ToolBodyPolicy::ByteCap(8000)), None);
        assert_eq!(
            shrink_tool_body("ok", ToolBodyPolicy::Digest { preview_len: 200 }),
            None
        );

        // Cap landing inside a multi-byte char must back off to a boundary.
        let s = format!("{}é tail", "x".repeat(9)); // 'é' spans bytes 9..11
        let capped = shrink_tool_body(&s, ToolBodyPolicy::ByteCap(10)).unwrap();
        assert_eq!(capped, "xxxxxxxxx\n...[tool output truncated]");

        // Over-threshold digest emits the digest marker.
        let long = "y".repeat(300);
        let digest = shrink_tool_body(&long, ToolBodyPolicy::Digest { preview_len: 200 }).unwrap();
        assert!(digest.starts_with("TOOL_OUTPUT_DIGEST v1 | sha256:"));
        assert!(digest.contains("len:300"));
    }

    #[test]
    fn test_dedup_keeps_last_tool_result_per_call_id() {
        let mut messages = vec![
            assistant_with_tool_call("tc_1"),
            tool_result("tc_1", "old result"),
            tool_result("tc_1", "new result"),
        ];
        dedup_tool_results(&mut messages);

        assert_eq!(messages.len(), 2);
        assert!(messages[1]["content"]
            .as_str()
            .unwrap()
            .contains("new result"));
    }

    #[test]
    fn test_strip_dangling_without_immediate_result() {
        let mut messages = vec![
            assistant_with_tool_call("tc_1"),
            user_message("next prompt"),
            tool_result("tc_1", "late result"),
        ];
        strip_dangling_tool_calls(&mut messages);

        let tool_calls = messages[0].get("tool_calls").and_then(|tc| tc.as_array());
        assert!(tool_calls.is_none() || tool_calls.unwrap().is_empty());
    }

    #[test]
    fn test_remove_orphaned_without_matching_call() {
        let mut messages = vec![
            tool_result("tc_orphan", "orphan result"),
            assistant_with_tool_call("tc_1"),
            tool_result("tc_1", "valid result"),
        ];
        remove_orphaned_tool_results(&mut messages);

        assert_eq!(messages.len(), 2);
        assert_eq!(get_role(&messages[0]), "assistant");
    }

    #[test]
    fn test_merge_consecutive_user_messages() {
        let mut messages = vec![user_message("hello"), user_message("world")];
        merge_consecutive_same_role(&mut messages);

        assert_eq!(messages.len(), 1);
        assert!(messages[0]["content"].as_str().unwrap().contains("hello"));
        assert!(messages[0]["content"].as_str().unwrap().contains("world"));
    }

    #[test]
    fn test_merge_does_not_merge_tool_messages() {
        let mut messages = vec![
            tool_result("tc_1", "result 1"),
            tool_result("tc_2", "result 2"),
        ];
        merge_consecutive_same_role(&mut messages);

        assert_eq!(messages.len(), 2);
    }

    #[test]
    fn test_full_pipeline_cleans_messy_context() {
        let mut messages = vec![
            tool_result("tc_orphan", "orphan"),
            assistant_with_tool_call("tc_old"),
            tool_result("tc_old", "old result"),
            tool_result("tc_old", "duplicate result"),
            assistant_with_tool_call("tc_new"),
            tool_result("tc_new", "valid result"),
        ];
        hygiene_pipeline(&mut messages, 20);

        assert!(messages.len() <= 4);
    }

    #[test]
    fn test_empty_messages_no_panic() {
        let mut messages: Vec<Value> = vec![];
        hygiene_pipeline(&mut messages, 20);
        assert!(messages.is_empty());
    }

    #[test]
    fn test_preserve_non_tool_conversation() {
        let mut messages = vec![
            user_message("hello"),
            assistant_plain("hi there"),
            user_message("how are you"),
            assistant_plain("doing well"),
        ];
        let before = messages.len();
        hygiene_pipeline(&mut messages, 20);
        assert_eq!(messages.len(), before);
    }

    #[test]
    fn test_truncate_old_assistant_messages_keeps_recent() {
        let mut messages = vec![
            assistant_plain("old 1"),
            assistant_plain("old 2"),
            assistant_plain("recent 1"),
            assistant_plain("recent 2"),
        ];
        truncate_old_assistant_messages(&mut messages, 2);

        assert_eq!(messages.len(), 4);
        assert!(messages[0]["content"]
            .as_str()
            .unwrap()
            .contains("truncated"));
        assert!(messages[1]["content"]
            .as_str()
            .unwrap()
            .contains("truncated"));
        assert_eq!(messages[2]["content"], "recent 1");
        assert_eq!(messages[3]["content"], "recent 2");
    }

    #[test]
    fn test_drop_old_tool_results_keeps_last_n() {
        let mut messages = vec![
            assistant_with_tool_call("tc_1"),
            tool_result("tc_1", "old"),
            assistant_with_tool_call("tc_2"),
            tool_result("tc_2", "newer"),
            assistant_with_tool_call("tc_3"),
            tool_result("tc_3", "newest"),
        ];
        drop_old_tool_results(&mut messages, 2);

        let remaining_results: Vec<_> = messages.iter().filter(|m| get_role(m) == "tool").collect();
        assert!(remaining_results.len() <= 2);
    }

    // Router tool injections tagged with _synthetic should NOT be merged with the preceding user prompt
    #[test]
    fn test_router_tool_injection_not_merged_with_user_prompt() {
        let mut messages = vec![
            user_message("Fetch https://news.ycombinator.com and summarize top 5"),
            json!({
                "role": "user",
                "content": "[router:tool:web_fetch] <html>Hacker News content...</html>",
                "_synthetic": true
            }),
        ];
        merge_consecutive_same_role(&mut messages);

        // The user's original prompt must remain in a SEPARATE message from the tool result.
        // The _synthetic flag prevents the merge function from combining them.
        assert_eq!(
            messages.len(),
            2,
            "router:tool injection must stay in its own message, not merged with user prompt"
        );
        let first_content = messages[0]["content"].as_str().unwrap();
        let second_content = messages[1]["content"].as_str().unwrap();
        assert!(
            first_content.contains("Fetch https://news.ycombinator.com"),
            "user prompt must be in the first message"
        );
        assert!(
            second_content.contains("[router:tool:web_fetch]"),
            "router tool injection must be in the second message"
        );
    }

    // RED: specialist injections tagged with _synthetic should NOT be merged
    #[test]
    fn test_specialist_injection_not_merged_with_user_prompt() {
        let mut messages = vec![
            user_message("Original question"),
            json!({
                "role": "user",
                "content": "[specialist:coding] Here is the specialist result...",
                "_synthetic": true
            }),
        ];
        merge_consecutive_same_role(&mut messages);

        // The specialist injection should stay separate from the user's original question.
        // This test will FAIL because merge_consecutive_same_role does not check _synthetic.
        assert_eq!(
            messages.len(),
            2,
            "specialist injection must stay in its own message, not merged with user prompt"
        );
        let first_content = messages[0]["content"].as_str().unwrap();
        let second_content = messages[1]["content"].as_str().unwrap();
        assert!(
            first_content.contains("Original question"),
            "user prompt must be in the first message"
        );
        assert!(
            second_content.contains("[specialist:coding]"),
            "specialist injection must be in the second message"
        );
    }

    #[test]
    fn test_dangling_tool_call_with_valid_result_preserved() {
        let mut messages = vec![
            assistant_with_tool_call("tc_1"),
            tool_result("tc_1", "valid result"),
            user_message("thanks"),
        ];
        strip_dangling_tool_calls(&mut messages);

        let tool_calls = messages[0].get("tool_calls").and_then(|tc| tc.as_array());
        assert!(tool_calls.is_some() && !tool_calls.unwrap().is_empty());
    }

    #[test]
    fn test_merge_preserves_turn_tag() {
        // Two consecutive user messages; the first carries _turn: 5.
        // After merging, the resulting message must still have _turn: 5.
        let mut messages = vec![
            json!({
                "role": "user",
                "content": "first message",
                "_turn": 5
            }),
            json!({
                "role": "user",
                "content": "second message"
            }),
        ];
        merge_consecutive_same_role(&mut messages);

        assert_eq!(
            messages.len(),
            1,
            "two consecutive user messages should be merged into one"
        );
        assert!(
            messages[0]["content"]
                .as_str()
                .unwrap()
                .contains("first message"),
            "merged content must include first message"
        );
        assert!(
            messages[0]["content"]
                .as_str()
                .unwrap()
                .contains("second message"),
            "merged content must include second message"
        );
        assert_eq!(
            messages[0].get("_turn").and_then(|v| v.as_u64()),
            Some(5),
            "_turn metadata from the first message must be preserved after merge"
        );
    }

    #[test]
    fn test_multiple_tool_calls_in_single_message() {
        let mut messages = vec![
            json!({
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "tc_1", "name": "read", "arguments": {}},
                    {"id": "tc_2", "name": "exec", "arguments": {}}
                ]
            }),
            json!({
                "role": "tool",
                "tool_call_id": "tc_1",
                "content": "result 1"
            }),
            json!({
                "role": "tool",
                "tool_call_id": "tc_2",
                "content": "result 2"
            }),
        ];
        strip_dangling_tool_calls(&mut messages);

        assert_eq!(messages.len(), 3);
    }

    #[test]
    fn test_partial_tool_results_strips_only_dangling_calls() {
        let mut messages = vec![
            json!({
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "tc_1", "name": "read", "arguments": {}},
                    {"id": "tc_2", "name": "exec", "arguments": {}}
                ]
            }),
            json!({
                "role": "tool",
                "tool_call_id": "tc_1",
                "content": "only tc_1 has result"
            }),
        ];
        strip_dangling_tool_calls(&mut messages);

        // Only the dangling call (tc_2, no result) is stripped; the matched
        // call (tc_1) and its result survive. Previously this wiped *all*
        // tool_calls, which destroyed retention-driven partial sets produced
        // by `drop_old_tool_results` at the keep boundary (see
        // `test_pipeline_preserves_straddling_multi_call_batch`).
        let tool_calls = messages[0].get("tool_calls").and_then(|tc| tc.as_array());
        assert!(
            tool_calls.is_some(),
            "assistant with a surviving call must not be dropped"
        );
        let kept_ids: Vec<String> = tool_calls
            .unwrap()
            .iter()
            .filter_map(|c| c.get("id").and_then(|v| v.as_str()).map(String::from))
            .collect();
        assert_eq!(kept_ids, vec!["tc_1".to_string()]);
        assert_eq!(messages.len(), 2, "matched call + its result remain");
        assert_eq!(get_role(&messages[1]), "tool");
        assert_eq!(get_tool_result_id(&messages[1]), Some("tc_1".to_string()));
    }

    /// `strip_dangling_tool_calls` must strip ONLY the call whose result is
    /// missing and keep the rest; a 3-call assistant with one missing result
    /// keeps the 2 matched calls and their results.
    #[test]
    fn test_strip_dangling_strips_only_missing_calls() {
        let mut messages = vec![
            json!({
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "tc_1", "name": "read", "arguments": {}},
                    {"id": "tc_2", "name": "exec", "arguments": {}},
                    {"id": "tc_3", "name": "exec", "arguments": {}}
                ]
            }),
            json!({"role":"tool","tool_call_id":"tc_1","content":"r1"}),
            json!({"role":"tool","tool_call_id":"tc_2","content":"r2"}),
            // tc_3 result missing -> dangling
        ];
        strip_dangling_tool_calls(&mut messages);

        let kept_ids: Vec<String> = messages[0]
            .get("tool_calls")
            .and_then(|tc| tc.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|c| c.get("id").and_then(|v| v.as_str()).map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        // Only tc_3 is stripped; tc_1 and tc_2 survive. The old all-or-nothing
        // wipe would have cleared all three.
        assert_eq!(kept_ids, vec!["tc_1".to_string(), "tc_2".to_string()]);
        assert_eq!(messages.len(), 3, "matched assistant + 2 results remain");
        let result_ids: Vec<String> = messages
            .iter()
            .skip(1)
            .filter_map(|m| get_tool_result_id(m))
            .collect();
        assert_eq!(result_ids, vec!["tc_1".to_string(), "tc_2".to_string()]);
    }

    /// `drop_old_tool_results` must keep/drop whole multi-call batches: when a
    /// 3-call batch straddles the keep boundary (keep_last_n=2), the whole
    /// batch is kept and the older batch is dropped — never a partial sibling
    /// set that downstream passes could mistake for corruption.
    #[test]
    fn test_drop_old_tool_results_never_splits_multi_call_batch() {
        let mut messages = vec![
            json!({"role":"assistant","content":"","tool_calls":[
                {"id":"old_1","name":"read","arguments":{}},
                {"id":"old_2","name":"read","arguments":{}}]}),
            json!({"role":"tool","tool_call_id":"old_1","content":"o1"}),
            json!({"role":"tool","tool_call_id":"old_2","content":"o2"}),
            json!({"role":"assistant","content":"","tool_calls":[
                {"id":"new_1","name":"read","arguments":{}},
                {"id":"new_2","name":"read","arguments":{}},
                {"id":"new_3","name":"read","arguments":{}}]}),
            json!({"role":"tool","tool_call_id":"new_1","content":"n1"}),
            json!({"role":"tool","tool_call_id":"new_2","content":"n2"}),
            json!({"role":"tool","tool_call_id":"new_3","content":"n3"}),
        ];

        drop_old_tool_results(&mut messages, 2);

        let survivor_ids: HashSet<String> = messages
            .iter()
            .filter_map(|m| get_tool_result_id(m))
            .collect();
        assert_eq!(
            survivor_ids,
            ["new_1", "new_2", "new_3"]
                .iter()
                .map(|s| s.to_string())
                .collect::<HashSet<_>>(),
            "straddling multi-call batch must be kept whole; got {:?}",
            survivor_ids
        );
        assert!(!survivor_ids.contains("old_1"));
        assert!(!survivor_ids.contains("old_2"));
    }

    /// End-to-end reproduction of the reported bug: a multi-call batch
    /// straddling the keep boundary used to yield `messages: []` after the
    /// full pipeline; the last N tool results must now survive.
    #[test]
    fn test_pipeline_preserves_straddling_multi_call_batch() {
        let mut messages = vec![
            // older multi-call batch
            json!({"role":"assistant","content":"","tool_calls":[
                {"id":"old_1","name":"read","arguments":{}},
                {"id":"old_2","name":"read","arguments":{}}]}),
            json!({"role":"tool","tool_call_id":"old_1","content":"o1"}),
            json!({"role":"tool","tool_call_id":"old_2","content":"o2"}),
            // multi-call batch that straddles the keep boundary
            json!({"role":"assistant","content":"","tool_calls":[
                {"id":"new_1","name":"read","arguments":{}},
                {"id":"new_2","name":"read","arguments":{}},
                {"id":"new_3","name":"read","arguments":{}}]}),
            json!({"role":"tool","tool_call_id":"new_1","content":"n1"}),
            json!({"role":"tool","tool_call_id":"new_2","content":"n2"}),
            json!({"role":"tool","tool_call_id":"new_3","content":"n3"}),
        ];

        hygiene_pipeline(&mut messages, 2);

        let survivor_ids: Vec<String> = messages
            .iter()
            .filter_map(|m| get_tool_result_id(m))
            .collect();
        assert!(
            survivor_ids.contains(&"new_2".to_string())
                && survivor_ids.contains(&"new_3".to_string()),
            "last 2 tool results must survive the pipeline; got {:?}",
            survivor_ids
        );
        assert!(!survivor_ids.contains(&"old_1".to_string()));
        assert!(!survivor_ids.contains(&"old_2".to_string()));
        // The recent multi-call assistant carrier must survive (not wiped).
        let has_recent_assistant = messages.iter().any(|m| {
            get_role(m) == "assistant"
                && m.get("tool_calls")
                    .and_then(|tc| tc.as_array())
                    .is_some_and(|a| !a.is_empty())
        });
        assert!(
            has_recent_assistant,
            "recent multi-call assistant must survive; messages: {:?}",
            messages
        );
        // Every surviving tool result must have a matching surviving tool call.
        let surviving_calls: HashSet<String> =
            messages.iter().flat_map(|m| get_tool_call_ids(m)).collect();
        for id in &survivor_ids {
            assert!(
                surviving_calls.contains(id),
                "orphaned result {} survived without a matching call",
                id
            );
        }
    }

    /// Single-call batches at the boundary never triggered the destructive
    /// interaction; the fix must not regress that shape.
    #[test]
    fn test_pipeline_single_call_batch_straddling_survives() {
        let mut messages = vec![
            assistant_with_tool_call("old"),
            tool_result("old", "old result"),
            assistant_with_tool_call("new"),
            tool_result("new", "new result"),
        ];
        hygiene_pipeline(&mut messages, 1);

        let survivor_ids: Vec<String> = messages
            .iter()
            .filter_map(|m| get_tool_result_id(m))
            .collect();
        assert_eq!(survivor_ids, vec!["new".to_string()]);
        assert!(
            messages.iter().any(|m| {
                get_role(m) == "assistant"
                    && m.get("tool_calls")
                        .and_then(|tc| tc.as_array())
                        .is_some_and(|a| !a.is_empty())
            }),
            "newest single-call assistant must survive; messages: {:?}",
            messages
        );
    }

    #[test]
    fn test_tool_result_ok_normalizes_known_failure_markers() {
        assert!(!tool_result_ok("Error: file missing"));
        assert!(!tool_result_ok(
            "[VERBATIM TOOL OUTPUT — do not paraphrase]\nError: file missing\n[END TOOL OUTPUT]"
        ));
        assert!(!tool_result_ok("response boundary: exec was not executed"));
        assert!(!tool_result_ok("(no result)"));
        assert!(!tool_result_ok(
            "No stored output for tool_call_id='missing' in this session"
        ));
        assert!(tool_result_ok("Finished dev profile"));
    }
}
