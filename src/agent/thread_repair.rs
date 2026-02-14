//! Message thread repair for OpenAI protocol compliance.
//!
//! Scans a message array for protocol violations (orphaned tool calls,
//! orphaned tool results, consecutive user messages, etc.) and repairs
//! them before sending to the LLM.

use std::collections::HashSet;

use serde_json::{json, Value};
use tracing::warn;

/// Repair protocol violations in a message array.
///
/// Fixes:
/// 1. Orphaned tool_use: assistant messages with tool_calls but no
///    corresponding tool result messages following them.
/// 2. Orphaned tool results: tool role messages with no preceding
///    assistant message containing the matching tool_call_id.
/// 3. Consecutive user messages (can happen after history truncation).
/// 4. First non-system message must be user role.
pub fn repair_messages(messages: &mut Vec<Value>) {
    if messages.is_empty() {
        return;
    }

    // Pass 0: Deduplicate tool results with the same tool_call_id.
    // Keeps only the first result per ID. This prevents HTTP 400 errors when
    // duplicate results are written due to stale new_start tracking.
    dedup_tool_results(messages);

    // Pass 1: Remove positionally-orphaned tool results (tool messages whose
    // matching assistant doesn't PRECEDE them). Must run before fix_orphaned_tool_calls
    // so that misplaced results don't suppress synthetic result generation.
    fix_orphaned_tool_results(messages);

    // Pass 2: Fix orphaned tool_use (assistant with tool_calls but missing results).
    // Runs after pass 1 because pass 1 may have removed misplaced results.
    fix_orphaned_tool_calls(messages);

    // Pass 3: Merge consecutive user messages.
    merge_consecutive_user_messages(messages);

    // Pass 4: Ensure first non-system message is user role.
    ensure_user_first(messages);

    // Pass 5: Ensure last message is not assistant role.
    // The Anthropic OpenAI-compat endpoint rejects conversations ending with
    // an assistant message ("does not support assistant message prefill").
    ensure_not_ending_with_assistant(messages);

    // Final validation: warn about any remaining protocol issues (shouldn't happen).
    debug_validate(messages);
}

/// Log warnings for any remaining protocol violations after repair.
/// This is a safety net for debugging — it doesn't modify messages.
fn debug_validate(messages: &[Value]) {
    let mut known_call_ids: HashSet<String> = HashSet::new();

    for (i, msg) in messages.iter().enumerate() {
        let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("");

        if role == "assistant" {
            if let Some(tcs) = msg.get("tool_calls").and_then(|tc| tc.as_array()) {
                for tc in tcs {
                    if let Some(id) = tc.get("id").and_then(|id| id.as_str()) {
                        known_call_ids.insert(id.to_string());
                    }
                }
            }
        } else if role == "tool" {
            let call_id = msg
                .get("tool_call_id")
                .and_then(|id| id.as_str())
                .unwrap_or_default();
            if !call_id.is_empty() && !known_call_ids.contains(call_id) {
                warn!(
                    "PROTOCOL BUG: tool result at index {} has tool_call_id '{}' \
                     not found in any preceding assistant's tool_calls",
                    i, call_id
                );
            }
        }
    }
}

/// Remove duplicate tool results with the same `tool_call_id`.
/// Keeps only the first result per ID.
fn dedup_tool_results(messages: &mut Vec<Value>) {
    let mut seen_ids: HashSet<String> = HashSet::new();
    let mut remove_indices: Vec<usize> = Vec::new();
    for (i, msg) in messages.iter().enumerate() {
        if msg.get("role").and_then(|r| r.as_str()) != Some("tool") {
            continue;
        }
        let call_id = msg
            .get("tool_call_id")
            .and_then(|id| id.as_str())
            .unwrap_or_default();
        if !call_id.is_empty() && !seen_ids.insert(call_id.to_string()) {
            remove_indices.push(i);
        }
    }
    for &i in remove_indices.iter().rev() {
        warn!(
            "Removing duplicate tool result at index {} (id: {})",
            i,
            messages[i]
                .get("tool_call_id")
                .and_then(|id| id.as_str())
                .unwrap_or("?")
        );
        messages.remove(i);
    }
}

/// Find assistant messages with tool_calls that don't have corresponding
/// tool result messages. Append synthetic error results for missing ones.
fn fix_orphaned_tool_calls(messages: &mut Vec<Value>) {
    // Collect all tool_call_ids that have results.
    let result_ids: HashSet<String> = messages
        .iter()
        .filter(|m| m.get("role").and_then(|r| r.as_str()) == Some("tool"))
        .filter_map(|m| {
            m.get("tool_call_id")
                .and_then(|id| id.as_str())
                .map(String::from)
        })
        .collect();

    // Find assistant messages with tool_calls that are missing results.
    let mut synthetic_results: Vec<(usize, Vec<Value>)> = Vec::new();

    for (i, msg) in messages.iter().enumerate() {
        if msg.get("role").and_then(|r| r.as_str()) != Some("assistant") {
            continue;
        }
        let tool_calls = match msg.get("tool_calls").and_then(|tc| tc.as_array()) {
            Some(tc) => tc,
            None => continue,
        };

        let mut missing: Vec<Value> = Vec::new();
        for tc in tool_calls {
            let call_id = tc.get("id").and_then(|id| id.as_str()).unwrap_or_default();
            if !call_id.is_empty() && !result_ids.contains(call_id) {
                let tool_name = tc
                    .get("function")
                    .and_then(|f| f.get("name"))
                    .and_then(|n| n.as_str())
                    .unwrap_or("unknown");
                missing.push(json!({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": tool_name,
                    "content": "[Error: tool call interrupted, result unavailable]"
                }));
            }
        }

        if !missing.is_empty() {
            synthetic_results.push((i, missing));
        }
    }

    // Insert synthetic results right after their assistant message.
    // Process in reverse order so indices stay valid.
    for (idx, results) in synthetic_results.into_iter().rev() {
        let insert_at = idx + 1;
        for (j, result) in results.into_iter().enumerate() {
            messages.insert(insert_at + j, result);
        }
    }
}

/// Remove tool result messages that don't have a matching tool_call_id
/// in a PRECEDING assistant message.
///
/// Uses a forward scan: tool_call_ids are only "known" once their
/// assistant message has been seen, ensuring positional correctness.
fn fix_orphaned_tool_results(messages: &mut Vec<Value>) {
    // Forward scan: accumulate known_call_ids as we encounter assistants,
    // then mark tool results whose ID hasn't been seen yet for removal.
    let mut known_call_ids: HashSet<String> = HashSet::new();
    let mut remove_indices: Vec<usize> = Vec::new();

    for (i, msg) in messages.iter().enumerate() {
        let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("");

        if role == "assistant" {
            if let Some(tool_calls) = msg.get("tool_calls").and_then(|tc| tc.as_array()) {
                for tc in tool_calls {
                    if let Some(id) = tc.get("id").and_then(|id| id.as_str()) {
                        known_call_ids.insert(id.to_string());
                    }
                }
            }
        } else if role == "tool" {
            let call_id = msg
                .get("tool_call_id")
                .and_then(|id| id.as_str())
                .unwrap_or_default();
            if call_id.is_empty() || !known_call_ids.contains(call_id) {
                remove_indices.push(i);
            }
        }
    }

    // Remove in reverse order to preserve indices.
    for &i in remove_indices.iter().rev() {
        messages.remove(i);
    }
}

/// Merge consecutive user messages into a single message.
fn merge_consecutive_user_messages(messages: &mut Vec<Value>) {
    let mut i = 0;
    while i + 1 < messages.len() {
        let is_user = messages[i].get("role").and_then(|r| r.as_str()) == Some("user");
        let next_is_user = messages[i + 1].get("role").and_then(|r| r.as_str()) == Some("user");

        if is_user && next_is_user {
            // Merge content from messages[i+1] into messages[i].
            let next_content = messages[i + 1]
                .get("content")
                .and_then(|c| c.as_str())
                .unwrap_or_default()
                .to_string();
            let current_content = messages[i]
                .get("content")
                .and_then(|c| c.as_str())
                .unwrap_or_default()
                .to_string();

            messages[i]["content"] =
                Value::String(format!("{}\n\n{}", current_content, next_content));
            messages.remove(i + 1);
            // Don't increment i — check if the next message is also user.
        } else {
            i += 1;
        }
    }
}

/// Ensure the first non-system message is a user message.
/// If the first non-system message is assistant, prepend a synthetic user message.
fn ensure_user_first(messages: &mut Vec<Value>) {
    let first_non_system = messages
        .iter()
        .position(|m| m.get("role").and_then(|r| r.as_str()) != Some("system"));

    if let Some(idx) = first_non_system {
        let role = messages[idx]
            .get("role")
            .and_then(|r| r.as_str())
            .unwrap_or_default();
        if role == "assistant" || role == "tool" {
            messages.insert(
                idx,
                json!({
                    "role": "user",
                    "content": "[Conversation resumed]"
                }),
            );
        }
    }
}

/// Ensure the conversation does not end with an assistant message.
/// The Anthropic OpenAI-compat endpoint rejects conversations that end
/// with role "assistant" (treats it as unsupported prefill). If the last
/// message is assistant, append a synthetic user continuation.
fn ensure_not_ending_with_assistant(messages: &mut Vec<Value>) {
    if let Some(last) = messages.last() {
        let role = last.get("role").and_then(|r| r.as_str()).unwrap_or("");
        if role == "assistant" {
            let content_preview: String = last.get("content")
                .and_then(|c| c.as_str())
                .unwrap_or("")
                .chars()
                .take(80)
                .collect();
            let has_tool_calls = last.get("tool_calls").and_then(|tc| tc.as_array()).map(|a| a.len()).unwrap_or(0);
            warn!(
                "Messages end with assistant role — appending user continuation to prevent prefill error \
                 (msg_count={}, content=\"{}\", tool_calls={})",
                messages.len(), content_preview, has_tool_calls
            );
            messages.push(json!({
                "role": "user",
                "content": "[system] Continue with the task."
            }));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_repair_empty_messages() {
        let mut messages: Vec<Value> = vec![];
        repair_messages(&mut messages);
        assert!(messages.is_empty());
    }

    #[test]
    fn test_repair_valid_messages_unchanged() {
        // Conversation ending with user (no assistant prefill issue).
        let mut messages = vec![
            json!({"role": "system", "content": "System prompt"}),
            json!({"role": "user", "content": "Hello"}),
        ];
        let original_len = messages.len();
        repair_messages(&mut messages);
        assert_eq!(messages.len(), original_len);
    }

    #[test]
    fn test_fix_orphaned_tool_calls() {
        let mut messages = vec![
            json!({"role": "system", "content": "System"}),
            json!({"role": "user", "content": "Read a file"}),
            json!({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "tc_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"}
                }]
            }),
            // Missing tool result for tc_1!
            json!({"role": "user", "content": "What happened?"}),
        ];

        repair_messages(&mut messages);

        // Should have inserted a synthetic tool result after the assistant message.
        assert_eq!(messages.len(), 5);
        assert_eq!(messages[3]["role"], "tool");
        assert_eq!(messages[3]["tool_call_id"], "tc_1");
        assert!(messages[3]["content"]
            .as_str()
            .unwrap()
            .contains("interrupted"));
    }

    #[test]
    fn test_fix_orphaned_tool_results() {
        let mut messages = vec![
            json!({"role": "system", "content": "System"}),
            json!({"role": "user", "content": "Hello"}),
            // Tool result with no matching assistant tool_call.
            json!({
                "role": "tool",
                "tool_call_id": "nonexistent_id",
                "name": "read_file",
                "content": "Some old result"
            }),
            json!({"role": "assistant", "content": "OK"}),
        ];

        repair_messages(&mut messages);

        // The orphaned tool result should be removed.
        // Remaining: system + user + assistant + user(anti-prefill) = 4
        assert_eq!(messages.len(), 4);
        assert!(!messages
            .iter()
            .any(|m| m.get("role").and_then(|r| r.as_str()) == Some("tool")));
        assert_eq!(
            messages.last().unwrap()["role"].as_str(),
            Some("user"),
            "Last message should be user continuation"
        );
    }

    #[test]
    fn test_merge_consecutive_user_messages() {
        let mut messages = vec![
            json!({"role": "system", "content": "System"}),
            json!({"role": "user", "content": "First message"}),
            json!({"role": "user", "content": "Second message"}),
            json!({"role": "user", "content": "Third message"}),
            json!({"role": "assistant", "content": "Reply"}),
        ];

        repair_messages(&mut messages);

        // Three user messages should be merged into one.
        // system + merged_user + assistant + user(anti-prefill) = 4
        assert_eq!(messages.len(), 4);
        let user_content = messages[1]["content"].as_str().unwrap();
        assert!(user_content.contains("First message"));
        assert!(user_content.contains("Second message"));
        assert!(user_content.contains("Third message"));
    }

    #[test]
    fn test_ensure_user_first_after_system() {
        let mut messages = vec![
            json!({"role": "system", "content": "System"}),
            json!({"role": "assistant", "content": "I was here first"}),
            json!({"role": "user", "content": "Hello"}),
        ];

        repair_messages(&mut messages);

        // Should have inserted a synthetic user message before the assistant.
        assert_eq!(messages[1]["role"], "user");
        assert!(messages[1]["content"].as_str().unwrap().contains("resumed"));
    }

    #[test]
    fn test_valid_tool_call_with_result_preserved() {
        let mut messages = vec![
            json!({"role": "system", "content": "System"}),
            json!({"role": "user", "content": "Read a file"}),
            json!({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "tc_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"}
                }]
            }),
            json!({
                "role": "tool",
                "tool_call_id": "tc_1",
                "name": "read_file",
                "content": "File contents here"
            }),
            json!({"role": "assistant", "content": "Here's the file contents."}),
        ];

        repair_messages(&mut messages);
        // Original 5 + user(anti-prefill) = 6
        assert_eq!(messages.len(), 6);
        assert_eq!(
            messages.last().unwrap()["role"].as_str(),
            Some("user"),
            "Last message should be user continuation"
        );
    }

    #[test]
    fn test_multiple_orphaned_tool_calls() {
        let mut messages = vec![
            json!({"role": "system", "content": "System"}),
            json!({"role": "user", "content": "Do two things"}),
            json!({
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "tc_1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}},
                    {"id": "tc_2", "type": "function", "function": {"name": "write_file", "arguments": "{}"}}
                ]
            }),
            // Only tc_1 has a result.
            json!({
                "role": "tool",
                "tool_call_id": "tc_1",
                "name": "read_file",
                "content": "File content"
            }),
        ];

        repair_messages(&mut messages);

        // tc_2 should get a synthetic result.
        let tool_msgs: Vec<&Value> = messages
            .iter()
            .filter(|m| m.get("role").and_then(|r| r.as_str()) == Some("tool"))
            .collect();
        assert_eq!(tool_msgs.len(), 2);
        assert!(tool_msgs.iter().any(|m| m["tool_call_id"] == "tc_2"));
    }

    #[test]
    fn test_tool_result_before_its_assistant_is_removed() {
        // A tool result that appears BEFORE its matching assistant should be
        // treated as orphaned (positional violation).
        let mut messages = vec![
            json!({"role": "system", "content": "System"}),
            json!({"role": "user", "content": "Hello"}),
            // Tool result appears BEFORE its matching assistant.
            json!({
                "role": "tool",
                "tool_call_id": "tc_1",
                "name": "read_file",
                "content": "data"
            }),
            json!({
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "tc_1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]
            }),
            json!({"role": "assistant", "content": "Done"}),
        ];

        repair_messages(&mut messages);

        // The mispositioned tool result should be removed.
        // fix_orphaned_tool_calls will then add a synthetic result after the assistant.
        let tool_msgs: Vec<&Value> = messages
            .iter()
            .filter(|m| m.get("role").and_then(|r| r.as_str()) == Some("tool"))
            .collect();
        // Should have exactly 1 tool result (the synthetic one, AFTER the assistant).
        assert_eq!(tool_msgs.len(), 1);
        assert!(
            tool_msgs[0]["content"].as_str().unwrap().contains("interrupted"),
            "Should be a synthetic result"
        );
    }

    #[test]
    fn test_tool_result_matching_later_assistant_is_orphan() {
        // Scenario from history windowing: tool result at the start of window,
        // matching assistant exists later but the result precedes it.
        let mut messages = vec![
            json!({"role": "system", "content": "System"}),
            json!({"role": "user", "content": "q1"}),
            json!({
                "role": "tool",
                "tool_call_id": "tc_old",
                "name": "exec",
                "content": "old result"
            }),
            json!({"role": "user", "content": "q2"}),
            json!({
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "tc_new", "type": "function", "function": {"name": "exec", "arguments": "{}"}}]
            }),
            json!({
                "role": "tool",
                "tool_call_id": "tc_new",
                "name": "exec",
                "content": "new result"
            }),
            json!({"role": "assistant", "content": "Done"}),
        ];

        repair_messages(&mut messages);

        // tc_old should be removed (no preceding assistant with that ID).
        // tc_new should survive (its assistant precedes it).
        let tool_msgs: Vec<&Value> = messages
            .iter()
            .filter(|m| m.get("role").and_then(|r| r.as_str()) == Some("tool"))
            .collect();
        assert_eq!(tool_msgs.len(), 1);
        assert_eq!(tool_msgs[0]["tool_call_id"], "tc_new");
    }

    #[test]
    fn test_dedup_tool_results() {
        // Two tool results with the same tool_call_id — the second should be removed.
        let mut messages = vec![
            json!({"role": "system", "content": "System"}),
            json!({"role": "user", "content": "Do something"}),
            json!({
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "tc_dup", "type": "function", "function": {"name": "exec", "arguments": "{}"}}]
            }),
            json!({
                "role": "tool",
                "tool_call_id": "tc_dup",
                "name": "exec",
                "content": "first result"
            }),
            json!({
                "role": "tool",
                "tool_call_id": "tc_dup",
                "name": "exec",
                "content": "duplicate result written 14s later"
            }),
            json!({"role": "assistant", "content": "Done"}),
        ];

        repair_messages(&mut messages);

        let tool_msgs: Vec<&Value> = messages
            .iter()
            .filter(|m| m.get("role").and_then(|r| r.as_str()) == Some("tool"))
            .collect();
        assert_eq!(tool_msgs.len(), 1, "Duplicate tool result should be removed");
        assert_eq!(
            tool_msgs[0]["content"].as_str().unwrap(),
            "first result",
            "First result should be kept"
        );
    }

    #[test]
    fn test_ensure_not_ending_with_assistant() {
        // Messages ending with assistant should get a user continuation.
        let mut messages = vec![
            json!({"role": "system", "content": "System"}),
            json!({"role": "user", "content": "Hello"}),
            json!({
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "tc_1", "type": "function", "function": {"name": "exec", "arguments": "{}"}}]
            }),
            json!({
                "role": "tool",
                "tool_call_id": "tc_1",
                "name": "exec",
                "content": "result"
            }),
            json!({"role": "assistant", "content": "Runner summary"}),
        ];

        repair_messages(&mut messages);

        let last = messages.last().unwrap();
        assert_eq!(
            last["role"].as_str(),
            Some("user"),
            "Last message should be user, not assistant. Got: {}",
            last
        );
    }

    #[test]
    fn test_not_ending_with_assistant_no_op_when_user() {
        // Messages ending with user should not be modified.
        let mut messages = vec![
            json!({"role": "system", "content": "System"}),
            json!({"role": "user", "content": "Hello"}),
        ];

        let len_before = messages.len();
        repair_messages(&mut messages);
        assert_eq!(messages.len(), len_before);
    }
}
