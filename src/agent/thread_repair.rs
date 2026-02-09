//! Message thread repair for OpenAI protocol compliance.
//!
//! Scans a message array for protocol violations (orphaned tool calls,
//! orphaned tool results, consecutive user messages, etc.) and repairs
//! them before sending to the LLM.

use std::collections::HashSet;

use serde_json::{json, Value};

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

    // Pass 1: Fix orphaned tool_use (assistant with tool_calls but missing results).
    fix_orphaned_tool_calls(messages);

    // Pass 2: Fix orphaned tool results (tool messages without matching assistant).
    fix_orphaned_tool_results(messages);

    // Pass 3: Merge consecutive user messages.
    merge_consecutive_user_messages(messages);

    // Pass 4: Ensure first non-system message is user role.
    ensure_user_first(messages);
}

/// Find assistant messages with tool_calls that don't have corresponding
/// tool result messages. Append synthetic error results for missing ones.
fn fix_orphaned_tool_calls(messages: &mut Vec<Value>) {
    // Collect all tool_call_ids that have results.
    let result_ids: HashSet<String> = messages
        .iter()
        .filter(|m| m.get("role").and_then(|r| r.as_str()) == Some("tool"))
        .filter_map(|m| m.get("tool_call_id").and_then(|id| id.as_str()).map(String::from))
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
            let call_id = tc
                .get("id")
                .and_then(|id| id.as_str())
                .unwrap_or_default();
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
/// in any preceding assistant message.
fn fix_orphaned_tool_results(messages: &mut Vec<Value>) {
    // Collect all tool_call_ids from assistant messages.
    let mut known_call_ids: HashSet<String> = HashSet::new();
    for msg in messages.iter() {
        if msg.get("role").and_then(|r| r.as_str()) != Some("assistant") {
            continue;
        }
        if let Some(tool_calls) = msg.get("tool_calls").and_then(|tc| tc.as_array()) {
            for tc in tool_calls {
                if let Some(id) = tc.get("id").and_then(|id| id.as_str()) {
                    known_call_ids.insert(id.to_string());
                }
            }
        }
    }

    // Remove tool messages whose tool_call_id isn't in any assistant's tool_calls.
    messages.retain(|msg| {
        if msg.get("role").and_then(|r| r.as_str()) != Some("tool") {
            return true;
        }
        let call_id = msg
            .get("tool_call_id")
            .and_then(|id| id.as_str())
            .unwrap_or_default();
        if call_id.is_empty() {
            return false; // malformed tool result
        }
        known_call_ids.contains(call_id)
    });
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
        let mut messages = vec![
            json!({"role": "system", "content": "System prompt"}),
            json!({"role": "user", "content": "Hello"}),
            json!({"role": "assistant", "content": "Hi there!"}),
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
        assert_eq!(messages.len(), 3);
        assert!(!messages.iter().any(|m| m.get("role").and_then(|r| r.as_str()) == Some("tool")));
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
        assert_eq!(messages.len(), 3);
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
        assert!(messages[1]["content"]
            .as_str()
            .unwrap()
            .contains("resumed"));
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

        let original_len = messages.len();
        repair_messages(&mut messages);
        assert_eq!(messages.len(), original_len);
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
        assert!(tool_msgs
            .iter()
            .any(|m| m["tool_call_id"] == "tc_2"));
    }
}
