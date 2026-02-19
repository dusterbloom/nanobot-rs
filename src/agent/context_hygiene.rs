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
//! 3. `truncate_old_tool_results` - Keep only last N tool results
//! 4. `truncate_old_assistant_messages` - Compress old assistant content
//! 5. `strip_dangling_tool_calls` - Remove tool calls without immediate results
//! 6. `remove_orphaned_tool_results` - Remove results without matching calls

use std::collections::{HashMap, HashSet};

use serde_json::{json, Value};
use tracing::debug;

const TRUNCATED_ASSISTANT_PLACEHOLDER: &str = "[assistant message truncated]";
pub const DEFAULT_KEEP_LAST_MESSAGES: usize = 20;

pub fn hygiene_pipeline(messages: &mut Vec<Value>) {
    if messages.is_empty() {
        return;
    }

    let before = messages.len();

    dedup_tool_results(messages);
    merge_consecutive_same_role(messages);
    truncate_old_tool_results(messages, DEFAULT_KEEP_LAST_MESSAGES);
    truncate_old_assistant_messages(messages, DEFAULT_KEEP_LAST_MESSAGES);
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

pub fn dedup_tool_results(messages: &mut Vec<Value>) {
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

pub fn merge_consecutive_same_role(messages: &mut Vec<Value>) {
    if messages.len() <= 1 {
        return;
    }

    let mut merged: Vec<Value> = Vec::with_capacity(messages.len());

    for message in messages.drain(..) {
        if let Some(prev) = merged.last_mut() {
            if get_role(prev) == get_role(&message) && get_role(&message) != "tool" {
                let prev_content = get_content(prev);
                let msg_content = get_content(&message);
                let combined = format!("{}\n{}", prev_content, msg_content);
                *prev = json!({
                    "role": get_role(&message),
                    "content": combined
                });
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

pub fn truncate_old_tool_results(messages: &mut Vec<Value>, keep_last_n: usize) {
    if keep_last_n == usize::MAX || messages.len() <= keep_last_n {
        return;
    }

    let tool_result_positions: Vec<(usize, String)> = messages
        .iter()
        .enumerate()
        .filter_map(|(idx, m)| get_tool_result_id(m).map(|id| (idx, id)))
        .collect();

    if tool_result_positions.len() <= keep_last_n {
        return;
    }

    let keep_from = tool_result_positions.len().saturating_sub(keep_last_n);
    let keep_ids: HashSet<String> = tool_result_positions
        .into_iter()
        .skip(keep_from)
        .map(|(_, id)| id)
        .collect();

    messages.retain(|m| {
        if let Some(id) = get_tool_result_id(m) {
            keep_ids.contains(&id)
        } else {
            true
        }
    });
}

pub fn truncate_old_assistant_messages(messages: &mut Vec<Value>, keep_last_n: usize) {
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
                    message["content"] = Value::String(TRUNCATED_ASSISTANT_PLACEHOLDER.to_string());
                }
            }
        } else {
            let content = message
                .get("content")
                .and_then(|c| c.as_str())
                .unwrap_or("");
            if !content.is_empty() {
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

        let has_matching_results =
            !all_results.is_empty() && tool_call_ids.iter().all(|id| all_results.contains(id));

        if has_matching_results {
            continue;
        }

        if let Some(tc) = messages[idx].get("tool_calls").and_then(|t| t.as_array()) {
            if !tc.is_empty() {
                messages[idx]["tool_calls"] = Value::Array(vec![]);
            }
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
        hygiene_pipeline(&mut messages);

        assert!(messages.len() <= 4);
    }

    #[test]
    fn test_empty_messages_no_panic() {
        let mut messages: Vec<Value> = vec![];
        hygiene_pipeline(&mut messages);
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
        hygiene_pipeline(&mut messages);
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
    fn test_truncate_old_tool_results_keeps_last_n() {
        let mut messages = vec![
            assistant_with_tool_call("tc_1"),
            tool_result("tc_1", "old"),
            assistant_with_tool_call("tc_2"),
            tool_result("tc_2", "newer"),
            assistant_with_tool_call("tc_3"),
            tool_result("tc_3", "newest"),
        ];
        truncate_old_tool_results(&mut messages, 2);

        let remaining_results: Vec<_> = messages.iter().filter(|m| get_role(m) == "tool").collect();
        assert!(remaining_results.len() <= 2);
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
    fn test_partial_tool_results_marks_dangling() {
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

        let tool_calls = messages[0].get("tool_calls").and_then(|tc| tc.as_array());
        assert!(tool_calls.is_none() || tool_calls.unwrap().is_empty());
    }
}
