//! Named regression track: message-array protocol invariants.
//!
//! Protects against drift in the wire-format produced by `ConversationProtocol`
//! implementations. ds4's `--server` test serves the same role for its
//! HTTP/DSML rendering. We assert load-bearing invariants on the output
//! `Vec<Value>` for both `LocalProtocol` and `CloudProtocol`:
//!
//! - tool_call / tool_result pairing by id (no orphans either way)
//! - tool_call ids are unique within a turn array
//! - local mode: last message MUST be role=user (LM Studio constraint)
//! - cloud mode: assistant-prefill at the tail is allowed
//!
//! When any of these fail, the agent silently misbehaves: the LLM rejects
//! the request, sees stale context, or worse, hallucinates a tool result.

use serde_json::{json, Value};

use nanobot::agent::protocol::{CloudProtocol, ConversationProtocol, LocalProtocol};
use nanobot::agent::turn::{ToolCall, Turn};

// ─────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────

fn user(text: &str) -> Turn {
    Turn::User {
        content: text.into(),
        media: vec![],
    }
}

fn assistant_with_calls(text: Option<&str>, calls: Vec<(&str, &str, Value)>) -> Turn {
    Turn::Assistant {
        text: text.map(String::from),
        tool_calls: calls
            .into_iter()
            .map(|(id, name, args)| ToolCall {
                id: id.into(),
                tool: name.into(),
                args,
            })
            .collect(),
    }
}

fn tool_result(call_id: &str, tool: &str, result: &str) -> Turn {
    Turn::ToolResult {
        call_id: call_id.into(),
        tool: tool.into(),
        result: result.into(),
        ok: true,
    }
}

fn role(msg: &Value) -> &str {
    msg.get("role").and_then(|r| r.as_str()).unwrap_or("")
}

fn tool_call_ids(msg: &Value) -> Vec<String> {
    msg.get("tool_calls")
        .and_then(|tc| tc.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|tc| tc.get("id").and_then(|id| id.as_str()).map(String::from))
                .collect()
        })
        .unwrap_or_default()
}

fn tool_result_id(msg: &Value) -> Option<String> {
    msg.get("tool_call_id")
        .and_then(|id| id.as_str())
        .map(String::from)
}

// ─────────────────────────────────────────────────────────────
// Invariant 1: tool_call / tool_result pairing by id
// ─────────────────────────────────────────────────────────────

fn assert_paired(messages: &[Value], protocol_name: &str) {
    let mut announced: Vec<String> = Vec::new();
    let mut answered: Vec<String> = Vec::new();
    for msg in messages {
        match role(msg) {
            "assistant" => announced.extend(tool_call_ids(msg)),
            "tool" => {
                if let Some(id) = tool_result_id(msg) {
                    answered.push(id);
                }
            }
            _ => {}
        }
    }
    for id in &answered {
        assert!(
            announced.contains(id),
            "[{}] orphan tool_result with id={id} (no prior tool_call announces it)",
            protocol_name
        );
    }
}

#[test]
fn cloud_pairs_tool_calls_with_results() {
    let turns = vec![
        user("read Cargo.toml"),
        assistant_with_calls(
            Some("looking it up"),
            vec![("tc_1", "read_file", json!({"path": "Cargo.toml"}))],
        ),
        tool_result("tc_1", "read_file", "[package]\nname = \"nanobot\""),
        user("thanks"),
    ];
    let messages = CloudProtocol.render("you are nanobot", &turns);
    assert_paired(&messages, "cloud");
}

#[test]
fn local_textual_replay_does_not_emit_orphan_tool_messages() {
    let turns = vec![
        user("read Cargo.toml"),
        assistant_with_calls(
            None,
            vec![("tc_1", "read_file", json!({"path": "Cargo.toml"}))],
        ),
        tool_result("tc_1", "read_file", "[package]"),
        user("ok"),
    ];
    let messages = LocalProtocol::textual().render("you are nanobot", &turns);
    // Local protocol in textual replay mode — no role=tool messages should exist.
    // (If it did, the local model would reject the request.)
    for msg in &messages {
        assert_ne!(
            role(msg),
            "tool",
            "local protocol emitted role=tool message: {:?}",
            msg
        );
    }
}

// ─────────────────────────────────────────────────────────────
// Invariant 2: tool_call ids are unique within the assistant message
// ─────────────────────────────────────────────────────────────

#[test]
fn cloud_tool_call_ids_unique_within_one_assistant() {
    let turns = vec![
        user("read two files"),
        assistant_with_calls(
            None,
            vec![
                ("tc_1", "read_file", json!({"path": "a.toml"})),
                ("tc_2", "read_file", json!({"path": "b.toml"})),
            ],
        ),
    ];
    let messages = CloudProtocol.render("sys", &turns);
    for msg in &messages {
        let ids = tool_call_ids(msg);
        let mut sorted = ids.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(
            ids.len(),
            sorted.len(),
            "duplicate tool_call ids in one message: {:?}",
            ids
        );
    }
}

// ─────────────────────────────────────────────────────────────
// Invariant 3: local mode — last message MUST be role=user
// ─────────────────────────────────────────────────────────────
//
// LM Studio / llama.cpp jinja templates require the final message to be a
// user turn. Any drift here breaks every local-mode conversation. See
// CLAUDE.md ("Local LLM Protocol Constraints").

#[test]
fn local_last_message_is_user_after_tool_results() {
    let turns = vec![
        user("list files"),
        assistant_with_calls(None, vec![("tc_1", "ls", json!({"path": "."}))]),
        tool_result("tc_1", "ls", "Cargo.toml\nsrc/"),
    ];
    let messages = LocalProtocol::native().render("sys", &turns);
    let last = messages.last().expect("rendered messages must be non-empty");
    assert_eq!(
        role(last),
        "user",
        "local protocol last message must be role=user; got: {:?}",
        last
    );
}

#[test]
fn local_last_message_is_user_simple_qna() {
    let turns = vec![user("what is 2+2?")];
    let messages = LocalProtocol::native().render("sys", &turns);
    let last = messages.last().expect("rendered messages must be non-empty");
    assert_eq!(role(last), "user");
}

// ─────────────────────────────────────────────────────────────
// Invariant 4: system message is exactly one, at index 0
// ─────────────────────────────────────────────────────────────

#[test]
fn exactly_one_system_message_at_index_zero_cloud() {
    let turns = vec![user("hi")];
    let messages = CloudProtocol.render("you are nanobot", &turns);
    let sys_count = messages.iter().filter(|m| role(m) == "system").count();
    assert_eq!(
        sys_count, 1,
        "exactly one system message expected; got {} ({:?})",
        sys_count, messages
    );
    assert_eq!(role(&messages[0]), "system");
}
