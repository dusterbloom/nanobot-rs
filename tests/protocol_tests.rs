// Phase 1 & 2: Tests for Turn enum and ConversationProtocol — written FIRST (Red).
// These will fail until Turn and the protocol modules are implemented.

use serde_json::json;

use nanobot::agent::turn::{turn_from_legacy, MediaAttachment, ToolCall, Turn};
use nanobot::agent::protocol::{CloudProtocol, ConversationProtocol, LocalProtocol};

// ─────────────────────────────────────────────────────────────
// Phase 1: Turn enum serialization and legacy conversion
// ─────────────────────────────────────────────────────────────

#[test]
fn turn_user_round_trips_through_jsonl() {
    let t = Turn::User {
        content: "hello".into(),
        media: vec![],
    };
    let s = serde_json::to_string(&t).unwrap();
    let t2: Turn = serde_json::from_str(&s).unwrap();
    assert!(matches!(t2, Turn::User { .. }));
    if let Turn::User { content, .. } = t2 {
        assert_eq!(content, "hello");
    }
}

#[test]
fn turn_assistant_with_tool_calls_round_trips() {
    let t = Turn::Assistant {
        text: Some("thinking".into()),
        tool_calls: vec![ToolCall {
            id: "tc_1".into(),
            tool: "read_file".into(),
            args: json!({"path": "Cargo.toml"}),
        }],
    };
    let s = serde_json::to_string(&t).unwrap();
    let t2: Turn = serde_json::from_str(&s).unwrap();
    if let Turn::Assistant { tool_calls, .. } = t2 {
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, "tc_1");
        assert_eq!(tool_calls[0].tool, "read_file");
    } else {
        panic!("Expected Turn::Assistant");
    }
}

#[test]
fn turn_tool_result_round_trips() {
    let t = Turn::ToolResult {
        call_id: "tc_1".into(),
        tool: "read_file".into(),
        result: "[package]\nname = \"nanobot\"".into(),
        ok: true,
    };
    let s = serde_json::to_string(&t).unwrap();
    let t2: Turn = serde_json::from_str(&s).unwrap();
    if let Turn::ToolResult { call_id, ok, .. } = t2 {
        assert_eq!(call_id, "tc_1");
        assert!(ok);
    } else {
        panic!("Expected Turn::ToolResult");
    }
}

#[test]
fn turn_summary_round_trips() {
    let t = Turn::Summary {
        text: "Messages 1-20 covered setup and file reads.".into(),
        source_ids: vec![0, 1, 2, 3],
        level: 1,
    };
    let s = serde_json::to_string(&t).unwrap();
    let t2: Turn = serde_json::from_str(&s).unwrap();
    if let Turn::Summary { text, source_ids, level } = t2 {
        assert_eq!(level, 1);
        assert_eq!(source_ids.len(), 4);
        assert!(text.contains("Messages 1-20"));
    } else {
        panic!("Expected Turn::Summary");
    }
}

#[test]
fn turn_kind_tag_is_snake_case() {
    let user = Turn::User { content: "hi".into(), media: vec![] };
    let v = serde_json::to_value(&user).unwrap();
    assert_eq!(v["kind"], "user");

    let result = Turn::ToolResult {
        call_id: "x".into(),
        tool: "exec".into(),
        result: "ok".into(),
        ok: true,
    };
    let v = serde_json::to_value(&result).unwrap();
    assert_eq!(v["kind"], "tool_result");

    let summary = Turn::Summary { text: "s".into(), source_ids: vec![], level: 1 };
    let v = serde_json::to_value(&summary).unwrap();
    assert_eq!(v["kind"], "summary");
}

// ─────────────────────────────────────────────────────────────
// Legacy conversion: OpenAI-format Value → Turn
// ─────────────────────────────────────────────────────────────

#[test]
fn legacy_user_message_converts_to_turn_user() {
    let legacy = json!({"role": "user", "content": "hello"});
    let t = turn_from_legacy(&legacy).unwrap();
    assert!(matches!(t, Turn::User { .. }));
    if let Turn::User { content, .. } = t {
        assert_eq!(content, "hello");
    }
}

#[test]
fn legacy_assistant_text_converts_to_turn_assistant() {
    let legacy = json!({"role": "assistant", "content": "I'll help you."});
    let t = turn_from_legacy(&legacy).unwrap();
    if let Turn::Assistant { text, tool_calls } = t {
        assert_eq!(text.as_deref(), Some("I'll help you."));
        assert!(tool_calls.is_empty());
    } else {
        panic!("Expected Turn::Assistant");
    }
}

#[test]
fn legacy_assistant_with_tool_calls_converts_correctly() {
    let legacy = json!({
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": "tc_1",
            "type": "function",
            "function": {
                "name": "read_file",
                "arguments": "{\"path\": \"Cargo.toml\"}"
            }
        }]
    });
    let t = turn_from_legacy(&legacy).unwrap();
    if let Turn::Assistant { tool_calls, .. } = t {
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, "tc_1");
        assert_eq!(tool_calls[0].tool, "read_file");
        assert_eq!(tool_calls[0].args["path"], "Cargo.toml");
    } else {
        panic!("Expected Turn::Assistant with tool_calls");
    }
}

#[test]
fn legacy_tool_result_converts_to_turn_tool_result() {
    let legacy = json!({
        "role": "tool",
        "tool_call_id": "tc_1",
        "name": "read_file",
        "content": "file contents here"
    });
    let t = turn_from_legacy(&legacy).unwrap();
    if let Turn::ToolResult { call_id, tool, result, ok } = t {
        assert_eq!(call_id, "tc_1");
        assert_eq!(tool, "read_file");
        assert_eq!(result, "file contents here");
        assert!(ok); // defaults to true for legacy results
    } else {
        panic!("Expected Turn::ToolResult");
    }
}

#[test]
fn legacy_system_message_converts_to_turn_system() {
    let legacy = json!({"role": "system", "content": "You are a helpful assistant."});
    let t = turn_from_legacy(&legacy).unwrap();
    assert!(matches!(t, Turn::System { .. }));
    if let Turn::System { content } = t {
        assert!(content.contains("helpful assistant"));
    }
}

#[test]
fn legacy_unknown_role_returns_none() {
    let legacy = json!({"role": "unknown_role", "content": "what"});
    let t = turn_from_legacy(&legacy);
    assert!(t.is_none());
}

// ─────────────────────────────────────────────────────────────
// Phase 2 tests — will be uncommented once protocol.rs exists
// ─────────────────────────────────────────────────────────────

// Helper shared by multiple tests
fn make_tool_calling_turns() -> Vec<Turn> {
    vec![
        Turn::User { content: "read Cargo.toml".into(), media: vec![] },
        Turn::Assistant {
            text: None,
            tool_calls: vec![ToolCall {
                id: "tc_1".into(),
                tool: "read_file".into(),
                args: json!({"path": "Cargo.toml"}),
            }],
        },
        Turn::ToolResult {
            call_id: "tc_1".into(),
            tool: "read_file".into(),
            result: "[package]\nname = \"nanobot\"".into(),
            ok: true,
        },
    ]
}

fn make_multi_tool_turns() -> Vec<Turn> {
    vec![
        Turn::User { content: "do two things".into(), media: vec![] },
        Turn::Assistant {
            text: None,
            tool_calls: vec![
                ToolCall { id: "tc_1".into(), tool: "read_file".into(), args: json!({"path": "a"}) },
                ToolCall { id: "tc_2".into(), tool: "exec".into(), args: json!({"cmd": "ls"}) },
            ],
        },
        Turn::ToolResult { call_id: "tc_1".into(), tool: "read_file".into(), result: "contents of a".into(), ok: true },
        Turn::ToolResult { call_id: "tc_2".into(), tool: "exec".into(), result: "file1\nfile2".into(), ok: true },
        Turn::Assistant { text: Some("Done".into()), tool_calls: vec![] },
    ]
}

// ─────────────────────────────────────────────────────────────
// Phase 2: ConversationProtocol render contracts
// ─────────────────────────────────────────────────────────────

// ---- LocalProtocol ----

#[test]
fn local_renders_system_as_first_message() {
    let protocol = LocalProtocol;
    let turns = vec![Turn::User { content: "hi".into(), media: vec![] }];
    let wire = protocol.render("You are helpful.", &turns);
    assert_eq!(wire[0]["role"], "system");
    assert!(wire[0]["content"].as_str().unwrap().contains("You are helpful."));
}

#[test]
fn local_renders_tool_result_as_user_message() {
    let protocol = LocalProtocol;
    let wire = protocol.render("sys", &make_tool_calling_turns());
    // Last non-appended message before possible continuation
    let tool_msgs: Vec<_> = wire.iter().filter(|m| m["role"] == "tool").collect();
    assert!(tool_msgs.is_empty(), "local protocol must never emit role:tool");
    // The tool result should appear as a user message
    let user_msgs: Vec<_> = wire.iter().filter(|m| m["role"] == "user").collect();
    let has_tool_result_as_user = user_msgs
        .iter()
        .any(|m| m["content"].as_str().unwrap_or("").contains("read_file"));
    assert!(has_tool_result_as_user, "tool result should appear in a user message");
}

#[test]
fn local_no_tool_role_in_any_message() {
    let protocol = LocalProtocol;
    let wire = protocol.render("sys", &make_multi_tool_turns());
    assert!(
        wire.iter().all(|m| m["role"] != "tool"),
        "local protocol must never emit role:tool"
    );
}

#[test]
fn local_always_ends_with_user() {
    let protocol = LocalProtocol;
    // Turns ending with assistant — local must append a continuation
    let turns = vec![
        Turn::User { content: "hi".into(), media: vec![] },
        Turn::Assistant { text: Some("hello".into()), tool_calls: vec![] },
    ];
    let wire = protocol.render("sys", &turns);
    assert_eq!(wire.last().unwrap()["role"], "user");
}

#[test]
fn local_tool_calling_sequence_ends_with_user() {
    let protocol = LocalProtocol;
    let wire = protocol.render("sys", &make_tool_calling_turns());
    assert_eq!(wire.last().unwrap()["role"], "user");
}

#[test]
fn local_no_mid_thread_system_messages() {
    let protocol = LocalProtocol;
    // A System turn that's not the first — must become a user message
    let turns = vec![
        Turn::User { content: "hi".into(), media: vec![] },
        Turn::Assistant { text: Some("hello".into()), tool_calls: vec![] },
        Turn::System { content: "New system notice".into() },
        Turn::User { content: "continue".into(), media: vec![] },
    ];
    let wire = protocol.render("sys", &turns);
    let non_first_system = wire.iter().skip(1).any(|m| m["role"] == "system");
    assert!(!non_first_system, "no system messages after index 0");
    // The injected system content must appear somewhere as a user message
    let has_notice = wire
        .iter()
        .any(|m| m["content"].as_str().unwrap_or("").contains("New system notice"));
    assert!(has_notice, "system notice content must appear in wire output");
}

#[test]
fn local_assistant_tool_calls_converted_to_text_summary() {
    let protocol = LocalProtocol;
    // Turn: assistant with tool_calls → must appear as assistant with text (no tool_calls field)
    let turns = vec![
        Turn::User { content: "read file".into(), media: vec![] },
        Turn::Assistant {
            text: None,
            tool_calls: vec![ToolCall {
                id: "tc_1".into(),
                tool: "read_file".into(),
                args: json!({"path": "Cargo.toml"}),
            }],
        },
        Turn::ToolResult { call_id: "tc_1".into(), tool: "read_file".into(), result: "data".into(), ok: true },
    ];
    let wire = protocol.render("sys", &turns);
    let assistant_msgs: Vec<_> = wire.iter().filter(|m| m["role"] == "assistant").collect();
    // At least one assistant message must exist
    assert!(!assistant_msgs.is_empty());
    // None of them should have a tool_calls field
    for msg in &assistant_msgs {
        assert!(msg.get("tool_calls").is_none(), "local assistant messages must not have tool_calls");
    }
    // The assistant message should mention the tool name
    let has_tool_mention = assistant_msgs
        .iter()
        .any(|m| m["content"].as_str().unwrap_or("").contains("read_file"));
    assert!(has_tool_mention, "local assistant turn should include tool name reference");
}

// ---- CloudProtocol ----

#[test]
fn cloud_renders_tool_result_with_role_tool() {
    let protocol = CloudProtocol;
    let wire = protocol.render("sys", &make_tool_calling_turns());
    let tool_msg = wire.iter().find(|m| m["role"] == "tool").expect("must have role:tool");
    assert_eq!(tool_msg["tool_call_id"], "tc_1");
    assert!(tool_msg["content"].as_str().unwrap_or("").contains("nanobot"));
}

#[test]
fn cloud_renders_assistant_with_tool_calls_field() {
    let protocol = CloudProtocol;
    let wire = protocol.render("sys", &make_tool_calling_turns());
    let asst = wire.iter().find(|m| m["role"] == "assistant").expect("must have assistant");
    assert!(asst.get("tool_calls").is_some(), "cloud assistant must include tool_calls field");
}

#[test]
fn cloud_does_not_end_with_assistant() {
    let protocol = CloudProtocol;
    // Turns ending with assistant — cloud must append a user continuation
    let turns = vec![
        Turn::User { content: "hi".into(), media: vec![] },
        Turn::Assistant { text: Some("hello".into()), tool_calls: vec![] },
    ];
    let wire = protocol.render("sys", &turns);
    assert_ne!(
        wire.last().unwrap()["role"],
        "assistant",
        "cloud protocol must not end with assistant"
    );
}

#[test]
fn cloud_system_message_is_first() {
    let protocol = CloudProtocol;
    let turns = vec![Turn::User { content: "hi".into(), media: vec![] }];
    let wire = protocol.render("You are a bot.", &turns);
    assert_eq!(wire[0]["role"], "system");
}

#[test]
fn cloud_multi_tool_has_all_tool_roles() {
    let protocol = CloudProtocol;
    let wire = protocol.render("sys", &make_multi_tool_turns());
    let tool_msgs: Vec<_> = wire.iter().filter(|m| m["role"] == "tool").collect();
    assert_eq!(tool_msgs.len(), 2, "should have 2 role:tool messages");
    let ids: Vec<&str> = tool_msgs
        .iter()
        .map(|m| m["tool_call_id"].as_str().unwrap())
        .collect();
    assert!(ids.contains(&"tc_1"));
    assert!(ids.contains(&"tc_2"));
}
