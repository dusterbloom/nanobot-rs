//! Conversation protocol — renders canonical `Turn` history to LLM wire format.
//!
//! Two implementations:
//! - [`CloudProtocol`] — standard OpenAI function-calling format (tool_calls / role:tool).
//! - [`LocalProtocol`] — strict user/assistant alternation for local models (LM Studio / vLLM).
//!   Tool results become user messages; assistant tool_calls become text summaries.
//!   No `role:tool`, no `role:system` after index 0, always ends with user.
//!
//! Protocol selection happens once per turn in `agent_loop.rs`:
//! ```ignore
//! let protocol: Arc<dyn ConversationProtocol> = if ctx.core.is_local {
//!     Arc::new(LocalProtocol)
//! } else {
//!     Arc::new(CloudProtocol)
//! };
//! ```

use serde_json::{json, Value};

use super::turn::{ToolCall, Turn};

// ─────────────────────────────────────────────────────────────
// Trait
// ─────────────────────────────────────────────────────────────

/// Renders a canonical `Turn` sequence to LLM wire format.
///
/// Implementations must enforce their invariants **structurally** — no post-hoc repair.
pub trait ConversationProtocol: Send + Sync {
    /// Render `turns` into a flat `Vec<Value>` ready to send to an LLM.
    ///
    /// `system` is injected as the first message (role:system for cloud,
    /// same for local — local only restricts *mid-thread* system messages).
    fn render(&self, system: &str, turns: &[Turn]) -> Vec<Value>;

    /// Human-readable name for logging / tracing.
    fn name(&self) -> &'static str;
}

// ─────────────────────────────────────────────────────────────
// CloudProtocol
// ─────────────────────────────────────────────────────────────

/// Standard OpenAI function-calling wire format.
///
/// Invariants enforced:
/// - First message is `role:system`.
/// - `Turn::Assistant { tool_calls }` emits the `tool_calls` JSON array.
/// - `Turn::ToolResult` emits `role:tool` with `tool_call_id`.
/// - If last rendered message is `role:assistant`, a user continuation is appended
///   (Anthropic OpenAI-compat endpoint rejects assistant prefill).
pub struct CloudProtocol;

impl ConversationProtocol for CloudProtocol {
    fn render(&self, system: &str, turns: &[Turn]) -> Vec<Value> {
        let mut out: Vec<Value> = Vec::with_capacity(turns.len() + 2);

        // System message always first.
        out.push(json!({"role": "system", "content": system}));

        for turn in turns {
            match turn {
                Turn::System { content } => {
                    // Mid-thread system — already have a leading system; skip duplicate.
                    // If needed, it could be added as a user note, but cloud APIs
                    // handle multiple system messages poorly. For now, skip.
                    let _ = content;
                }
                Turn::User { content, .. } => {
                    out.push(json!({"role": "user", "content": content}));
                }
                Turn::Assistant { text, tool_calls } => {
                    let content_val = match text {
                        Some(t) => Value::String(t.clone()),
                        None => Value::Null,
                    };
                    if tool_calls.is_empty() {
                        out.push(json!({"role": "assistant", "content": content_val}));
                    } else {
                        let tc_json: Vec<Value> = tool_calls
                            .iter()
                            .map(tool_call_to_openai_json)
                            .collect();
                        out.push(json!({
                            "role": "assistant",
                            "content": content_val,
                            "tool_calls": tc_json,
                        }));
                    }
                }
                Turn::ToolResult { call_id, tool, result, .. } => {
                    out.push(json!({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": tool,
                        "content": result,
                    }));
                }
                Turn::Summary { text, .. } => {
                    // Summaries render as assistant messages summarising prior context.
                    out.push(json!({"role": "assistant", "content": text}));
                }
            }
        }

        // Anthropic compat: must not end with assistant.
        if out.last().map(|m| m["role"] == "assistant").unwrap_or(false) {
            out.push(json!({"role": "user", "content": "Continue."}));
        }

        out
    }

    fn name(&self) -> &'static str {
        "cloud"
    }
}

// ─────────────────────────────────────────────────────────────
// LocalProtocol
// ─────────────────────────────────────────────────────────────

/// Strict user/assistant alternation for local models (LM Studio, Ollama, vLLM).
///
/// Invariants enforced:
/// - No `role:tool` — tool results become user messages.
/// - No `role:system` after index 0 — mid-thread system turns become user messages.
/// - Assistant `tool_calls` are converted to a text summary with arguments so the model
///   can verify results match: `"[I called: name(args)]"`.
/// - Consecutive same-role messages are merged.
/// - Output always ends with `role:user`.
pub struct LocalProtocol;

impl ConversationProtocol for LocalProtocol {
    fn render(&self, system: &str, turns: &[Turn]) -> Vec<Value> {
        let mut out: Vec<Value> = Vec::with_capacity(turns.len() + 2);

        // Leading system message (index 0 is the only allowed system position).
        out.push(json!({"role": "system", "content": system}));

        for turn in turns {
            let msg = match turn {
                Turn::System { content } => {
                    // Mid-thread system → user notice.
                    json!({"role": "user", "content": format!("[System notice] {}", content)})
                }
                Turn::User { content, .. } => {
                    json!({"role": "user", "content": content})
                }
                Turn::Assistant { text, tool_calls } => {
                    // Convert tool_calls to a text summary with arguments so the
                    // model can verify that the subsequent tool results match what
                    // was called.  Without arguments, small models re-call tools
                    // because they cannot confirm the result is authoritative.
                    let tool_summary = if tool_calls.is_empty() {
                        String::new()
                    } else {
                        let calls: Vec<String> = tool_calls
                            .iter()
                            .map(|tc| {
                                let args_str = serde_json::to_string(&tc.args)
                                    .unwrap_or_else(|_| "{}".to_string());
                                format!("{}({})", tc.tool, args_str)
                            })
                            .collect();
                        format!("[I called: {}]", calls.join(", "))
                    };
                    let content = match text.as_deref() {
                        Some(t) if !t.is_empty() && !tool_summary.is_empty() => {
                            format!("{}\n{}", t, tool_summary)
                        }
                        Some(t) if !t.is_empty() => t.to_string(),
                        _ if !tool_summary.is_empty() => tool_summary,
                        _ => String::new(),
                    };
                    json!({"role": "assistant", "content": content})
                }
                Turn::ToolResult { tool, call_id, result, .. } => {
                    json!({
                        "role": "user",
                        "content": format!("[System: tool execution complete — {}({}) returned]:\n{}", tool, call_id, result),
                    })
                }
                Turn::Summary { text, .. } => {
                    // Summaries render as user context blocks (local models may not follow
                    // assistant-role summaries reliably).
                    json!({
                        "role": "user",
                        "content": format!("[Context summary]: {}", text),
                    })
                }
            };
            out.push(msg);
        }

        // Merge consecutive same-role messages (avoids "consecutive user" violations).
        out = merge_consecutive_role(out);

        // Must always end with user.
        if out.last().map(|m| m["role"] != "user").unwrap_or(true) {
            out.push(json!({"role": "user", "content": "Continue."}));
        }

        out
    }

    fn name(&self) -> &'static str {
        "local"
    }
}

// ─────────────────────────────────────────────────────────────
// Public helpers
// ─────────────────────────────────────────────────────────────

/// Convert a raw wire-format message array to a protocol-rendered wire format.
///
/// Extracts the leading `role:system` message as the system prompt, converts
/// the remaining messages to canonical `Turn`s via `turn_from_legacy`, then
/// renders them using `protocol.render()`.
///
/// Metadata-only fields (e.g. `_turn`, `_synthetic`) on raw messages are not
/// forwarded to the rendered output — they are internal to the message store.
pub fn render_to_wire(protocol: &dyn ConversationProtocol, messages: &[Value]) -> Vec<Value> {
    use super::turn::turn_from_legacy;

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

// ─────────────────────────────────────────────────────────────
// Private helpers
// ─────────────────────────────────────────────────────────────

/// Convert a canonical `ToolCall` to the OpenAI wire representation.
fn tool_call_to_openai_json(tc: &ToolCall) -> Value {
    json!({
        "id": tc.id,
        "type": "function",
        "function": {
            "name": tc.tool,
            "arguments": serde_json::to_string(&tc.args).unwrap_or_else(|_| "{}".into()),
        }
    })
}

/// Merge consecutive messages with the same role by concatenating their content.
///
/// This operates on an already-built wire-format `Vec<Value>`.
/// The leading system message (index 0) is preserved as-is.
fn merge_consecutive_role(messages: Vec<Value>) -> Vec<Value> {
    if messages.is_empty() {
        return messages;
    }
    let mut out: Vec<Value> = Vec::with_capacity(messages.len());
    for msg in messages {
        let role = msg["role"].as_str().unwrap_or("").to_string();
        let content = msg["content"].as_str().unwrap_or("").to_string();

        // The system message at index 0 is never merged.
        if let Some(last) = out.last_mut() {
            let last_role = last["role"].as_str().unwrap_or("");
            // Don't merge system messages.
            if last_role == role && role != "system" {
                let last_content = last["content"].as_str().unwrap_or("").to_string();
                let merged = if last_content.is_empty() {
                    content
                } else if content.is_empty() {
                    last_content
                } else {
                    format!("{}\n\n{}", last_content, content)
                };
                last["content"] = Value::String(merged);
                continue;
            }
        }
        out.push(msg);
    }
    out
}

// ─────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tool_turns() -> Vec<Turn> {
        vec![
            Turn::User { content: "read file".into(), media: vec![] },
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
                result: "data".into(),
                ok: true,
            },
        ]
    }

    // ---- LocalProtocol unit tests ----

    #[test]
    fn local_no_tool_role() {
        let wire = LocalProtocol.render("sys", &tool_turns());
        assert!(wire.iter().all(|m| m["role"] != "tool"));
    }

    #[test]
    fn local_ends_with_user() {
        let turns = vec![
            Turn::User { content: "hi".into(), media: vec![] },
            Turn::Assistant { text: Some("hello".into()), tool_calls: vec![] },
        ];
        let wire = LocalProtocol.render("sys", &turns);
        assert_eq!(wire.last().unwrap()["role"], "user");
    }

    #[test]
    fn local_assistant_has_no_tool_calls_field() {
        let wire = LocalProtocol.render("sys", &tool_turns());
        for msg in &wire {
            if msg["role"] == "assistant" {
                assert!(msg.get("tool_calls").is_none());
            }
        }
    }

    #[test]
    fn local_mid_thread_system_becomes_user() {
        let turns = vec![
            Turn::User { content: "hi".into(), media: vec![] },
            Turn::System { content: "Injected notice".into() },
            Turn::User { content: "go on".into(), media: vec![] },
        ];
        let wire = LocalProtocol.render("sys", &turns);
        let non_first_system = wire.iter().skip(1).any(|m| m["role"] == "system");
        assert!(!non_first_system);
        let has_notice = wire
            .iter()
            .any(|m| m["content"].as_str().unwrap_or("").contains("Injected notice"));
        assert!(has_notice);
    }

    // ---- CloudProtocol unit tests ----

    #[test]
    fn cloud_has_tool_role() {
        let wire = CloudProtocol.render("sys", &tool_turns());
        let tool_msg = wire.iter().find(|m| m["role"] == "tool").unwrap();
        assert_eq!(tool_msg["tool_call_id"], "tc_1");
    }

    #[test]
    fn cloud_assistant_has_tool_calls_field() {
        let wire = CloudProtocol.render("sys", &tool_turns());
        let asst = wire.iter().find(|m| m["role"] == "assistant").unwrap();
        assert!(asst.get("tool_calls").is_some());
    }

    #[test]
    fn cloud_does_not_end_with_assistant() {
        let turns = vec![
            Turn::User { content: "hi".into(), media: vec![] },
            Turn::Assistant { text: Some("hello".into()), tool_calls: vec![] },
        ];
        let wire = CloudProtocol.render("sys", &turns);
        assert_ne!(wire.last().unwrap()["role"], "assistant");
    }

    #[test]
    fn merge_consecutive_role_helper() {
        let msgs = vec![
            json!({"role": "system", "content": "sys"}),
            json!({"role": "user", "content": "hello"}),
            json!({"role": "user", "content": "world"}),
            json!({"role": "assistant", "content": "hi"}),
        ];
        let merged = merge_consecutive_role(msgs);
        assert_eq!(merged.len(), 3); // system + merged_user + assistant
        assert!(merged[1]["content"].as_str().unwrap().contains("hello"));
        assert!(merged[1]["content"].as_str().unwrap().contains("world"));
    }
}
