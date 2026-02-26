//! Conversation protocol — renders canonical `Turn` history to LLM wire format.
//!
//! Two implementations:
//! - [`CloudProtocol`] — standard OpenAI function-calling format (tool_calls / role:tool).
//! - [`LocalProtocol`] — strict user/assistant alternation for local models (LM Studio / vLLM).
//!   Tool results become user messages. Assistant replay can be native `tool_calls`
//!   or textual summaries, selected by replay mode.
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
use std::collections::HashMap;

use super::turn::{ToolCall, Turn};
use crate::agent::model_capabilities::{lookup, ModelSizeClass};

const CONTINUE_SENTINEL: &str = "Continue.";
const SYSTEM_NOTICE_PREFIX: &str = "[System notice]";
const CONTEXT_SUMMARY_PREFIX: &str = "[Context summary]:";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocalReplayMode {
    NativeToolCalls,
    TextualReplay,
}

impl LocalReplayMode {
    fn from_env() -> Option<Self> {
        let raw = std::env::var("NANOBOT_LOCAL_PROTOCOL_MODE").ok()?;
        let value = raw.trim().to_ascii_lowercase();
        match value.as_str() {
            "native" | "tool_calls" | "native_tool_calls" => Some(Self::NativeToolCalls),
            "text" | "textual" | "textual_replay" => Some(Self::TextualReplay),
            _ => None,
        }
    }
}

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
                Turn::Clear => {
                    // Clear marker is not rendered - it only affects LCM rebuild.
                }
            }
        }

        // Anthropic compat: must not end with assistant.
        if out.last().map(|m| m["role"] == "assistant").unwrap_or(false) {
            out.push(json!({"role": "user", "content": CONTINUE_SENTINEL}));
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
/// - Assistant replay mode can preserve native `tool_calls` or convert to textual replay.
/// - Consecutive same-role messages are merged.
/// - Output always ends with `role:user`.
#[derive(Debug, Clone, Copy)]
pub struct LocalProtocol {
    replay_mode: LocalReplayMode,
}

impl LocalProtocol {
    pub const fn native() -> Self {
        Self {
            replay_mode: LocalReplayMode::NativeToolCalls,
        }
    }

    pub const fn textual() -> Self {
        Self {
            replay_mode: LocalReplayMode::TextualReplay,
        }
    }

    pub fn auto_for_model(model: &str) -> Self {
        if let Some(mode) = LocalReplayMode::from_env() {
            return Self { replay_mode: mode };
        }

        let caps = lookup(model, &HashMap::new());
        if !caps.tool_calling || caps.size_class == ModelSizeClass::Small {
            Self::textual()
        } else {
            Self::native()
        }
    }
}

impl Default for LocalProtocol {
    fn default() -> Self {
        Self::native()
    }
}

impl ConversationProtocol for LocalProtocol {
    fn render(&self, system: &str, turns: &[Turn]) -> Vec<Value> {
        let mut out: Vec<Value> = Vec::with_capacity(turns.len() + 2);

        // Leading system message (index 0 is the only allowed system position).
        out.push(json!({"role": "system", "content": system}));

        for turn in turns {
            let msg = match turn {
                Turn::System { content } => {
                    // Mid-thread system → user notice.
                    json!({"role": "user", "content": format!("{} {}", SYSTEM_NOTICE_PREFIX, content)})
                }
                Turn::User { content, .. } => {
                    json!({"role": "user", "content": content})
                }
                Turn::Assistant { text, tool_calls } => {
                    match self.replay_mode {
                        LocalReplayMode::NativeToolCalls => {
                            let content_val = match text {
                                Some(t) if !t.is_empty() => Value::String(t.clone()),
                                _ => Value::Null,
                            };
                            if tool_calls.is_empty() {
                                json!({"role": "assistant", "content": content_val})
                            } else {
                                let tc_json: Vec<Value> = tool_calls
                                    .iter()
                                    .map(tool_call_to_openai_json)
                                    .collect();
                                json!({
                                    "role": "assistant",
                                    "content": content_val,
                                    "tool_calls": tc_json,
                                })
                            }
                        }
                        LocalReplayMode::TextualReplay => {
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
                    }
                }
                Turn::ToolResult { tool, call_id, result, .. } => {
                    // Tool results become user messages for alternation compliance.
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
                        "content": format!("{} {}", CONTEXT_SUMMARY_PREFIX, text),
                    })
                }
                Turn::Clear => {
                    // Clear marker is not rendered - it only affects LCM rebuild.
                    continue;
                }
            };
            out.push(msg);
        }

        // Merge consecutive same-role messages (avoids "consecutive user" violations).
        out = merge_consecutive_role(out);

        // Must always end with user.
        if out.last().map(|m| m["role"] != "user").unwrap_or(true) {
            out.push(json!({"role": "user", "content": CONTINUE_SENTINEL}));
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
                if !is_merge_safe(last, &msg, &role) {
                    out.push(msg);
                    continue;
                }
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

fn is_merge_safe(last: &Value, current: &Value, role: &str) -> bool {
    if role == "assistant" {
        // Assistant metadata like tool_calls must never be dropped by merge.
        let last_has_tool_calls = last
            .get("tool_calls")
            .and_then(|v| v.as_array())
            .map(|arr| !arr.is_empty())
            .unwrap_or(false);
        let current_has_tool_calls = current
            .get("tool_calls")
            .and_then(|v| v.as_array())
            .map(|arr| !arr.is_empty())
            .unwrap_or(false);
        if last_has_tool_calls || current_has_tool_calls {
            return false;
        }
    }

    let last_has_extra_fields = has_non_content_fields(last);
    let current_has_extra_fields = has_non_content_fields(current);
    !(last_has_extra_fields || current_has_extra_fields)
}

fn has_non_content_fields(msg: &Value) -> bool {
    msg.as_object()
        .map(|obj| obj.keys().any(|k| k != "role" && k != "content"))
        .unwrap_or(false)
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
        let wire = LocalProtocol::default().render("sys", &tool_turns());
        assert!(wire.iter().all(|m| m["role"] != "tool"));
    }

    #[test]
    fn local_ends_with_user() {
        let turns = vec![
            Turn::User { content: "hi".into(), media: vec![] },
            Turn::Assistant { text: Some("hello".into()), tool_calls: vec![] },
        ];
        let wire = LocalProtocol::default().render("sys", &turns);
        assert_eq!(wire.last().unwrap()["role"], "user");
    }

    #[test]
    fn local_assistant_preserves_tool_calls() {
        let wire = LocalProtocol::native().render("sys", &tool_turns());
        // LocalProtocol now preserves tool_calls for native tool-calling support (LM Studio).
        let assistant_msgs: Vec<_> = wire.iter().filter(|m| m["role"] == "assistant").collect();
        assert!(!assistant_msgs.is_empty(), "Should have assistant messages");
        for msg in &assistant_msgs {
            if msg.get("tool_calls").is_some() {
                let tc = msg["tool_calls"].as_array().unwrap();
                assert!(!tc.is_empty(), "tool_calls should not be empty if present");
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
        let wire = LocalProtocol::default().render("sys", &turns);
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

    #[test]
    fn merge_consecutive_role_preserves_assistant_tool_call_metadata() {
        let msgs = vec![
            json!({"role": "system", "content": "sys"}),
            json!({
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "tc_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"}
                }]
            }),
            json!({"role": "assistant", "content": "Calling tool now"}),
            json!({"role": "user", "content": "continue"}),
        ];

        let merged = merge_consecutive_role(msgs);
        assert_eq!(merged.len(), 4, "assistant entries with metadata must not merge");
        assert!(merged[1].get("tool_calls").is_some());
    }

    #[test]
    fn local_textual_replay_includes_called_tool_arguments() {
        let wire = LocalProtocol::textual().render("sys", &tool_turns());
        let assistant = wire
            .iter()
            .find(|m| m["role"] == "assistant")
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .unwrap_or("");
        assert!(assistant.contains("[I called: read_file"));
        assert!(assistant.contains("\"path\":\"Cargo.toml\""));
    }

    #[test]
    fn local_textual_replay_formats_tool_result_as_system_completion() {
        let wire = LocalProtocol::textual().render("sys", &tool_turns());
        let tool_result = wire
            .iter()
            .find(|m| {
                m["role"] == "user"
                    && m["content"]
                        .as_str()
                        .unwrap_or("")
                        .contains("tool execution complete")
            })
            .and_then(|m| m["content"].as_str())
            .unwrap_or("");
        assert!(tool_result.contains("read_file(tc_1)"));
        assert!(tool_result.contains("data"));
    }
}
