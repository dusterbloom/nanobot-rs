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

use once_cell::sync::Lazy;
use regex::Regex;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::LazyLock;

use super::turn::{ToolCall, Turn};
use crate::agent::model_capabilities::{lookup, ModelSizeClass};

// Matches the outer `[I called: ...]` or `[Called: ...]` or `[called ...]` or
// `[Calling tool: ...]` bracket. Captures the inner content.
// The alternation handles both past tense (called/calling) and the extra "tool"
// word that local models sometimes insert.
static TEXTUAL_CALL_OUTER_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\[(?:I\s+)?call(?:ed|ing)(?:\s+tool)?[:\s]\s*(.*?)\]")
        .expect("textual call outer regex")
});

// Matches a single `tool_name({...})` pair within the inner content.
// The format rendered by TextualReplay is: tool_name({"arg": "val"})
// Captures: (1) tool name, (2) JSON args string (including the braces)
static TEXTUAL_CALL_ITEM_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(\w+)\s*\(\s*(\{[^}]*(?:\{[^}]*\}[^}]*)?\})\s*\)")
        .expect("textual call item regex")
});

/// A parsed tool call extracted from textual replay format.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedToolCall {
    pub tool: String,
    pub args: Value,
}

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

    /// Returns `true` when this protocol renders past tool calls as textual
    /// summaries (`[I called: tool_name({...})]`) instead of native `tool_calls`
    /// JSON.  Callers use this to:
    /// - Skip hallucination validation (the pattern is expected, not erroneous).
    /// - Parse textual tool call patterns from the model's response.
    fn is_textual_replay(&self) -> bool {
        false
    }
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
                        let tc_json: Vec<Value> =
                            tool_calls.iter().map(tool_call_to_openai_json).collect();
                        out.push(json!({
                            "role": "assistant",
                            "content": content_val,
                            "tool_calls": tc_json,
                        }));
                    }
                }
                Turn::ToolResult {
                    call_id,
                    tool,
                    result,
                    ..
                } => {
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
        if out
            .last()
            .map(|m| m["role"] == "assistant")
            .unwrap_or(false)
        {
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
                Turn::Assistant { text, tool_calls } => match self.replay_mode {
                    LocalReplayMode::NativeToolCalls => {
                        let content_val = match text {
                            Some(t) if !t.is_empty() => Value::String(t.clone()),
                            _ => Value::Null,
                        };
                        if tool_calls.is_empty() {
                            json!({"role": "assistant", "content": content_val})
                        } else {
                            let tc_json: Vec<Value> =
                                tool_calls.iter().map(tool_call_to_openai_json).collect();
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
                },
                Turn::ToolResult {
                    tool,
                    call_id,
                    result,
                    ..
                } => {
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

    fn is_textual_replay(&self) -> bool {
        self.replay_mode == LocalReplayMode::TextualReplay
    }
}

// ─────────────────────────────────────────────────────────────
// Public helpers
// ─────────────────────────────────────────────────────────────

/// Parse tool calls from a textual replay response.
///
/// TextualReplay-mode models express tool intent by writing bracket patterns like:
/// ```text
/// [I called: read_file({"path": "x"}), write_file({"path": "y", "content": "z"})]
/// [Called: shell_exec({"cmd": "ls"})]
/// [I called read_file({"path": "x"})]
/// ```
///
/// This function extracts each `tool_name({args})` pair and returns a
/// `Vec<ParsedToolCall>`. Entries with malformed JSON are silently skipped.
///
/// The caller is responsible for assigning call IDs and stripping the matched
/// text from the response content.
pub fn parse_textual_tool_calls(text: &str) -> Vec<ParsedToolCall> {
    let mut result = Vec::new();

    for outer_cap in TEXTUAL_CALL_OUTER_RE.captures_iter(text) {
        let inner = match outer_cap.get(1) {
            Some(m) => m.as_str(),
            None => continue,
        };

        for item_cap in TEXTUAL_CALL_ITEM_RE.captures_iter(inner) {
            let tool = item_cap[1].to_string();
            let args_str = &item_cap[2];

            match serde_json::from_str::<Value>(args_str) {
                Ok(args) => result.push(ParsedToolCall { tool, args }),
                Err(_) => {
                    // Best-effort: skip malformed JSON, don't abort the whole parse.
                }
            }
        }
    }

    result
}

/// Strip textual tool call brackets from response content.
///
/// Removes all `[I called: ...]` / `[Called: ...]` patterns, trims, and returns
/// the cleaned text.  Used after `parse_textual_tool_calls()` to avoid sending
/// bracket noise to the user or to downstream tools.
pub fn strip_textual_tool_calls(content: &str) -> String {
    TEXTUAL_CALL_OUTER_RE
        .replace_all(content, "")
        .trim()
        .to_string()
}

/// Parse tool calls using the model-appropriate parser from the registry.
///
/// Selects the correct parser for the given model by name substring matching,
/// with an optional config override to force a specific parser.
///
/// Returns a `Vec<(name, arguments)>` compatible with the existing tool call
/// dispatch format. The original `parse_textual_tool_calls` is kept for
/// backward compatibility; this is the new preferred entry point.
pub fn parse_tool_calls_for_model(
    text: &str,
    model_name: &str,
    parser_override: Option<&str>,
) -> Vec<(String, Value)> {
    use crate::agent::parsers::ParserRegistry;
    let registry = ParserRegistry::new();
    let parser = registry.select_for_model(model_name, parser_override);
    parser
        .parse(text)
        .into_iter()
        .map(|tc| (tc.name, tc.arguments))
        .collect()
}

// ─────────────────────────────────────────────────────────────
// XML tool-call parsing (Qwen-style <tool_call> blocks)
// ─────────────────────────────────────────────────────────────

// Matches `<tool_call>...</tool_call>` blocks (possibly multiline).
static XML_TOOL_CALL_BLOCK_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?si)<tool_call>\s*(.*?)\s*</tool_call>").expect("xml tool_call block regex")
});

// Extracts function name from `<function=NAME>` or `<function name="NAME">`.
static XML_FUNCTION_NAME_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?i)<function[= ]+"?([^">]+)"?>"#).expect("xml function name regex")
});

// Extracts `<parameter=KEY>VALUE</parameter>` pairs.
static XML_PARAMETER_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?si)<parameter=(\w+)>\s*(.*?)\s*</parameter>"#).expect("xml parameter regex")
});

/// Parse XML-style tool calls from response content.
///
/// Models like Qwen3.5-2B sometimes emit tool calls as:
/// ```text
/// <tool_call>
///   <function=web_search>
///   <parameter=query>latest news</parameter>
///   <parameter=count>10</parameter>
///   </function>
/// </tool_call>
/// ```
///
/// Returns a vec of `ParsedToolCall` with tool name and arguments as JSON.
pub fn parse_xml_tool_calls(text: &str) -> Vec<ParsedToolCall> {
    let mut result = Vec::new();

    for block_cap in XML_TOOL_CALL_BLOCK_RE.captures_iter(text) {
        let inner = match block_cap.get(1) {
            Some(m) => m.as_str(),
            None => continue,
        };

        let tool_name = match XML_FUNCTION_NAME_RE.captures(inner) {
            Some(cap) => cap[1].trim().to_string(),
            None => continue,
        };

        let mut args = serde_json::Map::new();
        for param_cap in XML_PARAMETER_RE.captures_iter(inner) {
            let key = param_cap[1].to_string();
            let value = param_cap[2].trim().to_string();
            // Try parsing as number/bool/null, fall back to string.
            let json_val =
                serde_json::from_str::<Value>(&value).unwrap_or_else(|_| Value::String(value));
            args.insert(key, json_val);
        }

        result.push(ParsedToolCall {
            tool: tool_name,
            args: Value::Object(args),
        });
    }

    result
}

/// Strip XML tool call blocks from response content.
pub fn strip_xml_tool_calls(content: &str) -> String {
    XML_TOOL_CALL_BLOCK_RE
        .replace_all(content, "")
        .trim()
        .to_string()
}

// ─────────────────────────────────────────────────────────────
// Streaming XML tool-call filter
// ─────────────────────────────────────────────────────────────

/// State machine that suppresses `<tool_call>...</tool_call>` blocks from
/// streaming text deltas so they don't render in the terminal.
///
/// Call `filter()` for each incoming delta. It returns the text to display
/// (possibly empty if everything was buffered).
pub struct XmlToolCallFilter {
    state: XmlFilterState,
    buf: String,
}

#[derive(Debug, PartialEq)]
enum XmlFilterState {
    Normal,
    /// We've seen a partial or full `<tool_call` prefix and are buffering
    /// until `</tool_call>` closes the block.
    Buffering,
}

impl XmlToolCallFilter {
    pub fn new() -> Self {
        Self {
            state: XmlFilterState::Normal,
            buf: String::new(),
        }
    }

    /// Filter a streaming delta. Returns text safe to display.
    pub fn filter(&mut self, delta: &str) -> String {
        // In Buffering state, just accumulate until closing tag.
        if self.state == XmlFilterState::Buffering {
            self.buf.push_str(delta);
            if self.buf.ends_with("</tool_call>") || self.buf.contains("</tool_call>") {
                // Find text after the closing tag (if any).
                let after = self
                    .buf
                    .find("</tool_call>")
                    .map(|i| &self.buf[i + 12..])
                    .unwrap_or("")
                    .to_string();
                self.buf.clear();
                self.state = XmlFilterState::Normal;
                if after.is_empty() {
                    return String::new();
                }
                // Recursively filter the remainder (might have another tool_call).
                return self.filter(&after);
            }
            return String::new();
        }

        // Normal state: if we have a pending partial buffer, prepend it.
        let combined;
        let text = if !self.buf.is_empty() {
            combined = std::mem::take(&mut self.buf) + delta;
            combined.as_str()
        } else {
            delta
        };

        // Look for `<tool_call` in the text.
        if let Some(start) = text.find("<tool_call") {
            let before = &text[..start];
            let rest = &text[start..];

            // Check if the closing tag is in this chunk too.
            if let Some(end) = rest.find("</tool_call>") {
                let after = &rest[end + 12..];
                let mut out = before.to_string();
                // Recursively filter remainder.
                if !after.is_empty() {
                    out.push_str(&self.filter(after));
                }
                return out;
            }

            // No closing tag yet — buffer the rest, return text before.
            self.state = XmlFilterState::Buffering;
            self.buf = rest.to_string();
            return before.to_string();
        }

        // No `<tool_call` found. But the end of the text might be a partial
        // prefix like `<tool_` that continues in the next delta.
        const TAG: &str = "<tool_call";
        for split in (1..TAG.len()).rev() {
            if text.ends_with(&TAG[..split]) {
                // Partial prefix — hold it back.
                let safe = &text[..text.len() - split];
                self.buf = text[text.len() - split..].to_string();
                return safe.to_string();
            }
        }

        text.to_string()
    }
}

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
            Turn::User {
                content: "read file".into(),
                media: vec![],
            },
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
            Turn::User {
                content: "hi".into(),
                media: vec![],
            },
            Turn::Assistant {
                text: Some("hello".into()),
                tool_calls: vec![],
            },
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
            Turn::User {
                content: "hi".into(),
                media: vec![],
            },
            Turn::System {
                content: "Injected notice".into(),
            },
            Turn::User {
                content: "go on".into(),
                media: vec![],
            },
        ];
        let wire = LocalProtocol::default().render("sys", &turns);
        let non_first_system = wire.iter().skip(1).any(|m| m["role"] == "system");
        assert!(!non_first_system);
        let has_notice = wire.iter().any(|m| {
            m["content"]
                .as_str()
                .unwrap_or("")
                .contains("Injected notice")
        });
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
            Turn::User {
                content: "hi".into(),
                media: vec![],
            },
            Turn::Assistant {
                text: Some("hello".into()),
                tool_calls: vec![],
            },
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
        assert_eq!(
            merged.len(),
            4,
            "assistant entries with metadata must not merge"
        );
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

    // ---- is_textual_replay() ----

    #[test]
    fn cloud_protocol_is_not_textual_replay() {
        assert!(!CloudProtocol.is_textual_replay());
    }

    #[test]
    fn local_native_is_not_textual_replay() {
        assert!(!LocalProtocol::native().is_textual_replay());
    }

    #[test]
    fn local_textual_is_textual_replay() {
        assert!(LocalProtocol::textual().is_textual_replay());
    }

    // ---- parse_textual_tool_calls() ----

    #[test]
    fn parse_single_call_with_colon() {
        let text = r#"[I called: read_file({"path": "/tmp/foo"})]"#;
        let calls = parse_textual_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].tool, "read_file");
        assert_eq!(calls[0].args["path"], "/tmp/foo");
    }

    #[test]
    fn parse_single_call_without_colon() {
        let text = r#"[I called read_file({"path": "/tmp/bar"})]"#;
        let calls = parse_textual_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].tool, "read_file");
        assert_eq!(calls[0].args["path"], "/tmp/bar");
    }

    #[test]
    fn parse_called_prefix() {
        let text = r#"[Called: shell_exec({"cmd": "ls"})]"#;
        let calls = parse_textual_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].tool, "shell_exec");
        assert_eq!(calls[0].args["cmd"], "ls");
    }

    #[test]
    fn parse_multiple_calls_comma_separated() {
        let text =
            r#"[I called: read_file({"path": "a"}), write_file({"path": "b", "content": "x"})]"#;
        let calls = parse_textual_tool_calls(text);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].tool, "read_file");
        assert_eq!(calls[0].args["path"], "a");
        assert_eq!(calls[1].tool, "write_file");
        assert_eq!(calls[1].args["path"], "b");
    }

    #[test]
    fn parse_empty_args_object() {
        let text = r#"[Called: get_time({})]"#;
        let calls = parse_textual_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].tool, "get_time");
        assert!(calls[0].args.is_object());
    }

    #[test]
    fn parse_skips_malformed_json() {
        // Only the valid call should be returned; the broken one is silently dropped.
        let text = r#"[I called: bad_tool({NOT JSON}), good_tool({"k": "v"})]"#;
        let calls = parse_textual_tool_calls(text);
        // `good_tool` is returned; `bad_tool` was skipped.
        assert!(calls.iter().any(|c| c.tool == "good_tool"));
        assert!(!calls.iter().any(|c| c.tool == "bad_tool"));
    }

    #[test]
    fn parse_no_match_returns_empty() {
        let text = "The answer is 42. No tool calls here.";
        let calls = parse_textual_tool_calls(text);
        assert!(calls.is_empty());
    }

    #[test]
    fn parse_case_insensitive_prefix() {
        let text = r#"[CALLED: read_file({"path": "x"})]"#;
        let calls = parse_textual_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].tool, "read_file");
    }

    // ---- "Calling tool" variant (local model format confusion) ----

    #[test]
    fn parse_calling_tool_format() {
        let text = r#"[Calling tool: write_file({"path": "/tmp/game.py", "content": "print('hi')"})]"#;
        let calls = parse_textual_tool_calls(text);
        assert_eq!(calls.len(), 1, "Should parse [Calling tool: ...] format");
        assert_eq!(calls[0].tool, "write_file");
        assert_eq!(calls[0].args["path"], "/tmp/game.py");
    }

    #[test]
    fn parse_calling_tool_without_colon() {
        let text = r#"[Calling tool read_file({"path": "/tmp/x"})]"#;
        let calls = parse_textual_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].tool, "read_file");
    }

    #[test]
    fn strip_calling_tool_format() {
        let text = r#"Some text. [Calling tool: write_file({"path": "x", "content": "y"})] Done."#;
        let stripped = strip_textual_tool_calls(text);
        assert!(!stripped.contains("[Calling tool:"));
        assert!(stripped.contains("Some text."));
        assert!(stripped.contains("Done."));
    }

    // ---- empty XML tool_call blocks ----

    #[test]
    fn parse_xml_empty_block_returns_empty() {
        let text = "<tool_call>\n</tool_call>";
        let calls = parse_xml_tool_calls(text);
        assert!(calls.is_empty(), "Empty <tool_call> block should yield no parsed calls");
    }

    #[test]
    fn strip_xml_removes_empty_blocks() {
        let text = "Some text. <tool_call>\n</tool_call> More text.";
        let stripped = strip_xml_tool_calls(text);
        assert!(!stripped.contains("<tool_call>"), "Empty XML blocks should be stripped");
        assert!(stripped.contains("Some text."));
        assert!(stripped.contains("More text."));
    }

    // ---- strip_textual_tool_calls() ----

    #[test]
    fn strip_removes_bracket_pattern() {
        let text = r#"Some text. [I called: read_file({"path": "x"})] Done."#;
        let stripped = strip_textual_tool_calls(text);
        assert!(!stripped.contains("[I called:"));
        assert!(stripped.contains("Some text."));
        assert!(stripped.contains("Done."));
    }

    #[test]
    fn strip_leaves_plain_text_unchanged() {
        let text = "The answer is 42.";
        assert_eq!(strip_textual_tool_calls(text), text);
    }

    // ---- parse_xml_tool_calls() ----

    #[test]
    fn parse_xml_single_tool_call() {
        let text = r#"<tool_call>
  <function=web_search>
  <parameter=query>Middle East latest news</parameter>
  <parameter=count>10</parameter>
  </function>
  </tool_call>"#;
        let calls = parse_xml_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].tool, "web_search");
        assert_eq!(calls[0].args["query"], "Middle East latest news");
        assert_eq!(calls[0].args["count"], 10); // parsed as number
    }

    #[test]
    fn parse_xml_multiple_tool_calls() {
        let text = r#"Let me search for that.
<tool_call>
  <function=web_search>
  <parameter=query>news</parameter>
  </function>
</tool_call>
And also:
<tool_call>
  <function=read_file>
  <parameter=path>/tmp/test.txt</parameter>
  </function>
</tool_call>"#;
        let calls = parse_xml_tool_calls(text);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].tool, "web_search");
        assert_eq!(calls[1].tool, "read_file");
        assert_eq!(calls[1].args["path"], "/tmp/test.txt");
    }

    #[test]
    fn parse_xml_no_match() {
        let text = "No tool calls here. Just some <b>HTML</b>.";
        assert!(parse_xml_tool_calls(text).is_empty());
    }

    #[test]
    fn strip_xml_removes_blocks() {
        let text = r#"Some text. <tool_call>
  <function=web_search>
  <parameter=query>test</parameter>
  </function>
  </tool_call> Done."#;
        let stripped = strip_xml_tool_calls(text);
        assert!(!stripped.contains("<tool_call>"));
        assert!(stripped.contains("Some text."));
        assert!(stripped.contains("Done."));
    }

    // ---- XmlToolCallFilter ----

    #[test]
    fn filter_passes_normal_text() {
        let mut f = XmlToolCallFilter::new();
        assert_eq!(f.filter("Hello world"), "Hello world");
    }

    #[test]
    fn filter_suppresses_tool_call_single_chunk() {
        let mut f = XmlToolCallFilter::new();
        let out =
            f.filter("<tool_call><function=test><parameter=a>b</parameter></function></tool_call>");
        assert!(out.is_empty());
    }

    #[test]
    fn filter_suppresses_tool_call_across_chunks() {
        let mut f = XmlToolCallFilter::new();
        assert_eq!(f.filter("Hi! "), "Hi! ");
        assert_eq!(f.filter("<tool_cal"), "");
        assert_eq!(f.filter("l><function=test>"), "");
        assert_eq!(f.filter("<parameter=q>x</parameter></function>"), "");
        assert_eq!(f.filter("</tool_call> bye"), " bye");
    }

    #[test]
    fn filter_passes_non_toolcall_angle_bracket() {
        let mut f = XmlToolCallFilter::new();
        // <b> is not <tool_call, should pass through
        assert_eq!(f.filter("some <b>bold</b> text"), "some <b>bold</b> text");
    }

    #[test]
    fn filter_mixed_content_and_tool_call() {
        let mut f = XmlToolCallFilter::new();
        assert_eq!(f.filter("Before "), "Before ");
        assert!(f
            .filter("<tool_call><function=x></function></tool_call>")
            .is_empty());
        assert_eq!(f.filter(" After"), " After");
    }
}
