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
use std::sync::LazyLock;

use super::turn::{ToolCall, Turn};
use crate::agent::model_capabilities::{lookup_default, ModelSizeClass};

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
        let caps = lookup_default(model);
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
                    ok,
                } => match self.replay_mode {
                    LocalReplayMode::NativeToolCalls => {
                        // Native OpenAI format: role:tool with
                        // tool_call_id + name + content. No
                        // [System: tool succeeded...] wrapper — the
                        // chat template handles tool messages natively
                        // via <|im_start|>tool. This keeps tool-result
                        // bytes deterministic (no UUID in a header
                        // string) and avoids the user-role conversion
                        // that caused repair_role_alternation to insert
                        // dynamic separators between consecutive
                        // results (the prefix-cache divergence root
                        // cause, 2026-07-27).
                        json!({
                            "role": "tool",
                            "tool_call_id": call_id,
                            "name": tool,
                            "content": result,
                        })
                    }
                    LocalReplayMode::TextualReplay => {
                        // Textual models have no native tool_calls —
                        // render as user messages with the [System: ...]
                        // header so the model can correlate results to
                        // calls by position and tool name.
                        let header = if *ok {
                            format!("[System: tool succeeded - {}({}) returned]", tool, call_id)
                        } else {
                            format!(
                                "[System: TOOL FAILED - {}({}) did not complete \
                                 successfully. Report this error exactly before \
                                 trying another path]",
                                tool, call_id
                            )
                        };
                        json!({
                            "role": "user",
                            "content": format!("{header}:\n{result}"),
                        })
                    }
                },
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
        out = repair_role_alternation(out);

        // Append the continuation sentinel only when the render ends with an
        // assistant turn (mirrors CloudProtocol at line 161). The previous
        // broader condition (`!= "user"`) fired after `role:tool` results too,
        // injecting a transient `Continue.` user message that was hashed into
        // the stored fingerprint but never persisted — so the next iteration's
        // model response filled that index and the prefix diverged
        // (~60s/turn re-prefills on local, every divergence in 2026-07-27/28
        // logs shared the literal sentinel hash 6787951084353679885).
        // Native tool-calling chat templates handle `tool → assistant` without
        // help; textual replay folds tool results into user messages already.
        if out.last().map(|m| m["role"] == "assistant").unwrap_or(false) {
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

/// Marker heading of the textual-tools prompt block. Callers use it to check
/// whether the block was already appended to a system prompt (idempotence).
pub const TEXTUAL_TOOLS_MARKER: &str = "## Tool Calls (textual)";

/// Render tool definitions as a system-prompt block that teaches the textual
/// call syntax parsed by [`parse_textual_tool_calls`].
///
/// For models with no tool-calling training (e.g. VibeThinker) the native
/// `tools` request parameter is useless — their chat templates either ignore
/// or reject it. Instead, the caller drops the `tools` parameter and appends
/// this block so the model can still act through the textual protocol.
pub fn textual_tools_block(tool_defs: &[Value]) -> String {
    let mut out = format!(
        "\n\n{TEXTUAL_TOOLS_MARKER}\n\
         You do not have native tool calling. To use a tool, write this exact \
         pattern on its own line, then stop and wait:\n\
         [I called: tool_name({{\"arg\": \"value\"}})]\n\
         The result arrives in the next user message. One call per turn, \
         arguments as strict JSON.\n\
         Available tools:\n"
    );
    for def in tool_defs {
        let name = def
            .pointer("/function/name")
            .and_then(|v| v.as_str())
            .unwrap_or("?");
        let desc = def
            .pointer("/function/description")
            .and_then(|v| v.as_str())
            .and_then(|d| d.lines().next())
            .unwrap_or("");
        let params: Vec<&str> = def
            .pointer("/function/parameters/properties")
            .and_then(|p| p.as_object())
            .map(|o| o.keys().map(|k| k.as_str()).collect())
            .unwrap_or_default();
        out.push_str(&format!("- {}({}) — {}\n", name, params.join(", "), desc));
    }
    out
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

// ─────────────────────────────────────────────────────────────
// XML tool-call parsing (Qwen-style <tool_call> blocks)
// ─────────────────────────────────────────────────────────────

// Matches `<tool_call>...</tool_call>` blocks (possibly multiline).
static XML_TOOL_CALL_BLOCK_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?si)<tool_call>\s*(.*?)\s*</tool_call>").expect("xml tool_call block regex")
});

// Extracts function name from `<function=NAME>` or `<function name="NAME">`.
//
// Keep this intentionally strict. Local models sometimes omit the `>` after
// the function name, producing fragments like `<function=list_dir\n</function>`.
// A permissive `[^">]+` capture turns that into a real tool name and leaks XML
// into the tool engine/TUI.
static XML_FUNCTION_NAME_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?i)<function(?:=|\s+name=)"?([A-Za-z_][A-Za-z0-9_]*)"?\s*>"#)
        .expect("xml function name regex")
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
    let mut stripped = String::with_capacity(content.len());
    let mut last_end = 0usize;

    for block_cap in XML_TOOL_CALL_BLOCK_RE.captures_iter(content) {
        let Some(block) = block_cap.get(0) else {
            continue;
        };
        let inner = block_cap.get(1).map(|m| m.as_str()).unwrap_or("");
        let should_strip = inner.trim().is_empty() || XML_FUNCTION_NAME_RE.is_match(inner);

        if should_strip {
            stripped.push_str(&content[last_end..block.start()]);
            last_end = block.end();
        }
    }

    stripped.push_str(&content[last_end..]);
    stripped.trim().to_string()
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
/// Repair user/assistant alternation by inserting an empty opposite-role
/// separator between consecutive same-role messages.
///
/// This deliberately replaces the old `merge_consecutive_role` concatenation.
/// Merging is not append-stable: when a same-role turn lands later (async tool
/// completion, parallel tool results, a dropped empty assistant reply), it
/// folds into an EARLIER message and changes its bytes across renders. That
/// mutation busts server-side prefix caches — the agent loop's
/// `prompt_prefix_diverged` warning, measured as ~60s full re-prefills per
/// affected turn on local models — and under higgs's message-boundary splice
/// it silently hides the folded content from the model instead. Separators are
/// position-stable by construction: whether a separator exists between log
/// entries i and i+1 depends only on those two entries, so a rendered prefix
/// never changes once written.
fn repair_role_alternation(messages: Vec<Value>) -> Vec<Value> {
    let mut out: Vec<Value> = Vec::with_capacity(messages.len() * 2);
    for msg in messages {
        if let Some(last) = out.last() {
            let last_role = last["role"].as_str().unwrap_or("");
            let role = msg["role"].as_str().unwrap_or("");
            // Consecutive tool messages are valid in the OpenAI format
            // (parallel tool results from one assistant turn). Inserting
            // a user separator between them would mutate the rendered
            // prefix on every new tool result — the exact cache-busting
            // pattern observed 2026-07-27.
            if last_role == role && role != "system" && role != "tool" {
                let sep = if role == "user" { "assistant" } else { "user" };
                out.push(json!({"role": sep, "content": ""}));
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
    fn textual_tools_block_teaches_parseable_syntax() {
        let defs = vec![json!({"type": "function", "function": {
            "name": "read_file",
            "description": "Read a file.\nSecond line is dropped.",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}
        }})];
        let block = textual_tools_block(&defs);
        assert!(block.contains(TEXTUAL_TOOLS_MARKER));
        assert!(block.contains("- read_file(path) — Read a file."));
        assert!(!block.contains("Second line"));
        // Self-consistency: the exact syntax the block teaches must round-trip
        // through our own parser.
        let parsed = parse_textual_tool_calls(r#"[I called: tool_name({"arg": "value"})]"#);
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].tool, "tool_name");
    }

    #[test]
    fn native_mode_uses_tool_role() {
        // NativeToolCalls mode (default for local with higgs) now
        // renders tool results as role:tool — the OpenAI-native format
        // the chat template handles via <|im_start|>tool. This replaced
        // the old role:user conversion that caused prefix-cache
        // instability (2026-07-27).
        let wire = LocalProtocol::default().render("sys", &tool_turns());
        assert!(
            wire.iter().any(|m| m["role"] == "tool"),
            "NativeToolCalls mode must include role:tool messages"
        );
    }

    #[test]
    fn textual_mode_has_no_tool_role() {
        // TextualReplay mode still converts tool results to role:user
        // because there are no native tool_calls to pair with.
        let wire = LocalProtocol::textual().render("sys", &tool_turns());
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
    fn alternation_repair_separates_consecutive_users_without_mutation() {
        let msgs = vec![
            json!({"role": "system", "content": "sys"}),
            json!({"role": "user", "content": "hello"}),
            json!({"role": "user", "content": "world"}),
            json!({"role": "assistant", "content": "hi"}),
        ];
        let repaired = repair_role_alternation(msgs);
        // system, user(hello), empty assistant separator, user(world), assistant(hi)
        assert_eq!(repaired.len(), 5);
        assert_eq!(repaired[1]["content"], "hello");
        assert_eq!(repaired[2]["role"], "assistant");
        assert_eq!(repaired[2]["content"], "");
        assert_eq!(repaired[3]["content"], "world");
    }

    #[test]
    fn alternation_repair_is_append_stable() {
        // The rendered prefix must not change when a same-role message lands
        // later (async tool completion) — this is the prefix-cache contract.
        let base = vec![
            json!({"role": "system", "content": "sys"}),
            json!({"role": "user", "content": "tool result 1"}),
        ];
        let mut grown = base.clone();
        grown.push(json!({"role": "user", "content": "tool result 2"}));

        let repaired_base = repair_role_alternation(base);
        let repaired_grown = repair_role_alternation(grown);
        assert_eq!(
            &repaired_grown[..repaired_base.len()],
            &repaired_base[..],
            "growing the log must only append rendered messages"
        );
    }

    #[test]
    fn alternation_repair_preserves_assistant_tool_call_metadata() {
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

        let repaired = repair_role_alternation(msgs);
        // A user separator lands between the two assistant messages; the
        // tool_calls metadata is untouched.
        assert_eq!(repaired.len(), 5);
        assert!(repaired[1].get("tool_calls").is_some());
        assert_eq!(repaired[2]["role"], "user");
        assert_eq!(repaired[3]["content"], "Calling tool now");
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
    fn local_textual_replay_formats_tool_result_as_success() {
        let wire = LocalProtocol::textual().render("sys", &tool_turns());
        let tool_result = wire
            .iter()
            .find(|m| {
                m["role"] == "user"
                    && m["content"]
                        .as_str()
                        .unwrap_or("")
                        .contains("tool succeeded")
            })
            .and_then(|m| m["content"].as_str())
            .unwrap_or("");
        assert!(tool_result.contains("read_file(tc_1)"));
        assert!(tool_result.contains("data"));
    }

    #[test]
    fn local_textual_replay_formats_failed_tool_result_as_failure() {
        let turns = vec![Turn::ToolResult {
            call_id: "tc_1".into(),
            tool: "list_dir".into(),
            result: "Error: Directory not found: /bad/path".into(),
            ok: false,
        }];
        let wire = LocalProtocol::textual().render("sys", &turns);
        let tool_result = wire
            .iter()
            .find(|m| {
                m["role"] == "user" && m["content"].as_str().unwrap_or("").contains("TOOL FAILED")
            })
            .and_then(|m| m["content"].as_str())
            .unwrap_or("");
        assert!(tool_result.contains("list_dir(tc_1)"));
        assert!(tool_result.contains("Error: Directory not found: /bad/path"));
        assert!(tool_result.contains("Report this error exactly"));
    }

    // -----------------------------------------------------------------
    // NativeToolCalls mode: tool results use role:"tool" (not role:"user")
    //
    // The old code converted ALL tool results to role:"user" with a
    // [System: tool succeeded - tool(call_id) returned] header. This
    // broke prefix-cache stability: (1) the unique call_id UUID made
    // every receipt hash differently, (2) consecutive tool results
    // became consecutive user messages → repair_role_alternation
    // inserted dynamic separators → position shifts on append.
    //
    // NativeToolCalls mode now emits the OpenAI-native tool format.
    // TextualReplay mode is unchanged (it still needs user-role
    // rendering because there's no native tool_calls to pair with).
    // -----------------------------------------------------------------

    #[test]
    fn native_tool_calls_renders_tool_result_as_role_tool() {
        let wire = LocalProtocol::native().render("sys", &tool_turns());
        let tool_msg = wire
            .iter()
            .find(|m| m["role"] == "tool")
            .expect("NativeToolCalls mode must render tool results as role:tool");
        assert_eq!(
            tool_msg["tool_call_id"], "tc_1",
            "tool message must carry tool_call_id for API compliance"
        );
        assert_eq!(
            tool_msg["name"], "read_file",
            "tool message must carry the tool name"
        );
        assert_eq!(
            tool_msg["content"], "data",
            "tool message content is the raw result, no wrapper"
        );
    }

    #[test]
    fn native_tool_calls_no_user_header_wrapper() {
        let wire = LocalProtocol::native().render("sys", &tool_turns());
        let has_system_header = wire.iter().any(|m| {
            m["content"]
                .as_str()
                .unwrap_or("")
                .contains("[System: tool succeeded")
        });
        assert!(
            !has_system_header,
            "NativeToolCalls mode must NOT wrap tool results in [System: tool succeeded...] user messages"
        );
    }

    #[test]
    fn native_tool_calls_consecutive_results_no_separator() {
        // Two parallel tool results from one assistant turn — the
        // OpenAI format allows consecutive role:"tool" messages.
        // repair_role_alternation must NOT insert user separators
        // between them.
        let turns = vec![
            Turn::Assistant {
                text: None,
                tool_calls: vec![
                    ToolCall {
                        id: "tc_1".into(),
                        tool: "read_file".into(),
                        args: json!({"path": "a"}),
                    },
                    ToolCall {
                        id: "tc_2".into(),
                        tool: "read_file".into(),
                        args: json!({"path": "b"}),
                    },
                ],
            },
            Turn::ToolResult {
                call_id: "tc_1".into(),
                tool: "read_file".into(),
                result: "content_a".into(),
                ok: true,
            },
            Turn::ToolResult {
                call_id: "tc_2".into(),
                tool: "read_file".into(),
                result: "content_b".into(),
                ok: true,
            },
        ];
        let wire = LocalProtocol::native().render("sys", &turns);
        // Collect roles to check alternation pattern.
        let roles: Vec<&str> = wire
            .iter()
            .map(|m| m["role"].as_str().unwrap_or(""))
            .collect();
        // There must be NO empty-user separator between consecutive
        // tool messages.
        let has_empty_user_between_tools = roles
            .windows(3)
            .any(|w| w[0] == "tool" && w[1] == "user" && w[2] == "tool");
        assert!(
            !has_empty_user_between_tools,
            "consecutive tool results must not get a user separator; roles: {:?}",
            roles
        );
    }

    #[test]
    fn textual_replay_still_renders_tool_result_as_user() {
        // TextualReplay mode is UNCHANGED — it still converts tool
        // results to role:"user" with the [System: ...] header because
        // there are no native tool_calls to pair with.
        let wire = LocalProtocol::textual().render("sys", &tool_turns());
        let has_user_tool_result = wire.iter().any(|m| {
            m["role"] == "user"
                && m["content"]
                    .as_str()
                    .unwrap_or("")
                    .contains("tool succeeded")
        });
        assert!(
            has_user_tool_result,
            "TextualReplay mode must still render tool results as user messages"
        );
    }

    #[test]
    fn native_tool_calls_failed_result_no_success_header() {
        let turns = vec![Turn::ToolResult {
            call_id: "tc_1".into(),
            tool: "exec".into(),
            result: "Error: command failed".into(),
            ok: false,
        }];
        let wire = LocalProtocol::native().render("sys", &turns);
        let tool_msg = wire
            .iter()
            .find(|m| m["role"] == "tool")
            .expect("failed tool results must also use role:tool");
        assert_eq!(
            tool_msg["content"], "Error: command failed",
            "failed tool result content is the raw error, no wrapper"
        );
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

    #[test]
    fn local_protocol_is_derived_from_small_model_capabilities() {
        assert!(LocalProtocol::auto_for_model("nanbeige-3b").is_textual_replay());
    }

    #[test]
    fn local_protocol_is_derived_from_native_model_capabilities() {
        assert!(!LocalProtocol::auto_for_model("qwen3.5-35b-a3b").is_textual_replay());
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
        let text =
            r#"[Calling tool: write_file({"path": "/tmp/game.py", "content": "print('hi')"})]"#;
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
        assert!(
            calls.is_empty(),
            "Empty <tool_call> block should yield no parsed calls"
        );
    }

    #[test]
    fn strip_xml_removes_empty_blocks() {
        let text = "Some text. <tool_call>\n</tool_call> More text.";
        let stripped = strip_xml_tool_calls(text);
        assert!(
            !stripped.contains("<tool_call>"),
            "Empty XML blocks should be stripped"
        );
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
    fn parse_xml_rejects_function_name_spanning_close_tag() {
        let text = r#"<tool_call>
  <function=list_dir
</function>
</tool_call>"#;
        let calls = parse_xml_tool_calls(text);
        assert!(
            calls.is_empty(),
            "malformed function tags must not become executable tool names"
        );
        let stripped = strip_xml_tool_calls(text);
        assert!(
            stripped.contains("<tool_call>") && stripped.contains("</function"),
            "malformed XML must remain visible for pathological-output recovery"
        );
    }

    #[test]
    fn parse_xml_rejects_function_name_spanning_parameter_tag() {
        let text = r#"<tool_call>
  <function=read_file
  <parameter=path>/tmp/x</parameter>
  </function>
</tool_call>"#;
        let calls = parse_xml_tool_calls(text);
        assert!(
            calls.is_empty(),
            "malformed function tags must not capture parameter markup"
        );
        let stripped = strip_xml_tool_calls(text);
        assert!(
            stripped.contains("<tool_call>") && stripped.contains("<parameter=path"),
            "malformed XML must remain visible for pathological-output recovery"
        );
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
