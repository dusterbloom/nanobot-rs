#![allow(dead_code)]
//! Response validation to detect hallucinated tool calls and context drift.
//!
//! When local SLMs get confused or context becomes polluted, they may:
//! 1. Write `[Called tool(...)]` in text instead of actually calling tools
//! 2. Claim "let me check/read/look" without executing any tools
//!
//! This module provides validation to catch these patterns and trigger retries.

use std::collections::HashMap;

use once_cell::sync::Lazy;
use regex::Regex;
use serde_json::Value;

static HALLUCINATED_CALL_RE: Lazy<Regex> = Lazy::new(|| {
    // Matches `[Called ...]`, `[I called: ...]`, `[Calling tool: ...]`, etc.
    // Both past tense (called) and present (calling) with optional "tool" word.
    Regex::new(r"(?i)\[(?:\w+\s+)*call(?:ed|ing)(?:\s+tool)?[\s:]").expect("hallucination regex")
});

/// `[get_skills()]`, `[exec(command='date')]` — a Python-ish call literal
/// emitted as plain content instead of a structured tool call. Observed from
/// lfm2-2.6b / lfm2.5-2.6b on higgs, which ignore `tool_choice=required` and
/// never populate `tool_calls`. Distinct from `[Called ...]` narration: there
/// is no verb, just the call.
///
static XML_HALLUCINATED_CALL_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r#"(?isx)
        <xml\b[^>]*>\s*
        <bigtag\b[^>]*\bname\s*=\s*["'][a-z][a-z0-9_]*["'][^>]*>
        .*?<arguments\b
        .*?</bigtag>\s*
        </xml>
        "#,
    )
    .expect("xml hallucinated tool-call regex")
});

static NAMED_TOOL_INTENT_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?ix)
        \b(?:i\s+(?:can|could|will|would|should|need\s+to)|i'll|let\s+me)\s+
        (?:use|run|call|invoke|execute)\s+
        (?:the\s+)?
        (?:
            `?[a-z][a-z0-9]*_[a-z0-9_]*`?
            |
            `?[a-z][a-z0-9_]*`?\s+tool
        )
        (?:\b|`)",
    )
    .expect("named tool intent regex")
});

const RAW_JSON_TOOL_NAMES: &[&str] = &[
    "backtrack",
    "browser",
    "check_inbox",
    "checkpoint",
    "cron",
    "edit_file",
    "exec",
    "execute_code",
    "list_dir",
    "message",
    "plan",
    "read_file",
    "get_skills",
    "recall",
    "remember",
    "send_email",
    "session_search",
    "spawn",
    "todo",
    "web_fetch",
    "web_search",
    "write_file",
];

// ---------------------------------------------------------------------------
// ValidationError
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ValidationError {
    #[error("HallucinatedToolCall")]
    HallucinatedToolCall,
    #[error("ClaimedButNotExecuted")]
    ClaimedButNotExecuted,
}

// ---------------------------------------------------------------------------
// Tool Intent Detection Patterns
// ---------------------------------------------------------------------------

const TOOL_INTENT_PATTERNS: &[&str] = &[
    "let me check",
    "let me read",
    "let me look",
    "i'll check",
    "i'll read",
    "i'll look",
    "i will check",
    "i will read",
    "i will look",
    "the file contains",
    "the result shows",
    "i found that",
    "i can see that",
];

pub(crate) fn has_claimed_tool_intent(content: &str) -> bool {
    let lower = content.to_lowercase();
    TOOL_INTENT_PATTERNS
        .iter()
        .any(|pattern| lower.contains(pattern))
        || NAMED_TOOL_INTENT_RE.is_match(content)
}

pub(crate) fn has_hallucinated_tool_call(content: &str) -> bool {
    HALLUCINATED_CALL_RE.is_match(content)
        || XML_HALLUCINATED_CALL_RE.is_match(content)
        || raw_json_tool_call_span(content).is_some()
}

pub(crate) fn has_xml_hallucinated_tool_call(content: &str) -> bool {
    XML_HALLUCINATED_CALL_RE.is_match(content)
}

pub(crate) fn has_raw_json_hallucinated_tool_call(content: &str) -> bool {
    raw_json_tool_call_span(content).is_some()
}

fn raw_json_tool_call_span(content: &str) -> Option<(usize, usize)> {
    if !content.contains("\"name\"")
        || !(content.contains("\"parameters\"") || content.contains("\"arguments\""))
    {
        return None;
    }

    for (start, _) in content.char_indices().filter(|(_, ch)| *ch == '{') {
        let Some(end) = json_object_end(content, start) else {
            continue;
        };
        let Ok(value) = serde_json::from_str::<Value>(&content[start..end]) else {
            continue;
        };
        if value_is_raw_json_tool_call(&value) {
            return Some((start, end));
        }
    }
    None
}

fn json_object_end(content: &str, start: usize) -> Option<usize> {
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;

    for (offset, ch) in content[start..].char_indices() {
        if in_string {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }

        match ch {
            '"' => in_string = true,
            '{' => depth += 1,
            '}' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return Some(start + offset + ch.len_utf8());
                }
            }
            _ => {}
        }
    }
    None
}

fn value_is_raw_json_tool_call(value: &Value) -> bool {
    let Some(obj) = value.as_object() else {
        return false;
    };
    let Some(name) = obj.get("name").and_then(Value::as_str) else {
        return false;
    };
    if !RAW_JSON_TOOL_NAMES.iter().any(|known| *known == name) {
        return false;
    }
    obj.get("parameters")
        .or_else(|| obj.get("arguments"))
        .is_some_and(Value::is_object)
}

// ---------------------------------------------------------------------------
// ValidationOutcome
// ---------------------------------------------------------------------------

/// Outcome of response validation.
#[derive(Debug, PartialEq)]
pub enum ValidationOutcome {
    /// Response is clean.
    Ok,
    /// Real tool_calls exist but response text contains hallucinated call syntax.
    /// Caller should strip the garbage text and continue.
    StripHallucination,
    /// Validation failed — caller should retry or terminate through control
    /// logic, without exposing the repair text to the user-facing stream.
    Error(ValidationError),
}

// ---------------------------------------------------------------------------
// Validation Functions
// ---------------------------------------------------------------------------

pub fn validate_response(
    content: &str,
    actual_tool_calls: &[HashMap<String, Value>],
    is_textual_replay: bool,
    had_blocked_calls: bool,
) -> ValidationOutcome {
    // In TextualReplay mode the model legitimately writes `[I called: ...]` to
    // express history and intent. The parser upstream extracts real tool calls
    // from those patterns — triggering validation errors here would create a
    // death spiral that exhausts the iteration budget.
    if is_textual_replay {
        return ValidationOutcome::Ok;
    }

    // When the tool guard recently blocked calls, the model may express intent
    // ("let me search/check") because it genuinely wanted to use tools but was
    // prevented. Punishing it with ClaimedButNotExecuted retries creates a
    // death spiral. Only check for hallucinated `[Called ...]` syntax.
    if had_blocked_calls && actual_tool_calls.is_empty() {
        if has_hallucinated_tool_call(content) {
            return ValidationOutcome::Error(ValidationError::HallucinatedToolCall);
        }
        return ValidationOutcome::Ok;
    }

    if has_hallucinated_tool_call(content) {
        if actual_tool_calls.is_empty() {
            return ValidationOutcome::Error(ValidationError::HallucinatedToolCall);
        } else {
            return ValidationOutcome::StripHallucination;
        }
    }

    if actual_tool_calls.is_empty() {
        if has_claimed_tool_intent(content) {
            return ValidationOutcome::Error(ValidationError::ClaimedButNotExecuted);
        }
    }

    ValidationOutcome::Ok
}

/// Strip hallucinated tool-call text from response content.
pub fn strip_hallucinated_text(content: &str) -> String {
    let content = XML_HALLUCINATED_CALL_RE.replace_all(content, "");
    let mut content = content.to_string();
    while let Some((start, end)) = raw_json_tool_call_span(&content) {
        content.replace_range(start..end, "");
    }
    HALLUCINATED_CALL_RE
        .replace_all(&content, "")
        .trim()
        .to_string()
}

pub fn generate_retry_prompt(error: &ValidationError, attempt: u8) -> String {
    match error {
        ValidationError::HallucinatedToolCall => format!(
            "[system] Your previous response described a tool call in text instead of emitting a structured tool_call. \
             Retry {}/3: emit a structured tool_call now, or provide a final answer with no tool-action wording.",
            attempt
        ),
        ValidationError::ClaimedButNotExecuted => format!(
            "[system] Your previous response described future tool use but did not emit a structured tool_call. \
             Retry {}/3: emit a structured tool_call now, or provide a final answer with no tool-action wording.",
            attempt
        ),
    }
}

pub const MAX_VALIDATION_RETRIES: u8 = 3;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tool_call(name: &str) -> HashMap<String, Value> {
        let mut tc = HashMap::new();
        tc.insert("name".to_string(), Value::String(name.to_string()));
        tc
    }

    #[test]
    fn test_reject_hallucinated_called_pattern() {
        let content = "I'll read the file.\n\n[Called read_file({\"path\":\"/tmp/test\"})]";
        let result = validate_response(content, &[], false, false);
        assert!(matches!(
            result,
            ValidationOutcome::Error(ValidationError::HallucinatedToolCall)
        ));
    }

    /// Rust attributes, macros, and indexing must never be flagged as
    /// hallucinated tool calls. The old `BRACKET_CALL_LITERAL_RE` matched
    /// `#[cfg(test)]`, `vec![foo()]`, and `items[len(items)]` — retracting
    /// correct final answers. Those shapes are now handled by
    /// `HALLUCINATED_CALL_RE` (verb-based) and `NAMED_TOOL_INTENT_RE`.
    #[test]
    fn rust_attributes_and_code_patterns_are_not_hallucinated() {
        for content in [
            "Use `#[cfg(test)]` above the module.",
            "The attribute #[derive(Debug, Clone)] is common.",
            "#[allow(dead_code)]",
            "#[serde(rename_all = camelCase)]",
            "#[cfg(feature = python-kernel)]",
            "You can write let v = vec![foo()]; in Rust.",
            "items[len(items)]",
            "arr[max(a,b)]",
        ] {
            assert!(
                matches!(
                    validate_response(content, &[], false, false),
                    ValidationOutcome::Ok
                ),
                "{content:?} must not be flagged as a hallucinated tool call"
            );
        }
    }

    #[test]
    fn test_reject_claimed_but_not_executed() {
        let content = "Let me check that file for you.";
        let result = validate_response(content, &[], false, false);
        assert!(matches!(
            result,
            ValidationOutcome::Error(ValidationError::ClaimedButNotExecuted)
        ));
    }

    #[test]
    fn test_accept_response_with_actual_tools() {
        let content = "Let me check that file for you.";
        let tool_calls = vec![make_tool_call("read_file")];
        let result = validate_response(content, &tool_calls, false, false);
        assert_eq!(result, ValidationOutcome::Ok);
    }

    #[test]
    fn test_accept_plain_response() {
        let content = "The answer is 42.";
        let result = validate_response(content, &[], false, false);
        assert_eq!(result, ValidationOutcome::Ok);
    }

    #[test]
    fn test_detect_multiple_hallucinations() {
        let content = "[Called spawn(...)] and [Called exec(...)]";
        let result = validate_response(content, &[], false, false);
        assert!(matches!(
            result,
            ValidationOutcome::Error(ValidationError::HallucinatedToolCall)
        ));
    }

    #[test]
    fn test_case_insensitive_patterns() {
        let content = "LET ME CHECK that for you.";
        let result = validate_response(content, &[], false, false);
        assert!(matches!(
            result,
            ValidationOutcome::Error(ValidationError::ClaimedButNotExecuted)
        ));
    }

    #[test]
    fn test_detect_claimed_intent_patterns() {
        let test_cases = [
            "the file contains important data",
            "the result shows that",
            "i found that the answer",
            "i can see that it works",
            "I'll check this now",
            "let me look at the code",
            "let me read the file",
        ];

        for content in test_cases {
            let result = validate_response(content, &[], false, false);
            assert!(
                matches!(
                    result,
                    ValidationOutcome::Error(ValidationError::ClaimedButNotExecuted)
                ),
                "Failed to detect intent in: {}",
                content
            );
        }
    }

    #[test]
    fn test_detect_named_tool_future_action() {
        let content = "I can use the `web_fetch` tool to get the content of that URL. Which part?";
        let result = validate_response(content, &[], false, false);
        assert!(
            matches!(
                result,
                ValidationOutcome::Error(ValidationError::ClaimedButNotExecuted)
            ),
            "Named tool narration should be treated as claimed tool intent"
        );
    }

    #[test]
    fn test_detect_xml_tool_call_hallucination() {
        let content = r#"<xml>
  <bigtag name="web_search">
    <arguments>
      <jsonobject>
        <parameters>
          <string>latest news</string>
        </parameters>
      </jsonobject>
    </arguments>
  </bigtag>
</xml>"#;
        let result = validate_response(content, &[], false, false);
        assert!(
            matches!(
                result,
                ValidationOutcome::Error(ValidationError::HallucinatedToolCall)
            ),
            "XML-ish tool-call envelopes should be treated as hallucinated tool calls"
        );
    }

    #[test]
    fn test_detect_raw_json_tool_call_hallucination() {
        let content = r#"{ "name": "exec", "parameters": { "command": "git clone https://github.com/dusterbloom/skybloom", "timeout": 60, "working_dir": "/home/your_user/Dev/nanobot-rs" } }"#;
        let result = validate_response(content, &[], false, false);
        assert!(
            matches!(
                result,
                ValidationOutcome::Error(ValidationError::HallucinatedToolCall)
            ),
            "Raw tool-call JSON should not be treated as a final answer"
        );
    }

    #[test]
    fn test_detect_raw_json_tool_call_after_reasoning() {
        let content = r#"We need to use the exec tool.
{ "name": "exec", "parameters": { "command": "git clone https://github.com/dusterbloom/skybloom", "timeout": 60, "working_dir": "/home/your_user/Dev/nanobot-rs" } }"#;
        let result = validate_response(content, &[], false, false);
        assert!(
            matches!(
                result,
                ValidationOutcome::Error(ValidationError::HallucinatedToolCall)
            ),
            "Raw tool-call JSON after leaked reasoning should still trigger recovery"
        );
    }

    #[test]
    fn test_raw_json_non_tool_name_passes() {
        let content = r#"{ "name": "skybloom", "parameters": { "stars": 42 } }"#;
        let result = validate_response(content, &[], false, false);
        assert_eq!(result, ValidationOutcome::Ok);
    }

    #[test]
    fn test_named_tool_intent_does_not_catch_plain_capability_talk() {
        let content = "I can use Rust, Python, and JavaScript for serious coding.";
        let result = validate_response(content, &[], false, false);
        assert_eq!(result, ValidationOutcome::Ok);
    }

    #[test]
    fn test_lower_case_called_pattern() {
        let content = "i will do it\n[called spawn({})]";
        let result = validate_response(content, &[], false, false);
        assert!(matches!(
            result,
            ValidationOutcome::Error(ValidationError::HallucinatedToolCall)
        ));
    }

    #[test]
    fn test_generate_retry_prompt_hallucinated() {
        let prompt = generate_retry_prompt(&ValidationError::HallucinatedToolCall, 1);
        assert!(prompt.contains("structured tool_call"));
        assert!(prompt.contains("1/3"));
    }

    #[test]
    fn test_generate_retry_prompt_claimed() {
        let prompt = generate_retry_prompt(&ValidationError::ClaimedButNotExecuted, 2);
        assert!(prompt.contains("future tool use"));
        assert!(prompt.contains("2/3"));
    }

    #[test]
    fn test_empty_content_passes() {
        let result = validate_response("", &[], false, false);
        assert_eq!(result, ValidationOutcome::Ok);
    }

    #[test]
    fn test_whitespace_only_passes() {
        let result = validate_response("   \n\t  ", &[], false, false);
        assert_eq!(result, ValidationOutcome::Ok);
    }

    #[test]
    fn test_response_with_code_block_passes() {
        let content = "Here's the code:\n```rust\nfn main() {}\n```";
        let result = validate_response(content, &[], false, false);
        assert_eq!(result, ValidationOutcome::Ok);
    }

    #[test]
    fn test_tool_intent_in_code_block_still_detected() {
        let content = "```\nlet me check this\n```";
        let result = validate_response(content, &[], false, false);
        assert!(
            matches!(
                result,
                ValidationOutcome::Error(ValidationError::ClaimedButNotExecuted)
            ),
            "Tool intent in code blocks should still be detected"
        );
    }

    #[test]
    fn test_i_called_colon_detected() {
        let content = "[I called: recall({\"query\": \"test\"})]";
        let result = validate_response(content, &[], false, false);
        assert_eq!(
            result,
            ValidationOutcome::Error(ValidationError::HallucinatedToolCall)
        );
    }

    #[test]
    fn test_i_called_space_detected() {
        let content = "[I called recall({\"query\": \"test\"})]";
        let result = validate_response(content, &[], false, false);
        assert_eq!(
            result,
            ValidationOutcome::Error(ValidationError::HallucinatedToolCall)
        );
    }

    #[test]
    fn test_hallucination_case_insensitive() {
        let content = "[I CALLED: tool()]";
        let result = validate_response(content, &[], false, false);
        assert_eq!(
            result,
            ValidationOutcome::Error(ValidationError::HallucinatedToolCall)
        );
    }

    #[test]
    fn test_calling_tool_detected_as_hallucination() {
        // Local models sometimes write [Calling tool: ...] instead of actual tool calls.
        let content = "[Calling tool: write_file({\"path\": \"/tmp/game.py\"})]";
        let result = validate_response(content, &[], false, false);
        assert_eq!(
            result,
            ValidationOutcome::Error(ValidationError::HallucinatedToolCall),
            "[Calling tool: ...] should be detected as hallucinated tool call"
        );
    }

    #[test]
    fn test_calling_tool_stripped_when_real_tools_present() {
        let content = "Processing... [Calling tool: read_file({\"path\": \"x\"})] Done.";
        let tool_calls = vec![make_tool_call("read_file")];
        let result = validate_response(content, &tool_calls, false, false);
        assert_eq!(result, ValidationOutcome::StripHallucination);
    }

    #[test]
    fn test_strip_when_real_tools_present() {
        let content = "Processing... [I called: recall({\"query\": \"test\"})] Done.";
        let tool_calls = vec![make_tool_call("recall")];
        let result = validate_response(content, &tool_calls, false, false);
        assert_eq!(result, ValidationOutcome::StripHallucination);
    }

    #[test]
    fn test_strip_raw_json_when_real_tools_present() {
        let content = r#"Starting clone.
{ "name": "exec", "parameters": { "command": "git clone https://github.com/dusterbloom/skybloom" } }
Done."#;
        let tool_calls = vec![make_tool_call("exec")];
        let result = validate_response(content, &tool_calls, false, false);
        assert_eq!(result, ValidationOutcome::StripHallucination);
        assert_eq!(strip_hallucinated_text(content), "Starting clone.\n\nDone.");
    }

    #[test]
    fn test_strip_hallucinated_text_helper() {
        let content = "[I called: recall({\"query\": \"test\"})] rest of text";
        let stripped = strip_hallucinated_text(content);
        assert!(!stripped.contains("[I called:"));
        assert!(stripped.contains("rest of text"));
    }

    #[test]
    fn test_strip_called_bracket_helper() {
        let content = "[Called recall({\"query\": \"test\"})] more text";
        let stripped = strip_hallucinated_text(content);
        assert!(!stripped.contains("[Called"));
        assert!(stripped.contains("more text"));
    }

    // --- TextualReplay mode: validation must be suppressed ---

    #[test]
    fn test_textual_replay_skips_hallucination_check() {
        // In textual replay mode the `[I called: ...]` pattern is legitimate history.
        // Validation must return Ok, not an error.
        let content = "[I called: read_file({\"path\":\"/tmp/test\"})]";
        let result = validate_response(content, &[], true, false);
        assert_eq!(
            result,
            ValidationOutcome::Ok,
            "Hallucination check must be suppressed in textual replay mode"
        );
    }

    #[test]
    fn test_textual_replay_skips_claimed_but_not_executed() {
        // Models in textual replay mode may write intent phrases while describing
        // what they are about to do — those should not trigger validation errors.
        let content = "Let me check that file for you.";
        let result = validate_response(content, &[], true, false);
        assert_eq!(
            result,
            ValidationOutcome::Ok,
            "ClaimedButNotExecuted check must be suppressed in textual replay mode"
        );
    }

    #[test]
    fn test_textual_replay_passes_all_patterns() {
        // All patterns that would otherwise fire must be silent in textual replay mode.
        let cases = [
            "[Called spawn({})]",
            "[I called: recall({\"q\":\"x\"})]",
            "let me read the file",
            "the file contains data",
            "i found that the answer is 42",
        ];
        for content in cases {
            let result = validate_response(content, &[], true, false);
            assert_eq!(
                result,
                ValidationOutcome::Ok,
                "Expected Ok in textual replay mode for: {}",
                content
            );
        }
    }
}
