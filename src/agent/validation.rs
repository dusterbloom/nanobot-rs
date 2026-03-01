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
    Regex::new(r"(?i)\[(?:\w+\s+)*called[\s:]").expect("hallucination regex")
});

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
    /// Validation failed — caller should retry with corrective prompt.
    Error(ValidationError),
}

// ---------------------------------------------------------------------------
// Validation Functions
// ---------------------------------------------------------------------------

pub fn validate_response(
    content: &str,
    actual_tool_calls: &[HashMap<String, Value>],
) -> ValidationOutcome {
    let lower = content.to_lowercase();

    if HALLUCINATED_CALL_RE.is_match(content) {
        if actual_tool_calls.is_empty() {
            return ValidationOutcome::Error(ValidationError::HallucinatedToolCall);
        } else {
            return ValidationOutcome::StripHallucination;
        }
    }

    if actual_tool_calls.is_empty() {
        for pattern in TOOL_INTENT_PATTERNS {
            if lower.contains(pattern) {
                return ValidationOutcome::Error(ValidationError::ClaimedButNotExecuted);
            }
        }
    }

    ValidationOutcome::Ok
}

/// Strip hallucinated tool-call text from response content.
pub fn strip_hallucinated_text(content: &str) -> String {
    HALLUCINATED_CALL_RE.replace_all(content, "").trim().to_string()
}

pub fn generate_retry_prompt(error: &ValidationError, attempt: u8) -> String {
    match error {
        ValidationError::HallucinatedToolCall => format!(
            "CRITICAL: Do NOT describe tool calls in text. \
             Use the tools array in your response structure. \
             Attempt {}/3. Actually call the tool or respond without tool intent.",
            attempt
        ),
        ValidationError::ClaimedButNotExecuted => format!(
            "CRITICAL: You expressed tool intent ('let me check/read/look') but called no tools. \
             Either ACTUALLY call the tool OR respond without implying tool use. \
             Attempt {}/3.",
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
        let result = validate_response(content, &[]);
        assert!(matches!(
            result,
            ValidationOutcome::Error(ValidationError::HallucinatedToolCall)
        ));
    }

    #[test]
    fn test_reject_claimed_but_not_executed() {
        let content = "Let me check that file for you.";
        let result = validate_response(content, &[]);
        assert!(matches!(
            result,
            ValidationOutcome::Error(ValidationError::ClaimedButNotExecuted)
        ));
    }

    #[test]
    fn test_accept_response_with_actual_tools() {
        let content = "Let me check that file for you.";
        let tool_calls = vec![make_tool_call("read_file")];
        let result = validate_response(content, &tool_calls);
        assert_eq!(result, ValidationOutcome::Ok);
    }

    #[test]
    fn test_accept_plain_response() {
        let content = "The answer is 42.";
        let result = validate_response(content, &[]);
        assert_eq!(result, ValidationOutcome::Ok);
    }

    #[test]
    fn test_detect_multiple_hallucinations() {
        let content = "[Called spawn(...)] and [Called exec(...)]";
        let result = validate_response(content, &[]);
        assert!(matches!(
            result,
            ValidationOutcome::Error(ValidationError::HallucinatedToolCall)
        ));
    }

    #[test]
    fn test_case_insensitive_patterns() {
        let content = "LET ME CHECK that for you.";
        let result = validate_response(content, &[]);
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
            let result = validate_response(content, &[]);
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
    fn test_lower_case_called_pattern() {
        let content = "i will do it\n[called spawn({})]";
        let result = validate_response(content, &[]);
        assert!(matches!(
            result,
            ValidationOutcome::Error(ValidationError::HallucinatedToolCall)
        ));
    }

    #[test]
    fn test_generate_retry_prompt_hallucinated() {
        let prompt = generate_retry_prompt(&ValidationError::HallucinatedToolCall, 1);
        assert!(prompt.contains("Do NOT describe tool calls in text"));
        assert!(prompt.contains("1/3"));
    }

    #[test]
    fn test_generate_retry_prompt_claimed() {
        let prompt = generate_retry_prompt(&ValidationError::ClaimedButNotExecuted, 2);
        assert!(prompt.contains("tool intent"));
        assert!(prompt.contains("2/3"));
    }

    #[test]
    fn test_empty_content_passes() {
        let result = validate_response("", &[]);
        assert_eq!(result, ValidationOutcome::Ok);
    }

    #[test]
    fn test_whitespace_only_passes() {
        let result = validate_response("   \n\t  ", &[]);
        assert_eq!(result, ValidationOutcome::Ok);
    }

    #[test]
    fn test_response_with_code_block_passes() {
        let content = "Here's the code:\n```rust\nfn main() {}\n```";
        let result = validate_response(content, &[]);
        assert_eq!(result, ValidationOutcome::Ok);
    }

    #[test]
    fn test_tool_intent_in_code_block_still_detected() {
        let content = "```\nlet me check this\n```";
        let result = validate_response(content, &[]);
        assert!(
            matches!(result, ValidationOutcome::Error(ValidationError::ClaimedButNotExecuted)),
            "Tool intent in code blocks should still be detected"
        );
    }

    #[test]
    fn test_i_called_colon_detected() {
        let content = "[I called: recall({\"query\": \"test\"})]";
        let result = validate_response(content, &[]);
        assert_eq!(
            result,
            ValidationOutcome::Error(ValidationError::HallucinatedToolCall)
        );
    }

    #[test]
    fn test_i_called_space_detected() {
        let content = "[I called recall({\"query\": \"test\"})]";
        let result = validate_response(content, &[]);
        assert_eq!(
            result,
            ValidationOutcome::Error(ValidationError::HallucinatedToolCall)
        );
    }

    #[test]
    fn test_hallucination_case_insensitive() {
        let content = "[I CALLED: tool()]";
        let result = validate_response(content, &[]);
        assert_eq!(
            result,
            ValidationOutcome::Error(ValidationError::HallucinatedToolCall)
        );
    }

    #[test]
    fn test_strip_when_real_tools_present() {
        let content = "Processing... [I called: recall({\"query\": \"test\"})] Done.";
        let tool_calls = vec![make_tool_call("recall")];
        let result = validate_response(content, &tool_calls);
        assert_eq!(result, ValidationOutcome::StripHallucination);
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
}
