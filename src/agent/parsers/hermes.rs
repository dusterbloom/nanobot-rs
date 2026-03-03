//! Hermes/generic textual replay parser.
//!
//! Handles the `[I called: tool_name({...})]` format used by Nanobot's
//! TextualReplay protocol mode, which is the default fallback for models
//! that don't use a recognized vendor-specific tool call format.
//!
//! Regex patterns are ported directly from `protocol.rs`.

use once_cell::sync::Lazy;
use regex::Regex;
use serde_json::Value;

use super::registry::{ParsedToolCall, ToolCallParser};

pub struct HermesParser;

// Matches the outer `[I called: ...]` or `[Called: ...]` or `[called ...]` bracket.
// Captures the inner content (everything between `[` ... `]`).
// Mirrors TEXTUAL_CALL_OUTER_RE in protocol.rs.
static HERMES_OUTER_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\[(?:I\s+)?called[:\s]\s*(.*?)\]").expect("hermes outer regex")
});

// Matches a single `tool_name({...})` pair within the inner content.
// Mirrors TEXTUAL_CALL_ITEM_RE in protocol.rs.
static HERMES_ITEM_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(\w+)\s*\(\s*(\{[^}]*(?:\{[^}]*\}[^}]*)?\})\s*\)").expect("hermes item regex")
});

impl ToolCallParser for HermesParser {
    fn name(&self) -> &str {
        "hermes"
    }

    fn parse(&self, text: &str) -> Vec<ParsedToolCall> {
        let mut result = Vec::new();

        for outer_cap in HERMES_OUTER_RE.captures_iter(text) {
            let inner = match outer_cap.get(1) {
                Some(m) => m.as_str(),
                None => continue,
            };

            for item_cap in HERMES_ITEM_RE.captures_iter(inner) {
                let name = item_cap[1].to_string();
                let args_str = &item_cap[2];

                match serde_json::from_str::<Value>(args_str) {
                    Ok(arguments) => result.push(ParsedToolCall { name, arguments }),
                    Err(_) => {
                        // Best-effort: skip malformed JSON.
                    }
                }
            }
        }

        result
    }

    fn strip(&self, text: &str) -> String {
        HERMES_OUTER_RE.replace_all(text, "").trim().to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hermes_parse_basic() {
        let text = r#"I'll search for that. [I called: web_search({"query": "rust news"})]"#;
        let calls = HermesParser.parse(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "web_search");
        assert_eq!(calls[0].arguments["query"], "rust news");
    }

    #[test]
    fn test_hermes_parse_called_prefix() {
        let text = r#"[Called: shell_exec({"cmd": "ls"})]"#;
        let calls = HermesParser.parse(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "shell_exec");
        assert_eq!(calls[0].arguments["cmd"], "ls");
    }

    #[test]
    fn test_hermes_parse_without_colon() {
        let text = r#"[I called read_file({"path": "/tmp/bar"})]"#;
        let calls = HermesParser.parse(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "read_file");
        assert_eq!(calls[0].arguments["path"], "/tmp/bar");
    }

    #[test]
    fn test_hermes_parse_multiple_calls() {
        let text = r#"[I called: read_file({"path": "a"}), write_file({"path": "b", "content": "x"})]"#;
        let calls = HermesParser.parse(text);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "read_file");
        assert_eq!(calls[1].name, "write_file");
    }

    #[test]
    fn test_hermes_parse_empty_args() {
        let text = r#"[Called: get_time({})]"#;
        let calls = HermesParser.parse(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_time");
        assert!(calls[0].arguments.is_object());
    }

    #[test]
    fn test_hermes_parse_skips_malformed_json() {
        let text = r#"[I called: bad_tool({NOT JSON}), good_tool({"k": "v"})]"#;
        let calls = HermesParser.parse(text);
        assert!(calls.iter().any(|c| c.name == "good_tool"));
        assert!(!calls.iter().any(|c| c.name == "bad_tool"));
    }

    #[test]
    fn test_hermes_parse_no_match() {
        let text = "The answer is 42. No tool calls here.";
        assert!(HermesParser.parse(text).is_empty());
    }

    #[test]
    fn test_hermes_parse_case_insensitive() {
        let text = r#"[CALLED: read_file({"path": "x"})]"#;
        let calls = HermesParser.parse(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "read_file");
    }

    #[test]
    fn test_hermes_strip() {
        let text = r#"Some text. [I called: read_file({"path": "x"})] Done."#;
        let stripped = HermesParser.strip(text);
        assert!(!stripped.contains("[I called:"));
        assert!(stripped.contains("Some text."));
        assert!(stripped.contains("Done."));
    }

    #[test]
    fn test_hermes_strip_plain_text_unchanged() {
        let text = "The answer is 42.";
        assert_eq!(HermesParser.strip(text), text);
    }
}
