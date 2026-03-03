//! Qwen model family tool call parser.
//!
//! Handles the `<tool_call>{"name": "...", "arguments": {...}}</tool_call>`
//! format used by Qwen2.5 and later Qwen instruct models.

use once_cell::sync::Lazy;
use regex::Regex;

use super::registry::{ParsedToolCall, ToolCallParser};

pub struct QwenParser;

static QWEN_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>").expect("qwen tool_call regex")
});

impl ToolCallParser for QwenParser {
    fn name(&self) -> &str {
        "qwen"
    }

    fn parse(&self, text: &str) -> Vec<ParsedToolCall> {
        QWEN_RE
            .captures_iter(text)
            .filter_map(|cap| {
                let json_str = cap.get(1)?.as_str();
                let v: serde_json::Value = serde_json::from_str(json_str).ok()?;
                Some(ParsedToolCall {
                    name: v["name"].as_str()?.to_string(),
                    arguments: v["arguments"].clone(),
                })
            })
            .collect()
    }

    fn strip(&self, text: &str) -> String {
        QWEN_RE.replace_all(text, "").trim().to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qwen_parse_basic() {
        let text = r#"Let me search. <tool_call>{"name": "web_search", "arguments": {"query": "rust news"}}</tool_call>"#;
        let calls = QwenParser.parse(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "web_search");
        assert_eq!(calls[0].arguments["query"], "rust news");
    }

    #[test]
    fn test_qwen_parse_multiple() {
        let text = r#"<tool_call>{"name": "a", "arguments": {}}</tool_call> then <tool_call>{"name": "b", "arguments": {}}</tool_call>"#;
        assert_eq!(QwenParser.parse(text).len(), 2);
    }

    #[test]
    fn test_qwen_parse_no_match() {
        let text = "No tool calls here.";
        assert!(QwenParser.parse(text).is_empty());
    }

    #[test]
    fn test_qwen_parse_missing_arguments_field() {
        // If "arguments" key is absent, it deserializes as Value::Null which is fine
        let text = r#"<tool_call>{"name": "ping", "arguments": null}</tool_call>"#;
        let calls = QwenParser.parse(text);
        // name is present, so we get a result (arguments is null)
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "ping");
    }

    #[test]
    fn test_qwen_parse_malformed_json_skipped() {
        let text = r#"<tool_call>NOT JSON</tool_call>"#;
        assert!(QwenParser.parse(text).is_empty());
    }

    #[test]
    fn test_qwen_strip() {
        let text = r#"Hello <tool_call>{"name": "a", "arguments": {}}</tool_call> world"#;
        let stripped = QwenParser.strip(text);
        assert!(!stripped.contains("<tool_call>"));
        assert!(stripped.contains("Hello"));
        assert!(stripped.contains("world"));
    }

    #[test]
    fn test_qwen_strip_multiline() {
        let text = "Before\n<tool_call>\n{\"name\": \"x\", \"arguments\": {}}\n</tool_call>\nAfter";
        let stripped = QwenParser.strip(text);
        assert!(!stripped.contains("<tool_call>"));
        assert!(stripped.contains("Before"));
        assert!(stripped.contains("After"));
    }
}
