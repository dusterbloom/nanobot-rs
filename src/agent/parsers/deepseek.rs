//! DeepSeek model family tool call parser.
//!
//! Handles the `<functioncall>{"name": "...", "arguments": {...}}</functioncall>`
//! format used by DeepSeek models.

use once_cell::sync::Lazy;
use regex::Regex;

use super::registry::{ParsedToolCall, ToolCallParser};

pub struct DeepSeekParser;

static DEEPSEEK_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"<functioncall>\s*(\{[\s\S]*?\})\s*</functioncall>")
        .expect("deepseek functioncall regex")
});

impl ToolCallParser for DeepSeekParser {
    fn name(&self) -> &str {
        "deepseek"
    }

    fn parse(&self, text: &str) -> Vec<ParsedToolCall> {
        DEEPSEEK_RE
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
        DEEPSEEK_RE.replace_all(text, "").trim().to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deepseek_parse() {
        let text = r#"<functioncall>{"name": "web_search", "arguments": {"query": "rust"}}</functioncall>"#;
        let calls = DeepSeekParser.parse(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "web_search");
        assert_eq!(calls[0].arguments["query"], "rust");
    }

    #[test]
    fn test_deepseek_parse_multiple() {
        let text = r#"<functioncall>{"name": "a", "arguments": {"x": 1}}</functioncall> then <functioncall>{"name": "b", "arguments": {"y": 2}}</functioncall>"#;
        let calls = DeepSeekParser.parse(text);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "a");
        assert_eq!(calls[1].name, "b");
    }

    #[test]
    fn test_deepseek_parse_malformed_json_skipped() {
        let text = r#"<functioncall>NOT JSON</functioncall>"#;
        assert!(DeepSeekParser.parse(text).is_empty());
    }

    #[test]
    fn test_deepseek_parse_no_match() {
        let text = "No tool calls here.";
        assert!(DeepSeekParser.parse(text).is_empty());
    }

    #[test]
    fn test_deepseek_strip() {
        let text = r#"Before <functioncall>{"name": "x", "arguments": {}}</functioncall> after"#;
        let stripped = DeepSeekParser.strip(text);
        assert!(!stripped.contains("<functioncall>"));
        assert!(stripped.contains("Before"));
        assert!(stripped.contains("after"));
    }

    #[test]
    fn test_deepseek_strip_multiline() {
        let text =
            "Before\n<functioncall>\n{\"name\": \"x\", \"arguments\": {}}\n</functioncall>\nAfter";
        let stripped = DeepSeekParser.strip(text);
        assert!(!stripped.contains("<functioncall>"));
        assert!(stripped.contains("Before"));
        assert!(stripped.contains("After"));
    }

    #[test]
    fn test_deepseek_parse_empty_arguments() {
        let text = r#"<functioncall>{"name": "ping", "arguments": {}}</functioncall>"#;
        let calls = DeepSeekParser.parse(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "ping");
        assert!(calls[0].arguments.is_object());
    }
}
