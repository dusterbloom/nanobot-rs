//! Llama / Nemotron model family tool call parser.
//!
//! Handles two formats:
//! - `<function=name>{"arg":"val"}</function>` (Llama 3.1+ function calling)
//! - `<|python_tag|>name.call({"arg":"val"})` (Llama python_tag variant)

use once_cell::sync::Lazy;
use regex::Regex;

use super::registry::{ParsedToolCall, ToolCallParser};

pub struct LlamaParser;

static LLAMA_FUNC_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"<function=(\w+)>\s*(\{[\s\S]*?\})\s*</function>")
        .expect("llama function tag regex")
});

static LLAMA_PYTHON_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"<\|python_tag\|>\s*(\w+)\.call\((\{[\s\S]*?\})\)").expect("llama python_tag regex")
});

impl ToolCallParser for LlamaParser {
    fn name(&self) -> &str {
        "llama"
    }

    fn parse(&self, text: &str) -> Vec<ParsedToolCall> {
        let mut calls: Vec<ParsedToolCall> = LLAMA_FUNC_RE
            .captures_iter(text)
            .filter_map(|cap| {
                let name = cap.get(1)?.as_str().to_string();
                let args: serde_json::Value = serde_json::from_str(cap.get(2)?.as_str()).ok()?;
                Some(ParsedToolCall {
                    name,
                    arguments: args,
                })
            })
            .collect();

        // Also try python_tag format
        for cap in LLAMA_PYTHON_RE.captures_iter(text) {
            if let (Some(name), Some(args_str)) = (cap.get(1), cap.get(2)) {
                if let Ok(args) = serde_json::from_str::<serde_json::Value>(args_str.as_str()) {
                    calls.push(ParsedToolCall {
                        name: name.as_str().to_string(),
                        arguments: args,
                    });
                }
            }
        }

        calls
    }

    fn strip(&self, text: &str) -> String {
        let s = LLAMA_FUNC_RE.replace_all(text, "");
        LLAMA_PYTHON_RE.replace_all(&s, "").trim().to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llama_function_format() {
        let text = r#"<function=web_search>{"query": "rust"}</function>"#;
        let calls = LlamaParser.parse(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "web_search");
        assert_eq!(calls[0].arguments["query"], "rust");
    }

    #[test]
    fn test_llama_python_tag_format() {
        let text = r#"<|python_tag|>web_search.call({"query": "rust"})"#;
        let calls = LlamaParser.parse(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "web_search");
        assert_eq!(calls[0].arguments["query"], "rust");
    }

    #[test]
    fn test_llama_both_formats_in_one_text() {
        let text =
            r#"<function=tool_a>{"x": 1}</function> and <|python_tag|>tool_b.call({"y": 2})"#;
        let calls = LlamaParser.parse(text);
        assert_eq!(calls.len(), 2);
        let names: Vec<&str> = calls.iter().map(|c| c.name.as_str()).collect();
        assert!(names.contains(&"tool_a"));
        assert!(names.contains(&"tool_b"));
    }

    #[test]
    fn test_llama_function_malformed_json_skipped() {
        let text = r#"<function=bad_tool>NOT JSON</function>"#;
        assert!(LlamaParser.parse(text).is_empty());
    }

    #[test]
    fn test_llama_no_match_returns_empty() {
        let text = "No tool calls here.";
        assert!(LlamaParser.parse(text).is_empty());
    }

    #[test]
    fn test_llama_strip_function_format() {
        let text = r#"Before <function=tool>{"x": 1}</function> after"#;
        let stripped = LlamaParser.strip(text);
        assert!(!stripped.contains("<function="));
        assert!(stripped.contains("Before"));
        assert!(stripped.contains("after"));
    }

    #[test]
    fn test_llama_strip_python_tag_format() {
        let text = r#"Before <|python_tag|>tool.call({"x": 1}) after"#;
        let stripped = LlamaParser.strip(text);
        assert!(!stripped.contains("<|python_tag|>"));
        assert!(stripped.contains("Before"));
        assert!(stripped.contains("after"));
    }

    #[test]
    fn test_llama_empty_args() {
        let text = r#"<function=get_time>{}</function>"#;
        let calls = LlamaParser.parse(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_time");
        assert!(calls[0].arguments.is_object());
    }
}
