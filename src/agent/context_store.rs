//! RLM ContextStore: symbolic handles to tool outputs.
//!
//! Instead of truncating large tool results, the delegation model receives
//! metadata (variable name, length, preview) and can inspect the full data
//! via micro-tools: `ctx_slice`, `ctx_grep`, `ctx_length`.

use std::collections::HashMap;

use serde_json::{json, Value};

/// Names of micro-tools that operate on the ContextStore.
pub const MICRO_TOOLS: &[&str] = &["ctx_slice", "ctx_grep", "ctx_length"];

/// Stores full tool outputs as named variables for micro-tool inspection.
pub struct ContextStore {
    variables: HashMap<String, String>,
    counter: usize,
}

impl ContextStore {
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
            counter: 0,
        }
    }

    /// Store data as a named variable. Returns `(name, metadata)`.
    ///
    /// Metadata is a compact string: `"Variable 'output_0': 45230 chars. Preview: <first 150>..."`
    pub fn store(&mut self, data: String) -> (String, String) {
        let name = format!("output_{}", self.counter);
        self.counter += 1;

        let preview: String = data.chars().take(150).collect();
        let ellipsis = if data.chars().count() > 150 { "..." } else { "" };
        let metadata = format!(
            "Variable '{}': {} chars. Preview: {}{}",
            name,
            data.chars().count(),
            preview,
            ellipsis
        );

        self.variables.insert(name.clone(), data);
        (name, metadata)
    }

    /// Get the full content of a variable.
    pub fn get(&self, name: &str) -> Option<&str> {
        self.variables.get(name).map(|s| s.as_str())
    }

    /// Char-based substring, capped at 2000 chars.
    pub fn slice(&self, name: &str, start: usize, end: usize) -> Option<String> {
        let data = self.variables.get(name)?;
        let char_count = data.chars().count();
        let clamped_start = start.min(char_count);
        let clamped_end = end.min(char_count).max(clamped_start);
        let max_len = 2000;
        let effective_end = clamped_end.min(clamped_start + max_len);
        let result: String = data.chars().skip(clamped_start).take(effective_end - clamped_start).collect();
        Some(result)
    }

    /// Matching lines with line numbers, max 20 lines / 2000 chars.
    pub fn grep(&self, name: &str, pattern: &str) -> Option<String> {
        let data = self.variables.get(name)?;
        let pattern_lower = pattern.to_lowercase();
        let mut matches = Vec::new();
        let mut total_chars = 0;

        for (i, line) in data.lines().enumerate() {
            if line.to_lowercase().contains(&pattern_lower) {
                let entry = format!("{}:{}", i + 1, line);
                total_chars += entry.len() + 1; // +1 for newline
                if matches.len() >= 20 || total_chars > 2000 {
                    break;
                }
                matches.push(entry);
            }
        }

        if matches.is_empty() {
            Some("No matching lines.".to_string())
        } else {
            Some(matches.join("\n"))
        }
    }

    /// Char count of a variable.
    pub fn length(&self, name: &str) -> Option<usize> {
        self.variables.get(name).map(|s| s.chars().count())
    }
}

/// Check if a tool name is a micro-tool.
pub fn is_micro_tool(name: &str) -> bool {
    MICRO_TOOLS.contains(&name)
}

/// JSON Schema definitions for the micro-tools.
pub fn micro_tool_definitions() -> Vec<Value> {
    vec![
        json!({
            "type": "function",
            "function": {
                "name": "ctx_slice",
                "description": "Read a character range from a stored variable. Returns up to 2000 chars.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "variable": {"type": "string", "description": "Variable name (e.g. 'output_0')"},
                        "start": {"type": "integer", "description": "Start character index (0-based)"},
                        "end": {"type": "integer", "description": "End character index (exclusive)"}
                    },
                    "required": ["variable", "start", "end"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "ctx_grep",
                "description": "Search for a pattern in a stored variable. Returns matching lines with line numbers (max 20 lines).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "variable": {"type": "string", "description": "Variable name (e.g. 'output_0')"},
                        "pattern": {"type": "string", "description": "Text pattern to search for (case-insensitive)"}
                    },
                    "required": ["variable", "pattern"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "ctx_length",
                "description": "Get the character count of a stored variable.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "variable": {"type": "string", "description": "Variable name (e.g. 'output_0')"}
                    },
                    "required": ["variable"]
                }
            }
        }),
    ]
}

/// Execute a micro-tool against the ContextStore.
pub fn execute_micro_tool(
    store: &ContextStore,
    name: &str,
    args: &HashMap<String, Value>,
) -> String {
    let variable = args
        .get("variable")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    match name {
        "ctx_slice" => {
            let start = args
                .get("start")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as usize;
            let end = args
                .get("end")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as usize;
            match store.slice(variable, start, end) {
                Some(s) => s,
                None => format!("Error: variable '{}' not found.", variable),
            }
        }
        "ctx_grep" => {
            let pattern = args
                .get("pattern")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            match store.grep(variable, pattern) {
                Some(s) => s,
                None => format!("Error: variable '{}' not found.", variable),
            }
        }
        "ctx_length" => match store.length(variable) {
            Some(len) => len.to_string(),
            None => format!("Error: variable '{}' not found.", variable),
        },
        _ => format!("Error: unknown micro-tool '{}'.", name),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_store_auto_naming() {
        let mut store = ContextStore::new();
        let (name0, _) = store.store("hello".to_string());
        let (name1, _) = store.store("world".to_string());
        assert_eq!(name0, "output_0");
        assert_eq!(name1, "output_1");
    }

    #[test]
    fn test_store_metadata_compact() {
        let mut store = ContextStore::new();
        let data = "x".repeat(45230);
        let (_, metadata) = store.store(data);
        assert!(metadata.len() < 250, "Metadata should be compact, got {} chars", metadata.len());
        assert!(metadata.contains("output_0"));
        assert!(metadata.contains("45230 chars"));
        assert!(metadata.contains("Preview:"));
    }

    #[test]
    fn test_get_existing() {
        let mut store = ContextStore::new();
        store.store("hello world".to_string());
        assert_eq!(store.get("output_0"), Some("hello world"));
    }

    #[test]
    fn test_get_missing() {
        let store = ContextStore::new();
        assert_eq!(store.get("nonexistent"), None);
    }

    #[test]
    fn test_slice_basic() {
        let mut store = ContextStore::new();
        store.store("hello world".to_string());
        assert_eq!(store.slice("output_0", 0, 5), Some("hello".to_string()));
        assert_eq!(store.slice("output_0", 6, 11), Some("world".to_string()));
    }

    #[test]
    fn test_slice_clamping() {
        let mut store = ContextStore::new();
        store.store("hello".to_string());
        // Out-of-bounds end should be clamped
        assert_eq!(store.slice("output_0", 0, 100), Some("hello".to_string()));
        // Start beyond end
        assert_eq!(store.slice("output_0", 100, 200), Some("".to_string()));
    }

    #[test]
    fn test_slice_max_2000() {
        let mut store = ContextStore::new();
        store.store("x".repeat(5000));
        let result = store.slice("output_0", 0, 5000).unwrap();
        assert_eq!(result.len(), 2000, "Slice should be capped at 2000 chars");
    }

    #[test]
    fn test_grep_matching() {
        let mut store = ContextStore::new();
        store.store("line one\nline two\nline three\nfoo bar".to_string());
        let result = store.grep("output_0", "line").unwrap();
        assert!(result.contains("1:line one"));
        assert!(result.contains("2:line two"));
        assert!(result.contains("3:line three"));
        assert!(!result.contains("foo bar"));
    }

    #[test]
    fn test_grep_no_match() {
        let mut store = ContextStore::new();
        store.store("hello world".to_string());
        assert_eq!(store.grep("output_0", "zzz"), Some("No matching lines.".to_string()));
    }

    #[test]
    fn test_grep_max_lines() {
        let mut store = ContextStore::new();
        let data: String = (0..50).map(|i| format!("match line {}", i)).collect::<Vec<_>>().join("\n");
        store.store(data);
        let result = store.grep("output_0", "match").unwrap();
        let line_count = result.lines().count();
        assert!(line_count <= 20, "Grep should stop after 20 matches, got {}", line_count);
    }

    #[test]
    fn test_length_returns_chars() {
        let mut store = ContextStore::new();
        // Unicode: each emoji is 1 char but multiple bytes
        // "héllo 🌍" = h é l l o <space> 🌍 = 7 chars
        store.store("héllo 🌍".to_string());
        let len = store.length("output_0").unwrap();
        assert_eq!(len, 7, "Length should count chars, not bytes");
    }

    #[test]
    fn test_execute_micro_tool_dispatch() {
        let mut store = ContextStore::new();
        store.store("hello world".to_string());

        // ctx_length
        let mut args = HashMap::new();
        args.insert("variable".to_string(), json!("output_0"));
        let result = execute_micro_tool(&store, "ctx_length", &args);
        assert_eq!(result, "11");

        // ctx_slice
        args.insert("start".to_string(), json!(0));
        args.insert("end".to_string(), json!(5));
        let result = execute_micro_tool(&store, "ctx_slice", &args);
        assert_eq!(result, "hello");

        // ctx_grep
        args.insert("pattern".to_string(), json!("world"));
        let result = execute_micro_tool(&store, "ctx_grep", &args);
        assert!(result.contains("world"));
    }

    #[test]
    fn test_execute_micro_tool_unknown_var() {
        let store = ContextStore::new();
        let mut args = HashMap::new();
        args.insert("variable".to_string(), json!("nonexistent"));
        let result = execute_micro_tool(&store, "ctx_length", &args);
        assert!(result.contains("not found"), "Should report missing variable: {}", result);
    }

    #[test]
    fn test_is_micro_tool() {
        assert!(is_micro_tool("ctx_slice"));
        assert!(is_micro_tool("ctx_grep"));
        assert!(is_micro_tool("ctx_length"));
        assert!(!is_micro_tool("exec"));
        assert!(!is_micro_tool("read_file"));
    }
}
