//! RLM ContextStore: symbolic handles to tool outputs.
//!
//! Instead of truncating large tool results, the delegation model receives
//! metadata (variable name, length, preview) and can inspect the full data
//! via micro-tools: `ctx_slice`, `ctx_grep`, `ctx_length`.

use std::collections::HashMap;
use std::sync::Arc;

use serde_json::{json, Value};

use crate::providers::base::LLMProvider;

/// Names of micro-tools that operate on the ContextStore.
pub const MICRO_TOOLS: &[&str] = &[
    "ctx_slice",
    "ctx_grep",
    "ctx_length",
    "ctx_summarize",
    "mem_store",
    "mem_recall",
    "set_phase",
];

/// Stores full tool outputs as named variables for micro-tool inspection.
pub struct ContextStore {
    variables: HashMap<String, String>,
    counter: usize,
    memory: HashMap<String, String>,
}

impl ContextStore {
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
            counter: 0,
            memory: HashMap::new(),
        }
    }

    /// Store data as a named variable. Returns `(name, metadata)`.
    ///
    /// Metadata is a compact string: `"Variable 'output_0': 45230 chars. Preview: <first 150>..."`
    pub fn store(&mut self, data: String) -> (String, String) {
        let name = format!("output_{}", self.counter);
        self.counter += 1;

        let preview: String = data.chars().take(150).collect();
        let ellipsis = if data.chars().count() > 150 {
            "..."
        } else {
            ""
        };
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
        let result: String = data
            .chars()
            .skip(clamped_start)
            .take(effective_end - clamped_start)
            .collect();
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

    /// Store a key-value pair in working memory.
    pub fn mem_store(&mut self, key: &str, value: String) -> String {
        self.memory.insert(key.to_string(), value);
        format!("Stored key '{}'.", key)
    }

    /// Recall a value from working memory by key.
    pub fn mem_recall(&self, key: &str) -> String {
        match self.memory.get(key) {
            Some(v) => v.clone(),
            None => format!("Key '{}' not found in memory.", key),
        }
    }

    /// List all keys in working memory.
    pub fn mem_keys(&self) -> Vec<&str> {
        self.memory.keys().map(|k| k.as_str()).collect()
    }

    /// Get all memory key-value pairs for state reconstruction.
    pub fn mem_entries(&self) -> &HashMap<String, String> {
        &self.memory
    }

    /// Get all variable metadata (name, length, preview) without full content.
    pub fn variable_metadata(&self) -> Vec<(String, usize, String)> {
        let mut entries: Vec<_> = self
            .variables
            .iter()
            .map(|(name, data)| {
                let preview: String = data.chars().take(100).collect();
                let ellipsis = if data.chars().count() > 100 {
                    "..."
                } else {
                    ""
                };
                (
                    name.clone(),
                    data.chars().count(),
                    format!("{}{}", preview, ellipsis),
                )
            })
            .collect();
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        entries
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
        json!({
            "type": "function",
            "function": {
                "name": "ctx_summarize",
                "description": "Summarize a stored variable using a sub-model. Returns the summary text.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "variable": {"type": "string", "description": "Variable name (e.g. 'output_0')"},
                        "instruction": {"type": "string", "description": "What to extract or summarize (e.g. 'List all function names')"}
                    },
                    "required": ["variable", "instruction"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "mem_store",
                "description": "Store a key-value pair in working memory. Persists across tool calls in this delegation session.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "key": {"type": "string", "description": "Memory key (e.g. 'findings', 'urls')"},
                        "value": {"type": "string", "description": "Value to store (overwrites existing)"}
                    },
                    "required": ["key", "value"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "mem_recall",
                "description": "Recall a value from working memory by key. Returns error message if not found.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "key": {"type": "string", "description": "Memory key to recall"}
                    },
                    "required": ["key"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "set_phase",
                "description": "Explicitly set the current task phase. This adjusts which tools are available. Valid phases: idle, understanding, planning, file_editing, code_execution, web_research, communication, reflection.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "phase": {"type": "string", "description": "Phase name (e.g. 'file_editing', 'web_research')"}
                    },
                    "required": ["phase"]
                }
            }
        }),
    ]
}

/// Execute a micro-tool against the ContextStore.
pub fn execute_micro_tool(
    store: &mut ContextStore,
    name: &str,
    args: &HashMap<String, Value>,
) -> String {
    let variable = args.get("variable").and_then(|v| v.as_str()).unwrap_or("");

    match name {
        "ctx_slice" => {
            let start = args.get("start").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            let end = args.get("end").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            match store.slice(variable, start, end) {
                Some(s) => s,
                None => format!("Error: variable '{}' not found.", variable),
            }
        }
        "ctx_grep" => {
            let pattern = args.get("pattern").and_then(|v| v.as_str()).unwrap_or("");
            match store.grep(variable, pattern) {
                Some(s) => s,
                None => format!("Error: variable '{}' not found.", variable),
            }
        }
        "ctx_length" => match store.length(variable) {
            Some(len) => len.to_string(),
            None => format!("Error: variable '{}' not found.", variable),
        },
        "mem_store" => {
            let key = args.get("key").and_then(|v| v.as_str()).unwrap_or("");
            let value = args
                .get("value")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            if key.is_empty() {
                "Error: 'key' parameter is required.".to_string()
            } else {
                store.mem_store(key, value)
            }
        }
        "mem_recall" => {
            let key = args.get("key").and_then(|v| v.as_str()).unwrap_or("");
            store.mem_recall(key)
        }
        "set_phase" => {
            let phase_str = args.get("phase").and_then(|v| v.as_str()).unwrap_or("");
            match crate::agent::system_state::TaskPhase::from_str_loose(phase_str) {
                Some(phase) => format!("Phase set to: {}", phase),
                None => format!("Error: unknown phase '{}'. Valid: idle, understanding, planning, file_editing, code_execution, web_research, communication, reflection.", phase_str),
            }
        }
        _ => format!("Error: unknown micro-tool '{}'.", name),
    }
}

/// Maximum recursion depth for ctx_summarize.
pub const MAX_SUMMARIZE_DEPTH: u32 = 2;

/// Sync micro-tool definitions (no ctx_summarize) for use in sub-loops.
fn sync_micro_tool_definitions() -> Vec<Value> {
    micro_tool_definitions()
        .into_iter()
        .filter(|d| d.pointer("/function/name").and_then(|v| v.as_str()) != Some("ctx_summarize"))
        .collect()
}

/// Execute ctx_summarize: run a mini summarization loop over a stored variable.
///
/// Creates a sub-ContextStore with the variable's content, gives the model
/// only sync micro-tools (slice/grep/length), and returns its text summary.
pub async fn execute_ctx_summarize(
    store: &ContextStore,
    variable: &str,
    instruction: &str,
    provider: &Arc<dyn LLMProvider>,
    model: &str,
    depth: u32,
    max_tokens: u32,
) -> String {
    // Depth guard: prevent infinite recursion.
    if depth >= MAX_SUMMARIZE_DEPTH {
        return format!(
            "Error: ctx_summarize depth limit reached ({}/{}). Use ctx_slice or ctx_grep instead.",
            depth, MAX_SUMMARIZE_DEPTH
        );
    }

    // Get the variable content.
    let content = match store.get(variable) {
        Some(c) => c.to_string(),
        None => return format!("Error: variable '{}' not found.", variable),
    };

    // Create a mini ContextStore with just this variable's content.
    let mut sub_store = ContextStore::new();
    let (var_name, metadata) = sub_store.store(content);

    // Build messages for the sub-loop.
    let system_msg = json!({
        "role": "system",
        "content": "You summarize data stored in variables. Use ctx_slice, ctx_grep, ctx_length to inspect the variable, then summarize."
    });
    let user_msg = json!({
        "role": "user",
        "content": format!("{}\n\n{}", instruction, metadata)
    });
    let mut messages = vec![system_msg, user_msg];

    // Only sync micro-tools (no ctx_summarize in sub-loop).
    let tool_defs = sync_micro_tool_definitions();
    let tool_defs_ref: Option<&[Value]> = if tool_defs.is_empty() {
        None
    } else {
        Some(&tool_defs)
    };

    // Mini 3-iteration loop: model can use micro-tools to inspect, then summarize.
    for _ in 0..3 {
        let response = match provider
            .chat(&messages, tool_defs_ref, Some(model), max_tokens, 0.3, None)
            .await
        {
            Ok(r) => r,
            Err(e) => return format!("Error: ctx_summarize LLM call failed: {}", e),
        };

        if response.has_tool_calls() {
            // Build assistant message with tool calls.
            let tc_json: Vec<Value> = response
                .tool_calls
                .iter()
                .enumerate()
                .map(|(i, tc)| {
                    json!({
                        "id": format!("sub{:06}", i),
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": serde_json::to_string(&tc.arguments)
                                .unwrap_or_else(|_| "{}".to_string()),
                        }
                    })
                })
                .collect();
            crate::agent::context::ContextBuilder::add_assistant_message(
                &mut messages,
                None,
                Some(&tc_json),
            );

            // Execute only sync micro-tools against the sub-store.
            for (i, tc) in response.tool_calls.iter().enumerate() {
                let id = format!("sub{:06}", i);
                if tc.name == "ctx_summarize" {
                    // Block recursive ctx_summarize in sub-loop.
                    crate::agent::context::ContextBuilder::add_tool_result(
                        &mut messages,
                        &id,
                        &tc.name,
                        "Error: ctx_summarize not available in sub-loop.",
                    );
                } else {
                    let result = execute_micro_tool(&mut sub_store, &tc.name, &tc.arguments);
                    crate::agent::context::ContextBuilder::add_tool_result(
                        &mut messages,
                        &id,
                        &tc.name,
                        &result,
                    );
                }
            }
        } else {
            // Model produced a text response — that's our summary.
            return response
                .content
                .unwrap_or_else(|| "No summary produced.".to_string());
        }
    }

    // Ran out of iterations — return what we have.
    "Error: ctx_summarize reached max iterations without producing a summary.".to_string()
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
        assert!(
            metadata.len() < 250,
            "Metadata should be compact, got {} chars",
            metadata.len()
        );
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
        assert_eq!(
            store.grep("output_0", "zzz"),
            Some("No matching lines.".to_string())
        );
    }

    #[test]
    fn test_grep_max_lines() {
        let mut store = ContextStore::new();
        let data: String = (0..50)
            .map(|i| format!("match line {}", i))
            .collect::<Vec<_>>()
            .join("\n");
        store.store(data);
        let result = store.grep("output_0", "match").unwrap();
        let line_count = result.lines().count();
        assert!(
            line_count <= 20,
            "Grep should stop after 20 matches, got {}",
            line_count
        );
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
        let result = execute_micro_tool(&mut store, "ctx_length", &args);
        assert_eq!(result, "11");

        // ctx_slice
        args.insert("start".to_string(), json!(0));
        args.insert("end".to_string(), json!(5));
        let result = execute_micro_tool(&mut store, "ctx_slice", &args);
        assert_eq!(result, "hello");

        // ctx_grep
        args.insert("pattern".to_string(), json!("world"));
        let result = execute_micro_tool(&mut store, "ctx_grep", &args);
        assert!(result.contains("world"));
    }

    #[test]
    fn test_execute_micro_tool_unknown_var() {
        let mut store = ContextStore::new();
        let mut args = HashMap::new();
        args.insert("variable".to_string(), json!("nonexistent"));
        let result = execute_micro_tool(&mut store, "ctx_length", &args);
        assert!(
            result.contains("not found"),
            "Should report missing variable: {}",
            result
        );
    }

    #[test]
    fn test_is_micro_tool() {
        assert!(is_micro_tool("ctx_slice"));
        assert!(is_micro_tool("ctx_grep"));
        assert!(is_micro_tool("ctx_length"));
        assert!(is_micro_tool("ctx_summarize"));
        assert!(!is_micro_tool("exec"));
        assert!(!is_micro_tool("read_file"));
    }

    #[test]
    fn test_ctx_summarize_definition_present() {
        let defs = micro_tool_definitions();
        let names: Vec<&str> = defs
            .iter()
            .filter_map(|d| d.pointer("/function/name").and_then(|v| v.as_str()))
            .collect();
        assert!(
            names.contains(&"ctx_summarize"),
            "micro_tool_definitions() should include ctx_summarize, got: {:?}",
            names
        );
        // Verify schema has required params
        let summarize_def = defs
            .iter()
            .find(|d| d.pointer("/function/name").and_then(|v| v.as_str()) == Some("ctx_summarize"))
            .unwrap();
        let required = summarize_def
            .pointer("/function/parameters/required")
            .and_then(|v| v.as_array())
            .unwrap();
        let req_names: Vec<&str> = required.iter().filter_map(|v| v.as_str()).collect();
        assert!(req_names.contains(&"variable"), "Should require 'variable'");
        assert!(
            req_names.contains(&"instruction"),
            "Should require 'instruction'"
        );
    }

    #[test]
    fn test_mem_store_and_recall() {
        let mut store = ContextStore::new();
        let result = store.mem_store("key1", "value1".to_string());
        assert!(result.contains("Stored"));
        assert_eq!(store.mem_recall("key1"), "value1");
    }

    #[test]
    fn test_mem_recall_missing_key() {
        let store = ContextStore::new();
        let result = store.mem_recall("nonexistent");
        assert!(result.contains("not found"));
    }

    #[test]
    fn test_mem_store_overwrites() {
        let mut store = ContextStore::new();
        store.mem_store("key1", "first".to_string());
        store.mem_store("key1", "second".to_string());
        assert_eq!(store.mem_recall("key1"), "second");
    }

    #[test]
    fn test_mem_keys() {
        let mut store = ContextStore::new();
        store.mem_store("a", "1".to_string());
        store.mem_store("b", "2".to_string());
        let mut keys = store.mem_keys();
        keys.sort();
        assert_eq!(keys, vec!["a", "b"]);
    }

    #[test]
    fn test_mem_store_via_micro_tool() {
        let mut store = ContextStore::new();
        let mut args = HashMap::new();
        args.insert("key".to_string(), json!("findings"));
        args.insert("value".to_string(), json!("Found 3 items"));
        let result = execute_micro_tool(&mut store, "mem_store", &args);
        assert!(result.contains("Stored"));

        // Recall via micro-tool
        let mut recall_args = HashMap::new();
        recall_args.insert("key".to_string(), json!("findings"));
        let recalled = execute_micro_tool(&mut store, "mem_recall", &recall_args);
        assert_eq!(recalled, "Found 3 items");
    }

    #[test]
    fn test_is_micro_tool_includes_memory() {
        assert!(is_micro_tool("mem_store"));
        assert!(is_micro_tool("mem_recall"));
    }

    #[test]
    fn test_mem_tool_definitions_present() {
        let defs = micro_tool_definitions();
        let names: Vec<&str> = defs
            .iter()
            .filter_map(|d| d.pointer("/function/name").and_then(|v| v.as_str()))
            .collect();
        assert!(
            names.contains(&"mem_store"),
            "Should include mem_store, got: {:?}",
            names
        );
        assert!(
            names.contains(&"mem_recall"),
            "Should include mem_recall, got: {:?}",
            names
        );
    }

    #[test]
    fn test_variable_metadata_returns_sorted_entries() {
        let mut store = ContextStore::new();
        store.store("short".to_string());
        store.store("x".repeat(200));

        let meta = store.variable_metadata();
        assert_eq!(meta.len(), 2);

        // Sorted by name: output_0 before output_1
        assert_eq!(meta[0].0, "output_0");
        assert_eq!(meta[0].1, 5); // "short" = 5 chars
        assert_eq!(meta[0].2, "short"); // preview, no ellipsis

        assert_eq!(meta[1].0, "output_1");
        assert_eq!(meta[1].1, 200);
        // Preview is first 100 chars + "..."
        assert!(
            meta[1].2.ends_with("..."),
            "Long preview should have ellipsis: {}",
            meta[1].2
        );
        assert_eq!(meta[1].2.len(), 103); // 100 x's + "..."
    }

    #[test]
    fn test_mem_entries_returns_all_memory() {
        let mut store = ContextStore::new();
        store.mem_store("finding_a", "value_a".to_string());
        store.mem_store("finding_b", "value_b".to_string());

        let entries = store.mem_entries();
        assert_eq!(entries.len(), 2);
        assert_eq!(
            entries.get("finding_a").map(|s| s.as_str()),
            Some("value_a")
        );
        assert_eq!(
            entries.get("finding_b").map(|s| s.as_str()),
            Some("value_b")
        );
    }
}
