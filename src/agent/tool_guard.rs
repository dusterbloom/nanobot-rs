//! In-turn tool deduplication and loop safety guard.

use std::collections::HashMap;

use serde_json::Value;

/// Read-only tools that benefit from higher repeat limits.
const READ_TOOL_LIMIT: u32 = 5;
const READ_TOOLS: &[&str] = &[
    "read_file",
    "list_dir",
    "recall",
    "read_skill",
    "web_search",
    "web_fetch",
];

pub struct ToolGuard {
    seen: HashMap<String, u32>,
    max_same_call: u32,
    tool_limits: HashMap<String, u32>,
    results: HashMap<String, String>,
}

impl ToolGuard {
    pub fn new(max_same_call: u32) -> Self {
        let mut tool_limits = HashMap::new();
        for &tool in READ_TOOLS {
            tool_limits.insert(tool.to_string(), READ_TOOL_LIMIT);
        }
        Self {
            seen: HashMap::new(),
            max_same_call: max_same_call.max(1),
            tool_limits,
            results: HashMap::new(),
        }
    }

    /// Store a tool result keyed by (name, args) so it can be replayed on duplicates.
    pub fn record_result(&mut self, name: &str, args: &HashMap<String, Value>, result: String) {
        let key = Self::key(name, args);
        self.results.insert(key, result);
    }

    /// Retrieve a previously cached result for the given call signature.
    pub fn get_cached_result(&self, key: &str) -> Option<&str> {
        self.results.get(key).map(|s| s.as_str())
    }

    pub fn key(name: &str, args: &HashMap<String, Value>) -> String {
        let mut keys: Vec<&String> = args.keys().collect();
        keys.sort();
        let mut parts = Vec::with_capacity(keys.len());
        for k in keys {
            parts.push(format!("{}={}", k, args.get(k).cloned().unwrap_or(Value::Null)));
        }
        format!("{}|{}", name, parts.join("&"))
    }

    pub fn allow(&mut self, name: &str, args: &HashMap<String, Value>) -> Result<(), String> {
        let key = Self::key(name, args);
        let count = self.seen.entry(key).or_insert(0);
        *count += 1;
        let limit = self.tool_limits.get(name).copied().unwrap_or(self.max_same_call);
        if *count > limit {
            return Err(format!(
                "duplicate tool call blocked for '{}': exceeded {} identical calls in one turn",
                name, limit
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(pairs: &[(&str, &str)]) -> HashMap<String, Value> {
        pairs.iter().map(|(k, v)| (k.to_string(), Value::String(v.to_string()))).collect()
    }

    #[test]
    fn test_tool_guard_blocks_duplicates() {
        let mut g = ToolGuard::new(1);
        let mut args = HashMap::new();
        args.insert("command".to_string(), Value::String("ls".to_string()));
        // exec is not in the read-tool list, so it uses the default limit of 1
        assert!(g.allow("exec", &args).is_ok());
        assert!(g.allow("exec", &args).is_err());
    }

    #[test]
    fn test_tool_guard_cache_hit_after_recording() {
        let mut g = ToolGuard::new(1);
        let mut args = HashMap::new();
        args.insert("path".to_string(), Value::String("/tmp/foo".to_string()));
        let key = ToolGuard::key("read_file", &args);
        g.record_result("read_file", &args, "file contents here".to_string());
        assert_eq!(g.get_cached_result(&key), Some("file contents here"));
    }

    #[test]
    fn test_tool_guard_cache_miss_without_recording() {
        let g = ToolGuard::new(1);
        let mut args = HashMap::new();
        args.insert("path".to_string(), Value::String("/tmp/bar".to_string()));
        let key = ToolGuard::key("read_file", &args);
        assert_eq!(g.get_cached_result(&key), None);
    }

    #[test]
    fn test_read_tool_higher_limit() {
        let mut guard = ToolGuard::new(1);
        let a = args(&[("path", "/tmp/a.txt")]);
        // read_file should allow up to 5 identical calls
        for _ in 0..5 {
            assert!(guard.allow("read_file", &a).is_ok());
        }
        // 6th should be blocked
        assert!(guard.allow("read_file", &a).is_err());
    }

    #[test]
    fn test_write_tool_uses_default_limit() {
        let mut guard = ToolGuard::new(1);
        let a = args(&[("path", "/tmp/a.txt"), ("content", "hello")]);
        // First call allowed
        assert!(guard.allow("write_file", &a).is_ok());
        // Second identical call blocked at default limit
        assert!(guard.allow("write_file", &a).is_err());
    }

    #[test]
    fn test_different_args_not_blocked() {
        let mut guard = ToolGuard::new(1);
        let a1 = args(&[("path", "/tmp/a.txt")]);
        let a2 = args(&[("path", "/tmp/b.txt")]);
        assert!(guard.allow("write_file", &a1).is_ok());
        assert!(guard.allow("write_file", &a2).is_ok());
    }

    #[test]
    fn test_list_dir_higher_limit() {
        let mut guard = ToolGuard::new(1);
        let a = args(&[("path", "/tmp")]);
        for _ in 0..5 {
            assert!(guard.allow("list_dir", &a).is_ok());
        }
        assert!(guard.allow("list_dir", &a).is_err());
    }

    #[test]
    fn test_recall_higher_limit() {
        let mut guard = ToolGuard::new(1);
        let a = args(&[("query", "test")]);
        for _ in 0..5 {
            assert!(guard.allow("recall", &a).is_ok());
        }
        assert!(guard.allow("recall", &a).is_err());
    }

    #[test]
    fn test_default_limit_raised_to_three() {
        let mut guard = ToolGuard::new(3);
        let a = args(&[("command", "ls")]);
        for _ in 0..3 {
            assert!(guard.allow("exec", &a).is_ok());
        }
        assert!(guard.allow("exec", &a).is_err());
    }
}

