//! In-turn tool deduplication and loop safety guard.

use std::collections::HashMap;

use serde_json::Value;

pub struct ToolGuard {
    seen: HashMap<String, u32>,
    max_same_call: u32,
    results: HashMap<String, String>,
}

impl ToolGuard {
    pub fn new(max_same_call: u32) -> Self {
        Self {
            seen: HashMap::new(),
            max_same_call: max_same_call.max(1),
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
        if *count > self.max_same_call {
            return Err(format!(
                "duplicate tool call blocked for '{}': exceeded {} identical calls in one turn",
                name, self.max_same_call
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_guard_blocks_duplicates() {
        let mut g = ToolGuard::new(1);
        let mut args = HashMap::new();
        args.insert("url".to_string(), Value::String("https://a".to_string()));
        assert!(g.allow("web_fetch", &args).is_ok());
        assert!(g.allow("web_fetch", &args).is_err());
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
}

