//! In-turn tool deduplication and loop safety guard.

use std::collections::HashMap;

use serde_json::Value;

/// Read-only tools that benefit from higher repeat limits.
const READ_TOOL_LIMIT: u32 = 2;
const READ_TOOLS: &[&str] = &[
    "read_file",
    "list_dir",
    "find_files",
    "search_files",
    "recall",
    "get_skills",
];
const READ_CACHE_INVALIDATORS: &[&str] =
    &["exec", "write_file", "edit_file", "apply_patch", "remember"];

/// Web tools need even higher limits — agents frequently search/fetch many
/// different URLs or refine queries within one turn.
const WEB_TOOL_LIMIT: u32 = 6;
const WEB_TOOLS: &[&str] = &["web_search", "web_fetch"];

pub struct ToolGuard {
    seen: HashMap<String, u32>,
    /// Times each (name, args) key was blocked via the cache-replay path.
    /// Drives receipt escalation: the first duplicate replays the cached
    /// data; later ones get directives instead of re-dumping the same bytes.
    cache_hits: HashMap<String, u32>,
    max_same_call: u32,
    tool_limits: HashMap<String, u32>,
    results: HashMap<String, String>,
    /// True if any tool call was blocked this turn. Used to suppress
    /// ClaimedButNotExecuted validation — the model wanted to use tools
    /// but was prevented, so "let me search" text is expected, not a hallucination.
    pub had_blocked_calls: bool,
}

impl ToolGuard {
    pub fn new(max_same_call: u32) -> Self {
        let mut tool_limits = HashMap::new();
        for &tool in READ_TOOLS {
            tool_limits.insert(tool.to_string(), READ_TOOL_LIMIT);
        }
        for &tool in WEB_TOOLS {
            tool_limits.insert(tool.to_string(), WEB_TOOL_LIMIT);
        }
        Self {
            seen: HashMap::new(),
            cache_hits: HashMap::new(),
            max_same_call: max_same_call.max(1),
            tool_limits,
            results: HashMap::new(),
            had_blocked_calls: false,
        }
    }

    /// Store a tool result keyed by (name, args) so it can be replayed on duplicates.
    #[cfg(test)]
    pub fn record_result(&mut self, name: &str, args: &HashMap<String, Value>, result: String) {
        self.record_result_with_status(name, args, result, true);
    }

    /// Store only successful results while still letting writes invalidate stale reads.
    pub fn record_result_with_status(
        &mut self,
        name: &str,
        args: &HashMap<String, Value>,
        result: String,
        ok: bool,
    ) {
        if READ_CACHE_INVALIDATORS.contains(&name) {
            self.invalidate_read_cache();
        }
        if !ok {
            return;
        }
        let key = Self::key(name, args);
        self.results.insert(key, result);
    }

    fn invalidate_read_cache(&mut self) {
        self.results.retain(|key, _| !Self::is_read_tool_key(key));
        self.seen.retain(|key, _| !Self::is_read_tool_key(key));
        self.cache_hits.retain(|key, _| !Self::is_read_tool_key(key));
    }

    fn is_read_tool_key(key: &str) -> bool {
        let Some((tool, _)) = key.split_once('|') else {
            return false;
        };
        READ_TOOLS.contains(&tool)
    }

    fn uses_cached_result(name: &str) -> bool {
        READ_TOOLS.contains(&name) || WEB_TOOLS.contains(&name)
    }

    /// Retrieve a previously cached result for the given call signature.
    pub fn get_cached_result(&self, key: &str) -> Option<&str> {
        self.results.get(key).map(|s| s.as_str())
    }

    /// How many times this (name, args) signature was blocked on the cache
    /// path this turn. 1 = first duplicate, 2+ = the model keeps replaying
    /// the same call despite the cached receipt.
    pub fn cache_hits(&self, key: &str) -> u32 {
        self.cache_hits.get(key).copied().unwrap_or(0)
    }

    pub fn key(name: &str, args: &HashMap<String, Value>) -> String {
        let mut keys: Vec<&String> = args.keys().collect();
        keys.sort();
        let mut parts = Vec::with_capacity(keys.len());
        for k in keys {
            parts.push(format!(
                "{}={}",
                k,
                args.get(k).cloned().unwrap_or(Value::Null)
            ));
        }
        format!("{}|{}", name, parts.join("&"))
    }

    pub fn allow(&mut self, name: &str, args: &HashMap<String, Value>) -> Result<(), String> {
        let key = Self::key(name, args);
        if Self::uses_cached_result(name) && self.results.contains_key(&key) {
            self.had_blocked_calls = true;
            *self.cache_hits.entry(key).or_insert(0) += 1;
            return Err(format!(
                "duplicate tool call blocked for '{}': cached result already exists in this turn",
                name
            ));
        }
        let count = self.seen.entry(key).or_insert(0);
        *count += 1;
        let limit = self
            .tool_limits
            .get(name)
            .copied()
            .unwrap_or(self.max_same_call);
        if *count > limit {
            self.had_blocked_calls = true;
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
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), Value::String(v.to_string())))
            .collect()
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
        assert!(
            g.allow("read_file", &args).is_err(),
            "cached read-only results should be replayed, not executed again"
        );
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
    fn test_tool_guard_failed_result_does_not_cache_read() {
        let mut g = ToolGuard::new(1);
        let read_args = args(&[("path", "/tmp/missing.txt")]);
        let key = ToolGuard::key("read_file", &read_args);

        g.record_result_with_status(
            "read_file",
            &read_args,
            "Error: missing file".to_string(),
            false,
        );

        assert_eq!(g.get_cached_result(&key), None);
        assert!(
            g.allow("read_file", &read_args).is_ok(),
            "failed read result must not block a retry"
        );
    }

    #[test]
    fn test_tool_guard_cache_hit_after_web_result() {
        let mut g = ToolGuard::new(1);
        let web_args = args(&[("query", "nanobot higgs retained kv")]);
        let key = ToolGuard::key("web_search", &web_args);
        g.record_result("web_search", &web_args, "web result".to_string());

        assert_eq!(g.get_cached_result(&key), Some("web result"));
        assert!(
            g.allow("web_search", &web_args).is_err(),
            "cached web results should be replayed, not executed again"
        );
    }

    #[test]
    fn test_tool_guard_write_invalidates_cached_reads() {
        let mut g = ToolGuard::new(1);
        let read_args = args(&[("path", "/tmp/a.txt")]);
        g.record_result("read_file", &read_args, "old".to_string());

        let write_args = args(&[("path", "/tmp/a.txt"), ("content", "new")]);
        g.record_result("write_file", &write_args, "written".to_string());

        let key = ToolGuard::key("read_file", &read_args);
        assert_eq!(g.get_cached_result(&key), None);
        assert!(
            g.allow("read_file", &read_args).is_ok(),
            "read after a write should execute again"
        );
    }

    #[test]
    fn test_tool_guard_write_invalidates_cached_file_searches() {
        let mut g = ToolGuard::new(1);
        let find_args = args(&[("path", "/tmp"), ("pattern", "*.rs")]);
        let search_args = args(&[("path", "/tmp"), ("query", "needle")]);
        g.record_result("find_files", &find_args, "old names".to_string());
        g.record_result("search_files", &search_args, "old matches".to_string());

        let write_args = args(&[("path", "/tmp/a.rs"), ("content", "needle")]);
        g.record_result("write_file", &write_args, "written".to_string());

        let find_key = ToolGuard::key("find_files", &find_args);
        let search_key = ToolGuard::key("search_files", &search_args);
        assert_eq!(g.get_cached_result(&find_key), None);
        assert_eq!(g.get_cached_result(&search_key), None);
        assert!(
            g.allow("find_files", &find_args).is_ok(),
            "find_files after a write should execute fresh"
        );
        assert!(
            g.allow("search_files", &search_args).is_ok(),
            "search_files after a write should execute fresh"
        );
    }

    #[test]
    fn test_tool_guard_memory_write_invalidates_cached_recall() {
        let mut g = ToolGuard::new(1);
        let recall_args = args(&[("query", "AGI bonsai")]);
        assert!(g.allow("recall", &recall_args).is_ok());
        g.record_result("recall", &recall_args, "old memory".to_string());

        let remember_args = args(&[("fact", "AGI bonsai: updated")]);
        g.record_result("remember", &remember_args, "Remembered".to_string());

        let key = ToolGuard::key("recall", &recall_args);
        assert_eq!(g.get_cached_result(&key), None);
        assert!(
            g.allow("recall", &recall_args).is_ok(),
            "recall after a memory write should execute fresh"
        );
    }

    #[test]
    fn test_tool_guard_invalidating_write_resets_read_counter() {
        let mut g = ToolGuard::new(1);
        let read_args = args(&[("path", "/tmp/a.txt")]);
        let edit_args = args(&[("path", "/tmp/a.txt"), ("old", "a"), ("new", "b")]);

        assert!(g.allow("read_file", &read_args).is_ok());
        g.record_result("read_file", &read_args, "one".to_string());
        g.record_result("edit_file", &edit_args, "edited".to_string());

        assert!(g.allow("read_file", &read_args).is_ok());
        g.record_result("read_file", &read_args, "two".to_string());
        g.record_result("edit_file", &edit_args, "edited again".to_string());

        assert!(
            g.allow("read_file", &read_args).is_ok(),
            "read counter must reset each time writes invalidate cached reads"
        );
    }

    #[test]
    fn test_read_tool_higher_limit() {
        let mut guard = ToolGuard::new(1);
        let a = args(&[("path", "/tmp/a.txt")]);
        // read_file allows 2 identical calls (original + one re-read after modification).
        for _ in 0..2 {
            assert!(guard.allow("read_file", &a).is_ok());
        }
        // 3rd identical call is blocked (cache replay handles it).
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
        for _ in 0..2 {
            assert!(guard.allow("list_dir", &a).is_ok());
        }
        assert!(guard.allow("list_dir", &a).is_err());
    }

    #[test]
    fn test_recall_higher_limit() {
        let mut guard = ToolGuard::new(1);
        let a = args(&[("query", "test")]);
        for _ in 0..2 {
            assert!(guard.allow("recall", &a).is_ok());
        }
        assert!(guard.allow("recall", &a).is_err());
    }

    #[test]
    fn test_read_tool_different_args_unlimited() {
        let mut guard = ToolGuard::new(1);
        // Different paths should each get their own counter.
        for i in 0..10 {
            let a = args(&[("path", &format!("/tmp/file_{}.txt", i))]);
            assert!(guard.allow("read_file", &a).is_ok());
        }
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

    /// Regression for the get_tools dedup-drop: a successful non-read/non-web
    /// tool (e.g. the `get_tools` meta-tool) gets its result stored in the
    /// `results` map by `record_result_with_status`, even though
    /// `uses_cached_result()` is false for it. The router's block classifier
    /// then calls `get_cached_result()` directly and finds the stored result,
    /// classifying the blocked call as `blocked_with_result` — which makes the
    /// circuit breaker fire at rounds=1 and discard the actionable receipt.
    ///
    /// This test pins the surprising-but-load-bearing behavior: get_tools is
    /// stored AND retrievable via get_cached_result, yet allow() blocks it via
    /// the count-limit path (not the cache path). See
    /// .planning/debug/get-tools-dedup-drop.md.
    #[test]
    fn test_get_tools_result_cached_but_blocked_via_count_limit() {
        let mut guard = ToolGuard::new(2);
        let empty = HashMap::new();
        let key = ToolGuard::key("get_tools", &empty);

        // get_tools is NOT a read/web tool, so the cache-block path in allow()
        // never fires — only the count-limit path does.
        assert!(!ToolGuard::uses_cached_result("get_tools"));

        // Two identical discovery calls succeed; each result is recorded.
        assert!(guard.allow("get_tools", &empty).is_ok());
        guard.record_result_with_status(
            "get_tools",
            &empty,
            "Available tools: a, b".to_string(),
            true,
        );
        assert!(guard.allow("get_tools", &empty).is_ok());
        guard.record_result_with_status(
            "get_tools",
            &empty,
            "Available tools: a, b".to_string(),
            true,
        );

        // The cached result is retrievable even though get_tools isn't a
        // read/web tool — this is what makes the router classify the next
        // block as `blocked_with_result`.
        assert_eq!(guard.get_cached_result(&key), Some("Available tools: a, b"));

        // Third identical call is blocked by the count-limit path (limit 2).
        let err = guard
            .allow("get_tools", &empty)
            .expect_err("third identical call must be blocked");
        assert!(
            err.contains("exceeded 2 identical calls"),
            "block must come from the count-limit path: {err}"
        );
    }

    /// Duplicate receipts must escalate: hit 1 replays the cached data, hits
    /// 2+ replace the bytes with a directive + progress signal. Regression
    /// for the live recall loop where the full cached payload was re-dumped
    /// every round and the model (temp 1.0) re-rolled identical calls until
    /// the breaker hard-stopped the turn.
    #[test]
    fn test_cache_hit_counter_drives_receipt_escalation() {
        let mut g = ToolGuard::new(2);
        let recall_args = args(&[("query", "PHASEONE big")]);

        assert!(g.allow("recall", &recall_args).is_ok());
        g.record_result("recall", &recall_args, "found: ...".to_string());

        // First duplicate: hit 1 → replay data.
        assert!(g.allow("recall", &recall_args).is_err());
        assert_eq!(g.cache_hits(&ToolGuard::key("recall", &recall_args)), 1);

        // Second duplicate: hit 2 → escalation threshold.
        assert!(g.allow("recall", &recall_args).is_err());
        assert_eq!(g.cache_hits(&ToolGuard::key("recall", &recall_args)), 2);
    }

    /// The stale-result receipts must not outlive a write: invalidation
    /// clears hit counters along with the cached bytes.
    #[test]
    fn test_cache_hits_reset_on_invalidation() {
        let mut g = ToolGuard::new(2);
        let recall_args = args(&[("query", "phase")]);
        let remember_args = args(&[("fact", "updated")]);

        assert!(g.allow("recall", &recall_args).is_ok());
        g.record_result("recall", &recall_args, "old".to_string());
        assert!(g.allow("recall", &recall_args).is_err());
        assert_eq!(g.cache_hits(&ToolGuard::key("recall", &recall_args)), 1);

        g.record_result("remember", &remember_args, "Remembered".to_string());

        assert_eq!(
            g.cache_hits(&ToolGuard::key("recall", &recall_args)),
            0,
            "invalidation must reset hit counters so fresh calls replay data again"
        );
    }
}
