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
    results: HashMap<String, CachedToolResult>,
    read_evidence_counts: HashMap<String, u32>,
    read_evidence_nudge_pending: bool,
    read_evidence_nudge_sent: bool,
    last_exec_result_digest: Option<String>,
    exec_same_output_streak: u32,
    exec_output_nudge_pending: bool,
    exec_output_nudge_sent: bool,
    /// True if any tool call was blocked this turn. Used to suppress
    /// ClaimedButNotExecuted validation — the model wanted to use tools
    /// but was prevented, so "let me search" text is expected, not a hallucination.
    pub had_blocked_calls: bool,
}

/// Model-visible result bytes plus the durable execution that produced them.
/// Replaying this value is a terminal disposition for a new call ID, not a
/// claim that the tool executed again.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CachedToolResult {
    pub(crate) result: String,
    pub(crate) source_tool_call_id: String,
    pub(crate) result_digest: String,
}

/// Disposition of a normalized tool call. A completed success is replayable
/// and must never be sent back to the implementation; failures and rejected
/// calls remain executable so the model can recover.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ToolGuardDecision {
    Execute,
    Replay(CachedToolResult),
    Reject(String),
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
            read_evidence_counts: HashMap::new(),
            read_evidence_nudge_pending: false,
            read_evidence_nudge_sent: false,
            last_exec_result_digest: None,
            exec_same_output_streak: 0,
            exec_output_nudge_pending: false,
            exec_output_nudge_sent: false,
            had_blocked_calls: false,
        }
    }

    /// Store a tool result keyed by (name, args) so it can be replayed on duplicates.
    #[cfg(test)]
    pub fn record_result(&mut self, name: &str, args: &HashMap<String, Value>, result: String) {
        self.record_result_with_status(name, args, result, true, "test-source", "test-digest");
    }

    /// Store only successful results while still letting writes invalidate stale reads.
    pub fn record_result_with_status(
        &mut self,
        name: &str,
        args: &HashMap<String, Value>,
        result: String,
        ok: bool,
        source_tool_call_id: &str,
        result_digest: &str,
    ) {
        if READ_CACHE_INVALIDATORS.contains(&name) {
            self.invalidate_read_cache();
        }
        if name == "exec" {
            if !ok || result.trim().is_empty() {
                self.last_exec_result_digest = None;
                self.exec_same_output_streak = 0;
                self.exec_output_nudge_pending = false;
            } else if self.last_exec_result_digest.as_deref() == Some(result_digest) {
                self.exec_same_output_streak = self.exec_same_output_streak.saturating_add(1);
            } else {
                self.last_exec_result_digest = Some(result_digest.to_owned());
                self.exec_same_output_streak = 1;
                self.exec_output_nudge_pending = false;
            }
            if self.exec_same_output_streak == 3 && !self.exec_output_nudge_sent {
                self.exec_output_nudge_pending = true;
                self.exec_output_nudge_sent = true;
            }
        }
        if !ok {
            return;
        }
        if READ_TOOLS.contains(&name) && !result.trim().is_empty() {
            let evidence_key = format!("{name}:{result_digest}");
            let repeats = self.read_evidence_counts.entry(evidence_key).or_insert(0);
            *repeats = repeats.saturating_add(1);
            if *repeats == 3 && !self.read_evidence_nudge_sent {
                self.read_evidence_nudge_pending = true;
                self.read_evidence_nudge_sent = true;
            }
        }
        let key = Self::key(name, args);
        tracing::debug!(
            tool = name,
            call_key_digest = %Self::key_digest(&key),
            source_tool_call_id,
            result_digest,
            "tool_guard_cache_recorded"
        );
        self.results.insert(
            key,
            CachedToolResult {
                result,
                source_tool_call_id: source_tool_call_id.to_owned(),
                result_digest: result_digest.to_owned(),
            },
        );
    }

    fn invalidate_read_cache(&mut self) {
        self.results.retain(|key, _| !Self::is_read_tool_key(key));
        self.seen.retain(|key, _| !Self::is_read_tool_key(key));
        self.cache_hits
            .retain(|key, _| !Self::is_read_tool_key(key));
        self.read_evidence_counts.clear();
        self.read_evidence_nudge_pending = false;
    }

    fn is_read_tool_key(key: &str) -> bool {
        let Some((tool, _)) = key.split_once(':') else {
            return false;
        };
        READ_TOOLS.contains(&tool)
    }

    fn uses_cached_result(name: &str) -> bool {
        // Tool catalogs describe current capabilities and may legitimately
        // change within a turn; replaying an old catalog would hide that.
        name != "get_tools"
    }

    /// Retrieve a previously cached result for the given call signature.
    #[cfg(test)]
    pub fn get_cached_result(&self, key: &str) -> Option<&str> {
        self.results.get(key).map(|cached| cached.result.as_str())
    }

    pub(crate) fn get_cached_result_entry(&self, key: &str) -> Option<&CachedToolResult> {
        self.results.get(key)
    }

    /// Classify a call without making duplicate success a blocking error.
    /// `allow` remains as a compatibility wrapper for older callers/tests.
    pub(crate) fn decide(
        &mut self,
        name: &str,
        args: &HashMap<String, Value>,
    ) -> ToolGuardDecision {
        let key = Self::key(name, args);
        if Self::uses_cached_result(name) {
            let cached = self.results.get(&key).cloned();
            tracing::debug!(
                tool = name,
                call_key_digest = %Self::key_digest(&key),
                hit = cached.is_some(),
                source_tool_call_id = cached.as_ref().map(|entry| entry.source_tool_call_id.as_str()),
                "tool_guard_cache_lookup"
            );
            if let Some(result) = cached {
                *self.cache_hits.entry(key).or_insert(0) += 1;
                return ToolGuardDecision::Replay(result);
            }
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
            return ToolGuardDecision::Reject(format!(
                "duplicate tool call blocked for '{}': exceeded {} identical calls in one turn",
                name, limit
            ));
        }
        ToolGuardDecision::Execute
    }

    /// How many times this (name, args) signature was blocked on the cache
    /// path this turn. 1 = first duplicate, 2+ = the model keeps replaying
    /// the same call despite the cached receipt.
    pub fn cache_hits(&self, key: &str) -> u32 {
        self.cache_hits.get(key).copied().unwrap_or(0)
    }

    /// Return the one bounded advisory raised by three independently executed
    /// reads with the same non-empty result bytes.
    pub(crate) fn take_read_evidence_nudge(&mut self) -> bool {
        std::mem::take(&mut self.read_evidence_nudge_pending)
    }

    /// Return one bounded advisory after three independently executed `exec`
    /// calls produce the same non-empty output bytes. This never changes call
    /// admission or claims that different commands are equivalent.
    pub(crate) fn take_exec_output_nudge(&mut self) -> bool {
        std::mem::take(&mut self.exec_output_nudge_pending)
    }

    pub fn key(name: &str, args: &HashMap<String, Value>) -> String {
        crate::agent::tool_runner::normalize_call_key(name, args)
    }

    fn key_digest(key: &str) -> String {
        use sha2::{Digest, Sha256};

        let mut hasher = Sha256::new();
        hasher.update(key.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    #[cfg(test)]
    pub fn allow(&mut self, name: &str, args: &HashMap<String, Value>) -> Result<(), String> {
        match self.decide(name, args) {
            ToolGuardDecision::Execute => Ok(()),
            ToolGuardDecision::Replay(_) => Err(format!(
                "duplicate tool call replayed for '{}': cached result already exists in this turn",
                name
            )),
            ToolGuardDecision::Reject(reason) => Err(reason),
        }
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
    fn cached_success_carries_durable_source_provenance() {
        let mut g = ToolGuard::new(1);
        let call_args = args(&[("command", "printf evidence"), ("working_dir", "/tmp")]);

        g.record_result_with_status(
            "exec",
            &call_args,
            "evidence".to_string(),
            true,
            "source-call",
            "raw-result-digest",
        );

        assert_eq!(
            g.decide("exec", &call_args),
            ToolGuardDecision::Replay(CachedToolResult {
                result: "evidence".to_string(),
                source_tool_call_id: "source-call".to_string(),
                result_digest: "raw-result-digest".to_string(),
            })
        );
    }

    #[test]
    fn advancing_a_read_cursor_is_a_distinct_call() {
        let mut guard = ToolGuard::new(1);
        let first = args(&[("tool_call_id", "large-result"), ("start_char", "0")]);
        let next = args(&[("tool_call_id", "large-result"), ("start_char", "1000")]);
        guard.record_result("inspect_tool_result", &first, "page one".to_string());

        assert_eq!(
            guard.decide("inspect_tool_result", &next),
            ToolGuardDecision::Execute
        );
    }

    #[test]
    fn repeated_nonempty_read_evidence_raises_one_advisory_without_blocking_variants() {
        let mut guard = ToolGuard::new(1);
        for index in 0..4 {
            let call_args = args(&[("path", &format!("/tmp/file-{index}"))]);
            assert_eq!(
                guard.decide("read_file", &call_args),
                ToolGuardDecision::Execute
            );
            guard.record_result_with_status(
                "read_file",
                &call_args,
                "same evidence".to_string(),
                true,
                &format!("read-{index}"),
                "same-raw-digest",
            );
            assert_eq!(guard.take_read_evidence_nudge(), index == 2);
        }
    }

    #[test]
    fn repeated_exec_output_is_advisory_and_changed_output_resets_the_streak() {
        let mut guard = ToolGuard::new(1);
        let record = |guard: &mut ToolGuard, index: usize, digest: &str| {
            let call_args = args(&[("command", &format!("variant-{index}"))]);
            assert_eq!(guard.decide("exec", &call_args), ToolGuardDecision::Execute);
            guard.record_result_with_status(
                "exec",
                &call_args,
                "same visible output".to_string(),
                true,
                &format!("exec-{index}"),
                digest,
            );
        };

        record(&mut guard, 0, "same-digest");
        record(&mut guard, 1, "same-digest");
        record(&mut guard, 2, "changed-digest");
        record(&mut guard, 3, "same-digest");
        record(&mut guard, 4, "same-digest");
        assert!(!guard.take_exec_output_nudge());
        record(&mut guard, 5, "same-digest");
        assert!(guard.take_exec_output_nudge());
        assert!(!guard.take_exec_output_nudge());
    }

    #[test]
    fn empty_evidence_and_changed_state_do_not_raise_recovery_advice() {
        let mut guard = ToolGuard::new(1);
        for index in 0..3 {
            let call_args = args(&[("path", &format!("/tmp/empty-{index}"))]);
            guard.record_result_with_status(
                "read_file",
                &call_args,
                String::new(),
                true,
                &format!("empty-{index}"),
                "empty-digest",
            );
        }
        assert!(!guard.take_read_evidence_nudge());

        for index in 0..2 {
            let call_args = args(&[("path", &format!("/tmp/before-{index}"))]);
            guard.record_result_with_status(
                "read_file",
                &call_args,
                "same evidence".to_string(),
                true,
                &format!("before-{index}"),
                "same-digest",
            );
        }
        let write_args = args(&[("path", "/tmp/changed"), ("content", "new")]);
        guard.record_result_with_status(
            "write_file",
            &write_args,
            "written".to_string(),
            true,
            "mutation",
            "mutation-digest",
        );
        let after = args(&[("path", "/tmp/after")]);
        guard.record_result_with_status(
            "read_file",
            &after,
            "same evidence".to_string(),
            true,
            "after",
            "same-digest",
        );
        assert!(!guard.take_read_evidence_nudge());
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
            "failed-read",
            "failed-digest",
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
    fn successful_write_is_replayed_without_execution_budget_or_block() {
        let mut g = ToolGuard::new(1);
        let write_args = args(&[("path", "/tmp/a.txt"), ("content", "new")]);
        g.record_result_with_status(
            "write_file",
            &write_args,
            "written".into(),
            true,
            "test-source",
            "test-digest",
        );
        assert_eq!(
            g.decide("write_file", &write_args),
            ToolGuardDecision::Replay(CachedToolResult {
                result: "written".into(),
                source_tool_call_id: "test-source".into(),
                result_digest: "test-digest".into(),
            })
        );
        assert_eq!(g.cache_hits(&ToolGuard::key("write_file", &write_args)), 1);
    }

    #[test]
    fn failed_and_rejected_calls_remain_retryable() {
        let mut g = ToolGuard::new(1);
        let call_args = args(&[("path", "/tmp/missing")]);
        g.record_result_with_status(
            "write_file",
            &call_args,
            "failed".into(),
            false,
            "failed-write",
            "failed-digest",
        );
        assert_eq!(
            g.decide("write_file", &call_args),
            ToolGuardDecision::Execute
        );
        assert!(matches!(
            g.decide("write_file", &call_args),
            ToolGuardDecision::Reject(_)
        ));
    }

    #[test]
    fn normalized_key_is_shared_with_router_for_reordered_arguments() {
        let first = args(&[("z", "last"), ("a", "first")]);
        let second = args(&[("a", "first"), ("z", "last")]);
        assert_eq!(
            ToolGuard::key("exec", &first),
            ToolGuard::key("exec", &second)
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
            "catalog-1",
            "catalog-digest-1",
        );
        assert!(guard.allow("get_tools", &empty).is_ok());
        guard.record_result_with_status(
            "get_tools",
            &empty,
            "Available tools: a, b".to_string(),
            true,
            "catalog-2",
            "catalog-digest-2",
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
