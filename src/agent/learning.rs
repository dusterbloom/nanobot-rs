//! Lightweight tool outcome tracking for agent learning.
//!
//! Records whether each tool invocation succeeded or failed, and provides
//! a short summary for injection into the system prompt.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::Duration;

use chrono::Utc;
use serde::{Deserialize, Serialize};

/// Stores and retrieves tool outcome data.
pub struct LearningStore {
    file_path: PathBuf,
    lock_path: PathBuf,
    legacy_file_path: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LearningEntry {
    timestamp: String,
    tool_name: String,
    succeeded: bool,
    /// Brief description (first 100 chars of command/query).
    context: String,
    error: Option<String>,
    /// Provider that executed this tool call (e.g. "openrouter", "local").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    provider: Option<String>,
    /// Model used for this tool call (e.g. "qwen2-0.5b").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    model: Option<String>,
    /// Latency in milliseconds for this tool execution.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    latency_ms: Option<u64>,
}

impl LearningStore {
    /// Create a new learning store rooted at the given workspace.
    ///
    /// Data is stored at `{workspace}/memory/learnings.jsonl`.
    /// Legacy `{workspace}/memory/learnings.json` is still read for backward
    /// compatibility until the first prune/write.
    pub fn new(workspace: &Path) -> Self {
        let memory_dir = workspace.join("memory");
        let file_path = memory_dir.join("learnings.jsonl");
        let lock_path = memory_dir.join("learnings.lock");
        let legacy_file_path = memory_dir.join("learnings.json");
        Self {
            file_path,
            lock_path,
            legacy_file_path,
        }
    }

    /// Record a tool outcome.
    pub fn record(&self, tool_name: &str, succeeded: bool, context: &str, error: Option<&str>) {
        let entry = LearningEntry {
            timestamp: Utc::now().to_rfc3339(),
            tool_name: tool_name.to_string(),
            succeeded,
            context: context.chars().take(100).collect(),
            error: error.map(|e| e.chars().take(200).collect()),
            provider: None,
            model: None,
            latency_ms: None,
        };

        let _guard = match self.acquire_lock() {
            Some(g) => g,
            None => return,
        };
        self.ensure_parent_dir();
        self.append_entry_jsonl(&entry);
    }

    /// Record a tool outcome with extended provider/model/latency info.
    pub fn record_extended(
        &self,
        tool_name: &str,
        succeeded: bool,
        context: &str,
        error: Option<&str>,
        provider: Option<&str>,
        model: Option<&str>,
        latency_ms: Option<u64>,
    ) {
        let entry = LearningEntry {
            timestamp: Utc::now().to_rfc3339(),
            tool_name: tool_name.to_string(),
            succeeded,
            context: context.chars().take(100).collect(),
            error: error.map(|e| e.chars().take(200).collect()),
            provider: provider.map(|s| s.to_string()),
            model: model.map(|s| s.to_string()),
            latency_ms,
        };

        let _guard = match self.acquire_lock() {
            Some(g) => g,
            None => return,
        };
        self.ensure_parent_dir();
        self.append_entry_jsonl(&entry);
    }

    /// Get a learning context summary for injection into the system prompt.
    ///
    /// Returns a short summary of recent tool success/failure patterns.
    /// Empty string if no data or no interesting patterns.
    pub fn get_learning_context(&self) -> String {
        let entries = self.load_entries();
        if entries.is_empty() {
            return String::new();
        }

        // Only look at the last 50 entries.
        let recent: Vec<&LearningEntry> = entries.iter().rev().take(50).collect();

        // Aggregate by tool name.
        let mut tool_stats: std::collections::HashMap<&str, (u32, u32)> =
            std::collections::HashMap::new();

        for entry in &recent {
            let stat = tool_stats.entry(&entry.tool_name).or_insert((0, 0));
            if entry.succeeded {
                stat.0 += 1;
            } else {
                stat.1 += 1;
            }
        }

        let mut lines: Vec<String> = Vec::new();

        // Only report tools with failures or notable patterns.
        let mut sorted_tools: Vec<(&&str, &(u32, u32))> = tool_stats.iter().collect();
        sorted_tools.sort_by_key(|(name, _)| name.to_string());

        for (tool_name, (success, failure)) in sorted_tools {
            let total = success + failure;
            if total < 2 {
                continue; // not enough data
            }
            if *failure > 0 {
                lines.push(format!(
                    "- {}: {}/{} succeeded recently",
                    tool_name, success, total
                ));
            }
        }

        // Add recent errors (last 3 unique).
        let mut seen_errors: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut error_lines: Vec<String> = Vec::new();
        for entry in &recent {
            if let Some(ref err) = entry.error {
                if seen_errors.insert(err.clone()) && error_lines.len() < 3 {
                    error_lines.push(format!(
                        "- {} failed: {}",
                        entry.tool_name,
                        err.chars().take(80).collect::<String>()
                    ));
                }
            }
        }

        // Compute average latency per tool (only entries with latency data).
        let mut latency_stats: std::collections::HashMap<&str, (u64, u32)> =
            std::collections::HashMap::new();
        for entry in &recent {
            if let Some(ms) = entry.latency_ms {
                let stat = latency_stats.entry(&entry.tool_name).or_insert((0, 0));
                stat.0 += ms;
                stat.1 += 1;
            }
        }
        let mut slow_lines: Vec<String> = Vec::new();
        let mut sorted_latency: Vec<(&&str, &(u64, u32))> = latency_stats.iter().collect();
        sorted_latency.sort_by_key(|(name, _)| name.to_string());
        for (tool_name, (total_ms, count)) in sorted_latency {
            let avg = total_ms / (*count as u64);
            if avg > 5000 {
                slow_lines.push(format!("- {}: avg {}ms (slow)", tool_name, avg));
            }
        }

        if lines.is_empty() && error_lines.is_empty() && slow_lines.is_empty() {
            return String::new();
        }

        let mut result = String::new();
        if !lines.is_empty() {
            result.push_str("Tool success rates:\n");
            result.push_str(&lines.join("\n"));
        }
        if !error_lines.is_empty() {
            if !result.is_empty() {
                result.push_str("\n\n");
            }
            result.push_str("Recent errors:\n");
            result.push_str(&error_lines.join("\n"));
        }
        if !slow_lines.is_empty() {
            if !result.is_empty() {
                result.push_str("\n\n");
            }
            result.push_str("Slow tools:\n");
            result.push_str(&slow_lines.join("\n"));
        }

        result
    }

    /// Prune entries older than 30 days.
    pub fn prune(&self) {
        let _guard = match self.acquire_lock() {
            Some(g) => g,
            None => return,
        };

        let mut entries = self.load_entries();
        let cutoff = Utc::now() - chrono::Duration::days(30);
        let cutoff_str = cutoff.to_rfc3339();

        entries.retain(|e| e.timestamp >= cutoff_str);
        self.save_entries(&entries);
    }

    // ---------------------------------------------------------------
    // Private helpers
    // ---------------------------------------------------------------

    fn load_entries(&self) -> Vec<LearningEntry> {
        // Prefer JSONL. If missing/empty, fall back to legacy JSON array.
        let mut entries = self.load_entries_from_jsonl();
        if entries.is_empty() && self.legacy_file_path.exists() {
            entries = self.load_entries_from_legacy_json();
        }
        entries
    }

    fn save_entries(&self, entries: &[LearningEntry]) {
        // Ensure parent directory exists.
        self.ensure_parent_dir();

        // Rewrite JSONL atomically via temp file + rename.
        let tmp_path = self.file_path.with_extension("jsonl.tmp");
        if let Ok(mut f) = fs::File::create(&tmp_path) {
            for entry in entries {
                if let Ok(line) = serde_json::to_string(entry) {
                    let _ = writeln!(f, "{}", line);
                }
            }
            let _ = f.sync_all();
            let _ = fs::rename(&tmp_path, &self.file_path);
        }
    }

    fn ensure_parent_dir(&self) {
        if let Some(parent) = self.file_path.parent() {
            let _ = fs::create_dir_all(parent);
        }
    }

    fn append_entry_jsonl(&self, entry: &LearningEntry) {
        if let Ok(mut f) = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.file_path)
        {
            if let Ok(line) = serde_json::to_string(entry) {
                let _ = writeln!(f, "{}", line);
            }
        }
    }

    fn load_entries_from_jsonl(&self) -> Vec<LearningEntry> {
        let mut out: Vec<LearningEntry> = Vec::new();
        let data = match fs::read_to_string(&self.file_path) {
            Ok(d) => d,
            Err(_) => return out,
        };

        for line in data.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Ok(entry) = serde_json::from_str::<LearningEntry>(line) {
                out.push(entry);
            }
        }
        out
    }

    fn load_entries_from_legacy_json(&self) -> Vec<LearningEntry> {
        match fs::read_to_string(&self.legacy_file_path) {
            Ok(data) => serde_json::from_str(&data).unwrap_or_default(),
            Err(_) => Vec::new(),
        }
    }

    fn acquire_lock(&self) -> Option<LearningLockGuard> {
        self.ensure_parent_dir();
        const MAX_ATTEMPTS: u32 = 50;
        const RETRY_DELAY_MS: u64 = 20;
        for _ in 0..MAX_ATTEMPTS {
            match fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&self.lock_path)
            {
                Ok(_) => {
                    return Some(LearningLockGuard {
                        lock_path: self.lock_path.clone(),
                    });
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                    thread::sleep(Duration::from_millis(RETRY_DELAY_MS));
                }
                Err(_) => return None,
            }
        }
        None
    }
}

struct LearningLockGuard {
    lock_path: PathBuf,
}

impl Drop for LearningLockGuard {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.lock_path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_store() -> (TempDir, LearningStore) {
        let tmp = TempDir::new().unwrap();
        let store = LearningStore::new(tmp.path());
        (tmp, store)
    }

    #[test]
    fn test_record_and_load() {
        let (_tmp, store) = make_store();
        store.record("read_file", true, "/tmp/test.txt", None);
        store.record("exec", false, "ls /nonexistent", Some("No such file"));

        let entries = store.load_entries();
        assert_eq!(entries.len(), 2);
        assert!(entries[0].succeeded);
        assert!(!entries[1].succeeded);
        assert_eq!(entries[1].error.as_deref(), Some("No such file"));
    }

    #[test]
    fn test_get_learning_context_empty() {
        let (_tmp, store) = make_store();
        assert_eq!(store.get_learning_context(), "");
    }

    #[test]
    fn test_get_learning_context_with_failures() {
        let (_tmp, store) = make_store();

        // Record some mixed outcomes.
        for _ in 0..3 {
            store.record("exec", true, "ls", None);
        }
        for _ in 0..2 {
            store.record("exec", false, "bad_cmd", Some("command not found"));
        }

        let context = store.get_learning_context();
        assert!(context.contains("exec"));
        assert!(context.contains("3/5 succeeded"));
    }

    #[test]
    fn test_record_writes_jsonl_lines() {
        let (_tmp, store) = make_store();
        store.record("read_file", true, "/tmp/a", None);
        store.record("exec", false, "bad", Some("oops"));

        let raw = fs::read_to_string(&store.file_path).unwrap();
        let lines: Vec<&str> = raw.lines().filter(|l| !l.trim().is_empty()).collect();
        assert_eq!(lines.len(), 2);
    }

    #[test]
    fn test_get_learning_context_all_success_no_report() {
        let (_tmp, store) = make_store();

        for _ in 0..5 {
            store.record("read_file", true, "/tmp/a.txt", None);
        }

        // All successes → nothing interesting to report.
        let context = store.get_learning_context();
        assert!(context.is_empty());
    }

    #[test]
    fn test_context_truncation() {
        let (_tmp, store) = make_store();
        let long_context = "x".repeat(500);
        store.record("exec", true, &long_context, None);

        let entries = store.load_entries();
        assert_eq!(entries[0].context.len(), 100);
    }

    #[test]
    fn test_prune_removes_old() {
        let (_tmp, store) = make_store();

        // Manually insert an old entry.
        let old_entry = LearningEntry {
            timestamp: "2020-01-01T00:00:00+00:00".to_string(),
            tool_name: "exec".to_string(),
            succeeded: true,
            context: "old".to_string(),
            error: None,
            provider: None,
            model: None,
            latency_ms: None,
        };
        let new_entry = LearningEntry {
            timestamp: Utc::now().to_rfc3339(),
            tool_name: "exec".to_string(),
            succeeded: true,
            context: "new".to_string(),
            error: None,
            provider: None,
            model: None,
            latency_ms: None,
        };
        store.save_entries(&[old_entry, new_entry]);

        store.prune();
        let entries = store.load_entries();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].context, "new");
    }

    #[test]
    fn test_record_with_provider_info() {
        let (_tmp, store) = make_store();
        store.record_extended(
            "exec",
            true,
            "ls -la",
            None,
            Some("local"),
            Some("qwen2-0.5b"),
            Some(150),
        );

        let entries = store.load_entries();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].provider.as_deref(), Some("local"));
        assert_eq!(entries[0].model.as_deref(), Some("qwen2-0.5b"));
        assert_eq!(entries[0].latency_ms, Some(150));
    }

    #[test]
    fn test_learning_context_shows_latency() {
        let (_tmp, store) = make_store();
        // Record several slow entries (>5s average).
        for _ in 0..3 {
            store.record_extended(
                "web_fetch",
                true,
                "fetch page",
                None,
                None,
                None,
                Some(8000),
            );
        }

        let context = store.get_learning_context();
        assert!(context.contains("Slow tools:"));
        assert!(context.contains("web_fetch"));
        assert!(context.contains("avg 8000ms"));
    }

    #[test]
    fn test_learning_context_no_slow_warning_for_fast_tools() {
        let (_tmp, store) = make_store();
        for _ in 0..3 {
            store.record_extended("read_file", true, "/tmp/a", None, None, None, Some(50));
        }

        let context = store.get_learning_context();
        assert!(!context.contains("Slow tools:"));
    }

    #[test]
    fn test_load_entries_falls_back_to_legacy_json() {
        let (_tmp, store) = make_store();
        let legacy = vec![LearningEntry {
            timestamp: Utc::now().to_rfc3339(),
            tool_name: "exec".to_string(),
            succeeded: false,
            context: "legacy".to_string(),
            error: Some("legacy error".to_string()),
            provider: None,
            model: None,
            latency_ms: None,
        }];
        let legacy_json = serde_json::to_string(&legacy).unwrap();
        if let Some(parent) = store.legacy_file_path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(&store.legacy_file_path, legacy_json).unwrap();

        let entries = store.load_entries();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].context, "legacy");
    }
}
