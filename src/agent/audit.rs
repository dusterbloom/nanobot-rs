#![allow(dead_code)]
//! Immutable audit log for tool call provenance.
//!
//! Records every tool invocation with arguments, results, timing, and a
//! SHA-256 hash chain so tampering is detectable.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Duration;

use chrono::Utc;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Maximum size of result_data stored per entry (8 KB).
const MAX_RESULT_SIZE: usize = 8192;

/// A single audit log entry recording one tool invocation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    /// Monotonically increasing sequence number within the session.
    pub seq: u64,
    /// ISO 8601 timestamp.
    pub timestamp: String,
    /// Name of the tool that was called.
    pub tool_name: String,
    /// Unique ID for this tool call (from the LLM).
    pub tool_call_id: String,
    /// Raw arguments JSON.
    pub arguments: serde_json::Value,
    /// Tool output, truncated to MAX_RESULT_SIZE.
    pub result_data: String,
    /// Whether the tool execution succeeded.
    pub result_ok: bool,
    /// Wall-clock duration of the tool execution.
    pub duration_ms: u64,
    /// Who executed the tool: "inline" or "tool_runner:{model}".
    pub executor: String,
    /// SHA-256 hex digest of this entry.
    pub hash: String,
    /// Hash of the previous entry (empty string for the first entry).
    pub prev_hash: String,
}

/// Append-only audit log with hash chain integrity.
pub struct AuditLog {
    file_path: PathBuf,
    lock_path: PathBuf,
    last_hash: Mutex<String>,
    seq_counter: AtomicU64,
}

impl AuditLog {
    /// Create a new audit log for the given workspace and session.
    ///
    /// Storage: `{workspace}/memory/audit/{session_key}.jsonl`
    pub fn new(workspace: &Path, session_key: &str) -> Self {
        let audit_dir = workspace.join("memory").join("audit");
        let _ = fs::create_dir_all(&audit_dir);
        // Sanitize session_key for use as filename
        let safe_key: String = session_key
            .chars()
            .map(|c| {
                if c.is_alphanumeric() || c == '-' || c == '_' {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        let file_path = audit_dir.join(format!("{}.jsonl", safe_key));
        let lock_path = audit_dir.join(format!("{}.lock", safe_key));

        // Resume sequence from existing file.
        let (last_hash, seq) = Self::read_last_entry(&file_path);

        Self {
            file_path,
            lock_path,
            last_hash: Mutex::new(last_hash),
            seq_counter: AtomicU64::new(seq),
        }
    }

    /// Record a tool invocation in the audit log.
    ///
    /// The entire seq-allocation → hash-computation → file-write → state-update
    /// sequence is serialized under the file lock to prevent race conditions
    /// both within a process (multiple threads) and across processes (parent +
    /// tool_runner subagent sharing the same JSONL file).
    pub fn record(
        &self,
        tool_name: &str,
        tool_call_id: &str,
        arguments: &serde_json::Value,
        result_data: &str,
        result_ok: bool,
        duration_ms: u64,
        executor: &str,
    ) {
        let timestamp = Utc::now().to_rfc3339();

        // Truncate result_data to MAX_RESULT_SIZE.
        let truncated_result: String = if result_data.len() > MAX_RESULT_SIZE {
            let mut s: String = result_data.chars().take(MAX_RESULT_SIZE).collect();
            s.push_str("...[truncated]");
            s
        } else {
            result_data.to_string()
        };

        let args_json = serde_json::to_string(arguments).unwrap_or_else(|_| "{}".to_string());

        // Acquire file lock FIRST — serializes across processes (parent + subagents).
        let _guard = match self.acquire_lock() {
            Some(g) => g,
            None => {
                tracing::warn!("Failed to acquire audit log lock, skipping entry");
                return;
            }
        };

        // Re-read last entry from file under lock to get authoritative seq + prev_hash.
        // This handles the cross-process case: another AuditLog instance may have
        // appended entries since we initialized our in-memory state.
        let (file_prev_hash, file_seq) = Self::read_last_entry(&self.file_path);

        // Use the file's authoritative state, not our potentially-stale in-memory state.
        let seq = file_seq;
        let prev_hash = file_prev_hash;

        let hash = Self::compute_hash(
            &prev_hash,
            seq,
            tool_name,
            tool_call_id,
            &args_json,
            &truncated_result,
            &timestamp,
        );

        let entry = AuditEntry {
            seq,
            timestamp,
            tool_name: tool_name.to_string(),
            tool_call_id: tool_call_id.to_string(),
            arguments: arguments.clone(),
            result_data: truncated_result,
            result_ok,
            duration_ms,
            executor: executor.to_string(),
            hash: hash.clone(),
            prev_hash,
        };

        if let Ok(mut f) = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.file_path)
        {
            if let Ok(line) = serde_json::to_string(&entry) {
                let _ = writeln!(f, "{}", line);
            }
        }

        // Update in-memory state for next call from this instance.
        *self.last_hash.lock().unwrap_or_else(|e| e.into_inner()) = hash;
        self.seq_counter.store(seq + 1, Ordering::SeqCst);
    }

    /// Load all entries from the audit log.
    pub fn get_entries(&self) -> Vec<AuditEntry> {
        let data = match fs::read_to_string(&self.file_path) {
            Ok(d) => d,
            Err(_) => return Vec::new(),
        };

        data.lines()
            .filter(|l| !l.trim().is_empty())
            .filter_map(|l| serde_json::from_str::<AuditEntry>(l).ok())
            .collect()
    }

    /// Verify the hash chain integrity of the audit log.
    ///
    /// Returns `Ok(entry_count)` if valid, `Err(message)` if tampered.
    pub fn verify_chain(&self) -> Result<usize, String> {
        let entries = self.get_entries();
        let mut expected_prev = String::new();

        for (i, entry) in entries.iter().enumerate() {
            // Check prev_hash linkage.
            if entry.prev_hash != expected_prev {
                return Err(format!(
                    "Entry {} prev_hash mismatch: expected '{}', got '{}'",
                    i, expected_prev, entry.prev_hash
                ));
            }

            // Recompute hash and verify.
            let args_json =
                serde_json::to_string(&entry.arguments).unwrap_or_else(|_| "{}".to_string());
            let computed = Self::compute_hash(
                &entry.prev_hash,
                entry.seq,
                &entry.tool_name,
                &entry.tool_call_id,
                &args_json,
                &entry.result_data,
                &entry.timestamp,
            );

            if entry.hash != computed {
                return Err(format!(
                    "Entry {} hash mismatch: expected '{}', got '{}'",
                    i, computed, entry.hash
                ));
            }

            expected_prev = entry.hash.clone();
        }

        Ok(entries.len())
    }

    /// Search entries for one whose result_data contains the given substring.
    pub fn find_matching_result(&self, substring: &str) -> Option<AuditEntry> {
        let entries = self.get_entries();
        entries
            .into_iter()
            .rev()
            .find(|e| e.result_data.contains(substring))
    }

    // --- Private helpers ---

    fn compute_hash(
        prev_hash: &str,
        seq: u64,
        tool_name: &str,
        tool_call_id: &str,
        args_json: &str,
        result_data: &str,
        timestamp: &str,
    ) -> String {
        let mut hasher = Sha256::new();
        hasher.update(format!(
            "{}|{}|{}|{}|{}|{}|{}",
            prev_hash, seq, tool_name, tool_call_id, args_json, result_data, timestamp
        ));
        format!("{:x}", hasher.finalize())
    }

    fn read_last_entry(file_path: &Path) -> (String, u64) {
        let data = match fs::read_to_string(file_path) {
            Ok(d) => d,
            Err(_) => return (String::new(), 0),
        };

        let last_entry = data
            .lines()
            .rev()
            .find(|l| !l.trim().is_empty())
            .and_then(|l| serde_json::from_str::<AuditEntry>(l).ok());

        match last_entry {
            Some(entry) => (entry.hash, entry.seq + 1),
            None => (String::new(), 0),
        }
    }

    fn acquire_lock(&self) -> Option<AuditLockGuard> {
        use backon::BlockingRetryable;

        let lock_path = &self.lock_path;
        let result = (|| {
            fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(lock_path)
                .map(|_| AuditLockGuard {
                    lock_path: lock_path.clone(),
                })
        })
        .retry(
            backon::ConstantBuilder::default()
                .with_delay(Duration::from_millis(20))
                .with_max_times(50),
        )
        .when(|e: &std::io::Error| e.kind() == std::io::ErrorKind::AlreadyExists)
        .call();

        result.ok()
    }
}

struct AuditLockGuard {
    lock_path: PathBuf,
}

impl Drop for AuditLockGuard {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.lock_path);
    }
}

/// Per-turn audit summary for the structured turn log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnSummary {
    pub turn: u64,
    pub timestamp: String,
    pub context_tokens: usize,
    pub message_count: usize,
    pub tools_called: Vec<TurnToolEntry>,
    pub working_memory_tokens: usize,
}

/// Tool call summary within a turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnToolEntry {
    pub name: String,
    pub id: String,
    pub ok: bool,
    pub duration_ms: u64,
    pub result_chars: usize,
}

/// Write a per-turn summary to the session audit JSONL.
///
/// Stored at `{workspace}/memory/audit/{session_key}.turns.jsonl`
pub fn write_turn_summary(workspace: &Path, session_key: &str, summary: &TurnSummary) {
    let audit_dir = workspace.join("memory").join("audit");
    let _ = fs::create_dir_all(&audit_dir);
    let safe_key: String = session_key
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let path = audit_dir.join(format!("{}.turns.jsonl", safe_key));
    if let Ok(json) = serde_json::to_string(summary) {
        if let Ok(mut f) = fs::OpenOptions::new().create(true).append(true).open(&path) {
            let _ = writeln!(f, "{}", json);
        }
    }
}

/// Events emitted during tool execution for REPL display.
#[derive(Debug, Clone)]
pub enum ToolEvent {
    /// Tool execution is starting.
    CallStart {
        tool_name: String,
        tool_call_id: String,
        arguments_preview: String,
    },
    /// Tool execution has completed.
    CallEnd {
        tool_name: String,
        tool_call_id: String,
        /// Full tool output (consumers truncate for display).
        result_data: String,
        ok: bool,
        duration_ms: u64,
    },
    /// Periodic progress update during tool execution.
    Progress {
        tool_name: String,
        tool_call_id: String,
        /// Milliseconds since tool execution started.
        elapsed_ms: u64,
        /// Last line of output (if streaming), or None for non-streaming tools.
        output_preview: Option<String>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::TempDir;

    fn make_audit() -> (TempDir, AuditLog) {
        let tmp = TempDir::new().unwrap();
        let log = AuditLog::new(tmp.path(), "test-session");
        (tmp, log)
    }

    #[test]
    fn test_record_and_load() {
        let (_tmp, log) = make_audit();
        log.record(
            "read_file",
            "call_1",
            &json!({"path": "/tmp/test.txt"}),
            "file contents here",
            true,
            12,
            "inline",
        );
        log.record(
            "exec",
            "call_2",
            &json!({"command": "ls"}),
            "Error: not found",
            false,
            5,
            "inline",
        );

        let entries = log.get_entries();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].seq, 0);
        assert_eq!(entries[0].tool_name, "read_file");
        assert!(entries[0].result_ok);
        assert_eq!(entries[1].seq, 1);
        assert!(!entries[1].result_ok);
    }

    #[test]
    fn test_hash_chain_verification() {
        let (_tmp, log) = make_audit();
        log.record("tool_a", "c1", &json!({}), "result_a", true, 10, "inline");
        log.record("tool_b", "c2", &json!({}), "result_b", true, 20, "inline");
        log.record("tool_c", "c3", &json!({}), "result_c", true, 30, "inline");

        let result = log.verify_chain();
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 3);
    }

    #[test]
    fn test_chain_tamper_detection() {
        let (_tmp, log) = make_audit();
        log.record("tool_a", "c1", &json!({}), "result_a", true, 10, "inline");
        log.record("tool_b", "c2", &json!({}), "result_b", true, 20, "inline");

        // Tamper with the file: replace a result
        let data = fs::read_to_string(&log.file_path).unwrap();
        let tampered = data.replace("result_a", "FAKE_result");
        fs::write(&log.file_path, tampered).unwrap();

        let result = log.verify_chain();
        assert!(result.is_err());
    }

    #[test]
    fn test_find_matching_result() {
        let (_tmp, log) = make_audit();
        log.record(
            "read_file",
            "c1",
            &json!({}),
            "hello world",
            true,
            5,
            "inline",
        );
        log.record(
            "exec",
            "c2",
            &json!({}),
            "total 42 files",
            true,
            10,
            "inline",
        );

        let found = log.find_matching_result("42 files");
        assert!(found.is_some());
        assert_eq!(found.unwrap().tool_name, "exec");

        let not_found = log.find_matching_result("nonexistent");
        assert!(not_found.is_none());
    }

    #[test]
    fn test_result_truncation() {
        let (_tmp, log) = make_audit();
        let big_result = "x".repeat(20000);
        log.record("tool", "c1", &json!({}), &big_result, true, 5, "inline");

        let entries = log.get_entries();
        assert!(entries[0].result_data.len() < 10000);
        assert!(entries[0].result_data.ends_with("...[truncated]"));
    }

    #[test]
    fn test_empty_log_verification() {
        let (_tmp, log) = make_audit();
        let result = log.verify_chain();
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0);
    }

    #[test]
    fn test_prev_hash_linkage() {
        let (_tmp, log) = make_audit();
        log.record("a", "c1", &json!({}), "r1", true, 1, "inline");
        log.record("b", "c2", &json!({}), "r2", true, 2, "inline");

        let entries = log.get_entries();
        assert!(entries[0].prev_hash.is_empty());
        assert_eq!(entries[1].prev_hash, entries[0].hash);
    }

    #[test]
    fn test_session_key_sanitization() {
        let tmp = TempDir::new().unwrap();
        let log = AuditLog::new(tmp.path(), "cli:user@123/test");
        log.record("tool", "c1", &json!({}), "ok", true, 1, "inline");
        let entries = log.get_entries();
        assert_eq!(entries.len(), 1);
    }

    #[test]
    fn test_resume_from_existing_log() {
        let tmp = TempDir::new().unwrap();

        // First session: write 2 entries.
        {
            let log = AuditLog::new(tmp.path(), "resume-test");
            log.record("a", "c1", &json!({}), "r1", true, 1, "inline");
            log.record("b", "c2", &json!({}), "r2", true, 2, "inline");
        }

        // Second session: resume and write more.
        {
            let log = AuditLog::new(tmp.path(), "resume-test");
            log.record("c", "c3", &json!({}), "r3", true, 3, "inline");
        }

        // Verify full chain.
        let log = AuditLog::new(tmp.path(), "resume-test");
        let result = log.verify_chain();
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 3);

        let entries = log.get_entries();
        assert_eq!(entries[2].seq, 2);
        assert_eq!(entries[2].prev_hash, entries[1].hash);
    }

    #[test]
    fn test_write_turn_summary() {
        let tmp = TempDir::new().unwrap();
        let summary = TurnSummary {
            turn: 5,
            timestamp: "2026-02-13T14:30:00Z".to_string(),
            context_tokens: 12400,
            message_count: 34,
            tools_called: vec![
                TurnToolEntry {
                    name: "read_file".to_string(),
                    id: "tc_1".to_string(),
                    ok: true,
                    duration_ms: 45,
                    result_chars: 1200,
                },
                TurnToolEntry {
                    name: "exec".to_string(),
                    id: "tc_2".to_string(),
                    ok: false,
                    duration_ms: 120,
                    result_chars: 340,
                },
            ],
            working_memory_tokens: 820,
        };

        write_turn_summary(tmp.path(), "test-session", &summary);

        // Read back and verify.
        let path = tmp
            .path()
            .join("memory")
            .join("audit")
            .join("test-session.turns.jsonl");
        assert!(path.exists());
        let content = fs::read_to_string(&path).unwrap();
        let parsed: TurnSummary = serde_json::from_str(content.trim()).unwrap();
        assert_eq!(parsed.turn, 5);
        assert_eq!(parsed.context_tokens, 12400);
        assert_eq!(parsed.tools_called.len(), 2);
        assert_eq!(parsed.tools_called[0].name, "read_file");
        assert!(!parsed.tools_called[1].ok);
    }

    #[test]
    fn test_progress_event_construction() {
        let event = ToolEvent::Progress {
            tool_name: "exec".to_string(),
            tool_call_id: "call_1".to_string(),
            elapsed_ms: 3500,
            output_preview: Some("building...".to_string()),
        };
        match event {
            ToolEvent::Progress {
                tool_name,
                elapsed_ms,
                output_preview,
                ..
            } => {
                assert_eq!(tool_name, "exec");
                assert_eq!(elapsed_ms, 3500);
                assert_eq!(output_preview.as_deref(), Some("building..."));
            }
            _ => panic!("Expected Progress variant"),
        }
    }

    #[test]
    fn test_progress_event_without_preview() {
        let event = ToolEvent::Progress {
            tool_name: "web_fetch".to_string(),
            tool_call_id: "call_2".to_string(),
            elapsed_ms: 1000,
            output_preview: None,
        };
        match event {
            ToolEvent::Progress { output_preview, .. } => {
                assert!(output_preview.is_none());
            }
            _ => panic!("Expected Progress variant"),
        }
    }
}
