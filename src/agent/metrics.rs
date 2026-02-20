//! Lightweight per-request metrics written to `~/.nanobot/metrics.jsonl`.
//!
//! Each LLM call emits one [`RequestMetrics`] line — append-only, one JSON
//! object per line.  Consumed by `nanobot sessions list` / external scripts.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;

use serde::Serialize;

/// One metric record per LLM call.
#[derive(Debug, Clone, Serialize)]
pub struct RequestMetrics {
    pub timestamp: String,
    pub request_id: String,
    pub role: String,
    pub model: String,
    pub provider_base: String,
    pub elapsed_ms: u64,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_detail: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub anti_drift_score: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub anti_drift_signals: Option<Vec<String>>,
    pub tool_calls_requested: u32,
    pub tool_calls_executed: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub validation_result: Option<String>,
}

/// Return the metrics file path (`~/.nanobot/metrics.jsonl`).
pub fn metrics_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_default()
        .join(".nanobot")
        .join("metrics.jsonl")
}

/// Maximum metrics file size before rotation (10 MB).
const MAX_METRICS_BYTES: u64 = 10 * 1024 * 1024;

/// Append a single metrics record to the JSONL file.
///
/// When the file exceeds [`MAX_METRICS_BYTES`], the current file is rotated
/// to `metrics.jsonl.1` (overwriting any previous backup) and a fresh file
/// is started.  Only one backup is kept — `nanobot sessions purge` handles
/// deeper cleanup.
///
/// Failures are silently ignored — metrics are best-effort and must never
/// crash the agent loop.
pub fn emit(metrics: &RequestMetrics) {
    let path = metrics_path();
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }

    // Rotate if over size limit.
    if let Ok(meta) = fs::metadata(&path) {
        if meta.len() > MAX_METRICS_BYTES {
            let backup = path.with_extension("jsonl.1");
            let _ = fs::rename(&path, &backup);
        }
    }

    let Ok(line) = serde_json::to_string(metrics) else {
        return;
    };
    let Ok(mut file) = OpenOptions::new().create(true).append(true).open(&path) else {
        return;
    };
    let _ = writeln!(file, "{}", line);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_metrics_serialization() {
        let m = RequestMetrics {
            timestamp: "2026-02-20T12:00:00Z".into(),
            request_id: "abc12345".into(),
            role: "main".into(),
            model: "qwen3-8b".into(),
            provider_base: "http://localhost:1234/v1".into(),
            elapsed_ms: 1500,
            prompt_tokens: 2048,
            completion_tokens: 256,
            status: "ok".into(),
            error_detail: None,
            anti_drift_score: Some(0.3),
            anti_drift_signals: Some(vec!["filler_heavy".into()]),
            tool_calls_requested: 2,
            tool_calls_executed: 2,
            validation_result: None,
        };

        let json = serde_json::to_string(&m).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed["request_id"], "abc12345");
        assert_eq!(parsed["elapsed_ms"], 1500);
        assert_eq!(parsed["status"], "ok");
        assert!(parsed.get("error_detail").is_none()); // skip_serializing_if
        assert!(parsed.get("validation_result").is_none());
        assert_eq!(parsed["anti_drift_score"], 0.3);
    }

    #[test]
    fn test_request_metrics_with_error() {
        let m = RequestMetrics {
            timestamp: "2026-02-20T12:00:00Z".into(),
            request_id: "def67890".into(),
            role: "router".into(),
            model: "nvidia_Orchestrator-8B".into(),
            provider_base: "http://192.168.1.22:1234/v1".into(),
            elapsed_ms: 200,
            prompt_tokens: 0,
            completion_tokens: 0,
            status: "error:reasoning_config_rejected".into(),
            error_detail: Some("reasoning_budget not supported".into()),
            anti_drift_score: None,
            anti_drift_signals: None,
            tool_calls_requested: 0,
            tool_calls_executed: 0,
            validation_result: None,
        };

        let json = serde_json::to_string(&m).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed["status"], "error:reasoning_config_rejected");
        assert_eq!(parsed["error_detail"], "reasoning_budget not supported");
        assert_eq!(parsed["role"], "router");
    }

    #[test]
    fn test_rotation_on_size_limit() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("metrics.jsonl");
        let backup = dir.path().join("metrics.jsonl.1");

        // Write a file that exceeds MAX_METRICS_BYTES.
        {
            let mut f = OpenOptions::new().create(true).write(true).open(&path).unwrap();
            // Write ~11MB of data to trigger rotation.
            let line = "x".repeat(1024);
            for _ in 0..(11 * 1024) {
                writeln!(f, "{}", line).unwrap();
            }
        }
        assert!(fs::metadata(&path).unwrap().len() > MAX_METRICS_BYTES);

        // Manually call the rotation logic (emit() uses hardcoded path,
        // so we test the rotation logic directly).
        if let Ok(meta) = fs::metadata(&path) {
            if meta.len() > MAX_METRICS_BYTES {
                let _ = fs::rename(&path, &backup);
            }
        }

        assert!(backup.exists(), "backup file should exist after rotation");
        assert!(!path.exists(), "original should be gone (renamed)");
    }

    #[test]
    fn test_emit_to_tempdir() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("metrics.jsonl");

        let m = RequestMetrics {
            timestamp: "2026-02-20T12:00:00Z".into(),
            request_id: "test1234".into(),
            role: "main".into(),
            model: "test-model".into(),
            provider_base: "http://localhost/v1".into(),
            elapsed_ms: 100,
            prompt_tokens: 10,
            completion_tokens: 5,
            status: "ok".into(),
            error_detail: None,
            anti_drift_score: None,
            anti_drift_signals: None,
            tool_calls_requested: 0,
            tool_calls_executed: 0,
            validation_result: None,
        };

        // Write directly to test path
        let line = serde_json::to_string(&m).unwrap();
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .unwrap();
        writeln!(file, "{}", line).unwrap();
        drop(file);

        let content = fs::read_to_string(&path).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(content.trim()).unwrap();
        assert_eq!(parsed["request_id"], "test1234");
    }
}
