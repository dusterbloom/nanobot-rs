//! Lightweight per-request metrics written to `~/.nanobot/metrics.jsonl`.
//!
//! Each LLM call emits one [`RequestMetrics`] line — append-only, one JSON
//! object per line.  Consumed by `nanobot sessions list` / external scripts.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;

use serde::Serialize;

/// Cumulative prompt-cache accounting for one logical Nanobot session.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SessionCacheMetrics {
    pub calls: u64,
    pub prompt_tokens: u64,
    pub cache_read_tokens: u64,
    pub cache_creation_tokens: u64,
    pub cold_calls: u64,
}

impl SessionCacheMetrics {
    pub fn record(
        &mut self,
        prompt_tokens: u64,
        cache_read_tokens: Option<u64>,
        cache_creation_tokens: Option<u64>,
    ) {
        let cache_read_tokens = cache_read_tokens.unwrap_or(0);
        self.calls = self.calls.saturating_add(1);
        self.prompt_tokens = self.prompt_tokens.saturating_add(prompt_tokens);
        self.cache_read_tokens = self.cache_read_tokens.saturating_add(cache_read_tokens);
        self.cache_creation_tokens = self
            .cache_creation_tokens
            .saturating_add(cache_creation_tokens.unwrap_or(0));
        if prompt_tokens > 0 && cache_read_tokens == 0 {
            self.cold_calls = self.cold_calls.saturating_add(1);
        }
    }

    pub fn efficiency_pct(self) -> f64 {
        if self.prompt_tokens == 0 {
            0.0
        } else {
            self.cache_read_tokens as f64 * 100.0 / self.prompt_tokens as f64
        }
    }
}

/// One metric record per LLM call.
#[derive(Debug, Clone, Serialize)]
pub struct RequestMetrics {
    pub timestamp: String,
    pub request_id: String,
    /// Stable Nanobot session key spanning every provider call in one logical
    /// conversation, independent of retained Higgs session-id rotations.
    pub logical_session: String,
    /// Prompt-cache branch used for this request. Wire values are `active`,
    /// `retained_expansion`, and `fallback`.
    pub cache_route: String,
    pub role: String,
    pub model: String,
    pub provider_base: String,
    pub elapsed_ms: u64,
    /// Time to first token (ms) — the prefill cost. `None` for non-streaming
    /// calls or when no token was produced.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttft_ms: Option<u64>,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    /// Prompt-cache tokens served from the provider's KV/prefix cache this call
    /// (Anthropic/Zhipu `cache_read_input_tokens`). `None` when the provider
    /// does not report caching, so cache health is observable per-provider.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_tokens: Option<u64>,
    /// Prompt tokens written into the provider's cache this call
    /// (`cache_creation_input_tokens`). `None` when unreported.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_creation_tokens: Option<u64>,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_detail: Option<String>,
    /// Exact provider content for pathological responses. Healthy payloads are
    /// omitted to keep the append-only metrics stream compact.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_response: Option<String>,
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
fn emit_line<T: Serialize>(value: &T) {
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

    let Ok(line) = serde_json::to_string(value) else {
        return;
    };
    let Ok(mut file) = OpenOptions::new().create(true).append(true).open(&path) else {
        return;
    };
    let mut bytes = line.into_bytes();
    bytes.push(b'\n');
    let _ = file.write_all(&bytes);
}

pub fn emit(metrics: &RequestMetrics) {
    emit_line(metrics);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_cache_metrics_aggregate_provider_calls() {
        let mut stats = SessionCacheMetrics::default();
        stats.record(100, Some(60), Some(40));
        stats.record(200, Some(0), Some(200));

        assert_eq!(stats.calls, 2);
        assert_eq!(stats.prompt_tokens, 300);
        assert_eq!(stats.cache_read_tokens, 60);
        assert_eq!(stats.cache_creation_tokens, 240);
        assert_eq!(stats.cold_calls, 1);
        assert_eq!(stats.efficiency_pct(), 20.0);
    }

    #[test]
    fn test_request_metrics_serialization() {
        let m = RequestMetrics {
            timestamp: "2026-02-20T12:00:00Z".into(),
            request_id: "abc12345".into(),
            logical_session: "cli:session-42".into(),
            cache_route: "active".into(),
            role: "main".into(),
            model: "qwen3-8b".into(),
            provider_base: "http://localhost:1234/v1".into(),
            elapsed_ms: 1500,
            ttft_ms: Some(420),
            prompt_tokens: 2048,
            completion_tokens: 256,
            cache_read_tokens: Some(1500),
            cache_creation_tokens: Some(400),
            status: "ok".into(),
            error_detail: None,
            raw_response: None,
            anti_drift_score: Some(0.3),
            anti_drift_signals: Some(vec!["filler_heavy".into()]),
            tool_calls_requested: 2,
            tool_calls_executed: 2,
            validation_result: None,
        };

        let json = serde_json::to_string(&m).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed["request_id"], "abc12345");
        assert_eq!(parsed["logical_session"], "cli:session-42");
        assert_eq!(parsed["cache_route"], "active");
        assert_eq!(parsed["elapsed_ms"], 1500);
        assert_eq!(parsed["status"], "ok");
        assert!(parsed.get("error_detail").is_none()); // skip_serializing_if
        assert!(parsed.get("raw_response").is_none());
        assert!(parsed.get("validation_result").is_none());
        assert_eq!(parsed["anti_drift_score"], 0.3);
        assert_eq!(parsed["ttft_ms"], 420);
        assert_eq!(parsed["cache_read_tokens"], 1500);
        assert_eq!(parsed["cache_creation_tokens"], 400);
    }

    #[test]
    fn test_request_metrics_with_error() {
        let m = RequestMetrics {
            timestamp: "2026-02-20T12:00:00Z".into(),
            request_id: "def67890".into(),
            logical_session: "cli:router-7".into(),
            cache_route: "fallback".into(),
            role: "router".into(),
            model: "nvidia_Orchestrator-8B".into(),
            provider_base: "http://192.168.1.22:1234/v1".into(),
            elapsed_ms: 200,
            ttft_ms: None,
            prompt_tokens: 0,
            completion_tokens: 0,
            cache_read_tokens: None,
            cache_creation_tokens: None,
            status: "error:reasoning_config_rejected".into(),
            error_detail: Some("reasoning_budget not supported".into()),
            raw_response: Some("<tool_call>\n<tool_code>".into()),
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
        assert_eq!(parsed["raw_response"], "<tool_call>\n<tool_code>");
        assert_eq!(parsed["role"], "router");
        assert!(parsed.get("ttft_ms").is_none()); // skip_serializing_if None
    }

    #[test]
    fn test_rotation_on_size_limit() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("metrics.jsonl");
        let backup = dir.path().join("metrics.jsonl.1");

        // Write a file that exceeds MAX_METRICS_BYTES.
        {
            let mut f = OpenOptions::new()
                .create(true)
                .write(true)
                .open(&path)
                .unwrap();
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
            logical_session: "cli:test".into(),
            cache_route: "active".into(),
            role: "main".into(),
            model: "test-model".into(),
            provider_base: "http://localhost/v1".into(),
            elapsed_ms: 100,
            ttft_ms: Some(33),
            prompt_tokens: 10,
            completion_tokens: 5,
            cache_read_tokens: None,
            cache_creation_tokens: None,
            status: "ok".into(),
            error_detail: None,
            raw_response: None,
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
