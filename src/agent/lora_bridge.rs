use anyhow::Result;
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

// =============================================================================
// Experience Buffer (SQLite-backed)
// =============================================================================

/// A recorded experience from a successful tool execution trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experience {
    /// Unique ID.
    pub id: i64,
    /// The task/prompt that was given.
    pub prompt: String,
    /// The tool calls that were made (JSON array of {name, arguments, result}).
    pub tool_trace: String,
    /// The final response/summary.
    pub response: String,
    /// Whether the execution was successful.
    pub success: bool,
    /// Quality score (0.0-1.0) — from user feedback or automated checks.
    pub quality: f64,
    /// Model used for this execution.
    pub model: String,
    /// Surprise score: how unexpected was this trace (higher = more worth learning).
    pub surprise: f64,
    /// Whether this experience has been exported for training.
    pub exported: bool,
    /// When this experience was recorded.
    pub created_at: String,
}

/// SQLite-backed experience buffer for collecting training data.
pub struct ExperienceBuffer {
    conn: Connection,
}

impl ExperienceBuffer {
    /// Open or create the experience buffer database.
    pub fn open(db_path: &Path) -> Result<Self> {
        let conn = Connection::open(db_path)?;
        conn.execute_batch(
            "
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = NORMAL;

            CREATE TABLE IF NOT EXISTS experiences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt TEXT NOT NULL,
                tool_trace TEXT NOT NULL,
                response TEXT NOT NULL,
                success INTEGER NOT NULL DEFAULT 1,
                quality REAL NOT NULL DEFAULT 0.0,
                model TEXT NOT NULL DEFAULT '',
                surprise REAL NOT NULL DEFAULT 0.0,
                exported INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_experiences_surprise ON experiences(surprise DESC);
            CREATE INDEX IF NOT EXISTS idx_experiences_exported ON experiences(exported);
            CREATE INDEX IF NOT EXISTS idx_experiences_quality ON experiences(quality DESC);
        ",
        )?;
        Ok(Self { conn })
    }

    /// Open at the default location: ~/.nanobot/experience.db
    pub fn open_default() -> Result<Self> {
        let db_path = dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".nanobot")
            .join("experience.db");
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        Self::open(&db_path)
    }

    /// Record a new experience.
    pub fn record(
        &self,
        prompt: &str,
        tool_trace: &str,
        response: &str,
        success: bool,
        quality: f64,
        model: &str,
    ) -> Result<i64> {
        let surprise = compute_surprise(prompt, tool_trace);
        self.conn.execute(
            "INSERT INTO experiences (prompt, tool_trace, response, success, quality, model, surprise) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![prompt, tool_trace, response, success as i32, quality, model, surprise],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Get the top N unexported experiences sorted by surprise (most surprising first).
    pub fn top_unexported(&self, limit: usize) -> Result<Vec<Experience>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, prompt, tool_trace, response, success, quality, model, surprise, exported, created_at
             FROM experiences WHERE exported = 0 AND success = 1
             ORDER BY surprise DESC, quality DESC
             LIMIT ?1"
        )?;
        let rows = stmt.query_map(params![limit as i64], |row| {
            Ok(Experience {
                id: row.get(0)?,
                prompt: row.get(1)?,
                tool_trace: row.get(2)?,
                response: row.get(3)?,
                success: row.get::<_, i32>(4)? != 0,
                quality: row.get(5)?,
                model: row.get(6)?,
                surprise: row.get(7)?,
                exported: row.get::<_, i32>(8)? != 0,
                created_at: row.get(9)?,
            })
        })?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    /// Mark experiences as exported.
    pub fn mark_exported(&self, ids: &[i64]) -> Result<usize> {
        if ids.is_empty() {
            return Ok(0);
        }
        let placeholders: Vec<String> = ids.iter().map(|_| "?".to_string()).collect();
        let sql = format!(
            "UPDATE experiences SET exported = 1 WHERE id IN ({})",
            placeholders.join(",")
        );
        let params: Vec<&dyn rusqlite::types::ToSql> = ids
            .iter()
            .map(|id| id as &dyn rusqlite::types::ToSql)
            .collect();
        let count = self.conn.execute(&sql, params.as_slice())?;
        Ok(count)
    }

    /// Export experiences as JSONL for the Python ContinualLearner.
    /// Returns the path to the written JSONL file.
    pub fn export_jsonl(&self, output_path: &Path, limit: usize) -> Result<ExportResult> {
        let experiences = self.top_unexported(limit)?;
        if experiences.is_empty() {
            return Ok(ExportResult {
                path: output_path.to_path_buf(),
                count: 0,
                ids: vec![],
            });
        }

        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let mut lines = Vec::new();
        let mut ids = Vec::new();
        for exp in &experiences {
            // Format as chat-style training data
            let training_entry = serde_json::json!({
                "messages": [
                    {"role": "user", "content": exp.prompt},
                    {"role": "assistant", "content": exp.response}
                ],
                "tool_trace": serde_json::from_str::<serde_json::Value>(&exp.tool_trace).unwrap_or(serde_json::Value::Null),
                "quality": exp.quality,
                "surprise": exp.surprise,
                "model": exp.model,
            });
            lines.push(serde_json::to_string(&training_entry)?);
            ids.push(exp.id);
        }

        std::fs::write(output_path, lines.join("\n") + "\n")?;
        self.mark_exported(&ids)?;

        Ok(ExportResult {
            path: output_path.to_path_buf(),
            count: ids.len(),
            ids,
        })
    }

    /// Get buffer statistics.
    pub fn stats(&self) -> Result<BufferStats> {
        let total: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM experiences", [], |r| r.get(0))?;
        let successful: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM experiences WHERE success = 1",
            [],
            |r| r.get(0),
        )?;
        let unexported: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM experiences WHERE exported = 0 AND success = 1",
            [],
            |r| r.get(0),
        )?;
        let avg_surprise: f64 = self.conn.query_row(
            "SELECT COALESCE(AVG(surprise), 0.0) FROM experiences WHERE success = 1",
            [],
            |r| r.get(0),
        )?;
        Ok(BufferStats {
            total,
            successful,
            unexported,
            avg_surprise,
        })
    }
}

#[derive(Debug, Clone)]
pub struct ExportResult {
    pub path: PathBuf,
    pub count: usize,
    pub ids: Vec<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BufferStats {
    pub total: i64,
    pub successful: i64,
    pub unexported: i64,
    pub avg_surprise: f64,
}

// =============================================================================
// Surprise Detection (heuristic)
// =============================================================================

/// Compute a surprise score for an experience.
///
/// Higher surprise = more worth learning from. Uses simple heuristics:
/// - Longer tool traces suggest more complex/novel workflows
/// - More unique tool names suggest novel combinations
/// - Shorter prompts with longer traces suggest efficient delegation
pub fn compute_surprise(prompt: &str, tool_trace: &str) -> f64 {
    let prompt_len = prompt.len() as f64;
    let trace_len = tool_trace.len() as f64;

    // Factor 1: Trace complexity (longer traces = more interesting)
    let complexity = (trace_len / 1000.0).min(5.0) / 5.0; // 0-1, saturates at 5K chars

    // Factor 2: Efficiency ratio (short prompt, long trace = efficient delegation)
    let efficiency = if prompt_len > 0.0 {
        (trace_len / prompt_len).min(10.0) / 10.0
    } else {
        0.0
    };

    // Factor 3: Tool diversity (count unique tool names in trace)
    let diversity = count_unique_tools(tool_trace) as f64 / 5.0; // Normalize to ~1.0 for 5 tools
    let diversity = diversity.min(1.0);

    // Weighted combination
    0.4 * complexity + 0.3 * efficiency + 0.3 * diversity
}

/// Count unique tool names in a JSON tool trace.
fn count_unique_tools(tool_trace: &str) -> usize {
    // Try to parse as JSON array
    if let Ok(arr) = serde_json::from_str::<Vec<serde_json::Value>>(tool_trace) {
        let names: std::collections::HashSet<&str> = arr
            .iter()
            .filter_map(|v| v.get("name").and_then(|n| n.as_str()))
            .collect();
        names.len()
    } else {
        0
    }
}

// =============================================================================
// LoRA Hot-Swap (local server API)
// =============================================================================

/// Configuration for LoRA adapter management.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraConfig {
    /// Base URL of the local inference server (e.g., "http://127.0.0.1:8080").
    pub server_url: String,
    /// Path to the LoRA adapter file (.gguf).
    pub adapter_path: Option<PathBuf>,
    /// Scale factor for the LoRA adapter (0.0-1.0).
    pub scale: f64,
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self {
            server_url: "http://127.0.0.1:8080".to_string(),
            adapter_path: None,
            scale: 0.5,
        }
    }
}

/// Result of a LoRA hot-swap operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HotSwapResult {
    pub success: bool,
    pub message: String,
    pub adapter_path: Option<String>,
    pub scale: f64,
}

/// Apply a LoRA adapter to a running local server via POST /lora-adapters.
///
/// Uses the hot-swap API to load/unload adapters without restarting the server.
pub async fn apply_lora_adapter(config: &LoraConfig) -> Result<HotSwapResult> {
    let client = reqwest::Client::new();
    let url = format!("{}/lora-adapters", config.server_url);

    let body = if let Some(ref adapter_path) = config.adapter_path {
        // Apply adapter
        serde_json::json!([{
            "path": adapter_path.to_string_lossy(),
            "scale": config.scale,
        }])
    } else {
        // Remove all adapters (empty array)
        serde_json::json!([])
    };

    let resp = client.post(&url).json(&body).send().await?;

    let status = resp.status();
    let text = resp.text().await.unwrap_or_default();

    if status.is_success() {
        Ok(HotSwapResult {
            success: true,
            message: format!("LoRA adapter applied successfully: {}", text),
            adapter_path: config
                .adapter_path
                .as_ref()
                .map(|p| p.to_string_lossy().to_string()),
            scale: config.scale,
        })
    } else {
        Ok(HotSwapResult {
            success: false,
            message: format!("LoRA hot-swap failed ({}): {}", status, text),
            adapter_path: config
                .adapter_path
                .as_ref()
                .map(|p| p.to_string_lossy().to_string()),
            scale: config.scale,
        })
    }
}

/// Remove all LoRA adapters from a running local server.
pub async fn remove_lora_adapters(server_url: &str) -> Result<HotSwapResult> {
    apply_lora_adapter(&LoraConfig {
        server_url: server_url.to_string(),
        adapter_path: None,
        scale: 0.0,
    })
    .await
}

/// Check if the local server supports LoRA hot-swap.
pub async fn check_lora_support(server_url: &str) -> Result<bool> {
    let client = reqwest::Client::new();
    let url = format!("{}/lora-adapters", server_url);

    // Try a GET request to see if the endpoint exists
    match client.get(&url).send().await {
        Ok(resp) => Ok(resp.status().is_success() || resp.status().as_u16() == 405),
        Err(_) => Ok(false),
    }
}

// =============================================================================
// Training Pipeline Orchestration
// =============================================================================

/// Default export path for training JSONL.
pub fn default_export_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".nanobot")
        .join("training")
        .join("experiences.jsonl")
}

/// Default adapter output path.
pub fn default_adapter_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".nanobot")
        .join("adapters")
        .join("latest.gguf")
}

/// Status of the training pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingStatus {
    /// Whether a training run is currently active.
    pub active: bool,
    /// Number of experiences available for training.
    pub pending_experiences: i64,
    /// Path to the latest adapter file (if any).
    pub latest_adapter: Option<PathBuf>,
    /// Whether the adapter is currently loaded in the local server.
    pub adapter_loaded: bool,
}

/// Check the training pipeline status.
pub fn check_training_status(buffer: &ExperienceBuffer) -> Result<TrainingStatus> {
    let stats = buffer.stats()?;
    let adapter_path = default_adapter_path();
    let has_adapter = adapter_path.exists();

    Ok(TrainingStatus {
        active: false, // Would be tracked by a PID file
        pending_experiences: stats.unexported,
        latest_adapter: if has_adapter {
            Some(adapter_path)
        } else {
            None
        },
        adapter_loaded: false, // Would need to query the local server
    })
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn test_experience_buffer_open() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("experience.db");

        let buffer = ExperienceBuffer::open(&db_path).unwrap();
        assert!(db_path.exists());

        // Verify WAL mode
        let stats = buffer.stats().unwrap();
        assert_eq!(stats.total, 0);
    }

    #[test]
    fn test_record_and_retrieve() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("experience.db");
        let buffer = ExperienceBuffer::open(&db_path).unwrap();

        let id = buffer
            .record(
                "Test task",
                r#"[{"name":"read_file","arguments":{},"result":"ok"}]"#,
                "Task complete",
                true,
                0.8,
                "test-model",
            )
            .unwrap();

        assert_eq!(id, 1);

        let experiences = buffer.top_unexported(10).unwrap();
        assert_eq!(experiences.len(), 1);
        assert_eq!(experiences[0].prompt, "Test task");
        assert_eq!(experiences[0].quality, 0.8);
        assert_eq!(experiences[0].model, "test-model");
        assert!(experiences[0].surprise > 0.0);
    }

    #[test]
    fn test_surprise_computation() {
        // Empty trace - low surprise
        let s1 = compute_surprise("hello", "[]");
        assert!(s1 < 0.2);

        // Simple trace - moderate surprise
        let s2 = compute_surprise("do something", r#"[{"name":"tool1"}]"#);
        assert!(s2 > 0.0 && s2 < 0.5);

        // Complex trace - high surprise
        let complex_trace = r#"[
            {"name":"read_file","arguments":{"path":"a.txt"}},
            {"name":"write_file","arguments":{"path":"b.txt"}},
            {"name":"execute_shell","arguments":{"cmd":"ls"}},
            {"name":"search_web","arguments":{"query":"test"}},
            {"name":"send_message","arguments":{"text":"done"}}
        ]"#;
        let s3 = compute_surprise("task", complex_trace);
        assert!(s3 > 0.5);

        // Short prompt, long trace - high efficiency factor
        let s4 = compute_surprise("x", &"a".repeat(5000));
        assert!(s4 > 0.5);
    }

    #[test]
    fn test_count_unique_tools() {
        let trace = r#"[
            {"name":"read_file"},
            {"name":"write_file"},
            {"name":"read_file"},
            {"name":"search_web"}
        ]"#;
        assert_eq!(count_unique_tools(trace), 3);

        // Empty array
        assert_eq!(count_unique_tools("[]"), 0);

        // Missing name fields
        let no_names = r#"[{"tool":"read_file"},{"tool":"write_file"}]"#;
        assert_eq!(count_unique_tools(no_names), 0);
    }

    #[test]
    fn test_count_unique_tools_invalid_json() {
        assert_eq!(count_unique_tools("not json"), 0);
        assert_eq!(count_unique_tools("{\"not\":\"array\"}"), 0);
    }

    #[test]
    fn test_export_jsonl() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("experience.db");
        let buffer = ExperienceBuffer::open(&db_path).unwrap();

        // Record multiple experiences
        buffer
            .record(
                "task1",
                r#"[{"name":"tool1"}]"#,
                "done1",
                true,
                0.9,
                "model1",
            )
            .unwrap();
        buffer
            .record(
                "task2",
                r#"[{"name":"tool2"}]"#,
                "done2",
                true,
                0.7,
                "model2",
            )
            .unwrap();
        buffer
            .record(
                "task3",
                r#"[{"name":"tool3"}]"#,
                "done3",
                false,
                0.5,
                "model3",
            )
            .unwrap(); // Failed - won't export

        let output_path = dir.path().join("training").join("export.jsonl");
        let result = buffer.export_jsonl(&output_path, 10).unwrap();

        assert_eq!(result.count, 2); // Only successful experiences
        assert_eq!(result.ids.len(), 2);
        assert!(output_path.exists());

        // Verify JSONL format
        let content = fs::read_to_string(&output_path).unwrap();
        let lines: Vec<&str> = content.trim().split('\n').collect();
        assert_eq!(lines.len(), 2);

        // Parse first line
        let entry: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(entry["messages"][0]["role"], "user");
        assert_eq!(entry["messages"][0]["content"], "task1");
        assert_eq!(entry["messages"][1]["role"], "assistant");
        assert_eq!(entry["quality"], 0.9);
        assert!(entry["surprise"].is_number());

        // Verify marked as exported
        let unexported = buffer.top_unexported(10).unwrap();
        assert_eq!(unexported.len(), 0);
    }

    #[test]
    fn test_mark_exported() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("experience.db");
        let buffer = ExperienceBuffer::open(&db_path).unwrap();

        let id1 = buffer
            .record("task1", "[]", "done1", true, 0.8, "model")
            .unwrap();
        let id2 = buffer
            .record("task2", "[]", "done2", true, 0.8, "model")
            .unwrap();

        // Before marking
        let unexported = buffer.top_unexported(10).unwrap();
        assert_eq!(unexported.len(), 2);

        // Mark first as exported
        let count = buffer.mark_exported(&[id1]).unwrap();
        assert_eq!(count, 1);

        // After marking
        let unexported = buffer.top_unexported(10).unwrap();
        assert_eq!(unexported.len(), 1);
        assert_eq!(unexported[0].id, id2);

        // Mark with empty array
        let count = buffer.mark_exported(&[]).unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn test_buffer_stats() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("experience.db");
        let buffer = ExperienceBuffer::open(&db_path).unwrap();

        buffer
            .record("task1", "[]", "done1", true, 0.8, "model")
            .unwrap();
        buffer
            .record("task2", "[]", "done2", true, 0.9, "model")
            .unwrap();
        buffer
            .record("task3", "[]", "done3", false, 0.5, "model")
            .unwrap();

        let stats = buffer.stats().unwrap();
        assert_eq!(stats.total, 3);
        assert_eq!(stats.successful, 2);
        assert_eq!(stats.unexported, 2); // Only successful ones
        assert!(stats.avg_surprise >= 0.0);
    }

    #[test]
    fn test_export_empty_buffer() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("experience.db");
        let buffer = ExperienceBuffer::open(&db_path).unwrap();

        let output_path = dir.path().join("export.jsonl");
        let result = buffer.export_jsonl(&output_path, 10).unwrap();

        assert_eq!(result.count, 0);
        assert_eq!(result.ids.len(), 0);
    }

    #[test]
    fn test_lora_config_default() {
        let config = LoraConfig::default();
        assert_eq!(config.server_url, "http://127.0.0.1:8080");
        assert!(config.adapter_path.is_none());
        assert_eq!(config.scale, 0.5);
    }

    #[test]
    fn test_surprise_empty_trace() {
        let surprise = compute_surprise("test prompt", "");
        assert!(surprise < 0.1); // Very low surprise for empty trace
    }

    #[test]
    fn test_surprise_complex_trace() {
        // Create a complex trace with 8 different tools
        let trace = r#"[
            {"name":"read_file","arguments":{"path":"a.txt"}},
            {"name":"write_file","arguments":{"path":"b.txt"}},
            {"name":"execute_shell","arguments":{"cmd":"ls"}},
            {"name":"search_web","arguments":{"query":"test"}},
            {"name":"send_message","arguments":{"text":"done"}},
            {"name":"create_file","arguments":{"path":"c.txt"}},
            {"name":"delete_file","arguments":{"path":"d.txt"}},
            {"name":"list_directory","arguments":{"path":"/"}}
        ]"#;

        let surprise = compute_surprise("do stuff", trace);
        assert!(surprise > 0.6); // High surprise for complex multi-tool trace
    }

    #[test]
    fn test_training_status_no_adapter() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("experience.db");
        let buffer = ExperienceBuffer::open(&db_path).unwrap();

        buffer
            .record("task1", "[]", "done1", true, 0.8, "model")
            .unwrap();

        let status = check_training_status(&buffer).unwrap();
        assert!(!status.active);
        assert_eq!(status.pending_experiences, 1);
        assert!(status.latest_adapter.is_none());
        assert!(!status.adapter_loaded);
    }

    #[test]
    fn test_export_preserves_order() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("experience.db");
        let buffer = ExperienceBuffer::open(&db_path).unwrap();

        // Record with different surprise scores
        buffer
            .record("low", "x", "done", true, 0.5, "model")
            .unwrap();
        buffer
            .record("high", &"x".repeat(10000), "done", true, 0.9, "model")
            .unwrap();
        buffer
            .record("medium", &"x".repeat(1000), "done", true, 0.7, "model")
            .unwrap();

        // Top unexported should be ordered by surprise DESC
        let experiences = buffer.top_unexported(10).unwrap();
        assert_eq!(experiences.len(), 3);

        // Higher surprise should come first
        assert!(experiences[0].surprise >= experiences[1].surprise);
        assert!(experiences[1].surprise >= experiences[2].surprise);
    }
}
