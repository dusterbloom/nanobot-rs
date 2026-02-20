#![allow(dead_code)]
//! Budget calibration using historical execution data.
//!
//! Tracks per-task-type performance stats in SQLite and provides budget
//! recommendations based on P75 iterations, historical success rates, and
//! task complexity patterns.

use anyhow::Result;
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// A recorded execution measurement for calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionRecord {
    /// Task type category (e.g., "shell", "web_search", "code_analysis", "delegate").
    pub task_type: String,
    /// Model used.
    pub model: String,
    /// Number of iterations used.
    pub iterations_used: u32,
    /// Maximum iterations allowed.
    pub max_iterations: u32,
    /// Whether the task succeeded.
    pub success: bool,
    /// Total cost in USD.
    pub cost_usd: f64,
    /// Wall-clock duration in milliseconds.
    pub duration_ms: u64,
    /// Depth in delegation tree.
    pub depth: u32,
    /// Number of tool calls made.
    pub tool_calls: u32,
    /// Timestamp.
    pub created_at: String,
}

/// Calibrated budget recommendation for a task type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetRecommendation {
    /// Task type this recommendation is for.
    pub task_type: String,
    /// Recommended max_iterations based on historical p75.
    pub recommended_iterations: u32,
    /// Recommended max_depth.
    pub recommended_depth: u32,
    /// Recommended budget_multiplier for children.
    pub recommended_multiplier: f32,
    /// Historical success rate for this task type.
    pub historical_success_rate: f64,
    /// Historical mean cost.
    pub mean_cost_usd: f64,
    /// Number of samples this recommendation is based on.
    pub sample_count: i64,
    /// Confidence: higher with more samples.
    pub confidence: f64,
}

/// Aggregate stats for a task type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskTypeStats {
    pub task_type: String,
    pub total_executions: i64,
    pub successful: i64,
    pub success_rate: f64,
    pub mean_iterations: f64,
    pub p75_iterations: f64,
    pub p95_iterations: f64,
    pub mean_cost_usd: f64,
    pub mean_duration_ms: f64,
    pub mean_tool_calls: f64,
}

/// SQLite-backed budget calibrator.
pub struct BudgetCalibrator {
    conn: Connection,
}

impl BudgetCalibrator {
    /// Open or create the calibration database.
    pub fn open(db_path: &Path) -> Result<Self> {
        let conn = Connection::open(db_path)?;
        conn.execute_batch(
            "
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = NORMAL;

            CREATE TABLE IF NOT EXISTS execution_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_type TEXT NOT NULL,
                model TEXT NOT NULL DEFAULT '',
                iterations_used INTEGER NOT NULL,
                max_iterations INTEGER NOT NULL,
                success INTEGER NOT NULL DEFAULT 1,
                cost_usd REAL NOT NULL DEFAULT 0.0,
                duration_ms INTEGER NOT NULL DEFAULT 0,
                depth INTEGER NOT NULL DEFAULT 0,
                tool_calls INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_records_task_type ON execution_records(task_type);
            CREATE INDEX IF NOT EXISTS idx_records_created ON execution_records(created_at DESC);
        ",
        )?;
        Ok(Self { conn })
    }

    /// Open at the default location: ~/.nanobot/calibration.db
    pub fn open_default() -> Result<Self> {
        let db_path = dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".nanobot")
            .join("calibration.db");
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        Self::open(&db_path)
    }

    /// Record an execution.
    pub fn record(&self, record: &ExecutionRecord) -> Result<i64> {
        self.conn.execute(
            "INSERT INTO execution_records (task_type, model, iterations_used, max_iterations, success, cost_usd, duration_ms, depth, tool_calls)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                record.task_type, record.model, record.iterations_used,
                record.max_iterations, record.success as i32, record.cost_usd,
                record.duration_ms as i64, record.depth, record.tool_calls,
            ],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Get aggregate stats for a task type.
    pub fn task_stats(&self, task_type: &str) -> Result<Option<TaskTypeStats>> {
        let total: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM execution_records WHERE task_type = ?1",
            params![task_type],
            |r| r.get(0),
        )?;

        if total == 0 {
            return Ok(None);
        }

        let successful: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM execution_records WHERE task_type = ?1 AND success = 1",
            params![task_type],
            |r| r.get(0),
        )?;

        let mean_iterations: f64 = self.conn.query_row(
            "SELECT AVG(iterations_used) FROM execution_records WHERE task_type = ?1",
            params![task_type],
            |r| r.get(0),
        )?;

        let mean_cost: f64 = self.conn.query_row(
            "SELECT AVG(cost_usd) FROM execution_records WHERE task_type = ?1",
            params![task_type],
            |r| r.get(0),
        )?;

        let mean_duration: f64 = self.conn.query_row(
            "SELECT AVG(duration_ms) FROM execution_records WHERE task_type = ?1",
            params![task_type],
            |r| r.get(0),
        )?;

        let mean_tools: f64 = self.conn.query_row(
            "SELECT AVG(tool_calls) FROM execution_records WHERE task_type = ?1",
            params![task_type],
            |r| r.get(0),
        )?;

        // P75 and P95 via window functions on sorted iterations
        let p75 = self.percentile_iterations(task_type, 0.75)?;
        let p95 = self.percentile_iterations(task_type, 0.95)?;

        Ok(Some(TaskTypeStats {
            task_type: task_type.to_string(),
            total_executions: total,
            successful,
            success_rate: successful as f64 / total as f64,
            mean_iterations,
            p75_iterations: p75,
            p95_iterations: p95,
            mean_cost_usd: mean_cost,
            mean_duration_ms: mean_duration,
            mean_tool_calls: mean_tools,
        }))
    }

    /// Compute percentile of iterations_used for a task type.
    fn percentile_iterations(&self, task_type: &str, percentile: f64) -> Result<f64> {
        let mut stmt = self.conn.prepare(
            "SELECT iterations_used FROM execution_records WHERE task_type = ?1 ORDER BY iterations_used ASC"
        )?;
        let values: Vec<u32> = stmt
            .query_map(params![task_type], |r| r.get(0))?
            .collect::<Result<Vec<_>, _>>()?;

        if values.is_empty() {
            return Ok(0.0);
        }

        let idx = ((values.len() as f64 * percentile).ceil() as usize)
            .saturating_sub(1)
            .min(values.len() - 1);
        Ok(values[idx] as f64)
    }

    /// Get a budget recommendation for a task type.
    pub fn recommend(&self, task_type: &str) -> Result<BudgetRecommendation> {
        match self.task_stats(task_type)? {
            Some(stats) => {
                // Use P75 + 2 as recommended iterations (some headroom over typical)
                let recommended_iterations = (stats.p75_iterations as u32 + 2).max(3);

                // Recommend depth based on whether delegation was historically used
                let recommended_depth = if stats.mean_tool_calls > 3.0 { 2 } else { 1 };

                // Multiplier: tighter for simple tasks, looser for complex
                let recommended_multiplier = if stats.mean_iterations < 3.0 {
                    0.3
                } else {
                    0.5
                };

                // Confidence based on sample size (logarithmic scale)
                // log2(1) = 0, log2(2) = 1, log2(32) = 5, so we add 1 to give some confidence to single samples
                let confidence = ((stats.total_executions as f64).log2() + 1.0).min(6.0) / 6.0;

                Ok(BudgetRecommendation {
                    task_type: task_type.to_string(),
                    recommended_iterations,
                    recommended_depth,
                    recommended_multiplier,
                    historical_success_rate: stats.success_rate,
                    mean_cost_usd: stats.mean_cost_usd,
                    sample_count: stats.total_executions,
                    confidence,
                })
            }
            None => {
                // No data: return defaults
                Ok(BudgetRecommendation {
                    task_type: task_type.to_string(),
                    recommended_iterations: 8,
                    recommended_depth: 2,
                    recommended_multiplier: 0.5,
                    historical_success_rate: 0.0,
                    mean_cost_usd: 0.0,
                    sample_count: 0,
                    confidence: 0.0,
                })
            }
        }
    }

    /// List all known task types with their stats.
    pub fn all_task_types(&self) -> Result<Vec<TaskTypeStats>> {
        let mut stmt = self
            .conn
            .prepare("SELECT DISTINCT task_type FROM execution_records ORDER BY task_type")?;
        let types: Vec<String> = stmt
            .query_map([], |r| r.get(0))?
            .collect::<Result<Vec<_>, _>>()?;

        let mut result = Vec::new();
        for task_type in &types {
            if let Some(stats) = self.task_stats(task_type)? {
                result.push(stats);
            }
        }
        Ok(result)
    }

    /// Prune old records (keep last N per task type).
    pub fn prune(&self, keep_per_type: usize) -> Result<usize> {
        let types = self.all_task_types()?;
        let mut total_deleted = 0;

        for stats in &types {
            if stats.total_executions > keep_per_type as i64 {
                let delete_count = stats.total_executions - keep_per_type as i64;
                let deleted = self.conn.execute(
                    "DELETE FROM execution_records WHERE task_type = ?1 AND id IN (
                        SELECT id FROM execution_records WHERE task_type = ?1
                        ORDER BY created_at ASC LIMIT ?2
                    )",
                    params![stats.task_type, delete_count],
                )?;
                total_deleted += deleted;
            }
        }

        Ok(total_deleted)
    }

    /// Get total record count.
    pub fn total_records(&self) -> Result<i64> {
        self.conn
            .query_row("SELECT COUNT(*) FROM execution_records", [], |r| r.get(0))
            .map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn test_db() -> BudgetCalibrator {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        BudgetCalibrator::open(&db_path).unwrap()
    }

    fn sample_record(task_type: &str, iterations_used: u32, success: bool) -> ExecutionRecord {
        ExecutionRecord {
            task_type: task_type.to_string(),
            model: "gpt-4".to_string(),
            iterations_used,
            max_iterations: 10,
            success,
            cost_usd: 0.05,
            duration_ms: 1000,
            depth: 0,
            tool_calls: 2,
            created_at: chrono::Utc::now().to_rfc3339(),
        }
    }

    #[test]
    fn test_open_creates_db() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let result = BudgetCalibrator::open(&db_path);
        assert!(result.is_ok());
        assert!(db_path.exists());
    }

    #[test]
    fn test_record_and_stats() {
        let cal = test_db();

        // Record 5 executions
        for i in 1..=5 {
            let rec = sample_record("shell", i, true);
            cal.record(&rec).unwrap();
        }

        let stats = cal.task_stats("shell").unwrap().unwrap();
        assert_eq!(stats.total_executions, 5);
        assert_eq!(stats.successful, 5);
        assert_eq!(stats.success_rate, 1.0);
        assert_eq!(stats.mean_iterations, 3.0); // (1+2+3+4+5)/5 = 3
    }

    #[test]
    fn test_percentile_iterations() {
        let cal = test_db();

        // Record iterations: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
        for i in 1..=10 {
            let rec = sample_record("code_analysis", i, true);
            cal.record(&rec).unwrap();
        }

        let stats = cal.task_stats("code_analysis").unwrap().unwrap();
        // P75 of [1..10] = 8th value (0.75 * 10 = 7.5, ceil = 8, idx = 7)
        assert_eq!(stats.p75_iterations, 8.0);
        // P95 of [1..10] = 10th value (0.95 * 10 = 9.5, ceil = 10, idx = 9)
        assert_eq!(stats.p95_iterations, 10.0);
    }

    #[test]
    fn test_recommend_with_data() {
        let cal = test_db();

        // Record data with P75 at 6
        for i in &[1, 2, 3, 4, 5, 6, 7, 8] {
            let rec = sample_record("web_search", *i, true);
            cal.record(&rec).unwrap();
        }

        let rec = cal.recommend("web_search").unwrap();
        // P75 = 6, so recommended = 6 + 2 = 8
        assert_eq!(rec.recommended_iterations, 8);
        assert_eq!(rec.historical_success_rate, 1.0);
        assert!(rec.confidence > 0.0);
        assert_eq!(rec.sample_count, 8);
    }

    #[test]
    fn test_recommend_no_data() {
        let cal = test_db();

        let rec = cal.recommend("nonexistent").unwrap();
        assert_eq!(rec.recommended_iterations, 8);
        assert_eq!(rec.recommended_depth, 2);
        assert_eq!(rec.recommended_multiplier, 0.5);
        assert_eq!(rec.historical_success_rate, 0.0);
        assert_eq!(rec.sample_count, 0);
        assert_eq!(rec.confidence, 0.0);
    }

    #[test]
    fn test_all_task_types() {
        let cal = test_db();

        cal.record(&sample_record("shell", 1, true)).unwrap();
        cal.record(&sample_record("web_search", 2, true)).unwrap();
        cal.record(&sample_record("delegate", 3, true)).unwrap();

        let types = cal.all_task_types().unwrap();
        assert_eq!(types.len(), 3);

        let type_names: Vec<String> = types.iter().map(|t| t.task_type.clone()).collect();
        assert!(type_names.contains(&"shell".to_string()));
        assert!(type_names.contains(&"web_search".to_string()));
        assert!(type_names.contains(&"delegate".to_string()));
    }

    #[test]
    fn test_prune() {
        let cal = test_db();

        // Record 20 shell executions
        for i in 1..=20 {
            let rec = sample_record("shell", i, true);
            cal.record(&rec).unwrap();
        }

        assert_eq!(cal.total_records().unwrap(), 20);

        // Prune to keep only 10
        let deleted = cal.prune(10).unwrap();
        assert_eq!(deleted, 10);
        assert_eq!(cal.total_records().unwrap(), 10);

        // Verify the oldest were deleted (should have iterations 11-20 remaining)
        let stats = cal.task_stats("shell").unwrap().unwrap();
        assert_eq!(stats.total_executions, 10);
        // Mean of 11..=20 is 15.5
        assert!((stats.mean_iterations - 15.5).abs() < 0.01);
    }

    #[test]
    fn test_success_rate() {
        let cal = test_db();

        // Record mixed success/failure
        for i in 1..=10 {
            let success = i % 2 == 0;
            let rec = sample_record("mixed", i, success);
            cal.record(&rec).unwrap();
        }

        let stats = cal.task_stats("mixed").unwrap().unwrap();
        assert_eq!(stats.total_executions, 10);
        assert_eq!(stats.successful, 5);
        assert_eq!(stats.success_rate, 0.5);
    }

    #[test]
    fn test_multiple_models() {
        let cal = test_db();

        // Record same task type with different models
        let mut rec1 = sample_record("code_analysis", 3, true);
        rec1.model = "gpt-4".to_string();
        cal.record(&rec1).unwrap();

        let mut rec2 = sample_record("code_analysis", 5, true);
        rec2.model = "claude-3".to_string();
        cal.record(&rec2).unwrap();

        let stats = cal.task_stats("code_analysis").unwrap().unwrap();
        assert_eq!(stats.total_executions, 2);
        assert_eq!(stats.mean_iterations, 4.0); // (3+5)/2
    }

    #[test]
    fn test_confidence_scales_with_samples() {
        let cal = test_db();

        // 1 sample
        cal.record(&sample_record("type1", 5, true)).unwrap();
        let rec1 = cal.recommend("type1").unwrap();

        // 100 samples
        for _ in 0..100 {
            cal.record(&sample_record("type2", 5, true)).unwrap();
        }
        let rec2 = cal.recommend("type2").unwrap();

        // More samples = higher confidence
        assert!(rec2.confidence > rec1.confidence);
        assert!(rec1.confidence > 0.0);
    }

    #[test]
    fn test_total_records() {
        let cal = test_db();

        assert_eq!(cal.total_records().unwrap(), 0);

        cal.record(&sample_record("shell", 1, true)).unwrap();
        cal.record(&sample_record("web", 2, true)).unwrap();

        assert_eq!(cal.total_records().unwrap(), 2);
    }

    #[test]
    fn test_record_fields() {
        let cal = test_db();

        let original = ExecutionRecord {
            task_type: "test_task".to_string(),
            model: "test-model".to_string(),
            iterations_used: 7,
            max_iterations: 12,
            success: false,
            cost_usd: 0.123,
            duration_ms: 5432,
            depth: 3,
            tool_calls: 9,
            created_at: "2025-01-01T00:00:00Z".to_string(),
        };

        let id = cal.record(&original).unwrap();
        assert!(id > 0);

        // Read back and verify all fields
        let retrieved: ExecutionRecord = cal.conn.query_row(
            "SELECT task_type, model, iterations_used, max_iterations, success, cost_usd, duration_ms, depth, tool_calls, created_at
             FROM execution_records WHERE id = ?1",
            params![id],
            |r| Ok(ExecutionRecord {
                task_type: r.get(0)?,
                model: r.get(1)?,
                iterations_used: r.get(2)?,
                max_iterations: r.get(3)?,
                success: r.get::<_, i32>(4)? != 0,
                cost_usd: r.get(5)?,
                duration_ms: r.get::<_, i64>(6)? as u64,
                depth: r.get(7)?,
                tool_calls: r.get(8)?,
                created_at: r.get(9)?,
            })
        ).unwrap();

        assert_eq!(retrieved.task_type, original.task_type);
        assert_eq!(retrieved.model, original.model);
        assert_eq!(retrieved.iterations_used, original.iterations_used);
        assert_eq!(retrieved.max_iterations, original.max_iterations);
        assert_eq!(retrieved.success, original.success);
        assert!((retrieved.cost_usd - original.cost_usd).abs() < 0.001);
        assert_eq!(retrieved.duration_ms, original.duration_ms);
        assert_eq!(retrieved.depth, original.depth);
        assert_eq!(retrieved.tool_calls, original.tool_calls);
    }

    #[test]
    fn test_recommended_depth_based_on_tool_calls() {
        let cal = test_db();

        // Low tool calls -> depth 1
        for _ in 0..5 {
            let mut rec = sample_record("simple", 2, true);
            rec.tool_calls = 1;
            cal.record(&rec).unwrap();
        }

        let rec1 = cal.recommend("simple").unwrap();
        assert_eq!(rec1.recommended_depth, 1);

        // High tool calls -> depth 2
        for _ in 0..5 {
            let mut rec = sample_record("complex", 5, true);
            rec.tool_calls = 8;
            cal.record(&rec).unwrap();
        }

        let rec2 = cal.recommend("complex").unwrap();
        assert_eq!(rec2.recommended_depth, 2);
    }

    #[test]
    fn test_recommended_multiplier_based_on_iterations() {
        let cal = test_db();

        // Simple tasks (low iterations) -> tighter multiplier
        for i in 1..=5 {
            let rec = sample_record("simple", i % 3 + 1, true); // iterations: 1-3
            cal.record(&rec).unwrap();
        }

        let rec1 = cal.recommend("simple").unwrap();
        assert_eq!(rec1.recommended_multiplier, 0.3);

        // Complex tasks (high iterations) -> looser multiplier
        for i in 1..=5 {
            let rec = sample_record("complex", i + 5, true); // iterations: 6-10
            cal.record(&rec).unwrap();
        }

        let rec2 = cal.recommend("complex").unwrap();
        assert_eq!(rec2.recommended_multiplier, 0.5);
    }
}
