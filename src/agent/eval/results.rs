use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Top-level evaluation result container.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalResult {
    /// Which benchmark was run.
    pub benchmark: BenchmarkType,
    /// When the evaluation started.
    pub started_at: String,
    /// When the evaluation completed.
    pub completed_at: String,
    /// Model used.
    pub model: String,
    /// Benchmark-specific results.
    pub data: BenchmarkData,
    /// Free-form metadata (config params, etc).
    pub metadata: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum BenchmarkType {
    Hanoi,
    Haystack,
    Learning,
    Sprint,
}

/// Benchmark-specific result data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BenchmarkData {
    HanoiCalibration {
        num_disks: u8,
        samples: usize,
        accuracy: f64,
        red_flag_rate: f64,
        median_latency_ms: f64,
        voters_needed: Option<usize>,
    },
    HanoiSolve {
        num_disks: u8,
        steps_completed: usize,
        steps_total: usize,
        errors: usize,
        avg_voters_per_step: f64,
        catts_savings: f64,
        total_duration_ms: u64,
    },
    HaystackRetrieval {
        num_facts: usize,
        total_length: usize,
        precision: f64,
        recall: f64,
        mrr: f64,
    },
    HaystackAggregation {
        num_facts: usize,
        total_length: usize,
        tasks_correct: usize,
        tasks_total: usize,
        accuracy: f64,
        mean_search_calls: f64,
    },
    Learning {
        family: String,
        total_tasks: usize,
        completed: usize,
        final_accuracy: f64,
        forward_transfer: f64,
        surprise_rate: f64,
    },
    Sprint {
        corpus_size: usize,
        questions_total: usize,
        questions_correct: usize,
        accuracy: f64,
        compound_score: f64,
        catts_savings_trend: Vec<f64>,
        time_per_question: Vec<f64>,
    },
}

/// Summary for display - one line per benchmark.
#[derive(Debug, Clone)]
pub struct ResultSummary {
    pub benchmark: String,
    pub model: String,
    pub date: String,
    pub headline: String,
    pub score: f64,
}

/// Returns current time as ISO 8601 string.
pub fn now_timestamp() -> String {
    chrono::Utc::now().to_rfc3339()
}

/// Replace colons with dashes for filename safety.
pub fn filename_safe_timestamp(ts: &str) -> String {
    ts.replace(':', "-")
}

/// Returns `~/.nanobot/eval_results/`.
pub fn default_results_dir() -> PathBuf {
    dirs::home_dir()
        .expect("Unable to determine home directory")
        .join(".nanobot")
        .join("eval_results")
}

/// Save to `{dir}/{benchmark}_{timestamp}.json`. Create dir if needed. Returns the path written.
pub fn save_result(result: &EvalResult, dir: &Path) -> Result<PathBuf, String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("Failed to create directory: {}", e))?;

    let benchmark_name = format!("{:?}", result.benchmark).to_lowercase();
    let safe_timestamp = filename_safe_timestamp(&result.completed_at);
    let filename = format!("{}_{}.json", benchmark_name, safe_timestamp);
    let path = dir.join(filename);

    let json = serde_json::to_string_pretty(result)
        .map_err(|e| format!("Failed to serialize result: {}", e))?;

    std::fs::write(&path, json).map_err(|e| format!("Failed to write file: {}", e))?;

    Ok(path)
}

/// Load a single result from JSON file.
pub fn load_result(path: &Path) -> Result<EvalResult, String> {
    let content =
        std::fs::read_to_string(path).map_err(|e| format!("Failed to read file: {}", e))?;
    serde_json::from_str(&content).map_err(|e| format!("Failed to parse JSON: {}", e))
}

/// Load all `.json` files from the directory, parse each as EvalResult, skip failures silently.
pub fn load_all_results(dir: &Path) -> Result<Vec<EvalResult>, String> {
    if !dir.exists() {
        return Ok(Vec::new());
    }

    let entries = std::fs::read_dir(dir).map_err(|e| format!("Failed to read directory: {}", e))?;

    let mut results = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("json") {
            if let Ok(result) = load_result(&path) {
                results.push(result);
            }
        }
    }

    Ok(results)
}

/// Human-readable report for a single result.
pub fn format_result(result: &EvalResult) -> String {
    let mut output = String::new();
    output.push_str(&format!(
        "=== {:?} Benchmark Results ===\n",
        result.benchmark
    ));
    output.push_str(&format!("Model: {}\n", result.model));
    output.push_str(&format!("Started: {}\n", result.started_at));
    output.push_str(&format!("Completed: {}\n", result.completed_at));
    output.push_str("\n");

    match &result.data {
        BenchmarkData::HanoiCalibration {
            num_disks,
            samples,
            accuracy,
            red_flag_rate,
            median_latency_ms,
            voters_needed,
        } => {
            output.push_str(&format!("Disks: {}\n", num_disks));
            output.push_str(&format!("Samples: {}\n", samples));
            output.push_str(&format!("Accuracy: {:.2}%\n", accuracy * 100.0));
            output.push_str(&format!("Red Flag Rate: {:.2}%\n", red_flag_rate * 100.0));
            output.push_str(&format!("Median Latency: {:.0}ms\n", median_latency_ms));
            if let Some(voters) = voters_needed {
                output.push_str(&format!("Voters Needed: {}\n", voters));
            }
        }
        BenchmarkData::HanoiSolve {
            num_disks,
            steps_completed,
            steps_total,
            errors,
            avg_voters_per_step,
            catts_savings,
            total_duration_ms,
        } => {
            output.push_str(&format!("Disks: {}\n", num_disks));
            output.push_str(&format!(
                "Completion: {}/{} steps ({:.1}%)\n",
                steps_completed,
                steps_total,
                (*steps_completed as f64 / *steps_total as f64) * 100.0
            ));
            output.push_str(&format!("Errors: {}\n", errors));
            output.push_str(&format!(
                "Avg Voters Per Step: {:.2}\n",
                avg_voters_per_step
            ));
            output.push_str(&format!("CATTS Savings: {:.2}%\n", catts_savings * 100.0));
            output.push_str(&format!("Duration: {}ms\n", total_duration_ms));
        }
        BenchmarkData::HaystackRetrieval {
            num_facts,
            total_length,
            precision,
            recall,
            mrr,
        } => {
            output.push_str(&format!("Facts: {}\n", num_facts));
            output.push_str(&format!("Total Length: {} chars\n", total_length));
            output.push_str(&format!("Precision: {:.2}%\n", precision * 100.0));
            output.push_str(&format!("Recall: {:.2}%\n", recall * 100.0));
            output.push_str(&format!("MRR: {:.3}\n", mrr));
        }
        BenchmarkData::HaystackAggregation {
            num_facts,
            total_length,
            tasks_correct,
            tasks_total,
            accuracy,
            mean_search_calls,
        } => {
            output.push_str(&format!("Facts: {}\n", num_facts));
            output.push_str(&format!("Total Length: {} chars\n", total_length));
            output.push_str(&format!(
                "Tasks: {}/{} correct\n",
                tasks_correct, tasks_total
            ));
            output.push_str(&format!("Accuracy: {:.2}%\n", accuracy * 100.0));
            output.push_str(&format!("Mean Search Calls: {:.2}\n", mean_search_calls));
        }
        BenchmarkData::Learning {
            family,
            total_tasks,
            completed,
            final_accuracy,
            forward_transfer,
            surprise_rate,
        } => {
            output.push_str(&format!("Family: {}\n", family));
            output.push_str(&format!("Tasks: {}/{}\n", completed, total_tasks));
            output.push_str(&format!("Final Accuracy: {:.2}%\n", final_accuracy * 100.0));
            output.push_str(&format!("Forward Transfer: {:.3}\n", forward_transfer));
            output.push_str(&format!("Surprise Rate: {:.2}%\n", surprise_rate * 100.0));
        }
        BenchmarkData::Sprint {
            corpus_size,
            questions_total,
            questions_correct,
            accuracy,
            compound_score,
            catts_savings_trend,
            time_per_question,
        } => {
            output.push_str(&format!("Corpus Size: {} docs\n", corpus_size));
            output.push_str(&format!(
                "Questions: {}/{} correct\n",
                questions_correct, questions_total
            ));
            output.push_str(&format!("Accuracy: {:.2}%\n", accuracy * 100.0));
            output.push_str(&format!("Compound Score: {:.3}\n", compound_score));
            output.push_str(&format!(
                "CATTS Savings Trend: [{:.2}..{:.2}]\n",
                catts_savings_trend.first().unwrap_or(&0.0),
                catts_savings_trend.last().unwrap_or(&0.0)
            ));
            output.push_str(&format!(
                "Time Per Question: avg {:.0}ms\n",
                time_per_question.iter().sum::<f64>() / time_per_question.len() as f64
            ));
        }
    }

    if !result.metadata.is_empty() {
        output.push_str("\nMetadata:\n");
        for (key, value) in &result.metadata {
            output.push_str(&format!("  {}: {}\n", key, value));
        }
    }

    output
}

/// Extract a one-line summary from any result.
pub fn summarize(result: &EvalResult) -> ResultSummary {
    let benchmark = format!("{:?}", result.benchmark);
    let date = result
        .completed_at
        .split('T')
        .next()
        .unwrap_or(&result.completed_at)
        .to_string();

    let (headline, score) = match &result.data {
        BenchmarkData::HanoiCalibration {
            accuracy,
            voters_needed,
            ..
        } => {
            let voters_str = voters_needed
                .map(|v| format!(", {} voters", v))
                .unwrap_or_default();
            (
                format!("{:.1}% accuracy{}", accuracy * 100.0, voters_str),
                *accuracy,
            )
        }
        BenchmarkData::HanoiSolve {
            steps_completed,
            steps_total,
            catts_savings,
            ..
        } => {
            let completion = *steps_completed as f64 / *steps_total as f64;
            (
                format!(
                    "{}/{} steps, {:.1}% savings",
                    steps_completed,
                    steps_total,
                    catts_savings * 100.0
                ),
                completion,
            )
        }
        BenchmarkData::HaystackRetrieval { mrr, recall, .. } => (
            format!("MRR {:.3}, Recall {:.1}%", mrr, recall * 100.0),
            *mrr,
        ),
        BenchmarkData::HaystackAggregation { accuracy, .. } => {
            (format!("{:.1}% accuracy", accuracy * 100.0), *accuracy)
        }
        BenchmarkData::Learning {
            final_accuracy,
            forward_transfer,
            ..
        } => (
            format!(
                "{:.1}% accuracy, FT {:.2}",
                final_accuracy * 100.0,
                forward_transfer
            ),
            *final_accuracy,
        ),
        BenchmarkData::Sprint {
            compound_score,
            accuracy,
            ..
        } => (
            format!(
                "Compound {:.3}, Acc {:.1}%",
                compound_score,
                accuracy * 100.0
            ),
            *compound_score,
        ),
    };

    ResultSummary {
        benchmark,
        model: result.model.clone(),
        date,
        headline,
        score,
    }
}

/// Table of all results: benchmark | model | date | headline | score
pub fn format_summary(results: &[EvalResult]) -> String {
    if results.is_empty() {
        return "No evaluation results found.\n".to_string();
    }

    let summaries: Vec<ResultSummary> = results.iter().map(summarize).collect();

    let mut output = String::new();
    output.push_str(&format!(
        "{:<12} {:<20} {:<12} {:<40} {:>8}\n",
        "Benchmark", "Model", "Date", "Headline", "Score"
    ));
    output.push_str(&"-".repeat(100));
    output.push('\n');

    for summary in summaries {
        output.push_str(&format!(
            "{:<12} {:<20} {:<12} {:<40} {:>8.3}\n",
            summary.benchmark, summary.model, summary.date, summary.headline, summary.score
        ));
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_result() -> EvalResult {
        EvalResult {
            benchmark: BenchmarkType::Hanoi,
            started_at: "2026-02-16T12:00:00Z".to_string(),
            completed_at: "2026-02-16T12:05:00Z".to_string(),
            model: "test-model".to_string(),
            data: BenchmarkData::HanoiCalibration {
                num_disks: 3,
                samples: 100,
                accuracy: 0.95,
                red_flag_rate: 0.02,
                median_latency_ms: 150.0,
                voters_needed: Some(3),
            },
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn test_save_and_load_roundtrip() {
        let temp_dir = TempDir::new().unwrap();
        let result = create_test_result();

        let path = save_result(&result, temp_dir.path()).unwrap();
        assert!(path.exists());

        let loaded = load_result(&path).unwrap();
        assert_eq!(loaded.benchmark, result.benchmark);
        assert_eq!(loaded.model, result.model);
        assert_eq!(loaded.started_at, result.started_at);
        assert_eq!(loaded.completed_at, result.completed_at);

        match loaded.data {
            BenchmarkData::HanoiCalibration {
                num_disks,
                samples,
                accuracy,
                red_flag_rate,
                median_latency_ms,
                voters_needed,
            } => {
                assert_eq!(num_disks, 3);
                assert_eq!(samples, 100);
                assert!((accuracy - 0.95).abs() < 0.001);
                assert!((red_flag_rate - 0.02).abs() < 0.001);
                assert!((median_latency_ms - 150.0).abs() < 0.1);
                assert_eq!(voters_needed, Some(3));
            }
            _ => panic!("Wrong data variant"),
        }
    }

    #[test]
    fn test_load_all_results() {
        let temp_dir = TempDir::new().unwrap();

        let result1 = create_test_result();
        let mut result2 = create_test_result();
        result2.completed_at = "2026-02-16T13:00:00Z".to_string();
        let mut result3 = create_test_result();
        result3.completed_at = "2026-02-16T14:00:00Z".to_string();

        save_result(&result1, temp_dir.path()).unwrap();
        save_result(&result2, temp_dir.path()).unwrap();
        save_result(&result3, temp_dir.path()).unwrap();

        let loaded = load_all_results(temp_dir.path()).unwrap();
        assert_eq!(loaded.len(), 3);
    }

    #[test]
    fn test_format_result_hanoi() {
        let result = create_test_result();
        let formatted = format_result(&result);

        assert!(formatted.contains("Hanoi"));
        assert!(formatted.contains("test-model"));
        assert!(formatted.contains("95.00%"));
        assert!(formatted.contains("Accuracy"));
        assert!(formatted.contains("Red Flag Rate"));
        assert!(formatted.contains("Voters Needed: 3"));
    }

    #[test]
    fn test_summarize() {
        let result = create_test_result();
        let summary = summarize(&result);

        assert_eq!(summary.benchmark, "Hanoi");
        assert_eq!(summary.model, "test-model");
        assert_eq!(summary.date, "2026-02-16");
        assert!(summary.headline.contains("95.0%"));
        assert!(summary.headline.contains("3 voters"));
        assert!((summary.score - 0.95).abs() < 0.001);
    }

    #[test]
    fn test_filename_safe_timestamp() {
        let input = "2026-02-16T12:00:00Z";
        let output = filename_safe_timestamp(input);
        assert_eq!(output, "2026-02-16T12-00-00Z");
        assert!(!output.contains(':'));
    }
}
