//! Evaluation benchmarks for the "Three Impossible Things" plan.
//!
//! Four benchmarks testing context retrieval, multi-step verification,
//! learning, and compound performance:
//!
//! - `hanoi` — Towers of Hanoi (MAKER process verification)
//! - `haystack` — Aggregation Haystack (Oolong-inspired context retrieval)
//! - `learning` — Learning Curve (SWE-Bench-CL inspired improvement over time)
//! - `sprint` — Research Sprint (compound challenge using all three)
//! - `results` — Result types, JSON persistence, reporting
//! - `runner` — LLM integration and orchestration

pub mod hanoi;
pub mod haystack;
pub mod learning;
pub mod results;
pub mod runner;
pub mod sprint;

use serde::{Deserialize, Serialize};

/// Top-level eval configuration shared across benchmarks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalConfig {
    /// Model to use for evaluation.
    pub model: String,
    /// Random seed for deterministic generation.
    pub seed: u64,
    /// Whether to save results to disk.
    pub save_results: bool,
    /// Results output directory (default: ~/.nanobot/eval_results/).
    pub output_dir: Option<String>,
}

impl Default for EvalConfig {
    fn default() -> Self {
        Self {
            model: "default".to_string(),
            seed: 42,
            save_results: true,
            output_dir: None,
        }
    }
}
