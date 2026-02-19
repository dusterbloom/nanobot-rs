//! Research Sprint: compound evaluation combining all three challenges.
//!
//! Tests context retrieval + multi-step verification + learning compounding
//! at test time over a 20-question sprint on a synthetic corpus.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::haystack::{self, Fact};

// ============================================================================
// Types
// ============================================================================

/// Difficulty tier for sprint questions.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum QuestionDifficulty {
    /// Q1-5: single-fact lookup.
    SimpleRetrieval,
    /// Q6-10: two-hop reasoning.
    MultiHop,
    /// Q11-15: aggregation across chunks.
    Aggregation,
    /// Q16-20: cross-domain synthesis.
    Synthesis,
}

impl QuestionDifficulty {
    pub fn label(&self) -> &'static str {
        match self {
            Self::SimpleRetrieval => "Simple Retrieval",
            Self::MultiHop => "Multi-Hop",
            Self::Aggregation => "Aggregation",
            Self::Synthesis => "Synthesis",
        }
    }
}

/// A single sprint question with ground truth.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SprintQuestion {
    /// Question index (0-based).
    pub index: usize,
    /// The question text.
    pub question: String,
    /// Expected answer.
    pub expected_answer: String,
    /// Difficulty tier.
    pub difficulty: QuestionDifficulty,
    /// Which topic domains are involved.
    pub domains: Vec<String>,
}

/// Execution record for a single sprint question.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuestionExecution {
    /// Question index.
    pub index: usize,
    /// Agent's answer.
    pub agent_answer: String,
    /// Whether the answer was correct.
    pub correct: bool,
    /// Number of search calls made.
    pub search_calls: usize,
    /// Number of voters used (MAKER).
    pub voters_used: usize,
    /// Whether CATTS accepted the pilot.
    pub catts_accepted_pilot: bool,
    /// Iterations used by the budget calibrator.
    pub iterations_used: u32,
    /// Duration in milliseconds.
    pub duration_ms: u64,
    /// Cost in USD.
    pub cost_usd: f64,
}

/// Configuration for the sprint benchmark.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SprintConfig {
    pub corpus_size: usize,
    pub num_questions: usize,
    pub num_facts: usize,
    pub num_domains: usize,
    pub seed: u64,
}

impl Default for SprintConfig {
    fn default() -> Self {
        Self {
            corpus_size: 500_000,
            num_questions: 20,
            num_facts: 200,
            num_domains: 10,
            seed: 42,
        }
    }
}

/// The compound scorecard — the key output of the sprint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SprintScorecard {
    /// Overall accuracy.
    pub accuracy: f64,
    /// Questions correct / total.
    pub questions_correct: usize,
    pub questions_total: usize,

    /// Average time per question per difficulty tier.
    pub avg_time_by_difficulty: HashMap<String, f64>,

    /// CATTS pilot acceptance rate per tier (fraction that didn't escalate).
    pub catts_acceptance_by_tier: HashMap<String, f64>,

    /// Budget calibration: mean iterations per tier.
    pub budget_iterations_by_tier: HashMap<String, f64>,

    /// Compound score: accuracy * speed_improvement * savings_ratio.
    pub compound_score: f64,

    /// Trend metrics: are later questions cheaper/faster?
    pub speed_trend: f64,
    pub savings_trend: f64,
}

/// Topic domain for cross-domain questions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopicDomain {
    pub name: String,
    pub facts: Vec<Fact>,
}

// ============================================================================
// Corpus generation
// ============================================================================

/// Domain names for the synthetic corpus.
const DOMAINS: &[&str] = &[
    "medicine",
    "engineering",
    "education",
    "finance",
    "agriculture",
    "technology",
    "law",
    "arts",
    "science",
    "sports",
];

/// Generate a multi-domain corpus with facts distributed across domains.
pub fn generate_corpus(config: &SprintConfig) -> (Vec<TopicDomain>, String) {
    let mut domains = Vec::new();
    let facts_per_domain = config.num_facts / config.num_domains.max(1);

    for (d_idx, domain_name) in DOMAINS.iter().take(config.num_domains).enumerate() {
        let domain_seed = config.seed.wrapping_add(d_idx as u64 * 1000);
        let base_id = d_idx * facts_per_domain;
        let facts = haystack::generate_facts(facts_per_domain, domain_seed)
            .into_iter()
            .enumerate()
            .map(|(i, mut f)| {
                f.person_id = base_id + i;
                f.name = format!("Person_{}", base_id + i);
                f
            })
            .collect();

        domains.push(TopicDomain {
            name: domain_name.to_string(),
            facts,
        });
    }

    // Assemble all facts into one document
    let all_facts: Vec<Fact> = domains.iter().flat_map(|d| d.facts.clone()).collect();
    let document = haystack::assemble_document(&all_facts, config.corpus_size, config.seed);

    (domains, document)
}

/// Get all facts from all domains.
pub fn all_facts(domains: &[TopicDomain]) -> Vec<Fact> {
    domains.iter().flat_map(|d| d.facts.clone()).collect()
}

// ============================================================================
// Question generation
// ============================================================================

/// Generate sprint questions with escalating difficulty.
pub fn generate_questions(
    domains: &[TopicDomain],
    num_questions: usize,
    seed: u64,
) -> Vec<SprintQuestion> {
    let all = all_facts(domains);
    let mut questions = Vec::new();
    let per_tier = num_questions / 4;
    let remainder = num_questions % 4;

    // Tier 1: Simple Retrieval
    for i in 0..per_tier {
        let fact_idx = (seed as usize + i) % all.len();
        let fact = &all[fact_idx];
        questions.push(SprintQuestion {
            index: questions.len(),
            question: format!("What is {}'s job?", fact.name),
            expected_answer: fact.job.clone(),
            difficulty: QuestionDifficulty::SimpleRetrieval,
            domains: vec![domain_for_fact(domains, fact.person_id)],
        });
    }

    // Tier 2: Multi-Hop
    for i in 0..per_tier {
        let idx_a = (seed as usize + i * 3) % all.len();
        let idx_b = (seed as usize + i * 3 + 1) % all.len();
        let fact_a = &all[idx_a];
        let fact_b = &all[idx_b];
        let same_city = fact_a.city == fact_b.city;
        questions.push(SprintQuestion {
            index: questions.len(),
            question: format!(
                "Does {} live in the same city as {}?",
                fact_a.name, fact_b.name
            ),
            expected_answer: if same_city {
                "yes".to_string()
            } else {
                "no".to_string()
            },
            difficulty: QuestionDifficulty::MultiHop,
            domains: vec![
                domain_for_fact(domains, fact_a.person_id),
                domain_for_fact(domains, fact_b.person_id),
            ],
        });
    }

    // Tier 3: Aggregation
    for i in 0..(per_tier + if remainder > 0 { 1 } else { 0 }) {
        if questions.len() >= num_questions {
            break;
        }
        let job_idx = (seed as usize + i) % haystack::JOBS.len();
        let job = haystack::JOBS[job_idx];
        let count = all.iter().filter(|f| f.job == job).count();
        questions.push(SprintQuestion {
            index: questions.len(),
            question: format!("How many people work as a {}?", job),
            expected_answer: count.to_string(),
            difficulty: QuestionDifficulty::Aggregation,
            domains: DOMAINS
                .iter()
                .take(domains.len())
                .map(|s| s.to_string())
                .collect(),
        });
    }

    // Tier 4: Synthesis
    while questions.len() < num_questions {
        let i = questions.len() - (per_tier * 3 + per_tier.min(remainder));
        let domain_idx = (seed as usize + i) % domains.len();
        let domain = &domains[domain_idx];

        // Count scientists in this domain
        let job = "scientist";
        let count_in_domain = domain.facts.iter().filter(|f| f.job == job).count();
        let city_idx = (seed as usize + i * 7) % haystack::CITIES.len();
        let city = haystack::CITIES[city_idx];
        let in_city = domain
            .facts
            .iter()
            .filter(|f| f.job == job && f.city == city)
            .count();

        questions.push(SprintQuestion {
            index: questions.len(),
            question: format!(
                "In the {} domain, how many scientists are in {}?",
                domain.name, city
            ),
            expected_answer: in_city.to_string(),
            difficulty: QuestionDifficulty::Synthesis,
            domains: vec![domain.name.clone()],
        });
    }

    questions.truncate(num_questions);
    questions
}

/// Find which domain a person belongs to.
fn domain_for_fact(domains: &[TopicDomain], person_id: usize) -> String {
    for domain in domains {
        if domain.facts.iter().any(|f| f.person_id == person_id) {
            return domain.name.clone();
        }
    }
    "unknown".to_string()
}

// ============================================================================
// Scorecard computation
// ============================================================================

/// Compute the compound scorecard from question executions.
pub fn compute_scorecard(
    questions: &[SprintQuestion],
    executions: &[QuestionExecution],
) -> SprintScorecard {
    let total = questions.len();
    let correct = executions.iter().filter(|e| e.correct).count();
    let accuracy = if total > 0 {
        correct as f64 / total as f64
    } else {
        0.0
    };

    // Group by difficulty tier
    let mut time_by_tier: HashMap<String, Vec<f64>> = HashMap::new();
    let mut catts_by_tier: HashMap<String, Vec<bool>> = HashMap::new();
    let mut iters_by_tier: HashMap<String, Vec<f64>> = HashMap::new();

    for exec in executions {
        if let Some(q) = questions.get(exec.index) {
            let tier = q.difficulty.label().to_string();
            time_by_tier
                .entry(tier.clone())
                .or_default()
                .push(exec.duration_ms as f64);
            catts_by_tier
                .entry(tier.clone())
                .or_default()
                .push(exec.catts_accepted_pilot);
            iters_by_tier
                .entry(tier.clone())
                .or_default()
                .push(exec.iterations_used as f64);
        }
    }

    let avg_time_by_difficulty: HashMap<String, f64> = time_by_tier
        .iter()
        .map(|(k, v)| (k.clone(), mean(v)))
        .collect();

    let catts_acceptance_by_tier: HashMap<String, f64> = catts_by_tier
        .iter()
        .map(|(k, v)| {
            let accepted = v.iter().filter(|&&b| b).count();
            (
                k.clone(),
                if v.is_empty() {
                    0.0
                } else {
                    accepted as f64 / v.len() as f64
                },
            )
        })
        .collect();

    let budget_iterations_by_tier: HashMap<String, f64> = iters_by_tier
        .iter()
        .map(|(k, v)| (k.clone(), mean(v)))
        .collect();

    // Trend: compare first half vs second half timing
    let speed_trend = compute_trend(
        &executions
            .iter()
            .map(|e| e.duration_ms as f64)
            .collect::<Vec<_>>(),
    );

    // Savings trend: CATTS acceptance rate over time
    let savings_trend = compute_trend(
        &executions
            .iter()
            .map(|e| if e.catts_accepted_pilot { 1.0 } else { 0.0 })
            .collect::<Vec<_>>(),
    );

    // Compound score: accuracy * speed_improvement * savings
    let overall_catts = if executions.is_empty() {
        0.0
    } else {
        executions.iter().filter(|e| e.catts_accepted_pilot).count() as f64
            / executions.len() as f64
    };
    let speed_improvement = if speed_trend > 0.0 {
        1.0 / speed_trend.max(0.01)
    } else {
        1.0
    };
    let compound_score = accuracy * speed_improvement.min(2.0) * (1.0 + overall_catts);

    SprintScorecard {
        accuracy,
        questions_correct: correct,
        questions_total: total,
        avg_time_by_difficulty,
        catts_acceptance_by_tier,
        budget_iterations_by_tier,
        compound_score,
        speed_trend,
        savings_trend,
    }
}

/// Format the scorecard as a human-readable report.
pub fn format_scorecard(sc: &SprintScorecard) -> String {
    let mut out = String::new();
    out.push_str("Research Sprint Scorecard\n");
    out.push_str(&"=".repeat(54));
    out.push('\n');

    out.push_str(&format!(
        "Questions answered correctly:  {}/{} ({:.0}%)\n\n",
        sc.questions_correct,
        sc.questions_total,
        sc.accuracy * 100.0,
    ));

    out.push_str("Average time per question:\n");
    for tier in &["Simple Retrieval", "Multi-Hop", "Aggregation", "Synthesis"] {
        if let Some(t) = sc.avg_time_by_difficulty.get(*tier) {
            out.push_str(&format!("  {:20} {:.1}ms\n", tier, t));
        }
    }

    out.push_str("\nCATTS pilot acceptance:\n");
    for tier in &["Simple Retrieval", "Multi-Hop", "Aggregation", "Synthesis"] {
        if let Some(r) = sc.catts_acceptance_by_tier.get(*tier) {
            out.push_str(&format!("  {:20} {:.0}%\n", tier, r * 100.0));
        }
    }

    out.push_str("\nBudget calibration (mean iterations):\n");
    for tier in &["Simple Retrieval", "Multi-Hop", "Aggregation", "Synthesis"] {
        if let Some(i) = sc.budget_iterations_by_tier.get(*tier) {
            out.push_str(&format!("  {:20} {:.1}\n", tier, i));
        }
    }

    out.push_str(&format!("\nCompound Score: {:.3}\n", sc.compound_score));
    out.push_str(&format!(
        "Speed trend:   {:.3} (< 1.0 = getting faster)\n",
        sc.speed_trend
    ));
    out.push_str(&format!(
        "Savings trend: {:.3} (> 1.0 = more CATTS savings)\n",
        sc.savings_trend
    ));
    out.push_str(&"=".repeat(54));
    out.push('\n');

    out
}

// ============================================================================
// Helpers
// ============================================================================

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// Compute trend as ratio of second-half mean to first-half mean.
/// < 1.0 means the metric decreased (e.g., faster times).
/// > 1.0 means the metric increased (e.g., more savings).
fn compute_trend(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 1.0;
    }
    let mid = values.len() / 2;
    let first_half = mean(&values[..mid]);
    let second_half = mean(&values[mid..]);
    if first_half.abs() < 1e-9 {
        return 1.0;
    }
    second_half / first_half
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_corpus_structure() {
        let config = SprintConfig {
            corpus_size: 10_000,
            num_questions: 8,
            num_facts: 40,
            num_domains: 4,
            seed: 42,
        };
        let (domains, document) = generate_corpus(&config);
        assert_eq!(domains.len(), 4);
        assert_eq!(domains[0].facts.len(), 10); // 40 / 4
        assert!(document.len() >= 5000); // should approach corpus_size
    }

    #[test]
    fn test_generate_corpus_deterministic() {
        let config = SprintConfig {
            corpus_size: 5_000,
            num_questions: 4,
            num_facts: 20,
            num_domains: 2,
            seed: 99,
        };
        let (d1, doc1) = generate_corpus(&config);
        let (d2, doc2) = generate_corpus(&config);
        assert_eq!(d1[0].facts[0].name, d2[0].facts[0].name);
        assert_eq!(doc1, doc2);
    }

    #[test]
    fn test_generate_questions_count() {
        let config = SprintConfig::default();
        let (domains, _) = generate_corpus(&config);
        let questions = generate_questions(&domains, 20, config.seed);
        assert_eq!(questions.len(), 20);
    }

    #[test]
    fn test_question_difficulty_escalation() {
        let config = SprintConfig::default();
        let (domains, _) = generate_corpus(&config);
        let questions = generate_questions(&domains, 20, config.seed);

        // First 5 should be SimpleRetrieval
        for q in &questions[..5] {
            assert_eq!(q.difficulty, QuestionDifficulty::SimpleRetrieval);
        }
        // Next 5 should be MultiHop
        for q in &questions[5..10] {
            assert_eq!(q.difficulty, QuestionDifficulty::MultiHop);
        }
    }

    #[test]
    fn test_compute_scorecard_basic() {
        let questions = vec![
            SprintQuestion {
                index: 0,
                question: "Q1?".into(),
                expected_answer: "A1".into(),
                difficulty: QuestionDifficulty::SimpleRetrieval,
                domains: vec!["medicine".into()],
            },
            SprintQuestion {
                index: 1,
                question: "Q2?".into(),
                expected_answer: "A2".into(),
                difficulty: QuestionDifficulty::MultiHop,
                domains: vec!["engineering".into()],
            },
        ];

        let executions = vec![
            QuestionExecution {
                index: 0,
                agent_answer: "A1".into(),
                correct: true,
                search_calls: 2,
                voters_used: 3,
                catts_accepted_pilot: false,
                iterations_used: 3,
                duration_ms: 1000,
                cost_usd: 0.01,
            },
            QuestionExecution {
                index: 1,
                agent_answer: "wrong".into(),
                correct: false,
                search_calls: 5,
                voters_used: 5,
                catts_accepted_pilot: true,
                iterations_used: 2,
                duration_ms: 800,
                cost_usd: 0.02,
            },
        ];

        let sc = compute_scorecard(&questions, &executions);
        assert_eq!(sc.questions_correct, 1);
        assert_eq!(sc.questions_total, 2);
        assert!((sc.accuracy - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_format_scorecard() {
        let sc = SprintScorecard {
            accuracy: 0.9,
            questions_correct: 18,
            questions_total: 20,
            avg_time_by_difficulty: [("Simple Retrieval".into(), 2300.0)].into(),
            catts_acceptance_by_tier: [("Simple Retrieval".into(), 0.0)].into(),
            budget_iterations_by_tier: [("Simple Retrieval".into(), 3.0)].into(),
            compound_score: 1.35,
            speed_trend: 0.85,
            savings_trend: 1.2,
        };
        let report = format_scorecard(&sc);
        assert!(report.contains("18/20"));
        assert!(report.contains("90%"));
        assert!(report.contains("Compound Score"));
    }
}
