use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A synthetic fact embedded in the haystack.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fact {
    pub person_id: usize,
    pub name: String,
    pub job: String,
    pub city: String,
}

/// Configuration for haystack evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HaystackConfig {
    pub num_facts: usize,
    pub total_length: usize,
    pub chunk_size: usize,
    pub chunk_overlap: usize,
}

impl Default for HaystackConfig {
    fn default() -> Self {
        Self {
            num_facts: 50,
            total_length: 100_000,
            chunk_size: 4096,
            chunk_overlap: 256,
        }
    }
}

/// Types of aggregation tasks.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AggregationTask {
    Count {
        job: String,
        expected: usize,
    },
    Distribution {
        expected_top_job: String,
    },
    Filter {
        city: String,
        expected_names: Vec<String>,
    },
    CrossRef {
        person_id: usize,
        expected_names: Vec<String>,
    },
    Temporal {
        position: TemporalPosition,
        expected_name: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TemporalPosition {
    First,
    Last,
}

/// Results from Tier 1 (pure retrieval) benchmark.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalMetrics {
    pub precision: f64,
    pub recall: f64,
    pub mrr: f64,
    pub facts_found: usize,
    pub facts_total: usize,
}

/// Result from a single aggregation task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregationResult {
    pub task: AggregationTask,
    pub correct: bool,
    pub agent_answer: String,
    pub search_calls: usize,
}

pub const JOBS: &[&str] = &[
    "engineer",
    "teacher",
    "doctor",
    "artist",
    "chef",
    "scientist",
    "lawyer",
    "writer",
    "nurse",
    "pilot",
];

pub const CITIES: &[&str] = &[
    "Helsinki", "Tokyo", "London", "New York", "Berlin", "Sydney", "Toronto", "Paris", "Mumbai",
    "Seoul",
];

const FILLER_SENTENCES: &[&str] = &[
    "The weather has been quite unpredictable this season.",
    "Recent studies suggest that regular exercise improves cognitive function.",
    "Technology continues to advance at an unprecedented pace.",
    "The local community gathered for the annual festival last weekend.",
    "Environmental conservation efforts have gained significant momentum.",
    "New research in renewable energy shows promising results.",
    "The historical significance of the region cannot be overstated.",
    "Modern architecture blends functionality with aesthetic appeal.",
    "The global economy faces both challenges and opportunities.",
    "Educational institutions are adapting to new teaching methodologies.",
];

/// Generate deterministic facts using seed.
pub fn generate_facts(num_facts: usize, seed: u64) -> Vec<Fact> {
    let mut facts = Vec::with_capacity(num_facts);
    for i in 0..num_facts {
        let job_idx = ((seed + i as u64) % JOBS.len() as u64) as usize;
        let city_idx = ((seed + i as u64 * 7) % CITIES.len() as u64) as usize;
        facts.push(Fact {
            person_id: i,
            name: format!("Person_{}", i),
            job: JOBS[job_idx].to_string(),
            city: CITIES[city_idx].to_string(),
        });
    }
    facts
}

/// Convert a fact to a sentence.
pub fn fact_to_sentence(fact: &Fact) -> String {
    format!("{} works as an {} in {}.", fact.name, fact.job, fact.city)
}

/// Generate filler text to reach target character count.
pub fn generate_filler(target_chars: usize, seed: u64) -> String {
    let mut result = String::new();
    let mut idx = seed as usize;

    while result.len() < target_chars {
        if !result.is_empty() {
            result.push(' ');
        }
        result.push_str(FILLER_SENTENCES[idx % FILLER_SENTENCES.len()]);
        idx += 1;
    }

    result.truncate(target_chars);
    result
}

/// Assemble document with facts embedded at evenly-spaced intervals in filler text.
pub fn assemble_document(facts: &[Fact], total_length: usize, seed: u64) -> String {
    if facts.is_empty() {
        return generate_filler(total_length, seed);
    }

    // Calculate fact sentences and their total length
    let fact_sentences: Vec<String> = facts.iter().map(fact_to_sentence).collect();
    let total_fact_chars: usize = fact_sentences.iter().map(|s| s.len() + 1).sum(); // +1 for space

    if total_fact_chars >= total_length {
        return fact_sentences.join(" ");
    }

    // Calculate filler needed
    let filler_chars = total_length - total_fact_chars;
    let chunk_size = filler_chars / (facts.len() + 1);

    let mut result = String::new();
    let mut filler_idx = seed as usize;

    // Add initial filler
    let initial_filler = generate_filler_chunk(chunk_size, &mut filler_idx);
    result.push_str(&initial_filler);

    // Interleave facts with filler
    for (i, fact_sentence) in fact_sentences.iter().enumerate() {
        if !result.is_empty() {
            result.push(' ');
        }
        result.push_str(fact_sentence);

        // Add filler after each fact (except the last one if we're running out of space)
        if i < facts.len() - 1 || result.len() < total_length - chunk_size {
            result.push(' ');
            let filler = generate_filler_chunk(chunk_size, &mut filler_idx);
            result.push_str(&filler);
        }
    }

    // Add any remaining filler to reach target length
    if result.len() < total_length {
        let remaining = total_length - result.len();
        if remaining > 1 {
            result.push(' ');
            let final_filler = generate_filler_chunk(remaining - 1, &mut filler_idx);
            result.push_str(&final_filler);
        }
    }

    result.truncate(total_length);
    result
}

/// Helper to generate a filler chunk of specified size.
fn generate_filler_chunk(target_chars: usize, filler_idx: &mut usize) -> String {
    let mut result = String::new();

    while result.len() < target_chars {
        if !result.is_empty() {
            result.push(' ');
        }
        result.push_str(FILLER_SENTENCES[*filler_idx % FILLER_SENTENCES.len()]);
        *filler_idx += 1;
    }

    result.truncate(target_chars);
    result
}

/// Count facts with given job.
pub fn compute_count(facts: &[Fact], job: &str) -> usize {
    facts.iter().filter(|f| f.job == job).count()
}

/// Find the most common job.
pub fn compute_distribution(facts: &[Fact]) -> String {
    let mut job_counts: HashMap<&str, usize> = HashMap::new();
    for fact in facts {
        *job_counts.entry(&fact.job).or_insert(0) += 1;
    }

    job_counts
        .into_iter()
        .max_by_key(|(_, count)| *count)
        .map(|(job, _)| job.to_string())
        .unwrap_or_default()
}

/// Find all person names in a given city.
pub fn compute_filter(facts: &[Fact], city: &str) -> Vec<String> {
    facts
        .iter()
        .filter(|f| f.city == city)
        .map(|f| f.name.clone())
        .collect()
}

/// Find all person names sharing a city with person_id (excluding person_id).
pub fn compute_cross_ref(facts: &[Fact], person_id: usize) -> Vec<String> {
    let target_city = facts
        .iter()
        .find(|f| f.person_id == person_id)
        .map(|f| &f.city);

    match target_city {
        Some(city) => facts
            .iter()
            .filter(|f| f.person_id != person_id && &f.city == city)
            .map(|f| f.name.clone())
            .collect(),
        None => Vec::new(),
    }
}

/// Generate aggregation tasks with ground truth.
pub fn generate_aggregation_tasks(facts: &[Fact]) -> Vec<AggregationTask> {
    if facts.is_empty() {
        return Vec::new();
    }

    let mut tasks = Vec::new();

    // Task 1: Count - pick first fact's job
    let first_job = &facts[0].job;
    let count = compute_count(facts, first_job);
    tasks.push(AggregationTask::Count {
        job: first_job.clone(),
        expected: count,
    });

    // Task 2: Distribution - most common job
    let top_job = compute_distribution(facts);
    tasks.push(AggregationTask::Distribution {
        expected_top_job: top_job,
    });

    // Task 3: Filter - pick first fact's city
    let first_city = &facts[0].city;
    let names = compute_filter(facts, first_city);
    tasks.push(AggregationTask::Filter {
        city: first_city.clone(),
        expected_names: names,
    });

    // Task 4: CrossRef - pick middle person
    let mid_person_id = facts.len() / 2;
    let cross_ref_names = compute_cross_ref(facts, mid_person_id);
    tasks.push(AggregationTask::CrossRef {
        person_id: mid_person_id,
        expected_names: cross_ref_names,
    });

    // Task 5: Temporal - first and last
    tasks.push(AggregationTask::Temporal {
        position: TemporalPosition::First,
        expected_name: facts[0].name.clone(),
    });

    tasks
}

/// Build aggregation prompt for each task type.
pub fn build_aggregation_prompt(task: &AggregationTask) -> String {
    match task {
        AggregationTask::Count { job, .. } => {
            format!("How many people work as {}s?", job)
        }
        AggregationTask::Distribution { .. } => {
            "What is the most common job among all people?".to_string()
        }
        AggregationTask::Filter { city, .. } => {
            format!("List all people who work in {}.", city)
        }
        AggregationTask::CrossRef { person_id, .. } => {
            format!(
                "List all people who work in the same city as Person_{}, excluding Person_{} themselves.",
                person_id, person_id
            )
        }
        AggregationTask::Temporal { position, .. } => match position {
            TemporalPosition::First => "Who was the first person mentioned?".to_string(),
            TemporalPosition::Last => "Who was the last person mentioned?".to_string(),
        },
    }
}

/// Evaluate retrieval quality using a search function.
pub fn evaluate_retrieval(
    facts: &[Fact],
    search_fn: impl Fn(&str) -> Vec<String>,
) -> RetrievalMetrics {
    if facts.is_empty() {
        return RetrievalMetrics {
            precision: 0.0,
            recall: 0.0,
            mrr: 0.0,
            facts_found: 0,
            facts_total: 0,
        };
    }

    let mut total_precision = 0.0;
    let mut total_recall = 0.0;
    let mut total_reciprocal_rank = 0.0;
    let mut facts_found = 0;

    for fact in facts {
        // Query using person name as the key term
        let query = &fact.name;
        let results = search_fn(query);

        if results.is_empty() {
            continue;
        }

        // Check if the fact sentence appears in results
        let fact_sentence = fact_to_sentence(fact);
        let mut found_rank: Option<usize> = None;

        for (idx, result) in results.iter().enumerate() {
            if result.contains(&fact.name)
                && result.contains(&fact.job)
                && result.contains(&fact.city)
            {
                found_rank = Some(idx + 1);
                break;
            }
        }

        if let Some(rank) = found_rank {
            facts_found += 1;
            // Precision: 1 relevant result / total results returned
            total_precision += 1.0 / results.len() as f64;
            // Recall: found (1) / expected (1)
            total_recall += 1.0;
            // Reciprocal rank
            total_reciprocal_rank += 1.0 / rank as f64;
        } else {
            // Not found: precision and recall are 0 for this query
            // MRR is also 0
        }
    }

    let num_queries = facts.len() as f64;

    RetrievalMetrics {
        precision: total_precision / num_queries,
        recall: total_recall / num_queries,
        mrr: total_reciprocal_rank / num_queries,
        facts_found,
        facts_total: facts.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_facts_deterministic() {
        let facts1 = generate_facts(10, 42);
        let facts2 = generate_facts(10, 42);

        assert_eq!(facts1.len(), facts2.len());
        for (f1, f2) in facts1.iter().zip(facts2.iter()) {
            assert_eq!(f1.person_id, f2.person_id);
            assert_eq!(f1.name, f2.name);
            assert_eq!(f1.job, f2.job);
            assert_eq!(f1.city, f2.city);
        }
    }

    #[test]
    fn test_generate_facts_count() {
        let facts = generate_facts(25, 123);
        assert_eq!(facts.len(), 25);

        // Verify each fact has correct person_id
        for (i, fact) in facts.iter().enumerate() {
            assert_eq!(fact.person_id, i);
            assert_eq!(fact.name, format!("Person_{}", i));
        }
    }

    #[test]
    fn test_fact_to_sentence() {
        let fact = Fact {
            person_id: 0,
            name: "Person_0".to_string(),
            job: "engineer".to_string(),
            city: "Helsinki".to_string(),
        };

        let sentence = fact_to_sentence(&fact);
        assert_eq!(sentence, "Person_0 works as an engineer in Helsinki.");
    }

    #[test]
    fn test_assemble_document_length() {
        let facts = generate_facts(5, 42);
        let target_length = 1000;
        let doc = assemble_document(&facts, target_length, 42);

        // Should be within 10% of target length (allowing for truncation)
        assert!(
            doc.len() >= target_length - 100 && doc.len() <= target_length,
            "Document length {} not close to target {}",
            doc.len(),
            target_length
        );
    }

    #[test]
    fn test_assemble_document_contains_facts() {
        let facts = generate_facts(5, 42);
        let doc = assemble_document(&facts, 2000, 42);

        // All fact names should appear in the document
        for fact in &facts {
            assert!(
                doc.contains(&fact.name),
                "Document missing fact: {}",
                fact.name
            );
            assert!(
                doc.contains(&fact.job),
                "Document missing job: {}",
                fact.job
            );
            assert!(
                doc.contains(&fact.city),
                "Document missing city: {}",
                fact.city
            );
        }
    }

    #[test]
    fn test_compute_count() {
        let facts = vec![
            Fact {
                person_id: 0,
                name: "Person_0".to_string(),
                job: "engineer".to_string(),
                city: "Helsinki".to_string(),
            },
            Fact {
                person_id: 1,
                name: "Person_1".to_string(),
                job: "engineer".to_string(),
                city: "Tokyo".to_string(),
            },
            Fact {
                person_id: 2,
                name: "Person_2".to_string(),
                job: "doctor".to_string(),
                city: "London".to_string(),
            },
        ];

        assert_eq!(compute_count(&facts, "engineer"), 2);
        assert_eq!(compute_count(&facts, "doctor"), 1);
        assert_eq!(compute_count(&facts, "pilot"), 0);
    }

    #[test]
    fn test_compute_distribution() {
        let facts = vec![
            Fact {
                person_id: 0,
                name: "Person_0".to_string(),
                job: "engineer".to_string(),
                city: "Helsinki".to_string(),
            },
            Fact {
                person_id: 1,
                name: "Person_1".to_string(),
                job: "engineer".to_string(),
                city: "Tokyo".to_string(),
            },
            Fact {
                person_id: 2,
                name: "Person_2".to_string(),
                job: "engineer".to_string(),
                city: "London".to_string(),
            },
            Fact {
                person_id: 3,
                name: "Person_3".to_string(),
                job: "doctor".to_string(),
                city: "Paris".to_string(),
            },
        ];

        assert_eq!(compute_distribution(&facts), "engineer");
    }

    #[test]
    fn test_compute_filter() {
        let facts = vec![
            Fact {
                person_id: 0,
                name: "Person_0".to_string(),
                job: "engineer".to_string(),
                city: "Helsinki".to_string(),
            },
            Fact {
                person_id: 1,
                name: "Person_1".to_string(),
                job: "teacher".to_string(),
                city: "Helsinki".to_string(),
            },
            Fact {
                person_id: 2,
                name: "Person_2".to_string(),
                job: "doctor".to_string(),
                city: "Tokyo".to_string(),
            },
        ];

        let helsinki_people = compute_filter(&facts, "Helsinki");
        assert_eq!(helsinki_people.len(), 2);
        assert!(helsinki_people.contains(&"Person_0".to_string()));
        assert!(helsinki_people.contains(&"Person_1".to_string()));

        let tokyo_people = compute_filter(&facts, "Tokyo");
        assert_eq!(tokyo_people.len(), 1);
        assert!(tokyo_people.contains(&"Person_2".to_string()));
    }

    #[test]
    fn test_compute_cross_ref() {
        let facts = vec![
            Fact {
                person_id: 0,
                name: "Person_0".to_string(),
                job: "engineer".to_string(),
                city: "Helsinki".to_string(),
            },
            Fact {
                person_id: 1,
                name: "Person_1".to_string(),
                job: "teacher".to_string(),
                city: "Helsinki".to_string(),
            },
            Fact {
                person_id: 2,
                name: "Person_2".to_string(),
                job: "doctor".to_string(),
                city: "Tokyo".to_string(),
            },
            Fact {
                person_id: 3,
                name: "Person_3".to_string(),
                job: "artist".to_string(),
                city: "Helsinki".to_string(),
            },
        ];

        let city_mates = compute_cross_ref(&facts, 0);
        assert_eq!(city_mates.len(), 2);
        assert!(city_mates.contains(&"Person_1".to_string()));
        assert!(city_mates.contains(&"Person_3".to_string()));
        assert!(!city_mates.contains(&"Person_0".to_string()));

        let city_mates_2 = compute_cross_ref(&facts, 2);
        assert_eq!(city_mates_2.len(), 0);
    }

    #[test]
    fn test_generate_aggregation_tasks() {
        let facts = generate_facts(10, 42);
        let tasks = generate_aggregation_tasks(&facts);

        assert_eq!(tasks.len(), 5);

        // Verify task types
        assert!(matches!(tasks[0], AggregationTask::Count { .. }));
        assert!(matches!(tasks[1], AggregationTask::Distribution { .. }));
        assert!(matches!(tasks[2], AggregationTask::Filter { .. }));
        assert!(matches!(tasks[3], AggregationTask::CrossRef { .. }));
        assert!(matches!(tasks[4], AggregationTask::Temporal { .. }));
    }
}
