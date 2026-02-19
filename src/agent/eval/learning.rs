//! Learning Curve benchmark (SWE-Bench-CL inspired).
//!
//! Measures whether the agent improves over repeated similar tasks.
//! Generates a curriculum of deterministic tasks with verifiable answers
//! and tracks metrics over time.

use serde::{Deserialize, Serialize};

/// A task family that produces deterministic tasks with verifiable answers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TaskFamily {
    /// Math word problems with the same structure, different numbers.
    /// depth = number of chained operations.
    ArithmeticChain { depth: usize },
    /// "Given these facts, answer this question" with varying fact counts.
    FactRetrieval { num_facts: usize },
    /// Multi-step tool use: search → extract → compute → verify.
    ToolChain { num_steps: usize },
}

/// A single generated task with its expected answer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurriculumTask {
    /// Task index in the sequence.
    pub index: usize,
    /// The task family this belongs to.
    pub family: TaskFamily,
    /// The task prompt/question.
    pub prompt: String,
    /// The expected correct answer (for verification).
    pub expected_answer: String,
    /// Difficulty hint (1-5).
    pub difficulty: u8,
}

/// Execution record for a single task in the curriculum.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskExecution {
    /// Which task was executed.
    pub task_index: usize,
    /// Whether the task was completed successfully.
    pub success: bool,
    /// Number of iterations used.
    pub iterations_used: u32,
    /// Cost in USD.
    pub cost_usd: f64,
    /// Duration in milliseconds.
    pub duration_ms: u64,
    /// Number of tool calls made.
    pub tool_calls: u32,
    /// The agent's answer.
    pub agent_answer: String,
}

/// Sliding window metrics for learning curve analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowMetrics {
    /// Window start index.
    pub start: usize,
    /// Window end index (exclusive).
    pub end: usize,
    /// Success rate within this window.
    pub success_rate: f64,
    /// Mean iterations within this window.
    pub mean_iterations: f64,
    /// Mean cost within this window.
    pub mean_cost: f64,
}

/// Full learning curve results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningCurveResults {
    /// Task family used.
    pub family: TaskFamily,
    /// Total tasks in curriculum.
    pub total_tasks: usize,
    /// Total completed.
    pub completed: usize,
    /// Accuracy curve (sliding windows).
    pub accuracy_curve: Vec<WindowMetrics>,
    /// Efficiency curve (sliding windows).
    pub efficiency_curve: Vec<WindowMetrics>,
    /// Cost curve (sliding windows).
    pub cost_curve: Vec<WindowMetrics>,
    /// Forward transfer metric: ratio of performance improvement.
    pub forward_transfer: f64,
    /// Surprise rate: fraction of experiences flagged as surprising.
    pub surprise_rate: f64,
}

/// Simple pseudo-random number generator for deterministic task generation.
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> u64 {
        // Linear congruential generator
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.state
    }

    fn next_range(&mut self, min: i32, max: i32) -> i32 {
        let range = (max - min) as u64;
        (self.next() % range) as i32 + min
    }
}

/// Generate a multi-step arithmetic problem with deterministic seed.
///
/// Example for depth=3: "A store has 15 apples. They receive 23 more. Then they sell 12. How many apples remain?"
/// Answer: "26"
pub fn generate_arithmetic_task(index: usize, depth: usize, seed: u64) -> CurriculumTask {
    let mut rng = SimpleRng::new(seed.wrapping_add(index as u64));

    let initial_value = rng.next_range(10, 50);
    let mut current_value = initial_value;
    let mut steps = Vec::new();

    let subjects = ["apples", "books", "coins", "marbles"];
    let subject = subjects[rng.next_range(0, subjects.len() as i32) as usize];

    steps.push(format!("A store has {} {}", initial_value, subject));

    for i in 0..depth {
        let operation = rng.next_range(0, 3);
        let operand = rng.next_range(5, 25);

        match operation {
            0 => {
                // Addition
                steps.push(format!("They receive {} more", operand));
                current_value += operand;
            }
            1 => {
                // Subtraction (ensure non-negative)
                let subtract_amount = operand.min(current_value - 1);
                steps.push(format!("They sell {}", subtract_amount));
                current_value -= subtract_amount;
            }
            _ => {
                // Multiplication by small number
                let multiplier = rng.next_range(2, 4);
                steps.push(format!("They multiply their stock by {}", multiplier));
                current_value *= multiplier;
            }
        }
    }

    steps.push(format!("How many {} remain?", subject));

    let difficulty = ((depth - 1).min(4) + 1) as u8;

    CurriculumTask {
        index,
        family: TaskFamily::ArithmeticChain { depth },
        prompt: steps.join(". "),
        expected_answer: current_value.to_string(),
        difficulty,
    }
}

/// Generate a fact retrieval task with deterministic seed.
///
/// Example: "Given: Alice is 30. Bob is 25. Carol is 35. Who is the oldest?"
/// Answer: "Carol"
pub fn generate_fact_retrieval_task(index: usize, num_facts: usize, seed: u64) -> CurriculumTask {
    let mut rng = SimpleRng::new(seed.wrapping_add(index as u64));

    let names = [
        "Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace", "Henry",
    ];
    let attribute_types = ["age", "height_cm", "score"];

    let attr_type = attribute_types[rng.next_range(0, attribute_types.len() as i32) as usize];
    let (attr_name, value_range, question_superlative, comparative) = match attr_type {
        "age" => ("age", (20, 60), "oldest", "older"),
        "height_cm" => ("height", (150, 200), "tallest", "taller"),
        _ => ("score", (60, 100), "highest score", "higher score"),
    };

    let mut facts = Vec::new();
    let mut max_value = i32::MIN;
    let mut max_name = "";

    for i in 0..num_facts.min(names.len()) {
        let name = names[i];
        let value = rng.next_range(value_range.0, value_range.1);

        facts.push(format!("{} has {} of {}", name, attr_name, value));

        if value > max_value {
            max_value = value;
            max_name = name;
        }
    }

    let prompt = format!(
        "Given: {}. Who has the {}?",
        facts.join(". "),
        question_superlative
    );

    let difficulty = ((num_facts - 1).min(4) + 1) as u8;

    CurriculumTask {
        index,
        family: TaskFamily::FactRetrieval { num_facts },
        prompt,
        expected_answer: max_name.to_string(),
        difficulty,
    }
}

/// Generate a multi-step tool chain task with deterministic seed.
///
/// Example for steps=3: "Step 1: Find the population of Country_0.
/// Step 2: Divide by 1000. Step 3: Round to nearest integer. What is the result?"
pub fn generate_tool_chain_task(index: usize, num_steps: usize, seed: u64) -> CurriculumTask {
    let mut rng = SimpleRng::new(seed.wrapping_add(index as u64));

    let country_id = rng.next_range(0, 10);
    let population = rng.next_range(10000, 100000) * 1000; // Millions

    let mut steps = Vec::new();
    let mut current_value = population as f64;

    steps.push(format!(
        "Step 1: In the database, Country_{} has population {}",
        country_id, population
    ));

    for i in 1..num_steps {
        let operation = rng.next_range(0, 3);

        match operation {
            0 => {
                // Divide
                let divisor = rng.next_range(100, 1000);
                steps.push(format!("Step {}: Divide by {}", i + 1, divisor));
                current_value /= divisor as f64;
            }
            1 => {
                // Multiply
                let multiplier = rng.next_range(2, 10);
                steps.push(format!("Step {}: Multiply by {}", i + 1, multiplier));
                current_value *= multiplier as f64;
            }
            _ => {
                // Add
                let addend = rng.next_range(100, 1000);
                steps.push(format!("Step {}: Add {}", i + 1, addend));
                current_value += addend as f64;
            }
        }
    }

    steps.push(format!(
        "Step {}: Round to nearest integer. What is the result?",
        num_steps + 1
    ));

    let expected = current_value.round() as i64;
    let difficulty = ((num_steps - 1).min(4) + 1) as u8;

    CurriculumTask {
        index,
        family: TaskFamily::ToolChain { num_steps },
        prompt: steps.join(". "),
        expected_answer: expected.to_string(),
        difficulty,
    }
}

/// Generate the full curriculum by dispatching to the appropriate generator.
pub fn generate_curriculum(
    family: &TaskFamily,
    num_tasks: usize,
    seed: u64,
) -> Vec<CurriculumTask> {
    (0..num_tasks)
        .map(|index| match family {
            TaskFamily::ArithmeticChain { depth } => generate_arithmetic_task(index, *depth, seed),
            TaskFamily::FactRetrieval { num_facts } => {
                generate_fact_retrieval_task(index, *num_facts, seed)
            }
            TaskFamily::ToolChain { num_steps } => {
                generate_tool_chain_task(index, *num_steps, seed)
            }
        })
        .collect()
}

/// Compute sliding window metrics. Each window of `window_size` tasks produces one WindowMetrics.
pub fn compute_sliding_window(
    executions: &[TaskExecution],
    window_size: usize,
) -> Vec<WindowMetrics> {
    if executions.len() < window_size {
        return Vec::new();
    }

    let mut windows = Vec::new();

    for start in 0..=(executions.len() - window_size) {
        let end = start + window_size;
        let window = &executions[start..end];

        let successes = window.iter().filter(|e| e.success).count();
        let success_rate = successes as f64 / window_size as f64;

        let total_iterations: u32 = window.iter().map(|e| e.iterations_used).sum();
        let mean_iterations = total_iterations as f64 / window_size as f64;

        let total_cost: f64 = window.iter().map(|e| e.cost_usd).sum();
        let mean_cost = total_cost / window_size as f64;

        windows.push(WindowMetrics {
            start,
            end,
            success_rate,
            mean_iterations,
            mean_cost,
        });
    }

    windows
}

/// Compare performance of second half vs first half.
/// Returns ratio: second_half_success_rate / first_half_success_rate.
/// Returns 1.0 if first half has 0 success rate (to avoid division by zero).
pub fn compute_forward_transfer(executions: &[TaskExecution]) -> f64 {
    if executions.len() < 2 {
        return 1.0;
    }

    let mid = executions.len() / 2;
    let first_half = &executions[..mid];
    let second_half = &executions[mid..];

    let first_success =
        first_half.iter().filter(|e| e.success).count() as f64 / first_half.len() as f64;
    let second_success =
        second_half.iter().filter(|e| e.success).count() as f64 / second_half.len() as f64;

    if first_success == 0.0 {
        return 1.0;
    }

    second_success / first_success
}

/// Fraction of tasks where iterations_used > 2 * mean_iterations (surprisingly hard).
pub fn compute_surprise_rate(executions: &[TaskExecution]) -> f64 {
    if executions.is_empty() {
        return 0.0;
    }

    let total_iterations: u32 = executions.iter().map(|e| e.iterations_used).sum();
    let mean_iterations = total_iterations as f64 / executions.len() as f64;

    let surprise_threshold = 2.0 * mean_iterations;
    let surprising_count = executions
        .iter()
        .filter(|e| e.iterations_used as f64 > surprise_threshold)
        .count();

    surprising_count as f64 / executions.len() as f64
}

/// Compute all metrics from a set of task executions.
pub fn compute_learning_curve(
    family: &TaskFamily,
    executions: &[TaskExecution],
    window_size: usize,
) -> LearningCurveResults {
    let total_tasks = executions.len();
    let completed = executions.iter().filter(|e| e.success).count();

    let accuracy_curve = compute_sliding_window(executions, window_size);
    let efficiency_curve = accuracy_curve.clone();
    let cost_curve = accuracy_curve.clone();

    let forward_transfer = compute_forward_transfer(executions);
    let surprise_rate = compute_surprise_rate(executions);

    LearningCurveResults {
        family: family.clone(),
        total_tasks,
        completed,
        accuracy_curve,
        efficiency_curve,
        cost_curve,
        forward_transfer,
        surprise_rate,
    }
}

/// Lenient answer matching: trim whitespace, case-insensitive, handle common variations.
pub fn verify_answer(expected: &str, actual: &str) -> bool {
    let expected = expected.trim().to_lowercase();
    let actual = actual.trim().to_lowercase();

    if expected == actual {
        return true;
    }

    // Try parsing as numbers for numeric comparison
    if let (Ok(exp_num), Ok(act_num)) = (expected.parse::<f64>(), actual.parse::<f64>()) {
        return (exp_num - act_num).abs() < 0.001;
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_arithmetic_deterministic() {
        let task1 = generate_arithmetic_task(5, 3, 12345);
        let task2 = generate_arithmetic_task(5, 3, 12345);

        assert_eq!(task1.prompt, task2.prompt);
        assert_eq!(task1.expected_answer, task2.expected_answer);
    }

    #[test]
    fn test_generate_arithmetic_verifiable() {
        let task = generate_arithmetic_task(0, 2, 42);

        // Parse the task to verify the answer manually
        // Task structure: "A store has X items. [operations]. How many items remain?"
        assert!(task.prompt.contains("How many"));
        assert!(!task.expected_answer.is_empty());

        // Verify the answer is a valid number
        let answer: i32 = task
            .expected_answer
            .parse()
            .expect("Answer should be a number");
        assert!(answer >= 0, "Answer should be non-negative");
    }

    #[test]
    fn test_generate_fact_retrieval() {
        let task = generate_fact_retrieval_task(0, 3, 42);

        assert!(task.prompt.contains("Given:"));
        assert!(task.prompt.contains("Who has the"));
        assert!(!task.expected_answer.is_empty());

        // The answer should be one of the standard names
        let valid_names = [
            "Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace", "Henry",
        ];
        assert!(valid_names.contains(&task.expected_answer.as_str()));
    }

    #[test]
    fn test_generate_curriculum_count() {
        let family = TaskFamily::ArithmeticChain { depth: 3 };
        let curriculum = generate_curriculum(&family, 10, 12345);

        assert_eq!(curriculum.len(), 10);

        for (i, task) in curriculum.iter().enumerate() {
            assert_eq!(task.index, i);
            assert_eq!(task.family, family);
        }
    }

    #[test]
    fn test_sliding_window_basic() {
        let executions: Vec<TaskExecution> = (0..10)
            .map(|i| TaskExecution {
                task_index: i,
                success: i % 2 == 0,
                iterations_used: (i + 1) as u32,
                cost_usd: 0.01,
                duration_ms: 1000,
                tool_calls: 2,
                agent_answer: "test".to_string(),
            })
            .collect();

        let windows = compute_sliding_window(&executions, 5);

        // 10 executions with window size 5 should produce 6 windows
        assert_eq!(windows.len(), 6);

        // Check first window
        assert_eq!(windows[0].start, 0);
        assert_eq!(windows[0].end, 5);

        // Check last window
        assert_eq!(windows[5].start, 5);
        assert_eq!(windows[5].end, 10);
    }

    #[test]
    fn test_forward_transfer_improving() {
        let executions: Vec<TaskExecution> = (0..10)
            .map(|i| TaskExecution {
                task_index: i,
                success: i >= 5, // Second half all succeed
                iterations_used: 1,
                cost_usd: 0.01,
                duration_ms: 1000,
                tool_calls: 2,
                agent_answer: "test".to_string(),
            })
            .collect();

        let transfer = compute_forward_transfer(&executions);

        // First half: 0/5 success = 0.0
        // Second half: 5/5 success = 1.0
        // Since first half is 0.0, should return 1.0 to avoid division by zero
        assert_eq!(transfer, 1.0);

        // Test with improving but non-zero first half
        let executions2: Vec<TaskExecution> = (0..10)
            .map(|i| TaskExecution {
                task_index: i,
                success: i >= 3, // First half: 2/5, second half: 5/5
                iterations_used: 1,
                cost_usd: 0.01,
                duration_ms: 1000,
                tool_calls: 2,
                agent_answer: "test".to_string(),
            })
            .collect();

        let transfer2 = compute_forward_transfer(&executions2);

        // First half: 2/5 = 0.4
        // Second half: 5/5 = 1.0
        // Transfer: 1.0 / 0.4 = 2.5
        assert!((transfer2 - 2.5).abs() < 0.001);
    }

    #[test]
    fn test_forward_transfer_constant() {
        // Use a pattern that ensures both halves have the same success rate
        let executions: Vec<TaskExecution> = (0..10)
            .map(|i| TaskExecution {
                task_index: i,
                success: i < 3 || (i >= 5 && i < 8), // First half: 3/5, second half: 3/5
                iterations_used: 1,
                cost_usd: 0.01,
                duration_ms: 1000,
                tool_calls: 2,
                agent_answer: "test".to_string(),
            })
            .collect();

        let transfer = compute_forward_transfer(&executions);

        // Both halves have exactly the same success rate (3/5 = 0.6)
        // Transfer should be exactly 1.0
        assert!((transfer - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_verify_answer_lenient() {
        // Exact match
        assert!(verify_answer("26", "26"));

        // Whitespace
        assert!(verify_answer("26", "  26  "));

        // Decimal equivalence
        assert!(verify_answer("26", "26.0"));
        assert!(verify_answer("26.0", "26"));

        // Case insensitive
        assert!(verify_answer("Carol", "carol"));
        assert!(verify_answer("CAROL", "Carol"));

        // Different values
        assert!(!verify_answer("26", "27"));
        assert!(!verify_answer("Alice", "Bob"));
    }
}
