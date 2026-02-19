use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for the step voter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoterConfig {
    /// Number of parallel completions per step (k).
    pub num_voters: usize,
    /// Margin needed to win (ahead-by-k). Default: 1.
    pub win_margin: usize,
    /// Maximum allowed red-flag (malformed) rate before aborting. Default: 0.5.
    pub max_red_flag_rate: f64,
    /// Temperature for voter completions. Default: 0.7.
    pub temperature: f64,
}

impl Default for VoterConfig {
    fn default() -> Self {
        Self {
            num_voters: 3,
            win_margin: 1,
            max_red_flag_rate: 0.5,
            temperature: 0.7,
        }
    }
}

/// A single voter's response for a step.
#[derive(Debug, Clone)]
pub struct VoterResponse {
    /// The raw text response from the model.
    pub raw: String,
    /// Parsed/normalized answer (for comparison). None if red-flagged.
    pub answer: Option<String>,
    /// Whether this response was red-flagged (malformed).
    pub red_flagged: bool,
    /// Reason for red-flagging, if applicable.
    pub red_flag_reason: Option<String>,
}

/// Result of voting on a single step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoteResult {
    /// The winning answer (None if no consensus or too many red flags).
    pub winner: Option<String>,
    /// Total votes cast (excluding red-flagged).
    pub valid_votes: usize,
    /// Total red-flagged responses.
    pub red_flagged: usize,
    /// Vote counts per answer.
    pub vote_counts: HashMap<String, usize>,
    /// Confidence: winner_votes / valid_votes (0.0 if no winner).
    pub confidence: f64,
    /// Whether the result was decided by margin (true) or majority fallback (false).
    pub decided_by_margin: bool,
}

/// Calibration sample for measuring model accuracy on micro-steps.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationSample {
    /// The prompt given to the model.
    pub prompt: String,
    /// Expected correct answer.
    pub expected: String,
    /// Model's answer.
    pub actual: String,
    /// Whether the model got it right.
    pub correct: bool,
    /// Response time in milliseconds.
    pub latency_ms: u64,
}

/// Calibration result: aggregate statistics from a calibration run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationResult {
    /// Model name used for calibration.
    pub model: String,
    /// Number of samples tested.
    pub num_samples: usize,
    /// Accuracy (p) = correct / num_samples.
    pub accuracy: f64,
    /// Median latency in ms.
    pub median_latency_ms: f64,
    /// Red-flag rate (malformed responses / total).
    pub red_flag_rate: f64,
    /// Individual samples.
    pub samples: Vec<CalibrationSample>,
}

/// Red-flag parser: detect malformed outputs.
///
/// Returns (normalized_answer, is_red_flagged, reason).
pub fn parse_and_validate(raw: &str) -> (Option<String>, bool, Option<String>) {
    let trimmed = raw.trim();

    // Red-flag: empty response
    if trimmed.is_empty() {
        return (None, true, Some("Empty response".to_string()));
    }

    // Red-flag: response is just an error message
    if trimmed.starts_with("Error:")
        || trimmed.starts_with("I'm sorry")
        || trimmed.starts_with("I cannot")
    {
        return (
            None,
            true,
            Some(format!(
                "Error/refusal response: {}",
                &trimmed[..trimmed.len().min(50)]
            )),
        );
    }

    // Red-flag: response exceeds reasonable length (likely hallucination/runaway)
    if trimmed.len() > 10000 {
        return (None, true, Some("Response exceeds 10K chars".to_string()));
    }

    // Red-flag: response contains obvious format errors for structured output
    // (JSON expected but got something else, etc.)
    // For now, accept any non-empty, non-error response as valid

    // Normalize: trim whitespace, lowercase for comparison
    let normalized = trimmed.to_string();
    (Some(normalized), false, None)
}

/// Run the first-to-ahead-by-k voting algorithm on a set of responses.
pub fn vote(responses: &[VoterResponse], config: &VoterConfig) -> VoteResult {
    // Separate valid from red-flagged
    let valid: Vec<&VoterResponse> = responses.iter().filter(|r| !r.red_flagged).collect();
    let red_flagged = responses.len() - valid.len();

    // Check if too many red flags
    if responses.is_empty()
        || (red_flagged as f64 / responses.len() as f64) > config.max_red_flag_rate
    {
        return VoteResult {
            winner: None,
            valid_votes: valid.len(),
            red_flagged,
            vote_counts: HashMap::new(),
            confidence: 0.0,
            decided_by_margin: false,
        };
    }

    // Count votes per answer
    let mut vote_counts: HashMap<String, usize> = HashMap::new();
    for resp in &valid {
        if let Some(ref answer) = resp.answer {
            *vote_counts.entry(answer.clone()).or_insert(0) += 1;
        }
    }

    // First-to-ahead-by-margin: process votes in order
    let mut running_counts: HashMap<String, usize> = HashMap::new();
    let mut decided_by_margin = false;
    let mut early_winner: Option<String> = None;

    for resp in &valid {
        if let Some(ref answer) = resp.answer {
            let count = running_counts.entry(answer.clone()).or_insert(0);
            *count += 1;

            // Check if this answer is ahead by margin over all others
            let my_count = *count;
            let max_other = running_counts
                .iter()
                .filter(|(k, _)| *k != answer)
                .map(|(_, v)| *v)
                .max()
                .unwrap_or(0);

            if my_count >= max_other + config.win_margin && my_count >= 2 {
                early_winner = Some(answer.clone());
                decided_by_margin = true;
                break;
            }
        }
    }

    // Fallback: majority vote
    let winner = early_winner.or_else(|| {
        vote_counts
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(answer, _)| answer.clone())
    });

    let confidence = if let Some(ref w) = winner {
        let winner_votes = vote_counts.get(w).copied().unwrap_or(0);
        if valid.is_empty() {
            0.0
        } else {
            winner_votes as f64 / valid.len() as f64
        }
    } else {
        0.0
    };

    VoteResult {
        winner,
        valid_votes: valid.len(),
        red_flagged,
        vote_counts,
        confidence,
        decided_by_margin,
    }
}

/// Estimate the number of voters needed to achieve target reliability.
///
/// Given model accuracy `p` and desired step reliability `target_p`,
/// returns the minimum k (number of voters) needed for majority-of-k voting.
///
/// Uses MAKER's formula: P(correct) = sum over majority..k of C(k,i) * p^i * (1-p)^(k-i)
pub fn estimate_voters_needed(p: f64, target_p: f64, max_k: usize) -> Option<usize> {
    if p <= 0.5 {
        return None; // Voting can't help if model is worse than random
    }
    if p >= target_p {
        return Some(1); // Model is already good enough
    }

    for k in (1..=max_k).step_by(2) {
        // Odd numbers only for clean majority
        let majority = k / 2 + 1;
        let mut prob_correct = 0.0;

        for i in majority..=k {
            let binom = binomial_coefficient(k, i);
            prob_correct += binom * p.powi(i as i32) * (1.0 - p).powi((k - i) as i32);
        }

        if prob_correct >= target_p {
            return Some(k);
        }
    }

    None // Can't reach target within max_k
}

/// Compute binomial coefficient C(n, k) as f64.
fn binomial_coefficient(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    if k == 0 || k == n {
        return 1.0;
    }
    let k = k.min(n - k); // Optimization: C(n,k) = C(n,n-k)
    let mut result = 1.0;
    for i in 0..k {
        result *= (n - i) as f64 / (i + 1) as f64;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_and_validate_normal() {
        let (answer, red_flagged, reason) = parse_and_validate("  42  ");
        assert_eq!(answer, Some("42".to_string()));
        assert!(!red_flagged);
        assert!(reason.is_none());
    }

    #[test]
    fn test_parse_and_validate_empty() {
        let (answer, red_flagged, reason) = parse_and_validate("   ");
        assert_eq!(answer, None);
        assert!(red_flagged);
        assert_eq!(reason, Some("Empty response".to_string()));
    }

    #[test]
    fn test_parse_and_validate_error() {
        let (answer, red_flagged, reason) = parse_and_validate("Error: something went wrong");
        assert_eq!(answer, None);
        assert!(red_flagged);
        assert!(reason.is_some());
        assert!(reason.unwrap().contains("Error/refusal response"));
    }

    #[test]
    fn test_parse_and_validate_refusal() {
        let (answer, red_flagged, reason) = parse_and_validate("I'm sorry, I can't help with that");
        assert_eq!(answer, None);
        assert!(red_flagged);
        assert!(reason.is_some());
    }

    #[test]
    fn test_parse_and_validate_too_long() {
        let long_string = "x".repeat(10001);
        let (answer, red_flagged, reason) = parse_and_validate(&long_string);
        assert_eq!(answer, None);
        assert!(red_flagged);
        assert_eq!(reason, Some("Response exceeds 10K chars".to_string()));
    }

    #[test]
    fn test_vote_unanimous() {
        let config = VoterConfig::default();
        let responses = vec![
            VoterResponse {
                raw: "42".to_string(),
                answer: Some("42".to_string()),
                red_flagged: false,
                red_flag_reason: None,
            },
            VoterResponse {
                raw: "42".to_string(),
                answer: Some("42".to_string()),
                red_flagged: false,
                red_flag_reason: None,
            },
            VoterResponse {
                raw: "42".to_string(),
                answer: Some("42".to_string()),
                red_flagged: false,
                red_flag_reason: None,
            },
        ];

        let result = vote(&responses, &config);
        assert_eq!(result.winner, Some("42".to_string()));
        assert_eq!(result.valid_votes, 3);
        assert_eq!(result.red_flagged, 0);
        assert_eq!(result.confidence, 1.0);
    }

    #[test]
    fn test_vote_majority() {
        let config = VoterConfig::default();
        let responses = vec![
            VoterResponse {
                raw: "42".to_string(),
                answer: Some("42".to_string()),
                red_flagged: false,
                red_flag_reason: None,
            },
            VoterResponse {
                raw: "42".to_string(),
                answer: Some("42".to_string()),
                red_flagged: false,
                red_flag_reason: None,
            },
            VoterResponse {
                raw: "43".to_string(),
                answer: Some("43".to_string()),
                red_flagged: false,
                red_flag_reason: None,
            },
        ];

        let result = vote(&responses, &config);
        assert_eq!(result.winner, Some("42".to_string()));
        assert_eq!(result.valid_votes, 3);
        assert_eq!(result.red_flagged, 0);
        assert!((result.confidence - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_vote_no_consensus() {
        let config = VoterConfig::default();
        let responses = vec![
            VoterResponse {
                raw: "42".to_string(),
                answer: Some("42".to_string()),
                red_flagged: false,
                red_flag_reason: None,
            },
            VoterResponse {
                raw: "43".to_string(),
                answer: Some("43".to_string()),
                red_flagged: false,
                red_flag_reason: None,
            },
            VoterResponse {
                raw: "44".to_string(),
                answer: Some("44".to_string()),
                red_flagged: false,
                red_flag_reason: None,
            },
        ];

        let result = vote(&responses, &config);
        assert!(result.winner.is_some()); // Picks one arbitrarily
        assert_eq!(result.valid_votes, 3);
        assert_eq!(result.red_flagged, 0);
        assert!((result.confidence - 0.333).abs() < 0.01);
    }

    #[test]
    fn test_vote_ahead_by_margin() {
        let mut config = VoterConfig::default();
        config.win_margin = 1;

        // First two votes go to "42", should win early
        let responses = vec![
            VoterResponse {
                raw: "42".to_string(),
                answer: Some("42".to_string()),
                red_flagged: false,
                red_flag_reason: None,
            },
            VoterResponse {
                raw: "42".to_string(),
                answer: Some("42".to_string()),
                red_flagged: false,
                red_flag_reason: None,
            },
            VoterResponse {
                raw: "43".to_string(),
                answer: Some("43".to_string()),
                red_flagged: false,
                red_flag_reason: None,
            },
        ];

        let result = vote(&responses, &config);
        assert_eq!(result.winner, Some("42".to_string()));
        assert!(result.decided_by_margin);
    }

    #[test]
    fn test_vote_all_red_flagged() {
        let config = VoterConfig::default();
        let responses = vec![
            VoterResponse {
                raw: "".to_string(),
                answer: None,
                red_flagged: true,
                red_flag_reason: Some("Empty".to_string()),
            },
            VoterResponse {
                raw: "Error: failed".to_string(),
                answer: None,
                red_flagged: true,
                red_flag_reason: Some("Error".to_string()),
            },
            VoterResponse {
                raw: "I'm sorry".to_string(),
                answer: None,
                red_flagged: true,
                red_flag_reason: Some("Refusal".to_string()),
            },
        ];

        let result = vote(&responses, &config);
        assert_eq!(result.winner, None);
        assert_eq!(result.valid_votes, 0);
        assert_eq!(result.red_flagged, 3);
    }

    #[test]
    fn test_vote_high_red_flag_rate() {
        let config = VoterConfig::default();
        let responses = vec![
            VoterResponse {
                raw: "42".to_string(),
                answer: Some("42".to_string()),
                red_flagged: false,
                red_flag_reason: None,
            },
            VoterResponse {
                raw: "".to_string(),
                answer: None,
                red_flagged: true,
                red_flag_reason: Some("Empty".to_string()),
            },
            VoterResponse {
                raw: "Error".to_string(),
                answer: None,
                red_flagged: true,
                red_flag_reason: Some("Error".to_string()),
            },
        ];

        let result = vote(&responses, &config);
        // Red-flag rate is 2/3 = 0.666, exceeds 0.5 threshold
        assert_eq!(result.winner, None);
        assert_eq!(result.red_flagged, 2);
    }

    #[test]
    fn test_vote_empty_responses() {
        let config = VoterConfig::default();
        let responses: Vec<VoterResponse> = vec![];

        let result = vote(&responses, &config);
        assert_eq!(result.winner, None);
        assert_eq!(result.valid_votes, 0);
        assert_eq!(result.red_flagged, 0);
        assert_eq!(result.confidence, 0.0);
    }

    #[test]
    fn test_estimate_voters_p90() {
        // With p=0.90, to get to 0.999 we should need k=5 or 7
        let k = estimate_voters_needed(0.90, 0.999, 15);
        assert!(k.is_some());
        let k = k.unwrap();
        assert!(k >= 5 && k <= 9); // Should be in this range
    }

    #[test]
    fn test_estimate_voters_p50_fails() {
        // With p=0.50, voting can't help
        let k = estimate_voters_needed(0.50, 0.999, 15);
        assert!(k.is_none());
    }

    #[test]
    fn test_estimate_voters_perfect() {
        // With p=0.999, we're already there
        let k = estimate_voters_needed(0.999, 0.999, 15);
        assert_eq!(k, Some(1));
    }

    #[test]
    fn test_estimate_voters_below_random() {
        // With p=0.3, voting can't help
        let k = estimate_voters_needed(0.3, 0.999, 15);
        assert!(k.is_none());
    }

    #[test]
    fn test_binomial_coefficient() {
        assert_eq!(binomial_coefficient(5, 2), 10.0);
        assert_eq!(binomial_coefficient(10, 3), 120.0);
        assert_eq!(binomial_coefficient(5, 0), 1.0);
        assert_eq!(binomial_coefficient(5, 5), 1.0);
        assert_eq!(binomial_coefficient(3, 4), 0.0); // k > n
    }

    #[test]
    fn test_binomial_symmetry() {
        // C(n, k) = C(n, n-k)
        assert_eq!(binomial_coefficient(10, 3), binomial_coefficient(10, 7));
        assert_eq!(binomial_coefficient(8, 2), binomial_coefficient(8, 6));
    }
}
