#![allow(dead_code)]
use crate::agent::step_voter::{vote, VoteResult, VoterConfig, VoterResponse};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for confidence-gated voting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceGateConfig {
    /// Number of pilot voters (initial cheap check). Default: 2.
    pub pilot_voters: usize,
    /// Full voter config (used when escalating). Uses step_voter's VoterConfig.
    pub full_config: VoterConfig,
    /// Entropy threshold below which we accept the pilot answer. Default: 0.3.
    /// Lower = more confident required to skip full voting.
    pub entropy_threshold: f64,
    /// Top-1/top-2 margin threshold. If top answer leads by this margin, accept. Default: 0.5.
    pub margin_threshold: f64,
    /// Minimum confidence to accept pilot result. Default: 0.8.
    pub min_pilot_confidence: f64,
}

impl Default for ConfidenceGateConfig {
    fn default() -> Self {
        Self {
            pilot_voters: 2,
            full_config: VoterConfig::default(),
            entropy_threshold: 0.3,
            margin_threshold: 0.5,
            min_pilot_confidence: 0.8,
        }
    }
}

/// Decision from the confidence gate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GateDecision {
    /// Accepted pilot answer (low compute).
    AcceptPilot {
        result: VoteResult,
        entropy: f64,
        margin: f64,
    },
    /// Escalated to full voting (high compute).
    Escalated {
        pilot_result: VoteResult,
        full_result: VoteResult,
        pilot_entropy: f64,
        pilot_margin: f64,
    },
}

impl GateDecision {
    /// Get the final vote result regardless of decision path.
    pub fn result(&self) -> &VoteResult {
        match self {
            GateDecision::AcceptPilot { result, .. } => result,
            GateDecision::Escalated { full_result, .. } => full_result,
        }
    }

    /// Whether the gate accepted the pilot (saved compute).
    pub fn was_accepted_early(&self) -> bool {
        matches!(self, GateDecision::AcceptPilot { .. })
    }

    /// Total voters used.
    pub fn total_voters_used(&self) -> usize {
        match self {
            GateDecision::AcceptPilot { result, .. } => result.valid_votes + result.red_flagged,
            GateDecision::Escalated { full_result, .. } => {
                // full_result already includes all voters (pilot + extra)
                full_result.valid_votes + full_result.red_flagged
            }
        }
    }
}

/// Compute Shannon entropy of a vote distribution.
///
/// Returns 0.0 for unanimous votes, higher for more disagreement.
/// Normalized to [0, 1] by dividing by log2(num_candidates).
pub fn vote_entropy(vote_counts: &HashMap<String, usize>) -> f64 {
    let total: usize = vote_counts.values().sum();
    if total == 0 {
        return 0.0;
    }

    let total_f = total as f64;
    let mut entropy = 0.0;

    for &count in vote_counts.values() {
        if count > 0 {
            let p = count as f64 / total_f;
            entropy -= p * p.log2();
        }
    }

    // Normalize by max possible entropy (uniform distribution)
    let num_candidates = vote_counts.len();
    if num_candidates <= 1 {
        return 0.0;
    }
    let max_entropy = (num_candidates as f64).log2();
    if max_entropy > 0.0 {
        entropy / max_entropy
    } else {
        0.0
    }
}

/// Compute the margin between the top-1 and top-2 vote counts.
///
/// Returns 1.0 if only one candidate, 0.0 if tied.
pub fn top_margin(vote_counts: &HashMap<String, usize>) -> f64 {
    let total: usize = vote_counts.values().sum();
    if total == 0 {
        return 0.0;
    }

    let mut counts: Vec<usize> = vote_counts.values().copied().collect();
    counts.sort_unstable_by(|a, b| b.cmp(a));

    let top1 = counts.first().copied().unwrap_or(0) as f64;
    let top2 = counts.get(1).copied().unwrap_or(0) as f64;

    (top1 - top2) / total as f64
}

/// Run confidence-gated voting on a set of responses.
///
/// Takes pilot_responses (small initial set) and extra_responses (used only if escalating).
/// The caller is responsible for generating responses — this function just decides.
pub fn gated_vote(
    pilot_responses: &[VoterResponse],
    extra_responses: &[VoterResponse],
    config: &ConfidenceGateConfig,
) -> GateDecision {
    // Step 1: Vote on pilot responses
    let pilot_result = vote(
        pilot_responses,
        &VoterConfig {
            num_voters: config.pilot_voters,
            ..config.full_config.clone()
        },
    );

    // Step 2: Compute confidence metrics
    let entropy = vote_entropy(&pilot_result.vote_counts);
    let margin = top_margin(&pilot_result.vote_counts);

    // Step 3: Decide whether to accept or escalate
    let should_accept = pilot_result.winner.is_some()
        && entropy < config.entropy_threshold
        && margin >= config.margin_threshold
        && pilot_result.confidence >= config.min_pilot_confidence;

    if should_accept {
        GateDecision::AcceptPilot {
            result: pilot_result,
            entropy,
            margin,
        }
    } else {
        // Escalate: combine pilot + extra responses for full voting
        let mut all_responses: Vec<VoterResponse> = pilot_responses.to_vec();
        all_responses.extend_from_slice(extra_responses);

        let full_result = vote(&all_responses, &config.full_config);

        GateDecision::Escalated {
            pilot_result,
            full_result,
            pilot_entropy: entropy,
            pilot_margin: margin,
        }
    }
}

/// Statistics tracker for gate decisions over time.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GateStats {
    /// Total decisions made.
    pub total_decisions: usize,
    /// Decisions accepted at pilot stage (saved compute).
    pub pilot_accepted: usize,
    /// Decisions that required full voting.
    pub escalated: usize,
    /// Total voters used across all decisions.
    pub total_voters_used: usize,
    /// Theoretical voters if always using full voting.
    pub theoretical_full_voters: usize,
}

impl GateStats {
    /// Record a gate decision.
    pub fn record(&mut self, decision: &GateDecision, full_k: usize) {
        self.total_decisions += 1;
        self.total_voters_used += decision.total_voters_used();
        self.theoretical_full_voters += full_k;

        match decision {
            GateDecision::AcceptPilot { .. } => self.pilot_accepted += 1,
            GateDecision::Escalated { .. } => self.escalated += 1,
        }
    }

    /// Compute savings ratio: actual_voters / theoretical_voters.
    /// Lower is better. 1.0 = no savings. 0.5 = 2x savings.
    pub fn savings_ratio(&self) -> f64 {
        if self.theoretical_full_voters == 0 {
            return 1.0;
        }
        self.total_voters_used as f64 / self.theoretical_full_voters as f64
    }

    /// Acceptance rate: fraction of decisions accepted at pilot stage.
    pub fn acceptance_rate(&self) -> f64 {
        if self.total_decisions == 0 {
            return 0.0;
        }
        self.pilot_accepted as f64 / self.total_decisions as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_response(answer: &str) -> VoterResponse {
        VoterResponse {
            raw: answer.to_string(),
            answer: Some(answer.to_string()),
            red_flagged: false,
            red_flag_reason: None,
        }
    }

    #[test]
    fn test_vote_entropy_unanimous() {
        let mut counts = HashMap::new();
        counts.insert("42".to_string(), 5);
        let entropy = vote_entropy(&counts);
        assert_eq!(entropy, 0.0);
    }

    #[test]
    fn test_vote_entropy_split() {
        let mut counts = HashMap::new();
        counts.insert("42".to_string(), 2);
        counts.insert("43".to_string(), 2);
        let entropy = vote_entropy(&counts);
        assert_eq!(entropy, 1.0); // Perfect uncertainty between two options
    }

    #[test]
    fn test_vote_entropy_majority() {
        let mut counts = HashMap::new();
        counts.insert("42".to_string(), 2);
        counts.insert("43".to_string(), 1);
        let entropy = vote_entropy(&counts);
        assert!(entropy > 0.0 && entropy < 1.0);
    }

    #[test]
    fn test_vote_entropy_empty() {
        let counts = HashMap::new();
        let entropy = vote_entropy(&counts);
        assert_eq!(entropy, 0.0);
    }

    #[test]
    fn test_top_margin_unanimous() {
        let mut counts = HashMap::new();
        counts.insert("42".to_string(), 3);
        let margin = top_margin(&counts);
        assert_eq!(margin, 1.0);
    }

    #[test]
    fn test_top_margin_tied() {
        let mut counts = HashMap::new();
        counts.insert("42".to_string(), 2);
        counts.insert("43".to_string(), 2);
        let margin = top_margin(&counts);
        assert_eq!(margin, 0.0);
    }

    #[test]
    fn test_top_margin_majority() {
        let mut counts = HashMap::new();
        counts.insert("42".to_string(), 3);
        counts.insert("43".to_string(), 1);
        let margin = top_margin(&counts);
        assert_eq!(margin, 0.5); // (3-1)/4 = 0.5
    }

    #[test]
    fn test_gated_vote_accepts_confident_pilot() {
        let config = ConfidenceGateConfig::default();

        // Two unanimous pilot votes
        let pilot = vec![make_response("42"), make_response("42")];
        let extra = vec![make_response("42"), make_response("42")];

        let decision = gated_vote(&pilot, &extra, &config);

        assert!(decision.was_accepted_early());
        assert_eq!(decision.result().winner, Some("42".to_string()));
        assert_eq!(decision.total_voters_used(), 2);

        if let GateDecision::AcceptPilot {
            entropy, margin, ..
        } = decision
        {
            assert_eq!(entropy, 0.0);
            assert_eq!(margin, 1.0);
        } else {
            panic!("Expected AcceptPilot");
        }
    }

    #[test]
    fn test_gated_vote_escalates_split_pilot() {
        let config = ConfidenceGateConfig::default();

        // Split pilot vote (1:1)
        let pilot = vec![make_response("42"), make_response("43")];
        let extra = vec![make_response("42"), make_response("42")];

        let decision = gated_vote(&pilot, &extra, &config);

        assert!(!decision.was_accepted_early());
        assert_eq!(decision.total_voters_used(), 4); // 2 pilot + 2 extra

        if let GateDecision::Escalated { full_result, .. } = decision {
            // With all 4 votes, "42" should win (3:1)
            assert_eq!(full_result.winner, Some("42".to_string()));
        } else {
            panic!("Expected Escalated");
        }
    }

    #[test]
    fn test_gated_vote_escalates_low_margin() {
        let mut config = ConfidenceGateConfig::default();
        config.margin_threshold = 0.6; // Require large margin

        // Pilot has small margin (2:1, margin = 0.333)
        let pilot = vec![
            make_response("42"),
            make_response("42"),
            make_response("43"),
        ];
        let extra = vec![make_response("42")];

        let decision = gated_vote(&pilot, &extra, &config);

        // Margin 0.333 < 0.6, should escalate
        assert!(!decision.was_accepted_early());
    }

    #[test]
    fn test_gate_stats_tracking() {
        let mut stats = GateStats::default();
        let config = ConfidenceGateConfig::default();

        // Simulate one accepted decision
        let pilot = vec![make_response("42"), make_response("42")];
        let extra = vec![make_response("42")];
        let decision1 = gated_vote(&pilot, &extra, &config);
        stats.record(&decision1, 5);

        assert_eq!(stats.total_decisions, 1);
        assert_eq!(stats.pilot_accepted, 1);
        assert_eq!(stats.escalated, 0);
        assert_eq!(stats.total_voters_used, 2);
        assert_eq!(stats.theoretical_full_voters, 5);

        // Simulate one escalated decision
        let pilot2 = vec![make_response("42"), make_response("43")];
        let extra2 = vec![
            make_response("42"),
            make_response("42"),
            make_response("42"),
        ];
        let decision2 = gated_vote(&pilot2, &extra2, &config);
        stats.record(&decision2, 5);

        assert_eq!(stats.total_decisions, 2);
        assert_eq!(stats.pilot_accepted, 1);
        assert_eq!(stats.escalated, 1);
        assert_eq!(stats.total_voters_used, 2 + 5); // 2 from accepted pilot1, 5 total from escalated2 (2 pilot + 3 extra)
        assert_eq!(stats.theoretical_full_voters, 10);
    }

    #[test]
    fn test_gate_stats_savings_with_all_accepted() {
        let mut stats = GateStats::default();
        let config = ConfidenceGateConfig::default();

        // All decisions accepted at pilot
        for _ in 0..10 {
            let pilot = vec![make_response("42"), make_response("42")];
            let extra = vec![];
            let decision = gated_vote(&pilot, &extra, &config);
            stats.record(&decision, 5);
        }

        assert_eq!(stats.acceptance_rate(), 1.0);
        assert_eq!(stats.savings_ratio(), 0.4); // 20 used / 50 theoretical = 0.4
    }

    #[test]
    fn test_gate_decision_accessors() {
        let config = ConfidenceGateConfig::default();

        // Accepted pilot
        let pilot = vec![make_response("42"), make_response("42")];
        let extra = vec![make_response("42")];
        let decision = gated_vote(&pilot, &extra, &config);

        assert_eq!(decision.result().winner, Some("42".to_string()));
        assert!(decision.was_accepted_early());
        assert_eq!(decision.total_voters_used(), 2);

        // Escalated
        let pilot2 = vec![make_response("42"), make_response("43")];
        let extra2 = vec![make_response("42"), make_response("42")];
        let decision2 = gated_vote(&pilot2, &extra2, &config);

        assert!(!decision2.was_accepted_early());
        assert_eq!(decision2.total_voters_used(), 4);
    }

    #[test]
    fn test_confidence_gate_config_default() {
        let config = ConfidenceGateConfig::default();

        assert_eq!(config.pilot_voters, 2);
        assert_eq!(config.entropy_threshold, 0.3);
        assert_eq!(config.margin_threshold, 0.5);
        assert_eq!(config.min_pilot_confidence, 0.8);
        assert_eq!(config.full_config.num_voters, 3);
    }
}
