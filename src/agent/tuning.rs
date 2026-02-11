//! Benchmark scoring for local SLM profile tuning.
//!
//! The goal is to pick the best speed/accuracy balance under hard reliability
//! constraints so local mode can stay competitive with cloud mode.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SLMProfile {
    pub id: String,
    pub model: String,
    pub ctx_size: usize,
    pub max_tokens: u32,
    pub temperature: f64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct BenchmarkSample {
    /// First-token latency (median) in milliseconds.
    pub ttft_ms: f64,
    /// Throughput in generated tokens per second.
    pub output_toks_per_sec: f64,
    /// Quality/pass-rate score in [0, 1].
    pub quality_score: f64,
    /// Tool-call success rate in [0, 1].
    pub tool_success_rate: f64,
    /// Context overflow error rate in [0, 1].
    pub context_overflow_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProfileMeasurement {
    pub profile: SLMProfile,
    pub sample: BenchmarkSample,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct OptimizationConstraints {
    pub min_quality_score: f64,
    pub min_tool_success_rate: f64,
    pub max_ttft_ms: f64,
    pub max_context_overflow_rate: f64,
}

impl Default for OptimizationConstraints {
    fn default() -> Self {
        Self {
            min_quality_score: 0.72,
            min_tool_success_rate: 0.90,
            max_ttft_ms: 1800.0,
            max_context_overflow_rate: 0.02,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct OptimizationWeights {
    pub quality: f64,
    pub speed: f64,
    pub ttft: f64,
    pub reliability: f64,
}

impl Default for OptimizationWeights {
    fn default() -> Self {
        Self {
            // Emphasize quality first, then latency/throughput.
            quality: 0.60,
            speed: 0.20,
            ttft: 0.10,
            reliability: 0.10,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ScoredProfile {
    pub profile: SLMProfile,
    pub sample: BenchmarkSample,
    pub score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OptimizationInput {
    pub measurements: Vec<ProfileMeasurement>,
    #[serde(default)]
    pub constraints: Option<OptimizationConstraints>,
    #[serde(default)]
    pub weights: Option<OptimizationWeights>,
}

impl OptimizationInput {
    pub fn resolved_constraints(&self) -> OptimizationConstraints {
        self.constraints.unwrap_or_default()
    }

    pub fn resolved_weights(&self) -> OptimizationWeights {
        self.weights.unwrap_or_default()
    }
}

/// Score and rank all feasible profiles (best first).
pub fn score_feasible_profiles(
    measurements: &[ProfileMeasurement],
    constraints: OptimizationConstraints,
    weights: OptimizationWeights,
) -> Vec<ScoredProfile> {
    let feasible: Vec<&ProfileMeasurement> = measurements
        .iter()
        .filter(|m| is_feasible(&m.sample, constraints))
        .collect();

    if feasible.is_empty() {
        return Vec::new();
    }

    let weights = normalize_weights(weights);
    let mins_maxs = compute_metric_ranges(&feasible);

    let mut scored: Vec<ScoredProfile> = feasible
        .iter()
        .map(|m| ScoredProfile {
            profile: m.profile.clone(),
            sample: m.sample,
            score: weighted_score(&m.sample, mins_maxs, constraints, weights),
        })
        .collect();

    // Best-first order.
    scored.sort_by(|a, b| compare_scored_profiles(b, a));
    scored
}

/// Select the best candidate from an optimization input bundle.
pub fn select_optimal_from_input(input: &OptimizationInput) -> Option<ScoredProfile> {
    select_optimal_profile(
        &input.measurements,
        input.resolved_constraints(),
        input.resolved_weights(),
    )
}

/// Select the best candidate among those that satisfy hard constraints.
pub fn select_optimal_profile(
    measurements: &[ProfileMeasurement],
    constraints: OptimizationConstraints,
    weights: OptimizationWeights,
) -> Option<ScoredProfile> {
    score_feasible_profiles(measurements, constraints, weights)
        .into_iter()
        .next()
}

fn is_feasible(sample: &BenchmarkSample, constraints: OptimizationConstraints) -> bool {
    if !sample.quality_score.is_finite()
        || !sample.tool_success_rate.is_finite()
        || !sample.output_toks_per_sec.is_finite()
        || !sample.ttft_ms.is_finite()
        || !sample.context_overflow_rate.is_finite()
    {
        return false;
    }

    if sample.output_toks_per_sec < 0.0
        || sample.ttft_ms < 0.0
        || sample.quality_score < 0.0
        || sample.tool_success_rate < 0.0
        || sample.context_overflow_rate < 0.0
    {
        return false;
    }

    sample.quality_score >= constraints.min_quality_score
        && sample.tool_success_rate >= constraints.min_tool_success_rate
        && sample.ttft_ms <= constraints.max_ttft_ms
        && sample.context_overflow_rate <= constraints.max_context_overflow_rate
}

fn normalize_weights(weights: OptimizationWeights) -> OptimizationWeights {
    let quality = weights.quality.max(0.0);
    let speed = weights.speed.max(0.0);
    let ttft = weights.ttft.max(0.0);
    let reliability = weights.reliability.max(0.0);
    let sum = quality + speed + ttft + reliability;

    if sum <= f64::EPSILON {
        return OptimizationWeights::default();
    }

    OptimizationWeights {
        quality: quality / sum,
        speed: speed / sum,
        ttft: ttft / sum,
        reliability: reliability / sum,
    }
}

#[derive(Debug, Clone, Copy)]
struct MetricRanges {
    speed_min: f64,
    speed_max: f64,
    ttft_min: f64,
    ttft_max: f64,
}

fn compute_metric_ranges(measurements: &[&ProfileMeasurement]) -> MetricRanges {
    let mut speed_min = f64::INFINITY;
    let mut speed_max = f64::NEG_INFINITY;
    let mut ttft_min = f64::INFINITY;
    let mut ttft_max = f64::NEG_INFINITY;

    for m in measurements {
        let s = m.sample;
        speed_min = speed_min.min(s.output_toks_per_sec);
        speed_max = speed_max.max(s.output_toks_per_sec);
        ttft_min = ttft_min.min(s.ttft_ms);
        ttft_max = ttft_max.max(s.ttft_ms);
    }

    MetricRanges {
        speed_min,
        speed_max,
        ttft_min,
        ttft_max,
    }
}

fn normalize_up(value: f64, min: f64, max: f64) -> f64 {
    if (max - min).abs() <= f64::EPSILON {
        return 1.0;
    }
    ((value - min) / (max - min)).clamp(0.0, 1.0)
}

fn normalize_down(value: f64, min: f64, max: f64) -> f64 {
    if (max - min).abs() <= f64::EPSILON {
        return 1.0;
    }
    ((max - value) / (max - min)).clamp(0.0, 1.0)
}

fn weighted_score(
    sample: &BenchmarkSample,
    ranges: MetricRanges,
    constraints: OptimizationConstraints,
    weights: OptimizationWeights,
) -> f64 {
    // Score quality by margin above the minimum accepted bar.
    let quality = normalize_up(sample.quality_score, constraints.min_quality_score, 1.0);
    let speed = normalize_up(
        sample.output_toks_per_sec,
        ranges.speed_min,
        ranges.speed_max,
    );
    let ttft = normalize_down(sample.ttft_ms, ranges.ttft_min, ranges.ttft_max);

    // Reliability balances tool execution quality and context stability.
    let tool = normalize_up(
        sample.tool_success_rate,
        constraints.min_tool_success_rate,
        1.0,
    );
    let overflow = normalize_down(
        sample.context_overflow_rate,
        0.0,
        constraints.max_context_overflow_rate,
    );
    let reliability = (tool + overflow) / 2.0;

    quality * weights.quality
        + speed * weights.speed
        + ttft * weights.ttft
        + reliability * weights.reliability
}

fn compare_scored_profiles(a: &ScoredProfile, b: &ScoredProfile) -> std::cmp::Ordering {
    // Primary: weighted score (descending).
    let primary = a
        .score
        .partial_cmp(&b.score)
        .unwrap_or(std::cmp::Ordering::Equal);
    if primary != std::cmp::Ordering::Equal {
        return primary;
    }

    // Tie-breakers prioritize reliability and usefulness.
    let quality = a
        .sample
        .quality_score
        .partial_cmp(&b.sample.quality_score)
        .unwrap_or(std::cmp::Ordering::Equal);
    if quality != std::cmp::Ordering::Equal {
        return quality;
    }

    let tool = a
        .sample
        .tool_success_rate
        .partial_cmp(&b.sample.tool_success_rate)
        .unwrap_or(std::cmp::Ordering::Equal);
    if tool != std::cmp::Ordering::Equal {
        return tool;
    }

    // Lower TTFT wins.
    let ttft = b
        .sample
        .ttft_ms
        .partial_cmp(&a.sample.ttft_ms)
        .unwrap_or(std::cmp::Ordering::Equal);
    if ttft != std::cmp::Ordering::Equal {
        return ttft;
    }

    let speed = a
        .sample
        .output_toks_per_sec
        .partial_cmp(&b.sample.output_toks_per_sec)
        .unwrap_or(std::cmp::Ordering::Equal);
    if speed != std::cmp::Ordering::Equal {
        return speed;
    }

    // Stable deterministic output.
    b.profile.id.cmp(&a.profile.id)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(
        id: &str,
        quality: f64,
        tool_success: f64,
        ttft_ms: f64,
        toks_per_sec: f64,
        overflow: f64,
    ) -> ProfileMeasurement {
        ProfileMeasurement {
            profile: SLMProfile {
                id: id.to_string(),
                model: "generic-slm".to_string(),
                ctx_size: 16_384,
                max_tokens: 768,
                temperature: 0.3,
            },
            sample: BenchmarkSample {
                ttft_ms,
                output_toks_per_sec: toks_per_sec,
                quality_score: quality,
                tool_success_rate: tool_success,
                context_overflow_rate: overflow,
            },
        }
    }

    #[test]
    fn rejects_profiles_failing_hard_constraints() {
        let candidates = vec![
            m("fast-but-bad", 0.65, 0.95, 500.0, 120.0, 0.0),
            m("balanced", 0.80, 0.96, 900.0, 60.0, 0.0),
        ];

        let best = select_optimal_profile(
            &candidates,
            OptimizationConstraints::default(),
            OptimizationWeights::default(),
        )
        .expect("expected a feasible profile");

        assert_eq!(best.profile.id, "balanced");
    }

    #[test]
    fn returns_none_if_no_profile_meets_constraints() {
        let candidates = vec![
            m("low-quality", 0.60, 0.95, 400.0, 90.0, 0.0),
            m("high-overflow", 0.90, 0.96, 700.0, 55.0, 0.10),
        ];

        let best = select_optimal_profile(
            &candidates,
            OptimizationConstraints::default(),
            OptimizationWeights::default(),
        );

        assert!(best.is_none());
    }

    #[test]
    fn prefers_speed_when_quality_and_reliability_are_close() {
        let candidates = vec![
            m("fast", 0.81, 0.95, 650.0, 95.0, 0.0),
            m("slightly-better-quality", 0.82, 0.95, 1300.0, 40.0, 0.0),
        ];

        let best = select_optimal_profile(
            &candidates,
            OptimizationConstraints::default(),
            OptimizationWeights::default(),
        )
        .expect("expected a feasible profile");

        assert_eq!(best.profile.id, "fast");
    }

    #[test]
    fn prefers_quality_when_gap_is_large() {
        let candidates = vec![
            m("very-fast", 0.76, 0.93, 500.0, 120.0, 0.0),
            m("accurate", 0.91, 0.98, 1000.0, 50.0, 0.0),
        ];

        let best = select_optimal_profile(
            &candidates,
            OptimizationConstraints::default(),
            OptimizationWeights::default(),
        )
        .expect("expected a feasible profile");

        assert_eq!(best.profile.id, "accurate");
    }

    #[test]
    fn penalizes_high_ttft_even_when_throughput_is_good() {
        let candidates = vec![
            m("slow-first-token", 0.84, 0.96, 2200.0, 110.0, 0.0),
            m("snappy", 0.83, 0.95, 700.0, 70.0, 0.0),
        ];

        let best = select_optimal_profile(
            &candidates,
            OptimizationConstraints::default(),
            OptimizationWeights::default(),
        )
        .expect("expected a feasible profile");

        assert_eq!(best.profile.id, "snappy");
    }
}
