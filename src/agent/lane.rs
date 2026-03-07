//! Lane-based policy: Answer vs Action execution lanes.
//!
//! Each [`Lane`] maps to a [`LanePolicy`] containing 5 profile types that
//! control prompt assembly, tool gating, memory budgets, learning, and parsing.

use crate::agent::model_capabilities::ModelSizeClass;
use crate::agent::prompt_contract::PromptSection;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

/// Execution lane: determines the policy profile for a given turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Lane {
    Answer,
    Action,
}

impl Default for Lane {
    fn default() -> Self {
        Lane::Action
    }
}

impl fmt::Display for Lane {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Lane::Answer => write!(f, "answer"),
            Lane::Action => write!(f, "action"),
        }
    }
}

impl FromStr for Lane {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "answer" => Ok(Lane::Answer),
            "action" => Ok(Lane::Action),
            other => Err(format!("unknown lane: {}", other)),
        }
    }
}

impl Lane {
    /// Returns the policy bundle for this lane.
    pub fn policy(&self) -> LanePolicy {
        match self {
            Lane::Answer => LanePolicy {
                prompt: PromptProfile::Answer,
                tools: ToolGateProfile::ReadOnly,
                memory: MemoryProfile {
                    budget_multiplier: 1.5,
                },
                learn: LearnProfile::Full,
                parser: ParserProfile::Default,
            },
            Lane::Action => LanePolicy {
                prompt: PromptProfile::Action,
                tools: ToolGateProfile::SizeClassBased,
                memory: MemoryProfile {
                    budget_multiplier: 1.0,
                },
                learn: LearnProfile::Full,
                parser: ParserProfile::Default,
            },
        }
    }
}

/// Complete policy bundle for a lane.
#[derive(Debug, Clone, Copy)]
pub struct LanePolicy {
    pub prompt: PromptProfile,
    pub tools: ToolGateProfile,
    pub memory: MemoryProfile,
    pub learn: LearnProfile,
    pub parser: ParserProfile,
}

/// Controls which prompt sections are included.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptProfile {
    /// All 13 sections included.
    Action,
    /// Excludes ToolPatterns and BackgroundTasks.
    Answer,
}

impl PromptProfile {
    /// Returns true if the given section should be included for this profile.
    pub fn includes(&self, section: PromptSection) -> bool {
        match self {
            PromptProfile::Action => true,
            PromptProfile::Answer => {
                !matches!(
                    section,
                    PromptSection::ToolPatterns | PromptSection::BackgroundTasks
                )
            }
        }
    }
}

/// Controls tool availability.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolGateProfile {
    /// Use normal size-class-based filtering.
    SizeClassBased,
    /// Force tiny tier (read-only) regardless of model size.
    ReadOnly,
}

impl ToolGateProfile {
    /// Resolve the effective ModelSizeClass for tool gating.
    ///
    /// ReadOnly forces Small (tiny tier). SizeClassBased passes through.
    pub fn effective_size_class(&self, actual: ModelSizeClass) -> ModelSizeClass {
        match self {
            ToolGateProfile::SizeClassBased => actual,
            ToolGateProfile::ReadOnly => ModelSizeClass::Small,
        }
    }
}

/// Controls memory token budget scaling.
#[derive(Debug, Clone, Copy)]
pub struct MemoryProfile {
    pub budget_multiplier: f64,
}

/// Controls learning behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LearnProfile {
    /// Full learning enabled.
    Full,
}

/// Controls response parsing behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParserProfile {
    /// Default parsing.
    Default,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lane_default_is_action() {
        assert_eq!(Lane::default(), Lane::Action);
    }

    #[test]
    fn lane_answer_policy() {
        let policy = Lane::Answer.policy();
        assert_eq!(policy.prompt, PromptProfile::Answer);
        assert_eq!(policy.tools, ToolGateProfile::ReadOnly);
        assert!((policy.memory.budget_multiplier - 1.5).abs() < f64::EPSILON);
        assert_eq!(policy.learn, LearnProfile::Full);
        assert_eq!(policy.parser, ParserProfile::Default);
    }

    #[test]
    fn lane_action_policy() {
        let policy = Lane::Action.policy();
        assert_eq!(policy.prompt, PromptProfile::Action);
        assert_eq!(policy.tools, ToolGateProfile::SizeClassBased);
        assert!((policy.memory.budget_multiplier - 1.0).abs() < f64::EPSILON);
        assert_eq!(policy.learn, LearnProfile::Full);
        assert_eq!(policy.parser, ParserProfile::Default);
    }

    #[test]
    fn prompt_profile_action_includes_all() {
        for section in PromptSection::all() {
            assert!(
                PromptProfile::Action.includes(*section),
                "Action should include {:?}",
                section
            );
        }
    }

    #[test]
    fn prompt_profile_answer_excludes_tool_patterns_and_background() {
        assert!(!PromptProfile::Answer.includes(PromptSection::ToolPatterns));
        assert!(!PromptProfile::Answer.includes(PromptSection::BackgroundTasks));
    }

    #[test]
    fn prompt_profile_answer_includes_other_sections() {
        let excluded = [PromptSection::ToolPatterns, PromptSection::BackgroundTasks];
        for section in PromptSection::all() {
            if excluded.contains(section) {
                continue;
            }
            assert!(
                PromptProfile::Answer.includes(*section),
                "Answer should include {:?}",
                section
            );
        }
    }

    #[test]
    fn lane_serde_roundtrip() {
        let json = serde_json::to_string(&Lane::Answer).unwrap();
        assert_eq!(json, "\"answer\"");
        let back: Lane = serde_json::from_str(&json).unwrap();
        assert_eq!(back, Lane::Answer);

        let json = serde_json::to_string(&Lane::Action).unwrap();
        assert_eq!(json, "\"action\"");
        let back: Lane = serde_json::from_str(&json).unwrap();
        assert_eq!(back, Lane::Action);
    }

    #[test]
    fn lane_from_str() {
        assert_eq!(Lane::from_str("answer").unwrap(), Lane::Answer);
        assert_eq!(Lane::from_str("action").unwrap(), Lane::Action);
        assert_eq!(Lane::from_str("Answer").unwrap(), Lane::Answer);
        assert_eq!(Lane::from_str("ACTION").unwrap(), Lane::Action);
        assert!(Lane::from_str("bogus").is_err());
    }

    #[test]
    fn lane_display() {
        assert_eq!(Lane::Answer.to_string(), "answer");
        assert_eq!(Lane::Action.to_string(), "action");
    }

    #[test]
    fn tool_gate_profile_read_only_forces_small() {
        assert_eq!(
            ToolGateProfile::ReadOnly.effective_size_class(ModelSizeClass::Large),
            ModelSizeClass::Small
        );
        assert_eq!(
            ToolGateProfile::ReadOnly.effective_size_class(ModelSizeClass::Medium),
            ModelSizeClass::Small
        );
    }

    #[test]
    fn tool_gate_profile_size_class_based_passes_through() {
        assert_eq!(
            ToolGateProfile::SizeClassBased.effective_size_class(ModelSizeClass::Large),
            ModelSizeClass::Large
        );
        assert_eq!(
            ToolGateProfile::SizeClassBased.effective_size_class(ModelSizeClass::Small),
            ModelSizeClass::Small
        );
    }

    #[test]
    fn answer_and_action_policies_differ() {
        let answer = Lane::Answer.policy();
        let action = Lane::Action.policy();
        assert_ne!(answer.prompt, action.prompt);
        assert_ne!(answer.tools, action.tools);
        assert!((answer.memory.budget_multiplier - action.memory.budget_multiplier).abs() > f64::EPSILON);
    }
}
