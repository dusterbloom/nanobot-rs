//! Tool gating by model size class.
//!
//! Maps [`ModelSizeClass`] to capability-based tool tiers. Small models get
//! a restricted "tiny" set, medium models get a "balanced" set, and large
//! models get everything (no filtering).

use crate::agent::capabilities::{resolve_capabilities, Capability};
use crate::agent::model_capabilities::ModelSizeClass;

/// Returns the capability set for tiny (small model) tier.
fn tiny_capabilities() -> &'static [Capability] {
    &[Capability::Read, Capability::Http, Capability::Skills]
}

/// Returns the capability set for balanced (medium model) tier.
fn balanced_capabilities() -> &'static [Capability] {
    &[
        Capability::Read,
        Capability::Http,
        Capability::Skills,
        Capability::Write,
        Capability::Memory,
        Capability::Execute,
    ]
}

/// Gates tool availability based on model size class.
///
/// When a config override is provided, it takes precedence over the
/// size-class-based tier for all size classes (including Large).
pub struct ToolGate;

impl ToolGate {
    /// Filter tools for a given model size class.
    ///
    /// Returns `Some(names)` when filtering should be applied, or `None`
    /// when all tools should be available (Large models without override).
    pub fn filter(
        size_class: ModelSizeClass,
        config_override: Option<&[String]>,
    ) -> Option<Vec<String>> {
        if let Some(names) = config_override {
            return Some(names.to_vec());
        }
        match size_class {
            ModelSizeClass::Small => Some(resolve_capabilities(tiny_capabilities())),
            ModelSizeClass::Medium => Some(resolve_capabilities(balanced_capabilities())),
            ModelSizeClass::Large => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_small_returns_tiny_tools() {
        let tools = ToolGate::filter(ModelSizeClass::Small, None).unwrap();
        // Should contain exactly the tools from Read + Http + Skills
        let expected: HashSet<String> = resolve_capabilities(tiny_capabilities())
            .into_iter()
            .collect();
        let actual: HashSet<String> = tools.into_iter().collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_small_contains_expected_tools() {
        let tools = ToolGate::filter(ModelSizeClass::Small, None).unwrap();
        // Verify specific tools from each capability group
        assert!(
            tools.contains(&"read_file".to_string()),
            "Read -> read_file"
        );
        assert!(tools.contains(&"list_dir".to_string()), "Read -> list_dir");
        assert!(
            tools.contains(&"web_search".to_string()),
            "Http -> web_search"
        );
        assert!(tools.contains(&"browser".to_string()), "Http -> browser");
        assert!(
            tools.contains(&"read_skill".to_string()),
            "Skills -> read_skill"
        );
    }

    #[test]
    fn test_medium_returns_balanced_tools() {
        let tools = ToolGate::filter(ModelSizeClass::Medium, None).unwrap();
        let expected: HashSet<String> = resolve_capabilities(balanced_capabilities())
            .into_iter()
            .collect();
        let actual: HashSet<String> = tools.into_iter().collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_medium_contains_write_and_execute() {
        let tools = ToolGate::filter(ModelSizeClass::Medium, None).unwrap();
        assert!(
            tools.contains(&"write_file".to_string()),
            "Write -> write_file"
        );
        assert!(tools.contains(&"exec".to_string()), "Execute -> exec");
        assert!(tools.contains(&"recall".to_string()), "Memory -> recall");
        assert!(
            tools.contains(&"remember".to_string()),
            "Memory -> remember"
        );
    }

    #[test]
    fn test_large_returns_none() {
        let result = ToolGate::filter(ModelSizeClass::Large, None);
        assert!(result.is_none(), "Large models should get all tools (None)");
    }

    #[test]
    fn test_config_override_small() {
        let override_list = vec!["foo".to_string()];
        let tools = ToolGate::filter(ModelSizeClass::Small, Some(&override_list)).unwrap();
        assert_eq!(tools, vec!["foo".to_string()]);
    }

    #[test]
    fn test_config_override_large() {
        let override_list = vec!["bar".to_string()];
        let tools = ToolGate::filter(ModelSizeClass::Large, Some(&override_list)).unwrap();
        assert_eq!(tools, vec!["bar".to_string()]);
    }

    #[test]
    fn test_tier_tests_are_capability_based() {
        // Verify that tiers resolve dynamically from capabilities, not hardcoded counts.
        // If someone adds a tool to Capability::Read, tiny tier should include it.
        let tiny_tools = resolve_capabilities(tiny_capabilities());
        let read_tools: Vec<String> = Capability::Read
            .tool_names()
            .iter()
            .map(|s| s.to_string())
            .collect();
        for tool in &read_tools {
            assert!(
                tiny_tools.contains(tool),
                "Tiny tier should contain Read tool: {}",
                tool
            );
        }
    }
}
