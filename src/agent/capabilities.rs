use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Semantic capabilities that map to groups of tools.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Capability {
    Read,
    Write,
    Execute,
    Http,
    Memory,
    Spawn,
    Skills,
    Cron,
    Message,
    Code,
}

impl Capability {
    /// Tool names this capability grants access to.
    pub fn tool_names(&self) -> &[&str] {
        match self {
            Capability::Read => &["read_file", "list_dir"],
            Capability::Write => &["write_file", "edit_file"],
            Capability::Execute => &["exec"],
            Capability::Http => &["web_search", "web_fetch"],
            Capability::Memory => &["recall", "remember", "session_search"],
            Capability::Spawn => &["spawn"],
            Capability::Skills => &["read_skill"],
            Capability::Cron => &["cron_schedule"],
            Capability::Message => &["message"],
            Capability::Code => &["execute_code"],
        }
    }
}

/// Resolve inherited capabilities: start from parent's capabilities and
/// remove any in the `deny` list.
///
/// When a subagent profile sets `inherit: true`, it starts with the parent's
/// full capability set and narrows it by removing anything in `deny_capabilities`.
pub fn inherit_capabilities(parent_caps: &[Capability], deny: &[Capability]) -> Vec<Capability> {
    parent_caps
        .iter()
        .filter(|c| !deny.contains(c))
        .cloned()
        .collect()
}

/// Resolve a list of capabilities to a deduplicated, sorted list of tool names.
pub fn resolve_capabilities(caps: &[Capability]) -> Vec<String> {
    let mut tools: HashSet<String> = HashSet::new();
    for cap in caps {
        for tool in cap.tool_names() {
            tools.insert(tool.to_string());
        }
    }
    let mut result: Vec<String> = tools.into_iter().collect();
    result.sort();
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_single_capability() {
        let tools = resolve_capabilities(&[Capability::Read]);
        assert!(tools.contains(&"read_file".to_string()));
        assert!(tools.contains(&"list_dir".to_string()));
        assert_eq!(tools.len(), 2);
    }

    #[test]
    fn test_resolve_multiple_capabilities() {
        let tools = resolve_capabilities(&[Capability::Read, Capability::Http]);
        assert!(tools.contains(&"read_file".to_string()));
        assert!(tools.contains(&"web_search".to_string()));
        assert!(tools.contains(&"web_fetch".to_string()));
    }

    #[test]
    fn test_resolve_dedup() {
        // Read specified twice — read_file must appear exactly once
        let tools = resolve_capabilities(&[Capability::Read, Capability::Read]);
        assert_eq!(tools.iter().filter(|t| *t == "read_file").count(), 1);
    }

    #[test]
    fn test_resolve_empty() {
        let tools = resolve_capabilities(&[]);
        assert!(tools.is_empty());
    }

    #[test]
    fn test_resolve_all_capabilities() {
        let all = vec![
            Capability::Read,
            Capability::Write,
            Capability::Execute,
            Capability::Http,
            Capability::Memory,
            Capability::Spawn,
            Capability::Skills,
            Capability::Cron,
            Capability::Message,
            Capability::Code,
        ];
        let tools = resolve_capabilities(&all);
        // Should have all unique tool names
        assert!(tools.len() >= 10); // at least 10 unique tools
    }

    #[test]
    fn test_capability_serde_roundtrip() {
        let cap = Capability::Read;
        let json = serde_json::to_string(&cap).unwrap();
        assert_eq!(json, "\"read\"");
        let parsed: Capability = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, Capability::Read);
    }

    #[test]
    fn test_resolve_sorted() {
        let tools = resolve_capabilities(&[Capability::Http, Capability::Read]);
        // Verify alphabetical order
        for i in 1..tools.len() {
            assert!(tools[i] >= tools[i - 1], "tools should be sorted");
        }
    }
}
