//! Agent profile loader.
//!
//! Loads named agent profiles from `.nanobot/agents/` directories.
//! Each profile is a markdown file with YAML frontmatter that defines
//! a specialized subagent configuration.

use std::collections::HashMap;
use std::path::Path;

use serde::Deserialize;
use tracing::{debug, warn};

use crate::agent::capabilities::{inherit_capabilities, resolve_capabilities, Capability};

/// A loaded agent profile.
#[derive(Debug, Clone)]
pub struct AgentProfile {
    /// Unique name (from frontmatter or filename).
    pub name: String,
    /// Human-readable description of when to use this agent.
    pub description: String,
    /// System prompt (the markdown body after frontmatter).
    pub system_prompt: String,
    /// Allowed tools. None = all tools available.
    pub tools: Option<Vec<String>>,
    /// Model to use. None = inherit from parent.
    pub model: Option<String>,
    /// Max iterations. None = use default.
    pub max_iterations: Option<u32>,
    /// If true, exclude write/edit tools even if listed.
    pub read_only: bool,
    /// When true, capabilities are inherited from the parent context
    /// (minus `deny_capabilities`). Explicit `capabilities` in frontmatter
    /// takes priority over this flag.
    pub inherit: bool,
    /// Capabilities stripped from inherited set when `inherit: true`.
    pub deny_capabilities: Vec<Capability>,
}

/// Raw YAML frontmatter (deserialized from the --- block).
#[derive(Debug, Deserialize)]
struct ProfileFrontmatter {
    name: Option<String>,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    tools: Option<Vec<String>>,
    /// Semantic capability groups. Resolved to tool names and merged with
    /// any explicit `tools` list. Both may be specified simultaneously.
    #[serde(default)]
    capabilities: Option<Vec<Capability>>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    max_iterations: Option<u32>,
    #[serde(default)]
    read_only: bool,
    /// When true, the subagent inherits capabilities from the parent (minus `deny_capabilities`).
    /// Explicit `capabilities` takes priority over `inherit`.
    #[serde(default)]
    inherit: bool,
    /// Capabilities to remove when `inherit: true` is set.
    #[serde(default)]
    deny_capabilities: Option<Vec<Capability>>,
}

/// Parse a markdown file with YAML frontmatter into an AgentProfile.
///
/// Format:
/// ```text
/// ---
/// name: explore
/// description: Fast codebase exploration
/// tools: [read_file, list_dir, exec]
/// model: haiku
/// max_iterations: 10
/// read_only: true
/// ---
/// You are a codebase explorer...
/// ```
pub fn parse_profile(content: &str, fallback_name: &str) -> Option<AgentProfile> {
    let content = content.trim();
    if !content.starts_with("---") {
        warn!("Agent profile missing YAML frontmatter (no opening ---)");
        return None;
    }

    // Find the closing ---
    let after_first = &content[3..];
    let end_idx = after_first.find("\n---")?;
    let yaml_block = &after_first[..end_idx];
    let body_start = 3 + end_idx + 4; // skip past \n---
    let body = if body_start < content.len() {
        content[body_start..].trim()
    } else {
        ""
    };

    let fm: ProfileFrontmatter = match serde_yaml::from_str(yaml_block) {
        Ok(fm) => fm,
        Err(e) => {
            warn!("Failed to parse agent profile frontmatter: {}", e);
            return None;
        }
    };

    let name = fm.name.unwrap_or_else(|| fallback_name.to_string());
    let description = fm.description.unwrap_or_else(|| format!("Agent: {}", name));

    // Merge capability-derived tools with any explicit tool list.
    // If both are absent, the result is None (all tools available).
    // If either is present, produce a merged, sorted, deduplicated list.
    //
    // `inherit: true` is a runtime hint — the actual parent capability list is
    // only known at spawn time, not at parse time.  We store `inherit` and
    // `deny_capabilities` on the profile and leave `tools` as None so that
    // callers can apply `inherit_capabilities(parent, deny)` themselves.
    let resolved_tools = match (fm.tools, fm.capabilities) {
        (None, None) => None,
        (Some(explicit), None) => Some(explicit),
        (None, Some(caps)) => Some(resolve_capabilities(&caps)),
        (Some(explicit), Some(caps)) => {
            let mut merged = resolve_capabilities(&caps);
            for t in explicit {
                if !merged.contains(&t) {
                    merged.push(t);
                }
            }
            merged.sort();
            Some(merged)
        }
    };

    let deny_capabilities = fm.deny_capabilities.unwrap_or_default();

    Some(AgentProfile {
        name,
        description,
        system_prompt: body.to_string(),
        tools: resolved_tools,
        model: fm.model,
        max_iterations: fm.max_iterations,
        read_only: fm.read_only,
        inherit: fm.inherit,
        deny_capabilities,
    })
}

/// Load all agent profiles from a directory.
///
/// Scans for `*.md` files and parses each one.
fn load_from_dir(dir: &Path) -> HashMap<String, AgentProfile> {
    let mut profiles = HashMap::new();

    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return profiles,
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("md") {
            continue;
        }

        let fallback_name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        let content = match std::fs::read_to_string(&path) {
            Ok(c) => c,
            Err(e) => {
                warn!("Failed to read agent profile {:?}: {}", path, e);
                continue;
            }
        };

        if let Some(profile) = parse_profile(&content, &fallback_name) {
            debug!("Loaded agent profile '{}' from {:?}", profile.name, path);
            profiles.insert(profile.name.clone(), profile);
        }
    }

    profiles
}

/// Load agent profiles from all standard locations.
///
/// Priority (higher overrides lower):
/// 1. Project-level: `{workspace}/../.nanobot/agents/` (or workspace itself if it contains agents/)
/// 2. User-level: `~/.nanobot/agents/`
///
/// Returns a merged map where project profiles override user profiles.
pub fn load_profiles(workspace: &Path) -> HashMap<String, AgentProfile> {
    let mut profiles = HashMap::new();

    // User-level: ~/.nanobot/agents/
    if let Some(home) = dirs::home_dir() {
        let user_dir = home.join(".nanobot").join("agents");
        let user_profiles = load_from_dir(&user_dir);
        debug!("Loaded {} user-level agent profiles", user_profiles.len());
        profiles.extend(user_profiles);
    }

    // Project-level: {workspace}/agents/  (workspace is typically ~/.nanobot/workspace)
    let project_dir = workspace.join("agents");
    let project_profiles = load_from_dir(&project_dir);
    debug!(
        "Loaded {} project-level agent profiles",
        project_profiles.len()
    );
    // Project overrides user
    profiles.extend(project_profiles);

    profiles
}

/// Resolve a model alias to a full model name.
///
/// Aliases:
/// - "local" → kept as "local" (handled by provider selection)
/// - "haiku" → "claude-haiku-4-5-20251001"
/// - "sonnet" → "claude-sonnet-4-5-20250929"
/// - "opus" → "claude-opus-4-6"
/// - anything else → passed through as-is
pub fn resolve_model_alias(alias: &str) -> String {
    match alias.to_lowercase().as_str() {
        "haiku" => "claude-haiku-4-5-20251001".to_string(),
        "sonnet" => "claude-sonnet-4-5-20250929".to_string(),
        "opus" => "claude-opus-4-6".to_string(),
        "local" => "local".to_string(),
        other => other.to_string(),
    }
}

/// Get a one-line summary of available profiles for the system prompt.
pub fn profiles_summary(profiles: &HashMap<String, AgentProfile>) -> String {
    if profiles.is_empty() {
        return String::new();
    }

    let mut lines = Vec::new();
    let mut names: Vec<&String> = profiles.keys().collect();
    names.sort();

    for name in names {
        let p = &profiles[name];
        let model_hint = p.model.as_deref().unwrap_or("inherit");
        let ro = if p.read_only { " (read-only)" } else { "" };
        lines.push(format!(
            "- **{}**: {}{} [model: {}]",
            name, p.description, ro, model_hint
        ));
    }

    format!(
        "## Subagent Profiles\n\n\
         Use `spawn(agent=\"name\", task=\"...\")` to delegate work to a specialized subagent.\n\
         Each runs in its own context — your context stays clean.\n\n\
         ### Available Agents\n{}\n\n\
         ### When to Delegate\n\
         - **explore**: Finding files, functions, patterns in large codebases\n\
         - **builder**: Running builds/tests and diagnosing failures\n\
         - **reviewer**: Checking code changes for bugs and style\n\
         - **researcher**: Multi-page web research and synthesis\n\
         - **Any agent**: When the task would burn >1000 tokens of your context on intermediate results\n\n\
         ### Do It Yourself When\n\
         - Reading a small file (<50 lines)\n\
         - Running a quick command you need immediately\n\
         - The answer feeds directly into your next sentence",
        lines.join("\n")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_profile_basic() {
        let content = r#"---
name: explore
description: Fast codebase exploration
tools: [read_file, list_dir, exec]
model: haiku
max_iterations: 10
read_only: true
---
You are a codebase explorer. Search efficiently."#;

        let profile = parse_profile(content, "fallback").unwrap();
        assert_eq!(profile.name, "explore");
        assert_eq!(profile.description, "Fast codebase exploration");
        assert_eq!(
            profile.tools,
            Some(vec![
                "read_file".to_string(),
                "list_dir".to_string(),
                "exec".to_string()
            ])
        );
        assert_eq!(profile.model, Some("haiku".to_string()));
        assert_eq!(profile.max_iterations, Some(10));
        assert!(profile.read_only);
        assert!(profile.system_prompt.contains("codebase explorer"));
    }

    #[test]
    fn test_parse_profile_minimal() {
        let content = r#"---
description: A minimal agent
---
Do stuff."#;

        let profile = parse_profile(content, "minimal").unwrap();
        assert_eq!(profile.name, "minimal"); // falls back to filename
        assert_eq!(profile.description, "A minimal agent");
        assert!(profile.tools.is_none());
        assert!(profile.model.is_none());
        assert!(!profile.read_only);
    }

    #[test]
    fn test_parse_profile_no_frontmatter() {
        let content = "Just some markdown without frontmatter.";
        assert!(parse_profile(content, "test").is_none());
    }

    #[test]
    fn test_resolve_model_alias() {
        assert_eq!(resolve_model_alias("haiku"), "claude-haiku-4-5-20251001");
        assert_eq!(resolve_model_alias("sonnet"), "claude-sonnet-4-5-20250929");
        assert_eq!(resolve_model_alias("opus"), "claude-opus-4-6");
        assert_eq!(resolve_model_alias("Haiku"), "claude-haiku-4-5-20251001"); // case insensitive
        assert_eq!(resolve_model_alias("local"), "local");
        assert_eq!(resolve_model_alias("custom-model-v2"), "custom-model-v2");
    }

    #[test]
    fn test_profile_with_capabilities() {
        let yaml = "---\ncapabilities: [read, http, skills]\n---\nYou are a researcher.";
        let profile = parse_profile(yaml, "researcher").unwrap();
        let tools = profile.tools.expect("capabilities should produce a tools list");
        // read -> [list_dir, read_file], http -> [web_fetch, web_search], skills -> [read_skill]
        assert!(tools.contains(&"read_file".to_string()));
        assert!(tools.contains(&"list_dir".to_string()));
        assert!(tools.contains(&"web_search".to_string()));
        assert!(tools.contains(&"web_fetch".to_string()));
        assert!(tools.contains(&"read_skill".to_string()));
        assert_eq!(tools.len(), 5);
        // Verify sorted
        for i in 1..tools.len() {
            assert!(tools[i] >= tools[i - 1], "tools should be sorted");
        }
    }

    #[test]
    fn test_profile_capabilities_and_explicit_tools_merge() {
        let yaml = "---\ncapabilities: [read]\ntools: [exec, read_file]\n---\nDo stuff.";
        let profile = parse_profile(yaml, "test").unwrap();
        let tools = profile.tools.expect("merged tools should be present");
        // capabilities read -> [list_dir, read_file], plus explicit exec; read_file deduped
        assert!(tools.contains(&"read_file".to_string()));
        assert!(tools.contains(&"list_dir".to_string()));
        assert!(tools.contains(&"exec".to_string()));
        assert_eq!(tools.iter().filter(|t| *t == "read_file").count(), 1, "read_file deduped");
        // Verify sorted
        for i in 1..tools.len() {
            assert!(tools[i] >= tools[i - 1], "tools should be sorted");
        }
    }

    #[test]
    fn test_profile_no_capabilities_no_tools_is_none() {
        let yaml = "---\ndescription: minimal\n---\nDo stuff.";
        let profile = parse_profile(yaml, "minimal").unwrap();
        assert!(profile.tools.is_none(), "neither capabilities nor tools => None means all tools");
    }

    // ----- inherit / deny_capabilities -----

    #[test]
    fn test_profile_inherit_flag_parsed() {
        let yaml = "---\ninherit: true\ndeny_capabilities: [write, execute]\n---\nDo restricted work.";
        let profile = parse_profile(yaml, "restricted").unwrap();
        assert!(profile.inherit, "inherit flag should be true");
        assert!(profile.deny_capabilities.contains(&Capability::Write));
        assert!(profile.deny_capabilities.contains(&Capability::Execute));
        // tools should be None because no explicit capabilities/tools listed
        assert!(profile.tools.is_none());
    }

    #[test]
    fn test_profile_inherit_false_by_default() {
        let yaml = "---\ndescription: normal agent\n---\nDo stuff.";
        let profile = parse_profile(yaml, "normal").unwrap();
        assert!(!profile.inherit);
        assert!(profile.deny_capabilities.is_empty());
    }

    #[test]
    fn test_profile_explicit_capabilities_override_inherit() {
        // When `capabilities` are explicit, they define `tools` regardless of `inherit`.
        let yaml = "---\ncapabilities: [read]\ninherit: true\ndeny_capabilities: [http]\n---\nDo stuff.";
        let profile = parse_profile(yaml, "mixed").unwrap();
        let tools = profile.tools.expect("should have tools from capabilities");
        assert!(tools.contains(&"read_file".to_string()));
        // inherit flag is recorded but explicit capabilities win for tools
        assert!(profile.inherit);
        assert!(profile.deny_capabilities.contains(&Capability::Http));
    }

    #[test]
    fn test_inherit_capabilities_integration() {
        // Simulate a spawn-time resolution: parent has Read + Http + Write,
        // child profile says inherit=true, deny=[write].
        let parent_caps = vec![Capability::Read, Capability::Http, Capability::Write];
        let deny = vec![Capability::Write];
        let resolved = inherit_capabilities(&parent_caps, &deny);
        let tool_names = resolve_capabilities(&resolved);
        assert!(tool_names.contains(&"read_file".to_string()));
        assert!(tool_names.contains(&"web_fetch".to_string()));
        assert!(!tool_names.contains(&"write_file".to_string()));
        assert!(!tool_names.contains(&"edit_file".to_string()));
    }

    #[test]
    fn test_profiles_summary_empty() {
        let profiles = HashMap::new();
        assert_eq!(profiles_summary(&profiles), "");
    }

    #[test]
    fn test_profiles_summary_with_profiles() {
        let mut profiles = HashMap::new();
        profiles.insert(
            "explore".to_string(),
            AgentProfile {
                name: "explore".to_string(),
                description: "Fast codebase exploration".to_string(),
                system_prompt: String::new(),
                tools: Some(vec!["read_file".to_string()]),
                model: Some("haiku".to_string()),
                max_iterations: Some(10),
                read_only: true,
                inherit: false,
                deny_capabilities: vec![],
            },
        );
        let summary = profiles_summary(&profiles);
        assert!(summary.contains("explore"));
        assert!(summary.contains("Fast codebase exploration"));
        assert!(summary.contains("read-only"));
        assert!(summary.contains("haiku"));
    }
}
