//! Model-specific instruction profiles for prompt engineering.
//!
//! Profiles are loaded from a YAML file and resolved at runtime by matching
//! the active model name (via glob pattern) and the current task kind.
//!
//! Resolution order: `base` messages → matching `model_profiles` → matching
//! `task_profiles`. All matching sections are concatenated in that order.

use serde::Deserialize;
use std::path::Path;

/// A single instruction message to inject into the prompt.
#[derive(Debug, Clone, Deserialize)]
pub struct InstructionMessage {
    /// Role: "system" or "developer"
    pub role: String,
    /// The content to inject
    pub content: String,
}

/// A model-specific profile matched by glob pattern.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelProfile {
    /// Glob pattern to match model names (e.g., "qwen*", "deepseek*", "gemma*")
    pub pattern: String,
    /// Messages to inject when this profile matches
    pub messages: Vec<InstructionMessage>,
}

/// A task-specific profile matched by task kind.
#[derive(Debug, Clone, Deserialize)]
pub struct TaskProfile {
    /// Task kind to match (e.g., "main", "routing", "specialist", "subagent")
    pub kind: String,
    /// Messages to inject when this profile matches
    pub messages: Vec<InstructionMessage>,
}

/// Root instruction profile configuration.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct InstructionProfiles {
    /// Base messages applied to all models and tasks
    #[serde(default)]
    pub base: Vec<InstructionMessage>,
    /// Model-specific profiles
    #[serde(default)]
    pub model_profiles: Vec<ModelProfile>,
    /// Task-specific profiles
    #[serde(default)]
    pub task_profiles: Vec<TaskProfile>,
}

impl InstructionProfiles {
    /// Load profiles from a YAML file path.
    pub fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let profiles: InstructionProfiles = serde_yaml::from_str(&content)?;
        Ok(profiles)
    }

    /// Resolve messages for a given model name and task kind.
    ///
    /// Returns base + matching model profiles + matching task profiles, in that
    /// order. Multiple model profiles can match (e.g., both `"qwen*"` and
    /// `"*coder*"`); all are included in declaration order.
    pub fn resolve(&self, model_name: &str, task_kind: &str) -> Vec<InstructionMessage> {
        let mut result = self.base.clone();

        // Match model profiles (all matching patterns are included).
        for profile in &self.model_profiles {
            if glob_match(&profile.pattern, model_name) {
                result.extend(profile.messages.clone());
            }
        }

        // Match task profiles (exact kind match).
        for profile in &self.task_profiles {
            if profile.kind == task_kind {
                result.extend(profile.messages.clone());
            }
        }

        result
    }
}

/// Simple glob matching supporting `*` (zero or more chars) and `?` (exactly
/// one char) wildcards. Matching is case-insensitive.
fn glob_match(pattern: &str, text: &str) -> bool {
    let pattern = pattern.to_lowercase();
    let text = text.to_lowercase();
    glob_match_recursive(pattern.as_bytes(), text.as_bytes())
}

fn glob_match_recursive(pattern: &[u8], text: &[u8]) -> bool {
    match (pattern.first(), text.first()) {
        (None, None) => true,
        (Some(b'*'), _) => {
            // `*` matches zero or more characters: try consuming zero (advance
            // pattern only) or one character from text (keep `*` in pattern).
            glob_match_recursive(&pattern[1..], text)
                || (!text.is_empty() && glob_match_recursive(pattern, &text[1..]))
        }
        (Some(b'?'), Some(_)) => glob_match_recursive(&pattern[1..], &text[1..]),
        (Some(a), Some(b)) if a == b => glob_match_recursive(&pattern[1..], &text[1..]),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ----- glob_match -----

    #[test]
    fn test_glob_match_exact() {
        assert!(glob_match("deepseek-r1", "deepseek-r1"));
        assert!(!glob_match("deepseek-r1", "qwen-2.5"));
    }

    #[test]
    fn test_glob_match_star() {
        assert!(glob_match("qwen*", "qwen-2.5-coder"));
        assert!(glob_match("*coder*", "qwen-2.5-coder-32b"));
        assert!(glob_match("deep*r1", "deepseek-r1"));
        assert!(!glob_match("qwen*", "deepseek-r1"));
    }

    #[test]
    fn test_glob_match_question() {
        assert!(glob_match("qwen-?.5", "qwen-2.5"));
        assert!(!glob_match("qwen-?.5", "qwen-25.5"));
    }

    #[test]
    fn test_glob_case_insensitive() {
        assert!(glob_match("Qwen*", "qwen-2.5"));
        assert!(glob_match("qwen*", "Qwen-2.5"));
    }

    #[test]
    fn test_glob_star_matches_empty() {
        assert!(glob_match("qwen*", "qwen"));
        assert!(glob_match("*", "anything"));
        assert!(glob_match("*", ""));
    }

    #[test]
    fn test_glob_no_match_trailing_chars() {
        assert!(!glob_match("qwen", "qwen-extra"));
    }

    // ----- serde / YAML loading -----

    #[test]
    fn test_load_from_yaml() {
        let yaml = r#"
base:
  - role: developer
    content: "Always respond concisely."

model_profiles:
  - pattern: "qwen*"
    messages:
      - role: developer
        content: "Use strict JSON for tool calls. Never embed tool calls in prose."
  - pattern: "deepseek*"
    messages:
      - role: developer
        content: "After </think>, give your response directly."

task_profiles:
  - kind: routing
    messages:
      - role: developer
        content: "You are a routing agent. Output only the tool name."
  - kind: specialist
    messages:
      - role: developer
        content: "Focus on the delegated task only."
"#;
        let profiles: InstructionProfiles = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(profiles.base.len(), 1);
        assert_eq!(profiles.model_profiles.len(), 2);
        assert_eq!(profiles.task_profiles.len(), 2);
    }

    #[test]
    fn test_empty_yaml_deserializes_to_default() {
        let yaml = r#"{}"#;
        let profiles: InstructionProfiles = serde_yaml::from_str(yaml).unwrap();
        assert!(profiles.base.is_empty());
        assert!(profiles.model_profiles.is_empty());
        assert!(profiles.task_profiles.is_empty());
    }

    // ----- resolve -----

    #[test]
    fn test_resolve_base_only() {
        let yaml = r#"
base:
  - role: developer
    content: "Be concise."
"#;
        let profiles: InstructionProfiles = serde_yaml::from_str(yaml).unwrap();
        let msgs = profiles.resolve("some-model", "main");
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].content, "Be concise.");
    }

    #[test]
    fn test_resolve_with_model_match() {
        let yaml = r#"
base:
  - role: developer
    content: "Base instruction."
model_profiles:
  - pattern: "qwen*"
    messages:
      - role: developer
        content: "Qwen-specific."
  - pattern: "deepseek*"
    messages:
      - role: developer
        content: "DeepSeek-specific."
"#;
        let profiles: InstructionProfiles = serde_yaml::from_str(yaml).unwrap();
        let msgs = profiles.resolve("qwen-2.5-coder", "main");
        assert_eq!(msgs.len(), 2); // base + qwen
        assert_eq!(msgs[1].content, "Qwen-specific.");
    }

    #[test]
    fn test_resolve_no_model_match() {
        let yaml = r#"
base:
  - role: developer
    content: "Base."
model_profiles:
  - pattern: "qwen*"
    messages:
      - role: developer
        content: "Qwen-specific."
"#;
        let profiles: InstructionProfiles = serde_yaml::from_str(yaml).unwrap();
        let msgs = profiles.resolve("gemma-3n", "main");
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].content, "Base.");
    }

    #[test]
    fn test_resolve_with_task_match() {
        let yaml = r#"
base: []
task_profiles:
  - kind: routing
    messages:
      - role: developer
        content: "Route only."
"#;
        let profiles: InstructionProfiles = serde_yaml::from_str(yaml).unwrap();
        let msgs = profiles.resolve("any-model", "routing");
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].content, "Route only.");
    }

    #[test]
    fn test_resolve_task_no_match() {
        let yaml = r#"
base: []
task_profiles:
  - kind: routing
    messages:
      - role: developer
        content: "Route only."
"#;
        let profiles: InstructionProfiles = serde_yaml::from_str(yaml).unwrap();
        let msgs = profiles.resolve("any-model", "main");
        assert!(msgs.is_empty());
    }

    #[test]
    fn test_resolve_combined() {
        let yaml = r#"
base:
  - role: developer
    content: "Base."
model_profiles:
  - pattern: "qwen*"
    messages:
      - role: developer
        content: "Model."
task_profiles:
  - kind: specialist
    messages:
      - role: developer
        content: "Task."
"#;
        let profiles: InstructionProfiles = serde_yaml::from_str(yaml).unwrap();
        let msgs = profiles.resolve("qwen-2.5", "specialist");
        assert_eq!(msgs.len(), 3); // base + model + task
        assert_eq!(msgs[0].content, "Base.");
        assert_eq!(msgs[1].content, "Model.");
        assert_eq!(msgs[2].content, "Task.");
    }

    #[test]
    fn test_resolve_multiple_model_profiles_match() {
        // Both `qwen*` and `*coder*` should match `qwen-2.5-coder`.
        let yaml = r#"
base: []
model_profiles:
  - pattern: "qwen*"
    messages:
      - role: developer
        content: "Qwen."
  - pattern: "*coder*"
    messages:
      - role: developer
        content: "Coder."
"#;
        let profiles: InstructionProfiles = serde_yaml::from_str(yaml).unwrap();
        let msgs = profiles.resolve("qwen-2.5-coder", "main");
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].content, "Qwen.");
        assert_eq!(msgs[1].content, "Coder.");
    }

    #[test]
    fn test_default_empty() {
        let profiles = InstructionProfiles::default();
        let msgs = profiles.resolve("any", "any");
        assert!(msgs.is_empty());
    }

    #[test]
    fn test_resolve_order_base_model_task() {
        // Verify the documented resolution order: base → model → task.
        let yaml = r#"
base:
  - role: developer
    content: "first"
model_profiles:
  - pattern: "*"
    messages:
      - role: developer
        content: "second"
task_profiles:
  - kind: main
    messages:
      - role: developer
        content: "third"
"#;
        let profiles: InstructionProfiles = serde_yaml::from_str(yaml).unwrap();
        let msgs = profiles.resolve("any-model", "main");
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[0].content, "first");
        assert_eq!(msgs[1].content, "second");
        assert_eq!(msgs[2].content, "third");
    }

    // ----- load from file -----

    #[test]
    fn test_load_from_nonexistent_file_returns_error() {
        let result = InstructionProfiles::load(std::path::Path::new("/nonexistent/path.yaml"));
        assert!(result.is_err());
    }

    #[test]
    fn test_load_from_temp_file() {
        use std::io::Write;
        let mut file = tempfile::NamedTempFile::new().unwrap();
        let yaml = r#"
base:
  - role: developer
    content: "From file."
"#;
        file.write_all(yaml.as_bytes()).unwrap();
        let profiles = InstructionProfiles::load(file.path()).unwrap();
        assert_eq!(profiles.base.len(), 1);
        assert_eq!(profiles.base[0].content, "From file.");
    }
}
