//! Skills loader for agent capabilities.
//!
//! Skills are markdown files (`SKILL.md`) that teach the agent how to use
//! specific tools or perform certain tasks.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use regex::Regex;
use tracing::debug;

/// Information about a discovered skill.
#[derive(Debug, Clone)]
pub struct SkillInfo {
    pub name: String,
    pub path: String,
    pub source: String,
}

/// Loads and manages agent skills from workspace and built-in directories.
pub struct SkillsLoader {
    workspace: PathBuf,
    workspace_skills: PathBuf,
    builtin_skills: PathBuf,
}

impl SkillsLoader {
    /// Create a new `SkillsLoader`.
    ///
    /// * `workspace`         - the agent workspace root.
    /// * `builtin_skills_dir` - optional override for the built-in skills directory.
    pub fn new(workspace: &Path, builtin_skills_dir: Option<&Path>) -> Self {
        let builtin = match builtin_skills_dir {
            Some(p) => p.to_path_buf(),
            None => workspace.join("builtin_skills"),
        };
        Self {
            workspace: workspace.to_path_buf(),
            workspace_skills: workspace.join("skills"),
            builtin_skills: builtin,
        }
    }

    /// List all available skills.
    ///
    /// When `filter_unavailable` is `true`, skills with unmet requirements are
    /// excluded from the result.
    pub fn list_skills(&self, filter_unavailable: bool) -> Vec<SkillInfo> {
        let mut skills: Vec<SkillInfo> = Vec::new();
        let mut seen_names: Vec<String> = Vec::new();

        // Workspace skills (highest priority).
        if self.workspace_skills.exists() {
            if let Ok(entries) = fs::read_dir(&self.workspace_skills) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_dir() {
                        let skill_file = path.join("SKILL.md");
                        if skill_file.exists() {
                            let name = path
                                .file_name()
                                .unwrap_or_default()
                                .to_string_lossy()
                                .to_string();
                            seen_names.push(name.clone());
                            skills.push(SkillInfo {
                                name,
                                path: skill_file.to_string_lossy().to_string(),
                                source: "workspace".to_string(),
                            });
                        }
                    }
                }
            }
        }

        // Built-in skills.
        if self.builtin_skills.exists() {
            if let Ok(entries) = fs::read_dir(&self.builtin_skills) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_dir() {
                        let skill_file = path.join("SKILL.md");
                        if skill_file.exists() {
                            let name = path
                                .file_name()
                                .unwrap_or_default()
                                .to_string_lossy()
                                .to_string();
                            if !seen_names.contains(&name) {
                                skills.push(SkillInfo {
                                    name,
                                    path: skill_file.to_string_lossy().to_string(),
                                    source: "builtin".to_string(),
                                });
                            }
                        }
                    }
                }
            }
        }

        if filter_unavailable {
            skills
                .into_iter()
                .filter(|s| {
                    let meta = self._get_skill_meta(&s.name);
                    _check_requirements(&meta)
                })
                .collect()
        } else {
            skills
        }
    }

    /// Load a skill's content by name.
    pub fn load_skill(&self, name: &str) -> Option<String> {
        // Workspace first.
        let workspace_skill = self.workspace_skills.join(name).join("SKILL.md");
        if workspace_skill.exists() {
            return fs::read_to_string(&workspace_skill).ok();
        }

        // Built-in.
        let builtin_skill = self.builtin_skills.join(name).join("SKILL.md");
        if builtin_skill.exists() {
            return fs::read_to_string(&builtin_skill).ok();
        }

        None
    }

    /// Load specific skills for inclusion in agent context.
    pub fn load_skills_for_context(&self, skill_names: &[String]) -> String {
        let mut parts: Vec<String> = Vec::new();

        for name in skill_names {
            if let Some(content) = self.load_skill(name) {
                let stripped = _strip_frontmatter(&content);
                parts.push(format!("### Skill: {}\n\n{}", name, stripped));
            }
        }

        if parts.is_empty() {
            String::new()
        } else {
            parts.join("\n\n---\n\n")
        }
    }

    /// Build an XML-formatted summary of all skills.
    pub fn build_skills_summary(&self) -> String {
        let all_skills = self.list_skills(false);
        if all_skills.is_empty() {
            return String::new();
        }

        let mut lines: Vec<String> = vec!["<skills>".to_string()];

        for s in &all_skills {
            let name = _escape_xml(&s.name);
            let path = &s.path;
            let desc = _escape_xml(&self._get_skill_description(&s.name));
            let skill_meta = self._get_skill_meta(&s.name);
            let available = _check_requirements(&skill_meta);

            lines.push(format!(
                "  <skill available=\"{}\">",
                if available { "true" } else { "false" }
            ));
            lines.push(format!("    <name>{}</name>", name));
            lines.push(format!("    <description>{}</description>", desc));
            lines.push(format!("    <location>{}</location>", path));

            if !available {
                let missing = _get_missing_requirements(&skill_meta);
                if !missing.is_empty() {
                    lines.push(format!(
                        "    <requires>{}</requires>",
                        _escape_xml(&missing)
                    ));
                }
            }

            lines.push("  </skill>".to_string());
        }

        lines.push("</skills>".to_string());
        lines.join("\n")
    }

    /// Get skills marked as `always=true` that also meet requirements.
    pub fn get_always_skills(&self) -> Vec<String> {
        let mut result: Vec<String> = Vec::new();

        for s in self.list_skills(true) {
            if let Some(meta) = self.get_skill_metadata(&s.name) {
                let skill_meta =
                    _parse_skill_metadata(meta.get("metadata").map(|s| s.as_str()).unwrap_or(""));
                if skill_meta
                    .get("always")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false)
                {
                    result.push(s.name.clone());
                    continue;
                }
                // Also check top-level "always" in frontmatter.
                if meta.get("always").map(|v| v == "true").unwrap_or(false) {
                    result.push(s.name.clone());
                }
            }
        }

        result
    }

    /// Parse YAML-like frontmatter metadata from a skill file.
    pub fn get_skill_metadata(&self, name: &str) -> Option<HashMap<String, String>> {
        let content = self.load_skill(name)?;

        if !content.starts_with("---") {
            return None;
        }

        let re = Regex::new(r"(?s)^---\n(.*?)\n---").ok()?;
        let caps = re.captures(&content)?;
        let frontmatter = caps.get(1)?.as_str();

        let mut metadata = HashMap::new();
        for line in frontmatter.lines() {
            if let Some((key, value)) = line.split_once(':') {
                let k = key.trim().to_string();
                let v = value
                    .trim()
                    .trim_matches(|c| c == '"' || c == '\'')
                    .to_string();
                metadata.insert(k, v);
            }
        }

        Some(metadata)
    }

    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    fn _get_skill_description(&self, name: &str) -> String {
        if let Some(meta) = self.get_skill_metadata(name) {
            if let Some(desc) = meta.get("description") {
                if !desc.is_empty() {
                    return desc.clone();
                }
            }
        }
        name.to_string()
    }

    fn _get_skill_meta(&self, name: &str) -> HashMap<String, serde_json::Value> {
        let meta = match self.get_skill_metadata(name) {
            Some(m) => m,
            None => return HashMap::new(),
        };
        let raw = meta.get("metadata").map(|s| s.as_str()).unwrap_or("");
        _parse_skill_metadata(raw)
    }
}

// ---------------------------------------------------------------------------
// Module-level helpers
// ---------------------------------------------------------------------------

/// Strip YAML frontmatter from markdown content.
fn _strip_frontmatter(content: &str) -> String {
    if content.starts_with("---") {
        if let Some(re) = Regex::new(r"(?s)^---\n.*?\n---\n").ok() {
            if let Some(m) = re.find(content) {
                return content[m.end()..].trim().to_string();
            }
        }
    }
    content.to_string()
}

/// Parse skill metadata JSON from frontmatter value.
/// Looks for a "nanobot" key for backward compatibility with upstream skill files.
fn _parse_skill_metadata(raw: &str) -> HashMap<String, serde_json::Value> {
    if raw.is_empty() {
        return HashMap::new();
    }
    match serde_json::from_str::<serde_json::Value>(raw) {
        Ok(serde_json::Value::Object(map)) => {
            if let Some(serde_json::Value::Object(nanobot)) = map.get("nanobot") {
                nanobot
                    .iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect()
            } else {
                HashMap::new()
            }
        }
        _ => HashMap::new(),
    }
}

/// Check if skill requirements are met (bins in PATH, env vars set).
fn _check_requirements(skill_meta: &HashMap<String, serde_json::Value>) -> bool {
    if let Some(requires) = skill_meta.get("requires") {
        // Check required binaries.
        if let Some(bins) = requires.get("bins").and_then(|v| v.as_array()) {
            for bin_val in bins {
                if let Some(bin_name) = bin_val.as_str() {
                    if !_command_exists(bin_name) {
                        debug!("Skill requirement not met: binary '{}' not found", bin_name);
                        return false;
                    }
                }
            }
        }
        // Check required environment variables.
        if let Some(env_vars) = requires.get("env").and_then(|v| v.as_array()) {
            for env_val in env_vars {
                if let Some(env_name) = env_val.as_str() {
                    if std::env::var(env_name).is_err() {
                        debug!("Skill requirement not met: env var '{}' not set", env_name);
                        return false;
                    }
                }
            }
        }
    }
    true
}

/// Get a description of what requirements are missing.
fn _get_missing_requirements(skill_meta: &HashMap<String, serde_json::Value>) -> String {
    let mut missing: Vec<String> = Vec::new();

    if let Some(requires) = skill_meta.get("requires") {
        if let Some(bins) = requires.get("bins").and_then(|v| v.as_array()) {
            for bin_val in bins {
                if let Some(bin_name) = bin_val.as_str() {
                    if !_command_exists(bin_name) {
                        missing.push(format!("CLI: {}", bin_name));
                    }
                }
            }
        }
        if let Some(env_vars) = requires.get("env").and_then(|v| v.as_array()) {
            for env_val in env_vars {
                if let Some(env_name) = env_val.as_str() {
                    if std::env::var(env_name).is_err() {
                        missing.push(format!("ENV: {}", env_name));
                    }
                }
            }
        }
    }

    missing.join(", ")
}

/// Check whether a binary exists on the PATH.
fn _command_exists(name: &str) -> bool {
    std::process::Command::new("which")
        .arg(name)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Escape XML special characters.
fn _escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    /// Helper: create a workspace temp dir with a skills/ subdirectory
    /// containing one skill named `test-skill` with a SKILL.md file.
    fn make_workspace_with_skill(frontmatter: Option<&str>, body: &str) -> (TempDir, SkillsLoader) {
        let tmp = TempDir::new().unwrap();
        let skill_dir = tmp.path().join("skills").join("test-skill");
        fs::create_dir_all(&skill_dir).unwrap();

        let content = match frontmatter {
            Some(fm) => format!("---\n{}\n---\n{}", fm, body),
            None => body.to_string(),
        };
        fs::write(skill_dir.join("SKILL.md"), &content).unwrap();

        // Point builtin_skills to a non-existent dir so it is ignored.
        let loader = SkillsLoader::new(tmp.path(), Some(&tmp.path().join("no_builtin")));
        (tmp, loader)
    }

    // ----- _strip_frontmatter -----

    #[test]
    fn test_strip_frontmatter_with_frontmatter() {
        let content = "---\ntitle: Test\n---\nBody content";
        let result = _strip_frontmatter(content);
        assert_eq!(result, "Body content");
    }

    #[test]
    fn test_strip_frontmatter_without_frontmatter() {
        let content = "Just plain markdown content";
        let result = _strip_frontmatter(content);
        assert_eq!(result, "Just plain markdown content");
    }

    #[test]
    fn test_strip_frontmatter_multiline_body() {
        let content = "---\nkey: val\n---\nLine 1\nLine 2\nLine 3";
        let result = _strip_frontmatter(content);
        assert_eq!(result, "Line 1\nLine 2\nLine 3");
    }

    // ----- _escape_xml -----

    #[test]
    fn test_escape_xml_ampersand() {
        assert_eq!(_escape_xml("a & b"), "a &amp; b");
    }

    #[test]
    fn test_escape_xml_angle_brackets() {
        assert_eq!(_escape_xml("<tag>"), "&lt;tag&gt;");
    }

    #[test]
    fn test_escape_xml_combined() {
        assert_eq!(_escape_xml("x < y & y > z"), "x &lt; y &amp; y &gt; z");
    }

    #[test]
    fn test_escape_xml_no_special_chars() {
        assert_eq!(_escape_xml("hello world"), "hello world");
    }

    // ----- _parse_skill_metadata -----

    #[test]
    fn test_parse_skill_metadata_empty_string() {
        let result = _parse_skill_metadata("");
        assert!(result.is_empty());
    }

    #[test]
    fn test_parse_skill_metadata_valid_json() {
        let raw = r#"{"nanobot": {"always": true, "priority": 1}}"#;
        let result = _parse_skill_metadata(raw);
        assert_eq!(result.get("always").and_then(|v| v.as_bool()), Some(true));
        assert_eq!(result.get("priority").and_then(|v| v.as_i64()), Some(1));
    }

    #[test]
    fn test_parse_skill_metadata_no_nanobot_key() {
        let raw = r#"{"other": {"key": "val"}}"#;
        let result = _parse_skill_metadata(raw);
        assert!(result.is_empty());
    }

    #[test]
    fn test_parse_skill_metadata_invalid_json() {
        let raw = "not json at all";
        let result = _parse_skill_metadata(raw);
        assert!(result.is_empty());
    }

    // ----- _check_requirements -----

    #[test]
    fn test_check_requirements_no_requires_key() {
        let meta: HashMap<String, serde_json::Value> = HashMap::new();
        assert!(_check_requirements(&meta));
    }

    #[test]
    fn test_check_requirements_with_existing_bin() {
        // "ls" should always exist on Linux/macOS.
        let mut meta: HashMap<String, serde_json::Value> = HashMap::new();
        meta.insert("requires".to_string(), serde_json::json!({"bins": ["ls"]}));
        assert!(_check_requirements(&meta));
    }

    #[test]
    fn test_check_requirements_with_missing_bin() {
        let mut meta: HashMap<String, serde_json::Value> = HashMap::new();
        meta.insert(
            "requires".to_string(),
            serde_json::json!({"bins": ["this_binary_does_not_exist_xyz_123"]}),
        );
        assert!(!_check_requirements(&meta));
    }

    #[test]
    fn test_check_requirements_with_missing_env() {
        let mut meta: HashMap<String, serde_json::Value> = HashMap::new();
        meta.insert(
            "requires".to_string(),
            serde_json::json!({"env": ["NANOBOT_TEST_NONEXISTENT_VAR_XYZ"]}),
        );
        assert!(!_check_requirements(&meta));
    }

    // ----- list_skills -----

    #[test]
    fn test_list_skills_finds_workspace_skill() {
        let (_tmp, loader) = make_workspace_with_skill(None, "# Test Skill\nDoes stuff.");
        let skills = loader.list_skills(false);
        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].name, "test-skill");
        assert_eq!(skills[0].source, "workspace");
    }

    #[test]
    fn test_list_skills_empty_workspace() {
        let tmp = TempDir::new().unwrap();
        let loader = SkillsLoader::new(tmp.path(), Some(&tmp.path().join("no_builtin")));
        let skills = loader.list_skills(false);
        assert!(skills.is_empty());
    }

    // ----- load_skill -----

    #[test]
    fn test_load_skill_returns_content() {
        let (_tmp, loader) = make_workspace_with_skill(None, "# My Skill\nInstructions here.");
        let content = loader.load_skill("test-skill");
        assert!(content.is_some());
        assert!(content.unwrap().contains("Instructions here."));
    }

    #[test]
    fn test_load_skill_nonexistent_returns_none() {
        let (_tmp, loader) = make_workspace_with_skill(None, "body");
        let content = loader.load_skill("no-such-skill");
        assert!(content.is_none());
    }

    // ----- get_skill_metadata -----

    #[test]
    fn test_get_skill_metadata_with_frontmatter() {
        let frontmatter = "description: A cool skill\nauthor: tester";
        let (_tmp, loader) = make_workspace_with_skill(Some(frontmatter), "body");
        let meta = loader.get_skill_metadata("test-skill");
        assert!(meta.is_some());
        let meta = meta.unwrap();
        assert_eq!(
            meta.get("description").map(|s| s.as_str()),
            Some("A cool skill")
        );
        assert_eq!(meta.get("author").map(|s| s.as_str()), Some("tester"));
    }

    #[test]
    fn test_get_skill_metadata_without_frontmatter() {
        let (_tmp, loader) = make_workspace_with_skill(None, "plain body");
        let meta = loader.get_skill_metadata("test-skill");
        assert!(meta.is_none());
    }

    #[test]
    fn test_get_skill_metadata_strips_quotes() {
        let frontmatter = "description: \"Quoted value\"";
        let (_tmp, loader) = make_workspace_with_skill(Some(frontmatter), "body");
        let meta = loader.get_skill_metadata("test-skill").unwrap();
        assert_eq!(
            meta.get("description").map(|s| s.as_str()),
            Some("Quoted value")
        );
    }

    // ----- build_skills_summary -----

    #[test]
    fn test_build_skills_summary_xml_format() {
        let frontmatter = "description: Test description";
        let (_tmp, loader) = make_workspace_with_skill(Some(frontmatter), "body");
        let summary = loader.build_skills_summary();
        assert!(summary.starts_with("<skills>"));
        assert!(summary.ends_with("</skills>"));
        assert!(summary.contains("<name>test-skill</name>"));
        assert!(summary.contains("<description>Test description</description>"));
    }

    #[test]
    fn test_build_skills_summary_empty() {
        let tmp = TempDir::new().unwrap();
        let loader = SkillsLoader::new(tmp.path(), Some(&tmp.path().join("no_builtin")));
        let summary = loader.build_skills_summary();
        assert_eq!(summary, "");
    }

    #[test]
    fn test_build_skills_summary_escapes_xml() {
        // Create a skill whose name contains XML-special characters.
        let tmp = TempDir::new().unwrap();
        let skill_dir = tmp.path().join("skills").join("a&b");
        fs::create_dir_all(&skill_dir).unwrap();
        fs::write(
            skill_dir.join("SKILL.md"),
            "---\ndescription: x < y\n---\nbody",
        )
        .unwrap();
        let loader = SkillsLoader::new(tmp.path(), Some(&tmp.path().join("no_builtin")));
        let summary = loader.build_skills_summary();
        assert!(summary.contains("a&amp;b"));
        assert!(summary.contains("x &lt; y"));
    }

    // ----- load_skills_for_context -----

    #[test]
    fn test_load_skills_for_context_strips_frontmatter() {
        let frontmatter = "description: Test";
        let (_tmp, loader) = make_workspace_with_skill(Some(frontmatter), "The body content.\n");
        let names = vec!["test-skill".to_string()];
        let result = loader.load_skills_for_context(&names);
        assert!(result.contains("### Skill: test-skill"));
        assert!(result.contains("The body content."));
        // Frontmatter should be stripped.
        assert!(!result.contains("description: Test"));
    }

    #[test]
    fn test_load_skills_for_context_nonexistent_skill() {
        let (_tmp, loader) = make_workspace_with_skill(None, "body");
        let names = vec!["no-such-skill".to_string()];
        let result = loader.load_skills_for_context(&names);
        assert_eq!(result, "");
    }

    // ----- get_always_skills -----

    #[test]
    fn test_get_always_skills_with_always_flag() {
        let frontmatter = "always: true\ndescription: Always on";
        let (_tmp, loader) = make_workspace_with_skill(Some(frontmatter), "body");
        let always = loader.get_always_skills();
        assert!(always.contains(&"test-skill".to_string()));
    }

    #[test]
    fn test_get_always_skills_without_flag() {
        let frontmatter = "description: Normal skill";
        let (_tmp, loader) = make_workspace_with_skill(Some(frontmatter), "body");
        let always = loader.get_always_skills();
        assert!(always.is_empty());
    }

    // ----- builtin skill priority -----

    #[test]
    fn test_workspace_skill_shadows_builtin() {
        let tmp = TempDir::new().unwrap();

        // Create workspace skill.
        let ws_skill = tmp.path().join("skills").join("overlap");
        fs::create_dir_all(&ws_skill).unwrap();
        fs::write(ws_skill.join("SKILL.md"), "workspace version").unwrap();

        // Create builtin skill with same name.
        let bi_dir = tmp.path().join("builtin");
        let bi_skill = bi_dir.join("overlap");
        fs::create_dir_all(&bi_skill).unwrap();
        fs::write(bi_skill.join("SKILL.md"), "builtin version").unwrap();

        let loader = SkillsLoader::new(tmp.path(), Some(&bi_dir));
        let skills = loader.list_skills(false);

        // Should only find one, from workspace.
        let overlap_skills: Vec<&SkillInfo> =
            skills.iter().filter(|s| s.name == "overlap").collect();
        assert_eq!(overlap_skills.len(), 1);
        assert_eq!(overlap_skills[0].source, "workspace");

        // load_skill should return workspace version.
        let content = loader.load_skill("overlap").unwrap();
        assert_eq!(content, "workspace version");
    }
}
