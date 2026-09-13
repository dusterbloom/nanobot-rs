//! Read-skill tool: fetch a skill's full content on demand.
//!
//! In lazy/RLM mode, skills are listed as names+descriptions in the system
//! prompt. This tool lets the agent fetch the full SKILL.md content when it
//! decides a skill is relevant — context as variable, not input.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use async_trait::async_trait;
use serde_json::{json, Value};

use super::base::{Tool, ToolConcurrency, ToolContext, ToolResult};
use crate::agent::skills::SkillsLoader;
use crate::errors::ToolError;

/// Tool that reads a skill's full content by name.
pub struct ReadSkillTool {
    workspace: PathBuf,
}

impl ReadSkillTool {
    pub fn new(workspace: &Path) -> Self {
        Self {
            workspace: workspace.to_path_buf(),
        }
    }
}

#[async_trait]
impl Tool for ReadSkillTool {
    fn name(&self) -> &str {
        "get_skills"
    }

    fn description(&self) -> &str {
        "List available skills when name is omitted, or load a known skill's full instructions by name. \
         Use the bounded listing for discovery; once the skill name is known, call get_skills with that name to load SKILL.md."
    }

    fn concurrency(&self) -> ToolConcurrency {
        ToolConcurrency::ParallelSafe
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Skill name to read (e.g. webradio, newsreader). Omit this field entirely to list all installed skills."
                }
            },
        })
    }

    async fn execute(&self, params: HashMap<String, Value>, _ctx: &ToolContext) -> ToolResult {
        // No name (or empty, or the legacy "__list__" sentinel) → list all
        // skills. Same discoverability contract as the `tool` proxy: call with
        // no args to discover, call with a name to load.
        let name = params
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim();
        let want_list = name.is_empty() || name == "__list__";

        if want_list {
            let summary = SkillsLoader::new(&self.workspace, None).build_compact_index();
            return Ok(if summary.is_empty() {
                "No skills are installed.".into()
            } else {
                summary.into()
            });
        }

        let loader = SkillsLoader::new(&self.workspace, None);
        match loader.load_skill(name) {
            Some(content) => Ok(content.into()),
            None => {
                // List available skills to help the agent.
                let available: Vec<String> = loader
                    .list_skills(false)
                    .into_iter()
                    .map(|s| s.name)
                    .collect();
                if available.is_empty() {
                    Err(ToolError::NotFound(format!(
                        "Skill '{}' not found. No skills are installed.",
                        name
                    )))
                } else {
                    Err(ToolError::NotFound(format!(
                        "Skill '{}' not found. Available skills: {}",
                        name,
                        available.join(", ")
                    )))
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn make_workspace_with_skill(name: &str, content: &str) -> (TempDir, ReadSkillTool) {
        let tmp = TempDir::new().unwrap();
        let skill_dir = tmp.path().join("skills").join(name);
        fs::create_dir_all(&skill_dir).unwrap();
        fs::write(skill_dir.join("SKILL.md"), content).unwrap();
        let tool = ReadSkillTool::new(tmp.path());
        (tmp, tool)
    }

    #[tokio::test]
    async fn test_read_existing_skill() {
        let (_tmp, tool) = make_workspace_with_skill("coding", "# Coding Skill\nWrite good code.");
        let mut params = HashMap::new();
        params.insert("name".to_string(), json!("coding"));
        let result = crate::agent::tools::base::render_result(
            tool.execute(params, &crate::agent::tools::base::ToolContext::sandbox())
                .await,
        );
        assert!(result.contains("Coding Skill"));
        assert!(result.contains("Write good code."));
    }

    #[tokio::test]
    async fn test_read_nonexistent_skill() {
        let (_tmp, tool) = make_workspace_with_skill("coding", "body");
        let mut params = HashMap::new();
        params.insert("name".to_string(), json!("nonexistent"));
        let result = crate::agent::tools::base::render_result(
            tool.execute(params, &crate::agent::tools::base::ToolContext::sandbox())
                .await,
        );
        assert!(result.starts_with("Error:"));
        assert!(result.contains("coding")); // lists available skills
    }

    #[tokio::test]
    async fn test_missing_name_param_lists_skills() {
        // No name → list all skills (same discoverability contract as the
        // `tool` proxy). Replaces the old error-on-missing-name behavior that
        // left the model unable to discover skills without the obscure
        // "__list__" sentinel.
        let (_tmp, tool) = make_workspace_with_skill("test", "body");
        let params = HashMap::new();
        let result = crate::agent::tools::base::render_result(
            tool.execute(params, &crate::agent::tools::base::ToolContext::sandbox())
                .await,
        );
        assert!(
            result.starts_with("Installed skills preview:"),
            "got: {result}"
        );
        assert!(result.contains("test"));
    }

    #[tokio::test]
    async fn test_skill_listing_is_useful_without_xml_opener() {
        let (_tmp, tool) = make_workspace_with_skill(
            "coding",
            "---\ndescription: Write and review code\n---\nbody",
        );
        let result = crate::agent::tools::base::render_result(
            tool.execute(
                HashMap::new(),
                &crate::agent::tools::base::ToolContext::sandbox(),
            )
            .await,
        );
        assert!(
            !result.starts_with("<skills>"),
            "listing should expose entries directly, got: {result}"
        );
        assert!(
            result.contains("coding"),
            "listing omitted skill name: {result}"
        );
        assert!(
            result.contains("Write and review code"),
            "listing omitted skill description: {result}"
        );
        assert!(
            result.contains("get_skills") && result.contains("coding"),
            "listing must explain how to load a known skill: {result}"
        );
    }

    #[tokio::test]
    async fn test_large_skill_listing_keeps_bounded_entries_visible() {
        let tmp = TempDir::new().unwrap();
        for index in 0..80 {
            let name = format!("skill-{index:02}");
            let skill_dir = tmp.path().join("skills").join(&name);
            fs::create_dir_all(&skill_dir).unwrap();
            fs::write(
                skill_dir.join("SKILL.md"),
                format!(
                    "---\ndescription: Skill {index} explains a bounded operational workflow for reliable discovery\n---\nbody"
                ),
            )
            .unwrap();
        }
        let tool = ReadSkillTool::new(tmp.path());
        let result = crate::agent::tools::base::render_result(
            tool.execute(
                HashMap::new(),
                &crate::agent::tools::base::ToolContext::sandbox(),
            )
            .await,
        );
        assert!(
            result.chars().count() > 4096,
            "fixture must exercise a large listing"
        );
        assert!(result.starts_with("Installed skills preview:"));
        assert!(
            result.contains("- skill-00:"),
            "first entry was lost: {result}"
        );
        assert!(
            result.contains("- skill-79:"),
            "last entry was lost: {result}"
        );
        assert!(
            result.contains("get_skills"),
            "load guidance was lost: {result}"
        );
        assert!(!result.starts_with("<skills>"));
    }

    #[test]
    fn test_tool_schema() {
        let tmp = TempDir::new().unwrap();
        let tool = ReadSkillTool::new(tmp.path());
        assert_eq!(tool.name(), "get_skills");
        let schema = tool.to_schema();
        assert_eq!(schema["function"]["name"], "get_skills");
        assert!(schema["function"]["parameters"]["properties"]["name"].is_object());
    }

    #[tokio::test]
    async fn test_read_skill_list_returns_bounded_index() {
        // Set up a workspace with a skill that has a description.
        let (_tmp, tool) = make_workspace_with_skill(
            "my-skill",
            "---\ndescription: My skill description\n---\n# My Skill\nDo things.",
        );
        let mut params = HashMap::new();
        params.insert("name".to_string(), json!("__list__"));
        let result = crate::agent::tools::base::render_result(
            tool.execute(params, &crate::agent::tools::base::ToolContext::sandbox())
                .await,
        );
        assert!(
            result.starts_with("Installed skills preview:"),
            "got: {result}"
        );
        assert!(result.contains("- my-skill: My skill description"));
        assert!(result.contains("get_skills"));
    }

    #[tokio::test]
    async fn test_read_skill_list_empty_workspace() {
        let tmp = TempDir::new().unwrap();
        let tool = ReadSkillTool::new(tmp.path());
        let mut params = HashMap::new();
        params.insert("name".to_string(), json!("__list__"));
        let result = crate::agent::tools::base::render_result(
            tool.execute(params, &crate::agent::tools::base::ToolContext::sandbox())
                .await,
        );
        assert_eq!(result, "No skills are installed.");
    }
}
