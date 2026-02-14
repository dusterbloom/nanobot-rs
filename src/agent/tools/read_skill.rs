//! Read-skill tool: fetch a skill's full content on demand.
//!
//! In lazy/RLM mode, skills are listed as names+descriptions in the system
//! prompt. This tool lets the agent fetch the full SKILL.md content when it
//! decides a skill is relevant — context as variable, not input.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use async_trait::async_trait;
use serde_json::{json, Value};

use super::base::Tool;
use crate::agent::skills::SkillsLoader;

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
        "read_skill"
    }

    fn description(&self) -> &str {
        "Read a skill's full instructions by name. Use this when the skills \
         summary in the system prompt mentions a skill you want to use."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The skill name (as shown in the <name> tag of the skills list)"
                }
            },
            "required": ["name"]
        })
    }

    async fn execute(&self, params: HashMap<String, Value>) -> String {
        let name = match params.get("name").and_then(|v| v.as_str()) {
            Some(n) => n,
            None => return "Error: 'name' parameter is required".to_string(),
        };

        let loader = SkillsLoader::new(&self.workspace, None);

        match loader.load_skill(name) {
            Some(content) => content,
            None => {
                // List available skills to help the agent.
                let available: Vec<String> = loader
                    .list_skills(false)
                    .into_iter()
                    .map(|s| s.name)
                    .collect();
                if available.is_empty() {
                    format!("Error: Skill '{}' not found. No skills are installed.", name)
                } else {
                    format!(
                        "Error: Skill '{}' not found. Available skills: {}",
                        name,
                        available.join(", ")
                    )
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
        let result = tool.execute(params).await;
        assert!(result.contains("Coding Skill"));
        assert!(result.contains("Write good code."));
    }

    #[tokio::test]
    async fn test_read_nonexistent_skill() {
        let (_tmp, tool) = make_workspace_with_skill("coding", "body");
        let mut params = HashMap::new();
        params.insert("name".to_string(), json!("nonexistent"));
        let result = tool.execute(params).await;
        assert!(result.starts_with("Error:"));
        assert!(result.contains("coding")); // lists available skills
    }

    #[tokio::test]
    async fn test_missing_name_param() {
        let (_tmp, tool) = make_workspace_with_skill("test", "body");
        let params = HashMap::new();
        let result = tool.execute(params).await;
        assert!(result.contains("Error:"));
        assert!(result.contains("'name' parameter"));
    }

    #[test]
    fn test_tool_schema() {
        let tmp = TempDir::new().unwrap();
        let tool = ReadSkillTool::new(tmp.path());
        assert_eq!(tool.name(), "read_skill");
        let schema = tool.to_schema();
        assert_eq!(schema["function"]["name"], "read_skill");
        assert!(schema["function"]["parameters"]["properties"]["name"].is_object());
    }
}
