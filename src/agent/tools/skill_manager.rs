//! Skill manager tool for dynamic skill creation and management.
//!
//! Allows the agent to create, update, list, read, and delete skills
//! at runtime. Skills are stored as `{workspace}/skills/{name}/SKILL.md`
//! files with optional YAML frontmatter.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use async_trait::async_trait;

use super::base::Tool;
use crate::utils::helpers::ensure_dir;

/// Tool for managing skills at runtime.
pub struct SkillManagerTool {
    workspace: PathBuf,
}

impl SkillManagerTool {
    pub fn new(workspace: PathBuf) -> Self {
        Self { workspace }
    }

    fn skills_dir(&self) -> PathBuf {
        self.workspace.join("skills")
    }

    fn skill_path(&self, name: &str) -> PathBuf {
        self.skills_dir().join(name).join("SKILL.md")
    }

    fn create_or_update(&self, name: &str, content: &str, description: Option<&str>, always: bool) -> String {
        let skill_dir = self.skills_dir().join(name);
        ensure_dir(&skill_dir);

        // Build frontmatter if metadata provided.
        let mut full_content = String::new();
        if description.is_some() || always {
            full_content.push_str("---\n");
            if let Some(desc) = description {
                full_content.push_str(&format!("description: {}\n", desc));
            }
            if always {
                full_content.push_str("always: true\n");
            }
            full_content.push_str("---\n\n");
        }
        full_content.push_str(content);

        let path = self.skill_path(name);
        match fs::write(&path, &full_content) {
            Ok(_) => format!("Skill '{}' saved at {}", name, path.display()),
            Err(e) => format!("Error: Failed to write skill '{}': {}", name, e),
        }
    }

    fn list_skills(&self) -> String {
        let dir = self.skills_dir();
        if !dir.is_dir() {
            return "No skills directory found.".to_string();
        }

        let mut skills = Vec::new();
        if let Ok(entries) = fs::read_dir(&dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    let skill_file = path.join("SKILL.md");
                    let name = path
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("unknown");
                    if skill_file.is_file() {
                        // Try to read description from frontmatter.
                        let desc = fs::read_to_string(&skill_file)
                            .ok()
                            .and_then(|content| extract_frontmatter_field(&content, "description"))
                            .unwrap_or_default();
                        skills.push(format!("- **{}**: {}", name, desc));
                    } else {
                        skills.push(format!("- **{}** (no SKILL.md)", name));
                    }
                }
            }
        }

        if skills.is_empty() {
            "No skills found.".to_string()
        } else {
            format!("Skills:\n{}", skills.join("\n"))
        }
    }

    fn read_skill(&self, name: &str) -> String {
        let path = self.skill_path(name);
        match fs::read_to_string(&path) {
            Ok(content) => content,
            Err(_) => format!("Error: Skill '{}' not found at {}", name, path.display()),
        }
    }

    fn delete_skill(&self, name: &str) -> String {
        let dir = self.skills_dir().join(name);
        if !dir.is_dir() {
            return format!("Error: Skill '{}' not found", name);
        }
        match fs::remove_dir_all(&dir) {
            Ok(_) => format!("Skill '{}' deleted", name),
            Err(e) => format!("Error: Failed to delete skill '{}': {}", name, e),
        }
    }
}

#[async_trait]
impl Tool for SkillManagerTool {
    fn name(&self) -> &str {
        "skill_manager"
    }

    fn description(&self) -> &str {
        "Create, update, list, read, or delete skills. Skills are reusable knowledge \
         and instructions that extend your capabilities. Use this to teach yourself \
         new capabilities that persist across conversations."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create", "update", "list", "read", "delete"],
                    "description": "Action to perform"
                },
                "name": {
                    "type": "string",
                    "description": "Skill name (required for create/update/read/delete)"
                },
                "content": {
                    "type": "string",
                    "description": "Skill content in markdown (required for create/update)"
                },
                "description": {
                    "type": "string",
                    "description": "Short description of the skill (optional, for create/update)"
                },
                "always": {
                    "type": "boolean",
                    "description": "If true, skill is always loaded into context (default: false)"
                }
            },
            "required": ["action"]
        })
    }

    async fn execute(&self, params: HashMap<String, serde_json::Value>) -> String {
        let action = match params.get("action").and_then(|v| v.as_str()) {
            Some(a) => a,
            None => return "Error: 'action' parameter is required".to_string(),
        };

        match action {
            "create" | "update" => {
                let name = match params.get("name").and_then(|v| v.as_str()) {
                    Some(n) => n,
                    None => return "Error: 'name' parameter is required for create/update".to_string(),
                };
                let content = match params.get("content").and_then(|v| v.as_str()) {
                    Some(c) => c,
                    None => return "Error: 'content' parameter is required for create/update".to_string(),
                };
                let description = params.get("description").and_then(|v| v.as_str());
                let always = params.get("always").and_then(|v| v.as_bool()).unwrap_or(false);
                self.create_or_update(name, content, description, always)
            }
            "list" => self.list_skills(),
            "read" => {
                let name = match params.get("name").and_then(|v| v.as_str()) {
                    Some(n) => n,
                    None => return "Error: 'name' parameter is required for read".to_string(),
                };
                self.read_skill(name)
            }
            "delete" => {
                let name = match params.get("name").and_then(|v| v.as_str()) {
                    Some(n) => n,
                    None => return "Error: 'name' parameter is required for delete".to_string(),
                };
                self.delete_skill(name)
            }
            _ => format!("Error: Unknown action '{}'. Use create, update, list, read, or delete.", action),
        }
    }
}

/// Extract a field value from YAML frontmatter.
fn extract_frontmatter_field(content: &str, field: &str) -> Option<String> {
    if !content.starts_with("---") {
        return None;
    }
    let rest = &content[3..];
    let end = rest.find("---")?;
    let frontmatter = &rest[..end];
    for line in frontmatter.lines() {
        if let Some(value) = line.strip_prefix(&format!("{}: ", field)) {
            return Some(value.trim().to_string());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_frontmatter_field() {
        let content = "---\ndescription: A test skill\nalways: true\n---\n\n# Content";
        assert_eq!(
            extract_frontmatter_field(content, "description"),
            Some("A test skill".to_string())
        );
        assert_eq!(
            extract_frontmatter_field(content, "always"),
            Some("true".to_string())
        );
        assert_eq!(extract_frontmatter_field(content, "missing"), None);
    }

    #[test]
    fn test_no_frontmatter() {
        let content = "# Just content\nNo frontmatter here.";
        assert_eq!(extract_frontmatter_field(content, "description"), None);
    }

    #[tokio::test]
    async fn test_skill_manager_crud() {
        let dir = tempfile::tempdir().unwrap();
        let tool = SkillManagerTool::new(dir.path().to_path_buf());

        // List (empty).
        let mut params = HashMap::new();
        params.insert("action".to_string(), serde_json::json!("list"));
        let result = tool.execute(params).await;
        assert!(result.contains("No skills"));

        // Create.
        let mut params = HashMap::new();
        params.insert("action".to_string(), serde_json::json!("create"));
        params.insert("name".to_string(), serde_json::json!("docker-deploy"));
        params.insert("content".to_string(), serde_json::json!("# Docker Deploy\nUse `docker compose up` to deploy."));
        params.insert("description".to_string(), serde_json::json!("Deploy with Docker"));
        let result = tool.execute(params).await;
        assert!(result.contains("saved"));

        // Read.
        let mut params = HashMap::new();
        params.insert("action".to_string(), serde_json::json!("read"));
        params.insert("name".to_string(), serde_json::json!("docker-deploy"));
        let result = tool.execute(params).await;
        assert!(result.contains("Docker Deploy"));
        assert!(result.contains("description: Deploy with Docker"));

        // List (one skill).
        let mut params = HashMap::new();
        params.insert("action".to_string(), serde_json::json!("list"));
        let result = tool.execute(params).await;
        assert!(result.contains("docker-deploy"));

        // Delete.
        let mut params = HashMap::new();
        params.insert("action".to_string(), serde_json::json!("delete"));
        params.insert("name".to_string(), serde_json::json!("docker-deploy"));
        let result = tool.execute(params).await;
        assert!(result.contains("deleted"));

        // List (empty again).
        let mut params = HashMap::new();
        params.insert("action".to_string(), serde_json::json!("list"));
        let result = tool.execute(params).await;
        assert!(result.contains("No skills"));
    }

    #[tokio::test]
    async fn test_skill_manager_missing_params() {
        let dir = tempfile::tempdir().unwrap();
        let tool = SkillManagerTool::new(dir.path().to_path_buf());

        let mut params = HashMap::new();
        params.insert("action".to_string(), serde_json::json!("create"));
        let result = tool.execute(params).await;
        assert!(result.contains("Error"));
    }

    #[test]
    fn test_tool_name() {
        let dir = tempfile::tempdir().unwrap();
        let tool = SkillManagerTool::new(dir.path().to_path_buf());
        assert_eq!(tool.name(), "skill_manager");
    }
}
