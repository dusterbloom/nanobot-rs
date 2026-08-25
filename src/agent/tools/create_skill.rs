//! Create-skill tool (v0.5 E2): the agent authors its own skills.
//!
//! Writes `workspace/skills/{name}/SKILL.md` with validated frontmatter.
//! Availability is immediate: `SkillsLoader` re-scans disk on every
//! system-prompt build, so the new skill appears next turn with no reload
//! machinery. Idle turns may only write here anyway (E1 allowlist); on
//! normal turns this tool is how the user asks the agent to codify a
//! procedure interactively.

use std::collections::HashMap;
use std::path::PathBuf;

use async_trait::async_trait;
use serde_json::{json, Value};

use super::base::{Tool, ToolConcurrency, ToolContext};
use crate::agent::tools::base::ToolResult;

const MAX_NAME_CHARS: usize = 48;
const MAX_DESCRIPTION_CHARS: usize = 200;
/// Matches the loader's skill lint threshold (skills.rs 200-line warning).
const MAX_BODY_LINES: usize = 200;

pub struct CreateSkillTool {
    workspace: PathBuf,
}

impl CreateSkillTool {
    pub fn new(workspace: &std::path::Path) -> Self {
        Self {
            workspace: workspace.to_path_buf(),
        }
    }
}

fn is_kebab_case(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= MAX_NAME_CHARS
        && name
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
        && !name.starts_with('-')
        && !name.ends_with('-')
        && !name.contains("--")
}

/// Render the SKILL.md the loader contract expects: YAML frontmatter with a
/// `description`, then the body (skills.rs `load_skill` / `validate_skill`).
fn render_skill_md(description: &str, body: &str) -> String {
    format!("---\ndescription: {description}\n---\n\n{body}\n")
}

#[async_trait]
impl Tool for CreateSkillTool {
    fn name(&self) -> &str {
        "create_skill"
    }

    fn description(&self) -> &str {
        "Create a new skill (reusable procedure) as workspace/skills/{name}/SKILL.md. \
         The skill becomes available on the next turn. Use when a procedure is worth \
         repeating: name it in kebab-case, describe when to use it, write the body as \
         step-by-step instructions."
    }

    fn concurrency(&self) -> ToolConcurrency {
        // Sequential: one physical write target per name.
        ToolConcurrency::Sequential
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Kebab-case skill name, e.g. release-checklist"
                },
                "description": {
                    "type": "string",
                    "description": "One sentence: when should this skill be used?"
                },
                "body": {
                    "type": "string",
                    "description": "The skill instructions (markdown, <=200 lines)"
                }
            },
            "required": ["name", "description", "body"]
        })
    }

    async fn execute(&self, params: HashMap<String, Value>) -> String {
        let name = params
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim();
        let description = params
            .get("description")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim();
        let body = params
            .get("body")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim();

        if name.is_empty() {
            return "Error: 'name' parameter is required (kebab-case, e.g. release-checklist)".to_string();
        }
        if !is_kebab_case(name) {
            return format!(
                "Error: skill name must be kebab-case (lowercase letters, digits, single hyphens; <= {MAX_NAME_CHARS} chars), got '{name}'."
            );
        }
        if description.is_empty() || description.chars().count() > MAX_DESCRIPTION_CHARS {
            return format!(
                "Error: description must be 1..={MAX_DESCRIPTION_CHARS} characters."
            );
        }
        let body_lines = body.lines().count();
        if body.is_empty() || body_lines > MAX_BODY_LINES {
            return format!("Error: body must be 1..={MAX_BODY_LINES} lines, got {body_lines}.");
        }

        let skill_dir = self.workspace.join("skills").join(name);
        let skill_path = skill_dir.join("SKILL.md");
        if skill_path.exists() {
            let loader = crate::agent::skills::SkillsLoader::new(&self.workspace, None);
            let available: Vec<String> = loader
                .list_skills(false)
                .into_iter()
                .map(|s| s.name)
                .collect();
            return format!(
                "Error: skill '{name}' already exists. Existing skills: {}",
                available.join(", ")
            );
        }

        if let Err(e) = tokio::fs::create_dir_all(&skill_dir).await {
            return format!("Error creating skill directory: {e}");
        }
        // Atomic publish: stage then rename, mirroring memory.rs writes.
        let tmp = skill_path.with_extension("md.tmp");
        if let Err(e) = tokio::fs::write(&tmp, render_skill_md(description, body)).await {
            return format!("Error writing skill: {e}");
        }
        if let Err(e) = tokio::fs::rename(&tmp, &skill_path).await {
            return format!("Error publishing skill: {e}");
        }

        format!(
            "Skill '{name}' created at {} and available on the next turn.",
            skill_path.display()
        )
    }

    /// Typed funnel (remember.rs pattern): validation failures classify as
    /// model-fixable via `from_legacy`; success passes the created-path text.
    async fn execute_typed(&self, params: HashMap<String, Value>, ctx: &ToolContext) -> ToolResult {
        let out = self.execute_with_context(params, ctx).await;
        crate::agent::tools::base::funnel_legacy(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(name: &str, description: &str, body: &str) -> HashMap<String, Value> {
        let mut m = HashMap::new();
        m.insert("name".to_string(), json!(name));
        m.insert("description".to_string(), json!(description));
        m.insert("body".to_string(), json!(body));
        m
    }

    #[tokio::test]
    async fn creates_valid_skill_visible_to_loader() {
        let tmp = tempfile::tempdir().unwrap();
        let tool = CreateSkillTool::new(tmp.path());
        let out = tool
            .execute(params("brew-coffee", "How to brew coffee", "1. Grind beans\n2. Brew"))
            .await;
        assert!(out.contains("brew-coffee"), "created: {out}");

        // Hot-reload is free: a fresh loader scan sees it.
        let loader = crate::agent::skills::SkillsLoader::new(tmp.path(), None);
        let content = loader.load_skill("brew-coffee").expect("loadable");
        assert!(content.contains("Grind beans"));
    }

    #[tokio::test]
    async fn rejects_duplicate_bad_name_and_oversized() {
        let tmp = tempfile::tempdir().unwrap();
        let tool = CreateSkillTool::new(tmp.path());

        let first = tool.execute(params("dup", "d", "b")).await;
        assert!(first.contains("created"));
        let dup = tool.execute(params("dup", "d", "b")).await;
        assert!(dup.starts_with("Error:"), "duplicate rejected: {dup}");

        let bad = tool.execute(params("Bad_Name", "d", "b")).await;
        assert!(bad.starts_with("Error:"), "non-kebab rejected");

        let long_body = "line\n".repeat(201);
        let oversized = tool.execute(params("big", "d", &long_body)).await;
        assert!(oversized.starts_with("Error:"), ">200 lines rejected");
    }

    #[tokio::test]
    async fn typed_funnel_classifies_validation_failures() {
        let tmp = tempfile::tempdir().unwrap();
        let tool = CreateSkillTool::new(tmp.path());
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let token = tokio_util::sync::CancellationToken::new();
        let ctx = ToolContext::new(None, tx, token, "test-call");
        let res = tool
            .execute_typed(params("BAD!", "d", "b"), &ctx)
            .await;
        assert!(res.is_err(), "funnel turns Error: string into ToolError");
    }
}
