//! Todo tool: session-scoped working-memory scratchpad.
//!
//! Small local models juggling multiple concurrent tasks routinely conflate
//! them ("which webradio script was I editing again?"). This tool gives the
//! agent a tiny add/list/complete scratchpad backed by `{workspace}/TODO.json`
//! so it can pin whatever it's doing and not lose the thread when tool
//! results swamp its context window.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::fs;

use super::base::{PermissionLevel, Tool};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Todo {
    id: u32,
    text: String,
    done: bool,
}

pub struct TodoTool {
    path: PathBuf,
}

impl TodoTool {
    pub fn new(workspace: &Path) -> Self {
        Self {
            path: workspace.join("TODO.json"),
        }
    }

    async fn load(&self) -> Vec<Todo> {
        let Ok(raw) = fs::read_to_string(&self.path).await else {
            return Vec::new();
        };
        serde_json::from_str(&raw).unwrap_or_default()
    }

    async fn save(&self, items: &[Todo]) -> Result<(), String> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)
                .await
                .map_err(|e| format!("mkdir: {e}"))?;
        }
        let raw = serde_json::to_string_pretty(items).map_err(|e| e.to_string())?;
        fs::write(&self.path, raw)
            .await
            .map_err(|e| format!("write: {e}"))
    }

    fn render(items: &[Todo]) -> String {
        if items.is_empty() {
            return "(no todos)".to_string();
        }
        items
            .iter()
            .map(|t| {
                let mark = if t.done { "x" } else { " " };
                format!("- [{mark}] {}: {}", t.id, t.text)
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

#[async_trait]
impl Tool for TodoTool {
    fn name(&self) -> &str {
        "todo"
    }

    fn permission(&self) -> PermissionLevel {
        PermissionLevel::Write
    }

    fn description(&self) -> &str {
        "Working-memory scratchpad for multi-step artifact work. Before starting, \
         use `action: add` for a short plan; use `list` to recover progress across \
         tool calls. After publishing, validate the artifact, fix any errors, then \
         use `complete` to mark the item done by id. Use `clear` to wipe the list. \
         Persists to {workspace}/TODO.json between turns."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add", "list", "complete", "clear"],
                    "description": "Operation to perform"
                },
                "text": {
                    "type": "string",
                    "description": "Task text (required for action=add)"
                },
                "id": {
                    "type": "integer",
                    "description": "Task id (required for action=complete)"
                }
            },
            "required": ["action"]
        })
    }

    async fn execute(&self, params: HashMap<String, Value>) -> String {
        let action = params.get("action").and_then(|v| v.as_str()).unwrap_or("");
        let mut items = self.load().await;

        match action {
            "list" => Self::render(&items),
            "add" => {
                let Some(text) = params.get("text").and_then(|v| v.as_str()) else {
                    return "Error: 'text' is required for action=add".to_string();
                };
                let next_id = items.iter().map(|t| t.id).max().unwrap_or(0) + 1;
                items.push(Todo {
                    id: next_id,
                    text: text.to_string(),
                    done: false,
                });
                if let Err(e) = self.save(&items).await {
                    return format!("Error: {e}");
                }
                format!("Added todo #{next_id}: {text}")
            }
            "complete" => {
                let Some(id) = params.get("id").and_then(|v| v.as_u64()) else {
                    return "Error: 'id' is required for action=complete".to_string();
                };
                let id = id as u32;
                let Some(t) = items.iter_mut().find(|t| t.id == id) else {
                    return format!("Error: no todo with id={id}");
                };
                t.done = true;
                let text = t.text.clone();
                if let Err(e) = self.save(&items).await {
                    return format!("Error: {e}");
                }
                format!("Completed todo #{id}: {text}")
            }
            "clear" => {
                if let Err(e) = self.save(&[]).await {
                    return format!("Error: {e}");
                }
                "Cleared all todos".to_string()
            }
            other => format!("Error: unknown action '{other}' (use add|list|complete|clear)"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn tool_in(tmp: &TempDir) -> TodoTool {
        TodoTool::new(tmp.path())
    }

    fn params(pairs: &[(&str, Value)]) -> HashMap<String, Value> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect()
    }

    #[tokio::test]
    async fn test_list_on_empty_is_friendly() {
        let tmp = TempDir::new().unwrap();
        let tool = tool_in(&tmp);
        let out = tool.execute(params(&[("action", json!("list"))])).await;
        assert_eq!(out, "(no todos)");
    }

    #[tokio::test]
    async fn test_add_then_list_shows_pending() {
        let tmp = TempDir::new().unwrap();
        let tool = tool_in(&tmp);
        let added = tool
            .execute(params(&[
                ("action", json!("add")),
                ("text", json!("fix bug")),
            ]))
            .await;
        assert!(added.starts_with("Added todo #1"));
        let listed = tool.execute(params(&[("action", json!("list"))])).await;
        assert!(listed.contains("[ ] 1: fix bug"), "got: {listed}");
    }

    #[tokio::test]
    async fn test_complete_marks_done() {
        let tmp = TempDir::new().unwrap();
        let tool = tool_in(&tmp);
        tool.execute(params(&[("action", json!("add")), ("text", json!("t1"))]))
            .await;
        let done = tool
            .execute(params(&[("action", json!("complete")), ("id", json!(1))]))
            .await;
        assert!(done.starts_with("Completed todo #1"));
        let listed = tool.execute(params(&[("action", json!("list"))])).await;
        assert!(listed.contains("[x] 1: t1"), "got: {listed}");
    }

    #[tokio::test]
    async fn test_add_assigns_sequential_ids() {
        let tmp = TempDir::new().unwrap();
        let tool = tool_in(&tmp);
        for n in 1..=3 {
            let out = tool
                .execute(params(&[
                    ("action", json!("add")),
                    ("text", json!(format!("task {n}"))),
                ]))
                .await;
            assert!(out.contains(&format!("#{n}")), "got: {out}");
        }
    }

    #[tokio::test]
    async fn test_state_isolated_between_workspaces() {
        let a = TempDir::new().unwrap();
        let b = TempDir::new().unwrap();
        tool_in(&a)
            .execute(params(&[
                ("action", json!("add")),
                ("text", json!("only-a")),
            ]))
            .await;
        let listed_b = tool_in(&b)
            .execute(params(&[("action", json!("list"))]))
            .await;
        assert_eq!(listed_b, "(no todos)");
    }

    #[tokio::test]
    async fn test_persists_across_tool_instances() {
        // New tool instance against the same workspace must see the earlier state.
        let tmp = TempDir::new().unwrap();
        tool_in(&tmp)
            .execute(params(&[
                ("action", json!("add")),
                ("text", json!("persisted")),
            ]))
            .await;
        let listed = tool_in(&tmp)
            .execute(params(&[("action", json!("list"))]))
            .await;
        assert!(listed.contains("persisted"), "got: {listed}");
    }

    #[tokio::test]
    async fn test_complete_with_bad_id_errors() {
        let tmp = TempDir::new().unwrap();
        let tool = tool_in(&tmp);
        let out = tool
            .execute(params(&[("action", json!("complete")), ("id", json!(42))]))
            .await;
        assert!(out.starts_with("Error:"), "got: {out}");
    }

    #[tokio::test]
    async fn test_clear_wipes_all() {
        let tmp = TempDir::new().unwrap();
        let tool = tool_in(&tmp);
        tool.execute(params(&[("action", json!("add")), ("text", json!("gone"))]))
            .await;
        tool.execute(params(&[("action", json!("clear"))])).await;
        let listed = tool.execute(params(&[("action", json!("list"))])).await;
        assert_eq!(listed, "(no todos)");
    }

    #[test]
    fn test_schema_has_action_enum() {
        let tool = TodoTool::new(Path::new("/tmp"));
        let schema = tool.parameters();
        let variants = schema["properties"]["action"]["enum"].as_array().unwrap();
        let names: Vec<&str> = variants.iter().filter_map(|v| v.as_str()).collect();
        assert_eq!(names, vec!["add", "list", "complete", "clear"]);
    }

    #[test]
    fn test_description_teaches_artifact_plan_and_validation_workflow() {
        let tool = TodoTool::new(Path::new("/tmp"));
        let description = tool.description().to_ascii_lowercase();
        assert!(description.contains("multi-step artifact"), "{description}");
        assert!(description.contains("before"), "{description}");
        assert!(description.contains("validate"), "{description}");
        assert!(description.contains("complete"), "{description}");
    }
}
