//! Task board tool for agent coordination.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;

use super::base::Tool;
use crate::agent::taskboard::{TaskBoard, TaskStatus};

/// Tool exposing the shared task board to the agent.
pub struct TaskBoardTool {
    board: Arc<TaskBoard>,
}

impl TaskBoardTool {
    pub fn new(board: Arc<TaskBoard>) -> Self {
        Self { board }
    }
}

#[async_trait]
impl Tool for TaskBoardTool {
    fn name(&self) -> &str {
        "task_board"
    }

    fn description(&self) -> &str {
        "Manage a shared task board for coordinating work between agents. \
         Create tasks, update status, add progress notes, set handoff context, \
         and list all tasks."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create", "update", "progress", "list", "read", "handoff"],
                    "description": "Action to perform"
                },
                "task_id": {
                    "type": "string",
                    "description": "Task ID (required for update/progress/read/handoff)"
                },
                "label": {
                    "type": "string",
                    "description": "Task label (required for create)"
                },
                "description": {
                    "type": "string",
                    "description": "Task description (for create)"
                },
                "parent_id": {
                    "type": "string",
                    "description": "Parent task ID (optional, for create)"
                },
                "status": {
                    "type": "string",
                    "enum": ["pending", "in_progress", "blocked", "completed", "failed"],
                    "description": "New status (for update)"
                },
                "note": {
                    "type": "string",
                    "description": "Progress note (for progress)"
                },
                "context": {
                    "type": "string",
                    "description": "Handoff context (for handoff)"
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
            "create" => {
                let label = params.get("label").and_then(|v| v.as_str()).unwrap_or("Untitled");
                let description = params.get("description").and_then(|v| v.as_str()).unwrap_or("");
                let parent_id = params.get("parent_id").and_then(|v| v.as_str());
                let task = self.board.create_task(label, description, parent_id);
                format!("Created task [{}]: {}", task.task_id, task.label)
            }
            "update" => {
                let task_id = match params.get("task_id").and_then(|v| v.as_str()) {
                    Some(id) => id,
                    None => return "Error: 'task_id' is required for update".to_string(),
                };
                let status = match params.get("status").and_then(|v| v.as_str()) {
                    Some("pending") => TaskStatus::Pending,
                    Some("in_progress") => TaskStatus::InProgress,
                    Some("blocked") => TaskStatus::Blocked,
                    Some("completed") => TaskStatus::Completed,
                    Some("failed") => TaskStatus::Failed,
                    _ => return "Error: 'status' must be one of: pending, in_progress, blocked, completed, failed".to_string(),
                };
                match self.board.update_status(task_id, status) {
                    Some(task) => format!("Updated [{}] to {}", task.task_id, task.status),
                    None => format!("Error: Task '{}' not found", task_id),
                }
            }
            "progress" => {
                let task_id = match params.get("task_id").and_then(|v| v.as_str()) {
                    Some(id) => id,
                    None => return "Error: 'task_id' is required for progress".to_string(),
                };
                let note = params.get("note").and_then(|v| v.as_str()).unwrap_or("");
                match self.board.add_progress(task_id, note) {
                    Some(_) => format!("Added progress note to [{}]", task_id),
                    None => format!("Error: Task '{}' not found", task_id),
                }
            }
            "list" => {
                let tasks = self.board.list_tasks();
                if tasks.is_empty() {
                    "No tasks on the board.".to_string()
                } else {
                    let lines: Vec<String> = tasks
                        .iter()
                        .map(|t| {
                            format!(
                                "[{}] {} — {} ({})",
                                t.task_id,
                                t.label,
                                t.status,
                                t.assigned_to.as_deref().unwrap_or("unassigned")
                            )
                        })
                        .collect();
                    lines.join("\n")
                }
            }
            "read" => {
                let task_id = match params.get("task_id").and_then(|v| v.as_str()) {
                    Some(id) => id,
                    None => return "Error: 'task_id' is required for read".to_string(),
                };
                match self.board.get_task(task_id) {
                    Some(task) => serde_json::to_string_pretty(&task).unwrap_or_else(|_| "Error serializing task".to_string()),
                    None => format!("Error: Task '{}' not found", task_id),
                }
            }
            "handoff" => {
                let task_id = match params.get("task_id").and_then(|v| v.as_str()) {
                    Some(id) => id,
                    None => return "Error: 'task_id' is required for handoff".to_string(),
                };
                let context = params.get("context").and_then(|v| v.as_str()).unwrap_or("");
                match self.board.set_handoff_context(task_id, context) {
                    Some(_) => format!("Set handoff context for [{}]", task_id),
                    None => format!("Error: Task '{}' not found", task_id),
                }
            }
            _ => format!("Error: Unknown action '{}'. Use create, update, progress, list, read, or handoff.", action),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_task_board_tool_crud() {
        let dir = tempfile::tempdir().unwrap();
        let board = Arc::new(TaskBoard::new(dir.path()));
        let tool = TaskBoardTool::new(board);

        // Create.
        let mut params = HashMap::new();
        params.insert("action".to_string(), serde_json::json!("create"));
        params.insert("label".to_string(), serde_json::json!("Build feature"));
        params.insert("description".to_string(), serde_json::json!("Implement X"));
        let result = tool.execute(params).await;
        assert!(result.contains("Created task"));

        // Extract task ID.
        let task_id = result
            .split('[')
            .nth(1)
            .and_then(|s| s.split(']').next())
            .unwrap();

        // Update.
        let mut params = HashMap::new();
        params.insert("action".to_string(), serde_json::json!("update"));
        params.insert("task_id".to_string(), serde_json::json!(task_id));
        params.insert("status".to_string(), serde_json::json!("in_progress"));
        let result = tool.execute(params).await;
        assert!(result.contains("Updated"));

        // List.
        let mut params = HashMap::new();
        params.insert("action".to_string(), serde_json::json!("list"));
        let result = tool.execute(params).await;
        assert!(result.contains("Build feature"));

        // Read.
        let mut params = HashMap::new();
        params.insert("action".to_string(), serde_json::json!("read"));
        params.insert("task_id".to_string(), serde_json::json!(task_id));
        let result = tool.execute(params).await;
        assert!(result.contains("in_progress"));
    }

    #[test]
    fn test_tool_name() {
        let dir = tempfile::tempdir().unwrap();
        let board = Arc::new(TaskBoard::new(dir.path()));
        let tool = TaskBoardTool::new(board);
        assert_eq!(tool.name(), "task_board");
    }
}
