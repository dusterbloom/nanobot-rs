//! Task board for multi-agent coordination.
//!
//! Provides a shared task board where agents can create tasks, update status,
//! record progress, and hand off work to other agents. Each task is persisted
//! as a JSON file in `{workspace}/taskboard/{task_id}.json`.

use std::fs;
use std::path::{Path, PathBuf};

use chrono::Utc;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::utils::helpers::ensure_dir;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Task status.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    Pending,
    InProgress,
    Blocked,
    Completed,
    Failed,
}

impl std::fmt::Display for TaskStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TaskStatus::Pending => write!(f, "pending"),
            TaskStatus::InProgress => write!(f, "in_progress"),
            TaskStatus::Blocked => write!(f, "blocked"),
            TaskStatus::Completed => write!(f, "completed"),
            TaskStatus::Failed => write!(f, "failed"),
        }
    }
}

/// A task entry on the board.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskEntry {
    pub task_id: String,
    pub parent_id: Option<String>,
    pub label: String,
    pub description: String,
    pub status: TaskStatus,
    pub assigned_to: Option<String>,
    pub created_at: String,
    pub updated_at: String,
    pub progress_notes: Vec<String>,
    pub artifacts: Vec<String>,
    pub handoff_context: Option<String>,
}

impl TaskEntry {
    fn new(label: &str, description: &str, parent_id: Option<&str>) -> Self {
        let now = Utc::now().to_rfc3339();
        Self {
            task_id: Uuid::new_v4().to_string()[..8].to_string(),
            parent_id: parent_id.map(|s| s.to_string()),
            label: label.to_string(),
            description: description.to_string(),
            status: TaskStatus::Pending,
            assigned_to: None,
            created_at: now.clone(),
            updated_at: now,
            progress_notes: Vec::new(),
            artifacts: Vec::new(),
            handoff_context: None,
        }
    }
}

// ---------------------------------------------------------------------------
// TaskBoard
// ---------------------------------------------------------------------------

/// Shared task board backed by filesystem.
pub struct TaskBoard {
    dir: PathBuf,
}

impl TaskBoard {
    pub fn new(workspace: &Path) -> Self {
        let dir = workspace.join("taskboard");
        Self { dir }
    }

    /// Create a new task and return it.
    pub fn create_task(
        &self,
        label: &str,
        description: &str,
        parent_id: Option<&str>,
    ) -> TaskEntry {
        ensure_dir(&self.dir);
        let task = TaskEntry::new(label, description, parent_id);
        self.save_task(&task);
        task
    }

    /// Update the status of a task.
    pub fn update_status(&self, task_id: &str, status: TaskStatus) -> Option<TaskEntry> {
        let mut task = self.get_task(task_id)?;
        task.status = status;
        task.updated_at = Utc::now().to_rfc3339();
        self.save_task(&task);
        Some(task)
    }

    /// Add a progress note to a task.
    pub fn add_progress(&self, task_id: &str, note: &str) -> Option<TaskEntry> {
        let mut task = self.get_task(task_id)?;
        task.progress_notes.push(format!(
            "[{}] {}",
            Utc::now().format("%H:%M:%S"),
            note
        ));
        task.updated_at = Utc::now().to_rfc3339();
        self.save_task(&task);
        Some(task)
    }

    /// Set handoff context for a task.
    pub fn set_handoff_context(&self, task_id: &str, context: &str) -> Option<TaskEntry> {
        let mut task = self.get_task(task_id)?;
        task.handoff_context = Some(context.to_string());
        task.updated_at = Utc::now().to_rfc3339();
        self.save_task(&task);
        Some(task)
    }

    /// Add an artifact path to a task.
    pub fn add_artifact(&self, task_id: &str, artifact: &str) -> Option<TaskEntry> {
        let mut task = self.get_task(task_id)?;
        task.artifacts.push(artifact.to_string());
        task.updated_at = Utc::now().to_rfc3339();
        self.save_task(&task);
        Some(task)
    }

    /// Get a task by ID.
    pub fn get_task(&self, task_id: &str) -> Option<TaskEntry> {
        let path = self.task_path(task_id);
        let content = fs::read_to_string(path).ok()?;
        serde_json::from_str(&content).ok()
    }

    /// Get subtasks of a parent task.
    pub fn get_subtasks(&self, parent_id: &str) -> Vec<TaskEntry> {
        self.list_tasks()
            .into_iter()
            .filter(|t| t.parent_id.as_deref() == Some(parent_id))
            .collect()
    }

    /// List all tasks.
    pub fn list_tasks(&self) -> Vec<TaskEntry> {
        if !self.dir.is_dir() {
            return Vec::new();
        }

        let mut tasks = Vec::new();
        if let Ok(entries) = fs::read_dir(&self.dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) == Some("json") {
                    if let Ok(content) = fs::read_to_string(&path) {
                        if let Ok(task) = serde_json::from_str::<TaskEntry>(&content) {
                            tasks.push(task);
                        }
                    }
                }
            }
        }

        tasks.sort_by(|a, b| a.created_at.cmp(&b.created_at));
        tasks
    }

    /// Format a brief summary of active tasks for the system prompt.
    pub fn summary(&self) -> String {
        let tasks = self.list_tasks();
        if tasks.is_empty() {
            return String::new();
        }

        let active: Vec<&TaskEntry> = tasks
            .iter()
            .filter(|t| {
                matches!(
                    t.status,
                    TaskStatus::Pending | TaskStatus::InProgress | TaskStatus::Blocked
                )
            })
            .collect();

        if active.is_empty() {
            return String::new();
        }

        let mut lines = Vec::new();
        for task in &active {
            let assigned = task
                .assigned_to
                .as_deref()
                .unwrap_or("unassigned");
            lines.push(format!(
                "- [{}] {} ({}) — {}",
                task.task_id, task.label, task.status, assigned
            ));
        }

        format!("Active tasks:\n{}", lines.join("\n"))
    }

    fn task_path(&self, task_id: &str) -> PathBuf {
        self.dir.join(format!("{}.json", task_id))
    }

    fn save_task(&self, task: &TaskEntry) {
        ensure_dir(&self.dir);
        let path = self.task_path(&task.task_id);
        if let Ok(json) = serde_json::to_string_pretty(task) {
            if let Err(e) = fs::write(path, json) {
                tracing::warn!("Failed to save task {}: {}", task.task_id, e);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_and_get_task() {
        let dir = tempfile::tempdir().unwrap();
        let board = TaskBoard::new(dir.path());

        let task = board.create_task("Build API", "Implement REST endpoints", None);
        assert_eq!(task.label, "Build API");
        assert_eq!(task.status, TaskStatus::Pending);

        let loaded = board.get_task(&task.task_id).unwrap();
        assert_eq!(loaded.label, "Build API");
    }

    #[test]
    fn test_update_status() {
        let dir = tempfile::tempdir().unwrap();
        let board = TaskBoard::new(dir.path());

        let task = board.create_task("Test task", "desc", None);
        let updated = board.update_status(&task.task_id, TaskStatus::InProgress).unwrap();
        assert_eq!(updated.status, TaskStatus::InProgress);

        let completed = board.update_status(&task.task_id, TaskStatus::Completed).unwrap();
        assert_eq!(completed.status, TaskStatus::Completed);
    }

    #[test]
    fn test_progress_notes() {
        let dir = tempfile::tempdir().unwrap();
        let board = TaskBoard::new(dir.path());

        let task = board.create_task("Test task", "desc", None);
        board.add_progress(&task.task_id, "Started work");
        board.add_progress(&task.task_id, "50% complete");

        let loaded = board.get_task(&task.task_id).unwrap();
        assert_eq!(loaded.progress_notes.len(), 2);
    }

    #[test]
    fn test_subtasks() {
        let dir = tempfile::tempdir().unwrap();
        let board = TaskBoard::new(dir.path());

        let parent = board.create_task("Parent", "Main task", None);
        board.create_task("Child 1", "Subtask 1", Some(&parent.task_id));
        board.create_task("Child 2", "Subtask 2", Some(&parent.task_id));

        let subtasks = board.get_subtasks(&parent.task_id);
        assert_eq!(subtasks.len(), 2);
    }

    #[test]
    fn test_list_tasks() {
        let dir = tempfile::tempdir().unwrap();
        let board = TaskBoard::new(dir.path());

        board.create_task("Task 1", "desc 1", None);
        board.create_task("Task 2", "desc 2", None);

        let tasks = board.list_tasks();
        assert_eq!(tasks.len(), 2);
    }

    #[test]
    fn test_handoff_context() {
        let dir = tempfile::tempdir().unwrap();
        let board = TaskBoard::new(dir.path());

        let task = board.create_task("Test", "desc", None);
        board.set_handoff_context(&task.task_id, "Continue from step 3");

        let loaded = board.get_task(&task.task_id).unwrap();
        assert_eq!(loaded.handoff_context.as_deref(), Some("Continue from step 3"));
    }

    #[test]
    fn test_artifacts() {
        let dir = tempfile::tempdir().unwrap();
        let board = TaskBoard::new(dir.path());

        let task = board.create_task("Test", "desc", None);
        board.add_artifact(&task.task_id, "/path/to/output.txt");

        let loaded = board.get_task(&task.task_id).unwrap();
        assert_eq!(loaded.artifacts.len(), 1);
        assert_eq!(loaded.artifacts[0], "/path/to/output.txt");
    }

    #[test]
    fn test_summary() {
        let dir = tempfile::tempdir().unwrap();
        let board = TaskBoard::new(dir.path());

        let task = board.create_task("Active task", "doing stuff", None);
        board.update_status(&task.task_id, TaskStatus::InProgress);
        board.create_task("Completed task", "done", None);

        let summary = board.summary();
        assert!(summary.contains("Active task"));
    }

    #[test]
    fn test_nonexistent_task() {
        let dir = tempfile::tempdir().unwrap();
        let board = TaskBoard::new(dir.path());

        assert!(board.get_task("nonexistent").is_none());
        assert!(board.update_status("nonexistent", TaskStatus::Failed).is_none());
    }
}
