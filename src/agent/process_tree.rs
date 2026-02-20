#![allow(dead_code)]
use anyhow::Result;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use uuid::Uuid;

/// Status of a worker node in the process tree.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeStatus {
    /// Not yet started.
    Pending,
    /// Currently executing.
    Running,
    /// Completed successfully.
    Completed,
    /// Failed with error message.
    Failed(String),
    /// Skipped (e.g., parent failed).
    Skipped,
}

/// A single node in the process tree, representing one worker task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerNode {
    /// Unique identifier for this node.
    pub id: String,
    /// Human-readable task description.
    pub task: String,
    /// Status of this node.
    pub status: NodeStatus,
    /// Result text from the worker (if completed).
    pub result: Option<String>,
    /// Error message (if failed).
    pub error: Option<String>,
    /// Child worker nodes (subtasks).
    pub children: Vec<WorkerNode>,
    /// Step number within the parent (0-indexed).
    pub step_index: usize,
    /// Total steps expected in this level.
    pub total_steps: usize,
    /// Wall-clock duration in milliseconds (if completed/failed).
    pub duration_ms: Option<u64>,
    /// Timestamp when this node started.
    pub started_at: Option<String>,
    /// Timestamp when this node completed/failed.
    pub finished_at: Option<String>,
    /// Number of verification votes (from StepVoter) if verified.
    pub verification_votes: Option<usize>,
    /// Confidence from StepVoter verification (0.0-1.0).
    pub verification_confidence: Option<f64>,
}

impl WorkerNode {
    /// Create a new pending worker node.
    pub fn new(task: &str, step_index: usize, total_steps: usize) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            task: task.to_string(),
            status: NodeStatus::Pending,
            result: None,
            error: None,
            children: Vec::new(),
            step_index,
            total_steps,
            duration_ms: None,
            started_at: None,
            finished_at: None,
            verification_votes: None,
            verification_confidence: None,
        }
    }

    /// Mark this node as running.
    pub fn start(&mut self) {
        self.status = NodeStatus::Running;
        self.started_at = Some(Utc::now().to_rfc3339());
    }

    /// Mark this node as completed with a result.
    pub fn complete(&mut self, result: String, duration_ms: u64) {
        self.status = NodeStatus::Completed;
        self.result = Some(result);
        self.duration_ms = Some(duration_ms);
        self.finished_at = Some(Utc::now().to_rfc3339());
    }

    /// Mark this node as failed.
    pub fn fail(&mut self, error: String, duration_ms: u64) {
        self.status = NodeStatus::Failed(error.clone());
        self.error = Some(error);
        self.duration_ms = Some(duration_ms);
        self.finished_at = Some(Utc::now().to_rfc3339());
    }

    /// Skip this node and all children recursively.
    pub fn skip_recursive(&mut self) {
        self.status = NodeStatus::Skipped;
        for child in &mut self.children {
            child.skip_recursive();
        }
    }

    /// Add a child worker node.
    pub fn add_child(&mut self, child: WorkerNode) {
        self.children.push(child);
    }

    /// Find a node by ID (DFS search).
    pub fn find_by_id(&self, id: &str) -> Option<&WorkerNode> {
        if self.id == id {
            return Some(self);
        }
        for child in &self.children {
            if let Some(found) = child.find_by_id(id) {
                return Some(found);
            }
        }
        None
    }

    /// Find a mutable node by ID (DFS search).
    pub fn find_by_id_mut(&mut self, id: &str) -> Option<&mut WorkerNode> {
        if self.id == id {
            return Some(self);
        }
        for child in &mut self.children {
            if let Some(found) = child.find_by_id_mut(id) {
                return Some(found);
            }
        }
        None
    }

    /// Find the next pending node (DFS, depth-first).
    pub fn next_pending(&self) -> Option<&WorkerNode> {
        if self.status == NodeStatus::Pending {
            return Some(self);
        }
        for child in &self.children {
            if let Some(found) = child.next_pending() {
                return Some(found);
            }
        }
        None
    }

    /// Count nodes by status recursively.
    pub fn count_by_status(&self) -> StatusCounts {
        let mut counts = StatusCounts::default();
        self.count_recursive(&mut counts);
        counts
    }

    fn count_recursive(&self, counts: &mut StatusCounts) {
        match &self.status {
            NodeStatus::Pending => counts.pending += 1,
            NodeStatus::Running => counts.running += 1,
            NodeStatus::Completed => counts.completed += 1,
            NodeStatus::Failed(_) => counts.failed += 1,
            NodeStatus::Skipped => counts.skipped += 1,
        }
        for child in &self.children {
            child.count_recursive(counts);
        }
    }

    /// Total number of nodes in this subtree (including self).
    pub fn total_nodes(&self) -> usize {
        1 + self.children.iter().map(|c| c.total_nodes()).sum::<usize>()
    }

    /// Check if this node and all children are terminal (completed/failed/skipped).
    pub fn is_terminal(&self) -> bool {
        match self.status {
            NodeStatus::Completed | NodeStatus::Failed(_) | NodeStatus::Skipped => {
                self.children.iter().all(|c| c.is_terminal())
            }
            _ => false,
        }
    }

    /// Record verification result from StepVoter.
    pub fn set_verification(&mut self, votes: usize, confidence: f64) {
        self.verification_votes = Some(votes);
        self.verification_confidence = Some(confidence);
    }
}

/// Counts of nodes by status.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StatusCounts {
    pub pending: usize,
    pub running: usize,
    pub completed: usize,
    pub failed: usize,
    pub skipped: usize,
}

impl StatusCounts {
    pub fn total(&self) -> usize {
        self.pending + self.running + self.completed + self.failed + self.skipped
    }

    pub fn progress_pct(&self) -> f64 {
        let total = self.total();
        if total == 0 {
            return 0.0;
        }
        (self.completed as f64 / total as f64) * 100.0
    }
}

/// Verification checkpoints: verify at these step counts.
pub const GRADUATED_CHECKPOINTS: &[usize] = &[10, 100, 1_000, 10_000, 100_000, 1_000_000];

/// Determines if step N should trigger a verification checkpoint.
pub fn should_verify_at_step(step: usize) -> bool {
    GRADUATED_CHECKPOINTS.contains(&step)
}

/// A complete process tree with metadata and checkpoint support.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessTree {
    /// Unique process ID.
    pub id: String,
    /// Human-readable name for this process.
    pub name: String,
    /// Root node of the process tree.
    pub root: WorkerNode,
    /// When the process was created.
    pub created_at: String,
    /// When the process was last checkpointed.
    pub last_checkpoint: Option<String>,
    /// Total steps completed so far.
    pub steps_completed: usize,
    /// Next graduation verification step.
    pub next_verification_step: usize,
}

impl ProcessTree {
    /// Create a new process tree with a root task.
    pub fn new(name: &str, root_task: &str, total_steps: usize) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            name: name.to_string(),
            root: WorkerNode::new(root_task, 0, total_steps),
            created_at: Utc::now().to_rfc3339(),
            last_checkpoint: None,
            steps_completed: 0,
            next_verification_step: GRADUATED_CHECKPOINTS.first().copied().unwrap_or(10),
        }
    }

    /// Save the process tree to a JSON file.
    pub fn checkpoint(&mut self, dir: &Path) -> Result<PathBuf> {
        fs::create_dir_all(dir)?;
        let file_path = dir.join(format!("{}.json", self.id));
        self.last_checkpoint = Some(Utc::now().to_rfc3339());
        let json = serde_json::to_string_pretty(self)?;
        // Atomic write: write to temp, then rename
        let tmp_path = dir.join(format!("{}.tmp", self.id));
        fs::write(&tmp_path, &json)?;
        fs::rename(&tmp_path, &file_path)?;
        Ok(file_path)
    }

    /// Load a process tree from a JSON checkpoint file.
    pub fn load(path: &Path) -> Result<Self> {
        let json = fs::read_to_string(path)?;
        let tree: ProcessTree = serde_json::from_str(&json)?;
        Ok(tree)
    }

    /// List all process checkpoints in a directory.
    pub fn list_checkpoints(dir: &Path) -> Result<Vec<ProcessSummary>> {
        let mut summaries = Vec::new();
        if !dir.exists() {
            return Ok(summaries);
        }
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().map_or(false, |e| e == "json") {
                if let Ok(tree) = Self::load(&path) {
                    let counts = tree.root.count_by_status();
                    summaries.push(ProcessSummary {
                        id: tree.id,
                        name: tree.name,
                        created_at: tree.created_at,
                        last_checkpoint: tree.last_checkpoint,
                        steps_completed: tree.steps_completed,
                        total_nodes: tree.root.total_nodes(),
                        is_complete: tree.root.is_terminal(),
                        counts,
                    });
                }
            }
        }
        summaries.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        Ok(summaries)
    }

    /// Find the next pending node to resume.
    pub fn next_pending(&self) -> Option<&WorkerNode> {
        self.root.next_pending()
    }

    /// Increment steps_completed and check if we should verify.
    pub fn record_step(&mut self) -> bool {
        self.steps_completed += 1;
        if should_verify_at_step(self.steps_completed) {
            // Advance next_verification_step
            self.next_verification_step = GRADUATED_CHECKPOINTS
                .iter()
                .find(|&&s| s > self.steps_completed)
                .copied()
                .unwrap_or(usize::MAX);
            true // Should verify
        } else {
            false
        }
    }

    /// Get the default checkpoint directory: ~/.nanobot/processes/
    pub fn default_dir() -> PathBuf {
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".nanobot")
            .join("processes")
    }

    /// Get overall status counts.
    pub fn status_counts(&self) -> StatusCounts {
        self.root.count_by_status()
    }
}

/// Summary of a process checkpoint (for listing).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessSummary {
    pub id: String,
    pub name: String,
    pub created_at: String,
    pub last_checkpoint: Option<String>,
    pub steps_completed: usize,
    pub total_nodes: usize,
    pub is_complete: bool,
    pub counts: StatusCounts,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_worker_node_new() {
        let node = WorkerNode::new("Test task", 0, 5);

        assert_eq!(node.task, "Test task");
        assert_eq!(node.status, NodeStatus::Pending);
        assert_eq!(node.step_index, 0);
        assert_eq!(node.total_steps, 5);
        assert!(node.result.is_none());
        assert!(node.error.is_none());
        assert!(node.children.is_empty());
        assert!(node.duration_ms.is_none());
        assert!(node.started_at.is_none());
        assert!(node.finished_at.is_none());
        assert!(node.verification_votes.is_none());
        assert!(node.verification_confidence.is_none());
        assert!(!node.id.is_empty());
    }

    #[test]
    fn test_worker_node_lifecycle() {
        // Test start → complete
        let mut node = WorkerNode::new("Task", 0, 1);
        assert_eq!(node.status, NodeStatus::Pending);

        node.start();
        assert_eq!(node.status, NodeStatus::Running);
        assert!(node.started_at.is_some());

        node.complete("Success!".to_string(), 1234);
        assert_eq!(node.status, NodeStatus::Completed);
        assert_eq!(node.result, Some("Success!".to_string()));
        assert_eq!(node.duration_ms, Some(1234));
        assert!(node.finished_at.is_some());

        // Test start → fail
        let mut node2 = WorkerNode::new("Task2", 0, 1);
        node2.start();
        node2.fail("Error occurred".to_string(), 5678);

        assert_eq!(
            node2.status,
            NodeStatus::Failed("Error occurred".to_string())
        );
        assert_eq!(node2.error, Some("Error occurred".to_string()));
        assert_eq!(node2.duration_ms, Some(5678));
        assert!(node2.finished_at.is_some());
    }

    #[test]
    fn test_worker_node_skip_recursive() {
        let mut root = WorkerNode::new("Root", 0, 1);
        let mut child1 = WorkerNode::new("Child1", 0, 2);
        let child2 = WorkerNode::new("Child2", 1, 2);

        child1.add_child(child2);
        root.add_child(child1);

        root.skip_recursive();

        assert_eq!(root.status, NodeStatus::Skipped);
        assert_eq!(root.children[0].status, NodeStatus::Skipped);
        assert_eq!(root.children[0].children[0].status, NodeStatus::Skipped);
    }

    #[test]
    fn test_worker_node_find_by_id() {
        let mut root = WorkerNode::new("Root", 0, 1);
        let child1 = WorkerNode::new("Child1", 0, 2);
        let child2 = WorkerNode::new("Child2", 1, 2);

        let child1_id = child1.id.clone();
        let child2_id = child2.id.clone();

        root.add_child(child1);
        root.children[0].add_child(child2);

        // Find root
        assert!(root.find_by_id(&root.id).is_some());
        assert_eq!(root.find_by_id(&root.id).unwrap().task, "Root");

        // Find child1
        assert!(root.find_by_id(&child1_id).is_some());
        assert_eq!(root.find_by_id(&child1_id).unwrap().task, "Child1");

        // Find child2
        assert!(root.find_by_id(&child2_id).is_some());
        assert_eq!(root.find_by_id(&child2_id).unwrap().task, "Child2");

        // Not found
        assert!(root.find_by_id("nonexistent").is_none());
    }

    #[test]
    fn test_worker_node_find_by_id_mut() {
        let mut root = WorkerNode::new("Root", 0, 1);
        let child = WorkerNode::new("Child", 0, 1);
        let child_id = child.id.clone();

        root.add_child(child);

        // Find and mutate
        if let Some(node) = root.find_by_id_mut(&child_id) {
            node.task = "Modified Child".to_string();
        }

        assert_eq!(root.children[0].task, "Modified Child");
    }

    #[test]
    fn test_worker_node_next_pending() {
        let mut root = WorkerNode::new("Root", 0, 1);
        root.start();
        root.complete("Done".to_string(), 100);

        let mut child1 = WorkerNode::new("Child1", 0, 2);
        child1.start();
        child1.complete("Done".to_string(), 100);

        let child2 = WorkerNode::new("Child2", 1, 2);
        let child3 = WorkerNode::new("Child3", 2, 2);

        root.add_child(child1);
        root.add_child(child2);
        root.add_child(child3);

        // Should return first pending node in DFS order
        let next = root.next_pending();
        assert!(next.is_some());
        assert_eq!(next.unwrap().task, "Child2");
    }

    #[test]
    fn test_worker_node_count_by_status() {
        let mut root = WorkerNode::new("Root", 0, 1);
        root.start();

        let mut child1 = WorkerNode::new("Child1", 0, 3);
        child1.complete("Done".to_string(), 100);

        let mut child2 = WorkerNode::new("Child2", 1, 3);
        child2.fail("Error".to_string(), 50);

        let mut child3 = WorkerNode::new("Child3", 2, 3);
        child3.skip_recursive();

        let child4 = WorkerNode::new("Child4", 0, 1);

        root.add_child(child1);
        root.add_child(child2);
        root.add_child(child3);
        root.add_child(child4);

        let counts = root.count_by_status();

        assert_eq!(counts.running, 1); // root
        assert_eq!(counts.completed, 1); // child1
        assert_eq!(counts.failed, 1); // child2
        assert_eq!(counts.skipped, 1); // child3
        assert_eq!(counts.pending, 1); // child4
        assert_eq!(counts.total(), 5);
    }

    #[test]
    fn test_worker_node_total_nodes() {
        let mut root = WorkerNode::new("Root", 0, 1);
        let mut child1 = WorkerNode::new("Child1", 0, 2);
        let child2 = WorkerNode::new("Child2", 1, 2);
        let grandchild = WorkerNode::new("Grandchild", 0, 1);

        child1.add_child(grandchild);
        root.add_child(child1);
        root.add_child(child2);

        // root + child1 + child2 + grandchild = 4
        assert_eq!(root.total_nodes(), 4);
    }

    #[test]
    fn test_worker_node_is_terminal() {
        let mut root = WorkerNode::new("Root", 0, 1);
        root.complete("Done".to_string(), 100);

        let mut child1 = WorkerNode::new("Child1", 0, 2);
        child1.complete("Done".to_string(), 100);

        let mut child2 = WorkerNode::new("Child2", 1, 2);
        child2.fail("Error".to_string(), 50);

        root.add_child(child1);
        root.add_child(child2);

        // All nodes are completed/failed
        assert!(root.is_terminal());

        // Add a pending child
        let child3 = WorkerNode::new("Child3", 2, 2);
        root.add_child(child3);

        // Now not terminal
        assert!(!root.is_terminal());
    }

    #[test]
    fn test_process_tree_checkpoint_and_load() {
        let tmpdir = tempdir().unwrap();
        let mut tree = ProcessTree::new("Test Process", "Root task", 10);

        tree.root.start();
        tree.root.complete("Success".to_string(), 5000);

        // Checkpoint
        let path = tree.checkpoint(tmpdir.path()).unwrap();
        assert!(path.exists());

        // Load
        let loaded = ProcessTree::load(&path).unwrap();

        assert_eq!(loaded.id, tree.id);
        assert_eq!(loaded.name, tree.name);
        assert_eq!(loaded.root.task, tree.root.task);
        assert_eq!(loaded.root.status, NodeStatus::Completed);
        assert_eq!(loaded.root.result, Some("Success".to_string()));
        assert!(loaded.last_checkpoint.is_some());
    }

    #[test]
    fn test_process_tree_list_checkpoints() {
        let tmpdir = tempdir().unwrap();

        // Create multiple trees
        let mut tree1 = ProcessTree::new("Process 1", "Task 1", 5);
        tree1.checkpoint(tmpdir.path()).unwrap();

        let mut tree2 = ProcessTree::new("Process 2", "Task 2", 10);
        tree2.root.complete("Done".to_string(), 100);
        tree2.checkpoint(tmpdir.path()).unwrap();

        let mut tree3 = ProcessTree::new("Process 3", "Task 3", 3);
        tree3.root.fail("Error".to_string(), 50);
        tree3.checkpoint(tmpdir.path()).unwrap();

        // List checkpoints
        let summaries = ProcessTree::list_checkpoints(tmpdir.path()).unwrap();

        assert_eq!(summaries.len(), 3);

        // Verify sorted by created_at (newest first)
        assert!(summaries[0].created_at >= summaries[1].created_at);
        assert!(summaries[1].created_at >= summaries[2].created_at);

        // Verify summaries
        let process2 = summaries.iter().find(|s| s.name == "Process 2").unwrap();
        assert_eq!(process2.total_nodes, 1);
        assert_eq!(process2.counts.completed, 1);
        assert!(process2.is_complete);
    }

    #[test]
    fn test_process_tree_record_step_graduated() {
        let mut tree = ProcessTree::new("Test", "Root", 1);

        assert_eq!(tree.steps_completed, 0);
        assert_eq!(tree.next_verification_step, 10);

        // Steps 1-9: no verification
        for _ in 0..9 {
            assert!(!tree.record_step());
        }
        assert_eq!(tree.steps_completed, 9);

        // Step 10: verify
        assert!(tree.record_step());
        assert_eq!(tree.steps_completed, 10);
        assert_eq!(tree.next_verification_step, 100);

        // Steps 11-99: no verification
        for _ in 0..89 {
            assert!(!tree.record_step());
        }
        assert_eq!(tree.steps_completed, 99);

        // Step 100: verify
        assert!(tree.record_step());
        assert_eq!(tree.steps_completed, 100);
        assert_eq!(tree.next_verification_step, 1_000);

        // Step 1000
        for _ in 0..899 {
            assert!(!tree.record_step());
        }
        assert!(tree.record_step());
        assert_eq!(tree.steps_completed, 1_000);
        assert_eq!(tree.next_verification_step, 10_000);
    }

    #[test]
    fn test_should_verify_at_step() {
        assert!(should_verify_at_step(10));
        assert!(should_verify_at_step(100));
        assert!(should_verify_at_step(1_000));
        assert!(should_verify_at_step(10_000));
        assert!(should_verify_at_step(100_000));
        assert!(should_verify_at_step(1_000_000));

        assert!(!should_verify_at_step(1));
        assert!(!should_verify_at_step(9));
        assert!(!should_verify_at_step(11));
        assert!(!should_verify_at_step(99));
        assert!(!should_verify_at_step(101));
        assert!(!should_verify_at_step(999));
        assert!(!should_verify_at_step(1_001));
    }

    #[test]
    fn test_status_counts_progress() {
        let counts = StatusCounts {
            pending: 10,
            running: 2,
            completed: 30,
            failed: 3,
            skipped: 5,
        };

        assert_eq!(counts.total(), 50);
        assert_eq!(counts.progress_pct(), 60.0); // 30/50 = 0.6 = 60%

        // Empty counts
        let empty = StatusCounts::default();
        assert_eq!(empty.total(), 0);
        assert_eq!(empty.progress_pct(), 0.0);
    }

    #[test]
    fn test_process_tree_atomic_write() {
        let tmpdir = tempdir().unwrap();
        let mut tree = ProcessTree::new("Test", "Root", 1);

        // Checkpoint multiple times
        let path1 = tree.checkpoint(tmpdir.path()).unwrap();

        tree.root.complete("Updated".to_string(), 100);
        let path2 = tree.checkpoint(tmpdir.path()).unwrap();

        // Should be same file
        assert_eq!(path1, path2);

        // Load and verify it has the updated content
        let loaded = ProcessTree::load(&path2).unwrap();
        assert_eq!(loaded.root.result, Some("Updated".to_string()));

        // Verify no .tmp file left behind
        let tmp_files: Vec<_> = fs::read_dir(tmpdir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map_or(false, |ext| ext == "tmp"))
            .collect();

        assert!(tmp_files.is_empty(), "Temp files should be cleaned up");
    }

    #[test]
    fn test_worker_node_verification() {
        let mut node = WorkerNode::new("Task", 0, 1);

        assert!(node.verification_votes.is_none());
        assert!(node.verification_confidence.is_none());

        node.set_verification(5, 0.95);

        assert_eq!(node.verification_votes, Some(5));
        assert_eq!(node.verification_confidence, Some(0.95));
    }
}
