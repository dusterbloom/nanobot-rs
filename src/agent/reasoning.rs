//! Branching reasoning engine for plan-guided execution and backtracking.
//!
//! Provides checkpointing, plan decomposition (as a DAG), and step-by-step
//! execution — enabling both thinking and non-thinking models to handle
//! complex multi-step tasks with recovery from dead ends.

use daggy::{Dag, NodeIndex, Walker};
use serde_json::Value;
use std::time::Instant;

/// A single step in a reasoning plan.
#[derive(Debug, Clone)]
pub struct PlanStep {
    pub id: usize,
    pub goal: String,
    pub status: StepStatus,
    pub result: Option<String>,
    pub max_iterations: u32,
    pub iterations_used: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub enum StepStatus {
    Pending,
    Active,
    Completed,
    Failed(String),
    Skipped,
}

/// Edge types between plan steps.
#[derive(Debug, Clone)]
pub enum EdgeType {
    /// Step B depends on step A completing successfully.
    Dependency,
    /// Step B is an alternative to step A (try if A fails).
    Alternative,
}

/// Snapshot of conversation state at a decision point.
#[derive(Debug, Clone)]
pub struct Checkpoint {
    pub label: String,
    pub messages: Vec<Value>,
    pub step_index: Option<usize>,
    pub iteration: u32,
    pub created_at: Instant,
}

/// Record of a branching attempt and its outcome.
#[derive(Debug, Clone)]
pub struct BranchAttempt {
    pub step_id: usize,
    pub approach: String,
    pub outcome: StepStatus,
    pub iterations_consumed: u32,
}

/// How the reasoning engine operates.
#[derive(Debug, Clone, PartialEq)]
pub enum ReasoningMode {
    /// No plan — behave like today's linear loop.
    Linear,
    /// Plan exists — feed steps one at a time.
    PlanGuided,
    /// Autonomous — engine decides when to checkpoint/backtrack.
    Autonomous,
}

/// The reasoning engine — owns plan DAG and checkpoint stack.
pub struct ReasoningEngine {
    plan: Option<Dag<PlanStep, EdgeType>>,
    checkpoints: Vec<Checkpoint>,
    current_step: Option<NodeIndex>,
    branch_history: Vec<BranchAttempt>,
    mode: ReasoningMode,
    max_checkpoints: usize,
    step_budget: u32,
    /// Messages snapshot for checkpoint tool to read from.
    current_messages: Vec<Value>,
    /// Pending restore from backtrack — agent loop consumes this.
    pending_restore: Option<Vec<Value>>,
}

impl ReasoningEngine {
    pub fn new() -> Self {
        Self {
            plan: None,
            checkpoints: Vec::new(),
            current_step: None,
            branch_history: Vec::new(),
            mode: ReasoningMode::Linear,
            max_checkpoints: 10,
            step_budget: 5,
            current_messages: Vec::new(),
            pending_restore: None,
        }
    }

    pub fn new_with_plan(mut plan: Dag<PlanStep, EdgeType>, step_budget: u32) -> Self {
        // Find the root node: the node with no incoming Dependency edges.
        // In daggy, parents(idx) walks incoming edges.
        // We look for a node that has no parents at all (no incoming edges).
        let root = {
            let mut found = None;
            let node_count = plan.node_count();
            'outer: for i in 0..node_count {
                let idx = NodeIndex::new(i);
                let mut parents = plan.parents(idx);
                if parents.walk_next(&plan).is_none() {
                    found = Some(idx);
                    break 'outer;
                }
            }
            found
        };

        if let Some(root_idx) = root {
            if let Some(step) = plan.node_weight_mut(root_idx) {
                step.status = StepStatus::Active;
            }
        }

        Self {
            plan: Some(plan),
            checkpoints: Vec::new(),
            current_step: root,
            branch_history: Vec::new(),
            mode: ReasoningMode::PlanGuided,
            max_checkpoints: 10,
            step_budget,
            current_messages: Vec::new(),
            pending_restore: None,
        }
    }

    // Checkpoint management

    pub fn save_checkpoint(&mut self, label: &str, messages: &[Value], iteration: u32) {
        let step_index = self.current_step.map(|idx| idx.index());
        let cp = Checkpoint {
            label: label.to_string(),
            messages: messages.to_vec(),
            step_index,
            iteration,
            created_at: Instant::now(),
        };
        self.checkpoints.push(cp);
        // Evict oldest if over limit
        while self.checkpoints.len() > self.max_checkpoints {
            self.checkpoints.remove(0);
        }
    }

    pub fn pop_checkpoint(&mut self) -> Option<Checkpoint> {
        self.checkpoints.pop()
    }

    pub fn find_checkpoint(&self, label: &str) -> Option<&Checkpoint> {
        self.checkpoints.iter().find(|cp| cp.label == label)
    }

    pub fn checkpoint_count(&self) -> usize {
        self.checkpoints.len()
    }

    // Plan navigation

    pub fn current_step(&self) -> Option<&PlanStep> {
        let plan = self.plan.as_ref()?;
        let idx = self.current_step?;
        plan.node_weight(idx)
    }

    pub fn advance(&mut self) -> Option<&PlanStep> {
        let plan = self.plan.as_mut()?;
        let current_idx = self.current_step?;

        // Walk children of current node, find first Dependency child that is Pending.
        let mut children_walker = plan.children(current_idx);
        let mut next_idx = None;
        loop {
            match children_walker.walk_next(plan) {
                None => break,
                Some((edge_idx, child_idx)) => {
                    let edge_type = plan.edge_weight(edge_idx)?;
                    if matches!(edge_type, EdgeType::Dependency) {
                        if let Some(child) = plan.node_weight(child_idx) {
                            if child.status == StepStatus::Pending {
                                next_idx = Some(child_idx);
                                break;
                            }
                        }
                    }
                }
            }
        }

        if let Some(idx) = next_idx {
            self.current_step = Some(idx);
            if let Some(step) = plan.node_weight_mut(idx) {
                step.status = StepStatus::Active;
            }
            plan.node_weight(idx)
        } else {
            self.current_step = None;
            None
        }
    }

    pub fn mark_current_failed(&mut self, reason: &str) {
        if let Some(plan) = self.plan.as_mut() {
            if let Some(idx) = self.current_step {
                if let Some(step) = plan.node_weight_mut(idx) {
                    step.status = StepStatus::Failed(reason.to_string());
                }
            }
        }
    }

    pub fn mark_current_completed(&mut self, result: Option<String>) {
        if let Some(plan) = self.plan.as_mut() {
            if let Some(idx) = self.current_step {
                if let Some(step) = plan.node_weight_mut(idx) {
                    step.status = StepStatus::Completed;
                    step.result = result;
                }
            }
        }
    }

    pub fn find_alternative(&self) -> Option<NodeIndex> {
        let plan = self.plan.as_ref()?;
        let current_idx = self.current_step?;

        // The current step must be failed.
        let current = plan.node_weight(current_idx)?;
        if !matches!(current.status, StepStatus::Failed(_)) {
            return None;
        }

        // Find parent of current step.
        let mut parents_walker = plan.parents(current_idx);
        let parent_idx = loop {
            match parents_walker.walk_next(plan) {
                None => return None,
                Some((_edge_idx, p_idx)) => break p_idx,
            }
        };

        // Walk parent's children, find one connected via Alternative edge that is Pending.
        let mut children_walker = plan.children(parent_idx);
        loop {
            match children_walker.walk_next(plan) {
                None => return None,
                Some((edge_idx, child_idx)) => {
                    let edge_type = plan.edge_weight(edge_idx)?;
                    if matches!(edge_type, EdgeType::Alternative) {
                        if let Some(child) = plan.node_weight(child_idx) {
                            if child.status == StepStatus::Pending {
                                return Some(child_idx);
                            }
                        }
                    }
                }
            }
        }
    }

    pub fn is_complete(&self) -> bool {
        let plan = match self.plan.as_ref() {
            Some(p) => p,
            None => return false,
        };
        // All nodes must be Completed or Skipped.
        let node_count = plan.node_count();
        for i in 0..node_count {
            let idx = NodeIndex::new(i);
            match plan.node_weight(idx) {
                Some(step) => {
                    if !matches!(step.status, StepStatus::Completed | StepStatus::Skipped) {
                        return false;
                    }
                }
                None => {}
            }
        }
        true
    }

    // Step instruction for dumb models

    pub fn step_instruction(&self) -> Option<String> {
        if self.mode != ReasoningMode::PlanGuided {
            return None;
        }
        let plan = self.plan.as_ref()?;
        let idx = self.current_step?;
        let step = plan.node_weight(idx)?;
        let total = plan.node_count();
        Some(format!("Step {}/{}: {}", step.id + 1, total, step.goal))
    }

    // Iteration budget per step

    pub fn step_budget_remaining(&self) -> u32 {
        if let Some(plan) = self.plan.as_ref() {
            if let Some(idx) = self.current_step {
                if let Some(step) = plan.node_weight(idx) {
                    return self.step_budget.saturating_sub(step.iterations_used);
                }
            }
        }
        self.step_budget
    }

    pub fn consume_iteration(&mut self) {
        if let Some(plan) = self.plan.as_mut() {
            if let Some(idx) = self.current_step {
                if let Some(step) = plan.node_weight_mut(idx) {
                    step.iterations_used += 1;
                }
            }
        }
    }

    // Getters

    pub fn mode(&self) -> &ReasoningMode {
        &self.mode
    }

    /// Update the maximum checkpoint stack depth (from config).
    pub fn set_max_checkpoints(&mut self, max: usize) {
        self.max_checkpoints = max;
    }

    pub fn branch_history(&self) -> &[BranchAttempt] {
        &self.branch_history
    }

    /// Sync current conversation messages so CheckpointTool can read them.
    pub fn sync_messages(&mut self, messages: &[Value]) {
        self.current_messages = messages.to_vec();
    }

    /// Get reference to current messages (for CheckpointTool).
    pub fn current_messages(&self) -> &[Value] {
        &self.current_messages
    }

    /// Set pending restore (called by BacktrackTool).
    pub fn set_pending_restore(&mut self, messages: Vec<Value>) {
        self.pending_restore = Some(messages);
    }

    /// Take pending restore (consumed by agent loop).
    pub fn take_pending_restore(&mut self) -> Option<Vec<Value>> {
        self.pending_restore.take()
    }

    /// Record a branch attempt.
    pub fn record_branch(&mut self, attempt: BranchAttempt) {
        self.branch_history.push(attempt);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to build a simple linear plan: A -> B -> C
    fn linear_plan() -> Dag<PlanStep, EdgeType> {
        let mut dag = Dag::new();
        let a = dag.add_node(PlanStep {
            id: 0,
            goal: "Step A".into(),
            status: StepStatus::Pending,
            result: None,
            max_iterations: 5,
            iterations_used: 0,
        });
        let b = dag.add_node(PlanStep {
            id: 1,
            goal: "Step B".into(),
            status: StepStatus::Pending,
            result: None,
            max_iterations: 5,
            iterations_used: 0,
        });
        let c = dag.add_node(PlanStep {
            id: 2,
            goal: "Step C".into(),
            status: StepStatus::Pending,
            result: None,
            max_iterations: 5,
            iterations_used: 0,
        });
        dag.add_edge(a, b, EdgeType::Dependency).unwrap();
        dag.add_edge(b, c, EdgeType::Dependency).unwrap();
        dag
    }

    // Helper to build a branching plan: A -> B, A -> C (C is alternative to B)
    fn branching_plan() -> Dag<PlanStep, EdgeType> {
        let mut dag = Dag::new();
        let a = dag.add_node(PlanStep {
            id: 0,
            goal: "Step A".into(),
            status: StepStatus::Pending,
            result: None,
            max_iterations: 5,
            iterations_used: 0,
        });
        let b = dag.add_node(PlanStep {
            id: 1,
            goal: "Step B".into(),
            status: StepStatus::Pending,
            result: None,
            max_iterations: 5,
            iterations_used: 0,
        });
        let c = dag.add_node(PlanStep {
            id: 2,
            goal: "Step C (alt)".into(),
            status: StepStatus::Pending,
            result: None,
            max_iterations: 5,
            iterations_used: 0,
        });
        dag.add_edge(a, b, EdgeType::Dependency).unwrap();
        dag.add_edge(a, c, EdgeType::Alternative).unwrap();
        dag
    }

    // --- Checkpoint tests ---

    #[test]
    fn test_checkpoint_save_and_restore() {
        let mut engine = ReasoningEngine::new();
        let msgs = vec![serde_json::json!({"role": "user", "content": "hello"})];
        engine.save_checkpoint("cp1", &msgs, 3);
        let cp = engine.pop_checkpoint().unwrap();
        assert_eq!(cp.label, "cp1");
        assert_eq!(cp.messages.len(), 1);
        assert_eq!(cp.iteration, 3);
    }

    #[test]
    fn test_checkpoint_stack_lifo_order() {
        let mut engine = ReasoningEngine::new();
        engine.save_checkpoint("first", &[], 1);
        engine.save_checkpoint("second", &[], 2);
        engine.save_checkpoint("third", &[], 3);
        assert_eq!(engine.pop_checkpoint().unwrap().label, "third");
        assert_eq!(engine.pop_checkpoint().unwrap().label, "second");
        assert_eq!(engine.pop_checkpoint().unwrap().label, "first");
        assert!(engine.pop_checkpoint().is_none());
    }

    #[test]
    fn test_checkpoint_find_by_label() {
        let mut engine = ReasoningEngine::new();
        engine.save_checkpoint("alpha", &[], 1);
        engine.save_checkpoint("beta", &[], 2);
        let found = engine.find_checkpoint("alpha").unwrap();
        assert_eq!(found.label, "alpha");
        assert!(engine.find_checkpoint("gamma").is_none());
    }

    #[test]
    fn test_checkpoint_max_eviction() {
        let mut engine = ReasoningEngine::new();
        // Default max_checkpoints should be reasonable (e.g. 10)
        // Save 12 checkpoints — first 2 should be evicted
        for i in 0..12 {
            engine.save_checkpoint(&format!("cp{}", i), &[], i as u32);
        }
        assert!(engine.checkpoint_count() <= 10);
        // Oldest should be gone
        assert!(engine.find_checkpoint("cp0").is_none());
    }

    // --- Plan DAG tests ---

    #[test]
    fn test_create_linear_plan() {
        let plan = linear_plan();
        let engine = ReasoningEngine::new_with_plan(plan, 5);
        assert_eq!(*engine.mode(), ReasoningMode::PlanGuided);
        assert!(!engine.is_complete());
    }

    #[test]
    fn test_advance_step_linear() {
        let plan = linear_plan();
        let mut engine = ReasoningEngine::new_with_plan(plan, 5);
        // First step should be A
        let step = engine.current_step().unwrap();
        assert_eq!(step.goal, "Step A");
        // Mark A completed, advance to B
        engine.mark_current_completed(Some("done A".into()));
        let next = engine.advance().unwrap();
        assert_eq!(next.goal, "Step B");
    }

    #[test]
    fn test_advance_through_all_steps() {
        let plan = linear_plan();
        let mut engine = ReasoningEngine::new_with_plan(plan, 5);
        // Complete A -> B -> C
        engine.mark_current_completed(None);
        engine.advance();
        engine.mark_current_completed(None);
        engine.advance();
        engine.mark_current_completed(None);
        assert!(engine.is_complete());
        assert!(engine.advance().is_none());
    }

    #[test]
    fn test_find_alternative_after_failure() {
        let plan = branching_plan();
        let mut engine = ReasoningEngine::new_with_plan(plan, 5);
        // Complete A
        engine.mark_current_completed(None);
        engine.advance(); // now at B
        // B fails
        engine.mark_current_failed("tool error");
        let alt = engine.find_alternative();
        assert!(alt.is_some()); // Should find C as alternative
    }

    #[test]
    fn test_no_alternative_when_none_exists() {
        let plan = linear_plan();
        let mut engine = ReasoningEngine::new_with_plan(plan, 5);
        engine.mark_current_failed("error");
        assert!(engine.find_alternative().is_none());
    }

    // --- ReasoningEngine mode tests ---

    #[test]
    fn test_engine_linear_mode_default() {
        let engine = ReasoningEngine::new();
        assert_eq!(*engine.mode(), ReasoningMode::Linear);
    }

    #[test]
    fn test_engine_plan_guided_has_instruction() {
        let plan = linear_plan();
        let engine = ReasoningEngine::new_with_plan(plan, 5);
        let instruction = engine.step_instruction().unwrap();
        assert!(instruction.contains("Step A"));
    }

    #[test]
    fn test_engine_linear_no_instruction() {
        let engine = ReasoningEngine::new();
        assert!(engine.step_instruction().is_none());
    }

    // --- Iteration budget tests ---

    #[test]
    fn test_step_budget_remaining() {
        let plan = linear_plan();
        let mut engine = ReasoningEngine::new_with_plan(plan, 5);
        assert_eq!(engine.step_budget_remaining(), 5);
        engine.consume_iteration();
        assert_eq!(engine.step_budget_remaining(), 4);
    }

    #[test]
    fn test_step_budget_exhausted() {
        let plan = linear_plan();
        let mut engine = ReasoningEngine::new_with_plan(plan, 2);
        engine.consume_iteration();
        engine.consume_iteration();
        assert_eq!(engine.step_budget_remaining(), 0);
    }

    // --- Backtracking integration test ---

    #[test]
    fn test_backtrack_restores_messages() {
        let mut engine = ReasoningEngine::new();
        let msgs_v1 = vec![serde_json::json!({"role": "user", "content": "v1"})];
        engine.save_checkpoint("before_attempt", &msgs_v1, 0);
        // Simulate: messages grew to v2, attempt failed
        let cp = engine.pop_checkpoint().unwrap();
        assert_eq!(cp.messages.len(), 1); // restored to v1
        assert_eq!(cp.messages[0]["content"], "v1");
    }
}
