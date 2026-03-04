//! Reasoning tools: checkpoint, backtrack, and plan.
//!
//! These tools expose the ReasoningEngine to the LLM so it can save
//! conversation snapshots, restore them on failure, and decompose tasks
//! into a dependency DAG.

use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::Mutex;

use async_trait::async_trait;
use daggy::Dag;

use super::base::Tool;
use crate::agent::reasoning::{EdgeType, PlanStep, ReasoningEngine, StepStatus};

/// Shared handle to the reasoning engine, passed into each tool.
pub type SharedEngine = Arc<Mutex<ReasoningEngine>>;

// ---------------------------------------------------------------------------
// CheckpointTool
// ---------------------------------------------------------------------------

/// Save a snapshot of the current conversation state for later backtracking.
///
/// Reads current messages from the engine's `current_messages` field, which
/// is kept in sync by the agent loop via `engine.sync_messages()` before each
/// tool execution.
pub struct CheckpointTool {
    engine: SharedEngine,
}

impl CheckpointTool {
    pub fn new(engine: SharedEngine) -> Self {
        Self { engine }
    }
}

#[async_trait]
impl Tool for CheckpointTool {
    fn name(&self) -> &str {
        "checkpoint"
    }

    fn description(&self) -> &str {
        "Save a snapshot of the current conversation state for potential backtracking"
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "label": {
                    "type": "string",
                    "description": "Name for this checkpoint (auto-generated if omitted)"
                }
            },
            "required": []
        })
    }

    async fn execute(&self, params: HashMap<String, serde_json::Value>) -> String {
        let label = params
            .get("label")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| {
                format!(
                    "cp-{}",
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_secs())
                        .unwrap_or(0)
                )
            });

        let mut engine = self.engine.lock();
        let msgs = engine.current_messages().to_vec();
        engine.save_checkpoint(&label, &msgs, 0);
        let n = engine.checkpoint_count();
        format!("Checkpoint '{}' saved. {} checkpoints on stack.", label, n)
    }
}

// ---------------------------------------------------------------------------
// BacktrackTool
// ---------------------------------------------------------------------------

/// Restore conversation state to a previous checkpoint.
///
/// Returns a JSON signal so the agent loop can apply the restored messages.
/// Format: `{"backtrack": true, "label": "...", "messages": [...]}`
pub struct BacktrackTool {
    engine: SharedEngine,
}

impl BacktrackTool {
    pub fn new(engine: SharedEngine) -> Self {
        Self { engine }
    }
}

#[async_trait]
impl Tool for BacktrackTool {
    fn name(&self) -> &str {
        "backtrack"
    }

    fn description(&self) -> &str {
        "Restore conversation state to a previous checkpoint, abandoning the current approach"
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Why the current approach failed"
                },
                "label": {
                    "type": "string",
                    "description": "Specific checkpoint label to restore (uses latest if omitted)"
                }
            },
            "required": ["reason"]
        })
    }

    async fn execute(&self, params: HashMap<String, serde_json::Value>) -> String {
        let reason = params
            .get("reason")
            .and_then(|v| v.as_str())
            .unwrap_or("unspecified");

        let label = params
            .get("label")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let mut engine = self.engine.lock();

        let checkpoint = if let Some(ref lbl) = label {
            // Named backtrack: find by label, then pop the stack to that point.
            // For simplicity, pop until we find the label or exhaust.
            let mut found = None;
            loop {
                match engine.pop_checkpoint() {
                    None => break,
                    Some(cp) if cp.label == *lbl => {
                        found = Some(cp);
                        break;
                    }
                    Some(_) => {}
                }
            }
            found
        } else {
            engine.pop_checkpoint()
        };

        match checkpoint {
            None => format!(
                "Error: No checkpoint available to backtrack to (reason: {})",
                reason
            ),
            Some(cp) => {
                let cp_label = cp.label.clone();
                engine.set_pending_restore(cp.messages.clone());
                format!(
                    "Backtracking to checkpoint '{}'. Reason: {}. Conversation state will be restored.",
                    cp_label, reason
                )
            }
        }
    }
}

// ---------------------------------------------------------------------------
// PlanTool
// ---------------------------------------------------------------------------

/// Decompose a task into numbered steps with optional dependencies.
pub struct PlanTool {
    engine: SharedEngine,
}

impl PlanTool {
    pub fn new(engine: SharedEngine) -> Self {
        Self { engine }
    }
}

#[async_trait]
impl Tool for PlanTool {
    fn name(&self) -> &str {
        "plan"
    }

    fn description(&self) -> &str {
        "Decompose a task into numbered steps with dependencies.\n\
         Example: [{\"goal\": \"read config file\"}, {\"goal\": \"validate settings\", \"depends_on\": [0]}]\n\
         depends_on uses 0-based indices: [0] means 'after step 0 (the first step)'."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "description": "Ordered list of plan steps",
                    "items": {
                        "type": "object",
                        "properties": {
                            "goal": {
                                "type": "string",
                                "description": "What this step should accomplish"
                            },
                            "depends_on": {
                                "type": "array",
                                "description": "0-based step indices this step waits for. [0]=after first step, [0,1]=after steps 0 and 1. Omit or [] for no dependencies.",
                                "items": { "type": "integer" }
                            }
                        },
                        "required": ["goal"]
                    }
                },
                "step_budget": {
                    "type": "integer",
                    "description": "Max tool iterations per step (default 5)"
                }
            },
            "required": ["steps"]
        })
    }

    async fn execute(&self, params: HashMap<String, serde_json::Value>) -> String {
        let steps_val = match params.get("steps") {
            Some(v) => v,
            None => return "Error: 'steps' parameter is required".to_string(),
        };

        let steps_arr = match steps_val.as_array() {
            Some(a) => a,
            None => return "Error: 'steps' must be an array".to_string(),
        };

        if steps_arr.is_empty() {
            return "Error: 'steps' array must not be empty".to_string();
        }

        let step_budget = params
            .get("step_budget")
            .and_then(|v| v.as_u64())
            .unwrap_or(5) as u32;

        // Build the DAG.
        let mut dag: Dag<PlanStep, EdgeType> = Dag::new();
        let mut node_indices = Vec::new();

        for (i, step_val) in steps_arr.iter().enumerate() {
            let goal = step_val
                .get("goal")
                .and_then(|g| g.as_str())
                .unwrap_or("(unnamed)")
                .to_string();

            let idx = dag.add_node(PlanStep {
                id: i,
                goal,
                status: StepStatus::Pending,
                result: None,
                max_iterations: step_budget,
                iterations_used: 0,
            });
            node_indices.push(idx);
        }

        // Wire dependency edges.
        for (i, step_val) in steps_arr.iter().enumerate() {
            if let Some(deps) = step_val.get("depends_on").and_then(|d| d.as_array()) {
                for dep_val in deps {
                    let dep_idx_raw = match dep_val.as_u64() {
                        Some(n) => n as usize,
                        None => continue,
                    };
                    if dep_idx_raw >= node_indices.len() {
                        return format!(
                            "Error: step {} depends_on index {} is out of range",
                            i, dep_idx_raw
                        );
                    }
                    let from = node_indices[dep_idx_raw];
                    let to = node_indices[i];
                    if dag.add_edge(from, to, EdgeType::Dependency).is_err() {
                        return format!(
                            "Error: dependency from step {} to step {} would create a cycle",
                            dep_idx_raw, i
                        );
                    }
                }
            }
        }

        let first_goal = steps_arr[0]
            .get("goal")
            .and_then(|g| g.as_str())
            .unwrap_or("(unnamed)")
            .to_string();

        let n = dag.node_count();

        // Install the new plan into the engine.
        let new_engine = ReasoningEngine::new_with_plan(dag, step_budget);
        let mut engine = self.engine.lock();
        *engine = new_engine;

        format!(
            "Plan created with {} steps. Starting step 1: {}",
            n, first_goal
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_engine() -> SharedEngine {
        Arc::new(Mutex::new(ReasoningEngine::new()))
    }

    // --- CheckpointTool ---

    #[tokio::test]
    async fn test_checkpoint_tool_saves_state() {
        let engine = make_engine();

        // Pre-load messages into the engine's current_messages field.
        {
            let mut e = engine.lock();
            e.sync_messages(&[json!({"role": "user", "content": "hello"})]);
        }

        let tool = CheckpointTool::new(Arc::clone(&engine));
        let mut params = HashMap::new();
        params.insert(
            "label".to_string(),
            serde_json::Value::String("before_tool_call".to_string()),
        );

        let result = tool.execute(params).await;
        assert!(
            result.contains("before_tool_call"),
            "result should mention label, got: {result}"
        );
        assert!(
            result.contains("1 checkpoints"),
            "result should report count, got: {result}"
        );

        // Verify the engine actually has the checkpoint with the right messages.
        let mut engine_guard = engine.lock();
        let cp = engine_guard.pop_checkpoint().unwrap();
        assert_eq!(cp.label, "before_tool_call");
        assert_eq!(cp.messages.len(), 1);
    }

    #[tokio::test]
    async fn test_checkpoint_tool_auto_label() {
        let engine = make_engine();
        let tool = CheckpointTool::new(Arc::clone(&engine));

        // Execute without a label parameter.
        let result = tool.execute(HashMap::new()).await;
        assert!(
            result.contains("cp-"),
            "auto-label should start with 'cp-', got: {result}"
        );
        assert!(
            result.contains("1 checkpoints"),
            "should report 1 checkpoint, got: {result}"
        );
    }

    #[tokio::test]
    async fn test_checkpoint_tool_increments_count() {
        let engine = make_engine();
        let tool = CheckpointTool::new(Arc::clone(&engine));

        for i in 0..3u32 {
            let mut params = HashMap::new();
            params.insert(
                "label".to_string(),
                serde_json::Value::String(format!("cp{}", i)),
            );
            tool.execute(params).await;
        }

        let count = engine.lock().checkpoint_count();
        assert_eq!(count, 3);
    }

    // --- BacktrackTool ---

    #[tokio::test]
    async fn test_backtrack_tool_returns_checkpoint() {
        let engine = make_engine();

        // Manually save a checkpoint.
        {
            let msgs = vec![json!({"role": "user", "content": "v1"})];
            let mut e = engine.lock();
            e.save_checkpoint("safe_point", &msgs, 0);
        }

        let tool = BacktrackTool::new(Arc::clone(&engine));
        let mut params = HashMap::new();
        params.insert(
            "reason".to_string(),
            serde_json::Value::String("tool failed".to_string()),
        );

        let result = tool.execute(params).await;
        // Should be a simple status message (not JSON).
        assert!(
            result.contains("Backtracking"),
            "should confirm backtracking, got: {result}"
        );
        assert!(
            result.contains("safe_point"),
            "should contain checkpoint label, got: {result}"
        );
        assert!(
            result.contains("tool failed"),
            "should echo the reason, got: {result}"
        );

        // Verify the pending restore was set on the engine.
        let mut e = engine.lock();
        let restored = e.take_pending_restore().unwrap();
        assert_eq!(restored.len(), 1);
        assert_eq!(restored[0]["content"], "v1");
    }

    #[tokio::test]
    async fn test_backtrack_tool_named_label() {
        let engine = make_engine();

        // Save two checkpoints.
        {
            let mut e = engine.lock();
            e.save_checkpoint("first", &[json!({"role":"user","content":"first"})], 0);
            e.save_checkpoint("second", &[json!({"role":"user","content":"second"})], 1);
        }

        let tool = BacktrackTool::new(Arc::clone(&engine));
        let mut params = HashMap::new();
        params.insert(
            "reason".to_string(),
            serde_json::Value::String("retry".to_string()),
        );
        params.insert(
            "label".to_string(),
            serde_json::Value::String("first".to_string()),
        );

        let result = tool.execute(params).await;
        assert!(
            result.contains("first"),
            "should mention 'first' checkpoint, got: {result}"
        );
        // Pending restore should have the first checkpoint messages.
        let mut e = engine.lock();
        let restored = e.take_pending_restore().unwrap();
        assert_eq!(restored[0]["content"], "first");
    }

    #[tokio::test]
    async fn test_backtrack_no_checkpoints_returns_error() {
        let engine = make_engine();
        let tool = BacktrackTool::new(Arc::clone(&engine));

        let mut params = HashMap::new();
        params.insert(
            "reason".to_string(),
            serde_json::Value::String("failed".to_string()),
        );

        let result = tool.execute(params).await;
        assert!(
            result.starts_with("Error:"),
            "should return error when no checkpoints, got: {result}"
        );
        assert!(
            result.contains("No checkpoint available"),
            "should mention no checkpoints, got: {result}"
        );
    }

    // --- PlanTool ---

    #[tokio::test]
    async fn test_plan_tool_creates_dag() {
        let engine = make_engine();
        let tool = PlanTool::new(Arc::clone(&engine));

        let steps = json!([
            {"goal": "Research the topic"},
            {"goal": "Write the report"},
            {"goal": "Review and edit"}
        ]);
        let mut params = HashMap::new();
        params.insert("steps".to_string(), steps);

        let result = tool.execute(params).await;
        assert!(
            result.contains("Plan created with 3 steps"),
            "should report 3 steps, got: {result}"
        );
        assert!(
            result.contains("Research the topic"),
            "should mention first step goal, got: {result}"
        );

        // Engine should now be in plan-guided mode.
        let engine_guard = engine.lock();
        let step = engine_guard.current_step().unwrap();
        assert_eq!(step.goal, "Research the topic");
    }

    #[tokio::test]
    async fn test_plan_tool_with_dependencies() {
        let engine = make_engine();
        let tool = PlanTool::new(Arc::clone(&engine));

        // Step 0: fetch data
        // Step 1: process data (depends on step 0)
        // Step 2: store result (depends on step 1)
        let steps = json!([
            {"goal": "Fetch data"},
            {"goal": "Process data", "depends_on": [0]},
            {"goal": "Store result", "depends_on": [1]}
        ]);
        let mut params = HashMap::new();
        params.insert("steps".to_string(), steps);

        let result = tool.execute(params).await;
        assert!(
            result.contains("Plan created with 3 steps"),
            "should create 3 steps, got: {result}"
        );

        // Verify the plan advances in order.
        let mut engine_guard = engine.lock();
        assert_eq!(engine_guard.current_step().unwrap().goal, "Fetch data");
        engine_guard.mark_current_completed(None);
        let next = engine_guard.advance().unwrap();
        assert_eq!(next.goal, "Process data");
    }

    #[tokio::test]
    async fn test_plan_tool_missing_steps_param() {
        let engine = make_engine();
        let tool = PlanTool::new(Arc::clone(&engine));

        let result = tool.execute(HashMap::new()).await;
        assert!(
            result.starts_with("Error:"),
            "should error on missing steps, got: {result}"
        );
    }

    #[tokio::test]
    async fn test_plan_tool_empty_steps() {
        let engine = make_engine();
        let tool = PlanTool::new(Arc::clone(&engine));

        let mut params = HashMap::new();
        params.insert("steps".to_string(), json!([]));

        let result = tool.execute(params).await;
        assert!(
            result.starts_with("Error:"),
            "should error on empty steps, got: {result}"
        );
    }

    #[tokio::test]
    async fn test_plan_tool_out_of_range_dependency() {
        let engine = make_engine();
        let tool = PlanTool::new(Arc::clone(&engine));

        let steps = json!([
            {"goal": "Step A"},
            {"goal": "Step B", "depends_on": [99]}
        ]);
        let mut params = HashMap::new();
        params.insert("steps".to_string(), steps);

        let result = tool.execute(params).await;
        assert!(
            result.starts_with("Error:"),
            "should error on out-of-range dep, got: {result}"
        );
        assert!(
            result.contains("out of range"),
            "should mention out of range, got: {result}"
        );
    }
}
