//! Lossless retrieval of digested tool-result outputs.
//!
//! Large tool results are reduced to a head+tail preview at ingestion
//! (`digest_tool_result` in `tool_engine.rs`) and the full body is stashed in
//! the per-agent `tool_result_store` keyed by `tool_call_id`. This tool lets
//! the model recover the full output (the truncated middle) on demand — one
//! tool call, no re-execution of the original tool.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Value};

use crate::agent::tools::base::Tool;

/// Retrieve the full, untruncated output of a prior tool call by its
/// `tool_call_id` (the id shown in a `[truncated: ...]` preview block).
pub struct RecallToolResultTool {
    store: Arc<parking_lot::Mutex<HashMap<String, String>>>,
}

impl RecallToolResultTool {
    pub fn new(store: Arc<parking_lot::Mutex<HashMap<String, String>>>) -> Self {
        Self { store }
    }
}

#[async_trait]
impl Tool for RecallToolResultTool {
    fn name(&self) -> &str {
        "recall_tool_result"
    }

    fn description(&self) -> &str {
        "Retrieve the FULL output of a prior tool call that was truncated in \
         context (shown as `[truncated: ... call recall_tool_result(...)]`). \
         Pass the `tool_call_id` from that preview block. Returns the complete \
         verbatim output. Use it when you need the middle of a truncated result \
         rather than re-running the tool."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "tool_call_id": {
                    "type": "string",
                    "description": "The tool_call_id from the [truncated: ...] preview block."
                }
            },
            "required": ["tool_call_id"]
        })
    }

    async fn execute(&self, params: HashMap<String, Value>) -> String {
        let id = match params.get("tool_call_id").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => {
                return "Error: tool_call_id is required. Pass the id from the \
                        [truncated: ...] preview block."
                    .to_string();
            }
        };
        match self.store.lock().get(id) {
            Some(full) => full.clone(),
            None => format!(
                "No stored output for tool_call_id='{id}'. It may be from a \
                 previous session (full outputs are in-memory only) or already \
                 small enough that it was never truncated. Re-run the original \
                 tool if you need fresh data."
            ),
        }
    }
}
