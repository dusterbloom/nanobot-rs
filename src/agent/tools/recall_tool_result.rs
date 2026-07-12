//! Lossless retrieval of digested tool-result outputs.
//!
//! Large tool results are reduced to a head+tail preview at ingestion
//! (`digest_tool_result` in `tool_engine.rs`) and the full body is stashed in
//! the per-agent `tool_result_store` and SQLite keyed by session +
//! `tool_call_id`. This tool lets the model recover the full output (the
//! truncated middle) on demand — one tool call, no re-execution, including
//! after a process restart.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Value};

use crate::agent::tools::base::Tool;

/// Retrieve the full, untruncated output of a prior tool call by its
/// `tool_call_id` (the id shown in a `[truncated: ...]` preview block).
pub struct RecallToolResultTool {
    store: Arc<parking_lot::Mutex<HashMap<String, String>>>,
    durable: (PathBuf, String),
}

impl RecallToolResultTool {
    pub fn with_db(
        store: Arc<parking_lot::Mutex<HashMap<String, String>>>,
        db_path: PathBuf,
        session_id: String,
    ) -> Self {
        Self {
            store,
            durable: (db_path, session_id),
        }
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
        if let Some(full) = self.store.lock().get(id).cloned() {
            return full;
        }
        let (db_path, session_id) = &self.durable;
        let db = crate::session::db::SessionDb::new(db_path);
        if let Some(full) = db.load_tool_result(session_id, id).await {
            self.store.lock().insert(id.to_string(), full.clone());
            return full;
        }
        format!(
            "No stored output for tool_call_id='{id}' in this session. It may \
             have been small enough that it was never stashed, or it may have \
             been removed. Re-run the original tool if you need fresh data."
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn recalls_from_sqlite_after_live_cache_is_empty() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("sessions.db");
        let db = crate::session::db::SessionDb::new(&db_path);
        let session = db.create_session("cli:restart-recall").await;
        assert!(
            db.store_tool_result(&session.id, "call_restart", "exec", "durable body")
                .await
        );

        let tool = RecallToolResultTool::with_db(
            Arc::new(parking_lot::Mutex::new(HashMap::new())),
            db_path,
            session.id,
        );
        let result = tool
            .execute(HashMap::from([(
                "tool_call_id".to_string(),
                Value::String("call_restart".to_string()),
            )]))
            .await;

        assert_eq!(result, "durable body");
    }
}
