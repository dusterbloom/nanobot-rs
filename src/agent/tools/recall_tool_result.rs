//! Lossless retrieval of digested tool-result outputs.
//!
//! Large tool results are reduced to a head+tail preview at ingestion
//! (`digest_tool_result` in `tool_engine.rs`) and the full body is stashed in
//! SQLite keyed by concrete session id + `tool_call_id`. This tool lets the
//! model recover the full output (the truncated middle) on demand — one tool
//! call, no re-execution, including after a process restart.

use std::collections::HashMap;
use std::path::PathBuf;

use async_trait::async_trait;
use serde_json::{json, Value};

use crate::agent::tools::base::Tool;

/// Retrieve the full, untruncated output of a prior tool call by its
/// `tool_call_id` (the id shown in a `[truncated: ...]` preview block).
pub struct RecallToolResultTool {
    db_path: PathBuf,
    session_id: String,
}

impl RecallToolResultTool {
    pub fn with_db(db_path: PathBuf, session_id: String) -> Self {
        Self {
            db_path,
            session_id,
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
        let db = crate::session::db::SessionDb::new(&self.db_path);
        if let Some(full) = db.load_tool_result(&self.session_id, id).await {
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
        assert!(matches!(
            db.store_tool_result_immutable(&session.id, "call_restart", "exec", "durable body")
                .await,
            crate::session::db::StoredResult::Stored { .. }
        ));

        let tool = RecallToolResultTool::with_db(db_path, session.id);
        let result = tool
            .execute(HashMap::from([(
                "tool_call_id".to_string(),
                Value::String("call_restart".to_string()),
            )]))
            .await;

        assert_eq!(result, "durable body");
    }

    #[tokio::test]
    async fn predictable_call_ids_are_isolated_by_session() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("sessions.db");
        let db = crate::session::db::SessionDb::new(&db_path);
        let first = db.create_session("cli:first").await;
        let second = db.create_session("cli:second").await;
        assert!(matches!(
            db.store_tool_result_immutable(&first.id, "call_1", "exec", "first body")
                .await,
            crate::session::db::StoredResult::Stored { .. }
        ));
        assert!(matches!(
            db.store_tool_result_immutable(&second.id, "call_1", "exec", "second body")
                .await,
            crate::session::db::StoredResult::Stored { .. }
        ));

        let params = HashMap::from([(
            "tool_call_id".to_string(),
            Value::String("call_1".to_string()),
        )]);
        let first_tool = RecallToolResultTool::with_db(db_path.clone(), first.id);
        let second_tool = RecallToolResultTool::with_db(db_path, second.id);

        assert_eq!(first_tool.execute(params.clone()).await, "first body");
        assert_eq!(second_tool.execute(params).await, "second body");
    }
}
