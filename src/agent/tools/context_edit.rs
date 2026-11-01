//! `context` tool — model-managed context editing (T4, nodecode port).
//!
//! Lets the model drop settled tool-call/result blocks it will not reread and
//! leave a checkpoint note in its own words. The tool only queues the edit on
//! the [`LcmEngine`]; `apply_pending_context_edit` (agent loop) resolves it
//! against the live message log and rewrites the committed context before the
//! next request goes out (T5). Nothing here touches `active`/`store` directly.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Value};
use tokio::sync::Mutex;

use crate::agent::lcm::{parse_message_ids, ContextEdit, DropSpec, LcmEngine};
use crate::agent::tools::base::{Tool, ToolContext, ToolResult};
use crate::errors::ToolError;

/// Parse the `drop` argument: `"all"` (case-insensitive), or the same lenient
/// id shapes `lcm_expand` accepts (range string, comma list, integer array).
/// `None`/empty input means no drop requested (note-only edit).
fn parse_drop(arg: Option<&Value>) -> Option<DropSpec> {
    let value = arg?;
    if let Some(s) = value.as_str() {
        if s.trim().eq_ignore_ascii_case("all") {
            return Some(DropSpec::All);
        }
    }
    let ids = parse_message_ids(value);
    if ids.is_empty() {
        None
    } else {
        Some(DropSpec::Ids(ids))
    }
}

/// Human-readable echo of a queued edit, used only for the tool's own
/// confirmation line (not re-parsed by anything).
fn describe(edit: &ContextEdit) -> String {
    let drop = match &edit.drop {
        None => "none".to_string(),
        Some(DropSpec::All) => "all".to_string(),
        Some(DropSpec::Ids(ids)) => format!("{ids:?}"),
    };
    let keep = if edit.keep.is_empty() {
        "none".to_string()
    } else {
        format!("{:?}", edit.keep)
    };
    let note = edit.note.as_deref().unwrap_or("none");
    format!(
        "context edit queued: drop {drop} keep {keep} note {note:?}; \
         applied before your next request"
    )
}

/// Tool that queues a model-issued context edit (drop settled blocks, leave a
/// checkpoint note) for the loop to apply before the next request.
pub struct ContextTool {
    engine: Arc<Mutex<LcmEngine>>,
}

impl ContextTool {
    pub fn new(engine: Arc<Mutex<LcmEngine>>) -> Self {
        Self { engine }
    }
}

#[async_trait]
impl Tool for ContextTool {
    fn name(&self) -> &str {
        "context"
    }

    fn description(&self) -> &str {
        "Edit your own context: drop settled tool-call/result blocks you will not reread, and/or \
         leave a checkpoint note in your own words. Every tool result ends with its seq, e.g. \
         [seq 57]. context({\"drop\":\"12-56\",\"note\":\"...\"}) drops those blocks; \
         context({\"drop\":\"all\",\"keep\":\"57\",\"note\":\"...\"}) keeps only seq 57. Dropping a \
         tool call or its result drops the other half of the pair too. The edit is lossless: \
         dropped blocks are never deleted, lcm_expand({\"message_ids\":\"57\"}) reads one back. \
         Requires at least drop or note. Applied before your next request, not immediately."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "drop": {
                    // No "type": accepts "all", a range string ("12-56",
                    // "12-20,30-40"), or an array of seq integers.
                    "description": "What to drop: \"all\", a seq range string like \"12-56\", or an array of seq integers. Omit for a note-only edit."
                },
                "keep": {
                    "description": "Seq ids exempted from drop (same shapes as drop). Optional."
                },
                "note": {
                    "type": "string",
                    "description": "Your own checkpoint, kept verbatim: constraints, decisions, verified state, open work, evidence pointers. Required if drop is omitted."
                }
            }
        })
    }

    async fn execute(&self, params: HashMap<String, Value>, _ctx: &ToolContext) -> ToolResult {
        let drop = parse_drop(params.get("drop"));
        let keep = params.get("keep").map(parse_message_ids).unwrap_or_default();
        let note = params
            .get("note")
            .and_then(Value::as_str)
            .map(str::to_string);

        if drop.is_none() && note.is_none() {
            return Err(ToolError::InvalidArgs {
                message: "requires at least one of 'drop' or 'note'. Call as \
                    context({\"drop\":\"12-56\",\"note\":\"...\"}) or context({\"note\":\"...\"})."
                    .to_string(),
            });
        }

        let edit = ContextEdit { drop, keep, note };
        let summary = describe(&edit);
        self.engine.lock().await.queue_context_edit(edit);
        Ok(summary.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::lcm::LcmConfig;

    fn make_tool() -> (ContextTool, Arc<Mutex<LcmEngine>>) {
        let engine = Arc::new(Mutex::new(LcmEngine::new(LcmConfig::default())));
        (ContextTool::new(engine.clone()), engine)
    }

    #[tokio::test]
    async fn execute_queues_context_edit_from_params() {
        let (tool, engine) = make_tool();

        // drop range + keep + note -> Ids(12..=56), keep [57], note "x".
        let mut params = HashMap::new();
        params.insert("drop".to_string(), json!("12-56"));
        params.insert("keep".to_string(), json!([57]));
        params.insert("note".to_string(), json!("x"));
        assert!(tool
            .execute(params, &ToolContext::sandbox())
            .await
            .is_ok());
        {
            let mut e = engine.lock().await;
            let edit = e.take_context_edit().expect("edit queued");
            assert_eq!(edit.drop, Some(DropSpec::Ids((12..=56).collect())));
            assert_eq!(edit.keep, vec![57]);
            assert_eq!(edit.note.as_deref(), Some("x"));
        }

        // drop "all" -> DropSpec::All.
        let mut params = HashMap::new();
        params.insert("drop".to_string(), json!("all"));
        assert!(tool
            .execute(params, &ToolContext::sandbox())
            .await
            .is_ok());
        {
            let mut e = engine.lock().await;
            let edit = e.take_context_edit().expect("edit queued");
            assert_eq!(edit.drop, Some(DropSpec::All));
            assert!(edit.keep.is_empty());
            assert!(edit.note.is_none());
        }

        // neither drop nor note -> InvalidArgs.
        let result = tool.execute(HashMap::new(), &ToolContext::sandbox()).await;
        assert!(matches!(result, Err(ToolError::InvalidArgs { .. })));
    }
}
