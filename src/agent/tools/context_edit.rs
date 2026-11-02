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

/// Canonical call shape echoed in the "note required" error so the model
/// sees exactly what to send instead of guessing.
const CANONICAL_CALL_SHAPE: &str = "context({\"drop\":\"12-56\",\"note\":\"...\"})";

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
        "Drop settled tool-call/result blocks by [seq] and leave a checkpoint note; see system \
         guidance."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "drop": {
                    // No "type": accepts "all", a range string ("12-56",
                    // "12-20,30-40"), or an array of seq integers.
                    "description": "\"all\", a seq range like \"12-56\", or an array of seq integers. Omit for a note-only edit."
                },
                "keep": {
                    "description": "Seq ids exempted from drop. Optional."
                },
                "note": {
                    "type": "string",
                    "description": "Checkpoint note, kept verbatim. Required whenever drop is set."
                }
            }
        })
    }

    /// Repair/validate before `execute`: a `drop` without a `note` is
    /// rejected here (not in `execute`) so every call path — direct or via
    /// the dispatcher envelope — gets the same canonical-shape error before
    /// any edit is queued.
    fn normalize_params(
        &self,
        _called_as: &str,
        params: &mut HashMap<String, Value>,
    ) -> Result<(), String> {
        let has_drop = parse_drop(params.get("drop")).is_some();
        let has_note = params
            .get("note")
            .and_then(Value::as_str)
            .is_some_and(|s| !s.trim().is_empty());
        if has_drop && !has_note {
            return Err(format!(
                "Tool 'context' requires 'note' whenever 'drop' is present. Call as {CANONICAL_CALL_SHAPE}."
            ));
        }
        Ok(())
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
                message: format!(
                    "requires at least one of 'drop' or 'note'. Call as {CANONICAL_CALL_SHAPE} \
                     or context({{\"note\":\"...\"}})."
                ),
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

    /// `normalize_params` — the hook every registry-routed call goes through
    /// before `execute` — rejects a `drop` with no `note` and gives the
    /// canonical call shape; a drop with a note, or a note alone, pass.
    #[test]
    fn normalize_params_requires_note_whenever_drop_is_present() {
        let (tool, _engine) = make_tool();

        // drop + no note -> rejected, canonical shape in the message.
        let mut params = HashMap::new();
        params.insert("drop".to_string(), json!("12-56"));
        let err = tool
            .normalize_params("context", &mut params)
            .expect_err("drop without note must be rejected");
        assert!(err.contains("note"));
        assert!(err.contains(CANONICAL_CALL_SHAPE));

        // drop + blank note -> still rejected (whitespace-only is not a note).
        let mut params = HashMap::new();
        params.insert("drop".to_string(), json!("12-56"));
        params.insert("note".to_string(), json!("   "));
        assert!(tool.normalize_params("context", &mut params).is_err());

        // drop + note -> accepted.
        let mut params = HashMap::new();
        params.insert("drop".to_string(), json!("12-56"));
        params.insert("note".to_string(), json!("settled the fix"));
        assert!(tool.normalize_params("context", &mut params).is_ok());

        // note only, no drop -> accepted (note-only edit).
        let mut params = HashMap::new();
        params.insert("note".to_string(), json!("checkpoint"));
        assert!(tool.normalize_params("context", &mut params).is_ok());

        // neither -> accepted here; `execute` is the layer that rejects an
        // empty call entirely (see `execute_queues_context_edit_from_params`).
        assert!(tool
            .normalize_params("context", &mut HashMap::new())
            .is_ok());
    }
}
