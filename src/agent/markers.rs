//! Shared marker strings used across agent routing/tool pipelines.

pub const TOOL_RUNNER_SUMMARY_PREFIX: &str = "[tool runner summary]";
pub const TOOL_RUNNER_OUTPUT_PREFIX: &str = "[tool runner output]";
pub const TOOL_ANALYSIS_SUMMARY_PREFIX: &str = "[Tool analysis summary]";
pub const TOOL_ANALYSIS_FULL_OUTPUT_MARKER: &str = "\n\n[Full output:";

/// Build an ephemeral turn-scaffolding user message.
///
/// These are injected continuation/grounding/boundary nudges — NOT real user
/// input. Marking them `_synthetic` means they (a) persist with `synthetic=1`
/// (see `session::db::insert_message_locked`), (b) are stripped from the next
/// turn's reloaded wire history by `session::filters::filter_history` (Stage 5),
/// and (c) are not counted as conversational turns by that filter's turn limit
/// (Stage 4). Together this keeps the rendered prompt prefix byte-stable across
/// turns so the server-side prefix cache survives instead of cold re-prefilling.
pub fn scaffold_user(content: impl Into<String>) -> serde_json::Value {
    serde_json::json!({
        "role": "user",
        "content": content.into(),
        "_synthetic": true,
    })
}
