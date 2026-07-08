//! Shared marker strings used across agent routing/tool pipelines.

pub const TOOL_RUNNER_SUMMARY_PREFIX: &str = "[tool runner summary]";
pub const TOOL_RUNNER_OUTPUT_PREFIX: &str = "[tool runner output]";
pub const TOOL_ANALYSIS_SUMMARY_PREFIX: &str = "[Tool analysis summary]";
pub const TOOL_ANALYSIS_FULL_OUTPUT_MARKER: &str = "\n\n[Full output:";

/// Build a turn-scaffolding user message.
///
/// These are injected continuation/grounding/boundary nudges, not real user
/// input. They are synthetic so turn-limit logic does not count them as user
/// turns, but they are also cache-replayable: once a scaffold is sent to the
/// model, the next reloaded turn must replay it byte-for-byte or the local
/// prefix cache diverges at the removed message.
pub fn scaffold_user(content: impl Into<String>) -> serde_json::Value {
    serde_json::json!({
        "role": "user",
        "content": content.into(),
        "_synthetic": true,
        "_cache_replay": true,
    })
}

/// True if a message is a synthetic scaffold (injected nudge, not real input).
pub fn is_synthetic(msg: &serde_json::Value) -> bool {
    msg.get("_synthetic").and_then(|v| v.as_bool()).unwrap_or(false)
}
