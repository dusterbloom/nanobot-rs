//! Tool-call helpers shared by the agent loop, subagents and pipelines.

use std::collections::HashMap;

use serde_json::Value;

use crate::agent::context::ContextBuilder;
use crate::agent::tools::ToolRegistry;

/// Normalize a tool call key for dedup: sort JSON keys and use compact serialization.
pub(crate) fn normalize_call_key(name: &str, arguments: &HashMap<String, Value>) -> String {
    let mut sorted: Vec<_> = arguments.iter().collect();
    sorted.sort_by_key(|(k, _)| *k);
    let normalized = serde_json::to_string(&sorted).unwrap_or_default();
    format!("{}:{}", name, normalized)
}

/// Process tool calls from an LLM response: build assistant message,
/// execute each tool, and add results to messages.
///
/// This is the common core shared by subagent and pipeline.
/// Returns `true` if tool calls were processed, `false` if none were present.
///
/// **Protocol compliance** is handled by the caller: before the next LLM call,
/// pass `messages` through `protocol::render_to_wire()` to get the correct wire
/// format. Do NOT call `repair_for_local` here — rendering at call time is the
/// correct approach.
pub async fn process_tool_response(
    response: &crate::providers::base::LLMResponse,
    messages: &mut Vec<Value>,
    tools: &ToolRegistry,
) -> bool {
    if !response.has_tool_calls() {
        return false;
    }

    let tc_json: Vec<Value> = response
        .tool_calls
        .iter()
        .map(|tc| tc.to_openai_json())
        .collect();

    ContextBuilder::add_assistant_message(messages, response.content.as_deref(), Some(&tc_json));

    for tc in &response.tool_calls {
        let result = tools.execute(&tc.name, tc.arguments.clone()).await;
        ContextBuilder::add_tool_result(messages, &tc.id, &tc.name, result.data());
    }

    true
}
