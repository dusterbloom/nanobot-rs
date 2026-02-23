//! Hard role-separation policies for local trio operation.
//!
//! This module contains pure helpers for:
//! - main-model tool-use bans
//! - strict router decision parsing
//! - role-scoped context pack shaping

use serde::{Deserialize, Serialize};
use serde_json::Value;

fn default_true_specialist() -> bool {
    true
}

/// Structured response returned by the specialist lane.
///
/// The LLM only produces `result` and `success`.
/// `vote_key` and `parsed_json` are computed post-hoc — never by the model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SpecialistResponse {
    pub result: String,
    #[serde(default = "default_true_specialist")]
    pub success: bool,
    /// Deterministic vote key for MAKER comparison — computed from `result`, never LLM-generated.
    #[serde(skip_deserializing)]
    pub vote_key: String,
    /// True if the raw output was valid JSON; false if we fell back to plain text.
    #[serde(skip_deserializing)]
    pub parsed_json: bool,
}

pub(crate) fn normalize_vote_key(s: &str) -> String {
    s.to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Build the specialist system prompt, optionally requiring strict JSON output.
pub(crate) fn build_specialist_system_prompt(schema_enabled: bool) -> String {
    if schema_enabled {
        r#"ROLE=SPECIALIST
You MUST respond ONLY with a JSON object:
{
  "result": "<your complete answer>",
  "success": true
}
No text outside the JSON object."#
            .to_string()
    } else {
        "ROLE=SPECIALIST\nReturn concise actionable output only. No markdown unless requested."
            .to_string()
    }
}

/// Parse raw specialist output into a `SpecialistResponse`.
///
/// Tries to extract and deserialize a JSON object from the raw text.
/// Falls back to a plain-text result when JSON is absent or malformed.
pub(crate) fn parse_specialist_response(raw: &str, _target: &str) -> SpecialistResponse {
    if let Some(json_str) = crate::agent::router::extract_json_object(raw) {
        if let Ok(mut sr) = serde_json::from_str::<SpecialistResponse>(&json_str) {
            sr.vote_key = normalize_vote_key(&sr.result);
            sr.parsed_json = true;
            return sr;
        }
    }
    SpecialistResponse {
        result: raw.to_string(),
        vote_key: normalize_vote_key(raw),
        success: true,
        parsed_json: false,
    }
}

/// High-level role in the trio.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    Main,
    Router,
    Specialist,
}

/// Strict router output schema.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct RouterDecision {
    pub action: String, // tool | subagent | specialist | ask_user
    pub target: String,
    #[serde(default)]
    pub args: Value,
    pub confidence: f64,
}

/// Decide whether direct tool use by the main model should be blocked.
pub fn should_block_main_tool_calls(strict_no_tools_main: bool, has_tool_calls: bool) -> bool {
    strict_no_tools_main && has_tool_calls
}

/// Parse and validate strict router decision JSON.
pub fn parse_router_decision_strict(raw: &str) -> Result<RouterDecision, String> {
    let parsed: RouterDecision =
        serde_json::from_str(raw).map_err(|e| format!("invalid router JSON: {}", e))?;
    let action_ok = matches!(
        parsed.action.as_str(),
        "tool" | "subagent" | "specialist" | "ask_user" | "respond" | "pipeline"
    );
    if !action_ok {
        return Err(format!("invalid router action: {}", parsed.action));
    }
    if parsed.target.trim().is_empty() && parsed.action != "respond" {
        return Err("router target cannot be empty".to_string());
    }
    if !(0.0..=1.0).contains(&parsed.confidence) {
        return Err(format!(
            "router confidence must be in [0,1], got {}",
            parsed.confidence
        ));
    }
    Ok(parsed)
}

/// Build role-scoped context pack text from shared inputs.
pub fn build_context_pack(
    role: Role,
    user_intent: &str,
    conversation_summary: &str,
    task_state: &str,
    available_tools: &[String],
    max_chars: usize,
) -> String {
    let body = match role {
        Role::Main => format!(
            "Role: main\nUser intent:\n{}\n\nConversation summary:\n{}\n\nTask state:\n{}\n",
            user_intent, conversation_summary, task_state
        ),
        Role::Router => format!(
            "Role: router\nTask state:\n{}\n\nAllowed actions:\n- respond (simple conversation, greetings, direct answers)\n- tool (use a specific tool)\n- specialist (delegate complex reasoning)\n- ask_user (request clarification)\n\nAvailable tools:\n{}\n",
            task_state,
            if available_tools.is_empty() {
                "(none)".to_string()
            } else {
                available_tools.join(", ")
            }
        ),
        Role::Specialist => format!(
            "Role: specialist\nTask state:\n{}\n\nFocused objective:\n{}\n",
            task_state, user_intent
        ),
    };
    if body.len() <= max_chars {
        body
    } else {
        let end = crate::utils::helpers::floor_char_boundary(&body, max_chars);
        body[..end].to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // T1 — struct deserializes the two LLM-produced fields
    #[test]
    fn test_specialist_response_deserializes_full_envelope() {
        let json = r#"{"result":"42","success":true}"#;
        let sr: SpecialistResponse = serde_json::from_str(json).unwrap();
        assert_eq!(sr.result, "42");
        assert!(sr.success);
        // vote_key and parsed_json are skip_deserializing — always default
        assert_eq!(sr.vote_key, "");
        assert!(!sr.parsed_json);
    }

    // T2 — missing success defaults to true
    #[test]
    fn test_specialist_response_missing_success_defaults_true() {
        let json = r#"{"result":"ok"}"#;
        let sr: SpecialistResponse = serde_json::from_str(json).unwrap();
        assert!(sr.success);
    }

    // T3 — raw text fallback produces non-empty result
    #[test]
    fn test_parse_specialist_response_raw_text_fallback() {
        let sr = parse_specialist_response("The answer is 42", "math");
        assert_eq!(sr.result, "The answer is 42");
        assert!(sr.success);
        assert!(!sr.parsed_json);
    }

    // T4 — JSON buried in prose is extracted
    #[test]
    fn test_parse_specialist_response_json_in_prose() {
        let raw = r#"Here is my answer: {"result":"42","success":true}"#;
        let sr = parse_specialist_response(raw, "math");
        assert_eq!(sr.result, "42");
        assert!(sr.parsed_json);
    }

    // T5 — vote_key is computed post-hoc, not from LLM
    #[test]
    fn test_parse_specialist_response_vote_key_computed_post_hoc() {
        let raw = r#"{"result":"The Answer Is 42","success":true}"#;
        let sr = parse_specialist_response(raw, "math");
        assert_eq!(sr.vote_key, "the answer is 42");
        assert!(sr.parsed_json);
    }

    // T6 — schema=false produces legacy system prompt (no JSON instruction)
    #[test]
    fn test_build_specialist_system_prompt_schema_disabled() {
        let prompt = build_specialist_system_prompt(false);
        assert!(prompt.contains("ROLE=SPECIALIST"));
        assert!(!prompt.contains("result"));
    }

    // T7 — schema=true produces lean JSON schema (result + success only)
    #[test]
    fn test_build_specialist_system_prompt_schema_enabled() {
        let prompt = build_specialist_system_prompt(true);
        assert!(prompt.contains("result"));
        assert!(prompt.contains("success"));
        // phantom fields must NOT appear
        assert!(!prompt.contains("tools_used"));
        assert!(!prompt.contains("steps"));
        assert!(!prompt.contains("vote_key"));
    }

    // T8 — two semantically equivalent results produce same vote_key
    #[test]
    fn test_vote_key_normalization_strips_whitespace() {
        let key1 = normalize_vote_key("The answer is 42");
        let key2 = normalize_vote_key("the  answer  is  42");
        assert_eq!(key1, key2);
    }

    // T9 — fallback vote_key is populated from result
    #[test]
    fn test_parse_specialist_response_fallback_vote_key_nonempty() {
        let sr = parse_specialist_response("The answer is 42", "math");
        assert!(!sr.vote_key.is_empty());
        assert_eq!(sr.vote_key, "the answer is 42");
    }

    // T10 — parsed_json telemetry: false on fallback, true on valid JSON
    #[test]
    fn test_parsed_json_telemetry() {
        let fallback = parse_specialist_response("plain text", "t");
        assert!(!fallback.parsed_json);

        let json_input = r#"{"result":"ok","success":true}"#;
        let parsed = parse_specialist_response(json_input, "t");
        assert!(parsed.parsed_json);
    }

    #[test]
    fn test_should_block_main_tool_calls() {
        assert!(should_block_main_tool_calls(true, true));
        assert!(!should_block_main_tool_calls(false, true));
        assert!(!should_block_main_tool_calls(true, false));
    }

    #[test]
    fn test_pipeline_action_validates() {
        let json_str = r#"{"action":"pipeline","target":"multi_fetch","args":{"steps":[{"instruction":"fetch weather"}]},"confidence":0.9}"#;
        let parsed: RouterDecision = serde_json::from_str(json_str).unwrap();
        assert_eq!(parsed.action, "pipeline");
    }

    #[test]
    fn test_parse_router_decision_strict_accepts_valid() {
        let raw = r#"{"action":"tool","target":"read_file","args":{"path":"README.md"},"confidence":0.92}"#;
        let d = parse_router_decision_strict(raw).expect("valid router decision");
        assert_eq!(d.action, "tool");
        assert_eq!(d.target, "read_file");
        assert_eq!(d.args["path"], "README.md");
        assert_eq!(d.confidence, 0.92);
    }

    #[test]
    fn test_parse_router_decision_strict_accepts_respond() {
        let raw = r#"{"action":"respond","target":"main","args":{},"confidence":0.95}"#;
        let d = parse_router_decision_strict(raw).expect("respond should be valid");
        assert_eq!(d.action, "respond");
    }

    #[test]
    fn test_parse_router_decision_strict_respond_allows_empty_target() {
        let raw = r#"{"action":"respond","target":"","args":{},"confidence":0.9}"#;
        parse_router_decision_strict(raw).expect("respond with empty target should be valid");
    }

    #[test]
    fn test_parse_router_decision_strict_rejects_invalid_action() {
        let raw = r#"{"action":"chat","target":"x","args":{},"confidence":0.5}"#;
        let err = parse_router_decision_strict(raw).unwrap_err();
        assert!(err.contains("invalid router action"));
    }

    #[test]
    fn test_parse_router_decision_strict_rejects_non_json() {
        let err = parse_router_decision_strict("not json").unwrap_err();
        assert!(err.contains("invalid router JSON"));
    }

    #[test]
    fn test_build_context_pack_is_role_scoped() {
        let tools = vec!["exec".to_string(), "read_file".to_string()];
        let main = build_context_pack(
            Role::Main,
            "solve user request",
            "short convo summary",
            "task-state",
            &tools,
            10_000,
        );
        assert!(main.contains("Role: main"));
        assert!(!main.contains("Available tools:"));

        let router = build_context_pack(
            Role::Router,
            "solve user request",
            "short convo summary",
            "task-state",
            &tools,
            10_000,
        );
        assert!(router.contains("Role: router"));
        assert!(router.contains("Available tools:"));
    }
}
