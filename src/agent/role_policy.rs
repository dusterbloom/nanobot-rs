//! Hard role-separation policies for local trio operation.
//!
//! This module contains pure helpers for:
//! - main-model tool-use bans
//! - strict router decision parsing
//! - role-scoped context pack shaping

use serde::{Deserialize, Serialize};
use serde_json::Value;

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
        "tool" | "subagent" | "specialist" | "ask_user"
    );
    if !action_ok {
        return Err(format!("invalid router action: {}", parsed.action));
    }
    if parsed.target.trim().is_empty() {
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
            "Role: router\nTask state:\n{}\n\nAllowed actions:\n- tool\n- subagent\n- specialist\n- ask_user\n\nAvailable tools:\n{}\n",
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

    #[test]
    fn test_should_block_main_tool_calls() {
        assert!(should_block_main_tool_calls(true, true));
        assert!(!should_block_main_tool_calls(false, true));
        assert!(!should_block_main_tool_calls(true, false));
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
