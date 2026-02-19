//! Canonical tool plan schema and validation.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::agent::role_policy::RouterDecision;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolPlanAction {
    Tool,
    Subagent,
    Specialist,
    AskUser,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolPlan {
    pub action: ToolPlanAction,
    pub target: String,
    #[serde(default)]
    pub args: Value,
    pub confidence: f64,
    #[serde(default)]
    pub idempotency_key: String,
}

impl ToolPlan {
    pub fn validate(&self) -> Result<(), String> {
        if self.target.trim().is_empty() {
            return Err("tool plan target cannot be empty".to_string());
        }
        if !(0.0..=1.0).contains(&self.confidence) {
            return Err(format!(
                "tool plan confidence must be in [0,1], got {}",
                self.confidence
            ));
        }
        Ok(())
    }
}

fn normalize_action(raw: &str) -> Option<ToolPlanAction> {
    let s = raw.to_lowercase();
    match s.as_str() {
        "tool" => Some(ToolPlanAction::Tool),
        "subagent" => Some(ToolPlanAction::Subagent),
        "specialist" => Some(ToolPlanAction::Specialist),
        "ask_user" | "askuser" => Some(ToolPlanAction::AskUser),
        _ => {
            // Local SLMs sometimes output enum-lists like "tool|subagent|..."
            if s.contains("ask_user") {
                Some(ToolPlanAction::AskUser)
            } else if s.contains("specialist") {
                Some(ToolPlanAction::Specialist)
            } else if s.contains("subagent") {
                Some(ToolPlanAction::Subagent)
            } else if s.contains("tool") {
                Some(ToolPlanAction::Tool)
            } else {
                None
            }
        }
    }
}

pub fn from_router_decision(decision: RouterDecision) -> Result<ToolPlan, String> {
    let action = normalize_action(&decision.action)
        .ok_or_else(|| format!("invalid tool plan action: {}", decision.action))?;
    let idempotency_key = format!(
        "{}:{}:{}",
        match action {
            ToolPlanAction::Tool => "tool",
            ToolPlanAction::Subagent => "subagent",
            ToolPlanAction::Specialist => "specialist",
            ToolPlanAction::AskUser => "ask_user",
        },
        decision.target,
        serde_json::to_string(&decision.args).unwrap_or_default()
    );
    let plan = ToolPlan {
        action,
        target: decision.target,
        args: decision.args,
        confidence: decision.confidence,
        idempotency_key,
    };
    plan.validate()?;
    Ok(plan)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_router_decision_normalizes_enum_list_action() {
        let rd = RouterDecision {
            action: "tool|subagent|specialist|ask_user".to_string(),
            target: "read_file".to_string(),
            args: serde_json::json!({"path":"README.md"}),
            confidence: 0.7,
        };
        let p = from_router_decision(rd).expect("normalized");
        assert_eq!(p.action, ToolPlanAction::AskUser);
    }
}

