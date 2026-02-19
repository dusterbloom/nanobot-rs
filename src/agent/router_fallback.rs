//! Deterministic fallback router used when SLM routing is invalid.

use serde_json::json;

use crate::agent::policy::SessionPolicy;
use crate::agent::toolplan::{ToolPlan, ToolPlanAction};

/// Build a deterministic fallback tool plan from user text and available tools.
pub fn route(user_text: &str, available_tools: &[String], policy: &SessionPolicy) -> ToolPlan {
    let lower = user_text.to_lowercase();
    let has_tool = |name: &str| available_tools.iter().any(|t| t == name);

    if (lower.contains("http://") || lower.contains("https://") || lower.contains("hacker news"))
        && has_tool("web_fetch")
    {
        let url = if lower.contains("hacker news") || lower.contains("hackernews") {
            "https://news.ycombinator.com/".to_string()
        } else {
            user_text
                .split_whitespace()
                .find(|w| w.starts_with("http://") || w.starts_with("https://"))
                .unwrap_or("https://example.com")
                .to_string()
        };
        return ToolPlan {
            action: ToolPlanAction::Tool,
            target: "web_fetch".to_string(),
            args: json!({ "url": url }),
            confidence: 0.4,
            idempotency_key: "fallback:web_fetch".to_string(),
        };
    }

    if lower.contains("latest news")
        && has_tool("spawn")
        && (lower.contains("local") || policy.local_only)
    {
        return ToolPlan {
            action: ToolPlanAction::Subagent,
            target: "researcher".to_string(),
            args: json!({
                "task":"Fetch latest news and summarize key points",
                "model":"local",
            }),
            confidence: 0.4,
            idempotency_key: "fallback:spawn_local_news".to_string(),
        };
    }

    ToolPlan {
        action: ToolPlanAction::AskUser,
        target: "clarify".to_string(),
        args: json!({"question":"Please clarify the exact task and target source."}),
        confidence: 0.2,
        idempotency_key: "fallback:ask_user".to_string(),
    }
}

