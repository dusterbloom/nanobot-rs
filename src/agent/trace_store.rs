//! Durable trace storage for specialist dispatch results.
//!
//! Captures the full `SpecialistResponse` at the point of creation,
//! before compaction can destroy the in-context specialist messages.

use crate::utils::helpers::{ensure_dir, today_date};
use serde_json::{json, Value};
use std::io::Write;

/// Max chars of user_content stored per trace line.
const MAX_USER_CONTENT: usize = 500;

/// Max chars of outcome summary stored per router decision trace line.
const MAX_OUTCOME: usize = 4000;

/// Full dispatch record including inputs and outputs.
#[derive(Clone)]
pub(crate) struct DispatchRecord {
    pub specialist_name: String,
    pub specialist_model: String,
    pub router_action: String,
    pub router_target: String,
    pub router_confidence: f64,
    pub router_args: serde_json::Value,
    pub user_content: String,
    pub messages_count: usize,
    pub tool_results: Vec<serde_json::Value>,
    pub specialist_response: String,
}

/// Full record of a single router decision (preflight or post-tool).
#[derive(Clone)]
pub(crate) struct RouterDecisionTrace {
    pub phase: String,          // "preflight" or "tool_routing"
    pub action: String,         // respond | tool | specialist | subagent | ask_user
    pub target: String,
    pub confidence: f64,
    pub args: serde_json::Value,
    pub user_content: String,
    pub router_elapsed_ms: u64,
    pub model: String,
    pub outcome: Option<String>, // None for respond/ask_user/specialist; Some for tool/subagent
}

/// Build the trace JSON value from a specialist dispatch (pure, testable).
pub(crate) fn build_trace_entry(record: &DispatchRecord) -> Value {
    let truncated: String = record.user_content.chars().take(MAX_USER_CONTENT).collect();
    json!({
        "type": "specialist_dispatch",
        "ts": chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        "specialist_name": record.specialist_name,
        "specialist_model": record.specialist_model,
        "router_action": record.router_action,
        "router_target": record.router_target,
        "router_confidence": record.router_confidence,
        "router_args": record.router_args,
        "user_content": truncated,
        "messages_count": record.messages_count,
        "tool_results": record.tool_results,
        "specialist_response": record.specialist_response,
    })
}

/// Build the trace JSON value from a router decision (pure, testable).
pub(crate) fn build_router_decision_entry(record: &RouterDecisionTrace) -> Value {
    let truncated_content: String = record.user_content.chars().take(MAX_USER_CONTENT).collect();
    let truncated_outcome: Option<String> = record.outcome.as_ref().map(|o| {
        o.chars().take(MAX_OUTCOME).collect()
    });
    json!({
        "type": "router_decision",
        "ts": chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        "phase": record.phase,
        "action": record.action,
        "target": record.target,
        "confidence": record.confidence,
        "args": record.args,
        "user_content": truncated_content,
        "router_elapsed_ms": record.router_elapsed_ms,
        "model": record.model,
        "outcome": truncated_outcome,
    })
}

/// Append a specialist trace to `~/.nanobot/traces/YYYY-MM-DD.jsonl`.
pub(crate) fn append_specialist_trace(record: &DispatchRecord) {
    let entry = build_trace_entry(record);
    let line = format!("{}\n", entry);

    let base = dirs::home_dir()
        .unwrap_or_default()
        .join(".nanobot")
        .join("traces");
    let dir = ensure_dir(&base);
    let path = dir.join(format!("{}.jsonl", today_date()));

    match std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    {
        Ok(mut f) => {
            if let Err(e) = f.write_all(line.as_bytes()) {
                tracing::warn!("Failed to append specialist trace: {}", e);
            }
        }
        Err(e) => tracing::warn!("Failed to open trace file {}: {}", path.display(), e),
    }
}

/// Append a router decision trace to `~/.nanobot/traces/YYYY-MM-DD.jsonl`.
pub(crate) fn append_router_decision_trace(record: &RouterDecisionTrace) {
    let entry = build_router_decision_entry(record);
    let line = format!("{}\n", entry);

    let base = dirs::home_dir()
        .unwrap_or_default()
        .join(".nanobot")
        .join("traces");
    let dir = ensure_dir(&base);
    let path = dir.join(format!("{}.jsonl", today_date()));

    match std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    {
        Ok(mut f) => {
            if let Err(e) = f.write_all(line.as_bytes()) {
                tracing::warn!("Failed to append router decision trace: {}", e);
            }
        }
        Err(e) => tracing::warn!("Failed to open trace file {}: {}", path.display(), e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_router_decision_entry_all_fields() {
        let record = RouterDecisionTrace {
            phase: "preflight".to_string(),
            action: "specialist".to_string(),
            target: "coding".to_string(),
            confidence: 0.85,
            args: json!({"task": "write code"}),
            user_content: "Write a binary search".to_string(),
            router_elapsed_ms: 340,
            model: "nvidia_orchestrator-8b".to_string(),
            outcome: None,
        };
        let entry = build_router_decision_entry(&record);
        assert_eq!(entry["type"], "router_decision");
        assert_eq!(entry["phase"], "preflight");
        assert_eq!(entry["action"], "specialist");
        assert_eq!(entry["target"], "coding");
        assert_eq!(entry["confidence"], 0.85);
        assert_eq!(entry["args"]["task"], "write code");
        assert_eq!(entry["user_content"], "Write a binary search");
        assert_eq!(entry["router_elapsed_ms"], 340);
        assert_eq!(entry["model"], "nvidia_orchestrator-8b");
        assert!(entry["outcome"].is_null());
        assert!(entry["ts"].as_str().is_some());
    }

    #[test]
    fn test_build_router_decision_entry_outcome_some() {
        let record = RouterDecisionTrace {
            phase: "preflight".to_string(),
            action: "tool".to_string(),
            target: "shell".to_string(),
            confidence: 0.9,
            args: json!({"command": "ls"}),
            user_content: "list files".to_string(),
            router_elapsed_ms: 200,
            model: "test-model".to_string(),
            outcome: Some("file1.txt\nfile2.txt".to_string()),
        };
        let entry = build_router_decision_entry(&record);
        assert_eq!(entry["outcome"], "file1.txt\nfile2.txt");
    }

    #[test]
    fn test_build_router_decision_entry_truncates_user_content() {
        let long_content = "x".repeat(1000);
        let record = RouterDecisionTrace {
            phase: "preflight".to_string(),
            action: "respond".to_string(),
            target: "".to_string(),
            confidence: 0.5,
            args: json!({}),
            user_content: long_content,
            router_elapsed_ms: 100,
            model: "test".to_string(),
            outcome: None,
        };
        let entry = build_router_decision_entry(&record);
        assert_eq!(entry["user_content"].as_str().unwrap().len(), MAX_USER_CONTENT);
    }

    #[test]
    fn test_build_router_decision_entry_truncates_outcome() {
        let long_outcome = "y".repeat(5000);
        let record = RouterDecisionTrace {
            phase: "tool_routing".to_string(),
            action: "tool".to_string(),
            target: "shell".to_string(),
            confidence: 0.7,
            args: json!({}),
            user_content: "test".to_string(),
            router_elapsed_ms: 50,
            model: "test".to_string(),
            outcome: Some(long_outcome),
        };
        let entry = build_router_decision_entry(&record);
        assert_eq!(entry["outcome"].as_str().unwrap().len(), MAX_OUTCOME);
    }

    #[test]
    fn test_build_router_decision_entry_json_roundtrip() {
        let record = RouterDecisionTrace {
            phase: "preflight".to_string(),
            action: "subagent".to_string(),
            target: "research".to_string(),
            confidence: 0.75,
            args: json!({"query": "test"}),
            user_content: "do research".to_string(),
            router_elapsed_ms: 500,
            model: "model-v1".to_string(),
            outcome: Some("found results".to_string()),
        };
        let entry = build_router_decision_entry(&record);
        let serialized = serde_json::to_string(&entry).unwrap();
        let parsed: Value = serde_json::from_str(&serialized).unwrap();
        assert_eq!(parsed["type"], "router_decision");
        assert_eq!(parsed["phase"], "preflight");
        assert_eq!(parsed["action"], "subagent");
        assert_eq!(parsed["target"], "research");
        assert_eq!(parsed["confidence"], 0.75);
        assert_eq!(parsed["outcome"], "found results");
    }

    #[test]
    fn test_build_router_decision_entry_all_actions() {
        for action in &["respond", "tool", "specialist", "subagent", "ask_user"] {
            let record = RouterDecisionTrace {
                phase: "preflight".to_string(),
                action: action.to_string(),
                target: "t".to_string(),
                confidence: 0.5,
                args: json!({}),
                user_content: "test".to_string(),
                router_elapsed_ms: 10,
                model: "m".to_string(),
                outcome: None,
            };
            let entry = build_router_decision_entry(&record);
            assert_eq!(entry["action"].as_str().unwrap(), *action);
        }
    }

    #[test]
    fn test_build_trace_entry_has_type_field() {
        let record = DispatchRecord {
            specialist_name: "coding".to_string(),
            specialist_model: "model".to_string(),
            router_action: "specialist".to_string(),
            router_target: "coding".to_string(),
            router_confidence: 0.9,
            router_args: json!({}),
            user_content: "test".to_string(),
            messages_count: 0,
            tool_results: vec![],
            specialist_response: "done".to_string(),
        };
        let entry = build_trace_entry(&record);
        assert_eq!(entry["type"], "specialist_dispatch");
    }
}
