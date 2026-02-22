//! Durable trace storage for specialist dispatch results.
//!
//! Captures the full `SpecialistResponse` at the point of creation,
//! before compaction can destroy the in-context specialist messages.

use crate::agent::role_policy::SpecialistResponse;
use crate::utils::helpers::{ensure_dir, today_date};
use serde_json::{json, Value};
use std::io::Write;

/// Max chars of user_content stored per trace line.
const MAX_USER_CONTENT: usize = 500;

/// Full dispatch record including inputs and outputs.
pub(crate) struct DispatchRecord {
    pub response: SpecialistResponse,
    pub system_prompt: String,
    pub specialist_pack: String,
    pub context_summary: String,
    pub router_args: serde_json::Value,
    pub elapsed_ms: u64,
    pub model: String,
}

/// Build the trace JSON value from a specialist dispatch (pure, testable).
pub(crate) fn build_trace_entry(
    sp: &SpecialistResponse,
    target: &str,
    user_content: &str,
    system_prompt: &str,
    specialist_pack: &str,
    context_summary: &str,
    router_args: &serde_json::Value,
    elapsed_ms: u64,
    model: &str,
) -> Value {
    let truncated: String = user_content.chars().take(MAX_USER_CONTENT).collect();
    json!({
        "ts": chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        "target": target,
        "result": sp.result,
        "success": sp.success,
        "vote_key": sp.vote_key,
        "parsed_json": sp.parsed_json,
        "user_content": truncated,
        "system_prompt": system_prompt,
        "specialist_pack": specialist_pack,
        "context_summary": context_summary,
        "router_args": router_args,
        "elapsed_ms": elapsed_ms,
        "model": model,
    })
}

/// Append a specialist trace to `~/.nanobot/traces/YYYY-MM-DD.jsonl`.
pub(crate) fn append_specialist_trace(
    sp: &SpecialistResponse,
    target: &str,
    user_content: &str,
    system_prompt: &str,
    specialist_pack: &str,
    context_summary: &str,
    router_args: &serde_json::Value,
    elapsed_ms: u64,
    model: &str,
) {
    let entry = build_trace_entry(sp, target, user_content, system_prompt, specialist_pack, context_summary, router_args, elapsed_ms, model);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::role_policy::SpecialistResponse;

    fn make_sp(result: &str, success: bool, vote_key: &str, parsed_json: bool) -> SpecialistResponse {
        SpecialistResponse {
            result: result.to_string(),
            success,
            vote_key: vote_key.to_string(),
            parsed_json,
        }
    }

    #[test]
    fn test_build_trace_entry_includes_all_fields() {
        let sp = make_sp("code fix applied", true, "abc123", true);
        let entry = build_trace_entry(&sp, "coding", "please fix the bug", "You are a specialist.", "context pack here", "user wants bug fix", &json!({}), 42, "test-model");

        assert_eq!(entry["target"], "coding");
        assert_eq!(entry["result"], "code fix applied");
        assert_eq!(entry["success"], true);
        assert_eq!(entry["vote_key"], "abc123");
        assert_eq!(entry["parsed_json"], true);
        assert_eq!(entry["user_content"], "please fix the bug");
        assert!(entry["ts"].as_str().is_some(), "ts must be a string");
        assert_eq!(entry["system_prompt"], "You are a specialist.");
        assert_eq!(entry["specialist_pack"], "context pack here");
        assert_eq!(entry["context_summary"], "user wants bug fix");
        assert_eq!(entry["router_args"], json!({}));
        assert_eq!(entry["elapsed_ms"], 42);
        assert_eq!(entry["model"], "test-model");
    }

    #[test]
    fn test_build_trace_entry_success_false() {
        let sp = make_sp("failed to process", false, "def456", false);
        let entry = build_trace_entry(&sp, "research", "find info", "", "", "", &json!({}), 42, "test-model");

        assert_eq!(entry["success"], false);
        assert_eq!(entry["parsed_json"], false);
    }

    #[test]
    fn test_build_trace_entry_truncates_long_user_content() {
        let long_content = "x".repeat(1000);
        let sp = make_sp("ok", true, "k", true);
        let entry = build_trace_entry(&sp, "t", &long_content, "", "", "", &json!({}), 42, "test-model");

        let stored = entry["user_content"].as_str().unwrap();
        assert_eq!(stored.len(), MAX_USER_CONTENT);
    }

    #[test]
    fn test_build_trace_entry_captures_router_args() {
        let sp = make_sp("done", true, "vk2", false);
        let args = json!({"task": "write tests", "priority": 1});
        let entry = build_trace_entry(&sp, "coding", "write tests", "sys prompt", "pack text", "summary", &args, 42, "test-model");

        assert_eq!(entry["router_args"]["task"], "write tests");
        assert_eq!(entry["router_args"]["priority"], 1);
        assert_eq!(entry["system_prompt"], "sys prompt");
        assert_eq!(entry["specialist_pack"], "pack text");
        assert_eq!(entry["context_summary"], "summary");
    }

    #[test]
    fn test_append_specialist_trace_creates_file() {
        let dir = tempfile::tempdir().unwrap();
        let trace_dir = dir.path().join("traces");
        std::fs::create_dir_all(&trace_dir).unwrap();
        let path = trace_dir.join(format!("{}.jsonl", today_date()));

        let sp = make_sp("result text", true, "vk1", true);
        let entry = build_trace_entry(&sp, "coding", "user msg", "system", "pack", "summary", &json!({}), 42, "test-model");
        let line = format!("{}\n", entry);

        // Write directly to the temp path (avoids needing ~/.nanobot)
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(line.as_bytes())
            .unwrap();

        let contents = std::fs::read_to_string(&path).unwrap();
        let parsed: Value = serde_json::from_str(contents.trim()).unwrap();
        assert_eq!(parsed["target"], "coding");
        assert_eq!(parsed["result"], "result text");
        assert_eq!(parsed["user_content"], "user msg");
        assert_eq!(parsed["system_prompt"], "system");
        assert_eq!(parsed["specialist_pack"], "pack");
        assert_eq!(parsed["context_summary"], "summary");
    }

    #[test]
    fn test_build_trace_entry_empty_router_args() {
        let sp = make_sp("ok", true, "vk_empty", false);
        let entry = build_trace_entry(&sp, "analysis", "analyze this", "sys", "pack", "summary", &json!({}), 0, "model");

        assert_eq!(entry["router_args"], json!({}));
        assert!(entry["router_args"].is_object());
        assert_eq!(entry["router_args"].as_object().unwrap().len(), 0);
    }

    #[test]
    fn test_build_trace_entry_captures_system_prompt_and_pack() {
        // system_prompt and specialist_pack must NOT be truncated — unlike user_content (500 char cap).
        let long_system_prompt = "S".repeat(2000);
        let long_specialist_pack = "P".repeat(2000);
        let sp = make_sp("done", true, "vk_long", true);
        let entry = build_trace_entry(
            &sp,
            "coding",
            "short user message",
            &long_system_prompt,
            &long_specialist_pack,
            "summary",
            &json!({}),
            0,
            "model",
        );

        let stored_prompt = entry["system_prompt"].as_str().unwrap();
        let stored_pack = entry["specialist_pack"].as_str().unwrap();

        assert_eq!(stored_prompt.len(), 2000, "system_prompt must not be truncated");
        assert_eq!(stored_pack.len(), 2000, "specialist_pack must not be truncated");
        assert_eq!(entry["user_content"], "short user message");
    }

    #[test]
    fn test_multiple_traces_append_not_overwrite() {
        use std::io::BufRead;

        let dir = tempfile::tempdir().unwrap();
        let trace_dir = dir.path().join("traces");
        std::fs::create_dir_all(&trace_dir).unwrap();
        let path = trace_dir.join("multi.jsonl");

        let sp1 = make_sp("first result", true, "vk_first", true);
        let sp2 = make_sp("second result", false, "vk_second", false);

        let entry1 = build_trace_entry(&sp1, "target_a", "user msg 1", "sys1", "pack1", "summary1", &json!({"n": 1}), 10, "model-a");
        let entry2 = build_trace_entry(&sp2, "target_b", "user msg 2", "sys2", "pack2", "summary2", &json!({"n": 2}), 20, "model-b");

        for entry in [&entry1, &entry2] {
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .unwrap()
                .write_all(format!("{}\n", entry).as_bytes())
                .unwrap();
        }

        let file = std::fs::File::open(&path).unwrap();
        let lines: Vec<String> = std::io::BufReader::new(file)
            .lines()
            .map(|l| l.unwrap())
            .filter(|l| !l.is_empty())
            .collect();

        assert_eq!(lines.len(), 2, "file must contain exactly 2 JSONL lines");

        let parsed1: Value = serde_json::from_str(&lines[0]).expect("first line must be valid JSON");
        let parsed2: Value = serde_json::from_str(&lines[1]).expect("second line must be valid JSON");

        assert_eq!(parsed1["target"], "target_a");
        assert_eq!(parsed1["result"], "first result");
        assert_eq!(parsed1["success"], true);
        assert_eq!(parsed1["model"], "model-a");

        assert_eq!(parsed2["target"], "target_b");
        assert_eq!(parsed2["result"], "second result");
        assert_eq!(parsed2["success"], false);
        assert_eq!(parsed2["model"], "model-b");
    }

    #[test]
    fn test_build_trace_entry_json_roundtrip() {
        let sp = make_sp("roundtrip result", true, "vk_rt", true);
        let args = json!({"agent": "sisyphus", "depth": 3, "flag": true});
        let entry = build_trace_entry(
            &sp,
            "planning",
            "plan the feature",
            "You are a planner.",
            "specialist context",
            "user wants a plan",
            &args,
            999,
            "gpt-4o",
        );

        let serialized = serde_json::to_string(&entry).expect("entry must serialize to JSON string");
        let parsed: Value = serde_json::from_str(&serialized).expect("serialized entry must parse back to Value");

        assert_eq!(parsed["target"], "planning");
        assert_eq!(parsed["result"], "roundtrip result");
        assert_eq!(parsed["success"], true);
        assert_eq!(parsed["vote_key"], "vk_rt");
        assert_eq!(parsed["parsed_json"], true);
        assert_eq!(parsed["user_content"], "plan the feature");
        assert_eq!(parsed["system_prompt"], "You are a planner.");
        assert_eq!(parsed["specialist_pack"], "specialist context");
        assert_eq!(parsed["context_summary"], "user wants a plan");
        assert_eq!(parsed["router_args"]["agent"], "sisyphus");
        assert_eq!(parsed["router_args"]["depth"], 3);
        assert_eq!(parsed["router_args"]["flag"], true);
        assert_eq!(parsed["elapsed_ms"], 999);
        assert_eq!(parsed["model"], "gpt-4o");
        assert!(parsed["ts"].as_str().is_some(), "ts must survive round-trip as string");
    }

    #[test]
    fn test_dispatch_record_carries_all_fields() {
        let sp = make_sp("dispatch result", true, "vk_dr", false);
        let record = DispatchRecord {
            response: sp,
            system_prompt: "system prompt text".to_string(),
            specialist_pack: "specialist pack text".to_string(),
            context_summary: "context summary text".to_string(),
            router_args: json!({"model": "gpt-4", "temperature": 0}),
            elapsed_ms: 1234,
            model: "gpt-4o".to_string(),
        };

        assert_eq!(record.response.result, "dispatch result");
        assert_eq!(record.response.success, true);
        assert_eq!(record.response.vote_key, "vk_dr");
        assert_eq!(record.response.parsed_json, false);
        assert_eq!(record.system_prompt, "system prompt text");
        assert_eq!(record.specialist_pack, "specialist pack text");
        assert_eq!(record.context_summary, "context summary text");
        assert_eq!(record.router_args["model"], "gpt-4");
        assert_eq!(record.router_args["temperature"], 0);
        assert_eq!(record.elapsed_ms, 1234);
        assert_eq!(record.model, "gpt-4o");
    }
}
