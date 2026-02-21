//! Deterministic fallback router used when SLM routing is invalid.

use serde_json::json;

use crate::agent::policy::SessionPolicy;
use crate::agent::toolplan::{ToolPlan, ToolPlanAction};

/// Build a deterministic fallback tool plan from user text and available tools.
///
/// Patterns are ordered most-specific-first. Each pattern is guarded by
/// `has_tool()` so missing tools cause graceful fallthrough.
pub fn route(user_text: &str, available_tools: &[String], policy: &SessionPolicy) -> ToolPlan {
    let lower = user_text.to_lowercase();
    let has_tool = |name: &str| available_tools.iter().any(|t| t == name);
    let has_url = lower.contains("http://") || lower.contains("https://");

    // 1. research/summarize + URL → spawn researcher
    //    Must precede plain URL to avoid web_fetch stealing research requests.
    if has_url
        && ["research", "report", "summarize", "summarise", "analyze", "analyse"]
            .iter()
            .any(|kw| lower.contains(kw))
        && has_tool("spawn")
    {
        return ToolPlan {
            action: ToolPlanAction::Subagent,
            target: "researcher".to_string(),
            args: json!({ "task": user_text }),
            confidence: 0.5,
            idempotency_key: "fallback:spawn_researcher".to_string(),
        };
    }

    // 2. Plain URL / hacker news → web_fetch
    if (has_url || lower.contains("hacker news")) && has_tool("web_fetch") {
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

    // 3. "latest news" + local → spawn researcher
    if lower.contains("latest news")
        && has_tool("spawn")
        && (lower.contains("local") || policy.local_only)
    {
        return ToolPlan {
            action: ToolPlanAction::Subagent,
            target: "researcher".to_string(),
            args: json!({
                "task": "Fetch latest news and summarize key points",
                "model": "local",
            }),
            confidence: 0.4,
            idempotency_key: "fallback:spawn_local_news".to_string(),
        };
    }

    // 4. read/show/cat + path → read_file
    if has_path_like(&lower)
        && ["read ", "show ", "cat ", "display ", "open "]
            .iter()
            .any(|kw| lower.contains(kw))
        && has_tool("read_file")
    {
        return ToolPlan {
            action: ToolPlanAction::Tool,
            target: "read_file".to_string(),
            args: json!({ "instruction": user_text }),
            confidence: 0.5,
            idempotency_key: "fallback:read_file".to_string(),
        };
    }

    // 5. write/create + path → write_file
    if has_path_like(&lower)
        && (lower.starts_with("write ")
            || lower.contains("write a new")
            || lower.contains("create a file")
            || lower.contains("save to "))
        && has_tool("write_file")
    {
        return ToolPlan {
            action: ToolPlanAction::Tool,
            target: "write_file".to_string(),
            args: json!({ "instruction": user_text }),
            confidence: 0.4,
            idempotency_key: "fallback:write_file".to_string(),
        };
    }

    // 6. edit/modify + path → edit_file
    if has_path_like(&lower)
        && ["edit ", "modify ", "change ", "fix "]
            .iter()
            .any(|kw| lower.contains(kw))
        && has_tool("edit_file")
    {
        return ToolPlan {
            action: ToolPlanAction::Tool,
            target: "edit_file".to_string(),
            args: json!({ "instruction": user_text }),
            confidence: 0.4,
            idempotency_key: "fallback:edit_file".to_string(),
        };
    }

    // 7. list/ls/directory → list_dir
    if (lower.contains("list ") || lower.contains("ls ") || lower.starts_with("what files"))
        && has_tool("list_dir")
    {
        return ToolPlan {
            action: ToolPlanAction::Tool,
            target: "list_dir".to_string(),
            args: json!({ "path": user_text }),
            confidence: 0.4,
            idempotency_key: "fallback:list_dir".to_string(),
        };
    }

    // 8. run/execute/cargo/npm/git → exec
    if (["run ", "execute "].iter().any(|kw| lower.starts_with(kw))
        || ["run the ", "execute the ", "build the ", "compile the ", "run my "]
            .iter()
            .any(|v| lower.contains(v))
        || ["cargo ", "npm ", "git ", "make ", "python "]
            .iter()
            .any(|cmd| lower.starts_with(cmd)))
        && has_tool("exec")
    {
        return ToolPlan {
            action: ToolPlanAction::Tool,
            target: "exec".to_string(),
            args: json!({ "command": user_text }),
            confidence: 0.3,
            idempotency_key: "fallback:exec".to_string(),
        };
    }

    // 9. search/look up (no path) → web_search
    if ["search for ", "search about ", "look up ", "find out about ", "google "]
        .iter()
        .any(|kw| lower.contains(kw))
        && has_tool("web_search")
    {
        return ToolPlan {
            action: ToolPlanAction::Tool,
            target: "web_search".to_string(),
            args: json!({ "query": user_text }),
            confidence: 0.4,
            idempotency_key: "fallback:web_search".to_string(),
        };
    }

    // 10. default → ask_user
    ToolPlan {
        action: ToolPlanAction::AskUser,
        target: "clarify".to_string(),
        args: json!({"question": "Please clarify the exact task and target source."}),
        confidence: 0.2,
        idempotency_key: "fallback:ask_user".to_string(),
    }
}

/// Heuristic: does the lowercased text look like it contains a file path?
fn has_path_like(lower: &str) -> bool {
    lower.contains('/')
        || lower.contains(".rs")
        || lower.contains(".txt")
        || lower.contains(".md")
        || lower.contains(".json")
        || lower.contains(".py")
        || lower.contains(".js")
        || lower.contains(".ts")
        || lower.contains(".toml")
        || lower.contains(".yaml")
        || lower.contains(".yml")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_tools() -> Vec<String> {
        ["read_file", "write_file", "edit_file", "list_dir", "exec", "web_search", "web_fetch", "spawn"]
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    fn policy() -> SessionPolicy {
        SessionPolicy { local_only: false }
    }

    // ---- Pattern 1: research + URL → researcher ----

    #[test]
    fn test_research_url_routes_to_researcher() {
        let plan = route(
            "research https://arxiv.org/abs/2401.00001 and summarize",
            &all_tools(),
            &policy(),
        );
        assert_eq!(plan.action, ToolPlanAction::Subagent);
        assert_eq!(plan.target, "researcher");
        assert_eq!(plan.confidence, 0.5);
    }

    #[test]
    fn test_summarize_url_routes_to_researcher() {
        let plan = route(
            "summarize this article https://blog.example.com/post",
            &all_tools(),
            &policy(),
        );
        assert_eq!(plan.action, ToolPlanAction::Subagent);
        assert_eq!(plan.target, "researcher");
    }

    // ---- Pattern 2: plain URL → web_fetch ----

    #[test]
    fn test_plain_url_routes_to_web_fetch() {
        let plan = route("check https://example.com", &all_tools(), &policy());
        assert_eq!(plan.action, ToolPlanAction::Tool);
        assert_eq!(plan.target, "web_fetch");
    }

    #[test]
    fn test_hacker_news_routes_to_web_fetch() {
        let plan = route("show me hacker news", &all_tools(), &policy());
        assert_eq!(plan.action, ToolPlanAction::Tool);
        assert_eq!(plan.target, "web_fetch");
        assert!(plan.args["url"].as_str().unwrap().contains("ycombinator"));
    }

    // ---- Pattern 3: latest news local → spawn ----

    #[test]
    fn test_latest_news_local_routes_to_spawn() {
        let local_policy = SessionPolicy { local_only: true };
        let plan = route("get the latest news", &all_tools(), &local_policy);
        assert_eq!(plan.action, ToolPlanAction::Subagent);
        assert_eq!(plan.target, "researcher");
    }

    // ---- Pattern 4: read + path → read_file ----

    #[test]
    fn test_read_path_routes_to_read_file() {
        let plan = route("read /home/user/notes.txt", &all_tools(), &policy());
        assert_eq!(plan.action, ToolPlanAction::Tool);
        assert_eq!(plan.target, "read_file");
    }

    #[test]
    fn test_show_file_routes_to_read_file() {
        let plan = route("show me src/main.rs", &all_tools(), &policy());
        assert_eq!(plan.action, ToolPlanAction::Tool);
        assert_eq!(plan.target, "read_file");
    }

    // ---- Pattern 5: write + path → write_file ----

    #[test]
    fn test_write_new_file_routes_to_write_file() {
        let plan = route("write a new file called output.txt with results", &all_tools(), &policy());
        assert_eq!(plan.action, ToolPlanAction::Tool);
        assert_eq!(plan.target, "write_file");
    }

    // ---- Pattern 6: edit + path → edit_file ----

    #[test]
    fn test_edit_file_routes_to_edit_file() {
        let plan = route("edit src/main.rs and fix the bug", &all_tools(), &policy());
        assert_eq!(plan.action, ToolPlanAction::Tool);
        assert_eq!(plan.target, "edit_file");
    }

    // ---- Pattern 7: list → list_dir ----

    #[test]
    fn test_list_dir_routes_to_list_dir() {
        let plan = route("list the files in /tmp", &all_tools(), &policy());
        assert_eq!(plan.action, ToolPlanAction::Tool);
        assert_eq!(plan.target, "list_dir");
    }

    // ---- Pattern 8: run/execute → exec ----

    #[test]
    fn test_run_cargo_routes_to_exec() {
        let plan = route("run cargo build", &all_tools(), &policy());
        assert_eq!(plan.action, ToolPlanAction::Tool);
        assert_eq!(plan.target, "exec");
    }

    #[test]
    fn test_cargo_direct_routes_to_exec() {
        let plan = route("cargo test --release", &all_tools(), &policy());
        assert_eq!(plan.action, ToolPlanAction::Tool);
        assert_eq!(plan.target, "exec");
    }

    // ---- Pattern 9: search → web_search ----

    #[test]
    fn test_search_routes_to_web_search() {
        let plan = route("search for rust async patterns", &all_tools(), &policy());
        assert_eq!(plan.action, ToolPlanAction::Tool);
        assert_eq!(plan.target, "web_search");
    }

    #[test]
    fn test_look_up_routes_to_web_search() {
        let plan = route("look up the weather in Helsinki", &all_tools(), &policy());
        assert_eq!(plan.action, ToolPlanAction::Tool);
        assert_eq!(plan.target, "web_search");
    }

    // ---- Edge cases ----

    #[test]
    fn test_missing_tool_falls_through() {
        let tools: Vec<String> = ["read_file", "web_search"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let plan = route("run cargo build", &tools, &policy());
        assert_eq!(plan.action, ToolPlanAction::AskUser);
    }

    #[test]
    fn test_ambiguous_defaults_to_ask_user() {
        let plan = route("hello, how are you?", &all_tools(), &policy());
        assert_eq!(plan.action, ToolPlanAction::AskUser);
    }
}
