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

    for rule in FALLBACK_RULES {
        if rule.matches(&lower, available_tools, policy) {
            return rule.plan(user_text, &lower);
        }
    }

    ToolPlan {
        action: ToolPlanAction::AskUser,
        target: "clarify".to_string(),
        args: json!({"question": "Please clarify the exact task and target source."}),
        confidence: 0.2,
        idempotency_key: "fallback:ask_user".to_string(),
    }
}

#[derive(Clone, Copy)]
struct FallbackRule(
    &'static str,
    Select,
    KeywordMatcher,
    ExtraPredicate,
    &'static str,
    f64,
);

#[rustfmt::skip]
const FALLBACK_RULES: &[FallbackRule] = &[
    // Must precede plain URL to avoid web_fetch stealing research requests.
    FallbackRule("spawn_researcher", Select::Subagent("researcher", ArgsKind::Task), KeywordMatcher::ContainsAny(&["research", "report", "summarize", "summarise", "analyze", "analyse"]), ExtraPredicate::ContainsUrl, "spawn", 0.5),
    FallbackRule("web_fetch", Select::Tool("web_fetch", ArgsKind::Url), KeywordMatcher::None, ExtraPredicate::UrlOrHackerNews, "web_fetch", 0.4),
    FallbackRule("spawn_local_news", Select::Subagent("researcher", ArgsKind::LocalNewsTask), KeywordMatcher::ContainsAny(&["latest news"]), ExtraPredicate::LocalNews, "spawn", 0.4),
    FallbackRule("read_file", Select::Tool("read_file", ArgsKind::Instruction), KeywordMatcher::ContainsAny(&["read ", "show ", "cat ", "display ", "open "]), ExtraPredicate::ContainsPath, "read_file", 0.5),
    FallbackRule("write_file", Select::Tool("write_file", ArgsKind::Instruction), KeywordMatcher::Mixed(&["write a new", "create a file", "save to "], &["write "]), ExtraPredicate::ContainsPath, "write_file", 0.4),
    FallbackRule("edit_file", Select::Tool("edit_file", ArgsKind::Instruction), KeywordMatcher::ContainsAny(&["edit ", "modify ", "change ", "fix "]), ExtraPredicate::ContainsPath, "edit_file", 0.4),
    FallbackRule("list_dir", Select::Tool("list_dir", ArgsKind::Path), KeywordMatcher::Mixed(&["list ", "ls "], &["what files"]), ExtraPredicate::None, "list_dir", 0.4),
    FallbackRule("exec", Select::Tool("exec", ArgsKind::Command), KeywordMatcher::Mixed(&["run the ", "execute the ", "build the ", "compile the ", "run my "], &["run ", "execute ", "cargo ", "npm ", "git ", "make ", "python "]), ExtraPredicate::None, "exec", 0.3),
    FallbackRule("web_search", Select::Tool("web_search", ArgsKind::Query), KeywordMatcher::ContainsAny(&["search for ", "search about ", "look up ", "find out about ", "google "]), ExtraPredicate::None, "web_search", 0.4),
];

impl FallbackRule {
    fn matches(&self, lower: &str, available_tools: &[String], policy: &SessionPolicy) -> bool {
        available_tools.iter().any(|tool| tool == self.4)
            && self.2.matches(lower)
            && self.3.matches(lower, policy)
    }

    fn plan(&self, user_text: &str, lower: &str) -> ToolPlan {
        let (action, target, args) = match self.1 {
            Select::Subagent(target, args) => (ToolPlanAction::Subagent, target, args),
            Select::Tool(target, args) => (ToolPlanAction::Tool, target, args),
        };
        ToolPlan {
            action,
            target: target.to_string(),
            args: args.to_json(user_text, lower),
            confidence: self.5,
            idempotency_key: format!("fallback:{}", self.0),
        }
    }
}

#[derive(Clone, Copy)]
enum Select {
    Subagent(&'static str, ArgsKind),
    Tool(&'static str, ArgsKind),
}

#[derive(Clone, Copy)]
enum ArgsKind {
    Command,
    Instruction,
    LocalNewsTask,
    Path,
    Query,
    Task,
    Url,
}

impl ArgsKind {
    fn to_json(self, user_text: &str, lower: &str) -> serde_json::Value {
        match self {
            Self::Command => json!({ "command": user_text }),
            Self::Instruction => json!({ "instruction": user_text }),
            Self::LocalNewsTask => json!({ "task": "Fetch latest news and summarize key points" }),
            Self::Path => json!({ "path": user_text }),
            Self::Query => json!({ "query": user_text }),
            Self::Task => json!({ "task": user_text }),
            Self::Url => json!({ "url": web_fetch_url(user_text, lower) }),
        }
    }
}

#[derive(Clone, Copy)]
enum KeywordMatcher {
    ContainsAny(&'static [&'static str]),
    Mixed(&'static [&'static str], &'static [&'static str]),
    None,
}

impl KeywordMatcher {
    fn matches(self, lower: &str) -> bool {
        match self {
            Self::ContainsAny(terms) => terms.iter().any(|term| lower.contains(term)),
            Self::Mixed(contains, starts) => {
                contains.iter().any(|term| lower.contains(term))
                    || starts.iter().any(|term| lower.starts_with(term))
            }
            Self::None => true,
        }
    }
}

#[derive(Clone, Copy)]
enum ExtraPredicate {
    ContainsPath,
    ContainsUrl,
    LocalNews,
    None,
    UrlOrHackerNews,
}

impl ExtraPredicate {
    fn matches(self, lower: &str, policy: &SessionPolicy) -> bool {
        match self {
            Self::ContainsPath => has_path_like(lower),
            Self::ContainsUrl => has_url(lower),
            Self::LocalNews => lower.contains("local") || policy.local_only,
            Self::None => true,
            Self::UrlOrHackerNews => has_url(lower) || lower.contains("hacker news"),
        }
    }
}

fn has_url(lower: &str) -> bool {
    lower.contains("http://") || lower.contains("https://")
}

fn web_fetch_url(user_text: &str, lower: &str) -> String {
    if lower.contains("hacker news") || lower.contains("hackernews") {
        return "https://news.ycombinator.com/".to_string();
    }

    user_text
        .split_whitespace()
        .find(|word| word.starts_with("http://") || word.starts_with("https://"))
        .unwrap_or("https://example.com")
        .to_string()
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
        [
            "read_file",
            "write_file",
            "edit_file",
            "list_dir",
            "exec",
            "web_search",
            "web_fetch",
            "spawn",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }

    fn policy() -> SessionPolicy {
        SessionPolicy::default()
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
        let local_policy = SessionPolicy {
            local_only: true,
            ..SessionPolicy::default()
        };
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
        let plan = route(
            "write a new file called output.txt with results",
            &all_tools(),
            &policy(),
        );
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
