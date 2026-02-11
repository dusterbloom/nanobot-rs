//! Tool registry for dynamic tool management.

use std::collections::{HashMap, HashSet};

use super::base::{Tool, ToolExecutionResult};

/// Registry for agent tools.
///
/// Allows dynamic registration and execution of tools.
pub struct ToolRegistry {
    tools: HashMap<String, Box<dyn Tool>>,
}

impl ToolRegistry {
    /// Create a new, empty registry.
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Register a tool. Replaces any existing tool with the same name.
    pub fn register(&mut self, tool: Box<dyn Tool>) {
        let name = tool.name().to_string();
        self.tools.insert(name, tool);
    }

    /// Unregister a tool by name.
    pub fn unregister(&mut self, name: &str) {
        self.tools.remove(name);
    }

    /// Get a reference to a tool by name.
    pub fn get(&self, name: &str) -> Option<&dyn Tool> {
        self.tools.get(name).map(|t| t.as_ref())
    }

    /// Check if a tool is registered.
    pub fn has(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// Get all tool definitions in OpenAI format.
    pub fn get_definitions(&self) -> Vec<serde_json::Value> {
        self.tools.values().map(|tool| tool.to_schema()).collect()
    }

    /// Execute a tool by name with given parameters.
    ///
    /// Returns a structured outcome (`ok`, `data`, `error`) so callers can
    /// reason about success/failure without parsing raw strings.
    /// Catches panics so a single tool failure doesn't crash the agent loop.
    pub async fn execute(
        &self,
        name: &str,
        params: HashMap<String, serde_json::Value>,
    ) -> ToolExecutionResult {
        let tool = match self.tools.get(name) {
            Some(t) => t,
            None => {
                return ToolExecutionResult::failure(format!("Tool '{}' not found", name));
            }
        };

        let fut = std::panic::AssertUnwindSafe(tool.execute_with_result(params));
        match futures_util::FutureExt::catch_unwind(fut).await {
            Ok(result) => result,
            Err(_) => {
                ToolExecutionResult::failure(format!("Tool '{}' panicked during execution", name))
            }
        }
    }

    /// Core tools that are always included in tool definitions.
    const CORE_TOOLS: &'static [&'static str] =
        &["read_file", "write_file", "edit_file", "list_dir", "exec"];

    /// Keyword-to-tool mapping for context-triggered tool selection.
    const KEYWORD_TRIGGERS: &'static [(&'static [&'static str], &'static str)] = &[
        (
            &["search", "find online", "look up", "google"],
            "web_search",
        ),
        (
            &["http", "url", "fetch", "website", "webpage", "download"],
            "web_fetch",
        ),
        (&["schedule", "cron", "every", "timer", "periodic"], "cron"),
        (
            &["send", "message", "notify", "tell", "reply to"],
            "message",
        ),
        (
            &["spawn", "agent", "background", "subagent", "delegate"],
            "spawn",
        ),
    ];

    /// Get tool definitions filtered by relevance to the current context.
    ///
    /// Core tools (filesystem, exec) are always included. Other tools are
    /// included if they were previously used in the conversation or if
    /// relevant keywords appear in recent messages.
    pub fn get_relevant_definitions(
        &self,
        messages: &[serde_json::Value],
        used_tools: &HashSet<String>,
    ) -> Vec<serde_json::Value> {
        let mut relevant: HashSet<String> = HashSet::new();

        // Always include core tools.
        for name in Self::CORE_TOOLS {
            if self.tools.contains_key(*name) {
                relevant.insert(name.to_string());
            }
        }

        // Include any tools already used in this conversation.
        for name in used_tools {
            if self.tools.contains_key(name) {
                relevant.insert(name.clone());
            }
        }

        // Scan recent messages for keyword triggers.
        let recent_text = Self::extract_recent_text(messages, 5);
        let lower_text = recent_text.to_lowercase();

        for (keywords, tool_name) in Self::KEYWORD_TRIGGERS {
            if self.tools.contains_key(*tool_name) {
                for kw in *keywords {
                    if lower_text.contains(kw) {
                        relevant.insert(tool_name.to_string());
                        break;
                    }
                }
            }
        }

        // If we end up with all tools anyway, just return everything.
        if relevant.len() >= self.tools.len() {
            return self.get_definitions();
        }

        self.tools
            .iter()
            .filter(|(name, _)| relevant.contains(name.as_str()))
            .map(|(_, tool)| tool.to_schema())
            .collect()
    }

    /// Extract text content from the last N messages for keyword scanning.
    fn extract_recent_text(messages: &[serde_json::Value], n: usize) -> String {
        messages
            .iter()
            .rev()
            .take(n)
            .filter_map(|m| m.get("content").and_then(|c| c.as_str()))
            .collect::<Vec<&str>>()
            .join(" ")
    }

    /// Get list of registered tool names.
    pub fn tool_names(&self) -> Vec<String> {
        self.tools.keys().cloned().collect()
    }

    /// Get the number of registered tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Check if the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Check if a tool name is in the registry.
    pub fn contains(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;

    /// A simple mock tool for registry tests.
    struct MockTool {
        tool_name: String,
    }

    impl MockTool {
        fn new(name: &str) -> Self {
            Self {
                tool_name: name.to_string(),
            }
        }
    }

    #[async_trait]
    impl Tool for MockTool {
        fn name(&self) -> &str {
            &self.tool_name
        }

        fn description(&self) -> &str {
            "A mock tool for testing"
        }

        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "value": { "type": "string" }
                },
                "required": ["value"]
            })
        }

        async fn execute(&self, params: HashMap<String, serde_json::Value>) -> String {
            let value = params
                .get("value")
                .and_then(|v| v.as_str())
                .unwrap_or("default");
            format!("{}:{}", self.tool_name, value)
        }
    }

    #[test]
    fn test_new_registry_is_empty() {
        let registry = ToolRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn test_default_registry_is_empty() {
        let registry = ToolRegistry::default();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn test_register_tool() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(MockTool::new("test_tool")));
        assert_eq!(registry.len(), 1);
        assert!(!registry.is_empty());
    }

    #[test]
    fn test_has_tool() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(MockTool::new("alpha")));

        assert!(registry.has("alpha"));
        assert!(!registry.has("beta"));
    }

    #[test]
    fn test_contains_tool() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(MockTool::new("alpha")));

        assert!(registry.contains("alpha"));
        assert!(!registry.contains("nonexistent"));
    }

    #[test]
    fn test_get_tool() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(MockTool::new("fetch")));

        let tool = registry.get("fetch");
        assert!(tool.is_some());
        assert_eq!(tool.unwrap().name(), "fetch");

        assert!(registry.get("missing").is_none());
    }

    #[test]
    fn test_unregister_tool() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(MockTool::new("to_remove")));
        assert!(registry.has("to_remove"));

        registry.unregister("to_remove");
        assert!(!registry.has("to_remove"));
        assert!(registry.is_empty());
    }

    #[test]
    fn test_unregister_nonexistent_does_nothing() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(MockTool::new("keeper")));
        registry.unregister("nonexistent");
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn test_register_replaces_existing() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(MockTool::new("dup")));
        registry.register(Box::new(MockTool::new("dup")));
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn test_tool_names() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(MockTool::new("alpha")));
        registry.register(Box::new(MockTool::new("beta")));

        let mut names = registry.tool_names();
        names.sort();
        assert_eq!(names, vec!["alpha", "beta"]);
    }

    #[test]
    fn test_get_definitions() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(MockTool::new("def_test")));

        let definitions = registry.get_definitions();
        assert_eq!(definitions.len(), 1);
        assert_eq!(definitions[0]["type"], "function");
        assert_eq!(definitions[0]["function"]["name"], "def_test");
    }

    #[test]
    fn test_len_multiple_tools() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(MockTool::new("a")));
        registry.register(Box::new(MockTool::new("b")));
        registry.register(Box::new(MockTool::new("c")));
        assert_eq!(registry.len(), 3);
    }

    #[tokio::test]
    async fn test_execute_tool() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(MockTool::new("echo")));

        let mut params = HashMap::new();
        params.insert(
            "value".to_string(),
            serde_json::Value::String("hello".to_string()),
        );

        let result = registry.execute("echo", params).await;
        assert!(result.ok);
        assert_eq!(result.data, "echo:hello");
        assert!(result.error.is_none());
    }

    #[tokio::test]
    async fn test_execute_missing_tool() {
        let registry = ToolRegistry::new();
        let params = HashMap::new();

        let result = registry.execute("nonexistent", params).await;
        assert!(!result.ok);
        assert!(result.data.contains("Error"));
        assert!(result.data.contains("nonexistent"));
        assert!(result
            .error
            .as_deref()
            .unwrap_or_default()
            .contains("not found"));
    }

    // -----------------------------------------------------------------------
    // get_relevant_definitions tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_relevant_defs_always_includes_core_tools() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(MockTool::new("read_file")));
        registry.register(Box::new(MockTool::new("write_file")));
        registry.register(Box::new(MockTool::new("exec")));
        registry.register(Box::new(MockTool::new("web_search")));

        let messages = vec![serde_json::json!({"role": "user", "content": "hello"})];
        let used = HashSet::new();
        let defs = registry.get_relevant_definitions(&messages, &used);

        // Core tools should be included; web_search should not (no keyword).
        let names: Vec<String> = defs
            .iter()
            .filter_map(|d| d["function"]["name"].as_str().map(String::from))
            .collect();
        assert!(names.contains(&"read_file".to_string()));
        assert!(names.contains(&"write_file".to_string()));
        assert!(names.contains(&"exec".to_string()));
        assert!(!names.contains(&"web_search".to_string()));
    }

    #[test]
    fn test_relevant_defs_keyword_trigger() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(MockTool::new("read_file")));
        registry.register(Box::new(MockTool::new("web_search")));

        let messages =
            vec![serde_json::json!({"role": "user", "content": "search for Rust async patterns"})];
        let used = HashSet::new();
        let defs = registry.get_relevant_definitions(&messages, &used);

        let names: Vec<String> = defs
            .iter()
            .filter_map(|d| d["function"]["name"].as_str().map(String::from))
            .collect();
        assert!(names.contains(&"web_search".to_string()));
    }

    #[test]
    fn test_relevant_defs_includes_used_tools() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(MockTool::new("read_file")));
        registry.register(Box::new(MockTool::new("web_fetch")));

        let messages = vec![serde_json::json!({"role": "user", "content": "hello"})];
        let mut used = HashSet::new();
        used.insert("web_fetch".to_string());

        let defs = registry.get_relevant_definitions(&messages, &used);
        let names: Vec<String> = defs
            .iter()
            .filter_map(|d| d["function"]["name"].as_str().map(String::from))
            .collect();
        assert!(names.contains(&"web_fetch".to_string()));
    }
}
