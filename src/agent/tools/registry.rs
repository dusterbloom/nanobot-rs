#![allow(dead_code)]
//! Tool registry for dynamic tool management.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use serde_json::Value;

use super::base::{Tool, ToolExecutionContext, ToolExecutionResult};
use super::{
    EditFileTool, ExecTool, ListDirTool, ReadFileTool, ReadSkillTool, RecallTool, WebFetchTool,
    WebSearchTool, WriteFileTool,
};
use crate::agent::system_state::TaskPhase;

/// Configuration for building a standard tool registry.
///
/// Consolidates the divergent parameters across agent_loop, subagent, and pipeline
/// into a single source of truth.
pub struct ToolConfig {
    pub workspace: PathBuf,
    pub exec_timeout: u64,
    pub restrict_to_workspace: bool,
    pub max_tool_result_chars: usize,
    pub brave_api_key: Option<String>,
    /// When true, exclude write_file and edit_file.
    pub read_only: bool,
    /// If set, only register tools in this list. Empty = register all.
    pub tools_filter: Option<Vec<String>>,
    /// Optional override for exec working directory.
    pub exec_working_dir: Option<String>,
    /// Search backend: "searxng" (default) or "brave".
    pub search_provider: String,
    /// Base URL of the SearXNG instance (default: "http://localhost:8888").
    pub searxng_url: String,
}

impl ToolConfig {
    /// Sensible defaults for a given workspace.
    pub fn new(workspace: &Path) -> Self {
        Self {
            workspace: workspace.to_path_buf(),
            exec_timeout: 30,
            restrict_to_workspace: false,
            max_tool_result_chars: 30_000,
            brave_api_key: None,
            read_only: false,
            tools_filter: None,
            exec_working_dir: None,
            search_provider: "searxng".to_string(),
            searxng_url: "http://localhost:8888".to_string(),
        }
    }
}

/// Registry for agent tools.
///
/// Allows dynamic registration and execution of tools.
pub struct ToolRegistry {
    tools: HashMap<String, Box<dyn Tool>>,
}

impl ToolRegistry {
    /// Normalize model-emitted tool names/params into canonical tool contract.
    ///
    /// This keeps small/local models focused by repairing common drift:
    /// - alias tool names (`wait/check/list/cancel` -> `spawn` with action)
    /// - parameter aliases (`q` -> `query`, `file` -> `path`, etc.)
    /// - strict required-arg validation by action
    fn normalize_tool_request(
        name: &str,
        mut params: HashMap<String, Value>,
    ) -> Result<(String, HashMap<String, Value>), String> {
        let canonical_name = match name {
            "wait" | "check" | "list" | "cancel" => "spawn",
            other => other,
        };

        if canonical_name == "spawn" {
            // If the model called alias tools directly, translate to spawn actions.
            if !params.contains_key("action") {
                if matches!(name, "wait" | "check" | "list" | "cancel") {
                    params.insert("action".to_string(), Value::String(name.to_string()));
                }
            }

            let action = params
                .get("action")
                .and_then(|v| v.as_str())
                .unwrap_or("spawn")
                .to_ascii_lowercase();
            params.insert("action".to_string(), Value::String(action.clone()));

            // Alias normalization for task_id.
            if !params.contains_key("task_id") {
                if let Some(id) = params.get("id").cloned() {
                    params.insert("task_id".to_string(), id);
                }
            }

            match action.as_str() {
                "spawn" | "loop" => {
                    Self::require_non_empty_string(&params, "task", "spawn")?;
                }
                "check" | "wait" | "cancel" => {
                    Self::require_non_empty_string(&params, "task_id", "spawn")?;
                }
                "pipeline" => match params.get("steps").and_then(|v| v.as_array()) {
                    Some(arr) if !arr.is_empty() => {}
                    _ => {
                        return Err(
                            "Tool 'spawn' with action='pipeline' requires non-empty 'steps' array"
                                .to_string(),
                        )
                    }
                },
                "list" => {}
                _ => {
                    return Err(format!(
                        "Tool 'spawn' has invalid action '{}'. Allowed: spawn, list, check, wait, cancel, pipeline, loop",
                        action
                    ))
                }
            }
        }

        if canonical_name == "web_search" {
            if !params.contains_key("query") {
                if let Some(v) = params.get("q").cloned() {
                    params.insert("query".to_string(), v);
                } else if let Some(v) = params.get("search_query").cloned() {
                    params.insert("query".to_string(), v);
                }
            }
            Self::require_non_empty_string(&params, "query", "web_search")?;
        }

        if canonical_name == "web_fetch" {
            if !params.contains_key("url") {
                if let Some(v) = params.get("link").cloned() {
                    params.insert("url".to_string(), v);
                }
            }
            Self::require_non_empty_string(&params, "url", "web_fetch")?;
        }

        if canonical_name == "read_file" {
            if !params.contains_key("path") {
                if let Some(v) = params
                    .get("file_path")
                    .cloned()
                    .or_else(|| params.get("filepath").cloned())
                    .or_else(|| params.get("file").cloned())
                {
                    params.insert("path".to_string(), v);
                }
            }
            Self::require_non_empty_string(&params, "path", "read_file")?;
        }

        if canonical_name == "write_file" {
            if !params.contains_key("path") {
                if let Some(v) = params
                    .get("file_path")
                    .cloned()
                    .or_else(|| params.get("filepath").cloned())
                    .or_else(|| params.get("file").cloned())
                {
                    params.insert("path".to_string(), v);
                }
            }
            Self::require_non_empty_string(&params, "path", "write_file")?;
        }

        if canonical_name == "edit_file" {
            if !params.contains_key("path") {
                if let Some(v) = params
                    .get("file_path")
                    .cloned()
                    .or_else(|| params.get("filepath").cloned())
                    .or_else(|| params.get("file").cloned())
                {
                    params.insert("path".to_string(), v);
                }
            }
            Self::require_non_empty_string(&params, "path", "edit_file")?;
        }

        if canonical_name == "list_dir" {
            if !params.contains_key("path") {
                if let Some(v) = params
                    .get("dir_path")
                    .cloned()
                    .or_else(|| params.get("directory").cloned())
                    .or_else(|| params.get("dir").cloned())
                {
                    params.insert("path".to_string(), v);
                }
            }
        }

        Ok((canonical_name.to_string(), params))
    }

    fn require_non_empty_string(
        params: &HashMap<String, Value>,
        key: &str,
        tool_name: &str,
    ) -> Result<(), String> {
        match params.get(key).and_then(|v| v.as_str()).map(str::trim) {
            Some(s) if !s.is_empty() => Ok(()),
            _ => Err(format!(
                "Tool '{}' requires non-empty '{}' parameter",
                tool_name, key
            )),
        }
    }

    /// Create a new, empty registry.
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Create a registry pre-populated with standard stateless tools.
    ///
    /// This is the single place that maps ToolConfig → registered tools.
    /// Agent loop, subagent, and pipeline all call this instead of
    /// duplicating tool registration.
    pub fn with_standard_tools(config: &ToolConfig) -> Self {
        let mut registry = Self::new();
        registry.register_standard_tools(config);
        registry
    }

    /// Register the standard stateless tools based on config.
    ///
    /// Handles filtering (read_only, tools_filter) and wires all params
    /// from ToolConfig. Callers can add stateful tools (MessageTool, SpawnTool)
    /// after this.
    pub fn register_standard_tools(&mut self, config: &ToolConfig) {
        let should_include = |name: &str| -> bool {
            if config.read_only && matches!(name, "write_file" | "edit_file") {
                return false;
            }
            if let Some(ref filter) = config.tools_filter {
                return filter.iter().any(|t| t == name);
            }
            true
        };

        if should_include("read_file") {
            self.register(Box::new(ReadFileTool));
        }
        if should_include("write_file") {
            self.register(Box::new(WriteFileTool));
        }
        if should_include("edit_file") {
            self.register(Box::new(EditFileTool));
        }
        if should_include("list_dir") {
            self.register(Box::new(ListDirTool));
        }
        if should_include("exec") {
            let exec_cwd = config
                .exec_working_dir
                .clone()
                .unwrap_or_else(|| {
                    std::env::current_dir()
                        .map(|p| p.to_string_lossy().to_string())
                        .unwrap_or_else(|_| config.workspace.to_string_lossy().to_string())
                });
            self.register(Box::new(ExecTool::new(
                config.exec_timeout,
                Some(exec_cwd),
                None,
                None,
                config.restrict_to_workspace,
                config.max_tool_result_chars,
            )));
        }
        if should_include("web_search") {
            self.register(Box::new(WebSearchTool::new(
                config.brave_api_key.clone(),
                5,
                config.search_provider.clone(),
                config.searxng_url.clone(),
            )));
        }
        if should_include("web_fetch") {
            self.register(Box::new(WebFetchTool::new(config.max_tool_result_chars)));
        }
        if should_include("recall") {
            self.register(Box::new(RecallTool::new(&config.workspace)));
        }
        if should_include("read_skill") {
            self.register(Box::new(ReadSkillTool::new(&config.workspace)));
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
        let (name, params) = match Self::normalize_tool_request(name, params) {
            Ok(v) => v,
            Err(e) => return ToolExecutionResult::failure(e),
        };

        let tool = match self.tools.get(&name) {
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

    /// Execute a tool by name with a [`ToolExecutionContext`] for progress
    /// reporting and cancellation support.
    ///
    /// Same as [`execute`] but passes the context through to the tool.
    pub async fn execute_with_context(
        &self,
        name: &str,
        params: HashMap<String, serde_json::Value>,
        ctx: &ToolExecutionContext,
    ) -> ToolExecutionResult {
        let (name, params) = match Self::normalize_tool_request(name, params) {
            Ok(v) => v,
            Err(e) => return ToolExecutionResult::failure(e),
        };

        let tool = match self.tools.get(&name) {
            Some(t) => t,
            None => {
                return ToolExecutionResult::failure(format!("Tool '{}' not found", name));
            }
        };

        let fut = std::panic::AssertUnwindSafe(tool.execute_with_result_and_context(params, ctx));
        match futures_util::FutureExt::catch_unwind(fut).await {
            Ok(result) => result,
            Err(_) => {
                ToolExecutionResult::failure(format!("Tool '{}' panicked during execution", name))
            }
        }
    }

    /// Core tools that are always included in tool definitions.
    const CORE_TOOLS: &'static [&'static str] = &[
        "read_file",
        "write_file",
        "edit_file",
        "list_dir",
        "exec",
        "spawn",
    ];

    /// Minimal tool set for local models — saves context tokens.
    const LOCAL_CORE_TOOLS: &'static [&'static str] = &[
        "read_file",
        "write_file",
        "list_dir",
        "exec",
    ];

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
        (
            &[
                "recall",
                "remember",
                "memory",
                "past",
                "previous",
                "earlier",
                "last time",
            ],
            "recall",
        ),
        (
            &["skill", "capability", "how to", "technique", "method"],
            "read_skill",
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

    /// Get tool definitions for local models — smaller core set with keyword
    /// expansion to conserve context tokens.
    pub fn get_local_definitions(
        &self,
        messages: &[serde_json::Value],
        used_tools: &HashSet<String>,
    ) -> Vec<serde_json::Value> {
        let mut relevant: HashSet<String> = HashSet::new();

        // Minimal core for local models.
        for name in Self::LOCAL_CORE_TOOLS {
            if self.tools.contains_key(*name) {
                relevant.insert(name.to_string());
            }
        }

        // Include tools the model has already used (it expects them).
        for name in used_tools {
            if self.tools.contains_key(name) {
                relevant.insert(name.clone());
            }
        }

        // Keyword triggers still apply, but only last 3 messages (tighter).
        let recent_text = Self::extract_recent_text(messages, 3);
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

    /// Get the tool names appropriate for a given task phase.
    ///
    /// Returns `None` for phases where all tools should be available
    /// (Idle, Understanding, Planning, Reflection).
    pub fn tools_for_phase(phase: &TaskPhase) -> Option<&'static [&'static str]> {
        match phase {
            TaskPhase::FileEditing => {
                Some(&["read_file", "write_file", "edit_file", "list_dir", "exec"])
            }
            TaskPhase::CodeExecution => Some(&["exec", "read_file", "list_dir"]),
            TaskPhase::WebResearch => Some(&["web_search", "web_fetch", "read_file"]),
            TaskPhase::Communication => Some(&["message", "send_email", "check_inbox"]),
            _ => None, // Idle/Understanding/Planning/Reflection -> all tools
        }
    }

    /// Get tool definitions scoped for the main agent (additive).
    ///
    /// Includes phase tools + keyword-triggered tools + used tools.
    /// This is a gentle scoping — tools are added, not removed.
    pub fn get_scoped_definitions(
        &self,
        phase: &TaskPhase,
        messages: &[serde_json::Value],
        used_tools: &HashSet<String>,
    ) -> Vec<serde_json::Value> {
        let mut relevant: HashSet<String> = HashSet::new();

        // Phase-specific tools (if any).
        if let Some(phase_tools) = Self::tools_for_phase(phase) {
            for name in phase_tools {
                if self.tools.contains_key(*name) {
                    relevant.insert(name.to_string());
                }
            }
        }

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

    /// Get tool definitions scoped for the delegation model (strict).
    ///
    /// Only returns tools appropriate for the current phase.
    /// For phases with no specific tool set, returns all tools.
    pub fn get_delegation_definitions(&self, phase: &TaskPhase) -> Vec<serde_json::Value> {
        match Self::tools_for_phase(phase) {
            Some(phase_tools) => {
                let allowed: HashSet<&str> = phase_tools.iter().copied().collect();
                self.tools
                    .iter()
                    .filter(|(name, _)| allowed.contains(name.as_str()))
                    .map(|(_, tool)| tool.to_schema())
                    .collect()
            }
            None => self.get_definitions(),
        }
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

    /// Echoes the full params as JSON (used to validate request normalization).
    struct ParamEchoTool {
        tool_name: String,
    }

    impl ParamEchoTool {
        fn new(name: &str) -> Self {
            Self {
                tool_name: name.to_string(),
            }
        }
    }

    #[async_trait]
    impl Tool for ParamEchoTool {
        fn name(&self) -> &str {
            &self.tool_name
        }

        fn description(&self) -> &str {
            "Echo params as JSON"
        }

        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({
                "type": "object",
                "properties": {}
            })
        }

        async fn execute(&self, params: HashMap<String, serde_json::Value>) -> String {
            serde_json::to_string(&params).unwrap_or_else(|_| "{}".to_string())
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

    #[tokio::test]
    async fn test_execute_alias_wait_maps_to_spawn_action() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(ParamEchoTool::new("spawn")));

        let mut params = HashMap::new();
        params.insert(
            "task_id".to_string(),
            serde_json::Value::String("abc123".to_string()),
        );

        let result = registry.execute("wait", params).await;
        assert!(result.ok, "{}", result.data);
        let parsed: serde_json::Value = serde_json::from_str(&result.data).unwrap();
        assert_eq!(parsed["action"], "wait");
        assert_eq!(parsed["task_id"], "abc123");
    }

    #[tokio::test]
    async fn test_execute_spawn_requires_task_for_spawn_action() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(ParamEchoTool::new("spawn")));

        let result = registry.execute("spawn", HashMap::new()).await;
        assert!(!result.ok);
        assert!(result
            .error
            .as_deref()
            .unwrap_or_default()
            .contains("requires non-empty 'task'"));
    }

    #[tokio::test]
    async fn test_execute_spawn_check_requires_task_id() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(ParamEchoTool::new("spawn")));

        let mut params = HashMap::new();
        params.insert(
            "action".to_string(),
            serde_json::Value::String("check".to_string()),
        );

        let result = registry.execute("spawn", params).await;
        assert!(!result.ok);
        assert!(result
            .error
            .as_deref()
            .unwrap_or_default()
            .contains("requires non-empty 'task_id'"));
    }

    #[tokio::test]
    async fn test_execute_web_search_normalizes_q_to_query() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(ParamEchoTool::new("web_search")));

        let mut params = HashMap::new();
        params.insert(
            "q".to_string(),
            serde_json::Value::String("latest news".to_string()),
        );

        let result = registry.execute("web_search", params).await;
        assert!(result.ok, "{}", result.data);
        let parsed: serde_json::Value = serde_json::from_str(&result.data).unwrap();
        assert_eq!(parsed["query"], "latest news");
    }

    #[tokio::test]
    async fn test_read_file_file_path_alias() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(ParamEchoTool::new("read_file")));

        let mut params = HashMap::new();
        params.insert(
            "file_path".to_string(),
            serde_json::Value::String("/tmp/test.txt".to_string()),
        );

        let result = registry.execute("read_file", params).await;
        assert!(result.ok, "Expected ok, got error: {:?}", result.error);
        let parsed: serde_json::Value = serde_json::from_str(&result.data).unwrap();
        assert_eq!(parsed["path"], "/tmp/test.txt");
    }

    #[tokio::test]
    async fn test_read_file_filepath_alias() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(ParamEchoTool::new("read_file")));

        let mut params = HashMap::new();
        params.insert(
            "filepath".to_string(),
            serde_json::Value::String("/tmp/test.txt".to_string()),
        );

        let result = registry.execute("read_file", params).await;
        assert!(result.ok, "Expected ok, got error: {:?}", result.error);
        let parsed: serde_json::Value = serde_json::from_str(&result.data).unwrap();
        assert_eq!(parsed["path"], "/tmp/test.txt");
    }

    #[tokio::test]
    async fn test_read_file_file_alias_still_works() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(ParamEchoTool::new("read_file")));

        let mut params = HashMap::new();
        params.insert(
            "file".to_string(),
            serde_json::Value::String("/tmp/test.txt".to_string()),
        );

        let result = registry.execute("read_file", params).await;
        assert!(result.ok, "Expected ok, got error: {:?}", result.error);
        let parsed: serde_json::Value = serde_json::from_str(&result.data).unwrap();
        assert_eq!(parsed["path"], "/tmp/test.txt");
    }

    #[tokio::test]
    async fn test_read_file_path_alias_priority_file_path_over_file() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(ParamEchoTool::new("read_file")));

        // When both file_path and file are present, file_path wins.
        let mut params = HashMap::new();
        params.insert(
            "file_path".to_string(),
            serde_json::Value::String("/correct.txt".to_string()),
        );
        params.insert(
            "file".to_string(),
            serde_json::Value::String("/wrong.txt".to_string()),
        );

        let result = registry.execute("read_file", params).await;
        assert!(result.ok, "Expected ok, got error: {:?}", result.error);
        let parsed: serde_json::Value = serde_json::from_str(&result.data).unwrap();
        assert_eq!(parsed["path"], "/correct.txt");
    }

    #[tokio::test]
    async fn test_write_file_file_path_alias() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(ParamEchoTool::new("write_file")));

        let mut params = HashMap::new();
        params.insert(
            "file_path".to_string(),
            serde_json::Value::String("/tmp/out.txt".to_string()),
        );
        params.insert(
            "content".to_string(),
            serde_json::Value::String("hello".to_string()),
        );

        let result = registry.execute("write_file", params).await;
        assert!(result.ok, "Expected ok, got error: {:?}", result.error);
        let parsed: serde_json::Value = serde_json::from_str(&result.data).unwrap();
        assert_eq!(parsed["path"], "/tmp/out.txt");
    }

    #[tokio::test]
    async fn test_write_file_filepath_alias() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(ParamEchoTool::new("write_file")));

        let mut params = HashMap::new();
        params.insert(
            "filepath".to_string(),
            serde_json::Value::String("/tmp/out.txt".to_string()),
        );
        params.insert(
            "content".to_string(),
            serde_json::Value::String("hello".to_string()),
        );

        let result = registry.execute("write_file", params).await;
        assert!(result.ok, "Expected ok, got error: {:?}", result.error);
        let parsed: serde_json::Value = serde_json::from_str(&result.data).unwrap();
        assert_eq!(parsed["path"], "/tmp/out.txt");
    }

    #[tokio::test]
    async fn test_edit_file_file_path_alias() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(ParamEchoTool::new("edit_file")));

        let mut params = HashMap::new();
        params.insert(
            "file_path".to_string(),
            serde_json::Value::String("/tmp/edit.txt".to_string()),
        );

        let result = registry.execute("edit_file", params).await;
        assert!(result.ok, "Expected ok, got error: {:?}", result.error);
        let parsed: serde_json::Value = serde_json::from_str(&result.data).unwrap();
        assert_eq!(parsed["path"], "/tmp/edit.txt");
    }

    #[tokio::test]
    async fn test_edit_file_filepath_alias() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(ParamEchoTool::new("edit_file")));

        let mut params = HashMap::new();
        params.insert(
            "filepath".to_string(),
            serde_json::Value::String("/tmp/edit.txt".to_string()),
        );

        let result = registry.execute("edit_file", params).await;
        assert!(result.ok, "Expected ok, got error: {:?}", result.error);
        let parsed: serde_json::Value = serde_json::from_str(&result.data).unwrap();
        assert_eq!(parsed["path"], "/tmp/edit.txt");
    }

    #[tokio::test]
    async fn test_list_dir_directory_alias() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(ParamEchoTool::new("list_dir")));

        let mut params = HashMap::new();
        params.insert(
            "directory".to_string(),
            serde_json::Value::String("/tmp".to_string()),
        );

        let result = registry.execute("list_dir", params).await;
        assert!(result.ok, "Expected ok, got error: {:?}", result.error);
        let parsed: serde_json::Value = serde_json::from_str(&result.data).unwrap();
        assert_eq!(parsed["path"], "/tmp");
    }

    #[tokio::test]
    async fn test_list_dir_dir_alias() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(ParamEchoTool::new("list_dir")));

        let mut params = HashMap::new();
        params.insert(
            "dir".to_string(),
            serde_json::Value::String("/tmp".to_string()),
        );

        let result = registry.execute("list_dir", params).await;
        assert!(result.ok, "Expected ok, got error: {:?}", result.error);
        let parsed: serde_json::Value = serde_json::from_str(&result.data).unwrap();
        assert_eq!(parsed["path"], "/tmp");
    }

    #[tokio::test]
    async fn test_list_dir_dir_path_alias() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(ParamEchoTool::new("list_dir")));

        let mut params = HashMap::new();
        params.insert(
            "dir_path".to_string(),
            serde_json::Value::String("/tmp".to_string()),
        );

        let result = registry.execute("list_dir", params).await;
        assert!(result.ok, "Expected ok, got error: {:?}", result.error);
        let parsed: serde_json::Value = serde_json::from_str(&result.data).unwrap();
        assert_eq!(parsed["path"], "/tmp");
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

    #[tokio::test]
    async fn test_execute_with_context() {
        use crate::agent::audit::ToolEvent;
        use crate::agent::tools::base::ToolExecutionContext;

        let mut registry = ToolRegistry::new();
        registry.register(Box::new(MockTool::new("echo")));

        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel::<ToolEvent>();
        let token = tokio_util::sync::CancellationToken::new();
        let ctx = ToolExecutionContext {
            event_tx: tx,
            cancellation_token: token,
            tool_call_id: "call_ctx".to_string(),
        };

        let mut params = HashMap::new();
        params.insert(
            "value".to_string(),
            serde_json::Value::String("world".to_string()),
        );

        let result = registry.execute_with_context("echo", params, &ctx).await;
        assert!(result.ok);
        assert_eq!(result.data, "echo:world");
    }

    #[tokio::test]
    async fn test_execute_with_context_missing_tool() {
        use crate::agent::audit::ToolEvent;
        use crate::agent::tools::base::ToolExecutionContext;

        let registry = ToolRegistry::new();
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel::<ToolEvent>();
        let token = tokio_util::sync::CancellationToken::new();
        let ctx = ToolExecutionContext {
            event_tx: tx,
            cancellation_token: token,
            tool_call_id: "call_missing".to_string(),
        };

        let result = registry
            .execute_with_context("nonexistent", HashMap::new(), &ctx)
            .await;
        assert!(!result.ok);
        assert!(result.data.contains("not found"));
    }

    // -----------------------------------------------------------------------
    // Phase 2: Dynamic Tool Scoping tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_tools_for_phase_file_editing() {
        let tools = ToolRegistry::tools_for_phase(&TaskPhase::FileEditing).unwrap();
        assert!(tools.contains(&"read_file"));
        assert!(tools.contains(&"edit_file"));
        assert!(tools.contains(&"write_file"));
        assert_eq!(tools.len(), 5);
    }

    #[test]
    fn test_tools_for_phase_code_execution() {
        let tools = ToolRegistry::tools_for_phase(&TaskPhase::CodeExecution).unwrap();
        assert!(tools.contains(&"exec"));
        assert!(tools.contains(&"read_file"));
        assert_eq!(tools.len(), 3);
    }

    #[test]
    fn test_tools_for_phase_web_research() {
        let tools = ToolRegistry::tools_for_phase(&TaskPhase::WebResearch).unwrap();
        assert!(tools.contains(&"web_search"));
        assert!(tools.contains(&"web_fetch"));
        assert_eq!(tools.len(), 3);
    }

    #[test]
    fn test_tools_for_phase_idle_returns_none() {
        assert!(ToolRegistry::tools_for_phase(&TaskPhase::Idle).is_none());
    }

    #[test]
    fn test_tools_for_phase_understanding_returns_none() {
        assert!(ToolRegistry::tools_for_phase(&TaskPhase::Understanding).is_none());
    }

    #[test]
    fn test_tools_for_phase_planning_returns_none() {
        assert!(ToolRegistry::tools_for_phase(&TaskPhase::Planning).is_none());
    }

    #[test]
    fn test_delegation_defs_file_editing_strict() {
        let mut registry = ToolRegistry::new();
        for name in &[
            "read_file",
            "write_file",
            "edit_file",
            "list_dir",
            "exec",
            "web_search",
            "message",
        ] {
            registry.register(Box::new(MockTool::new(name)));
        }

        let defs = registry.get_delegation_definitions(&TaskPhase::FileEditing);
        let names: HashSet<String> = defs
            .iter()
            .filter_map(|d| d["function"]["name"].as_str().map(String::from))
            .collect();

        // Strict: only phase tools, no web_search or message
        assert_eq!(names.len(), 5);
        assert!(names.contains("read_file"));
        assert!(names.contains("write_file"));
        assert!(names.contains("edit_file"));
        assert!(!names.contains("web_search"));
        assert!(!names.contains("message"));
    }

    #[test]
    fn test_delegation_defs_idle_returns_all() {
        let mut registry = ToolRegistry::new();
        for name in &["read_file", "web_search", "exec"] {
            registry.register(Box::new(MockTool::new(name)));
        }

        let defs = registry.get_delegation_definitions(&TaskPhase::Idle);
        assert_eq!(defs.len(), 3); // All tools returned for Idle
    }

    #[test]
    fn test_scoped_defs_includes_phase_and_used() {
        let mut registry = ToolRegistry::new();
        for name in &[
            "read_file",
            "write_file",
            "edit_file",
            "list_dir",
            "exec",
            "web_search",
            "web_fetch",
        ] {
            registry.register(Box::new(MockTool::new(name)));
        }

        let messages = vec![serde_json::json!({"role": "user", "content": "edit the code"})];
        let mut used = HashSet::new();
        used.insert("web_fetch".to_string());

        let defs = registry.get_scoped_definitions(&TaskPhase::FileEditing, &messages, &used);
        let names: HashSet<String> = defs
            .iter()
            .filter_map(|d| d["function"]["name"].as_str().map(String::from))
            .collect();

        // Should include phase tools + used tools (web_fetch)
        assert!(names.contains("read_file"));
        assert!(names.contains("edit_file"));
        assert!(names.contains("web_fetch")); // used tool, added back
    }

    #[test]
    fn test_local_defs_minimal_core() {
        let mut registry = ToolRegistry::new();
        for name in &[
            "read_file",
            "write_file",
            "edit_file",
            "list_dir",
            "exec",
            "spawn",
            "web_search",
            "web_fetch",
            "message",
        ] {
            registry.register(Box::new(MockTool::new(name)));
        }

        let messages = vec![serde_json::json!({"role": "user", "content": "fix the bug"})];
        let used = HashSet::new();

        let defs = registry.get_local_definitions(&messages, &used);
        let names: HashSet<String> = defs
            .iter()
            .filter_map(|d| d["function"]["name"].as_str().map(String::from))
            .collect();

        // Local core: read_file, write_file, list_dir, exec
        assert!(names.contains("read_file"));
        assert!(names.contains("list_dir"));
        assert!(names.contains("exec"));
        // Excluded from local core:
        assert!(!names.contains("edit_file"));
        assert!(!names.contains("spawn"));
        assert!(!names.contains("web_search"));
        assert!(!names.contains("message"));
    }

    #[test]
    fn test_local_defs_keyword_expansion() {
        let mut registry = ToolRegistry::new();
        for name in &["read_file", "list_dir", "exec", "web_search"] {
            registry.register(Box::new(MockTool::new(name)));
        }

        // "search" triggers web_search
        let messages = vec![serde_json::json!({"role": "user", "content": "search for rust docs"})];
        let used = HashSet::new();

        let defs = registry.get_local_definitions(&messages, &used);
        let names: HashSet<String> = defs
            .iter()
            .filter_map(|d| d["function"]["name"].as_str().map(String::from))
            .collect();

        assert!(names.contains("web_search")); // keyword-triggered
        assert!(names.contains("read_file"));  // local core
    }

    #[test]
    fn test_local_defs_used_tools_preserved() {
        let mut registry = ToolRegistry::new();
        for name in &["read_file", "list_dir", "exec", "spawn", "message"] {
            registry.register(Box::new(MockTool::new(name)));
        }

        let messages = vec![serde_json::json!({"role": "user", "content": "continue"})];
        let mut used = HashSet::new();
        used.insert("spawn".to_string());

        let defs = registry.get_local_definitions(&messages, &used);
        let names: HashSet<String> = defs
            .iter()
            .filter_map(|d| d["function"]["name"].as_str().map(String::from))
            .collect();

        assert!(names.contains("spawn")); // used tool preserved
        assert!(!names.contains("message")); // not used, not in local core
    }
}
