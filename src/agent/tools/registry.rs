#![allow(dead_code)]
//! Tool registry for dynamic tool management.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use serde_json::Value;

use super::base::{PermissionLevel, Tool, ToolExecutionContext, ToolExecutionResult};
use super::{
    BrowserTool, CodeExecutionTool, EditFileTool, ExecTool, ListDirTool, ReadFileTool,
    ReadSkillTool, RecallTool, RememberTool, SessionSearchTool, WebFetchTool, WebSearchTool,
    WriteFileTool,
};
use crate::agent::system_state::TaskPhase;
use crate::config::schema::CodeExecutionConfig;

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
    /// Maximum search results to return (default: 5).
    pub search_max_results: u32,
    /// Path to the SQLite sessions database for session_search tool.
    /// When `None`, the session_search tool is not registered.
    pub db_path: Option<PathBuf>,
    /// Code execution tool config. Disabled by default.
    pub code_execution: CodeExecutionConfig,
}

impl ToolConfig {
    /// Sensible defaults for a given workspace.
    pub fn new(workspace: &Path) -> Self {
        Self {
            workspace: workspace.to_path_buf(),
            exec_timeout: 30,
            restrict_to_workspace: false,
            max_tool_result_chars: crate::config::schema::DEFAULT_MAX_TOOL_RESULT_CHARS,
            brave_api_key: None,
            read_only: false,
            tools_filter: None,
            exec_working_dir: None,
            search_provider: "searxng".to_string(),
            searxng_url: "http://localhost:8888".to_string(),
            search_max_results: 5,
            db_path: None,
            code_execution: CodeExecutionConfig::default(),
        }
    }
}

/// Registry for agent tools.
///
/// Allows dynamic registration and execution of tools.
pub struct ToolRegistry {
    tools: HashMap<String, Box<dyn Tool>>,
    /// Maximum permission level allowed. Tools above this ceiling are denied.
    max_permission: PermissionLevel,
    /// Optional hook scripts that run before/after tool calls.
    hooks: Option<crate::config::schema::HooksConfig>,
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

        if canonical_name == "browser" {
            // Normalize element/selector → ref
            if !params.contains_key("ref") {
                if let Some(v) = params
                    .get("element")
                    .cloned()
                    .or_else(|| params.get("selector").cloned())
                {
                    params.insert("ref".to_string(), v);
                }
            }
            Self::require_non_empty_string(&params, "action", "browser")?;
        }

        // Normalize path aliases for file tools.
        let file_aliases: &[&str] = &["file_path", "filepath", "file"];
        let dir_aliases: &[&str] = &["dir_path", "directory", "dir"];

        match canonical_name {
            "read_file" | "write_file" | "edit_file" => {
                Self::normalize_param_aliases(&mut params, "path", file_aliases);
                Self::require_non_empty_string(&params, "path", canonical_name)?;
            }
            "list_dir" => {
                Self::normalize_param_aliases(&mut params, "path", dir_aliases);
            }
            _ => {}
        }

        Ok((canonical_name.to_string(), params))
    }

    fn normalize_param_aliases(
        params: &mut HashMap<String, Value>,
        canonical: &str,
        aliases: &[&str],
    ) {
        if params.contains_key(canonical) {
            return;
        }
        for alias in aliases {
            if let Some(v) = params.get(*alias).cloned() {
                params.insert(canonical.to_string(), v);
                return;
            }
        }
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
            max_permission: PermissionLevel::System,
            hooks: None,
        }
    }

    /// Create a registry with a permission ceiling.
    ///
    /// Tools whose [`PermissionLevel`] exceeds `max` will be denied at
    /// execution time.
    pub fn with_max_permission(max: PermissionLevel) -> Self {
        Self {
            tools: HashMap::new(),
            max_permission: max,
            hooks: None,
        }
    }

    /// Set the maximum permission level for this registry.
    pub fn set_max_permission(&mut self, max: PermissionLevel) {
        self.max_permission = max;
    }

    /// Configure hook scripts that run before/after tool calls.
    pub fn set_hooks(&mut self, hooks: crate::config::schema::HooksConfig) {
        self.hooks = Some(hooks);
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
            let exec_cwd = config.exec_working_dir.clone().unwrap_or_else(|| {
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
                config.search_max_results,
                config.search_provider.clone(),
                config.searxng_url.clone(),
            )));
        }
        if should_include("web_fetch") {
            self.register(Box::new(WebFetchTool::new(config.max_tool_result_chars)));
        }
        if should_include("browser") {
            self.register(Box::new(BrowserTool::new(config.max_tool_result_chars)));
        }
        if should_include("recall") {
            self.register(Box::new(RecallTool::new(&config.workspace)));
        }
        if should_include("remember") {
            self.register(Box::new(RememberTool::new(config.workspace.clone())));
        }
        if should_include("read_skill") {
            self.register(Box::new(ReadSkillTool::new(&config.workspace)));
        }
        if should_include("session_search") {
            if let Some(ref db_path) = config.db_path {
                self.register(Box::new(SessionSearchTool::new(db_path.clone())));
            }
        }
        if config.code_execution.enabled && should_include("execute_code") {
            // Collect tool names for the Python stub (excluding execute_code itself).
            let available_tools: Vec<String> = self
                .tools
                .keys()
                .filter(|n| n.as_str() != "execute_code")
                .cloned()
                .collect();
            self.register(Box::new(CodeExecutionTool::new(
                true,
                config.code_execution.timeout,
                config.code_execution.max_tool_calls,
                available_tools,
                None, // No nested tool_config — scripts get a stub-only registry.
            )));
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
    ///
    /// Tools where [`Tool::is_available`] returns `false` are excluded.
    pub fn get_definitions(&self) -> Vec<serde_json::Value> {
        let mut defs: Vec<serde_json::Value> = self
            .tools
            .values()
            .filter(|tool| tool.is_available())
            .map(|tool| tool.to_schema())
            .collect();
        Self::sort_definitions(&mut defs);
        defs
    }

    /// Order tool definitions deterministically by function name.
    ///
    /// The registry is rebuilt per message (`build_tools`), so its `tools`
    /// HashMap gets a fresh random seed each turn and `values()` yields a
    /// different order every time. The tool block renders at the FRONT of the
    /// chat-template prompt, so an unstable order changes the prompt prefix
    /// every turn — busting the inference server's prefix cache and forcing a
    /// full re-prefill (measured: a warm 3.4k-token turn re-prefills in ~10s
    /// instead of reusing the cache in ~0.4s). A stable order keeps the tools
    /// block byte-identical across turns so the cache hits.
    fn sort_definitions(defs: &mut [serde_json::Value]) {
        defs.sort_by(|a, b| {
            let name = |v: &serde_json::Value| {
                v.pointer("/function/name")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("")
                    .to_owned()
            };
            name(a).cmp(&name(b))
        });
    }

    /// Get tool definitions for only the named tools, in OpenAI format.
    ///
    /// Tools not in the registry or where [`Tool::is_available`] returns `false`
    /// are silently skipped.
    pub fn definitions_for(&self, names: &[String]) -> Vec<serde_json::Value> {
        let name_set: HashSet<&str> = names.iter().map(|s| s.as_str()).collect();
        let mut defs: Vec<serde_json::Value> = self
            .tools
            .values()
            .filter(|tool| tool.is_available() && name_set.contains(tool.name()))
            .map(|tool| tool.to_schema())
            .collect();
        Self::sort_definitions(&mut defs);
        defs
    }

    /// Truncate tool descriptions to their first sentence to save tokens.
    fn condense_definitions(defs: &mut [serde_json::Value]) {
        for def in defs.iter_mut() {
            if let Some(func) = def.get_mut("function") {
                if let Some(desc) = func.get("description").and_then(|d| d.as_str()) {
                    let condensed = Self::condense_description(desc);
                    func["description"] = serde_json::Value::String(condensed);
                }
            }
        }
    }

    /// Condense a description to its first two sentences for richer guidance.
    ///
    /// A single sentence (e.g. "Execute a shell command.") often drops critical
    /// context like safe/blocked commands or output format hints that help
    /// local models use tools correctly.
    fn condense_description(desc: &str) -> String {
        let bytes = desc.as_bytes();
        let mut sentences = 0u8;
        let mut end = 0usize;
        for i in 0..bytes.len() {
            if bytes[i] == b'.' {
                let next = bytes.get(i + 1).copied();
                if matches!(next, Some(b' ') | Some(b'\n') | None) {
                    sentences += 1;
                    end = i + 1;
                    if sentences >= 2 {
                        break;
                    }
                }
            }
        }
        if end > 0 && end < desc.len() {
            desc[..end].trim().to_string()
        } else {
            desc.to_string()
        }
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
        // Proxy intercept: "tool" is the meta-tool, not a registered tool.
        if name == "tool" {
            return self.execute_proxy(params).await;
        }
        self.execute_inner(name, params).await
    }

    /// Run the pre-tool-use hook if configured. Returns `Some(result)` to
    /// short-circuit execution when the hook blocks.
    async fn run_pre_hook(
        &self,
        name: &str,
        params: &HashMap<String, serde_json::Value>,
    ) -> Option<ToolExecutionResult> {
        let script = self.hooks.as_ref()?.pre_tool_use.as_ref()?;
        let path = std::path::Path::new(script);
        let hook_result = crate::agent::hooks::run_hook(
            path,
            crate::agent::hooks::HookPhase::PreToolUse,
            name,
            params,
            None,
        )
        .await?;
        if !hook_result.allowed {
            return Some(ToolExecutionResult::failure(format!(
                "Blocked by PreToolUse hook: {}",
                hook_result.output.trim()
            )));
        }
        None
    }

    /// Run the post-tool-use hook if configured (fire-and-forget semantics).
    async fn run_post_hook(
        &self,
        name: &str,
        params: &HashMap<String, serde_json::Value>,
        result: &ToolExecutionResult,
    ) {
        let script = match self.hooks.as_ref().and_then(|h| h.post_tool_use.as_ref()) {
            Some(s) => s,
            None => return,
        };
        let path = std::path::Path::new(script);
        let _ = crate::agent::hooks::run_hook(
            path,
            crate::agent::hooks::HookPhase::PostToolUse,
            name,
            params,
            Some((&result.data, result.ok)),
        )
        .await;
    }

    /// Core execute logic (no proxy intercept). Called by both `execute()`
    /// and `execute_proxy()` dispatch mode.
    async fn execute_inner(
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

        if tool.permission() > self.max_permission {
            return ToolExecutionResult::failure(format!(
                "Permission denied: tool '{}' requires {:?} but max allowed is {:?}",
                name,
                tool.permission(),
                self.max_permission
            ));
        }

        if let Some(blocked) = self.run_pre_hook(&name, &params).await {
            return blocked;
        }

        let fut = std::panic::AssertUnwindSafe(tool.execute_with_result(params.clone()));
        let result = match futures_util::FutureExt::catch_unwind(fut).await {
            Ok(result) => result,
            Err(_) => {
                ToolExecutionResult::failure(format!("Tool '{}' panicked during execution", name))
            }
        };

        self.run_post_hook(&name, &params, &result).await;
        result
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
        // Proxy intercept: "tool" is the meta-tool, not a registered tool.
        if name == "tool" {
            return self.execute_proxy_with_context(params, ctx).await;
        }
        self.execute_inner_with_context(name, params, ctx).await
    }

    /// Core execute-with-context logic (no proxy intercept).
    async fn execute_inner_with_context(
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

        if tool.permission() > self.max_permission {
            return ToolExecutionResult::failure(format!(
                "Permission denied: tool '{}' requires {:?} but max allowed is {:?}",
                name,
                tool.permission(),
                self.max_permission
            ));
        }

        if let Some(blocked) = self.run_pre_hook(&name, &params).await {
            return blocked;
        }

        let fut =
            std::panic::AssertUnwindSafe(tool.execute_with_result_and_context(params.clone(), ctx));
        let result = match futures_util::FutureExt::catch_unwind(fut).await {
            Ok(result) => result,
            Err(_) => {
                ToolExecutionResult::failure(format!("Tool '{}' panicked during execution", name))
            }
        };

        self.run_post_hook(&name, &params, &result).await;
        result
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

    /// Keyword-to-tool mapping for context-triggered tool selection (cloud path).
    const KEYWORD_TRIGGERS: &'static [(&'static [&'static str], &'static str)] = &[
        (
            &[
                "search",
                "find online",
                "look up",
                "google",
                "news",
                "latest",
                "current events",
                "what's happening",
                "headlines",
                "update on",
                "weather",
                "stock",
                "price of",
            ],
            "web_search",
        ),
        (
            &[
                "fetch",
                "download",
                "read url",
                "get page",
                "web_fetch",
                "scrape",
            ],
            "web_fetch",
        ),
        (
            &[
                "browser", "browse", "click", "navigate", "http", "url", "website", "webpage",
            ],
            "browser",
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
                "memory",
                "past",
                "previous",
                "earlier",
                "last time",
            ],
            "recall",
        ),
        (&["remember", "save", "store", "note this"], "remember"),
        (
            &["skill", "capability", "how to", "technique", "method"],
            "read_skill",
        ),
    ];

    /// Shared logic for building filtered tool definitions.
    ///
    /// `core_tools` — always-included tool names.
    /// `scan_depth` — how many recent messages to scan for keyword triggers.
    fn collect_filtered_definitions(
        &self,
        core_tools: &[&str],
        messages: &[serde_json::Value],
        used_tools: &HashSet<String>,
        scan_depth: usize,
    ) -> Vec<serde_json::Value> {
        let mut relevant: HashSet<String> = HashSet::new();

        for name in core_tools {
            if self.tools.contains_key(*name) {
                relevant.insert(name.to_string());
            }
        }

        for name in used_tools {
            if self.tools.contains_key(name) {
                relevant.insert(name.clone());
            }
        }

        let recent_text = Self::extract_recent_text(messages, scan_depth);
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

        if relevant.len() >= self.tools.len() {
            return self.get_definitions();
        }

        self.tools
            .iter()
            .filter(|(name, tool)| relevant.contains(name.as_str()) && tool.is_available())
            .map(|(_, tool)| tool.to_schema())
            .collect()
    }

    /// Get tool definitions for local models.
    ///
    /// Returns ALL registered + available tools with condensed (two-sentence)
    /// descriptions. Registration is the source of truth — no hardcoded subset.
    /// Token cost is ~350 tokens for 12 tools, affordable even in 32K context.
    pub fn get_local_definitions(&self) -> Vec<serde_json::Value> {
        let mut defs: Vec<serde_json::Value> = self
            .tools
            .values()
            .filter(|tool| tool.is_available())
            .map(|tool| tool.to_schema())
            .collect();
        Self::condense_definitions(&mut defs);
        Self::sort_definitions(&mut defs);
        defs
    }

    /// Returns individual tool schemas with condensed descriptions AND stripped
    /// parameter descriptions. Keeps property names, types, and required list
    /// but removes per-parameter `"description"` fields that consume most tokens.
    pub fn get_slim_definitions(&self) -> Vec<serde_json::Value> {
        // Tools whose parameter semantics are load-bearing and must survive
        // slimming. read_file's `lines` paging syntax is the prime case: strip
        // it and the local model can't page large files and re-prefills the
        // whole file each turn.
        const KEEP_PARAM_DESCRIPTIONS: &[&str] = &["read_file"];
        let mut defs = self.get_local_definitions();
        for def in &mut defs {
            let name = def
                .pointer("/function/name")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if KEEP_PARAM_DESCRIPTIONS.contains(&name) {
                continue;
            }
            if let Some(params) = def.pointer_mut("/function/parameters/properties") {
                if let Some(props) = params.as_object_mut() {
                    for (_key, prop) in props.iter_mut() {
                        if let Some(obj) = prop.as_object_mut() {
                            obj.remove("description");
                        }
                    }
                }
            }
        }
        defs
    }

    // -------------------------------------------------------------------
    // Tool proxy: single compact schema for local models
    // -------------------------------------------------------------------

    /// Sorted list of available tool names. Used by proxy error messages.
    fn available_tool_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .tools
            .values()
            .filter(|t| t.is_available())
            .map(|t| t.name().to_string())
            .collect();
        names.sort();
        names
    }

    /// Extract the first required parameter name from a tool's JSON Schema.
    fn primary_arg_hint(tool: &dyn Tool) -> Option<String> {
        let params = tool.parameters();
        let required = params.get("required")?.as_array()?;
        required.first()?.as_str().map(String::from)
    }

    /// Build arg hints for all required params: `"name(path,content)"`.
    fn tool_hint(tool: &dyn Tool) -> String {
        let params = tool.parameters();
        let hints: Vec<&str> = params
            .get("required")
            .and_then(|r| r.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect::<Vec<&str>>())
            .unwrap_or_default();
        if hints.is_empty() {
            tool.name().to_string()
        } else {
            format!("{}({})", tool.name(), hints.join(","))
        }
    }

    /// Build a single compact proxy schema that lists all available tools.
    ///
    /// Returns one tool definition called `"tool"` whose description embeds
    /// the full catalog with arg hints. The model calls `tool(name: "X")`
    /// to inspect a tool's schema, or `tool(name: "X", args: {...})` to execute.
    ///
    /// Token cost: ~90 tokens vs ~2045 for 15 individual schemas.
    pub fn get_proxy_definition(&self) -> Vec<serde_json::Value> {
        let mut hints: Vec<String> = self
            .tools
            .values()
            .filter(|t| t.is_available())
            .map(|t| Self::tool_hint(t.as_ref()))
            .collect();
        hints.sort();

        let description = format!(
            "Use any tool. Available: {}. \
             Pass name only to see full parameters, or name+args to execute.",
            hints.join(", ")
        );

        vec![serde_json::json!({
            "type": "function",
            "function": {
                "name": "tool",
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": { "type": "string", "description": "Tool name" },
                        "args": { "type": "object", "description": "Tool arguments (omit to see schema)" }
                    },
                    "required": ["name"]
                }
            }
        })]
    }

    /// Handle a proxy call: inspect (no args) or dispatch (with args).
    pub async fn execute_proxy(
        &self,
        params: HashMap<String, serde_json::Value>,
    ) -> ToolExecutionResult {
        let tool_name = match params.get("name").and_then(|v| v.as_str()) {
            Some(n) => n.to_string(),
            None => {
                let names = self.available_tool_names();
                return ToolExecutionResult::failure(format!(
                    "Missing 'name'. Available tools: {}",
                    names.join(", ")
                ));
            }
        };

        match params.get("args") {
            None | Some(serde_json::Value::Null) => {
                // Inspect mode: return tool's full schema
                match self.tools.get(&tool_name) {
                    Some(tool) if tool.is_available() => {
                        let schema = serde_json::json!({
                            "name": tool.name(),
                            "description": tool.description(),
                            "parameters": tool.parameters(),
                        });
                        ToolExecutionResult::success(
                            serde_json::to_string_pretty(&schema).unwrap_or_else(|_| "{}".into()),
                        )
                    }
                    _ => {
                        let names = self.available_tool_names();
                        ToolExecutionResult::failure(format!(
                            "Tool '{}' not found. Available: {}",
                            tool_name,
                            names.join(", ")
                        ))
                    }
                }
            }
            Some(args_val) => {
                // Dispatch mode: extract args and call the real tool.
                // Call execute_inner directly to avoid recursion through execute().
                let inner_params: HashMap<String, serde_json::Value> = match args_val {
                    serde_json::Value::Object(map) => {
                        map.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
                    }
                    _ => {
                        return ToolExecutionResult::failure(
                            "'args' must be a JSON object".to_string(),
                        );
                    }
                };
                self.execute_inner(&tool_name, inner_params).await
            }
        }
    }

    /// Handle a proxy call with execution context passthrough.
    pub async fn execute_proxy_with_context(
        &self,
        params: HashMap<String, serde_json::Value>,
        ctx: &ToolExecutionContext,
    ) -> ToolExecutionResult {
        let tool_name = match params.get("name").and_then(|v| v.as_str()) {
            Some(n) => n.to_string(),
            None => {
                let names = self.available_tool_names();
                return ToolExecutionResult::failure(format!(
                    "Missing 'name'. Available tools: {}",
                    names.join(", ")
                ));
            }
        };

        match params.get("args") {
            None | Some(serde_json::Value::Null) => {
                // Inspect mode (same as non-context version)
                match self.tools.get(&tool_name) {
                    Some(tool) if tool.is_available() => {
                        let schema = serde_json::json!({
                            "name": tool.name(),
                            "description": tool.description(),
                            "parameters": tool.parameters(),
                        });
                        ToolExecutionResult::success(
                            serde_json::to_string_pretty(&schema).unwrap_or_else(|_| "{}".into()),
                        )
                    }
                    _ => {
                        let names = self.available_tool_names();
                        ToolExecutionResult::failure(format!(
                            "Tool '{}' not found. Available: {}",
                            tool_name,
                            names.join(", ")
                        ))
                    }
                }
            }
            Some(args_val) => {
                let inner_params: HashMap<String, serde_json::Value> = match args_val {
                    serde_json::Value::Object(map) => {
                        map.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
                    }
                    _ => {
                        return ToolExecutionResult::failure(
                            "'args' must be a JSON object".to_string(),
                        );
                    }
                };
                self.execute_inner_with_context(&tool_name, inner_params, ctx)
                    .await
            }
        }
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
            TaskPhase::WebResearch => Some(&["web_search", "web_fetch", "browser", "read_file"]),
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
        let phase_tools = Self::tools_for_phase(phase)
            .map(|pt| pt.to_vec())
            .unwrap_or_default();
        let mut core: Vec<&str> = phase_tools.iter().copied().collect();
        for name in Self::CORE_TOOLS {
            if !core.contains(name) {
                core.push(name);
            }
        }
        self.collect_filtered_definitions(&core, messages, used_tools, 5)
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

    /// Tool definitions must be byte-identical across registry instances
    /// regardless of registration order or HashMap seed. The registry is
    /// rebuilt per message, so an unstable order changes the prompt's tool
    /// block every turn and busts the inference server's prefix cache
    /// (measured: a warm turn re-prefilling ~10s instead of ~1s). Build two
    /// registries with the SAME tools registered in DIFFERENT orders and
    /// require identical output from every definition accessor.
    #[test]
    fn test_slim_keeps_read_file_param_descriptions() {
        // The local model gets slim definitions by default. read_file's `lines`
        // paging syntax is load-bearing — it must survive slimming, while a
        // normal tool's param descriptions are still stripped to save tokens.
        let mut reg = ToolRegistry::new();
        reg.register(Box::new(ReadFileTool));
        reg.register(Box::new(ListDirTool));

        let slim = reg.get_slim_definitions();
        let find = |name: &str| {
            slim.iter()
                .find(|d| d.pointer("/function/name").and_then(|v| v.as_str()) == Some(name))
                .unwrap()
                .clone()
        };

        // read_file keeps its `lines` description (paging guidance preserved).
        let rf = find("read_file");
        let lines_desc = rf.pointer("/function/parameters/properties/lines/description");
        assert!(
            lines_desc.and_then(|v| v.as_str()).is_some_and(|s| s.contains("1:")),
            "read_file lines description must survive slim: {rf:?}"
        );

        // A non-allowlisted tool still has every param description stripped.
        let other = find("list_dir");
        if let Some(props) = other
            .pointer("/function/parameters/properties")
            .and_then(|v| v.as_object())
        {
            for (k, prop) in props {
                assert!(
                    prop.get("description").is_none(),
                    "list_dir param '{k}' description should be stripped in slim"
                );
            }
        }
    }

    #[test]
    fn test_definitions_order_is_deterministic() {
        let names = ["zeta", "alpha", "mike", "bravo", "yankee"];
        let mut reg_a = ToolRegistry::new();
        for n in names {
            reg_a.register(Box::new(MockTool::new(n)));
        }
        let mut reg_b = ToolRegistry::new();
        for n in names.iter().rev() {
            reg_b.register(Box::new(MockTool::new(n)));
        }

        let extract = |defs: &[serde_json::Value]| -> Vec<String> {
            defs.iter()
                .map(|d| {
                    d.pointer("/function/name")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("")
                        .to_owned()
                })
                .collect()
        };

        let want = vec!["alpha", "bravo", "mike", "yankee", "zeta"];
        // Same order from both registries (seed/registration-order independent).
        assert_eq!(extract(&reg_a.get_definitions()), want);
        assert_eq!(extract(&reg_b.get_definitions()), want);
        // And from the local + slim accessors that the prefix-cache path uses.
        assert_eq!(extract(&reg_a.get_local_definitions()), want);
        assert_eq!(extract(&reg_b.get_local_definitions()), want);
        assert_eq!(extract(&reg_a.get_slim_definitions()), want);
        assert_eq!(extract(&reg_b.get_slim_definitions()), want);
        // definitions_for preserves the deterministic order for a subset.
        let subset = ["zeta".to_owned(), "alpha".to_owned(), "mike".to_owned()];
        assert_eq!(
            extract(&reg_a.definitions_for(&subset)),
            vec!["alpha", "mike", "zeta"]
        );
    }

    #[test]
    fn test_default_registry_is_empty() {
        let registry = ToolRegistry::default();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn test_tool_config_new_uses_global_default_max_tool_result_chars() {
        let cfg = ToolConfig::new(std::path::Path::new("/tmp"));
        assert_eq!(
            cfg.max_tool_result_chars,
            crate::config::schema::DEFAULT_MAX_TOOL_RESULT_CHARS
        );
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
    // condense_description tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_condense_description_with_period() {
        let desc = "Read a file from disk. Supports binary and text files. Very useful.";
        assert_eq!(
            ToolRegistry::condense_description(desc),
            "Read a file from disk. Supports binary and text files."
        );
    }

    #[test]
    fn test_condense_description_without_period() {
        let desc = "A mock tool for testing";
        assert_eq!(
            ToolRegistry::condense_description(desc),
            "A mock tool for testing"
        );
    }

    #[test]
    fn test_get_definitions_preserves_full_descriptions() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(MockTool::new("test_tool")));
        let defs = registry.get_definitions();
        let desc = defs[0]["function"]["description"].as_str().unwrap();
        assert_eq!(desc, "A mock tool for testing");
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
        assert!(tools.contains(&"browser"));
        assert_eq!(tools.len(), 4);
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
    fn test_scoped_defs_includes_phase_and_used() {
        let mut registry = ToolRegistry::new();
        for name in &[
            "read_file",
            "write_file",
            "edit_file",
            "list_dir",
            "exec",
            "web_search",
            "browser",
        ] {
            registry.register(Box::new(MockTool::new(name)));
        }

        let messages = vec![serde_json::json!({"role": "user", "content": "edit the code"})];
        let mut used = HashSet::new();
        used.insert("browser".to_string());

        let defs = registry.get_scoped_definitions(&TaskPhase::FileEditing, &messages, &used);
        let names: HashSet<String> = defs
            .iter()
            .filter_map(|d| d["function"]["name"].as_str().map(String::from))
            .collect();

        // Should include phase tools + used tools (browser)
        assert!(names.contains("read_file"));
        assert!(names.contains("edit_file"));
        assert!(names.contains("browser")); // used tool, added back
    }

    /// Old behavior hid tools from local models. New behavior: all registered
    /// tools are visible regardless of message content or used_tools.
    #[test]
    fn test_local_defs_all_registered_visible() {
        let mut registry = ToolRegistry::new();
        for name in &[
            "read_file",
            "write_file",
            "edit_file",
            "list_dir",
            "exec",
            "spawn",
            "web_search",
            "browser",
            "message",
        ] {
            registry.register(Box::new(MockTool::new(name)));
        }

        let defs = registry.get_local_definitions();
        let names: HashSet<String> = defs
            .iter()
            .filter_map(|d| d["function"]["name"].as_str().map(String::from))
            .collect();

        assert_eq!(
            names.len(),
            9,
            "All 9 registered tools must be visible: {:?}",
            names
        );
        for tool in &[
            "read_file",
            "write_file",
            "edit_file",
            "list_dir",
            "exec",
            "spawn",
            "web_search",
            "browser",
            "message",
        ] {
            assert!(names.contains(*tool), "Missing '{}' in {:?}", tool, names);
        }
    }

    // -----------------------------------------------------------------------
    // is_available() gating tests
    // -----------------------------------------------------------------------

    /// A tool that reports itself as unavailable.
    struct UnavailableTool;

    #[async_trait]
    impl Tool for UnavailableTool {
        fn name(&self) -> &str {
            "unavailable_test"
        }
        fn description(&self) -> &str {
            "test tool that is unavailable"
        }
        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({"type": "object", "properties": {}})
        }
        async fn execute(&self, _params: HashMap<String, serde_json::Value>) -> String {
            "executed".to_string()
        }
        fn is_available(&self) -> bool {
            false
        }
    }

    #[test]
    fn test_unavailable_tool_excluded_from_get_definitions() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(MockTool::new("available_tool")));
        registry.register(Box::new(UnavailableTool));

        let defs = registry.get_definitions();
        let names: Vec<String> = defs
            .iter()
            .filter_map(|d| d["function"]["name"].as_str().map(String::from))
            .collect();

        assert!(
            names.contains(&"available_tool".to_string()),
            "Available tool should appear in definitions"
        );
        assert!(
            !names.contains(&"unavailable_test".to_string()),
            "Unavailable tool must NOT appear in definitions"
        );
    }

    #[test]
    fn test_available_tool_included_in_get_definitions() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(MockTool::new("my_tool")));

        let defs = registry.get_definitions();
        let names: Vec<String> = defs
            .iter()
            .filter_map(|d| d["function"]["name"].as_str().map(String::from))
            .collect();

        assert!(names.contains(&"my_tool".to_string()));
    }

    #[tokio::test]
    async fn test_unavailable_tool_can_still_be_executed() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(UnavailableTool));

        // The tool is registered but not in definitions; execute() should still work.
        let result = registry.execute("unavailable_test", HashMap::new()).await;
        assert!(
            result.ok,
            "Unavailable tool should still execute when called directly: {:?}",
            result.error
        );
        assert_eq!(result.data, "executed");
    }

    #[test]
    fn test_unavailable_tool_excluded_from_local_definitions() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(MockTool::new("read_file")));
        registry.register(Box::new(MockTool::new("list_dir")));
        registry.register(Box::new(MockTool::new("exec")));
        registry.register(Box::new(UnavailableTool));

        let defs = registry.get_local_definitions();

        let names: Vec<String> = defs
            .iter()
            .filter_map(|d| d["function"]["name"].as_str().map(String::from))
            .collect();

        assert!(!names.contains(&"unavailable_test".to_string()));
    }

    // -----------------------------------------------------------------------
    // Integration tests: capability resolution -> tool availability gating
    // -----------------------------------------------------------------------

    /// Test 1: Capability-resolved tools are gated by is_available().
    ///
    /// Verifies that resolve_capabilities() produces the expected tool names and
    /// that the registry's get_definitions() correctly excludes unavailable tools
    /// even when both available and unavailable tools are registered.
    #[test]
    fn test_integration_capability_tools_gated_by_availability() {
        use crate::agent::capabilities::{resolve_capabilities, Capability};

        // Step 1: Resolve capabilities to tool names.
        let tool_names = resolve_capabilities(&[Capability::Read, Capability::Http]);

        // Step 2: Verify expected tools are in the resolved set.
        assert!(tool_names.contains(&"read_file".to_string()));
        assert!(tool_names.contains(&"web_search".to_string()));
        assert!(tool_names.contains(&"list_dir".to_string()));
        assert!(tool_names.contains(&"browser".to_string()));

        // Step 3: Build a registry where read_file is available but web_search is not
        // (simulating the case where the search backend key is missing).
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(MockTool::new("read_file")));
        registry.register(Box::new(UnavailableTool)); // stands in for unavailable web_search

        let defs = registry.get_definitions();
        let def_names: Vec<String> = defs
            .iter()
            .filter_map(|d| d["function"]["name"].as_str().map(String::from))
            .collect();

        // read_file (available) should appear; unavailable_test should not.
        assert!(def_names.contains(&"read_file".to_string()));
        assert!(!def_names.contains(&"unavailable_test".to_string()));
    }

    /// Test 3: Full chain — profile capabilities -> resolved tools -> no forbidden tools.
    ///
    /// Simulates what happens when a subagent profile declares Read + Memory + Skills
    /// capabilities: verifies the correct tool set is produced and forbidden tools
    /// (Execute, Write, Http) are excluded.
    #[test]
    fn test_integration_profile_capabilities_to_registry() {
        use crate::agent::capabilities::{resolve_capabilities, Capability};

        // Simulate a subagent profile that declares these capabilities.
        let profile_caps = vec![Capability::Read, Capability::Memory, Capability::Skills];
        let allowed_tools = resolve_capabilities(&profile_caps);

        // These tools should be allowed.
        assert!(allowed_tools.contains(&"read_file".to_string()));
        assert!(allowed_tools.contains(&"list_dir".to_string()));
        assert!(allowed_tools.contains(&"recall".to_string()));
        assert!(allowed_tools.contains(&"remember".to_string()));
        assert!(allowed_tools.contains(&"session_search".to_string()));
        assert!(allowed_tools.contains(&"read_skill".to_string()));

        // These should NOT be allowed (not in the declared capabilities).
        assert!(!allowed_tools.contains(&"exec".to_string()));
        assert!(!allowed_tools.contains(&"write_file".to_string()));
        assert!(!allowed_tools.contains(&"edit_file".to_string()));
        assert!(!allowed_tools.contains(&"web_search".to_string()));
        assert!(!allowed_tools.contains(&"browser".to_string()));
        assert!(!allowed_tools.contains(&"spawn".to_string()));
    }

    /// Test 5: CodeExecutionTool availability gating via is_available().
    ///
    /// Verifies that the Tool::is_available() contract works for CodeExecutionTool:
    /// disabled -> excluded from definitions, enabled -> included in definitions.
    #[test]
    fn test_integration_code_execution_availability() {
        use crate::agent::tools::CodeExecutionTool;

        // Disabled tool should not be available.
        let disabled = CodeExecutionTool::new(false, 30, 20, vec![], None);
        assert!(!disabled.is_available());

        // Enabled tool should be available.
        let enabled = CodeExecutionTool::new(true, 30, 20, vec![], None);
        assert!(enabled.is_available());

        // Registry gating: disabled execute_code must not appear in definitions.
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(MockTool::new("read_file")));
        registry.register(Box::new(CodeExecutionTool::new(
            false,
            30,
            20,
            vec![],
            None,
        )));

        let defs = registry.get_definitions();
        let def_names: Vec<String> = defs
            .iter()
            .filter_map(|d| d["function"]["name"].as_str().map(String::from))
            .collect();

        assert!(def_names.contains(&"read_file".to_string()));
        assert!(!def_names.contains(&"execute_code".to_string()));

        // Registry gating: enabled execute_code must appear in definitions.
        let mut registry2 = ToolRegistry::new();
        registry2.register(Box::new(CodeExecutionTool::new(true, 30, 20, vec![], None)));

        let defs2 = registry2.get_definitions();
        let def_names2: Vec<String> = defs2
            .iter()
            .filter_map(|d| d["function"]["name"].as_str().map(String::from))
            .collect();

        assert!(def_names2.contains(&"execute_code".to_string()));
    }

    #[test]
    fn test_definitions_for_filters_by_name() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(MockTool::new("alpha")));
        registry.register(Box::new(MockTool::new("beta")));
        registry.register(Box::new(MockTool::new("gamma")));

        let names = vec!["alpha".to_string(), "gamma".to_string()];
        let defs = registry.definitions_for(&names);
        let def_names: Vec<String> = defs
            .iter()
            .filter_map(|d| d["function"]["name"].as_str().map(String::from))
            .collect();
        assert_eq!(def_names.len(), 2);
        assert!(def_names.contains(&"alpha".to_string()));
        assert!(def_names.contains(&"gamma".to_string()));
        assert!(!def_names.contains(&"beta".to_string()));
    }

    #[test]
    fn test_definitions_for_missing_tool_skipped() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(MockTool::new("alpha")));

        let names = vec!["alpha".to_string(), "nonexistent".to_string()];
        let defs = registry.definitions_for(&names);
        assert_eq!(defs.len(), 1);
    }

    // -----------------------------------------------------------------------
    // Local tool selection: all registered tools visible
    // -----------------------------------------------------------------------

    /// Local models must see ALL registered+available tools — not a hardcoded
    /// subset. Registration is the source of truth; `is_available()` gates
    /// visibility; keyword triggers are irrelevant for local (everything shows).
    #[test]
    fn test_local_defs_include_all_registered_tools() {
        let mut registry = ToolRegistry::new();
        let all_tools = [
            "read_file",
            "write_file",
            "edit_file",
            "list_dir",
            "exec",
            "web_search",
            "web_fetch",
            "recall",
            "remember",
            "read_skill",
            "browser",
            "spawn",
        ];
        for name in &all_tools {
            registry.register(Box::new(MockTool::new(name)));
        }

        let defs = registry.get_local_definitions();
        let names: HashSet<String> = defs
            .iter()
            .filter_map(|d| d["function"]["name"].as_str().map(String::from))
            .collect();

        // Every registered tool must be present — no keyword gating.
        for tool in &all_tools {
            assert!(
                names.contains(*tool),
                "Local model must see '{}' — got: {:?}",
                tool,
                names,
            );
        }
    }

    /// Unavailable tools must still be excluded from local definitions.
    #[test]
    fn test_local_defs_exclude_unavailable() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(MockTool::new("read_file")));
        registry.register(Box::new(MockTool::new("web_search")));
        registry.register(Box::new(UnavailableTool)); // unavailable_test

        let defs = registry.get_local_definitions();
        let names: HashSet<String> = defs
            .iter()
            .filter_map(|d| d["function"]["name"].as_str().map(String::from))
            .collect();

        assert!(names.contains("read_file"));
        assert!(names.contains("web_search"));
        assert!(
            !names.contains("unavailable_test"),
            "Unavailable tools must be excluded from local definitions"
        );
    }

    /// Local definitions must have condensed (two-sentence) descriptions
    /// to save tokens without hiding tools.
    #[test]
    fn test_local_defs_are_condensed() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(MockTool::new("read_file")));

        let defs = registry.get_local_definitions();
        assert_eq!(defs.len(), 1);

        let desc = defs[0]["function"]["description"].as_str().unwrap();
        // MockTool description is "A mock tool for testing" (no period, no
        // second sentence) — condense_description returns it unchanged.
        // The important thing: condense IS applied (verified by the multi-
        // sentence test below).
        assert_eq!(desc, "A mock tool for testing");
    }

    /// Verify condensation truncates to two sentences (not one, not all).
    #[test]
    fn test_local_defs_condense_truncates_multi_sentence() {
        // Use a real tool that has a 3-sentence description.
        // WebSearchTool: "Search the web. Returns titles, URLs, and snippets. Use web_fetch..."
        let tool = super::WebSearchTool::new(
            None,
            5,
            "searxng".to_string(),
            "http://localhost:8888".to_string(),
        );
        let full_desc = tool.description().to_string();
        // Count sentences in full description (periods followed by space).
        let sentence_breaks = full_desc.matches(". ").count();
        assert!(
            sentence_breaks >= 2,
            "WebSearchTool should have 3+ sentences, got {} breaks in: {}",
            sentence_breaks,
            full_desc,
        );

        let mut registry = ToolRegistry::new();
        registry.register(Box::new(tool));

        let defs = registry.get_local_definitions();
        let condensed_desc = defs[0]["function"]["description"].as_str().unwrap();

        // Should keep two sentences but drop the third.
        assert!(condensed_desc.ends_with('.'), "Should end with period");
        assert!(
            condensed_desc.len() < full_desc.len(),
            "Condensed should be shorter than full: {} vs {}",
            condensed_desc.len(),
            full_desc.len(),
        );
        // Two sentences means exactly one ". " break in the condensed output.
        let condensed_breaks = condensed_desc.matches(". ").count();
        assert_eq!(
            condensed_breaks, 1,
            "Two-sentence condensation should have 1 internal break, got {}: {}",
            condensed_breaks, condensed_desc,
        );
    }

    // -------------------------------------------------------------------
    // Tool proxy tests
    // -------------------------------------------------------------------

    #[test]
    fn test_proxy_definition_single_schema() {
        let mut registry = ToolRegistry::new();
        for name in &["read_file", "list_dir", "exec"] {
            registry.register(Box::new(MockTool::new(name)));
        }

        let defs = registry.get_proxy_definition();
        assert_eq!(defs.len(), 1, "Proxy must return exactly 1 tool schema");
        assert_eq!(
            defs[0]["function"]["name"].as_str().unwrap(),
            "tool",
            "Proxy tool must be named 'tool'"
        );

        let props = &defs[0]["function"]["parameters"]["properties"];
        assert!(props.get("name").is_some(), "Must have 'name' param");
        assert!(props.get("args").is_some(), "Must have 'args' param");
    }

    #[test]
    fn test_proxy_definition_has_arg_hints() {
        let mut registry = ToolRegistry::new();
        // MockTool has required: ["value"]
        registry.register(Box::new(MockTool::new("read_file")));

        let defs = registry.get_proxy_definition();
        let desc = defs[0]["function"]["description"].as_str().unwrap();

        assert!(
            desc.contains("read_file(value)"),
            "Description should contain arg hint 'read_file(value)', got: {}",
            desc
        );
    }

    #[test]
    fn test_proxy_definition_excludes_unavailable() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(MockTool::new("read_file")));
        registry.register(Box::new(UnavailableTool));

        let defs = registry.get_proxy_definition();
        let desc = defs[0]["function"]["description"].as_str().unwrap();

        assert!(desc.contains("read_file"), "Available tool must be listed");
        assert!(
            !desc.contains("unavailable_test"),
            "Unavailable tool must NOT be listed"
        );
    }

    #[tokio::test]
    async fn test_proxy_inspect_returns_schema() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(MockTool::new("read_file")));

        let mut params = HashMap::new();
        params.insert("name".to_string(), serde_json::json!("read_file"));

        let result = registry.execute("tool", params).await;
        assert!(result.ok, "Inspect should succeed: {:?}", result.error);
        assert!(
            result.data.contains("parameters"),
            "Inspect result should contain parameters schema: {}",
            result.data
        );
        assert!(
            result.data.contains("value"),
            "Schema should mention required param 'value': {}",
            result.data
        );
    }

    #[tokio::test]
    async fn test_proxy_inspect_unknown_lists_available() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(MockTool::new("read_file")));

        let mut params = HashMap::new();
        params.insert("name".to_string(), serde_json::json!("nonexistent"));

        let result = registry.execute("tool", params).await;
        assert!(!result.ok, "Unknown tool inspect should fail");
        assert!(
            result.data.contains("read_file"),
            "Error should list available tools: {}",
            result.data
        );
    }

    #[tokio::test]
    async fn test_proxy_dispatch_executes_real_tool() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(MockTool::new("mock_tool")));

        let mut params = HashMap::new();
        params.insert("name".to_string(), serde_json::json!("mock_tool"));
        params.insert("args".to_string(), serde_json::json!({"value": "hello"}));

        let result = registry.execute("tool", params).await;
        assert!(
            result.ok,
            "Proxy dispatch should succeed: {:?}",
            result.error
        );
        assert!(
            result.data.contains("hello"),
            "Should contain dispatched tool output: {}",
            result.data
        );
    }

    #[tokio::test]
    async fn test_proxy_dispatch_runs_normalization() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(ParamEchoTool::new("read_file")));

        let mut params = HashMap::new();
        params.insert("name".to_string(), serde_json::json!("read_file"));
        // Use alias "file_path" which normalize_tool_request converts to "path"
        params.insert(
            "args".to_string(),
            serde_json::json!({"file_path": "/foo", "path": "/foo"}),
        );

        let result = registry.execute("tool", params).await;
        assert!(
            result.ok,
            "Dispatch with alias should succeed: {:?}",
            result.error
        );
        assert!(
            result.data.contains("path"),
            "Normalization should convert file_path to path: {}",
            result.data
        );
    }

    #[tokio::test]
    async fn test_proxy_missing_name_returns_error() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(MockTool::new("read_file")));

        let result = registry.execute("tool", HashMap::new()).await;
        assert!(!result.ok, "Missing name should fail");
        assert!(
            result.data.contains("Missing 'name'"),
            "Error should mention missing name: {}",
            result.data
        );
    }

    #[tokio::test]
    async fn test_proxy_intercept_via_execute() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(MockTool::new("list_dir")));

        // Call through the main execute() path with "tool" as the tool name
        let mut params = HashMap::new();
        params.insert("name".to_string(), serde_json::json!("list_dir"));
        params.insert("args".to_string(), serde_json::json!({"value": "test"}));

        let result = registry.execute("tool", params).await;
        assert!(
            result.ok,
            "Proxy intercept via execute() should work: {:?}",
            result.error
        );
        assert!(
            result.data.contains("test"),
            "Should dispatch to real tool: {}",
            result.data
        );
    }

    #[tokio::test]
    async fn test_direct_call_bypasses_proxy() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(MockTool::new("mock_tool")));

        // Call mock_tool directly, not through proxy
        let mut params = HashMap::new();
        params.insert("value".to_string(), serde_json::json!("direct"));

        let result = registry.execute("mock_tool", params).await;
        assert!(
            result.ok,
            "Direct call should still work: {:?}",
            result.error
        );
        assert!(
            result.data.contains("direct"),
            "Should get direct tool output: {}",
            result.data
        );
    }

    /// A mock tool with param descriptions for testing slim stripping.
    struct DescribedTool;
    #[async_trait]
    impl Tool for DescribedTool {
        fn name(&self) -> &str {
            "described_tool"
        }
        fn description(&self) -> &str {
            "A tool with described params."
        }
        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "The file path" },
                    "content": { "type": "string", "description": "The content to write" }
                },
                "required": ["path", "content"]
            })
        }
        async fn execute(&self, _params: HashMap<String, serde_json::Value>) -> String {
            "ok".into()
        }
    }

    #[test]
    fn test_slim_definitions_strip_param_descriptions() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(DescribedTool));

        let slim = registry.get_slim_definitions();
        let full = registry.get_local_definitions();

        assert_eq!(slim.len(), 1);
        assert_eq!(full.len(), 1);

        // Slim defs should have no "description" keys in parameter properties
        let slim_props = slim[0].pointer("/function/parameters/properties").unwrap();
        for (key, prop) in slim_props.as_object().unwrap() {
            assert!(
                prop.get("description").is_none(),
                "Slim def should strip description from param '{}': {:?}",
                key,
                prop
            );
            // But type should still be present
            assert!(
                prop.get("type").is_some(),
                "Slim def should keep type for param '{}'",
                key
            );
        }

        // Full defs should retain descriptions
        let full_props = full[0].pointer("/function/parameters/properties").unwrap();
        let has_desc = full_props
            .as_object()
            .unwrap()
            .values()
            .any(|v| v.get("description").is_some());
        assert!(has_desc, "Full defs should retain param descriptions");
    }

    // -----------------------------------------------------------------------
    // Permission enforcement tests
    // -----------------------------------------------------------------------

    struct ExecuteTool;

    #[async_trait]
    impl Tool for ExecuteTool {
        fn name(&self) -> &str {
            "exec_mock"
        }
        fn description(&self) -> &str {
            "mock execute-level tool"
        }
        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({"type": "object", "properties": {}})
        }
        fn permission(&self) -> PermissionLevel {
            PermissionLevel::Execute
        }
        async fn execute(&self, _params: HashMap<String, serde_json::Value>) -> String {
            "executed".to_string()
        }
    }

    #[tokio::test]
    async fn test_permission_denied_when_above_ceiling() {
        let mut registry = ToolRegistry::with_max_permission(PermissionLevel::ReadOnly);
        registry.register(Box::new(ExecuteTool));

        let result = registry.execute("exec_mock", HashMap::new()).await;
        assert!(!result.ok);
        assert!(result.data.contains("Permission denied"));
    }

    #[tokio::test]
    async fn test_permission_allowed_at_ceiling() {
        let mut registry = ToolRegistry::with_max_permission(PermissionLevel::Execute);
        registry.register(Box::new(ExecuteTool));

        let result = registry.execute("exec_mock", HashMap::new()).await;
        assert!(result.ok);
        assert_eq!(result.data, "executed");
    }

    #[tokio::test]
    async fn test_permission_allowed_above_ceiling() {
        let mut registry = ToolRegistry::with_max_permission(PermissionLevel::System);
        registry.register(Box::new(ExecuteTool));

        let result = registry.execute("exec_mock", HashMap::new()).await;
        assert!(result.ok);
    }

    #[test]
    fn test_set_max_permission() {
        let mut registry = ToolRegistry::new();
        assert_eq!(registry.max_permission, PermissionLevel::System);
        registry.set_max_permission(PermissionLevel::Write);
        assert_eq!(registry.max_permission, PermissionLevel::Write);
    }
}
