// Error-protocol layer-3 backlog (docs/research/2026-08-06-error-conventions-and-host-bridge.md §3.6):
// the deny regime in Cargo.toml is live; this module still carries pre-existing
// violations of the lints below. Remove this allow as the module migrates onto
// the regime.
#![allow(
    clippy::indexing_slicing,
    clippy::shadow_reuse,
)]
#![allow(dead_code)]
//! Tool registry for dynamic tool management.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde_json::Value;
use tracing::warn;

use super::base::{
    PermissionLevel, Tool, ToolConcurrency, ToolContext, ToolExecutionResult,
};
use super::filesystem::MAX_WRITE_FILE_PIECE_CHARS;
use super::{
    ApplyPatchTool, BrowserTool, CodeExecutionTool, EditFileTool, ExecTool, FileInfoTool,
    FilePreviewTool, FindFilesTool, ListDirTool, ReadFileTool, ReadSkillTool, RecallTool,
    RememberTool, SearchFilesTool, SystemInfoTool, ToolStatusTool, WebFetchTool, WebSearchTool,
    WorkspaceDiffTool, WriteFileTool,
};
use crate::config::schema::CodeExecutionConfig;
#[cfg(feature = "python-kernel")]
use crate::config::schema::PythonKernelConfig;

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
    /// When true, exclude file mutation tools.
    pub read_only: bool,
    /// If set, only register tools in this list. Empty = register all.
    pub tools_filter: Option<Vec<String>>,
    /// Optional override for exec working directory.
    pub exec_working_dir: Option<String>,
    /// Search backend: "searxng" (default) or "brave".
    pub search_provider: String,
    /// Base URL of the SearXNG instance (default: "http://localhost:8888").
    pub searxng_url: String,
    /// Base URL of a local crw-server for web_fetch; empty = disabled.
    pub crw_url: String,
    /// Maximum search results to return (default: 5).
    pub search_max_results: u32,
    /// Path to the SQLite sessions database for session_search tool.
    /// When `None`, the session_search tool is not registered.
    pub db_path: Option<PathBuf>,
    /// Code execution tool config. Disabled by default.
    pub code_execution: CodeExecutionConfig,
    /// Stateful Python kernel tool config. Feature: `python-kernel`.
    #[cfg(feature = "python-kernel")]
    pub python_kernel: PythonKernelConfig,
    /// Optional health-registry handle. When set, the web_search tool checks
    /// the "searxng" probe before calling SearXNG and returns a clear
    /// "degraded, restart the container" message instead of silent zero results.
    pub health_registry: Option<Arc<crate::heartbeat::health::HealthRegistry>>,
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
            crw_url: String::new(),
            search_max_results: 5,
            db_path: None,
            code_execution: CodeExecutionConfig::default(),
            #[cfg(feature = "python-kernel")]
            python_kernel: PythonKernelConfig::default(),
            health_registry: None,
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
    /// Typed host bridge injected at the registry boundary (research §3.5).
    /// `None` in sandboxed registries (scripts, tests) — tools that require
    /// the host fail with a typed `ToolError::Execution`.
    host: Option<Arc<dyn crate::agent::host_bridge::HostBridge>>,
}

/// Fully decoded meaning of the compact `tool` proxy envelope.
///
/// This is the only place that knows the current and legacy wire keys. Both
/// routing and execution project from this value, so a call cannot execute as
/// one tool while being guarded, cached, or persisted as another.
enum ProxyCall {
    Catalog,
    Inspect {
        tool_name: String,
    },
    Dispatch {
        tool_name: String,
        arguments: HashMap<String, Value>,
    },
    MissingSelector,
    InvalidArguments,
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
            // Dissolution: session_search and search_context collapse into
            // recall. Which recall path runs is decided by the params present
            // (query -> search; session/message_ids/mode=latest -> fetch), so
            // a plain name rewrite here is sufficient.
            "session_search" | "search_context" => "recall",
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
            "read_file" | "write_file" | "edit_file" | "file_info" | "workspace_diff" => {
                Self::normalize_param_aliases(&mut params, "path", file_aliases);
                if canonical_name != "workspace_diff" {
                    Self::require_non_empty_string(&params, "path", canonical_name)?;
                }
            }
            "list_dir" | "find_files" | "search_files" => {
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
            host: None,
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
            host: None,
        }
    }

    /// Set the maximum permission level for this registry.
    pub fn set_max_permission(&mut self, max: PermissionLevel) {
        self.max_permission = max;
    }

    /// Inject the typed host bridge (spawn/pipeline/loop/message). Builder
    /// style so the registry stays immutable after construction; the single
    /// production injection point is `tool_wiring::build_tools`.
    pub fn with_host(mut self, host: Option<Arc<dyn crate::agent::host_bridge::HostBridge>>) -> Self {
        self.host = host;
        self
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
            if config.read_only && matches!(name, "write_file" | "edit_file" | "apply_patch") {
                return false;
            }
            if let Some(ref filter) = config.tools_filter {
                return filter.iter().any(|t| t == name);
            }
            true
        };

        if should_include("read_file") {
            self.register(Box::new(ReadFileTool::new(config.max_tool_result_chars)));
        }
        if should_include("file_preview") {
            self.register(Box::new(FilePreviewTool));
        }
        if should_include("write_file") {
            self.register(Box::new(WriteFileTool::default()));
        }
        if should_include("edit_file") {
            self.register(Box::new(EditFileTool));
        }
        if should_include("apply_patch") {
            self.register(Box::new(ApplyPatchTool));
        }
        if should_include("list_dir") {
            self.register(Box::new(ListDirTool));
        }
        if should_include("find_files") {
            self.register(Box::new(FindFilesTool));
        }
        if should_include("search_files") {
            self.register(Box::new(SearchFilesTool));
        }
        if should_include("file_info") {
            self.register(Box::new(FileInfoTool));
        }
        if should_include("workspace_diff") {
            self.register(Box::new(WorkspaceDiffTool));
        }
        if should_include("system_info") {
            self.register(Box::new(SystemInfoTool));
        }
        if should_include("tool_status") {
            self.register(Box::new(ToolStatusTool::new(config.workspace.clone())));
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
            self.register(Box::new(
                WebSearchTool::new(
                    config.brave_api_key.clone(),
                    config.search_max_results,
                    config.search_provider.clone(),
                    config.searxng_url.clone(),
                )
                .with_health_registry(config.health_registry.clone()),
            ));
        }
        if should_include("web_fetch") {
            self.register(Box::new(
                WebFetchTool::new(config.max_tool_result_chars).with_crw(config.crw_url.clone()),
            ));
        }
        if should_include("browser") {
            self.register(Box::new(BrowserTool::new(config.max_tool_result_chars)));
        }
        if should_include("recall") {
            // recall is the unified retrieval tool: it absorbs the dissolved
            // session_search + search_context. Attach the sessions database so
            // the session fetch/search legs and the trust-ranked merge work.
            let mut tool = RecallTool::new(&config.workspace);
            if let Some(ref db_path) = config.db_path {
                tool = tool.with_db(db_path.clone());
            }
            self.register(Box::new(tool));
        }
        if should_include("remember") {
            self.register(Box::new(RememberTool::new(config.workspace.clone())));
        }
        if should_include("get_skills") {
            self.register(Box::new(ReadSkillTool::new(&config.workspace)));
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
        #[cfg(feature = "python-kernel")]
        if config.python_kernel.enabled && should_include("python") {
            self.register(Box::new(crate::agent::tools::PythonKernel::new(
                config.python_kernel.timeout,
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

    /// Return the implementation-declared execution policy. Hooks may mutate
    /// external state, so their presence conservatively serializes every tool.
    pub fn concurrency(&self, name: &str) -> ToolConcurrency {
        if self.hooks.is_some() {
            return ToolConcurrency::Sequential;
        }
        self.get(name)
            .map(Tool::concurrency)
            .unwrap_or(ToolConcurrency::Sequential)
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
        // Proxy intercept: "get_tools" (alias "tool") is the meta-tool, not a registered tool.
        if name == "get_tools" || name == "tool" {
            return self.execute_proxy(params, None).await;
        }
        self.execute_inner(name, params, None).await
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
            Some((result.data(), result.ok())),
        )
        .await;
    }

    /// Execute a tool by name with a [`ToolContext`] for progress
    /// reporting and cancellation support.
    ///
    /// Same as [`Self::execute`] but passes the context through to the tool.
    pub async fn execute_with_context(
        &self,
        name: &str,
        params: HashMap<String, serde_json::Value>,
        ctx: &ToolContext,
    ) -> ToolExecutionResult {
        // Proxy intercept: "get_tools" (alias "tool") is the meta-tool, not a registered tool.
        if name == "get_tools" || name == "tool" {
            return self.execute_proxy(params, Some(ctx)).await;
        }
        self.execute_inner(name, params, Some(ctx)).await
    }

    /// Core execute logic (no proxy intercept). Called by both `execute()`
    /// variants and `execute_proxy()` dispatch mode. `ctx` enables progress
    /// reporting and cancellation when the caller has one.
    async fn execute_inner(
        &self,
        name: &str,
        params: HashMap<String, serde_json::Value>,
        ctx: Option<&ToolContext>,
    ) -> ToolExecutionResult {
        let (name, params) = match Self::normalize_tool_request(name, params) {
            Ok(v) => v,
            Err(e) => return ToolExecutionResult::failure(e),
        };
        // The registry is the single ToolContext builder (research §3.5):
        // the host is injected here, once, at the boundary. A caller-supplied
        // context (streaming tools) keeps its event/cancel/call_id parts;
        // sandboxed callers get a fresh default context.
        let ctx = match ctx {
            Some(c) => c.clone().with_host(self.host.clone()),
            None => {
                let (event_tx, _rx) = tokio::sync::mpsc::unbounded_channel();
                ToolContext::new(
                    self.host.clone(),
                    event_tx,
                    tokio_util::sync::CancellationToken::new(),
                    String::new(),
                )
            }
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

        let unwound = std::panic::AssertUnwindSafe(
            tool.execute_with_result_and_context(params.clone(), &ctx),
        );
        let unwound = futures_util::FutureExt::catch_unwind(unwound).await;
        let mut result = match unwound {
            Ok(result) => result,
            Err(_) => {
                ToolExecutionResult::failure(format!("Tool '{}' panicked during execution", name))
            }
        };

        // When a tool rejects a call for a missing required arg, append the
        // exact call shape derived from its schema. This is the moment a weak
        // (temp 0) model pays attention; a passive "Error: 'query' is
        // required" does not teach the shape, so the model retries identically
        // until the dedup guard terminates the turn. Generic across every tool
        // — reads the schema, no per-tool hardcoding.
        // Structural path: a tool that set `error_kind = MissingArg` carries
        // the canonical example directly — no string matching. Back-compat:
        // legacy tools whose error string still contains "is required" (e.g.
        // recall's "'query' parameter is required") keep getting a
        // schema-derived example until they migrate to the structured path.
        if !result.ok() {
            let example = match result.error_kind() {
                Some(crate::errors::ToolErrorKind::MissingArg { example, .. }) => {
                    Some(example.clone())
                }
                _ if result.data().contains("is required") => {
                    Self::worked_example_call(&name, &tool.parameters())
                }
                _ => None,
            };
            if let Some(example) = example {
                // Migrated tools render the example into the wire string
                // already (`ToolError::MissingArg::render` ends with
                // "call as {example}"); appending again would double it.
                if !result.data().contains(example.as_str()) {
                    result.append_worked_example(&example);
                }
            }
        }

        self.run_post_hook(&name, &params, &result).await;
        result
    }

    /// Core tools that are always included in tool definitions.
    const CORE_TOOLS: &'static [&'static str] = &[
        "read_file",
        "write_file",
        "edit_file",
        "list_dir",
        "search_files",
        "exec",
        "spawn",
    ];

    /// Extra tools included (when registered) in the Lean production surface,
    /// on top of `CORE_TOOLS`. Everything else is reachable via the proxy
    /// meta-tool appended by `get_lean_definitions`.
    const LEAN_EXTRA_TOOLS: &'static [&'static str] = &[
        "get_skills",
        "web_search",
        "web_fetch",
        "message",
        "recall_tool_result",
        // Bounded retrieval over stashed results (search/slice without loading
        // the full body). See stash_search.rs.
        "search_tool_result",
        "slice_tool_result",
        "todo",
        // The system prompt instructs the model to expand LCM summaries; the
        // schema must be advertised or local models fall back to guessing
        // (observed: recall_tool_result with invented ids).
        "lcm_expand",
        // Memory tools: medium local models (e.g. Qwen3.6-35B-A3M) fail to route
        // through the `tool` proxy meta-tool reliably, so they get dedicated
        // slim schemas instead of proxy-only reachability.
        "recall",
        "remember",
    ];

    /// Hot tools advertised as native schemas at turn 1. Kept to the 5 the
    /// model uses every turn — the rest go through the `tool` proxy to keep
    /// the tool-schema prefix small (~670 tok vs ~2000 tok for 14 native).
    /// `get_skills` is native (not proxied) so the model reads skill content
    /// directly instead of hitting the proxy's inspect mode.
    /// See commit f03c6e8 for the prior pure-proxy attempt; this is the
    /// middle ground that avoids the proxy/native arg confusion that killed
    /// pure-proxy while still cutting cold-start prefill by ~60%.
    const CORE_NATIVE_TOOLS: &'static [&'static str] = &[
        "read_file",
        "edit_file",
        "write_file",
        "exec",
        "get_skills",
    ];

    /// Tools kept registered and reachable via the `get_tools` proxy (call with
    /// no `tool_name` to list) but omitted from the per-turn prompt catalog to
    /// save cold-prefill tokens. Empirically 0 invocations across 17 days of
    /// sessions and peripheral to the hot path. `remember` and `lcm_expand` are
    /// intentionally NOT here (core to memory formation / LCM expansion).
    ///
    /// `recall_tool_result` is advertised because `TOOL_RESULT_HANDLE v1`
    /// receipts point the model at it (`fetch:"recall_tool_result"`); an
    /// unadvertised target made local models substitute `recall` and burn turns
    /// failing to resolve handles (session 20260804_204406_c16eb0).
    const RARELY_ADVERTISED_TOOLS: &'static [&'static str] = &[
        "file_preview",
        "file_info",
        "workspace_diff",
        "system_info",
        "tool_status",
        "browser",
        "search_tool_result",
        "slice_tool_result",
    ];

    /// Internal Lean-catalog builder: condense every available schema before
    /// selecting the fixed production subset. `pub(crate)` so the delegation
    /// (sub-agent) path can also send condensed descriptions without losing
    /// any tool (unlike `get_lean_definitions`, which is a fixed subset).
    pub(crate) fn get_local_definitions(&self) -> Vec<serde_json::Value> {
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
    fn get_slim_definitions(&self) -> Vec<serde_json::Value> {
        // Tools whose parameter semantics are load-bearing and must survive
        // slimming. read_file's `lines` paging syntax is the prime case: strip
        // it and the local model can't page large files and re-prefills the
        // whole file each turn.
        // lcm_expand's param teaches the copyable range-string form ("120-158")
        // — strip it and small models invent shapes.
        const KEEP_PARAM_DESCRIPTIONS: &[&str] =
            &["read_file", "write_file", "edit_file", "lcm_expand"];
        let mut defs = self.get_local_definitions();
        for def in &mut defs {
            Self::remove_local_hot_model_hazards(def);
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

    /// Lean production surface: a SINGLE meta-tool (`tool`) advertised at turn 1.
    ///
    /// Lazy-load contract — the proxy unlocks every registered tool on demand:
    /// `tool("name")` returns the tool's full parameter schema, `tool("name", args)`
    /// executes it. No per-tool schemas live in the prompt head, so the
    /// cold-prefill tool block is one small definition (~150 tokens) instead of
    /// the ~1.4k of individual slim schemas. The model pulls a tool's schema
    /// only when it actually needs it, and because inspect returns the FULL
    /// (non-slimmed) schema, it forms correct arguments instead of guessing.
    pub fn get_lean_definitions(&self) -> Vec<serde_json::Value> {
        self.get_proxy_definition()
    }

    /// Core-plus-proxy surface: hot local tools as native schemas at turn 1,
    /// plus the `tool` proxy meta-tool for everything else. The model no longer
    /// pays an inspect round-trip for the tools it uses every turn, while the
    /// proxy keeps the long tail executable on demand (no tool lost).
    pub fn get_core_plus_proxy_definitions(&self) -> Vec<serde_json::Value> {
        self.get_core_plus_proxy_definitions_for(Self::CORE_NATIVE_TOOLS, Self::CORE_NATIVE_TOOLS)
    }

    /// Artifact surface for local models.
    ///
    /// Keep this byte-identical to the normal core-plus-proxy surface. The chat
    /// template renders tool schemas at the prompt head, so switching tool
    /// catalogs mid-session busts the retained local KV cache.
    pub fn get_artifact_core_plus_proxy_definitions(&self) -> Vec<serde_json::Value> {
        self.get_core_plus_proxy_definitions()
    }

    fn get_core_plus_proxy_definitions_for(
        &self,
        native_tools: &[&'static str],
        proxy_excludes: &[&'static str],
    ) -> Vec<serde_json::Value> {
        let mut defs: Vec<serde_json::Value> = native_tools
            .iter()
            .filter_map(|name| {
                self.tools
                    .get(*name)
                    .filter(|t| t.is_available())
                    .map(|t| t.to_schema())
            })
            .collect();
        Self::condense_definitions(&mut defs);
        // Slim parameter descriptions (pi-style leanness); keep the discovery
        // hooks' param descriptions so the model knows how to call them without
        // a proxy round-trip.
        const KEEP_PARAM_DESCRIPTIONS: &[&str] = &[
            "read_file",
            "get_skills",
            "recall",
            "remember",
        ];
        for def in &mut defs {
            Self::remove_local_hot_model_hazards(def);
            let name = def
                .pointer("/function/name")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if KEEP_PARAM_DESCRIPTIONS.contains(&name) {
                continue;
            }
            if let Some(params) = def.pointer_mut("/function/parameters/properties") {
                if let Some(props) = params.as_object_mut() {
                    for (_k, prop) in props.iter_mut() {
                        if let Some(obj) = prop.as_object_mut() {
                            obj.remove("description");
                        }
                    }
                }
            }
        }
        Self::sort_definitions(&mut defs);
        defs.extend(self.get_proxy_definition_excluding(proxy_excludes));
        defs
    }

    fn remove_local_hot_model_hazards(def: &mut serde_json::Value) {
        let name = def
            .pointer("/function/name")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        match name {
            "edit_file" => {
                if let Some(props) = def
                    .pointer_mut("/function/parameters/properties")
                    .and_then(|v| v.as_object_mut())
                {
                    // Optional hashes/patches stay executable, but local hot
                    // schemas should not invite fabricated guards. Inspect the
                    // full schema through `tool("edit_file")` when needed.
                    props.remove("expected_sha256");
                    props.remove("patch");
                }
                if let Some(required) = def.pointer_mut("/function/parameters/required") {
                    *required = serde_json::json!(["path", "old_text", "new_text"]);
                }
            }
            _ => {}
        }
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

    fn flattened_proxy_key_allowed(tool_name: &str, key: &str, tool: &dyn Tool) -> bool {
        let params = tool.parameters();
        let in_schema = params
            .get("properties")
            .and_then(|v| v.as_object())
            .is_some_and(|props| props.contains_key(key));
        if in_schema {
            return true;
        }

        match tool_name {
            "read_file" | "write_file" | "edit_file" | "file_info" | "workspace_diff" => {
                matches!(key, "file_path" | "filepath" | "file")
            }
            "list_dir" | "find_files" | "search_files" => {
                matches!(key, "dir_path" | "directory" | "dir")
            }
            "web_search" => matches!(key, "q" | "search_query"),
            "spawn" => matches!(key, "id"),
            _ => false,
        }
    }

    fn flattened_proxy_args_for_dispatch(
        &self,
        tool_name: &str,
        extras: serde_json::Map<String, Value>,
    ) -> Option<Value> {
        if extras.is_empty() {
            return None;
        }
        let tool = self.tools.get(tool_name)?;
        if extras
            .keys()
            .all(|key| Self::flattened_proxy_key_allowed(tool_name, key, tool.as_ref()))
        {
            Some(Value::Object(extras))
        } else {
            None
        }
    }

    /// Build a single compact proxy schema that lists all available tools.
    ///
    /// Returns one tool definition called `"get_tools"` whose description embeds
    /// the full catalog with arg hints. The model calls `tool(name: "X")`
    /// to inspect a tool's schema, or `tool(name: "X", args: {...})` to execute.
    ///
    /// Token cost: ~90 tokens vs ~2045 for 15 individual schemas.
    fn get_proxy_definition(&self) -> Vec<serde_json::Value> {
        self.get_proxy_definition_excluding(&[])
    }

    /// Like [`get_proxy_definition`], but omits `exclude` tool names from the
    /// advertised catalog. Used by `get_core_plus_proxy_definitions` so the
    /// four native core tools aren't double-listed (once as native schemas,
    /// once in the proxy catalog).
    fn get_proxy_definition_excluding(&self, exclude: &[&str]) -> Vec<serde_json::Value> {
        // Rarely-used tools stay registered and reachable via `get_tools({})`
        // (omit tool_name to list) but are not enumerated in the per-turn prompt
        // catalog, to save cold-prefill tokens. See `RARELY_ADVERTISED_TOOLS`.
        let mut hints: Vec<String> = self
            .tools
            .values()
            .filter(|t| t.is_available())
            .map(|t| Self::tool_hint(t.as_ref()))
            .filter(|h| !exclude.iter().any(|e| h.starts_with(e)))
            .filter(|h| !Self::RARELY_ADVERTISED_TOOLS.iter().any(|r| h.starts_with(r)))
            .collect();
        hints.sort();

        let description = format!(
            "Gateway to all tools. Omit tool_name to list every available tool. \
             Provide {{\"tool_name\":\"NAME\"}} to inspect that tool's full parameter \
             schema, or {{\"tool_name\":\"NAME\",\"tool_args\":{{\"arg\":\"value\"}}}} to invoke it. \
             Starter tools: read_file, get_skills (lists SKILLS, not tools), recall \
             (memory search), remember, todo (plan multi-step artifact work), edit_file, \
             exec, write_file. A complete write_file call may contain the whole file. \
             For voluntary staged writes, keep state=more pieces to 4096 characters or \
             less; state=complete publishes the final piece. After publication, validate \
             the artifact before claiming completion. \
             Full catalog: {}.",
            hints.join(", ")
        );

        vec![serde_json::json!({
            "type": "function",
            "function": {
                "name": "get_tools",
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        // The proxy selector and argument envelope are named
                        // `tool_name`/`tool_args` rather than `name`/`args`.
                        // The old names overloaded with inner-tool params
                        // (e.g. `get_skills` has its own `name`), causing
                        // small local models to collapse everything into
                        // `args` and omit the top-level selector. Live
                        // failure 2026-07-27 17:21 (session
                        // 20260727_173522_263450) showed this 3× in a row.
                        // Back-compat: the dispatcher still accepts the
                        // legacy `name`/`args` keys for in-flight sessions.
                        "tool_name": { "type": "string", "description": "Name of the tool to invoke (omit to list all available tools)" },
                        "tool_args": { "type": "object", "description": "Arguments object for the tool named above (omit to inspect its schema)" }
                    }
                }
            }
        })]
    }

    /// Result for a proxy call that carries no usable tool name.
    ///
    /// A bare `tool({})` is the documented discovery path and must stay a
    /// success — failing it made models think the proxy was broken. Anything
    /// else means the model forgot `name`; answering that with the catalog
    /// reads as "it worked", so the model loops on the same mistake instead
    /// of correcting it. Carrying `args` counts as intent just as much as a
    /// flattened `url` does.
    fn missing_name_result(
        &self,
        params: &HashMap<String, serde_json::Value>,
    ) -> ToolExecutionResult {
        // Only `tool_name`/`name` (the selector) is filtered out as a pure
        // control key. `tool_args`/`args` is intentionally KEPT — its
        // presence means the model was trying to invoke a tool (intent),
        // not just discover them. Answering "args without selector" with
        // the success-catalog reads as "it worked", so the model loops on
        // the same malformed call instead of correcting it.
        let mut sent = Self::proxy_intent_keys(params);
        let names = self.available_tool_names();
        if sent.is_empty() {
            // The catalog result is the feedback a weak model actually reads.
            // A bare name list gives no way to reach schemas, so the model
            // loops calling get_tools({}) identically until the dedup guard
            // terminates the turn. State the inspect path in the result itself.
            return ToolExecutionResult::success(format!(
                "Available tools: {}. Pass {{\"tool_name\":\"NAME\"}} to inspect NAME's \
                 full parameter schema.",
                names.join(", ")
            ));
        }
        sent.sort_unstable();
        warn!(
            params = %serde_json::to_string(params).unwrap_or_default(),
            "proxy_call_missing_name"
        );
        ToolExecutionResult::failure(format!(
            "Error: 'tool_name' is required. You sent parameters {{{}}} with no tool selector. \
             Call tool with {{\"tool_name\":\"TOOLNAME\",\"tool_args\":{{...}}}} — omit tool_name only to list tools. \
             Available: {}",
            sent.join(", "),
            names.join(", ")
        ))
    }

    /// A parameter that carries no information: null, empty string, empty
    /// object, or empty array. Models emit these as filler around a call they
    /// meant to make, so they must not be mistaken for real intent.
    fn is_blank_param(v: &serde_json::Value) -> bool {
        match v {
            serde_json::Value::Null => true,
            serde_json::Value::String(s) => s.trim().is_empty(),
            serde_json::Value::Object(m) => m.is_empty(),
            serde_json::Value::Array(a) => a.is_empty(),
            _ => false,
        }
    }

    fn proxy_intent_keys(params: &HashMap<String, Value>) -> Vec<&str> {
        params
            .iter()
            .filter(|(key, value)| {
                !matches!(key.as_str(), "tool_name" | "name") && !Self::is_blank_param(value)
            })
            .map(|(key, _)| key.as_str())
            .collect()
    }

    /// Build a worked-example call shape from a tool's parameter schema, for
    /// appending to missing-required-arg error messages. Returns `None` only
    /// when the tool genuinely takes no parameters. Declared `required` params
    /// are used when present; otherwise the first property is used, so
    /// mode-defaulted tools (e.g. `session_search`, whose `required` array is
    /// empty because the default mode doesn't need `query`) still surface a
    /// useful example at the moment the model is paying attention.
    fn worked_example_call(name: &str, schema_params: &serde_json::Value) -> Option<String> {
        let props = schema_params.get("properties")?.as_object()?;
        if props.is_empty() {
            return None;
        }
        let required: Vec<&str> = schema_params
            .get("required")
            .and_then(|r| r.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
            .unwrap_or_default();
        let keys: Vec<&str> = if !required.is_empty() {
            required
        } else {
            vec![props.keys().next()?.as_str()]
        };
        let inner = keys
            .iter()
            .map(|k| format!("\"{k}\":\"...\""))
            .collect::<Vec<_>>()
            .join(", ");
        Some(format!("{name}({{{inner}}})"))
    }

    /// Decode the compact proxy envelope once at its wire boundary.
    ///
    /// Current keys win when both forms are present. A JSON string containing
    /// an object is accepted because local models sometimes double-encode the
    /// argument envelope. Flattened arguments are dispatched only when every
    /// key belongs to the selected tool; otherwise the call remains an inspect.
    fn resolve_proxy_call(&self, params: &HashMap<String, Value>) -> ProxyCall {
        let tool_name = params
            .get("tool_name")
            .or_else(|| params.get("name"))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|name| !name.is_empty())
            .map(str::to_string);
        let Some(tool_name) = tool_name else {
            return if Self::proxy_intent_keys(params).is_empty() {
                ProxyCall::Catalog
            } else {
                ProxyCall::MissingSelector
            };
        };

        let envelope = params.get("tool_args").or_else(|| params.get("args"));
        match envelope {
            Some(Value::Object(map)) => {
                return ProxyCall::Dispatch {
                    tool_name,
                    arguments: map
                        .iter()
                        .map(|(key, value)| (key.clone(), value.clone()))
                        .collect(),
                };
            }
            Some(Value::String(encoded)) => {
                return match serde_json::from_str::<Value>(encoded) {
                    Ok(Value::Object(map)) => ProxyCall::Dispatch {
                        tool_name,
                        arguments: map.into_iter().collect(),
                    },
                    _ => ProxyCall::InvalidArguments,
                };
            }
            Some(Value::Null) | None => {}
            Some(_) => return ProxyCall::InvalidArguments,
        }

        let extras: serde_json::Map<String, Value> = params
            .iter()
            .filter(|(key, _)| !matches!(key.as_str(), "tool_name" | "tool_args" | "name" | "args"))
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect();
        match self.flattened_proxy_args_for_dispatch(&tool_name, extras) {
            Some(Value::Object(map)) => ProxyCall::Dispatch {
                tool_name,
                arguments: map.into_iter().collect(),
            },
            _ => ProxyCall::Inspect { tool_name },
        }
    }

    /// Project a tool call into its semantic execution identity.
    ///
    /// `None` means a proxy catalog, inspect, or malformed call and instructs
    /// routing to preserve the outer `tool` call so execution can return its
    /// normal catalog/schema/error response.
    pub(crate) fn canonical_proxy_dispatch(
        &self,
        outer_name: &str,
        params: &HashMap<String, Value>,
    ) -> Option<(String, HashMap<String, Value>)> {
        if outer_name != "get_tools" && outer_name != "tool" {
            return Some((outer_name.to_string(), params.clone()));
        }
        match self.resolve_proxy_call(params) {
            ProxyCall::Dispatch {
                tool_name,
                arguments,
            } => Some((tool_name, arguments)),
            _ => None,
        }
    }

    /// Handle a proxy call: inspect (no args) or dispatch (with args).
    /// `ctx` is passed through to the dispatched tool when present.
    async fn execute_proxy(
        &self,
        params: HashMap<String, serde_json::Value>,
        ctx: Option<&ToolContext>,
    ) -> ToolExecutionResult {
        match self.resolve_proxy_call(&params) {
            ProxyCall::Catalog | ProxyCall::MissingSelector => self.missing_name_result(&params),
            ProxyCall::InvalidArguments => {
                ToolExecutionResult::failure("'args' must be a JSON object".to_string())
            }
            ProxyCall::Inspect { tool_name } => {
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
            ProxyCall::Dispatch {
                tool_name,
                arguments: inner_params,
            } => {
                // Dispatch mode: extract args and call the real tool.
                // Call execute_inner directly to avoid recursion through execute().
                if tool_name == "write_file"
                    && inner_params
                        .get("state")
                        .and_then(Value::as_str)
                        .is_some_and(|state| state.trim().eq_ignore_ascii_case("more"))
                    && inner_params
                        .get("content")
                        .and_then(Value::as_str)
                        .is_some_and(|content| content.chars().count() > MAX_WRITE_FILE_PIECE_CHARS)
                {
                    return ToolExecutionResult::failure(format!(
                        "write_file state=\"more\" content exceeds the {}-character staged-piece limit; retry with 4096 characters or less, or send the full artifact with state=\"complete\"",
                        MAX_WRITE_FILE_PIECE_CHARS
                    ));
                }
                self.execute_inner(&tool_name, inner_params, ctx).await
            }
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

    fn register_test_result_recall(registry: &mut ToolRegistry, db_path: std::path::PathBuf) {
        registry.register(Box::new(
            crate::agent::tools::recall_tool_result::RecallToolResultTool::with_db(
                db_path.clone(),
                "test-session".to_string(),
            ),
        ));
        registry.register(Box::new(
            crate::agent::tools::stash_search::SearchToolResultTool::with_db(
                db_path.clone(),
                "test-session".to_string(),
            ),
        ));
        registry.register(Box::new(
            crate::agent::tools::stash_search::SliceToolResultTool::with_db(
                db_path,
                "test-session".to_string(),
            ),
        ));
    }

    /// Wire-cost budget for the local tool surface. The tool block is
    /// rendered into the prompt HEAD by the chat template, so its size is
    /// paid on every cold prefill and its hash must stay stable for the
    /// prefix cache. Run with --nocapture to see the actual numbers.
    #[test]
    fn test_local_tool_surface_token_budget() {
        use crate::agent::token_budget::TokenBudget;
        let ws = tempfile::tempdir().unwrap();
        let mut config = ToolConfig::new(ws.path());
        config.db_path = Some(ws.path().join("sessions.db"));
        let mut reg = ToolRegistry::with_standard_tools(&config);
        register_test_result_recall(&mut reg, ws.path().join("sessions.db"));
        let count = reg.get_local_definitions().len();
        let full = TokenBudget::estimate_tool_def_tokens(&reg.get_definitions());
        let local = TokenBudget::estimate_tool_def_tokens(&reg.get_local_definitions());
        let slim = TokenBudget::estimate_tool_def_tokens(&reg.get_slim_definitions());
        let lean = TokenBudget::estimate_tool_def_tokens(&reg.get_lean_definitions());
        let proxy = TokenBudget::estimate_tool_def_tokens(&reg.get_proxy_definition());
        let core_proxy =
            TokenBudget::estimate_tool_def_tokens(&reg.get_core_plus_proxy_definitions());
        println!(
            "tool surface: count={count} full={full} local={local} slim={slim} lean={lean} proxy={proxy} core_proxy={core_proxy}"
        );
        // Lean surface is now a SINGLE proxy definition (~150 tokens), paid on
        // every cold prefill. This is the lazy-load contract: one tool at turn 1.
        assert_eq!(
            lean, proxy,
            "lean surface must be exactly the proxy definition (proxy-only lean)"
        );
        assert!(
            lean <= 400,
            "lean tool defs (the local default) ballooned to {lean} tokens (budget 400) — \
             every token here is cold-prefill cost on the local path"
        );
        assert!(
            slim <= 2500,
            "slim tool defs ballooned to {slim} tokens (budget 2500) across {count} tools"
        );
        assert!(
            core_proxy <= 1900,
            "core+proxy surface (cloud path) ballooned to {core_proxy} tokens (budget 1900)"
        );
        assert!(
            lean <= 400,
            "lean surface (local path) ballooned to {lean} tokens (budget 400) — \
             every token here is cold-prefill cost on the local path"
        );
    }

    /// Core-plus-proxy surface: 14 hot native schemas + the `tool` proxy for
    /// the CLOUD path. Local uses lean (proxy-only) via select_tool_definitions.
    #[test]
    fn test_core_plus_proxy_surface() {
        let ws = tempfile::tempdir().unwrap();
        let mut config = ToolConfig::new(ws.path());
        config.db_path = Some(ws.path().join("sessions.db"));
        let mut reg = ToolRegistry::with_standard_tools(&config);
        register_test_result_recall(&mut reg, ws.path().join("sessions.db"));
        reg.register(Box::new(crate::agent::tools::TodoTool::new(ws.path())));
        let defs = reg.get_core_plus_proxy_definitions();
        let names: Vec<&str> = defs
            .iter()
            .filter_map(|d| d.pointer("/function/name").and_then(|v| v.as_str()))
            .collect();
        // 5 hot native tools + 1 proxy = 6 total.
        assert_eq!(
            names.len(),
            6,
            "core+proxy must be 5 native + 1 proxy, got {names:?}"
        );
        assert!(names.contains(&"get_tools"), "missing proxy: {names:?}");
        for expected in [
            "read_file",
            "edit_file",
            "write_file",
            "exec",
            "get_skills",
        ] {
            assert!(
                names.contains(&expected),
                "core missing {expected}: {names:?}"
            );
        }
        let write_content = defs
            .iter()
            .find(|d| d.pointer("/function/name").and_then(|v| v.as_str()) == Some("write_file"))
            .and_then(|d| d.pointer("/function/parameters/properties/content"))
            .expect("write_file content schema");
        assert!(
            write_content.get("maxLength").is_none(),
            "complete writes must not inherit the voluntary staged-piece limit"
        );
        // The proxy must teach all three modes and name starter tools.
        let proxy_desc = defs
            .iter()
            .find(|d| d.pointer("/function/name").and_then(|v| v.as_str()) == Some("get_tools"))
            .and_then(|d| d.pointer("/function/description").and_then(|v| v.as_str()))
            .unwrap_or("");
        assert!(proxy_desc.contains("Omit tool_name to list"), "{proxy_desc}");
        assert!(proxy_desc.contains("read_file"), "{proxy_desc}");
        assert!(proxy_desc.contains("todo"), "{proxy_desc}");
        assert!(proxy_desc.contains("validate"), "{proxy_desc}");
    }

    #[test]
    fn test_artifact_core_plus_proxy_surface_matches_core_for_prefix_stability() {
        let ws = tempfile::tempdir().unwrap();
        let mut reg = ToolRegistry::with_standard_tools(&ToolConfig::new(ws.path()));
        register_test_result_recall(&mut reg, ws.path().join("sessions.db"));
        let core_defs = reg.get_core_plus_proxy_definitions();
        let defs = reg.get_artifact_core_plus_proxy_definitions();
        // Both surfaces are pure-proxy; they must be byte-identical so the chat
        // template's prompt-head tool block stays hash-stable for prefix cache.
        assert_eq!(
            serde_json::to_string(&defs).unwrap(),
            serde_json::to_string(&core_defs).unwrap(),
            "artifact turns must not switch tool catalogs; the chat template renders tools at the prompt head"
        );
        let proxy_desc = defs
            .iter()
            .find(|d| d.pointer("/function/name").and_then(|v| v.as_str()) == Some("get_tools"))
            .and_then(|d| d.pointer("/function/description").and_then(|v| v.as_str()))
            .unwrap_or("");
        assert!(
            !proxy_desc.contains("write_file_chunk"),
            "proxy must expose one file-writing path: {proxy_desc}"
        );
        assert!(
            proxy_desc.contains("write_file"),
            "proxy must keep transactional write_file reachable: {proxy_desc}"
        );
    }

    /// Lean surface contract: EXACTLY one tool (the `tool` proxy) advertised at
    /// turn 1. It must teach inspect-then-execute and advertise every reachable
    /// lean tool by name, and stay byte-identical across calls (the tool block
    /// is hashed for prefix-cache stability).
    #[test]
    fn test_lean_definitions_surface() {
        let ws = tempfile::tempdir().unwrap();
        let mut reg = ToolRegistry::with_standard_tools(&ToolConfig::new(ws.path()));
        register_test_result_recall(&mut reg, ws.path().join("sessions.db"));
        let lean = reg.get_lean_definitions();

        // Lazy-load contract: exactly ONE tool advertised at turn 1.
        assert_eq!(lean.len(), 1, "lean surface must be the single proxy tool");
        let name = lean[0].pointer("/function/name").and_then(|v| v.as_str());
        assert_eq!(name, Some("get_tools"), "only the proxy meta-tool is advertised");

        let desc = lean[0]
            .pointer("/function/description")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        // The proxy description must teach inspect-then-execute so the model
        // forms correct arguments instead of guessing (the earlier failure mode).
        assert!(
            desc.contains("full parameter"),
            "proxy must teach inspect-then-execute: {desc}"
        );
        // Hot tools must be named in the catalog so the model routes without a
        // round-trip; rarely-used tools are intentionally omitted (discoverable
        // via the omit-tool_name list path — see RARELY_ADVERTISED_TOOLS).
        assert!(
            desc.contains("Omit tool_name to list every available tool"),
            "proxy must teach the list-all discovery path: {desc}"
        );
        let available: std::collections::HashSet<String> =
            reg.available_tool_names().into_iter().collect();
        for t in ToolRegistry::CORE_TOOLS
            .iter()
            .chain(ToolRegistry::LEAN_EXTRA_TOOLS.iter())
        {
            if !available.contains(*t) || ToolRegistry::RARELY_ADVERTISED_TOOLS.contains(t) {
                continue;
            }
            assert!(
                desc.contains(t),
                "proxy must advertise hot tool '{t}': {desc}"
            );
        }
        // Rarely-advertised tools must NOT clutter the per-turn catalog.
        for t in ToolRegistry::RARELY_ADVERTISED_TOOLS {
            if available.contains(*t) {
                assert!(
                    !desc.contains(t),
                    "rarely-used tool '{t}' should be de-listed from catalog: {desc}"
                );
            }
        }

        // Determinism: byte-identical across two calls on the same registry.
        let a = serde_json::to_vec(&lean).unwrap();
        let b = serde_json::to_vec(&reg.get_lean_definitions()).unwrap();
        assert_eq!(a, b, "lean definitions must be byte-identical across calls");
    }

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
        param_names: Vec<String>,
    }

    impl ParamEchoTool {
        fn new(name: &str) -> Self {
            Self {
                tool_name: name.to_string(),
                param_names: Vec::new(),
            }
        }

        fn with_params(name: &str, param_names: &[&str]) -> Self {
            Self {
                tool_name: name.to_string(),
                param_names: param_names.iter().map(|p| p.to_string()).collect(),
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
            let properties: serde_json::Map<String, serde_json::Value> = self
                .param_names
                .iter()
                .map(|name| {
                    (
                        name.clone(),
                        serde_json::json!({
                            "type": "string",
                        }),
                    )
                })
                .collect();
            serde_json::json!({
                "type": "object",
                "properties": properties
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
        reg.register(Box::new(ReadFileTool::default()));
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
            lines_desc
                .and_then(|v| v.as_str())
                .is_some_and(|s| s.contains("1:")),
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
        assert!(result.ok());
        assert_eq!(result.data(), "echo:hello");
        assert!(result.error().is_none());
    }

    #[tokio::test]
    async fn test_execute_missing_tool() {
        let registry = ToolRegistry::new();
        let params = HashMap::new();

        let result = registry.execute("nonexistent", params).await;
        assert!(!result.ok());
        assert!(result.data().contains("Error"));
        assert!(result.data().contains("nonexistent"));
        assert!(result
            .error()
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
        assert!(result.ok(), "{}", result.data());
        let parsed: serde_json::Value = serde_json::from_str(&result.data()).unwrap();
        assert_eq!(parsed["action"], "wait");
        assert_eq!(parsed["task_id"], "abc123");
    }

    #[tokio::test]
    async fn test_execute_spawn_requires_task_for_spawn_action() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(ParamEchoTool::new("spawn")));

        let result = registry.execute("spawn", HashMap::new()).await;
        assert!(!result.ok());
        assert!(result
            .error()
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
        assert!(!result.ok());
        assert!(result
            .error()
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
        assert!(result.ok(), "{}", result.data());
        let parsed: serde_json::Value = serde_json::from_str(&result.data()).unwrap();
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
        assert!(result.ok(), "Expected ok, got error: {:?}", result.error());
        let parsed: serde_json::Value = serde_json::from_str(&result.data()).unwrap();
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
        assert!(result.ok(), "Expected ok, got error: {:?}", result.error());
        let parsed: serde_json::Value = serde_json::from_str(&result.data()).unwrap();
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
        assert!(result.ok(), "Expected ok, got error: {:?}", result.error());
        let parsed: serde_json::Value = serde_json::from_str(&result.data()).unwrap();
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
        assert!(result.ok(), "Expected ok, got error: {:?}", result.error());
        let parsed: serde_json::Value = serde_json::from_str(&result.data()).unwrap();
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
        assert!(result.ok(), "Expected ok, got error: {:?}", result.error());
        let parsed: serde_json::Value = serde_json::from_str(&result.data()).unwrap();
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
        assert!(result.ok(), "Expected ok, got error: {:?}", result.error());
        let parsed: serde_json::Value = serde_json::from_str(&result.data()).unwrap();
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
        assert!(result.ok(), "Expected ok, got error: {:?}", result.error());
        let parsed: serde_json::Value = serde_json::from_str(&result.data()).unwrap();
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
        assert!(result.ok(), "Expected ok, got error: {:?}", result.error());
        let parsed: serde_json::Value = serde_json::from_str(&result.data()).unwrap();
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
        assert!(result.ok(), "Expected ok, got error: {:?}", result.error());
        let parsed: serde_json::Value = serde_json::from_str(&result.data()).unwrap();
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
        assert!(result.ok(), "Expected ok, got error: {:?}", result.error());
        let parsed: serde_json::Value = serde_json::from_str(&result.data()).unwrap();
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
        assert!(result.ok(), "Expected ok, got error: {:?}", result.error());
        let parsed: serde_json::Value = serde_json::from_str(&result.data()).unwrap();
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
        use crate::agent::tools::base::ToolContext;

        let mut registry = ToolRegistry::new();
        registry.register(Box::new(MockTool::new("echo")));

        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel::<ToolEvent>();
        let token = tokio_util::sync::CancellationToken::new();
        let ctx = ToolContext::new(None, tx, token, "call_ctx");

        let mut params = HashMap::new();
        params.insert(
            "value".to_string(),
            serde_json::Value::String("world".to_string()),
        );

        let result = registry.execute_with_context("echo", params, &ctx).await;
        assert!(result.ok());
        assert_eq!(result.data(), "echo:world");
    }

    #[tokio::test]
    async fn test_execute_with_context_missing_tool() {
        use crate::agent::audit::ToolEvent;
        use crate::agent::tools::base::ToolContext;

        let registry = ToolRegistry::new();
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel::<ToolEvent>();
        let token = tokio_util::sync::CancellationToken::new();
        let ctx = ToolContext::new(None, tx, token, "call_missing");

        let result = registry
            .execute_with_context("nonexistent", HashMap::new(), &ctx)
            .await;
        assert!(!result.ok());
        assert!(result.data().contains("not found"));
    }

    #[test]
    fn test_standard_registry_omits_redundant_batch_tool() {
        let registry =
            ToolRegistry::with_standard_tools(&ToolConfig::new(std::path::Path::new(".")));
        assert!(
            !registry.has("batch"),
            "native multi-tool responses are the single batching path"
        );
    }

    #[test]
    fn test_standard_registry_exposes_one_file_writer() {
        let registry =
            ToolRegistry::with_standard_tools(&ToolConfig::new(std::path::Path::new(".")));

        assert!(registry.has("write_file"));
        assert!(
            !registry.has("write_file_chunk"),
            "write_file must own both complete and staged writes"
        );
    }

    /// The internal condensed builder starts from every registered tool before
    /// the Lean production subset is selected.
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
            result.ok(),
            "Unavailable tool should still execute when called directly: {:?}",
            result.error()
        );
        assert_eq!(result.data(), "executed");
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
        assert!(allowed_tools.contains(&"get_skills".to_string()));

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
            "get_skills",
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
        // Inline mock with a 3-sentence description so this test doesn't
        // depend on any real tool's prose.
        struct ThreeSentenceTool;

        #[async_trait]
        impl Tool for ThreeSentenceTool {
            fn name(&self) -> &str {
                "three_sentence"
            }
            fn description(&self) -> &str {
                "Does a thing. Returns the result. This third sentence should be dropped."
            }
            fn parameters(&self) -> serde_json::Value {
                serde_json::json!({ "type": "object", "properties": {} })
            }
            async fn execute(&self, _params: HashMap<String, serde_json::Value>) -> String {
                String::new()
            }
        }

        let tool = ThreeSentenceTool;
        let full_desc = tool.description().to_string();
        // Count sentences in full description (periods followed by space).
        let sentence_breaks = full_desc.matches(". ").count();
        assert!(
            sentence_breaks >= 2,
            "fixture should have 3+ sentences, got {} breaks in: {}",
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
            "get_tools",
            "Proxy tool must be named 'get_tools'"
        );

        let props = &defs[0]["function"]["parameters"]["properties"];
        assert!(
            props.get("tool_name").is_some(),
            "Must have 'tool_name' param"
        );
        assert!(
            props.get("tool_args").is_some(),
            "Must have 'tool_args' param"
        );
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

        let result = registry.execute("get_tools", params).await;
        assert!(result.ok(), "Inspect should succeed: {:?}", result.error());
        assert!(
            result.data().contains("parameters"),
            "Inspect result should contain parameters schema: {}",
            result.data()
        );
        assert!(
            result.data().contains("value"),
            "Schema should mention required param 'value': {}",
            result.data()
        );
    }

    #[tokio::test]
    async fn test_proxy_inspect_unknown_lists_available() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(MockTool::new("read_file")));

        let mut params = HashMap::new();
        params.insert("name".to_string(), serde_json::json!("nonexistent"));

        let result = registry.execute("get_tools", params).await;
        assert!(!result.ok(), "Unknown tool inspect should fail");
        assert!(
            result.data().contains("read_file"),
            "Error should list available tools: {}",
            result.data()
        );
    }

    #[tokio::test]
    async fn test_proxy_dispatch_executes_real_tool() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(MockTool::new("mock_tool")));

        let mut params = HashMap::new();
        params.insert("name".to_string(), serde_json::json!("mock_tool"));
        params.insert("args".to_string(), serde_json::json!({"value": "hello"}));

        let result = registry.execute("get_tools", params).await;
        assert!(
            result.ok(),
            "Proxy dispatch should succeed: {:?}",
            result.error()
        );
        assert!(
            result.data().contains("hello"),
            "Should contain dispatched tool output: {}",
            result.data()
        );
    }

    // -----------------------------------------------------------------
    // Proxy parameter rename (2026-07-27) — `name`/`args` →
    // `tool_name`/`tool_args`. The old names overloaded with inner
    // tool params (e.g. `get_skills` has its own `name`), causing
    // small local models to collapse everything into `args` and omit
    // the top-level tool name. See live failure 2026-07-27 17:21
    // (session 20260727_173522_263450).
    // -----------------------------------------------------------------

    /// The proxy schema MUST advertise `tool_name` (not `name`) as the
    /// parameter that selects the inner tool. The old `name` parameter
    /// is reserved for the inner tool's own use (e.g. `get_skills(name)`).
    #[test]
    fn proxy_schema_uses_tool_name_not_name() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(MockTool::new("mock_tool")));
        let defs = registry.get_proxy_definition();
        let params = defs[0]["function"]["parameters"]["properties"]
            .as_object()
            .expect("proxy parameters must be an object");
        assert!(
            params.contains_key("tool_name"),
            "proxy schema must advertise `tool_name` (not `name`), got keys: {:?}",
            params.keys().collect::<Vec<_>>()
        );
        assert!(
            params.contains_key("tool_args"),
            "proxy schema must advertise `tool_args` (not `args`), got keys: {:?}",
            params.keys().collect::<Vec<_>>()
        );
        assert!(
            !params.contains_key("name"),
            "proxy schema must NOT advertise `name` at the proxy level — that name belongs to inner tools"
        );
    }

    /// Calling `tool({tool_name: "X", tool_args: {...}})` MUST dispatch
    /// the inner tool correctly. This is the corrected shape that small
    /// local models can produce reliably (no nested-`name` ambiguity).
    #[tokio::test]
    async fn proxy_dispatch_via_tool_name_tool_args() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(MockTool::new("mock_tool")));

        let mut params = HashMap::new();
        params.insert("tool_name".to_string(), serde_json::json!("mock_tool"));
        params.insert(
            "tool_args".to_string(),
            serde_json::json!({"value": "via_tool_name"}),
        );

        let result = registry.execute("get_tools", params).await;
        assert!(
            result.ok(),
            "dispatch via tool_name/tool_args must succeed: {:?}",
            result.error()
        );
        assert!(
            result.data().contains("via_tool_name"),
            "should contain dispatched tool output: {}",
            result.data()
        );
    }

    /// Back-compat: old persisted calls may still arrive with `name`/`args`.
    /// The proxy must accept them — a hard rename would break in-flight
    /// sessions and tests with old fixtures. Old form continues to work;
    /// new form is the documented preferred shape.
    #[tokio::test]
    async fn proxy_dispatch_legacy_name_args_still_accepted() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(MockTool::new("mock_tool")));

        let mut params = HashMap::new();
        params.insert("name".to_string(), serde_json::json!("mock_tool"));
        params.insert("args".to_string(), serde_json::json!({"value": "legacy"}));

        let result = registry.execute("get_tools", params).await;
        assert!(
            result.ok(),
            "legacy name/args must still be accepted for back-compat: {:?}",
            result.error()
        );
        assert!(result.data().contains("legacy"));
    }

    /// The model emitting flattened args with no tool selector MUST produce
    /// an error message that names `tool_name` as the missing field, not
    /// the old `name`. The error message is the model's primary correction
    /// signal — it must reference the new schema's parameter name.
    #[tokio::test]
    async fn proxy_missing_name_error_references_tool_name() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(MockTool::new("mock_tool")));

        // Flattened-form: caller put inner-tool args directly at the proxy
        // level (no `tool_name`, no `args` envelope). The proxy can't
        // dispatch without knowing which tool — and the corrective error
        // must reference `tool_name` so the model knows what to add.
        let mut params = HashMap::new();
        params.insert("url".to_string(), serde_json::json!("https://example.com"));

        let result = registry.execute("get_tools", params).await;
        assert!(
            !result.ok(),
            "flattened-form without tool selector must fail (not silently succeed)"
        );
        let err = result.error().unwrap_or("");
        assert!(
            err.contains("tool_name"),
            "error must reference `tool_name` (the new parameter), got: {err}"
        );
    }

    #[tokio::test]
    async fn test_proxy_accepts_large_complete_write_file_payload() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("artifact.html");
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(WriteFileTool::default()));

        assert_eq!(
            MAX_WRITE_FILE_PIECE_CHARS, 4096,
            "local models must use the same bounded piece size the writer stages"
        );
        let oversized = "x".repeat(MAX_WRITE_FILE_PIECE_CHARS + 1);

        let mut params = HashMap::new();
        params.insert("name".to_string(), serde_json::json!("write_file"));
        params.insert(
            "args".to_string(),
            serde_json::json!({
                "path": file_path,
                "content": oversized,
                "state": "complete",
            }),
        );

        let result = registry.execute("get_tools", params).await;

        assert!(result.ok(), "large complete proxy write failed: {result:?}");
        assert!(
            result.data().contains("Validate the published artifact"),
            "publication must require validation: {}",
            result.data()
        );
        assert_eq!(std::fs::metadata(file_path).unwrap().len(), 4097);
    }

    #[tokio::test]
    async fn test_proxy_rejects_only_large_more_write_file_payload() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("artifact.html");
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(WriteFileTool::default()));

        let mut params = HashMap::new();
        params.insert("name".to_string(), serde_json::json!("write_file"));
        params.insert(
            "args".to_string(),
            serde_json::json!({
                "path": file_path,
                "content": "x".repeat(MAX_WRITE_FILE_PIECE_CHARS + 1),
                "state": "more",
            }),
        );

        let result = registry.execute("get_tools", params).await;

        assert!(!result.ok(), "oversized state=more must be rejected");
        assert!(result.data().contains("state=\"more\""), "{}", result.data());
        assert!(!file_path.exists());
    }

    #[tokio::test]
    async fn test_direct_write_file_keeps_large_payload_support() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("direct.txt");
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(WriteFileTool::default()));

        let mut params = HashMap::new();
        params.insert("path".to_string(), serde_json::json!(file_path));
        params.insert("content".to_string(), serde_json::json!("x".repeat(4097)));

        let result = registry.execute("write_file", params).await;

        assert!(
            result.ok(),
            "proxy guard must not alter direct/cloud write_file calls: {:?}",
            result.error()
        );
        assert_eq!(std::fs::metadata(file_path).unwrap().len(), 4097);
    }

    #[tokio::test]
    async fn test_proxy_dispatch_accepts_flattened_args() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(ParamEchoTool::with_params("recall", &["mode"])));

        let mut params = HashMap::new();
        params.insert("name".to_string(), serde_json::json!("recall"));
        params.insert("mode".to_string(), serde_json::json!("latest"));

        let result = registry.execute("get_tools", params).await;
        assert!(
            result.ok(),
            "Flattened proxy dispatch should succeed: {:?}",
            result.error()
        );
        assert!(
            result.data().contains("latest"),
            "Flattened mode should be moved into args: {}",
            result.data()
        );
    }

    #[tokio::test]
    async fn test_proxy_stray_metadata_keeps_inspect_mode() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(MockTool::new("read_file")));

        let mut params = HashMap::new();
        params.insert("name".to_string(), serde_json::json!("read_file"));
        params.insert("reason".to_string(), serde_json::json!("need schema first"));

        let result = registry.execute("get_tools", params).await;
        assert!(
            result.ok(),
            "Stray metadata should not dispatch tool: {:?}",
            result.error()
        );
        assert!(
            result.data().contains("\"parameters\""),
            "Stray metadata should leave proxy in inspect mode: {}",
            result.data()
        );
        assert!(
            !result.data().contains("read_file:default"),
            "Inspect mode must not execute the underlying tool: {}",
            result.data()
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

        let result = registry.execute("get_tools", params).await;
        assert!(
            result.ok(),
            "Dispatch with alias should succeed: {:?}",
            result.error()
        );
        assert!(
            result.data().contains("path"),
            "Normalization should convert file_path to path: {}",
            result.data()
        );
    }

    #[tokio::test]
    async fn test_proxy_missing_name_returns_tool_list() {
        // No name → list mode (success). The prompt teaches "omit tool_name to list"
        // so this MUST return a success with the catalog, not a failure error.
        // Returning a failure here was the bonsai confabulation trigger.
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(MockTool::new("read_file")));

        let result = registry.execute("get_tools", HashMap::new()).await;
        assert!(
            result.ok(),
            "Missing name should return success with tool list"
        );
        assert!(
            result.data().contains("read_file"),
            "Should list available tools: {}",
            result.data()
        );
    }

    #[tokio::test]
    async fn test_execute_proxy_missing_name_with_args_is_actionable_error() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(MockTool::new("web_fetch")));
        registry.register(Box::new(MockTool::new("read_file")));

        // Case 1: No name but WITH other parameters → should be FAILURE (not success catalog).
        // This is the bug fix: model likely forgot to pass tool_name, sending
        // {"url": "..."} instead of {"tool_name": "web_fetch", "tool_args": {"url": "..."}}.
        let mut params = HashMap::new();
        params.insert("url".to_string(), serde_json::json!("https://example.com"));

        let result = registry.execute("get_tools", params).await;
        assert!(
            !result.ok(),
            "Missing tool_name with stray params should be FAILURE, not success catalog: {}",
            result.data()
        );
        assert!(
            result.data().contains("'tool_name' is required"),
            "Error should mention tool_name is required: {}",
            result.data()
        );
        assert!(
            result.data().contains("url"),
            "Error should mention the stray parameter sent: {}",
            result.data()
        );
        assert!(
            result.data().contains("web_fetch"),
            "Error should list available tools including web_fetch: {}",
            result.data()
        );

        // Case 2: Empty params (genuine discovery) → should still be SUCCESS with catalog.
        // This is the regression guard: the documented discovery path must work.
        let result = registry.execute("get_tools", HashMap::new()).await;
        assert!(
            result.ok(),
            "Empty params should return success with tool list (discovery path)"
        );
        assert!(
            result.data().contains("Available tools:"),
            "Should list available tools: {}",
            result.data()
        );
        assert!(
            result.data().contains("web_fetch"),
            "Should include web_fetch in tool list: {}",
            result.data()
        );

        // Case 3: `args` present but no name. Observed in production — the model
        // sends {"args": {"url": "..."}}, and answering with the catalog reads as
        // success, so it repeats the same malformed call instead of correcting it.
        let mut params = HashMap::new();
        params.insert(
            "args".to_string(),
            serde_json::json!({"url": "https://example.com"}),
        );
        let result = registry.execute("get_tools", params).await;
        assert!(
            !result.ok(),
            "args without tool_name must fail, not return the catalog: {}",
            result.data()
        );
        assert!(
            result.data().contains("'tool_name' is required"),
            "{}",
            result.data()
        );

        // Case 4: blank filler around an otherwise bare call is still discovery.
        let mut params = HashMap::new();
        params.insert("args".to_string(), serde_json::json!({}));
        params.insert("name".to_string(), serde_json::Value::Null);
        let result = registry.execute("get_tools", params).await;
        assert!(
            result.ok() && result.data().contains("Available tools:"),
            "empty args/null name is a bare discovery call: {}",
            result.data()
        );
    }

    /// Regression: a bare `get_tools({})` discovery call must tell the model
    /// HOW to retrieve a tool's schema. Without an inspect hint in the result,
    /// weak local models loop calling `get_tools({})` identically (each call
    /// returns the same flat name list) until the dedup guard terminates the
    /// turn with a generic message. The catalog result is the feedback the
    /// model actually reads, so the hint must live here, not only in the tool
    /// description. See .planning/debug/get-tools-dedup-drop.md.
    #[tokio::test]
    async fn test_proxy_catalog_directs_model_to_inspect_path() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(MockTool::new("edit_file")));
        registry.register(Box::new(MockTool::new("read_file")));

        let result = registry.execute("get_tools", HashMap::new()).await;
        assert!(result.ok(), "discovery must succeed: {:?}", result.data());
        assert!(
            result.data().contains("Available tools:"),
            "catalog must still list tools: {}",
            result.data()
        );
        // The actionable hint: the model must learn, from THIS result, that
        // passing tool_name yields a schema. A flat "Available tools: a, b"
        // alone does not break the identical-retry loop.
        assert!(
            result.data().contains("tool_name")
                && (result.data().contains("schema")
                    || result.data().contains("inspect")
                    || result.data().contains("parameters")),
            "catalog must direct the model to the inspect path (tool_name → schema): {}",
            result.data()
        );
    }

    /// Unit test for the schema-derived worked-example builder. Pins the three
    /// branches: declared required params, fallback to first property (for
    /// mode-defaulted tools like session_search whose `required` is empty), and
    /// None for genuinely no-arg tools.
    #[test]
    fn worked_example_call_derives_from_schema() {
        // Declared required params → example includes exactly those.
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"query": {"type": "string"}, "mode": {"type": "string"}},
            "required": ["query"]
        });
        assert_eq!(
            ToolRegistry::worked_example_call("recall", &schema).as_deref(),
            Some("recall({\"query\":\"...\"})")
        );

        // No required declared → falls back to first property (session_search
        // declares required: [] because its default mode needs no args, but the
        // empty-arg loop is on `query`).
        let schema_no_required = serde_json::json!({
            "type": "object",
            "properties": {"query": {"type": "string"}, "mode": {"type": "string"}}
        });
        let example =
            ToolRegistry::worked_example_call("session_search", &schema_no_required)
                .expect("first-property fallback must yield an example");
        assert!(
            example.starts_with("session_search({\""),
            "fallback example must include a property: {example}"
        );

        // Genuinely no-arg tool → None (no augmentation possible).
        let no_args = serde_json::json!({"type": "object", "properties": {}});
        assert_eq!(ToolRegistry::worked_example_call("system_info", &no_args), None);
    }

    /// Old tool names from prior sessions must keep resolving to their dissolved
    /// target via `normalize_tool_request`, so in-flight and replayed
    /// `tool_calls` rows never hit "Tool not found". (Recovery-tool renames are
    /// a separate follow-up — not yet applied — so they aren't aliased here.)
    #[test]
    fn normalize_routes_old_tool_names_to_new_targets() {
        for (old, new) in [
            ("session_search", "recall"),
            ("search_context", "recall"),
        ] {
            let (c, _) = ToolRegistry::normalize_tool_request(old, HashMap::new()).unwrap();
            assert_eq!(c, new, "alias {old} -> {new}");
        }
    }

    /// Regression (defect 2): a tool that rejects empty args with the canonical
    /// "Error: 'X' is required" must, at the registry dispatch boundary, get a
    /// corrective worked example appended — derived from the tool's OWN schema.
    /// This is the moment a zero-temp weak model pays attention; without the
    /// shape it retries identically until the dedup guard kills the turn.
    /// See .planning/debug/get-tools-dedup-drop.md.
    #[tokio::test]
    async fn test_missing_required_arg_error_appends_schema_derived_example() {
        struct RequireQueryTool;
        #[async_trait]
        impl Tool for RequireQueryTool {
            fn name(&self) -> &str {
                "require_query"
            }
            fn description(&self) -> &str {
                "test tool requiring query"
            }
            fn parameters(&self) -> serde_json::Value {
                serde_json::json!({
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"]
                })
            }
            async fn execute(&self, _params: HashMap<String, serde_json::Value>) -> String {
                "Error: 'query' parameter is required and must be non-empty.".to_string()
            }
        }

        let mut registry = ToolRegistry::new();
        registry.register(Box::new(RequireQueryTool));

        let result = registry.execute("require_query", HashMap::new()).await;
        assert!(
            !result.ok(),
            "tool must still report failure (no silent success)"
        );
        // The corrective shape, derived from the schema's required params:
        assert!(
            result.data().contains("Call as require_query({\"query\":\"...\"})"),
            "missing-arg error must echo the schema-derived worked example: {}",
            result.data()
        );
        // The original error text is preserved (augmented, not replaced):
        assert!(
            result.data().contains("'query' parameter is required"),
            "augmentation must preserve the original error: {}",
            result.data()
        );
    }

    /// Structural path: a tool that sets `error_kind = MissingArg` gets its
    /// worked example appended EVEN when its data string lacks the "is
    /// required" substring — the failure mode that left `remember` and
    /// `lcm_expand` unaugmented under the old substring gate.
    #[tokio::test]
    async fn missing_arg_error_kind_appends_structured_example() {
        use crate::agent::tools::base::ToolExecutionResult;

        struct StructuredMissingArg;
        #[async_trait]
        impl Tool for StructuredMissingArg {
            fn name(&self) -> &str {
                "structured_missing_arg"
            }
            fn description(&self) -> &str {
                "test"
            }
            fn parameters(&self) -> serde_json::Value {
                serde_json::json!({"type":"object","properties":{"facts":{"type":"array"}}})
            }
            async fn execute(&self, _: HashMap<String, serde_json::Value>) -> String {
                "Error: provide facts".to_string() // NOTE: no "is required" substring
            }
            async fn execute_with_result_and_context(
                &self,
                _: HashMap<String, serde_json::Value>,
                _ctx: &ToolContext,
            ) -> ToolExecutionResult {
                ToolExecutionResult::failure_with_kind(
                    "Error: provide facts".to_string(),
                    crate::errors::ToolErrorKind::MissingArg {
                        param: "facts".to_string(),
                        example: r#"structured_missing_arg({"facts":["..."]})"#.to_string(),
                    },
                )
            }
        }

        let mut registry = ToolRegistry::new();
        registry.register(Box::new(StructuredMissingArg));

        let result = registry
            .execute("structured_missing_arg", HashMap::new())
            .await;
        assert!(!result.ok(), "must still report failure");
        assert!(
            result.data().contains(r#"Call as structured_missing_arg({"facts":["..."]})"#),
            "structural MissingArg must append the example from error_kind: {}",
            result.data()
        );
    }

    #[tokio::test]
    async fn test_proxy_intercept_via_execute() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(MockTool::new("list_dir")));

        // Back-compat: the legacy "tool" alias must still dispatch to the proxy.
        let mut params = HashMap::new();
        params.insert("name".to_string(), serde_json::json!("list_dir"));
        params.insert("args".to_string(), serde_json::json!({"value": "test"}));

        let result = registry.execute("tool", params).await;
        assert!(
            result.ok(),
            "Proxy intercept via execute() should work: {:?}",
            result.error()
        );
        assert!(
            result.data().contains("test"),
            "Should dispatch to real tool: {}",
            result.data()
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
            result.ok(),
            "Direct call should still work: {:?}",
            result.error()
        );
        assert!(
            result.data().contains("direct"),
            "Should get direct tool output: {}",
            result.data()
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
        assert!(!result.ok());
        assert!(result.data().contains("Permission denied"));
    }

    #[tokio::test]
    async fn test_permission_allowed_at_ceiling() {
        let mut registry = ToolRegistry::with_max_permission(PermissionLevel::Execute);
        registry.register(Box::new(ExecuteTool));

        let result = registry.execute("exec_mock", HashMap::new()).await;
        assert!(result.ok());
        assert_eq!(result.data(), "executed");
    }

    #[tokio::test]
    async fn test_permission_allowed_above_ceiling() {
        let mut registry = ToolRegistry::with_max_permission(PermissionLevel::System);
        registry.register(Box::new(ExecuteTool));

        let result = registry.execute("exec_mock", HashMap::new()).await;
        assert!(result.ok());
    }

    #[test]
    fn test_set_max_permission() {
        let mut registry = ToolRegistry::new();
        assert_eq!(registry.max_permission, PermissionLevel::System);
        registry.set_max_permission(PermissionLevel::Write);
        assert_eq!(registry.max_permission, PermissionLevel::Write);
    }
}
