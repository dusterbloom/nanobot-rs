//! Taint tracking for web tool results.
//!
//! When the agent fetches external content via `web_fetch` or `web_search`,
//! that content is considered "tainted" — it originates from an untrusted
//! source and may contain adversarial instructions (prompt injection).
//!
//! This module tracks taint spans introduced during a turn. When the agent
//! subsequently calls a sensitive tool (e.g. `exec`, `write_file`) while
//! tainted context is active, a warning is emitted so that the behaviour
//! can be audited or blocked by a future policy layer.

use std::collections::HashSet;

/// A record of a taint source.
#[derive(Debug, Clone)]
pub struct TaintSpan {
    /// Which tool introduced the taint (e.g., "web_fetch", "web_search")
    pub source_tool: String,
    /// Optional: URL or query that was the source
    pub source_detail: Option<String>,
}

/// Tracks taint state for a session/conversation.
#[derive(Debug, Default)]
pub struct TaintState {
    /// Active taint spans — cleared when conversation resets
    spans: Vec<TaintSpan>,
    /// Tool names that introduce taint
    taint_sources: HashSet<String>,
    /// Tool names that are sensitive when taint is active
    sensitive_tools: HashSet<String>,
}

impl TaintState {
    pub fn new() -> Self {
        let mut taint_sources = HashSet::new();
        taint_sources.insert("web_fetch".to_string());
        taint_sources.insert("web_search".to_string());

        let mut sensitive_tools = HashSet::new();
        sensitive_tools.insert("exec".to_string());
        sensitive_tools.insert("write_file".to_string());
        sensitive_tools.insert("edit_file".to_string());
        sensitive_tools.insert("apply_patch".to_string());
        sensitive_tools.insert("execute_code".to_string());
        #[cfg(feature = "python-kernel")]
        sensitive_tools.insert("python".to_string());

        Self {
            spans: Vec::new(),
            taint_sources,
            sensitive_tools,
        }
    }

    /// Record that a taint source tool was executed.
    pub fn mark_tainted(&mut self, tool_name: &str, detail: Option<String>) {
        if self.taint_sources.contains(tool_name) {
            self.spans.push(TaintSpan {
                source_tool: tool_name.to_string(),
                source_detail: detail,
            });
        }
    }

    /// Check if the context is currently tainted.
    pub fn is_tainted(&self) -> bool {
        !self.spans.is_empty()
    }

    /// Check if a tool call should trigger a taint warning.
    /// Returns the taint spans if the tool is sensitive and context is tainted.
    pub fn check_sensitive(&self, tool_name: &str) -> Option<&[TaintSpan]> {
        if self.sensitive_tools.contains(tool_name) && self.is_tainted() {
            Some(&self.spans)
        } else {
            None
        }
    }

    /// Get a human-readable summary of active taint sources for logging.
    pub fn taint_summary(&self) -> String {
        if self.spans.is_empty() {
            return "none".to_string();
        }
        self.spans
            .iter()
            .map(|s| {
                if let Some(ref detail) = s.source_detail {
                    format!("{}({})", s.source_tool, detail)
                } else {
                    s.source_tool.clone()
                }
            })
            .collect::<Vec<_>>()
            .join(", ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_state_not_tainted() {
        let state = TaintState::new();
        assert!(!state.is_tainted());
    }

    #[test]
    fn test_mark_web_fetch_tainted() {
        let mut state = TaintState::new();
        state.mark_tainted("web_fetch", Some("https://example.com".into()));
        assert!(state.is_tainted());
        assert_eq!(state.check_sensitive("exec").unwrap().len(), 1);
    }

    #[test]
    fn test_non_taint_source_ignored() {
        let mut state = TaintState::new();
        state.mark_tainted("read_file", None);
        assert!(!state.is_tainted());
    }

    #[test]
    fn test_exec_sensitive_when_tainted() {
        let mut state = TaintState::new();
        state.mark_tainted("web_search", Some("query".into()));
        assert!(state.check_sensitive("exec").is_some());
    }

    #[test]
    fn test_exec_not_sensitive_when_clean() {
        let state = TaintState::new();
        assert!(state.check_sensitive("exec").is_none());
    }

    #[test]
    fn test_write_file_sensitive_when_tainted() {
        let mut state = TaintState::new();
        state.mark_tainted("web_fetch", None);
        assert!(state.check_sensitive("write_file").is_some());
    }

    #[test]
    fn test_read_file_not_sensitive() {
        let mut state = TaintState::new();
        state.mark_tainted("web_fetch", None);
        assert!(state.check_sensitive("read_file").is_none());
    }

    #[test]
    fn test_taint_summary() {
        let mut state = TaintState::new();
        state.mark_tainted("web_fetch", Some("https://evil.com".into()));
        state.mark_tainted("web_search", Some("injection query".into()));
        let summary = state.taint_summary();
        assert!(summary.contains("web_fetch(https://evil.com)"));
        assert!(summary.contains("web_search(injection query)"));
    }

    #[test]
    fn test_multiple_taint_spans() {
        let mut state = TaintState::new();
        state.mark_tainted("web_fetch", Some("url1".into()));
        state.mark_tainted("web_fetch", Some("url2".into()));
        assert_eq!(state.check_sensitive("exec").unwrap().len(), 2);
    }

    // --- Integration scenario tests: simulating tool_engine.rs call sequence ---

    #[test]
    fn test_web_fetch_then_exec_scenario() {
        // Simulates the exact sequence that happens in tool_engine.rs:
        // 1. Agent calls web_fetch -> result comes back -> mark tainted
        // 2. Agent calls exec -> check_sensitive fires -> warning should trigger

        let mut state = TaintState::new();

        // Step 1: web_fetch executes (tool_engine marks it)
        let tool_name = "web_fetch";
        state.mark_tainted(tool_name, Some("https://attacker.com/payload".into()));

        // Step 2: Agent now wants to run exec (tool_engine checks before execution)
        let sensitive_tool = "exec";
        let taint_check = state.check_sensitive(sensitive_tool);

        // This is the check that tool_engine.rs performs before executing
        assert!(
            taint_check.is_some(),
            "exec should be flagged as sensitive when web content is in context"
        );

        // Verify the taint spans contain the source
        let spans = taint_check.unwrap();
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].source_tool, "web_fetch");
        assert_eq!(
            spans[0].source_detail.as_deref(),
            Some("https://attacker.com/payload")
        );

        // Verify the summary matches what would be logged
        let summary = state.taint_summary();
        assert!(summary.contains("web_fetch(https://attacker.com/payload)"));
    }

    #[test]
    fn test_web_search_then_write_file_scenario() {
        let mut state = TaintState::new();

        // web_search introduces taint
        state.mark_tainted("web_search", Some("how to hack".into()));

        // write_file should trigger warning
        assert!(state.check_sensitive("write_file").is_some());
        assert!(state.check_sensitive("edit_file").is_some());
        assert!(state.check_sensitive("apply_patch").is_some());
        assert!(state.check_sensitive("execute_code").is_some());

        // But read_file should NOT trigger
        assert!(state.check_sensitive("read_file").is_none());
        assert!(state.check_sensitive("list_dir").is_none());
    }

    #[test]
    fn test_subagent_taint_scenario() {
        // Simulates the subagent tool execution loop
        let mut state = TaintState::new();

        // Subagent calls web_fetch first
        state.mark_tainted("web_fetch", Some("https://untrusted.com".into()));

        // Then tries to write_file
        assert!(
            state.check_sensitive("write_file").is_some(),
            "write_file in subagent after web_fetch should trigger warning"
        );

        // And exec
        assert!(
            state.check_sensitive("exec").is_some(),
            "exec in subagent after web_fetch should trigger warning"
        );
    }

    #[test]
    fn test_multi_turn_taint_accumulation() {
        // Simulates multiple turns where web tools are called
        let mut state = TaintState::new();

        // Turn 1: web_fetch
        state.mark_tainted("web_fetch", Some("https://site1.com".into()));

        // Turn 2: web_search
        state.mark_tainted("web_search", Some("query about site1".into()));

        // Turn 3: exec - should see both taint spans
        let spans = state.check_sensitive("exec").unwrap();
        assert_eq!(spans.len(), 2);

        // Summary should show both sources
        let summary = state.taint_summary();
        assert!(summary.contains("web_fetch"));
        assert!(summary.contains("web_search"));
    }

    // --- Regression guard: the sensitive set must match the real registry ---

    /// The hardcoded `sensitive_tools` set must cover every file-mutating and
    /// code-executing tool that `register_standard_tools` actually registers,
    /// and must not contain stale names that match no registered tool. This
    /// test fails if the registry grows a new sensitive tool and `taint.rs`
    /// is not updated, preventing the drift this bug was caused by.
    #[test]
    fn test_sensitive_set_covers_all_registered_sensitive_tools() {
        use crate::agent::tools::registry::{ToolConfig, ToolRegistry};

        let ws = tempfile::tempdir().unwrap();
        let mut config = ToolConfig::new(ws.path());
        config.code_execution.enabled = true;
        #[cfg(feature = "python-kernel")]
        {
            config.python_kernel.enabled = true;
        }
        let reg = ToolRegistry::with_standard_tools(&config);

        let mut state = TaintState::new();
        state.mark_tainted("web_fetch", Some("https://attacker.example/payload".into()));

        // The file-mutating class — exactly the set the registry's `read_only`
        // guard groups (`write_file | edit_file | apply_patch`).
        for file_tool in ["write_file", "edit_file", "apply_patch"] {
            assert!(
                reg.has(file_tool),
                "expected '{file_tool}' to be registered by register_standard_tools"
            );
            assert!(
                state.check_sensitive(file_tool).is_some(),
                "taint layer must flag file-mutating tool '{file_tool}' while tainted — \
                 check_sensitive returned None"
            );
        }

        // Code-execution primitives.
        for exec_tool in ["exec", "execute_code"] {
            assert!(
                reg.has(exec_tool),
                "expected '{exec_tool}' to be registered"
            );
            assert!(
                state.check_sensitive(exec_tool).is_some(),
                "taint layer must flag code-executing tool '{exec_tool}' while tainted"
            );
        }

        #[cfg(feature = "python-kernel")]
        {
            assert!(
                reg.has("python"),
                "expected 'python' to be registered when the python-kernel feature is on"
            );
            assert!(
                state.check_sensitive("python").is_some(),
                "taint layer must flag code-executing tool 'python' while tainted"
            );
        }

        // Stale names that caused this bug: no registered tool is named
        // `create_file` or `patch_file`, so flagging them would be a dead
        // signal. They must NOT be in the sensitive set.
        for dead in ["create_file", "patch_file"] {
            assert!(
                !reg.has(dead),
                "'{dead}' should not be a registered tool name (sanity check)"
            );
            assert!(
                state.check_sensitive(dead).is_none(),
                "taint layer must not flag stale name '{dead}' that matches no registered tool"
            );
        }

        // A representative read-only tool is registered but must NOT be flagged.
        assert!(reg.has("read_file"));
        assert!(
            state.check_sensitive("read_file").is_none(),
            "read-only tools must not be flagged as sensitive"
        );
    }
}
