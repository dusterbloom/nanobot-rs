// Error-protocol layer-3 backlog (docs/research/2026-08-06-error-conventions-and-host-bridge.md §3.6):
// the deny regime in Cargo.toml is live; this module still carries pre-existing
// violations of the lints below. Remove this allow as the module migrates onto
// the regime.
// Tracking: docs/error-protocol-backlog.md
#![allow(
    clippy::as_conversions,
    clippy::format_push_string,
    clippy::shadow_reuse
)]
//! Tool status: summarize tool health, audit history, and skill validation.

use std::collections::{BTreeMap, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};

use super::base::{Tool, ToolContext, ToolResult};
use crate::agent::audit::AuditEntry;
use crate::agent::skills::SkillsLoader;
use crate::errors::ToolError;

const DEFAULT_RECENT_LIMIT: usize = 80;
const MAX_RECENT_LIMIT: usize = 500;
const SUMMARY_ROWS: usize = 5;
const DETAIL_ROWS: usize = 10;
const MAX_AUDIT_FILES: usize = 8;

/// Tool to inspect existing observability data without shelling out.
pub struct ToolStatusTool {
    workspace: PathBuf,
}

impl ToolStatusTool {
    pub fn new(workspace: PathBuf) -> Self {
        Self { workspace }
    }
}

#[async_trait]
impl Tool for ToolStatusTool {
    fn name(&self) -> &str {
        "tool_status"
    }

    fn description(&self) -> &str {
        "Summarize tool health from learning logs, audit logs, and skill validation. Use this for tool observability, tool failures, and skill metadata hygiene."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["summary", "learning", "audit", "skills", "all"],
                    "description": "Which status section to show. Default: summary"
                },
                "recent_limit": {
                    "type": "integer",
                    "description": "Recent learning/audit entries to inspect. Default: 80, max: 500"
                }
            }
        })
    }

    async fn execute_typed(
        &self,
        params: std::collections::HashMap<String, Value>,
        _ctx: &ToolContext,
    ) -> ToolResult {
        let action = params
            .get("action")
            .and_then(|v| v.as_str())
            .unwrap_or("summary");
        if ToolStatusAction::parse(action).is_none() {
            return Err(ToolError::InvalidArgs {
                message: "'action' must be one of: summary, learning, audit, skills, all"
                    .to_string(),
            });
        }
        let recent_limit = params
            .get("recent_limit")
            .and_then(|v| v.as_u64())
            .map(|v| (v as usize).clamp(1, MAX_RECENT_LIMIT))
            .unwrap_or(DEFAULT_RECENT_LIMIT);

        Ok(build_tool_status_report(&self.workspace, action, recent_limit).into())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ToolStatusAction {
    Summary,
    Learning,
    Audit,
    Skills,
    All,
}

impl ToolStatusAction {
    fn parse(action: &str) -> Option<Self> {
        match action {
            "summary" => Some(Self::Summary),
            "learning" => Some(Self::Learning),
            "audit" => Some(Self::Audit),
            "skills" => Some(Self::Skills),
            "all" => Some(Self::All),
            _ => None,
        }
    }
}

/// Build a human-readable status report from existing on-disk observability.
pub(crate) fn build_tool_status_report(
    workspace: &Path,
    action: &str,
    recent_limit: usize,
) -> String {
    let action = ToolStatusAction::parse(action).unwrap_or(ToolStatusAction::Summary);
    let max_rows = if action == ToolStatusAction::Summary {
        SUMMARY_ROWS
    } else {
        DETAIL_ROWS
    };

    let mut sections = Vec::new();
    match action {
        ToolStatusAction::Summary => {
            sections.push(render_learning_status(workspace, recent_limit, max_rows));
            sections.push(render_audit_status(workspace, recent_limit, max_rows));
            sections.push(render_skill_status(workspace, max_rows));
        }
        ToolStatusAction::Learning => {
            sections.push(render_learning_status(workspace, recent_limit, max_rows));
        }
        ToolStatusAction::Audit => {
            sections.push(render_audit_status(workspace, recent_limit, max_rows));
        }
        ToolStatusAction::Skills => {
            sections.push(render_skill_status(workspace, max_rows));
        }
        ToolStatusAction::All => {
            sections.push(render_learning_status(workspace, recent_limit, max_rows));
            sections.push(render_audit_status(workspace, recent_limit, max_rows));
            sections.push(render_skill_status(workspace, max_rows));
        }
    }

    format!(
        "# Tool Status\nWorkspace: {}\n\n{}",
        workspace.display(),
        sections.join("\n\n")
    )
}

#[derive(Debug, Deserialize)]
struct LearningEntry {
    #[serde(default)]
    tool_name: String,
    #[serde(default)]
    succeeded: bool,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    latency_ms: Option<u64>,
}

#[derive(Debug, Default)]
struct ToolStats {
    ok: usize,
    failed: usize,
    latency_total_ms: u64,
    latency_samples: usize,
    last_error: Option<String>,
}

impl ToolStats {
    fn record(&mut self, ok: bool, latency_ms: Option<u64>, error: Option<&str>) {
        if ok {
            self.ok += 1;
        } else {
            self.failed += 1;
            if let Some(err) = error {
                if !err.trim().is_empty() {
                    self.last_error = Some(err.chars().take(100).collect());
                }
            }
        }
        if let Some(ms) = latency_ms {
            self.latency_total_ms += ms;
            self.latency_samples += 1;
        }
    }

    fn total(&self) -> usize {
        self.ok + self.failed
    }

    fn avg_latency_ms(&self) -> Option<u64> {
        (self.latency_samples > 0).then(|| self.latency_total_ms / self.latency_samples as u64)
    }
}

fn read_recent_learning_entries(workspace: &Path, limit: usize) -> Vec<LearningEntry> {
    let path = workspace.join("memory").join("learnings.jsonl");
    let Ok(data) = fs::read_to_string(path) else {
        return Vec::new();
    };

    let mut recent = VecDeque::with_capacity(limit.saturating_add(1));
    for line in data.lines() {
        let Ok(entry) = serde_json::from_str::<LearningEntry>(line) else {
            continue;
        };
        if entry.tool_name.trim().is_empty() {
            continue;
        }
        recent.push_back(entry);
        if recent.len() > limit {
            recent.pop_front();
        }
    }
    recent.into_iter().collect()
}

fn render_learning_status(workspace: &Path, recent_limit: usize, max_rows: usize) -> String {
    let entries = read_recent_learning_entries(workspace, recent_limit);
    if entries.is_empty() {
        return "## Learning\nNo learning entries found.".to_string();
    }

    let mut stats: BTreeMap<String, ToolStats> = BTreeMap::new();
    for entry in &entries {
        stats.entry(entry.tool_name.clone()).or_default().record(
            entry.succeeded,
            entry.latency_ms,
            entry.error.as_deref(),
        );
    }

    render_stats("Learning", &stats, max_rows, Some(entries.len()))
}

fn read_recent_audit_entries(workspace: &Path, limit: usize) -> Vec<AuditEntry> {
    let audit_dir = workspace.join("memory").join("audit");
    let Ok(read_dir) = fs::read_dir(audit_dir) else {
        return Vec::new();
    };

    let mut files: Vec<(SystemTime, PathBuf)> = read_dir
        .flatten()
        .filter_map(|entry| {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("jsonl") {
                return None;
            }
            if path
                .file_name()
                .and_then(|s| s.to_str())
                .map(|name| name.contains(".turns."))
                .unwrap_or(false)
            {
                return None;
            }
            let modified = entry
                .metadata()
                .and_then(|m| m.modified())
                .unwrap_or(UNIX_EPOCH);
            Some((modified, path))
        })
        .collect();
    files.sort_by(|a, b| b.0.cmp(&a.0));

    let mut entries = Vec::new();
    for (_, path) in files.into_iter().take(MAX_AUDIT_FILES) {
        let Ok(data) = fs::read_to_string(path) else {
            continue;
        };
        for line in data.lines() {
            if let Ok(entry) = serde_json::from_str::<AuditEntry>(line) {
                entries.push(entry);
            }
        }
    }

    entries.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
    if entries.len() > limit {
        entries.split_off(entries.len() - limit)
    } else {
        entries
    }
}

fn render_audit_status(workspace: &Path, recent_limit: usize, max_rows: usize) -> String {
    let entries = read_recent_audit_entries(workspace, recent_limit);
    if entries.is_empty() {
        return "## Audit\nNo audit entries found.".to_string();
    }

    let mut stats: BTreeMap<String, ToolStats> = BTreeMap::new();
    let mut executors: BTreeMap<String, usize> = BTreeMap::new();
    for entry in &entries {
        stats.entry(entry.tool_name.clone()).or_default().record(
            entry.result_ok,
            Some(entry.duration_ms),
            None,
        );
        *executors.entry(entry.executor.clone()).or_default() += 1;
    }

    let mut out = render_stats("Audit", &stats, max_rows, Some(entries.len()));
    if !executors.is_empty() {
        let mut parts: Vec<String> = executors
            .into_iter()
            .map(|(name, count)| format!("{}={}", name, count))
            .collect();
        parts.sort();
        out.push_str(&format!("\nExecutors: {}", parts.join(", ")));
    }
    out
}

fn render_skill_status(workspace: &Path, max_rows: usize) -> String {
    let loader = SkillsLoader::new(workspace, None);
    let results = loader.validate_all();
    if results.is_empty() {
        return "## Skill Validation\nNo skills found.".to_string();
    }

    let valid = results.iter().filter(|r| r.is_valid()).count();
    let error_count: usize = results.iter().map(|r| r.errors.len()).sum();
    let warning_count: usize = results.iter().map(|r| r.warnings.len()).sum();
    let mut out = format!(
        "## Skill Validation\n{} skill(s), {} valid, {} error(s), {} warning(s)",
        results.len(),
        valid,
        error_count,
        warning_count
    );

    let issue_rows: Vec<String> = results
        .iter()
        .filter(|r| !r.errors.is_empty() || !r.warnings.is_empty())
        .take(max_rows)
        .map(|r| {
            let mut issues: Vec<String> =
                r.errors.iter().map(|e| format!("ERROR: {}", e)).collect();
            issues.extend(r.warnings.iter().map(|w| format!("WARN: {}", w)));
            format!("- {}: {}", r.name, issues.join("; "))
        })
        .collect();
    if issue_rows.is_empty() {
        out.push_str("\nAll discovered skills have required metadata.");
    } else {
        out.push('\n');
        out.push_str(&issue_rows.join("\n"));
    }
    out
}

fn render_stats(
    title: &str,
    stats: &BTreeMap<String, ToolStats>,
    max_rows: usize,
    source_entries: Option<usize>,
) -> String {
    let calls: usize = stats.values().map(ToolStats::total).sum();
    let ok: usize = stats.values().map(|s| s.ok).sum();
    let failed: usize = stats.values().map(|s| s.failed).sum();
    let mut out = if let Some(entries) = source_entries {
        format!(
            "## {}\n{} recent entries, {} call(s), {} ok, {} failed",
            title, entries, calls, ok, failed
        )
    } else {
        format!(
            "## {}\n{} call(s), {} ok, {} failed",
            title, calls, ok, failed
        )
    };

    let mut rows: Vec<(&String, &ToolStats)> = stats.iter().collect();
    rows.sort_by(|a, b| b.1.total().cmp(&a.1.total()).then_with(|| a.0.cmp(b.0)));
    for (name, stat) in rows.into_iter().take(max_rows) {
        let mut line = format!("- {}: {}/{} ok", name, stat.ok, stat.total());
        if let Some(avg) = stat.avg_latency_ms() {
            line.push_str(&format!(", avg {}ms", avg));
        }
        if let Some(err) = &stat.last_error {
            line.push_str(&format!(", last error: {}", err));
        }
        out.push('\n');
        out.push_str(&line);
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_tool_status_reports_learning_and_skills() {
        let tmp = TempDir::new().unwrap();
        let memory = tmp.path().join("memory");
        fs::create_dir_all(&memory).unwrap();
        fs::write(
            memory.join("learnings.jsonl"),
            format!(
                "{}\n{}\n",
                json!({"timestamp":"2026-01-01T00:00:00Z","tool_name":"read_file","succeeded":true,"context":"x","latency_ms":10}),
                json!({"timestamp":"2026-01-01T00:00:01Z","tool_name":"exec","succeeded":false,"context":"x","error":"timeout","latency_ms":20})
            ),
        )
        .unwrap();
        let skill_dir = tmp.path().join("skills").join("status");
        fs::create_dir_all(&skill_dir).unwrap();
        fs::write(
            skill_dir.join("SKILL.md"),
            "---\ndescription: Show tool status\n---\nbody",
        )
        .unwrap();

        let tool = ToolStatusTool::new(tmp.path().to_path_buf());
        let out = tool.execute(std::collections::HashMap::new()).await;
        assert!(out.contains("## Learning"), "{out}");
        assert!(out.contains("read_file: 1/1 ok"), "{out}");
        assert!(out.contains("exec: 0/1 ok"), "{out}");
        assert!(out.contains("## Skill Validation"), "{out}");
        assert!(out.contains("1 skill(s), 1 valid"), "{out}");
    }

    #[tokio::test]
    async fn test_tool_status_reports_skill_metadata_errors() {
        let tmp = TempDir::new().unwrap();
        let skill_dir = tmp.path().join("skills").join("newsreader");
        fs::create_dir_all(&skill_dir).unwrap();
        fs::write(skill_dir.join("SKILL.md"), "# Newsreader\nbody").unwrap();

        let tool = ToolStatusTool::new(tmp.path().to_path_buf());
        let mut params = std::collections::HashMap::new();
        params.insert("action".to_string(), json!("skills"));
        let out = tool.execute(params).await;
        assert!(out.contains("newsreader"), "{out}");
        assert!(out.contains("Missing description"), "{out}");
    }
}
