// Error-protocol layer-3 backlog (docs/research/2026-08-06-error-conventions-and-host-bridge.md §3.6):
// the deny regime in Cargo.toml is live; this module still carries pre-existing
// violations of the lints below. Remove this allow as the module migrates onto
// the regime.
// Tracking: docs/error-protocol-backlog.md
#![allow(
    clippy::as_conversions,
    clippy::format_push_string,
    clippy::indexing_slicing,
    clippy::shadow_reuse,
    clippy::shadow_unrelated
)]
//! File system tools: read, write, edit, list, search, metadata, and diffs.
//!
//! Each tool lives in its own file (`read.rs`, `write.rs`, ...). This file
//! holds the shared helpers (`require_param`, `expand_path`, parameter
//! parsing, path resolution) that every tool uses.

mod write;

pub use write::WriteFileTool;
pub(crate) use write::MAX_WRITE_FILE_PIECE_CHARS;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use regex::RegexBuilder;
use sha2::{Digest, Sha256};
use tokio::process::Command;

use super::base::{PermissionLevel, Tool, ToolConcurrency};
use crate::agent::context_hygiene::TOOL_RESULT_REPLAY_MAX_BYTES;

/// Leave room below the replay ceiling for wrappers while retaining a useful
/// direct-file page. Crossing the ceiling stashes an otherwise contiguous
/// source window and replaces its native `lines=` continuation with a generic
/// artifact lookup, which costs both a tool round and prompt space.
const READ_FILE_REPLAY_SAFE_BYTES: usize = TOOL_RESULT_REPLAY_MAX_BYTES - 400;

/// Extract a required string parameter, returning an error string on missing.
fn require_param<'a>(
    params: &'a HashMap<String, serde_json::Value>,
    key: &str,
) -> Result<&'a str, String> {
    params
        .get(key)
        .and_then(|v| v.as_str())
        .ok_or_else(|| format!("Error: '{}' parameter is required", key))
}

/// Build a `HashMap<String, Value>` from `&str` pairs. Shared test helper used
/// by every tool's tests; declared at module level (not inside `mod tests`) so
/// that per-tool submodules can reach it via `super::super::make_params`.
#[cfg(test)]
fn make_params(pairs: &[(&str, &str)]) -> HashMap<String, serde_json::Value> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), serde_json::Value::String(v.to_string())))
        .collect()
}

// ---------------------------------------------------------------------------
// ReadFileTool
// ---------------------------------------------------------------------------

/// Default number of lines a bare `read_file` returns. Modeled on ds4's
/// `AGENT_READ_DEFAULT_LINES` (antirez/DwarfStar): reading defaults to a
/// bounded chunk, never the whole file, so the model only pulls more lines
/// (via the `lines` param) where it actually needs them. This keeps each read
/// a small, individually cacheable prefill instead of dumping an entire file
/// into context.
const DEFAULT_READ_LINES: usize = 1000;

/// Hard cap for a single bare read. Explicit `lines` ranges can still request
/// more when the model really needs it; this cap just prevents accidental dumps.
const MAX_READ_LINES: usize = 5000;

/// Tool to read file contents.
///
/// `char_budget` bounds the rendered line window so it stays UNDER the
/// tool-result cap (`max_tool_result_chars`). This is essential: if the window
/// exceeded the cap, `digest_tool_result` would head+tail-truncate it and
/// insert a `"[...]"` gap — read_file is randomly accessible, so a complete
/// window plus a next-range pointer always beats a hole. Production constructs
/// with `new(max_tool_result_chars)`; `Default` keeps the historical 7000 for
/// tests.
pub struct ReadFileTool {
    char_budget: usize,
}

impl ReadFileTool {
    pub fn new(char_budget: usize) -> Self {
        // Floor so an absurdly small cap still renders at least a few lines.
        Self {
            char_budget: char_budget.clamp(512, READ_FILE_REPLAY_SAFE_BYTES),
        }
    }
}

impl Default for ReadFileTool {
    fn default() -> Self {
        Self::new(7_000)
    }
}

#[async_trait]
impl Tool for ReadFileTool {
    fn name(&self) -> &str {
        "read_file"
    }

    fn description(&self) -> &str {
        "Read a file: returns the first chunk of numbered lines that fits the read budget (~7000 chars), plus the total line count and the exact next range to read. Output is always complete, contiguous lines — never a mid-cut — so page forward using the suggested next range (or lines=\"START:END\" for an exact 1-indexed inclusive range). Use lines=\"1:\" only on small files."
    }

    fn concurrency(&self) -> ToolConcurrency {
        ToolConcurrency::ParallelSafe
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The file path to read"
                },
                "lines": {
                    "type": "string",
                    "description": "Line range, e.g. \"10:50\" (1-indexed, inclusive). Omit for the first chunk; use \"1:\" for the whole file. The output header reports the total line count and the next range to read."
                },
                "max_lines": {
                    "type": "integer",
                    "description": "Number of lines for a bare read when lines is omitted. Default: 1000, max: 5000"
                }
            },
            "required": ["path"]
        })
    }

    async fn execute(&self, params: HashMap<String, serde_json::Value>) -> String {
        let path = match require_param(&params, "path") {
            Ok(p) => p,
            Err(e) => return e,
        };

        let file_path = resolve_read_path(path);

        if !file_path.exists() {
            return format!("Error: File not found: {}. Hint: verify the path exists. Use list_dir to browse the directory.", path);
        }
        if !file_path.is_file() {
            return format!("Error: Not a file: {}. Hint: this path is a directory, not a file. Use list_dir to see its contents.", path);
        }

        // Read raw bytes first for binary detection.
        let bytes = match tokio::fs::read(&file_path).await {
            Ok(b) => b,
            Err(e) => {
                return if e.kind() == std::io::ErrorKind::PermissionDenied {
                    format!("Error: Permission denied: {}. Hint: check file permissions or try a different path.", path)
                } else {
                    format!("Error reading file: {}", e)
                }
            }
        };

        // Binary detection: null bytes in first 512 bytes.
        let display_path = file_path.to_string_lossy().to_string();
        if crate::utils::helpers::is_binary(&bytes) {
            return format!("[Binary file: {}, {} bytes]", display_path, bytes.len());
        }

        let content = String::from_utf8_lossy(&bytes).to_string();
        let content_sha256 = sha256_hex(&bytes);
        let total = content.lines().count();
        if total == 0 {
            return format!("# {} (0 lines) sha256={}\n", display_path, content_sha256);
        }

        // Explicit range → render it. Bare read → first DEFAULT_READ_LINES,
        // ds4-style, so the model never dumps a whole file unless it asks
        // (lines="1:"). Both paths share the deterministic renderer below.
        if let Some(lines_param) = params.get("lines").and_then(|v| v.as_str()) {
            return extract_line_range(
                &content,
                lines_param,
                &display_path,
                self.char_budget,
                &content_sha256,
            );
        }
        let max_lines =
            bounded_usize_param(&params, "max_lines", DEFAULT_READ_LINES, MAX_READ_LINES);
        render_range(
            &content,
            1,
            max_lines.min(total),
            &display_path,
            total,
            self.char_budget,
            &content_sha256,
        )
    }
}

// ---------------------------------------------------------------------------
// WriteFileTool — see `write.rs`
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// EditFileTool
// ---------------------------------------------------------------------------

/// Explain why `old_text` didn't match `content` byte-for-byte by trying
/// progressively looser comparisons. Returns an error string that names the
/// specific whitespace issue when possible, so the model can fix its input
/// instead of re-reading the whole file.
fn diagnose_missing_old_text(content: &str, old_text: &str) -> String {
    if content
        .replace("\r\n", "\n")
        .contains(&old_text.replace("\r\n", "\n"))
    {
        return "Error: old_text not found — line endings differ (file uses CRLF, old_text uses LF or vice versa). Normalize to LF in old_text and retry.".to_string();
    }
    let strip_trailing = |s: &str| {
        s.lines()
            .map(|l| l.trim_end())
            .collect::<Vec<_>>()
            .join("\n")
    };
    if strip_trailing(content).contains(&strip_trailing(old_text)) {
        return "Error: old_text not found — trailing whitespace on one or more lines differs. Match trailing spaces/tabs exactly, or re-read the file first.".to_string();
    }
    "Error: old_text not found in file. Make sure it matches exactly. Hint: use read_file to see the current file contents, then copy the exact text to match.".to_string()
}

fn is_sha256_hex(value: &str) -> bool {
    let trimmed = value.trim();
    trimmed.len() == 64 && trimmed.bytes().all(|b| b.is_ascii_hexdigit())
}

/// Idle-turn write gate (v0.5 E1). Entry forms: workspace-relative subtree
/// ("skills/**"), workspace-relative exact file ("MEMORY.md"), or absolute
/// path, optionally with a "/**" subtree suffix. Only idle turns pass an
/// allowlist at all — a human watches normal turns.
///
/// Note: relative paths resolve against cwd first (see [`expand_path`]), so
/// an idle "MEMORY.md" with cwd outside the workspace resolves outside the
/// allowlist and is denied. Deny-over-surprise is the correct direction for
/// an unattended turn; memory maintenance goes through remember/recall.
pub(crate) fn idle_write_allowed(entries: &[String], target: &Path, workspace: &Path) -> bool {
    let mut allowed = false;
    for entry in entries {
        let (raw, subtree) = match entry.strip_suffix("/**") {
            Some(dir) => (dir, true),
            None => (entry.as_str(), false),
        };
        if raw.is_empty() {
            continue;
        }
        let base = Path::new(raw);
        let matched = if base.is_absolute() {
            subtree && target.starts_with(base) || (!subtree && target == base)
        } else {
            let ws = workspace.join(base);
            subtree && target.starts_with(&ws) || (!subtree && target == ws)
        };
        if matched {
            allowed = true;
            break;
        }
    }
    allowed
}

/// Consistent denial message for idle-turn writes outside the allowlist.
pub(crate) fn idle_write_denied(target: &Path) -> String {
    format!(
        "Error: idle turns may only write to configured paths (idle.writePaths); denied: {}. Use the message tool to ask the user for wider access.",
        target.display()
    )
}

/// Tool to edit a file by replacing text.
pub struct EditFileTool {
    /// Idle-turn write allowlist; `None` on normal turns.
    pub idle_paths: Option<Vec<String>>,
}

impl Default for EditFileTool {
    fn default() -> Self {
        Self { idle_paths: None }
    }
}

impl EditFileTool {
    pub fn new(idle_paths: Option<Vec<String>>) -> Self {
        Self { idle_paths }
    }
}

#[async_trait]
impl Tool for EditFileTool {
    fn name(&self) -> &str {
        "edit_file"
    }

    fn description(&self) -> &str {
        "Edit a file by replacing old_text with new_text, or apply a unified diff in patch. Use patch for multi-line edits when exact old_text matching would be brittle. Optional expected_sha256 prevents overwriting a file that changed since file_info/read_file."
    }

    fn permission(&self) -> PermissionLevel {
        PermissionLevel::Write
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The file path to edit"
                },
                "old_text": {
                    "type": "string",
                    "description": "The exact text to find and replace"
                },
                "new_text": {
                    "type": "string",
                    "description": "The text to replace with"
                },
                "patch": {
                    "type": "string",
                    "description": "Unified diff hunk(s) to apply to this file, e.g. @@ -1,2 +1,2 @@. If patch is provided, old_text/new_text are ignored."
                },
                "expected_sha256": {
                    "type": "string",
                    "description": "Optional SHA-256 of the current file contents. The edit is rejected if the file hash differs, which catches concurrent edits."
                }
            },
            "required": ["path"]
        })
    }

    async fn execute(&self, params: HashMap<String, serde_json::Value>) -> String {
        let path = match require_param(&params, "path") {
            Ok(p) => p,
            Err(e) => return e,
        };

        let file_path = expand_path(path);
        if let Some(paths) = &self.idle_paths {
            let workspace = crate::utils::helpers::get_workspace_path(None);
            if !idle_write_allowed(paths, &file_path, &workspace) {
                return idle_write_denied(&file_path);
            }
        }

        if !file_path.exists() {
            return format!("Error: File not found: {}. Hint: verify the path exists. Use list_dir to browse the directory.", path);
        }

        let content = match tokio::fs::read_to_string(&file_path).await {
            Ok(c) => c,
            Err(e) => return format!("Error reading file: {}", e),
        };

        if let Some(expected) = params.get("expected_sha256").and_then(|v| v.as_str()) {
            if !is_sha256_hex(expected) {
                return format!(
                    "Error: invalid expected_sha256 '{}'. Omit expected_sha256 unless you copied the 64-character sha256 value from read_file or file_info.",
                    expected.trim()
                );
            }
            let actual = sha256_hex(content.as_bytes());
            if !expected.trim().eq_ignore_ascii_case(&actual) {
                return format!(
                    "Error: File changed before edit. expected_sha256={}, actual_sha256={}. Re-read the file or inspect workspace_diff before retrying; omit expected_sha256 if you did not copy it from read_file or file_info.",
                    expected.trim(),
                    actual
                );
            }
        }

        if let Some(patch) = params.get("patch").and_then(|v| v.as_str()) {
            if patch.trim().is_empty() {
                return "Error: 'patch' parameter cannot be empty".to_string();
            }
            let (new_content, hunks) =
                match super::apply_patch::apply_unified_patch_to_content(&content, patch) {
                    Ok(result) => result,
                    Err(e) => return e,
                };
            return match tokio::fs::write(&file_path, new_content).await {
                Ok(()) => format!("Successfully patched {} ({} hunk(s))", path, hunks),
                Err(e) => {
                    if e.kind() == std::io::ErrorKind::PermissionDenied {
                        format!("Error: Permission denied: {}. Hint: check file permissions or try a different path.", path)
                    } else {
                        format!("Error writing file: {}", e)
                    }
                }
            };
        }

        let old_text = match params.get("old_text").and_then(|v| v.as_str()) {
            Some(t) => t,
            None => {
                return "Error: either 'patch' or both 'old_text' and 'new_text' are required"
                    .to_string()
            }
        };
        let new_text = match params.get("new_text").and_then(|v| v.as_str()) {
            Some(t) => t,
            None => {
                return "Error: either 'patch' or both 'old_text' and 'new_text' are required"
                    .to_string()
            }
        };
        if old_text == new_text {
            return "Error: old_text and new_text are identical; no change was made. Provide different replacement text, or call write_file with state=append to add a suffix."
                .to_string();
        }

        if !content.contains(old_text) {
            return diagnose_missing_old_text(&content, old_text);
        }

        // Count occurrences.
        let count = content.matches(old_text).count();
        if count > 1 {
            return format!(
                "Error: old_text appears {} times. Please provide more context to make it unique.",
                count
            );
        }

        let new_content = content.replacen(old_text, new_text, 1);

        match tokio::fs::write(&file_path, new_content).await {
            Ok(()) => format!("Successfully edited {}", path),
            Err(e) => {
                if e.kind() == std::io::ErrorKind::PermissionDenied {
                    format!("Error: Permission denied: {}. Hint: check file permissions or try a different path.", path)
                } else {
                    format!("Error writing file: {}", e)
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ListDirTool
// ---------------------------------------------------------------------------

/// Tool to list directory contents.
pub struct ListDirTool;

#[async_trait]
impl Tool for ListDirTool {
    fn name(&self) -> &str {
        "list_dir"
    }

    fn description(&self) -> &str {
        "List the contents of a directory."
    }

    fn concurrency(&self) -> ToolConcurrency {
        ToolConcurrency::ParallelSafe
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The directory path to list"
                }
            },
            "required": ["path"]
        })
    }

    async fn execute(&self, params: HashMap<String, serde_json::Value>) -> String {
        let path = match require_param(&params, "path") {
            Ok(p) => p,
            Err(e) => return e,
        };

        let dir_path = expand_path(path);

        if !dir_path.exists() {
            return format!("Error: Directory not found: {}. Hint: parent directory does not exist. Use list_dir to find the correct path.", path);
        }
        if !dir_path.is_dir() {
            return format!("Error: Not a directory: {}. Hint: this path is a file, not a directory. Use read_file instead.", path);
        }

        // Detect when the model is listing the workspace instead of the project.
        let workspace = crate::utils::helpers::get_workspace_path(None);
        let is_workspace = dir_path
            .canonicalize()
            .ok()
            .and_then(|canon| {
                workspace
                    .canonicalize()
                    .ok()
                    .map(|ws| canon.starts_with(ws))
            })
            .unwrap_or(false);

        match tokio::fs::read_dir(&dir_path).await {
            Ok(mut entries) => {
                let mut items: Vec<(bool, String)> = Vec::new();

                loop {
                    match entries.next_entry().await {
                        Ok(Some(entry)) => {
                            let name = entry.file_name().to_string_lossy().to_string();
                            let is_dir = entry
                                .file_type()
                                .await
                                .map(|ft| ft.is_dir())
                                .unwrap_or(false);
                            items.push((is_dir, name));
                        }
                        Ok(None) => break,
                        Err(e) => return format!("Error reading directory: {}", e),
                    }
                }

                if items.is_empty() {
                    return format!("Directory {} is empty", path);
                }

                // Sort alphabetically.
                items.sort_by(|a, b| a.1.cmp(&b.1));

                let lines: Vec<String> = items
                    .into_iter()
                    .map(|(is_dir, name)| {
                        if is_dir {
                            format!("[dir]  {}", name)
                        } else {
                            format!("[file] {}", name)
                        }
                    })
                    .collect();

                let mut output = lines.join("\n");

                if is_workspace {
                    let cwd = std::env::current_dir()
                        .map(|p| p.display().to_string())
                        .unwrap_or_else(|_| ".".to_string());
                    output.push_str(&format!(
                        "\n\nNote: this is your internal workspace (memory, skills, config). \
                         The user's project is at: {cwd} - use list_dir on that path instead."
                    ));
                }

                output
            }
            Err(e) => {
                if e.kind() == std::io::ErrorKind::PermissionDenied {
                    format!("Error: Permission denied: {}. Hint: check file permissions or try a different path.", path)
                } else {
                    format!("Error listing directory: {}", e)
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// FindFilesTool
// ---------------------------------------------------------------------------

/// Tool to recursively find files/directories without requiring shell `find`.
pub struct FindFilesTool;

#[async_trait]
impl Tool for FindFilesTool {
    fn name(&self) -> &str {
        "find_files"
    }

    fn description(&self) -> &str {
        "Recursively find files or directories under a path. Supports simple glob patterns with * and ?, depth/limit bounds, and an optional tree view."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory to search. Defaults to the current working directory."
                },
                "pattern": {
                    "type": "string",
                    "description": "File name or relative-path pattern. Supports * and ?. Plain text matches as a case-insensitive substring. Default: *"
                },
                "kind": {
                    "type": "string",
                    "enum": ["file", "dir", "all"],
                    "description": "Return files, directories, or both. Default: file"
                },
                "max_depth": {
                    "type": "integer",
                    "description": "Maximum recursive depth below path. Direct children are depth 1. Default: 5, max: 20"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum matches to return. Default: 200, max: 1000"
                },
                "include_hidden": {
                    "type": "boolean",
                    "description": "Include dotfiles and hidden directories. Default: false"
                },
                "tree": {
                    "type": "boolean",
                    "description": "Render matches as an indented tree. Default: false"
                }
            }
        })
    }

    async fn execute(&self, params: HashMap<String, serde_json::Value>) -> String {
        let path = params.get("path").and_then(|v| v.as_str()).unwrap_or(".");
        let pattern = params
            .get("pattern")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .unwrap_or("*");
        let kind = params
            .get("kind")
            .and_then(|v| v.as_str())
            .unwrap_or("file");
        if !matches!(kind, "file" | "dir" | "all") {
            return "Error: 'kind' must be one of: file, dir, all".to_string();
        }

        let max_depth = bounded_usize_param(&params, "max_depth", 5, 20);
        let limit = bounded_usize_param(&params, "limit", 200, 1000);
        let include_hidden = bool_param(&params, "include_hidden", false);
        let tree = bool_param(&params, "tree", false);

        let root = expand_path(path);
        if !root.exists() {
            return format!(
                "Error: Directory not found: {}. Hint: verify the path or use list_dir on its parent.",
                path
            );
        }
        if !root.is_dir() {
            return format!(
                "Error: Not a directory: {}. Hint: use file_info or read_file for file paths.",
                path
            );
        }

        let root_canon = root.canonicalize().unwrap_or(root.clone());
        let mut stack = vec![(root.clone(), 0usize)];
        let mut matches = Vec::new();

        while let Some((dir, depth)) = stack.pop() {
            if depth >= max_depth {
                continue;
            }

            let mut children = match std::fs::read_dir(&dir) {
                Ok(rd) => rd
                    .filter_map(Result::ok)
                    .collect::<Vec<std::fs::DirEntry>>(),
                Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => continue,
                Err(e) => return format!("Error reading directory {}: {}", dir.display(), e),
            };
            children.sort_by_key(|e| e.file_name());

            for entry in children {
                let name = entry.file_name().to_string_lossy().to_string();
                if !include_hidden && name.starts_with('.') {
                    continue;
                }
                let entry_path = entry.path();
                let file_type = match entry.file_type() {
                    Ok(ft) => ft,
                    Err(_) => continue,
                };
                let is_dir = file_type.is_dir();
                let child_depth = depth + 1;

                let rel = entry_path
                    .strip_prefix(&root_canon)
                    .or_else(|_| entry_path.strip_prefix(&root))
                    .unwrap_or(&entry_path)
                    .to_string_lossy()
                    .replace('\\', "/");

                let kind_matches = match kind {
                    "file" => file_type.is_file(),
                    "dir" => is_dir,
                    "all" => true,
                    _ => false,
                };
                if kind_matches && pattern_matches(pattern, &name, &rel) {
                    let size = entry.metadata().ok().map(|m| m.len()).unwrap_or(0);
                    matches.push(FindMatch {
                        rel,
                        depth: child_depth,
                        is_dir,
                        size,
                    });
                }

                if is_dir && child_depth < max_depth {
                    stack.push((entry_path, child_depth));
                }
            }
        }

        matches.sort_by(|a, b| a.rel.cmp(&b.rel));
        let total = matches.len();
        let shown = total.min(limit);
        if total == 0 {
            return format!("No matches under {} for pattern=\"{}\".", path, pattern);
        }
        let mut out = format!("{} matches (showing {}):", total, shown);
        for item in matches.iter().take(limit) {
            out.push('\n');
            if tree {
                let indent = "  ".repeat(item.depth.saturating_sub(1));
                out.push_str(&format!(
                    "{}{}{}",
                    indent,
                    item.rel,
                    if item.is_dir { "/" } else { "" }
                ));
            } else if item.is_dir {
                out.push_str(&format!("[dir]  {}", item.rel));
            } else {
                out.push_str(&format!("[file] {} ({} bytes)", item.rel, item.size));
            }
        }
        if shown < total {
            out.push_str(&format!("\n[{} more matches not shown]", total - shown));
        }
        out
    }
}

#[derive(Debug)]
struct FindMatch {
    rel: String,
    depth: usize,
    is_dir: bool,
    size: u64,
}

// ---------------------------------------------------------------------------
// SearchFilesTool
// ---------------------------------------------------------------------------

/// Tool to recursively search text file contents without requiring shell grep.
pub struct SearchFilesTool;

#[async_trait]
impl Tool for SearchFilesTool {
    fn name(&self) -> &str {
        "search_files"
    }

    fn description(&self) -> &str {
        "Recursively search file contents under a directory. Supports plain text or regex queries, file glob filters, context lines, depth/size/limit bounds, and skips common vendor/build directories by default."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory to search. Defaults to the current working directory."
                },
                "query": {
                    "type": "string",
                    "description": "Text or regex pattern to search for"
                },
                "regex": {
                    "type": "boolean",
                    "description": "Treat query as a regular expression. Default: false"
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Use case-sensitive matching. Default: false"
                },
                "pattern": {
                    "type": "string",
                    "description": "File name or relative-path glob/substring filter. Supports * and ?. Default: *"
                },
                "exclude_pattern": {
                    "type": "string",
                    "description": "Optional file name or relative-path glob/substring to skip"
                },
                "max_depth": {
                    "type": "integer",
                    "description": "Maximum recursive depth below path. Direct children are depth 1. Default: 8, max: 30"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum matching lines to return. Default: 200, max: 1000"
                },
                "context": {
                    "type": "integer",
                    "description": "Context lines before/after each match. Default: 0, max: 5"
                },
                "max_file_bytes": {
                    "type": "integer",
                    "description": "Skip files larger than this many bytes. Default: 1048576, max: 10485760"
                },
                "include_hidden": {
                    "type": "boolean",
                    "description": "Include dotfiles and hidden directories. Default: false"
                },
                "skip_vendor": {
                    "type": "boolean",
                    "description": "Skip common heavy directories like target, node_modules, dist, build, and vendor. Default: true"
                }
            },
            "required": ["query"]
        })
    }

    async fn execute(&self, params: HashMap<String, serde_json::Value>) -> String {
        let query = match require_param(&params, "query") {
            Ok(q) if !q.trim().is_empty() => q.trim(),
            Ok(_) => return "Error: 'query' parameter cannot be empty".to_string(),
            Err(e) => return e,
        };
        let path = params.get("path").and_then(|v| v.as_str()).unwrap_or(".");
        let pattern = params
            .get("pattern")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .unwrap_or("*");
        let exclude_pattern = params
            .get("exclude_pattern")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty());
        let regex = bool_param(&params, "regex", false);
        let case_sensitive = bool_param(&params, "case_sensitive", false);
        let include_hidden = bool_param(&params, "include_hidden", false);
        let skip_vendor = bool_param(&params, "skip_vendor", true);
        let max_depth = bounded_usize_param(&params, "max_depth", 8, 30);
        let limit = bounded_usize_param(&params, "limit", 200, 1000);
        let context = bounded_usize_param(&params, "context", 0, 5);
        let max_file_bytes = bounded_u64_param(&params, "max_file_bytes", 1_048_576, 10_485_760);

        let matcher = match SearchMatcher::new(query, regex, case_sensitive) {
            Ok(m) => m,
            Err(e) => return e,
        };

        let root = expand_path(path);
        if !root.exists() {
            return format!(
                "Error: Directory not found: {}. Hint: verify the path or use list_dir on its parent.",
                path
            );
        }
        if !root.is_dir() {
            return format!(
                "Error: Not a directory: {}. Hint: use read_file for a single file.",
                path
            );
        }

        let root_canon = root.canonicalize().unwrap_or(root.clone());
        let mut stack = vec![(root.clone(), 0usize)];
        let mut files_searched = 0usize;
        let mut skipped_binary = 0usize;
        let mut skipped_large = 0usize;
        let mut skipped_unreadable = 0usize;
        let mut matched_lines = 0usize;
        let mut out_lines: Vec<String> = Vec::new();
        let mut limit_hit = false;

        'walk: while let Some((dir, depth)) = stack.pop() {
            if depth >= max_depth {
                continue;
            }

            let mut children = match std::fs::read_dir(&dir) {
                Ok(rd) => rd
                    .filter_map(Result::ok)
                    .collect::<Vec<std::fs::DirEntry>>(),
                Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => {
                    skipped_unreadable += 1;
                    continue;
                }
                Err(e) => return format!("Error reading directory {}: {}", dir.display(), e),
            };
            children.sort_by_key(|e| e.file_name());

            for entry in children {
                let name = entry.file_name().to_string_lossy().to_string();
                if !include_hidden && name.starts_with('.') {
                    continue;
                }
                if skip_vendor && is_common_vendor_dir(&name) {
                    continue;
                }

                let entry_path = entry.path();
                let file_type = match entry.file_type() {
                    Ok(ft) => ft,
                    Err(_) => {
                        skipped_unreadable += 1;
                        continue;
                    }
                };
                let child_depth = depth + 1;
                if file_type.is_dir() {
                    if child_depth < max_depth {
                        stack.push((entry_path, child_depth));
                    }
                    continue;
                }
                if !file_type.is_file() {
                    continue;
                }

                let rel = relative_display_path(&entry_path, &root_canon, &root);
                if !pattern_matches(pattern, &name, &rel) {
                    continue;
                }
                if exclude_pattern
                    .map(|p| pattern_matches(p, &name, &rel))
                    .unwrap_or(false)
                {
                    continue;
                }

                let metadata = match entry.metadata() {
                    Ok(m) => m,
                    Err(_) => {
                        skipped_unreadable += 1;
                        continue;
                    }
                };
                if metadata.len() > max_file_bytes {
                    skipped_large += 1;
                    continue;
                }

                let bytes = match std::fs::read(&entry_path) {
                    Ok(b) => b,
                    Err(_) => {
                        skipped_unreadable += 1;
                        continue;
                    }
                };
                if crate::utils::helpers::is_binary(&bytes) {
                    skipped_binary += 1;
                    continue;
                }

                files_searched += 1;
                let content = String::from_utf8_lossy(&bytes);
                let lines: Vec<&str> = content.lines().collect();
                for (idx, line) in lines.iter().enumerate() {
                    if !matcher.is_match(line) {
                        continue;
                    }
                    matched_lines += 1;
                    append_search_hit(&mut out_lines, &rel, &lines, idx, context);
                    if matched_lines >= limit {
                        limit_hit = true;
                        break 'walk;
                    }
                }
            }
        }

        let shown = out_lines.len().min(limit);
        let mut out = if out_lines.is_empty() {
            format!("No matches for {:?} under {}.", query, path)
        } else {
            format!(
                "Found {} matching line(s) in {} file(s); showing {}",
                matched_lines, files_searched, shown
            )
        };
        if skipped_large > 0 || skipped_binary > 0 || skipped_unreadable > 0 {
            out.push_str(&format!(
                "\nSkipped: {} large, {} binary, {} unreadable",
                skipped_large, skipped_binary, skipped_unreadable
            ));
        }
        if !out_lines.is_empty() {
            out.push('\n');
            out.push_str(&out_lines.join("\n"));
        }
        if limit_hit {
            out.push_str(&format!(
                "\n[limit reached at {} matching line(s); narrow pattern/path or raise limit]",
                limit
            ));
        }
        out
    }
}

enum SearchMatcher {
    Plain {
        needle: String,
        case_sensitive: bool,
    },
    Regex(regex::Regex),
}

impl SearchMatcher {
    fn new(query: &str, regex: bool, case_sensitive: bool) -> Result<Self, String> {
        if regex {
            return RegexBuilder::new(query)
                .case_insensitive(!case_sensitive)
                .build()
                .map(Self::Regex)
                .map_err(|e| format!("Error: invalid regex query: {}", e));
        }
        Ok(Self::Plain {
            needle: if case_sensitive {
                query.to_string()
            } else {
                query.to_ascii_lowercase()
            },
            case_sensitive,
        })
    }

    fn is_match(&self, line: &str) -> bool {
        match self {
            Self::Plain {
                needle,
                case_sensitive,
            } => {
                if *case_sensitive {
                    line.contains(needle)
                } else {
                    line.to_ascii_lowercase().contains(needle)
                }
            }
            Self::Regex(re) => re.is_match(line),
        }
    }
}

fn append_search_hit(out: &mut Vec<String>, rel: &str, lines: &[&str], idx: usize, context: usize) {
    let start = idx.saturating_sub(context);
    let end = (idx + context + 1).min(lines.len());
    for line_idx in start..end {
        let sep = if line_idx == idx { ":" } else { "-" };
        out.push(format!(
            "{}:{}{} {}",
            rel,
            line_idx + 1,
            sep,
            truncate_line(lines[line_idx], 300)
        ));
    }
}

// ---------------------------------------------------------------------------
// FileInfoTool
// ---------------------------------------------------------------------------

/// Tool to inspect file metadata and optional content hash.
pub struct FileInfoTool;

#[async_trait]
impl Tool for FileInfoTool {
    fn name(&self) -> &str {
        "file_info"
    }

    fn description(&self) -> &str {
        "Return metadata for a file or directory: type, size, permissions, timestamps, and SHA-256 for files. Use expected_sha256 with edit_file to catch concurrent edits."
    }

    fn concurrency(&self) -> ToolConcurrency {
        ToolConcurrency::ParallelSafe
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File or directory path to inspect"
                },
                "hash": {
                    "type": "boolean",
                    "description": "Compute SHA-256 for regular files. Default: true"
                }
            },
            "required": ["path"]
        })
    }

    async fn execute(&self, params: HashMap<String, serde_json::Value>) -> String {
        let path = match require_param(&params, "path") {
            Ok(p) => p,
            Err(e) => return e,
        };
        let include_hash = bool_param(&params, "hash", true);
        let file_path = expand_path(path);

        let metadata = match tokio::fs::symlink_metadata(&file_path).await {
            Ok(m) => m,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return format!(
                    "Error: Path not found: {}. Hint: use find_files to locate it.",
                    path
                )
            }
            Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => {
                return format!(
                    "Error: Permission denied: {}. Hint: check file permissions.",
                    path
                )
            }
            Err(e) => return format!("Error reading metadata: {}", e),
        };

        let kind = if metadata.file_type().is_symlink() {
            "symlink"
        } else if metadata.is_file() {
            "file"
        } else if metadata.is_dir() {
            "directory"
        } else {
            "other"
        };

        let mut out = format!(
            "Path: {}\nResolved: {}\nType: {}\nSize: {} bytes\nReadonly: {}",
            path,
            file_path.display(),
            kind,
            metadata.len(),
            metadata.permissions().readonly()
        );

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            out.push_str(&format!(
                "\nMode: {:o}",
                metadata.permissions().mode() & 0o7777
            ));
        }

        if let Ok(modified) = metadata.modified() {
            out.push_str(&format!("\nModified: {}", format_system_time(modified)));
            out.push_str(&format!(
                "\nModified_unix: {}",
                unix_timestamp_secs(modified).unwrap_or(0)
            ));
        }
        if let Ok(created) = metadata.created() {
            out.push_str(&format!("\nCreated: {}", format_system_time(created)));
        }
        if let Ok(accessed) = metadata.accessed() {
            out.push_str(&format!("\nAccessed: {}", format_system_time(accessed)));
        }

        if include_hash && metadata.is_file() {
            match tokio::fs::read(&file_path).await {
                Ok(bytes) => out.push_str(&format!("\nSHA-256: {}", sha256_hex(&bytes))),
                Err(e) => out.push_str(&format!("\nSHA-256: Error reading file: {}", e)),
            }
        }

        out
    }
}

// ---------------------------------------------------------------------------
// WorkspaceDiffTool
// ---------------------------------------------------------------------------

/// Tool to summarize git status/diff without making the model parse shell output.
pub struct WorkspaceDiffTool;

#[async_trait]
impl Tool for WorkspaceDiffTool {
    fn name(&self) -> &str {
        "workspace_diff"
    }

    fn description(&self) -> &str {
        "Show what changed in the current git workspace: status, staged/unstaged diff stats, and optionally the patch. Use after edits or before reporting completion."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Git repo, subdirectory, or file to inspect. Defaults to current directory."
                },
                "include_diff": {
                    "type": "boolean",
                    "description": "Include full staged and unstaged patches, truncated by max_chars. Default: false"
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Maximum characters when include_diff=true. Default: 12000, max: 50000"
                }
            }
        })
    }

    async fn execute(&self, params: HashMap<String, serde_json::Value>) -> String {
        let path = params.get("path").and_then(|v| v.as_str()).unwrap_or(".");
        let include_diff = bool_param(&params, "include_diff", false);
        let max_chars = bounded_usize_param(&params, "max_chars", 12_000, 50_000);

        let requested = expand_path(path);
        let probe = if requested.is_file() {
            requested
                .parent()
                .map(Path::to_path_buf)
                .unwrap_or_else(|| PathBuf::from("."))
        } else {
            requested.clone()
        };
        if !probe.exists() {
            return format!(
                "Error: Path not found: {}. Hint: use find_files or list_dir first.",
                path
            );
        }

        let root_text =
            match run_git_command(&probe, vec!["rev-parse".into(), "--show-toplevel".into()]).await
            {
                Ok(out) => out.trim().to_string(),
                Err(e) => return format!("Error: Not a git workspace or git unavailable: {}", e),
            };
        let root = PathBuf::from(root_text);
        let pathspec = git_pathspec(&root, &requested);

        let mut out = format!("Git root: {}", root.display());
        if let Some(ref spec) = pathspec {
            out.push_str(&format!("\nPath filter: {}", spec));
        }

        let status = run_git_with_optional_pathspec(
            &root,
            vec!["status".into(), "--short".into()],
            pathspec.as_deref(),
        )
        .await
        .unwrap_or_else(|e| format!("Error: {}", e));
        out.push_str("\n\n## Status\n");
        out.push_str(if status.trim().is_empty() {
            "clean"
        } else {
            status.trim_end()
        });

        let unstaged_stat = run_git_with_optional_pathspec(
            &root,
            vec!["diff".into(), "--stat".into()],
            pathspec.as_deref(),
        )
        .await
        .unwrap_or_else(|e| format!("Error: {}", e));
        out.push_str("\n\n## Unstaged Diff Stat\n");
        out.push_str(if unstaged_stat.trim().is_empty() {
            "none"
        } else {
            unstaged_stat.trim_end()
        });

        let staged_stat = run_git_with_optional_pathspec(
            &root,
            vec!["diff".into(), "--cached".into(), "--stat".into()],
            pathspec.as_deref(),
        )
        .await
        .unwrap_or_else(|e| format!("Error: {}", e));
        out.push_str("\n\n## Staged Diff Stat\n");
        out.push_str(if staged_stat.trim().is_empty() {
            "none"
        } else {
            staged_stat.trim_end()
        });

        if include_diff {
            let unstaged =
                run_git_with_optional_pathspec(&root, vec!["diff".into()], pathspec.as_deref())
                    .await
                    .unwrap_or_else(|e| format!("Error: {}", e));
            let staged = run_git_with_optional_pathspec(
                &root,
                vec!["diff".into(), "--cached".into()],
                pathspec.as_deref(),
            )
            .await
            .unwrap_or_else(|e| format!("Error: {}", e));
            let patches = format!(
                "\n\n## Unstaged Diff\n{}\n\n## Staged Diff\n{}",
                if unstaged.trim().is_empty() {
                    "none"
                } else {
                    unstaged.trim_end()
                },
                if staged.trim().is_empty() {
                    "none"
                } else {
                    staged.trim_end()
                }
            );
            out.push_str(&truncate_chars_with_notice(&patches, max_chars));
        }

        out
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn bool_param(params: &HashMap<String, serde_json::Value>, key: &str, default: bool) -> bool {
    params.get(key).and_then(|v| v.as_bool()).unwrap_or(default)
}

fn bounded_usize_param(
    params: &HashMap<String, serde_json::Value>,
    key: &str,
    default: usize,
    max: usize,
) -> usize {
    params
        .get(key)
        .and_then(|v| v.as_u64())
        .map(|v| (v as usize).clamp(1, max))
        .unwrap_or(default)
}

fn bounded_u64_param(
    params: &HashMap<String, serde_json::Value>,
    key: &str,
    default: u64,
    max: u64,
) -> u64 {
    params
        .get(key)
        .and_then(|v| v.as_u64())
        .map(|v| v.clamp(1, max))
        .unwrap_or(default)
}

pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn unix_timestamp_secs(time: SystemTime) -> Option<u64> {
    time.duration_since(UNIX_EPOCH).ok().map(|d| d.as_secs())
}

fn format_system_time(time: SystemTime) -> String {
    let datetime: chrono::DateTime<chrono::Utc> = time.into();
    datetime.to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

fn pattern_matches(pattern: &str, name: &str, rel: &str) -> bool {
    if pattern == "*" || pattern.is_empty() {
        return true;
    }
    if pattern.contains('*') || pattern.contains('?') {
        return glob_match(pattern, name) || glob_match(pattern, rel);
    }
    let needle = pattern.to_ascii_lowercase();
    name.to_ascii_lowercase().contains(&needle) || rel.to_ascii_lowercase().contains(&needle)
}

fn relative_display_path(path: &Path, root_canon: &Path, root: &Path) -> String {
    path.strip_prefix(root_canon)
        .or_else(|_| path.strip_prefix(root))
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

fn is_common_vendor_dir(name: &str) -> bool {
    matches!(
        name,
        "target"
            | "node_modules"
            | "vendor"
            | "dist"
            | "build"
            | "coverage"
            | ".git"
            | ".hg"
            | ".svn"
            | ".venv"
            | "venv"
            | ".next"
    )
}

fn truncate_line(line: &str, max_chars: usize) -> String {
    if line.chars().count() <= max_chars {
        return line.to_string();
    }
    let mut out: String = line.chars().take(max_chars).collect();
    out.push_str("...");
    out
}

fn glob_match(pattern: &str, text: &str) -> bool {
    fn inner(pattern: &[u8], text: &[u8]) -> bool {
        match (pattern.first(), text.first()) {
            (None, None) => true,
            (None, Some(_)) => false,
            (Some(b'*'), _) => {
                inner(&pattern[1..], text) || (!text.is_empty() && inner(pattern, &text[1..]))
            }
            (Some(b'?'), Some(_)) => inner(&pattern[1..], &text[1..]),
            (Some(a), Some(b)) if a.eq_ignore_ascii_case(b) => inner(&pattern[1..], &text[1..]),
            _ => false,
        }
    }
    inner(pattern.as_bytes(), text.as_bytes())
}

fn git_pathspec(root: &Path, requested: &Path) -> Option<String> {
    let absolute = if requested.is_absolute() {
        requested.to_path_buf()
    } else {
        std::env::current_dir()
            .map(|cwd| cwd.join(requested))
            .unwrap_or_else(|_| requested.to_path_buf())
    };
    let absolute = absolute.canonicalize().unwrap_or(absolute);
    if absolute == root {
        return None;
    }
    absolute
        .strip_prefix(root)
        .ok()
        .map(|p| p.to_string_lossy().replace('\\', "/"))
        .filter(|s| !s.is_empty())
}

async fn run_git_with_optional_pathspec(
    cwd: &Path,
    mut args: Vec<String>,
    pathspec: Option<&str>,
) -> Result<String, String> {
    if let Some(spec) = pathspec {
        args.push("--".to_string());
        args.push(spec.to_string());
    }
    run_git_command(cwd, args).await
}

async fn run_git_command(cwd: &Path, args: Vec<String>) -> Result<String, String> {
    let output = tokio::time::timeout(
        Duration::from_secs(10),
        Command::new("git")
            .args(&args)
            .current_dir(cwd)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .output(),
    )
    .await
    .map_err(|_| "git command timed out".to_string())?
    .map_err(|e| e.to_string())?;

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    if output.status.success() {
        Ok(stdout)
    } else {
        Err(if stderr.trim().is_empty() {
            stdout.trim().to_string()
        } else {
            stderr.trim().to_string()
        })
    }
}

fn truncate_chars_with_notice(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max_chars).collect();
    out.push_str("\n...[truncated]");
    out
}

/// Extract a line range from content.
///
/// `range` format: "start:end" (1-indexed, inclusive) or "start:" (to end).
fn extract_line_range(
    content: &str,
    range: &str,
    path: &str,
    char_budget: usize,
    sha256: &str,
) -> String {
    let parts: Vec<&str> = range.splitn(2, ':').collect();
    if parts.len() != 2 {
        return format!(
            "Error: Invalid lines format '{}'. Use 'start:end' (e.g. '10:50').",
            range
        );
    }

    let start: usize = match parts[0].trim().parse() {
        Ok(n) if n >= 1 => n,
        _ => {
            return format!(
                "Error: Invalid start line '{}'. Must be a positive integer.",
                parts[0]
            )
        }
    };

    let lines: Vec<&str> = content.lines().collect();
    let total = lines.len();

    let end: usize = if parts[1].trim().is_empty() {
        total
    } else {
        match parts[1].trim().parse::<usize>() {
            Ok(n) => n.min(total),
            _ => {
                return format!(
                    "Error: Invalid end line '{}'. Must be a positive integer.",
                    parts[1]
                )
            }
        }
    };

    if start > total {
        return format!(
            "Error: Start line {} exceeds file length ({} lines).",
            start, total
        );
    }
    if start > end {
        return format!("Error: Start line {} is after end line {}.", start, end);
    }

    render_range(content, start, end, path, total, char_budget, sha256)
}

/// Render a 1-indexed inclusive line range with line numbers, a header
/// reporting the file's total length, and — when the range stops short of EOF
/// — the exact next range to read (ds4's `continue_offset`). This nudges the
/// model to page forward contiguously rather than re-reading overlapping
/// ranges, keeping appended reads cache-friendly.
///
/// Output is a pure function of `(content, start, end, path)`: no timestamps,
/// ids, or mtimes, so two reads of the same file+range are byte-identical and
/// the inference server's prefix cache stays warm.
///
/// `char_budget` is the per-read ceiling (sourced from `max_tool_result_chars`
/// in production). The window is bounded to complete lines that fit within it
/// (minus a header/marker reserve) so the result stays UNDER the tool-result
/// cap — otherwise `digest_tool_result` head+tail-truncates it and inserts a
/// `"[...]"` gap, which defeats the point of a randomly-accessible file.
fn render_range(
    content: &str,
    start: usize,
    end: usize,
    path: &str,
    total: usize,
    char_budget: usize,
    sha256: &str,
) -> String {
    let lines: Vec<&str> = content.lines().collect();
    let end = end.min(total);

    // Bound the window to COMPLETE lines within the read byte budget. A file is
    // randomly accessible, so the model is best served by a contiguous, whole
    // window plus the exact next range to read — never a head+tail cut that
    // destroys the middle and forces a `recall_tool_result` round-trip.
    // Always render at least one line. The reserve covers the "# path (lines…)"
    // header, the VERBATIM wrapper, and the next-chunk marker so the total
    // stays under the cap and `digest_tool_result` never fires its `"[...]"`.
    let budget = char_budget.saturating_sub(path.len().saturating_add(320));
    let mut eff_end = start;
    let mut cost: usize = 0;
    for i in start..=end {
        // Use bytes, not Unicode scalar count: the replay guard is byte-based,
        // and a Unicode-heavy source must not cross it after this renderer has
        // declared the page safe.
        let line_cost = format!("{:>4}: ", i).len() + lines[i - 1].len() + 1;
        if i > start && cost + line_cost > budget {
            break;
        }
        cost += line_cost;
        eff_end = i;
    }
    let end = eff_end;

    let selected: Vec<String> = lines[start - 1..end]
        .iter()
        .enumerate()
        .map(|(i, line)| format!("{:>4}: {}", start + i, line))
        .collect();

    let mut out = format!(
        "# {} (lines {}-{} of {}) sha256={}\n{}",
        path,
        start,
        end,
        total,
        sha256,
        selected.join("\n")
    );
    if end < total {
        let chunk_len = end.saturating_sub(start).saturating_add(1).max(1);
        let next_end = (end + chunk_len).min(total);
        out.push_str(&format!(
            "\n[{} more lines; next: read_file lines=\"{}:{}\"]",
            total - end,
            end + 1,
            next_end
        ));
    }
    out
}

/// Expand a path: `~` → home dir, relative paths → workspace-aware.
///
/// Small/delegation models sometimes omit the full workspace prefix and
/// pass bare filenames like `MEMORY.md`. Resolve against the project first,
/// then the workspace, then the workspace memory directory for that alias.
pub(crate) fn expand_path(path: &str) -> PathBuf {
    if path.starts_with('~') {
        return crate::utils::helpers::expand_tilde(path);
    }
    let p = PathBuf::from(path);
    if p.is_absolute() {
        p
    } else {
        // Resolve relative paths against the shell working directory first
        // (matches exec/pwd behavior), then fall back to workspace for
        // memory/bootstrap convenience.
        let cwd = std::env::current_dir().ok();
        let workspace = crate::utils::helpers::get_workspace_path(None);
        resolve_relative_path(&p, cwd.as_deref(), &workspace)
    }
}

pub(crate) fn resolve_read_path(path: &str) -> PathBuf {
    let expanded = expand_path(path);
    if expanded.exists() {
        return expanded;
    }

    // SLMs sometimes hallucinate an absolute path under ~/.nanobot/workspace
    // even when CWD is a repository root. If that exact workspace-prefixed path
    // does not exist, map the relative tail onto CWD for read-only resolution.
    if expanded.is_absolute() {
        if let Ok(cwd) = std::env::current_dir() {
            let workspace = crate::utils::helpers::get_workspace_path(None);
            if let Ok(rel) = expanded.strip_prefix(&workspace) {
                let cwd_candidate = cwd.join(rel);
                if cwd_candidate.exists() {
                    return cwd_candidate;
                }
            }
        }
    }

    expanded
}

fn resolve_relative_path(relative: &Path, cwd: Option<&Path>, workspace: &Path) -> PathBuf {
    if let Some(memory_alias) = workspace_memory_alias(relative, workspace) {
        if memory_alias.exists() {
            return memory_alias;
        }
    }

    if let Some(cwd_path) = cwd {
        let cwd_resolved = cwd_path.join(relative);
        if cwd_resolved.exists() {
            return cwd_resolved;
        }
    }

    let workspace_resolved = workspace.join(relative);
    if workspace_resolved.exists() {
        return workspace_resolved;
    }

    relative.to_path_buf()
}

fn workspace_memory_alias(relative: &Path, workspace: &Path) -> Option<PathBuf> {
    if relative.components().count() == 1 && relative == Path::new("MEMORY.md") {
        Some(workspace.join("memory").join("MEMORY.md"))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use tempfile::TempDir;

    // -----------------------------------------------------------------------
    // idle write-allowlist tests
    // -----------------------------------------------------------------------

    fn idle_entries() -> Vec<String> {
        vec![
            "skills/**".to_string(),
            "MEMORY.md".to_string(),
            "/etc/nanobot/allow/**".to_string(),
        ]
    }

    #[test]
    fn idle_allowlist_subtree_exact_and_absolute() {
        let ws = Path::new("/ws");
        let e = idle_entries();
        assert!(idle_write_allowed(&e, &ws.join("skills/foo/SKILL.md"), ws));
        assert!(idle_write_allowed(&e, &ws.join("MEMORY.md"), ws));
        assert!(idle_write_allowed(&e, &Path::new("/etc/nanobot/allow/x.toml"), ws));
        // Denials: outside subtree, exact-file mismatch, foreign absolute.
        // ("skills" the directory itself DOES match "skills/**" —
        // starts_with includes the base — but a directory is not a
        // writable file target, so this is harmless by construction.)
        assert!(idle_write_allowed(&e, &ws.join("skills"), ws), "base dir matches its own /**");
        assert!(!idle_write_allowed(&e, &ws.join("MEMORY.md.bak"), ws));
        assert!(!idle_write_allowed(&e, &ws.join("workspace/other.md"), ws));
        assert!(!idle_write_allowed(&e, &Path::new("/etc/passwd"), ws));
        assert!(!idle_write_allowed(&e, &ws.join("secrets/MEMORY.md"), ws));
        assert!(!idle_write_allowed(&[], &ws.join("MEMORY.md"), ws), "empty allowlist denies");
    }

    #[tokio::test]
    async fn write_tool_enforces_idle_paths() {
        let tool = crate::agent::tools::filesystem::write::WriteFileTool::new(Some(idle_entries()));
        let ws = crate::utils::helpers::get_workspace_path(None);
        let mut params = HashMap::new();
        // Absolute path outside the allowlist (whatever the workspace is).
        let outside = ws.parent().unwrap_or(&ws).join("definitely-not-allowed.txt");
        params.insert(
            "path".to_string(),
            serde_json::json!(outside.to_string_lossy()),
        );
        params.insert("content".to_string(), serde_json::json!("x"));
        let out = tool.execute(params).await;
        assert!(
            out.starts_with("Error: idle turns may only write"),
            "denied write returned: {out}"
        );
        assert!(!outside.exists(), "denied write must not touch disk");
    }

    // -----------------------------------------------------------------------
    // expand_path tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_expand_path_absolute() {
        let result = expand_path("/tmp/test.txt");
        assert_eq!(result, PathBuf::from("/tmp/test.txt"));
    }

    #[test]
    fn test_expand_path_relative() {
        let result = expand_path("foo/bar.txt");
        assert_eq!(result, PathBuf::from("foo/bar.txt"));
    }

    #[test]
    fn test_resolve_relative_path_prefers_cwd() {
        let tmp = TempDir::new().unwrap();
        let cwd = tmp.path().join("cwd");
        let ws = tmp.path().join("workspace");
        std::fs::create_dir_all(&cwd).unwrap();
        std::fs::create_dir_all(&ws).unwrap();
        std::fs::write(cwd.join("note.txt"), "from-cwd").unwrap();
        std::fs::write(ws.join("note.txt"), "from-workspace").unwrap();

        let out = resolve_relative_path(Path::new("note.txt"), Some(&cwd), &ws);
        assert_eq!(out, cwd.join("note.txt"));
    }

    #[test]
    fn test_resolve_relative_path_falls_back_to_workspace() {
        let tmp = TempDir::new().unwrap();
        let cwd = tmp.path().join("cwd");
        let ws = tmp.path().join("workspace");
        std::fs::create_dir_all(&cwd).unwrap();
        std::fs::create_dir_all(&ws).unwrap();
        std::fs::write(ws.join("note.txt"), "from-workspace").unwrap();

        let out = resolve_relative_path(Path::new("note.txt"), Some(&cwd), &ws);
        assert_eq!(out, ws.join("note.txt"));
    }

    #[test]
    fn test_resolve_relative_path_maps_bare_memory_to_workspace_memory() {
        let tmp = TempDir::new().unwrap();
        let cwd = tmp.path().join("cwd");
        let ws = tmp.path().join("workspace");
        let memory_dir = ws.join("memory");
        std::fs::create_dir_all(&cwd).unwrap();
        std::fs::create_dir_all(&memory_dir).unwrap();
        std::fs::write(memory_dir.join("MEMORY.md"), "from-memory").unwrap();

        let out = resolve_relative_path(Path::new("MEMORY.md"), Some(&cwd), &ws);
        assert_eq!(out, memory_dir.join("MEMORY.md"));
    }

    #[test]
    fn test_resolve_relative_path_reserves_bare_memory_alias() {
        let tmp = TempDir::new().unwrap();
        let cwd = tmp.path().join("cwd");
        let ws = tmp.path().join("workspace");
        let memory_dir = ws.join("memory");
        std::fs::create_dir_all(&cwd).unwrap();
        std::fs::create_dir_all(&memory_dir).unwrap();
        std::fs::write(cwd.join("MEMORY.md"), "from-cwd").unwrap();
        std::fs::write(memory_dir.join("MEMORY.md"), "from-memory").unwrap();

        let out = resolve_relative_path(Path::new("MEMORY.md"), Some(&cwd), &ws);
        assert_eq!(out, memory_dir.join("MEMORY.md"));

        let project_out = resolve_relative_path(Path::new("./MEMORY.md"), Some(&cwd), &ws);
        assert_eq!(project_out, cwd.join("./MEMORY.md"));
    }

    #[test]
    fn test_resolve_read_path_maps_missing_workspace_prefixed_file_to_cwd() {
        let tmp = TempDir::new().unwrap();
        let cwd = tmp.path().join("repo");
        std::fs::create_dir_all(&cwd).unwrap();
        let name = "__unit_workspace_alias_resolution__.md";
        std::fs::write(cwd.join(name), "repo-arch").unwrap();

        let original_cwd = std::env::current_dir().unwrap();
        std::env::set_current_dir(&cwd).unwrap();

        let input = crate::utils::helpers::get_workspace_path(None).join(name);
        let out = resolve_read_path(input.to_str().unwrap());

        std::env::set_current_dir(original_cwd).unwrap();
        let out_canon = std::fs::canonicalize(out).unwrap();
        let expected_canon = std::fs::canonicalize(cwd.join(name)).unwrap();
        assert_eq!(out_canon, expected_canon);
    }

    #[test]
    fn test_expand_path_tilde() {
        let result = expand_path("~");
        // Should be the home directory (or "." if none).
        assert!(result.is_absolute() || result == PathBuf::from("."));
    }

    #[test]
    fn test_expand_path_tilde_subpath() {
        let result = expand_path("~/Documents/file.txt");
        // Should end with Documents/file.txt.
        assert!(result.to_string_lossy().ends_with("Documents/file.txt"));
    }

    // -----------------------------------------------------------------------
    // ReadFileTool tests
    // -----------------------------------------------------------------------

    // make_params is defined at module level (above this `mod tests`) so that
    // per-tool submodules can access it via `super::super::make_params`.
    #[tokio::test]
    async fn test_read_file_existing() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("hello.txt");
        std::fs::write(&file_path, "hello world").unwrap();

        let tool = ReadFileTool::default();
        let params = make_params(&[("path", file_path.to_str().unwrap())]);
        let result = tool.execute(params).await;
        // Bounded-default format: numbered line + header reporting the total.
        // A small file fits in one read → no continuation hint.
        assert!(result.contains("(lines 1-1 of 1)"), "{result}");
        assert!(result.contains("   1: hello world"), "{result}");
        assert!(!result.contains("more lines"), "{result}");
    }

    #[tokio::test]
    async fn test_read_file_default_is_bounded_with_paging_hint() {
        // Budget-aware: a bare read returns the largest chunk of complete lines
        // that fits the read char-budget, NOT a fixed 1000 lines. The old 1000-
        // line default overflowed the 10k-char tool-result cap and got head+tail
        // truncated — the #1 source of read_file truncation. Now the window is
        // whole, under budget, and points at the next range.
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("big.txt");
        let content = (1..=1200)
            .map(|i| format!("line {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(&file_path, &content).unwrap();

        let tool = ReadFileTool::default();
        let result = tool
            .execute(make_params(&[("path", file_path.to_str().unwrap())]))
            .await;

        assert!(result.contains("of 1200)"), "header: {}", &result[..80]);
        assert!(
            !result.contains("[truncated:"),
            "read_file must never head+tail-truncate"
        );
        assert!(
            result.contains("more lines; next: read_file lines="),
            "must tell the model the next range: {result}"
        );
        // Budget-bounded: never reaches line 1000 (each line renders ~13 chars,
        // budget allows ~500). The first line is always present.
        assert!(result.contains("   1: line 1"));
        assert!(
            !result.contains("1000: line 1000"),
            "must not dump past the char budget"
        );
        assert!(
            result.chars().count() <= 7_600,
            "output must stay under budget+slack, got {}",
            result.chars().count()
        );
    }

    #[tokio::test]
    async fn test_read_file_respects_replay_safe_cap_when_configured_higher() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("replay-boundary.txt");
        let content = (1..=2_000)
            .map(|i| format!("line {i}: {}", "x".repeat(40)))
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(&file_path, &content).unwrap();

        // The user-facing 10K result limit must not create a 8K replay-cap
        // cliff: direct source-file pagination is cheaper and clearer than
        // stashing an otherwise complete window.
        let tool = ReadFileTool::new(10_000);
        let result = tool
            .execute(make_params(&[("path", file_path.to_str().unwrap())]))
            .await;

        assert!(
            result.len() < crate::agent::context_hygiene::TOOL_RESULT_REPLAY_MAX_BYTES,
            "read_file must fit under the replay cap, got {} bytes",
            result.len()
        );
        assert!(
            result.contains("more lines; next: read_file lines="),
            "{result}"
        );
    }

    #[tokio::test]
    async fn test_read_file_replay_safe_cap_counts_utf8_bytes() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("utf8-replay-boundary.txt");
        let content = (1..=500)
            .map(|i| format!("line {i}: {}", "界".repeat(30)))
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(&file_path, &content).unwrap();

        let tool = ReadFileTool::new(10_000);
        let result = tool
            .execute(make_params(&[("path", file_path.to_str().unwrap())]))
            .await;

        assert!(
            result.len() < crate::agent::context_hygiene::TOOL_RESULT_REPLAY_MAX_BYTES,
            "read_file must use bytes for the replay ceiling, got {} bytes",
            result.len()
        );
        assert!(
            result.contains("more lines; next: read_file lines="),
            "{result}"
        );
    }

    #[tokio::test]
    async fn test_read_file_max_lines_caps_window() {
        // max_lines is an upper bound honored when it sits below the char-budget
        // line count. (It can no longer EXPAND past the budget — a multi-thousand-
        // line dump was exactly what overflowed the cap and got head+tail cut.)
        // Here max_lines=40 binds: 40 short lines fit the budget easily, so the
        // window is exactly 40.
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("window.txt");
        let content = (1..=1500)
            .map(|i| format!("line {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(&file_path, &content).unwrap();

        let tool = ReadFileTool::default();
        let mut params = make_params(&[("path", file_path.to_str().unwrap())]);
        params.insert("max_lines".to_string(), serde_json::json!(40));
        let result = tool.execute(params).await;

        assert!(result.contains("(lines 1-40 of 1500)"), "{result}");
        assert!(result.contains("  40: line 40"), "{result}");
        assert!(!result.contains(" 41: line 41"), "{result}");
        assert!(
            result.contains("more lines; next: read_file lines="),
            "must point at the next range"
        );
    }

    #[tokio::test]
    async fn test_read_file_whole_via_open_range() {
        // The model can still read the entire file when it fits the budget:
        // lines="1:" on a small file returns every line with no continuation.
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("whole.txt");
        let content = (1..=200)
            .map(|i| format!("L{i}"))
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(&file_path, &content).unwrap();

        let tool = ReadFileTool::default();
        let mut params = make_params(&[("path", file_path.to_str().unwrap())]);
        params.insert("lines".to_string(), serde_json::json!("1:"));
        let result = tool.execute(params).await;

        assert!(result.contains("(lines 1-200 of 200)"));
        assert!(result.contains(" 200: L200"));
        assert!(
            !result.contains("more lines"),
            "full read of a small file has no continuation"
        );
    }

    #[tokio::test]
    async fn test_read_file_output_is_deterministic() {
        // The prefix-cache contract: identical (path, args) must yield
        // byte-identical output every call, so the inference server reuses the
        // cached prefix instead of re-prefilling. No timestamps/ids may leak in.
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("det.txt");
        let content = (1..=800)
            .map(|i| format!("x{i}"))
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(&file_path, &content).unwrap();
        let tool = ReadFileTool::default();
        let p = file_path.to_str().unwrap();

        let bare1 = tool.execute(make_params(&[("path", p)])).await;
        let bare2 = tool.execute(make_params(&[("path", p)])).await;
        assert_eq!(
            bare1, bare2,
            "bare read must be byte-identical across calls"
        );

        let ranged = |s: &str| {
            let mut m = make_params(&[("path", p)]);
            m.insert("lines".to_string(), serde_json::json!(s));
            m
        };
        let r1 = tool.execute(ranged("100:200")).await;
        let r2 = tool.execute(ranged("100:200")).await;
        assert_eq!(r1, r2, "ranged read must be byte-identical across calls");
    }

    #[tokio::test]
    async fn test_read_file_missing() {
        let tool = ReadFileTool::default();
        let params = make_params(&[("path", "/tmp/nonexistent_nanobot_test_file_xyz.txt")]);
        let result = tool.execute(params).await;
        assert!(result.starts_with("Error: File not found"));
    }

    #[tokio::test]
    async fn test_read_file_missing_param() {
        let tool = ReadFileTool::default();
        let params = HashMap::new();
        let result = tool.execute(params).await;
        assert!(result.contains("'path' parameter is required"));
    }

    #[tokio::test]
    async fn test_read_file_not_a_file() {
        let dir = TempDir::new().unwrap();
        let tool = ReadFileTool::default();
        let params = make_params(&[("path", dir.path().to_str().unwrap())]);
        let result = tool.execute(params).await;
        assert!(result.starts_with("Error: Not a file"));
    }

    #[test]
    fn test_read_file_name() {
        let tool = ReadFileTool::default();
        assert_eq!(tool.name(), "read_file");
    }

    #[test]
    fn test_read_file_parameters_schema() {
        let tool = ReadFileTool::default();
        let params = tool.parameters();
        assert_eq!(params["type"], "object");
        assert!(params["properties"]["path"].is_object());
        assert!(params["properties"]["lines"].is_object());
    }

    #[tokio::test]
    async fn test_read_file_with_line_range() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("lines.txt");
        let content = (1..=20)
            .map(|i| format!("line {}", i))
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(&file_path, &content).unwrap();

        let tool = ReadFileTool::default();
        let mut params = make_params(&[("path", file_path.to_str().unwrap())]);
        params.insert("lines".to_string(), serde_json::json!("5:10"));
        let result = tool.execute(params).await;

        assert!(result.contains("lines 5-10 of 20"));
        assert!(result.contains("line 5"));
        assert!(result.contains("line 10"));
        assert!(!result.contains("line 4\n"));
        assert!(!result.contains("line 11"));
    }

    #[tokio::test]
    async fn test_read_file_lines_open_end() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("open.txt");
        let content = (1..=5)
            .map(|i| format!("line {}", i))
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(&file_path, &content).unwrap();

        let tool = ReadFileTool::default();
        let mut params = make_params(&[("path", file_path.to_str().unwrap())]);
        params.insert("lines".to_string(), serde_json::json!("3:"));
        let result = tool.execute(params).await;

        assert!(result.contains("lines 3-5 of 5"));
        assert!(result.contains("line 3"));
        assert!(result.contains("line 5"));
    }

    #[test]
    fn test_extract_line_range_basic() {
        let content = "alpha\nbeta\ngamma\ndelta\nepsilon";
        let hash = sha256_hex(content.as_bytes());
        let result = extract_line_range(content, "2:4", "test.txt", 7_000, &hash);
        assert!(result.contains("lines 2-4 of 5"));
        assert!(result.contains(&format!("sha256={hash}")));
        assert!(result.contains("beta"));
        assert!(result.contains("delta"));
    }

    #[test]
    fn test_extract_line_range_invalid_format() {
        let result = extract_line_range("content", "bad", "test.txt", 7_000, "hash");
        assert!(result.contains("Error"));
    }

    #[test]
    fn test_extract_line_range_out_of_bounds() {
        let result = extract_line_range("one\ntwo", "5:10", "test.txt", 7_000, "hash");
        assert!(result.contains("exceeds file length"));
    }

    #[test]
    fn render_range_bounds_to_char_budget_no_truncation() {
        // A realistic large source file: 5000 lines × ~40 chars. The old behavior
        // rendered the whole requested range and let the downstream cap head+tail-
        // truncate it. Now render_range itself bounds to complete lines within the
        // read budget, so the output is whole, contiguous, and never mid-cut.
        let content: String = (1..=5000)
            .map(|i| format!("    let x_{i} = some_call(arg_one, arg_two);"))
            .collect::<Vec<_>>()
            .join("\n");
        let hash = sha256_hex(content.as_bytes());
        let out = render_range(&content, 1, 1000, "src/big_module.rs", 5000, 7_000, &hash);

        assert!(
            !out.contains("[truncated:"),
            "read_file must never head+tail-truncate: {}",
            &out[..80.min(out.len())]
        );
        assert!(
            out.chars().count() <= 7_600,
            "output must stay under budget+slack, got {}",
            out.chars().count()
        );
        assert!(out.contains("of 5000)"), "header reports the true total");
        assert!(
            out.contains("more lines; next: read_file lines="),
            "must point at the next range"
        );
        assert!(out.contains("   1:"), "range starts at line 1");
        // Contiguous + whole: the first line is rendered complete (numbered
        // prefix + full statement), not a fragment from a head+tail cut.
        assert!(
            out.contains("let x_1 = some_call(arg_one, arg_two);"),
            "first line must be whole, not mid-cut"
        );
    }

    #[test]
    fn render_range_under_small_cap_never_emits_ellipsis_gap() {
        // Regression: a small `max_tool_result_chars` (e.g. 2500) must bound the
        // window so the result stays UNDER that cap. Previously render_range
        // hardcoded 7000, so a 2500-cap config would let read_file return ~7000
        // chars → digest_tool_result head+tail-truncated it → a "[...]" gap. With
        // the cap threaded in, the window fits and no "[...]" can appear.
        let content: String = (1..=5000)
            .map(|i| format!("    let x_{i} = some_call(arg_one, arg_two);"))
            .collect::<Vec<_>>()
            .join("\n");
        let hash = sha256_hex(content.as_bytes());
        let out = render_range(&content, 1, 1000, "src/big.rs", 5000, 2_500, &hash);

        assert!(
            !out.contains("[...]"),
            "small-cap read must not produce a '[...]' head+tail gap: {}",
            &out[..80.min(out.len())]
        );
        assert!(
            out.chars().count() <= 2_500,
            "output must stay under the 2500 cap, got {}",
            out.chars().count()
        );
        assert!(
            out.contains("more lines; next: read_file lines="),
            "must still point at the next range"
        );
        assert!(out.contains("   1:"), "range starts at line 1");
    }

    // -----------------------------------------------------------------------
    // WriteFileTool tests — see `write.rs`
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    // EditFileTool tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_edit_file_replace_string() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("edit_me.txt");
        std::fs::write(&file_path, "Hello World! This is a test.").unwrap();

        let tool = EditFileTool::default();
        let params = make_params(&[
            ("path", file_path.to_str().unwrap()),
            ("old_text", "World"),
            ("new_text", "Rust"),
        ]);
        let result = tool.execute(params).await;
        assert!(result.starts_with("Successfully edited"));

        let content = std::fs::read_to_string(&file_path).unwrap();
        assert_eq!(content, "Hello Rust! This is a test.");
    }

    #[tokio::test]
    async fn test_edit_file_rejects_identical_old_and_new_text() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("no_op.txt");
        std::fs::write(&file_path, "unchanged").unwrap();

        let result = EditFileTool::default()
            .execute(make_params(&[
                ("path", file_path.to_str().unwrap()),
                ("old_text", "unchanged"),
                ("new_text", "unchanged"),
            ]))
            .await;

        assert!(result.starts_with("Error:"), "{result}");
        assert!(result.contains("identical"), "{result}");
        assert!(result.contains("state=append"), "{result}");
        assert_eq!(std::fs::read_to_string(file_path).unwrap(), "unchanged");
    }

    #[tokio::test]
    async fn test_edit_file_applies_unified_patch() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("patch_me.txt");
        std::fs::write(&file_path, "zero\nalpha\nbeta\ngamma\n").unwrap();

        // Header points at line 1, but the hunk content has drifted to line 2.
        // The applier should relocate by context instead of failing purely on
        // line numbers.
        let patch = "\
@@ -1,2 +1,2 @@
 alpha
-beta
+BETTA";

        let tool = EditFileTool::default();
        let mut params = make_params(&[("path", file_path.to_str().unwrap())]);
        params.insert("patch".to_string(), serde_json::json!(patch));
        let result = tool.execute(params).await;
        assert!(result.starts_with("Successfully patched"), "{result}");

        let content = std::fs::read_to_string(&file_path).unwrap();
        assert_eq!(content, "zero\nalpha\nBETTA\ngamma\n");
    }

    #[tokio::test]
    async fn test_edit_file_expected_sha256_rejects_stale_edit() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("guarded.txt");
        std::fs::write(&file_path, "current\n").unwrap();

        let tool = EditFileTool::default();
        let mut params = make_params(&[
            ("path", file_path.to_str().unwrap()),
            ("old_text", "current"),
            ("new_text", "next"),
        ]);
        params.insert(
            "expected_sha256".to_string(),
            serde_json::json!("0".repeat(64)),
        );
        let result = tool.execute(params).await;
        assert!(
            result.starts_with("Error: File changed before edit"),
            "{result}"
        );

        let content = std::fs::read_to_string(&file_path).unwrap();
        assert_eq!(content, "current\n");
    }

    #[tokio::test]
    async fn test_edit_file_rejects_malformed_expected_sha256() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("guarded.txt");
        std::fs::write(&file_path, "current\n").unwrap();

        let tool = EditFileTool::default();
        let mut params = make_params(&[
            ("path", file_path.to_str().unwrap()),
            ("old_text", "current"),
            ("new_text", "next"),
        ]);
        params.insert(
            "expected_sha256".to_string(),
            serde_json::json!("not-a-real-hash"),
        );
        let result = tool.execute(params).await;
        assert!(
            result.starts_with("Error: invalid expected_sha256"),
            "{result}"
        );

        let content = std::fs::read_to_string(&file_path).unwrap();
        assert_eq!(content, "current\n");
    }

    #[tokio::test]
    async fn test_edit_file_old_text_not_found() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("edit_me.txt");
        std::fs::write(&file_path, "Hello World!").unwrap();

        let tool = EditFileTool::default();
        let params = make_params(&[
            ("path", file_path.to_str().unwrap()),
            ("old_text", "nonexistent text"),
            ("new_text", "replacement"),
        ]);
        let result = tool.execute(params).await;
        assert!(result.contains("old_text not found"));
    }

    #[tokio::test]
    async fn test_edit_file_multiple_occurrences() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("dup.txt");
        std::fs::write(&file_path, "aaa bbb aaa").unwrap();

        let tool = EditFileTool::default();
        let params = make_params(&[
            ("path", file_path.to_str().unwrap()),
            ("old_text", "aaa"),
            ("new_text", "ccc"),
        ]);
        let result = tool.execute(params).await;
        // Must be surfaced as an Error, not a Warning — small models
        // routinely misread "Warning:" as a non-fatal success.
        assert!(
            result.starts_with("Error:"),
            "multi-match should return Error, got: {result}"
        );
        assert!(result.contains("appears 2 times"));
    }

    #[tokio::test]
    async fn test_edit_file_line_ending_mismatch_gives_hint() {
        // File on disk uses CRLF; model remembered LF. Bytes differ, so
        // `contains` returns false, but a whitespace-normalized check would
        // match. The error must tell the model exactly what's wrong instead
        // of the generic "not found" dead-end.
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("crlf.txt");
        std::fs::write(&file_path, "foo\r\nbar\r\n").unwrap();

        let tool = EditFileTool::default();
        let params = make_params(&[
            ("path", file_path.to_str().unwrap()),
            ("old_text", "foo\nbar"),
            ("new_text", "baz"),
        ]);
        let result = tool.execute(params).await;
        assert!(
            result.starts_with("Error:"),
            "expected Error prefix, got: {result}"
        );
        assert!(
            result.to_lowercase().contains("line ending")
                || result.to_lowercase().contains("whitespace")
                || result.to_lowercase().contains("crlf"),
            "expected a whitespace/line-ending hint, got: {result}"
        );
    }

    #[tokio::test]
    async fn test_edit_file_trailing_whitespace_mismatch_gives_hint() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("trail.txt");
        // On-disk content has a trailing space after `hello`.
        std::fs::write(&file_path, "hello \nworld\n").unwrap();

        let tool = EditFileTool::default();
        let params = make_params(&[
            ("path", file_path.to_str().unwrap()),
            ("old_text", "hello\nworld"), // no trailing space
            ("new_text", "bye"),
        ]);
        let result = tool.execute(params).await;
        assert!(
            result.starts_with("Error:"),
            "expected Error prefix, got: {result}"
        );
        assert!(
            result.to_lowercase().contains("whitespace")
                || result.to_lowercase().contains("trailing"),
            "expected a whitespace hint, got: {result}"
        );
    }

    #[tokio::test]
    async fn test_edit_file_missing_file() {
        let tool = EditFileTool::default();
        let params = make_params(&[
            ("path", "/tmp/nonexistent_nanobot_edit_test_xyz.txt"),
            ("old_text", "a"),
            ("new_text", "b"),
        ]);
        let result = tool.execute(params).await;
        assert!(result.starts_with("Error: File not found"));
    }

    #[test]
    fn test_edit_file_name() {
        let tool = EditFileTool::default();
        assert_eq!(tool.name(), "edit_file");
    }

    // -----------------------------------------------------------------------
    // ListDirTool tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_list_dir_basic() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("file_a.txt"), "").unwrap();
        std::fs::write(dir.path().join("file_b.txt"), "").unwrap();
        std::fs::create_dir(dir.path().join("subdir")).unwrap();

        let tool = ListDirTool;
        let params = make_params(&[("path", dir.path().to_str().unwrap())]);
        let result = tool.execute(params).await;

        assert!(result.contains("[file] file_a.txt"));
        assert!(result.contains("[file] file_b.txt"));
        assert!(result.contains("[dir]  subdir"));
    }

    #[tokio::test]
    async fn test_list_dir_empty() {
        let dir = TempDir::new().unwrap();

        let tool = ListDirTool;
        let params = make_params(&[("path", dir.path().to_str().unwrap())]);
        let result = tool.execute(params).await;
        assert!(result.contains("is empty"));
    }

    #[tokio::test]
    async fn test_list_dir_not_found() {
        let tool = ListDirTool;
        let params = make_params(&[("path", "/tmp/nonexistent_nanobot_dir_xyz")]);
        let result = tool.execute(params).await;
        assert!(result.starts_with("Error: Directory not found"));
    }

    #[tokio::test]
    async fn test_list_dir_not_a_directory() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("file.txt");
        std::fs::write(&file_path, "content").unwrap();

        let tool = ListDirTool;
        let params = make_params(&[("path", file_path.to_str().unwrap())]);
        let result = tool.execute(params).await;
        assert!(result.starts_with("Error: Not a directory"));
    }

    #[test]
    fn test_list_dir_name() {
        let tool = ListDirTool;
        assert_eq!(tool.name(), "list_dir");
    }

    #[tokio::test]
    async fn test_list_dir_missing_param() {
        let tool = ListDirTool;
        let params = HashMap::new();
        let result = tool.execute(params).await;
        assert!(result.contains("'path' parameter is required"));
    }

    // -----------------------------------------------------------------------
    // FindFilesTool tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_find_files_recursive_pattern() {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir_all(dir.path().join("src").join("nested")).unwrap();
        std::fs::write(dir.path().join("src").join("main.rs"), "fn main() {}").unwrap();
        std::fs::write(dir.path().join("src").join("nested").join("lib.rs"), "").unwrap();
        std::fs::write(dir.path().join("README.md"), "").unwrap();

        let tool = FindFilesTool;
        let mut params =
            make_params(&[("path", dir.path().to_str().unwrap()), ("pattern", "*.rs")]);
        params.insert("max_depth".to_string(), serde_json::json!(3));
        let result = tool.execute(params).await;

        assert!(result.contains("src/main.rs"), "{result}");
        assert!(result.contains("src/nested/lib.rs"), "{result}");
        assert!(!result.contains("README.md"), "{result}");
    }

    #[tokio::test]
    async fn test_find_files_skips_hidden_by_default() {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir_all(dir.path().join(".git")).unwrap();
        std::fs::write(dir.path().join(".git").join("config"), "").unwrap();

        let tool = FindFilesTool;
        let result = tool
            .execute(make_params(&[
                ("path", dir.path().to_str().unwrap()),
                ("pattern", "config"),
            ]))
            .await;
        assert!(result.starts_with("No matches"), "{result}");
    }

    // -----------------------------------------------------------------------
    // SearchFilesTool tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_search_files_finds_plain_text_with_context() {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir_all(dir.path().join("src")).unwrap();
        std::fs::write(
            dir.path().join("src").join("lib.rs"),
            "alpha\nNeedle here\nomega\n",
        )
        .unwrap();
        std::fs::write(dir.path().join("README.md"), "no match\n").unwrap();

        let tool = SearchFilesTool;
        let mut params = make_params(&[
            ("path", dir.path().to_str().unwrap()),
            ("query", "needle"),
            ("pattern", "*.rs"),
        ]);
        params.insert("context".to_string(), serde_json::json!(1));
        let result = tool.execute(params).await;

        assert!(result.contains("Found 1 matching line"), "{result}");
        assert!(result.contains("src/lib.rs:1- alpha"), "{result}");
        assert!(result.contains("src/lib.rs:2: Needle here"), "{result}");
        assert!(result.contains("src/lib.rs:3- omega"), "{result}");
        assert!(!result.contains("README.md"), "{result}");
    }

    #[tokio::test]
    async fn test_search_files_regex_and_limit() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("a.txt"), "item 1\nitem 2\nitem 3\n").unwrap();

        let tool = SearchFilesTool;
        let mut params = make_params(&[
            ("path", dir.path().to_str().unwrap()),
            ("query", r"item \d"),
        ]);
        params.insert("regex".to_string(), serde_json::json!(true));
        params.insert("limit".to_string(), serde_json::json!(2));
        let result = tool.execute(params).await;

        assert!(result.contains("a.txt:1: item 1"), "{result}");
        assert!(result.contains("a.txt:2: item 2"), "{result}");
        assert!(!result.contains("a.txt:3: item 3"), "{result}");
        assert!(result.contains("limit reached"), "{result}");
    }

    // -----------------------------------------------------------------------
    // FileInfoTool tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_file_info_includes_sha256() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("hash.txt");
        std::fs::write(&file_path, "hash me").unwrap();

        let tool = FileInfoTool;
        let result = tool
            .execute(make_params(&[("path", file_path.to_str().unwrap())]))
            .await;

        assert!(result.contains("Type: file"), "{result}");
        assert!(result.contains("Size: 7 bytes"), "{result}");
        assert!(
            result.contains(&format!("SHA-256: {}", sha256_hex(b"hash me"))),
            "{result}"
        );
    }

    #[test]
    fn test_apply_unified_patch_adds_line_to_empty_file() {
        let patch = "\
@@ -1,0 +1,2 @@
+alpha
+beta";
        let (updated, hunks) =
            crate::agent::tools::apply_patch::apply_unified_patch_to_content("", patch).unwrap();
        assert_eq!(hunks, 1);
        assert_eq!(updated, "alpha\nbeta");
    }

    #[tokio::test]
    async fn test_read_binary_file() {
        let dir = tempfile::tempdir().unwrap();
        let bin_path = dir.path().join("test.bin");
        // Write binary content with null bytes.
        std::fs::write(&bin_path, b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR").unwrap();

        let tool = ReadFileTool::default();
        let mut params = HashMap::new();
        params.insert(
            "path".to_string(),
            serde_json::Value::String(bin_path.to_string_lossy().to_string()),
        );
        let result = tool.execute(params).await;
        assert!(
            result.starts_with("[Binary file:"),
            "Expected binary detection, got: {}",
            result
        );
        assert!(result.contains("bytes]"));
    }

    // -----------------------------------------------------------------------
    // Recovery hint tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_read_file_not_found_has_hint() {
        let tool = ReadFileTool::default();
        let params = make_params(&[("path", "/tmp/nanobot_hint_test_nonexistent_xyz.txt")]);
        let result = tool.execute(params).await;
        assert!(
            result.contains("Hint:"),
            "Expected hint in error: {}",
            result
        );
        assert!(
            result.contains("list_dir"),
            "Expected list_dir hint: {}",
            result
        );
    }

    #[tokio::test]
    async fn test_read_file_not_a_file_has_hint() {
        let dir = TempDir::new().unwrap();
        let tool = ReadFileTool::default();
        let params = make_params(&[("path", dir.path().to_str().unwrap())]);
        let result = tool.execute(params).await;
        assert!(
            result.contains("Hint:"),
            "Expected hint in error: {}",
            result
        );
        assert!(
            result.contains("list_dir"),
            "Expected list_dir hint: {}",
            result
        );
    }

    #[tokio::test]
    async fn test_edit_file_not_found_has_hint() {
        let tool = EditFileTool::default();
        let params = make_params(&[
            ("path", "/tmp/nanobot_hint_test_nonexistent_edit_xyz.txt"),
            ("old_text", "a"),
            ("new_text", "b"),
        ]);
        let result = tool.execute(params).await;
        assert!(
            result.contains("Hint:"),
            "Expected hint in error: {}",
            result
        );
        assert!(
            result.contains("list_dir"),
            "Expected list_dir hint: {}",
            result
        );
    }

    #[tokio::test]
    async fn test_edit_file_old_text_not_found_has_hint() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("edit_hint.txt");
        std::fs::write(&file_path, "some existing content").unwrap();

        let tool = EditFileTool::default();
        let params = make_params(&[
            ("path", file_path.to_str().unwrap()),
            ("old_text", "text that does not exist in file"),
            ("new_text", "replacement"),
        ]);
        let result = tool.execute(params).await;
        assert!(
            result.contains("Hint:"),
            "Expected hint in error: {}",
            result
        );
        assert!(
            result.contains("read_file"),
            "Expected read_file hint: {}",
            result
        );
    }

    #[tokio::test]
    async fn test_list_dir_not_found_has_hint() {
        let tool = ListDirTool;
        let params = make_params(&[("path", "/tmp/nanobot_hint_test_nonexistent_dir_xyz")]);
        let result = tool.execute(params).await;
        assert!(
            result.contains("Hint:"),
            "Expected hint in error: {}",
            result
        );
        assert!(
            result.contains("list_dir"),
            "Expected list_dir hint: {}",
            result
        );
    }

    #[tokio::test]
    async fn test_list_dir_not_a_directory_has_hint() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("not_a_dir.txt");
        std::fs::write(&file_path, "content").unwrap();

        let tool = ListDirTool;
        let params = make_params(&[("path", file_path.to_str().unwrap())]);
        let result = tool.execute(params).await;
        assert!(
            result.contains("Hint:"),
            "Expected hint in error: {}",
            result
        );
        assert!(
            result.contains("read_file"),
            "Expected read_file hint: {}",
            result
        );
    }
}
