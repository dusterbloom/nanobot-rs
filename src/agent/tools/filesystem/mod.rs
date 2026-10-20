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
use std::path::{Component, Path, PathBuf};

use async_trait::async_trait;
use regex::RegexBuilder;
use sha2::{Digest, Sha256};

use super::base::{PermissionLevel, Tool, ToolConcurrency, ToolContext, ToolResult};
use crate::agent::context_hygiene::INLINE_TOOL_RESULT_MAX_BYTES;
use crate::errors::ToolError;

/// Extract a required string parameter, returning an error string on missing.
/// Typed param extractor (error protocol Phase 2): the message preserves
/// the legacy bytes so the registry's "is required" worked-example arm and
/// exact-substring tests are unchanged.
fn require_param<'a>(
    params: &'a HashMap<String, serde_json::Value>,
    key: &str,
) -> Result<&'a str, crate::errors::ToolError> {
    params
        .get(key)
        .and_then(|v| v.as_str())
        .ok_or_else(|| crate::errors::ToolError::InvalidArgs {
            message: format!("'{}' parameter is required", key),
        })
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
/// `char_budget` is measured in bytes and bounds the complete rendered page to
/// the shared inline result ceiling. The renderer reserves wrapper space and
/// emits whole contiguous source lines with an exact next-range pointer. A
/// single line larger than that budget remains intact and can flow through the
/// existing lossless result-handle/inspection path.
pub struct ReadFileTool {
    char_budget: usize,
}

impl ReadFileTool {
    pub fn new(char_budget: usize) -> Self {
        // Floor so an absurdly small cap still renders at least a few lines.
        Self {
            char_budget: char_budget.clamp(512, INLINE_TOOL_RESULT_MAX_BYTES),
        }
    }
}

impl Default for ReadFileTool {
    fn default() -> Self {
        Self::new(INLINE_TOOL_RESULT_MAX_BYTES)
    }
}

#[async_trait]
impl Tool for ReadFileTool {
    fn name(&self) -> &str {
        "read_file"
    }

    fn description(&self) -> &str {
        "Read a file as a page of numbered, complete contiguous lines within a 4,096-byte inline budget, with the total line count, SHA-256, and exact next range. Page forward using the suggested range or lines=\"START:END\" (1-indexed, inclusive). An exceptionally oversized line stays whole; if it pushes the result over the inline limit, the result is stored as a lossless handle that can be read with inspect_tool_result. Use lines=\"1:\" only when the file fits inline."
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
                    "description": "Line range, e.g. \"10:50\" (1-indexed, inclusive). Omit for the first page; use \"1:\" only when the file fits inline. The output header reports the total line count and the next range to read."
                },
                "max_lines": {
                    "type": "integer",
                    "description": "Number of lines for a bare read when lines is omitted. Default: 1000, max: 5000"
                }
            },
            "required": ["path"]
        })
    }

    async fn execute(
        &self,
        params: HashMap<String, serde_json::Value>,
        _ctx: &ToolContext,
    ) -> ToolResult {
        let path = require_param(&params, "path")?;

        let file_path = resolve_read_path(path);

        if !file_path.exists() {
            return Err(ToolError::NotFound(format!(
                "File not found: {}. Hint: verify the path exists. Use list_dir to browse the directory.",
                path
            )));
        }
        if !file_path.is_file() {
            return Err(ToolError::InvalidArgs {
                message: format!(
                    "Not a file: {}. Hint: this path is a directory, not a file. Use list_dir to see its contents.",
                    path
                ),
            });
        }

        // Read raw bytes first for binary detection.
        let bytes = match tokio::fs::read(&file_path).await {
            Ok(b) => b,
            Err(e) => {
                return if e.kind() == std::io::ErrorKind::PermissionDenied {
                    Err(ToolError::PermissionDenied(format!(
                        "Permission denied: {}. Hint: check file permissions or try a different path.",
                        path
                    )))
                } else {
                    // Legacy quirk preserved: no "Error:" prefix → success channel.
                    Ok(format!("Error reading file: {}", e).into())
                };
            }
        };

        // Binary detection: null bytes in first 512 bytes.
        let display_path = file_path.to_string_lossy().to_string();
        if crate::utils::helpers::is_binary(&bytes) {
            return Ok(format!("[Binary file: {}, {} bytes]", display_path, bytes.len()).into());
        }

        let content = String::from_utf8_lossy(&bytes).to_string();
        let content_sha256 = sha256_hex(&bytes);
        let total = content.lines().count();
        if total == 0 {
            return Ok(format!("# {} (0 lines) sha256={}\n", display_path, content_sha256).into());
        }

        // Explicit range → render it. Bare read → first DEFAULT_READ_LINES,
        // ds4-style, so the model never dumps a whole file unless it asks
        // (lines="1:"). Both paths share the deterministic renderer below.
        if let Some(lines_param) = params.get("lines").and_then(|v| v.as_str()) {
            return Ok(extract_line_range(
                &content,
                lines_param,
                &display_path,
                self.char_budget,
                &content_sha256,
            )
            .into());
        }
        let max_lines =
            bounded_usize_param(&params, "max_lines", DEFAULT_READ_LINES, MAX_READ_LINES);
        Ok(render_range(
            &content,
            1,
            max_lines.min(total),
            &display_path,
            total,
            self.char_budget,
            &content_sha256,
        )
        .into())
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
    let target = normalize_lexical(target);
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
            let base = normalize_lexical(base);
            subtree && target.starts_with(&base) || (!subtree && target == base)
        } else {
            let ws = normalize_lexical(&workspace.join(base));
            subtree && target.starts_with(&ws) || (!subtree && target == ws)
        };
        if matched {
            allowed = true;
            break;
        }
    }
    allowed
}

/// Normalize `.` and `..` without touching the filesystem. Relative paths
/// retain parents that cannot be cancelled; absolute paths stop at root.
pub(crate) fn normalize_lexical(path: &Path) -> PathBuf {
    let absolute = path.is_absolute();
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Prefix(prefix) => normalized.push(prefix.as_os_str()),
            Component::RootDir => normalized.push(component.as_os_str()),
            Component::CurDir => {}
            Component::ParentDir => match normalized.components().next_back() {
                Some(Component::Normal(_)) => {
                    normalized.pop();
                }
                _ if !absolute => normalized.push(component.as_os_str()),
                _ => {}
            },
            Component::Normal(part) => normalized.push(part),
        }
    }
    normalized
}

/// Consistent denial message for idle-turn writes outside the allowlist.
pub(crate) fn idle_write_denied(target: &Path) -> String {
    format!(
        "Error: idle turns may only write to configured paths (idle.writePaths); denied: {}. Use the message tool to ask the user for wider access.",
        target.display()
    )
}

/// Typed form of [`idle_write_denied`] for tools returning [`ToolResult`].
pub(crate) fn idle_write_denied_err(target: &Path) -> crate::errors::ToolError {
    crate::errors::ToolError::PermissionDenied(
        idle_write_denied(target)
            .trim_start_matches("Error: ")
            .to_string(),
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
        "Edit a file by replacing old_text with new_text, or apply a unified diff in patch. Use patch for multi-line edits when exact old_text matching would be brittle. Optional expected_sha256 prevents overwriting a file that changed since it was last read."
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

    async fn execute(
        &self,
        params: HashMap<String, serde_json::Value>,
        _ctx: &ToolContext,
    ) -> ToolResult {
        let path = require_param(&params, "path")?;

        let mut file_path = expand_path(path);
        if let Some(paths) = &self.idle_paths {
            file_path = normalize_lexical(&file_path);
            let workspace = crate::utils::helpers::get_workspace_path(None);
            if !idle_write_allowed(paths, &file_path, &workspace) {
                return Err(idle_write_denied_err(&file_path));
            }
        }

        if !file_path.exists() {
            return Err(ToolError::NotFound(format!(
                "File not found: {}. Hint: verify the path exists. Use list_dir to browse the directory.",
                path
            )));
        }

        // Legacy quirk preserved: no "Error:" prefix → success channel.
        let content = match tokio::fs::read_to_string(&file_path).await {
            Ok(c) => c,
            Err(e) => return Ok(format!("Error reading file: {}", e).into()),
        };

        if let Some(expected) = params.get("expected_sha256").and_then(|v| v.as_str()) {
            if !is_sha256_hex(expected) {
                return Err(ToolError::InvalidArgs {
                    message: format!(
                        "invalid expected_sha256 '{}'. Omit expected_sha256 unless you copied the 64-character sha256 value from read_file.",
                        expected.trim()
                    ),
                });
            }
            let actual = sha256_hex(content.as_bytes());
            if !expected.trim().eq_ignore_ascii_case(&actual) {
                return Err(ToolError::InvalidArgs {
                    message: format!(
                        "File changed before edit. expected_sha256={}, actual_sha256={}. Re-read the file before retrying; omit expected_sha256 if you did not copy it from read_file.",
                        expected.trim(),
                        actual
                    ),
                });
            }
        }

        if let Some(patch) = params.get("patch").and_then(|v| v.as_str()) {
            if patch.trim().is_empty() {
                return Err(ToolError::InvalidArgs {
                    message: "'patch' parameter cannot be empty".to_string(),
                });
            }
            let (new_content, hunks) =
                match super::apply_patch::apply_unified_patch_to_content(&content, patch) {
                    Ok(result) => result,
                    // Parser errors carry the legacy prefix; strip so render
                    // re-prefixes exactly once.
                    Err(e) => {
                        return Err(ToolError::Execution {
                            message: e.trim_start_matches("Error: ").to_string(),
                        })
                    }
                };
            return match tokio::fs::write(&file_path, new_content).await {
                Ok(()) => Ok(format!("Successfully patched {} ({} hunk(s))", path, hunks).into()),
                Err(e) => {
                    if e.kind() == std::io::ErrorKind::PermissionDenied {
                        Err(ToolError::PermissionDenied(format!(
                            "Permission denied: {}. Hint: check file permissions or try a different path.",
                            path
                        )))
                    } else {
                        // Legacy quirk preserved: success channel.
                        Ok(format!("Error writing file: {}", e).into())
                    }
                }
            };
        }

        let old_text = match params.get("old_text").and_then(|v| v.as_str()) {
            Some(t) => t,
            None => {
                return Err(ToolError::InvalidArgs {
                    message: "either 'patch' or both 'old_text' and 'new_text' are required"
                        .to_string(),
                })
            }
        };
        let new_text = match params.get("new_text").and_then(|v| v.as_str()) {
            Some(t) => t,
            None => {
                return Err(ToolError::InvalidArgs {
                    message: "either 'patch' or both 'old_text' and 'new_text' are required"
                        .to_string(),
                })
            }
        };
        if old_text == new_text {
            return Err(ToolError::InvalidArgs {
                message: "old_text and new_text are identical; no change was made. Provide different replacement text, or call write_file with state=append to add a suffix.".to_string(),
            });
        }

        if !content.contains(old_text) {
            return Err(ToolError::NotFound(
                diagnose_missing_old_text(&content, old_text)
                    .trim_start_matches("Error: ")
                    .to_string(),
            ));
        }

        // Count occurrences.
        let count = content.matches(old_text).count();
        if count > 1 {
            return Err(ToolError::InvalidArgs {
                message: format!(
                    "old_text appears {} times. Please provide more context to make it unique.",
                    count
                ),
            });
        }

        let new_content = content.replacen(old_text, new_text, 1);

        match tokio::fs::write(&file_path, new_content).await {
            Ok(()) => Ok(format!("Successfully edited {}", path).into()),
            Err(e) => {
                if e.kind() == std::io::ErrorKind::PermissionDenied {
                    Err(ToolError::PermissionDenied(format!(
                        "Permission denied: {}. Hint: check file permissions or try a different path.",
                        path
                    )))
                } else {
                    // Legacy quirk preserved: success channel.
                    Ok(format!("Error writing file: {}", e).into())
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

    async fn execute(
        &self,
        params: HashMap<String, serde_json::Value>,
        _ctx: &ToolContext,
    ) -> ToolResult {
        let path = require_param(&params, "path")?;

        let dir_path = expand_path(path);

        if !dir_path.exists() {
            return Err(ToolError::NotFound(format!(
                "Directory not found: {}. Hint: parent directory does not exist. Use list_dir to find the correct path.",
                path
            )));
        }
        if !dir_path.is_dir() {
            return Err(ToolError::InvalidArgs {
                message: format!(
                    "Not a directory: {}. Hint: this path is a file, not a directory. Use read_file instead.",
                    path
                ),
            });
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
                        // Legacy quirk preserved: no "Error:" prefix → success channel.
                        Err(e) => return Ok(format!("Error reading directory: {}", e).into()),
                    }
                }

                if items.is_empty() {
                    return Ok(format!("Directory {} is empty", path).into());
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

                Ok(output.into())
            }
            Err(e) => {
                if e.kind() == std::io::ErrorKind::PermissionDenied {
                    Err(ToolError::PermissionDenied(format!(
                        "Permission denied: {}. Hint: check file permissions or try a different path.",
                        path
                    )))
                } else {
                    // Legacy quirk preserved: no "Error:" prefix → success channel.
                    Ok(format!("Error listing directory: {}", e).into())
                }
            }
        }
    }
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

    async fn execute(
        &self,
        params: HashMap<String, serde_json::Value>,
        _ctx: &ToolContext,
    ) -> ToolResult {
        let query = match require_param(&params, "query") {
            Ok(q) if !q.trim().is_empty() => q.trim(),
            Ok(_) => {
                return Err(ToolError::InvalidArgs {
                    message: "'query' parameter cannot be empty".to_string(),
                })
            }
            Err(e) => return Err(e),
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
            Err(e) => {
                return Err(ToolError::InvalidArgs {
                    message: e.trim_start_matches("Error: ").to_string(),
                })
            }
        };

        let root = expand_path(path);
        if !root.exists() {
            return Err(ToolError::NotFound(format!(
                "Directory not found: {}. Hint: verify the path or use list_dir on its parent.",
                path
            )));
        }
        if !root.is_dir() {
            return Err(ToolError::InvalidArgs {
                message: format!(
                    "Not a directory: {}. Hint: use read_file for a single file.",
                    path
                ),
            });
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
                // Legacy quirk preserved: no "Error:" prefix → success channel.
                Err(e) => {
                    return Ok(format!("Error reading directory {}: {}", dir.display(), e).into())
                }
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
        Ok(out.into())
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
/// `char_budget` is measured in bytes and capped at the shared inline limit.
/// The window includes only complete lines that fit after reserving space for
/// the path, header, wrapper, and continuation marker.
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

    // Bound the window to COMPLETE lines within the inline byte budget. A file is
    // randomly accessible, so the model is best served by a contiguous, whole
    // window plus the exact next range to read — never a head+tail cut that
    // destroys the middle and forces an `inspect_tool_result` round-trip.
    // Always render at least one full line. The reserve covers the header,
    // result wrapper, and next-chunk marker so ordinary pages fit inline. An
    // exceptional oversized line remains lossless and uses the handle path.
    let budget = char_budget.saturating_sub(path.len().saturating_add(320));
    let mut eff_end = start;
    let mut cost: usize = 0;
    for i in start..=end {
        // Use bytes, not Unicode scalar count: the inline guard is byte-based,
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
        assert!(idle_write_allowed(
            &e,
            &Path::new("/etc/nanobot/allow/x.toml"),
            ws
        ));
        // Denials: outside subtree, exact-file mismatch, foreign absolute.
        // ("skills" the directory itself DOES match "skills/**" —
        // starts_with includes the base — but a directory is not a
        // writable file target, so this is harmless by construction.)
        assert!(
            idle_write_allowed(&e, &ws.join("skills"), ws),
            "base dir matches its own /**"
        );
        assert!(!idle_write_allowed(&e, &ws.join("MEMORY.md.bak"), ws));
        assert!(!idle_write_allowed(&e, &ws.join("workspace/other.md"), ws));
        assert!(!idle_write_allowed(&e, &Path::new("/etc/passwd"), ws));
        assert!(!idle_write_allowed(&e, &ws.join("secrets/MEMORY.md"), ws));
        assert!(
            !idle_write_allowed(&[], &ws.join("MEMORY.md"), ws),
            "empty allowlist denies"
        );
    }

    #[test]
    fn idle_allowlist_rejects_parent_traversal() {
        let workspace = Path::new("/workspace");
        let target = workspace.join("skills/../../outside.txt");
        assert!(!idle_write_allowed(
            &["skills/**".into()],
            &target,
            workspace
        ));
    }

    #[tokio::test]
    async fn write_tool_enforces_idle_paths() {
        let tool = crate::agent::tools::filesystem::write::WriteFileTool::new(Some(idle_entries()));
        let ws = crate::utils::helpers::get_workspace_path(None);
        let mut params = HashMap::new();
        // Absolute path outside the allowlist (whatever the workspace is).
        let outside = ws
            .parent()
            .unwrap_or(&ws)
            .join("definitely-not-allowed.txt");
        params.insert(
            "path".to_string(),
            serde_json::json!(outside.to_string_lossy()),
        );
        params.insert("content".to_string(), serde_json::json!("x"));
        let out = crate::agent::tools::base::render_result(
            tool.execute(params, &crate::agent::tools::base::ToolContext::sandbox())
                .await,
        );
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
        let result = crate::agent::tools::base::render_result(
            tool.execute(params, &crate::agent::tools::base::ToolContext::sandbox())
                .await,
        );
        // Bounded-default format: numbered line + header reporting the total.
        // A small file fits in one read → no continuation hint.
        assert!(result.contains("(lines 1-1 of 1)"), "{result}");
        assert!(result.contains("   1: hello world"), "{result}");
        assert!(!result.contains("more lines"), "{result}");
    }

    #[tokio::test]
    async fn test_read_file_default_is_bounded_with_paging_hint() {
        // A bare read returns the largest chunk of complete lines that fits
        // the inline byte budget, NOT a fixed 1000 lines. The window stays
        // whole and points at the exact next range.
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("big.txt");
        let content = (1..=1200)
            .map(|i| format!("line {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(&file_path, &content).unwrap();

        let tool = ReadFileTool::default();
        let result = crate::agent::tools::base::render_result(
            tool.execute(
                make_params(&[("path", file_path.to_str().unwrap())]),
                &crate::agent::tools::base::ToolContext::sandbox(),
            )
            .await,
        );

        assert!(result.contains("of 1200)"), "header: {}", &result[..80]);
        assert!(
            !result.contains("[truncated:"),
            "read_file must never head+tail-truncate"
        );
        assert!(
            result.contains("more lines; next: read_file lines="),
            "must tell the model the next range: {result}"
        );
        // Budget-bounded: never reaches line 1000 (the inline page holds only
        // a few hundred short lines). The first line is always present.
        assert!(result.contains("   1: line 1"));
        assert!(
            !result.contains("1000: line 1000"),
            "must not dump past the byte budget"
        );
        assert!(
            result.len() <= crate::agent::context_hygiene::INLINE_TOOL_RESULT_MAX_BYTES,
            "ordinary output must stay within the inline byte limit, got {}",
            result.len()
        );
    }

    #[tokio::test]
    async fn test_read_file_configured_budget_stays_within_inline_bytes() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("large-ascii.txt");
        let content = (0..400)
            .map(|i| format!("item {i}: useful evidence"))
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(&file_path, &content).unwrap();

        // A large configured result cap must not let an ordinary first page
        // become a handle at the shared inline ingestion boundary.
        let tool = ReadFileTool::new(20_000);
        let result = crate::agent::tools::base::render_result(
            tool.execute(
                make_params(&[("path", file_path.to_str().unwrap())]),
                &crate::agent::tools::base::ToolContext::sandbox(),
            )
            .await,
        );

        assert!(
            result.len() <= crate::agent::context_hygiene::INLINE_TOOL_RESULT_MAX_BYTES,
            "ordinary source page must fit the inline byte limit, got {} bytes",
            result.len()
        );
        assert!(result.contains("item 0: useful evidence"), "{result}");
        assert!(
            result.contains("sha256="),
            "page must retain its source digest"
        );
        assert!(
            result.contains("more lines; next: read_file lines=\""),
            "ordinary source page must provide a line continuation: {result}"
        );
        assert!(!result.contains("[truncated:"), "{result}");

        let displayed_end = result
            .split_once("(lines 1-")
            .and_then(|(_, rest)| rest.split_once(" of "))
            .and_then(|(end, _)| end.parse::<usize>().ok())
            .expect("page header must report the last displayed line");
        let next_end = (displayed_end * 2).min(400);
        assert!(
            result.contains(&format!(
                "read_file lines=\"{}:{}\"",
                displayed_end + 1,
                next_end
            )),
            "continuation must start at the exact next line: {result}"
        );
    }

    #[tokio::test]
    async fn test_read_file_long_path_stays_within_inline_bytes() {
        let temp = TempDir::new().unwrap();
        let mut directory = temp.path().to_path_buf();
        for _ in 0..10 {
            directory.push(format!("segment_{}", "x".repeat(70)));
            std::fs::create_dir_all(&directory).unwrap();
        }
        let file_path = directory.join("long-path.txt");
        let content = (0..300)
            .map(|i| format!("item {i}: long-path evidence"))
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(&file_path, &content).unwrap();

        let tool = ReadFileTool::new(20_000);
        let result = crate::agent::tools::base::render_result(
            tool.execute(
                make_params(&[("path", file_path.to_str().unwrap())]),
                &crate::agent::tools::base::ToolContext::sandbox(),
            )
            .await,
        );

        assert!(
            result.len() <= crate::agent::context_hygiene::INLINE_TOOL_RESULT_MAX_BYTES,
            "long source path plus page wrapper must fit the inline byte limit, got {} bytes",
            result.len()
        );
        assert!(result.contains("item 0: long-path evidence"), "{result}");
        assert!(
            result.contains("more lines; next: read_file lines="),
            "{result}"
        );
    }

    #[tokio::test]
    async fn test_read_file_unicode_pages_are_bounded_and_reconstruct_exactly() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("unicode.txt");
        let content = (1..=220)
            .map(|i| format!("line {i}: {}", "界🦀é".repeat(20)))
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(&file_path, &content).unwrap();

        let tool = ReadFileTool::new(20_000);
        let mut start = 1;
        let mut reconstructed = Vec::new();
        loop {
            let result = crate::agent::tools::base::render_result(
                tool.execute(
                    make_params(&[
                        ("path", file_path.to_str().unwrap()),
                        ("lines", &format!("{start}:220")),
                    ]),
                    &crate::agent::tools::base::ToolContext::sandbox(),
                )
                .await,
            );

            assert!(
                result.len() <= crate::agent::context_hygiene::INLINE_TOOL_RESULT_MAX_BYTES,
                "Unicode source page must fit the inline byte limit, got {} bytes",
                result.len()
            );
            let header_range = result
                .lines()
                .next()
                .and_then(|header| header.split_once("(lines "))
                .and_then(|(_, rest)| rest.split_once(" of "))
                .map(|(range, _)| range)
                .expect("page header must report its line range");
            let (page_start, page_end) = header_range
                .split_once('-')
                .and_then(|(first, last)| {
                    Some((first.parse::<usize>().ok()?, last.parse::<usize>().ok()?))
                })
                .expect("page header range must parse");
            assert_eq!(page_start, start, "pages must be contiguous: {result}");

            for rendered_line in result.lines().skip(1) {
                if rendered_line.starts_with('[') {
                    break;
                }
                let (_, source_line) = rendered_line
                    .split_once(": ")
                    .expect("each source line must retain its numbered prefix");
                reconstructed.push(source_line.to_string());
            }

            if page_end == 220 {
                break;
            }
            let next_end = (page_end + (page_end - page_start + 1)).min(220);
            assert!(
                result.contains(&format!("read_file lines=\"{}:{next_end}\"", page_end + 1)),
                "page continuation must name the exact next range: {result}"
            );
            start = page_end + 1;
        }

        assert_eq!(reconstructed.join("\n"), content);
    }

    #[test]
    fn test_read_file_oversized_line_stays_whole_for_handle_inspection() {
        let long_line = "é🦀".repeat(2_500);
        let content = format!("{long_line}\nfollowing evidence");
        let hash = sha256_hex(content.as_bytes());
        let page = render_range(
            &content,
            1,
            2,
            "source.txt",
            2,
            crate::agent::context_hygiene::INLINE_TOOL_RESULT_MAX_BYTES,
            &hash,
        );

        assert!(
            page.len() > crate::agent::context_hygiene::INLINE_TOOL_RESULT_MAX_BYTES,
            "an indivisible oversized line must flow to the existing handle path"
        );
        assert!(
            page.contains(&long_line),
            "the renderer must keep the full source line"
        );
        assert!(page.contains("lines 1-1 of 2"), "{page}");
        assert!(
            page.contains("next: read_file lines=\"2:2\""),
            "continuation must advance by a whole line without claiming skipped bytes: {page}"
        );
        let tool = ReadFileTool::default();
        let description = tool.description();
        assert!(description.contains("oversized line"), "{description}");
        assert!(description.contains("inspect_tool_result"), "{description}");
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

        // A user-facing result limit above the inline threshold must still
        // produce an ordinary source page that can be read immediately.
        let tool = ReadFileTool::new(10_000);
        let result = crate::agent::tools::base::render_result(
            tool.execute(
                make_params(&[("path", file_path.to_str().unwrap())]),
                &crate::agent::tools::base::ToolContext::sandbox(),
            )
            .await,
        );

        assert!(
            result.len() <= crate::agent::context_hygiene::INLINE_TOOL_RESULT_MAX_BYTES,
            "read_file must fit under the inline cap, got {} bytes",
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
        let file_path = dir.path().join("utf8-inline-boundary.txt");
        let content = (1..=500)
            .map(|i| format!("line {i}: {}", "界".repeat(30)))
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(&file_path, &content).unwrap();

        let tool = ReadFileTool::new(10_000);
        let result = crate::agent::tools::base::render_result(
            tool.execute(
                make_params(&[("path", file_path.to_str().unwrap())]),
                &crate::agent::tools::base::ToolContext::sandbox(),
            )
            .await,
        );

        assert!(
            result.len() <= crate::agent::context_hygiene::INLINE_TOOL_RESULT_MAX_BYTES,
            "read_file must use bytes for the inline ceiling, got {} bytes",
            result.len()
        );
        assert!(
            result.contains("more lines; next: read_file lines="),
            "{result}"
        );
    }

    #[tokio::test]
    async fn test_read_file_max_lines_caps_window() {
        // max_lines is an upper bound honored when it sits below the byte-budget
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
        let result = crate::agent::tools::base::render_result(
            tool.execute(params, &crate::agent::tools::base::ToolContext::sandbox())
                .await,
        );

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
        let result = crate::agent::tools::base::render_result(
            tool.execute(params, &crate::agent::tools::base::ToolContext::sandbox())
                .await,
        );

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

        let bare1 = crate::agent::tools::base::render_result(
            tool.execute(
                make_params(&[("path", p)]),
                &crate::agent::tools::base::ToolContext::sandbox(),
            )
            .await,
        );
        let bare2 = crate::agent::tools::base::render_result(
            tool.execute(
                make_params(&[("path", p)]),
                &crate::agent::tools::base::ToolContext::sandbox(),
            )
            .await,
        );
        assert_eq!(
            bare1, bare2,
            "bare read must be byte-identical across calls"
        );

        let ranged = |s: &str| {
            let mut m = make_params(&[("path", p)]);
            m.insert("lines".to_string(), serde_json::json!(s));
            m
        };
        let r1 = crate::agent::tools::base::render_result(
            tool.execute(
                ranged("100:200"),
                &crate::agent::tools::base::ToolContext::sandbox(),
            )
            .await,
        );
        let r2 = crate::agent::tools::base::render_result(
            tool.execute(
                ranged("100:200"),
                &crate::agent::tools::base::ToolContext::sandbox(),
            )
            .await,
        );
        assert_eq!(r1, r2, "ranged read must be byte-identical across calls");
    }

    #[tokio::test]
    async fn test_read_file_missing() {
        let tool = ReadFileTool::default();
        let params = make_params(&[("path", "/tmp/nonexistent_nanobot_test_file_xyz.txt")]);
        let result = crate::agent::tools::base::render_result(
            tool.execute(params, &crate::agent::tools::base::ToolContext::sandbox())
                .await,
        );
        assert!(result.starts_with("Error: File not found"));
    }

    #[tokio::test]
    async fn test_read_file_missing_param() {
        let tool = ReadFileTool::default();
        let params = HashMap::new();
        let result = crate::agent::tools::base::render_result(
            tool.execute(params, &crate::agent::tools::base::ToolContext::sandbox())
                .await,
        );
        assert!(result.contains("'path' parameter is required"));
    }

    #[tokio::test]
    async fn test_read_file_not_a_file() {
        let dir = TempDir::new().unwrap();
        let tool = ReadFileTool::default();
        let params = make_params(&[("path", dir.path().to_str().unwrap())]);
        let result = crate::agent::tools::base::render_result(
            tool.execute(params, &crate::agent::tools::base::ToolContext::sandbox())
                .await,
        );
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
        let result = crate::agent::tools::base::render_result(
            tool.execute(params, &crate::agent::tools::base::ToolContext::sandbox())
                .await,
        );

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
        let result = crate::agent::tools::base::render_result(
            tool.execute(params, &crate::agent::tools::base::ToolContext::sandbox())
                .await,
        );

        assert!(result.contains("lines 3-5 of 5"));
        assert!(result.contains("line 3"));
        assert!(result.contains("line 5"));
    }

    #[test]
    fn test_extract_line_range_basic() {
        let content = "alpha\nbeta\ngamma\ndelta\nepsilon";
        let hash = sha256_hex(content.as_bytes());
        let result = extract_line_range(
            content,
            "2:4",
            "test.txt",
            INLINE_TOOL_RESULT_MAX_BYTES,
            &hash,
        );
        assert!(result.contains("lines 2-4 of 5"));
        assert!(result.contains(&format!("sha256={hash}")));
        assert!(result.contains("beta"));
        assert!(result.contains("delta"));
    }

    #[test]
    fn test_extract_line_range_invalid_format() {
        let result = extract_line_range(
            "content",
            "bad",
            "test.txt",
            INLINE_TOOL_RESULT_MAX_BYTES,
            "hash",
        );
        assert!(result.contains("Error"));
    }

    #[test]
    fn test_extract_line_range_out_of_bounds() {
        let result = extract_line_range(
            "one\ntwo",
            "5:10",
            "test.txt",
            INLINE_TOOL_RESULT_MAX_BYTES,
            "hash",
        );
        assert!(result.contains("exceeds file length"));
    }

    #[test]
    fn render_range_bounds_to_char_budget_no_truncation() {
        // A realistic large source file: 5000 lines × ~40 chars. The old behavior
        // rendered the whole requested range and let the downstream cap head+tail-
        // truncate it. Now render_range itself bounds to complete lines within the
        // inline byte budget, so the output is whole, contiguous, and never mid-cut.
        let content: String = (1..=5000)
            .map(|i| format!("    let x_{i} = some_call(arg_one, arg_two);"))
            .collect::<Vec<_>>()
            .join("\n");
        let hash = sha256_hex(content.as_bytes());
        let out = render_range(
            &content,
            1,
            1000,
            "src/big_module.rs",
            5000,
            INLINE_TOOL_RESULT_MAX_BYTES,
            &hash,
        );

        assert!(
            !out.contains("[truncated:"),
            "read_file must never head+tail-truncate: {}",
            &out[..80.min(out.len())]
        );
        assert!(
            out.len() <= INLINE_TOOL_RESULT_MAX_BYTES,
            "output must stay within the inline byte limit, got {} bytes",
            out.len()
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
        // Regression: a smaller configured budget must bound the window so the
        // result stays within that byte cap. The full wrapper reservation keeps
        // the downstream renderer from inserting a head+tail gap.
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
            out.len() <= 2_500,
            "output must stay under the 2500-byte cap, got {} bytes",
            out.len()
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
        let result = crate::agent::tools::base::render_result(
            tool.execute(params, &crate::agent::tools::base::ToolContext::sandbox())
                .await,
        );
        assert!(result.starts_with("Successfully edited"));

        let content = std::fs::read_to_string(&file_path).unwrap();
        assert_eq!(content, "Hello Rust! This is a test.");
    }

    #[tokio::test]
    async fn test_edit_file_rejects_identical_old_and_new_text() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("no_op.txt");
        std::fs::write(&file_path, "unchanged").unwrap();

        let result = crate::agent::tools::base::render_result(
            EditFileTool::default()
                .execute(
                    make_params(&[
                        ("path", file_path.to_str().unwrap()),
                        ("old_text", "unchanged"),
                        ("new_text", "unchanged"),
                    ]),
                    &crate::agent::tools::base::ToolContext::sandbox(),
                )
                .await,
        );

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
        let result = crate::agent::tools::base::render_result(
            tool.execute(params, &crate::agent::tools::base::ToolContext::sandbox())
                .await,
        );
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
        let result = crate::agent::tools::base::render_result(
            tool.execute(params, &crate::agent::tools::base::ToolContext::sandbox())
                .await,
        );
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
        let result = crate::agent::tools::base::render_result(
            tool.execute(params, &crate::agent::tools::base::ToolContext::sandbox())
                .await,
        );
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
        let result = crate::agent::tools::base::render_result(
            tool.execute(params, &crate::agent::tools::base::ToolContext::sandbox())
                .await,
        );
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
        let result = crate::agent::tools::base::render_result(
            tool.execute(params, &crate::agent::tools::base::ToolContext::sandbox())
                .await,
        );
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
        let result = crate::agent::tools::base::render_result(
            tool.execute(params, &crate::agent::tools::base::ToolContext::sandbox())
                .await,
        );
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
        let result = crate::agent::tools::base::render_result(
            tool.execute(params, &crate::agent::tools::base::ToolContext::sandbox())
                .await,
        );
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
        let result = crate::agent::tools::base::render_result(
            tool.execute(params, &crate::agent::tools::base::ToolContext::sandbox())
                .await,
        );
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
        let result = crate::agent::tools::base::render_result(
            tool.execute(params, &crate::agent::tools::base::ToolContext::sandbox())
                .await,
        );

        assert!(result.contains("[file] file_a.txt"));
        assert!(result.contains("[file] file_b.txt"));
        assert!(result.contains("[dir]  subdir"));
    }

    #[tokio::test]
    async fn test_list_dir_empty() {
        let dir = TempDir::new().unwrap();

        let tool = ListDirTool;
        let params = make_params(&[("path", dir.path().to_str().unwrap())]);
        let result = crate::agent::tools::base::render_result(
            tool.execute(params, &crate::agent::tools::base::ToolContext::sandbox())
                .await,
        );
        assert!(result.contains("is empty"));
    }

    #[tokio::test]
    async fn test_list_dir_not_found() {
        let tool = ListDirTool;
        let params = make_params(&[("path", "/tmp/nonexistent_nanobot_dir_xyz")]);
        let result = crate::agent::tools::base::render_result(
            tool.execute(params, &crate::agent::tools::base::ToolContext::sandbox())
                .await,
        );
        assert!(result.starts_with("Error: Directory not found"));
    }

    #[tokio::test]
    async fn test_list_dir_not_a_directory() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("file.txt");
        std::fs::write(&file_path, "content").unwrap();

        let tool = ListDirTool;
        let params = make_params(&[("path", file_path.to_str().unwrap())]);
        let result = crate::agent::tools::base::render_result(
            tool.execute(params, &crate::agent::tools::base::ToolContext::sandbox())
                .await,
        );
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
        let result = crate::agent::tools::base::render_result(
            tool.execute(params, &crate::agent::tools::base::ToolContext::sandbox())
                .await,
        );
        assert!(result.contains("'path' parameter is required"));
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
        let result = crate::agent::tools::base::render_result(
            tool.execute(params, &crate::agent::tools::base::ToolContext::sandbox())
                .await,
        );

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
        let result = crate::agent::tools::base::render_result(
            tool.execute(params, &crate::agent::tools::base::ToolContext::sandbox())
                .await,
        );

        assert!(result.contains("a.txt:1: item 1"), "{result}");
        assert!(result.contains("a.txt:2: item 2"), "{result}");
        assert!(!result.contains("a.txt:3: item 3"), "{result}");
        assert!(result.contains("limit reached"), "{result}");
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
        let result = crate::agent::tools::base::render_result(
            tool.execute(params, &crate::agent::tools::base::ToolContext::sandbox())
                .await,
        );
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
        let result = crate::agent::tools::base::render_result(
            tool.execute(params, &crate::agent::tools::base::ToolContext::sandbox())
                .await,
        );
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
        let result = crate::agent::tools::base::render_result(
            tool.execute(params, &crate::agent::tools::base::ToolContext::sandbox())
                .await,
        );
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
        let result = crate::agent::tools::base::render_result(
            tool.execute(params, &crate::agent::tools::base::ToolContext::sandbox())
                .await,
        );
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
        let result = crate::agent::tools::base::render_result(
            tool.execute(params, &crate::agent::tools::base::ToolContext::sandbox())
                .await,
        );
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
        let result = crate::agent::tools::base::render_result(
            tool.execute(params, &crate::agent::tools::base::ToolContext::sandbox())
                .await,
        );
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
        let result = crate::agent::tools::base::render_result(
            tool.execute(params, &crate::agent::tools::base::ToolContext::sandbox())
                .await,
        );
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
