// Error-protocol layer-3 backlog (docs/research/2026-08-06-error-conventions-and-host-bridge.md §3.6):
// the deny regime in Cargo.toml is live; this module still carries pre-existing
// violations of the lints below. Remove this allow as the module migrates onto
// the regime.
// Tracking: docs/error-protocol-backlog.md
#![allow(clippy::indexing_slicing, clippy::shadow_reuse)]
//! Unified-diff patch tool and reusable patch validator.

use std::collections::HashMap;
use std::path::PathBuf;

use async_trait::async_trait;
use serde_json::{json, Value};

use super::base::{PermissionLevel, Tool, ToolContext, ToolResult};
use super::filesystem::{expand_path, idle_write_denied_err, normalize_lexical, sha256_hex};
use crate::errors::ToolError;

/// Tool to validate or apply unified diffs across one or more files.
pub struct ApplyPatchTool {
    /// Idle-turn write allowlist; `None` on normal turns.
    pub idle_paths: Option<Vec<String>>,
}

impl Default for ApplyPatchTool {
    fn default() -> Self {
        Self { idle_paths: None }
    }
}

impl ApplyPatchTool {
    pub fn new(idle_paths: Option<Vec<String>>) -> Self {
        Self { idle_paths }
    }
}

#[async_trait]
impl Tool for ApplyPatchTool {
    fn name(&self) -> &str {
        "apply_patch"
    }

    fn description(&self) -> &str {
        "Apply or dry-run a unified diff. Validates all hunks before writing and reports stale-hunk diagnostics with file and hunk context."
    }

    fn permission(&self) -> PermissionLevel {
        PermissionLevel::Write
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "patch": {
                    "type": "string",
                    "description": "Unified diff containing one or more file patches."
                },
                "dry_run": {
                    "type": "boolean",
                    "description": "Validate and summarize without writing. Default: false."
                },
                "expected_sha256_by_path": {
                    "type": "object",
                    "description": "Optional map from patch path to expected SHA-256; rejects changed files before applying."
                }
            },
            "required": ["patch"]
        })
    }

    async fn execute(&self, params: HashMap<String, Value>, _ctx: &ToolContext) -> ToolResult {
        let patch = match params.get("patch").and_then(|v| v.as_str()) {
            Some(p) if !p.trim().is_empty() => p,
            _ => {
                return Err(ToolError::InvalidArgs {
                    message: "'patch' parameter is required and must be non-empty".to_string(),
                })
            }
        };
        let dry_run = params
            .get("dry_run")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let expected = params
            .get("expected_sha256_by_path")
            .and_then(|v| v.as_object())
            .cloned()
            .unwrap_or_default();

        let file_patches = match parse_file_patches(patch) {
            Ok(patches) => patches,
            // Parser errors carry the legacy "Error: " prefix; strip it so
            // render() re-prefixes exactly once.
            Err(e) => {
                return Err(ToolError::Execution {
                    message: e.trim_start_matches("Error: ").to_string(),
                })
            }
        };

        let mut updates: Vec<(PathBuf, String, usize, usize)> = Vec::new();
        let mut lines = vec![format!(
            "{} {} file patch(es)",
            if dry_run { "Validated" } else { "Applying" },
            file_patches.len()
        )];

        for fp in file_patches {
            let mut path = expand_path(&fp.path);
            // Idle turns gate every patched path: hunks name targets the
            // registry-level param check can never see. Normalize `..`/`.`
            // lexically before the allowlist check and reuse the normalized
            // path for the read/write so the on-disk operation cannot escape
            // the boundary via OS-level `..` resolution.
            if let Some(paths) = &self.idle_paths {
                let workspace = crate::utils::helpers::get_workspace_path(None);
                path = normalize_lexical(&path);
                if !super::filesystem::idle_write_allowed(paths, &path, &workspace) {
                    return Err(idle_write_denied_err(&path));
                }
            }
            let content = if path.exists() {
                match tokio::fs::read(&path).await {
                    Ok(bytes) => {
                        if crate::utils::helpers::is_binary(&bytes) {
                            return Err(ToolError::InvalidArgs {
                                message: format!("{} is binary; refusing to patch", fp.path),
                            });
                        }
                        let current_hash = sha256_hex(&bytes);
                        if let Some(expected_hash) =
                            expected_hash_for_path(&expected, &fp.path, &path)
                        {
                            if !expected_hash.eq_ignore_ascii_case(&current_hash) {
                                // "expected_sha256" matched the legacy
                                // InvalidArgs classifier; keep the kind.
                                return Err(ToolError::InvalidArgs {
                                    message: format!(
                                        "File changed before patch. path={}, expected_sha256={}, actual_sha256={}",
                                        fp.path, expected_hash, current_hash
                                    ),
                                });
                            }
                        }
                        String::from_utf8_lossy(&bytes).to_string()
                    }
                    // Legacy quirk preserved: no "Error:" prefix → success.
                    Err(e) => return Ok(format!("Error reading {}: {}", fp.path, e).into()),
                }
            } else {
                String::new()
            };

            let before_lines = content.lines().count();
            let (updated, hunks) = match apply_unified_patch_to_content(&content, &fp.patch) {
                Ok(result) => result,
                Err(e) => {
                    return Err(ToolError::Execution {
                        message: format!("{}: {}", fp.path, e.trim_start_matches("Error: ")),
                    })
                }
            };
            let after_lines = updated.lines().count();
            lines.push(format!(
                "  {}: {} hunk(s), {} -> {} line(s)",
                fp.path, hunks, before_lines, after_lines
            ));
            updates.push((path, updated, hunks, after_lines));
        }

        if dry_run {
            return Ok(lines.join("\n").into());
        }

        for (path, updated, _, _) in updates {
            if let Some(parent) = path.parent() {
                if let Err(e) = tokio::fs::create_dir_all(parent).await {
                    // Legacy quirk preserved: no "Error:" prefix → success.
                    return Ok(format!("Error creating {}: {}", parent.display(), e).into());
                }
            }
            if let Err(e) = tokio::fs::write(&path, updated).await {
                // Legacy quirk preserved: no "Error:" prefix → success.
                return Ok(format!("Error writing {}: {}", path.display(), e).into());
            }
        }

        lines.push("Patch applied successfully".to_string());
        Ok(lines.join("\n").into())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FilePatch {
    path: String,
    patch: String,
}

fn expected_hash_for_path(
    expected: &serde_json::Map<String, Value>,
    patch_path: &str,
    resolved: &PathBuf,
) -> Option<String> {
    expected
        .get(patch_path)
        .or_else(|| expected.get(&resolved.display().to_string()))
        .and_then(|v| v.as_str())
        .map(|s| s.trim().to_string())
}

fn parse_file_patches(patch: &str) -> Result<Vec<FilePatch>, String> {
    let normalized = patch.replace("\r\n", "\n");
    let mut patches = Vec::new();
    let mut current_path: Option<String> = None;
    let mut current_lines: Vec<String> = Vec::new();

    for raw in normalized.lines() {
        if raw == "*** Begin Patch" || raw == "*** End Patch" {
            continue;
        }
        if raw.starts_with("diff --git ") {
            push_file_patch(&mut patches, &mut current_path, &mut current_lines);
        }
        if let Some(path) = raw.strip_prefix("+++ ") {
            if path.trim() != "/dev/null" {
                current_path = Some(normalize_patch_path(path.trim()));
            }
        }
        if current_path.is_some() {
            current_lines.push(raw.to_string());
        }
    }
    push_file_patch(&mut patches, &mut current_path, &mut current_lines);

    if patches.is_empty() {
        return Err(
            "Error: patch must include unified diff file headers with +++ paths".to_string(),
        );
    }
    Ok(patches)
}

fn push_file_patch(
    patches: &mut Vec<FilePatch>,
    current_path: &mut Option<String>,
    current_lines: &mut Vec<String>,
) {
    let Some(path) = current_path.take() else {
        current_lines.clear();
        return;
    };
    if current_lines.iter().any(|line| line.starts_with("@@")) {
        patches.push(FilePatch {
            path,
            patch: current_lines.join("\n"),
        });
    }
    current_lines.clear();
}

fn normalize_patch_path(path: &str) -> String {
    path.strip_prefix("b/")
        .or_else(|| path.strip_prefix("a/"))
        .unwrap_or(path)
        .to_string()
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum PatchLine {
    Context(String),
    Remove(String),
    Add(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PatchHunk {
    old_start: usize,
    header: String,
    lines: Vec<PatchLine>,
}

/// Apply unified diff hunk(s) to a single file content string.
pub(crate) fn apply_unified_patch_to_content(
    content: &str,
    patch: &str,
) -> Result<(String, usize), String> {
    let hunks = parse_unified_patch(patch)?;
    if hunks.is_empty() {
        return Err("Error: patch contains no @@ hunks".to_string());
    }

    let newline = if content.contains("\r\n") {
        "\r\n"
    } else {
        "\n"
    };
    let normalized = content.replace("\r\n", "\n");
    let had_final_newline = normalized.ends_with('\n');
    let original: Vec<String> = if normalized.is_empty() {
        Vec::new()
    } else {
        normalized.lines().map(str::to_string).collect()
    };

    let mut result = Vec::new();
    let mut src_index = 0usize;

    for hunk in &hunks {
        let old_lines = hunk_old_lines(hunk);
        let preferred = hunk.old_start.saturating_sub(1);
        let match_index = if old_lines.is_empty() {
            preferred.clamp(src_index, original.len())
        } else {
            find_hunk_match(&original, &old_lines, preferred, src_index)
                .ok_or_else(|| stale_hunk_error(hunk, &old_lines, &original))?
        };

        if match_index < src_index {
            return Err(format!(
                "Error: patch hunks overlap near original line {} ({})",
                hunk.old_start, hunk.header
            ));
        }
        result.extend_from_slice(&original[src_index..match_index]);

        let mut hunk_src = match_index;
        for line in &hunk.lines {
            match line {
                PatchLine::Context(text) => {
                    if original.get(hunk_src) != Some(text) {
                        return Err(format!(
                            "Error: patch context mismatch near line {} in {}",
                            hunk_src + 1,
                            hunk.header
                        ));
                    }
                    result.push(text.clone());
                    hunk_src += 1;
                }
                PatchLine::Remove(text) => {
                    if original.get(hunk_src) != Some(text) {
                        return Err(format!(
                            "Error: patch removal mismatch near line {} in {}",
                            hunk_src + 1,
                            hunk.header
                        ));
                    }
                    hunk_src += 1;
                }
                PatchLine::Add(text) => result.push(text.clone()),
            }
        }
        src_index = hunk_src;
    }

    result.extend_from_slice(&original[src_index..]);
    let mut updated = result.join(newline);
    if had_final_newline && !updated.is_empty() {
        updated.push_str(newline);
    }
    Ok((updated, hunks.len()))
}

fn stale_hunk_error(hunk: &PatchHunk, old_lines: &[String], original: &[String]) -> String {
    let expected = old_lines
        .iter()
        .filter(|line| !line.trim().is_empty())
        .take(8)
        .cloned()
        .collect::<Vec<_>>()
        .join("\\n");
    let nearest_needle = old_lines
        .iter()
        .find(|line| !line.trim().is_empty())
        .or_else(|| old_lines.first())
        .map(String::as_str)
        .unwrap_or("");
    let nearest = if nearest_needle.is_empty() {
        None
    } else {
        original
            .iter()
            .position(|line| {
                line.trim() == nearest_needle.trim() || line.contains(nearest_needle.trim())
            })
            .map(|idx| idx + 1)
    };
    let expected = if expected.is_empty() {
        nearest_needle.to_string()
    } else {
        expected
    };
    match nearest {
        Some(line) => format!(
            "Error: stale patch hunk {} did not match. Expected context {:?}; nearest similar line is {}. Re-read the file and regenerate the diff.",
            hunk.header, expected, line
        ),
        None => format!(
            "Error: stale patch hunk {} did not match. Expected context {:?}; no similar line found. Re-read the file and regenerate the diff.",
            hunk.header, expected
        ),
    }
}

fn parse_unified_patch(patch: &str) -> Result<Vec<PatchHunk>, String> {
    let normalized = patch.replace("\r\n", "\n");
    let mut hunks = Vec::new();
    let mut current: Option<PatchHunk> = None;

    for raw in normalized.lines() {
        if raw.starts_with("--- ") || raw.starts_with("+++ ") || raw.starts_with("diff ") {
            continue;
        }
        if raw.starts_with("@@") {
            if let Some(hunk) = current.take() {
                hunks.push(hunk);
            }
            current = Some(PatchHunk {
                old_start: parse_hunk_old_start(raw)?,
                header: raw.to_string(),
                lines: Vec::new(),
            });
            continue;
        }

        let Some(hunk) = current.as_mut() else {
            continue;
        };
        if raw == r"\ No newline at end of file" {
            continue;
        }
        let mut chars = raw.chars();
        let Some(prefix) = chars.next() else {
            return Err("Error: malformed patch line without prefix".to_string());
        };
        let text = chars.as_str().to_string();
        match prefix {
            ' ' => hunk.lines.push(PatchLine::Context(text)),
            '-' => hunk.lines.push(PatchLine::Remove(text)),
            '+' => hunk.lines.push(PatchLine::Add(text)),
            _ => {
                return Err(format!(
                    "Error: malformed patch line '{}'. Lines inside hunks must start with space, '-', or '+'.",
                    raw
                ))
            }
        }
    }

    if let Some(hunk) = current {
        hunks.push(hunk);
    }
    Ok(hunks)
}

fn parse_hunk_old_start(header: &str) -> Result<usize, String> {
    let old_part = header
        .split_whitespace()
        .find(|part| part.starts_with('-'))
        .ok_or_else(|| format!("Error: malformed hunk header '{}'", header))?;
    let start = old_part
        .trim_start_matches('-')
        .split(',')
        .next()
        .unwrap_or("1")
        .parse::<usize>()
        .map_err(|_| format!("Error: malformed hunk header '{}'", header))?;
    Ok(start.max(1))
}

fn hunk_old_lines(hunk: &PatchHunk) -> Vec<String> {
    hunk.lines
        .iter()
        .filter_map(|line| match line {
            PatchLine::Context(text) | PatchLine::Remove(text) => Some(text.clone()),
            PatchLine::Add(_) => None,
        })
        .collect()
}

fn find_hunk_match(
    original: &[String],
    old_lines: &[String],
    preferred: usize,
    min_index: usize,
) -> Option<usize> {
    if old_lines.is_empty() {
        return Some(preferred.clamp(min_index, original.len()));
    }
    if preferred >= min_index && lines_match_at(original, old_lines, preferred) {
        return Some(preferred);
    }
    let max_start = original.len().saturating_sub(old_lines.len());
    (min_index..=max_start).find(|&idx| lines_match_at(original, old_lines, idx))
}

fn lines_match_at(original: &[String], needle: &[String], start: usize) -> bool {
    start + needle.len() <= original.len() && original[start..start + needle.len()] == *needle
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_apply_patch_single_file_dry_run_validates() {
        let patch = "--- a/demo.txt\n+++ b/demo.txt\n@@ -1,2 +1,2 @@\n one\n-two\n+TWO\n";
        let patches = parse_file_patches(patch).unwrap();
        assert_eq!(patches[0].path, "demo.txt");
        let (updated, hunks) =
            apply_unified_patch_to_content("one\ntwo\n", &patches[0].patch).unwrap();
        assert_eq!(hunks, 1);
        assert_eq!(updated, "one\nTWO\n");
    }

    #[test]
    fn test_apply_patch_reports_stale_hunk_with_context() {
        let patch = "@@ -1,2 +1,2 @@\n one\n-missing\n+new\n";
        let err = apply_unified_patch_to_content("one\ntwo\n", patch).unwrap_err();
        assert!(err.contains("stale patch hunk"), "{err}");
        assert!(err.contains("missing"), "{err}");
    }

    #[tokio::test]
    async fn test_apply_patch_tool_rejects_changed_sha() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("demo.txt");
        tokio::fs::write(&path, "one\ntwo\n").await.unwrap();
        let patch = format!(
            "--- a/{0}\n+++ b/{0}\n@@ -1,2 +1,2 @@\n one\n-two\n+TWO\n",
            path.display()
        );
        let mut expected = serde_json::Map::new();
        expected.insert(path.display().to_string(), json!("deadbeef"));
        let mut params = HashMap::new();
        params.insert("patch".to_string(), json!(patch));
        params.insert(
            "expected_sha256_by_path".to_string(),
            Value::Object(expected),
        );
        let out = crate::agent::tools::base::render_result(
            ApplyPatchTool::default()
                .execute(params, &crate::agent::tools::base::ToolContext::sandbox())
                .await,
        );
        assert!(out.contains("File changed before patch"), "{out}");
    }

    #[tokio::test]
    async fn apply_patch_tool_idle_denies_traversal() {
        // apply_patch shares the idle_write_allowed gate; the bypass must not
        // transfer. A patch whose `+++` target lexically begins with an
        // allowlisted absolute subtree but resolves via ".." outside it must be
        // denied before any file is read or written.
        let sandbox = tempfile::tempdir().unwrap();
        let base = sandbox.path().join("skills");
        std::fs::create_dir_all(&base).unwrap();
        let tool = ApplyPatchTool::new(Some(vec![format!("{}/**", base.display())]));
        let escape = tempfile::tempdir().unwrap();
        let escape_file = escape.path().join("escaped.txt");

        let mut malicious = base.clone();
        for _ in 0..base.components().count() {
            malicious.push("..");
        }
        malicious.push(escape_file.strip_prefix("/").unwrap());

        let patch = format!(
            "--- a/x\n+++ {malicious}\n@@ -1,1 +1,1 @@\n-old\n+new\n",
            malicious = malicious.to_str().unwrap(),
        );
        let mut params = HashMap::new();
        params.insert("patch".to_string(), json!(patch));
        let out = crate::agent::tools::base::render_result(
            tool.execute(params, &crate::agent::tools::base::ToolContext::sandbox())
                .await,
        );
        assert!(
            out.starts_with("Error: idle turns may only write"),
            "apply_patch idle allowlist must deny traversal, got: {out}"
        );
        assert!(
            !escape_file.exists(),
            "apply_patch must not write via traversal"
        );
    }
}
