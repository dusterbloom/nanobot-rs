// Error-protocol layer-3 backlog (docs/research/2026-08-06-error-conventions-and-host-bridge.md §3.6):
// the deny regime in Cargo.toml is live; this module still carries pre-existing
// violations of the lints below. Remove this allow as the module migrates onto
// the regime.
// Tracking: docs/error-protocol-backlog.md
#![allow(clippy::indexing_slicing, clippy::shadow_reuse)]
//! Reusable unified-diff patch validator.
//!
//! The standalone `apply_patch` Tool surface was removed (lean-ecosystem
//! Task 3); this module now only holds `apply_unified_patch_to_content`,
//! which `EditFileTool`'s patch mode (`filesystem/mod.rs`) still calls.

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
        if current.is_none()
            && (raw.starts_with("--- ") || raw.starts_with("+++ ") || raw.starts_with("diff "))
        {
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

    #[test]
    fn test_apply_patch_single_file_dry_run_validates() {
        let patch = "@@ -1,2 +1,2 @@\n one\n-two\n+TWO\n";
        let (updated, hunks) = apply_unified_patch_to_content("one\ntwo\n", patch).unwrap();
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

    #[test]
    fn patch_removes_double_dash_comment() {
        let content = "-- comment\nkeep\n";
        let patch = "@@ -1,2 +1,1 @@\n--- comment\n keep\n";
        let (updated, _) = apply_unified_patch_to_content(content, patch).unwrap();
        assert_eq!(updated, "keep\n");
    }
}
