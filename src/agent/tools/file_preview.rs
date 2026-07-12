//! File preview tool: metadata, snippets, outline, and ranged-read hints.

use std::collections::HashMap;
use std::path::Path;

use async_trait::async_trait;
use serde_json::{json, Value};

use super::base::{Tool, ToolConcurrency};
use super::filesystem::{resolve_read_path, sha256_hex};

const SNIPPET_LINES: usize = 12;
const OUTLINE_LIMIT: usize = 40;

/// Tool to inspect file shape before choosing ranges to read.
pub struct FilePreviewTool;

#[async_trait]
impl Tool for FilePreviewTool {
    fn name(&self) -> &str {
        "file_preview"
    }

    fn description(&self) -> &str {
        "Preview a file before reading it: metadata, line count, hash, small head/tail snippets, simple outline, and suggested read_file ranges."
    }

    fn concurrency(&self) -> ToolConcurrency {
        ToolConcurrency::ParallelSafe
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path to preview"
                },
                "outline_limit": {
                    "type": "integer",
                    "description": "Maximum outline entries. Default: 40, max: 200"
                }
            },
            "required": ["path"]
        })
    }

    async fn execute(&self, params: HashMap<String, Value>) -> String {
        let path = match params.get("path").and_then(|v| v.as_str()) {
            Some(p) if !p.trim().is_empty() => p,
            _ => return "Error: 'path' parameter is required".to_string(),
        };
        let outline_limit = params
            .get("outline_limit")
            .and_then(|v| v.as_u64())
            .map(|v| (v as usize).clamp(1, 200))
            .unwrap_or(OUTLINE_LIMIT);

        let file_path = resolve_read_path(path);
        let metadata = match tokio::fs::metadata(&file_path).await {
            Ok(m) => m,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return format!("Error: File not found: {}", path)
            }
            Err(e) => return format!("Error reading metadata: {}", e),
        };
        if !metadata.is_file() {
            return format!("Error: Not a file: {}", path);
        }

        let bytes = match tokio::fs::read(&file_path).await {
            Ok(b) => b,
            Err(e) => return format!("Error reading file: {}", e),
        };
        let lang = detect_language(path);
        let mut out = format!(
            "# File preview: {}\nResolved: {}\nType: file\nSize: {} bytes\nLanguage: {}\nSHA-256: {}",
            path,
            file_path.display(),
            metadata.len(),
            lang,
            sha256_hex(&bytes)
        );

        if bytes.is_empty() {
            out.push_str("\nEmpty: true\nLines: 0");
            return out;
        }
        if crate::utils::helpers::is_binary(&bytes) {
            out.push_str("\nBinary: true");
            return out;
        }

        let content = String::from_utf8_lossy(&bytes);
        let lines: Vec<&str> = content.lines().collect();
        let total = lines.len();
        out.push_str(&format!("\nBinary: false\nEmpty: false\nLines: {}", total));

        out.push_str("\n\n## Suggested ranges");
        for range in suggested_ranges(total) {
            out.push_str(&format!(
                "\nread_file {{\"path\":\"{}\",\"lines\":\"{}\"}}",
                path, range
            ));
        }

        let outline = outline_lines(path, &lines, outline_limit);
        if !outline.is_empty() {
            out.push_str("\n\n## Outline");
            for line in outline {
                out.push('\n');
                out.push_str(&line);
            }
        }

        out.push_str("\n\n## Head");
        append_numbered_snippet(&mut out, &lines, 1, SNIPPET_LINES.min(total));
        if total > SNIPPET_LINES {
            let tail_start = total.saturating_sub(SNIPPET_LINES) + 1;
            out.push_str("\n\n## Tail");
            append_numbered_snippet(&mut out, &lines, tail_start, total);
        }
        out
    }
}

fn suggested_ranges(total: usize) -> Vec<String> {
    if total == 0 {
        return Vec::new();
    }
    let first = total.min(200);
    let mut ranges = vec![format!("1:{first}")];
    if total > 400 {
        let mid = total / 2;
        ranges.push(format!(
            "{}:{}",
            mid.saturating_sub(100).max(1),
            (mid + 100).min(total)
        ));
    }
    if total > 200 {
        ranges.push(format!("{}:{}", total.saturating_sub(199).max(1), total));
    }
    ranges
}

fn append_numbered_snippet(out: &mut String, lines: &[&str], start: usize, end: usize) {
    if start == 0 || end == 0 || start > end {
        return;
    }
    for idx in start..=end {
        if let Some(line) = lines.get(idx - 1) {
            out.push_str(&format!("\n{:>4}: {}", idx, truncate(line, 240)));
        }
    }
}

fn outline_lines(path: &str, lines: &[&str], limit: usize) -> Vec<String> {
    let lang = detect_language(path);
    let mut out = Vec::new();
    for (idx, line) in lines.iter().enumerate() {
        if let Some(label) = outline_label(lang, line) {
            out.push(format!("{:>4}: {}", idx + 1, label));
            if out.len() >= limit {
                break;
            }
        }
    }
    out
}

fn outline_label(lang: &str, line: &str) -> Option<String> {
    let trimmed = line.trim_start();
    match lang {
        "markdown" => trimmed
            .starts_with('#')
            .then(|| trimmed.chars().take(120).collect()),
        "rust" => {
            let rest = trimmed.strip_prefix("pub ").unwrap_or(trimmed);
            ["fn ", "struct ", "enum ", "trait ", "impl ", "mod "]
                .iter()
                .any(|prefix| rest.starts_with(prefix))
                .then(|| trimmed.chars().take(160).collect())
        }
        "python" => ["def ", "class "]
            .iter()
            .any(|prefix| trimmed.starts_with(prefix))
            .then(|| trimmed.chars().take(160).collect()),
        "javascript" | "typescript" => {
            let rest = trimmed.strip_prefix("export ").unwrap_or(trimmed);
            ["function ", "class ", "const ", "let ", "async function "]
                .iter()
                .any(|prefix| rest.starts_with(prefix))
                .then(|| trimmed.chars().take(160).collect())
        }
        _ => None,
    }
}

fn detect_language(path: &str) -> &'static str {
    match Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
    {
        "rs" => "rust",
        "py" => "python",
        "js" | "jsx" => "javascript",
        "ts" | "tsx" => "typescript",
        "md" | "markdown" => "markdown",
        "json" => "json",
        "toml" => "toml",
        "yaml" | "yml" => "yaml",
        "sh" | "zsh" | "bash" => "shell",
        _ => "text",
    }
}

fn truncate(line: &str, max: usize) -> String {
    if line.chars().count() <= max {
        return line.to_string();
    }
    let mut out: String = line.chars().take(max).collect();
    out.push_str("...");
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn test_file_preview_reports_outline_and_ranges() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("lib.rs");
        tokio::fs::write(
            &path,
            "pub struct Thing {}\n\nimpl Thing {\n    pub fn run(&self) {}\n}\n",
        )
        .await
        .unwrap();
        let mut params = HashMap::new();
        params.insert("path".to_string(), json!(path.display().to_string()));
        let out = FilePreviewTool.execute(params).await;
        assert!(out.contains("Language: rust"), "{out}");
        assert!(out.contains("Lines: 5"), "{out}");
        assert!(out.contains("pub struct Thing"), "{out}");
        assert!(out.contains("read_file"), "{out}");
    }
}
