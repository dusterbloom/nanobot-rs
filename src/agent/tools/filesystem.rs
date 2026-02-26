//! File system tools: read, write, edit, list.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use async_trait::async_trait;

use super::base::Tool;

// ---------------------------------------------------------------------------
// ReadFileTool
// ---------------------------------------------------------------------------

/// Tool to read file contents.
pub struct ReadFileTool;

#[async_trait]
impl Tool for ReadFileTool {
    fn name(&self) -> &str {
        "read_file"
    }

    fn description(&self) -> &str {
        "Read the contents of a file at the given path. Optionally read a specific line range."
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
                    "description": "Optional line range to read, e.g. \"10:50\" (1-indexed, inclusive). Omit to read the entire file."
                }
            },
            "required": ["path"]
        })
    }

    async fn execute(&self, params: HashMap<String, serde_json::Value>) -> String {
        let path = match params.get("path").and_then(|v| v.as_str()) {
            Some(p) => p,
            None => return "Error: 'path' parameter is required".to_string(),
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
        let check_len = bytes.len().min(512);
        if bytes[..check_len].contains(&0u8) {
            return format!("[Binary file: {}, {} bytes]", path, bytes.len());
        }

        let content = String::from_utf8_lossy(&bytes).to_string();

        // If lines parameter is provided, extract the range.
        if let Some(lines_param) = params.get("lines").and_then(|v| v.as_str()) {
            return extract_line_range(&content, lines_param, path);
        }

        content
    }
}

// ---------------------------------------------------------------------------
// WriteFileTool
// ---------------------------------------------------------------------------

/// Tool to write content to a file.
pub struct WriteFileTool;

#[async_trait]
impl Tool for WriteFileTool {
    fn name(&self) -> &str {
        "write_file"
    }

    fn description(&self) -> &str {
        "Write content to a file at the given path. Creates parent directories if needed."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The file path to write to"
                },
                "content": {
                    "type": "string",
                    "description": "The content to write"
                }
            },
            "required": ["path", "content"]
        })
    }

    async fn execute(&self, params: HashMap<String, serde_json::Value>) -> String {
        let path = match params.get("path").and_then(|v| v.as_str()) {
            Some(p) => p,
            None => return "Error: 'path' parameter is required".to_string(),
        };
        let content = match params.get("content").and_then(|v| v.as_str()) {
            Some(c) => c,
            None => return "Error: 'content' parameter is required".to_string(),
        };

        let file_path = expand_path(path);

        // Create parent directories.
        if let Some(parent) = file_path.parent() {
            if let Err(e) = tokio::fs::create_dir_all(parent).await {
                return format!("Error creating directories: {}. Hint: check file permissions or try a different path.", e);
            }
        }

        match tokio::fs::write(&file_path, content).await {
            Ok(()) => format!("Successfully wrote {} bytes to {}", content.len(), path),
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
// EditFileTool
// ---------------------------------------------------------------------------

/// Tool to edit a file by replacing text.
pub struct EditFileTool;

#[async_trait]
impl Tool for EditFileTool {
    fn name(&self) -> &str {
        "edit_file"
    }

    fn description(&self) -> &str {
        "Edit a file by replacing old_text with new_text. The old_text must exist exactly in the file."
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
                }
            },
            "required": ["path", "old_text", "new_text"]
        })
    }

    async fn execute(&self, params: HashMap<String, serde_json::Value>) -> String {
        let path = match params.get("path").and_then(|v| v.as_str()) {
            Some(p) => p,
            None => return "Error: 'path' parameter is required".to_string(),
        };
        let old_text = match params.get("old_text").and_then(|v| v.as_str()) {
            Some(t) => t,
            None => return "Error: 'old_text' parameter is required".to_string(),
        };
        let new_text = match params.get("new_text").and_then(|v| v.as_str()) {
            Some(t) => t,
            None => return "Error: 'new_text' parameter is required".to_string(),
        };

        let file_path = expand_path(path);

        if !file_path.exists() {
            return format!("Error: File not found: {}. Hint: verify the path exists. Use list_dir to browse the directory.", path);
        }

        let content = match tokio::fs::read_to_string(&file_path).await {
            Ok(c) => c,
            Err(e) => return format!("Error reading file: {}", e),
        };

        if !content.contains(old_text) {
            return "Error: old_text not found in file. Make sure it matches exactly. Hint: use read_file to see the current file contents, then copy the exact text to match.".to_string();
        }

        // Count occurrences.
        let count = content.matches(old_text).count();
        if count > 1 {
            return format!(
                "Warning: old_text appears {} times. Please provide more context to make it unique.",
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
        let path = match params.get("path").and_then(|v| v.as_str()) {
            Some(p) => p,
            None => return "Error: 'path' parameter is required".to_string(),
        };

        let dir_path = expand_path(path);

        if !dir_path.exists() {
            return format!("Error: Directory not found: {}. Hint: parent directory does not exist. Use list_dir to find the correct path.", path);
        }
        if !dir_path.is_dir() {
            return format!("Error: Not a directory: {}. Hint: this path is a file, not a directory. Use read_file instead.", path);
        }

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

                lines.join("\n")
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
// Helpers
// ---------------------------------------------------------------------------

/// Extract a line range from content.
///
/// `range` format: "start:end" (1-indexed, inclusive) or "start:" (to end).
fn extract_line_range(content: &str, range: &str, path: &str) -> String {
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

    let selected: Vec<String> = lines[start - 1..end]
        .iter()
        .enumerate()
        .map(|(i, line)| format!("{:>4}: {}", start + i, line))
        .collect();

    format!(
        "# {} (lines {}-{} of {})\n{}",
        path,
        start,
        end,
        total,
        selected.join("\n")
    )
}

/// Expand a path: `~` → home dir, relative paths → workspace-relative.
///
/// Small/delegation models sometimes omit the full workspace prefix and
/// pass bare filenames like `MEMORY.md`. Resolving against the workspace
/// makes these succeed instead of failing with "File not found".
fn expand_path(path: &str) -> PathBuf {
    if let Some(rest) = path.strip_prefix("~/") {
        let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
        home.join(rest)
    } else if path == "~" {
        dirs::home_dir().unwrap_or_else(|| PathBuf::from("."))
    } else {
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
}

fn resolve_read_path(path: &str) -> PathBuf {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use tempfile::TempDir;

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

    fn make_params(pairs: &[(&str, &str)]) -> HashMap<String, serde_json::Value> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), serde_json::Value::String(v.to_string())))
            .collect()
    }

    #[tokio::test]
    async fn test_read_file_existing() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("hello.txt");
        std::fs::write(&file_path, "hello world").unwrap();

        let tool = ReadFileTool;
        let params = make_params(&[("path", file_path.to_str().unwrap())]);
        let result = tool.execute(params).await;
        assert_eq!(result, "hello world");
    }

    #[tokio::test]
    async fn test_read_file_missing() {
        let tool = ReadFileTool;
        let params = make_params(&[("path", "/tmp/nonexistent_nanobot_test_file_xyz.txt")]);
        let result = tool.execute(params).await;
        assert!(result.starts_with("Error: File not found"));
    }

    #[tokio::test]
    async fn test_read_file_missing_param() {
        let tool = ReadFileTool;
        let params = HashMap::new();
        let result = tool.execute(params).await;
        assert!(result.contains("'path' parameter is required"));
    }

    #[tokio::test]
    async fn test_read_file_not_a_file() {
        let dir = TempDir::new().unwrap();
        let tool = ReadFileTool;
        let params = make_params(&[("path", dir.path().to_str().unwrap())]);
        let result = tool.execute(params).await;
        assert!(result.starts_with("Error: Not a file"));
    }

    #[test]
    fn test_read_file_name() {
        let tool = ReadFileTool;
        assert_eq!(tool.name(), "read_file");
    }

    #[test]
    fn test_read_file_parameters_schema() {
        let tool = ReadFileTool;
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

        let tool = ReadFileTool;
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

        let tool = ReadFileTool;
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
        let result = extract_line_range(content, "2:4", "test.txt");
        assert!(result.contains("lines 2-4 of 5"));
        assert!(result.contains("beta"));
        assert!(result.contains("delta"));
    }

    #[test]
    fn test_extract_line_range_invalid_format() {
        let result = extract_line_range("content", "bad", "test.txt");
        assert!(result.contains("Error"));
    }

    #[test]
    fn test_extract_line_range_out_of_bounds() {
        let result = extract_line_range("one\ntwo", "5:10", "test.txt");
        assert!(result.contains("exceeds file length"));
    }

    // -----------------------------------------------------------------------
    // WriteFileTool tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_write_file_creates_file() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("output.txt");

        let tool = WriteFileTool;
        let params = make_params(&[
            ("path", file_path.to_str().unwrap()),
            ("content", "test content"),
        ]);
        let result = tool.execute(params).await;
        assert!(result.starts_with("Successfully wrote"));

        let content = std::fs::read_to_string(&file_path).unwrap();
        assert_eq!(content, "test content");
    }

    #[tokio::test]
    async fn test_write_file_creates_parent_dirs() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("subdir").join("nested").join("file.txt");

        let tool = WriteFileTool;
        let params = make_params(&[
            ("path", file_path.to_str().unwrap()),
            ("content", "nested content"),
        ]);
        let result = tool.execute(params).await;
        assert!(result.starts_with("Successfully wrote"));
        assert!(file_path.exists());
    }

    #[tokio::test]
    async fn test_write_file_missing_path() {
        let tool = WriteFileTool;
        let params = make_params(&[("content", "test")]);
        let result = tool.execute(params).await;
        assert!(result.contains("'path' parameter is required"));
    }

    #[tokio::test]
    async fn test_write_file_missing_content() {
        let tool = WriteFileTool;
        let params = make_params(&[("path", "/tmp/test.txt")]);
        let result = tool.execute(params).await;
        assert!(result.contains("'content' parameter is required"));
    }

    #[test]
    fn test_write_file_name() {
        let tool = WriteFileTool;
        assert_eq!(tool.name(), "write_file");
    }

    // -----------------------------------------------------------------------
    // EditFileTool tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_edit_file_replace_string() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("edit_me.txt");
        std::fs::write(&file_path, "Hello World! This is a test.").unwrap();

        let tool = EditFileTool;
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
    async fn test_edit_file_old_text_not_found() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("edit_me.txt");
        std::fs::write(&file_path, "Hello World!").unwrap();

        let tool = EditFileTool;
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

        let tool = EditFileTool;
        let params = make_params(&[
            ("path", file_path.to_str().unwrap()),
            ("old_text", "aaa"),
            ("new_text", "ccc"),
        ]);
        let result = tool.execute(params).await;
        assert!(result.contains("appears 2 times"));
    }

    #[tokio::test]
    async fn test_edit_file_missing_file() {
        let tool = EditFileTool;
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
        let tool = EditFileTool;
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

    #[tokio::test]
    async fn test_read_binary_file() {
        let dir = tempfile::tempdir().unwrap();
        let bin_path = dir.path().join("test.bin");
        // Write binary content with null bytes.
        std::fs::write(&bin_path, b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR").unwrap();

        let tool = ReadFileTool;
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
        let tool = ReadFileTool;
        let params = make_params(&[("path", "/tmp/nanobot_hint_test_nonexistent_xyz.txt")]);
        let result = tool.execute(params).await;
        assert!(result.contains("Hint:"), "Expected hint in error: {}", result);
        assert!(result.contains("list_dir"), "Expected list_dir hint: {}", result);
    }

    #[tokio::test]
    async fn test_read_file_not_a_file_has_hint() {
        let dir = TempDir::new().unwrap();
        let tool = ReadFileTool;
        let params = make_params(&[("path", dir.path().to_str().unwrap())]);
        let result = tool.execute(params).await;
        assert!(result.contains("Hint:"), "Expected hint in error: {}", result);
        assert!(result.contains("list_dir"), "Expected list_dir hint: {}", result);
    }

    #[tokio::test]
    async fn test_edit_file_not_found_has_hint() {
        let tool = EditFileTool;
        let params = make_params(&[
            ("path", "/tmp/nanobot_hint_test_nonexistent_edit_xyz.txt"),
            ("old_text", "a"),
            ("new_text", "b"),
        ]);
        let result = tool.execute(params).await;
        assert!(result.contains("Hint:"), "Expected hint in error: {}", result);
        assert!(result.contains("list_dir"), "Expected list_dir hint: {}", result);
    }

    #[tokio::test]
    async fn test_edit_file_old_text_not_found_has_hint() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("edit_hint.txt");
        std::fs::write(&file_path, "some existing content").unwrap();

        let tool = EditFileTool;
        let params = make_params(&[
            ("path", file_path.to_str().unwrap()),
            ("old_text", "text that does not exist in file"),
            ("new_text", "replacement"),
        ]);
        let result = tool.execute(params).await;
        assert!(result.contains("Hint:"), "Expected hint in error: {}", result);
        assert!(result.contains("read_file"), "Expected read_file hint: {}", result);
    }

    #[tokio::test]
    async fn test_list_dir_not_found_has_hint() {
        let tool = ListDirTool;
        let params = make_params(&[("path", "/tmp/nanobot_hint_test_nonexistent_dir_xyz")]);
        let result = tool.execute(params).await;
        assert!(result.contains("Hint:"), "Expected hint in error: {}", result);
        assert!(result.contains("list_dir"), "Expected list_dir hint: {}", result);
    }

    #[tokio::test]
    async fn test_list_dir_not_a_directory_has_hint() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("not_a_dir.txt");
        std::fs::write(&file_path, "content").unwrap();

        let tool = ListDirTool;
        let params = make_params(&[("path", file_path.to_str().unwrap())]);
        let result = tool.execute(params).await;
        assert!(result.contains("Hint:"), "Expected hint in error: {}", result);
        assert!(result.contains("read_file"), "Expected read_file hint: {}", result);
    }
}
