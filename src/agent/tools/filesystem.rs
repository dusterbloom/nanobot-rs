//! File system tools: read, write, edit, list.

use std::collections::HashMap;
use std::path::PathBuf;

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
        "Read the contents of a file at the given path."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The file path to read"
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

        let file_path = expand_path(path);

        if !file_path.exists() {
            return format!("Error: File not found: {}", path);
        }
        if !file_path.is_file() {
            return format!("Error: Not a file: {}", path);
        }

        match tokio::fs::read_to_string(&file_path).await {
            Ok(content) => content,
            Err(e) => {
                if e.kind() == std::io::ErrorKind::PermissionDenied {
                    format!("Error: Permission denied: {}", path)
                } else {
                    format!("Error reading file: {}", e)
                }
            }
        }
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
                return format!("Error creating directories: {}", e);
            }
        }

        match tokio::fs::write(&file_path, content).await {
            Ok(()) => format!("Successfully wrote {} bytes to {}", content.len(), path),
            Err(e) => {
                if e.kind() == std::io::ErrorKind::PermissionDenied {
                    format!("Error: Permission denied: {}", path)
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
            return format!("Error: File not found: {}", path);
        }

        let content = match tokio::fs::read_to_string(&file_path).await {
            Ok(c) => c,
            Err(e) => return format!("Error reading file: {}", e),
        };

        if !content.contains(old_text) {
            return "Error: old_text not found in file. Make sure it matches exactly.".to_string();
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
                    format!("Error: Permission denied: {}", path)
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
            return format!("Error: Directory not found: {}", path);
        }
        if !dir_path.is_dir() {
            return format!("Error: Not a directory: {}", path);
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
                    format!("Error: Permission denied: {}", path)
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

/// Expand a leading `~` to the user's home directory.
fn expand_path(path: &str) -> PathBuf {
    if let Some(rest) = path.strip_prefix("~/") {
        let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
        home.join(rest)
    } else if path == "~" {
        dirs::home_dir().unwrap_or_else(|| PathBuf::from("."))
    } else {
        PathBuf::from(path)
    }
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
}
