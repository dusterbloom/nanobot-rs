//! `write_file` tool.

use std::collections::HashMap;

use async_trait::async_trait;

use super::super::base::{PermissionLevel, Tool};
use super::{expand_path, require_param};

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

    fn permission(&self) -> PermissionLevel {
        PermissionLevel::Write
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
        let path = match require_param(&params, "path") {
            Ok(p) => p,
            Err(e) => return e,
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

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::make_params;
    use tempfile::TempDir;

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
}
