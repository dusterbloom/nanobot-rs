//! `write_file` tool.

use std::collections::HashMap;
use std::path::{Component, Path, PathBuf};

use async_trait::async_trait;
use tokio::io::AsyncWriteExt;

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

        let file_path = expand_write_path(path);

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

/// Tool to write large generated files in bounded chunks.
pub struct WriteFileChunkTool;

#[async_trait]
impl Tool for WriteFileChunkTool {
    fn name(&self) -> &str {
        "write_file_chunk"
    }

    fn description(&self) -> &str {
        "Chunk-write rich files. start stages; append/finish need expected_offset; finish publishes."
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
                    "description": "Target path"
                },
                "mode": {
                    "type": "string",
                    "enum": ["start", "append", "finish"],
                    "description": "start|append|finish"
                },
                "content": {
                    "type": "string",
                    "maxLength": 4096,
                    "description": "Chunk <=4096 bytes"
                },
                "expected_offset": {
                    "type": "integer",
                    "description": "Prior next expected_offset"
                },
                "final_sha256": {
                    "type": "string",
                    "description": "Optional SHA-256"
                }
            },
            "required": ["path", "mode", "content"]
        })
    }

    async fn execute(&self, params: HashMap<String, serde_json::Value>) -> String {
        let path = match require_param(&params, "path") {
            Ok(p) => p,
            Err(e) => return e,
        };
        let mode = match require_param(&params, "mode") {
            Ok(m) => m.trim().to_ascii_lowercase(),
            Err(e) => return e,
        };
        let content = match params.get("content").and_then(|v| v.as_str()) {
            Some(c) => c,
            None => return "Error: 'content' parameter is required".to_string(),
        };
        if !matches!(mode.as_str(), "start" | "append" | "finish") {
            return "Error: 'mode' must be one of: start, append, finish".to_string();
        }
        if content.len() > 4096 {
            return format!(
                "Error: write_file_chunk content is {} bytes; split it into chunks of 4096 bytes or less.",
                content.len()
            );
        }

        let target_path = expand_write_path(path);
        if let Some(parent) = target_path.parent() {
            if let Err(e) = tokio::fs::create_dir_all(parent).await {
                return format!("Error creating directories: {}. Hint: check file permissions or try a different path.", e);
            }
        }

        let stage_path = staged_chunk_path(&target_path);
        let wrote = content.len();
        let write_result = if mode == "start" {
            tokio::fs::write(&stage_path, content).await
        } else {
            if !stage_path.exists() {
                return format!(
                    "Error: Cannot {mode} chunk because no staged write exists for {path}. Call write_file_chunk with mode=start first."
                );
            }
            let Some(expected_offset) = params.get("expected_offset").and_then(|v| v.as_u64())
            else {
                return format!(
                    "Error: expected_offset is required for mode={mode}. Use the next expected_offset returned by the previous write_file_chunk call."
                );
            };
            let current_offset = tokio::fs::metadata(&stage_path)
                .await
                .map(|m| m.len())
                .unwrap_or(0);
            if current_offset != expected_offset {
                return format!(
                    "Error: staged write offset mismatch for {path}. expected_offset={expected_offset}, actual_offset={current_offset}. Re-read the latest write_file_chunk receipt before continuing."
                );
            }
            match tokio::fs::OpenOptions::new()
                .append(true)
                .open(&stage_path)
                .await
            {
                Ok(mut file) => file.write_all(content.as_bytes()).await,
                Err(e) => Err(e),
            }
        };

        if let Err(e) = write_result {
            return if e.kind() == std::io::ErrorKind::PermissionDenied {
                format!("Error: Permission denied: {}. Hint: check file permissions or try a different path.", path)
            } else {
                format!("Error writing file chunk: {}", e)
            };
        }

        let total = tokio::fs::metadata(&stage_path)
            .await
            .map(|m| m.len())
            .unwrap_or(0);
        if mode == "finish" {
            let bytes = match tokio::fs::read(&stage_path).await {
                Ok(bytes) => bytes,
                Err(e) => return format!("Error reading staged file before publish: {}", e),
            };
            let actual_hash = super::sha256_hex(&bytes);
            if let Some(expected_hash) = params.get("final_sha256").and_then(|v| v.as_str()) {
                if !super::is_sha256_hex(expected_hash) {
                    return format!(
                        "Error: invalid final_sha256 '{}'. Provide a 64-character SHA-256 hex digest or omit final_sha256.",
                        expected_hash.trim()
                    );
                }
                if !expected_hash.trim().eq_ignore_ascii_case(&actual_hash) {
                    return format!(
                        "Error: final_sha256 mismatch for {path}. expected={}, actual={}. Staged file was not published.",
                        expected_hash.trim(),
                        actual_hash
                    );
                }
            }
            if let Err(e) = tokio::fs::rename(&stage_path, &target_path).await {
                return if e.kind() == std::io::ErrorKind::PermissionDenied {
                    format!("Error: Permission denied publishing {}. Hint: check file permissions or try a different path.", path)
                } else {
                    format!("Error publishing staged file: {}", e)
                };
            }
            format!("Finished writing {total} bytes to {path}; sha256={actual_hash}")
        } else {
            format!(
                "Staged {wrote} byte chunk for {path} with mode={mode}; total bytes now {total}; next expected_offset={total}"
            )
        }
    }
}

fn expand_write_path(path: &str) -> PathBuf {
    let expanded = expand_path(path);
    normalize_portable_home_alias(&expanded).unwrap_or(expanded)
}

fn normalize_portable_home_alias(path: &Path) -> Option<PathBuf> {
    let mut components = path.components();
    if components.next() != Some(Component::RootDir) {
        return None;
    }
    if components.next()?.as_os_str() != "home" {
        return None;
    }

    let alias = components.next()?.as_os_str().to_string_lossy();
    let home = dirs::home_dir()?;
    let current_user = home.file_name()?.to_string_lossy();
    if alias != "user" && alias != current_user {
        return None;
    }

    let mut remapped = home;
    for component in components {
        remapped.push(component.as_os_str());
    }
    Some(remapped)
}

fn staged_chunk_path(target: &Path) -> PathBuf {
    let file_name = target
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("nanobot-write");
    let staged_name = format!(".{file_name}.nanobot-part");
    target.with_file_name(staged_name)
}

#[cfg(test)]
mod tests {
    use super::super::make_params;
    use super::*;
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

    #[test]
    fn test_write_file_maps_synthetic_linux_home_alias() {
        let Some(home) = dirs::home_dir() else {
            return;
        };
        let expected = home.join("tetris").join("index.html");

        let actual = expand_write_path("/home/user/tetris/index.html");

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_write_file_maps_current_user_linux_home_alias() {
        let Some(home) = dirs::home_dir() else {
            return;
        };
        let Some(user) = home.file_name().and_then(|name| name.to_str()) else {
            return;
        };
        let input = format!("/home/{user}/tetris/index.html");
        let expected = home.join("tetris").join("index.html");

        let actual = expand_write_path(&input);

        assert_eq!(actual, expected);
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

    #[test]
    fn test_write_file_chunk_description_routes_large_artifacts() {
        let tool = WriteFileChunkTool;

        assert!(tool.description().contains("rich files"));
        assert!(tool.description().contains("start"));
        assert!(tool.description().contains("expected_offset"));
    }

    #[tokio::test]
    async fn test_write_file_chunk_stages_then_publishes_in_order() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("artifact.html");
        std::fs::write(&file_path, "old").unwrap();
        let tool = WriteFileChunkTool;

        let start = tool
            .execute(make_params(&[
                ("path", file_path.to_str().unwrap()),
                ("mode", "start"),
                ("content", "<html>"),
            ]))
            .await;
        assert!(start.starts_with("Staged 6 byte chunk"));
        assert!(start.contains("next expected_offset=6"));
        assert_eq!(std::fs::read_to_string(&file_path).unwrap(), "old");

        let mut append_params = make_params(&[
            ("path", file_path.to_str().unwrap()),
            ("mode", "append"),
            ("content", "<body>ok</body>"),
        ]);
        append_params.insert("expected_offset".to_string(), serde_json::json!(6));
        let append = tool.execute(append_params).await;
        assert!(append.contains("mode=append"));
        assert!(append.contains("next expected_offset=21"));
        assert_eq!(std::fs::read_to_string(&file_path).unwrap(), "old");

        let body = "<html><body>ok</body></html>";
        let expected_hash = crate::agent::tools::filesystem::sha256_hex(body.as_bytes());
        let mut finish_params = make_params(&[
            ("path", file_path.to_str().unwrap()),
            ("mode", "finish"),
            ("content", "</html>"),
        ]);
        finish_params.insert("expected_offset".to_string(), serde_json::json!(21));
        finish_params.insert("final_sha256".to_string(), serde_json::json!(expected_hash));
        let finish = tool.execute(finish_params).await;
        assert!(finish.starts_with("Finished writing"));
        assert!(finish.contains("sha256="));
        assert_eq!(std::fs::read_to_string(&file_path).unwrap(), body);
        assert!(!staged_chunk_path(&file_path).exists());
    }

    #[tokio::test]
    async fn test_write_file_chunk_requires_start_before_append() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("artifact.html");
        let tool = WriteFileChunkTool;

        let result = tool
            .execute(make_params(&[
                ("path", file_path.to_str().unwrap()),
                ("mode", "append"),
                ("content", "late"),
            ]))
            .await;

        assert!(result.contains("mode=start first"));
    }

    #[tokio::test]
    async fn test_write_file_chunk_rejects_missing_expected_offset_after_start() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("artifact.html");
        let tool = WriteFileChunkTool;

        let start = tool
            .execute(make_params(&[
                ("path", file_path.to_str().unwrap()),
                ("mode", "start"),
                ("content", "first"),
            ]))
            .await;
        assert!(start.contains("next expected_offset=5"));

        let result = tool
            .execute(make_params(&[
                ("path", file_path.to_str().unwrap()),
                ("mode", "append"),
                ("content", "second"),
            ]))
            .await;

        assert!(result.contains("expected_offset is required"));
        assert_eq!(
            std::fs::read_to_string(staged_chunk_path(&file_path)).unwrap(),
            "first"
        );
    }

    #[tokio::test]
    async fn test_write_file_chunk_rejects_offset_mismatch() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("artifact.html");
        let tool = WriteFileChunkTool;

        let start = tool
            .execute(make_params(&[
                ("path", file_path.to_str().unwrap()),
                ("mode", "start"),
                ("content", "first"),
            ]))
            .await;
        assert!(start.contains("next expected_offset=5"));

        let mut append_params = make_params(&[
            ("path", file_path.to_str().unwrap()),
            ("mode", "append"),
            ("content", "second"),
        ]);
        append_params.insert("expected_offset".to_string(), serde_json::json!(4));
        let result = tool.execute(append_params).await;

        assert!(result.contains("offset mismatch"));
        assert_eq!(
            std::fs::read_to_string(staged_chunk_path(&file_path)).unwrap(),
            "first"
        );
    }

    #[tokio::test]
    async fn test_write_file_chunk_rejects_final_hash_mismatch_without_publish() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("artifact.html");
        let tool = WriteFileChunkTool;

        let start = tool
            .execute(make_params(&[
                ("path", file_path.to_str().unwrap()),
                ("mode", "start"),
                ("content", "first"),
            ]))
            .await;
        assert!(start.contains("next expected_offset=5"));

        let mut finish_params = make_params(&[
            ("path", file_path.to_str().unwrap()),
            ("mode", "finish"),
            ("content", "second"),
        ]);
        finish_params.insert("expected_offset".to_string(), serde_json::json!(5));
        finish_params.insert(
            "final_sha256".to_string(),
            serde_json::json!("0".repeat(64)),
        );
        let result = tool.execute(finish_params).await;

        assert!(result.contains("final_sha256 mismatch"));
        assert!(!file_path.exists());
        assert_eq!(
            std::fs::read_to_string(staged_chunk_path(&file_path)).unwrap(),
            "firstsecond"
        );
    }
}
