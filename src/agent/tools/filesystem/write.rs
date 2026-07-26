//! `write_file` tool.

use std::collections::{HashMap, HashSet};
use std::path::{Component, Path, PathBuf};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use tokio::io::AsyncWriteExt;
use tokio::sync::Mutex;

use super::super::base::{PermissionLevel, Tool, ToolExecutionContext};
use super::{expand_path, require_param};

pub(crate) const MAX_WRITE_FILE_PIECE_CHARS: usize = 4096;
const STAGED_WRITE_TTL: Duration = Duration::from_secs(30 * 60);

struct StagedWrite {
    stage_path: PathBuf,
    total_bytes: u64,
    delivered_call_ids: HashSet<String>,
    updated_at: Instant,
}

struct CompletedWrite {
    target_path: PathBuf,
    content_digest: String,
    total_bytes: u64,
    updated_at: Instant,
}

struct RemoveOnDrop {
    path: PathBuf,
    armed: bool,
}

impl RemoveOnDrop {
    fn new(path: PathBuf) -> Self {
        Self { path, armed: true }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for RemoveOnDrop {
    fn drop(&mut self) {
        if self.armed {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

struct TruncateOnDrop {
    path: PathBuf,
    len: u64,
    armed: bool,
}

impl TruncateOnDrop {
    fn new(path: PathBuf, len: u64) -> Self {
        Self {
            path,
            len,
            armed: true,
        }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for TruncateOnDrop {
    fn drop(&mut self) {
        if self.armed {
            if let Ok(file) = std::fs::OpenOptions::new().write(true).open(&self.path) {
                let _ = file.set_len(self.len);
            }
        }
    }
}

#[derive(Default)]
struct WriteState {
    staged_writes: HashMap<PathBuf, StagedWrite>,
    completed_calls: HashMap<String, CompletedWrite>,
}

/// Tool to write content to a file.
pub struct WriteFileTool {
    state: Mutex<WriteState>,
}

impl Default for WriteFileTool {
    fn default() -> Self {
        Self {
            state: Mutex::new(WriteState::default()),
        }
    }
}

impl Drop for WriteFileTool {
    fn drop(&mut self) {
        // A registry lives for one inbound message. Incomplete transactions
        // must disappear with it instead of leaving hidden part files behind.
        for staged in self.state.get_mut().staged_writes.values() {
            let _ = std::fs::remove_file(&staged.stage_path);
        }
    }
}

impl WriteFileTool {
    async fn execute_write(
        &self,
        params: HashMap<String, serde_json::Value>,
        tool_call_id: Option<&str>,
    ) -> String {
        let path = match require_param(&params, "path") {
            Ok(p) => p,
            Err(e) => return e,
        };
        let content = match params.get("content").and_then(|v| v.as_str()) {
            Some(c) => c,
            None => return "Error: 'content' parameter is required".to_string(),
        };
        let state = params
            .get("state")
            .and_then(|v| v.as_str())
            .unwrap_or("complete")
            .trim()
            .to_ascii_lowercase();
        if !matches!(state.as_str(), "more" | "complete" | "append") {
            return "Error: 'state' must be one of: more, complete, append".to_string();
        }
        let piece_chars = content.chars().count();
        if tool_call_id.is_some() && piece_chars > MAX_WRITE_FILE_PIECE_CHARS {
            return oversized_piece_error(piece_chars);
        }

        let target_path = expand_write_path(path);
        if let Some(parent) = target_path.parent() {
            if let Err(e) = tokio::fs::create_dir_all(parent).await {
                return format!(
                    "Error creating directories: {e}. Hint: check file permissions or try a different path."
                );
            }
        }

        let content_digest = super::sha256_hex(content.as_bytes());
        let mut writer_state = self.state.lock().await;
        let now = Instant::now();
        let expired_targets: Vec<PathBuf> = writer_state
            .staged_writes
            .iter()
            .filter(|(_, staged)| now.duration_since(staged.updated_at) >= STAGED_WRITE_TTL)
            .map(|(target, _)| target.clone())
            .collect();
        for expired_target in expired_targets {
            if let Some(expired) = writer_state.staged_writes.remove(&expired_target) {
                let _ = tokio::fs::remove_file(expired.stage_path).await;
            }
        }
        writer_state
            .completed_calls
            .retain(|_, completed| now.duration_since(completed.updated_at) < STAGED_WRITE_TTL);

        if let Some(completed) = tool_call_id
            .and_then(|id| writer_state.completed_calls.get(id))
            .filter(|completed| {
                completed.target_path == target_path && completed.content_digest == content_digest
            })
        {
            return format!(
                "This write_file call already completed for {path}; total={} bytes.",
                completed.total_bytes
            );
        }

        if let Some(staged) = writer_state.staged_writes.get_mut(&target_path) {
            if state == "append" {
                return "Error: cannot append while a staged write is active for this path; finish it with state=\"complete\" first."
                    .to_string();
            }
            if tool_call_id.is_some_and(|id| staged.delivered_call_ids.contains(id)) {
                return format!(
                    "This write_file call was already staged for {path}; total={} bytes. Continue with state=\"more\", or send the final piece with state=\"complete\".",
                    staged.total_bytes
                );
            }
            if piece_chars > MAX_WRITE_FILE_PIECE_CHARS {
                return oversized_piece_error(piece_chars);
            }

            let previous_total = staged.total_bytes;
            let mut rollback = TruncateOnDrop::new(staged.stage_path.clone(), previous_total);
            let append_result = match tokio::fs::OpenOptions::new()
                .append(true)
                .open(&staged.stage_path)
                .await
            {
                Ok(mut file) => file.write_all(content.as_bytes()).await,
                Err(e) => Err(e),
            };
            if let Err(e) = append_result {
                return write_error(path, e);
            }

            staged.total_bytes += content.len() as u64;
            staged.updated_at = now;
            if let Some(id) = tool_call_id {
                staged.delivered_call_ids.insert(id.to_string());
            }

            if state == "more" {
                rollback.disarm();
                return staged_receipt(path, content.len(), staged.total_bytes);
            }

            let stage_path = staged.stage_path.clone();
            let total_bytes = staged.total_bytes;
            if let Err(e) = tokio::fs::rename(&stage_path, &target_path).await {
                staged.total_bytes = previous_total;
                if let Some(id) = tool_call_id {
                    staged.delivered_call_ids.remove(id);
                }
                return publish_error(path, e);
            }
            rollback.disarm();
            writer_state.staged_writes.remove(&target_path);
            if let Some(id) = tool_call_id {
                writer_state.completed_calls.insert(
                    id.to_string(),
                    CompletedWrite {
                        target_path,
                        content_digest,
                        total_bytes,
                        updated_at: now,
                    },
                );
            }
            return format!("Successfully wrote {total_bytes} bytes to {path}");
        }

        if state == "more" {
            if piece_chars > MAX_WRITE_FILE_PIECE_CHARS {
                return oversized_piece_error(piece_chars);
            }
            let stage_path = unique_staged_write_path(&target_path);
            let mut cleanup = RemoveOnDrop::new(stage_path.clone());
            if let Err(e) = tokio::fs::write(&cleanup.path, content).await {
                return write_error(path, e);
            }
            let mut delivered_call_ids = HashSet::new();
            if let Some(id) = tool_call_id {
                delivered_call_ids.insert(id.to_string());
            }
            let total_bytes = content.len() as u64;
            writer_state.staged_writes.insert(
                target_path,
                StagedWrite {
                    stage_path,
                    total_bytes,
                    delivered_call_ids,
                    updated_at: now,
                },
            );
            cleanup.disarm();
            return staged_receipt(path, content.len(), total_bytes);
        }

        if state == "append" {
            return match atomic_append(&target_path, content).await {
                Ok(total_bytes) => {
                    if let Some(id) = tool_call_id {
                        writer_state.completed_calls.insert(
                            id.to_string(),
                            CompletedWrite {
                                target_path,
                                content_digest,
                                total_bytes,
                                updated_at: now,
                            },
                        );
                    }
                    format!(
                        "Successfully appended {} bytes to {}; total={} bytes",
                        content.len(),
                        path,
                        total_bytes
                    )
                }
                Err(e) => write_error(path, e),
            };
        }

        match atomic_replace(&target_path, content).await {
            Ok(()) => {
                if let Some(id) = tool_call_id {
                    writer_state.completed_calls.insert(
                        id.to_string(),
                        CompletedWrite {
                            target_path,
                            content_digest,
                            total_bytes: content.len() as u64,
                            updated_at: now,
                        },
                    );
                }
                format!("Successfully wrote {} bytes to {}", content.len(), path)
            }
            Err(e) => write_error(path, e),
        }
    }
}

#[async_trait]
impl Tool for WriteFileTool {
    fn name(&self) -> &str {
        "write_file"
    }

    fn description(&self) -> &str {
        "Write a file atomically; content MUST be 4096 characters or less per call. For larger replacements, call write_file repeatedly on the same path with state=more for non-final pieces and state=complete for the final piece; use state=append to add a small suffix, omit state for a smaller complete file, and never send offsets, hashes, IDs, or temporary paths."
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
                    "maxLength": MAX_WRITE_FILE_PIECE_CHARS,
                    "description": "One content piece, at most 4096 characters"
                },
                "state": {
                    "type": "string",
                    "enum": ["more", "complete", "append"],
                    "default": "complete",
                    "description": "Omit for a complete one-call write. Use more for non-final pieces, complete for the final piece, or append to add content to an existing file."
                }
            },
            "required": ["path", "content"]
        })
    }

    async fn execute(&self, params: HashMap<String, serde_json::Value>) -> String {
        self.execute_write(params, None).await
    }

    async fn execute_with_context(
        &self,
        params: HashMap<String, serde_json::Value>,
        ctx: &ToolExecutionContext,
    ) -> String {
        self.execute_write(params, Some(&ctx.tool_call_id)).await
    }
}

async fn atomic_replace(target_path: &Path, content: &str) -> std::io::Result<()> {
    let stage_path = unique_staged_write_path(target_path);
    let cleanup = RemoveOnDrop::new(stage_path);
    if let Err(e) = tokio::fs::write(&cleanup.path, content).await {
        return Err(e);
    }
    if let Err(e) = tokio::fs::rename(&cleanup.path, target_path).await {
        return Err(e);
    }
    Ok(())
}

async fn atomic_append(target_path: &Path, content: &str) -> std::io::Result<u64> {
    let mut combined = match tokio::fs::read_to_string(target_path).await {
        Ok(existing) => existing,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(e) => return Err(e),
    };
    combined.push_str(content);
    let total_bytes = combined.len() as u64;
    atomic_replace(target_path, &combined).await?;
    Ok(total_bytes)
}

fn staged_receipt(path: &str, wrote: usize, total: u64) -> String {
    format!(
        "Staged {wrote} bytes for {path}; total={total}. Continue with write_file(path, content, state=\"more\"), or send the final piece with state=\"complete\"."
    )
}

fn oversized_piece_error(actual_chars: usize) -> String {
    format!(
        "Error: write_file staged content is {actual_chars} characters; send pieces of {MAX_WRITE_FILE_PIECE_CHARS} characters or less."
    )
}

fn write_error(path: &str, error: std::io::Error) -> String {
    if error.kind() == std::io::ErrorKind::PermissionDenied {
        format!(
            "Error: Permission denied: {path}. Hint: check file permissions or try a different path."
        )
    } else {
        format!("Error writing file: {error}")
    }
}

fn publish_error(path: &str, error: std::io::Error) -> String {
    if error.kind() == std::io::ErrorKind::PermissionDenied {
        format!(
            "Error: Permission denied publishing {path}. Hint: check file permissions or try a different path."
        )
    } else {
        format!("Error publishing staged file: {error}")
    }
}

fn unique_staged_write_path(target: &Path) -> PathBuf {
    let file_name = target
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("nanobot-write");
    let staged_name = format!(".{file_name}.nanobot-part-{}", uuid::Uuid::new_v4());
    target.with_file_name(staged_name)
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

#[cfg(test)]
mod tests {
    use super::super::make_params;
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_write_file_creates_file() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("output.txt");

        let tool = WriteFileTool::default();
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

        let tool = WriteFileTool::default();
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
        let tool = WriteFileTool::default();
        let params = make_params(&[("content", "test")]);
        let result = tool.execute(params).await;
        assert!(result.contains("'path' parameter is required"));
    }

    #[tokio::test]
    async fn test_write_file_missing_content() {
        let tool = WriteFileTool::default();
        let params = make_params(&[("path", "/tmp/test.txt")]);
        let result = tool.execute(params).await;
        assert!(result.contains("'content' parameter is required"));
    }

    #[test]
    fn test_write_file_name() {
        let tool = WriteFileTool::default();
        assert_eq!(tool.name(), "write_file");
    }

    #[test]
    fn test_write_file_schema_teaches_simple_transactional_writes() {
        let tool = WriteFileTool::default();
        let schema = tool.parameters();
        let description = tool.description();

        assert!(description.contains("MUST be 4096 characters or less"));
        assert!(description.contains("For larger replacements"));
        assert!(description.contains("state=more"));
        assert!(description.contains("state=complete"));
        assert!(description.contains("never send offsets"));
        assert!(description.contains("hashes"));
        assert_eq!(
            schema.pointer("/properties/state/enum"),
            Some(&serde_json::json!(["more", "complete", "append"]))
        );
        assert!(schema.pointer("/properties/expected_offset").is_none());
        assert!(schema.pointer("/properties/final_sha256").is_none());
        assert_eq!(
            schema.pointer("/required"),
            Some(&serde_json::json!(["path", "content"]))
        );
    }

    #[tokio::test]
    async fn test_write_file_more_stages_until_complete() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("artifact.html");
        std::fs::write(&file_path, "old").unwrap();
        let tool = WriteFileTool::default();

        let first = tool
            .execute(make_params(&[
                ("path", file_path.to_str().unwrap()),
                ("content", "<html>"),
                ("state", "more"),
            ]))
            .await;
        assert!(first.contains("Staged 6 bytes"));
        assert!(first.contains("state=\"more\""));
        assert!(first.contains("state=\"complete\""));
        assert_eq!(std::fs::read_to_string(&file_path).unwrap(), "old");

        let second = tool
            .execute(make_params(&[
                ("path", file_path.to_str().unwrap()),
                ("content", "<body>ok</body>"),
                ("state", "more"),
            ]))
            .await;
        assert!(second.contains("total=21"));
        assert_eq!(std::fs::read_to_string(&file_path).unwrap(), "old");

        let final_result = tool
            .execute(make_params(&[
                ("path", file_path.to_str().unwrap()),
                ("content", "</html>"),
                ("state", "complete"),
            ]))
            .await;
        assert!(final_result.starts_with("Successfully wrote 28 bytes"));
        assert_eq!(
            std::fs::read_to_string(&file_path).unwrap(),
            "<html><body>ok</body></html>"
        );
    }

    #[tokio::test]
    async fn test_write_file_rejects_oversized_staged_piece_without_touching_target() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("artifact.html");
        std::fs::write(&file_path, "old").unwrap();
        let tool = WriteFileTool::default();
        let mut params = HashMap::new();
        params.insert("path".to_string(), serde_json::json!(file_path));
        params.insert("content".to_string(), serde_json::json!("x".repeat(4097)));
        params.insert("state".to_string(), serde_json::json!("more"));

        let result = tool.execute(params).await;

        assert!(result.contains("4097 characters"));
        assert!(result.contains("4096 characters or less"));
        assert_eq!(std::fs::read_to_string(&file_path).unwrap(), "old");
    }

    #[tokio::test]
    async fn test_model_write_rejects_oversized_complete_call_without_touching_target() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("artifact.html");
        std::fs::write(&file_path, "old").unwrap();
        let tool = WriteFileTool::default();
        let (event_tx, _event_rx) = tokio::sync::mpsc::unbounded_channel();
        let ctx = crate::agent::tools::base::ToolExecutionContext {
            event_tx,
            cancellation_token: tokio_util::sync::CancellationToken::new(),
            tool_call_id: "call-oversized".to_string(),
        };

        let result = tool
            .execute_with_context(
                make_params(&[
                    ("path", file_path.to_str().unwrap()),
                    ("content", &"x".repeat(MAX_WRITE_FILE_PIECE_CHARS + 1)),
                ]),
                &ctx,
            )
            .await;

        assert!(result.starts_with("Error:"), "{result}");
        assert!(result.contains("4097 characters"), "{result}");
        assert_eq!(std::fs::read_to_string(file_path).unwrap(), "old");
    }

    #[tokio::test]
    async fn test_write_file_accepts_schema_valid_unicode_piece() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("unicode.txt");
        let tool = WriteFileTool::default();
        let content = "é".repeat(MAX_WRITE_FILE_PIECE_CHARS);

        let result = tool
            .execute(make_params(&[
                ("path", path.to_str().unwrap()),
                ("content", &content),
                ("state", "more"),
            ]))
            .await;

        assert!(!result.starts_with("Error:"), "{result}");
        assert!(!path.exists(), "staged writes must not publish early");
    }

    #[tokio::test]
    async fn test_incomplete_write_removes_staging_file_when_tool_is_dropped() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("unfinished.txt");
        {
            let tool = WriteFileTool::default();
            let result = tool
                .execute(make_params(&[
                    ("path", path.to_str().unwrap()),
                    ("content", "partial"),
                    ("state", "more"),
                ]))
                .await;
            assert!(!result.starts_with("Error:"), "{result}");
            assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
        }

        assert!(!path.exists());
        assert_eq!(
            std::fs::read_dir(dir.path()).unwrap().count(),
            0,
            "dropping a per-turn writer must not orphan its staging file"
        );
    }

    #[tokio::test]
    async fn test_write_file_rejects_unknown_state() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("artifact.html");
        let tool = WriteFileTool::default();

        let result = tool
            .execute(make_params(&[
                ("path", file_path.to_str().unwrap()),
                ("content", "content"),
                ("state", "merge"),
            ]))
            .await;

        assert!(result.contains("'state' must be one of: more, complete, append"));
        assert!(!file_path.exists());
    }

    #[tokio::test]
    async fn test_write_file_ignores_redelivery_of_the_same_staged_tool_call() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("artifact.html");
        std::fs::write(&file_path, "old").unwrap();
        let tool = WriteFileTool::default();
        let (event_tx, _event_rx) = tokio::sync::mpsc::unbounded_channel();
        let ctx = crate::agent::tools::base::ToolExecutionContext {
            event_tx,
            cancellation_token: tokio_util::sync::CancellationToken::new(),
            tool_call_id: "call-1".to_string(),
        };

        let params = make_params(&[
            ("path", file_path.to_str().unwrap()),
            ("content", "<html>"),
            ("state", "more"),
        ]);
        let first = tool.execute_with_context(params.clone(), &ctx).await;
        let duplicate = tool.execute_with_context(params, &ctx).await;
        assert!(first.contains("total=6"));
        assert!(duplicate.contains("already staged"));

        let final_result = tool
            .execute(make_params(&[
                ("path", file_path.to_str().unwrap()),
                ("content", "</html>"),
                ("state", "complete"),
            ]))
            .await;
        assert!(final_result.starts_with("Successfully wrote 13 bytes"));
        assert_eq!(
            std::fs::read_to_string(&file_path).unwrap(),
            "<html></html>"
        );
    }

    #[tokio::test]
    async fn test_write_file_append_is_atomic_and_idempotent() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("artifact.html");
        std::fs::write(&file_path, "<script>run()").unwrap();
        let tool = WriteFileTool::default();
        let (event_tx, _event_rx) = tokio::sync::mpsc::unbounded_channel();
        let ctx = crate::agent::tools::base::ToolExecutionContext {
            event_tx,
            cancellation_token: tokio_util::sync::CancellationToken::new(),
            tool_call_id: "call-append".to_string(),
        };
        let params = make_params(&[
            ("path", file_path.to_str().unwrap()),
            ("content", "</script></html>"),
            ("state", "append"),
        ]);

        let first = tool.execute_with_context(params.clone(), &ctx).await;
        let duplicate = tool.execute_with_context(params, &ctx).await;

        assert!(first.starts_with("Successfully appended"), "{first}");
        assert!(duplicate.contains("already completed"), "{duplicate}");
        assert_eq!(
            std::fs::read_to_string(file_path).unwrap(),
            "<script>run()</script></html>"
        );
    }

    #[tokio::test]
    async fn test_write_file_ignores_redelivery_after_publish() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("artifact.html");
        let tool = WriteFileTool::default();
        let (event_tx, _event_rx) = tokio::sync::mpsc::unbounded_channel();
        let first_ctx = crate::agent::tools::base::ToolExecutionContext {
            event_tx: event_tx.clone(),
            cancellation_token: tokio_util::sync::CancellationToken::new(),
            tool_call_id: "call-1".to_string(),
        };
        let final_ctx = crate::agent::tools::base::ToolExecutionContext {
            event_tx,
            cancellation_token: tokio_util::sync::CancellationToken::new(),
            tool_call_id: "call-2".to_string(),
        };

        tool.execute_with_context(
            make_params(&[
                ("path", file_path.to_str().unwrap()),
                ("content", "<html>"),
                ("state", "more"),
            ]),
            &first_ctx,
        )
        .await;
        let final_params = make_params(&[
            ("path", file_path.to_str().unwrap()),
            ("content", "</html>"),
            ("state", "complete"),
        ]);
        let first_publish = tool
            .execute_with_context(final_params.clone(), &final_ctx)
            .await;
        let duplicate_publish = tool.execute_with_context(final_params, &final_ctx).await;

        assert!(first_publish.starts_with("Successfully wrote 13 bytes"));
        assert!(duplicate_publish.contains("already completed"));
        assert_eq!(
            std::fs::read_to_string(&file_path).unwrap(),
            "<html></html>"
        );
    }

    #[test]
    fn cancellation_guards_remove_new_stage_and_rollback_append() {
        let dir = TempDir::new().unwrap();
        let new_stage = dir.path().join("new.part");
        std::fs::write(&new_stage, "partial").unwrap();
        drop(RemoveOnDrop::new(new_stage.clone()));
        assert!(!new_stage.exists());

        let appended_stage = dir.path().join("append.part");
        std::fs::write(&appended_stage, "stable-partial").unwrap();
        drop(TruncateOnDrop::new(appended_stage.clone(), 6));
        assert_eq!(std::fs::read_to_string(appended_stage).unwrap(), "stable");
    }
}
