//! Batch wrapper for independent read-only tool operations.

use std::collections::HashMap;
use std::path::PathBuf;

use async_trait::async_trait;
use serde_json::{json, Value};

use super::base::Tool;
use super::{
    FileInfoTool, FilePreviewTool, FindFilesTool, ListDirTool, ReadFileTool, SearchContextTool,
    SearchFilesTool,
};

const DEFAULT_CONCURRENCY: usize = 8;
const MAX_CONCURRENCY: usize = 16;

/// Tool to execute independent read-only operations in one model call.
pub struct BatchTool {
    workspace: PathBuf,
    db_path: Option<PathBuf>,
}

impl BatchTool {
    pub fn new(workspace: PathBuf, db_path: Option<PathBuf>) -> Self {
        Self { workspace, db_path }
    }
}

#[async_trait]
impl Tool for BatchTool {
    fn name(&self) -> &str {
        "batch"
    }

    fn description(&self) -> &str {
        "Run multiple independent read-only tool operations concurrently. Preserves input order and rejects unsafe tools."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "operations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "description": "Optional caller-supplied id"},
                            "tool": {
                                "type": "string",
                                "enum": ["read_file", "list_dir", "file_info", "find_files", "search_files", "file_preview", "search_context"]
                            },
                            "args": {"type": "object", "description": "Arguments for the selected tool"}
                        },
                        "required": ["tool", "args"]
                    }
                },
                "max_concurrency": {
                    "type": "integer",
                    "description": "Max operations to run at once. Default: 8, max: 16."
                }
            },
            "required": ["operations"]
        })
    }

    async fn execute(&self, params: HashMap<String, Value>) -> String {
        let operations = match params.get("operations").and_then(|v| v.as_array()) {
            Some(ops) if !ops.is_empty() => ops.clone(),
            _ => return "Error: 'operations' must be a non-empty array".to_string(),
        };
        let max_concurrency = params
            .get("max_concurrency")
            .and_then(|v| v.as_u64())
            .map(|v| (v as usize).clamp(1, MAX_CONCURRENCY))
            .unwrap_or(DEFAULT_CONCURRENCY);

        let mut parsed = Vec::with_capacity(operations.len());
        for (index, op) in operations.iter().enumerate() {
            let id = op
                .get("id")
                .and_then(|v| v.as_str())
                .map(ToString::to_string);
            let tool = match op.get("tool").and_then(|v| v.as_str()) {
                Some(t) => t.to_string(),
                None => {
                    parsed.push(BatchOperation::invalid(
                        index,
                        id,
                        "missing string 'tool'".to_string(),
                    ));
                    continue;
                }
            };
            let args = match op.get("args").and_then(|v| v.as_object()) {
                Some(map) => map.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
                None => {
                    parsed.push(BatchOperation::invalid(
                        index,
                        id,
                        "missing object 'args'".to_string(),
                    ));
                    continue;
                }
            };
            if !is_allowed_batch_tool(&tool) {
                parsed.push(BatchOperation::invalid(
                    index,
                    id,
                    format!("tool '{}' is not allowed in batch", tool),
                ));
                continue;
            }
            parsed.push(BatchOperation {
                index,
                id,
                tool,
                args,
                pre_error: None,
            });
        }

        let mut results: Vec<Option<Value>> = vec![None; parsed.len()];
        for chunk in parsed.chunks(max_concurrency) {
            let futs = chunk.iter().cloned().map(|op| {
                let workspace = self.workspace.clone();
                let db_path = self.db_path.clone();
                async move { execute_batch_operation(op, workspace, db_path).await }
            });
            for (idx, result) in futures_util::future::join_all(futs).await {
                results[idx] = Some(result);
            }
        }

        let rendered: Vec<Value> = results
            .into_iter()
            .enumerate()
            .map(|(index, value)| {
                value.unwrap_or_else(|| {
                    json!({
                        "index": index,
                        "ok": false,
                        "error": "operation did not run"
                    })
                })
            })
            .collect();
        serde_json::to_string_pretty(&json!({ "results": rendered }))
            .unwrap_or_else(|_| "Error: failed to serialize batch results".to_string())
    }
}

#[derive(Clone)]
struct BatchOperation {
    index: usize,
    id: Option<String>,
    tool: String,
    args: HashMap<String, Value>,
    pre_error: Option<String>,
}

impl BatchOperation {
    fn invalid(index: usize, id: Option<String>, error: String) -> Self {
        Self {
            index,
            id,
            tool: String::new(),
            args: HashMap::new(),
            pre_error: Some(error),
        }
    }
}

async fn execute_batch_operation(
    op: BatchOperation,
    workspace: PathBuf,
    db_path: Option<PathBuf>,
) -> (usize, Value) {
    if let Some(error) = op.pre_error {
        return (
            op.index,
            batch_result(op.index, op.id, "", false, None, Some(error)),
        );
    }
    let output = match op.tool.as_str() {
        "read_file" => ReadFileTool.execute(op.args.clone()).await,
        "list_dir" => ListDirTool.execute(op.args.clone()).await,
        "file_info" => FileInfoTool.execute(op.args.clone()).await,
        "find_files" => FindFilesTool.execute(op.args.clone()).await,
        "search_files" => SearchFilesTool.execute(op.args.clone()).await,
        "file_preview" => FilePreviewTool.execute(op.args.clone()).await,
        "search_context" => {
            SearchContextTool::new(workspace, db_path)
                .execute(op.args.clone())
                .await
        }
        _ => format!("Error: tool '{}' is not allowed in batch", op.tool),
    };
    let ok = !output.starts_with("Error:");
    let error = (!ok).then(|| output.trim_start_matches("Error:").trim().to_string());
    (
        op.index,
        batch_result(op.index, op.id, &op.tool, ok, Some(output), error),
    )
}

fn batch_result(
    index: usize,
    id: Option<String>,
    tool: &str,
    ok: bool,
    output: Option<String>,
    error: Option<String>,
) -> Value {
    json!({
        "index": index,
        "id": id,
        "tool": tool,
        "ok": ok,
        "output": output,
        "error": error,
    })
}

fn is_allowed_batch_tool(tool: &str) -> bool {
    matches!(
        tool,
        "read_file"
            | "list_dir"
            | "file_info"
            | "find_files"
            | "search_files"
            | "file_preview"
            | "search_context"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn test_batch_preserves_order_and_rejects_write_tool() {
        let tmp = tempfile::tempdir().unwrap();
        let a = tmp.path().join("a.txt");
        let b = tmp.path().join("b.txt");
        tokio::fs::write(&a, "alpha\n").await.unwrap();
        tokio::fs::write(&b, "beta\n").await.unwrap();
        let tool = BatchTool::new(tmp.path().to_path_buf(), None);
        let mut params = HashMap::new();
        params.insert(
            "operations".to_string(),
            json!([
                {"id": "one", "tool": "read_file", "args": {"path": a.display().to_string()}},
                {"id": "bad", "tool": "write_file", "args": {"path": b.display().to_string(), "content": "x"}},
                {"id": "two", "tool": "read_file", "args": {"path": b.display().to_string()}},
            ]),
        );
        let out = tool.execute(params).await;
        let parsed: Value = serde_json::from_str(&out).unwrap();
        let results = parsed["results"].as_array().unwrap();
        assert_eq!(results[0]["id"], "one");
        assert_eq!(results[1]["id"], "bad");
        assert_eq!(results[1]["ok"], false);
        assert_eq!(results[2]["id"], "two");
        assert!(results[0]["output"].as_str().unwrap().contains("alpha"));
        assert!(results[2]["output"].as_str().unwrap().contains("beta"));
    }
}
