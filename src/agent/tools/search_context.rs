//! Unified search tool across workspace files, memory, and session history.

use std::collections::HashMap;
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;

use async_trait::async_trait;
use serde_json::{json, Value};

use super::base::Tool;
use super::{RecallTool, SearchFilesTool, SessionSearchTool};

/// Tool that searches local files, memory, and sessions through one interface.
pub struct SearchContextTool {
    workspace: PathBuf,
    db_path: Option<PathBuf>,
}

impl SearchContextTool {
    pub fn new(workspace: PathBuf, db_path: Option<PathBuf>) -> Self {
        Self { workspace, db_path }
    }
}

#[async_trait]
impl Tool for SearchContextTool {
    fn name(&self) -> &str {
        "search_context"
    }

    fn description(&self) -> &str {
        "Search across workspace files, long-term memory, and past sessions. Results are labeled by source."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "sources": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["files", "memory", "sessions"]},
                    "description": "Sources to search. Default: all available sources."
                },
                "path": {
                    "type": "string",
                    "description": "Workspace path for file search. Default: current directory."
                },
                "pattern": {
                    "type": "string",
                    "description": "File pattern for file search. Default: *."
                },
                "limit": {
                    "type": "integer",
                    "description": "Per-source result limit. Default: 10, max: 100."
                },
                "channel": {
                    "type": "string",
                    "description": "Optional session key/channel prefix for session search."
                },
                "mode": {
                    "type": "string",
                    "enum": ["auto", "keyword", "semantic"],
                    "description": "Memory search mode. Default: auto."
                }
            },
            "required": ["query"]
        })
    }

    async fn execute(&self, params: HashMap<String, Value>) -> String {
        let query = match params.get("query").and_then(|v| v.as_str()) {
            Some(q) if !q.trim().is_empty() => q.trim().to_string(),
            _ => return "Error: 'query' parameter is required and must be non-empty".to_string(),
        };
        let limit = params
            .get("limit")
            .and_then(|v| v.as_u64())
            .map(|v| (v as usize).clamp(1, 100))
            .unwrap_or(10);
        let sources = parse_sources(params.get("sources"));

        let mut futs: Vec<Pin<Box<dyn Future<Output = (String, String)> + Send>>> = Vec::new();

        if sources.contains(&"files") {
            let mut file_params = HashMap::new();
            file_params.insert("query".to_string(), json!(query.clone()));
            file_params.insert(
                "path".to_string(),
                params
                    .get("path")
                    .cloned()
                    .unwrap_or_else(|| json!(self.workspace.display().to_string())),
            );
            if let Some(pattern) = params.get("pattern").cloned() {
                file_params.insert("pattern".to_string(), pattern);
            }
            file_params.insert("limit".to_string(), json!(limit));
            futs.push(Box::pin(async move {
                let out = SearchFilesTool.execute(file_params).await;
                ("files".to_string(), out)
            }));
        }

        if sources.contains(&"memory") {
            let workspace = self.workspace.clone();
            let mode = params
                .get("mode")
                .and_then(|v| v.as_str())
                .unwrap_or("auto")
                .to_string();
            let q = query.clone();
            futs.push(Box::pin(async move {
                let tool = RecallTool::new(&workspace);
                let mut memory_params = HashMap::new();
                memory_params.insert("query".to_string(), json!(q));
                memory_params.insert("mode".to_string(), json!(mode));
                let out = tool.execute(memory_params).await;
                ("memory".to_string(), out)
            }));
        }

        if sources.contains(&"sessions") {
            if let Some(db_path) = self.db_path.clone() {
                let q = query.clone();
                let channel = params.get("channel").cloned();
                futs.push(Box::pin(async move {
                    let tool = SessionSearchTool::new(db_path);
                    let mut session_params = HashMap::new();
                    session_params.insert("query".to_string(), json!(q));
                    session_params.insert("limit".to_string(), json!(limit));
                    if let Some(channel) = channel {
                        session_params.insert("channel".to_string(), channel);
                    }
                    let out = tool.execute(session_params).await;
                    ("sessions".to_string(), out)
                }));
            }
        }

        if futs.is_empty() {
            return "Error: no searchable sources are available".to_string();
        }

        let sections = futures_util::future::join_all(futs).await;
        let mut out = format!("Search context results for {:?}\n", query);
        for (source, body) in sections {
            out.push_str(&format!("\n## {}\n{}\n", source, body.trim_end()));
        }
        out
    }
}

fn parse_sources(value: Option<&Value>) -> Vec<&'static str> {
    let Some(Value::Array(items)) = value else {
        return vec!["files", "memory", "sessions"];
    };
    let mut out = Vec::new();
    for item in items {
        match item.as_str() {
            Some("files") if !out.contains(&"files") => out.push("files"),
            Some("memory") if !out.contains(&"memory") => out.push("memory"),
            Some("sessions") if !out.contains(&"sessions") => out.push("sessions"),
            _ => {}
        }
    }
    if out.is_empty() {
        vec!["files", "memory", "sessions"]
    } else {
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn test_search_context_labels_file_results() {
        let tmp = tempfile::tempdir().unwrap();
        let file = tmp.path().join("notes.txt");
        tokio::fs::write(&file, "alpha beta gamma\n").await.unwrap();
        let tool = SearchContextTool::new(tmp.path().to_path_buf(), None);
        let mut params = HashMap::new();
        params.insert("query".to_string(), json!("beta"));
        params.insert("sources".to_string(), json!(["files"]));
        params.insert("path".to_string(), json!(tmp.path().display().to_string()));
        let out = tool.execute(params).await;
        assert!(out.contains("## files"), "{out}");
        assert!(out.contains("notes.txt"), "{out}");
    }
}
