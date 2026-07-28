//! Unified search tool across workspace files and curated memory.

use std::collections::HashMap;
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;

use async_trait::async_trait;
use serde_json::{json, Value};

use super::base::Tool;
use super::{RecallTool, SearchFilesTool};

/// Tool that searches local files and curated memory through one interface.
pub struct SearchContextTool {
    workspace: PathBuf,
}

impl SearchContextTool {
    pub fn new(workspace: PathBuf) -> Self {
        Self { workspace }
    }
}

#[async_trait]
impl Tool for SearchContextTool {
    fn name(&self) -> &str {
        "search_context"
    }

    fn description(&self) -> &str {
        "Search workspace files and curated long-term memory. Results are labeled by source; use session_search for past conversations."
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
                    "items": {"type": "string", "enum": ["files", "memory"]},
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
        return vec!["files", "memory"];
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
    out
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
        let tool = SearchContextTool::new(tmp.path().to_path_buf());
        let mut params = HashMap::new();
        params.insert("query".to_string(), json!("beta"));
        params.insert("sources".to_string(), json!(["files"]));
        params.insert("path".to_string(), json!(tmp.path().display().to_string()));
        let out = tool.execute(params).await;
        assert!(out.contains("## files"), "{out}");
        assert!(out.contains("notes.txt"), "{out}");
    }

    #[tokio::test]
    async fn test_search_context_does_not_bypass_session_search_ownership() {
        let tmp = tempfile::tempdir().unwrap();
        let db_path = tmp.path().join("sessions.db");
        let db = crate::session::db::SessionDb::new(&db_path);
        let session = db.create_session("cli:past").await;
        db.add_messages(
            &session.id,
            &[json!({"role": "user", "content": "solepathtranscript"})],
        )
        .await;
        let tool = SearchContextTool::new(tmp.path().to_path_buf());

        let out = tool
            .execute(HashMap::from([
                ("query".to_string(), json!("solepathtranscript")),
                ("sources".to_string(), json!(["sessions"])),
            ]))
            .await;

        assert!(
            out.contains("no searchable sources"),
            "session_search must be the sole user-facing transcript path: {out}"
        );
    }
}
