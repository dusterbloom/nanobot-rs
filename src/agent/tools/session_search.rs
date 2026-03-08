//! Session search tool — FTS5 full-text search across past conversations.
//!
//! Unlike `recall` (which searches curated long-term memory), this tool
//! searches raw conversation history stored in `sessions.db`.

use std::collections::HashMap;
use std::path::PathBuf;

use async_trait::async_trait;
use serde_json::{json, Value};

use super::base::Tool;
use crate::session::db::SessionDb;

/// Tool that searches past session conversations via FTS5.
pub struct SessionSearchTool {
    db_path: PathBuf,
}

impl SessionSearchTool {
    pub fn new(db_path: PathBuf) -> Self {
        Self { db_path }
    }
}

#[async_trait]
impl Tool for SessionSearchTool {
    fn name(&self) -> &str {
        "session_search"
    }

    fn description(&self) -> &str {
        "Search past conversations by keyword. Use this to recall what was discussed \
         in previous sessions — e.g. 'how did we fix the Docker issue?' or 'what model \
         did we use for the benchmark?'. Returns matching messages with context."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (keywords). FTS5 supports AND, OR, NOT, and phrase \"quotes\"."
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results to return. Default: 10."
                },
                "channel": {
                    "type": "string",
                    "description": "Filter to a specific channel prefix (e.g. 'cli:', 'telegram:'). Optional."
                }
            },
            "required": ["query"]
        })
    }

    async fn execute(&self, params: HashMap<String, Value>) -> String {
        let query = match params.get("query").and_then(|v| v.as_str()) {
            Some(q) if !q.trim().is_empty() => q.trim().to_string(),
            _ => return "Error: 'query' parameter is required and must be non-empty.".to_string(),
        };

        let limit = params.get("limit").and_then(|v| v.as_u64()).unwrap_or(10) as usize;

        let channel = params
            .get("channel")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let db = SessionDb::new(&self.db_path);
        let results = db.search_messages(&query, limit, channel.as_deref()).await;

        if results.is_empty() {
            return format!("No results found for '{}'.", query);
        }

        let mut output = format!("Found {} result(s) for '{}':\n\n", results.len(), query);
        for (i, r) in results.iter().enumerate() {
            let snippet = if !r.snippet.is_empty() {
                &r.snippet
            } else {
                &r.content
            };
            output.push_str(&format!(
                "--- Result {} ---\n\
                 Session: {} ({})\n\
                 Time: {}\n\
                 Role: {}\n\
                 {}\n\n",
                i + 1,
                r.session_id,
                r.session_key,
                r.timestamp,
                r.role,
                snippet,
            ));
        }

        // Truncate to avoid huge tool results (UTF-8 safe).
        if output.len() > 8000 {
            let truncated: String = output.chars().take(8000).collect();
            format!("{}\n... (truncated)", truncated)
        } else {
            output
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::TempDir;

    fn make_tool() -> (TempDir, SessionSearchTool) {
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("sessions.db");
        let tool = SessionSearchTool::new(db_path);
        (tmp, tool)
    }

    #[test]
    fn test_name() {
        let (_tmp, tool) = make_tool();
        assert_eq!(tool.name(), "session_search");
    }

    #[test]
    fn test_parameters_schema() {
        let (_tmp, tool) = make_tool();
        let params = tool.parameters();
        assert_eq!(params["type"], "object");
        assert!(params["properties"]["query"].is_object());
        assert!(params["properties"]["limit"].is_object());
        assert!(params["properties"]["channel"].is_object());
        let required = params["required"].as_array().unwrap();
        assert_eq!(required.len(), 1);
        assert_eq!(required[0], "query");
    }

    #[tokio::test]
    async fn test_empty_query_returns_error() {
        let (_tmp, tool) = make_tool();
        let mut params = HashMap::new();
        params.insert("query".to_string(), json!(""));
        let result = tool.execute(params).await;
        assert!(
            result.contains("Error"),
            "Empty query must return Error: {}",
            result
        );
    }

    #[tokio::test]
    async fn test_missing_query_returns_error() {
        let (_tmp, tool) = make_tool();
        let params = HashMap::new();
        let result = tool.execute(params).await;
        assert!(
            result.contains("Error"),
            "Missing query must return Error: {}",
            result
        );
    }

    #[tokio::test]
    async fn test_no_results_message() {
        let (_tmp, tool) = make_tool();
        let mut params = HashMap::new();
        params.insert("query".to_string(), json!("xyznonexistent_abc"));
        let result = tool.execute(params).await;
        assert!(
            result.contains("No results found"),
            "Expected no-results message, got: {}",
            result
        );
    }

    #[tokio::test]
    async fn test_finds_message_by_keyword() {
        let (tmp, tool) = make_tool();
        // Seed the DB with a known message.
        let db = SessionDb::new(&tmp.path().join("sessions.db"));
        let session = db.create_session("cli:default").await;
        db.add_message(
            &session.id,
            &json!({"role": "user", "content": "How do I configure Rustfmt?"}),
        )
        .await;

        let mut params = HashMap::new();
        params.insert("query".to_string(), json!("Rustfmt"));
        let result = tool.execute(params).await;
        assert!(
            result.contains("Rustfmt") || result.contains("rustfmt"),
            "Expected Rustfmt in results, got: {}",
            result
        );
    }

    #[tokio::test]
    async fn test_channel_filter_limits_results() {
        let (tmp, tool) = make_tool();
        let db = SessionDb::new(&tmp.path().join("sessions.db"));

        // Insert into two different channels.
        let cli = db.create_session("cli:default").await;
        let tg = db.create_session("telegram:42").await;
        db.add_message(
            &cli.id,
            &json!({"role": "user", "content": "CLI benchmark result"}),
        )
        .await;
        db.add_message(
            &tg.id,
            &json!({"role": "user", "content": "Telegram benchmark result"}),
        )
        .await;

        let mut params = HashMap::new();
        params.insert("query".to_string(), json!("benchmark"));
        params.insert("channel".to_string(), json!("cli:"));
        let result = tool.execute(params).await;
        assert!(
            result.contains("cli:default"),
            "Should contain cli session: {}",
            result
        );
        assert!(
            !result.contains("telegram:"),
            "Should NOT contain telegram session when filtered: {}",
            result
        );
    }

    #[tokio::test]
    async fn test_limit_parameter() {
        let (tmp, tool) = make_tool();
        let db = SessionDb::new(&tmp.path().join("sessions.db"));
        let session = db.create_session("cli:default").await;

        // Insert 5 matching messages.
        for i in 0..5 {
            db.add_message(
                &session.id,
                &json!({"role": "user", "content": format!("Tokio async message {}", i)}),
            )
            .await;
        }

        let mut params = HashMap::new();
        params.insert("query".to_string(), json!("Tokio"));
        params.insert("limit".to_string(), json!(2));
        let result = tool.execute(params).await;
        // Should contain "Found 2 result(s)" (limit applied).
        assert!(
            result.contains("Found 2 result"),
            "Expected 2 results with limit=2, got: {}",
            result
        );
    }

    #[tokio::test]
    async fn test_to_schema_structure() {
        let (_tmp, tool) = make_tool();
        let schema = tool.to_schema();
        assert_eq!(schema["type"], "function");
        assert_eq!(schema["function"]["name"], "session_search");
        assert!(schema["function"]["description"].as_str().unwrap().len() > 10);
    }
}
