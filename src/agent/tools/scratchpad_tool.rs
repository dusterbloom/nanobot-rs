//! Scratchpad tool for inter-agent communication.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;

use super::base::Tool;
use crate::agent::scratchpad::SharedScratchpad;

/// Tool exposing the shared scratchpad to the agent.
pub struct ScratchpadTool {
    pad: Arc<SharedScratchpad>,
}

impl ScratchpadTool {
    pub fn new(pad: Arc<SharedScratchpad>) -> Self {
        Self { pad }
    }
}

#[async_trait]
impl Tool for ScratchpadTool {
    fn name(&self) -> &str {
        "scratchpad"
    }

    fn description(&self) -> &str {
        "A shared scratchpad for storing and retrieving notes, data, and context \
         that persists across conversations and is accessible by all agents. \
         Use this to leave notes for future sessions, share data between agents, \
         or store intermediate results."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["write", "read", "list", "append", "delete"],
                    "description": "Action to perform"
                },
                "key": {
                    "type": "string",
                    "description": "Key name (required for write/read/append/delete)"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write or append (required for write/append)"
                }
            },
            "required": ["action"]
        })
    }

    async fn execute(&self, params: HashMap<String, serde_json::Value>) -> String {
        let action = match params.get("action").and_then(|v| v.as_str()) {
            Some(a) => a,
            None => return "Error: 'action' parameter is required".to_string(),
        };

        match action {
            "write" => {
                let key = match params.get("key").and_then(|v| v.as_str()) {
                    Some(k) => k,
                    None => return "Error: 'key' is required for write".to_string(),
                };
                let content = match params.get("content").and_then(|v| v.as_str()) {
                    Some(c) => c,
                    None => return "Error: 'content' is required for write".to_string(),
                };
                self.pad.write(key, content)
            }
            "read" => {
                let key = match params.get("key").and_then(|v| v.as_str()) {
                    Some(k) => k,
                    None => return "Error: 'key' is required for read".to_string(),
                };
                self.pad
                    .read(key)
                    .unwrap_or_else(|| format!("Error: Key '{}' not found", key))
            }
            "list" => {
                let keys = self.pad.list();
                if keys.is_empty() {
                    "Scratchpad is empty.".to_string()
                } else {
                    format!("Keys: {}", keys.join(", "))
                }
            }
            "append" => {
                let key = match params.get("key").and_then(|v| v.as_str()) {
                    Some(k) => k,
                    None => return "Error: 'key' is required for append".to_string(),
                };
                let content = match params.get("content").and_then(|v| v.as_str()) {
                    Some(c) => c,
                    None => return "Error: 'content' is required for append".to_string(),
                };
                self.pad.append(key, content)
            }
            "delete" => {
                let key = match params.get("key").and_then(|v| v.as_str()) {
                    Some(k) => k,
                    None => return "Error: 'key' is required for delete".to_string(),
                };
                self.pad.delete(key)
            }
            _ => format!(
                "Error: Unknown action '{}'. Use write, read, list, append, or delete.",
                action
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_scratchpad_tool_crud() {
        let dir = tempfile::tempdir().unwrap();
        let pad = Arc::new(SharedScratchpad::new(dir.path()));
        let tool = ScratchpadTool::new(pad);

        // Write.
        let mut params = HashMap::new();
        params.insert("action".to_string(), serde_json::json!("write"));
        params.insert("key".to_string(), serde_json::json!("notes"));
        params.insert("content".to_string(), serde_json::json!("Important data"));
        let result = tool.execute(params).await;
        assert!(result.contains("Written"));

        // Read.
        let mut params = HashMap::new();
        params.insert("action".to_string(), serde_json::json!("read"));
        params.insert("key".to_string(), serde_json::json!("notes"));
        let result = tool.execute(params).await;
        assert_eq!(result, "Important data");

        // List.
        let mut params = HashMap::new();
        params.insert("action".to_string(), serde_json::json!("list"));
        let result = tool.execute(params).await;
        assert!(result.contains("notes"));

        // Append.
        let mut params = HashMap::new();
        params.insert("action".to_string(), serde_json::json!("append"));
        params.insert("key".to_string(), serde_json::json!("notes"));
        params.insert("content".to_string(), serde_json::json!("More data"));
        let result = tool.execute(params).await;
        assert!(result.contains("Appended"));

        // Delete.
        let mut params = HashMap::new();
        params.insert("action".to_string(), serde_json::json!("delete"));
        params.insert("key".to_string(), serde_json::json!("notes"));
        let result = tool.execute(params).await;
        assert!(result.contains("Deleted"));
    }

    #[test]
    fn test_tool_name() {
        let dir = tempfile::tempdir().unwrap();
        let pad = Arc::new(SharedScratchpad::new(dir.path()));
        let tool = ScratchpadTool::new(pad);
        assert_eq!(tool.name(), "scratchpad");
    }
}
