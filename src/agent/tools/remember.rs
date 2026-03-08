//! Remember tool: append a fact or preference to long-term memory (MEMORY.md).

use std::collections::HashMap;
use std::path::PathBuf;

use async_trait::async_trait;
use chrono::Local;
use serde_json::{json, Value};
use tokio::fs;

use super::base::Tool;

/// Tool that writes a single fact into MEMORY.md under a dated section.
pub struct RememberTool {
    workspace: PathBuf,
}

impl RememberTool {
    pub fn new(workspace: PathBuf) -> Self {
        Self { workspace }
    }
}

/// Pure helper — appends `fact` under `## Remembered (<date>)` in `current`.
///
/// If a section for `date` already exists the entry is appended under it.
/// Otherwise a new section is added at the end of the document.
pub fn append_fact(current: &str, fact: &str, date: &str) -> String {
    let header = format!("## Remembered ({})", date);
    let entry = format!("- {}", fact.trim());

    if let Some(pos) = current.find(&header) {
        // Same-date section exists — append the entry before the next `## ` heading.
        let after_header = pos + header.len();
        let section_end = current[after_header..]
            .find("\n## ")
            .map(|p| after_header + p)
            .unwrap_or(current.len());
        let mut result = current[..section_end].to_string();
        result.push('\n');
        result.push_str(&entry);
        result.push_str(&current[section_end..]);
        result
    } else {
        // New date — append a fresh section at the end.
        let mut result = current.trim_end().to_string();
        if !result.is_empty() {
            result.push_str("\n\n");
        }
        result.push_str(&header);
        result.push('\n');
        result.push_str(&entry);
        result.push('\n');
        result
    }
}

#[async_trait]
impl Tool for RememberTool {
    fn name(&self) -> &str {
        "remember"
    }

    fn description(&self) -> &str {
        "Save a fact or preference to long-term memory (MEMORY.md). \
         Use when the user says 'remember this', 'note that', or shares \
         a preference worth keeping."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "fact": {
                    "type": "string",
                    "description": "The fact, preference, or note to remember"
                }
            },
            "required": ["fact"]
        })
    }

    async fn execute(&self, args: HashMap<String, Value>) -> String {
        let fact = match args.get("fact").and_then(|v| v.as_str()) {
            Some(f) => f,
            None => return "Error: Missing required parameter: fact".to_string(),
        };

        if fact.trim().is_empty() {
            return "Error: Fact cannot be empty".to_string();
        }

        let memory_path = self.workspace.join("memory").join("MEMORY.md");

        // Read existing content (start fresh if file doesn't exist yet).
        let current = fs::read_to_string(&memory_path).await.unwrap_or_default();

        let date = Local::now().format("%Y-%m-%d").to_string();
        let updated = append_fact(&current, fact, &date);

        // Write atomically via temp file so a crash never corrupts MEMORY.md.
        let tmp_path = memory_path.with_extension("md.tmp");
        if let Some(parent) = memory_path.parent() {
            if let Err(e) = fs::create_dir_all(parent).await {
                return format!("Error: Failed to create memory dir: {}", e);
            }
        }
        if let Err(e) = fs::write(&tmp_path, &updated).await {
            return format!("Error: Failed to write: {}", e);
        }
        if let Err(e) = fs::rename(&tmp_path, &memory_path).await {
            return format!("Error: Failed to save: {}", e);
        }

        format!("Remembered: {}", fact.trim())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    // ---------------------------------------------------------------
    // Pure-function tests (no filesystem)
    // ---------------------------------------------------------------

    #[test]
    fn test_append_fact_empty_doc() {
        let result = append_fact("", "I prefer dark mode", "2026-03-02");
        assert!(result.contains("## Remembered (2026-03-02)"));
        assert!(result.contains("- I prefer dark mode"));
    }

    #[test]
    fn test_append_fact_existing_content_preserved() {
        let current = "# Memory\n\nSome existing content\n";
        let result = append_fact(current, "Use Rust for CLI tools", "2026-03-02");
        assert!(result.contains("Some existing content"));
        assert!(result.contains("## Remembered (2026-03-02)"));
        assert!(result.contains("- Use Rust for CLI tools"));
    }

    #[test]
    fn test_append_fact_same_day_no_duplicate_header() {
        let current = "## Remembered (2026-03-02)\n- First fact\n";
        let result = append_fact(current, "Second fact", "2026-03-02");
        assert!(result.contains("- First fact"));
        assert!(result.contains("- Second fact"));
        assert_eq!(
            result.matches("## Remembered (2026-03-02)").count(),
            1,
            "must not duplicate the header"
        );
    }

    #[test]
    fn test_append_fact_same_day_order_preserved() {
        let current = "## Remembered (2026-03-02)\n- First fact\n";
        let result = append_fact(current, "Second fact", "2026-03-02");
        let first_pos = result.find("First fact").unwrap();
        let second_pos = result.find("Second fact").unwrap();
        assert!(
            first_pos < second_pos,
            "first fact must appear before second"
        );
    }

    #[test]
    fn test_append_fact_different_day_new_section() {
        let current = "## Remembered (2026-03-01)\n- Old fact\n";
        let result = append_fact(current, "New fact", "2026-03-02");
        assert!(result.contains("## Remembered (2026-03-01)"));
        assert!(result.contains("## Remembered (2026-03-02)"));
        assert!(result.contains("- Old fact"));
        assert!(result.contains("- New fact"));
    }

    #[test]
    fn test_append_fact_trims_whitespace() {
        let result = append_fact("", "  padded fact  ", "2026-03-02");
        assert!(
            result.contains("- padded fact"),
            "leading/trailing space trimmed"
        );
    }

    // ---------------------------------------------------------------
    // Tool trait tests
    // ---------------------------------------------------------------

    #[test]
    fn test_tool_name() {
        let tool = RememberTool::new(PathBuf::from("/tmp"));
        assert_eq!(tool.name(), "remember");
    }

    #[test]
    fn test_tool_params_require_fact() {
        let tool = RememberTool::new(PathBuf::from("/tmp"));
        let params = tool.parameters();
        let required = params["required"].as_array().unwrap();
        assert!(required.contains(&json!("fact")));
    }

    #[test]
    fn test_tool_params_schema_type() {
        let tool = RememberTool::new(PathBuf::from("/tmp"));
        let params = tool.parameters();
        assert_eq!(params["type"], "object");
        assert!(params["properties"]["fact"].is_object());
    }

    // ---------------------------------------------------------------
    // Async execute tests (filesystem)
    // ---------------------------------------------------------------

    #[tokio::test]
    async fn test_empty_fact_returns_error() {
        let dir = TempDir::new().unwrap();
        let tool = RememberTool::new(dir.path().to_path_buf());
        let mut args = HashMap::new();
        args.insert("fact".to_string(), json!("  "));
        let result = tool.execute(args).await;
        assert!(result.starts_with("Error:"), "got: {}", result);
    }

    #[tokio::test]
    async fn test_missing_fact_param_returns_error() {
        let dir = TempDir::new().unwrap();
        let tool = RememberTool::new(dir.path().to_path_buf());
        let result = tool.execute(HashMap::new()).await;
        assert!(result.starts_with("Error:"), "got: {}", result);
    }

    #[tokio::test]
    async fn test_filesystem_round_trip() {
        let dir = TempDir::new().unwrap();
        let tool = RememberTool::new(dir.path().to_path_buf());

        let mut args = HashMap::new();
        args.insert("fact".to_string(), json!("Testing round trip"));
        let result = tool.execute(args).await;
        assert!(
            result.starts_with("Remembered:"),
            "expected success, got: {}",
            result
        );

        let content = std::fs::read_to_string(dir.path().join("memory").join("MEMORY.md")).unwrap();
        assert!(content.contains("- Testing round trip"));
    }

    #[tokio::test]
    async fn test_filesystem_appends_on_second_call() {
        let dir = TempDir::new().unwrap();
        let tool = RememberTool::new(dir.path().to_path_buf());

        let mut args1 = HashMap::new();
        args1.insert("fact".to_string(), json!("First fact"));
        tool.execute(args1).await;

        let mut args2 = HashMap::new();
        args2.insert("fact".to_string(), json!("Second fact"));
        tool.execute(args2).await;

        let content = std::fs::read_to_string(dir.path().join("memory").join("MEMORY.md")).unwrap();
        assert!(content.contains("- First fact"), "first fact missing");
        assert!(content.contains("- Second fact"), "second fact missing");
    }

    #[tokio::test]
    async fn test_memory_dir_created_if_missing() {
        let dir = TempDir::new().unwrap();
        // Intentionally do NOT create the memory subdirectory.
        let tool = RememberTool::new(dir.path().to_path_buf());

        let mut args = HashMap::new();
        args.insert("fact".to_string(), json!("Auto-create dir test"));
        let result = tool.execute(args).await;
        assert!(
            !result.starts_with("Error:"),
            "should succeed even with missing dir, got: {}",
            result
        );
        assert!(dir.path().join("memory").join("MEMORY.md").exists());
    }
}
