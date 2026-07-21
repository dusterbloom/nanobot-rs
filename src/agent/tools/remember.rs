//! Remember tool: manage facts and preferences in long-term memory (MEMORY.md).

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use async_trait::async_trait;
use chrono::Local;
use serde_json::{json, Value};
use tokio::fs;

use super::base::{PermissionLevel, Tool};

const MAX_FACT_CHARS: usize = 180;

/// Tool that manages facts in MEMORY.md under dated sections.
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

pub fn memory_has_fact(current: &str, fact: &str) -> bool {
    let needle = normalize_fact(fact);
    current
        .lines()
        .filter_map(fact_text_from_line)
        .any(|existing| normalize_fact(existing) == needle)
}

pub fn replace_fact(current: &str, old_fact: &str, new_fact: &str) -> (String, usize) {
    let needle = normalize_fact(old_fact);
    let mut replaced = 0usize;
    let lines: Vec<String> = current
        .lines()
        .map(|line| {
            if fact_text_from_line(line)
                .map(|fact| normalize_fact(fact) == needle)
                .unwrap_or(false)
            {
                replaced += 1;
                format!("{}- {}", fact_indent(line), new_fact.trim())
            } else {
                line.to_string()
            }
        })
        .collect();
    (join_preserving_final_newline(lines, current), replaced)
}

pub fn delete_fact(current: &str, fact: &str) -> (String, usize) {
    let needle = normalize_fact(fact);
    let mut removed = 0usize;
    let lines: Vec<String> = current
        .lines()
        .filter_map(|line| {
            let should_remove = fact_text_from_line(line)
                .map(|existing| normalize_fact(existing) == needle)
                .unwrap_or(false);
            if should_remove {
                removed += 1;
                None
            } else {
                Some(line.to_string())
            }
        })
        .collect();
    (join_preserving_final_newline(lines, current), removed)
}

pub fn dedupe_facts(current: &str) -> (String, usize) {
    let mut seen = HashSet::new();
    let mut removed = 0usize;
    let lines: Vec<String> = current
        .lines()
        .filter_map(|line| {
            if let Some(fact) = fact_text_from_line(line) {
                let key = normalize_fact(fact);
                if !seen.insert(key) {
                    removed += 1;
                    return None;
                }
            }
            Some(line.to_string())
        })
        .collect();
    (join_preserving_final_newline(lines, current), removed)
}

pub fn list_facts(current: &str, limit: usize) -> Vec<String> {
    current
        .lines()
        .filter_map(fact_text_from_line)
        .take(limit)
        .map(|fact| fact.to_string())
        .collect()
}

#[async_trait]
impl Tool for RememberTool {
    fn name(&self) -> &str {
        "remember"
    }

    fn permission(&self) -> PermissionLevel {
        PermissionLevel::Write
    }

    fn description(&self) -> &str {
        "Manage curated long-term facts in MEMORY.md. Add only concise, durable facts explicitly stated by the user; do not infer emotions, intent, causality, or narrative context. One fact per call, at most 180 characters. Default action is add when fact is present; otherwise list."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add", "list", "replace", "delete", "dedupe"],
                    "description": "Memory operation. Default: add"
                },
                "fact": {
                    "type": "string",
                    "maxLength": MAX_FACT_CHARS,
                    "description": "One concise fact to add, or the exact fact to delete. For add, use only an explicit user-stated fact; no interpretation or session narrative."
                },
                "old_fact": {
                    "type": "string",
                    "description": "Exact fact to replace when action='replace'"
                },
                "new_fact": {
                    "type": "string",
                    "description": "Replacement fact when action='replace'"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum facts to show for action='list'. Default: 50, max: 200"
                }
            },
            "required": []
        })
    }

    async fn execute(&self, args: HashMap<String, Value>) -> String {
        let action = args
            .get("action")
            .and_then(|v| v.as_str())
            .map(str::to_ascii_lowercase)
            .unwrap_or_else(|| {
                if args.contains_key("fact") {
                    "add".to_string()
                } else {
                    "list".to_string()
                }
            });

        let memory_path = self.workspace.join("memory").join("MEMORY.md");

        // Share one read/modify/write transaction with reflection. Reflection
        // may spend time deriving a new MEMORY.md from SQLite, so waiting here
        // prevents either writer from replacing facts based on stale content.
        let _memory_guard = crate::agent::memory::memory_transaction_lock().lock().await;

        // Read existing content (start fresh if file doesn't exist yet).
        let current = fs::read_to_string(&memory_path).await.unwrap_or_default();

        let (updated, message, should_write) = match action.as_str() {
            "add" => {
                let fact = match required_string(&args, "fact") {
                    Ok(f) => f,
                    Err(e) => return e,
                };
                if let Err(error) = validate_new_fact(fact) {
                    return error;
                }
                if memory_has_fact(&current, fact) {
                    return format!("Already remembered: {}", fact.trim());
                }
                let date = Local::now().format("%Y-%m-%d").to_string();
                (
                    append_fact(&current, fact, &date),
                    format!("Remembered: {}", fact.trim()),
                    true,
                )
            }
            "list" => {
                let limit = args
                    .get("limit")
                    .and_then(|v| v.as_u64())
                    .map(|v| (v as usize).clamp(1, 200))
                    .unwrap_or(50);
                let facts = list_facts(&current, limit);
                if facts.is_empty() {
                    return "No memory facts found.".to_string();
                }
                let mut out = format!("Memory facts (showing {}):", facts.len());
                for fact in facts {
                    out.push_str(&format!("\n- {}", fact));
                }
                return out;
            }
            "replace" => {
                let old_fact = match required_string(&args, "old_fact") {
                    Ok(f) => f,
                    Err(e) => return e,
                };
                let new_fact = match required_string(&args, "new_fact") {
                    Ok(f) => f,
                    Err(e) => return e,
                };
                if let Err(error) = validate_new_fact(new_fact) {
                    return error;
                }
                let (updated, count) = replace_fact(&current, old_fact, new_fact);
                if count == 0 {
                    return format!(
                        "Error: memory fact not found for replace: {}",
                        old_fact.trim()
                    );
                }
                (updated, format!("Replaced {} memory fact(s).", count), true)
            }
            "delete" => {
                let fact = match required_string(&args, "fact") {
                    Ok(f) => f,
                    Err(e) => return e,
                };
                let (updated, count) = delete_fact(&current, fact);
                if count == 0 {
                    return format!("Error: memory fact not found for delete: {}", fact.trim());
                }
                (updated, format!("Deleted {} memory fact(s).", count), true)
            }
            "dedupe" => {
                let (updated, count) = dedupe_facts(&current);
                if count == 0 {
                    return "No duplicate memory facts found.".to_string();
                }
                (
                    updated,
                    format!("Removed {} duplicate memory fact(s).", count),
                    true,
                )
            }
            _ => {
                return "Error: action must be one of: add, list, replace, delete, dedupe"
                    .to_string()
            }
        };

        if !should_write {
            return message;
        }

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

        message
    }
}

fn required_string<'a>(args: &'a HashMap<String, Value>, key: &str) -> Result<&'a str, String> {
    match args.get(key).and_then(|v| v.as_str()).map(str::trim) {
        Some(s) if !s.is_empty() => Ok(s),
        _ => Err(format!("Error: Missing required parameter: {}", key)),
    }
}

fn validate_new_fact(fact: &str) -> Result<(), String> {
    let fact = fact.trim();
    let chars = fact.chars().count();
    if chars > MAX_FACT_CHARS {
        return Err(format!(
            "Error: Memory fact is too verbose ({chars} characters; maximum {MAX_FACT_CHARS}). Store one concise, explicit user-stated fact without interpretation."
        ));
    }
    if fact.contains('\n')
        || fact.contains('\r')
        || fact.starts_with("- ")
        || fact.starts_with("# ")
    {
        return Err(
            "Error: Memory fact must be one plain-text fact, not a list or multi-line narrative."
                .to_string(),
        );
    }
    Ok(())
}

fn fact_text_from_line(line: &str) -> Option<&str> {
    line.trim_start().strip_prefix("- ").map(str::trim)
}

fn fact_indent(line: &str) -> &str {
    let trimmed_len = line.trim_start().len();
    &line[..line.len().saturating_sub(trimmed_len)]
}

fn normalize_fact(fact: &str) -> String {
    fact.trim().to_ascii_lowercase()
}

fn join_preserving_final_newline(lines: Vec<String>, original: &str) -> String {
    let mut out = lines.join("\n");
    if original.ends_with('\n') && !out.is_empty() {
        out.push('\n');
    }
    out
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
        assert!(
            required.is_empty(),
            "action-specific parameters are validated at execution time"
        );
        assert!(params["properties"]["fact"].is_object());
        assert!(params["properties"]["action"].is_object());
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
    async fn test_missing_fact_param_defaults_to_list() {
        let dir = TempDir::new().unwrap();
        let tool = RememberTool::new(dir.path().to_path_buf());
        let result = tool.execute(HashMap::new()).await;
        assert_eq!(result, "No memory facts found.");
    }

    #[tokio::test]
    async fn test_add_rejects_verbose_interpretive_fact() {
        let dir = TempDir::new().unwrap();
        let tool = RememberTool::new(dir.path().to_path_buf());
        let verbose = "Peppi said something memorable during this session, which means the user was expressing a complex emotional reaction to the agent and the surrounding project context, with additional speculative narrative details that were never explicitly established.";
        let result = tool
            .execute(HashMap::from([("fact".to_string(), json!(verbose))]))
            .await;
        assert!(result.contains("too verbose"), "got: {result}");
        assert!(!dir.path().join("memory").join("MEMORY.md").exists());
    }

    #[tokio::test]
    async fn test_add_rejects_multiline_fact() {
        let dir = TempDir::new().unwrap();
        let tool = RememberTool::new(dir.path().to_path_buf());
        let result = tool
            .execute(HashMap::from([(
                "fact".to_string(),
                json!("First claim\n- inferred second claim"),
            )]))
            .await;
        assert!(result.contains("one plain-text fact"), "got: {result}");
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
    async fn test_duplicate_fact_is_not_appended() {
        let dir = TempDir::new().unwrap();
        let tool = RememberTool::new(dir.path().to_path_buf());

        let mut args = HashMap::new();
        args.insert("fact".to_string(), json!("Use concise answers"));
        assert!(tool.execute(args.clone()).await.starts_with("Remembered:"));
        let result = tool.execute(args).await;
        assert!(result.starts_with("Already remembered:"), "got: {result}");

        let content = std::fs::read_to_string(dir.path().join("memory").join("MEMORY.md")).unwrap();
        assert_eq!(content.matches("Use concise answers").count(), 1);
    }

    #[tokio::test]
    async fn test_replace_delete_and_list_memory_facts() {
        let dir = TempDir::new().unwrap();
        let tool = RememberTool::new(dir.path().to_path_buf());

        let mut add = HashMap::new();
        add.insert("fact".to_string(), json!("Old preference"));
        tool.execute(add).await;

        let mut replace = HashMap::new();
        replace.insert("action".to_string(), json!("replace"));
        replace.insert("old_fact".to_string(), json!("Old preference"));
        replace.insert("new_fact".to_string(), json!("New preference"));
        let result = tool.execute(replace).await;
        assert!(result.starts_with("Replaced 1"), "got: {result}");

        let mut list = HashMap::new();
        list.insert("action".to_string(), json!("list"));
        let result = tool.execute(list).await;
        assert!(result.contains("New preference"), "got: {result}");
        assert!(!result.contains("Old preference"), "got: {result}");

        let mut delete = HashMap::new();
        delete.insert("action".to_string(), json!("delete"));
        delete.insert("fact".to_string(), json!("New preference"));
        let result = tool.execute(delete).await;
        assert!(result.starts_with("Deleted 1"), "got: {result}");
    }

    #[test]
    fn test_dedupe_facts_removes_later_duplicates() {
        let current = "\
## Remembered (2026-03-01)
- Prefer Rust
- prefer rust
- Prefer tests
";
        let (updated, removed) = dedupe_facts(current);
        assert_eq!(removed, 1);
        assert_eq!(updated.matches("Prefer").count(), 2);
        assert!(updated.contains("- Prefer Rust"));
        assert!(updated.contains("- Prefer tests"));
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

    #[tokio::test]
    async fn test_execute_waits_for_memory_transaction_lock() {
        use std::time::Duration;

        let guard = crate::agent::memory::memory_transaction_lock().lock().await;
        let dir = TempDir::new().unwrap();
        let tool = RememberTool::new(dir.path().to_path_buf());
        let mut args = HashMap::new();
        args.insert("fact".to_string(), json!("Serialized memory write"));

        let mut task = tokio::spawn(async move { tool.execute(args).await });
        assert!(
            tokio::time::timeout(Duration::from_millis(50), &mut task)
                .await
                .is_err(),
            "remember should wait while reflection owns the memory transaction"
        );

        drop(guard);
        let result = tokio::time::timeout(Duration::from_secs(1), task)
            .await
            .expect("remember should resume after the transaction releases")
            .expect("remember task should complete");
        assert!(result.starts_with("Remembered:"), "got: {result}");
    }
}
