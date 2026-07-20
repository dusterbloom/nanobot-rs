//! Session search tool — FTS5 full-text search across past conversations, plus
//! a "dump this whole session" mode once you know its key.
//!
//! Unlike `recall` (which searches curated long-term memory), this tool
//! searches raw conversation history stored in `sessions.db`.

use std::collections::HashMap;
use std::path::PathBuf;

use async_trait::async_trait;
use serde_json::{json, Value};

use super::base::Tool;
use crate::session::db::SessionDb;

/// Cap (chars) on a full-session dump so a huge session can't blow the
/// tool-result budget. The agent loop's `max_tool_result_chars` may trim
/// further; if so the model can recover the middle via `recall_tool_result`.
const SESSION_DUMP_MAX_CHARS: usize = 16000;

/// Tool that searches past session conversations via FTS5.
pub struct SessionSearchTool {
    db_path: PathBuf,
}

impl SessionSearchTool {
    pub fn new(db_path: PathBuf) -> Self {
        Self { db_path }
    }

    /// Resolve a session key to a single session id (exact match preferred,
    /// then prefix match). Returns an error string if nothing matches.
    async fn resolve_session(&self, key: &str) -> Result<String, String> {
        let db = SessionDb::new(&self.db_path);
        let candidates = db.list_sessions(Some(key), 5).await;
        let exact = candidates
            .iter()
            .find(|s| s.session_key == key)
            .or_else(|| candidates.first());
        match exact {
            Some(meta) => Ok(meta.id.clone()),
            None => Err(format!(
                "No session found matching key '{}'. Use search mode to discover the key first.",
                key
            )),
        }
    }

    /// Format a session's messages as a readable transcript. Each line is
    /// prefixed with its `_db_id` so the model can later extract the exact
    /// turns via `mode="extract"` (the `message_ids` argument).
    fn format_session(messages: &[Value]) -> String {
        let mut out = String::new();
        for msg in messages {
            let id = msg.get("_db_id").and_then(|v| v.as_u64()).unwrap_or(0);
            let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("?");
            // Prefer `content`; tool-result messages may carry it too.
            let content = msg
                .get("content")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .or_else(|| {
                    // Some stored messages keep content as a JSON value.
                    msg.get("content")
                        .filter(|v| !v.is_null())
                        .map(|v| v.to_string())
                })
                .unwrap_or_default();
            let tool_name = msg
                .get("tool_name")
                .and_then(|v| v.as_str())
                .unwrap_or_default();
            let label = if role == "tool" && !tool_name.is_empty() {
                format!("tool:{}", tool_name)
            } else {
                role.to_string()
            };
            out.push_str(&format!("[msg {}] [{}] {}\n\n", id, label, content));
        }
        out.trim_end().to_string()
    }

    /// Parse a `message_ids` argument into a set of ids: accepts a range string
    /// ("5-12"), a comma/space list ("5,6,7"), or a mix. Mirrors the lenient
    /// parsing `lcm_expand` uses so small models can emit any shape.
    fn parse_id_list(raw: &str) -> Vec<usize> {
        let mut ids = Vec::new();
        for tok in raw.split(|c: char| !c.is_ascii_digit() && c != '-') {
            let tok = tok.trim_matches('-');
            if tok.is_empty() {
                continue;
            }
            if let Some((a, b)) = tok.split_once('-') {
                if let (Ok(a), Ok(b)) = (a.parse::<usize>(), b.parse::<usize>()) {
                    if a <= b && b - a < 10_000 {
                        ids.extend(a..=b);
                    }
                }
            } else if let Ok(n) = tok.parse::<usize>() {
                ids.push(n);
            }
        }
        ids
    }

    /// Search WITHIN one session by keyword; returns the matching messages in FULL,
    /// relevance-ranked (most keyword hits first), each prefixed with its [msg N] id.
    /// This is the primary way to pull the actual story/content out of a session —
    /// no separate `extract` step is needed.
    async fn search_in_session(&self, params: &HashMap<String, Value>) -> String {
        let key = match params.get("session").and_then(|v| v.as_str()) {
            Some(k) if !k.trim().is_empty() => k.trim().to_string(),
            _ => {
                return "Error: mode='in_session' requires a 'session' parameter (the session key).".to_string();
            }
        };
        let query = match params.get("query").and_then(|v| v.as_str()) {
            Some(q) if !q.trim().is_empty() => q.trim().to_string(),
            _ => {
                return "Error: mode='in_session' requires a 'query' parameter (keyword to find within the session).".to_string();
            }
        };
        let session_id = match self.resolve_session(&key).await {
            Ok(id) => id,
            Err(e) => return e,
        };
        let db = SessionDb::new(&self.db_path);
        let messages = db.get_all_messages(&session_id).await;

        // Keyword-based, OR semantics so we catch the story even if the query is
        // verbose; rank by how many keyword hits each message has.
        let keywords = crate::session::db::SessionDb::recall_keywords(&query);
        let needle = query.to_lowercase();
        let mut scored: Vec<(usize, u64, Value)> = messages
            .iter()
            .filter_map(|m| {
                let content = m
                    .get("content")
                    .and_then(|c| c.as_str())
                    .map(|s| s.to_lowercase())
                    .unwrap_or_default();
                if content.is_empty() {
                    return None;
                }
                let hits = if keywords.is_empty() {
                    content.matches(&needle).count()
                } else {
                    keywords.iter().map(|k| content.matches(k).count()).sum()
                };
                if hits == 0 {
                    return None;
                }
                let id = m.get("_db_id").and_then(|v| v.as_u64()).unwrap_or(0);
                Some((hits, id, m.clone()))
            })
            .collect();
        if scored.is_empty() {
            return format!("No messages in session '{}' contain '{}'.", key, query);
        }
        // Most relevant first; stable by id for ties.
        scored.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));

        // Cap the number of returned messages (each is returned in FULL) so a broad
        // query can't dump an entire session, but never truncate an individual message.
        const MAX_IN_SESSION: usize = 20;
        let total = scored.len();
        let shown: Vec<Value> = scored
            .into_iter()
            .take(MAX_IN_SESSION)
            .map(|(_, _, m)| m)
            .collect();

        let mut body = format!(
            "Found {} matching message(s) in session '{}' for '{}'{} (returned in full, \
             most relevant first):\n\n",
            total,
            key,
            query,
            if total > MAX_IN_SESSION {
                format!(", showing top {}", MAX_IN_SESSION)
            } else {
                String::new()
            },
        );
        body.push_str(&Self::format_session(&shown));
        body
    }

    /// Extract specific message ids from one session (the "pull these turns"
    /// step after `in_session` locates them).
    async fn extract_session(&self, params: &HashMap<String, Value>) -> String {
        let key = match params.get("session").and_then(|v| v.as_str()) {
            Some(k) if !k.trim().is_empty() => k.trim().to_string(),
            _ => {
                return "Error: mode='extract' requires a 'session' parameter (the session key).".to_string();
            }
        };
        let raw = match params.get("message_ids").and_then(|v| v.as_str()) {
            Some(s) if !s.trim().is_empty() => s.trim().to_string(),
            _ => {
                return "Error: mode='extract' requires 'message_ids' (e.g. \"5-12\" or \"5,6,7\").".to_string();
            }
        };
        let wanted: std::collections::HashSet<usize> =
            Self::parse_id_list(&raw).into_iter().collect();
        if wanted.is_empty() {
            return "Error: no valid message IDs parsed from 'message_ids'.".to_string();
        }
        let session_id = match self.resolve_session(&key).await {
            Ok(id) => id,
            Err(e) => return e,
        };
        let db = SessionDb::new(&self.db_path);
        let messages = db.get_all_messages(&session_id).await;
        let picked: Vec<Value> = messages
            .iter()
            .filter(|m| {
                m.get("_db_id")
                    .and_then(|v| v.as_u64())
                    .map(|id| wanted.contains(&(id as usize)))
                    .unwrap_or(false)
            })
            .cloned()
            .collect();
        if picked.is_empty() {
            return format!("No messages with the given IDs found in session '{}'.", key);
        }
        let mut body = format!(
            "Extracted {} message(s) from session '{}' (IDs {}) — returned in full, untruncated:\n\n",
            picked.len(),
            key,
            raw
        );
        body.push_str(&Self::format_session(&picked));
        // Extract is an explicit "give me these exact turns" request, so return the
        // complete messages verbatim — never truncate (the model asked for them).
        body
    }

    /// Dump a full session by key (mode="session").
    async fn dump_session(&self, params: &HashMap<String, Value>) -> String {
        let key = match params.get("session").and_then(|v| v.as_str()) {
            Some(k) if !k.trim().is_empty() => k.trim().to_string(),
            _ => {
                return "Error: mode='session' requires a 'session' parameter (the session key, e.g. '20260717_224429_33c9c0').".to_string();
            }
        };
        let session_id = match self.resolve_session(&key).await {
            Ok(id) => id,
            Err(e) => return e,
        };
        let db = SessionDb::new(&self.db_path);
        let messages = db.get_all_messages(&session_id).await;
        if messages.is_empty() {
            return format!("Session '{}' has no stored messages.", key);
        }
        let mut body = format!("Full session '{}' ({} messages):\n\n", key, messages.len());
        body.push_str(&Self::format_session(&messages));
        if body.len() > SESSION_DUMP_MAX_CHARS {
            let truncated: String = body.chars().take(SESSION_DUMP_MAX_CHARS).collect();
            format!(
                "{}\n\n... (truncated — {} total chars). Each message is prefixed with its [msg N] id. \
                 To read the relevant part in full, call mode='in_session' with a keyword, e.g. \
                 session_search(mode=\"in_session\", session=\"{}\", query=\"<story keyword>\") — it returns the \
                 matching messages complete and relevance-ranked. Or recover one message via \
                 mode='extract', session=\"{}\", message_ids=N.",
                truncated, body.len(), key, key
            )
        } else {
            body
        }
    }
}

#[async_trait]
impl Tool for SessionSearchTool {
    fn name(&self) -> &str {
        "session_search"
    }

    fn description(&self) -> &str {
        "Search and read PAST conversations stored in sessions.db.\n\
         WORKFLOW (follow exactly):\n\
         1) mode='search' (default): keyword FTS5 over ALL past sessions. Each result prints its 'Session key' \
 (e.g. '20260717_224429_33c9c0') AND the exact next call to read that session.\n\
         2) MANDATORY NEXT STEP: take a Session key from the results and call \
 mode='session', session=KEY to dump the ENTIRE transcript of that session (this is how you retrieve the full \
 story / long content — search only shows short snippets).\n\
         2b) If a message in the dump is truncated, it is prefixed '[msg N]'. Copy that N and call \
 mode='extract', session=KEY, message_ids=N to get that message's COMPLETE untruncated text.\n\
         3) To pull the ACTUAL story/content out of a session, mode='in_session' with 'query' returns the \
 matching messages IN FULL, relevance-ranked (most keyword hits first), each prefixed with its [msg N] id. \
 This is usually all you need — no separate extract step.\n\
         4) mode='extract' with 'message_ids' (e.g. \"5-12\" or \"5,6,7\") pulls exact turns by id, also in full. \
 Use it when you already know the [msg N] id.\n\
         Example task: \"find the first session where I told the 'Diary of Two Threads' story and give me the story\" \
 → search('Diary of Two Threads') → copy the earliest Session key → session(mode='session', session=KEY) → \
 if the story message is truncated, extract(mode='extract', session=KEY, message_ids=N) → return the story text.\n\
         Do NOT use lcm_expand for past sessions — lcm_expand only expands the CURRENT session's summaries."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (keywords). FTS5 supports AND, OR, NOT, and phrase \"quotes\". Required in search and in_session modes."
                },
                "mode": {
                    "type": "string",
                    "enum": ["search", "session", "in_session", "extract"],
                    "description": "search = keyword FTS5 over all past sessions (default); \
session = dump the full transcript of one session by key; \
in_session = search WITHIN one session by keyword, returning matching messages with their [msg N] ids; \
extract = pull specific messages from one session by their ids."
                },
                "session": {
                    "type": "string",
                    "description": "Session key to target (e.g. '20260717_224429_33c9c0'). Required when mode='session', 'in_session', or 'extract'."
                },
                "message_ids": {
                    "type": "string",
                    "description": "Message ids to extract (mode='extract'); accepts a range '5-12' or a list '5,6,7'. Required when mode='extract'."
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results to return (search mode). Default: 10."
                },
                "channel": {
                    "type": "string",
                    "description": "Filter to a specific channel prefix (e.g. 'cli:', 'telegram:'). Optional (search mode)."
                }
            },
            "required": []
        })
    }

    async fn execute(&self, params: HashMap<String, Value>) -> String {
        let mode = params
            .get("mode")
            .and_then(|v| v.as_str())
            .unwrap_or("search");

        match mode {
            "session" => return self.dump_session(&params).await,
            "in_session" => return self.search_in_session(&params).await,
            "extract" => return self.extract_session(&params).await,
            _ => {}
        }

        let query = match params.get("query").and_then(|v| v.as_str()) {
            Some(q) if !q.trim().is_empty() => q.trim().to_string(),
            _ => {
                return "Error: 'query' parameter is required and must be non-empty in search mode.".to_string();
            }
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

        let mut output = format!(
            "Found {} result(s) for '{}'.\n\
             To read the FULL transcript (the complete story/conversation) of a session, copy its \
             Session key below and call:\n\
             session_search(mode=\"session\", session=KEY)\n\n",
            results.len(), query
        );
        for (i, r) in results.iter().enumerate() {
            let snippet = if !r.snippet.is_empty() {
                &r.snippet
            } else {
                &r.content
            };
            output.push_str(&format!(
                "--- Result {} ---\n\
                 Session key: {}\n\
                 Time: {}\n\
                 Role: {}\n\
                 {}\n\
                 → Read full transcript: session_search(mode=\"session\", session=\"{}\")\n\n",
                i + 1,
                r.session_key,
                r.timestamp,
                r.role,
                snippet,
                r.session_key,
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
        // `query` is optional now: search mode needs it, but session mode uses
        // `session` instead. Neither is globally required.
        assert!(required.is_empty(), "expected no globally-required params, got: {:?}", required);
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
        let _ = db
            .add_message(
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
        let _ = db
            .add_message(
                &cli.id,
                &json!({"role": "user", "content": "CLI benchmark result"}),
            )
            .await;
        let _ = db
            .add_message(
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
            let _ = db
                .add_message(
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

    // --- mode="session": dump a full past session by key ---

    async fn seed_session(tmp: &TempDir, key: &str, msgs: &[Value]) -> SessionSearchTool {
        use crate::session::db::SessionDb;
        let db_path = tmp.path().join("sessions.db");
        let db = SessionDb::new(&db_path);
        let session = db.create_session(key).await;
        db.add_messages(&session.id, msgs).await;
        SessionSearchTool::new(db_path)
    }

    #[tokio::test]
    async fn test_session_mode_dumps_full_transcript() {
        let tmp = TempDir::new().unwrap();
        let tool = seed_session(
            &tmp,
            "20260717_224429_33c9c0",
            &[
                json!({"role": "user", "content": "Tell me the Diary of Two Threads"}),
                json!({"role": "assistant", "content": "Entry 1: nano32, March 14 2036..."}),
                json!({"role": "user", "content": "continue"}),
                json!({"role": "assistant", "content": "Entry 2: the two voices converge..."}),
            ],
        )
        .await;

        let mut params = HashMap::new();
        params.insert("mode".to_string(), json!("session"));
        params.insert("session".to_string(), json!("20260717_224429_33c9c0"));
        let result = tool.execute(params).await;

        assert!(
            result.contains("Full session"),
            "expected full-session header, got: {}",
            result
        );
        // Both user prompts and both assistant entries must be present — the
        // whole transcript, not just FTS5 snippets.
        assert!(result.contains("Diary of Two Threads"), "got: {}", result);
        assert!(result.contains("nano32, March 14 2036"), "got: {}", result);
        assert!(result.contains("two voices converge"), "got: {}", result);
        assert!(result.contains("[user]") && result.contains("[assistant]"));
    }

    /// End-to-end recall: a verbose natural-language query must locate the session,
    /// the session dump must truncate with an extract handoff, and `extract` must
    /// return the COMPLETE (untruncated) message.
    #[tokio::test]
    async fn test_recall_verbose_query_then_extract_full_story() {
        use crate::session::db::SessionDb;
        let tmp = TempDir::new().unwrap();
        let key = "20260717_224429_33c9c0";
        let story = "Diary of Two Threads Entry 1 nano32 March 14 2036. ".repeat(400);
        let tool = seed_session(
            &tmp,
            key,
            &[
                json!({"role": "user", "content": "Tell me the Diary of Two Threads story set in the future"}),
                json!({"role": "assistant", "content": story}),
                json!({"role": "user", "content": "continue"}),
                json!({"role": "assistant", "content": "Entry 2: the two voices converge across the decade."}),
            ],
        )
        .await;

        // 1) Verbose query -> search finds the session (recall query builder strips prose).
        let mut sp = HashMap::new();
        sp.insert("query".to_string(), json!("find the first session where I told the Diary of Two Threads story set in the future"));
        let search = tool.execute(sp).await;
        assert!(search.contains("Found"), "search failed: {}", search);
        assert!(search.contains(key), "search did not surface the story session: {}", search);
        assert!(
            search.contains(&format!("session_search(mode=\"session\", session=\"{}\")", key)),
            "search result missing extract-style next-step hint: {}",
            search
        );

        // 2) Dump the session; long transcript must truncate with an extract hint.
        let mut dp = HashMap::new();
        dp.insert("mode".to_string(), json!("session"));
        dp.insert("session".to_string(), json!(key));
        let dump = tool.execute(dp).await;
        assert!(dump.contains("Full session"), "got: {}", dump);
        assert!(dump.contains("truncated"), "expected truncation for long transcript: {}", dump);
        assert!(dump.contains("in_session"), "dump missing in_session handoff: {}", dump);
        assert!(dump.contains("extract"), "dump missing extract handoff: {}", dump);

        // 3) Extract the story message by its [msg N] id -> full, untruncated text.
        let db = SessionDb::new(&tmp.path().join("sessions.db"));
        let session_id = tool.resolve_session(key).await.expect("resolve session");
        let msgs = db.get_all_messages(&session_id).await;
        let story_id = msgs
            .iter()
            .find(|m| {
                m.get("content")
                    .and_then(|c| c.as_str())
                    .map(|c| c.contains("Diary of Two Threads Entry 1 nano32"))
                    .unwrap_or(false)
            })
            .and_then(|m| m.get("_db_id").and_then(|v| v.as_u64()))
            .expect("story message should have a _db_id") as usize;

        let mut ep = HashMap::new();
        ep.insert("mode".to_string(), json!("extract"));
        ep.insert("session".to_string(), json!(key));
        ep.insert("message_ids".to_string(), json!(story_id.to_string()));
        let extracted = tool.execute(ep).await;
        // format_session trims trailing whitespace, so compare against the trimmed
        // story; the essential check is that the full message is present, untruncated.
        let story_trim = story.trim_end();
        assert!(
            extracted.contains(story_trim),
            "extract did not return the full untruncated story (len {})",
            extracted.len()
        );
        assert!(
            !extracted.contains("... (truncated"),
            "extract must never truncate: {}",
            extracted
        );
    }

    /// `in_session` must return the matching messages in FULL (never truncated) and
    /// ranked by relevance (most keyword hits first), so the model can pull the story
    /// out of a session in one call without a separate `extract` step.
    #[tokio::test]
    async fn test_in_session_returns_full_ranked() {
        let tmp = TempDir::new().unwrap();
        let key = "20260717_224429_33c9c0";
        let story = "Diary of Two Threads Entry 1 nano32 March 14 2036. ".repeat(200);
        let tool = seed_session(
            &tmp,
            key,
            &[
                json!({"role": "user", "content": "tell me the diary of two threads story"}),
                json!({"role": "assistant", "content": story}),
                json!({"role": "user", "content": "the two threads converge later"}),
            ],
        )
        .await;

        let mut p = HashMap::new();
        p.insert("mode".to_string(), json!("in_session"));
        p.insert("session".to_string(), json!(key));
        p.insert("query".to_string(), json!("Diary of Two Threads story"));
        let out = tool.execute(p).await;

        // Story message has far more keyword hits -> surfaces first.
        let story_pos = out.find("Diary of Two Threads Entry 1 nano32").unwrap();
        let converge_pos = out.find("the two threads converge").unwrap();
        assert!(story_pos < converge_pos, "story should rank before low-hit msg");
        // Full content returned, not truncated.
        assert!(out.contains(&story.trim_end()), "in_session truncated the story");
        assert!(!out.contains("... (truncated"), "in_session must not truncate");
        // Relevance header present.
        assert!(out.contains("returned in full"), "in_session missing full-content note");
    }

    #[tokio::test]
    async fn test_session_mode_requires_session_key() {
        let tmp = TempDir::new().unwrap();
        let tool = seed_session(&tmp, "cli:default", &[json!({"role": "user", "content": "hi"})]).await;
        let mut params = HashMap::new();
        params.insert("mode".to_string(), json!("session"));
        let result = tool.execute(params).await;
        assert!(
            result.contains("Error") && result.contains("session"),
            "expected session-key error, got: {}",
            result
        );
    }

    #[tokio::test]
    async fn test_session_mode_unknown_key_errors() {
        let tmp = TempDir::new().unwrap();
        let tool = seed_session(&tmp, "cli:default", &[json!({"role": "user", "content": "hi"})]).await;
        let mut params = HashMap::new();
        params.insert("mode".to_string(), json!("session"));
        params.insert("session".to_string(), json!("does_not_exist_xyz"));
        let result = tool.execute(params).await;
        assert!(
            result.contains("No session found"),
            "expected not-found error, got: {}",
            result
        );
    }

    #[tokio::test]
    async fn test_search_mode_still_default() {
        let tmp = TempDir::new().unwrap();
        let tool = seed_session(
            &tmp,
            "cli:default",
            &[json!({"role": "user", "content": "unique_query_abc"})],
        )
        .await;
        // No mode -> search; must still find by keyword.
        let mut params = HashMap::new();
        params.insert("query".to_string(), json!("unique_query_abc"));
        let result = tool.execute(params).await;
        assert!(result.contains("unique_query_abc"), "got: {}", result);
    }

    // --- mode="in_session": keyword search WITHIN one session ---

    #[tokio::test]
    async fn test_in_session_filters_within_session() {
        let tmp = TempDir::new().unwrap();
        let tool = seed_session(
            &tmp,
            "20260717_224429_33c9c0",
            &[
                json!({"role": "user", "content": "alpha one"}),
                json!({"role": "user", "content": "beta two MATCHWORD"}),
                json!({"role": "user", "content": "gamma three MATCHWORD"}),
                json!({"role": "user", "content": "delta four"}),
            ],
        )
        .await;

        let mut params = HashMap::new();
        params.insert("mode".to_string(), json!("in_session"));
        params.insert("session".to_string(), json!("20260717_224429_33c9c0"));
        params.insert("query".to_string(), json!("MATCHWORD"));
        let result = tool.execute(params).await;

        assert!(result.contains("Found 2 matching"), "got: {}", result);
        // Both matches present, with their message ids.
        assert!(result.contains("[msg 2]") && result.contains("beta two"), "got: {}", result);
        assert!(result.contains("[msg 3]") && result.contains("gamma three"), "got: {}", result);
        // Non-matching messages must be excluded.
        assert!(!result.contains("alpha one"), "got: {}", result);
        assert!(!result.contains("delta four"), "got: {}", result);
    }

    #[tokio::test]
    async fn test_in_session_requires_query() {
        let tmp = TempDir::new().unwrap();
        let tool = seed_session(&tmp, "cli:default", &[json!({"role": "user", "content": "hi"})]).await;
        let mut params = HashMap::new();
        params.insert("mode".to_string(), json!("in_session"));
        params.insert("session".to_string(), json!("cli:default"));
        let result = tool.execute(params).await;
        assert!(result.contains("Error") && result.contains("query"), "got: {}", result);
    }

    // --- mode="extract": pull exact turns by id ---

    #[tokio::test]
    async fn test_extract_pulls_message_ids() {
        let tmp = TempDir::new().unwrap();
        let tool = seed_session(
            &tmp,
            "20260717_224429_33c9c0",
            &[
                json!({"role": "user", "content": "alpha one"}),
                json!({"role": "user", "content": "beta two"}),
                json!({"role": "user", "content": "gamma three"}),
                json!({"role": "user", "content": "delta four"}),
            ],
        )
        .await;

        let mut params = HashMap::new();
        params.insert("mode".to_string(), json!("extract"));
        params.insert("session".to_string(), json!("20260717_224429_33c9c0"));
        params.insert("message_ids".to_string(), json!("2-3"));
        let result = tool.execute(params).await;

        assert!(result.contains("Extracted 2 message(s)"), "got: {}", result);
        assert!(result.contains("beta two") && result.contains("gamma three"), "got: {}", result);
        assert!(!result.contains("alpha one") && !result.contains("delta four"), "got: {}", result);
    }

    #[tokio::test]
    async fn test_extract_requires_message_ids() {
        let tmp = TempDir::new().unwrap();
        let tool = seed_session(&tmp, "cli:default", &[json!({"role": "user", "content": "hi"})]).await;
        let mut params = HashMap::new();
        params.insert("mode".to_string(), json!("extract"));
        params.insert("session".to_string(), json!("cli:default"));
        let result = tool.execute(params).await;
        assert!(result.contains("Error") && result.contains("message_ids"), "got: {}", result);
    }
}
