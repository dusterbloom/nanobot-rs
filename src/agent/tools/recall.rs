// Error-protocol layer-3 backlog (docs/research/2026-08-06-error-conventions-and-host-bridge.md §3.6):
// the deny regime in Cargo.toml is live; this module still carries pre-existing
// violations of the lints below. Remove this allow as the module migrates onto
// the regime.
// Tracking: docs/error-protocol-backlog.md
#![allow(
    clippy::as_conversions,
    clippy::format_push_string,
    clippy::shadow_reuse
)]
//! Unified retrieval tool: trust-ranked search across curated memory, indexed
//! knowledge docs, workspace files, and past conversations, plus fetch modes
//! that dump or extract a specific session's messages.
//!
//! Trust order when `scope=all` (the dissolve safety net — preserves the
//! guardrail's intent that canonical facts outrank stale transcripts):
//!   curated MEMORY.md > knowledge docs > workspace files > raw sessions.
//! Each source becomes its own labelled section; the curated section always
//! appears first and can never be drowned by transcript volume.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use async_trait::async_trait;
use serde_json::{json, Value};

use super::base::{Tool, ToolContext, ToolResult};
use crate::agent::knowledge_store::{KnowledgeStore, SearchHit};

/// Cap (chars) on the merged search output so a broad query can't blow context.
const OUTPUT_MAX_CHARS: usize = 8000;
/// Default per-source result cap for the trust-ranked merge.
const DEFAULT_PER_SOURCE: usize = 3;
/// Hard ceiling on the per-source cap.
const MAX_PER_SOURCE: usize = 10;
/// Cap (chars) on a full-session dump (matches the former session_search tool).
const SESSION_DUMP_MAX_CHARS: usize = 16000;
/// Cap on messages returned by an in-session keyword search.
const MAX_IN_SESSION: usize = 20;

/// Legacy working-memory files may still be present in an existing
/// `knowledge.db`. They are no longer canonical and must not compete with
/// curated retrieval.
fn is_legacy_session_source(source_name: &str) -> bool {
    let basename = source_name
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or(source_name);
    basename.starts_with("SESSION_") && basename.ends_with(".md")
}

fn format_knowledge_hits(hits: &[SearchHit]) -> Option<String> {
    let mut output = String::new();
    for hit in hits
        .iter()
        .filter(|hit| !is_legacy_session_source(&hit.source_name))
    {
        output.push_str(&format!(
            "### {} (chunk {})\n{}\n\n",
            hit.source_name, hit.chunk_idx, hit.snippet
        ));
    }

    if output.is_empty() {
        None
    } else {
        Some(output.trim_end().to_string())
    }
}

/// Rank curated `MEMORY.md` facts against the meaningful words in a natural
/// language query. Unlike a full-line substring check, this handles calls such
/// as "what does Peppi prefer?" without missing the fact just because the whole
/// sentence is not in the file.
fn matching_memory_facts<'a>(content: &'a str, query: &str, limit: usize) -> Vec<&'a str> {
    let terms = crate::session::db::SessionDb::recall_keywords(query);
    if terms.is_empty() {
        return Vec::new();
    }

    let mut ranked: Vec<(usize, usize, &str)> = content
        .lines()
        .enumerate()
        .filter_map(|(index, line)| {
            let fact = line.trim_start().strip_prefix("- ")?.trim();
            let lower = fact.to_lowercase();
            let score = terms.iter().filter(|term| lower.contains(*term)).count();
            // Require ≥2 matching terms on multi-word queries so a single
            // common word (e.g. "design") can't surface an unrelated fact
            // (e.g. "Project Zephyr Prime" for an architecture query).
            let min_score = if terms.len() >= 3 { 2 } else { 1 };
            (score >= min_score).then_some((score, index, fact))
        })
        .collect();
    ranked.sort_by_key(|(score, index, _)| (std::cmp::Reverse(*score), *index));
    ranked
        .into_iter()
        .take(limit)
        .map(|(_, _, fact)| fact)
        .collect()
}

/// How a recall query is executed against the knowledge store.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QueryMode {
    /// BM25/FTS only — exact-match shaped queries.
    Keyword,
    /// BM25 + vector fusion (degrades to BM25 when vectors are unavailable).
    Hybrid,
}

/// Pick the query mode. An explicit `mode` param wins; otherwise multi-word
/// queries (≥2 whitespace-separated tokens) use hybrid when the semantic
/// capability is compiled in, and single-word queries stay keyword.
fn choose_query_mode(explicit: Option<&str>, query: &str, semantic_available: bool) -> QueryMode {
    match explicit {
        Some("keyword") => QueryMode::Keyword,
        Some("semantic") | Some("hybrid") => QueryMode::Hybrid,
        _ => {
            let multi_word = query.split_whitespace().nth(1).is_some();
            if multi_word && semantic_available {
                QueryMode::Hybrid
            } else {
                QueryMode::Keyword
            }
        }
    }
}

/// Truncate to `OUTPUT_MAX_CHARS` on a UTF-8 boundary with an overflow note.
fn truncate_output(output: String) -> String {
    if output.chars().count() <= OUTPUT_MAX_CHARS {
        return output;
    }
    let truncated: String = output.chars().take(OUTPUT_MAX_CHARS).collect();
    format!(
        "{}\n\n[truncated — {} total chars]",
        truncated,
        output.chars().count()
    )
}

/// Unified retrieval: trust-ranked search across all stores, or fetch a
/// specific session's messages by key / message ids.
pub struct RecallTool {
    workspace: PathBuf,
    /// SQLite sessions database. When `None`, the session fetch/search legs
    /// are unavailable (recall degrades to memory + knowledge + files).
    db_path: Option<PathBuf>,
    /// Concrete active session id, excluded from past-conversation discovery
    /// so a session never echoes the turn in progress.
    current_session_id: Option<String>,
}

impl RecallTool {
    pub fn new(workspace: &Path) -> Self {
        Self {
            workspace: workspace.to_path_buf(),
            db_path: None,
            current_session_id: None,
        }
    }

    /// Attach the sessions database (enables the sessions search + all fetch
    /// modes).
    pub fn with_db(mut self, db_path: PathBuf) -> Self {
        self.db_path = Some(db_path);
        self
    }

    /// Bind the active concrete SQLite session. Explicit session dumps remain
    /// addressable, while discovery modes cannot echo the turn in progress.
    pub fn with_current_session_id(mut self, session_id: Option<String>) -> Self {
        self.current_session_id = session_id;
        self
    }

    // -----------------------------------------------------------------
    // Search legs
    // -----------------------------------------------------------------

    /// Search curated MEMORY.md facts. Returns the matching bullets (no
    /// header) or `None` when nothing matches.
    async fn grep_memory(&self, query: &str, max_results: usize) -> Option<String> {
        let memory_file = self.workspace.join("memory").join("MEMORY.md");
        let content = tokio::fs::read_to_string(&memory_file).await.ok()?;
        let facts = matching_memory_facts(&content, query, max_results);
        if facts.is_empty() {
            return None;
        }
        Some(
            facts
                .into_iter()
                .map(|fact| format!("- {fact}"))
                .collect::<Vec<_>>()
                .join("\n"),
        )
    }

    /// Search the knowledge store (hybrid BM25 + vector when semantic feature
    /// is enabled). Returns formatted hits or `None`.
    fn knowledge_search(&self, query: &str, n: usize, mode: QueryMode) -> Option<String> {
        let store = KnowledgeStore::open_default().ok()?;
        let hits = match mode {
            QueryMode::Keyword => store.search(query, n).ok()?,
            QueryMode::Hybrid => store.hybrid_search(query, n).ok()?,
        };
        format_knowledge_hits(&hits)
    }

    /// Search workspace files via `SearchFilesTool`. Returns `None` when the
    /// tool reports no matches.
    async fn files_search(&self, query: &str, n: usize) -> Option<String> {
        let mut file_params = HashMap::new();
        file_params.insert("query".to_string(), json!(query));
        file_params.insert(
            "path".to_string(),
            json!(self.workspace.display().to_string()),
        );
        file_params.insert("limit".to_string(), json!(n));
        let out = super::SearchFilesTool.execute(file_params).await;
        let trimmed = out.trim();
        if trimmed.is_empty() || trimmed.starts_with("No matches") || trimmed.starts_with("Error") {
            return None;
        }
        Some(out)
    }

    /// Search past conversations via SessionDb FTS. Returns `None` when no
    /// session database is configured or nothing matches.
    async fn sessions_search(
        &self,
        query: &str,
        n: usize,
        params: &HashMap<String, Value>,
    ) -> Option<String> {
        let db_path = self.db_path.as_ref()?;
        let channel = params
            .get("channel")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let db = crate::session::db::SessionDb::new(db_path);
        let results = db
            .search_conversation_messages(
                query,
                n,
                channel.as_deref(),
                self.current_session_id.as_deref(),
            )
            .await;
        if results.is_empty() {
            return None;
        }
        let mut out = String::new();
        for r in &results {
            let snippet = if !r.snippet.is_empty() {
                &r.snippet
            } else {
                &r.content
            };
            out.push_str(&format!(
                "- [{}] ({}): {}\n",
                r.session_key, r.timestamp, snippet
            ));
        }
        Some(out.trim_end().to_string())
    }

    // -----------------------------------------------------------------
    // Fetch legs (migrated from the dissolved session_search tool)
    // -----------------------------------------------------------------

    /// Deterministically list recent completed conversation tails without FTS.
    async fn latest_sessions(&self, db_path: &Path, count: usize) -> String {
        let db = crate::session::db::SessionDb::new(db_path);
        let exclude_session_id = self.current_session_id.as_deref().unwrap_or("");
        let tails = db.latest_session_tails(exclude_session_id, count).await;
        if tails.is_empty() {
            return "No previous sessions found in the session database.".to_string();
        }
        let now = chrono::Utc::now();
        tails
            .iter()
            .map(|tail| {
                format!(
                    "- {}",
                    crate::agent::continuity::format_continuity_line(tail, now)
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Resolve a session key to a single session id (exact match preferred,
    /// then prefix match). Returns an error string if nothing matches.
    async fn resolve_session(&self, db_path: &Path, key: &str) -> Result<String, String> {
        let db = crate::session::db::SessionDb::new(db_path);
        let candidates = db.list_sessions(Some(key), 5).await;
        let exact = candidates
            .iter()
            .find(|s| s.session_key == key)
            .or_else(|| candidates.first());
        match exact {
            Some(meta) => Ok(meta.id.clone()),
            None => Err(format!(
                "No session found matching key '{}'. Use recall with a query to discover the key first.",
                key
            )),
        }
    }

    /// Format a session's messages as a readable transcript. Each line is
    /// prefixed with its `_db_id` so the model can later extract the exact
    /// turns via `message_ids`.
    fn format_session(messages: &[Value]) -> String {
        let mut out = String::new();
        for msg in messages {
            let id = msg.get("_db_id").and_then(|v| v.as_u64()).unwrap_or(0);
            let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("?");
            let content = msg
                .get("content")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .or_else(|| {
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

    /// Dump a full session by key (`session` param).
    async fn dump_session(&self, db_path: &Path, params: &HashMap<String, Value>) -> String {
        let key = match params.get("session").and_then(|v| v.as_str()) {
            Some(k) if !k.trim().is_empty() => k.trim().to_string(),
            _ => {
                return "Error: dumping a session requires a 'session' key (e.g. '20260717_224429_33c9c0').".to_string();
            }
        };
        let session_id = match self.resolve_session(db_path, &key).await {
            Ok(id) => id,
            Err(e) => return e,
        };
        let db = crate::session::db::SessionDb::new(db_path);
        let messages = db.get_all_messages(&session_id).await;
        if messages.is_empty() {
            return format!("Session '{}' has no stored messages.", key);
        }
        let mut body = format!("Full session '{}' ({} messages):\n\n", key, messages.len());
        body.push_str(&Self::format_session(&messages));
        if body.chars().count() > SESSION_DUMP_MAX_CHARS {
            let truncated: String = body.chars().take(SESSION_DUMP_MAX_CHARS).collect();
            format!(
                "{}\n\n... (truncated — {} total chars). Each message is prefixed with its [msg N] id. \
                 To read the relevant part in full, call recall with mode=\"in_session\", \
                 session=\"{}\", query=\"<keyword>\" — it returns the matching messages complete and \
                 relevance-ranked. Or recover one message via mode=\"extract\", session=\"{}\", \
                 message_ids=N.",
                truncated,
                body.chars().count(),
                key,
                key
            )
        } else {
            body
        }
    }

    /// Search WITHIN one session by keyword (`session` + `query`).
    async fn search_in_session(&self, db_path: &Path, params: &HashMap<String, Value>) -> String {
        let key = match params.get("session").and_then(|v| v.as_str()) {
            Some(k) if !k.trim().is_empty() => k.trim().to_string(),
            _ => {
                return "Error: mode='in_session' requires a 'session' parameter (the session key)."
                    .to_string();
            }
        };
        let query = match params.get("query").and_then(|v| v.as_str()) {
            Some(q) if !q.trim().is_empty() => q.trim().to_string(),
            _ => {
                return "Error: mode='in_session' requires a 'query' parameter (keyword to find within the session).".to_string();
            }
        };
        let session_id = match self.resolve_session(db_path, &key).await {
            Ok(id) => id,
            Err(e) => return e,
        };
        let db = crate::session::db::SessionDb::new(db_path);
        let messages = db.get_all_messages(&session_id).await;

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
        scored.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));

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

    /// Extract specific message ids from one session (`session` + `message_ids`).
    async fn extract_session(&self, db_path: &Path, params: &HashMap<String, Value>) -> String {
        let key = match params.get("session").and_then(|v| v.as_str()) {
            Some(k) if !k.trim().is_empty() => k.trim().to_string(),
            _ => {
                return "Error: extracting messages requires a 'session' parameter (the session key)."
                    .to_string();
            }
        };
        let raw = match params.get("message_ids").and_then(|v| v.as_str()) {
            Some(s) if !s.trim().is_empty() => s.trim().to_string(),
            _ => {
                return "Error: extraction requires 'message_ids' (e.g. \"5-12\" or \"5,6,7\")."
                    .to_string();
            }
        };
        let wanted: std::collections::HashSet<usize> =
            Self::parse_id_list(&raw).into_iter().collect();
        if wanted.is_empty() {
            return "Error: no valid message IDs parsed from 'message_ids'.".to_string();
        }
        let session_id = match self.resolve_session(db_path, &key).await {
            Ok(id) => id,
            Err(e) => return e,
        };
        let db = crate::session::db::SessionDb::new(db_path);
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
        body
    }

    // -----------------------------------------------------------------
    // Dispatch
    // -----------------------------------------------------------------

    /// Route a fetch-mode request to the right session leg.
    async fn execute_fetch(&self, params: &HashMap<String, Value>, mode: &str) -> String {
        let Some(db_path) = self.db_path.as_ref() else {
            return "Error: session history is not available (no session database configured)."
                .to_string();
        };
        match mode {
            "latest" => {
                let count = params
                    .get("count")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(3)
                    .clamp(1, 10) as usize;
                self.latest_sessions(db_path, count).await
            }
            "session" => self.dump_session(db_path, params).await,
            "in_session" => self.search_in_session(db_path, params).await,
            "extract" => self.extract_session(db_path, params).await,
            _ => "Error: unsupported fetch mode.".to_string(),
        }
    }

    /// Trust-ranked search across the stores selected by `scope`.
    async fn execute_search(&self, params: &HashMap<String, Value>, query: &str) -> String {
        // A query with no content keywords cannot match anything useful across
        // any store — fail fast with a corrective message instead of running
        // four empty searches.
        if crate::session::db::SessionDb::recall_keywords(query).is_empty() {
            return "No specific memory topic found in the query. Ask about a person, preference, decision, project, or other concrete detail."
                .to_string();
        }

        let scope = params
            .get("scope")
            .and_then(|v| v.as_str())
            .unwrap_or("all");
        // `n` is the documented per-source cap; accept the legacy `limit` alias
        // so old `session_search`/`search_context` calls (now aliased to recall)
        // keep working without a param rewrite.
        let n = params
            .get("n")
            .and_then(|v| v.as_u64())
            .or_else(|| params.get("limit").and_then(|v| v.as_u64()))
            .map(|v| v.clamp(1, MAX_PER_SOURCE as u64) as usize)
            .unwrap_or(DEFAULT_PER_SOURCE);
        let mode_str = params
            .get("mode")
            .and_then(|v| v.as_str())
            .filter(|m| matches!(*m, "auto" | "keyword" | "semantic"));
        let qmode = choose_query_mode(mode_str, query, cfg!(feature = "semantic"));

        let want_memory = matches!(scope, "all" | "memory");
        // Files excluded from default `all` — the leg searches the agent's own
        // workspace (~/.nanobot/workspace: memory/audit/skills), which is noise
        // for conceptual queries and echoes past search recordings. Opt in with
        // scope="files" explicitly. See recall tool description.
        let want_files = scope == "files";
        let want_sessions = matches!(scope, "all" | "sessions");

        // Trust order: curated memory > knowledge docs > workspace files > sessions.
        let mut sections: Vec<(&'static str, String)> = Vec::new();
        if want_memory {
            if let Some(body) = self.grep_memory(query, n).await {
                sections.push(("Curated memory", body));
            }
            if let Some(body) = self.knowledge_search(query, n, qmode) {
                sections.push(("Knowledge docs", body));
            }
        }
        if want_files {
            if let Some(body) = self.files_search(query, n).await {
                sections.push(("Workspace files", body));
            }
        }
        if want_sessions {
            if let Some(body) = self.sessions_search(query, n, params).await {
                sections.push(("Past conversations", body));
            }
        }

        if sections.is_empty() {
            return format!(
                "No matches found for '{}' across memory, files, or past conversations.",
                query
            );
        }

        let mut output = String::new();
        for (header, body) in &sections {
            if !output.is_empty() {
                output.push_str("\n\n");
            }
            output.push_str(&format!("## {header}\n{body}"));
        }
        truncate_output(output)
    }
}

#[async_trait]
impl Tool for RecallTool {
    fn name(&self) -> &str {
        "recall"
    }

    fn description(&self) -> &str {
        "Unified retrieval. SEARCH curated memory, indexed knowledge, and past conversations \
         (trust-ranked: canonical facts first) — workspace files are opt-in via scope=\"files\" \
         (they search the agent workspace, rarely useful). OR FETCH a specific session. \
         Search: {\"query\":\"...\",\"scope\":\"all|memory|files|sessions\"} (default 'all' = \
         memory + knowledge + sessions, NO files). Fetch: {\"session\":\"KEY\"} to dump a \
         transcript, {\"session\":\"KEY\",\"message_ids\":\"5-12\"} to extract turns, or \
         {\"mode\":\"latest\"} to list recent sessions."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (search path). Ignored when fetching by session/message_ids."
                },
                "scope": {
                    "type": "string",
                    "enum": ["all", "memory", "files", "sessions"],
                    "description": "Sources to search. Default 'all' = memory + knowledge + sessions (files excluded — workspace internals are noise; use 'files' to opt in). 'memory' = curated MEMORY.md + indexed knowledge docs."
                },
                "n": {
                    "type": "integer",
                    "description": "Per-source result cap (search path). Default 3, max 10."
                },
                "mode": {
                    "type": "string",
                    "enum": ["auto", "keyword", "semantic", "latest", "session", "in_session", "extract"],
                    "description": "Search tuning (auto/keyword/semantic) OR fetch operation (latest=session list, session=full dump, in_session=keyword within one session, extract=pull ids)."
                },
                "session": {
                    "type": "string",
                    "description": "Session key (e.g. '20260717_224429_33c9c0'). Triggers fetch mode (dump) when present without a query."
                },
                "message_ids": {
                    "type": "string",
                    "description": "Message ids to extract from a session, e.g. \"5-12\" or \"5,6,7\". Requires 'session'."
                },
                "count": {
                    "type": "integer",
                    "description": "Number of recent sessions for mode='latest'. Default 3, max 10."
                },
                "channel": {
                    "type": "string",
                    "description": "Filter sessions search to a channel prefix (e.g. 'cli:', 'telegram:'). Optional."
                }
            },
            "required": []
        })
    }

    async fn execute(&self, params: HashMap<String, Value>) -> String {
        let mode = params.get("mode").and_then(|v| v.as_str()).unwrap_or("");
        let session = params
            .get("session")
            .and_then(|v| v.as_str())
            .filter(|s| !s.trim().is_empty());
        let message_ids = params
            .get("message_ids")
            .and_then(|v| v.as_str())
            .filter(|s| !s.trim().is_empty());
        let query = params
            .get("query")
            .and_then(|v| v.as_str())
            .filter(|s| !s.trim().is_empty());

        // Fetch path: an explicit fetch mode, or a session/message_ids key
        // present (param-presence dispatch — no separate tool needed).
        let fetch_mode = match mode {
            "latest" | "session" | "in_session" | "extract" => Some(mode),
            _ if message_ids.is_some() => Some("extract"),
            _ if session.is_some() => Some("session"),
            _ => None,
        };
        if let Some(fm) = fetch_mode {
            return self.execute_fetch(&params, fm).await;
        }

        // Search path.
        if let Some(q) = query {
            return self.execute_search(&params, q).await;
        }

        "Error: recall needs one of: 'query' (to search), 'session' or 'message_ids' (to fetch), or mode=\"latest\".".to_string()
    }

    /// Empty-arg is self-correcting: produce a structural
    /// [`crate::errors::ToolError::MissingArg`] (model-fixable) whose render
    /// carries a worked call shape instead of looping on the same bare call.
    /// Everything else funnels through the legacy string path unchanged.
    async fn execute_typed(&self, params: HashMap<String, Value>, ctx: &ToolContext) -> ToolResult {
        let has_query = params
            .get("query")
            .and_then(|v| v.as_str())
            .is_some_and(|s| !s.trim().is_empty());
        let has_session = params
            .get("session")
            .and_then(|v| v.as_str())
            .is_some_and(|s| !s.trim().is_empty());
        let has_message_ids = params
            .get("message_ids")
            .and_then(|v| v.as_str())
            .is_some_and(|s| !s.trim().is_empty());
        let mode = params.get("mode").and_then(|v| v.as_str()).unwrap_or("");
        let is_fetch = matches!(mode, "latest" | "session" | "in_session" | "extract")
            || has_session
            || has_message_ids;

        if !is_fetch && !has_query {
            return Err(crate::errors::ToolError::MissingArg {
                param: "query".to_string(),
                example: r#"recall({"query":"..."})"#.to_string(),
            });
        }
        // Funnel through the legacy string path (error protocol §2.2).
        // NB: call the shared helper, NOT `Tool::execute_typed(self, ...)` —
        // under #[async_trait] that qualified call re-dispatches to this
        // override and recurses until stack overflow (see funnel_legacy docs).
        let out = self.execute_with_context(params, ctx).await;
        crate::agent::tools::base::funnel_legacy(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_tool() -> (TempDir, RecallTool) {
        let tmp = TempDir::new().unwrap();
        std::fs::create_dir_all(tmp.path().join("memory")).unwrap();
        let tool = RecallTool::new(tmp.path());
        (tmp, tool)
    }

    fn make_tool_with_db() -> (TempDir, RecallTool) {
        let tmp = TempDir::new().unwrap();
        std::fs::create_dir_all(tmp.path().join("memory")).unwrap();
        let tool = RecallTool::new(tmp.path()).with_db(tmp.path().join("sessions.db"));
        (tmp, tool)
    }

    async fn seed_session(tmp: &TempDir, key: &str, msgs: &[Value]) {
        use crate::session::db::SessionDb;
        let db_path = tmp.path().join("sessions.db");
        let db = SessionDb::new(&db_path);
        let session = db.create_session(key).await;
        db.add_messages(&session.id, msgs).await;
    }

    #[test]
    fn test_recall_tool_name() {
        let (_tmp, tool) = make_tool();
        assert_eq!(tool.name(), "recall");
    }

    #[test]
    fn test_choose_query_mode_explicit_wins() {
        assert_eq!(
            choose_query_mode(Some("keyword"), "several words here", true),
            QueryMode::Keyword
        );
        assert_eq!(
            choose_query_mode(Some("semantic"), "word", false),
            QueryMode::Hybrid
        );
    }

    #[test]
    fn test_choose_query_mode_heuristic_word_count() {
        assert_eq!(choose_query_mode(None, "rust", true), QueryMode::Keyword);
        assert_eq!(
            choose_query_mode(None, "how compaction was configured", true),
            QueryMode::Hybrid
        );
        assert_eq!(
            choose_query_mode(None, "how compaction was configured", false),
            QueryMode::Keyword
        );
        assert_eq!(
            choose_query_mode(Some("auto"), "one two three", true),
            QueryMode::Hybrid
        );
        assert_eq!(
            choose_query_mode(Some("auto"), "one", true),
            QueryMode::Keyword
        );
    }

    #[test]
    fn test_recall_description_mentions_mode() {
        let (_tmp, tool) = make_tool();
        assert!(tool.description().contains("mode"));
    }

    #[test]
    fn test_recall_tool_parameters_schema_supports_fetch_and_search() {
        let (_tmp, tool) = make_tool();
        let params = tool.parameters();
        assert_eq!(params["type"], "object");
        assert!(params["properties"]["query"].is_object());
        assert!(params["properties"]["scope"].is_object());
        assert!(params["properties"]["n"].is_object());
        assert!(params["properties"]["session"].is_object());
        assert!(params["properties"]["message_ids"].is_object());
        assert!(params["properties"]["mode"].is_object());
        let mode_enum = params["properties"]["mode"]["enum"].as_array().unwrap();
        // Fetch modes (absorbed from session_search) now valid here.
        assert!(mode_enum.contains(&json!("latest")));
        assert!(mode_enum.contains(&json!("session")));
        assert!(mode_enum.contains(&json!("extract")));
        // Search tuning modes still present.
        assert!(mode_enum.contains(&json!("keyword")));
        assert!(mode_enum.contains(&json!("semantic")));
        let scope_enum = params["properties"]["scope"]["enum"].as_array().unwrap();
        assert!(scope_enum.contains(&json!("all")));
        assert!(scope_enum.contains(&json!("memory")));
        assert!(scope_enum.contains(&json!("sessions")));
    }

    #[tokio::test]
    async fn test_recall_empty_query_returns_error() {
        let (_tmp, tool) = make_tool();
        let mut params = HashMap::new();
        params.insert("query".to_string(), json!(""));
        let result = tool.execute(params).await;
        assert!(result.contains("Error"));
    }

    /// Empty-arg must produce a structural MissingArg naming the entry params.
    #[tokio::test]
    async fn test_recall_empty_arg_returns_structural_missing_arg() {
        let (_tmp, tool) = make_tool();
        let res = tool.execute_with_result(HashMap::new()).await;
        assert!(!res.ok());
        assert!(matches!(
            res.error_kind(),
            Some(crate::errors::ToolErrorKind::MissingArg { ref param, .. }) if param == "query"
        ));
        // Canonical MissingArg render names the required param and carries
        // the worked call shape (error protocol Phase 2 canonicalization).
        assert!(
            res.data().contains("'query' parameter is required"),
            "{}",
            res.data()
        );
        assert!(res.data().contains("call as recall("), "{}", res.data());
    }

    #[tokio::test]
    async fn test_recall_grep_finds_memory_md() {
        let (tmp, tool) = make_tool();
        std::fs::write(
            tmp.path().join("memory").join("MEMORY.md"),
            "- User prefers dark mode\n- Favorite language is Rust\n- Lives in Helsinki",
        )
        .unwrap();

        let mut params = HashMap::new();
        params.insert("query".to_string(), json!("Rust"));
        params.insert("mode".to_string(), json!("keyword"));
        let result = tool.execute(params).await;
        assert!(result.contains("Rust"), "Should find Rust in MEMORY.md");
    }

    #[tokio::test]
    async fn test_recall_natural_query_finds_curated_fact() {
        let (tmp, tool) = make_tool();
        std::fs::write(
            tmp.path().join("memory").join("MEMORY.md"),
            "- Peppi prefers concise answers\n- Favorite language is Rust\n",
        )
        .unwrap();

        let result = tool
            .grep_memory("what does Peppi prefer", 5)
            .await
            .unwrap_or_default();
        assert!(
            result.contains("Peppi prefers concise answers"),
            "got: {result}"
        );
    }

    #[tokio::test]
    async fn test_recall_grep_returns_none_when_no_match() {
        let (tmp, tool) = make_tool();
        std::fs::write(
            tmp.path().join("memory").join("MEMORY.md"),
            "- unrelated fact\n",
        )
        .unwrap();
        let result = tool.grep_memory("nonexistent_xyz_123_qqq", 5).await;
        assert!(result.is_none(), "no match must be None: {:?}", result);
    }

    #[tokio::test]
    async fn test_recall_does_not_index_raw_session_messages_into_memory() {
        let (tmp, tool) = make_tool_with_db();
        let db = crate::session::db::SessionDb::new(&tmp.path().join("sessions.db"));
        let session = db.create_session("cli:test").await;
        db.add_messages(
            &session.id,
            &[json!({
                "role": "user",
                "content": "Discussed async Rust patterns."
            })],
        )
        .await;
        // The memory leg reads only MEMORY.md, never raw transcripts.
        let result = tool.grep_memory("async", 10).await;
        assert!(
            result.is_none(),
            "recall's memory leg owns curated memory, not raw transcripts: {:?}",
            result
        );
    }

    #[tokio::test]
    async fn test_recall_rejects_topicless_memory_query() {
        let (_tmp, tool) = make_tool();
        let result = tool
            .execute(HashMap::from([(
                "query".to_string(),
                json!("what do you remember about me"),
            )]))
            .await;
        assert!(result.contains("No specific memory topic"), "got: {result}");
    }

    #[tokio::test]
    async fn test_recall_utf8_truncation_no_panic() {
        // The truncation path must never slice a multi-byte CJK boundary.
        // (The original byte-slice `&output[..8000]` panicked on CJK; the
        // char-based `truncate_output` is UTF-8 safe.)
        let cjk = "日本語テスト\n".repeat(2000); // ~12k chars of CJK
        let out = truncate_output(cjk.clone());
        assert!(
            out.contains("日本語"),
            "truncated output must preserve the CJK head: {}",
            &out[..out.len().min(200)]
        );
        assert!(out.contains("[truncated"), "must mark truncation: {out}");
    }

    #[tokio::test]
    async fn test_recall_missing_query_param_returns_error() {
        let (_tmp, tool) = make_tool();
        let result = tool.execute(HashMap::new()).await;
        assert!(
            result.contains("Error"),
            "missing curated-memory query must be rejected, got: {result}"
        );
    }

    #[test]
    fn test_knowledge_search_on_empty_store() {
        let tool = RecallTool::new(Path::new("/tmp/nonexistent"));
        let result = tool.knowledge_search("test query", 5, QueryMode::Hybrid);
        if let Some(ref text) = result {
            assert!(!text.is_empty());
        }
    }

    #[test]
    fn test_knowledge_results_exclude_legacy_session_markdown() {
        let hits = vec![
            SearchHit {
                source_name: "/tmp/memory/sessions/SESSION_cli_123.md".to_string(),
                chunk_idx: 0,
                snippet: "stale session summary".to_string(),
                rank: -2.0,
            },
            SearchHit {
                source_name: r"C:\memory\sessions\SESSION_cli_456.md".to_string(),
                chunk_idx: 0,
                snippet: "stale Windows session summary".to_string(),
                rank: -1.5,
            },
            SearchHit {
                source_name: "project-notes.md".to_string(),
                chunk_idx: 2,
                snippet: "current durable knowledge".to_string(),
                rank: -1.0,
            },
        ];

        let output = format_knowledge_hits(&hits).expect("non-session hit should remain");
        assert!(!output.contains("SESSION_cli_123.md"));
        assert!(!output.contains("SESSION_cli_456.md"));
        assert!(!output.contains("stale session summary"));
        assert!(output.contains("project-notes.md"));
        assert!(output.contains("current durable knowledge"));
    }

    // -----------------------------------------------------------------
    // scope + trust-ranking (the dissolve safety net)
    // -----------------------------------------------------------------

    #[tokio::test]
    async fn test_scope_memory_excludes_sessions() {
        let (tmp, tool) = make_tool_with_db();
        std::fs::write(
            tmp.path().join("memory").join("MEMORY.md"),
            "- The canonical quasarledger note\n",
        )
        .unwrap();
        seed_session(
            &tmp,
            "cli:past",
            &[json!({"role": "user", "content": "quasarledger session transcript"})],
        )
        .await;

        let result = tool
            .execute(HashMap::from([
                ("query".to_string(), json!("quasarledger")),
                ("scope".to_string(), json!("memory")),
            ]))
            .await;
        assert!(
            result.contains("## Curated memory") && result.contains("quasarledger note"),
            "memory scope must surface curated fact: {result}"
        );
        assert!(
            !result.contains("Past conversations"),
            "scope=memory must NOT search sessions: {result}"
        );
    }

    /// Trust-ranking invariant: when scope=all matches both MEMORY.md and a raw
    /// session on the same term, the "## Curated memory" section MUST precede
    /// "## Past conversations", and the canonical fact must be present. Stale
    /// transcripts can never drown canonical facts.
    #[tokio::test]
    async fn test_trust_ranking_curated_memory_precedes_sessions() {
        let (tmp, tool) = make_tool_with_db();
        std::fs::write(
            tmp.path().join("memory").join("MEMORY.md"),
            "- Peppi's canonical preference is Rust over Python\n",
        )
        .unwrap();
        seed_session(
            &tmp,
            "cli:past",
            &[json!({
                "role": "user",
                "content": "we talked about rust vs python earlier"
            })],
        )
        .await;

        let result = tool
            .execute(HashMap::from([("query".to_string(), json!("rust"))]))
            .await;
        let mem_pos = result.find("## Curated memory");
        let sess_pos = result.find("## Past conversations");
        assert!(
            mem_pos.is_some(),
            "curated section must be present: {result}"
        );
        assert!(
            sess_pos.is_some(),
            "sessions section must be present (both matched): {result}"
        );
        assert!(
            mem_pos < sess_pos,
            "curated memory MUST precede past conversations: {result}"
        );
        assert!(
            result.contains("canonical preference is Rust"),
            "canonical fact must surface: {result}"
        );
    }

    // -----------------------------------------------------------------
    // Fetch modes (absorbed from session_search)
    // -----------------------------------------------------------------

    #[tokio::test]
    async fn test_mode_latest_lists_recent_sessions_excluding_current() {
        let (tmp, tool) = make_tool_with_db();
        let previous = crate::session::db::SessionDb::new(&tmp.path().join("sessions.db"))
            .create_session("cli:previous")
            .await;
        crate::session::db::SessionDb::new(&tmp.path().join("sessions.db"))
            .add_messages(
                &previous.id,
                &[json!({"role": "user", "content": "previous canonical question"})],
            )
            .await;
        let current = crate::session::db::SessionDb::new(&tmp.path().join("sessions.db"))
            .create_session("cli:current")
            .await;
        crate::session::db::SessionDb::new(&tmp.path().join("sessions.db"))
            .add_messages(
                &current.id,
                &[json!({"role": "user", "content": "active question"})],
            )
            .await;
        let tool = tool.with_current_session_id(Some(current.id));

        let result = tool
            .execute(HashMap::from([
                ("mode".to_string(), json!("latest")),
                ("count".to_string(), json!(3)),
            ]))
            .await;
        assert!(result.contains("cli:previous"), "got: {result}");
        assert!(!result.contains("cli:current"), "got: {result}");
    }

    #[tokio::test]
    async fn test_session_param_dumps_full_transcript() {
        let (tmp, tool) = make_tool_with_db();
        seed_session(
            &tmp,
            "20260717_224429_33c9c0",
            &[
                json!({"role": "user", "content": "Tell me the Diary of Two Threads"}),
                json!({"role": "assistant", "content": "Entry 1: the two voices converge"}),
            ],
        )
        .await;
        let result = tool
            .execute(HashMap::from([(
                "session".to_string(),
                json!("20260717_224429_33c9c0"),
            )]))
            .await;
        assert!(result.contains("Full session"), "got: {result}");
        assert!(result.contains("Diary of Two Threads"), "got: {result}");
        assert!(result.contains("two voices converge"), "got: {result}");
    }

    #[tokio::test]
    async fn test_message_ids_extract_turns() {
        let (tmp, tool) = make_tool_with_db();
        seed_session(
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
        let result = tool
            .execute(HashMap::from([
                ("session".to_string(), json!("20260717_224429_33c9c0")),
                ("message_ids".to_string(), json!("2-3")),
            ]))
            .await;
        assert!(result.contains("Extracted 2 message(s)"), "got: {result}");
        assert!(
            result.contains("beta two") && result.contains("gamma three"),
            "got: {result}"
        );
        assert!(
            !result.contains("alpha one") && !result.contains("delta four"),
            "got: {result}"
        );
    }

    #[tokio::test]
    async fn test_in_session_keyword_search_returns_full_ranked() {
        let (tmp, tool) = make_tool_with_db();
        let story = "Diary of Two Threads Entry 1 nano32 March 14 2036. ".repeat(200);
        seed_session(
            &tmp,
            "20260717_224429_33c9c0",
            &[
                json!({"role": "user", "content": "tell me the diary of two threads story"}),
                json!({"role": "assistant", "content": story}),
            ],
        )
        .await;
        let out = tool
            .execute(HashMap::from([
                ("mode".to_string(), json!("in_session")),
                ("session".to_string(), json!("20260717_224429_33c9c0")),
                ("query".to_string(), json!("Diary of Two Threads story")),
            ]))
            .await;
        assert!(out.contains("returned in full"), "got: {out}");
        assert!(
            out.contains("Diary of Two Threads Entry 1 nano32"),
            "got: {out}"
        );
        assert!(
            !out.contains("... (truncated"),
            "in_session must not truncate: {out}"
        );
    }

    #[tokio::test]
    async fn test_fetch_without_db_returns_clear_error() {
        let (_tmp, tool) = make_tool();
        let result = tool
            .execute(HashMap::from([("mode".to_string(), json!("latest"))]))
            .await;
        assert!(
            result.contains("not available") || result.contains("Error"),
            "fetch without db_path must error clearly: {result}"
        );
    }

    #[tokio::test]
    async fn test_recall_full_typed_chain_with_query_does_not_recurse() {
        let (tmp, tool) = make_tool();
        std::fs::write(
            tmp.path().join("memory").join("MEMORY.md"),
            "- User prefers dark mode\n- Favorite language is Rust\n",
        )
        .unwrap();
        let mut params = HashMap::new();
        params.insert("query".to_string(), json!("Rust"));
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let token = tokio_util::sync::CancellationToken::new();
        let ctx = ToolContext::new(None, tx, token, "test-call");
        let res = tool.execute_with_result_and_context(params, &ctx).await;
        assert!(
            res.ok(),
            "should succeed via typed chain: {:?}",
            res.error()
        );
        assert!(
            res.data().contains("Rust"),
            "should find Rust: {}",
            res.data()
        );
    }
}
