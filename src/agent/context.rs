//! Context builder for assembling agent prompts.
//!
//! Assembles bootstrap files, memory, skills, and conversation history into
//! a coherent prompt for the LLM.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use base64::Engine;
use chrono::{DateTime, Duration as ChronoDuration, Local};
use serde_json::{json, Value};

use crate::agent::learning::LearningStore;
use crate::agent::memory::MemoryStore;
use crate::agent::observer::ObservationStore;
use crate::agent::semantic::SemanticIndex;
use crate::agent::skills::SkillsLoader;
use crate::agent::token_budget::TokenBudget;

/// Well-known files that are loaded from the workspace root when present.
const BOOTSTRAP_FILES: &[&str] = &["AGENTS.md", "SOUL.md", "USER.md", "TOOLS.md", "IDENTITY.md"];

/// Builds the context (system prompt + messages) for the agent.
pub struct ContextBuilder {
    pub workspace: PathBuf,
    pub memory: MemoryStore,
    pub skills: SkillsLoader,
    pub model_name: String,
    /// Max tokens for bootstrap instruction files in the system prompt.
    pub bootstrap_budget: usize,
    /// Max tokens for long-term memory (`MEMORY.md`) in the system prompt.
    pub long_term_memory_budget: usize,
    /// Max tokens for today's notes in the system prompt.
    pub today_notes_budget: usize,
    /// Max tokens for observations in the system prompt (default: 2000).
    pub observation_budget: usize,
    /// Max tokens for learning context in the system prompt.
    pub learning_budget: usize,
    /// Optional semantic index for BM25-based memory retrieval.
    pub semantic_index: Option<Arc<SemanticIndex>>,
    /// Max tokens for semantic retrieval context.
    pub semantic_budget: usize,
    /// Optional environment summary for the system prompt.
    pub environment_summary: Option<String>,
    /// Timestamp of the last user interaction (for time awareness).
    pub last_interaction: Option<DateTime<Local>>,
    /// Git change summary detected at session start.
    pub git_changes: Option<String>,
}

impl ContextBuilder {
    /// Create a new context builder for the given workspace.
    pub fn new(workspace: &Path) -> Self {
        Self {
            workspace: workspace.to_path_buf(),
            memory: MemoryStore::new(workspace),
            skills: SkillsLoader::new(workspace, None),
            model_name: String::new(),
            bootstrap_budget: 3000,
            long_term_memory_budget: 2000,
            today_notes_budget: 1200,
            observation_budget: 2000,
            learning_budget: 800,
            semantic_index: None,
            semantic_budget: 1500,
            environment_summary: None,
            last_interaction: None,
            git_changes: None,
        }
    }

    // ------------------------------------------------------------------
    // Public API
    // ------------------------------------------------------------------

    /// Build the system prompt from bootstrap files, memory, and skills.
    pub fn build_system_prompt(&self, skill_names: Option<&[String]>) -> String {
        let mut parts: Vec<String> = Vec::new();

        // Core identity.
        parts.push(self._get_identity());

        // Bootstrap files.
        let bootstrap =
            Self::_truncate_to_budget(&self._load_bootstrap_files(), self.bootstrap_budget);
        if !bootstrap.is_empty() {
            parts.push(bootstrap);
        }

        // Memory context.
        let mut memory_parts: Vec<String> = Vec::new();
        let long_term =
            Self::_truncate_to_budget(&self.memory.read_long_term(), self.long_term_memory_budget);
        if !long_term.is_empty() {
            memory_parts.push(format!("## Long-term Memory\n{}", long_term));
        }
        let today_notes =
            Self::_truncate_to_budget(&self.memory.read_today(), self.today_notes_budget);
        if !today_notes.is_empty() {
            memory_parts.push(format!("## Today's Notes\n{}", today_notes));
        }
        if !memory_parts.is_empty() {
            parts.push(format!("# Memory\n\n{}", memory_parts.join("\n\n")));
        }

        // Observations from past conversations.
        let observer = ObservationStore::new(&self.workspace);
        let obs_context = observer.get_context(self.observation_budget);
        if !obs_context.is_empty() {
            parts.push(format!(
                "# Observations from Past Conversations\n\n{}",
                obs_context
            ));
        }

        // Learning context (tool outcome patterns).
        let learning = LearningStore::new(&self.workspace);
        let learning_context =
            Self::_truncate_to_budget(&learning.get_learning_context(), self.learning_budget);
        if !learning_context.is_empty() {
            parts.push(format!("# Recent Tool Patterns\n\n{}", learning_context));
        }

        // Environment capabilities.
        if let Some(ref env_summary) = self.environment_summary {
            parts.push(format!("# Environment\n\n{}", env_summary));
        }

        // Skills -- progressive loading:
        // 1. Always-loaded skills: full content included directly.
        let always_skills = self.skills.get_always_skills();
        if !always_skills.is_empty() {
            let always_content = self.skills.load_skills_for_context(&always_skills);
            if !always_content.is_empty() {
                parts.push(format!("# Active Skills\n\n{}", always_content));
            }
        }

        // 2. Available skills: summary only (agent can read_file for details).
        let skills_summary = self.skills.build_skills_summary();
        if !skills_summary.is_empty() {
            parts.push(format!(
                "# Skills\n\n\
                 The following skills extend your capabilities. \
                 To use a skill, read its SKILL.md file using the read_file tool.\n\
                 Skills with available=\"false\" need dependencies installed first \
                 - you can try installing them with apt/brew.\n\n\
                 {}",
                skills_summary
            ));
        }

        // 3. Explicitly requested skills.
        if let Some(names) = skill_names {
            if !names.is_empty() {
                let requested = self.skills.load_skills_for_context(names);
                if !requested.is_empty() {
                    parts.push(format!("# Requested Skills\n\n{}", requested));
                }
            }
        }

        parts.join("\n\n---\n\n")
    }

    /// Build the complete message list for an LLM call.
    pub fn build_messages(
        &self,
        history: &[Value],
        current_message: &str,
        skill_names: Option<&[String]>,
        media: Option<&[String]>,
        channel: Option<&str>,
        chat_id: Option<&str>,
        is_voice_message: bool,
        detected_language: Option<&str>,
    ) -> Vec<Value> {
        let mut messages: Vec<Value> = Vec::new();

        // System prompt.
        let mut system_prompt = self.build_system_prompt(skill_names);
        if let (Some(ch), Some(cid)) = (channel, chat_id) {
            system_prompt.push_str(&format!(
                "\n\n## Current Session\nChannel: {}\nChat ID: {}",
                ch, cid
            ));
            if ch == "voice" || is_voice_message {
                system_prompt.push_str(concat!(
                    "\n\n## Voice Mode (IMPORTANT)\n",
                    "The user is speaking via microphone and your response will be read aloud by TTS. ",
                    "STRICT RULES:\n",
                    "- Keep responses to 1-3 sentences. Be concise.\n",
                    "- Use plain spoken language only.\n",
                    "- NEVER use emoji, emoticons, or unicode symbols.\n",
                    "- NEVER use markdown: no **, no ##, no ```, no bullet points, no numbered lists.\n",
                    "- NEVER output code blocks or technical formatting.\n",
                    "- If asked a complex question, give a brief spoken answer, not a written essay.",
                ));
                if let Some(lang) = detected_language {
                    if lang == "en" {
                        system_prompt.push_str(
                            "\n- The user is speaking in English. You MUST respond in English.",
                        );
                    } else {
                        system_prompt.push_str(&format!(
                            "\n- The user is speaking in {}. You MUST respond in the same language.",
                            lang_code_to_name(lang)
                        ));
                    }
                }
            }
        }
        // Semantic retrieval: inject relevant memories based on the user's message.
        if let Some(ref index) = self.semantic_index {
            let semantic_ctx = index.get_relevant_context(current_message, self.semantic_budget);
            if !semantic_ctx.is_empty() {
                system_prompt.push_str(&format!(
                    "\n\n---\n\n# Relevant Memories\n\n{}",
                    semantic_ctx
                ));
            }
        }

        messages.push(json!({"role": "system", "content": system_prompt}));

        // History.
        messages.extend(history.iter().cloned());

        // Current user message (with optional image attachments).
        let user_content = Self::_build_user_content(current_message, media);
        messages.push(json!({"role": "user", "content": user_content}));

        messages
    }

    /// Add a tool result to the message list and return the updated list.
    pub fn add_tool_result(
        messages: &mut Vec<Value>,
        tool_call_id: &str,
        tool_name: &str,
        result: &str,
    ) {
        messages.push(json!({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": result,
        }));
    }

    /// Add an assistant message (possibly with tool calls) to the message list.
    pub fn add_assistant_message(
        messages: &mut Vec<Value>,
        content: Option<&str>,
        tool_calls: Option<&[Value]>,
    ) {
        let mut msg = json!({
            "role": "assistant",
            "content": content.unwrap_or(""),
        });

        if let Some(tc) = tool_calls {
            if !tc.is_empty() {
                msg["tool_calls"] = Value::Array(tc.to_vec());
            }
        }

        messages.push(msg);
    }

    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    /// Core identity section including current time, time awareness, and workspace info.
    fn _get_identity(&self) -> String {
        let now = Local::now();
        let now_str = now.format("%Y-%m-%d %H:%M (%A)").to_string();
        let workspace_path = self
            .workspace
            .canonicalize()
            .unwrap_or_else(|_| self.workspace.clone())
            .to_string_lossy()
            .to_string();

        let model_section = if self.model_name.is_empty() {
            String::new()
        } else if let Some(gguf_name) = self.model_name.strip_prefix("local:") {
            format!(
                "\n\n## Model\nYou are running locally via llama.cpp. \
                 Your model file: {}. You are NOT Claude or any cloud AI. \
                 Respond as nanobot powered by this local model.",
                gguf_name
            )
        } else {
            format!("\n\n## Model\nYou are powered by: {}", self.model_name)
        };

        // Time awareness: show gap since last interaction with behavioral hint.
        let time_awareness = if let Some(last) = self.last_interaction {
            let gap = now.signed_duration_since(last);
            let gap_str = Self::format_time_gap(gap);
            let hint = Self::time_gap_hint(gap);
            format!("\nLast session: {}. {}", gap_str, hint)
        } else {
            String::new()
        };

        // Git changes detected since last session.
        let git_section = match self.git_changes {
            Some(ref changes) if !changes.is_empty() => {
                format!("\n\n## External Changes\n{}", changes)
            }
            _ => String::new(),
        };

        format!(
            "# nanobot\n\n\
             ## Current Time\n{now_str}{time_awareness}{model_section}\n\n\
             ## Workspace\n{workspace_path}{git_section}"
        )
    }

    /// Format a duration into a human-readable gap string.
    fn format_time_gap(gap: ChronoDuration) -> String {
        let minutes = gap.num_minutes();
        if minutes < 2 {
            "Just now".to_string()
        } else if minutes < 60 {
            format!("{} min ago", minutes)
        } else if gap.num_hours() < 24 {
            let hours = gap.num_hours();
            if hours == 1 {
                "1 hour ago".to_string()
            } else {
                format!("{} hours ago", hours)
            }
        } else if gap.num_days() == 1 {
            "Yesterday".to_string()
        } else {
            format!("{} days ago", gap.num_days())
        }
    }

    /// Behavioral hint based on the time gap.
    fn time_gap_hint(gap: ChronoDuration) -> &'static str {
        let hours = gap.num_hours();
        if hours < 1 {
            "Continue naturally from where you left off."
        } else if hours < 8 {
            "The user is returning after a short break."
        } else {
            "The user is starting a new session — briefly re-orient if context changed."
        }
    }

    /// Load all bootstrap files from workspace.
    fn _load_bootstrap_files(&self) -> String {
        let mut parts: Vec<String> = Vec::new();

        for filename in BOOTSTRAP_FILES {
            let file_path = self.workspace.join(filename);
            if file_path.exists() {
                if let Ok(content) = fs::read_to_string(&file_path) {
                    parts.push(format!("## {}\n\n{}", filename, content));
                }
            }
        }

        parts.join("\n\n")
    }

    /// Build user message content with optional base64-encoded images.
    ///
    /// If media contains image files, returns a JSON array of content parts.
    /// Otherwise returns a plain string value.
    fn _build_user_content(text: &str, media: Option<&[String]>) -> Value {
        let media = match media {
            Some(m) if !m.is_empty() => m,
            _ => return Value::String(text.to_string()),
        };

        let mut images: Vec<Value> = Vec::new();

        for path_str in media {
            let path = Path::new(path_str);
            if !path.is_file() {
                continue;
            }
            let mime = _guess_mime(path_str);
            if !mime.starts_with("image/") {
                continue;
            }
            if let Ok(bytes) = fs::read(path) {
                let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
                images.push(json!({
                    "type": "image_url",
                    "image_url": {
                        "url": format!("data:{};base64,{}", mime, b64),
                    }
                }));
            }
        }

        if images.is_empty() {
            return Value::String(text.to_string());
        }

        // Append text part after images.
        images.push(json!({"type": "text", "text": text}));
        Value::Array(images)
    }

    /// Truncate a section to fit an approximate token budget.
    fn _truncate_to_budget(text: &str, max_tokens: usize) -> String {
        if text.is_empty() || max_tokens == 0 {
            return String::new();
        }

        if TokenBudget::estimate_str_tokens(text) <= max_tokens {
            return text.to_string();
        }

        let max_chars = max_tokens.saturating_mul(4);
        let marker = "\n\n[truncated to fit token budget]";
        let keep_chars = max_chars.saturating_sub(marker.len());
        let mut out: String = text.chars().take(keep_chars).collect();
        out.push_str(marker);
        out
    }
}

/// Map ISO 639-1 language code to a human-readable name for LLM instruction.
fn lang_code_to_name(code: &str) -> &'static str {
    match code {
        "es" => "Spanish",
        "fr" => "French",
        "hi" => "Hindi",
        "it" => "Italian",
        "ja" => "Japanese",
        "pt" => "Portuguese",
        "zh" => "Chinese",
        "en" => "English",
        _ => "the user's language",
    }
}

/// Guess MIME type from a file extension.
fn _guess_mime(path: &str) -> String {
    let lower = path.to_lowercase();
    if lower.ends_with(".jpg") || lower.ends_with(".jpeg") {
        "image/jpeg".to_string()
    } else if lower.ends_with(".png") {
        "image/png".to_string()
    } else if lower.ends_with(".gif") {
        "image/gif".to_string()
    } else if lower.ends_with(".webp") {
        "image/webp".to_string()
    } else if lower.ends_with(".svg") {
        "image/svg+xml".to_string()
    } else {
        "application/octet-stream".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    /// Helper: create a ContextBuilder backed by a temporary workspace.
    fn make_context() -> (TempDir, ContextBuilder) {
        let tmp = TempDir::new().unwrap();
        let cb = ContextBuilder::new(tmp.path());
        (tmp, cb)
    }

    // ----- _guess_mime -----

    #[test]
    fn test_guess_mime_jpg() {
        assert_eq!(_guess_mime("photo.jpg"), "image/jpeg");
    }

    #[test]
    fn test_guess_mime_jpeg() {
        assert_eq!(_guess_mime("photo.jpeg"), "image/jpeg");
    }

    #[test]
    fn test_guess_mime_png() {
        assert_eq!(_guess_mime("image.png"), "image/png");
    }

    #[test]
    fn test_guess_mime_gif() {
        assert_eq!(_guess_mime("anim.gif"), "image/gif");
    }

    #[test]
    fn test_guess_mime_webp() {
        assert_eq!(_guess_mime("pic.webp"), "image/webp");
    }

    #[test]
    fn test_guess_mime_svg() {
        assert_eq!(_guess_mime("icon.svg"), "image/svg+xml");
    }

    #[test]
    fn test_guess_mime_unknown() {
        assert_eq!(_guess_mime("archive.tar.gz"), "application/octet-stream");
    }

    #[test]
    fn test_guess_mime_case_insensitive() {
        assert_eq!(_guess_mime("PHOTO.JPG"), "image/jpeg");
        assert_eq!(_guess_mime("image.PNG"), "image/png");
    }

    // ----- build_system_prompt -----

    #[test]
    fn test_build_system_prompt_contains_nanobot_identity() {
        let (_tmp, cb) = make_context();
        let prompt = cb.build_system_prompt(None);
        assert!(
            prompt.contains("# nanobot"),
            "system prompt should contain 'nanobot' heading"
        );
        assert!(
            prompt.contains("Current Time"),
            "system prompt should contain current time section"
        );
    }

    #[test]
    fn test_build_system_prompt_contains_workspace_path() {
        let (_tmp, cb) = make_context();
        let prompt = cb.build_system_prompt(None);
        let workspace_str = cb
            .workspace
            .canonicalize()
            .unwrap_or_else(|_| cb.workspace.clone())
            .to_string_lossy()
            .to_string();
        assert!(
            prompt.contains(&workspace_str),
            "system prompt should contain workspace path"
        );
    }

    #[test]
    fn test_build_system_prompt_includes_bootstrap_file() {
        let tmp = TempDir::new().unwrap();
        // Write a bootstrap file that ContextBuilder looks for.
        fs::write(tmp.path().join("SOUL.md"), "I am a helpful bot").unwrap();
        let cb = ContextBuilder::new(tmp.path());
        let prompt = cb.build_system_prompt(None);
        assert!(
            prompt.contains("I am a helpful bot"),
            "system prompt should include content from SOUL.md"
        );
    }

    #[test]
    fn test_build_system_prompt_includes_memory() {
        let tmp = TempDir::new().unwrap();
        let memory_dir = tmp.path().join("memory");
        fs::create_dir_all(&memory_dir).unwrap();
        fs::write(memory_dir.join("MEMORY.md"), "User prefers dark mode").unwrap();
        let cb = ContextBuilder::new(tmp.path());
        let prompt = cb.build_system_prompt(None);
        assert!(
            prompt.contains("User prefers dark mode"),
            "system prompt should include long-term memory"
        );
    }

    #[test]
    fn test_build_system_prompt_applies_bootstrap_budget() {
        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("AGENTS.md"), "x".repeat(4000)).unwrap();
        let mut cb = ContextBuilder::new(tmp.path());
        cb.bootstrap_budget = 40;
        let prompt = cb.build_system_prompt(None);
        assert!(prompt.contains("[truncated to fit token budget]"));
    }

    #[test]
    fn test_build_system_prompt_applies_learning_budget() {
        let tmp = TempDir::new().unwrap();
        let store = LearningStore::new(tmp.path());
        for _ in 0..6 {
            store.record("exec", false, "bad", Some(&"very long error ".repeat(30)));
        }
        let mut cb = ContextBuilder::new(tmp.path());
        cb.learning_budget = 40;
        let prompt = cb.build_system_prompt(None);
        assert!(prompt.contains("# Recent Tool Patterns"));
        assert!(prompt.contains("[truncated to fit token budget]"));
    }

    // ----- build_messages -----

    #[test]
    fn test_build_messages_structure() {
        let (_tmp, cb) = make_context();
        let history: Vec<Value> = vec![json!({"role": "user", "content": "hello"})];
        let messages =
            cb.build_messages(&history, "what's up?", None, None, None, None, false, None);

        // Should have: system, history entry, current user message.
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[1]["role"], "user");
        assert_eq!(messages[1]["content"], "hello");
        assert_eq!(messages[2]["role"], "user");
        assert_eq!(messages[2]["content"], "what's up?");
    }

    #[test]
    fn test_build_messages_includes_channel_and_chat_id() {
        let (_tmp, cb) = make_context();
        let messages = cb.build_messages(
            &[],
            "hi",
            None,
            None,
            Some("telegram"),
            Some("12345"),
            false,
            None,
        );
        let system_content = messages[0]["content"].as_str().unwrap();
        assert!(system_content.contains("Channel: telegram"));
        assert!(system_content.contains("Chat ID: 12345"));
    }

    #[test]
    fn test_build_messages_without_history() {
        let (_tmp, cb) = make_context();
        let messages = cb.build_messages(&[], "test", None, None, None, None, false, None);
        // system + current user message
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[1]["role"], "user");
    }

    // ----- add_tool_result -----

    #[test]
    fn test_add_tool_result() {
        let mut messages: Vec<Value> = Vec::new();
        ContextBuilder::add_tool_result(&mut messages, "call_123", "read_file", "file content");

        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["role"], "tool");
        assert_eq!(messages[0]["tool_call_id"], "call_123");
        assert_eq!(messages[0]["name"], "read_file");
        assert_eq!(messages[0]["content"], "file content");
    }

    #[test]
    fn test_add_tool_result_multiple() {
        let mut messages: Vec<Value> = Vec::new();
        ContextBuilder::add_tool_result(&mut messages, "c1", "tool_a", "result_a");
        ContextBuilder::add_tool_result(&mut messages, "c2", "tool_b", "result_b");
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["name"], "tool_a");
        assert_eq!(messages[1]["name"], "tool_b");
    }

    // ----- add_assistant_message -----

    #[test]
    fn test_add_assistant_message_content_only() {
        let mut messages: Vec<Value> = Vec::new();
        ContextBuilder::add_assistant_message(&mut messages, Some("Hello!"), None);

        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["role"], "assistant");
        assert_eq!(messages[0]["content"], "Hello!");
        assert!(messages[0].get("tool_calls").is_none());
    }

    #[test]
    fn test_add_assistant_message_with_tool_calls() {
        let mut messages: Vec<Value> = Vec::new();
        let tool_calls = vec![json!({
            "id": "tc_1",
            "type": "function",
            "function": {"name": "read_file", "arguments": "{}"}
        })];
        ContextBuilder::add_assistant_message(
            &mut messages,
            Some("Let me check."),
            Some(&tool_calls),
        );

        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["role"], "assistant");
        assert_eq!(messages[0]["content"], "Let me check.");
        let tc = messages[0]["tool_calls"].as_array().unwrap();
        assert_eq!(tc.len(), 1);
        assert_eq!(tc[0]["id"], "tc_1");
    }

    #[test]
    fn test_add_assistant_message_no_content() {
        let mut messages: Vec<Value> = Vec::new();
        ContextBuilder::add_assistant_message(&mut messages, None, None);
        assert_eq!(messages[0]["content"], "");
    }

    #[test]
    fn test_add_assistant_message_empty_tool_calls() {
        let mut messages: Vec<Value> = Vec::new();
        let empty: Vec<Value> = vec![];
        ContextBuilder::add_assistant_message(&mut messages, Some("ok"), Some(&empty));
        // Empty tool_calls should not add the key.
        assert!(messages[0].get("tool_calls").is_none());
    }

    // ----- observations -----

    #[test]
    fn test_system_prompt_includes_observations_when_present() {
        let tmp = TempDir::new().unwrap();
        let obs_dir = tmp.path().join("memory").join("observations");
        fs::create_dir_all(&obs_dir).unwrap();
        fs::write(
            obs_dir.join("20260101T000000Z_test.md"),
            "---\ntimestamp: 2026-01-01T00:00:00Z\nsession: test\n---\n\nUser likes dark mode and Rust.",
        )
        .unwrap();
        let cb = ContextBuilder::new(tmp.path());
        let prompt = cb.build_system_prompt(None);
        assert!(
            prompt.contains("Observations from Past Conversations"),
            "system prompt should include observations header"
        );
        assert!(
            prompt.contains("User likes dark mode and Rust"),
            "system prompt should include observation content"
        );
    }

    #[test]
    fn test_system_prompt_respects_observation_budget() {
        let tmp = TempDir::new().unwrap();
        let obs_dir = tmp.path().join("memory").join("observations");
        fs::create_dir_all(&obs_dir).unwrap();
        // Write a large observation.
        let big_content = "x".repeat(40000); // ~10000 tokens
        fs::write(
            obs_dir.join("20260101T000000Z_big.md"),
            format!(
                "---\ntimestamp: 2026-01-01T00:00:00Z\nsession: big\n---\n\n{}",
                big_content
            ),
        )
        .unwrap();
        let mut cb = ContextBuilder::new(tmp.path());
        cb.observation_budget = 100; // very small budget

        let prompt = cb.build_system_prompt(None);
        // If the observation exceeds budget, it should not appear.
        // (The single entry is too big for 100 tokens, so get_context returns empty.)
        assert!(
            !prompt.contains("xxxx"),
            "oversized observation should not appear in prompt"
        );
    }

    // ----- time awareness -----

    #[test]
    fn test_format_time_gap_just_now() {
        let gap = ChronoDuration::seconds(30);
        assert_eq!(ContextBuilder::format_time_gap(gap), "Just now");
    }

    #[test]
    fn test_format_time_gap_minutes() {
        let gap = ChronoDuration::minutes(15);
        assert_eq!(ContextBuilder::format_time_gap(gap), "15 min ago");
    }

    #[test]
    fn test_format_time_gap_hours() {
        let gap = ChronoDuration::hours(3);
        assert_eq!(ContextBuilder::format_time_gap(gap), "3 hours ago");
    }

    #[test]
    fn test_format_time_gap_one_hour() {
        let gap = ChronoDuration::hours(1);
        assert_eq!(ContextBuilder::format_time_gap(gap), "1 hour ago");
    }

    #[test]
    fn test_format_time_gap_yesterday() {
        let gap = ChronoDuration::days(1);
        assert_eq!(ContextBuilder::format_time_gap(gap), "Yesterday");
    }

    #[test]
    fn test_format_time_gap_days() {
        let gap = ChronoDuration::days(5);
        assert_eq!(ContextBuilder::format_time_gap(gap), "5 days ago");
    }

    #[test]
    fn test_time_awareness_in_identity() {
        let (_tmp, mut cb) = make_context();
        cb.last_interaction = Some(Local::now() - ChronoDuration::hours(2));
        let prompt = cb.build_system_prompt(None);
        assert!(
            prompt.contains("Last session: 2 hours ago"),
            "prompt should contain time gap"
        );
        assert!(
            prompt.contains("returning after a short break"),
            "prompt should contain behavioral hint"
        );
    }

    #[test]
    fn test_no_time_awareness_without_last_interaction() {
        let (_tmp, cb) = make_context();
        let prompt = cb.build_system_prompt(None);
        assert!(
            !prompt.contains("Last session"),
            "prompt should not contain time gap without last_interaction"
        );
    }

    // ----- git changes -----

    #[test]
    fn test_git_changes_in_identity() {
        let (_tmp, mut cb) = make_context();
        cb.git_changes = Some("3 file(s) with uncommitted changes".to_string());
        let prompt = cb.build_system_prompt(None);
        assert!(
            prompt.contains("## External Changes"),
            "prompt should contain git changes header"
        );
        assert!(
            prompt.contains("3 file(s) with uncommitted changes"),
            "prompt should contain git change details"
        );
    }

    #[test]
    fn test_no_git_section_when_clean() {
        let (_tmp, cb) = make_context();
        let prompt = cb.build_system_prompt(None);
        assert!(
            !prompt.contains("External Changes"),
            "prompt should not contain git changes when none detected"
        );
    }

    // ----- SLM budget mode -----

    #[test]
    fn test_slm_budget_zeroes_out_sections() {
        let tmp = TempDir::new().unwrap();
        // Create observations and memory that would normally be included.
        let obs_dir = tmp.path().join("memory").join("observations");
        fs::create_dir_all(&obs_dir).unwrap();
        fs::write(
            obs_dir.join("20260101T000000Z_test.md"),
            "---\ntimestamp: 2026-01-01T00:00:00Z\nsession: test\n---\n\nSome observation.",
        )
        .unwrap();
        let memory_dir = tmp.path().join("memory");
        fs::write(memory_dir.join("MEMORY.md"), "User prefers dark mode").unwrap();

        let mut cb = ContextBuilder::new(tmp.path());
        // Simulate SLM budget mode.
        cb.observation_budget = 0;
        cb.today_notes_budget = 0;
        cb.learning_budget = 0;

        let prompt = cb.build_system_prompt(None);
        assert!(
            !prompt.contains("Observations from Past Conversations"),
            "observations should be excluded with zero budget"
        );
        // Memory should still appear (budget not zeroed in this test).
        assert!(
            prompt.contains("User prefers dark mode"),
            "memory should still appear when its budget is nonzero"
        );
    }
}
