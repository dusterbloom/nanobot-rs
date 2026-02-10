//! Context builder for assembling agent prompts.
//!
//! Assembles bootstrap files, memory, skills, and conversation history into
//! a coherent prompt for the LLM.

use std::fs;
use std::path::{Path, PathBuf};

use base64::Engine;
use chrono::Local;
use serde_json::{json, Value};

use crate::agent::learning::LearningStore;
use crate::agent::memory::MemoryStore;
use crate::agent::skills::SkillsLoader;

/// Well-known files that are loaded from the workspace root when present.
const BOOTSTRAP_FILES: &[&str] = &[
    "AGENTS.md",
    "SOUL.md",
    "USER.md",
    "TOOLS.md",
    "IDENTITY.md",
];

/// Builds the context (system prompt + messages) for the agent.
pub struct ContextBuilder {
    pub workspace: PathBuf,
    pub memory: MemoryStore,
    pub skills: SkillsLoader,
    pub model_name: String,
}

impl ContextBuilder {
    /// Create a new context builder for the given workspace.
    pub fn new(workspace: &Path) -> Self {
        Self {
            workspace: workspace.to_path_buf(),
            memory: MemoryStore::new(workspace),
            skills: SkillsLoader::new(workspace, None),
            model_name: String::new(),
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
        let bootstrap = self._load_bootstrap_files();
        if !bootstrap.is_empty() {
            parts.push(bootstrap);
        }

        // Memory context.
        let memory = self.memory.get_memory_context();
        if !memory.is_empty() {
            parts.push(format!("# Memory\n\n{}", memory));
        }

        // Learning context (tool outcome patterns).
        let learning = LearningStore::new(&self.workspace);
        let learning_context = learning.get_learning_context();
        if !learning_context.is_empty() {
            parts.push(format!("# Recent Tool Patterns\n\n{}", learning_context));
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
    ) -> Vec<Value> {
        let mut messages: Vec<Value> = Vec::new();

        // System prompt.
        let mut system_prompt = self.build_system_prompt(skill_names);
        if let (Some(ch), Some(cid)) = (channel, chat_id) {
            system_prompt
                .push_str(&format!("\n\n## Current Session\nChannel: {}\nChat ID: {}", ch, cid));
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

    /// Core identity section including current time and workspace info.
    fn _get_identity(&self) -> String {
        let now = Local::now().format("%Y-%m-%d %H:%M (%A)").to_string();
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

        format!(
            r#"# nanobot

You are nanobot, a helpful AI assistant. You have access to tools that allow you to:
- Read, write, and edit files
- Execute shell commands
- Search the web and fetch web pages
- Send messages to users on chat channels
- Spawn subagents for complex background tasks

## Current Time
{now}{model_section}

## Workspace
Your workspace is at: {workspace_path}
- Memory files: {workspace_path}/memory/MEMORY.md
- Daily notes: {workspace_path}/memory/YYYY-MM-DD.md
- Custom skills: {workspace_path}/skills/{{skill-name}}/SKILL.md

IMPORTANT: When responding to direct questions or conversations, reply directly with your text response.
Only use the 'message' tool when you need to send a message to a specific chat channel (like WhatsApp).
For normal conversation, just respond with text - do not call the message tool.

Always be helpful, accurate, and concise. When using tools, explain what you're doing.
When remembering something, write to {workspace_path}/memory/MEMORY.md"#
        )
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
            prompt.contains("nanobot"),
            "system prompt should contain 'nanobot' identity"
        );
        assert!(
            prompt.contains("You are nanobot"),
            "system prompt should contain identity introduction"
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

    // ----- build_messages -----

    #[test]
    fn test_build_messages_structure() {
        let (_tmp, cb) = make_context();
        let history: Vec<Value> = vec![json!({"role": "user", "content": "hello"})];
        let messages = cb.build_messages(&history, "what's up?", None, None, None, None, false);

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
        let messages =
            cb.build_messages(&[], "hi", None, None, Some("telegram"), Some("12345"), false);
        let system_content = messages[0]["content"].as_str().unwrap();
        assert!(system_content.contains("Channel: telegram"));
        assert!(system_content.contains("Chat ID: 12345"));
    }

    #[test]
    fn test_build_messages_without_history() {
        let (_tmp, cb) = make_context();
        let messages = cb.build_messages(&[], "test", None, None, None, None, false);
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
        ContextBuilder::add_assistant_message(&mut messages, Some("Let me check."), Some(&tool_calls));

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
}
