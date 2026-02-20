//! Context builder for assembling agent prompts.
//!
//! Assembles bootstrap files, memory, skills, and conversation history into
//! a coherent prompt for the LLM.

use std::fs;
use std::path::{Path, PathBuf};

use base64::Engine;
use chrono::Local;
use serde_json::{json, Value};

use crate::agent::memory::MemoryStore;
use crate::agent::skills::SkillsLoader;
use crate::agent::token_budget::TokenBudget;

/// Sanitize a tool result before injecting into context.
///
/// - Detects binary content (null bytes in first 512 bytes)
/// - Detects base64 blobs (>50% alphanumeric+/= chars and length > 1000)
/// - Truncates to `max_chars` if too long
/// - Returns the original string unchanged otherwise
pub fn sanitize_tool_result(result: &str, max_chars: usize) -> String {
    if result.is_empty() || max_chars == 0 {
        return result.to_string();
    }

    // Binary detection: null bytes in first 512 bytes.
    let check_len = result.len().min(512);
    if result.as_bytes()[..check_len].contains(&0u8) {
        return format!("[Binary content, {} bytes]", result.len());
    }

    // Base64 detection: >50% base64-alphabet chars and length > 1000.
    if result.len() > 1000 {
        let b64_chars = result
            .bytes()
            .filter(|b| b.is_ascii_alphanumeric() || *b == b'+' || *b == b'/' || *b == b'=')
            .count();
        if b64_chars * 2 > result.len() {
            return format!("[Base64 data, {} chars]", result.len());
        }
    }

    // Truncation.
    if result.len() > max_chars {
        let end = crate::utils::helpers::floor_char_boundary(result, max_chars);
        return format!(
            "{}\n... [truncated, {} chars total]",
            &result[..end],
            result.len()
        );
    }

    result.to_string()
}

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
    /// Max tokens for skills (always-loaded + summary) in the system prompt.
    pub skills_budget: usize,
    /// Max tokens for subagent profiles in the system prompt.
    pub profiles_budget: usize,
    /// Hard cap on the total assembled system prompt (0 = no cap).
    /// Acts as end-to-end safety net: even if individual component budgets
    /// sum to more than this, the final prompt is truncated to fit.
    pub system_prompt_cap: usize,
    /// Whether to inject provenance verification rules into the system prompt.
    pub provenance_enabled: bool,
    /// When true, all skills are loaded as summaries only (RLM lazy mode).
    /// The agent uses `read_skill` to fetch full content on demand.
    pub lazy_skills: bool,
    /// Pre-rendered subagent profiles section (from `profiles_summary()`).
    /// Injected into the system prompt when non-empty.
    pub agent_profiles: String,
}

impl ContextBuilder {
    /// Create a new context builder for the given workspace.
    pub fn new(workspace: &Path) -> Self {
        Self {
            workspace: workspace.to_path_buf(),
            memory: MemoryStore::new(workspace),
            skills: SkillsLoader::new(workspace, None),
            model_name: String::new(),
            bootstrap_budget: 1500,
            long_term_memory_budget: 400,
            skills_budget: 1000,
            profiles_budget: 500,
            system_prompt_cap: 0, // no cap by default (cloud models)
            provenance_enabled: false,
            lazy_skills: false,
            agent_profiles: String::new(),
        }
    }

    /// Create a context builder optimized for local/small models.
    ///
    /// Uses much smaller budgets to keep the system prompt under ~2k tokens,
    /// leaving more room for conversation in limited context windows.
    pub fn new_lite(workspace: &Path) -> Self {
        Self {
            workspace: workspace.to_path_buf(),
            memory: MemoryStore::new(workspace),
            skills: SkillsLoader::new(workspace, None),
            model_name: String::new(),
            bootstrap_budget: 500,
            long_term_memory_budget: 200,
            skills_budget: 300,
            profiles_budget: 200,
            system_prompt_cap: 800, // ~20% of typical 4K local model
            provenance_enabled: false,
            lazy_skills: false,
            agent_profiles: String::new(),
        }
    }

    /// Convert this builder to lite mode (for local models).
    pub fn set_lite_mode(&mut self, max_context_tokens: usize) {
        // Local models: scale down but with tighter clamps.
        self.bootstrap_budget = (max_context_tokens / 50).clamp(300, 2_000); // 2%
        self.long_term_memory_budget = (max_context_tokens / 100).clamp(100, 1_000); // 1%
        self.skills_budget = (max_context_tokens / 50).clamp(200, 1_500); // 2%
        self.profiles_budget = (max_context_tokens / 100).clamp(100, 800); // 1%
        // Hard cap: 30% of context for system prompt, leaving 70% for conversation.
        self.system_prompt_cap = (max_context_tokens * 3 / 10).clamp(500, 4_000);
    }

    /// Scale prompt component budgets proportionally to the model's context window.
    ///
    /// Without this, a 1M-context model gets the same 1500-token bootstrap cap
    /// as a 16K model — wasting 99.85% of available system prompt space.
    pub fn scale_budgets(&mut self, max_context_tokens: usize) {
        self.bootstrap_budget = (max_context_tokens / 50).clamp(500, 20_000); // 2%
        self.long_term_memory_budget = (max_context_tokens / 100).clamp(200, 10_000); // 1%
        self.skills_budget = (max_context_tokens / 25).clamp(500, 20_000); // 4%
        self.profiles_budget = (max_context_tokens / 50).clamp(300, 10_000); // 2%
        // Cloud models: generous cap (40% of context), or 0 to disable.
        self.system_prompt_cap = max_context_tokens * 2 / 5; // 40%
    }

    // ------------------------------------------------------------------
    // Public API
    // ------------------------------------------------------------------

    /// Build the system prompt from bootstrap files, memory, and skills.
    ///
    /// When `channel` is provided, reads `CONTEXT-{channel}.md` for per-channel
    /// session context (falls back to `CONTEXT.md` for backwards compatibility).
    pub fn build_system_prompt(
        &self,
        skill_names: Option<&[String]>,
        channel: Option<&str>,
    ) -> String {
        let mut parts: Vec<String> = Vec::new();

        // Core identity.
        parts.push(self._get_identity());

        // Provenance verification rules.
        if self.provenance_enabled {
            parts.push(
                "## Verification Protocol\n\n\
                 Tool calls are audit-logged and mechanically verified. Unverified claims are redacted.\n\
                 1. QUOTE VERBATIM — exact tool output, no paraphrasing.\n\
                 2. NEVER FABRICATE — no invented paths, outputs, or numbers.\n\
                 3. REPORT ERRORS — exact error messages from tools.\n\
                 4. NO PHANTOM ACTIONS — only claim actions with matching tool calls.\n\
                 5. STATE UNCERTAINTY — say \"truncated\" or \"timed out\" when applicable.\n\
                 6. USE [VERBATIM TOOL OUTPUT] markers — quote from these blocks directly.\n\
                 Every \"let me\" is a PROMISE requiring an immediate tool call."
                    .to_string(),
            );
        }

        // Bootstrap files — include complete files in priority order, skip the rest.
        let bootstrap = self._load_bootstrap_files_within_budget(self.bootstrap_budget);
        if !bootstrap.is_empty() {
            parts.push(bootstrap);
        }

        // Session context — structured snapshot from last compaction.
        // Per-channel file (CONTEXT-cli.md, CONTEXT-telegram.md) prevents
        // concurrent sessions from clobbering each other. Falls back to
        // legacy CONTEXT.md for backwards compatibility.
        let context_path = if let Some(ch) = channel {
            let per_channel = self.workspace.join(format!("CONTEXT-{}.md", ch));
            if per_channel.exists() {
                per_channel
            } else {
                self.workspace.join("CONTEXT.md")
            }
        } else {
            self.workspace.join("CONTEXT.md")
        };
        if context_path.exists() {
            if let Ok(ctx) = fs::read_to_string(&context_path) {
                if !ctx.trim().is_empty() {
                    parts.push(format!("# Session Context\n\n{}", ctx.trim()));
                }
            }
        }

        // Memory context (long-term facts only — observations, daily notes, and
        // learnings have been moved out of the system prompt).
        // Use tail truncation so newest facts (appended by reflector) survive.
        let long_term = Self::_truncate_to_budget_tail(
            &self.memory.read_long_term(),
            self.long_term_memory_budget,
        );
        if !long_term.is_empty() {
            parts.push(format!("# Memory\n\n## Long-term Memory\n{}", long_term));
        }

        // Skills -- progressive loading (with lazy/RLM mode support).
        // All skills content is accumulated then capped at skills_budget.
        let fetch_hint = if self.lazy_skills {
            "Use the read_skill tool to load a skill's full instructions."
        } else {
            "To use a skill, read its SKILL.md file using the read_file tool."
        };

        let mut skills_parts: Vec<String> = Vec::new();

        if !self.lazy_skills {
            // Eager mode: always-loaded skills get full content.
            let always_skills = self.skills.get_always_skills();
            if !always_skills.is_empty() {
                let always_content = self.skills.load_skills_for_context(&always_skills);
                if !always_content.is_empty() {
                    skills_parts.push(format!("# Active Skills\n\n{}", always_content));
                }
            }
        }
        // In lazy mode, always-skills appear in the summary below instead.

        // Available skills: summary only (name + description).
        let skills_summary = self.skills.build_skills_summary();
        if !skills_summary.is_empty() {
            skills_parts.push(format!(
                "# Skills\n\n\
                 The following skills extend your capabilities. {}\n\
                 Skills with available=\"false\" need dependencies installed first \
                 - you can try installing them with apt/brew.\n\n\
                 {}",
                fetch_hint, skills_summary
            ));
        }

        if !skills_parts.is_empty() {
            let combined = skills_parts.join("\n\n");
            let capped = Self::_truncate_to_budget_head(&combined, self.skills_budget);
            if !capped.is_empty() {
                parts.push(capped);
            }
        }

        // Subagent profiles — tells the model what agents exist and when to delegate.
        // Capped at profiles_budget to avoid blowing context for small models.
        if !self.agent_profiles.is_empty() {
            let capped = Self::_truncate_to_budget_head(
                &self.agent_profiles,
                self.profiles_budget,
            );
            if !capped.is_empty() {
                parts.push(capped);
            }
        }

        // Explicitly requested skills (always loaded in full, even in lazy mode).
        if let Some(names) = skill_names {
            if !names.is_empty() {
                let requested = self.skills.load_skills_for_context(names);
                if !requested.is_empty() {
                    parts.push(format!("# Requested Skills\n\n{}", requested));
                }
            }
        }

        let assembled = parts.join("\n\n---\n\n");

        // End-to-end safety net: hard cap on total system prompt size.
        if self.system_prompt_cap > 0 {
            Self::_truncate_to_budget_head(&assembled, self.system_prompt_cap)
        } else {
            assembled
        }
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
        let mut system_prompt = self.build_system_prompt(skill_names, channel);
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
                    "- If asked a complex question, give a brief spoken answer, not a written essay.\n",
                    "- Be concise in SPOKEN OUTPUT, but still be resourceful with tools. ",
                    "If one tool or approach fails, try alternatives (spawn agents, web_fetch, exec curl, etc.) before giving up. ",
                    "Never say 'sorry I can not' without exhausting your options.",
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

    /// Add a tool result wrapped with verbatim markers (provenance mode).
    ///
    /// When provenance is enabled, tool results are marked as immutable so the
    /// LLM is instructed to quote rather than paraphrase.
    pub fn add_tool_result_immutable(
        messages: &mut Vec<Value>,
        tool_call_id: &str,
        tool_name: &str,
        result: &str,
    ) {
        let wrapped = format!(
            "[VERBATIM TOOL OUTPUT — do not paraphrase]\n{}\n[END TOOL OUTPUT]",
            result
        );
        messages.push(json!({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": wrapped,
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
                "\n\n## Model\nYou are running locally via LM Studio. \
                 Your model file: {}. You are NOT Claude or any cloud AI. \
                 Respond as nanobot powered by this local model.",
                gguf_name
            )
        } else {
            format!("\n\n## Model\nYou are powered by: {}", self.model_name)
        };

        // Cost-aware delegation hint for expensive models.
        let is_expensive = self.model_name.contains("opus")
            || self.model_name.contains("gpt-4o")
            || self.model_name.contains("claude-max");
        let delegation_hint = if is_expensive && !self.agent_profiles.is_empty() {
            "\n\n## Cost Efficiency\n\
             You are an expensive model. Delegate mechanical tasks to cheap subagents via `spawn`:\n\
             - File writes, web fetches, shell commands whose output you don't need immediately\n\
             - Multi-step research that would burn >1000 tokens of intermediate results\n\
             - Build/test cycles\n\
             Only do it yourself when the result feeds directly into your next sentence."
        } else {
            ""
        };

        format!(
            r#"# nanobot

You are nanobot, a helpful AI assistant with tools for file I/O, shell, web, messaging, and subagents.

## Context
Time: {now}{model_section}
Workspace: {workspace_path}

Be concise (1-5 sentences) unless asked for detail. Use tools to verify; never fabricate.
Reply directly for conversation; use 'message' tool only for chat channels.{delegation_hint}

## Memory
Working Memory is injected automatically (session state). Long-term facts: {workspace_path}/memory/MEMORY.md.
Use `recall` to search all memory (sessions, facts, archives).

If you see a [PRIORITY USER MESSAGE], acknowledge it and adjust your approach — it takes precedence."#
        )
    }

    /// Load bootstrap files within budget using file-granularity inclusion.
    ///
    /// Each file is included in full if it fits the remaining budget; otherwise
    /// it is skipped entirely. Skipped files are listed in a brief note so the
    /// model knows they exist and can fetch them with `read_file`.
    ///
    /// This avoids mid-content truncation that would show broken/incomplete
    /// instructions to the model — a complete file or nothing.
    fn _load_bootstrap_files_within_budget(&self, budget_tokens: usize) -> String {
        if budget_tokens == 0 {
            return String::new();
        }

        let mut included: Vec<String> = Vec::new();
        let mut skipped: Vec<&str> = Vec::new();
        let mut remaining = budget_tokens;

        for filename in BOOTSTRAP_FILES {
            let file_path = self.workspace.join(filename);
            if !file_path.exists() {
                continue;
            }
            let content = match fs::read_to_string(&file_path) {
                Ok(c) if !c.trim().is_empty() => c,
                _ => continue,
            };

            let section = format!("## {}\n\n{}", filename, content);
            let cost = TokenBudget::estimate_str_tokens(&section);

            if cost <= remaining {
                included.push(section);
                remaining = remaining.saturating_sub(cost);
            } else {
                skipped.push(filename);
            }
        }

        // Tell the model about skipped files (never a "truncated" marker).
        if !skipped.is_empty() {
            let note = format!(
                "_Workspace files available via read_file: {}_",
                skipped.join(", ")
            );
            let note_cost = TokenBudget::estimate_str_tokens(&note);
            if note_cost <= remaining {
                included.push(note);
            }
        }

        included.join("\n\n")
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

    /// Truncate keeping the HEAD of the text (most important content first).
    ///
    /// Used for skills and profiles where the beginning is most relevant.
    /// Truncates at line boundaries.
    fn _truncate_to_budget_head(text: &str, max_tokens: usize) -> String {
        if text.is_empty() || max_tokens == 0 {
            return String::new();
        }

        if TokenBudget::estimate_str_tokens(text) <= max_tokens {
            return text.to_string();
        }

        let max_chars = max_tokens.saturating_mul(4);
        let mut kept = String::new();

        for line in text.lines() {
            let line_cost = line.len() + 1;
            if kept.len() + line_cost > max_chars && !kept.is_empty() {
                break;
            }
            if !kept.is_empty() {
                kept.push('\n');
            }
            kept.push_str(line);
        }

        kept
    }

    /// Truncate keeping the TAIL of the text (newest content), silently.
    ///
    /// Used for MEMORY.md where the reflector appends new facts at the bottom.
    /// Truncates at line boundaries — never mid-line, never with a visible marker.
    /// The model sees clean content, not broken instructions.
    fn _truncate_to_budget_tail(text: &str, max_tokens: usize) -> String {
        if text.is_empty() || max_tokens == 0 {
            return String::new();
        }

        if TokenBudget::estimate_str_tokens(text) <= max_tokens {
            return text.to_string();
        }

        // Keep roughly max_tokens worth of text from the tail, at line boundaries.
        let max_chars = max_tokens.saturating_mul(4);
        let lines: Vec<&str> = text.lines().collect();
        let mut kept: Vec<&str> = Vec::new();
        let mut char_count = 0;

        for line in lines.iter().rev() {
            let line_cost = line.len() + 1; // +1 for newline
            if char_count + line_cost > max_chars && !kept.is_empty() {
                break;
            }
            kept.push(line);
            char_count += line_cost;
        }

        kept.reverse();
        kept.join("\n")
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
        let prompt = cb.build_system_prompt(None, None);
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
        let prompt = cb.build_system_prompt(None, None);
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
        let prompt = cb.build_system_prompt(None, None);
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
        let prompt = cb.build_system_prompt(None, None);
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
        cb.bootstrap_budget = 40; // Too small to fit AGENTS.md
        let prompt = cb.build_system_prompt(None, None);
        // File should be skipped entirely — never truncated mid-content.
        assert!(!prompt.contains("[truncated"));
        // Model should be told the file exists and can be fetched.
        assert!(prompt.contains("AGENTS.md"));
        assert!(prompt.contains("read_file"));
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

    // ----- add_tool_result_immutable -----

    #[test]
    fn test_add_tool_result_immutable_wraps_content() {
        let mut messages: Vec<Value> = Vec::new();
        ContextBuilder::add_tool_result_immutable(
            &mut messages,
            "call_1",
            "read_file",
            "file content",
        );

        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["role"], "tool");
        assert_eq!(messages[0]["tool_call_id"], "call_1");
        assert_eq!(messages[0]["name"], "read_file");
        let content = messages[0]["content"].as_str().unwrap();
        assert!(content.starts_with("[VERBATIM TOOL OUTPUT"));
        assert!(content.contains("file content"));
        assert!(content.ends_with("[END TOOL OUTPUT]"));
    }

    #[test]
    fn test_add_tool_result_immutable_empty_result() {
        let mut messages: Vec<Value> = Vec::new();
        ContextBuilder::add_tool_result_immutable(&mut messages, "c1", "exec", "");
        let content = messages[0]["content"].as_str().unwrap();
        assert!(content.contains("[VERBATIM TOOL OUTPUT"));
        assert!(content.contains("[END TOOL OUTPUT]"));
    }

    #[test]
    fn test_truncate_to_budget_tail_keeps_newest() {
        // Long text should keep the tail (newest facts), silently.
        let lines: Vec<String> = (0..500).map(|i| format!("FACT {}", i)).collect();
        let text = lines.join("\n");
        let result = ContextBuilder::_truncate_to_budget_tail(&text, 40);
        assert!(
            result.contains("FACT 499"),
            "newest facts should survive tail truncation"
        );
        assert!(!result.contains("FACT 0"), "oldest facts should be dropped");
        // No truncation marker visible to the model.
        assert!(!result.contains("[truncated"), "no truncation markers");
        assert!(!result.contains("[earlier"), "no earlier-memory markers");
    }

    #[test]
    fn test_truncate_to_budget_tail_short_text_no_truncation() {
        let text = "Short text.";
        let result = ContextBuilder::_truncate_to_budget_tail(text, 100);
        assert_eq!(result, text);
    }

    #[test]
    fn test_truncate_to_budget_tail_empty() {
        assert_eq!(ContextBuilder::_truncate_to_budget_tail("", 100), "");
        assert_eq!(ContextBuilder::_truncate_to_budget_tail("hello", 0), "");
    }

    #[test]
    fn test_build_system_prompt_memory_uses_tail_truncation() {
        // Verify that with a tight budget, newest facts survive.
        let tmp = TempDir::new().unwrap();
        let memory_dir = tmp.path().join("memory");
        fs::create_dir_all(&memory_dir).unwrap();
        let content = format!("- OLD FACT 1\n{}\n- NEWEST FACT", "- filler\n".repeat(500));
        fs::write(memory_dir.join("MEMORY.md"), &content).unwrap();
        let mut cb = ContextBuilder::new(tmp.path());
        cb.long_term_memory_budget = 20; // very tight
        let prompt = cb.build_system_prompt(None, None);
        assert!(
            prompt.contains("NEWEST FACT"),
            "newest fact should survive tight budget"
        );
    }

    #[test]
    fn test_system_prompt_no_longer_includes_observations() {
        let tmp = TempDir::new().unwrap();
        let obs_dir = tmp.path().join("memory").join("observations");
        fs::create_dir_all(&obs_dir).unwrap();
        fs::write(
            obs_dir.join("20260101T000000Z_test.md"),
            "---\ntimestamp: 2026-01-01T00:00:00Z\nsession: test\n---\n\nUser likes dark mode.",
        )
        .unwrap();
        let cb = ContextBuilder::new(tmp.path());
        let prompt = cb.build_system_prompt(None, None);
        assert!(
            !prompt.contains("Observations from Past Conversations"),
            "observations should no longer be injected into system prompt"
        );
    }

    // ----- sanitize_tool_result -----

    #[test]
    fn test_sanitize_passthrough() {
        let result = sanitize_tool_result("normal text output", 30000);
        assert_eq!(result, "normal text output");
    }

    #[test]
    fn test_sanitize_binary_detection() {
        let mut data = "some text\0with null bytes".to_string();
        let result = sanitize_tool_result(&data, 30000);
        assert!(result.starts_with("[Binary content,"));
    }

    #[test]
    fn test_sanitize_base64_detection() {
        // Create a string that's >50% base64 characters and >1000 chars
        let b64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="
            .repeat(20);
        let result = sanitize_tool_result(&b64, 30000);
        assert!(result.starts_with("[Base64 data,"), "got: {}", result);
    }

    #[test]
    fn test_sanitize_truncation() {
        let long = "x".repeat(1000);
        let result = sanitize_tool_result(&long, 100);
        assert!(result.len() < 200); // 100 + truncation message
        assert!(result.contains("[truncated, 1000 chars total]"));
    }

    #[test]
    fn test_sanitize_empty() {
        assert_eq!(sanitize_tool_result("", 100), "");
    }

    // ----- _truncate_to_budget_head -----

    #[test]
    fn test_system_prompt_cap_truncates() {
        let tmp = TempDir::new().unwrap();
        // Create a large AGENTS.md to blow the budget.
        fs::write(tmp.path().join("AGENTS.md"), "A".repeat(10_000)).unwrap();
        let mut cb = ContextBuilder::new(tmp.path());
        cb.system_prompt_cap = 50; // very tight cap
        let prompt = cb.build_system_prompt(None, None);
        let tokens = TokenBudget::estimate_str_tokens(&prompt);
        assert!(
            tokens <= 75, // allow overshoot due to line boundary truncation
            "system prompt ({} tokens) should be near cap (50)",
            tokens
        );
    }

    #[test]
    fn test_system_prompt_no_cap_by_default() {
        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("AGENTS.md"), "A".repeat(10_000)).unwrap();
        let cb = ContextBuilder::new(tmp.path()); // default: cap=0 (disabled)
        let prompt = cb.build_system_prompt(None, None);
        assert!(
            prompt.contains("AAAA"),
            "without cap, large content should pass through"
        );
    }

    #[test]
    fn test_truncate_to_budget_head_keeps_beginning() {
        let lines: Vec<String> = (0..500).map(|i| format!("SKILL {}", i)).collect();
        let text = lines.join("\n");
        let result = ContextBuilder::_truncate_to_budget_head(&text, 40);
        assert!(
            result.contains("SKILL 0"),
            "beginning should survive head truncation"
        );
        assert!(
            !result.contains("SKILL 499"),
            "end should be dropped"
        );
    }

    #[test]
    fn test_truncate_to_budget_head_short_passthrough() {
        let text = "Short text.";
        let result = ContextBuilder::_truncate_to_budget_head(text, 100);
        assert_eq!(result, text);
    }

    #[test]
    fn test_truncate_to_budget_head_empty() {
        assert_eq!(ContextBuilder::_truncate_to_budget_head("", 100), "");
        assert_eq!(ContextBuilder::_truncate_to_budget_head("hello", 0), "");
    }
}
