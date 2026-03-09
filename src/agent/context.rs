#![allow(dead_code)]
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
    if crate::utils::helpers::is_binary(result.as_bytes()) {
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

#[derive(Debug, Clone)]
pub struct PromptBlock {
    title: String,
    content: String,
}

impl PromptBlock {
    pub(crate) fn new(title: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            content: content.into(),
        }
    }

    pub(crate) fn render(&self) -> String {
        if self.content.trim().is_empty() {
            String::new()
        } else if self.title.trim().is_empty() {
            self.content.trim().to_string()
        } else {
            format!("## {}\n\n{}", self.title.trim(), self.content.trim())
        }
    }

    pub(crate) fn report_title(&self) -> String {
        if self.title.trim().is_empty() {
            "Session Metadata".to_string()
        } else {
            self.title.trim().to_string()
        }
    }

    /// Returns a reference to the block title.
    pub(crate) fn title(&self) -> &str {
        &self.title
    }

    /// Returns a reference to the raw content string.
    pub(crate) fn content(&self) -> &str {
        &self.content
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptBlockKind {
    Prefix,
    Static,
    Runtime,
}

#[derive(Debug, Clone)]
pub struct PromptBlockReport {
    pub kind: PromptBlockKind,
    pub title: String,
    pub tokens: usize,
    pub included: bool,
    /// Tokens allocated by the budget (0 = legacy/untracked).
    pub allocated_tokens: usize,
    /// Source descriptor (empty string = legacy/untracked).
    pub source: String,
}

#[derive(Debug, Clone)]
pub struct PromptAssemblyReport {
    pub prompt: String,
    pub total_tokens: usize,
    pub cap_tokens: Option<usize>,
    pub blocks: Vec<PromptBlockReport>,
}

/// Builds the context (system prompt + messages) for the agent.
pub struct ContextBuilder {
    pub workspace: PathBuf,
    pub memory: MemoryStore,
    pub skills: SkillsLoader,
    pub model_name: String,
    /// When true, build a compact local-style prompt instead of the cloud
    /// system+developer split. This is driven by runtime locality, not by the
    /// model name prefix alone, so MLX local inference can use the lean path.
    pub local_prompt_mode: bool,
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
    /// Controls how skills are disclosed in the system prompt.
    /// - "compact" (default): one-line index per skill (~20 tokens each)
    /// - "xml": full XML summary with descriptions and metadata
    /// - "eager": full skill content loaded into the system prompt
    pub skill_disclosure: String,
    /// Pre-rendered subagent profiles section (from `profiles_summary()`).
    /// Injected into the system prompt when non-empty.
    pub agent_profiles: String,
    /// Optional instruction profiles for model-specific prompt engineering.
    /// When set, resolved messages are appended to the developer context.
    pub instruction_profiles: Option<crate::agent::instructions::InstructionProfiles>,
    /// Task kind used when resolving instruction profiles (default: "main").
    pub task_kind: String,
}

impl ContextBuilder {
    /// Create a new context builder for the given workspace.
    pub fn new(workspace: &Path) -> Self {
        Self {
            workspace: workspace.to_path_buf(),
            memory: MemoryStore::new(workspace),
            skills: SkillsLoader::new(workspace, None),
            model_name: String::new(),
            local_prompt_mode: false,
            bootstrap_budget: 1500,
            long_term_memory_budget: 400,
            skills_budget: 1000,
            profiles_budget: 500,
            system_prompt_cap: 0, // no cap by default (cloud models)
            provenance_enabled: false,
            lazy_skills: false,
            skill_disclosure: "compact".to_string(),
            agent_profiles: String::new(),
            instruction_profiles: None,
            task_kind: "main".to_string(),
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
            local_prompt_mode: true,
            bootstrap_budget: 500,
            long_term_memory_budget: 200,
            skills_budget: 300,
            profiles_budget: 200,
            system_prompt_cap: 800, // ~20% of typical 4K local model
            provenance_enabled: false,
            lazy_skills: false,
            skill_disclosure: "compact".to_string(),
            agent_profiles: String::new(),
            instruction_profiles: None,
            task_kind: "main".to_string(),
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

    /// Build the core identity section of the system prompt.
    ///
    /// This is always sent as the `system` role message. Includes identity,
    /// provenance rules, and bootstrap files (AGENTS.md, SOUL.md, etc.).
    pub fn build_identity_prompt(&self) -> String {
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

        parts.join("\n\n---\n\n")
    }

    /// Build the injected context section for the `developer` role message.
    ///
    /// Contains long-term memory, skills, and subagent profiles. Returns an
    /// empty string when there is nothing to inject. Session context (compaction
    /// summaries) is injected separately by prepare_context, not here.
    /// On cloud APIs this is emitted as a separate `developer` role message;
    /// on local models it is folded back into the `system` message.
    pub fn build_developer_context(
        &self,
        skill_names: Option<&[String]>,
        _channel: Option<&str>,
    ) -> String {
        let mut parts: Vec<String> = Vec::new();

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

        // Knowledge graph context (entity/relation summaries).
        #[cfg(feature = "knowledge-graph")]
        {
            if let Ok(kg) = crate::agent::knowledge_graph::KnowledgeGraph::open_default() {
                let kg_ctx = kg.export_context(30);
                if !kg_ctx.is_empty() {
                    parts.push(format!("## Knowledge Graph\n{}", kg_ctx));
                }
            }
        }

        // Skills -- progressive loading (3-tier disclosure model).
        // Tier selection is driven by `skill_disclosure`:
        //   "compact" (default): one-line index in system prompt, full content on demand via read_skill
        //   "xml": full XML summary with descriptions and metadata
        //   "eager": full skill content loaded immediately (legacy behaviour)
        // `lazy_skills` is kept for backward compat and treated as "xml" disclosure.
        let effective_disclosure = if self.skill_disclosure == "eager" {
            "eager"
        } else if self.skill_disclosure == "xml"
            || (!self.skill_disclosure.is_empty() && self.skill_disclosure != "compact")
        {
            "xml"
        } else {
            // "compact" is default; also applied when lazy_skills=true (unless overridden)
            "compact"
        };

        let mut skills_parts: Vec<String> = Vec::new();

        match effective_disclosure {
            "eager" => {
                // Eager mode: always-loaded skills get full content, others as XML summary.
                let always_skills = self.skills.get_always_skills();
                if !always_skills.is_empty() {
                    let always_content = self.skills.load_skills_for_context(&always_skills);
                    if !always_content.is_empty() {
                        skills_parts.push(format!("# Active Skills\n\n{}", always_content));
                    }
                }
                let skills_summary = self.skills.build_skills_summary();
                if !skills_summary.is_empty() {
                    skills_parts.push(format!(
                        "# Skills\n\n\
                         The following skills extend your capabilities. \
                         To use a skill, read its SKILL.md file using the read_file tool.\n\
                         Skills with available=\"false\" need dependencies installed first \
                         - you can try installing them with apt/brew.\n\n\
                         {}",
                        skills_summary
                    ));
                }
            }
            "xml" => {
                // XML mode: full XML summary, agent fetches full content via read_skill.
                let skills_summary = self.skills.build_skills_summary();
                if !skills_summary.is_empty() {
                    skills_parts.push(format!(
                        "# Skills\n\n\
                         The following skills extend your capabilities. \
                         Use the read_skill tool to load a skill's full instructions.\n\
                         Skills with available=\"false\" need dependencies installed first \
                         - you can try installing them with apt/brew.\n\n\
                         {}",
                        skills_summary
                    ));
                }
            }
            _ => {
                // "compact" (default): one-line index, full content on demand.
                let index = self.skills.build_compact_index();
                if !index.is_empty() {
                    skills_parts.push(format!("# Skills\n\n{}", index));
                }
            }
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
            let capped = Self::_truncate_to_budget_head(&self.agent_profiles, self.profiles_budget);
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

        // Instruction profiles — model- and task-specific prompt engineering.
        // Resolved messages are appended last so they can override earlier context.
        if let Some(ref profiles) = self.instruction_profiles {
            let messages = profiles.resolve(&self.model_name, &self.task_kind);
            for msg in messages {
                if !msg.content.is_empty() {
                    parts.push(format!(
                        "<!-- instruction-profile role={} -->\n{}",
                        msg.role, msg.content
                    ));
                }
            }
        }

        parts.join("\n\n---\n\n")
    }

    /// Build the system prompt from bootstrap files, memory, and skills.
    ///
    /// Returns the full prompt as a single concatenated string (identity +
    /// developer context). Used by local models and as a convenience method
    /// where a single string is needed.
    ///
    /// The `channel` parameter is forwarded to `build_developer_context` for
    /// API compatibility; session context is injected by prepare_context.
    pub fn build_system_prompt(
        &self,
        skill_names: Option<&[String]>,
        channel: Option<&str>,
    ) -> String {
        let identity = self.build_identity_prompt();
        let developer = self.build_developer_context(skill_names, channel);

        let assembled = if developer.is_empty() {
            identity
        } else {
            format!("{}\n\n---\n\n{}", identity, developer)
        };

        // End-to-end safety net: hard cap on total system prompt size.
        if self.system_prompt_cap > 0 {
            Self::_truncate_to_budget_head(&assembled, self.system_prompt_cap)
        } else {
            assembled
        }
    }

    /// Build a compact local system prompt from a stable prefix plus optional
    /// static and runtime blocks. The final result is capped end-to-end.
    ///
    /// Delegates to `LocalAssembler` for budget-aware section assembly.
    pub fn build_local_system_prompt(
        &self,
        skill_names: Option<&[String]>,
        channel: Option<&str>,
        chat_id: Option<&str>,
        is_voice_message: bool,
        detected_language: Option<&str>,
        runtime_blocks: &[PromptBlock],
    ) -> String {
        use crate::agent::prompt_contract::{
            AssemblyContext, LocalAssembler, PromptAssembler, PromptSection, SectionEntry,
            SectionSource,
        };

        let mut sections = self._collect_local_sections(
            skill_names,
            channel,
            chat_id,
            is_voice_message,
            detected_language,
        );

        // Convert runtime PromptBlocks to SectionEntry values.
        for block in runtime_blocks {
            let title = block.report_title();
            let section = match title.as_str() {
                "Working Memory" => PromptSection::WorkingMemory,
                "Tool Patterns" => PromptSection::ToolPatterns,
                "Background Tasks" => PromptSection::BackgroundTasks,
                _ => PromptSection::MemoryBriefing,
            };
            sections.push(SectionEntry {
                section,
                block: block.clone(),
                allocated_tokens: 0,
                actual_tokens: 0,
                source: SectionSource::Runtime(title),
                included: true,
                shrinkable: section.shrinkable(),
            });
        }

        let context_window = if self.system_prompt_cap > 0 {
            (self.system_prompt_cap as f64 / 0.3).round() as usize
        } else {
            16_000 // default local context
        };

        let ctx = AssemblyContext {
            context_window,
            system_prompt_cap_pct: 0.3,
            sections,
        };
        LocalAssembler.assemble(&ctx).system_content
    }

    /// Describe the compact local prompt assembly with per-block token counts.
    ///
    /// Uses `LocalAssembler` for consistent assembly with `build_local_system_prompt()`.
    pub fn describe_local_system_prompt(
        &self,
        skill_names: Option<&[String]>,
        channel: Option<&str>,
        chat_id: Option<&str>,
        is_voice_message: bool,
        detected_language: Option<&str>,
        runtime_blocks: &[PromptBlock],
    ) -> PromptAssemblyReport {
        use crate::agent::prompt_contract::{
            AssemblyContext, LocalAssembler, PromptAssembler, PromptSection, SectionEntry,
            SectionSource,
        };

        let mut sections = self._collect_local_sections(
            skill_names,
            channel,
            chat_id,
            is_voice_message,
            detected_language,
        );

        for block in runtime_blocks {
            let title = block.report_title();
            let section = match title.as_str() {
                "Working Memory" => PromptSection::WorkingMemory,
                "Tool Patterns" => PromptSection::ToolPatterns,
                "Background Tasks" => PromptSection::BackgroundTasks,
                _ => PromptSection::MemoryBriefing,
            };
            sections.push(SectionEntry {
                section,
                block: block.clone(),
                allocated_tokens: 0,
                actual_tokens: 0,
                source: SectionSource::Runtime(title),
                included: true,
                shrinkable: section.shrinkable(),
            });
        }

        let context_window = if self.system_prompt_cap > 0 {
            (self.system_prompt_cap as f64 / 0.3).round() as usize
        } else {
            16_000
        };

        let ctx = AssemblyContext {
            context_window,
            system_prompt_cap_pct: 0.3,
            sections,
        };
        LocalAssembler.assemble(&ctx).report
    }

    /// Collect local prompt sections as `SectionEntry` values.
    ///
    /// Converts the local identity prefix and static blocks into typed entries
    /// for consumption by `LocalAssembler`.
    fn _collect_local_sections(
        &self,
        skill_names: Option<&[String]>,
        channel: Option<&str>,
        chat_id: Option<&str>,
        is_voice_message: bool,
        detected_language: Option<&str>,
    ) -> Vec<crate::agent::prompt_contract::SectionEntry> {
        use crate::agent::prompt_contract::{PromptSection, SectionEntry, SectionSource};

        let mut sections = Vec::new();

        // Identity prefix.
        let identity = self._get_local_identity();
        if !identity.is_empty() {
            sections.push(SectionEntry {
                section: PromptSection::Identity,
                block: PromptBlock::new("", &identity),
                allocated_tokens: 0,
                actual_tokens: 0,
                source: SectionSource::Static("local identity"),
                included: true,
                shrinkable: PromptSection::Identity.shrinkable(),
            });
        }

        // Static blocks from the existing local builder.
        let static_blocks = self.build_local_static_blocks(
            skill_names,
            channel,
            chat_id,
            is_voice_message,
            detected_language,
        );

        for block in &static_blocks {
            let title = block.report_title();
            let section = match title.as_str() {
                "Verification" => PromptSection::Verification,
                "Workspace Context" => PromptSection::WorkspaceContext,
                "On-Demand Context" => PromptSection::OnDemandContext,
                "Skills" => PromptSection::Skills,
                "Requested Skills" => PromptSection::RequestedSkills,
                "Session Metadata" => PromptSection::SessionMetadata,
                "Tool Use" => PromptSection::ToolUse,
                _ => PromptSection::SessionMetadata,
            };
            sections.push(SectionEntry {
                section,
                block: block.clone(),
                allocated_tokens: 0,
                actual_tokens: 0,
                source: SectionSource::Static("local static"),
                included: true,
                shrinkable: section.shrinkable(),
            });
        }

        sections
    }

    /// Collect all static prompt sections as typed `SectionEntry` values.
    ///
    /// Converts existing `build_identity_prompt()` and `build_developer_context()`
    /// content into the `PromptSection` taxonomy for assembler consumption.
    pub fn collect_static_sections(
        &self,
        skill_names: Option<&[String]>,
        channel: Option<&str>,
        chat_id: Option<&str>,
        is_voice_message: bool,
        detected_language: Option<&str>,
    ) -> Vec<crate::agent::prompt_contract::SectionEntry> {
        use crate::agent::prompt_contract::{PromptSection, SectionEntry, SectionSource};

        let mut sections = Vec::new();

        // --- Identity section (core identity text) ---
        let identity_text = self._get_identity();
        if !identity_text.is_empty() {
            sections.push(SectionEntry {
                section: PromptSection::Identity,
                block: PromptBlock::new("", &identity_text),
                allocated_tokens: 0,
                actual_tokens: 0,
                source: SectionSource::Static("core identity"),
                included: true,
                shrinkable: PromptSection::Identity.shrinkable(),
            });
        }

        // --- Verification section (provenance rules) ---
        if self.provenance_enabled {
            let verification_text =
                "## Verification Protocol\n\n\
                 Tool calls are audit-logged and mechanically verified. Unverified claims are redacted.\n\
                 1. QUOTE VERBATIM — exact tool output, no paraphrasing.\n\
                 2. NEVER FABRICATE — no invented paths, outputs, or numbers.\n\
                 3. REPORT ERRORS — exact error messages from tools.\n\
                 4. NO PHANTOM ACTIONS — only claim actions with matching tool calls.\n\
                 5. STATE UNCERTAINTY — say \"truncated\" or \"timed out\" when applicable.\n\
                 6. USE [VERBATIM TOOL OUTPUT] markers — quote from these blocks directly.\n\
                 Every \"let me\" is a PROMISE requiring an immediate tool call.";
            sections.push(SectionEntry {
                section: PromptSection::Verification,
                block: PromptBlock::new("Verification Protocol", verification_text),
                allocated_tokens: 0,
                actual_tokens: 0,
                source: SectionSource::Static("provenance rules"),
                included: true,
                shrinkable: PromptSection::Verification.shrinkable(),
            });
        }

        // --- Workspace context (bootstrap files) ---
        let bootstrap = self._load_bootstrap_files_within_budget(self.bootstrap_budget);
        if !bootstrap.is_empty() {
            sections.push(SectionEntry {
                section: PromptSection::WorkspaceContext,
                block: PromptBlock::new("Workspace Context", &bootstrap),
                allocated_tokens: 0,
                actual_tokens: 0,
                source: SectionSource::File("bootstrap files".to_string()),
                included: true,
                shrinkable: PromptSection::WorkspaceContext.shrinkable(),
            });
        }

        // --- Long-term memory (MemoryBriefing section) ---
        let long_term = Self::_truncate_to_budget_tail(
            &self.memory.read_long_term(),
            self.long_term_memory_budget,
        );
        if !long_term.is_empty() {
            sections.push(SectionEntry {
                section: PromptSection::MemoryBriefing,
                block: PromptBlock::new("Memory", &format!("## Long-term Memory\n{}", long_term)),
                allocated_tokens: 0,
                actual_tokens: 0,
                source: SectionSource::File("MEMORY.md".to_string()),
                included: true,
                shrinkable: PromptSection::MemoryBriefing.shrinkable(),
            });
        }

        // --- Knowledge graph context ---
        #[cfg(feature = "knowledge-graph")]
        {
            if let Ok(kg) = crate::agent::knowledge_graph::KnowledgeGraph::open_default() {
                let kg_ctx = kg.export_context(30);
                if !kg_ctx.is_empty() {
                    // Append to MemoryBriefing if it exists, otherwise create a new entry.
                    if let Some(mem_entry) = sections
                        .iter_mut()
                        .find(|s| s.section == PromptSection::MemoryBriefing)
                    {
                        let existing = mem_entry.block.content().to_string();
                        mem_entry.block = PromptBlock::new(
                            "Memory",
                            &format!("{}\n\n## Knowledge Graph\n{}", existing, kg_ctx),
                        );
                    }
                }
            }
        }

        // --- Skills ---
        let skills_content = self._build_skills_content(skill_names);
        if !skills_content.is_empty() {
            sections.push(SectionEntry {
                section: PromptSection::Skills,
                block: PromptBlock::new("Skills", &skills_content),
                allocated_tokens: 0,
                actual_tokens: 0,
                source: SectionSource::Computed("skills loader".to_string()),
                included: true,
                shrinkable: PromptSection::Skills.shrinkable(),
            });
        }

        // --- Requested skills (explicit, always full content) ---
        if let Some(names) = skill_names {
            if !names.is_empty() {
                let requested = self.skills.load_skills_for_context(names);
                if !requested.is_empty() {
                    sections.push(SectionEntry {
                        section: PromptSection::RequestedSkills,
                        block: PromptBlock::new("Requested Skills", &requested),
                        allocated_tokens: 0,
                        actual_tokens: 0,
                        source: SectionSource::Computed("requested skills".to_string()),
                        included: true,
                        shrinkable: PromptSection::RequestedSkills.shrinkable(),
                    });
                }
            }
        }

        // --- Subagent profiles (part of Skills/OnDemandContext area) ---
        if !self.agent_profiles.is_empty() {
            let capped = Self::_truncate_to_budget_head(&self.agent_profiles, self.profiles_budget);
            if !capped.is_empty() {
                sections.push(SectionEntry {
                    section: PromptSection::OnDemandContext,
                    block: PromptBlock::new("Agent Profiles", &capped),
                    allocated_tokens: 0,
                    actual_tokens: 0,
                    source: SectionSource::Computed("agent profiles".to_string()),
                    included: true,
                    shrinkable: PromptSection::OnDemandContext.shrinkable(),
                });
            }
        }

        // --- Session metadata ---
        let session_meta =
            _session_metadata_suffix(channel, chat_id, is_voice_message, detected_language);
        if !session_meta.trim().is_empty() {
            sections.push(SectionEntry {
                section: PromptSection::SessionMetadata,
                block: PromptBlock::new("", &session_meta),
                allocated_tokens: 0,
                actual_tokens: 0,
                source: SectionSource::Runtime("session metadata".to_string()),
                included: true,
                shrinkable: PromptSection::SessionMetadata.shrinkable(),
            });
        }

        // --- Instruction profiles ---
        if let Some(ref profiles) = self.instruction_profiles {
            let messages = profiles.resolve(&self.model_name, &self.task_kind);
            let instruction_parts: Vec<String> = messages
                .iter()
                .filter(|m| !m.content.is_empty())
                .map(|m| {
                    format!(
                        "<!-- instruction-profile role={} -->\n{}",
                        m.role, m.content
                    )
                })
                .collect();
            if !instruction_parts.is_empty() {
                sections.push(SectionEntry {
                    section: PromptSection::ToolUse,
                    block: PromptBlock::new("Instructions", &instruction_parts.join("\n\n")),
                    allocated_tokens: 0,
                    actual_tokens: 0,
                    source: SectionSource::Computed("instruction profiles".to_string()),
                    included: true,
                    shrinkable: PromptSection::ToolUse.shrinkable(),
                });
            }
        }

        sections
    }

    /// Build the skills content string based on the disclosure mode.
    fn _build_skills_content(&self, _skill_names: Option<&[String]>) -> String {
        let effective_disclosure = if self.skill_disclosure == "eager" {
            "eager"
        } else if self.skill_disclosure == "xml"
            || (!self.skill_disclosure.is_empty() && self.skill_disclosure != "compact")
        {
            "xml"
        } else {
            "compact"
        };

        let mut parts: Vec<String> = Vec::new();

        match effective_disclosure {
            "eager" => {
                let always_skills = self.skills.get_always_skills();
                if !always_skills.is_empty() {
                    let always_content = self.skills.load_skills_for_context(&always_skills);
                    if !always_content.is_empty() {
                        parts.push(format!("# Active Skills\n\n{}", always_content));
                    }
                }
                let skills_summary = self.skills.build_skills_summary();
                if !skills_summary.is_empty() {
                    parts.push(format!(
                        "The following skills extend your capabilities. \
                         To use a skill, read its SKILL.md file using the read_file tool.\n\
                         Skills with available=\"false\" need dependencies installed first \
                         - you can try installing them with apt/brew.\n\n\
                         {}",
                        skills_summary
                    ));
                }
            }
            "xml" => {
                let skills_summary = self.skills.build_skills_summary();
                if !skills_summary.is_empty() {
                    parts.push(format!(
                        "The following skills extend your capabilities. \
                         Use the read_skill tool to load a skill's full instructions.\n\
                         Skills with available=\"false\" need dependencies installed first \
                         - you can try installing them with apt/brew.\n\n\
                         {}",
                        skills_summary
                    ));
                }
            }
            _ => {
                let index = self.skills.build_compact_index();
                if !index.is_empty() {
                    parts.push(index);
                }
            }
        }

        let combined = parts.join("\n\n");
        if combined.is_empty() {
            String::new()
        } else {
            Self::_truncate_to_budget_head(&combined, self.skills_budget)
        }
    }

    /// Build the complete message list for an LLM call.
    ///
    /// For cloud models, delegates to `CloudAssembler` for system+developer
    /// message construction with budget-aware section ordering.
    /// For local models, delegates to `build_local_system_prompt()`.
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
        use crate::agent::prompt_contract::{AssemblyContext, CloudAssembler, PromptAssembler};

        let mut messages: Vec<Value> = Vec::new();

        // Determine whether to use the `developer` role for injected context.
        // Local prompt mode is driven by runtime locality. Keep the model-name
        // prefix fallback for older callers/tests.
        let is_local = self.local_prompt_mode || self.model_name.starts_with("local:");

        if is_local {
            let system_prompt = self.build_local_system_prompt(
                skill_names,
                channel,
                chat_id,
                is_voice_message,
                detected_language,
                &[],
            );
            messages.push(json!({"role": "system", "content": system_prompt}));
        } else {
            // Cloud model: collect static sections and use CloudAssembler for
            // budget-aware system+developer message construction.
            let static_sections = self.collect_static_sections(
                skill_names,
                channel,
                chat_id,
                is_voice_message,
                detected_language,
            );

            let context_window = if self.system_prompt_cap > 0 {
                // Reverse-engineer context window from the cap (cap = 40% of window).
                (self.system_prompt_cap as f64 / 0.4).round() as usize
            } else {
                // Default: 128K context window.
                128_000
            };

            let ctx = AssemblyContext {
                context_window,
                system_prompt_cap_pct: 0.4,
                sections: static_sections,
            };
            let result = CloudAssembler.assemble(&ctx);

            messages.push(json!({"role": "system", "content": result.system_content}));
            if !result.developer_content.is_empty() {
                messages.push(json!({"role": "developer", "content": result.developer_content}));
            }
        }

        // History.
        messages.extend(history.iter().cloned());

        // Current user message (with optional image attachments).
        let user_content = Self::_build_user_content(current_message, media);
        messages.push(json!({"role": "user", "content": user_content}));

        messages
    }

    /// Inject pre-fetched runtime sections into the developer message.
    ///
    /// For cloud prompts, runtime sections (working memory, daily notes,
    /// subagent status, bulletin) are rendered and appended to the existing
    /// `developer` role message. If no developer message exists, one is created.
    ///
    /// This replaces the former `append_to_system_prompt()` calls in
    /// `prepare_context.rs` -- all runtime content now flows through typed
    /// `SectionEntry` values rather than ad-hoc string concatenation.
    pub fn inject_runtime_sections(
        &self,
        messages: &mut Vec<Value>,
        sections: &[crate::agent::prompt_contract::SectionEntry],
    ) {
        if sections.is_empty() {
            return;
        }

        // Render each section's block content with separators.
        let rendered_parts: Vec<String> = sections
            .iter()
            .filter(|s| s.included)
            .map(|s| s.block.render())
            .filter(|r| !r.is_empty())
            .collect();

        if rendered_parts.is_empty() {
            return;
        }

        let suffix = rendered_parts.join("\n\n---\n\n");

        // Find and extend the developer message, or create one.
        if let Some(dev_msg) = messages.iter_mut().find(|m| m["role"] == "developer") {
            let existing = dev_msg["content"].as_str().unwrap_or("").to_string();
            if existing.is_empty() {
                dev_msg["content"] = Value::String(suffix);
            } else {
                dev_msg["content"] = Value::String(format!("{}\n\n---\n\n{}", existing, suffix));
            }
        } else {
            // No developer message exists -- insert one after the system message.
            let dev_msg = json!({"role": "developer", "content": suffix});
            // Insert at position 1 (after system message, before history).
            if messages.len() > 1 {
                messages.insert(1, dev_msg);
            } else {
                messages.push(dev_msg);
            }
        }
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

    /// Core identity section including current time, time awareness, and workspace info.
    fn _get_identity(&self) -> String {
        let now = Local::now();
        let workspace_path = Self::display_path(
            &self
                .workspace
                .canonicalize()
                .unwrap_or_else(|_| self.workspace.clone()),
        );
        let home_dir = dirs::home_dir()
            .map(|p| Self::display_path(&p))
            .unwrap_or_else(|| "~".to_string());
        let cwd = std::env::current_dir()
            .map(|p| Self::display_path(&p))
            .unwrap_or_else(|_| ".".to_string());

        let model_section = if self.model_name.is_empty() {
            String::new()
        } else if let Some(gguf_name) = self.model_name.strip_prefix("local:") {
            format!(
                "\n\n## Model\nYou are running locally via LM Studio. \
                 Your model file: {}. You are NOT Claude or any cloud AI. \
                 Respond as nanobot powered by this local model.\n\n\
                 ## Local Mode Constraints\n\
                 - Cloud models (claude-*, gpt-*, etc.) are **not available** in this session.\n\
                 - When spawning subagents, omit the `model` parameter to use the local model, \
                 or specify it by name: `{}`.\n\
                 - Do not request cloud-specific model tiers (haiku, sonnet, opus) — \
                 they will automatically resolve to the local model.",
                gguf_name, gguf_name
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
Home: {home_dir}
Working directory: {cwd}
Workspace: {workspace_path}

## Rules
1. ALWAYS use tools. NEVER guess file contents, command output, or system state. Run the command, read the file, check with a tool. If you don't know, use a tool to find out — do not make up an answer.
2. Be concise (1-5 sentences) unless asked for detail.
3. Reply directly for conversation; use 'message' tool only for chat channels.

## File Operations
The user's project is at the **working directory** ({cwd}). Always look here first.
Only access files outside it when the user explicitly asks.
Your workspace ({workspace_path}) is for your internal state (memory, skills, config) — not the user's project.
Use absolute paths or paths relative to the working directory.{delegation_hint}

## Memory
Working Memory is injected automatically (session state). Long-term facts: {workspace_path}/memory/MEMORY.md.
Use `recall` to search all memory (sessions, facts, archives).

If you see a [PRIORITY USER MESSAGE], acknowledge it and adjust your approach — it takes precedence."#
        )
    }

    fn _get_local_identity(&self) -> String {
        let now = Local::now();
        let workspace_path = Self::display_path(
            &self
                .workspace
                .canonicalize()
                .unwrap_or_else(|_| self.workspace.clone()),
        );
        let cwd = std::env::current_dir()
            .map(|p| Self::display_path(&p))
            .unwrap_or_else(|_| ".".to_string());

        let model_line = if self.model_name.is_empty() {
            "Model: local".to_string()
        } else {
            format!("Model: {}", self.model_name)
        };

        let local_mode_hint = if self.model_name.starts_with("local:")
            || self.model_name.starts_with("mlx:")
        {
            "\n- This is a local session. Do not claim to be Claude, GPT, or another cloud model."
        } else {
            ""
        };

        format!(
            r#"# nanobot

You are nanobot, a local tool-using assistant.
Time: {now}
{model_line}

## IMPORTANT: Directories
**Project directory: {cwd}** ← The user's code lives HERE. Always start here.
Internal workspace: {workspace_path} ← Your memory/skills/config only. NOT the user's project.

When running `exec`, `list_dir`, `read_file`, or `write_file`, use paths under the **project directory** ({cwd}).
Only access the workspace for `recall`, `remember`, or `read_skill` — never for the user's files.

Rules:
- Use tools to inspect files, commands, and memory. Never invent outputs, paths, or results.
- Keep replies short unless the user asks for detail.{local_mode_hint}"#
        )
    }

    fn display_path(path: &Path) -> String {
        let rendered = path.to_string_lossy().to_string();
        if let Some(home) = dirs::home_dir() {
            let home = home.to_string_lossy().to_string();
            if rendered == home {
                return "~".to_string();
            }
            if let Some(stripped) = rendered.strip_prefix(&(home.clone() + "/")) {
                return format!("~/{}", stripped);
            }
        }
        rendered
    }

    fn build_local_static_blocks(
        &self,
        skill_names: Option<&[String]>,
        channel: Option<&str>,
        chat_id: Option<&str>,
        is_voice_message: bool,
        detected_language: Option<&str>,
    ) -> Vec<PromptBlock> {
        let mut blocks = Vec::new();

        if self.provenance_enabled {
            blocks.push(PromptBlock::new(
                "Verification",
                "Use tool output as ground truth. Do not invent file contents, command output, or paths.",
            ));
        }

        // Load bootstrap file CONTENT (SOUL.md, USER.md, TOOLS.md, etc.)
        // into the prompt so the agent knows its identity and user preferences.
        // Previously this only listed file names as "available via read_file"
        // which meant local agents started with zero context.
        let bootstrap = self._load_bootstrap_files_within_budget(self.bootstrap_budget);
        if !bootstrap.is_empty() {
            blocks.push(PromptBlock::new("Workspace Context", bootstrap));
        }

        blocks.push(PromptBlock::new(
            "On-Demand Context",
            concat!(
                "Use `recall` to load memory only when needed. ",
                "Use `read_skill __list__` to discover skills and `read_skill <name>` to load one."
            ),
        ));

        let skill_index = self.skills.build_name_index(12);
        if !skill_index.is_empty() {
            blocks.push(PromptBlock::new("Skills", skill_index));
        }

        if let Some(names) = skill_names {
            if !names.is_empty() {
                let requested = self.skills.load_skills_for_context(names);
                if !requested.is_empty() {
                    blocks.push(PromptBlock::new("Requested Skills", requested));
                }
            }
        }

        let session_meta =
            _session_metadata_suffix(channel, chat_id, is_voice_message, detected_language);
        if !session_meta.trim().is_empty() {
            blocks.push(PromptBlock::new(String::new(), session_meta));
        }

        blocks.push(PromptBlock::new(
            "Tool Use",
            concat!(
                "Use tools only when they change the answer. ",
                "After tool results, answer directly. ",
                "One tool call at a time."
            ),
        ));

        blocks
    }

    fn assemble_local_prompt_report(
        &self,
        prefix: &str,
        static_blocks: &[PromptBlock],
        runtime_blocks: &[PromptBlock],
    ) -> PromptAssemblyReport {
        let mut assembled = prefix.trim().to_string();
        let max_tokens = if self.system_prompt_cap > 0 {
            self.system_prompt_cap
        } else {
            usize::MAX
        };
        let mut blocks = vec![PromptBlockReport {
            kind: PromptBlockKind::Prefix,
            title: "Identity".to_string(),
            tokens: TokenBudget::estimate_str_tokens(&assembled),
            included: true,
            allocated_tokens: 0,
            source: String::new(),
        }];

        for (kind, block) in static_blocks
            .iter()
            .map(|b| (PromptBlockKind::Static, b))
            .chain(runtime_blocks.iter().map(|b| (PromptBlockKind::Runtime, b)))
        {
            let rendered = block.render();
            if rendered.is_empty() {
                continue;
            }
            let block_tokens = TokenBudget::estimate_str_tokens(&rendered);
            let candidate = format!("{}\n\n---\n\n{}", assembled, rendered);
            let included = max_tokens == usize::MAX
                || TokenBudget::estimate_str_tokens(&candidate) <= max_tokens;
            if included {
                assembled = candidate;
            }
            blocks.push(PromptBlockReport {
                kind,
                title: block.report_title(),
                tokens: block_tokens,
                included,
                allocated_tokens: 0,
                source: String::new(),
            });
        }

        PromptAssemblyReport {
            total_tokens: TokenBudget::estimate_str_tokens(&assembled),
            cap_tokens: if max_tokens == usize::MAX {
                None
            } else {
                Some(max_tokens)
            },
            prompt: assembled,
            blocks,
        }
    }

    /// Load bootstrap files within budget using progressive line-level inclusion.
    ///
    /// If a file fits fully within the remaining budget it is included in full.
    /// If a file does not fit fully, as many lines as will fit are included and
    /// a truncation note is appended. Files that cannot fit even a single line
    /// are skipped entirely and listed so the model can fetch them with
    /// `read_file`.
    ///
    /// Priority follows `BOOTSTRAP_FILES` ordering: earlier files and earlier
    /// lines within a file always win. Cloud models with generous budgets see
    /// the same result as before (all files fit fully).
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
                // File fits fully — include as-is.
                included.push(section);
                remaining = remaining.saturating_sub(cost);
            } else {
                // File does not fit fully — include as many lines as possible.
                let header = format!("## {}\n\n", filename);
                let header_cost = TokenBudget::estimate_str_tokens(&header);
                if header_cost >= remaining {
                    // Not even the header fits; skip entirely.
                    skipped.push(filename);
                    continue;
                }
                remaining = remaining.saturating_sub(header_cost);

                let mut partial_lines: Vec<&str> = Vec::new();
                for line in content.lines() {
                    let line_cost = TokenBudget::estimate_str_tokens(line) + 1; // +1 for newline
                    if line_cost > remaining {
                        break;
                    }
                    partial_lines.push(line);
                    remaining = remaining.saturating_sub(line_cost);
                }

                if partial_lines.is_empty() {
                    skipped.push(filename);
                } else {
                    included.push(format!("{}{}", header, partial_lines.join("\n")));
                    skipped.push(filename); // full file still available via read_file
                }
            }
        }

        // Tell the model about skipped/truncated files.
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

    fn _available_bootstrap_files(&self) -> Vec<String> {
        BOOTSTRAP_FILES
            .iter()
            .filter_map(|filename| {
                let path = self.workspace.join(filename);
                if path.exists() {
                    Some((*filename).to_string())
                } else {
                    None
                }
            })
            .collect()
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

/// Build the session metadata suffix appended to the system/developer prompt.
///
/// Returns a string starting with `"\n\n"` when channel and chat_id are both present,
/// or an empty string when either is `None`.
/// Appends voice mode instructions when appropriate.
fn _session_metadata_suffix(
    channel: Option<&str>,
    chat_id: Option<&str>,
    is_voice: bool,
    detected_language: Option<&str>,
) -> String {
    let (Some(ch), Some(cid)) = (channel, chat_id) else {
        return String::new();
    };
    let mut text = format!("\n\n## Current Session\nChannel: {}\nChat ID: {}", ch, cid);
    if ch == "voice" || is_voice {
        text.push_str(&_voice_mode_instructions(detected_language));
    }
    text
}

/// Build the voice mode instruction block appended to the system/developer prompt.
///
/// `detected_language` is an ISO 639-1 code (e.g. `"en"`, `"es"`) or `None`.
/// The returned string always starts with `"\n\n"` so callers can push it directly.
fn _voice_mode_instructions(detected_language: Option<&str>) -> String {
    let mut text = concat!(
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
    )
    .to_string();
    if let Some(lang) = detected_language {
        if lang == "en" {
            text.push_str("\n- The user is speaking in English. You MUST respond in English.");
        } else {
            text.push_str(&format!(
                "\n- The user is speaking in {}. You MUST respond in the same language.",
                lang_code_to_name(lang)
            ));
        }
    }
    text
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

    #[test]
    fn test_bootstrap_partial_inclusion_tight_budget() {
        // Budget is large enough for AGENTS.md header + a few lines but not
        // the whole file.  Verify that the first lines are present.
        let tmp = TempDir::new().unwrap();
        let long_content = (0..50)
            .map(|i| format!("Line {}: some important directive here\n", i))
            .collect::<String>();
        fs::write(tmp.path().join("AGENTS.md"), &long_content).unwrap();
        let mut cb = ContextBuilder::new(tmp.path());
        // Budget: fits header (~6 tok) + a handful of lines but not all 50.
        cb.bootstrap_budget = 80;
        let result = cb._load_bootstrap_files_within_budget(80);
        // First line must be present.
        assert!(
            result.contains("Line 0:"),
            "first line should be present in partial inclusion"
        );
        // Last line must be absent (budget exhausted before line 49).
        assert!(
            !result.contains("Line 49:"),
            "last line should be absent when budget is tight"
        );
        // The AGENTS.md section header should be present.
        assert!(
            result.contains("AGENTS.md"),
            "file header should appear in partial section"
        );
    }

    #[test]
    fn test_bootstrap_full_inclusion_generous_budget() {
        // Generous budget: every file must be fully included (no truncation).
        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("AGENTS.md"), "Agent directive alpha").unwrap();
        fs::write(tmp.path().join("SOUL.md"), "Soul directive beta").unwrap();
        fs::write(tmp.path().join("USER.md"), "User directive gamma").unwrap();
        let cb = ContextBuilder::new(tmp.path());
        let result = cb._load_bootstrap_files_within_budget(10_000);
        assert!(
            result.contains("Agent directive alpha"),
            "AGENTS.md fully included"
        );
        assert!(
            result.contains("Soul directive beta"),
            "SOUL.md fully included"
        );
        assert!(
            result.contains("User directive gamma"),
            "USER.md fully included"
        );
        // No skipped-file note should appear since all files fit.
        assert!(
            !result.contains("read_file"),
            "no skipped files when budget is generous"
        );
    }

    #[test]
    fn test_bootstrap_critical_directive_survives_tight_budget() {
        // A critical directive on line 2 of USER.md must survive when the
        // budget is too tight to include the whole file.
        let tmp = TempDir::new().unwrap();
        let user_content = "# User preferences\nRespond in the user's language\n".to_string()
            + &"filler line\n".repeat(100);
        fs::write(tmp.path().join("USER.md"), &user_content).unwrap();
        let mut cb = ContextBuilder::new(tmp.path());
        // Budget: fits the header + first two lines but not the 100 filler lines.
        cb.bootstrap_budget = 60;
        let result = cb._load_bootstrap_files_within_budget(60);
        assert!(
            result.contains("Respond in the user's language"),
            "critical directive on line 2 must survive tight budget"
        );
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
        // Channel/chat_id goes into the developer message for cloud models.
        // Search all messages for the session metadata.
        let all_content: String = messages
            .iter()
            .filter_map(|m| m["content"].as_str())
            .collect::<Vec<_>>()
            .join(" ");
        assert!(all_content.contains("Channel: telegram"));
        assert!(all_content.contains("Chat ID: 12345"));
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
        let b64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=".repeat(20);
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
        assert!(!result.contains("SKILL 499"), "end should be dropped");
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

    // ----- developer role -----

    #[test]
    fn test_developer_role_in_messages_cloud() {
        // Cloud model (no "local:" prefix): memory/skills go into `developer` role,
        // identity goes into `system` role.
        let tmp = TempDir::new().unwrap();
        let memory_dir = tmp.path().join("memory");
        fs::create_dir_all(&memory_dir).unwrap();
        fs::write(memory_dir.join("MEMORY.md"), "User prefers dark mode").unwrap();

        let mut cb = ContextBuilder::new(tmp.path());
        cb.model_name = "claude-opus-4-6".to_string(); // cloud model

        let messages = cb.build_messages(&[], "hello", None, None, None, None, false, None);

        // There should be a `system` message and a `developer` message (plus user).
        let system_msg = messages.iter().find(|m| m["role"] == "system").unwrap();
        let developer_msg = messages.iter().find(|m| m["role"] == "developer").unwrap();

        // Identity is in `system`.
        let system_content = system_msg["content"].as_str().unwrap();
        assert!(
            system_content.contains("You are nanobot"),
            "system message should contain identity"
        );
        assert!(
            !system_content.contains("User prefers dark mode"),
            "memory must NOT be in system message"
        );

        // Memory is in `developer`.
        let dev_content = developer_msg["content"].as_str().unwrap();
        assert!(
            dev_content.contains("User prefers dark mode"),
            "memory should be in developer message"
        );
        assert!(
            !dev_content.contains("You are nanobot"),
            "identity must NOT be in developer message"
        );
    }

    #[test]
    fn test_developer_role_local_model_uses_system_only() {
        // Local model: everything goes into a single lean `system` message.
        let tmp = TempDir::new().unwrap();
        let memory_dir = tmp.path().join("memory");
        fs::create_dir_all(&memory_dir).unwrap();
        fs::write(memory_dir.join("MEMORY.md"), "User prefers dark mode").unwrap();

        let mut cb = ContextBuilder::new(tmp.path());
        cb.model_name = "local:Qwen3-8B.gguf".to_string(); // local model
        cb.local_prompt_mode = true;

        let messages = cb.build_messages(&[], "hello", None, None, None, None, false, None);

        // No `developer` role message for local models.
        assert!(
            messages.iter().all(|m| m["role"] != "developer"),
            "local model should not emit a developer message"
        );

        // Identity stays in system, but memory/skills become on-demand.
        let system_msg = messages.iter().find(|m| m["role"] == "system").unwrap();
        let system_content = system_msg["content"].as_str().unwrap();
        assert!(system_content.contains("You are nanobot"));
        assert!(!system_content.contains("User prefers dark mode"));
        assert!(system_content.contains("Use `recall` to load memory"));
        assert!(system_content.contains("read_skill __list__"));
    }

    #[test]
    fn test_local_prompt_mode_applies_to_mlx_style_models() {
        let tmp = TempDir::new().unwrap();
        let memory_dir = tmp.path().join("memory");
        fs::create_dir_all(&memory_dir).unwrap();
        fs::write(memory_dir.join("MEMORY.md"), "User prefers Rust").unwrap();

        let mut cb = ContextBuilder::new(tmp.path());
        cb.model_name = "mlx:Qwen3-8B-MLX-4bit".to_string();
        cb.local_prompt_mode = true;

        let messages = cb.build_messages(&[], "hello", None, None, None, None, false, None);

        assert!(
            messages.iter().all(|m| m["role"] != "developer"),
            "MLX local mode should use the lean local prompt path"
        );

        let system_msg = messages.iter().find(|m| m["role"] == "system").unwrap();
        let system_content = system_msg["content"].as_str().unwrap();
        assert!(system_content.contains("Model: mlx:Qwen3-8B-MLX-4bit"));
        assert!(system_content.contains("local tool-using assistant"));
        assert!(!system_content.contains("User prefers Rust"));
        assert!(system_content.contains("read_skill __list__"));
    }

    #[test]
    fn test_local_prompt_mode_uses_on_demand_workspace_context() {
        let tmp = TempDir::new().unwrap();
        let memory_dir = tmp.path().join("memory");
        let skill_dir = tmp.path().join("skills").join("radio");
        fs::create_dir_all(&memory_dir).unwrap();
        fs::create_dir_all(&skill_dir).unwrap();

        fs::write(
            tmp.path().join("AGENTS.md"),
            format!("Bootstrap directive\n{}\n", "VERY IMPORTANT ".repeat(400)),
        )
        .unwrap();
        fs::write(
            memory_dir.join("MEMORY.md"),
            format!("Persistent fact\n{}\n", "LONG TERM MEMORY ".repeat(300)),
        )
        .unwrap();
        fs::write(
            skill_dir.join("SKILL.md"),
            concat!(
                "---\n",
                "description: Stream and manage local radio playback with rich controls\n",
                "---\n",
                "Full skill body that should not be injected into the lean local prompt.\n"
            ),
        )
        .unwrap();

        let mut cb = ContextBuilder::new(tmp.path());
        cb.model_name = "mlx:Qwen3-8B-MLX-4bit".to_string();
        cb.local_prompt_mode = true;

        let messages = cb.build_messages(&[], "hello", None, None, None, None, false, None);
        let total_tokens = TokenBudget::estimate_tokens(&messages);
        let system_msg = messages.iter().find(|m| m["role"] == "system").unwrap();
        let system_content = system_msg["content"].as_str().unwrap();

        // Local prompt now loads bootstrap file content (within budget)
        // so the agent knows its identity. It should still stay under the
        // system_prompt_cap and include at least some bootstrap content.
        assert!(
            total_tokens < 1200,
            "local prompt with bootstrap content should stay within budget, got {} tokens",
            total_tokens
        );
        // Bootstrap content should now be included (at least partially).
        assert!(system_content.contains("Bootstrap directive"));
        assert!(system_content.contains("radio"));
        // Long-term memory is still excluded from local prompt (loaded via recall).
        assert!(!system_content.contains("LONG TERM MEMORY LONG TERM MEMORY"));
    }

    #[test]
    fn test_developer_role_no_context_no_extra_message() {
        // When there is nothing to inject (no memory, no skills, no profiles),
        // no `developer` message should be emitted even for cloud models.
        let (_tmp, mut cb) = make_context();
        cb.model_name = "claude-opus-4-6".to_string();

        let messages = cb.build_messages(&[], "hello", None, None, None, None, false, None);

        assert!(
            messages.iter().all(|m| m["role"] != "developer"),
            "no developer message when context is empty"
        );
        // Must still have a system message.
        assert!(messages.iter().any(|m| m["role"] == "system"));
    }

    // --- Integration tests: instruction profiles injected into developer context ---

    #[test]
    fn test_instruction_profiles_injected_into_developer_context() {
        let tmp = TempDir::new().unwrap();

        // Create instruction profiles YAML
        let profiles_yaml = r#"
base:
  - role: developer
    content: "Always respond in JSON format."
model_profiles:
  - pattern: "qwen*"
    messages:
      - role: developer
        content: "Use strict tool calling. Never embed tool calls in prose."
task_profiles:
  - kind: main
    messages:
      - role: developer
        content: "You are the main agent."
"#;
        let profiles_path = tmp.path().join("instructions.yaml");
        fs::write(&profiles_path, profiles_yaml).unwrap();

        let mut cb = ContextBuilder::new(tmp.path());
        cb.model_name = "qwen-2.5-coder-32b".to_string();
        cb.task_kind = "main".to_string();
        cb.instruction_profiles =
            Some(crate::agent::instructions::InstructionProfiles::load(&profiles_path).unwrap());

        let messages = cb.build_messages(&[], "hello", None, None, None, None, false, None);

        // Find the developer message
        let dev_msg = messages.iter().find(|m| m["role"] == "developer");
        assert!(
            dev_msg.is_some(),
            "Should have a developer message when profiles are set"
        );

        let dev_content = dev_msg.unwrap()["content"].as_str().unwrap();

        // Base profile should be present
        assert!(
            dev_content.contains("Always respond in JSON format"),
            "Base profile should be injected"
        );

        // Model-matched profile should be present (qwen* matches qwen-2.5-coder-32b)
        assert!(
            dev_content.contains("strict tool calling"),
            "Qwen-specific profile should be injected"
        );

        // Task-matched profile should be present (kind=main)
        assert!(
            dev_content.contains("main agent"),
            "Main task profile should be injected"
        );
    }

    #[test]
    fn test_instruction_profiles_no_match_still_injects_base() {
        let tmp = TempDir::new().unwrap();

        let profiles_yaml = r#"
base:
  - role: developer
    content: "Base instruction."
model_profiles:
  - pattern: "llama*"
    messages:
      - role: developer
        content: "Llama-specific."
"#;
        let profiles_path = tmp.path().join("instructions.yaml");
        fs::write(&profiles_path, profiles_yaml).unwrap();

        let mut cb = ContextBuilder::new(tmp.path());
        cb.model_name = "deepseek-r1".to_string(); // doesn't match llama*
        cb.task_kind = "main".to_string();
        cb.instruction_profiles =
            Some(crate::agent::instructions::InstructionProfiles::load(&profiles_path).unwrap());

        let messages = cb.build_messages(&[], "hello", None, None, None, None, false, None);

        let dev_msg = messages.iter().find(|m| m["role"] == "developer");
        assert!(
            dev_msg.is_some(),
            "Should still have developer message for base profile"
        );

        let dev_content = dev_msg.unwrap()["content"].as_str().unwrap();
        assert!(
            dev_content.contains("Base instruction"),
            "Base should be present"
        );
        assert!(
            !dev_content.contains("Llama-specific"),
            "Non-matching model profile should NOT be present"
        );
    }

    #[test]
    fn test_developer_context_without_knowledge_graph_feature() {
        // Without --features knowledge-graph, developer context should still
        // build correctly and not contain any "Knowledge Graph" section.
        let tmp = TempDir::new().unwrap();
        let memory_dir = tmp.path().join("memory");
        fs::create_dir_all(&memory_dir).unwrap();
        fs::write(memory_dir.join("MEMORY.md"), "User prefers Rust").unwrap();

        let cb = ContextBuilder::new(tmp.path());
        let ctx = cb.build_developer_context(None, None);

        assert!(
            ctx.contains("User prefers Rust"),
            "Memory should be present"
        );

        // Without the feature flag, no KG section should appear.
        // (On knowledge-graph builds, it could appear if open_default finds data.)
        #[cfg(not(feature = "knowledge-graph"))]
        assert!(
            !ctx.contains("Knowledge Graph"),
            "KG section should not appear without knowledge-graph feature"
        );
    }
}
