#![allow(dead_code)]
//! Typed prompt contract: section enum, budget tracking, and assemblers.
//!
//! Defines the canonical ordering and budget allocation for prompt sections,
//! with CloudAssembler and LocalAssembler implementations that handle
//! overflow via two-pass drop-then-shrink logic.

use crate::agent::context::{
    PromptAssemblyReport, PromptBlock, PromptBlockKind, PromptBlockReport,
};
use crate::agent::token_budget::TokenBudget;

/// Canonical prompt sections in fixed priority order.
///
/// Lower discriminant = higher priority (dropped last during overflow).
/// The two-pass overflow algorithm drops from the tail (highest discriminant)
/// first, then shrinks the lowest remaining shrinkable section.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum PromptSection {
    Identity = 0,
    Verification = 1,
    WorkspaceContext = 2,
    OnDemandContext = 3,
    Skills = 4,
    RequestedSkills = 5,
    SessionMetadata = 6,
    ToolUse = 7,
    WorkingMemory = 8,
    ToolPatterns = 9,
    RecentNotes = 10,
    BackgroundTasks = 11,
    MemoryBriefing = 12,
}

static ALL_SECTIONS: [PromptSection; 13] = [
    PromptSection::Identity,
    PromptSection::Verification,
    PromptSection::WorkspaceContext,
    PromptSection::OnDemandContext,
    PromptSection::Skills,
    PromptSection::RequestedSkills,
    PromptSection::SessionMetadata,
    PromptSection::ToolUse,
    PromptSection::WorkingMemory,
    PromptSection::ToolPatterns,
    PromptSection::RecentNotes,
    PromptSection::BackgroundTasks,
    PromptSection::MemoryBriefing,
];

impl PromptSection {
    /// Returns all 13 variants in discriminant order.
    pub fn all() -> &'static [PromptSection] {
        &ALL_SECTIONS
    }

    /// Maps this section to the legacy PromptBlockKind for backward compatibility.
    pub fn kind(&self) -> PromptBlockKind {
        match self {
            Self::Identity => PromptBlockKind::Prefix,
            Self::Verification
            | Self::WorkspaceContext
            | Self::OnDemandContext
            | Self::Skills
            | Self::RequestedSkills => PromptBlockKind::Static,
            Self::SessionMetadata
            | Self::ToolUse
            | Self::WorkingMemory
            | Self::ToolPatterns
            | Self::RecentNotes
            | Self::BackgroundTasks
            | Self::MemoryBriefing => PromptBlockKind::Runtime,
        }
    }

    /// Whether this section's content can be truncated during overflow.
    pub fn shrinkable(&self) -> bool {
        matches!(
            self,
            Self::WorkingMemory
                | Self::ToolPatterns
                | Self::RecentNotes
                | Self::MemoryBriefing
                | Self::Skills
        )
    }

    /// Default budget percentage for this section.
    ///
    /// These percentages are scaled proportionally against the actual
    /// system prompt cap (which varies by context window size).
    pub fn default_budget_pct(&self) -> f64 {
        match self {
            Self::Identity => 10.0,
            Self::Verification => 3.0,
            Self::WorkspaceContext => 8.0,
            Self::OnDemandContext => 5.0,
            Self::Skills => 10.0,
            Self::RequestedSkills => 5.0,
            Self::SessionMetadata => 3.0,
            Self::ToolUse => 8.0,
            Self::WorkingMemory => 15.0,
            Self::ToolPatterns => 8.0,
            Self::RecentNotes => 8.0,
            Self::BackgroundTasks => 5.0,
            Self::MemoryBriefing => 12.0,
        }
    }
}

/// Describes where a section's content originated.
#[derive(Debug, Clone)]
pub enum SectionSource {
    /// Hard-coded content baked into the binary.
    Static(&'static str),
    /// Loaded from a file at the given path.
    File(String),
    /// Generated at runtime (e.g., session metadata).
    Runtime(String),
    /// Computed from multiple sources.
    Computed(String),
}

/// A single section entry with budget tracking metadata.
#[derive(Debug, Clone)]
pub struct SectionEntry {
    /// Which canonical section this entry belongs to.
    pub section: PromptSection,
    /// The rendered content block.
    pub block: PromptBlock,
    /// Tokens allocated by the proportional budget.
    pub allocated_tokens: usize,
    /// Actual measured token count of the content.
    pub actual_tokens: usize,
    /// Where the content came from.
    pub source: SectionSource,
    /// Whether this section is included in the final prompt.
    pub included: bool,
    /// Whether this section can be truncated during overflow.
    pub shrinkable: bool,
}

/// Pre-fetched context bundle consumed by assemblers.
///
/// The assembler handles ordering, budgeting, and overflow -- NOT data
/// fetching. This avoids borrow checker fights with the ContextBuilder.
#[derive(Debug, Clone)]
pub struct AssemblyContext {
    /// Total context window in tokens.
    pub context_window: usize,
    /// Fraction of context window reserved for the system prompt (e.g., 0.4 cloud, 0.3 local).
    pub system_prompt_cap_pct: f64,
    /// Pre-populated section entries with content already fetched.
    pub sections: Vec<SectionEntry>,
}

/// Result of prompt assembly.
#[derive(Debug, Clone)]
pub struct AssemblyResult {
    /// System-level content (Identity for cloud, full prompt for local).
    pub system_content: String,
    /// Developer-level content (non-Identity sections for cloud, empty for local).
    pub developer_content: String,
    /// Backward-compatible assembly report.
    pub report: PromptAssemblyReport,
}

/// Assembles prompt sections into a final prompt string.
pub trait PromptAssembler {
    /// Assemble sections into a final prompt result with overflow handling.
    fn assemble(&self, ctx: &AssemblyContext) -> AssemblyResult;
}

/// Two-pass overflow enforcement.
///
/// **Pass 1 (drop):** Walk from highest discriminant (lowest priority) to lowest,
/// excluding sections until total fits within `cap`.
///
/// **Pass 2 (shrink):** If still over budget, find the last included shrinkable
/// section and truncate its content to fit.
fn enforce_budget(sections: &mut Vec<SectionEntry>, cap: usize) {
    // Measure actual tokens for each included section.
    let mut total: usize = sections
        .iter()
        .filter(|s| s.included)
        .map(|s| s.actual_tokens)
        .sum();

    if total <= cap {
        return;
    }

    // Pass 1: Drop from tail (highest discriminant = lowest priority).
    // Skip shrinkable sections on the first pass -- they get a chance to be
    // truncated in Pass 2 before being dropped entirely.
    for i in (0..sections.len()).rev() {
        if total <= cap {
            break;
        }
        if !sections[i].included || sections[i].shrinkable {
            continue;
        }
        tracing::warn!(
            "Prompt overflow: dropping section {:?} ({} tokens)",
            sections[i].section,
            sections[i].actual_tokens,
        );
        total -= sections[i].actual_tokens;
        sections[i].included = false;
    }

    // Pass 1b: If still over budget, drop shrinkable sections from tail too,
    // but leave at least one shrinkable section for Pass 2 to truncate.
    if total > cap {
        let shrinkable_count = sections.iter().filter(|s| s.included && s.shrinkable).count();
        let mut remaining_shrinkable = shrinkable_count;
        for i in (0..sections.len()).rev() {
            if total <= cap {
                break;
            }
            if !sections[i].included || !sections[i].shrinkable {
                continue;
            }
            // Keep the last shrinkable section for Pass 2.
            if remaining_shrinkable <= 1 {
                break;
            }
            tracing::warn!(
                "Prompt overflow: dropping section {:?} ({} tokens)",
                sections[i].section,
                sections[i].actual_tokens,
            );
            total -= sections[i].actual_tokens;
            sections[i].included = false;
            remaining_shrinkable -= 1;
        }
    }

    if total <= cap {
        return;
    }

    // Pass 2: Shrink the last included shrinkable section.
    if let Some(idx) = sections
        .iter()
        .enumerate()
        .rev()
        .filter(|(_, s)| s.included && s.shrinkable)
        .map(|(i, _)| i)
        .next()
    {
        let overshoot = total.saturating_sub(cap);
        let target_tokens = sections[idx].actual_tokens.saturating_sub(overshoot);
        // Estimate chars from target tokens (inverse: ~4 chars per token).
        let target_chars = target_tokens.saturating_mul(4);
        let content = sections[idx].block.content();
        let truncated_end =
            crate::utils::helpers::floor_char_boundary(content, target_chars);
        let truncated_content = &content[..truncated_end];
        let old_tokens = sections[idx].actual_tokens;
        let new_block = PromptBlock::new(
            sections[idx].block.report_title(),
            truncated_content,
        );
        let new_tokens = TokenBudget::estimate_str_tokens(&new_block.render());
        tracing::warn!(
            "Prompt overflow: shrinking section {:?} from {} to {} tokens",
            sections[idx].section,
            old_tokens,
            new_tokens,
        );
        sections[idx].block = new_block;
        sections[idx].actual_tokens = new_tokens;
    }
}

/// Build a PromptAssemblyReport from processed section entries.
fn build_report(
    sections: &[SectionEntry],
    prompt: &str,
    cap: Option<usize>,
) -> PromptAssemblyReport {
    let blocks = sections
        .iter()
        .map(|s| PromptBlockReport {
            kind: s.section.kind(),
            title: s.block.report_title(),
            tokens: s.actual_tokens,
            included: s.included,
            allocated_tokens: s.allocated_tokens,
            source: match &s.source {
                SectionSource::Static(label) => label.to_string(),
                SectionSource::File(path) => path.clone(),
                SectionSource::Runtime(desc) => desc.clone(),
                SectionSource::Computed(desc) => desc.clone(),
            },
        })
        .collect();

    PromptAssemblyReport {
        prompt: prompt.to_string(),
        total_tokens: TokenBudget::estimate_str_tokens(prompt),
        cap_tokens: cap,
        blocks,
    }
}

/// Prepare sections: sort by discriminant, measure tokens, set allocated budgets.
fn prepare_sections(sections: &mut Vec<SectionEntry>, cap: usize) {
    sections.sort_by_key(|s| s.section);
    for entry in sections.iter_mut() {
        let rendered = entry.block.render();
        entry.actual_tokens = if rendered.is_empty() {
            entry.included = false;
            0
        } else {
            TokenBudget::estimate_str_tokens(&rendered)
        };
        entry.allocated_tokens =
            ((entry.section.default_budget_pct() / 100.0) * cap as f64).round() as usize;

        tracing::debug!(
            "Section {:?}: {} tokens, source={:?}, included={}",
            entry.section,
            entry.actual_tokens,
            entry.source,
            entry.included,
        );
    }
}

/// Assembler for cloud providers (Anthropic, OpenAI, etc.).
///
/// Identity section goes into `system_content`. All other included sections
/// are concatenated into `developer_content` with `---` separators.
pub struct CloudAssembler;

impl PromptAssembler for CloudAssembler {
    fn assemble(&self, ctx: &AssemblyContext) -> AssemblyResult {
        let cap = ((ctx.context_window as f64) * ctx.system_prompt_cap_pct).round() as usize;
        let mut sections = ctx.sections.clone();
        prepare_sections(&mut sections, cap);
        enforce_budget(&mut sections, cap);

        let system_content = sections
            .iter()
            .filter(|s| s.included && s.section == PromptSection::Identity)
            .map(|s| s.block.render())
            .collect::<Vec<_>>()
            .join("");

        let developer_content = sections
            .iter()
            .filter(|s| s.included && s.section != PromptSection::Identity)
            .map(|s| s.block.render())
            .filter(|r| !r.is_empty())
            .collect::<Vec<_>>()
            .join("\n\n---\n\n");

        let full_prompt = if developer_content.is_empty() {
            system_content.clone()
        } else {
            format!("{}\n\n---\n\n{}", system_content, developer_content)
        };

        let report = build_report(&sections, &full_prompt, Some(cap));

        AssemblyResult {
            system_content,
            developer_content,
            report,
        }
    }
}

/// Assembler for local models (LM Studio, mlx-lm, etc.).
///
/// All included sections are concatenated into a single `system_content`
/// string with `---` separators. `developer_content` is always empty.
pub struct LocalAssembler;

impl PromptAssembler for LocalAssembler {
    fn assemble(&self, ctx: &AssemblyContext) -> AssemblyResult {
        let cap = ((ctx.context_window as f64) * ctx.system_prompt_cap_pct).round() as usize;
        let mut sections = ctx.sections.clone();
        prepare_sections(&mut sections, cap);
        enforce_budget(&mut sections, cap);

        let system_content = sections
            .iter()
            .filter(|s| s.included)
            .map(|s| s.block.render())
            .filter(|r| !r.is_empty())
            .collect::<Vec<_>>()
            .join("\n\n---\n\n");

        let report = build_report(&sections, &system_content, Some(cap));

        AssemblyResult {
            system_content,
            developer_content: String::new(),
            report,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::context::PromptBlockKind;

    #[test]
    fn test_prompt_section_ord_ordering() {
        // Identity must be less than Verification, which must be less than WorkspaceContext, etc.
        assert!(PromptSection::Identity < PromptSection::Verification);
        assert!(PromptSection::Verification < PromptSection::WorkspaceContext);
        assert!(PromptSection::WorkspaceContext < PromptSection::OnDemandContext);
        assert!(PromptSection::OnDemandContext < PromptSection::Skills);
        assert!(PromptSection::Skills < PromptSection::RequestedSkills);
        assert!(PromptSection::RequestedSkills < PromptSection::SessionMetadata);
        assert!(PromptSection::SessionMetadata < PromptSection::ToolUse);
        assert!(PromptSection::ToolUse < PromptSection::WorkingMemory);
        assert!(PromptSection::WorkingMemory < PromptSection::ToolPatterns);
        assert!(PromptSection::ToolPatterns < PromptSection::RecentNotes);
        assert!(PromptSection::RecentNotes < PromptSection::BackgroundTasks);
        assert!(PromptSection::BackgroundTasks < PromptSection::MemoryBriefing);
    }

    #[test]
    fn test_prompt_section_all_returns_13_variants() {
        let all = PromptSection::all();
        assert_eq!(all.len(), 13);
        // Must be in order
        for i in 1..all.len() {
            assert!(all[i - 1] < all[i], "all() not sorted at index {}", i);
        }
    }

    #[test]
    fn test_prompt_section_kind_mapping() {
        assert_eq!(PromptSection::Identity.kind(), PromptBlockKind::Prefix);
        // Static sections
        assert_eq!(
            PromptSection::WorkspaceContext.kind(),
            PromptBlockKind::Static
        );
        assert_eq!(
            PromptSection::OnDemandContext.kind(),
            PromptBlockKind::Static
        );
        assert_eq!(PromptSection::Skills.kind(), PromptBlockKind::Static);
        assert_eq!(
            PromptSection::RequestedSkills.kind(),
            PromptBlockKind::Static
        );
        assert_eq!(PromptSection::Verification.kind(), PromptBlockKind::Static);
        // Runtime sections
        assert_eq!(
            PromptSection::SessionMetadata.kind(),
            PromptBlockKind::Runtime
        );
        assert_eq!(PromptSection::ToolUse.kind(), PromptBlockKind::Runtime);
        assert_eq!(
            PromptSection::WorkingMemory.kind(),
            PromptBlockKind::Runtime
        );
        assert_eq!(
            PromptSection::ToolPatterns.kind(),
            PromptBlockKind::Runtime
        );
        assert_eq!(PromptSection::RecentNotes.kind(), PromptBlockKind::Runtime);
        assert_eq!(
            PromptSection::BackgroundTasks.kind(),
            PromptBlockKind::Runtime
        );
        assert_eq!(
            PromptSection::MemoryBriefing.kind(),
            PromptBlockKind::Runtime
        );
    }

    #[test]
    fn test_prompt_section_shrinkable() {
        // Shrinkable sections
        assert!(PromptSection::WorkingMemory.shrinkable());
        assert!(PromptSection::ToolPatterns.shrinkable());
        assert!(PromptSection::RecentNotes.shrinkable());
        assert!(PromptSection::MemoryBriefing.shrinkable());
        assert!(PromptSection::Skills.shrinkable());
        // Non-shrinkable
        assert!(!PromptSection::Identity.shrinkable());
        assert!(!PromptSection::Verification.shrinkable());
        assert!(!PromptSection::OnDemandContext.shrinkable());
        assert!(!PromptSection::ToolUse.shrinkable());
        assert!(!PromptSection::SessionMetadata.shrinkable());
    }

    #[test]
    fn test_prompt_section_default_budget_pct_sums_to_100_or_less() {
        let all = PromptSection::all();
        let total: f64 = all.iter().map(|s| s.default_budget_pct()).sum();
        assert!(
            total <= 100.0 + f64::EPSILON,
            "budget percentages sum to {} which exceeds 100",
            total
        );
        // Each section should have a non-negative percentage
        for s in all {
            assert!(
                s.default_budget_pct() >= 0.0,
                "{:?} has negative budget pct",
                s
            );
        }
    }

    // --- Task 2 tests: Assembler trait, CloudAssembler, LocalAssembler, overflow ---

    /// Helper to create a SectionEntry for tests.
    fn make_entry(section: PromptSection, title: &str, content: &str) -> SectionEntry {
        SectionEntry {
            section,
            block: PromptBlock::new(title, content),
            allocated_tokens: 0,
            actual_tokens: 0,
            source: SectionSource::Runtime("test".to_string()),
            included: true,
            shrinkable: section.shrinkable(),
        }
    }

    fn make_ctx(context_window: usize, cap_pct: f64, sections: Vec<SectionEntry>) -> AssemblyContext {
        AssemblyContext {
            context_window,
            system_prompt_cap_pct: cap_pct,
            sections,
        }
    }

    #[test]
    fn test_local_assembler_concatenates_with_separators() {
        let sections = vec![
            make_entry(PromptSection::Identity, "Identity", "I am nanobot"),
            make_entry(PromptSection::Verification, "Verification", "Check stuff"),
        ];
        let ctx = make_ctx(128_000, 0.3, sections);
        let result = LocalAssembler.assemble(&ctx);
        assert!(result.system_content.contains("I am nanobot"));
        assert!(result.system_content.contains("Check stuff"));
        assert!(result.system_content.contains("---"));
        assert!(result.developer_content.is_empty());
    }

    #[test]
    fn test_cloud_assembler_splits_identity_and_developer() {
        let sections = vec![
            make_entry(PromptSection::Identity, "Identity", "I am nanobot"),
            make_entry(PromptSection::Verification, "Verification", "Check stuff"),
            make_entry(PromptSection::WorkingMemory, "Working Memory", "Remember this"),
        ];
        let ctx = make_ctx(128_000, 0.4, sections);
        let result = CloudAssembler.assemble(&ctx);
        // Identity goes to system_content
        assert!(result.system_content.contains("I am nanobot"));
        // Other sections go to developer_content
        assert!(result.developer_content.contains("Check stuff"));
        assert!(result.developer_content.contains("Remember this"));
        // Identity should NOT be in developer_content
        assert!(!result.developer_content.contains("I am nanobot"));
    }

    #[test]
    fn test_overflow_drops_lowest_priority_first() {
        // BackgroundTasks (11, non-shrinkable) should be dropped before
        // SessionMetadata (6, non-shrinkable) because it has a higher discriminant.
        let mut sm = make_entry(PromptSection::SessionMetadata, "SM", "session info");
        sm.shrinkable = false; // explicit
        let mut bg = make_entry(PromptSection::BackgroundTasks, "BG", &"y".repeat(5000));
        bg.shrinkable = false;
        let sections = vec![
            make_entry(PromptSection::Identity, "Identity", "Core identity text"),
            sm,
            bg,
        ];
        // 1000 * 0.3 = 300 token cap -- Identity + SM fit, BG doesn't
        let ctx = make_ctx(1000, 0.3, sections);
        let result = LocalAssembler.assemble(&ctx);
        // Identity should be included
        assert!(result.system_content.contains("Core identity text"));
        // SM should be included (lower discriminant = higher priority)
        assert!(result.system_content.contains("session info"));
        // BG should be dropped (highest discriminant among non-shrinkable)
        let bg_block = result.report.blocks.iter().find(|b| b.title == "BG").unwrap();
        assert!(!bg_block.included, "BackgroundTasks should be dropped");
    }

    #[test]
    fn test_overflow_shrinks_after_drop_fails() {
        // One large shrinkable section that exceeds the budget on its own.
        // After Pass 1 tries to drop (but nothing else to drop), Pass 2 should shrink it.
        let big_content = "word ".repeat(2000); // ~2000 tokens worth
        let sections = vec![
            make_entry(PromptSection::Identity, "", "id"),
            make_entry(PromptSection::WorkingMemory, "WM", &big_content),
        ];
        // 200 token cap -- Identity is ~1 token, WM needs to be shrunk
        let ctx = make_ctx(1000, 0.2, sections);
        let result = LocalAssembler.assemble(&ctx);
        // Working memory should still be included (shrunk, not dropped)
        let wm_block = result.report.blocks.iter().find(|b| b.title == "WM").unwrap();
        assert!(wm_block.included, "WorkingMemory should be shrunk, not dropped");
        // Its token count should be less than original
        assert!(wm_block.tokens < 2000, "should have been shrunk, got {} tokens", wm_block.tokens);
    }

    #[test]
    fn test_proportional_budgets_scale_with_context_window() {
        let sections_small = vec![
            make_entry(PromptSection::Identity, "Identity", "I am nanobot"),
            make_entry(PromptSection::WorkingMemory, "WM", "some memory"),
        ];
        let sections_large = sections_small.clone();

        let ctx_small = make_ctx(4_000, 0.3, sections_small);  // 1200 token cap
        let ctx_large = make_ctx(128_000, 0.4, sections_large); // 51200 token cap

        let result_small = LocalAssembler.assemble(&ctx_small);
        let result_large = LocalAssembler.assemble(&ctx_large);

        // Both should report allocated_tokens, and large should have bigger allocations
        let wm_small = result_small.report.blocks.iter().find(|b| b.title == "WM").unwrap();
        let wm_large = result_large.report.blocks.iter().find(|b| b.title == "WM").unwrap();
        assert!(
            wm_large.allocated_tokens > wm_small.allocated_tokens,
            "larger context window should produce larger allocated budget: small={} large={}",
            wm_small.allocated_tokens,
            wm_large.allocated_tokens,
        );
    }

    #[test]
    fn test_empty_sections_excluded() {
        let sections = vec![
            make_entry(PromptSection::Identity, "Identity", "I am nanobot"),
            make_entry(PromptSection::Verification, "Verification", ""),
            make_entry(PromptSection::WorkingMemory, "WM", "   "),
        ];
        let ctx = make_ctx(128_000, 0.4, sections);
        let result = CloudAssembler.assemble(&ctx);
        // Empty sections should not appear in developer_content
        assert!(!result.developer_content.contains("Verification"));
        assert!(!result.developer_content.contains("WM"));
        // Report should mark them as not included
        let ver = result.report.blocks.iter().find(|b| b.title == "Verification").unwrap();
        assert!(!ver.included, "empty section should be excluded");
    }

    #[test]
    fn test_assembly_report_has_correct_block_counts() {
        let sections = vec![
            make_entry(PromptSection::Identity, "Identity", "I am nanobot"),
            make_entry(PromptSection::Verification, "Verification", "Check stuff"),
            make_entry(PromptSection::WorkingMemory, "WM", "Remember this"),
        ];
        let ctx = make_ctx(128_000, 0.4, sections);
        let result = CloudAssembler.assemble(&ctx);
        // Should have 3 blocks in the report
        assert_eq!(result.report.blocks.len(), 3);
        // All should be included (plenty of budget)
        assert!(result.report.blocks.iter().all(|b| b.included));
        // Total tokens should be > 0
        assert!(result.report.total_tokens > 0);
        // Cap tokens should be set
        assert!(result.report.cap_tokens.is_some());
    }

    #[test]
    fn test_overflow_produces_report_with_dropped_sections() {
        // When overflow occurs, the report should reflect which sections were
        // dropped or shrunk. This indirectly verifies the tracing::warn paths
        // fire (they share the same code paths that set included=false).
        let sections = vec![
            make_entry(PromptSection::Identity, "", "id"),
            make_entry(PromptSection::MemoryBriefing, "Memory", &"x".repeat(5000)),
        ];
        // 500 * 0.3 = 150 token cap. Identity (~1 tok) fits. MemoryBriefing
        // is shrinkable so Pass 2 shrinks it rather than dropping.
        let ctx = make_ctx(500, 0.3, sections);
        let result = LocalAssembler.assemble(&ctx);

        let memory_block = result
            .report
            .blocks
            .iter()
            .find(|b| b.title == "Memory")
            .unwrap();
        // MemoryBriefing should have been shrunk (still included but fewer tokens)
        assert!(memory_block.included, "shrinkable section should be shrunk, not dropped");
        assert!(
            memory_block.tokens < 1250,
            "should be shrunk below original ~1250 tokens, got {}",
            memory_block.tokens,
        );
    }
}
