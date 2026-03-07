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
}
