// Error-protocol layer-3 backlog (docs/research/2026-08-06-error-conventions-and-host-bridge.md §3.6):
// the deny regime in Cargo.toml is live; this module still carries pre-existing
// violations of the lints below. Remove this allow as the module migrates onto
// the regime.
// Tracking: docs/error-protocol-backlog.md
#![allow(
    clippy::as_conversions,
    clippy::format_push_string,
    clippy::indexing_slicing,
    clippy::shadow_reuse
)]
//! Lossless Context Management (LCM)
//!
//! Implements the LCM architecture from Ehrlich & Blackman (2026):
//! - **Immutable Store**: Every raw message persisted verbatim in SQLite.
//! - **Active Context**: Window sent to LLM = recent raw messages + summary nodes.
//! - **Summary DAG**: Hierarchical summaries with lossless pointers to originals.
//! - **Two-threshold control loop**: τ_soft (async) / τ_hard (blocking).
//! - **Escalating summarization**: preserve_details → bullet_points. No
//!   deterministic-truncation fallback for a missing/failing compactor or a
//!   babble/degenerate summary — those leave the context uncompacted this
//!   round instead. Deterministic truncation still fires as an instant path
//!   for oversized blocks, where an LLM call would be prohibitively slow.
//!
//! SQLite message rows and summary nodes provide restart-safe storage. This
//! module manages the in-memory DAG and active context assembly.

#![allow(clippy::disallowed_types)] // anyhow is the app convention — the ban targets tool boundaries (error protocol §2.5)
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

use crate::agent::anti_drift;
use crate::agent::compaction::ContextCompactor;
use crate::agent::token_budget::TokenBudget;
use crate::agent::turn::Turn;
use crate::config::schema::LcmSchemaConfig;

// ---------------------------------------------------------------------------
// Summary DAG
// ---------------------------------------------------------------------------

/// Unique ID for a message in the immutable store.
///
/// This is the SQLite `messages.id` rowid (`_db_id` on reconstructed message
/// Values) — stable across restarts and independent of any windowing or
/// filtering applied to the live context. Messages that have not been
/// persisted yet carry no `_db_id` and are not ingested; they are picked up
/// on the next turn, after persistence.
pub type MessageId = usize;

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ManifestItem {
    #[serde(default)]
    pub text: String,
    /// Message IDs this item references. The summarization prompt asks the
    /// model for `[<integer id>, ...]`, but local models in the wild emit
    /// strings like `"msg 55531"` (matching the `[message_id: 55531]`
    /// transcript labels) or `"55531"` (numeric strings). A single malformed
    /// source value used to fail the entire manifest deserialization and
    /// collapse 100% of the recovered state (live 2026-07-27 12:34:03). The
    /// custom deserializer extracts the first integer from each value and
    /// silently skips non-numeric entries.
    #[serde(default, deserialize_with = "deserialize_message_id_array")]
    pub sources: Vec<MessageId>,
}

/// Extract a `MessageId` from a JSON value: integers pass through; strings
/// contribute their first contiguous run of ASCII digits (so `"msg 55531"`,
/// `"55531"`, and `"[message_id: 55531]"` all yield `55531`); anything else
/// yields `None` and is skipped by the array visitor.
fn extract_message_id(value: &Value) -> Option<MessageId> {
    match value {
        Value::Number(n) => n.as_u64().map(|x| x as MessageId),
        Value::String(s) => {
            let mut start: Option<usize> = None;
            let mut end: usize = 0;
            for (i, c) in s.char_indices() {
                if c.is_ascii_digit() {
                    if start.is_none() {
                        start = Some(i);
                    }
                    end = i + c.len_utf8();
                } else if start.is_some() {
                    break;
                }
            }
            start.and_then(|begin| s[begin..end].parse().ok())
        }
        _ => None,
    }
}

fn deserialize_message_id_array<'de, D>(
    deserializer: D,
) -> std::result::Result<Vec<MessageId>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::SeqAccess;

    struct MessageIdArrayVisitor;

    impl<'de> serde::de::Visitor<'de> for MessageIdArrayVisitor {
        type Value = Vec<MessageId>;

        fn expecting(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
            write!(
                fmt,
                "an array of integer message ids or strings containing message ids"
            )
        }

        fn visit_seq<A>(self, mut seq: A) -> std::result::Result<Self::Value, A::Error>
        where
            A: SeqAccess<'de>,
        {
            let mut out = Vec::new();
            while let Some(value) = seq.next_element::<Value>()? {
                if let Some(id) = extract_message_id(&value) {
                    out.push(id);
                }
            }
            Ok(out)
        }
    }

    deserializer.deserialize_seq(MessageIdArrayVisitor)
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct SummaryManifest {
    #[serde(default)]
    pub open_loops: Vec<ManifestItem>,
    #[serde(default)]
    pub failed_approaches: Vec<ManifestItem>,
    #[serde(default)]
    pub decisions: Vec<ManifestItem>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ManifestCategory {
    OpenLoop,
    FailedApproach,
    Decision,
}

impl SummaryManifest {
    pub fn merge(parts: &[&SummaryManifest]) -> Self {
        let mut merged: Vec<(ManifestCategory, ManifestItem)> = Vec::new();
        for part in parts {
            for (category, items) in [
                (ManifestCategory::OpenLoop, &part.open_loops),
                (ManifestCategory::FailedApproach, &part.failed_approaches),
                (ManifestCategory::Decision, &part.decisions),
            ] {
                for item in items {
                    let key = normalized_manifest_text(&item.text);
                    if let Some((existing_category, existing)) = merged
                        .iter_mut()
                        .find(|(_, existing)| normalized_manifest_text(&existing.text) == key)
                    {
                        *existing_category = category;
                        existing.sources.extend(item.sources.iter().copied());
                        existing.sources.sort_unstable();
                        existing.sources.dedup();
                        continue;
                    }

                    let mut item = item.clone();
                    item.sources.sort_unstable();
                    item.sources.dedup();
                    merged.push((category, item));
                }
            }
        }

        let mut manifest = Self::default();
        for (category, item) in merged {
            match category {
                ManifestCategory::OpenLoop => manifest.open_loops.push(item),
                ManifestCategory::FailedApproach => manifest.failed_approaches.push(item),
                ManifestCategory::Decision => manifest.decisions.push(item),
            }
        }
        manifest
    }
}

fn normalized_manifest_text(text: &str) -> String {
    text.trim().to_lowercase()
}

fn extract_summary_manifest(reply: String) -> (String, SummaryManifest) {
    const JSON_FENCE: &str = "```json";
    const MANIFEST_KEYS: [&str; 3] = ["\"open_loops\"", "\"failed_approaches\"", "\"decisions\""];

    let Some(fence_start) = reply.rfind(JSON_FENCE) else {
        return (reply, SummaryManifest::default());
    };
    let json_start = fence_start + JSON_FENCE.len();
    let Some(fence_end_offset) = reply[json_start..].find("```") else {
        let json = &reply[json_start..];
        if !MANIFEST_KEYS.iter().all(|key| json.contains(key)) {
            return (reply, SummaryManifest::default());
        }
        debug!("LCM: summary manifest JSON fence was not closed; using empty manifest");
        return (
            reply[..fence_start].trim_end().to_string(),
            SummaryManifest::default(),
        );
    };
    let fence_end = json_start + fence_end_offset;
    let json = &reply[json_start..fence_end];
    if !reply[fence_end + 3..].trim().is_empty()
        || !MANIFEST_KEYS.iter().all(|key| json.contains(key))
    {
        return (reply, SummaryManifest::default());
    }

    let prose = reply[..fence_start].trim_end().to_string();
    let manifest = match serde_json::from_str::<SummaryManifest>(json) {
        Ok(manifest) => manifest,
        Err(error) => {
            debug!(%error, "LCM: failed to parse summary manifest JSON; using empty manifest");
            SummaryManifest::default()
        }
    };
    (prose, manifest)
}

/// A summary node in the DAG.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummaryNode {
    /// Unique ID of this summary node.
    pub id: usize,
    /// IDs of the original messages this summary covers.
    pub source_ids: Vec<MessageId>,
    /// IDs of child summary nodes (if this is a merge of summaries).
    pub child_summaries: Vec<usize>,
    /// The summary text.
    pub text: String,
    /// Structured state that survives repeated summary merges.
    #[serde(default)]
    pub manifest: SummaryManifest,
    /// Estimated token count of the summary.
    pub tokens: usize,
    /// Escalation level that produced this summary (1, 2, or 3).
    pub level: u8,
    /// Session turn when this node was created, stamped by `LcmEngine` from
    /// its `current_turn` counter. Used by `auto_expand`'s fresh-summary
    /// cooldown to prevent the just-compacted detail from being reinjected
    /// the very next turn (live failure 2026-07-27 12:13:06 saw +12463
    /// tokens reinjected 24 seconds after compaction). Persisted with
    /// `#[serde(default)]` so older SQLite rows restore as 0 — those are
    /// treated as ancient history by the cooldown check.
    #[serde(default)]
    pub created_at_turn: u64,
}

/// Compress sorted message IDs into a compact range string: `5-8,12,14-20`.
///
/// Summary headers used to embed every ID (144 five-digit rowids ≈ 1KB of
/// noise per summary). Ranges keep the header tiny and are directly accepted
/// by `lcm_expand` (`parse_id_runs` understands `a-b`).
pub fn format_id_ranges(ids: &[MessageId]) -> String {
    let mut sorted: Vec<MessageId> = ids.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    let mut parts: Vec<String> = Vec::new();
    let mut i = 0;
    while i < sorted.len() {
        let start = sorted[i];
        let mut end = start;
        while i + 1 < sorted.len() && sorted[i + 1] == end + 1 {
            i += 1;
            end = sorted[i];
        }
        parts.push(if start == end {
            start.to_string()
        } else {
            format!("{start}-{end}")
        });
        i += 1;
    }
    parts.join(",")
}

/// Build the wire message that represents a summary node in the active
/// context. Single source of truth for the header format (live compaction and
/// restart rebuild must render byte-identically for the prompt prefix cache).
fn summary_wire_message(source_ids: &[MessageId], text: &str, manifest: &SummaryManifest) -> Value {
    let ranges = format_id_ranges(source_ids);
    let mut content = format!(
        "[Summary of messages {ranges}. To read the exact originals call \
         lcm_expand({{\"message_ids\": \"{ranges}\"}}).]\n\n{text}"
    );
    if !manifest.open_loops.is_empty()
        || !manifest.failed_approaches.is_empty()
        || !manifest.decisions.is_empty()
    {
        content.push_str("\n\n[State manifest]");
        for (label, items) in [
            ("Open loops:", &manifest.open_loops),
            ("Failed approaches:", &manifest.failed_approaches),
            ("Decisions:", &manifest.decisions),
        ] {
            if items.is_empty() {
                continue;
            }
            content.push('\n');
            content.push_str(label);
            for item in items {
                content.push_str("\n- ");
                content.push_str(&item.text);
                if !item.sources.is_empty() {
                    content.push_str(" [sources: ");
                    content.push_str(&format_id_ranges(&item.sources));
                    content.push(']');
                }
            }
        }
    }
    json!({
        "role": "user",
        "_lcm_summary": true,
        "content": content
    })
}

/// The summary DAG: tracks all summary nodes and the active context composition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummaryDag {
    /// All summary nodes, indexed by their ID.
    nodes: Vec<SummaryNode>,
    /// Next node ID.
    next_id: usize,
}

impl SummaryDag {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            next_id: 0,
        }
    }

    /// Create a new summary node covering the given source messages.
    pub fn create_node(
        &mut self,
        source_ids: Vec<MessageId>,
        child_summaries: Vec<usize>,
        text: String,
        manifest: SummaryManifest,
        level: u8,
    ) -> &SummaryNode {
        let tokens = TokenBudget::estimate_str_tokens(&text);
        let id = self.next_id;
        self.next_id += 1;
        self.nodes.push(SummaryNode {
            id,
            source_ids,
            child_summaries,
            text,
            manifest,
            tokens,
            level,
            created_at_turn: 0,
        });
        &self.nodes[self.nodes.len() - 1]
    }

    /// Stamp the session turn when a node was created. Called by
    /// `LcmEngine::compact` right after `create_node` so the cooldown check
    /// in `auto_expand` can tell fresh nodes from established ones. Only
    /// mutates the in-memory DAG; persistence stores the field via
    /// `save_compaction_checkpoint`.
    pub(crate) fn stamp_node_creation_turn(&mut self, node_id: usize, turn: u64) {
        if let Some(node) = self.nodes.iter_mut().find(|n| n.id == node_id) {
            node.created_at_turn = turn;
        }
    }

    /// Get a summary node by ID.
    pub fn get(&self, id: usize) -> Option<&SummaryNode> {
        self.nodes.iter().find(|n| n.id == id)
    }

    /// Get all source message IDs covered by a summary (recursively).
    pub fn all_source_ids(&self, node_id: usize) -> Vec<MessageId> {
        let mut result = Vec::new();
        if let Some(node) = self.get(node_id) {
            result.extend(&node.source_ids);
            for &child_id in &node.child_summaries {
                result.extend(self.all_source_ids(child_id));
            }
        }
        result
    }

    /// Total number of summary nodes.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Most recently allocated node, regardless of sparse persisted IDs.
    ///
    /// Restart reconstruction preserves SQLite node IDs, so `len() - 1` is
    /// not necessarily a node ID. Live compaction always allocates above the
    /// greatest restored ID, making the greatest ID the node just created.
    pub(crate) fn newest(&self) -> Option<&SummaryNode> {
        self.nodes.iter().max_by_key(|node| node.id)
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Active Context
// ---------------------------------------------------------------------------

/// An entry in the active context: either a raw message or a summary pointer.
#[derive(Debug, Clone)]
pub enum ContextEntry {
    /// A raw message from the immutable store.
    Raw { msg_id: MessageId, message: Value },
    /// A summary node replacing a block of older messages.
    Summary {
        node_id: usize,
        /// The summary formatted as a message Value.
        message: Value,
    },
}

impl ContextEntry {
    /// Get the message Value for sending to the LLM.
    pub fn message(&self) -> &Value {
        match self {
            ContextEntry::Raw { message, .. } => message,
            ContextEntry::Summary { message, .. } => message,
        }
    }
}

// ---------------------------------------------------------------------------
// LCM Engine
// ---------------------------------------------------------------------------

/// Configuration for the LCM engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LcmConfig {
    /// Soft threshold as fraction of available context (0.0-1.0).
    /// Triggers async compaction. Default: 0.5 (50%).
    pub tau_soft: f64,
    /// Hard threshold as fraction of available context (0.0-1.0).
    /// Triggers blocking compaction. Default: 0.85 (85%).
    pub tau_hard: f64,
    /// Target tokens for Level 3 deterministic truncation.
    pub deterministic_target: usize,
}

impl Default for LcmConfig {
    fn default() -> Self {
        Self {
            tau_soft: 0.5,
            tau_hard: 0.85,
            deterministic_target: 512,
        }
    }
}

impl From<&LcmSchemaConfig> for LcmConfig {
    fn from(schema: &LcmSchemaConfig) -> Self {
        Self {
            tau_soft: schema.tau_soft,
            tau_hard: schema.tau_hard,
            deterministic_target: schema.deterministic_target,
        }
    }
}

/// Default recent-raw token budget for the test helper / fallback (≈1k tokens,
/// matching the production protect target so post-compaction lands ~1–2k).
#[cfg(test)]
const DEFAULT_PROTECT_TOKENS: usize = 1024;

/// Merge accumulated summaries once their mass becomes material to the prompt.
const SUMMARY_MERGE_BUDGET_FRACTION: f64 = 0.25;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BlockSelection {
    AppendOnly,
    MergeSummaries,
}

/// Token budget for the recent raw messages kept verbatim (not summarized),
/// scaled to the available context. ~5% of the budget, clamped to [512, 2048]:
/// on a ~22k effective budget this keeps ~1k tokens of recent turns raw (the
/// rest summarized), so post-compaction active context lands ~1–2k and a
/// cache-miss re-prefill is a few seconds, not minutes. Token-based (not
/// message-count) so a single huge tool result can't blow past the target.
fn protect_tokens_for_budget(available_tokens: usize) -> usize {
    (available_tokens / 20).clamp(512, 2048)
}

/// A freshly created summary node is ineligible for `auto_expand` while its
/// age in turns is `<=` this value. Age is `current_turn - created_at_turn`.
/// Cooldown of 1 means a node created during turn N's compaction cannot be
/// re-expanded until turn N+2 — the very next turn (N+1, age=1) is still
/// blocked, which is the live failure window (session 20260727_094539_eeab48,
/// 2026-07-27 12:12:42 compact → 12:13:06 +12463 token reinject).
///
/// Nodes with `created_at_turn == 0` (back-compat rows, or test scaffolding
/// that never set the turn) are always eligible.
const FRESH_SUMMARY_COOLDOWN_TURNS: u64 = 1;

/// One relevant summary selected for lossless auto-expansion. Planning is
/// intentionally inert: callers decide how to represent the candidate on the
/// wire and commit its node ID only after a successful provider response.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct AutoExpansionCandidate {
    pub(crate) node_id: usize,
    pub(crate) source_ids: Vec<MessageId>,
    pub(crate) estimated_tokens: usize,
    pub(crate) flattened_fallback: Value,
    pub(crate) summary_message: Value,
}

/// The LCM engine: manages the active context with lossless compaction.
pub struct LcmEngine {
    config: LcmConfig,
    /// The summary DAG.
    dag: SummaryDag,
    /// Active context entries (system prompt + summaries + raw messages).
    active: Vec<ContextEntry>,
    /// All raw messages in the immutable store, keyed by their stable DB
    /// rowid (`_db_id`). Ordered iteration (BTreeMap) preserves session order
    /// for rebuild and active-context assembly. This is the in-memory mirror;
    /// the session DB is the durable copy.
    store: std::collections::BTreeMap<MessageId, Value>,
    /// Whether async compaction has been requested but not yet completed.
    async_compaction_pending: bool,
    /// Summary node IDs already auto-expanded into the tail this session, so the
    /// per-turn auto_expand pass doesn't append the same detail repeatedly.
    auto_expanded: std::collections::HashSet<usize>,
    /// Session turn counter, bumped by the caller at turn start via
    /// `set_current_turn`. Stamped onto every freshly-created summary node
    /// so `auto_expand` can apply `FRESH_SUMMARY_COOLDOWN_TURNS`.
    current_turn: u64,
}

/// Mutable LCM state that must advance only after its SQLite checkpoint does.
/// The immutable raw-message mirror is deliberately excluded from rollback.
pub(crate) struct LcmCompactionState {
    dag: SummaryDag,
    active: Vec<ContextEntry>,
}

impl LcmEngine {
    pub fn new(config: LcmConfig) -> Self {
        Self {
            config,
            dag: SummaryDag::new(),
            active: Vec::new(),
            store: std::collections::BTreeMap::new(),
            async_compaction_pending: false,
            auto_expanded: std::collections::HashSet::new(),
            current_turn: 0,
        }
    }

    /// Record the session's current turn. Called by `agent_loop` at turn
    /// start so subsequent `compact()` calls stamp the new node's
    /// `created_at_turn`, and `auto_expand`'s cooldown check knows the
    /// present turn. Tests that never call this default to 0, which
    /// disables the cooldown (back-compat with summaries created without a
    /// turn stamp).
    pub fn set_current_turn(&mut self, turn: u64) {
        self.current_turn = turn;
    }

    pub fn current_turn(&self) -> u64 {
        self.current_turn
    }

    pub(crate) fn compaction_state(&self) -> LcmCompactionState {
        LcmCompactionState {
            dag: self.dag.clone(),
            active: self.active.clone(),
        }
    }

    pub(crate) fn restore_compaction_state(&mut self, state: LcmCompactionState) {
        self.dag = state.dag;
        self.active = state.active;
        self.async_compaction_pending = false;
    }

    /// Ingest a message into the immutable store and active context.
    ///
    /// The MessageId is the message's `_db_id` (SQLite rowid), so IDs are
    /// stable across restarts regardless of how the live context was
    /// windowed. Ingest is an idempotent upsert:
    /// - No `_db_id` → the message has not been persisted yet; it is NOT
    ///   ingested (returns `None`). Such messages are always the newest, live
    ///   inside the protect window, and are ingested on the next turn once
    ///   `get_history` supplies their rowid.
    /// - `_db_id` already in the store → skipped (returns the existing id).
    pub fn ingest(&mut self, message: Value) -> Option<MessageId> {
        // Summary wire blocks are views of rows already in the store, never
        // originals. (Rows persisted by a pre-fix bug carry `_lcm_summary` in
        // metadata; refusing them here keeps old DBs from re-polluting.)
        if message.get("_lcm_summary").is_some() {
            return None;
        }
        let msg_id = message.get("_db_id").and_then(|v| v.as_u64())? as usize;
        if self.store.contains_key(&msg_id) {
            return Some(msg_id);
        }
        self.store.insert(msg_id, message.clone());
        self.active.push(ContextEntry::Raw { msg_id, message });
        Some(msg_id)
    }

    /// Get the active context as a message array for the LLM.
    pub fn active_context(&self) -> Vec<Value> {
        self.active.iter().map(|e| e.message().clone()).collect()
    }

    /// Estimate the token count of the active context (including system prompt).
    pub fn active_tokens(&self) -> usize {
        let messages: Vec<Value> = self.active_context();
        TokenBudget::estimate_tokens(&messages)
    }

    /// Estimate token count of conversation only (excludes system prompt).
    ///
    /// The system prompt is a fixed cost that cannot be compacted, so it must
    /// not count toward the compaction threshold. Without this, a large system
    /// prompt (~8K tokens on a 32K window with tau_soft=0.5) leaves only ~7K
    /// for conversation, triggering compaction after just 1-2 exchanges.
    pub fn conversation_tokens(&self) -> usize {
        let messages: Vec<Value> = self
            .active
            .iter()
            .filter(|e| e.message().get("role").and_then(|r| r.as_str()) != Some("system"))
            .map(|e| e.message().clone())
            .collect();
        TokenBudget::estimate_tokens(&messages)
    }

    pub fn tau_soft(&self) -> f64 {
        self.config.tau_soft
    }
    pub fn tau_hard(&self) -> f64 {
        self.config.tau_hard
    }

    /// Check thresholds and return what action is needed.
    pub fn check_thresholds(
        &self,
        budget: &TokenBudget,
        tool_def_tokens: usize,
    ) -> CompactionAction {
        let available = budget.available_budget(tool_def_tokens);
        self.check_thresholds_with_available(available)
    }

    /// Check thresholds against an already-adjusted conversation budget.
    ///
    /// The normal path uses the model context window minus response/tool
    /// reserves. Callers with a stricter backend admission cap can pass that
    /// effective conversation budget here without rewriting the shared
    /// `TokenBudget`.
    pub fn check_thresholds_with_available(&self, available: usize) -> CompactionAction {
        // Use conversation tokens only — the system prompt is fixed overhead
        // that cannot be compacted and must not trigger the threshold.
        let current = self.conversation_tokens();

        let hard_limit = (available as f64 * self.config.tau_hard) as usize;
        let soft_limit = (available as f64 * self.config.tau_soft) as usize;

        if current >= hard_limit {
            CompactionAction::Blocking
        } else if current >= soft_limit && !self.async_compaction_pending {
            CompactionAction::Async
        } else {
            CompactionAction::None
        }
    }

    /// Perform compaction using the three-level escalation protocol.
    ///
    /// Algorithm 3 from the paper:
    /// - Level 1: LLM summarize with mode="preserve_details", target T tokens
    /// - Level 2: LLM summarize with mode="bullet_points", target T/2 tokens
    /// - Level 3: Deterministic truncation to 512 tokens (no LLM)
    ///
    /// Returns a `Turn::Summary` if compaction occurred. The caller persists
    /// its DAG node in SQLite so the active hierarchy can be rebuilt on restart.
    pub async fn compact(
        &mut self,
        compactor: Option<&ContextCompactor>,
        budget: &TokenBudget,
        tool_def_tokens: usize,
        // Retained for call-site intent (soft vs blocking). Since huge blocks
        // in BOTH modes now take deterministic truncation (the >80 guard no
        // longer refuses soft blocks), the mode no longer branches here.
        _failure_mode: CompactionFailureMode,
    ) -> Option<Turn> {
        let available = budget.available_budget(tool_def_tokens);
        let target = (available as f64 * self.config.tau_soft * 0.8) as usize;

        // Token-budgeted protection: keep ~1k tokens of the most recent raw
        // turns verbatim (the rest summarized), so post-compaction active
        // context lands ~1–2k regardless of individual message size. Cap at
        // half the conversation so we always compact a meaningful portion
        // (matters when messages are small relative to the protect target).
        let protect_tokens =
            protect_tokens_for_budget(available).min(self.conversation_tokens().max(1) / 2);
        let selection = self.block_selection(available);

        // Find the oldest contiguous block of raw messages to compact.
        let (block_start, block_end) =
            match self.find_oldest_raw_block_impl(protect_tokens, selection) {
                Some(range) => range,
                None => {
                    debug!("LCM: no raw block to compact");
                    self.async_compaction_pending = false;
                    return None;
                }
            };

        // Collect messages and source ids from the block. Merge mode may
        // include prior Summary entries; append mode never does. When present,
        // their text is folded into the summarization input and their source
        // ids are unioned so lcm_expand stays lossless.
        let mut source_ids: Vec<MessageId> = Vec::new();
        let mut block_messages = Vec::new();
        let mut merged_node_ids: Vec<usize> = Vec::new();
        let mut raw_count = 0usize;
        for entry in &self.active[block_start..block_end] {
            match entry {
                ContextEntry::Raw { msg_id, message } => {
                    source_ids.push(*msg_id);
                    block_messages.push(message.clone());
                    raw_count += 1;
                }
                ContextEntry::Summary { node_id, .. } => {
                    // Fold the node's TEXT into the summarization input — not
                    // the wire message, whose "[Summary … IDs: …]" header is ID
                    // noise. Feeding the header made deterministic truncation
                    // keep the ID list as the "first sentence", producing
                    // summaries that were literally ID spam with content lost.
                    let text = self
                        .dag
                        .get(*node_id)
                        .map(|n| n.text.clone())
                        .unwrap_or_default();
                    block_messages.push(json!({
                        "role": "user",
                        "content": format!("[Earlier summary]\n{text}")
                    }));
                    for sid in self.dag.all_source_ids(*node_id) {
                        if !source_ids.contains(&sid) {
                            source_ids.push(sid);
                        }
                    }
                    // Track the subsumed node so it can be retired in the DAG
                    // and DB (rebuild_from_db_nodes skips children), keeping the
                    // merge durable across restarts instead of re-accumulating.
                    merged_node_ids.push(*node_id);
                }
            }
        }

        // Need at least one RAW (new content) to merge — re-summarizing a lone
        // summary with no new raws is wasted work.
        if raw_count == 0 {
            debug!("LCM: compact block had no raws (only summaries); nothing new to merge");
            self.async_compaction_pending = false;
            return None;
        }

        if block_messages.is_empty() {
            self.async_compaction_pending = false;
            return None;
        }

        let block_tokens = TokenBudget::estimate_tokens(&block_messages);

        // Skip compaction when the block is too small to be worth an LLM call.
        // An LLM summarization of <200 tokens wastes more GPU time than it saves.
        const MIN_COMPACTION_TOKENS: usize = 200;
        if block_tokens < MIN_COMPACTION_TOKENS {
            debug!(
                "LCM: skipping compaction — block too small ({} tokens < {})",
                block_tokens, MIN_COMPACTION_TOKENS
            );
            self.async_compaction_pending = false;
            return None;
        }

        info!(
            "LCM: compacting {} messages ({} tokens) from positions {}..{}",
            block_messages.len(),
            block_tokens,
            block_start,
            block_end
        );

        // Guard: if the block is enormous, skip LLM summarization entirely.
        // The historical rationale (a 0.8B summarizer doing 60-75+ sequential
        // chunk+merge calls) is dead — summarize_with_prompt makes ONE call.
        // What matters now: a huge block exceeds the summarizer's transcript
        // band, and REFUSING to compact it (the old PreserveContext behavior)
        // left the store in async-band limbo — every turn re-fired Async, hit
        // this guard, and installed nothing while the prompt grew to the
        // server's cap (session 20260828_142425: 0 summaries in 83 messages,
        // then a 16K-token raw re-prefill after the overflow). Large blocks in
        // soft mode now take the same deterministic truncation the blocking
        // path uses.
        const MAX_COMPACTION_BLOCK_MESSAGES: usize = 80;
        let (summary_text, fresh_manifest, level) = if block_messages.len()
            > MAX_COMPACTION_BLOCK_MESSAGES
        {
            info!(
                "LCM: block too large ({} msgs > {}) for LLM summarization, using deterministic truncation",
                block_messages.len(),
                MAX_COMPACTION_BLOCK_MESSAGES
            );
            let truncated =
                deterministic_truncate(&block_messages, self.config.deterministic_target);
            (truncated, SummaryManifest::default(), 3)
        } else {
            // Escalating LLM summarization (Algorithm 3, levels 1-2). There is
            // no deterministic-truncation fallback here: a missing compactor,
            // a failed LLM call, or a babble/degenerate summary all leave the
            // active context uncompacted this round rather than installing a
            // lossy truncation silently. Compaction simply retries next turn.
            match escalated_summary(&block_messages, target, compactor).await {
                Ok(Some(summary)) => summary,
                Ok(None) => {
                    self.async_compaction_pending = false;
                    return None;
                }
                Err(error) => {
                    warn!(
                        %error,
                        "LCM: compaction summarization failed, leaving context uncompacted this round"
                    );
                    self.async_compaction_pending = false;
                    return None;
                }
            }
        };

        let child_manifests: Vec<SummaryManifest> = merged_node_ids
            .iter()
            .filter_map(|node_id| self.dag.get(*node_id))
            .map(|node| node.manifest.clone())
            .collect();
        let mut manifest_parts: Vec<&SummaryManifest> = child_manifests.iter().collect();
        manifest_parts.push(&fresh_manifest);
        let manifest = SummaryManifest::merge(&manifest_parts);
        let summary_tokens = TokenBudget::estimate_str_tokens(&summary_text);

        // Only accept if summary is smaller than original.
        if summary_tokens >= block_tokens {
            warn!(
                "LCM: summary ({} tokens) not smaller than original ({} tokens), skipping",
                summary_tokens, block_tokens
            );
            self.async_compaction_pending = false;
            return None;
        }

        info!(
            "LCM: compacted {} -> {} tokens (level {}, {:.0}% reduction)",
            block_tokens,
            summary_tokens,
            level,
            (1.0 - summary_tokens as f64 / block_tokens as f64) * 100.0
        );

        // Create summary node in DAG, recording the subsumed children so the
        // merge is durable (rebuild_from_db_nodes skips child nodes; the
        // persistence site stores them as child_ids).
        let node = self.dag.create_node(
            source_ids.clone(),
            merged_node_ids.clone(),
            summary_text.clone(),
            manifest,
            level,
        );
        let node_id = node.id;
        let summary_source_ids = node.source_ids.clone();
        let summary_text_clone = node.text.clone();
        let summary_manifest = node.manifest.clone();
        // Stamp the creation turn so `auto_expand`'s cooldown can tell this
        // fresh node apart from established summaries on the next turn.
        self.dag
            .stamp_node_creation_turn(node_id, self.current_turn);

        // Build the summary message with lossless pointers (compact ranges).
        let summary_message =
            summary_wire_message(&summary_source_ids, &summary_text_clone, &summary_manifest);

        // Replace the block in active context with the summary.
        let mut new_active = Vec::with_capacity(self.active.len());
        new_active.extend_from_slice(&self.active[..block_start]);
        new_active.push(ContextEntry::Summary {
            node_id,
            message: summary_message,
        });
        if block_end < self.active.len() {
            new_active.extend_from_slice(&self.active[block_end..]);
        }
        self.active = new_active;
        self.async_compaction_pending = false;

        Some(Turn::Summary {
            text: summary_text,
            source_ids,
            level,
        })
    }

    fn block_selection(&self, available: usize) -> BlockSelection {
        let summary_tokens = self
            .active
            .iter()
            .filter_map(|entry| match entry {
                ContextEntry::Summary { node_id, .. } => {
                    self.dag.get(*node_id).map(|node| node.tokens)
                }
                ContextEntry::Raw { .. } => None,
            })
            .fold(0usize, usize::saturating_add);

        if summary_tokens as f64 > available as f64 * SUMMARY_MERGE_BUDGET_FRACTION {
            BlockSelection::MergeSummaries
        } else {
            BlockSelection::AppendOnly
        }
    }

    /// Find the block to compact: either the oldest raw-only run or everything
    /// from the first compactible entry up to the recent-protect boundary.
    ///
    /// `AppendOnly` leaves existing summaries byte-stable at the prompt front.
    /// `MergeSummaries` includes them in the block, bounding summary mass by
    /// replacing all accumulated summaries with one new node.
    ///
    /// Protection is TOKEN-based (not message-count): walk back from the end
    /// accumulating tokens of real raw messages (2× for tool results, which are
    /// retrievable via recall_tool_result); the boundary where we cross
    /// `protect_tokens` is the oldest still-protected index. Summaries are
    /// compactible only in `MergeSummaries` mode.
    fn find_oldest_raw_block_impl(
        &self,
        protect_tokens: usize,
        selection: BlockSelection,
    ) -> Option<(usize, usize)> {
        let start = (0..self.active.len()).find(|&i| match &self.active[i] {
            ContextEntry::Raw { message, .. } => {
                let role = message.get("role").and_then(|r| r.as_str()).unwrap_or("");
                role != "system" && !crate::agent::markers::is_synthetic(message)
            }
            ContextEntry::Summary { .. } => selection == BlockSelection::MergeSummaries,
        })?;

        // Protect the most recent RAW messages (token-based, 2× tool weighting).
        // `boundary` = oldest protected raw index; the compact block is [start, boundary).
        let mut acc: usize = 0;
        let mut boundary: usize = 0;
        let mut seen_any = false;
        for i in (0..self.active.len()).rev() {
            if let ContextEntry::Raw { message, .. } = &self.active[i] {
                let role = message.get("role").and_then(|r| r.as_str()).unwrap_or("");
                if role == "system" || crate::agent::markers::is_synthetic(message) {
                    continue;
                }
                let t = TokenBudget::estimate_message_tokens(message);
                let weighted = if role == "tool" {
                    t.saturating_mul(2)
                } else {
                    t
                };
                if seen_any && acc + weighted > protect_tokens {
                    break; // protect window full; this older message is compacted
                }
                acc += weighted;
                boundary = i;
                seen_any = true;
            }
        }
        if boundary > start {
            Some((start, boundary))
        } else {
            None
        }
    }

    /// Mark that async compaction has been requested.
    pub fn request_async_compaction(&mut self) {
        self.async_compaction_pending = true;
    }

    /// Clear the soft-pending flag on cancelled jobs. Without this, one
    /// cancelled job permanently disables soft compaction for the session
    /// (`check_thresholds_with_available` gates Async on `!pending`).
    pub fn clear_async_compaction_pending(&mut self) {
        self.async_compaction_pending = false;
    }

    /// Retrieve original messages by IDs from the immutable store.
    ///
    /// This is the `lcm_expand` operation — lossless retrieval.
    pub fn expand(&self, msg_ids: &[MessageId]) -> Vec<(MessageId, &Value)> {
        msg_ids
            .iter()
            .filter_map(|&id| self.store.get(&id).map(|msg| (id, msg)))
            .collect()
    }

    /// Format expanded messages for display (used by lcm_expand tool).
    pub fn format_expanded(&self, msg_ids: &[MessageId]) -> String {
        let mut output = String::new();
        for (id, msg) in self.expand(msg_ids) {
            let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("?");
            let content = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
            output.push_str(&format!("[msg {}] {}: {}\n\n", id, role, content));
        }
        if output.is_empty() {
            output = "No messages found for the given IDs. lcm_expand only expands \
                      LCM summary blocks from the CURRENT session — the IDs you passed \
                      are not in this session's compaction graph (they may belong to a \
                      past session). To retrieve a different past session's full \
                      transcript, use session_search(mode=\"session\", session=KEY) with \
                      that session's key."
                .to_string();
        }
        output
    }

    /// Get the summary DAG (for serialization/debugging).
    pub fn dag(&self) -> &SummaryDag {
        &self.dag
    }

    /// Get the immutable store size.
    pub fn store_len(&self) -> usize {
        self.store.len()
    }

    /// All message IDs currently in the immutable store, ascending.
    pub fn store_ids(&self) -> Vec<MessageId> {
        self.store.keys().copied().collect()
    }

    /// Get the active context entry count.
    pub fn active_len(&self) -> usize {
        self.active.len()
    }

    /// Access the active context entries (for inspection in tests).
    pub fn active_entries(&self) -> &[ContextEntry] {
        &self.active
    }

    /// Find the oldest contiguous block of raw messages with an explicit
    /// protect-token budget (for testing the token-based boundary).
    pub fn find_oldest_raw_block_with_tokens(
        &self,
        protect_tokens: usize,
    ) -> Option<(usize, usize)> {
        self.find_oldest_raw_block_impl(protect_tokens, BlockSelection::AppendOnly)
    }

    /// Plan summaries that are relevant to the latest user message.
    ///
    /// The flattened fallback on each candidate is the synthetic message the
    /// caller may append after any frozen cache prefix. Selection never updates
    /// `auto_expanded`; the caller must explicitly commit the node only after a
    /// successful provider response.
    ///
    /// `wire_tokens` is the actual rendered prompt size for this turn — what
    /// the provider will prefill. Headroom is computed against `wire_tokens`,
    /// NOT against `self.active_tokens()`: reinjected originals live in the
    /// wire (not the engine's internal active), so the engine's view
    /// under-counts. Counting the wire keeps reinjection from pushing the
    /// prompt past τ_hard (the failure mode that broke the Higgs retained
    /// session and forced 60s+ ExactBootstrap prefills).
    ///
    /// Relevance uses semantic embeddings when available, else keyword overlap.
    ///
    /// Returns bounded candidates (empty if nothing is eligible).
    pub(crate) fn plan_auto_expansion(
        &self,
        budget: &TokenBudget,
        tool_def_tokens: usize,
        wire_tokens: usize,
    ) -> Vec<AutoExpansionCandidate> {
        // Find the latest user message content.
        let user_text = self.active.iter().rev().find_map(|entry| {
            let msg = entry.message();
            if msg.get("role").and_then(|r| r.as_str()) == Some("user") {
                msg.get("content")
                    .and_then(|c| c.as_str())
                    .map(|s| s.to_string())
            } else {
                None
            }
        });

        let user_text = match user_text {
            Some(t) if !t.is_empty() => t,
            _ => return Vec::new(),
        };

        let user_keywords = extract_keywords(&user_text);
        if user_keywords.is_empty() {
            return Vec::new();
        }
        // Embed the user message once; None when the semantic feature is off or
        // the model is unavailable, in which case we fall back to keyword overlap.
        let user_embedding = crate::agent::embedder::embed_one(&user_text).ok();

        // Budget: never let expansion push the *wire* past τ_hard. Because we
        // APPEND originals (the summary stays in place, keeping the frozen prefix
        // byte-stable and the prompt cache warm), the cost is the full expansion,
        // not summary→original net. Counting the wire (not internal active) is
        // what prevents the feedback loop where reinjection exceeds headroom
        // because the engine's view misses the previously-appended wire tail.
        let available = budget.available_budget(tool_def_tokens);
        let hard_limit = (available as f64 * self.config.tau_hard) as usize;
        let mut headroom = hard_limit.saturating_sub(wire_tokens);
        if headroom < 100 {
            debug!(
                wire_tokens,
                hard_limit, "LCM auto_expand: no wire headroom, skipping"
            );
            return Vec::new();
        }

        let mut planned = Vec::new();
        let candidates: Vec<(usize, Value)> = self
            .active
            .iter()
            .filter_map(|e| match e {
                ContextEntry::Summary { node_id, message }
                    if !self.auto_expanded.contains(node_id) =>
                {
                    Some((*node_id, message.clone()))
                }
                _ => None,
            })
            .filter(|(node_id, _)| {
                // Fresh-summary cooldown: a node created within the last
                // FRESH_SUMMARY_COOLDOWN_TURNS turns is ineligible. Without
                // this, auto_expand on turn N+1 reinjects the originals of
                // a summary created on turn N — undoing the compaction
                // (live 2026-07-27 12:13:06 saw +12463 tokens reinjected 24
                // seconds after a successful 12463→1398 compaction).
                //
                // Nodes with `created_at_turn == 0` (back-compat rows from
                // older persisted sessions, or test scaffolding that never
                // set the turn) are always eligible — we have no creation
                // signal for them, and treating them as ancient history
                // preserves pre-cooldown behaviour.
                let Some(node) = self.dag.get(*node_id) else {
                    return true;
                };
                if node.created_at_turn == 0 {
                    return true;
                }
                let age = self.current_turn.saturating_sub(node.created_at_turn);
                age > FRESH_SUMMARY_COOLDOWN_TURNS
            })
            .collect();

        for (node_id, summary_message) in candidates {
            let source_ids = self.dag.all_source_ids(node_id);
            let source_messages: Vec<Value> = source_ids
                .iter()
                .filter_map(|&id| self.store.get(&id).cloned())
                .collect();
            if source_messages.is_empty() {
                continue;
            }
            let source_text = source_messages
                .iter()
                .filter_map(|m| m.get("content").and_then(|c| c.as_str()))
                .collect::<Vec<_>>()
                .join(" ");

            let relevance = self.relevance(
                &user_text,
                &user_keywords,
                user_embedding.as_deref(),
                &source_text,
            );
            if relevance < 0.3 {
                continue;
            }

            let expansion_tokens = TokenBudget::estimate_tokens(&source_messages);
            if expansion_tokens > headroom {
                debug!(
                    "LCM auto_expand: node {} relevant but needs {} tokens, {} headroom",
                    node_id, expansion_tokens, headroom
                );
                continue;
            }

            debug!(
                "LCM auto_expand: appending originals for node {} (relevance={:.2}, +{} tokens)",
                node_id, relevance, expansion_tokens
            );
            let first = source_ids.first().copied().unwrap_or(0);
            let last = source_ids.last().copied().unwrap_or(0);
            let body = source_ids
                .iter()
                .zip(source_messages.iter())
                .map(|(id, m)| {
                    let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("?");
                    let content = m.get("content").and_then(|c| c.as_str()).unwrap_or("");
                    format!("[msg {id}] {role}: {content}")
                })
                .collect::<Vec<_>>()
                .join("\n\n");
            let flattened_fallback = json!({
                "role": "user",
                "content": format!(
                    "[Auto-expanded originals for the summary of messages {first}-{last}:]\n\n{body}"
                ),
                // Synthetic: dropped from the wire's turn tags, and skipped by
                // find_oldest_raw_block so it is never itself re-compacted.
                "_synthetic": true,
            });
            headroom = headroom.saturating_sub(expansion_tokens);
            planned.push(AutoExpansionCandidate {
                node_id,
                source_ids,
                estimated_tokens: expansion_tokens,
                flattened_fallback,
                summary_message,
            });
        }

        planned
    }

    /// Mark one planned node consumed after the provider accepted the prompt.
    pub(crate) fn commit_auto_expansion(&mut self, node_id: usize) -> bool {
        self.dag.get(node_id).is_some() && self.auto_expanded.insert(node_id)
    }

    /// Atomically mark one provider attempt's planned nodes consumed.
    /// A stale or duplicate node rejects the whole batch so prompt publication
    /// cannot get ahead of only a partial eligibility commit.
    pub(crate) fn commit_auto_expansions(&mut self, node_ids: &[usize]) -> bool {
        let mut unique = std::collections::HashSet::with_capacity(node_ids.len());
        if node_ids.is_empty()
            || node_ids.iter().any(|node_id| {
                self.dag.get(*node_id).is_none()
                    || self.auto_expanded.contains(node_id)
                    || !unique.insert(*node_id)
            })
        {
            return false;
        }
        for node_id in node_ids {
            let committed = self.commit_auto_expansion(*node_id);
            debug_assert!(committed, "batch was fully validated before commit");
        }
        true
    }

    /// Relevance of a summary's source text to the user message. Uses semantic
    /// cosine similarity when embeddings are available, else keyword overlap.
    fn relevance(
        &self,
        _user_text: &str,
        user_keywords: &std::collections::HashSet<String>,
        user_embedding: Option<&[f32]>,
        source_text: &str,
    ) -> f64 {
        // ponytail: re-embeds each summary per turn. Cache on SummaryNode if the
        // summary count ever grows enough for ~5ms/embed to matter.
        if let Some(u) = user_embedding {
            if let Ok(s) = crate::agent::embedder::embed_one(source_text) {
                return crate::agent::embedder::cosine_similarity(u, &s) as f64;
            }
        }
        let source_keywords = extract_keywords(source_text);
        if source_keywords.is_empty() {
            return 0.0;
        }
        let overlap = user_keywords
            .iter()
            .filter(|kw| source_keywords.contains(*kw))
            .count();
        overlap as f64 / user_keywords.len() as f64
    }

    /// Rebuild the LCM engine from persisted summary nodes (from SQLite DB).
    ///
    /// This is the preferred way to restore the DAG after a restart — it uses
    /// the summary_nodes table instead of scanning Turn::Summary entries from
    /// the message stream.
    ///
    /// The store is keyed by each row's stable `_db_id` (SQLite rowid), so
    /// persisted `source_ids` resolve to exactly the same originals they were
    /// created against, no matter how the live context was windowed.
    ///
    /// Pre-rowid nodes with POSITIONAL source ids are purged once when the
    /// DB is opened (`SessionDb::new`) and filtered by `load_summary_nodes`,
    /// so every node received here is db-id-keyed (the 8th tuple element is
    /// always `"db_id"`).
    pub fn rebuild_from_db_nodes(
        raw_messages: &[Value],
        nodes: &[(
            usize,
            Vec<usize>,
            Vec<usize>,
            String,
            usize,
            u8,
            SummaryManifest,
            String,
        )],
        config: LcmConfig,
    ) -> Self {
        let mut engine = Self::new(config);

        // `/clear` is an append-only boundary in SQLite. Rebuild only the raw
        // rows and summary nodes wholly after the newest marker; otherwise an
        // old persisted summary can resurrect cleared conversation on restart,
        // and the internal `role: "clear"` marker itself can reach the provider.
        let clear_index = raw_messages
            .iter()
            .rposition(|message| message.get("role").and_then(Value::as_str) == Some("clear"));
        let clear_db_id = clear_index.map(|index| {
            raw_messages[index]
                .get("_db_id")
                .and_then(Value::as_u64)
                .map_or(usize::MAX, |id| id as usize)
        });
        let raw_after_clear = clear_index.map_or(raw_messages, |index| &raw_messages[index + 1..]);
        let node_is_current = |source_ids: &[usize]| {
            clear_db_id.is_none_or(|boundary| {
                !source_ids.is_empty() && source_ids.iter().all(|source_id| *source_id > boundary)
            })
        };

        // Ingest raw messages keyed by rowid. Persisted `role: "summary"` rows
        // reference store entries and never occupy store slots themselves;
        // synthetic scaffolds are not originals. Rows without a `_db_id`
        // cannot be addressed losslessly and are skipped (get_all_messages
        // always supplies the rowid, so this is defensive only).
        for msg in raw_after_clear {
            let role = msg.get("role").and_then(|r| r.as_str());
            if matches!(role, Some("summary" | "clear"))
                || crate::agent::markers::is_synthetic(msg)
                || msg.get("_lcm_summary").is_some()
            {
                continue;
            }
            let Some(db_id) = msg.get("_db_id").and_then(|v| v.as_u64()) else {
                warn!("LCM rebuild: skipping message without _db_id");
                continue;
            };
            engine.store.insert(db_id as usize, msg.clone());
        }

        // Track which message IDs are covered by summaries.
        let mut summarized_ids: std::collections::HashSet<MessageId> =
            std::collections::HashSet::new();

        // Only current nodes participate in reconstruction. In particular, a
        // stale/corrupt row must not retire a valid node merely by naming it
        // as a child.
        let valid_ids: std::collections::HashSet<usize> = nodes
            .iter()
            .filter(|(_, source_ids, _, _, _, _, _, _)| node_is_current(source_ids))
            .map(|(id, _, _, _, _, _, _, _)| *id)
            .collect();

        // Nodes merged into a newer summary are retired from the active DAG.
        // Re-activating them would duplicate content and re-accumulate summary
        // mass. Restrict child references to known valid nodes so malformed
        // dangling IDs cannot suppress unrelated roots.
        let subsumed: std::collections::HashSet<usize> = nodes
            .iter()
            .filter(|(_, source_ids, _, _, _, _, _, _)| node_is_current(source_ids))
            .flat_map(|(_, _, child_ids, _, _, _, _, _)| child_ids.iter().copied())
            .filter(|child_id| valid_ids.contains(child_id))
            .collect();

        // Reserve above every persisted ID, including retired children. This
        // prevents the next live compaction from reusing a sparse/subsumed ID
        // that still exists in SQLite.
        engine.dag.next_id = nodes
            .iter()
            .map(|(id, _, _, _, _, _, _, _)| *id)
            .max()
            .map_or(0, |id| id.saturating_add(1));

        // Reconstruct active DAG roots with their persisted IDs. Calling
        // create_node() here used to renumber sparse roots from zero while
        // leaving child_ids unchanged; a restored root could therefore point
        // at itself and recurse forever. Preserving IDs means every retained
        // child reference targets a retired (absent) node, so the reconstructed
        // in-memory graph cannot contain a cycle.
        let mut restored_ids = std::collections::HashSet::new();
        for (id, source_ids, child_ids, text, tokens, level, manifest, _id_kind) in nodes {
            if !node_is_current(source_ids) {
                debug!(
                    node_id = id,
                    "LCM rebuild: skipping summary node before the latest clear boundary"
                );
                continue;
            }
            if !restored_ids.insert(*id) {
                warn!(
                    node_id = id,
                    "LCM rebuild: skipping duplicate summary node ID"
                );
                continue;
            }
            if subsumed.contains(id) {
                debug!(
                    node_id = id,
                    "LCM rebuild: skipping subsumed (merged) summary node"
                );
                continue;
            }
            engine.dag.nodes.push(SummaryNode {
                id: *id,
                source_ids: source_ids.clone(),
                child_summaries: child_ids.clone(),
                text: text.clone(),
                manifest: manifest.clone(),
                tokens: *tokens,
                level: *level,
                // Persisted rows restore with their original creation turn
                // where available; older rows without the column deserialize
                // as 0 and are treated as ancient history by the cooldown.
                created_at_turn: 0,
            });
            for &sid in source_ids {
                summarized_ids.insert(sid);
            }
        }
        engine.dag.nodes.sort_by_key(|node| node.id);

        let unsummarized: Vec<(MessageId, Value)> = engine
            .store
            .iter()
            .filter(|(id, _)| !summarized_ids.contains(id))
            .map(|(&id, message)| (id, message.clone()))
            .collect();

        // Live compaction retains the leading system message, inserts summaries
        // after it, then keeps the unsummarized conversational tail. Rebuild
        // must use the same order or a restart changes the prompt bytes.
        for (msg_id, message) in &unsummarized {
            if message.get("role").and_then(Value::as_str) == Some("system") {
                engine.active.push(ContextEntry::Raw {
                    msg_id: *msg_id,
                    message: message.clone(),
                });
            }
        }
        for node in &engine.dag.nodes {
            let summary_message =
                summary_wire_message(&node.source_ids, &node.text, &node.manifest);
            engine.active.push(ContextEntry::Summary {
                node_id: node.id,
                message: summary_message,
            });
        }

        // BTreeMap iteration is ascending by id — session order is preserved.
        for (msg_id, message) in unsummarized {
            if message.get("role").and_then(Value::as_str) == Some("system") {
                continue;
            }
            engine.active.push(ContextEntry::Raw { msg_id, message });
        }

        debug!(
            "LCM rebuild_from_db: {} store entries, {} DAG nodes, {} active entries",
            engine.store.len(),
            engine.dag.len(),
            engine.active.len(),
        );

        engine
    }
}

// ---------------------------------------------------------------------------
// Compaction Action
// ---------------------------------------------------------------------------

/// What the control loop should do after checking thresholds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompactionAction {
    /// Context is within soft threshold — no action needed.
    None,
    /// Context exceeds soft threshold — trigger async compaction (non-blocking).
    Async,
    /// Context exceeds hard threshold — must compact NOW (blocking).
    Blocking,
}

/// What to do when model-backed summarization cannot produce a valid smaller
/// summary. Soft pressure preserves context; hard pressure must converge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompactionFailureMode {
    PreserveContext,
    Deterministic,
}

// ---------------------------------------------------------------------------
// Three-Level Escalation (Algorithm 3)
// ---------------------------------------------------------------------------


/// Extract one mechanical headline line per message in the eviction span.
/// No LLM calls — deterministic, zero-latency. Returns (text, manifest).
/// The manifest is default (no structured extraction) — open_loops and
/// decisions require model comprehension that this path deliberately avoids.
fn mechanical_headlines(messages: &[Value]) -> (String, SummaryManifest) {
    let mut lines = Vec::new();
    for msg in messages {
        let role = msg.get("role").and_then(Value::as_str).unwrap_or("");
        let content = msg.get("content").and_then(Value::as_str).unwrap_or("");
        if content.trim().is_empty() || role == "system" {
            continue;
        }
        // For tool results: use the TOOL_RESULT_HANDLE excerpt (first line
        // after the marker, already truncated to 160 chars at ingestion).
        let first_line = if role == "tool" {
            content
                .lines()
                .find(|l| !l.trim().is_empty() && !l.starts_with("TOOL_RESULT_HANDLE"))
                .unwrap_or("")
                .trim()
        } else {
            content.lines().next().unwrap_or("").trim()
        };
        if first_line.is_empty() {
            continue;
        }
        let boundary = crate::utils::helpers::floor_char_boundary(first_line, 150);
        let headline = &first_line[..boundary];
        match role {
            "user" => lines.push(format!("· user: {headline}")),
            "assistant" => lines.push(format!("· assistant: {headline}")),
            _ => lines.push(format!("· {role}: {headline}")),
        }
    }
    (lines.join("\n"), SummaryManifest::default())
}

/// Escalated summarization: tries increasingly aggressive LLM strategies.
///
/// Returns `Ok(Some((summary_text, manifest, escalation_level)))` on success,
/// `Ok(None)` when there is no compactor to summarize with or every level
/// produced output that failed the acceptance checks, and `Err` when the LLM
/// call itself failed or the output scored as babble/degenerate — both fatal
/// to this compaction attempt, with no retry ladder. There is no deterministic
/// truncation fallback: any failure here leaves the context uncompacted this
/// round rather than silently degrading it.
async fn escalated_summary(
    messages: &[Value],
    _target_tokens: usize,
    compactor: Option<&ContextCompactor>,
) -> Result<Option<(String, SummaryManifest, u8)>> {
    let original_tokens = TokenBudget::estimate_tokens(messages);

    let Some(compactor) = compactor else {
        debug!("LCM escalation: no compactor available, leaving context uncompacted");
        return Ok(None);
    };

    // Level 0: Mechanical headlines — one line per message from the eviction
    // span. Zero LLM calls, zero latency. Scroll (arXiv:2608.21690 §4.3)
    // shows lossy LLM summarization degrades accuracy 73→20 while
    // recoverable eviction (originals in SQLite, lcm_expand) stays ≥86.
    // Mechanical headlines preserve the same recoverability invariant.
    // Mechanical headlines only when the eviction span contains tool
    // results (large stashed outputs — the paging-loop pain point). Pure
    // conversation messages go through the LLM path for richer summaries.
    let has_tool_results = messages
        .iter()
        .any(|m| m.get("role").and_then(Value::as_str) == Some("tool"));
    if has_tool_results {
        let (mech_summary, mech_manifest) = mechanical_headlines(messages);
        if mech_summary.chars().count() >= 200 {
            info!(
                chars = mech_summary.chars().count(),
                "LCM Level 0: mechanical headlines (zero LLM calls)"
            );
            return Ok(Some((mech_summary, mech_manifest, 0)));
        }
    }

    // Level 1: Preserve details. Only a completed but insufficient summary may
    // escalate. A transport or fidelity-gate error is indeterminate and must
    // not launch another generation behind work the backend may still own.
    match compactor
        .summarize_for_lcm(messages, "preserve_details")
        .await
    {
        Ok(reply) => {
            let (summary, manifest) = extract_summary_manifest(reply);
            if summary_is_acceptable(&summary, original_tokens, 1)? {
                return Ok(Some((summary, manifest, 1)));
            }
        }
        Err(error) => {
            return Err(error);
        }
    }

    // Level 2: Bullet points, more aggressive compression. This is the last
    // level — an error here has nowhere left to escalate to, so it
    // propagates as the caller-visible failure for this compaction attempt.
    let reply = compactor
        .summarize_for_lcm(messages, "bullet_points")
        .await?;
    let (summary, manifest) = extract_summary_manifest(reply);
    if summary_is_acceptable(&summary, original_tokens, 2)? {
        return Ok(Some((summary, manifest, 2)));
    }
    Ok(None)
}

fn summary_is_acceptable(summary: &str, original_tokens: usize, level: u8) -> Result<bool> {
    if summary.trim().is_empty() {
        debug!(
            "LCM escalation: Level {} had no prose after manifest extraction",
            level
        );
        return Ok(false);
    }
    if let Some(score) = anti_drift::score_summary_babble(summary) {
        warn!(
            score,
            "LCM escalation: Level {} summary scored as babble, rejecting", level
        );
        anyhow::bail!("LCM summary rejected as babble (score={:.2})", score);
    }
    if contains_refusal_pattern(summary) {
        debug!("LCM escalation: Level {} contained refusal pattern", level);
        return Ok(false);
    }
    let tokens = TokenBudget::estimate_str_tokens(summary);
    if tokens >= original_tokens {
        debug!(
            "LCM escalation: Level {} failed (output {} >= input {})",
            level, tokens, original_tokens
        );
        return Ok(false);
    }
    debug!(
        "LCM escalation: Level {} succeeded ({} -> {} tokens)",
        level, original_tokens, tokens
    );
    Ok(true)
}

/// Deterministic truncation: extract key facts without any LLM call.
///
/// Strategy: Keep first sentence of each user message, skip tool results,
/// keep first sentence of assistant responses. Guaranteed to produce
/// output ≤ target_tokens.
fn deterministic_truncate(messages: &[Value], target_tokens: usize) -> String {
    let mut lines = Vec::new();

    for msg in messages {
        let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("");
        let content = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");

        match role {
            "user" => {
                // Keep first sentence.
                let first = first_sentence(content);
                if !first.is_empty() {
                    lines.push(format!("User: {}", first));
                }
            }
            "assistant" => {
                // Keep first sentence, skip if tool-call-only.
                if !content.is_empty() {
                    let first = first_sentence(content);
                    if !first.is_empty() {
                        lines.push(format!("Assistant: {}", first));
                    }
                }
            }
            "tool" => {
                // Just note the tool name and result length.
                let name = msg.get("name").and_then(|n| n.as_str()).unwrap_or("tool");
                let len = content.len();
                lines.push(format!("[Tool {}: {} chars]", name, len));
            }
            _ => {}
        }

        // Check token budget after each message.
        let current = lines.join("\n");
        if TokenBudget::estimate_str_tokens(&current) >= target_tokens {
            break;
        }
    }

    let result = lines.join("\n");
    // Final clamp: if still over budget, hard-truncate by characters.
    let target_chars = target_tokens * 4; // ~4 chars per token
    if result.len() > target_chars {
        result.chars().take(target_chars).collect()
    } else {
        result
    }
}

/// Extract the first sentence from text.
fn first_sentence(text: &str) -> &str {
    let trimmed = text.trim();
    // Find first sentence boundary (. ! ? followed by space or end).
    for (i, c) in trimmed.char_indices() {
        if (c == '.' || c == '!' || c == '?') && i > 0 {
            let next = trimmed[i + c.len_utf8()..].chars().next();
            if next.is_none() || next == Some(' ') || next == Some('\n') {
                return &trimmed[..=i];
            }
        }
    }
    // No sentence boundary found — take first 200 chars.
    let end = trimmed
        .char_indices()
        .nth(200)
        .map(|(i, _)| i)
        .unwrap_or(trimmed.len());
    &trimmed[..end]
}

/// Check if text contains an LLM refusal pattern.
///
/// Refusal patterns indicate the LLM declined to help, which should not
/// be captured in summaries as it pollutes future context.
pub fn contains_refusal_pattern(text: &str) -> bool {
    let lower = text.to_lowercase();
    let refusal_indicators = [
        "i cannot",
        "i can't",
        "i'm unable",
        "i am unable",
        "i apologize",
        "i'm sorry",
        "as an ai",
        "as a language model",
        "unable to help",
        "cannot assist",
        "can't assist",
        "cannot fulfill",
        "can't fulfill",
        "not able to provide",
        "unable to provide",
        "i won't",
        "i will not",
        "against my guidelines",
        "violates my",
        "i'm not comfortable",
        "i am not comfortable",
        // NB: bare topic words like "harmful"/"ethical"/"unethical" were removed —
        // they flag legitimate summaries of conversations *about* harm or ethics
        // (e.g. "summary of the discussion on harmful algae blooms") as refusals.
        // Only first-person refusal phrasing belongs here.
    ];

    for indicator in &refusal_indicators {
        if lower.contains(indicator) {
            return true;
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Keyword extraction for auto-expand relevance scoring
// ---------------------------------------------------------------------------

/// Extract significant keywords from text for relevance matching.
///
/// Lowercases, splits on non-alphanumeric boundaries, and filters out
/// stopwords and short words. Returns a set of unique keywords.
fn extract_keywords(text: &str) -> std::collections::HashSet<String> {
    static STOPWORDS: &[&str] = &[
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "will", "would", "could", "should", "may", "might", "shall", "can",
        "need", "must", "ought", "i", "me", "my", "we", "our", "you", "your", "he", "she", "it",
        "they", "them", "their", "this", "that", "these", "those", "what", "which", "who", "whom",
        "how", "when", "where", "why", "and", "but", "or", "nor", "not", "no", "so", "if", "then",
        "for", "with", "about", "from", "into", "of", "to", "in", "on", "at", "by", "as", "up",
        "out", "off", "over", "under", "just", "also", "very", "really", "quite", "much", "more",
        "some", "any", "all", "each", "every", "both", "few", "many", "use", "tell", "let", "see",
        "get", "got", "make", "made", "know", "think", "want", "like", "said", "say", "help",
    ];

    let lower = text.to_lowercase();
    lower
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|w| w.len() >= 3 && !STOPWORDS.contains(w))
        .map(|w| w.to_string())
        .collect()
}

// ---------------------------------------------------------------------------
// LCM Expand Tool
// ---------------------------------------------------------------------------

/// Tool that retrieves original messages from the LCM immutable store.
///
/// When summaries replace message blocks in the active context, the LLM
/// can call this tool to recover the full original messages by their IDs.
pub struct LcmExpandTool {
    engine: Arc<Mutex<LcmEngine>>,
}

impl LcmExpandTool {
    pub fn new(engine: Arc<Mutex<LcmEngine>>) -> Self {
        Self { engine }
    }
}

#[async_trait]
impl crate::agent::tools::base::Tool for LcmExpandTool {
    fn name(&self) -> &str {
        "lcm_expand"
    }

    fn description(&self) -> &str {
        "Retrieve the exact original messages behind a compressed \
         [Summary of messages …] block FROM THE CURRENT SESSION. Copy the range \
         from that block, e.g. lcm_expand({\"message_ids\": \"120-158\"}). \
         Expansion is lossless — the originals are always available. \
         NOTE: this only works on summary blocks in the current conversation; it \
         cannot reach past sessions. To retrieve the FULL transcript of a DIFFERENT \
         past session, use session_search(mode=\"session\", session=KEY) with that \
         session's key instead."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "message_ids": {
                    // No "type": accepts the range string from the summary
                    // header, an integer array, or a single integer — small
                    // models emit all three shapes.
                    "description": "Message IDs to expand: the range string shown in the summary block (e.g. \"120-158\" or \"120-140,150-158\"), or an array of integers."
                }
            },
            "required": ["message_ids"]
        })
    }

    async fn execute(
        &self,
        params: HashMap<String, Value>,
        _ctx: &crate::agent::tools::base::ToolContext,
    ) -> crate::agent::tools::base::ToolResult {
        // Lenient by design: small models emit arrays, bare strings, ranges, or
        // stray prose. Accept all of them rather than silently returning nothing.
        let msg_ids = match params.get("message_ids") {
            Some(v) => parse_message_ids(v),
            None => Vec::new(),
        };

        if msg_ids.is_empty() {
            return Err(crate::errors::ToolError::InvalidArgs {
                message: "no valid message IDs provided. lcm_expand only expands LCM \
                    summary blocks from the CURRENT session, and needs IDs from a \
                    [Summary … (IDs: …)] block as an array, e.g. [5, 6, 7, 8]. To read a \
                    PAST session's full transcript, use session_search(mode=\"session\", \
                    session=KEY) with that session's key (search first to get the key)."
                    .to_string(),
            });
        }

        let engine = self.engine.lock().await;
        Ok(engine.format_expanded(&msg_ids).into())
    }
}

/// Extract message IDs from whatever shape the model produced: an integer array,
/// a comma/space-separated string, a `"5-8"` range, or numbers embedded in prose.
fn parse_message_ids(v: &Value) -> Vec<usize> {
    match v {
        Value::Array(arr) => arr
            .iter()
            .flat_map(|e| match e {
                Value::Number(n) => n.as_u64().map(|x| x as usize).into_iter().collect(),
                Value::String(s) => parse_id_runs(s),
                _ => Vec::new(),
            })
            .collect(),
        Value::String(s) => parse_id_runs(s),
        Value::Number(n) => n.as_u64().map(|x| x as usize).into_iter().collect(),
        _ => Vec::new(),
    }
}

/// Parse integer IDs and `a-b` ranges out of a free-form string.
fn parse_id_runs(s: &str) -> Vec<usize> {
    let mut ids = Vec::new();
    // Tokens are runs of digits and '-'; everything else (commas, brackets,
    // spaces, prose) is a separator.
    for tok in s.split(|c: char| !c.is_ascii_digit() && c != '-') {
        let tok = tok.trim_matches('-');
        if tok.is_empty() {
            continue;
        }
        if let Some((a, b)) = tok.split_once('-') {
            if let (Ok(a), Ok(b)) = (a.parse::<usize>(), b.parse::<usize>()) {
                // Cap runaway ranges so a typo can't allocate millions of IDs.
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::tools::base::Tool;
    use crate::providers::base::{FinishReason, LLMProvider, LLMResponse};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    /// Mock LLM that returns a short summary — short enough that Level 1
    /// succeeds (fewer tokens than the original block).
    struct SummarizerMock;

    #[async_trait]
    impl LLMProvider for SummarizerMock {
        async fn chat(
            &self,
            _messages: &[Value],
            _tools: Option<&[Value]>,
            _model: Option<&str>,
            _max_tokens: u32,
            _temperature: f64,
            _thinking_budget: Option<u32>,
            _top_p: Option<f64>,
        ) -> anyhow::Result<LLMResponse> {
            Ok(LLMResponse {
                content: Some("- User asked multiple questions about Rust ownership.".to_string()),
                tool_calls: vec![],
                finish_reason: FinishReason::Stop,
                usage: std::collections::HashMap::new(),
            })
        }

        fn get_default_model(&self) -> &str {
            "mock-summarizer"
        }
    }

    struct ManifestSummarizerMock;

    #[async_trait]
    impl LLMProvider for ManifestSummarizerMock {
        async fn chat(
            &self,
            _messages: &[Value],
            _tools: Option<&[Value]>,
            _model: Option<&str>,
            _max_tokens: u32,
            _temperature: f64,
            _thinking_budget: Option<u32>,
            _top_p: Option<f64>,
        ) -> anyhow::Result<LLMResponse> {
            Ok(LLMResponse {
                content: Some(
                    "- Preserved the restart rendering contract.\n\n\
                     ```json\n\
                     {\"open_loops\":[{\"text\":\"Verify restart rendering\",\"sources\":[2,3]}],\
                     \"failed_approaches\":[{\"text\":\"JSON-only prompt state\",\"sources\":[4]}],\
                     \"decisions\":[{\"text\":\"Keep one wire formatter\",\"sources\":[5,6]}]}\n\
                     ```"
                    .to_string(),
                ),
                tool_calls: vec![],
                finish_reason: FinishReason::Stop,
                usage: std::collections::HashMap::new(),
            })
        }

        fn get_default_model(&self) -> &str {
            "mock-manifest-summarizer"
        }
    }

    /// Mock LLM that returns an error — forces Level 3 deterministic fallback.
    struct FailingMock;

    #[async_trait]
    impl LLMProvider for FailingMock {
        async fn chat(
            &self,
            _messages: &[Value],
            _tools: Option<&[Value]>,
            _model: Option<&str>,
            _max_tokens: u32,
            _temperature: f64,
            _thinking_budget: Option<u32>,
            _top_p: Option<f64>,
        ) -> anyhow::Result<LLMResponse> {
            Err(anyhow::anyhow!("No LLM available"))
        }

        fn get_default_model(&self) -> &str {
            "mock-failing"
        }
    }

    struct CountingFailingMock {
        calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl LLMProvider for CountingFailingMock {
        async fn chat(
            &self,
            _messages: &[Value],
            _tools: Option<&[Value]>,
            _model: Option<&str>,
            _max_tokens: u32,
            _temperature: f64,
            _thinking_budget: Option<u32>,
            _top_p: Option<f64>,
        ) -> anyhow::Result<LLMResponse> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Err(anyhow::anyhow!("indeterminate transport timeout"))
        }

        fn get_default_model(&self) -> &str {
            "mock-counting-failing"
        }
    }

    #[tokio::test]
    async fn transport_failure_does_not_launch_level_two_retry() {
        let calls = Arc::new(AtomicUsize::new(0));
        let compactor = ContextCompactor::new(
            Arc::new(CountingFailingMock {
                calls: calls.clone(),
            }),
            "mock".to_string(),
            4096,
        );
        let messages = vec![
            json!({"role": "user", "content": "Preserve this compaction source."}),
            json!({"role": "assistant", "content": "Preserve this response too."}),
        ];

        let error = escalated_summary(&messages, 64, Some(&compactor))
            .await
            .unwrap_err()
            .to_string();

        assert!(error.contains("indeterminate transport timeout"), "{error}");
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "an indeterminate generation must not be retried"
        );
    }

    /// Mock LLM that reflects real input content back as short bullets
    /// instead of a fixed canned string. Unlike `SummarizerMock`, this
    /// survives the compaction fidelity gate's topic-anchor checks
    /// (`compaction.rs::collect_topic_anchors`) across arbitrary/evolving
    /// input, because it echoes the leading words of every source line
    /// rather than always returning the same text.
    struct EchoSummarizerMock;

    #[async_trait]
    impl LLMProvider for EchoSummarizerMock {
        async fn chat(
            &self,
            messages: &[Value],
            _tools: Option<&[Value]>,
            _model: Option<&str>,
            _max_tokens: u32,
            _temperature: f64,
            _thinking_budget: Option<u32>,
            _top_p: Option<f64>,
        ) -> anyhow::Result<LLMResponse> {
            let request = messages
                .iter()
                .rev()
                .find_map(|m| m.get("content").and_then(|c| c.as_str()))
                .unwrap_or("");
            let source = request
                .split("[SOURCE_BEGIN]\n")
                .nth(1)
                .and_then(|s| s.split("\n[SOURCE_END]").next())
                .unwrap_or(request);
            let heads: Vec<String> = source
                .lines()
                .filter(|line| !line.trim().is_empty())
                .map(|line| {
                    line.split_whitespace()
                        .take(3)
                        .collect::<Vec<_>>()
                        .join(" ")
                })
                .collect();
            // The fidelity gate caps a handoff at 15 bullets; group source
            // lines into at most 15 buckets so every line still contributes
            // its distinguishing words somewhere in the output.
            let chunk_size = heads.len().div_ceil(15).max(1);
            let bullets: Vec<String> = heads
                .chunks(chunk_size)
                .map(|group| format!("- {}", group.join("; ")))
                .collect();
            Ok(LLMResponse {
                content: Some(bullets.join("\n")),
                tool_calls: vec![],
                finish_reason: FinishReason::Stop,
                usage: std::collections::HashMap::new(),
            })
        }

        fn get_default_model(&self) -> &str {
            "mock-echo-summarizer"
        }
    }

    /// Mock LLM that returns a degenerate, filler-heavy wall of text — scores
    /// as babble under the anti-drift pollution heuristics regardless of
    /// input, so the babble gate must reject it before persistence.
    struct BabbleMock;

    #[async_trait]
    impl LLMProvider for BabbleMock {
        async fn chat(
            &self,
            _messages: &[Value],
            _tools: Option<&[Value]>,
            _model: Option<&str>,
            _max_tokens: u32,
            _temperature: f64,
            _thinking_budget: Option<u32>,
            _top_p: Option<f64>,
        ) -> anyhow::Result<LLMResponse> {
            let babble = format!(
                "well so basically actually honestly {}I ran the whole thing myself",
                "well so basically actually honestly ".repeat(40)
            );
            Ok(LLMResponse {
                content: Some(babble),
                tool_calls: vec![],
                finish_reason: FinishReason::Stop,
                usage: std::collections::HashMap::new(),
            })
        }

        fn get_default_model(&self) -> &str {
            "mock-babble"
        }
    }

    /// Mock LLM that panics if called — proves the LLM was never invoked.
    struct PanickingMock;

    #[async_trait]
    impl LLMProvider for PanickingMock {
        async fn chat(
            &self,
            _messages: &[Value],
            _tools: Option<&[Value]>,
            _model: Option<&str>,
            _max_tokens: u32,
            _temperature: f64,
            _thinking_budget: Option<u32>,
            _top_p: Option<f64>,
        ) -> anyhow::Result<LLMResponse> {
            panic!("LLM should not be called for huge compaction blocks")
        }

        fn get_default_model(&self) -> &str {
            "mock-panicking"
        }
    }

    /// Test message tagged with an explicit `_db_id`, mirroring what
    /// get_history supplies for persisted messages.
    fn msg(id: usize, role: &str, content: &str) -> Value {
        json!({"role": role, "content": content, "_db_id": id})
    }

    /// Ingest a `_db_id`-tagged message, asserting it was accepted.
    fn ingest(engine: &mut LcmEngine, id: usize, role: &str, content: &str) {
        assert_eq!(engine.ingest(msg(id, role, content)), Some(id));
    }

    fn plan_and_commit_auto_expansion(
        engine: &mut LcmEngine,
        budget: &TokenBudget,
        tool_def_tokens: usize,
        wire_tokens: usize,
    ) -> Vec<Value> {
        let candidates = engine.plan_auto_expansion(budget, tool_def_tokens, wire_tokens);
        candidates
            .into_iter()
            .map(|candidate| {
                assert!(engine.commit_auto_expansion(candidate.node_id));
                candidate.flattened_fallback
            })
            .collect()
    }

    fn serialize_wire(entries: &[ContextEntry]) -> Vec<u8> {
        let mut bytes = Vec::new();
        for entry in entries {
            bytes.extend(serde_json::to_vec(entry.message()).unwrap());
            bytes.push(b'\n');
        }
        bytes
    }

    /// Append-by-default must still self-regulate: over a long run the engine
    /// has to reach the merge threshold on its own and collapse accumulated
    /// summaries, without a test forcing the mode. If `block_selection` never
    /// returned `MergeSummaries`, summary mass would grow without bound and
    /// every other compaction test would still pass.
    #[tokio::test]
    async fn summary_mass_stays_bounded_across_many_compactions() {
        let mut engine = LcmEngine::new(LcmConfig {
            tau_soft: 0.3,
            tau_hard: 0.6,
            deterministic_target: 64,
        });
        let budget = TokenBudget::new(4096, 1024);
        let compactor = ContextCompactor::new(
            Arc::new(SummarizerMock) as Arc<dyn LLMProvider>,
            "mock".to_string(),
            16_384,
        );
        let body = "bounded summary mass over a long agentic session with many compactions ";
        let ceiling = (budget.available_budget(0) as f64 * SUMMARY_MERGE_BUDGET_FRACTION) as usize;

        let summary_tokens = |engine: &LcmEngine| -> usize {
            engine
                .active_entries()
                .iter()
                .filter_map(|entry| match entry {
                    ContextEntry::Summary { node_id, .. } => {
                        engine.dag().get(*node_id).map(|node| node.tokens)
                    }
                    ContextEntry::Raw { .. } => None,
                })
                .fold(0usize, usize::saturating_add)
        };

        ingest(&mut engine, 1, "system", "System");
        let mut next_id = 2;
        let mut merge_observed = false;
        let mut peak_mass = 0usize;

        for _round in 0..120 {
            for i in 0..12 {
                ingest(&mut engine, next_id, "user", &format!("{i}: {body}"));
                next_id += 1;
                ingest(
                    &mut engine,
                    next_id,
                    "assistant",
                    &format!("reply {i}: {body}"),
                );
                next_id += 1;
            }
            let before = summary_tokens(&engine);
            engine
                .compact(
                    Some(&compactor),
                    &budget,
                    0,
                    CompactionFailureMode::Deterministic,
                )
                .await;
            let after = summary_tokens(&engine);
            eprintln!(
                "round: summaries={} mass={} -> {}",
                engine
                    .active_entries()
                    .iter()
                    .filter(|e| matches!(e, ContextEntry::Summary { .. }))
                    .count(),
                before,
                after
            );
            // A merge is the only way accumulated summary mass goes down.
            merge_observed |= after < before;
            peak_mass = peak_mass.max(after);
        }

        assert!(
            merge_observed,
            "engine never merged on its own across 120 compactions — summary mass \
             is not self-regulating (peak {peak_mass} tokens, ceiling {ceiling})"
        );
        // One summary may push past the threshold before the next call merges,
        // so the bound is the ceiling plus a single summary's worth.
        let slack = ceiling * 2;
        assert!(
            peak_mass <= slack,
            "summary mass peaked at {peak_mass} tokens, unbounded past {slack}"
        );
    }

    #[test]
    fn manifest_merge_unions_open_loops_and_deduplicates_sources() {
        let first = SummaryManifest {
            open_loops: vec![
                ManifestItem {
                    text: "Ship the release".to_string(),
                    sources: vec![3, 1],
                },
                ManifestItem {
                    text: "Write the changelog".to_string(),
                    sources: vec![4],
                },
            ],
            ..SummaryManifest::default()
        };
        let second = SummaryManifest {
            open_loops: vec![ManifestItem {
                text: "  ship THE release  ".to_string(),
                sources: vec![2, 3],
            }],
            ..SummaryManifest::default()
        };

        let merged = SummaryManifest::merge(&[&first, &second]);

        assert_eq!(
            merged.open_loops,
            vec![
                ManifestItem {
                    text: "Ship the release".to_string(),
                    sources: vec![1, 2, 3],
                },
                ManifestItem {
                    text: "Write the changelog".to_string(),
                    sources: vec![4],
                },
            ]
        );
    }

    #[test]
    fn manifest_merge_moves_later_category_to_decisions() {
        let earlier = SummaryManifest {
            open_loops: vec![ManifestItem {
                text: "Use SQLite for persistence".to_string(),
                sources: vec![7],
            }],
            ..SummaryManifest::default()
        };
        let later = SummaryManifest {
            decisions: vec![ManifestItem {
                text: " use sqlite FOR persistence ".to_string(),
                sources: vec![9],
            }],
            ..SummaryManifest::default()
        };

        let merged = SummaryManifest::merge(&[&earlier, &later]);

        assert!(merged.open_loops.is_empty());
        assert_eq!(
            merged.decisions,
            vec![ManifestItem {
                text: "Use SQLite for persistence".to_string(),
                sources: vec![7, 9],
            }]
        );
    }

    #[tokio::test]
    async fn non_empty_manifest_renders_identically_live_and_after_sqlite_restart() {
        let temp = tempfile::tempdir().unwrap();
        let db_path = temp.path().join("sessions.db");
        let db = crate::session::SessionDb::new(&db_path);
        let session = db.create_session("manifest-wire-restart").await;
        let detail = "prefix stability restart persistence manifest rendering ".repeat(8);
        let mut messages = vec![json!({"role": "system", "content": "System prompt."})];
        for i in 0..12 {
            messages.push(json!({
                "role": "user",
                "content": format!("Question {i}: {detail}")
            }));
            messages.push(json!({
                "role": "assistant",
                "content": format!("Answer {i}: {detail}")
            }));
        }
        db.add_messages(&session.id, &messages).await;

        let raw_messages = db.get_all_messages(&session.id).await;
        let mut live = LcmEngine::new(LcmConfig::default());
        for message in &raw_messages {
            live.ingest(message.clone());
        }
        let compactor = ContextCompactor::new(
            Arc::new(ManifestSummarizerMock) as Arc<dyn LLMProvider>,
            "mock".to_string(),
            16_384,
        );
        let result = live
            .compact(
                Some(&compactor),
                &TokenBudget::new(4096, 1024),
                0,
                CompactionFailureMode::Deterministic,
            )
            .await;
        assert!(result.is_some(), "live compaction must produce a summary");

        let node = live.dag().newest().unwrap().clone();
        let live_message = live
            .active_entries()
            .iter()
            .find_map(|entry| match entry {
                ContextEntry::Summary { node_id, message } if *node_id == node.id => {
                    Some(message.clone())
                }
                _ => None,
            })
            .unwrap();
        let live_content = live_message["content"].as_str().unwrap();
        assert!(live_content.contains("[State manifest]"));
        assert!(live_content.contains("Open loops:\n- Verify restart rendering [sources: 2-3]"));
        assert!(live_content.contains("Failed approaches:\n- JSON-only prompt state [sources: 4]"));
        assert!(live_content.contains("Decisions:\n- Keep one wire formatter [sources: 5-6]"));

        db.save_summary_node(
            &session.id,
            node.id,
            &node.source_ids,
            &node.child_summaries,
            &node.text,
            node.tokens,
            node.level,
            &node.manifest,
        )
        .await;
        let persisted_nodes = db.load_summary_nodes(&session.id).await;
        let reloaded_messages = db.get_all_messages(&session.id).await;
        let rebuilt = LcmEngine::rebuild_from_db_nodes(
            &reloaded_messages,
            &persisted_nodes,
            LcmConfig::default(),
        );
        let rebuilt_message = rebuilt
            .active_entries()
            .iter()
            .find_map(|entry| match entry {
                ContextEntry::Summary { node_id, message } if *node_id == node.id => {
                    Some(message.clone())
                }
                _ => None,
            })
            .unwrap();

        assert_eq!(live_message, rebuilt_message);
        assert_eq!(
            live.active_context(),
            rebuilt.active_context(),
            "restart must preserve the complete live wire ordering"
        );
    }

    #[test]
    fn test_summary_dag_create_and_retrieve() {
        let mut dag = SummaryDag::new();
        dag.create_node(
            vec![0, 1, 2],
            vec![],
            "Summary of first 3 messages.".to_string(),
            SummaryManifest::default(),
            1,
        );
        assert_eq!(dag.len(), 1);
        let node = dag.get(0).unwrap();
        assert_eq!(node.source_ids, vec![0, 1, 2]);
        assert_eq!(node.level, 1);
    }

    #[test]
    fn test_summary_dag_all_source_ids() {
        let mut dag = SummaryDag::new();
        dag.create_node(
            vec![0, 1, 2],
            vec![],
            "First batch.".to_string(),
            SummaryManifest::default(),
            1,
        );
        dag.create_node(
            vec![3, 4, 5],
            vec![],
            "Second batch.".to_string(),
            SummaryManifest::default(),
            1,
        );
        let ids = dag.all_source_ids(0);
        assert_eq!(ids, vec![0, 1, 2]);
    }

    #[test]
    fn test_lcm_engine_ingest() {
        let engine = &mut LcmEngine::new(LcmConfig::default());
        let id0 = engine.ingest(msg(1, "system", "You are helpful."));
        let id1 = engine.ingest(msg(2, "user", "Hello"));
        assert_eq!(id0, Some(1));
        assert_eq!(id1, Some(2));
        assert_eq!(engine.store_len(), 2);
        assert_eq!(engine.active_len(), 2);
        assert_eq!(engine.store_ids(), vec![1, 2]);
    }

    #[test]
    fn test_ingest_skips_unpersisted_messages() {
        let engine = &mut LcmEngine::new(LcmConfig::default());

        // No `_db_id` → not persisted yet → not ingested.
        assert_eq!(
            engine.ingest(json!({"role": "user", "content": "not persisted"})),
            None
        );
        assert_eq!(engine.store_len(), 0);
        assert_eq!(engine.active_len(), 0);

        // Same message re-offered WITH a `_db_id` ingests exactly once.
        let m = msg(7, "user", "now persisted");
        assert_eq!(engine.ingest(m.clone()), Some(7));
        assert_eq!(engine.ingest(m), Some(7), "upsert must be idempotent");
        assert_eq!(engine.store_len(), 1);
        assert_eq!(engine.active_len(), 1, "no duplicate active entry");
    }

    #[test]
    fn test_lcm_engine_expand() {
        let engine = &mut LcmEngine::new(LcmConfig::default());
        ingest(engine, 1, "user", "Hello");
        ingest(engine, 2, "assistant", "Hi there!");
        ingest(engine, 3, "user", "How are you?");

        let expanded = engine.expand(&[1, 3]);
        assert_eq!(expanded.len(), 2);
        assert_eq!(expanded[0].0, 1);
        assert_eq!(expanded[1].0, 3);
    }

    #[test]
    fn test_lcm_engine_format_expanded() {
        let engine = &mut LcmEngine::new(LcmConfig::default());
        ingest(engine, 1, "user", "Hello");
        ingest(engine, 2, "assistant", "Hi!");

        let output = engine.format_expanded(&[1, 2]);
        assert!(output.contains("[msg 1] user: Hello"));
        assert!(output.contains("[msg 2] assistant: Hi!"));
    }

    #[test]
    fn test_check_thresholds_none() {
        let engine = &mut LcmEngine::new(LcmConfig {
            tau_soft: 0.5,
            tau_hard: 0.85,
            deterministic_target: 512,
        });
        ingest(engine, 1, "system", "S");
        ingest(engine, 2, "user", "Hi");

        let budget = TokenBudget::new(100_000, 8192);
        assert_eq!(
            engine.check_thresholds(&budget, 500),
            CompactionAction::None
        );
    }

    /// A realistic system prompt (~8K tokens) should NOT trigger compaction
    /// on a 32K context window after just 1-2 turns of conversation.
    ///
    /// Before the fix, `active_tokens()` counted the system prompt, so with
    /// tau_soft=0.5 on a 32K window (soft limit ~15K), an 8K system prompt
    /// left only ~7K for conversation before compaction fired at msg_count=2.
    #[test]
    fn test_system_prompt_excluded_from_compaction_threshold() {
        let mut engine = LcmEngine::new(LcmConfig {
            tau_soft: 0.5,
            tau_hard: 0.85,
            deterministic_target: 512,
        });

        // Simulate a realistic ~8K-token system prompt (32K chars ≈ 8K tokens)
        let big_system = "x".repeat(32_000);
        ingest(&mut engine, 1, "system", &big_system);

        // Add 2 short conversation turns (~200 tokens total)
        ingest(&mut engine, 2, "user", "Hello, how are you?");
        ingest(
            &mut engine,
            3,
            "assistant",
            "I'm fine, thanks for asking! How can I help you today?",
        );
        ingest(&mut engine, 4, "user", "What is the weather like?");
        ingest(
            &mut engine,
            5,
            "assistant",
            "I don't have access to weather data, but I can help with other things.",
        );

        // 32K context, 2K reserve → 30K available, tau_soft=0.5 → soft=15K
        let budget = TokenBudget::new(32_768, 2048);

        // With the fix: conversation tokens (~200) are well under 15K → None
        // Before the fix: 8K system + 200 conv = 8.2K, but the real bug is
        // that it triggers even earlier with slightly more conversation.
        assert_eq!(
            engine.check_thresholds(&budget, 500),
            CompactionAction::None,
            "System prompt should not count toward compaction threshold"
        );
    }

    /// Verify that conversation tokens (excluding system prompt) DO trigger
    /// compaction when they genuinely exceed the soft threshold.
    #[test]
    fn test_conversation_tokens_trigger_compaction_when_over_threshold() {
        let mut engine = LcmEngine::new(LcmConfig {
            tau_soft: 0.5,
            tau_hard: 0.85,
            deterministic_target: 512,
        });

        // Small system prompt
        ingest(&mut engine, 1, "system", "You are helpful.");

        // Fill conversation with enough tokens to exceed tau_soft on a small window
        // 4K context, 512 reserve → 3.5K available, tau_soft=0.5 → soft=1.75K tokens
        // Add ~2K tokens of conversation (8K chars / 4)
        let big_msg = "y".repeat(8_000);
        ingest(&mut engine, 2, "user", &big_msg);

        let budget = TokenBudget::new(4096, 512);
        assert_eq!(
            engine.check_thresholds(&budget, 0),
            CompactionAction::Async,
            "Conversation tokens over soft threshold should trigger async compaction"
        );
    }

    #[test]
    fn test_thresholds_can_use_retained_session_available_budget() {
        let mut engine = LcmEngine::new(LcmConfig {
            tau_soft: 0.5,
            tau_hard: 0.85,
            deterministic_target: 512,
        });
        ingest(&mut engine, 1, "system", "You are helpful.");
        ingest(
            &mut engine,
            2,
            "user",
            &"retained session pressure ".repeat(400),
        );

        let large_model_budget = TokenBudget::new(32_768, 2048);
        assert_eq!(
            engine.check_thresholds(&large_model_budget, 0),
            CompactionAction::None,
            "model context alone should not compact yet"
        );

        let retained_available = engine.conversation_tokens();
        assert_eq!(
            engine.check_thresholds_with_available(retained_available),
            CompactionAction::Blocking,
            "retained cap pressure should force blocking compaction first"
        );
    }

    /// When a block has more messages than MAX_COMPACTION_BLOCK_MESSAGES,
    /// compact() should skip LLM summarization and use deterministic
    /// truncation directly. This prevents a 0.8B model from making 75+
    /// sequential LLM calls to summarize a massive block.
    #[tokio::test]
    async fn test_huge_block_skips_llm_uses_deterministic() {
        let mut engine = LcmEngine::new(LcmConfig {
            tau_soft: 0.1,
            tau_hard: 0.3,
            deterministic_target: 64,
        });

        ingest(&mut engine, 1, "system", "System");
        // Add 200 messages — way more than any sane compaction block
        for i in 0..100 {
            ingest(
                &mut engine,
                2 + 2 * i,
                "user",
                &format!("Question {} about topic {}", i, i * 7),
            );
            ingest(
                &mut engine,
                3 + 2 * i,
                "assistant",
                &format!("Answer {} with details about {}", i, i * 3),
            );
        }

        // Use a mock that PANICS if called — proving LLM was never invoked.
        let compactor = ContextCompactor::new(
            Arc::new(PanickingMock) as Arc<dyn LLMProvider>,
            "mock".to_string(),
            4096,
        );
        let budget = TokenBudget::new(32_768, 2048);

        // compact() should succeed via deterministic truncation (level 3),
        // never calling the LLM.
        let result = engine
            .compact(
                Some(&compactor),
                &budget,
                0,
                CompactionFailureMode::Deterministic,
            )
            .await;
        assert!(
            result.is_some(),
            "Should produce a summary via deterministic truncation"
        );
        if let Some(Turn::Summary { level, .. }) = &result {
            assert_eq!(
                *level, 3,
                "Should use level 3 (deterministic), not LLM levels 1-2"
            );
        }
    }

    /// `conversation_tokens()` should return 0 when only a system prompt exists.
    #[test]
    fn test_conversation_tokens_zero_for_system_only() {
        let mut engine = LcmEngine::new(LcmConfig::default());
        let big_system = "z".repeat(40_000); // ~10K tokens
        ingest(&mut engine, 1, "system", &big_system);

        assert_eq!(engine.conversation_tokens(), 0);
    }

    #[test]
    fn test_deterministic_truncate_basic() {
        let messages = vec![
            json!({"role": "user", "content": "Please read the file and analyze it."}),
            json!({"role": "tool", "name": "read_file", "content": "x".repeat(5000)}),
            json!({"role": "assistant", "content": "I found several issues in the code. Let me explain."}),
        ];

        let result = deterministic_truncate(&messages, 100);
        assert!(result.contains("User:"));
        assert!(result.contains("[Tool read_file:"));
        assert!(result.contains("Assistant:"));
        assert!(TokenBudget::estimate_str_tokens(&result) <= 100);
    }

    #[test]
    fn test_first_sentence() {
        assert_eq!(first_sentence("Hello world. More text."), "Hello world.");
        assert_eq!(first_sentence("No period here"), "No period here");
        assert_eq!(first_sentence("Question? Yes."), "Question?");
        assert_eq!(first_sentence(""), "");
    }

    // -----------------------------------------------------------------------
    // Manifest sources lenient deserialization (Bug #3)
    // -----------------------------------------------------------------------

    /// The live 2026-07-27 12:34:03 failure was
    /// `invalid type: string "msg 55531", expected usize at line 17 column 23`.
    /// The transcript labels render as `[message_id: 55531]`, and the model
    /// emitted `"msg 55531"` in the sources array. The whole manifest was
    /// discarded. Sources must accept integers, numeric strings, and
    /// `"msg <id>"`-style strings, extracting the integer from each value.
    #[test]
    fn manifest_sources_accepts_string_with_msg_prefix() {
        let json = r#"{"text":"x","sources":["msg 55531", 123, "456"]}"#;
        let item: ManifestItem = serde_json::from_str(json).unwrap();
        assert_eq!(item.text, "x");
        assert_eq!(item.sources, vec![55531, 123, 456]);
    }

    /// Non-numeric strings are skipped, not fatal: a single junk value must
    /// not discard the whole manifest (the live bug collapsed 100% of the
    /// manifest state because of one malformed source id).
    #[test]
    fn manifest_sources_skips_non_numeric_strings() {
        let json = r#"{"text":"x","sources":["hello", 789, "msg not-a-number"]}"#;
        let item: ManifestItem = serde_json::from_str(json).unwrap();
        assert_eq!(item.text, "x");
        assert_eq!(item.sources, vec![789]);
    }

    #[test]
    fn test_find_oldest_raw_block_token_based_protects_recent() {
        let engine = &mut LcmEngine::new(LcmConfig::default());
        ingest(engine, 1, "system", "System");
        // 10 messages, each ~100 tokens (400 chars). Total raw ≈ 2000 tokens.
        let body = "x".repeat(400);
        for i in 0..10 {
            ingest(engine, 2 + 2 * i, "user", &format!("Msg {i}: {body}"));
            ingest(
                engine,
                3 + 2 * i,
                "assistant",
                &format!("Reply {i}: {body}"),
            );
        }
        // Protect ~400 tokens (≈ the 4 most recent messages); compact the older run.
        let (start, end) = engine.find_oldest_raw_block_with_tokens(400).unwrap();
        assert_eq!(start, 1, "block starts after the system message");
        assert!(end <= engine.active.len() - 4, "recent tail is protected");
        assert!(end > start + 1, "a non-trivial block is compacted");
    }

    #[test]
    fn test_find_oldest_raw_block_returns_first_block() {
        let engine = &mut LcmEngine::new(LcmConfig::default());
        // Realistic-sized messages: token-based protect (default ~1k tokens) only
        // yields a compact block once the conversation exceeds the protect budget,
        // so use ~60-token messages (not 3-token ones).
        let body = "the quick brown fox jumps over the lazy dog while the model prefills tokens "
            .repeat(5);

        ingest(engine, 1, "system", "System");
        for i in 0..10 {
            ingest(engine, 2 + 2 * i, "user", &format!("User {}: {}", i, body));
            ingest(
                engine,
                3 + 2 * i,
                "assistant",
                &format!("Assistant {}: {}", i, body),
            );
        }

        let block =
            engine.find_oldest_raw_block_impl(DEFAULT_PROTECT_TOKENS, BlockSelection::AppendOnly);
        assert!(block.is_some());

        let (start, end) = block.unwrap();
        assert!(start >= 1, "Block should start after system message");
        assert!(
            end <= engine.active_len() - 4,
            "Block should leave the recent messages protected"
        );
    }

    #[test]
    fn test_find_oldest_raw_block_none_when_all_fit_in_protect_budget() {
        let engine = &mut LcmEngine::new(LcmConfig::default());
        ingest(engine, 1, "system", "System");
        for i in 0..5 {
            ingest(engine, 2 + i, "user", &format!("Msg {i}")); // tiny (~3 tokens)
        }
        // Protect budget larger than all raws → nothing to compact.
        assert!(engine.find_oldest_raw_block_with_tokens(1024).is_none());
    }

    #[test]
    fn test_protect_prefers_reasoning_over_tool_results() {
        // Tool results are weighted 2× in the protect budget (they're
        // retrievable via recall_tool_result), so a tool result that would
        // otherwise fit gets compacted in favour of keeping reasoning raw.
        let engine = &mut LcmEngine::new(LcmConfig::default());
        let body = "the quick brown fox jumps over the lazy dog while the model prefills ";
        ingest(engine, 1, "system", "System");
        ingest(engine, 2, "assistant", &body.repeat(4)); // older reasoning (~50 tok)
        ingest(engine, 3, "tool", &body.repeat(4)); // ~50 tok, weighted ~100 → overflows
        ingest(engine, 4, "assistant", "ok"); // tiny, newest
                                              // protect=50: newest assistant fits (~5); tool (weighted ~100) overflows → compacted.
        let (start, end) = engine.find_oldest_raw_block_with_tokens(50).unwrap();
        assert_eq!(start, 1);
        // tool at active index 2 is in the compact block (end > 2) while the
        // newest assistant at index 3 is protected (end <= 3) → end == 3.
        assert_eq!(end, 3, "tool result compacted, newest reasoning protected");
    }

    #[test]
    fn test_find_oldest_raw_block_huge_old_message_is_compacted() {
        // The point of token-based protect: one giant OLD tool result must not
        // be protected as "a single message". The recent small replies are
        // protected by the token budget; the giant is the compact block.
        let engine = &mut LcmEngine::new(LcmConfig::default());
        ingest(engine, 1, "system", "System");
        ingest(engine, 2, "user", &"x".repeat(8000)); // ~2k-token giant, oldest
        for i in 0..4 {
            ingest(engine, 3 + i, "assistant", &format!("reply {i}")); // ~3 tokens each
        }
        let (start, end) = engine.find_oldest_raw_block_with_tokens(100).unwrap();
        assert_eq!(start, 1, "the giant at index 1 starts the compact block");
        assert_eq!(
            end, 2,
            "only the giant is compacted; the 4 recent replies are protected"
        );
    }

    // -----------------------------------------------------------------------
    // E2E: full compact→expand cycle with mock LLM (Level 1 succeeds)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_e2e_compact_level1_then_expand() {
        // Tiny context window so we can trigger compaction with few messages.
        let mut engine = LcmEngine::new(LcmConfig {
            tau_soft: 0.3,
            tau_hard: 0.6,
            deterministic_target: 64,
        });

        // System prompt.
        ingest(&mut engine, 1, "system", "You are a helpful assistant.");

        // 10 turns of verbose conversation to fill the context.
        for i in 0..10 {
            ingest(
                &mut engine,
                2 + 2 * i,
                "user",
                &format!(
                    "Tell me about Rust ownership, borrowing, and lifetimes in detail. Turn {}. \
                     I need a comprehensive explanation with examples and edge cases.",
                    i
                ),
            );
            ingest(
                &mut engine,
                3 + 2 * i,
                "assistant",
                &format!(
                    "Rust ownership is a memory safety feature. Each value has exactly one owner. \
                     When the owner goes out of scope, the value is dropped. Borrowing allows \
                     temporary references. Lifetimes annotate how long references are valid. \
                     This is turn {} of our conversation about memory management in Rust.",
                    i
                ),
            );
        }

        let pre_compact_active = engine.active_len();
        let pre_compact_store = engine.store_len();
        assert_eq!(pre_compact_active, 21); // 1 system + 20 messages
        assert_eq!(pre_compact_store, 21);

        // Budget: small enough that 21 messages exceed τ_soft.
        let budget = TokenBudget::new(4096, 2048);
        let action = engine.check_thresholds(&budget, 100);
        assert!(
            action == CompactionAction::Async || action == CompactionAction::Blocking,
            "With 21 messages in 2048-token budget, should trigger compaction, got {:?}",
            action
        );

        // Compact with mock LLM that returns a short summary (Level 1 succeeds).
        let compactor = ContextCompactor::new(
            Arc::new(SummarizerMock) as Arc<dyn LLMProvider>,
            "mock".to_string(),
            4096,
        );
        let result = engine
            .compact(
                Some(&compactor),
                &budget,
                100,
                CompactionFailureMode::Deterministic,
            )
            .await;
        assert!(result.is_some(), "Compaction should produce a summary");

        let summary_turn = result.unwrap();
        let summary_text = match &summary_turn {
            Turn::Summary { text, .. } => text.clone(),
            _ => panic!("Expected Turn::Summary"),
        };
        assert!(
            summary_text.contains("Rust ownership"),
            "Summary should contain key content"
        );

        // Active context should be shorter.
        assert!(
            engine.active_len() < pre_compact_active,
            "Active context should shrink: was {}, now {}",
            pre_compact_active,
            engine.active_len()
        );

        // Immutable store is unchanged.
        assert_eq!(
            engine.store_len(),
            pre_compact_store,
            "Store must never lose messages"
        );

        // DAG should have a summary node.
        assert_eq!(engine.dag.len(), 1, "Should have exactly 1 summary node");
        let node = engine.dag.get(0).unwrap();
        assert_eq!(node.level, 1, "Should be Level 1 (LLM succeeded)");
        assert!(!node.source_ids.is_empty());

        // Active context should contain a Summary entry.
        let has_summary = engine
            .active
            .iter()
            .any(|e| matches!(e, ContextEntry::Summary { .. }));
        assert!(has_summary, "Active context must contain a Summary entry");

        // Expand: retrieve originals via the IDs stored in the summary node.
        let expanded = engine.expand(&node.source_ids);
        assert_eq!(
            expanded.len(),
            node.source_ids.len(),
            "All source messages must be retrievable"
        );
        for (id, msg) in &expanded {
            let content = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
            assert!(
                !content.is_empty(),
                "Expanded message {} should have content",
                id
            );
        }

        // format_expanded should produce readable output.
        let formatted = engine.format_expanded(&node.source_ids);
        assert!(formatted.contains("[msg "));
        assert!(formatted.contains("user:") || formatted.contains("assistant:"));
    }

    // -----------------------------------------------------------------------
    // E2E: compact with failing LLM → no deterministic fallback
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_e2e_compact_llm_error_leaves_context_uncompacted() {
        let mut engine = LcmEngine::new(LcmConfig {
            tau_soft: 0.3,
            tau_hard: 0.6,
            deterministic_target: 64,
        });

        ingest(&mut engine, 1, "system", "System prompt.");
        for i in 0..10 {
            ingest(
                &mut engine,
                2 + 2 * i,
                "user",
                &format!("Question {i} about lifetimes and borrowing rules in Rust, with examples of move semantics and shared references across scopes. "),
            );
            ingest(
                &mut engine,
                3 + 2 * i,
                "assistant",
                &format!("Answer {i} explains ownership semantics in detail, covering how values are dropped, how borrows prevent aliasing, and where lifetimes annotations are required. "),
            );
        }

        let active_before = engine.active_context();
        let budget = TokenBudget::new(4096, 2048);

        // Compact with a failing LLM: the summarization call errors, so
        // compaction must NOT fall back to deterministic truncation — it
        // leaves the context untouched and retryable next turn.
        let compactor = ContextCompactor::new(
            Arc::new(FailingMock) as Arc<dyn LLMProvider>,
            "mock".to_string(),
            4096,
        );
        let result = engine
            .compact(
                Some(&compactor),
                &budget,
                100,
                CompactionFailureMode::Deterministic,
            )
            .await;

        assert!(
            result.is_none(),
            "an LLM error must never fall back to silent deterministic truncation"
        );
        assert_eq!(
            engine.active_context(),
            active_before,
            "context must be preserved byte for byte on failure"
        );
        assert_eq!(engine.dag.len(), 0, "no summary node created on failure");

        // Lossless: originals still directly retrievable from the store.
        for id in engine.store_ids() {
            assert_eq!(engine.expand(&[id]).len(), 1);
        }
    }

    // -----------------------------------------------------------------------
    // E2E: compact with a babble-scoring summary → rejected, never persisted
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_e2e_compact_babble_summary_leaves_context_uncompacted() {
        let mut engine = LcmEngine::new(LcmConfig {
            tau_soft: 0.3,
            tau_hard: 0.6,
            deterministic_target: 64,
        });

        ingest(&mut engine, 1, "system", "System prompt.");
        for i in 0..10 {
            ingest(
                &mut engine,
                2 + 2 * i,
                "user",
                &format!("Question {i} about lifetimes and borrowing rules in Rust, with examples of move semantics and shared references across scopes. "),
            );
            ingest(
                &mut engine,
                3 + 2 * i,
                "assistant",
                &format!("Answer {i} explains ownership semantics in detail, covering how values are dropped, how borrows prevent aliasing, and where lifetimes annotations are required. "),
            );
        }

        let active_before = engine.active_context();
        let budget = TokenBudget::new(4096, 2048);

        // The LLM returns a degenerate, filler-heavy, repetitive wall of
        // text — real-world motivation is a small local model producing
        // babble that used to get persisted verbatim as memory. The
        // anti-drift babble gate must reject it before it is ever
        // persisted, exactly like an LLM error: uncompacted this round, no
        // retry ladder to a different level.
        let compactor = ContextCompactor::new(
            Arc::new(BabbleMock) as Arc<dyn LLMProvider>,
            "mock".to_string(),
            4096,
        );
        let result = engine
            .compact(
                Some(&compactor),
                &budget,
                100,
                CompactionFailureMode::Deterministic,
            )
            .await;

        assert!(
            result.is_none(),
            "a babble/degenerate summary must never be persisted"
        );
        assert_eq!(
            engine.active_context(),
            active_before,
            "context must be preserved byte for byte when the summary is rejected"
        );
        assert_eq!(
            engine.dag.len(),
            0,
            "no summary node created for a rejected babble summary"
        );
    }

    #[tokio::test]
    async fn soft_compaction_without_model_preserves_active_context() {
        let mut engine = LcmEngine::new(LcmConfig {
            tau_soft: 0.3,
            tau_hard: 0.6,
            deterministic_target: 64,
        });
        ingest(&mut engine, 1, "system", "System prompt.");
        for i in 0..10 {
            let detail =
                "ownership borrowing lifetimes move semantics shared references ".repeat(3);
            ingest(
                &mut engine,
                2 + 2 * i,
                "user",
                &format!("Question {i}: {detail}"),
            );
            ingest(
                &mut engine,
                3 + 2 * i,
                "assistant",
                &format!("Answer {i}: {detail}"),
            );
        }
        let active_before = engine.active_context();
        engine.request_async_compaction();

        let result = engine
            .compact(
                None,
                &TokenBudget::new(4096, 2048),
                100,
                CompactionFailureMode::PreserveContext,
            )
            .await;

        assert!(result.is_none());
        assert_eq!(engine.active_context(), active_before);
        assert_eq!(engine.dag.len(), 0);
        assert_ne!(
            engine.check_thresholds(&TokenBudget::new(4096, 2048), 100),
            CompactionAction::None,
            "a failed soft pass must clear the pending bit so it can retry"
        );
    }

    #[tokio::test]
    async fn soft_compaction_huge_block_installs_deterministic_truncation() {
        // Regression pin for the async-band limbo: PreserveContext used to
        // REFUSE blocks >80 messages outright, so a store sitting in the
        // soft band re-fired Async every turn and installed nothing while
        // the prompt grew to the server cap (session 20260828_142425: zero
        // summaries in 83 messages). Soft mode now takes the same
        // deterministic truncation the blocking path uses.
        let mut engine = LcmEngine::new(LcmConfig {
            tau_soft: 0.1,
            tau_hard: 0.3,
            deterministic_target: 64,
        });
        ingest(&mut engine, 1, "system", "System prompt.");
        for i in 0..100 {
            let detail = "ownership borrowing lifetimes and stable identifiers ".repeat(3);
            ingest(
                &mut engine,
                2 + 2 * i,
                "user",
                &format!("Question {i}: {detail}"),
            );
            ingest(
                &mut engine,
                3 + 2 * i,
                "assistant",
                &format!("Answer {i}: {detail}"),
            );
        }
        engine.request_async_compaction();

        let result = engine
            .compact(
                None,
                &TokenBudget::new(32_768, 2048),
                0,
                CompactionFailureMode::PreserveContext,
            )
            .await;

        let summary = result.expect("huge soft block must install deterministic truncation");
        let Turn::Summary { text, level, .. } = summary else {
            panic!("expected a Turn::Summary, got {summary:?}");
        };
        assert_eq!(level, 3, "huge block must take the deterministic path");
        assert!(
            !text.is_empty(),
            "deterministic truncation must produce summary text"
        );
        assert!(
            engine.active_context().len() < 201,
            "block must be replaced by the summary entry"
        );
        assert!(
            !engine.dag().is_empty(),
            "summary node must be committed to the DAG"
        );
        assert_eq!(
            engine.check_thresholds(&TokenBudget::new(32_768, 2048), 0),
            CompactionAction::None,
            "installed truncation must bring the store back under thresholds"
        );
    }

    #[tokio::test]
    async fn hard_compaction_without_model_leaves_context_uncompacted() {
        // With no compactor at all, hard pressure can no longer converge via
        // deterministic truncation — that fallback was deleted (silent
        // truncation is never acceptable). It leaves the context uncompacted
        // and retryable, same as a soft failure; a separate hard trim
        // elsewhere in the pipeline is what actually protects the request.
        let mut engine = LcmEngine::new(LcmConfig {
            tau_soft: 0.3,
            tau_hard: 0.6,
            deterministic_target: 64,
        });
        ingest(&mut engine, 1, "system", "System prompt.");
        for i in 0..10 {
            let detail =
                "ownership borrowing lifetimes move semantics shared references ".repeat(3);
            ingest(
                &mut engine,
                2 + 2 * i,
                "user",
                &format!("Question {i}: {detail}"),
            );
            ingest(
                &mut engine,
                3 + 2 * i,
                "assistant",
                &format!("Answer {i}: {detail}"),
            );
        }
        let active_before = engine.active_context();

        let result = engine
            .compact(
                None,
                &TokenBudget::new(4096, 2048),
                100,
                CompactionFailureMode::Deterministic,
            )
            .await;

        assert!(
            result.is_none(),
            "no compactor means no compaction, never silent truncation"
        );
        assert_eq!(engine.active_context(), active_before);
        assert_eq!(engine.dag.len(), 0);
    }

    // -----------------------------------------------------------------------
    // E2E: lcm_expand tool round-trip
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_e2e_lcm_expand_tool_roundtrip() {
        let engine = Arc::new(Mutex::new(LcmEngine::new(LcmConfig::default())));

        // Ingest messages.
        {
            let mut e = engine.lock().await;
            ingest(&mut e, 1, "user", "What is Rust?");
            ingest(
                &mut e,
                2,
                "assistant",
                "Rust is a systems programming language.",
            );
            ingest(&mut e, 3, "user", "Tell me about ownership.");
        }

        let tool = LcmExpandTool::new(engine.clone());

        // Valid IDs.
        let mut params = HashMap::new();
        params.insert("message_ids".to_string(), json!("1,2,3"));
        let output = crate::agent::tools::base::render_result(
            tool.execute(params, &crate::agent::tools::base::ToolContext::sandbox())
                .await,
        );
        assert!(output.contains("[msg 1] user: What is Rust?"));
        assert!(output.contains("[msg 2] assistant: Rust is a systems programming language."));
        assert!(output.contains("[msg 3] user: Tell me about ownership."));

        // Integer array (the shape small models actually emit) works too.
        let mut params = HashMap::new();
        params.insert("message_ids".to_string(), json!([1, 2, 3]));
        let output = crate::agent::tools::base::render_result(
            tool.execute(params, &crate::agent::tools::base::ToolContext::sandbox())
                .await,
        );
        assert!(output.contains("[msg 1] user: What is Rust?"));
        assert!(output.contains("[msg 3] user: Tell me about ownership."));

        // A range string expands inclusively.
        let mut params = HashMap::new();
        params.insert("message_ids".to_string(), json!("1-3"));
        let output = crate::agent::tools::base::render_result(
            tool.execute(params, &crate::agent::tools::base::ToolContext::sandbox())
                .await,
        );
        assert!(output.contains("[msg 2]"));

        // Invalid IDs.
        let mut params = HashMap::new();
        params.insert("message_ids".to_string(), json!("99,100"));
        let output = crate::agent::tools::base::render_result(
            tool.execute(params, &crate::agent::tools::base::ToolContext::sandbox())
                .await,
        );
        assert!(output.contains("No messages found"));

        // Empty input.
        let mut params = HashMap::new();
        params.insert("message_ids".to_string(), json!(""));
        let output = crate::agent::tools::base::render_result(
            tool.execute(params, &crate::agent::tools::base::ToolContext::sandbox())
                .await,
        );
        assert!(output.contains("Error: no valid message IDs"));
    }

    // -----------------------------------------------------------------------
    // E2E: double compaction (compact twice, expand both)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_e2e_double_compaction_lossless() {
        let mut engine = LcmEngine::new(LcmConfig {
            tau_soft: 0.2,
            tau_hard: 0.5,
            deterministic_target: 64,
        });

        ingest(&mut engine, 1, "system", "System.");
        // 20 turns — enough for two compaction rounds.
        for i in 0..20 {
            ingest(
                &mut engine,
                2 + 2 * i,
                "user",
                &format!(
                    "Detailed question {} about async Rust with tokio examples.",
                    i
                ),
            );
            ingest(
                &mut engine,
                3 + 2 * i,
                "assistant",
                &format!(
                    "Detailed answer {} covering spawn, select, and channels.",
                    i
                ),
            );
        }

        let total_messages = engine.store_len();
        let budget = TokenBudget::new(4096, 2048);
        let compactor = ContextCompactor::new(
            Arc::new(SummarizerMock) as Arc<dyn LLMProvider>,
            "mock".to_string(),
            4096,
        );

        // First compaction.
        let r1 = engine
            .compact(
                Some(&compactor),
                &budget,
                100,
                CompactionFailureMode::Deterministic,
            )
            .await;
        assert!(r1.is_some(), "First compaction should succeed");
        let active_after_first = engine.active_len();

        // Second compaction (if there's still a raw block).
        let r2 = engine
            .compact(
                Some(&compactor),
                &budget,
                100,
                CompactionFailureMode::Deterministic,
            )
            .await;
        if r2.is_some() {
            assert!(
                engine.active_len() <= active_after_first,
                "Second compaction should not increase context"
            );
        }

        // Lossless invariant: every original message still in store.
        assert_eq!(engine.store_len(), total_messages);

        // Every summary node's source IDs resolve to real messages.
        for i in 0..engine.dag.len() {
            let node = engine.dag.get(i).unwrap();
            let expanded = engine.expand(&node.source_ids);
            assert_eq!(
                expanded.len(),
                node.source_ids.len(),
                "Summary node {} has dangling source IDs",
                i
            );
        }
    }

    #[tokio::test]
    async fn compaction_keeps_existing_wire_prefix_until_summary_merge_is_needed() {
        let mut engine = LcmEngine::new(LcmConfig {
            tau_soft: 0.3,
            tau_hard: 0.6,
            deterministic_target: 64,
        });
        let budget = TokenBudget::new(4096, 1024);
        let compactor = ContextCompactor::new(
            Arc::new(SummarizerMock) as Arc<dyn LLMProvider>,
            "mock".to_string(),
            16_384,
        );
        let body =
            "the quick brown fox preserves the local model prompt prefix across compactions ";

        ingest(&mut engine, 1, "system", "System");
        for i in 0..12 {
            ingest(&mut engine, 2 + 2 * i, "user", &format!("{i}: {body}"));
            ingest(
                &mut engine,
                3 + 2 * i,
                "assistant",
                &format!("reply {i}: {body}"),
            );
        }
        assert!(engine
            .compact(
                Some(&compactor),
                &budget,
                0,
                CompactionFailureMode::Deterministic,
            )
            .await
            .is_some());
        let last_summary = engine
            .active_entries()
            .iter()
            .rposition(|entry| matches!(entry, ContextEntry::Summary { .. }))
            .unwrap();
        let stable_after_first = serialize_wire(&engine.active_entries()[..=last_summary]);

        for i in 0..12 {
            ingest(
                &mut engine,
                100 + 2 * i,
                "user",
                &format!("more {i}: {body}"),
            );
            ingest(
                &mut engine,
                101 + 2 * i,
                "assistant",
                &format!("more reply {i}: {body}"),
            );
        }
        assert!(engine
            .compact(
                Some(&compactor),
                &budget,
                0,
                CompactionFailureMode::Deterministic,
            )
            .await
            .is_some());
        let after_second = serialize_wire(engine.active_entries());

        assert!(
            after_second.starts_with(&stable_after_first),
            "the first compacted prompt, excluding its protected raw tail, \
             must remain a byte-prefix after the next append-only compaction"
        );
        assert_eq!(
            engine
                .active_entries()
                .iter()
                .filter(|entry| matches!(entry, ContextEntry::Summary { .. }))
                .count(),
            2,
            "low summary mass should append instead of merge"
        );
    }

    #[test]
    fn test_format_id_ranges_compact_and_roundtrip() {
        assert_eq!(format_id_ranges(&[5, 6, 7, 8]), "5-8");
        assert_eq!(format_id_ranges(&[50756]), "50756");
        assert_eq!(
            format_id_ranges(&[50756, 50757, 50758, 50760, 50762, 50763]),
            "50756-50758,50760,50762-50763"
        );
        // Unsorted + duplicate input still yields canonical ranges.
        assert_eq!(format_id_ranges(&[8, 5, 6, 6, 7]), "5-8");

        // The header range must round-trip through the lcm_expand parser —
        // this is the contract that lets a small model copy it verbatim.
        let ids: Vec<usize> = vec![50756, 50757, 50758, 50760, 50762, 50763];
        assert_eq!(parse_id_runs(&format_id_ranges(&ids)), ids);
    }

    #[tokio::test]
    async fn test_merge_recompaction_does_not_keep_id_spam() {
        // Regression: re-compacting a block containing a prior summary used to
        // feed the summary's WIRE message (header with every source ID) into
        // deterministic truncation, whose "first sentence" was the ID list —
        // the merged summary became ID spam with all content lost.
        let mut engine = LcmEngine::new(LcmConfig {
            tau_soft: 0.3,
            tau_hard: 0.6,
            deterministic_target: 64,
        });
        let budget = TokenBudget::new(4096, 1024);
        let body = "the quick brown fox jumps over the lazy dog while the model prefills tokens ";

        ingest(&mut engine, 1, "system", "System");
        for i in 0..12 {
            ingest(&mut engine, 2 + 2 * i, "user", &format!("{i}: {body}"));
            ingest(
                &mut engine,
                3 + 2 * i,
                "assistant",
                &format!("re {i}: {body}"),
            );
        }
        // Round 1: LLM-backed compaction (Level 1). Uses a content-echoing
        // mock (not the fixed-text SummarizerMock) because round 2 re-feeds
        // evolving input through the same compactor, and a canned reply
        // fails the compaction fidelity gate's topic-anchor check on the
        // second round.
        let compactor = ContextCompactor::new(
            Arc::new(EchoSummarizerMock) as Arc<dyn LLMProvider>,
            "mock".to_string(),
            4096,
        );
        let r1 = engine
            .compact(
                Some(&compactor),
                &budget,
                0,
                CompactionFailureMode::Deterministic,
            )
            .await;
        assert!(r1.is_some());
        engine.dag.nodes[0].tokens =
            (budget.available_budget(0) as f64 * SUMMARY_MERGE_BUDGET_FRACTION) as usize + 1;
        assert_eq!(
            engine.block_selection(budget.available_budget(0)),
            BlockSelection::MergeSummaries
        );

        // Round 2: merge the prior summary with new raws.
        for i in 0..12 {
            ingest(
                &mut engine,
                100 + 2 * i,
                "user",
                &format!("more {i}: {body}"),
            );
            ingest(
                &mut engine,
                101 + 2 * i,
                "assistant",
                &format!("more re {i}: {body}"),
            );
        }
        let r2 = engine
            .compact(
                Some(&compactor),
                &budget,
                0,
                CompactionFailureMode::Deterministic,
            )
            .await;
        assert!(r2.is_some());

        let node = engine.dag().newest().unwrap();
        assert!(
            !node.text.contains("Summary of messages"),
            "merged summary text must not contain a prior wire header: {}",
            node.text
        );
        assert!(
            !node.text.contains("lcm_expand"),
            "merged summary text must not contain expand instructions: {}",
            node.text
        );

        // The wire message header carries compact ranges that round-trip.
        let wire = engine
            .active_entries()
            .iter()
            .find_map(|e| match e {
                ContextEntry::Summary { message, .. } => message
                    .get("content")
                    .and_then(|c| c.as_str())
                    .map(|s| s.to_string()),
                _ => None,
            })
            .unwrap();
        let ranges = format_id_ranges(&node.source_ids);
        assert!(
            wire.contains(&format!("{{\"message_ids\": \"{ranges}\"}}")),
            "wire header must show a copyable lcm_expand call: {wire}"
        );
        let mut expected = node.source_ids.clone();
        expected.sort_unstable();
        assert_eq!(parse_id_runs(&ranges), expected);
    }

    #[tokio::test]
    async fn test_compaction_merges_summaries_when_mass_exceeds_threshold() {
        let mut engine = LcmEngine::new(LcmConfig {
            tau_soft: 0.3,
            tau_hard: 0.6,
            deterministic_target: 64,
        });
        let compactor = ContextCompactor::new(
            Arc::new(SummarizerMock) as Arc<dyn LLMProvider>,
            "mock".to_string(),
            16_384,
        );
        let budget = TokenBudget::new(4096, 1024);
        let body = "the quick brown fox jumps over the lazy dog while the model prefills tokens ";
        ingest(&mut engine, 1, "system", "System");
        ingest(&mut engine, 2, "user", "Already summarized source");
        for i in 0..16 {
            ingest(&mut engine, 2 + 2 * i, "user", &format!("{i}: {body}"));
            ingest(
                &mut engine,
                3 + 2 * i,
                "assistant",
                &format!("reply {i}: {body}"),
            );
        }
        let prior_node_id = engine
            .dag
            .create_node(
                vec![2],
                vec![],
                "Earlier compacted state.".to_string(),
                SummaryManifest::default(),
                1,
            )
            .id;
        engine.active[1] = ContextEntry::Summary {
            node_id: prior_node_id,
            message: json!({
                "role": "user",
                "_lcm_summary": true,
                "content": "Earlier compacted state."
            }),
        };
        let available = budget.available_budget(0);
        engine.dag.nodes[prior_node_id].tokens = (available as f64 * 0.25) as usize + 1;

        assert_eq!(
            engine.block_selection(available),
            BlockSelection::MergeSummaries
        );
        let result = engine
            .compact(
                Some(&compactor),
                &budget,
                0,
                CompactionFailureMode::Deterministic,
            )
            .await;
        assert!(result.is_some(), "threshold-triggered merge must compact");
        assert_eq!(
            engine
                .active_entries()
                .iter()
                .filter(|entry| matches!(entry, ContextEntry::Summary { .. }))
                .count(),
            1,
            "merge mode must retire the prior summary"
        );
    }

    // -----------------------------------------------------------------------
    // Benchmark: LCM compaction quality across matched local models.
    //
    // Runs the same 10-turn conversation through each model's compaction
    // and reports: escalation level, compression ratio, latency, summary.
    //
    // Requires an OpenAI-compatible endpoint at NANOBOT_LCM_BENCH_BASE with
    // every comma-separated NANOBOT_LCM_BENCH_MODELS id available. This lets
    // the release check compare candidate foreground models with identical
    // input, budget, prompt, and measurement code.
    //
    // Run: cargo test test_bench_lcm_compaction_models -- --ignored --nocapture
    // -----------------------------------------------------------------------

    async fn load_real_benchmark_sessions(
        db_path: &std::path::Path,
        session_ids: &[String],
    ) -> Vec<(String, Vec<Value>)> {
        let db = crate::session::SessionDb::new(db_path);
        let mut sessions = Vec::with_capacity(session_ids.len());
        for session_id in session_ids {
            let messages = db.get_all_messages(session_id).await;
            if !messages.is_empty() {
                sessions.push((session_id.clone(), messages));
            }
        }
        sessions
    }

    #[tokio::test]
    async fn real_session_benchmark_loader_uses_persisted_messages() {
        let temp = tempfile::tempdir().unwrap();
        let db_path = temp.path().join("sessions.db");
        let db = crate::session::SessionDb::new(&db_path);
        let session = db.create_session("benchmark-fixture").await;
        db.add_messages(
            &session.id,
            &[
                json!({"role": "system", "content": "hidden bootstrap"}),
                json!({
                    "role": "user",
                    "content": "synthetic reminder",
                    "_synthetic": true
                }),
                json!({
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "write_file",
                            "arguments": "{\"path\":\"index.html\"}"
                        }
                    }]
                }),
                json!({
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "name": "write_file",
                    "content": "wrote complete asteroids game"
                }),
                json!({"role": "user", "content": "real Asteroids acceptance"}),
            ],
        )
        .await;

        let loaded =
            load_real_benchmark_sessions(&db_path, std::slice::from_ref(&session.id)).await;

        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].0, session.id);
        assert_eq!(loaded[0].1.len(), 5);
        assert_eq!(
            loaded[0].1[4].get("content").and_then(Value::as_str),
            Some("real Asteroids acceptance")
        );
        assert!(loaded[0]
            .1
            .iter()
            .all(|message| message.to_string().contains("Rust ownership") == false));
    }

    #[tokio::test]
    #[ignore = "requires a real sessions DB and OpenAI-compatible endpoint"]
    async fn test_bench_lcm_real_sessions() {
        use crate::providers::openai_compat::OpenAICompatProvider;
        use std::time::Instant;

        let db_path = std::env::var("NANOBOT_LCM_BENCH_SESSIONS_DB")
            .expect("NANOBOT_LCM_BENCH_SESSIONS_DB is required");
        let session_ids = std::env::var("NANOBOT_LCM_BENCH_SESSION_IDS")
            .expect("NANOBOT_LCM_BENCH_SESSION_IDS is required")
            .split(',')
            .map(str::trim)
            .filter(|id| !id.is_empty())
            .map(str::to_string)
            .collect::<Vec<_>>();
        assert!(
            !session_ids.is_empty(),
            "NANOBOT_LCM_BENCH_SESSION_IDS must contain at least one id"
        );
        let api_base = std::env::var("NANOBOT_LCM_BENCH_BASE")
            .unwrap_or_else(|_| "http://127.0.0.1:9001/v1".to_string());
        let model = std::env::var("NANOBOT_LCM_BENCH_MODELS")
            .unwrap_or_else(|_| "Qwen3.5-2B-MLX-8bit".to_string())
            .split(',')
            .next()
            .unwrap()
            .trim()
            .to_string();
        let max_context = std::env::var("NANOBOT_LCM_BENCH_MAX_CONTEXT")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(262_144);
        let sessions =
            load_real_benchmark_sessions(std::path::Path::new(&db_path), &session_ids).await;
        assert_eq!(
            sessions.len(),
            session_ids.len(),
            "every requested session must exist and contain messages"
        );

        let provider: Arc<dyn crate::providers::base::LLMProvider> = Arc::new(
            OpenAICompatProvider::new("local", Some(&api_base), Some(&model)),
        );
        let compactor = ContextCompactor::new(provider, model.clone(), max_context);

        for (session_id, messages) in sessions {
            let transcript = crate::agent::compaction::build_transcript(&messages);
            let transcript_tokens = TokenBudget::estimate_str_tokens(&transcript);
            let required_context =
                compactor.required_context_for_lcm(&messages, "preserve_details");
            eprintln!("\n{}", "=".repeat(80));
            eprintln!("SESSION: {session_id}");
            eprintln!("MODEL: {model}");
            eprintln!(
                "CONTEXT: transcript={transcript_tokens} required={required_context} max={max_context}"
            );
            eprintln!("--- SEMANTIC TRANSCRIPT ---\n{transcript}\n--- END TRANSCRIPT ---");

            let started = Instant::now();
            let result = compactor
                .summarize_for_lcm(&messages, "preserve_details")
                .await;
            let elapsed = started.elapsed().as_millis();
            match result {
                Ok(summary) => {
                    let output_tokens = TokenBudget::estimate_str_tokens(&summary);
                    eprintln!(
                        "--- RAW SUMMARY ---\n{summary}\n--- END SUMMARY ---\n\
                         finish_reason=stop output_tokens={output_tokens} latency_ms={elapsed}"
                    );
                }
                Err(error) => {
                    eprintln!(
                        "--- SUMMARY ERROR ---\n{error:#}\n--- END ERROR ---\nlatency_ms={elapsed}"
                    );
                }
            }
        }
    }

    #[tokio::test]
    #[ignore = "requires LM Studio with multiple models loaded"]
    async fn test_bench_lcm_compaction_models() {
        use crate::providers::openai_compat::OpenAICompatProvider;
        use std::time::Instant;

        let api_base = std::env::var("NANOBOT_LCM_BENCH_BASE")
            .unwrap_or_else(|_| "http://127.0.0.1:8000/v1".to_string());

        let models: Vec<String> = std::env::var("NANOBOT_LCM_BENCH_MODELS")
            .unwrap_or_else(|_| {
                "qwen3-0.6b,qwen3-1.7b,gemma-3n-e4b-it,nvidia-nemotron-nano-12b-v2-vl".to_string()
            })
            .split(',')
            .map(str::trim)
            .filter(|model| !model.is_empty())
            .map(str::to_string)
            .collect();
        assert!(
            !models.is_empty(),
            "NANOBOT_LCM_BENCH_MODELS must contain at least one model id"
        );

        // Build a realistic 10-turn conversation (fixed, deterministic input).
        let mut conversation: Vec<Value> = Vec::new();
        conversation.push(
            json!({"role": "system", "content": "You are a helpful Rust programming assistant."}),
        );

        let turns = [
            ("user", "Explain Rust ownership rules in detail. Each value in Rust has exactly one owner at a time. When the owner goes out of scope, the value is dropped. Ownership can be transferred via moves. For example, let s1 = String::from(\"hello\"); let s2 = s1; after this, s1 is invalid."),
            ("assistant", "Rust's ownership model is the foundation of its memory safety guarantees. Here are the three core rules: 1) Each value has exactly one owner. 2) When the owner goes out of scope, the value is automatically dropped (freed). 3) Ownership can be transferred through moves. When you write `let s2 = s1`, the String data moves from s1 to s2, and s1 becomes invalid. This prevents double-free bugs. For types that implement Copy (like integers), assignment creates a copy instead of a move."),
            ("user", "Now explain borrowing. What's the difference between &T and &mut T? How do the borrowing rules prevent data races at compile time? Give examples of when you'd use each."),
            ("assistant", "Borrowing lets you reference data without taking ownership. There are two kinds: shared references (&T) and mutable references (&mut T). The rules are: 1) You can have either one &mut T OR any number of &T at the same time, never both. 2) References must always be valid (no dangling). Shared borrows (&T) allow read-only access and can coexist. Mutable borrows (&mut T) give exclusive read-write access. Example: fn calculate_length(s: &String) -> usize { s.len() } borrows without ownership transfer."),
            ("user", "Describe lifetime annotations. Why does Rust need them? When does the compiler require explicit annotations vs when can it infer them? Give an example with a struct holding a reference."),
            ("assistant", "Lifetimes are Rust's way of tracking how long references are valid. Most of the time, the compiler infers them automatically (lifetime elision). You need explicit annotations when the compiler can't determine the relationship between input and output lifetimes. The three elision rules are: 1) Each reference parameter gets its own lifetime. 2) If there's exactly one input lifetime, it's assigned to all outputs. 3) If &self is a parameter, its lifetime is assigned to outputs. Example: struct Excerpt<'a> { part: &'a str } — this struct can't outlive the string it references."),
            ("user", "Explain async/await in Rust. How do Futures work? What is the role of the executor, waker, and poll method? How does the compiler transform async functions into state machines?"),
            ("assistant", "Async/await in Rust enables non-blocking concurrency. An async function returns a Future — a state machine generated by the compiler. The Future trait has one method: poll(cx: &mut Context) -> Poll<Output>. Executors (like tokio) drive futures by calling poll(). When a future can't make progress, it returns Poll::Pending and registers a Waker. When the underlying I/O completes, the Waker notifies the executor to poll again. The compiler transforms each .await point into a state in the state machine, saving local variables across suspension points. This is zero-cost: no heap allocation for the state machine itself."),
            ("user", "What are smart pointers in Rust? Explain Box<T>, Rc<T>, Arc<T>, Cow<T>, and when to use each. How do they differ in ownership semantics and thread safety?"),
            ("assistant", "Smart pointers in Rust manage memory with additional metadata and capabilities. Box<T> provides heap allocation with single ownership — use for recursive types or large data. Rc<T> enables shared ownership via reference counting (single-threaded only). Arc<T> is the atomic version of Rc for thread-safe shared ownership. Cow<T> (Clone-on-Write) starts as a borrow but clones to owned data when mutation is needed — great for functions that sometimes need to modify input. Box is Send+Sync, Rc is neither, Arc is both. Use Box for simple heap allocation, Rc for shared graphs, Arc for concurrent access, Cow for optional mutation."),
        ];

        for (role, content) in &turns {
            conversation.push(json!({"role": role, "content": content}));
        }

        let input_tokens = TokenBudget::estimate_tokens(&conversation[1..]); // skip system

        eprintln!("\n{}", "=".repeat(70));
        eprintln!("LCM COMPACTION BENCHMARK");
        eprintln!("{}", "=".repeat(70));
        eprintln!("Input: {} messages, {} tokens", turns.len(), input_tokens);
        eprintln!("API: {}", api_base);
        eprintln!("{:-<70}", "");
        eprintln!(
            "{:<35} {:>5} {:>6} {:>7} {:>8}  {}",
            "Model", "Level", "In", "Out", "Ratio", "Latency"
        );
        eprintln!("{:-<70}", "");

        struct BenchResult {
            model: String,
            level: u8,
            input_tokens: usize,
            output_tokens: usize,
            compression_ratio: f64,
            latency_ms: u128,
            summary_preview: String,
            success: bool,
            error: Option<String>,
        }

        let mut results: Vec<BenchResult> = Vec::new();
        let budget = TokenBudget::new(8192, 4096);

        for model_name in &models {
            let provider: Arc<dyn crate::providers::base::LLMProvider> = Arc::new(
                OpenAICompatProvider::new("local", Some(&api_base), Some(model_name)),
            );

            // Warm up: verify this model responds.
            let warmup = provider
                .chat(
                    &[json!({"role": "user", "content": "Reply: ok"})],
                    None,
                    Some(model_name),
                    16,
                    0.0,
                    None,
                    None,
                )
                .await;

            if let Err(e) = warmup {
                let msg = format!("warmup failed: {}", e);
                eprintln!("{:<35} SKIP  ({})", model_name, msg);
                results.push(BenchResult {
                    model: model_name.to_string(),
                    level: 0,
                    input_tokens,
                    output_tokens: 0,
                    compression_ratio: 0.0,
                    latency_ms: 0,
                    summary_preview: String::new(),
                    success: false,
                    error: Some(msg),
                });
                continue;
            }

            // Build a fresh LCM engine with the conversation.
            let mut engine = LcmEngine::new(LcmConfig {
                tau_soft: 0.3,
                tau_hard: 0.6,
                deterministic_target: 128,
            });

            for (i, m) in conversation.iter().enumerate() {
                let mut tagged = m.clone();
                tagged["_db_id"] = json!(i + 1);
                let _ = engine.ingest(tagged);
            }

            let compactor = ContextCompactor::new(provider.clone(), model_name.to_string(), 4096);

            // Run compaction, measure time.
            let start = Instant::now();
            let result = engine
                .compact(
                    Some(&compactor),
                    &budget,
                    100,
                    CompactionFailureMode::Deterministic,
                )
                .await;
            let elapsed = start.elapsed().as_millis();

            match result {
                Some(summary_turn) => {
                    let summary_text = match &summary_turn {
                        Turn::Summary { text, .. } => text.clone(),
                        _ => String::new(),
                    };
                    let out_tokens = TokenBudget::estimate_str_tokens(&summary_text);
                    let ratio = if input_tokens > 0 {
                        out_tokens as f64 / input_tokens as f64
                    } else {
                        1.0
                    };

                    let node = engine.dag.get(engine.dag.len() - 1).unwrap();
                    let level = node.level;

                    // Preview: first 60 chars of summary.
                    let preview: String = summary_text
                        .chars()
                        .take(60)
                        .collect::<String>()
                        .replace('\n', " ");

                    eprintln!(
                        "{:<35} L{:<4} {:>6} {:>7} {:>7.1}%  {}ms",
                        model_name,
                        level,
                        input_tokens,
                        out_tokens,
                        (1.0 - ratio) * 100.0,
                        elapsed
                    );

                    results.push(BenchResult {
                        model: model_name.to_string(),
                        level,
                        input_tokens,
                        output_tokens: out_tokens,
                        compression_ratio: ratio,
                        latency_ms: elapsed,
                        summary_preview: preview,
                        success: true,
                        error: None,
                    });
                }
                None => {
                    eprintln!("{:<35} FAIL  compaction returned None", model_name);
                    results.push(BenchResult {
                        model: model_name.to_string(),
                        level: 0,
                        input_tokens,
                        output_tokens: 0,
                        compression_ratio: 0.0,
                        latency_ms: elapsed,
                        summary_preview: String::new(),
                        success: false,
                        error: Some("compaction returned None".to_string()),
                    });
                }
            }
        }

        // Print summary table.
        eprintln!("\n{:=<70}", "");
        eprintln!("RESULTS SUMMARY");
        eprintln!("{:=<70}", "");
        eprintln!(
            "{:<35} {:>5} {:>6} {:>7} {:>8} {:>8}",
            "Model", "Level", "In", "Out", "Compr%", "ms"
        );
        eprintln!("{:-<70}", "");

        for r in &results {
            if r.success {
                eprintln!(
                    "{:<35} L{:<4} {:>6} {:>7} {:>7.1}% {:>7}ms",
                    r.model,
                    r.level,
                    r.input_tokens,
                    r.output_tokens,
                    (1.0 - r.compression_ratio) * 100.0,
                    r.latency_ms
                );
            } else {
                eprintln!(
                    "{:<35} {:>5} {:>27}",
                    r.model,
                    "FAIL",
                    r.error.as_deref().unwrap_or("unknown")
                );
            }
        }

        eprintln!("\nSummary Previews:");
        eprintln!("{:-<70}", "");
        for r in &results {
            if r.success && !r.summary_preview.is_empty() {
                eprintln!("  {}: {}", r.model, r.summary_preview);
            }
        }

        // Lossless invariant check on all successful runs.
        let successful = results.iter().filter(|r| r.success).count();
        assert!(
            successful >= 1,
            "At least 1 model should produce a successful compaction"
        );

        eprintln!(
            "\n{}/{} models completed successfully.",
            successful,
            models.len()
        );
    }

    // -----------------------------------------------------------------------
    // Non-contiguous summary coverage: rebuild must not orphan messages whose
    // ids sit between summarized ids (e.g. users when only assistants were
    // summarized).
    // -----------------------------------------------------------------------

    #[test]
    fn test_rebuild_non_contiguous_source_ids() {
        // 8 messages, db ids 1..=8; summary covers only assistants 2, 4, 6.
        let raw_messages: Vec<Value> = (1..=8)
            .map(|i| {
                if i % 2 == 1 {
                    msg(i, "user", &format!("user{i}"))
                } else {
                    msg(i, "assistant", &format!("asst{i}"))
                }
            })
            .collect();
        let nodes = vec![(
            0usize,
            vec![2usize, 4, 6],
            vec![],
            "Summary of assistant messages.".to_string(),
            10usize,
            1u8,
            SummaryManifest::default(),
            "db_id".to_string(),
        )];

        let engine = LcmEngine::rebuild_from_db_nodes(&raw_messages, &nodes, LcmConfig::default());

        let raw_ids: Vec<usize> = engine
            .active
            .iter()
            .filter_map(|e| {
                if let ContextEntry::Raw { msg_id, .. } = e {
                    Some(*msg_id)
                } else {
                    None
                }
            })
            .collect();

        // Unsummarized ids (1, 3, 5, 7, 8) must all be active — not orphaned.
        assert_eq!(
            raw_ids,
            vec![1, 3, 5, 7, 8],
            "unsummarized raws must stay active"
        );

        let summary_count = engine
            .active
            .iter()
            .filter(|e| matches!(e, ContextEntry::Summary { .. }))
            .count();
        assert_eq!(
            summary_count, 1,
            "Expected exactly 1 Summary entry in active"
        );
    }

    // -----------------------------------------------------------------------
    // Auto-expand: relevant summary gets expanded
    // -----------------------------------------------------------------------

    #[test]
    fn test_auto_expand_relevant_summary() {
        let mut engine = LcmEngine::new(LcmConfig {
            tau_soft: 0.3,
            tau_hard: 0.85,
            deterministic_target: 64,
        });

        // Ingest messages about Rust ownership.
        ingest(&mut engine, 1, "system", "You are helpful.");
        ingest(
            &mut engine,
            2,
            "user",
            "Explain Rust ownership and borrowing rules.",
        );
        ingest(
            &mut engine,
            3,
            "assistant",
            "Rust ownership means each value has one owner. Borrowing allows references.",
        );
        ingest(&mut engine, 4, "user", "How do lifetimes work in Rust?");
        ingest(
            &mut engine,
            5,
            "assistant",
            "Lifetimes track how long references are valid.",
        );

        // Manually create a summary covering messages 2-5.
        let node = engine.dag.create_node(
            vec![2, 3, 4, 5],
            vec![],
            "Discussion about Rust ownership, borrowing, and lifetimes.".to_string(),
            SummaryManifest::default(),
            1,
        );
        let node_id = node.id;
        let summary_msg = json!({
            "role": "user",
            "content": "[Summary of messages 2-5 (IDs: 2,3,4,5). Use lcm_expand to retrieve originals.]\n\nDiscussion about Rust ownership, borrowing, and lifetimes."
        });

        // Replace raw entries 2-5 with the summary in active context.
        engine.active = vec![
            engine.active[0].clone(), // system
            ContextEntry::Summary {
                node_id,
                message: summary_msg,
            },
        ];

        // Now add a new user message about ownership (should trigger expansion).
        ingest(
            &mut engine,
            6,
            "user",
            "Tell me more about Rust ownership rules and borrow checker.",
        );

        let budget = TokenBudget::new(100_000, 8192);
        let active_tokens = engine.active_tokens();
        let appended = plan_and_commit_auto_expansion(&mut engine, &budget, 100, active_tokens);
        assert_eq!(
            appended.len(),
            1,
            "Auto-expand should append one expansion message for the relevant query"
        );

        // The summary STAYS in place (cache-safe): expansion is appended, not spliced.
        let has_summary = engine
            .active
            .iter()
            .any(|e| matches!(e, ContextEntry::Summary { .. }));
        assert!(
            has_summary,
            "Summary must remain — expansion is append-only"
        );

        // The appended message carries the original content, tagged synthetic.
        let content = appended[0]["content"].as_str().unwrap_or("");
        assert!(content.contains("Auto-expanded"), "should be labelled");
        assert!(
            content.contains("each value has one owner"),
            "should contain the original message text, got: {content}"
        );
        assert_eq!(appended[0]["_synthetic"], json!(true));

        // Idempotent: a second pass does not re-append the same node.
        let active_tokens = engine.active_tokens();
        assert!(
            plan_and_commit_auto_expansion(&mut engine, &budget, 100, active_tokens).is_empty(),
            "already-expanded summary must not be appended twice"
        );
    }

    #[test]
    fn auto_expansion_planning_is_non_mutating_until_explicit_commit() {
        let mut engine = LcmEngine::new(LcmConfig {
            tau_soft: 0.3,
            tau_hard: 0.85,
            deterministic_target: 64,
        });
        ingest(&mut engine, 1, "system", "You are helpful.");
        ingest(
            &mut engine,
            2,
            "user",
            "Explain Rust ownership and borrowing rules.",
        );
        ingest(
            &mut engine,
            3,
            "assistant",
            "Rust ownership means each value has one owner.",
        );
        let node_id = engine
            .dag
            .create_node(
                vec![2, 3],
                vec![],
                "Discussion about Rust ownership and borrowing.".to_string(),
                SummaryManifest::default(),
                1,
            )
            .id;
        engine.active = vec![
            engine.active[0].clone(),
            ContextEntry::Summary {
                node_id,
                message: json!({
                    "role": "user",
                    "content": "[Summary of messages 2-3.] Rust ownership and borrowing."
                }),
            },
        ];
        ingest(
            &mut engine,
            4,
            "user",
            "Tell me more about Rust ownership and borrowing.",
        );
        let budget = TokenBudget::new(100_000, 8192);

        let first = engine.plan_auto_expansion(&budget, 0, engine.active_tokens());
        assert_eq!(first.len(), 1);
        assert_eq!(first[0].node_id, node_id);
        assert_eq!(first[0].source_ids, vec![2, 3]);
        assert!(first[0].estimated_tokens > 0);
        assert!(first[0]
            .flattened_fallback
            .get("content")
            .and_then(Value::as_str)
            .is_some_and(|content| content.contains("each value has one owner")));

        assert_eq!(
            engine
                .plan_auto_expansion(&budget, 0, engine.active_tokens())
                .len(),
            1,
            "selection alone must not consume expansion eligibility"
        );
        assert!(engine.commit_auto_expansion(node_id));
        assert!(engine
            .plan_auto_expansion(&budget, 0, engine.active_tokens())
            .is_empty());
        assert!(!engine.commit_auto_expansion(node_id));
    }

    // -----------------------------------------------------------------------
    // Auto-expand: budget-aware — doesn't expand when near τ_hard
    // -----------------------------------------------------------------------

    #[test]
    fn test_auto_expand_budget_aware() {
        let mut engine = LcmEngine::new(LcmConfig {
            tau_soft: 0.3,
            tau_hard: 0.85,
            deterministic_target: 64,
        });

        // Fill with enough messages to be near the hard limit.
        ingest(&mut engine, 1, "system", "S");
        for i in 0..20 {
            ingest(
                &mut engine,
                2 + 2 * i,
                "user",
                &format!("Long question {} about Rust ownership and memory management details with many words.", i),
            );
            ingest(
                &mut engine,
                3 + 2 * i,
                "assistant",
                &format!(
                    "Long answer {} about ownership, borrowing, lifetimes, and the borrow checker.",
                    i
                ),
            );
        }

        // Create a summary covering messages 2-11.
        let source_ids: Vec<usize> = (2..=11).collect();
        let node = engine.dag.create_node(
            source_ids.clone(),
            vec![],
            "Summary of first 10 messages.".to_string(),
            SummaryManifest::default(),
            1,
        );
        let node_id = node.id;
        let summary_msg = json!({
            "role": "user",
            "content": "[Summary of messages 2-11 (IDs: 2..11).]\n\nSummary of first 10 messages."
        });

        // Replace messages 2-11 in active with summary.
        let mut new_active = vec![engine.active[0].clone()];
        new_active.push(ContextEntry::Summary {
            node_id,
            message: summary_msg,
        });
        for entry in engine.active.iter().skip(11) {
            new_active.push(entry.clone());
        }
        engine.active = new_active;

        // Use a tiny budget where there's no headroom.
        let budget = TokenBudget::new(256, 128); // Very small
        let active_tokens = engine.active_tokens();
        assert!(
            plan_and_commit_auto_expansion(&mut engine, &budget, 50, active_tokens).is_empty(),
            "Should NOT expand when budget headroom is insufficient"
        );
    }

    // -----------------------------------------------------------------------
    // Auto-expand: irrelevant query doesn't trigger expansion
    // -----------------------------------------------------------------------

    #[test]
    fn test_auto_expand_irrelevant_query() {
        let mut engine = LcmEngine::new(LcmConfig {
            tau_soft: 0.3,
            tau_hard: 0.85,
            deterministic_target: 64,
        });

        ingest(&mut engine, 1, "system", "S");
        ingest(
            &mut engine,
            2,
            "user",
            "Explain Rust ownership and borrowing.",
        );
        ingest(
            &mut engine,
            3,
            "assistant",
            "Ownership means each value has one owner.",
        );

        let node = engine.dag.create_node(
            vec![2, 3],
            vec![],
            "Discussion about Rust ownership.".to_string(),
            SummaryManifest::default(),
            1,
        );
        let node_id = node.id;
        let summary_msg = json!({
            "role": "user",
            "content": "[Summary of messages 1-2.]\n\nDiscussion about Rust ownership."
        });

        engine.active = vec![
            engine.active[0].clone(),
            ContextEntry::Summary {
                node_id,
                message: summary_msg,
            },
        ];

        // Ask about something completely unrelated.
        ingest(
            &mut engine,
            4,
            "user",
            "What is the weather forecast for Tokyo tomorrow?",
        );

        let budget = TokenBudget::new(100_000, 8192);
        let active_tokens = engine.active_tokens();
        assert!(
            plan_and_commit_auto_expansion(&mut engine, &budget, 100, active_tokens).is_empty(),
            "Should NOT expand for an irrelevant query"
        );
    }

    // -----------------------------------------------------------------------
    // Auto-expand: fresh-summary cooldown (Bug #2 — feedback loop)
    // -----------------------------------------------------------------------

    /// A summary created during turn N's compaction MUST NOT be auto-expanded
    /// at turn N+1. The live failure (2026-07-27 12:13:06, session
    /// 20260727_094539_eeab48) saw +12463 tokens of originals reinjected 24
    /// seconds after a successful 12463→1398 compaction, blowing the wire
    /// back to pre-compaction size and forcing 5 more failed compactions in
    /// the next 10 minutes. `FRESH_SUMMARY_COOLDOWN_TURNS` makes this
    /// structurally impossible: the summary is ineligible until it has aged
    /// past the cooldown.
    #[tokio::test]
    async fn auto_expand_skips_freshly_created_summary() {
        let mut engine = LcmEngine::new(LcmConfig {
            tau_soft: 0.3,
            tau_hard: 0.6,
            deterministic_target: 64,
        });
        // Large budget so the ONLY barrier to expansion is the cooldown
        // (with wire_tokens=0, headroom = hard_limit ≈ 59k tokens).
        let budget = TokenBudget::new(100_000, 1_024);
        let compactor = ContextCompactor::new(
            Arc::new(EchoSummarizerMock) as Arc<dyn LLMProvider>,
            "mock".to_string(),
            100_000,
        );
        // Distinctive topic shared between compacted sources and the next
        // user message so keyword overlap is high. `body` is sized so the
        // compactable block clears MIN_COMPACTION_TOKENS (200).
        let topic = "serramanna weather forecast desert bakery recipes";
        let body = format!("{} ", topic).repeat(20);
        ingest(&mut engine, 1, "system", "System");
        for i in 0..12 {
            ingest(&mut engine, 2 + 2 * i, "user", &format!("{i}: {body}"));
            ingest(
                &mut engine,
                3 + 2 * i,
                "assistant",
                &format!("reply {i}: {body}"),
            );
        }

        // Turn 5: compact — new node stamps created_at_turn=5.
        engine.set_current_turn(5);
        let compacted = engine
            .compact(
                Some(&compactor),
                &budget,
                0,
                CompactionFailureMode::Deterministic,
            )
            .await;
        assert!(compacted.is_some(), "compaction must succeed first");
        let node_id = engine.dag().newest().unwrap().id;
        assert_eq!(
            engine.dag().get(node_id).unwrap().created_at_turn,
            5,
            "compact must stamp the new node's creation turn"
        );

        // Turn 6 (age=1): ingest a highly-relevant user message and give
        // auto_expand huge wire headroom so ONLY the cooldown can block it.
        engine.set_current_turn(6);
        ingest(&mut engine, 100, "user", topic);
        let appended = plan_and_commit_auto_expansion(&mut engine, &budget, 0, 0);
        assert!(
            appended.is_empty(),
            "freshly-created summary (age=1, cooldown=1) must NOT be \
             auto-expanded; this is the exact live failure pattern that \
             undid the 12463→1398 compaction 24 seconds later"
        );

        // Sanity: at turn 8 (age=3, past cooldown), the same user message
        // CAN trigger expansion — the cooldown is not a permanent block.
        engine.set_current_turn(8);
        let appended_later = plan_and_commit_auto_expansion(&mut engine, &budget, 0, 0);
        assert!(
            !appended_later.is_empty(),
            "at age=3 (past FRESH_SUMMARY_COOLDOWN_TURNS=1) the summary must \
             be eligible again; cooldown must not be permanent"
        );
    }

    /// Reinjection consumes WIRE budget, not the engine's internal active.
    /// The engine's active under-counts because reinjected originals live in
    /// the wire (not `self.active`). Counting the wire is what prevents
    /// auto_expand from pushing the prompt past τ_hard and blowing the Higgs
    /// retained-session cap.
    #[tokio::test]
    async fn auto_expand_budget_uses_wire_tokens_not_active() {
        let mut engine = LcmEngine::new(LcmConfig {
            tau_soft: 0.3,
            tau_hard: 0.6,
            deterministic_target: 64,
        });
        let budget = TokenBudget::new(100_000, 1_024);
        let compactor = ContextCompactor::new(
            Arc::new(EchoSummarizerMock) as Arc<dyn LLMProvider>,
            "mock".to_string(),
            100_000,
        );
        let body = "serramanna weather forecast desert bakery recipes ".repeat(20);
        ingest(&mut engine, 1, "system", "System");
        for i in 0..12 {
            ingest(&mut engine, 2 + 2 * i, "user", &format!("{i}: {body}"));
            ingest(
                &mut engine,
                3 + 2 * i,
                "assistant",
                &format!("reply {i}: {body}"),
            );
        }

        // Turn 1: compact — node stamps created_at_turn=1.
        engine.set_current_turn(1);
        engine
            .compact(
                Some(&compactor),
                &budget,
                0,
                CompactionFailureMode::Deterministic,
            )
            .await;

        // Turn 100: way past the cooldown, so wire_tokens is the only barrier.
        engine.set_current_turn(100);
        ingest(&mut engine, 200, "user", &body); // highly relevant
        let available = budget.available_budget(0);
        let hard_limit = (available as f64 * engine.tau_hard()) as usize;
        let active_tokens = engine.active_tokens();

        // Wire is already at hard_limit. Internal active is much smaller
        // (just the summary + recent raws), but the wire bound must win.
        let appended = plan_and_commit_auto_expansion(&mut engine, &budget, 0, hard_limit);
        assert!(
            appended.is_empty(),
            "wire_tokens={} (= hard_limit) must block reinjection even when \
             engine.active_tokens()={} has nominal headroom; counting the wire \
             is what prevents the Higgs retained-session cap blow-up",
            hard_limit,
            active_tokens
        );
    }

    /// End-to-end: the exact live failure pattern. Ingest ~10k+ tokens of
    /// conversation, compact successfully, then immediately call auto_expand
    /// on the very next turn with a same-topic user message. The reinjection
    /// of freshly-compacted originals must be impossible.
    #[tokio::test]
    async fn auto_expand_cannot_reinject_more_than_just_compacted() {
        let mut engine = LcmEngine::new(LcmConfig {
            tau_soft: 0.3,
            tau_hard: 0.6,
            deterministic_target: 64,
        });
        let budget = TokenBudget::new(100_000, 1_024);
        let compactor = ContextCompactor::new(
            Arc::new(EchoSummarizerMock) as Arc<dyn LLMProvider>,
            "mock".to_string(),
            100_000,
        );
        // Build a body of substantial conversation around one topic — large
        // enough that compaction actually relieves pressure.
        let topic = "higgs retained session compaction feedback loop ";
        let body = topic.repeat(20);
        ingest(&mut engine, 1, "system", "System");
        for i in 0..12 {
            ingest(&mut engine, 2 + 2 * i, "user", &format!("{i}: {body}"));
            ingest(
                &mut engine,
                3 + 2 * i,
                "assistant",
                &format!("reply {i}: {body}"),
            );
        }

        let pre_compact_tokens = engine.active_tokens();
        engine.set_current_turn(1);
        let compacted = engine
            .compact(
                Some(&compactor),
                &budget,
                0,
                CompactionFailureMode::Deterministic,
            )
            .await;
        assert!(compacted.is_some(), "compaction must succeed first");
        let post_compact_tokens = engine.active_tokens();
        assert!(
            post_compact_tokens < pre_compact_tokens,
            "compaction must reduce active size: pre={} post={}",
            pre_compact_tokens,
            post_compact_tokens
        );

        // Turn 2 (age=1): same-topic user msg, wire at post-compact active
        // size. This is the 12:13:06 pattern. Reinjection must be blocked.
        engine.set_current_turn(2);
        ingest(&mut engine, 200, "user", topic);
        let appended = plan_and_commit_auto_expansion(&mut engine, &budget, 0, post_compact_tokens);
        assert!(
            appended.is_empty(),
            "fresh summary (age=1) must NOT be re-expanded; would reinject \
             ~{} freshly-compacted tokens back into the wire and undo the \
             compaction we just measured ({} → {})",
            pre_compact_tokens - post_compact_tokens,
            pre_compact_tokens,
            post_compact_tokens
        );
    }

    // -----------------------------------------------------------------------
    // rebuild_from_db_nodes: round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn test_rebuild_from_db_nodes() {
        // Simulate raw messages with stable db ids 1..=6.
        let raw_messages = vec![
            msg(1, "system", "System prompt."),
            msg(2, "user", "Hello"),
            msg(3, "assistant", "Hi there!"),
            msg(4, "user", "How are you?"),
            msg(5, "assistant", "I'm good, thanks!"),
            msg(6, "user", "Tell me a joke."),
        ];

        // Simulate a DB node covering db ids 2-5.
        let db_nodes = vec![(
            0usize,                           // node_id
            vec![2usize, 3, 4, 5],            // source_ids
            vec![],                           // child_ids
            "Greeting exchange.".to_string(), // text
            10usize,                          // tokens
            1u8,                              // level
            SummaryManifest::default(),       // manifest
            "db_id".to_string(),              // id_kind
        )];

        let config = LcmConfig {
            tau_soft: 0.5,
            tau_hard: 0.85,
            deterministic_target: 512,
        };

        let engine = LcmEngine::rebuild_from_db_nodes(&raw_messages, &db_nodes, config);

        // Should have all 6 messages in store.
        assert_eq!(engine.store_len(), 6);

        // DAG should have 1 node.
        assert_eq!(engine.dag().len(), 1);
        let node = engine.dag().get(0).unwrap();
        assert_eq!(node.source_ids, vec![2, 3, 4, 5]);
        assert_eq!(node.level, 1);

        // Active context: 1 summary + 2 raw (system + msg 6).
        let summary_count = engine
            .active_entries()
            .iter()
            .filter(|e| matches!(e, ContextEntry::Summary { .. }))
            .count();
        assert_eq!(summary_count, 1);

        let raw_count = engine
            .active_entries()
            .iter()
            .filter(|e| matches!(e, ContextEntry::Raw { .. }))
            .count();
        assert_eq!(raw_count, 2, "system + msg 6 should be raw");

        // Expand still works.
        let expanded = engine.expand(&[2, 3, 4, 5]);
        assert_eq!(expanded.len(), 4);
    }

    #[test]
    fn rebuild_honors_latest_clear_boundary() {
        let raw_messages = vec![
            msg(1, "user", "old question"),
            msg(2, "assistant", "old answer"),
            msg(3, "clear", ""),
            msg(4, "user", "new question"),
            msg(5, "assistant", "new answer"),
            msg(6, "user", "new tail"),
        ];
        let db_nodes = vec![
            (
                40,
                vec![1, 2],
                vec![],
                "old summary".to_string(),
                4,
                1,
                SummaryManifest::default(),
                "db_id".to_string(),
            ),
            (
                1,
                vec![4, 5],
                vec![],
                "new summary".to_string(),
                4,
                1,
                SummaryManifest::default(),
                "db_id".to_string(),
            ),
        ];

        let engine =
            LcmEngine::rebuild_from_db_nodes(&raw_messages, &db_nodes, LcmConfig::default());
        let active = engine.active_context();
        let wire = serde_json::to_string(&active).unwrap();

        assert_eq!(
            engine.store_len(),
            3,
            "only post-clear raws enter the store"
        );
        assert_eq!(engine.dag().len(), 1, "pre-clear summaries stay retired");
        assert_eq!(
            engine.dag.next_id, 41,
            "post-clear compaction must not reuse a persisted pre-clear node ID"
        );
        assert!(wire.contains("new summary"));
        assert!(wire.contains("new tail"));
        assert!(!wire.contains("old question"));
        assert!(!wire.contains("old summary"));
        assert!(!active
            .iter()
            .any(|message| { message.get("role").and_then(Value::as_str) == Some("clear") }));
        assert!(engine.expand(&[1, 2]).is_empty());
        assert_eq!(engine.expand(&[4, 5]).len(), 2);
    }

    /// THE id-drift repro. Live sessions ingest a FILTERED WINDOW of history
    /// (get_history caps counts/tokens and drops synthetics), while restart
    /// rebuilds from the FULL history. Under the old positional scheme a
    /// summary created against the window (positions 0..) misresolved after
    /// restart — expand() silently returned the wrong originals. With
    /// db-id-keyed MessageIds the same ids resolve identically in both worlds.
    #[test]
    fn test_windowed_resume_ids_stable() {
        // Full history: db ids 1..=10.
        let full: Vec<Value> = (1..=10)
            .map(|i| {
                let role = if i % 2 == 1 { "user" } else { "assistant" };
                msg(i, role, &format!("MSG_{i}"))
            })
            .collect();

        // Live session: the engine only ever saw a window (ids 6..=10)...
        let mut live = LcmEngine::new(LcmConfig::default());
        for m in &full[5..] {
            let _ = live.ingest(m.clone());
        }
        // ...and compacted ids 6,7 into a summary persisted with those ids.
        live.dag.create_node(
            vec![6, 7],
            vec![],
            "Summary of 6-7.".to_string(),
            SummaryManifest::default(),
            1,
        );
        // Even before restart, the live engine resolves by db id, not window
        // position (positionally, "6" would have been out of range or wrong).
        let live_expanded = live.expand(&[6, 7]);
        assert_eq!(live_expanded.len(), 2);
        assert_eq!(live_expanded[0].1["content"], "MSG_6");
        assert_eq!(live_expanded[1].1["content"], "MSG_7");

        // Restart: rebuild from the FULL history + the persisted node.
        let nodes = vec![(
            0usize,
            vec![6usize, 7],
            vec![],
            "Summary of 6-7.".to_string(),
            5usize,
            1u8,
            SummaryManifest::default(),
            "db_id".to_string(),
        )];
        let engine = LcmEngine::rebuild_from_db_nodes(&full, &nodes, LcmConfig::default());

        // Lossless guarantee: expand(6,7) returns EXACTLY messages 6 and 7.
        // (Old positional code returned the messages at positions 6,7 of the
        // full history — MSG_7 and MSG_8 — silent corruption.)
        let expanded = engine.expand(&[6, 7]);
        assert_eq!(expanded.len(), 2);
        assert_eq!(expanded[0].1["content"], "MSG_6");
        assert_eq!(expanded[1].1["content"], "MSG_7");

        // The summarized ids are not active as raws; everything else is.
        let raw_ids: Vec<usize> = engine
            .active
            .iter()
            .filter_map(|e| match e {
                ContextEntry::Raw { msg_id, .. } => Some(*msg_id),
                _ => None,
            })
            .collect();
        assert_eq!(raw_ids, vec![1, 2, 3, 4, 5, 8, 9, 10]);
    }

    /// A merged summary records the nodes it subsumed as child_ids. On rebuild,
    /// those subsumed nodes must be SKIPPED — otherwise they re-activate
    /// alongside the merged node (duplicate content + renewed accumulation).
    #[test]
    fn rebuild_skips_subsumed_summary_nodes() {
        // Node 0 = an old summary (subsumed). Node 1 = the merged summary whose
        // child_ids = [0]. No raw messages needed for this DAG-only check.
        let nodes: Vec<(
            usize,
            Vec<usize>,
            Vec<usize>,
            String,
            usize,
            u8,
            SummaryManifest,
            String,
        )> = vec![
            (
                0,
                vec![1, 2, 3],
                vec![],
                "old summary".to_string(),
                10,
                1,
                SummaryManifest::default(),
                "db_id".to_string(),
            ),
            (
                1,
                vec![1, 2, 3, 4, 5],
                vec![0],
                "merged summary".to_string(),
                15,
                1,
                SummaryManifest::default(),
                "db_id".to_string(),
            ),
        ];
        let engine = LcmEngine::rebuild_from_db_nodes(&[], &nodes, LcmConfig::default());

        // Only the merged node survives (the subsumed old node is skipped).
        assert_eq!(
            engine.dag().len(),
            1,
            "subsumed node must be skipped on rebuild"
        );
        let only = engine
            .dag()
            .get(1)
            .expect("persisted root ID must survive rebuild");
        assert_eq!(only.text, "merged summary");
        assert_eq!(only.child_summaries, vec![0]);
    }

    /// Sparse persisted IDs used to be densely renumbered while their child
    /// references were left untouched. If the only active root became ID 0 and
    /// named retired child 0, `all_source_ids(0)` recursed forever. Preserve
    /// the root ID, reserve above every persisted ID, and keep expansion keyed
    /// exclusively by stable SQLite message IDs.
    #[test]
    fn restart_rebuild_preserves_sparse_subsumed_ids_without_cycles() {
        let raw_messages: Vec<Value> = (10..=14)
            .map(|id| msg(id, "user", &format!("ORIGINAL_{id}")))
            .collect();
        let nodes = vec![
            (
                0usize,
                vec![10usize, 11],
                vec![],
                "retired child".to_string(),
                4usize,
                1u8,
                SummaryManifest::default(),
                "db_id".to_string(),
            ),
            (
                17usize,
                vec![10usize, 11, 12],
                vec![0usize],
                "sparse merged root".to_string(),
                6usize,
                1u8,
                SummaryManifest::default(),
                "db_id".to_string(),
            ),
        ];

        let mut engine =
            LcmEngine::rebuild_from_db_nodes(&raw_messages, &nodes, LcmConfig::default());

        assert!(
            engine.dag().get(0).is_none(),
            "retired child stays inactive"
        );
        let root = engine
            .dag()
            .get(17)
            .expect("sparse persisted root ID must be preserved");
        assert_eq!(root.child_summaries, vec![0]);
        assert_eq!(
            engine.dag().all_source_ids(17),
            vec![10, 11, 12],
            "retired child reference must not alias the restored root"
        );

        let expanded = engine.expand(&[10, 11, 12]);
        assert_eq!(expanded.len(), 3);
        assert_eq!(expanded[0].1["content"], "ORIGINAL_10");
        assert_eq!(expanded[2].1["content"], "ORIGINAL_12");

        let next = engine.dag.create_node(
            vec![13, 14],
            vec![17],
            "new live root".to_string(),
            SummaryManifest::default(),
            1,
        );
        assert_eq!(next.id, 18, "new nodes must not reuse persisted sparse IDs");
        assert_eq!(engine.dag.newest().map(|node| node.id), Some(18));
    }

    #[test]
    fn restart_rebuild_drops_persisted_cycles() {
        let raw_messages = vec![msg(10, "user", "still available")];
        let nodes = vec![
            (
                4usize,
                vec![10usize],
                vec![9usize],
                "cycle a".to_string(),
                2usize,
                1u8,
                SummaryManifest::default(),
                "db_id".to_string(),
            ),
            (
                9usize,
                vec![10usize],
                vec![4usize],
                "cycle b".to_string(),
                2usize,
                1u8,
                SummaryManifest::default(),
                "db_id".to_string(),
            ),
        ];

        let engine = LcmEngine::rebuild_from_db_nodes(&raw_messages, &nodes, LcmConfig::default());
        assert!(engine.dag().is_empty(), "cyclic nodes cannot become roots");
        assert_eq!(engine.expand(&[10])[0].1["content"], "still available");
    }

    /// CRIT-1 seam: after a compaction the loop used to call
    /// `reset_from_messages()`, clearing the store and corrupting `lcm_expand`.
    /// The old e2e tests missed this because they called `compact()` directly.
    /// The store is now append-only; originals survive compaction + resync.
    #[tokio::test]
    async fn test_store_sacred_across_compaction_and_resync() {
        let config = LcmConfig {
            tau_soft: 0.5,
            tau_hard: 0.85,
            deterministic_target: 512,
        };
        let mut engine = LcmEngine::new(config);

        ingest(&mut engine, 1, "system", "System prompt.");
        for i in 0..8 {
            ingest(
                &mut engine,
                2 + 2 * i,
                "user",
                &format!(
                    "MARKER_USER_{i} tell me about rust ownership borrowing lifetimes \
                     in detail with worked examples and edge cases please"
                ),
            );
            ingest(
                &mut engine,
                3 + 2 * i,
                "assistant",
                &format!(
                    "MARKER_ASSISTANT_{i} rust ownership is a memory safety feature; \
                     each value has exactly one owner and is dropped at scope end"
                ),
            );
        }
        let store_before = engine.store_len();

        let budget = TokenBudget::new(4096, 2048);
        // Content-echoing mock: the per-message MARKER_USER_{i}/
        // MARKER_ASSISTANT_{i} tokens are unique-per-message topic anchors
        // that a fixed canned reply can't cover.
        let compactor = ContextCompactor::new(
            Arc::new(EchoSummarizerMock) as Arc<dyn LLMProvider>,
            "mock".to_string(),
            4096,
        );
        let summary = engine
            .compact(
                Some(&compactor),
                &budget,
                100,
                CompactionFailureMode::Deterministic,
            )
            .await
            .expect("should compact");
        let source_ids = match &summary {
            Turn::Summary { source_ids, .. } => source_ids.clone(),
            _ => panic!("expected Turn::Summary"),
        };

        // Simulate the loop's post-swap resync: ctx.messages = active_context(),
        // engine is NOT reset. The active window shrinks...
        assert!(
            engine.active_context().len() < store_before,
            "active context should shrink after compaction"
        );
        // ...but the store keeps every original.
        assert_eq!(
            engine.store_len(),
            store_before,
            "store must never be cleared"
        );
        for &id in &source_ids {
            let got = engine.expand(&[id]);
            assert_eq!(got.len(), 1, "original id {id} must still resolve");
            let content = got[0]
                .1
                .get("content")
                .and_then(|c| c.as_str())
                .unwrap_or("");
            assert!(
                content.contains("MARKER"),
                "expand({id}) returned corrupted/shifted content: {content}"
            );
        }

        // A new persisted message on the next turn keeps its own db id; the
        // store is never renumbered.
        let next_id = store_before + 1; // ids 1..=17 used above
        let new_id = engine.ingest(msg(next_id, "user", "next turn"));
        assert_eq!(new_id, Some(next_id));
        assert_eq!(engine.store_len(), store_before + 1);
        assert_eq!(engine.expand(&[next_id])[0].1["content"], "next turn");
    }

    /// CRIT-2 seam: a persisted `role:"summary"` row must never occupy a
    /// store slot on restart. Under db-id keying its rowid is simply absent
    /// from the store, and every other id resolves to exactly its own row.
    #[test]
    fn test_rebuild_skips_summary_rows_preserving_ids() {
        let raw_messages = vec![
            msg(1, "system", "System prompt."),
            msg(2, "user", "FIRST question"),
            msg(3, "assistant", "FIRST answer"),
            // Poison row: a legacy summary persisted into messages (rowid 4).
            json!({"role": "summary", "text": "a summary", "source_ids": [2, 3], "level": 1, "_db_id": 4}),
            msg(5, "user", "SECOND question"),
            msg(6, "assistant", "SECOND answer"),
        ];
        let db_nodes = vec![(
            0usize,
            vec![2usize, 3],
            vec![],
            "a summary".to_string(),
            10usize,
            1u8,
            SummaryManifest::default(),
            "db_id".to_string(),
        )];

        let engine =
            LcmEngine::rebuild_from_db_nodes(&raw_messages, &db_nodes, LcmConfig::default());

        assert_eq!(
            engine.store_len(),
            5,
            "the summary row must be skipped, not stored"
        );
        assert!(
            engine.expand(&[4]).is_empty(),
            "the summary row's own id must not resolve to anything"
        );
        let first = engine.expand(&[2, 3]);
        assert_eq!(first[0].1["content"], "FIRST question");
        assert_eq!(first[1].1["content"], "FIRST answer");
        let after = engine.expand(&[5]);
        assert_eq!(
            after[0].1["content"], "SECOND question",
            "ids must resolve to their own rows, never shifted"
        );
    }

    #[test]
    fn test_parse_message_ids_lenient() {
        // Integer array — the canonical shape.
        assert_eq!(parse_message_ids(&json!([5, 6, 7, 8])), vec![5, 6, 7, 8]);
        // Comma string, with spaces.
        assert_eq!(parse_message_ids(&json!("5, 6, 7, 8")), vec![5, 6, 7, 8]);
        // Bracketed string (model echoed the block literally).
        assert_eq!(parse_message_ids(&json!("[5,6,7,8]")), vec![5, 6, 7, 8]);
        // Inclusive range.
        assert_eq!(parse_message_ids(&json!("5-8")), vec![5, 6, 7, 8]);
        // Mixed singles and a range.
        assert_eq!(parse_message_ids(&json!("1, 5-7")), vec![1, 5, 6, 7]);
        // Numbers embedded in prose.
        assert_eq!(parse_message_ids(&json!("messages 5 and 6")), vec![5, 6]);
        // Bare number.
        assert_eq!(parse_message_ids(&json!(5)), vec![5]);
        // Array of strings.
        assert_eq!(parse_message_ids(&json!(["5", "6"])), vec![5, 6]);
        // Nothing parseable.
        assert!(parse_message_ids(&json!("none")).is_empty());
        // Runaway range is rejected, not expanded to millions.
        assert!(parse_message_ids(&json!("0-999999")).is_empty());
    }

    // -----------------------------------------------------------------------
    // Keyword extraction
    // -----------------------------------------------------------------------

    #[test]
    fn test_extract_keywords() {
        let keywords = extract_keywords("Explain Rust ownership and borrowing rules in detail.");
        assert!(keywords.contains("rust"));
        assert!(keywords.contains("ownership"));
        assert!(keywords.contains("borrowing"));
        assert!(keywords.contains("rules"));
        assert!(keywords.contains("detail"));
        // Stopwords should be excluded.
        assert!(!keywords.contains("and"));
        assert!(!keywords.contains("in"));
    }
}
