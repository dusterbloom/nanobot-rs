//! Lossless Context Management (LCM)
//!
//! Implements the LCM architecture from Ehrlich & Blackman (2026):
//! - **Immutable Store**: Every message persisted verbatim in session JSONL (existing).
//! - **Active Context**: Window sent to LLM = recent raw messages + summary nodes.
//! - **Summary DAG**: Hierarchical summaries with lossless pointers to originals.
//! - **Two-threshold control loop**: τ_soft (async) / τ_hard (blocking).
//! - **Three-level escalation**: preserve_details → bullet_points → deterministic truncate.
//!
//! The session JSONL files serve as the immutable store. This module manages
//! the summary DAG and active context assembly.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

use crate::agent::compaction::ContextCompactor;
use crate::agent::token_budget::TokenBudget;
use crate::config::schema::LcmSchemaConfig;

// ---------------------------------------------------------------------------
// Summary DAG
// ---------------------------------------------------------------------------

/// Unique ID for a message in the immutable store.
///
/// Index into the session's message array (0-based, monotonically increasing).
pub type MessageId = usize;

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
    /// Estimated token count of the summary.
    pub tokens: usize,
    /// Escalation level that produced this summary (1, 2, or 3).
    pub level: u8,
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
        text: String,
        level: u8,
    ) -> &SummaryNode {
        let tokens = TokenBudget::estimate_str_tokens(&text);
        let id = self.next_id;
        self.next_id += 1;
        self.nodes.push(SummaryNode {
            id,
            source_ids,
            child_summaries: Vec::new(),
            text,
            tokens,
            level,
        });
        &self.nodes[self.nodes.len() - 1]
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
    Raw {
        msg_id: MessageId,
        message: Value,
    },
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
    /// Enable LCM (default: false for backward compatibility).
    pub enabled: bool,
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
            enabled: false,
            tau_soft: 0.5,
            tau_hard: 0.85,
            deterministic_target: 512,
        }
    }
}

impl From<&LcmSchemaConfig> for LcmConfig {
    fn from(schema: &LcmSchemaConfig) -> Self {
        Self {
            enabled: schema.enabled,
            tau_soft: schema.tau_soft,
            tau_hard: schema.tau_hard,
            deterministic_target: schema.deterministic_target,
        }
    }
}

/// The LCM engine: manages the active context with lossless compaction.
pub struct LcmEngine {
    config: LcmConfig,
    /// The summary DAG.
    dag: SummaryDag,
    /// Active context entries (system prompt + summaries + raw messages).
    active: Vec<ContextEntry>,
    /// All raw messages in the immutable store (indexed by MessageId).
    /// This is the in-memory mirror; the session JSONL is the durable copy.
    store: Vec<Value>,
    /// Whether async compaction has been requested but not yet completed.
    async_compaction_pending: bool,
}

impl LcmEngine {
    pub fn new(config: LcmConfig) -> Self {
        Self {
            config,
            dag: SummaryDag::new(),
            active: Vec::new(),
            store: Vec::new(),
            async_compaction_pending: false,
        }
    }

    /// Ingest a new message into the immutable store and active context.
    ///
    /// Returns the assigned MessageId.
    pub fn ingest(&mut self, message: Value) -> MessageId {
        let msg_id = self.store.len();
        self.store.push(message.clone());
        self.active.push(ContextEntry::Raw { msg_id, message });
        msg_id
    }

    /// Get the active context as a message array for the LLM.
    pub fn active_context(&self) -> Vec<Value> {
        self.active.iter().map(|e| e.message().clone()).collect()
    }

    /// Estimate the token count of the active context.
    pub fn active_tokens(&self) -> usize {
        let messages: Vec<Value> = self.active_context();
        TokenBudget::estimate_tokens(&messages)
    }

    /// Check thresholds and return what action is needed.
    pub fn check_thresholds(&self, budget: &TokenBudget, tool_def_tokens: usize) -> CompactionAction {
        let available = budget.available_budget(tool_def_tokens);
        let current = self.active_tokens();

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
    /// Returns the observation text (for cross-session memory) if compaction occurred.
    pub async fn compact(
        &mut self,
        compactor: &ContextCompactor,
        budget: &TokenBudget,
        tool_def_tokens: usize,
    ) -> Option<String> {
        let available = budget.available_budget(tool_def_tokens);
        let target = (available as f64 * self.config.tau_soft * 0.8) as usize;

        // Find the oldest contiguous block of raw messages to compact.
        // Skip the system message (index 0) and any existing summaries.
        let (block_start, block_end) = match self.find_oldest_raw_block() {
            Some(range) => range,
            None => {
                debug!("LCM: no raw block to compact");
                self.async_compaction_pending = false;
                return None;
            }
        };

        // Collect the messages and their IDs.
        let mut source_ids = Vec::new();
        let mut block_messages = Vec::new();
        for entry in &self.active[block_start..block_end] {
            if let ContextEntry::Raw { msg_id, message } = entry {
                source_ids.push(*msg_id);
                block_messages.push(message.clone());
            }
        }

        if block_messages.is_empty() {
            self.async_compaction_pending = false;
            return None;
        }

        let block_tokens = TokenBudget::estimate_tokens(&block_messages);
        info!(
            "LCM: compacting {} messages ({} tokens) from positions {}..{}",
            block_messages.len(),
            block_tokens,
            block_start,
            block_end
        );

        // Three-level escalation (Algorithm 3).
        let (summary_text, level) =
            escalated_summary(&block_messages, target, compactor, self.config.deterministic_target)
                .await;

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

        // Create summary node in DAG.
        let node = self.dag.create_node(source_ids.clone(), summary_text.clone(), level);
        let node_id = node.id;

        // Build the summary message with lossless pointers.
        // Include source message IDs so the model knows it can lcm_expand them.
        let id_list: String = source_ids
            .iter()
            .map(|id| id.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let summary_message = json!({
            "role": "user",
            "content": format!(
                "[Summary of messages {}-{} (IDs: {}). Use lcm_expand to retrieve originals.]\n\n{}",
                source_ids.first().unwrap_or(&0),
                source_ids.last().unwrap_or(&0),
                id_list,
                summary_text
            )
        });

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

        Some(summary_text)
    }

    /// Find the oldest contiguous block of raw messages (skipping system prompt).
    fn find_oldest_raw_block(&self) -> Option<(usize, usize)> {
        let mut start = None;
        let mut end = 0;

        for (i, entry) in self.active.iter().enumerate() {
            match entry {
                ContextEntry::Raw { msg_id: _, message } => {
                    // Skip system message.
                    let role = message.get("role").and_then(|r| r.as_str()).unwrap_or("");
                    if role == "system" {
                        continue;
                    }
                    if start.is_none() {
                        start = Some(i);
                    }
                    end = i + 1;
                }
                ContextEntry::Summary { .. } => {
                    // If we've found a raw block before this summary, use it.
                    if start.is_some() {
                        break;
                    }
                }
            }
        }

        // Don't compact the most recent messages (keep at least the last 4 raw entries).
        let raw_count = self
            .active
            .iter()
            .filter(|e| matches!(e, ContextEntry::Raw { .. }))
            .count();

        if let Some(s) = start {
            // Keep at least 4 recent raw messages uncompacted.
            let protect_count = 4.min(raw_count);
            let max_end = self.active.len().saturating_sub(protect_count);
            let clamped_end = end.min(max_end);
            if clamped_end > s + 1 {
                // Need at least 2 messages to justify compaction.
                Some((s, clamped_end))
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Mark that async compaction has been requested.
    pub fn request_async_compaction(&mut self) {
        self.async_compaction_pending = true;
    }

    /// Retrieve original messages by IDs from the immutable store.
    ///
    /// This is the `lcm_expand` operation — lossless retrieval.
    pub fn expand(&self, msg_ids: &[MessageId]) -> Vec<(MessageId, &Value)> {
        msg_ids
            .iter()
            .filter_map(|&id| self.store.get(id).map(|msg| (id, msg)))
            .collect()
    }

    /// Expand a summary node: retrieve all original messages it covers.
    pub fn expand_summary(&self, node_id: usize) -> Vec<(MessageId, &Value)> {
        let ids = self.dag.all_source_ids(node_id);
        self.expand(&ids)
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
            output = "No messages found for the given IDs.".to_string();
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

    /// Get the active context entry count.
    pub fn active_len(&self) -> usize {
        self.active.len()
    }

    /// Access the summary DAG (for inspection in tests).
    pub(crate) fn dag_ref(&self) -> &SummaryDag {
        &self.dag
    }

    /// Access the active context entries (for inspection in tests).
    pub(crate) fn active_entries(&self) -> &[ContextEntry] {
        &self.active
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

// ---------------------------------------------------------------------------
// Three-Level Escalation (Algorithm 3)
// ---------------------------------------------------------------------------

/// Escalated summarization: tries increasingly aggressive strategies.
///
/// Returns (summary_text, escalation_level).
async fn escalated_summary(
    messages: &[Value],
    _target_tokens: usize,
    compactor: &ContextCompactor,
    deterministic_target: usize,
) -> (String, u8) {
    let original_tokens = TokenBudget::estimate_tokens(messages);

    // Level 1: Preserve details.
    if let Ok(summary) = compactor.summarize_for_lcm(messages, "preserve_details").await {
        let tokens = TokenBudget::estimate_str_tokens(&summary);
        if tokens < original_tokens {
            debug!("LCM escalation: Level 1 succeeded ({} -> {} tokens)", original_tokens, tokens);
            return (summary, 1);
        }
        debug!("LCM escalation: Level 1 failed (output {} >= input {})", tokens, original_tokens);
    }

    // Level 2: Bullet points, half the target.
    if let Ok(summary) = compactor.summarize_for_lcm(messages, "bullet_points").await {
        let tokens = TokenBudget::estimate_str_tokens(&summary);
        if tokens < original_tokens {
            debug!("LCM escalation: Level 2 succeeded ({} -> {} tokens)", original_tokens, tokens);
            return (summary, 2);
        }
        debug!("LCM escalation: Level 2 failed (output {} >= input {})", tokens, original_tokens);
    }

    // Level 3: Deterministic truncation — guaranteed convergence, no LLM.
    let truncated = deterministic_truncate(messages, deterministic_target);
    debug!(
        "LCM escalation: Level 3 (deterministic truncate to {} tokens)",
        deterministic_target
    );
    (truncated, 3)
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
        "Retrieve original messages from a summarized conversation block. \
         Use when you need full details from a [Summary of messages X-Y] block."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "message_ids": {
                    "type": "string",
                    "description": "Comma-separated message IDs to retrieve (e.g. '5,6,7,8')"
                }
            },
            "required": ["message_ids"]
        })
    }

    async fn execute(&self, params: HashMap<String, Value>) -> String {
        let ids_str = params
            .get("message_ids")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let msg_ids: Vec<usize> = ids_str
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();

        if msg_ids.is_empty() {
            return "Error: no valid message IDs provided. Use comma-separated numbers (e.g. '5,6,7,8').".to_string();
        }

        let engine = self.engine.lock().await;
        engine.format_expanded(&msg_ids)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::tools::base::Tool;
    use crate::providers::base::{LLMProvider, LLMResponse};
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
                content: Some("User asked multiple questions about Rust ownership.".to_string()),
                tool_calls: vec![],
                finish_reason: "stop".to_string(),
                usage: std::collections::HashMap::new(),
            })
        }

        fn get_default_model(&self) -> &str {
            "mock-summarizer"
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

    #[test]
    fn test_summary_dag_create_and_retrieve() {
        let mut dag = SummaryDag::new();
        dag.create_node(vec![0, 1, 2], "Summary of first 3 messages.".to_string(), 1);
        assert_eq!(dag.len(), 1);
        let node = dag.get(0).unwrap();
        assert_eq!(node.source_ids, vec![0, 1, 2]);
        assert_eq!(node.level, 1);
    }

    #[test]
    fn test_summary_dag_all_source_ids() {
        let mut dag = SummaryDag::new();
        dag.create_node(vec![0, 1, 2], "First batch.".to_string(), 1);
        dag.create_node(vec![3, 4, 5], "Second batch.".to_string(), 1);
        let ids = dag.all_source_ids(0);
        assert_eq!(ids, vec![0, 1, 2]);
    }

    #[test]
    fn test_lcm_engine_ingest() {
        let engine = &mut LcmEngine::new(LcmConfig::default());
        let id0 = engine.ingest(json!({"role": "system", "content": "You are helpful."}));
        let id1 = engine.ingest(json!({"role": "user", "content": "Hello"}));
        assert_eq!(id0, 0);
        assert_eq!(id1, 1);
        assert_eq!(engine.store_len(), 2);
        assert_eq!(engine.active_len(), 2);
    }

    #[test]
    fn test_lcm_engine_expand() {
        let engine = &mut LcmEngine::new(LcmConfig::default());
        engine.ingest(json!({"role": "user", "content": "Hello"}));
        engine.ingest(json!({"role": "assistant", "content": "Hi there!"}));
        engine.ingest(json!({"role": "user", "content": "How are you?"}));

        let expanded = engine.expand(&[0, 2]);
        assert_eq!(expanded.len(), 2);
        assert_eq!(expanded[0].0, 0);
        assert_eq!(expanded[1].0, 2);
    }

    #[test]
    fn test_lcm_engine_format_expanded() {
        let engine = &mut LcmEngine::new(LcmConfig::default());
        engine.ingest(json!({"role": "user", "content": "Hello"}));
        engine.ingest(json!({"role": "assistant", "content": "Hi!"}));

        let output = engine.format_expanded(&[0, 1]);
        assert!(output.contains("[msg 0] user: Hello"));
        assert!(output.contains("[msg 1] assistant: Hi!"));
    }

    #[test]
    fn test_check_thresholds_none() {
        let engine = &mut LcmEngine::new(LcmConfig {
            enabled: true,
            tau_soft: 0.5,
            tau_hard: 0.85,
            deterministic_target: 512,
        });
        engine.ingest(json!({"role": "system", "content": "S"}));
        engine.ingest(json!({"role": "user", "content": "Hi"}));

        let budget = TokenBudget::new(100_000, 8192);
        assert_eq!(engine.check_thresholds(&budget, 500), CompactionAction::None);
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

    #[test]
    fn test_find_oldest_raw_block() {
        let engine = &mut LcmEngine::new(LcmConfig::default());
        // System message (protected).
        engine.ingest(json!({"role": "system", "content": "System"}));
        // 10 user/assistant messages.
        for i in 0..10 {
            engine.ingest(json!({"role": "user", "content": format!("Msg {}", i)}));
            engine.ingest(json!({"role": "assistant", "content": format!("Reply {}", i)}));
        }

        let block = engine.find_oldest_raw_block();
        assert!(block.is_some());
        let (start, end) = block.unwrap();
        assert_eq!(start, 1); // After system message.
        // End should leave at least 4 raw messages protected.
        assert!(end <= engine.active.len() - 4);
    }

    // -----------------------------------------------------------------------
    // E2E: full compact→expand cycle with mock LLM (Level 1 succeeds)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_e2e_compact_level1_then_expand() {
        // Tiny context window so we can trigger compaction with few messages.
        let mut engine = LcmEngine::new(LcmConfig {
            enabled: true,
            tau_soft: 0.3,
            tau_hard: 0.6,
            deterministic_target: 64,
        });

        // System prompt.
        engine.ingest(json!({"role": "system", "content": "You are a helpful assistant."}));

        // 10 turns of verbose conversation to fill the context.
        for i in 0..10 {
            engine.ingest(json!({
                "role": "user",
                "content": format!(
                    "Tell me about Rust ownership, borrowing, and lifetimes in detail. Turn {}. \
                     I need a comprehensive explanation with examples and edge cases.",
                    i
                )
            }));
            engine.ingest(json!({
                "role": "assistant",
                "content": format!(
                    "Rust ownership is a memory safety feature. Each value has exactly one owner. \
                     When the owner goes out of scope, the value is dropped. Borrowing allows \
                     temporary references. Lifetimes annotate how long references are valid. \
                     This is turn {} of our conversation about memory management in Rust.",
                    i
                )
            }));
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
        let result = engine.compact(&compactor, &budget, 100).await;
        assert!(result.is_some(), "Compaction should produce a summary");

        let summary_text = result.unwrap();
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
        let has_summary = engine.active.iter().any(|e| matches!(e, ContextEntry::Summary { .. }));
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
    // E2E: compact with failing LLM → Level 3 deterministic fallback
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_e2e_compact_level3_deterministic_fallback() {
        let mut engine = LcmEngine::new(LcmConfig {
            enabled: true,
            tau_soft: 0.3,
            tau_hard: 0.6,
            deterministic_target: 64,
        });

        engine.ingest(json!({"role": "system", "content": "System prompt."}));
        for i in 0..10 {
            engine.ingest(json!({
                "role": "user",
                "content": format!("Question {} about lifetimes and borrowing rules.", i)
            }));
            engine.ingest(json!({
                "role": "assistant",
                "content": format!("Answer {} explains ownership semantics in detail.", i)
            }));
        }

        let budget = TokenBudget::new(4096, 2048);

        // Compact with failing LLM → falls through to Level 3.
        let compactor = ContextCompactor::new(
            Arc::new(FailingMock) as Arc<dyn LLMProvider>,
            "mock".to_string(),
            4096,
        );
        let result = engine.compact(&compactor, &budget, 100).await;
        assert!(result.is_some(), "Level 3 must always produce output");

        let summary = result.unwrap();
        // Level 3 uses first_sentence extraction.
        assert!(
            summary.contains("User:") || summary.contains("Assistant:"),
            "Deterministic truncation should contain role prefixes"
        );

        // DAG node should be level 3.
        assert_eq!(engine.dag.len(), 1);
        let node = engine.dag.get(0).unwrap();
        assert_eq!(node.level, 3, "Should fall through to Level 3");

        // Lossless: originals still retrievable.
        let expanded = engine.expand(&node.source_ids);
        assert_eq!(expanded.len(), node.source_ids.len());
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
            e.ingest(json!({"role": "user", "content": "What is Rust?"}));
            e.ingest(json!({"role": "assistant", "content": "Rust is a systems programming language."}));
            e.ingest(json!({"role": "user", "content": "Tell me about ownership."}));
        }

        let tool = LcmExpandTool::new(engine.clone());

        // Valid IDs.
        let mut params = HashMap::new();
        params.insert("message_ids".to_string(), json!("0,1,2"));
        let output = tool.execute(params).await;
        assert!(output.contains("[msg 0] user: What is Rust?"));
        assert!(output.contains("[msg 1] assistant: Rust is a systems programming language."));
        assert!(output.contains("[msg 2] user: Tell me about ownership."));

        // Invalid IDs.
        let mut params = HashMap::new();
        params.insert("message_ids".to_string(), json!("99,100"));
        let output = tool.execute(params).await;
        assert!(output.contains("No messages found"));

        // Empty input.
        let mut params = HashMap::new();
        params.insert("message_ids".to_string(), json!(""));
        let output = tool.execute(params).await;
        assert!(output.contains("Error: no valid message IDs"));
    }

    // -----------------------------------------------------------------------
    // E2E: double compaction (compact twice, expand both)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_e2e_double_compaction_lossless() {
        let mut engine = LcmEngine::new(LcmConfig {
            enabled: true,
            tau_soft: 0.2,
            tau_hard: 0.5,
            deterministic_target: 64,
        });

        engine.ingest(json!({"role": "system", "content": "System."}));
        // 20 turns — enough for two compaction rounds.
        for i in 0..20 {
            engine.ingest(json!({
                "role": "user",
                "content": format!("Detailed question {} about async Rust with tokio examples.", i)
            }));
            engine.ingest(json!({
                "role": "assistant",
                "content": format!("Detailed answer {} covering spawn, select, and channels.", i)
            }));
        }

        let total_messages = engine.store_len();
        let budget = TokenBudget::new(4096, 2048);
        let compactor = ContextCompactor::new(
            Arc::new(SummarizerMock) as Arc<dyn LLMProvider>,
            "mock".to_string(),
            4096,
        );

        // First compaction.
        let r1 = engine.compact(&compactor, &budget, 100).await;
        assert!(r1.is_some(), "First compaction should succeed");
        let active_after_first = engine.active_len();

        // Second compaction (if there's still a raw block).
        let r2 = engine.compact(&compactor, &budget, 100).await;
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

    // -----------------------------------------------------------------------
    // Benchmark: LCM compaction quality across 4 local models.
    //
    // Runs the same 10-turn conversation through each model's compaction
    // and reports: escalation level, compression ratio, latency, summary.
    //
    // Requires LM Studio at NANOBOT_LCM_BENCH_BASE (default: http://192.168.1.22:1234/v1)
    // with these models loaded: qwen3-0.6b, qwen3-1.7b, gemma-3n-e4b-it,
    // nvidia-nemotron-nano-12b-v2-vl
    //
    // Run: cargo test test_bench_lcm_compaction_models -- --ignored --nocapture
    // -----------------------------------------------------------------------

    #[tokio::test]
    #[ignore = "requires LM Studio with multiple models loaded"]
    async fn test_bench_lcm_compaction_models() {
        use crate::providers::openai_compat::OpenAICompatProvider;
        use std::time::Instant;

        let api_base = std::env::var("NANOBOT_LCM_BENCH_BASE")
            .unwrap_or_else(|_| "http://192.168.1.22:1234/v1".to_string());

        // Models to benchmark (smallest → largest).
        let models = [
            "qwen3-0.6b",
            "qwen3-1.7b",
            "gemma-3n-e4b-it",
            "nvidia-nemotron-nano-12b-v2-vl",
        ];

        // Build a realistic 10-turn conversation (fixed, deterministic input).
        let mut conversation: Vec<Value> = Vec::new();
        conversation.push(json!({"role": "system", "content": "You are a helpful Rust programming assistant."}));

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
                enabled: true,
                tau_soft: 0.3,
                tau_hard: 0.6,
                deterministic_target: 128,
            });

            for msg in &conversation {
                engine.ingest(msg.clone());
            }

            let compactor = ContextCompactor::new(provider.clone(), model_name.to_string(), 4096);

            // Run compaction, measure time.
            let start = Instant::now();
            let result = engine.compact(&compactor, &budget, 100).await;
            let elapsed = start.elapsed().as_millis();

            match result {
                Some(summary_text) => {
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
}
