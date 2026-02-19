//! Token budget management for context window overflow prevention.
//!
//! Uses tiktoken-rs (cl100k_base BPE) for accurate token counting,
//! with a char/4 fallback if the encoder fails to initialize.

use std::collections::HashSet;
use std::sync::OnceLock;

use serde_json::Value;
use tiktoken_rs::CoreBPE;

/// Manages the token budget for LLM context windows.
///
/// Unified budget system: handles both context-window trimming (for message
/// arrays) and per-turn consumption tracking (for the content gate).
pub struct TokenBudget {
    /// Total context window size in tokens (e.g. 128K for Claude, 16K for local).
    max_context: usize,
    /// Tokens reserved for the LLM's response (max_tokens from config).
    reserve_response: usize,
    /// Tokens currently consumed this turn (system prompt + messages + tool results).
    /// Used by ContentGate for admission decisions. Reset each turn.
    used_tokens: usize,
    /// Fraction of context reserved for output generation (0.0–1.0).
    /// When set, `available()` uses this instead of `reserve_response`.
    output_reserve: Option<f64>,
}

impl TokenBudget {
    /// Create a new token budget.
    ///
    /// * `max_context` - Total context window in tokens.
    /// * `max_response` - Tokens reserved for the response.
    pub fn new(max_context: usize, max_response: usize) -> Self {
        Self {
            max_context,
            reserve_response: max_response,
            used_tokens: 0,
            output_reserve: None,
        }
    }

    /// Create a budget with a fractional output reserve (used by ContentGate).
    ///
    /// `output_reserve` is the fraction reserved for generation (0.0–1.0).
    pub fn with_output_reserve(max_context: usize, output_reserve: f32) -> Self {
        Self {
            max_context,
            reserve_response: 0,
            used_tokens: 0,
            output_reserve: Some((output_reserve as f64).clamp(0.0, 0.95)),
        }
    }

    /// Return the maximum context window size in tokens.
    pub fn max_context(&self) -> usize {
        self.max_context
    }

    /// Tokens available for new content (accounting for reserve and usage).
    pub fn available(&self) -> usize {
        let ceiling = if let Some(reserve) = self.output_reserve {
            ((self.max_context as f64) * (1.0 - reserve)).round() as usize
        } else {
            self.max_context.saturating_sub(self.reserve_response)
        };
        ceiling.saturating_sub(self.used_tokens)
    }

    /// Record tokens consumed this turn.
    pub fn consume(&mut self, tokens: usize) {
        self.used_tokens = self.used_tokens.saturating_add(tokens);
    }

    /// Reset consumption tracking (e.g. after compaction or new turn).
    pub fn reset_used(&mut self, used: usize) {
        self.used_tokens = used;
    }

    /// Current tokens consumed this turn.
    pub fn used(&self) -> usize {
        self.used_tokens
    }

    /// Estimate token count for a string using BPE (cl100k_base).
    /// Falls back to char/4 heuristic if the encoder is unavailable.
    pub fn estimate_str_tokens(s: &str) -> usize {
        static ENCODER: OnceLock<Option<CoreBPE>> = OnceLock::new();
        let enc = ENCODER.get_or_init(|| tiktoken_rs::cl100k_base().ok());
        match enc {
            Some(e) => e.encode_ordinary(s).len(),
            None => (s.len() + 3) / 4, // fallback
        }
    }

    /// Estimate token count for a single message Value (public variant).
    pub fn estimate_message_tokens_pub(msg: &Value) -> usize {
        Self::estimate_message_tokens(msg)
    }

    /// Estimate token count for a single message Value.
    fn estimate_message_tokens(msg: &Value) -> usize {
        let mut tokens = 4; // per-message overhead (role, delimiters)

        if let Some(content) = msg.get("content") {
            match content {
                Value::String(s) => tokens += Self::estimate_str_tokens(s),
                Value::Array(parts) => {
                    for part in parts {
                        if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                            tokens += Self::estimate_str_tokens(text);
                        }
                        // Image parts: rough estimate of 85 tokens per image.
                        if part.get("image_url").is_some() {
                            tokens += 85;
                        }
                    }
                }
                _ => {}
            }
        }

        // Tool calls in assistant messages.
        if let Some(Value::Array(tool_calls)) = msg.get("tool_calls") {
            for tc in tool_calls {
                tokens += 10; // overhead per tool call
                if let Some(f) = tc.get("function") {
                    if let Some(name) = f.get("name").and_then(|n| n.as_str()) {
                        tokens += Self::estimate_str_tokens(name);
                    }
                    if let Some(args) = f.get("arguments").and_then(|a| a.as_str()) {
                        tokens += Self::estimate_str_tokens(args);
                    }
                }
            }
        }

        tokens
    }

    /// Estimate total tokens for a message array.
    pub fn estimate_tokens(messages: &[Value]) -> usize {
        messages.iter().map(Self::estimate_message_tokens).sum()
    }

    /// Estimate tokens for tool definitions.
    pub fn estimate_tool_def_tokens(tool_defs: &[Value]) -> usize {
        let json = serde_json::to_string(tool_defs).unwrap_or_default();
        Self::estimate_str_tokens(&json)
    }

    /// Available budget for messages (after reserving response + tool defs).
    pub fn available_budget(&self, tool_def_tokens: usize) -> usize {
        self.max_context
            .saturating_sub(self.reserve_response)
            .saturating_sub(tool_def_tokens)
    }

    /// Trim message history to fit within the token budget.
    ///
    /// Strategy (4 stages):
    /// 1. **Soft**: Truncate old tool results to summaries.
    /// 1.5. **Age-based**: Drop messages older than `max_age_turns` (if set).
    /// 2. **Medium**: Drop oldest history messages (keep system + recent).
    /// 3. **Hard**: Keep only system prompt + last user message + summary.
    ///
    /// The system prompt (first message) and the most recent user message
    /// (last message) are always preserved.
    pub fn trim_to_fit(&self, messages: &[Value], tool_def_tokens: usize) -> Vec<Value> {
        self.trim_to_fit_with_age(messages, tool_def_tokens, 0, 0)
    }

    /// Like `trim_to_fit`, but with age-based eviction.
    ///
    /// `current_turn` is the current turn number (from learning_turn_counter).
    /// `max_age_turns` is the maximum age in turns before a message is preferred
    /// for eviction (0 = disabled).
    pub fn trim_to_fit_with_age(
        &self,
        messages: &[Value],
        tool_def_tokens: usize,
        current_turn: u64,
        max_age_turns: usize,
    ) -> Vec<Value> {
        let budget = self.available_budget(tool_def_tokens);
        let mut msgs = messages.to_vec();

        // Stage 0: Proactive age-based eviction — runs even when within budget.
        // This prevents context rot from old messages accumulating in large windows.
        if max_age_turns > 0 && current_turn > 0 && msgs.len() > 2 {
            let age_threshold = current_turn.saturating_sub(max_age_turns as u64);
            let last_idx = msgs.len() - 1;
            msgs = msgs
                .into_iter()
                .enumerate()
                .filter(|(i, m)| {
                    if *i == 0 || *i == last_idx {
                        return true;
                    }
                    if let Some(turn) = m.get("_turn").and_then(|v| v.as_u64()) {
                        turn >= age_threshold
                    } else {
                        true
                    }
                })
                .map(|(_, m)| m)
                .collect();

            // Remove orphaned tool results whose assistant was age-evicted.
            let known_call_ids: HashSet<String> = msgs
                .iter()
                .filter(|m| m.get("role").and_then(|r| r.as_str()) == Some("assistant"))
                .filter_map(|m| m.get("tool_calls").and_then(|tc| tc.as_array()))
                .flat_map(|tcs| tcs.iter())
                .filter_map(|tc| tc.get("id").and_then(|id| id.as_str()).map(String::from))
                .collect();
            msgs.retain(|m| {
                if m.get("role").and_then(|r| r.as_str()) != Some("tool") {
                    return true;
                }
                m.get("tool_call_id")
                    .and_then(|id| id.as_str())
                    .map(|id| known_call_ids.contains(id))
                    .unwrap_or(true) // keep tool msgs without tool_call_id (legacy)
            });
        }

        // Already within budget?
        if Self::estimate_tokens(&msgs) <= budget {
            return msgs;
        }

        // Stage 1: Truncate old tool results (keeping most recent 4).
        let tool_msg_indices: Vec<usize> = msgs
            .iter()
            .enumerate()
            .filter(|(_, m)| m.get("role").and_then(|r| r.as_str()) == Some("tool"))
            .map(|(i, _)| i)
            .collect();

        if tool_msg_indices.len() > 4 {
            let truncate_up_to = tool_msg_indices.len() - 4;
            for &idx in &tool_msg_indices[..truncate_up_to] {
                if let Some(content) = msgs[idx].get("content").and_then(|c| c.as_str()) {
                    if content.len() > 200 {
                        let summary = format!(
                            "[truncated: {}... ({} chars)]",
                            &content[..crate::utils::helpers::floor_char_boundary(content, 100)],
                            content.len()
                        );
                        msgs[idx]["content"] = Value::String(summary);
                    }
                }
            }
        }

        if Self::estimate_tokens(&msgs) <= budget {
            return msgs;
        }

        // Stage 2: Drop oldest non-system, non-recent messages.
        // Keep: system (index 0) + last N messages that fit.
        if msgs.len() > 2 {
            let system_msg = msgs[0].clone();
            let system_tokens = Self::estimate_message_tokens(&system_msg);

            let mut kept_tail: Vec<Value> = Vec::new();
            let mut tail_tokens = 0;
            let remaining_budget = budget.saturating_sub(system_tokens);

            // Walk backwards from the end, keeping messages that fit.
            // Track tool_call IDs from skipped assistant messages so we also
            // skip their orphaned tool results (protocol safety).
            let mut skipped_call_ids: HashSet<String> = HashSet::new();

            for msg in msgs[1..].iter().rev() {
                let msg_tokens = Self::estimate_message_tokens(msg);
                let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("");

                // Skip tool results whose assistant was already skipped.
                if role == "tool" {
                    if let Some(id) = msg.get("tool_call_id").and_then(|v| v.as_str()) {
                        if skipped_call_ids.contains(id) {
                            continue;
                        }
                    }
                }

                if tail_tokens + msg_tokens <= remaining_budget {
                    kept_tail.push(msg.clone());
                    tail_tokens += msg_tokens;
                } else {
                    // Track tool_call IDs from skipped assistant messages.
                    if role == "assistant" {
                        if let Some(tcs) = msg.get("tool_calls").and_then(|v| v.as_array()) {
                            for tc in tcs {
                                if let Some(id) = tc.get("id").and_then(|v| v.as_str()) {
                                    skipped_call_ids.insert(id.to_string());
                                }
                            }
                        }
                    }
                    continue;
                }
            }

            kept_tail.reverse();

            // Post-walk cleanup: remove tool results whose assistant was
            // skipped. This catches cases where the backward walk saw the tool
            // result before discovering its assistant would be skipped.
            if !skipped_call_ids.is_empty() {
                kept_tail.retain(|m| {
                    if m.get("role").and_then(|r| r.as_str()) != Some("tool") {
                        return true;
                    }
                    m.get("tool_call_id")
                        .and_then(|id| id.as_str())
                        .map(|id| !skipped_call_ids.contains(id))
                        .unwrap_or(true)
                });
            }

            let mut result = vec![system_msg];
            result.extend(kept_tail);
            msgs = result;
        }

        if Self::estimate_tokens(&msgs) <= budget {
            return msgs;
        }

        // Stage 3 (hard): System prompt + summary + last user message.
        if msgs.len() > 2 {
            let system_msg = msgs[0].clone();
            let last_msg = msgs.last().cloned().unwrap();
            let summary_msg = serde_json::json!({
                "role": "user",
                "content": "[Previous conversation truncated due to context limits. Please continue from the latest message.]"
            });
            msgs = vec![system_msg, summary_msg, last_msg];
        }

        msgs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_estimate_str_tokens() {
        // With tiktoken (cl100k_base): "hello" is 1 token.
        assert_eq!(TokenBudget::estimate_str_tokens("hello"), 1);
        // Empty string → 0
        assert_eq!(TokenBudget::estimate_str_tokens(""), 0);
        // Longer strings should produce reasonable counts.
        let s = "a".repeat(100);
        let tokens = TokenBudget::estimate_str_tokens(&s);
        assert!(
            tokens > 0 && tokens < 50,
            "100 'a' chars should be <50 tokens, got {}",
            tokens
        );
    }

    #[test]
    fn test_estimate_tokens_basic() {
        let messages = vec![
            json!({"role": "system", "content": "You are helpful."}),
            json!({"role": "user", "content": "Hello"}),
        ];
        let tokens = TokenBudget::estimate_tokens(&messages);
        assert!(tokens > 0);
        // System message: 4 overhead + ~4 for content = ~8
        // User message: 4 overhead + ~2 for content = ~6
        assert!(tokens < 50, "tokens={}", tokens);
    }

    #[test]
    fn test_estimate_tool_call_tokens() {
        let msg = json!({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "tc_1",
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": "{\"path\": \"/tmp/test.txt\"}"
                }
            }]
        });
        let tokens = TokenBudget::estimate_message_tokens(&msg);
        assert!(tokens > 10, "should count tool call tokens, got {}", tokens);
    }

    #[test]
    fn test_trim_no_op_when_within_budget() {
        let budget = TokenBudget::new(100_000, 8192);
        let messages = vec![
            json!({"role": "system", "content": "You are helpful."}),
            json!({"role": "user", "content": "Hello"}),
        ];
        let trimmed = budget.trim_to_fit(&messages, 500);
        assert_eq!(trimmed.len(), 2);
    }

    #[test]
    fn test_trim_stage1_truncates_old_tool_results() {
        // Create a scenario with many large tool results.
        let budget = TokenBudget::new(1000, 200);
        let mut messages = vec![
            json!({"role": "system", "content": "System"}),
            json!({"role": "user", "content": "Do something"}),
        ];

        // Add 6 tool call / result pairs with large results.
        for i in 0..6 {
            messages.push(json!({
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": format!("tc_{}", i), "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]
            }));
            messages.push(json!({
                "role": "tool",
                "tool_call_id": format!("tc_{}", i),
                "name": "read_file",
                "content": "x".repeat(500)
            }));
        }
        messages.push(json!({"role": "user", "content": "Continue"}));

        let trimmed = budget.trim_to_fit(&messages, 100);
        // Should have truncated some old tool results.
        let total_tokens = TokenBudget::estimate_tokens(&trimmed);
        assert!(
            total_tokens <= 700,
            "should be within budget, got {} tokens",
            total_tokens
        );
    }

    #[test]
    fn test_trim_stage2_drops_old_messages() {
        // Very tight budget that forces dropping old messages.
        let budget = TokenBudget::new(200, 50);
        let mut messages = vec![json!({"role": "system", "content": "Sys"})];
        for i in 0..20 {
            messages.push(json!({"role": "user", "content": format!("Message {}", i)}));
            messages.push(json!({"role": "assistant", "content": format!("Reply {}", i)}));
        }
        messages.push(json!({"role": "user", "content": "Latest question"}));

        let trimmed = budget.trim_to_fit(&messages, 20);
        // Should have system + some recent messages.
        assert!(trimmed.len() < messages.len());
        assert_eq!(trimmed[0]["role"], "system");
        // Last message should be preserved.
        let last = trimmed.last().unwrap();
        assert_eq!(last["content"], "Latest question");
    }

    #[test]
    fn test_trim_stage3_hard_reset() {
        // Extremely tight budget: only room for system + 1-2 messages.
        let budget = TokenBudget::new(50, 10);
        let mut messages = vec![json!({"role": "system", "content": "S"})];
        for _ in 0..50 {
            messages.push(json!({"role": "user", "content": "a".repeat(100)}));
        }

        let trimmed = budget.trim_to_fit(&messages, 5);
        // Stage 3: system + summary + last message = 3.
        assert!(trimmed.len() <= 3, "got {} messages", trimmed.len());
        assert_eq!(trimmed[0]["role"], "system");
    }

    #[test]
    fn test_trim_stage2_skips_oversized_keeps_smaller() {
        // One oversized message in the middle should be skipped, not block
        // earlier messages from being kept.
        let budget = TokenBudget::new(400, 50);
        let mut messages = vec![json!({"role": "system", "content": "S"})];
        // Small message.
        messages.push(json!({"role": "user", "content": "tiny"}));
        // Oversized message (fills most of the budget).
        messages.push(json!({"role": "assistant", "content": "x".repeat(2000)}));
        // Small message at the end.
        messages.push(json!({"role": "user", "content": "latest"}));

        let trimmed = budget.trim_to_fit(&messages, 10);
        // The oversized message should be skipped, but both small messages
        // should survive (system + tiny + latest).
        let contents: Vec<&str> = trimmed
            .iter()
            .filter_map(|m| m["content"].as_str())
            .collect();
        assert!(contents.contains(&"latest"), "latest message must survive");
        assert!(
            contents.contains(&"tiny"),
            "earlier small message should survive when oversized one is skipped"
        );
    }

    #[test]
    fn test_available_budget() {
        let budget = TokenBudget::new(128000, 8192);
        let available = budget.available_budget(2000);
        assert_eq!(available, 128000 - 8192 - 2000);
    }

    #[test]
    fn test_estimate_tool_def_tokens() {
        let defs = vec![json!({
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}
            }
        })];
        let tokens = TokenBudget::estimate_tool_def_tokens(&defs);
        assert!(tokens > 10, "tool def should have tokens, got {}", tokens);
    }

    #[test]
    fn test_age_based_eviction_drops_old_messages() {
        // Budget is generous — age eviction should fire before size eviction.
        let budget = TokenBudget::new(100_000, 8192);
        let mut messages = vec![json!({"role": "system", "content": "System prompt"})];

        // Add messages with turn tags. Turns 1-10 are "old", turn 50 is current.
        for turn in 1..=10 {
            messages.push(
                json!({"role": "user", "content": format!("Old msg turn {}", turn), "_turn": turn}),
            );
            messages.push(json!({"role": "assistant", "content": format!("Old reply {}", turn), "_turn": turn}));
        }
        // Recent messages.
        messages.push(json!({"role": "user", "content": "Recent question", "_turn": 50}));

        let original_count = messages.len();

        // max_age_turns=10, current_turn=50 → threshold = 40. All turns 1-10 < 40 → evicted.
        let trimmed = budget.trim_to_fit_with_age(&messages, 500, 50, 10);

        // Old messages (turns 1-10) should be evicted, system + recent kept.
        assert!(
            trimmed.len() < original_count,
            "should have evicted old messages"
        );
        assert_eq!(trimmed[0]["role"], "system");
        let last = trimmed.last().unwrap();
        assert_eq!(last["content"], "Recent question");
    }

    #[test]
    fn test_age_based_eviction_keeps_recent_messages() {
        let budget = TokenBudget::new(100_000, 8192);
        let mut messages = vec![json!({"role": "system", "content": "System"})];

        // All messages are recent (turns 45-50, threshold=40 with max_age=10, current=50).
        for turn in 45..=50 {
            messages
                .push(json!({"role": "user", "content": format!("Msg {}", turn), "_turn": turn}));
        }

        let trimmed = budget.trim_to_fit_with_age(&messages, 500, 50, 10);
        // No eviction — all messages are within age window.
        assert_eq!(trimmed.len(), messages.len());
    }

    #[test]
    fn test_age_based_eviction_preserves_untagged_messages() {
        let budget = TokenBudget::new(100_000, 8192);
        let mut messages = vec![json!({"role": "system", "content": "System"})];

        // Mix of tagged and untagged messages.
        messages.push(json!({"role": "user", "content": "No turn tag"})); // no _turn
        messages.push(json!({"role": "user", "content": "Old", "_turn": 1}));
        messages.push(json!({"role": "user", "content": "Recent", "_turn": 50}));

        let trimmed = budget.trim_to_fit_with_age(&messages, 500, 50, 10);
        // Untagged message preserved, old tagged dropped.
        let contents: Vec<&str> = trimmed
            .iter()
            .filter_map(|m| m["content"].as_str())
            .collect();
        assert!(
            contents.contains(&"No turn tag"),
            "untagged messages must survive"
        );
        assert!(contents.contains(&"Recent"), "recent messages must survive");
        assert!(
            !contents.contains(&"Old"),
            "old tagged messages should be evicted"
        );
    }

    #[test]
    fn test_stage2_skips_tool_results_when_assistant_skipped() {
        // Tight budget: system + large assistant+tool_calls should be skipped,
        // and the tool result must also be skipped (not orphaned).
        let budget = TokenBudget::new(300, 50);
        let messages = vec![
            json!({"role": "system", "content": "sys"}),
            json!({"role": "user", "content": "question"}),
            json!({
                "role": "assistant", "content": "x".repeat(2000),
                "tool_calls": [{"id": "tc_1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]
            }),
            json!({"role": "tool", "tool_call_id": "tc_1", "name": "read_file", "content": "data"}),
            json!({"role": "user", "content": "latest question"}),
        ];
        let trimmed = budget.trim_to_fit(&messages, 10);
        // The tool result should NOT survive if its assistant was dropped.
        let has_orphan = trimmed.iter().any(|m| {
            m.get("role").and_then(|r| r.as_str()) == Some("tool")
                && !trimmed.iter().any(|a| {
                    a.get("tool_calls")
                        .and_then(|tc| tc.as_array())
                        .map(|tcs| {
                            tcs.iter()
                                .any(|t| t.get("id").and_then(|i| i.as_str()) == Some("tc_1"))
                        })
                        .unwrap_or(false)
                })
        });
        assert!(
            !has_orphan,
            "Tool result must not survive when its assistant is dropped"
        );
    }

    #[test]
    fn test_age_eviction_removes_orphaned_tool_results() {
        // Age-evict a user message at the start of a turn, which should also
        // clean up the subsequent tool results from that turn's assistant.
        let budget = TokenBudget::new(100_000, 8192);
        let messages = vec![
            json!({"role": "system", "content": "System"}),
            // Old turn with tool calls — user gets _turn tag, assistant+tool don't.
            json!({"role": "user", "content": "Old question", "_turn": 1}),
            json!({
                "role": "assistant", "content": "",
                "tool_calls": [{"id": "tc_old", "type": "function", "function": {"name": "exec", "arguments": "{}"}}]
            }),
            json!({"role": "tool", "tool_call_id": "tc_old", "name": "exec", "content": "result"}),
            json!({"role": "assistant", "content": "Done with old task"}),
            // Recent turn.
            json!({"role": "user", "content": "New question", "_turn": 50}),
        ];

        let trimmed = budget.trim_to_fit_with_age(&messages, 500, 50, 10);
        // The tool result for tc_old should be removed because its assistant survived
        // but only if the assistant also got removed. Let's check for orphans.
        for m in &trimmed {
            if m.get("role").and_then(|r| r.as_str()) == Some("tool") {
                let tool_call_id = m.get("tool_call_id").and_then(|v| v.as_str()).unwrap_or("");
                // Verify the matching assistant is also present.
                let has_matching_assistant = trimmed.iter().any(|a| {
                    a.get("tool_calls")
                        .and_then(|tc| tc.as_array())
                        .map(|tcs| {
                            tcs.iter()
                                .any(|t| t.get("id").and_then(|i| i.as_str()) == Some(tool_call_id))
                        })
                        .unwrap_or(false)
                });
                assert!(
                    has_matching_assistant,
                    "Tool result for '{}' is orphaned — its assistant was evicted",
                    tool_call_id
                );
            }
        }
    }
}
