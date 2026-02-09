//! Token budget management for context window overflow prevention.
//!
//! Uses character-based estimation (1 token ~ 4 chars) rather than a
//! tokenizer crate. Good enough for budget management.

use serde_json::Value;

/// Manages the token budget for LLM context windows.
pub struct TokenBudget {
    /// Total context window size in tokens (e.g. 128K for Claude, 16K for local).
    max_context: usize,
    /// Tokens reserved for the LLM's response (max_tokens from config).
    reserve_response: usize,
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
        }
    }

    /// Estimate token count for a string (~4 chars per token).
    pub fn estimate_str_tokens(s: &str) -> usize {
        // Add 1 to avoid underestimating short strings.
        (s.len() + 3) / 4
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
    fn available_budget(&self, tool_def_tokens: usize) -> usize {
        self.max_context
            .saturating_sub(self.reserve_response)
            .saturating_sub(tool_def_tokens)
    }

    /// Trim message history to fit within the token budget.
    ///
    /// Strategy (3 stages):
    /// 1. **Soft**: Truncate old tool results to summaries.
    /// 2. **Medium**: Drop oldest history messages (keep system + recent).
    /// 3. **Hard**: Keep only system prompt + last user message + summary.
    ///
    /// The system prompt (first message) and the most recent user message
    /// (last message) are always preserved.
    pub fn trim_to_fit(&self, messages: &[Value], tool_def_tokens: usize) -> Vec<Value> {
        let budget = self.available_budget(tool_def_tokens);
        let mut msgs = messages.to_vec();

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
                            &content[..100],
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
            for msg in msgs[1..].iter().rev() {
                let msg_tokens = Self::estimate_message_tokens(msg);
                if tail_tokens + msg_tokens <= remaining_budget {
                    kept_tail.push(msg.clone());
                    tail_tokens += msg_tokens;
                } else {
                    break;
                }
            }

            kept_tail.reverse();
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
        // "hello" = 5 chars → (5+3)/4 = 2 tokens
        assert_eq!(TokenBudget::estimate_str_tokens("hello"), 2);
        // Empty string → (0+3)/4 = 0
        assert_eq!(TokenBudget::estimate_str_tokens(""), 0);
        // 100 chars → 25 tokens
        let s = "a".repeat(100);
        assert_eq!(TokenBudget::estimate_str_tokens(&s), 25);
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
}
