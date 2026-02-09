//! Context compaction via LLM-powered summarization.
//!
//! When conversation history exceeds the token budget, this module summarizes
//! old messages via a cheap LLM call before falling back to hard truncation.

use std::sync::Arc;

use anyhow::Result;
use serde_json::{json, Value};
use tracing::{debug, warn};

use crate::agent::token_budget::TokenBudget;
use crate::providers::base::LLMProvider;

/// Summarization prompt sent to the LLM.
const SUMMARIZE_PROMPT: &str = "\
Summarize this conversation history concisely. Focus on:
- Key facts and decisions made
- Current task/topic being discussed
- Any pending actions or commitments
Keep it under 500 words.";

/// Compacts conversation context by summarizing old messages via an LLM call.
pub struct ContextCompactor {
    provider: Arc<dyn LLMProvider>,
    model: String,
    /// Max tokens for the summarization response.
    summary_max_tokens: u32,
}

impl ContextCompactor {
    /// Create a new compactor that uses the given provider/model for summaries.
    pub fn new(provider: Arc<dyn LLMProvider>, model: String) -> Self {
        Self {
            provider,
            model,
            summary_max_tokens: 1024,
        }
    }

    /// Compact messages to fit within the token budget.
    ///
    /// If messages already fit, returns them unchanged. Otherwise:
    /// 1. Splits into system + old messages + recent messages (~60% budget for recent)
    /// 2. Summarizes old messages via an LLM call
    /// 3. Returns system + summary + recent messages
    /// 4. Falls back to `TokenBudget::trim_to_fit()` on failure
    pub async fn compact(
        &self,
        messages: &[Value],
        budget: &TokenBudget,
        tool_def_tokens: usize,
    ) -> Vec<Value> {
        let available = budget.available_budget(tool_def_tokens);
        let current = TokenBudget::estimate_tokens(messages);

        if current <= available {
            return messages.to_vec();
        }

        debug!(
            "Context needs compaction: {} tokens > {} available",
            current, available
        );

        // Try LLM summarization.
        match self.compact_via_summary(messages, available).await {
            Ok(compacted) => {
                let new_size = TokenBudget::estimate_tokens(&compacted);
                debug!("Compacted {} -> {} tokens", current, new_size);
                compacted
            }
            Err(e) => {
                warn!("Compaction summarization failed, falling back to trim: {}", e);
                budget.trim_to_fit(messages, tool_def_tokens)
            }
        }
    }

    /// Perform the actual summarization-based compaction.
    async fn compact_via_summary(
        &self,
        messages: &[Value],
        available_budget: usize,
    ) -> Result<Vec<Value>> {
        if messages.is_empty() {
            return Ok(messages.to_vec());
        }

        // Always keep the system message (first).
        let system_msg = messages[0].clone();
        let system_tokens = TokenBudget::estimate_tokens(&[system_msg.clone()]);

        // Reserve space: system + summary overhead + some headroom.
        let summary_headroom = 400; // ~400 tokens for the summary message
        let remaining = available_budget
            .saturating_sub(system_tokens)
            .saturating_sub(summary_headroom);
        if remaining == 0 {
            anyhow::bail!("Budget too small for summarization");
        }
        let recent_budget = (remaining as f64 * 0.6) as usize;

        // Walk backwards from the end to find recent messages that fit in ~60% of budget.
        let body = &messages[1..];
        let mut recent_start = body.len();
        let mut recent_tokens = 0;

        for (i, msg) in body.iter().enumerate().rev() {
            let msg_tokens = TokenBudget::estimate_tokens(&[msg.clone()]);
            if recent_tokens + msg_tokens > recent_budget {
                recent_start = i + 1;
                break;
            }
            recent_tokens += msg_tokens;
            if i == 0 {
                recent_start = 0;
            }
        }

        // If everything is "recent" (nothing to summarize), fall back.
        if recent_start == 0 {
            anyhow::bail!("All messages fit as recent; nothing to summarize");
        }

        let old_messages = &body[..recent_start];
        let recent_messages = &body[recent_start..];

        // Build the summarization request.
        let summary = self.summarize(old_messages).await?;

        // Assemble: system + summary + recent
        let mut result = Vec::with_capacity(2 + recent_messages.len());
        result.push(system_msg);
        result.push(json!({
            "role": "user",
            "content": format!("[Conversation summary: {}]", summary)
        }));
        result.extend_from_slice(recent_messages);

        Ok(result)
    }

    /// Call the LLM to summarize a set of messages.
    async fn summarize(&self, messages: &[Value]) -> Result<String> {
        // Format messages into a readable transcript for the summarizer.
        let mut transcript = String::new();
        for msg in messages {
            let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("unknown");
            let content = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
            // Skip very long tool results - just note they existed.
            if role == "tool" {
                let name = msg.get("name").and_then(|n| n.as_str()).unwrap_or("tool");
                if content.len() > 500 {
                    transcript.push_str(&format!("{}: [result from {} - {} chars]\n", role, name, content.len()));
                } else {
                    transcript.push_str(&format!("{} ({}): {}\n", role, name, content));
                }
            } else if role == "assistant" && msg.get("tool_calls").is_some() {
                // Summarize tool call requests briefly.
                if let Some(Value::Array(calls)) = msg.get("tool_calls") {
                    let names: Vec<&str> = calls.iter()
                        .filter_map(|c| c.get("function").and_then(|f| f.get("name")).and_then(|n| n.as_str()))
                        .collect();
                    transcript.push_str(&format!("assistant: [called tools: {}]\n", names.join(", ")));
                }
                if let Some(text) = msg.get("content").and_then(|c| c.as_str()) {
                    if !text.is_empty() {
                        transcript.push_str(&format!("assistant: {}\n", text));
                    }
                }
            } else {
                transcript.push_str(&format!("{}: {}\n", role, content));
            }
        }

        let summary_messages = vec![
            json!({
                "role": "system",
                "content": SUMMARIZE_PROMPT
            }),
            json!({
                "role": "user",
                "content": transcript
            }),
        ];

        let response = self
            .provider
            .chat(
                &summary_messages,
                None,
                Some(&self.model),
                self.summary_max_tokens,
                0.3, // low temperature for factual summaries
            )
            .await?;

        response
            .content
            .ok_or_else(|| anyhow::anyhow!("Summarization returned no content"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::base::{LLMProvider, LLMResponse};
    use async_trait::async_trait;
    use std::collections::HashMap;

    /// Mock provider that returns a fixed summary.
    struct MockProvider {
        response: String,
    }

    impl MockProvider {
        fn new(response: &str) -> Self {
            Self {
                response: response.to_string(),
            }
        }
    }

    #[async_trait]
    impl LLMProvider for MockProvider {
        async fn chat(
            &self,
            _messages: &[Value],
            _tools: Option<&[Value]>,
            _model: Option<&str>,
            _max_tokens: u32,
            _temperature: f64,
        ) -> Result<LLMResponse> {
            Ok(LLMResponse {
                content: Some(self.response.clone()),
                tool_calls: vec![],
                finish_reason: "stop".to_string(),
                usage: HashMap::new(),
            })
        }

        fn get_default_model(&self) -> &str {
            "mock"
        }
    }

    /// Mock provider that always fails.
    struct FailingProvider;

    #[async_trait]
    impl LLMProvider for FailingProvider {
        async fn chat(
            &self,
            _messages: &[Value],
            _tools: Option<&[Value]>,
            _model: Option<&str>,
            _max_tokens: u32,
            _temperature: f64,
        ) -> Result<LLMResponse> {
            Err(anyhow::anyhow!("LLM unavailable"))
        }

        fn get_default_model(&self) -> &str {
            "mock"
        }
    }

    #[tokio::test]
    async fn test_compact_within_budget_is_noop() {
        let provider = Arc::new(MockProvider::new("summary"));
        let compactor = ContextCompactor::new(provider, "test".into());
        let budget = TokenBudget::new(100_000, 8192);

        let messages = vec![
            json!({"role": "system", "content": "You are helpful."}),
            json!({"role": "user", "content": "Hello"}),
        ];

        let result = compactor.compact(&messages, &budget, 500).await;
        assert_eq!(result.len(), 2);
    }

    #[tokio::test]
    async fn test_compact_summarizes_old_messages() {
        let provider = Arc::new(MockProvider::new("User discussed weather and cats."));
        let compactor = ContextCompactor::new(provider, "test".into());
        // Budget must be smaller than total message tokens to trigger compaction,
        // but large enough for system + summary + a few recent messages.
        let budget = TokenBudget::new(1200, 200);

        let mut messages = vec![json!({"role": "system", "content": "You are a helpful assistant."})];
        // Generate enough messages with long content to exceed the budget.
        for i in 0..30 {
            messages.push(json!({"role": "user", "content": format!("This is a long user message number {} discussing many important topics about the world and various subjects at length", i)}));
            messages.push(json!({"role": "assistant", "content": format!("This is a detailed assistant response number {} providing thorough analysis and helpful information about the topics", i)}));
        }

        let result = compactor.compact(&messages, &budget, 50).await;
        // Should have: system + summary + some recent messages (fewer than 61)
        assert!(
            result.len() < messages.len(),
            "Expected fewer messages after compaction: got {} vs original {}",
            result.len(),
            messages.len()
        );
        assert_eq!(result[0]["role"], "system");
        // Second message should be the summary.
        let summary_content = result[1]["content"].as_str().unwrap();
        assert!(summary_content.contains("Conversation summary"));
    }

    #[tokio::test]
    async fn test_compact_falls_back_on_failure() {
        let provider = Arc::new(FailingProvider);
        let compactor = ContextCompactor::new(provider, "test".into());
        let budget = TokenBudget::new(200, 50);

        let mut messages = vec![json!({"role": "system", "content": "Sys"})];
        for i in 0..20 {
            messages.push(json!({"role": "user", "content": format!("Long message {}", i)}));
        }

        let result = compactor.compact(&messages, &budget, 20).await;
        // Should still return valid messages (via trim_to_fit fallback).
        assert!(!result.is_empty());
        assert_eq!(result[0]["role"], "system");
    }
}
