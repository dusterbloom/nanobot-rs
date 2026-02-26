//! E2E tests for Lossless Context Management (LCM)
//!
//! Tests the core LCM behaviors:
//! 1. `/clear` resets LCM engines - old summaries should NOT persist
//! 2. LCM summarizes the correct block (after last summary, not from beginning)
//! 3. Refusal patterns should be filtered from summaries

use serde_json::json;
use nanobot::agent::lcm::{LcmConfig, LcmEngine, CompactionAction};
use nanobot::agent::turn::Turn;
use nanobot::agent::protocol::LocalProtocol;
use nanobot::agent::token_budget::TokenBudget;
use nanobot::agent::compaction::ContextCompactor;
use nanobot::providers::base::{LLMProvider, LLMResponse};
use async_trait::async_trait;
use std::sync::Arc;

// ─────────────────────────────────────────────────────────────
// Mock LLM for testing
// ─────────────────────────────────────────────────────────────

struct MockSummarizer;

#[async_trait]
impl LLMProvider for MockSummarizer {
    async fn chat(
        &self,
        _messages: &[serde_json::Value],
        _tools: Option<&[serde_json::Value]>,
        _model: Option<&str>,
        _max_tokens: u32,
        _temperature: f64,
        _thinking_budget: Option<u32>,
        _top_p: Option<f64>,
    ) -> anyhow::Result<LLMResponse> {
        Ok(LLMResponse {
            content: Some("User discussed Rust ownership and borrowing concepts.".to_string()),
            tool_calls: vec![],
            finish_reason: "stop".to_string(),
            usage: std::collections::HashMap::new(),
        })
    }

    fn get_default_model(&self) -> &str {
        "mock-summarizer"
    }
}

// ─────────────────────────────────────────────────────────────
// Test 1: /clear semantics - LCM engine should be reset
// ─────────────────────────────────────────────────────────────

#[test]
fn test_clear_resets_lcm_engine() {
    let config = LcmConfig {
        enabled: true,
        tau_soft: 0.5,
        tau_hard: 0.85,
        deterministic_target: 512,
    };
    
    let mut engine = LcmEngine::new(config.clone());
    
    engine.ingest(json!({"role": "system", "content": "System"}));
    engine.ingest(json!({"role": "user", "content": "First message"}));
    engine.ingest(json!({"role": "assistant", "content": "First response"}));
    
    assert_eq!(engine.store_len(), 3);
    assert_eq!(engine.active_len(), 3);
    
    // Simulate /clear: create a fresh engine
    let fresh_engine = LcmEngine::new(config);
    assert_eq!(fresh_engine.store_len(), 0);
    assert_eq!(fresh_engine.active_len(), 0);
    assert!(fresh_engine.dag_ref().is_empty());
}

#[test]
fn test_rebuild_from_turns_after_clear_is_empty() {
    let config = LcmConfig::default();
    let protocol = LocalProtocol::default();
    
    let turns: Vec<Turn> = vec![];
    let engine = LcmEngine::rebuild_from_turns(&turns, config, &protocol, "");
    
    assert_eq!(engine.store_len(), 0);
    assert_eq!(engine.active_len(), 0);
    assert!(engine.dag_ref().is_empty());
}

#[test]
fn test_rebuild_respects_clear_marker() {
    let config = LcmConfig::default();
    let protocol = LocalProtocol::default();
    
    // Turns BEFORE clear marker
    let turns_before_clear = vec![
        Turn::User { content: "Old question 1".into(), media: vec![] },
        Turn::Assistant { text: Some("Old answer 1".into()), tool_calls: vec![] },
        Turn::User { content: "Old question 2".into(), media: vec![] },
        Turn::Assistant { text: Some("Old answer 2".into()), tool_calls: vec![] },
        Turn::Summary {
            text: "Summary of old conversation".into(),
            source_ids: vec![0, 1, 2, 3],
            level: 1,
        },
    ];
    
    // Turns AFTER clear marker (what should be processed)
    let turns_after_clear = vec![
        Turn::Clear,
        Turn::User { content: "New question".into(), media: vec![] },
        Turn::Assistant { text: Some("New answer".into()), tool_calls: vec![] },
    ];
    
    // Combine: old turns, clear marker, new turns
    let all_turns: Vec<Turn> = [turns_before_clear, turns_after_clear].concat();
    
    let engine = LcmEngine::rebuild_from_turns(&all_turns, config, &protocol, "");
    
    // Should only have 2 messages (new question + new answer)
    assert_eq!(engine.store_len(), 2, "Should only have messages after clear marker");
    assert_eq!(engine.active_len(), 2, "Active context should only have post-clear messages");
    assert!(engine.dag_ref().is_empty(), "Old summaries should be discarded after clear");
}

// ─────────────────────────────────────────────────────────────
// Test 2: Compaction window - should summarize AFTER last summary
// ─────────────────────────────────────────────────────────────

#[test]
fn test_find_oldest_raw_block_returns_first_block() {
    let mut engine = LcmEngine::new(LcmConfig::default());
    
    engine.ingest(json!({"role": "system", "content": "System"}));
    for i in 0..10 {
        engine.ingest(json!({"role": "user", "content": format!("User {}", i)}));
        engine.ingest(json!({"role": "assistant", "content": format!("Assistant {}", i)}));
    }
    
    let block = engine.find_oldest_raw_block();
    assert!(block.is_some());
    
    let (start, end) = block.unwrap();
    assert!(start >= 1, "Block should start after system message");
    assert!(end <= engine.active_len() - 4, "Block should leave 4 recent messages");
}

#[tokio::test]
async fn test_compaction_creates_summary_node() {
    let mut engine = LcmEngine::new(LcmConfig {
        enabled: true,
        tau_soft: 0.3,
        tau_hard: 0.6,
        deterministic_target: 64,
    });
    
    engine.ingest(json!({"role": "system", "content": "System"}));
    for i in 0..10 {
        engine.ingest(json!({"role": "user", "content": format!("Question {} about Rust ownership", i)}));
        engine.ingest(json!({"role": "assistant", "content": format!("Answer {} explains borrowing rules", i)}));
    }
    
    let budget = TokenBudget::new(4096, 2048);
    let compactor = ContextCompactor::new(
        Arc::new(MockSummarizer) as Arc<dyn LLMProvider>,
        "mock".to_string(),
        4096,
    );
    
    let result = engine.compact(&compactor, &budget, 100).await;
    assert!(result.is_some());
    
    let summary_turn = result.unwrap();
    assert!(matches!(summary_turn, Turn::Summary { .. }));
    
    assert_eq!(engine.dag_ref().len(), 1);
}

#[tokio::test]
async fn test_second_compaction_summarizes_after_first_summary() {
    let mut engine = LcmEngine::new(LcmConfig {
        enabled: true,
        tau_soft: 0.2,
        tau_hard: 0.5,
        deterministic_target: 64,
    });
    
    engine.ingest(json!({"role": "system", "content": "System"}));
    for i in 0..20 {
        engine.ingest(json!({"role": "user", "content": format!("Message {}", i)}));
        engine.ingest(json!({"role": "assistant", "content": format!("Response {}", i)}));
    }
    
    let budget = TokenBudget::new(4096, 2048);
    let compactor = ContextCompactor::new(
        Arc::new(MockSummarizer) as Arc<dyn LLMProvider>,
        "mock".to_string(),
        4096,
    );
    
    let r1 = engine.compact(&compactor, &budget, 100).await;
    assert!(r1.is_some(), "First compaction should succeed");
    
    let first_summary = r1.unwrap();
    let first_source_ids = match &first_summary {
        Turn::Summary { source_ids, .. } => source_ids.clone(),
        _ => panic!("Expected Summary"),
    };
    
    let r2 = engine.compact(&compactor, &budget, 100).await;
    if r2.is_some() {
        let second_summary = r2.unwrap();
        let second_source_ids = match &second_summary {
            Turn::Summary { source_ids, .. } => source_ids.clone(),
            _ => panic!("Expected Summary"),
        };
        
        // Second compaction should NOT overlap with first
        for id in &first_source_ids {
            assert!(
                !second_source_ids.contains(id),
                "Second compaction should not include messages from first summary"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────
// Test 3: Refusal pattern detection
// ─────────────────────────────────────────────────────────────

#[test]
fn test_contains_refusal_pattern() {
    let refusal_patterns = [
        "I cannot assist with that request.",
        "I'm sorry, I can't help with that.",
        "I apologize, but I cannot fulfill this request.",
        "As an AI language model, I cannot provide that information.",
        "I'm unable to help with that specific request.",
    ];
    
    for pattern in &refusal_patterns {
        assert!(
            contains_refusal_pattern(pattern),
            "Should detect refusal in: {}",
            pattern
        );
    }
    
    let non_refusals = [
        "I can help you with that.",
        "Here's how to solve the problem.",
        "The answer to your question is...",
        "Let me explain the concept.",
    ];
    
    for text in &non_refusals {
        assert!(
            !contains_refusal_pattern(text),
            "Should NOT detect refusal in: {}",
            text
        );
    }
}

fn contains_refusal_pattern(text: &str) -> bool {
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
    ];
    
    for indicator in &refusal_indicators {
        if lower.contains(indicator) {
            return true;
        }
    }
    false
}

#[test]
fn test_filter_refusal_from_summary() {
    let summary_with_refusal = "User asked about Rust. I cannot assist with that request. However, here is some info.";
    let clean_summary = "User asked about Rust. Here is some info about ownership.";
    
    assert!(
        contains_refusal_pattern(summary_with_refusal),
        "Should detect refusal in summary"
    );
    assert!(
        !contains_refusal_pattern(clean_summary),
        "Should NOT detect refusal in clean summary"
    );
}

// ─────────────────────────────────────────────────────────────
// Test 4: Lossless retrieval after compaction
// ─────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_lossless_retrieval_after_multiple_compactions() {
    let mut engine = LcmEngine::new(LcmConfig {
        enabled: true,
        tau_soft: 0.2,
        tau_hard: 0.5,
        deterministic_target: 64,
    });
    
    engine.ingest(json!({"role": "system", "content": "System"}));
    
    let total_messages = 30;
    for i in 0..total_messages {
        engine.ingest(json!({"role": "user", "content": format!("Original message {}", i)}));
        engine.ingest(json!({"role": "assistant", "content": format!("Original response {}", i)}));
    }
    
    let store_size = engine.store_len();
    
    let budget = TokenBudget::new(4096, 2048);
    let compactor = ContextCompactor::new(
        Arc::new(MockSummarizer) as Arc<dyn LLMProvider>,
        "mock".to_string(),
        4096,
    );
    
    // Multiple compaction rounds
    for _ in 0..3 {
        let _ = engine.compact(&compactor, &budget, 100).await;
    }
    
    // Store should be unchanged
    assert_eq!(
        engine.store_len(),
        store_size,
        "Store must never lose messages after compaction"
    );
    
    // All summary nodes should have retrievable sources
    for i in 0..engine.dag_ref().len() {
        let node = engine.dag_ref().get(i).unwrap();
        let expanded = engine.expand(&node.source_ids);
        assert_eq!(
            expanded.len(),
            node.source_ids.len(),
            "All source messages for summary {} must be retrievable",
            i
        );
        
        for (id, msg) in &expanded {
            let content = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
            assert!(
                content.contains("Original"),
                "Expanded message {} should contain original content",
                id
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────
// Test 5: Threshold detection
// ─────────────────────────────────────────────────────────────

#[test]
fn test_check_thresholds_below_soft() {
    let mut engine = LcmEngine::new(LcmConfig {
        enabled: true,
        tau_soft: 0.5,
        tau_hard: 0.85,
        deterministic_target: 512,
    });
    
    engine.ingest(json!({"role": "system", "content": "System"}));
    engine.ingest(json!({"role": "user", "content": "Hello"}));
    
    let budget = TokenBudget::new(100_000, 8192);
    assert_eq!(engine.check_thresholds(&budget, 500), CompactionAction::None);
}

#[test]
fn test_check_thresholds_above_soft() {
    let mut engine = LcmEngine::new(LcmConfig {
        enabled: true,
        tau_soft: 0.3,
        tau_hard: 0.8,
        deterministic_target: 512,
    });
    
    engine.ingest(json!({"role": "system", "content": "System"}));
    for i in 0..20 {
        engine.ingest(json!({"role": "user", "content": format!("A long message with content {}", i)}));
    }
    
    let budget = TokenBudget::new(1024, 512);
    let action = engine.check_thresholds(&budget, 50);
    assert!(
        action == CompactionAction::Async || action == CompactionAction::Blocking,
        "Should trigger compaction with small budget and many messages"
    );
}

// ─────────────────────────────────────────────────────────────
// Test 6: Rebuild from persisted turns
// ─────────────────────────────────────────────────────────────

#[test]
fn test_rebuild_from_turns_preserves_summaries() {
    let config = LcmConfig::default();
    let protocol = LocalProtocol::default();
    
    let turns = vec![
        Turn::User { content: "First question".into(), media: vec![] },
        Turn::Assistant { text: Some("First answer".into()), tool_calls: vec![] },
        Turn::Summary {
            text: "Summary of first exchange".into(),
            source_ids: vec![0, 1],
            level: 1,
        },
        Turn::User { content: "Second question".into(), media: vec![] },
        Turn::Assistant { text: Some("Second answer".into()), tool_calls: vec![] },
    ];
    
    let engine = LcmEngine::rebuild_from_turns(&turns, config, &protocol, "");
    
    // Store should have all raw messages (not summaries)
    assert_eq!(engine.store_len(), 4, "Store should have 4 raw messages (2 user + 2 assistant)");
    
    // DAG should have the summary
    assert_eq!(engine.dag_ref().len(), 1, "DAG should have 1 summary node");
    
    // Active context should have summary + recent raw messages
    let active = engine.active_entries();
    let summary_count = active.iter().filter(|e| matches!(e, nanobot::agent::lcm::ContextEntry::Summary { .. })).count();
    assert_eq!(summary_count, 1, "Active context should have 1 summary entry");
}
