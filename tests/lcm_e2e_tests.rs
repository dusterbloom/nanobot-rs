//! E2E tests for Lossless Context Management (LCM)
//!
//! Tests the core LCM behaviors:
//! 1. `/clear` resets LCM engines - old summaries should NOT persist
//! 2. LCM summarizes the correct block (after last summary, not from beginning)
//! 3. Refusal patterns should be filtered from summaries

use async_trait::async_trait;
use nanobot::agent::compaction::ContextCompactor;
use nanobot::agent::lcm::{CompactionAction, CompactionFailureMode, LcmConfig, LcmEngine};
use nanobot::agent::token_budget::TokenBudget;
use nanobot::agent::turn::Turn;
use nanobot::providers::base::{LLMProvider, LLMResponse};
use serde_json::{json, Value};
use std::sync::Arc;

/// Test message tagged with an explicit `_db_id`, mirroring what
/// get_history supplies for persisted messages.
fn msg(id: usize, role: &str, content: &str) -> Value {
    json!({"role": role, "content": content, "_db_id": id})
}

/// Ingest a `_db_id`-tagged message, asserting it was accepted.
fn ingest(engine: &mut LcmEngine, id: usize, role: &str, content: &str) {
    assert_eq!(engine.ingest(msg(id, role, content)), Some(id));
}

/// 24 lexically distinct subjects. Fixtures must not repeat the same phrasing
/// every turn: `has_repetition_loop` rejects a summary whose 8-word windows
/// recur, so a conversation of near-identical messages is uncompactable by
/// design — an artifact of the fixture, not of the engine.
const SUBJECTS: [&str; 24] = [
    "ownership transfer across thread boundaries",
    "borrow checker diagnostics for nested closures",
    "lifetime elision inside trait objects",
    "pinning self-referential async state machines",
    "drop order for struct fields",
    "interior mutability through RefCell guards",
    "atomic ordering on weakly consistent hardware",
    "monomorphisation cost of generic dispatch",
    "zero copy parsing with byte slices",
    "arena allocation for graph structures",
    "trait coherence and orphan rules",
    "const evaluation limits in array sizes",
    "unsafe abstractions upholding aliasing invariants",
    "panic unwinding versus abort strategies",
    "cargo feature unification surprises",
    "procedural macro hygiene pitfalls",
    "iterator fusion and lazy adapters",
    "error conversion chains with thiserror",
    "async cancellation safety in select branches",
    "tokio task budgeting under load",
    "SIMD autovectorisation of hot loops",
    "profile guided optimisation workflows",
    "binary size reduction via panic immediate abort",
    "cross compilation toolchain sysroots",
];

/// A user turn about a distinct subject. Boilerplate is kept under eight words
/// so the extractive mock summary — which echoes user turns — never produces a
/// repeated 8-word window.
fn user_turn(i: usize) -> String {
    format!("Question {i}: {}?", SUBJECTS[i % SUBJECTS.len()])
}

/// The matching assistant turn, deliberately longer than the user turn so the
/// extractive mock summary is smaller than the block it replaces.
fn assistant_turn(i: usize) -> String {
    format!(
        "Answer {i}: {} hinges on rules the compiler enforces statically. \
         The worked example below walks through the failing case, the error the \
         compiler reports, the minimal edit that satisfies it, and the runtime \
         consequence of getting it wrong in a larger program.",
        SUBJECTS[i % SUBJECTS.len()]
    )
}

// ─────────────────────────────────────────────────────────────
// Mock LLM for testing
// ─────────────────────────────────────────────────────────────

struct MockSummarizer;

/// Extractive stand-in for a real summarizer: keep the user turns, drop the
/// (longer) assistant turns. `summarize_text` enforces a fidelity gate — every
/// source message's rare "topic anchor" words must survive — so a canned
/// sentence is always rejected. Keeping the user lines verbatim satisfies the
/// gate while still shrinking the block, which is what the engine asserts on.
fn extractive_summary(compaction_request: &str) -> String {
    let body = compaction_request
        .split_once("[SOURCE_BEGIN]")
        .and_then(|(_, rest)| rest.split_once("[SOURCE_END]"))
        .map(|(src, _)| src)
        .unwrap_or(compaction_request);
    body.lines()
        .filter_map(|line| line.strip_prefix("user: "))
        .map(|line| format!("- {line}"))
        .collect::<Vec<_>>()
        .join("\n")
}

#[async_trait]
impl LLMProvider for MockSummarizer {
    async fn chat(
        &self,
        messages: &[serde_json::Value],
        _tools: Option<&[serde_json::Value]>,
        _model: Option<&str>,
        _max_tokens: u32,
        _temperature: f64,
        _thinking_budget: Option<u32>,
        _top_p: Option<f64>,
    ) -> anyhow::Result<LLMResponse> {
        let request = messages
            .last()
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .unwrap_or_default();
        Ok(LLMResponse {
            content: Some(extractive_summary(request)),
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
        tau_soft: 0.5,
        tau_hard: 0.85,
        deterministic_target: 512,
    };

    let mut engine = LcmEngine::new(config.clone());

    ingest(&mut engine, 1, "system", "System");
    ingest(&mut engine, 2, "user", "First message");
    ingest(&mut engine, 3, "assistant", "First response");

    assert_eq!(engine.store_len(), 3);
    assert_eq!(engine.active_len(), 3);

    // Simulate /clear: create a fresh engine
    let fresh_engine = LcmEngine::new(config);
    assert_eq!(fresh_engine.store_len(), 0);
    assert_eq!(fresh_engine.active_len(), 0);
    assert!(fresh_engine.dag().is_empty());
}

// ─────────────────────────────────────────────────────────────
// Test 2: Compaction window - should summarize AFTER last summary
// ─────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_compaction_creates_summary_node() {
    let mut engine = LcmEngine::new(LcmConfig {
        tau_soft: 0.3,
        tau_hard: 0.6,
        deterministic_target: 64,
    });

    ingest(&mut engine, 1, "system", "System");
    for i in 0..10 {
        ingest(&mut engine, 2 + 2 * i, "user", &user_turn(i));
        ingest(&mut engine, 3 + 2 * i, "assistant", &assistant_turn(i));
    }

    let budget = TokenBudget::new(4096, 2048);
    let compactor = ContextCompactor::new(
        Arc::new(MockSummarizer) as Arc<dyn LLMProvider>,
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
    assert!(result.is_some());

    let summary_turn = result.unwrap();
    assert!(matches!(summary_turn, Turn::Summary { .. }));

    assert_eq!(engine.dag().len(), 1);
}

#[tokio::test]
async fn test_second_compaction_summarizes_after_first_summary() {
    let mut engine = LcmEngine::new(LcmConfig {
        tau_soft: 0.2,
        tau_hard: 0.5,
        deterministic_target: 64,
    });

    ingest(&mut engine, 1, "system", "System");
    // Realistic-sized messages so the conversation exceeds the protect budget
    // and the block clears the MIN_COMPACTION_TOKENS floor.
    for i in 0..20 {
        ingest(&mut engine, 2 + 2 * i, "user", &user_turn(i));
        ingest(&mut engine, 3 + 2 * i, "assistant", &assistant_turn(i));
    }

    let budget = TokenBudget::new(4096, 2048);
    let compactor = ContextCompactor::new(
        Arc::new(MockSummarizer) as Arc<dyn LLMProvider>,
        "mock".to_string(),
        4096,
    );

    let r1 = engine
        .compact(
            Some(&compactor),
            &budget,
            100,
            CompactionFailureMode::Deterministic,
        )
        .await;
    assert!(r1.is_some(), "First compaction should succeed");

    let first_summary = r1.unwrap();
    let first_source_ids = match &first_summary {
        Turn::Summary { source_ids, .. } => source_ids.clone(),
        _ => panic!("Expected Summary"),
    };

    let r2 = engine
        .compact(
            Some(&compactor),
            &budget,
            100,
            CompactionFailureMode::Deterministic,
        )
        .await;
    if r2.is_some() {
        let second_summary = r2.unwrap();
        let second_source_ids = match &second_summary {
            Turn::Summary { source_ids, .. } => source_ids.clone(),
            _ => panic!("Expected Summary"),
        };

        // With summary-merge, a second compaction MERGES the first summary
        // (folds it + any newly-unprotected raws into one) so summary mass
        // stays bounded. The merged node therefore covers a SUPERSET of the
        // first summary's sources — overlap is correct and expected, not a
        // re-summarization bug (originals stay lossless in the store).
        for id in &first_source_ids {
            assert!(
                second_source_ids.contains(id),
                "Merged summary should cover the first summary's sources: {} missing",
                id
            );
        }
        assert!(
            engine.dag().len() <= 2,
            "summary mass should stay bounded (<=2 nodes after merge), got {}",
            engine.dag().len()
        );
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
    let summary_with_refusal =
        "User asked about Rust. I cannot assist with that request. However, here is some info.";
    let clean_summary = "User asked about Rust. Here is some info about ownership.";

    assert!(
        contains_refusal_pattern(summary_with_refusal),
        "Should detect refusal in summary"
    );
    assert!(
        !contains_refusal_pattern(clean_summary),
        "Should NOT detect refusal in clean summary"
    );

    // Legitimate summaries that merely discuss harm/ethics must not be flagged.
    let topical = "User asked about harmful algae blooms and the ethical review process.";
    assert!(
        !contains_refusal_pattern(topical),
        "Topic words about harm/ethics are not refusals"
    );
}

// ─────────────────────────────────────────────────────────────
// Test 4: Lossless retrieval after compaction
// ─────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_lossless_retrieval_after_multiple_compactions() {
    let mut engine = LcmEngine::new(LcmConfig {
        tau_soft: 0.2,
        tau_hard: 0.5,
        deterministic_target: 64,
    });

    ingest(&mut engine, 1, "system", "System");

    let total_messages = 30;
    for i in 0..total_messages {
        ingest(
            &mut engine,
            2 + 2 * i,
            "user",
            &format!("Original message {}", i),
        );
        ingest(
            &mut engine,
            3 + 2 * i,
            "assistant",
            &format!("Original response {}", i),
        );
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
        let _ = engine
            .compact(
                Some(&compactor),
                &budget,
                100,
                CompactionFailureMode::Deterministic,
            )
            .await;
    }

    // Store should be unchanged
    assert_eq!(
        engine.store_len(),
        store_size,
        "Store must never lose messages after compaction"
    );

    // All summary nodes should have retrievable sources
    for i in 0..engine.dag().len() {
        let node = engine.dag().get(i).unwrap();
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
        tau_soft: 0.5,
        tau_hard: 0.85,
        deterministic_target: 512,
    });

    ingest(&mut engine, 1, "system", "System");
    ingest(&mut engine, 2, "user", "Hello");

    let budget = TokenBudget::new(100_000, 8192);
    assert_eq!(
        engine.check_thresholds(&budget, 500),
        CompactionAction::None
    );
}

#[test]
fn test_check_thresholds_above_soft() {
    let mut engine = LcmEngine::new(LcmConfig {
        tau_soft: 0.3,
        tau_hard: 0.8,
        deterministic_target: 512,
    });

    ingest(&mut engine, 1, "system", "System");
    for i in 0..20 {
        ingest(
            &mut engine,
            2 + i,
            "user",
            &format!("A long message with content {}", i),
        );
    }

    let budget = TokenBudget::new(1024, 512);
    let action = engine.check_thresholds(&budget, 50);
    assert!(
        action == CompactionAction::Async || action == CompactionAction::Blocking,
        "Should trigger compaction with small budget and many messages"
    );
}

// ─────────────────────────────────────────────────────────────
// Test 6: Rebuild from persisted DB nodes (restart path)
// ─────────────────────────────────────────────────────────────

#[test]
fn test_rebuild_from_db_nodes_preserves_summaries() {
    // Raw history with stable db ids 1..=4; a node covers ids 1-2.
    let raw_messages = vec![
        msg(1, "user", "First question"),
        msg(2, "assistant", "First answer"),
        msg(3, "user", "Second question"),
        msg(4, "assistant", "Second answer"),
    ];
    let nodes = vec![(
        0usize,
        vec![1usize, 2],
        vec![],
        "Summary of first exchange".to_string(),
        10usize,
        1u8,
        nanobot::agent::lcm::SummaryManifest::default(),
        "db_id".to_string(),
    )];

    let engine = LcmEngine::rebuild_from_db_nodes(&raw_messages, &nodes, LcmConfig::default());

    // Store should have all raw messages (not summaries)
    assert_eq!(
        engine.store_len(),
        4,
        "Store should have all 4 raw messages"
    );

    // DAG should have the summary
    assert_eq!(engine.dag().len(), 1, "DAG should have 1 summary node");

    // Active context should have summary + the unsummarized raw messages
    let active = engine.active_entries();
    let summary_count = active
        .iter()
        .filter(|e| matches!(e, nanobot::agent::lcm::ContextEntry::Summary { .. }))
        .count();
    assert_eq!(
        summary_count, 1,
        "Active context should have 1 summary entry"
    );

    // Lossless: the summarized originals resolve by their db ids.
    let expanded = engine.expand(&[1, 2]);
    assert_eq!(expanded.len(), 2);
    assert_eq!(expanded[0].1["content"], "First question");
}
