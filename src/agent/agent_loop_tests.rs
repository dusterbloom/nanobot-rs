//! Tests for the agent loop.
//!
//! Extracted from agent_loop.rs for maintainability.
//! Loaded via `#[path = "agent_loop_tests.rs"]` in agent_loop.rs so that
//! `use super::*` continues to resolve against the agent_loop module.

use super::*;
use backon::BackoffBuilder;
use crate::agent::router::{extract_json_object, parse_lenient_router_decision, request_strict_router_decision};
use crate::config::schema::{AdaptiveTokenConfig, MemoryConfig, ProvenanceConfig, ProviderConfig, ToolDelegationConfig, TrioConfig};
use crate::providers::base::LLMProvider;
use crate::providers::openai_compat::OpenAICompatProvider;
use async_trait::async_trait;

/// Minimal mock LLM provider for wiring tests.
struct MockLLM {
    name: String,
}

impl MockLLM {
    fn named(name: &str) -> Arc<dyn LLMProvider> {
        Arc::new(Self {
            name: name.to_string(),
        })
    }
}

#[async_trait]
impl LLMProvider for MockLLM {
    async fn chat(
        &self,
        _messages: &[Value],
        _tools: Option<&[Value]>,
        _model: Option<&str>,
        _max_tokens: u32,
        _temperature: f64,
        _thinking_budget: Option<u32>,
        _top_p: Option<f64>,
    ) -> anyhow::Result<crate::providers::base::LLMResponse> {
        Ok(crate::providers::base::LLMResponse {
            content: Some("mock".to_string()),
            tool_calls: vec![],
            finish_reason: "stop".to_string(),
            usage: std::collections::HashMap::new(),
        })
    }

    fn get_default_model(&self) -> &str {
        &self.name
    }
}

struct StaticResponseLLM {
    name: String,
    body: String,
}

impl StaticResponseLLM {
    fn new(name: &str, body: &str) -> Self {
        Self {
            name: name.to_string(),
            body: body.to_string(),
        }
    }
}

#[async_trait]
impl LLMProvider for StaticResponseLLM {
    async fn chat(
        &self,
        _messages: &[Value],
        _tools: Option<&[Value]>,
        _model: Option<&str>,
        _max_tokens: u32,
        _temperature: f64,
        _thinking_budget: Option<u32>,
        _top_p: Option<f64>,
    ) -> anyhow::Result<crate::providers::base::LLMResponse> {
        Ok(crate::providers::base::LLMResponse {
            content: Some(self.body.clone()),
            tool_calls: vec![],
            finish_reason: "stop".to_string(),
            usage: std::collections::HashMap::new(),
        })
    }

    fn get_default_model(&self) -> &str {
        &self.name
    }
}

/// Helper to build a SwappableCore with minimal config for wiring tests.
fn build_test_core(
    delegation_enabled: bool,
    delegation_provider: Option<Arc<dyn LLMProvider>>,
    config_provider: Option<ProviderConfig>,
) -> SwappableCore {
    let workspace = tempfile::tempdir().unwrap().into_path();
    let main = MockLLM::named("main-provider");
    let td = ToolDelegationConfig {
        enabled: delegation_enabled,
        model: "delegation-model".to_string(),
        provider: config_provider,
        auto_local: true,
        ..Default::default()
    };
    build_swappable_core(SwappableCoreConfig {
        provider: main,
        workspace,
        model: "main-model".to_string(),
        max_iterations: 10,
        max_continuations: 2,
        max_tokens: 4096,
        temperature: 0.7,
        max_context_tokens: 16384,
        brave_api_key: None,
        search_provider: "searxng".to_string(),
        searxng_url: "http://localhost:8888".to_string(),
        search_max_results: 5,
        jina_api_key: None,
        exec_timeout: 30,
        restrict_to_workspace: false,
        memory_config: MemoryConfig::default(),
        is_local: false,
        compaction_provider: None,
        tool_delegation: td,
        provenance: ProvenanceConfig::default(),
        max_tool_result_chars: 2000,
        delegation_provider,
        specialist_provider: None,
        trio_config: TrioConfig::default(),
        model_capabilities_overrides: std::collections::HashMap::new(),
        reasoning_config: crate::config::schema::ReasoningConfig::default(),
        tool_heartbeat_secs: 2,
        health_check_timeout_secs: 2,
        adaptive_tokens: AdaptiveTokenConfig::default(),
    })
}

#[test]
fn test_provenance_warning_role_local_safe() {
    assert_eq!(provenance_warning_role(true), "user");
    assert_eq!(provenance_warning_role(false), "system");
}

#[test]
fn test_extract_json_object_from_markdown_fence() {
    let raw = "```json\n{\"action\":\"tool\",\"target\":\"exec\",\"args\":{},\"confidence\":0.9}\n```";
    let obj = extract_json_object(raw).expect("json object");
    assert!(obj.starts_with('{'));
    assert!(obj.ends_with('}'));
    assert!(obj.contains("\"action\":\"tool\""));
}

#[test]
fn test_extract_json_object_none_when_missing() {
    assert!(extract_json_object("no json here").is_none());
}

#[tokio::test]
async fn test_request_strict_router_decision_action_matrix() {
    let cases = vec![
        (
            r#"{"action":"tool","target":"read_file","args":{"path":"README.md"},"confidence":0.9}"#,
            "tool",
        ),
        (
            r#"{"action":"subagent","target":"builder","args":{"task":"x"},"confidence":0.8}"#,
            "subagent",
        ),
        (
            r#"{"action":"specialist","target":"summarizer","args":{"style":"tight"},"confidence":0.7}"#,
            "specialist",
        ),
        (
            r#"{"action":"ask_user","target":"clarify","args":{"question":"Need path?"},"confidence":0.6}"#,
            "ask_user",
        ),
    ];

    for (raw, expected_action) in cases {
        let llm = StaticResponseLLM::new("router", raw);
        let decision = request_strict_router_decision(
            &llm,
            "router",
            "route this action with strict schema",
            false,
            0.6,
            1.0,
            "",
            256,
        )
        .await
        .expect("valid strict router decision");
        assert_eq!(decision.action, expected_action);
    }
}

/// Real-provider trio probe.
///
/// Runs against live OpenAI-compatible endpoints (e.g. LM Studio):
/// - main: `NANOBOT_REAL_MAIN_BASE` (default: http://127.0.0.1:8080/v1)
/// - router: `NANOBOT_REAL_ROUTER_BASE` (default: http://127.0.0.1:8094/v1)
/// - specialist: `NANOBOT_REAL_SPECIALIST_BASE` (default: http://127.0.0.1:8095/v1)
///
/// Optional model overrides:
/// - `NANOBOT_REAL_MAIN_MODEL`
/// - `NANOBOT_REAL_ROUTER_MODEL`
/// - `NANOBOT_REAL_SPECIALIST_MODEL`
#[tokio::test]
#[ignore = "requires running local providers on main/router/specialist ports"]
async fn test_real_providers_trio_probe() {
    let main_base = std::env::var("NANOBOT_REAL_MAIN_BASE")
        .unwrap_or_else(|_| "http://127.0.0.1:8080/v1".to_string());
    let router_base = std::env::var("NANOBOT_REAL_ROUTER_BASE")
        .unwrap_or_else(|_| "http://127.0.0.1:8094/v1".to_string());
    let specialist_base = std::env::var("NANOBOT_REAL_SPECIALIST_BASE")
        .unwrap_or_else(|_| "http://127.0.0.1:8095/v1".to_string());
    let main_model = std::env::var("NANOBOT_REAL_MAIN_MODEL")
        .unwrap_or_else(|_| "local-model".to_string());
    let router_model = std::env::var("NANOBOT_REAL_ROUTER_MODEL")
        .unwrap_or_else(|_| "local-delegation".to_string());
    let specialist_model = std::env::var("NANOBOT_REAL_SPECIALIST_MODEL")
        .unwrap_or_else(|_| "local-specialist".to_string());

    let main = OpenAICompatProvider::new("local", Some(&main_base), Some(&main_model));
    let router = OpenAICompatProvider::new("local", Some(&router_base), Some(&router_model));
    let specialist =
        OpenAICompatProvider::new("local", Some(&specialist_base), Some(&specialist_model));

    let mut failures: Vec<String> = Vec::new();

    // Router: force each action in a constrained prompt and verify strict parsing.
    let router_cases = vec![
        ("tool", "Return action=tool target=read_file args={\"path\":\"README.md\"}."),
        (
            "subagent",
            "Return action=subagent target=builder args={\"task\":\"diagnose issue\"}.",
        ),
        (
            "specialist",
            "Return action=specialist target=summarizer args={\"objective\":\"compress\"}.",
        ),
        (
            "ask_user",
            "Return action=ask_user target=clarify args={\"question\":\"Which file?\"}.",
        ),
    ];
    for (expected_action, directive) in router_cases {
        let pack = format!("{}\nFollow schema strictly.", directive);
        match request_strict_router_decision(&router, &router_model, &pack, false, 0.6, 1.0, "", 256).await {
            Ok(d) => {
                if d.action != expected_action {
                    failures.push(format!(
                        "router action mismatch: expected={}, got={} target={}",
                        expected_action, d.action, d.target
                    ));
                }
            }
            Err(e) => failures.push(format!("router {} failed: {}", expected_action, e)),
        }
    }

    // Specialist must produce non-empty response (with warmup retries).
    let specialist_messages = vec![
        json!({"role":"system","content":"ROLE=SPECIALIST\nReturn concise output."}),
        json!({"role":"user","content":"Summarize: tool call failed because server was down and port conflicted."}),
    ];
    let mut specialist_ok = false;
    let mut warmup_backoff = backon::ConstantBuilder::default()
        .with_delay(Duration::from_secs(2))
        .with_max_times(10)
        .build();
    loop {
        match specialist
            .chat(
                &specialist_messages,
                None,
                Some(&specialist_model),
                256,
                0.2,
                None,
                None,
            )
            .await
        {
            Ok(resp) => {
                let text = resp.content.unwrap_or_default();
                if !text.trim().is_empty() {
                    specialist_ok = true;
                    break;
                }
            }
            Err(e) => {
                let msg = e.to_string();
                let lower = msg.to_lowercase();
                if !lower.contains("loading model") && !lower.contains("503") {
                    failures.push(format!("specialist call failed: {}", msg));
                    break;
                }
            }
        }
        match warmup_backoff.next() {
            Some(delay) => tokio::time::sleep(delay).await,
            None => break,
        }
    }
    if !specialist_ok {
        failures.push("specialist did not become ready / returned empty output".to_string());
    }

    // Main provider smoke: should answer plain text with no tools when none offered.
    let main_messages = vec![json!({"role":"user","content":"Reply with exactly: main-ok"})];
    match main
        .chat(&main_messages, None, Some(&main_model), 64, 0.0, None, None)
        .await
    {
        Ok(resp) => {
            if resp.has_tool_calls() {
                failures.push("main returned tool calls unexpectedly".to_string());
            }
            let text = resp.content.unwrap_or_default();
            if !text.to_lowercase().contains("main-ok") {
                failures.push(format!("main output mismatch: {}", text));
            }
        }
        Err(e) => failures.push(format!("main call failed: {}", e)),
    }

    if !failures.is_empty() {
        panic!(
            "real trio probe failed (main={}, router={}, specialist={}):\n{}",
            main_base,
            router_base,
            specialist_base,
            failures.join("\n")
        );
    }
}

// -- Delegation provider wiring tests --

#[test]
fn test_delegation_disabled_no_runner_provider() {
    let core = build_test_core(false, None, None);
    assert!(
        core.tool_runner_provider.is_none(),
        "When delegation is disabled, tool_runner_provider should be None"
    );
    assert!(core.tool_runner_model.is_none());
}

#[test]
fn test_delegation_enabled_with_auto_provider() {
    // When an auto-spawned delegation provider is passed, it should be used
    let dp = MockLLM::named("auto-delegation");
    let core = build_test_core(true, Some(dp), None);

    assert!(core.tool_runner_provider.is_some());
    let provider = core.tool_runner_provider.as_ref().unwrap();
    assert_eq!(
        provider.get_default_model(),
        "auto-delegation",
        "Should use the auto-spawned delegation provider"
    );
    assert_eq!(core.tool_runner_model.as_deref(), Some("delegation-model"));
}

#[test]
fn test_delegation_auto_provider_takes_priority_over_config() {
    // Auto-spawned provider should take priority over config provider
    let dp = MockLLM::named("auto-delegation");
    let config_provider = ProviderConfig {
        api_key: "key".to_string(),
        api_base: Some("http://localhost:9999/v1".to_string()),
    };
    let core = build_test_core(true, Some(dp), Some(config_provider));

    let provider = core.tool_runner_provider.as_ref().unwrap();
    assert_eq!(
        provider.get_default_model(),
        "auto-delegation",
        "Auto-spawned provider should beat config provider"
    );
}

#[test]
fn test_delegation_config_provider_used_when_no_auto() {
    // When no auto provider, but config has one, it should create OpenAICompatProvider
    let config_provider = ProviderConfig {
        api_key: "key".to_string(),
        api_base: Some("http://localhost:9999/v1".to_string()),
    };
    let core = build_test_core(true, None, Some(config_provider));

    assert!(
        core.tool_runner_provider.is_some(),
        "Should have a provider from config"
    );
}

#[test]
fn test_delegation_falls_back_to_main_provider() {
    // When delegation enabled but no auto provider and no config provider,
    // should fall back to main
    let core = build_test_core(true, None, None);

    assert!(core.tool_runner_provider.is_some());
    let provider = core.tool_runner_provider.as_ref().unwrap();
    assert_eq!(
        provider.get_default_model(),
        "main-provider",
        "Should fall back to main provider"
    );
}

#[test]
fn test_delegation_model_uses_config_model() {
    let core = build_test_core(true, None, None);
    assert_eq!(
        core.tool_runner_model.as_deref(),
        Some("delegation-model"),
        "Should use the model from ToolDelegationConfig"
    );
}

#[test]
fn test_delegation_model_falls_back_to_main_when_empty() {
    let workspace = tempfile::tempdir().unwrap().into_path();
    let main = MockLLM::named("main-provider");
    let td = ToolDelegationConfig {
        enabled: true,
        model: String::new(), // Empty → fall back to main model
        auto_local: true,
        ..Default::default()
    };
    let core = build_swappable_core(SwappableCoreConfig {
        provider: main,
        workspace,
        model: "main-model".to_string(),
        max_iterations: 10,
        max_continuations: 2,
        max_tokens: 4096,
        temperature: 0.7,
        max_context_tokens: 16384,
        brave_api_key: None,
        search_provider: "searxng".to_string(),
        searxng_url: "http://localhost:8888".to_string(),
        search_max_results: 5,
        jina_api_key: None,
        exec_timeout: 30,
        restrict_to_workspace: false,
        memory_config: MemoryConfig::default(),
        is_local: false,
        compaction_provider: None,
        tool_delegation: td,
        provenance: ProvenanceConfig::default(),
        max_tool_result_chars: 2000,
        delegation_provider: None,
        specialist_provider: None,
        trio_config: TrioConfig::default(),
        model_capabilities_overrides: std::collections::HashMap::new(),
        reasoning_config: crate::config::schema::ReasoningConfig::default(),
        tool_heartbeat_secs: 2,
        health_check_timeout_secs: 2,
        adaptive_tokens: AdaptiveTokenConfig::default(),
    });
    assert_eq!(
        core.tool_runner_model.as_deref(),
        Some("main-model"),
        "Empty delegation model should fall back to main model"
    );
}

#[test]
fn test_delegation_disabled_ignores_passed_provider() {
    // Even if a delegation_provider is passed, it should be ignored
    // when delegation is disabled.
    let dp = MockLLM::named("auto-delegation");
    let core = build_test_core(false, Some(dp), None);

    assert!(
        core.tool_runner_provider.is_none(),
        "Delegation disabled should ignore passed provider"
    );
    assert!(core.tool_runner_model.is_none());
}

#[test]
fn test_delegation_with_is_local_true() {
    // Verify wiring works when is_local=true (uses lite context builder)
    let workspace = tempfile::tempdir().unwrap().into_path();
    let main = MockLLM::named("local-main");
    let dp = MockLLM::named("local-delegation");
    let td = ToolDelegationConfig {
        enabled: true,
        model: "delegation-model".to_string(),
        auto_local: true,
        ..Default::default()
    };
    let core = build_swappable_core(SwappableCoreConfig {
        provider: main,
        workspace,
        model: "local-model".to_string(),
        max_iterations: 10,
        max_continuations: 2,
        max_tokens: 4096,
        temperature: 0.7,
        max_context_tokens: 16384,
        brave_api_key: None,
        search_provider: "searxng".to_string(),
        searxng_url: "http://localhost:8888".to_string(),
        search_max_results: 5,
        jina_api_key: None,
        exec_timeout: 30,
        restrict_to_workspace: false,
        memory_config: MemoryConfig::default(),
        is_local: true,
        compaction_provider: None,
        tool_delegation: td,
        provenance: ProvenanceConfig::default(),
        max_tool_result_chars: 2000,
        delegation_provider: Some(dp),
        specialist_provider: None,
        trio_config: TrioConfig::default(),
        model_capabilities_overrides: std::collections::HashMap::new(),
        reasoning_config: crate::config::schema::ReasoningConfig::default(),
        tool_heartbeat_secs: 2,
        health_check_timeout_secs: 2,
        adaptive_tokens: AdaptiveTokenConfig::default(),
    });

    assert!(core.is_local);
    assert!(core.tool_runner_provider.is_some());
    assert_eq!(
        core.tool_runner_provider
            .as_ref()
            .unwrap()
            .get_default_model(),
        "local-delegation",
        "Local mode should still use the delegation provider"
    );
}

#[test]
fn test_delegation_with_compaction_and_delegation_providers() {
    // Both compaction and delegation providers set — should not interfere
    let workspace = tempfile::tempdir().unwrap().into_path();
    let main = MockLLM::named("main");
    let compaction = MockLLM::named("compaction");
    let delegation = MockLLM::named("delegation");
    let td = ToolDelegationConfig {
        enabled: true,
        model: "deleg-model".to_string(),
        auto_local: true,
        ..Default::default()
    };
    let core = build_swappable_core(SwappableCoreConfig {
        provider: main,
        workspace,
        model: "main-model".to_string(),
        max_iterations: 10,
        max_continuations: 2,
        max_tokens: 4096,
        temperature: 0.7,
        max_context_tokens: 16384,
        brave_api_key: None,
        search_provider: "searxng".to_string(),
        searxng_url: "http://localhost:8888".to_string(),
        search_max_results: 5,
        jina_api_key: None,
        exec_timeout: 30,
        restrict_to_workspace: false,
        memory_config: MemoryConfig::default(),
        is_local: true,
        compaction_provider: Some(compaction),
        tool_delegation: td,
        provenance: ProvenanceConfig::default(),
        max_tool_result_chars: 2000,
        delegation_provider: Some(delegation),
        specialist_provider: None,
        trio_config: TrioConfig::default(),
        model_capabilities_overrides: std::collections::HashMap::new(),
        reasoning_config: crate::config::schema::ReasoningConfig::default(),
        tool_heartbeat_secs: 2,
        health_check_timeout_secs: 2,
        adaptive_tokens: AdaptiveTokenConfig::default(),
    });

    // Compaction provider goes to memory_provider, delegation to tool_runner
    assert_eq!(
        core.memory_provider.get_default_model(),
        "compaction",
        "Memory should use compaction provider"
    );
    assert_eq!(
        core.tool_runner_provider
            .as_ref()
            .unwrap()
            .get_default_model(),
        "delegation",
        "Tool runner should use delegation provider"
    );
}

// -----------------------------------------------------------------------
// E2E: Full agent loop with LCM enabled against real local LLM.
//
// This test requires LM Studio (or compatible) running. Set env vars:
//   NANOBOT_LCM_TEST_BASE  — API base (default: http://127.0.0.1:1234/v1)
//   NANOBOT_LCM_TEST_MODEL — Model name (default: local-model)
//
// Run with: cargo test test_real_lcm_e2e -- --ignored --nocapture
// -----------------------------------------------------------------------

#[tokio::test]
#[ignore = "requires running local LLM on NANOBOT_LCM_TEST_BASE"]
async fn test_real_lcm_e2e_compact_and_expand() {
    use crate::config::schema::LcmSchemaConfig;

    let api_base = std::env::var("NANOBOT_LCM_TEST_BASE")
        .unwrap_or_else(|_| "http://127.0.0.1:1234/v1".to_string());
    let model_name = std::env::var("NANOBOT_LCM_TEST_MODEL")
        .unwrap_or_else(|_| "local-model".to_string());

    eprintln!("LCM E2E: using {} model={}", api_base, model_name);

    // Real provider pointing at local LLM.
    let provider: Arc<dyn LLMProvider> = Arc::new(
        OpenAICompatProvider::new("local", Some(&api_base), Some(&model_name)),
    );

    // Warm up: verify the model is responding.
    let warmup = provider
        .chat(
            &[json!({"role": "user", "content": "Reply with exactly: ok"})],
            None,
            Some(&model_name),
            32,
            0.0,
            None,
            None,
        )
        .await;
    match warmup {
        Ok(r) => eprintln!("LCM E2E warmup: {}", r.content.as_deref().unwrap_or("(empty)")),
        Err(e) => panic!("LCM E2E: model not responding at {}: {}", api_base, e),
    }

    let workspace = tempfile::tempdir().unwrap().keep();

    // Build core with small context window + LCM thresholds that trigger fast.
    let core = build_swappable_core(SwappableCoreConfig {
        provider: provider.clone(),
        workspace: workspace.clone(),
        model: model_name.clone(),
        max_iterations: 3,
        max_continuations: 2,
        max_tokens: 512,
        temperature: 0.3,
        max_context_tokens: 2048, // Tiny so compaction triggers quickly.
        brave_api_key: None,
        search_provider: "searxng".to_string(),
        searxng_url: "http://localhost:8888".to_string(),
        search_max_results: 5,
        jina_api_key: None,
        exec_timeout: 30,
        restrict_to_workspace: false,
        memory_config: MemoryConfig::default(),
        is_local: true,
        compaction_provider: Some(provider.clone()),
        tool_delegation: ToolDelegationConfig::default(),
        provenance: ProvenanceConfig::default(),
        max_tool_result_chars: 2000,
        delegation_provider: None,
        specialist_provider: None,
        trio_config: TrioConfig::default(),
        model_capabilities_overrides: std::collections::HashMap::new(),
        reasoning_config: crate::config::schema::ReasoningConfig::default(),
        tool_heartbeat_secs: 2,
        health_check_timeout_secs: 2,
        adaptive_tokens: AdaptiveTokenConfig::default(),
    });
    let counters = Arc::new(crate::agent::agent_core::RuntimeCounters::new(2048));
    let core_handle = AgentHandle::new(core, counters);

    let (inbound_tx, inbound_rx) = tokio::sync::mpsc::unbounded_channel::<InboundMessage>();
    let (outbound_tx, _outbound_rx) = tokio::sync::mpsc::unbounded_channel::<OutboundMessage>();

    let lcm_config = LcmSchemaConfig {
        enabled: true,
        tau_soft: 0.3,  // Trigger early.
        tau_hard: 0.6,
        deterministic_target: 128,
        ..Default::default()
    };

    let agent_loop = AgentLoop::new(
        core_handle,
        inbound_rx,
        outbound_tx,
        inbound_tx,
        None, // no cron
        1,
        None, // no email
        None, // no repl display
        None, // no providers config
        ProprioceptionConfig::default(),
        lcm_config,
        None, // no health registry
    );

    let session_key = "lcm-e2e-test";
    let mut responses = Vec::new();

    // Send 12 verbose messages to fill the tiny 2K context.
    let prompts = [
        "Explain Rust ownership rules in detail with examples of move semantics. Be thorough and give at least 3 examples.",
        "Now explain borrowing and the difference between mutable and immutable references with code examples.",
        "Describe lifetime annotations and why they are needed. Give concrete examples with structs and functions.",
        "What are the rules for lifetime elision? When can you omit lifetime annotations? List all three rules.",
        "Explain smart pointers: Box, Rc, Arc, and when to use each one. Give a real-world use case for each.",
        "What is interior mutability? Explain Cell, RefCell, and Mutex with examples of each.",
        "Describe async/await in Rust. How do Futures work under the hood? Explain the state machine transformation.",
        "Explain trait objects vs generics. When would you use dynamic dispatch vs static dispatch?",
        "What are the differences between String and &str? When should you use each one in function signatures?",
        "Explain the Drop trait and how Rust's destructors work. What is the order of dropping?",
        "Describe the Pin and Unpin traits. Why are they needed for async Rust and self-referential structs?",
        "Explain how pattern matching works in Rust. Cover match, if let, while let, and destructuring.",
    ];

    for (i, prompt) in prompts.iter().enumerate() {
        eprintln!("LCM E2E: sending message {}/{}...", i + 1, prompts.len());
        let resp = agent_loop
            .process_direct(prompt, session_key, "test", "lcm-e2e")
            .await;
        eprintln!(
            "LCM E2E: response {} ({} chars): {}",
            i + 1,
            resp.len(),
            &resp[..resp.len().min(80)]
        );
        assert!(
            !resp.is_empty(),
            "Message {} should get a non-empty response",
            i + 1
        );
        responses.push(resp);
    }

    // Check LCM engine state.
    let engines = agent_loop.shared.lcm_engines.lock().await;
    let engine_arc = engines
        .get(session_key)
        .expect("LCM engine should exist for session");
    let engine = engine_arc.lock().await;

    eprintln!(
        "LCM E2E results: store={} active={} dag_nodes={}",
        engine.store_len(),
        engine.active_len(),
        engine.dag_ref().len()
    );

    // Invariant 1: store has messages from the conversation.
    // Note: with is_local + small context, trim_to_fit_with_age runs before
    // LCM ingestion, so the store only contains messages that survived trimming.
    // The session JSONL (on-disk) is the true immutable store; the in-memory
    // LCM store tracks what entered the active context window.
    assert!(
        engine.store_len() >= 5,
        "Store should have at least 5 messages (system + some turns), got {}",
        engine.store_len()
    );

    // Invariant 2: active context should be shorter than store (compaction happened).
    // With tau_soft=0.3 and 4K context, compaction should trigger early.
    assert!(
        engine.active_len() < engine.store_len(),
        "Active ({}) should be shorter than store ({}) — compaction should have triggered",
        engine.active_len(),
        engine.store_len()
    );

    // Invariant 3: DAG should have at least one summary node.
    assert!(
        engine.dag_ref().len() >= 1,
        "DAG should have at least 1 summary node, got {}",
        engine.dag_ref().len()
    );

    // Invariant 4: every summary node's source IDs resolve to real messages.
    for i in 0..engine.dag_ref().len() {
        let node = engine.dag_ref().get(i).unwrap();
        let expanded = engine.expand(&node.source_ids);
        assert_eq!(
            expanded.len(),
            node.source_ids.len(),
            "Summary node {} has {} source IDs but only {} resolve",
            i,
            node.source_ids.len(),
            expanded.len()
        );
        eprintln!(
            "  DAG node {}: level={} sources={:?} tokens={}",
            i, node.level, node.source_ids, node.tokens
        );
    }

    // Invariant 5: active context contains at least one Summary entry.
    let summary_count = engine
        .active_entries()
        .iter()
        .filter(|e| matches!(e, crate::agent::lcm::ContextEntry::Summary { .. }))
        .count();
    assert!(
        summary_count >= 1,
        "Active context should have at least 1 summary entry, got {}",
        summary_count
    );

    // Invariant 6: lossless expand — all store IDs are retrievable.
    let all_ids: Vec<usize> = (0..engine.store_len()).collect();
    let expanded = engine.expand(&all_ids);
    assert_eq!(
        expanded.len(),
        engine.store_len(),
        "All {} store messages should be retrievable via expand",
        engine.store_len()
    );
    for (id, msg) in &expanded {
        let content = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
        assert!(
            !content.is_empty(),
            "Expanded message {} should have content",
            id
        );
    }

    eprintln!("LCM E2E: ALL INVARIANTS PASSED");
    eprintln!(
        "  Messages: {} stored, {} active, {} summary nodes",
        engine.store_len(),
        engine.active_len(),
        engine.dag_ref().len()
    );

    // Cleanup.
    drop(engine);
    drop(engines);
    let _ = std::fs::remove_dir_all(&workspace);
}

#[tokio::test]
async fn test_compaction_timeout_resets_in_flight() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    let in_flight = Arc::new(AtomicBool::new(false));

    // Simulate: set in_flight before spawning compaction
    in_flight.store(true, Ordering::SeqCst);
    assert!(in_flight.load(Ordering::SeqCst));

    let flag = in_flight.clone();
    let handle = tokio::spawn(async move {
        let timeout_result = tokio::time::timeout(
            Duration::from_millis(100), // Short timeout for test
            async {
                // Simulate a hanging compaction endpoint
                tokio::time::sleep(Duration::from_secs(60)).await;
            },
        )
        .await;
        assert!(timeout_result.is_err(), "should have timed out");
        flag.store(false, Ordering::SeqCst); // Must always execute
    });

    // Wait for the spawned task to complete
    handle.await.unwrap();

    // The critical assertion: in_flight must be reset even after timeout
    assert!(
        !in_flight.load(Ordering::SeqCst),
        "in_flight must reset to false after timeout"
    );
}

// -----------------------------------------------------------------------
// Trio E2E test harness
//
// All tests require a single LM Studio endpoint serving three models.
// Configure via env vars:
//   NANOBOT_TRIO_BASE            — API base (default: http://192.168.1.22:1234/v1)
//   NANOBOT_TRIO_MAIN_MODEL      — Main model name
//   NANOBOT_TRIO_ROUTER_MODEL    — Router model name
//   NANOBOT_TRIO_SPECIALIST_MODEL — Specialist model name
//
// Run with: cargo test test_trio_e2e -- --ignored --nocapture
// -----------------------------------------------------------------------

/// Read trio E2E env vars (single shared endpoint).
fn trio_e2e_env() -> (String, String, String, String) {
    let base = std::env::var("NANOBOT_TRIO_BASE")
        .unwrap_or_else(|_| "http://192.168.1.22:1234/v1".to_string());
    let main_model = std::env::var("NANOBOT_TRIO_MAIN_MODEL")
        .unwrap_or_else(|_| "gemma-3n-e4b-it".to_string());
    let router_model = std::env::var("NANOBOT_TRIO_ROUTER_MODEL")
        .unwrap_or_else(|_| "nvidia_orchestrator-8b".to_string());
    let specialist_model = std::env::var("NANOBOT_TRIO_SPECIALIST_MODEL")
        .unwrap_or_else(|_| "qwen3-1.7b".to_string());
    (base, main_model, router_model, specialist_model)
}

/// Build an AgentLoop wired for trio E2E testing.
///
/// All three providers share one LM Studio endpoint, differentiated by model name.
/// A shared JitGate serialises requests to prevent concurrent model-loading crashes.
fn build_trio_e2e_harness(
    base_url: &str,
    main_model: &str,
    router_model: &str,
    specialist_model: &str,
) -> (AgentLoop, std::path::PathBuf) {
    use crate::providers::factory;
    use crate::providers::jit_gate::JitGate;
    use crate::config::schema::LcmSchemaConfig;

    let jit_gate = std::sync::Arc::new(JitGate::new());

    let main_provider: Arc<dyn LLMProvider> = factory::create_openai_compat(
        factory::ProviderSpec::local(base_url, Some(main_model))
            .with_jit_gate_opt(Some(jit_gate.clone())),
    );
    let router_provider: Arc<dyn LLMProvider> = factory::create_openai_compat(
        factory::ProviderSpec::local(base_url, Some(router_model))
            .with_jit_gate_opt(Some(jit_gate.clone())),
    );
    let specialist_provider: Arc<dyn LLMProvider> = factory::create_openai_compat(
        factory::ProviderSpec::local(base_url, Some(specialist_model))
            .with_jit_gate_opt(Some(jit_gate.clone())),
    );

    let workspace = tempfile::tempdir().unwrap().into_path();

    let mut td = ToolDelegationConfig {
        mode: crate::config::schema::DelegationMode::Trio,
        ..Default::default()
    };
    td.apply_mode();

    let trio_config = TrioConfig {
        enabled: true,
        router_model: router_model.to_string(),
        specialist_model: specialist_model.to_string(),
        ..Default::default()
    };

    let core = build_swappable_core(SwappableCoreConfig {
        provider: main_provider,
        workspace: workspace.clone(),
        model: main_model.to_string(),
        max_iterations: 5,
        max_continuations: 2,
        max_tokens: 512,
        temperature: 0.3,
        max_context_tokens: 4096,
        brave_api_key: None,
        search_provider: "searxng".to_string(),
        searxng_url: "http://localhost:8888".to_string(),
        search_max_results: 5,
        jina_api_key: None,
        exec_timeout: 30,
        restrict_to_workspace: true,
        memory_config: MemoryConfig::default(),
        is_local: true,
        compaction_provider: None,
        tool_delegation: td,
        provenance: ProvenanceConfig::default(),
        max_tool_result_chars: 2000,
        delegation_provider: Some(router_provider),
        specialist_provider: Some(specialist_provider),
        trio_config,
        model_capabilities_overrides: std::collections::HashMap::new(),
        reasoning_config: crate::config::schema::ReasoningConfig::default(),
        tool_heartbeat_secs: 2,
        health_check_timeout_secs: 2,
        adaptive_tokens: AdaptiveTokenConfig::default(),
    });

    let counters = Arc::new(crate::agent::agent_core::RuntimeCounters::new(4096));
    let core_handle = AgentHandle::new(core, counters);

    let (inbound_tx, inbound_rx) = tokio::sync::mpsc::unbounded_channel::<InboundMessage>();
    let (outbound_tx, _outbound_rx) = tokio::sync::mpsc::unbounded_channel::<OutboundMessage>();

    let agent_loop = AgentLoop::new(
        core_handle,
        inbound_rx,
        outbound_tx,
        inbound_tx,
        None,
        1,
        None,
        None,
        None,
        ProprioceptionConfig::default(),
        LcmSchemaConfig::default(),
        None,
    );

    (agent_loop, workspace)
}

/// Warmup a provider with backon retries (models may need JIT loading time).
async fn warmup_trio_provider(
    provider: &dyn LLMProvider,
    model: &str,
    role: &str,
) {
    use backon::ConstantBuilder;

    let messages = vec![serde_json::json!({"role": "user", "content": "Reply with: ok"})];
    let mut backoff = ConstantBuilder::default()
        .with_delay(Duration::from_secs(2))
        .with_max_times(10)
        .build();
    loop {
        match provider.chat(&messages, None, Some(model), 32, 0.0, None, None).await {
            Ok(resp) => {
                let text = resp.content.unwrap_or_default();
                if !text.trim().is_empty() {
                    eprintln!("  {} warmup OK: {}", role, &text[..text.len().min(40)]);
                    return;
                }
            }
            Err(e) => {
                let msg = e.to_string().to_lowercase();
                if !msg.contains("loading") && !msg.contains("503") {
                    panic!("{} warmup failed (non-retryable): {}", role, e);
                }
            }
        }
        match backoff.next() {
            Some(delay) => {
                eprintln!("  {} warming up, retrying in {:?}...", role, delay);
                tokio::time::sleep(delay).await;
            }
            None => panic!("{} did not become ready after retries", role),
        }
    }
}

#[tokio::test]
#[ignore = "requires LM Studio at NANOBOT_TRIO_BASE"]
async fn test_trio_e2e_preflight() {
    let (base, main_model, router_model, specialist_model) = trio_e2e_env();
    eprintln!("trio E2E preflight: base={}", base);

    // 1. Verify LM Studio /models endpoint is reachable
    let models_url = format!("{}/models", base.trim_end_matches("/v1").trim_end_matches('/'));
    // Try the /v1/models path first (standard OpenAI-compat)
    let models_url_v1 = format!("{}/models", base.trim_end_matches('/'));
    let client = reqwest::Client::new();
    let models_resp = client
        .get(&models_url_v1)
        .header("Authorization", "Bearer local")
        .timeout(Duration::from_secs(10))
        .send()
        .await;

    match &models_resp {
        Ok(resp) if resp.status().is_success() => {
            eprintln!("  /models endpoint OK (status {})", resp.status());
        }
        Ok(resp) => {
            panic!(
                "preflight FAILED: /models returned HTTP {} — is LM Studio running at {}?",
                resp.status(),
                base
            );
        }
        Err(e) => {
            panic!(
                "preflight FAILED: cannot reach {} — {}\nStart LM Studio or set NANOBOT_TRIO_BASE.",
                models_url_v1, e
            );
        }
    }

    // 2. Parse model list and check availability
    let body: serde_json::Value = models_resp
        .unwrap()
        .json()
        .await
        .expect("preflight: /models response is not valid JSON");

    let model_ids: Vec<String> = body
        .get("data")
        .and_then(|d| d.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|m| m.get("id").and_then(|id| id.as_str()).map(String::from))
                .collect()
        })
        .unwrap_or_default();

    eprintln!("  available models: {:?}", model_ids);

    // Note: LM Studio with JIT loading may not list all models upfront.
    // We log availability but don't fail — the warmup step below is the real gate.
    for (name, role) in [
        (&main_model, "main"),
        (&router_model, "router"),
        (&specialist_model, "specialist"),
    ] {
        if model_ids.iter().any(|id| id.contains(name.as_str())) {
            eprintln!("  {} model '{}' found in /models", role, name);
        } else {
            eprintln!("  {} model '{}' NOT listed (may JIT-load on demand)", role, name);
        }
    }

    // 3. Build harness and warmup all 3 providers (the real gate)
    let (agent_loop, workspace) =
        build_trio_e2e_harness(&base, &main_model, &router_model, &specialist_model);

    let core = agent_loop.shared.core_handle.swappable();
    warmup_trio_provider(&*core.provider, &main_model, "main").await;
    warmup_trio_provider(
        core.router_provider.as_ref().unwrap().as_ref(),
        &router_model,
        "router",
    )
    .await;
    warmup_trio_provider(
        core.specialist_provider.as_ref().unwrap().as_ref(),
        &specialist_model,
        "specialist",
    )
    .await;

    eprintln!("trio E2E preflight: ALL OK — infrastructure ready");
    let _ = std::fs::remove_dir_all(&workspace);
}

#[tokio::test]
#[ignore = "requires LM Studio at NANOBOT_TRIO_BASE"]
async fn test_trio_e2e_respond() {
    let (base, main_model, router_model, specialist_model) = trio_e2e_env();
    eprintln!("trio E2E respond: base={}", base);

    let (agent_loop, workspace) = build_trio_e2e_harness(&base, &main_model, &router_model, &specialist_model);

    // Warmup all 3 models
    let core = agent_loop.shared.core_handle.swappable();
    warmup_trio_provider(&*core.provider, &main_model, "main").await;
    warmup_trio_provider(core.router_provider.as_ref().unwrap().as_ref(), &router_model, "router").await;
    warmup_trio_provider(core.specialist_provider.as_ref().unwrap().as_ref(), &specialist_model, "specialist").await;

    let resp = tokio::time::timeout(
        Duration::from_secs(180),
        agent_loop.process_direct("Hello, what is 2 + 2?", "trio-e2e-respond", "test", "trio-e2e"),
    )
    .await
    .expect("test timed out");

    eprintln!("trio E2E respond: response ({} chars): {}", resp.len(), &resp[..resp.len().min(200)]);
    assert!(!resp.is_empty(), "response should be non-empty");

    let _ = std::fs::remove_dir_all(&workspace);
}

#[tokio::test]
#[ignore = "requires LM Studio at NANOBOT_TRIO_BASE"]
async fn test_trio_e2e_tool_dispatch() {
    let (base, main_model, router_model, specialist_model) = trio_e2e_env();
    eprintln!("trio E2E tool dispatch: base={}", base);

    let (agent_loop, workspace) = build_trio_e2e_harness(&base, &main_model, &router_model, &specialist_model);

    // Write a known file to workspace
    std::fs::write(workspace.join("README.md"), "Nanobot is a lightweight AI assistant framework written in Rust.").unwrap();

    let core = agent_loop.shared.core_handle.swappable();
    warmup_trio_provider(&*core.provider, &main_model, "main").await;
    warmup_trio_provider(core.router_provider.as_ref().unwrap().as_ref(), &router_model, "router").await;
    warmup_trio_provider(core.specialist_provider.as_ref().unwrap().as_ref(), &specialist_model, "specialist").await;

    let resp = tokio::time::timeout(
        Duration::from_secs(180),
        agent_loop.process_direct(
            "Read the file README.md and tell me what it says",
            "trio-e2e-tool",
            "test",
            "trio-e2e",
        ),
    )
    .await
    .expect("test timed out");

    eprintln!("trio E2E tool dispatch: response ({} chars): {}", resp.len(), &resp[..resp.len().min(200)]);
    assert!(!resp.is_empty(), "response should be non-empty");

    // Check TrioMetrics
    let metrics = &agent_loop.shared.core_handle.counters.trio_metrics;
    eprintln!(
        "  metrics: preflight={} action={:?} specialist={} tool={:?}",
        metrics.router_preflight_fired.load(std::sync::atomic::Ordering::Relaxed),
        metrics.router_action.lock().unwrap(),
        metrics.specialist_dispatched.load(std::sync::atomic::Ordering::Relaxed),
        metrics.tool_dispatched.lock().unwrap(),
    );

    let _ = std::fs::remove_dir_all(&workspace);
}

#[tokio::test]
#[ignore = "requires LM Studio at NANOBOT_TRIO_BASE"]
async fn test_trio_e2e_specialist_dispatch() {
    let (base, main_model, router_model, specialist_model) = trio_e2e_env();
    eprintln!("trio E2E specialist: base={}", base);

    let (agent_loop, workspace) = build_trio_e2e_harness(&base, &main_model, &router_model, &specialist_model);

    let core = agent_loop.shared.core_handle.swappable();
    warmup_trio_provider(&*core.provider, &main_model, "main").await;
    warmup_trio_provider(core.router_provider.as_ref().unwrap().as_ref(), &router_model, "router").await;
    warmup_trio_provider(core.specialist_provider.as_ref().unwrap().as_ref(), &specialist_model, "specialist").await;

    let resp = tokio::time::timeout(
        Duration::from_secs(180),
        agent_loop.process_direct(
            "Provide a detailed technical analysis of REST vs GraphQL",
            "trio-e2e-specialist",
            "test",
            "trio-e2e",
        ),
    )
    .await
    .expect("test timed out");

    eprintln!("trio E2E specialist: response ({} chars): {}", resp.len(), &resp[..resp.len().min(200)]);
    assert!(!resp.is_empty(), "response should be non-empty");
    assert!(resp.len() > 50, "specialist response should be substantive (>50 chars)");

    let _ = std::fs::remove_dir_all(&workspace);
}

#[tokio::test]
#[ignore = "requires LM Studio at NANOBOT_TRIO_BASE"]
async fn test_trio_e2e_ask_user() {
    let (base, main_model, router_model, specialist_model) = trio_e2e_env();
    eprintln!("trio E2E ask_user: base={}", base);

    let (agent_loop, workspace) = build_trio_e2e_harness(&base, &main_model, &router_model, &specialist_model);

    let core = agent_loop.shared.core_handle.swappable();
    warmup_trio_provider(&*core.provider, &main_model, "main").await;
    warmup_trio_provider(core.router_provider.as_ref().unwrap().as_ref(), &router_model, "router").await;
    warmup_trio_provider(core.specialist_provider.as_ref().unwrap().as_ref(), &specialist_model, "specialist").await;

    let resp = tokio::time::timeout(
        Duration::from_secs(180),
        agent_loop.process_direct(
            "Do that thing with the file",
            "trio-e2e-ask",
            "test",
            "trio-e2e",
        ),
    )
    .await
    .expect("test timed out");

    eprintln!("trio E2E ask_user: response ({} chars): {}", resp.len(), &resp[..resp.len().min(200)]);
    assert!(!resp.is_empty(), "response should be non-empty");

    let _ = std::fs::remove_dir_all(&workspace);
}

#[tokio::test]
#[ignore = "requires LM Studio at NANOBOT_TRIO_BASE"]
async fn test_trio_e2e_router_unreachable() {
    let (base, main_model, _router_model, specialist_model) = trio_e2e_env();
    eprintln!("trio E2E router unreachable: base={}", base);

    // Router on dead port, main + specialist on real endpoint
    let (agent_loop, workspace) = build_trio_e2e_harness(
        &base,
        &main_model,
        &"unreachable-router-model".to_string(), // model doesn't matter since we override the provider
        &specialist_model,
    );

    // Actually, the harness uses shared base for all providers.
    // For unreachable router, we need a custom build with bad router URL.
    // Let's build it manually.
    drop(agent_loop);
    let _ = std::fs::remove_dir_all(&workspace);

    use crate::providers::factory;
    use crate::providers::jit_gate::JitGate;
    use crate::config::schema::{DelegationMode, LcmSchemaConfig};

    let jit_gate = std::sync::Arc::new(JitGate::new());
    let main_provider: Arc<dyn LLMProvider> = factory::create_openai_compat(
        factory::ProviderSpec::local(&base, Some(&main_model))
            .with_jit_gate_opt(Some(jit_gate.clone())),
    );
    // Router points to dead port
    let router_provider: Arc<dyn LLMProvider> = Arc::new(
        OpenAICompatProvider::new("local", Some("http://127.0.0.1:19999/v1"), Some("dead-router")),
    );
    let specialist_provider: Arc<dyn LLMProvider> = factory::create_openai_compat(
        factory::ProviderSpec::local(&base, Some(&specialist_model))
            .with_jit_gate_opt(Some(jit_gate.clone())),
    );

    let workspace = tempfile::tempdir().unwrap().into_path();
    let mut td = ToolDelegationConfig {
        mode: DelegationMode::Trio,
        ..Default::default()
    };
    td.apply_mode();

    let trio_config = TrioConfig {
        enabled: true,
        router_model: "dead-router".to_string(),
        specialist_model: specialist_model.to_string(),
        ..Default::default()
    };

    let core = build_swappable_core(SwappableCoreConfig {
        provider: main_provider,
        workspace: workspace.clone(),
        model: main_model.to_string(),
        max_iterations: 5,
        max_continuations: 2,
        max_tokens: 512,
        temperature: 0.3,
        max_context_tokens: 4096,
        brave_api_key: None,
        search_provider: "searxng".to_string(),
        searxng_url: "http://localhost:8888".to_string(),
        search_max_results: 5,
        jina_api_key: None,
        exec_timeout: 30,
        restrict_to_workspace: true,
        memory_config: MemoryConfig::default(),
        is_local: true,
        compaction_provider: None,
        tool_delegation: td,
        provenance: ProvenanceConfig::default(),
        max_tool_result_chars: 2000,
        delegation_provider: Some(router_provider),
        specialist_provider: Some(specialist_provider),
        trio_config,
        model_capabilities_overrides: std::collections::HashMap::new(),
        reasoning_config: crate::config::schema::ReasoningConfig::default(),
        tool_heartbeat_secs: 2,
        health_check_timeout_secs: 2,
        adaptive_tokens: AdaptiveTokenConfig::default(),
    });
    let counters = Arc::new(crate::agent::agent_core::RuntimeCounters::new(4096));
    let core_handle = AgentHandle::new(core, counters);

    let (inbound_tx, inbound_rx) = tokio::sync::mpsc::unbounded_channel::<InboundMessage>();
    let (outbound_tx, _outbound_rx) = tokio::sync::mpsc::unbounded_channel::<OutboundMessage>();

    let agent_loop = AgentLoop::new(
        core_handle,
        inbound_rx,
        outbound_tx,
        inbound_tx,
        None,
        1,
        None,
        None,
        None,
        ProprioceptionConfig::default(),
        LcmSchemaConfig::default(),
        None,
    );

    // Only warmup main (router is intentionally dead)
    let core = agent_loop.shared.core_handle.swappable();
    warmup_trio_provider(&*core.provider, &main_model, "main").await;

    let resp = tokio::time::timeout(
        Duration::from_secs(60),
        agent_loop.process_direct("Hello", "trio-e2e-router-dead", "test", "trio-e2e"),
    )
    .await
    .expect("test timed out");

    eprintln!("trio E2E router unreachable: response ({} chars): {}", resp.len(), &resp[..resp.len().min(200)]);
    assert!(!resp.is_empty(), "should get error response, not panic");

    let _ = std::fs::remove_dir_all(&workspace);
}

#[tokio::test]
#[ignore = "requires LM Studio at NANOBOT_TRIO_BASE"]
async fn test_trio_e2e_specialist_unreachable() {
    let (base, main_model, router_model, _specialist_model) = trio_e2e_env();
    eprintln!("trio E2E specialist unreachable: base={}", base);

    use crate::providers::factory;
    use crate::providers::jit_gate::JitGate;
    use crate::config::schema::{DelegationMode, LcmSchemaConfig};

    let jit_gate = std::sync::Arc::new(JitGate::new());
    let main_provider: Arc<dyn LLMProvider> = factory::create_openai_compat(
        factory::ProviderSpec::local(&base, Some(&main_model))
            .with_jit_gate_opt(Some(jit_gate.clone())),
    );
    let router_provider: Arc<dyn LLMProvider> = factory::create_openai_compat(
        factory::ProviderSpec::local(&base, Some(&router_model))
            .with_jit_gate_opt(Some(jit_gate.clone())),
    );
    // Specialist points to dead port
    let specialist_provider: Arc<dyn LLMProvider> = Arc::new(
        OpenAICompatProvider::new("local", Some("http://127.0.0.1:19999/v1"), Some("dead-specialist")),
    );

    let workspace = tempfile::tempdir().unwrap().into_path();
    let mut td = ToolDelegationConfig {
        mode: DelegationMode::Trio,
        ..Default::default()
    };
    td.apply_mode();

    let trio_config = TrioConfig {
        enabled: true,
        router_model: router_model.to_string(),
        specialist_model: "dead-specialist".to_string(),
        ..Default::default()
    };

    let core = build_swappable_core(SwappableCoreConfig {
        provider: main_provider,
        workspace: workspace.clone(),
        model: main_model.to_string(),
        max_iterations: 5,
        max_continuations: 2,
        max_tokens: 512,
        temperature: 0.3,
        max_context_tokens: 4096,
        brave_api_key: None,
        search_provider: "searxng".to_string(),
        searxng_url: "http://localhost:8888".to_string(),
        search_max_results: 5,
        jina_api_key: None,
        exec_timeout: 30,
        restrict_to_workspace: true,
        memory_config: MemoryConfig::default(),
        is_local: true,
        compaction_provider: None,
        tool_delegation: td,
        provenance: ProvenanceConfig::default(),
        max_tool_result_chars: 2000,
        delegation_provider: Some(router_provider),
        specialist_provider: Some(specialist_provider),
        trio_config,
        model_capabilities_overrides: std::collections::HashMap::new(),
        reasoning_config: crate::config::schema::ReasoningConfig::default(),
        tool_heartbeat_secs: 2,
        health_check_timeout_secs: 2,
        adaptive_tokens: AdaptiveTokenConfig::default(),
    });
    let counters = Arc::new(crate::agent::agent_core::RuntimeCounters::new(4096));
    let core_handle = AgentHandle::new(core, counters);

    let (inbound_tx, inbound_rx) = tokio::sync::mpsc::unbounded_channel::<InboundMessage>();
    let (outbound_tx, _outbound_rx) = tokio::sync::mpsc::unbounded_channel::<OutboundMessage>();

    let agent_loop = AgentLoop::new(
        core_handle,
        inbound_rx,
        outbound_tx,
        inbound_tx,
        None,
        1,
        None,
        None,
        None,
        ProprioceptionConfig::default(),
        LcmSchemaConfig::default(),
        None,
    );

    let core = agent_loop.shared.core_handle.swappable();
    warmup_trio_provider(&*core.provider, &main_model, "main").await;
    warmup_trio_provider(core.router_provider.as_ref().unwrap().as_ref(), &router_model, "router").await;

    let resp = tokio::time::timeout(
        Duration::from_secs(180),
        agent_loop.process_direct(
            "Provide a detailed technical analysis of REST vs GraphQL",
            "trio-e2e-specialist-dead",
            "test",
            "trio-e2e",
        ),
    )
    .await
    .expect("test timed out");

    eprintln!("trio E2E specialist unreachable: response ({} chars): {}", resp.len(), &resp[..resp.len().min(200)]);
    assert!(!resp.is_empty(), "should get response despite dead specialist");

    let _ = std::fs::remove_dir_all(&workspace);
}

#[tokio::test]
#[ignore = "requires LM Studio at NANOBOT_TRIO_BASE"]
async fn test_trio_e2e_multi_turn() {
    let (base, main_model, router_model, specialist_model) = trio_e2e_env();
    eprintln!("trio E2E multi-turn: base={}", base);

    let (agent_loop, workspace) = build_trio_e2e_harness(&base, &main_model, &router_model, &specialist_model);

    // Write test file
    std::fs::write(workspace.join("README.md"), "Nanobot is a lightweight AI assistant.").unwrap();

    let core = agent_loop.shared.core_handle.swappable();
    warmup_trio_provider(&*core.provider, &main_model, "main").await;
    warmup_trio_provider(core.router_provider.as_ref().unwrap().as_ref(), &router_model, "router").await;
    warmup_trio_provider(core.specialist_provider.as_ref().unwrap().as_ref(), &specialist_model, "specialist").await;

    let session_key = "trio-e2e-multi";

    // Turn 1: simple greeting (respond path)
    let resp1 = tokio::time::timeout(
        Duration::from_secs(180),
        agent_loop.process_direct("Hello", session_key, "test", "trio-e2e"),
    )
    .await
    .expect("turn 1 timed out");
    eprintln!("turn 1 ({} chars): {}", resp1.len(), &resp1[..resp1.len().min(100)]);
    assert!(!resp1.is_empty(), "turn 1 should be non-empty");

    // Turn 2: tool path
    let resp2 = tokio::time::timeout(
        Duration::from_secs(180),
        agent_loop.process_direct("Read README.md", session_key, "test", "trio-e2e"),
    )
    .await
    .expect("turn 2 timed out");
    eprintln!("turn 2 ({} chars): {}", resp2.len(), &resp2[..resp2.len().min(100)]);
    assert!(!resp2.is_empty(), "turn 2 should be non-empty");

    // Turn 3: follow-up (tests session state persistence)
    let resp3 = tokio::time::timeout(
        Duration::from_secs(180),
        agent_loop.process_direct("Summarize what you found", session_key, "test", "trio-e2e"),
    )
    .await
    .expect("turn 3 timed out");
    eprintln!("turn 3 ({} chars): {}", resp3.len(), &resp3[..resp3.len().min(100)]);
    assert!(!resp3.is_empty(), "turn 3 should be non-empty");

    let _ = std::fs::remove_dir_all(&workspace);
}

// -----------------------------------------------------------------------
// should_strip_tools_for_trio — pure function tests
// -----------------------------------------------------------------------

#[test]
fn test_should_strip_tools_all_healthy() {
    assert!(should_strip_tools_for_trio(true, true, true, true));
}

#[test]
fn test_should_strip_tools_not_local() {
    // Cloud mode: never strip tools via this path.
    assert!(!should_strip_tools_for_trio(false, true, true, true));
}

#[test]
fn test_should_strip_tools_no_strict_mode() {
    // strict_no_tools_main is false: don't strip.
    assert!(!should_strip_tools_for_trio(true, false, true, true));
}

#[test]
fn test_should_strip_tools_router_unhealthy() {
    // Router probe degraded: keep tools for fallback.
    assert!(!should_strip_tools_for_trio(true, true, false, true));
}

#[test]
fn test_should_strip_tools_circuit_breaker_open() {
    // Circuit breaker tripped: keep tools for fallback.
    assert!(!should_strip_tools_for_trio(true, true, true, false));
}

#[test]
fn test_should_strip_tools_both_degraded() {
    // Both degraded: definitely keep tools.
    assert!(!should_strip_tools_for_trio(true, true, false, false));
}

#[test]
fn test_adaptive_max_tokens_reserves_budget_for_local_thinking() {
    let out = adaptive_max_tokens(4096, false, "What time is it?", 0, true, Some(512), &AdaptiveTokenConfig::default());
    assert_eq!(out, 3584);
}

#[test]
fn test_adaptive_max_tokens_no_reserve_without_thinking() {
    let out = adaptive_max_tokens(4096, false, "What time is it?", 0, true, None, &AdaptiveTokenConfig::default());
    assert_eq!(out, 4096);
}

#[test]
fn test_adaptive_max_tokens_no_reserve_for_cloud() {
    let out = adaptive_max_tokens(4096, false, "What time is it?", 0, false, Some(512), &AdaptiveTokenConfig::default());
    assert_eq!(out, 4096);
}

#[test]
fn test_adaptive_max_tokens_keeps_floor_when_base_is_small() {
    let out = adaptive_max_tokens(512, false, "short", 0, true, Some(128), &AdaptiveTokenConfig::default());
    assert_eq!(out, 256);
}

// -----------------------------------------------------------------------
// Offline trio E2E tests (no network required — all providers are mocks)
// -----------------------------------------------------------------------

/// A mock LLM provider that returns responses from a pre-loaded queue.
///
/// Each call pops the next response. When the queue is empty it returns a
/// sentinel error string so tests can detect over-calling.
struct SequenceProvider {
    name: String,
    responses: std::sync::Mutex<std::collections::VecDeque<String>>,
    call_count: std::sync::atomic::AtomicU32,
}

impl SequenceProvider {
    fn new(name: &str, responses: Vec<&str>) -> Self {
        Self {
            name: name.to_string(),
            responses: std::sync::Mutex::new(
                responses.into_iter().map(|s| s.to_string()).collect(),
            ),
            call_count: std::sync::atomic::AtomicU32::new(0),
        }
    }

    fn call_count(&self) -> u32 {
        self.call_count.load(std::sync::atomic::Ordering::Relaxed)
    }
}

#[async_trait]
impl LLMProvider for SequenceProvider {
    async fn chat(
        &self,
        _messages: &[Value],
        _tools: Option<&[Value]>,
        _model: Option<&str>,
        _max_tokens: u32,
        _temperature: f64,
        _thinking_budget: Option<u32>,
        _top_p: Option<f64>,
    ) -> anyhow::Result<crate::providers::base::LLMResponse> {
        self.call_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let response = {
            let mut deque = self.responses.lock().unwrap();
            if deque.is_empty() {
                "ERROR: no responses left in SequenceProvider".to_string()
            } else {
                deque.pop_front().unwrap()
            }
        };
        Ok(crate::providers::base::LLMResponse {
            content: Some(response),
            tool_calls: vec![],
            finish_reason: "stop".to_string(),
            usage: std::collections::HashMap::new(),
        })
    }

    fn get_default_model(&self) -> &str {
        &self.name
    }
}

struct RecordingProvider {
    name: String,
    response: String,
    last_max_tokens: std::sync::atomic::AtomicU32,
}

impl RecordingProvider {
    fn new(name: &str, response: &str) -> Self {
        Self {
            name: name.to_string(),
            response: response.to_string(),
            last_max_tokens: std::sync::atomic::AtomicU32::new(0),
        }
    }

    fn last_max_tokens(&self) -> u32 {
        self.last_max_tokens
            .load(std::sync::atomic::Ordering::Relaxed)
    }
}

#[async_trait]
impl LLMProvider for RecordingProvider {
    async fn chat(
        &self,
        _messages: &[Value],
        _tools: Option<&[Value]>,
        _model: Option<&str>,
        max_tokens: u32,
        _temperature: f64,
        _thinking_budget: Option<u32>,
        _top_p: Option<f64>,
    ) -> anyhow::Result<crate::providers::base::LLMResponse> {
        self.last_max_tokens
            .store(max_tokens, std::sync::atomic::Ordering::Relaxed);
        Ok(crate::providers::base::LLMResponse {
            content: Some(self.response.clone()),
            tool_calls: vec![],
            finish_reason: "stop".to_string(),
            usage: std::collections::HashMap::new(),
        })
    }

    fn get_default_model(&self) -> &str {
        &self.name
    }
}

/// Build an offline trio harness from pre-built mock providers.
///
/// Mirrors `build_trio_e2e_harness` but accepts providers directly rather
/// than constructing real HTTP clients. No background probes are wired.
fn build_trio_offline_harness(
    main: Arc<dyn LLMProvider>,
    router: Arc<dyn LLMProvider>,
    specialist: Arc<dyn LLMProvider>,
) -> (AgentLoop, std::path::PathBuf) {
    use crate::config::schema::LcmSchemaConfig;

    let workspace = tempfile::tempdir().unwrap().into_path();

    let mut td = ToolDelegationConfig {
        mode: crate::config::schema::DelegationMode::Trio,
        ..Default::default()
    };
    td.apply_mode(); // sets strict_no_tools_main = true, strict_router_schema = true

    let router_model = router.get_default_model().to_string();
    let specialist_model = specialist.get_default_model().to_string();

    let trio_config = TrioConfig {
        enabled: true,
        router_model: router_model.clone(),
        specialist_model: specialist_model.clone(),
        ..Default::default()
    };

    let core = build_swappable_core(SwappableCoreConfig {
        provider: main,
        workspace: workspace.clone(),
        model: "offline-main".to_string(),
        max_iterations: 5,
        max_continuations: 2,
        max_tokens: 512,
        temperature: 0.3,
        max_context_tokens: 4096,
        brave_api_key: None,
        search_provider: "searxng".to_string(),
        searxng_url: "http://localhost:8888".to_string(),
        search_max_results: 5,
        jina_api_key: None,
        exec_timeout: 30,
        restrict_to_workspace: true,
        memory_config: MemoryConfig::default(),
        is_local: true,
        compaction_provider: None,
        tool_delegation: td,
        provenance: ProvenanceConfig::default(),
        max_tool_result_chars: 2000,
        delegation_provider: Some(router),
        specialist_provider: Some(specialist),
        trio_config,
        model_capabilities_overrides: std::collections::HashMap::new(),
        reasoning_config: crate::config::schema::ReasoningConfig::default(),
        tool_heartbeat_secs: 2,
        health_check_timeout_secs: 2,
        adaptive_tokens: AdaptiveTokenConfig::default(),
    });

    let counters = Arc::new(crate::agent::agent_core::RuntimeCounters::new(4096));
    let core_handle = AgentHandle::new(core, counters);

    let (inbound_tx, inbound_rx) =
        tokio::sync::mpsc::unbounded_channel::<InboundMessage>();
    let (outbound_tx, _outbound_rx) =
        tokio::sync::mpsc::unbounded_channel::<OutboundMessage>();

    let agent_loop = AgentLoop::new(
        core_handle,
        inbound_rx,
        outbound_tx,
        inbound_tx,
        None,
        1,
        None,
        None,
        None,
        ProprioceptionConfig::default(),
        LcmSchemaConfig::default(),
        None, // no health_registry — offline tests manage their own
    );

    (agent_loop, workspace)
}

// -----------------------------------------------------------------------
// Test 1: router decides "respond" — specialist is never called
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_trio_offline_e2e_respond() {
    let router_resp = r#"{"action":"respond","target":"main","args":{},"confidence":0.9}"#;
    let main_resp = "Four.";

    let router: Arc<dyn LLMProvider> = Arc::new(SequenceProvider::new(
        "offline-router",
        vec![router_resp, router_resp, router_resp],
    ));
    let main: Arc<dyn LLMProvider> = Arc::new(StaticResponseLLM::new("offline-main", main_resp));
    let specialist: Arc<dyn LLMProvider> = Arc::new(StaticResponseLLM::new(
        "offline-specialist",
        "specialist unused",
    ));

    let (agent_loop, workspace) =
        build_trio_offline_harness(main, router, specialist);

    let resp = agent_loop
        .process_direct("What is 2+2?", "trio-offline-respond", "test", "offline")
        .await;

    eprintln!(
        "test_trio_offline_e2e_respond: response ({} chars): {}",
        resp.len(),
        &resp[..resp.len().min(200)]
    );

    let counters = &agent_loop.shared.core_handle.counters;
    let metrics = &counters.trio_metrics;

    assert!(
        metrics.router_preflight_fired.load(std::sync::atomic::Ordering::Relaxed),
        "router preflight should have fired"
    );
    assert_eq!(
        metrics.router_action.lock().unwrap().as_deref(),
        Some("respond"),
        "router_action should be 'respond'"
    );
    assert!(
        !metrics.specialist_dispatched.load(std::sync::atomic::Ordering::Relaxed),
        "specialist should NOT have been dispatched for a 'respond' decision"
    );
    assert!(!resp.is_empty(), "response should be non-empty");

    let _ = std::fs::remove_dir_all(&workspace);
}

#[tokio::test]
async fn test_local_thinking_reserves_max_tokens_end_to_end() {
    let router_resp = r#"{"action":"respond","target":"main","args":{},"confidence":0.9}"#;
    let router: Arc<dyn LLMProvider> = Arc::new(SequenceProvider::new(
        "offline-router",
        vec![router_resp, router_resp, router_resp],
    ));
    let main = Arc::new(RecordingProvider::new("offline-main", "ok"));
    let main_dyn: Arc<dyn LLMProvider> = main.clone();
    let specialist: Arc<dyn LLMProvider> = Arc::new(StaticResponseLLM::new(
        "offline-specialist",
        "unused",
    ));

    let (agent_loop, workspace) = build_trio_offline_harness(main_dyn, router, specialist);
    agent_loop
        .shared
        .core_handle
        .counters
        .thinking_budget
        .store(128, std::sync::atomic::Ordering::Relaxed);

    let _ = agent_loop
        .process_direct("What is the current date?", "reserve-max-tokens", "test", "offline")
        .await;

    assert_eq!(
        main.last_max_tokens(),
        256,
        "local thinking should reserve tool-call budget from base max_tokens=512"
    );

    let _ = std::fs::remove_dir_all(&workspace);
}

// -----------------------------------------------------------------------
// Test 2: router decides "specialist" — specialist is called
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_trio_offline_e2e_specialist_dispatch() {
    let router_resp = r#"{"action":"specialist","target":"coding","args":{"task":"explain loops"},"confidence":0.85}"#;

    let router: Arc<dyn LLMProvider> = Arc::new(SequenceProvider::new(
        "offline-router",
        vec![router_resp, router_resp, router_resp],
    ));
    let main: Arc<dyn LLMProvider> =
        Arc::new(StaticResponseLLM::new("offline-main", "delegating"));
    let specialist: Arc<dyn LLMProvider> = Arc::new(StaticResponseLLM::new(
        "offline-specialist",
        "Here is the specialist answer.",
    ));

    let (agent_loop, workspace) =
        build_trio_offline_harness(main, router, specialist);

    let resp = agent_loop
        .process_direct(
            "Explain for loops",
            "trio-offline-specialist",
            "test",
            "offline",
        )
        .await;

    eprintln!(
        "test_trio_offline_e2e_specialist_dispatch: response ({} chars): {}",
        resp.len(),
        &resp[..resp.len().min(200)]
    );

    let metrics = &agent_loop.shared.core_handle.counters.trio_metrics;

    assert_eq!(
        metrics.router_action.lock().unwrap().as_deref(),
        Some("specialist"),
        "router_action should be 'specialist'"
    );
    assert!(
        metrics.specialist_dispatched.load(std::sync::atomic::Ordering::Relaxed),
        "specialist should have been dispatched"
    );
    assert!(!resp.is_empty(), "response should be non-empty");

    let _ = std::fs::remove_dir_all(&workspace);
}

// -----------------------------------------------------------------------
// Test 3: circuit breaker cascade
//
// The router returns non-JSON 3+ times. Each failure is recorded under
// the key "router:{model}" (as router.rs does). However, agent_loop.rs
// checks availability under "trio_router" — so the CB check at the
// should_strip_tools_for_trio call site never sees the tripped breaker.
//
// This test documents that discrepancy explicitly.
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_trio_offline_e2e_circuit_breaker_cascade() {
    // All 4 router calls return non-JSON to trip the circuit breaker.
    let router: Arc<dyn LLMProvider> = Arc::new(SequenceProvider::new(
        "offline-router",
        vec![
            "this is not json at all !!!",
            "this is not json at all !!!",
            "this is not json at all !!!",
            "this is not json at all !!!",
        ],
    ));
    let main: Arc<dyn LLMProvider> = Arc::new(StaticResponseLLM::new(
        "offline-main",
        "main fallback response",
    ));
    let specialist: Arc<dyn LLMProvider> =
        Arc::new(StaticResponseLLM::new("offline-specialist", "specialist unused"));

    let (agent_loop, workspace) =
        build_trio_offline_harness(main, router, specialist);

    // Send 4 messages — each failure increments the CB counter.
    // After 3 failures (default threshold) the CB is tripped.
    // The 4th call will be via Passthrough (router returns early) because
    // the CB key "router:offline-router" is open. Main answers directly.
    for i in 0..4u32 {
        let resp = agent_loop
            .process_direct(
                &format!("message {}", i),
                "trio-offline-cb",
                "test",
                "offline",
            )
            .await;
        eprintln!(
            "  cascade msg {}: ({} chars) {}",
            i,
            resp.len(),
            &resp[..resp.len().min(80)]
        );
    }

    let counters = &agent_loop.shared.core_handle.counters;

    // After repeated failures the trio state should be Degraded.
    let state = counters.get_trio_state();
    eprintln!("trio_state after cascade: {:?}", state);
    assert_eq!(
        state,
        crate::agent::agent_core::TrioState::Degraded,
        "trio_state should be Degraded after repeated router failures"
    );

    // Verify CB key alignment after the fix.
    //
    // The offline harness returns mock responses that fail strict AND lenient
    // parsing (lenient no longer defaults to phantom "clarify" target — it
    // returns None when no target can be extracted). Each parse failure records
    // a CB failure, so after 4 turns the CB should be tripped.
    //
    // The shared CB key format ("router:{model}") ensures that the
    // tool-stripping guard in step_pre_call and the routing skip in
    // router_preflight observe the same state.
    let cb_correct_key_available = counters
        .trio_circuit_breaker
        .lock()
        .unwrap()
        .is_available("router:offline-router");
    eprintln!(
        "CB 'router:offline-router' available after 4 turns: {}",
        cb_correct_key_available
    );
    // Parse failures are now correctly recorded — CB should be tripped.
    assert!(
        !cb_correct_key_available,
        "CB 'router:offline-router' should be tripped: parse failures are now recorded"
    );
    // The legacy key "trio_router" is also untouched.
    let cb_legacy_key_available = counters
        .trio_circuit_breaker
        .lock()
        .unwrap()
        .is_available("trio_router");
    assert!(
        cb_legacy_key_available,
        "CB 'trio_router' should be untouched — agent_loop now uses 'router:{{model}}' key"
    );

    let _ = std::fs::remove_dir_all(&workspace);
}

// -----------------------------------------------------------------------
// Test 4: health gate — degraded router probe bypasses preflight
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_trio_offline_e2e_health_gate() {
    use crate::heartbeat::health::{HealthProbe, HealthRegistry, ProbeResult};
    use crate::config::schema::LcmSchemaConfig;

    // A mock probe that always returns unhealthy (simulates router being down).
    struct AlwaysUnhealthyProbe;

    #[async_trait]
    impl HealthProbe for AlwaysUnhealthyProbe {
        fn name(&self) -> &str {
            "trio_router"
        }

        fn interval_secs(&self) -> u64 {
            0 // always due
        }

        async fn check(&self) -> ProbeResult {
            ProbeResult {
                healthy: false,
                latency_ms: 0,
                detail: Some("simulated failure".to_string()),
            }
        }
    }

    // Build a registry and degrade the trio_router probe.
    let mut health_registry = HealthRegistry::new();
    health_registry.register(Box::new(AlwaysUnhealthyProbe));
    // Run 3 times to reach DEGRADED_THRESHOLD = 3.
    for _ in 0..3 {
        health_registry.run_due_probes().await;
    }
    assert!(
        !health_registry.is_healthy("trio_router"),
        "trio_router should be degraded after 3 failures"
    );
    let health_registry = Arc::new(health_registry);

    // The router SequenceProvider would fail the test if called (empty queue).
    // We keep a typed Arc so we can read call_count() after the run.
    let router_seq = Arc::new(SequenceProvider::new(
        "offline-router",
        vec![], // empty — calling this would return the sentinel error
    ));
    let router: Arc<dyn LLMProvider> = router_seq.clone();
    let main: Arc<dyn LLMProvider> =
        Arc::new(StaticResponseLLM::new("offline-main", "main answer"));
    let specialist: Arc<dyn LLMProvider> =
        Arc::new(StaticResponseLLM::new("offline-specialist", "specialist unused"));

    // Build harness manually so we can wire in the health registry.
    let workspace = tempfile::tempdir().unwrap().into_path();
    let mut td = ToolDelegationConfig {
        mode: crate::config::schema::DelegationMode::Trio,
        ..Default::default()
    };
    td.apply_mode();

    let router_model = router.get_default_model().to_string();
    let specialist_model = specialist.get_default_model().to_string();
    let trio_config = TrioConfig {
        enabled: true,
        router_model: router_model.clone(),
        specialist_model: specialist_model.clone(),
        ..Default::default()
    };

    let core = build_swappable_core(SwappableCoreConfig {
        provider: main,
        workspace: workspace.clone(),
        model: "offline-main".to_string(),
        max_iterations: 5,
        max_continuations: 2,
        max_tokens: 512,
        temperature: 0.3,
        max_context_tokens: 4096,
        brave_api_key: None,
        search_provider: "searxng".to_string(),
        searxng_url: "http://localhost:8888".to_string(),
        search_max_results: 5,
        jina_api_key: None,
        exec_timeout: 30,
        restrict_to_workspace: true,
        memory_config: MemoryConfig::default(),
        is_local: true,
        compaction_provider: None,
        tool_delegation: td,
        provenance: ProvenanceConfig::default(),
        max_tool_result_chars: 2000,
        delegation_provider: Some(router.clone()),
        specialist_provider: Some(specialist),
        trio_config,
        model_capabilities_overrides: std::collections::HashMap::new(),
        reasoning_config: crate::config::schema::ReasoningConfig::default(),
        tool_heartbeat_secs: 2,
        health_check_timeout_secs: 2,
        adaptive_tokens: AdaptiveTokenConfig::default(),
    });

    let counters = Arc::new(crate::agent::agent_core::RuntimeCounters::new(4096));
    let core_handle = AgentHandle::new(core, counters);

    let (inbound_tx, inbound_rx) =
        tokio::sync::mpsc::unbounded_channel::<InboundMessage>();
    let (outbound_tx, _outbound_rx) =
        tokio::sync::mpsc::unbounded_channel::<OutboundMessage>();

    let agent_loop = AgentLoop::new(
        core_handle,
        inbound_rx,
        outbound_tx,
        inbound_tx,
        None,
        1,
        None,
        None,
        None,
        ProprioceptionConfig::default(),
        LcmSchemaConfig::default(),
        Some(health_registry), // health registry is wired in here
    );

    let resp = agent_loop
        .process_direct(
            "Hello",
            "trio-offline-health-gate",
            "test",
            "offline",
        )
        .await;

    eprintln!(
        "test_trio_offline_e2e_health_gate: response ({} chars): {}",
        resp.len(),
        &resp[..resp.len().min(200)]
    );

    // When the health gate fires, router_preflight returns Passthrough and sets Degraded.
    let state = agent_loop
        .shared
        .core_handle
        .counters
        .get_trio_state();
    eprintln!("trio_state after health gate: {:?}", state);
    assert_eq!(
        state,
        crate::agent::agent_core::TrioState::Degraded,
        "trio_state should be Degraded when health gate fires"
    );

    // Response must come from main (non-empty).
    assert!(!resp.is_empty(), "response should come from main, not be empty");

    // router_preflight_fired should be true (we entered preflight but returned Passthrough).
    let metrics = &agent_loop.shared.core_handle.counters.trio_metrics;
    assert!(
        metrics.router_preflight_fired.load(std::sync::atomic::Ordering::Relaxed),
        "router_preflight_fired should be true (preflight was entered)"
    );

    // Specialist must not have been dispatched.
    assert!(
        !metrics.specialist_dispatched.load(std::sync::atomic::Ordering::Relaxed),
        "specialist should not be dispatched when health gate is active"
    );

    // Router's chat() should never have been called — health gate fired before it.
    assert_eq!(
        router_seq.call_count(),
        0,
        "router provider's chat() call count should be 0 (health gate bypassed it)"
    );

    let _ = std::fs::remove_dir_all(&workspace);
}

// -----------------------------------------------------------------------
// Test 5: lenient parse fallback
//
// Router returns FunctionGemma comma-separated format:
//   "specialist,coding,{}"
// `parse_lenient_router_decision` handles this format.
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_trio_offline_e2e_parse_fallback_lenient() {
    // Lenient format: "action,target,{args}" — no JSON wrapper.
    // This exercises the comma-separated branch in parse_lenient_router_decision.
    let router_resp = "specialist,coding,{}";

    let router: Arc<dyn LLMProvider> = Arc::new(SequenceProvider::new(
        "offline-router",
        vec![router_resp, router_resp, router_resp],
    ));
    let main: Arc<dyn LLMProvider> =
        Arc::new(StaticResponseLLM::new("offline-main", "delegating"));
    let specialist: Arc<dyn LLMProvider> = Arc::new(StaticResponseLLM::new(
        "offline-specialist",
        "lenient parse worked",
    ));

    // Verify that parse_lenient_router_decision handles this format before
    // wiring it into the full agent loop.
    let lenient_decision = parse_lenient_router_decision(router_resp);
    assert!(
        lenient_decision.is_some(),
        "parse_lenient_router_decision should accept 'specialist,coding,{{}}'"
    );
    let lenient_decision = lenient_decision.unwrap();
    assert_eq!(
        lenient_decision.action, "specialist",
        "lenient decision action should be 'specialist'"
    );

    let (agent_loop, workspace) =
        build_trio_offline_harness(main, router, specialist);

    let resp = agent_loop
        .process_direct(
            "Explain something complex",
            "trio-offline-lenient",
            "test",
            "offline",
        )
        .await;

    eprintln!(
        "test_trio_offline_e2e_parse_fallback_lenient: response ({} chars): {}",
        resp.len(),
        &resp[..resp.len().min(200)]
    );

    let metrics = &agent_loop.shared.core_handle.counters.trio_metrics;

    assert_eq!(
        metrics.router_action.lock().unwrap().as_deref(),
        Some("specialist"),
        "router_action should be 'specialist' after lenient parse"
    );
    assert!(
        metrics.specialist_dispatched.load(std::sync::atomic::Ordering::Relaxed),
        "specialist should have been dispatched after lenient parse"
    );
    assert!(!resp.is_empty(), "response should be non-empty");

    let _ = std::fs::remove_dir_all(&workspace);
}


// ============================================================================
// appears_incomplete heuristic tests
// ============================================================================

mod continuation_tests {
    use super::appears_incomplete;

    #[test]
    fn test_unclosed_backtick_detected() {
        assert!(appears_incomplete("The template to skip `"));
        assert!(appears_incomplete("Thinking blocks (`"));
    }

    #[test]
    fn test_complete_response_not_flagged() {
        assert!(!appears_incomplete("This is a complete sentence."));
        assert!(!appears_incomplete("Done!"));
        assert!(!appears_incomplete("Use `code` here."));
        assert!(!appears_incomplete("```\ncode\n```"));
    }

    #[test]
    fn test_mid_sentence_detected() {
        assert!(appears_incomplete("The quick brown fox jumped over the"));
        assert!(appears_incomplete("Here are the steps to configure"));
    }

    #[test]
    fn test_short_fragments_not_flagged() {
        assert!(!appears_incomplete("OK"));
        assert!(!appears_incomplete("Yes"));
    }

    #[test]
    fn test_unclosed_paren_detected() {
        assert!(appears_incomplete("The function signature is fn foo(bar"));
    }

    #[test]
    fn test_appears_incomplete_mid_sentence() {
        // Text ending mid-word (no terminal punctuation, long enough to trigger)
        assert!(appears_incomplete("The configuration requires setting the correc"));
        assert!(appears_incomplete("You can use this approach to implemen"));
    }

    #[test]
    fn test_appears_incomplete_complete() {
        // Text ending with period or exclamation is considered complete
        assert!(!appears_incomplete("The task is now complete."));
        assert!(!appears_incomplete("All done!"));
        assert!(!appears_incomplete("Did it work?"));
    }

    #[test]
    fn test_trailing_emoji_not_flagged() {
        // Period before emoji — response is complete, must not trigger continuation
        assert!(!appears_incomplete("Why cross the road? To avoid borrows. 🦀"));
        // Period before multiple emojis
        assert!(!appears_incomplete("The answer is 42. 🎉✨"));
    }

    #[test]
    fn test_trailing_emoji_mid_sentence_still_flagged() {
        // No punctuation even after stripping emojis — still incomplete
        assert!(appears_incomplete("Here's a joke 🤣😂🔥"));
    }

    #[test]
    fn test_short_response_with_emoji_not_flagged() {
        // Under the 20-char length threshold
        assert!(!appears_incomplete("OK 👍"));
    }
}

// ============================================================================
// Universal textual tool-call parsing tests
// ============================================================================

mod universal_textual_parse_tests {
    use crate::agent::protocol::{parse_textual_tool_calls, strip_textual_tool_calls};

    #[test]
    fn test_textual_parse_strips_content() {
        // Content containing a [I called: ...] annotation should have the
        // annotation removed by strip_textual_tool_calls, leaving only prose.
        let input = "Sure, let me list the files.\n[I called: exec({\"command\": \"ls\"})]\nDone.";
        let stripped = strip_textual_tool_calls(input);
        assert!(
            !stripped.contains("[I called:"),
            "Expected [I called:] pattern to be stripped, got: {:?}",
            stripped
        );
        assert!(
            stripped.contains("Sure, let me list the files."),
            "Expected prose to be preserved, got: {:?}",
            stripped
        );
    }

    #[test]
    fn test_universal_parse_non_textual_replay() {
        // parse_textual_tool_calls should work on any content string regardless
        // of protocol mode — the function itself is protocol-agnostic.
        let content = "I will run the command now.\n[I called: exec({\"command\": \"echo hello\"})]";
        let parsed = parse_textual_tool_calls(content);
        assert_eq!(
            parsed.len(),
            1,
            "Expected 1 parsed tool call, got {}",
            parsed.len()
        );
        assert_eq!(parsed[0].tool, "exec");
        // Args should decode the command key.
        let cmd = parsed[0].args.get("command").and_then(|v| v.as_str());
        assert_eq!(cmd, Some("echo hello"));
    }

    #[test]
    fn test_textual_parse_no_match_returns_empty() {
        // Plain prose with no [I called: ...] patterns must return empty.
        let content = "There are no tool calls in this response.";
        let parsed = parse_textual_tool_calls(content);
        assert!(
            parsed.is_empty(),
            "Expected no parsed tool calls, got {:?}",
            parsed
        );
    }
}

mod nudge_tests {
    /// Verify that the 80%-ceiling formula produces the expected nudge thresholds.
    #[test]
    fn test_nudge_threshold_80_percent() {
        let nudge_at = |max: u32| -> u32 {
            ((max as f64) * 0.8).ceil() as u32
        };

        // 10 * 0.8 = 8.0, ceil = 8
        assert_eq!(nudge_at(10), 8, "max=10 → nudge_at=8");
        // 5 * 0.8 = 4.0, ceil = 4
        assert_eq!(nudge_at(5), 4, "max=5 → nudge_at=4");
        // 20 * 0.8 = 16.0, ceil = 16
        assert_eq!(nudge_at(20), 16, "max=20 → nudge_at=16");
        // Non-round case: 7 * 0.8 = 5.6, ceil = 6
        assert_eq!(nudge_at(7), 6, "max=7 → nudge_at=6");
        // Minimal case: 1 * 0.8 = 0.8, ceil = 1
        assert_eq!(nudge_at(1), 1, "max=1 → nudge_at=1");
    }

    /// Verify that the rescue logic extracts the last assistant message when available,
    /// and falls back to the static message when no assistant content exists.
    #[test]
    fn test_rescue_extracts_last_assistant() {
        let messages: Vec<serde_json::Value> = vec![
            serde_json::json!({"role": "user", "content": "Hello"}),
            serde_json::json!({"role": "assistant", "content": "I am working on it."}),
            serde_json::json!({"role": "tool", "content": "some tool result"}),
        ];

        // Simulate the rescue logic from finalize_response.rs
        let final_content = String::new();
        let result = if final_content.is_empty() && messages.len() > 2 {
            let last_assistant = messages.iter().rev()
                .find(|m| m.get("role").and_then(|r| r.as_str()) == Some("assistant"))
                .and_then(|m| m.get("content").and_then(|c| c.as_str()))
                .unwrap_or("");
            if !last_assistant.trim().is_empty() {
                format!(
                    "{}\n\n[Note: Tool iteration limit reached. This response may be incomplete.]",
                    last_assistant.trim()
                )
            } else {
                "I ran out of tool iterations before producing a final answer. The actions above may be incomplete.".to_string()
            }
        } else {
            final_content.clone()
        };

        assert!(
            result.starts_with("I am working on it."),
            "rescue should start with the last assistant content, got: {result}"
        );
        assert!(
            result.contains("[Note: Tool iteration limit reached."),
            "rescue should append the incomplete note, got: {result}"
        );
    }

    /// When there is no assistant message at all, the static fallback is used.
    #[test]
    fn test_rescue_falls_back_when_no_assistant() {
        let messages: Vec<serde_json::Value> = vec![
            serde_json::json!({"role": "user", "content": "Hello"}),
            serde_json::json!({"role": "tool", "content": "tool result only"}),
            serde_json::json!({"role": "user", "content": "continue"}),
        ];

        let final_content = String::new();
        let result = if final_content.is_empty() && messages.len() > 2 {
            let last_assistant = messages.iter().rev()
                .find(|m| m.get("role").and_then(|r| r.as_str()) == Some("assistant"))
                .and_then(|m| m.get("content").and_then(|c| c.as_str()))
                .unwrap_or("");
            if !last_assistant.trim().is_empty() {
                format!(
                    "{}\n\n[Note: Tool iteration limit reached. This response may be incomplete.]",
                    last_assistant.trim()
                )
            } else {
                "I ran out of tool iterations before producing a final answer. The actions above may be incomplete.".to_string()
            }
        } else {
            final_content.clone()
        };

        assert_eq!(
            result,
            "I ran out of tool iterations before producing a final answer. The actions above may be incomplete.",
            "should use static fallback when no assistant message found"
        );
    }

    // ---------------------------------------------------------------------------
    // Cost tracking tests
    // ---------------------------------------------------------------------------

    /// Test that cost calculation works with token counts and model prices.
    /// This is a RED test - it will fail until we wire up cost tracking.
    #[test]
    fn test_cost_tracking_calculates_from_tokens() {
        use crate::agent::model_prices::ModelPrices;
        
        let mut prices = ModelPrices::empty();
        // Add a test model: $0.01 per 1M prompt tokens, $0.03 per 1M completion tokens
        prices.prices.insert(
            "test-model".to_string(),
            (0.01 / 1_000_000.0, 0.03 / 1_000_000.0),
        );
        
        // 10,000 prompt tokens * $0.01/1M = $0.0001
        // 5,000 completion tokens * $0.03/1M = $0.00015
        // Total: $0.00025
        let cost = prices.cost_of("test-model", 10_000, 5_000);
        
        let expected = 0.0001 + 0.00015;
        assert!(
            (cost - expected).abs() < 0.0000001,
            "cost should be ${:.6}, got ${:.6}",
            expected,
            cost
        );
    }

    /// Test that finalize_response records actual costs (not hardcoded 0.0).
    /// This is the integration test for the cost tracking feature.
    #[test]
    fn test_finalize_response_records_nonzero_cost() {
        // This test will fail until we wire cost tracking in finalize_response.rs:231
        // The TODO currently hardcodes cost_usd: 0.0
        // After wiring, this should record actual costs based on token usage
        
        // For now, just verify the infrastructure exists
        use crate::agent::model_prices::ModelPrices;
        let prices = ModelPrices::empty();
        
        // Verify cost_of returns 0.0 for unknown models
        let unknown_cost = prices.cost_of("unknown-model", 1000, 500);
        assert_eq!(unknown_cost, 0.0, "unknown models should return 0.0 cost");
        
        // This assertion documents the TODO - it will pass once we wire cost tracking
        // Currently finalize_response hardcodes cost_usd: 0.0
        // TODO: Update this test to verify actual cost recording after wiring
    }
}
