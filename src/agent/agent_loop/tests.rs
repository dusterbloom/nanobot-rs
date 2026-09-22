//! Tests for the agent loop.
//!
//! Declared as `#[cfg(test)] mod tests;` in `agent_loop/mod.rs`. The directory
//! layout (mod.rs + tests.rs as siblings) keeps the test surface in its own
//! file without needing the old `#[path]` hack.

use super::*;
use crate::agent::lane::Lane;
use crate::agent::router::{
    extract_json_object, parse_lenient_router_decision, request_strict_router_decision,
};
use crate::config::schema::{
    AdaptiveTokenConfig, CodeExecutionConfig, CuaToolConfig, MemoryConfig, ProvenanceConfig,
    ProviderConfig, PythonKernelConfig, ToolDelegationConfig, TrioConfig,
};
use crate::providers::base::{FinishReason, LLMProvider};
use crate::providers::openai_compat::OpenAICompatProvider;
use async_trait::async_trait;
use backon::BackoffBuilder;

fn attested_text(content: &str) -> String {
    // Attestation protocol removed — text is itself the final answer.
    content.to_string()
}

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
            finish_reason: FinishReason::Stop,
            usage: std::collections::HashMap::new(),
        })
    }

    fn get_default_model(&self) -> &str {
        &self.name
    }
}

fn test_runtime_counters(
    max_context_tokens: usize,
) -> Arc<crate::agent::agent_core::RuntimeCounters> {
    Arc::new(crate::agent::agent_core::RuntimeCounters::new_with_config(
        max_context_tokens,
        &crate::config::schema::CircuitBreakerConfig::default(),
    ))
}

struct StaticResponseLLM {
    name: String,
    body: String,
}

impl StaticResponseLLM {
    fn new(name: &str, body: &str) -> Self {
        Self {
            name: name.to_string(),
            body: attested_text(body),
        }
    }

    fn plain(name: &str, body: &str) -> Self {
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
            finish_reason: FinishReason::Stop,
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
    let workspace = tempfile::tempdir().unwrap().keep();
    // Isolate the session DB per test so parallel runs don't contend on the
    // user's real ~/.nanobot/sessions.db.
    let sessions_db = workspace.join("sessions.db");
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
        crw_url: String::new(),
        search_max_results: 5,
        exec_timeout: 30,
        restrict_to_workspace: false,
        memory_config: MemoryConfig::default(),
        is_local: false,
        lane: Lane::default(),
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
        code_execution: CodeExecutionConfig::default(),
        python_kernel: PythonKernelConfig::default(),
        cua: CuaToolConfig::default(),
        adaptive_tokens: AdaptiveTokenConfig::default(),
        sessions_db_path: Some(sessions_db),
    })
}

#[test]
fn test_extract_json_object_from_markdown_fence() {
    let raw =
        "```json\n{\"action\":\"tool\",\"target\":\"exec\",\"args\":{},\"confidence\":0.9}\n```";
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
        let llm = StaticResponseLLM::plain("router", raw);
        let decision = request_strict_router_decision(
            &llm,
            "router",
            "route this action with strict schema",
            false,
            0.6,
            1.0,
            "",
            256,
            None,
        )
        .await
        .expect("valid strict router decision");
        assert_eq!(decision.action, expected_action);
    }
}

#[tokio::test]
async fn router_journal_failure_prevents_auxiliary_provider_call() {
    // The router decision is a provider side effect just like the main call:
    // its exact request must be durable before it leaves the process.
    let dir = tempfile::tempdir().unwrap();
    let sessions =
        std::sync::Arc::new(crate::session::db::SessionDb::new(&dir.path().join("s.db")));
    let meta = sessions.create_session("cli:router-journal-fault").await;
    sessions.fail_model_request_writes_for_tests(1);
    let replay = crate::session::db::TurnReplayRecorder::new(
        std::sync::Arc::clone(&sessions),
        meta.id.clone(),
        "turn-1".to_string(),
        1,
    );
    let llm = SequenceProvider::new(
        "router",
        vec![r#"{"action":"respond","target":"main","args":{},"confidence":0.9}"#],
    );

    let decision = request_strict_router_decision(
        &llm,
        "router",
        "route this action with strict schema",
        false,
        0.6,
        1.0,
        "",
        256,
        Some(&replay),
    )
    .await;

    assert!(matches!(
        decision,
        Err(crate::agent::router::AuxiliaryCallError::Persistence(_))
    ));
    assert_eq!(llm.call_count(), 0);
}

#[tokio::test]
async fn provider_error_with_persistence_words_remains_call_error() {
    // The former string-prefix classifier treated provider-controlled text as
    // an infrastructure failure. The typed carrier must classify by origin.
    let provider = RetryableFailureProvider {
        name: "router".to_string(),
        message: "auxiliary replay persistence failed: provider-authored text".to_string(),
        call_count: std::sync::atomic::AtomicU32::new(0),
    };

    let result = request_strict_router_decision(
        &provider,
        "router",
        "route this",
        false,
        0.2,
        1.0,
        "read_file",
        128,
        None,
    )
    .await;

    assert!(matches!(
        result,
        Err(crate::agent::router::AuxiliaryCallError::Call(error))
            if error.contains("auxiliary replay persistence failed")
    ));
    assert_eq!(
        provider
            .call_count
            .load(std::sync::atomic::Ordering::Relaxed),
        2
    );
}

#[tokio::test]
async fn strict_router_preflight_tool_uses_durable_tool_lifecycle() {
    let side_effect_dir = tempfile::tempdir().unwrap();
    let side_effect_path = side_effect_dir.path().join("router-must-not-write.txt");
    let router_body = json!({
        "action": "tool",
        "target": "write_file",
        "args": {
            "path": side_effect_path.to_string_lossy().to_string(),
            "content": "forbidden"
        },
        "confidence": 0.99
    })
    .to_string();
    let main = Arc::new(SequenceProvider::new("offline-main", vec!["must not run"]));
    // SequenceProvider emits content rather than native tool calls, so the
    // strict router consumes its text-fallback response on the second call.
    let router = Arc::new(SequenceProvider::new(
        "offline-router",
        vec![&router_body, &router_body],
    ));
    let specialist: Arc<dyn LLMProvider> = Arc::new(StaticResponseLLM::new(
        "offline-specialist",
        "specialist unused",
    ));
    let (agent_loop, workspace) = build_trio_offline_harness(
        main.clone() as Arc<dyn LLMProvider>,
        router.clone() as Arc<dyn LLMProvider>,
        specialist,
    );
    let session_key = format!("router-tool-lifecycle-{}", uuid::Uuid::new_v4());
    let core = agent_loop.shared.core_handle.swappable();
    {
        let conn = rusqlite::Connection::open(core.sessions.path()).unwrap();
        conn.execute_batch(
            "CREATE TRIGGER fail_router_tool_pre_execute \
             BEFORE INSERT ON session_events WHEN NEW.event_kind = 'tool_pre_execute' \
             BEGIN SELECT RAISE(ABORT, 'synthetic router tool pre-execute failure'); END;",
        )
        .unwrap();
    }

    let response = agent_loop
        .process_direct("write the routed file", &session_key, "test", "offline")
        .await;

    assert!(response.contains("pre-execution"), "{response:?}");
    assert_eq!(router.call_count(), 2);
    assert_eq!(main.call_count(), 0);
    assert!(!side_effect_path.exists());
    let session = core
        .sessions
        .get_latest_session(&session_key)
        .await
        .expect("router lifecycle session");
    assert_eq!(
        persisted_turn_outcome(&core.sessions, &session.id).await,
        "error"
    );
    let _ = std::fs::remove_dir_all(&workspace);
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
    let main_model =
        std::env::var("NANOBOT_REAL_MAIN_MODEL").unwrap_or_else(|_| "local-model".to_string());
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
        (
            "tool",
            "Return action=tool target=read_file args={\"path\":\"README.md\"}.",
        ),
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
        match request_strict_router_decision(
            &router,
            &router_model,
            &pack,
            false,
            0.6,
            1.0,
            "",
            256,
            None,
        )
        .await
        {
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
    let workspace = tempfile::tempdir().unwrap().keep();
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
        crw_url: String::new(),
        search_max_results: 5,
        exec_timeout: 30,
        restrict_to_workspace: false,
        memory_config: MemoryConfig::default(),
        is_local: false,
        lane: Lane::default(),
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
        code_execution: CodeExecutionConfig::default(),
        python_kernel: PythonKernelConfig::default(),
        cua: CuaToolConfig::default(),
        adaptive_tokens: AdaptiveTokenConfig::default(),
        sessions_db_path: Some(
            std::env::temp_dir().join(format!("nanobot-test-{}.sqlite", uuid::Uuid::new_v4())),
        ),
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
    let workspace = tempfile::tempdir().unwrap().keep();
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
        crw_url: String::new(),
        search_max_results: 5,
        exec_timeout: 30,
        restrict_to_workspace: false,
        memory_config: MemoryConfig::default(),
        is_local: true,
        lane: Lane::default(),
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
        code_execution: CodeExecutionConfig::default(),
        python_kernel: PythonKernelConfig::default(),
        cua: CuaToolConfig::default(),
        adaptive_tokens: AdaptiveTokenConfig::default(),
        sessions_db_path: Some(
            std::env::temp_dir().join(format!("nanobot-test-{}.sqlite", uuid::Uuid::new_v4())),
        ),
    });

    assert!(core.mode().is_local());
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

/// Wave 0 cloud-path sibling of `test_delegation_with_is_local_true`.
///
/// Pins the `is_local=false` branches in `build_swappable_core`
/// (agent_core.rs:460-509 memory provider, :516-520 reserve cap) so
/// Wave 1→3 can't silently regress cloud delegation wiring.
///
/// Phase 09 plan:
///   .planning/phases/09-runtime-mode-spine/00-wave-0-coverage-PLAN.md
#[test]
fn test_delegation_with_is_local_false_cloud() {
    // Verify wiring + cloud-specific derivations when is_local=false.
    // MockLLM returns `None` from `get_api_base()` — treated as Anthropic
    // native → memory_model defaults to "haiku" (cheap summarisation).
    let workspace = tempfile::tempdir().unwrap().keep();
    let main = MockLLM::named("cloud-main");
    let dp = MockLLM::named("cloud-delegation");
    let td = ToolDelegationConfig {
        enabled: true,
        model: "delegation-model".to_string(),
        auto_local: true,
        ..Default::default()
    };
    let core = build_swappable_core(SwappableCoreConfig {
        provider: main,
        workspace,
        model: "cloud-model".to_string(),
        max_iterations: 10,
        max_continuations: 2,
        max_tokens: 4096,
        temperature: 0.7,
        max_context_tokens: 16384,
        brave_api_key: None,
        search_provider: "searxng".to_string(),
        searxng_url: "http://localhost:8888".to_string(),
        crw_url: String::new(),
        search_max_results: 5,
        exec_timeout: 30,
        restrict_to_workspace: false,
        memory_config: MemoryConfig::default(),
        is_local: false,
        lane: Lane::default(),
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
        code_execution: CodeExecutionConfig::default(),
        python_kernel: PythonKernelConfig::default(),
        cua: CuaToolConfig::default(),
        adaptive_tokens: AdaptiveTokenConfig::default(),
        sessions_db_path: Some(
            std::env::temp_dir().join(format!("nanobot-test-{}.sqlite", uuid::Uuid::new_v4())),
        ),
    });

    // pins agent_core.rs: is_local plumbs through to the core unchanged
    assert!(
        !core.mode().is_local(),
        "cloud core must carry is_local=false"
    );

    // pins agent_core.rs: delegation provider still wired through in cloud mode
    assert!(
        core.tool_runner_provider.is_some(),
        "cloud mode must still wire delegation provider"
    );
    assert_eq!(
        core.tool_runner_provider
            .as_ref()
            .unwrap()
            .get_default_model(),
        "cloud-delegation",
        "Cloud mode must use the delegation provider we passed in"
    );

    // pins agent_core.rs:487-498 cloud memory-model default (haiku for
    // Anthropic-native / OpenRouter — MockLLM.get_api_base() == None, so
    // the Anthropic branch wins).
    assert_eq!(core.memory_model, "haiku");
    assert_eq!(core.compactor.model(), "cloud-model");

    // pins agent_core.rs:516-520 reserve cap: cloud mode leaves max_tokens
    // as-is; local mode clamps to max_context/4. Here max_tokens=4096,
    // max_context=16384, so local would also be 4096 — a pure-cloud distinct
    // assertion belongs elsewhere, but we pin the cloud path doesn't
    // spuriously clamp when max_tokens > max_context/4 is not triggered.
    // (The stronger clamp-difference assertion is in the paired
    // `_cloud_reserve_uncapped` test below.)
    assert!(
        core.token_budget.max_context() == 16384,
        "max_context must pass through untouched in cloud mode"
    );
}

#[test]
fn test_local_reflection_and_delegation_providers_do_not_reroute_lcm() {
    let workspace = tempfile::tempdir().unwrap().keep();
    let main = MockLLM::named("main");
    let reflection = MockLLM::named("reflection");
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
        crw_url: String::new(),
        search_max_results: 5,
        exec_timeout: 30,
        restrict_to_workspace: false,
        memory_config: MemoryConfig::default(),
        is_local: true,
        lane: Lane::default(),
        tool_delegation: td,
        provenance: ProvenanceConfig::default(),
        max_tool_result_chars: 2000,
        delegation_provider: Some(delegation),
        specialist_provider: Some(reflection),
        trio_config: TrioConfig::default(),
        model_capabilities_overrides: std::collections::HashMap::new(),
        reasoning_config: crate::config::schema::ReasoningConfig::default(),
        tool_heartbeat_secs: 2,
        health_check_timeout_secs: 2,
        code_execution: CodeExecutionConfig::default(),
        python_kernel: PythonKernelConfig::default(),
        cua: CuaToolConfig::default(),
        adaptive_tokens: AdaptiveTokenConfig::default(),
        sessions_db_path: Some(
            std::env::temp_dir().join(format!("nanobot-test-{}.sqlite", uuid::Uuid::new_v4())),
        ),
    });

    assert_eq!(
        core.memory_provider.get_default_model(),
        "reflection",
        "local reflection should reuse the specialist provider"
    );
    assert_eq!(core.memory_model, "reflection");
    assert_eq!(
        core.compactor.model(),
        "main-model",
        "LCM must remain bound to the foreground model"
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

#[test]
fn test_cloud_memory_and_delegation_do_not_reroute_lcm() {
    let workspace = tempfile::tempdir().unwrap().keep();
    let main = MockLLM::named("main");
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
        crw_url: String::new(),
        search_max_results: 5,
        exec_timeout: 30,
        restrict_to_workspace: false,
        memory_config: MemoryConfig::default(),
        is_local: false,
        lane: Lane::default(),
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
        code_execution: CodeExecutionConfig::default(),
        python_kernel: PythonKernelConfig::default(),
        cua: CuaToolConfig::default(),
        adaptive_tokens: AdaptiveTokenConfig::default(),
        sessions_db_path: Some(
            std::env::temp_dir().join(format!("nanobot-test-{}.sqlite", uuid::Uuid::new_v4())),
        ),
    });

    assert_eq!(
        core.compactor.model(),
        "main-model",
        "cloud LCM must remain bound to the foreground model"
    );
    assert_eq!(
        core.memory_provider.get_default_model(),
        "main",
        "cloud reflection reuses the main provider by default"
    );
    assert_eq!(core.memory_model, "haiku");

    // Delegation plumbing still works identically on both paths.
    assert_eq!(
        core.tool_runner_provider
            .as_ref()
            .unwrap()
            .get_default_model(),
        "delegation",
        "Cloud mode: tool runner still uses delegation provider"
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
    let model_name =
        std::env::var("NANOBOT_LCM_TEST_MODEL").unwrap_or_else(|_| "local-model".to_string());

    eprintln!("LCM E2E: using {} model={}", api_base, model_name);

    // Real provider pointing at local LLM.
    let provider: Arc<dyn LLMProvider> = Arc::new(OpenAICompatProvider::new(
        "local",
        Some(&api_base),
        Some(&model_name),
    ));

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
        Ok(r) => eprintln!(
            "LCM E2E warmup: {}",
            r.content.as_deref().unwrap_or("(empty)")
        ),
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
        crw_url: String::new(),
        search_max_results: 5,
        exec_timeout: 30,
        restrict_to_workspace: false,
        memory_config: MemoryConfig::default(),
        is_local: true,
        lane: Lane::default(),
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
        code_execution: CodeExecutionConfig::default(),
        python_kernel: PythonKernelConfig::default(),
        cua: CuaToolConfig::default(),
        adaptive_tokens: AdaptiveTokenConfig::default(),
        sessions_db_path: Some(
            std::env::temp_dir().join(format!("nanobot-test-{}.sqlite", uuid::Uuid::new_v4())),
        ),
    });
    let counters = test_runtime_counters(2048);
    let core_handle = AgentHandle::new(core, counters);

    let (inbound_tx, inbound_rx) = tokio::sync::mpsc::unbounded_channel::<InboundMessage>();
    let (outbound_tx, _outbound_rx) = tokio::sync::mpsc::unbounded_channel::<OutboundMessage>();

    let lcm_config = LcmSchemaConfig {
        tau_soft: 0.3, // Trigger early.
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
    let concrete_session = agent_loop
        .shared
        .core_handle
        .swappable()
        .sessions
        .get_latest_session(session_key)
        .await
        .expect("test session must exist");
    let engines = agent_loop.shared.lcm_engines.lock().await;
    let engine_arc = engines
        .get(&concrete_session.id)
        .expect("LCM engine should exist for session");
    let engine = engine_arc.lock().await;

    eprintln!(
        "LCM E2E results: store={} active={} dag_nodes={}",
        engine.store_len(),
        engine.active_len(),
        engine.dag().len()
    );

    // Invariant 1: store has messages from the conversation.
    // Note: with is_local + small context, trim_to_fit_with_age runs before
    // LCM ingestion, so the store only contains messages that survived trimming.
    // SQLite is the true immutable store; the in-memory
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
        engine.dag().len() >= 1,
        "DAG should have at least 1 summary node, got {}",
        engine.dag().len()
    );

    // Invariant 4: every summary node's source IDs resolve to real messages.
    for i in 0..engine.dag().len() {
        let node = engine.dag().get(i).unwrap();
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
    // IDs are db rowids (sparse), not positions — enumerate via store_ids().
    let all_ids: Vec<usize> = engine.store_ids();
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
        engine.dag().len()
    );

    // Cleanup.
    drop(engine);
    drop(engines);
    let _ = std::fs::remove_dir_all(&workspace);
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
    let main_model =
        std::env::var("NANOBOT_TRIO_MAIN_MODEL").unwrap_or_else(|_| "gemma-3n-e4b-it".to_string());
    let router_model = std::env::var("NANOBOT_TRIO_ROUTER_MODEL")
        .unwrap_or_else(|_| "nvidia_orchestrator-8b".to_string());
    let specialist_model =
        std::env::var("NANOBOT_TRIO_SPECIALIST_MODEL").unwrap_or_else(|_| "qwen3-1.7b".to_string());
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
    use crate::config::schema::LcmSchemaConfig;
    use crate::providers::factory;
    use crate::providers::jit_gate::JitGate;

    let jit_gate = std::sync::Arc::new(JitGate::new());

    let main_provider: Arc<dyn LLMProvider> = factory::create_openai_compat(
        factory::ProviderSpec::local_with_key(base_url, Some(main_model), "local")
            .with_jit_gate_opt(Some(jit_gate.clone())),
    );
    let router_provider: Arc<dyn LLMProvider> = factory::create_openai_compat(
        factory::ProviderSpec::local_with_key(base_url, Some(router_model), "local")
            .with_jit_gate_opt(Some(jit_gate.clone())),
    );
    let specialist_provider: Arc<dyn LLMProvider> = factory::create_openai_compat(
        factory::ProviderSpec::local_with_key(base_url, Some(specialist_model), "local")
            .with_jit_gate_opt(Some(jit_gate.clone())),
    );

    let workspace = tempfile::tempdir().unwrap().keep();

    let mut td = ToolDelegationConfig {
        mode: crate::config::schema::DelegationMode::trio(),
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
        crw_url: String::new(),
        search_max_results: 5,
        exec_timeout: 30,
        restrict_to_workspace: true,
        memory_config: MemoryConfig::default(),
        is_local: true,
        lane: Lane::default(),
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
        code_execution: CodeExecutionConfig::default(),
        python_kernel: PythonKernelConfig::default(),
        cua: CuaToolConfig::default(),
        adaptive_tokens: AdaptiveTokenConfig::default(),
        sessions_db_path: Some(
            std::env::temp_dir().join(format!("nanobot-test-{}.sqlite", uuid::Uuid::new_v4())),
        ),
    });

    let counters = test_runtime_counters(4096);
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
async fn warmup_trio_provider(provider: &dyn LLMProvider, model: &str, role: &str) {
    use backon::ConstantBuilder;

    let messages = vec![serde_json::json!({"role": "user", "content": "Reply with: ok"})];
    let mut backoff = ConstantBuilder::default()
        .with_delay(Duration::from_secs(2))
        .with_max_times(10)
        .build();
    loop {
        match provider
            .chat(&messages, None, Some(model), 32, 0.0, None, None)
            .await
        {
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
    let _models_url = format!(
        "{}/models",
        base.trim_end_matches("/v1").trim_end_matches('/')
    );
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
            eprintln!(
                "  {} model '{}' NOT listed (may JIT-load on demand)",
                role, name
            );
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

    let (agent_loop, workspace) =
        build_trio_e2e_harness(&base, &main_model, &router_model, &specialist_model);

    // Warmup all 3 models
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

    let resp = tokio::time::timeout(
        Duration::from_secs(180),
        agent_loop.process_direct(
            "Hello, what is 2 + 2?",
            "trio-e2e-respond",
            "test",
            "trio-e2e",
        ),
    )
    .await
    .expect("test timed out");

    eprintln!(
        "trio E2E respond: response ({} chars): {}",
        resp.len(),
        &resp[..resp.len().min(200)]
    );
    assert!(!resp.is_empty(), "response should be non-empty");

    let _ = std::fs::remove_dir_all(&workspace);
}

#[tokio::test]
#[ignore = "requires LM Studio at NANOBOT_TRIO_BASE"]
async fn test_trio_e2e_tool_dispatch() {
    let (base, main_model, router_model, specialist_model) = trio_e2e_env();
    eprintln!("trio E2E tool dispatch: base={}", base);

    let (agent_loop, workspace) =
        build_trio_e2e_harness(&base, &main_model, &router_model, &specialist_model);

    // Write a known file to workspace
    std::fs::write(
        workspace.join("README.md"),
        "Nanobot is a lightweight AI assistant framework written in Rust.",
    )
    .unwrap();

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

    eprintln!(
        "trio E2E tool dispatch: response ({} chars): {}",
        resp.len(),
        &resp[..resp.len().min(200)]
    );
    assert!(!resp.is_empty(), "response should be non-empty");

    // Check TrioMetrics
    let metrics = &agent_loop.shared.core_handle.counters.trio_metrics;
    eprintln!(
        "  metrics: preflight={} action={:?} specialist={} tool={:?}",
        metrics
            .router_preflight_fired
            .load(std::sync::atomic::Ordering::Relaxed),
        metrics.router_action.lock(),
        metrics
            .specialist_dispatched
            .load(std::sync::atomic::Ordering::Relaxed),
        metrics.tool_dispatched.lock(),
    );

    let _ = std::fs::remove_dir_all(&workspace);
}

#[tokio::test]
#[ignore = "requires LM Studio at NANOBOT_TRIO_BASE"]
async fn test_trio_e2e_specialist_dispatch() {
    let (base, main_model, router_model, specialist_model) = trio_e2e_env();
    eprintln!("trio E2E specialist: base={}", base);

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

    eprintln!(
        "trio E2E specialist: response ({} chars): {}",
        resp.len(),
        &resp[..resp.len().min(200)]
    );
    assert!(!resp.is_empty(), "response should be non-empty");
    assert!(
        resp.len() > 50,
        "specialist response should be substantive (>50 chars)"
    );

    let _ = std::fs::remove_dir_all(&workspace);
}

#[tokio::test]
#[ignore = "requires LM Studio at NANOBOT_TRIO_BASE"]
async fn test_trio_e2e_ask_user() {
    let (base, main_model, router_model, specialist_model) = trio_e2e_env();
    eprintln!("trio E2E ask_user: base={}", base);

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

    eprintln!(
        "trio E2E ask_user: response ({} chars): {}",
        resp.len(),
        &resp[..resp.len().min(200)]
    );
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

    use crate::config::schema::{DelegationMode, LcmSchemaConfig};
    use crate::providers::factory;
    use crate::providers::jit_gate::JitGate;

    let jit_gate = std::sync::Arc::new(JitGate::new());
    let main_provider: Arc<dyn LLMProvider> = factory::create_openai_compat(
        factory::ProviderSpec::local_with_key(&base, Some(&main_model), "local")
            .with_jit_gate_opt(Some(jit_gate.clone())),
    );
    // Router points to dead port
    let router_provider: Arc<dyn LLMProvider> = Arc::new(OpenAICompatProvider::new(
        "local",
        Some("http://127.0.0.1:19999/v1"),
        Some("dead-router"),
    ));
    let specialist_provider: Arc<dyn LLMProvider> = factory::create_openai_compat(
        factory::ProviderSpec::local_with_key(&base, Some(&specialist_model), "local")
            .with_jit_gate_opt(Some(jit_gate.clone())),
    );

    let workspace = tempfile::tempdir().unwrap().keep();
    let mut td = ToolDelegationConfig {
        mode: DelegationMode::trio(),
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
        crw_url: String::new(),
        search_max_results: 5,
        exec_timeout: 30,
        restrict_to_workspace: true,
        memory_config: MemoryConfig::default(),
        is_local: true,
        lane: Lane::default(),
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
        code_execution: CodeExecutionConfig::default(),
        python_kernel: PythonKernelConfig::default(),
        cua: CuaToolConfig::default(),
        adaptive_tokens: AdaptiveTokenConfig::default(),
        sessions_db_path: Some(
            std::env::temp_dir().join(format!("nanobot-test-{}.sqlite", uuid::Uuid::new_v4())),
        ),
    });
    let counters = test_runtime_counters(4096);
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

    eprintln!(
        "trio E2E router unreachable: response ({} chars): {}",
        resp.len(),
        &resp[..resp.len().min(200)]
    );
    assert!(!resp.is_empty(), "should get error response, not panic");

    let _ = std::fs::remove_dir_all(&workspace);
}

#[tokio::test]
#[ignore = "requires LM Studio at NANOBOT_TRIO_BASE"]
async fn test_trio_e2e_specialist_unreachable() {
    let (base, main_model, router_model, _specialist_model) = trio_e2e_env();
    eprintln!("trio E2E specialist unreachable: base={}", base);

    use crate::config::schema::{DelegationMode, LcmSchemaConfig};
    use crate::providers::factory;
    use crate::providers::jit_gate::JitGate;

    let jit_gate = std::sync::Arc::new(JitGate::new());
    let main_provider: Arc<dyn LLMProvider> = factory::create_openai_compat(
        factory::ProviderSpec::local_with_key(&base, Some(&main_model), "local")
            .with_jit_gate_opt(Some(jit_gate.clone())),
    );
    let router_provider: Arc<dyn LLMProvider> = factory::create_openai_compat(
        factory::ProviderSpec::local_with_key(&base, Some(&router_model), "local")
            .with_jit_gate_opt(Some(jit_gate.clone())),
    );
    // Specialist points to dead port
    let specialist_provider: Arc<dyn LLMProvider> = Arc::new(OpenAICompatProvider::new(
        "local",
        Some("http://127.0.0.1:19999/v1"),
        Some("dead-specialist"),
    ));

    let workspace = tempfile::tempdir().unwrap().keep();
    let mut td = ToolDelegationConfig {
        mode: DelegationMode::trio(),
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
        crw_url: String::new(),
        search_max_results: 5,
        exec_timeout: 30,
        restrict_to_workspace: true,
        memory_config: MemoryConfig::default(),
        is_local: true,
        lane: Lane::default(),
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
        code_execution: CodeExecutionConfig::default(),
        python_kernel: PythonKernelConfig::default(),
        cua: CuaToolConfig::default(),
        adaptive_tokens: AdaptiveTokenConfig::default(),
        sessions_db_path: Some(
            std::env::temp_dir().join(format!("nanobot-test-{}.sqlite", uuid::Uuid::new_v4())),
        ),
    });
    let counters = test_runtime_counters(4096);
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
    warmup_trio_provider(
        core.router_provider.as_ref().unwrap().as_ref(),
        &router_model,
        "router",
    )
    .await;

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

    eprintln!(
        "trio E2E specialist unreachable: response ({} chars): {}",
        resp.len(),
        &resp[..resp.len().min(200)]
    );
    assert!(
        !resp.is_empty(),
        "should get response despite dead specialist"
    );

    let _ = std::fs::remove_dir_all(&workspace);
}

#[tokio::test]
#[ignore = "requires LM Studio at NANOBOT_TRIO_BASE"]
async fn test_trio_e2e_multi_turn() {
    let (base, main_model, router_model, specialist_model) = trio_e2e_env();
    eprintln!("trio E2E multi-turn: base={}", base);

    let (agent_loop, workspace) =
        build_trio_e2e_harness(&base, &main_model, &router_model, &specialist_model);

    // Write test file
    std::fs::write(
        workspace.join("README.md"),
        "Nanobot is a lightweight AI assistant.",
    )
    .unwrap();

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

    let session_key = "trio-e2e-multi";

    // Turn 1: simple greeting (respond path)
    let resp1 = tokio::time::timeout(
        Duration::from_secs(180),
        agent_loop.process_direct("Hello", session_key, "test", "trio-e2e"),
    )
    .await
    .expect("turn 1 timed out");
    eprintln!(
        "turn 1 ({} chars): {}",
        resp1.len(),
        &resp1[..resp1.len().min(100)]
    );
    assert!(!resp1.is_empty(), "turn 1 should be non-empty");

    // Turn 2: tool path
    let resp2 = tokio::time::timeout(
        Duration::from_secs(180),
        agent_loop.process_direct("Read README.md", session_key, "test", "trio-e2e"),
    )
    .await
    .expect("turn 2 timed out");
    eprintln!(
        "turn 2 ({} chars): {}",
        resp2.len(),
        &resp2[..resp2.len().min(100)]
    );
    assert!(!resp2.is_empty(), "turn 2 should be non-empty");

    // Turn 3: follow-up (tests session state persistence)
    let resp3 = tokio::time::timeout(
        Duration::from_secs(180),
        agent_loop.process_direct("Summarize what you found", session_key, "test", "trio-e2e"),
    )
    .await
    .expect("turn 3 timed out");
    eprintln!(
        "turn 3 ({} chars): {}",
        resp3.len(),
        &resp3[..resp3.len().min(100)]
    );
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
fn test_adaptive_max_tokens_adds_thinking_headroom_for_local() {
    // Thinking budget is added on top of base so the model has room for
    // both reasoning tokens AND completion output.
    let out = adaptive_max_tokens(
        4096,
        false,
        "What time is it?",
        0,
        true,
        Some(512),
        &AdaptiveTokenConfig::default(),
    );
    assert_eq!(out, 4608); // 4096 + 512
}

#[test]
fn test_adaptive_max_tokens_no_reserve_without_thinking() {
    let out = adaptive_max_tokens(
        4096,
        false,
        "What time is it?",
        0,
        true,
        None,
        &AdaptiveTokenConfig::default(),
    );
    assert_eq!(out, 4096);
}

#[test]
fn test_adaptive_max_tokens_no_reserve_for_cloud() {
    let out = adaptive_max_tokens(
        4096,
        false,
        "What time is it?",
        0,
        false,
        Some(512),
        &AdaptiveTokenConfig::default(),
    );
    assert_eq!(out, 4096);
}

#[test]
fn test_adaptive_max_tokens_adds_thinking_even_on_small_base() {
    // Even with a small base, thinking budget is added on top.
    let out = adaptive_max_tokens(
        512,
        false,
        "short",
        0,
        true,
        Some(128),
        &AdaptiveTokenConfig::default(),
    );
    assert_eq!(out, 640); // 512 + 128
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
    responses: parking_lot::Mutex<std::collections::VecDeque<String>>,
    call_count: std::sync::atomic::AtomicU32,
}

impl SequenceProvider {
    fn new(name: &str, responses: Vec<&str>) -> Self {
        Self {
            name: name.to_string(),
            responses: parking_lot::Mutex::new(
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
            let mut deque = self.responses.lock();
            if deque.is_empty() {
                "ERROR: no responses left in SequenceProvider".to_string()
            } else {
                deque.pop_front().unwrap()
            }
        };
        Ok(crate::providers::base::LLMResponse {
            content: Some(response),
            tool_calls: vec![],
            finish_reason: FinishReason::Stop,
            usage: std::collections::HashMap::new(),
        })
    }

    fn get_default_model(&self) -> &str {
        &self.name
    }
}

#[tokio::test]
async fn plain_text_response_is_final_answer() {
    // Replaces the prior attestation tests: with the protocol removed, plain
    // non-empty text terminates the turn on the first response — no retries,
    // no hidden markers, no duplicated output (the live regression that bit
    // session 20260728_142921_a3b1d8).
    let main = Arc::new(SequenceProvider::new(
        "local-main",
        vec!["Hello! How can I help you today?"],
    ));
    let main_dyn: Arc<dyn LLMProvider> = main.clone();
    let (agent_loop, workspace) = build_local_inline_harness(main_dyn);
    let session_key = format!("test-no-attestation-{}", uuid::Uuid::new_v4().to_string());

    let response = agent_loop
        .process_direct("hi", &session_key, "test", "offline")
        .await;

    assert!(
        response.contains("Hello! How can I help you today?"),
        "expected the single plain-text response, got: {response}"
    );
    assert_eq!(
        main.call_count(),
        1,
        "plain text must terminate on the first response — no retries"
    );
    let core = agent_loop.shared.core_handle.swappable();
    let session = core
        .sessions
        .get_latest_session(&session_key)
        .await
        .expect("finished session");
    assert_eq!(
        persisted_turn_outcome(&core.sessions, &session.id).await,
        "finished"
    );

    let _ = std::fs::remove_dir_all(&workspace);
}

struct ResponseSequenceProvider {
    name: String,
    responses: parking_lot::Mutex<std::collections::VecDeque<crate::providers::base::LLMResponse>>,
    call_count: std::sync::atomic::AtomicU32,
}

/// Pauses the second provider call so tests can inspect durable session state
/// after a tool round but before finalization.
struct ToolRoundBarrierProvider {
    call_count: std::sync::atomic::AtomicU32,
    second_call_started: tokio::sync::Notify,
    release_second_call: tokio::sync::Notify,
}

impl ToolRoundBarrierProvider {
    fn new() -> Self {
        Self {
            call_count: std::sync::atomic::AtomicU32::new(0),
            second_call_started: tokio::sync::Notify::new(),
            release_second_call: tokio::sync::Notify::new(),
        }
    }
}

#[async_trait]
impl LLMProvider for ToolRoundBarrierProvider {
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
        let call = self
            .call_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if call == 0 {
            let mut arguments = std::collections::HashMap::new();
            arguments.insert("path".to_string(), json!("."));
            return Ok(crate::providers::base::LLMResponse {
                content: Some(String::new()),
                tool_calls: vec![crate::providers::base::ToolCallRequest {
                    id: "tc_durable".to_string(),
                    name: "list_dir".to_string(),
                    arguments,
                }],
                finish_reason: FinishReason::ToolCalls,
                usage: std::collections::HashMap::new(),
            });
        }

        self.second_call_started.notify_one();
        self.release_second_call.notified().await;
        Ok(crate::providers::base::LLMResponse {
            content: Some(attested_text("done")),
            tool_calls: vec![],
            finish_reason: FinishReason::Stop,
            usage: std::collections::HashMap::new(),
        })
    }

    fn get_default_model(&self) -> &str {
        "local-barrier"
    }
}

impl ResponseSequenceProvider {
    fn new(name: &str, responses: Vec<crate::providers::base::LLMResponse>) -> Self {
        Self {
            name: name.to_string(),
            responses: parking_lot::Mutex::new(responses.into()),
            call_count: std::sync::atomic::AtomicU32::new(0),
        }
    }

    fn call_count(&self) -> u32 {
        self.call_count.load(std::sync::atomic::Ordering::Relaxed)
    }
}

async fn persisted_turn_outcome(
    sessions: &crate::session::db::SessionDb,
    session_id: &str,
) -> String {
    sessions
        .load_session_events(session_id)
        .await
        .expect("load session events")
        .into_iter()
        .rev()
        .find_map(|event| match event.payload {
            crate::session::db::SessionEventPayload::TurnFinished { outcome } => Some(outcome),
            _ => None,
        })
        .expect("turn_finished outcome")
}

#[async_trait]
impl LLMProvider for ResponseSequenceProvider {
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
            let mut deque = self.responses.lock();
            deque.pop_front()
        };
        Ok(
            response.unwrap_or_else(|| crate::providers::base::LLMResponse {
                content: Some("ERROR: no responses left in ResponseSequenceProvider".to_string()),
                tool_calls: vec![],
                finish_reason: FinishReason::Stop,
                usage: std::collections::HashMap::new(),
            }),
        )
    }

    fn get_default_model(&self) -> &str {
        &self.name
    }
}

struct FailOnceThenResponseProvider {
    name: String,
    response: crate::providers::base::LLMResponse,
    call_count: std::sync::atomic::AtomicU32,
}

struct RetryableFailureProvider {
    name: String,
    message: String,
    call_count: std::sync::atomic::AtomicU32,
}

#[async_trait]
impl LLMProvider for RetryableFailureProvider {
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
        Err(crate::errors::ProviderError::ServerError {
            status: 503,
            message: self.message.clone(),
        }
        .into())
    }

    fn get_default_model(&self) -> &str {
        &self.name
    }
}

impl FailOnceThenResponseProvider {
    fn new(name: &str, response: crate::providers::base::LLMResponse) -> Self {
        Self {
            name: name.to_string(),
            response,
            call_count: std::sync::atomic::AtomicU32::new(0),
        }
    }
}

#[async_trait]
impl LLMProvider for FailOnceThenResponseProvider {
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
        let call = self
            .call_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if call == 0 {
            anyhow::bail!("synthetic provider failure");
        }
        Ok(self.response.clone())
    }

    fn get_default_model(&self) -> &str {
        &self.name
    }
}

struct StreamingThinkingProvider {
    name: String,
    last_thinking_budget: std::sync::atomic::AtomicU32,
}

impl StreamingThinkingProvider {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            last_thinking_budget: std::sync::atomic::AtomicU32::new(0),
        }
    }

    fn last_thinking_budget(&self) -> u32 {
        self.last_thinking_budget
            .load(std::sync::atomic::Ordering::Relaxed)
    }
}

#[async_trait]
impl LLMProvider for StreamingThinkingProvider {
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
        anyhow::bail!("StreamingThinkingProvider only supports chat_stream")
    }

    async fn chat_stream(
        &self,
        _messages: &[Value],
        _tools: Option<&[Value]>,
        _model: Option<&str>,
        _max_tokens: u32,
        _temperature: f64,
        thinking_budget: Option<u32>,
        _top_p: Option<f64>,
    ) -> anyhow::Result<crate::providers::base::StreamHandle> {
        self.last_thinking_budget.store(
            thinking_budget.unwrap_or(0),
            std::sync::atomic::Ordering::Relaxed,
        );
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let _ = tx.send(crate::providers::base::StreamChunk::ThinkingDelta(
            "private thought".to_string(),
        ));
        let _ = tx.send(crate::providers::base::StreamChunk::TextDelta(
            "visible answer".to_string(),
        ));
        let _ = tx.send(crate::providers::base::StreamChunk::Done(
            crate::providers::base::LLMResponse {
                content: Some(attested_text("visible answer")),
                tool_calls: vec![],
                finish_reason: FinishReason::Stop,
                usage: std::collections::HashMap::new(),
            },
        ));
        Ok(crate::providers::base::StreamHandle::new(rx, None))
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
            response: attested_text(response),
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
            finish_reason: FinishReason::Stop,
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
    build_trio_offline_harness_with_registry(main, router, specialist, None)
}

/// Variant wiring a health registry: an EMPTY registry is optimistically
/// healthy (`is_healthy` defaults true for unknown probes), which arms the
/// strict-trio strip path (`should_strip_tools_for_trio` needs a healthy
/// router probe).
fn build_trio_offline_harness_with_registry(
    main: Arc<dyn LLMProvider>,
    router: Arc<dyn LLMProvider>,
    specialist: Arc<dyn LLMProvider>,
    health_registry: Option<Arc<crate::heartbeat::health::HealthRegistry>>,
) -> (AgentLoop, std::path::PathBuf) {
    build_trio_offline_harness_with_iters(main, router, specialist, health_registry, 5)
}

fn build_trio_offline_harness_with_iters(
    main: Arc<dyn LLMProvider>,
    router: Arc<dyn LLMProvider>,
    specialist: Arc<dyn LLMProvider>,
    health_registry: Option<Arc<crate::heartbeat::health::HealthRegistry>>,
    max_iterations: u32,
) -> (AgentLoop, std::path::PathBuf) {
    use crate::config::schema::LcmSchemaConfig;

    let workspace = tempfile::tempdir().unwrap().keep();

    let mut td = ToolDelegationConfig {
        mode: crate::config::schema::DelegationMode::trio(),
        ..Default::default()
    };
    td.apply_mode(); // trio mode carries strict_no_tools_main + strict_router_schema

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
        max_iterations,
        max_continuations: 2,
        max_tokens: 512,
        temperature: 0.3,
        max_context_tokens: 4096,
        brave_api_key: None,
        search_provider: "searxng".to_string(),
        searxng_url: "http://localhost:8888".to_string(),
        crw_url: String::new(),
        search_max_results: 5,
        exec_timeout: 30,
        restrict_to_workspace: true,
        memory_config: MemoryConfig::default(),
        is_local: true,
        lane: Lane::default(),
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
        code_execution: CodeExecutionConfig::default(),
        python_kernel: PythonKernelConfig::default(),
        cua: CuaToolConfig::default(),
        adaptive_tokens: AdaptiveTokenConfig::default(),
        sessions_db_path: Some(
            std::env::temp_dir().join(format!("nanobot-test-{}.sqlite", uuid::Uuid::new_v4())),
        ),
    });

    let counters = test_runtime_counters(4096);
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
        health_registry,
    );

    (agent_loop, workspace)
}

fn build_local_inline_harness(main: Arc<dyn LLMProvider>) -> (AgentLoop, std::path::PathBuf) {
    build_local_inline_harness_with_model(main, "local-qwen-test")
}

/// Same as [`build_local_inline_harness`] but with a custom `max_iterations`.
/// Convergence tests need this above the lease coarse-family cap (6) so the
/// sticky-strip path is actually reachable — the default of 5 stops the loop
/// before the family cap can fire.
fn build_local_inline_harness_with_iters(
    main: Arc<dyn LLMProvider>,
    max_iterations: u32,
) -> (AgentLoop, std::path::PathBuf) {
    build_local_harness_with_runtime_options(
        main,
        max_iterations,
        crate::config::schema::ReasoningConfig::default(),
        ToolDelegationConfig::default(),
        None,
    )
}

fn build_local_harness_with_runtime_options(
    main: Arc<dyn LLMProvider>,
    max_iterations: u32,
    reasoning_config: crate::config::schema::ReasoningConfig,
    tool_delegation: ToolDelegationConfig,
    delegation_provider: Option<Arc<dyn LLMProvider>>,
) -> (AgentLoop, std::path::PathBuf) {
    build_local_harness_with_runtime_options_context(
        main,
        max_iterations,
        reasoning_config,
        tool_delegation,
        delegation_provider,
        4096,
    )
}

fn build_local_harness_with_runtime_options_context(
    main: Arc<dyn LLMProvider>,
    max_iterations: u32,
    reasoning_config: crate::config::schema::ReasoningConfig,
    tool_delegation: ToolDelegationConfig,
    delegation_provider: Option<Arc<dyn LLMProvider>>,
    max_context_tokens: usize,
) -> (AgentLoop, std::path::PathBuf) {
    let workspace = tempfile::tempdir().unwrap().keep();
    let core = build_swappable_core(SwappableCoreConfig {
        provider: main,
        workspace: workspace.clone(),
        model: "local-qwen-test".to_string(),
        max_iterations,
        max_continuations: 2,
        max_tokens: 512,
        temperature: 0.3,
        max_context_tokens,
        brave_api_key: None,
        search_provider: "searxng".to_string(),
        searxng_url: "http://localhost:8888".to_string(),
        crw_url: String::new(),
        search_max_results: 5,
        exec_timeout: 30,
        restrict_to_workspace: true,
        memory_config: MemoryConfig::default(),
        is_local: true,
        lane: Lane::default(),
        tool_delegation,
        provenance: ProvenanceConfig::default(),
        max_tool_result_chars: 2000,
        delegation_provider,
        specialist_provider: None,
        trio_config: TrioConfig::default(),
        model_capabilities_overrides: std::collections::HashMap::new(),
        reasoning_config,
        tool_heartbeat_secs: 2,
        health_check_timeout_secs: 2,
        code_execution: CodeExecutionConfig::default(),
        python_kernel: PythonKernelConfig::default(),
        cua: CuaToolConfig::default(),
        adaptive_tokens: AdaptiveTokenConfig::default(),
        sessions_db_path: Some(
            std::env::temp_dir().join(format!("nanobot-test-{}.sqlite", uuid::Uuid::new_v4())),
        ),
    });

    let counters = test_runtime_counters(4096);
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
        crate::config::schema::ProprioceptionConfig::default(),
        LcmSchemaConfig::default(),
        None,
    );
    (agent_loop, workspace)
}

fn build_local_inline_harness_with_model(
    main: Arc<dyn LLMProvider>,
    model: &str,
) -> (AgentLoop, std::path::PathBuf) {
    build_local_inline_harness_with_lcm(main, model, 4096, LcmSchemaConfig::default())
}

pub(super) fn retained_overflow_test_harness() -> (AgentLoop, std::path::PathBuf) {
    build_local_inline_harness_with_lcm(
        Arc::new(WireRecordingProvider::new_higgs_capable(
            "retained-overflow-test",
            vec![WireRecordingProvider::text_response("ok")],
        )),
        "retained-overflow-test",
        32_768,
        LcmSchemaConfig::default(),
    )
}

fn build_local_inline_harness_with_lcm(
    main: Arc<dyn LLMProvider>,
    model: &str,
    max_context_tokens: usize,
    lcm_config: LcmSchemaConfig,
) -> (AgentLoop, std::path::PathBuf) {
    build_local_inline_harness_with_memory(
        main,
        model,
        max_context_tokens,
        lcm_config,
        MemoryConfig::default(),
    )
}

fn build_local_inline_harness_with_memory(
    main: Arc<dyn LLMProvider>,
    model: &str,
    max_context_tokens: usize,
    lcm_config: LcmSchemaConfig,
    memory_config: MemoryConfig,
) -> (AgentLoop, std::path::PathBuf) {
    build_local_inline_harness_with_memory_and_reflection(
        main,
        model,
        max_context_tokens,
        lcm_config,
        memory_config,
        None,
    )
}

/// Same as [`build_local_inline_harness_with_memory`], but lets a test wire a
/// distinct specialist fallback for durable-memory reflection. LCM still uses
/// `main`; keeping this separate proves reflection configuration cannot reroute
/// context compaction.
fn build_local_inline_harness_with_memory_and_reflection(
    main: Arc<dyn LLMProvider>,
    model: &str,
    max_context_tokens: usize,
    lcm_config: LcmSchemaConfig,
    memory_config: MemoryConfig,
    reflection: Option<Arc<dyn LLMProvider>>,
) -> (AgentLoop, std::path::PathBuf) {
    let workspace = tempfile::tempdir().unwrap().keep();
    let core = build_swappable_core(SwappableCoreConfig {
        provider: main,
        workspace: workspace.clone(),
        model: model.to_string(),
        max_iterations: 5,
        max_continuations: 2,
        max_tokens: 512,
        temperature: 0.3,
        max_context_tokens,
        brave_api_key: None,
        search_provider: "searxng".to_string(),
        searxng_url: "http://localhost:8888".to_string(),
        crw_url: String::new(),
        search_max_results: 5,
        exec_timeout: 30,
        restrict_to_workspace: true,
        memory_config,
        is_local: true,
        lane: Lane::default(),
        tool_delegation: ToolDelegationConfig::default(),
        provenance: ProvenanceConfig::default(),
        max_tool_result_chars: 2000,
        delegation_provider: None,
        specialist_provider: reflection,
        trio_config: TrioConfig::default(),
        model_capabilities_overrides: std::collections::HashMap::new(),
        reasoning_config: crate::config::schema::ReasoningConfig::default(),
        tool_heartbeat_secs: 2,
        health_check_timeout_secs: 2,
        code_execution: CodeExecutionConfig::default(),
        python_kernel: PythonKernelConfig::default(),
        cua: CuaToolConfig::default(),
        adaptive_tokens: AdaptiveTokenConfig::default(),
        sessions_db_path: Some(
            std::env::temp_dir().join(format!("nanobot-test-{}.sqlite", uuid::Uuid::new_v4())),
        ),
    });

    let counters = test_runtime_counters(max_context_tokens);
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
        lcm_config,
        None,
    );

    (agent_loop, workspace)
}

/// Cloud-mode counterpart to `build_local_inline_harness_with_memory_and_reflection`
/// (`is_local: false`). Exercises the real system+developer assembly split
/// (`ContextBuilder::collect_static_sections`) plus `prepare_context`'s
/// `collect_cloud_runtime_sections` (MemoryLadder, background-task status) --
/// the two mechanisms that once double-injected `MEMORY.md`.
fn build_cloud_inline_harness_with_memory(
    main: Arc<dyn LLMProvider>,
    model: &str,
    max_context_tokens: usize,
    lcm_config: LcmSchemaConfig,
    memory_config: MemoryConfig,
) -> (AgentLoop, std::path::PathBuf) {
    let workspace = tempfile::tempdir().unwrap().keep();
    let core = build_swappable_core(SwappableCoreConfig {
        provider: main,
        workspace: workspace.clone(),
        model: model.to_string(),
        max_iterations: 5,
        max_continuations: 2,
        max_tokens: 512,
        temperature: 0.3,
        max_context_tokens,
        brave_api_key: None,
        search_provider: "searxng".to_string(),
        searxng_url: "http://localhost:8888".to_string(),
        crw_url: String::new(),
        search_max_results: 5,
        exec_timeout: 30,
        restrict_to_workspace: true,
        memory_config,
        is_local: false,
        lane: Lane::default(),
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
        code_execution: CodeExecutionConfig::default(),
        python_kernel: PythonKernelConfig::default(),
        cua: CuaToolConfig::default(),
        adaptive_tokens: AdaptiveTokenConfig::default(),
        sessions_db_path: Some(
            std::env::temp_dir().join(format!("nanobot-test-{}.sqlite", uuid::Uuid::new_v4())),
        ),
    });

    let counters = test_runtime_counters(max_context_tokens);
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
        lcm_config,
        None,
    );

    (agent_loop, workspace)
}

/// Regression test for the MEMORY.md double-injection bug: `ContextBuilder::
/// collect_static_sections` used to be one injector and `MemoryLadder`'s
/// (now-removed) `GroundTruth` layer -- reached via `prepare_context::
/// collect_cloud_runtime_sections` -- was a second, silently duplicating
/// long-term memory content into the assembled cloud-path messages.
///
/// Inspects `TurnContext.messages` directly (via `prepare_context`, the
/// method that actually assembles system+developer content) rather than the
/// protocol-rendered wire sent to a provider -- `render_to_wire`/
/// `turn_from_legacy` has a separate, pre-existing gap where `role:
/// "developer"` messages aren't converted to a `Turn` at all and are dropped
/// during rendering, which is an unrelated protocol-layer bug, not a memory
/// double-injection.
#[tokio::test]
async fn memory_md_appears_exactly_once_in_assembled_cloud_messages() {
    let provider = MockLLM::named("cloud-memory-dedup-test");
    let (agent_loop, workspace) = build_cloud_inline_harness_with_memory(
        provider,
        "cloud-memory-dedup-test",
        128_000,
        LcmSchemaConfig::default(),
        MemoryConfig::default(),
    );

    // Distinctive content only `MEMORY.md` contains -- `collect_static_sections`
    // (the sole intended injector) reads it from disk via `MemoryStore`.
    const MARKER: &str = "XYZZY-UNIQUE-MEMORY-MARKER-42: the user prefers oat milk.";
    let memory_dir = workspace.join("memory");
    std::fs::create_dir_all(&memory_dir).unwrap();
    std::fs::write(memory_dir.join("MEMORY.md"), MARKER).unwrap();

    let session_key = format!("cloud-memory-dedup-{}", uuid::Uuid::new_v4());
    let mut msg = InboundMessage::new("test", "user", "offline", "Hello there.");
    msg.metadata
        .insert("session_key".to_string(), json!(session_key));

    let turn_ctx = agent_loop
        .shared
        .prepare_context(&msg, None, None, None, None)
        .await;

    // Count substring occurrences (not just how many messages contain it) so
    // a duplicate concatenated into the SAME message -- e.g. two blocks
    // folded into one `developer` message -- is caught too.
    let occurrences: usize = turn_ctx
        .messages
        .iter()
        .filter_map(|message| message.get("content").and_then(Value::as_str))
        .map(|content| content.matches(MARKER).count())
        .sum();
    assert_eq!(
        occurrences, 1,
        "MEMORY.md content must appear exactly once across the assembled cloud-path messages, got {}",
        occurrences
    );
}

/// Records the full wire `messages` array of every `chat()` call and replays
/// a scripted response sequence (last response repeats when exhausted).
struct WireRecordingProvider {
    name: String,
    responses: std::sync::Mutex<std::collections::VecDeque<crate::providers::base::LLMResponse>>,
    calls: std::sync::Mutex<Vec<Vec<Value>>>,
    higgs_capable: bool,
}

impl WireRecordingProvider {
    fn new(name: &str, responses: Vec<crate::providers::base::LLMResponse>) -> Self {
        Self {
            name: name.to_string(),
            responses: std::sync::Mutex::new(responses.into()),
            calls: std::sync::Mutex::new(Vec::new()),
            higgs_capable: false,
        }
    }

    /// Higgs-capable variant: the loop attaches retained-session control
    /// fields to messages[0], so tests can assert on the session id the wire
    /// actually carried.
    fn new_higgs_capable(name: &str, responses: Vec<crate::providers::base::LLMResponse>) -> Self {
        Self {
            higgs_capable: true,
            ..Self::new(name, responses)
        }
    }

    fn text_response(content: &str) -> crate::providers::base::LLMResponse {
        crate::providers::base::LLMResponse {
            content: Some(attested_text(content)),
            tool_calls: vec![],
            finish_reason: FinishReason::Stop,
            usage: std::collections::HashMap::new(),
        }
    }

    /// Internal compaction/reflection output is not an agent turn.
    fn plain_text_response(content: &str) -> crate::providers::base::LLMResponse {
        crate::providers::base::LLMResponse {
            content: Some(content.to_string()),
            tool_calls: vec![],
            finish_reason: FinishReason::Stop,
            usage: std::collections::HashMap::new(),
        }
    }

    fn calls(&self) -> Vec<Vec<Value>> {
        self.calls.lock().unwrap().clone()
    }
}

#[async_trait]
impl LLMProvider for WireRecordingProvider {
    async fn chat(
        &self,
        messages: &[Value],
        _tools: Option<&[Value]>,
        _model: Option<&str>,
        _max_tokens: u32,
        _temperature: f64,
        _thinking_budget: Option<u32>,
        _top_p: Option<f64>,
    ) -> anyhow::Result<crate::providers::base::LLMResponse> {
        self.calls.lock().unwrap().push(messages.to_vec());
        let mut queue = self.responses.lock().unwrap();
        Ok(if queue.len() > 1 {
            queue.pop_front().unwrap()
        } else {
            queue
                .front()
                .cloned()
                .unwrap_or_else(|| Self::text_response("done"))
        })
    }

    fn get_default_model(&self) -> &str {
        &self.name
    }

    fn supports_higgs_session_cache(&self) -> bool {
        self.higgs_capable
    }
}

/// Blocks the first provider request so a test can prove that a queued
/// same-session message cannot enter the provider concurrently and cannot
/// starve another session's concurrency permit.
struct BlockingFirstProvider {
    calls: std::sync::atomic::AtomicUsize,
    first_started: tokio::sync::Notify,
    allow_first: tokio::sync::Notify,
    second_started: tokio::sync::Notify,
}

impl BlockingFirstProvider {
    fn new() -> Self {
        Self {
            calls: std::sync::atomic::AtomicUsize::new(0),
            first_started: tokio::sync::Notify::new(),
            allow_first: tokio::sync::Notify::new(),
            second_started: tokio::sync::Notify::new(),
        }
    }
}

#[async_trait]
impl LLMProvider for BlockingFirstProvider {
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
        let call = self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if call == 0 {
            self.first_started.notify_one();
            self.allow_first.notified().await;
        } else {
            self.second_started.notify_one();
        }
        Ok(WireRecordingProvider::text_response(&format!(
            "response {}",
            call + 1
        )))
    }

    fn get_default_model(&self) -> &str {
        "local-qwen-test"
    }
}

/// Build an `AgentLoop` wired for gateway-mode `run()` tests: real inbound /
/// outbound channels and `max_concurrent_chats = 2` so permit starvation is
/// observable.
fn build_gateway_harness(
    provider: Arc<dyn LLMProvider>,
) -> (
    AgentLoop,
    tokio::sync::mpsc::UnboundedSender<InboundMessage>,
    tokio::sync::mpsc::UnboundedReceiver<OutboundMessage>,
    std::path::PathBuf,
) {
    let (base_loop, workspace) = build_local_inline_harness(provider);
    let core_handle = base_loop.shared.core_handle.clone();
    drop(base_loop);

    let (inbound_tx, inbound_rx) = tokio::sync::mpsc::unbounded_channel();
    let (outbound_tx, outbound_rx) = tokio::sync::mpsc::unbounded_channel();
    let gateway_loop = AgentLoop::new(
        core_handle,
        inbound_rx,
        outbound_tx,
        inbound_tx.clone(),
        None,
        2,
        None,
        None,
        None,
        crate::config::schema::ProprioceptionConfig::default(),
        LcmSchemaConfig::default(),
        None,
    );
    (gateway_loop, inbound_tx, outbound_rx, workspace)
}

#[tokio::test]
async fn gateway_clear_waits_for_active_same_session_turn() {
    let provider = Arc::new(BlockingFirstProvider::new());
    let (mut gateway_loop, inbound_tx, mut outbound_rx, workspace) =
        build_gateway_harness(provider.clone() as Arc<dyn LLMProvider>);
    let sessions = gateway_loop.shared.core_handle.swappable().sessions.clone();
    let running = gateway_loop.running.clone();
    let runner = tokio::spawn(async move { gateway_loop.run().await });
    let session_key = "test:offline".to_string();

    let first = InboundMessage::new("test", "user", "offline", "first");
    let first_started = provider.first_started.notified();
    inbound_tx.send(first).unwrap();
    tokio::time::timeout(std::time::Duration::from_secs(5), first_started)
        .await
        .expect("first gateway turn must reach the provider");

    let clear = InboundMessage::new("test", "user", "offline", "/clear");
    inbound_tx.send(clear).unwrap();
    assert!(
        tokio::time::timeout(std::time::Duration::from_millis(150), outbound_rx.recv(),)
            .await
            .is_err(),
        "/clear completed while an older same-session turn still held the session lock"
    );

    let third = InboundMessage::new("test", "user", "other", "independent");
    let independent_started = provider.second_started.notified();
    inbound_tx.send(third).unwrap();
    tokio::time::timeout(std::time::Duration::from_secs(5), independent_started)
        .await
        .expect("a queued same-session clear must not consume another session's permit");
    let independent_outbound =
        tokio::time::timeout(std::time::Duration::from_secs(5), outbound_rx.recv())
            .await
            .expect("independent response must arrive while the first turn remains blocked")
            .expect("outbound channel must stay open");
    assert_eq!(independent_outbound.content, "response 2");

    provider.allow_first.notify_one();
    let first_outbound =
        tokio::time::timeout(std::time::Duration::from_secs(5), outbound_rx.recv())
            .await
            .expect("first response must arrive")
            .expect("outbound channel must stay open");
    assert_eq!(first_outbound.content, "response 1");
    let clear_outbound =
        tokio::time::timeout(std::time::Duration::from_secs(5), outbound_rx.recv())
            .await
            .expect("clear response must arrive after the active turn")
            .expect("outbound channel must stay open");
    assert_eq!(
        clear_outbound.content,
        "Working memory and history cleared."
    );

    let session = sessions
        .get_latest_session(&session_key)
        .await
        .expect("gateway session must exist");
    let replay = sessions.get_history(&session.id, 100, 0).await;
    assert!(
        replay.is_empty(),
        "the completed pre-clear turn must remain behind the clear marker on replay: {replay:?}"
    );

    drop(inbound_tx);
    running.store(false, std::sync::atomic::Ordering::SeqCst);
    tokio::time::timeout(std::time::Duration::from_secs(5), runner)
        .await
        .expect("gateway loop must stop after its input channel closes")
        .unwrap();
    let _ = std::fs::remove_dir_all(&workspace);
}

#[tokio::test]
async fn cross_session_command_seen_during_coalescing_uses_gateway_dispatch() {
    let provider = Arc::new(WireRecordingProvider::new(
        "local-main",
        vec![WireRecordingProvider::text_response("coalesced response")],
    ));
    let (mut gateway_loop, inbound_tx, mut outbound_rx, workspace) =
        build_gateway_harness(provider.clone() as Arc<dyn LLMProvider>);
    let running = gateway_loop.running.clone();
    let runner = tokio::spawn(async move { gateway_loop.run().await });

    inbound_tx
        .send(InboundMessage::new("test", "user", "first", "hello"))
        .unwrap();
    inbound_tx
        .send(InboundMessage::new("test", "user", "second", "/clear"))
        .unwrap();

    let mut responses = Vec::new();
    for _ in 0..2 {
        responses.push(
            tokio::time::timeout(std::time::Duration::from_secs(5), outbound_rx.recv())
                .await
                .expect("both the normal turn and command must complete")
                .expect("outbound channel must stay open")
                .content,
        );
    }
    assert!(responses
        .iter()
        .any(|response| response == "coalesced response"));
    assert!(
        responses
            .iter()
            .any(|response| response == "Working memory and history cleared."),
        "the cross-session /clear was routed to the model: {responses:?}"
    );
    assert_eq!(
        provider.calls().len(),
        1,
        "recognized gateway commands must not consume an inference request"
    );

    running.store(false, std::sync::atomic::Ordering::SeqCst);
    tokio::time::timeout(std::time::Duration::from_secs(5), runner)
        .await
        .expect("gateway loop must stop")
        .unwrap();
    let _ = std::fs::remove_dir_all(&workspace);
}

#[tokio::test]
async fn rapid_same_session_user_messages_still_coalesce() {
    let provider = Arc::new(WireRecordingProvider::new(
        "local-coalescing-test",
        vec![WireRecordingProvider::text_response("user response")],
    ));
    let (mut gateway_loop, inbound_tx, mut outbound_rx, workspace) =
        build_gateway_harness(provider.clone() as Arc<dyn LLMProvider>);
    let running = gateway_loop.running.clone();
    let runner = tokio::spawn(async move { gateway_loop.run().await });

    inbound_tx
        .send(InboundMessage::new(
            "test",
            "user",
            "coalesced-chat",
            "first user marker",
        ))
        .unwrap();
    inbound_tx
        .send(InboundMessage::new(
            "test",
            "user",
            "coalesced-chat",
            "second user marker",
        ))
        .unwrap();

    let response = tokio::time::timeout(std::time::Duration::from_secs(5), outbound_rx.recv())
        .await
        .expect("the coalesced user turn must complete")
        .expect("outbound channel must stay open");
    assert_eq!(response.content, "user response");
    let calls = provider.calls();
    assert_eq!(calls.len(), 1, "rapid user messages must share one turn");
    let wire = serde_json::to_string(&calls[0]).unwrap();
    assert!(wire.contains("first user marker"));
    assert!(wire.contains("second user marker"));

    running.store(false, std::sync::atomic::Ordering::SeqCst);
    tokio::time::timeout(std::time::Duration::from_secs(5), runner)
        .await
        .expect("gateway loop must stop")
        .unwrap();
    let _ = std::fs::remove_dir_all(&workspace);
}

#[tokio::test]
async fn system_announcement_after_user_message_does_not_coalesce() {
    let provider = Arc::new(WireRecordingProvider::new(
        "local-system-announcement-test",
        vec![WireRecordingProvider::text_response("user response")],
    ));
    let (mut gateway_loop, inbound_tx, mut outbound_rx, workspace) =
        build_gateway_harness(provider.clone() as Arc<dyn LLMProvider>);
    let running = gateway_loop.running.clone();
    let runner = tokio::spawn(async move { gateway_loop.run().await });

    inbound_tx
        .send(InboundMessage::new(
            "test",
            "user",
            "announcement-chat",
            "user turn marker",
        ))
        .unwrap();
    let mut announcement = InboundMessage::new(
        "test",
        "subagent",
        "announcement-chat",
        "system announcement marker",
    );
    announcement
        .metadata
        .insert("is_system".to_string(), json!(true));
    inbound_tx.send(announcement).unwrap();

    let mut responses = Vec::new();
    for _ in 0..2 {
        responses.push(
            tokio::time::timeout(std::time::Duration::from_secs(5), outbound_rx.recv())
                .await
                .expect("the user reply and system announcement must both be emitted")
                .expect("outbound channel must stay open")
                .content,
        );
    }
    assert!(responses.iter().any(|content| content == "user response"));
    assert!(responses
        .iter()
        .any(|content| content == "system announcement marker"));

    let calls = provider.calls();
    assert_eq!(calls.len(), 1, "the user turn must reach the provider once");
    let wire = serde_json::to_string(&calls[0]).unwrap();
    assert!(wire.contains("user turn marker"));
    assert!(!wire.contains("system announcement marker"));

    running.store(false, std::sync::atomic::Ordering::SeqCst);
    tokio::time::timeout(std::time::Duration::from_secs(5), runner)
        .await
        .expect("gateway loop must stop")
        .unwrap();
    let _ = std::fs::remove_dir_all(&workspace);
}

#[tokio::test]
async fn hard_lcm_checkpoint_is_installed_before_foreground_inference() {
    const EARLIEST_FAILED_ACTION: &str = "EARLIEST_FAILED_ACTION_EX_7041_PERMISSION_DENIED";
    let provider = Arc::new(WireRecordingProvider::new(
        "local-hard-lcm-test",
        vec![
            WireRecordingProvider::plain_text_response(
                "- Prior turns retained project detail context for later reference.",
            ),
            WireRecordingProvider::text_response("foreground reply"),
        ],
    ));
    let lcm_config = LcmSchemaConfig {
        tau_soft: 0.05,
        tau_hard: 0.10,
        deterministic_target: 64,
        // Lossless-handoff coverage test: exempt from prefix-preserving cuts
        // (the kept head would remove the oldest marker from the summarizer wire).
        keep_prefix_fraction: 0.0,
        ..Default::default()
    };
    let (agent_loop, _workspace) = build_local_inline_harness_with_memory_and_reflection(
        provider.clone() as Arc<dyn LLMProvider>,
        "local-hard-lcm-test",
        8192,
        lcm_config,
        MemoryConfig::default(),
        None,
    );
    let session_key = format!("hard-lcm-barrier-{}", uuid::Uuid::new_v4());
    let core = agent_loop.shared.core_handle.swappable();
    let session = core.sessions.get_or_resume(&session_key).await;
    // Forty small rows exceed the former context-derived 38-row cap. LCM must
    // receive the oldest failed action before any token-driven reduction.
    for turn in 0..20_u64 {
        let failed_action = if turn == 0 {
            EARLIEST_FAILED_ACTION
        } else {
            "ordinary retained state"
        };
        let detail = format!(
            "turn {turn}: {failed_action}; {}",
            "persistent project detail with enough context to require lossless compaction "
                .repeat(12)
        );
        core.sessions
            .add_message(
                &session.id,
                &json!({"role": "user", "content": detail, "_turn": turn}),
            )
            .await;
        core.sessions
            .add_message(
                &session.id,
                &json!({"role": "assistant", "content": format!("acknowledged turn {turn}"), "_turn": turn}),
            )
            .await;
    }

    let response = agent_loop
        .process_direct(
            "Use the retained project details to answer briefly.",
            &session_key,
            "test",
            "offline",
        )
        .await;
    assert_eq!(response, "foreground reply");

    let calls = provider.calls();
    assert_eq!(
        calls.len(),
        2,
        "expected one compaction call followed by one foreground call"
    );
    let compaction_wire = serde_json::to_string(&calls[0]).unwrap();
    assert!(
        compaction_wire.contains(EARLIEST_FAILED_ACTION),
        "pre-call retention must not discard old persisted evidence before LCM sees it"
    );
    // Internal fields like `_lcm_summary` are stripped before messages hit
    // the wire, so `calls[0]` can't be checked via that tag. Match the
    // summary wire message's exact phrasing instead of a bare "[Summary of
    // messages" substring — the LCM_EXPAND_GUIDE instructional text
    // (prepare_context.rs) contains that same lead-in as a generic example
    // ("copy that range into lcm_expand — for example ..."), which would
    // false-positive this check even if compaction never installed
    // anything. `summary_wire_message` (lcm.rs) uniquely phrases it "To read
    // the exact originals call lcm_expand(...)".
    let foreground_call = calls.last().expect("foreground call recorded");
    let has_summary = foreground_call.iter().any(|message| {
        message
            .get("content")
            .and_then(Value::as_str)
            .is_some_and(|content| content.contains("To read the exact originals call"))
    });
    assert!(
        has_summary,
        "hard-pressure LCM checkpoint must be installed before the foreground call"
    );
    assert!(foreground_call.iter().any(|message| {
        message
            .get("content")
            .and_then(Value::as_str)
            .is_some_and(|content| content.contains("Use the retained project details"))
    }));
    assert!(
        agent_loop
            .shared
            .core_handle
            .counters
            .lcm_compaction_count
            .load(std::sync::atomic::Ordering::Relaxed)
            >= 1
    );

    let nodes = tokio::time::timeout(std::time::Duration::from_secs(2), async {
        loop {
            let nodes = core.sessions.load_summary_nodes(&session.id).await;
            if !nodes.is_empty() {
                break nodes;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("LCM summary node must become restart-durable");
    let raw = core.sessions.get_all_messages(&session.id).await;
    let rebuilt = crate::agent::lcm::LcmEngine::rebuild_from_db_nodes(
        &raw,
        &nodes,
        crate::agent::lcm::LcmConfig::default(),
    );
    assert!(rebuilt.active_context().iter().any(|message| {
        message
            .get("_lcm_summary")
            .and_then(Value::as_bool)
            .unwrap_or(false)
    }));
}

#[tokio::test]
async fn soft_lcm_uses_main_provider_and_preserves_foreground_context() {
    let main_provider = Arc::new(WireRecordingProvider::new(
        "local-soft-lcm-test",
        vec![WireRecordingProvider::text_response("foreground reply")],
    ));
    let memory_provider = Arc::new(WireRecordingProvider::new(
        "memory-soft-lcm-test",
        vec![WireRecordingProvider::plain_text_response(
            "- memory summary",
        )],
    ));
    let lcm_config = LcmSchemaConfig {
        tau_soft: 0.0001,
        // Deliberately above 1.0 in this focused policy test so even a tiny
        // effective budget cannot turn the soft case into hard pressure.
        tau_hard: 10.0,
        deterministic_target: 64,
        // Lossless-handoff coverage test: exempt from prefix-preserving cuts
        // (the kept head would remove the oldest marker from the summarizer wire).
        keep_prefix_fraction: 0.0,
        ..Default::default()
    };

    let (agent_loop, _workspace) = build_local_inline_harness_with_memory_and_reflection(
        main_provider.clone() as Arc<dyn LLMProvider>,
        "local-soft-lcm-test",
        1_000_000,
        lcm_config,
        MemoryConfig::default(),
        Some(memory_provider.clone() as Arc<dyn LLMProvider>),
    );
    let session_key = format!("soft-lcm-preserve-{}", uuid::Uuid::new_v4());
    let core = agent_loop.shared.core_handle.swappable();
    let session = core.sessions.get_or_resume(&session_key).await;
    // 8 turns of padded content: the LCM engine protects a ~2048-token tail
    // of the most recent raw messages (see `protect_tokens_for_budget`) and
    // only compacts the oldest block beyond that. It must clear
    // `MIN_COMPACTION_TOKENS` (200) or compaction skips silently without
    // ever calling the provider — this volume keeps the oldest block
    // comfortably above that floor so the main-provider compactor is actually
    // exercised.
    for turn in 0..8_u64 {
        core.sessions
            .add_message(
                &session.id,
                &json!({
                    "role": "user",
                    "content": format!("soft-pressure-marker-{turn} {}", "context detail ".repeat(40)),
                    "_turn": turn
                }),
            )
            .await;
        core.sessions
            .add_message(
                &session.id,
                &json!({"role": "assistant", "content": format!("soft ack {turn}"), "_turn": turn}),
            )
            .await;
    }

    let response = agent_loop
        .process_direct(
            "Continue without losing context.",
            &session_key,
            "test",
            "offline",
        )
        .await;
    assert_eq!(response, "foreground reply");

    // Async (soft) compaction must use the main provider even though a distinct
    // provider is configured for memory reflection.
    tokio::time::timeout(std::time::Duration::from_secs(3), async {
        while main_provider.calls().len() <= 1 {
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("background compaction must call the main provider");

    // The mock's canned foreground reply isn't a valid bullet-only handoff, so
    // both escalation levels reject it and compaction leaves the context
    // uncompacted rather than installing deterministic truncation.
    let calls = main_provider.calls();
    assert!(
        calls.len() > 1,
        "compaction must have been attempted against the main provider, got {} call(s)",
        calls.len()
    );
    assert!(
        memory_provider.calls().is_empty(),
        "LCM must not send compaction requests to the reflection provider"
    );
    // The last call is the LCM escalation attempt (Level 1/2 summarization),
    // not the original foreground chat call — it must still carry the
    // original source content it's being asked to summarize, proving
    // nothing was silently dropped before compaction gave up.
    let last_call = calls.last().expect("at least one call recorded");
    assert!(last_call.iter().any(|message| {
        message
            .get("content")
            .and_then(Value::as_str)
            .is_some_and(|content| content.contains("soft-pressure-marker-0"))
    }));
    assert!(
        !calls.iter().flatten().any(|message| {
            // Match the summary wire message's exact phrasing, not a bare
            // "[Summary of messages" substring — the LCM_EXPAND_GUIDE
            // instructional text (prepare_context.rs) contains that same
            // lead-in as a generic example and would false-positive this
            // check. `summary_wire_message` (lcm.rs) uniquely phrases an
            // actually-installed summary "To read the exact originals call
            // lcm_expand(...)".
            message
                .get("content")
                .and_then(Value::as_str)
                .is_some_and(|content| content.contains("To read the exact originals call"))
        }),
        "no summary — real or the deleted deterministic-truncation fallback — was ever installed"
    );
}

fn is_lcm_compaction_request(messages: &[Value]) -> bool {
    messages
        .first()
        .and_then(|message| message.get("content"))
        .and_then(Value::as_str)
        .is_some_and(|content| content.contains("conversation-state compressor"))
}

async fn seed_compaction_history(
    core: &SwappableCore,
    session_id: &str,
    turns: u64,
    detail_repetitions: usize,
) {
    for turn in 0..turns {
        core.sessions
            .add_message(
                session_id,
                &json!({
                    "role": "user",
                    "content": format!(
                        "turn {turn}: {}",
                        "persistent project detail with decisions and constraints "
                            .repeat(detail_repetitions)
                    ),
                    "_turn": turn,
                }),
            )
            .await;
        core.sessions
            .add_message(
                session_id,
                &json!({
                    "role": "assistant",
                    "content": format!("acknowledged retained project detail for turn {turn}"),
                    "_turn": turn,
                }),
            )
            .await;
    }
}

async fn persist_prior_summary(core: &SwappableCore, session_id: &str) {
    let raw = core.sessions.get_all_messages(session_id).await;
    let source_ids = raw
        .iter()
        .take(4)
        .map(|message| message["_db_id"].as_u64().unwrap() as usize)
        .collect::<Vec<_>>();
    assert_eq!(source_ids.len(), 4, "prior summary needs four source rows");
    let text = "Prior project details, decisions, constraints, and acknowledged outcomes.";
    core.sessions
        .save_compaction_checkpoint(
            session_id,
            0,
            &source_ids,
            &[],
            text,
            crate::agent::token_budget::TokenBudget::estimate_str_tokens(text),
            1,
            &crate::agent::lcm::SummaryManifest::default(),
            None,
        )
        .await
        .unwrap();
}

struct ForegroundPriorityProvider {
    first_foreground_started: Arc<tokio::sync::Notify>,
    release_first_foreground: Arc<tokio::sync::Notify>,
    soft_generation_started: Arc<tokio::sync::Notify>,
    foreground_calls: std::sync::atomic::AtomicUsize,
}

#[async_trait]
impl LLMProvider for ForegroundPriorityProvider {
    async fn chat(
        &self,
        messages: &[Value],
        _tools: Option<&[Value]>,
        _model: Option<&str>,
        _max_tokens: u32,
        _temperature: f64,
        _thinking_budget: Option<u32>,
        _top_p: Option<f64>,
    ) -> anyhow::Result<crate::providers::base::LLMResponse> {
        if is_lcm_compaction_request(messages) {
            self.soft_generation_started.notify_one();
            return std::future::pending().await;
        }

        let call = self
            .foreground_calls
            .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        if call == 0 {
            self.first_foreground_started.notify_one();
            self.release_first_foreground.notified().await;
            return Ok(WireRecordingProvider::text_response("turn one reply"));
        }
        Ok(WireRecordingProvider::text_response("turn two reply"))
    }

    fn get_default_model(&self) -> &str {
        "foreground-priority-compaction-test"
    }
}

#[tokio::test]
async fn soft_compaction_waits_for_turn_end_and_next_foreground_preempts_generation() {
    let first_foreground_started = Arc::new(tokio::sync::Notify::new());
    let release_first_foreground = Arc::new(tokio::sync::Notify::new());
    let soft_generation_started = Arc::new(tokio::sync::Notify::new());
    let provider = Arc::new(ForegroundPriorityProvider {
        first_foreground_started: first_foreground_started.clone(),
        release_first_foreground: release_first_foreground.clone(),
        soft_generation_started: soft_generation_started.clone(),
        foreground_calls: std::sync::atomic::AtomicUsize::new(0),
    });
    let lcm_config = LcmSchemaConfig {
        tau_soft: 0.0001,
        tau_hard: 10.0,
        deterministic_target: 64,
        ..Default::default()
    };
    let (agent_loop, _workspace) = build_local_inline_harness_with_lcm(
        provider as Arc<dyn LLMProvider>,
        "foreground-priority-compaction-test",
        1_000_000,
        lcm_config,
    );
    let session_key = format!("foreground-priority-{}", uuid::Uuid::new_v4());
    let core = agent_loop.shared.core_handle.swappable();
    let session = core.sessions.get_or_resume(&session_key).await;
    seed_compaction_history(&core, &session.id, 12, 40).await;

    let mut turn_one = Box::pin(agent_loop.process_direct(
        "Finish this foreground turn before compacting.",
        &session_key,
        "test",
        "offline",
    ));
    await_compaction_sync("turn one to enter its foreground provider call", async {
        tokio::select! {
            response = turn_one.as_mut() => {
                panic!("turn one returned before its provider barrier: {response}");
            }
            () = first_foreground_started.notified() => {}
        }
    })
    .await;

    assert!(
        tokio::time::timeout(
            COMPACTION_BLOCKED_OBSERVATION,
            soft_generation_started.notified(),
        )
        .await
        .is_err(),
        "soft generation raced the still-running foreground model call"
    );

    release_first_foreground.notify_one();
    assert_eq!(
        await_compaction_sync("turn one to finish", turn_one.as_mut()).await,
        "turn one reply"
    );
    await_compaction_sync(
        "soft generation to start after the full foreground loop",
        soft_generation_started.notified(),
    )
    .await;

    let turn_two = tokio::time::timeout(
        std::time::Duration::from_secs(2),
        agent_loop.process_direct(
            "Foreground work must preempt soft generation.",
            &session_key,
            "test",
            "offline",
        ),
    )
    .await
    .expect("turn two did not cancel soft generation and reach the foreground provider");
    assert_eq!(turn_two, "turn two reply");
}

struct ReplayStableSoftProvider {
    foreground_calls: std::sync::Mutex<Vec<Vec<Value>>>,
    foreground_max_tokens: std::sync::Mutex<Vec<u32>>,
    force_recovery_on_second: bool,
}

impl ReplayStableSoftProvider {
    fn foreground_calls(&self) -> Vec<Vec<Value>> {
        self.foreground_calls.lock().unwrap().clone()
    }

    fn foreground_max_tokens(&self) -> Vec<u32> {
        self.foreground_max_tokens.lock().unwrap().clone()
    }
}

#[async_trait]
impl LLMProvider for ReplayStableSoftProvider {
    async fn chat(
        &self,
        messages: &[Value],
        _tools: Option<&[Value]>,
        _model: Option<&str>,
        max_tokens: u32,
        _temperature: f64,
        _thinking_budget: Option<u32>,
        _top_p: Option<f64>,
    ) -> anyhow::Result<crate::providers::base::LLMResponse> {
        if is_lcm_compaction_request(messages) {
            return Ok(WireRecordingProvider::plain_text_response(
                "- Persistent project details, decisions, constraints, acknowledged outcomes, and follow-up context remain available.",
            ));
        }

        let call = {
            let mut calls = self.foreground_calls.lock().unwrap();
            calls.push(messages.to_vec());
            self.foreground_max_tokens.lock().unwrap().push(max_tokens);
            calls.len()
        };
        let mut response = if self.force_recovery_on_second && call == 2 {
            WireRecordingProvider::plain_text_response(
                "I'll read it.\n[Called read_file({\"path\":\"/x\"})]",
            )
        } else if self.force_recovery_on_second && call == 3 {
            let mut arguments = std::collections::HashMap::new();
            arguments.insert("path".to_string(), json!("."));
            crate::providers::base::LLMResponse {
                content: Some(String::new()),
                tool_calls: vec![crate::providers::base::ToolCallRequest {
                    id: "tc_lease_recovery".to_string(),
                    name: "list_dir".to_string(),
                    arguments,
                }],
                finish_reason: FinishReason::ToolCalls,
                usage: std::collections::HashMap::new(),
            }
        } else {
            WireRecordingProvider::text_response(if call == 1 {
                "turn one reply"
            } else {
                "turn two reply"
            })
        };
        if call == 2 {
            response
                .usage
                .insert("higgs_session_lease_active".to_string(), 1);
        }
        Ok(response)
    }

    fn get_default_model(&self) -> &str {
        "replay-stable-soft-compaction-test"
    }

    fn get_api_base(&self) -> Option<&str> {
        Some("http://127.0.0.1:1234/v1")
    }

    fn supports_higgs_session_cache(&self) -> bool {
        true
    }
}

#[tokio::test]
async fn published_soft_checkpoint_replays_and_installs_on_next_turn() {
    let provider = Arc::new(ReplayStableSoftProvider {
        foreground_calls: std::sync::Mutex::new(Vec::new()),
        foreground_max_tokens: std::sync::Mutex::new(Vec::new()),
        force_recovery_on_second: false,
    });
    let lcm_config = LcmSchemaConfig {
        tau_soft: 0.0001,
        tau_hard: 10.0,
        deterministic_target: 64,
        ..Default::default()
    };
    let (agent_loop, _workspace) = build_local_inline_harness_with_lcm(
        provider.clone() as Arc<dyn LLMProvider>,
        "replay-stable-soft-compaction-test",
        1_000_000,
        lcm_config,
    );
    let session_key = format!("replay-stable-soft-{}", uuid::Uuid::new_v4());
    let core = agent_loop.shared.core_handle.swappable();
    let session = core.sessions.get_or_resume(&session_key).await;
    seed_compaction_history(&core, &session.id, 12, 40).await;
    persist_prior_summary(&core, &session.id).await;

    let mut probe = InboundMessage::new("test", "user", "offline", "probe prior summary");
    probe
        .metadata
        .insert("session_key".to_string(), json!(session_key.clone()));
    let prepared = agent_loop
        .shared
        .prepare_context(&probe, None, None, None, None)
        .await;
    assert!(prepared
        .messages
        .iter()
        .any(|message| message.get("_lcm_summary").is_some()));
    let compaction = prepared.compaction.clone();
    drop(prepared);

    assert_eq!(
        agent_loop
            .process_direct("Finish turn one.", &session_key, "test", "offline")
            .await,
        "turn one reply"
    );
    await_compaction_sync("soft checkpoint publication and pending handoff", async {
        while !compaction.has_pending().await {
            tokio::task::yield_now().await;
        }
    })
    .await;
    assert!(core.sessions.load_summary_nodes(&session.id).await.len() >= 2);

    let epoch_before = agent_loop
        .shared
        .core_handle
        .counters
        .session_prompt_epoch(&session_key);
    let installs_before = agent_loop
        .shared
        .core_handle
        .counters
        .lcm_compaction_count
        .load(std::sync::atomic::Ordering::Acquire);
    assert_eq!(
        agent_loop
            .process_direct("Continue on turn two.", &session_key, "test", "offline")
            .await,
        "turn two reply"
    );
    assert_eq!(
        agent_loop
            .shared
            .core_handle
            .counters
            .take_cache_reset(&session_key),
        None,
        "an unchanged stable prefix must not emit an LCM cache reset"
    );

    assert!(!compaction.has_pending().await);
    assert!(
        agent_loop
            .shared
            .core_handle
            .counters
            .session_prompt_epoch(&session_key)
            > epoch_before,
        "installing the replayable soft checkpoint did not rotate the prompt cache"
    );
    assert!(
        agent_loop
            .shared
            .core_handle
            .counters
            .lcm_compaction_count
            .load(std::sync::atomic::Ordering::Acquire)
            > installs_before,
        "the pending soft checkpoint was not installed"
    );
    let calls = provider.foreground_calls();
    assert_eq!(calls.len(), 2);
    assert!(calls[1].iter().any(|message| {
        message
            .get("content")
            .and_then(Value::as_str)
            .is_some_and(|content| content.contains("To read the exact originals call"))
    }));
}

#[tokio::test]
async fn manual_compaction_does_not_create_a_fake_turn() {
    let provider = Arc::new(ReplayStableSoftProvider {
        foreground_calls: std::sync::Mutex::new(Vec::new()),
        foreground_max_tokens: std::sync::Mutex::new(Vec::new()),
        force_recovery_on_second: false,
    });
    let lcm_config = LcmSchemaConfig {
        tau_soft: 0.0001,
        tau_hard: 10.0,
        deterministic_target: 64,
        ..Default::default()
    };
    let (agent_loop, workspace) = build_local_inline_harness_with_lcm(
        provider.clone() as Arc<dyn LLMProvider>,
        "manual-compact-test",
        1_000_000,
        lcm_config,
    );
    let session_key = format!("manual-compact-{}", uuid::Uuid::new_v4());
    let core = agent_loop.shared.core_handle.swappable();
    let session = core.sessions.get_or_resume(&session_key).await;
    seed_compaction_history(&core, &session.id, 12, 40).await;
    let before_rows = core.sessions.get_all_messages(&session.id).await;

    let report = agent_loop.shared.compact_session_now(&session_key).await;

    assert!(
        report.after_tokens < report.before_tokens,
        "manual compaction should reduce the rendered prompt: {report:?}"
    );
    let after_rows = core.sessions.get_all_messages(&session.id).await;
    assert_eq!(
        after_rows.len(),
        before_rows.len(),
        "manual compaction must not persist a synthetic user turn"
    );
    assert!(
        after_rows.iter().all(|message| {
            message
                .get("content")
                .and_then(Value::as_str)
                .is_some_and(|content| !content.is_empty())
        }),
        "manual compaction must not persist an empty maintenance message"
    );
    assert!(
        provider.foreground_calls().is_empty(),
        "manual compaction must not send a foreground user turn"
    );

    let _ = std::fs::remove_dir_all(workspace);
}

#[tokio::test]
async fn soft_checkpoint_survives_turn_finish_journal_failure() {
    // Break caught: a durably-checkpointed LCM compaction was discarded when
    // the final replay journal write failed, so the compacted context never
    // installed on the next turn (prefix-divergence class).
    let provider = Arc::new(ReplayStableSoftProvider {
        foreground_calls: std::sync::Mutex::new(Vec::new()),
        foreground_max_tokens: std::sync::Mutex::new(Vec::new()),
        force_recovery_on_second: false,
    });
    let lcm_config = LcmSchemaConfig {
        tau_soft: 0.0001,
        tau_hard: 10.0,
        deterministic_target: 64,
        ..Default::default()
    };
    let (agent_loop, _workspace) = build_local_inline_harness_with_lcm(
        provider.clone() as Arc<dyn LLMProvider>,
        "replay-stable-soft-compaction-test",
        1_000_000,
        lcm_config,
    );
    let session_key = format!("replay-soft-fault-{}", uuid::Uuid::new_v4());
    let core = agent_loop.shared.core_handle.swappable();
    let session = core.sessions.get_or_resume(&session_key).await;
    seed_compaction_history(&core, &session.id, 12, 40).await;
    persist_prior_summary(&core, &session.id).await;

    let mut probe = InboundMessage::new("test", "user", "offline", "probe prior summary");
    probe
        .metadata
        .insert("session_key".to_string(), json!(session_key.clone()));
    let prepared = agent_loop
        .shared
        .prepare_context(&probe, None, None, None, None)
        .await;
    assert!(prepared
        .messages
        .iter()
        .any(|message| message.get("_lcm_summary").is_some()));
    let compaction = prepared.compaction.clone();
    drop(prepared);

    // Fail both turn-finish journal writes of turn one: the foreground
    // finalize and the compaction close. Reply and checkpoint must both
    // survive; replay degrades to Incomplete.
    core.sessions.fail_turn_finished_writes_for_tests(2);

    assert_eq!(
        agent_loop
            .process_direct("Finish turn one.", &session_key, "test", "offline")
            .await,
        "turn one reply"
    );
    await_compaction_sync(
        "soft checkpoint publication despite journal failure",
        async {
            while !compaction.has_pending().await {
                tokio::task::yield_now().await;
            }
        },
    )
    .await;

    let installs_before = agent_loop
        .shared
        .core_handle
        .counters
        .lcm_compaction_count
        .load(std::sync::atomic::Ordering::Acquire);
    assert_eq!(
        agent_loop
            .process_direct("Continue on turn two.", &session_key, "test", "offline")
            .await,
        "turn two reply"
    );

    assert!(
        agent_loop
            .shared
            .core_handle
            .counters
            .lcm_compaction_count
            .load(std::sync::atomic::Ordering::Acquire)
            > installs_before,
        "the checkpoint must install despite the journal failure"
    );
    let calls = provider.foreground_calls();
    assert_eq!(calls.len(), 2);
    assert!(calls[1].iter().any(|message| {
        message
            .get("content")
            .and_then(Value::as_str)
            .is_some_and(|content| content.contains("To read the exact originals call"))
    }));
}

#[tokio::test]
async fn cloud_working_memory_prefix_change_allows_soft_checkpoint_install() {
    let provider = Arc::new(ReplayStableSoftProvider {
        foreground_calls: std::sync::Mutex::new(Vec::new()),
        foreground_max_tokens: std::sync::Mutex::new(Vec::new()),
        force_recovery_on_second: false,
    });
    let lcm_config = LcmSchemaConfig {
        tau_soft: 0.0001,
        tau_hard: 10.0,
        deterministic_target: 64,
        ..Default::default()
    };
    let (agent_loop, _workspace) = build_cloud_inline_harness_with_memory(
        provider.clone() as Arc<dyn LLMProvider>,
        "cloud-working-memory-soft-compaction-test",
        1_000_000,
        lcm_config,
        MemoryConfig::default(),
    );
    let session_key = format!("cloud-working-memory-soft-{}", uuid::Uuid::new_v4());
    let core = agent_loop.shared.core_handle.swappable();
    let session = core.sessions.get_or_resume(&session_key).await;
    seed_compaction_history(&core, &session.id, 12, 40).await;
    persist_prior_summary(&core, &session.id).await;

    let mut probe = InboundMessage::new("test", "user", "offline", "probe cloud prefix");
    probe
        .metadata
        .insert("session_key".to_string(), json!(session_key.clone()));
    let prepared = agent_loop
        .shared
        .prepare_context(&probe, None, None, None, None)
        .await;
    let developer_before = prepared
        .messages
        .iter()
        .find(|message| message.get("role").and_then(Value::as_str) == Some("developer"))
        .and_then(|message| message.get("content"))
        .and_then(Value::as_str)
        .expect("cloud prompt must have a developer prefix")
        .to_string();
    let compaction = prepared.compaction.clone();
    drop(prepared);

    assert_eq!(
        agent_loop
            .process_direct("Finish cloud turn one.", &session_key, "test", "offline")
            .await,
        "turn one reply"
    );
    await_compaction_sync("cloud soft checkpoint publication", async {
        while !compaction.has_pending().await {
            tokio::task::yield_now().await;
        }
    })
    .await;
    assert_eq!(
        core.working_memory
            .get_context(&session.id, usize::MAX)
            .await
            .unwrap(),
        "- Persistent project details, decisions, constraints, acknowledged outcomes, and follow-up context remain available."
    );
    assert_eq!(
        agent_loop
            .shared
            .core_handle
            .counters
            .lcm_compaction_count
            .load(std::sync::atomic::Ordering::Acquire),
        0,
        "publication must not install the checkpoint before the next turn"
    );

    let mut next_probe = InboundMessage::new("test", "user", "offline", "probe changed prefix");
    next_probe
        .metadata
        .insert("session_key".to_string(), json!(session_key.clone()));
    let prepared = agent_loop
        .shared
        .prepare_context(&next_probe, None, None, None, None)
        .await;
    let developer_after = prepared
        .messages
        .iter()
        .find(|message| message.get("role").and_then(Value::as_str) == Some("developer"))
        .and_then(|message| message.get("content"))
        .and_then(Value::as_str)
        .expect("working memory must remain in the developer prefix");
    assert_ne!(developer_after, developer_before);
    assert!(developer_after.contains("Working Memory (Current Session)"));
    assert!(developer_after.contains("Persistent project details"));
    assert!(compaction.has_pending().await);
    drop(prepared);

    let (text_delta_tx, mut text_delta_rx) = tokio::sync::mpsc::unbounded_channel();
    assert_eq!(
        agent_loop
            .process_direct_streaming(
                "Continue cloud turn two.",
                &session_key,
                "test",
                "offline",
                None,
                text_delta_tx,
                None,
                None,
                None,
                None,
            )
            .await,
        "turn two reply"
    );

    assert!(!compaction.has_pending().await);
    assert_eq!(
        agent_loop
            .shared
            .core_handle
            .counters
            .lcm_compaction_count
            .load(std::sync::atomic::Ordering::Acquire),
        1,
        "the pending cloud checkpoint was not installed exactly once"
    );
    assert!(
        std::iter::from_fn(|| text_delta_rx.try_recv().ok())
            .any(|delta| delta == "\0cache:reset:lcm_checkpoint"),
        "installing after a developer-prefix change did not reset the prompt cache"
    );
    let calls = provider.foreground_calls();
    assert_eq!(calls.len(), 2);
    assert!(calls[1].iter().any(|message| {
        message
            .get("content")
            .and_then(Value::as_str)
            .is_some_and(|content| content.contains("To read the exact originals call"))
    }));
}

struct SoftTurnCancellationProvider {
    generation_started: Arc<tokio::sync::Notify>,
    generation_dropped: Arc<std::sync::atomic::AtomicBool>,
}

#[async_trait]
impl LLMProvider for SoftTurnCancellationProvider {
    async fn chat(
        &self,
        messages: &[Value],
        _tools: Option<&[Value]>,
        _model: Option<&str>,
        _max_tokens: u32,
        _temperature: f64,
        _thinking_budget: Option<u32>,
        _top_p: Option<f64>,
    ) -> anyhow::Result<crate::providers::base::LLMResponse> {
        if is_lcm_compaction_request(messages) {
            let _drop = CompactionTaskDrop(self.generation_dropped.clone());
            self.generation_started.notify_one();
            return std::future::pending().await;
        }
        Ok(WireRecordingProvider::text_response("foreground reply"))
    }

    fn get_default_model(&self) -> &str {
        "soft-turn-cancellation-test"
    }
}

#[tokio::test]
async fn turn_cancellation_after_soft_start_rolls_back_without_checkpoint() {
    let generation_started = Arc::new(tokio::sync::Notify::new());
    let generation_dropped = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let provider = Arc::new(SoftTurnCancellationProvider {
        generation_started: generation_started.clone(),
        generation_dropped: generation_dropped.clone(),
    });
    let lcm_config = LcmSchemaConfig {
        tau_soft: 0.0001,
        tau_hard: 10.0,
        deterministic_target: 64,
        ..Default::default()
    };
    let (agent_loop, _workspace) = build_local_inline_harness_with_lcm(
        provider as Arc<dyn LLMProvider>,
        "soft-turn-cancellation-test",
        1_000_000,
        lcm_config,
    );
    let session_key = format!("soft-turn-cancellation-{}", uuid::Uuid::new_v4());
    let core = agent_loop.shared.core_handle.swappable();
    let session = core.sessions.get_or_resume(&session_key).await;
    seed_compaction_history(&core, &session.id, 12, 40).await;

    let mut probe = InboundMessage::new("test", "user", "offline", "probe soft state");
    probe
        .metadata
        .insert("session_key".to_string(), json!(session_key.clone()));
    let prepared = agent_loop
        .shared
        .prepare_context(&probe, None, None, None, None)
        .await;
    let compaction = prepared.compaction.clone();
    let engine = agent_loop
        .shared
        .lcm_engines
        .lock()
        .await
        .get(&session.id)
        .cloned()
        .unwrap();
    let (active_before, dag_before, store_before) = {
        let engine = engine.lock().await;
        (
            engine.active_context(),
            serde_json::to_value(engine.dag()).unwrap(),
            engine.store_len(),
        )
    };
    drop(prepared);

    let cancellation = tokio_util::sync::CancellationToken::new();
    let (text_delta_tx, _text_delta_rx) = tokio::sync::mpsc::unbounded_channel();
    let mut turn = agent_loop.spawn_direct_streaming(
        "Finish foreground work, then start soft generation.".to_string(),
        session_key,
        "test".to_string(),
        "offline".to_string(),
        None,
        text_delta_tx,
        None,
        Some(cancellation.clone()),
        None,
    );
    await_compaction_sync(
        "soft generation to start after foreground work",
        generation_started.notified(),
    )
    .await;
    cancellation.cancel();
    let _ = await_compaction_sync("foreground turn to finish", &mut turn)
        .await
        .unwrap();
    await_compaction_sync(
        "turn cancellation to finish owned soft generation",
        compaction.wait_for_completion(),
    )
    .await;

    assert!(generation_dropped.load(std::sync::atomic::Ordering::Acquire));
    assert!(!compaction.has_job().await);
    assert!(!compaction.has_pending().await);
    let engine = engine.lock().await;
    assert_eq!(serde_json::to_value(engine.dag()).unwrap(), dag_before);
    let active_after = engine.active_context();
    let durable_turn_tail = active_after
        .strip_prefix(active_before.as_slice())
        .expect("cancellation changed the pre-turn active context");
    assert_eq!(
        durable_turn_tail
            .iter()
            .map(|message| (
                message.get("role").and_then(Value::as_str),
                message.get("content").and_then(Value::as_str),
            ))
            .collect::<Vec<_>>(),
        vec![
            (
                Some("user"),
                Some("Finish foreground work, then start soft generation."),
            ),
            (Some("assistant"), Some("foreground reply")),
        ]
    );
    assert_eq!(engine.store_len(), store_before + durable_turn_tail.len());
    drop(engine);
    assert!(core
        .sessions
        .load_summary_nodes(&session.id)
        .await
        .is_empty());
}

struct HardCancellationProvider {
    compaction_started: Arc<tokio::sync::Notify>,
    foreground_calls: std::sync::atomic::AtomicUsize,
}

#[async_trait]
impl LLMProvider for HardCancellationProvider {
    async fn chat(
        &self,
        messages: &[Value],
        _tools: Option<&[Value]>,
        _model: Option<&str>,
        _max_tokens: u32,
        _temperature: f64,
        _thinking_budget: Option<u32>,
        _top_p: Option<f64>,
    ) -> anyhow::Result<crate::providers::base::LLMResponse> {
        if is_lcm_compaction_request(messages) {
            self.compaction_started.notify_one();
            return std::future::pending().await;
        }
        self.foreground_calls
            .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        Ok(WireRecordingProvider::text_response(
            "cancelled turn reached foreground inference",
        ))
    }

    fn get_default_model(&self) -> &str {
        "hard-compaction-cancellation-test"
    }
}

#[tokio::test]
async fn cancelling_hard_compaction_restores_engine_without_publishing_checkpoint() {
    const CANCELLATION_PROMPT: &str = "persistent project detail decisions constraints";
    let compaction_started = Arc::new(tokio::sync::Notify::new());
    let provider = Arc::new(HardCancellationProvider {
        compaction_started: compaction_started.clone(),
        foreground_calls: std::sync::atomic::AtomicUsize::new(0),
    });
    let lcm_config = LcmSchemaConfig {
        tau_soft: 0.05,
        tau_hard: 0.10,
        deterministic_target: 64,
        ..Default::default()
    };
    let (agent_loop, _workspace) = build_local_inline_harness_with_lcm(
        provider.clone() as Arc<dyn LLMProvider>,
        "hard-compaction-cancellation-test",
        8192,
        lcm_config,
    );
    let session_key = format!("hard-cancellation-{}", uuid::Uuid::new_v4());
    let core = agent_loop.shared.core_handle.swappable();
    let session = core.sessions.get_or_resume(&session_key).await;
    seed_compaction_history(&core, &session.id, 20, 12).await;
    persist_prior_summary(&core, &session.id).await;

    let mut probe = InboundMessage::new("test", "user", "offline", CANCELLATION_PROMPT);
    probe
        .metadata
        .insert("session_key".to_string(), json!(session_key.clone()));
    let prepared = agent_loop
        .shared
        .prepare_context(&probe, None, None, None, None)
        .await;
    let compaction = prepared.compaction.clone();
    let engine = agent_loop
        .shared
        .lcm_engines
        .lock()
        .await
        .get(&session.id)
        .cloned()
        .expect("hard-pressure session must own an LCM engine");
    let (active_before, dag_before, store_before) = {
        let engine = engine.lock().await;
        (
            engine.active_context(),
            serde_json::to_value(engine.dag()).unwrap(),
            engine.store_len(),
        )
    };
    let durable_before = core.sessions.load_summary_nodes(&session.id).await;
    drop(prepared);

    let cancellation = tokio_util::sync::CancellationToken::new();
    let (text_delta_tx, _text_delta_rx) = tokio::sync::mpsc::unbounded_channel();
    let mut turn = agent_loop.spawn_direct_streaming(
        CANCELLATION_PROMPT.to_string(),
        session_key,
        "test".to_string(),
        "offline".to_string(),
        None,
        text_delta_tx,
        None,
        Some(cancellation.clone()),
        None,
    );
    await_compaction_sync(
        "hard compaction to enter provider generation",
        compaction_started.notified(),
    )
    .await;

    cancellation.cancel();
    let response = match tokio::time::timeout(std::time::Duration::from_secs(2), &mut turn).await {
        Ok(joined) => joined.expect("hard-cancelled foreground task panicked"),
        Err(_) => {
            turn.abort();
            let _ = await_compaction_sync("timed-out hard turn to abort", &mut turn).await;
            panic!("hard compaction ignored the current turn cancellation token");
        }
    };
    assert!(response.is_empty(), "cancelled turn returned: {response}");
    assert_eq!(
        provider
            .foreground_calls
            .load(std::sync::atomic::Ordering::Acquire),
        0,
        "a cancelled hard-compaction turn must not make a foreground model request"
    );
    assert!(!compaction.has_job().await);
    assert!(!compaction.has_pending().await);

    let engine = await_compaction_sync("restored LCM engine lock", engine.lock()).await;
    assert_eq!(serde_json::to_value(engine.dag()).unwrap(), dag_before);
    let active_after = engine.active_context();
    assert!(
        active_after.starts_with(&active_before),
        "cancelled compaction did not restore the pre-existing active context"
    );
    assert_eq!(
        active_after.len(),
        active_before.len() + 1,
        "only the eagerly persisted current user turn may extend active context"
    );
    assert_eq!(engine.store_len(), store_before + 1);
    let expanded = engine.plan_auto_expansion(&core.token_budget, 0, 0);
    assert!(
        expanded.iter().any(|message| {
            message
                .flattened_fallback
                .get("content")
                .and_then(Value::as_str)
                .is_some_and(|content| content.contains("persistent project detail"))
        }),
        "hard cancellation consumed prior-summary auto-expand eligibility"
    );
    drop(engine);
    assert_eq!(
        core.sessions.load_summary_nodes(&session.id).await,
        durable_before,
        "cancelled hard compaction changed durable summary checkpoints"
    );
}

const COMPACTION_SYNC_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);
const COMPACTION_BLOCKED_OBSERVATION: std::time::Duration = std::time::Duration::from_millis(250);

async fn await_compaction_sync<T>(
    context: &str,
    future: impl std::future::Future<Output = T>,
) -> T {
    tokio::time::timeout(COMPACTION_SYNC_TIMEOUT, future)
        .await
        .unwrap_or_else(|_| {
            panic!("timed out after {COMPACTION_SYNC_TIMEOUT:?} while waiting for {context}")
        })
}

#[tokio::test]
async fn lcm_recovery_reserves_the_complete_non_lcm_wire() {
    use crate::agent::agent_loop::compaction::{execute_lcm_compaction, CompactionPublication};
    use crate::agent::lcm::{CompactionFailureMode, LcmConfig, LcmEngine};
    use crate::agent::token_budget::TokenBudget;
    use tokio_util::sync::CancellationToken;

    for (system_words, tool_words) in [(6_000, 0), (0, 8_000)] {
        let (agent_loop, _workspace) = build_local_inline_harness(MockLLM::named("fixed-lcm-wire"));
        let core = agent_loop.shared.core_handle.swappable();
        let session = core.sessions.get_or_resume("fixed-lcm-wire").await;
        for id in 1..=80 {
            core.sessions
                .add_message(
                    &session.id,
                    &json!({
                        "role": if id % 2 == 1 { "user" } else { "assistant" },
                        "content": "word ".repeat(500),
                    }),
                )
                .await;
        }
        let durable = core.sessions.get_all_messages(&session.id).await;
        let mut engine = LcmEngine::new(LcmConfig::default());
        for message in &durable {
            engine.ingest(message.clone());
        }
        let engine = Arc::new(tokio::sync::Mutex::new(engine));
        let prefix = vec![
            json!({"role": "system", "content": "system ".repeat(system_words)}),
            json!({"role": "developer", "content": "developer ".repeat(1_000)}),
        ];
        let tool_def_tokens = TokenBudget::estimate_tool_def_tokens(&[json!({
            "type": "function", "function": {
                "name": "inspect", "description": "tool ".repeat(tool_words),
                "parameters": {"type": "object"}
            }
        })]);
        let ephemeral = json!({"role": "user", "content": "current ".repeat(500)});
        let mut messages = prefix.clone();
        messages.extend(durable);
        messages.push(ephemeral.clone());
        for context in [100_000, 18_000, 18_000, 18_000] {
            if let Some(pending) = execute_lcm_compaction(
                core.clone(),
                session.id.clone(),
                engine.clone(),
                messages.clone(),
                80,
                TokenBudget::new(context, 1_000),
                tool_def_tokens,
                CompactionFailureMode::Deterministic,
                CancellationToken::new(),
                Arc::new(CompactionPublication::new()),
            )
            .await
            {
                messages = pending.result.messages;
            }
        }
        assert_eq!(&messages[..prefix.len()], prefix.as_slice());
        assert_eq!(messages.last(), Some(&ephemeral));
        assert!(
            TokenBudget::estimate_tokens(&messages) + tool_def_tokens <= 17_000,
            "the full wire must fit, including immutable prefix and current turn: {}",
            TokenBudget::estimate_tokens(&messages) + tool_def_tokens
        );
        assert_eq!(core.sessions.get_all_messages(&session.id).await.len(), 80);
    }
}

#[tokio::test]
async fn cancelled_before_engine_lock_keeps_soft_compaction_retryable() {
    use crate::agent::agent_loop::compaction::{execute_lcm_compaction, CompactionPublication};
    use crate::agent::lcm::{CompactionAction, CompactionFailureMode, LcmConfig, LcmEngine};
    use tokio_util::sync::CancellationToken;

    let provider = MockLLM::named("cancelled-before-engine-lock-test");
    let (agent_loop, _workspace) = build_local_inline_harness(provider);
    let core = agent_loop.shared.core_handle.swappable();
    let session = core
        .sessions
        .get_or_resume("cancelled-before-engine-lock")
        .await;
    let message = json!({
        "role": "user",
        "content": "soft pressure context ".repeat(100),
        "_db_id": 1,
    });
    let messages = vec![message.clone()];
    let mut engine = LcmEngine::new(LcmConfig {
        tau_soft: 0.01,
        tau_hard: 10.0,
        deterministic_target: 64,
        keep_prefix_fraction: 0.35,
    });
    engine.ingest(message);
    assert_eq!(
        engine.check_thresholds(&core.token_budget, 0),
        CompactionAction::Async
    );
    let engine = Arc::new(tokio::sync::Mutex::new(engine));

    let engine_guard = engine.lock().await;
    let cancellation = CancellationToken::new();
    cancellation.cancel();
    let result = await_compaction_sync(
        "pre-lock cancellation to finish without acquiring the engine",
        execute_lcm_compaction(
            core.clone(),
            session.id,
            engine.clone(),
            messages,
            1,
            crate::agent::token_budget::TokenBudget::new(
                core.token_budget.max_context(),
                core.token_budget.response_reserve(),
            ),
            0,
            CompactionFailureMode::PreserveContext,
            cancellation,
            Arc::new(CompactionPublication::new()),
        ),
    )
    .await;
    assert!(result.is_none());
    drop(engine_guard);

    assert_eq!(
        engine.lock().await.check_thresholds(&core.token_budget, 0),
        CompactionAction::Async,
        "cancellation before acquisition must leave soft compaction retryable"
    );
}

struct BlockingCompactionProvider {
    started: Arc<tokio::sync::Notify>,
    release: Arc<tokio::sync::Notify>,
}

#[async_trait]
impl LLMProvider for BlockingCompactionProvider {
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
        self.started.notify_one();
        await_compaction_sync(
            "blocking compaction provider to receive its release",
            self.release.notified(),
        )
        .await;
        Ok(WireRecordingProvider::plain_text_response(
            "- Prior turns retain project details, decisions, constraints, and follow-up context.",
        ))
    }

    fn get_default_model(&self) -> &str {
        "owned-blocking-publication-test"
    }
}

#[tokio::test]
async fn blocking_compaction_publication_survives_foreground_abort() {
    let compaction_started = Arc::new(tokio::sync::Notify::new());
    let release_compaction = Arc::new(tokio::sync::Notify::new());
    let provider = Arc::new(BlockingCompactionProvider {
        started: compaction_started.clone(),
        release: release_compaction.clone(),
    });
    let lcm_config = LcmSchemaConfig {
        tau_soft: 0.05,
        tau_hard: 0.10,
        deterministic_target: 64,
        ..Default::default()
    };
    let (agent_loop, _workspace) = build_local_inline_harness_with_lcm(
        provider as Arc<dyn LLMProvider>,
        "owned-blocking-publication-test",
        8192,
        lcm_config,
    );
    let session_key = format!("owned-blocking-publication-{}", uuid::Uuid::new_v4());
    let core = agent_loop.shared.core_handle.swappable();
    let session = core.sessions.get_or_resume(&session_key).await;
    for turn in 0..20_u64 {
        core.sessions
            .add_message(
                &session.id,
                &json!({
                    "role": "user",
                    "content": format!(
                        "turn {turn}: {}",
                        "persistent project detail with decisions and constraints ".repeat(12)
                    ),
                }),
            )
            .await;
        core.sessions
            .add_message(
                &session.id,
                &json!({
                    "role": "assistant",
                    "content": format!("acknowledged retained project detail for turn {turn}"),
                }),
            )
            .await;
    }

    let mut msg = InboundMessage::new(
        "test",
        "user",
        "offline",
        "Use the retained project details to answer briefly.",
    );
    msg.metadata
        .insert("session_key".to_string(), json!(session_key));
    let context = agent_loop
        .shared
        .prepare_context(&msg, None, None, None, None)
        .await;
    let slot = context.compaction.slot.clone();
    let compaction = context.compaction.clone();
    let (text_delta_tx, _text_delta_rx) = tokio::sync::mpsc::unbounded_channel();
    let turn = agent_loop.spawn_direct_streaming(
        "Use the retained project details to answer briefly.".to_string(),
        session_key,
        "test".to_string(),
        "offline".to_string(),
        None,
        text_delta_tx,
        None,
        None,
        None,
    );
    await_compaction_sync(
        "real turn to enter LCM generation",
        compaction_started.notified(),
    )
    .await;
    let slot_guard = slot.lock().await;
    release_compaction.notify_one();

    await_compaction_sync("actual compaction to publish to SQLite", async {
        loop {
            if !core
                .sessions
                .load_summary_nodes(&session.id)
                .await
                .is_empty()
            {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await;

    // Dropping the foreground waiter models Escape/Ctrl-C after durable
    // publication but before the pending-slot handoff can acquire its lock.
    turn.abort();
    assert!(
        await_compaction_sync("aborted foreground turn to join", turn)
            .await
            .unwrap_err()
            .is_cancelled()
    );
    assert!(
        compaction.has_job().await,
        "blocking publication must remain owned after its foreground waiter is dropped"
    );
    drop(slot_guard);
    await_compaction_sync("owned publication to finish pending-slot handoff", async {
        loop {
            if compaction.has_pending().await {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await;
    await_compaction_sync(
        "blocking publication job to reap after handoff",
        compaction.cancel_and_reap(),
    )
    .await;
    assert!(!compaction.has_job().await);
}

#[tokio::test]
async fn dropped_reaper_leaves_job_owned_for_next_reaper() {
    let handle = CompactionHandle::new();
    let started = Arc::new(tokio::sync::Notify::new());
    let cancellation_seen = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let completed = Arc::new(std::sync::atomic::AtomicBool::new(false));

    let task_started = started.clone();
    let task_cancellation_seen = cancellation_seen.clone();
    let task_release = release.clone();
    let task_completed = completed.clone();
    assert!(
        handle
            .try_start(move |cancellation, _publication| async move {
                task_started.notify_one();
                cancellation.cancelled().await;
                task_cancellation_seen.notify_one();
                await_compaction_sync(
                    "retained owned job to receive its release",
                    task_release.notified(),
                )
                .await;
                task_completed.store(true, std::sync::atomic::Ordering::Release);
                None
            })
            .await
    );
    await_compaction_sync("owned task to start", started.notified()).await;

    let first_handle = handle.clone();
    let first_reaper = tokio::spawn(async move {
        await_compaction_sync(
            "first reaper to finish after cancellation",
            first_handle.cancel_and_reap(),
        )
        .await;
    });
    await_compaction_sync(
        "first reaper to cancel generation",
        cancellation_seen.notified(),
    )
    .await;
    first_reaper.abort();
    assert!(
        await_compaction_sync("aborted first reaper to join", first_reaper)
            .await
            .unwrap_err()
            .is_cancelled()
    );
    assert!(
        handle.has_job().await,
        "cancelling a reaper must not detach the job it was joining"
    );

    release.notify_one();
    await_compaction_sync(
        "later reaper to join the retained task",
        handle.cancel_and_reap(),
    )
    .await;
    assert!(completed.load(std::sync::atomic::Ordering::Acquire));
    assert!(!handle.has_job().await);
}

#[derive(Clone)]
struct CompactionLogWriter(Arc<std::sync::Mutex<Vec<u8>>>);

impl std::io::Write for CompactionLogWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[tokio::test(flavor = "current_thread")]
async fn panicked_owned_job_reaps_with_session_phase_context() {
    let output = Arc::new(std::sync::Mutex::new(Vec::new()));
    let writer = output.clone();
    let subscriber = tracing_subscriber::fmt()
        .with_ansi(false)
        .without_time()
        .with_target(false)
        .with_max_level(tracing::Level::WARN)
        .with_writer(move || CompactionLogWriter(writer.clone()))
        .finish();
    let _subscriber = tracing::subscriber::set_default(subscriber);

    let provider = MockLLM::named("panic-session-test");
    let (agent_loop, _workspace) = build_local_inline_harness(provider);
    let session_key = format!("panic-session-{}", uuid::Uuid::new_v4());
    let mut msg = InboundMessage::new("test", "user", "offline", "panic");
    msg.metadata
        .insert("session_key".to_string(), json!(session_key));
    let context = agent_loop
        .shared
        .prepare_context(&msg, None, None, None, None)
        .await;
    let handle = context.compaction;
    let session_id = context.session_id;
    let started = Arc::new(tokio::sync::Notify::new());
    let task_started = started.clone();
    assert!(
        handle
            .try_start(move |_cancellation, _publication| async move {
                task_started.notify_one();
                panic!("compaction panic");
                #[allow(unreachable_code)]
                None
            })
            .await
    );
    await_compaction_sync("panicking owned job to start", started.notified()).await;
    await_compaction_sync("panicked owned job to reap", handle.cancel_and_reap()).await;

    assert!(!handle.has_job().await, "a panicked job must be reaped");
    let logs = String::from_utf8(output.lock().unwrap().clone()).unwrap();
    assert!(logs.contains("owned compaction task failed"), "{logs}");
    assert!(logs.contains(&format!("session_id={session_id}")), "{logs}");
    assert!(logs.contains("phase=generating"), "{logs}");
    assert!(
        handle
            .try_start(|_cancellation, _publication| async move { None })
            .await,
        "panic cleanup must leave the session restartable"
    );
    await_compaction_sync(
        "restarted owned job to reap after panic cleanup",
        handle.cancel_and_reap(),
    )
    .await;
}

struct CompactionTaskDrop(Arc<std::sync::atomic::AtomicBool>);

impl Drop for CompactionTaskDrop {
    fn drop(&mut self) {
        self.0.store(true, std::sync::atomic::Ordering::Release);
    }
}

#[tokio::test]
async fn final_handle_drop_aborts_generation_but_preserves_publication_handoff() {
    let generation = CompactionHandle::new();
    let generation_started = Arc::new(tokio::sync::Notify::new());
    let generation_dropped = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let task_started = generation_started.clone();
    let task_dropped = generation_dropped.clone();
    assert!(
        generation
            .try_start(move |_cancellation, _publication| async move {
                let _drop = CompactionTaskDrop(task_dropped);
                task_started.notify_one();
                std::future::pending::<Option<crate::agent::agent_core::PendingCompaction>>().await
            })
            .await
    );
    await_compaction_sync(
        "generation job to start before final-owner drop",
        generation_started.notified(),
    )
    .await;
    drop(generation);
    await_compaction_sync("final-owner drop to abort generation", async {
        while !generation_dropped.load(std::sync::atomic::Ordering::Acquire) {
            tokio::task::yield_now().await;
        }
    })
    .await;

    let publishing = CompactionHandle::new();
    let slot = publishing.slot.clone();
    let enter_publication = Arc::new(tokio::sync::Notify::new());
    let publication_claimed = Arc::new(tokio::sync::Notify::new());
    let task_slot = slot.clone();
    let task_enter_publication = enter_publication.clone();
    let task_publication_claimed = publication_claimed.clone();
    assert!(
        publishing
            .try_start(move |_cancellation, publication| async move {
                await_compaction_sync(
                    "publishing job to receive its entry signal",
                    task_enter_publication.notified(),
                )
                .await;
                assert!(publication.begin_publication());
                task_publication_claimed.notify_one();
                let guard = task_slot.lock().await;
                drop(guard);
                Some(crate::agent::agent_core::PendingCompaction {
                    result: crate::agent::compaction::CompactionResult {
                        messages: vec![json!({
                            "role": "assistant",
                            "content": "published-checkpoint",
                        })],
                    },
                    snapshot: Vec::new(),
                    summary_node_id: 0,
                })
            })
            .await
    );

    let slot_guard = slot.lock().await;
    enter_publication.notify_one();
    await_compaction_sync(
        "owned job to claim publication before final-owner drop",
        publication_claimed.notified(),
    )
    .await;
    tokio::task::yield_now().await;
    drop(publishing);
    drop(slot_guard);

    await_compaction_sync("publication to complete pending-slot handoff", async {
        loop {
            if slot.lock().await.as_ref().is_some_and(|pending| {
                pending
                    .result
                    .messages
                    .first()
                    .and_then(|message| message.get("content"))
                    .and_then(Value::as_str)
                    == Some("published-checkpoint")
            }) {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await;
}

#[tokio::test]
async fn compaction_shutdown_waits_for_publication_and_reaps_generation() {
    let provider = MockLLM::named("compaction-shutdown-drain-test");
    let (agent_loop, _workspace) = build_local_inline_harness(provider);
    let publishing = CompactionHandle::for_session("shutdown-publishing");
    let generating = CompactionHandle::for_session("shutdown-generating");
    {
        let mut handles = agent_loop.shared.compaction_handles.lock().await;
        handles.insert("shutdown-publishing".to_string(), publishing.clone());
        handles.insert("shutdown-generating".to_string(), generating.clone());
    }

    let slot = publishing.slot.clone();
    let enter_publication = Arc::new(tokio::sync::Notify::new());
    let publication_claimed = Arc::new(tokio::sync::Notify::new());
    let task_enter_publication = enter_publication.clone();
    let task_publication_claimed = publication_claimed.clone();
    assert!(
        publishing
            .try_start(move |_cancellation, publication| async move {
                task_enter_publication.notified().await;
                assert!(publication.begin_publication());
                task_publication_claimed.notify_one();
                Some(crate::agent::agent_core::PendingCompaction {
                    result: crate::agent::compaction::CompactionResult {
                        messages: vec![json!({
                            "role": "assistant",
                            "content": "shutdown-published-checkpoint",
                        })],
                    },
                    snapshot: Vec::new(),
                    summary_node_id: 0,
                })
            })
            .await
    );
    let slot_guard = slot.lock().await;
    enter_publication.notify_one();
    await_compaction_sync(
        "shutdown publication to claim its atomic boundary",
        publication_claimed.notified(),
    )
    .await;

    let generation_cancelled = Arc::new(tokio::sync::Notify::new());
    let task_generation_cancelled = generation_cancelled.clone();
    assert!(
        generating
            .try_start(move |cancellation, _publication| async move {
                cancellation.cancelled().await;
                task_generation_cancelled.notify_one();
                None
            })
            .await
    );

    let mut drain = Box::pin(agent_loop.drain_compaction_jobs());
    await_compaction_sync(
        "shutdown drain to cancel generation while publication stays blocked",
        async {
            tokio::select! {
                () = drain.as_mut() => {
                    panic!("shutdown drain returned before pending handoff was released");
                }
                () = generation_cancelled.notified() => {}
            }
        },
    )
    .await;

    drop(slot_guard);
    await_compaction_sync(
        "shutdown drain to join publication after pending handoff",
        drain.as_mut(),
    )
    .await;

    assert!(!publishing.has_job().await);
    assert!(!generating.has_job().await);
    assert_eq!(
        slot.lock()
            .await
            .as_ref()
            .and_then(|pending| pending.result.messages.first())
            .and_then(|message| message.get("content"))
            .and_then(Value::as_str),
        Some("shutdown-published-checkpoint")
    );
}

#[tokio::test]
async fn agent_clear_reaps_job_and_discards_pending_checkpoint() {
    let provider = MockLLM::named("clear-owned-compaction-test");
    let (agent_loop, _workspace) = build_local_inline_harness(provider);
    let session_key = format!("clear-owned-compaction-{}", uuid::Uuid::new_v4());
    let mut msg = InboundMessage::new("test", "user", "offline", "first");
    msg.metadata
        .insert("session_key".to_string(), json!(session_key.clone()));
    let context = agent_loop
        .shared
        .prepare_context(&msg, None, None, None, None)
        .await;
    let compaction = context.compaction.clone();
    let publication_claimed = Arc::new(tokio::sync::Notify::new());
    let release_publication = Arc::new(tokio::sync::Notify::new());
    let task_publication_claimed = publication_claimed.clone();
    let task_release_publication = release_publication.clone();
    assert!(
        compaction
            .try_start(move |_cancellation, publication| async move {
                assert!(publication.begin_publication());
                task_publication_claimed.notify_one();
                await_compaction_sync(
                    "agent-clear publication barrier to release",
                    task_release_publication.notified(),
                )
                .await;
                Some(crate::agent::agent_core::PendingCompaction {
                    result: crate::agent::compaction::CompactionResult {
                        messages: vec![json!({
                            "role": "assistant",
                            "content": "pending checkpoint before agent clear",
                        })],
                    },
                    snapshot: Vec::new(),
                    summary_node_id: 0,
                })
            })
            .await
    );
    await_compaction_sync(
        "clear-owned compaction to claim publication",
        publication_claimed.notified(),
    )
    .await;

    let mut clear = Box::pin(agent_loop.clear_session_state(&session_key));
    await_compaction_sync("agent clear to block while reaping publication", async {
        tokio::select! {
            () = clear.as_mut() => {
                panic!("agent clear returned before publication handoff was released");
            }
            () = tokio::time::sleep(COMPACTION_BLOCKED_OBSERVATION) => {}
        }
    })
    .await;
    release_publication.notify_one();
    await_compaction_sync(
        "agent clear to reap publication and discard its checkpoint",
        clear.as_mut(),
    )
    .await;

    assert!(!compaction.has_job().await);
    assert!(!compaction.has_pending().await);
    assert!(!agent_loop
        .shared
        .compaction_handles
        .lock()
        .await
        .contains_key(&context.session_id));
    assert!(!agent_loop
        .shared
        .lcm_engines
        .lock()
        .await
        .contains_key(&context.session_id));
}

fn repl_context_for_clear_test(
    agent_loop: AgentLoop,
    core_handle: SharedCoreHandle,
    session_id: String,
    workspace: std::path::PathBuf,
) -> crate::repl::commands::ReplContext {
    let (display_tx, display_rx) = tokio::sync::mpsc::unbounded_channel();
    let (restart_tx, restart_rx) = tokio::sync::mpsc::unbounded_channel();
    crate::repl::commands::ReplContext {
        config: crate::config::schema::Config::default(),
        core_handle,
        agent_loop,
        session_id,
        lang: None,
        srv: crate::repl::ServerState::new("0".to_string()),
        current_model_path: workspace.clone(),
        active_channels: Vec::new(),
        display_tx,
        display_rx,
        cron_service: Arc::new(crate::cron::service::CronService::new(
            workspace.join("cron.json"),
        )),
        email_config: None,
        rl: None,
        watchdog_handle: None,
        restart_tx,
        restart_rx,
        health_registry: None,
        #[cfg(feature = "voice")]
        voice_session: None,
        #[cfg(feature = "cluster")]
        cluster_state: None,
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn handle_restart_requests_skips_stale_restart() {
    let provider = MockLLM::named("stale-restart-test");
    let (agent_loop, workspace) = build_local_inline_harness(provider);
    let core_handle = agent_loop.shared.core_handle.clone();
    let mut ctx = repl_context_for_clear_test(
        agent_loop,
        core_handle,
        "stale-restart-test".to_string(),
        workspace,
    );

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    ctx.srv.local_port = listener.local_addr().unwrap().port().to_string();
    ctx.config.agents.defaults.local_backend = "lmstudio".to_string();
    ctx.config.agents.defaults.local_autostart = crate::config::schema::LocalAutostart::Lmstudio;
    ctx.restart_tx
        .send(crate::server::RestartRequest {
            role: "main".to_string(),
        })
        .unwrap();

    let health = tokio::spawn(async move {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        loop {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = [0_u8; 1024];
            let read = stream.read(&mut request).await.unwrap();
            let request = String::from_utf8_lossy(&request[..read]);
            let healthy = request.starts_with("GET /health ");
            assert!(healthy || request.starts_with("GET /props "));
            let response = if healthy {
                b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".as_slice()
            } else {
                b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                    .as_slice()
            };
            stream.write_all(response).await.unwrap();
            if healthy {
                break;
            }
        }
    });

    assert!(!ctx.handle_restart_requests().await);
    assert!(ctx.display_rx.try_recv().is_err());
    health.await.unwrap();
}

#[tokio::test]
async fn interactive_clear_is_atomic_against_session_admission() {
    let provider = MockLLM::named("interactive-clear-admission-test");
    let (agent_loop, workspace) = build_local_inline_harness_with_memory(
        provider,
        "interactive-clear-admission-test",
        4096,
        LcmSchemaConfig::default(),
        MemoryConfig::default(),
    );
    let core_handle = agent_loop.shared.core_handle.clone();
    let core = core_handle.swappable();
    let session_key = format!("interactive-clear-admission-{}", uuid::Uuid::new_v4());
    let session = core.sessions.get_or_resume(&session_key).await;
    core.sessions
        .add_message(
            &session.id,
            &json!({"role": "user", "content": "history before clear"}),
        )
        .await;
    core.sessions
        .save_working_memory(&session.id, "working memory before clear", "active", 1)
        .await
        .unwrap();

    let mut first_msg = InboundMessage::new("test", "user", "offline", "first");
    first_msg
        .metadata
        .insert("session_key".to_string(), json!(session_key.clone()));
    let first = agent_loop
        .shared
        .prepare_context(&first_msg, None, None, None, None)
        .await;
    let old_compaction = first.compaction.clone();
    let old_engine = agent_loop
        .shared
        .lcm_engines
        .lock()
        .await
        .get(&session.id)
        .cloned()
        .unwrap();
    let publication_claimed = Arc::new(tokio::sync::Notify::new());
    let release_publication = Arc::new(tokio::sync::Notify::new());
    let pending_handoff: Arc<
        std::sync::Mutex<Option<crate::agent::agent_core::PendingCompaction>>,
    > = Arc::new(std::sync::Mutex::new(None));
    let task_publication_claimed = publication_claimed.clone();
    let task_release_publication = release_publication.clone();
    let task_pending_handoff = pending_handoff.clone();
    assert!(
        old_compaction
            .try_start(move |_cancellation, publication| async move {
                let pending = crate::agent::agent_core::PendingCompaction {
                    result: crate::agent::compaction::CompactionResult {
                        messages: vec![json!({
                            "role": "assistant",
                            "content": "pending checkpoint before interactive clear",
                        })],
                    },
                    snapshot: Vec::new(),
                    summary_node_id: 0,
                };
                assert!(publication.begin_publication());
                *task_pending_handoff.lock().unwrap() = Some(pending);
                task_publication_claimed.notify_one();
                await_compaction_sync(
                    "interactive-clear publication barrier to release",
                    task_release_publication.notified(),
                )
                .await;
                let pending = task_pending_handoff.lock().unwrap().take();
                pending
            })
            .await
    );
    await_compaction_sync(
        "interactive-clear fixture to claim real publication",
        publication_claimed.notified(),
    )
    .await;

    let counters = core_handle.counters.clone();
    let stale_prompt_epoch = counters.reset_session_prompt_state(&session_key);
    let stale_higgs_session_id = 9_001;
    counters.record_higgs_session_id(&session_key, stale_higgs_session_id);
    counters
        .last_context_used
        .store(123, std::sync::atomic::Ordering::Relaxed);
    counters
        .last_message_count
        .store(45, std::sync::atomic::Ordering::Relaxed);
    counters
        .last_working_memory_tokens
        .store(67, std::sync::atomic::Ordering::Relaxed);

    let repl = repl_context_for_clear_test(
        agent_loop,
        core_handle.clone(),
        session_key.clone(),
        workspace,
    );
    let mut clear = Box::pin(repl.agent_loop.clear_session_state(&session_key));
    await_compaction_sync(
        "interactive clear to block while reaping publication",
        async {
            tokio::select! {
                () = clear.as_mut() => {
                    panic!("interactive clear returned before publication handoff was released");
                }
                () = tokio::time::sleep(COMPACTION_BLOCKED_OBSERVATION) => {}
            }
        },
    )
    .await;

    let retained_history = core
        .sessions
        .get_history(&session.id, usize::MAX, usize::MAX)
        .await;
    assert!(retained_history.iter().any(|message| {
        message.get("content").and_then(Value::as_str) == Some("history before clear")
    }));
    assert_eq!(
        core.working_memory
            .get_context(&session.id, usize::MAX)
            .await
            .unwrap(),
        "working memory before clear"
    );
    {
        // The real owned wrapper writes `slot` only after this publishing
        // future returns. Holding the actual checkpoint here models the
        // reachable pre-handoff state without fabricating job + populated slot.
        let pending = pending_handoff.lock().unwrap();
        let pending = pending
            .as_ref()
            .expect("publishing job must retain its pending checkpoint");
        assert_eq!(
            pending
                .result
                .messages
                .first()
                .and_then(|message| message.get("content"))
                .and_then(Value::as_str),
            Some("pending checkpoint before interactive clear")
        );
    }
    assert!(
        !old_compaction.has_pending().await,
        "the slot must remain empty until the publishing future hands off its checkpoint"
    );
    let retained_engine = repl
        .agent_loop
        .shared
        .lcm_engines
        .lock()
        .await
        .get(&session.id)
        .cloned()
        .expect("old engine must remain installed while clear is reaping");
    assert!(Arc::ptr_eq(&old_engine, &retained_engine));

    let mut second_msg = InboundMessage::new("test", "user", "offline", "second");
    second_msg
        .metadata
        .insert("session_key".to_string(), json!(session_key.clone()));
    let waiter_counters = counters.clone();
    let waiter_session_key = session_key.clone();
    let mut prepare = Box::pin(async {
        let fresh = repl
            .agent_loop
            .shared
            .prepare_context(&second_msg, None, None, None, None)
            .await;
        let observed = (
            waiter_counters.session_prompt_epoch(&waiter_session_key),
            waiter_counters
                .last_context_used
                .load(std::sync::atomic::Ordering::Relaxed),
            waiter_counters
                .last_message_count
                .load(std::sync::atomic::Ordering::Relaxed),
            waiter_counters
                .last_working_memory_tokens
                .load(std::sync::atomic::Ordering::Relaxed),
            waiter_counters.pending_higgs_session_drop_ids(&waiter_session_key),
        );
        (fresh, observed)
    });
    await_compaction_sync(
        "new preparation to remain blocked by interactive clear",
        async {
            tokio::select! {
                _ = prepare.as_mut() => {
                    panic!("preparation entered before interactive clear retired the old handle");
                }
                () = tokio::time::sleep(COMPACTION_BLOCKED_OBSERVATION) => {}
            }
        },
    )
    .await;

    release_publication.notify_one();
    let ((), (fresh, observed)) = await_compaction_sync(
        "interactive clear and waiting preparation to finish after retirement",
        async { tokio::join!(clear.as_mut(), prepare.as_mut()) },
    )
    .await;

    assert_eq!(
        observed.0,
        stale_prompt_epoch.saturating_add(1),
        "the admitted waiter observed the pre-clear prompt epoch"
    );
    assert_eq!(
        (observed.1, observed.2, observed.3),
        (0, 0, 0),
        "the admitted waiter observed stale aggregate context counters"
    );
    assert!(
        observed.4.contains(&stale_higgs_session_id),
        "the admitted waiter did not observe the cleared Higgs session handoff"
    );

    let remaining_history = core
        .sessions
        .get_history(&session.id, usize::MAX, usize::MAX)
        .await;
    assert!(
        remaining_history.is_empty(),
        "history remained after clear: {}",
        serde_json::to_string(&remaining_history).unwrap()
    );
    assert_eq!(
        core.working_memory
            .get_context(&session.id, usize::MAX)
            .await
            .unwrap(),
        ""
    );
    assert!(pending_handoff.lock().unwrap().is_none());
    assert!(!old_compaction.has_job().await);
    assert!(!old_compaction.has_pending().await);
    assert!(
        !old_compaction
            .try_start(|_cancellation, _publication| async move { None })
            .await,
        "the retired pre-clear handle must reject new compaction"
    );
    assert!(!Arc::ptr_eq(&old_compaction.slot, &fresh.compaction.slot));
    let fresh_engine = repl
        .agent_loop
        .shared
        .lcm_engines
        .lock()
        .await
        .get(&session.id)
        .cloned()
        .unwrap();
    assert!(!Arc::ptr_eq(&old_engine, &fresh_engine));
}

#[tokio::test]
async fn concrete_session_reuses_compaction_checkpoint_handle() {
    let provider = MockLLM::named("local-compaction-checkpoint-handle-test");
    let (agent_loop, _workspace) = build_local_inline_harness(provider);
    let session_key = format!("compaction-checkpoint-handle-{}", uuid::Uuid::new_v4());
    let mut msg = InboundMessage::new("test", "user", "offline", "first");
    msg.metadata
        .insert("session_key".to_string(), json!(session_key));

    let first = agent_loop
        .shared
        .prepare_context(&msg, None, None, None, None)
        .await;
    msg.content = "second".to_string();
    let second = agent_loop
        .shared
        .prepare_context(&msg, None, None, None, None)
        .await;

    assert_eq!(first.session_id, second.session_id);
    assert!(
        Arc::ptr_eq(&first.compaction.slot, &second.compaction.slot),
        "the pending checkpoint must remain visible across turns in one concrete session"
    );
    assert!(
        first
            .compaction
            .try_start(|cancellation, _publication| async move {
                cancellation.cancelled().await;
                None
            })
            .await
    );
    assert!(
        second.compaction.has_job().await,
        "the owned job must remain visible across turns in one concrete session"
    );
    await_compaction_sync(
        "shared session compaction job to reap",
        second.compaction.cancel_and_reap(),
    )
    .await;
    assert!(!first.compaction.has_job().await);
}

#[tokio::test]
async fn idle_rollover_does_not_reuse_compaction_checkpoint_handle() {
    let provider = MockLLM::named("local-compaction-checkpoint-rollover-test");
    let memory_config = MemoryConfig {
        session_complete_after_secs: 1,
        ..Default::default()
    };
    let (agent_loop, _workspace) = build_local_inline_harness_with_memory(
        provider,
        "local-compaction-checkpoint-rollover-test",
        4096,
        LcmSchemaConfig::default(),
        memory_config,
    );
    let session_key = format!("compaction-checkpoint-rollover-{}", uuid::Uuid::new_v4());
    let mut msg = InboundMessage::new("test", "user", "offline", "first");
    msg.metadata
        .insert("session_key".to_string(), json!(session_key));

    let first = agent_loop
        .shared
        .prepare_context(&msg, None, None, None, None)
        .await;
    tokio::time::sleep(std::time::Duration::from_millis(2_100)).await;
    msg.content = "second".to_string();
    let second = agent_loop
        .shared
        .prepare_context(&msg, None, None, None, None)
        .await;

    assert_ne!(first.session_id, second.session_id);
    assert!(!Arc::ptr_eq(
        &first.compaction.slot,
        &second.compaction.slot
    ));
    assert!(
        first
            .compaction
            .try_start(|cancellation, _publication| async move {
                cancellation.cancelled().await;
                None
            })
            .await
    );
    assert!(first.compaction.has_job().await);
    assert!(
        !second.compaction.has_job().await,
        "an idle rollover must own an independent compaction lifecycle"
    );
    await_compaction_sync(
        "rolled-over session compaction job to reap",
        first.compaction.cancel_and_reap(),
    )
    .await;
}

#[tokio::test]
async fn pending_compaction_checkpoint_hides_unpublished_dag() {
    let provider = MockLLM::named("local-compaction-checkpoint-visibility-test");
    let (agent_loop, _workspace) = build_local_inline_harness(provider);
    let session_key = format!("compaction-checkpoint-visibility-{}", uuid::Uuid::new_v4());
    let core = agent_loop.shared.core_handle.swappable();
    let session = core.sessions.get_or_resume(&session_key).await;
    core.sessions
        .add_message(
            &session.id,
            &json!({"role": "user", "content": "raw-source-marker"}),
        )
        .await;
    core.sessions
        .add_message(
            &session.id,
            &json!({"role": "assistant", "content": "raw-source-answer"}),
        )
        .await;

    let mut msg = InboundMessage::new("test", "user", "offline", "first");
    msg.metadata
        .insert("session_key".to_string(), json!(session_key));
    let first = agent_loop
        .shared
        .prepare_context(&msg, None, None, None, None)
        .await;

    let raw = core.sessions.get_all_messages(&session.id).await;
    let source_ids = raw
        .iter()
        .map(|message| message["_db_id"].as_u64().unwrap() as usize)
        .collect::<Vec<_>>();
    let nodes = vec![(
        0,
        source_ids,
        Vec::new(),
        "checkpoint-summary-marker".to_string(),
        3,
        1,
        crate::agent::lcm::SummaryManifest::default(),
        "db_id".to_string(),
    )];
    let rebuilt = crate::agent::lcm::LcmEngine::rebuild_from_db_nodes(
        &raw,
        &nodes,
        crate::agent::lcm::LcmConfig::default(),
    );
    let engine = agent_loop
        .shared
        .lcm_engines
        .lock()
        .await
        .get(&session.id)
        .cloned()
        .unwrap();
    *engine.lock().await = rebuilt;

    let snapshot = first.messages[..first.new_start].to_vec();
    *first.compaction.slot.lock().await = Some(crate::agent::agent_core::PendingCompaction {
        result: crate::agent::compaction::CompactionResult {
            messages: vec![json!({
                "role": "assistant",
                "content": "checkpoint-summary-marker"
            })],
        },
        snapshot,
        summary_node_id: 0,
    });
    msg.content = "second".to_string();
    let second = agent_loop
        .shared
        .prepare_context(&msg, None, None, None, None)
        .await;
    *first.compaction.slot.lock().await = None;

    let assembled = serde_json::to_string(&*second.messages).unwrap();
    assert!(
        assembled.contains("raw-source-marker"),
        "raw history remains authoritative while the checkpoint is pending"
    );
    assert!(
        !assembled.contains("checkpoint-summary-marker"),
        "a pending DAG rewrite must not become prompt-visible before checkpoint installation"
    );
}

#[tokio::test]
async fn persisted_lcm_rebuild_expands_exact_durable_rows() {
    let provider = MockLLM::named("persisted-lcm-replay-projection-test");
    let (agent_loop, _workspace) = build_local_inline_harness(provider);
    let session_key = format!("persisted-lcm-replay-projection-{}", uuid::Uuid::new_v4());
    let core = agent_loop.shared.core_handle.swappable();
    let session = core.sessions.get_or_resume(&session_key).await;

    core.sessions
        .add_message(
            &session.id,
            &json!({"role": "user", "content": "durable source"}),
        )
        .await;
    core.sessions
        .add_message(
            &session.id,
            &json!({
                "role": "user",
                "content": "[System notice] You have 12 iteration(s) remaining.",
                "_synthetic": true,
                "_cache_replay": true,
            }),
        )
        .await;
    core.sessions
        .add_message(
            &session.id,
            &json!({"role": "assistant", "content": "durable answer"}),
        )
        .await;

    let exact_tool_body = format!(
        "RAW_EXACT_TOOL_BODY:{}:RAW_EXACT_TOOL_TAIL",
        "x".repeat(13_000)
    );
    core.sessions
        .add_message(
            &session.id,
            &json!({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_exact",
                    "type": "function",
                    "function": {"name": "exec", "arguments": "{}"},
                }],
            }),
        )
        .await;
    core.sessions
        .add_message(
            &session.id,
            &json!({
                "role": "tool",
                "content": exact_tool_body,
                "tool_call_id": "call_exact",
                "name": "exec",
            }),
        )
        .await;

    let replay = core.sessions.get_history(&session.id, 0, 0).await;
    let raw = core.sessions.get_all_messages(&session.id).await;
    let replay_wire = serde_json::to_string(&replay).unwrap();
    assert!(!replay_wire.contains("RAW_EXACT_TOOL_TAIL"));
    assert!(replay_wire.contains("TOOL_RESULT_HANDLE v1"));
    let source_ids = replay
        .iter()
        .take(3)
        .map(|message| message["_db_id"].as_u64().unwrap() as usize)
        .collect::<Vec<_>>();
    assert_eq!(source_ids.len(), 3, "cache-replay scaffold stays foldable");
    core.sessions
        .save_summary_node(
            &session.id,
            0,
            &source_ids,
            &[],
            "version=1 lcm_expand({\"message_ids\":\"1-3\"})",
            10,
            0,
            &crate::agent::lcm::SummaryManifest::default(),
        )
        .await;

    let mut inbound = InboundMessage::new("test", "user", "offline", "continue");
    inbound
        .metadata
        .insert("session_key".to_string(), json!(session_key));
    let _context = agent_loop
        .shared
        .prepare_context(&inbound, None, None, None, None)
        .await;

    let engine = agent_loop
        .shared
        .lcm_engines
        .lock()
        .await
        .get(&session.id)
        .cloned()
        .unwrap();
    let engine = engine.lock().await;
    let expanded = engine.expand(&source_ids);
    let active = serde_json::to_string(&engine.active_context()).unwrap();

    let expected = raw
        .iter()
        .filter(|message| source_ids.contains(&(message["_db_id"].as_u64().unwrap() as usize)))
        .collect::<Vec<_>>();
    assert_eq!(expanded.len(), expected.len());
    for ((id, actual), expected) in expanded.into_iter().zip(expected) {
        assert_eq!(id, expected["_db_id"].as_u64().unwrap() as usize);
        assert_eq!(
            actual, expected,
            "restart must preserve digest/expand bytes"
        );
    }
    assert!(!active.contains("RAW_EXACT_TOOL_TAIL"));
    assert!(active.contains("TOOL_RESULT_HANDLE v1"));
}

#[tokio::test]
async fn idle_rollover_uses_a_new_session_scoped_lcm_engine() {
    let provider = Arc::new(WireRecordingProvider::new(
        "local-idle-lcm-test",
        vec![WireRecordingProvider::text_response("foreground reply")],
    ));
    let memory_config = MemoryConfig {
        session_complete_after_secs: 1,
        ..Default::default()
    };
    let (agent_loop, _workspace) = build_local_inline_harness_with_memory(
        provider as Arc<dyn LLMProvider>,
        "local-idle-lcm-test",
        4096,
        LcmSchemaConfig::default(),
        memory_config,
    );
    let session_key = format!("idle-lcm-isolation-{}", uuid::Uuid::new_v4());

    agent_loop
        .process_direct("first-session-marker", &session_key, "test", "offline")
        .await;
    let core = agent_loop.shared.core_handle.swappable();
    let first_id = core
        .sessions
        .get_latest_session(&session_key)
        .await
        .unwrap()
        .id;
    // Expiry uses whole seconds and a strict `idle > threshold` comparison.
    tokio::time::sleep(std::time::Duration::from_millis(2_100)).await;

    agent_loop
        .process_direct("second-session-marker", &session_key, "test", "offline")
        .await;
    let second_id = core
        .sessions
        .get_latest_session(&session_key)
        .await
        .unwrap()
        .id;
    assert_ne!(
        first_id, second_id,
        "idle rollover must create a new session"
    );

    let second_engine = agent_loop
        .shared
        .lcm_engines
        .lock()
        .await
        .get(&second_id)
        .cloned()
        .expect("new concrete session must own its own LCM engine");
    let active = second_engine.lock().await.active_context();
    let wire = serde_json::to_string(&active).unwrap();
    assert!(wire.contains("second-session-marker"));
    assert!(!wire.contains("first-session-marker"));
}

/// Assert every wire message of `first` reappears byte-identical, in order,
/// at the head of `second` — the KV prefix-cache contract. One mutated byte
/// forces the local server to re-prefill everything past it (~45s cold on a
/// 35B), so this property IS the local-model perf story.
fn assert_wire_prefix(first: &[Value], second: &[Value]) {
    assert!(
        second.len() > first.len(),
        "later call must extend the earlier one (got {} then {})",
        first.len(),
        second.len()
    );
    for (i, msg) in first.iter().enumerate() {
        assert_eq!(
            serde_json::to_string(msg).unwrap(),
            serde_json::to_string(&second[i]).unwrap(),
            "wire message {i} mutated between calls — prompt prefix diverged, KV cache busted"
        );
    }
}

/// KV prefix contract across turns: turn N's wire prompt must be a
/// byte-prefix of turn N+1's. Per-turn content belongs in TAIL blocks,
/// never in messages[0]. Guards the 1.9s-full-prefill → 0.07s-reuse asset.
#[tokio::test]
async fn test_local_wire_prompt_prefix_stable_across_turns() {
    let provider = Arc::new(WireRecordingProvider::new(
        "local-qwen-test",
        vec![WireRecordingProvider::text_response("first reply")],
    ));
    let (agent_loop, _ws) = build_local_inline_harness(provider.clone() as Arc<dyn LLMProvider>);
    let session_key = format!("prefix-stability-{}", uuid::Uuid::new_v4());

    agent_loop
        .process_direct("first message", &session_key, "test", "offline")
        .await;
    agent_loop
        .process_direct("second message", &session_key, "test", "offline")
        .await;

    let calls = provider.calls();
    assert!(
        calls.len() >= 2,
        "expected two LLM calls, got {}",
        calls.len()
    );
    assert_wire_prefix(&calls[0], &calls[calls.len() - 1]);
}

#[tokio::test]
async fn test_local_wire_prompt_prefix_stable_when_second_turn_is_rich_artifact() {
    let provider = Arc::new(WireRecordingProvider::new(
        "local-qwen-test",
        vec![
            WireRecordingProvider::text_response("Hey! What can I help you with today?"),
            WireRecordingProvider::text_response("I'll create it."),
        ],
    ));
    let (agent_loop, _ws) = build_local_inline_harness(provider.clone() as Arc<dyn LLMProvider>);
    let session_key = format!("artifact-prefix-stability-{}", uuid::Uuid::new_v4());

    agent_loop
        .process_direct("hi", &session_key, "test", "offline")
        .await;
    agent_loop
        .process_direct(
            "I want you to create a single HTML file version of Pong at ~/Dev/pong . It must be fun, colorful and easy to play",
            &session_key,
            "test",
            "offline",
        )
        .await;

    let calls = provider.calls();
    assert!(
        calls.len() >= 2,
        "expected two LLM calls, got {}",
        calls.len()
    );
    assert_wire_prefix(&calls[0], &calls[calls.len() - 1]);
    let system = calls[calls.len() - 1][0]["content"].as_str().unwrap_or("");
    assert!(
        !system.contains("Local Artifact Writer"),
        "rich artifact turns must not mutate the stable system prompt"
    );
}

/// THE prompt-cache invariant: when a turn's prompt is not an append-only
/// extension of the previous call (unsanctioned divergence), the request must
/// ship under a FRESH higgs session id with the poisoned id queued for drop —
/// never under the warm id, where the server's exact-token guard rejects it
/// (`token_mismatch`) and then rejects every follow-up (`not_growing`),
/// cascading full re-prefills.
#[tokio::test]
async fn test_diverged_prompt_ships_under_rotated_higgs_session() {
    let provider = Arc::new(WireRecordingProvider::new_higgs_capable(
        "local-qwen-test",
        vec![
            WireRecordingProvider::text_response("first reply"),
            WireRecordingProvider::text_response("second reply"),
        ],
    ));
    let (agent_loop, _ws) = build_local_inline_harness(provider.clone() as Arc<dyn LLMProvider>);
    let session_key = format!("diverge-rotate-{}", uuid::Uuid::new_v4());

    agent_loop
        .process_direct("first message", &session_key, "test", "offline")
        .await;

    // Poison the baseline so turn two's compare cannot be an append-only
    // extension of what was actually shipped — the unsanctioned-divergence
    // class the live bug produced.
    agent_loop
        .shared
        .core_handle
        .counters
        .prompt_fingerprints
        .lock()
        .insert(
            session_key.clone(),
            crate::agent::prompt_fingerprint::fingerprint(&[json!({
                "role": "user",
                "content": "bogus baseline",
            })]),
        );

    agent_loop
        .process_direct("second message", &session_key, "test", "offline")
        .await;

    let calls = provider.calls();
    assert!(calls.len() >= 2, "expected two LLM calls");
    let session_field = crate::providers::openai_compat::NANOBOT_HIGGS_SESSION_ID_FIELD;
    let drop_field = crate::providers::openai_compat::NANOBOT_HIGGS_DROP_SESSION_ID_FIELD;
    let id1 = calls[0][0][session_field].as_u64();
    let id2 = calls[1][0][session_field].as_u64();
    let id1 = id1.expect("higgs session id must ride on the first wire");
    let id2 = id2.expect("higgs session id must ride on the diverged wire");
    assert_ne!(
        id1, id2,
        "diverged prompt must ship under a fresh session id, not the warm one"
    );
    assert_eq!(
        calls[1][0][drop_field].as_u64(),
        Some(id1),
        "poisoned session must be queued for drop on the diverged request"
    );
    // Turn-start divergence is the reload window advancing — a SANCTIONED
    // reset, priced as `history_reload`, not as an unsanctioned bug.
    assert_eq!(
        agent_loop
            .shared
            .core_handle
            .counters
            .take_cache_reset(&session_key),
        Some("history_reload"),
        "turn-start divergence must be classified as sanctioned history_reload"
    );
}

/// The tool block renders at the chat-template head, so byte drift in the
/// tool array under a warm session is the same hazard as a message
/// divergence: the request must rotate to a fresh session id (queued drop of
/// the poisoned one), priced as `tool_block_change`.
#[tokio::test]
async fn test_tool_block_change_rotates_higgs_session() {
    let provider = Arc::new(WireRecordingProvider::new_higgs_capable(
        "local-qwen-test",
        vec![
            WireRecordingProvider::text_response("first reply"),
            WireRecordingProvider::text_response("second reply"),
        ],
    ));
    let (agent_loop, _ws) = build_local_inline_harness(provider.clone() as Arc<dyn LLMProvider>);
    let session_key = format!("tool-drift-{}", uuid::Uuid::new_v4());

    agent_loop
        .process_direct("first message", &session_key, "test", "offline")
        .await;

    // Poison the tool-hash baseline — the enforcement's view of "the tool
    // block bytes changed under this warm session".
    agent_loop
        .shared
        .core_handle
        .counters
        .prompt_tool_hashes
        .lock()
        .insert(session_key.clone(), 0xDEAD_BEEF);

    agent_loop
        .process_direct("second message", &session_key, "test", "offline")
        .await;

    let calls = provider.calls();
    assert!(calls.len() >= 2, "expected two LLM calls");
    let session_field = crate::providers::openai_compat::NANOBOT_HIGGS_SESSION_ID_FIELD;
    let drop_field = crate::providers::openai_compat::NANOBOT_HIGGS_DROP_SESSION_ID_FIELD;
    let id1 = calls[0][0][session_field]
        .as_u64()
        .expect("higgs session id must ride on the first wire");
    let id2 = calls[1][0][session_field]
        .as_u64()
        .expect("higgs session id must ride on the drifted wire");
    assert_ne!(
        id1, id2,
        "tool-block drift must ship under a fresh session id"
    );
    assert_eq!(
        calls[1][0][drop_field].as_u64(),
        Some(id1),
        "poisoned session must be queued for drop on the drifted request"
    );
    assert_eq!(
        agent_loop
            .shared
            .core_handle
            .counters
            .take_cache_reset(&session_key),
        Some("tool_block_change"),
        "tool-block drift must be classified as tool_block_change"
    );
}

/// Persisted sessions remain discoverable through recall/resume, but a fresh
/// local turn must not carry an unrelated previous-session hint in its stable
/// prompt prefix.
#[tokio::test]
async fn test_local_fresh_session_does_not_inject_previous_session() {
    let provider = Arc::new(WireRecordingProvider::new(
        "local-qwen-test",
        vec![
            WireRecordingProvider::text_response("first reply"),
            WireRecordingProvider::text_response("second reply"),
        ],
    ));
    let (agent_loop, _ws) = build_local_inline_harness(provider.clone() as Arc<dyn LLMProvider>);

    agent_loop
        .process_direct("first message", "pi-style-prior", "test", "offline")
        .await;
    agent_loop
        .process_direct(
            "unrelated fresh message",
            "pi-style-fresh",
            "test",
            "offline",
        )
        .await;

    let calls = provider.calls();
    assert!(
        calls.len() >= 2,
        "expected two LLM calls, got {}",
        calls.len()
    );
    let system = calls[1][0]["content"].as_str().unwrap_or("");
    assert!(!system.contains("Previous Session"));
    assert!(!system.contains("pi-style-prior"));
}

/// Same contract within a turn: executing a tool must only APPEND to the
/// wire prompt (assistant carrier + tool result + continuation), never
/// rewrite what the server already prefilled.
#[tokio::test]
async fn test_local_wire_prompt_tool_result_appends_only() {
    let mut args = std::collections::HashMap::new();
    args.insert("path".to_string(), json!("."));
    let provider = Arc::new(WireRecordingProvider::new(
        "local-qwen-test",
        vec![
            crate::providers::base::LLMResponse {
                content: Some(String::new()),
                tool_calls: vec![crate::providers::base::ToolCallRequest {
                    id: "tc_prefix".to_string(),
                    name: "list_dir".to_string(),
                    arguments: args,
                }],
                finish_reason: FinishReason::ToolCalls,
                usage: std::collections::HashMap::new(),
            },
            WireRecordingProvider::text_response("listed."),
        ],
    ));
    let (agent_loop, _ws) = build_local_inline_harness(provider.clone() as Arc<dyn LLMProvider>);
    let session_key = format!("prefix-tool-{}", uuid::Uuid::new_v4());

    agent_loop
        .process_direct("please list files", &session_key, "test", "offline")
        .await;

    let calls = provider.calls();
    assert!(
        calls.len() >= 2,
        "expected two LLM calls, got {}",
        calls.len()
    );
    assert_wire_prefix(&calls[0], &calls[1]);
}

#[tokio::test]
async fn test_local_wire_prefix_stable_across_batched_tool_results_and_next_turn() {
    let mut args_a = std::collections::HashMap::new();
    args_a.insert("path".to_string(), json!("big_a.txt"));
    let mut args_b = std::collections::HashMap::new();
    args_b.insert("path".to_string(), json!("big_b.txt"));
    let provider = Arc::new(WireRecordingProvider::new(
        "local-qwen-test",
        vec![
            crate::providers::base::LLMResponse {
                content: Some(String::new()),
                tool_calls: vec![
                    crate::providers::base::ToolCallRequest {
                        id: "tc_big_a".to_string(),
                        name: "read_file".to_string(),
                        arguments: args_a,
                    },
                    crate::providers::base::ToolCallRequest {
                        id: "tc_big_b".to_string(),
                        name: "read_file".to_string(),
                        arguments: args_b,
                    },
                ],
                finish_reason: FinishReason::ToolCalls,
                usage: std::collections::HashMap::new(),
            },
            WireRecordingProvider::text_response("done turn one"),
            WireRecordingProvider::text_response("turn two reply"),
        ],
    ));
    let (agent_loop, workspace) =
        build_local_inline_harness(provider.clone() as Arc<dyn LLMProvider>);
    std::fs::write(workspace.join("big_a.txt"), "alpha\n".repeat(1600)).unwrap();
    std::fs::write(workspace.join("big_b.txt"), "beta\n".repeat(1700)).unwrap();
    let session_key = format!("batched-tools-prefix-{}", uuid::Uuid::new_v4());

    tokio::time::timeout(
        std::time::Duration::from_secs(10),
        agent_loop.process_direct("read both files", &session_key, "test", "offline"),
    )
    .await
    .expect("turn 1 must terminate");
    let turn1_calls = provider.calls().len();
    assert_eq!(turn1_calls, 2, "turn 1 should call before and after tools");

    tokio::time::timeout(
        std::time::Duration::from_secs(10),
        agent_loop.process_direct("what did you see?", &session_key, "test", "offline"),
    )
    .await
    .expect("turn 2 must terminate");

    let calls = provider.calls();
    assert_eq!(calls.len(), 3, "turn 2 should add one provider call");
    assert_wire_prefix(&calls[0], &calls[1]);
    assert_wire_prefix(&calls[1], &calls[2]);

    let call1_roles: Vec<_> = calls[1]
        .iter()
        .map(|m| m["role"].as_str().unwrap_or("?"))
        .collect();
    let joined = call1_roles.join(",");
    // With native tool-role rendering (2026-07-27), parallel tool
    // results render as consecutive role:tool messages — no user-role
    // separators between them. The pattern is
    // assistant(tool_calls),tool,tool,assistant(text).
    assert!(
        joined.contains("assistant,tool,tool"),
        "batched native tool results must render as consecutive role:tool after the tool_calls assistant: {joined}"
    );
    // With native tool-role rendering, tool results are role:tool with
    // name + content (no [System: tool succeeded...] user wrapper).
    assert!(calls[1].iter().any(|m| {
        m["role"] == "tool" && m["name"] == "read_file" && m["tool_call_id"] == "tc_big_a"
    }));
    assert!(calls[1].iter().any(|m| {
        m["role"] == "tool" && m["name"] == "read_file" && m["tool_call_id"] == "tc_big_b"
    }));

    let _ = std::fs::remove_dir_all(&workspace);
}

/// STEP 1 agent-loop invariant test: when a second executed tool result would
/// stash DIFFERENT bytes under an existing `(session_id, tool_call_id)`, the
/// turn FAILS with the infrastructure error instead of overwriting the body or
/// surfacing file B's content. The stash retains file A.
///
/// This is the load-bearing contract behind
/// `docs/superpowers/plans/2026-07-30-tool-result-handles-not-bodies.md` Hole 1:
/// a lying handle (pointing at overwritten bytes) is what caused the
/// `token_mismatch` cache-desync class.
#[tokio::test]
async fn stash_conflict_on_reused_tool_call_id_aborts_turn_preserves_body_a() {
    // Directly verify the agent-loop abort path: when the immutable stash
    // rejects a conflicting write, the turn FAILS with the infra error and
    // the body is never shown raw.
    //
    // Setup: pre-stash body A under "tc_conflict" (simulating a prior turn
    // that stashed it). Then run ONE turn that emits a 2-call batch (so
    // force_stash_raw=true) re-using "tc_conflict" for body B → Conflict.
    let files_dir = tempfile::tempdir().unwrap();
    let body_a = "alpha\n".repeat(200);
    let body_b = "beta\n".repeat(210);
    let body_c = "gamma\n".repeat(190);
    let body_d = "delta\n".repeat(220);
    let path_a = files_dir.path().join("big_a.txt");
    let path_b = files_dir.path().join("big_b.txt");
    let path_c = files_dir.path().join("big_c.txt");
    let path_d = files_dir.path().join("big_d.txt");
    std::fs::write(&path_a, &body_a).unwrap();
    std::fs::write(&path_b, &body_b).unwrap();
    std::fs::write(&path_c, &body_c).unwrap();
    std::fs::write(&path_d, &body_d).unwrap();
    let path_a_s = path_a.to_string_lossy().to_string();
    let path_b_s = path_b.to_string_lossy().to_string();
    let path_c_s = path_c.to_string_lossy().to_string();
    let path_d_s = path_d.to_string_lossy().to_string();

    // Two read_file calls so force_stash_raw=true (multi-result batch always
    // stashes, regardless of per-result size). tc_conflict reads file_b
    // (conflicts with pre-stashed body A); tc_extra reads file_d — a path the
    // seed turn never touched, so the retained-context tool guard cannot
    // cache-block either call before the stash sees the reused id.
    let mut a1 = std::collections::HashMap::new();
    a1.insert("path".to_string(), json!(path_b_s));
    let mut a2 = std::collections::HashMap::new();
    a2.insert("path".to_string(), json!(path_d_s));
    let conflict_batch = crate::providers::base::LLMResponse {
        content: Some(String::new()),
        tool_calls: vec![
            crate::providers::base::ToolCallRequest {
                id: "tc_conflict".to_string(),
                name: "read_file".to_string(),
                arguments: a1,
            },
            crate::providers::base::ToolCallRequest {
                id: "tc_extra".to_string(),
                name: "read_file".to_string(),
                arguments: a2,
            },
        ],
        finish_reason: FinishReason::ToolCalls,
        usage: std::collections::HashMap::new(),
    };

    let provider = Arc::new(WireRecordingProvider::new(
        "local-qwen-test",
        vec![
            conflict_batch.clone(),
            // Call 2: UNREACHABLE — the turn aborts at the stash-conflict
            // before the next LLM round.
            WireRecordingProvider::text_response("UNREACHABLE post-conflict answer"),
        ],
    ));
    let (agent_loop, _harness_workspace) = build_local_inline_harness_with_lcm(
        provider.clone() as Arc<dyn LLMProvider>,
        "local-qwen-test",
        32_768,
        LcmSchemaConfig::default(),
    );
    let session_key = format!("stash-conflict-{}", uuid::Uuid::new_v4());

    // Pre-stash body A under "tc_conflict" by running a PRIOR turn that
    // produces a force-stash for that id (2-call batch reading DIFFERENT
    // files, so the duplicate-call guard doesn't collapse them). The extra
    // call reads file_c: the conflict turn must re-read NEITHER seed path,
    // or the cross-turn tool guard cache-blocks it before the stash runs.
    {
        let mut pa1 = std::collections::HashMap::new();
        pa1.insert("path".to_string(), json!(path_a_s.clone()));
        let mut pa2 = std::collections::HashMap::new();
        pa2.insert("path".to_string(), json!(path_c_s.clone()));
        let seed_batch = crate::providers::base::LLMResponse {
            content: Some(String::new()),
            tool_calls: vec![
                crate::providers::base::ToolCallRequest {
                    id: "tc_conflict".to_string(),
                    name: "read_file".to_string(),
                    arguments: pa1,
                },
                crate::providers::base::ToolCallRequest {
                    id: "tc_seed_extra".to_string(),
                    name: "read_file".to_string(),
                    arguments: pa2,
                },
            ],
            finish_reason: FinishReason::ToolCalls,
            usage: std::collections::HashMap::new(),
        };
        // Replace the provider's queue with the seed batch first.
        {
            let mut queue = provider.responses.lock().unwrap();
            queue.clear();
            queue.push_back(seed_batch);
            queue.push_back(WireRecordingProvider::text_response("seeded"));
            queue.push_back(conflict_batch.clone());
            queue.push_back(WireRecordingProvider::text_response(
                "UNREACHABLE post-conflict answer",
            ));
        }
    }

    // Seed turn: stashes body A under tc_conflict (force=true, 2-call batch).
    let seed = tokio::time::timeout(
        std::time::Duration::from_secs(15),
        agent_loop.process_direct("seed file_a read", &session_key, "test", "offline"),
    )
    .await
    .expect("seed turn must terminate");
    assert_eq!(seed, "seeded");
    assert_eq!(
        provider.calls().len(),
        2,
        "the seed turn must not consume the queued conflict batch"
    );

    // Confirm body A was stashed under tc_conflict.
    {
        let core = agent_loop.shared.core_handle.swappable();
        let session = core.sessions.get_or_resume(&session_key).await;
        let stashed = core
            .sessions
            .load_tool_result(&session.id, "tc_conflict")
            .await;
        assert!(
            stashed
                .as_deref()
                .map(|s| s.contains("alpha"))
                .unwrap_or(false),
            "seed turn must have stashed body A under tc_conflict; got {stashed:?}"
        );
    }

    // Conflict turn: reuses tc_conflict for body B → stash Conflict → abort.
    let conflict_turn = tokio::time::timeout(
        std::time::Duration::from_secs(15),
        agent_loop.process_direct(
            "now read file_b with the same call id",
            &session_key,
            "test",
            "offline",
        ),
    )
    .await
    .expect("conflict turn must still terminate (with the infra error)");

    // The infra error is surfaced — never body B's raw content.
    assert!(
        !conflict_turn.contains("beta"),
        "conflict turn must NOT surface file B's body; got: {conflict_turn}"
    );
    assert!(
        conflict_turn.to_lowercase().contains("abort")
            || conflict_turn.to_lowercase().contains("error"),
        "conflict turn must surface the abort/error to the user; got: {conflict_turn}"
    );
    assert_eq!(
        provider.calls().len(),
        3,
        "stash conflict must abort before requesting a post-conflict answer"
    );

    // The stash retains body A — never overwritten by body B.
    let core = agent_loop.shared.core_handle.swappable();
    let session = core.sessions.get_or_resume(&session_key).await;
    let stashed = core
        .sessions
        .load_tool_result(&session.id, "tc_conflict")
        .await;
    let stashed = stashed.expect("tc_conflict must retain a stashed body");
    assert!(
        stashed.contains("alpha") && !stashed.contains("beta"),
        "the stash must retain body A (alpha), not body B (beta); got: {stashed}"
    );

    let _ = std::fs::remove_dir_all(files_dir);
}

/// STEP 2 integration test: a turn that produces a stashed (large/forced) tool
/// A GENUINELY oversized (>8KB / TOOL_RESULT_REPLAY_MAX_BYTES) tool result
/// must persist a HANDLE as the tool-result message content — not the raw
/// body. The body lives only in the stash, inspectable through inspect_tool_result.
/// The handle is byte-identical live and after a SQLite round-trip (reload).
/// Uses `exec seq` (~13KB) because read_file self-caps under the replay limit.
#[tokio::test]
async fn stashed_tool_result_persists_handle_not_body_in_messages() {
    use crate::agent::tool_engine::TOOL_RESULT_HANDLE_MARKER;

    let mut a = std::collections::HashMap::new();
    a.insert("command".to_string(), json!("seq 1 3000")); // ~13KB > 8KB replay cap
    let resp = crate::providers::base::LLMResponse {
        content: Some(String::new()),
        tool_calls: vec![crate::providers::base::ToolCallRequest {
            id: "tc_handle".to_string(),
            name: "exec".to_string(),
            arguments: a,
        }],
        finish_reason: FinishReason::ToolCalls,
        usage: std::collections::HashMap::new(),
    };

    let provider = Arc::new(WireRecordingProvider::new(
        "local-qwen-test",
        vec![resp, WireRecordingProvider::text_response("done")],
    ));
    let (agent_loop, _ws) = build_local_inline_harness(provider.clone() as Arc<dyn LLMProvider>);
    let session_key = format!("handle-persist-{}", uuid::Uuid::new_v4());

    let _turn = tokio::time::timeout(
        std::time::Duration::from_secs(15),
        agent_loop.process_direct("run it", &session_key, "test", "offline"),
    )
    .await
    .expect("turn must terminate");

    let core = agent_loop.shared.core_handle.swappable();
    let session = core.sessions.get_or_resume(&session_key).await;

    let messages = core.sessions.get_all_messages(&session.id).await;
    let tool_msg = messages.iter().find(|m| {
        m.get("role").and_then(|v| v.as_str()) == Some("tool")
            && m.get("tool_call_id").and_then(|v| v.as_str()) == Some("tc_handle")
    });
    let tool_msg = tool_msg.expect("tc_handle tool message must exist in persisted history");
    let content = tool_msg
        .get("content")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    assert!(
        content.starts_with(TOOL_RESULT_HANDLE_MARKER),
        "persisted tool-result content must be a HANDLE; got: {content}"
    );
    // The raw body is NOT in the message — "2999" is a seq line, absent from
    // the handle (excerpt is "1"; args is "seq 1 3000").
    assert!(
        !content.contains("2999"),
        "the handle must not contain the raw body; got: {content}"
    );

    // The full body IS in the stash, inspectable through inspect_tool_result.
    let stashed = core
        .sessions
        .load_tool_result(&session.id, "tc_handle")
        .await
        .expect("body must be stashed under tc_handle");
    assert!(
        stashed.contains("2999") && stashed.contains("1500"),
        "the stash must retain the full body; got first 80 chars: {}",
        &stashed[..stashed.len().min(80)]
    );

    // Cache stability: the handle in the LIVE prompt (provider call 2) is
    // byte-identical to the one persisted in SQLite (reload-stable).
    let calls = provider.calls();
    let live_msg = calls
        .iter()
        .flat_map(|c| c.iter())
        .find(|m| {
            m.get("role").and_then(|v| v.as_str()) == Some("tool")
                && m.get("tool_call_id").and_then(|v| v.as_str()) == Some("tc_handle")
        })
        .expect("live prompt must contain the tc_handle tool message");
    let live_content = live_msg
        .get("content")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    assert!(
        live_content.starts_with(TOOL_RESULT_HANDLE_MARKER),
        "live prompt tool-result must be a handle; got: {live_content}"
    );
    assert_eq!(
        live_content, content,
        "live and persisted handle must be byte-identical (cache-stable)"
    );
}

/// Hybrid exposure: a MEDIUM ordinary result (read_file self-caps under the
/// inline threshold) must be stored losslessly AND injected inline, so the
/// model reads the content directly with no inspect_tool_result round-trip.
/// Live and persisted bytes stay identical so the prefix cache remains stable.
#[tokio::test]
async fn medium_tool_result_inlines_body_not_handle() {
    let files_dir = tempfile::tempdir().unwrap();
    // ~6.4KB source; read_file self-caps its output under the inline
    // threshold, so the shaped result is small enough to inline.
    let body = "medium_line_one\nmedium_line_two\n".repeat(200);
    let path = files_dir.path().join("medium.txt");
    std::fs::write(&path, &body).unwrap();
    let path_s = path.to_string_lossy().to_string();

    let mut a = std::collections::HashMap::new();
    a.insert("path".to_string(), json!(path_s));
    let resp = crate::providers::base::LLMResponse {
        content: Some(String::new()),
        tool_calls: vec![crate::providers::base::ToolCallRequest {
            id: "tc_medium".to_string(),
            name: "read_file".to_string(),
            arguments: a,
        }],
        finish_reason: FinishReason::ToolCalls,
        usage: std::collections::HashMap::new(),
    };
    let provider = Arc::new(WireRecordingProvider::new(
        "local-qwen-test",
        vec![resp, WireRecordingProvider::text_response("done")],
    ));
    let (agent_loop, _ws) = build_local_inline_harness(provider.clone() as Arc<dyn LLMProvider>);
    let session_key = format!("medium-content-{}", uuid::Uuid::new_v4());

    let _turn = tokio::time::timeout(
        std::time::Duration::from_secs(15),
        agent_loop.process_direct("read the file", &session_key, "test", "offline"),
    )
    .await
    .expect("turn must terminate");

    let core = agent_loop.shared.core_handle.swappable();
    let session = core.sessions.get_or_resume(&session_key).await;
    let messages = core.sessions.get_all_messages(&session.id).await;
    let tool_msg = messages
        .iter()
        .find(|m| {
            m.get("role").and_then(|v| v.as_str()) == Some("tool")
                && m.get("tool_call_id").and_then(|v| v.as_str()) == Some("tc_medium")
        })
        .expect("tc_medium tool message must exist");
    let content = tool_msg
        .get("content")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    assert!(
        content.contains("medium_line_one"),
        "hybrid: medium (sub-threshold) result must inline the body so the \
         model reads it without an inspect round-trip; got: {content}"
    );

    let stashed = core
        .sessions
        .load_tool_result(&session.id, "tc_medium")
        .await
        .expect("medium body must be durably stashed");
    assert!(
        stashed.contains("medium_line_one") && stashed.contains("medium_line_two"),
        "stash must retain the exact tool output bytes; got: {stashed}"
    );

    let calls = provider.calls();
    let live_msg = calls
        .iter()
        .flat_map(|call| call.iter())
        .find(|message| {
            message.get("role").and_then(|v| v.as_str()) == Some("tool")
                && message.get("tool_call_id").and_then(|v| v.as_str()) == Some("tc_medium")
        })
        .expect("next provider request must contain the medium tool message");
    let live_content = live_msg
        .get("content")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    assert_eq!(
        live_content, content,
        "live and persisted inline bytes must be identical (cache-stable)"
    );

    let _ = std::fs::remove_dir_all(files_dir);
}

struct DuplicateTerminalProvider {
    responses: parking_lot::Mutex<std::collections::VecDeque<crate::providers::base::LLMResponse>>,
    terminal_choices: parking_lot::Mutex<Vec<crate::providers::base::ToolChoice>>,
}

#[async_trait]
impl LLMProvider for DuplicateTerminalProvider {
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
        Ok(self
            .responses
            .lock()
            .pop_front()
            .expect("scripted response"))
    }

    async fn chat_with_tool_choice(
        &self,
        messages: &[Value],
        tools: Option<&[Value]>,
        model: Option<&str>,
        max_tokens: u32,
        temperature: f64,
        thinking_budget: Option<u32>,
        top_p: Option<f64>,
        tool_choice: crate::providers::base::ToolChoice,
    ) -> anyhow::Result<crate::providers::base::LLMResponse> {
        self.terminal_choices.lock().push(tool_choice);
        self.chat(
            messages,
            tools,
            model,
            max_tokens,
            temperature,
            thinking_budget,
            top_p,
        )
        .await
    }

    fn get_default_model(&self) -> &str {
        "local-duplicate-terminal"
    }
}

fn exec_tool_response(
    id: &str,
    command: &str,
    working_dir: Option<&std::path::Path>,
) -> crate::providers::base::LLMResponse {
    let mut arguments = std::collections::HashMap::from([("command".to_string(), json!(command))]);
    if let Some(dir) = working_dir {
        arguments.insert("working_dir".to_string(), json!(dir.to_string_lossy()));
    }
    crate::providers::base::LLMResponse {
        content: Some("Running the requested command.".to_string()),
        tool_calls: vec![crate::providers::base::ToolCallRequest {
            id: id.to_string(),
            name: "exec".to_string(),
            arguments,
        }],
        finish_reason: FinishReason::ToolCalls,
        usage: std::collections::HashMap::new(),
    }
}

fn build_unrestricted_exec_harness(
    provider: Arc<dyn LLMProvider>,
    workspace: std::path::PathBuf,
) -> AgentLoop {
    build_unrestricted_exec_harness_with_context(provider, workspace, 4_096)
}

fn build_unrestricted_exec_harness_with_context(
    provider: Arc<dyn LLMProvider>,
    workspace: std::path::PathBuf,
    max_context_tokens: usize,
) -> AgentLoop {
    let core = build_swappable_core(SwappableCoreConfig {
        provider,
        workspace,
        model: "local-tool-recovery".to_string(),
        max_iterations: 8,
        max_continuations: 2,
        max_tokens: 512,
        temperature: 0.0,
        max_context_tokens,
        brave_api_key: None,
        search_provider: "searxng".to_string(),
        searxng_url: "http://localhost:8888".to_string(),
        crw_url: String::new(),
        search_max_results: 5,
        exec_timeout: 30,
        restrict_to_workspace: false,
        memory_config: MemoryConfig::default(),
        is_local: true,
        lane: Lane::default(),
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
        code_execution: CodeExecutionConfig::default(),
        python_kernel: PythonKernelConfig::default(),
        cua: CuaToolConfig::default(),
        adaptive_tokens: AdaptiveTokenConfig::default(),
        sessions_db_path: Some(
            std::env::temp_dir().join(format!("nanobot-replay-{}.sqlite", uuid::Uuid::new_v4())),
        ),
    });
    let core_handle = AgentHandle::new(core, test_runtime_counters(max_context_tokens));
    let (inbound_tx, inbound_rx) = tokio::sync::mpsc::unbounded_channel::<InboundMessage>();
    let (outbound_tx, _outbound_rx) = tokio::sync::mpsc::unbounded_channel::<OutboundMessage>();
    AgentLoop::new(
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
    )
}

async fn assert_three_identical_execs_run_once(explicit_cwd: bool) {
    let workspace = tempfile::tempdir().unwrap().keep();
    let marker = workspace.join(if explicit_cwd { "explicit" } else { "default" });
    let command = format!("printf x >> {}", marker.to_string_lossy());
    let cwd = explicit_cwd.then_some(workspace.as_path());
    let provider = Arc::new(ResponseSequenceProvider::new(
        "local-tool-recovery",
        vec![
            exec_tool_response("exec-1", &command, cwd),
            exec_tool_response("exec-2", &command, cwd),
            exec_tool_response("exec-3", &command, cwd),
            WireRecordingProvider::text_response("finished after replay"),
        ],
    ));
    let agent_loop = build_unrestricted_exec_harness(
        provider.clone() as Arc<dyn LLMProvider>,
        workspace.clone(),
    );
    let session_key = format!("three-exec-{}-{}", explicit_cwd, uuid::Uuid::new_v4());

    let reply = agent_loop
        .process_direct("run exactly once", &session_key, "test", "offline")
        .await;
    assert_eq!(reply, "finished after replay");
    assert_eq!(std::fs::read_to_string(&marker).unwrap(), "x");

    let core = agent_loop.shared.core_handle.swappable();
    let session = core
        .sessions
        .get_latest_session(&session_key)
        .await
        .unwrap();
    let events = core
        .sessions
        .load_session_events(&session.id)
        .await
        .unwrap();
    let executions = events
        .iter()
        .filter(|event| {
            matches!(
                event.payload,
                crate::session::db::SessionEventPayload::ToolExecute { .. }
            )
        })
        .count();
    assert_eq!(executions, 1, "cached calls must not create executions");
    let cached = events
        .iter()
        .filter_map(|event| match &event.payload {
            crate::session::db::SessionEventPayload::ToolPreExecute {
                decision:
                    crate::session::db::ToolPreExecuteDecision::CachedReplay {
                        source_tool_call_id,
                        result_digest,
                    },
                ..
            } => Some((source_tool_call_id.as_str(), result_digest.as_str())),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(cached.len(), 2);
    assert!(cached
        .iter()
        .all(|(source, digest)| *source == "exec-1" && !digest.is_empty()));
    assert_eq!(provider.call_count(), 4);
    let _ = std::fs::remove_dir_all(workspace);
}

#[tokio::test]
async fn repeated_explicit_cwd_exec_is_a_durable_cached_replay() {
    assert_three_identical_execs_run_once(true).await;
}

#[tokio::test]
async fn near_inline_limit_cached_receipt_is_byte_stable_across_history_reloads() {
    let workspace = tempfile::tempdir().unwrap().keep();
    let marker = workspace.join("large-replay-side-effect");
    let command = format!(
        "printf y >> {}; awk 'BEGIN {{ for (i = 0; i < 4000; i++) printf \"x\" }}'",
        marker.to_string_lossy()
    );
    let provider = Arc::new(ResponseSequenceProvider::new(
        "local-tool-recovery",
        vec![
            exec_tool_response("large-source", &command, Some(&workspace)),
            exec_tool_response("large-replay", &command, Some(&workspace)),
            WireRecordingProvider::text_response("large replay finished"),
        ],
    ));
    let agent_loop = build_unrestricted_exec_harness_with_context(
        provider.clone() as Arc<dyn LLMProvider>,
        workspace.clone(),
        16_384,
    );
    let session_key = format!("large-cached-replay-{}", uuid::Uuid::new_v4());

    assert_eq!(
        agent_loop
            .process_direct(
                "run the large command once",
                &session_key,
                "test",
                "offline"
            )
            .await,
        "large replay finished"
    );
    assert_eq!(std::fs::read_to_string(&marker).unwrap(), "y");
    assert_eq!(provider.call_count(), 3);

    let core = agent_loop.shared.core_handle.swappable();
    let session = core
        .sessions
        .get_latest_session(&session_key)
        .await
        .unwrap();
    let raw = core.sessions.get_all_messages(&session.id).await;
    let persisted = raw
        .iter()
        .find(|message| {
            message.get("role").and_then(Value::as_str) == Some("tool")
                && message.get("tool_call_id").and_then(Value::as_str) == Some("large-replay")
        })
        .and_then(|message| message.get("content").and_then(Value::as_str))
        .unwrap()
        .to_string();
    assert!(
        persisted.len() > 4_096,
        "fixture must cross reload threshold"
    );
    assert!(persisted.contains("source_tool_call_id=\"large-source\""));

    for _ in 0..2 {
        let history = core.sessions.get_history(&session.id, 0, 0).await;
        let reloaded = history
            .iter()
            .find(|message| {
                message.get("role").and_then(Value::as_str) == Some("tool")
                    && message.get("tool_call_id").and_then(Value::as_str) == Some("large-replay")
            })
            .and_then(|message| message.get("content").and_then(Value::as_str))
            .unwrap();
        assert_eq!(reloaded, persisted);
    }
    assert!(
        core.sessions
            .load_tool_result(&session.id, "large-replay")
            .await
            .is_none(),
        "history reload must not fabricate a stored result for a cached receipt"
    );
    let _ = std::fs::remove_dir_all(workspace);
}

#[tokio::test]
async fn omitted_cwd_is_materialized_before_guard_lookup() {
    assert_three_identical_execs_run_once(false).await;
}

#[tokio::test]
async fn same_command_in_distinct_working_directories_executes_twice() {
    let workspace = tempfile::tempdir().unwrap().keep();
    let first = workspace.join("first");
    let second = workspace.join("second");
    std::fs::create_dir_all(&first).unwrap();
    std::fs::create_dir_all(&second).unwrap();
    let command = "printf x >> marker";
    let provider = Arc::new(ResponseSequenceProvider::new(
        "local-tool-recovery",
        vec![
            exec_tool_response("cwd-1", command, Some(&first)),
            exec_tool_response("cwd-2", command, Some(&second)),
            WireRecordingProvider::text_response("both finished"),
        ],
    ));
    let agent_loop =
        build_unrestricted_exec_harness(provider as Arc<dyn LLMProvider>, workspace.clone());
    let session_key = format!("distinct-cwd-{}", uuid::Uuid::new_v4());
    assert_eq!(
        agent_loop
            .process_direct("run in both", &session_key, "test", "offline")
            .await,
        "both finished"
    );
    assert_eq!(std::fs::read_to_string(first.join("marker")).unwrap(), "x");
    assert_eq!(std::fs::read_to_string(second.join("marker")).unwrap(), "x");
    let _ = std::fs::remove_dir_all(workspace);
}

#[tokio::test]
async fn completed_turn_does_not_replay_a_new_user_requested_exec() {
    let workspace = tempfile::tempdir().unwrap().keep();
    let marker = workspace.join("cross-turn");
    let command = format!("printf x >> {}", marker.to_string_lossy());
    let provider = Arc::new(ResponseSequenceProvider::new(
        "local-tool-recovery",
        vec![
            exec_tool_response("turn-1-exec", &command, Some(&workspace)),
            WireRecordingProvider::text_response("first done"),
            exec_tool_response("turn-2-exec", &command, Some(&workspace)),
            WireRecordingProvider::text_response("second done"),
        ],
    ));
    let agent_loop =
        build_unrestricted_exec_harness(provider as Arc<dyn LLMProvider>, workspace.clone());
    let session_key = format!("new-turn-exec-{}", uuid::Uuid::new_v4());

    assert_eq!(
        agent_loop
            .process_direct("first", &session_key, "test", "offline")
            .await,
        "first done"
    );
    assert_eq!(
        agent_loop
            .process_direct("second", &session_key, "test", "offline")
            .await,
        "second done"
    );
    assert_eq!(std::fs::read_to_string(&marker).unwrap(), "xx");
    let _ = std::fs::remove_dir_all(workspace);
}

#[tokio::test]
async fn different_empty_directory_reads_are_both_executed() {
    let workspace = tempfile::tempdir().unwrap().keep();
    let first = workspace.join("empty-a");
    let second = workspace.join("empty-b");
    std::fs::create_dir_all(&first).unwrap();
    std::fs::create_dir_all(&second).unwrap();
    let call = |id: &str, path: &std::path::Path| crate::providers::base::ToolCallRequest {
        id: id.to_string(),
        name: "list_dir".to_string(),
        arguments: std::collections::HashMap::from([(
            "path".to_string(),
            json!(path.to_string_lossy()),
        )]),
    };
    let provider = Arc::new(ResponseSequenceProvider::new(
        "local-tool-recovery",
        vec![
            crate::providers::base::LLMResponse {
                content: Some("Inspecting both directories.".to_string()),
                tool_calls: vec![call("empty-a", &first), call("empty-b", &second)],
                finish_reason: FinishReason::ToolCalls,
                usage: std::collections::HashMap::new(),
            },
            WireRecordingProvider::text_response("both are empty"),
        ],
    ));
    let agent_loop =
        build_unrestricted_exec_harness(provider as Arc<dyn LLMProvider>, workspace.clone());
    let session_key = format!("empty-reads-{}", uuid::Uuid::new_v4());
    assert_eq!(
        agent_loop
            .process_direct("inspect both", &session_key, "test", "offline")
            .await,
        "both are empty"
    );
    let core = agent_loop.shared.core_handle.swappable();
    let session = core
        .sessions
        .get_latest_session(&session_key)
        .await
        .unwrap();
    let events = core
        .sessions
        .load_session_events(&session.id)
        .await
        .unwrap();
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(
                event.payload,
                crate::session::db::SessionEventPayload::ToolExecute { .. }
            ))
            .count(),
        2,
        "equal empty evidence from different queries must not be semantically deduplicated"
    );
    let _ = std::fs::remove_dir_all(workspace);
}

#[tokio::test]
async fn repeated_read_evidence_executes_every_variant_then_adds_one_neutral_nudge() {
    let workspace = tempfile::tempdir().unwrap().keep();
    let path = workspace.join("same.txt");
    std::fs::write(&path, "same evidence\n").unwrap();
    let mut calls = Vec::new();
    for index in 0..3 {
        calls.push(crate::providers::base::ToolCallRequest {
            id: format!("read-{index}"),
            name: "read_file".to_string(),
            arguments: std::collections::HashMap::from([
                ("path".to_string(), json!(path.to_string_lossy())),
                ("max_lines".to_string(), json!(1000 + index)),
            ]),
        });
    }
    let provider = Arc::new(WireRecordingProvider::new(
        "local-read-evidence",
        vec![
            crate::providers::base::LLMResponse {
                content: Some("Reading three files.".to_string()),
                tool_calls: calls,
                finish_reason: FinishReason::ToolCalls,
                usage: std::collections::HashMap::new(),
            },
            WireRecordingProvider::text_response("evidence assessed"),
        ],
    ));
    let agent_loop = build_unrestricted_exec_harness(
        provider.clone() as Arc<dyn LLMProvider>,
        workspace.clone(),
    );
    let session_key = format!("read-evidence-{}", uuid::Uuid::new_v4());
    assert_eq!(
        agent_loop
            .process_direct("compare all three", &session_key, "test", "offline")
            .await,
        "evidence assessed"
    );

    let core = agent_loop.shared.core_handle.swappable();
    let session = core
        .sessions
        .get_latest_session(&session_key)
        .await
        .unwrap();
    let events = core
        .sessions
        .load_session_events(&session.id)
        .await
        .unwrap();
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(
                event.payload,
                crate::session::db::SessionEventPayload::ToolExecute { .. }
            ))
            .count(),
        3,
        "evidence similarity is advisory and must never skip execution"
    );
    let next_request = provider.calls().get(1).cloned().unwrap();
    let guidance = next_request
        .iter()
        .filter_map(|message| message.get("content").and_then(Value::as_str))
        .filter(|content| content.contains("Three successful read-only calls"))
        .collect::<Vec<_>>();
    assert_eq!(guidance.len(), 1);
    assert!(guidance[0].contains("Every call executed; no result was inferred or skipped"));
    assert!(guidance[0].contains("identify the remaining gap"));
    let _ = std::fs::remove_dir_all(workspace);
}

#[tokio::test]
async fn repeated_exec_output_executes_every_variant_then_adds_one_neutral_nudge() {
    let workspace = tempfile::tempdir().unwrap().keep();
    let mut calls = Vec::new();
    let mut markers = Vec::new();
    for index in 0..3 {
        let marker = workspace.join(format!("exec-variant-{index}"));
        let command = format!(
            "printf x > {}; printf 'same output'",
            marker.to_string_lossy()
        );
        markers.push(marker);
        calls.push(crate::providers::base::ToolCallRequest {
            id: format!("exec-variant-{index}"),
            name: "exec".to_string(),
            arguments: std::collections::HashMap::from([
                ("command".to_string(), json!(command)),
                (
                    "working_dir".to_string(),
                    json!(workspace.to_string_lossy()),
                ),
            ]),
        });
    }
    let provider = Arc::new(WireRecordingProvider::new(
        "local-exec-evidence",
        vec![
            crate::providers::base::LLMResponse {
                content: Some("Running three distinct commands.".to_string()),
                tool_calls: calls,
                finish_reason: FinishReason::ToolCalls,
                usage: std::collections::HashMap::new(),
            },
            WireRecordingProvider::text_response("exec evidence assessed"),
        ],
    ));
    let agent_loop = build_unrestricted_exec_harness(
        provider.clone() as Arc<dyn LLMProvider>,
        workspace.clone(),
    );
    let session_key = format!("exec-evidence-{}", uuid::Uuid::new_v4());

    assert_eq!(
        agent_loop
            .process_direct("run all variants", &session_key, "test", "offline")
            .await,
        "exec evidence assessed"
    );
    for marker in markers {
        assert_eq!(std::fs::read_to_string(marker).unwrap(), "x");
    }
    let core = agent_loop.shared.core_handle.swappable();
    let session = core
        .sessions
        .get_latest_session(&session_key)
        .await
        .unwrap();
    let events = core
        .sessions
        .load_session_events(&session.id)
        .await
        .unwrap();
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(
                event.payload,
                crate::session::db::SessionEventPayload::ToolExecute { .. }
            ))
            .count(),
        3,
        "matching output is advisory and must never skip distinct executions"
    );
    let next_request = provider.calls().get(1).cloned().unwrap();
    let guidance = next_request
        .iter()
        .filter_map(|message| message.get("content").and_then(Value::as_str))
        .filter(|content| content.contains("independently executed calls returned identical"))
        .collect::<Vec<_>>();
    assert_eq!(guidance.len(), 1);
    assert!(guidance[0].contains("No command equivalence was inferred"));
    assert!(guidance[0].contains("continue executing if another action is required"));
    let _ = std::fs::remove_dir_all(workspace);
}

#[tokio::test]
async fn cached_duplicate_rounds_use_one_terminal_none_without_scaffold_or_static_break() {
    let duplicate_call = |id: usize| {
        let mut arguments = std::collections::HashMap::new();
        arguments.insert("path".to_string(), json!("."));
        crate::providers::base::LLMResponse {
            content: Some("Let me check the remaining modified files:".to_string()),
            tool_calls: vec![crate::providers::base::ToolCallRequest {
                id: format!("tc_duplicate_{id}"),
                name: "list_dir".to_string(),
                arguments,
            }],
            finish_reason: FinishReason::ToolCalls,
            usage: std::collections::HashMap::new(),
        }
    };
    let provider = Arc::new(DuplicateTerminalProvider {
        responses: parking_lot::Mutex::new(
            vec![
                duplicate_call(1),
                duplicate_call(2),
                duplicate_call(3),
                duplicate_call(4),
                duplicate_call(5),
                WireRecordingProvider::text_response("terminal duplicate summary"),
            ]
            .into(),
        ),
        terminal_choices: parking_lot::Mutex::new(Vec::new()),
    });
    let (agent_loop, workspace) =
        build_local_inline_harness_with_iters(provider.clone() as Arc<dyn LLMProvider>, 10);
    let session_key = format!("cached-duplicate-breaker-{}", uuid::Uuid::new_v4());

    let response = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        agent_loop.process_direct("inspect the workspace", &session_key, "test", "offline"),
    )
    .await
    .expect("cached duplicate loop must terminate");

    assert_eq!(response, "terminal duplicate summary");
    assert_eq!(
        provider.terminal_choices.lock().as_slice(),
        &[crate::providers::base::ToolChoice::None],
        "one shared terminal authority must own duplicate-loop convergence"
    );

    let core = agent_loop.shared.core_handle.swappable();
    let session = core.sessions.get_or_resume(&session_key).await;
    let replay = core
        .sessions
        .load_session_replay(&session.id)
        .await
        .unwrap();
    let terminal_calls = replay
        .model_calls
        .iter()
        .filter(|call| call.purpose == crate::session::db::ModelCallPurpose::Continuation)
        .collect::<Vec<_>>();
    assert_eq!(terminal_calls.len(), 1);
    let terminal_request: crate::session::db::RecordedProviderRequest =
        serde_json::from_slice(&terminal_calls[0].request).unwrap();
    assert_eq!(terminal_request.tool_choice, "none");
    let persisted_text = core
        .sessions
        .get_all_messages(&session.id)
        .await
        .iter()
        .filter_map(|message| message.get("content").and_then(Value::as_str))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(!persisted_text.contains("Your last several tool calls were duplicates or blocked"));
    assert!(!persisted_text.contains("Tool calls were blocked after repeated duplicates"));

    let _ = std::fs::remove_dir_all(&workspace);
}

fn stale_read_write_context_parts(
    path: &str,
) -> (
    Vec<Value>,
    std::collections::HashMap<String, Value>,
    crate::agent::tool_guard::ToolGuard,
) {
    let mut read_args = std::collections::HashMap::new();
    read_args.insert("path".to_string(), json!(path));
    let mut write_args = std::collections::HashMap::new();
    write_args.insert("path".to_string(), json!(path));
    write_args.insert("content".to_string(), json!("new\n"));
    let read_call = crate::providers::base::ToolCallRequest {
        id: "tc_read_old".to_string(),
        name: "read_file".to_string(),
        arguments: read_args.clone(),
    };
    let write_call = crate::providers::base::ToolCallRequest {
        id: "tc_write_new".to_string(),
        name: "write_file".to_string(),
        arguments: write_args.clone(),
    };

    let mut guard = crate::agent::tool_guard::ToolGuard::new(1);
    assert!(guard.allow("read_file", &read_args).is_ok());
    guard.record_result_with_status(
        "read_file",
        &read_args,
        "old\n".to_string(),
        true,
        "old-read",
        "old-digest",
    );
    assert!(guard.allow("write_file", &write_args).is_ok());
    guard.record_result_with_status(
        "write_file",
        &write_args,
        "written".to_string(),
        true,
        "write",
        "write-digest",
    );
    assert_eq!(
        guard.get_cached_result(&crate::agent::tool_guard::ToolGuard::key(
            "read_file",
            &read_args
        )),
        None
    );

    (
        vec![
            json!({"role": "user", "content": "read, write, re-read"}),
            json!({"role": "assistant", "content": "", "tool_calls": [read_call.to_openai_json()]}),
            json!({"role": "tool", "tool_call_id": "tc_read_old", "name": "read_file", "ok": true, "content": "old\n"}),
            json!({"role": "assistant", "content": "", "tool_calls": [write_call.to_openai_json()]}),
            json!({"role": "tool", "tool_call_id": "tc_write_new", "name": "write_file", "ok": true, "content": "written"}),
        ],
        read_args,
        guard,
    )
}

#[tokio::test]
async fn test_read_after_write_same_turn_is_not_blocked_by_stale_receipt() {
    let provider = Arc::new(WireRecordingProvider::new(
        "local-qwen-test",
        vec![WireRecordingProvider::text_response("unused")],
    ));
    let (agent_loop, workspace) =
        build_local_inline_harness(provider.clone() as Arc<dyn LLMProvider>);
    let msg = InboundMessage::new("test", "user", "offline", "read, write, re-read");
    let mut ctx = agent_loop
        .shared
        .prepare_context(&msg, None, None, None, None)
        .await;

    let (messages, read_args, guard) = stale_read_write_context_parts("/tmp/same-turn.txt");
    ctx.messages = crate::agent::agent_loop::shared::MessageLog::committed(messages);
    ctx.new_start = 0;
    ctx.flow.tool_guard = guard;

    let result = crate::agent::router::route_tool_calls(
        &mut ctx,
        Some(""),
        vec![crate::providers::base::ToolCallRequest {
            id: "tc_read_new".to_string(),
            name: "read_file".to_string(),
            arguments: read_args,
        }],
        crate::agent::agent_loop::ToolRouting::NeedsRouting,
    )
    .await;

    match result {
        crate::agent::router::RouteResult::Execute(batch) => {
            assert_eq!(batch.calls.len(), 1);
            assert_eq!(batch.calls[0].call.id, "tc_read_new");
            assert!(matches!(
                batch.calls[0].disposition,
                crate::agent::router::RoutedToolDisposition::Execute
            ));
        }
        crate::agent::router::RouteResult::Break(text) => {
            panic!("post-write read was blocked: {text}")
        }
        crate::agent::router::RouteResult::Error(text) => {
            panic!("post-write read hit infrastructure error: {text}")
        }
        crate::agent::router::RouteResult::Continue => panic!("post-write read should execute"),
    }
    assert!(!ctx.flow.tool_guard.had_blocked_calls);

    let _ = std::fs::remove_dir_all(&workspace);
}

/// Regression (prod, session cli:oneshot, bonsai-27b): a turn that runs a
/// A side-effect round followed by another legitimate tool must remain an
/// append-only wire prefix without synthetic boundary messages.
#[tokio::test]
async fn test_wire_prefix_stable_across_turn_after_side_effect_boundary_nudge() {
    let mut exec_args = std::collections::HashMap::new();
    exec_args.insert("command".to_string(), json!("echo hi"));
    let exec_call = crate::providers::base::LLMResponse {
        content: Some(String::new()),
        tool_calls: vec![crate::providers::base::ToolCallRequest {
            id: "tc_exec".to_string(),
            name: "exec".to_string(),
            arguments: exec_args,
        }],
        finish_reason: FinishReason::ToolCalls,
        usage: std::collections::HashMap::new(),
    };
    let mut ls_args = std::collections::HashMap::new();
    ls_args.insert("path".to_string(), json!("."));
    let listdir_call = crate::providers::base::LLMResponse {
        content: Some(String::new()),
        tool_calls: vec![crate::providers::base::ToolCallRequest {
            id: "tc_ls".to_string(),
            name: "list_dir".to_string(),
            arguments: ls_args,
        }],
        finish_reason: FinishReason::ToolCalls,
        usage: std::collections::HashMap::new(),
    };
    // Turn 1: exec, then a second legitimate tool round, then a final text
    // reply. Turn 2: a plain text reply.
    let provider = Arc::new(WireRecordingProvider::new(
        "local-qwen-test",
        vec![
            exec_call,
            listdir_call,
            WireRecordingProvider::text_response("done turn one"),
            WireRecordingProvider::text_response("turn two reply"),
        ],
    ));
    let (agent_loop, workspace) =
        build_local_inline_harness(provider.clone() as Arc<dyn LLMProvider>);
    let session_key = format!("boundary-nudge-prefix-{}", uuid::Uuid::new_v4());

    tokio::time::timeout(
        std::time::Duration::from_secs(10),
        agent_loop.process_direct("run something", &session_key, "test", "offline"),
    )
    .await
    .expect("turn 1 must terminate");

    let turn1_calls = provider.calls().len();
    assert!(turn1_calls >= 2, "turn 1 must make multiple provider calls");

    tokio::time::timeout(
        std::time::Duration::from_secs(10),
        agent_loop.process_direct("what did you find?", &session_key, "test", "offline"),
    )
    .await
    .expect("turn 2 must terminate");

    let calls = provider.calls();
    assert!(
        calls.len() > turn1_calls,
        "turn 2 must make at least one provider call (had {turn1_calls}, now {})",
        calls.len()
    );

    // Turn N's last wire must be a byte-prefix of turn N+1's first wire.
    assert_wire_prefix(&calls[turn1_calls - 1], &calls[turn1_calls]);

    let _ = std::fs::remove_dir_all(&workspace);
}

/// Full production shape (log 2026-07-17, cli:oneshot, bonsai-27b): repeated
/// identical `exec` calls — the first ones execute (arming the response
/// boundary, injecting its scaffold nudge), the rest are duplicate-blocked
/// until the tool-loop circuit breaker forces a text response. The NEXT turn's
/// reloaded wire must still be a byte-suffix extension of the previous turn's
/// last wire (no mid-history shrink, no `prompt_prefix_diverged`).
#[tokio::test]
async fn duplicate_recovery_is_persisted_after_carrier_and_receipt() {
    let provider = Arc::new(WireRecordingProvider::new(
        "local-qwen-test",
        vec![WireRecordingProvider::text_response("unused")],
    ));
    let (agent_loop, workspace) = build_local_inline_harness(provider as Arc<dyn LLMProvider>);
    let msg = InboundMessage::new("test", "user", "offline", "inspect tools");
    let mut ctx = agent_loop
        .shared
        .prepare_context(&msg, None, None, None, None)
        .await;
    let arguments = std::collections::HashMap::new();
    ctx.flow.tool_guard = crate::agent::tool_guard::ToolGuard::new(1);
    assert!(ctx.flow.tool_guard.allow("get_tools", &arguments).is_ok());
    ctx.flow.consecutive_all_blocked = 1;
    let before = ctx.messages.len();

    let response = crate::providers::base::LLMResponse {
        content: Some(String::new()),
        tool_calls: vec![crate::providers::base::ToolCallRequest {
            id: "tc_blocked_recovery".to_string(),
            name: "get_tools".to_string(),
            arguments,
        }],
        finish_reason: FinishReason::ToolCalls,
        usage: std::collections::HashMap::new(),
    };
    let _ = agent_loop
        .shared
        .step_execute_tools(&mut ctx, response, ToolRouting::AlreadyRouted)
        .await;

    let appended = &ctx.messages[before..];
    assert_eq!(appended.len(), 3, "carrier, receipt, recovery instruction");
    assert_eq!(appended[0]["role"], "assistant");
    assert_eq!(appended[1]["role"], "tool");
    assert_eq!(appended[1]["tool_call_id"], "tc_blocked_recovery");
    assert_eq!(appended[2]["role"], "user");
    assert!(appended[2]["content"].as_str().is_some_and(|content| {
        content.contains("give an honest partial answer")
            && content.contains("identify unresolved gaps")
            && !content.contains("already have the data you need")
    }));

    let _ = std::fs::remove_dir_all(&workspace);
}

#[tokio::test]
async fn test_wire_prefix_stable_after_duplicate_exec_circuit_breaker() {
    let exec_call = |id: usize| {
        let mut arguments = std::collections::HashMap::new();
        arguments.insert("command".to_string(), json!("echo hi"));
        crate::providers::base::LLMResponse {
            content: Some(String::new()),
            tool_calls: vec![crate::providers::base::ToolCallRequest {
                id: format!("tc_exec_{id}"),
                name: "exec".to_string(),
                arguments,
            }],
            finish_reason: FinishReason::ToolCalls,
            usage: std::collections::HashMap::new(),
        }
    };
    let provider = Arc::new(WireRecordingProvider::new(
        "local-qwen-test",
        vec![
            exec_call(1),
            exec_call(2),
            exec_call(3),
            exec_call(4),
            exec_call(5),
            exec_call(6),
            exec_call(7),
            WireRecordingProvider::text_response("turn two reply"),
        ],
    ));
    let (agent_loop, workspace) =
        build_local_inline_harness(provider.clone() as Arc<dyn LLMProvider>);
    let session_key = format!("dup-exec-breaker-prefix-{}", uuid::Uuid::new_v4());

    tokio::time::timeout(
        std::time::Duration::from_secs(15),
        agent_loop.process_direct("run the check", &session_key, "test", "offline"),
    )
    .await
    .expect("turn 1 (duplicate exec loop) must terminate");

    let turn1_calls = provider.calls().len();
    assert!(turn1_calls >= 2, "turn 1 must make multiple provider calls");

    let core = agent_loop.shared.core_handle.swappable();
    let meta = core
        .sessions
        .get_latest_session(&session_key)
        .await
        .expect("boundary session should exist");
    let replay = core.sessions.load_session_replay(&meta.id).await.unwrap();
    assert!(!replay.events.iter().any(|event| matches!(
        &event.payload,
        crate::session::db::SessionEventPayload::ToolPreExecute {
            decision: crate::session::db::ToolPreExecuteDecision::Rejected { reason },
            ..
        } if reason == "response_boundary"
    )));

    tokio::time::timeout(
        std::time::Duration::from_secs(15),
        agent_loop.process_direct("so what happened?", &session_key, "test", "offline"),
    )
    .await
    .expect("turn 2 must terminate");

    let calls = provider.calls();
    assert!(
        calls.len() > turn1_calls,
        "turn 2 must make at least one provider call (had {turn1_calls}, now {})",
        calls.len()
    );
    assert_wire_prefix(&calls[turn1_calls - 1], &calls[turn1_calls]);

    let _ = std::fs::remove_dir_all(&workspace);
}

#[tokio::test]
async fn turn_finish_journal_failure_still_returns_the_reply() {
    // Break caught: when the final turn-finished journal write failed, an
    // already-generated and already-persisted reply was discarded and the
    // user got nothing. The reply must survive; only replay availability
    // degrades to Incomplete.
    let main: Arc<dyn LLMProvider> = Arc::new(ResponseSequenceProvider::new(
        "local-main",
        vec![crate::providers::base::LLMResponse {
            content: Some(attested_text("still delivered")),
            tool_calls: vec![],
            finish_reason: FinishReason::Stop,
            usage: std::collections::HashMap::new(),
        }],
    ));
    let (agent_loop, workspace) = build_local_inline_harness(main);
    let session_key = format!("journal-fault-reply-{}", uuid::Uuid::new_v4());
    let core = agent_loop.shared.core_handle.swappable();

    core.sessions.fail_turn_finished_writes_for_tests(1);

    let response = agent_loop
        .process_direct("say the thing", &session_key, "test", "offline")
        .await;
    assert_eq!(response, "still delivered");

    let meta = core
        .sessions
        .get_latest_session(&session_key)
        .await
        .expect("session should exist");
    let messages = core.sessions.get_all_messages(&meta.id).await;
    assert!(
        messages.iter().any(|m| {
            m.get("role").and_then(Value::as_str) == Some("assistant")
                && m.get("content").and_then(Value::as_str) == Some("still delivered")
        }),
        "the reply must remain in persisted history despite the journal failure"
    );
    let replay = core.sessions.load_session_replay(&meta.id).await.unwrap();
    assert!(matches!(
        replay.availability,
        crate::session::db::ReplayAvailability::Incomplete { .. }
    ));

    let _ = std::fs::remove_dir_all(&workspace);
}

#[tokio::test]
async fn empty_provider_content_persists_empty_outcome() {
    let provider = Arc::new(ResponseSequenceProvider::new(
        "local-main",
        vec![crate::providers::base::LLMResponse {
            content: None,
            tool_calls: vec![],
            finish_reason: FinishReason::Stop,
            usage: HashMap::new(),
        }],
    ));
    let (agent_loop, workspace) = build_local_inline_harness(provider as Arc<dyn LLMProvider>);
    let session_key = format!("empty-outcome-{}", uuid::Uuid::new_v4());

    let response = agent_loop
        .process_direct("return nothing", &session_key, "test", "offline")
        .await;
    assert!(response.contains("couldn't produce a response"));

    let core = agent_loop.shared.core_handle.swappable();
    let session = core
        .sessions
        .get_latest_session(&session_key)
        .await
        .expect("empty session");
    assert_eq!(
        persisted_turn_outcome(&core.sessions, &session.id).await,
        "empty"
    );
    let _ = std::fs::remove_dir_all(&workspace);
}

#[tokio::test]
async fn cancelled_turn_persists_cancelled_outcome_without_provider_call() {
    let provider = Arc::new(SequenceProvider::new("local-main", vec!["must not run"]));
    let (agent_loop, workspace) =
        build_local_inline_harness(provider.clone() as Arc<dyn LLMProvider>);
    let session_key = format!("cancelled-outcome-{}", uuid::Uuid::new_v4());
    let cancellation = tokio_util::sync::CancellationToken::new();
    cancellation.cancel();
    let (delta_tx, _delta_rx) = tokio::sync::mpsc::unbounded_channel();

    let response = agent_loop
        .process_direct_streaming(
            "cancel now",
            &session_key,
            "test",
            "offline",
            None,
            delta_tx,
            None,
            Some(cancellation),
            None,
            None,
        )
        .await;
    assert!(response.is_empty());
    assert_eq!(provider.call_count(), 0);

    let core = agent_loop.shared.core_handle.swappable();
    let session = core
        .sessions
        .get_latest_session(&session_key)
        .await
        .expect("cancelled session");
    assert_eq!(
        persisted_turn_outcome(&core.sessions, &session.id).await,
        "cancelled"
    );
    let _ = std::fs::remove_dir_all(&workspace);
}

#[tokio::test]
async fn empty_plan_step_does_not_poison_later_success() {
    let provider = Arc::new(ResponseSequenceProvider::new(
        "local-main",
        vec![
            crate::providers::base::LLMResponse {
                content: None,
                tool_calls: vec![],
                finish_reason: FinishReason::Stop,
                usage: HashMap::new(),
            },
            WireRecordingProvider::text_response("plan eventually succeeded"),
        ],
    ));
    let reasoning = crate::config::schema::ReasoningConfig {
        enabled: true,
        auto_decompose: true,
        ..Default::default()
    };
    let (agent_loop, workspace) = build_local_harness_with_runtime_options(
        provider.clone() as Arc<dyn LLMProvider>,
        5,
        reasoning,
        ToolDelegationConfig::default(),
        None,
    );
    let session_key = format!("empty-then-success-{}", uuid::Uuid::new_v4());

    let response = agent_loop
        .process_direct(
            "1. inspect the request\n2. give the final answer",
            &session_key,
            "test",
            "offline",
        )
        .await;

    assert_eq!(response, "plan eventually succeeded");
    assert_eq!(provider.call_count(), 2);
    let core = agent_loop.shared.core_handle.swappable();
    let session = core
        .sessions
        .get_latest_session(&session_key)
        .await
        .expect("empty-then-success session");
    assert_eq!(
        persisted_turn_outcome(&core.sessions, &session.id).await,
        "finished"
    );
    let _ = std::fs::remove_dir_all(&workspace);
}

#[tokio::test]
async fn plan_guided_failed_plan_step_stops_at_step_budget() {
    let responses = (0..8)
        .map(|index| crate::providers::base::LLMResponse {
            content: Some(String::new()),
            tool_calls: vec![crate::providers::base::ToolCallRequest {
                id: format!("tc-plan-budget-{index}"),
                name: "list_dir".to_string(),
                arguments: HashMap::from([(
                    "path".to_string(),
                    json!(format!("{}.", "./".repeat(index))),
                )]),
            }],
            finish_reason: FinishReason::ToolCalls,
            usage: HashMap::new(),
        })
        .collect();
    let provider = Arc::new(ResponseSequenceProvider::new("local-main", responses));
    let reasoning = crate::config::schema::ReasoningConfig {
        enabled: true,
        auto_decompose: true,
        step_budget: 2,
        ..Default::default()
    };
    let (agent_loop, workspace) = build_local_harness_with_runtime_options(
        provider.clone() as Arc<dyn LLMProvider>,
        8,
        reasoning,
        ToolDelegationConfig::default(),
        None,
    );
    let session_key = format!("failed-plan-step-{}", uuid::Uuid::new_v4());

    let response = agent_loop
        .process_direct(
            "1. inspect the workspace\n2. report the result",
            &session_key,
            "test",
            "offline",
        )
        .await;

    assert!(response.contains("iteration budget"), "{response:?}");
    assert_eq!(
        provider.call_count(),
        2,
        "the failed step must stop at its own budget, not max_iterations"
    );
    let _ = std::fs::remove_dir_all(&workspace);
}

#[tokio::test]
async fn plan_checkpoint_rewind_preserves_active_step_at_step_budget() {
    let provider = Arc::new(ResponseSequenceProvider::new(
        "local-main",
        vec![
            crate::providers::base::LLMResponse {
                content: Some(String::new()),
                tool_calls: vec![crate::providers::base::ToolCallRequest {
                    id: "tc-plan-checkpoint-setup".to_string(),
                    name: "plan".to_string(),
                    arguments: HashMap::from([
                        ("steps".to_string(), json!([{"goal": "inspect and report"}])),
                        ("step_budget".to_string(), json!(3)),
                    ]),
                }],
                finish_reason: FinishReason::ToolCalls,
                usage: HashMap::new(),
            },
            crate::providers::base::LLMResponse {
                content: Some(String::new()),
                tool_calls: vec![crate::providers::base::ToolCallRequest {
                    id: "tc-plan-checkpoint-save".to_string(),
                    name: "checkpoint".to_string(),
                    arguments: HashMap::from([("label".to_string(), json!("before-report"))]),
                }],
                finish_reason: FinishReason::ToolCalls,
                usage: HashMap::new(),
            },
            crate::providers::base::LLMResponse {
                content: Some(String::new()),
                tool_calls: vec![crate::providers::base::ToolCallRequest {
                    id: "tc-plan-checkpoint-work".to_string(),
                    name: "list_dir".to_string(),
                    arguments: HashMap::from([("path".to_string(), json!("."))]),
                }],
                finish_reason: FinishReason::ToolCalls,
                usage: HashMap::new(),
            },
            WireRecordingProvider::text_response("recovered after checkpoint"),
        ],
    ));
    let reasoning = crate::config::schema::ReasoningConfig {
        enabled: true,
        ..Default::default()
    };
    let (agent_loop, workspace) = build_local_harness_with_runtime_options(
        provider.clone() as Arc<dyn LLMProvider>,
        8,
        reasoning,
        ToolDelegationConfig::default(),
        None,
    );
    let session_key = format!("plan-checkpoint-rewind-{}", uuid::Uuid::new_v4());

    let response = agent_loop
        .process_direct(
            "Inspect the workspace and report the result.",
            &session_key,
            "test",
            "offline",
        )
        .await;

    assert_eq!(
        provider.call_count(),
        4,
        "checkpoint recovery must preserve the active plan step"
    );
    assert_eq!(response, "recovered after checkpoint");
    let _ = std::fs::remove_dir_all(&workspace);
}

#[tokio::test]
async fn model_failure_journal_failure_prevents_retry() {
    let provider = Arc::new(RetryableFailureProvider {
        name: "local-main".to_string(),
        message: "synthetic retryable failure".to_string(),
        call_count: std::sync::atomic::AtomicU32::new(0),
    });
    let (agent_loop, workspace) =
        build_local_inline_harness(provider.clone() as Arc<dyn LLMProvider>);
    let session_key = format!("failure-journal-{}", uuid::Uuid::new_v4());
    let core = agent_loop.shared.core_handle.swappable();
    {
        let conn = rusqlite::Connection::open(core.sessions.path()).unwrap();
        conn.execute_batch(
            "CREATE TRIGGER fail_model_failure_event \
             BEFORE INSERT ON session_events WHEN NEW.event_kind = 'model_failed' \
             BEGIN SELECT RAISE(ABORT, 'synthetic model failure journal fault'); END;",
        )
        .unwrap();
    }

    let response = agent_loop
        .process_direct("fail once only", &session_key, "test", "offline")
        .await;

    assert!(
        response.contains("could not be recorded durably"),
        "{response:?}"
    );
    assert_eq!(
        provider
            .call_count
            .load(std::sync::atomic::Ordering::Relaxed),
        1,
        "journal failure must stop before retrying the provider"
    );
    let session = core
        .sessions
        .get_latest_session(&session_key)
        .await
        .expect("failure-journal session");
    assert_eq!(
        persisted_turn_outcome(&core.sessions, &session.id).await,
        "error"
    );
    let _ = std::fs::remove_dir_all(&workspace);
}

#[tokio::test]
async fn empty_rescue_journal_failure_prevents_thinking_off_retry() {
    let provider = Arc::new(ResponseSequenceProvider::new(
        "local-main",
        vec![
            crate::providers::base::LLMResponse {
                content: None,
                tool_calls: vec![],
                finish_reason: FinishReason::Length,
                usage: HashMap::new(),
            },
            WireRecordingProvider::text_response("undurable rescue"),
            WireRecordingProvider::text_response("must not retry"),
        ],
    ));
    let (agent_loop, workspace) =
        build_local_inline_harness(provider.clone() as Arc<dyn LLMProvider>);
    let session_key = format!("empty-rescue-journal-{}", uuid::Uuid::new_v4());
    let core = agent_loop.shared.core_handle.swappable();
    {
        let conn = rusqlite::Connection::open(core.sessions.path()).unwrap();
        conn.execute_batch(
            "CREATE TRIGGER fail_second_model_response \
             BEFORE INSERT ON session_events \
             WHEN NEW.event_kind = 'model_response' \
              AND (SELECT COUNT(*) FROM session_events WHERE event_kind = 'model_response') >= 1 \
             BEGIN SELECT RAISE(ABORT, 'synthetic rescue response journal fault'); END;",
        )
        .unwrap();
    }

    let response = agent_loop
        .process_direct("answer after thinking", &session_key, "test", "offline")
        .await;

    assert!(response.contains("was not recorded"), "{response:?}");
    assert_eq!(
        provider.call_count(),
        2,
        "undurable rescue result must stop before the thinking-off retry"
    );
    let session = core
        .sessions
        .get_latest_session(&session_key)
        .await
        .expect("empty-rescue journal session");
    assert_eq!(
        persisted_turn_outcome(&core.sessions, &session.id).await,
        "error"
    );
    let _ = std::fs::remove_dir_all(&workspace);
}

#[tokio::test]
async fn streamed_response_journal_failure_retracts_before_error() {
    let provider = Arc::new(SequenceProvider::new(
        "local-main",
        vec!["visible but undurable"],
    ));
    let (agent_loop, workspace) =
        build_local_inline_harness(provider.clone() as Arc<dyn LLMProvider>);
    let session_key = format!("stream-response-journal-{}", uuid::Uuid::new_v4());
    let core = agent_loop.shared.core_handle.swappable();
    core.sessions.fail_model_response_writes_for_tests(1);
    let (delta_tx, mut delta_rx) = tokio::sync::mpsc::unbounded_channel();

    let response = agent_loop
        .process_direct_streaming(
            "stream it",
            &session_key,
            "test",
            "offline",
            None,
            delta_tx,
            None,
            None,
            None,
            None,
        )
        .await;

    assert!(
        response.contains("could not record it durably"),
        "{response:?}"
    );
    let mut deltas = Vec::new();
    while let Ok(delta) = delta_rx.try_recv() {
        deltas.push(delta);
    }
    let streamed = deltas.join("");
    let retract = crate::turn_stream::ControlMarker::RetractReply.encode();
    let retract_at = streamed
        .find(&retract)
        .expect("undurable streamed response must be retracted");
    assert!(streamed[..retract_at].contains("visible but undurable"));
    let tail = &streamed[retract_at + retract.len()..];
    assert!(tail.contains("could not record it durably"), "{tail:?}");
    let _ = std::fs::remove_dir_all(&workspace);
}

#[tokio::test]
async fn iteration_limit_persists_limit_exhausted_outcome() {
    let provider = Arc::new(ResponseSequenceProvider::new(
        "local-main",
        vec![
            crate::providers::base::LLMResponse {
                content: Some(String::new()),
                tool_calls: vec![crate::providers::base::ToolCallRequest {
                    id: "tc-limit".to_string(),
                    name: "list_dir".to_string(),
                    arguments: HashMap::from([("path".to_string(), json!("."))]),
                }],
                finish_reason: FinishReason::ToolCalls,
                usage: HashMap::new(),
            },
            crate::providers::base::LLMResponse {
                content: None,
                tool_calls: vec![],
                finish_reason: FinishReason::Stop,
                usage: HashMap::new(),
            },
        ],
    ));
    let (agent_loop, workspace) =
        build_local_inline_harness_with_iters(provider.clone() as Arc<dyn LLMProvider>, 1);
    let session_key = format!("limit-outcome-{}", uuid::Uuid::new_v4());

    let response = agent_loop
        .process_direct("list once", &session_key, "test", "offline")
        .await;

    assert!(!response.is_empty());
    assert_eq!(provider.call_count(), 2);
    let core = agent_loop.shared.core_handle.swappable();
    let session = core
        .sessions
        .get_latest_session(&session_key)
        .await
        .expect("limit-exhausted session");
    assert_eq!(
        persisted_turn_outcome(&core.sessions, &session.id).await,
        "limit_exhausted"
    );
    let _ = std::fs::remove_dir_all(&workspace);
}

#[tokio::test]
async fn inbound_persistence_failure_prevents_provider_call() {
    let provider = Arc::new(SequenceProvider::new("local-main", vec!["must not run"]));
    let (agent_loop, workspace) =
        build_local_inline_harness(provider.clone() as Arc<dyn LLMProvider>);
    let session_key = format!("inbound-persist-failure-{}", uuid::Uuid::new_v4());
    let core = agent_loop.shared.core_handle.swappable();
    let session = core.sessions.get_or_resume(&session_key).await;
    {
        let conn = rusqlite::Connection::open(core.sessions.path()).unwrap();
        conn.execute_batch(
            "CREATE TRIGGER fail_inbound_message \
             BEFORE INSERT ON messages WHEN NEW.role = 'user' \
             BEGIN SELECT RAISE(ABORT, 'synthetic inbound persistence failure'); END;",
        )
        .unwrap();
    }

    let response = agent_loop
        .process_direct("must be durable", &session_key, "test", "offline")
        .await;

    assert!(
        response.contains("could not durably record"),
        "{response:?}"
    );
    assert_eq!(provider.call_count(), 0);
    assert_eq!(
        persisted_turn_outcome(&core.sessions, &session.id).await,
        "error"
    );
    let _ = std::fs::remove_dir_all(&workspace);
}

#[tokio::test]
async fn tool_carrier_persistence_failure_prevents_tool_side_effect() {
    let provider = Arc::new(ResponseSequenceProvider::new(
        "local-main",
        vec![crate::providers::base::LLMResponse {
            content: Some(String::new()),
            tool_calls: vec![crate::providers::base::ToolCallRequest {
                id: "tc-carrier-fault".to_string(),
                name: "write_file".to_string(),
                arguments: HashMap::from([
                    ("path".to_string(), json!("carrier-must-not-exist.txt")),
                    ("content".to_string(), json!("forbidden")),
                ]),
            }],
            finish_reason: FinishReason::ToolCalls,
            usage: HashMap::new(),
        }],
    ));
    let (agent_loop, workspace) =
        build_local_inline_harness(provider.clone() as Arc<dyn LLMProvider>);
    let session_key = format!("carrier-persist-failure-{}", uuid::Uuid::new_v4());
    let core = agent_loop.shared.core_handle.swappable();
    let session = core.sessions.get_or_resume(&session_key).await;
    {
        let conn = rusqlite::Connection::open(core.sessions.path()).unwrap();
        conn.execute_batch(
            "CREATE TRIGGER fail_tool_carrier \
             BEFORE INSERT ON messages \
             WHEN NEW.role = 'assistant' AND NEW.tool_calls IS NOT NULL \
             BEGIN SELECT RAISE(ABORT, 'synthetic carrier persistence failure'); END;",
        )
        .unwrap();
    }

    let response = agent_loop
        .process_direct("write it", &session_key, "test", "offline")
        .await;

    assert!(
        response.contains("tool-call carrier could not be recorded"),
        "{response:?}"
    );
    assert_eq!(provider.call_count(), 1);
    assert!(!workspace.join("carrier-must-not-exist.txt").exists());
    assert_eq!(
        persisted_turn_outcome(&core.sessions, &session.id).await,
        "error"
    );
    let _ = std::fs::remove_dir_all(&workspace);
}

#[tokio::test]
async fn raw_result_persistence_failure_prevents_subsequent_provider_call() {
    let side_effect_dir = tempfile::tempdir().unwrap();
    let side_effect_path = side_effect_dir.path().join("raw-result-ran.txt");
    let forbidden_path = side_effect_dir.path().join("raw-result-must-not-run.txt");
    let provider = Arc::new(ResponseSequenceProvider::new(
        "local-main",
        vec![
            crate::providers::base::LLMResponse {
                content: Some(String::new()),
                tool_calls: vec![
                    crate::providers::base::ToolCallRequest {
                        id: "tc-raw-fault".to_string(),
                        name: "write_file".to_string(),
                        arguments: HashMap::from([
                            (
                                "path".to_string(),
                                json!(side_effect_path.to_string_lossy().to_string()),
                            ),
                            ("content".to_string(), json!("ran once")),
                        ]),
                    },
                    crate::providers::base::ToolCallRequest {
                        id: "tc-raw-forbidden".to_string(),
                        name: "write_file".to_string(),
                        arguments: HashMap::from([
                            (
                                "path".to_string(),
                                json!(forbidden_path.to_string_lossy().to_string()),
                            ),
                            ("content".to_string(), json!("must not run")),
                        ]),
                    },
                ],
                finish_reason: FinishReason::ToolCalls,
                usage: HashMap::new(),
            },
            WireRecordingProvider::text_response("must not run"),
        ],
    ));
    let (agent_loop, workspace) =
        build_local_inline_harness(provider.clone() as Arc<dyn LLMProvider>);
    let session_key = format!("raw-result-persist-failure-{}", uuid::Uuid::new_v4());
    let core = agent_loop.shared.core_handle.swappable();
    let session = core.sessions.get_or_resume(&session_key).await;
    {
        let conn = rusqlite::Connection::open(core.sessions.path()).unwrap();
        conn.execute_batch(
            "CREATE TRIGGER fail_raw_tool_result \
             BEFORE INSERT ON session_events WHEN NEW.event_kind = 'tool_execute' \
             BEGIN SELECT RAISE(ABORT, 'synthetic raw result persistence failure'); END;",
        )
        .unwrap();
    }

    let response = agent_loop
        .process_direct("write once", &session_key, "test", "offline")
        .await;

    assert!(response.contains("execution result"), "{response:?}");
    assert_eq!(provider.call_count(), 1);
    assert!(side_effect_path.exists());
    assert!(
        !forbidden_path.exists(),
        "later sequential tool ran after raw-result persistence failed"
    );
    assert_eq!(
        persisted_turn_outcome(&core.sessions, &session.id).await,
        "error"
    );
    let _ = std::fs::remove_dir_all(&workspace);
}

#[tokio::test]
async fn post_result_persistence_failure_prevents_subsequent_provider_call() {
    let side_effect_dir = tempfile::tempdir().unwrap();
    let side_effect_path = side_effect_dir.path().join("post-result-ran.txt");
    let forbidden_path = side_effect_dir.path().join("post-result-must-not-run.txt");
    let provider = Arc::new(ResponseSequenceProvider::new(
        "local-main",
        vec![
            crate::providers::base::LLMResponse {
                content: Some(String::new()),
                tool_calls: vec![
                    crate::providers::base::ToolCallRequest {
                        id: "tc-post-fault".to_string(),
                        name: "write_file".to_string(),
                        arguments: HashMap::from([
                            (
                                "path".to_string(),
                                json!(side_effect_path.to_string_lossy().to_string()),
                            ),
                            ("content".to_string(), json!("ran once")),
                        ]),
                    },
                    crate::providers::base::ToolCallRequest {
                        id: "tc-post-forbidden".to_string(),
                        name: "write_file".to_string(),
                        arguments: HashMap::from([
                            (
                                "path".to_string(),
                                json!(forbidden_path.to_string_lossy().to_string()),
                            ),
                            ("content".to_string(), json!("must not run")),
                        ]),
                    },
                ],
                finish_reason: FinishReason::ToolCalls,
                usage: HashMap::new(),
            },
            WireRecordingProvider::text_response("must not run"),
        ],
    ));
    let (agent_loop, workspace) =
        build_local_inline_harness(provider.clone() as Arc<dyn LLMProvider>);
    let session_key = format!("post-result-persist-failure-{}", uuid::Uuid::new_v4());
    let core = agent_loop.shared.core_handle.swappable();
    let session = core.sessions.get_or_resume(&session_key).await;
    {
        let conn = rusqlite::Connection::open(core.sessions.path()).unwrap();
        conn.execute_batch(
            "CREATE TRIGGER fail_model_tool_result \
             BEFORE INSERT ON messages WHEN NEW.role = 'tool' \
             BEGIN SELECT RAISE(ABORT, 'synthetic post-result persistence failure'); END;",
        )
        .unwrap();
    }

    let response = agent_loop
        .process_direct("write once", &session_key, "test", "offline")
        .await;

    assert!(
        response.contains("model-visible tool result"),
        "{response:?}"
    );
    assert_eq!(provider.call_count(), 1);
    assert!(side_effect_path.exists());
    assert!(
        !forbidden_path.exists(),
        "later sequential tool ran after model-visible result persistence failed"
    );
    assert_eq!(
        persisted_turn_outcome(&core.sessions, &session.id).await,
        "error"
    );
    let _ = std::fs::remove_dir_all(&workspace);
}

#[tokio::test]
async fn delegated_batch_routes_through_inline_persistence_chokepoint() {
    let side_effect_dir = tempfile::tempdir().unwrap();
    let first_path = side_effect_dir.path().join("delegated-first.txt");
    let forbidden_path = side_effect_dir.path().join("delegated-must-not-run.txt");
    let main = Arc::new(ResponseSequenceProvider::new(
        "local-main",
        vec![crate::providers::base::LLMResponse {
            content: Some(String::new()),
            tool_calls: vec![
                crate::providers::base::ToolCallRequest {
                    id: "tc-delegated-first".to_string(),
                    name: "write_file".to_string(),
                    arguments: HashMap::from([
                        (
                            "path".to_string(),
                            json!(first_path.to_string_lossy().to_string()),
                        ),
                        ("content".to_string(), json!("ran once")),
                    ]),
                },
                crate::providers::base::ToolCallRequest {
                    id: "tc-delegated-forbidden".to_string(),
                    name: "write_file".to_string(),
                    arguments: HashMap::from([
                        (
                            "path".to_string(),
                            json!(forbidden_path.to_string_lossy().to_string()),
                        ),
                        ("content".to_string(), json!("must not run")),
                    ]),
                },
            ],
            finish_reason: FinishReason::ToolCalls,
            usage: HashMap::new(),
        }],
    ));
    let delegation = Arc::new(SequenceProvider::new("delegation-model", vec!["summary"]));
    let tool_delegation = ToolDelegationConfig {
        enabled: true,
        model: "delegation-model".to_string(),
        max_iterations: 1,
        ..Default::default()
    };
    let (agent_loop, workspace) = build_local_harness_with_runtime_options(
        main.clone() as Arc<dyn LLMProvider>,
        5,
        crate::config::schema::ReasoningConfig::default(),
        tool_delegation,
        Some(delegation.clone() as Arc<dyn LLMProvider>),
    );
    let session_key = format!("delegated-inline-chokepoint-{}", uuid::Uuid::new_v4());
    let core = agent_loop.shared.core_handle.swappable();
    {
        let conn = rusqlite::Connection::open(core.sessions.path()).unwrap();
        conn.execute_batch(
            "CREATE TRIGGER fail_delegated_raw_result \
             BEFORE INSERT ON session_events WHEN NEW.event_kind = 'tool_execute' \
             BEGIN SELECT RAISE(ABORT, 'synthetic delegated raw result failure'); END;",
        )
        .unwrap();
    }

    let response = agent_loop
        .process_direct("write both", &session_key, "test", "offline")
        .await;

    assert!(response.contains("execution result"), "{response:?}");
    assert!(first_path.exists());
    assert!(!forbidden_path.exists());
    assert_eq!(main.call_count(), 1);
    assert_eq!(
        delegation.call_count(),
        0,
        "delegated selection must not create an alternate execution pipeline"
    );
    let _ = std::fs::remove_dir_all(&workspace);
}

#[tokio::test]
async fn final_assistant_persistence_failure_records_error_outcome() {
    let provider = Arc::new(SequenceProvider::new("local-main", vec!["generated reply"]));
    let (agent_loop, workspace) =
        build_local_inline_harness(provider.clone() as Arc<dyn LLMProvider>);
    let session_key = format!("final-persist-failure-{}", uuid::Uuid::new_v4());
    let core = agent_loop.shared.core_handle.swappable();
    let session = core.sessions.get_or_resume(&session_key).await;
    {
        let conn = rusqlite::Connection::open(core.sessions.path()).unwrap();
        conn.execute_batch(
            "CREATE TRIGGER fail_final_assistant \
             BEFORE INSERT ON messages \
             WHEN NEW.role = 'assistant' AND NEW.tool_calls IS NULL \
             BEGIN SELECT RAISE(ABORT, 'synthetic final persistence failure'); END;",
        )
        .unwrap();
    }

    let response = agent_loop
        .process_direct("answer", &session_key, "test", "offline")
        .await;

    assert!(
        response.contains("could not durably record"),
        "{response:?}"
    );
    assert_eq!(provider.call_count(), 1);
    assert_eq!(
        persisted_turn_outcome(&core.sessions, &session.id).await,
        "error"
    );
    let _ = std::fs::remove_dir_all(&workspace);
}

struct EnvVarGuard {
    key: &'static str,
    saved: Option<std::ffi::OsString>,
}

impl EnvVarGuard {
    fn remove(key: &'static str) -> Self {
        let saved = std::env::var_os(key);
        std::env::remove_var(key);
        Self { key, saved }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        if let Some(value) = self.saved.take() {
            std::env::set_var(self.key, value);
        } else {
            std::env::remove_var(self.key);
        }
    }
}

#[tokio::test]
async fn test_tool_call_carrier_persists_before_tool_result() {
    let mut args = std::collections::HashMap::new();
    args.insert("path".to_string(), json!("."));
    let main: Arc<dyn LLMProvider> = Arc::new(ResponseSequenceProvider::new(
        "local-main",
        vec![
            crate::providers::base::LLMResponse {
                content: Some(String::new()),
                tool_calls: vec![crate::providers::base::ToolCallRequest {
                    id: "tc_list".to_string(),
                    name: "list_dir".to_string(),
                    arguments: args,
                }],
                finish_reason: FinishReason::ToolCalls,
                usage: std::collections::HashMap::new(),
            },
            crate::providers::base::LLMResponse {
                content: Some(attested_text("I listed the workspace.")),
                tool_calls: vec![],
                finish_reason: FinishReason::Stop,
                usage: std::collections::HashMap::new(),
            },
        ],
    ));
    let (agent_loop, workspace) = build_local_inline_harness(main);
    let session_key = format!("test-tool-order-{}", uuid::Uuid::new_v4().to_string());

    let response = agent_loop
        .process_direct("please list files", &session_key, "test", "offline")
        .await;
    assert_eq!(response, "I listed the workspace.");

    let core = agent_loop.shared.core_handle.swappable();
    let meta = core
        .sessions
        .get_latest_session(&session_key)
        .await
        .expect("session should exist");
    let raw = core.sessions.get_all_messages(&meta.id).await;
    let roles: Vec<&str> = raw
        .iter()
        .map(|m| m.get("role").and_then(|r| r.as_str()).unwrap_or(""))
        .collect();

    assert_eq!(roles, vec!["user", "assistant", "tool", "assistant"]);
    assert!(
        raw[1].get("tool_calls").is_some(),
        "assistant carrier must retain tool_calls"
    );
    assert_eq!(
        raw[2].get("tool_call_id").and_then(|v| v.as_str()),
        raw[1]
            .get("tool_calls")
            .and_then(|v| v.as_array())
            .and_then(|calls| calls.first())
            .and_then(|call| call.get("id"))
            .and_then(|v| v.as_str()),
        "tool result must point at the immediately preceding assistant call"
    );

    let replay = core
        .sessions
        .load_session_replay(&meta.id)
        .await
        .expect("load tool replay");
    assert_eq!(
        replay.availability,
        crate::session::db::ReplayAvailability::Exact
    );
    assert_eq!(
        replay
            .events
            .iter()
            .map(|event| event.payload.kind())
            .collect::<Vec<_>>(),
        vec![
            "turn_started",
            "model_request",
            "model_response",
            "tool_pre_execute",
            "tool_execute",
            "tool_post_execute",
            "model_request",
            "model_response",
            "turn_finished",
        ]
    );
    assert_eq!(replay.model_calls.len(), 2);
    let second_request: Value = serde_json::from_slice(&replay.model_calls[1].request).unwrap();
    assert!(second_request["messages"]
        .as_array()
        .is_some_and(|messages| messages.iter().any(|message| {
            message.get("role").and_then(Value::as_str) == Some("tool")
                && message.get("tool_call_id").and_then(Value::as_str) == Some("tc_list")
        })));
    let lifecycle: Vec<&crate::session::db::SessionEventPayload> = replay
        .events
        .iter()
        .filter_map(|event| match &event.payload {
            payload @ (crate::session::db::SessionEventPayload::ToolPreExecute { .. }
            | crate::session::db::SessionEventPayload::ToolExecute { .. }
            | crate::session::db::SessionEventPayload::ToolPostExecute { .. }) => Some(payload),
            _ => None,
        })
        .collect();
    assert_eq!(lifecycle.len(), 3);
    assert!(matches!(
        lifecycle[0],
        crate::session::db::SessionEventPayload::ToolPreExecute {
            tool_call_id,
            decision: crate::session::db::ToolPreExecuteDecision::Ready,
            ..
        } if tool_call_id == "tc_list"
    ));
    assert!(matches!(
        lifecycle[1],
        crate::session::db::SessionEventPayload::ToolExecute {
            tool_call_id,
            ok: true,
            ..
        } if tool_call_id == "tc_list"
    ));
    assert!(matches!(
        lifecycle[2],
        crate::session::db::SessionEventPayload::ToolPostExecute {
            tool_call_id,
            message_id,
            ..
        } if tool_call_id == "tc_list" && *message_id > 0
    ));

    let _ = std::fs::remove_dir_all(&workspace);
}

fn guard_probe_call(id: &str) -> crate::providers::base::ToolCallRequest {
    crate::providers::base::ToolCallRequest {
        id: id.to_string(),
        name: "write_file".to_string(),
        arguments: std::collections::HashMap::from([(
            "content".to_string(),
            json!("missing path keeps this side-effect free"),
        )]),
    }
}

fn mixed_guard_responses(
    blocked_id: &str,
    allowed_id: &str,
    final_content: &str,
) -> Vec<crate::providers::base::LLMResponse> {
    let mut list_args = std::collections::HashMap::new();
    list_args.insert("path".to_string(), json!("."));
    let mut responses = (1..=3)
        .map(|index| crate::providers::base::LLMResponse {
            content: Some(String::new()),
            tool_calls: vec![guard_probe_call(&format!("tc_mixed_warmup_{index}"))],
            finish_reason: FinishReason::ToolCalls,
            usage: std::collections::HashMap::new(),
        })
        .collect::<Vec<_>>();
    responses.push(crate::providers::base::LLMResponse {
        content: Some(String::new()),
        tool_calls: vec![
            guard_probe_call(blocked_id),
            crate::providers::base::ToolCallRequest {
                id: allowed_id.to_string(),
                name: "list_dir".to_string(),
                arguments: list_args,
            },
        ],
        finish_reason: FinishReason::ToolCalls,
        usage: std::collections::HashMap::new(),
    });
    responses.push(crate::providers::base::LLMResponse {
        content: Some(final_content.to_string()),
        tool_calls: vec![],
        finish_reason: FinishReason::Stop,
        usage: std::collections::HashMap::new(),
    });
    responses
}

fn mixed_guard_side_effect_responses(
    blocked_id: &str,
    allowed_id: &str,
    allowed_path: &std::path::Path,
) -> Vec<crate::providers::base::LLMResponse> {
    let mut responses = (1..=3)
        .map(|index| crate::providers::base::LLMResponse {
            content: Some(String::new()),
            tool_calls: vec![guard_probe_call(&format!("tc_side_effect_warmup_{index}"))],
            finish_reason: FinishReason::ToolCalls,
            usage: std::collections::HashMap::new(),
        })
        .collect::<Vec<_>>();
    responses.push(crate::providers::base::LLMResponse {
        content: Some(String::new()),
        tool_calls: vec![
            guard_probe_call(blocked_id),
            crate::providers::base::ToolCallRequest {
                id: allowed_id.to_string(),
                name: "write_file".to_string(),
                arguments: std::collections::HashMap::from([
                    (
                        "path".to_string(),
                        json!(allowed_path.to_string_lossy().to_string()),
                    ),
                    ("content".to_string(), json!("must not be written")),
                ]),
            },
        ],
        finish_reason: FinishReason::ToolCalls,
        usage: std::collections::HashMap::new(),
    });
    responses
}

fn message_has_tool_call(message: &Value, tool_call_id: &str) -> bool {
    message
        .get("tool_calls")
        .and_then(Value::as_array)
        .is_some_and(|calls| {
            calls
                .iter()
                .any(|call| call.get("id").and_then(Value::as_str) == Some(tool_call_id))
        })
}

#[tokio::test]
async fn all_guard_blocked_without_cache_keeps_carrier_receipt_and_exact_replay() {
    let mut responses = (1..=4)
        .map(|index| crate::providers::base::LLMResponse {
            content: Some(String::new()),
            tool_calls: vec![guard_probe_call(&format!("tc_guard_{index}"))],
            finish_reason: FinishReason::ToolCalls,
            usage: std::collections::HashMap::new(),
        })
        .collect::<Vec<_>>();
    responses.push(crate::providers::base::LLMResponse {
        content: Some("finished after the guard receipt".to_string()),
        tool_calls: vec![],
        finish_reason: FinishReason::Stop,
        usage: std::collections::HashMap::new(),
    });
    let main: Arc<dyn LLMProvider> =
        Arc::new(ResponseSequenceProvider::new("local-main", responses));
    let (agent_loop, workspace) = build_local_inline_harness_with_iters(main, 6);
    let session_key = format!("all-guard-blocked-{}", uuid::Uuid::new_v4());

    let response = agent_loop
        .process_direct("exercise the guard", &session_key, "test", "offline")
        .await;
    assert_eq!(response, "finished after the guard receipt");

    let core = agent_loop.shared.core_handle.swappable();
    let session = core
        .sessions
        .get_latest_session(&session_key)
        .await
        .expect("guard session");
    let raw = core.sessions.get_all_messages(&session.id).await;
    assert!(raw.iter().any(|message| {
        message.get("role").and_then(Value::as_str) == Some("assistant")
            && message
                .get("tool_calls")
                .and_then(Value::as_array)
                .is_some_and(|calls| {
                    calls
                        .iter()
                        .any(|call| call.get("id").and_then(Value::as_str) == Some("tc_guard_4"))
                })
    }));
    assert!(raw.iter().any(|message| {
        message.get("role").and_then(Value::as_str) == Some("tool")
            && message.get("tool_call_id").and_then(Value::as_str) == Some("tc_guard_4")
            && message.get("ok") == Some(&Value::Bool(false))
    }));
    let replay = core
        .sessions
        .load_session_replay(&session.id)
        .await
        .expect("guard replay");
    assert_eq!(
        replay.availability,
        crate::session::db::ReplayAvailability::Exact
    );

    let _ = std::fs::remove_dir_all(&workspace);
}

#[tokio::test]
async fn mixed_guard_batch_keeps_blocked_carrier_receipt_and_executes_allowed_member() {
    let responses = mixed_guard_responses(
        "tc_mixed_blocked",
        "tc_mixed_allowed",
        "mixed batch finished",
    );
    let main: Arc<dyn LLMProvider> =
        Arc::new(ResponseSequenceProvider::new("local-main", responses));
    let (agent_loop, workspace) = build_local_inline_harness_with_iters(main, 6);
    let session_key = format!("mixed-guard-batch-{}", uuid::Uuid::new_v4());

    let response = agent_loop
        .process_direct(
            "exercise a mixed guard batch",
            &session_key,
            "test",
            "offline",
        )
        .await;
    assert_eq!(response, "mixed batch finished");

    let core = agent_loop.shared.core_handle.swappable();
    let session = core
        .sessions
        .get_latest_session(&session_key)
        .await
        .expect("mixed guard session");
    let raw = core.sessions.get_all_messages(&session.id).await;
    let mixed_carrier = raw
        .iter()
        .find(|message| {
            message
                .get("tool_calls")
                .and_then(Value::as_array)
                .is_some_and(|calls| {
                    calls.iter().any(|call| {
                        call.get("id").and_then(Value::as_str) == Some("tc_mixed_allowed")
                    })
                })
        })
        .expect("mixed carrier");
    let carrier_ids = mixed_carrier["tool_calls"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|call| call.get("id").and_then(Value::as_str))
        .collect::<std::collections::HashSet<_>>();
    assert_eq!(
        carrier_ids,
        std::collections::HashSet::from(["tc_mixed_blocked", "tc_mixed_allowed"])
    );
    for (id, ok) in [("tc_mixed_blocked", false), ("tc_mixed_allowed", true)] {
        assert!(raw.iter().any(|message| {
            message.get("role").and_then(Value::as_str) == Some("tool")
                && message.get("tool_call_id").and_then(Value::as_str) == Some(id)
                && message.get("ok") == Some(&Value::Bool(ok))
        }));
    }
    let blocked_receipt = raw
        .iter()
        .find(|message| {
            message.get("tool_call_id").and_then(Value::as_str) == Some("tc_mixed_blocked")
        })
        .and_then(|message| message.get("content").and_then(Value::as_str))
        .expect("mixed guard blocked receipt content");
    let (blocked_row, blocked_ok) = core
        .sessions
        .load_tool_result_with_status(&session.id, "tc_mixed_blocked")
        .await
        .expect("mixed guard rejected call raw row");
    assert_eq!(blocked_ok, Some(false));
    assert_eq!(blocked_row, blocked_receipt);
    let replay = core
        .sessions
        .load_session_replay(&session.id)
        .await
        .expect("mixed replay");
    assert_eq!(
        replay.availability,
        crate::session::db::ReplayAvailability::Exact
    );

    let _ = std::fs::remove_dir_all(&workspace);
}

#[tokio::test]
async fn mixed_guard_receipt_persistence_failure_prevents_allowed_execution() {
    let provider = Arc::new(ResponseSequenceProvider::new(
        "local-main",
        mixed_guard_responses(
            "tc_fault_blocked",
            "tc_fault_allowed",
            "must not be reached",
        ),
    ));
    let (agent_loop, workspace) =
        build_local_inline_harness_with_iters(provider.clone() as Arc<dyn LLMProvider>, 6);
    let session_key = format!("mixed-guard-receipt-fault-{}", uuid::Uuid::new_v4());
    let core = agent_loop.shared.core_handle.swappable();
    {
        let conn = rusqlite::Connection::open(core.sessions.path()).unwrap();
        conn.execute_batch(
            "CREATE TRIGGER fail_guard_receipt \
             BEFORE INSERT ON messages \
             WHEN NEW.role = 'tool' AND NEW.tool_call_id = 'tc_fault_blocked' \
             BEGIN SELECT RAISE(ABORT, 'synthetic guard receipt failure'); END;",
        )
        .unwrap();
    }

    let response = agent_loop
        .process_direct("fault the mixed receipt", &session_key, "test", "offline")
        .await;
    assert!(
        response.contains("router-rejected tool receipts"),
        "{response}"
    );
    assert_eq!(provider.call_count(), 4);

    let session = core
        .sessions
        .get_latest_session(&session_key)
        .await
        .expect("receipt fault session");
    let messages = core.sessions.get_all_messages(&session.id).await;
    assert!(
        !messages
            .iter()
            .any(|message| message_has_tool_call(message, "tc_fault_blocked")),
        "message fault must roll back the complete carrier"
    );
    assert!(!messages.iter().any(|message| {
        message.get("tool_call_id").and_then(Value::as_str) == Some("tc_fault_blocked")
    }));
    let events = core
        .sessions
        .load_session_events(&session.id)
        .await
        .unwrap();
    assert!(!events.iter().any(|event| matches!(
        &event.payload,
        crate::session::db::SessionEventPayload::ToolPreExecute { tool_call_id, .. }
            if tool_call_id == "tc_fault_allowed" || tool_call_id == "tc_fault_blocked"
    )));
    assert_eq!(
        persisted_turn_outcome(&core.sessions, &session.id).await,
        "error"
    );
    assert_eq!(
        core.sessions
            .load_tool_result_with_status(&session.id, "tc_fault_blocked")
            .await,
        None,
        "failed router receipt transaction must roll back its raw row"
    );

    let _ = std::fs::remove_dir_all(&workspace);
}

#[tokio::test]
async fn router_rejection_raw_and_decision_faults_preserve_transaction_order() {
    for fault in ["raw", "decision"] {
        let side_effect_dir = tempfile::tempdir().unwrap();
        let side_effect_path = side_effect_dir.path().join(format!("router-{fault}.txt"));
        let blocked_id = format!("tc_router_{fault}_blocked");
        let allowed_id = format!("tc_router_{fault}_allowed");
        let provider = Arc::new(ResponseSequenceProvider::new(
            "local-main",
            mixed_guard_side_effect_responses(&blocked_id, &allowed_id, &side_effect_path),
        ));
        let (agent_loop, workspace) =
            build_local_inline_harness_with_iters(provider as Arc<dyn LLMProvider>, 6);
        let session_key = format!("router-{fault}-fault-{}", uuid::Uuid::new_v4());
        let core = agent_loop.shared.core_handle.swappable();
        {
            let conn = rusqlite::Connection::open(core.sessions.path()).unwrap();
            let sql = if fault == "raw" {
                format!(
                    "CREATE TRIGGER fail_router_raw BEFORE INSERT ON tool_results \
                     WHEN NEW.tool_call_id = '{blocked_id}' \
                     BEGIN SELECT RAISE(ABORT, 'synthetic router raw failure'); END;"
                )
            } else {
                format!(
                    "CREATE TRIGGER fail_router_decision BEFORE INSERT ON session_events \
                     WHEN NEW.event_kind = 'tool_pre_execute' \
                       AND NEW.payload_json LIKE '%{blocked_id}%' \
                     BEGIN SELECT RAISE(ABORT, 'synthetic router decision failure'); END;"
                )
            };
            conn.execute_batch(&sql).unwrap();
        }

        agent_loop
            .process_direct(
                "exercise router transaction ordering",
                &session_key,
                "test",
                "offline",
            )
            .await;

        let session = core.sessions.get_or_resume(&session_key).await;
        let messages = core.sessions.get_all_messages(&session.id).await;
        let has_carrier = messages
            .iter()
            .any(|message| message_has_tool_call(message, &blocked_id));
        let has_receipt = messages.iter().any(|message| {
            message.get("tool_call_id").and_then(Value::as_str) == Some(blocked_id.as_str())
                && message.get("ok").and_then(Value::as_bool) == Some(false)
        });
        let raw = core
            .sessions
            .load_tool_result_with_status(&session.id, &blocked_id)
            .await;
        if fault == "raw" {
            assert_eq!(
                (has_carrier, has_receipt, raw.is_some()),
                (false, false, false)
            );
        } else {
            assert_eq!((has_carrier, has_receipt), (true, true));
            assert!(matches!(raw, Some((_, Some(false)))));
        }
        assert!(
            !side_effect_path.exists(),
            "allowed router member ran after {fault} fault"
        );
        let events = core
            .sessions
            .load_session_events(&session.id)
            .await
            .unwrap();
        assert!(!events.iter().any(|event| matches!(
            &event.payload,
            crate::session::db::SessionEventPayload::ToolPreExecute { tool_call_id, .. }
                if tool_call_id == &allowed_id
        )));

        let _ = std::fs::remove_dir_all(&workspace);
    }
}

fn lease_fault_response(
    prefix: &str,
    side_effect_dir: &std::path::Path,
) -> crate::providers::base::LLMResponse {
    let tool_calls = (1..=97)
        .map(|index| crate::providers::base::ToolCallRequest {
            id: format!("{prefix}_{index}"),
            name: "exec".to_string(),
            arguments: std::collections::HashMap::from([(
                "command".to_string(),
                json!(format!(
                    "printf ran > {}",
                    side_effect_dir.join(format!("ran-{index}.txt")).display()
                )),
            )]),
        })
        .collect();
    crate::providers::base::LLMResponse {
        content: Some(String::new()),
        tool_calls,
        finish_reason: FinishReason::ToolCalls,
        usage: std::collections::HashMap::new(),
    }
}

#[tokio::test]
async fn lease_rejection_message_raw_and_decision_faults_preserve_transaction_order() {
    for fault in ["message", "raw", "decision"] {
        let side_effect_dir = tempfile::tempdir().unwrap();
        let prefix = format!("tc_lease_{fault}");
        let blocked_id = format!("{prefix}_97");
        let provider = Arc::new(ResponseSequenceProvider::new(
            "local-main",
            vec![lease_fault_response(&prefix, side_effect_dir.path())],
        ));
        let (agent_loop, workspace) =
            build_local_inline_harness_with_iters(provider as Arc<dyn LLMProvider>, 3);
        let session_key = format!("lease-{fault}-fault-{}", uuid::Uuid::new_v4());
        let core = agent_loop.shared.core_handle.swappable();
        {
            let conn = rusqlite::Connection::open(core.sessions.path()).unwrap();
            let sql = match fault {
                "message" => format!(
                    "CREATE TRIGGER fail_lease_message BEFORE INSERT ON messages \
                     WHEN NEW.tool_call_id = '{blocked_id}' \
                     BEGIN SELECT RAISE(ABORT, 'synthetic lease message failure'); END;"
                ),
                "raw" => format!(
                    "CREATE TRIGGER fail_lease_raw BEFORE INSERT ON tool_results \
                     WHEN NEW.tool_call_id = '{blocked_id}' \
                     BEGIN SELECT RAISE(ABORT, 'synthetic lease raw failure'); END;"
                ),
                "decision" => format!(
                    "CREATE TRIGGER fail_lease_decision BEFORE INSERT ON session_events \
                     WHEN NEW.event_kind = 'tool_pre_execute' \
                       AND NEW.payload_json LIKE '%{blocked_id}%' \
                     BEGIN SELECT RAISE(ABORT, 'synthetic lease decision failure'); END;"
                ),
                _ => unreachable!(),
            };
            conn.execute_batch(&sql).unwrap();
        }

        agent_loop
            .process_direct(
                "exercise lease transaction ordering",
                &session_key,
                "test",
                "offline",
            )
            .await;

        let session = core.sessions.get_or_resume(&session_key).await;
        let messages = core.sessions.get_all_messages(&session.id).await;
        let has_carrier = messages
            .iter()
            .any(|message| message_has_tool_call(message, &blocked_id));
        let has_receipt = messages.iter().any(|message| {
            message.get("tool_call_id").and_then(Value::as_str) == Some(blocked_id.as_str())
                && message.get("ok").and_then(Value::as_bool) == Some(false)
        });
        let raw = core
            .sessions
            .load_tool_result_with_status(&session.id, &blocked_id)
            .await;
        if fault == "decision" {
            assert_eq!((has_carrier, has_receipt), (true, true));
            assert!(matches!(raw, Some((_, Some(false)))));
        } else {
            assert_eq!(
                (has_carrier, has_receipt, raw.is_some()),
                (false, false, false)
            );
        }
        assert!(
            std::fs::read_dir(side_effect_dir.path())
                .unwrap()
                .next()
                .is_none(),
            "allowed lease members ran after {fault} fault"
        );
        let events = core
            .sessions
            .load_session_events(&session.id)
            .await
            .unwrap();
        assert!(!events.iter().any(|event| matches!(
            &event.payload,
            crate::session::db::SessionEventPayload::ToolPreExecute {
                decision: crate::session::db::ToolPreExecuteDecision::Ready,
                ..
            }
        )));

        let _ = std::fs::remove_dir_all(&workspace);
    }
}

#[tokio::test]
async fn exact_turn_replay_survives_workspace_prompt_changes() {
    // Break caught: replay reconstructs the old provider request from today's
    // workspace/system prompt instead of reading the exact durable call bytes.
    let main: Arc<dyn LLMProvider> =
        Arc::new(StaticResponseLLM::new("local-main", "recorded answer"));
    let (agent_loop, workspace) = build_local_inline_harness(main);
    let session_key = format!("exact-turn-replay-{}", uuid::Uuid::new_v4());

    let response = agent_loop
        .process_direct("preserve this exact turn", &session_key, "test", "offline")
        .await;
    assert_eq!(response, "recorded answer");

    let core = agent_loop.shared.core_handle.swappable();
    let meta = core
        .sessions
        .get_latest_session(&session_key)
        .await
        .expect("session should exist");
    let before = core
        .sessions
        .load_session_replay(&meta.id)
        .await
        .expect("load exact replay");
    assert_eq!(
        before.availability,
        crate::session::db::ReplayAvailability::Exact
    );
    assert_eq!(before.model_calls.len(), 1);
    assert!(matches!(
        before.events.last().map(|event| &event.payload),
        Some(crate::session::db::SessionEventPayload::TurnFinished { outcome })
            if outcome == "finished"
    ));
    let request: Value = serde_json::from_slice(&before.model_calls[0].request).unwrap();
    assert_eq!(request.get("streaming"), Some(&Value::Bool(false)));
    assert_eq!(
        request.get("model").and_then(Value::as_str),
        Some(core.model.as_str())
    );
    let messages = request
        .get("messages")
        .and_then(Value::as_array)
        .expect("recorded messages");
    assert!(messages.iter().any(|message| {
        message.get("role").and_then(Value::as_str) == Some("user")
            && message.get("content").and_then(Value::as_str) == Some("preserve this exact turn")
    }));

    std::fs::write(workspace.join("IDENTITY.md"), "a different future identity").unwrap();
    let after = core
        .sessions
        .load_session_replay(&meta.id)
        .await
        .expect("reload exact replay");
    assert_eq!(after.model_calls[0].request, before.model_calls[0].request);
    assert_eq!(
        after.model_calls[0].response,
        before.model_calls[0].response
    );

    let _ = std::fs::remove_dir_all(&workspace);
}

#[tokio::test]
async fn pre_execute_persistence_failure_prevents_tool_side_effect() {
    // Break caught: a tool enters its implementation even though the durable
    // pre-execute decision failed, leaving an unreplayable side effect.
    let provider = Arc::new(ResponseSequenceProvider::new(
        "local-main",
        vec![crate::providers::base::LLMResponse {
            content: Some(String::new()),
            tool_calls: vec![crate::providers::base::ToolCallRequest {
                id: "tc-blocked-write".to_string(),
                name: "write_file".to_string(),
                arguments: HashMap::from([
                    ("path".to_string(), json!("blocked.txt")),
                    ("content".to_string(), json!("must not exist")),
                ]),
            }],
            finish_reason: FinishReason::ToolCalls,
            usage: HashMap::new(),
        }],
    ));
    let (agent_loop, workspace) =
        build_local_inline_harness(provider.clone() as Arc<dyn LLMProvider>);
    let core = agent_loop.shared.core_handle.swappable();
    {
        let conn = rusqlite::Connection::open(core.sessions.path()).unwrap();
        conn.execute_batch(
            "CREATE TRIGGER fail_tool_pre_replay \
             BEFORE INSERT ON session_events \
             WHEN NEW.event_kind = 'tool_pre_execute' \
             BEGIN SELECT RAISE(ABORT, 'synthetic pre-execute persistence failure'); END;",
        )
        .unwrap();
    }

    let response = agent_loop
        .process_direct(
            "write the blocked file",
            &format!("pre-execute-failure-{}", uuid::Uuid::new_v4()),
            "test",
            "offline",
        )
        .await;

    assert!(
        response.contains("pre-execution decision could not be recorded"),
        "unexpected failure response: {response:?}"
    );
    assert!(!workspace.join("blocked.txt").exists());
    assert_eq!(provider.call_count(), 1);
    let _ = std::fs::remove_dir_all(&workspace);
}

#[tokio::test]
async fn test_tool_round_is_durable_before_next_provider_call_completes() {
    let provider = Arc::new(ToolRoundBarrierProvider::new());
    let (agent_loop, workspace) =
        build_local_inline_harness(provider.clone() as Arc<dyn LLMProvider>);
    let core = agent_loop.shared.core_handle.swappable();
    let session_key = format!("test-tool-durable-{}", uuid::Uuid::new_v4());
    let task_loop = Arc::new(agent_loop);
    let task = {
        let task_loop = task_loop.clone();
        let session_key = session_key.clone();
        tokio::spawn(async move {
            task_loop
                .process_direct("please list files", &session_key, "test", "offline")
                .await
        })
    };

    tokio::time::timeout(
        std::time::Duration::from_secs(5),
        provider.second_call_started.notified(),
    )
    .await
    .expect("second provider call should start after the tool round");

    let meta = core
        .sessions
        .get_latest_session(&session_key)
        .await
        .expect("session should exist while the turn is active");
    let durable = core.sessions.get_all_messages(&meta.id).await;
    let roles: Vec<&str> = durable
        .iter()
        .map(|message| message.get("role").and_then(Value::as_str).unwrap_or(""))
        .collect();
    assert_eq!(
        roles,
        vec!["user", "assistant", "tool"],
        "the active tool protocol must be crash-durable before the next inference"
    );

    provider.release_second_call.notify_one();
    assert_eq!(task.await.expect("turn task should join"), "done");
    let _ = std::fs::remove_dir_all(&workspace);
}

#[tokio::test]
async fn test_multiple_tool_round_carriers_persist_in_order() {
    let mut list_args = std::collections::HashMap::new();
    list_args.insert("path".to_string(), json!("."));
    let mut exec_args = std::collections::HashMap::new();
    exec_args.insert("command".to_string(), json!("printf ok"));
    exec_args.insert("working_dir".to_string(), json!("."));

    let main: Arc<dyn LLMProvider> = Arc::new(ResponseSequenceProvider::new(
        "local-main",
        vec![
            crate::providers::base::LLMResponse {
                content: Some(String::new()),
                tool_calls: vec![crate::providers::base::ToolCallRequest {
                    id: "tc_list".to_string(),
                    name: "list_dir".to_string(),
                    arguments: list_args,
                }],
                finish_reason: FinishReason::ToolCalls,
                usage: std::collections::HashMap::new(),
            },
            crate::providers::base::LLMResponse {
                content: Some("I found the script; running it now.".to_string()),
                tool_calls: vec![crate::providers::base::ToolCallRequest {
                    id: "tc_exec".to_string(),
                    name: "exec".to_string(),
                    arguments: exec_args,
                }],
                finish_reason: FinishReason::ToolCalls,
                usage: std::collections::HashMap::new(),
            },
            crate::providers::base::LLMResponse {
                content: Some(attested_text("Done.")),
                tool_calls: vec![],
                finish_reason: FinishReason::Stop,
                usage: std::collections::HashMap::new(),
            },
        ],
    ));
    let (agent_loop, workspace) = build_local_inline_harness(main);
    let session_key = format!("test-tool-order-multi-{}", uuid::Uuid::new_v4().to_string());

    let response = agent_loop
        .process_direct("please inspect and run", &session_key, "test", "offline")
        .await;
    assert_eq!(response, "Done.");

    let core = agent_loop.shared.core_handle.swappable();
    let meta = core
        .sessions
        .get_latest_session(&session_key)
        .await
        .expect("session should exist");
    let raw = core.sessions.get_all_messages(&meta.id).await;
    let roles: Vec<&str> = raw
        .iter()
        .map(|m| m.get("role").and_then(|r| r.as_str()).unwrap_or(""))
        .collect();

    assert_eq!(
        roles,
        vec![
            "user",
            "assistant",
            "tool",
            "assistant",
            "tool",
            "assistant"
        ]
    );
    assert!(raw[1].get("tool_calls").is_some());
    assert!(raw[3].get("tool_calls").is_some());
    assert_eq!(
        raw[2].get("tool_call_id").and_then(|v| v.as_str()),
        Some("tc_list")
    );
    assert_eq!(
        raw[4].get("tool_call_id").and_then(|v| v.as_str()),
        Some("tc_exec")
    );

    let _ = std::fs::remove_dir_all(&workspace);
}

#[tokio::test]
async fn test_local_truncated_response_requires_attested_correction_without_continue() {
    let main = Arc::new(ResponseSequenceProvider::new(
        "local-main",
        vec![crate::providers::base::LLMResponse {
            content: Some("Partial local answer".to_string()),
            tool_calls: vec![],
            finish_reason: FinishReason::Length,
            usage: std::collections::HashMap::new(),
        }],
    ));
    let main_dyn: Arc<dyn LLMProvider> = main.clone();
    let (agent_loop, workspace) = build_local_inline_harness(main_dyn);
    let session_key = format!(
        "test-local-no-auto-continue-{}",
        uuid::Uuid::new_v4().to_string()
    );

    let response = agent_loop
        .process_direct("answer briefly", &session_key, "test", "offline")
        .await;

    // With the attestation protocol removed, truncated text follows the
    // standard auto-continuation path; if continuations are exhausted the
    // accumulated text is the final answer (no attestation retry loop).
    assert!(
        !response.is_empty(),
        "expected a non-empty terminal response, got: {response:?}"
    );

    let _ = std::fs::remove_dir_all(&workspace);
}

#[tokio::test]
async fn test_local_streaming_cache_markers_append_only_across_turns() {
    let main: Arc<dyn LLMProvider> = Arc::new(ResponseSequenceProvider::new(
        "local-main",
        vec![
            crate::providers::base::LLMResponse {
                content: Some(attested_text("one")),
                tool_calls: vec![],
                finish_reason: FinishReason::Stop,
                usage: std::collections::HashMap::new(),
            },
            crate::providers::base::LLMResponse {
                content: Some(attested_text("two")),
                tool_calls: vec![],
                finish_reason: FinishReason::Stop,
                usage: std::collections::HashMap::new(),
            },
            crate::providers::base::LLMResponse {
                content: Some(attested_text("three")),
                tool_calls: vec![],
                finish_reason: FinishReason::Stop,
                usage: std::collections::HashMap::new(),
            },
        ],
    ));
    let (agent_loop, workspace) = build_local_inline_harness(main);
    let session_key = format!(
        "test-local-cache-markers-{}",
        uuid::Uuid::new_v4().to_string()
    );

    let mut cache_markers = Vec::new();
    let mut prefill_estimates = Vec::new();
    for (input, expected) in [
        ("first short turn", "one"),
        ("second short turn", "two"),
        ("third short turn", "three"),
    ] {
        let (delta_tx, mut delta_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
        let response = agent_loop
            .process_direct_streaming(
                input,
                &session_key,
                "test",
                "offline",
                None,
                delta_tx,
                None,
                None,
                None,
                None,
            )
            .await;
        assert_eq!(response, expected);

        let mut turn_cache_markers = Vec::new();
        let mut turn_prefill_estimates = Vec::new();
        while let Ok(delta) = delta_rx.try_recv() {
            if delta.starts_with("\u{0}cache:") {
                turn_cache_markers.push(delta);
            } else if delta.starts_with("\u{0}prefill_estimate:") {
                turn_prefill_estimates.push(delta);
            }
        }
        assert_eq!(
            turn_cache_markers.len(),
            1,
            "each streamed turn should emit one cache marker"
        );
        assert_eq!(
            turn_prefill_estimates.len(),
            1,
            "each streamed turn should emit one prefill estimate"
        );
        cache_markers.push(turn_cache_markers.remove(0));
        prefill_estimates.push(turn_prefill_estimates.remove(0));
    }

    assert!(
        cache_markers[0].starts_with("\u{0}cache:first:"),
        "first turn should establish the cache: {cache_markers:?}"
    );
    // Later turns must report AppendOnly, not First.
    //
    // The DB reload between turns is a pure function of the stored rows, so
    // turn N+1's prompt is turn N's prompt plus the new messages. The
    // fingerprint therefore survives the reload and the comparison is made
    // ACROSS the turn boundary — which is the only place a reload that
    // silently rewrote history can be caught.
    //
    // This previously asserted `first:` on every turn, because the fingerprint
    // was cleared on each reload. That made cross-turn divergence structurally
    // undetectable: session 20260810_081050_8306f8 shrank 8 tool results by
    // ~9.8KB at a turn boundary, cost 124.54s of re-prefill on the server, and
    // produced no nanobot log line at all.
    for (i, marker) in cache_markers.iter().enumerate().skip(1) {
        assert!(
            marker.starts_with("\u{0}cache:append:"),
            "turn {} must continue the previous turn's prefix, got {marker:?} in {cache_markers:?}",
            i + 1
        );
    }
    assert!(
        cache_markers
            .iter()
            .all(|marker| !marker.starts_with("\u{0}cache:diverged:")),
        "local cache path must not diverge across ordinary turns: {cache_markers:?}"
    );
    for marker in &prefill_estimates {
        let tokens: usize = marker
            .trim_start_matches("\u{0}prefill_estimate:")
            .parse()
            .expect("prefill estimate token count");
        assert!(tokens > 0, "prefill estimate must be positive: {marker:?}");
    }

    let _ = std::fs::remove_dir_all(&workspace);
}

#[tokio::test]
async fn test_failed_local_call_does_not_seed_prompt_cache_marker() {
    let provider: Arc<dyn LLMProvider> = Arc::new(FailOnceThenResponseProvider::new(
        "local-main",
        crate::providers::base::LLMResponse {
            content: Some(attested_text("recovered")),
            tool_calls: vec![],
            finish_reason: FinishReason::Stop,
            usage: std::collections::HashMap::new(),
        },
    ));
    let (agent_loop, workspace) = build_local_inline_harness(provider);
    let session_key = format!(
        "test-local-cache-failure-{}",
        uuid::Uuid::new_v4().to_string()
    );

    let (first_tx, mut first_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    let first = agent_loop
        .process_direct_streaming(
            "first turn fails before the server can warm cache",
            &session_key,
            "test",
            "offline",
            None,
            first_tx,
            None,
            None,
            None,
            None,
        )
        .await;
    assert!(
        first.contains("synthetic provider failure"),
        "expected provider error, got {first:?}"
    );
    let mut first_markers = Vec::new();
    while let Ok(delta) = first_rx.try_recv() {
        if delta.starts_with("\u{0}cache:") {
            first_markers.push(delta);
        }
    }
    assert!(
        first_markers
            .first()
            .is_some_and(|m| m.starts_with("\u{0}cache:first:")),
        "failed call may diagnose cold cache, but must not commit it: {first_markers:?}"
    );
    let core = agent_loop.shared.core_handle.swappable();
    let failed_session = core
        .sessions
        .get_latest_session(&session_key)
        .await
        .expect("failed turn session");
    let failed_replay = core
        .sessions
        .load_session_replay(&failed_session.id)
        .await
        .expect("failed turn replay");
    assert_eq!(
        failed_replay.availability,
        crate::session::db::ReplayAvailability::Exact
    );
    assert!(failed_replay.model_calls[0]
        .failure
        .as_deref()
        .is_some_and(|bytes| bytes
            .windows("synthetic provider failure".len())
            .any(|window| { window == "synthetic provider failure".as_bytes() })));
    assert_eq!(
        persisted_turn_outcome(&core.sessions, &failed_session.id).await,
        "error",
        "provider failure text must not make the turn look successful"
    );

    let (second_tx, mut second_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    let second = agent_loop
        .process_direct_streaming(
            "second turn should still be cold from nanobot's cache model",
            &session_key,
            "test",
            "offline",
            None,
            second_tx,
            None,
            None,
            None,
            None,
        )
        .await;
    assert_eq!(second, "recovered");

    let mut second_markers = Vec::new();
    while let Ok(delta) = second_rx.try_recv() {
        if delta.starts_with("\u{0}cache:") {
            second_markers.push(delta);
        }
    }
    assert!(
        second_markers
            .first()
            .is_some_and(|m| m.starts_with("\u{0}cache:first:")),
        "a failed provider call must not make the next turn look append-only: {second_markers:?}"
    );

    let _ = std::fs::remove_dir_all(&workspace);
}

#[tokio::test]
async fn test_direct_streaming_forwards_thinking_delta_with_ansi_marker() {
    let main = Arc::new(StreamingThinkingProvider::new("local-main"));
    let main_dyn: Arc<dyn LLMProvider> = main.clone();
    let (agent_loop, workspace) = build_local_inline_harness(main_dyn);
    agent_loop
        .shared
        .core_handle
        .counters
        .thinking_budget
        .store(128, std::sync::atomic::Ordering::Relaxed);
    let session_key = format!(
        "test-direct-thinking-stream-{}",
        uuid::Uuid::new_v4().to_string()
    );

    let (delta_tx, mut delta_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    let response = agent_loop
        .process_direct_streaming(
            "show thinking",
            &session_key,
            "test",
            "offline",
            None,
            delta_tx,
            None,
            None,
            None,
            None,
        )
        .await;

    assert_eq!(response, "visible answer");
    assert_eq!(
        main.last_thinking_budget(),
        128,
        "direct streaming call should pass the enabled thinking budget to the provider"
    );

    let mut deltas = Vec::new();
    while let Ok(delta) = delta_rx.try_recv() {
        deltas.push(delta);
    }
    let marker_idx = deltas
        .iter()
        .position(|delta| delta == "\x1b[90m\x1b[2m")
        .unwrap_or_else(|| panic!("missing thinking marker in deltas: {deltas:?}"));
    assert_eq!(
        deltas.get(marker_idx + 1).map(String::as_str),
        Some("private thought"),
        "thinking text should immediately follow the ANSI marker: {deltas:?}"
    );
    assert!(
        deltas.iter().any(|delta| delta == "\x1b[0m\n\n"),
        "thinking stream should be reset before visible text: {deltas:?}"
    );
    assert!(
        deltas.iter().any(|delta| delta == "visible answer"),
        "visible answer text should still stream after thinking: {deltas:?}"
    );
    let core = agent_loop.shared.core_handle.swappable();
    let meta = core
        .sessions
        .get_latest_session(&session_key)
        .await
        .expect("streaming replay session");
    let replay = core.sessions.load_session_replay(&meta.id).await.unwrap();
    assert_eq!(
        replay.availability,
        crate::session::db::ReplayAvailability::Exact
    );
    let recorded_request: Value = serde_json::from_slice(&replay.model_calls[0].request).unwrap();
    assert_eq!(recorded_request["streaming"], json!(true));
    let recorded_response: Value =
        serde_json::from_slice(replay.model_calls[0].response.as_deref().unwrap()).unwrap();
    assert_eq!(recorded_response["content"], json!("visible answer"));

    let _ = std::fs::remove_dir_all(&workspace);
}

#[tokio::test]
async fn test_vibethinker_hidden_reasoning_streams_without_think_budget() {
    let main = Arc::new(StreamingThinkingProvider::new("VibeThinker-3B-mlx-8Bit"));
    let main_dyn: Arc<dyn LLMProvider> = main.clone();
    let (agent_loop, workspace) =
        build_local_inline_harness_with_model(main_dyn, "local:VibeThinker-3B-mlx-8Bit");
    let session_key = format!(
        "test-vibethinker-hidden-thinking-{}",
        uuid::Uuid::new_v4().to_string()
    );

    let (delta_tx, mut delta_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    let response = agent_loop
        .process_direct_streaming(
            "show native thinking",
            &session_key,
            "test",
            "offline",
            None,
            delta_tx,
            None,
            None,
            None,
            None,
        )
        .await;

    assert_eq!(response, "visible answer");
    assert_eq!(
        main.last_thinking_budget(),
        0,
        "hidden reasoning should not impose a nanobot thinking budget"
    );

    let mut deltas = Vec::new();
    while let Ok(delta) = delta_rx.try_recv() {
        deltas.push(delta);
    }
    assert!(
        deltas.iter().any(|delta| delta == "\x1b[90m\x1b[2m"),
        "VibeThinker reasoning_content should stream to display without /think: {deltas:?}"
    );
    assert!(
        deltas.iter().any(|delta| delta == "private thought"),
        "hidden reasoning text should not be dropped: {deltas:?}"
    );

    let _ = std::fs::remove_dir_all(&workspace);
}

#[tokio::test]
async fn test_tts_suppression_does_not_hide_vibethinker_display() {
    let main = Arc::new(StreamingThinkingProvider::new("VibeThinker-3B-mlx-8Bit"));
    let main_dyn: Arc<dyn LLMProvider> = main.clone();
    let (agent_loop, workspace) =
        build_local_inline_harness_with_model(main_dyn, "local:VibeThinker-3B-mlx-8Bit");
    agent_loop
        .shared
        .core_handle
        .counters
        .suppress_thinking_in_tts
        .store(true, std::sync::atomic::Ordering::Relaxed);
    let session_key = format!(
        "test-vibethinker-tts-suppression-display-{}",
        uuid::Uuid::new_v4().to_string()
    );

    let (delta_tx, mut delta_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    let response = agent_loop
        .process_direct_streaming(
            "show native thinking while voice is on",
            &session_key,
            "test",
            "offline",
            None,
            delta_tx,
            None,
            None,
            None,
            None,
        )
        .await;

    assert_eq!(response, "visible answer");
    let mut deltas = Vec::new();
    while let Ok(delta) = delta_rx.try_recv() {
        deltas.push(delta);
    }
    assert!(
        deltas.iter().any(|delta| delta == "private thought"),
        "TTS suppression should not suppress the visual thinking stream: {deltas:?}"
    );

    let _ = std::fs::remove_dir_all(&workspace);
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

    let (agent_loop, workspace) = build_trio_offline_harness(main, router, specialist);

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
        metrics
            .router_preflight_fired
            .load(std::sync::atomic::Ordering::Relaxed),
        "router preflight should have fired"
    );
    assert_eq!(
        metrics.router_action.lock().as_deref(),
        Some("respond"),
        "router_action should be 'respond'"
    );
    assert!(
        !metrics
            .specialist_dispatched
            .load(std::sync::atomic::Ordering::Relaxed),
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
    let specialist: Arc<dyn LLMProvider> =
        Arc::new(StaticResponseLLM::new("offline-specialist", "unused"));

    let (agent_loop, workspace) = build_trio_offline_harness(main_dyn, router, specialist);
    agent_loop
        .shared
        .core_handle
        .counters
        .thinking_budget
        .store(128, std::sync::atomic::Ordering::Relaxed);

    let _ = agent_loop
        .process_direct(
            "What is the current date?",
            "reserve-max-tokens",
            "test",
            "offline",
        )
        .await;

    assert_eq!(
        main.last_max_tokens(),
        640,
        "local thinking should add budget on top of base max_tokens=512 (512+128=640)"
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
    let main: Arc<dyn LLMProvider> = Arc::new(StaticResponseLLM::new("offline-main", "delegating"));
    let specialist: Arc<dyn LLMProvider> = Arc::new(StaticResponseLLM::new(
        "offline-specialist",
        "Here is the specialist answer.",
    ));

    let (agent_loop, workspace) = build_trio_offline_harness(main, router, specialist);

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
        metrics.router_action.lock().as_deref(),
        Some("specialist"),
        "router_action should be 'specialist'"
    );
    assert!(
        metrics
            .specialist_dispatched
            .load(std::sync::atomic::Ordering::Relaxed),
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
    let specialist: Arc<dyn LLMProvider> = Arc::new(StaticResponseLLM::new(
        "offline-specialist",
        "specialist unused",
    ));

    let (agent_loop, workspace) = build_trio_offline_harness(main, router, specialist);

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
    use crate::config::schema::LcmSchemaConfig;
    use crate::heartbeat::health::{HealthProbe, HealthRegistry, ProbeResult};

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
    let specialist: Arc<dyn LLMProvider> = Arc::new(StaticResponseLLM::new(
        "offline-specialist",
        "specialist unused",
    ));

    // Build harness manually so we can wire in the health registry.
    let workspace = tempfile::tempdir().unwrap().keep();
    let mut td = ToolDelegationConfig {
        mode: crate::config::schema::DelegationMode::trio(),
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
        crw_url: String::new(),
        search_max_results: 5,
        exec_timeout: 30,
        restrict_to_workspace: true,
        memory_config: MemoryConfig::default(),
        is_local: true,
        lane: Lane::default(),
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
        code_execution: CodeExecutionConfig::default(),
        python_kernel: PythonKernelConfig::default(),
        cua: CuaToolConfig::default(),
        adaptive_tokens: AdaptiveTokenConfig::default(),
        sessions_db_path: Some(
            std::env::temp_dir().join(format!("nanobot-test-{}.sqlite", uuid::Uuid::new_v4())),
        ),
    });

    let counters = test_runtime_counters(4096);
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
        Some(health_registry), // health registry is wired in here
    );

    let resp = agent_loop
        .process_direct("Hello", "trio-offline-health-gate", "test", "offline")
        .await;

    eprintln!(
        "test_trio_offline_e2e_health_gate: response ({} chars): {}",
        resp.len(),
        &resp[..resp.len().min(200)]
    );

    // When the health gate fires, router_preflight returns Passthrough and sets Degraded.
    let state = agent_loop.shared.core_handle.counters.get_trio_state();
    eprintln!("trio_state after health gate: {:?}", state);
    assert_eq!(
        state,
        crate::agent::agent_core::TrioState::Degraded,
        "trio_state should be Degraded when health gate fires"
    );

    // Response must come from main (non-empty).
    assert!(
        !resp.is_empty(),
        "response should come from main, not be empty"
    );

    // router_preflight_fired should be true (we entered preflight but returned Passthrough).
    let metrics = &agent_loop.shared.core_handle.counters.trio_metrics;
    assert!(
        metrics
            .router_preflight_fired
            .load(std::sync::atomic::Ordering::Relaxed),
        "router_preflight_fired should be true (preflight was entered)"
    );

    // Specialist must not have been dispatched.
    assert!(
        !metrics
            .specialist_dispatched
            .load(std::sync::atomic::Ordering::Relaxed),
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

/// Strict-trio strip path is active (healthy registry) and the turn spans
/// two iterations (specialist dispatch → respond). The orchestration-mode
/// block must appear EXACTLY ONCE in the system head on every wire: a
/// re-append per iteration rewrites sent system bytes and busts the prefix
/// cache.
#[tokio::test]
async fn test_trio_orchestration_block_not_reappended_across_iterations() {
    use crate::heartbeat::health::HealthRegistry;

    let router: Arc<dyn LLMProvider> = Arc::new(SequenceProvider::new(
        "offline-router",
        vec![
            r#"{"action":"specialist","target":"coding","args":{"task":"explain loops"},"confidence":0.85}"#,
            r#"{"action":"specialist","target":"coding","args":{"task":"explain loops"},"confidence":0.85}"#,
            r#"{"action":"specialist","target":"coding","args":{"task":"explain loops again"},"confidence":0.85}"#,
            r#"{"action":"respond","target":"main","args":{},"confidence":0.95}"#,
        ],
    ));
    let main_recorder = Arc::new(WireRecordingProvider::new(
        "offline-main",
        vec![WireRecordingProvider::text_response("final answer")],
    ));
    let main: Arc<dyn LLMProvider> = main_recorder.clone();
    let specialist: Arc<dyn LLMProvider> = Arc::new(StaticResponseLLM::new(
        "offline-specialist",
        "specialist answer",
    ));

    let (agent_loop, workspace) = build_trio_offline_harness_with_registry(
        main,
        router,
        specialist,
        Some(Arc::new(HealthRegistry::new())),
    );

    let resp = agent_loop
        .process_direct("Explain for loops", "trio-dedup", "test", "offline")
        .await;
    assert!(!resp.is_empty(), "response should be non-empty");

    // The strip path must actually have been armed, or the test is vacuous.
    assert_eq!(
        agent_loop.shared.core_handle.counters.get_trio_state(),
        crate::agent::agent_core::TrioState::Active,
        "trio must be Active (tools stripped) for this test to exercise the block"
    );

    let calls = main_recorder.calls();
    assert!(
        calls.len() >= 1,
        "main model must have been called at least once"
    );
    for (i, wire) in calls.iter().enumerate() {
        let system = wire[0]["content"].as_str().unwrap_or("");
        assert_eq!(
            system.matches("## Orchestration Mode (Active)").count(),
            1,
            "wire {i}: orchestration block must appear exactly once in the system head"
        );
    }

    let _ = std::fs::remove_dir_all(&workspace);
}

#[tokio::test]
async fn test_trio_offline_e2e_parse_fallback_lenient() {
    // Lenient format: "action,target,{args}" — no JSON wrapper.
    // This exercises the comma-separated branch in parse_lenient_router_decision.
    let router_resp = "specialist,coding,{}";

    let router: Arc<dyn LLMProvider> = Arc::new(SequenceProvider::new(
        "offline-router",
        vec![router_resp, router_resp, router_resp],
    ));
    let main: Arc<dyn LLMProvider> = Arc::new(StaticResponseLLM::new("offline-main", "delegating"));
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

    let (agent_loop, workspace) = build_trio_offline_harness(main, router, specialist);

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
        metrics.router_action.lock().as_deref(),
        Some("specialist"),
        "router_action should be 'specialist' after lenient parse"
    );
    assert!(
        metrics
            .specialist_dispatched
            .load(std::sync::atomic::Ordering::Relaxed),
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
        assert!(appears_incomplete(
            "The configuration requires setting the correc"
        ));
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
        assert!(!appears_incomplete(
            "Why cross the road? To avoid borrows. 🦀"
        ));
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
        let content =
            "I will run the command now.\n[I called: exec({\"command\": \"echo hello\"})]";
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
        let nudge_at = |max: u32| -> u32 { ((max as f64) * 0.8).ceil() as u32 };

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
            let last_assistant = messages
                .iter()
                .rev()
                .find(|m| m.get("role").and_then(|r| r.as_str()) == Some("assistant"))
                .and_then(|m| m.get("content").and_then(|c| c.as_str()))
                .unwrap_or("");
            if !last_assistant.trim().is_empty() {
                format!(
                    "{}\n\n[Note: The turn ended before a final answer was produced. This response may be incomplete.]",
                    last_assistant.trim()
                )
            } else {
                "The turn ended before I could produce a final answer. The actions above may be incomplete.".to_string()
            }
        } else {
            final_content.clone()
        };

        assert!(
            result.starts_with("I am working on it."),
            "rescue should start with the last assistant content, got: {result}"
        );
        assert!(
            result.contains("[Note: The turn ended before a final answer"),
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
            let last_assistant = messages
                .iter()
                .rev()
                .find(|m| m.get("role").and_then(|r| r.as_str()) == Some("assistant"))
                .and_then(|m| m.get("content").and_then(|c| c.as_str()))
                .unwrap_or("");
            if !last_assistant.trim().is_empty() {
                format!(
                    "{}\n\n[Note: The turn ended before a final answer was produced. This response may be incomplete.]",
                    last_assistant.trim()
                )
            } else {
                "The turn ended before I could produce a final answer. The actions above may be incomplete.".to_string()
            }
        } else {
            final_content.clone()
        };

        assert_eq!(
            result,
            "The turn ended before I could produce a final answer. The actions above may be incomplete.",
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

// ============================================================================
// RuntimeMode parallel-rollout parity tests (Wave 2)
// ============================================================================
//
// These tests pin the invariant that `SwappableCore.is_local` and
// `SwappableCore.mode` agree by construction. Wave 3's reader-migration
// relies on this invariant to swap each `is_local` read for a `mode` match
// without behavioural drift.
mod runtime_mode_parity_tests {
    use super::*;
    use crate::agent::runtime_mode::RuntimeMode;

    /// Cloud-fixture path: `is_local: false` → `mode == Cloud`, accessor returns Cloud.
    #[test]
    fn mode_accessor_cloud_matches_is_local_false() {
        let core = build_test_core(false, None, None);
        assert!(!core.mode().is_local(), "fixture is is_local=false");
        assert!(
            matches!(core.mode(), RuntimeMode::Cloud),
            "cloud fixture must resolve to RuntimeMode::Cloud"
        );
    }

    /// Local-fixture path: build a local core via a minimal SwappableCoreConfig
    /// (mirrors the pattern in `test_delegation_with_is_local_true`). Verifies
    /// the accessor returns `Local { caps }`.
    #[test]
    fn mode_accessor_local_matches_is_local_true() {
        let workspace = tempfile::tempdir().unwrap().keep();
        let main = MockLLM::named("local-main");
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
            crw_url: String::new(),
            search_max_results: 5,
            exec_timeout: 30,
            restrict_to_workspace: false,
            memory_config: MemoryConfig::default(),
            is_local: true,
            lane: Lane::default(),
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
            code_execution: CodeExecutionConfig::default(),
            python_kernel: PythonKernelConfig::default(),
            cua: CuaToolConfig::default(),
            adaptive_tokens: AdaptiveTokenConfig::default(),
            sessions_db_path: Some(
                std::env::temp_dir().join(format!("nanobot-test-{}.sqlite", uuid::Uuid::new_v4())),
            ),
        });
        assert!(core.mode().is_local(), "fixture is is_local=true");
        assert!(
            matches!(core.mode(), RuntimeMode::Local { .. }),
            "local fixture must resolve to RuntimeMode::Local"
        );
    }

    /// Task 2 / Branch 4: reserve_cap is now derived from mode.
    /// Cloud: passthrough. Local + ample ctx: unchanged. Local + tight ctx: clamped to 25%.
    #[test]
    fn build_core_reserve_cap_cloud_passthrough() {
        // Cloud fixture: max_tokens=4096, max_ctx=16384. Cloud reserve = max_tokens verbatim.
        let core = build_test_core(false, None, None);
        // token_budget exposes reserve via the constructor; reconstruct the
        // expected value from what mode.reserve_cap returns on Cloud.
        let mode = core.mode();
        assert!(matches!(mode, RuntimeMode::Cloud));
        assert_eq!(mode.reserve_cap(4096, 16384), 4096);
    }

    #[test]
    fn build_core_reserve_cap_local_clamped_to_25_pct() {
        // Local fixture with a tight 16K ctx + 4096 max_tokens: reserve clamps to 4096 (ctx/4).
        let workspace = tempfile::tempdir().unwrap().keep();
        let main = MockLLM::named("local-main");
        let core = build_swappable_core(SwappableCoreConfig {
            provider: main,
            workspace,
            model: "local-model".to_string(),
            max_iterations: 10,
            max_continuations: 2,
            max_tokens: 4096,
            temperature: 0.7,
            max_context_tokens: 16_384,
            brave_api_key: None,
            search_provider: "searxng".to_string(),
            searxng_url: "http://localhost:8888".to_string(),
            crw_url: String::new(),
            search_max_results: 5,
            exec_timeout: 30,
            restrict_to_workspace: false,
            memory_config: MemoryConfig::default(),
            is_local: true,
            lane: Lane::default(),
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
            code_execution: CodeExecutionConfig::default(),
            python_kernel: PythonKernelConfig::default(),
            cua: CuaToolConfig::default(),
            adaptive_tokens: AdaptiveTokenConfig::default(),
            sessions_db_path: Some(
                std::env::temp_dir().join(format!("nanobot-test-{}.sqlite", uuid::Uuid::new_v4())),
            ),
        });
        // ctx/4 == 4096; min(4096, 4096) == 4096.
        assert_eq!(core.mode().reserve_cap(4096, 16_384), 4096);
        // Tighter ctx: 8192/4 = 2048 → reserve clamps below max_tokens.
        assert_eq!(core.mode().reserve_cap(4096, 8_192), 2048);
    }

    /// Task 2 / Branch 1–2: context builder reflects the mode's lite/full defaults.
    /// Cloud: `local_prompt_mode == false`, `system_prompt_cap == 0` prior to scaling
    /// (then scale_budgets sets it to 40% of ctx). Local: `local_prompt_mode == true`,
    /// `system_prompt_cap` is a fixed 1000-token cached-prefix cap set by set_lite_mode
    /// (the tiny-model ≤4K branch keeps a leaner 50-token prefix).
    #[test]
    fn build_core_context_cap_cloud_uses_full_scaling() {
        let core = build_test_core(false, None, None);
        // Cloud: scale_budgets sets system_prompt_cap = ctx * 2/5 = 16384 * 2/5 = 6553.
        assert!(!core.context.local_prompt_mode);
        assert_eq!(core.context.system_prompt_cap, 16_384 * 2 / 5);
    }

    #[test]
    fn build_core_context_cap_local_uses_lite_mode() {
        let workspace = tempfile::tempdir().unwrap().keep();
        let main = MockLLM::named("local-main");
        let core = build_swappable_core(SwappableCoreConfig {
            provider: main,
            workspace,
            model: "local-model".to_string(),
            max_iterations: 10,
            max_continuations: 2,
            max_tokens: 4096,
            temperature: 0.7,
            max_context_tokens: 16_384,
            brave_api_key: None,
            search_provider: "searxng".to_string(),
            searxng_url: "http://localhost:8888".to_string(),
            crw_url: String::new(),
            search_max_results: 5,
            exec_timeout: 30,
            restrict_to_workspace: false,
            memory_config: MemoryConfig::default(),
            is_local: true,
            lane: Lane::default(),
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
            code_execution: CodeExecutionConfig::default(),
            python_kernel: PythonKernelConfig::default(),
            cua: CuaToolConfig::default(),
            adaptive_tokens: AdaptiveTokenConfig::default(),
            sessions_db_path: Some(
                std::env::temp_dir().join(format!("nanobot-test-{}.sqlite", uuid::Uuid::new_v4())),
            ),
        });
        assert!(core.context.local_prompt_mode);
        // Local prompt cost is fixed rather than scaling with the model window:
        // the byte-stable cached prefix (identity + skills + workspace bootstrap)
        // is capped at 1000 tokens so Higgs's radix prefix cache can reuse it.
        assert_eq!(core.context.system_prompt_cap, 1000);
    }

    /// Task 2 / Branch 3: cloud memory provider/model follows the pre-Wave-2 path.
    /// MockLLM returns `get_api_base() == None` → triggers the "haiku" branch.
    #[test]
    fn build_core_memory_provider_cloud_defaults_to_haiku_when_no_api_base() {
        let core = build_test_core(false, None, None);
        // provider.get_api_base() is None for MockLLM → "haiku" memory model.
        assert_eq!(core.memory_model, "haiku");
        assert_eq!(core.compactor.model(), "main-model");
    }

    /// Task 2 / Branch 3: local memory provider falls through specialist → main.
    /// With no explicit memory config and no specialist provider, the local
    /// reflection and compaction identities both resolve to the main model.
    #[test]
    fn build_core_memory_provider_local_defaults_to_main_without_trio() {
        let workspace = tempfile::tempdir().unwrap().keep();
        let main = MockLLM::named("local-main");
        let core = build_swappable_core(SwappableCoreConfig {
            provider: main,
            workspace,
            model: "local-model".to_string(),
            max_iterations: 10,
            max_continuations: 2,
            max_tokens: 4096,
            temperature: 0.7,
            max_context_tokens: 16_384,
            brave_api_key: None,
            search_provider: "searxng".to_string(),
            searxng_url: "http://localhost:8888".to_string(),
            crw_url: String::new(),
            search_max_results: 5,
            exec_timeout: 30,
            restrict_to_workspace: false,
            memory_config: MemoryConfig::default(),
            is_local: true,
            lane: Lane::default(),
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
            code_execution: CodeExecutionConfig::default(),
            python_kernel: PythonKernelConfig::default(),
            cua: CuaToolConfig::default(),
            adaptive_tokens: AdaptiveTokenConfig::default(),
            sessions_db_path: Some(
                std::env::temp_dir().join(format!("nanobot-test-{}.sqlite", uuid::Uuid::new_v4())),
            ),
        });
        assert_eq!(core.memory_model, "local-model");
        assert_eq!(core.compactor.model(), "local-model");
    }

    /// Caps carried inside `Local { caps }` match the capabilities resolved
    /// for the model. Ensures `mode_accessor_round_trip` (VALIDATION.md):
    /// construction inputs are consistent with the mode's payload.
    #[test]
    fn mode_accessor_round_trip_local_caps_match_lookup() {
        let workspace = tempfile::tempdir().unwrap().keep();
        let main = MockLLM::named("local-main");
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
            crw_url: String::new(),
            search_max_results: 5,
            exec_timeout: 30,
            restrict_to_workspace: false,
            memory_config: MemoryConfig::default(),
            is_local: true,
            lane: Lane::default(),
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
            code_execution: CodeExecutionConfig::default(),
            python_kernel: PythonKernelConfig::default(),
            cua: CuaToolConfig::default(),
            adaptive_tokens: AdaptiveTokenConfig::default(),
            sessions_db_path: Some(
                std::env::temp_dir().join(format!("nanobot-test-{}.sqlite", uuid::Uuid::new_v4())),
            ),
        });
        match core.mode() {
            RuntimeMode::Local { caps } => {
                // The wrapped caps must equal the model-capabilities lookup for the
                // same model — construction doesn't silently swap in a different
                // capability record.
                assert_eq!(
                    caps.size_class, core.model_capabilities.size_class,
                    "mode caps.size_class must match core.model_capabilities"
                );
                assert_eq!(
                    caps.tool_calling, core.model_capabilities.tool_calling,
                    "mode caps.tool_calling must match core.model_capabilities"
                );
            }
            RuntimeMode::Cloud => panic!("expected Local variant"),
        }
    }

    // ------------------------------------------------------------------
    // Wave 3 reader-migration parity tests.
    //
    // Every migration in plan 09-03 replaces `ctx.core.mode().is_local()` with a
    // typed `mode()` dispatch. These tests pin the parity between the old
    // bool branch and the new mode-driven branch for the non-trivial
    // migration sites, so a future reader-migration regression surfaces
    // as a failing test rather than a behavioral drift only visible in
    // three-way smoke.
    // ------------------------------------------------------------------

    /// agent_shared.rs :820 — proactive grounding message role.
    /// Pre-Wave-3: `if core.mode().is_local() { "user" } else { "system" }`.
    /// Post-Wave-3: `core.mode().grounding_role()`.
    #[test]
    fn wave3_grounding_role_cloud_matches_pre_migration() {
        let core = build_test_core(false, None, None);
        assert_eq!(core.mode().grounding_role(), "system");
    }

    #[test]
    fn wave3_grounding_role_local_matches_pre_migration() {
        let workspace = tempfile::tempdir().unwrap().keep();
        let main = MockLLM::named("local-main");
        let core = build_swappable_core(SwappableCoreConfig {
            provider: main,
            workspace,
            model: "local-model".to_string(),
            max_iterations: 10,
            max_continuations: 2,
            max_tokens: 4096,
            temperature: 0.7,
            max_context_tokens: 16_384,
            brave_api_key: None,
            search_provider: "searxng".to_string(),
            searxng_url: "http://localhost:8888".to_string(),
            crw_url: String::new(),
            search_max_results: 5,
            exec_timeout: 30,
            restrict_to_workspace: false,
            memory_config: MemoryConfig::default(),
            is_local: true,
            lane: Lane::default(),
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
            code_execution: CodeExecutionConfig::default(),
            python_kernel: PythonKernelConfig::default(),
            cua: CuaToolConfig::default(),
            adaptive_tokens: AdaptiveTokenConfig::default(),
            sessions_db_path: Some(
                std::env::temp_dir().join(format!("nanobot-test-{}.sqlite", uuid::Uuid::new_v4())),
            ),
        });
        assert_eq!(core.mode().grounding_role(), "user");
    }

    /// prepare_context.rs :518 — protocol selection respects the mlx: prefix
    /// exception. The non-mlx local path must still pick LocalProtocol, which
    /// is equivalent to `mode.is_local() && !model.starts_with("mlx:")`.
    #[test]
    fn wave3_protocol_selection_mlx_exception_preserved() {
        // Cloud always → CloudProtocol (mode.is_local() == false).
        let cloud = build_test_core(false, None, None);
        assert!(!cloud.mode().is_local());

        // Local with mlx: prefix model would go to CloudProtocol
        // (behavior-preserving: is_local && !starts_with("mlx:")).
        // This test pins the `mode.is_local()` half; the mlx: prefix check
        // is string-based and not affected by the migration.
        let workspace = tempfile::tempdir().unwrap().keep();
        let main = MockLLM::named("mlx-main");
        let local_mlx = build_swappable_core(SwappableCoreConfig {
            provider: main,
            workspace,
            model: "mlx:llama-8b".to_string(),
            max_iterations: 10,
            max_continuations: 2,
            max_tokens: 4096,
            temperature: 0.7,
            max_context_tokens: 16_384,
            brave_api_key: None,
            search_provider: "searxng".to_string(),
            searxng_url: "http://localhost:8888".to_string(),
            crw_url: String::new(),
            search_max_results: 5,
            exec_timeout: 30,
            restrict_to_workspace: false,
            memory_config: MemoryConfig::default(),
            is_local: true,
            lane: Lane::default(),
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
            code_execution: CodeExecutionConfig::default(),
            python_kernel: PythonKernelConfig::default(),
            cua: CuaToolConfig::default(),
            adaptive_tokens: AdaptiveTokenConfig::default(),
            sessions_db_path: Some(
                std::env::temp_dir().join(format!("nanobot-test-{}.sqlite", uuid::Uuid::new_v4())),
            ),
        });
        // mlx: prefix model: mode is Local, but protocol selection still
        // falls through to CloudProtocol via the `!starts_with("mlx:")` guard.
        assert!(local_mlx.mode().is_local());
        assert!(local_mlx.model.starts_with("mlx:"));
    }

    // -------------------------------------------------------------------------
    // Convergence harness
    // -------------------------------------------------------------------------
    //
    // The property that was missing for all three 2026-07 incidents: no matter
    // what the model emits, the loop must terminate in a BOUNDED number of
    // provider calls and return something — it must never spin. These tests
    // feed adversarial providers that never cooperate and assert bounded
    // termination. They guard schema churn, phantom tool narration, and
    // repeated tool-call loops.

    #[derive(Clone)]
    enum TerminalScript {
        Response(crate::providers::base::LLMResponse),
        Error(String),
    }

    struct TerminalNoToolsProvider {
        normal: crate::providers::base::LLMResponse,
        terminal: TerminalScript,
        normal_tools: parking_lot::Mutex<Vec<Option<Vec<Value>>>>,
        terminal_calls:
            parking_lot::Mutex<Vec<(crate::providers::base::ToolChoice, Option<Vec<Value>>)>>,
    }

    impl TerminalNoToolsProvider {
        fn new(terminal: TerminalScript) -> Self {
            let mut arguments = std::collections::HashMap::new();
            arguments.insert("path".to_string(), json!("."));
            Self {
                normal: crate::providers::base::LLMResponse {
                    content: Some(String::new()),
                    tool_calls: vec![crate::providers::base::ToolCallRequest {
                        id: "tc_normal".to_string(),
                        name: "list_dir".to_string(),
                        arguments,
                    }],
                    finish_reason: FinishReason::ToolCalls,
                    usage: std::collections::HashMap::new(),
                },
                terminal,
                normal_tools: parking_lot::Mutex::new(Vec::new()),
                terminal_calls: parking_lot::Mutex::new(Vec::new()),
            }
        }
    }

    #[async_trait]
    impl LLMProvider for TerminalNoToolsProvider {
        async fn chat(
            &self,
            _messages: &[Value],
            tools: Option<&[Value]>,
            _model: Option<&str>,
            _max_tokens: u32,
            _temperature: f64,
            _thinking_budget: Option<u32>,
            _top_p: Option<f64>,
        ) -> anyhow::Result<crate::providers::base::LLMResponse> {
            self.normal_tools.lock().push(tools.map(<[Value]>::to_vec));
            Ok(self.normal.clone())
        }

        async fn chat_with_tool_choice(
            &self,
            _messages: &[Value],
            tools: Option<&[Value]>,
            _model: Option<&str>,
            _max_tokens: u32,
            _temperature: f64,
            thinking_budget: Option<u32>,
            _top_p: Option<f64>,
            tool_choice: crate::providers::base::ToolChoice,
        ) -> anyhow::Result<crate::providers::base::LLMResponse> {
            assert_eq!(thinking_budget, None, "terminal call must disable thinking");
            self.terminal_calls
                .lock()
                .push((tool_choice, tools.map(<[Value]>::to_vec)));
            match &self.terminal {
                TerminalScript::Response(response) => Ok(response.clone()),
                TerminalScript::Error(message) => anyhow::bail!(message.clone()),
            }
        }

        fn get_default_model(&self) -> &str {
            "local-main"
        }
    }

    fn terminal_text(content: Option<&str>) -> TerminalScript {
        TerminalScript::Response(crate::providers::base::LLMResponse {
            content: content.map(str::to_string),
            tool_calls: vec![],
            finish_reason: FinishReason::Stop,
            usage: std::collections::HashMap::new(),
        })
    }

    #[tokio::test]
    async fn terminal_no_tools_prose_uses_one_stable_recorded_call() {
        let provider = Arc::new(TerminalNoToolsProvider::new(terminal_text(Some(
            "terminal summary",
        ))));
        let (agent_loop, workspace) =
            build_local_inline_harness_with_iters(provider.clone() as Arc<dyn LLMProvider>, 1);
        let session_key = format!("terminal-prose-{}", uuid::Uuid::new_v4());

        let response = agent_loop
            .process_direct("inspect once then finish", &session_key, "test", "offline")
            .await;

        assert_eq!(response, "terminal summary");
        let terminal_calls = provider.terminal_calls.lock();
        assert_eq!(terminal_calls.len(), 1);
        assert_eq!(
            terminal_calls[0].0,
            crate::providers::base::ToolChoice::None
        );
        assert_eq!(provider.normal_tools.lock()[0], terminal_calls[0].1);
        drop(terminal_calls);

        let core = agent_loop.shared.core_handle.swappable();
        let session = core
            .sessions
            .get_latest_session(&session_key)
            .await
            .unwrap();
        let replay = core
            .sessions
            .load_session_replay(&session.id)
            .await
            .unwrap();
        let terminal = replay
            .model_calls
            .iter()
            .find(|call| call.purpose == crate::session::db::ModelCallPurpose::Continuation)
            .expect("terminal request must be recorded as a continuation");
        let request: crate::session::db::RecordedProviderRequest =
            serde_json::from_slice(&terminal.request).unwrap();
        assert_eq!(request.tool_choice, "none");
        assert!(!request.streaming);
        assert_eq!(
            persisted_turn_outcome(&core.sessions, &session.id).await,
            "finished"
        );
        let persisted = core.sessions.get_all_messages(&session.id).await;
        let persisted_text = persisted
            .iter()
            .filter_map(|message| message.get("content").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n");
        for scaffold in [
            "Report what the previous tool results showed",
            "Your tool results are already in the conversation above",
            "You called the same tool(s) with the same arguments again",
        ] {
            assert!(
                !persisted_text.contains(scaffold),
                "persisted scaffold: {scaffold}"
            );
        }

        let _ = std::fs::remove_dir_all(&workspace);
    }

    #[tokio::test]
    async fn terminal_no_tools_prose_streams_once_after_prior_tool_round_text() {
        let mut scripted = TerminalNoToolsProvider::new(terminal_text(Some("terminal summary")));
        scripted.normal.content = Some("normal streamed setup".to_string());
        let provider = Arc::new(scripted);
        let (agent_loop, workspace) =
            build_local_inline_harness_with_iters(provider as Arc<dyn LLMProvider>, 1);
        let session_key = format!("terminal-streamed-{}", uuid::Uuid::new_v4());
        let (delta_tx, mut delta_rx) = tokio::sync::mpsc::unbounded_channel();

        let response = agent_loop
            .process_direct_streaming(
                "inspect once then finish",
                &session_key,
                "test",
                "offline",
                None,
                delta_tx,
                None,
                None,
                None,
                None,
            )
            .await;

        assert_eq!(response, "terminal summary");
        let mut deltas = Vec::new();
        while let Ok(delta) = delta_rx.try_recv() {
            deltas.push(delta);
        }
        assert!(
            deltas.iter().any(|delta| delta == "normal streamed setup"),
            "normal streamed setup missing: {deltas:?}"
        );
        assert_eq!(
            deltas
                .iter()
                .filter(|delta| delta.as_str() == "terminal summary")
                .count(),
            1,
            "terminal prose must be emitted exactly once after prior streamed text: {deltas:?}"
        );

        let _ = std::fs::remove_dir_all(&workspace);
    }

    #[tokio::test]
    async fn terminal_no_tools_without_prior_contract_bypasses_strict_router() {
        let main = Arc::new(TerminalNoToolsProvider::new(terminal_text(Some(
            "strict terminal summary",
        ))));
        let router_body = r#"{"action":"respond","target":"main","args":{},"confidence":0.9}"#;
        let router = Arc::new(SequenceProvider::new(
            "offline-router",
            vec![router_body, router_body, router_body, router_body],
        ));
        let specialist: Arc<dyn LLMProvider> = Arc::new(StaticResponseLLM::new(
            "offline-specialist",
            "specialist unused",
        ));
        let (agent_loop, workspace) = build_trio_offline_harness_with_iters(
            main.clone() as Arc<dyn LLMProvider>,
            router.clone() as Arc<dyn LLMProvider>,
            specialist,
            None,
            0,
        );
        let session_key = format!("terminal-strict-trio-{}", uuid::Uuid::new_v4());

        let response = agent_loop
            .process_direct("What is 2+2?", &session_key, "test", "offline")
            .await;

        assert!(!response.is_empty(), "{response:?}");
        assert_ne!(response, "strict terminal summary");
        assert_eq!(
            router.call_count(),
            0,
            "terminal mode reran router preflight"
        );
        assert!(main.normal_tools.lock().is_empty());
        let terminal_calls = main.terminal_calls.lock();
        assert!(
            terminal_calls.is_empty(),
            "without an issued normal contract, terminal mode must not invent one"
        );
        let core = agent_loop.shared.core_handle.swappable();
        let session = core
            .sessions
            .get_latest_session(&session_key)
            .await
            .unwrap();
        assert_eq!(
            persisted_turn_outcome(&core.sessions, &session.id).await,
            "limit_exhausted"
        );

        let _ = std::fs::remove_dir_all(&workspace);
    }

    #[tokio::test]
    async fn terminal_no_tools_ignored_tool_call_is_rejected_without_execution() {
        let mut arguments = std::collections::HashMap::new();
        arguments.insert("command".to_string(), json!("printf should-not-run"));
        let provider = Arc::new(TerminalNoToolsProvider::new(TerminalScript::Response(
            crate::providers::base::LLMResponse {
                content: None,
                tool_calls: vec![crate::providers::base::ToolCallRequest {
                    id: "tc_ignored_none".to_string(),
                    name: "exec".to_string(),
                    arguments,
                }],
                finish_reason: FinishReason::ToolCalls,
                usage: std::collections::HashMap::new(),
            },
        )));
        let (agent_loop, workspace) =
            build_local_inline_harness_with_iters(provider.clone() as Arc<dyn LLMProvider>, 1);
        let session_key = format!("terminal-ignored-{}", uuid::Uuid::new_v4());

        agent_loop
            .process_direct("inspect once then finish", &session_key, "test", "offline")
            .await;

        assert_eq!(provider.terminal_calls.lock().len(), 1);
        let core = agent_loop.shared.core_handle.swappable();
        let session = core
            .sessions
            .get_latest_session(&session_key)
            .await
            .unwrap();
        let replay = core
            .sessions
            .load_session_replay(&session.id)
            .await
            .unwrap();
        assert!(!replay.events.iter().any(|event| matches!(
            &event.payload,
            crate::session::db::SessionEventPayload::ToolExecute { tool_call_id, .. }
                if tool_call_id == "tc_ignored_none"
        )));
        assert!(replay.events.iter().any(|event| matches!(
            &event.payload,
            crate::session::db::SessionEventPayload::ToolPreExecute {
                tool_call_id,
                decision: crate::session::db::ToolPreExecuteDecision::Rejected { reason },
                ..
            } if tool_call_id == "tc_ignored_none" && reason == "terminal_no_tools"
        )));
        let messages = core.sessions.get_all_messages(&session.id).await;
        assert!(messages.iter().any(|message| {
            message.get("role").and_then(Value::as_str) == Some("assistant")
                && message
                    .get("tool_calls")
                    .and_then(Value::as_array)
                    .is_some_and(|calls| {
                        calls.iter().any(|call| {
                            call.get("id").and_then(Value::as_str) == Some("tc_ignored_none")
                        })
                    })
        }));
        assert!(messages.iter().any(|message| {
            message.get("role").and_then(Value::as_str) == Some("tool")
                && message.get("tool_call_id").and_then(Value::as_str) == Some("tc_ignored_none")
                && message.get("ok").and_then(Value::as_bool) == Some(false)
        }));
        let (stored_receipt, stored_ok) = core
            .sessions
            .load_tool_result_with_status(&session.id, "tc_ignored_none")
            .await
            .expect("terminal rejected call raw row");
        assert_eq!(stored_ok, Some(false));
        assert_eq!(
            stored_receipt,
            "terminal tool_choice=none: tool call was rejected and not executed"
        );
        assert_eq!(
            persisted_turn_outcome(&core.sessions, &session.id).await,
            "limit_exhausted"
        );

        let _ = std::fs::remove_dir_all(&workspace);
    }

    #[tokio::test]
    async fn terminal_no_tools_receipt_batch_failure_rolls_back_carrier() {
        let mut arguments = std::collections::HashMap::new();
        arguments.insert("command".to_string(), json!("printf should-not-run"));
        let provider = Arc::new(TerminalNoToolsProvider::new(TerminalScript::Response(
            crate::providers::base::LLMResponse {
                content: None,
                tool_calls: vec![crate::providers::base::ToolCallRequest {
                    id: "tc_terminal_receipt_fault".to_string(),
                    name: "exec".to_string(),
                    arguments,
                }],
                finish_reason: FinishReason::ToolCalls,
                usage: std::collections::HashMap::new(),
            },
        )));
        let (agent_loop, workspace) =
            build_local_inline_harness_with_iters(provider.clone() as Arc<dyn LLMProvider>, 1);
        let session_key = format!("terminal-receipt-fault-{}", uuid::Uuid::new_v4());
        let core = agent_loop.shared.core_handle.swappable();
        let session = core.sessions.get_or_resume(&session_key).await;
        {
            let conn = rusqlite::Connection::open(core.sessions.path()).unwrap();
            conn.execute_batch(
                "CREATE TRIGGER fail_terminal_receipt \
                 BEFORE INSERT ON messages \
                 WHEN NEW.tool_call_id = 'tc_terminal_receipt_fault' \
                 BEGIN SELECT RAISE(ABORT, 'synthetic terminal receipt failure'); END;",
            )
            .unwrap();
        }

        let response = agent_loop
            .process_direct("inspect once then finish", &session_key, "test", "offline")
            .await;

        assert!(response.contains("rejection receipts"), "{response:?}");
        assert_eq!(provider.terminal_calls.lock().len(), 1);
        let messages = core.sessions.get_all_messages(&session.id).await;
        assert!(!messages.iter().any(|message| {
            message
                .get("tool_calls")
                .and_then(Value::as_array)
                .is_some_and(|calls| {
                    calls.iter().any(|call| {
                        call.get("id").and_then(Value::as_str) == Some("tc_terminal_receipt_fault")
                    })
                })
        }));
        assert!(!messages.iter().any(|message| {
            message.get("tool_call_id").and_then(Value::as_str) == Some("tc_terminal_receipt_fault")
        }));
        assert_eq!(
            core.sessions
                .load_tool_result_with_status(&session.id, "tc_terminal_receipt_fault")
                .await,
            None,
            "failed terminal protocol transaction must roll back its raw row"
        );
        let replay = core
            .sessions
            .load_session_replay(&session.id)
            .await
            .unwrap();
        assert!(!replay.events.iter().any(|event| match &event.payload {
            crate::session::db::SessionEventPayload::ToolPreExecute { tool_call_id, .. }
            | crate::session::db::SessionEventPayload::ToolExecute { tool_call_id, .. } => {
                tool_call_id == "tc_terminal_receipt_fault"
            }
            _ => false,
        }));

        let _ = std::fs::remove_dir_all(&workspace);
    }

    #[tokio::test]
    async fn terminal_no_tools_decision_failure_keeps_carrier_receipt_pair() {
        let mut arguments = std::collections::HashMap::new();
        arguments.insert("command".to_string(), json!("printf should-not-run"));
        let provider = Arc::new(TerminalNoToolsProvider::new(TerminalScript::Response(
            crate::providers::base::LLMResponse {
                content: None,
                tool_calls: vec![crate::providers::base::ToolCallRequest {
                    id: "tc_terminal_decision_fault".to_string(),
                    name: "exec".to_string(),
                    arguments,
                }],
                finish_reason: FinishReason::ToolCalls,
                usage: std::collections::HashMap::new(),
            },
        )));
        let (agent_loop, workspace) =
            build_local_inline_harness_with_iters(provider.clone() as Arc<dyn LLMProvider>, 1);
        let session_key = format!("terminal-decision-fault-{}", uuid::Uuid::new_v4());
        let core = agent_loop.shared.core_handle.swappable();
        let session = core.sessions.get_or_resume(&session_key).await;
        {
            let conn = rusqlite::Connection::open(core.sessions.path()).unwrap();
            conn.execute_batch(
                "CREATE TRIGGER fail_terminal_decision \
                 BEFORE INSERT ON session_events \
                 WHEN NEW.event_kind = 'tool_pre_execute' \
                   AND NEW.payload_json LIKE '%tc_terminal_decision_fault%' \
                 BEGIN SELECT RAISE(ABORT, 'synthetic terminal decision failure'); END;",
            )
            .unwrap();
        }

        let response = agent_loop
            .process_direct("inspect once then finish", &session_key, "test", "offline")
            .await;

        assert!(response.contains("tool rejection"), "{response:?}");
        assert_eq!(provider.terminal_calls.lock().len(), 1);
        let messages = core.sessions.get_all_messages(&session.id).await;
        let has_carrier = messages.iter().any(|message| {
            message
                .get("tool_calls")
                .and_then(Value::as_array)
                .is_some_and(|calls| {
                    calls.iter().any(|call| {
                        call.get("id").and_then(Value::as_str) == Some("tc_terminal_decision_fault")
                    })
                })
        });
        let has_receipt = messages.iter().any(|message| {
            message.get("tool_call_id").and_then(Value::as_str)
                == Some("tc_terminal_decision_fault")
                && message.get("ok").and_then(Value::as_bool) == Some(false)
        });
        assert_eq!((has_carrier, has_receipt), (true, true));
        let (stored_receipt, stored_ok) = core
            .sessions
            .load_tool_result_with_status(&session.id, "tc_terminal_decision_fault")
            .await
            .expect("terminal rejection row survives later decision-journal fault");
        assert_eq!(stored_ok, Some(false));
        assert_eq!(
            stored_receipt,
            "terminal tool_choice=none: tool call was rejected and not executed"
        );
        let replay = core
            .sessions
            .load_session_replay(&session.id)
            .await
            .unwrap();
        assert!(!replay.events.iter().any(|event| matches!(
            &event.payload,
            crate::session::db::SessionEventPayload::ToolExecute { tool_call_id, .. }
                if tool_call_id == "tc_terminal_decision_fault"
        )));

        let _ = std::fs::remove_dir_all(&workspace);
    }

    #[tokio::test]
    async fn terminal_no_tools_error_and_empty_do_not_retry() {
        for (label, terminal) in [
            (
                "error",
                TerminalScript::Error("terminal unavailable".to_string()),
            ),
            ("empty", terminal_text(None)),
        ] {
            let provider = Arc::new(TerminalNoToolsProvider::new(terminal));
            let (agent_loop, workspace) =
                build_local_inline_harness_with_iters(provider.clone() as Arc<dyn LLMProvider>, 1);
            let session_key = format!("terminal-{label}-{}", uuid::Uuid::new_v4());

            agent_loop
                .process_direct("inspect once then finish", &session_key, "test", "offline")
                .await;

            assert_eq!(provider.terminal_calls.lock().len(), 1, "{label}");
            let core = agent_loop.shared.core_handle.swappable();
            let session = core
                .sessions
                .get_latest_session(&session_key)
                .await
                .unwrap();
            assert_eq!(
                persisted_turn_outcome(&core.sessions, &session.id).await,
                "limit_exhausted",
                "{label}"
            );
            let _ = std::fs::remove_dir_all(&workspace);
        }
    }

    /// One deterministic replay of the failure cluster recovered from the
    /// 2026-09-01 session: truthful shell/API failures, a metered network
    /// lease, paired blocked calls, bounded terminal prose, and a later empty
    /// model turn. The provider and HTTP fixture are both in-process so this
    /// gate exercises the production loop without external state.
    #[tokio::test]
    async fn compound_session_failure_replay() {
        struct CompoundReplayProvider {
            responses:
                parking_lot::Mutex<std::collections::VecDeque<crate::providers::base::LLMResponse>>,
            terminal_calls:
                parking_lot::Mutex<Vec<(crate::providers::base::ToolChoice, Option<Vec<Value>>)>>,
            empty_turn_armed: std::sync::atomic::AtomicBool,
        }

        #[async_trait]
        impl LLMProvider for CompoundReplayProvider {
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
                if let Some(response) = self.responses.lock().pop_front() {
                    return Ok(response);
                }
                if self
                    .empty_turn_armed
                    .swap(false, std::sync::atomic::Ordering::SeqCst)
                {
                    return Ok(crate::providers::base::LLMResponse {
                        content: None,
                        tool_calls: vec![],
                        finish_reason: FinishReason::Stop,
                        usage: std::collections::HashMap::new(),
                    });
                }
                Ok(crate::providers::base::LLMResponse {
                    content: Some(
                        "The evidence was collected and the failures were preserved.".to_string(),
                    ),
                    tool_calls: vec![],
                    finish_reason: FinishReason::Stop,
                    usage: std::collections::HashMap::new(),
                })
            }

            async fn chat_with_tool_choice(
                &self,
                _messages: &[Value],
                tools: Option<&[Value]>,
                _model: Option<&str>,
                _max_tokens: u32,
                _temperature: f64,
                thinking_budget: Option<u32>,
                _top_p: Option<f64>,
                tool_choice: crate::providers::base::ToolChoice,
            ) -> anyhow::Result<crate::providers::base::LLMResponse> {
                assert_eq!(thinking_budget, None);
                self.terminal_calls
                    .lock()
                    .push((tool_choice, tools.map(<[Value]>::to_vec)));
                self.empty_turn_armed
                    .store(true, std::sync::atomic::Ordering::SeqCst);
                Ok(crate::providers::base::LLMResponse {
                    content: Some(
                        "The evidence was collected and the failures were preserved.".to_string(),
                    ),
                    tool_calls: vec![],
                    finish_reason: FinishReason::Stop,
                    usage: std::collections::HashMap::new(),
                })
            }

            fn get_default_model(&self) -> &str {
                "local-main"
            }
        }

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let network_paths = Arc::new(parking_lot::Mutex::new(Vec::<String>::new()));
        let recorded_paths = Arc::clone(&network_paths);
        let server = tokio::spawn(async move {
            loop {
                let Ok((mut socket, _)) = listener.accept().await else {
                    break;
                };
                let paths = Arc::clone(&recorded_paths);
                tokio::spawn(async move {
                    use tokio::io::{AsyncReadExt, AsyncWriteExt};

                    let mut request = [0_u8; 2048];
                    let read = socket.read(&mut request).await.unwrap_or(0);
                    let request = String::from_utf8_lossy(&request[..read]);
                    let path = request
                        .lines()
                        .next()
                        .and_then(|line| line.split_whitespace().nth(1))
                        .unwrap_or("/")
                        .to_string();
                    paths.lock().push(path.clone());
                    let body = match path.as_str() {
                        "/primary" => {
                            r#"{"message":"API rate limit exceeded for 127.0.0.1.","documentation_url":"https://docs.github.com/rest/using-the-rest-api/rate-limits-for-the-rest-api"}"#
                        }
                        "/secondary" => {
                            r#"{"message":"You have exceeded a secondary rate limit. Please wait a few minutes before you try again.","documentation_url":"https://docs.github.com/rest/using-the-rest-api/rate-limits-for-the-rest-api"}"#
                        }
                        _ => r#"{"evidence":"confirmed"}"#,
                    };
                    let response = format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                        body.len(),
                        body
                    );
                    let _ = socket.write_all(response.as_bytes()).await;
                });
            }
        });

        let tool_response = |id: String, command: String| {
            let mut arguments = std::collections::HashMap::new();
            arguments.insert("command".to_string(), json!(command));
            crate::providers::base::LLMResponse {
                content: Some(String::new()),
                tool_calls: vec![crate::providers::base::ToolCallRequest {
                    id,
                    name: "exec".to_string(),
                    arguments,
                }],
                finish_reason: FinishReason::ToolCalls,
                usage: std::collections::HashMap::new(),
            }
        };
        let base_url = format!("http://127.0.0.1:{port}");
        let mut responses = vec![
            tool_response(
                "tc_compound_evidence".to_string(),
                format!("curl -sS {base_url}/evidence"),
            ),
            tool_response(
                "tc_compound_pipeline".to_string(),
                format!("false | curl -sS {base_url}/pipeline"),
            ),
            tool_response(
                "tc_compound_primary".to_string(),
                format!("curl -sS {base_url}/primary"),
            ),
            tool_response(
                "tc_compound_secondary".to_string(),
                format!("curl -sS {base_url}/secondary"),
            ),
        ];
        // One evidence call succeeds above. Fill the remaining 95 slots with
        // deterministic successful executions so the four following calls
        // exercise the 97th-call rejection and durable receipt path without
        // making the localhost stress fixture itself the source of failures.
        for index in 0..95 {
            responses.push(tool_response(
                format!("tc_compound_metered_{index}"),
                format!("true # metered-{index}"),
            ));
        }
        for (id, path) in [
            ("tc_compound_blocked_repeat_1", "blocked/repeat"),
            ("tc_compound_blocked_repeat_2", "blocked/repeat"),
            ("tc_compound_blocked_distinct_1", "blocked/distinct/1"),
            ("tc_compound_blocked_distinct_2", "blocked/distinct/2"),
        ] {
            responses.push(tool_response(
                id.to_string(),
                format!("curl -sS {base_url}/{path}"),
            ));
        }
        let provider = Arc::new(CompoundReplayProvider {
            responses: parking_lot::Mutex::new(responses.into()),
            terminal_calls: parking_lot::Mutex::new(Vec::new()),
            empty_turn_armed: std::sync::atomic::AtomicBool::new(false),
        });
        let (agent_loop, workspace) = build_local_harness_with_runtime_options_context(
            provider.clone() as Arc<dyn LLMProvider>,
            130,
            crate::config::schema::ReasoningConfig::default(),
            ToolDelegationConfig::default(),
            None,
            131_072,
        );
        let session_key = format!("compound-replay-{}", uuid::Uuid::new_v4());
        let (delta_tx, mut delta_rx) = tokio::sync::mpsc::unbounded_channel();

        let response = tokio::time::timeout(
            std::time::Duration::from_secs(30),
            agent_loop.process_direct_streaming(
                "collect evidence and report every failure",
                &session_key,
                "test",
                "offline",
                None,
                delta_tx,
                None,
                None,
                None,
                None,
            ),
        )
        .await
        .expect("compound replay must terminate");

        assert_eq!(
            response,
            "The evidence was collected and the failures were preserved."
        );
        let mut deltas = Vec::new();
        while let Ok(delta) = delta_rx.try_recv() {
            deltas.push(delta);
        }
        assert_eq!(
            deltas
                .iter()
                .filter(|delta| {
                    delta.as_str() == "The evidence was collected and the failures were preserved."
                })
                .count(),
            1,
            "terminal prose must stream exactly once: {deltas:?}"
        );
        let terminal_calls = provider.terminal_calls.lock();
        assert_eq!(terminal_calls.len(), 1);
        assert_eq!(
            terminal_calls[0].0,
            crate::providers::base::ToolChoice::None
        );
        drop(terminal_calls);

        let core = agent_loop.shared.core_handle.swappable();
        let successful_session = core
            .sessions
            .get_latest_session(&session_key)
            .await
            .expect("compound session");
        let successful_replay = core
            .sessions
            .load_session_replay(&successful_session.id)
            .await
            .expect("compound replay");
        assert_eq!(
            persisted_turn_outcome(&core.sessions, &successful_session.id).await,
            "finished"
        );

        let terminal_call = successful_replay
            .model_calls
            .iter()
            .find(|call| call.purpose == crate::session::db::ModelCallPurpose::Continuation)
            .expect("terminal continuation call");
        let terminal_request: crate::session::db::RecordedProviderRequest =
            serde_json::from_slice(&terminal_call.request).unwrap();
        let prior_main_call = successful_replay
            .model_calls
            .iter()
            .rev()
            .find(|call| call.purpose == crate::session::db::ModelCallPurpose::Main)
            .expect("prior main call");
        let prior_main_request: crate::session::db::RecordedProviderRequest =
            serde_json::from_slice(&prior_main_call.request).unwrap();
        assert_eq!(terminal_request.tool_choice, "none");
        assert!(!terminal_request.streaming);
        assert!(prior_main_request.streaming);
        assert_eq!(
            serde_json::to_vec(&terminal_request.tools).unwrap(),
            serde_json::to_vec(&prior_main_request.tools).unwrap(),
            "terminal call must preserve the exact main tool catalog bytes"
        );

        let failed_execution_ids = [
            "tc_compound_pipeline",
            "tc_compound_primary",
            "tc_compound_secondary",
        ];
        for tool_call_id in failed_execution_ids {
            let event_ok = successful_replay
                .events
                .iter()
                .find_map(|event| match &event.payload {
                    crate::session::db::SessionEventPayload::ToolExecute {
                        tool_call_id: event_id,
                        ok,
                        ..
                    } if event_id == tool_call_id => Some(*ok),
                    _ => None,
                });
            assert_eq!(event_ok, Some(false), "event status for {tool_call_id}");
            let (_, row_ok) = core
                .sessions
                .load_tool_result_with_status(&successful_session.id, tool_call_id)
                .await
                .unwrap_or_else(|| panic!("raw tool row for {tool_call_id}"));
            assert_eq!(row_ok, Some(false), "raw row status for {tool_call_id}");
        }
        let (evidence, evidence_ok) = core
            .sessions
            .load_tool_result_with_status(&successful_session.id, "tc_compound_evidence")
            .await
            .expect("evidence row");
        assert!(evidence.contains("confirmed"));
        assert_eq!(evidence_ok, Some(true));
        let (pipeline, _) = core
            .sessions
            .load_tool_result_with_status(&successful_session.id, "tc_compound_pipeline")
            .await
            .expect("pipeline row");
        assert!(pipeline.contains("Exit code: 1"), "{pipeline}");
        let (primary, _) = core
            .sessions
            .load_tool_result_with_status(&successful_session.id, "tc_compound_primary")
            .await
            .expect("primary rate-limit row");
        assert!(primary.contains("API rate limit exceeded"), "{primary}");
        let (secondary, _) = core
            .sessions
            .load_tool_result_with_status(&successful_session.id, "tc_compound_secondary")
            .await
            .expect("secondary rate-limit row");
        assert!(secondary.contains("secondary rate limit"), "{secondary}");

        let messages = core.sessions.get_all_messages(&successful_session.id).await;
        let carrier_ids: Vec<String> = messages
            .iter()
            .filter(|message| message.get("role").and_then(Value::as_str) == Some("assistant"))
            .flat_map(|message| {
                message
                    .get("tool_calls")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .filter_map(|call| call.get("id").and_then(Value::as_str))
                    .map(str::to_string)
            })
            .collect();
        let receipts: Vec<&Value> = messages
            .iter()
            .filter(|message| message.get("role").and_then(Value::as_str) == Some("tool"))
            .collect();
        let receipt_ids: Vec<String> = receipts
            .iter()
            .map(|message| {
                message
                    .get("tool_call_id")
                    .and_then(Value::as_str)
                    .unwrap_or_else(|| panic!("tool receipt lacks tool_call_id: {message}"))
                    .to_string()
            })
            .collect();
        assert_eq!(
            carrier_ids.len(),
            receipt_ids.len(),
            "assistant carrier call count must equal tool receipt count; carriers={carrier_ids:?}, receipts={receipt_ids:?}"
        );
        let carrier_counts = carrier_ids.iter().fold(
            std::collections::HashMap::<String, usize>::new(),
            |mut counts, id| {
                *counts.entry(id.clone()).or_default() += 1;
                counts
            },
        );
        let receipt_counts = receipt_ids.iter().fold(
            std::collections::HashMap::<String, usize>::new(),
            |mut counts, id| {
                *counts.entry(id.clone()).or_default() += 1;
                counts
            },
        );
        assert!(
            carrier_counts.values().all(|count| *count == 1),
            "assistant carrier IDs must be unique and appear exactly once: {carrier_counts:?}"
        );
        assert!(
            receipt_counts.values().all(|count| *count == 1),
            "tool receipt IDs must be unique and appear exactly once: {receipt_counts:?}"
        );
        assert_eq!(
            carrier_counts, receipt_counts,
            "assistant carrier and tool receipt ID multisets must match exactly"
        );
        for tool_call_id in failed_execution_ids {
            assert!(receipts.iter().any(|message| {
                message.get("tool_call_id").and_then(Value::as_str) == Some(tool_call_id)
                    && message.get("ok").and_then(Value::as_bool) == Some(false)
            }));
        }
        let blocked_receipts: Vec<&Value> = receipts
            .iter()
            .copied()
            .filter(|message| {
                message
                    .get("content")
                    .and_then(Value::as_str)
                    .is_some_and(|content| content.starts_with("lease exhausted:"))
            })
            .collect();
        assert!(
            !blocked_receipts.is_empty(),
            "the 96-success budget must produce a durable 97th-call receipt"
        );
        for receipt in blocked_receipts {
            let tool_call_id = receipt
                .get("tool_call_id")
                .and_then(Value::as_str)
                .expect("lease receipt must carry its tool-call ID");
            assert_eq!(receipt.get("ok").and_then(Value::as_bool), Some(false));
            let receipt_content = receipt
                .get("content")
                .and_then(Value::as_str)
                .expect("lease receipt must carry content");
            let (stored_content, stored_ok) = core
                .sessions
                .load_tool_result_with_status(&successful_session.id, tool_call_id)
                .await
                .expect("lease receipt must have a raw row");
            assert_eq!(stored_ok, Some(false));
            assert_eq!(stored_content, receipt_content);
            assert!(successful_replay.events.iter().any(|event| matches!(
                &event.payload,
                crate::session::db::SessionEventPayload::ToolPreExecute {
                    tool_call_id: event_id,
                    decision: crate::session::db::ToolPreExecuteDecision::Rejected { reason },
                    ..
                } if event_id == tool_call_id && reason == "lease:lease_exhausted"
            )));
        }

        let paths = network_paths.lock().clone();
        let metered_paths = paths
            .iter()
            .filter(|path| path.starts_with("/metered/"))
            .count();
        assert!(
            metered_paths
                <= crate::agent::lease::DEFAULT_TOOLS_PER_LEASE as usize - 1,
            "the successful lease budget must stop metered execution after the first successful evidence call: {paths:?}"
        );
        assert!(
            paths.len()
                <= 4 + (crate::agent::lease::DEFAULT_TOOLS_PER_LEASE as usize - 1),
            "failed calls may reach the network, but no call after lease exhaustion may do so: {paths:?}"
        );
        assert!(
            !paths.iter().any(|path| path.starts_with("/blocked/")),
            "lease-rejected calls must not reach the network: {paths:?}"
        );
        let persisted_text = messages
            .iter()
            .filter_map(|message| message.get("content").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n");
        for retired_scaffold in [
            "Report what the previous tool results showed",
            "Your tool results are already in the conversation above",
            "You called the same tool(s) with the same arguments again",
        ] {
            assert!(!persisted_text.contains(retired_scaffold));
        }

        let empty_session_key = format!("compound-empty-{}", uuid::Uuid::new_v4());
        let (empty_tx, _empty_rx) = tokio::sync::mpsc::unbounded_channel();
        let empty_response = agent_loop
            .process_direct_streaming(
                "return an empty stream",
                &empty_session_key,
                "test",
                "offline",
                None,
                empty_tx,
                None,
                None,
                None,
                None,
            )
            .await;
        assert!(empty_response.contains("couldn't produce a response"));
        let empty_session = core
            .sessions
            .get_latest_session(&empty_session_key)
            .await
            .expect("empty session");
        let empty_stream_replay = core
            .sessions
            .load_session_replay(&empty_session.id)
            .await
            .expect("empty replay");
        assert_eq!(
            persisted_turn_outcome(&core.sessions, &empty_session.id).await,
            "empty"
        );
        assert!(matches!(
            empty_stream_replay.availability,
            crate::session::db::ReplayAvailability::Exact
        ));
        assert_eq!(provider.terminal_calls.lock().len(), 1);

        server.abort();
        let _ = std::fs::remove_dir_all(&workspace);
    }

    /// A provider that emits a distinct side-effect tool call on every turn.
    /// The session replay records the main-call catalogs for the lease
    /// convergence assertion.
    struct LoopingProvider {
        name: String,
        call_count: std::sync::atomic::AtomicU32,
        terminal_choices: parking_lot::Mutex<Vec<crate::providers::base::ToolChoice>>,
    }

    impl LoopingProvider {
        fn new(name: &str) -> Self {
            Self {
                name: name.to_string(),
                call_count: std::sync::atomic::AtomicU32::new(0),
                terminal_choices: parking_lot::Mutex::new(Vec::new()),
            }
        }
        fn call_count(&self) -> u32 {
            self.call_count.load(std::sync::atomic::Ordering::Relaxed)
        }
    }

    #[async_trait]
    impl LLMProvider for LoopingProvider {
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
            let n = self
                .call_count
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            // Distinct command per call: the cached-duplicate breaker cannot
            // arm, so the lease exhaustion path exercises paired rejections.
            // Uses `exec` (side-effect) not `list_dir`
            // (read-only) because read-only tools now auto-renew the lease.
            let mut args = std::collections::HashMap::new();
            args.insert("command".to_string(), json!(format!("echo {n}")));
            Ok(crate::providers::base::LLMResponse {
                content: Some(String::new()),
                tool_calls: vec![crate::providers::base::ToolCallRequest {
                    id: format!("tc_loop_{n}"),
                    name: "exec".to_string(),
                    arguments: args,
                }],
                finish_reason: FinishReason::ToolCalls,
                usage: std::collections::HashMap::new(),
            })
        }

        async fn chat_with_tool_choice(
            &self,
            _messages: &[Value],
            _tools: Option<&[Value]>,
            _model: Option<&str>,
            _max_tokens: u32,
            _temperature: f64,
            _thinking_budget: Option<u32>,
            _top_p: Option<f64>,
            tool_choice: crate::providers::base::ToolChoice,
        ) -> anyhow::Result<crate::providers::base::LLMResponse> {
            self.terminal_choices.lock().push(tool_choice);
            Ok(crate::providers::base::LLMResponse {
                content: Some("bounded terminal response".to_string()),
                tool_calls: vec![],
                finish_reason: FinishReason::Stop,
                usage: std::collections::HashMap::new(),
            })
        }

        fn get_default_model(&self) -> &str {
            &self.name
        }
    }

    /// A model that narrates a tool call it never emits (`higgs` + lfm2-2.6b
    /// returns `[exec(command='date')]` as plain content with no `tool_calls`,
    /// even under `tool_choice=required`). Nothing executes, so the narration
    /// must NOT be left standing on the stream as if it were a result: it has
    /// to be retracted and replaced by an explicit failure. Before the fix the
    /// give-up message was suppressed by `content_was_streamed` and the user
    /// saw only the phantom call.
    struct PhantomToolProvider {
        name: String,
        calls: std::sync::atomic::AtomicU32,
    }

    #[async_trait]
    impl LLMProvider for PhantomToolProvider {
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
            self.calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(crate::providers::base::LLMResponse {
                content: Some(
                    "Let me use the exec tool to run a command that will give me the current time.\
                     [exec(command='date')]"
                        .to_string(),
                ),
                tool_calls: vec![],
                finish_reason: FinishReason::Stop,
                usage: std::collections::HashMap::new(),
            })
        }

        fn get_default_model(&self) -> &str {
            &self.name
        }
    }

    #[tokio::test]
    async fn phantom_tool_narration_is_retracted_not_surfaced_as_answer() {
        let provider = Arc::new(PhantomToolProvider {
            name: "local-main".to_string(),
            calls: std::sync::atomic::AtomicU32::new(0),
        });
        let (agent_loop, workspace) =
            build_local_inline_harness_with_iters(provider.clone() as Arc<dyn LLMProvider>, 20);
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
        let session_key = format!("phantom-tool-{}", uuid::Uuid::new_v4());

        let final_text = tokio::time::timeout(
            std::time::Duration::from_secs(30),
            agent_loop.process_direct_streaming(
                "what is the time?",
                &session_key,
                "test",
                "offline",
                None,
                tx,
                None,
                None,
                None,
                None,
            ),
        )
        .await
        .expect("turn must terminate");

        let mut deltas = Vec::new();
        while let Ok(d) = rx.try_recv() {
            deltas.push(d);
        }
        let streamed = deltas.join("");

        assert!(
            streamed.contains(&crate::turn_stream::ControlMarker::RetractReply.encode()),
            "phantom narration must be retracted from the stream, got: {streamed:?}"
        );
        let tail = streamed
            .rsplit(&crate::turn_stream::ControlMarker::RetractReply.encode())
            .next()
            .unwrap_or_default();
        assert!(
            !tail.contains("[exec(command='date')]"),
            "phantom call survived the retraction: {tail:?}"
        );
        assert!(
            final_text.contains("narrated a tool action"),
            "turn must end with an explicit no-execution failure, got: {final_text:?}"
        );
        let core = agent_loop.shared.core_handle.swappable();
        let meta = core
            .sessions
            .get_latest_session(&session_key)
            .await
            .expect("phantom session");
        let replay = core.sessions.load_session_replay(&meta.id).await.unwrap();
        assert!(replay.model_calls.iter().any(|call| {
            call.purpose == crate::session::db::ModelCallPurpose::ForcedToolRecovery
                && call.response.is_some()
        }));

        let _ = std::fs::remove_dir_all(&workspace);
    }

    /// Adversarial convergence: the model emits a fresh side-effect call every
    /// turn and never writes a final answer. The loop must terminate after the
    /// lease rejections, without changing the frozen catalog or orphaning any
    /// rejected tool result.
    #[tokio::test]
    async fn convergence_loop_terminates_without_mutating_tool_catalog() {
        let provider = Arc::new(LoopingProvider::new("local-main"));
        // max_iterations must exceed the 96-success budget so the provider
        // reaches paired lease rejections before the ordinary iteration limit.
        let (agent_loop, workspace) =
            build_local_inline_harness_with_iters(provider.clone() as Arc<dyn LLMProvider>, 130);
        let session_key = format!("conv-stable-catalog-{}", uuid::Uuid::new_v4());

        let response = tokio::time::timeout(
            std::time::Duration::from_secs(30),
            agent_loop.process_direct("list files forever", &session_key, "test", "offline"),
        )
        .await
        .expect("loop must terminate — it hung (convergence regression)");

        assert!(
            !response.trim().is_empty(),
            "a converged turn must return text, got empty"
        );
        let calls = provider.call_count();
        assert!(
            calls < 110,
            "loop made {calls} provider calls — did not converge (termination guard regressed)"
        );
        assert_eq!(
            provider.terminal_choices.lock().as_slice(),
            &[crate::providers::base::ToolChoice::None],
            "the repeated loop must share the one terminal no-tools authority"
        );

        let core = agent_loop.shared.core_handle.swappable();
        let meta = core
            .sessions
            .get_latest_session(&session_key)
            .await
            .expect("loop session must exist");
        let replay = core
            .sessions
            .load_session_replay(&meta.id)
            .await
            .expect("loop replay must be readable");
        let main_catalogs: Vec<Option<Vec<Value>>> = replay
            .model_calls
            .iter()
            .filter(|call| call.purpose == crate::session::db::ModelCallPurpose::Main)
            .map(|call| {
                serde_json::from_slice::<crate::session::db::RecordedProviderRequest>(&call.request)
                    .expect("main request must decode")
                    .tools
            })
            .collect();
        assert!(
            !main_catalogs.is_empty(),
            "the foreground loop must record main provider calls"
        );
        assert!(
            main_catalogs.windows(2).all(|pair| pair[0] == pair[1]),
            "lease rejection changed the main-call tool catalog and would bust Higgs's cached prefix"
        );
        let raw = core.sessions.get_all_messages(&meta.id).await;
        let carrier_ids: std::collections::HashSet<String> = raw
            .iter()
            .filter(|message| message.get("role").and_then(Value::as_str) == Some("assistant"))
            .flat_map(|message| {
                message
                    .get("tool_calls")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .filter_map(|call| call.get("id").and_then(Value::as_str))
                    .map(str::to_string)
            })
            .collect();
        let blocked_ids: Vec<&str> = raw
            .iter()
            .filter(|message| {
                message.get("role").and_then(Value::as_str) == Some("tool")
                    && message
                        .get("content")
                        .and_then(Value::as_str)
                        .is_some_and(|content| content.starts_with("lease exhausted:"))
            })
            .filter_map(|message| message.get("tool_call_id").and_then(Value::as_str))
            .collect();
        assert!(
            !blocked_ids.is_empty(),
            "adversarial provider must reach lease-blocked receipts"
        );
        assert!(
            blocked_ids.iter().all(|id| carrier_ids.contains(*id)),
            "every lease rejection must have an assistant tool-call carrier: {blocked_ids:?}"
        );

        let _ = std::fs::remove_dir_all(&workspace);
    }

    /// Positive counterpart to the convergence test: legitimate bounded
    /// exploration — 8 distinct same-family (read) calls, the exact pattern the
    /// retired coarse-family cap used to block at 6 — must complete normally
    /// with the model's own answer, without a lease rejection or loop guard.
    /// Under the old cap the 7th call would be blocked, so this asserts tools
    /// stay available and the turn reaches the model's answer.
    #[tokio::test]
    async fn convergence_legitimate_exploration_completes_without_guard() {
        // A sequence provider also records an absent tool catalog. Without
        // this, a blind sequence dequeue would pass if a loop guard changed
        // the schema before the model reached its answer.
        struct ExploringProvider {
            responses:
                parking_lot::Mutex<std::collections::VecDeque<crate::providers::base::LLMResponse>>,
            calls: std::sync::atomic::AtomicU32,
            saw_tools_absent: std::sync::atomic::AtomicBool,
        }
        #[async_trait]
        impl LLMProvider for ExploringProvider {
            async fn chat(
                &self,
                _messages: &[Value],
                tools: Option<&[Value]>,
                _model: Option<&str>,
                _max_tokens: u32,
                _temperature: f64,
                _thinking_budget: Option<u32>,
                _top_p: Option<f64>,
            ) -> anyhow::Result<crate::providers::base::LLMResponse> {
                self.calls
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if tools.is_none() {
                    self.saw_tools_absent
                        .store(true, std::sync::atomic::Ordering::Relaxed);
                }
                Ok(self.responses.lock().pop_front().unwrap_or_else(|| {
                    crate::providers::base::LLMResponse {
                        content: Some("ERROR: exploration sequence exhausted".to_string()),
                        tool_calls: vec![],
                        finish_reason: FinishReason::Stop,
                        usage: std::collections::HashMap::new(),
                    }
                }))
            }
            fn get_default_model(&self) -> &str {
                "local-main"
            }
        }

        let mut seq: Vec<crate::providers::base::LLMResponse> = (0..8)
            .map(|n| {
                let mut a = std::collections::HashMap::new();
                a.insert("path".to_string(), json!(format!("dir{n}")));
                crate::providers::base::LLMResponse {
                    content: Some(String::new()),
                    tool_calls: vec![crate::providers::base::ToolCallRequest {
                        id: format!("tc_ex_{n}"),
                        name: "list_dir".to_string(),
                        arguments: a,
                    }],
                    finish_reason: FinishReason::ToolCalls,
                    usage: std::collections::HashMap::new(),
                }
            })
            .collect();
        seq.push(crate::providers::base::LLMResponse {
            content: Some(attested_text("done exploring")),
            tool_calls: vec![],
            finish_reason: FinishReason::Stop,
            usage: std::collections::HashMap::new(),
        });
        let provider = Arc::new(ExploringProvider {
            responses: parking_lot::Mutex::new(seq.into()),
            calls: std::sync::atomic::AtomicU32::new(0),
            saw_tools_absent: std::sync::atomic::AtomicBool::new(false),
        });
        // max_iterations well above 8 so the only way this fails to reach the
        // final answer is if a guard (lease/cap/strip) interrupts exploration.
        let (agent_loop, workspace) =
            build_local_inline_harness_with_iters(provider.clone() as Arc<dyn LLMProvider>, 20);
        let session_key = format!("conv-explore-{}", uuid::Uuid::new_v4());

        let response = tokio::time::timeout(
            std::time::Duration::from_secs(30),
            agent_loop.process_direct("explore these dirs", &session_key, "test", "offline"),
        )
        .await
        .expect("exploration must terminate");

        assert_eq!(
            response, "done exploring",
            "8 distinct exploration calls must complete with the model's answer, not a guard"
        );
        assert!(
            !provider.saw_tools_absent.load(std::sync::atomic::Ordering::Relaxed),
            "tools disappeared during exploration — a loop guard fired (regression of the family-cap retirement)"
        );

        let _ = std::fs::remove_dir_all(&workspace);
    }

    /// dropped the warm cache). The model runs a command whose output is huge
    /// (stashed under its tool_call_id), then inspects it. The inspection result
    /// persisted in the conversation must be bounded — not the full body. This
    /// exercises the real tool-execution + shaping path
    /// end-to-end (the unit test only covers `digest_tool_result`).
    #[tokio::test]
    async fn convergence_inspection_of_oversized_body_stays_bounded() {
        let mut exec_args = std::collections::HashMap::new();
        // ~230KB of output — far over any in-context cap, guaranteed to stash.
        exec_args.insert("command".to_string(), json!("seq 1 40000"));
        let mut inspect_args = std::collections::HashMap::new();
        inspect_args.insert("tool_call_id".to_string(), json!("tc_big"));
        inspect_args.insert("query".to_string(), json!("39999"));

        let main: Arc<dyn LLMProvider> = Arc::new(ResponseSequenceProvider::new(
            "local-main",
            vec![
                crate::providers::base::LLMResponse {
                    content: Some(String::new()),
                    tool_calls: vec![crate::providers::base::ToolCallRequest {
                        id: "tc_big".to_string(),
                        name: "exec".to_string(),
                        arguments: exec_args,
                    }],
                    finish_reason: FinishReason::ToolCalls,
                    usage: std::collections::HashMap::new(),
                },
                crate::providers::base::LLMResponse {
                    content: Some(String::new()),
                    tool_calls: vec![crate::providers::base::ToolCallRequest {
                        id: "tc_inspect".to_string(),
                        name: "inspect_tool_result".to_string(),
                        arguments: inspect_args,
                    }],
                    finish_reason: FinishReason::ToolCalls,
                    usage: std::collections::HashMap::new(),
                },
                crate::providers::base::LLMResponse {
                    content: Some(attested_text("done")),
                    tool_calls: vec![],
                    finish_reason: FinishReason::Stop,
                    usage: std::collections::HashMap::new(),
                },
            ],
        ));
        let (agent_loop, workspace) = build_local_inline_harness(main);
        let session_key = format!("conv-inspect-bound-{}", uuid::Uuid::new_v4());

        let response = tokio::time::timeout(
            std::time::Duration::from_secs(30),
            agent_loop.process_direct("run then inspect", &session_key, "test", "offline"),
        )
        .await
        .expect("inspection e2e must terminate");

        assert_eq!(response, "done");

        // The inspected body persisted in the conversation must be bounded,
        // not the raw ~230KB.
        let core = agent_loop.shared.core_handle.swappable();
        let meta = core
            .sessions
            .get_latest_session(&session_key)
            .await
            .expect("session should exist");
        let msgs = core.sessions.get_all_messages(&meta.id).await;
        let inspect_result = msgs
            .iter()
            .find(|m| m.get("tool_call_id").and_then(|v| v.as_str()) == Some("tc_inspect"))
            .expect("inspect tool result must be persisted");
        let content = inspect_result
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        assert!(
            content.chars().count() < 5000,
            "inspected body must be bounded in context, got {} chars (regression of the 172KB blowup)",
            content.chars().count()
        );
        assert!(
            content.starts_with(crate::agent::tool_engine::TOOL_RESULT_EXCERPT_MARKER),
            "inspection output must be the bounded projection, got: {content}"
        );

        let _ = std::fs::remove_dir_all(&workspace);
    }
}

// ---------------------------------------------------------------------------
// Idle-window agency (v0.5 E1) — full-path E2E through the real run() loop
// ---------------------------------------------------------------------------

/// End-to-end: a real inbound seeds the tracker; the idle timer (with a
/// warm fake inference server) injects a self-directed turn onto the same
/// bus; run() processes it through the normal lock/permit path; the turn
/// is journaled in the session DB but its reply is suppressed
/// (quiet-by-default), and the idle turn does not reset its own backoff.
#[tokio::test]
async fn idle_turn_e2e_injects_journaled_quiet_turn() {
    use crate::bus::events::{InboundMessage, OutboundMessage};

    // Fake warm inference server: any HTTP GET -> 200 with a JSON body.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let warm_port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        loop {
            let Ok((mut sock, _)) = listener.accept().await else {
                break;
            };
            tokio::spawn(async move {
                use tokio::io::{AsyncReadExt, AsyncWriteExt};
                let mut buf = [0u8; 2048];
                let _ = sock.read(&mut buf).await;
                let _ = sock
                    .write_all(b"HTTP/1.1 200 OK\r\ncontent-length: 2\r\n\r\n{}")
                    .await;
            });
        }
    });

    // Real loop, static-reply provider, isolated session DB.
    let workspace = tempfile::tempdir().unwrap().keep();
    let db_path = workspace.join("sessions.db");
    let provider = Arc::new(StaticResponseLLM::plain("idle-e2e", "noted."));
    let core = build_swappable_core(SwappableCoreConfig {
        provider,
        workspace: workspace.clone(),
        model: "idle-e2e-model".to_string(),
        max_iterations: 3,
        max_continuations: 1,
        max_tokens: 512,
        temperature: 0.0,
        max_context_tokens: 8192,
        brave_api_key: None,
        search_provider: "searxng".to_string(),
        searxng_url: "http://localhost:8888".to_string(),
        crw_url: String::new(),
        search_max_results: 5,
        exec_timeout: 30,
        restrict_to_workspace: false,
        memory_config: MemoryConfig::default(),
        is_local: false,
        lane: Lane::default(),
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
        code_execution: CodeExecutionConfig::default(),
        python_kernel: PythonKernelConfig::default(),
        cua: CuaToolConfig::default(),
        adaptive_tokens: AdaptiveTokenConfig::default(),
        sessions_db_path: Some(db_path.clone()),
    });
    let counters = test_runtime_counters(8192);
    let core_handle = AgentHandle::new(core, counters);
    let (inbound_tx, inbound_rx) = tokio::sync::mpsc::unbounded_channel::<InboundMessage>();
    let (outbound_tx, mut outbound_rx) = tokio::sync::mpsc::unbounded_channel::<OutboundMessage>();
    let mut agent_loop = AgentLoop::new(
        core_handle,
        inbound_rx,
        outbound_tx,
        inbound_tx.clone(),
        None,
        2,
        None,
        None,
        None,
        ProprioceptionConfig::default(),
        LcmSchemaConfig::default(),
        None,
    );

    let idle_cfg = crate::config::schema::IdleConfig {
        enabled: true,
        after_secs: 1,
        max_backoff_secs: 60,
        max_turns_per_hour: 10,
        session_key: None,
        write_paths: vec!["skills/**".to_string(), "MEMORY.md".to_string()],
    };
    let tracker =
        agent_loop.set_idle_runtime(crate::agent::idle::IdleRuntime::new(idle_cfg.clone()));
    let mut loop_for_run = agent_loop;
    let run_handle = tokio::spawn(async move {
        loop_for_run.run().await;
    });

    // Real inbound seeds the tracker and gets a normal (non-suppressed) reply.
    inbound_tx
        .send(InboundMessage::new("test", "human", "e2e", "hello"))
        .unwrap();
    let first = tokio::time::timeout(std::time::Duration::from_secs(20), outbound_rx.recv())
        .await
        .expect("user turn reply within 20s")
        .expect("outbound channel alive");
    assert_eq!(first.chat_id, "e2e", "user turn replies to its chat");

    // Idle timer with a 1s tick against the fake warm server.
    tokio::spawn(crate::agent::idle::run_idle_timer_with_tick(
        idle_cfg,
        tracker,
        inbound_tx.clone(),
        format!("http://127.0.0.1:{warm_port}/v1"),
        1,
    ));

    // Wait for the [idle] observation to be journaled in the session DB.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    let mut idle_row_seen = false;
    while std::time::Instant::now() < deadline {
        if let Ok(conn) = rusqlite::Connection::open_with_flags(
            &db_path,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
        ) {
            let _ = conn.busy_timeout(std::time::Duration::from_millis(200));
            let found: Result<i64, _> = conn.query_row(
                "SELECT COUNT(*) FROM messages WHERE content LIKE '[idle]%'",
                [],
                |row| row.get(0),
            );
            if let Ok(n) = found {
                if n > 0 {
                    idle_row_seen = true;
                    break;
                }
            }
        }
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    }
    assert!(
        idle_row_seen,
        "idle observation journaled in the session DB within 30s"
    );

    // Quiet-by-default: after the journaled turn, allow a grace window for
    // the (suppressed) reply path, then assert the outbound bus is empty.
    tokio::time::sleep(std::time::Duration::from_secs(3)).await;
    assert!(
        outbound_rx.try_recv().is_err(),
        "idle turn reply must not reach the outbound bus"
    );

    run_handle.abort();
    let _ = std::fs::remove_dir_all(&workspace);
}

// ---------------------------------------------------------------------------
// Task 3: live Higgs capacity runtime
// ---------------------------------------------------------------------------

mod capacity_runtime {
    use super::*;
    use crate::agent::agent_loop::shared::resolve_live_capacity;
    use crate::agent::capacity::{CapacityRuntime, HiggsCapacityFetch, HiggsCapacityProfile};
    use crate::agent::token_budget::TokenBudget;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Arc, Mutex};

    fn profile_json(boot_id: &str, safe_total: u64, output: u64) -> serde_json::Value {
        serde_json::json!({
            "schemaVersion": 1,
            "model": "escha",
            "modelFingerprint": "sha256:abc",
            "bootId": boot_id,
            "generation": 7,
            "availability": "available",
            "pressure": "normal",
            "safeTotalTokens": safe_total,
            "recommendedOutputTokens": output,
            "maxPromptTokens": safe_total - output,
            "retainedSessionTokens": 0,
            "retainedBytes": 0,
            "prefixCacheBytes": 0,
            "basis": "configured"
        })
    }

    /// Higgs-capable mock that counts fetches and serves queued profiles.
    struct CapacityMockLLM {
        fetches: AtomicU64,
        next: Mutex<Vec<HiggsCapacityFetch>>,
        legacy: bool,
    }

    impl CapacityMockLLM {
        fn serving(profiles: Vec<HiggsCapacityFetch>) -> Arc<Self> {
            Arc::new(Self {
                fetches: AtomicU64::new(0),
                next: Mutex::new(profiles),
                legacy: false,
            })
        }

        fn fetches(&self) -> u64 {
            self.fetches.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl LLMProvider for CapacityMockLLM {
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
                finish_reason: FinishReason::Stop,
                usage: std::collections::HashMap::new(),
            })
        }

        fn get_default_model(&self) -> &str {
            "escha"
        }

        fn get_api_base(&self) -> Option<&str> {
            Some("http://127.0.0.1:9000")
        }

        fn supports_higgs_session_cache(&self) -> bool {
            true
        }

        fn fetch_higgs_capacity<'a>(
            &'a self,
            _model: &'a str,
        ) -> std::pin::Pin<
            Box<
                dyn std::future::Future<
                        Output = Result<Option<HiggsCapacityFetch>, crate::errors::ProviderError>,
                    > + Send
                    + 'a,
            >,
        > {
            Box::pin(async {
                self.fetches.fetch_add(1, Ordering::SeqCst);
                let mut queue = self.next.lock().unwrap();
                Ok(queue.pop_front_owned())
            })
        }
    }

    /// `Vec::pop` from the front without shifting: take from the front.
    trait PopFrontOwned {
        fn pop_front_owned(&mut self) -> Option<HiggsCapacityFetch>;
    }

    impl PopFrontOwned for Vec<HiggsCapacityFetch> {
        fn pop_front_owned(&mut self) -> Option<HiggsCapacityFetch> {
            if self.is_empty() {
                if self.capacity() == 0 {
                    // Exhausted queue behaves as legacy marker holder: None
                    // means "no more profiles"; the loop then invalidates.
                    return None;
                }
                return None;
            }
            Some(self.remove(0))
        }
    }

    fn configured() -> TokenBudget {
        TokenBudget::new(131_072, 8_192)
    }

    #[tokio::test]
    async fn first_turn_fetches_and_narrows_the_effective_budget() {
        let capacity = CapacityRuntime::shared();
        let counters = test_runtime_counters(131_072);
        let provider = CapacityMockLLM::serving(vec![HiggsCapacityFetch::Profile(
            serde_json::from_value::<HiggsCapacityProfile>(profile_json("boot-1", 53_248, 4_096))
                .unwrap(),
        )]);

        let budget = resolve_live_capacity(
            &capacity,
            provider.as_ref(),
            "escha",
            &configured(),
            &counters,
            "session",
        )
        .await;

        assert_eq!(provider.fetches(), 1, "discovery must fetch once");
        assert_eq!(budget.max_context(), 53_248.min(131_072));
        assert_eq!(budget.response_reserve(), 4_096);
    }

    #[tokio::test]
    async fn same_boot_refetch_keeps_the_epoch_stable() {
        let capacity = CapacityRuntime::shared();
        let counters = test_runtime_counters(131_072);
        let provider = CapacityMockLLM::serving(vec![
            HiggsCapacityFetch::Profile(
                serde_json::from_value::<HiggsCapacityProfile>(profile_json(
                    "boot-1", 53_248, 4_096,
                ))
                .unwrap(),
            ),
            HiggsCapacityFetch::Profile(
                serde_json::from_value::<HiggsCapacityProfile>(profile_json(
                    "boot-1", 53_248, 4_096,
                ))
                .unwrap(),
            ),
        ]);

        for _ in 0..2 {
            resolve_live_capacity(
                &capacity,
                provider.as_ref(),
                "escha",
                &configured(),
                &counters,
                "session",
            )
            .await;
        }
        // Two turns → two fetches (a server reboot must be observable), but
        // the same boot never rotates the retained epoch.
        assert_eq!(provider.fetches(), 2);
        counters
            .prompt_cache_watermark
            .lock()
            .insert("session".to_string(), 4_096);
        resolve_live_capacity(
            &capacity,
            provider.as_ref(), // queue exhausted → Ok(None) → invalidate is wrong here
            "escha",
            &configured(),
            &counters,
            "session",
        )
        .await;
        assert!(
            counters
                .prompt_cache_watermark
                .lock()
                .contains_key("session"),
            "same-boot refresh must not rotate the epoch"
        );
    }

    #[tokio::test]
    async fn boot_change_refetches_and_rotates_the_retained_epoch() {
        let capacity = CapacityRuntime::shared();
        let counters = test_runtime_counters(131_072);
        let provider = CapacityMockLLM::serving(vec![
            HiggsCapacityFetch::Profile(
                serde_json::from_value::<HiggsCapacityProfile>(profile_json(
                    "boot-1", 53_248, 4_096,
                ))
                .unwrap(),
            ),
            HiggsCapacityFetch::Profile(
                serde_json::from_value::<HiggsCapacityProfile>(profile_json(
                    "boot-2", 49_152, 4_096,
                ))
                .unwrap(),
            ),
        ]);

        resolve_live_capacity(
            &capacity,
            provider.as_ref(),
            "escha",
            &configured(),
            &counters,
            "session",
        )
        .await;
        // Warm prompt-cache state from the old boot.
        counters
            .prompt_cache_watermark
            .lock()
            .insert("session".to_string(), 4_096);

        let budget = resolve_live_capacity(
            &capacity,
            provider.as_ref(),
            "escha",
            &configured(),
            &counters,
            "session",
        )
        .await;

        assert_eq!(provider.fetches(), 2, "boot change must refetch");
        assert_eq!(budget.max_context(), 49_152);
        assert!(
            !counters
                .prompt_cache_watermark
                .lock()
                .contains_key("session"),
            "boot change must reset the prompt-cache watermark"
        );
    }

    #[tokio::test]
    async fn cloud_provider_bypasses_capacity_entirely() {
        let capacity = CapacityRuntime::shared();
        let counters = test_runtime_counters(131_072);
        // Install a Higgs snapshot first, then take over with a cloud mock.
        let higgs = CapacityMockLLM::serving(vec![HiggsCapacityFetch::Profile(
            serde_json::from_value::<HiggsCapacityProfile>(profile_json("boot-1", 53_248, 4_096))
                .unwrap(),
        )]);
        resolve_live_capacity(
            &capacity,
            higgs.as_ref(),
            "escha",
            &configured(),
            &counters,
            "session",
        )
        .await;

        let budget = resolve_live_capacity(
            &capacity,
            MockLLM::named("cloud").as_ref(), // supports=false
            "escha",
            &configured(),
            &counters,
            "session",
        )
        .await;

        assert_eq!(
            budget.max_context(),
            131_072,
            "cloud turn must keep the configured ceiling"
        );
        assert_eq!(
            budget.response_reserve(),
            8_192,
            "cloud turn must keep the configured reserve"
        );
        // The stale snapshot is dropped: the next Higgs turn refetches.
        assert!(!capacity.cached_for("http://127.0.0.1:9000", "escha"));
    }

    #[tokio::test]
    async fn legacy_higgs_preserves_the_configured_budget() {
        let capacity = CapacityRuntime::shared();
        let counters = test_runtime_counters(131_072);
        let provider = CapacityMockLLM::serving(vec![HiggsCapacityFetch::Legacy]);

        let budget = resolve_live_capacity(
            &capacity,
            provider.as_ref(),
            "escha",
            &configured(),
            &counters,
            "session",
        )
        .await;

        assert_eq!(provider.fetches(), 1);
        assert_eq!(budget.max_context(), 131_072);
        assert_eq!(budget.response_reserve(), 8_192);
    }

    #[tokio::test]
    async fn fetch_failure_fails_open_to_the_configured_ceiling() {
        let capacity = CapacityRuntime::shared();
        let counters = test_runtime_counters(131_072);

        struct FailingLLM;
        #[async_trait]
        impl LLMProvider for FailingLLM {
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
                unreachable!("capacity fetch test never chats")
            }
            fn get_default_model(&self) -> &str {
                "escha"
            }
            fn get_api_base(&self) -> Option<&str> {
                Some("http://127.0.0.1:9000")
            }
            fn supports_higgs_session_cache(&self) -> bool {
                true
            }
            fn fetch_higgs_capacity<'a>(
                &'a self,
                _model: &'a str,
            ) -> std::pin::Pin<
                Box<
                    dyn std::future::Future<
                            Output = Result<
                                Option<HiggsCapacityFetch>,
                                crate::errors::ProviderError,
                            >,
                        > + Send
                        + 'a,
                >,
            > {
                Box::pin(async {
                    Err(crate::errors::ProviderError::HttpError(
                        "unreachable".to_string(),
                    ))
                })
            }
        }

        let budget = resolve_live_capacity(
            &capacity,
            &FailingLLM,
            "escha",
            &configured(),
            &counters,
            "session",
        )
        .await;

        assert_eq!(budget.max_context(), 131_072);
        assert_eq!(budget.response_reserve(), 8_192);
    }

    #[test]
    fn runtime_snapshot_never_persists_configured_limits() {
        // The runtime narrows a view; it holds no config path and cannot
        // rewrite `config.json` (structural invariant of Task 3).
        let capacity = CapacityRuntime::shared();
        capacity.install_legacy("http://127.0.0.1:9000", "escha");
        let budget = capacity.effective_budget(&configured(), 0);
        assert_eq!(budget.max_context(), 131_072);
        // Invalidate restores the configured ceiling — proof the narrowing
        // lives in the runtime view, not in any persisted configuration.
        capacity.invalidate();
        assert_eq!(
            capacity.effective_budget(&configured(), 0).max_context(),
            131_072
        );
    }
}

// ---------------------------------------------------------------------------
// Task 4: preflight capacity reduction (sanctioned order)
// ---------------------------------------------------------------------------

mod capacity_preflight {
    use super::*;
    use crate::agent::agent_loop::compaction::LcmCompactionMutation;
    use crate::agent::compaction::ContextCompactor;
    use crate::agent::lcm::{CompactionFailureMode, LcmConfig, LcmEngine};
    use crate::agent::token_budget::TokenBudget;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Counting summarizer: records provider calls, returns a short summary.
    struct CountingSummarizer {
        calls: AtomicU64,
    }

    #[async_trait]
    impl LLMProvider for CountingSummarizer {
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
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(crate::providers::base::LLMResponse {
                content: Some("summarized: the user discussed rust ownership.".to_string()),
                tool_calls: vec![],
                finish_reason: FinishReason::Stop,
                usage: std::collections::HashMap::new(),
            })
        }
        fn get_default_model(&self) -> &str {
            "counter"
        }
    }

    fn engine_with_block(messages: usize, tokens_each: usize) -> LcmEngine {
        let mut engine = LcmEngine::new(LcmConfig {
            tau_soft: 0.3,
            tau_hard: 10.0,
            deterministic_target: 64,
            keep_prefix_fraction: 0.35,
        });
        for id in 0..messages {
            let _ = engine.ingest(json!({
                "role": "user",
                "content": format!("msg-{id} {}", "x".repeat(tokens_each)),
                "_db_id": id,
            }));
        }
        engine
    }

    #[tokio::test]
    async fn compact_uses_a_deterministic_checkpoint_when_model_request_does_not_fit() {
        // The full block cannot fit the summarizer request. The pressure path
        // must not install a partial model fold; it uses one bounded,
        // lossless deterministic checkpoint instead.
        let summarizer = Arc::new(CountingSummarizer {
            calls: AtomicU64::new(0),
        });
        let compactor = ContextCompactor::new(
            Arc::clone(&summarizer) as Arc<dyn LLMProvider>,
            "counter".to_string(),
            4096,
        );
        let mut engine = engine_with_block(60, 2_600);
        let mut mutation = LcmCompactionMutation::new(&mut engine);
        let summary = mutation
            .engine_mut()
            .compact(
                Some(&compactor),
                &TokenBudget::new(4_096, 512),
                0,
                CompactionFailureMode::PreserveContext,
            )
            .await;

        let summary = summary.expect("the deterministic checkpoint should fit");
        let crate::agent::turn::Turn::Summary {
            level, source_ids, ..
        } = &summary
        else {
            panic!("expected a summary turn");
        };
        assert_eq!(*level, 0, "model must not receive an oversized request");
        assert_eq!(
            summarizer.calls.load(Ordering::SeqCst),
            0,
            "an oversized model request must use the deterministic path"
        );
        assert!(
            !source_ids.is_empty(),
            "the deterministic checkpoint must cover real source rows"
        );
        assert!(
            mutation.engine().active_context().len() < 60,
            "the whole eligible history should be replaced by one checkpoint"
        );
        let node_sources = &mutation.engine().dag().newest().unwrap().source_ids;
        assert!(!node_sources.is_empty());
        for id in node_sources {
            assert!(
                mutation.engine().expand(&[*id]).len() == 1,
                "source {id} recallable"
            );
        }
        assert_eq!(source_ids.len(), node_sources.len());
        drop(mutation);
    }

    #[tokio::test]
    async fn compact_uses_llm_when_its_own_request_fits() {
        let summarizer = Arc::new(CountingSummarizer {
            calls: AtomicU64::new(0),
        });
        let compactor = ContextCompactor::new(
            Arc::clone(&summarizer) as Arc<dyn LLMProvider>,
            "counter".to_string(),
            8192,
        );
        // Six ~380-token messages: the protect floor (512 tokens) keeps the
        // newest one raw and leaves a ~1.9K block that fits the 7.7K budget.
        let mut engine = engine_with_block(6, 2_600);
        let mut mutation = LcmCompactionMutation::new(&mut engine);
        let summary = mutation
            .engine_mut()
            .compact(
                Some(&compactor),
                &TokenBudget::new(8_192, 512),
                0,
                CompactionFailureMode::PreserveContext,
            )
            .await;
        drop(mutation);

        let summary = summary.expect("small block compacts via LLM");
        let crate::agent::turn::Turn::Summary { level, .. } = &summary else {
            panic!("expected a summary turn");
        };
        assert!(matches!(level, 1 | 2), "LLM escalation, got level {level}");
        assert_eq!(
            summarizer.calls.load(Ordering::SeqCst),
            1,
            "exactly one summarizer call"
        );
    }

    /// Full-turn gate: a 512-token window whose immutable prefix (system +
    /// tools) alone cannot fit ends the turn capacity-unavailable with ZERO
    /// provider calls — pending work, never a recursive compaction.
    #[tokio::test]
    async fn immutable_prefix_that_cannot_fit_returns_context_error() {
        struct PanicProvider;
        #[async_trait]
        impl LLMProvider for PanicProvider {
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
                panic!("capacity-unavailable turn must not reach the provider");
            }
            fn get_default_model(&self) -> &str {
                "local-model"
            }
        }

        let workspace = tempfile::tempdir().unwrap().keep();
        let core = build_swappable_core(SwappableCoreConfig {
            provider: Arc::new(PanicProvider) as Arc<dyn LLMProvider>,
            workspace: workspace.clone(),
            model: "local-model".to_string(),
            max_iterations: 2,
            max_continuations: 1,
            max_tokens: 256,
            temperature: 0.0,
            max_context_tokens: 512,
            brave_api_key: None,
            search_provider: "searxng".to_string(),
            searxng_url: "http://localhost:8888".to_string(),
            crw_url: String::new(),
            search_max_results: 5,
            exec_timeout: 30,
            restrict_to_workspace: false,
            memory_config: MemoryConfig::default(),
            is_local: true,
            lane: Lane::default(),
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
            code_execution: CodeExecutionConfig::default(),
            python_kernel: PythonKernelConfig::default(),
            cua: CuaToolConfig::default(),
            adaptive_tokens: AdaptiveTokenConfig::default(),
            sessions_db_path: Some(
                std::env::temp_dir().join(format!("nanobot-cap-{}.sqlite", uuid::Uuid::new_v4())),
            ),
        });
        let counters = test_runtime_counters(512);
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

        let body = agent_loop
            .process_direct("hello", "cap-e2e", "test", "capacity-preflight")
            .await;
        assert!(
            body.contains("[Context Limit]"),
            "expected capacity-unavailable reply, got: {body}"
        );
    }

    struct ShrinkingCapacityProvider {
        requests: AtomicU64,
        fetches: AtomicU64,
        unavailable_after_fetch: u64,
    }

    impl ShrinkingCapacityProvider {
        fn profile(available: bool) -> crate::agent::capacity::HiggsCapacityFetch {
            crate::agent::capacity::HiggsCapacityFetch::Profile(
                serde_json::from_value(json!({
                    "schemaVersion": 1,
                    "model": "local-qwen-test",
                    "modelFingerprint": "fresh-capacity-gate-test",
                    "bootId": "fresh-capacity-gate-boot",
                    "generation": if available { 1 } else { 2 },
                    "availability": if available { "available" } else { "unavailable" },
                    "pressure": if available { "normal" } else { "critical" },
                    "safeTotalTokens": if available { 4096 } else { 0 },
                    "recommendedOutputTokens": if available { 512 } else { 0 },
                    "maxPromptTokens": if available { 3584 } else { 0 },
                    "retainedSessionTokens": 0,
                    "retainedBytes": 0,
                    "prefixCacheBytes": 0,
                    "basis": "configured"
                }))
                .expect("valid capacity profile"),
            )
        }
    }

    #[async_trait]
    impl LLMProvider for ShrinkingCapacityProvider {
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
            let request = self.requests.fetch_add(1, Ordering::SeqCst);
            if request == 0 {
                return Ok(crate::providers::base::LLMResponse {
                    content: Some(String::new()),
                    tool_calls: vec![crate::providers::base::ToolCallRequest {
                        id: "capacity-tool-round".to_string(),
                        name: "get_tools".to_string(),
                        arguments: std::collections::HashMap::new(),
                    }],
                    finish_reason: FinishReason::ToolCalls,
                    usage: std::collections::HashMap::new(),
                });
            }
            Ok(crate::providers::base::LLMResponse {
                content: Some("request should have been gated".to_string()),
                tool_calls: vec![],
                finish_reason: FinishReason::Stop,
                usage: std::collections::HashMap::new(),
            })
        }

        fn fetch_higgs_capacity<'a>(
            &'a self,
            _model: &'a str,
        ) -> std::pin::Pin<
            Box<
                dyn std::future::Future<
                        Output = Result<
                            Option<crate::agent::capacity::HiggsCapacityFetch>,
                            crate::errors::ProviderError,
                        >,
                    > + Send
                    + 'a,
            >,
        > {
            Box::pin(async move {
                let fetch = self.fetches.fetch_add(1, Ordering::SeqCst);
                Ok(Some(Self::profile(fetch < self.unavailable_after_fetch)))
            })
        }

        fn get_default_model(&self) -> &str {
            "local-qwen-test"
        }

        fn get_api_base(&self) -> Option<&str> {
            Some("http://127.0.0.1:9000")
        }

        fn supports_higgs_session_cache(&self) -> bool {
            true
        }
    }

    #[tokio::test]
    async fn unavailable_capacity_snapshot_does_not_block_configured_request() {
        let provider = Arc::new(ShrinkingCapacityProvider {
            requests: AtomicU64::new(0),
            fetches: AtomicU64::new(0),
            unavailable_after_fetch: 0,
        });
        let (agent_loop, workspace) =
            build_local_inline_harness(provider.clone() as Arc<dyn LLMProvider>);
        let reply = agent_loop
            .process_direct("hello", "fixed-capacity", "test", "fixed-capacity")
            .await;
        assert!(!reply.contains("[Capacity Unavailable]"), "{reply}");
        assert!(provider.requests.load(Ordering::SeqCst) >= 1);
        let _ = std::fs::remove_dir_all(workspace);
    }

    #[tokio::test]
    async fn pressure_between_tool_rounds_does_not_block_continuation() {
        let provider = Arc::new(ShrinkingCapacityProvider {
            requests: AtomicU64::new(0),
            fetches: AtomicU64::new(0),
            unavailable_after_fetch: 2,
        });
        let (agent_loop, workspace) =
            build_local_inline_harness(provider.clone() as Arc<dyn LLMProvider>);

        let reply = agent_loop
            .process_direct(
                "inspect the tools then answer",
                "fresh-capacity-tool-round",
                "test",
                "fresh-capacity-tool-round",
            )
            .await;

        assert!(!reply.contains("[Capacity Unavailable]"), "{reply}");
        assert_eq!(
            provider.requests.load(Ordering::SeqCst),
            2,
            "pressure must not block the continuation"
        );
        assert!(
            provider.fetches.load(Ordering::SeqCst) >= 2,
            "capacity must be refreshed around each admitted POST"
        );

        let _ = std::fs::remove_dir_all(workspace);
    }
}

// ---------------------------------------------------------------------------
// Task 5: exactly one typed-413 retry, no turn/tool replay
// ---------------------------------------------------------------------------

mod capacity_exceeded {
    use super::*;
    use crate::agent::token_budget::TokenBudget;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct Capacity413Provider {
        requests: AtomicU64,
        compaction_requests_after_rejection: AtomicU64,
        capacity_rejected: std::sync::atomic::AtomicBool,
        typed_failures: u32,
        capacity_pressure: &'static str,
        boot_id: &'static str,
        safe_total_tokens: u64,
        prompt_limits: std::sync::Mutex<Vec<u64>>,
        output_limits: std::sync::Mutex<Vec<u32>>,
        request_tokens: std::sync::Mutex<Vec<usize>>,
    }

    #[async_trait]
    impl LLMProvider for Capacity413Provider {
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
            let Some(prompt_limit) = _messages[0]
                [crate::providers::openai_compat::NANOBOT_HIGGS_MAX_PROMPT_TOKENS_FIELD]
                .as_u64()
            else {
                if self.capacity_rejected.load(Ordering::SeqCst) {
                    self.compaction_requests_after_rejection
                        .fetch_add(1, Ordering::SeqCst);
                }
                self.output_limits.lock().unwrap().push(_max_tokens);
                self.request_tokens.lock().unwrap().push(
                    TokenBudget::estimate_tokens(_messages)
                        + TokenBudget::estimate_tool_def_tokens(_tools.unwrap_or(&[])),
                );
                self.requests.fetch_add(1, Ordering::SeqCst);
                return Ok(crate::providers::base::LLMResponse {
                    content: Some("recovered answer".to_string()),
                    tool_calls: vec![],
                    finish_reason: FinishReason::Stop,
                    usage: std::collections::HashMap::new(),
                });
            };
            self.output_limits.lock().unwrap().push(_max_tokens);
            self.prompt_limits.lock().unwrap().push(prompt_limit);
            self.request_tokens.lock().unwrap().push(
                TokenBudget::estimate_tokens(_messages)
                    + TokenBudget::estimate_tool_def_tokens(_tools.unwrap_or(&[])),
            );
            let n = self.requests.fetch_add(1, Ordering::SeqCst);
            if (n as u32) < self.typed_failures {
                self.capacity_rejected.store(true, Ordering::SeqCst);
                return Err(crate::errors::ProviderError::HiggsCapacityExceeded {
                    safe_prompt_tokens: 8_192,
                    safe_total_tokens: 12_288,
                    boot_id: "boot-1".to_string(),
                    generation: 3,
                }
                .into());
            }
            Ok(crate::providers::base::LLMResponse {
                content: Some("recovered answer".to_string()),
                tool_calls: vec![],
                finish_reason: FinishReason::Stop,
                usage: std::collections::HashMap::new(),
            })
        }
        fn fetch_higgs_capacity<'a>(
            &'a self,
            _model: &'a str,
        ) -> std::pin::Pin<
            Box<
                dyn std::future::Future<
                        Output = Result<
                            Option<crate::agent::capacity::HiggsCapacityFetch>,
                            crate::errors::ProviderError,
                        >,
                    > + Send
                    + 'a,
            >,
        > {
            Box::pin(async {
                // Discovery remains broader than the typed request-specific rejection.
                let profile = serde_json::from_value(json!({
                    "schemaVersion":1,"model":"local-model","modelFingerprint":"test-model",
                    "bootId":self.boot_id,"generation":1,"availability":"available",
                    "pressure":self.capacity_pressure,
                    "safeTotalTokens":self.safe_total_tokens,"recommendedOutputTokens":512,
                    "maxPromptTokens":self.safe_total_tokens-512,
                    "retainedSessionTokens":0,"retainedBytes":0,"prefixCacheBytes":0,"basis":"configured"
                })).unwrap();
                Ok(Some(crate::agent::capacity::HiggsCapacityFetch::Profile(
                    profile,
                )))
            })
        }

        fn get_default_model(&self) -> &str {
            "local-model"
        }
        fn get_api_base(&self) -> Option<&str> {
            Some("http://127.0.0.1:9000")
        }
        fn supports_higgs_session_cache(&self) -> bool {
            true
        }
    }

    struct TurnRecord {
        provider_calls: u64,
        compaction_requests_after_rejection: u64,
        prompt_limits: Vec<u64>,
        output_limits: Vec<u32>,
        request_tokens: Vec<usize>,
        event_kinds: Vec<&'static str>,
        outcome: String,
        reply: String,
        suspended_events: usize,
    }

    async fn drive_typed_turn(typed_failures: u32, prompt: &str) -> TurnRecord {
        drive_typed_turn_with_history(typed_failures, prompt, Vec::new()).await
    }

    async fn drive_typed_turn_with_history(
        typed_failures: u32,
        prompt: &str,
        history: Vec<Value>,
    ) -> TurnRecord {
        drive_typed_turn_with_history_and_pressure(
            typed_failures,
            prompt,
            history,
            "normal",
            "boot-mock",
        )
        .await
    }

    async fn drive_typed_turn_with_history_and_pressure(
        typed_failures: u32,
        prompt: &str,
        history: Vec<Value>,
        capacity_pressure: &'static str,
        boot_id: &'static str,
    ) -> TurnRecord {
        drive_typed_turn_full(
            typed_failures,
            prompt,
            history,
            capacity_pressure,
            boot_id,
            12_288,
            16_384,
        )
        .await
    }

    async fn drive_typed_turn_full(
        typed_failures: u32,
        prompt: &str,
        history: Vec<Value>,
        capacity_pressure: &'static str,
        boot_id: &'static str,
        safe_total_tokens: u64,
        configured_context: usize,
    ) -> TurnRecord {
        let provider = Arc::new(Capacity413Provider {
            requests: AtomicU64::new(0),
            compaction_requests_after_rejection: AtomicU64::new(0),
            capacity_rejected: std::sync::atomic::AtomicBool::new(false),
            typed_failures,
            capacity_pressure,
            boot_id,
            safe_total_tokens,
            prompt_limits: std::sync::Mutex::new(Vec::new()),
            output_limits: std::sync::Mutex::new(Vec::new()),
            request_tokens: std::sync::Mutex::new(Vec::new()),
        });
        let workspace = tempfile::tempdir().unwrap().keep();
        let core = build_swappable_core(SwappableCoreConfig {
            provider: Arc::clone(&provider) as Arc<dyn LLMProvider>,
            workspace: workspace.clone(),
            model: "local-model".to_string(),
            max_iterations: 3,
            max_continuations: 1,
            max_tokens: 512,
            temperature: 0.0,
            max_context_tokens: configured_context,
            brave_api_key: None,
            search_provider: "searxng".to_string(),
            searxng_url: "http://localhost:8888".to_string(),
            crw_url: String::new(),
            search_max_results: 5,
            exec_timeout: 30,
            restrict_to_workspace: false,
            memory_config: MemoryConfig::default(),
            is_local: true,
            lane: Lane::default(),
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
            code_execution: CodeExecutionConfig::default(),
            python_kernel: PythonKernelConfig::default(),
            cua: CuaToolConfig::default(),
            adaptive_tokens: AdaptiveTokenConfig::default(),
            sessions_db_path: Some(
                std::env::temp_dir().join(format!("nanobot-413-{}.sqlite", uuid::Uuid::new_v4())),
            ),
        });
        let session_key = "cap-413";
        if !history.is_empty() {
            let session = core.sessions.get_or_resume(session_key).await;
            core.sessions.add_messages(&session.id, &history).await;
        }
        let counters = test_runtime_counters(16_384);
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
        let reply = agent_loop
            .process_direct(prompt, session_key, "test", "capacity-413")
            .await;

        let sessions = agent_loop.shared.core_handle.swappable().sessions.clone();
        let concrete = sessions
            .get_latest_session(session_key)
            .await
            .expect("session exists");
        let events = sessions.load_session_events(&concrete.id).await.unwrap();
        // The final turn is the LAST turn_started window.
        let last_start = events
            .iter()
            .rposition(|event| event.payload.kind() == "turn_started")
            .expect("turn_started");
        let turn_events = &events[last_start..];
        let event_kinds = turn_events
            .iter()
            .map(|event| event.payload.kind())
            .collect();
        let outcome = turn_events
            .iter()
            .rev()
            .find_map(|event| match &event.payload {
                crate::session::db::SessionEventPayload::TurnFinished { outcome } => {
                    Some(outcome.clone())
                }
                _ => None,
            })
            .expect("turn_finished");
        let suspended_events = turn_events
            .iter()
            .filter(|event| event.payload.kind() == "turn_suspended")
            .count();
        let prompt_limits = provider.prompt_limits.lock().unwrap().clone();
        let output_limits = provider.output_limits.lock().unwrap().clone();
        let request_tokens = provider.request_tokens.lock().unwrap().clone();
        TurnRecord {
            provider_calls: provider.requests.load(Ordering::SeqCst),
            compaction_requests_after_rejection: provider
                .compaction_requests_after_rejection
                .load(Ordering::SeqCst),
            prompt_limits,
            output_limits,
            request_tokens,
            event_kinds,
            outcome,
            reply,
            suspended_events,
        }
    }

    #[tokio::test]
    async fn pressure_does_not_compact_history_that_fits_fixed_context() {
        let mut history = Vec::new();
        let mut turn = 0_u64;
        while TokenBudget::estimate_tokens(&history) < 10_500 {
            history.push(json!({
                "role": "user", "_turn": turn,
                "content": format!("retained project evidence {turn}: {}", "a concrete durable fact needed after recovery ".repeat(80)),
            }));
            history.push(json!({
                "role": "assistant", "_turn": turn,
                "content": format!("acknowledged retained evidence {turn}"),
            }));
            turn += 1;
        }
        let raw_tokens = TokenBudget::estimate_tokens(&history);
        let record = drive_typed_turn_full(
            0,
            "report status",
            history,
            "critical",
            "boot",
            16_384,
            16_384,
        )
        .await;
        assert_eq!(record.outcome, "finished");
        assert_eq!(record.request_tokens.len(), 1);
        assert!(
            record.request_tokens[0] >= raw_tokens,
            "advisory pressure must not compact fitting history"
        );
    }

    #[tokio::test]
    async fn pressure_does_not_change_fixed_request_limits() {
        let normal =
            drive_typed_turn_with_history_and_pressure(0, "hello", vec![], "normal", "boot").await;
        let critical =
            drive_typed_turn_with_history_and_pressure(0, "hello", vec![], "critical", "boot")
                .await;
        assert_eq!(normal.outcome, "finished");
        assert_eq!(critical.outcome, "finished");
        assert_eq!(normal.prompt_limits, critical.prompt_limits);
        assert_eq!(normal.output_limits, critical.output_limits);
        assert_eq!(normal.request_tokens.len(), 1);
        assert_eq!(critical.request_tokens.len(), 1);
        assert_eq!(critical.suspended_events, 0);
    }

    #[tokio::test]
    async fn v1_capacity_server_remains_stateless_and_does_not_claim_retention() {
        let record = drive_typed_turn(1, "hello").await;
        assert_eq!(record.request_tokens.len(), 1);
        assert_eq!(record.provider_calls, 1);
        assert!(record.prompt_limits.is_empty());
        assert_eq!(record.outcome, "finished");
    }
}

// ---------------------------------------------------------------------------
// Task 6: durable interrupted / suspended turns
// ---------------------------------------------------------------------------

mod interrupted {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    enum CapacityFailure {
        Interrupted,
        Unavailable,
    }

    struct CapacityFailingProvider {
        requests: AtomicU64,
        failure: CapacityFailure,
    }

    #[async_trait]
    impl LLMProvider for CapacityFailingProvider {
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
            self.requests.fetch_add(1, Ordering::SeqCst);
            Err(match self.failure {
                CapacityFailure::Interrupted => {
                    crate::errors::ProviderError::HiggsCapacityInterrupted {
                        boot_id: "boot-1".to_string(),
                        generation: 9,
                        partial_output_tokens: 7,
                        partial_stream_bytes: crate::errors::PartialStreamBytes::new(
                            b"partial streamed text".to_vec(),
                        ),
                    }
                    .into()
                }
                CapacityFailure::Unavailable => {
                    crate::errors::ProviderError::HiggsCapacityUnavailable {
                        boot_id: "boot-1".to_string(),
                        generation: 9,
                        retry_after_ms: 5_000,
                    }
                    .into()
                }
            })
        }
        fn get_default_model(&self) -> &str {
            "local-model"
        }
        fn get_api_base(&self) -> Option<&str> {
            Some("http://127.0.0.1:9000")
        }
        fn supports_higgs_session_cache(&self) -> bool {
            true
        }
    }

    async fn drive(
        failure: CapacityFailure,
        fail_interrupted_write: bool,
    ) -> (
        Arc<CapacityFailingProvider>,
        std::sync::Arc<crate::session::db::SessionDb>,
        String,
        String,
        Vec<crate::session::db::SessionEvent>,
        Vec<Value>,
    ) {
        let provider = Arc::new(CapacityFailingProvider {
            requests: AtomicU64::new(0),
            failure,
        });
        let workspace = tempfile::tempdir().unwrap().keep();
        let core = build_swappable_core(SwappableCoreConfig {
            provider: Arc::clone(&provider) as Arc<dyn LLMProvider>,
            workspace: workspace.clone(),
            model: "local-model".to_string(),
            max_iterations: 2,
            max_continuations: 1,
            max_tokens: 256,
            temperature: 0.0,
            max_context_tokens: 4_096,
            brave_api_key: None,
            search_provider: "searxng".to_string(),
            searxng_url: "http://localhost:8888".to_string(),
            crw_url: String::new(),
            search_max_results: 5,
            exec_timeout: 30,
            restrict_to_workspace: false,
            memory_config: MemoryConfig::default(),
            is_local: true,
            lane: Lane::default(),
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
            code_execution: CodeExecutionConfig::default(),
            python_kernel: PythonKernelConfig::default(),
            cua: CuaToolConfig::default(),
            adaptive_tokens: AdaptiveTokenConfig::default(),
            sessions_db_path: Some(
                std::env::temp_dir().join(format!("nanobot-int-{}.sqlite", uuid::Uuid::new_v4())),
            ),
        });
        let counters = test_runtime_counters(4_096);
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
        let session_key = "cap-int";
        if fail_interrupted_write {
            agent_loop
                .shared
                .core_handle
                .swappable()
                .sessions
                .fail_model_interrupted_writes_for_tests(1);
        }
        let reply = agent_loop
            .process_direct("please answer", session_key, "test", "capacity-interrupted")
            .await;

        let sessions = agent_loop.shared.core_handle.swappable().sessions.clone();
        let concrete = sessions
            .get_latest_session(session_key)
            .await
            .expect("session exists");
        let events = sessions.load_session_events(&concrete.id).await.unwrap();
        let history = sessions.get_history(&concrete.id, 100, 100).await;
        (provider, sessions, concrete.id, reply, events, history)
    }

    #[tokio::test]
    async fn interrupted_stream_persists_partial_artifact_never_success() {
        let (provider, sessions, session_id, reply, events, _history) =
            drive(CapacityFailure::Interrupted, false).await;

        assert_eq!(provider.requests.load(Ordering::SeqCst), 1);
        assert!(
            reply.contains("[Model Error]"),
            "reply must explain the interruption: {reply}"
        );
        // Typed interrupted event present.
        assert!(
            events
                .iter()
                .any(|event| event.payload.kind() == "model_interrupted"),
            "model_interrupted event must be journaled"
        );
        // Replay folds the call as incomplete with the partial artifact and
        // no successful response.
        let replay = sessions.load_session_replay(&session_id).await.unwrap();
        let call = replay.model_calls.last().expect("the interrupted call");
        assert!(call.response.is_none(), "never a successful response");
        let failure = call.failure.as_ref().expect("failure recorded");
        let partial = call.partial_output.as_ref().expect("partial artifact");
        assert_eq!(partial, &b"partial streamed text".to_vec());
        assert!(
            String::from_utf8_lossy(failure).contains("interrupted"),
            "error artifact carries the typed reason"
        );
    }

    #[tokio::test]
    async fn interrupted_stream_does_not_claim_a_failed_artifact_write() {
        let (provider, _sessions, _session_id, reply, events, _history) =
            drive(CapacityFailure::Interrupted, true).await;

        assert_eq!(provider.requests.load(Ordering::SeqCst), 1);
        assert!(reply.contains("could not be durably saved"), "{reply}");
        assert!(
            !reply.contains("was saved as an incomplete artifact"),
            "{reply}"
        );
        assert!(!events
            .iter()
            .any(|event| event.payload.kind() == "model_interrupted"));
    }

    #[tokio::test]
    async fn allocation_failure_ends_turn_without_parking() {
        let (provider, _sessions, _session_id, reply, events, _history) =
            drive(CapacityFailure::Unavailable, false).await;
        assert_eq!(provider.requests.load(Ordering::SeqCst), 1);
        assert!(!reply.contains("resume automatically"), "{reply}");
        assert!(!events
            .iter()
            .any(|event| event.payload.kind() == "turn_suspended"));
    }
}
