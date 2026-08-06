// Test crate — sanctioned escape hatch (research doc §3.6): integration tests
// keep pragmatic unwraps without blocking the production deny regime.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable,
    clippy::todo,
    clippy::unimplemented,
    clippy::dbg_macro,
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::shadow_reuse,
    clippy::shadow_unrelated,
    clippy::shadow_same,
    clippy::string_add,
    clippy::format_push_string,
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::pedantic,
    clippy::nursery,
)]
//! E2E integration tests for the memory & continual-learning pipeline.
//!
//! Tests cover: KnowledgeStore, Embedder, SQLite working memory, Reflector,
//! KnowledgeGraph, LoRA Bridge (ExperienceBuffer + D2L/T2L), and RecallTool.
//!
//! Tests 1, 2, 5 run without LM Studio (CI-safe).
//! Tests 3, 4 are `#[ignore]` and require a local OpenAI-compatible endpoint.

#[cfg(feature = "semantic")]
use std::fs;
#[cfg(feature = "semantic")]
use std::path::Path;
#[cfg(all(feature = "semantic", feature = "knowledge-graph"))]
use std::path::PathBuf;
#[cfg(all(feature = "semantic", feature = "knowledge-graph"))]
use std::process::Command;
#[cfg(all(feature = "semantic", feature = "knowledge-graph"))]
use std::thread;
#[cfg(all(feature = "semantic", feature = "knowledge-graph"))]
use std::time::Duration;

#[cfg(feature = "semantic")]
use tempfile::TempDir;

#[cfg(feature = "semantic")]
use nanobot::agent::knowledge_store::KnowledgeStore;
#[cfg(all(feature = "semantic", feature = "knowledge-graph"))]
use nanobot::config::schema::Config;

// =============================================================================
// Helpers (shared with agent_local_cli_smoke.rs pattern)
// =============================================================================

#[cfg(all(feature = "semantic", feature = "knowledge-graph"))]
fn resolve_nanobot_bin() -> PathBuf {
    if let Ok(path) = std::env::var("CARGO_BIN_EXE_nanobot") {
        return PathBuf::from(path);
    }
    if let Ok(path) = std::env::var("NANOBOT_BIN") {
        return PathBuf::from(path);
    }
    let exe = std::env::current_exe().expect("failed to resolve current_exe");
    let debug_dir = exe
        .parent()
        .and_then(|p| p.parent())
        .expect("failed to resolve target/debug directory");
    let candidate = debug_dir.join("nanobot");
    assert!(
        candidate.exists(),
        "nanobot binary not found at {}; set NANOBOT_BIN",
        candidate.display()
    );
    candidate
}

#[cfg(all(feature = "semantic", feature = "knowledge-graph"))]
fn write_isolated_config(home: &Path, local_api_base: &str, local_model: &str) {
    let nanobot_dir = home.join(".nanobot");
    let workspace = nanobot_dir.join("workspace");
    fs::create_dir_all(&workspace).expect("create workspace");
    fs::create_dir_all(nanobot_dir.join("sessions")).expect("create sessions dir");
    fs::create_dir_all(nanobot_dir.join("logs")).expect("create logs dir");
    fs::create_dir_all(workspace.join("memory")).expect("create memory dir");

    let mut cfg = Config::default();
    cfg.agents.defaults.workspace = workspace.to_string_lossy().to_string();
    cfg.agents.defaults.local_api_base = local_api_base.to_string();
    cfg.agents.defaults.local_model = local_model.to_string();
    cfg.agents.defaults.skip_jit_gate = true;

    let cfg_path = nanobot_dir.join("config.json");
    let cfg_json = serde_json::to_string_pretty(&cfg).expect("serialize config");
    fs::write(cfg_path, cfg_json).expect("write config");
}

#[cfg(all(feature = "semantic", feature = "knowledge-graph"))]
fn run_local_agent_once(
    bin: &Path,
    home: &Path,
    session: &str,
    prompt: &str,
) -> std::process::Output {
    Command::new(bin)
        .env("HOME", home)
        .arg("agent")
        .arg("-l")
        .arg("-s")
        .arg(session)
        .arg("-m")
        .arg(prompt)
        .output()
        .expect("failed to run nanobot agent -l")
}

#[cfg(all(feature = "semantic", feature = "knowledge-graph"))]
fn run_local_agent_with_retry(
    bin: &Path,
    home: &Path,
    session: &str,
    prompt: &str,
    attempts: usize,
) -> std::process::Output {
    let mut last = run_local_agent_once(bin, home, session, prompt);
    for _ in 1..attempts {
        let stderr = String::from_utf8_lossy(&last.stderr);
        let stdout = String::from_utf8_lossy(&last.stdout);
        let transient = stderr.contains("error sending request for url")
            || stdout.contains("error sending request for url");
        if last.status.success() && !transient {
            return last;
        }
        if !transient {
            return last;
        }
        thread::sleep(Duration::from_millis(500));
        last = run_local_agent_once(bin, home, session, prompt);
    }
    last
}

// =============================================================================
// Test 1: KnowledgeStore ingest + hybrid search (no LM Studio)
// =============================================================================

#[test]
#[cfg(feature = "semantic")]
fn knowledge_store_ingest_and_hybrid_search() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("knowledge.db");

    let store = KnowledgeStore::open(&db_path).unwrap();

    // Ingest two documents with real fastembed embeddings.
    let doc1 = "Rust is a systems programming language focused on safety, speed, and concurrency. \
                It achieves memory safety without garbage collection through its ownership system.";
    let doc2 =
        "Python is a high-level interpreted language known for its readability and extensive \
                standard library. It is widely used in data science and machine learning.";

    let r1 = store
        .ingest_with_embeddings("rust_doc", None, doc1, 256, 32)
        .unwrap();
    let r2 = store
        .ingest_with_embeddings("python_doc", None, doc2, 256, 32)
        .unwrap();

    assert!(r1.chunks_created > 0, "Rust doc should produce chunks");
    assert!(r2.chunks_created > 0, "Python doc should produce chunks");

    // Semantic search: "memory safety ownership" should rank Rust doc first.
    let hits = store.hybrid_search("memory safety ownership", 5).unwrap();
    assert!(!hits.is_empty(), "hybrid_search should return results");
    assert_eq!(
        hits[0].source_name,
        "rust_doc",
        "Rust doc should rank first for 'memory safety ownership'; got {:?}",
        hits.iter().map(|h| &h.source_name).collect::<Vec<_>>()
    );

    // BM25 keyword search: "garbage collection" should also find Rust doc.
    let keyword_hits = store.search("garbage collection", 5).unwrap();
    assert!(
        keyword_hits.iter().any(|h| h.source_name == "rust_doc"),
        "BM25 search should find 'garbage collection' in Rust doc"
    );

    // Verify stats.
    let stats = store.stats().unwrap();
    assert_eq!(stats.total_sources, 2);
    assert!(stats.total_chunks >= 2);
}

// =============================================================================
// Test 2: Reflector distills sessions → MEMORY.md + KnowledgeGraph
//         (requires LM Studio)
// =============================================================================

#[test]
#[ignore = "requires running local OpenAI-compatible endpoint (e.g. LM Studio)"]
#[cfg(all(feature = "semantic", feature = "knowledge-graph"))]
fn reflector_distills_to_memory_and_graph() {
    use nanobot::agent::knowledge_graph::KnowledgeGraph;
    use nanobot::agent::reflector::Reflector;
    use nanobot::agent::working_memory::WorkingMemoryStore;
    use nanobot::providers::openai_compat::OpenAICompatProvider;
    use nanobot::session::db::SessionDb;

    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let tmp = TempDir::new().unwrap();
        let home = tmp.path();
        unsafe { std::env::set_var("HOME", home) };

        let workspace = home.join(".nanobot").join("workspace");
        let mem_dir = workspace.join("memory");
        fs::create_dir_all(&mem_dir).unwrap();
        let sessions = std::sync::Arc::new(SessionDb::new(
            &home.join(".nanobot").join("sessions.db"),
        ));

        // Populate workspace with completed working sessions containing
        // distinctive user preferences (short sessions to fit 8K context).
        let wm = WorkingMemoryStore::new(sessions.clone());
        let prefs = sessions.create_session("cli:prefs").await;
        sessions
            .save_working_memory(
                &prefs.id,
                "User said their favorite programming language is Haskell and they always use the Catppuccin color theme. They prefer functional programming paradigms.",
                "active",
                0,
            )
            .await
            .unwrap();
        wm.complete(&prefs.id).await.unwrap();

        let tools = sessions.create_session("cli:tools").await;
        sessions
            .save_working_memory(
                &tools.id,
                "User mentioned they use Helix as their primary editor and prefer nix for package management. They run NixOS on their server.",
                "active",
                0,
            )
            .await
            .unwrap();
        wm.complete(&tools.id).await.unwrap();

        // Create Reflector with real LLM provider pointed at LM Studio.
        let api_base = std::env::var("NANOBOT_TEST_LOCAL_API_BASE")
            .unwrap_or_else(|_| "http://127.0.0.1:1234/v1".to_string());
        let model = std::env::var("NANOBOT_TEST_LOCAL_MODEL")
            .unwrap_or_else(|_| "qwen3.5-35b-a3b".to_string());

        let provider = std::sync::Arc::new(OpenAICompatProvider::new(
            "not-needed", // API key not needed for local
            Some(&api_base),
            Some(model.as_str()),
        ));

        let reflector = Reflector::new(
            provider,
            model,
            &workspace,
            0, // threshold=0 to force reflection
            sessions,
        );

        assert!(
            reflector.should_reflect().await,
            "should_reflect must be true with completed sessions"
        );

        let result = reflector.reflect().await;
        assert!(
            result.is_ok(),
            "reflect() failed: {:?}",
            result.err()
        );

        // Verify MEMORY.md was written with non-empty content.
        let memory_path = mem_dir.join("MEMORY.md");
        assert!(memory_path.exists(), "MEMORY.md should be created");
        let memory_content = fs::read_to_string(&memory_path).unwrap();
        assert!(
            !memory_content.trim().is_empty(),
            "MEMORY.md should have content after reflection"
        );
        eprintln!("--- MEMORY.md ({} chars) ---", memory_content.len());
        eprintln!("{}", &memory_content[..memory_content.len().min(500)]);

        // Verify completed rows were marked reflected.
        let remaining = wm.list_completed().await.unwrap();
        assert!(
            remaining.is_empty(),
            "completed sessions should be consumed after reflection"
        );
        assert_eq!(wm.list_reflected().await.unwrap().len(), 2);

        // Log knowledge graph counts (non-deterministic LLM output).
        let kg = KnowledgeGraph::open_default().unwrap();
        eprintln!(
            "--- KnowledgeGraph: {} entities, {} relations ---",
            kg.entity_count(),
            kg.relation_count()
        );
    });
}

// =============================================================================
// Test 3: Full pipeline — agent resumes canonical SQLite history
//         (requires LM Studio + built binary)
// =============================================================================

#[test]
#[ignore = "requires running local OpenAI-compatible endpoint (e.g. LM Studio)"]
#[cfg(all(feature = "semantic", feature = "knowledge-graph"))]
fn full_pipeline_agent_resumes_sqlite_session() {
    let bin = resolve_nanobot_bin();
    let tmp = TempDir::new().unwrap();
    let home = tmp.path();

    let api_base = std::env::var("NANOBOT_TEST_LOCAL_API_BASE")
        .unwrap_or_else(|_| "http://127.0.0.1:1234/v1".to_string());
    let model =
        std::env::var("NANOBOT_TEST_LOCAL_MODEL").unwrap_or_else(|_| "qwen3.5-35b-a3b".to_string());

    write_isolated_config(home, &api_base, &model);

    // Session 1: Tell the agent a distinctive fact.
    let session1 = format!("cli:mem_store_{}", uuid::Uuid::new_v4());
    let output1 = run_local_agent_with_retry(
        &bin,
        home,
        &session1,
        "Remember this: my favorite programming language is Haskell and I use the Catppuccin color theme everywhere.",
        3,
    );
    assert!(
        output1.status.success(),
        "Session 1 failed: stderr={}",
        String::from_utf8_lossy(&output1.stderr)
    );

    let rt = tokio::runtime::Runtime::new().unwrap();
    let db = nanobot::session::db::SessionDb::new(&home.join(".nanobot").join("sessions.db"));
    let stored = rt
        .block_on(db.get_latest_session(&session1))
        .expect("session should be present in SQLite");
    let messages = rt.block_on(db.get_all_messages(&stored.id));
    assert!(messages.iter().any(|message| {
        message
            .get("content")
            .and_then(serde_json::Value::as_str)
            .is_some_and(|content| content.contains("Haskell"))
    }));

    // A second process with the same session key reconstructs raw history from SQLite.
    let output2 = run_local_agent_with_retry(
        &bin,
        home,
        &session1,
        "What favorite programming language did I just tell you?",
        3,
    );
    assert!(
        output2.status.success(),
        "Resumed session failed: stderr={}",
        String::from_utf8_lossy(&output2.stderr)
    );

    let stdout2 = String::from_utf8_lossy(&output2.stdout);
    eprintln!("--- Resumed session stdout ---\n{}", stdout2);
    assert!(
        stdout2.to_lowercase().contains("haskell"),
        "Agent should recover 'Haskell' from SQLite history; stdout={}",
        stdout2
    );
}
