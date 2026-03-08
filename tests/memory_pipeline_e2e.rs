//! E2E integration tests for the memory & continual-learning pipeline.
//!
//! Tests cover: KnowledgeStore, Embedder, SessionIndexer, Reflector,
//! KnowledgeGraph, LoRA Bridge (ExperienceBuffer + D2L/T2L), and RecallTool.
//!
//! Tests 1, 2, 5 run without LM Studio (CI-safe).
//! Tests 3, 4 are `#[ignore]` and require a local OpenAI-compatible endpoint.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread;
use std::time::Duration;

use chrono::Local;
use serde_json::Value;
use tempfile::TempDir;

use nanobot::agent::knowledge_store::KnowledgeStore;
use nanobot::agent::lora_bridge::{
    build_d2l_document, build_t2l_description, compute_surprise, ExperienceBuffer,
};
use nanobot::agent::session_indexer::index_sessions;
use nanobot::config::schema::Config;
use nanobot::utils::helpers::safe_filename;

// =============================================================================
// Helpers (shared with agent_local_cli_smoke.rs pattern)
// =============================================================================

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

fn write_isolated_config(home: &Path, local_api_base: &str, local_model: &str) {
    let nanobot_dir = home.join(".nanobot");
    let workspace = nanobot_dir.join("workspace");
    fs::create_dir_all(&workspace).expect("create workspace");
    fs::create_dir_all(nanobot_dir.join("sessions")).expect("create sessions dir");
    fs::create_dir_all(nanobot_dir.join("logs")).expect("create logs dir");
    fs::create_dir_all(workspace.join("memory")).expect("create memory dir");
    fs::create_dir_all(workspace.join("memory").join("sessions")).expect("create memory/sessions");

    let mut cfg = Config::default();
    cfg.agents.defaults.workspace = workspace.to_string_lossy().to_string();
    cfg.agents.defaults.local_api_base = local_api_base.to_string();
    cfg.agents.defaults.local_model = local_model.to_string();
    cfg.agents.defaults.skip_jit_gate = true;

    let cfg_path = nanobot_dir.join("config.json");
    let cfg_json = serde_json::to_string_pretty(&cfg).expect("serialize config");
    fs::write(cfg_path, cfg_json).expect("write config");
}

fn expected_session_path(home: &Path, session_key: &str) -> PathBuf {
    let safe = safe_filename(&session_key.replace(':', "_"));
    let date = Local::now().format("%Y-%m-%d");
    home.join(".nanobot")
        .join("sessions")
        .join(format!("{}_{}.jsonl", safe, date))
}

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

/// Write a realistic JSONL session file with metadata + user/assistant turns.
fn write_session_jsonl(dir: &Path, filename: &str, session_key: &str, turns: &[(&str, &str)]) {
    let mut lines = vec![format!(
        r#"{{"_type":"metadata","session_key":"{}","created_at":"2026-02-27T10:00:00Z","updated_at":"2026-02-27T10:30:00Z"}}"#,
        session_key
    )];
    for (role, content) in turns {
        lines.push(format!(r#"{{"role":"{}","content":"{}"}}"#, role, content));
    }
    fs::write(dir.join(filename), lines.join("\n")).expect("write session JSONL");
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
// Test 2: Session indexer → searchable knowledge (no LM Studio)
// =============================================================================

#[test]
#[cfg(feature = "semantic")]
fn session_indexer_produces_searchable_knowledge() {
    let tmp = TempDir::new().unwrap();
    let home = tmp.path();

    // Set HOME for this test so open_default() uses our tempdir.
    // SAFETY: this is a test binary; parallel tests may conflict but #[serial]
    // is not needed because each test uses unique HOME paths via tempdir.
    unsafe { std::env::set_var("HOME", home) };

    let nanobot_dir = home.join(".nanobot");
    let sessions_dir = nanobot_dir.join("sessions");
    let memory_sessions_dir = nanobot_dir
        .join("workspace")
        .join("memory")
        .join("sessions");
    fs::create_dir_all(&sessions_dir).unwrap();
    fs::create_dir_all(&memory_sessions_dir).unwrap();

    // Write a realistic session JSONL about Rust async patterns.
    write_session_jsonl(
        &sessions_dir,
        "cli_async_rust_2026-02-27.jsonl",
        "cli:async_rust",
        &[
            ("user", "How does tokio handle task cancellation in Rust?"),
            ("assistant", "Tokio uses structured concurrency with select! macro and drop guards. When a task is dropped, its future is cancelled. You can use tokio::select! to race multiple futures and cancel the losers."),
            ("user", "What about graceful shutdown patterns?"),
            ("assistant", "For graceful shutdown, use a broadcast channel or tokio::sync::watch. Signal the shutdown, then await all tasks with JoinSet or tokio::join!. The CancellationToken from tokio-util is also excellent for propagating shutdown across task trees."),
        ],
    );

    let (indexed, _skipped, errors) = index_sessions(&sessions_dir, &memory_sessions_dir);

    assert_eq!(errors, 0, "no indexing errors expected");
    assert_eq!(indexed, 1, "should index exactly 1 session");

    // Verify SESSION_*.md was created with YAML frontmatter.
    let md_files: Vec<_> = fs::read_dir(&memory_sessions_dir)
        .unwrap()
        .flatten()
        .filter(|e| {
            e.file_name()
                .to_str()
                .map_or(false, |n| n.starts_with("SESSION_") && n.ends_with(".md"))
        })
        .collect();
    assert_eq!(md_files.len(), 1, "should create exactly 1 SESSION_*.md");

    let md_content = fs::read_to_string(md_files[0].path()).unwrap();
    assert!(
        md_content.contains("session_key:"),
        "SESSION_*.md should have YAML frontmatter"
    );
    assert!(
        md_content.contains("tokio") || md_content.contains("cancellation"),
        "SESSION_*.md should contain session content"
    );

    // Verify content is searchable via KnowledgeStore (ingested by index_sessions).
    let store = KnowledgeStore::open_default().unwrap();
    let hits = store
        .hybrid_search("tokio cancellation shutdown", 5)
        .unwrap();
    assert!(
        !hits.is_empty(),
        "Session content should be searchable in KnowledgeStore after indexing"
    );

    // Restore HOME (best effort — tempdir cleanup handles the rest).
    if let Ok(real_home) = std::env::var("USER") {
        let _ = unsafe { std::env::set_var("HOME", format!("/Users/{}", real_home)) };
    }
}

// =============================================================================
// Test 3: Reflector distills sessions → MEMORY.md + KnowledgeGraph
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

    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let tmp = TempDir::new().unwrap();
        let home = tmp.path();
        unsafe { std::env::set_var("HOME", home) };

        let workspace = home.join(".nanobot").join("workspace");
        let mem_dir = workspace.join("memory");
        fs::create_dir_all(&mem_dir).unwrap();
        fs::create_dir_all(mem_dir.join("sessions")).unwrap();

        // Populate workspace with completed working sessions containing
        // distinctive user preferences (short sessions to fit 8K context).
        let wm = WorkingMemoryStore::new(&workspace);
        wm.update_from_compaction(
            "cli:prefs",
            "User said their favorite programming language is Haskell and they always use the Catppuccin color theme. They prefer functional programming paradigms.",
            0,
        );
        wm.complete("cli:prefs");

        wm.update_from_compaction(
            "cli:tools",
            "User mentioned they use Helix as their primary editor and prefer nix for package management. They run NixOS on their server.",
            0,
        );
        wm.complete("cli:tools");

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
        );

        assert!(
            reflector.should_reflect(),
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

        // Verify sessions were archived.
        let remaining = wm.list_completed();
        assert!(
            remaining.is_empty(),
            "completed sessions should be archived after reflection"
        );

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
// Test 4: Full pipeline — agent remembers across sessions
//         (requires LM Studio + built binary)
// =============================================================================

#[test]
#[ignore = "requires running local OpenAI-compatible endpoint (e.g. LM Studio)"]
#[cfg(all(feature = "semantic", feature = "knowledge-graph"))]
fn full_pipeline_agent_remembers_across_sessions() {
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

    // Verify session JSONL was created.
    let session1_path = expected_session_path(home, &session1);
    assert!(
        session1_path.exists(),
        "Session 1 JSONL not created at {}",
        session1_path.display()
    );

    // Index step: convert JSONL → SESSION_*.md + KnowledgeStore ingestion.
    // Set HOME so open_default() resolves to our tempdir.
    unsafe { std::env::set_var("HOME", home) };
    let sessions_dir = home.join(".nanobot").join("sessions");
    let memory_sessions_dir = home
        .join(".nanobot")
        .join("workspace")
        .join("memory")
        .join("sessions");
    fs::create_dir_all(&memory_sessions_dir).unwrap();

    let (indexed, _, errors) = index_sessions(&sessions_dir, &memory_sessions_dir);
    assert!(indexed >= 1, "should index at least 1 session");
    assert_eq!(errors, 0, "no indexing errors expected");

    // Session 2: Ask the agent what it remembers (different session key).
    let session2 = format!("cli:mem_recall_{}", uuid::Uuid::new_v4());
    let output2 = run_local_agent_with_retry(
        &bin,
        home,
        &session2,
        "What is my favorite programming language? Check your memory using the recall tool.",
        3,
    );
    assert!(
        output2.status.success(),
        "Session 2 failed: stderr={}",
        String::from_utf8_lossy(&output2.stderr)
    );

    let stdout2 = String::from_utf8_lossy(&output2.stdout);
    eprintln!("--- Session 2 stdout ---\n{}", stdout2);
    assert!(
        stdout2.to_lowercase().contains("haskell"),
        "Agent should recall 'Haskell' from memory; stdout={}",
        stdout2
    );
}

// =============================================================================
// Test 5: ExperienceBuffer + D2L/T2L generation (no LM Studio)
// =============================================================================

#[test]
#[cfg(feature = "knowledge-graph")]
fn experience_buffer_and_d2l_generation() {
    use nanobot::agent::knowledge_graph::KnowledgeGraph;

    let tmp = TempDir::new().unwrap();
    let workspace = tmp.path().join("workspace");
    let mem_dir = workspace.join("memory");
    fs::create_dir_all(&mem_dir).unwrap();

    // --- ExperienceBuffer: record experiences with different tool traces ---
    let db_path = tmp.path().join("experience.db");
    let buffer = ExperienceBuffer::open(&db_path).unwrap();

    // Experience 1: Simple file read
    let _id1 = buffer
        .record(
            "Read the config file",
            r#"[{"name":"read_file","arguments":{"path":"config.json"},"result":"ok"}]"#,
            "Config file contains database settings.",
            true,
            0.9,
            "test-model",
        )
        .unwrap();

    // Experience 2: Multi-tool workflow (higher surprise expected)
    let _id2 = buffer
        .record(
            "Deploy the application",
            r#"[{"name":"read_file","arguments":{"path":"Dockerfile"}},{"name":"execute_shell","arguments":{"cmd":"docker build ."}},{"name":"execute_shell","arguments":{"cmd":"docker push"}},{"name":"send_message","arguments":{"text":"deployed"}}]"#,
            "Application deployed successfully to production.",
            true,
            0.95,
            "test-model",
        )
        .unwrap();

    // Experience 3: Failed experience (should not appear in unexported)
    let _id3 = buffer
        .record(
            "Delete everything",
            r#"[{"name":"execute_shell","arguments":{"cmd":"rm -rf /"}}]"#,
            "Permission denied",
            false,
            0.1,
            "test-model",
        )
        .unwrap();

    // Verify compute_surprise scores: multi-tool should be higher.
    let simple_trace = r#"[{"name":"read_file"}]"#;
    let complex_trace = r#"[{"name":"read_file"},{"name":"execute_shell"},{"name":"send_message"},{"name":"docker_build"}]"#;
    let s_simple = compute_surprise("Read file", simple_trace);
    let s_complex = compute_surprise("Deploy", complex_trace);
    assert!(
        s_complex > s_simple,
        "Complex trace ({:.3}) should have higher surprise than simple ({:.3})",
        s_complex,
        s_simple
    );

    // Verify buffer stats.
    let stats = buffer.stats().unwrap();
    assert_eq!(stats.total, 3);
    assert_eq!(stats.successful, 2);
    assert_eq!(stats.unexported, 2);

    // --- D2L document generation ---
    // Write MEMORY.md in workspace.
    fs::write(
        mem_dir.join("MEMORY.md"),
        "- User prefers Haskell\n- Uses Catppuccin theme\n- Runs NixOS on server",
    )
    .unwrap();

    // Write a KnowledgeGraph with entities.
    let kg_path = tmp.path().join("knowledge_graph.json");
    {
        let mut kg = KnowledgeGraph::open(&kg_path).unwrap();
        kg.upsert_entity("Haskell", "language", "User's favorite language");
        kg.upsert_entity("Catppuccin", "theme", "Preferred color theme");
        kg.add_relation("User", "prefers", "Haskell", "stated preference");
        kg.add_relation("User", "uses", "Catppuccin", "stated preference");
        kg.save().unwrap();
    }

    // build_d2l_document reads from workspace/memory/MEMORY.md.
    let d2l_doc = build_d2l_document(&workspace);
    assert!(
        d2l_doc.contains("Haskell"),
        "D2L doc should include MEMORY.md content; got: {}",
        &d2l_doc[..d2l_doc.len().min(200)]
    );
    assert!(
        d2l_doc.contains("User Memory"),
        "D2L doc should have User Memory header"
    );

    // --- T2L description generation ---
    let t2l_desc = build_t2l_description(&buffer);
    assert!(
        !t2l_desc.is_empty(),
        "T2L description should not be empty with experiences"
    );
    assert!(
        t2l_desc.contains("read_file"),
        "T2L should list tool names; got: {}",
        &t2l_desc[..t2l_desc.len().min(200)]
    );
    assert!(
        t2l_desc.contains("2 experiences"),
        "T2L should report 2 successful experiences; got: {}",
        &t2l_desc[..t2l_desc.len().min(200)]
    );

    // --- JSONL export ---
    let export_path = tmp.path().join("training").join("export.jsonl");
    let export_result = buffer.export_jsonl(&export_path, 10).unwrap();
    assert_eq!(
        export_result.count, 2,
        "should export 2 successful experiences"
    );
    assert!(export_path.exists(), "JSONL file should be created");

    let jsonl_content = fs::read_to_string(&export_path).unwrap();
    let jsonl_lines: Vec<&str> = jsonl_content.trim().lines().collect();
    assert_eq!(jsonl_lines.len(), 2, "JSONL should have 2 lines");

    // Verify JSONL structure.
    for line in &jsonl_lines {
        let entry: Value = serde_json::from_str(line).expect("each line should be valid JSON");
        assert!(
            entry.get("messages").is_some(),
            "entry should have messages"
        );
        assert!(entry.get("quality").is_some(), "entry should have quality");
        assert!(
            entry.get("surprise").is_some(),
            "entry should have surprise"
        );
    }

    // After export, nothing left unexported.
    let post_stats = buffer.stats().unwrap();
    assert_eq!(
        post_stats.unexported, 0,
        "all successful experiences should be marked exported"
    );
}
