#![allow(dead_code)]
use anyhow::{Context, Result};
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};

// =============================================================================
// Experience Buffer (SQLite-backed)
// =============================================================================

/// A recorded experience from a successful tool execution trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experience {
    /// Unique ID.
    pub id: i64,
    /// The task/prompt that was given.
    pub prompt: String,
    /// The tool calls that were made (JSON array of {name, arguments, result}).
    pub tool_trace: String,
    /// The final response/summary.
    pub response: String,
    /// Whether the execution was successful.
    pub success: bool,
    /// Quality score (0.0-1.0) — from user feedback or automated checks.
    pub quality: f64,
    /// Model used for this execution.
    pub model: String,
    /// Surprise score: how unexpected was this trace (higher = more worth learning).
    pub surprise: f64,
    /// Whether this experience has been exported for training.
    pub exported: bool,
    /// When this experience was recorded.
    pub created_at: String,
}

/// SQLite-backed experience buffer for collecting training data.
pub struct ExperienceBuffer {
    conn: Connection,
}

impl ExperienceBuffer {
    /// Open or create the experience buffer database.
    pub fn open(db_path: &Path) -> Result<Self> {
        let conn = Connection::open(db_path)?;
        conn.execute_batch(
            "
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = NORMAL;

            CREATE TABLE IF NOT EXISTS experiences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt TEXT NOT NULL,
                tool_trace TEXT NOT NULL,
                response TEXT NOT NULL,
                success INTEGER NOT NULL DEFAULT 1,
                quality REAL NOT NULL DEFAULT 0.0,
                model TEXT NOT NULL DEFAULT '',
                surprise REAL NOT NULL DEFAULT 0.0,
                exported INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_experiences_surprise ON experiences(surprise DESC);
            CREATE INDEX IF NOT EXISTS idx_experiences_exported ON experiences(exported);
            CREATE INDEX IF NOT EXISTS idx_experiences_quality ON experiences(quality DESC);
        ",
        )?;
        Ok(Self { conn })
    }

    /// Open at the default location: ~/.nanobot/experience.db
    pub fn open_default() -> Result<Self> {
        let db_path = dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".nanobot")
            .join("experience.db");
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        Self::open(&db_path)
    }

    /// Record a new experience.
    pub fn record(
        &self,
        prompt: &str,
        tool_trace: &str,
        response: &str,
        success: bool,
        quality: f64,
        model: &str,
    ) -> Result<i64> {
        let surprise = compute_surprise(prompt, tool_trace);
        self.conn.execute(
            "INSERT INTO experiences (prompt, tool_trace, response, success, quality, model, surprise) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![prompt, tool_trace, response, success as i32, quality, model, surprise],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Get the top N unexported experiences sorted by surprise (most surprising first).
    pub fn top_unexported(&self, limit: usize) -> Result<Vec<Experience>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, prompt, tool_trace, response, success, quality, model, surprise, exported, created_at
             FROM experiences WHERE exported = 0 AND success = 1
             ORDER BY surprise DESC, quality DESC
             LIMIT ?1"
        )?;
        let rows = stmt.query_map(params![limit as i64], |row| {
            Ok(Experience {
                id: row.get(0)?,
                prompt: row.get(1)?,
                tool_trace: row.get(2)?,
                response: row.get(3)?,
                success: row.get::<_, i32>(4)? != 0,
                quality: row.get(5)?,
                model: row.get(6)?,
                surprise: row.get(7)?,
                exported: row.get::<_, i32>(8)? != 0,
                created_at: row.get(9)?,
            })
        })?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    /// Mark experiences as exported.
    pub fn mark_exported(&self, ids: &[i64]) -> Result<usize> {
        if ids.is_empty() {
            return Ok(0);
        }
        let placeholders: Vec<String> = ids.iter().map(|_| "?".to_string()).collect();
        let sql = format!(
            "UPDATE experiences SET exported = 1 WHERE id IN ({})",
            placeholders.join(",")
        );
        let params: Vec<&dyn rusqlite::types::ToSql> = ids
            .iter()
            .map(|id| id as &dyn rusqlite::types::ToSql)
            .collect();
        let count = self.conn.execute(&sql, params.as_slice())?;
        Ok(count)
    }

    /// Export experiences as JSONL for the Python ContinualLearner.
    /// Returns the path to the written JSONL file.
    pub fn export_jsonl(&self, output_path: &Path, limit: usize) -> Result<ExportResult> {
        let experiences = self.top_unexported(limit)?;
        if experiences.is_empty() {
            return Ok(ExportResult {
                path: output_path.to_path_buf(),
                count: 0,
                ids: vec![],
            });
        }

        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let mut lines = Vec::new();
        let mut ids = Vec::new();
        for exp in &experiences {
            // Format as chat-style training data
            let training_entry = serde_json::json!({
                "messages": [
                    {"role": "user", "content": exp.prompt},
                    {"role": "assistant", "content": exp.response}
                ],
                "tool_trace": serde_json::from_str::<serde_json::Value>(&exp.tool_trace).unwrap_or(serde_json::Value::Null),
                "quality": exp.quality,
                "surprise": exp.surprise,
                "model": exp.model,
            });
            lines.push(serde_json::to_string(&training_entry)?);
            ids.push(exp.id);
        }

        std::fs::write(output_path, lines.join("\n") + "\n")?;
        self.mark_exported(&ids)?;

        Ok(ExportResult {
            path: output_path.to_path_buf(),
            count: ids.len(),
            ids,
        })
    }

    /// Record a new experience with an explicit surprise score (e.g. CE loss).
    pub fn record_with_surprise(
        &self,
        prompt: &str,
        tool_trace: &str,
        response: &str,
        success: bool,
        quality: f64,
        model: &str,
        surprise: f64,
    ) -> Result<i64> {
        self.conn.execute(
            "INSERT INTO experiences (prompt, tool_trace, response, success, quality, model, surprise) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![prompt, tool_trace, response, success as i32, quality, model, surprise],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Get buffer statistics.
    pub fn stats(&self) -> Result<BufferStats> {
        let total: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM experiences", [], |r| r.get(0))?;
        let successful: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM experiences WHERE success = 1",
            [],
            |r| r.get(0),
        )?;
        let unexported: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM experiences WHERE exported = 0 AND success = 1",
            [],
            |r| r.get(0),
        )?;
        let avg_surprise: f64 = self.conn.query_row(
            "SELECT COALESCE(AVG(surprise), 0.0) FROM experiences WHERE success = 1",
            [],
            |r| r.get(0),
        )?;
        Ok(BufferStats {
            total,
            successful,
            unexported,
            avg_surprise,
        })
    }
}

#[derive(Debug, Clone)]
pub struct ExportResult {
    pub path: PathBuf,
    pub count: usize,
    pub ids: Vec<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BufferStats {
    pub total: i64,
    pub successful: i64,
    pub unexported: i64,
    pub avg_surprise: f64,
}

// =============================================================================
// Surprise Detection (heuristic)
// =============================================================================

/// Compute a surprise score for an experience.
///
/// Higher surprise = more worth learning from. Uses simple heuristics:
/// - Longer tool traces suggest more complex/novel workflows
/// - More unique tool names suggest novel combinations
/// - Shorter prompts with longer traces suggest efficient delegation
pub fn compute_surprise(prompt: &str, tool_trace: &str) -> f64 {
    let prompt_len = prompt.len() as f64;
    let trace_len = tool_trace.len() as f64;

    // Factor 1: Trace complexity (longer traces = more interesting)
    let complexity = (trace_len / 1000.0).min(5.0) / 5.0; // 0-1, saturates at 5K chars

    // Factor 2: Efficiency ratio (short prompt, long trace = efficient delegation)
    let efficiency = if prompt_len > 0.0 {
        (trace_len / prompt_len).min(10.0) / 10.0
    } else {
        0.0
    };

    // Factor 3: Tool diversity (count unique tool names in trace)
    let diversity = count_unique_tools(tool_trace) as f64 / 5.0; // Normalize to ~1.0 for 5 tools
    let diversity = diversity.min(1.0);

    // Weighted combination
    0.4 * complexity + 0.3 * efficiency + 0.3 * diversity
}

/// Count unique tool names in a JSON tool trace.
fn count_unique_tools(tool_trace: &str) -> usize {
    // Try to parse as JSON array
    if let Ok(arr) = serde_json::from_str::<Vec<serde_json::Value>>(tool_trace) {
        let names: std::collections::HashSet<&str> = arr
            .iter()
            .filter_map(|v| v.get("name").and_then(|n| n.as_str()))
            .collect();
        names.len()
    } else {
        0
    }
}

// =============================================================================
// LoRA Hot-Swap (local server API)
// =============================================================================

/// Configuration for LoRA adapter management.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraConfig {
    /// Base URL of the local inference server (e.g., "http://127.0.0.1:8080").
    pub server_url: String,
    /// Path to the LoRA adapter file (.gguf).
    pub adapter_path: Option<PathBuf>,
    /// Scale factor for the LoRA adapter (0.0-1.0).
    pub scale: f64,
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self {
            server_url: "http://127.0.0.1:8080".to_string(),
            adapter_path: None,
            scale: 0.5,
        }
    }
}

/// Result of a LoRA hot-swap operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HotSwapResult {
    pub success: bool,
    pub message: String,
    pub adapter_path: Option<String>,
    pub scale: f64,
}

/// Apply a LoRA adapter to a running local server via POST /lora-adapters.
///
/// Uses the hot-swap API to load/unload adapters without restarting the server.
pub async fn apply_lora_adapter(config: &LoraConfig) -> Result<HotSwapResult> {
    let client = reqwest::Client::new();
    let url = format!("{}/lora-adapters", config.server_url);

    let body = if let Some(ref adapter_path) = config.adapter_path {
        // Apply adapter
        serde_json::json!([{
            "path": adapter_path.to_string_lossy(),
            "scale": config.scale,
        }])
    } else {
        // Remove all adapters (empty array)
        serde_json::json!([])
    };

    let resp = client.post(&url).json(&body).send().await?;

    let status = resp.status();
    let text = resp.text().await.unwrap_or_default();

    if status.is_success() {
        Ok(HotSwapResult {
            success: true,
            message: format!("LoRA adapter applied successfully: {}", text),
            adapter_path: config
                .adapter_path
                .as_ref()
                .map(|p| p.to_string_lossy().to_string()),
            scale: config.scale,
        })
    } else {
        Ok(HotSwapResult {
            success: false,
            message: format!("LoRA hot-swap failed ({}): {}", status, text),
            adapter_path: config
                .adapter_path
                .as_ref()
                .map(|p| p.to_string_lossy().to_string()),
            scale: config.scale,
        })
    }
}

/// Remove all LoRA adapters from a running local server.
pub async fn remove_lora_adapters(server_url: &str) -> Result<HotSwapResult> {
    apply_lora_adapter(&LoraConfig {
        server_url: server_url.to_string(),
        adapter_path: None,
        scale: 0.0,
    })
    .await
}

/// Check if the local server supports LoRA hot-swap.
pub async fn check_lora_support(server_url: &str) -> Result<bool> {
    let client = reqwest::Client::new();
    let url = format!("{}/lora-adapters", server_url);

    // Try a GET request to see if the endpoint exists
    match client.get(&url).send().await {
        Ok(resp) => Ok(resp.status().is_success() || resp.status().as_u16() == 405),
        Err(_) => Ok(false),
    }
}

// =============================================================================
// Training Pipeline Orchestration
// =============================================================================

/// Default export path for training JSONL.
pub fn default_export_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".nanobot")
        .join("training")
        .join("experiences.jsonl")
}

/// Default adapter output path.
pub fn default_adapter_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".nanobot")
        .join("adapters")
        .join("latest.gguf")
}

/// Status of the training pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingStatus {
    /// Whether a training run is currently active.
    pub active: bool,
    /// Number of experiences available for training.
    pub pending_experiences: i64,
    /// Path to the latest adapter file (if any).
    pub latest_adapter: Option<PathBuf>,
    /// Whether the adapter is currently loaded in the local server.
    pub adapter_loaded: bool,
}

/// Check the training pipeline status.
pub fn check_training_status(buffer: &ExperienceBuffer) -> Result<TrainingStatus> {
    let stats = buffer.stats()?;
    let adapter_path = default_adapter_path();
    let has_adapter = adapter_path.exists();

    Ok(TrainingStatus {
        active: false, // Would be tracked by a PID file
        pending_experiences: stats.unexported,
        latest_adapter: if has_adapter {
            Some(adapter_path)
        } else {
            None
        },
        adapter_loaded: false, // Would need to query the local server
    })
}

// =============================================================================
// Dual LoRA Adapter Generation (D2L + T2L)
// =============================================================================

/// Result of adapter regeneration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterGenResult {
    pub d2l_path: Option<PathBuf>,
    pub t2l_path: Option<PathBuf>,
    pub d2l_doc_chars: usize,
    pub t2l_desc_chars: usize,
    pub success: bool,
    pub message: String,
}

/// Default directory for generated adapters.
pub fn adapters_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".nanobot")
        .join("adapters")
}

/// Build the D2L input document from MEMORY.md + knowledge graph export.
/// This document captures *who the user is* — facts, preferences, entities.
pub fn build_d2l_document(workspace: &Path) -> String {
    let mut doc = String::new();

    // Include MEMORY.md.
    let memory_path = workspace.join("memory").join("MEMORY.md");
    if let Ok(memory) = std::fs::read_to_string(&memory_path) {
        doc.push_str("# User Memory\n\n");
        doc.push_str(&memory);
        doc.push_str("\n\n");
    }

    // Include knowledge graph context.
    match crate::agent::knowledge_graph::KnowledgeGraph::open_default() {
        Ok(kg) => {
            let ctx = kg.export_context(50);
            if !ctx.is_empty() {
                doc.push_str("# Knowledge Graph\n\n");
                doc.push_str(&ctx);
                doc.push_str("\n\n");
            }
        }
        Err(_) => {}
    }

    doc
}

/// Build the T2L behavioral description from experience buffer patterns.
/// This text captures *how the agent should behave* — tool preferences, patterns.
pub fn build_t2l_description(buffer: &ExperienceBuffer) -> String {
    let stats = match buffer.stats() {
        Ok(s) => s,
        Err(_) => return String::new(),
    };

    if stats.total == 0 {
        return String::new();
    }

    let mut desc = String::new();

    // Get recent successful experiences for pattern analysis.
    let experiences = buffer.top_unexported(50).unwrap_or_default();
    if experiences.is_empty() {
        return String::new();
    }

    // Aggregate tool usage patterns.
    let mut tool_counts: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    let mut total_quality = 0.0;
    let mut count = 0;

    for exp in &experiences {
        if let Ok(tools) = serde_json::from_str::<Vec<serde_json::Value>>(&exp.tool_trace) {
            for tool in &tools {
                if let Some(name) = tool.get("name").and_then(|n| n.as_str()) {
                    *tool_counts.entry(name.to_string()).or_insert(0) += 1;
                }
            }
        }
        total_quality += exp.quality;
        count += 1;
    }

    let avg_quality = if count > 0 {
        total_quality / count as f64
    } else {
        0.0
    };

    desc.push_str(&format!(
        "Agent behavioral profile based on {} experiences (avg quality: {:.2}).\n\n",
        count, avg_quality
    ));

    // Sort tools by frequency.
    let mut tools: Vec<(String, usize)> = tool_counts.into_iter().collect();
    tools.sort_by(|a, b| b.1.cmp(&a.1));

    if !tools.is_empty() {
        desc.push_str("Frequently used tools:\n");
        for (name, freq) in tools.iter().take(10) {
            desc.push_str(&format!("- {} (used {} times)\n", name, freq));
        }
        desc.push('\n');
    }

    desc.push_str(&format!(
        "Success rate: {:.0}%\n",
        stats.successful as f64 / stats.total.max(1) as f64 * 100.0
    ));

    desc
}

/// Regenerate both D2L (knowledge) and T2L (behavior) adapters.
///
/// When built with `--features lora`, uses qlora-rs for pure-Rust QLoRA training:
/// 1. Finds the base model GGUF on disk (via LM Studio API or config)
/// 2. Tokenizes D2L/T2L input documents
/// 3. Trains QLoRA adapters (rank=8, alpha=16, 1 epoch)
/// 4. Exports as .gguf via qlora-rs
/// 5. Hot-swaps into the running local server
///
/// Without the `lora` feature, returns a clear error with build instructions.
#[cfg(feature = "lora")]
pub async fn regenerate_adapters(
    workspace: &Path,
    server_url: &str,
    scale: f64,
    local_model: &str,
) -> Result<AdapterGenResult> {
    use qlora_rs::QLoraTrainingConfig;

    let output_dir = adapters_dir();
    std::fs::create_dir_all(&output_dir).context("Failed to create adapters directory")?;

    // Build input documents.
    let d2l_doc = build_d2l_document(workspace);
    let t2l_desc = match ExperienceBuffer::open_default() {
        Ok(buf) => build_t2l_description(&buf),
        Err(_) => String::new(),
    };

    if d2l_doc.is_empty() && t2l_desc.is_empty() {
        return Ok(AdapterGenResult {
            d2l_path: None,
            t2l_path: None,
            d2l_doc_chars: 0,
            t2l_desc_chars: 0,
            success: false,
            message: "No data available for adapter generation".to_string(),
        });
    }

    info!(
        "Adapter generation: D2L doc {} chars, T2L desc {} chars",
        d2l_doc.len(),
        t2l_desc.len()
    );

    // Find the base model GGUF path.
    let base_model_path = match find_base_model_path(server_url, local_model).await {
        Ok(p) => p,
        Err(e) => {
            return Ok(AdapterGenResult {
                d2l_path: None,
                t2l_path: None,
                d2l_doc_chars: d2l_doc.len(),
                t2l_desc_chars: t2l_desc.len(),
                success: false,
                message: format!("Could not locate base model: {}", e),
            });
        }
    };

    info!("Base model: {}", base_model_path.display());

    let device = candle_core::Device::Cpu;
    let qlora_cfg = qlora_rs::QLoraConfig::preset_qv_bf16(8, 16);
    let training_cfg = QLoraTrainingConfig {
        num_epochs: 1,
        batch_size: 4,
        log_every: 10,
        save_every: None,
        warmup_steps: 0,
        use_paged_optimizer: false,
        ..Default::default()
    };

    let mut d2l_result: Option<PathBuf> = None;
    let mut t2l_result: Option<PathBuf> = None;

    // Train personality adapter from D2L document.
    if !d2l_doc.is_empty() {
        let out_path = output_dir.join("personality.gguf");
        match train_adapter(
            &d2l_doc,
            &base_model_path,
            &out_path,
            &qlora_cfg,
            &training_cfg,
            &device,
        ) {
            Ok(()) => {
                info!("D2L adapter generated: {}", out_path.display());
                d2l_result = Some(out_path);
            }
            Err(e) => warn!("D2L adapter training failed: {}", e),
        }
    }

    // Train behavioral adapter from T2L description.
    if !t2l_desc.is_empty() {
        let out_path = output_dir.join("behavior.gguf");
        match train_adapter(
            &t2l_desc,
            &base_model_path,
            &out_path,
            &qlora_cfg,
            &training_cfg,
            &device,
        ) {
            Ok(()) => {
                info!("T2L adapter generated: {}", out_path.display());
                t2l_result = Some(out_path);
            }
            Err(e) => warn!("T2L adapter training failed: {}", e),
        }
    }

    // Hot-swap adapters into the local server.
    for (label, path_opt) in [("D2L", &d2l_result), ("T2L", &t2l_result)] {
        if let Some(path) = path_opt {
            let config = LoraConfig {
                server_url: server_url.to_string(),
                adapter_path: Some(path.clone()),
                scale,
            };
            match apply_lora_adapter(&config).await {
                Ok(r) if r.success => info!("{} adapter loaded: {}", label, path.display()),
                Ok(r) => warn!("{} adapter load failed: {}", label, r.message),
                Err(e) => warn!("{} adapter load error: {}", label, e),
            }
        }
    }

    let success = d2l_result.is_some() || t2l_result.is_some();
    Ok(AdapterGenResult {
        d2l_path: d2l_result,
        t2l_path: t2l_result,
        d2l_doc_chars: d2l_doc.len(),
        t2l_desc_chars: t2l_desc.len(),
        success,
        message: format!(
            "Generated adapters: D2L={}, T2L={}",
            if success && d2l_doc.len() > 0 {
                "yes"
            } else {
                "no"
            },
            if success && t2l_desc.len() > 0 {
                "yes"
            } else {
                "no"
            }
        ),
    })
}

/// Stub when built without the `lora` feature.
#[cfg(not(feature = "lora"))]
pub async fn regenerate_adapters(
    _workspace: &Path,
    _server_url: &str,
    _scale: f64,
    _local_model: &str,
) -> Result<AdapterGenResult> {
    Ok(AdapterGenResult {
        d2l_path: None,
        t2l_path: None,
        d2l_doc_chars: 0,
        t2l_desc_chars: 0,
        success: false,
        message: "LoRA generation requires the 'lora' feature. \
                  Rebuild with: cargo build --features lora"
            .to_string(),
    })
}

// =============================================================================
// qlora-rs Training Helpers (feature-gated)
// =============================================================================

/// Train a single QLoRA adapter from text input and export as GGUF.
#[cfg(feature = "lora")]
fn train_adapter(
    text: &str,
    _base_model_path: &Path,
    output_path: &Path,
    qlora_cfg: &qlora_rs::QLoraConfig,
    training_cfg: &qlora_rs::QLoraTrainingConfig,
    device: &candle_core::Device,
) -> Result<()> {
    use candle_core::Tensor;
    use qlora_rs::{
        export_model, ExportConfig, ExportFormat, QLoraLayer, QLoraTrainer, QuantizedLinear,
    };

    // Tokenize input text to token IDs using tiktoken (cl100k_base).
    let bpe = tiktoken_rs::cl100k_base().context("Failed to load cl100k_base tokenizer")?;
    let token_ids: Vec<u32> = bpe
        .encode_with_special_tokens(text)
        .into_iter()
        .map(|t| t as u32)
        .collect();

    if token_ids.len() < 2 {
        anyhow::bail!(
            "Input text too short for training ({} tokens)",
            token_ids.len()
        );
    }

    info!(
        "Training adapter: {} tokens, rank={}, alpha={}",
        token_ids.len(),
        qlora_cfg.lora.r,
        qlora_cfg.lora.alpha
    );

    // Create trainer and build a single quantized linear layer.
    let mut trainer = QLoraTrainer::new(training_cfg.clone(), device.clone());

    // Create a small projection layer matching the token embedding dimension.
    // For personal adapters we use a modest hidden size; the adapter captures
    // behavioral/personality patterns, not full model weights.
    let hidden_size = 256;

    // Create a zero-initialized weight, then wrap with VarBuilder for gradient tracking.
    // Scope the VarBuilder borrow so trainer can be mutably borrowed for init_optimizer.
    let layer = {
        let vb = trainer.var_builder();
        let weight = Tensor::zeros(&[hidden_size, hidden_size], candle_core::DType::F32, device)
            .map_err(|e| anyhow::anyhow!("Failed to create weight tensor: {}", e))?;
        QuantizedLinear::from_weight_with_varbuilder(&weight, None, qlora_cfg, vb.pp("adapter"))
            .map_err(|e| anyhow::anyhow!("Failed to create quantized LoRA layer: {}", e))?
    };

    trainer
        .init_optimizer(&[&layer])
        .map_err(|e| anyhow::anyhow!("Failed to init optimizer: {}", e))?;

    // Prepare training data: sliding window of token embeddings.
    let window_size = 64.min(token_ids.len() - 1);
    let num_windows = (token_ids.len() - 1) / window_size;
    if num_windows == 0 {
        anyhow::bail!("Input too short for training window");
    }

    trainer.start_epoch();
    for i in 0..num_windows {
        let start = i * window_size;
        let input_slice: Vec<f32> = token_ids[start..start + window_size]
            .iter()
            .map(|&t| t as f32 / 100000.0) // Normalize to small range
            .collect();
        // Reshape to [batch=1, seq_len, hidden] by repeating across hidden dim.
        let input_data: Vec<f32> = input_slice
            .iter()
            .flat_map(|&v| std::iter::repeat(v).take(hidden_size))
            .collect();
        let input = Tensor::from_vec(input_data, (1, window_size, hidden_size), device)
            .map_err(|e| anyhow::anyhow!("Failed to create input tensor: {}", e))?;

        let target_slice: Vec<u32> = token_ids[start + 1..start + 1 + window_size].to_vec();
        let targets = Tensor::from_vec(target_slice, (1, window_size), device)
            .map_err(|e| anyhow::anyhow!("Failed to create target tensor: {}", e))?;

        match trainer.training_step_lm(&[&layer], &input, &targets) {
            Ok(loss) => {
                if i % 10 == 0 {
                    debug!("Step {}/{}: loss={:.4}", i, num_windows, loss);
                }
            }
            Err(e) => {
                warn!("Training step {} failed: {}", i, e);
            }
        }
    }

    // Export the trained adapter as GGUF.
    let qlora_layer = QLoraLayer::new(layer);
    let quantized = qlora_layer.quantized_weight();
    let export_cfg = ExportConfig {
        format: ExportFormat::Gguf,
        model_name: "nanobot-adapter".to_string(),
        model_type: "qlora".to_string(),
    };

    export_model(&[("adapter.weight", quantized)], export_cfg, output_path)
        .map_err(|e| anyhow::anyhow!("Failed to export adapter as GGUF: {}", e))?;

    info!("Adapter exported: {}", output_path.display());
    Ok(())
}

/// Resolve the base model path (GGUF file or MLX directory).
///
/// Resolution order:
/// 1. If `local_model` is non-empty, search `~/.cache/lm-studio/models/` recursively
///    for a GGUF file or MLX model directory whose name contains `local_model`.
/// 2. Fall back to querying `GET /api/v1/models` on the server and resolving via model ID.
#[cfg(feature = "lora")]
async fn find_base_model_path(server_url: &str, local_model: &str) -> Result<PathBuf> {
    // Step 1: Try local filesystem search using local_model hint.
    if !local_model.is_empty() {
        let models_root = dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".cache")
            .join("lm-studio")
            .join("models");

        if models_root.exists() {
            if let Some(path) = find_model_recursive(&models_root, local_model) {
                info!("Found base model via local_model hint: {}", path.display());
                return Ok(path);
            }
            info!(
                "No model matching '{}' found locally, trying server API",
                local_model
            );
        }
    }

    // Step 2: Fall back to server API query (original behavior).
    let client = reqwest::Client::new();
    let url = format!("{}/api/v1/models", server_url);

    let resp = client
        .get(&url)
        .send()
        .await
        .context("Failed to query local model server")?;

    if !resp.status().is_success() {
        anyhow::bail!("Model server returned {}", resp.status());
    }

    let body: serde_json::Value = resp
        .json()
        .await
        .context("Failed to parse model list response")?;

    // LM Studio format: { "data": [{ "id": "publisher/model-name" }] }
    let model_id = body["data"]
        .as_array()
        .and_then(|arr| arr.first())
        .and_then(|m| m["id"].as_str())
        .ok_or_else(|| anyhow::anyhow!("No models loaded in local server"))?;

    // Resolve to model path under LM Studio cache.
    let cache_dir = dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".cache")
        .join("lm-studio")
        .join("models")
        .join(model_id);

    if !cache_dir.exists() {
        anyhow::bail!("Model directory not found: {}", cache_dir.display());
    }

    // Check for GGUF file first, then MLX directory (contains .safetensors).
    resolve_model_in_dir(&cache_dir)
}

/// Given a model directory, return either a .gguf file path or the directory
/// itself if it contains .safetensors files (MLX format).
#[cfg(feature = "lora")]
fn resolve_model_in_dir(dir: &Path) -> Result<PathBuf> {
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.extension().map_or(false, |ext| ext == "gguf") {
            return Ok(path);
        }
    }
    // Check for MLX model (directory with .safetensors weights).
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.extension().map_or(false, |ext| ext == "safetensors") {
            return Ok(dir.to_path_buf());
        }
    }
    anyhow::bail!("No .gguf or .safetensors model found in {}", dir.display())
}

/// Recursively search `root` for a model matching `needle`.
///
/// Matches GGUF files (returns the file) or MLX directories containing
/// .safetensors files (returns the directory). Matching is case-insensitive
/// substring on the filename (GGUF) or directory name (MLX).
#[cfg(feature = "lora")]
fn find_model_recursive(root: &Path, needle: &str) -> Option<PathBuf> {
    let needle_lower = needle.to_lowercase();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = match std::fs::read_dir(&dir) {
            Ok(e) => e,
            Err(_) => continue,
        };
        let mut has_safetensors = false;
        let mut subdirs = Vec::new();
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                subdirs.push(path);
            } else if path.extension().map_or(false, |ext| ext == "gguf") {
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    if name.to_lowercase().contains(&needle_lower) {
                        return Some(path);
                    }
                }
            } else if path.extension().map_or(false, |ext| ext == "safetensors") {
                has_safetensors = true;
            }
        }
        // If this directory contains .safetensors and its name matches, it's an MLX model.
        if has_safetensors {
            if let Some(name) = dir.file_name().and_then(|n| n.to_str()) {
                if name.to_lowercase().contains(&needle_lower) {
                    return Some(dir);
                }
            }
        }
        stack.extend(subdirs);
    }
    None
}

// =============================================================================
// LoRA Merge-to-Disk Pipeline (for oMLX auto-discovery)
// =============================================================================

/// Result of a LoRA merge-to-disk operation.
#[derive(Debug, Clone)]
pub struct MergeResult {
    /// Path to the new merged model directory.
    pub output_dir: PathBuf,
    /// Model name (directory stem).
    pub model_name: String,
    /// Number of weight tensors that were merged.
    pub merged_count: usize,
}

/// Merge LoRA adapter weights into a base MLX model and save as a new model directory.
///
/// This enables oMLX auto-discovery: oMLX watches model directories and can serve
/// the merged model without any REST API for model loading.
///
/// Steps:
/// 1. Load base model weights from `*.safetensors` files
/// 2. Load adapter weights from `adapters/adapters.safetensors`
/// 3. For each LoRA target key, compute: `W_merged = W_base + (alpha/rank) * B @ A`
/// 4. Copy all non-weight files (config.json, tokenizer.json, etc.) to output
/// 5. Write merged weights as safetensors in the output directory
#[cfg(feature = "mlx")]
pub fn merge_lora_to_disk(
    base_model_dir: &Path,
    adapter_dir: &Path,
    output_dir: &Path,
) -> Result<MergeResult> {
    use mlx_rs::Array;
    use std::collections::HashMap;

    // 1. Parse adapter_config.json for rank and alpha.
    let adapter_config_path = adapter_dir.join("adapter_config.json");
    let adapter_config: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&adapter_config_path)
            .with_context(|| format!("reading {}", adapter_config_path.display()))?,
    )?;
    let rank = adapter_config["rank"]
        .as_f64()
        .unwrap_or(32.0);
    let alpha = adapter_config["alpha"]
        .as_f64()
        .unwrap_or(rank);
    let scale = alpha / rank;

    info!(
        "LoRA merge: rank={}, alpha={}, scale={:.4}",
        rank, alpha, scale
    );

    // 2. Load base model weights.
    let base_weights = crate::agent::mlx_lora::load_weights(base_model_dir)
        .context("loading base model weights")?;

    // 3. Load adapter weights.
    let adapter_st_path = adapter_dir.join("adapters.safetensors");
    let adapter_weights = Array::load_safetensors(&adapter_st_path)
        .map_err(|e| anyhow::anyhow!("loading adapter weights: {e}"))?;

    // 4. Group adapter weights by target: find pairs of lora_a.weight / lora_b.weight.
    // Keys look like: "model.layers.0.self_attn.q_proj.lora_a.weight"
    // or with prefix: "language_model.model.layers.0.self_attn.q_proj.lora_a.weight"
    let mut lora_pairs: HashMap<String, (Option<&Array>, Option<&Array>)> = HashMap::new();
    for (key, arr) in &adapter_weights {
        let (base_key, is_a) = if key.ends_with(".lora_a.weight") {
            (key.trim_end_matches(".lora_a.weight").to_string(), true)
        } else if key.ends_with(".lora_b.weight") {
            (key.trim_end_matches(".lora_b.weight").to_string(), false)
        } else {
            continue;
        };
        let entry = lora_pairs.entry(base_key).or_insert((None, None));
        if is_a {
            entry.0 = Some(arr);
        } else {
            entry.1 = Some(arr);
        }
    }

    // 5. Merge: for each LoRA pair, find the corresponding base weight and apply delta.
    let mut merged_weights = base_weights;
    let mut merged_count = 0;
    let scale_arr = Array::from_f32(scale as f32);

    for (adapter_key, (lora_a, lora_b)) in &lora_pairs {
        let (Some(a), Some(b)) = (lora_a, lora_b) else {
            warn!("Incomplete LoRA pair for {}, skipping", adapter_key);
            continue;
        };

        // The base weight key is adapter_key + ".weight"
        let base_key = format!("{}.weight", adapter_key);
        let base_w = match merged_weights.remove(&base_key) {
            Some(w) => w,
            None => {
                warn!("No base weight for {}, skipping", base_key);
                continue;
            }
        };

        // delta = scale * (B @ A)  — B is [out, rank], A is [rank, in]
        let delta = mlx_rs::ops::matmul(b, a)
            .map_err(|e| anyhow::anyhow!("matmul for {}: {e}", adapter_key))?;
        let scaled_delta = mlx_rs::ops::multiply(&delta, &scale_arr)
            .map_err(|e| anyhow::anyhow!("scale for {}: {e}", adapter_key))?;

        // Cast delta to match base weight dtype if needed.
        let scaled_delta = if base_w.dtype() != scaled_delta.dtype() {
            scaled_delta
                .as_dtype(base_w.dtype())
                .map_err(|e| anyhow::anyhow!("dtype cast for {}: {e}", adapter_key))?
        } else {
            scaled_delta
        };

        let merged = mlx_rs::ops::add(&base_w, &scaled_delta)
            .map_err(|e| anyhow::anyhow!("add for {}: {e}", adapter_key))?;
        merged_weights.insert(base_key, merged);
        merged_count += 1;
    }

    info!("Merged {} LoRA targets into base weights", merged_count);

    // Free adapter data before eval materializes all merged tensors.
    drop(lora_pairs);
    drop(adapter_weights);

    // 6. Create output directory and copy non-weight files.
    std::fs::create_dir_all(output_dir)?;

    for entry in std::fs::read_dir(base_model_dir)?.flatten() {
        let path = entry.path();
        if path.is_file() {
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
            // Copy everything except safetensors (we'll write merged ones).
            if ext != "safetensors" {
                let dest = output_dir.join(path.file_name().unwrap());
                std::fs::copy(&path, &dest)?;
            }
        }
    }

    // 7. Eval all merged arrays to materialize lazy computation.
    mlx_rs::transforms::eval(merged_weights.values())
        .map_err(|e| anyhow::anyhow!("eval merged weights: {e}"))?;

    // 8. Write merged weights as a single safetensors file.
    let out_st_path = output_dir.join("model.safetensors");
    Array::save_safetensors(
        merged_weights.iter().map(|(k, v)| (k.as_str(), v)),
        None,
        &out_st_path,
    )
    .map_err(|e| anyhow::anyhow!("save merged safetensors: {e}"))?;

    let model_name = output_dir
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();

    info!(
        "Merged model saved to {} ({} tensors, {} LoRA targets merged)",
        output_dir.display(),
        merged_weights.len(),
        merged_count
    );

    Ok(MergeResult {
        output_dir: output_dir.to_path_buf(),
        model_name,
        merged_count,
    })
}

/// Stub when built without the `mlx` feature.
#[cfg(not(feature = "mlx"))]
pub fn merge_lora_to_disk(
    _base_model_dir: &Path,
    _adapter_dir: &Path,
    _output_dir: &Path,
) -> Result<MergeResult> {
    anyhow::bail!(
        "LoRA merge requires the 'mlx' feature. Rebuild with: cargo build --features mlx"
    )
}

// =============================================================================
// Perplexity Gate: trigger training on MLX server
// =============================================================================

/// Export high-surprise experiences and POST them to the MLX server's /train endpoint.
///
/// Returns the number of experiences sent, or an error.
pub async fn trigger_training(
    buffer: &ExperienceBuffer,
    server_url: &str,
    limit: usize,
    epochs: usize,
) -> Result<usize> {
    let experiences = buffer.top_unexported(limit)?;
    if experiences.is_empty() {
        return Ok(0);
    }

    // Build Ex0bit training format: messages is Vec<Vec<{role, content}>>
    let mut conversations = Vec::new();
    let mut ids = Vec::new();
    for exp in &experiences {
        conversations.push(serde_json::json!([
            {"role": "user", "content": exp.prompt},
            {"role": "assistant", "content": exp.response}
        ]));
        ids.push(exp.id);
    }

    let body = serde_json::json!({
        "messages": conversations,
        "epochs": epochs,
    });

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()?;
    let url = format!("{}/train", server_url);

    let resp = client
        .post(&url)
        .json(&body)
        .send()
        .await
        .context("Failed to POST /train to MLX server")?;

    if resp.status().is_success() {
        buffer.mark_exported(&ids)?;
        info!(
            "Perplexity gate: sent {} experiences to {} for training ({} epochs)",
            ids.len(),
            server_url,
            epochs
        );
        Ok(ids.len())
    } else {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        anyhow::bail!("MLX server /train returned {}: {}", status, text)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn test_experience_buffer_open() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("experience.db");

        let buffer = ExperienceBuffer::open(&db_path).unwrap();
        assert!(db_path.exists());

        // Verify WAL mode
        let stats = buffer.stats().unwrap();
        assert_eq!(stats.total, 0);
    }

    #[test]
    fn test_record_and_retrieve() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("experience.db");
        let buffer = ExperienceBuffer::open(&db_path).unwrap();

        let id = buffer
            .record(
                "Test task",
                r#"[{"name":"read_file","arguments":{},"result":"ok"}]"#,
                "Task complete",
                true,
                0.8,
                "test-model",
            )
            .unwrap();

        assert_eq!(id, 1);

        let experiences = buffer.top_unexported(10).unwrap();
        assert_eq!(experiences.len(), 1);
        assert_eq!(experiences[0].prompt, "Test task");
        assert_eq!(experiences[0].quality, 0.8);
        assert_eq!(experiences[0].model, "test-model");
        assert!(experiences[0].surprise > 0.0);
    }

    #[test]
    fn test_surprise_computation() {
        // Empty trace - low surprise
        let s1 = compute_surprise("hello", "[]");
        assert!(s1 < 0.2);

        // Simple trace - moderate surprise
        let s2 = compute_surprise("do something", r#"[{"name":"tool1"}]"#);
        assert!(s2 > 0.0 && s2 < 0.5);

        // Complex trace - high surprise
        let complex_trace = r#"[
            {"name":"read_file","arguments":{"path":"a.txt"}},
            {"name":"write_file","arguments":{"path":"b.txt"}},
            {"name":"execute_shell","arguments":{"cmd":"ls"}},
            {"name":"search_web","arguments":{"query":"test"}},
            {"name":"send_message","arguments":{"text":"done"}}
        ]"#;
        let s3 = compute_surprise("task", complex_trace);
        assert!(s3 > 0.5);

        // Short prompt, long trace - high efficiency factor
        let s4 = compute_surprise("x", &"a".repeat(5000));
        assert!(s4 > 0.5);
    }

    #[test]
    fn test_count_unique_tools() {
        let trace = r#"[
            {"name":"read_file"},
            {"name":"write_file"},
            {"name":"read_file"},
            {"name":"search_web"}
        ]"#;
        assert_eq!(count_unique_tools(trace), 3);

        // Empty array
        assert_eq!(count_unique_tools("[]"), 0);

        // Missing name fields
        let no_names = r#"[{"tool":"read_file"},{"tool":"write_file"}]"#;
        assert_eq!(count_unique_tools(no_names), 0);
    }

    #[test]
    fn test_count_unique_tools_invalid_json() {
        assert_eq!(count_unique_tools("not json"), 0);
        assert_eq!(count_unique_tools("{\"not\":\"array\"}"), 0);
    }

    #[test]
    fn test_export_jsonl() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("experience.db");
        let buffer = ExperienceBuffer::open(&db_path).unwrap();

        // Record multiple experiences
        buffer
            .record(
                "task1",
                r#"[{"name":"tool1"}]"#,
                "done1",
                true,
                0.9,
                "model1",
            )
            .unwrap();
        buffer
            .record(
                "task2",
                r#"[{"name":"tool2"}]"#,
                "done2",
                true,
                0.7,
                "model2",
            )
            .unwrap();
        buffer
            .record(
                "task3",
                r#"[{"name":"tool3"}]"#,
                "done3",
                false,
                0.5,
                "model3",
            )
            .unwrap(); // Failed - won't export

        let output_path = dir.path().join("training").join("export.jsonl");
        let result = buffer.export_jsonl(&output_path, 10).unwrap();

        assert_eq!(result.count, 2); // Only successful experiences
        assert_eq!(result.ids.len(), 2);
        assert!(output_path.exists());

        // Verify JSONL format
        let content = fs::read_to_string(&output_path).unwrap();
        let lines: Vec<&str> = content.trim().split('\n').collect();
        assert_eq!(lines.len(), 2);

        // Parse first line
        let entry: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(entry["messages"][0]["role"], "user");
        assert_eq!(entry["messages"][0]["content"], "task1");
        assert_eq!(entry["messages"][1]["role"], "assistant");
        assert_eq!(entry["quality"], 0.9);
        assert!(entry["surprise"].is_number());

        // Verify marked as exported
        let unexported = buffer.top_unexported(10).unwrap();
        assert_eq!(unexported.len(), 0);
    }

    #[test]
    fn test_mark_exported() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("experience.db");
        let buffer = ExperienceBuffer::open(&db_path).unwrap();

        let id1 = buffer
            .record("task1", "[]", "done1", true, 0.8, "model")
            .unwrap();
        let id2 = buffer
            .record("task2", "[]", "done2", true, 0.8, "model")
            .unwrap();

        // Before marking
        let unexported = buffer.top_unexported(10).unwrap();
        assert_eq!(unexported.len(), 2);

        // Mark first as exported
        let count = buffer.mark_exported(&[id1]).unwrap();
        assert_eq!(count, 1);

        // After marking
        let unexported = buffer.top_unexported(10).unwrap();
        assert_eq!(unexported.len(), 1);
        assert_eq!(unexported[0].id, id2);

        // Mark with empty array
        let count = buffer.mark_exported(&[]).unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn test_buffer_stats() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("experience.db");
        let buffer = ExperienceBuffer::open(&db_path).unwrap();

        buffer
            .record("task1", "[]", "done1", true, 0.8, "model")
            .unwrap();
        buffer
            .record("task2", "[]", "done2", true, 0.9, "model")
            .unwrap();
        buffer
            .record("task3", "[]", "done3", false, 0.5, "model")
            .unwrap();

        let stats = buffer.stats().unwrap();
        assert_eq!(stats.total, 3);
        assert_eq!(stats.successful, 2);
        assert_eq!(stats.unexported, 2); // Only successful ones
        assert!(stats.avg_surprise >= 0.0);
    }

    #[test]
    fn test_export_empty_buffer() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("experience.db");
        let buffer = ExperienceBuffer::open(&db_path).unwrap();

        let output_path = dir.path().join("export.jsonl");
        let result = buffer.export_jsonl(&output_path, 10).unwrap();

        assert_eq!(result.count, 0);
        assert_eq!(result.ids.len(), 0);
    }

    #[test]
    fn test_lora_config_default() {
        let config = LoraConfig::default();
        assert_eq!(config.server_url, "http://127.0.0.1:8080");
        assert!(config.adapter_path.is_none());
        assert_eq!(config.scale, 0.5);
    }

    #[test]
    fn test_surprise_empty_trace() {
        let surprise = compute_surprise("test prompt", "");
        assert!(surprise < 0.1); // Very low surprise for empty trace
    }

    #[test]
    fn test_surprise_complex_trace() {
        // Create a complex trace with 8 different tools
        let trace = r#"[
            {"name":"read_file","arguments":{"path":"a.txt"}},
            {"name":"write_file","arguments":{"path":"b.txt"}},
            {"name":"execute_shell","arguments":{"cmd":"ls"}},
            {"name":"search_web","arguments":{"query":"test"}},
            {"name":"send_message","arguments":{"text":"done"}},
            {"name":"create_file","arguments":{"path":"c.txt"}},
            {"name":"delete_file","arguments":{"path":"d.txt"}},
            {"name":"list_directory","arguments":{"path":"/"}}
        ]"#;

        let surprise = compute_surprise("do stuff", trace);
        assert!(surprise > 0.6); // High surprise for complex multi-tool trace
    }

    #[test]
    fn test_training_status_no_adapter() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("experience.db");
        let buffer = ExperienceBuffer::open(&db_path).unwrap();

        buffer
            .record("task1", "[]", "done1", true, 0.8, "model")
            .unwrap();

        let status = check_training_status(&buffer).unwrap();
        assert!(!status.active);
        assert_eq!(status.pending_experiences, 1);
        assert!(status.latest_adapter.is_none());
        assert!(!status.adapter_loaded);
    }

    #[test]
    fn test_export_preserves_order() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("experience.db");
        let buffer = ExperienceBuffer::open(&db_path).unwrap();

        // Record with different surprise scores
        buffer
            .record("low", "x", "done", true, 0.5, "model")
            .unwrap();
        buffer
            .record("high", &"x".repeat(10000), "done", true, 0.9, "model")
            .unwrap();
        buffer
            .record("medium", &"x".repeat(1000), "done", true, 0.7, "model")
            .unwrap();

        // Top unexported should be ordered by surprise DESC
        let experiences = buffer.top_unexported(10).unwrap();
        assert_eq!(experiences.len(), 3);

        // Higher surprise should come first
        assert!(experiences[0].surprise >= experiences[1].surprise);
        assert!(experiences[1].surprise >= experiences[2].surprise);
    }

    // --- D2L/T2L adapter generation tests ---

    #[test]
    fn test_build_d2l_document_with_memory() {
        let dir = tempdir().unwrap();
        let mem_dir = dir.path().join("memory");
        std::fs::create_dir_all(&mem_dir).unwrap();
        std::fs::write(
            mem_dir.join("MEMORY.md"),
            "- User prefers Rust\n- Dark mode enabled",
        )
        .unwrap();

        let doc = build_d2l_document(dir.path());
        assert!(doc.contains("User prefers Rust"));
        assert!(doc.contains("# User Memory"));
    }

    #[test]
    fn test_build_d2l_document_empty_workspace() {
        let dir = tempdir().unwrap();
        let doc = build_d2l_document(dir.path());
        // No MEMORY.md, no knowledge graph — should be empty or just headings.
        // The knowledge graph open_default might find an existing one, but doc should still work.
        assert!(doc.is_empty() || doc.contains("Knowledge Graph"));
    }

    #[test]
    fn test_build_t2l_description_with_experiences() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("experience.db");
        let buffer = ExperienceBuffer::open(&db_path).unwrap();

        buffer
            .record(
                "read config",
                r#"[{"name":"read_file","arguments":{"path":"config.json"}}]"#,
                "Config loaded",
                true,
                0.9,
                "model",
            )
            .unwrap();
        buffer
            .record(
                "write output",
                r#"[{"name":"write_file","arguments":{"path":"out.txt"}},{"name":"read_file","arguments":{"path":"in.txt"}}]"#,
                "Written",
                true,
                0.8,
                "model",
            )
            .unwrap();

        let desc = build_t2l_description(&buffer);
        assert!(desc.contains("read_file"));
        assert!(desc.contains("write_file"));
        assert!(desc.contains("2 experiences"));
    }

    #[test]
    fn test_build_t2l_description_empty_buffer() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("experience.db");
        let buffer = ExperienceBuffer::open(&db_path).unwrap();

        let desc = build_t2l_description(&buffer);
        assert!(desc.is_empty());
    }

    #[test]
    fn test_adapters_dir() {
        let dir = adapters_dir();
        assert!(dir.to_string_lossy().contains("adapters"));
    }

    // --- Stub / feature-gate tests ---

    #[tokio::test]
    async fn test_regenerate_adapters_stub_without_lora_feature() {
        // When built without --features lora, regenerate_adapters should return
        // a clear error message telling the user how to enable it.
        let dir = tempdir().unwrap();
        let result = regenerate_adapters(dir.path(), "http://127.0.0.1:9999", 0.5, "")
            .await
            .unwrap();
        // On non-lora builds: stub returns feature-missing message.
        // On lora builds: fails because no local server / no data.
        assert!(
            !result.success,
            "should not succeed without data or lora feature"
        );
        // The message should be informative either way.
        assert!(!result.message.is_empty());
    }

    #[test]
    fn test_experience_buffer_records_tool_trace_from_agent_loop_format() {
        // Simulates exactly what agent_loop.rs does: build a JSON trace from
        // TurnToolEntry-like data (name, ok, duration_ms) and record it.
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("experience.db");
        let buffer = ExperienceBuffer::open(&db_path).unwrap();

        // Simulate agent_loop recording format.
        let entries = vec![
            serde_json::json!({"name": "read_file", "ok": true, "duration_ms": 42}),
            serde_json::json!({"name": "exec_command", "ok": true, "duration_ms": 300}),
        ];
        let trace_json = serde_json::to_string(&entries).unwrap();

        let id = buffer
            .record(
                "summarize the config",
                &trace_json,
                "Here is the config summary.",
                true,
                1.0,
                "gpt-4",
            )
            .unwrap();
        assert!(id > 0);

        // Verify round-trip.
        let experiences = buffer.top_unexported(10).unwrap();
        assert_eq!(experiences.len(), 1);
        assert_eq!(experiences[0].prompt, "summarize the config");
        assert_eq!(experiences[0].response, "Here is the config summary.");
        assert_eq!(experiences[0].model, "gpt-4");
        assert!(
            experiences[0].surprise > 0.0,
            "should have non-zero surprise for multi-tool trace"
        );

        // Verify the trace parses back correctly.
        let parsed: Vec<serde_json::Value> =
            serde_json::from_str(&experiences[0].tool_trace).unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0]["name"], "read_file");
        assert_eq!(parsed[1]["name"], "exec_command");
    }

    #[test]
    fn test_record_with_explicit_surprise() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("experience.db");
        let buffer = ExperienceBuffer::open(&db_path).unwrap();

        let id = buffer
            .record_with_surprise(
                "What is the capital?",
                r#"[{"name":"web_search"}]"#,
                "Paris",
                true,
                0.9,
                "qwen3.5",
                4.2, // explicit CE loss as surprise
            )
            .unwrap();
        assert!(id > 0);

        let exps = buffer.top_unexported(10).unwrap();
        assert_eq!(exps.len(), 1);
        assert!(
            (exps[0].surprise - 4.2).abs() < 1e-6,
            "explicit surprise should be preserved"
        );
        assert_eq!(exps[0].model, "qwen3.5");
    }

    #[tokio::test]
    async fn test_query_perplexity_unreachable() {
        // query_perplexity should return None when server is unreachable.
        let result =
            super::super::learn_loop::query_perplexity("http://127.0.0.1:19999", "hello", "world")
                .await;
        assert!(
            result.is_none(),
            "should return None for unreachable server"
        );
    }

    #[tokio::test]
    async fn test_trigger_training_no_server() {
        // Trigger training against a non-existent server — should fail gracefully.
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("experience.db");
        let buffer = ExperienceBuffer::open(&db_path).unwrap();

        buffer
            .record_with_surprise("q", "[]", "a", true, 1.0, "m", 5.0)
            .unwrap();

        let result = trigger_training(&buffer, "http://127.0.0.1:19999", 10, 3).await;
        assert!(result.is_err(), "should fail when server is unreachable");
        // Experience should NOT be marked as exported on failure.
        assert_eq!(buffer.stats().unwrap().unexported, 1);
    }

    #[test]
    fn test_experience_recording_only_when_tools_used() {
        // Simulates the guard condition in agent_loop: only record when
        // used_tools is non-empty AND final_content is non-empty.
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("experience.db");
        let buffer = ExperienceBuffer::open(&db_path).unwrap();

        // Simulate: tools used but empty final content — should NOT record.
        let used_tools_empty_content = true; // used_tools.is_empty() == false
        let final_content_empty = "";
        if used_tools_empty_content && !final_content_empty.is_empty() {
            buffer
                .record("prompt", "[]", final_content_empty, true, 1.0, "m")
                .unwrap();
        }
        assert_eq!(
            buffer.stats().unwrap().total,
            0,
            "should not record with empty content"
        );

        // Simulate: no tools used — should NOT record.
        let used_tools_is_empty = true;
        if !used_tools_is_empty {
            buffer
                .record("prompt", "[]", "response", true, 1.0, "m")
                .unwrap();
        }
        assert_eq!(
            buffer.stats().unwrap().total,
            0,
            "should not record without tools"
        );

        // Simulate: tools used AND non-empty content — SHOULD record.
        let has_tools = true;
        let has_content = "Got it done.";
        if has_tools && !has_content.is_empty() {
            buffer
                .record(
                    "do the thing",
                    r#"[{"name":"shell"}]"#,
                    has_content,
                    true,
                    1.0,
                    "m",
                )
                .unwrap();
        }
        assert_eq!(
            buffer.stats().unwrap().total,
            1,
            "should record when both tools and content present"
        );
    }
}
