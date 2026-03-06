//! OpenAI-compatible + Ex0bit daemon HTTP server for MLX inference + LoRA training.
//!
//! Exposes both protocols:
//!   - OpenAI: `/v1/chat/completions` (JSON)
//!   - Ex0bit: `/chat` (SSE), `/train`, `/perplexity`, `/reset`, `/status`, `/config`
//!
//! Model lives on a dedicated thread (mlx-rs uses Rc, so !Send).
//! HTTP handlers communicate via mpsc channels.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;

use axum::body::Body;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post, put};
use axum::{Json, Router};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use tokio::sync::oneshot;

use super::mlx_lora::{LoraConfig, ModelConfig, MlxLoraModel, MlxTokenizer};

// ---------------------------------------------------------------------------
// Chat template (ChatML for Qwen)
// ---------------------------------------------------------------------------

const IM_START: &str = "<|im_start|>";
const IM_END: &str = "<|im_end|>";

/// Apply ChatML template with empty think-block prefill (Qwen3.5 style).
pub fn apply_chat_template(messages: &[ChatMessage]) -> String {
    apply_chat_template_with_think(messages, true)
}

/// Apply ChatML template for thinking models: prefill `<think>\n` (matching
/// Qwen3's Jinja template), append `/no_think` to suppress reasoning.
/// Output is stripped of `<think>...</think>` blocks by the provider.
pub fn apply_chat_template_nothink(messages: &[ChatMessage]) -> String {
    apply_chat_template_with_think(messages, false)
}

fn apply_chat_template_with_think(messages: &[ChatMessage], close_think: bool) -> String {
    let mut prompt = String::new();
    for (i, msg) in messages.iter().enumerate() {
        prompt.push_str(IM_START);
        prompt.push_str(&msg.role);
        prompt.push('\n');
        prompt.push_str(&msg.content);
        // For nothink mode, append /no_think to last user message
        if !close_think && msg.role == "user" && i == messages.len() - 1 {
            prompt.push_str(" /no_think");
        }
        prompt.push_str(IM_END);
        prompt.push('\n');
    }
    prompt.push_str(IM_START);
    prompt.push_str("assistant\n");
    if close_think {
        // Qwen3.5: empty think block prefill closes reasoning immediately
        prompt.push_str("<think>\n\n</think>\n\n");
    } else {
        // Qwen3 thinking: open think block (matches Jinja template)
        prompt.push_str("<think>\n");
    }
    prompt
}

// ---------------------------------------------------------------------------
// Request/Response types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

// OpenAI-compat request
#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub stream: Option<bool>,
}

fn default_max_tokens() -> usize { 256 }

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: ChatUsage,
}

#[derive(Debug, Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatResponseMessage,
    pub finish_reason: String,
}

#[derive(Debug, Serialize)]
pub struct ChatResponseMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct ChatUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

// Ex0bit-compat train request: {messages: [[{role,content},...], ...], epochs: N}
#[derive(Debug, Deserialize)]
pub struct ExobitTrainRequest {
    pub messages: Vec<Vec<ChatMessage>>,
    #[serde(default = "default_epochs")]
    pub epochs: usize,
}

fn default_epochs() -> usize { 15 }

// Old train request format (prompt/completion pairs)
#[derive(Debug, Deserialize)]
pub struct SimpleTrainRequest {
    pub data: Vec<SimpleTrainSample>,
    #[serde(default = "default_epochs")]
    pub epochs: usize,
    #[serde(default = "default_early_stop")]
    pub early_stop_loss: f32,
    #[serde(default = "default_patience")]
    pub patience: usize,
}

/// `/perplexity` request: compute mean CE loss for a conversation.
#[derive(Debug, Deserialize)]
pub struct PerplexityRequest {
    pub messages: Vec<ChatMessage>,
}

fn default_early_stop() -> f32 { 0.5 }
fn default_patience() -> usize { 10 }

#[derive(Debug, Deserialize)]
pub struct SimpleTrainSample {
    pub prompt: String,
    pub completion: String,
}

// ---------------------------------------------------------------------------
// Model worker (runs on dedicated thread, owns the !Send model)
// ---------------------------------------------------------------------------

pub enum ModelRequest {
    Chat {
        prompt: String,
        max_tokens: usize,
        temperature: f32,
        reply: oneshot::Sender<Result<(String, usize, usize), String>>,
    },
    Train {
        samples: Vec<(Vec<i32>, Vec<i32>)>,
        epochs: usize,
        early_stop_loss: f32,
        patience: usize,
        /// If None, training is fire-and-forget (async mode for Ex0bit compat).
        reply: Option<oneshot::Sender<Result<(Vec<f32>, usize), String>>>,
    },
    Reset {
        reply: oneshot::Sender<Result<(), String>>,
    },
    /// Forward pass only — return mean CE loss (perplexity = exp(loss)).
    Perplexity {
        tokens: Vec<i32>,
        targets: Vec<i32>,
        reply: oneshot::Sender<Result<f32, String>>,
    },
    /// Export trained LoRA adapters to disk in mlx-lm safetensors format.
    ExportAdapters {
        output_dir: std::path::PathBuf,
        reply: oneshot::Sender<Result<usize, String>>,
    },
    /// Hot-swap LoRA weights from ANE training into the live model.
    #[cfg(feature = "ane")]
    ApplyLoraDeltas {
        deltas: super::ane_mlx_bridge::LoraDeltas,
        reply: Option<oneshot::Sender<Result<usize, String>>>,
    },
}

/// Shared training state visible to status endpoint.
pub struct TrainState {
    training: AtomicBool,
    total_steps: AtomicU32,
    last_loss: Mutex<f32>,
    initial_loss: Mutex<f32>,
    trainable_params: AtomicU32,
}

impl TrainState {
    pub fn new() -> Self {
        TrainState {
            training: AtomicBool::new(false),
            total_steps: AtomicU32::new(0),
            last_loss: Mutex::new(0.0),
            initial_loss: Mutex::new(0.0),
            trainable_params: AtomicU32::new(0),
        }
    }
}

pub fn run_model_worker(
    model_dir: PathBuf,
    cfg: ModelConfig,
    lora_cfg: LoraConfig,
    train_state: Arc<TrainState>,
    rx: std::sync::mpsc::Receiver<ModelRequest>,
    post_train_hook: Option<Box<dyn Fn() + Send>>,
) {
    use mlx_rs::module::ModuleParameters;

    let tokenizer = MlxTokenizer::load(&model_dir)
        .expect("Failed to load tokenizer");

    // Try to load the model; may fail for unsupported architectures.
    // When mlx-lm server handles inference, the in-process model is only
    // needed for training/perplexity — so we can still serve requests.
    let mut model: Option<MlxLoraModel> = match MlxLoraModel::load(&model_dir, &cfg, &lora_cfg) {
        Ok(m) => Some(m),
        Err(e) => {
            tracing::warn!("in-process model load failed: {e} (training unavailable, mlx-lm inference still works)");
            None
        }
    };

    // Count trainable params
    if let Some(ref m) = model {
        let trainable = m.trainable_parameters().flatten();
        let total_trainable: u32 = trainable.values()
            .map(|a| a.shape().iter().product::<i32>() as u32)
            .sum();
        train_state.trainable_params.store(total_trainable, Ordering::Relaxed);
    }

    // Resolve stop tokens from tokenizer (works across Qwen3/3.5 vocab sizes)
    let resolve_token = |text: &str| -> i32 {
        tokenizer.encode(text).ok()
            .and_then(|ids| ids.first().copied())
            .unwrap_or(-1)
    };
    let im_end_id = resolve_token("<|im_end|>");
    let eos_id = tokenizer.eos_token_id().map(|id| id as i32).unwrap_or(-1);
    let mut stop_tokens: Vec<i32> = [im_end_id, eos_id]
        .iter().copied().filter(|&id| id >= 0).collect();
    // For non-thinking models (Qwen3.5), also stop on <think> to prevent re-entering
    // thinking after the empty prefill. Thinking models need <think> to flow through.
    if !cfg.thinking_model {
        let think_id = resolve_token("<think>");
        if think_id >= 0 {
            stop_tokens.push(think_id);
        }
    }

    tracing::info!(model_loaded = model.is_some(), "model worker ready");

    while let Ok(req) = rx.recv() {
        match req {
            ModelRequest::Chat { prompt, max_tokens, temperature, reply } => {
                let result = (|| {
                    let m = model.as_mut().ok_or("in-process model not loaded (use mlx-lm server for inference)")?;
                    let prompt_tokens = tokenizer.encode(&prompt)
                        .map_err(|e| format!("encode: {e}"))?;
                    let prompt_len = prompt_tokens.len();
                    tracing::debug!(prompt_len, max_tokens, temperature, "generate");
                    let t0 = std::time::Instant::now();
                    let generated = m.generate(
                        &prompt_tokens, max_tokens, temperature, &stop_tokens,
                    ).map_err(|e| format!("generate: {e}"))?;
                    let elapsed = t0.elapsed();
                    let gen_len = generated.len();
                    tracing::debug!(
                        gen_len, secs = format!("{:.1}", elapsed.as_secs_f64()),
                        ms_per_tok = format!("{:.0}", elapsed.as_millis() as f64 / gen_len.max(1) as f64),
                        prefill = prompt_len, "generate done"
                    );
                    let text = tokenizer.decode(&generated)
                        .map_err(|e| format!("decode: {e}"))?;
                    Ok((text, prompt_len, gen_len))
                })();
                let _ = reply.send(result);
            }
            ModelRequest::Train { samples, epochs, early_stop_loss, patience, reply } => {
                train_state.training.store(true, Ordering::Relaxed);
                train_state.total_steps.store(0, Ordering::Relaxed);
                *train_state.initial_loss.lock() = 0.0;
                *train_state.last_loss.lock() = 0.0;

                let result = (|| {
                    let m = model.as_mut().ok_or_else(|| "in-process model not loaded (training unavailable for this model)".to_string())?;
                    use mlx_rs::Array;
                    for (i, (toks, tgts)) in samples.iter().enumerate() {
                        tracing::debug!(sample = i, tokens = toks.len(), targets = tgts.len(), "training sample");
                    }
                    let token_batches: Vec<Array> = samples.iter()
                        .map(|(toks, _)| Array::from_slice(toks, &[1, toks.len() as i32]))
                        .collect();
                    let target_batches: Vec<Array> = samples.iter()
                        .map(|(_, tgts)| Array::from_slice(tgts, &[1, tgts.len() as i32]))
                        .collect();

                    // Live progress callback for /status polling
                    let ts = Arc::clone(&train_state);
                    let callback: super::mlx_lora::TrainCallback = Box::new(move |step, loss, _grad_norm| {
                        ts.total_steps.store((step + 1) as u32, Ordering::Relaxed);
                        *ts.last_loss.lock() = loss;
                        if step == 0 {
                            *ts.initial_loss.lock() = loss;
                        }
                    });

                    let losses = super::mlx_lora::train_loop_with_callback(
                        m, &token_batches, &target_batches,
                        &lora_cfg, epochs, early_stop_loss, patience,
                        Some(callback),
                    ).map_err(|e| format!("train: {e}"))?;

                    let steps = losses.len();
                    Ok((losses, steps))
                })();

                // Auto-export adapters after successful training
                if result.is_ok() {
                    if let Some(ref m) = model {
                        let adapter_dir = model_dir.join("adapters");
                        if let Err(e) = super::mlx_lora::export_adapters(m, &lora_cfg, &cfg, &adapter_dir) {
                            tracing::warn!("auto-export adapters failed: {e}");
                        } else if let Some(ref hook) = post_train_hook {
                            hook();
                        }
                    }
                }

                train_state.training.store(false, Ordering::Relaxed);
                if let Some(reply) = reply {
                    let _ = reply.send(result);
                }
            }
            ModelRequest::Perplexity { tokens, targets, reply } => {
                let result = (|| {
                    let m = model.as_mut().ok_or("in-process model not loaded")?;
                    use mlx_rs::Array;
                    let tok_arr = Array::from_slice(&tokens, &[1, tokens.len() as i32]);
                    let tgt_arr = Array::from_slice(&targets, &[1, targets.len() as i32]);
                    let logits = m.forward_logits(&tok_arr)
                        .map_err(|e| format!("forward: {e}"))?;
                    let loss = super::mlx_lora::cross_entropy_loss(&logits, &tgt_arr)
                        .map_err(|e| format!("ce_loss: {e}"))?;
                    let loss_val: f32 = loss.item();
                    Ok(loss_val)
                })();
                let _ = reply.send(result);
            }
            ModelRequest::Reset { reply } => {
                train_state.total_steps.store(0, Ordering::Relaxed);
                *train_state.last_loss.lock() = 0.0;
                *train_state.initial_loss.lock() = 0.0;

                let result = (|| {
                    match MlxLoraModel::load(&model_dir, &cfg, &lora_cfg) {
                        Ok(m) => {
                            let trainable = m.trainable_parameters().flatten();
                            let total: u32 = trainable.values()
                                .map(|a| a.shape().iter().product::<i32>() as u32)
                                .sum();
                            train_state.trainable_params.store(total, Ordering::Relaxed);
                            model = Some(m);
                            Ok(())
                        }
                        Err(e) => {
                            model = None;
                            Err(format!("reload: {e}"))
                        }
                    }
                })();
                let _ = reply.send(result);
            }
            ModelRequest::ExportAdapters { output_dir, reply } => {
                let result = if let Some(ref m) = model {
                    super::mlx_lora::export_adapters(m, &lora_cfg, &cfg, &output_dir)
                        .map_err(|e| format!("export: {e}"))
                } else {
                    Err("in-process model not loaded".into())
                };
                let _ = reply.send(result);
            }
            #[cfg(feature = "ane")]
            ModelRequest::ApplyLoraDeltas { deltas, reply } => {
                use mlx_rs::Array;
                let result: Result<usize, String> = (|| {
                    let m = model.as_mut().ok_or("in-process model not loaded")?;
                    let mut applied = 0usize;
                    for delta in &deltas.layers {
                        if delta.layer_idx >= m.layers.len() {
                            return Err(format!(
                                "layer {} out of range ({})", delta.layer_idx, m.layers.len()
                            ));
                        }
                        let d = &delta.delta;
                        let new_a = Array::from_slice(&d.a, &[d.rank as i32, d.d_in as i32]);
                        let new_b = Array::from_slice(&d.b, &[d.d_out as i32, d.rank as i32]);
                        if m.layers[delta.layer_idx].apply_lora_weights(
                            delta.target.mlx_name(), new_a, new_b,
                        ) {
                            applied += 1;
                        }
                    }
                    Ok(applied)
                })();
                if let Ok(n) = &result {
                    tracing::info!(deltas = n, "applied LoRA deltas from ANE");
                }
                if let Some(tx) = reply {
                    let _ = tx.send(result);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Shared app state
// ---------------------------------------------------------------------------

struct AppState {
    tx: std::sync::mpsc::SyncSender<ModelRequest>,
    model_name: String,
    model_dir: String,
    tokenizer: MlxTokenizer,
    train_state: Arc<TrainState>,
}

// ---------------------------------------------------------------------------
// Helper: tokenize message pairs for training
// ---------------------------------------------------------------------------

pub fn tokenize_conversation(
    tokenizer: &MlxTokenizer,
    messages: &[ChatMessage],
) -> Result<(Vec<i32>, Vec<i32>), String> {
    let mut text = String::new();
    for msg in messages {
        text.push_str(IM_START);
        text.push_str(&msg.role);
        text.push('\n');
        text.push_str(&msg.content);
        text.push_str(IM_END);
        text.push('\n');
    }
    let tokens = tokenizer.encode(&text)
        .map_err(|e| format!("tokenize: {e}"))?;
    if tokens.len() < 2 {
        return Err("conversation too short".into());
    }
    let input = tokens[..tokens.len() - 1].to_vec();
    let target = tokens[1..].to_vec();
    Ok((input, target))
}

// ---------------------------------------------------------------------------
// Handlers: OpenAI-compat
// ---------------------------------------------------------------------------

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    let prompt = apply_chat_template(&req.messages);
    let temperature = req.temperature.unwrap_or(0.0);
    let max_tokens = req.max_tokens;

    let (reply_tx, reply_rx) = oneshot::channel();
    state.tx.send(ModelRequest::Chat {
        prompt,
        max_tokens,
        temperature,
        reply: reply_tx,
    }).map_err(|_| (StatusCode::INTERNAL_SERVER_ERROR, "model worker died".into()))?;

    let result = reply_rx.await
        .map_err(|_| (StatusCode::INTERNAL_SERVER_ERROR, "model worker dropped reply".into()))?
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

    let (text, prompt_tokens, completion_tokens) = result;

    let resp = ChatCompletionResponse {
        id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
        object: "chat.completion".into(),
        created: now_epoch(),
        model: req.model.unwrap_or_else(|| state.model_name.clone()),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatResponseMessage {
                role: "assistant".into(),
                content: text,
            },
            finish_reason: "stop".into(),
        }],
        usage: ChatUsage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    };

    Ok(Json(resp))
}

// ---------------------------------------------------------------------------
// Handlers: Ex0bit daemon compat
// ---------------------------------------------------------------------------

/// `/chat` — SSE streaming response (Ex0bit format).
/// The test script reads `data: {json}` lines with choices[0].delta.content.
async fn chat_sse(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    let prompt = apply_chat_template(&req.messages);
    let temperature = req.temperature.unwrap_or(0.0);
    let max_tokens = req.max_tokens;

    let (reply_tx, reply_rx) = oneshot::channel();
    state.tx.send(ModelRequest::Chat {
        prompt,
        max_tokens,
        temperature,
        reply: reply_tx,
    }).map_err(|_| (StatusCode::INTERNAL_SERVER_ERROR, "model worker died".into()))?;

    let result = reply_rx.await
        .map_err(|_| (StatusCode::INTERNAL_SERVER_ERROR, "model worker dropped reply".into()))?
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

    let (text, _prompt_tokens, _completion_tokens) = result;

    // Build SSE response: one event with full content, then [DONE]
    let chunk = serde_json::json!({
        "choices": [{
            "index": 0,
            "delta": {"content": text},
            "finish_reason": "stop"
        }]
    });
    let body = format!("data: {}\n\ndata: [DONE]\n\n", chunk);

    Ok(axum::response::Response::builder()
        .header("Content-Type", "text/event-stream")
        .header("Cache-Control", "no-cache")
        .body(Body::from(body))
        .unwrap())
}

/// `/status` — Ex0bit daemon status format.
async fn status_exobit(
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    let ts = &state.train_state;
    let last = *ts.last_loss.lock();
    let initial = *ts.initial_loss.lock();
    Json(serde_json::json!({
        "active": true,
        "model_key": state.model_name,
        "model_dir": state.model_dir,
        "training": ts.training.load(Ordering::Relaxed),
        "total_steps": ts.total_steps.load(Ordering::Relaxed),
        "last_loss": if last.is_nan() { 0.0 } else { last },
        "initial_loss": if initial.is_nan() { 0.0 } else { initial },
        "trainable_params": ts.trainable_params.load(Ordering::Relaxed),
        "n_adapters": 1,
        "mamba_architecture": false,
    }))
}

/// `/train` — Ex0bit format: {messages: [[{role,content},...], ...], epochs: N}
/// Training is synchronous (blocks until complete). The test script polls /status.
async fn train_exobit(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ExobitTrainRequest>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    let mut samples = Vec::new();
    for conversation in &req.messages {
        match tokenize_conversation(&state.tokenizer, conversation) {
            Ok(pair) => samples.push(pair),
            Err(_) => continue,
        }
    }

    if samples.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "no valid training samples".into()));
    }

    let injected = samples.len();
    let epochs = req.epochs;
    let max_steps = injected * epochs; // epochs × samples = total gradient steps
    // Early stopping patience: stop if no improvement for 2 full epochs worth of steps.
    let patience = injected * 2;

    // Set training=true BEFORE enqueueing so /status immediately reflects it.
    state.train_state.training.store(true, Ordering::Relaxed);

    // Fire-and-forget: return immediately, training runs in background.
    state.tx.send(ModelRequest::Train {
        samples,
        epochs: max_steps,
        early_stop_loss: 0.0,
        patience,
        reply: None,
    }).map_err(|_| (StatusCode::INTERNAL_SERVER_ERROR, "model worker died".into()))?;

    Ok(Json(serde_json::json!({
        "injected": injected,
        "epochs": epochs,
    })))
}

/// `/reset` — reload model from disk (clear LoRA weights).
async fn reset(
    State(state): State<Arc<AppState>>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    let (reply_tx, reply_rx) = oneshot::channel();
    state.tx.send(ModelRequest::Reset { reply: reply_tx })
        .map_err(|_| (StatusCode::INTERNAL_SERVER_ERROR, "model worker died".into()))?;

    reply_rx.await
        .map_err(|_| (StatusCode::INTERNAL_SERVER_ERROR, "model worker dropped reply".into()))?
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

    Ok(Json(serde_json::json!({"status": "reset"})))
}

/// `/config` PUT — no-op (auto_train not applicable to our server).
async fn config_put() -> impl IntoResponse {
    Json(serde_json::json!({"status": "ok"}))
}

/// `/perplexity` — compute mean CE loss for a conversation (forward pass only).
/// Returns `{"loss": f32, "perplexity": f32, "tokens": usize}`.
async fn perplexity(
    State(state): State<Arc<AppState>>,
    Json(req): Json<PerplexityRequest>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    let (tokens, targets) = tokenize_conversation(&state.tokenizer, &req.messages)
        .map_err(|e| (StatusCode::BAD_REQUEST, e))?;
    let n_tokens = tokens.len();

    let (reply_tx, reply_rx) = oneshot::channel();
    state.tx.send(ModelRequest::Perplexity {
        tokens,
        targets,
        reply: reply_tx,
    }).map_err(|_| (StatusCode::INTERNAL_SERVER_ERROR, "model worker died".into()))?;

    let loss = reply_rx.await
        .map_err(|_| (StatusCode::INTERNAL_SERVER_ERROR, "model worker dropped reply".into()))?
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

    Ok(Json(serde_json::json!({
        "loss": loss,
        "perplexity": (loss as f64).exp(),
        "tokens": n_tokens,
    })))
}

/// `/export` POST — export LoRA adapters to disk in mlx-lm format.
/// Body: `{"output_dir": "/path/to/adapter_dir"}` (optional, defaults to model_dir/adapters).
async fn export_adapters(
    State(state): State<Arc<AppState>>,
    Json(req): Json<serde_json::Value>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    let output_dir = req.get("output_dir")
        .and_then(|v| v.as_str())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(&state.model_dir).join("adapters"));

    let (reply_tx, reply_rx) = oneshot::channel();
    state.tx.send(ModelRequest::ExportAdapters {
        output_dir: output_dir.clone(),
        reply: reply_tx,
    }).map_err(|_| (StatusCode::INTERNAL_SERVER_ERROR, "model worker died".into()))?;

    let n = reply_rx.await
        .map_err(|_| (StatusCode::INTERNAL_SERVER_ERROR, "model worker dropped reply".into()))?
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

    Ok(Json(serde_json::json!({
        "exported": n,
        "output_dir": output_dir.display().to_string(),
    })))
}

async fn health() -> impl IntoResponse {
    Json(serde_json::json!({"status": "ok"}))
}

fn now_epoch() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// ---------------------------------------------------------------------------
// Public API: start the server
// ---------------------------------------------------------------------------

pub struct MlxServerConfig {
    pub model_dir: PathBuf,
    pub model_config: ModelConfig,
    pub lora_config: LoraConfig,
    pub host: String,
    pub port: u16,
}

impl Default for MlxServerConfig {
    fn default() -> Self {
        MlxServerConfig {
            model_dir: PathBuf::new(),
            model_config: ModelConfig::qwen3_5_2b(),
            lora_config: LoraConfig {
                lr: 1e-5, // Ex0bit-compatible: 1e-5 for real conversation training
                ..LoraConfig::default()
            },
            host: "127.0.0.1".into(),
            port: 8766,
        }
    }
}

pub async fn serve(config: MlxServerConfig) -> Result<(), anyhow::Error> {
    let tokenizer = MlxTokenizer::load(&config.model_dir)?;
    let model_name = config.model_dir
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "mlx-model".into());

    let train_state = Arc::new(TrainState::new());

    let (tx, rx) = std::sync::mpsc::sync_channel::<ModelRequest>(4);

    let model_dir = config.model_dir.clone();
    let cfg = config.model_config;
    let lora_cfg_worker = config.lora_config.clone();
    let ts_clone = Arc::clone(&train_state);

    std::thread::Builder::new()
        .name("mlx-model-worker".into())
        .spawn(move || {
            run_model_worker(model_dir, cfg, lora_cfg_worker, ts_clone, rx, None);
        })?;

    let state = Arc::new(AppState {
        tx,
        model_name: model_name.clone(),
        model_dir: config.model_dir.display().to_string(),
        tokenizer,
        train_state,
    });

    let app = Router::new()
        // OpenAI-compat
        .route("/v1/chat/completions", post(chat_completions))
        // Ex0bit daemon compat
        .route("/chat", post(chat_sse))
        .route("/train", post(train_exobit))
        .route("/perplexity", post(perplexity))
        .route("/reset", post(reset))
        .route("/status", get(status_exobit))
        .route("/config", put(config_put))
        .route("/export", post(export_adapters))
        // Common
        .route("/health", get(health))
        .with_state(state);

    let addr = format!("{}:{}", config.host, config.port);
    eprintln!("MLX inference server listening on http://{addr}");
    eprintln!("  model: {model_name}");
    eprintln!("  endpoints:");
    eprintln!("    OpenAI:  POST /v1/chat/completions");
    eprintln!("    Ex0bit:  POST /chat (SSE), /train, /perplexity, /export, /reset | GET /status | PUT /config");

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_template() {
        let messages = vec![
            ChatMessage { role: "system".into(), content: "You are helpful.".into() },
            ChatMessage { role: "user".into(), content: "Hi".into() },
        ];
        let prompt = apply_chat_template(&messages);
        assert!(prompt.contains("<|im_start|>system\nYou are helpful.<|im_end|>"));
        assert!(prompt.contains("<|im_start|>user\nHi<|im_end|>"));
        // Ends with assistant prefill including empty think block (think-skip)
        assert!(prompt.contains("<|im_start|>assistant\n"));
        assert!(prompt.ends_with("<think>\n\n</think>\n\n"));
    }

    #[test]
    fn test_chat_template_single_user() {
        let messages = vec![
            ChatMessage { role: "user".into(), content: "What is 2+2?".into() },
        ];
        let prompt = apply_chat_template(&messages);
        assert_eq!(
            prompt,
            "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }
}
