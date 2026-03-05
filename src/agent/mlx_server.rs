//! OpenAI-compatible + Ex0bit daemon HTTP server for MLX inference + LoRA training.
//!
//! Exposes both protocols:
//!   - OpenAI: `/v1/chat/completions` (JSON)
//!   - Ex0bit: `/chat` (SSE), `/train`, `/reset`, `/status`, `/config`
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

fn apply_chat_template(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        prompt.push_str(IM_START);
        prompt.push_str(&msg.role);
        prompt.push('\n');
        prompt.push_str(&msg.content);
        prompt.push_str(IM_END);
        prompt.push('\n');
    }
    prompt.push_str(IM_START);
    prompt.push_str("assistant\n");
    // Prefill empty think block so Qwen3.5 skips reasoning and answers directly.
    // Qwen3.5 lacks /nothink support — this is the standard prefill approach.
    prompt.push_str("<think>\n\n</think>\n\n");
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

enum ModelRequest {
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
}

/// Shared training state visible to status endpoint.
struct TrainState {
    training: AtomicBool,
    total_steps: AtomicU32,
    last_loss: Mutex<f32>,
    initial_loss: Mutex<f32>,
    trainable_params: AtomicU32,
}

impl TrainState {
    fn new() -> Self {
        TrainState {
            training: AtomicBool::new(false),
            total_steps: AtomicU32::new(0),
            last_loss: Mutex::new(0.0),
            initial_loss: Mutex::new(0.0),
            trainable_params: AtomicU32::new(0),
        }
    }
}

fn run_model_worker(
    model_dir: PathBuf,
    cfg: ModelConfig,
    lora_cfg: LoraConfig,
    train_state: Arc<TrainState>,
    rx: std::sync::mpsc::Receiver<ModelRequest>,
) {
    use mlx_rs::module::ModuleParameters;

    let tokenizer = MlxTokenizer::load(&model_dir)
        .expect("Failed to load tokenizer");
    let mut model = MlxLoraModel::load(&model_dir, &cfg, &lora_cfg)
        .expect("Failed to load model");

    // Count trainable params
    let trainable = model.trainable_parameters().flatten();
    let total_trainable: u32 = trainable.values()
        .map(|a| a.shape().iter().product::<i32>() as u32)
        .sum();
    train_state.trainable_params.store(total_trainable, Ordering::Relaxed);

    let im_end_id = 248046i32;
    let eos_id = 248044i32;
    let think_id = 248068i32; // <think> — prevent re-entering thinking after prefill
    let stop_tokens = [im_end_id, eos_id, think_id];

    eprintln!("model worker ready ({total_trainable} trainable params)");

    while let Ok(req) = rx.recv() {
        match req {
            ModelRequest::Chat { prompt, max_tokens, temperature, reply } => {
                let result = (|| {
                    let prompt_tokens = tokenizer.encode(&prompt)
                        .map_err(|e| format!("encode: {e}"))?;
                    let prompt_len = prompt_tokens.len();
                    let generated = model.generate(
                        &prompt_tokens, max_tokens, temperature, &stop_tokens,
                    ).map_err(|e| format!("generate: {e}"))?;
                    let gen_len = generated.len();
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
                    use mlx_rs::Array;
                    for (i, (toks, tgts)) in samples.iter().enumerate() {
                        eprintln!("  sample {i}: tokens={}, targets={}, first_5_toks={:?}",
                            toks.len(), tgts.len(), &toks[..toks.len().min(5)]);
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
                        &mut model, &token_batches, &target_batches,
                        &lora_cfg, epochs, early_stop_loss, patience,
                        Some(callback),
                    ).map_err(|e| format!("train: {e}"))?;

                    let steps = losses.len();
                    Ok((losses, steps))
                })();

                train_state.training.store(false, Ordering::Relaxed);
                if let Some(reply) = reply {
                    let _ = reply.send(result);
                }
            }
            ModelRequest::Reset { reply } => {
                train_state.total_steps.store(0, Ordering::Relaxed);
                *train_state.last_loss.lock() = 0.0;
                *train_state.initial_loss.lock() = 0.0;

                let result = (|| {
                    model = MlxLoraModel::load(&model_dir, &cfg, &lora_cfg)
                        .map_err(|e| format!("reload: {e}"))?;
                    // Recount trainable params
                    let trainable = model.trainable_parameters().flatten();
                    let total: u32 = trainable.values()
                        .map(|a| a.shape().iter().product::<i32>() as u32)
                        .sum();
                    train_state.trainable_params.store(total, Ordering::Relaxed);
                    Ok(())
                })();
                let _ = reply.send(result);
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

fn tokenize_conversation(
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
    let lora_cfg_worker = LoraConfig {
        rank: config.lora_config.rank,
        alpha: config.lora_config.alpha,
        lr: config.lora_config.lr,
        weight_decay: config.lora_config.weight_decay,
        grad_clip: config.lora_config.grad_clip,
    };
    let ts_clone = Arc::clone(&train_state);

    std::thread::Builder::new()
        .name("mlx-model-worker".into())
        .spawn(move || {
            run_model_worker(model_dir, cfg, lora_cfg_worker, ts_clone, rx);
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
        .route("/reset", post(reset))
        .route("/status", get(status_exobit))
        .route("/config", put(config_put))
        // Common
        .route("/health", get(health))
        .with_state(state);

    let addr = format!("{}:{}", config.host, config.port);
    eprintln!("MLX inference server listening on http://{addr}");
    eprintln!("  model: {model_name}");
    eprintln!("  endpoints:");
    eprintln!("    OpenAI:  POST /v1/chat/completions");
    eprintln!("    Ex0bit:  POST /chat (SSE), /train, /reset | GET /status | PUT /config");

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
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn test_chat_template_single_user() {
        let messages = vec![
            ChatMessage { role: "user".into(), content: "What is 2+2?".into() },
        ];
        let prompt = apply_chat_template(&messages);
        assert_eq!(
            prompt,
            "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n"
        );
    }
}
