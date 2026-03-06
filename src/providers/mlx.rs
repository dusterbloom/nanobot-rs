//! In-process MLX LLM provider — closes the online learning loop.
//!
//! Implements [`LLMProvider`] by sending requests through a channel to the
//! MLX model worker thread. Same model serves inference, perplexity scoring,
//! and LoRA training. No HTTP, no separate process.

#[cfg(feature = "mlx")]
mod inner {
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::sync::Arc;

    use anyhow::{Context, Result};
    use async_trait::async_trait;
    use tokio::sync::oneshot;

    use crate::agent::mlx_lora::{LoraConfig, MlxLoraModel, MlxTokenizer, ModelConfig};
    use crate::agent::mlx_server::ModelRequest;
    use crate::providers::base::{LLMProvider, LLMResponse};

    /// Strip thinking content from model output. Handles two cases:
    /// 1. `<think>...</think>` blocks (full tags in output)
    /// 2. `...</think>` at start (opening tag was in prompt prefill)
    fn strip_think_blocks(text: &str) -> String {
        let mut result = String::new();
        let mut rest = text;

        // Case 2: output starts mid-think (opening <think> was in prompt prefill)
        if !rest.contains("<think>") {
            if let Some(end) = rest.find("</think>") {
                rest = &rest[end + "</think>".len()..];
            }
        }

        // Case 1: full <think>...</think> blocks
        while let Some(start) = rest.find("<think>") {
            result.push_str(&rest[..start]);
            if let Some(end) = rest[start..].find("</think>") {
                rest = &rest[start + end + "</think>".len()..];
            } else {
                // Unclosed think block — discard the rest
                return result.trim().to_string();
            }
        }
        result.push_str(rest);
        result.trim().to_string()
    }

    /// In-process MLX provider. Owns the channel to the model worker thread.
    ///
    /// When `mlx_lm_url` is set, inference is delegated to an external mlx-lm
    /// server via OpenAI-compat HTTP. Training and perplexity remain in-process.
    pub struct MlxProvider {
        tx: std::sync::mpsc::SyncSender<ModelRequest>,
        tokenizer: Arc<MlxTokenizer>,
        model_name: String,
        api_base: String,
        thinking_model: bool,
        /// When set, chat() delegates to this mlx-lm server URL.
        mlx_lm_url: Option<String>,
        http_client: reqwest::Client,
        /// Managed mlx-lm subprocess (when mlxLmUrl is "auto").
        managed_server: Option<Arc<parking_lot::Mutex<crate::agent::mlx_lm::MlxLmServer>>>,
    }

    impl MlxProvider {
        /// Start the model worker thread and return the provider.
        ///
        /// The model is loaded once (~2GB for 8-bit Qwen3.5-2B) and serves
        /// all inference, perplexity, and training requests.
        ///
        /// When `mlx_lm_url` is set, inference is delegated to the external
        /// mlx-lm server while training/perplexity stay in-process.
        pub fn start(
            model_dir: PathBuf,
            model_config: ModelConfig,
            lora_config: LoraConfig,
        ) -> Result<Self> {
            Self::start_with_mlx_lm(model_dir, model_config, lora_config, None)
        }

        /// Start with optional mlx-lm server URL for inference delegation.
        ///
        /// When `mlx_lm_url` is `Some("auto")`, spawns a managed `mlx_lm.server`
        /// subprocess on port 8090. Any other `Some(url)` connects to that URL.
        pub fn start_with_mlx_lm(
            model_dir: PathBuf,
            model_config: ModelConfig,
            lora_config: LoraConfig,
            mlx_lm_url: Option<String>,
        ) -> Result<Self> {
            let tokenizer = MlxTokenizer::load(&model_dir)
                .context("Failed to load tokenizer")?;
            let model_name = model_dir
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| "mlx-model".into());

            let train_state = Arc::new(crate::agent::mlx_server::TrainState::new());
            let (tx, rx) = std::sync::mpsc::sync_channel::<ModelRequest>(4);

            let thinking_model = model_config.thinking_model;
            let dir = model_dir.clone();
            let cfg = model_config;
            let lora_cfg = lora_config;
            let ts = train_state;
            std::thread::Builder::new()
                .name("mlx-model-worker".into())
                .spawn(move || {
                    crate::agent::mlx_server::run_model_worker(dir, cfg, lora_cfg, ts, rx);
                })
                .context("Failed to spawn model worker thread")?;

            // Auto-start managed mlx-lm server when "auto"
            let (resolved_url, managed_server) = match mlx_lm_url.as_deref() {
                Some("auto") => {
                    let port = 8090u16;
                    let adapter_path = model_dir.join("adapters");
                    let adapter = if adapter_path.join("adapters.safetensors").exists() {
                        Some(adapter_path)
                    } else {
                        None
                    };
                    eprintln!("starting managed mlx-lm server on port {port}...");
                    match crate::agent::mlx_lm::MlxLmServer::start(model_dir, adapter, port) {
                        Ok(srv) => {
                            let url = srv.base_url();
                            eprintln!("mlx-lm server ready at {url}");
                            (Some(url), Some(std::sync::Arc::new(parking_lot::Mutex::new(srv))))
                        }
                        Err(e) => {
                            eprintln!("warning: mlx-lm server failed to start: {e}");
                            eprintln!("falling back to in-process inference");
                            (None, None)
                        }
                    }
                }
                Some(url) => {
                    eprintln!("MLX inference delegated to mlx-lm server at {url}");
                    (Some(url.to_string()), None)
                }
                None => (None, None),
            };

            let api_base = resolved_url.as_deref()
                .unwrap_or("mlx://in-process")
                .to_string();

            Ok(Self {
                tx,
                tokenizer: Arc::new(tokenizer),
                model_name,
                api_base,
                thinking_model,
                mlx_lm_url: resolved_url,
                http_client: reqwest::Client::new(),
                managed_server,
            })
        }

        /// Compute perplexity (CE loss) for a user/assistant exchange.
        /// Returns the mean cross-entropy loss.
        pub async fn perplexity(&self, user: &str, assistant: &str) -> Result<f32> {
            let messages = vec![
                crate::agent::mlx_server::ChatMessage {
                    role: "user".into(),
                    content: user.into(),
                },
                crate::agent::mlx_server::ChatMessage {
                    role: "assistant".into(),
                    content: assistant.into(),
                },
            ];
            let (tokens, targets) =
                crate::agent::mlx_server::tokenize_conversation(&self.tokenizer, &messages)
                    .map_err(|e| anyhow::anyhow!("{}", e))?;

            let (reply_tx, reply_rx) = oneshot::channel();
            self.tx
                .send(ModelRequest::Perplexity {
                    tokens,
                    targets,
                    reply: reply_tx,
                })
                .map_err(|_| anyhow::anyhow!("model worker died"))?;

            reply_rx
                .await
                .map_err(|_| anyhow::anyhow!("model worker dropped reply"))?
                .map_err(|e| anyhow::anyhow!("{}", e))
        }

        /// Get a clone of the model worker channel sender.
        /// Used by the ANE training thread to send ApplyLoraDeltas directly.
        pub fn model_tx(&self) -> std::sync::mpsc::SyncSender<ModelRequest> {
            self.tx.clone()
        }

        /// Switch the managed mlx-lm server to a different model.
        /// Returns the new model name, or error if no managed server.
        pub fn switch_model(&self, model_dir: PathBuf) -> Result<String> {
            let srv = self.managed_server.as_ref()
                .ok_or_else(|| anyhow::anyhow!("no managed mlx-lm server"))?;
            let adapter_path = model_dir.join("adapters");
            let adapter = if adapter_path.join("adapters.safetensors").exists() {
                Some(adapter_path)
            } else {
                None
            };
            let name = model_dir.file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();
            srv.lock().switch_model(model_dir, adapter)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            Ok(name)
        }

        /// Get reference to the managed server (for REPL model switching).
        pub fn managed_server(&self) -> Option<&Arc<parking_lot::Mutex<crate::agent::mlx_lm::MlxLmServer>>> {
            self.managed_server.as_ref()
        }

        /// Delegate inference to an external mlx-lm server via OpenAI-compat API.
        async fn chat_via_mlx_lm(
            &self,
            base_url: &str,
            messages: &[serde_json::Value],
            max_tokens: u32,
            temperature: f64,
        ) -> Result<LLMResponse> {
            let url = format!("{}/v1/chat/completions", base_url.trim_end_matches('/'));
            let body = serde_json::json!({
                "model": &self.model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            });

            let resp = self.http_client
                .post(&url)
                .json(&body)
                .send()
                .await
                .context("mlx-lm server request failed")?;

            if !resp.status().is_success() {
                let status = resp.status();
                let text = resp.text().await.unwrap_or_default();
                anyhow::bail!("mlx-lm server returned {status}: {text}");
            }

            let json: serde_json::Value = resp.json().await
                .context("mlx-lm server returned invalid JSON")?;

            let content = json.pointer("/choices/0/message/content")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            let text = if self.thinking_model {
                strip_think_blocks(&content)
            } else {
                content
            };

            let finish_reason = json.pointer("/choices/0/finish_reason")
                .and_then(|v| v.as_str())
                .unwrap_or("stop")
                .to_string();

            let mut usage = HashMap::new();
            if let Some(u) = json.get("usage") {
                if let Some(n) = u.get("prompt_tokens").and_then(|v| v.as_i64()) {
                    usage.insert("prompt_tokens".to_string(), n);
                }
                if let Some(n) = u.get("completion_tokens").and_then(|v| v.as_i64()) {
                    usage.insert("completion_tokens".to_string(), n);
                }
                if let Some(n) = u.get("total_tokens").and_then(|v| v.as_i64()) {
                    usage.insert("total_tokens".to_string(), n);
                }
            }

            Ok(LLMResponse {
                content: Some(text),
                tool_calls: Vec::new(),
                finish_reason,
                usage,
            })
        }

        /// Export LoRA adapters to disk in mlx-lm safetensors format.
        /// Returns the number of adapter tensors exported.
        pub async fn export_adapters(&self, output_dir: std::path::PathBuf) -> Result<usize> {
            let (reply_tx, reply_rx) = oneshot::channel();
            self.tx
                .send(ModelRequest::ExportAdapters {
                    output_dir,
                    reply: reply_tx,
                })
                .map_err(|_| anyhow::anyhow!("model worker died"))?;

            reply_rx
                .await
                .map_err(|_| anyhow::anyhow!("model worker dropped reply"))?
                .map_err(|e| anyhow::anyhow!("{}", e))
        }

        /// Trigger LoRA training on the model worker.
        /// Training updates the live model weights — next inference uses them.
        pub async fn train(
            &self,
            conversations: Vec<Vec<crate::agent::mlx_server::ChatMessage>>,
            epochs: usize,
        ) -> Result<()> {
            let mut samples = Vec::new();
            for conv in &conversations {
                if let Ok(pair) =
                    crate::agent::mlx_server::tokenize_conversation(&self.tokenizer, conv)
                {
                    samples.push(pair);
                }
            }
            if samples.is_empty() {
                anyhow::bail!("no valid training samples");
            }

            let injected = samples.len();
            let max_steps = injected * epochs;
            let patience = injected * 2;

            self.tx
                .send(ModelRequest::Train {
                    samples,
                    epochs: max_steps,
                    early_stop_loss: 0.0,
                    patience,
                    reply: None, // fire-and-forget
                })
                .map_err(|_| anyhow::anyhow!("model worker died"))?;

            Ok(())
        }
    }

    #[async_trait]
    impl LLMProvider for MlxProvider {
        async fn chat(
            &self,
            messages: &[serde_json::Value],
            _tools: Option<&[serde_json::Value]>,
            _model: Option<&str>,
            max_tokens: u32,
            temperature: f64,
            _thinking_budget: Option<u32>,
            _top_p: Option<f64>,
        ) -> Result<LLMResponse> {
            // Delegate to mlx-lm server when configured
            if let Some(ref url) = self.mlx_lm_url {
                return self.chat_via_mlx_lm(url, messages, max_tokens, temperature).await;
            }

            // In-process: apply chat template and generate
            let chat_messages: Vec<crate::agent::mlx_server::ChatMessage> = messages
                .iter()
                .filter_map(|m| {
                    let role = m.get("role")?.as_str()?.to_string();
                    let content = m.get("content")?.as_str()?.to_string();
                    Some(crate::agent::mlx_server::ChatMessage { role, content })
                })
                .collect();

            let prompt = if self.thinking_model {
                crate::agent::mlx_server::apply_chat_template_nothink(&chat_messages)
            } else {
                crate::agent::mlx_server::apply_chat_template(&chat_messages)
            };

            let (reply_tx, reply_rx) = oneshot::channel();
            self.tx
                .send(ModelRequest::Chat {
                    prompt,
                    max_tokens: max_tokens as usize,
                    temperature: temperature as f32,
                    reply: reply_tx,
                })
                .map_err(|_| anyhow::anyhow!("model worker died"))?;

            let (raw_text, prompt_tokens, completion_tokens) = reply_rx
                .await
                .map_err(|_| anyhow::anyhow!("model worker dropped reply"))?
                .map_err(|e| anyhow::anyhow!("{}", e))?;

            // Strip <think>...</think> blocks from thinking models
            let text = if self.thinking_model {
                strip_think_blocks(&raw_text)
            } else {
                raw_text
            };

            let mut usage = HashMap::new();
            usage.insert("prompt_tokens".to_string(), prompt_tokens as i64);
            usage.insert("completion_tokens".to_string(), completion_tokens as i64);
            usage.insert(
                "total_tokens".to_string(),
                (prompt_tokens + completion_tokens) as i64,
            );

            Ok(LLMResponse {
                content: Some(text),
                tool_calls: Vec::new(),
                finish_reason: "stop".to_string(),
                usage,
            })
        }

        fn get_default_model(&self) -> &str {
            &self.model_name
        }

        fn get_api_base(&self) -> Option<&str> {
            Some(&self.api_base)
        }
    }
}

#[cfg(feature = "mlx")]
pub use inner::MlxProvider;

#[cfg(all(test, feature = "mlx"))]
mod tests {
    use super::*;
    use crate::agent::mlx_lora::{LoraConfig, ModelConfig};
    use crate::providers::base::LLMProvider;
    use std::path::PathBuf;

    fn model_dir() -> PathBuf {
        dirs::home_dir()
            .unwrap()
            .join(".cache/lm-studio/models/mlx-community/Qwen3.5-2B-MLX-8bit")
    }

    fn skip_if_no_model() -> bool {
        !model_dir().join("tokenizer.json").exists()
    }

    /// E2E: start MlxProvider, run chat inference, verify we get a non-empty response.
    #[tokio::test]
    async fn test_mlx_provider_chat_inference() {
        if skip_if_no_model() {
            eprintln!("SKIP: model not found at {:?}", model_dir());
            return;
        }

        let provider = MlxProvider::start(
            model_dir(),
            ModelConfig::qwen3_5_2b(),
            LoraConfig { lr: 1e-5, ..LoraConfig::default() },
        ).expect("MlxProvider::start failed");

        let messages = vec![serde_json::json!({
            "role": "user",
            "content": "What is 2+2? Answer with just the number."
        })];

        let resp = provider.chat(&messages, None, None, 32, 0.0, None, None)
            .await
            .expect("chat failed");

        assert!(resp.content.is_some(), "response should have content");
        let text = resp.content.unwrap();
        assert!(!text.is_empty(), "response text should not be empty");
        eprintln!("MLX inference response: {text:?}");
        assert_eq!(resp.finish_reason, "stop");
        assert!(*resp.usage.get("prompt_tokens").unwrap() > 0);
        assert!(*resp.usage.get("completion_tokens").unwrap() > 0);
    }

    /// E2E: compute perplexity on a known exchange.
    #[tokio::test]
    async fn test_mlx_provider_perplexity() {
        if skip_if_no_model() {
            eprintln!("SKIP: model not found");
            return;
        }

        let provider = MlxProvider::start(
            model_dir(),
            ModelConfig::qwen3_5_2b(),
            LoraConfig { lr: 1e-5, ..LoraConfig::default() },
        ).expect("MlxProvider::start failed");

        let loss = provider.perplexity(
            "What is the capital of France?",
            "The capital of France is Paris.",
        ).await.expect("perplexity failed");

        eprintln!("Perplexity (CE loss): {loss}");
        assert!(loss > 0.0, "CE loss should be positive");
        assert!(loss < 20.0, "CE loss should be reasonable (< 20)");
    }

    /// E2E: fire-and-forget training on a small sample.
    #[tokio::test]
    async fn test_mlx_provider_train() {
        if skip_if_no_model() {
            eprintln!("SKIP: model not found");
            return;
        }

        let provider = MlxProvider::start(
            model_dir(),
            ModelConfig::qwen3_5_2b(),
            LoraConfig { lr: 1e-5, ..LoraConfig::default() },
        ).expect("MlxProvider::start failed");

        let conversations = vec![vec![
            crate::agent::mlx_server::ChatMessage {
                role: "user".into(),
                content: "What is the speed of light?".into(),
            },
            crate::agent::mlx_server::ChatMessage {
                role: "assistant".into(),
                content: "Approximately 299,792,458 meters per second.".into(),
            },
        ]];

        // Training is fire-and-forget but should not error on send.
        provider.train(conversations, 2)
            .await
            .expect("train should not error");
    }

    /// E2E: full closed loop — inference, perplexity, train on same provider.
    #[tokio::test]
    async fn test_mlx_provider_closed_loop() {
        if skip_if_no_model() {
            eprintln!("SKIP: model not found");
            return;
        }

        let provider = MlxProvider::start(
            model_dir(),
            ModelConfig::qwen3_5_2b(),
            LoraConfig { lr: 1e-5, ..LoraConfig::default() },
        ).expect("MlxProvider::start failed");

        // 1. Inference
        let messages = vec![serde_json::json!({
            "role": "user",
            "content": "Capital of Japan?"
        })];
        let resp = provider.chat(&messages, None, None, 16, 0.0, None, None)
            .await
            .expect("chat failed");
        let answer = resp.content.unwrap_or_default();
        eprintln!("Inference: {answer:?}");
        assert!(!answer.is_empty());

        // 2. Perplexity scoring
        let loss = provider.perplexity("Capital of Japan?", &answer)
            .await
            .expect("perplexity failed");
        eprintln!("Perplexity: {loss}");
        assert!(loss > 0.0);

        // 3. Training (fire-and-forget)
        let convos = vec![vec![
            crate::agent::mlx_server::ChatMessage {
                role: "user".into(),
                content: "Capital of Japan?".into(),
            },
            crate::agent::mlx_server::ChatMessage {
                role: "assistant".into(),
                content: answer.clone(),
            },
        ]];
        provider.train(convos, 1).await.expect("train should not error");

        eprintln!("Closed loop complete: inference → perplexity → train");
    }

    // --- Qwen3-4B tests ---

    fn model_dir_4b() -> PathBuf {
        dirs::home_dir()
            .unwrap()
            .join(".cache/lm-studio/models/lmstudio-community/Qwen3-4B-Thinking-2507-MLX-4bit")
    }

    fn skip_if_no_4b() -> bool {
        !model_dir_4b().join("tokenizer.json").exists()
    }

    /// E2E: Qwen3-4B closed loop — inference, perplexity, train.
    #[tokio::test]
    async fn test_mlx_qwen3_4b_closed_loop() {
        if skip_if_no_4b() {
            eprintln!("SKIP: Qwen3-4B not found at {:?}", model_dir_4b());
            return;
        }

        let provider = MlxProvider::start(
            model_dir_4b(),
            ModelConfig::qwen3_4b(),
            LoraConfig { lr: 1e-5, ..LoraConfig::default() },
        ).expect("MlxProvider::start failed (4B)");

        // 1. Inference
        let messages = vec![serde_json::json!({
            "role": "user",
            "content": "What is 2+2? Answer with just the number."
        })];
        // Thinking models need more tokens: ~100 for reasoning + answer
        let resp = provider.chat(&messages, None, None, 256, 0.0, None, None)
            .await
            .expect("chat failed");
        let answer = resp.content.unwrap_or_default();
        eprintln!("4B Inference: {answer:?}");
        assert!(!answer.is_empty(), "response empty after think-strip (raw may have had thinking)");

        // 2. Perplexity
        let loss = provider.perplexity("What is 2+2?", &answer)
            .await
            .expect("perplexity failed");
        eprintln!("4B Perplexity: {loss}");
        assert!(loss > 0.0);

        // 3. Train
        let convos = vec![vec![
            crate::agent::mlx_server::ChatMessage {
                role: "user".into(),
                content: "What is 2+2?".into(),
            },
            crate::agent::mlx_server::ChatMessage {
                role: "assistant".into(),
                content: answer.clone(),
            },
        ]];
        provider.train(convos, 1).await.expect("train should not error");

        eprintln!("4B closed loop complete: inference → perplexity → train");
    }
}
