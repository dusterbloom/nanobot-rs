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

    /// In-process MLX provider. Owns the channel to the model worker thread.
    pub struct MlxProvider {
        tx: std::sync::mpsc::SyncSender<ModelRequest>,
        tokenizer: Arc<MlxTokenizer>,
        model_name: String,
        api_base: String,
    }

    impl MlxProvider {
        /// Start the model worker thread and return the provider.
        ///
        /// The model is loaded once (~2GB for 8-bit Qwen3.5-2B) and serves
        /// all inference, perplexity, and training requests.
        pub fn start(
            model_dir: PathBuf,
            model_config: ModelConfig,
            lora_config: LoraConfig,
        ) -> Result<Self> {
            let tokenizer = MlxTokenizer::load(&model_dir)
                .context("Failed to load tokenizer")?;
            let model_name = model_dir
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| "mlx-model".into());

            let train_state = Arc::new(crate::agent::mlx_server::TrainState::new());
            let (tx, rx) = std::sync::mpsc::sync_channel::<ModelRequest>(4);

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

            Ok(Self {
                tx,
                tokenizer: Arc::new(tokenizer),
                model_name,
                api_base: "mlx://in-process".to_string(),
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
            // Convert JSON messages to ChatMessage format and apply template.
            let chat_messages: Vec<crate::agent::mlx_server::ChatMessage> = messages
                .iter()
                .filter_map(|m| {
                    let role = m.get("role")?.as_str()?.to_string();
                    let content = m.get("content")?.as_str()?.to_string();
                    Some(crate::agent::mlx_server::ChatMessage { role, content })
                })
                .collect();

            let prompt = crate::agent::mlx_server::apply_chat_template(&chat_messages);

            let (reply_tx, reply_rx) = oneshot::channel();
            self.tx
                .send(ModelRequest::Chat {
                    prompt,
                    max_tokens: max_tokens as usize,
                    temperature: temperature as f32,
                    reply: reply_tx,
                })
                .map_err(|_| anyhow::anyhow!("model worker died"))?;

            let (text, prompt_tokens, completion_tokens) = reply_rx
                .await
                .map_err(|_| anyhow::anyhow!("model worker dropped reply"))?
                .map_err(|e| anyhow::anyhow!("{}", e))?;

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
}
