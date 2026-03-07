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
    use crate::providers::base::{LLMProvider, LLMResponse, ToolCallRequest};

    /// Parse tool calls from model text output when the model doesn't generate
    /// proper tool_calls JSON. Detects `{"name": "...", "arguments": {...}}` blocks.
    /// Returns (parsed_tool_calls, remaining_text_with_blocks_removed).
    fn parse_tool_calls_from_text(text: &str) -> (Vec<ToolCallRequest>, String) {
        let mut tool_calls = Vec::new();
        let mut remaining = text.to_string();

        // Try to find JSON objects with "name" and "arguments" keys
        let mut search_from = 0;
        while let Some(start) = remaining[search_from..].find('{') {
            let start = search_from + start;
            // Try to find matching closing brace by counting nesting
            let mut depth = 0i32;
            let mut end = None;
            for (i, ch) in remaining[start..].char_indices() {
                match ch {
                    '{' => depth += 1,
                    '}' => {
                        depth -= 1;
                        if depth == 0 {
                            end = Some(start + i + 1);
                            break;
                        }
                    }
                    _ => {}
                }
            }
            let Some(end) = end else {
                search_from = start + 1;
                continue;
            };

            let candidate = &remaining[start..end];
            if let Ok(obj) = serde_json::from_str::<serde_json::Value>(candidate) {
                if let (Some(name), Some(args)) = (
                    obj.get("name").and_then(|v| v.as_str()),
                    obj.get("arguments"),
                ) {
                    let arguments: HashMap<String, serde_json::Value> = match args {
                        serde_json::Value::Object(m) => {
                            m.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
                        }
                        serde_json::Value::String(s) => serde_json::from_str(s).unwrap_or_default(),
                        _ => HashMap::new(),
                    };
                    let id = format!("call_{}", tool_calls.len());
                    tool_calls.push(ToolCallRequest {
                        id,
                        name: name.to_string(),
                        arguments,
                    });
                    // Remove the JSON block from remaining text
                    remaining = format!("{}{}", &remaining[..start], &remaining[end..]);
                    continue; // don't advance search_from since we removed text
                }
            }
            search_from = start + 1;
        }

        (tool_calls, remaining)
    }

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
        /// Full model directory path — used as the "model" field in mlx-lm requests.
        model_path: String,
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
            let tokenizer = MlxTokenizer::load(&model_dir).context("Failed to load tokenizer")?;
            let model_name = model_dir
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| "mlx-model".into());
            // Full path needed for mlx-lm server requests — it matches by path, not leaf name.
            let model_path_str = model_dir.to_string_lossy().to_string();

            let train_state = Arc::new(crate::agent::mlx_server::TrainState::new());
            let (tx, rx) = std::sync::mpsc::sync_channel::<ModelRequest>(4);

            let thinking_model = model_config.thinking_model;

            // Start managed mlx-lm server BEFORE model worker so we can pass
            // a post-train hook that reloads adapters on the managed server.
            let (resolved_url, managed_server) = match mlx_lm_url.as_deref() {
                Some("auto") => {
                    let port = 8090u16;
                    let adapter_path = model_dir.join("adapters");
                    let adapter = if adapter_path.join("adapters.safetensors").exists() {
                        Some(adapter_path)
                    } else {
                        None
                    };
                    let server_options = crate::agent::mlx_lm::MlxLmServerOptions {
                        decode_concurrency: std::env::var("NANOBOT_MLX_LM_DECODE_CONCURRENCY")
                            .ok()
                            .and_then(|v| v.parse().ok()),
                        prompt_concurrency: std::env::var("NANOBOT_MLX_LM_PROMPT_CONCURRENCY")
                            .ok()
                            .and_then(|v| v.parse().ok()),
                        chat_template_args: thinking_model
                            .then(|| r#"{"enable_thinking": false}"#.to_string()),
                    };
                    tracing::info!(port, "starting managed mlx-lm server");
                    match crate::agent::mlx_lm::MlxLmServer::start(
                        model_dir.clone(),
                        adapter,
                        port,
                        server_options,
                    ) {
                        Ok(srv) => {
                            let url = srv.base_url();
                            tracing::info!(url = %url, "mlx-lm server ready");
                            (
                                Some(url),
                                Some(std::sync::Arc::new(parking_lot::Mutex::new(srv))),
                            )
                        }
                        Err(e) => {
                            tracing::warn!(
                                "mlx-lm server failed to start: {e}, falling back to in-process"
                            );
                            (None, None)
                        }
                    }
                }
                Some(url) => {
                    tracing::info!(url, "MLX inference delegated to mlx-lm server");
                    (Some(url.to_string()), None)
                }
                None => (None, None),
            };

            // Build post-train hook that reloads adapters on the managed server.
            let post_train_hook: Option<Box<dyn Fn() + Send>> =
                managed_server.as_ref().map(|srv| {
                    let srv = Arc::clone(srv);
                    let adapter_dir = model_dir.join("adapters");
                    Box::new(move || {
                        if adapter_dir.join("adapters.safetensors").exists() {
                            match srv.lock().reload_adapters(adapter_dir.clone()) {
                                Ok(()) => {
                                    tracing::info!("mlx-lm: adapters reloaded after training")
                                }
                                Err(e) => tracing::warn!("mlx-lm: adapter reload failed: {e}"),
                            }
                        }
                    }) as Box<dyn Fn() + Send>
                });

            let dir = model_dir.clone();
            let cfg = model_config;
            let lora_cfg = lora_config;
            let ts = train_state;
            std::thread::Builder::new()
                .name("mlx-model-worker".into())
                .spawn(move || {
                    crate::agent::mlx_server::run_model_worker(
                        dir,
                        cfg,
                        lora_cfg,
                        ts,
                        rx,
                        post_train_hook,
                    );
                })
                .context("Failed to spawn model worker thread")?;

            let api_base = resolved_url
                .as_deref()
                .unwrap_or("mlx://in-process")
                .to_string();

            Ok(Self {
                tx,
                tokenizer: Arc::new(tokenizer),
                model_name,
                model_path: model_path_str,
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
            let srv = self
                .managed_server
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("no managed mlx-lm server"))?;
            let adapter_path = model_dir.join("adapters");
            let adapter = if adapter_path.join("adapters.safetensors").exists() {
                Some(adapter_path)
            } else {
                None
            };
            let name = model_dir
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();
            srv.lock()
                .switch_model(model_dir, adapter)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            Ok(name)
        }

        /// Get reference to the managed server (for REPL model switching).
        pub fn managed_server(
            &self,
        ) -> Option<&Arc<parking_lot::Mutex<crate::agent::mlx_lm::MlxLmServer>>> {
            self.managed_server.as_ref()
        }

        /// Kill the managed mlx-lm server subprocess if one is running.
        /// Call before starting a new provider to free the port.
        pub fn kill_managed_server(&self) {
            if let Some(ref srv) = self.managed_server {
                srv.lock().kill();
            }
        }

        /// Inject `/nothink` into system message for thinking models.
        /// This tells Qwen3 to skip the `<think>` block, dramatically reducing
        /// output tokens (200+ thinking tokens → 0).
        fn inject_nothink(&self, messages: &[serde_json::Value]) -> Vec<serde_json::Value> {
            if !self.thinking_model {
                return messages.to_vec();
            }
            let mut msgs = messages.to_vec();
            // Find system message and append /nothink
            if let Some(sys) = msgs
                .iter_mut()
                .find(|m| m.get("role").and_then(|r| r.as_str()) == Some("system"))
            {
                if let Some(content) = sys.get("content").and_then(|c| c.as_str()) {
                    if !content.contains("/nothink") {
                        sys["content"] = serde_json::json!(format!("{content}\n/nothink"));
                    }
                }
            } else {
                // No system message — prepend one
                msgs.insert(
                    0,
                    serde_json::json!({
                        "role": "system",
                        "content": "/nothink"
                    }),
                );
            }
            msgs
        }

        /// Delegate inference to an external mlx-lm server via OpenAI-compat API.
        async fn chat_via_mlx_lm(
            &self,
            base_url: &str,
            messages: &[serde_json::Value],
            tools: Option<&[serde_json::Value]>,
            max_tokens: u32,
            temperature: f64,
        ) -> Result<LLMResponse> {
            let url = format!("{}/v1/chat/completions", base_url.trim_end_matches('/'));
            let msgs = self.inject_nothink(messages);
            let mut body = serde_json::json!({
                "model": &self.model_path,
                "messages": msgs,
                "max_tokens": max_tokens,
                "temperature": temperature,
            });
            if let Some(tools) = tools {
                if !tools.is_empty() {
                    body["tools"] = serde_json::json!(tools);
                }
            }

            let resp = self
                .http_client
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

            let json: serde_json::Value = resp
                .json()
                .await
                .context("mlx-lm server returned invalid JSON")?;

            let message = json
                .pointer("/choices/0/message")
                .cloned()
                .unwrap_or_default();

            let content = message
                .get("content")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            let text = strip_think_blocks(&content);

            let finish_reason = json
                .pointer("/choices/0/finish_reason")
                .and_then(|v| v.as_str())
                .unwrap_or("stop")
                .to_string();

            // Parse tool_calls from response (same format as OpenAI)
            let mut tool_calls = Vec::new();
            if let Some(tc_array) = message.get("tool_calls").and_then(|v| v.as_array()) {
                for tc in tc_array {
                    let id = tc
                        .get("id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let function = tc.get("function").cloned().unwrap_or_default();
                    let name = function
                        .get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let arguments_raw = function
                        .get("arguments")
                        .cloned()
                        .unwrap_or(serde_json::Value::String("{}".into()));
                    let arguments: HashMap<String, serde_json::Value> =
                        if let Some(s) = arguments_raw.as_str() {
                            serde_json::from_str(s).unwrap_or_default()
                        } else if let Some(obj) = arguments_raw.as_object() {
                            obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
                        } else {
                            HashMap::new()
                        };
                    tool_calls.push(ToolCallRequest {
                        id,
                        name,
                        arguments,
                    });
                }
            }

            // Fallback: parse tool calls from text when model doesn't generate
            // proper tool_calls JSON (e.g. models without tool-calling templates).
            // Looks for {"name": "...", "arguments": {...}} blocks in content.
            let (tool_calls, text) = if tool_calls.is_empty() {
                parse_tool_calls_from_text(&text)
            } else {
                (tool_calls, text)
            };

            let finish_reason = if !tool_calls.is_empty() && finish_reason == "stop" {
                "tool_calls".to_string()
            } else {
                finish_reason
            };

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
                content: if text.trim().is_empty() && !tool_calls.is_empty() {
                    None
                } else {
                    Some(text)
                },
                tool_calls,
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
            tools: Option<&[serde_json::Value]>,
            _model: Option<&str>,
            max_tokens: u32,
            temperature: f64,
            _thinking_budget: Option<u32>,
            _top_p: Option<f64>,
        ) -> Result<LLMResponse> {
            // Delegate to mlx-lm server when configured
            if let Some(ref url) = self.mlx_lm_url {
                return self
                    .chat_via_mlx_lm(url, messages, tools, max_tokens, temperature)
                    .await;
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

            let text = strip_think_blocks(&raw_text);

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

        async fn chat_stream(
            &self,
            messages: &[serde_json::Value],
            tools: Option<&[serde_json::Value]>,
            _model: Option<&str>,
            max_tokens: u32,
            temperature: f64,
            _thinking_budget: Option<u32>,
            _top_p: Option<f64>,
        ) -> Result<crate::providers::base::StreamHandle> {
            use crate::providers::base::{StreamChunk, StreamHandle};
            use futures_util::StreamExt;

            // Only stream when delegating to mlx-lm server
            let Some(ref base_url) = self.mlx_lm_url else {
                // In-process: fall back to buffered default
                let response = self
                    .chat(messages, tools, None, max_tokens, temperature, None, None)
                    .await?;
                let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
                if let Some(ref content) = response.content {
                    let _ = tx.send(StreamChunk::TextDelta(content.clone()));
                }
                let _ = tx.send(StreamChunk::Done(response));
                return Ok(StreamHandle { rx });
            };

            let url = format!("{}/v1/chat/completions", base_url.trim_end_matches('/'));
            let msgs = self.inject_nothink(messages);
            let mut body = serde_json::json!({
                "model": &self.model_path,
                "messages": msgs,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": true,
            });
            if let Some(tools) = tools {
                if !tools.is_empty() {
                    body["tools"] = serde_json::json!(tools);
                }
            }

            let resp = self
                .http_client
                .post(&url)
                .json(&body)
                .send()
                .await
                .context("mlx-lm server stream request failed")?;

            if !resp.status().is_success() {
                let status = resp.status();
                let text = resp.text().await.unwrap_or_default();
                anyhow::bail!("mlx-lm server returned {status}: {text}");
            }

            let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

            // Spawn task to parse SSE stream from mlx-lm server
            let byte_stream = resp.bytes_stream();
            tokio::spawn(async move {
                let mut stream = Box::pin(byte_stream);
                let mut line_buffer = String::new();
                let mut full_content = String::new();
                let mut finish_reason = "stop".to_string();
                let mut usage: HashMap<String, i64> = HashMap::new();
                let mut tool_calls_acc: HashMap<u64, (String, String, String)> = HashMap::new();
                let mut think_split = crate::providers::openai_compat::ThinkSplitState::default();

                while let Some(result) = stream.next().await {
                    let bytes = match result {
                        Ok(b) => b,
                        Err(e) => {
                            tracing::warn!("mlx-lm SSE error: {e}");
                            break;
                        }
                    };

                    line_buffer.push_str(&String::from_utf8_lossy(&bytes));

                    while let Some(newline_pos) = line_buffer.find('\n') {
                        let line = line_buffer[..newline_pos]
                            .trim_end_matches('\r')
                            .to_string();
                        line_buffer = line_buffer[newline_pos + 1..].to_string();

                        if line.is_empty() || !line.starts_with("data: ") {
                            continue;
                        }
                        let data = &line[6..];
                        if data == "[DONE]" {
                            break;
                        }

                        let Ok(chunk) = serde_json::from_str::<serde_json::Value>(data) else {
                            continue;
                        };

                        // Extract text delta
                        if let Some(delta) = chunk
                            .pointer("/choices/0/delta/content")
                            .and_then(|v| v.as_str())
                        {
                            if !delta.is_empty() {
                                // Split <think>/<thinking> blocks using the
                                // tested splitter from openai_compat (handles
                                // tags split across chunk boundaries).
                                let (visible, thinking) =
                                    crate::providers::openai_compat::split_thinking_from_content_delta(
                                        &mut think_split, delta,
                                    );
                                if !thinking.is_empty() {
                                    let _ = tx.send(StreamChunk::ThinkingDelta(thinking));
                                }
                                if !visible.is_empty() {
                                    full_content.push_str(&visible);
                                    let _ = tx.send(StreamChunk::TextDelta(visible));
                                }
                            }
                        }

                        // Accumulate tool calls from deltas
                        if let Some(tc_array) = chunk
                            .pointer("/choices/0/delta/tool_calls")
                            .and_then(|v| v.as_array())
                        {
                            for tc in tc_array {
                                let idx = tc.get("index").and_then(|v| v.as_u64()).unwrap_or(0);
                                let entry = tool_calls_acc.entry(idx).or_insert_with(|| {
                                    (String::new(), String::new(), String::new())
                                });
                                if let Some(id) = tc.get("id").and_then(|v| v.as_str()) {
                                    entry.0 = id.to_string();
                                }
                                if let Some(f) = tc.get("function") {
                                    if let Some(name) = f.get("name").and_then(|v| v.as_str()) {
                                        entry.1 = name.to_string();
                                    }
                                    if let Some(args) = f.get("arguments").and_then(|v| v.as_str())
                                    {
                                        entry.2.push_str(args);
                                    }
                                }
                            }
                        }

                        if let Some(fr) = chunk
                            .pointer("/choices/0/finish_reason")
                            .and_then(|v| v.as_str())
                        {
                            finish_reason = fr.to_string();
                        }

                        // Usage in final chunk
                        if let Some(u) = chunk.get("usage") {
                            for key in &["prompt_tokens", "completion_tokens", "total_tokens"] {
                                if let Some(n) = u.get(*key).and_then(|v| v.as_i64()) {
                                    usage.insert(key.to_string(), n);
                                }
                            }
                        }
                    }
                }

                // Flush any carried-over content from the think splitter.
                let (tail_visible, tail_thinking) =
                    crate::providers::openai_compat::flush_thinking_split_state(&mut think_split);
                if !tail_thinking.is_empty() {
                    let _ = tx.send(StreamChunk::ThinkingDelta(tail_thinking));
                }
                if !tail_visible.is_empty() {
                    full_content.push_str(&tail_visible);
                    let _ = tx.send(StreamChunk::TextDelta(tail_visible));
                }

                // Build tool calls
                let mut tool_calls = Vec::new();
                let mut indices: Vec<u64> = tool_calls_acc.keys().copied().collect();
                indices.sort();
                for idx in indices {
                    let (id, name, args_json) = tool_calls_acc.remove(&idx).unwrap();
                    let arguments: HashMap<String, serde_json::Value> =
                        serde_json::from_str(&args_json).unwrap_or_default();
                    tool_calls.push(ToolCallRequest {
                        id,
                        name,
                        arguments,
                    });
                }

                // Fallback: parse tool calls from text
                let (tool_calls, text) = if tool_calls.is_empty() {
                    parse_tool_calls_from_text(&full_content)
                } else {
                    (tool_calls, full_content)
                };

                let finish_reason = if !tool_calls.is_empty() && finish_reason == "stop" {
                    "tool_calls".to_string()
                } else {
                    finish_reason
                };

                let _ = tx.send(StreamChunk::Done(LLMResponse {
                    content: if text.trim().is_empty() && !tool_calls.is_empty() {
                        None
                    } else {
                        Some(text)
                    },
                    tool_calls,
                    finish_reason,
                    usage,
                }));
            });

            Ok(StreamHandle { rx })
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
            LoraConfig {
                lr: 1e-5,
                ..LoraConfig::default()
            },
        )
        .expect("MlxProvider::start failed");

        let messages = vec![serde_json::json!({
            "role": "user",
            "content": "What is 2+2? Answer with just the number."
        })];

        let resp = provider
            .chat(&messages, None, None, 32, 0.0, None, None)
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
            LoraConfig {
                lr: 1e-5,
                ..LoraConfig::default()
            },
        )
        .expect("MlxProvider::start failed");

        let loss = provider
            .perplexity(
                "What is the capital of France?",
                "The capital of France is Paris.",
            )
            .await
            .expect("perplexity failed");

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
            LoraConfig {
                lr: 1e-5,
                ..LoraConfig::default()
            },
        )
        .expect("MlxProvider::start failed");

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
        provider
            .train(conversations, 2)
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
            LoraConfig {
                lr: 1e-5,
                ..LoraConfig::default()
            },
        )
        .expect("MlxProvider::start failed");

        // 1. Inference
        let messages = vec![serde_json::json!({
            "role": "user",
            "content": "Capital of Japan?"
        })];
        let resp = provider
            .chat(&messages, None, None, 16, 0.0, None, None)
            .await
            .expect("chat failed");
        let answer = resp.content.unwrap_or_default();
        eprintln!("Inference: {answer:?}");
        assert!(!answer.is_empty());

        // 2. Perplexity scoring
        let loss = provider
            .perplexity("Capital of Japan?", &answer)
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
        provider
            .train(convos, 1)
            .await
            .expect("train should not error");

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
            LoraConfig {
                lr: 1e-5,
                ..LoraConfig::default()
            },
        )
        .expect("MlxProvider::start failed (4B)");

        // 1. Inference
        let messages = vec![serde_json::json!({
            "role": "user",
            "content": "What is 2+2? Answer with just the number."
        })];
        // Thinking models need more tokens: ~100 for reasoning + answer
        let resp = provider
            .chat(&messages, None, None, 256, 0.0, None, None)
            .await
            .expect("chat failed");
        let answer = resp.content.unwrap_or_default();
        eprintln!("4B Inference: {answer:?}");
        assert!(
            !answer.is_empty(),
            "response empty after think-strip (raw may have had thinking)"
        );

        // 2. Perplexity
        let loss = provider
            .perplexity("What is 2+2?", &answer)
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
        provider
            .train(convos, 1)
            .await
            .expect("train should not error");

        eprintln!("4B closed loop complete: inference → perplexity → train");
    }
}
