#![allow(dead_code)]
//! Feishu/Lark channel implementation.
//!
//! This is a simplified stub: receiving messages via WebSocket long connection
//! requires the Feishu Lark SDK which is not available in Rust.  Sending
//! messages is done via the Feishu HTTP API directly.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use serde_json::{json, Value};
use tokio::sync::mpsc::UnboundedSender;
use tokio::sync::Mutex as TokioMutex;
use tracing::{error, info, warn};

use crate::bus::events::{InboundMessage, OutboundMessage};
use crate::channels::base::Channel;
use crate::config::schema::FeishuConfig;

/// Feishu/Lark channel.
///
/// Receiving messages is currently not supported because the Lark SDK WebSocket
/// long connection API has no Rust equivalent.  Sending messages uses the
/// Feishu HTTP API.
pub struct FeishuChannel {
    config: FeishuConfig,
    bus_tx: UnboundedSender<InboundMessage>,
    running: Arc<AtomicBool>,
    client: reqwest::Client,
    /// Cached tenant access token.
    token: Arc<TokioMutex<Option<String>>>,
}

impl FeishuChannel {
    /// Create a new `FeishuChannel`.
    pub fn new(config: FeishuConfig, bus_tx: UnboundedSender<InboundMessage>) -> Self {
        Self {
            config,
            bus_tx,
            running: Arc::new(AtomicBool::new(false)),
            client: reqwest::Client::new(),
            token: Arc::new(TokioMutex::new(None)),
        }
    }

    /// Refresh the tenant access token via the Feishu API.
    async fn _refresh_token(&self) -> Result<String> {
        let resp = self
            .client
            .post("https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal")
            .json(&json!({
                "app_id": self.config.app_id,
                "app_secret": self.config.app_secret,
            }))
            .send()
            .await?;

        let data: Value = resp.json().await?;
        let code = data.get("code").and_then(|v| v.as_i64()).unwrap_or(-1);

        if code != 0 {
            let msg = data
                .get("msg")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown error");
            return Err(anyhow::anyhow!(
                "Feishu token refresh failed (code {}): {}",
                code,
                msg
            ));
        }

        let token = data
            .get("tenant_access_token")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing tenant_access_token in response"))?
            .to_string();

        {
            let mut slot = self.token.lock().await;
            *slot = Some(token.clone());
        }

        Ok(token)
    }

    /// Get a valid tenant access token (refresh if needed).
    async fn _get_token(&self) -> Result<String> {
        {
            let slot = self.token.lock().await;
            if let Some(ref t) = *slot {
                return Ok(t.clone());
            }
        }
        self._refresh_token().await
    }
}

#[async_trait]
impl Channel for FeishuChannel {
    fn name(&self) -> &str {
        "feishu"
    }

    async fn start(&mut self) -> Result<()> {
        if self.config.app_id.is_empty() || self.config.app_secret.is_empty() {
            return Err(anyhow::anyhow!(
                "Feishu app_id and app_secret not configured"
            ));
        }

        self.running.store(true, Ordering::SeqCst);

        info!(
            "Feishu channel started (send-only mode). \
             Receiving messages via WebSocket long connection requires the Lark SDK, \
             which is not yet available in Rust."
        );

        // Pre-fetch a token so that send() works right away.
        match self._refresh_token().await {
            Ok(_) => info!("Feishu tenant access token acquired"),
            Err(e) => warn!("Failed to acquire Feishu token: {}", e),
        }

        Ok(())
    }

    async fn stop(&mut self) -> Result<()> {
        self.running.store(false, Ordering::SeqCst);
        info!("Feishu channel stopped");
        Ok(())
    }

    async fn send(&self, msg: &OutboundMessage) -> Result<()> {
        let token = match self._get_token().await {
            Ok(t) => t,
            Err(e) => {
                // Try refreshing once.
                match self._refresh_token().await {
                    Ok(t) => t,
                    Err(e2) => {
                        error!("Cannot send Feishu message: {}, {}", e, e2);
                        return Err(e2);
                    }
                }
            }
        };

        // Determine receive_id_type based on chat_id format.
        let receive_id_type = if msg.chat_id.starts_with("oc_") {
            "chat_id"
        } else {
            "open_id"
        };

        let content = json!({"text": msg.content}).to_string();

        let resp = self
            .client
            .post(format!(
                "https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type={}",
                receive_id_type
            ))
            .header("Authorization", format!("Bearer {}", token))
            .header("Content-Type", "application/json; charset=utf-8")
            .json(&json!({
                "receive_id": msg.chat_id,
                "msg_type": "text",
                "content": content,
            }))
            .send()
            .await?;

        let data: Value = resp.json().await.unwrap_or_default();
        let code = data.get("code").and_then(|v| v.as_i64()).unwrap_or(-1);

        if code != 0 {
            let err_msg = data
                .get("msg")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown error");
            // If token expired, try refreshing.
            if code == 99991663 || code == 99991664 {
                warn!("Feishu token expired, refreshing and retrying...");
                let new_token = self._refresh_token().await?;
                let retry_resp = self
                    .client
                    .post(format!(
                        "https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type={}",
                        receive_id_type
                    ))
                    .header("Authorization", format!("Bearer {}", new_token))
                    .header("Content-Type", "application/json; charset=utf-8")
                    .json(&json!({
                        "receive_id": msg.chat_id,
                        "msg_type": "text",
                        "content": content,
                    }))
                    .send()
                    .await?;

                let retry_data: Value = retry_resp.json().await.unwrap_or_default();
                let retry_code = retry_data
                    .get("code")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(-1);
                if retry_code != 0 {
                    let retry_msg = retry_data
                        .get("msg")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown");
                    return Err(anyhow::anyhow!(
                        "Feishu send failed after retry (code {}): {}",
                        retry_code,
                        retry_msg
                    ));
                }
            } else {
                return Err(anyhow::anyhow!(
                    "Feishu send failed (code {}): {}",
                    code,
                    err_msg
                ));
            }
        }

        Ok(())
    }

    fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }
}
