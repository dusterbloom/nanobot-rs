//! Channel manager for coordinating chat channels.
//!
//! Initialises enabled channels, starts them, and dispatches outbound messages
//! to the correct channel.

use std::collections::HashMap;
use std::sync::Arc;

use serde_json::{json, Value};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use tokio::sync::Mutex as TokioMutex;
use tracing::{error, info, warn};

use crate::bus::events::{InboundMessage, OutboundMessage};
use crate::channels::base::Channel;
use crate::channels::email::EmailChannel;
use crate::channels::feishu::FeishuChannel;
use crate::channels::telegram::TelegramChannel;
use crate::channels::whatsapp::WhatsAppChannel;
use crate::config::schema::Config;

#[cfg(feature = "voice")]
use crate::voice_pipeline::VoicePipeline;

/// Manages chat channels and coordinates message routing.
pub struct ChannelManager {
    channels: HashMap<String, Arc<TokioMutex<Box<dyn Channel>>>>,
    bus_outbound_rx: Arc<TokioMutex<UnboundedReceiver<OutboundMessage>>>,
}

impl ChannelManager {
    /// Create a new `ChannelManager`, initialising enabled channels.
    pub fn new(
        config: &Config,
        bus_inbound_tx: UnboundedSender<InboundMessage>,
        bus_outbound_rx: UnboundedReceiver<OutboundMessage>,
        #[cfg(feature = "voice")] voice_pipeline: Option<Arc<VoicePipeline>>,
    ) -> Self {
        let mut channels: HashMap<String, Arc<TokioMutex<Box<dyn Channel>>>> = HashMap::new();

        // Telegram.
        if config.channels.telegram.enabled {
            let groq_key = config.providers.groq.api_key.clone();
            let ch = TelegramChannel::new(
                config.channels.telegram.clone(),
                bus_inbound_tx.clone(),
                groq_key,
                #[cfg(feature = "voice")]
                voice_pipeline.clone(),
            );
            channels.insert(
                "telegram".to_string(),
                Arc::new(TokioMutex::new(Box::new(ch))),
            );
            info!("Telegram channel enabled");
        }

        // WhatsApp.
        if config.channels.whatsapp.enabled {
            let ch = WhatsAppChannel::new(
                config.channels.whatsapp.clone(),
                bus_inbound_tx.clone(),
                #[cfg(feature = "voice")]
                voice_pipeline.clone(),
            );
            channels.insert(
                "whatsapp".to_string(),
                Arc::new(TokioMutex::new(Box::new(ch))),
            );
            info!("WhatsApp channel enabled");
        }

        // Feishu.
        if config.channels.feishu.enabled {
            let ch = FeishuChannel::new(
                config.channels.feishu.clone(),
                bus_inbound_tx.clone(),
            );
            channels.insert(
                "feishu".to_string(),
                Arc::new(TokioMutex::new(Box::new(ch))),
            );
            info!("Feishu channel enabled");
        }

        // Email.
        if config.channels.email.enabled {
            let ch = EmailChannel::new(
                config.channels.email.clone(),
                bus_inbound_tx.clone(),
            );
            channels.insert(
                "email".to_string(),
                Arc::new(TokioMutex::new(Box::new(ch))),
            );
            info!("Email channel enabled");
        }

        Self {
            channels,
            bus_outbound_rx: Arc::new(TokioMutex::new(bus_outbound_rx)),
        }
    }

    /// Start all enabled channels and the outbound message dispatcher.
    pub async fn start_all(&self) {
        if self.channels.is_empty() {
            warn!("No channels enabled");
            return;
        }

        // Start each channel.
        for (name, channel) in &self.channels {
            let ch = channel.clone();
            let channel_name = name.clone();
            tokio::spawn(async move {
                let mut guard = ch.lock().await;
                if let Err(e) = guard.start().await {
                    error!("Failed to start {} channel: {}", channel_name, e);
                }
            });
        }

        // Start the outbound dispatcher.
        let channels = self.channels.clone();
        let rx = self.bus_outbound_rx.clone();

        tokio::spawn(async move {
            info!("Outbound dispatcher started");
            loop {
                let msg = {
                    let mut guard = rx.lock().await;
                    match guard.recv().await {
                        Some(m) => m,
                        None => {
                            info!("Outbound channel closed, dispatcher stopping");
                            break;
                        }
                    }
                };

                if let Some(channel) = channels.get(&msg.channel) {
                    let guard = channel.lock().await;
                    if let Err(e) = guard.send(&msg).await {
                        error!("Error sending to {}: {}", msg.channel, e);
                    }
                } else {
                    warn!("Unknown channel: {}", msg.channel);
                }
            }
        });
    }

    /// Stop all channels.
    pub async fn stop_all(&self) {
        info!("Stopping all channels...");

        for (name, channel) in &self.channels {
            let mut guard = channel.lock().await;
            if let Err(e) = guard.stop().await {
                error!("Error stopping {} channel: {}", name, e);
            } else {
                info!("Stopped {} channel", name);
            }
        }
    }

    /// Get status information for all channels.
    pub fn get_status(&self) -> HashMap<String, Value> {
        let mut status = HashMap::new();

        for name in self.channels.keys() {
            status.insert(
                name.clone(),
                json!({
                    "enabled": true,
                    "type": name,
                }),
            );
        }

        status
    }

    /// Get the list of enabled channel names.
    pub fn enabled_channels(&self) -> Vec<String> {
        self.channels.keys().cloned().collect()
    }
}
