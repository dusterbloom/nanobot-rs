//! Base trait for chat channels.

use anyhow::Result;
use async_trait::async_trait;

use crate::bus::events::OutboundMessage;

/// Trait that every chat channel must implement.
///
/// Channels are responsible for receiving user messages and forwarding them to
/// the message bus, and for delivering outbound agent responses.
#[async_trait]
pub trait Channel: Send + Sync {
    /// Human-readable channel name (e.g. `"telegram"`).
    fn name(&self) -> &str;

    /// Start the channel.
    ///
    /// Implementations should set up any polling / WebSocket listeners and
    /// spawn internal background tasks, then return.  The background tasks
    /// should keep running until [`stop`](Self::stop) is called.
    async fn start(&mut self) -> Result<()>;

    /// Stop the channel gracefully.
    async fn stop(&mut self) -> Result<()>;

    /// Send an outbound message through this channel.
    async fn send(&self, msg: &OutboundMessage) -> Result<()>;

    /// Check whether `sender_id` is in the allow list.
    ///
    /// If the allow list is empty every sender is permitted.  The
    /// `sender_id` may contain pipe-separated parts (e.g. `"id|username"`)
    /// that are checked individually.
    fn is_allowed(&self, sender_id: &str, allow_list: &[String]) -> bool {
        if allow_list.is_empty() {
            return true;
        }
        let sender_str = sender_id.to_string();
        if allow_list.contains(&sender_str) {
            return true;
        }
        if sender_str.contains('|') {
            for part in sender_str.split('|') {
                if !part.is_empty() && allow_list.contains(&part.to_string()) {
                    return true;
                }
            }
        }
        false
    }

    /// Check whether the channel is currently running.
    fn is_running(&self) -> bool;
}
