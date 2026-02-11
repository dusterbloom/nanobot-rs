//! Async message queue for decoupled channel-agent communication.
//!
//! Uses `tokio::sync::mpsc::unbounded_channel` for inbound/outbound queues and
//! supports subscriber callbacks for routing outbound messages to channels.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use tokio::sync::mpsc::{self, UnboundedReceiver, UnboundedSender};
use tokio::sync::Mutex;
use tracing::error;

use crate::bus::events::{InboundMessage, OutboundMessage};

/// Callback type for outbound message subscribers.
///
/// Each callback receives an `OutboundMessage` and returns a pinned future
/// that resolves to `()`.
pub type OutboundCallback =
    Arc<dyn Fn(OutboundMessage) -> Pin<Box<dyn Future<Output = ()> + Send>> + Send + Sync>;

/// Async message bus that decouples chat channels from the agent core.
///
/// Channels push messages to the inbound queue, and the agent processes them
/// and pushes responses to the outbound queue. The bus can be cloned cheaply
/// because all internal state is behind `Arc`.
#[derive(Clone)]
pub struct MessageBus {
    /// Sender half for inbound messages.
    inbound_tx: UnboundedSender<InboundMessage>,
    /// Receiver half for inbound messages (shared so only one consumer drains).
    inbound_rx: Arc<Mutex<UnboundedReceiver<InboundMessage>>>,
    /// Sender half for outbound messages.
    outbound_tx: UnboundedSender<OutboundMessage>,
    /// Receiver half for outbound messages (shared so only one consumer drains).
    outbound_rx: Arc<Mutex<UnboundedReceiver<OutboundMessage>>>,
    /// Subscribers keyed by channel name.
    subscribers: Arc<Mutex<HashMap<String, Vec<OutboundCallback>>>>,
    /// Flag to control the dispatch loop.
    running: Arc<AtomicBool>,
}

impl MessageBus {
    /// Create a new `MessageBus`.
    pub fn new() -> Self {
        let (inbound_tx, inbound_rx) = mpsc::unbounded_channel();
        let (outbound_tx, outbound_rx) = mpsc::unbounded_channel();
        Self {
            inbound_tx,
            inbound_rx: Arc::new(Mutex::new(inbound_rx)),
            outbound_tx,
            outbound_rx: Arc::new(Mutex::new(outbound_rx)),
            subscribers: Arc::new(Mutex::new(HashMap::new())),
            running: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Publish a message from a channel to the agent (inbound).
    pub fn publish_inbound(&self, msg: InboundMessage) {
        let _ = self.inbound_tx.send(msg);
    }

    /// Consume the next inbound message (blocks until available).
    ///
    /// Returns `None` if all senders have been dropped.
    pub async fn consume_inbound(&self) -> Option<InboundMessage> {
        let mut rx = self.inbound_rx.lock().await;
        rx.recv().await
    }

    /// Publish a response from the agent to channels (outbound).
    pub fn publish_outbound(&self, msg: OutboundMessage) {
        let _ = self.outbound_tx.send(msg);
    }

    /// Consume the next outbound message (blocks until available).
    ///
    /// Returns `None` if all senders have been dropped.
    pub async fn consume_outbound(&self) -> Option<OutboundMessage> {
        let mut rx = self.outbound_rx.lock().await;
        rx.recv().await
    }

    /// Subscribe to outbound messages for a specific channel.
    ///
    /// The callback will be invoked whenever a message targeted at `channel` is
    /// dispatched.
    pub async fn subscribe_outbound(&self, channel: impl Into<String>, callback: OutboundCallback) {
        let mut subs = self.subscribers.lock().await;
        subs.entry(channel.into())
            .or_insert_with(Vec::new)
            .push(callback);
    }

    /// Dispatch outbound messages to subscribed channel callbacks.
    ///
    /// This should be run as a background task (e.g. `tokio::spawn`). It loops
    /// until [`stop`](Self::stop) is called.
    pub async fn dispatch_outbound(&self) {
        self.running.store(true, Ordering::SeqCst);

        while self.running.load(Ordering::SeqCst) {
            let msg = {
                let mut rx = self.outbound_rx.lock().await;
                // Use a timeout so we can check the running flag periodically.
                match tokio::time::timeout(std::time::Duration::from_secs(1), rx.recv()).await {
                    Ok(Some(msg)) => msg,
                    Ok(None) => {
                        // Channel closed -- all senders dropped.
                        break;
                    }
                    Err(_) => {
                        // Timeout -- loop back and check running flag.
                        continue;
                    }
                }
            };

            let subs = self.subscribers.lock().await;
            if let Some(callbacks) = subs.get(&msg.channel) {
                for callback in callbacks {
                    let fut = callback(msg.clone());
                    if let Err(e) = tokio::spawn(fut).await {
                        error!("Error dispatching to {}: {}", msg.channel, e);
                    }
                }
            }
        }
    }

    /// Stop the dispatcher loop.
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
    }

    /// Check whether the dispatcher is currently running.
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }
}

impl Default for MessageBus {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_inbound_publish_consume() {
        let bus = MessageBus::new();
        let msg = InboundMessage::new("telegram", "user1", "chat1", "hello");
        bus.publish_inbound(msg);

        let received = bus.consume_inbound().await.unwrap();
        assert_eq!(received.channel, "telegram");
        assert_eq!(received.content, "hello");
    }

    #[tokio::test]
    async fn test_outbound_publish_consume() {
        let bus = MessageBus::new();
        let msg = OutboundMessage::new("whatsapp", "+1234", "response");
        bus.publish_outbound(msg);

        let received = bus.consume_outbound().await.unwrap();
        assert_eq!(received.channel, "whatsapp");
        assert_eq!(received.content, "response");
    }

    #[tokio::test]
    async fn test_subscribe_and_dispatch() {
        let bus = MessageBus::new();
        let received = Arc::new(Mutex::new(Vec::<String>::new()));
        let received_clone = received.clone();

        let callback: OutboundCallback = Arc::new(move |msg: OutboundMessage| {
            let captured = received_clone.clone();
            Box::pin(async move {
                let mut v = captured.lock().await;
                v.push(msg.content);
            })
        });

        bus.subscribe_outbound("telegram", callback).await;

        // Spawn dispatcher.
        let bus_clone = bus.clone();
        let handle = tokio::spawn(async move {
            bus_clone.dispatch_outbound().await;
        });

        // Give dispatcher time to start.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        // Publish a message.
        bus.publish_outbound(OutboundMessage::new("telegram", "chat1", "dispatched!"));

        // Give dispatcher time to process.
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        bus.stop();
        let _ = handle.await;

        let messages = received.lock().await;
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0], "dispatched!");
    }

    #[tokio::test]
    async fn test_stop() {
        let bus = MessageBus::new();
        assert!(!bus.is_running());

        let bus_clone = bus.clone();
        let handle = tokio::spawn(async move {
            bus_clone.dispatch_outbound().await;
        });

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        assert!(bus.is_running());

        bus.stop();
        let _ = handle.await;
        assert!(!bus.is_running());
    }
}
