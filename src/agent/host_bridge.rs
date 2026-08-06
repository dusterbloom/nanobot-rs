//! Typed host bridge — the single protocol between agent tools and the
//! agent-loop host (research §3: docs/research/2026-08-06-error-conventions-and-host-bridge.md).
//!
//! Replaces the 8 `Arc<Mutex<Option<…>>>` callback slots (7 on `SpawnTool`,
//! 1 on `MessageTool`) with one typed request/reply protocol, a registry
//! dispatcher (OCP), and per-capability traits (ISP). Mirrors prime-agent's
//! `host.request` comm target: a type-tagged request envelope, a discriminated
//! reply union (`ok | error`), and strict typed validation at the boundary.
//!
//! Layout (matching the research doc's migration path):
//! - wire protocol: [`HostRequest`] / [`HostReply`] + typed payload structs
//! - capability traits: [`SpawnHost`], [`PipelineHost`], [`LoopHost`],
//!   [`MessageHost`], composed into the [`HostBridge`] blanket trait (DIP)
//! - [`HostDispatcher`]: the single `match` (OCP seam) + the `From<Result<…>>`
//!   envelope construction (DRY)
//! - [`HostBridgeAdapter`]: transitional (doc §3.7 Step 1) — implements the
//!   traits by invoking the *legacy* callback slots, so the dispatcher can be
//!   exercised end-to-end against the pre-bridge wiring byte-for-byte.

// Additive module landing ahead of its consumers: the protocol types are
// exercised by the tests in this commit; production wiring (AgentHost in
// `tool_wiring`, `ToolRegistry::with_host`) consumes them in the next steps.
// Remove this allow when the wiring lands (scoped-allow style, error
// protocol §2.5).
#![allow(dead_code)]

use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::agent::tools::message::SendCallback;
use crate::agent::tools::spawn::{
    CancelCallback, CheckCallback, ListCallback, LoopCallback, PipelineCallback, SpawnCallback,
    WaitCallback,
};
use crate::bus::events::OutboundMessage;
use crate::errors::ToolError;

// ---------------------------------------------------------------------------
// Wire protocol (serde) — transport-agnostic
// ---------------------------------------------------------------------------

/// One typed request the agent-loop host can answer. The variant name is the
/// wire discriminator (`"type"`), exactly like prime-agent's `host.request`.
///
/// The tag is the *only* `type` key on the wire: with serde's internal
/// tagging the discriminator is consumed as the route, so a payload can never
/// carry its own `type` field and spoof the dispatch target.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum HostRequest {
    Spawn(SpawnRequest),
    ListSubagents(ListSubagentsRequest),
    CancelSubagent(CancelRequest),
    WaitSubagent(WaitRequest),
    CheckSubagent(CheckRequest),
    RunPipeline(PipelineRequest),
    RunLoop(LoopRequest),
    SendMessage(SendMessageRequest),
}

/// Spawn a background subagent (the contract that replaces the positional
/// `SpawnCallback`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpawnRequest {
    pub task: String,
    #[serde(default)]
    pub label: Option<String>,
    #[serde(default)]
    pub profile: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub working_dir: Option<String>,
    /// Origin channel/chat — baked in by the host at construction; never sent
    /// over the wire (`#[serde(skip)]` ⇒ absent from the envelope).
    #[serde(skip)]
    pub channel: String,
    #[serde(skip)]
    pub chat_id: String,
}

/// List running subagents. Unit payload — the route alone is the request.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ListSubagentsRequest {}

/// Cancel a running subagent by task-id prefix.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CancelRequest {
    pub task_id: String,
}

/// Block until a subagent completes or the timeout elapses.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WaitRequest {
    pub task_id: String,
    #[serde(default = "default_wait_timeout")]
    pub timeout_secs: u64,
}

/// Default wait timeout, matching the tool schema's `timeout` default (120s).
const fn default_wait_timeout() -> u64 {
    120
}

/// Non-blocking result lookup for a subagent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckRequest {
    pub task_id: String,
}

/// Run a multi-step pipeline (the `pipeline` spawn action).
#[allow(clippy::derive_partial_eq_without_eq)] // steps is Vec<serde_json::Value> — Value is not Eq
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PipelineRequest {
    /// Raw pipeline steps array (the tool's `steps` param, verbatim).
    #[serde(default)]
    pub steps: Vec<serde_json::Value>,
    #[serde(default)]
    pub ahead_by_k: usize,
}

/// Run an autonomous refinement loop (the `loop` spawn action).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoopRequest {
    pub task: String,
    #[serde(default = "default_max_rounds")]
    pub max_rounds: u32,
    #[serde(default)]
    pub tools: Option<Vec<String>>,
    #[serde(default)]
    pub stop_condition: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub working_dir: Option<String>,
}

/// Default loop rounds, matching the tool schema's `max_rounds` default (5).
const fn default_max_rounds() -> u32 {
    5
}

/// Send an explicit out-of-band notification (the `SendCallback` contract).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SendMessageRequest {
    /// Resolved target channel (defaults already applied by the caller).
    pub channel: String,
    /// Resolved target chat id (defaults already applied by the caller).
    pub chat_id: String,
    pub content: String,
}

// ---------------------------------------------------------------------------
// Typed replies — the contract that replaces positional `String` returns
// ---------------------------------------------------------------------------

/// Every reply carries the byte-stable model-visible `text`: the exact string
/// the legacy callback produced, so adopting the bridge cannot change what
/// the model sees. Structured fields (e.g. `task_id`) are additive; the
/// rendered string remains the single source of truth for tool output.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpawnReply {
    /// Structured task id (best-effort extraction from the spawn line).
    pub task_id: String,
    /// The model-visible spawn confirmation text.
    pub text: String,
}

macro_rules! text_reply {
    ($name:ident, $doc:literal) => {
        #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
        #[doc = $doc]
        pub struct $name {
            /// The model-visible output text (byte-stable with the legacy
            /// callback's formatted string).
            pub text: String,
        }
    };
}

text_reply!(ListReply, "Formatted listing of running + recently completed subagents.");
text_reply!(CancelReply, "Cancellation confirmation text.");
text_reply!(WaitReply, "Blocking wait result text.");
text_reply!(CheckReply, "Non-blocking status/result text.");
text_reply!(PipelineReply, "Formatted pipeline completion summary.");
text_reply!(LoopReply, "Autonomous refinement loop result text.");
text_reply!(SendMessageReply, "Send confirmation text.");

/// Discriminated reply envelope — the single error channel of the bridge
/// (prime-agent's `{status:"ok"|"error"}` union, typed).
///
/// `Error` carries [`ToolError::render()`] output today; the design is ready
/// to carry the typed [`ToolError`] directly once the wire is fully typed
/// (research §4 risk 3).
#[allow(clippy::derive_partial_eq_without_eq)] // flattened payload is serde_json::Value — not Eq
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum HostReply {
    Ok {
        #[serde(flatten)]
        data: serde_json::Value,
    },
    Error { error: String },
}

// ---------------------------------------------------------------------------
// Capability traits (ISP) + composite (DIP)
// ---------------------------------------------------------------------------

/// Subagent lifecycle capabilities (spawn/list/cancel/wait/check).
#[async_trait]
pub trait SpawnHost: Send + Sync {
    async fn spawn(&self, req: SpawnRequest) -> Result<SpawnReply, ToolError>;
    async fn list_subagents(&self) -> Result<ListReply, ToolError>;
    async fn cancel(&self, req: CancelRequest) -> Result<CancelReply, ToolError>;
    async fn wait(&self, req: WaitRequest) -> Result<WaitReply, ToolError>;
    async fn check(&self, req: CheckRequest) -> Result<CheckReply, ToolError>;
}

/// Multi-step pipeline capability.
#[async_trait]
pub trait PipelineHost: Send + Sync {
    async fn run_pipeline(&self, req: PipelineRequest) -> Result<PipelineReply, ToolError>;
}

/// Autonomous refinement loop capability.
#[async_trait]
pub trait LoopHost: Send + Sync {
    async fn run_loop(&self, req: LoopRequest) -> Result<LoopReply, ToolError>;
}

/// Out-of-band message capability.
#[async_trait]
pub trait MessageHost: Send + Sync {
    async fn send(&self, req: SendMessageRequest) -> Result<SendMessageReply, ToolError>;
}

/// Composite capability set. Blanket impl: any type implementing all four
/// traits IS a [`HostBridge`] (DIP: depend on this, never on
/// `SubagentManager` or callback slots).
pub trait HostBridge: SpawnHost + PipelineHost + LoopHost + MessageHost {}
impl<T: SpawnHost + PipelineHost + LoopHost + MessageHost> HostBridge for T {}

// ---------------------------------------------------------------------------
// HostDispatcher — the single `match` (OCP), the DRY replacement for the
// 8 `Arc<Mutex<Option<…>>>` fields + setters + lock/clone/drop/await blocks
// ---------------------------------------------------------------------------

/// Registry dispatcher: envelope in, envelope out. This is what an in-process
/// `call()` or a future UDS/ZeroMQ server both invoke. Implements the four
/// capability traits *by delegation*, so the dispatcher itself IS a composite
/// [`HostBridge`] — `Arc<dyn HostBridge>` can be a dispatcher over mocks in
/// tests or over the production host in `build_tools`.
pub struct HostDispatcher {
    spawn: Arc<dyn SpawnHost>,
    pipeline: Arc<dyn PipelineHost>,
    loop_: Arc<dyn LoopHost>,
    message: Arc<dyn MessageHost>,
}

impl HostDispatcher {
    pub fn new(
        spawn: Arc<dyn SpawnHost>,
        pipeline: Arc<dyn PipelineHost>,
        loop_: Arc<dyn LoopHost>,
        message: Arc<dyn MessageHost>,
    ) -> Self {
        Self { spawn, pipeline, loop_, message }
    }

    /// Transport entry point: request in, reply out. The single OCP `match`
    /// over the request enum — a new capability adds a variant + one arm here
    /// (or none, if routed through an existing host) and nothing else.
    pub async fn dispatch(&self, req: HostRequest) -> HostReply {
        match req {
            HostRequest::Spawn(r) => self.spawn.spawn(r).await.into(),
            HostRequest::ListSubagents(_) => self.spawn.list_subagents().await.into(),
            HostRequest::CancelSubagent(r) => self.spawn.cancel(r).await.into(),
            HostRequest::WaitSubagent(r) => self.spawn.wait(r).await.into(),
            HostRequest::CheckSubagent(r) => self.spawn.check(r).await.into(),
            HostRequest::RunPipeline(r) => self.pipeline.run_pipeline(r).await.into(),
            HostRequest::RunLoop(r) => self.loop_.run_loop(r).await.into(),
            HostRequest::SendMessage(r) => self.message.send(r).await.into(),
        }
    }

    /// Typed in-process call used by tools (skips the serde round-trip).
    ///
    /// The error channel is round-tripped byte-losslessly: the envelope
    /// carries `ToolError::render()` output, and converting it back through
    /// [`ToolError::from_legacy`] re-renders the identical string, so a tool
    /// that `?`-propagates the error never double-prefixes or rewrites it.
    pub async fn call(&self, req: HostRequest) -> Result<serde_json::Value, ToolError> {
        match self.dispatch(req).await {
            HostReply::Ok { data } => Ok(data),
            HostReply::Error { error } => {
                let message = error
                    .strip_prefix("Error:")
                    .map(str::trim)
                    .unwrap_or(&error)
                    .to_string();
                Err(ToolError::from_legacy(&message))
            }
        }
    }
}

#[async_trait]
impl SpawnHost for HostDispatcher {
    async fn spawn(&self, req: SpawnRequest) -> Result<SpawnReply, ToolError> {
        self.spawn.spawn(req).await
    }
    async fn list_subagents(&self) -> Result<ListReply, ToolError> {
        self.spawn.list_subagents().await
    }
    async fn cancel(&self, req: CancelRequest) -> Result<CancelReply, ToolError> {
        self.spawn.cancel(req).await
    }
    async fn wait(&self, req: WaitRequest) -> Result<WaitReply, ToolError> {
        self.spawn.wait(req).await
    }
    async fn check(&self, req: CheckRequest) -> Result<CheckReply, ToolError> {
        self.spawn.check(req).await
    }
}

#[async_trait]
impl PipelineHost for HostDispatcher {
    async fn run_pipeline(&self, req: PipelineRequest) -> Result<PipelineReply, ToolError> {
        self.pipeline.run_pipeline(req).await
    }
}

#[async_trait]
impl LoopHost for HostDispatcher {
    async fn run_loop(&self, req: LoopRequest) -> Result<LoopReply, ToolError> {
        self.loop_.run_loop(req).await
    }
}

#[async_trait]
impl MessageHost for HostDispatcher {
    async fn send(&self, req: SendMessageRequest) -> Result<SendMessageReply, ToolError> {
        self.message.send(req).await
    }
}

/// One `From` impl — every capability method returns `Result<_, ToolError>`
/// and every reply envelope is produced here (DRY: no per-method envelope
/// construction). The single fallible-serialization site of the bridge
/// (research §3.6): an unserializable success payload degrades to `null`
/// rather than losing the error channel.
impl<T> From<Result<T, ToolError>> for HostReply
where
    T: Serialize,
{
    fn from(r: Result<T, ToolError>) -> Self {
        match r {
            Ok(v) => Self::Ok {
                data: serde_json::to_value(v).unwrap_or(serde_json::Value::Null),
            },
            Err(e) => Self::Error { error: e.render() },
        }
    }
}

// ---------------------------------------------------------------------------
// HostBridgeAdapter — transitional bridge over the legacy callback slots
// (doc §3.7 Step 1). Deleted in Step 3 once the closure bodies are fully
// ported into the production host (AgentHost in tool_wiring).
// ---------------------------------------------------------------------------

/// Implements the capability traits by invoking the *legacy* callback
/// closures, so the dispatcher can be driven end-to-end against the
/// pre-bridge wiring and proven byte-identical to `SpawnTool`/`MessageTool`.
pub struct HostBridgeAdapter {
    spawn: Option<SpawnCallback>,
    list: Option<ListCallback>,
    cancel: Option<CancelCallback>,
    wait: Option<WaitCallback>,
    check: Option<CheckCallback>,
    pipeline: Option<PipelineCallback>,
    loop_: Option<LoopCallback>,
    send: Option<SendCallback>,
}

impl HostBridgeAdapter {
    #[allow(clippy::too_many_arguments)] // 8 legacy callback slots, one per capability method — transitional type, deleted in Step 3
    pub fn new(
        spawn: Option<SpawnCallback>,
        list: Option<ListCallback>,
        cancel: Option<CancelCallback>,
        wait: Option<WaitCallback>,
        check: Option<CheckCallback>,
        pipeline: Option<PipelineCallback>,
        loop_: Option<LoopCallback>,
        send: Option<SendCallback>,
    ) -> Self {
        Self { spawn, list, cancel, wait, check, pipeline, loop_, send }
    }
}

#[async_trait]
impl SpawnHost for HostBridgeAdapter {
    async fn spawn(&self, req: SpawnRequest) -> Result<SpawnReply, ToolError> {
        let cb = self.spawn.clone().ok_or_else(|| ToolError::Execution {
            message: "Spawn callback not configured".to_string(),
        })?;
        let text = into_model_text(
            cb(req.task, req.label, req.profile, req.model, req.channel, req.chat_id, req.working_dir)
                .await,
        )?;
        Ok(SpawnReply { task_id: spawn_task_id(&text), text })
    }

    async fn list_subagents(&self) -> Result<ListReply, ToolError> {
        let cb = self.list.clone().ok_or_else(|| ToolError::Execution {
            message: "List callback not configured".to_string(),
        })?;
        Ok(ListReply { text: into_model_text(cb().await)? })
    }

    async fn cancel(&self, req: CancelRequest) -> Result<CancelReply, ToolError> {
        let cb = self.cancel.clone().ok_or_else(|| ToolError::Execution {
            message: "Cancel callback not configured".to_string(),
        })?;
        Ok(CancelReply { text: into_model_text(cb(req.task_id).await)? })
    }

    async fn wait(&self, req: WaitRequest) -> Result<WaitReply, ToolError> {
        let cb = self.wait.clone().ok_or_else(|| ToolError::Execution {
            message: "Wait callback not configured".to_string(),
        })?;
        Ok(WaitReply { text: into_model_text(cb(req.task_id, req.timeout_secs).await)? })
    }

    async fn check(&self, req: CheckRequest) -> Result<CheckReply, ToolError> {
        let cb = self.check.clone().ok_or_else(|| ToolError::Execution {
            message: "Check callback not configured".to_string(),
        })?;
        Ok(CheckReply { text: into_model_text(cb(req.task_id).await)? })
    }
}

#[async_trait]
impl PipelineHost for HostBridgeAdapter {
    async fn run_pipeline(&self, req: PipelineRequest) -> Result<PipelineReply, ToolError> {
        let cb = self.pipeline.clone().ok_or_else(|| ToolError::Execution {
            message: "Pipeline callback not configured".to_string(),
        })?;
        let steps_json = serde_json::to_string(&req.steps).unwrap_or_else(|_| "[]".to_string());
        Ok(PipelineReply { text: into_model_text(cb(steps_json, req.ahead_by_k).await)? })
    }
}

#[async_trait]
impl LoopHost for HostBridgeAdapter {
    async fn run_loop(&self, req: LoopRequest) -> Result<LoopReply, ToolError> {
        let cb = self.loop_.clone().ok_or_else(|| ToolError::Execution {
            message: "Loop callback not configured".to_string(),
        })?;
        Ok(LoopReply {
            text: into_model_text(
                cb(req.task, req.max_rounds, req.tools, req.stop_condition, req.model, req.working_dir)
                    .await,
            )?,
        })
    }
}

#[async_trait]
impl MessageHost for HostBridgeAdapter {
    async fn send(&self, req: SendMessageRequest) -> Result<SendMessageReply, ToolError> {
        let cb = self.send.clone().ok_or_else(|| ToolError::Execution {
            message: "Message sending not configured".to_string(),
        })?;
        let msg = OutboundMessage::new(&req.channel, &req.chat_id, &req.content);
        cb(msg).await.map_err(|e| ToolError::Execution {
            message: format!("Error sending message: {e}"),
        })?;
        Ok(SendMessageReply { text: format!("Message sent to {}:{}", req.channel, req.chat_id) })
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Funnel a legacy callback's `String` output into the typed channel: an
/// `Error:`-prefixed string becomes a typed [`ToolError`] via
/// [`ToolError::from_legacy`] (byte-identical classification); anything else
/// is the model-visible text of a success reply. Mirrors `spawn.rs`'s
/// `into_result`.
fn into_model_text(out: String) -> Result<String, ToolError> {
    out.strip_prefix("Error:")
        .map(|s| s.trim().to_string())
        .map_or(Ok(out), |err| Err(ToolError::from_legacy(&err)))
}

/// Best-effort structured task id from the spawn confirmation line
/// (`"Subagent '…' spawned (id: abc12345, …). …"`). Empty when the line shape
/// changes — `text` remains the authoritative model-visible output.
pub fn spawn_task_id(text: &str) -> String {
    text.split(" (id: ")
        .nth(1)
        .and_then(|rest| rest.split([',', ')']).next())
        .unwrap_or_default()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    use crate::agent::tools::base::Tool;
    use crate::agent::tools::{MessageTool, SpawnTool};
    use serde_json::json;

    // -----------------------------------------------------------------------
    // Serde wire round-trips
    // -----------------------------------------------------------------------

    fn spawn_request() -> SpawnRequest {
        SpawnRequest {
            task: "investigate".to_string(),
            label: Some("explore".to_string()),
            profile: None,
            model: Some("haiku".to_string()),
            working_dir: None,
            channel: "telegram".to_string(),
            chat_id: "42".to_string(),
        }
    }

    fn all_requests() -> Vec<HostRequest> {
        // SpawnRequest's channel/chat_id are `#[serde(skip)]`: the wire-visible
        // parts must round-trip exactly. The baked-origin case (non-empty
        // channel/chat_id) is covered by `spawn_request_skips_channel_and_chat_id_on_the_wire`.
        let mut spawn = spawn_request();
        spawn.channel.clear();
        spawn.chat_id.clear();
        vec![
            HostRequest::Spawn(spawn),
            HostRequest::ListSubagents(ListSubagentsRequest {}),
            HostRequest::CancelSubagent(CancelRequest {
                task_id: "abc123".to_string(),
            }),
            HostRequest::WaitSubagent(WaitRequest {
                task_id: "abc123".to_string(),
                timeout_secs: 30,
            }),
            HostRequest::CheckSubagent(CheckRequest {
                task_id: "abc123".to_string(),
            }),
            HostRequest::RunPipeline(PipelineRequest {
                steps: vec![json!({"prompt": "step one"})],
                ahead_by_k: 2,
            }),
            HostRequest::RunLoop(LoopRequest {
                task: "refine".to_string(),
                max_rounds: 3,
                tools: Some(vec!["read_file".to_string()]),
                stop_condition: None,
                model: None,
                working_dir: None,
            }),
            HostRequest::SendMessage(SendMessageRequest {
                channel: "telegram".to_string(),
                chat_id: "42".to_string(),
                content: "hello".to_string(),
            }),
        ]
    }

    #[test]
    fn spawn_request_skips_channel_and_chat_id_on_the_wire() {
        let wire = serde_json::to_string(&HostRequest::Spawn(spawn_request())).unwrap();
        assert!(wire.contains(r#""type":"spawn""#));
        assert!(wire.contains(r#""task":"investigate""#));
        // The baked-in origin context must never travel on the wire.
        assert!(!wire.contains("channel"));
        assert!(!wire.contains("chat_id"));

        let back: HostRequest = serde_json::from_str(&wire).unwrap();
        let HostRequest::Spawn(back) = back else {
            panic!("variant mismatch");
        };
        assert_eq!(back.task, "investigate");
        assert_eq!(back.label, Some("explore".to_string()));
        assert_eq!(back.model, Some("haiku".to_string()));
        // Skipped fields deserialize to their defaults, not the baked values.
        assert_eq!(back.channel, "");
        assert_eq!(back.chat_id, "");
    }

    #[test]
    fn all_request_variants_round_trip() {
        for req in all_requests() {
            let wire = serde_json::to_string(&req).unwrap();
            let back: HostRequest = serde_json::from_str(&wire).unwrap();
            assert_eq!(back, req, "round-trip failed for {wire}");
        }
    }

    #[test]
    fn type_tag_is_the_route_not_a_payload_key() {
        // A payload cannot carry its own `type`: the tag IS the discriminator,
        // so no data field can spoof the dispatch target (doc §3.1 property 6).
        let wire = r#"{"type": "list_subagents"}"#;
        let req: HostRequest = serde_json::from_str(wire).unwrap();
        assert!(matches!(req, HostRequest::ListSubagents(_)));
    }

    #[test]
    fn host_reply_ok_flattens_payload_onto_status_envelope() {
        let reply = HostReply::Ok {
            data: json!({"task_id": "abc12345", "text": "Subagent 'x' spawned"}),
        };
        let wire = serde_json::to_string(&reply).unwrap();
        // The payload is flattened next to the status tag (no nested "data").
        assert!(wire.contains(r#""status":"ok""#));
        assert!(wire.contains(r#""task_id":"abc12345""#));
        let back: HostReply = serde_json::from_str(&wire).unwrap();
        assert_eq!(back, reply);
    }

    #[test]
    fn host_reply_error_round_trip() {
        let reply = HostReply::Error {
            error: "Error: Command timed out after 30s".to_string(),
        };
        let wire = serde_json::to_string(&reply).unwrap();
        assert!(wire.contains(r#""status":"error""#));
        let back: HostReply = serde_json::from_str(&wire).unwrap();
        assert_eq!(back, reply);
    }

    // -----------------------------------------------------------------------
    // Typed dispatch (OCP seam) over mocks
    // -----------------------------------------------------------------------

    struct MockSpawn;
    struct MockPipeline;
    struct MockLoop;
    struct MockMessage;

    #[async_trait]
    impl SpawnHost for MockSpawn {
        async fn spawn(&self, req: SpawnRequest) -> Result<SpawnReply, ToolError> {
            Ok(SpawnReply {
                task_id: "abc12345".to_string(),
                text: format!("spawned {}", req.task),
            })
        }
        async fn list_subagents(&self) -> Result<ListReply, ToolError> {
            Ok(ListReply { text: "list".to_string() })
        }
        async fn cancel(&self, _req: CancelRequest) -> Result<CancelReply, ToolError> {
            Ok(CancelReply { text: "cancelled".to_string() })
        }
        async fn wait(&self, _req: WaitRequest) -> Result<WaitReply, ToolError> {
            Ok(WaitReply { text: "waited".to_string() })
        }
        async fn check(&self, _req: CheckRequest) -> Result<CheckReply, ToolError> {
            Ok(CheckReply { text: "checked".to_string() })
        }
    }

    #[async_trait]
    impl PipelineHost for MockPipeline {
        async fn run_pipeline(&self, _req: PipelineRequest) -> Result<PipelineReply, ToolError> {
            Ok(PipelineReply { text: "pipeline done".to_string() })
        }
    }

    #[async_trait]
    impl LoopHost for MockLoop {
        async fn run_loop(&self, _req: LoopRequest) -> Result<LoopReply, ToolError> {
            Ok(LoopReply { text: "loop done".to_string() })
        }
    }

    #[async_trait]
    impl MessageHost for MockMessage {
        async fn send(&self, req: SendMessageRequest) -> Result<SendMessageReply, ToolError> {
            Ok(SendMessageReply {
                text: format!("sent to {}", req.channel),
            })
        }
    }

    fn mock_dispatcher() -> HostDispatcher {
        HostDispatcher::new(
            Arc::new(MockSpawn),
            Arc::new(MockPipeline),
            Arc::new(MockLoop),
            Arc::new(MockMessage),
        )
    }

    #[tokio::test]
    async fn dispatch_routes_each_request_to_its_capability() {
        let d = mock_dispatcher();
        for req in all_requests() {
            let reply = d.dispatch(req.clone()).await;
            assert!(
                matches!(reply, HostReply::Ok { .. }),
                "expected Ok envelope for {req:?}, got {reply:?}"
            );
        }
    }

    #[tokio::test]
    async fn call_returns_flattened_data_for_ok() {
        let d = mock_dispatcher();
        let data = d
            .call(HostRequest::Spawn(SpawnRequest {
                task: "t".to_string(),
                ..spawn_request()
            }))
            .await
            .unwrap();
        assert_eq!(data["text"], "spawned t");
        assert_eq!(data["task_id"], "abc12345");
    }

    // -----------------------------------------------------------------------
    // Structural error classification + rendered error envelope
    // -----------------------------------------------------------------------

    struct FailingSpawn;

    #[async_trait]
    impl SpawnHost for FailingSpawn {
        async fn spawn(&self, _req: SpawnRequest) -> Result<SpawnReply, ToolError> {
            Err(ToolError::Timeout(30))
        }
        async fn list_subagents(&self) -> Result<ListReply, ToolError> {
            Err(ToolError::Execution {
                message: "List callback not configured".to_string(),
            })
        }
        async fn cancel(&self, _req: CancelRequest) -> Result<CancelReply, ToolError> {
            Err(ToolError::PermissionDenied("Permission denied: no".to_string()))
        }
        async fn wait(&self, _req: WaitRequest) -> Result<WaitReply, ToolError> {
            Err(ToolError::MissingArg {
                param: "task_id".to_string(),
                example: r#"spawn({"action":"wait","task_id":"x"})"#.to_string(),
            })
        }
        async fn check(&self, _req: CheckRequest) -> Result<CheckReply, ToolError> {
            Err(ToolError::Network("Network error: dns".to_string()))
        }
    }

    #[tokio::test]
    async fn typed_error_variants_render_into_the_error_envelope() {
        let d = HostDispatcher::new(
            Arc::new(FailingSpawn),
            Arc::new(MockPipeline),
            Arc::new(MockLoop),
            Arc::new(MockMessage),
        );
        let cases = [
            (
                HostRequest::Spawn(spawn_request()),
                "Error: Command timed out after 30s",
            ),
            (
                HostRequest::ListSubagents(ListSubagentsRequest {}),
                "Error: List callback not configured",
            ),
            (
                HostRequest::CancelSubagent(CancelRequest {
                    task_id: "x".to_string(),
                }),
                "Error: Permission denied: no",
            ),
            (
                HostRequest::WaitSubagent(WaitRequest {
                    task_id: "x".to_string(),
                    timeout_secs: 1,
                }),
                "Error: 'task_id' parameter is required; call as spawn({\"action\":\"wait\",\"task_id\":\"x\"})",
            ),
            (
                HostRequest::CheckSubagent(CheckRequest {
                    task_id: "x".to_string(),
                }),
                "Error: Network error: dns",
            ),
        ];
        for (req, expected) in cases {
            let reply = d.dispatch(req).await;
            match reply {
                HostReply::Error { error } => assert_eq!(error, expected),
                other => panic!("expected Error envelope, got {other:?}"),
            }
        }
    }

    #[tokio::test]
    async fn call_converts_error_envelope_into_typed_execution_error() {
        let d = HostDispatcher::new(
            Arc::new(FailingSpawn),
            Arc::new(MockPipeline),
            Arc::new(MockLoop),
            Arc::new(MockMessage),
        );
        let err = d.call(HostRequest::Spawn(spawn_request())).await.unwrap_err();
        assert_eq!(err.render(), "Error: Command timed out after 30s");
    }

    // -----------------------------------------------------------------------
    // Dispatcher over the legacy adapter — byte-stability proof
    // -----------------------------------------------------------------------

    fn legacy_callbacks() -> (
        SpawnCallback,
        ListCallback,
        CancelCallback,
        WaitCallback,
        CheckCallback,
        PipelineCallback,
        LoopCallback,
        SendCallback,
    ) {
        let spawn: SpawnCallback = Arc::new(
            |_task, label, _agent, model, _ch, _cid, _wd| {
                Box::pin(async move {
                    format!(
                        "Subagent '{}' spawned (id: abc12345, model: {}). It will announce results when done.",
                        label.unwrap_or_default(),
                        model.unwrap_or_else(|| "default".to_string())
                    )
                })
            },
        );
        let list: ListCallback =
            Arc::new(|| Box::pin(async { "No subagents currently running.".to_string() }));
        let cancel: CancelCallback =
            Arc::new(|id| Box::pin(async move { format!("Subagent '{id}' cancelled.") }));
        let wait: WaitCallback = Arc::new(|id, timeout| {
            Box::pin(async move { format!("waited {id} for {timeout}s") })
        });
        let check: CheckCallback =
            Arc::new(|id| Box::pin(async move { format!("checked {id}") }));
        let pipeline: PipelineCallback = Arc::new(|steps, ahead| {
            Box::pin(async move { format!("pipeline over {steps} ahead {ahead}") })
        });
        let loop_: LoopCallback = Arc::new(|task, rounds, _t, _s, _m, _w| {
            Box::pin(async move { format!("loop {task} r{rounds}") })
        });
        let send: SendCallback = Arc::new(|_msg| Box::pin(async { Ok(()) }));
        (spawn, list, cancel, wait, check, pipeline, loop_, send)
    }

    fn adapter_dispatcher(
        spawn: SpawnCallback,
        list: ListCallback,
        cancel: CancelCallback,
        wait: WaitCallback,
        check: CheckCallback,
        pipeline: PipelineCallback,
        loop_: LoopCallback,
        send: SendCallback,
    ) -> HostDispatcher {
        let adapter = Arc::new(HostBridgeAdapter::new(
            Some(spawn),
            Some(list),
            Some(cancel),
            Some(wait),
            Some(check),
            Some(pipeline),
            Some(loop_),
            Some(send),
        ));
        HostDispatcher::new(
            adapter.clone(),
            adapter.clone(),
            adapter.clone(),
            adapter.clone(),
        )
    }

    #[tokio::test]
    async fn dispatcher_over_legacy_callbacks_matches_spawn_tool_output() {
        let (spawn, list, cancel, wait, check, pipeline, loop_, send) = legacy_callbacks();
        let dispatcher = adapter_dispatcher(
            spawn.clone(),
            list.clone(),
            cancel.clone(),
            wait.clone(),
            check.clone(),
            pipeline.clone(),
            loop_.clone(),
            send.clone(),
        );

        // Legacy surface: the same callbacks wired into SpawnTool.
        let tool = SpawnTool::new();
        tool.set_callback(spawn).await;
        tool.set_list_callback(list).await;
        tool.set_cancel_callback(cancel).await;
        tool.set_wait_callback(wait).await;
        tool.set_check_callback(check).await;
        tool.set_pipeline_callback(pipeline).await;
        tool.set_loop_callback(loop_).await;
        tool.set_context("telegram", "42").await;

        // spawn
        let mut p = HashMap::new();
        p.insert("task".to_string(), json!("do x"));
        p.insert("label".to_string(), json!("explore"));
        p.insert("model".to_string(), json!("haiku"));
        let legacy = tool.execute(p).await;
        let reply = dispatcher
            .dispatch(HostRequest::Spawn(SpawnRequest {
                task: "do x".to_string(),
                label: Some("explore".to_string()),
                profile: None,
                model: Some("haiku".to_string()),
                working_dir: None,
                channel: "telegram".to_string(),
                chat_id: "42".to_string(),
            }))
            .await;
        match reply {
            HostReply::Ok { data } => {
                assert_eq!(data["text"].as_str().unwrap(), legacy, "spawn text diverged");
                assert_eq!(data["task_id"].as_str().unwrap(), "abc12345");
            }
            other => panic!("expected Ok, got {other:?}"),
        }

        // list
        let mut p = HashMap::new();
        p.insert("action".to_string(), json!("list"));
        let legacy = tool.execute(p).await;
        let reply = dispatcher.dispatch(HostRequest::ListSubagents(ListSubagentsRequest {})).await;
        assert_eq!(reply_text(&reply), legacy, "list text diverged");

        // cancel
        let mut p = HashMap::new();
        p.insert("action".to_string(), json!("cancel"));
        p.insert("task_id".to_string(), json!("abc12345"));
        let legacy = tool.execute(p).await;
        let reply = dispatcher
            .dispatch(HostRequest::CancelSubagent(CancelRequest {
                task_id: "abc12345".to_string(),
            }))
            .await;
        assert_eq!(reply_text(&reply), legacy, "cancel text diverged");

        // wait
        let mut p = HashMap::new();
        p.insert("action".to_string(), json!("wait"));
        p.insert("task_id".to_string(), json!("abc12345"));
        p.insert("timeout".to_string(), json!(10));
        let legacy = tool.execute(p).await;
        let reply = dispatcher
            .dispatch(HostRequest::WaitSubagent(WaitRequest {
                task_id: "abc12345".to_string(),
                timeout_secs: 10,
            }))
            .await;
        assert_eq!(reply_text(&reply), legacy, "wait text diverged");

        // check
        let mut p = HashMap::new();
        p.insert("action".to_string(), json!("check"));
        p.insert("task_id".to_string(), json!("abc12345"));
        let legacy = tool.execute(p).await;
        let reply = dispatcher
            .dispatch(HostRequest::CheckSubagent(CheckRequest {
                task_id: "abc12345".to_string(),
            }))
            .await;
        assert_eq!(reply_text(&reply), legacy, "check text diverged");

        // pipeline
        let mut p = HashMap::new();
        p.insert("action".to_string(), json!("pipeline"));
        p.insert("steps".to_string(), json!([{"prompt": "one"}, {"prompt": "two"}]));
        p.insert("ahead_by_k".to_string(), json!(2));
        let legacy = tool.execute(p).await;
        let reply = dispatcher
            .dispatch(HostRequest::RunPipeline(PipelineRequest {
                steps: vec![json!({"prompt": "one"}), json!({"prompt": "two"})],
                ahead_by_k: 2,
            }))
            .await;
        assert_eq!(reply_text(&reply), legacy, "pipeline text diverged");

        // loop
        let mut p = HashMap::new();
        p.insert("action".to_string(), json!("loop"));
        p.insert("task".to_string(), json!("refine"));
        p.insert("max_rounds".to_string(), json!(3));
        let legacy = tool.execute(p).await;
        let reply = dispatcher
            .dispatch(HostRequest::RunLoop(LoopRequest {
                task: "refine".to_string(),
                max_rounds: 3,
                tools: None,
                stop_condition: None,
                model: None,
                working_dir: None,
            }))
            .await;
        assert_eq!(reply_text(&reply), legacy, "loop text diverged");
    }

    #[tokio::test]
    async fn legacy_error_strings_classify_identically_through_the_bridge() {
        let failing: SpawnCallback =
            Arc::new(|_task, _l, _a, _m, _ch, _cid, _wd| {
                Box::pin(async { "Error: Connection refused".to_string() })
            });
        let adapter = Arc::new(HostBridgeAdapter::new(
            Some(failing),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ));
        let dispatcher = HostDispatcher::new(
            adapter.clone(),
            adapter.clone(),
            adapter.clone(),
            adapter.clone(),
        );
        let reply = dispatcher
            .dispatch(HostRequest::Spawn(SpawnRequest {
                task: "t".to_string(),
                label: None,
                profile: None,
                model: None,
                working_dir: None,
                channel: "cli".to_string(),
                chat_id: "direct".to_string(),
            }))
            .await;
        match reply {
            HostReply::Error { error } => assert_eq!(error, "Error: Connection refused"),
            other => panic!("expected Error envelope, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn adapter_missing_callback_reports_legacy_error_text() {
        let adapter = Arc::new(HostBridgeAdapter::new(
            None, None, None, None, None, None, None, None,
        ));
        let reply = adapter.list_subagents().await.unwrap_err();
        assert_eq!(reply.render(), "Error: List callback not configured");
    }

    #[tokio::test]
    async fn message_path_matches_legacy_message_tool_output() {
        let (_, _, _, _, _, _, _, send) = legacy_callbacks();
        let adapter = Arc::new(HostBridgeAdapter::new(
            None, None, None, None, None, None, None, Some(send.clone()),
        ));
        let dispatcher = HostDispatcher::new(
            adapter.clone(),
            adapter.clone(),
            adapter.clone(),
            adapter.clone(),
        );

        let legacy_tool = MessageTool::new(Some(send), "telegram", "42");
        let mut p = HashMap::new();
        p.insert("content".to_string(), json!("hi"));
        let legacy = legacy_tool.execute(p).await;

        let reply = dispatcher
            .dispatch(HostRequest::SendMessage(SendMessageRequest {
                channel: "telegram".to_string(),
                chat_id: "42".to_string(),
                content: "hi".to_string(),
            }))
            .await;
        assert_eq!(reply_text(&reply), legacy, "message text diverged");
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn reply_text(reply: &HostReply) -> String {
        match reply {
            HostReply::Ok { data } => data["text"]
                .as_str()
                .unwrap_or_else(|| panic!("no text in {data}"))
                .to_string(),
            HostReply::Error { error } => panic!("unexpected error envelope: {error}"),
        }
    }

    #[test]
    fn spawn_task_id_extraction() {
        assert_eq!(
            spawn_task_id("Subagent 'explore: investigate' spawned (id: abc12345, agent: explore, model: haiku). It will announce results when done."),
            "abc12345"
        );
        assert_eq!(spawn_task_id("no id here"), "");
    }
}
