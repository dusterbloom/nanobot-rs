//! Shared streaming-turn engine for the REPL and TUI frontends.
//!
//! Both frontends drive the same agent streaming surface (a text-delta
//! channel, an optional tool-event channel, a [`CancellationToken`], and —
//! for the TUI — the agent's [`JoinHandle`]). Before this module each
//! frontend hand-rolled the same `tokio::select!` lifecycle: biased
//! delta-first ordering, "drain buffered deltas before showing a tool row",
//! post-completion buffer drains, and cancel-discards-everything semantics.
//!
//! [`TurnStream`] owns that lifecycle and exposes it as a sequence of
//! [`TurnEvent`]s. Rendering stays per-frontend; deltas are yielded as raw
//! strings (control markers included — each frontend keeps its own marker
//! parser, see `repl::parse_control_marker` and `tui_app::App::on_delta`).
//!
//! # Completion policies ([`Completion`])
//!
//! * [`Completion::AgentHandle`] — TUI mode. The agent turn runs on its own
//!   task; the stream finishes when the handle resolves **and** every
//!   buffered delta/tool event has been yielded. `Finished` carries the
//!   join result (the final response text).
//! * [`Completion::ChannelClose`] — REPL print-task mode. The agent future
//!   is awaited elsewhere (the REPL awaits `process_direct_streaming`
//!   inline), so the stream finishes when **both** channels have closed
//!   (an absent tool channel counts as closed). `Finished` carries an
//!   empty string.
//!
//! # Ordering rule (structural, not best-effort)
//!
//! A tool event is never delivered while a delta is buffered: every delta
//! buffered at the moment a tool event would be yielded is yielded first.
//! This is the rule both frontends previously enforced with ad-hoc
//! `drain_pending_deltas` helpers / biased select arms.

use tokio::sync::mpsc::error::TryRecvError;
use tokio::sync::mpsc::UnboundedReceiver;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::agent::audit::ToolEvent;

/// One observable step of a streaming turn.
#[derive(Debug)]
pub(crate) enum TurnEvent {
    /// Raw text delta (control markers included).
    Delta(String),
    /// Tool lifecycle event (start / progress / end).
    Tool(ToolEvent),
    /// The turn is over. Payload is the final response text
    /// ([`Completion::AgentHandle`]) or empty ([`Completion::ChannelClose`]
    /// and every cancelled turn).
    Finished(String),
}

/// When a turn counts as finished. See the module docs for the two modes.
pub(crate) enum Completion {
    /// Finish when the agent's spawned task resolves (TUI).
    AgentHandle(JoinHandle<String>),
    /// Finish when both channels close (REPL print task).
    ChannelClose,
}

/// Where the agent JoinHandle is in its lifecycle.
enum HandleState {
    /// Still running; polled in the streaming select.
    Pending(JoinHandle<String>),
    /// Already joined (result moved into [`Phase::Draining`]) — must not be
    /// polled again.
    Joined,
    /// No handle at all ([`Completion::ChannelClose`]).
    Absent,
}

/// Lifecycle state of the stream.
enum Phase {
    /// Live turn: yielding events as they arrive.
    Streaming,
    /// Agent handle resolved; yielding the buffered backlog, then `Finished`.
    Draining { response: String },
    /// [`TurnStream::cancel`] was called: discard everything, yield
    /// `Finished("")` once the agent task has wound down.
    Cancelled,
    /// `Finished` already yielded; further `next()` calls repeat
    /// `Finished("")`.
    Done,
}

/// Owns one streaming turn's channels, cancellation token, and (optionally)
/// the agent task handle, and yields [`TurnEvent`]s in the shared order.
pub(crate) struct TurnStream {
    /// `None` once the channel has closed.
    delta_rx: Option<UnboundedReceiver<String>>,
    /// `None` when absent (REPL without provenance) or once closed.
    tool_rx: Option<UnboundedReceiver<ToolEvent>>,
    /// Tool event received from the live select but not yet yielded because
    /// deltas may have raced in behind it.
    pending_tool: Option<ToolEvent>,
    handle: HandleState,
    cancel: Option<CancellationToken>,
    phase: Phase,
}

impl TurnStream {
    pub(crate) fn new(
        delta_rx: UnboundedReceiver<String>,
        tool_rx: Option<UnboundedReceiver<ToolEvent>>,
        completion: Completion,
        cancel: Option<CancellationToken>,
    ) -> Self {
        let handle = match completion {
            Completion::AgentHandle(h) => HandleState::Pending(h),
            Completion::ChannelClose => HandleState::Absent,
        };
        Self {
            delta_rx: Some(delta_rx),
            tool_rx,
            pending_tool: None,
            handle,
            cancel,
            phase: Phase::Streaming,
        }
    }

    /// Cancel the turn: fires the token (if any) and switches the stream to
    /// discard mode — every buffered or late event is dropped and the next
    /// `next()` call resolves to `Finished("")` once the agent task (if
    /// owned) has wound down. Under [`Completion::ChannelClose`] there is no
    /// task to wait for, so `Finished("")` comes immediately.
    pub(crate) fn cancel(&mut self) {
        if let Some(token) = &self.cancel {
            token.cancel();
        }
        self.phase = Phase::Cancelled;
    }

    /// Yield the next event of the turn. Cancel-safe: dropping the returned
    /// future (e.g. losing a `tokio::select!` race) never loses an event —
    /// every received value is either returned within the same poll or
    /// parked on `self` before the next await point.
    pub(crate) async fn next(&mut self) -> TurnEvent {
        loop {
            match self.phase {
                Phase::Done => return TurnEvent::Finished(String::new()),
                Phase::Cancelled => return self.finish_cancelled().await,
                Phase::Draining { .. } => return self.next_draining(),
                Phase::Streaming => {
                    let Some(event) = self.streaming_step().await else {
                        // Internal state advanced (handle resolved / channel
                        // closed) without producing an event — re-dispatch.
                        continue;
                    };
                    return event;
                }
            }
        }
    }

    /// One live-turn step: buffered backlog first, then the biased select
    /// over agent completion / deltas / tool events. `None` means state
    /// advanced without an event and the caller should re-dispatch.
    async fn streaming_step(&mut self) -> Option<TurnEvent> {
        if let Some(event) = self.buffered_event() {
            return Some(event);
        }
        if self.channels_exhausted() {
            self.phase = Phase::Done;
            return Some(TurnEvent::Finished(String::new()));
        }
        tokio::select! {
            biased;
            response = join_agent(&mut self.handle) => {
                self.phase = Phase::Draining { response };
                None
            }
            delta = recv_opt(&mut self.delta_rx) => match delta {
                Some(d) => Some(TurnEvent::Delta(d)),
                None => {
                    self.delta_rx = None;
                    None
                }
            },
            event = recv_opt(&mut self.tool_rx) => match event {
                // Park it: a delta may have raced in behind this tool event.
                // `buffered_event` re-checks deltas before releasing it.
                Some(e) => {
                    self.pending_tool = Some(e);
                    None
                }
                None => {
                    self.tool_rx = None;
                    None
                }
            },
        }
    }

    /// Post-completion backlog: buffered events (deltas first), then
    /// `Finished` with the agent's response. Mirrors the drain the TUI did
    /// after its select loop broke.
    fn next_draining(&mut self) -> TurnEvent {
        if let Some(event) = self.buffered_event() {
            return event;
        }
        let Phase::Draining { response } = std::mem::replace(&mut self.phase, Phase::Done) else {
            unreachable!("next_draining is only dispatched from Phase::Draining");
        };
        TurnEvent::Finished(response)
    }

    /// Cancelled wind-down: wait for the agent task we own (the TUI's old
    /// loop kept selecting until the join arm fired), then discard whatever
    /// is buffered and finish empty.
    async fn finish_cancelled(&mut self) -> TurnEvent {
        if matches!(self.handle, HandleState::Pending(_)) {
            let _ = join_agent(&mut self.handle).await;
        }
        self.discard_buffers();
        self.phase = Phase::Done;
        TurnEvent::Finished(String::new())
    }

    /// One buffered event, deltas strictly first (the ordering rule from the
    /// module docs). `None` when nothing is buffered right now.
    fn buffered_event(&mut self) -> Option<TurnEvent> {
        if let Some(d) = try_recv_opt(&mut self.delta_rx) {
            return Some(TurnEvent::Delta(d));
        }
        if let Some(e) = self.pending_tool.take() {
            return Some(TurnEvent::Tool(e));
        }
        try_recv_opt(&mut self.tool_rx).map(TurnEvent::Tool)
    }

    /// True when there is no agent handle to wait for and both channels have
    /// closed — the [`Completion::ChannelClose`] finish condition.
    fn channels_exhausted(&self) -> bool {
        matches!(self.handle, HandleState::Absent)
            && self.delta_rx.is_none()
            && self.tool_rx.is_none()
    }

    fn discard_buffers(&mut self) {
        self.pending_tool = None;
        while try_recv_opt(&mut self.delta_rx).is_some() {}
        while try_recv_opt(&mut self.tool_rx).is_some() {}
    }
}

/// Poll the agent task if we own one; pend forever otherwise (keeps the
/// select arm structurally present without ever firing). On join the slot
/// flips to [`HandleState::Joined`] so the handle is never polled again.
async fn join_agent(handle: &mut HandleState) -> String {
    let HandleState::Pending(h) = handle else {
        return std::future::pending().await;
    };
    let response = h.await.unwrap_or_default();
    *handle = HandleState::Joined;
    response
}

/// `recv` through the `Option` wrapper; pends forever on an absent channel.
async fn recv_opt<T>(rx: &mut Option<UnboundedReceiver<T>>) -> Option<T> {
    match rx.as_mut() {
        Some(receiver) => receiver.recv().await,
        None => std::future::pending().await,
    }
}

/// `try_recv` through the `Option` wrapper; clears the slot on disconnect so
/// a closed channel reads as closed everywhere.
fn try_recv_opt<T>(rx: &mut Option<UnboundedReceiver<T>>) -> Option<T> {
    let receiver = rx.as_mut()?;
    match receiver.try_recv() {
        Ok(value) => Some(value),
        Err(TryRecvError::Empty) => None,
        Err(TryRecvError::Disconnected) => {
            *rx = None;
            None
        }
    }
}

// ============================================================================
// Control-marker wire protocol
// ============================================================================
//
// Telemetry smuggled through the text-delta channel as `\x00`-prefixed
// strings (never rendered). This module owns BOTH directions: the agent loop
// emits via [`ControlMarker::encode`], the frontends decode via
// [`parse_control_marker`]. Hand-written `format!("\x00...")` at emit sites
// is what let the two ends drift — the round-trip test below makes the
// protocol correct by construction.

/// Control markers carried on the delta channel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ControlMarker {
    RetractReply,
    FinishReason(String),
    Tokens(u64),
    PromptTokens(u64),
    /// Real decode time (milliseconds) for the just-finished LLM call, measured
    /// agent-side as `call_wall_time − ttft`. Renderers sum these to report a
    /// true decode tok/s that excludes tool-execution and re-prefill time.
    DecodeMs(u64),
    PrefillEstimate(u64),
    PrefillProgress {
        processed: u64,
        total: u64,
    },
    BackendActivity {
        phase: BackendActivity,
        idle_ms: u64,
    },
    CacheStatus(CacheStatus),
}

/// Agent-side backend progress state for a live LLM call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BackendActivity {
    WaitingForHeaders,
    Prefill,
    AwaitingToolPayload,
    ToolPayload,
    Decoding,
}

impl BackendActivity {
    fn as_wire(self) -> &'static str {
        match self {
            BackendActivity::WaitingForHeaders => "waiting_headers",
            BackendActivity::Prefill => "prefill",
            BackendActivity::AwaitingToolPayload => "awaiting_tool_payload",
            BackendActivity::ToolPayload => "tool_payload",
            BackendActivity::Decoding => "decoding",
        }
    }

    fn from_wire(raw: &str) -> Option<Self> {
        match raw {
            "waiting_headers" => Some(BackendActivity::WaitingForHeaders),
            "prefill" => Some(BackendActivity::Prefill),
            "awaiting_tool_payload" => Some(BackendActivity::AwaitingToolPayload),
            "tool_payload" => Some(BackendActivity::ToolPayload),
            "decoding" => Some(BackendActivity::Decoding),
            _ => None,
        }
    }
}

/// Prompt-cache relationship between this LLM call and the previous one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CacheStatus {
    First {
        messages: usize,
    },
    AppendOnly {
        added: usize,
        messages: usize,
    },
    Diverged {
        at: usize,
        prev: usize,
        messages: usize,
    },
    Reset {
        reason: CacheResetReason,
    },
}

/// Why the agent knowingly invalidated the prompt-cache prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CacheResetReason {
    Trim,
    EmergencyTrim,
    LcmCheckpoint,
    StalledProviderRequest,
}

impl CacheResetReason {
    fn as_wire(self) -> &'static str {
        match self {
            CacheResetReason::Trim => "trim",
            CacheResetReason::EmergencyTrim => "emergency_trim",
            CacheResetReason::LcmCheckpoint => "lcm_checkpoint",
            CacheResetReason::StalledProviderRequest => "stalled_provider_request",
        }
    }
}

impl ControlMarker {
    /// Render this marker in the wire syntax [`parse_control_marker`] reads.
    /// The round-trip test pins `parse(encode(m)) == m` for every variant.
    pub(crate) fn encode(&self) -> String {
        match self {
            ControlMarker::RetractReply => "\x00retract_reply".to_string(),
            ControlMarker::FinishReason(fr) => format!("\x00finish_reason:{fr}"),
            ControlMarker::Tokens(n) => format!("\x00tokens:{n}"),
            ControlMarker::PromptTokens(n) => format!("\x00prompt_tokens:{n}"),
            ControlMarker::DecodeMs(ms) => format!("\x00decode_ms:{ms}"),
            ControlMarker::PrefillEstimate(n) => format!("\x00prefill_estimate:{n}"),
            ControlMarker::PrefillProgress { processed, total } => {
                format!("\x00prefill:{processed}/{total}")
            }
            ControlMarker::BackendActivity { phase, idle_ms } => {
                format!("\x00backend:{}:{idle_ms}", phase.as_wire())
            }
            ControlMarker::CacheStatus(status) => match status {
                CacheStatus::First { messages } => format!("\x00cache:first:{messages}"),
                CacheStatus::AppendOnly { added, messages } => {
                    format!("\x00cache:append:{added}:{messages}")
                }
                CacheStatus::Diverged { at, prev, messages } => {
                    format!("\x00cache:diverged:{at}:{prev}:{messages}")
                }
                CacheStatus::Reset { reason } => {
                    format!("\x00cache:reset:{}", reason.as_wire())
                }
            },
        }
    }
}

/// Parse a delta-channel control marker. `None` means renderable text.
pub(crate) fn parse_control_marker(d: &str) -> Option<ControlMarker> {
    let rest = d.strip_prefix('\x00')?;
    if rest == "retract_reply" {
        return Some(ControlMarker::RetractReply);
    }
    if let Some(fr) = rest.strip_prefix("finish_reason:") {
        return Some(ControlMarker::FinishReason(fr.to_string()));
    }
    if let Some(tok) = rest.strip_prefix("tokens:") {
        return tok.parse().ok().map(ControlMarker::Tokens);
    }
    if let Some(tok) = rest.strip_prefix("prompt_tokens:") {
        return tok.parse().ok().map(ControlMarker::PromptTokens);
    }
    if let Some(ms) = rest.strip_prefix("decode_ms:") {
        return ms.parse().ok().map(ControlMarker::DecodeMs);
    }
    if let Some(tok) = rest.strip_prefix("prefill_estimate:") {
        return tok.parse().ok().map(ControlMarker::PrefillEstimate);
    }
    if let Some(pp) = rest.strip_prefix("prefill:") {
        let (p, t) = pp.split_once('/')?;
        return Some(ControlMarker::PrefillProgress {
            processed: p.parse().ok()?,
            total: t.parse().ok()?,
        });
    }
    if let Some(backend) = rest.strip_prefix("backend:") {
        let (phase, idle_ms) = backend.split_once(':')?;
        return Some(ControlMarker::BackendActivity {
            phase: BackendActivity::from_wire(phase)?,
            idle_ms: idle_ms.parse().ok()?,
        });
    }
    if let Some(cache) = rest.strip_prefix("cache:") {
        let mut parts = cache.split(':');
        return match parts.next()? {
            "first" => Some(ControlMarker::CacheStatus(CacheStatus::First {
                messages: parts.next()?.parse().ok()?,
            })),
            "append" => Some(ControlMarker::CacheStatus(CacheStatus::AppendOnly {
                added: parts.next()?.parse().ok()?,
                messages: parts.next()?.parse().ok()?,
            })),
            "diverged" => Some(ControlMarker::CacheStatus(CacheStatus::Diverged {
                at: parts.next()?.parse().ok()?,
                prev: parts.next()?.parse().ok()?,
                messages: parts.next()?.parse().ok()?,
            })),
            "reset" => {
                let reason = match parts.next()? {
                    "trim" => CacheResetReason::Trim,
                    "emergency_trim" => CacheResetReason::EmergencyTrim,
                    "lcm_checkpoint" => CacheResetReason::LcmCheckpoint,
                    "stalled_provider_request" => CacheResetReason::StalledProviderRequest,
                    _ => return None,
                };
                Some(ControlMarker::CacheStatus(CacheStatus::Reset { reason }))
            }
            _ => None,
        };
    }
    None
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tokio::sync::mpsc::unbounded_channel;

    fn tool_event(id: &str) -> ToolEvent {
        ToolEvent::CallStart {
            tool_name: "exec".into(),
            tool_call_id: id.into(),
            arguments_preview: "ls".into(),
        }
    }

    /// `next()` bounded by a short timeout so a buggy engine hangs the test
    /// with a clear message instead of stalling the suite.
    async fn ev(stream: &mut TurnStream) -> TurnEvent {
        tokio::time::timeout(Duration::from_millis(500), stream.next())
            .await
            .expect("TurnStream::next() stalled")
    }

    /// A handle that stays pending until `tx` is dropped or fired, then
    /// resolves to `result`.
    fn gated_handle(result: &str) -> (JoinHandle<String>, tokio::sync::oneshot::Sender<()>) {
        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        let result = result.to_string();
        let handle = tokio::spawn(async move {
            let _ = rx.await;
            result
        });
        (handle, tx)
    }

    /// The protocol is correct by construction: every variant must survive
    /// encode → parse unchanged. Adding a marker without wiring both ends
    /// fails here, not in production drift.
    #[test]
    fn control_marker_round_trips_every_variant() {
        let variants = vec![
            ControlMarker::RetractReply,
            ControlMarker::FinishReason("stop".into()),
            ControlMarker::Tokens(42),
            ControlMarker::PromptTokens(30_000),
            ControlMarker::DecodeMs(5_120),
            ControlMarker::PrefillEstimate(1_000),
            ControlMarker::PrefillProgress {
                processed: 1_200,
                total: 2_400,
            },
            ControlMarker::BackendActivity {
                phase: BackendActivity::WaitingForHeaders,
                idle_ms: 8_000,
            },
            ControlMarker::BackendActivity {
                phase: BackendActivity::AwaitingToolPayload,
                idle_ms: 2_500,
            },
            ControlMarker::BackendActivity {
                phase: BackendActivity::ToolPayload,
                idle_ms: 500,
            },
            ControlMarker::CacheStatus(CacheStatus::First { messages: 3 }),
            ControlMarker::CacheStatus(CacheStatus::AppendOnly {
                added: 2,
                messages: 9,
            }),
            ControlMarker::CacheStatus(CacheStatus::Diverged {
                at: 2,
                prev: 140,
                messages: 209,
            }),
            ControlMarker::CacheStatus(CacheStatus::Reset {
                reason: CacheResetReason::Trim,
            }),
            ControlMarker::CacheStatus(CacheStatus::Reset {
                reason: CacheResetReason::EmergencyTrim,
            }),
            ControlMarker::CacheStatus(CacheStatus::Reset {
                reason: CacheResetReason::LcmCheckpoint,
            }),
            ControlMarker::CacheStatus(CacheStatus::Reset {
                reason: CacheResetReason::StalledProviderRequest,
            }),
        ];
        for m in variants {
            let wire = m.encode();
            assert!(
                wire.starts_with('\x00'),
                "markers are NUL-prefixed: {wire:?}"
            );
            assert_eq!(
                parse_control_marker(&wire),
                Some(m.clone()),
                "round-trip failed for {m:?} (wire: {wire:?})"
            );
        }
        // Plain text never parses as a marker.
        assert_eq!(parse_control_marker("hello"), None);
    }

    #[tokio::test]
    async fn delta_yields_before_tool_regardless_of_send_order() {
        let (delta_tx, delta_rx) = unbounded_channel::<String>();
        let (tool_tx, tool_rx) = unbounded_channel::<ToolEvent>();
        let (handle, _gate) = gated_handle("done");
        let mut stream = TurnStream::new(
            delta_rx,
            Some(tool_rx),
            Completion::AgentHandle(handle),
            None,
        );

        // Tool event sent FIRST, delta second — both buffered before the
        // first poll. The delta must still come out first.
        tool_tx.send(tool_event("t1")).unwrap();
        delta_tx.send("hello".into()).unwrap();

        assert!(matches!(ev(&mut stream).await, TurnEvent::Delta(d) if d == "hello"));
        assert!(matches!(ev(&mut stream).await, TurnEvent::Tool(_)));
    }

    /// Pinned rule: ALL deltas buffered at the moment a tool event would be
    /// yielded are yielded first. So d1, tool, d2 (all buffered) comes out
    /// as d1, d2, tool — the tool event only surfaces once the delta buffer
    /// is empty.
    #[tokio::test]
    async fn all_buffered_deltas_flush_before_a_tool_event() {
        let (delta_tx, delta_rx) = unbounded_channel::<String>();
        let (tool_tx, tool_rx) = unbounded_channel::<ToolEvent>();
        let (handle, _gate) = gated_handle("done");
        let mut stream = TurnStream::new(
            delta_rx,
            Some(tool_rx),
            Completion::AgentHandle(handle),
            None,
        );

        delta_tx.send("d1".into()).unwrap();
        tool_tx.send(tool_event("t1")).unwrap();
        delta_tx.send("d2".into()).unwrap();

        assert!(matches!(ev(&mut stream).await, TurnEvent::Delta(d) if d == "d1"));
        assert!(matches!(ev(&mut stream).await, TurnEvent::Delta(d) if d == "d2"));
        assert!(matches!(ev(&mut stream).await, TurnEvent::Tool(_)));
    }

    #[tokio::test]
    async fn finished_carries_join_result_after_buffers_drain() {
        let (delta_tx, delta_rx) = unbounded_channel::<String>();
        let (tool_tx, tool_rx) = unbounded_channel::<ToolEvent>();
        let handle = tokio::spawn(async { "final answer".to_string() });
        // Let the agent task resolve before the first poll: buffered events
        // must STILL be delivered before Finished.
        handle_settled(&handle).await;

        let mut stream = TurnStream::new(
            delta_rx,
            Some(tool_rx),
            Completion::AgentHandle(handle),
            None,
        );
        delta_tx.send("d1".into()).unwrap();
        tool_tx.send(tool_event("t1")).unwrap();
        drop(delta_tx);
        drop(tool_tx);

        assert!(matches!(ev(&mut stream).await, TurnEvent::Delta(d) if d == "d1"));
        assert!(matches!(ev(&mut stream).await, TurnEvent::Tool(_)));
        assert!(matches!(ev(&mut stream).await, TurnEvent::Finished(r) if r == "final answer"));
    }

    async fn handle_settled(handle: &JoinHandle<String>) {
        while !handle.is_finished() {
            tokio::task::yield_now().await;
        }
    }

    #[tokio::test]
    async fn cancel_discards_everything_and_finishes_empty() {
        let (delta_tx, delta_rx) = unbounded_channel::<String>();
        let (tool_tx, tool_rx) = unbounded_channel::<ToolEvent>();
        let token = CancellationToken::new();
        // Agent task mimics the real loop: runs until the token fires, then
        // returns a late (must-be-discarded) response.
        let agent_token = token.clone();
        let handle = tokio::spawn(async move {
            agent_token.cancelled().await;
            "late response".to_string()
        });
        let mut stream = TurnStream::new(
            delta_rx,
            Some(tool_rx),
            Completion::AgentHandle(handle),
            Some(token.clone()),
        );

        delta_tx.send("buffered".into()).unwrap();
        tool_tx.send(tool_event("t1")).unwrap();
        stream.cancel();
        assert!(token.is_cancelled(), "cancel() must fire the token");
        // Late events after cancellation are discarded too.
        delta_tx.send("late delta".into()).unwrap();

        assert!(matches!(ev(&mut stream).await, TurnEvent::Finished(r) if r.is_empty()));
    }

    #[tokio::test]
    async fn channel_close_policy_finishes_when_both_channels_close() {
        let (delta_tx, delta_rx) = unbounded_channel::<String>();
        let (tool_tx, tool_rx) = unbounded_channel::<ToolEvent>();
        let mut stream = TurnStream::new(delta_rx, Some(tool_rx), Completion::ChannelClose, None);

        delta_tx.send("d1".into()).unwrap();
        tool_tx.send(tool_event("t1")).unwrap();
        drop(delta_tx);
        drop(tool_tx);

        assert!(matches!(ev(&mut stream).await, TurnEvent::Delta(d) if d == "d1"));
        assert!(matches!(ev(&mut stream).await, TurnEvent::Tool(_)));
        assert!(matches!(ev(&mut stream).await, TurnEvent::Finished(r) if r.is_empty()));
    }

    #[tokio::test]
    async fn channel_close_policy_without_tool_channel_finishes_on_delta_close() {
        let (delta_tx, delta_rx) = unbounded_channel::<String>();
        let mut stream = TurnStream::new(delta_rx, None, Completion::ChannelClose, None);

        delta_tx.send("only".into()).unwrap();
        drop(delta_tx);

        assert!(matches!(ev(&mut stream).await, TurnEvent::Delta(d) if d == "only"));
        assert!(matches!(ev(&mut stream).await, TurnEvent::Finished(r) if r.is_empty()));
    }
}
