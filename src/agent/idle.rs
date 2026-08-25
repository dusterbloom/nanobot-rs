//! Idle-window agency — self-directed turns when nobody is talking (v0.5 E1).
//!
//! A gateway-side timer watches per-session inbound activity. When the
//! designated session has been quiet past `after_secs` AND the local
//! inference server is already warm (no cold model loads because the agent
//! was bored), the timer injects one synthetic observation onto the *same*
//! inbound bus `AgentLoop::run()` drains — the cron-executor injection
//! precedent — so the idle turn flows through the normal session lock /
//! permit / lease machinery and gets its own per-turn budget. Idle turns
//! are quiet by default: `run()` suppresses their final reply; the agent
//! reaches the human only via an explicit `message` tool call (whose
//! channel defaults are baked from the session's real channel/chat_id).
//! Backoff doubles per consecutive fire (capped) and resets on any real
//! inbound. File tools enforce a write allowlist during idle turns because
//! no human is watching (see `filesystem::idle_write_allowed`).

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;
use tracing::{debug, info, warn};

use crate::bus::events::InboundMessage;
use crate::config::schema::IdleConfig;

/// Metadata flag marking an injected idle turn. `run()` keys coalescing
/// skip, reply suppression, and tracker reset off this flag; file tools
/// key the write allowlist off the same turn.
pub const IDLE_TURN_META: &str = "idle_turn";

/// Timer granularity: how often the idle timer reconsiders firing.
const IDLE_TICK_SECS: u64 = 30;

/// Warm-probe timeout: the local server must answer /v1/models this fast.
const WARM_PROBE_TIMEOUT_SECS: u64 = 2;

/// The session the idle agent inhabits, with its last real inbound time.
#[derive(Debug, Clone)]
pub struct IdleTarget {
    pub channel: String,
    pub chat_id: String,
    pub last_inbound_ms: i64,
}

impl IdleTarget {
    pub fn session_key(&self) -> String {
        format!("{}:{}", self.channel, self.chat_id)
    }
}

/// Inbound-activity tracker, shared between `run()` (writer: notes every
/// real inbound) and the idle timer task (reader: resolves the designated
/// session and quiet duration). Idle turns themselves do not update it —
/// an agent's own thoughts must not reset its idle backoff.
#[derive(Default)]
pub struct IdleTracker {
    inner: Mutex<HashMap<String, IdleTarget>>,
}

impl IdleTracker {
    /// Record a real inbound message for its session.
    pub fn note_inbound(&self, channel: &str, chat_id: &str) {
        if channel.is_empty() || chat_id.is_empty() {
            return;
        }
        let key = format!("{}:{}", channel, chat_id);
        let now = now_ms();
        let mut inner = self.inner.lock();
        match inner.get_mut(&key) {
            Some(target) => target.last_inbound_ms = now,
            None => {
                inner.insert(
                    key,
                    IdleTarget {
                        channel: channel.to_string(),
                        chat_id: chat_id.to_string(),
                        last_inbound_ms: now,
                    },
                );
            }
        }
    }

    /// Test seam: note inbound at an explicit epoch-ms (production always
    /// uses `note_inbound`, which stamps now).
    #[cfg(test)]
    pub(crate) fn note_inbound_at(&self, channel: &str, chat_id: &str, at_ms: i64) {
        let key = format!("{}:{}", channel, chat_id);
        self.inner.lock().insert(
            key,
            IdleTarget {
                channel: channel.to_string(),
                chat_id: chat_id.to_string(),
                last_inbound_ms: at_ms,
            },
        );
    }

    /// Resolve the designated idle target: the configured session key if
    /// set and seen, else the most-recently-active session. None until at
    /// least one real inbound has been observed (a gateway restart safely
    /// disables idle agency until the user speaks).
    pub fn designated(&self, configured_key: Option<&str>) -> Option<IdleTarget> {
        let inner = self.inner.lock();
        if let Some(key) = configured_key {
            return inner.get(key).cloned();
        }
        inner
            .values()
            .max_by_key(|target| target.last_inbound_ms)
            .cloned()
    }
}

/// True if this inbound message is an injected idle turn.
pub fn is_idle_message(msg: &InboundMessage) -> bool {
    msg.metadata
        .get(IDLE_TURN_META)
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
}

/// Effective wait before the next idle turn: `after * 2^consecutive_fires`,
/// capped at `max_backoff`. Pure.
pub fn effective_wait_secs(after_secs: u64, consecutive_fires: u32, max_backoff_secs: u64) -> u64 {
    // Saturating shift: beyond 63 doublings the cap governs anyway.
    let factor = 1u64.saturating_mul(1 << consecutive_fires.min(62));
    (after_secs.saturating_mul(factor)).min(max_backoff_secs).max(after_secs)
}

/// Sliding one-hour window cap. Prunes entries older than an hour and
/// returns whether one more fire fits under `cap`. Pure given `now_ms`.
pub fn hour_allows(fire_times_ms: &mut VecDeque<i64>, now_ms: i64, cap: u32) -> bool {
    while let Some(&oldest) = fire_times_ms.front() {
        if now_ms - oldest > 3_600_000 {
            fire_times_ms.pop_front();
        } else {
            break;
        }
    }
    (fire_times_ms.len() as u32) < cap
}

/// The self-directed observation injected as the idle turn's user message.
/// Kept short for weak local models; lists what the agent may do, names
/// the notify channel, and forbids fabricated user requests.
fn build_observation(quiet_mins: u64, notify_channel: &str) -> String {
    format!(
        "[idle] {quiet_mins} minutes of quiet — self-directed turn. Your reply here is silent; \
use the `message` tool to reach the user on {notify_channel} only if you have something worth saying.\n\
Continue useful background work in short steps: consolidate memory (remember/recall), review or \
improve a skill, or reflect on recent sessions. If nothing is genuinely worth doing, reply with \
one line and stop. Never invent user requests or notifications."
    )
}

pub fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// Config + tracker bundle stored on `AgentLoopShared`. Only the gateway
/// wires a live config (REPL/TUI direct mode has no timer task, mirroring
/// the cron executor's gateway-only stance).
#[derive(Default)]
pub struct IdleRuntime {
    pub config: IdleConfig,
    pub tracker: Arc<IdleTracker>,
}

impl IdleRuntime {
    pub fn new(config: IdleConfig) -> Self {
        Self {
            config,
            tracker: Arc::new(IdleTracker::default()),
        }
    }
}

/// The warm-probe URL. `local_api_base` may or may not already carry the
/// `/v1` suffix (production config uses "http://host:9000/v1"), so append
/// only what is missing — a doubled /v1 would 404 and freeze idle agency.
fn models_url(local_api_base: &str) -> String {
    let base = local_api_base.trim_end_matches('/');
    if base.ends_with("/v1") {
        format!("{}/models", base)
    } else {
        format!("{}/v1/models", base)
    }
}

/// The gateway idle timer. Runs until the process exits (the tracker and
/// backoff state are in-memory; a restart simply resets them).
pub async fn run_idle_timer(
    config: IdleConfig,
    tracker: Arc<IdleTracker>,
    inbound_tx: tokio::sync::mpsc::UnboundedSender<InboundMessage>,
    local_api_base: String,
) {
    if local_api_base.is_empty() {
        info!("idle timer disabled: no local inference base configured");
        return;
    }
    info!(
        after_secs = config.after_secs,
        max_turns_per_hour = config.max_turns_per_hour,
        "idle timer started"
    );
    run_idle_timer_with_tick(config, tracker, inbound_tx, local_api_base, IDLE_TICK_SECS).await
}

/// Tick-configurable core for tests (production always uses IDLE_TICK_SECS).
pub(crate) async fn run_idle_timer_with_tick(
    config: IdleConfig,
    tracker: Arc<IdleTracker>,
    inbound_tx: tokio::sync::mpsc::UnboundedSender<InboundMessage>,
    local_api_base: String,
    tick_secs: u64,
) {
    let base = local_api_base.trim_end_matches('/').to_string();
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(WARM_PROBE_TIMEOUT_SECS))
        .build()
        .expect("idle warm-probe client");
    let mut consecutive_fires: u32 = 0;
    let mut fire_times: VecDeque<i64> = VecDeque::new();
    let mut next_due_ms: i64 = 0;

    loop {
        tokio::time::sleep(Duration::from_secs(tick_secs)).await;
        let now = now_ms();
        let Some(target) = tracker.designated(config.session_key.as_deref()) else {
            continue;
        };

        // Any real inbound since our last observation resets the backoff —
        // the agent was just used; idle urgency is gone.
        let quiet_ms = now.saturating_sub(target.last_inbound_ms);
        if quiet_ms < (config.after_secs.saturating_mul(1000)) as i64 {
            consecutive_fires = 0;
            continue;
        }
        if now < next_due_ms {
            continue;
        }
        if !hour_allows(&mut fire_times, now, config.max_turns_per_hour) {
            continue;
        }

        // Warm-only: never cold-load a model for an idle turn.
        if let Err(error) = client.get(models_url(&base)).send().await {
            debug!(
                %error,
                "idle turn skipped: local inference server not warm"
            );
            continue;
        }

        let mut msg = InboundMessage::new(
            target.channel.clone(),
            "idle",
            target.chat_id.clone(),
            &build_observation((quiet_ms / 60_000).max(1) as u64, &target.channel),
        );
        msg.metadata
            .insert(IDLE_TURN_META.to_string(), serde_json::json!(true));
        if inbound_tx.send(msg).is_err() {
            warn!("idle timer: inbound bus closed, stopping");
            return;
        }
        fire_times.push_back(now);
        consecutive_fires += 1;
        next_due_ms = now + effective_wait_secs(
            config.after_secs,
            consecutive_fires,
            config.max_backoff_secs,
        )
        .saturating_mul(1000) as i64;
        info!(session = %target.session_key(), "idle turn injected");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(metadata_idle: bool) -> InboundMessage {
        let mut m = InboundMessage::new("telegram", "u", "c", "hi");
        if metadata_idle {
            m.metadata
                .insert(IDLE_TURN_META.to_string(), serde_json::json!(true));
        }
        m
    }

    #[test]
    fn effective_wait_doubles_then_caps() {
        assert_eq!(effective_wait_secs(900, 0, 3600), 900);
        assert_eq!(effective_wait_secs(900, 1, 3600), 1800);
        assert_eq!(effective_wait_secs(900, 2, 3600), 3600);
        assert_eq!(effective_wait_secs(900, 9, 3600), 3600, "capped");
        assert_eq!(effective_wait_secs(900, 3, 10_000), 7200, "cap > doublings");
        // Never below after_secs.
        assert_eq!(effective_wait_secs(900, 0, 100), 900);
    }

    #[test]
    fn hour_window_prunes_and_caps() {
        // All three fires inside the last hour: a 4th is blocked at cap 3.
        let now = 5_000_000i64;
        let mut fires = VecDeque::from(vec![now - 3000, now - 2000, now - 1000]);
        assert!(!hour_allows(&mut fires, now, 3), "window still full");
        // Just past the oldest fire's hour (but inside the others'):
        // exactly one entry is pruned, freeing one slot.
        let later = now - 3000 + 3_600_001;
        assert!(hour_allows(&mut fires, later, 3));
        assert_eq!(fires.len(), 2, "oldest pruned; recording is the caller's job");
        let mut few = VecDeque::new();
        assert!(hour_allows(&mut few, now, 1));
    }

    #[test]
    fn tracker_resolves_configured_or_most_recent() {
        let t = IdleTracker::default();
        assert!(t.designated(None).is_none(), "empty until real inbound");
        t.note_inbound_at("telegram", "111", 100);
        t.note_inbound_at("telegram", "222", 200);
        t.note_inbound_at("cli", "default", 300);
        let most = t.designated(None).unwrap();
        assert_eq!(most.session_key(), "cli:default", "latest wins");
        let pinned = t.designated(Some("telegram:111")).unwrap();
        assert_eq!(pinned.chat_id, "111");
        assert!(t.designated(Some("nope:x")).is_none());
    }

    #[test]
    fn idle_messages_are_flagged_only_via_metadata() {
        assert!(is_idle_message(&msg(true)));
        assert!(!is_idle_message(&msg(false)));
    }

    #[test]
    fn observation_names_notify_channel_and_forbids_invention() {
        let text = build_observation(34, "telegram");
        assert!(text.contains("[idle] 34 minutes"));
        assert!(text.contains("`message` tool"));
        assert!(text.contains("telegram"));
        assert!(text.contains("Never invent"));
    }

    #[test]
    fn models_url_never_doubles_the_v1_suffix() {
        assert_eq!(
            models_url("http://127.0.0.1:9000/v1"),
            "http://127.0.0.1:9000/v1/models"
        );
        assert_eq!(
            models_url("http://127.0.0.1:9000"),
            "http://127.0.0.1:9000/v1/models"
        );
        assert_eq!(
            models_url("http://127.0.0.1:9000/v1/"),
            "http://127.0.0.1:9000/v1/models"
        );
    }
}
