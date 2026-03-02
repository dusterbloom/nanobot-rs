# Nanobot Architecture Review

## Dependency Graph Summary

```
                       ┌─────────────┐
                       │   main.rs   │
                       │   cli.rs    │
                       └──────┬──────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
      ┌───────▼──────┐ ┌─────▼─────┐ ┌───────▼──────┐
      │  AgentLoop   │ │ ChannelMgr│ │   CronSvc    │
      │ (agent_loop) │ │ (channels)│ │   (cron)     │
      └───────┬──────┘ └─────┬─────┘ └───────┬──────┘
              │               │               │
    ┌─────────┼──────┐    bus_tx (mpsc)       │
    │         │      │        │               │
┌───▼──┐ ┌───▼───┐ ┌▼────┐   │               │
│Tools │ │Context│ │Sub- │   │               │
│Regis │ │Builder│ │agent│   │               │
└───┬──┘ └───────┘ └─────┘   │               │
    │                         │               │
┌───▼─────────────┐   ┌──────▼──────┐  ┌─────▼─────┐
│ Tool impls      │   │  Telegram   │  │ Session   │
│ (shell,fs,web,  │   │  WhatsApp   │  │ Manager   │
│  spawn,cron,msg)│   │  Feishu     │  │ (JSONL)   │
└───────┬─────────┘   │  Email      │  └───────────┘
        │             └─────────────┘
┌───────▼─────────┐
│ LLM Providers   │
│ (OpenAI-compat, │
│  Anthropic)     │
└─────────────────┘
```

## Identified Concerns & Implemented Fixes

### Fix 1: Bounded MessageBus Channels (Backpressure)

**Component:** `bus/queue.rs`, `channels/manager.rs`, `channels/{telegram,whatsapp,feishu,email}.rs`, `cli.rs`

**Risk:** HIGH — Under sustained load (e.g., Telegram group with many messages), the unbounded `mpsc::unbounded_channel` could grow without limit, eventually causing OOM.

**Root cause:** Both `MessageBus` and the direct gateway channel path used `mpsc::unbounded_channel()` with no flow control between fast producers (channel adapters) and the slower consumer (agent loop doing LLM inference).

**Fix:**
- `MessageBus` inbound channel changed from `UnboundedSender`/`UnboundedReceiver` to bounded `Sender`/`Receiver` with capacity 256
- Gateway path in `cli.rs` uses a bounded channel (cap 256) from channel adapters → bridge task → unbounded channel to `AgentLoop`
- All channel adapters (`Telegram`, `WhatsApp`, `Feishu`, `Email`) updated to use `Sender<InboundMessage>` and `try_send()` with logged warnings on backpressure

**Key code:**
```rust
// bus/queue.rs
const INBOUND_CHANNEL_CAPACITY: usize = 256;
let (inbound_tx, inbound_rx) = mpsc::channel(INBOUND_CHANNEL_CAPACITY);

// publish_inbound uses try_send for non-blocking backpressure
pub fn publish_inbound(&self, msg: InboundMessage) {
    if let Err(e) = self.inbound_tx.try_send(msg) {
        warn!("Inbound queue full (capacity {}), dropping: {}", INBOUND_CHANNEL_CAPACITY, e);
    }
}
```

**Integration test strategy:**
- Spawn N producer tasks that flood `publish_inbound` at max rate
- Verify memory usage stays bounded (no growth beyond capacity * msg_size)
- Verify backpressure warning is logged when capacity reached
- Verify no messages lost under normal load (capacity not exceeded)

---

### Fix 2: Atomic Cron Persistence

**Component:** `cron/service.rs`

**Risk:** MEDIUM — A crash or power loss during `fs::write()` can corrupt `cron.json`, losing all scheduled job definitions.

**Root cause:** `CronService::persist()` wrote directly to the store file. Unlike `SessionManager::save_session()` which already uses atomic temp-then-rename, cron used a single `fs::write()`.

**Fix:** Write to `*.json.tmp` first, then atomically `fs::rename()` to the final path.

**Key code:**
```rust
fn persist(&self) {
    let tmp_path = self.store_path.with_extension("json.tmp");
    if let Err(e) = std::fs::write(&tmp_path, &json) {
        warn!("Failed to write temp cron store: {}", e);
        return;
    }
    if let Err(e) = std::fs::rename(&tmp_path, &self.store_path) {
        warn!("Failed to rename cron store: {}", e);
    }
}
```

**Integration test strategy:**
- Create a cron service, add jobs, call persist
- Kill the process mid-write (simulate with a signal during a large write)
- Verify either the old file or the new file is valid JSON — never corrupted
- Verify existing tests still pass (they do)

---

### Fix 3: Shell Tool Workspace Restriction Hardening

**Component:** `agent/tools/shell.rs`

**Risk:** HIGH — The workspace restriction could be bypassed by putting paths inside quotes (e.g., `cat '/etc/passwd'`), because the regex `/[^\s"']+` stopped matching at quote boundaries.

**Root cause:** The path extraction regex only matched unquoted absolute paths. Paths inside single or double quotes were invisible to the guard.

**Fix:** Added two additional regex patterns to extract paths from inside single-quoted and double-quoted strings:

```rust
// Match paths inside double quotes: "/etc/passwd"
let posix_double_quoted = Regex::new(r#""(/[^"]+)""#);
// Match paths inside single quotes: '/etc/passwd'
let posix_single_quoted = Regex::new(r"'(/[^']+)'");
```

**New tests added:**
- `test_guard_blocks_single_quoted_absolute_path` — verifies `cat '/etc/passwd'` is blocked
- `test_guard_blocks_double_quoted_absolute_path` — verifies `cat "/etc/passwd"` is blocked

**Integration test strategy:**
- Fuzz the `guard_command` function with various quoting styles (`"`, `'`, backticks, `$()`)
- Verify all known bypass patterns are caught
- Verify legitimate workspace-local quoted paths still work (e.g., `cat "my file.txt"`)

---

### Fix 4: Channel Message Deduplication

**Component:** `bus/events.rs`, `channels/telegram.rs`, `channels/whatsapp.rs`

**Risk:** MEDIUM — Duplicate message processing wastes LLM tokens and can confuse the agent. Telegram relies solely on offset tracking; WhatsApp has no dedup at all.

**Root cause:** No content-based deduplication. If the Telegram offset tracker breaks (e.g., JSON parse failure) or WhatsApp bridge reconnects, the same message can be processed twice.

**Fix:** Added `MessageDedup` — a lightweight fingerprint-based LRU cache:

```rust
pub struct MessageDedup {
    seen: Mutex<VecDeque<u64>>,  // ring buffer of fingerprints
    capacity: usize,
}

impl MessageDedup {
    pub fn fingerprint(sender_id: &str, chat_id: &str, content: &str) -> u64 { ... }
    pub fn is_duplicate(&self, fingerprint: u64) -> bool { ... }
}
```

Both Telegram and WhatsApp now check `dedup.is_duplicate()` before publishing to the bus. The capacity is 512 fingerprints (covers ~500 recent messages per channel).

**Integration test strategy:**
- Unit tests for `MessageDedup`: duplicate rejection, different-message acceptance, LRU eviction
- End-to-end: send the same message twice via Telegram test API, verify agent only processes it once
- Verify legitimate rapid-fire different messages are all processed

---

### Fix 5: Bounded Session Cache with LRU Eviction

**Component:** `session/manager.rs`

**Risk:** LOW-MEDIUM — In long-running gateway deployments with many unique chat IDs, the unbounded `HashMap<String, Session>` cache grows without limit.

**Root cause:** `SessionManager.cache` never evicts entries. Every unique `session_key` that ever loads stays in memory forever.

**Fix:** Added `max_cached_sessions` (default: 128) with LRU eviction. When the cache is full and a new session is requested, the session with the oldest `updated_at` timestamp is saved to disk and evicted:

```rust
if cache.len() >= max_cached_sessions {
    let oldest_key = cache.iter()
        .min_by_key(|(_, s)| s.updated_at)
        .map(|(k, _)| k.clone());
    if let Some(evict_key) = oldest_key {
        Self::save_session(evicted, sessions_dir);
        cache.remove(&evict_key);
    }
}
```

**Integration test strategy:**
- Create `SessionManager` with `max_cached_sessions = 3`
- Load 5 different sessions, verify cache size never exceeds 3
- Verify evicted sessions are persisted to disk and can be reloaded
- Verify the most-recently-used sessions survive eviction

---

## Composition Strategy

These 5 fixes are designed to be incrementally composable:

1. **Fix 1** (bounded channels) and **Fix 4** (dedup) are independent and can be deployed separately
2. **Fix 2** (atomic cron) is fully isolated — no dependencies on other fixes
3. **Fix 3** (shell hardening) is self-contained in the tool guard
4. **Fix 5** (session cache) is independent of the channel/bus changes
5. Deploying all 5 together: each fix reduces a different failure mode, and they don't interact

## Remaining Concerns (Not Addressed)

- **Cron overlapping execution:** Jobs that run longer than their interval can overlap (no in-flight guard)
- **WhatsApp bridge crash detection:** Node.js child process death is not actively monitored
- **Config validation at load time:** Bad URLs/ports only fail at runtime
- **Streaming delta queue:** `UnboundedReceiver` for text deltas could spike memory during fast streaming
