# One Typed Error Protocol + a Typed Host Bridge for nanobot-rs

Date: 2026-08-06. Scope: research only — no code changed. Companion to
`docs/research/2026-08-06-rlm-prime-agent-review.md`.

Two coupled problems are investigated:

1. **Mixed error conventions** — four coexisting error formats make failures
   un-matchable, un-retryable, and un-auditable. Research collapses them into
   one typed protocol.
2. **Typed host bridge** — string-based callbacks (`SpawnTool` with 7
   callbacks) keep the tool layer coupled to the agent loop. Research ports
   prime-agent's typed `host.request` bridge to Rust.

The two are coupled: a bridge is only as typed as its error channel. The error
protocol (Topic 1) is the prerequisite; the bridge (Topic 2) is its first major
consumer.

---

## 1. Current-state audit

### 1.1 The four coexisting error formats

| # | Format | Where | Sites | Semantics |
|---|--------|-------|-------|-----------|
| A | `anyhow::Result<T>` | `LLMProvider::chat` (`src/providers/base.rs`), `SendCallback` (`src/agent/tools/message.rs:17`), misc. helpers | ~dozens | App-level context; downcast to `ProviderError` |
| B | `String` returns with `"Error: ..."` prefix | `Tool::execute` (`src/agent/tools/base.rs:144`), `require_str!` macro, `remember.rs:357` `Result<_, String>`, `registry.rs` `normalize_tool_request` | **297** `"Error:` sites in `src/` | Model-facing wire format; parsed by substring |
| C | `ToolExecutionResult` struct `{ok, data, error, error_kind}` | `base.rs:65`, registry execute path, tool_runner retry, `code_execution` RPC | 3 direct struct-literal constructions; ~100 `.ok` reads, ~184 `.data` reads | Structured outcome; `ok: bool` + `Option<String> error` invariant is unenforced |
| D | `finish_reason: "error"` | `providers/openai_compat.rs:1842` (synthesized), `providers/base.rs:49-54` `is_error()`, `agent_loop/response.rs:124` → `ResponseKind::ProviderError` | ~15 | LLM-response-level error; also magic strings `"aborted"`, `"cancelled"` matched at `response.rs:1024` |

### 1.2 Why each format exists

- **A (anyhow)** is the app boundary convention. `errors.rs` already embeds the
  typed `ProviderError` in `anyhow::Error` so callers downcast
  (`errors.rs:57` `is_retryable_provider_error`). This is the textbook
  thiserror-library / anyhow-app split — but it leaks into the tool layer
  (`SendCallback` returns `anyhow::Result<()>`).
- **B (String)** is the model-facing contract. The model reads `Error: ...`
  lines and the loop decides retry/repair from them. `context_hygiene.rs:80`
  `tool_result_ok()` treats a `"Error:"`-prefixed string as failure, and
  `errors.rs` `classify_tool_error()` *re-parses the string into a taxonomy* by
  substring matching (`"timed out"`, `"429"`, `"permission denied"`, …). This is
  the smell that drives everything else: the taxonomy already exists
  (`ToolErrorKind`), it is just reconstructed from text at runtime instead of
  being produced at the source.
- **C (ToolExecutionResult)** is the first attempt at structure. It still
  carries the string (`data`, `error`) *and* the classification (`error_kind`),
  and its `ok: bool` invites inconsistent states (nothing stops
  `ok: true` + `error: Some(..)`).
- **D (finish_reason="error")** is a provider-stream artifact: a stream that
  ends before producing content/tool-calls is reported as a response whose
  `finish_reason` is the literal string `"error"` (`openai_compat.rs:1831-1845`).
  The loop then classifies it as `ResponseKind::ProviderError`
  (`response.rs:124-125`). It is an error *encoded in a status field*.

### 1.3 Related inconsistencies found

- `Result<(), String>` in `crw.rs` (`ensure_crw`) — "Err is informational"
  (a third string-error channel).
- `ToolRunResult.error: Option<String>` (`tool_runner/mod.rs`) — a fourth
  string slot.
- 10 callback type aliases with **10 different arities and return types**
  (`spawn.rs:19-59`, `message.rs:17`, `bus/queue.rs:22`,
  `heartbeat/service.rs:39`): `SpawnCallback` takes **7 positional args**,
  `LoopCallback` 6, `SendCallback` returns `anyhow::Result<()>`,
  `OutboundCallback` returns `()`, `HeartbeatCallback` returns `Option<String>`.
- `ToolErrorKind::MissingArg` is already produced structurally by 3 tools
  (`recall.rs:717`, `remember.rs:333`, registry fallback at
  `registry.rs:699-713`), while the other ~290 error sites still rely on
  `classify_tool_error` substring matching. The codebase is mid-migration; this
  research completes the design so the migration has a target.

### 1.4 Compile-time reality today

`Tool::execute` **returns `String`**. A tool author can return `"Error: ..."`
or `"worked fine"` — the type system cannot tell them apart. `ok`/`error`/`data`
invariants on `ToolExecutionResult` are documentation, not law. `finish_reason`
is `String`, so `"error"` is one typo away from being a *successful* stop
reason.

---

## 2. Topic 1 — One typed error protocol

### 2.1 Design goals

1. **One type for every tool failure**: `Result<ToolOutput, ToolError>`.
2. **The wire string is a rendering, not a type**: `"Error: ..."` survives
   byte-for-byte (297 sites and dozens of tests assert exact substrings), but
   it is produced by `ToolError::render()` at the registry boundary.
3. **Classification is structural**: `is_retryable()`, model-fixable, infra
   flags are methods on the enum — `classify_tool_error` dies.
4. **Anyhow stays at the app edge**: tool layer never sees `anyhow::Error`.
5. **Compile-time enforcement**: the trait signature is the law; lints close
   the escape hatches.

### 2.2 The unified error enum

```rust
//! src/errors.rs — the single typed failure for the tool layer.
//!
//! Every failure that crosses a tool boundary is this enum: never a bare
//! string, never `anyhow::Error`, never a struct with an `ok: bool` hole.

/// Severity/action axes, collapsed into one enum so a single `match` gives
/// retryability, model-fixability, and infra attribution.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ToolError {
    // ---- model-recoverable (the model can repair its own call) ----
    #[error("Missing required argument '{param}'; call as {example}")]
    MissingArg { param: String, example: String },

    #[error("Invalid arguments: {message}")]
    InvalidArgs { message: String },

    // ---- infra / policy (the model cannot fix these) ----
    #[error("Tool '{name}' not found")]
    ToolNotFound { name: String },

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Permission denied: {0}")]
    PermissionDenied(String),

    // ---- transient (safe to retry) ----
    #[error("Command timed out after {0}s")]
    Timeout(u64),

    #[error("Network error: {0}")]
    Network(String),

    #[error("Rate limited")]
    RateLimited,

    #[error("Service unavailable: {0}")]
    ServiceUnavailable(String),

    /// Everything else. The registry converts panics here; unmigrated tools
    /// funnel legacy `"Error: ..."` strings through [`Self::from_legacy`] —
    /// the *only* string-to-error path left in the codebase.
    #[error("Execution failed: {message}")]
    Execution { message: String },
}

impl ToolError {
    /// Transient ⇒ the tool runner may retry with backoff.
    /// Mirrors today's `ToolErrorKind::is_retryable` (`errors.rs:124`).
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            Self::Timeout(_) | Self::Network(_) | Self::RateLimited | Self::ServiceUnavailable(_)
        )
    }

    /// Model-recoverable ⇒ the loop re-injects the call with a corrected
    /// shape instead of failing the turn. Replaces the `MissingArg` special
    /// case in `registry.rs:699-713` and the `"is required"` string fallback.
    pub fn is_model_fixable(&self) -> bool {
        matches!(self, Self::MissingArg { .. } | Self::InvalidArgs { .. })
    }

    /// The exact wire string the model sees. Byte-stable with today's
    /// `"Error: ..."` convention so `tool_result_ok` (`context_hygiene.rs:80`)
    /// and the 297 exact-substring tests keep passing unmodified.
    pub fn render(&self) -> String {
        match self {
            Self::MissingArg { param, example } => {
                format!("Error: '{}' parameter is required; call as {}", param, example)
            }
            other => format!("Error: {other}"),
        }
    }

    /// The single legacy bridge. Called only by the migration adapter in
    /// §2.6 Phase 1. Maps the exact strings `classify_tool_error` matched
    /// today, so retry behavior is preserved.
    pub fn from_legacy(msg: &str) -> Self {
        if let Some(kind) = classify_tool_error(msg) {
            return match kind {
                ToolErrorKind::Timeout(s) => Self::Timeout(s),
                ToolErrorKind::PermissionDenied(m) => Self::PermissionDenied(m),
                ToolErrorKind::NotFound(m) => Self::NotFound(m),
                ToolErrorKind::InvalidArgs(m) => Self::InvalidArgs { message: m },
                ToolErrorKind::ToolNotFound(m) => Self::ToolNotFound { name: m },
                ToolErrorKind::ExecutionFailed(m) => Self::Execution { message: m },
                ToolErrorKind::NetworkError(m) => Self::Network(m),
                ToolErrorKind::RateLimited => Self::RateLimited,
                ToolErrorKind::ServiceUnavailable(m) => Self::ServiceUnavailable(m),
                ToolErrorKind::MissingArg { param, example } => Self::MissingArg { param, example },
            };
        }
        Self::Execution { message: msg.to_string() }
    }
}
```

The success side is a plain struct, not `String`, so the registry can attach
audit metadata later without a breaking change:

```rust
/// The typed success payload of a tool call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolOutput {
    /// The model-facing text. May carry `TOOL_RESULT_HANDLE v1` receipts,
    /// `[truncated: …]` markers, or raw output.
    pub text: String,
}

/// The one result type of the tool layer.
pub type ToolResult = Result<ToolOutput, ToolError>;
```

`ToolExecutionResult` becomes a *rendered view* for the two call sites that
still need the legacy shape (the Python RPC in `code_execution.rs` and the
audit log), produced by a single conversion:

```rust
impl From<ToolResult> for ToolExecutionResult {
    fn from(r: ToolResult) -> Self {
        match r {
            Ok(out) => ToolExecutionResult {
                ok: true,
                data: out.text,
                error: None,
                error_kind: None,
            },
            Err(e) => ToolExecutionResult {
                ok: false,
                data: e.render(),
                error: Some(e.to_string()),
                error_kind: legacy_kind_from_tool_error(&e),
            },
        }
    }
}
```

### 2.3 thiserror/anyhow integration pattern

nanobot already follows the community-standard split and should keep it:

- **`thiserror`** for domain enums (`ProviderError`, `ToolError`, higgs's
  `ServerError`/`EngineError`): cheap, zero-dep, `#[from]` chains foreign
  errors, `Display` derives the message.
- **`anyhow`** for application plumbing: `?` on heterogeneous errors,
  `context()`, and **downcasting to the domain enum** when a decision is
  needed. nanobot already does this for providers (`errors.rs:88`); the rule
  to adopt is *downcast-or-propagate, never string-match an anyhow error*.

The boundary rule, enforced by lint (§2.5):

> **No `anyhow::Error` crosses the tool boundary.** `Tool::execute` returns
> `ToolResult`. A tool that calls an anyhow-returning function converts at the
> edge with `.map_err(|e| ToolError::Execution { message: e.to_string() })` —
> or better, downcasts: `.map_err(|e| e.downcast::<ToolError>().unwrap_or_else(...))`
> — but the *type* that leaves the tool is always `ToolError`.

For `finish_reason` (format D), the fix is the same shape: replace the magic
string with an enum *at the provider boundary*, keep a `String`-compatible
wire parse:

```rust
/// providers/base.rs — replaces `pub finish_reason: String`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
    ToolCalls,
    /// Stream died before producing content/tool-calls
    /// (was the literal `"error"` string, `openai_compat.rs:1842`).
    ProviderFailure,
    /// Aborted/cancelled — was magic strings matched at `response.rs:1024`.
    Aborted,
    Cancelled,
    /// Unknown provider value — kept so foreign backends don't break parsing.
    Other(String),
}

impl LLMResponse {
    /// `Result` view: `Ok(())` for stop/length/tool_calls,
    /// `Err(ProviderError::EmptyStream)` for the dead-stream case.
    pub fn outcome(&self) -> Result<(), ProviderError> {
        match self.finish_reason {
            FinishReason::ProviderFailure => Err(ProviderError::EmptyStream(
                self.content.clone().unwrap_or_else(|| "Unknown LLM error".to_string()),
            )),
            _ => Ok(()),
        }
    }
}
```

`ProviderError::EmptyStream(String)` is a new variant in `errors.rs` (replace
the current string-encoded `error_detail()`). `classify_response`
(`response.rs:115-126`) then matches `response.outcome()` instead of
`response.is_error()`.

### 2.4 Error classification taxonomy

`ToolErrorKind` already *is* a classification taxonomy; the problem is it is
reconstructed from strings. The taxonomy to institutionalize has three axes:

1. **Recoverability** — transient (retry) vs permanent (report).
2. **Blame** — model (bad call), policy (permission/config), infra (network,
   backend), environment (timeout/fs).
3. **Loop action** — retry with backoff / re-inject corrected call / fail turn
   / surface to user.

The enum §2.2 encodes all three; the loop's three consumers map directly:

| Consumer | Today (fragile) | Tomorrow (structural) |
|---|---|---|
| Tool runner retry (`tool_runner/mod.rs:1188`) | `result.is_retryable()` on parsed `error_kind` | `ToolError::is_retryable()` |
| Registry call-shape hint (`registry.rs:699-713`) | `error_kind == MissingArg` + `"is required"` string fallback | `ToolError::is_model_fixable()` → render includes example |
| Metrics/audit (`metrics.rs:38`, `audit.rs`) | `ok: bool` + raw string | `ToolError` variant name + `render()` |

Reference taxonomies that validate this shape: OpenAI/Anthropic API error types
(`invalid_request_error` / `authentication_error` / `rate_limit_error` /
`overloaded_error` …) which higgs already mirrors in `ServerError::IntoResponse`
(`crates/higgs/src/error.rs:33-88` — enum → `(status, error_type, message)`
table); AWS/Google retry classifications (throttle vs service unavailable vs
permanent); and the SRE error-budget split (deployment-visible vs
user-visible). The Rust encoding is always the same: **one enum, one
`is_retryable()` method, one render/into-wire method.**

### 2.5 Enforcing a single error convention at compile time

Four enforcement layers, in increasing strength:

**Layer 1 — the type system (the real enforcement).**
Change `Tool::execute` to return `ToolResult`. `"Error: ..."` as a bare string
becomes *unrepresentable* — the 297 sites stop compiling and must be migrated.
This is the same move higgs made with `unsafe_code = "deny"` plus type-level
gates (e.g. `mlx_exec` token): the convention is not a review comment, it is a
build failure.

```rust
// base.rs — new Tool trait (ISP-split shown in §3.3)
#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters(&self) -> serde_json::Value;
    fn permission(&self) -> PermissionLevel { PermissionLevel::ReadOnly }
    fn concurrency(&self) -> ToolConcurrency { ToolConcurrency::Sequential }
    fn is_available(&self) -> bool { true }

    /// The single execution entry point. Returns the typed outcome.
    async fn execute(
        &self,
        params: HashMap<String, serde_json::Value>,
        ctx: &ToolContext,
    ) -> ToolResult;
}
```

**Layer 2 — `clippy::disallowed_methods`/`disallowed_types` (kills the escape
hatches).** higgs already uses `disallowed-methods` in `clippy.toml` for the
MLX gate; the same mechanism bans the legacy constructors:

```toml
# clippy.toml additions
disallowed-methods = [
    # The legacy string-error channel. New code must construct ToolError
    # variants; rendering happens once, at the registry boundary.
    { path = "crate::agent::tools::base::ToolExecutionResult::from_output",
      reason = "legacy string-error adapter — migrate to ToolResult" },
    # Substring classification is fragile; return typed ToolError variants.
    { path = "crate::errors::classify_tool_error",
      reason = "string classification — produce ToolError at the source" },
]

disallowed-types = [
    # No anyhow at the tool boundary.
    { path = "anyhow::Error",
      reason = "tool layer must return ToolResult; convert at the edge" },
]
```

**Layer 3 — workspace lint gates (higgs-style deny regime).** nanobot has no
`[workspace.lints]` today. Adopt higgs's:

```toml
# Cargo.toml
[lints.rust]
unsafe_code = "deny"

[lints.clippy]
pedantic = { level = "warn", priority = -1 }
nursery = { level = "warn", priority = -1 }
unwrap_used = "deny"
expect_used = "deny"
panic = "deny"
todo = "deny"
unimplemented = "deny"
unreachable = "deny"
indexing_slicing = "deny"
as_conversions = "deny"
dbg_macro = "deny"
print_stdout = "deny"
print_stderr = "deny"
shadow_reuse = "deny"
shadow_same = "deny"
shadow_unrelated = "deny"
try_err = "deny"
string_add = "deny"
format_push_string = "deny"
```

Test modules get the sanctioned escape hatch higgs uses
(`crates/higgs/src/error.rs` test module):

```rust
#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used, clippy::indexing_slicing)]
mod tests { /* … */ }
```

**Layer 4 — compile-fail tests (trybuild).** Lock the protocol against
regression with a UI test that asserts the legacy forms do NOT compile:

```rust
// tests/ui/error_protocol.rs — run by trybuild::TestCases::compile_fail
// (fragment; the full file exercises each banned form)
fn main() {
    // A tool returning a bare String for an error must not exist:
    // Tool::execute returns ToolResult, so this is a type error already.
    // The test locks the trait signature and the disallowed constructors.
}
```

During migration, gate enforcement by *module*: unmigrated tools carry
`#![allow(clippy::disallowed_methods)]` at the top; the allow is removed as each
tool lands on `ToolResult`, and a grep for `from_output`/`classify_tool_error`
returns zero before the lints flip to `deny` globally.

### 2.6 Migration path for the error protocol

Four phases; the crate compiles and passes tests at the end of each.

**Phase 0 — make `ToolExecutionResult` safe (no behavior change).**
Make its fields private; expose `ok()`, `data()`, `error()`, `error_kind()`.
Only `base.rs`, `recall.rs`, `remember.rs` construct it, so the churn is tiny;
readers migrate to the accessors. `#[must_use]` on the constructors.

**Phase 1 — introduce `ToolError` + `ToolResult` alongside (additive).**
Add the types and `ToolError::from_legacy` (§2.2). Change the default
`execute_with_result` to `ToolExecutionResult::from(Ok/Err)` so the *existing*
trait still compiles. Add `Tool::execute_typed` as a new default method that
delegates to `execute` + `from_legacy`:

```rust
// base.rs, Phase 1 (additive)
async fn execute_typed(
    &self,
    params: HashMap<String, serde_json::Value>,
    ctx: &ToolExecutionContext,
) -> ToolResult {
    let out = self.execute_with_result_and_context(params, ctx).await;
    match out.ok {
        true => Ok(ToolOutput { text: out.data }),
        false => Err(ToolError::from_legacy(&out.error.unwrap_or(out.data))),
    }
}
```

**Phase 2 — migrate tools one at a time (the big codemod).**
Per tool: implement `execute_typed` directly, building `ToolError` variants at
the failure sites (the 297 sites shrink to ~30 per tool); the legacy
`execute` body becomes a thin `render()`-wrapped call or is deleted. Each tool
loses its `#![allow(clippy::disallowed_methods)]` when done. Order by value:
`recall`, `remember`, `spawn`, `message` first (they already touch
`error_kind`), then the filesystem/exec/web family.

**Phase 3 — collapse the trait and delete the legacy channel.**
`execute_typed` becomes `execute`; the old `execute`/`execute_with_result`/
`execute_with_context`/`execute_with_result_and_context` four-method ladder is
deleted. `ToolExecutionContext` → `ToolContext` (§3.5). Delete
`from_legacy`, `classify_tool_error`, and the `ToolExecutionResult::from_output`
adapter; flip `disallowed-methods` to `deny`; grep must return zero.

**Phase 4 — finish_reason enum.**
Switch `LLMResponse.finish_reason` to `FinishReason` (§2.3) with
`parse_finish_reason(&str)` at the wire boundary; delete `is_error()`/
`error_detail()` in favor of `outcome()`.

Byte-stability note: `ToolError::render()` must be tuned against the existing
test suite (which asserts exact substrings like `"'task' parameter is required"`,
`"Call as …"` suffixes, `"Spawn callback not configured"`) — the migration
intentionally preserves the model-visible contract; only the transport type
changes.

---

## 3. Topic 2 — Typed host bridge

### 3.1 Reference: prime-agent's `host.request`

prime-agent implements a typed TS ↔ Python bridge over Jupyter comms. The
kernel side (`prime-agent-runtime/src/rlm/__init__.py`):

- `HOST_COMM_TARGET = "host.request"` — one comm target for all request types.
- `host_request(request_type, payload)` opens a **fresh one-shot comm**,
  sends `data={**payload, "type": request_type}` (type last so a payload `type`
  key cannot reroute), and awaits a future resolved by `comm.on_msg`.
- Reply protocol is a **discriminated union**: `{status: "ok", ...data}` or
  `{status: "error", error: msg}`; error ⇒ `RuntimeError` on the kernel side.
- Typed wrappers (`run`, `find_models`, `list_subagents`, …) validate the
  reply shape and build typed dataclasses — the type lives in the wrappers,
  the wire stays JSON.

The host side (`dist/core/kernel/index.js`): `handleCommMessage` routes
`comm_open`/`comm_msg` for `host.request` to `startHostRequestFromComm`, which
dedupes by comm id, runs `hostHandlers[type]` in an async task, and replies
`{status:"ok", ...result}` / `{status:"error", error}`. `handleHostRequest`
validates `data.type`, looks up the handler **registry**, injects
`cellSourceCode`, and throws "not available in this session" for unknown types.
Handlers are registered in one place
(`agent-session.js` `_createKernelHostHandlers`): `rlm.run`, `rlm.find_models`,
`goal.get`, `compact.run`, `rlm_heartbeat.*`, …

**Design properties worth copying:**
1. Type-tagged request envelope (`type` is the discriminator).
2. Discriminated reply envelope (`status: ok|error`) — one error channel for
   the whole bridge.
3. Per-request lifecycle (one comm per call; no shared mutable state).
4. Strict validation **on both sides** (host validates payload; kernel
   validates reply into typed dataclasses).
5. Registry-based dispatch (OCP): new capability = new type string + one
   handler registration; dispatcher is untouched.
6. Route key placed last in the payload so data cannot spoof the route.

### 3.2 Mapping to nanobot-rs

nanobot already has the transport (`code_execution.rs` UDS JSON-lines RPC,
`src/agent/tools/code_execution.rs:17-34`). What it lacks is the **typed
protocol** — its callbacks are positional closures with ad-hoc arities and
`String` returns. The bridge design below is transport-agnostic (in-process
tokio mpsc today; UDS/ZeroMQ for a future Python bridge) because the wire
envelope is plain serde.

```rust
//! src/agent/host_bridge.rs — the typed host bridge (new module).
//!
//! Replaces the 8 SpawnTool callbacks + SendCallback + OutboundCallback +
//! HeartbeatCallback with one protocol: a typed request, a typed reply, a
//! registry dispatcher, and per-capability traits. Mirrors prime-agent's
//! `host.request` comm target.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Wire protocol (serde) — transport-agnostic
// ---------------------------------------------------------------------------

/// One typed request the agent-loop host can answer. The variant name is the
/// wire discriminator (`"type"`), exactly like prime-agent's `host.request`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum HostRequest {
    Spawn(SpawnRequest),
    ListSubagents(ListSubagentsRequest),
    CancelSubagent(CancelSubagentRequest),
    WaitSubagent(WaitSubagentRequest),
    CheckSubagent(CheckSubagentRequest),
    RunPipeline(RunPipelineRequest),
    RunLoop(RunLoopRequest),
    SendMessage(SendMessageRequest),
}

// ---- typed payloads (the contract that replaces positional callbacks) ----

#[derive(Debug, Clone, Serialize, Deserialize)]
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
    /// Baked in by the registry at construction; never sent by the model.
    #[serde(skip)]
    pub channel: String,
    #[serde(skip)]
    pub chat_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpawnReply {
    pub task_id: String,
}

// … ListSubagentsRequest/ListReply, CancelRequest/CancelReply,
// WaitRequest/WaitReply, CheckRequest/CheckReply, PipelineRequest/PipelineReply,
// LoopRequest/LoopReply, SendMessageRequest/SendMessageReply — plain structs.

/// Discriminated reply envelope — the single error channel of the bridge
/// (prime-agent's `{status:"ok"|"error"}` union, typed).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum HostReply {
    Ok {
        #[serde(flatten)]
        data: serde_json::Value,
    },
    Error { error: String },
}
```

### 3.3 SOLID mapping

**SRP — split `SpawnTool::execute`.** Today one method does arg extraction +
action routing + callback lookup + callback invocation + string erroring
(`spawn.rs:300-340` (the `SpawnTool` impl; `impl Tool for SpawnTool` at `spawn.rs:159` starts the lite variant)). Split into:

- `SpawnAction` (parsing, via serde derive) — one responsibility.
- `HostDispatcher::dispatch` (routing) — one match.
- `AgentHost` impls (invocation) — one method per capability.
- `ToolError` (errors) — typed.

**OCP — add an action without touching the dispatcher or the tool.** New
capability (`archive`, `notify`): add `HostRequest::Archive(ArchiveRequest)` +
`ArchiveHost` trait + register in `HostDispatcher::new`. No `match` arm in
`SpawnTool`, no new `Arc<Mutex<Option<…>>>` field, no new setter.

**LSP — trait objects replace closures.** `SpawnTool` depends on
`Arc<dyn HostBridge>`; the production impl (`AgentHost` over `SubagentManager`)
and test mocks (`MockHost`) are interchangeable. Today the 7 callbacks are
closures pinned to `SubagentManager` internals; a test must construct real
managers to exercise `SpawnTool` (see `spawn.rs` tests, which only test
unconfigured-callback paths for exactly this reason).

**ISP — per-capability traits, composed.** `HostBridge` is the union of five
small traits so a tool (or a future Python bridge client) can depend on only
what it uses:

```rust
#[async_trait]
pub trait SpawnHost: Send + Sync {
    async fn spawn(&self, req: SpawnRequest) -> Result<SpawnReply, ToolError>;
    async fn list_subagents(&self) -> Result<ListReply, ToolError>;
    async fn cancel(&self, req: CancelRequest) -> Result<CancelReply, ToolError>;
    async fn wait(&self, req: WaitRequest) -> Result<WaitReply, ToolError>;
    async fn check(&self, req: CheckRequest) -> Result<CheckReply, ToolError>;
}

#[async_trait]
pub trait PipelineHost: Send + Sync {
    async fn run_pipeline(&self, req: PipelineRequest) -> Result<PipelineReply, ToolError>;
}

#[async_trait]
pub trait LoopHost: Send + Sync {
    async fn run_loop(&self, req: LoopRequest) -> Result<LoopReply, ToolError>;
}

#[async_trait]
pub trait MessageHost: Send + Sync {
    async fn send(&self, req: SendMessageRequest) -> Result<SendMessageReply, ToolError>;
}

/// Composite capability set. Blanket impl: any type implementing all five
/// traits IS a HostBridge (DIP: depend on this, never on `SubagentManager`).
pub trait HostBridge: SpawnHost + PipelineHost + LoopHost + MessageHost {}
impl<T: SpawnHost + PipelineHost + LoopHost + MessageHost> HostBridge for T {}

/// Registry dispatcher — the single `match`, the OCP seam, and the DRY
/// replacement for 8 `Arc<Mutex<Option<…>>>` fields + 8 setters + 8
/// lock/clone/drop/await blocks. It also implements the five capability
/// traits *by delegation* (one small `impl` per trait), so the dispatcher
/// itself IS the composite `HostBridge` — `Arc<dyn HostBridge>` can be a
/// dispatcher over mocks in tests or over `AgentHost` in production.
/// (Validated by compilation + an end-to-end mock run in this research.)
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

    /// Transport entry point: envelope in, envelope out. This is what an
    /// in-process `call()` or a UDS/ZeroMQ server both invoke.
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
    pub async fn call(&self, req: HostRequest) -> Result<serde_json::Value, ToolError> {
        match self.dispatch(req).await {
            HostReply::Ok { data } => Ok(data),
            HostReply::Error { error } => Err(ToolError::Execution { message: error }),
        }
    }
}

/// One `From` impl — every capability method returns `Result<_, ToolError>`
/// and every reply envelope is produced here (DRY: no per-method envelope
/// construction).
impl<T> From<Result<T, ToolError>> for HostReply
where
    T: Serialize,
{
    fn from(r: Result<T, ToolError>) -> Self {
        match r {
            Ok(v) => HostReply::Ok {
                data: serde_json::to_value(v).unwrap_or_else(|_| serde_json::Value::Null),
            },
            Err(e) => HostReply::Error { error: e.render() },
        }
    }
}
```

**DIP — inject, don't construct.** `SpawnTool::new(host: Arc<dyn HostBridge>)`
replaces the 8 setters + `set_context` (origin channel/chat move into
`SpawnRequest`, baked by the registry — see §3.5). The wiring in
`tool_wiring.rs` becomes:

```rust
// tool_wiring.rs — DIP: build the real host once, inject everywhere.
let host: Arc<dyn HostBridge> = Arc::new(AgentHost {
    subagents: self.subagents.clone(),
    session_policies: self.session_policies.clone(),
    pipeline_provider: core.tool_runner_provider.clone()
        .unwrap_or_else(|| core.provider.clone()),
    pipeline_model: core.tool_runner_model.clone().unwrap_or_else(|| core.model.clone()),
    workspace: core.workspace.clone(),
    outbound: self.bus_outbound_tx.clone(),
    channel: channel.to_string(),
    chat_id: chat_id.to_string(),
});

let spawn_tool = Arc::new(SpawnTool::new(host.clone()));
// Local models keep the lite schema; the host is shared, not re-cloned.
if core.mode().is_local() {
    tools.register(Box::new(SpawnToolLite(spawn_tool)));
} else {
    tools.register(Box::new(ArcToolProxy(spawn_tool)));
}
```

Each closure body in today's `build_tools` (spawn, list, cancel, wait, check,
pipeline, loop — `tool_wiring.rs:112-294`) becomes one method on `AgentHost`.
That is the LSP/DIP payoff: the logic is now a named, testable type instead of
7 anonymous closures, and `SpawnTool`'s unit tests can use `MockHost`.

### 3.4 DRY

The 7 callbacks pattern repeats this boilerplate **8 times** in `spawn.rs`
and once in `message.rs`:

```rust
// spawn.rs today — repeated per action (lock, clone, drop, await, stringify)
let cb_guard = self.wait_callback.lock().await;
match cb_guard.as_ref() {
    Some(cb) => { let cb = cb.clone(); drop(cb_guard); cb(task_id, timeout).await }
    None => "Error: Wait callback not configured".to_string(),
}
```

The bridge replaces it with:

```rust
// spawn.rs after — one call, no locks, no Option, no stringify
async fn execute(&self, params: Params, _ctx: &ToolContext) -> ToolResult {
    let action: SpawnAction = serde_json::from_value(Value::Object(params.into_iter().collect()))
        .map_err(|e| ToolError::InvalidArgs { message: e.to_string() })?;
    let reply = self.host.call(action.into()).await?;
    Ok(ToolOutput { text: render_spawn_reply(&reply) })
}
```

Also DRY'd: the 7 hand-rolled `params.get("x").and_then(|v| v.as_str())`
extraction blocks become one `#[derive(Deserialize)]` struct; the
`require_str!` macro (`base.rs:19-35`) becomes serde's `#[serde(deny_unknown_fields)]`
+ required fields with `ToolError::MissingArg` produced by one mapping helper
(shared by every tool, replacing 21 copy-pasted match blocks per commit
`27dca59`).

### 3.5 Composable `ToolContext`

Today `ToolExecutionContext` (`base.rs:117-125`) has three fields and is built
by every caller (`registry.rs:622-627`, `tool_runner/mod.rs:1175-1181`). The
composable version encapsulates **all** tool dependencies:

```rust
/// Everything a tool may depend on, composed once at the registry boundary.
/// Tools depend on this type — never on globals, never on concrete managers,
/// never on `Arc<Mutex<Option<…>>>` callback slots.
#[derive(Clone)]
pub struct ToolContext {
    /// Typed host bridge (spawn/pipeline/loop/message). `None` in sandboxed
    /// registries (scripts, tests) — tools that require it fail with a typed
    /// `ToolError::Execution { message: "host not available" }` instead of
    /// `"Error: Spawn callback not configured"`.
    host: Option<Arc<dyn HostBridge>>,
    /// Progress/audit event sink.
    events: tokio::sync::mpsc::UnboundedSender<ToolEvent>,
    /// Cooperative cancellation (the existing token, unchanged).
    cancel: tokio_util::sync::CancellationToken,
    /// Correlates this call across audit + REPL.
    call_id: String,
}

impl ToolContext {
    pub fn new(
        host: Option<Arc<dyn HostBridge>>,
        events: tokio::sync::mpsc::UnboundedSender<ToolEvent>,
        cancel: tokio_util::sync::CancellationToken,
        call_id: impl Into<String>,
    ) -> Self {
        Self { host, events, cancel, call_id: call_id.into() }
    }

    pub fn host(&self) -> Result<&dyn HostBridge, ToolError> {
        self.host.as_deref().ok_or_else(|| ToolError::Execution {
            message: "host capability not available in this context".to_string(),
        })
    }

    pub fn emit(&self, event: ToolEvent) -> Result<(), ToolError> {
        self.events.send(event).map_err(|_| ToolError::Execution {
            message: "event channel closed".to_string(),
        })
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancel.is_cancelled()
    }

    pub fn cancellation_token(&self) -> &tokio_util::sync::CancellationToken {
        &self.cancel
    }

    pub fn call_id(&self) -> &str {
        &self.call_id
    }
}
```

The registry is the single builder (DRY — no more per-caller construction):

```rust
// registry.rs — the one place a ToolContext is built.
impl ToolRegistry {
    pub fn with_host(mut self, host: Option<Arc<dyn HostBridge>>) -> Self {
        self.host = host;
        self
    }

    async fn execute_inner(
        &self,
        name: &str,
        params: HashMap<String, Value>,
        event_tx: Option<tokio::sync::mpsc::UnboundedSender<ToolEvent>>,
        cancel: Option<&tokio_util::sync::CancellationToken>,
        tool_call_id: String,
    ) -> ToolResult {
        // … normalize, lookup, permission, pre-hook (unchanged) …
        let ctx = ToolContext::new(
            self.host.clone(),
            event_tx.unwrap_or_else(|| {
                let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
                tx
            }),
            cancel.map_or_else(tokio_util::sync::CancellationToken::new, |t| t.child_token()),
            // the loop passes the LLM tool_call_id down with the call
            tool_call_id,
        );
        let unwound = std::panic::AssertUnwindSafe(tool.execute(params, &ctx));
        match futures_util::FutureExt::catch_unwind(unwound).await {
            Ok(r) => r,
            Err(_) => Err(ToolError::Execution {
                message: format!("Tool '{name}' panicked during execution"),
            }),
        }
    }
}
```

`tool_runner`'s retry loop then simplifies to `registry.execute(name, params)`
(its custom channel + child token are now registry internals), and `ExecTool`'s
streaming (`shell.rs:456` `execute_with_context`) just calls `ctx.emit(...)` /
`ctx.is_cancelled()`.

### 3.6 Strict lint/audit compliance (higgs deny regime)

The proposed code is written against higgs's regime (`crates/higgs/Cargo.toml`
`[workspace.lints]`): no `unwrap`/`expect`/`panic`/indexing/`as` casts in
production code, pedantic+nursery at warn. Concretely:

- `ToolError::render()` and all capability methods use `?`/`map_err`/let-else —
  no `unwrap_or_else(|_| …)` except in the two serde fallbacks shown, which can
  be replaced by typed constructors if the strictest reading is wanted.
- `HostReply::from`'s `serde_json::to_value(...).unwrap_or_else(…)` is the one
  unavoidable fallible-serialization site; note it, or route through
  `serde_json::to_value(v)?` and push the `Ok(Null)` decision to the caller.
- No `panic!` in production paths: the registry's `catch_unwind` boundary
  (already present at `registry.rs:669-686`) is preserved — a panicking tool
  becomes `ToolError::Execution`, never a process abort.
- `indexing_slicing = "deny"`: the bridge uses no `[i]` indexing (serde
  structs + `match` everywhere; the current code's `arr[i]`-style iteration in
  `web.rs`, `pipeline.rs` etc. is migrated to iterators as a side effect).
- `shadow_*`, `try_err`, `string_add` are satisfied by construction (`let-else`
  instead of shadowing, `format!` instead of `+`, `?` instead of `Err(e)?`-style
  patterns).
- Test modules carry the sanctioned
  `#[cfg(test)] #[allow(clippy::panic, clippy::unwrap_used, clippy::indexing_slicing)]`
  (higgs's own escape hatch), so the 335 `unwrap()` sites in
  `src/agent/tools` test code do not block the flip.

### 3.7 Migration path for the bridge

**Step 1 — additive: introduce `host_bridge.rs` (§3.2) + `ToolContext` (§3.5),
wire `AgentHost` in parallel.** `build_tools` keeps the 7 callbacks for the
legacy `SpawnTool`, and additionally builds `AgentHost`; add a
`HostBridgeAdapter` that implements `SpawnHost`/`PipelineHost`/`LoopHost`/
`MessageHost` by calling the *existing* callback slots, so the dispatcher is
exercised end-to-end before any tool changes.

**Step 2 — swap `SpawnTool` internals.** `SpawnTool` holds
`Arc<dyn HostBridge>`; its `execute` parses `SpawnAction` and calls
`host.call(...)`. The adapter keeps `SpawnToolLite` and the tests working.
Delete the 7 callback fields, 7 setters, `set_context`.

**Step 3 — move logic out of closures.** Port each closure body in
`build_tools` into `AgentHost` methods (`spawn`, `list`, `cancel`, `wait`,
`check`, `pipeline`, `loop`, `message`); delete the callbacks and the adapter.
`AgentHost` is a plain struct — now unit-testable without the agent loop.

**Step 4 — transport independence.** The dispatcher is the single choke point.
Add a UDS JSON-lines server (reusing the `code_execution.rs` transport
`run_rpc_server`) that deserializes `HostRequest` and calls
`HostDispatcher::dispatch`, enabling a future Python bridge (the
`bridge/` directory today holds only ANE/WhatsApp binaries — a
`bridge/python` host client would speak this protocol). Optionally gate with
`clippy.toml` `disallowed-methods` on the old `set_*_callback` API so no new
callback-coupled tool can be written.

### 3.8 File-by-file impact map

| File | Change |
|---|---|
| `src/errors.rs` | Add `ToolError` (+`render`, `is_retryable`, `is_model_fixable`, `from_legacy`); add `ProviderError::EmptyStream`; keep `ToolErrorKind` during migration, delete at Phase 4 (after `ToolExecutionResult` retirement — still used by `classify_tool_error` via `ToolExecutionResult::failure` and `ToolError::from_legacy`) |
| `src/agent/tools/base.rs` | `Tool` trait returns `ToolResult`; `ToolOutput`; `ToolContext` replaces `ToolExecutionContext`; `ToolExecutionResult` becomes private-fields + `From<ToolResult>` |
| `src/agent/host_bridge.rs` | **new** — wire protocol, 5 capability traits, `HostBridge`, `HostDispatcher`, `HostReply` |
| `src/agent/tools/spawn.rs` | `SpawnTool::new(Arc<dyn HostBridge>)`; `SpawnAction` derive; delete 7 callbacks + `set_context` |
| `src/agent/tools/message.rs` | `SendCallback` → `MessageHost`; `MessageTool` takes host |
| `src/agent/tool_wiring.rs` | Build `AgentHost` once; inject; port 8 closure bodies into `AgentHost` methods |
| `src/agent/tools/registry.rs` | `with_host`; single `ToolContext` builder; `execute` returns `ToolResult`; catch_unwind → `ToolError::Execution` |
| `src/agent/tool_runner/mod.rs` | Retry loop uses `ToolError::is_retryable`; drop per-call context construction |
| `src/agent/agent_loop/response.rs` | `response.outcome()` replaces `is_error()`/`error_detail()` |
| `src/providers/base.rs`, `openai_compat.rs` | `FinishReason` enum; `parse_finish_reason` at wire boundary |
| `src/agent/tools/code_execution.rs` | RPC response built from `ToolResult` (`{ok, result}` / `{error}` stays Python-compatible) |
| `Cargo.toml`, `clippy.toml` | `[workspace.lints]` deny regime; `disallowed-methods` for legacy constructors; trybuild dev-dep |
| `quality-sentinel.sh` | Add greps: zero `from_output`, zero `classify_tool_error`, zero `set_*_callback` |

---

## 4. Risks and trade-offs

1. **Byte-stability of the model-visible string is the #1 constraint.** 297
   string sites and dozens of tests assert exact `"Error: ..."` substrings;
   `ToolError::render()` must reproduce them. Mitigation: `from_legacy` maps
   1:1 from the existing `classify_tool_error` output, and Phase 2 runs the
   full test suite after every tool migration.
2. **Migration size.** 168 Rust files, ~30 tools. The phased plan keeps the
   crate green at each step; the additive adapter (Phase 1) means no big-bang
   rewrite. The 335 `unwrap()` sites in tools are almost all `#[cfg(test)]` —
   the sanctioned test-module allow makes the lint flip non-blocking.
3. **`HostReply::Ok { data: Value }` is weakly typed on the wire.** The typed
   contract lives in the per-capability traits; the envelope is deliberately
   JSON-shaped for future cross-process transport. If stricter typing is
   wanted later, switch the envelope to a generic `HostReply<T>` or use
   `serde(tag)` per-reply types — the dispatcher seam contains the change.
4. **Overhead of one enum vs. string.** `ToolError` is a `String`-carrying
   enum; `render()` runs once per failure. Negligible on the LLM-dominated hot
   path; the win is debuggability (variant names in logs/metrics) and retry
   correctness.
5. **Parallel effort coupling.** The error protocol (Topic 1) must land before
   the bridge's reply envelope can be typed end-to-end (`HostReply::Error`
   carries `ToolError`). Steps 1-2 of the bridge migration only need `ToolError`
   as a name; Phase 2 of the error migration and Step 3 of the bridge migration
   can then proceed together.

## 5. References

- nanobot-rs: `src/errors.rs`, `src/agent/tools/base.rs`, `spawn.rs`,
  `message.rs`, `registry.rs`, `tool_wiring.rs`, `tool_runner/mod.rs`,
  `code_execution.rs`, `src/providers/base.rs`, `openai_compat.rs`,
  `src/agent/agent_loop/response.rs`, `src/agent/context_hygiene.rs`,
  `src/agent/audit.rs`, `clippy.toml`, `CLAUDE.md` (Quality Gates).
- prime-agent: `dist/prime-agent-runtime/src/rlm/__init__.py`
  (`HOST_COMM_TARGET`, `host_request`, typed wrappers),
  `dist/core/kernel/index.js` (`handleCommMessage`,
  `startHostRequestFromComm`, `handleHostRequest`),
  `dist/core/agent-session.js` (`_createKernelHostHandlers`),
  `dist/core/kernel/index.d.ts` (`HostRequestHandler`,
  `HostRequestHandlers`).
- higgs deny regime: `crates/higgs/Cargo.toml` `[workspace.lints]`,
  `clippy.toml` (`disallowed-methods` MLX gate),
  `crates/higgs/src/error.rs` (`ServerError` → wire-type mapping, test-module
  lint allows).
- Related nanobot research: `docs/research/2026-08-06-rlm-prime-agent-review.md`.
