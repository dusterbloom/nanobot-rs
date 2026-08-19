# Tool-result handle wire design

## Problem

Ordinary tool results in the current refactoring branch are inline whenever
they fit under `TOOL_RESULT_REPLAY_MAX_BYTES`. A result around 7 KB therefore
lands in the session message, is replayed on every turn, and can grow the live
prompt even though the exact bytes already have a durable SQLite home.

## Decision

Every ordinary tool result is persisted first and rendered as a deterministic
`TOOL_RESULT_HANDLE v1` receipt. The receipt contains only stable metadata,
the tool-call id, a bounded deterministic excerpt, and the retrieval affordance.
The exact result remains in `tool_results` and is never placed in the ordinary
tool-result message.

Explicit retrieval tools (`recall_tool_result`, `slice_tool_result`, and
`search_tool_result`) are the exception: their outputs are already bounded
continuations and may be rendered as bounded excerpts. A recalled full body is
stored under its own call id before it is shaped, so slice/search can recover it
without replaying the body.

The store-then-render operation is the single provider-facing chokepoint. If
SQLite cannot prove that the exact bytes were stored, the turn aborts with an
infrastructure error; it must not emit a raw fallback or a handle pointing at
missing bytes. Handles are deterministic from the stored bytes and arguments,
so live prompts and SQLite-reloaded history remain cache-stable.

Legacy session messages are upgraded lazily at history replay: an old ordinary
raw body is stored (or paired with its existing stash row) and projected as the
same handle. Protocol/infrastructure receipts such as response-boundary errors
are excluded from this upgrade because changing them would itself break the
prefix cache.

## Verification

- ordinary results of 0, 2, 7.4 KB, 8 KB, and 172 KB all reach the model as
  handles;
- exact bytes remain recoverable by the existing retrieval tools;
- explicit retrieval responses remain bounded and form valid continuations;
- provider-facing tool messages contain no ordinary raw body;
- stash failure never falls back to raw prompt content.
- legacy medium bodies stop replaying raw after the first history load.
