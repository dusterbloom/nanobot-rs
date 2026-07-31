# TUI Paste Admission Design

## Goal

Establish this invariant for the full-screen TUI:

> For every `Event::Paste(String)` delivered by crossterm, the application
> either accepts a bounded, terminal-safe representation of the paste or
> rejects it without mutating user input. Paste content must never panic,
> terminate, or wedge the TUI.

This invariant covers both idle input and typeahead while an assistant turn is
streaming.

The guarantee begins at the `Event::Paste(String)` boundary. Crossterm has
already allocated the event string before application code receives it, so
protecting against allocation failure inside crossterm itself is outside this
change.

## Admission Policy

Each model has one stable paste-token limit:

```text
paste_token_limit = max_context_tokens / 8
```

Examples:

- 8,192-token context → 1,024-token paste limit
- 65,536-token context → 8,192-token paste limit

The limit is based on the selected model's maximum context window and does not
shrink as session history grows. Normal context management remains responsible
for accumulated history.

If the runtime cannot report a nonzero context window, use an 8,192-token
context fallback, producing a 1,024-token paste limit. This keeps unknown
models usable without creating an unbounded path.

Admission measures the prospective complete input:

```text
existing input + normalized paste
```

Repeated individually-small pastes therefore cannot bypass the limit.

## Data Flow

Paste handling remains in `src/tui_app/app.rs`; this change does not introduce
a new module.

Both idle and streaming event handlers route `Event::Paste` through the same
admission function:

1. Receive the model's stable context-window size from the outer TUI loop.
2. Apply a cheap byte-size preflight to the prospective input.
3. Normalize the paste into terminal-safe text.
4. Estimate tokens for the prospective complete input.
5. Reject atomically when either bound is exceeded.
6. Otherwise insert the normalized text into the existing `TextArea`.

The byte preflight is a defensive CPU and memory-work bound before token
estimation:

```text
paste_byte_limit = paste_token_limit * 128
```

The multiplier accommodates highly compressible BPE input such as long
whitespace runs while still placing a finite ceiling on normalization and token
estimation. It is a safety bound rather than a second context policy. The token
limit remains the normal reported limit.

The application must update the supplied context-window size whenever model
selection or snapshot restoration changes the active model.

## Terminal-Safe Normalization

Accepted text preserves ordinary Unicode, box-drawing characters, tabs, and
newlines.

- Normalize `\r\n` and lone `\r` to `\n`.
- Preserve `\n` and `\t`.
- Convert all other Unicode control characters to visible `\u{...}` text.

Raw ESC, NUL, C0, or C1 control characters must never enter the textarea,
transcript renderer, terminal backend, prompt history, or outbound message.
Visible escaping preserves evidence of what was pasted without allowing pasted
data to act as terminal protocol.

Normalization happens before sizing because the normalized representation is
the text the application stores and submits.

## Rejection Semantics

Rejection is atomic:

- Keep the prior textarea contents and cursor unchanged.
- Keep prompt history unchanged.
- Keep attachments unchanged.
- Do not submit or cancel an active turn.
- Keep the event loop running.
- For token-budget rejection, show the estimated requested and allowed token
  counts.
- For byte-preflight rejection, show the requested and allowed byte counts
  without running token estimation.

No partial insertion or silent truncation is allowed.

## Rendering and Submission

Once admitted, pasted text follows the ordinary input path. Rendering and
submission do not gain table-specific, Unicode-specific, or paste-specific
fallback branches.

This keeps the production path singular:

```text
paste event → admission → TextArea → ordinary submit → agent
```

The recovered box-table payload from session `20260731_074806_8e1338` is a
regression fixture. Although it arrived with collapsed rows and stray visible
glyphs, those are valid user data and must remain harmless.

## Testing

Tests exercise public application behavior rather than normalization internals
alone.

1. Replay the recovered table through idle `Event::Paste`, submit it, and
   render the resulting user cell at narrow terminal widths.
2. Replay the same payload through the streaming typeahead path and verify
   Enter returns `CancelAndSubmit`.
3. Verify exact-limit acceptance and one-token-over rejection.
4. Verify multiple small pastes cannot exceed the complete-input limit.
5. Verify an oversized paste leaves input, cursor, history, attachments, and
   streaming state unchanged.
6. Verify CR/LF normalization and visible escaping of ESC, NUL, C0, and C1
   controls.
7. Exercise representative difficult Unicode: box drawing, combining marks,
   zero-width joiners, emoji sequences, bidirectional text, and long unbroken
   strings.
8. Render accepted fixtures at widths including one column to prove wrapping
   remains total.

The regression test must fail against the current implementation for the
missing admission and control-character guarantees before production code is
changed.

## Non-Goals

- Recovering malformed table layout from clipboard text.
- Persisting assistant final text.
- Dynamically shrinking paste limits based on resident session history.
- Accepting unbounded input.
- Catching panics after state mutation.
- Replacing crossterm's terminal parser or allocation behavior.
