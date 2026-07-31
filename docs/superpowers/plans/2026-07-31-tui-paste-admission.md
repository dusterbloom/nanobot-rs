# TUI Paste Admission Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Guarantee that every paste delivered to the TUI is either admitted as bounded, terminal-safe input or rejected atomically without stopping the application.

**Architecture:** `App` owns one context-derived paste budget and applies it at the existing shared `on_paste` boundary used by idle and streaming input. Admission clones the `TextArea`, inserts normalized text into the clone, sizes the exact prospective input, and swaps the clone into live state only after both byte and token checks pass; the outer loops refresh the stable budget from the active model's `TokenBudget`.

**Tech Stack:** Rust 2021, crossterm `Event::Paste`, ratatui/`ratatui-textarea`, existing `TokenBudget` BPE estimator, existing `TestBackend`.

## Global Constraints

- `paste_token_limit = max_context_tokens / 8`.
- A zero/unknown context window falls back to 8,192 tokens, yielding a 1,024-token paste limit.
- `paste_byte_limit = paste_token_limit * 128`, using saturating arithmetic.
- Admission measures the complete prospective textarea, including prior accepted pastes.
- Normalize `\r\n` and lone `\r` to `\n`; preserve `\n` and `\t`.
- Convert every other Unicode control character to visible lowercase `\u{...}` text.
- Rejection preserves textarea contents, cursor, prompt history, attachments, and streaming state.
- Rejection never partially inserts or silently truncates.
- Both idle and streaming paste events use the same `on_paste` hot path.
- Do not add a new module, dependency, protocol flag, or alternate input pipeline.
- Do not change assistant-response persistence or clipboard table-layout recovery.
- The invariant begins after crossterm has successfully delivered `Event::Paste(String)`.
- Preserve all unrelated dirty-worktree changes.

---

### Task 1: Atomic, Context-Bounded Paste Admission

**Files:**
- Modify: `src/tui_app/app.rs:14-24,444-610,1560-1720,5420-5660`
- Modify: `src/tui_app/mod.rs:240-310,916-990`

**Interfaces:**
- Consumes: `SwappableCore::token_budget.max_context() -> usize`.
- Produces: `App::set_paste_context_tokens(&mut self, max_context_tokens: usize)`.
- Produces: `paste_limits(max_context_tokens: usize) -> PasteLimits`.
- Produces: `normalize_paste(text: &str) -> String`.
- Preserves: `App::on_idle_event(Event) -> Action` and `App::on_streaming_event(Event) -> StreamingAction`.

- [ ] **Step 1: Run required GitNexus impact analysis before editing production symbols**

Run these upstream analyses against repository `nanobot-rs`:

```text
gitnexus_impact({target:"on_paste", direction:"upstream", repo:"nanobot-rs"})
gitnexus_impact({target:"App::new", direction:"upstream", repo:"nanobot-rs"})
gitnexus_impact({target:"event_loop", direction:"upstream", repo:"nanobot-rs"})
gitnexus_impact({target:"run_turn", direction:"upstream", repo:"nanobot-rs"})
```

Report direct callers, affected processes, and risk levels before editing. If
any result is HIGH or CRITICAL, warn the user and wait before continuing.

- [ ] **Step 2: Add the recovered production payload fixture to the existing `app.rs` test module**

Add a test-only constant near `input_text`:

```rust
const RECOVERED_TABLE_PASTE: &str = r##"no that is fine here the plan I put together

┌─┬─────────────────────────────────┬───────┬────────────────────────────────────────────────────────┐ │#│Fix │Verdict│Why │ ├─┼─────────────────────────────────┼───────┼────────────────────────────────────────────────────────┤ │1│Memory section worked examples ( │LAND │True root cause. Fixes an asymmetry with the LCM guide │ │ │context.rs:1326-1327) │ │— not a new feature. │ ├─┼─────────────────────────────────┼───────┼────────────────────────────────────────────────────────┤ │2│Empty-arg error echoes a correct │LAND │Generalizes the debugger's defect-1 hint into one │ │ │example (registry.rs:1078) │ │reactive mechanism; "the moment the model is paying │ │ │ │ │attention." │ ped? ├─┼─────────────────────────────────┼───────┼────────────────────────────────────────────────────────┤ │3│Make dedup-block message │LAND │This is defect-2 option B. Resolves the checkpoint │ │ │corrective (tool_runner/mod.rs: │ │decision. │ │ │316) │ │ │ ├─┼─────────────────────────────────┼───────┼────────────────────────────────────────────────────────┤ │5│get_skills vs get_tools │LAND │One-liner; prevents a real, repeated confusion. │ ▀▀▀▀▀▀▀ │ │disambiguator (registry.rs:1014 /│ │ │ │ │ identity) │ │"##;
```

This fixture deliberately preserves the collapsed rows, `ped?`, block glyphs,
and absent row 4 because the admission layer treats them as ordinary data.

- [ ] **Step 3: Write failing idle/streaming tests for the recovered paste**

Add tests named with the `paste_` prefix:

```rust
#[test]
fn paste_recovered_table_remains_submittable_and_renderable() {
    let mut app = App::new();
    app.set_paste_context_tokens(65_536);

    assert!(matches!(
        app.on_idle_event(Event::Paste(RECOVERED_TABLE_PASTE.to_string())),
        Action::Continue
    ));
    let turn = match app.on_idle_event(Event::Key(KeyEvent::new(
        KeyCode::Enter,
        KeyModifiers::NONE,
    ))) {
        Action::Submit(turn) => turn,
        _ => panic!("accepted table paste must remain submittable"),
    };
    assert_eq!(turn.text, RECOVERED_TABLE_PASTE);

    app.begin_turn(&turn.text);
    let rows = app.transcript_rows(1);
    assert!(!rows.is_empty());
}

#[test]
fn paste_recovered_table_during_streaming_remains_submittable() {
    let mut app = App::new();
    app.set_paste_context_tokens(65_536);
    app.begin_turn("current request");

    assert!(matches!(
        app.on_streaming_event(Event::Paste(RECOVERED_TABLE_PASTE.to_string())),
        StreamingAction::Continue
    ));
    match app.on_streaming_event(Event::Key(KeyEvent::new(
        KeyCode::Enter,
        KeyModifiers::NONE,
    ))) {
        StreamingAction::CancelAndSubmit(turn) => {
            assert_eq!(turn.text, RECOVERED_TABLE_PASTE)
        }
        _ => panic!("accepted streaming paste must remain submittable"),
    }
}
```

The full test group added in later steps must fail on current production
behavior; this fixture also protects the exact incident payload from future
rendering regressions.

- [ ] **Step 4: Write failing atomic-limit tests**

Add:

```rust
#[test]
fn paste_limit_applies_to_complete_input_and_rejects_atomically() {
    let mut app = App::new();
    app.set_paste_context_tokens(8); // one-token paste budget

    let _ = app.on_idle_event(Event::Paste("hello".into()));
    assert_eq!(input_text(&app), "hello");

    let before_input = input_text(&app);
    let before_cursor = app.input.cursor();
    let before_history = app.history.clone();
    let before_attachments = app.attachments.clone();
    let before_streaming = app.streaming;

    let _ = app.on_idle_event(Event::Paste(" world".into()));

    assert_eq!(input_text(&app), before_input);
    assert_eq!(app.input.cursor(), before_cursor);
    assert_eq!(app.history, before_history);
    assert_eq!(app.attachments, before_attachments);
    assert_eq!(app.streaming, before_streaming);
    assert!(matches!(
        app.transcript.last(),
        Some(Cell::Note(note))
            if note.contains("paste rejected")
                && note.contains("2 tokens")
                && note.contains("limit 1")
    ));
}

#[test]
fn paste_byte_preflight_rejects_before_tokenization() {
    let mut app = App::new();
    app.set_paste_context_tokens(8); // one token → 128-byte safety bound

    let _ = app.on_idle_event(Event::Paste("x".repeat(129)));

    assert_eq!(input_text(&app), "");
    assert!(matches!(
        app.transcript.last(),
        Some(Cell::Note(note))
            if note.contains("129 bytes") && note.contains("limit 128")
    ));
}

#[test]
fn paste_rejection_does_not_cancel_streaming_or_mutate_draft() {
    let mut app = App::new();
    app.set_paste_context_tokens(8);
    app.begin_turn("current request");
    app.input.insert_str("hello");

    let before_cursor = app.input.cursor();
    let before_history = app.history.clone();
    let before_attachments = app.attachments.clone();

    assert!(matches!(
        app.on_streaming_event(Event::Paste(" world".into())),
        StreamingAction::Continue
    ));
    assert_eq!(input_text(&app), "hello");
    assert_eq!(app.input.cursor(), before_cursor);
    assert_eq!(app.history, before_history);
    assert_eq!(app.attachments, before_attachments);
    assert!(app.streaming);
}
```

These tests catch removal of the prospective-input check, mutation-before-
validation, and accidental streaming cancellation.

- [ ] **Step 5: Write failing terminal-control and Unicode tests**

Add:

```rust
#[test]
fn paste_normalizes_line_endings_and_escapes_terminal_controls() {
    let mut app = App::new();
    app.set_paste_context_tokens(8_192);

    let _ = app.on_idle_event(Event::Paste(
        "a\r\nb\rc\u{1b}\0\u{85}\td".into(),
    ));
    let turn = match app.on_idle_event(Event::Key(KeyEvent::new(
        KeyCode::Enter,
        KeyModifiers::NONE,
    ))) {
        Action::Submit(turn) => turn,
        _ => panic!("normalized paste must submit"),
    };

    assert_eq!(turn.text, "a\nb\nc\\u{1b}\\u{0}\\u{85}\td");
    assert!(!turn.text.chars().any(|c| c.is_control() && c != '\n' && c != '\t'));
}

#[test]
fn paste_accepts_difficult_unicode_without_changing_it() {
    let input = "┌─表─┐ e\u{301} 👩\u{200d}💻 العربية אבגדה";
    let mut app = App::new();
    app.set_paste_context_tokens(8_192);

    let _ = app.on_idle_event(Event::Paste(input.into()));
    let turn = match app.on_idle_event(Event::Key(KeyEvent::new(
        KeyCode::Enter,
        KeyModifiers::NONE,
    ))) {
        Action::Submit(turn) => turn,
        _ => panic!("Unicode paste must submit"),
    };

    assert_eq!(turn.text, input);
    app.begin_turn(&turn.text);
    assert!(!app.transcript_rows(1).is_empty());
}
```

The normalization test must fail because current code stores raw CR, ESC, NUL,
and C1 controls.

- [ ] **Step 6: Preserve image-only paste behavior with a real-file regression test**

Add:

```rust
#[test]
fn paste_image_only_still_attaches_without_inserting_text() {
    let dir = tempfile::tempdir().unwrap();
    let image = dir.path().join("table.png");
    std::fs::write(&image, b"png").unwrap();
    let mut app = App::new();
    app.set_paste_context_tokens(8_192);

    let _ = app.on_idle_event(Event::Paste(image.display().to_string()));

    assert_eq!(input_text(&app), "");
    assert_eq!(
        app.attachments,
        vec![std::fs::canonicalize(image)
            .unwrap()
            .to_string_lossy()
            .to_string()]
    );
}
```

This catches an admission implementation that bypasses the existing
`extract_image_attachments` behavior.

- [ ] **Step 7: Run the new test group and verify RED**

Run:

```bash
cargo test --lib tui_app::app::tests::paste_ -- --nocapture
```

Expected: compilation fails because `set_paste_context_tokens` does not exist.
After adding only a temporary test-local no-op is forbidden; proceed directly
to the production implementation. The normalization and limit assertions must
be capable of failing against the old `on_paste` behavior.

- [ ] **Step 8: Add the paste budget and normalizer beside the existing TUI constants**

Import the existing estimator:

```rust
use crate::agent::token_budget::TokenBudget;
```

Add:

```rust
const UNKNOWN_CONTEXT_TOKENS: usize = 8_192;
const PASTE_CONTEXT_DIVISOR: usize = 8;
const PASTE_BYTES_PER_TOKEN: usize = 128;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PasteLimits {
    tokens: usize,
    bytes: usize,
}

fn paste_limits(max_context_tokens: usize) -> PasteLimits {
    let context = if max_context_tokens == 0 {
        UNKNOWN_CONTEXT_TOKENS
    } else {
        max_context_tokens
    };
    let tokens = (context / PASTE_CONTEXT_DIVISOR).max(1);
    PasteLimits {
        tokens,
        bytes: tokens.saturating_mul(PASTE_BYTES_PER_TOKEN),
    }
}

fn normalize_paste(text: &str) -> String {
    use std::fmt::Write;

    let mut normalized = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\r' => {
                if chars.peek() == Some(&'\n') {
                    chars.next();
                }
                normalized.push('\n');
            }
            '\n' | '\t' => normalized.push(c),
            c if c.is_control() => {
                write!(&mut normalized, "\\u{{{:x}}}", c as u32)
                    .expect("writing to String cannot fail");
            }
            c => normalized.push(c),
        }
    }
    normalized
}

fn textarea_bytes(input: &TextArea<'_>) -> usize {
    input
        .lines()
        .iter()
        .map(|line| line.len())
        .sum::<usize>()
        .saturating_add(input.lines().len().saturating_sub(1))
}
```

These functions are private to `app.rs`; do not expand the public agent API or
create another module.

- [ ] **Step 9: Add stable budget state to `App`**

Add this field near `input`:

```rust
/// Active model context size used only to derive the stable paste admission budget.
paste_context_tokens: usize,
```

Initialize it in `App::new`:

```rust
paste_context_tokens: UNKNOWN_CONTEXT_TOKENS,
```

Add:

```rust
pub(crate) fn set_paste_context_tokens(&mut self, max_context_tokens: usize) {
    self.paste_context_tokens = if max_context_tokens == 0 {
        UNKNOWN_CONTEXT_TOKENS
    } else {
        max_context_tokens
    };
}
```

The setter stores the selected model's stable maximum; it does not use current
session occupancy.

- [ ] **Step 10: Replace `on_paste` with clone-validate-swap admission**

Replace the current method with:

```rust
fn on_paste(&mut self, text: &str) {
    let limits = paste_limits(self.paste_context_tokens);
    let prospective_raw_bytes = textarea_bytes(&self.input).saturating_add(text.len());
    if prospective_raw_bytes > limits.bytes {
        self.push_note(format!(
            "paste rejected: {prospective_raw_bytes} bytes exceeds limit {}",
            limits.bytes
        ));
        return;
    }

    let normalized = normalize_paste(text);
    let mut candidate = self.input.clone();
    candidate.insert_str(&normalized);
    let candidate_text = candidate.lines().join("\n");

    if candidate_text.len() > limits.bytes {
        self.push_note(format!(
            "paste rejected: {} normalized bytes exceeds limit {}",
            candidate_text.len(),
            limits.bytes
        ));
        return;
    }

    let tokens = TokenBudget::estimate_str_tokens(&candidate_text);
    if tokens > limits.tokens {
        self.push_note(format!(
            "paste rejected: {tokens} tokens exceeds limit {}",
            limits.tokens
        ));
        return;
    }

    let (cleaned, media) = extract_image_attachments(&normalized);
    if !media.is_empty() && cleaned.trim().is_empty() {
        self.add_attachments(media);
    } else {
        self.input = candidate;
    }
}
```

The live textarea is assigned only on acceptance. All rejection branches occur
before history, attachments, or input mutation.

- [ ] **Step 11: Supply the selected model's stable context window in both outer loops**

In `event_loop`, immediately before dispatching the idle event:

```rust
let paste_context_tokens = ctx
    .core_handle
    .swappable()
    .token_budget
    .max_context();
app.set_paste_context_tokens(paste_context_tokens);
match app.on_idle_event(ev) {
```

In `run_turn`, set the value once before `app.begin_turn`:

```rust
app.set_paste_context_tokens(session.core.swappable().token_budget.max_context());
app.begin_turn(&turn.display_text());
```

This reads the same `TokenBudget` used by the active agent core. Model switches
are reflected on the next idle event; the selected model cannot change inside
one streaming `run_turn`.

- [ ] **Step 12: Run targeted tests and verify GREEN**

Run:

```bash
cargo test --lib tui_app::app::tests::paste_ -- --nocapture
```

Expected: all `paste_` tests pass, including the recovered payload, atomic
limits, terminal controls, Unicode, streaming, and image attachment.

- [ ] **Step 13: Run all TUI tests**

Run:

```bash
cargo test --lib tui_app::app::tests -- --nocapture
```

Expected: all existing and new TUI tests pass with no panic or failure.

- [ ] **Step 14: Format and inspect the exact diff**

Run:

```bash
cargo fmt --all
git diff --check
git diff -- src/tui_app/app.rs src/tui_app/mod.rs
```

Confirm only the shared admission path, runtime context wiring, and tests
changed. Do not stage or alter unrelated dirty files.

- [ ] **Step 15: Run repository-level validation**

Run:

```bash
cargo build
cargo test
```

Expected: both commands exit 0. `scripts/turn_bench.sh` is not required because
this change does not touch the agent loop, provider client, or context builder.

- [ ] **Step 16: Run required GitNexus change detection**

Stage only the two implementation files:

```bash
git add src/tui_app/app.rs src/tui_app/mod.rs
```

Then run:

```text
gitnexus_detect_changes({scope:"staged", repo:"nanobot-rs"})
```

Expected: only TUI input/render flows are affected. Review every changed symbol
and affected process; do not commit if unrelated agent/provider/session flows
appear.

- [ ] **Step 17: Commit the verified implementation**

```bash
git commit -m "fix(tui): make paste admission total"
```

After committing, refresh the GitNexus index without embeddings:

```bash
node .gitnexus/run.cjs analyze
```

Preserve the pre-existing contents of `AGENTS.md` and `CLAUDE.md` if the
analyzer refreshes their generated statistics.
