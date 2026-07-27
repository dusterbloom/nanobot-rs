# Higgs Literal ChatML Delimiter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve retained-session prefix reuse when user or tool content
contains the literal `<|im_end|>` string.

**Architecture:** Keep the existing rendered-text continuation path, but parse
each ChatML frame using its next start marker or EOF as the boundary and select
the rightmost structural closer. Ambiguous embedded start markers remain
fail-closed.

**Tech Stack:** Rust 2021, Higgs simple engine, Qwen ChatML templates.

## Global Constraints

- Modify only `crates/higgs-engine/src/simple.rs`.
- Preserve all unrelated dirty PFlash changes already present in that file.
- Add no flag, alternate renderer, or session-cache representation.
- Run Cargo tests and builds with `--release`.

---

### Task 1: Frame-Aware ChatML Closer

**Files:**
- Modify: `/Users/peppi/Dev/higgs-nightly/crates/higgs-engine/src/simple.rs:776`
- Test: `/Users/peppi/Dev/higgs-nightly/crates/higgs-engine/src/simple.rs:12900`

**Interfaces:**
- Consumes: `rendered_message_segments(text: &str)`.
- Produces: unchanged `Option<Vec<RenderedMessageSegment<'_>>>`, with literal
  end markers retained inside `text_without_end`.

- [ ] **Step 1: Add the failing user-content and adjacent-marker tests**

```rust
#[test]
fn boundary_delta_accepts_literal_im_end_in_covered_user_content() {
    let retained = "<|im_start|>system\nsys<|im_end|>\n\
        <|im_start|>user\nliteral <|im_end|> still data<|im_end|>\n\
        <|im_start|>assistant\nanswer";
    let full = "<|im_start|>system\nsys<|im_end|>\n\
        <|im_start|>user\nliteral <|im_end|> still data<|im_end|>\n\
        <|im_start|>assistant\nstored<|im_end|>\n\
        <|im_start|>user\nnext<|im_end|>\n<|im_start|>assistant\n";
    assert_eq!(
        message_boundary_delta(retained, full),
        Some("<|im_end|>\n<|im_start|>user\nnext<|im_end|>\n<|im_start|>assistant\n")
    );
}

#[test]
fn boundary_delta_uses_second_of_adjacent_im_end_markers() {
    let retained = "<|im_start|>user\nliteral ends <|im_end|><|im_end|>";
    let full = "<|im_start|>user\nliteral ends <|im_end|><|im_end|>\n\
        <|im_start|>assistant\n";
    assert_eq!(
        message_boundary_delta(retained, full),
        Some("\n<|im_start|>assistant\n")
    );
}
```

- [ ] **Step 2: Add the failing appended-tool-result test**

```rust
#[test]
fn boundary_delta_accepts_literal_im_end_in_new_tool_result() {
    let retained =
        "<|im_start|>user\nread<|im_end|>\n<|im_start|>assistant\ncall";
    let full =
        "<|im_start|>user\nread<|im_end|>\n<|im_start|>assistant\nstored<|im_end|>\n\
         <|im_start|>tool\nquoted <|im_end|> marker<|im_end|>\n\
         <|im_start|>assistant\n";
    assert_eq!(
        message_boundary_delta(retained, full),
        Some(
            "<|im_end|>\n<|im_start|>tool\nquoted <|im_end|> marker<|im_end|>\n\
             <|im_start|>assistant\n"
        )
    );
}
```

- [ ] **Step 3: Run the focused release test and confirm RED**

Run:

```bash
cargo test --release -p higgs-engine --lib boundary_delta -- --nocapture
```

Expected: the three new literal-delimiter tests fail because the first
`<|im_end|>` is selected.

- [ ] **Step 4: Implement the minimal frame-aware parser**

Inside `rendered_message_segments`, compute the next start/EOF frame boundary
and replace the first-closer lookup with:

```rust
let next_start = text
    .get(body_start..)?
    .find(IM_START)
    .map(|relative| body_start + relative);
let frame_end = next_start.unwrap_or(text.len());
let frame = text.get(body_start..frame_end)?;
let structural_end = frame
    .rmatch_indices(IM_END)
    .map(|(relative, _)| body_start + relative)
    .find(|candidate| {
        let after = candidate + IM_END.len();
        text.get(after..frame_end)
            .is_some_and(|tail| tail.trim().is_empty())
    });
```

When `structural_end` is absent, return a partial final segment only if
`next_start.is_none()`; otherwise return `None`. Retain the existing
`RenderedMessageSegment` fields and message compatibility rules.

- [ ] **Step 5: Run focused tests and the release build**

```bash
cargo test --release -p higgs-engine --lib boundary_delta -- --nocapture
cargo build --release -p higgs
```

Expected: all boundary tests pass and the Higgs release binary builds.

- [ ] **Step 6: Reproduce the two-turn cache case**

Restart the inspectable `higgs-nightly` tmux session with the rebuilt release
binary, send a first turn containing a literal `<|im_end|>`, then append one
ordinary turn with the same Higgs session ID. Require `continued=true`,
`prefill_saved > 0`, and no `ExactBootstrap` token mismatch.
