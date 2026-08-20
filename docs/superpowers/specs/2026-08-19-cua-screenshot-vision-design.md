# Design: Cua Screenshot Vision — Loop-Injected Image Turn

Date: 2026-08-19
Status: Approved (in-memory-only images; 10 MiB embed cap)

## Summary

Give the agent *vision* of the host desktop: when the `cua` tool saves a
screenshot, the agent loop injects that image into the conversation as an
`image_url` content part so vision-capable models see the actual screen,
not just the file path. This is the deferred vision workstream from
`2026-08-19-cua-driver-design.md`, now unblocked by the higgs nightly
vision tower.

Decisions locked in with the user:

- **Scope**: cua tool-result screenshots only. No general tool-image
  sidecar, no user-attachment rework.
- **Injection point**: the agent loop (`inject_tool_result` in
  `tool_engine.rs`), not the cua tool, not the result types.
- **Shape**: a synthetic user turn appended right after the cua tool
  result, carrying `[text + image_url]` content parts.
- **Images are in-memory only**: the base64 image lives only in the
  current turn's in-memory message array and is never persisted to the
  SQLite session history (the path text is what persists).
- **Size cap**: 10 MiB decoded bytes per embedded screenshot; larger
  images are skipped with a note (the text result already tells the model
  the path).

## Background

### higgs nightly vision tower (the enabler)

The higgs `nightly` branch merged first-class vision support
(`84af237d5 Merge feat/vision-models into nightly`):

- **OpenAI-compat wire**: `{"type": "image_url", "image_url": {"url":
  "data:image/png;base64,..."}}` content parts (base64 data URIs only;
  no HTTP fetching; 20 MiB byte cap; 4096px max dimension via
  `MediaExtractor` in `crates/higgs/src/media.rs`).
- **Anthropic**: native image blocks wired too.
- **Models**: LLaVA-Qwen2, Gemma 3/4 (pan-and-scan), Qwen-VL families,
  native preprocessing, batch-engine VLM routing.
- **Gating**: `engine.is_vlm()`; `disable_vision` per-model escape hatch;
  strict 400 for images on text models.

**Critical for this design**: higgs **bypasses the KV prefix cache for
multimodal requests entirely** (`crates/higgs-engine/src/simple.rs:4556`
"Skip prefix caching for multimodal requests: different images...",
`batch_engine.rs:1100/1746/1930`). Image-bearing turns never enter the
radix cache, so the nanobot byte-prefix stability contract (pinned by
`test_local_wire_prompt_prefix_stable_across_turns`) is untouched by this
feature — no TAIL-block gymnastics are needed.

### nanobot current state (verified)

- `ModelCapabilities.vision: bool` exists with `is_vision_model()` name
  matching (vision/vl/llava/pixtral/gpt-4o/gemini/claude-3/qwen-vl markers
  + qwen3.5/3.6 `a3b` MoE family) and the
  `modelCapabilities.<name>.vision` config override — the capability gate
  is already built.
- User-message images already flow end-to-end: `_build_user_content`
  (`context.rs:1572`) builds `image_url` data-URI parts from media
  attachments; `prepare_context.rs` passes `media_paths` into
  `build_messages`; `translate_user_content` (`anthropic.rs:198`) passes
  array content through; `LocalProtocol::render` (`protocol.rs:130`) pushes
  user content as-is.
- The cua tool (merged) returns `"Screenshot saved: <path>"` in its result
  text, where `<path>` is `<workspace>/cua/cua-<call_id>.png` (call_id
  sanitized to `[A-Za-z0-9_-]`).

## Architecture

```
cua tool → "Screenshot saved: <workspace>/cua/cua-<id>.png" (text tool result)
                    │
                    ▼
inject_tool_result (tool_engine.rs:1344) — after the tool result is added
    │ maybe_inject_cua_screenshot(ctx, r):
    │   1. r.tool_name == "cua"?
    │   2. ctx.core.model_capabilities.vision == true?
    │   3. parse "Screenshot saved: <path>" marker from r.result.data()
    │   4. file exists, ≤ 10 MiB decoded?
    │   5. read → base64 → append synthetic user turn to ctx.messages:
    │        {"role":"user","content":[
    │          {"type":"text","text":"[cua screenshot: <path>]"},
    │          {"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}]}
    │
    ▼
next provider call renders the array (LocalProtocol passes arrays through;
OpenAI-compat and Anthropic translate_user_content already handle arrays)
```

## Components

### 1. `maybe_inject_cua_screenshot` — `src/agent/tool_engine.rs`

A private async helper called from `inject_tool_result` immediately after
the tool result is added to `ctx.messages` (after
`record_tool_post_execute` succeeds, before the function returns).

Signature: `async fn maybe_inject_cua_screenshot(ctx: &mut TurnContext, r: &SingleToolResult)`

Logic (all conditions must hold):

1. `r.tool_name == "cua"`.
2. `ctx.core.model_capabilities.vision`.
3. `r.result.ok()` — only inject on success.
4. Parse the marker from `r.result.data()`: the last line starting with
   `Screenshot saved: `; extract the path. Absent → return.
5. Validate the path is inside the workspace cua dir (`<workspace>/cua/`)
   — reuse the same confinement the tool already enforces; reject anything
   else (defense in depth).
6. `fs::read` the file; if `bytes.len() > 10 * 1024 * 1024` → return
   (skip; the text result already told the model the path).
7. Guess MIME from the extension (`image/png`/`image/jpeg`/`image/webp`;
   default `image/png`) — **reuse** `ContextBuilder::_guess_mime` from
   `context.rs` (second-use rule: extract it to a shared free fn
   `pub(crate) fn guess_mime(path: &str) -> String` in `context.rs` and
   have both call sites use it, rather than duplicating the match).
8. Append to `ctx.messages`:

```rust
ctx.messages.push(json!({
    "role": "user",
    "content": [
        {"type": "text", "text": format!("[cua screenshot: {path}]")},
        {"type": "image_url", "image_url": {"url": format!("data:{mime};base64,{b64}")}}
    ]
}));
```

Notes:
- The synthetic turn is **in-memory only** — it is added to `ctx.messages`
  after the persistence of the tool result, and this design does NOT
  persist it to the session DB. `persist_pending_protocol_messages` must
  not be re-run after injection (or the image would be written to SQLite);
  the helper is placed after the persist call, so nothing persists it.
- The local protocol's "end on user" invariant is satisfied — the
  synthetic user turn is the last message.
- No config changes: the existing `modelCapabilities.<name>.vision` is the
  gate.

### 2. `src/agent/tools/cua.rs` — unchanged

The tool already emits the `Screenshot saved: <path>` marker and confines
screenshots to `<workspace>/cua/`.

### 3. `src/agent/context.rs` — one small extraction

Extract `_guess_mime` to a shared free fn `pub(crate) fn guess_mime(path: &str) -> String`
and update its existing caller; the vision helper calls the same fn (no
duplication). `_build_user_content` is the shape reference; the helper
reuses the same `image_url` part structure inline.

### 4. README — one sentence

In the cua README section: note that screenshots are fed back to the model
when the active model is vision-capable (`modelCapabilities.<name>.vision`),
and that images are in-memory only (the path persists).

## Error handling & safety

- File missing / unreadable / oversized / non-cua / non-vision → skip
  injection silently; never fail the turn for a screenshot (the text
  result already gave the model the path).
- Path confinement: only accept paths under `<workspace>/cua/` (defense in
  depth on top of the tool's own sanitization).
- No new config surface; gate is the existing `caps.vision`.
- Images never persisted to SQLite (memory-only); session history stays
  small and text-only.

## Explicitly out of scope

- General tool-image sidecar for arbitrary tools — rejected (cua only).
- User-message attachment rework — existing path already works.
- Multi-image turns, image streaming, image-in-tool-result messages —
  not now.
- Anthropic `image` block rendering — existing `translate_user_content`
  passes `image_url` parts through (verified); if a cloud Anthropic vision
  gap surfaces in testing, that is a follow-up.

## Testing

- Unit (`tool_engine.rs` tests): marker detected → user turn appended with
  `image_url` part and base64 roundtrip (decode == file bytes); marker
  absent → untouched; `vision == false` → untouched; file missing →
  untouched; oversized (> 10 MiB) → untouched; non-cua tool → untouched;
  path outside workspace cua dir → untouched.
- Regression: `test_local_wire_prompt_prefix_stable_across_turns` still
  passes (text turns unchanged); full suite green (7 known pre-existing
  sandbox failures).
- Manual: with cua-driver installed + a vision-capable higgs model, run
  `cua` screenshot then ask the model what's on screen.
