# Cua Screenshot Vision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When the `cua` tool saves a screenshot, the agent loop injects that image into the conversation as an `image_url` content part (gated on `modelCapabilities.<name>.vision`), so vision-capable models see the actual screen.

**Architecture:** One pure decision fn `cua_screenshot_candidate(tool_name, ok, vision, data, workspace) -> Option<PathBuf>` (all gates + marker parse + path confinement, fully unit-testable without a TurnContext) plus a thin IO shell `append_cua_screenshot_turn(messages, path)` in tool_engine.rs that reads the file (≤ 10 MiB) and appends a synthetic `{"role":"user","content":[text, image_url]}` turn. The call site sits at the end of `inject_tool_result` — after the tool result is durably persisted, so the image is in-memory only (never persisted to SQLite; no re-persist after injection). MIME guessing is extracted from `context.rs::_guess_mime` into a shared `guess_mime` free fn (second-use rule).

**Tech Stack:** Rust 2021, tokio, serde_json, base64 (already a dependency — used in context.rs). No new dependencies.

## Global Constraints

- Rust 2021; strict lints: `unwrap_used`, `expect_used`, `panic`, `indexing_slicing`, `as_conversions`, `format_push_string` are `deny` in non-test code (tests may unwrap — existing tool_engine tests do).
- Never fail a turn for a screenshot: every skip path returns silently (the text result already told the model the path).
- The synthetic user turn must be appended AFTER `persist_pending_protocol_messages()` and must not be persisted — do NOT call persist again after injection.
- Do not duplicate logic: `_guess_mime` is extracted to a shared `pub(crate) fn guess_mime` and both call sites use it.
- Path confinement: only inject paths under `<workspace>/cua/` (defense in depth).
- 7 pre-existing test failures (pid_file/cli, sandbox-environment artifacts) are known and unrelated — do not chase them.
- Higgs bypasses the KV prefix cache for multimodal requests, so `test_local_wire_prompt_prefix_stable_across_turns` must still pass unchanged — the feature never mutates text-only turns.

---
---

### Task 1: Extract `guess_mime` shared fn in context.rs

**Files:**
- Modify: `src/agent/context.rs` (`_guess_mime` at ~line 1744, its caller in `_build_user_content` at ~line 1585, tests at ~line 1920)

**Interfaces:**
- Produces: `pub(crate) fn guess_mime(path: &str) -> String` (free fn, same body as `_guess_mime`), and `_build_user_content` calls it. Task 2 consumes `guess_mime`.

- [ ] **Step 1: Write the failing test**

In `src/agent/context.rs`, add to `mod tests`:

```rust
#[test]
fn test_guess_mime_shared_fn() {
    // Same behavior as the former private helper, now a shared free fn.
    assert_eq!(guess_mime("photo.jpg"), "image/jpeg");
    assert_eq!(guess_mime("photo.jpeg"), "image/jpeg");
    assert_eq!(guess_mime("image.png"), "image/png");
    assert_eq!(guess_mime("anim.gif"), "image/gif");
    assert_eq!(guess_mime("pic.webp"), "image/webp");
    assert_eq!(guess_mime("img.svg"), "image/svg+xml");
    assert_eq!(guess_mime("file.bin"), "application/octet-stream");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib agent::context::tests::test_guess_mime_shared_fn`
Expected: FAIL — `guess_mime` not found (only the private `_guess_mime` exists).

- [ ] **Step 3: Extract the fn**

Replace the existing `fn _guess_mime(path: &str) -> String { ... }` (keep its exact body) with:

```rust
/// Guess a MIME type from a file extension (used for image content parts).
pub(crate) fn guess_mime(path: &str) -> String {
    let lower = path.to_lowercase();
    if lower.ends_with(".jpg") || lower.ends_with(".jpeg") {
        "image/jpeg".to_string()
    } else if lower.ends_with(".png") {
        "image/png".to_string()
    } else if lower.ends_with(".gif") {
        "image/gif".to_string()
    } else if lower.ends_with(".webp") {
        "image/webp".to_string()
    } else if lower.ends_with(".svg") {
        "image/svg+xml".to_string()
    } else {
        "application/octet-stream".to_string()
    }
}
```

Update the single caller in `_build_user_content` (the `let mime = _guess_mime(path_str);` line) to `let mime = guess_mime(path_str);`. If any test referenced `_guess_mime`, update those too (`grep -n "_guess_mime" src/agent/context.rs` after editing — expect 0 remaining references to the old name).

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --lib agent::context::tests::test_guess_mime_shared_fn`
Expected: PASS. Also `cargo test --lib agent::context` — all context tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/agent/context.rs
git commit -m "refactor(context): extract guess_mime as shared free fn"
```

---
---

### Task 2: cua screenshot candidate + append shell + wiring in tool_engine.rs

**Files:**
- Modify: `src/agent/tool_engine.rs` — add the helper fn near `inject_tool_result` (~line 1344), call it at the end of `inject_tool_result` (after the `ctx.turn_tool_entries.push(...)` block, before the closing of the fn at line 1494); add tests to the existing `mod tests`.

**Interfaces:**
- Consumes: `guess_mime` from Task 1; `ctx.core.model_capabilities.vision`; `r.tool_name`, `r.result.ok()`, `r.result.data()` from `SingleToolResult`; `ctx.messages` (Vec<Value>); `ctx.core.workspace` (PathBuf).
- Produces:
  - `fn cua_screenshot_candidate(tool_name: &str, ok: bool, vision: bool, data: &str, workspace: &Path) -> Option<PathBuf>` — pure: returns the screenshot path iff tool_name == "cua", ok, vision, the data carries a `Screenshot saved: <path>` line, and the path starts with `<workspace>/cua`. No IO.
  - `async fn append_cua_screenshot_turn(messages: &mut Vec<Value>, path: PathBuf)` — IO shell: reads the file (≤ 10 MiB), base64-encodes, appends the synthetic user message. Testable with a plain `Vec<Value>` — no TurnContext needed.
  - Call site inside `inject_tool_result`: `if let Some(path) = cua_screenshot_candidate(...) { append_cua_screenshot_turn(&mut ctx.messages, path).await; }`

- [ ] **Step 1: Write the failing tests**

Add to `mod tests` in `src/agent/tool_engine.rs`. The gate tests are pure; the async test uses a plain `Vec<Value>` + a tempdir file (no TurnContext construction — `TurnContext` has ~50 fields and no test builder):

```rust
const PNG_B64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";

fn cua_result(data: &str) -> SingleToolResult {
    SingleToolResult {
        tool_name: "cua".to_string(),
        tool_id: "tc_1".to_string(),
        arguments: HashMap::new(),
        result: crate::agent::tools::base::ToolExecutionResult::success(data.to_string()),
        duration_ms: 5,
        replay_error: None,
    }
}

#[test]
fn test_cua_screenshot_candidate_detects_marker() {
    let ws = std::path::Path::new("/tmp/ws");
    let shot = "/tmp/ws/cua/cua-tc_1.png";
    let data = format!("click OK\n\nScreenshot saved: {shot}");
    let got = cua_screenshot_candidate("cua", true, true, &data, ws);
    assert_eq!(got.as_deref(), Some(std::path::Path::new(shot)));
}

#[test]
fn test_cua_screenshot_candidate_skips_without_marker() {
    let ws = std::path::Path::new("/tmp/ws");
    let got = cua_screenshot_candidate("cua", true, true, "click OK", ws);
    assert_eq!(got, None);
}

#[test]
fn test_cua_screenshot_candidate_skips_non_cua_tool() {
    let ws = std::path::Path::new("/tmp/ws");
    let data = "Screenshot saved: /tmp/ws/cua/x.png";
    let got = cua_screenshot_candidate("read_file", true, true, data, ws);
    assert_eq!(got, None);
}

#[test]
fn test_cua_screenshot_candidate_skips_on_failure() {
    let ws = std::path::Path::new("/tmp/ws");
    let data = "Screenshot saved: /tmp/ws/cua/x.png";
    let got = cua_screenshot_candidate("cua", false, true, data, ws);
    assert_eq!(got, None);
}

#[test]
fn test_cua_screenshot_candidate_skips_without_vision() {
    let ws = std::path::Path::new("/tmp/ws");
    let data = "Screenshot saved: /tmp/ws/cua/x.png";
    let got = cua_screenshot_candidate("cua", true, false, data, ws);
    assert_eq!(got, None);
}

#[test]
fn test_cua_screenshot_candidate_skips_path_outside_cua_dir() {
    let ws = std::path::Path::new("/tmp/ws");
    let data = "Screenshot saved: /tmp/evil.png";
    let got = cua_screenshot_candidate("cua", true, true, data, ws);
    assert_eq!(got, None);
}

#[test]
fn test_cua_screenshot_candidate_skips_relative_path() {
    let ws = std::path::Path::new("/tmp/ws");
    let data = "Screenshot saved: cua/relative.png";
    let got = cua_screenshot_candidate("cua", true, true, data, ws);
    assert_eq!(got, None);
}

/// IO shell: real file ≤ 10 MiB → image turn appended with a base64
/// roundtrip. Uses a plain Vec<Value> — no TurnContext required.
#[tokio::test]
async fn test_append_cua_screenshot_turn_embeds_image() {
    let dir = tempfile::tempdir().unwrap();
    let shot = dir.path().join("cua-tc_1.png");
    let png = base64::engine::general_purpose::STANDARD.decode(PNG_B64).unwrap();
    std::fs::write(&shot, &png).unwrap();

    let mut messages: Vec<serde_json::Value> = Vec::new();
    append_cua_screenshot_turn(&mut messages, shot.clone()).await;

    let last = messages.last().unwrap();
    assert_eq!(last["role"], "user");
    let content = last["content"].as_array().unwrap();
    assert_eq!(content[0]["type"], "text");
    assert!(content[0]["text"].as_str().unwrap().contains("cua screenshot"));
    assert_eq!(content[1]["type"], "image_url");
    let url = content[1]["image_url"]["url"].as_str().unwrap();
    assert!(url.starts_with("data:image/png;base64,"), "got: {url}");
    let b64 = url.trim_start_matches("data:image/png;base64,");
    let decoded = base64::engine::general_purpose::STANDARD.decode(b64).unwrap();
    assert_eq!(decoded, png);
}

/// IO shell: file missing → nothing appended.
#[tokio::test]
async fn test_append_cua_screenshot_turn_missing_file() {
    let mut messages: Vec<serde_json::Value> = Vec::new();
    append_cua_screenshot_turn(&mut messages, std::path::PathBuf::from("/nonexistent/x.png")).await;
    assert!(messages.is_empty());
}

/// IO shell: oversized file (> 10 MiB) → nothing appended.
#[tokio::test]
async fn test_append_cua_screenshot_turn_oversized() {
    let dir = tempfile::tempdir().unwrap();
    let shot = dir.path().join("big.png");
    std::fs::write(&shot, vec![0u8; 10 * 1024 * 1024 + 1]).unwrap();

    let mut messages: Vec<serde_json::Value> = Vec::new();
    append_cua_screenshot_turn(&mut messages, shot).await;
    assert!(messages.is_empty());
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --lib agent::tool_engine::tests::test_cua_screenshot_candidate agent::tool_engine::tests::test_append_cua_screenshot_turn`
Expected: FAIL — `cua_screenshot_candidate` / `append_cua_screenshot_turn` not found.

- [ ] **Step 3: Implement the pure fn + IO shell + wiring**

Add next to `inject_tool_result` (module scope, before it):

```rust
/// Parse the cua screenshot marker and apply every gate: tool name, success,
/// vision capability, marker presence, and path confinement under
/// `<workspace>/cua/`. Pure — no IO — so the gate logic is unit-testable
/// without a TurnContext.
fn cua_screenshot_candidate(
    tool_name: &str,
    ok: bool,
    vision: bool,
    data: &str,
    workspace: &std::path::Path,
) -> Option<std::path::PathBuf> {
    if tool_name != "cua" || !ok || !vision {
        return None;
    }
    let path_str = data.lines().rev().find_map(|line| {
        line.strip_prefix("Screenshot saved: ").map(str::to_string)
    })?;
    let path = std::path::PathBuf::from(&path_str);
    // Defense in depth: only accept paths under <workspace>/cua/.
    let cua_dir = workspace.join("cua");
    if !path.starts_with(&cua_dir) {
        return None;
    }
    Some(path)
}

/// Read + embed one screenshot: read the file (≤ 10 MiB), base64-encode it,
/// and append a synthetic user turn carrying it as an `image_url` content
/// part so the model sees the screen. The image is in-memory only — the
/// caller appends this AFTER the tool result is durably persisted and never
/// re-persists. Every skip path is silent: the tool's text result already
/// told the model the path.
async fn append_cua_screenshot_turn(messages: &mut Vec<serde_json::Value>, path: std::path::PathBuf) {
    let path_str = path.to_string_lossy().to_string();
    let Ok(bytes) = tokio::fs::read(&path).await else {
        return;
    };
    const MAX_EMBED_BYTES: usize = 10 * 1024 * 1024;
    if bytes.len() > MAX_EMBED_BYTES {
        return;
    }
    let mime = crate::agent::context::guess_mime(&path_str);
    let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
    messages.push(serde_json::json!({
        "role": "user",
        "content": [
            {"type": "text", "text": format!("[cua screenshot: {path_str}]")},
            {"type": "image_url", "image_url": {"url": format!("data:{mime};base64,{b64}")}}
        ]
    }));
}
```

Wire the call at the END of `inject_tool_result` — after the `ctx.turn_tool_entries.push(...)` block and before the closing brace of the fn (line ~1493, right before the `// NOTE: response-boundary arming...` comment):

```rust
    // Cua screenshot vision: append the image as a user turn (in-memory only).
    // The image is NOT persisted — this runs after the tool result's own
    // persistence and nothing re-persists it.
    if let Some(path) = cua_screenshot_candidate(
        &r.tool_name,
        r.result.ok(),
        ctx.core.model_capabilities.vision,
        r.result.data(),
        &ctx.core.workspace,
    ) {
        append_cua_screenshot_turn(&mut ctx.messages, path).await;
    }
```

Imports: `base64` is a crate dependency (used in context.rs) — reference it fully-qualified (`base64::engine::...`) as the tests do; no import change needed unless the module prefers a `use`. `tokio::fs` is available (tokio full features). `guess_mime` is `pub(crate)` in `crate::agent::context` (Task 1).

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --lib agent::tool_engine::tests::test_cua_screenshot_candidate agent::tool_engine::tests::test_append_cua_screenshot_turn`
Expected: all 10 PASS (7 pure gates + 3 IO-shell).

- [ ] **Step 5: Full suite + prefix regression**

Run: `cargo test`
Expected: full suite green (2767 passed / 7 known pre-existing sandbox failures — verify nothing new), including `test_local_wire_prompt_prefix_stable_across_turns`.

- [ ] **Step 6: Clippy**

Run: `cargo clippy --lib 2>&1 | grep -E "tool_engine|context.rs"`
Expected: no NEW findings in the changed files (baseline is 3749 warnings + 13 pre-existing errors in untouched files; do not fix those).

- [ ] **Step 7: Commit**

```bash
git add src/agent/tool_engine.rs
git commit -m "feat(agent): inject cua screenshot as vision turn when model is vision-capable"
```

---
---

### Task 3: README note

**Files:**
- Modify: `README.md` (cua section, after the config block note)

- [ ] **Step 1: Add the sentence**

In the `## Cua Driver — local desktop computer-use` section, at the end of the `### Configuration` prose (after the `Screenshots are saved to a file and their path returned; feeding images back to vision models is a planned follow-up.` sentence), replace that last sentence with:

```markdown
Screenshots are saved to a file and their path returned; when the active
model is vision-capable (`modelCapabilities.<name>.vision`), the image is
also fed back to the model as an `image_url` content part so it can see
the screen. Images are in-memory only — the path is what persists to
session history.
```

- [ ] **Step 2: Verify**

Run: `grep -n "vision-capable" README.md` — expect exactly 1 match.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document cua screenshot vision injection"
```

---
---

### Task 4: End-to-end validation

**Files:** none (verification only).

- [ ] **Step 1: Full build + tests**

Run: `cargo build` and `cargo test`
Expected: clean build; full suite green (2767 passed / 7 known pre-existing sandbox failures, nothing new).

- [ ] **Step 2: Clippy (feature files only)**

Run: `cargo clippy --lib 2>&1 | grep -E "tool_engine|context.rs"` — no new findings.

- [ ] **Step 3: Spec-coverage checklist**

Open `docs/superpowers/specs/2026-08-19-cua-screenshot-vision-design.md` and confirm each component has a task:
- `cua_screenshot_candidate` + `append_cua_screenshot_turn` in inject_tool_result → Task 2 ✅
- gates: tool_name == cua, ok, caps.vision, marker, path confinement, ≤10 MiB → Task 2 ✅
- synthetic user turn `[text + image_url]`, in-memory only (after persist) → Task 2 ✅
- `guess_mime` extraction (second-use) → Task 1 ✅
- README sentence → Task 3 ✅
- images never persisted to SQLite → Task 2 (call placed after persist; no re-persist) ✅
- prefix-stability regression passes unchanged → Task 2 Step 5 ✅

- [ ] **Step 4: Final review**

Run: `git log --oneline -6` — expect the 3 feature commits from Tasks 1-3 plus docs. Review the diff with `git diff HEAD~3 --stat`.
