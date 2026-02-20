# Streaming Rewrite: Live Markdown Rendering

## Status Quo

The current flow during an LLM response:

1. User input is erased and re-rendered as a styled box
2. A **progress line** replaces itself on one line: `И 42w 12w/s ... last 50 chars`
3. Tool events appear inline as cyan lines, overwriting the progress line
4. When streaming ends, the progress line is erased
5. The **entire response** is re-rendered through `render_response()` (termimad + syntect)

**Problems:**
- User cannot read the response as it streams -- only sees a word count + tail preview
- The erase-and-rerender of the full response is a single atomic dump, losing the "streaming" feel
- Only 1 line is erased (the trailing `\r\n`), but tool event lines stay -- this means the final render appends BELOW the tool events, creating a disjointed layout where tool call boxes sit above the formatted response
- The progress line was designed to prevent "scrollback pollution" but it over-corrected: it hides ALL content

## Design Goals

1. **Readable streaming** -- user sees actual text as it arrives, not a progress counter
2. **No full-response erase-rerender** -- eliminate the pattern of hiding content then dumping it all at once
3. **Clean tool event separation** -- tool calls visually distinct from text, but integrated in the flow
4. **Code blocks still get syntax highlighting** -- via targeted block-level rerender
5. **Voice mode unaffected** -- TTS accumulator continues to work on raw deltas
6. **Provenance/claim verification still works** -- applied as a post-pass on the final text

## Constraints

- **termimad** needs complete prose blocks to format (headers, bold, lists, links)
- **syntect** needs complete code blocks to highlight
- Terminal has no "go back and reformat" without cursor movement + erase
- We send `String` deltas through an unbounded channel -- no structured tokens
- The delta stream mixes text and thinking deltas (thinking prefixed with ANSI dim)

## Architecture: Incremental Line Renderer

Replace the progress-line print task with an **incremental line renderer** that emits formatted lines as they complete.

### Core Idea

Buffer incoming deltas into a **line accumulator**. When a newline arrives, the completed line is classified and rendered immediately:

```
Delta arrives: "Here is a **bold" 
  -> buffer: "Here is a **bold"
  -> no newline yet, show as plain text on current line (overwritable)

Delta arrives: "** word\nAnd next"
  -> completes line: "Here is a **bold** word"
  -> render with inline markdown (bold applied)
  -> println!() -- line is now permanent in scrollback
  -> buffer: "And next" (partial, shown on current overwritable line)
```

### Line Classification State Machine

```
enum StreamState {
    Prose,              // default -- render lines through inline markdown
    CodeBlock {         // inside ``` ... ```
        lang: String,
        lines: Vec<String>,
    },
    Thinking,           // thinking deltas (dim, not rendered with markdown)
}
```

**Transitions:**
- `Prose` + line starts with `` ``` `` -> enter `CodeBlock`, print dim header
- `CodeBlock` + line starts with `` ``` `` -> flush block through syntect, transition to `Prose`
- Thinking delta detected (ANSI prefix) -> `Thinking` state, print dim

### What Gets Rendered When

| Content | During streaming | After completion |
|---------|-----------------|-----------------|
| Plain text | Inline markdown (bold, italic, inline code) via simple regex | No re-render needed |
| Headings | Colored on detection (`## ` prefix) | No re-render |
| Bullet lists | Indented + bullet character | No re-render |
| Code block lines | **Syntax-highlighted via syntect** as each line completes (HighlightLines is stateful) | Footer line printed, no re-render |
| Diff blocks | Colored via syntect Diff syntax or line-level `+`/`-` coloring | No re-render |
| Tool event diffs | Diff-detected output gets `+` green / `-` red in tool box | No re-render |
| Tool events | Printed on own line with cyan prefix (unchanged) | No re-render |
| Tables | Printed raw as they arrive | Optional: rerender aligned at end (rare case) |

### Code Block: Line-by-Line Syntax Highlighting (No Rerender)

Syntect's `HighlightLines` is stateful -- it carries parser state across lines.
This means we can highlight each code line the moment it arrives, with no
block-level rerender at all:

```
Streaming a code block:
  ─── rust ──────────        <- printed when ``` rust detected
    let x = 42;              <- highlighted via syntect as line completes
    println!("{}", x);       <- same, highlighter carries state from previous line
  ──────────────────         <- printed when closing ``` detected (just 1 println)
```

Implementation:

```rust
// When opening ``` lang is detected:
let syntax = SYNTAX_SET.find_syntax_by_token(&lang)
    .or_else(|| SYNTAX_SET.find_syntax_by_extension(&lang))
    .unwrap_or_else(|| SYNTAX_SET.find_syntax_plain_text());
let theme = &THEME_SET.themes["base16-ocean.dark"];
let highlighter = HighlightLines::new(syntax, theme);
// Store highlighter in CodeBlock state

// Each completed line inside the block:
let ranges = highlighter.highlight_line(line, &SYNTAX_SET)?;
let colored = as_24_bit_terminal_escaped(&ranges, false);
println!("  {}\x1b[0m", colored);  // permanent, no erase needed
```

The existing `render_code_block()` already uses this exact per-line API.
We just move it from post-hoc batch to real-time streaming. Zero rerender.

### Diff Highlighting

Diffs get proper coloring through two paths:

1. **Fenced diff blocks** (` ```diff `): Syntect has a built-in `Diff` syntax
   definition. `find_syntax_by_token("diff")` returns it. Each line gets
   highlighted automatically: `+` green, `-` red, `@@` cyan. No special case.

2. **Tool output diffs** (edit_file results in tool boxes): Detect diff-like
   content in `result_data` and apply line-level coloring:
   ```rust
   fn colorize_diff_line(line: &str) -> String {
       if line.starts_with('+') { format!("\x1b[32m{}\x1b[0m", line) }
       else if line.starts_with('-') { format!("\x1b[31m{}\x1b[0m", line) }
       else if line.starts_with("@@") { format!("\x1b[36m{}\x1b[0m", line) }
       else { line.to_string() }
   }
   ```
   Applied inside the tool event box renderer when output looks like a diff
   (heuristic: >50% of lines start with `+`, `-`, `@@`, or ` `).

### Inline Markdown Renderer

A lightweight function that handles completed prose lines:

```rust
fn render_inline_markdown(line: &str) -> String {
    // Applied in order:
    // 1. Heading: ^#{1,6} \s -> colored + bold
    // 2. Bullet: ^[\s]*[-*] \s -> indent + bullet char
    // 3. Numbered list: ^[\s]*\d+\. \s -> indent + number
    // 4. Bold: **text** -> \x1b[1m text \x1b[0m
    // 5. Italic: *text* or _text_ -> \x1b[3m text \x1b[0m  
    // 6. Inline code: `text` -> \x1b[38;5;223m text \x1b[0m
    // 7. Links: [text](url) -> text (dim url)
    // 
    // This does NOT need termimad. It's ~50 lines of regex/string ops.
    // termimad's term_text() does paragraph reflow which we don't want
    // for line-by-line rendering anyway.
}
```

**Why not termimad for streaming?** termimad's `term_text()` does paragraph-level reflow and needs the full block. For line-by-line streaming, a simple inline renderer is more appropriate and gives us full control.

### Partial Line Display

The current line (not yet terminated by `\n`) is shown as-is on an overwritable line:

```rust
// Partial line -- overwrite in place like the old progress line
print!("\r\x1b[K{}", partial_line_content);
```

When the newline arrives, the partial is replaced by the fully-rendered line via `println!()`, making it permanent.

### Tool Event Integration

Tool events already print on their own lines. The change:

- **Before tool event**: if there's a partial line being displayed, clear it (save content to buffer)
- **Print tool event lines** (unchanged cyan format with boxes)
- **After tool event**: restore partial line display

This keeps tool events visually interleaved at the correct position in the output.

## Implementation Plan

### Phase 1: IncrementalRenderer struct

New file: `src/repl/incremental.rs`

```rust
pub struct IncrementalRenderer {
    state: StreamState,
    line_buffer: String,        // partial line accumulator
    code_block_lines: Vec<String>, // buffered code block content
    code_lang: String,
    total_lines_printed: usize, // for potential cleanup
}

impl IncrementalRenderer {
    pub fn new() -> Self { ... }
    
    /// Feed a delta chunk. May print 0 or more lines to stdout.
    pub fn push(&mut self, delta: &str) { ... }
    
    /// Called when stream ends. Flushes any remaining partial line
    /// and rerenders any open code block.
    pub fn finish(&mut self) { ... }
    
    /// Clear the current partial line display (for tool event insertion).
    pub fn clear_partial(&mut self) { ... }
    
    /// Restore partial line display after tool event.
    pub fn restore_partial(&mut self) { ... }
}
```

### Phase 2: Replace print task

In `stream_and_render_inner`, replace the delta handling in the print task:

```rust
// OLD:
Some(d) => {
    full_text.push_str(&d);
    let progress = format_progress_line(&full_text, ...);
    print!("\r\x1b[K{}", progress);
}

// NEW:
Some(d) => {
    full_text.push_str(&d);  // keep for TTS + provenance
    renderer.push(&d);       // live rendering
}
```

Tool event handling wraps with `clear_partial`/`restore_partial`:

```rust
// Before tool event display:
renderer.clear_partial();
// ... print tool event lines (unchanged) ...
// After tool event display:
renderer.restore_partial();
```

### Phase 3: Remove full-response rerender

The post-streaming block that does `render_turn(&response, TurnRole::Assistant)` is removed for the normal (non-provenance) path. The response was already rendered incrementally.

**Provenance path**: When `verify_claims` is enabled, we still need the full response for claim verification. Options:
- (a) Run verification on `full_text`, print annotations as a summary below the already-rendered text
- (b) Keep the full rerender only when provenance is enabled (acceptable since it's an opt-in debug feature)

Recommend (b) for simplicity in v1.

### Phase 4: Thinking mode

Thinking deltas arrive with an ANSI prefix (`\x1b[90m...`). Detection:
- If delta starts with `\x1b[90m` or contains the thinking marker, switch to `Thinking` state
- Thinking lines are printed dim, no markdown processing
- When a non-thinking delta arrives, switch back to `Prose`

### Phase 5: Voice mode compatibility

The TTS `SentenceAccumulator` already receives raw deltas via `acc.push(&d)`. This is unchanged -- it operates on `full_text` / raw deltas, independent of the renderer.

## What Changes, What Doesn't

| Component | Changes? | Notes |
|-----------|----------|-------|
| `format_progress_line()` | **Removed** | Replaced by incremental renderer |
| Print task delta handler | **Rewritten** | Calls `renderer.push()` instead of progress line |
| Print task tool handler | **Minor** | Wraps with clear_partial/restore_partial |
| `render_response()` | **Kept** | Used only for code block rerender + provenance fallback |
| `render_turn()` | **Kept** | Still used for user turn rendering |
| `render_code_block()` | **Kept** | Called by incremental renderer for code block rerender |
| Post-stream rerender | **Removed** (normal path) | Kept only for provenance mode |
| User turn erase-rerender | **Unchanged** | This is fine -- it's instant and small |
| TTS accumulator | **Unchanged** | Operates on raw deltas |
| Input watcher | **Unchanged** | Cancel/inject still works |
| `terminal_rows()` | **Unchanged** | Still used for user turn |

## New File: `src/repl/incremental.rs`

Estimated ~200 lines:
- `IncrementalRenderer` struct + impl (~100 lines)
- `render_inline_markdown()` function (~50 lines)  
- `StreamState` enum + transitions (~30 lines)
- Tests (~20 lines)

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Inline markdown regex misparses | Medium | Conservative: only handle `**`, `` ` ``, `#`, `-/*` bullets. Skip edge cases. |
| Thinking mode detection fragile | Medium | Check for known ANSI prefix pattern. Fall through to prose if unsure. |
| Tool events interleave mid-word | Low | clear_partial/restore_partial handles this. |
| Wide Unicode / CJK line counting | Medium | Use `unicode-width` crate (already a dep?) for accurate column counting. |
| Syntect HighlightLines state drift | Low | If language detection wrong, falls back to plain text syntax. Reset state on block boundary. |

## Migration Path

1. Build `IncrementalRenderer` behind a feature flag or config toggle
2. Test with `NANOBOT_STREAM=incremental` env var
3. Once stable, make it the default
4. Remove `format_progress_line()` and old progress-line code path
5. Consider removing termimad dependency entirely if inline renderer covers all cases

## Future Enhancements (not in v1)

- **Table detection and alignment**: Buffer table rows, rerender when complete
- **Link preview**: Fetch title for URLs inline
- **Progress indicator**: Small spinner or dots while waiting for first delta
- **Semantic diff rendering**: Side-by-side old/new for edit_file tool results
- **Collapsible tool output**: Show summary line, expand on keypress
