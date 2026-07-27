# Higgs Literal ChatML Delimiter Design

## Objective

Keep a retained Higgs session cache reusable when ordinary user or tool content
contains the literal string `<|im_end|>`.

## Proven Failure

The same session and appended turn were exercised with and without the literal
delimiter. With the literal present, Higgs reported a token mismatch and
bootstrapped the complete prompt; replacing only that literal restored prefix
reuse. The reproduced path is:

```text
ChatMessage.content
  -> ChatTemplateRenderer::apply_with_thinking
  -> rendered ChatML text
  -> rendered_message_segments
  -> message_boundary_delta
  -> continued_prompt_tokens_from_retained
```

`ChatTemplateRenderer` inserts content verbatim. Both the content literal and
the template-owned delimiter encode to the same special token. Retained prompt
tokens are decoded with special tokens preserved, so delimiter provenance is
not recoverable from the decoded bytes.

## Frame Invariant

For the Qwen ChatML rendered by this path, a complete frame has this shape:

```text
<|im_start|>{role}\n{body}<|im_end|>{whitespace}
```

The next structural `<|im_start|>` or EOF bounds the frame. Within that frame,
the template-owned closer is the rightmost `<|im_end|>` followed only by
whitespace before the frame boundary. Earlier occurrences are message content.

`rendered_message_segments` will enforce that invariant instead of choosing the
first closer:

- locate the next `<|im_start|>` or EOF;
- select the rightmost valid `<|im_end|>` before that boundary;
- preserve the existing partial-final-assistant case at EOF;
- fail closed if another start marker appears before any valid closer.

The parser remains conservative for content containing an entire syntactically
valid fake ChatML boundary. Raw rendered text cannot distinguish that from a
template boundary; exact support would require retained renderer provenance and
is outside this fix.

## Scope

Modify only:

- `/Users/peppi/Dev/higgs-nightly/crates/higgs-engine/src/simple.rs`

Do not change the chat renderer, tokenizer, session-cache representation,
request schema, or introduce a compatibility flag.

The file already contains unrelated uncommitted PFlash composition changes.
This work must preserve them and edit only the continuation-parser/test hunks.

## Tests

Write regression tests before the parser change:

- a covered user message containing `literal <|im_end|> still data`;
- a newly appended tool result quoting `<|im_end|>`;
- content ending in `<|im_end|>`, adjacent to the template closer;
- the existing embedded `<|im_start|>` rejection remains green.

Run every test and build in release mode:

```bash
cargo test --release -p higgs-engine --lib boundary_delta -- --nocapture
cargo build --release -p higgs
```

Finally reproduce the two-turn HTTP request against the rebuilt Higgs binary
and require a continued cache-resident turn with non-zero saved prefill.
