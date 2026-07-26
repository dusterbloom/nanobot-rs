# Full-Context Compaction Design

## Objective

Make LCM summaries reflect the real conversation segment being compacted. The
summarizer must see the complete semantic segment in one request without
validation checklists, synthetic turns, protocol reminders, lossy tool-output
previews, chunk summaries, or merge summaries.

## Semantic Transcript

The compactor builds one transcript from the selected SQLite-backed message
range:

- Keep real user and assistant text.
- Keep complete assistant tool-call names and arguments.
- Keep complete tool results.
- Exclude system and developer prompts.
- Exclude messages marked synthetic or as LCM summary records.
- Remove nanobot transport and protocol envelopes while retaining their
  semantic payload.

Transcript construction never truncates a message or substitutes a digest.

## Per-Request Context

There is no fixed working context for compaction. Before each request, nanobot
computes:

```text
required_context =
    summarizer_system_prompt_tokens
  + complete_transcript_tokens
  + chat_template_and_wrapper_allowance
  + requested_summary_output_tokens
  + safety_margin
```

The summary output allowance is the maximum requested by that compaction call
(currently 512 tokens). It is part of every capacity calculation rather than a
hidden deduction from a fixed input budget. If an LCM level later requests a
different output maximum, the required context changes with it automatically.
The request therefore has enough room for both the complete source and the
summary it asks the model to produce.

The configured or model-reported maximum is only a hard ceiling. Qwen3.5 2B
advertises a 262,144-token maximum, and the installed Higgs server accepts
prompt lengths dynamically rather than exposing a fixed startup context flag.
Nanobot must not use `lcm.compactionContextSize` to split, merge, or truncate
input.

If a complete selected range plus its output allowance exceeds the model
ceiling, compaction does not run on a partial source. The original SQLite
history and active context remain available, and the capacity error identifies
the required and available token counts. The LCM range selector may choose a
smaller complete range on a later attempt.

## Model Request

The request contains:

1. A concise system instruction defining the summary format and prohibiting
   invention.
2. One user message containing the complete semantic transcript.

It does not contain `TOPIC_ANCHORS`, `REQUIRED_LITERALS`, synthetic validation
turns, prior chunk summaries, or repeated protocol instructions. There is one
model call and no map-reduce merge.

Post-response handling is limited to transport and structural failures:
provider errors, incomplete generation, empty output, obvious response loops,
and echoed source envelopes. It does not append missing literals or accept a
summary because it copied validation terms.

## Real-Session Evaluation

Run the Qwen3.5 2B MLX sidecar against representative real sessions from
`sessions.db`, including the Asteroids repair and a session whose previous
summary passed despite copying anchors. Preserve the raw prompt transcript,
raw model summary, token counts, and finish reason for inspection.

Judge each summary on:

- current user goal and constraints;
- durable completed work;
- unresolved work, failures, and uncertainty;
- exact paths, commands, identifiers, ports, and model names where material;
- absence of invented completion or causes;
- absence of protocol and validation scaffolding.

This evaluation does not rewrite a poor summary or conceal it behind a quality
score. The observed outputs determine whether the 2B sidecar remains the
production compactor or must be replaced by a stronger model.

## Tests

Tests are written before implementation and prove:

- synthetic and protocol-only records are excluded;
- complete tool calls and outputs survive transcript construction;
- one request contains the entire selected semantic range;
- request capacity and summary output are calculated per invocation;
- over-capacity input fails without truncation or provider invocation;
- no validation anchors or literal checklists reach the model;
- no chunk or merge request occurs.

After targeted tests pass, run `cargo test`, `cargo build`, and
`scripts/turn_bench.sh`. Run GitNexus impact analysis before production edits
and `detect_changes` before any implementation commit.
