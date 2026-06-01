# Contributing

Two regression tracks: **correctness** and **speed**. PR notes must include
which tracks were run, the machine, the provider/model, and any notable
failures.

The only acceptable speed regression is one that fixes a correctness bug and
requires a speed penalty.

## Correctness Regression Tests

Build first:

```sh
cargo build --release
```

Run all tests:

```sh
cargo test
```

Useful narrower checks (one per protected contract):

```sh
cargo test --test protocol_invariants
cargo test --test tool_call_quality
cargo test --test context_byte_stability
cargo test --test session_resume
cargo test --test channel_smoke
cargo test --test provider_parsers
```

What they cover:

- `protocol_invariants`: message-array protocol — tool-call/tool-result
  pairing, "last message must be user" for local models, no orphan tool_calls.
  This is the best quick check for prompt-rendering changes.
- `tool_call_quality`: tool-call parser regressions across families
  (Hermes/Qwen/Llama/DeepSeek/native). Compares each recorded model response
  against a golden parsed `ToolCall` in `tests/fixtures/tool_calls/`.
- `context_byte_stability`: prompt-builder drift. Hashes the output of
  `ContextBuilder` for a fixed `TurnContext` and compares to a checked-in
  SHA-256. Any change must update the hash with a commit message explaining why.
- `session_resume`: JSONL session save/load round-trip. Catches `session/`,
  `working_memory/`, `session_indexer/` regressions.
- `channel_smoke`: channel-adapter outbound formatting. No network — mocks the
  transport and asserts the rendered message bytes.
- `provider_parsers`: OpenAI/Anthropic-compatible response parsing for each
  supported provider, from recorded SSE transcripts in
  `tests/fixtures/provider_responses/`.

Override fixture paths when needed:

```sh
NANOBOT_FIXTURE_DIR=/path/to/alternate cargo test --test tool_call_quality
```

## Speed Regression Tests

Use `nanobot-bench` for end-to-end speed regressions. It reports instantaneous
per-task timings, not whole-run averages.

```sh
cargo run --release --bin nanobot-bench -- \
  --tasks t01,t02,t05 \
  --csv /tmp/nanobot-speed.csv
```

Use the same machine, provider, model, power state, and background load when
comparing two commits. For loop or context-builder work, run at least one
before/after CSV and compare both `context_ms` and `total_ms`.

To compare two CSVs:

```sh
python3 scripts/bench_diff.py benches/baseline.csv /tmp/nanobot-speed.csv
```

The headline metric is `total_ms` summed across the run. A PR that regresses
it by **more than 5%** without explanation in the PR description should not
merge.

## Reporting Bugs

For debugging a failing run, keep the trace:

```sh
NANOBOT_TRACE=/tmp/nanobot-trace.txt cargo run -- agent -m "..."
```

The trace captures inbound message, context build, provider request/response,
tool-call parser events, and the final outbound message for the session.
