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
cargo test --test protocol_tests
cargo test --test lcm_e2e_tests
```

What they cover:

- `protocol_invariants`: message-array protocol — tool-call/tool-result
  pairing, "last message must be user" for local models, no orphan tool_calls.
  This is the best quick check for prompt-rendering changes.
- `protocol_tests`: provider and parser protocol regressions, including native
  and fallback tool-call shapes.
- `lcm_e2e_tests`: lossless-context compaction, summary selection, and reset
  behavior.

## Speed Regression Tests

Build the release binary, then use the real local turn harness. It records wall
time, `turn_timing` context-build spans, and the new rows written to
`~/.nanobot/metrics.jsonl`.

```sh
cargo build --release
OUT=/tmp/nanobot-before scripts/turn_bench.sh 20 bench:before
# check out the candidate build, rebuild, then:
OUT=/tmp/nanobot-after scripts/turn_bench.sh 20 bench:after
```

Use the same machine, provider, model, power state, and background load when
comparing two commits. For loop or context-builder work, compare the emitted
context timing, TTFT, elapsed time, token counts, and failures.

Historical baselines remain in `benches/baseline.csv`. If a run has been
normalized to the same CSV schema, compare it with:

```sh
python3 scripts/bench_diff.py benches/baseline.csv /tmp/nanobot-speed.csv
```

Do not claim a speedup from the historical baseline alone: include matched
before/after measurements from the same environment in the PR notes.

## Reporting Bugs

For debugging a failing run, keep the trace:

```sh
NANOBOT_TRACE=/tmp/nanobot-trace.txt cargo run -- agent -m "..."
```

The trace captures inbound message, context build, provider request/response,
tool-call parser events, and the final outbound message for the session.
