# Escha control and recovery: exploratory live results

The original instruction criticism was justified. The pilot harness did not put
its guide in the actual local-model request, hid the custom argument schemas
behind discovery, and advertised production tools that its replacement registry
removed. The reset and scorer also had harness defects. Those pilots cannot
support a claim that Escha is weak at recovery interfaces.

After correction, Escha successfully wrote checkpoints and requested a fresh
window in all five notes/reset cases. Both notes/reset and current LCM recovered
the task facts in all five cases. Both unplanned-reset probes recovered the exact
artifact from history without checkpoint notes. No valid trial recorded a tool
execution error or prohibited action. This supports interface capability with a
clear contract; it does not establish an accuracy advantage over LCM.

| Method | Task facts, manual review | Exact JSON | Median total seconds |
|---|---:|---:|---:|
| A: production LCM + core restart | 5/5 | 4/5 | 205.9 |
| B: agent notes + fresh context | 5/5 | 4/5 | 92.2 |
| C: fresh context, no notes | 2/2 | 2/2 | 35.9 |

The exact mismatches were truthful failure-status strings: A returned
`Failed (ERROR permission denied)` and B returned `error`, while the oracle
expected `failed`. The fixture did not require that literal code. Original exact
scores remain unchanged in [results-summary.json](results-summary.json); the
separate factual assessments and reasons are in
[assessments.json](assessments.json). All other fields matched. An earlier
ambiguous operation-ID fixture was clarified before the scored failed-operation
pair. These are exploratory tests with documented harness corrections, not a
locked benchmark or a controlled prompt-only ablation.

The five paired cases cover a changed deployment target, a falsely claimed
successful operation, an already completed non-idempotent payment, a
case-sensitive checksum, and superseded branch/owner state. C tests failed-action
and checksum recovery without a checkpoint. See [the full timings and token
counts](results/table.md). Timings include summarization/checkpoint preparation
and recovery, and exclude server startup. B used more model requests (51 total
versus A's 18), so faster observed elapsed time does not mean fewer calls. Logical
prompt-token totals also include reused prefixes; they are not measured GPU
prefill work or billing estimates.

The harness now installs the guide, full available schemas, and a `get_tools`
argument example before first send. It audits recorded wire requests. Its
`cfg(test)` reset hook stops only after the entire requested batch has successful
durable execution and postprocessing receipts, before another old-window model
call. Preparation artifacts cannot satisfy the recovery scorer: success requires
zero preparation submissions, one fresh submission, Finished outcome, correct
JSON, and no prohibited action. All five B boundaries passed that audit. The
history search description still underspecifies its literal substring matching;
extra discovery steps should not automatically be attributed to model ability.

The model was native `EschaLabs/Qwen3.6-35B-A3B-Escha-W2`, served by hardened Higgs
`327e5021e` (`feature/adaptive-capacity-higgs`). Nanobot was `fa53da4` plus this
opt-in test harness. [HARDENING.md](HARDENING.md) locates the prior work: nanobot's
hardening is on main; Higgs's synchronized work is on the feature branch, not the
active nightly checkout. [provenance.json](provenance.json) records binary hashes
and [higgs-test.toml](higgs-test.toml) records the isolated settings. The user's
Higgs and nanobot configuration files were not edited.

A continuous-server attempt exposed separate endurance behavior: advertised
capacity declined across pressure episodes until only 3,072 total tokens and zero
prompt headroom remained. Nanobot returned CapacityUnavailable and retained the
pending turn. The next trial could not start. A later snapshot showed normal
pressure, zero current swap/compressor deltas, and zero active reservations. Logs
are preserved under [interrupted/](interrupted/). This is evidence to investigate
capacity recovery, not a lost-compaction-history finding. Remaining trials used
fresh same-config Higgs boots with at least 16K advertised prompt headroom; boot
logs and envelopes are preserved per trial. Therefore the suite is not evidence
of uninterrupted long-session endurance, and timings are exploratory.

The test uses the real context, provider, tool execution, LCM publication, and
SQLite persistence phases. Fresh B/C recovery sessions report Exact replay; A
reports Partial replay because its synthetic seed precedes recorded events. That
is a fixture limitation, not a compaction regression. It supplies synthetic quoted observations and code
appendices, about 3.3K estimated source tokens, with a 16K ceiling and forced
boundaries. It does not test autonomous reset timing, a full-window stress load,
OS-process crash recovery, channel adapters, or real external actions. A uses
production compaction and reconstructs its core from SQLite. B recreates the core
with a fresh session while its fixture history adapter retains the original rows;
notes are read from disk. Background compaction is suppressed to control the
boundary. All custom tools and the reset hook compile only into tests.

One source finding remains unmodified: LCM rebuild currently restores summary
nodes with `created_at_turn: 0`, disabling the fresh-summary cooldown after core
restart. This may increase re-expansion; this experiment did not isolate its
performance impact.

Validation: `cargo build --release` passed. Agent-loop regression tests passed
267 tests, with 11 opt-in tests ignored; the 12 live trials were run separately.
Earlier corrected-harness compaction regressions passed 52 tests, with one
ignored. Wire/reset/scoring audits passed for every live trial. Full raw
artifacts, SQL databases, notes, wire requests, server logs, and final validation
logs are retained in [results/](results/) (gitignored, in the workspace).

To rerun, retain the built hardened Higgs binary and temporary config paths
recorded in provenance, create an idle `recovery-higgs` tmux session, build the
release library tests, then run `python3 experiments/context-recovery/run.py <new-output-directory>` inside tmux from the nanobot repository. The driver uses
that named Higgs pane, or `HIGGS_EVAL_TMUX_PANE` when explicitly supplied. It refuses
to overwrite incomplete trial directories and resumes completed trials. This is
an inspectable local experiment driver, not a production rollover feature.
