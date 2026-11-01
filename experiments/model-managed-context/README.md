# T9 — model-managed context benchmark

Implements T9 of `docs/superpowers/plans/2026-11-01-model-managed-context.md`:
before/after comparison of Arm A (`main`, no `[seq N]` tags, no `context`
tool) vs Arm B (this branch's working tree, T1-T8 landed) on one matched
tool-heavy task.

Primary arm is **local, against Higgs** (not cloud) — the 32k local context
window is small enough that a 9-step tool-heavy task produces real pressure,
which is the regime the `context` tool is meant to help with. The cloud arm
from the original plan text is intentionally **not** run here: this repo's
config resolves the default model via a provider-prefixed model string
(`zhipu-coding/glm-5.2`), and local mode is *also* auto-selected by
`cmd_agent` whenever `agents.defaults.localApiBase` is non-empty (true in
this config) — forcing pure cloud would mean mutating that field in the
isolated config copy, adding complexity for a secondary arm. Skipped per
explicit instruction to keep spend/time small.

## What it does

`run.sh`:
1. Resolves `agents.defaults.mlxModelDir` out of the real `~/.nanobot/config.json`
   (never prints it — only reads the one field it needs).
2. For each arm: makes a throwaway `$HOME` (`mktemp -d`), copies the real
   `config.json` into `$HOME/.nanobot/` (isolation — neither arm touches your
   real `~/.nanobot/sessions.db` or `metrics.jsonl`), starts a **cold** Higgs
   server (`higgs serve --model <mlxModelDir> --port 9000 --mlx-profile
   throughput`, the same invocation `src/higgs.rs::server_start` uses),
   waits for `/health`, then runs the arm binary once:
   `agent -m "<task>" -s bench:<arm> --local`, cwd = this repo.
3. Kills that Higgs instance, repeats for the other arm (so each arm gets a
   genuinely cold prefix cache — no carryover between arms).
4. Hands both arms' isolated `$HOME`s to `summarize.py`, which prints the
   results table.

### The task (9 steps, matched across arms)

Reads 6 files under `src/agent/tools/` at fixed line ranges (chosen to fit
Higgs' 4096-byte inline-read budget in one page each, so the step count stays
deterministic instead of ballooning into dozens of pagination calls), runs
`cargo test --lib -- lcm::tests` twice via `exec`, then must answer a factual
question whose answer lives *only* in step 2's file
(`src/agent/tools/cron_tool.rs`, lines 150-260): the exact string the `cron`
tool returns for an unrecognized `action` value. That file is untouched by
this branch's diff (`git diff main...HEAD -- src/agent/tools/cron_tool.rs` is
empty), so the correct answer — `Unknown action: foo` — is identical on both
arms; grading is a strict substring check against the merged log, nothing
fuzzy.

The model is never told to use the `context` tool. Arm B only has it because
`src/agent/prepare_context.rs` registers it in `CORE_NATIVE_TOOLS`, and the
standing skill text (T7) only *suggests* using it "during substantial tool
work ... not after every tool call." Whether it actually gets called is part
of the result, not an assumption.

## Metrics and how they're read

`summarize.py` reads each arm's isolated `$HOME`:
- **requests, mean/total prompt tokens** — `~/.nanobot/metrics.jsonl`
  (`RequestMetrics`, filtered to `role == "main"`).
- **prefill cache hit %** — `RequestMetrics.cache_read_tokens` when the
  server reports it. In testing against this Higgs build it never did
  (`cache_read_tokens` was always `None` — Higgs logged
  `higgs_context_discovery_failed: HTTP 409 "fast-session contract fields
  must be positive"` on every call here, a pre-existing Higgs/nanobot
  integration quirk unrelated to T1-T9 and out of this experiment's scope to
  fix). When that happens, the table falls back to the **PromptDelta proxy**
  named in the plan: counts of `prompt_cache_cold_start` /
  `prompt_append_only` / `prompt_prefix_diverged` /
  `prompt_cache_sanctioned_reset`, logged by
  `src/agent/agent_loop/shared.rs` at target
  `nanobot::agent::agent_loop::shared` and captured by running each arm with
  `RUST_LOG="warn,nanobot::agent::agent_loop::shared=debug"`. This is a
  client-side fingerprint comparison (ours, not Higgs'), so it's unaffected
  by the 409.
- **context tool calls** (arm B) — `sessions.db`'s `session_events` table,
  counting `tool_pre_execute` rows with `tool_name == "context"`.
- **task success** — the exact string `Unknown action: foo` found anywhere
  in the arm's merged log.

Note: nanobot's non-interactive tracing writer defaults to **stdout**, not
stderr (checked empirically — `tracing_subscriber::fmt::layer()` with no
`with_writer` override writes to stdout); `run.sh` therefore merges both
streams into one `run.log` per arm rather than splitting them.

`read_metrics.py`/`summarize()` from `experiments/context-recovery/endurance.py`
was considered for reuse per the plan, but that reader hard-codes assertions
specific to a different harness (a `higgs-endurance` wire contract with
`submit_result`/`notes` schemas, continuous-tmux-server semantics, no
`main`-vs-branch binary split) — it doesn't fit this simpler matched-task
shape, so `summarize.py` here is a small standalone script instead.

## Building the two arm binaries

```bash
cd /Users/peppi/Dev/nanobot-rs
cargo build --release                                    # arm B: this tree
cp target/release/nanobot /tmp/mmc-bin/nanobot-armB-branch

# Arm A must be built from main. This repo's Cargo.toml pins jack-voice via
# [patch."https://github.com/dusterbloom/jack-voice.git"] to the RELATIVE
# path "../jack-voice/jack-voice" — Cargo resolves every [patch] target even
# when the `voice` feature is off, so the worktree must sit as a SIBLING of
# nanobot-rs (not under /tmp) for that path to exist.
git worktree add /Users/peppi/Dev/nanobot-main main
cd /Users/peppi/Dev/nanobot-main
CARGO_TARGET_DIR=/Users/peppi/Dev/nanobot-rs/target cargo build --release   # reuses arm B's already-built deps (same Cargo.lock) — ~25s instead of a full cold build
cp /Users/peppi/Dev/nanobot-rs/target/release/nanobot /tmp/mmc-bin/nanobot-armA-main
cd /Users/peppi/Dev/nanobot-rs
git worktree remove /Users/peppi/Dev/nanobot-main
```

`run.sh` expects the two binaries at `/tmp/mmc-bin/nanobot-armA-main` and
`/tmp/mmc-bin/nanobot-armB-branch` (override with `BIN_A=... BIN_B=...`).

## Running

```bash
experiments/model-managed-context/run.sh
```

Env overrides: `BIN_A`, `BIN_B`, `HIGGS_PORT` (default 9000), `TASK_TIMEOUT_S`
(default 900s per arm), `REAL_CONFIG` (default `$HOME/.nanobot/config.json`).

To rerun later against Higgs (the only arm this script runs):

```bash
experiments/model-managed-context/run.sh
```

To point at a different local server instead of Higgs, this script would
need a second code path (it currently hardcodes the `higgs serve` startup
command and `--local` discovery) — not built here since only Higgs was
available to test against.

## Results (last run, 2026-11-01)

| arm | requests | mean prompt tok/request | prefill cache hit % | total prompt tok | task success |
|-----|----------|--------------------------|----------------------|-------------------|--------------|
| A:main | 9 | 7934.8 | 83.5% | 71413 | True |
| B:branch | 12 | 9426.6 | 87.0% | 113119 | True |

Context tool calls — A:main: 0 (doesn't exist on main) | **B:branch: 0**.
PromptDelta proxy (both arms mostly `append_only`, 0 `diverged`, 1
`cold_start` each, as expected for one cold-started turn 1): A =
cold=1/append=8/diverged=0/reset=0; B = cold=1/append=11/diverged=0/reset=0.

**The model never called the `context` tool, even though it was available
and the standing skill (T7) was in the system prompt.** This is the honest
headline result, not a tooling failure on this script's part — both arms
answered the factual question correctly (task success true/true), and
nothing in the transcript suggests the model was blocked from calling it.
Cache-hit % and append-only-dominant PromptDelta came out close between
arms, consistent with a task this short never actually needing an edit.

**Known confound in this run, reported rather than hidden:** isolating
`$HOME` per arm also isolates `~/.rustup`/`~/.cargo`, so the temp home has no
default toolchain. Both arms' first `cargo test` (step 7) failed identically
with `rustup could not choose a version of cargo to run`. Arm A's model gave
up after the second identical failure and moved on. Arm B's model instead
ran `rustup default stable` to self-heal, which installed a *different*
rustc than whatever already populated `target/debug` in this repo, forcing
a full from-scratch recompile that took several minutes and 3 extra `exec`
calls beyond the 2 instructed. That is why arm B's request/token counts run
higher — it reflects the model's differing persistence on a broken troubleshooting
side-quest, not anything about `[seq N]` tags or the `context` tool. The
factual-answer and context-tool-usage columns are unaffected by this (both
come from step 2's file read and the standing skill, neither of which
touches cargo/rustup), but the token/request columns for this run should be
read with that in mind. A cleaner rerun would pre-seed each temp
`$HOME/.rustup` + `$HOME/.cargo` from the real ones (not done here to avoid
copying several more GB under already-tight disk).

## Caveats

- **n=1 per arm.** One run each, as instructed ("each arm run once"). No
  variance estimate — a single cold run's prefill/tokenize timing and the
  model's own non-determinism (temperature, sampling) both land in these
  numbers.
- **Local only.** Cloud arm skipped (see above) — results say nothing about
  provider-side prompt caching (Anthropic/OpenAI-style), only about the
  client-side wire-size and the model's own voluntary tool use.
- **Higgs 409 on every call** (`fast-session contract fields must be
  positive`) meant the primary cache-hit signal (`cache_read_tokens`) was
  unavailable in this environment; the table reports the PromptDelta proxy
  instead, as the plan anticipates.
- **Task success is a strict substring match.** A semantically-correct but
  differently-worded answer (e.g. missing the literal colon-space) is
  scored as failure. Check the per-arm `run.log` directly for nuance.

## Rerun 2026-11-01 (toolchain confound fixed)

Higgs, Clover-35B-A3B-Preview-MLX-4bit, temp 0, n=1 per arm, cold restart between arms.

| arm | requests | mean prompt tok/req | cache hit % | total prompt tok | success | `context` calls |
|-----|----------|---------------------|-------------|------------------|---------|-----------------|
| A:main | 9 | 8,009 | 82.9% | 72,079 | yes | 0 |
| B:branch | 12 | 9,889 | 87.0% | 118,667 (+65%) | yes | 0 |

PromptDelta: 0 diverged in either arm (the `[seq N]` render-time tag kept the prefix append-only).

The +65% is three extra `inspect_tool_result` pages the branch arm took over the
8,106-char `cargo test` output that main answered from the preview. Same command,
same repo, temp 0, so the likeliest cause is the changed prompt (standing skill,
`context` schema, `[seq N]` tags), not noise; n=1 cannot separate which.

The task does not exercise the mechanism: it is a numbered script (no settled
subproblem to checkpoint after) peaking near 9k of a 32k window. Whether Clover
adopts `context` needs an open-ended task that crosses the window.
