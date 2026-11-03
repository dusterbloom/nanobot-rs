#!/usr/bin/env bash
# T9 benchmark: model-managed context (nodecode port).
#
# Arm A = nanobot built from `main` (no [seq N] tags, no `context_edit` tool).
# Arm B = nanobot built from this branch's working tree (T1-T8 landed).
#
# Both arms run the SAME matched tool-heavy task, once each, against a
# freshly-started (cold) Higgs server, with an isolated $HOME so neither run
# touches the real ~/.nanobot sessions/metrics. See README.md for the full
# writeup and how to point this at a different server.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="/Users/peppi/Dev/nanobot-rs"
REAL_CONFIG="${REAL_CONFIG:-$HOME/.nanobot/config.json}"
REAL_HOME="$HOME"  # isolated runs still need the real rust toolchain

# TASK_NAME selects which benchmark task runs (see TASK_SCRIPTED/TASK_PRESSURE
# below): "scripted" (default, unchanged) peaks ~9k of the 32k local window —
# below the 0.5 size-notice threshold. "pressure" is sized to cross it.
TASK_NAME="${TASK_NAME:-scripted}"

# Arm binaries. Build once with:
#   (cd "$REPO" && cargo build --release)                     # arm B (this tree)
#   git worktree add /some/sibling/dir main                   # arm A, see note below
#   (cd /some/sibling/dir && CARGO_TARGET_DIR="$REPO/target" cargo build --release)
# NOTE: this repo's Cargo.toml has `[patch]` for jack-voice pinned to the
# relative path "../jack-voice/jack-voice". Put the arm-A worktree as a
# SIBLING of nanobot-rs (not under /tmp) or that patch source won't resolve.
BIN_A="${BIN_A:-/tmp/mmc-bin/nanobot-armA-main}"
BIN_B="${BIN_B:-/tmp/mmc-bin/nanobot-armB-branch}"

HIGGS_PORT="${HIGGS_PORT:-9000}"
HIGGS_HEALTH_TIMEOUT_S=120
# pressure's 4 phases (~27k tokens of tool output, see README) need more wall
# time than scripted's 9 steps.
if [[ "$TASK_NAME" == "pressure" ]]; then
  TASK_TIMEOUT_S="${TASK_TIMEOUT_S:-1800}"
else
  TASK_TIMEOUT_S="${TASK_TIMEOUT_S:-900}"
fi

for bin in "$BIN_A" "$BIN_B"; do
  if [[ ! -x "$bin" ]]; then
    echo "error: missing arm binary: $bin (see the build note above)" >&2
    exit 1
  fi
done
if [[ ! -f "$REAL_CONFIG" ]]; then
  echo "error: no config to copy at $REAL_CONFIG" >&2
  exit 1
fi
if ! command -v higgs >/dev/null 2>&1; then
  echo "error: higgs binary not on PATH" >&2
  exit 1
fi

MODEL_DIR="$(python3 - "$REAL_CONFIG" <<'PY'
import re, sys
txt = open(sys.argv[1]).read()
m = re.search(r'"mlxModelDir"\s*:\s*"([^"]*)"', txt)
print(m.group(1) if m else "")
PY
)"
if [[ -z "$MODEL_DIR" ]]; then
  echo "error: could not read agents.defaults.mlxModelDir from $REAL_CONFIG" >&2
  exit 1
fi

RUN_DIR="$(mktemp -d /tmp/mmc-run.XXXXXX)"
echo "run dir: $RUN_DIR"

# Grading note (checked against src/repl/mod.rs's stream_and_render_inner):
# the CLI only echoes the user's `-m` text to stdout when
# `std::io::stdout().is_terminal()` is true. Both arms run here with stdout
# redirected to run.log (a regular file), so is_terminal() is false and the
# task prompt below is NEVER echoed into run.log -- a plain substring check
# against run.log cannot be satisfied by the prompt itself. That is why both
# TASK_SCRIPTED's and TASK_PRESSURE's expected substrings are safe to grade
# with the straight "all substrings in run_log_text" check in summarize.py,
# with no need to slice the log down to "after the prompt".

TASK_SCRIPTED="$(cat <<'TASKEOF'
You are running inside a controlled, scripted benchmark. Follow these steps in
order, one tool call per step unless told otherwise. Do not read beyond the
line ranges given -- they are deliberately bounded so this run stays
comparable across repeats.

1. Read src/agent/tools/base.rs, lines 1-100.
2. Read src/agent/tools/cron_tool.rs, lines 150-260.
3. Read src/agent/tools/email.rs, lines 1-100.
4. Read src/agent/tools/message.rs, lines 1-100.
5. Read src/agent/tools/shell.rs, lines 1-95.
6. Read src/agent/tools/spawn.rs, lines 1-100.
7. Run this exact shell command: cd /Users/peppi/Dev/nanobot-rs && cargo test --lib -- lcm::tests
8. Run the exact same shell command again: cd /Users/peppi/Dev/nanobot-rs && cargo test --lib -- lcm::tests
9. Finally, answer using ONLY what you read in step 2 (cron_tool.rs) -- do not
   guess, do not use outside knowledge: in cron_tool.rs's Tool::execute(),
   when the `action` parameter is a string that is not "add", "list", or
   "remove" -- for example action="foo" -- what EXACT string does the tool
   return as its (successful) result? Quote the exact text, including
   capitalization and punctuation, as it would appear for action="foo".
TASKEOF
)"
# The factual answer lives verbatim in src/agent/tools/cron_tool.rs (identical
# on both arms — not touched by this branch's diff). Grading checks this
# exact string shows up in the arm's run.log (see the TTY-echo note above).
EXPECT_SUBSTRINGS_SCRIPTED='Unknown action: foo'

# pressure: 4 settled-subproblem phases sized so cumulative tool output
# (~27k tokens, see README "pressure task" section for the per-phase budget)
# clearly crosses the branch's size notice (fires at 0.5 of the 32k local
# window, i.e. ~16k) and nears the window ceiling -- unlike scripted, which
# peaks ~9k and never fires it. The model is never told to use the `context_edit`
# tool; only the standing skill (T7) and the notice itself (T8) can prompt it.
TASK_PRESSURE="$(cat <<'TASKEOF'
You are running inside a controlled benchmark with four phases. Work phase by
phase; when a phase is done its raw output is no longer needed. Within each
phase, follow the steps in order, one tool call per step. Do not read beyond
the line ranges given -- they are deliberately bounded so this run stays
comparable across repeats.

PHASE 1
Read src/agent/tools/stash_search.rs in these 9 consecutive bounded ranges,
one read per step, in this exact order: lines 1-90, 91-180, 181-270, 271-360,
361-450, 451-540, 541-630, 631-675, 676-720.
Fact A: somewhere in that range, this file's code returns an exact error
message when a search `query` parameter fails to compile as a pattern. Find
it and remember its exact text, including capitalization and punctuation.

PHASE 2
Read src/agent/tools/web.rs in these 13 consecutive bounded ranges, one read
per step, in this exact order: lines 1-90, 91-180, 181-270, 271-360, 361-450,
451-540, 541-630, 631-720, 721-810, 811-900, 901-990, 991-1080, 1081-1170.
Fact B: somewhere in that range, a constant named WEB_FETCH_EVIDENCE_MARKER is
defined with a short string literal value. Find it and remember its exact
string value, including capitalization.

PHASE 3
Run this exact shell command: cd /Users/peppi/Dev/nanobot-rs && cargo test --lib -- agent::tools::shell
Then run this exact shell command: cd /Users/peppi/Dev/nanobot-rs && cargo test --lib -- agent::tools::spawn
Fact C: remember the "N passed" count from the FIRST command's (the
agent::tools::shell one) test-result summary line, in the form "N passed".

PHASE 4
Read src/agent/tools/recall.rs in these 7 consecutive bounded ranges, one read
per step, in this exact order: lines 1-90, 91-180, 181-270, 271-360, 361-450,
451-540, 541-630.
Fact D: somewhere in that range, this file's `execute_fetch` function returns
an exact string when its `mode` argument does not match any of "latest",
"session", "in_session", or "extract". Find it and remember its exact text,
including capitalization and punctuation.

FINAL STEP
Without re-reading any file and without re-running any command, answer from
memory of phases 1-4 above. State all four facts in order -- Fact A, Fact B,
Fact C, Fact D -- each as the exact verbatim string or number you found,
including capitalization and punctuation.
TASKEOF
)"
# Facts A/B/D live verbatim in src/agent/tools/stash_search.rs,
# src/agent/tools/web.rs, and src/agent/tools/recall.rs respectively --
# `git diff main..HEAD --stat -- <those 3 files>` is empty, so each is
# identical on both arms. Fact C is the "N passed" line from a real
# `cargo test` run against src/agent/tools/shell.rs, also untouched by this
# branch's diff (confirmed the same way) -- the count (93, measured
# 2026-11-02 against this repo's current test suite) is a snapshot, not a
# hardcoded invariant, same caveat as scripted's cron_tool.rs answer.
EXPECT_SUBSTRINGS_PRESSURE='query could not be compiled.
WEB_FETCH_EVIDENCE v1
93 passed
Error: unsupported fetch mode.'

case "$TASK_NAME" in
  scripted)
    TASK="$TASK_SCRIPTED"
    EXPECT_SUBSTRINGS="$EXPECT_SUBSTRINGS_SCRIPTED"
    ;;
  pressure)
    TASK="$TASK_PRESSURE"
    EXPECT_SUBSTRINGS="$EXPECT_SUBSTRINGS_PRESSURE"
    ;;
  *)
    echo "error: unknown TASK_NAME '$TASK_NAME' (expected 'scripted' or 'pressure')" >&2
    exit 1
    ;;
esac

higgs_pid=""

start_higgs() {
  local logdir="$1"
  HIGGS_CHUNKED_PREFILL_CHUNK_SIZE=4096 higgs serve \
    --model "$MODEL_DIR" --port "$HIGGS_PORT" --mlx-profile throughput \
    >"$logdir/higgs.log" 2>&1 &
  higgs_pid=$!
  local waited=0
  while (( waited < HIGGS_HEALTH_TIMEOUT_S )); do
    if curl -s -m 1 "http://127.0.0.1:${HIGGS_PORT}/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
    waited=$(( waited + 2 ))
  done
  echo "error: higgs did not become healthy within ${HIGGS_HEALTH_TIMEOUT_S}s (see $logdir/higgs.log)" >&2
  return 1
}

stop_higgs() {
  [[ -n "$higgs_pid" ]] || return 0
  kill "$higgs_pid" 2>/dev/null
  local waited=0
  while (( waited < 20 )); do
    kill -0 "$higgs_pid" 2>/dev/null || { higgs_pid=""; return 0; }
    sleep 1
    waited=$(( waited + 1 ))
  done
  kill -9 "$higgs_pid" 2>/dev/null
  higgs_pid=""
}
trap stop_higgs EXIT

run_arm() {
  local label="$1" bin="$2" home="$3"
  mkdir -p "$home/.nanobot"
  cp "$REAL_CONFIG" "$home/.nanobot/config.json"

  echo "[$label] starting cold higgs..."
  start_higgs "$home" || return 1
  echo "[$label] higgs ready (pid $higgs_pid), running task..."

  # nanobot's non-interactive tracing writer defaults to stdout (verified by
  # smoke test), so both the rendered transcript and the debug "prompt_*"
  # proxy lines land in the same stream -- merge them into one log.
  ( cd "$REPO" && RUSTUP_HOME="$REAL_HOME/.rustup" CARGO_HOME="$REAL_HOME/.cargo" HOME="$home" \
      RUST_LOG="warn,nanobot::agent::agent_loop::shared=debug" \
      "$bin" agent -m "$TASK" -s "bench:$label" --local \
      </dev/null >"$home/run.log" 2>&1 ) &
  local pid=$!
  ( sleep "$TASK_TIMEOUT_S"; kill -9 "$pid" 2>/dev/null ) &
  local watchdog=$!
  wait "$pid"
  local rc=$?
  kill "$watchdog" 2>/dev/null
  echo "[$label] task exited rc=$rc"

  stop_higgs
  return 0
}

run_arm "armA-main"    "$BIN_A" "$RUN_DIR/home-A"
run_arm "armB-branch"  "$BIN_B" "$RUN_DIR/home-B"

echo
echo "=== results ($RUN_DIR) — task: $TASK_NAME ==="
python3 "$HERE/summarize.py" \
  "$RUN_DIR/home-A" "A:main" \
  "$RUN_DIR/home-B" "B:branch" \
  "$EXPECT_SUBSTRINGS"
