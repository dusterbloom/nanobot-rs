#!/usr/bin/env bash
# T9 benchmark: model-managed context (nodecode port).
#
# Arm A = nanobot built from `main` (no [seq N] tags, no `context` tool).
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
TASK_TIMEOUT_S="${TASK_TIMEOUT_S:-900}"

# The factual answer lives verbatim in src/agent/tools/cron_tool.rs (identical
# on both arms — not touched by this branch's diff). Grading just checks this
# exact string shows up in the arm's stdout transcript.
EXPECT_SUBSTRING='Unknown action: foo'

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

TASK="$(cat <<'TASKEOF'
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
echo "=== results ($RUN_DIR) ==="
python3 "$HERE/summarize.py" \
  "$RUN_DIR/home-A" "A:main" \
  "$RUN_DIR/home-B" "B:branch" \
  "$EXPECT_SUBSTRING"
