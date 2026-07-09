#!/usr/bin/env bash
# Phase 1 baseline (plan cozy-squishing-galaxy): run an N-turn local session
# and collect per-turn overhead (turn_timing tracing lines) + per-call
# metrics.jsonl rows (ttft/elapsed/tokens) for benches/baseline.csv.
#
# Usage: scripts/turn_bench.sh [N_TURNS] [SESSION]
# Env:   BIN=target/release/nanobot  OUT=/tmp/turn_bench.<pid>
set -euo pipefail

N="${1:-20}"
SESSION="${2:-bench:baseline}"
BIN="${BIN:-target/release/nanobot}"
OUT="${OUT:-/tmp/turn_bench.$$}"
METRICS="$HOME/.nanobot/metrics.jsonl"

mkdir -p "$OUT"
start_lines=$(wc -l <"$METRICS" 2>/dev/null || echo 0)

echo "session=$SESSION turns=$N out=$OUT"
for i in $(seq 1 "$N"); do
  t0=$(python3 -c 'import time;print(int(time.time()*1000))')
  RUST_LOG=turn_timing=info "$BIN" agent --local --session "$SESSION" \
    -m "Turn $i: reply with one short sentence mentioning the number $i." \
    >"$OUT/turn_$i.stdout" 2>"$OUT/turn_$i.stderr"
  t1=$(python3 -c 'import time;print(int(time.time()*1000))')
  wall=$((t1 - t0))
  timing=$(grep -h 'prepare_context_timing' "$OUT/turn_$i.stderr" | tail -1 || true)
  echo "turn=$i wall_ms=$wall ${timing:-<no timing line>}"
done

echo "--- turn_timing lines ---"
grep -h 'turn_timing' "$OUT"/turn_*.stderr || true
echo "--- new metrics.jsonl rows ---"
tail -n "+$((start_lines + 1))" "$METRICS" 2>/dev/null || true
echo "logs: $OUT"
