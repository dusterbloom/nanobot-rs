#!/usr/bin/env bash
# Smoke test for the 2026-07-27 LCM fixes. Sends substantial prompts to a
# fresh agent session to force compaction within ~6 turns, then greps the
# live log for the new session to verify:
#   1. No "Summarization ended with unsafe finish reason: length" (Fix 1)
#   2. No "Auto-expanded originals" within 1 turn of fresh compaction (Fix 2)
#   3. cache_read_tokens stays high after compaction (Fix 2 / criterion #3)
set -euo pipefail

BIN="${BIN:-target/release/nanobot}"
SESSION="bench:lcm-smoke-$(date +%s)"
LOG="$HOME/.nanobot/logs/nanobot.log.$(date +%Y-%m-%d)"

echo "session=$SESSION"
echo "log=$LOG"

# Approximate prompt size needed to cross tau_soft (~11k tokens on the
# local model's 22k effective budget). 5k-char prompts × ~6 turns ≈ ~30k
# chars ≈ ~7.5k tokens of conversation; with tool overhead and system
# prompt that's enough to trigger soft compaction.
PROMPT_FILE=$(mktemp)
trap 'rm -f "$PROMPT_FILE"' EXIT
python3 -c "print('Discuss topic alpha beta gamma delta epsilon zeta eta theta iota kappa lambda: ' + ('distinct evidence and constraints about serramanna bakery recipes and higgs retained session state '*40))" > "$PROMPT_FILE"

for i in 1 2 3 4 5 6; do
    echo "--- turn $i ---"
    t0=$(python3 -c 'import time;print(int(time.time()*1000))')
    RUST_LOG=info "$BIN" agent --local --session "$SESSION" \
        -m "$(cat "$PROMPT_FILE")" 2>&1 | tail -5 || echo "turn $i exit $?"
    t1=$(python3 -c 'import time;print(int(time.time()*1000))')
    echo "turn=$i wall_ms=$((t1 - t0))"
done

echo ""
echo "=== compaction events for this session ==="
rg -a "$SESSION" "$LOG" 2>/dev/null | rg -o '"timestamp":"[^"]+".*?"message":"[^"]+"' | rg 'compact|Auto-expand|auto_expand|summariz' || echo "(none)"

echo ""
echo "=== finish_reason=length failures for this session ==="
rg -a "$SESSION" "$LOG" 2>/dev/null | rg 'unsafe finish reason' || echo "(none — Fix 1 working)"

echo ""
echo "=== auto_expand events for this session ==="
rg -a "$SESSION" "$LOG" 2>/dev/null | rg 'auto_expand|Auto-expand' || echo "(none)"

echo ""
echo "=== cache_read tokens (last 8 rows) ==="
tail -8 "$HOME/.nanobot/metrics.jsonl" | rg -o '"prompt_tokens":[0-9]+|"cache_read_tokens":[0-9]+|"session":"[^"]+"'
