#!/usr/bin/env bash
# Run trio E2E tests against a live LM Studio instance.
#
# Usage:
#   bash scripts/test_trio_e2e.sh
#   NANOBOT_TRIO_BASE=http://192.168.1.22:1234/v1 bash scripts/test_trio_e2e.sh
set -euo pipefail

BASE="${NANOBOT_TRIO_BASE:-http://192.168.1.22:1234/v1}"

# Health check
if ! curl -sf "$BASE/models" > /dev/null 2>&1; then
    echo "ERROR: LM Studio not reachable at $BASE"
    echo "Set NANOBOT_TRIO_BASE or start LM Studio."
    exit 1
fi
echo "OK: LM Studio responding at $BASE"

# Run trio E2E tests (single-threaded for JIT model loading)
NANOBOT_TRIO_BASE="$BASE" \
RUST_LOG=debug \
cargo test trio_e2e -- --ignored --nocapture --test-threads=1
