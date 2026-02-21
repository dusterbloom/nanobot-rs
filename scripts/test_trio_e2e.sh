#!/usr/bin/env bash
# Trio E2E test runner with auto-repair protocol.
#
# Features:
#   - Pre-flight health gate (endpoint + model availability)
#   - Per-test retry with failure classification
#   - Summary report with pass/fail/retry counts
#
# Usage:
#   bash scripts/test_trio_e2e.sh
#   NANOBOT_TRIO_BASE=http://192.168.1.22:1234/v1 bash scripts/test_trio_e2e.sh
#   MAX_RETRIES=5 bash scripts/test_trio_e2e.sh
set -uo pipefail

BASE="${NANOBOT_TRIO_BASE:-http://192.168.1.22:1234/v1}"
MAX_RETRIES="${MAX_RETRIES:-3}"
RUST_LOG="${RUST_LOG:-debug}"

# Counters
TOTAL=0
PASSED=0
FAILED=0
RETRIED=0
SKIPPED=0

# ── Pre-flight ──────────────────────────────────────────────

preflight() {
    echo "=== PRE-FLIGHT ==="

    # 1. Endpoint health
    if ! curl -sf "${BASE}/models" > /dev/null 2>&1; then
        echo "ABORT: LM Studio not reachable at $BASE"
        echo "  Start LM Studio or set NANOBOT_TRIO_BASE."
        exit 1
    fi
    echo "OK: LM Studio responding at $BASE"

    # 2. List available models
    local models
    models=$(curl -sf "${BASE}/models" 2>/dev/null \
        | python3 -c "import sys,json; [print(m['id']) for m in json.load(sys.stdin).get('data',[])]" 2>/dev/null || true)

    if [ -n "$models" ]; then
        echo "Models available:"
        echo "$models" | sed 's/^/  /'
    else
        echo "WARN: Could not list models (JIT loading may still work)"
    fi

    echo ""
}

# ── Failure classification ──────────────────────────────────

# Classify test output into failure categories.
# Returns: PASS, INFRA_DOWN, JIT_LOADING, TIMEOUT, MODEL_QUALITY, BUG
classify() {
    local output="$1"
    local exit_code="$2"

    if [ "$exit_code" -eq 0 ]; then
        echo "PASS"
        return
    fi

    # Check patterns (order matters — most specific first)
    if echo "$output" | grep -qi "connection refused\|connect error\|not reachable"; then
        echo "INFRA_DOWN"
    elif echo "$output" | grep -qi "loading model\|no models loaded\|model is loading\|503"; then
        echo "JIT_LOADING"
    elif echo "$output" | grep -qi "timed out\|deadline has elapsed\|test timed out"; then
        echo "TIMEOUT"
    elif echo "$output" | grep -qi "response should be non-empty\|should be substantive\|warmup failed"; then
        echo "MODEL_QUALITY"
    else
        echo "BUG"
    fi
}

# Delay before retry based on failure category.
retry_delay() {
    local category="$1"
    case "$category" in
        JIT_LOADING)   echo 30 ;;
        TIMEOUT)       echo 60 ;;
        MODEL_QUALITY) echo 5 ;;
        *)             echo 0 ;;
    esac
}

# ── Per-test runner ─────────────────────────────────────────

run_test() {
    local test_name="$1"
    local attempt=1
    local last_category=""

    TOTAL=$((TOTAL + 1))

    while [ "$attempt" -le "$MAX_RETRIES" ]; do
        if [ "$attempt" -gt 1 ]; then
            RETRIED=$((RETRIED + 1))
            local delay
            delay=$(retry_delay "$last_category")
            echo "  retry $attempt/$MAX_RETRIES (${last_category}, wait ${delay}s)..."
            sleep "$delay"
        fi

        # Run single test, capture output
        local output
        output=$(NANOBOT_TRIO_BASE="$BASE" RUST_LOG="$RUST_LOG" \
            cargo test "$test_name" -- --ignored --nocapture --test-threads=1 2>&1)
        local rc=$?

        last_category=$(classify "$output" "$rc")

        case "$last_category" in
            PASS)
                if [ "$attempt" -gt 1 ]; then
                    echo "  PASS (after $attempt attempts)"
                else
                    echo "  PASS"
                fi
                PASSED=$((PASSED + 1))
                return 0
                ;;
            INFRA_DOWN)
                echo "  ABORT: infrastructure down — $test_name"
                echo "$output" | tail -5
                FAILED=$((FAILED + 1))
                # Abort entire suite
                echo ""
                echo "=== ABORTING: LM Studio infrastructure failure ==="
                summary
                exit 1
                ;;
            BUG)
                echo "  FAIL (BUG — not retrying)"
                echo "$output" | grep -i "panicked\|assertion\|error" | head -5
                FAILED=$((FAILED + 1))
                return 1
                ;;
            *)
                # JIT_LOADING, TIMEOUT, MODEL_QUALITY — retryable
                if [ "$attempt" -eq "$MAX_RETRIES" ]; then
                    echo "  FAIL (${last_category} — exhausted $MAX_RETRIES retries)"
                    echo "$output" | grep -i "panicked\|assertion\|error\|timed out" | head -3
                    FAILED=$((FAILED + 1))
                    return 1
                fi
                ;;
        esac

        attempt=$((attempt + 1))
    done
}

# ── Summary ─────────────────────────────────────────────────

summary() {
    echo ""
    echo "=== SUMMARY ==="
    echo "  Total:   $TOTAL"
    echo "  Passed:  $PASSED"
    echo "  Failed:  $FAILED"
    echo "  Retried: $RETRIED"
    echo ""
    if [ "$FAILED" -eq 0 ]; then
        echo "ALL TESTS PASSED"
    else
        echo "SOME TESTS FAILED"
    fi
}

# ── Main ────────────────────────────────────────────────────

preflight

# Discover trio E2E test names
TESTS=$(cargo test trio_e2e -- --list --ignored 2>/dev/null \
    | grep '::test_trio_e2e_' | sed 's/: test$//')

if [ -z "$TESTS" ]; then
    echo "ERROR: No trio_e2e tests found. Run cargo build first."
    exit 1
fi

TEST_COUNT=$(echo "$TESTS" | wc -l)
echo "=== RUNNING $TEST_COUNT TESTS (max $MAX_RETRIES attempts each) ==="
echo ""

# Run preflight canary first (if it exists)
PREFLIGHT_TEST=$(echo "$TESTS" | grep "preflight" || true)
if [ -n "$PREFLIGHT_TEST" ]; then
    echo "[canary] $PREFLIGHT_TEST"
    if ! run_test "$PREFLIGHT_TEST"; then
        echo ""
        echo "=== ABORTING: preflight canary failed ==="
        summary
        exit 1
    fi
    echo ""
fi

# Run remaining tests
for test in $TESTS; do
    # Skip preflight (already ran)
    if echo "$test" | grep -q "preflight"; then
        continue
    fi
    echo "[test] $test"
    run_test "$test"
    echo ""
done

summary

# Exit with failure if any test failed
[ "$FAILED" -eq 0 ]
