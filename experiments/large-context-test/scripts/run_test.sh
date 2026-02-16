#!/bin/bash
# run_test.sh - Execute the full large-context test pipeline

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$ROOT_DIR/results"

# Cleanup function to restore config on exit
CONFIG_BACKUP="$HOME/.nanobot/config.json.backup"
CONFIG_FILE="$HOME/.nanobot/config.json"
restore_config() {
    if [ -f "$CONFIG_BACKUP" ]; then
        mv "$CONFIG_BACKUP" "$CONFIG_FILE" 2>/dev/null || true
    fi
}
trap restore_config EXIT

echo "=============================================="
echo "NANOBOT LARGE-CONTEXT MULTI-AGENT TEST"
echo "=============================================="
echo ""

# Step 0: Check prerequisites
echo "[0/5] Checking prerequisites..."

if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found"
    exit 1
fi

NANOBOT_BIN="${NANOBOT_BIN:-$(which nanobot 2>/dev/null || echo "$HOME/Dev/nanobot/target/release/nanobot")}"
if [ ! -x "$NANOBOT_BIN" ]; then
    echo "ERROR: nanobot binary not found at $NANOBOT_BIN"
    echo "Set NANOBOT_BIN environment variable or build with: cargo build --release"
    exit 1
fi
echo "  nanobot: $NANOBOT_BIN"

# Step 1: Fetch data
echo ""
echo "[1/5] Fetching ArXiv data..."
if [ ! -f "$ROOT_DIR/data/arxiv_papers.csv" ]; then
    python3 "$ROOT_DIR/data/fetch_arxiv.py"
else
    echo "  Data already exists, skipping fetch."
    echo "  Delete $ROOT_DIR/data/arxiv_papers.csv to re-fetch."
fi

# Step 2: Generate ground truth
echo ""
echo "[2/5] Generating ground truth..."
python3 "$SCRIPT_DIR/ground_truth.py"

# Step 3: Start llama servers
echo ""
echo "[3/5] Starting llama-server cluster..."
"$SCRIPT_DIR/setup.sh" start

# Give servers extra time to stabilize
echo "  Stabilizing..."
sleep 5

# Step 4: Run nanobot
echo ""
echo "[4/5] Running nanobot test..."
PROMPT=$(cat "$ROOT_DIR/prompts/test_prompt.txt")
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="$RESULTS_DIR/nanobot_output_${TIMESTAMP}.txt"

# Backup and swap config (nanobot doesn't support --config flag yet)
if [ -f "$CONFIG_FILE" ]; then
    cp "$CONFIG_FILE" "$CONFIG_BACKUP"
    echo "  Backed up existing config to $CONFIG_BACKUP"
fi
cp "$ROOT_DIR/config.local.json" "$CONFIG_FILE"
echo "  Installed test config"
echo "  Output: $OUTPUT_FILE"
echo ""

# Time the execution
START_TIME=$(date +%s.%N)

cd "$ROOT_DIR"
time "$NANOBOT_BIN" agent -m "$PROMPT" 2>&1 | tee "$OUTPUT_FILE"

END_TIME=$(date +%s.%N)
ELAPSED=$(echo "$END_TIME - $START_TIME" | bc)

# Restore original config (also handled by trap on exit)
if [ -f "$CONFIG_BACKUP" ]; then
    mv "$CONFIG_BACKUP" "$CONFIG_FILE"
    echo "  Restored original config"
fi

echo ""
echo "  Elapsed time: ${ELAPSED}s"

# Save timing
echo "{\"elapsed_seconds\": $ELAPSED, \"timestamp\": \"$TIMESTAMP\"}" > "$RESULTS_DIR/timing_${TIMESTAMP}.json"

# Step 5: Validate results
echo ""
echo "[5/5] Validating results..."
python3 "$SCRIPT_DIR/validate.py" "$OUTPUT_FILE" || true

echo ""
echo "=============================================="
echo "TEST COMPLETE"
echo "=============================================="
echo "Results saved to: $RESULTS_DIR"
echo ""

# Ask if user wants to stop servers
read -p "Stop llama-server cluster? [Y/n] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    "$SCRIPT_DIR/setup.sh" stop
fi
