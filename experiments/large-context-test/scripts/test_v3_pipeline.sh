#!/bin/bash
# test_v3_pipeline.sh - Tests the new pipeline + subagent features post-overhaul
#
# Designed for SLMs (Ministral-8B main, Ministral-3B subagent).
# Each test targets one capability in isolation before combining.

set -e

TEST_DIR="/home/peppi/Dev/nanobot/experiments/large-context-test"
DATA_FILE="$TEST_DIR/data/arxiv_papers.csv"
NANOBOT="/home/peppi/Dev/nanobot/target/release/nanobot"
RESULTS_DIR="$TEST_DIR/results"

mkdir -p "$RESULTS_DIR"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

run_test() {
    local num="$1"
    local name="$2"
    local timeout_sec="$3"
    local prompt="$4"
    local output_file="$RESULTS_DIR/v3_test${num}_output.txt"

    echo ""
    echo -e "${YELLOW}=================================================="
    echo "TEST $num: $name"
    echo -e "==================================================${NC}"
    echo "Timeout: ${timeout_sec}s"
    echo "Output: $output_file"
    echo ""

    if timeout "$timeout_sec" "$NANOBOT" agent --local -s "v3test$num" -m "$prompt" 2>&1 | tee "$output_file"; then
        echo -e "${GREEN}[PASS] Test $num completed${NC}"
    else
        echo -e "${RED}[FAIL] Test $num failed (exit code: $?)${NC}"
    fi

    echo ""
    if [ "${AUTO:-}" != "1" ]; then
        read -p "Press Enter to continue (or Ctrl+C to stop)..."
    fi
}

echo "=================================================="
echo "NANOBOT v3 PIPELINE TEST SUITE (Post-Overhaul)"
echo "=================================================="
echo "Data: $DATA_FILE ($(wc -l < "$DATA_FILE") lines)"
echo "Binary: $NANOBOT"
echo ""

# ---- TEST 1: Basic exec (smoke test) ----
run_test 1 "Basic Exec - smoke test" 60 \
"Use the exec tool to run this command and tell me the result:
head -2 $DATA_FILE"

# ---- TEST 2: Author extraction (single exec) ----
run_test 2 "Author Extraction via exec" 90 \
"Use the exec tool to run this exact command and report the top 5 authors:
cut -d',' -f3 $DATA_FILE | tail -n +2 | tr ';' '\n' | sed 's/^ *//;s/ *$//' | sed 's/^\"//;s/\"\$//' | sort | uniq -c | sort -rn | head -5"

# ---- TEST 3: Keyword search (single exec) ----
run_test 3 "Keyword Search via exec" 90 \
"Use the exec tool to run this exact command. Report which papers mention emergence-related topics:
grep -i -E 'emergence|emergent|scaling law|phase transition|chain-of-thought' $DATA_FILE | cut -d',' -f2,3"

# ---- TEST 4: Subagent spawn (tests Jinja fix) ----
# First create a small filtered file that the subagent can safely read
echo "Creating pre-filtered file for subagent test..."
grep -i "Ming Zhang" "$DATA_FILE" > /tmp/author_ming_zhang.csv 2>/dev/null || true
echo "  /tmp/author_ming_zhang.csv: $(wc -l < /tmp/author_ming_zhang.csv 2>/dev/null || echo 0) lines"

run_test 4 "Single Subagent Spawn (Jinja fix)" 120 \
"Spawn a subagent to analyze a small CSV file. Use the spawn tool with these parameters:
- action: spawn
- task: Read the file /tmp/author_ming_zhang.csv using the exec tool. Run: cat /tmp/author_ming_zhang.csv. Then look for any mentions of emergence, scaling, or chain-of-thought in the text. Report what you find.

Wait for the subagent to finish and report its findings."

# ---- TEST 5: Pipeline action (tests new pipeline feature) ----
run_test 5 "Pipeline Action (3-step)" 180 \
"Use the spawn tool with action=pipeline and these steps:

{
  \"action\": \"pipeline\",
  \"steps\": [
    {
      \"prompt\": \"Run this command and report the result: head -2 $DATA_FILE\",
      \"tools\": [\"exec\"]
    },
    {
      \"prompt\": \"Run this command to find the top 5 most prolific authors. Report each author name and count: cut -d',' -f3 $DATA_FILE | tail -n +2 | tr ';' '\\n' | sed 's/^ *//;s/ *\$//' | sort | uniq -c | sort -rn | head -5\",
      \"tools\": [\"exec\"]
    },
    {
      \"prompt\": \"Run this command to find papers about emergence topics. Report the paper titles and authors: grep -i -E 'emergence|emergent|scaling law|phase transition|chain-of-thought' $DATA_FILE | cut -d',' -f2,3\",
      \"tools\": [\"exec\"]
    }
  ]
}"

# ---- TEST 6: Full workflow (combines exec + spawn + synthesis) ----
run_test 6 "Full Workflow" 240 \
"You have a CSV at $DATA_FILE. DO NOT read it directly.

Step 1: Use exec to find the top 5 authors:
cut -d',' -f3 $DATA_FILE | tail -n +2 | tr ';' '\n' | sed 's/^ *//;s/ *\$//' | sed 's/^\"//;s/\"\$//' | sort | uniq -c | sort -rn | head -5

Step 2: Use exec to find emergence-related papers:
grep -i -E 'emergence|emergent|scaling law|phase transition|chain-of-thought' $DATA_FILE | cut -d',' -f2,3

Step 3: Write a summary listing:
- The top 5 authors with paper counts
- Which authors discuss emergence/scaling topics
- Key findings about emergence in the dataset"

echo ""
echo "=================================================="
echo "ALL v3 TESTS COMPLETE"
echo "=================================================="
echo ""
echo "Results in: $RESULTS_DIR/v3_test*_output.txt"
echo ""
echo "Validate test 6 (full workflow):"
echo "  python3 $TEST_DIR/scripts/validate.py $RESULTS_DIR/v3_test6_output.txt"
