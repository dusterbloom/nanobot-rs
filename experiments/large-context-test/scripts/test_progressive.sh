#!/bin/bash
# Progressive tests to validate each component of the multi-agent workflow

set -e

TEST_DIR="/home/peppi/Dev/nanobot/experiments/large-context-test"
DATA_FILE="$TEST_DIR/data/arxiv_papers.csv"
NANOBOT="/home/peppi/Dev/nanobot/target/release/nanobot"

run_test() {
    local num="$1"
    local name="$2"
    local timeout_sec="$3"
    local prompt="$4"
    
    echo ""
    echo "=================================================="
    echo "TEST $num: $name"
    echo "=================================================="
    echo "Timeout: ${timeout_sec}s"
    echo ""
    
    timeout "$timeout_sec" "$NANOBOT" agent --local -s "test$num" -m "$prompt" 2>&1 | tee "$TEST_DIR/results/test${num}_output.txt"
    
    echo ""
    echo "Test $num output saved to: $TEST_DIR/results/test${num}_output.txt"
    echo ""
    read -p "Press Enter to continue to next test (or Ctrl+C to stop)..."
}

echo "=================================================="
echo "PROGRESSIVE MULTI-AGENT TEST SUITE"
echo "=================================================="
echo "Data file: $DATA_FILE"
echo ""

# Test 1: Shell Commands Only
run_test 1 "Shell Commands Only" 60 "
Use the exec tool to run these commands and report results:
1. head -1 $DATA_FILE
2. wc -l $DATA_FILE
Report what you found.
"

# Test 2: Author Extraction
run_test 2 "Author Extraction" 90 "
Use the exec tool with shell commands to find the top 3 authors in this CSV file:
$DATA_FILE

The authors are in column 3, with multiple authors per row separated by semicolons.

Use this pipeline:
cut -d',' -f3 $DATA_FILE | tail -n +2 | tr ';' '\n' | sed 's/^ *//;s/ *$//' | sort | uniq -c | sort -rn | head -3

Report the top 3 authors and their paper counts.
"

# Test 3: Pre-filter Data
run_test 3 "Pre-filter Data" 120 "
Run these exec commands to create filtered temp files:

First, get the top author:
TOP_AUTHOR=\$(cut -d',' -f3 $DATA_FILE | tail -n +2 | tr ';' '\n' | sed 's/^ *//;s/ *$//' | sort | uniq -c | sort -rn | head -1 | awk '{print \$2\" \"\$3\" \"\$4}')

Then filter their data:
grep -i \"\$TOP_AUTHOR\" $DATA_FILE > /tmp/author_top.csv

Verify:
wc -l /tmp/author_top.csv
head -1 /tmp/author_top.csv

Report how many rows were extracted.
"

# Test 4: Single Subagent
run_test 4 "Single Subagent Spawn" 120 "
Spawn ONE subagent with action=spawn and this exact task:

'Search the file /tmp/author_top.csv for any mentions of emergence, scaling, or phase transition. Use grep to find these terms in the abstracts. Report what you find.'

Wait for the subagent to complete and report its findings.
"

# Test 5: Full Workflow
echo ""
echo "=================================================="
echo "TEST 5: Full Multi-Agent Workflow"
echo "=================================================="
echo "This is the complete test from test_prompt_v2.txt"
echo ""

PROMPT=$(cat "$TEST_DIR/prompts/test_prompt_v2.txt")
timeout 300 "$NANOBOT" agent --local -s "test5" -m "$PROMPT" 2>&1 | tee "$TEST_DIR/results/test5_output.txt"

echo ""
echo "=================================================="
echo "ALL TESTS COMPLETE"
echo "=================================================="
echo ""
echo "Run validation:"
echo "  python3 $TEST_DIR/scripts/validate.py $TEST_DIR/results/test5_output.txt"
