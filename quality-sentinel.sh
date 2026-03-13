#!/usr/bin/env bash
# quality-sentinel.sh — Catch Quality Gate violations that Clippy doesn't cover.
#
# Clippy handles: G1 (bool params), G3 (cognitive complexity), G4 (function length)
# This script handles: G1 (mutable bool flags), G5 (else-if chains 3+), G2 (reminder)
#
# Usage:
#   ./quality-sentinel.sh              # Check all staged/modified .rs files
#   ./quality-sentinel.sh src/agent/    # Check specific path
#   ./quality-sentinel.sh --all        # Check entire src/ tree
#   ./quality-sentinel.sh --full       # Run clippy + sentinel together
#
# Exit codes: 0 = clean, 1 = warnings found

YEL='\033[0;33m'
CYN='\033[0;36m'
BLD='\033[1m'
RED='\033[0;31m'
RST='\033[0m'

TMPFILE=$(mktemp /tmp/sentinel.XXXXXX)
trap "rm -f $TMPFILE" EXIT

warn() {
    printf "${YEL}[G%s]${RST} %s${CYN}:%s${RST} — %s\n" "$1" "$2" "$3" "$4"
    echo "$1" >> "$TMPFILE"
}

# --- Run clippy first if --full ---
if [[ "${1:-}" == "--full" ]]; then
    echo -e "${BLD}Running cargo clippy with Quality Gate lints...${RST}\n"
    cargo clippy --all-targets -- \
        -W clippy::fn_params_excessive_bools \
        -W clippy::struct_excessive_bools \
        -W clippy::cognitive_complexity \
        -W clippy::too_many_lines \
        -W clippy::collapsible_else_if \
        -W clippy::collapsible_if \
        -W clippy::needless_bool \
        -W clippy::match_bool \
        2>&1
    echo ""
    echo -e "${BLD}Now running sentinel for gaps clippy misses...${RST}\n"
    shift || true
fi

# --- Determine files to check ---
if [[ "${1:-}" == "--all" ]]; then
    FILES=$(find src -name '*.rs' -type f 2>/dev/null)
elif [[ -n "${1:-}" ]]; then
    if [[ -d "$1" ]]; then
        FILES=$(find "$1" -name '*.rs' -type f 2>/dev/null)
    else
        FILES="$1"
    fi
else
    FILES=$(git diff --name-only --diff-filter=ACMR HEAD 2>/dev/null | grep '\.rs$' || true)
    STAGED=$(git diff --cached --name-only --diff-filter=ACMR 2>/dev/null | grep '\.rs$' || true)
    FILES=$(printf '%s\n%s' "$FILES" "$STAGED" | sort -u | grep -v '^$' || true)
    if [[ -z "$FILES" ]]; then
        echo "No modified .rs files. Use --all to scan everything, or --full for clippy + sentinel."
        exit 0
    fi
fi

FCOUNT=$(echo "$FILES" | wc -l | tr -d ' ')
echo -e "${BLD}Sentinel — checking ${FCOUNT} file(s) for gaps clippy misses${RST}\n"

# ============================================================
# G1 (supplement): Mutable bool flags — state machine candidates
# Clippy catches bool PARAMETERS but not "let mut done = false" in loops
# ============================================================
echo -e "${BLD}G1: Mutable bool flags (state machine candidates)${RST}"

for f in $FILES; do
    [[ -f "$f" ]] || continue
    grep -nE 'let mut [a-z_]+ = (true|false)' "$f" 2>/dev/null | while IFS=: read -r lineno rest; do
        varname=$(echo "$rest" | grep -oE 'let mut [a-z_]+' | sed 's/let mut //')
        warn 1 "$f" "$lineno" "Mutable bool '${varname}' — state machine? Consider an enum."
    done
done

# ============================================================
# G5: else-if chains with 3+ branches — should be enum + match
# Clippy checks collapsibility, not chain length
# ============================================================
echo -e "\n${BLD}G5: else-if chains (3+ branches)${RST}"

for f in $FILES; do
    [[ -f "$f" ]] || continue
    awk '
    /else if / {
        chain++;
        if (chain == 1) start = NR;
        last = NR;
        next;
    }
    /} else \{/ && chain >= 1 {
        chain++;
        last = NR;
        next;
    }
    {
        if (chain >= 3 && (NR - last) > 1) {
            printf "%d|%d\n", start, chain;
            chain = 0;
        } else if (chain > 0 && (NR - last) > 15) {
            if (chain >= 3) printf "%d|%d\n", start, chain;
            chain = 0;
        }
    }
    END { if (chain >= 3) printf "%d|%d\n", start, chain; }
    ' "$f" 2>/dev/null | while IFS='|' read -r lineno count; do
        warn 5 "$f" "$lineno" "${count}-branch else-if chain — replace with enum + match"
    done
done

# ============================================================
# Summary
# ============================================================
echo ""

TOTAL=$(wc -l < "$TMPFILE" 2>/dev/null | tr -d ' ')
if [[ "$TOTAL" -eq 0 ]]; then
    echo -e "${BLD}All sentinel checks passed.${RST}"
    echo -e "Run ${CYN}./quality-sentinel.sh --full${RST} to include clippy checks too."
    exit 0
fi

G1=$(grep -c '^1$' "$TMPFILE" 2>/dev/null || echo 0)
G5=$(grep -c '^5$' "$TMPFILE" 2>/dev/null || echo 0)

echo -e "${BLD}${RED}${TOTAL} sentinel warning(s):${RST}"
echo -e "  G1 (Mutable bool flags): ${G1}"
echo -e "  G5 (else-if chains 3+):  ${G5}"
echo ""
echo -e "Clippy covers the rest. Run: ${CYN}cargo clippy -- -W clippy::fn_params_excessive_bools -W clippy::cognitive_complexity -W clippy::too_many_lines${RST}"
echo -e "Or run: ${CYN}./quality-sentinel.sh --full${RST} for everything at once."
echo ""
echo -e "G2 (duplication) requires review: ${CYN}ast-grep --pattern '\$PATTERN' --lang rust src/${RST}"
exit 1
