#!/usr/bin/env bash
# Sanitize JSONL session files for cleaner qmd indexing.
#
# Strips noise (tool call JSON, tool results, metadata) and keeps
# only user + assistant conversation text. Output goes to
# ~/.nanobot/sessions-clean/ with the same filenames.
#
# Usage: bash scripts/sanitize-sessions.sh

set -euo pipefail

SRC="${HOME}/.nanobot/sessions"
DST="${HOME}/.nanobot/sessions-clean"

if [ ! -d "$SRC" ]; then
    echo "Error: $SRC not found"
    exit 1
fi

mkdir -p "$DST"

count=0
skipped=0

for f in "$SRC"/*.jsonl; do
    [ -f "$f" ] || continue
    base=$(basename "$f")
    out="$DST/$base"

    # Skip if clean file is newer than source
    if [ -f "$out" ] && [ "$out" -nt "$f" ]; then
        skipped=$((skipped + 1))
        continue
    fi

    # Process: keep user+assistant, strip tool_calls, skip tool/metadata/clear
    jq -c '
        # Skip non-conversation lines
        if .role == "tool" then empty
        elif ._type == "metadata" then empty
        elif .role == "clear" then empty
        elif .role == "system" then empty
        # For assistant messages: keep content, strip tool_calls
        elif .role == "assistant" then
            {role, content: (.content // ""), timestamp: .timestamp}
            | if .content == "" then empty else . end
        # For user messages: keep as-is but slim down
        elif .role == "user" then
            {role, content: (.content // ""), timestamp: .timestamp}
            | if ._turn then . + {_turn: ._turn} else . end
            | if .content == "" then empty else . end
        else empty
        end
    ' "$f" > "$out" 2>/dev/null || {
        # If jq fails (malformed JSON), copy raw but filter by grep
        grep -E '"role":"(user|assistant)"' "$f" | \
            jq -c '{role, content: (.content // ""), timestamp: .timestamp} | if .content == "" then empty else . end' \
            > "$out" 2>/dev/null || true
    }

    count=$((count + 1))
done

echo "Sanitized $count files ($skipped skipped, up-to-date)"
echo "Clean sessions at: $DST"

# Show size comparison
src_size=$(du -sh "$SRC" 2>/dev/null | cut -f1)
dst_size=$(du -sh "$DST" 2>/dev/null | cut -f1)
echo "Original: $src_size → Clean: $dst_size"
