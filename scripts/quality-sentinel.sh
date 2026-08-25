#!/usr/bin/env bash
# scripts/quality-sentinel.sh — error-protocol legacy-channel sentinel.
#
# Fails if (1) the legacy string-classification channels (`from_output`,
# `classify_tool_error`) are used outside their allowed sites, or (2) any
# legacy `set_*_callback` setter exists (host bridge Phase 3 deleted them). These are the
# pre-typed-error parsing paths; the typed error protocol (ToolError /
# ToolResult, error-protocol phase 2) replaces them, and the research doc
# (docs/research/2026-08-06-error-conventions-and-host-bridge.md §3.8) plans
# zero remaining usages by Phase 3.
#
# Allowed sites (see scripts/quality-sentinel.allow):
#   * src/errors.rs — the legacy `classify_tool_error` bridge definition, its
#     `from_legacy` caller, and the mapping-pinning tests.
#   * src/agent/tools/base.rs:142 — legacy `ToolExecutionResult::failure`
#     constructor (deleted in Phase 3).
#   * Comment lines anywhere (`//`), which may reference the symbols
#     descriptively without using them.
#
# Usage:
#   ./scripts/quality-sentinel.sh          # scan the whole src/ tree
# Exit codes: 0 = clean, 1 = violations found
set -euo pipefail
cd "$(dirname "$0")/.."

pattern='from_output|classify_tool_error'
allowfile="$(dirname "$0")/quality-sentinel.allow"

# Build a set of allowed `path` / `path:lineno` keys from the allow file.
allowed_keys() {
    awk -F: '
        /^[[:space:]]*#/ || /^[[:space:]]*$/ { next }
        {
            sub(/[[:space:]]+—.*$/, "");   # strip trailing reason
            gsub(/[[:space:]]/, "");
            if ($0 != "") print
        }
    ' "$allowfile"
}

violations="$(grep -rnE "$pattern" src/ --include='*.rs' \
  | grep -vE '^[^:]+:[0-9]+:[[:space:]]*//' \
  | { while IFS= read -r line; do
        path="${line%%:*}"; lineno="${line#*:}"; lineno="${lineno%%:*}";
        if ! grep -qxF "$path" <(allowed_keys) && ! grep -qxF "$path:$lineno" <(allowed_keys); then
            echo "$line";
        fi;
    done; } \
  || true)"

if [[ -n "$violations" ]]; then
    echo "error-protocol sentinel: legacy from_output/classify_tool_error usage outside allowed sites:" >&2
    echo "$violations" >&2
    echo >&2
    echo "Allowed sites are listed in $allowfile (see also docs/error-protocol-backlog.md)." >&2
    exit 1
fi

echo "error-protocol sentinel OK: no from_output/classify_tool_error usage outside allowed sites."

# Host-bridge legacy callback setters (research doc §3.8): zero
# set_*_callback may exist — Phase 3 deleted the callback slots and the typed
# HostBridge / MessageHost traits replaced them. A new callback-coupled tool
# is a regression against the migration.
callback_pattern='fn set_[a-z_]*callback'
callback_violations="$(grep -rnE "$callback_pattern" src/ --include='*.rs'   | grep -vE '^[^:]+:[0-9]+:[[:space:]]*//'   || true)"

if [[ -n "$callback_violations" ]]; then
    echo "error-protocol sentinel: legacy set_*_callback usage:" >&2
    echo "$callback_violations" >&2
    echo >&2
    echo "Callback slots were deleted in Phase 3 (host bridge, research doc §3.8);" >&2
    echo "use the typed HostBridge / MessageHost traits instead." >&2
    exit 1
fi

echo "error-protocol sentinel OK: no set_*_callback usage."
