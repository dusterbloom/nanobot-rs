#!/usr/bin/env bash
# Foreground Higgs launcher for the C-first KV-cache gate. Start this script as
# the tmux pane command: its shell PID is preserved when it execs Higgs, so
# `tmux display-message -p -t SESSION '#{pane_pid}'` is valid gate evidence.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: c_first_kv_cache_launch_higgs.sh \
  --higgs-bin /absolute/path/to/higgs \
  --server-config /absolute/path/to/higgs.toml \
  --model-path /absolute/path/to/model \
  --server-log /absolute/path/to/higgs.log

The launcher validates every path, emits exactly one startup binding JSON
record, tees stdout/stderr to --server-log, and then runs in the foreground as:
  HIGGS_BIN --config SERVER_CONFIG serve

It does not create a tmux session. Invoke it directly as the tmux pane command.
Run `c_first_kv_cache_launch_higgs.sh --self-test` for offline path-safety tests.
EOF
}

fail() {
    printf 'launcher error: %s\n' "$*" >&2
    exit 2
}

launcher_self_test() {
    local test_root launcher fake_bin config model_dir protected log_path before after output status
    test_root=$(mktemp -d "${TMPDIR:-/tmp}/c-first-launcher.XXXXXX")
    trap 'rm -rf -- "$test_root"' RETURN
    launcher=$(python3 -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' "${BASH_SOURCE[0]}")
    fake_bin="$test_root/higgs"
    config="$test_root/higgs.toml"
    model_dir="$test_root/model"
    protected="$test_root/protected"
    mkdir "$model_dir"
    printf '#!/bin/sh\nexit 91\n' >"$fake_bin"
    chmod +x "$fake_bin"
    printf 'config sentinel\n' >"$config"
    printf 'protected sentinel\n' >"$protected"

    reject_unchanged() {
        local label=$1
        before=$(shasum -a 256 "$protected" | awk 'NR == 1 {print $1}')
        set +e
        output=$("$launcher" --higgs-bin "$fake_bin" --server-config "$config" \
            --model-path "$model_dir" --server-log "$log_path" 2>&1)
        status=$?
        set -e
        after=$(shasum -a 256 "$protected" | awk 'NR == 1 {print $1}')
        [[ $status == 2 ]] || fail "self-test $label was not rejected before exec: $status $output"
        [[ $before == "$after" ]] || fail "self-test $label modified its protected target"
    }

    log_path="$test_root/symlink.log"
    ln -s "$protected" "$log_path"
    reject_unchanged "symlink log"
    rm -f -- "$log_path"

    protected=$config
    log_path="$test_root/config-hardlink.log"
    ln "$protected" "$log_path"
    reject_unchanged "config hardlink log"
    rm -f -- "$log_path"

    protected=$fake_bin
    log_path="$test_root/binary-hardlink.log"
    ln "$protected" "$log_path"
    reject_unchanged "binary hardlink log"
    printf 'launcher self-test: PASS\n'
}

canonical_existing() {
    python3 -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' "$1"
}

higgs_bin=
server_config=
model_path=
server_log=

if (($# == 1)) && [[ $1 == --self-test ]]; then
    launcher_self_test
    exit 0
fi

while (($#)); do
    case "$1" in
        --higgs-bin|--server-config|--model-path|--server-log)
            (($# >= 2)) || fail "$1 requires a value"
            [[ -n $2 ]] || fail "$1 requires a nonempty value"
            case "$1" in
                --higgs-bin) [[ -z $higgs_bin ]] || fail "$1 specified twice"; higgs_bin=$2 ;;
                --server-config) [[ -z $server_config ]] || fail "$1 specified twice"; server_config=$2 ;;
                --model-path) [[ -z $model_path ]] || fail "$1 specified twice"; model_path=$2 ;;
                --server-log) [[ -z $server_log ]] || fail "$1 specified twice"; server_log=$2 ;;
            esac
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *) fail "unknown argument: $1" ;;
    esac
done

[[ -n $higgs_bin ]] || fail "--higgs-bin is required"
[[ -n $server_config ]] || fail "--server-config is required"
[[ -n $model_path ]] || fail "--model-path is required"
[[ -n $server_log ]] || fail "--server-log is required"

command -v python3 >/dev/null 2>&1 || fail "python3 is required for canonical paths and JSON"
command -v shasum >/dev/null 2>&1 || fail "shasum is required for config hashing"
command -v tee >/dev/null 2>&1 || fail "tee is required for foreground log capture"

[[ -f $higgs_bin && -x $higgs_bin ]] || fail "--higgs-bin must be an executable file"
[[ -f $server_config && -r $server_config ]] || fail "--server-config must be a readable file"
[[ -d $model_path && -r $model_path ]] || fail "--model-path must be a readable directory"
[[ ! -L $server_log ]] || fail "--server-log must not be a symlink"
if [[ -e $server_log ]]; then
    [[ -f $server_log && -w $server_log ]] \
        || fail "--server-log must be a writable regular file when it exists"
    [[ ! $server_log -ef $higgs_bin ]] || fail "--server-log aliases --higgs-bin"
    [[ ! $server_log -ef $server_config ]] || fail "--server-log aliases --server-config"
    [[ ! $server_log -ef $model_path ]] || fail "--server-log aliases --model-path"
fi
log_parent=$(canonical_existing "$(dirname -- "$server_log")") \
    || fail "--server-log parent directory does not exist"
[[ -d $log_parent && -w $log_parent ]] || fail "--server-log parent must be writable"

higgs_bin=$(canonical_existing "$higgs_bin") || fail "cannot canonicalize --higgs-bin"
server_config=$(canonical_existing "$server_config") \
    || fail "cannot canonicalize --server-config"
model_path=$(canonical_existing "$model_path") || fail "cannot canonicalize --model-path"
server_log="$log_parent/$(basename -- "$server_log")"
[[ $server_log != "$server_config" ]] || fail "--server-log resolves to --server-config"
[[ $server_log != "$higgs_bin" ]] || fail "--server-log resolves to --higgs-bin"
[[ $server_log != "$model_path" ]] || fail "--server-log resolves to --model-path"

config_sha256=$(shasum -a 256 "$server_config" | awk 'NR == 1 {print $1}')
[[ $config_sha256 =~ ^[0-9a-fA-F]{64}$ ]] || fail "could not hash exact config bytes"
binding_record=$(python3 -c '
import json, sys
print(json.dumps({
    "kind": "c_first_kv_cache_server_start",
    "pid": int(sys.argv[1]),
    "config_path": sys.argv[2],
    "config_sha256": sys.argv[3].lower(),
    "model_path": sys.argv[4],
}, separators=(",", ":"), sort_keys=True))
' "$$" "$server_config" "$config_sha256" "$model_path") \
    || fail "could not encode startup binding record"

# Process substitution keeps this shell out of a pipeline. The following exec
# therefore preserves $$ as the real Higgs PID while tee remains a child.
exec > >(tee -a -- "$server_log") 2>&1
printf '%s\n' "$binding_record"
exec "$higgs_bin" --config "$server_config" serve
