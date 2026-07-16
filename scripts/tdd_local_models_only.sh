#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"

if [[ -z "$MODE" ]]; then
  echo "usage: $0 [red|green]"
  exit 2
fi

ORIGINAL_HOME="$HOME"
export CARGO_HOME="${CARGO_HOME:-$ORIGINAL_HOME/.cargo}"
export RUSTUP_HOME="${RUSTUP_HOME:-$ORIGINAL_HOME/.rustup}"
TEST_HOME="$(mktemp -d "${TMPDIR:-/tmp}/nanobot-local-tdd.XXXXXX")"
trap 'rm -rf "$TEST_HOME"' EXIT
export HOME="$TEST_HOME"

run_expect_fail() {
  local test_name="$1"
  local cmd=(cargo test --lib "$test_name" -- --exact --test-threads=1)
  echo "RED   > ${cmd[*]}"
  if "${cmd[@]}" >/tmp/nanobot-red.log 2>&1; then
    echo "expected failure, but command passed"
    tail -n 40 /tmp/nanobot-red.log || true
    exit 1
  fi
}

run_expect_pass() {
  local test_name="$1"
  local cmd=(cargo test --lib "$test_name" -- --exact --test-threads=1)
  echo "GREEN > ${cmd[*]}"
  "${cmd[@]}" 2>&1 | tee /tmp/nanobot-green.log
  if ! grep -Fq "test $test_name ... ok" /tmp/nanobot-green.log; then
    echo "test filter matched no test: $test_name"
    exit 1
  fi
}

case "$MODE" in
  red)
    export BRAVE_API_KEY="force-red-path"
    run_expect_fail "repl::commands::tests::test_normalize_alias_all_aliases"
    run_expect_fail "agent::tools::web::tests::test_web_search_no_api_key"
    run_expect_fail "agent::working_memory::tests::lifecycle_is_persisted_in_sqlite"
    run_expect_fail "agent::reflector::tests::test_reflect_marks_completed_sessions_reflected"
    run_expect_fail "server::tests::test_find_available_port_skips_occupied"
    run_expect_fail "config::schema::tests::test_local_vllm_provider_selected_when_cloud_disabled"
    echo
    echo "red phase confirmed: known regressions are reproducible"
    ;;

  green)
    export BRAVE_API_KEY="force-green-verification"
    run_expect_pass "repl::commands::tests::test_normalize_alias_all_aliases"
    run_expect_pass "agent::tools::web::tests::test_web_search_no_api_key"
    run_expect_pass "agent::working_memory::tests::lifecycle_is_persisted_in_sqlite"
    run_expect_pass "agent::reflector::tests::test_reflect_marks_completed_sessions_reflected"
    run_expect_pass "server::tests::test_find_available_port_skips_occupied"
    run_expect_pass "cli::tests::test_build_core_handle_local_forces_local_provider_even_with_cloud_keys"
    run_expect_pass "config::schema::tests::test_local_vllm_provider_selected_when_cloud_disabled"
    run_expect_pass "local_discovery::tests::test_decide_no_server_and_autostart_off_is_note_not_spawn"
    run_expect_pass "local_discovery::tests::test_decide_no_server_spawns_only_with_explicit_autostart"
    run_expect_pass "local_discovery::tests::test_candidates_cover_configured_higgs_lms_and_cluster"
    run_expect_pass "higgs::tests::compaction_manager_respects_explicit_higgs_autostart"
    run_expect_pass "higgs::tests::compaction_manager_never_spawns_for_lmstudio_autostart"
    echo
    echo "green phase confirmed: regressions fixed + local discovery/autostart wiring validated"
    ;;

  *)
    echo "unknown mode: $MODE"
    echo "usage: $0 [red|green]"
    exit 2
    ;;
esac
