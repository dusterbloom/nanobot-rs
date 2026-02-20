#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"

if [[ -z "$MODE" ]]; then
  echo "usage: $0 [red|green]"
  exit 2
fi

run_expect_fail() {
  local cmd="$1"
  echo "RED   > $cmd"
  if eval "$cmd" >/tmp/nanobot-red.log 2>&1; then
    echo "expected failure, but command passed"
    tail -n 40 /tmp/nanobot-red.log || true
    exit 1
  fi
}

run_expect_pass() {
  local cmd="$1"
  echo "GREEN > $cmd"
  eval "$cmd"
}

case "$MODE" in
  red)
    export BRAVE_API_KEY="force-red-path"
    run_expect_fail "cargo test repl::commands::tests::test_normalize_alias_all_aliases -- --test-threads=1"
    run_expect_fail "cargo test agent::tools::web::tests::test_web_search_no_api_key -- --test-threads=1"
    run_expect_fail "cargo test agent::observer::tests::test_archive_moves_files -- --test-threads=1"
    run_expect_fail "cargo test agent::working_memory::tests::test_archive_session -- --test-threads=1"
    run_expect_fail "cargo test agent::reflector::tests::test_reflect_archives_completed_sessions -- --test-threads=1"
    run_expect_fail "cargo test server::tests::test_find_available_port_skips_occupied -- --test-threads=1"
    run_expect_fail "cargo test config::schema::tests::test_local_vllm_provider_selected_when_cloud_disabled -- --test-threads=1"
    echo
    echo "red phase confirmed: known regressions are reproducible"
    ;;

  green)
    export BRAVE_API_KEY="force-green-verification"
    run_expect_pass "cargo test repl::commands::tests::test_normalize_alias_all_aliases -- --test-threads=1"
    run_expect_pass "cargo test agent::tools::web::tests::test_web_search_no_api_key -- --test-threads=1"
    run_expect_pass "cargo test agent::observer::tests::test_archive_moves_files -- --test-threads=1"
    run_expect_pass "cargo test agent::working_memory::tests::test_archive_session -- --test-threads=1"
    run_expect_pass "cargo test agent::reflector::tests::test_reflect_archives_completed_sessions -- --test-threads=1"
    run_expect_pass "cargo test server::tests::test_find_available_port_skips_occupied -- --test-threads=1"
    run_expect_pass "cargo test cli::tests::test_build_core_handle_local_forces_local_provider_even_with_cloud_keys -- --test-threads=1"
    run_expect_pass "cargo test cli::tests::test_make_eval_provider_local_uses_local_endpoint -- --test-threads=1"
    run_expect_pass "cargo test cli::tests::test_eval_model_name_local_is_port_scoped -- --test-threads=1"
    run_expect_pass "cargo test config::schema::tests::test_local_vllm_provider_selected_when_cloud_disabled -- --test-threads=1"
    echo
    echo "green phase confirmed: regressions fixed + local-only provider wiring validated"
    ;;

  *)
    echo "unknown mode: $MODE"
    echo "usage: $0 [red|green]"
    exit 2
    ;;
esac
