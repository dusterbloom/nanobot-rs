#!/usr/bin/env python3
"""Blocking live gate for Higgs retained-session KV reuse.

This script never starts or stops Higgs. Run the service separately in an
inspectable tmux pane, tee its output to --server-log, then run this script in
another pane. Results are append-only JSONL plus a final summary JSON.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import http.client
import json
import os
import re
import shlex
import statistics
import subprocess
import sys
import tempfile
import time
import tomllib
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


DEFAULT_MODEL_PATH = Path(
    "/Users/peppi/.cache/lm-studio/models/mlx-community/Qwen3.6-35B-A3B-4bit"
)
TARGET_CONFIG = {
    "max_retained_sessions": 2,
    "max_session_tokens": 32_768,
    "max_suffix_prefill_tokens": 24_576,
    "idle_seconds": 300,
    "max_retained_bytes": 2_147_483_648,
}
CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}
SELF_TEST_CONFIG_TEXT = f"""
[[models]]
name = "Qwen3.6-35B-A3B-4bit"
path = "{DEFAULT_MODEL_PATH}"
kv_max_sessions = 2
kv_max_session_tokens = 32768
kv_max_suffix_prefill_tokens = 24576
kv_retained_idle_secs = 300
kv_max_retained_bytes = 2147483648
"""
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "echo_probe",
            "description": "Echo a probe string when explicitly requested.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    }
]
ANSWER_WORDS = (
    "amber",
    "river",
    "forest",
    "silver",
    "earth",
    "star",
    "golden",
    "east",
    "blue",
    "green",
    "red",
    "white",
    "black",
    "stone",
    "light",
    "night",
    "water",
    "fire",
    "wind",
    "cloud",
    "rain",
    "snow",
    "sun",
    "moon",
    "north",
    "south",
    "west",
    "field",
    "tree",
    "bird",
    "lake",
    "sky",
)
FATAL_LOG_PATTERN = re.compile(
    r"SIGABRT|out[ -]of[ -]memory|\bOOM\b|stale reuse|malformed tool",
    re.IGNORECASE,
)
REQUIRED_METRICS = (
    "retained_sessions",
    "retained_bytes",
    "active_leases",
    "expired_leases",
    "broken_leases",
    "session_bootstrap_exact",
    "session_bootstrap_pflash",
    "required_continuation_misses",
    "prefill_only_requests",
)
VALID_FINISH_REASONS = {"stop", "length"}


class GateError(RuntimeError):
    pass


@dataclass(frozen=True)
class Endpoint:
    scheme: str
    host: str
    port: int
    prefix: str

    @classmethod
    def parse(cls, base_url: str) -> "Endpoint":
        parsed = urlparse(base_url.rstrip("/"))
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise GateError(f"invalid --base-url: {base_url}")
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        return cls(parsed.scheme, parsed.hostname, port, parsed.path.rstrip("/"))

    def connection(self, timeout: float) -> http.client.HTTPConnection:
        connection_type = (
            http.client.HTTPSConnection if self.scheme == "https" else http.client.HTTPConnection
        )
        return connection_type(self.host, self.port, timeout=timeout)

    def api_path(self, suffix: str) -> str:
        return f"{self.prefix}{suffix}"

    def root_path(self, suffix: str) -> str:
        root = self.prefix[:-3] if self.prefix.endswith("/v1") else self.prefix
        return f"{root}{suffix}" or "/"


@dataclass(frozen=True)
class LogCursor:
    path: Path
    device: int
    inode: int
    offset: int


def capture_log_cursor(path: Path) -> LogCursor:
    stat = path.stat()
    return LogCursor(path.resolve(), stat.st_dev, stat.st_ino, stat.st_size)


def scan_server_log(cursor: LogCursor) -> tuple[list[str], str]:
    stat = cursor.path.stat()
    if (stat.st_dev, stat.st_ino) != (cursor.device, cursor.inode):
        raise GateError("server log rotated during gate")
    if stat.st_size < cursor.offset:
        raise GateError("server log truncated during gate")
    with cursor.path.open("r", encoding="utf-8", errors="replace") as stream:
        stream.seek(cursor.offset)
        text = stream.read()
    matches = sorted(set(match.group(0) for match in FATAL_LOG_PATTERN.finditer(text)))
    return matches, text


def write_all_bytes(stream: Any, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = stream.write(payload[offset:])
        remaining = len(payload) - offset
        if (
            isinstance(written, bool)
            or not isinstance(written, int)
            or written <= 0
            or written > remaining
        ):
            raise GateError(f"evidence write made invalid progress: {written!r}")
        offset += written


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    encoded = (json.dumps(record, sort_keys=True) + "\n").encode()
    with path.open("ab", buffering=0) as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            write_all_bytes(stream, encoded)
            os.fsync(stream.fileno())
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def write_summary_atomic(path: Path, summary: dict[str, Any]) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def request_json(
    endpoint: Endpoint,
    method: str,
    path: str,
    timeout: float,
    body: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any], float]:
    encoded = None if body is None else json.dumps(body, separators=(",", ":")).encode()
    headers = {"accept": "application/json"}
    if encoded is not None:
        headers["content-type"] = "application/json"
        headers["content-length"] = str(len(encoded))
    connection = endpoint.connection(timeout)
    started = time.perf_counter()
    try:
        connection.request(method, path, body=encoded, headers=headers)
        response = connection.getresponse()
        payload = response.read()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
    finally:
        connection.close()
    try:
        decoded = json.loads(payload) if payload else {}
    except json.JSONDecodeError as error:
        raise GateError(f"{method} {path} returned non-JSON HTTP {response.status}") from error
    if response.status >= 400:
        raise GateError(f"{method} {path} returned HTTP {response.status}: {decoded}")
    return response.status, decoded, elapsed_ms


def stream_chat(
    endpoint: Endpoint, body: dict[str, Any], timeout: float
) -> dict[str, Any]:
    encoded = json.dumps(body, separators=(",", ":")).encode()
    connection = endpoint.connection(timeout)
    started = time.perf_counter()
    first_token_ms: float | None = None
    data_events: list[str] = []
    try:
        connection.request(
            "POST",
            endpoint.api_path("/chat/completions"),
            body=encoded,
            headers={
                "content-type": "application/json",
                "content-length": str(len(encoded)),
                "accept": "text/event-stream",
            },
        )
        response = connection.getresponse()
        if response.status >= 400:
            payload = response.read().decode(errors="replace")
            raise GateError(f"chat returned HTTP {response.status}: {payload}")
        while True:
            raw_line = response.readline()
            if not raw_line:
                break
            line = raw_line.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            data_events.append(data)
            if data != "[DONE]" and first_token_ms is None:
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    event = {}
                for choice in event.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if delta.get("content") or delta.get("reasoning_content") or delta.get("tool_calls"):
                        first_token_ms = (time.perf_counter() - started) * 1000.0
        wall_ms = (time.perf_counter() - started) * 1000.0
    finally:
        connection.close()

    parsed = parse_sse_data(data_events)
    return {
        "status": response.status,
        "ttft_ms": first_token_ms,
        "wall_ms": wall_ms,
        **parsed,
    }


def validate_usage(usage: Any, *, completion_tokens: int | None = None) -> dict[str, Any]:
    if not isinstance(usage, dict):
        raise GateError("response is missing its usage object")
    for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise GateError(f"usage.{name} must be a nonnegative integer")
    if usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]:
        raise GateError("usage total_tokens is inconsistent")
    if completion_tokens is not None and usage["completion_tokens"] != completion_tokens:
        raise GateError(f"usage completion_tokens must equal {completion_tokens}")
    details = usage.get("prompt_tokens_details")
    if details is not None:
        if not isinstance(details, dict):
            raise GateError("usage.prompt_tokens_details must be an object when present")
        cached = details.get("cached_tokens")
        if isinstance(cached, bool) or not isinstance(cached, int) or cached < 0:
            raise GateError("usage.prompt_tokens_details.cached_tokens must be a nonnegative integer")
        if cached > usage["prompt_tokens"]:
            raise GateError("usage cached_tokens cannot exceed prompt_tokens")
    lease = usage.get("higgs_session_lease_active")
    if lease not in (None, 1):
        raise GateError("usage.higgs_session_lease_active must be 1 when present")
    return usage


def parse_sse_data(data_events: list[str]) -> dict[str, Any]:
    done_count = 0
    usage_envelopes: list[dict[str, Any]] = []
    finish_reasons: list[str] = []
    text_parts: list[str] = []
    tool_fragments: dict[int, dict[str, str]] = {}
    for data in data_events:
        if data == "[DONE]":
            done_count += 1
            continue
        if done_count:
            raise GateError("SSE data appeared after [DONE]")
        try:
            event = json.loads(data)
        except json.JSONDecodeError as error:
            raise GateError(f"malformed SSE JSON: {data[:200]}") from error
        if not isinstance(event, dict):
            raise GateError("SSE event must be a JSON object")
        if "error" in event:
            raise GateError(f"SSE error object: {event['error']}")
        if event.get("usage") is not None:
            usage_envelopes.append(event["usage"])
        for choice in event.get("choices") or []:
            finish_reason = choice.get("finish_reason")
            if finish_reason is not None:
                finish_reasons.append(finish_reason)
            delta = choice.get("delta") or {}
            content = delta.get("content") or ""
            if content:
                text_parts.append(content)
            for call in delta.get("tool_calls") or []:
                index = int(call.get("index", 0))
                item = tool_fragments.setdefault(index, {"id": "", "name": "", "arguments": ""})
                item["id"] += call.get("id") or ""
                function = call.get("function") or {}
                item["name"] += function.get("name") or ""
                item["arguments"] += function.get("arguments") or ""
    if done_count != 1:
        raise GateError(f"SSE must contain exactly one [DONE], observed {done_count}")
    if len(usage_envelopes) != 1:
        raise GateError(f"SSE must contain exactly one usage envelope, observed {len(usage_envelopes)}")
    if len(finish_reasons) != 1 or finish_reasons[0] not in VALID_FINISH_REASONS:
        raise GateError(f"SSE must contain one valid finish reason: {finish_reasons}")
    usage = validate_usage(usage_envelopes[0])
    return {
        "usage": usage,
        "text": "".join(text_parts),
        "finish_reason": finish_reasons[0],
        "tool_calls": list(tool_fragments.values()),
    }


def blocking_chat(
    endpoint: Endpoint, body: dict[str, Any], timeout: float
) -> dict[str, Any]:
    status, response, wall_ms = request_json(
        endpoint, "POST", endpoint.api_path("/chat/completions"), timeout, body
    )
    return {"status": status, "wall_ms": wall_ms, "response": response}


def validate_prefill(
    result: dict[str, Any], *, expect_lease_ack: bool
) -> dict[str, Any]:
    if result.get("status") != 200:
        raise GateError("prefill response must be HTTP 200")
    response = result.get("response")
    if not isinstance(response, dict):
        raise GateError("prefill response must be an object")
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise GateError("prefill response must contain exactly one choice")
    choice = choices[0]
    message = choice.get("message") or {}
    if message.get("role") != "assistant" or message.get("content") != "":
        raise GateError("prefill response must contain one empty assistant choice")
    if message.get("tool_calls") not in (None, []):
        raise GateError("prefill response must not contain tool calls")
    if choice.get("finish_reason") != "length":
        raise GateError("prefill response finish_reason must be length")
    usage = validate_usage(response.get("usage"), completion_tokens=0)
    if cached_tokens(usage) != 0:
        raise GateError("prefill-only response must not report cached prompt tokens")
    lease_ack = usage.get("higgs_session_lease_active")
    if expect_lease_ack and lease_ack != 1:
        raise GateError("lease-carrying prefill response is missing lease acknowledgement")
    if not expect_lease_ack and lease_ack is not None:
        raise GateError("non-lease prefill response emitted a lease acknowledgement")
    return usage


def cache_view(metrics: dict[str, Any]) -> dict[str, int]:
    cache = metrics.get("cache")
    if not isinstance(cache, dict):
        raise GateError("metrics response is missing the cache object")
    view = {}
    for name in REQUIRED_METRICS:
        value = cache.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise GateError(f"metrics cache.{name} must be a nonnegative integer")
        view[name] = value
    return view


def cached_tokens(usage: dict[str, Any]) -> int:
    details = validate_usage(usage).get("prompt_tokens_details")
    return 0 if details is None else details["cached_tokens"]


def validate_delete_response(
    session_id: int, response: Any, *, expected_dropped: int | None = None
) -> int:
    if not isinstance(response, dict) or response.get("session_id") != session_id:
        raise GateError(f"DELETE response must identify session {session_id}")
    dropped = response.get("dropped")
    if isinstance(dropped, bool) or not isinstance(dropped, int) or dropped not in (0, 1):
        raise GateError("DELETE response dropped must be exactly 0 or 1")
    if expected_dropped is not None and dropped != expected_dropped:
        raise GateError(f"DELETE response dropped must equal {expected_dropped}")
    return dropped


def require_cache_state(
    metrics: dict[str, int], expected: dict[str, int], context: str
) -> None:
    names = ("retained_sessions", "retained_bytes", "active_leases")
    actual_state = {name: metrics[name] for name in names}
    expected_state = {name: expected[name] for name in names}
    if actual_state != expected_state:
        raise GateError(f"{context} cache state mismatch: expected {expected_state}, got {actual_state}")


def metric_delta(after: dict[str, int], before: dict[str, int], name: str) -> int:
    return after[name] - before[name]


def retained_bootstrap_delta(after_retained: dict[str, int], after_b: dict[str, int]) -> int:
    return metric_delta(after_retained, after_b, "session_bootstrap_exact") + metric_delta(
        after_retained, after_b, "session_bootstrap_pflash"
    )


def request_record(run_id: str, step: str, body: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    messages = body.get("messages") or []
    canonical_messages = json.dumps(messages, sort_keys=True, separators=(",", ":")).encode()
    return {
        "kind": "step",
        "run_id": run_id,
        "step": step,
        "request": {
            "session_id_present": "session_id" in body,
            "session_id": body.get("session_id"),
            "drop_session_id": body.get("drop_session_id"),
            "drop_session_ids": body.get("drop_session_ids"),
            "session_lease": body.get("session_lease"),
            "session_cache_policy": body.get("session_cache_policy"),
            "cache_mode": body.get("cache_mode"),
            "max_prompt_tokens": body.get("max_prompt_tokens"),
            "max_tokens": body.get("max_tokens"),
            "message_count": len(messages),
            "tool_count": len(body.get("tools") or []),
            "message_bytes": len(canonical_messages),
            "messages_sha256": hashlib.sha256(canonical_messages).hexdigest(),
        },
        "result": result,
    }


def evaluate(summary: dict[str, Any]) -> dict[str, Any]:
    trials = summary["trials"]
    retained_ttft = [trial["retained"]["ttft_ms"] for trial in trials]
    cold_ttft = [trial["cold"]["ttft_ms"] for trial in trials]
    checks: dict[str, bool] = {
        "three_pairs": len(trials) == 3,
        "health_ok": summary["health_before"].get("status") == "ok"
        and summary["health_after"].get("status") == "ok",
        "fixed_prompt_range": all(
            13_000 <= trial["prefill_prompt_tokens"] <= 20_000 for trial in trials
        ),
        "cached_prefix_exact": all(
            trial["retained_cached_tokens"] == trial["prepared_retained_prefix_tokens"]
            and trial["retained_cached_tokens"] > 0
            for trial in trials
        ),
        "lease_confirmed": all(trial["lease_active"] == 1 for trial in trials),
        "cold_is_not_retained_reuse": all(trial["cold_cached_tokens"] == 0 for trial in trials),
        "cold_requests_are_stateless": all(trial["cold_request_stateless"] for trial in trials),
        "prefill_contracts_valid": all(trial["prefills_valid"] for trial in trials),
        "prefill_counter_incremented": all(
            trial["prefill_a_counter_delta"] == 1 and trial["prefill_b_counter_delta"] == 1
            for trial in trials
        ),
        "metrics_sampled_every_mutation": all(
            trial["metric_steps"]
            == [
                "before_a",
                "after_a",
                "after_b",
                "after_retained",
                "after_cold",
                "after_release",
                "after_trial_cleanup",
            ]
            for trial in trials
        ),
        "matching_comparator_prompt_counts": all(
            trial["retained_prompt_tokens"] == trial["cold_prompt_tokens"] for trial in trials
        ),
        "no_bootstrap_fallback": all(trial["retained_bootstrap_delta"] == 0 for trial in trials),
        "ttft_complete": all(value is not None for value in retained_ttft + cold_ttft),
        "bounded_sessions": summary["max_observed_retained_sessions"] <= 2,
        "bounded_bytes": summary["max_observed_retained_bytes"] <= 2_147_483_648,
        "bounded_active_leases": summary["max_observed_active_leases"] <= 2,
        "explicit_release_reduced_state": all(trial["release_reduced_state"] for trial in trials),
        "no_broken_or_expired_lease": all(
            trial["lease_failure_delta"] == 0 for trial in trials
        ),
        "no_required_miss": all(trial["required_miss_delta"] == 0 for trial in trials),
        "zero_tool_calls": all(
            not trial["retained"]["tool_calls"] and not trial["cold"]["tool_calls"]
            for trial in trials
        ),
        "exact_fact_recovery": all(
            trial["retained"]["text"] == trial["answer_marker"]
            and trial["cold"]["text"] == trial["answer_marker"]
            for trial in trials
        ),
        "no_fatal_server_log": not summary["fatal_server_log_matches"],
        "server_log_not_rotated_or_truncated": summary["server_log_integrity"],
        "runtime_config_and_model_verified": summary["runtime_config_verified"],
        "cleanup_all_sessions_succeeded": summary["cleanup_succeeded"],
    }
    if checks["ttft_complete"]:
        retained_median = statistics.median(retained_ttft)
        cold_median = statistics.median(cold_ttft)
        checks["median_retained_ttft_40pct_lower"] = retained_median <= cold_median * 0.60
    else:
        retained_median = None
        cold_median = None
        checks["median_retained_ttft_40pct_lower"] = False
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "median_retained_ttft_ms": retained_median,
        "median_cold_ttft_ms": cold_median,
    }


def answer_marker(run_id: str, trial: int) -> str:
    digest = hashlib.sha256(f"{run_id}:{trial}".encode()).digest()
    return " ".join(ANSWER_WORDS[byte & 31] for byte in digest[:4])


def make_messages(run_id: str, trial: int) -> tuple[list[dict[str, str]], list[dict[str, str]], str]:
    long_body = " the" * 15_000
    marker = answer_marker(run_id, trial)
    original = [
        {
            "role": "system",
            "content": "Retain the document exactly. Follow the latest instruction exactly.",
        },
        {
            "role": "user",
            "content": f"Document {trial}:\nThe verification fact is {marker}.\n{long_body}",
        },
    ]
    expanded = original + [
        {"role": "assistant", "content": "Document retained."},
        {
            "role": "user",
            "content": "Reply with exactly the four-word verification fact from the retained document and no other text. Do not call tools.",
        },
    ]
    return original, expanded, marker


def build_cold_body(common: dict[str, Any], expanded: list[dict[str, str]]) -> dict[str, Any]:
    return {
        **common,
        "messages": expanded,
        "cache_mode": "bypass",
        "max_tokens": 4,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def validate_answer_vocabulary(model_path: Path) -> dict[str, dict[str, int]]:
    try:
        from tokenizers import Tokenizer
    except ImportError as error:
        raise GateError("the live gate requires the Python tokenizers package") from error
    tokenizer_path = model_path / "tokenizer.json"
    if not tokenizer_path.is_file():
        raise GateError(f"target tokenizer does not exist: {tokenizer_path}")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    evidence: dict[str, dict[str, int]] = {}
    for word in ANSWER_WORDS:
        initial = tokenizer.encode(word, add_special_tokens=False).ids
        prefixed = tokenizer.encode(f" {word}", add_special_tokens=False).ids
        if len(initial) != 1 or len(prefixed) != 1:
            raise GateError(
                f"answer word {word!r} is not one target token in both positions: "
                f"initial={initial}, prefixed={prefixed}"
            )
        evidence[word] = {
            "initial_token_id": initial[0],
            "prefixed_token_id": prefixed[0],
        }
    return evidence


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def process_command_for_pid(pid: int) -> str:
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "command="],
        check=False,
        capture_output=True,
        text=True,
    )
    command = result.stdout.strip()
    if result.returncode != 0 or not command:
        raise GateError(f"server PID {pid} is not running")
    return command


def config_path_from_command(command: str) -> Path:
    try:
        arguments = shlex.split(command)
    except ValueError as error:
        raise GateError("server process command is not valid shell syntax") from error
    for index, argument in enumerate(arguments):
        if argument == "--config" and index + 1 < len(arguments):
            return Path(arguments[index + 1]).expanduser().resolve()
        if argument.startswith("--config="):
            return Path(argument.split("=", 1)[1]).expanduser().resolve()
    raise GateError("server process command does not contain --config")


def startup_binding_record(log_text: str, server_pid: int) -> dict[str, Any]:
    records = []
    for line in log_text.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            isinstance(value, dict)
            and value.get("kind") == "c_first_kv_cache_server_start"
            and value.get("pid") == server_pid
        ):
            records.append(value)
    if len(records) != 1:
        raise GateError("startup log must contain exactly one matching server binding record")
    return records[0]


def validate_runtime_evidence(
    config: dict[str, Any],
    log_text: str,
    model: str,
    model_path: Path,
    *,
    config_path: Path,
    server_pid: int,
    process_command: str,
) -> dict[str, Any]:
    models = config.get("models")
    if not isinstance(models, list):
        raise GateError("server config has no [[models]] entries")
    expected_path = str(model_path.expanduser().resolve())
    candidates = [
        item
        for item in models
        if isinstance(item, dict)
        and (item.get("name") == model or item.get("path") in (str(model_path), expected_path))
    ]
    if len(candidates) != 1:
        raise GateError("server config does not identify exactly one requested model")
    configured = candidates[0]
    expected = {
        "kv_max_sessions": TARGET_CONFIG["max_retained_sessions"],
        "kv_max_session_tokens": TARGET_CONFIG["max_session_tokens"],
        "kv_max_suffix_prefill_tokens": TARGET_CONFIG["max_suffix_prefill_tokens"],
        "kv_retained_idle_secs": TARGET_CONFIG["idle_seconds"],
        "kv_max_retained_bytes": TARGET_CONFIG["max_retained_bytes"],
    }
    actual = {name: configured.get(name) for name in expected}
    if actual != expected:
        raise GateError(f"running model cache config mismatch: expected {expected}, got {actual}")
    configured_path = str(configured.get("path", ""))
    configured_canonical = str(Path(configured_path).expanduser().resolve())
    if configured_canonical != expected_path:
        raise GateError(
            f"configured model path {configured_canonical!r} does not equal --model-path {expected_path!r}"
        )
    config_canonical = config_path.expanduser().resolve()
    process_config = config_path_from_command(process_command)
    if process_config != config_canonical:
        raise GateError(
            f"running process config {process_config} does not equal --server-config {config_canonical}"
        )
    config_sha256 = sha256_file(config_canonical)
    binding = startup_binding_record(log_text, server_pid)
    expected_binding = {
        "pid": server_pid,
        "config_path": str(config_canonical),
        "config_sha256": config_sha256,
        "model_path": expected_path,
    }
    actual_binding = {name: binding.get(name) for name in expected_binding}
    if actual_binding != expected_binding:
        raise GateError(
            f"startup binding mismatch: expected {expected_binding}, got {actual_binding}"
        )
    if configured_path not in log_text and expected_path not in log_text:
        raise GateError("startup log does not identify the configured model path")
    if "Loading model" not in log_text:
        raise GateError("startup log has no Loading model evidence")
    return {
        "model": model,
        "model_path": configured_canonical,
        "cache_config": actual,
        "server_pid": server_pid,
        "server_config": str(config_canonical),
        "server_config_sha256": config_sha256,
        "process_command": process_command,
        "startup_binding": binding,
    }


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    endpoint = Endpoint.parse(args.base_url)
    transport = getattr(args, "_transport", None)
    json_request = request_json if transport is None else transport.request_json
    blocking_request = blocking_chat if transport is None else transport.blocking_chat
    streaming_request = stream_chat if transport is None else transport.stream_chat
    process_lookup = process_command_for_pid if transport is None else transport.process_command
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not args.server_log.exists():
        raise GateError(f"--server-log does not exist: {args.server_log}")
    if not args.server_config.exists():
        raise GateError(f"--server-config does not exist: {args.server_config}")
    run_id = str(uuid.uuid4())
    log_cursor = capture_log_cursor(args.server_log)
    startup_log = args.server_log.read_text(encoding="utf-8", errors="replace")
    startup_fatal_matches = sorted(
        set(match.group(0) for match in FATAL_LOG_PATTERN.finditer(startup_log))
    )

    def get_root(path: str) -> dict[str, Any]:
        return json_request(endpoint, "GET", endpoint.root_path(path), args.timeout)[1]

    def emit(record: dict[str, Any]) -> None:
        append_jsonl(output, {"run_id": run_id, **record})

    runtime_evidence: dict[str, Any] = {}
    answer_vocabulary_evidence: dict[str, dict[str, int]] = {}
    health_before: dict[str, Any] = {}
    health_after: dict[str, Any] = {}
    observations: list[dict[str, int]] = []
    trials: list[dict[str, Any]] = []
    cleanup_outcomes: list[dict[str, Any]] = []
    all_session_ids: list[int] = []
    initial_cache_baseline: dict[str, int] | None = None
    post_cleanup_metrics: dict[str, int] | None = None
    post_cleanup_verified = False
    fatal_matches: list[str] = []
    log_integrity = False
    run_error: str | None = None
    summary: dict[str, Any]
    session_seed = args.session_id_base + (uuid.UUID(run_id).int % 10_000_000) * 100
    try:
        process_command = process_lookup(args.server_pid)
        runtime_evidence = validate_runtime_evidence(
            tomllib.loads(args.server_config.read_text(encoding="utf-8")),
            startup_log,
            args.model,
            args.model_path,
            config_path=args.server_config,
            server_pid=args.server_pid,
            process_command=process_command,
        )
        if transport is None:
            answer_vocabulary_evidence = validate_answer_vocabulary(args.model_path)
        models = json_request(endpoint, "GET", endpoint.api_path("/models"), args.timeout)[1]
        model_ids = [item.get("id") for item in models.get("data", []) if isinstance(item, dict)]
        if args.model not in model_ids:
            raise GateError(f"running /v1/models does not contain requested model {args.model!r}")
        health_before = get_root("/health")
        initial_cache_baseline = cache_view(get_root("/metrics"))
        observations.append(initial_cache_baseline)
        require_cache_state(
            initial_cache_baseline,
            {"retained_sessions": 0, "retained_bytes": 0, "active_leases": 0},
            "run isolated baseline",
        )
        emit(
            {
                "kind": "run_start",
                "base_url": args.base_url,
                "model": args.model,
                "model_path": str(args.model_path),
                "server_config": str(args.server_config.resolve()),
                "runtime_evidence": runtime_evidence,
                "answer_vocabulary_evidence": answer_vocabulary_evidence,
                "health": health_before,
                "baseline_metrics": initial_cache_baseline,
            }
        )
        for trial_index in range(1, 4):
            session_a = session_seed + trial_index * 10
            session_b = session_a + 1
            session_c = session_a + 2
            all_session_ids.extend((session_a, session_b, session_c))
            original, expanded, marker = make_messages(run_id, trial_index)
            common = {
                "model": args.model,
                "temperature": 0.0,
                "top_p": 1.0,
                "max_prompt_tokens": TARGET_CONFIG["max_session_tokens"],
                "chat_template_kwargs": CHAT_TEMPLATE_KWARGS,
                "tools": TOOLS,
                "tool_choice": "auto",
            }

            metric_samples: list[tuple[str, dict[str, int]]] = []

            def sample(name: str) -> dict[str, int]:
                view = cache_view(get_root("/metrics"))
                observations.append(view)
                metric_samples.append((name, view))
                emit({"kind": "metric", "trial": trial_index, "step": name, "metrics": view})
                return view

            metrics_before_a = sample("before_a")
            require_cache_state(
                metrics_before_a,
                initial_cache_baseline,
                f"trial {trial_index} isolated baseline",
            )
            prefill_a_body = {
                **common,
                "messages": original,
                "session_id": session_a,
                "session_cache_policy": "best_effort",
                "max_tokens": 0,
                "stream": False,
            }
            prefill_a = blocking_request(endpoint, prefill_a_body, args.timeout)
            prefill_usage = validate_prefill(prefill_a, expect_lease_ack=False)
            emit(request_record(run_id, f"trial_{trial_index}_prefill_original_a", prefill_a_body, prefill_a))
            metrics_after_a = sample("after_a")
            if metrics_after_a["active_leases"] != metrics_before_a["active_leases"]:
                raise GateError("prefill A unexpectedly changed active leases")
            prefill_prompt_tokens = prefill_usage["prompt_tokens"]
            prepared_prefix = max(0, prefill_prompt_tokens - 1)

            prefill_b_body = {
                **common,
                "messages": [
                    {"role": "system", "content": "Compacted retained context."},
                    {"role": "user", "content": f"Document {trial_index} retained."},
                ],
                "session_id": session_b,
                "session_lease": {"session_id": session_a, "ttl_seconds": 300},
                "session_cache_policy": "best_effort",
                "max_tokens": 0,
                "stream": False,
            }
            prefill_b = blocking_request(endpoint, prefill_b_body, args.timeout)
            prefill_b_usage = validate_prefill(prefill_b, expect_lease_ack=True)
            emit(request_record(run_id, f"trial_{trial_index}_prefill_compacted_b_lease_a", prefill_b_body, prefill_b))
            metrics_after_b = sample("after_b")
            if metrics_after_b["active_leases"] != metrics_before_a["active_leases"] + 1:
                raise GateError("prefill B did not create exactly one active lease")
            lease_active = prefill_b_usage["higgs_session_lease_active"]

            retained_body = {
                **common,
                "messages": expanded,
                "session_id": session_a,
                "session_cache_policy": "require_continuation",
                "max_tokens": 4,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            retained = streaming_request(endpoint, retained_body, args.timeout)
            retained_usage = validate_usage(retained["usage"])
            if retained_usage.get("higgs_session_lease_active") is not None:
                raise GateError("retained generation unexpectedly acknowledged a new lease")
            metrics_after_retained = sample("after_retained")
            if metrics_after_retained["active_leases"] != metrics_after_b["active_leases"]:
                raise GateError("lease did not remain active through retained generation")
            emit(request_record(run_id, f"trial_{trial_index}_retained_a", retained_body, retained))

            cold_body = build_cold_body(common, expanded)
            cold = streaming_request(endpoint, cold_body, args.timeout)
            cold_usage = validate_usage(cold["usage"])
            if cold_usage.get("higgs_session_lease_active") is not None:
                raise GateError("cold comparator unexpectedly acknowledged a lease")
            if retained_usage["prompt_tokens"] != cold_usage["prompt_tokens"]:
                raise GateError("retained and cold comparators used different prompt token counts")
            metrics_after_cold = sample("after_cold")
            if metrics_after_cold["active_leases"] != metrics_after_b["active_leases"]:
                raise GateError("cold comparator unexpectedly changed active leases")
            emit(request_record(run_id, f"trial_{trial_index}_cold_stateless", cold_body, cold))

            release_status, release_body, release_ms = json_request(
                endpoint,
                "DELETE",
                endpoint.api_path(f"/cache/sessions/{session_a}"),
                args.timeout,
            )
            validate_delete_response(session_a, release_body, expected_dropped=1)
            metrics_after_release = sample("after_release")
            if metrics_after_release["active_leases"] != metrics_before_a["active_leases"]:
                raise GateError("deleting leased session A did not release its active lease")
            release = {
                "status": release_status,
                "wall_ms": release_ms,
                "response": release_body,
                "retained_sessions_delta": metric_delta(
                    metrics_after_release, metrics_after_cold, "retained_sessions"
                ),
                "retained_bytes_delta": metric_delta(
                    metrics_after_release, metrics_after_cold, "retained_bytes"
                ),
            }
            emit({"kind": "step", "step": f"trial_{trial_index}_release_a", "result": release})

            cleanup_b_status, cleanup_b_body, cleanup_b_ms = json_request(
                endpoint,
                "DELETE",
                endpoint.api_path(f"/cache/sessions/{session_b}"),
                args.timeout,
            )
            validate_delete_response(session_b, cleanup_b_body, expected_dropped=1)
            emit(
                {
                    "kind": "step",
                    "step": f"trial_{trial_index}_cleanup_b",
                    "result": {
                        "status": cleanup_b_status,
                        "wall_ms": cleanup_b_ms,
                        "response": cleanup_b_body,
                    },
                }
            )
            cleanup_c_status, cleanup_c_body, cleanup_c_ms = json_request(
                endpoint,
                "DELETE",
                endpoint.api_path(f"/cache/sessions/{session_c}"),
                args.timeout,
            )
            validate_delete_response(session_c, cleanup_c_body, expected_dropped=0)
            emit(
                {
                    "kind": "step",
                    "step": f"trial_{trial_index}_cleanup_c",
                    "result": {
                        "status": cleanup_c_status,
                        "wall_ms": cleanup_c_ms,
                        "response": cleanup_c_body,
                    },
                }
            )
            metrics_after_trial_cleanup = sample("after_trial_cleanup")
            require_cache_state(
                metrics_after_trial_cleanup,
                metrics_before_a,
                f"trial {trial_index} post-cleanup",
            )
            trials.append(
                {
                    "trial": trial_index,
                    "session_ids": {"original_a": session_a, "compacted_b": session_b, "cold_c": session_c},
                    "answer_marker": marker,
                    "prefill_prompt_tokens": prefill_prompt_tokens,
                    "prepared_retained_prefix_tokens": prepared_prefix,
                    "lease_active": lease_active,
                    "retained_cached_tokens": cached_tokens(retained_usage),
                    "cold_cached_tokens": cached_tokens(cold_usage),
                    "retained_prompt_tokens": retained_usage["prompt_tokens"],
                    "cold_prompt_tokens": cold_usage["prompt_tokens"],
                    "cold_request_stateless": "session_id" not in cold_body,
                    "prefills_valid": True,
                    "prefill_a_counter_delta": metric_delta(
                        metrics_after_a, metrics_before_a, "prefill_only_requests"
                    ),
                    "prefill_b_counter_delta": metric_delta(
                        metrics_after_b, metrics_after_a, "prefill_only_requests"
                    ),
                    "metric_steps": [name for name, _ in metric_samples],
                    "retained_bootstrap_delta": retained_bootstrap_delta(
                        metrics_after_retained, metrics_after_b
                    ),
                    "required_miss_delta": metric_delta(
                        metrics_after_release, metrics_after_a, "required_continuation_misses"
                    ),
                    "lease_failure_delta": metric_delta(
                        metrics_after_release, metrics_after_a, "expired_leases"
                    )
                    + metric_delta(metrics_after_release, metrics_after_a, "broken_leases"),
                    "retained": retained,
                    "cold": cold,
                    "release": release,
                    "release_reduced_state": release_body.get("dropped", 0) > 0
                    and (
                        release["retained_sessions_delta"] < 0
                        or release["retained_bytes_delta"] < 0
                    ),
                }
            )
        health_after = get_root("/health")
    except (GateError, OSError, TimeoutError, ValueError) as error:
        run_error = str(error)
    finally:
        for session_id in all_session_ids:
            try:
                status, response, elapsed_ms = json_request(
                    endpoint, "DELETE", endpoint.api_path(f"/cache/sessions/{session_id}"), args.timeout
                )
                validate_delete_response(session_id, response)
                outcome = {
                    "session_id": session_id,
                    "ok": True,
                    "status": status,
                    "wall_ms": elapsed_ms,
                    "response": response,
                }
            except (GateError, OSError, TimeoutError) as cleanup_error:
                outcome = {"session_id": session_id, "ok": False, "error": str(cleanup_error)}
            cleanup_outcomes.append(outcome)
        try:
            post_cleanup_metrics = cache_view(get_root("/metrics"))
            observations.append(post_cleanup_metrics)
            if initial_cache_baseline is None:
                raise GateError("run failed before an isolated cache baseline was captured")
            require_cache_state(post_cleanup_metrics, initial_cache_baseline, "terminal cleanup")
            post_cleanup_verified = True
        except (GateError, OSError, TimeoutError) as post_cleanup_error:
            if run_error is None:
                run_error = str(post_cleanup_error)
        try:
            new_fatal_matches, _ = scan_server_log(log_cursor)
            fatal_matches = sorted(set(startup_fatal_matches + new_fatal_matches))
            log_integrity = True
        except (GateError, OSError) as log_error:
            if run_error is None:
                run_error = str(log_error)
        cleanup_succeeded = len(cleanup_outcomes) == len(all_session_ids) and all(
            outcome["ok"] for outcome in cleanup_outcomes
        ) and post_cleanup_verified
        summary = {
            "kind": "summary",
            "run_id": run_id,
            "model": args.model,
            "model_path": str(args.model_path),
            "runtime_evidence": runtime_evidence,
            "answer_vocabulary_evidence": answer_vocabulary_evidence,
            "runtime_config_verified": bool(runtime_evidence),
            "health_before": health_before,
            "health_after": health_after,
            "run_error": run_error,
            "max_observed_retained_sessions": max(
                (item["retained_sessions"] for item in observations), default=0
            ),
            "max_observed_retained_bytes": max(
                (item["retained_bytes"] for item in observations), default=0
            ),
            "max_observed_active_leases": max(
                (item["active_leases"] for item in observations), default=0
            ),
            "fatal_server_log_matches": fatal_matches,
            "server_log_integrity": log_integrity,
            "cleanup_succeeded": cleanup_succeeded,
            "cleanup_outcomes": cleanup_outcomes,
            "initial_cache_baseline": initial_cache_baseline,
            "post_cleanup_metrics": post_cleanup_metrics,
            "trials": trials,
        }
        if run_error is None:
            summary["gate"] = evaluate(summary)
        else:
            summary["gate"] = {"pass": False, "checks": {"run_completed": False}}
        emit(
            {
                "kind": "terminal",
                "success": run_error is None and summary["gate"]["pass"],
                "run_error": run_error,
                "fatal_server_log_matches": fatal_matches,
                "server_log_integrity": log_integrity,
                "cleanup_outcomes": cleanup_outcomes,
                "post_cleanup_metrics": post_cleanup_metrics,
            }
        )
        emit(summary)
    summary_path = output.with_suffix(".summary.json")
    write_summary_atomic(summary_path, summary)
    return summary


def exercise_failure_cleanup() -> None:
    zero_metrics = {name: 0 for name in REQUIRED_METRICS}

    class FailureTransport:
        def __init__(self) -> None:
            self.deleted: list[int] = []

        def process_command(self, _pid: int) -> str:
            return "higgs serve --config PLACEHOLDER"

        def request_json(
            self,
            _endpoint: Endpoint,
            method: str,
            path: str,
            _timeout: float,
            _body: dict[str, Any] | None = None,
        ) -> tuple[int, dict[str, Any], float]:
            if method == "GET" and path == "/v1/models":
                return 200, {"data": [{"id": DEFAULT_MODEL_PATH.name}]}, 1.0
            if method == "GET" and path == "/health":
                return 200, {"status": "ok"}, 1.0
            if method == "GET" and path == "/metrics":
                return 200, {"cache": zero_metrics}, 1.0
            if method == "DELETE":
                session_id = int(path.rsplit("/", 1)[-1])
                self.deleted.append(session_id)
                return 200, {"session_id": session_id, "dropped": 0}, 1.0
            raise AssertionError(f"unexpected offline request: {method} {path}")

        def blocking_chat(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
            raise GateError("intentional offline mid-trial failure")

        def stream_chat(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
            raise AssertionError("streaming must not run after the injected prefill failure")

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        config_path = root / "higgs.toml"
        config_path.write_text(SELF_TEST_CONFIG_TEXT, encoding="utf-8")
        transport = FailureTransport()
        transport.process_command = lambda _pid: f"higgs serve --config {config_path}"
        server_pid = 123
        log_path = root / "server.log"
        binding = {
            "kind": "c_first_kv_cache_server_start",
            "pid": server_pid,
            "config_path": str(config_path.resolve()),
            "config_sha256": sha256_file(config_path),
            "model_path": str(DEFAULT_MODEL_PATH.resolve()),
        }
        log_path.write_text(
            json.dumps(binding) + f"\nLoading model model={DEFAULT_MODEL_PATH.resolve()}\n",
            encoding="utf-8",
        )
        output = root / "gate.jsonl"
        summary = run_gate(
            argparse.Namespace(
                base_url="http://offline.invalid/v1",
                model_path=DEFAULT_MODEL_PATH,
                model=DEFAULT_MODEL_PATH.name,
                server_log=log_path,
                server_config=config_path,
                server_pid=server_pid,
                output=output,
                session_id_base=4_600_000_000,
                timeout=5.0,
                _transport=transport,
            )
        )
        assert summary["gate"]["pass"] is False and summary["run_error"]
        assert len(summary["cleanup_outcomes"]) == 3
        assert all(outcome["ok"] for outcome in summary["cleanup_outcomes"])
        assert transport.deleted == [outcome["session_id"] for outcome in summary["cleanup_outcomes"]]
        assert summary["post_cleanup_metrics"] == zero_metrics
        records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        assert records and all(record["run_id"] == summary["run_id"] for record in records)
        terminal = next(record for record in records if record.get("kind") == "terminal")
        assert terminal["success"] is False
        assert len(terminal["cleanup_outcomes"]) == 3
        assert terminal["post_cleanup_metrics"] == zero_metrics


def self_test() -> None:
    class ShortWriter:
        def __init__(self) -> None:
            self.data = bytearray()

        def write(self, payload: bytes) -> int:
            written = min(3, len(payload))
            self.data.extend(payload[:written])
            return written

    short_writer = ShortWriter()
    write_all_bytes(short_writer, b"short-write-proof")
    assert bytes(short_writer.data) == b"short-write-proof"

    class ZeroProgressWriter:
        def write(self, _payload: bytes) -> int:
            return 0

    try:
        write_all_bytes(ZeroProgressWriter(), b"must-fail")
    except GateError:
        pass
    else:
        raise AssertionError("zero-progress evidence write must fail closed")

    exercise_failure_cleanup()
    zero_state = {"retained_sessions": 0, "retained_bytes": 0, "active_leases": 0}
    require_cache_state(zero_state, zero_state, "isolated baseline")
    try:
        require_cache_state(
            {"retained_sessions": 1, "retained_bytes": 0, "active_leases": 0},
            zero_state,
            "post cleanup",
        )
    except GateError:
        pass
    else:
        raise AssertionError("non-baseline cache state passed")
    validate_delete_response(41, {"session_id": 41, "dropped": 1}, expected_dropped=1)
    for invalid_delete in (
        {"session_id": 42, "dropped": 1},
        {"session_id": 41, "dropped": 0},
    ):
        try:
            validate_delete_response(41, invalid_delete, expected_dropped=1)
        except GateError:
            pass
        else:
            raise AssertionError(f"invalid DELETE response passed: {invalid_delete}")
    original, expanded, marker = make_messages("offline-run-nonce", 1)
    assert len(original[1]["content"]) > 55_000
    assert [message["role"] for message in original] == ["system", "user"]
    assert [message["role"] for message in expanded] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert expanded[2] == {"role": "assistant", "content": "Document retained."}
    assert expanded[:2] == original
    assert CHAT_TEMPLATE_KWARGS == {"enable_thinking": False}
    assert marker == "lake fire earth red"
    assert marker in original[1]["content"]
    assert marker not in expanded[2]["content"]
    assert marker not in expanded[-1]["content"]
    cold_probe = build_cold_body({"model": "m"}, expanded)
    assert "session_id" not in cold_probe and cold_probe["cache_mode"] == "bypass"
    assert cold_probe["max_tokens"] == 4
    try:
        cache_view({"cache": {}})
    except GateError:
        pass
    else:
        raise AssertionError("missing metrics must fail closed")
    complete_metrics = {name: 0 for name in REQUIRED_METRICS}
    assert cache_view({"cache": complete_metrics}) == complete_metrics
    after_b = dict(complete_metrics)
    after_b["session_bootstrap_exact"] = 7
    after_retained = dict(after_b)
    after_retained["session_bootstrap_exact"] = 8
    assert retained_bootstrap_delta(after_retained, after_b) == 1
    for invalid in (-1, "0", True):
        broken_metrics = dict(complete_metrics)
        broken_metrics["active_leases"] = invalid
        try:
            cache_view({"cache": broken_metrics})
        except GateError:
            pass
        else:
            raise AssertionError(f"invalid metric value passed: {invalid!r}")
    valid_usage = {
        "prompt_tokens": 15_100,
        "completion_tokens": 1,
        "total_tokens": 15_101,
        "prompt_tokens_details": {"cached_tokens": 15_099},
        "higgs_session_lease_active": 1,
    }
    cold_usage = {
        "prompt_tokens": 15_100,
        "completion_tokens": 1,
        "total_tokens": 15_101,
    }
    assert cached_tokens(validate_usage(cold_usage)) == 0
    impossible_cached = dict(valid_usage)
    impossible_cached["prompt_tokens_details"] = {"cached_tokens": 15_101}
    try:
        validate_usage(impossible_cached)
    except GateError:
        pass
    else:
        raise AssertionError("cached tokens above prompt tokens must fail closed")
    valid_stream = parse_sse_data(
        [
            json.dumps({"choices": [{"delta": {"content": marker}, "finish_reason": None}]}),
            json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
            json.dumps({"choices": [], "usage": valid_usage}),
            "[DONE]",
        ]
    )
    assert valid_stream["text"] == marker and valid_stream["tool_calls"] == []
    invalid_streams = [
        [],
        [json.dumps({"error": {"message": "boom"}}), "[DONE]"],
        [json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]}), "[DONE]"],
        [json.dumps({"choices": [], "usage": valid_usage})],
    ]
    for events in invalid_streams:
        try:
            parse_sse_data(events)
        except GateError:
            pass
        else:
            raise AssertionError(f"invalid SSE passed: {events}")
    prefill = {
        "status": 200,
        "response": {
            "choices": [
                {"message": {"role": "assistant", "content": ""}, "finish_reason": "length"}
            ],
            "usage": {
                **cold_usage,
                "completion_tokens": 0,
                "total_tokens": cold_usage["prompt_tokens"],
            },
        }
    }
    assert validate_prefill(prefill, expect_lease_ack=False)["completion_tokens"] == 0
    leased_prefill = json.loads(json.dumps(prefill))
    leased_prefill["response"]["usage"]["higgs_session_lease_active"] = 1
    assert validate_prefill(leased_prefill, expect_lease_ack=True)["higgs_session_lease_active"] == 1
    try:
        validate_prefill(prefill, expect_lease_ack=True)
    except GateError:
        pass
    else:
        raise AssertionError("B missing its lease acknowledgement must fail closed")
    missing_usage = json.loads(json.dumps(prefill))
    del missing_usage["response"]["usage"]
    try:
        validate_prefill(missing_usage, expect_lease_ack=False)
    except GateError:
        pass
    else:
        raise AssertionError("missing prefill usage must fail closed")
    trials = []
    for index in range(3):
        trials.append(
            {
                "prefill_prompt_tokens": 15_100,
                "prepared_retained_prefix_tokens": 15_099,
                "retained_cached_tokens": 15_099,
                "cold_cached_tokens": 0,
                "cold_request_stateless": True,
                "prefills_valid": True,
                "prefill_a_counter_delta": 1,
                "prefill_b_counter_delta": 1,
                "metric_steps": [
                    "before_a",
                    "after_a",
                    "after_b",
                    "after_retained",
                    "after_cold",
                    "after_release",
                    "after_trial_cleanup",
                ],
                "retained_prompt_tokens": 15_100,
                "cold_prompt_tokens": 15_100,
                "lease_active": 1,
                "retained_bootstrap_delta": 0,
                "required_miss_delta": 0,
                "lease_failure_delta": 0,
                "release_reduced_state": True,
                "answer_marker": f"ok-{index}",
                "retained": {
                    "ttft_ms": 400.0 + index,
                    "text": f"ok-{index}",
                    "tool_calls": [],
                },
                "cold": {
                    "ttft_ms": 1_000.0 + index,
                    "text": f"ok-{index}",
                    "tool_calls": [],
                },
            }
        )
    valid_summary = {
        "trials": trials,
        "health_before": {"status": "ok"},
        "health_after": {"status": "ok"},
        "max_observed_retained_sessions": 2,
        "max_observed_retained_bytes": 2_000_000_000,
        "max_observed_active_leases": 1,
        "fatal_server_log_matches": [],
        "server_log_integrity": True,
        "runtime_config_verified": True,
        "cleanup_succeeded": True,
    }
    valid_result = evaluate(valid_summary)
    assert valid_result["checks"]["matching_comparator_prompt_counts"] is True
    assert valid_result["pass"] is True
    surrounding_whitespace = json.loads(json.dumps(valid_summary))
    surrounding_whitespace["trials"][0]["retained"]["text"] = " ok-0"
    assert evaluate(surrounding_whitespace)["checks"]["exact_fact_recovery"] is False
    exact_sixty = json.loads(json.dumps(valid_summary))
    for trial in exact_sixty["trials"]:
        trial["retained"]["ttft_ms"] = 600.0
        trial["cold"]["ttft_ms"] = 1_000.0
    assert evaluate(exact_sixty)["checks"]["median_retained_ttft_40pct_lower"] is True
    over_sixty = json.loads(json.dumps(exact_sixty))
    for trial in over_sixty["trials"]:
        trial["retained"]["ttft_ms"] = 600.1
    assert evaluate(over_sixty)["checks"]["median_retained_ttft_40pct_lower"] is False
    valid_summary["trials"][0]["retained_cached_tokens"] = 0
    failed = evaluate(valid_summary)
    assert failed["pass"] is False
    assert failed["checks"]["cached_prefix_exact"] is False
    valid_summary["trials"][0]["retained_cached_tokens"] = 15_099
    valid_summary["runtime_config_verified"] = False
    assert evaluate(valid_summary)["pass"] is False
    valid_summary["runtime_config_verified"] = True
    valid_summary["cleanup_succeeded"] = False
    assert evaluate(valid_summary)["pass"] is False
    valid_summary["cleanup_succeeded"] = True
    valid_summary["trials"][0]["release_reduced_state"] = False
    assert evaluate(valid_summary)["pass"] is False
    with tempfile.TemporaryDirectory() as config_temporary:
        config_path = Path(config_temporary) / "higgs.toml"
        config_path.write_text(SELF_TEST_CONFIG_TEXT, encoding="utf-8")
        binding = json.dumps(
            {
                "kind": "c_first_kv_cache_server_start",
                "pid": 123,
                "config_path": str(config_path.resolve()),
                "config_sha256": sha256_file(config_path),
                "model_path": str(DEFAULT_MODEL_PATH.resolve()),
            }
        )
        runtime_log = f"{binding}\nLoading model model={DEFAULT_MODEL_PATH.resolve()}"
        validate_runtime_evidence(
            tomllib.loads(SELF_TEST_CONFIG_TEXT),
            runtime_log,
            "Qwen3.6-35B-A3B-4bit",
            DEFAULT_MODEL_PATH,
            config_path=config_path,
            server_pid=123,
            process_command=f"higgs serve --config {config_path}",
        )
        wrong_config = tomllib.loads(SELF_TEST_CONFIG_TEXT)
        wrong_config["models"][0]["kv_max_sessions"] = 3
        try:
            validate_runtime_evidence(
                wrong_config,
                runtime_log,
                "Qwen3.6-35B-A3B-4bit",
                DEFAULT_MODEL_PATH,
                config_path=config_path,
                server_pid=123,
                process_command=f"higgs serve --config {config_path}",
            )
        except GateError:
            pass
        else:
            raise AssertionError("wrong running config must fail closed")
    with tempfile.TemporaryDirectory() as temporary:
        evidence = Path(temporary) / "evidence.jsonl"
        append_jsonl(evidence, {"run_id": "offline", "kind": "one"})
        append_jsonl(evidence, {"run_id": "offline", "kind": "two"})
        assert len(evidence.read_text(encoding="utf-8").splitlines()) == 2
        summary_path = Path(temporary) / "summary.json"
        write_summary_atomic(summary_path, {"run_id": "offline", "pass": True})
        assert json.loads(summary_path.read_text(encoding="utf-8"))["pass"] is True
        log_path = Path(temporary) / "server.log"
        log_path.write_text("startup\n", encoding="utf-8")
        cursor = capture_log_cursor(log_path)
        replacement = Path(temporary) / "replacement.log"
        replacement.write_text("rotated\n", encoding="utf-8")
        os.replace(replacement, log_path)
        try:
            scan_server_log(cursor)
        except GateError:
            pass
        else:
            raise AssertionError("fatal-log rotation must fail closed")
        truncation_path = Path(temporary) / "truncated.log"
        truncation_path.write_text("startup evidence\n", encoding="utf-8")
        truncation_cursor = capture_log_cursor(truncation_path)
        truncation_path.write_text("x", encoding="utf-8")
        try:
            scan_server_log(truncation_cursor)
        except GateError:
            pass
        else:
            raise AssertionError("fatal-log truncation must fail closed")
        fatal_path = Path(temporary) / "fatal.log"
        fatal_path.write_text("startup\n", encoding="utf-8")
        fatal_cursor = capture_log_cursor(fatal_path)
        with fatal_path.open("a", encoding="utf-8") as stream:
            stream.write("fatal SIGABRT\n")
        fatal_matches, _ = scan_server_log(fatal_cursor)
        assert fatal_matches == ["SIGABRT"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8091/v1")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH.name)
    parser.add_argument("--server-log", type=Path, help="tmux-served Higgs log to scan")
    parser.add_argument("--server-config", type=Path, help="exact config used to launch Higgs")
    parser.add_argument("--server-pid", type=int, help="PID of the Higgs process using --server-config")
    parser.add_argument("--output", type=Path, default=Path("artifacts/c-first-kv-cache-live-gate.jsonl"))
    parser.add_argument("--session-id-base", type=int, default=4_600_000_000)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test and (
        args.server_log is None or args.server_config is None or args.server_pid is None
    ):
        parser.error("--server-log, --server-config, and --server-pid are required for the gate")
    return args


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        print("self-test: PASS")
        return 0
    try:
        summary = run_gate(args)
    except (GateError, OSError, TimeoutError) as error:
        print(f"live gate ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(summary["gate"], indent=2, sort_keys=True))
    return 0 if summary["gate"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
