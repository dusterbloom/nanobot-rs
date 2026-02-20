#!/usr/bin/env python3
"""Audit local-model failures from nanobot JSONL sessions for a specific date."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class FailurePattern:
    category: str
    scope: str
    first_action: str
    regex: re.Pattern[str]


@dataclass
class FailureEvent:
    session_file: Path
    line_no: int
    timestamp: str
    role: str
    category: str
    scope: str
    first_action: str
    excerpt: str


PATTERNS: list[FailurePattern] = [
    FailurePattern(
        category="context_overflow",
        scope="local",
        first_action="Lower prompt/context size or increase runtime n_ctx.",
        regex=re.compile(
            r"(exceed_context_size_error|exceeds the available context size|n_ctx)",
            re.IGNORECASE,
        ),
    ),
    FailurePattern(
        category="tool_call_protocol_mismatch",
        scope="local",
        first_action="Ensure tool_call_id round-trip matches provider format.",
        regex=re.compile(
            r"(tool_call_id.+not found in 'tool_calls'|tool call id.+not found)",
            re.IGNORECASE,
        ),
    ),
    FailurePattern(
        category="oom_or_memory_pressure",
        scope="local",
        first_action="Reduce model/context/concurrency and keep memory headroom.",
        regex=re.compile(
            r"((out of memory|cuda.*memory|metal.*memory|\boom\b).*(error|fail)?)",
            re.IGNORECASE,
        ),
    ),
    FailurePattern(
        category="timeout",
        scope="local",
        first_action="Lower work per turn; add watchdog timeout and retries.",
        regex=re.compile(
            r"(timed out|deadline exceeded|request timeout|timeout error|timed-out)",
            re.IGNORECASE,
        ),
    ),
    FailurePattern(
        category="connection_failure",
        scope="local",
        first_action="Verify local server is up; confirm host, port, and health checks.",
        regex=re.compile(
            r"(connection refused|failed to connect|error sending request for url|connection reset|unreachable|host not found)",
            re.IGNORECASE,
        ),
    ),
    FailurePattern(
        category="rate_limited",
        scope="cloud",
        first_action="Reduce request rate or switch model/provider.",
        regex=re.compile(
            r"(http 429|rate limit|too many requests)",
            re.IGNORECASE,
        ),
    ),
    FailurePattern(
        category="auth_failure",
        scope="cloud",
        first_action="Fix credentials and provider configuration.",
        regex=re.compile(
            r"(http 401|unauthorized|no cookie auth credentials)",
            re.IGNORECASE,
        ),
    ),
    FailurePattern(
        category="billing_or_quota",
        scope="cloud",
        first_action="Increase credits/quota or lower token budget.",
        regex=re.compile(
            r"(http 402|payment required|requires more credits|quota)",
            re.IGNORECASE,
        ),
    ),
    FailurePattern(
        category="model_not_found",
        scope="mixed",
        first_action="Validate model ID and provider-specific naming.",
        regex=re.compile(
            r"(http 404|not_found_error|model: .+not found)",
            re.IGNORECASE,
        ),
    ),
    FailurePattern(
        category="invalid_request",
        scope="mixed",
        first_action="Inspect payload shape, tool schema, and provider compatibility.",
        regex=re.compile(
            r"(http 400|invalid_request_error|bad request)",
            re.IGNORECASE,
        ),
    ),
]

GENERIC_FAILURE_RE = re.compile(
    r"^\s*(error calling llm|i encountered an error|panic[: ]|traceback\b|exception[: ])",
    re.IGNORECASE,
)
FAILURE_CANDIDATE_RE = re.compile(
    r"(error calling llm|encountered an error|llm api returned status|http [45]\d\d|timed out|deadline exceeded|request timeout|connection refused|failed to connect|error sending request for url|out of memory|panic|traceback|invalid_request_error|exceed_context_size_error)",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit failures from nanobot session JSONL files.",
    )
    parser.add_argument(
        "--sessions-dir",
        default="~/.nanobot/sessions",
        help="Session directory (default: ~/.nanobot/sessions).",
    )
    parser.add_argument(
        "--date",
        default=dt.date.today().isoformat(),
        help="Date to scan in YYYY-MM-DD (default: local today).",
    )
    parser.add_argument(
        "--scope",
        choices=["local", "all"],
        default="local",
        help="Failure scope filter (default: local).",
    )
    parser.add_argument(
        "--roles",
        default="assistant",
        help="Comma-separated roles to inspect (default: assistant).",
    )
    parser.add_argument(
        "--top-sessions",
        type=int,
        default=10,
        help="Number of top session files to show.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=2,
        help="Sample events per category to print.",
    )
    return parser.parse_args()


def normalize_roles(raw_roles: str) -> set[str]:
    roles = {part.strip() for part in raw_roles.split(",") if part.strip()}
    return roles or {"assistant"}


def match_pattern(text: str) -> tuple[str, str, str] | None:
    if not FAILURE_CANDIDATE_RE.search(text):
        return None

    for pattern in PATTERNS:
        if pattern.regex.search(text):
            return pattern.category, pattern.scope, pattern.first_action
    if GENERIC_FAILURE_RE.search(text):
        return "uncategorized_failure", "mixed", "Inspect raw event and classify manually."
    return None


def should_include_scope(scope_filter: str, event_scope: str) -> bool:
    if scope_filter == "all":
        return True
    return event_scope in {"local", "mixed"}


def iter_session_files(root: Path) -> Iterable[Path]:
    return sorted(path for path in root.glob("*.jsonl") if path.is_file())


def clip_excerpt(text: str, max_len: int = 180) -> str:
    single_line = " ".join(text.split())
    if len(single_line) <= max_len:
        return single_line
    return single_line[: max_len - 3] + "..."


def extract_timestamp(entry: dict) -> str | None:
    ts = entry.get("timestamp")
    if isinstance(ts, str) and len(ts) >= 10:
        return ts
    return None


def main() -> int:
    args = parse_args()
    roles = normalize_roles(args.roles)
    sessions_dir = Path(args.sessions_dir).expanduser()
    target_date = args.date

    if not sessions_dir.exists():
        print(f"Error: sessions directory not found: {sessions_dir}", file=sys.stderr)
        return 2

    files = list(iter_session_files(sessions_dir))
    parse_errors = 0
    total_lines = 0
    date_matched_lines = 0
    events: list[FailureEvent] = []

    for session_file in files:
        try:
            raw_lines = session_file.read_text(errors="replace").splitlines()
        except OSError as exc:
            print(f"Warning: could not read {session_file}: {exc}", file=sys.stderr)
            continue

        for idx, raw_line in enumerate(raw_lines, start=1):
            total_lines += 1
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            try:
                entry = json.loads(raw_line)
            except json.JSONDecodeError:
                parse_errors += 1
                continue

            if not isinstance(entry, dict):
                continue

            timestamp = extract_timestamp(entry)
            if not timestamp or not timestamp.startswith(target_date):
                continue

            date_matched_lines += 1
            role = entry.get("role")
            if role not in roles:
                continue

            content = entry.get("content")
            if not isinstance(content, str):
                continue

            match = match_pattern(content)
            if match is None:
                continue

            category, scope, first_action = match
            if not should_include_scope(args.scope, scope):
                continue

            events.append(
                FailureEvent(
                    session_file=session_file,
                    line_no=idx,
                    timestamp=timestamp,
                    role=role,
                    category=category,
                    scope=scope,
                    first_action=first_action,
                    excerpt=clip_excerpt(content),
                )
            )

    category_counts = Counter(event.category for event in events)
    session_counts = Counter(event.session_file.name for event in events)

    print("Local Model Failure Audit")
    print(f"date: {target_date}")
    print(f"sessions_dir: {sessions_dir}")
    print(f"scope: {args.scope}")
    print(f"roles: {','.join(sorted(roles))}")
    print(f"files_scanned: {len(files)}")
    print(f"lines_scanned: {total_lines}")
    print(f"lines_on_date: {date_matched_lines}")
    print(f"json_parse_errors: {parse_errors}")
    print(f"failure_events: {len(events)}")

    if not events:
        print("No matching failure events found.")
        return 0

    print("\nCategory counts:")
    for category, count in category_counts.most_common():
        first_action = next(event.first_action for event in events if event.category == category)
        print(f"- {category}: {count} | first_action: {first_action}")

    print("\nTop sessions:")
    for session_name, count in session_counts.most_common(max(1, args.top_sessions)):
        print(f"- {session_name}: {count}")

    print("\nSamples:")
    for category, _ in category_counts.most_common():
        printed = 0
        for event in events:
            if event.category != category:
                continue
            print(
                f"- [{category}] {event.timestamp} "
                f"{event.session_file.name}:{event.line_no}"
            )
            print(f"  {event.excerpt}")
            printed += 1
            if printed >= max(1, args.samples):
                break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
