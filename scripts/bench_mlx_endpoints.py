#!/usr/bin/env python3
"""Benchmark local MLX-compatible chat endpoints.

Example:
  scripts/bench_mlx_endpoints.py \
    --prompt "The capital of France is" \
    --max-tokens 16 \
    --target "nanobot,http://127.0.0.1:8772,,/chat" \
    --target "mlx-lm,http://127.0.0.1:8774,/abs/model/path,/v1/chat/completions" \
    --target "lms,http://127.0.0.1:1234,qwen3_17b_bench,/v1/chat/completions"

Target format:
  name,base_url,model,stream_path

Notes:
  - `model` can be empty for endpoints that ignore the OpenAI `model` field.
  - `stream_path` can be `/chat` for nanobot's Ex0bit SSE endpoint.
"""

from __future__ import annotations

import argparse
import dataclasses
import http.client
import json
import statistics
import time
import urllib.parse
import urllib.request
from typing import Optional


@dataclasses.dataclass
class Target:
    name: str
    base_url: str
    model: Optional[str]
    stream_path: str


def parse_target(raw: str) -> Target:
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "target must be name,base_url,model,stream_path"
        )
    name, base_url, model, stream_path = parts
    if not name or not base_url:
        raise argparse.ArgumentTypeError("target name and base_url are required")
    return Target(
        name=name,
        base_url=base_url.rstrip("/"),
        model=model or None,
        stream_path=stream_path or "/v1/chat/completions",
    )


def build_messages(system: Optional[str], prompt: str) -> list[dict[str, str]]:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return messages


def post_json(url: str, payload: dict) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())


def build_payload(
    target: Target,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    stream: bool,
) -> dict:
    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if target.model:
        payload["model"] = target.model
    if stream and target.stream_path.endswith("/v1/chat/completions"):
        payload["stream"] = True
    return payload


def measure_nonstream(target: Target, payload: dict) -> dict:
    t0 = time.perf_counter()
    obj = post_json(f"{target.base_url}/v1/chat/completions", payload)
    elapsed = time.perf_counter() - t0
    usage = obj.get("usage", {})
    completion_tokens = usage.get("completion_tokens", 0)
    prompt_tokens = usage.get("prompt_tokens", 0)
    tok_s = completion_tokens / elapsed if elapsed > 0 else 0.0
    content = (
        obj.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )
    return {
        "elapsed_s": elapsed,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "tok_s": tok_s,
        "content_preview": content[:120],
    }


def measure_first_chunk(target: Target, payload: dict) -> dict:
    url = urllib.parse.urlparse(target.base_url)
    conn = http.client.HTTPConnection(url.hostname, url.port, timeout=300)
    t0 = time.perf_counter()
    conn.request(
        "POST",
        target.stream_path,
        body=json.dumps(payload),
        headers={"Content-Type": "application/json"},
    )
    resp = conn.getresponse()
    first_line = None
    while True:
        line = resp.fp.readline()
        if not line:
            break
        if line.startswith(b"data: "):
            first_line = line.decode(errors="ignore").strip()
            break
    elapsed = time.perf_counter() - t0
    conn.close()
    return {
        "first_chunk_s": elapsed,
        "line_preview": first_line[:120] if first_line else None,
    }


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True, help="User prompt for the benchmark")
    parser.add_argument("--system", help="Optional system message")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measure-runs", type=int, default=2)
    parser.add_argument(
        "--target",
        action="append",
        type=parse_target,
        required=True,
        help="Target in the form name,base_url,model,stream_path",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON only")
    args = parser.parse_args()

    messages = build_messages(args.system, args.prompt)
    rows = []

    for target in args.target:
        nonstream_payload = build_payload(
            target, messages, args.max_tokens, args.temperature, stream=False
        )
        for _ in range(args.warmup_runs):
            measure_nonstream(target, nonstream_payload)

        samples = [
            measure_nonstream(target, nonstream_payload)
            for _ in range(args.measure_runs)
        ]
        stream = measure_first_chunk(
            target,
            build_payload(
                target, messages, args.max_tokens, args.temperature, stream=True
            ),
        )
        rows.append(
            {
                "target": target.name,
                "base_url": target.base_url,
                "model": target.model,
                "stream_path": target.stream_path,
                "tok_s": summarize([sample["tok_s"] for sample in samples]),
                "elapsed_s": summarize([sample["elapsed_s"] for sample in samples]),
                "prompt_tokens": samples[-1]["prompt_tokens"],
                "completion_tokens": samples[-1]["completion_tokens"],
                "first_chunk_s": stream["first_chunk_s"],
                "content_preview": samples[-1]["content_preview"],
                "line_preview": stream["line_preview"],
            }
        )

    if args.json:
        print(json.dumps(rows, indent=2))
        return 0

    print("MLX endpoint benchmark")
    print(f"  prompt: {args.prompt!r}")
    print(f"  max_tokens: {args.max_tokens}")
    print(f"  measure_runs: {args.measure_runs}")
    for row in rows:
        print(f"\n[{row['target']}]")
        print(f"  url: {row['base_url']}")
        if row["model"]:
            print(f"  model: {row['model']}")
        print(
            "  tok/s: "
            f"{row['tok_s']['mean']:.2f} "
            f"(min {row['tok_s']['min']:.2f}, max {row['tok_s']['max']:.2f})"
        )
        print(
            "  elapsed_s: "
            f"{row['elapsed_s']['mean']:.3f} "
            f"(min {row['elapsed_s']['min']:.3f}, max {row['elapsed_s']['max']:.3f})"
        )
        print(
            f"  usage: prompt={row['prompt_tokens']} completion={row['completion_tokens']}"
        )
        print(f"  first_chunk_s: {row['first_chunk_s']:.3f}")
        print(f"  content: {row['content_preview']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
