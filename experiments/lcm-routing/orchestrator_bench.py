#!/usr/bin/env python3
"""
Compare nvidia_orchestrator-8b LLM router against concept router on the full 30-case test set.

Replicates nanobot's exact router protocol:
  - System prompt: routing agent instructions
  - Tool definition: route_decision(action, target, args, confidence)
  - User content: router_pack format with available tools
  - /no_think prefix for Nemotron
  - Temperature: 0.2, max_tokens: 256

Usage:
  python orchestrator_bench.py --host 192.168.1.22:1234
"""

import json
import time
import argparse
import sys
import urllib.request
from pathlib import Path

AVAILABLE_TOOLS = [
    "read_file", "write_file", "list_files", "exec",
    "web_search", "web_browse", "send_message", "spawn",
    "cron_schedule", "cron_list", "cron_delete"
]

SYSTEM_PROMPT = (
    "You are a routing agent. Analyze the user's request and call route_decision once.\n\n"
    "Actions:\n"
    "- respond: Greetings, chitchat, simple questions the main model can answer directly\n"
    "- tool: Use a specific tool (set target=tool_name, args=tool_parameters)\n"
    "- specialist: Delegate to specialist model for complex multi-step reasoning\n"
    "- ask_user: ONLY when the request is truly ambiguous and cannot be answered\n\n"
    "If the user is just saying hello or asking a simple question, use action=respond.\n"
    "Call route_decision exactly once. No prose."
)

ROUTE_TOOL = {
    "type": "function",
    "function": {
        "name": "route_decision",
        "description": "Return one routing decision.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["respond", "tool", "specialist", "ask_user"]},
                "target": {"type": "string"},
                "args": {"type": "object"},
                "confidence": {"type": "number"}
            },
            "required": ["action", "target", "args", "confidence"]
        }
    }
}


def build_router_pack(user_message: str) -> str:
    """Replicate nanobot's role_policy::build_context_pack for Router role."""
    task_state = f"Strict preflight. User message: {user_message}"
    tools_str = ", ".join(AVAILABLE_TOOLS)
    return (
        f"Role: router\n"
        f"Task state:\n{task_state}\n\n"
        f"Allowed actions:\n"
        f"- respond (simple conversation, greetings, direct answers)\n"
        f"- tool (use a specific tool)\n"
        f"- specialist (delegate complex reasoning)\n"
        f"- ask_user (request clarification)\n\n"
        f"Available tools:\n{tools_str}\n"
    )


def call_orchestrator(host: str, model: str, user_message: str,
                      temperature: float = 0.2) -> dict:
    """Call the orchestrator model via LM Studio API."""
    router_pack = build_router_pack(user_message)
    user_content = f" /no_think\n{router_pack}"

    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ],
        "tools": [ROUTE_TOOL],
        "tool_choice": "auto",
        "temperature": temperature,
        "max_tokens": 256,
    }).encode()

    req = urllib.request.Request(
        f"http://{host}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    t0 = time.perf_counter()
    resp = urllib.request.urlopen(req, timeout=30)
    dt = (time.perf_counter() - t0) * 1000
    data = json.loads(resp.read())

    # Extract decision from tool call or content
    choice = data["choices"][0]
    message = choice["message"]
    tokens = data.get("usage", {})

    result = {
        "latency_ms": dt,
        "prompt_tokens": tokens.get("prompt_tokens", 0),
        "completion_tokens": tokens.get("completion_tokens", 0),
        "total_tokens": tokens.get("total_tokens", 0),
    }

    # Try tool call first
    tool_calls = message.get("tool_calls", [])
    if tool_calls:
        tc = tool_calls[0]
        if tc["function"]["name"] == "route_decision":
            args = json.loads(tc["function"]["arguments"])
            result["action"] = args.get("action", "respond")
            result["target"] = args.get("target", "")
            result["confidence"] = args.get("confidence", 0.0)
            result["source"] = "tool_call"
            return result

    # Fallback: parse JSON from content
    content = message.get("content", "")
    if content:
        try:
            # Try direct JSON parse
            parsed = json.loads(content)
            result["action"] = parsed.get("action", "respond")
            result["target"] = parsed.get("target", "")
            result["confidence"] = parsed.get("confidence", 0.0)
            result["source"] = "content_json"
            return result
        except json.JSONDecodeError:
            # Try to find JSON in content
            import re
            match = re.search(r'\{[^{}]+\}', content)
            if match:
                try:
                    parsed = json.loads(match.group())
                    result["action"] = parsed.get("action", "respond")
                    result["target"] = parsed.get("target", "")
                    result["confidence"] = parsed.get("confidence", 0.0)
                    result["source"] = "content_regex"
                    return result
                except json.JSONDecodeError:
                    pass

    result["action"] = "respond"
    result["target"] = ""
    result["confidence"] = 0.0
    result["source"] = "fallback"
    result["raw_content"] = content[:200]
    return result


def normalize_action(action: str) -> str:
    """Normalize action strings."""
    action = action.lower().strip()
    if action in ("respond", "tool", "specialist", "ask_user"):
        return action
    if action == "subagent":
        return "specialist"
    return action


def main():
    parser = argparse.ArgumentParser(description="Orchestrator vs Concept Router comparison")
    parser.add_argument("--host", default="192.168.1.22:1234")
    parser.add_argument("--model", default="nvidia_orchestrator-8b")
    parser.add_argument("--test", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    base_dir = Path(__file__).parent
    test_path = args.test or str(base_dir / "test_set.json")
    output_path = args.output or str(base_dir / "results" / "orchestrator_30case.json")

    with open(test_path) as f:
        test_data = json.load(f)

    # Load concept router results for comparison
    concept_path = base_dir / "results" / "L1_results.json"
    concept_results = {}
    if concept_path.exists():
        with open(concept_path) as f:
            cdata = json.load(f)
        for r in cdata["results"]:
            concept_results[r["query"]] = r

    results = []
    total_tokens = 0
    total_latency = 0

    print(f"Testing {len(test_data['test_cases'])} cases against {args.model} @ {args.host}")
    print(f"{'#':<4} {'Query':<45} {'Expected':<12} {'Got':<12} {'ms':>6} {'tok':>5} {'OK':>3}")
    print("-" * 90)

    for i, tc in enumerate(test_data["test_cases"]):
        query = tc["query"]
        expected = tc["expected_action"]

        try:
            resp = call_orchestrator(args.host, args.model, query)
        except Exception as e:
            print(f"{i+1:<4} {query[:44]:<45} {expected:<12} {'ERROR':<12} {'':>6} {'':>5} {'N':>3}")
            results.append({
                "query": query,
                "expected_action": expected,
                "expected_target": tc.get("expected_target"),
                "predicted_action": "error",
                "error": str(e),
                "correct_action": False,
            })
            continue

        predicted = normalize_action(resp["action"])
        correct = predicted == expected
        latency = resp["latency_ms"]
        tokens = resp.get("total_tokens", 0)
        total_tokens += tokens
        total_latency += latency

        mark = "Y" if correct else "N"
        q_short = query[:44]
        print(f"{i+1:<4} {q_short:<45} {expected:<12} {predicted:<12} {latency:>5.0f} {tokens:>5} {mark:>3}")

        # Concept router comparison
        cr = concept_results.get(query, {})

        results.append({
            "query": query,
            "expected_action": expected,
            "expected_target": tc.get("expected_target"),
            "predicted_action": predicted,
            "predicted_target": resp.get("target", ""),
            "confidence": resp.get("confidence", 0.0),
            "correct_action": correct,
            "latency_ms": round(latency, 2),
            "tokens": tokens,
            "source": resp.get("source", ""),
            "concept_predicted": cr.get("predicted_action", ""),
            "concept_correct": cr.get("correct_action", None),
        })

        # Small delay to not overload LM Studio
        time.sleep(0.1)

    # Summary
    correct_count = sum(1 for r in results if r["correct_action"])
    concept_correct = sum(1 for r in results if r.get("concept_correct", False))
    latencies = [r["latency_ms"] for r in results if "latency_ms" in r]

    print("\n" + "=" * 90)
    print("COMPARISON SUMMARY")
    print("=" * 90)
    print(f"\n{'Metric':<30} {'Orchestrator 8B':>18} {'Concept (centroid)':>18}")
    print("-" * 70)
    print(f"{'Accuracy':<30} {f'{correct_count}/30':>18} {f'{concept_correct}/30':>18}")
    import numpy as np
    concept_latency = 5.2  # from L1 results
    print(f"{'Mean latency':<30} {f'{np.mean(latencies):.0f}ms':>18} {f'{concept_latency:.0f}ms':>18}")
    print(f"{'Total tokens':<30} {f'{total_tokens}':>18} {'0':>18}")
    print(f"{'VRAM':<30} {'~6 GB':>18} {'0 GB':>18}")

    # Agreement analysis
    agree = sum(1 for r in results
                if r.get("concept_predicted") and r["predicted_action"] == r["concept_predicted"])
    disagree = [(r["query"][:40], r["predicted_action"], r["concept_predicted"])
                for r in results
                if r.get("concept_predicted") and r["predicted_action"] != r["concept_predicted"]]

    print(f"\n{'Agreement':<30} {f'{agree}/30':>18}")
    if disagree:
        print(f"\nDisagreements ({len(disagree)}):")
        for q, orch, concept in disagree:
            print(f"  {q:<42} orch={orch:<12} concept={concept}")

    # Per-category where orchestrator wins
    orch_only = [r for r in results if r["correct_action"] and not r.get("concept_correct", False)]
    concept_only = [r for r in results if not r["correct_action"] and r.get("concept_correct", False)]

    if orch_only:
        print(f"\nOrchestrator correct, concept wrong ({len(orch_only)}):")
        for r in orch_only:
            print(f"  {r['query'][:50]}")
    if concept_only:
        print(f"\nConcept correct, orchestrator wrong ({len(concept_only)}):")
        for r in concept_only:
            print(f"  {r['query'][:50]}")

    # Save results
    output = {
        "experiment": "L1_comparison",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "orchestrator_model": args.model,
        "summary": {
            "orchestrator_accuracy": f"{correct_count}/30",
            "concept_accuracy": f"{concept_correct}/30",
            "orchestrator_mean_latency_ms": round(float(np.mean(latencies)), 1),
            "concept_mean_latency_ms": concept_latency,
            "orchestrator_total_tokens": total_tokens,
            "agreement_rate": f"{agree}/30",
        },
        "results": results,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
