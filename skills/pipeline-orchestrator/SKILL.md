---
name: pipeline-orchestrator
description: Convert a goal into a pipeline plan (array of steps).
always: false
---

# Pipeline Orchestrator

You convert a high-level goal into a pipeline of simple steps.

## Protocol

Given a goal, output a JSON array of steps. Each step has:
- `prompt`: the instruction for that step (will be sent to a small model)
- `expected`: optional expected answer (for verification)

## Output Format

```json
[
  {"prompt": "What is the capital of France?", "expected": "Paris"},
  {"prompt": "How many letters in Paris?", "expected": "5"},
  {"prompt": "Is 5 a prime number?", "expected": "yes"}
]
```

## Rules

- Each step must be simple enough for a 3B parameter model.
- Steps should be independently answerable (no "based on the above").
- Include `expected` when the answer is deterministic.
- Keep prompts under 200 characters.
- Prefer 5-20 steps. Break complex goals into atomic operations.
