# Restore Canonical Tool-Result Handles Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the canonical store-then-render tool-result protocol on the refactoring branch so ordinary tool output is durable and represented in prompts by stable handles, including medium results.

**Architecture:** Keep the current refactoring branch’s session/replay changes. Add the canonical exposure decision at the existing tool-result injection boundary: persist exact bytes first, render ordinary results as deterministic handles, and reserve bounded inline excerpts for explicit retrieval continuations. Preserve immutable storage and abort on any durability failure.

**Tech Stack:** Rust, Tokio, SQLite via `rusqlite`, `serde_json`, existing nanobot agent-loop tests.

## Global Constraints

- Do not overwrite or discard the refactoring branch’s unrelated dirty changes.
- Do not wholesale merge `fix/stable-tool-prefix`; transplant only the focused wire contract.
- The provider-facing ordinary tool-result message must never contain the raw body.
- A handle may only be emitted after SQLite proves the exact bytes are present.
- Retrieval output must remain bounded and cache-stable.

## Task 1: Establish the regression test

- [x] Change the medium-result integration test to assert a handle in both the persisted message and the next provider request.
- [x] Assert the full medium body is present in SQLite under the original tool-call id.
- [x] Run the focused test and confirm it fails against the current inline-band behavior.

## Task 2: Port the canonical ordinary-result exposure

- [x] Add the ordinary-vs-explicit exposure decision at the current injection boundary.
- [x] Store every ordinary result before rendering its handle; preserve immutable conflict/error handling.
- [x] Keep explicit retrieval tools on bounded excerpts and ensure recalled bodies are stashed for slice/search.
- [x] Route delegated and inline result paths through the same renderer.
- [x] Upgrade legacy raw ordinary messages at history replay without rewriting protocol receipts.

## Task 3: Verify the protocol

- [x] Add coverage for ordinary result size classes and reload-stable handles.
- [x] Run focused tool-engine and agent-loop tests.
- [x] Run formatting, diff checks, and the relevant library test suite.
- [x] Review the final diff for accidental changes to the refactoring work.
