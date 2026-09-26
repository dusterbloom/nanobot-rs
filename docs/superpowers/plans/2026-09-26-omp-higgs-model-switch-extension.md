# OMP Higgs Model-Switch Extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and install an OMP extension that discovers Higgs models and switches them safely with `/higgs <model>`.

**Architecture:** Keep a canonical, dependency-free TypeScript extension in Higgs under `integrations/omp/`. Its async factory discovers the Higgs catalog, registers an OMP provider, and exposes a serialized command that unloads resident models before loading and selecting the target. Pure catalog and switch functions accept `fetch` and registry callbacks so Bun tests can exercise every state transition with no real model load.

**Tech Stack:** TypeScript, Bun test runner, OMP extension API, Higgs HTTP API.

## Global Constraints

- Do not change Higgs server behavior.
- Do not add dependencies.
- Default to `http://127.0.0.1:9000` and Bearer token `higgs`; allow `HIGGS_URL` and `HIGGS_API_KEY` overrides.
- Never shell-interpolate model IDs or paths; use `fetch`, `encodeURIComponent`, and JSON serialization.
- OMP's active model changes only after Higgs confirms the target is loaded.
- Runtime switch operations must be serialized.

---

### Task 1: Tested OMP extension

**Files:**
- Create: `/Users/peppi/Dev/higgs/integrations/omp/higgs-model-switch.ts`
- Create: `/Users/peppi/Dev/higgs/integrations/omp/higgs-model-switch.test.ts`

**Interfaces:**
- Produces: `fetchCatalog(fetcher, baseUrl, headers): Promise<CatalogRecord[]>`
- Produces: `switchHiggsModel(options): Promise<string>` returning the confirmed runtime model ID.
- Produces: default async OMP extension factory registering provider `higgs` and command `/higgs`.

- [ ] **Step 1: Write failing catalog and switching tests**

Use `bun:test` with a queued fake `fetch`. Assert:

```typescript
expect(await fetchCatalog(fakeFetch, baseUrl, headers)).toEqual([
  { id: "small", path: "/models/small", loaded: false },
]);
expect(requests.map((request) => `${request.method} ${request.url}`)).toEqual([
  "GET http://127.0.0.1:9000/v1/models/available",
  "GET http://127.0.0.1:9000/v1/models",
  "DELETE http://127.0.0.1:9000/v1/models/large",
  "GET http://127.0.0.1:9000/v1/models",
  "POST http://127.0.0.1:9000/v1/models",
  "GET http://127.0.0.1:9000/v1/models",
]);
expect(JSON.parse(requests[4].body)).toEqual({ name: "small", path: "/models/small" });
```

Add focused tests for an already-loaded target, unknown target, unload failure,
load failure, confirmation failure, URL-encoded IDs, and two concurrent calls
whose HTTP sequences never overlap.

- [ ] **Step 2: Run tests and verify the red state**

Run: `bun test integrations/omp/higgs-model-switch.test.ts`

Expected: FAIL because `higgs-model-switch.ts` and its exports do not exist.

- [ ] **Step 3: Implement HTTP and switch primitives**

Define these exact shapes:

```typescript
export interface CatalogRecord {
  id: string;
  path: string;
  loaded: boolean;
}

export interface SwitchOptions {
  fetcher: typeof fetch;
  baseUrl: string;
  headers: Record<string, string>;
  target: CatalogRecord;
  listLoaded(): Promise<string[]>;
  pollAttempts?: number;
  pollDelayMs?: number;
}

export async function fetchCatalog(...): Promise<CatalogRecord[]>;
export async function switchHiggsModel(options: SwitchOptions): Promise<string>;
```

`switchHiggsModel` must list residents, return immediately for an exact loaded
target, unload every other resident, poll until none remain, POST `{name,path}`
to `/v1/models`, and confirm the target through `/v1/models`. Throw concise
errors containing the HTTP status and at most 300 response characters.

- [ ] **Step 4: Implement provider registration and `/higgs`**

The async factory must fetch the catalog, register provider `higgs` with
`openai-completions`, and map each record to a zero-cost text model with a
49,152-token context and 4,096 output-token conservative default. Register
`/higgs` so no argument lists IDs with loaded markers; an argument resolves an
exact ID, enters a promise-chain mutex, refreshes state, calls
`switchHiggsModel`, then resolves `ctx.modelRegistry.find("higgs", runtimeId)`
and awaits `pi.setModel(model)`. Notify success or the exact failure; never
change OMP selection on failure.

- [ ] **Step 5: Run tests and static checks**

Run:

```bash
bun test integrations/omp/higgs-model-switch.test.ts
bunx tsc --noEmit --target ES2022 --module ESNext --moduleResolution bundler --skipLibCheck integrations/omp/higgs-model-switch.ts integrations/omp/higgs-model-switch.test.ts
```

Expected: all tests pass and TypeScript exits 0.

- [ ] **Step 6: Commit the tested extension**

```bash
git add integrations/omp/higgs-model-switch.ts integrations/omp/higgs-model-switch.test.ts
git commit -m "feat(omp): add Higgs model switch command"
```

### Task 2: Install and validate the personal extension

**Files:**
- Create: `/Users/peppi/.omp/agent/extensions/higgs-model-switch.ts`
- Modify: `/Users/peppi/.omp/agent/models.yml` only if the extension/provider conflict requires removing the earlier static `higgs` entry.

**Interfaces:**
- Consumes: default extension factory from Task 1.
- Produces: an automatically discovered OMP extension and `higgs/*` model roster.

- [ ] **Step 1: Install the exact tested artifact**

Create `~/.omp/agent/extensions/` and copy the canonical extension byte-for-byte.
Compare SHA-256 digests after copying.

- [ ] **Step 2: Validate offline startup behavior**

With Higgs stopped, run:

```bash
omp models higgs --json
```

Expected: OMP exits successfully; the extension does not make startup fail.

- [ ] **Step 3: Validate live discovery without switching**

If Higgs is already running, run `omp models higgs --json` and confirm every
record from `GET /v1/models/available` has the selector `higgs/<id>`. Do not
start Higgs or load/unload a real model solely for this check.

- [ ] **Step 4: Verify repository and installed state**

Run the focused Bun test again, `git diff --check`, Higgs GitNexus
`detect-changes --scope all`, and compare installed/source SHA-256 digests.
Expected: tests pass, no whitespace errors, graph impact is limited to the new
integration files, and digests match.
