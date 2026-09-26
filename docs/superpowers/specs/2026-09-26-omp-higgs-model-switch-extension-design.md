# OMP Higgs Model-Switch Extension Design

## Goal

Make every model returned by Higgs `GET /v1/models/available` selectable from
OMP without changing Higgs. The first version exposes an explicit
`/higgs <model>` command so a long model load cannot race the next inference
request or exceed OMP's 30-second event-handler budget.

## Scope

Create one personal OMP TypeScript extension at
`~/.omp/agent/extensions/higgs-model-switch.ts`. It has no dependencies and
uses the existing Higgs API at `http://127.0.0.1:9000` with the existing
Bearer token. Environment variables may override the URL and token.

Transparent switching through OMP's generic `/model`, `/switch`, or Ctrl+P
picker is deliberately deferred. Changing Higgs routing behavior is also out
of scope for this first version.

## Startup and Catalog

The async extension factory requests `GET /v1/models/available`. It registers
one OMP provider named `higgs` whose OpenAI-compatible base URL is `/v1` and
whose model list is derived from the returned records. Each registered model
retains its Higgs `id` and canonical `path`; the path remains private extension
state and is never sent in an inference request.

If Higgs is offline at OMP startup, the extension registers the already-loaded
models obtainable from `GET /v1/models` when possible. If neither endpoint is
reachable, startup continues and `/higgs` reports that Higgs is unavailable.

## Switching Flow

`/higgs` without an argument lists catalog entries and their loaded state.
`/higgs <id>` performs these steps under a single in-process switch lock:

1. Refresh the available and loaded model lists.
2. Reject unknown or unavailable-on-disk model IDs before mutating Higgs.
3. If the target is already resident, skip server mutation.
4. Otherwise unload every resident model except the exact target using
   `DELETE /v1/models/{id}` and wait until the loaded list confirms removal.
5. Load the target with `POST /v1/models` using its catalog `id` and `path`.
6. Confirm the target is resident, resolve it from OMP's model registry, and
   call `pi.setModel`.

OMP changes its active model only after Higgs confirms the target. Concurrent
commands queue behind one switch promise, preventing overlapping unload/load
cycles while preserving each command's requested target.

## Failures and Safety

- Network and non-2xx responses are shown through OMP notifications with the
  Higgs response body truncated to a useful length.
- A failed unload stops the switch before loading another multi-GB model.
- A failed load leaves OMP on its prior model and clearly reports that Higgs
  may temporarily have no resident model.
- Model IDs are URL-encoded for DELETE requests. Request bodies use JSON
  serialization; shell execution is not used.
- The extension never reads commented TOML. Higgs remains the authority for
  discoverable artifacts and canonical paths.

## Verification

Use a small mocked HTTP server to verify catalog registration and the command's
request ordering. Cover an already-loaded target, unload-then-load success,
unknown model rejection, unload failure, load failure, and two concurrent
switch requests. Finally, run OMP against the real extension and confirm that
`omp models higgs --json` lists the catalog. An end-to-end switch is run only
while Higgs is already running, because loading models is expensive and changes
resident GPU state.
