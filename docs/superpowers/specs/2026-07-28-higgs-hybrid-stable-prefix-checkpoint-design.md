# Higgs Hybrid Stable-Prefix Checkpoint Design

Date: 2026-07-28

## Objective

Reduce first-turn time to first token for Nanobot sessions served by
`higgs-nightly` without changing the system prompt or adding a second inference
path.

The optimization targets the recurring, exact prefix before the first real user
message: chat template bytes, system instructions, tool definitions, and any
other server-rendered stable content. Higgs evaluates that prefix once, persists
the complete Qwen3.6 hybrid cache, and restores it for a later new session or
process restart. It then prefills only the user-specific suffix.

This accelerates the first turn of subsequent sessions after one successful seed.
The first request for a new prefix or model remains a cold prefill and creates
the checkpoint for later requests.

## Proven Current Behavior

The active process is
`/Users/peppi/Dev/higgs-nightly/target/release/higgs`, serving the configured
Qwen3.6-35B-A3B model on port 9000. Disk prefix caching is enabled, but the
configured `.higgs-prefix-cache.bin` contains only its 32-byte header.

The cache is not being populated because
`crates/higgs-engine/src/cache/disk_prefix_cache.rs::snapshot_layers` accepts
only `AnyCache::KV` and returns
`Unsupported("hybrid caches are memory-only")` for the active
`AnyCache::Hybrid` model. A Qwen3.6 hybrid cache contains both:

- `LayerCache::KV(SteppingKeyValueCache)` for attention layers; and
- `LayerCache::Arrays(ArraysCache)` for GDN layers, including `conv_state`,
  `ssm_state`, `conv_pos`, and `offset`.

Nanobot's session bootstrap already calls `prepare_generation(..., true)` when
no retained session can be forked, so it already consults the prefix cache.
However, the session path deliberately does not publish through the existing
hybrid in-memory radix store because that path previously caused unsafe Metal
materialization. A session-specific durable capture is therefore required; a
configuration toggle alone cannot create a usable checkpoint.

## Scope

This design changes the active Qwen3.6 hybrid path only.

It preserves:

- the existing dense `HIGGSKV` v1 file and dense-model behavior;
- the retained `session_id` continuation path after the first turn;
- the canonical chat rendering, system prompt, tools, and request payload;
- the single `SimpleEngine` generation path and global MLX execution gate; and
- exact cold prefill as the fallback for every cache failure.

It does not:

- promise acceleration for the very first request after installation;
- persist dFlash/dSpark drafter state;
- persist TurboQuant-compressed hybrid state in v1;
- add a second server, proxy, prompt mode, or protocol flag; or
- change Nanobot production code.

## Invariants

### The checkpoint boundary is exact and stable

For a cacheable request, Higgs renders the prompt exactly as it does today and
finds the token boundary immediately before the earliest real user message.
Tool-response pseudo-user messages do not count as the first real user query.
The result must be an exact prefix of the final prompt token array.

The persisted boundary is:

```text
floor(first_real_user_start_token / block_size) * block_size
```

Higgs skips capture when the result is below the existing
`min_tokens_to_persist` threshold. Flooring reuses the existing cache block
quantum and makes lookup and stored cache offsets agree by construction. It does
not remove or change prompt tokens; the floored tail is simply prefilled on
every request.

The boundary helper uses the existing renderer and tokenizer logic rather than
searching for textual delimiters. Any system prompt, tool schema, chat template,
tokenizer, or thinking-mode change that changes the stable prefix tokens
naturally misses.

### A restored cache represents every layer at one absolute position

Every layer record carries one of three explicit tags: empty, KV, or arrays.

- KV records preserve keys and values, tensor dtype, shape, and absolute offset.
- Arrays records preserve optional `conv_state` and `ssm_state` tensors,
  including dtype and shape, plus `conv_pos` and `offset`.
- Empty records preserve the layer slot.

Restore rejects mixed or incomplete boundaries. After reconstruction,
`AnyCache::validate_absolute_boundary(stored_token_count)` must succeed before
the cache can reach model execution.

The writer never silently downcasts hybrid tensors to f16. V1 supports the
floating dtypes actually used by the served hybrid model and rejects any
unsupported dtype. This keeps disk restore equivalent to an in-memory deep
clone instead of creating a new lossy inference mode.

### Disk state is optional

A missing, stale, truncated, corrupt, incompatible, or unsupported checkpoint is
a cache miss. Higgs logs a content-free reason, increments a failure metric, and
runs the existing exact cold path. Cache I/O must never fail a completion.

### MLX work remains serialized

Tensor evaluation, deep cloning, and restored-array materialization occur only
while the model lock and the process-wide MLX execution gate are held. File
indexing, checksum validation, and file I/O occur outside that critical section.
No background task may evaluate an MLX graph concurrently with generation.

## Chosen Architecture

Keep the dense disk format unchanged and add a sibling, versioned hybrid
checkpoint store owned by `DiskPrefixCache`.

If the configured dense path is:

```text
.../.higgs-prefix-cache.bin
```

the hybrid store is:

```text
.../.higgs-prefix-cache.hybrid-v1/
```

Each content-addressed entry is named from the model identity, token count, and
SHA-256 token digest. The directory contains no alternate runtime or cache
authority; `DiskPrefixCache` remains the single component that chooses the
longest valid memory, dense-disk, or hybrid-disk prefix.

The separate format is intentional:

- existing dense files remain byte-compatible;
- hybrid layer records are structurally different from paged dense KV blocks;
- corruption or migration in one store cannot invalidate the other; and
- rollback consists of ignoring/removing the sibling directory, not migrating
  the production dense file.

## Capture Flow

On a cold session bootstrap with no valid hybrid checkpoint:

1. Render and tokenize the canonical prompt once.
2. Derive and validate the stable, block-aligned prefix boundary.
3. Prefill the stable prefix on the ordinary fresh hybrid cache.
4. Deep-clone and evaluate the cache at that boundary while the model lock and
   MLX gate are held.
5. Continue prefill over the remaining prompt tokens and decode through the
   existing path.
6. After decode, while the existing model lock and MLX gate are still held,
   convert the evaluated checkpoint tensors into an owned CPU payload.
7. Hand only that CPU payload and its exact prefix digest to one bounded
   persistence worker.
8. The worker checksums the payload, writes a temporary file in the same
   directory, `fsync`s it, atomically renames it to the content-addressed name,
   and `fsync`s the directory.

The phase-boundary deep clone is the only new seed-turn work before first token.
It is necessary because attention KV buffers append in place; serializing after
the suffix prefill cannot reconstruct the earlier GDN/SSM state. CPU
extraction happens after decode, and disk serialization/publication happens on
the worker; neither is on the TTFT path.

The persistence queue has capacity one. If a matching write is already present,
the new candidate is deduplicated. If a different write is already pending, the
new candidate is dropped and the request still succeeds. There is no unbounded
task creation and no generation thread waiting for disk I/O. The worker never
owns an MLX `Array` and therefore never needs the model lock or MLX gate. Engine
shutdown closes and joins the single worker; an interrupted temporary file is
ignored on the next startup.

## Restore Flow

Before acquiring the model lock, `prepare_generation` asks `DiskPrefixCache` for
the longest metadata-only candidate whose token hash matches the request.
Hybrid candidate selection checks:

- format version;
- model identity;
- cache topology;
- checkpoint token count and token hash;
- file length and payload checksum; and
- requested minimum prefix length.

Only the winning candidate payload is read. Under the model lock and MLX gate,
Higgs reconstructs every layer, evaluates the arrays, and validates the absolute
boundary. On success, `PreparedGeneration` receives:

```text
cache = restored hybrid checkpoint
actual_prompt_tokens = prompt_tokens[stored_token_count..]
reused_prefix_tokens = stored_token_count
```

The existing prefill and decode code then operates on the suffix. The existing
retained-session publication remains authoritative for turn two and later.

A full-prompt match still follows the current rule: because a checkpoint does
not persist generation logits, Higgs falls back to a fresh prefill rather than
sampling from nonexistent cached logits.

## File Format and Identity

The hybrid file has:

- a fixed magic value and format version;
- model-identity digest;
- token count and SHA-256 digest of the little-endian token bytes;
- layer count;
- payload byte length; and
- payload checksum.

Each tensor record has a dtype tag, rank, dimensions, and byte length before its
raw evaluated data. All integer fields use an explicitly documented byte order.
Readers use checked arithmetic and reject trailing, missing, or oversized data
before allocating tensors.

The model identity includes:

- canonical model directory;
- architecture and quantization configuration;
- hashes of tokenizer, chat-template, and small model configuration artifacts;
- selected weight-file names, sizes, and modification identities; and
- the hybrid checkpoint schema version.

This makes ordinary model replacement, re-quantization, tokenizer changes, and
template changes miss without hashing all model weights on every startup. Token
hash matching independently invalidates system-prompt and tool-schema changes.

The engine records its cache topology when it constructs `DiskPrefixCache`.
Hybrid engines consider memory-hybrid and hybrid-disk candidates; dense engines
consider memory-dense and existing `HIGGSKV` candidates. A stale dense entry can
therefore never be reconstructed and passed to a hybrid model, even if its token
prefix happens to match.

Only atomically published, checksummed entries are indexed. At startup, Higgs
ignores temporary files, removes no user data, and lazily validates payloads
when they become lookup candidates.

The store keeps at most the existing prefix-entry limit (currently eight).
After a successful publication, it evicts least-recently-used entries beyond
that bound. The active entry is never removed before its replacement is
atomically visible.

## API Shape

The existing `bool` that controls prefix-cache publication must not gain another
meaning. Replace the downstream behavior selector with a narrow enum/plan:

```text
PrefixCapture::Disabled
PrefixCapture::ConversationBoundary
PrefixCapture::StableFirstUserBoundary { token_count }
```

The stateless path uses `ConversationBoundary`, preserving current in-memory and
dense-disk behavior. A cold session bootstrap uses
`StableFirstUserBoundary`. The enum owns boundary validation and makes it
impossible for a call site to accidentally enable the previously unsafe hybrid
radix behavior.

`DiskPrefixCache` exposes separate prepare/commit operations:

- prepare a metadata candidate without the model lock;
- materialize the selected candidate under the MLX gate;
- prepare an immutable hybrid snapshot under the MLX gate; and
- enqueue/publish the prepared CPU payload without the gate.

Format parsing and serialization live in one dedicated
`hybrid_disk_checkpoint.rs` module. Cache selection and fallback remain in
`disk_prefix_cache.rs`; generation boundary choice and two-phase prefill remain
beside the hot path in `simple.rs`.

## Observability

Add content-free counters to `CacheStats` and `/metrics`:

- hybrid disk lookups;
- hybrid disk hits;
- hybrid disk reused tokens;
- hybrid checkpoint publications;
- publication drops/deduplications;
- load failures; and
- write failures.

Add count plus cumulative milliseconds for candidate read, materialization, and
seed deep-clone time. This follows the existing atomic-counter metrics path
instead of introducing a second metrics mechanism. Logs include model identity
prefix, token count, layer count, byte count, and failure category, never prompt
text or raw token IDs.

The existing `prompt_tokens_details.cached_tokens` remains the client-visible
proof of reuse and must include a restored hybrid prefix.

## TDD and Verification

Implementation proceeds red to green.

### Serialization contract

Write failing tests first for:

1. Round-trip of a mixed hybrid cache containing empty, KV, and arrays layers.
2. Preservation of KV/arrays dtype, shapes, offsets, `conv_pos`, and `ssm_state`.
3. Rejection of bad magic, version, checksum, model identity, token hash,
   truncated data, trailing data, overflowed dimensions, and unsupported dtype.
4. Atomic publication and startup ignoring incomplete temporary files.
5. Bounded LRU eviction and content-addressed deduplication.
6. Dense `HIGGSKV` v1 header and round-trip tests remaining byte-for-byte
   unchanged.

### Boundary and hot-path contract

Write failing tests first for:

1. Stable boundary immediately before the earliest real user message.
2. Tool-response pseudo-user messages not selecting the boundary.
3. Exact token-prefix validation and block flooring.
4. Short stable prefixes skipping capture.
5. Session cold bootstrap selecting stable capture while the stateless path
   preserves conversation-boundary capture.
6. A hybrid disk hit prefilling only the suffix and reporting the restored token
   count as cached.
7. Any prepare, read, checksum, materialization, or validation failure reaching
   the exact cold path once, without a second routing branch.
8. A bounded/full writer queue never blocking or failing generation.

### Model-level equivalence

For the real configured Qwen3.6 model:

1. Run a cold greedy request and persist the stable checkpoint.
2. Restart Higgs so no in-memory or retained session state survives.
3. Run the same system/tools prefix with the same user input under a new
   `session_id`.
4. Assert a hybrid-disk hit, nonzero cached tokens, and a cache boundary equal
   to the stable prefix.
5. Compare cold and restored next-token logits or deterministic greedy output;
   they must match within the existing exact-cache tolerance.
6. Repeat with a changed system prompt, tool schema, tokenizer identity, and
   model identity; every case must miss and remain correct.
7. Corrupt a copied checkpoint and prove the request succeeds through cold
   prefill.

### Performance gates

Run `scripts/turn_bench.sh` and a restart benchmark with matched prompt,
generation length, thermal state, and server configuration.

The feature is releasable only when:

- the restored first-turn path is materially faster than matched cold prefill;
- cached-token reporting equals the restored stable prefix;
- seed-turn deep-clone overhead is no more than 250 ms or 10% of cold TTFT,
  whichever is larger;
- decode throughput does not regress by more than 3%;
- warm retained-session TTFT does not regress by more than 5%; and
- dense-cache tests, full `cargo test`, and `cargo build` pass.

If the seed deep clone exceeds the gate, the implementation is not hidden behind
a permanent mode flag. The copy mechanism must be improved or the feature must
remain unreleased.

## Expected Change Surface

Higgs implementation is expected in:

- `crates/higgs-engine/src/cache/hybrid_disk_checkpoint.rs` — versioned hybrid
  serialization, atomic publication, indexing, and LRU;
- `crates/higgs-engine/src/cache/mod.rs` — narrow module/export wiring;
- `crates/higgs-engine/src/cache/disk_prefix_cache.rs` — unified candidate
  selection, hybrid prepare/commit, and cold fallback;
- `crates/higgs-engine/src/simple.rs` — stable boundary, capture enum, two-phase
  seed prefill, restoration metrics, and bounded publication scheduling;
- `crates/higgs/src/routes/metrics.rs` — content-free counters; and
- focused tests beside those implementations.

No change is expected in Nanobot's production path. Nanobot continues sending
the same canonical request and `session_id`; Higgs supplies the transport-level
acceleration.

Before editing any Higgs symbol, run GitNexus upstream impact analysis and warn
on HIGH or CRITICAL risk. Before any commit, run GitNexus change detection
against `main`, then the full correctness and speed tracks required by
`AGENTS.md`.

## Acceptance

The design is complete when a Qwen3.6 stable prefix is seeded once, survives a
Higgs restart, accelerates a later session's first turn, reports the exact reused
token count, and cannot alter or fail a response when its durable state is
missing or invalid.
