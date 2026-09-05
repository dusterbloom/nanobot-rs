# nanobot

A Rust assistant that connects your conversations, models, tools, and durable memory.

## TL;DR

Use cloud models or local inference through Higgs and other compatible servers. Nanobot manages the conversation and executes tools; the model server handles inference. SQLite history, LCM summaries, and workspace memory support long sessions.

```sh
cargo build --release
./target/release/nanobot onboard
./target/release/nanobot agent
```

Configure providers in `~/.nanobot/config.json`; use `/local` to switch to local inference. The default local autostart backend is Higgs. Workspace memory lives in `~/.nanobot/workspace/memory/MEMORY.md`.

## Features

- Terminal conversations and messaging-channel adapters, including Telegram and email.
- Cloud and local providers through one agent loop.
- File, shell, search, scheduling, and delegation tools with workspace restrictions and execution checks.
- Durable session history, replay evidence, LCM compaction, and persistent memory.
- Capacity-aware local requests, preserved interrupted turns, and durable pending work when capacity remains unavailable.
- Constrained recovery for explicit reset announcements when the required tools are available.
- Optional voice support through the `voice` build feature.

Notes/reset tools and FP16 Escha experiments remain evaluation work; they are not production defaults. See the [validation tracker](experiments/context-recovery/VALIDATION-TRACKER.md).

## Architecture

```text
channel / CLI → agent loop ⇄ provider ⇄ local or cloud model
                    ⇅
              tools + durable context
                    ↓
                   reply
```

The loop coordinates requests and tool receipts. Providers translate model protocols; SQLite, LCM, and memory preserve context across turns.

Open [the architecture companion](docs/architecture.html) locally for a source-linked map. Edit `docs/architecture.json`, then run `python3 scripts/architecture.py`. CI checks mapped source paths and generated-file freshness; reviewers maintain the relationships.

## Contribute

Read [CONTRIBUTING.md](CONTRIBUTING.md) and [AGENTS.md](AGENTS.md). Keep changes focused. Run `cargo build --release`, `cargo test --release`, and `cargo fmt --all -- --check`. For agent-loop, provider, or context changes, include matched `scripts/turn_bench.sh` results. Update the architecture map with structural changes and include reproduction steps and validation in your PR.
