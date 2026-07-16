# nanobot

```
 _____             _       _
|   | |___ ___ ___| |_ ___| |_
| | | | .'|   | . | . | . |  _|
|_|___|__,|_|_|___|___|___|_|
```

A personal AI assistant that runs on your terms. Cloud or local. Text or voice. Your machine, your models, your data.

## Why

Most AI assistants are cloud-locked SaaS products. nanobot is a single binary that talks to whatever LLM you point it at -- Claude, GPT, Gemini, Groq, or a model running on your own hardware. Add voice and it becomes a conversational assistant you can interrupt mid-sentence. Add channels and it lives in your Telegram, WhatsApp, or email.

No containers. No Python. No dependencies beyond what `cargo build` pulls in.

## Quick start

```bash
cargo build --release

# Initialize config and workspace
nanobot onboard

# Add your API key to ~/.nanobot/config.json

# Start chatting
nanobot agent
```

**Provider selection:** the first non-empty API key wins, in priority order
OpenRouter → DeepSeek → Anthropic → OpenAI → Gemini → Zhipu → ZhipuCoding → Groq → vLLM.
To force a specific provider, keep only its key in the config. `nanobot status` shows which provider is active.

## Features

### Talk to any LLM

All providers speak the same OpenAI-compatible protocol. First API key found wins:

OpenRouter / DeepSeek / Anthropic / OpenAI / Gemini / Groq / vLLM

```
You: What's the weather like?
```

### Go local with `/local`

Toggle between cloud and local inference mid-conversation. nanobot first adopts
an already-running compatible endpoint. It starts a server only when
`agents.defaults.localAutostart` explicitly selects `"higgs"` or
`"lmstudio"`; the default, `"off"`, never spawns one.

```
You: /local  # with localAutostart: "lmstudio"
  Starting LM Studio server on port 1234...
  Loading model...

  LOCAL MODE LM Studio on port 1234
  Model: NVIDIA-Nemotron-Nano-9B-v2-Q4_K_M.gguf

You: /model
Available models:
  [1] gemma-3n-E4B-it-Q4_K_S.gguf (3923 MB)
  [2] Ministral-8B-Instruct-Q4_K_M.gguf (4815 MB)
  [3] NVIDIA-Nemotron-Nano-9B-v2-Q4_K_M.gguf (5352 MB) (active)
  ...
Select model [1-12] or Enter to cancel:
```

Switch models on the fly. The server process is monitored -- if it crashes during loading, you get the error immediately instead of waiting for a timeout. Stale servers from previous sessions are cleaned up automatically.

### Voice mode

```bash
cargo build --release --features voice
```

```
You: /voice
Voice mode ON. Ctrl+Space or Enter to speak, type for text.

Recording... (press Enter or Ctrl+Space to stop)
You said: "What time is it in Tokyo?"

It's currently about two in the morning in Tokyo.
```

Voice mode uses on-device models -- no cloud STT/TTS:
- **Speech-to-text**: Whisper (via jack-voice)
- **Text-to-speech**: SuperTonic 3, or macOS `say`

Audio is streamed sentence-by-sentence through PulseAudio. First audio plays in ~300-500ms while remaining sentences synthesize in the background.

**Interrupt anytime**: press Enter during playback to cut the response short and start speaking. The assistant stops talking and listens.

### MLX inference (Apple Silicon)

> **Note:** In-process MLX inference was experimental and has been removed. The
> code path is preserved on the `claude/pensive-lumiere` branch. To use MLX
> models today, run them via LM Studio (or any OpenAI-compatible server) and
> point `agents.defaults.localApiBase` in `~/.nanobot/config.json` at it.

### Tools

The agent has hands. It can read and write files, run shell commands, search the web, spawn sub-agents, and schedule recurring tasks:

| Tool | What it does |
|------|-------------|
| File read/write/edit | Workspace file operations |
| Shell exec | Run commands with timeout and sandboxing |
| Web search + fetch | SearXNG (default) or Brave Search API + page fetching |
| Message | Send messages to channels |
| Spawn | Launch sub-agent conversations |
| Cron | Schedule recurring tasks with cron expressions |

#### Web Search Setup

By default, `web_search` uses [SearXNG](https://github.com/searxng/searxng) running locally. To set it up:

```bash
# Run SearXNG with JSON API enabled
docker run -d --name searxng -p 8888:8080 \
  -e SEARXNG_BASE_URL=http://localhost:8888 \
  searxng/searxng:latest

# Enable JSON format (required for API access)
docker exec searxng sed -i 's/^formats:$/formats:\n    - html\n    - json/' /etc/searxng/settings.yml
docker restart searxng
```

Add to `~/.nanobot/config.json`:
```json
{
  "tools": {
    "web": {
      "provider": "searxng",
      "searxngUrl": "http://localhost:8888"
    }
  }
}
```

Alternatively, set `"provider": "brave"` and add a `braveApiKey` to use Brave Search API (cloud).

### Channels

Deploy as a bot on your messaging platforms -- or start them right from the REPL:

| Channel | Transport | Quick start |
|---------|-----------|-------------|
| Telegram | Long-polling (POST) | `/telegram` or `/tg` |
| WhatsApp | WebSocket bridge | `/whatsapp` or `/wa` |
| Email | IMAP polling + SMTP | `/email` |

Channels run in the background while you keep chatting. Inbound messages and bot responses are displayed in the REPL as they flow through:

```
[telegram] 4815162342: What's the capital of France?
[telegram] bot: The capital of France is Paris.
You: (you keep chatting locally)
```

#### Voice messages on channels

With the `voice` feature enabled, voice messages sent via Telegram or WhatsApp are automatically transcribed using on-device STT (same Whisper model as `/voice` mode). The bot replies with both text and a voice note synthesized via TTS. No cloud transcription -- everything runs locally. Requires `ffmpeg` for audio codec conversion.

### Context compaction

Long conversations don't lose their source history. LCM stores every raw
message in `~/.nanobot/sessions.db` and replaces older active context with
hierarchical summaries that point back to those originals. Local compaction
uses the configured on-demand Higgs model; deterministic compression is the
hard-limit fallback when that model is unavailable.

The compaction sidecar is loaded only while LCM or reflection is using it. Its
canonical configuration lives under `lcm`. Autonomous spawning is explicit:
set `agents.defaults.localAutostart` to `"higgs"`; `"off"` and `"lmstudio"`
may reuse a healthy sidecar but never start one.

```json
{
  "agents": {
    "defaults": {
      "localAutostart": "higgs"
    }
  },
  "lcm": {
    "compactionModelDir": "/path/to/small-higgs-model",
    "compactionPort": 8092,
    "compactionContextSize": 4096
  }
}
```

For one release, `agents.defaults.higgsCompactionModelDir` and
`agents.defaults.higgsCompactionPort` are accepted as migration aliases. New
configuration should not use them. The removed `higgsCompactionModel`,
`higgsCompactionOnDemand`, and `compactionEndpoint` fields are not part of the
new schema: Higgs reports the model it actually serves, the sidecar is always
on demand, and its endpoint comes from the canonical directory/port fields.

### Concurrent message processing

In gateway mode, messages from different chats are processed in parallel (up to `maxConcurrentChats`, default 4). A WhatsApp user and a Telegram user get responses simultaneously instead of waiting in a queue. Messages within the same conversation stay serialized to preserve ordering.

### Memory and skills

- **Memory**: Curated cross-session facts in
  `~/.nanobot/workspace/memory/MEMORY.md`. Reflection reads completed session
  working state from SQLite, atomically updates this file, then marks those rows
  reflected.
- **Skills**: Markdown files with YAML frontmatter at `{workspace}/skills/{name}/SKILL.md`. Skills marked `always: true` are always loaded; others appear as summaries the agent can read on demand
- **Sessions**: Raw messages, stable IDs, LCM nodes, tool results, snapshots,
  and session-scoped working memory live in `~/.nanobot/sessions.db`. JSONL is
  an explicit import/export format, not a live session store.

## Interactive commands

| Command | Description |
|---------|-------------|
| `/local`, `/l` | Toggle local/cloud mode |
| `/model`, `/m` | Select local GGUF model |
| `/think`, `/t`, `/thinking` | Toggle/adjust thinking (`on`, `off`, or budget tokens) |
| `/nothink`, `/nt` | Suppress streamed thinking output |
| `/voice`, `/v` | Toggle voice mode |
| `/telegram`, `/tg` | Start Telegram channel in background |
| `/whatsapp`, `/wa` | Start WhatsApp channel in background |
| `/email` | Start Email channel in background |
| `/paste`, `/p` | Paste mode -- multiline input until `---` |
| `/stop` | Stop all running channels |
| `/status`, `/s` | Show current mode, model, and channels |
| `/lcm stats` | Show read-only LCM compaction statistics |
| `/help`, `/h` | Show help |
| `Ctrl+C` | Exit |

## CLI commands

| Command | Description |
|---------|-------------|
| `nanobot onboard` | Initialize config and workspace |
| `nanobot agent` | Interactive chat |
| `nanobot agent -m "..."` | Single message |
| `nanobot gateway` | Start with channel adapters |
| `nanobot status` | Configuration status |
| `nanobot tune --input bench.json` | Pick best local profile from benchmark JSON |
| `nanobot sessions import-jsonl [path]` | Import legacy session JSONL once |
| `nanobot sessions export <session> --format jsonl` | Export a SQLite session as JSONL |
| `nanobot sessions delete <session-id>` | Transactionally delete one session and all owned rows |
| `nanobot channels status` | Channel status |
| `nanobot cron list` | List scheduled jobs |
| `nanobot cron add` | Add a scheduled job |
| `nanobot voice list --engine supertonic` | List Supertonic voices |
| `nanobot voice config` | Show voice configuration help |

## Building

```bash
# Standard build
cargo build --release

# With voice mode (requires jack-voice)
cargo build --release --features voice

# Debug with logging
RUST_LOG=debug cargo run -- agent -m "Hello"
```

## Configuration

Config lives at `~/.nanobot/config.json` (camelCase keys). Workspace defaults to `~/.nanobot/workspace/`.

Key agent settings in `config.json`:

| Key | Default | Description |
|-----|---------|-------------|
| `agents.defaults.model` | `anthropic/claude-opus-4-5` | LLM model |
| `agents.defaults.maxTokens` | `8192` | Max response tokens |
| `agents.defaults.maxContextTokens` | `128000` | Context window size |
| `agents.defaults.maxConcurrentChats` | `4` | Parallel chat limit (gateway) |
| `agents.defaults.localAutostart` | `off` | Spawn policy when discovery finds no local endpoint: `off`, `higgs`, or `lmstudio` |
| `agents.defaults.higgsPort` | `8091` | Main managed Higgs endpoint |
| `agents.defaults.lmsPort` | `1234` | LM Studio endpoint |

Local discovery prefers an explicitly configured endpoint, then Higgs, then LM
Studio. `localBackend` is derived from the endpoint that answers; it is not a
second spawn switch. For LM Studio autostart, install
[LM Studio](https://lmstudio.ai/) and its CLI (`lms`).

### Voice settings (`voice` block in `config.json`)

```json
{
  "voice": {
    "ttsEngine": "supertonic",
    "ttsVoice": "F5",
    "language": "it"
  }
}
```

| Key | Values | Effect |
|-----|--------|--------|
| `ttsEngine` | `supertonic` (default), `say` | Which TTS backend to load. SuperTonic 3 supports 31 languages at 44.1 kHz; `say` uses the native macOS voice. |
| `ttsVoice` | engine-specific ID, or `null` | If `null` and engine is `supertonic`, the curated per-language voice is auto-selected (see below). |
| `language` | ISO-639-1 code (`"it"`, `"en"`, `"es"`, `"de"`, …) or `null` | Used to pick the curated Supertonic voice when `ttsVoice` is `null`. |

**SuperTonic 3 voice IDs (ear-checked May 2026):**

| Voice | Notes |
|-------|-------|
| `M2` | Default male, recommended for general use |
| `F5` | Recommended female (Italian + general) |
| `F1`, `F3` | Also good for Italian |
| `M1`, `M3`, `M4`, `M5` | Untested per-language; usable |
| `F2`, `F4` | ⚠️ Drift off pronunciation on Italian — not auto-selected |

When `ttsVoice: null` and `ttsEngine: "supertonic"`, the curated picker resolves to:
- `language: "it"` → `M2` (male) / `F5` (female via `ttsVoice: "F5"`)
- `language: "en"` → `M2` (male) / `F1` (female via `ttsVoice: "F1"`)
- Any other language → `M2` global default

## Architecture

```
              Channels (Telegram / WhatsApp / Email)
                              |
                              v
User --> CLI / Voice / Realtime --> AgentLoop --> LLM Provider
                           |   ^        (any OpenAI-compat API)
                           |   |
                           v   |
                        ToolRegistry --> file, shell, web,
                                         message, spawn, cron
```

Single-binary. No microservices. The agent loop is the core -- it takes a message, builds context (identity + memory + skills + history), calls the LLM, executes any tool calls, and returns a response. Voice mode wraps this with STT on input and streaming TTS on output.

On startup, the TUI clears the terminal, shows an ASCII splash with mode info, and renders LLM responses as styled markdown (headers, code blocks, bold/italic) via termimad. Input uses rustyline with arrow-key history.

## Origin

Originally inspired by [nanobot](https://github.com/HKUDS/nanobot) by [HKUDS](https://github.com/HKUDS) (Python, MIT). Rebuilt from scratch in Rust with a different architecture, feature set, and direction.

## License

MIT
