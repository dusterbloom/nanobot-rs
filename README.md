# nanobot

```
 _____             _       _
|   | |___ ___ ___| |_ ___| |_
| | | | .'|   | . | . | . |  _|
|_|___|__,|_|_|___|___|___|_|
```

A personal AI assistant that runs on your terms. Cloud or local. Text or voice. Your machine, your models, your data.

Rust port of [nanobot](https://github.com/HKUDS/nanobot) by HKUDS -- rebuilt from scratch for speed, portability, and offline-first operation.

## Why

Most AI assistants are cloud-locked SaaS products. nanobot is a single binary that talks to whatever LLM you point it at -- Claude, GPT, Gemini, Groq, or a GGUF running on your own hardware. Add voice and it becomes a conversational assistant you can interrupt mid-sentence. Add channels and it lives in your Telegram, WhatsApp, or Feishu.

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

## Features

### Talk to any LLM

All providers speak the same OpenAI-compatible protocol. First API key found wins:

OpenRouter / DeepSeek / Anthropic / OpenAI / Gemini / Groq / vLLM

```
You: What's the weather like?
```

### Go local with `/local`

Toggle between cloud and local inference mid-conversation. nanobot connects to LM Studio, loads a model, and switches over.

```
You: /local
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
- **Text-to-speech**: Pocket TTS (Candle, 24kHz, CPU real-time)

Audio is streamed sentence-by-sentence through PulseAudio. First audio plays in ~300-500ms while remaining sentences synthesize in the background.

**Interrupt anytime**: press Enter during playback to cut the response short and start speaking. The assistant stops talking and listens.

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
| Feishu (Lark) | WebSocket | gateway mode |

Channels run in the background while you keep chatting. Inbound messages and bot responses are displayed in the REPL as they flow through:

```
[telegram] 4815162342: What's the capital of France?
[telegram] bot: The capital of France is Paris.
You: (you keep chatting locally)
```

#### Voice messages on channels

With the `voice` feature enabled, voice messages sent via Telegram or WhatsApp are automatically transcribed using on-device STT (same Whisper model as `/voice` mode). The bot replies with both text and a voice note synthesized via TTS. No cloud transcription -- everything runs locally. Requires `ffmpeg` for audio codec conversion.

### Context compaction

Long conversations don't lose context. When history exceeds the token budget, nanobot summarizes older messages via a cheap LLM call instead of silently dropping them. The summary preserves key facts, decisions, and pending actions. Falls back to hard truncation if summarization fails.

### Concurrent message processing

In gateway mode, messages from different chats are processed in parallel (up to `maxConcurrentChats`, default 4). A WhatsApp user and a Telegram user get responses simultaneously instead of waiting in a queue. Messages within the same conversation stay serialized to preserve ordering.

### Memory and skills

- **Memory**: Daily notes + long-term MEMORY.md, loaded into every prompt
- **Skills**: Markdown files with YAML frontmatter at `{workspace}/skills/{name}/SKILL.md`. Skills marked `always: true` are always loaded; others appear as summaries the agent can read on demand
- **Sessions**: JSONL persistence at `~/.nanobot/sessions/`

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
| `nanobot channels status` | Channel status |
| `nanobot cron list` | List scheduled jobs |
| `nanobot cron add` | Add a scheduled job |

## Building

```bash
# Standard build
cargo build --release

# With voice mode (requires jack-voice + pocket-tts)
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

For local mode, install [LM Studio](https://lmstudio.ai/) and its CLI (`lms`). Models are managed through LM Studio.

## Architecture

```
              Channels (Telegram / WhatsApp / Feishu)
                              |
                              v
User --> CLI / Voice --> AgentLoop --> LLM Provider
                           |   ^        (any OpenAI-compat API)
                           |   |
                           v   |
                        ToolRegistry --> file, shell, web,
                                         message, spawn, cron
```

Single-binary. No microservices. The agent loop is the core -- it takes a message, builds context (identity + memory + skills + history), calls the LLM, executes any tool calls, and returns a response. Voice mode wraps this with STT on input and streaming TTS on output.

On startup, the TUI clears the terminal, shows an ASCII splash with mode info, and renders LLM responses as styled markdown (headers, code blocks, bold/italic) via termimad. Input uses rustyline with arrow-key history.

## Attribution

Rust port of [nanobot](https://github.com/HKUDS/nanobot) by [HKUDS](https://github.com/HKUDS). Original Python implementation licensed under MIT.

## License

MIT
