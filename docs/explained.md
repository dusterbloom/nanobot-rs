# Nanobot — Explained (No Tech Jargon)

## What is Nanobot?

Nanobot is a personal AI assistant that lives on your computer. It helps you get things done by reading files, running commands, searching the web, sending messages to chat apps, and even creating helper agents for complex tasks. 

Think of it as a smart co-pilot that learns from how you work and remembers what you've done together.

---

## How Does It Work?

```
┌─────────┐     ┌──────────┐     ┌─────────┐     ┌─────────────┐
│ You     │  →  │ Nanobot  │  →  │ Takes   │  →  │ Shows result│
│ Speak   │     │ Thinks   │     │ Action  │     │ & learns    │
└─────────┘     └──────────┘     └─────────┘     └─────────────┘
```

1. **You ask** — Type or speak what you need help with
2. **It thinks** — Nanobot figures out the best way to handle your request
3. **It acts** — Runs commands, reads files, searches online as needed
4. **It remembers** — Each interaction teaches it more about how you work

---

## What Can It Do?

### 🗂️ File Management
- Read and write files on your computer
- Search for specific content across directories
- Organize and manage your project structure

### 💻 Command Runner
- Execute shell commands safely (with timeout limits)
- Run scripts, compile code, deploy services
- Everything logged in an audit trail for safety

### 🌐 Web Research
- Search the internet via DuckDuckGo or SerpAPI
- Fetch and clean content from any URL
- Summarize articles and documentation

### 💬 Communication
- Send messages to Telegram, WhatsApp, Feishu (Lark)
- Read and reply to emails via IMAP/SMTP
- Proactively notify you about important events

### 🤖 Multi-Agent Teams
- Spawn helper agents for sub-tasks
- Each agent has its own context and tools
- Perfect for long or complex projects

---

## What Makes Nanobot Special?

| Feature | Why It Matters |
|---------|----------------|
| **Runs Locally** | Your data never leaves your machine. No cloud dependencies required. |
| **Privacy-First** | Everything stays on your computer. Great for sensitive work. |
| **Works Offline** | Uses small models that don't need internet access. |
| **Remembers You** | Builds a personal knowledge base (MEMORY.md) of what you've done together. |
| **Learns Patterns** | Adapts to how *you* think and work over time. |
| **Extensible** | Add new tools, skills, or agents as your needs grow. |

---

## Typical Use Cases

### For Developers
```bash
# "Help me understand this codebase"
→ Reads files, explains structure, finds patterns

# "Run tests and fix failures"
→ Executes commands, analyzes errors, suggests fixes

# "Deploy my app to production"
→ Runs deployment scripts, monitors output, reports status
```

### For Writers & Researchers
```bash
# "Summarize these 10 articles on AI ethics"
→ Fetches URLs, extracts content, synthesizes key points

# "Research the best practices for X"
→ Searches web, evaluates sources, provides citations
```

### For Project Managers
```bash
# "Send status update to my team chat"
→ Compiles progress info, sends message to Telegram/WhatsApp

# "Check if all services are healthy"
→ Runs health checks, reports issues in plain language
```

---

## Behind the Scenes (Optional)

If you're curious about how everything fits together under the hood:

- **Agent Core** — The brain that decides what actions to take
- **Tool System** — A collection of capabilities it can use (file read, shell exec, web search, etc.)
- **Context & Memory** — Keeps track of conversations and learns from experience
- **Channels** — Connects to chat apps so you can talk to it anywhere
- **Providers** — Supports multiple AI models (local or cloud)

For the full technical architecture with code structure and data flows, see [architecture.html](/home/peppi/Dev/nanobot/architecture.html).

---

## Getting Started

1. Install: `cargo build --release`
2. Configure: Add your API keys in `~/.nanobot/config.json`
3. Run: `./target/release/nanobot agent`
4. Try asking: *"Help me understand this project"*

Full documentation available at [`/home/peppi/Dev/nanobot/README.md`](/home/peppi/Dev/nanobot/README.md).

---

## FAQ (Frequently Asked Questions)

### Does it store my conversations?
Yes, but only locally in `~/.nanobot/sessions/`. You control what's shared.

### Can I use it without an internet connection?
Absolutely. It works fully offline with local models like Qwen3.5.

### Is it safe to run commands?
All tool calls are logged and auditable. Commands have timeout limits to prevent runaway processes.

### How does it learn from me?
Your MEMORY.md file stores important facts about you, your preferences, and past work. The agent reads this every time we talk.

---

## Want to Contribute?

Nanobot is built by builders who want better tools for themselves. If you're interested in helping:

- Add new tool capabilities
- Improve the Italian localization
- Test voice/audio modes
- Write documentation or tutorials

Check out [`AGENTS.md`](/home/peppi/Dev/nanobot/AGENTS.md) for how to get started as a contributor.
