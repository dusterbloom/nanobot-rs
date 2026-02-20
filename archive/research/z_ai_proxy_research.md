# Z.ai / Zhipu AI Web Chat Proxy Projects - Research Summary

## Overview
Multiple GitHub projects exist that proxy Z.ai's Zhipu AI (GLM models) as OpenAI-compatible and Anthropic-compatible API endpoints. These allow users with Z.ai subscriptions to expose their access as APIs compatible with standard AI tools and frameworks.

---

## Active Projects Found

### 1. **GLM Proxy** (Most Comprehensive)
**Repository:** https://github.com/dejay2/glmproxy  
**Language:** Node.js / TypeScript  
**Status:** Active and well-documented

#### What It Does
- HTTP proxy server that transforms Anthropic Messages API requests to Z.ai GLM-4.7 API format
- Enables Claude-compatible tools and applications to use GLM models
- Provides both Anthropic-compatible and OpenAI-compatible endpoints
- Includes smart backend routing (automatically selects text vs vision models)

#### Key Features
- **Web Dashboard**: Settings panel and MCP (Model Context Protocol) management
- **Intelligent Model Selection**: Auto-detects images/video in messages → routes to glm-4.6v for vision, switches back to glm-4.7 for text
- **Video Analysis**: Full support for video files up to ~1 hour (128K context)
- **Reasoning Injection**: Automatic reasoning prompt injection with `<reasoning_content>` tag parsing
- **Web Search**: MCP server integration for web_search and web_reader tools
- **Tool Execution**: Internal tool loop with MCP servers
- **Streaming**: Full SSE streaming support for both backend paths
- **Claude Code Integration**: Drop-in compatibility with Claude Code CLI

#### How to Use
```bash
# Install globally for ccglm command
npm install -g .

# Start proxy and launch Claude Code in one command
ccglm

# Or start proxy manually
npm start
```

#### Configuration
- **Port**: 4567 (default)
- **Host**: 127.0.0.1 (localhost only)
- **Required**: `ZAI_API_KEY` environment variable
- **Optional**: `REF_API_KEY`, `CONTEXT7_API_KEY` for MCP servers

#### Security Notes
- **Localhost-only by default** (127.0.0.1:4567)
- Not intended for production or multi-user environments
- API keys stored in memory (server) and localStorage (dashboard)
- No authentication layer (assumes trusted localhost environment)

---

### 2. **OpenAI-Compatible-API-Proxy-for-Z**
**Repository:** https://github.com/Idkwhattona/OpenAI-Compatible-API-Proxy-for-Z  
**Language:** Unknown (appears to be compiled binary)  
**Status:** Active

#### What It Does
- Simple OpenAI-compatible API proxy for Z.ai's GLM-4.5 model
- Desktop application with GUI for easier use
- Provides standard OpenAI API format access to GLM models

#### Key Features
- User-friendly interface designed for all skill levels
- Fast responses
- Seamless integration with OpenAI API calls
- Cross-platform: Windows 10+, macOS 10.15+, Linux

#### How to Use
1. Download from releases page
2. Install for your OS (exe for Windows, dmg for macOS, archive for Linux)
3. Open application and input API requests
4. Configure API keys in settings menu

---

### 3. **Copilot-Proxy** (GitHub Copilot Integration)
**Repository:** https://github.com/modpotato/copilot-proxy  
**Language:** Python  
**Status:** Active

#### What It Does
- Bridges GitHub Copilot Chat with GLM coding models
- Mimics Ollama API interface to work with Copilot
- Intercepts requests from GitHub Copilot's Ollama provider and forwards to GLM backend
- Advertises GLM Coding Plan lineup for seamless model switching

#### Key Features
- **Ollama API Compatibility**: Implements Ollama API specification
- **Model Switching**: Copilot can switch between GLM models seamlessly
- **Configuration Management**: CLI-based config for API keys, context length, temperature, model selection
- **PyPI Package**: Installable via `uv pip install copilot-proxy`

#### Supported Models
- GLM-4.7 (next-gen flagship)
- GLM-4-Plus (high-throughput)
- GLM-4.6 (flagship coding)
- GLM-4.5 (balanced)
- GLM-4.5-Air (lightweight)
- GLM-4.5-AirX (accelerated)
- GLM-4.5-Flash (ultra-fast)
- GLM-4.6V (multimodal)
- GLM-4.6V-Flash (fast multimodal)
- And more...

#### How to Use
```bash
# Install
uv pip install copilot-proxy

# Quick start
uvx copilot-proxy serve --host 127.0.0.1 --port 11434

# Interactive setup (first use)
copilot-proxy
```

#### Configuration
- **Port**: 11434 (default, same as Ollama)
- **Host**: 127.0.0.1 (localhost)
- **Required**: Z.ai Coding Plan API key
- **Environment Variables**: `ZAI_API_KEY`, `ZAI_API_BASE_URL` (optional)
- **Config File**: Persistent configuration via CLI commands

---

### 4. **Z2api-Go**
**Repository:** https://github.com/Tylerx404/z2api-go  
**Language:** Go  
**Status:** Active

#### What It Does
- Lightweight proxy API for Z.ai compatible with OpenAI and Anthropic
- Written in Go for performance and minimal dependencies
- Supports both OpenAI and Anthropic API formats

#### Key Features
- OpenAI compatibility
- Anthropic compatibility
- Docker support via docker-compose
- Configurable via environment variables

#### How to Use
```bash
# Using Go
git clone https://github.com/Tylerx404/z2api-go.git
cd z2api-go
go mod download
go run main.go

# Using Docker
docker-compose up -d
```

#### Configuration
- **TOKEN**: Z.ai token (optional for anonymous mode)
- **PORT**: 8080 (default)
- **DEBUG**: Enable debug mode
- **MODEL**: Default model (e.g., glm-5)
- **THINK_TAGS_MODE**: Processing mode for thinking tags

---

### 5. **ZtoApi-Deno**
**Repository:** https://github.com/LousyBook94/ZtoApi-Deno  
**Language:** Deno/TypeScript  
**Status:** Active

#### What It Does
- High-performance OpenAI AND Anthropic Claude compatible API proxy
- Designed specifically for Z.ai's GLM-4.5 and GLM-4.5V models
- Dual endpoint support (OpenAI and Claude formats)
- Built-in web dashboard with live request monitoring

#### Key Features
- **Dual API Support**: Both `/v1/` (OpenAI) and `/anthropic/v1/` (Claude) endpoints
- **Tool Calling**: Native tool support with built-in tools (time, URL fetching, hashing, math)
- **SSE Streaming**: Real-time token delivery for both APIs
- **Advanced Thinking**: 5 modes for processing thinking content
- **Web Dashboard**: Live request stats and monitoring
- **Deployable**: Deno Deploy or self-hosted options
- **Token Authentication**: Optional API key or anonymous fallback

#### Supported Models
See [models documentation](https://github.com/LousyBook94/ZtoApi-Deno/blob/master/docs/models.md)

#### How to Use
```bash
# Get Z.ai API token from https://chat.z.ai
# Set environment variables

# Run locally
deno run --allow-net --allow-env --allow-read main.ts
```

#### Configuration
- **Port**: 9090 (default)
- **ZAI_TOKEN**: Z.ai API token
- **OpenAI Endpoint**: http://localhost:9090/v1
- **Claude Endpoint**: http://localhost:9090/anthropic/v1

---

### 6. **Codex-Proxy**
**Repository:** https://github.com/cornellsh/codex-proxy  
**Language:** Python  
**Status:** Active

#### What It Does
- OpenAI Responses API proxy for Gemini and Z.AI (GLM) providers
- Translates OpenAI's Responses API to Gemini and Z.AI APIs
- Handles wire format differences, role mapping, and SSE stream formatting
- Allows Codex and other tools to use these providers instead of GPT

#### Key Features
- Full Responses API lifecycle with SSE events
- Multi-provider support (Gemini OAuth2 and Z.AI)
- Context compaction for both providers
- Tool support (function calling and web search)
- Docker-ready with hot-reload
- Configurable reasoning effort levels

#### How to Use
```bash
git clone https://github.com/cornellsh/codex-proxy.git
cd codex-proxy

# Docker
./scripts/control.sh start

# Direct (Python 3.14+ required)
python -m codex_proxy
```

#### Configuration
- **Port**: 8765 (default)
- **CODEX_PROXY_ZAI_API_KEY**: Z.ai API key
- **CODEX_PROXY_GEMINI_API_KEY**: Google AI Studio API key
- **Config File**: ~/.config/codex-proxy/config.json
- **Environment Variables**: Override all settings

---

## Comparison Table

| Project | Language | API Format | Key Feature | Port | Status |
|---------|----------|-----------|-------------|------|--------|
| **GLM Proxy** | Node.js | Anthropic + OpenAI | Claude Code integration, MCP, video analysis | 4567 | ⭐ Most Complete |
| **OpenAI-Compatible-API-Proxy-for-Z** | Binary | OpenAI | Simple GUI desktop app | ? | Active |
| **Copilot-Proxy** | Python | Ollama API | GitHub Copilot integration | 11434 | Active |
| **Z2api-Go** | Go | OpenAI + Anthropic | Lightweight, Docker support | 8080 | Active |
| **ZtoApi-Deno** | Deno | OpenAI + Anthropic | Dual endpoints, tool calling, dashboard | 9090 | Active |
| **Codex-Proxy** | Python | OpenAI Responses API | Multi-provider (Gemini + Z.ai) | 8765 | Active |

---

## Common Architecture Patterns

### Authentication
- All require Z.ai API key (or token)
- Some support environment variables: `ZAI_API_KEY`, `ZAI_TOKEN`
- Some support config files for persistence

### Endpoints
- **Anthropic Format**: `/v1/messages` or `/anthropic/v1/messages`
- **OpenAI Format**: `/v1/chat/completions`
- **Ollama Format**: Ollama API spec (port 11434)
- **Responses Format**: OpenAI Responses API

### Model Routing
- Auto-detection of vision vs text content
- Automatic model switching (glm-4.7 for text, glm-4.6v for vision)
- Some support video analysis

### Deployment
- All designed for localhost development
- Most have security warnings about not exposing to public internet
- Some support Docker deployment
- Some deployable to cloud (Deno Deploy, etc.)

---

## Security Considerations

**All projects include warnings:**
- ⚠️ Localhost-only by default (127.0.0.1)
- ⚠️ Not intended for production multi-user environments
- ⚠️ No built-in authentication layer
- ⚠️ API keys stored in memory or browser localStorage
- ⚠️ Designed for trusted development environments

---

## Use Cases

These proxies enable:

1. **Claude Code Integration**: Use GLM models with Claude Code CLI
2. **GitHub Copilot**: Use GLM Coding Plan with Copilot Chat
3. **Generic OpenAI Tools**: Any tool supporting OpenAI API can use GLM
4. **Anthropic Tools**: Any tool supporting Anthropic API can use GLM
5. **Web Search**: Add web search capabilities to GLM models
6. **Video Analysis**: Analyze videos with GLM-4.6V
7. **Tool Calling**: Execute functions via API with GLM

---

## Installation Recommendations

**For Claude Code Users**: Use **GLM Proxy** (dejay2/glmproxy)
- Best integration with Claude Code
- Most features (web search, MCP, video)
- Active development

**For GitHub Copilot Users**: Use **Copilot-Proxy** (modpotato/copilot-proxy)
- Direct Copilot Chat integration
- Easy model switching
- Python-based

**For Generic OpenAI Compatibility**: Use **ZtoApi-Deno** or **Z2api-Go**
- Lightweight options
- Both OpenAI and Anthropic support
- Good documentation

**For Simplicity**: Use **OpenAI-Compatible-API-Proxy-for-Z**
- Desktop GUI application
- No command line required
- Cross-platform

---

## Key Takeaway

The Z.ai/Zhipu AI ecosystem has a vibrant community of proxy projects that allow users to:
- Expose Z.ai subscriptions as standard APIs
- Integrate with existing AI tools and workflows
- Support both OpenAI and Anthropic API formats
- Add advanced features like web search, video analysis, and tool calling

**Most active and recommended**: GLM Proxy (dejay2/glmproxy) for comprehensive features and active maintenance.
