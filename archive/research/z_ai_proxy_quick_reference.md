# Z.ai Proxy Projects - Quick Reference

## 🔗 GitHub Repositories

### Primary Projects

1. **GLM Proxy** (Most Complete)
   - URL: https://github.com/dejay2/glmproxy
   - Language: Node.js/TypeScript
   - Best for: Claude Code, comprehensive features
   - Port: 4567
   - Features: MCP, web search, video analysis, reasoning injection

2. **Copilot-Proxy** (GitHub Copilot)
   - URL: https://github.com/modpotato/copilot-proxy
   - Language: Python
   - Best for: GitHub Copilot Chat integration
   - Port: 11434 (Ollama API)
   - Features: Model switching, PyPI installable

3. **ZtoApi-Deno** (Dual Endpoints)
   - URL: https://github.com/LousyBook94/ZtoApi-Deno
   - Language: Deno/TypeScript
   - Best for: Both OpenAI and Anthropic APIs
   - Port: 9090
   - Features: Tool calling, dashboard, streaming

4. **Z2api-Go** (Lightweight)
   - URL: https://github.com/Tylerx404/z2api-go
   - Language: Go
   - Best for: Minimal overhead, Docker
   - Port: 8080
   - Features: OpenAI + Anthropic compatible

5. **OpenAI-Compatible-API-Proxy-for-Z** (GUI)
   - URL: https://github.com/Idkwhattona/OpenAI-Compatible-API-Proxy-for-Z
   - Language: Binary/Desktop App
   - Best for: Non-technical users
   - Features: Desktop GUI, cross-platform

6. **Codex-Proxy** (Multi-Provider)
   - URL: https://github.com/cornellsh/codex-proxy
   - Language: Python
   - Best for: Gemini + Z.ai support
   - Port: 8765
   - Features: Multi-provider, context compaction

---

## 🚀 Quick Start Commands

### GLM Proxy (Recommended)
```bash
npm install -g dejay2/glmproxy
export ZAI_API_KEY="your-key-here"
ccglm  # Start proxy + Claude Code
```

### Copilot-Proxy
```bash
uv pip install copilot-proxy
export ZAI_API_KEY="your-key-here"
uvx copilot-proxy serve
```

### ZtoApi-Deno
```bash
git clone https://github.com/LousyBook94/ZtoApi-Deno
cd ZtoApi-Deno
export ZAI_TOKEN="your-token-here"
deno run --allow-net --allow-env --allow-read main.ts
```

### Z2api-Go
```bash
git clone https://github.com/Tylerx404/z2api-go
cd z2api-go
export TOKEN="your-token-here"
go run main.go
```

---

## 📋 API Endpoints by Format

### OpenAI Compatible (Most Common)
- Endpoint: `http://localhost:PORT/v1/chat/completions`
- Used by: Z2api-Go, ZtoApi-Deno, OpenAI-Compatible-API-Proxy-for-Z
- Port: 8080 (Go), 9090 (Deno), varies (GUI)

### Anthropic Compatible
- Endpoint: `http://localhost:PORT/v1/messages` or `/anthropic/v1/messages`
- Used by: GLM Proxy, ZtoApi-Deno, Codex-Proxy
- Port: 4567 (GLM Proxy), 9090 (ZtoApi-Deno), 8765 (Codex-Proxy)

### Ollama Compatible
- Endpoint: `http://localhost:11434/api/generate`
- Used by: Copilot-Proxy
- Port: 11434
- Best for: GitHub Copilot Chat

---

## 🔑 Environment Variables

### Common Variables
```bash
# Z.ai Authentication
ZAI_API_KEY="your-z-ai-api-key"        # Most projects
ZAI_TOKEN="your-z-ai-token"            # Some projects
TOKEN="your-token"                      # Z2api-Go

# Server Configuration
PORT=8080                               # Server port
HOST=127.0.0.1                          # Server host
LOG_LEVEL=info                          # Logging level
DEBUG=false                             # Debug mode
```

### Project-Specific Variables
```bash
# GLM Proxy
ZAI_API_KEY=                            # Required
REF_API_KEY=                            # Optional (Ref Tools MCP)
CONTEXT7_API_KEY=                       # Optional (Context7 MCP)
STREAMING_ENABLED=false                 # Enable SSE streaming

# Copilot-Proxy
ZAI_API_KEY=                            # Required
ZAI_API_BASE_URL=https://api.z.ai/...  # Optional (custom endpoint)

# Codex-Proxy
CODEX_PROXY_ZAI_API_KEY=                # Z.ai key
CODEX_PROXY_GEMINI_API_KEY=             # Google key
CODEX_PROXY_PORT=8765                   # Server port
```

---

## 🎯 Feature Comparison

| Feature | GLM Proxy | Copilot | ZtoApi | Z2api-Go | GUI | Codex |
|---------|-----------|---------|--------|----------|-----|-------|
| OpenAI API | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Anthropic API | ✓ | ✗ | ✓ | ✓ | ✗ | ✓ |
| Web Search | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ |
| Video Analysis | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Tool Calling | ✓ | ✗ | ✓ | ✗ | ✗ | ✓ |
| Streaming | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Dashboard | ✓ | ✗ | ✓ | ✗ | ✓ | ✗ |
| Docker | ✗ | ✓ | ✗ | ✓ | ✗ | ✓ |
| PyPI Package | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ |
| Claude Code | ✓ | ✗ | ✓ | ✓ | ✗ | ✗ |
| Copilot Chat | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ |

---

## 📊 Default Ports

- **4567**: GLM Proxy
- **8080**: Z2api-Go
- **8765**: Codex-Proxy
- **9090**: ZtoApi-Deno
- **11434**: Copilot-Proxy (Ollama API)
- **Varies**: OpenAI-Compatible-API-Proxy-for-Z (GUI app)

---

## ⚠️ Security Notes

**All projects are designed for localhost development only:**
- Default binding: `127.0.0.1` (localhost only)
- No built-in authentication
- API keys stored in memory or localStorage
- Not suitable for production or multi-user environments
- Do not expose to public internet

---

## 📝 Getting Z.ai API Credentials

1. Visit https://z.ai or https://chat.z.ai
2. Sign up or log in
3. Get your API key/token from settings
4. For Coding Plan: Get dedicated API key from Z.ai Coding Plan dashboard

---

## 🔧 Troubleshooting

### "Port already in use"
- Check if Ollama is running (uses 11434)
- Use `lsof -i :PORT` (Unix/Mac) or `netstat -ano | findstr :PORT` (Windows)
- Kill conflicting process or use different port

### "API key invalid"
- Verify key from Z.ai dashboard
- Check key hasn't expired
- Ensure correct environment variable name

### "Requests timing out"
- GLM models can take 10-30 seconds
- Try shorter prompts or reduce max_tokens
- Check internet connection

### "Streaming not working"
- Enable streaming in configuration
- Check if backend supports streaming
- Verify client supports SSE

---

## 📚 Additional Resources

- Z.ai Official: https://z.ai
- Z.ai Chat: https://chat.z.ai
- GLM Models Docs: https://open.bigmodel.cn/

---

## 🏆 Recommendation Summary

**Best Overall**: GLM Proxy (dejay2/glmproxy)
- Most features
- Active development
- Claude Code integration
- Web search & video support

**Best for Copilot**: Copilot-Proxy (modpotato/copilot-proxy)
- Direct Copilot Chat integration
- Easy installation (PyPI)
- Model switching

**Best for Simplicity**: OpenAI-Compatible-API-Proxy-for-Z
- GUI application
- No command line
- Cross-platform

**Best for Lightweight**: Z2api-Go
- Single binary
- Docker support
- Minimal dependencies
