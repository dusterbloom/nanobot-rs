# Continuity Ledger: Channels Expansion

## Goal
Expand nanobot beyond Telegram with Email, WhatsApp, and enhanced Telegram channels — all production-ready with proper config support.

## Constraints
- All channels use the same InboundMessage/OutboundMessage bus
- Config via `~/.nanobot/config.json` (camelCase, serde)
- WhatsApp uses external Node.js bridge (bridge/whatsapp/)
- Email uses IMAP idle + SMTP (lettre crate)

## Key Decisions
- **WhatsApp via Node bridge**: whatsapp-web.js has no Rust equivalent; bridge/whatsapp/index.js connects via WebSocket to nanobot [2026-02-10]
- **Email as full channel**: IMAP IDLE for incoming, SMTP for outgoing, thread tracking via Message-ID/In-Reply-To [2026-02-10]
- **Email tool for agent**: Separate `EmailTool` lets the agent compose/send emails proactively [2026-02-10]
- **Telegram enhancements**: Extended with richer message handling (+154 lines) [2026-02-10]
- **Per-message ToolRegistry**: Tools built per-message with channel/chat_id baked in (from graduated-features Phase 7) [2026-02-09]

## State
- Done:
  - [x] Email channel (`src/channels/email.rs` — 1,360 lines, IMAP+SMTP)
  - [x] Email tool (`src/agent/tools/email.rs` — 260 lines)
  - [x] WhatsApp bridge (`bridge/whatsapp/` — Node.js, whatsapp-web.js)
  - [x] WhatsApp channel refactor (`src/channels/whatsapp.rs` — +110 lines)
  - [x] Telegram enhancements (`src/channels/telegram.rs` — +154 lines)
  - [x] Config schema for new channels (`src/config/schema.rs` — +54 lines)
  - [x] Channel manager registration (`src/channels/manager.rs`)
  - [x] Main.rs integration (+416 lines)
  - [x] Pushed to origin at `1603896`
- Now: Idle
- Remaining:
  - [ ] End-to-end test: email channel with real IMAP/SMTP
  - [ ] End-to-end test: WhatsApp bridge connection
  - [ ] Telegram enhanced features testing
  - [ ] Error handling / reconnect logic for email IMAP idle
  - [ ] WhatsApp bridge auto-restart on disconnect

## Open Questions
- UNCONFIRMED: Does email IMAP idle handle connection drops gracefully?
- UNCONFIRMED: WhatsApp bridge QR code auth flow — tested end-to-end?
- UNCONFIRMED: Are new config fields documented in README?

## Working Set
- Files: `src/channels/{email,whatsapp,telegram,manager}.rs`, `src/agent/tools/email.rs`, `src/config/schema.rs`, `src/main.rs`, `bridge/whatsapp/`
- Branch: main
- Commit: `1603896`
- Build: `cargo build --release`
- Test: `cargo test`
