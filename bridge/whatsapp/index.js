#!/usr/bin/env node
// nanoclaw WhatsApp bridge - WebSocket server that bridges whatsapp-web.js to Rust.
//
// Usage: node index.js [--port 3001] [--session-dir ~/.nanoclaw/whatsapp-session]
//
// Protocol (JSON over WebSocket):
//   Bridge -> Rust: { type: "message"|"status"|"qr"|"error", ... }
//   Rust -> Bridge: { type: "send", to: "...", text: "..." }

const { Client, LocalAuth } = require("whatsapp-web.js");
const qrcode = require("qrcode-terminal");
const { WebSocketServer } = require("ws");
const path = require("path");
const os = require("os");

// Parse CLI args.
const args = process.argv.slice(2);
function getArg(name, fallback) {
  const idx = args.indexOf(name);
  return idx !== -1 && args[idx + 1] ? args[idx + 1] : fallback;
}

const PORT = parseInt(getArg("--port", "3001"), 10);
const SESSION_DIR = getArg(
  "--session-dir",
  path.join(os.homedir(), ".nanoclaw", "whatsapp-session")
);

// Track connected Rust clients.
const clients = new Set();

function broadcast(obj) {
  const msg = JSON.stringify(obj);
  for (const ws of clients) {
    if (ws.readyState === 1) {
      ws.send(msg);
    }
  }
}

// WhatsApp client with local auth persistence.
const wa = new Client({
  authStrategy: new LocalAuth({ dataPath: SESSION_DIR }),
  puppeteer: {
    headless: true,
    args: [
      "--no-sandbox",
      "--disable-setuid-sandbox",
      "--disable-dev-shm-usage",
      "--disable-gpu",
      "--single-process",
    ],
  },
});

wa.on("qr", (qr) => {
  qrcode.generate(qr, { small: true });
  broadcast({ type: "qr", qr });
});

wa.on("ready", () => {
  console.log("WhatsApp client ready");
  broadcast({ type: "status", status: "ready" });
});

wa.on("authenticated", () => {
  console.log("WhatsApp authenticated");
  broadcast({ type: "status", status: "authenticated" });
});

wa.on("auth_failure", (msg) => {
  console.error("WhatsApp auth failure:", msg);
  broadcast({ type: "error", error: `auth_failure: ${msg}` });
});

wa.on("disconnected", (reason) => {
  console.log("WhatsApp disconnected:", reason);
  broadcast({ type: "status", status: "disconnected" });
});

wa.on("message", async (msg) => {
  const chat = await msg.getChat();
  const payload = {
    type: "message",
    id: msg.id._serialized,
    sender: msg.from,
    content: msg.body,
    timestamp: msg.timestamp,
    isGroup: chat.isGroup,
  };

  // Handle voice messages (push-to-talk).
  if (msg.hasMedia && (msg.type === "ptt" || msg.type === "audio")) {
    try {
      const media = await msg.downloadMedia();
      if (media && media.data) {
        const fs = require("fs");
        const mediaDir = path.join(os.homedir(), ".nanoclaw", "media");
        fs.mkdirSync(mediaDir, { recursive: true });
        const ext = msg.type === "ptt" ? ".ogg" : ".ogg";
        const filename = `wa_voice_${Date.now()}${ext}`;
        const filePath = path.join(mediaDir, filename);
        fs.writeFileSync(filePath, Buffer.from(media.data, "base64"));
        payload.voiceFile = filePath;
        payload.content = "[Voice Message]";
        console.log(`Downloaded voice message to ${filePath}`);
      }
    } catch (err) {
      console.error("Failed to download voice media:", err.message);
      payload.content = "[Voice Message: download failed]";
    }
  }

  broadcast(payload);
});

// WebSocket server for Rust to connect.
const wss = new WebSocketServer({ port: PORT }, () => {
  console.log(`WhatsApp bridge WebSocket server listening on port ${PORT}`);
});

wss.on("connection", (ws) => {
  console.log("Rust client connected");
  clients.add(ws);

  ws.on("message", async (data) => {
    try {
      const msg = JSON.parse(data.toString());
      const { MessageMedia } = require("whatsapp-web.js");

      if (msg.type === "send" && msg.to && msg.text) {
        // Ensure JID format: add @c.us if missing.
        const to = msg.to.includes("@") ? msg.to : `${msg.to}@c.us`;
        await wa.sendMessage(to, msg.text);
      } else if (msg.type === "sendMedia" && msg.to && msg.media) {
        // Send media (voice note, image, etc.)
        const to = msg.to.includes("@") ? msg.to : `${msg.to}@c.us`;
        const media = new MessageMedia(
          msg.mimetype || "audio/ogg",
          msg.media, // base64 data
          msg.filename || "voice.ogg"
        );
        const options = {};
        if (msg.caption) options.caption = msg.caption;
        if (msg.mimetype && msg.mimetype.startsWith("audio/")) {
          options.sendAudioAsVoice = true;
        }
        await wa.sendMessage(to, media, options);
      }
    } catch (err) {
      console.error("Error handling message from Rust:", err.message);
    }
  });

  ws.on("close", () => {
    console.log("Rust client disconnected");
    clients.delete(ws);
  });
});

// Graceful shutdown.
async function shutdown() {
  console.log("Shutting down WhatsApp bridge...");
  try {
    await wa.destroy();
  } catch (_) {}
  wss.close();
  process.exit(0);
}

process.on("SIGTERM", shutdown);
process.on("SIGINT", shutdown);

// Start WhatsApp client.
console.log(`Starting WhatsApp bridge (port=${PORT}, session=${SESSION_DIR})`);
wa.initialize().catch((err) => {
  console.error("Failed to initialize WhatsApp client:", err.message);
  process.exit(1);
});
