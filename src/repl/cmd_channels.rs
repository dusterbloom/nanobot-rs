//! Channel REPL commands: /whatsapp, /telegram, /email, /voice.

use std::io::{self, Write as _};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use super::*;

impl ReplContext {
    /// /whatsapp — start WhatsApp channel in background.
    pub(super) fn cmd_whatsapp(&mut self) {
        self.active_channels.retain(|ch| !ch.handle.is_finished());
        if self.active_channels.iter().any(|ch| ch.name == "whatsapp") {
            println!("\n  WhatsApp is already running. Use /stop to stop channels.\n");
            return;
        }
        let mut gw_config = load_config(None);
        cli::check_api_key(&gw_config);
        gw_config.channels.whatsapp.enabled = true;
        gw_config.channels.telegram.enabled = false;
        gw_config.channels.feishu.enabled = false;
        gw_config.channels.email.enabled = false;
        let stop = Arc::new(AtomicBool::new(false));
        let stop2 = stop.clone();
        let dtx = self.display_tx.clone();
        let ch = self.core_handle.clone();
        println!("\n  Scan the QR code when it appears");
        let handle = tokio::spawn(async move {
            cli::run_gateway_async(gw_config, ch, Some(stop2), Some(dtx), None).await;
        });
        self.active_channels.push(super::super::ActiveChannel {
            name: "whatsapp".to_string(),
            stop,
            handle,
        });
        println!("  WhatsApp running in background. Continue chatting.\n");
    }

    /// /telegram — start Telegram channel in background.
    pub(super) fn cmd_telegram(&mut self) {
        self.active_channels.retain(|ch| !ch.handle.is_finished());
        if self.active_channels.iter().any(|ch| ch.name == "telegram") {
            println!("\n  Telegram is already running. Use /stop to stop channels.\n");
            return;
        }
        println!();
        let mut gw_config = load_config(None);
        cli::check_api_key(&gw_config);
        let saved_token = &gw_config.channels.telegram.token;
        let token = if !saved_token.is_empty() {
            println!("  Using saved bot token");
            saved_token.clone()
        } else {
            println!("  No Telegram bot token found.");
            println!("  Get one from @BotFather on Telegram.\n");
            let tok_prompt = "  Enter bot token: ";
            let t = match self.rl.as_mut().unwrap().readline(tok_prompt) {
                Ok(line) => line.trim().to_string(),
                Err(_) => {
                    return;
                }
            };
            if t.is_empty() {
                println!("  Cancelled.\n");
                return;
            }
            print!("  Validating token... ");
            io::stdout().flush().ok();
            if cli::validate_telegram_token(&t) {
                println!("valid!\n");
            } else {
                println!("invalid!");
                println!("  Check the token and try again.\n");
                return;
            }
            let save_prompt = "  Save token to config for next time? [Y/n] ";
            if let Ok(answer) = self.rl.as_mut().unwrap().readline(save_prompt) {
                if !answer.trim().eq_ignore_ascii_case("n") {
                    let mut save_cfg = load_config(None);
                    save_cfg.channels.telegram.token = t.clone();
                    save_config(&save_cfg, None);
                    println!("  Token saved to ~/.nanobot/config.json\n");
                }
            }
            t
        };
        gw_config.channels.telegram.token = token;
        gw_config.channels.telegram.enabled = true;
        gw_config.channels.whatsapp.enabled = false;
        gw_config.channels.feishu.enabled = false;
        gw_config.channels.email.enabled = false;
        let stop = Arc::new(AtomicBool::new(false));
        let stop2 = stop.clone();
        let dtx = self.display_tx.clone();
        let ch = self.core_handle.clone();
        let handle = tokio::spawn(async move {
            cli::run_gateway_async(gw_config, ch, Some(stop2), Some(dtx), None).await;
        });
        self.active_channels.push(super::super::ActiveChannel {
            name: "telegram".to_string(),
            stop,
            handle,
        });
        println!("  Telegram running in background. Continue chatting.\n");
    }

    /// /email — start Email channel in background.
    pub(super) fn cmd_email(&mut self) {
        self.active_channels.retain(|ch| !ch.handle.is_finished());
        if self.active_channels.iter().any(|ch| ch.name == "email") {
            println!("\n  Email is already running. Use /stop to stop channels.\n");
            return;
        }
        println!();
        let mut gw_config = load_config(None);
        cli::check_api_key(&gw_config);
        let email_cfg = &gw_config.channels.email;
        if email_cfg.imap_host.is_empty()
            || email_cfg.username.is_empty()
            || email_cfg.password.is_empty()
        {
            println!("  Email not configured. Run `nanobot email` first or add settings to config.json.\n");
            return;
        }
        println!("  Starting Email channel...");
        println!("  Polling {}", email_cfg.imap_host);
        gw_config.channels.email.enabled = true;
        gw_config.channels.whatsapp.enabled = false;
        gw_config.channels.telegram.enabled = false;
        gw_config.channels.feishu.enabled = false;
        let stop = Arc::new(AtomicBool::new(false));
        let stop2 = stop.clone();
        let dtx = self.display_tx.clone();
        let ch = self.core_handle.clone();
        let handle = tokio::spawn(async move {
            cli::run_gateway_async(gw_config, ch, Some(stop2), Some(dtx), None).await;
        });
        self.active_channels.push(super::super::ActiveChannel {
            name: "email".to_string(),
            stop,
            handle,
        });
        println!("  Email running in background. Continue chatting.\n");
    }

    /// /voice — toggle voice mode.
    #[cfg(feature = "voice")]
    pub(super) async fn cmd_voice(&mut self) {
        if self.voice_session.is_some() {
            if let Some(ref mut vs) = self.voice_session {
                vs.stop_playback();
            }
            self.voice_session = None;
            // Restore thinking display when voice mode turns off.
            self.core_handle
                .counters
                .suppress_thinking_in_tts
                .store(false, Ordering::Relaxed);
            println!("\nVoice mode OFF\n");
        } else {
            match crate::voice_pipeline::VoiceSession::with_lang(self.lang.as_deref()).await {
                Ok(vs) => {
                    self.voice_session = Some(vs);
                    // Auto-suppress thinking tokens from TTS.
                    self.core_handle
                        .counters
                        .suppress_thinking_in_tts
                        .store(true, Ordering::Relaxed);
                    println!(
                        "\nVoice mode ON. Ctrl+Space or Enter to speak/interrupt, type for text.\n"
                    );
                }
                Err(e) => eprintln!("\nFailed to start voice mode: {}\n", e),
            }
        }
    }
}
