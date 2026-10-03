// Error-protocol layer-3 backlog (docs/research/2026-08-06-error-conventions-and-host-bridge.md §3.6):
// the deny regime in Cargo.toml is live; this module still carries pre-existing
// violations of the lints below. Remove this allow as the module migrates onto
// the regime.
// Tracking: docs/error-protocol-backlog.md
#![allow(clippy::shadow_reuse)]
//! Prompt prefix-divergence diagnostic.
//!
//! Local inference servers (and Anthropic prompt caching) reuse work via
//! longest-prefix matching on the rendered prompt. Any call whose prompt is
//! NOT an append-only extension of the previous call's prompt forces a
//! re-prefill of everything after the divergence point — for a 14k-token
//! context at local prefill speeds (~250 tok/s) that is ~60s of dead wait.
//!
//! This module fingerprints each call's prompt (one hash per message) and
//! classifies how consecutive calls in a session relate, so prefix-cache
//! misses become one-line diagnosable instead of invisible. Cost per call:
//! hashing the already-rendered messages (microseconds) and ~8 bytes per
//! message of retained state.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use serde_json::Value;

use crate::agent::token_budget::TokenBudget;

/// The inference server's own token count for one fingerprinted prompt,
/// paired with the client estimate of the same bytes. The pair is the
/// observed tokenizer ratio: the client counts with cl100k while local
/// servers tokenize with their own BPE and chat template, which measured
/// 11–17% higher on a tool-heavy Qwen session (`20261002_141711_389268`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ServerMeasurement {
    server_tokens: usize,
    client_estimate: usize,
}

/// One hash per rendered message, plus — once the prompt has shipped — what
/// the server counted for exactly these bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptFingerprint {
    msg_hashes: Vec<u64>,
    measured: Option<ServerMeasurement>,
}

impl PromptFingerprint {
    /// Hash at a given message index. Exposed for diagnostic logging
    /// at divergence points (shared.rs dumps prev vs new hash to
    /// pinpoint which message's rendered bytes changed).
    pub fn msg_hash_at(&self, idx: usize) -> Option<u64> {
        self.msg_hashes.get(idx).copied()
    }

    /// Attach the server-reported prompt size for this exact prompt. A zero
    /// count (providers without usage) leaves the fingerprint unmeasured, so
    /// [`anchored_prompt_tokens`] falls back to the full client estimate.
    #[must_use]
    pub const fn with_server_measurement(
        mut self,
        server_tokens: usize,
        client_estimate: usize,
    ) -> Self {
        if server_tokens > 0 && client_estimate > 0 {
            self.measured = Some(ServerMeasurement {
                server_tokens,
                client_estimate,
            });
        }
        self
    }
}

/// Prompt size of `rendered` plus `tool_def_tokens`, anchored on the server's
/// truth where it exists: when `prev` was measured and `rendered` extends it
/// append-only, the shipped prefix costs exactly what the server counted and
/// only the new suffix is estimated — scaled by the ratio the server showed
/// on the prefix, rounded up so the anchored count never trails the server's.
/// Any other relation (first call, rotation, trim, compaction, epoch change —
/// each of which replaces the stored fingerprint) is the full client estimate.
pub fn anchored_prompt_tokens(
    prev: Option<&PromptFingerprint>,
    rendered: &[Value],
    tool_def_tokens: usize,
) -> usize {
    let full_estimate = || TokenBudget::estimate_tokens(rendered).saturating_add(tool_def_tokens);
    let Some(measured) = prev.and_then(|shipped| shipped.measured) else {
        return full_estimate();
    };
    let PromptDelta::AppendOnly { added_msgs } = compare(prev, &fingerprint(rendered)) else {
        return full_estimate();
    };
    let Some(suffix) = rendered
        .len()
        .checked_sub(added_msgs)
        .and_then(|start| rendered.get(start..))
    else {
        return full_estimate();
    };
    let scaled_suffix = TokenBudget::estimate_tokens(suffix)
        .saturating_mul(measured.server_tokens)
        .div_ceil(measured.client_estimate);
    measured.server_tokens.saturating_add(scaled_suffix)
}

/// How this call's prompt relates to the previous call's in the same session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptDelta {
    /// No previous fingerprint for this session in this process.
    First,
    /// Strict prefix extension — server-side prefix caches fully apply.
    AppendOnly { added_msgs: usize },
    /// The prompt changed before its end: everything after
    /// `first_divergent_msg` re-prefills.
    Diverged {
        first_divergent_msg: usize,
        prev_msgs: usize,
        new_msgs: usize,
    },
}

pub fn hash_value(v: &Value) -> u64 {
    let mut h = DefaultHasher::new();
    // Serialized form is what reaches the server; hash exactly that.
    v.to_string().hash(&mut h);
    h.finish()
}

/// Fingerprint a rendered prompt: per-message hashes only.
///
/// Tool schemas are metadata about *what tools are available*, not about the
/// conversation content. Changing available tools (e.g. via subcalls, router
/// trio stripping, or dynamic scoping) should **not** invalidate the prefix
/// cache for the conversation messages themselves — the server can still reuse
/// the prompt prefix for messages that haven't changed.
pub fn fingerprint(messages: &[Value]) -> PromptFingerprint {
    PromptFingerprint {
        msg_hashes: messages.iter().map(hash_value).collect(),
        measured: None,
    }
}

/// Hash the tool-definition array actually sent to the provider.
///
/// Unlike [`fingerprint`], this DOES track tool schemas, because chat templates
/// render the tool block into the token stream (often at the prompt head), so a
/// changing tool block busts the prefix cache even when messages are append-only.
/// Use this alongside the message fingerprint to catch that case.
pub fn hash_tools(tools: &[Value]) -> u64 {
    let mut h = DefaultHasher::new();
    for t in tools {
        // Serialized form is what reaches the template; hash exactly that.
        t.to_string().hash(&mut h);
    }
    h.finish()
}

/// Hash the exact prompt-bearing parts of a provider request. This combines
/// rendered messages (including transport metadata) and the tool schema block,
/// so an identical value means the model would receive no new evidence.
pub fn hash_provider_request(messages: &[Value], tools: &[Value]) -> u64 {
    let mut h = DefaultHasher::new();
    messages.len().hash(&mut h);
    for message in messages {
        message.to_string().hash(&mut h);
    }
    tools.len().hash(&mut h);
    for tool in tools {
        tool.to_string().hash(&mut h);
    }
    h.finish()
}

/// Classify the new fingerprint against the session's previous one.
pub fn compare(prev: Option<&PromptFingerprint>, new: &PromptFingerprint) -> PromptDelta {
    let Some(prev) = prev else {
        return PromptDelta::First;
    };
    let common = prev
        .msg_hashes
        .iter()
        .zip(new.msg_hashes.iter())
        .take_while(|(a, b)| a == b)
        .count();
    if common == prev.msg_hashes.len() && new.msg_hashes.len() >= common {
        return PromptDelta::AppendOnly {
            added_msgs: new.msg_hashes.len() - common,
        };
    }
    PromptDelta::Diverged {
        first_divergent_msg: common,
        prev_msgs: prev.msg_hashes.len(),
        new_msgs: new.msg_hashes.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn msgs(texts: &[&str]) -> Vec<Value> {
        texts
            .iter()
            .map(|t| json!({"role": "user", "content": t}))
            .collect()
    }

    /// One test, all classifications: first call, identical retry,
    /// append-only growth, mid-prompt mutation, and tail truncation.
    #[test]
    fn test_prompt_delta_classification() {
        let base = fingerprint(&msgs(&["sys", "a", "b"]));

        // First call of a session.
        assert_eq!(compare(None, &base), PromptDelta::First);

        // Identical prompt (retry) — append of zero.
        assert_eq!(
            compare(Some(&base), &base),
            PromptDelta::AppendOnly { added_msgs: 0 }
        );

        // Append-only growth — the cache-friendly steady state.
        let grown = fingerprint(&msgs(&["sys", "a", "b", "c", "d"]));
        assert_eq!(
            compare(Some(&base), &grown),
            PromptDelta::AppendOnly { added_msgs: 2 }
        );

        // Mid-prompt mutation (e.g. history trim rewrote message 1).
        let mutated = fingerprint(&msgs(&["sys", "CHANGED", "b", "c"]));
        assert_eq!(
            compare(Some(&base), &mutated),
            PromptDelta::Diverged {
                first_divergent_msg: 1,
                prev_msgs: 3,
                new_msgs: 4,
            }
        );

        // Tail truncation: new prompt is a strict prefix of the previous one.
        let truncated = fingerprint(&msgs(&["sys", "a"]));
        assert_eq!(
            compare(Some(&base), &truncated),
            PromptDelta::Diverged {
                first_divergent_msg: 2,
                prev_msgs: 3,
                new_msgs: 2,
            }
        );
    }

    /// The general rule behind the anchor: against a server whose tokenizer
    /// counts a fixed fraction more than the client (any ratio up to the
    /// observed ~17%), the anchored count of every append-only extension is
    /// never below the server's count and at most rounding above it — so it
    /// crosses any soft/hard wall no later than the server would. Without a
    /// measured, matching prefix it is exactly the full client estimate.
    #[test]
    fn anchored_count_never_trails_a_skewed_server_tokenizer() {
        let server_count = |messages: &[Value], tools: usize, ratio_pct: usize| {
            (TokenBudget::estimate_tokens(messages) + tools)
                .saturating_mul(ratio_pct)
                .div_ceil(100)
        };
        let tools = 300;
        let shipped = msgs(&[
            "You are nanobot. Native tools: read_file, exec.",
            "Compare the two posts about durable agents and summarise them.",
            "I will fetch both pages first.",
            "<html>first page body with several paragraphs of text</html>",
        ]);
        let client_estimate = TokenBudget::estimate_tokens(&shipped) + tools;
        let additions = msgs(&[
            "I called web_fetch and read the second page as well.",
            "<html>second page body, longer, with lists and code blocks</html>",
            "Both posts argue for checkpointed state; the second adds replay.",
            "Now write the comparison to notes.md.",
        ]);

        for ratio_pct in [100, 105, 110, 117] {
            let prev = fingerprint(&shipped)
                .with_server_measurement(server_count(&shipped, tools, ratio_pct), client_estimate);
            for added in 0..=additions.len() {
                let mut rendered = shipped.clone();
                rendered.extend(additions.iter().take(added).cloned());
                let actual = server_count(&rendered, tools, ratio_pct);
                let anchored = anchored_prompt_tokens(Some(&prev), &rendered, tools);
                assert!(
                    anchored >= actual,
                    "ratio {ratio_pct}%, {added} new messages: anchored {anchored} trails server {actual}"
                );
                assert!(
                    anchored <= actual + 3,
                    "ratio {ratio_pct}%, {added} new messages: anchored {anchored} overshoots server {actual}"
                );
            }
        }

        // No anchor, unmeasured anchor, or a diverged prefix: the full estimate.
        let mut rendered = shipped.clone();
        rendered.extend(additions.iter().take(2).cloned());
        let full = TokenBudget::estimate_tokens(&rendered) + tools;
        assert_eq!(anchored_prompt_tokens(None, &rendered, tools), full);
        assert_eq!(
            anchored_prompt_tokens(Some(&fingerprint(&shipped)), &rendered, tools),
            full
        );
        let measured =
            fingerprint(&shipped).with_server_measurement(client_estimate * 2, client_estimate);
        let mut diverged = rendered.clone();
        if let Some(slot) = diverged.get_mut(1) {
            *slot = json!({"role": "user", "content": "rewritten history"});
        }
        assert_eq!(
            anchored_prompt_tokens(Some(&measured), &diverged, tools),
            TokenBudget::estimate_tokens(&diverged) + tools
        );
    }
}
