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

/// One hash per rendered message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptFingerprint {
    msg_hashes: Vec<u64>,
}

impl PromptFingerprint {
    /// Hash at a given message index. Exposed for diagnostic logging
    /// at divergence points (shared.rs dumps prev vs new hash to
    /// pinpoint which message's rendered bytes changed).
    pub fn msg_hash_at(&self, idx: usize) -> Option<u64> {
        self.msg_hashes.get(idx).copied()
    }
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
}
