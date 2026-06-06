//! Prompt prefix-divergence diagnostic.
//!
//! Local inference servers (and Anthropic prompt caching) reuse work via
//! longest-prefix matching on the rendered prompt. Any call whose prompt is
//! NOT an append-only extension of the previous call's prompt forces a
//! re-prefill of everything after the divergence point — for a 14k-token
//! context at local prefill speeds (~250 tok/s) that is ~60s of dead wait.
//!
//! This module fingerprints each call's prompt (one hash per message plus one
//! for the tool schema) and classifies how consecutive calls in a session
//! relate, so prefix-cache misses become one-line diagnosable instead of
//! invisible. Cost per call: hashing the already-rendered messages
//! (microseconds) and ~8 bytes per message of retained state.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use serde_json::Value;

/// One hash per rendered message plus one for the tool schema.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptFingerprint {
    msg_hashes: Vec<u64>,
    tools_hash: u64,
}

/// How this call's prompt relates to the previous call's in the same session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptDelta {
    /// No previous fingerprint for this session in this process.
    First,
    /// Strict prefix extension — server-side prefix caches fully apply.
    AppendOnly { added_msgs: usize },
    /// The prompt changed before its end: everything after
    /// `first_divergent_msg` re-prefills. `tools_changed` flags schema churn,
    /// which diverges the rendered prompt at its very head (token ~0).
    Diverged {
        first_divergent_msg: usize,
        prev_msgs: usize,
        new_msgs: usize,
        tools_changed: bool,
    },
}

fn hash_value(v: &Value) -> u64 {
    let mut h = DefaultHasher::new();
    // Serialized form is what reaches the server; hash exactly that.
    v.to_string().hash(&mut h);
    h.finish()
}

/// Fingerprint a rendered prompt: per-message hashes + tool-schema hash.
pub fn fingerprint(messages: &[Value], tools: Option<&[Value]>) -> PromptFingerprint {
    let msg_hashes = messages.iter().map(hash_value).collect();
    let mut h = DefaultHasher::new();
    if let Some(defs) = tools {
        for def in defs {
            def.to_string().hash(&mut h);
        }
    }
    PromptFingerprint {
        msg_hashes,
        tools_hash: h.finish(),
    }
}

/// Classify the new fingerprint against the session's previous one.
pub fn compare(prev: Option<&PromptFingerprint>, new: &PromptFingerprint) -> PromptDelta {
    let Some(prev) = prev else {
        return PromptDelta::First;
    };
    let tools_changed = prev.tools_hash != new.tools_hash;
    let common = prev
        .msg_hashes
        .iter()
        .zip(new.msg_hashes.iter())
        .take_while(|(a, b)| a == b)
        .count();
    if !tools_changed && common == prev.msg_hashes.len() && new.msg_hashes.len() >= common {
        return PromptDelta::AppendOnly {
            added_msgs: new.msg_hashes.len() - common,
        };
    }
    PromptDelta::Diverged {
        first_divergent_msg: common,
        prev_msgs: prev.msg_hashes.len(),
        new_msgs: new.msg_hashes.len(),
        tools_changed,
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
    /// append-only growth, mid-prompt mutation, tool-schema churn, and
    /// tail truncation — the failure modes that distinguish a prefix-cache
    /// hit from a 60s re-prefill.
    #[test]
    fn test_prompt_delta_classification() {
        let tools = vec![json!({"function": {"name": "exec"}})];
        let base = fingerprint(&msgs(&["sys", "a", "b"]), Some(&tools));

        // First call of a session.
        assert_eq!(compare(None, &base), PromptDelta::First);

        // Identical prompt (retry) — append of zero.
        assert_eq!(
            compare(Some(&base), &base),
            PromptDelta::AppendOnly { added_msgs: 0 }
        );

        // Append-only growth — the cache-friendly steady state.
        let grown = fingerprint(&msgs(&["sys", "a", "b", "c", "d"]), Some(&tools));
        assert_eq!(
            compare(Some(&base), &grown),
            PromptDelta::AppendOnly { added_msgs: 2 }
        );

        // Mid-prompt mutation (e.g. history trim rewrote message 1).
        let mutated = fingerprint(&msgs(&["sys", "CHANGED", "b", "c"]), Some(&tools));
        assert_eq!(
            compare(Some(&base), &mutated),
            PromptDelta::Diverged {
                first_divergent_msg: 1,
                prev_msgs: 3,
                new_msgs: 4,
                tools_changed: false,
            }
        );

        // Tool-schema churn: messages append-only but the schema changed —
        // the rendered prompt head moves, so this must NOT read as append.
        let fewer_tools = fingerprint(&msgs(&["sys", "a", "b", "c"]), None);
        assert_eq!(
            compare(Some(&base), &fewer_tools),
            PromptDelta::Diverged {
                first_divergent_msg: 3,
                prev_msgs: 3,
                new_msgs: 4,
                tools_changed: true,
            }
        );

        // Tail truncation: new prompt is a strict prefix of the previous one.
        let truncated = fingerprint(&msgs(&["sys", "a"]), Some(&tools));
        assert_eq!(
            compare(Some(&base), &truncated),
            PromptDelta::Diverged {
                first_divergent_msg: 2,
                prev_msgs: 3,
                new_msgs: 2,
                tools_changed: false,
            }
        );
    }
}
