//! Frozen-prefix guard for prefix-cache-safe context cleanup.
//!
//! Local inference servers reuse work via longest-prefix matching on the
//! rendered prompt: a call is fast only if its prompt is an append-only
//! extension of the previous call's. Any cleanup pass that rewrites a message
//! *before the tail* moves the first-divergent token earlier, so the server
//! discards its KV cache from that point and re-prefills everything after —
//! ~65s for a 19k-token context at local prefill speeds, with the GPU pegged.
//!
//! The per-iteration cleanup passes ([`super::context_hygiene::hygiene_pipeline`]
//! and [`super::anti_drift::pre_completion_pipeline`]) exist to keep a small
//! model's context clean, but they rewrite the middle of the array every
//! iteration — busting the cache exactly when the context is largest. This is
//! the "cure became the disease" failure documented in
//! [`super::anti_drift::collapse_repetitive_attempts`].
//!
//! [`with_frozen_prefix`] makes those passes prefix-safe *by construction*:
//! the already-sent prefix is physically split off before the pass runs, so the
//! pass cannot see or mutate it — it operates only on the uncached tail. The
//! companion diagnostic [`super::prompt_fingerprint`] classifies the result;
//! under this guard, consecutive mid-turn prompts stay `AppendOnly`.
//!
//! # The head is the worst place to diverge
//!
//! `messages[0]` (the system prompt) is the *first* token block the template
//! renders, so mutating it moves the first-divergent token to offset ~0: the
//! server throws away the entire KV cache and re-prefills from scratch. Measured
//! on a 9267-token prompt in session `20260810_081050_8306f8`: **124s** of dead
//! wait for what should have been a warm-cache append.
//!
//! Six sites in this codebase rewrite `messages[0]` or insert before the last
//! user turn (`prepare_context::append_continuity_to_system`,
//! `agent_core::append_to_system_prompt`, the developer-message rewrite and
//! `insert_tail_before_user` in `context`, and the developer→system fold in
//! `providers::openai_compat`). Each was guarded only by a doc comment telling
//! callers to pass the same bytes every turn. Comments are not enforcement:
//! [`assert_stable_head`] replaces all six comment-contracts with one runtime
//! check that hashes the head per session and screams when it moves.

use std::collections::HashMap;

use parking_lot::Mutex;
use serde_json::Value;

use crate::agent::agent_loop::MessageLog;
use crate::agent::prompt_fingerprint::hash_value;

/// Chars of the mutated head echoed into the warning — enough to identify which
/// of the six mutation sites fired without dumping a 9k-token prompt into logs.
const PREVIEW_CHARS: usize = 200;

/// The frozen-prefix mechanism lives on the log itself
/// ([`MessageLog::edit_tail_from`]); this wrapper exists so the cleanup call
/// site and the tests can name the *contract* (freeze `messages[..watermark]`,
/// edit only the tail) rather than the mechanism. See the module doc and the
/// simulation tests below for why the boundary matters.
///
/// `watermark` is the number of leading messages already sent to the server
/// (hence warm in its prefix cache). It is always a send-time boundary, so no
/// `(tool_call, tool_result)` pair straddles it — tail-only orphan/dangling
/// removal stays correct.
///
/// - `watermark == 0` (cold start, or right after a sanctioned re-prefill such
///   as an over-budget trim or compaction) → `edit` sees the whole array, i.e.
///   the unrestricted cleanup behavior. Those warm bytes were already paid
///   for by the sanctioned reset, so rewriting them costs nothing new.
/// - `watermark >= messages.len()` → the tail is empty; `edit` is a no-op.
pub fn with_frozen_prefix(
    messages: &mut MessageLog,
    watermark: usize,
    edit: impl FnOnce(&mut Vec<Value>),
) {
    messages.edit_tail_from(watermark, edit);
}

/// Enforce that `messages[0]` is byte-stable across every call in a session.
///
/// Hashes the head with [`hash_value`] (the same serialized form that reaches
/// the server, so this sees exactly what the template renders) and compares it
/// against the last hash recorded for `session_key` in `store`. The new hash is
/// always recorded, so a permanently-changed head warns once rather than every
/// turn thereafter.
///
/// Returns `true` when the head is stable — including the first observation of
/// a session and an empty `messages`, which have nothing to diverge from.
///
/// On divergence: emits a `warn!` carrying the session, both hashes, and a
/// [`PREVIEW_CHARS`]-char preview of the new head, then trips a `debug_assert!`
/// so test and CI builds fail loudly while release builds degrade to the
/// warning and return `false`. A caller that gets `false` has already lost the
/// prefix cache for this turn; the value is there so it can be counted, not
/// recovered from.
pub fn assert_stable_head(
    session_key: &str,
    messages: &[Value],
    store: &Mutex<HashMap<String, u64>>,
) -> bool {
    let Some(head) = messages.first() else {
        return true;
    };
    let new_hash = hash_value(head);
    // Lock scope ends with the statement: logging below must not hold it.
    let Some(prev_hash) = store.lock().insert(session_key.to_owned(), new_hash) else {
        return true; // first observation for this session
    };
    if prev_hash == new_hash {
        return true;
    }

    // `chars().take(..)` rather than a byte slice: heads are UTF-8 prose.
    let content_preview: String = head
        .get("content")
        .and_then(Value::as_str)
        .map_or_else(|| head.to_string(), str::to_owned)
        .chars()
        .take(PREVIEW_CHARS)
        .collect();
    tracing::warn!(
        session = %session_key,
        prev_hash,
        new_hash,
        content_preview = %content_preview,
        "prompt_head_changed — messages[0] mutated mid-session; server re-prefills the whole context"
    );
    debug_assert!(
        false,
        "prompt_head_changed — messages[0] mutated mid-session; server re-prefills the \
         whole context (session={session_key}, prev_hash={prev_hash}, new_hash={new_hash}, \
         preview={content_preview})"
    );
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::prompt_fingerprint::{compare, fingerprint, PromptDelta};
    use crate::config::schema::AntiDriftConfig;
    use serde_json::json;

    // --- message builders (house style: module-local json! helpers) ---

    fn system(text: &str) -> Value {
        json!({"role": "system", "content": text})
    }
    fn user(text: &str) -> Value {
        json!({"role": "user", "content": text})
    }
    fn assistant(text: &str) -> Value {
        json!({"role": "assistant", "content": text})
    }
    /// Assistant message carrying a tool call. The `id` (read by
    /// `context_hygiene`) and `function.{name,arguments}` (read by
    /// `anti_drift`) mirror the real wire shape so both passes engage.
    fn assistant_call(id: &str, name: &str, args: &str) -> Value {
        json!({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": id,
                "type": "function",
                "function": {"name": name, "arguments": args},
            }],
        })
    }
    fn tool_result(id: &str, content: &str) -> Value {
        json!({"role": "tool", "tool_call_id": id, "content": content})
    }

    fn drift_config() -> AntiDriftConfig {
        AntiDriftConfig {
            enabled: true,
            anchor_interval: 0, // no tail anchors — keep the simulation deterministic
            pollution_threshold: 0.6,
            babble_max_tokens: 500,
            repetition_min_count: 3,
        }
    }

    // --- Test 1: the frozen prefix is preserved by construction ---

    #[test]
    fn test_with_frozen_prefix_preserves_prefix() {
        let original = vec![
            system("sys"),
            user("u0"),
            assistant("a0"),
            user("u1"),
            assistant("a1"),
        ];

        // 0 < w < len: a destructive edit must not reach the frozen prefix.
        let mut m = MessageLog::committed(original.clone());
        with_frozen_prefix(&mut m, 3, |tail| {
            tail.clear();
            tail.push(assistant("replaced"));
        });
        assert_eq!(
            &m[..3],
            &original[..3],
            "frozen prefix must be byte-identical"
        );
        assert_eq!(m.len(), 4, "tail was replaced with one message");
        assert_eq!(m[3], assistant("replaced"));

        // w == 0: edit sees the whole array (unrestricted, = today's behavior).
        let mut m0 = MessageLog::committed(original.clone());
        with_frozen_prefix(&mut m0, 0, |all| {
            assert_eq!(all.len(), original.len(), "w==0 exposes the whole array");
            all.clear();
        });
        assert!(m0.is_empty());

        // w > len: clamps; tail is empty; array unchanged.
        let mut mover = MessageLog::committed(original.clone());
        with_frozen_prefix(&mut mover, 999, |tail| {
            assert!(tail.is_empty(), "w>len yields an empty tail");
            tail.push(user("should-not-survive-as-prefix-break"));
        });
        assert_eq!(
            &mover[..original.len()],
            &original[..],
            "prefix intact when w clamps"
        );
    }

    // --- Test 2: the append-only invariant (control diverges, treatment holds) ---
    //
    // Simulates the session that triggered this work: a multi-iteration turn
    // that repeatedly issues the SAME web_search (drift → collapse fires) and
    // emits filler assistant turns (pollution → evict fires). Each iteration:
    //   1. run hygiene + anti-drift cleanup
    //   2. "send": fingerprint the array, classify vs. the previous send
    //   3. record the send length as the next watermark
    //   4. "response": append a fresh assistant call + tool result
    //
    // CONTROL (watermark pinned to 0) reproduces today's behavior: cleanup
    // rewrites the middle once enough repetition/pollution accumulates → at
    // least one transition is Diverged (a full re-prefill).
    //
    // TREATMENT (rolling watermark) freezes the sent prefix: cleanup only ever
    // touches the fresh tail → every mid-turn transition is AppendOnly/First.

    fn run_simulated_turn(freeze: bool) -> Vec<PromptDelta> {
        let cfg = drift_config();
        let keep_last = 50; // large, so hygiene truncation never fires in-test
        let mut messages = MessageLog::committed(vec![system("you are nano"), user("what is new about ANE?")]);
        let mut prev = None;
        let mut watermark = 0usize;
        let mut deltas = Vec::new();

        for iter in 0..8u32 {
            let frozen = if freeze { watermark } else { 0 };
            with_frozen_prefix(&mut messages, frozen, |m| {
                crate::agent::context_hygiene::hygiene_pipeline(m, keep_last);
                crate::agent::anti_drift::pre_completion_pipeline(m, iter, &cfg, false);
            });

            let fp = fingerprint(&messages);
            deltas.push(compare(prev.as_ref(), &fp));
            prev = Some(fp);
            watermark = messages.len(); // the "send"

            // "response": identical search call each round → drift loop.
            messages.push_draft(assistant_call(
                "call_search",
                "web_search",
                "{\"q\":\"ANE\"}",
            ));
            messages.push_draft(tool_result("call_search", "stale generic results again"));
            // every other round, a filler (polluted) assistant turn.
            if iter % 2 == 0 {
                messages.push_draft(assistant(
                    "Certainly! Of course, I'd be happy to help. Absolutely, let me think.",
                ));
            }
        }
        deltas
    }

    #[test]
    fn test_append_only_invariant_holds_under_freeze() {
        // Treatment: every transition is cache-safe.
        let treated = run_simulated_turn(true);
        for (i, d) in treated.iter().enumerate() {
            assert!(
                matches!(d, PromptDelta::First | PromptDelta::AppendOnly { .. }),
                "treated iter {i}: expected append-only, got {d:?}",
            );
        }

        // Control: without the freeze, the same workload re-prefills mid-turn.
        let control = run_simulated_turn(false);
        assert!(
            control
                .iter()
                .any(|d| matches!(d, PromptDelta::Diverged { .. })),
            "control must diverge at least once (proving the mechanism bites): {control:?}",
        );
    }

    // --- Test 3: cleanup still does its job at a sanctioned boundary (w==0) ---

    #[test]
    fn test_unfrozen_boundary_still_collapses_drift() {
        let cfg = drift_config();
        // Three identical assistant calls = a genuine drift run.
        let mut messages = MessageLog::committed(vec![
            system("sys"),
            user("go"),
            assistant_call("c1", "web_fetch", "{\"url\":\"x\"}"),
            tool_result("c1", "r1"),
            assistant_call("c2", "web_fetch", "{\"url\":\"x\"}"),
            tool_result("c2", "r2"),
            assistant_call("c3", "web_fetch", "{\"url\":\"x\"}"),
            tool_result("c3", "r3"),
            user("still nothing"),
        ]);
        with_frozen_prefix(&mut messages, 0, |m| {
            crate::agent::anti_drift::pre_completion_pipeline(m, 1, &cfg, false);
        });
        // At w==0 the pass is unrestricted: earlier identical calls collapse,
        // exactly as the direct-call anti_drift tests assert.
        let collapsed = messages
            .iter()
            .filter(|m| {
                m.get("content")
                    .and_then(|c| c.as_str())
                    .is_some_and(|s| s.contains("previous similar attempts removed"))
            })
            .count();
        assert!(
            collapsed >= 1,
            "w==0 must still collapse drift; got {collapsed}"
        );
    }

    // --- Test 4: a tool_call/result pair living in the tail survives ---
    //
    // At a real watermark, the model's assistant(tool_calls) and its results are
    // appended together AFTER the send, so the pair is wholly in the tail. This
    // asserts tail-only orphan removal keeps such a valid pair (it is not seen
    // as orphaned), which is the boundary-safety guarantee the watermark relies
    // on.

    #[test]
    fn test_tool_pair_in_tail_survives_frozen_cleanup() {
        let mut messages = MessageLog::committed(vec![
            system("sys"),
            user("research ANE"),
            assistant("earlier answer"),
            user("dig deeper"), // frozen prefix ends on a clean user-message boundary
            // --- tail (fresh response block) ---
            assistant_call("call_live", "web_fetch", "{\"url\":\"apple.com\"}"),
            tool_result("call_live", "fetched body"),
        ]);
        let watermark = 4; // freeze through the user message; tail = [call, result]

        with_frozen_prefix(&mut messages, watermark, |m| {
            crate::agent::context_hygiene::remove_orphaned_tool_results(m);
        });

        // The valid result must survive — its matching call is in the same tail.
        let kept = messages.iter().any(|m| {
            m.get("role").and_then(|r| r.as_str()) == Some("tool")
                && m.get("content").and_then(|c| c.as_str()) == Some("fetched body")
        });
        assert!(
            kept,
            "tail-resident tool result with an in-tail call must not be dropped"
        );
        assert_eq!(messages.len(), 6, "nothing should be removed");
    }

    // --- Test 5/6: the head-stability guard ---

    /// A prompt whose head is `text`. Only `messages[0]` is under contract, so
    /// the tail here is incidental.
    fn headed(text: &str) -> Vec<Value> {
        vec![system(text), user("turn")]
    }

    #[test]
    fn test_stable_head_holds_across_turns_and_sessions() {
        let store = Mutex::new(HashMap::new());

        // Empty prompt: no head, nothing to diverge from.
        assert!(assert_stable_head("s1", &[], &store));

        // An unchanged head stays stable no matter how the tail grows —
        // append-only tail growth is the whole point of the prefix cache.
        let mut messages = headed("you are nano");
        for turn in 0..3 {
            assert!(
                assert_stable_head("s1", &messages, &store),
                "turn {turn}: unchanged head must be stable"
            );
            messages.push(assistant("appended tail"));
        }

        // A second session carrying a DIFFERENT head must not clobber the first.
        assert!(assert_stable_head(
            "s2",
            &headed("you are someone else"),
            &store
        ));
        assert!(
            assert_stable_head("s1", &messages, &store),
            "sessions must be keyed independently"
        );
    }

    #[test]
    fn test_mutated_head_is_reported() {
        let store = Mutex::new(HashMap::new());
        assert!(assert_stable_head("s1", &headed("you are nano"), &store));

        // Debug/test builds trip the `debug_assert!`; release builds only warn
        // and return `false`. Assert whichever contract the current profile
        // actually promises rather than pinning the test to one of them.
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            assert_stable_head("s1", &headed("you are nano + continuity note"), &store)
        }));
        match outcome {
            Err(payload) => {
                let msg = payload.downcast_ref::<String>().map_or("", String::as_str);
                assert!(
                    msg.contains("prompt_head_changed"),
                    "debug build must trip the guard's debug_assert; got {msg:?}"
                );
            }
            Ok(stable) => assert!(
                !stable,
                "release build must report the mutated head as unstable"
            ),
        }

        // The mutated hash was recorded before reporting, so the guard
        // re-baselines instead of firing on every turn that follows.
        assert!(assert_stable_head(
            "s1",
            &headed("you are nano + continuity note"),
            &store
        ));
    }
}
