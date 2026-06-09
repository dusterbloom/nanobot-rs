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

use serde_json::Value;

/// Run `edit` on only the mutable tail (`messages[watermark..]`), leaving the
/// frozen prefix `messages[..watermark]` byte-identical.
///
/// `watermark` is the number of leading messages already sent to the server
/// (hence warm in its prefix cache). It is always a send-time boundary, so no
/// `(tool_call, tool_result)` pair straddles it — tail-only orphan/dangling
/// removal stays correct.
///
/// - `watermark == 0` (cold start, or right after a sanctioned re-prefill such
///   as an over-budget trim or compaction) → `edit` sees the whole array, i.e.
///   the unrestricted cleanup behavior.
/// - `watermark >= messages.len()` → the tail is empty; `edit` is a no-op on an
///   empty vec and the array is returned unchanged.
///
/// The frozen prefix is preserved by ownership, not by a runtime bound check:
/// `edit` never receives a reference to it.
pub fn with_frozen_prefix(
    messages: &mut Vec<Value>,
    watermark: usize,
    edit: impl FnOnce(&mut Vec<Value>),
) {
    let w = watermark.min(messages.len());
    let mut tail = messages.split_off(w);
    edit(&mut tail);
    messages.append(&mut tail);
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
        let mut m = original.clone();
        with_frozen_prefix(&mut m, 3, |tail| {
            tail.clear();
            tail.push(assistant("replaced"));
        });
        assert_eq!(&m[..3], &original[..3], "frozen prefix must be byte-identical");
        assert_eq!(m.len(), 4, "tail was replaced with one message");
        assert_eq!(m[3], assistant("replaced"));

        // w == 0: edit sees the whole array (unrestricted, = today's behavior).
        let mut m0 = original.clone();
        with_frozen_prefix(&mut m0, 0, |all| {
            assert_eq!(all.len(), original.len(), "w==0 exposes the whole array");
            all.clear();
        });
        assert!(m0.is_empty());

        // w > len: clamps; tail is empty; array unchanged.
        let mut mover = original.clone();
        with_frozen_prefix(&mut mover, 999, |tail| {
            assert!(tail.is_empty(), "w>len yields an empty tail");
            tail.push(user("should-not-survive-as-prefix-break"));
        });
        assert_eq!(&mover[..original.len()], &original[..], "prefix intact when w clamps");
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
        let mut messages = vec![system("you are nano"), user("what is new about ANE?")];
        let mut prev = None;
        let mut watermark = 0usize;
        let mut deltas = Vec::new();

        for iter in 0..8u32 {
            let frozen = if freeze { watermark } else { 0 };
            with_frozen_prefix(&mut messages, frozen, |m| {
                crate::agent::context_hygiene::hygiene_pipeline(m, keep_last);
                crate::agent::anti_drift::pre_completion_pipeline(m, iter, &cfg);
            });

            let fp = fingerprint(&messages, None);
            deltas.push(compare(prev.as_ref(), &fp));
            prev = Some(fp);
            watermark = messages.len(); // the "send"

            // "response": identical search call each round → drift loop.
            messages.push(assistant_call("call_search", "web_search", "{\"q\":\"ANE\"}"));
            messages.push(tool_result("call_search", "stale generic results again"));
            // every other round, a filler (polluted) assistant turn.
            if iter % 2 == 0 {
                messages.push(assistant(
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
        let mut messages = vec![
            system("sys"),
            user("go"),
            assistant_call("c1", "web_fetch", "{\"url\":\"x\"}"),
            tool_result("c1", "r1"),
            assistant_call("c2", "web_fetch", "{\"url\":\"x\"}"),
            tool_result("c2", "r2"),
            assistant_call("c3", "web_fetch", "{\"url\":\"x\"}"),
            tool_result("c3", "r3"),
            user("still nothing"),
        ];
        with_frozen_prefix(&mut messages, 0, |m| {
            crate::agent::anti_drift::pre_completion_pipeline(m, 1, &cfg);
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
        assert!(collapsed >= 1, "w==0 must still collapse drift; got {collapsed}");
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
        let mut messages = vec![
            system("sys"),
            user("research ANE"),
            assistant("earlier answer"),
            user("dig deeper"), // frozen prefix ends on a clean user-message boundary
            // --- tail (fresh response block) ---
            assistant_call("call_live", "web_fetch", "{\"url\":\"apple.com\"}"),
            tool_result("call_live", "fetched body"),
        ];
        let watermark = 4; // freeze through the user message; tail = [call, result]

        with_frozen_prefix(&mut messages, watermark, |m| {
            crate::agent::context_hygiene::remove_orphaned_tool_results(m);
        });

        // The valid result must survive — its matching call is in the same tail.
        let kept = messages.iter().any(|m| {
            m.get("role").and_then(|r| r.as_str()) == Some("tool")
                && m.get("content").and_then(|c| c.as_str()) == Some("fetched body")
        });
        assert!(kept, "tail-resident tool result with an in-tail call must not be dropped");
        assert_eq!(messages.len(), 6, "nothing should be removed");
    }
}
