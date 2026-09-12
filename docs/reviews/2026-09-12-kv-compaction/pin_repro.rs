use nanobot::agent::lcm::{CompactionFailureMode, LcmConfig, LcmEngine};
use nanobot::agent::token_budget::TokenBudget;
use std::{
    future::Future,
    sync::Arc,
    task::{Context, Poll, Wake, Waker},
};
struct N;
impl Wake for N {
    fn wake(self: Arc<Self>) {}
}
fn run<F: Future>(f: F) -> F::Output {
    let w = Waker::from(Arc::new(N));
    let mut c = Context::from_waker(&w);
    let mut f = std::pin::pin!(f);
    match f.as_mut().poll(&mut c) {
        Poll::Ready(v) => v,
        _ => panic!("unexpected pending"),
    }
}
fn main() {
    for frac in [0.35, 0.0] {
        let mut e = LcmEngine::new(LcmConfig {
            keep_prefix_fraction: frac,
            ..LcmConfig::default()
        });
        for id in 1..=80 {
            let role = if id % 2 == 1 { "user" } else { "assistant" };
            e.ingest(
                format!(
                    r#"{{"role":"{role}","content":"{}","_db_id":{id}}}"#,
                    "word ".repeat(500)
                )
                .parse()
                .unwrap(),
            );
        }
        let b = TokenBudget::new(49152, 4096);
        println!(
            "fraction={frac} before={} first={}",
            e.active_tokens(),
            run(e.compact(None, &b, 0, CompactionFailureMode::Deterministic)).is_some()
        );
        let narrow = TokenBudget::new(4000, 1000);
        for i in 0..8 {
            let ok =
                run(e.compact(None, &narrow, 0, CompactionFailureMode::Deterministic)).is_some();
            println!("narrow fold={i} success={ok} tokens={}", e.active_tokens());
        }
    }
}
