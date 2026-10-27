//! Rewinding the KV cache must reproduce the logits of the original run.
//! Needs the model files: `cargo test --release -p engine -- --ignored`
//!
//! Comparisons are between runs taking the same incremental path: a one-shot
//! prefill of the same tokens differs by ~0.4 in the logits from quantized
//! matmul batching alone.

use engine::Engine;

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

#[test]
#[ignore = "needs models/ (see README)"]
fn rewind_reproduces_original_logits() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../models");
    let mut engine = Engine::load(dir).unwrap();
    let prompt = engine.chat_prompt("You are terse.", "Count to ten.").unwrap();
    let mut seq = prompt.clone();
    let mut history = vec![];
    for _ in 0..12 {
        let logits = engine.logits(&seq).unwrap();
        seq.push(engine::top_n(&logits, 1)[0] as u32);
        history.push(logits);
    }

    // Undo 5 tokens: same logits as when we were first there.
    let short = &seq[..seq.len() - 5];
    let rewound = engine.logits(short).unwrap();
    assert!(max_diff(&rewound, &history[short.len() - prompt.len()]) < 1e-4);

    // Branch off with the runner-up token; compare with a fresh engine that
    // walks to the same branch step by step.
    let mut branch = short.to_vec();
    branch.push(engine::top_n(&rewound, 2)[1] as u32);
    let branched = engine.logits(&branch).unwrap();
    let mut fresh = Engine::load(dir).unwrap();
    for n in prompt.len()..branch.len() {
        fresh.logits(&branch[..n]).unwrap();
    }
    assert!(max_diff(&branched, &fresh.logits(&branch).unwrap()) < 1e-4);
}
