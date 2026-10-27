//! Constrained decoding demo: the same model, with and without a JSON mask.
//!
//!   jsonc "Maria is 34 and lives in Aarhus."   # compare one prompt
//!   jsonc --bench                              # valid-JSON rate over a prompt set

use std::time::Instant;

use anyhow::{Result, bail};
use engine::{Engine, Rng, Sampling, json::JsonState};

const SYSTEM: &str = "You extract structured data. Reply with a JSON object only.";

const BENCH: &[&str] = &[
    "Maria is 34 and lives in Aarhus. Extract name, age and city.",
    "Extract the product, price and currency: 'The new headphones cost 899 kroner.'",
    "List three primary colors as JSON with a key \"colors\".",
    "Give me a JSON object describing a cat with fields name, age, indoor (boolean).",
    "Order #4471: 2x coffee, 1x croissant, paid by card. Return order id, items with quantities, and payment method.",
    "Convert to JSON: Title: Dune; Author: Frank Herbert; Year: 1965; Genres: sci-fi, adventure.",
    "What's the weather like on Mars? Answer as JSON with keys summary and temperature_c.",
    "Parse this log line into JSON (level, time, message): '2026-10-27 18:04:11 ERROR disk full on /var'",
    "Summarize Rust's ownership model in JSON with keys idea and rules (an array).",
    "Return a JSON object mapping the numbers one to five in Danish to their digits.",
];

struct Outcome {
    text: String,
    tokens: usize,
    /// Steps where the model's favourite token was masked out.
    interventions: usize,
    secs: f32,
    valid: bool,
}

fn generate(engine: &mut Engine, prompt: &[u32], constrained: bool, s: Sampling, max: usize, seed: u64) -> Result<Outcome> {
    let start = Instant::now();
    let mut rng = Rng::new(seed);
    let mut seq = prompt.to_vec();
    let mut state = JsonState::default();
    let mut interventions = 0;
    for _ in 0..max {
        let logits = engine.logits(&seq)?;
        let mask = constrained.then(|| engine.json_mask(&state));
        if let Some(m) = &mask {
            let favourite = engine::top_n(&logits, 1)[0];
            interventions += !m[favourite] as usize;
        }
        let id = engine::sample(&logits, s, mask.as_deref(), &mut rng);
        if engine.is_eos(id) {
            break;
        }
        if constrained {
            for &b in engine.token_bytes(id).unwrap_or_default() {
                state.feed(b);
            }
        }
        seq.push(id);
    }
    let generated = &seq[prompt.len()..];
    let text = engine.decode(generated);
    let valid = serde_json::from_str::<serde_json::Value>(&text).is_ok();
    Ok(Outcome { text, tokens: generated.len(), interventions, secs: start.elapsed().as_secs_f32(), valid })
}

fn report(label: &str, o: &Outcome) {
    let verdict = if o.valid { "\x1b[32mvalid JSON\x1b[0m" } else { "\x1b[31mINVALID\x1b[0m" };
    println!(
        "── {label}: {verdict} · {} tokens · {:.1} tok/s · {} interventions",
        o.tokens,
        o.tokens as f32 / o.secs,
        o.interventions
    );
    println!("{}\n", o.text.trim_end());
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let mut sampling = Sampling { temperature: 0.0, ..Sampling::default() };
    let mut max = 256;
    let mut model_dir = engine::DEFAULT_MODEL_DIR.to_string();
    let mut bench = false;
    let mut prompt = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--temp" => sampling.temperature = args.next().unwrap_or_default().parse()?,
            "--max" => max = args.next().unwrap_or_default().parse()?,
            "--model-dir" => model_dir = args.next().unwrap_or_default(),
            "--bench" => bench = true,
            "-h" | "--help" => {
                println!("usage: jsonc [--temp T] [--max N] [--model-dir DIR] (--bench | PROMPT)");
                return Ok(());
            }
            _ => prompt = Some(a),
        }
    }

    let mut engine = Engine::load(&model_dir)?;
    if bench {
        let (mut free_ok, mut json_ok, mut interventions) = (0, 0, 0);
        println!("{:<4} {:<11} {:<11} {}", "#", "free", "constrained", "interventions");
        for (i, p) in BENCH.iter().enumerate() {
            let ids = engine.chat_prompt(SYSTEM, p)?;
            let free = generate(&mut engine, &ids, false, sampling, max, i as u64 + 1)?;
            let json = generate(&mut engine, &ids, true, sampling, max, i as u64 + 1)?;
            free_ok += free.valid as usize;
            json_ok += json.valid as usize;
            interventions += json.interventions;
            let mark = |v: bool| if v { "ok" } else { "INVALID" };
            println!("{:<4} {:<11} {:<11} {}", i + 1, mark(free.valid), mark(json.valid), json.interventions);
        }
        let n = BENCH.len();
        println!("\nvalid JSON: free {free_ok}/{n}, constrained {json_ok}/{n} ({interventions} interventions total)");
        return Ok(());
    }

    let Some(prompt) = prompt else { bail!("give a prompt or --bench (see --help)") };
    let ids = engine.chat_prompt(SYSTEM, &prompt)?;
    report("free", &generate(&mut engine, &ids, false, sampling, max, 1)?);
    report("constrained", &generate(&mut engine, &ids, true, sampling, max, 1)?);
    Ok(())
}
