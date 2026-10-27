//! Constrained decoding demo: the same model free, with a JSON mask, and with
//! a JSON Schema mask.
//!
//!   jsonc "Maria is 34 and lives in Aarhus."                 # free vs JSON
//!   jsonc --schema person.json "Maria is 34 and lives in Aarhus."
//!   jsonc --bench                                            # all three over a prompt set

use std::time::Instant;

use anyhow::{Result, bail};
use engine::{
    Engine, Rng, Sampling,
    json::{JsonState, Schema},
};

const SYSTEM: &str = "You extract structured data. Reply with a JSON object only.";

/// (prompt, JSON Schema) pairs. The prompts don't spell out the schema.
const BENCH: &[(&str, &str)] = &[
    (
        "Maria is 34 and lives in Aarhus. Extract name, age and city.",
        r#"{"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}, "city": {"type": "string"}}}"#,
    ),
    (
        "Extract the product, price and currency: 'The new headphones cost 899 kroner.'",
        r#"{"type": "object", "properties": {"product": {"type": "string"}, "price": {"type": "number"}, "currency": {"enum": ["DKK", "EUR", "USD"]}}}"#,
    ),
    (
        "List three primary colors as JSON with a key \"colors\".",
        r#"{"type": "object", "properties": {"colors": {"type": "array", "items": {"type": "string"}}}}"#,
    ),
    (
        "Give me a JSON object describing a cat with fields name, age, indoor (boolean).",
        r#"{"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}, "indoor": {"type": "boolean"}}}"#,
    ),
    (
        "Order #4471: 2x coffee, 1x croissant, paid by card. Return order id, items with quantities, and payment method.",
        r#"{"type": "object", "properties": {"order_id": {"type": "integer"}, "items": {"type": "array", "items": {"type": "object", "properties": {"name": {"type": "string"}, "quantity": {"type": "integer"}}}}, "payment_method": {"enum": ["card", "cash", "mobile"]}}}"#,
    ),
    (
        "Convert to JSON: Title: Dune; Author: Frank Herbert; Year: 1965; Genres: sci-fi, adventure.",
        r#"{"type": "object", "properties": {"title": {"type": "string"}, "author": {"type": "string"}, "year": {"type": "integer"}, "genres": {"type": "array", "items": {"type": "string"}}}}"#,
    ),
    (
        "What's the weather like on Mars? Answer as JSON with keys summary and temperature_c.",
        r#"{"type": "object", "properties": {"summary": {"type": "string"}, "temperature_c": {"type": "number"}}}"#,
    ),
    (
        "Parse this log line into JSON (level, time, message): '2026-10-27 18:04:11 ERROR disk full on /var'",
        r#"{"type": "object", "properties": {"level": {"enum": ["DEBUG", "INFO", "WARN", "ERROR"]}, "time": {"type": "string"}, "message": {"type": "string"}}}"#,
    ),
    (
        "Summarize Rust's ownership model in JSON with keys idea and rules (an array).",
        r#"{"type": "object", "properties": {"idea": {"type": "string"}, "rules": {"type": "array", "items": {"type": "string"}}}}"#,
    ),
    (
        "Return a JSON object mapping the numbers one to five in Danish to their digits.",
        r#"{"type": "object", "properties": {"en": {"type": "integer"}, "to": {"type": "integer"}, "tre": {"type": "integer"}, "fire": {"type": "integer"}, "fem": {"type": "integer"}}}"#,
    ),
];

struct Outcome {
    text: String,
    tokens: usize,
    /// Steps where the model's favourite token was masked out.
    interventions: usize,
    secs: f32,
    parsed: Option<serde_json::Value>,
}

impl Outcome {
    fn matches(&self, schema: &Schema) -> bool {
        self.parsed.as_ref().is_some_and(|v| schema.validate(v))
    }

    fn verdict(&self, schema: &Schema) -> &'static str {
        match (&self.parsed, self.matches(schema)) {
            (_, true) => "ok",
            (Some(_), false) => "wrong shape",
            (None, _) => "INVALID",
        }
    }
}

/// `constraint: None` is free generation.
fn generate(engine: &mut Engine, prompt: &[u32], constraint: Option<&Schema>, s: Sampling, max: usize, seed: u64) -> Result<Outcome> {
    let start = Instant::now();
    let mut rng = Rng::new(seed);
    let mut seq = prompt.to_vec();
    let mut state = JsonState::default();
    let mut interventions = 0;
    for _ in 0..max {
        let logits = engine.logits(&seq)?;
        let mask = constraint.map(|schema| engine.json_mask(schema, &state));
        if let Some(m) = &mask {
            let favourite = engine::top_n(&logits, 1)[0];
            interventions += !m[favourite] as usize;
        }
        let id = engine::sample(&logits, s, mask.as_deref(), &mut rng);
        if engine.is_eos(id) {
            break;
        }
        if let Some(schema) = constraint {
            for &b in engine.token_bytes(id).unwrap_or_default() {
                state.feed(schema, b);
            }
        }
        seq.push(id);
    }
    let generated = &seq[prompt.len()..];
    let text = engine.decode(generated);
    let parsed = serde_json::from_str(&text).ok();
    Ok(Outcome { text, tokens: generated.len(), interventions, secs: start.elapsed().as_secs_f32(), parsed })
}

fn report(label: &str, o: &Outcome, schema: Option<&Schema>) {
    let verdict = match (schema, &o.parsed) {
        (Some(s), _) if o.matches(s) => "\x1b[32mmatches schema\x1b[0m",
        (Some(_), Some(_)) => "\x1b[33mvalid JSON, wrong shape\x1b[0m",
        (None, Some(_)) => "\x1b[32mvalid JSON\x1b[0m",
        (_, None) => "\x1b[31mINVALID\x1b[0m",
    };
    println!(
        "── {label}: {verdict} · {} tokens · {:.1} tok/s · {} interventions",
        o.tokens,
        o.tokens as f32 / o.secs,
        o.interventions
    );
    println!("{}\n", o.text.trim_end());
}

/// A schema given inline (`{...}`) or as a file path.
fn load_schema(arg: &str) -> Result<Schema> {
    let text = if arg.trim_start().starts_with('{') { arg.to_string() } else { std::fs::read_to_string(arg)? };
    Schema::parse(&text)
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let mut sampling = Sampling { temperature: 0.0, ..Sampling::default() };
    let mut max = 256;
    let mut model_dir = engine::DEFAULT_MODEL_DIR.to_string();
    let mut bench = false;
    let mut schema = None;
    let mut prompt = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--temp" => sampling.temperature = args.next().unwrap_or_default().parse()?,
            "--max" => max = args.next().unwrap_or_default().parse()?,
            "--model-dir" => model_dir = args.next().unwrap_or_default(),
            "--schema" => schema = Some(load_schema(&args.next().unwrap_or_default())?),
            "--bench" => bench = true,
            "-h" | "--help" => {
                println!("usage: jsonc [--temp T] [--max N] [--model-dir DIR] (--bench | [--schema FILE|JSON] PROMPT)");
                return Ok(());
            }
            _ => prompt = Some(a),
        }
    }

    let mut engine = Engine::load(&model_dir)?;
    let any_object = Schema::default();
    if bench {
        let n = BENCH.len();
        let mut valid = [0; 3];
        let mut matching = [0; 3];
        println!("{:<4} {:<13} {:<13} {:<13} {}", "#", "free", "json", "schema", "interventions (json/schema)");
        for (i, (p, schema_text)) in BENCH.iter().enumerate() {
            let schema = Schema::parse(schema_text)?;
            let ids = engine.chat_prompt(SYSTEM, p)?;
            let seed = i as u64 + 1;
            let runs = [
                generate(&mut engine, &ids, None, sampling, max, seed)?,
                generate(&mut engine, &ids, Some(&any_object), sampling, max, seed)?,
                generate(&mut engine, &ids, Some(&schema), sampling, max, seed)?,
            ];
            for (k, o) in runs.iter().enumerate() {
                valid[k] += o.parsed.is_some() as usize;
                matching[k] += o.matches(&schema) as usize;
            }
            let [free, json, sch] = &runs;
            println!(
                "{:<4} {:<13} {:<13} {:<13} {}/{}",
                i + 1,
                free.verdict(&schema),
                json.verdict(&schema),
                sch.verdict(&schema),
                json.interventions,
                sch.interventions
            );
        }
        println!("\n{:<8} {:>10} {:>14}", "", "valid JSON", "matches schema");
        for (k, label) in ["free", "json", "schema"].iter().enumerate() {
            println!("{label:<8} {:>7}/{n} {:>11}/{n}", valid[k], matching[k]);
        }
        return Ok(());
    }

    let Some(prompt) = prompt else { bail!("give a prompt or --bench (see --help)") };
    let ids = engine.chat_prompt(SYSTEM, &prompt)?;
    report("free", &generate(&mut engine, &ids, None, sampling, max, 1)?, schema.as_ref());
    report("json", &generate(&mut engine, &ids, Some(&any_object), sampling, max, 1)?, schema.as_ref());
    if let Some(schema) = &schema {
        report("schema", &generate(&mut engine, &ids, Some(schema), sampling, max, 1)?, Some(schema));
    }
    Ok(())
}
