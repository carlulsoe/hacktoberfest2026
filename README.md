# Rust AI Hack Night: LLM X-ray + schema-constrained decoding

Qwen2.5-0.5B-Instruct (Q8_0 GGUF) running on CPU through [candle](https://github.com/huggingface/candle). No Python, no API keys.

```
engine/  model loading, rewindable KV cache, sampling, JSON/JSON Schema automaton (engine/src/json.rs)
xray/    TUI: watch the model pick each token, and override it
jsonc/   CLI: the same model free, with a JSON mask, and with a JSON Schema mask
schemas/ example schema
```

## Setup

```sh
mkdir -p models && cd models
curl -LO https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q8_0.gguf
curl -LO https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/tokenizer.json
cd .. && cargo build --release
```

## xray

```sh
./target/release/xray "Write a haiku about the borrow checker."
./target/release/xray --schema schemas/person.json "Describe a fictional Danish programmer."
```

The output is coloured by how confident the model was in each token: green is sure, red is a long shot. Forced tokens are underlined.

| key | action |
| --- | --- |
| space / → | sample one token |
| ↑ ↓, enter | pick a candidate and force it |
| ← / backspace | undo |
| a | autoplay |
| j | JSON mode (follows `--schema` if given): masked tokens are crossed out |
| t/T, k/K, p/P | temperature, top-k, top-p |
| e / r | new prompt / restart |

## jsonc

```sh
./target/release/jsonc "Maria is 34 and lives in Aarhus. Extract name, age and city."
./target/release/jsonc --schema schemas/person.json "Maria is 34, lives in Aarhus and writes Rust."
./target/release/jsonc --bench            # greedy
./target/release/jsonc --bench --temp 0.9
```

Each step, every token in the vocabulary (151k) is checked against the parser state. Tokens that would break the JSON, or stray from the schema, get masked out. The end-of-text token is only allowed once the object is closed. An "intervention" counts a step where the model's favourite token was masked.

Supported schemas are a subset of JSON Schema: `type` (object, array, string, integer, number, boolean), `properties`, `items` and string `enum`. Every property is required, must appear in the order written, and no other keys are allowed. The benchmark scores "matches schema" without regard to key order, so unconstrained output isn't penalized for ordering.

Benchmark on bluefin-1, greedy:

| | valid JSON | matches schema |
| --- | --- | --- |
| free | 3/10 | 2/10 |
| json | 10/10 | 5/10 |
| schema | 10/10 | 10/10 |

The free model mostly fails by wrapping its JSON in a code fence. JSON mode gets the syntax right but not the shape (`"order_id": "4471"`, `items` as an object instead of an array).

### Visual demo

`demo/order.html` is a self-contained replay of a real run: the same prompt decoded free, as any JSON, and against `schemas/order.json`, side by side and token by token. Wherever the mask overrode the model, the token the model wanted is struck out above the one it was allowed to write. Open it in a browser (no model needed). Space plays, ← → step, and hovering a token shows its candidates.

Make one for any prompt and schema:

```sh
./target/release/jsonc --schema schemas/order.json --demo demo/order.html \
  "Order #4471: 2x coffee, 1x croissant, paid by card. Return order id, items with quantities, and payment method."
```

## Tests

```sh
cargo test --release -p engine                 # parser and schema
cargo test --release -p engine -- --ignored    # KV-cache rewind; needs models/
```
