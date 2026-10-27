# Rust AI Hack Night: LLM X-ray + constrained decoding

Qwen2.5-0.5B-Instruct (Q8_0 GGUF) running on CPU through [candle](https://github.com/huggingface/candle). No Python, no API keys.

```
engine/  model loading, KV cache, sampling, JSON pushdown automaton (engine/src/json.rs)
xray/    TUI: watch the model pick each token, and override it
jsonc/   CLI: the same model with and without a JSON grammar mask
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
```

The output is coloured by how confident the model was in each token: green is sure, red is a long shot. Forced tokens are underlined.

| key | action |
| --- | --- |
| space / → | sample one token |
| ↑ ↓, enter | pick a candidate and force it |
| ← / backspace | undo |
| a | autoplay |
| j | JSON mode: masked tokens are crossed out |
| t/T, k/K, p/P | temperature, top-k, top-p |
| e / r | new prompt / restart |

## jsonc

```sh
./target/release/jsonc "Maria is 34 and lives in Aarhus. Extract name, age and city."
./target/release/jsonc --bench            # greedy
./target/release/jsonc --bench --temp 0.9
```

Each step, every token in the vocabulary (151k) is checked against the JSON parser state. Tokens that would break the JSON get masked out, and the end-of-text token is only allowed once the object is closed. An "intervention" counts a step where the model's favourite token was masked.
