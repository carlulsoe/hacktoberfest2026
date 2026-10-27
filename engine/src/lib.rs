//! Shared core: load a quantized Qwen2 GGUF, get next-token logits, sample.
//!
//! Both apps work on the same loop: `logits = engine.logits(&tokens)`,
//! tweak/mask the logits, pick a token, push it, repeat.

pub mod json;

use std::path::Path;

use anyhow::{Context, Result, anyhow};
use candle_core::{Device, Tensor, quantized::gguf_file};
use candle_transformers::models::quantized_qwen2::ModelWeights;
use tokenizers::Tokenizer;

pub const DEFAULT_MODEL_DIR: &str = "models";
const GGUF: &str = "qwen2.5-0.5b-instruct-q8_0.gguf";

pub struct Engine {
    model: ModelWeights,
    tokenizer: Tokenizer,
    device: Device,
    /// Raw bytes of every token id (None for ids the tokenizer doesn't know).
    token_bytes: Vec<Option<Vec<u8>>>,
    /// Special/control tokens like `<|im_end|>`.
    special: Vec<bool>,
    pub eos: Vec<u32>,
    /// Tokens currently held in the KV cache.
    cached: Vec<u32>,
}

impl Engine {
    pub fn load(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let device = Device::Cpu;
        let gguf_path = dir.join(GGUF);
        let mut file = std::fs::File::open(&gguf_path)
            .with_context(|| format!("opening {}", gguf_path.display()))?;
        let content = gguf_file::Content::read(&mut file)?;
        // The embedding table is padded past the tokenizer's vocab; size tables to the logits.
        let n_vocab = content
            .tensor_infos
            .get("token_embd.weight")
            .and_then(|t| t.shape.dims().iter().max().copied())
            .context("token_embd.weight missing from gguf")?;
        let model = ModelWeights::from_gguf(content, &mut file, &device)?;
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| anyhow!(e))?;

        let byte_decoder = gpt2_byte_decoder();
        let mut token_bytes = vec![None; n_vocab];
        let mut special = vec![false; n_vocab];
        for (text, id) in tokenizer.get_vocab(false) {
            let bytes = text.chars().map(|c| byte_decoder.get(&c).copied()).collect();
            token_bytes[id as usize] = bytes;
        }
        for (id, added) in tokenizer.get_added_tokens_decoder() {
            token_bytes[id as usize] = Some(added.content.clone().into_bytes());
            special[id as usize] = added.special;
        }
        let eos = ["<|im_end|>", "<|endoftext|>"]
            .iter()
            .filter_map(|t| tokenizer.token_to_id(t))
            .collect();

        Ok(Self { model, tokenizer, device, token_bytes, special, eos, cached: Vec::new() })
    }

    pub fn n_vocab(&self) -> usize {
        self.token_bytes.len()
    }

    /// Qwen chat template, ending where the assistant's reply starts.
    pub fn chat_prompt(&self, system: &str, user: &str) -> Result<Vec<u32>> {
        let text = format!(
            "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
        );
        let enc = self.tokenizer.encode(text, false).map_err(|e| anyhow!(e))?;
        Ok(enc.get_ids().to_vec())
    }

    /// Logits for the token after `tokens`. Reuses the KV cache when `tokens`
    /// extends what was evaluated last time; otherwise re-runs from scratch
    /// (that's how undo / forcing a different token works).
    pub fn logits(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        let reuse = tokens.len() > self.cached.len() && tokens.starts_with(&self.cached);
        if !reuse {
            self.model.clear_kv_cache();
            self.cached.clear();
        }
        let new = &tokens[self.cached.len()..];
        let input = Tensor::new(new, &self.device)?.unsqueeze(0)?;
        let logits = self.model.forward(&input, self.cached.len())?;
        self.cached = tokens.to_vec();
        Ok(logits.squeeze(0)?.to_vec1::<f32>()?)
    }

    pub fn token_bytes(&self, id: u32) -> Option<&[u8]> {
        self.token_bytes.get(id as usize)?.as_deref()
    }

    pub fn is_special(&self, id: u32) -> bool {
        self.special.get(id as usize).copied().unwrap_or(false)
    }

    pub fn is_eos(&self, id: u32) -> bool {
        self.eos.contains(&id)
    }

    pub fn token_text(&self, id: u32) -> String {
        self.token_bytes(id).map(|b| String::from_utf8_lossy(b).into_owned()).unwrap_or_default()
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let bytes: Vec<u8> = ids
            .iter()
            .filter(|&&id| !self.is_special(id))
            .filter_map(|&id| self.token_bytes(id))
            .flatten()
            .copied()
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Which tokens may follow, given a JSON parser state. EOS is allowed only
    /// once the document is complete.
    pub fn json_mask(&self, state: &json::JsonState) -> Vec<bool> {
        (0..self.n_vocab() as u32)
            .map(|id| {
                if self.is_eos(id) {
                    state.is_complete()
                } else if self.is_special(id) {
                    false
                } else {
                    self.token_bytes(id).is_some_and(|b| !b.is_empty() && state.accepts(b))
                }
            })
            .collect()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Sampling {
    /// 0 means greedy.
    pub temperature: f32,
    /// 0 means off.
    pub top_k: usize,
    /// 1.0 means off.
    pub top_p: f32,
}

impl Default for Sampling {
    fn default() -> Self {
        Self { temperature: 0.8, top_k: 40, top_p: 0.95 }
    }
}

/// Softmax with temperature; masked-out tokens get probability 0.
/// Temperature 0 is treated as 1 here so the display still shows a distribution.
pub fn probs(logits: &[f32], temperature: f32, mask: Option<&[bool]>) -> Vec<f32> {
    let t = if temperature > 0.0 { temperature } else { 1.0 };
    let allowed = |i: usize| mask.is_none_or(|m| m[i]);
    let max = (0..logits.len())
        .filter(|&i| allowed(i))
        .map(|i| logits[i])
        .fold(f32::NEG_INFINITY, f32::max);
    let mut p: Vec<f32> = (0..logits.len())
        .map(|i| if allowed(i) { ((logits[i] - max) / t).exp() } else { 0.0 })
        .collect();
    let sum: f32 = p.iter().sum();
    p.iter_mut().for_each(|x| *x /= sum);
    p
}

/// Indices of the `n` most probable entries, most probable first.
pub fn top_n(p: &[f32], n: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..p.len()).collect();
    let n = n.min(idx.len());
    idx.select_nth_unstable_by(n.saturating_sub(1), |&a, &b| p[b].total_cmp(&p[a]));
    idx.truncate(n);
    idx.sort_by(|&a, &b| p[b].total_cmp(&p[a]));
    idx
}

/// Pick a token from logits, honoring an optional mask.
pub fn sample(logits: &[f32], s: Sampling, mask: Option<&[bool]>, rng: &mut Rng) -> u32 {
    let p = probs(logits, s.temperature, mask);
    if s.temperature <= 0.0 {
        return top_n(&p, 1)[0] as u32;
    }
    let k = if s.top_k == 0 { 1000 } else { s.top_k };
    let mut kept = Vec::new();
    let mut mass = 0.0;
    for i in top_n(&p, k) {
        if p[i] == 0.0 || (mass >= s.top_p && !kept.is_empty()) {
            break;
        }
        mass += p[i];
        kept.push(i);
    }
    let mut r = rng.next_f32() * mass;
    for &i in &kept {
        r -= p[i];
        if r <= 0.0 {
            return i as u32;
        }
    }
    *kept.last().unwrap() as u32
}

/// Tiny xorshift RNG; plenty for sampling.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        // splitmix64 scramble: small seeds would otherwise give near-zero first outputs.
        let mut z = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        Self((z ^ (z >> 31)).max(1))
    }

    pub fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Byte-level BPE vocab entries are bytes mapped to printable chars (GPT-2 scheme).
fn gpt2_byte_decoder() -> std::collections::HashMap<char, u8> {
    let mut bs: Vec<u32> = (b'!' as u32..=b'~' as u32).chain(0xA1..=0xAC).chain(0xAE..=0xFF).collect();
    let mut cs = bs.clone();
    let mut n = 0;
    for b in 0..256 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    cs.into_iter().map(|c| char::from_u32(c).unwrap()).zip(bs.into_iter().map(|b| b as u8)).collect()
}
