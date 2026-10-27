//! Byte-level JSON prefix checker for constrained decoding.
//!
//! `JsonState` answers one question: "can these bytes extend what we have so
//! far and still be the start of a valid JSON object?" Masking every token that
//! fails the check makes the model unable to produce invalid JSON.
//!
//! The top level must be an object. The state is `Copy` (the bracket stack is
//! a bitset), so testing the whole vocabulary each step is cheap.

const MAX_DEPTH: u8 = 64;
/// Stops the model from padding forever with whitespace, a common failure
/// mode once everything else is masked.
const MAX_WS_RUN: u8 = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Start,
    /// Expecting a value (after `:`, `,` in an array).
    Value,
    /// Just after `[`: a value or `]`.
    ArrayFirst,
    /// Just after `{`: a key or `}`.
    ObjectFirst,
    /// After `,` in an object: a key.
    Key,
    String { key: bool, esc: Esc },
    /// After a key: `:`.
    Colon,
    /// After a value: `,` or a closer.
    AfterValue,
    Number(Num),
    Literal { word: &'static [u8], at: u8 },
    Done,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Esc {
    None,
    Backslash,
    /// Hex digits still expected after `\u`.
    Hex(u8),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Num {
    Minus,
    Zero,
    Int,
    Dot,
    Frac,
    E,
    ESign,
    Exp,
}

impl Num {
    fn complete(self) -> bool {
        matches!(self, Num::Zero | Num::Int | Num::Frac | Num::Exp)
    }

    fn step(self, b: u8) -> Option<Num> {
        use Num::*;
        Some(match (self, b) {
            (Minus, b'0') => Zero,
            (Minus, b'1'..=b'9') => Int,
            (Int, b'0'..=b'9') => Int,
            (Zero | Int, b'.') => Dot,
            (Dot | Frac, b'0'..=b'9') => Frac,
            (Zero | Int | Frac, b'e' | b'E') => E,
            (E, b'+' | b'-') => ESign,
            (E | ESign | Exp, b'0'..=b'9') => Exp,
            _ => return None,
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct JsonState {
    mode: Mode,
    /// Bit i set = level i is an object, clear = array.
    stack: u64,
    depth: u8,
    ws_run: u8,
}

impl Default for JsonState {
    fn default() -> Self {
        Self { mode: Mode::Start, stack: 0, depth: 0, ws_run: 0 }
    }
}

impl JsonState {
    pub fn is_complete(&self) -> bool {
        self.mode == Mode::Done
    }

    /// Would `bytes` keep the document valid?
    pub fn accepts(&self, bytes: &[u8]) -> bool {
        let mut s = *self;
        bytes.iter().all(|&b| s.feed(b))
    }

    /// Advance by one byte; returns false (and leaves garbage) if invalid.
    pub fn feed(&mut self, b: u8) -> bool {
        let ws = matches!(b, b' ' | b'\t' | b'\n' | b'\r');
        let in_token = matches!(self.mode, Mode::String { .. } | Mode::Number(_) | Mode::Literal { .. });
        if ws && !in_token {
            self.ws_run += 1;
            return self.mode != Mode::Done && self.ws_run <= MAX_WS_RUN;
        }
        self.ws_run = 0;

        match self.mode {
            Mode::Start => b == b'{' && self.push(true),
            Mode::Value => self.value_start(b),
            Mode::ArrayFirst => if b == b']' { self.pop(false) } else { self.value_start(b) },
            Mode::ObjectFirst => match b {
                b'"' => self.go(Mode::String { key: true, esc: Esc::None }),
                b'}' => self.pop(true),
                _ => false,
            },
            Mode::Key => b == b'"' && self.go(Mode::String { key: true, esc: Esc::None }),
            Mode::String { key, esc } => self.string(key, esc, b),
            Mode::Colon => b == b':' && self.go(Mode::Value),
            Mode::AfterValue => self.after_value(b),
            Mode::Number(n) => match n.step(b) {
                Some(n) => self.go(Mode::Number(n)),
                None if n.complete() => {
                    self.mode = Mode::AfterValue;
                    self.feed(b)
                }
                None => false,
            },
            Mode::Literal { word, at } => {
                if word[at as usize] != b {
                    return false;
                }
                if at as usize + 1 == word.len() {
                    self.end_value()
                } else {
                    self.go(Mode::Literal { word, at: at + 1 })
                }
            }
            Mode::Done => false,
        }
    }

    fn go(&mut self, mode: Mode) -> bool {
        self.mode = mode;
        true
    }

    fn value_start(&mut self, b: u8) -> bool {
        match b {
            b'{' => self.push(true),
            b'[' => self.push(false),
            b'"' => self.go(Mode::String { key: false, esc: Esc::None }),
            b'-' => self.go(Mode::Number(Num::Minus)),
            b'0' => self.go(Mode::Number(Num::Zero)),
            b'1'..=b'9' => self.go(Mode::Number(Num::Int)),
            b't' => self.go(Mode::Literal { word: b"true", at: 1 }),
            b'f' => self.go(Mode::Literal { word: b"false", at: 1 }),
            b'n' => self.go(Mode::Literal { word: b"null", at: 1 }),
            _ => false,
        }
    }

    fn string(&mut self, key: bool, esc: Esc, b: u8) -> bool {
        let esc = match esc {
            Esc::None => match b {
                b'"' if key => return self.go(Mode::Colon),
                b'"' => return self.end_value(),
                b'\\' => Esc::Backslash,
                0x00..=0x1f => return false,
                _ => Esc::None,
            },
            Esc::Backslash => match b {
                b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => Esc::None,
                b'u' => Esc::Hex(4),
                _ => return false,
            },
            Esc::Hex(n) if b.is_ascii_hexdigit() => if n == 1 { Esc::None } else { Esc::Hex(n - 1) },
            Esc::Hex(_) => return false,
        };
        self.go(Mode::String { key, esc })
    }

    fn after_value(&mut self, b: u8) -> bool {
        let in_object = self.top_is_object();
        match b {
            b',' if in_object => self.go(Mode::Key),
            b',' => self.go(Mode::Value),
            b'}' if in_object => self.pop(true),
            b']' if !in_object => self.pop(false),
            _ => false,
        }
    }

    fn top_is_object(&self) -> bool {
        self.depth > 0 && self.stack >> (self.depth - 1) & 1 == 1
    }

    fn push(&mut self, object: bool) -> bool {
        if self.depth == MAX_DEPTH {
            return false;
        }
        self.stack = (self.stack & !(1 << self.depth)) | ((object as u64) << self.depth);
        self.depth += 1;
        self.go(if object { Mode::ObjectFirst } else { Mode::ArrayFirst })
    }

    fn pop(&mut self, object: bool) -> bool {
        if self.depth == 0 || self.top_is_object() != object {
            return false;
        }
        self.depth -= 1;
        self.end_value()
    }

    fn end_value(&mut self) -> bool {
        self.go(if self.depth == 0 { Mode::Done } else { Mode::AfterValue })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid(s: &str) -> bool {
        let mut st = JsonState::default();
        s.bytes().all(|b| st.feed(b)) && st.is_complete()
    }

    fn prefix_ok(s: &str) -> bool {
        JsonState::default().accepts(s.as_bytes())
    }

    #[test]
    fn accepts_valid_documents() {
        for s in [
            "{}",
            r#"{"a": 1}"#,
            r#"{"a": [1, -2.5e+3, true, false, null, "x\"é"], "b": {"c": {}}}"#,
            "{\n  \"name\": \"Maria\",\n  \"age\": 34\n}",
            r#"{"n": 0, "m": 0.5, "k": []}"#,
        ] {
            assert!(valid(s), "{s}");
            assert!(serde_json::from_str::<serde_json::Value>(s).is_ok());
        }
    }

    #[test]
    fn rejects_invalid() {
        for s in [
            "```json",
            "[1]",
            r#"{"a": 01}"#,
            r#"{"a": 1,}"#,
            r#"{"a": [1,]}"#,
            r#"{a: 1}"#,
            r#"{"a" 1}"#,
            r#"{"a": tru}x"#,
            r#"{"a": 1]"#,
            r#"{"a": "\x"}"#,
            "{} ",
            "{}{}",
        ] {
            assert!(!valid(s), "{s}");
        }
    }

    #[test]
    fn prefixes() {
        assert!(prefix_ok(r#"{"na"#));
        assert!(prefix_ok(r#"{"a": -"#));
        assert!(prefix_ok(r#"{"a": 12"#));
        assert!(!valid(r#"{"a": -}"#));
        assert!(!valid(r#"{"a": 1.}"#));
        assert!(!prefix_ok(&format!("{{{}", " ".repeat(40))));
    }
}
