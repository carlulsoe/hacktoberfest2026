//! Byte-level JSON prefix checker for constrained decoding, optionally
//! following a schema.
//!
//! `JsonState` answers one question: "can these bytes extend what we have so
//! far and still be the start of a JSON object matching the schema?" Masking
//! every token that fails the check makes the model unable to produce anything
//! else.
//!
//! The state is `Copy` (a fixed-size stack of schema positions), so testing
//! the whole vocabulary each step is cheap.

use anyhow::{Context, Result, bail};
use serde_json::Value;

const MAX_DEPTH: usize = 32;
/// Whitespace is the one escape hatch left when the model's favourite token
/// is masked, so it is kept tidy: spaces plus at most one newline per run,
/// and not too many of them.
const MAX_WS_RUN: u8 = 16;

type NodeId = u16;
const ANY: NodeId = 0;
const ANY_OBJECT: NodeId = 1;
const ANY_ARRAY: NodeId = 2;

#[derive(Debug)]
enum Node {
    Any,
    AnyObject,
    /// Exactly these keys, in this order.
    Object(Vec<(String, NodeId)>),
    Array(NodeId),
    /// Allowed values; empty means any string.
    String(Vec<String>),
    Integer,
    Number,
    Boolean,
}

#[derive(Debug)]
pub struct Schema {
    nodes: Vec<Node>,
    root: NodeId,
}

impl Default for Schema {
    /// Any JSON object.
    fn default() -> Self {
        Self { nodes: vec![Node::Any, Node::AnyObject, Node::Array(ANY)], root: ANY_OBJECT }
    }
}

impl Schema {
    /// A subset of JSON Schema: `type` (object, array, string, integer,
    /// number, boolean), `properties`, `items` and string `enum`. Every
    /// property is required, in the order written, and no others are allowed.
    pub fn from_json_schema(v: &Value) -> Result<Self> {
        let mut s = Self::default();
        s.root = s.add(v)?;
        if !matches!(s.nodes[s.root as usize], Node::Object(_) | Node::AnyObject) {
            bail!("the top-level schema must be an object");
        }
        Ok(s)
    }

    pub fn parse(text: &str) -> Result<Self> {
        Self::from_json_schema(&serde_json::from_str(text).context("schema is not valid JSON")?)
    }

    fn add(&mut self, v: &Value) -> Result<NodeId> {
        let node = if let Some(values) = v.get("enum") {
            let values = values
                .as_array()
                .context("enum must be an array")?
                .iter()
                .map(|x| x.as_str().map(str::to_owned).context("only string enums are supported"))
                .collect::<Result<Vec<_>>>()?;
            if values.is_empty() || values.len() > 32 {
                bail!("enum needs 1 to 32 values");
            }
            values.iter().try_for_each(|s| check_plain(s))?;
            Node::String(values)
        } else {
            match v.get("type").and_then(Value::as_str) {
                None => return Ok(ANY),
                Some("object") => match v.get("properties").and_then(Value::as_object) {
                    None => return Ok(ANY_OBJECT),
                    Some(props) => {
                        let mut fields = Vec::new();
                        for (key, sub) in props {
                            check_plain(key)?;
                            fields.push((key.clone(), self.add(sub)?));
                        }
                        Node::Object(fields)
                    }
                },
                Some("array") => match v.get("items") {
                    None => return Ok(ANY_ARRAY),
                    Some(items) => Node::Array(self.add(items)?),
                },
                Some("string") => Node::String(vec![]),
                Some("integer") => Node::Integer,
                Some("number") => Node::Number,
                Some("boolean") => Node::Boolean,
                Some(t) => bail!("unsupported type {t:?}"),
            }
        };
        self.nodes.push(node);
        Ok((self.nodes.len() - 1) as NodeId)
    }

    /// Does a parsed document match? Key order doesn't matter here, so this is
    /// a fair check for unconstrained output too.
    pub fn validate(&self, v: &Value) -> bool {
        self.check(self.root, v)
    }

    fn check(&self, id: NodeId, v: &Value) -> bool {
        match &self.nodes[id as usize] {
            Node::Any => true,
            Node::AnyObject => v.is_object(),
            Node::Object(fields) => v.as_object().is_some_and(|o| {
                o.len() == fields.len() && fields.iter().all(|(k, n)| o.get(k).is_some_and(|x| self.check(*n, x)))
            }),
            Node::Array(items) => v.as_array().is_some_and(|a| a.iter().all(|x| self.check(*items, x))),
            Node::String(e) => v.as_str().is_some_and(|s| e.is_empty() || e.iter().any(|x| x == s)),
            Node::Integer => v.is_i64() || v.is_u64(),
            Node::Number => v.is_number(),
            Node::Boolean => v.is_boolean(),
        }
    }

    fn node(&self, id: NodeId) -> &Node {
        &self.nodes[id as usize]
    }
}

/// Fixed strings are matched byte by byte, so keep them escape-free.
fn check_plain(s: &str) -> Result<()> {
    if s.bytes().any(|b| b < 0x20 || b == b'"' || b == b'\\') {
        bail!("{s:?}: keys and enum values must not need escaping");
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Start,
    /// Expecting a value of this schema node.
    Value(NodeId),
    /// Just after `[`: a value or `]`.
    ArrayFirst,
    /// Just after `{`: a key or `}`.
    ObjectFirst,
    /// After `,` in an object: a key.
    Key,
    /// `at`/`cand` track progress through fixed keys and enum values:
    /// `cand` has a bit per still-possible value.
    Str { key: bool, node: NodeId, esc: Esc, at: u16, cand: u32 },
    /// After a key: `:`.
    Colon,
    /// After a value: `,` or a closer.
    AfterValue,
    Number { num: Num, int: bool },
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

    fn step(self, b: u8, int: bool) -> Option<Num> {
        use Num::*;
        let next = match (self, b) {
            (Minus, b'0') => Zero,
            (Minus, b'1'..=b'9') => Int,
            (Int, b'0'..=b'9') => Int,
            (Zero | Int, b'.') => Dot,
            (Dot | Frac, b'0'..=b'9') => Frac,
            (Zero | Int | Frac, b'e' | b'E') => E,
            (E, b'+' | b'-') => ESign,
            (E | ESign | Exp, b'0'..=b'9') => Exp,
            _ => return None,
        };
        (!int || matches!(next, Zero | Int)).then_some(next)
    }
}

/// One open object/array: its schema node and, for objects, the current key.
#[derive(Clone, Copy, Debug, Default)]
struct Frame {
    node: NodeId,
    field: u16,
}

#[derive(Clone, Copy, Debug)]
pub struct JsonState {
    mode: Mode,
    stack: [Frame; MAX_DEPTH],
    depth: u8,
    ws_run: u8,
    ws_newline: bool,
}

impl Default for JsonState {
    fn default() -> Self {
        Self { mode: Mode::Start, stack: [Frame::default(); MAX_DEPTH], depth: 0, ws_run: 0, ws_newline: false }
    }
}

impl JsonState {
    pub fn is_complete(&self) -> bool {
        self.mode == Mode::Done
    }

    /// Would `bytes` keep the document valid?
    pub fn accepts(&self, schema: &Schema, bytes: &[u8]) -> bool {
        let mut s = *self;
        bytes.iter().all(|&b| s.feed(schema, b))
    }

    /// Advance by one byte; returns false (and leaves garbage) if invalid.
    pub fn feed(&mut self, schema: &Schema, b: u8) -> bool {
        let ws = matches!(b, b' ' | b'\t' | b'\n' | b'\r');
        let in_token = matches!(self.mode, Mode::Str { .. } | Mode::Number { .. } | Mode::Literal { .. });
        if ws && !in_token {
            let newline = b == b'\n';
            let ok = matches!(b, b' ' | b'\n') && !(newline && self.ws_newline);
            self.ws_run += 1;
            self.ws_newline |= newline;
            return ok && self.mode != Mode::Done && self.ws_run <= MAX_WS_RUN;
        }
        self.ws_run = 0;
        self.ws_newline = false;

        match self.mode {
            Mode::Start => b == b'{' && self.push(schema, schema.root),
            Mode::Value(node) => self.value_start(schema, node, b),
            Mode::ArrayFirst => match (b, schema.node(self.top().node)) {
                (b']', _) => self.pop(),
                (_, Node::Array(items)) => self.value_start(schema, *items, b),
                _ => false,
            },
            Mode::ObjectFirst => {
                let (keys, close) = match schema.node(self.top().node) {
                    Node::Object(fields) => (!fields.is_empty(), fields.is_empty()),
                    _ => (true, true),
                };
                match b {
                    b'"' if keys => self.start_key(),
                    b'}' if close => self.pop(),
                    _ => false,
                }
            }
            Mode::Key => b == b'"' && self.start_key(),
            Mode::Str { key, node, esc, at, cand } => self.string(schema, key, node, esc, at, cand, b),
            Mode::Colon => {
                let top = self.top();
                let node = match schema.node(top.node) {
                    Node::Object(fields) => fields[top.field as usize].1,
                    _ => ANY,
                };
                b == b':' && self.go(Mode::Value(node))
            }
            Mode::AfterValue => self.after_value(schema, b),
            Mode::Number { num, int } => match num.step(b, int) {
                Some(num) => self.go(Mode::Number { num, int }),
                None if num.complete() => {
                    self.mode = Mode::AfterValue;
                    self.feed(schema, b)
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

    fn top(&self) -> Frame {
        self.stack[self.depth as usize - 1]
    }

    fn start_key(&mut self) -> bool {
        self.go(Mode::Str { key: true, node: ANY, esc: Esc::None, at: 0, cand: 1 })
    }

    fn value_start(&mut self, schema: &Schema, node: NodeId, b: u8) -> bool {
        let number = |int| {
            let num = match b {
                b'-' => Num::Minus,
                b'0' => Num::Zero,
                _ => Num::Int,
            };
            Some(Mode::Number { num, int })
        };
        let literal = |word: &'static [u8]| Mode::Literal { word, at: 1 };
        let mode = match (schema.node(node), b) {
            (Node::Any, b'{') => return self.push(schema, ANY_OBJECT),
            (Node::Any, b'[') => return self.push(schema, ANY_ARRAY),
            (Node::AnyObject | Node::Object(_), b'{') | (Node::Array(_), b'[') => return self.push(schema, node),
            (Node::Any | Node::String(_), b'"') => {
                let all = match schema.node(node) {
                    Node::String(e) if !e.is_empty() => u32::MAX >> (32 - e.len()),
                    _ => 0,
                };
                Some(Mode::Str { key: false, node, esc: Esc::None, at: 0, cand: all })
            }
            (Node::Any | Node::Number, b'-' | b'0'..=b'9') => number(false),
            (Node::Integer, b'-' | b'0'..=b'9') => number(true),
            (Node::Any | Node::Boolean, b't') => Some(literal(b"true")),
            (Node::Any | Node::Boolean, b'f') => Some(literal(b"false")),
            (Node::Any, b'n') => Some(literal(b"null")),
            _ => None,
        };
        mode.is_some_and(|m| self.go(m))
    }

    /// The strings this string must be one of, if it is fixed by the schema.
    fn fixed<'s>(&self, schema: &'s Schema, key: bool, node: NodeId) -> Option<&'s [String]> {
        if key {
            let top = self.top();
            match schema.node(top.node) {
                Node::Object(fields) => Some(std::slice::from_ref(&fields[top.field as usize].0)),
                _ => None,
            }
        } else {
            match schema.node(node) {
                Node::String(e) if !e.is_empty() => Some(e),
                _ => None,
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn string(&mut self, schema: &Schema, key: bool, node: NodeId, esc: Esc, at: u16, cand: u32, b: u8) -> bool {
        if let Some(options) = self.fixed(schema, key, node) {
            let live = |i: usize| cand >> i & 1 == 1;
            if b == b'"' {
                return (0..options.len()).any(|i| live(i) && options[i].len() == at as usize) && self.end_string(key);
            }
            let cand = (0..options.len())
                .filter(|&i| live(i) && options[i].as_bytes().get(at as usize) == Some(&b))
                .fold(0, |m, i| m | 1 << i);
            return cand != 0 && self.go(Mode::Str { key, node, esc, at: at + 1, cand });
        }
        let esc = match esc {
            Esc::None => match b {
                b'"' => return self.end_string(key),
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
        self.go(Mode::Str { key, node, esc, at, cand })
    }

    fn end_string(&mut self, key: bool) -> bool {
        if key { self.go(Mode::Colon) } else { self.end_value() }
    }

    fn after_value(&mut self, schema: &Schema, b: u8) -> bool {
        let top = self.top();
        match (schema.node(top.node), b) {
            (Node::AnyObject, b',') => self.go(Mode::Key),
            (Node::AnyObject, b'}') => self.pop(),
            (Node::Object(fields), b',') if (top.field as usize + 1) < fields.len() => {
                self.stack[self.depth as usize - 1].field += 1;
                self.go(Mode::Key)
            }
            (Node::Object(fields), b'}') if top.field as usize + 1 == fields.len() => self.pop(),
            (Node::Array(items), b',') => self.go(Mode::Value(*items)),
            (Node::Array(_), b']') => self.pop(),
            _ => false,
        }
    }

    fn push(&mut self, schema: &Schema, node: NodeId) -> bool {
        if self.depth as usize == MAX_DEPTH {
            return false;
        }
        self.stack[self.depth as usize] = Frame { node, field: 0 };
        self.depth += 1;
        let array = matches!(schema.node(node), Node::Array(_));
        self.go(if array { Mode::ArrayFirst } else { Mode::ObjectFirst })
    }

    /// Close the innermost container; the caller has checked the bracket kind.
    fn pop(&mut self) -> bool {
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

    fn valid_with(schema: &Schema, s: &str) -> bool {
        let mut st = JsonState::default();
        s.bytes().all(|b| st.feed(schema, b)) && st.is_complete()
    }

    fn valid(s: &str) -> bool {
        valid_with(&Schema::default(), s)
    }

    fn prefix_ok(s: &str) -> bool {
        JsonState::default().accepts(&Schema::default(), s.as_bytes())
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
            assert!(serde_json::from_str::<Value>(s).is_ok());
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
            r#"{"a": [1}"#,
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
        assert!(!prefix_ok("{\n\n"));
        assert!(!prefix_ok("{\t"));
        assert!(prefix_ok("{\n  \"a\": [\n    1,\n    2\n  ]\n}"));
    }

    fn person() -> Schema {
        Schema::parse(
            r#"{"type": "object", "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "level": {"enum": ["INFO", "ERROR"]},
                "ok": {"type": "boolean"},
                "meta": {"type": "object"}
            }}"#,
        )
        .unwrap()
    }

    const PERSON: &str =
        r#"{"name": "Ann", "age": 3, "tags": ["a", "b"], "level": "INFO", "ok": true, "meta": {"k": [1]}}"#;

    #[test]
    fn schema_accepts_matching_document() {
        let s = person();
        assert!(valid_with(&s, PERSON));
        assert!(valid_with(&s, &PERSON.replace(r#"["a", "b"]"#, "[]").replace("INFO", "ERROR")));
        assert!(s.validate(&serde_json::from_str(PERSON).unwrap()));
    }

    #[test]
    fn schema_rejects_mismatches() {
        let s = person();
        for (from, to) in [
            (r#""name": "Ann", "age": 3"#, r#""age": 3, "name": "Ann""#), // order
            (r#""age": 3"#, r#""age": 3.5"#),
            (r#""age": 3"#, r#""age": "3""#),
            (r#""age""#, r#""Age""#),
            (r#""INFO""#, r#""WARN""#),
            (r#""INFO""#, r#""INF""#),
            (r#"["a", "b"]"#, "[1]"),
            (r#""ok": true"#, r#""ok": null"#),
            (r#", "meta": {"k": [1]}"#, ""), // missing key
            (r#""meta": {"k": [1]}"#, r#""meta": {}, "x": 1"#), // extra key
        ] {
            let doc = PERSON.replace(from, to);
            assert_ne!(doc, PERSON);
            assert!(!valid_with(&s, &doc), "{doc}");
        }
        assert!(!valid_with(&s, "{}"));
        assert!(JsonState::default().accepts(&s, br#"{"name": "Ann", "age": 3, "tags": [], "level": "IN"#));
    }

    #[test]
    fn validate_ignores_key_order() {
        let s = person();
        let reordered = r#"{"age": 3, "name": "Ann", "tags": [], "level": "INFO", "ok": false, "meta": {}}"#;
        assert!(s.validate(&serde_json::from_str(reordered).unwrap()));
        let extra = PERSON.replace(r#""ok": true"#, r#""ok": true, "x": 1"#);
        assert!(!s.validate(&serde_json::from_str(&extra).unwrap()));
    }

    #[test]
    fn schema_errors() {
        assert!(Schema::parse(r#"{"type": "array"}"#).is_err());
        assert!(Schema::parse(r#"{"type": "object", "properties": {"a": {"type": "date"}}}"#).is_err());
        assert!(Schema::parse(r#"{"type": "object", "properties": {"a\"b": {}}}"#).is_err());
    }
}
