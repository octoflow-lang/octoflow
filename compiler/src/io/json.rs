//! Pure-Rust JSON parser and serializer — zero external dependencies.
//!
//! Supports the JSON subset needed by OctoFlow:
//!   - Objects   `{"key": value, ...}`
//!   - Arrays    `[value, ...]`
//!   - Strings   `"text"` (with `\"` `\\` `\n` `\r` `\t` `\uXXXX` escapes)
//!   - Numbers   integers and decimals (no exponent required)
//!   - Booleans  `true` / `false`
//!   - Null      `null`

use std::collections::HashMap;
use crate::{CliError, Value};

// ── Public JSON value type ────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum JsonVal {
    Null,
    Bool(bool),
    Number(f64),
    Str(String),
    Array(Vec<JsonVal>),
    Object(Vec<(String, JsonVal)>),   // preserve insertion order
}

// ── Parser ────────────────────────────────────────────────────────────

/// Maximum nesting depth for JSON objects/arrays (prevents stack exhaustion).
const MAX_JSON_DEPTH: usize = 100;

struct Parser<'a> {
    src: &'a [u8],
    pos: usize,
    depth: usize,
}

impl<'a> Parser<'a> {
    fn new(s: &'a str) -> Self { Parser { src: s.as_bytes(), pos: 0, depth: 0 } }

    fn peek(&self) -> Option<u8> {
        self.src.get(self.pos).copied()
    }

    fn advance(&mut self) -> Option<u8> {
        let ch = self.src.get(self.pos).copied();
        self.pos += 1;
        ch
    }

    fn skip_ws(&mut self) {
        while let Some(b) = self.peek() {
            if b == b' ' || b == b'\t' || b == b'\n' || b == b'\r' {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn expect(&mut self, ch: u8) -> Result<(), String> {
        self.skip_ws();
        match self.advance() {
            Some(c) if c == ch => Ok(()),
            Some(c) => Err(format!("expected '{}' got '{}'", ch as char, c as char)),
            None => Err(format!("unexpected end, expected '{}'", ch as char)),
        }
    }

    fn parse_value(&mut self) -> Result<JsonVal, String> {
        self.skip_ws();
        match self.peek() {
            Some(b'"') => self.parse_string().map(JsonVal::Str),
            Some(b'{') => self.parse_object(),
            Some(b'[') => self.parse_array(),
            Some(b't') => { self.parse_literal(b"true")?; Ok(JsonVal::Bool(true)) }
            Some(b'f') => { self.parse_literal(b"false")?; Ok(JsonVal::Bool(false)) }
            Some(b'n') => { self.parse_literal(b"null")?; Ok(JsonVal::Null) }
            Some(c) if c == b'-' || c.is_ascii_digit() => self.parse_number(),
            Some(c) => Err(format!("unexpected character '{}'", c as char)),
            None => Err("unexpected end of input".into()),
        }
    }

    fn parse_literal(&mut self, lit: &[u8]) -> Result<(), String> {
        for &expected in lit {
            match self.advance() {
                Some(c) if c == expected => {}
                Some(c) => return Err(format!("expected '{}' got '{}'", expected as char, c as char)),
                None => return Err("unexpected end".into()),
            }
        }
        Ok(())
    }

    fn parse_string(&mut self) -> Result<String, String> {
        self.expect(b'"')?;
        let mut s = String::new();
        loop {
            match self.advance() {
                None => return Err("unterminated string".into()),
                Some(b'"') => break,
                Some(b'\\') => {
                    match self.advance() {
                        Some(b'"')  => s.push('"'),
                        Some(b'\\') => s.push('\\'),
                        Some(b'/')  => s.push('/'),
                        Some(b'n')  => s.push('\n'),
                        Some(b'r')  => s.push('\r'),
                        Some(b't')  => s.push('\t'),
                        Some(b'b')  => s.push('\x08'),
                        Some(b'f')  => s.push('\x0C'),
                        Some(b'u')  => {
                            let mut hex = [0u8; 4];
                            for slot in &mut hex {
                                *slot = self.advance().ok_or("unexpected end in \\u")?;
                            }
                            let hex_str = std::str::from_utf8(&hex)
                                .map_err(|_| "invalid \\u escape")?;
                            let cp = u32::from_str_radix(hex_str, 16)
                                .map_err(|_| format!("invalid \\u{}", hex_str))?;
                            let ch = char::from_u32(cp)
                                .ok_or_else(|| format!("invalid codepoint U+{:04X}", cp))?;
                            s.push(ch);
                        }
                        Some(c) => return Err(format!("invalid escape \\{}", c as char)),
                        None => return Err("unexpected end after backslash".into()),
                    }
                }
                Some(c) => s.push(c as char),
            }
        }
        Ok(s)
    }

    fn parse_number(&mut self) -> Result<JsonVal, String> {
        let start = self.pos;
        if self.peek() == Some(b'-') { self.pos += 1; }
        while matches!(self.peek(), Some(c) if c.is_ascii_digit()) { self.pos += 1; }
        if self.peek() == Some(b'.') {
            self.pos += 1;
            while matches!(self.peek(), Some(c) if c.is_ascii_digit()) { self.pos += 1; }
        }
        // optional exponent
        if matches!(self.peek(), Some(b'e') | Some(b'E')) {
            self.pos += 1;
            if matches!(self.peek(), Some(b'+') | Some(b'-')) { self.pos += 1; }
            while matches!(self.peek(), Some(c) if c.is_ascii_digit()) { self.pos += 1; }
        }
        let slice = std::str::from_utf8(&self.src[start..self.pos])
            .map_err(|_| "invalid number bytes")?;
        let n: f64 = slice.parse().map_err(|_| format!("invalid number '{}'", slice))?;
        Ok(JsonVal::Number(n))
    }

    fn parse_object(&mut self) -> Result<JsonVal, String> {
        self.depth += 1;
        if self.depth > MAX_JSON_DEPTH {
            return Err(format!("JSON nesting depth exceeds maximum of {}", MAX_JSON_DEPTH));
        }
        self.expect(b'{')?;
        let mut pairs = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b'}') { self.pos += 1; self.depth -= 1; return Ok(JsonVal::Object(pairs)); }
        loop {
            self.skip_ws();
            let key = self.parse_string()?;
            self.expect(b':')?;
            let val = self.parse_value()?;
            pairs.push((key, val));
            self.skip_ws();
            match self.peek() {
                Some(b',') => { self.pos += 1; }
                Some(b'}') => { self.pos += 1; break; }
                Some(c) => return Err(format!("expected ',' or '}}', got '{}'", c as char)),
                None => return Err("unexpected end in object".into()),
            }
        }
        self.depth -= 1;
        Ok(JsonVal::Object(pairs))
    }

    fn parse_array(&mut self) -> Result<JsonVal, String> {
        self.depth += 1;
        if self.depth > MAX_JSON_DEPTH {
            return Err(format!("JSON nesting depth exceeds maximum of {}", MAX_JSON_DEPTH));
        }
        self.expect(b'[')?;
        let mut items = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') { self.pos += 1; self.depth -= 1; return Ok(JsonVal::Array(items)); }
        loop {
            let val = self.parse_value()?;
            items.push(val);
            self.skip_ws();
            match self.peek() {
                Some(b',') => { self.pos += 1; }
                Some(b']') => { self.pos += 1; break; }
                Some(c) => return Err(format!("expected ',' or ']', got '{}'", c as char)),
                None => return Err("unexpected end in array".into()),
            }
        }
        self.depth -= 1;
        Ok(JsonVal::Array(items))
    }
}

/// Parse a JSON string into a `JsonVal`.
pub fn parse(s: &str) -> Result<JsonVal, String> {
    let mut p = Parser::new(s);
    let v = p.parse_value()?;
    p.skip_ws();
    if p.pos != p.src.len() {
        return Err(format!("trailing content at position {}", p.pos));
    }
    Ok(v)
}

// ── Serializer ────────────────────────────────────────────────────────

/// Serialize a `JsonVal` to a compact JSON string.
pub fn stringify(v: &JsonVal) -> String {
    match v {
        JsonVal::Null => "null".into(),
        JsonVal::Bool(b) => if *b { "true".into() } else { "false".into() },
        JsonVal::Number(n) => {
            // Emit integer if whole, else decimal
            if n.is_finite() && *n == n.trunc() && n.abs() < 1e15_f64 {
                format!("{}", *n as i64)
            } else {
                format!("{}", n)
            }
        }
        JsonVal::Str(s) => escape_json_string(s),
        JsonVal::Array(items) => {
            let parts: Vec<String> = items.iter().map(stringify).collect();
            format!("[{}]", parts.join(","))
        }
        JsonVal::Object(pairs) => {
            let parts: Vec<String> = pairs.iter()
                .map(|(k, v)| format!("{}:{}", escape_json_string(k), stringify(v)))
                .collect();
            format!("{{{}}}", parts.join(","))
        }
    }
}

fn escape_json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"'  => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\x08' => out.push_str("\\b"),
            '\x0C' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04X}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

// ── OctoFlow Value ↔ JsonVal conversions ─────────────────────────────

/// Convert a top-level `JsonVal` to an OctoFlow `Value`.
pub fn json_to_value(v: &JsonVal) -> Value {
    match v {
        JsonVal::Number(n) => {
            let f = *n;
            if f.fract() == 0.0 && f >= i64::MIN as f64 && f <= i64::MAX as f64 {
                Value::Int(f as i64)
            } else {
                Value::Float(f as f32)
            }
        }
        JsonVal::Str(s)    => Value::Str(s.clone()),
        JsonVal::Bool(b)   => Value::Float(if *b { 1.0 } else { 0.0 }),
        JsonVal::Null      => Value::None,
        other              => Value::Str(stringify(other)),
    }
}

/// Convert an OctoFlow `Value` to a `JsonVal`.
pub fn value_to_json(v: &Value) -> JsonVal {
    match v {
        Value::Float(f) => {
            let n = *f as f64;
            if f.is_finite() && *f == (*f as i64 as f32) {
                JsonVal::Number(n)
            } else if f.is_finite() {
                JsonVal::Number(n)
            } else {
                JsonVal::Null
            }
        }
        Value::Int(i) => JsonVal::Number(*i as f64),
        Value::Str(s) => {
            // If string looks like a JSON array or object, re-parse it
            if (s.starts_with('[') && s.ends_with(']'))
                || (s.starts_with('{') && s.ends_with('}'))
            {
                if let Ok(parsed) = parse(s) {
                    return parsed;
                }
            }
            JsonVal::Str(s.clone())
        }
        Value::Map(map) => {
            let pairs: Vec<(String, JsonVal)> = {
                let mut keys: Vec<&String> = map.keys().collect();
                keys.sort();
                keys.into_iter().map(|k| (k.clone(), value_to_json(&map[k]))).collect()
            };
            JsonVal::Object(pairs)
        }
        Value::None => JsonVal::Null,
    }
}

/// Flatten a JSON object into `HashMap<String, Value>` with dot-notation keys.
pub fn flatten_json(prefix: &str, v: &JsonVal, map: &mut HashMap<String, Value>) {
    match v {
        JsonVal::Object(pairs) => {
            for (key, val) in pairs {
                let full = if prefix.is_empty() {
                    key.clone()
                } else {
                    format!("{}.{}", prefix, key)
                };
                flatten_json(&full, val, map);
            }
        }
        JsonVal::Number(n) => {
            map.insert(prefix.to_string(), Value::Float(*n as f32));
        }
        JsonVal::Str(s) => {
            map.insert(prefix.to_string(), Value::Str(s.clone()));
        }
        JsonVal::Bool(b) => {
            map.insert(prefix.to_string(), Value::Float(if *b { 1.0 } else { 0.0 }));
        }
        JsonVal::Null => {
            map.insert(prefix.to_string(), Value::None);
        }
        JsonVal::Array(_) => {
            // Store arrays as JSON string representation
            map.insert(prefix.to_string(), Value::Str(stringify(v)));
        }
    }
}

/// Unflatten dot-notation `HashMap` back into a nested `JsonVal::Object`.
pub fn unflatten_to_json(map: &HashMap<String, Value>) -> JsonVal {
    let mut pairs: Vec<(String, JsonVal)> = Vec::new();
    let mut sorted_keys: Vec<&String> = map.keys().collect();
    sorted_keys.sort();
    for key in sorted_keys {
        let val = &map[key];
        let parts: Vec<&str> = key.splitn(2, '.').collect();
        if parts.len() == 1 {
            pairs.push((key.clone(), value_to_json(val)));
        } else {
            // Nested key — find or create object entry for parts[0]
            let top = parts[0].to_string();
            let rest = parts[1].to_string();
            // Build a sub-map for this top key
            let sub: HashMap<String, Value> = map.iter()
                .filter_map(|(k, v)| {
                    k.strip_prefix(&format!("{}.", top)).map(|stripped| (stripped.to_string(), v.clone()))
                })
                .collect();
            // Only add if we haven't already added this top key
            if !pairs.iter().any(|(k, _)| k == &top) {
                pairs.push((top, unflatten_to_json(&sub)));
            }
            let _ = rest; // consumed by sub-map recursion above
        }
    }
    JsonVal::Object(pairs)
}

/// Stringify the OctoFlow hashmaps/arrays for json_stringify().
pub fn stringify_map(map: &HashMap<String, Value>) -> Result<String, CliError> {
    Ok(stringify(&unflatten_to_json(map)))
}

pub fn stringify_array(arr: &[Value]) -> Result<String, CliError> {
    let items: Vec<JsonVal> = arr.iter().map(value_to_json).collect();
    Ok(stringify(&JsonVal::Array(items)))
}

/// Parse a JSON string into HashMap (for json_parse).
pub fn parse_object(s: &str) -> Result<HashMap<String, Value>, CliError> {
    let v = parse(s).map_err(|e| CliError::Compile(format!("json_parse(): invalid JSON: {}", e)))?;
    match &v {
        JsonVal::Object(_) => {
            let mut result = HashMap::new();
            flatten_json("", &v, &mut result);
            Ok(result)
        }
        _ => Err(CliError::Compile("json_parse(): JSON value is not an object — use json_parse_array() for arrays".into())),
    }
}

/// Parse a JSON string into Vec<Value> (for json_parse_array).
pub fn parse_array(s: &str) -> Result<Vec<Value>, CliError> {
    let v = parse(s).map_err(|e| CliError::Compile(format!("json_parse_array(): invalid JSON: {}", e)))?;
    match v {
        JsonVal::Array(arr) => Ok(arr.iter().map(json_to_value).collect()),
        _ => Err(CliError::Compile("json_parse_array(): JSON value is not an array".into())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_object() {
        let v = parse(r#"{"name": "alice", "age": 30}"#).unwrap();
        match v {
            JsonVal::Object(pairs) => {
                assert_eq!(pairs.len(), 2);
                assert_eq!(pairs[0].0, "name");
            }
            _ => panic!("expected object"),
        }
    }

    #[test]
    fn test_parse_array() {
        let v = parse("[1, 2, 3]").unwrap();
        match v {
            JsonVal::Array(arr) => assert_eq!(arr.len(), 3),
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn test_parse_nested_within_limits() {
        // 50 levels deep — well within the 100 limit
        let mut s = String::new();
        for _ in 0..50 { s.push('['); }
        s.push('1');
        for _ in 0..50 { s.push(']'); }
        assert!(parse(&s).is_ok());
    }

    #[test]
    fn test_parse_rejects_excessive_nesting() {
        // 101 levels deep — exceeds the 100 limit
        let mut s = String::new();
        for _ in 0..101 { s.push('['); }
        s.push('1');
        for _ in 0..101 { s.push(']'); }
        let result = parse(&s);
        assert!(result.is_err());
        let msg = result.unwrap_err();
        assert!(msg.contains("nesting depth"), "got: {}", msg);
    }

    #[test]
    fn test_parse_rejects_deep_objects() {
        // Deeply nested objects
        let mut s = String::new();
        for i in 0..101 {
            s.push_str(&format!("{{\"k{}\":", i));
        }
        s.push_str("1");
        for _ in 0..101 {
            s.push('}');
        }
        let result = parse(&s);
        assert!(result.is_err());
        let msg = result.unwrap_err();
        assert!(msg.contains("nesting depth"), "got: {}", msg);
    }

    #[test]
    fn test_stringify_roundtrip() {
        let input = r#"{"a": [1, 2], "b": "hello"}"#;
        let v = parse(input).unwrap();
        let output = stringify(&v);
        let v2 = parse(&output).unwrap();
        // Verify structure preserved
        match (v, v2) {
            (JsonVal::Object(a), JsonVal::Object(b)) => assert_eq!(a.len(), b.len()),
            _ => panic!("mismatch"),
        }
    }
}

