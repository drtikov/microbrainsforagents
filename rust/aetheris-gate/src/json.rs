//! Minimal RFC 8259 JSON parser and serializer using pure Rust std.

use std::fmt::Write as _;

#[derive(Debug, Clone, PartialEq)]
pub enum JsonValue {
    Null,
    Bool(bool),
    Number(f64),
    Str(String),
    Array(Vec<JsonValue>),
    Object(Vec<(String, JsonValue)>),
}

impl JsonValue {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            JsonValue::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            JsonValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            JsonValue::Number(n) if n.is_finite() => Some(*n as i64),
            _ => None,
        }
    }

    pub fn as_object(&self) -> Option<&[(String, JsonValue)]> {
        match self {
            JsonValue::Object(entries) => Some(entries.as_slice()),
            _ => None,
        }
    }

    pub fn get(&self, key: &str) -> Option<&JsonValue> {
        match self {
            JsonValue::Object(entries) => {
                for (k, v) in entries.iter().rev() {
                    if k == key {
                        return Some(v);
                    }
                }
                None
            }
            _ => None,
        }
    }

    pub fn is_null(&self) -> bool {
        matches!(self, JsonValue::Null)
    }
}

pub fn obj(entries: Vec<(&str, JsonValue)>) -> JsonValue {
    JsonValue::Object(
        entries
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect(),
    )
}

pub fn to_json_string(val: &JsonValue) -> String {
    let mut out = String::with_capacity(128);
    write_json(val, &mut out);
    out
}

fn write_json(val: &JsonValue, out: &mut String) {
    match val {
        JsonValue::Null => out.push_str("null"),
        JsonValue::Bool(true) => out.push_str("true"),
        JsonValue::Bool(false) => out.push_str("false"),
        JsonValue::Number(n) => {
            if !n.is_finite() {
                out.push_str("null");
            } else if n.fract() == 0.0 && *n >= (i64::MIN as f64) && *n <= (i64::MAX as f64) {
                let _ = write!(out, "{}", *n as i64);
            } else {
                let _ = write!(out, "{n}");
            }
        }
        JsonValue::Str(s) => write_escaped_str(s, out),
        JsonValue::Array(items) => {
            out.push('[');
            for (idx, item) in items.iter().enumerate() {
                if idx > 0 {
                    out.push(',');
                }
                write_json(item, out);
            }
            out.push(']');
        }
        JsonValue::Object(entries) => {
            out.push('{');
            for (idx, (k, v)) in entries.iter().enumerate() {
                if idx > 0 {
                    out.push(',');
                }
                write_escaped_str(k, out);
                out.push(':');
                write_json(v, out);
            }
            out.push('}');
        }
    }
}

fn write_escaped_str(s: &str, out: &mut String) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\x08' => out.push_str("\\b"),
            '\x0c' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn skip_ws(&mut self) {
        while self.pos < self.bytes.len() && matches!(self.bytes[self.pos], b' ' | b'\t' | b'\r' | b'\n') {
            self.pos += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn parse_value(&mut self) -> Result<JsonValue, String> {
        self.skip_ws();
        match self.peek() {
            Some(b'n') => self.parse_lit(b"null", JsonValue::Null),
            Some(b't') => self.parse_lit(b"true", JsonValue::Bool(true)),
            Some(b'f') => self.parse_lit(b"false", JsonValue::Bool(false)),
            Some(b'"') => self.parse_string().map(JsonValue::Str),
            Some(b'[') => self.parse_array(),
            Some(b'{') => self.parse_object(),
            Some(b'-' | b'0'..=b'9') => self.parse_number(),
            Some(b) => Err(format!("unexpected byte 0x{b:02x} at pos {}", self.pos)),
            None => Err("unexpected EOF".to_string()),
        }
    }

    fn parse_lit(&mut self, lit: &[u8], val: JsonValue) -> Result<JsonValue, String> {
        if self.bytes[self.pos..].starts_with(lit) {
            self.pos += lit.len();
            Ok(val)
        } else {
            Err(format!("invalid literal at pos {}", self.pos))
        }
    }

    fn parse_hex4(&mut self) -> Result<u16, String> {
        if self.pos + 4 > self.bytes.len() {
            return Err("truncated unicode escape".to_string());
        }
        let s = std::str::from_utf8(&self.bytes[self.pos..self.pos + 4])
            .map_err(|_| "invalid utf8 in escape".to_string())?;
        let code = u16::from_str_radix(s, 16).map_err(|_| "invalid hex escape".to_string())?;
        self.pos += 4;
        Ok(code)
    }

    fn parse_string(&mut self) -> Result<String, String> {
        if self.peek() != Some(b'"') {
            return Err("expected '\"'".to_string());
        }
        self.pos += 1;
        let mut out = String::new();
        while self.pos < self.bytes.len() {
            let b = self.bytes[self.pos];
            if b == b'"' {
                self.pos += 1;
                return Ok(out);
            }
            if b == b'\\' {
                self.pos += 1;
                if self.pos >= self.bytes.len() {
                    return Err("unterminated escape".to_string());
                }
                let esc = self.bytes[self.pos];
                self.pos += 1;
                match esc {
                    b'"' => out.push('"'),
                    b'\\' => out.push('\\'),
                    b'/' => out.push('/'),
                    b'b' => out.push('\x08'),
                    b'f' => out.push('\x0c'),
                    b'n' => out.push('\n'),
                    b'r' => out.push('\r'),
                    b't' => out.push('\t'),
                    b'u' => {
                        let u1 = self.parse_hex4()?;
                        if (0xD800..=0xDBFF).contains(&u1) {
                            if self.pos + 2 <= self.bytes.len() && self.bytes[self.pos..self.pos + 2] == *b"\\u" {
                                self.pos += 2;
                                let u2 = self.parse_hex4()?;
                                if (0xDC00..=0xDFFF).contains(&u2) {
                                    let cp = 0x10000 + (((u1 as u32 - 0xD800) << 10) | (u2 as u32 - 0xDC00));
                                    if let Some(ch) = char::from_u32(cp) {
                                        out.push(ch);
                                        continue;
                                    }
                                }
                            }
                            return Err("invalid surrogate pair".to_string());
                        }
                        if let Some(ch) = char::from_u32(u1 as u32) {
                            out.push(ch);
                        } else {
                            return Err("invalid unicode codepoint".to_string());
                        }
                    }
                    _ => return Err(format!("invalid escape \\{}", esc as char)),
                }
                continue;
            }
            // Fast scan of plain UTF-8 bytes without quotes or backslashes
            let start = self.pos;
            while self.pos < self.bytes.len() && self.bytes[self.pos] != b'"' && self.bytes[self.pos] != b'\\' {
                if self.bytes[self.pos] < 0x20 {
                    return Err("unescaped control character in string".to_string());
                }
                self.pos += 1;
            }
            let slice = std::str::from_utf8(&self.bytes[start..self.pos])
                .map_err(|_| "invalid utf8 in string".to_string())?;
            out.push_str(slice);
        }
        Err("unterminated string".to_string())
    }

    fn parse_number(&mut self) -> Result<JsonValue, String> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        match self.peek() {
            Some(b'0') => self.pos += 1,
            Some(b'1'..=b'9') => {
                while matches!(self.peek(), Some(b'0'..=b'9')) {
                    self.pos += 1;
                }
            }
            _ => return Err(format!("invalid number at pos {start}")),
        }
        if self.peek() == Some(b'.') {
            self.pos += 1;
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err("expected digit after '.'".to_string());
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.pos += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err("expected exponent digit".to_string());
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }
        let s = std::str::from_utf8(&self.bytes[start..self.pos]).map_err(|_| "utf8 err")?;
        let num = s.parse::<f64>().map_err(|e| format!("bad number: {e}"))?;
        Ok(JsonValue::Number(num))
    }

    fn parse_array(&mut self) -> Result<JsonValue, String> {
        self.pos += 1;
        self.skip_ws();
        let mut items = Vec::new();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(JsonValue::Array(items));
        }
        loop {
            items.push(self.parse_value()?);
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b']') => {
                    self.pos += 1;
                    return Ok(JsonValue::Array(items));
                }
                _ => return Err("expected ',' or ']' in array".to_string()),
            }
        }
    }

    fn parse_object(&mut self) -> Result<JsonValue, String> {
        self.pos += 1;
        self.skip_ws();
        let mut entries = Vec::new();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(JsonValue::Object(entries));
        }
        loop {
            self.skip_ws();
            let key = self.parse_string()?;
            self.skip_ws();
            if self.peek() != Some(b':') {
                return Err("expected ':' in object".to_string());
            }
            self.pos += 1;
            let val = self.parse_value()?;
            entries.push((key, val));
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(JsonValue::Object(entries));
                }
                _ => return Err("expected ',' or '}' in object".to_string()),
            }
        }
    }
}

pub fn parse_json(input: &str) -> Result<JsonValue, String> {
    let mut p = Parser {
        bytes: input.as_bytes(),
        pos: 0,
    };
    let val = p.parse_value()?;
    p.skip_ws();
    if p.pos != p.bytes.len() {
        return Err(format!("trailing bytes at pos {}", p.pos));
    }
    Ok(val)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_nested_json() {
        let raw = r#"{"toolCall":{"name":"write_file","args":{"Path":".env","Content":"A=1\nB=\"x\""}},"ok":true,"n":42}"#;
        let parsed = parse_json(raw).expect("valid json");
        let tc = parsed.get("toolCall").expect("toolCall");
        assert_eq!(tc.get("name").and_then(|v| v.as_str()), Some("write_file"));
        assert_eq!(parsed.get("ok").and_then(|v| v.as_bool()), Some(true));
        assert_eq!(parsed.get("n").and_then(|v| v.as_i64()), Some(42));
        let re = to_json_string(&parsed);
        assert_eq!(parse_json(&re).unwrap(), parsed);
    }

    #[test]
    fn parses_unicode_surrogate_and_rejects_trailing() {
        let v = parse_json(r#""\uD83D\uDE80""#).unwrap();
        assert_eq!(v.as_str(), Some("\u{1F680}"));
        assert!(parse_json("null extra").is_err());
    }
}
