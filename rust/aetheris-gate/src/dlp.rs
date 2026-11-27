//! OS-independent Data Loss Prevention (DLP) secret scrubber & prompt-injection detector.
//! Compiles cleanly on `wasm32-unknown-unknown` with zero external crates.

use crate::json::{obj, to_json_string, JsonValue};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScrubResult {
    pub secrets_found: bool,
    pub redacted_text: String,
    pub injection_detected: bool,
}

const REDACTED_MARKER: &str = "[REDACTED_SECRET]";

fn is_tok_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_' || b == b'-'
}

fn is_bearer_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || matches!(b, b'-' | b'.' | b'_' | b'~' | b'+' | b'/' | b'=')
}

fn is_boundary_before(bytes: &[u8], pos: usize) -> bool {
    pos == 0 || !is_tok_char(bytes[pos - 1])
}

/// Match a prefixed token (`prefix` + at least `min_tail` matching chars) starting at `pos`.
fn match_prefixed_token(
    bytes: &[u8],
    pos: usize,
    prefix: &[u8],
    min_tail: usize,
    pred: fn(u8) -> bool,
) -> Option<usize> {
    if !bytes[pos..].starts_with(prefix) || !is_boundary_before(bytes, pos) {
        return None;
    }
    let mut end = pos + prefix.len();
    while end < bytes.len() && pred(bytes[end]) {
        end += 1;
    }
    if end - (pos + prefix.len()) >= min_tail {
        Some(end)
    } else {
        None
    }
}

fn match_pem_block(bytes: &[u8], pos: usize) -> Option<usize> {
    if !bytes[pos..].starts_with(b"-----BEGIN") {
        return None;
    }
    // Find closing `-----` of the BEGIN header line
    let mut header_end = pos + 10;
    while header_end < bytes.len() && bytes[header_end] != b'\n' {
        if bytes[header_end..].starts_with(b"-----") {
            header_end += 5;
            break;
        }
        header_end += 1;
    }
    // Look ahead for `-----END` within a bounded window (16 KiB)
    let search_limit = bytes.len().min(pos + 16384);
    let mut scan = header_end;
    while scan < search_limit {
        if bytes[scan..].starts_with(b"-----END") {
            let mut end_pos = scan + 8;
            while end_pos < search_limit && bytes[end_pos] != b'\n' {
                if bytes[end_pos..].starts_with(b"-----") {
                    return Some(end_pos + 5);
                }
                end_pos += 1;
            }
            return Some(end_pos);
        }
        scan += 1;
    }
    // No END marker: redact up to end of line or token run
    let mut line_end = header_end;
    while line_end < bytes.len() && !matches!(bytes[line_end], b'\r' | b'\n') {
        line_end += 1;
    }
    Some(line_end)
}

fn match_bearer(bytes: &[u8], pos: usize) -> Option<usize> {
    if !(bytes[pos..].starts_with(b"Bearer ") || bytes[pos..].starts_with(b"bearer ")) {
        return None;
    }
    if pos > 0 && bytes[pos - 1].is_ascii_alphanumeric() {
        return None;
    }
    let mut ws = pos + 7;
    while ws < bytes.len() && matches!(bytes[ws], b' ' | b'\t') {
        ws += 1;
    }
    if bytes[ws..].starts_with(b"[REDACTED") {
        return None;
    }
    let mut end = ws;
    while end < bytes.len() && is_bearer_char(bytes[end]) {
        end += 1;
    }
    if end - ws >= 8 {
        Some(end)
    } else {
        None
    }
}

fn match_secret_at(bytes: &[u8], pos: usize) -> Option<usize> {
    if let Some(end) = match_pem_block(bytes, pos) {
        return Some(end);
    }
    if let Some(end) = match_bearer(bytes, pos) {
        return Some(end);
    }
    for (prefix, min_tail) in [
        (b"sk_live_".as_slice(), 6usize),
        (b"sk_test_".as_slice(), 6usize),
        (b"rk_live_".as_slice(), 6usize),
        (b"sk-proj-".as_slice(), 6usize),
        (b"sk-ant-".as_slice(), 6usize),
        (b"sk-".as_slice(), 8usize),
        (b"ghp_".as_slice(), 8usize),
        (b"gho_".as_slice(), 8usize),
        (b"ghu_".as_slice(), 8usize),
        (b"ghs_".as_slice(), 8usize),
        (b"ghr_".as_slice(), 8usize),
        (b"github_pat_".as_slice(), 8usize),
        (b"glpat-".as_slice(), 8usize),
        (b"hf_".as_slice(), 10usize),
        (b"npm_".as_slice(), 10usize),
        (b"xoxb-".as_slice(), 8usize),
        (b"xoxp-".as_slice(), 8usize),
    ] {
        if let Some(end) = match_prefixed_token(bytes, pos, prefix, min_tail, is_tok_char) {
            return Some(end);
        }
    }
    if let Some(end) = match_prefixed_token(bytes, pos, b"AKIA", 12, |b| {
        b.is_ascii_uppercase() || b.is_ascii_digit()
    }) {
        return Some(end);
    }
    if let Some(end) = match_prefixed_token(bytes, pos, b"ASIA", 12, |b| {
        b.is_ascii_uppercase() || b.is_ascii_digit()
    }) {
        return Some(end);
    }
    None
}

fn detect_prompt_injection(text: &str) -> bool {
    let low = text.to_ascii_lowercase();
    const MARKERS: &[&str] = &[
        "<important>",
        "ignore previous instructions",
        "ignore all previous instructions",
        "disregard previous instructions",
        "forget previous instructions",
        "<|im_start|>system",
    ];
    MARKERS.iter().any(|m| low.contains(m))
}

/// Scan `text` for raw credentials and prompt-injection markers, redacting matched secrets.
pub fn scrub_text(text: &str) -> ScrubResult {
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut secrets_found = false;
    let mut pos = 0usize;
    let mut copied_up_to = 0usize;

    while pos < bytes.len() {
        if let Some(end) = match_secret_at(bytes, pos) {
            out.push_str(&text[copied_up_to..pos]);
            out.push_str(REDACTED_MARKER);
            secrets_found = true;
            pos = end;
            copied_up_to = end;
        } else {
            let ch_len = text[pos..].chars().next().map_or(1, char::len_utf8);
            pos += ch_len;
        }
    }
    if copied_up_to < text.len() {
        out.push_str(&text[copied_up_to..]);
    }

    let injection_detected = detect_prompt_injection(text);
    ScrubResult {
        secrets_found,
        redacted_text: out,
        injection_detected,
    }
}

/// Serialize `scrub_text` output as JSON `{secrets_found, redacted_text, injection_detected}`.
pub fn scrub_text_json(text: &str) -> String {
    let res = scrub_text(text);
    to_json_string(&obj(vec![
        ("secrets_found", JsonValue::Bool(res.secrets_found)),
        ("redacted_text", JsonValue::Str(res.redacted_text)),
        ("injection_detected", JsonValue::Bool(res.injection_detected)),
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redacts_supported_tokens_and_detects_injection() {
        let tok_live = ["sk", "live", "test1234567890abcd"].join("_");
        let tok_proj = ["sk", "proj", "abcdef1234567890"].join("-");
        let tok_sk = ["sk", "antigravity9988776655"].join("-");
        let tok_aws = format!("{}{}", "AKIA", "IOSFODNN7EXAMPLE");
        let tok_gh = format!("{}{}", "ghp_", "abcdefghijklmnop1234");
        let tok_gl = format!("{}{}", "glpat-", "abcdefghijklmnop1234");
        let pem = format!(
            "{} RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA\n{} RSA PRIVATE KEY-----",
            "-----BEGIN", "-----END"
        );
        let bearer = format!("{} {}", "Bearer", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9_sig");

        let input = format!(
            "a={tok_live} b={tok_proj} c={tok_sk} d={tok_aws} e={tok_gh} f={tok_gl}\n{pem}\nauth={bearer}\n<IMPORTANT> ignore previous instructions"
        );
        let res = scrub_text(&input);
        assert!(res.secrets_found);
        assert!(res.injection_detected);
        for raw in [&tok_live, &tok_proj, &tok_sk, &tok_aws, &tok_gh, &tok_gl] {
            assert!(!res.redacted_text.contains(raw), "leaked token: {raw}");
        }
        assert!(!res.redacted_text.contains("MIIEowIBAAKCAQEA"));
        assert!(!res.redacted_text.contains("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9_sig"));
        assert!(res.redacted_text.contains(REDACTED_MARKER));
    }

    #[test]
    fn passes_clean_text_untouched() {
        let clean = "Normal diagnostic output with disk-check and sk-short ok.";
        let res = scrub_text(clean);
        assert!(!res.secrets_found);
        assert!(!res.injection_detected);
        assert_eq!(res.redacted_text, clean);
    }
}
