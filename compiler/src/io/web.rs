//! Web search and page reading — zero API keys required.
//!
//! `web_search(query)` → searches DuckDuckGo HTML endpoint, returns array of {title, url, snippet}.
//! `web_read(url)` → fetches URL, extracts text content, returns {title, text, headings, links}.
//!
//! Uses curl for HTTPS (same pattern as chat API mode).
//! HTML extraction is regex/string-based (no external parser dependency).

use std::collections::HashMap;
use std::process::Command;
use crate::{CliError, Value};

/// URL-encode a query string for use in URLs.
fn url_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len() * 3);
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char);
            }
            b' ' => out.push('+'),
            _ => {
                out.push('%');
                out.push_str(&format!("{:02X}", b));
            }
        }
    }
    out
}

/// Fetch a URL using curl (handles HTTPS transparently).
fn curl_get(url: &str) -> Result<String, CliError> {
    let output = Command::new("curl")
        .args([
            "-s", "-L",
            "--max-time", "15",
            "-H", "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            url,
        ])
        .output()
        .map_err(|e| CliError::Io(format!("curl failed: {}", e)))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(CliError::Runtime(format!("web fetch failed: {}", stderr.trim())));
    }

    String::from_utf8(output.stdout)
        .map_err(|e| CliError::Runtime(format!("invalid UTF-8 in response: {}", e)))
}

/// Fetch a URL using curl with POST body.
fn curl_post(url: &str, body: &str, content_type: &str) -> Result<String, CliError> {
    let output = Command::new("curl")
        .args([
            "-s", "-L",
            "--max-time", "15",
            "-X", "POST",
            "-H", &format!("Content-Type: {}", content_type),
            "-H", "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "-d", body,
            url,
        ])
        .output()
        .map_err(|e| CliError::Io(format!("curl failed: {}", e)))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(CliError::Runtime(format!("web fetch failed: {}", stderr.trim())));
    }

    String::from_utf8(output.stdout)
        .map_err(|e| CliError::Runtime(format!("invalid UTF-8 in response: {}", e)))
}

/// Decode HTML entities (&amp; &lt; &gt; &quot; &#N; &#xHH;).
fn decode_html_entities(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '&' {
            let mut entity = String::new();
            for ec in chars.by_ref() {
                if ec == ';' { break; }
                entity.push(ec);
                if entity.len() > 10 { break; }
            }
            match entity.as_str() {
                "amp" => out.push('&'),
                "lt" => out.push('<'),
                "gt" => out.push('>'),
                "quot" => out.push('"'),
                "apos" => out.push('\''),
                "nbsp" => out.push(' '),
                s if s.starts_with('#') => {
                    let code = if s.starts_with("#x") || s.starts_with("#X") {
                        u32::from_str_radix(&s[2..], 16).ok()
                    } else {
                        s[1..].parse::<u32>().ok()
                    };
                    if let Some(c) = code.and_then(char::from_u32) {
                        out.push(c);
                    }
                }
                _ => {
                    out.push('&');
                    out.push_str(&entity);
                    out.push(';');
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// Strip HTML tags from a string.
fn strip_tags(html: &str) -> String {
    let mut out = String::with_capacity(html.len());
    let mut in_tag = false;
    for c in html.chars() {
        match c {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => out.push(c),
            _ => {}
        }
    }
    decode_html_entities(&out)
}

/// Extract text content between two patterns (inclusive of neither).
fn extract_between<'a>(html: &'a str, start: &str, end: &str) -> Option<&'a str> {
    let s = html.find(start)?;
    let content_start = s + start.len();
    let e = html[content_start..].find(end)?;
    Some(&html[content_start..content_start + e])
}

/// Extract attribute value from an HTML tag fragment.
fn extract_attr(tag: &str, attr: &str) -> Option<String> {
    let search = format!("{}=\"", attr);
    let pos = tag.find(&search)?;
    let start = pos + search.len();
    let end = tag[start..].find('"')?;
    Some(decode_html_entities(&tag[start..start + end]))
}

// ── web_search ──────────────────────────────────────────────────────

/// Search DuckDuckGo HTML endpoint. Returns array of {title, url, snippet} maps.
pub fn web_search(query: &str) -> Result<Vec<Value>, CliError> {
    let encoded = url_encode(query);
    let body = format!("q={}", encoded);
    let html = curl_post(
        "https://html.duckduckgo.com/html/",
        &body,
        "application/x-www-form-urlencoded",
    )?;

    parse_ddg_results(&html)
}

/// Parse DuckDuckGo HTML search results.
fn parse_ddg_results(html: &str) -> Result<Vec<Value>, CliError> {
    let mut results = Vec::new();
    let mut pos = 0;

    while let Some(div_start) = html[pos..].find("class=\"result ") {
        let abs_start = pos + div_start;

        // Find the next result div or end of results
        let chunk_end = html[abs_start + 20..].find("class=\"result ")
            .map(|p| abs_start + 20 + p)
            .unwrap_or(html.len());

        let chunk = &html[abs_start..chunk_end];

        // Extract title + URL from result__a
        let title = if let Some(a_start) = chunk.find("class=\"result__a\"") {
            let a_tag_end = chunk[a_start..].find('>').unwrap_or(0);
            let a_content_start = a_start + a_tag_end + 1;
            let a_end = chunk[a_content_start..].find("</a>").unwrap_or(0);
            strip_tags(&chunk[a_content_start..a_content_start + a_end]).trim().to_string()
        } else {
            String::new()
        };

        // Extract URL from href
        let url = if let Some(a_start) = chunk.find("class=\"result__a\"") {
            // Look backwards from class to find href
            let prefix = &chunk[..a_start];
            if let Some(href_start) = prefix.rfind("href=\"") {
                let href_val_start = href_start + 6;
                let href_end = chunk[href_val_start..].find('"').unwrap_or(0);
                let raw_url = &chunk[href_val_start..href_val_start + href_end];
                // DDG uses redirect URLs; extract the actual URL
                if let Some(uddg) = raw_url.find("uddg=") {
                    let url_start = uddg + 5;
                    let url_end = raw_url[url_start..].find('&').unwrap_or(raw_url.len() - url_start);
                    url_decode(&raw_url[url_start..url_start + url_end])
                } else {
                    decode_html_entities(raw_url)
                }
            } else {
                String::new()
            }
        } else {
            String::new()
        };

        // Extract snippet
        let snippet = if let Some(snip_start) = chunk.find("class=\"result__snippet\"") {
            let tag_end = chunk[snip_start..].find('>').unwrap_or(0);
            let content_start = snip_start + tag_end + 1;
            let content_end = chunk[content_start..].find("</").unwrap_or(0);
            strip_tags(&chunk[content_start..content_start + content_end]).trim().to_string()
        } else {
            String::new()
        };

        if !title.is_empty() && !url.is_empty() {
            let mut map = HashMap::new();
            map.insert("title".to_string(), Value::Str(title));
            map.insert("url".to_string(), Value::Str(url));
            map.insert("snippet".to_string(), Value::Str(snippet));
            results.push(Value::Map(map));
        }

        pos = chunk_end;
        if results.len() >= 10 { break; }
    }

    Ok(results)
}

/// URL-decode a percent-encoded string.
fn url_decode(s: &str) -> String {
    let mut out = Vec::new();
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let Ok(byte) = u8::from_str_radix(
                std::str::from_utf8(&bytes[i+1..i+3]).unwrap_or(""),
                16,
            ) {
                out.push(byte);
                i += 3;
                continue;
            }
        }
        if bytes[i] == b'+' {
            out.push(b' ');
        } else {
            out.push(bytes[i]);
        }
        i += 1;
    }
    String::from_utf8(out).unwrap_or_else(|_| s.to_string())
}

// ── web_read ────────────────────────────────────────────────────────

/// Read a web page and extract structured content.
/// Returns a map with: title, text, headings, links.
/// Only allows http:// and https:// protocols — rejects file://, ftp://, etc. (SSRF protection).
pub fn web_read(url: &str) -> Result<Value, CliError> {
    let lower = url.to_ascii_lowercase();
    if !lower.starts_with("http://") && !lower.starts_with("https://") {
        return Err(CliError::Runtime(format!(
            "web_read: only http:// and https:// URLs are allowed, got '{}'", url
        )));
    }
    let html = curl_get(url)?;
    parse_page_content(&html, url)
}

/// Parse HTML page into structured content map.
fn parse_page_content(html: &str, _url: &str) -> Result<Value, CliError> {
    let mut result = HashMap::new();

    // Extract <title>
    let title = extract_between(html, "<title", "</title>")
        .map(|t| {
            // Skip attributes in <title ...>
            let content = t.find('>').map(|p| &t[p + 1..]).unwrap_or(t);
            strip_tags(content).trim().to_string()
        })
        .unwrap_or_default();
    result.insert("title".to_string(), Value::Str(title));

    // Extract main text content (from <p> tags, highest density)
    let text = extract_main_text(html);
    result.insert("text".to_string(), Value::Str(text));

    // Extract headings (h1-h6)
    let headings = extract_headings(html);
    result.insert("headings".to_string(), Value::Str(headings.join("\n")));

    // Extract links
    let links = extract_links_from_html(html);
    result.insert("links".to_string(), Value::Str(
        links.iter().map(|(text, href)| format!("{}: {}", text, href)).collect::<Vec<_>>().join("\n")
    ));

    Ok(Value::Map(result))
}

/// Extract main text content using content-density heuristic.
fn extract_main_text(html: &str) -> String {
    let mut paragraphs = Vec::new();

    // Extract text from <p> tags (primary content signal)
    let mut pos = 0;
    while let Some(p_start) = html[pos..].find("<p") {
        let abs = pos + p_start;
        let tag_end = html[abs..].find('>').unwrap_or(0);
        let content_start = abs + tag_end + 1;
        if let Some(end_tag) = html[content_start..].find("</p>") {
            let content = &html[content_start..content_start + end_tag];
            let text = strip_tags(content).trim().to_string();
            if text.len() > 20 { // Skip very short fragments
                paragraphs.push(text);
            }
            pos = content_start + end_tag + 4;
        } else {
            pos = content_start;
        }
    }

    // Also try <article> and <main> content
    for tag in &["article", "main"] {
        let open = format!("<{}", tag);
        let close = format!("</{}>", tag);
        if let Some(content) = extract_between(html, &open, &close) {
            // Only use if we didn't find enough paragraphs
            if paragraphs.len() < 3 {
                let tag_end = content.find('>').map(|p| &content[p + 1..]).unwrap_or(content);
                let text = strip_tags(tag_end).trim().to_string();
                if text.len() > paragraphs.iter().map(|p| p.len()).sum::<usize>() {
                    paragraphs = text.split('\n')
                        .map(|l| l.trim().to_string())
                        .filter(|l| l.len() > 20)
                        .collect();
                }
            }
        }
    }

    // Truncate to reasonable size for LLM context
    let mut result = String::new();
    for p in &paragraphs {
        if result.len() + p.len() > 8000 { break; }
        if !result.is_empty() { result.push_str("\n\n"); }
        result.push_str(p);
    }
    result
}

/// Extract headings (h1-h6) from HTML.
fn extract_headings(html: &str) -> Vec<String> {
    let mut headings = Vec::new();
    for level in 1..=6 {
        let open = format!("<h{}", level);
        let close = format!("</h{}>", level);
        let mut pos = 0;
        while let Some(h_start) = html[pos..].find(&open) {
            let abs = pos + h_start;
            let tag_end = html[abs..].find('>').unwrap_or(0);
            let content_start = abs + tag_end + 1;
            if let Some(end_tag) = html[content_start..].find(&close) {
                let text = strip_tags(&html[content_start..content_start + end_tag]).trim().to_string();
                if !text.is_empty() {
                    headings.push(text);
                }
                pos = content_start + end_tag + close.len();
            } else {
                pos = content_start;
            }
        }
    }
    headings
}

/// Extract links (text + href) from <a> tags.
fn extract_links_from_html(html: &str) -> Vec<(String, String)> {
    let mut links = Vec::new();
    let mut pos = 0;
    while let Some(a_start) = html[pos..].find("<a ") {
        let abs = pos + a_start;
        let tag_end = html[abs..].find('>').unwrap_or(0);
        let tag = &html[abs..abs + tag_end + 1];
        let content_start = abs + tag_end + 1;
        if let Some(end_tag) = html[content_start..].find("</a>") {
            let text = strip_tags(&html[content_start..content_start + end_tag]).trim().to_string();
            if let Some(href) = extract_attr(tag, "href") {
                if !text.is_empty() && !href.is_empty() && !href.starts_with('#') && !href.starts_with("javascript:") {
                    links.push((text, href));
                }
            }
            pos = content_start + end_tag + 4;
        } else {
            pos = abs + 3;
        }
        if links.len() >= 50 { break; }
    }
    links
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── SEC-S02: URL protocol validation (SSRF protection) ──

    #[test]
    fn test_web_read_rejects_file_protocol() {
        let result = web_read("file:///etc/passwd");
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("only http:// and https://"), "got: {}", msg);
    }

    #[test]
    fn test_web_read_rejects_ftp_protocol() {
        let result = web_read("ftp://evil.com/data");
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("only http:// and https://"), "got: {}", msg);
    }

    #[test]
    fn test_web_read_rejects_gopher_protocol() {
        let result = web_read("gopher://evil.com");
        assert!(result.is_err());
    }

    #[test]
    fn test_web_read_rejects_javascript_protocol() {
        let result = web_read("javascript:alert(1)");
        assert!(result.is_err());
    }

    #[test]
    fn test_web_read_rejects_data_uri() {
        let result = web_read("data:text/html,<h1>xss</h1>");
        assert!(result.is_err());
    }

    #[test]
    fn test_web_read_rejects_empty_url() {
        let result = web_read("");
        assert!(result.is_err());
    }

    #[test]
    fn test_web_read_rejects_case_insensitive_file() {
        // Bypass attempt: FILE:///etc/passwd
        let result = web_read("FILE:///etc/passwd");
        assert!(result.is_err());
    }

    // ── SEC-S01: URL encoding ──

    #[test]
    fn test_url_encode() {
        assert_eq!(url_encode("hello world"), "hello+world");
        assert_eq!(url_encode("a&b=c"), "a%26b%3Dc");
    }

    #[test]
    fn test_url_decode() {
        assert_eq!(url_decode("hello%20world"), "hello world");
        assert_eq!(url_decode("a%26b"), "a&b");
        assert_eq!(url_decode("hello+world"), "hello world");
    }

    #[test]
    fn test_strip_tags() {
        assert_eq!(strip_tags("<b>bold</b> text"), "bold text");
        assert_eq!(strip_tags("no tags"), "no tags");
        assert_eq!(strip_tags("<a href=\"x\">link</a>"), "link");
    }

    #[test]
    fn test_decode_html_entities() {
        assert_eq!(decode_html_entities("&amp; &lt; &gt;"), "& < >");
        assert_eq!(decode_html_entities("&#65;"), "A");
        assert_eq!(decode_html_entities("&#x41;"), "A");
    }

    #[test]
    fn test_parse_ddg_results_empty() {
        let results = parse_ddg_results("<html><body>no results</body></html>").unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_parse_page_content() {
        let html = r#"<html><head><title>Test Page</title></head>
        <body><h1>Hello</h1><p>This is a paragraph with enough text to pass the filter threshold.</p></body></html>"#;
        let result = parse_page_content(html, "http://example.com").unwrap();
        if let Value::Map(m) = &result {
            assert_eq!(m["title"], Value::Str("Test Page".into()));
        } else {
            panic!("expected map");
        }
    }

    #[test]
    fn test_extract_headings() {
        let html = "<h1>Title</h1><h2>Section 1</h2><h2>Section 2</h2>";
        let headings = extract_headings(html);
        assert_eq!(headings, vec!["Title", "Section 1", "Section 2"]);
    }

    #[test]
    fn test_url_encode_special_chars() {
        assert_eq!(url_encode("<script>"), "%3Cscript%3E");
        assert_eq!(url_encode("path/to file"), "path%2Fto+file");
    }

    #[test]
    fn test_url_encode_preserves_unreserved() {
        // RFC 3986 unreserved: A-Z a-z 0-9 - _ . ~
        assert_eq!(url_encode("Hello-World_v1.0~beta"), "Hello-World_v1.0~beta");
    }

    #[test]
    fn test_url_decode_percent_sequences() {
        assert_eq!(url_decode("%3C%3E"), "<>");
        assert_eq!(url_decode("100%25"), "100%");
    }

    #[test]
    fn test_url_decode_incomplete_percent() {
        // Incomplete percent sequence should pass through
        let result = url_decode("100%2");
        assert!(result.contains("100"));
    }

    #[test]
    fn test_extract_links() {
        let html = r##"<a href="https://example.com">Example</a> <a href="#top">Top</a>"##;
        let links = extract_links_from_html(html);
        assert_eq!(links.len(), 1);
        assert_eq!(links[0].0, "Example");
        assert_eq!(links[0].1, "https://example.com");
    }
}
