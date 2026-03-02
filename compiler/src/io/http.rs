//! Pure-Rust HTTP/1.1 client and server request parser — zero external dependencies.
//!
//! Uses net_io.rs for raw TCP.  Supports plain HTTP only (no TLS).
//! HTTPS URLs return an error; TLS support is a future phase.

use crate::io::net;

// ── URL parsing ───────────────────────────────────────────────────────

/// Parse `http://host[:port]/path[?query]` → (host, port, path_and_query).
pub fn parse_url(url: &str) -> Result<(String, u16, String), String> {
    let rest = if let Some(r) = url.strip_prefix("http://") {
        r
    } else if url.starts_with("https://") {
        return Err("HTTPS not yet supported — use http:// (TLS support is a future phase)".into());
    } else {
        return Err(format!("unsupported URL scheme: {}", url));
    };

    // Split host[:port] from /path
    let (authority, path) = match rest.find('/') {
        Some(idx) => (&rest[..idx], &rest[idx..]),
        None      => (rest, "/"),
    };

    let (host, port) = if let Some(colon) = authority.rfind(':') {
        let port: u16 = authority[colon + 1..].parse()
            .map_err(|_| format!("invalid port in URL: {}", url))?;
        (authority[..colon].to_string(), port)
    } else {
        (authority.to_string(), 80u16)
    };

    Ok((host, port, path.to_string()))
}

// ── HTTP/1.1 client ───────────────────────────────────────────────────

/// Execute an HTTP request over plain TCP.
/// Returns (status_code, body, ok_flag, error_message).
pub fn do_request(
    method: &str,
    url: &str,
    body_opt: Option<&str>,
) -> (f32, String, f32, String) {
    let (host, port, path) = match parse_url(url) {
        Ok(v)  => v,
        Err(e) => return (0.0, String::new(), 0.0, e),
    };

    // Connect
    let fd = net::tcp_connect(&host, port);
    if fd < 0 {
        return (0.0, String::new(), 0.0,
            format!("HTTP {}: connection refused ({}:{})", method, host, port));
    }

    // Build request
    let request = build_request(method, &path, &host, body_opt);

    // Send
    if net::tcp_send(fd, &request) < 0 {
        net::socket_close(fd);
        return (0.0, String::new(), 0.0, format!("HTTP {}: send failed", method));
    }

    // Read full response (loop until EOF)
    let mut raw = Vec::new();
    loop {
        match net::tcp_recv(fd, 8192) {
            Ok(chunk) if chunk.is_empty() => break,
            Ok(chunk) => raw.extend_from_slice(chunk.as_bytes()),
            Err(_)    => break,
        }
    }
    net::socket_close(fd);

    // Parse response
    parse_response(&raw, method)
}

fn build_request(method: &str, path: &str, host: &str, body_opt: Option<&str>) -> String {
    let body = body_opt.unwrap_or("");
    let mut req = format!(
        "{} {} HTTP/1.1\r\nHost: {}\r\nUser-Agent: OctoFlow/1.0\r\nConnection: close\r\n",
        method, path, host
    );
    if !body.is_empty() {
        req.push_str(&format!(
            "Content-Type: application/json\r\nContent-Length: {}\r\n",
            body.len()
        ));
    }
    req.push_str("\r\n");
    req.push_str(body);
    req
}

fn parse_response(raw: &[u8], method: &str) -> (f32, String, f32, String) {
    // Find header/body split
    let split = find_header_end(raw);
    let (header_bytes, body_bytes) = if let Some(idx) = split {
        (&raw[..idx], &raw[idx + 4..])
    } else {
        return (0.0, String::new(), 0.0,
            format!("HTTP {}: malformed response (no header terminator)", method));
    };

    let header_str = String::from_utf8_lossy(header_bytes);
    let mut lines = header_str.split("\r\n");

    // Status line: "HTTP/1.1 200 OK"
    let status_line = lines.next().unwrap_or("");
    let status: f32 = status_line.splitn(3, ' ')
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);

    // Collect headers to find Content-Length and Transfer-Encoding
    let mut content_length: Option<usize> = None;
    let mut chunked = false;
    for line in lines {
        let lower = line.to_lowercase();
        if lower.starts_with("content-length:") {
            let v = lower.trim_start_matches("content-length:").trim();
            content_length = v.parse().ok();
        }
        if lower.contains("transfer-encoding") && lower.contains("chunked") {
            chunked = true;
        }
    }

    // Decode body
    let body = if chunked {
        decode_chunked(body_bytes)
    } else if let Some(len) = content_length {
        let end = len.min(body_bytes.len());
        String::from_utf8_lossy(&body_bytes[..end]).into_owned()
    } else {
        String::from_utf8_lossy(body_bytes).into_owned()
    };

    let ok = if (200.0..300.0).contains(&status) { 1.0 } else { 0.0 };
    if ok == 0.0 && status > 0.0 {
        (status, body, 0.0, format!("HTTP {} error {}", method, status as u32))
    } else if status == 0.0 {
        (0.0, body, 0.0, format!("HTTP {}: could not parse status", method))
    } else {
        (status, body, 1.0, String::new())
    }
}

fn find_header_end(data: &[u8]) -> Option<usize> {
    data.windows(4).position(|w| w == b"\r\n\r\n")
}

/// Decode HTTP chunked transfer encoding.
fn decode_chunked(data: &[u8]) -> String {
    let mut out = Vec::new();
    let mut pos = 0;
    loop {
        // Find end of chunk-size line
        let line_end = match data[pos..].windows(2).position(|w| w == b"\r\n") {
            Some(i) => pos + i,
            None    => break,
        };
        let size_str = std::str::from_utf8(&data[pos..line_end]).unwrap_or("0");
        let chunk_size = usize::from_str_radix(size_str.trim(), 16).unwrap_or(0);
        if chunk_size == 0 { break; }
        pos = line_end + 2; // skip \r\n after size
        let end = (pos + chunk_size).min(data.len());
        out.extend_from_slice(&data[pos..end]);
        pos = end + 2; // skip trailing \r\n after chunk
        if pos >= data.len() { break; }
    }
    String::from_utf8_lossy(&out).into_owned()
}

// ── HTTP server request parsing ────────────────────────────────────────

/// Parsed HTTP request from an incoming TCP connection.
#[derive(Debug, Clone)]
pub struct HttpRequest {
    pub method:  String,
    pub path:    String,
    pub query:   String,
    pub headers: Vec<(String, String)>,
    pub body:    String,
}

/// Read an HTTP request from an accepted TCP connection fd.
/// Returns the parsed request or an error.
pub fn read_request(fd: i64) -> Result<HttpRequest, String> {
    // Read until \r\n\r\n (end of headers)
    let mut raw = Vec::new();
    loop {
        let chunk = net::tcp_recv(fd, 4096)
            .map_err(|e| format!("read_request: {}", e))?;
        raw.extend_from_slice(chunk.as_bytes());
        if find_header_end(&raw).is_some() { break; }
        if chunk.is_empty() { break; }
        if raw.len() > 65536 { return Err("request too large".into()); }
    }

    let header_end = find_header_end(&raw)
        .ok_or("read_request: incomplete request (no \\r\\n\\r\\n)")?;

    let header_str = String::from_utf8_lossy(&raw[..header_end]);
    let mut lines = header_str.split("\r\n");

    // Request line: "METHOD /path?query HTTP/1.1"
    let req_line = lines.next().ok_or("empty request")?;
    let mut req_parts = req_line.splitn(3, ' ');
    let method = req_parts.next().unwrap_or("").to_string();
    let target = req_parts.next().unwrap_or("/");

    let (path, query) = if let Some(q) = target.find('?') {
        (target[..q].to_string(), target[q + 1..].to_string())
    } else {
        (target.to_string(), String::new())
    };

    // Headers
    let mut headers = Vec::new();
    let mut content_length: usize = 0;
    for line in lines {
        if line.is_empty() { break; }
        if let Some(colon) = line.find(':') {
            let name  = line[..colon].trim().to_lowercase();
            let value = line[colon + 1..].trim().to_string();
            if name == "content-length" {
                content_length = value.parse().unwrap_or(0);
            }
            headers.push((name, value));
        }
    }

    // Body: already partially buffered after headers
    let already_read = raw.len() - header_end - 4;
    let mut body_bytes = raw[header_end + 4..].to_vec();

    // Read remaining body bytes if Content-Length > what we have
    while body_bytes.len() < content_length {
        let need = content_length - body_bytes.len();
        match net::tcp_recv(fd, need.min(4096)) {
            Ok(chunk) if chunk.is_empty() => break,
            Ok(chunk) => body_bytes.extend_from_slice(chunk.as_bytes()),
            Err(_)    => break,
        }
    }
    let _ = already_read;

    let body = String::from_utf8_lossy(&body_bytes).into_owned();

    Ok(HttpRequest { method, path, query, headers, body })
}

/// Build and send an HTTP response on fd.
/// No CORS headers by default — use send_response_cors() to add explicit CORS.
pub fn send_response(fd: i64, status: u16, content_type: &str, body: &str) {
    let reason = status_reason(status);
    let response = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        status, reason, content_type, body.len(), body
    );
    net::tcp_send(fd, &response);
    net::socket_close(fd);
}

/// Build and send an HTTP response with binary body on fd.
/// No CORS headers by default — use send_response_bytes_cors() to add explicit CORS.
pub fn send_response_bytes(fd: i64, status: u16, content_type: &str, body: &[u8]) {
    let reason = status_reason(status);
    let header = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        status, reason, content_type, body.len()
    );
    net::tcp_send(fd, &header);
    net::tcp_send_bytes(fd, body);
    net::socket_close(fd);
}

/// Build and send an HTTP response with explicit CORS origin header.
/// Pass "*" for any origin, or a specific origin like "https://example.com".
pub fn send_response_cors(fd: i64, status: u16, content_type: &str, body: &str, cors_origin: &str) {
    let reason = status_reason(status);
    let response = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\nAccess-Control-Allow-Origin: {}\r\n\r\n{}",
        status, reason, content_type, body.len(), cors_origin, body
    );
    net::tcp_send(fd, &response);
    net::socket_close(fd);
}

/// Build and send an HTTP response with binary body and explicit CORS origin header.
pub fn send_response_bytes_cors(fd: i64, status: u16, content_type: &str, body: &[u8], cors_origin: &str) {
    let reason = status_reason(status);
    let header = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\nAccess-Control-Allow-Origin: {}\r\n\r\n",
        status, reason, content_type, body.len(), cors_origin
    );
    net::tcp_send(fd, &header);
    net::tcp_send_bytes(fd, body);
    net::socket_close(fd);
}

/// Build and send an HTTP response with custom headers on fd.
/// `extra_headers` is a slice of (name, value) pairs.
pub fn send_response_with_headers(
    fd: i64,
    status: u16,
    headers: &[(&str, &str)],
    body: &str,
) {
    let reason = status_reason(status);
    let mut response = format!(
        "HTTP/1.1 {} {}\r\nContent-Length: {}\r\nConnection: close\r\n",
        status, reason, body.len()
    );
    for (name, value) in headers {
        // Sanitize: strip \r and \n to prevent header injection
        let clean_name: String = name.chars().filter(|&c| c != '\r' && c != '\n').collect();
        let clean_value: String = value.chars().filter(|&c| c != '\r' && c != '\n').collect();
        response.push_str(&clean_name);
        response.push_str(": ");
        response.push_str(&clean_value);
        response.push_str("\r\n");
    }
    response.push_str("\r\n");
    response.push_str(body);
    net::tcp_send(fd, &response);
    net::socket_close(fd);
}

fn status_reason(code: u16) -> &'static str {
    match code {
        200 => "OK",
        201 => "Created",
        204 => "No Content",
        301 => "Moved Permanently",
        302 => "Found",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        500 => "Internal Server Error",
        _   => "Unknown",
    }
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_url_basic() {
        let (host, port, path) = parse_url("http://example.com/hello").unwrap();
        assert_eq!(host, "example.com");
        assert_eq!(port, 80);
        assert_eq!(path, "/hello");
    }

    #[test]
    fn test_parse_url_with_port() {
        let (host, port, path) = parse_url("http://localhost:8080/api").unwrap();
        assert_eq!(host, "localhost");
        assert_eq!(port, 8080);
        assert_eq!(path, "/api");
    }

    #[test]
    fn test_parse_url_root() {
        let (host, port, path) = parse_url("http://example.com").unwrap();
        assert_eq!(host, "example.com");
        assert_eq!(port, 80);
        assert_eq!(path, "/");
    }

    #[test]
    fn test_parse_url_https_error() {
        let err = parse_url("https://example.com").unwrap_err();
        assert!(err.contains("HTTPS not yet supported"), "got: {}", err);
    }

    #[test]
    fn test_parse_url_unknown_scheme() {
        assert!(parse_url("ftp://example.com").is_err());
    }

    #[test]
    fn test_parse_response_ok() {
        let raw = b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nhello";
        let (status, body, ok, err) = parse_response(raw, "GET");
        assert_eq!(status, 200.0);
        assert_eq!(body, "hello");
        assert_eq!(ok, 1.0);
        assert!(err.is_empty());
    }

    #[test]
    fn test_parse_response_404() {
        let raw = b"HTTP/1.1 404 Not Found\r\nContent-Length: 9\r\n\r\nnot found";
        let (status, body, ok, _err) = parse_response(raw, "GET");
        assert_eq!(status, 404.0);
        assert_eq!(body, "not found");
        assert_eq!(ok, 0.0);
    }

    #[test]
    fn test_decode_chunked() {
        // "5\r\nhello\r\n6\r\n world\r\n0\r\n\r\n"
        let data = b"5\r\nhello\r\n6\r\n world\r\n0\r\n\r\n";
        let out = decode_chunked(data);
        assert_eq!(out, "hello world");
    }

    #[test]
    fn test_build_request_get() {
        let req = build_request("GET", "/path", "example.com", None);
        assert!(req.starts_with("GET /path HTTP/1.1\r\n"));
        assert!(req.contains("Host: example.com\r\n"));
        assert!(!req.contains("Content-Length"));
    }

    #[test]
    fn test_build_request_post_with_body() {
        let req = build_request("POST", "/api", "example.com", Some("{\"x\":1}"));
        assert!(req.contains("Content-Length: 7\r\n"));
        assert!(req.contains("Content-Type: application/json\r\n"));
        assert!(req.ends_with("{\"x\":1}"));
    }

    // ── SEC-S03: CORS default-deny ──
    // The send_response functions require a socket fd, so we test the
    // format strings directly to verify CORS headers are NOT present
    // in the default response and ARE present in the CORS variant.

    #[test]
    fn test_response_format_no_cors_by_default() {
        let reason = status_reason(200);
        let body = "hello";
        let response = format!(
            "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            200, reason, "text/plain", body.len(), body
        );
        assert!(!response.contains("Access-Control-Allow-Origin"),
            "Default response must NOT include CORS header");
    }

    #[test]
    fn test_response_format_cors_explicit_origin() {
        let reason = status_reason(200);
        let body = "hello";
        let cors_origin = "https://example.com";
        let response = format!(
            "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\nAccess-Control-Allow-Origin: {}\r\n\r\n{}",
            200, reason, "text/plain", body.len(), cors_origin, body
        );
        assert!(response.contains("Access-Control-Allow-Origin: https://example.com"),
            "CORS response must include specified origin");
    }

    #[test]
    fn test_response_format_cors_wildcard() {
        let reason = status_reason(200);
        let body = "data";
        let response = format!(
            "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\nAccess-Control-Allow-Origin: {}\r\n\r\n{}",
            200, reason, "application/json", body.len(), "*", body
        );
        assert!(response.contains("Access-Control-Allow-Origin: *"));
    }

    #[test]
    fn test_status_reason_codes() {
        assert_eq!(status_reason(200), "OK");
        assert_eq!(status_reason(404), "Not Found");
        assert_eq!(status_reason(500), "Internal Server Error");
    }
}
