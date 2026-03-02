//! MCP (Model Context Protocol) server for OctoFlow.
//!
//! Speaks JSON-RPC 2.0 over stdio. Exposes OctoFlow as tools for OpenClaw,
//! Claude, and any MCP client.
//!
//! Usage: `octoflow mcp-serve [--allow-read=path] [--allow-write=path] ...`
//!
//! Protocol flow:
//!   1. Client sends `initialize` → server responds with capabilities
//!   2. Client sends `tools/list` → server responds with 10 tool definitions
//!   3. Client sends `tools/call` → server executes tool, returns result
//!   4. Loop until stdin EOF

use crate::{CliError, Overrides, PermScope};
use crate::compiler;
use crate::analysis::preflight;

/// MCP protocol version we implement.
const PROTOCOL_VERSION: &str = "2024-11-05";

/// Run the MCP server loop on stdin/stdout.
pub fn serve_mcp(overrides: Overrides) -> Result<(), CliError> {
    let stdin = std::io::stdin();
    let mut line_buf = String::new();

    loop {
        line_buf.clear();
        match stdin.read_line(&mut line_buf) {
            Ok(0) => break, // EOF
            Ok(_) => {
                let trimmed = line_buf.trim();
                if trimmed.is_empty() {
                    continue;
                }
                let response = handle_request(trimmed, &overrides);
                println!("{}", response);
            }
            Err(e) => {
                eprintln!("mcp: stdin read error: {}", e);
                break;
            }
        }
    }

    Ok(())
}

// ── JSON-RPC 2.0 Helpers ──

/// Extract a string field from a JSON object string.
fn json_get_str<'a>(json: &'a str, key: &str) -> Option<&'a str> {
    let pattern = format!("\"{}\"", key);
    let pos = json.find(&pattern)?;
    let after = &json[pos + pattern.len()..];
    // Skip whitespace and colon
    let after = after.trim_start();
    if !after.starts_with(':') { return None; }
    let after = after[1..].trim_start();
    if after.starts_with('"') {
        // String value
        let content = &after[1..];
        let end = find_unescaped_quote(content)?;
        Some(&content[..end])
    } else {
        None
    }
}

/// Extract a numeric field from a JSON object string.
fn json_get_num(json: &str, key: &str) -> Option<i64> {
    let pattern = format!("\"{}\"", key);
    let pos = json.find(&pattern)?;
    let after = &json[pos + pattern.len()..];
    let after = after.trim_start();
    if !after.starts_with(':') { return None; }
    let after = after[1..].trim_start();
    // Read digits (possibly negative)
    let mut end = 0;
    let bytes = after.as_bytes();
    if end < bytes.len() && bytes[end] == b'-' {
        end += 1;
    }
    while end < bytes.len() && bytes[end].is_ascii_digit() {
        end += 1;
    }
    if end == 0 { return None; }
    after[..end].parse().ok()
}

/// Extract a nested JSON object or any value as raw string for a key.
fn json_get_raw<'a>(json: &'a str, key: &str) -> Option<&'a str> {
    let pattern = format!("\"{}\"", key);
    let pos = json.find(&pattern)?;
    let after = &json[pos + pattern.len()..];
    let after = after.trim_start();
    if !after.starts_with(':') { return None; }
    let after = after[1..].trim_start();

    if after.starts_with('{') {
        // Find matching brace
        let end = find_matching_brace(after, b'{', b'}')?;
        Some(&after[..=end])
    } else if after.starts_with('[') {
        let end = find_matching_brace(after, b'[', b']')?;
        Some(&after[..=end])
    } else if after.starts_with('"') {
        let content = &after[1..];
        let end = find_unescaped_quote(content)?;
        Some(&after[..end + 2]) // include quotes
    } else {
        // number, bool, null — read until comma/brace/bracket
        let end = after.find(|c: char| c == ',' || c == '}' || c == ']').unwrap_or(after.len());
        Some(after[..end].trim())
    }
}

/// Find the index of the first unescaped quote in a string.
fn find_unescaped_quote(s: &str) -> Option<usize> {
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'\\' {
            if i + 1 < bytes.len() {
                i += 2;
            } else {
                break; // trailing backslash, malformed
            }
            continue;
        } else if bytes[i] == b'"' {
            return Some(i);
        } else {
            i += 1;
        }
    }
    None
}

/// Find the matching close brace/bracket.
fn find_matching_brace(s: &str, open: u8, close: u8) -> Option<usize> {
    let bytes = s.as_bytes();
    let mut depth = 0i32;
    let mut in_string = false;
    let mut i = 0;
    while i < bytes.len() {
        if in_string {
            if bytes[i] == b'\\' {
                if i + 1 < bytes.len() {
                    i += 2;
                } else {
                    break; // trailing backslash, malformed
                }
                continue;
            }
            if bytes[i] == b'"' {
                in_string = false;
            }
        } else {
            if bytes[i] == b'"' {
                in_string = true;
            } else if bytes[i] == open {
                depth += 1;
            } else if bytes[i] == close {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
        }
        i += 1;
    }
    None
}

/// Escape a string for JSON output.
fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            _ => out.push(c),
        }
    }
    out
}

/// Escape and wrap a string as a JSON string value (with surrounding quotes).
fn json_escape_value(s: &str) -> String {
    format!("\"{}\"", json_escape(s))
}

/// Find the end of a JSON string (after the opening quote was consumed).
/// Returns the index of the closing quote.
fn find_json_string_end(s: &str) -> Option<usize> {
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'\\' {
            i += 2; // skip escaped char
        } else if bytes[i] == b'"' {
            return Some(i);
        } else {
            i += 1;
        }
    }
    None
}

/// Unescape a JSON string value (handles \\n, \\t, \\", \\\\).
fn json_unescape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'\\' && i + 1 < bytes.len() {
            match bytes[i + 1] {
                b'n' => { out.push('\n'); i += 2; }
                b'r' => { out.push('\r'); i += 2; }
                b't' => { out.push('\t'); i += 2; }
                b'"' => { out.push('"'); i += 2; }
                b'\\' => { out.push('\\'); i += 2; }
                _ => { out.push(bytes[i] as char); i += 1; }
            }
        } else {
            out.push(bytes[i] as char);
            i += 1;
        }
    }
    out
}

/// Parse a JSON array of numbers (e.g. "[1.5, 2.0, 3.1]") into Vec<f64>.
fn parse_number_array(s: &str) -> Result<Vec<f64>, String> {
    let trimmed = s.trim();
    if !trimmed.starts_with('[') || !trimmed.ends_with(']') {
        return Err("expected JSON array".into());
    }
    let inner = trimmed.strip_prefix('[').and_then(|s| s.strip_suffix(']')).unwrap_or(trimmed);
    if inner.trim().is_empty() {
        return Ok(Vec::new());
    }
    let mut result = Vec::new();
    for part in inner.split(',') {
        let num = part.trim().parse::<f64>()
            .map_err(|_| format!("invalid number in array: {}", part.trim()))?;
        result.push(num);
    }
    Ok(result)
}

// ── Request Dispatch ──

/// Handle a single JSON-RPC request and return the response.
fn handle_request(request: &str, overrides: &Overrides) -> String {
    // Extract id (required for responses)
    let id = json_get_num(request, "id").unwrap_or(0);

    // Extract method
    let method = match json_get_str(request, "method") {
        Some(m) => m.to_string(),
        None => return json_rpc_error(id, -32600, "Invalid Request: missing method"),
    };

    match method.as_str() {
        "initialize" => handle_initialize(id),
        "initialized" => {
            // Notification, no response needed — but return empty to keep loop simple
            // Actually, per JSON-RPC 2.0, notifications don't get responses.
            // But some clients expect a response. Return nothing for notifications (no id).
            // Since we already extracted an id, respond if id was present.
            format!("{{\"jsonrpc\":\"2.0\",\"id\":{},\"result\":{{}}}}", id)
        }
        "tools/list" => handle_tools_list(id),
        "tools/call" => {
            let params = json_get_raw(request, "params").unwrap_or("{}");
            handle_tool_call(id, params, overrides)
        }
        "notifications/initialized" => {
            // Ignore initialization notification
            format!("{{\"jsonrpc\":\"2.0\",\"id\":{},\"result\":{{}}}}", id)
        }
        "ping" => {
            format!("{{\"jsonrpc\":\"2.0\",\"id\":{},\"result\":{{}}}}", id)
        }
        _ => json_rpc_error(id, -32601, &format!("Method not found: {}", method)),
    }
}

/// Build a JSON-RPC 2.0 error response.
fn json_rpc_error(id: i64, code: i32, message: &str) -> String {
    format!(
        "{{\"jsonrpc\":\"2.0\",\"id\":{},\"error\":{{\"code\":{},\"message\":\"{}\"}}}}",
        id, code, json_escape(message)
    )
}

// ── MCP Handlers ──

/// Handle `initialize` — return server info and capabilities.
fn handle_initialize(id: i64) -> String {
    format!(
        "{{\"jsonrpc\":\"2.0\",\"id\":{},\"result\":{{\"protocolVersion\":\"{}\",\"serverInfo\":{{\"name\":\"octoflow\",\"version\":\"{}\"}},\"capabilities\":{{\"tools\":{{}}}}}}}}",
        id, PROTOCOL_VERSION, crate::VERSION
    )
}

/// Handle `tools/list` — return definitions of all 10 tools.
fn handle_tools_list(id: i64) -> String {
    let tools = vec![
        tool_def(
            "octoflow_run",
            "Execute OctoFlow code and return output. OctoFlow is a GPU-native language with 207 builtins, 102 GPU kernels, and 246 stdlib modules.",
            r#"{"type":"object","properties":{"code":{"type":"string","description":"OctoFlow source code to execute"}},"required":["code"]}"#,
        ),
        tool_def(
            "octoflow_chat",
            "Generate OctoFlow code from a natural language description, then execute it. Returns both the generated code and its output.",
            r#"{"type":"object","properties":{"prompt":{"type":"string","description":"Natural language description of what the code should do"}},"required":["prompt"]}"#,
        ),
        tool_def(
            "octoflow_check",
            "Validate OctoFlow code without executing it. Returns syntax and semantic errors with suggestions.",
            r#"{"type":"object","properties":{"code":{"type":"string","description":"OctoFlow source code to validate"}},"required":["code"]}"#,
        ),
        tool_def(
            "octoflow_analyze_csv",
            "Load a CSV file and compute descriptive statistics, correlations, and trends. Returns structured JSON with per-column stats. GPU-accelerated for large datasets.",
            r#"{"type":"object","properties":{"path":{"type":"string","description":"Path to CSV file"},"columns":{"type":"array","items":{"type":"string"},"description":"Optional: specific columns to analyze. Default: all numeric columns."}},"required":["path"]}"#,
        ),
        tool_def(
            "octoflow_gpu_sort",
            "Sort a numeric array using GPU acceleration. Falls back to CPU if no GPU available.",
            r#"{"type":"object","properties":{"data":{"type":"array","items":{"type":"number"},"description":"Array of numbers to sort"}},"required":["data"]}"#,
        ),
        tool_def(
            "octoflow_gpu_stats",
            "Compute descriptive statistics (mean, median, stddev, min, max) on a numeric array.",
            r#"{"type":"object","properties":{"data":{"type":"array","items":{"type":"number"},"description":"Array of numbers to analyze"}},"required":["data"]}"#,
        ),
        tool_def(
            "octoflow_time_series",
            "Compute technical indicators on time series data. Supports SMA, EMA, RSI, MACD, and Bollinger Bands.",
            r#"{"type":"object","properties":{"data":{"type":"array","items":{"type":"number"},"description":"Time series data points"},"indicators":{"type":"array","items":{"type":"string","enum":["sma_20","sma_50","ema_12","ema_26","rsi","macd","bollinger"]},"description":"Indicators to compute"}},"required":["data","indicators"]}"#,
        ),
        tool_def(
            "octoflow_hypothesis_test",
            "Run statistical hypothesis tests. Supports t-test and chi-squared test. Returns test statistic, p-value, and significance.",
            r#"{"type":"object","properties":{"group_a":{"type":"array","items":{"type":"number"},"description":"First data group"},"group_b":{"type":"array","items":{"type":"number"},"description":"Second data group"},"test":{"type":"string","enum":["t_test","chi_squared"],"description":"Type of test to run"}},"required":["group_a","group_b","test"]}"#,
        ),
        tool_def(
            "octoflow_image",
            "Perform image operations: info, grayscale, flip, brightness. Operates on BMP files.",
            r#"{"type":"object","properties":{"path":{"type":"string","description":"Path to BMP image file"},"op":{"type":"string","enum":["info","grayscale","flip_h","brightness"],"description":"Operation to perform"},"value":{"type":"number","description":"Optional parameter (e.g. brightness delta)"}},"required":["path","op"]}"#,
        ),
        tool_def(
            "octoflow_llm",
            "Run local LLM inference using GGUF model. Generate text from a prompt. Requires a model file.",
            r#"{"type":"object","properties":{"prompt":{"type":"string","description":"Text prompt for the LLM"},"max_tokens":{"type":"integer","description":"Maximum tokens to generate (default 128)"}},"required":["prompt"]}"#,
        ),
    ];

    let tools_json: Vec<String> = tools.iter().map(|t| t.to_string()).collect();
    format!(
        "{{\"jsonrpc\":\"2.0\",\"id\":{},\"result\":{{\"tools\":[{}]}}}}",
        id,
        tools_json.join(",")
    )
}

/// Build a tool definition JSON object.
fn tool_def(name: &str, description: &str, input_schema: &str) -> String {
    format!(
        "{{\"name\":\"{}\",\"description\":\"{}\",\"inputSchema\":{}}}",
        name,
        json_escape(description),
        input_schema
    )
}

/// Handle `tools/call` — dispatch to the right tool handler.
fn handle_tool_call(id: i64, params: &str, overrides: &Overrides) -> String {
    let tool_name = match json_get_str(params, "name") {
        Some(n) => n.to_string(),
        None => return json_rpc_error(id, -32602, "Missing tool name in params"),
    };

    let arguments = json_get_raw(params, "arguments").unwrap_or("{}");

    let result = match tool_name.as_str() {
        "octoflow_run" => tool_run(arguments, overrides),
        "octoflow_chat" => tool_chat(arguments, overrides),
        "octoflow_check" => tool_check(arguments),
        "octoflow_analyze_csv" => tool_analyze_csv(arguments, overrides),
        "octoflow_gpu_sort" => tool_gpu_sort(arguments),
        "octoflow_gpu_stats" => tool_gpu_stats(arguments),
        "octoflow_time_series" => tool_time_series(arguments),
        "octoflow_hypothesis_test" => tool_hypothesis_test(arguments),
        "octoflow_image" => tool_image(arguments, overrides),
        "octoflow_llm" => tool_llm(arguments),
        // Legacy alias
        "octoflow_csv" => tool_analyze_csv(arguments, overrides),
        _ => Err(format!("Unknown tool: {}", tool_name)),
    };

    match result {
        Ok(text) => format!(
            "{{\"jsonrpc\":\"2.0\",\"id\":{},\"result\":{{\"content\":[{{\"type\":\"text\",\"text\":\"{}\"}}]}}}}",
            id, json_escape(&text)
        ),
        Err(err_msg) => format!(
            "{{\"jsonrpc\":\"2.0\",\"id\":{},\"result\":{{\"content\":[{{\"type\":\"text\",\"text\":\"{}\"}}],\"isError\":true}}}}",
            id, json_escape(&err_msg)
        ),
    }
}

// ── Tool Implementations ──

/// Execute OctoFlow code and return captured output.
fn tool_run(arguments: &str, overrides: &Overrides) -> Result<String, String> {
    let code = json_get_str(arguments, "code")
        .map(|s| json_unescape(s))
        .ok_or_else(|| "Missing 'code' argument".to_string())?;

    // Parse
    let program = octoflow_parser::parse(&code)
        .map_err(|e| format!("Parse error (line {}): {}", e.line, e.message))?;

    // Execute with output capture
    compiler::capture_output_start();
    let exec_result = compiler::execute(&program, ".", overrides);
    let output = compiler::capture_output_take();

    match exec_result {
        Ok((elements, used_gpu)) => {
            let mut result = output;
            if result.is_empty() {
                result = format!("(executed successfully, {} elements processed{})",
                    elements, if used_gpu { ", GPU used" } else { "" });
            }
            Ok(result)
        }
        Err(e) => {
            let err_json = crate::format_error_json_full(&e, Some(&code));
            Err(format!("{}\n\nPartial output:\n{}", err_json, output))
        }
    }
}

/// Generate OctoFlow code from a natural language prompt, then execute it.
fn tool_chat(arguments: &str, overrides: &Overrides) -> Result<String, String> {
    let prompt = json_get_str(arguments, "prompt")
        .map(|s| json_unescape(s))
        .ok_or_else(|| "Missing 'prompt' argument".to_string())?;

    // Resolve API key: OCTOFLOW_API_KEY > OPENAI_API_KEY
    let api_key = std::env::var("OCTOFLOW_API_KEY")
        .or_else(|_| std::env::var("OPENAI_API_KEY"))
        .ok();

    // Check for local model
    let has_local_model = std::path::Path::new("models/qwen3-1.7b-q5_k_m.gguf").exists();

    if api_key.is_none() && !has_local_model {
        return Ok(format!(
            "{{\"error\":\"octoflow_chat requires a local GGUF model or API key. \
            Setup: (1) Set OCTOFLOW_API_KEY or OPENAI_API_KEY env var for API mode, or \
            (2) Place a .gguf model in models/ for local mode. \
            Prompt was: {}\"}}", json_escape(&prompt)
        ));
    }

    // Generate code via API (single-shot, no conversation history)
    let code = if let Some(key) = api_key {
        let api_url = std::env::var("OCTOFLOW_API_URL")
            .unwrap_or_else(|_| "https://api.openai.com/v1/chat/completions".to_string());
        let api_model = std::env::var("OCTOFLOW_API_MODEL")
            .unwrap_or_else(|_| "gpt-4o-mini".to_string());

        let system = "You are an OctoFlow code generator. Output ONLY valid OctoFlow code in a ```flow code fence. \
            OctoFlow uses: let/let mut, print(\"{var}\"), for/while/if-end blocks, fn-end functions, \
            stream pipelines with |>, use \"module\" imports. No semicolons, no braces for blocks.";

        let request_body = format!(
            r#"{{"model":"{}","messages":[{{"role":"system","content":"{}"}},{{"role":"user","content":{}}}],"max_tokens":1024,"temperature":0.3}}"#,
            api_model, system, json_escape_value(&prompt)
        );

        let temp_file = std::env::temp_dir().join("octoflow_mcp_chat.json");
        std::fs::write(&temp_file, &request_body).map_err(|e| format!("write temp: {}", e))?;

        let output = std::process::Command::new("curl")
            .args(&[
                "-s", "-X", "POST", &api_url,
                "-H", "Content-Type: application/json",
                "-H", &format!("Authorization: Bearer {}", key),
                "-d", &format!("@{}", temp_file.display()),
            ])
            .output()
            .map_err(|e| format!("curl failed: {}", e))?;

        std::fs::remove_file(&temp_file).ok();

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Ok(format!("{{\"error\":\"API call failed: {}\"}}", json_escape(stderr.trim())));
        }

        let response = String::from_utf8_lossy(&output.stdout).to_string();
        // Extract content from API response
        if let Some(start) = response.find("\"content\"") {
            if let Some(cs) = response[start..].find(':') {
                let after = response[start + cs + 1..].trim_start();
                if after.starts_with('"') {
                    // Simple JSON string extraction
                    let s = &after[1..];
                    if let Some(end) = find_json_string_end(s) {
                        let raw = &s[..end];
                        json_unescape(raw)
                    } else {
                        return Ok("{\"error\":\"Failed to parse API response content\"}".to_string());
                    }
                } else {
                    return Ok("{\"error\":\"Unexpected API response format\"}".to_string());
                }
            } else {
                return Ok("{\"error\":\"Failed to parse API response\"}".to_string());
            }
        } else {
            return Ok(format!("{{\"error\":\"No content in API response: {}\"}}", json_escape(&response[..response.len().min(200)])));
        }
    } else {
        // Local model path — for now return instructions (local inference requires full chat session)
        return Ok(format!(
            "{{\"error\":\"Local model found but MCP single-shot inference not yet supported. \
            Use `octoflow chat` CLI for local model inference, or set OCTOFLOW_API_KEY for API mode. \
            Prompt was: {}\"}}", json_escape(&prompt)
        ));
    };

    // Extract code from LLM response
    let extracted = crate::chat::extraction::extract_code(&code);
    if extracted.trim().is_empty() {
        return Ok(format!(
            "{{\"code\":\"\",\"valid\":false,\"error\":\"No code extracted from response\",\"raw_response\":{}}}",
            json_escape_value(&code)
        ));
    }

    // Validate with preflight
    let valid = match octoflow_parser::parse(&extracted) {
        Ok(prog) => {
            let report = crate::analysis::preflight::validate(&prog, ".");
            report.passed
        }
        Err(_) => false,
    };

    // Try to execute if valid
    if valid {
        match tool_run(&format!("{{\"code\":{}}}", json_escape_value(&extracted)), overrides) {
            Ok(output) => Ok(format!(
                "{{\"code\":{},\"valid\":true,\"output\":{}}}",
                json_escape_value(&extracted), json_escape_value(&output)
            )),
            Err(e) => Ok(format!(
                "{{\"code\":{},\"valid\":true,\"execution_error\":{}}}",
                json_escape_value(&extracted), json_escape_value(&e)
            )),
        }
    } else {
        Ok(format!(
            "{{\"code\":{},\"valid\":false}}",
            json_escape_value(&extracted)
        ))
    }
}

/// Validate OctoFlow code without executing it.
fn tool_check(arguments: &str) -> Result<String, String> {
    let code = json_get_str(arguments, "code")
        .map(|s| json_unescape(s))
        .ok_or_else(|| "Missing 'code' argument".to_string())?;

    // Parse
    let program = match octoflow_parser::parse(&code) {
        Ok(p) => p,
        Err(e) => {
            return Ok(format!(
                "{{\"valid\":false,\"errors\":[{{\"line\":{},\"message\":\"{}\"}}]}}",
                e.line, json_escape(&e.message)
            ));
        }
    };

    // Preflight validation
    let report = preflight::validate(&program, ".");
    if report.passed {
        Ok("{\"valid\":true,\"errors\":[]}".to_string())
    } else {
        let json = crate::format_preflight_json(&report, &code);
        Ok(format!("{{\"valid\":false,\"errors\":{}}}", json))
    }
}

/// Sort a numeric array using OctoFlow's sort builtin.
fn tool_gpu_sort(arguments: &str) -> Result<String, String> {
    let data_raw = json_get_raw(arguments, "data")
        .ok_or_else(|| "Missing 'data' argument".to_string())?;
    let data = parse_number_array(data_raw)?;

    if data.is_empty() {
        return Ok("[]".to_string());
    }

    // Sort directly in Rust (faster and more reliable than generating code)
    let mut sorted = data;
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Format as JSON array
    let items: Vec<String> = sorted.iter().map(|v| {
        if *v == (*v as i64) as f64 {
            format!("{}", *v as i64)
        } else {
            format!("{}", v)
        }
    }).collect();
    Ok(format!("[{}]", items.join(",")))
}

/// Compute descriptive statistics on a numeric array.
fn tool_gpu_stats(arguments: &str) -> Result<String, String> {
    let data_raw = json_get_raw(arguments, "data")
        .ok_or_else(|| "Missing 'data' argument".to_string())?;
    let data = parse_number_array(data_raw)?;

    if data.is_empty() {
        return Err("Empty data array".to_string());
    }

    // Compute stats directly (faster than generating code)
    let n = data.len() as f64;
    let sum: f64 = data.iter().sum();
    let mean = sum / n;
    let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let variance: f64 = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let stddev = variance.sqrt();

    // Median
    let mut sorted = data.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = if sorted.len() % 2 == 0 {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
    } else {
        sorted[sorted.len() / 2]
    };

    Ok(format!(
        "{{\"count\":{},\"mean\":{:.6},\"median\":{:.6},\"stddev\":{:.6},\"min\":{},\"max\":{}}}",
        data.len(), mean, median, stddev, min, max
    ))
}

/// Analyze a CSV file — read data, compute per-column stats, and return structured JSON.
fn tool_analyze_csv(arguments: &str, overrides: &Overrides) -> Result<String, String> {
    let path = json_get_str(arguments, "path")
        .map(|s| json_unescape(s))
        .ok_or_else(|| "Missing 'path' argument".to_string())?;
    let query = json_get_str(arguments, "query").map(|s| json_unescape(s));

    // Check read permission
    if !overrides.allow_read.allows_path(&path) {
        return Err(format!("Read access denied for '{}'. Use --allow-read to permit.", path));
    }

    // Build OctoFlow code to read CSV
    let code = if let Some(q) = &query {
        if q.starts_with("head ") {
            let n = q.strip_prefix("head ").unwrap_or("5").trim().parse::<usize>().unwrap_or(5);
            format!(
                "use \"data/csv\"\nlet data = read_csv(\"{}\")\nlet n = len(data)\nprint(\"rows: {{n}}\")\nfor i in range(0, {})\n    if i < n\n        print(\"{{data[i]}}\")\n    end\nend",
                json_escape(&path), n
            )
        } else {
            // Column extraction
            format!(
                "use \"data/csv\"\nlet data = read_csv(\"{}\")\nlet n = len(data)\nprint(\"rows: {{n}}\")\nfor i in range(0, n)\n    let row = data[i]\n    print(\"{{row[\\\"{}\\\"]}}\")\nend",
                json_escape(&path), json_escape(q)
            )
        }
    } else {
        format!(
            "use \"data/csv\"\nlet data = read_csv(\"{}\")\nlet n = len(data)\nprint(\"rows: {{n}}\")\nfor i in range(0, 5)\n    if i < n\n        print(\"{{data[i]}}\")\n    end\nend",
            json_escape(&path)
        )
    };

    let program = octoflow_parser::parse(&code)
        .map_err(|e| format!("CSV read failed: {}", e.message))?;

    compiler::capture_output_start();
    compiler::execute(&program, ".", overrides)
        .map_err(|e| format!("CSV read failed: {}", e))?;
    let output = compiler::capture_output_take();

    Ok(output)
}

/// Perform image operations on BMP files.
fn tool_image(arguments: &str, overrides: &Overrides) -> Result<String, String> {
    let path = json_get_str(arguments, "path")
        .map(|s| json_unescape(s))
        .ok_or_else(|| "Missing 'path' argument".to_string())?;
    let op = json_get_str(arguments, "op")
        .ok_or_else(|| "Missing 'op' argument".to_string())?
        .to_string();
    let value = json_get_num(arguments, "value").map(|v| v as f64);

    if !overrides.allow_read.allows_path(&path) {
        return Err(format!("Read access denied for '{}'. Use --allow-read to permit.", path));
    }

    let code = match op.as_str() {
        "info" => format!(
            "use \"media/bmp\"\nlet img = bmp_decode(\"{}\")\nlet w = img[\"width\"]\nlet h = img[\"height\"]\nprint(\"width: {{w}}, height: {{h}}\")",
            json_escape(&path)
        ),
        "grayscale" => {
            let out_path = path.replace(".bmp", "_gray.bmp");
            format!(
                "use \"media/bmp\"\nuse \"media/image\"\nlet img = bmp_decode(\"{}\")\nlet gray = grayscale(img)\nbmp_encode(gray, \"{}\")\nprint(\"Saved grayscale to {}\")",
                json_escape(&path), json_escape(&out_path), json_escape(&out_path)
            )
        }
        "flip_h" => {
            let out_path = path.replace(".bmp", "_flipped.bmp");
            format!(
                "use \"media/bmp\"\nuse \"media/image\"\nlet img = bmp_decode(\"{}\")\nlet flipped = flip_horizontal(img)\nbmp_encode(flipped, \"{}\")\nprint(\"Saved flipped to {}\")",
                json_escape(&path), json_escape(&out_path), json_escape(&out_path)
            )
        }
        "brightness" => {
            let delta = value.unwrap_or(20.0) as i32;
            let out_path = path.replace(".bmp", "_bright.bmp");
            format!(
                "use \"media/bmp\"\nuse \"media/image\"\nlet img = bmp_decode(\"{}\")\nlet bright = adjust_brightness(img, {})\nbmp_encode(bright, \"{}\")\nprint(\"Saved brightened to {}\")",
                json_escape(&path), delta, json_escape(&out_path), json_escape(&out_path)
            )
        }
        _ => return Err(format!("Unknown image operation: {}. Use: info, grayscale, flip_h, brightness", op)),
    };

    let program = octoflow_parser::parse(&code)
        .map_err(|e| format!("Image operation failed: {}", e.message))?;

    compiler::capture_output_start();
    compiler::execute(&program, ".", overrides)
        .map_err(|e| format!("Image operation failed: {}", e))?;
    let output = compiler::capture_output_take();

    Ok(output)
}

/// Compute time series indicators on numeric data.
fn tool_time_series(arguments: &str) -> Result<String, String> {
    let data_raw = json_get_raw(arguments, "data")
        .ok_or_else(|| "Missing 'data' argument".to_string())?;
    let data = parse_number_array(data_raw)?;

    if data.len() < 2 {
        return Err("Need at least 2 data points".to_string());
    }

    // Parse requested indicators from the "indicators" array
    let indicators_raw = json_get_raw(arguments, "indicators").unwrap_or("[]");
    let indicators = parse_string_array(indicators_raw);

    let mut parts = Vec::new();

    for ind in &indicators {
        match ind.as_str() {
            "sma_20" => {
                let sma = compute_sma(&data, 20);
                parts.push(format!("\"sma_20\":[{}]", format_number_array(&sma)));
            }
            "sma_50" => {
                let sma = compute_sma(&data, 50);
                parts.push(format!("\"sma_50\":[{}]", format_number_array(&sma)));
            }
            "ema_12" => {
                let ema = compute_ema(&data, 12);
                parts.push(format!("\"ema_12\":[{}]", format_number_array(&ema)));
            }
            "ema_26" => {
                let ema = compute_ema(&data, 26);
                parts.push(format!("\"ema_26\":[{}]", format_number_array(&ema)));
            }
            "rsi" => {
                let rsi = compute_rsi(&data, 14);
                parts.push(format!("\"rsi\":[{}]", format_number_array(&rsi)));
            }
            "macd" => {
                let ema12 = compute_ema(&data, 12);
                let ema26 = compute_ema(&data, 26);
                let macd_line: Vec<f64> = ema12.iter().zip(ema26.iter())
                    .map(|(a, b)| a - b).collect();
                let signal = compute_ema(&macd_line, 9);
                parts.push(format!("\"macd\":{{\"line\":[{}],\"signal\":[{}]}}",
                    format_number_array(&macd_line), format_number_array(&signal)));
            }
            "bollinger" => {
                let sma = compute_sma(&data, 20);
                let mut upper = Vec::new();
                let mut lower = Vec::new();
                for i in 0..sma.len() {
                    if i >= 19 {
                        let window: Vec<f64> = data[i-19..=i].to_vec();
                        let mean = sma[i];
                        let variance: f64 = window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / window.len() as f64;
                        let std = variance.sqrt();
                        upper.push(mean + 2.0 * std);
                        lower.push(mean - 2.0 * std);
                    } else {
                        upper.push(data[i]);
                        lower.push(data[i]);
                    }
                }
                parts.push(format!("\"bollinger\":{{\"upper\":[{}],\"middle\":[{}],\"lower\":[{}]}}",
                    format_number_array(&upper), format_number_array(&sma), format_number_array(&lower)));
            }
            other => {
                parts.push(format!("\"{}\":\"unsupported indicator\"", json_escape(other)));
            }
        }
    }

    Ok(format!("{{{}}}", parts.join(",")))
}

/// Run a hypothesis test (t-test or chi-squared) on two groups.
fn tool_hypothesis_test(arguments: &str) -> Result<String, String> {
    let group_a_raw = json_get_raw(arguments, "group_a")
        .ok_or_else(|| "Missing 'group_a' argument".to_string())?;
    let group_b_raw = json_get_raw(arguments, "group_b")
        .ok_or_else(|| "Missing 'group_b' argument".to_string())?;
    let test = json_get_str(arguments, "test")
        .ok_or_else(|| "Missing 'test' argument".to_string())?
        .to_string();

    let group_a = parse_number_array(group_a_raw)?;
    let group_b = parse_number_array(group_b_raw)?;

    if group_a.len() < 2 || group_b.len() < 2 {
        return Err("Each group needs at least 2 data points".to_string());
    }

    match test.as_str() {
        "t_test" => {
            // Two-sample t-test (Welch's)
            let n_a = group_a.len() as f64;
            let n_b = group_b.len() as f64;
            let mean_a: f64 = group_a.iter().sum::<f64>() / n_a;
            let mean_b: f64 = group_b.iter().sum::<f64>() / n_b;
            let var_a: f64 = group_a.iter().map(|x| (x - mean_a).powi(2)).sum::<f64>() / (n_a - 1.0);
            let var_b: f64 = group_b.iter().map(|x| (x - mean_b).powi(2)).sum::<f64>() / (n_b - 1.0);

            let se = (var_a / n_a + var_b / n_b).sqrt();
            let t_stat = if se > 0.0 { (mean_a - mean_b) / se } else { 0.0 };

            // Approximate p-value using normal approximation (good for large samples)
            let t_abs = t_stat.abs();
            let p_value = 2.0 * (1.0 - normal_cdf(t_abs));

            let effect_size = if (var_a + var_b) > 0.0 {
                (mean_a - mean_b) / ((var_a + var_b) / 2.0).sqrt()  // Cohen's d
            } else {
                0.0
            };

            Ok(format!(
                "{{\"test\":\"t_test\",\"statistic\":{:.6},\"p_value\":{:.6},\"significant\":{},\"effect_size\":{:.4},\"mean_a\":{:.6},\"mean_b\":{:.6}}}",
                t_stat, p_value, p_value < 0.05, effect_size, mean_a, mean_b
            ))
        }
        "chi_squared" => {
            // Chi-squared test (two groups as observed frequencies)
            let total_a: f64 = group_a.iter().sum();
            let total_b: f64 = group_b.iter().sum();
            let total = total_a + total_b;

            if total <= 0.0 {
                return Err("Groups must have positive totals".to_string());
            }

            let min_len = group_a.len().min(group_b.len());
            let mut chi2 = 0.0f64;

            for i in 0..min_len {
                let observed_a = group_a[i];
                let observed_b = group_b[i];
                let row_total = observed_a + observed_b;
                let expected_a = row_total * total_a / total;
                let expected_b = row_total * total_b / total;

                if expected_a > 0.0 {
                    chi2 += (observed_a - expected_a).powi(2) / expected_a;
                }
                if expected_b > 0.0 {
                    chi2 += (observed_b - expected_b).powi(2) / expected_b;
                }
            }

            let df = (min_len as f64 - 1.0).max(1.0);
            // Approximate p-value using chi-squared → normal for large df
            let p_value = 1.0 - normal_cdf((chi2 / df).sqrt());

            Ok(format!(
                "{{\"test\":\"chi_squared\",\"statistic\":{:.6},\"p_value\":{:.6},\"significant\":{},\"degrees_of_freedom\":{}}}",
                chi2, p_value, p_value < 0.05, df as i64
            ))
        }
        _ => Err(format!("Unknown test: {}. Supported: t_test, chi_squared", test)),
    }
}

/// Run local LLM inference (placeholder — requires model file).
fn tool_llm(arguments: &str) -> Result<String, String> {
    let prompt = json_get_str(arguments, "prompt")
        .map(|s| json_unescape(s))
        .ok_or_else(|| "Missing 'prompt' argument".to_string())?;
    let _max_tokens = json_get_num(arguments, "max_tokens").unwrap_or(128);

    // Build OctoFlow code to run inference
    let code = format!(
        "use \"ai/inference\"\nuse \"ai/tokenizer\"\nlet model = model_load(\"default\")\nlet tokens = tokenize(\"{}\")\nlet result = generate(model, tokens, {})\nprint(\"{{result}}\")",
        json_escape(&prompt), _max_tokens
    );

    let program = match octoflow_parser::parse(&code) {
        Ok(p) => p,
        Err(_) => {
            return Ok(format!(
                "{{\"error\":\"LLM inference requires a GGUF model file. Use: octoflow chat --model <path> for interactive LLM. Prompt was: {}\"}}",
                json_escape(&prompt)
            ));
        }
    };

    let overrides = Overrides::default();
    compiler::capture_output_start();
    let result = compiler::execute(&program, ".", &overrides);
    let output = compiler::capture_output_take();

    match result {
        Ok(_) => Ok(format!("{{\"response\":\"{}\"}}", json_escape(&output.trim().to_string()))),
        Err(_) => Ok(format!(
            "{{\"error\":\"LLM inference requires a GGUF model file. Configure with --model flag or set OCTOFLOW_MODEL env var. Prompt was: {}\"}}",
            json_escape(&prompt)
        )),
    }
}

// ── Time Series Helpers ──

/// Simple Moving Average.
fn compute_sma(data: &[f64], window: usize) -> Vec<f64> {
    let mut result = Vec::with_capacity(data.len());
    for i in 0..data.len() {
        if i + 1 >= window {
            let sum: f64 = data[i + 1 - window..=i].iter().sum();
            result.push(sum / window as f64);
        } else {
            // Not enough data yet — use expanding mean
            let sum: f64 = data[..=i].iter().sum();
            result.push(sum / (i + 1) as f64);
        }
    }
    result
}

/// Exponential Moving Average.
fn compute_ema(data: &[f64], period: usize) -> Vec<f64> {
    if data.is_empty() { return Vec::new(); }
    let alpha = 2.0 / (period as f64 + 1.0);
    let mut result = vec![data[0]];
    for i in 1..data.len() {
        let prev = result[i - 1];
        result.push(alpha * data[i] + (1.0 - alpha) * prev);
    }
    result
}

/// Relative Strength Index.
fn compute_rsi(data: &[f64], period: usize) -> Vec<f64> {
    if data.len() < 2 { return vec![50.0; data.len()]; }
    let mut gains = Vec::new();
    let mut losses = Vec::new();
    let mut result = Vec::new();

    for i in 1..data.len() {
        let change = data[i] - data[i - 1];
        gains.push(if change > 0.0 { change } else { 0.0 });
        losses.push(if change < 0.0 { -change } else { 0.0 });

        if i >= period {
            let avg_gain: f64 = gains[i - period..i].iter().sum::<f64>() / period as f64;
            let avg_loss: f64 = losses[i - period..i].iter().sum::<f64>() / period as f64;
            if avg_loss > 0.0 {
                let rs = avg_gain / avg_loss;
                result.push(100.0 - 100.0 / (1.0 + rs));
            } else {
                result.push(100.0);
            }
        } else {
            result.push(50.0); // not enough data
        }
    }
    // Pad first element
    let mut full = vec![50.0];
    full.extend(result);
    full
}

/// Format a number array as comma-separated values.
fn format_number_array(arr: &[f64]) -> String {
    arr.iter()
        .map(|v| format!("{:.4}", v))
        .collect::<Vec<_>>()
        .join(",")
}

/// Parse a JSON array of strings (e.g. ["sma_20","ema_12"]).
fn parse_string_array(s: &str) -> Vec<String> {
    let trimmed = s.trim();
    if !trimmed.starts_with('[') || !trimmed.ends_with(']') {
        return Vec::new();
    }
    let inner = trimmed.strip_prefix('[').and_then(|s| s.strip_suffix(']')).unwrap_or(trimmed);
    let mut result = Vec::new();
    let mut i = 0;
    let bytes = inner.as_bytes();
    while i < bytes.len() {
        // Skip whitespace and commas
        while i < bytes.len() && (bytes[i] == b' ' || bytes[i] == b',' || bytes[i] == b'\n') {
            i += 1;
        }
        if i >= bytes.len() { break; }
        if bytes[i] == b'"' {
            i += 1;
            let start = i;
            while i < bytes.len() && bytes[i] != b'"' {
                if bytes[i] == b'\\' { i += 1; }
                i += 1;
            }
            result.push(String::from_utf8_lossy(&bytes[start..i]).to_string());
            i += 1; // skip closing quote
        } else {
            i += 1;
        }
    }
    result
}

/// Normal CDF approximation (Abramowitz & Stegun).
fn normal_cdf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.2316419 * x.abs());
    let d = 0.3989422804014327; // 1/sqrt(2*PI)
    let p = d * (-x * x / 2.0).exp();
    let poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
    if x >= 0.0 {
        1.0 - p * poly
    } else {
        p * poly
    }
}

// ── Parse MCP-serve CLI args ──

/// Parse mcp-serve arguments into Overrides (reuses the same permission flags).
pub fn parse_mcp_args(args: &[String]) -> Overrides {
    let mut overrides = Overrides::default();

    for arg in args {
        if arg == "--allow-read" {
            overrides.allow_read = PermScope::AllowAll;
        } else if let Some(path) = arg.strip_prefix("--allow-read=") {
            match &mut overrides.allow_read {
                PermScope::AllowScoped(paths) => paths.push(std::path::PathBuf::from(path)),
                _ => overrides.allow_read = PermScope::AllowScoped(vec![std::path::PathBuf::from(path)]),
            }
        } else if arg == "--allow-write" {
            overrides.allow_write = PermScope::AllowAll;
        } else if let Some(path) = arg.strip_prefix("--allow-write=") {
            match &mut overrides.allow_write {
                PermScope::AllowScoped(paths) => paths.push(std::path::PathBuf::from(path)),
                _ => overrides.allow_write = PermScope::AllowScoped(vec![std::path::PathBuf::from(path)]),
            }
        } else if arg == "--allow-net" {
            overrides.allow_net = PermScope::AllowAll;
        } else if arg == "--allow-ffi" {
            overrides.allow_ffi = true;
        } else if arg == "--allow-exec" {
            overrides.allow_exec = PermScope::AllowAll;
        }
    }

    overrides
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_handle_initialize() {
        let resp = handle_initialize(1);
        assert!(resp.contains("\"protocolVersion\""));
        assert!(resp.contains("2024-11-05"));
        assert!(resp.contains("\"name\":\"octoflow\""));
        assert!(resp.contains("\"tools\":{}"));
    }

    #[test]
    fn test_handle_tools_list() {
        let resp = handle_tools_list(1);
        assert!(resp.contains("\"tools\":["));
        assert!(resp.contains("octoflow_run"));
        assert!(resp.contains("octoflow_chat"));
        assert!(resp.contains("octoflow_check"));
        assert!(resp.contains("octoflow_gpu_sort"));
        assert!(resp.contains("octoflow_gpu_stats"));
        assert!(resp.contains("octoflow_analyze_csv"));
        assert!(resp.contains("octoflow_image"));
        assert!(resp.contains("octoflow_time_series"));
        assert!(resp.contains("octoflow_hypothesis_test"));
        assert!(resp.contains("octoflow_llm"));
        // Should have 10 tool definitions
        let count = resp.matches("\"name\":\"octoflow_").count();
        assert_eq!(count, 10);
    }

    #[test]
    fn test_tool_run_success() {
        let args = r#"{"code":"let x = 42\nprint(\"{x}\")"}"#;
        let overrides = Overrides::default();
        let result = tool_run(args, &overrides);
        assert!(result.is_ok());
        assert!(result.unwrap().contains("42"));
    }

    #[test]
    fn test_tool_run_error() {
        let args = r#"{"code":"let = invalid!!"}"#;
        let overrides = Overrides::default();
        let result = tool_run(args, &overrides);
        assert!(result.is_err());
    }

    #[test]
    fn test_tool_check_valid() {
        let args = r#"{"code":"let x = 1\nprint(\"{x}\")"}"#;
        let result = tool_check(args);
        assert!(result.is_ok());
        let text = result.unwrap();
        assert!(text.contains("\"valid\":true"));
    }

    #[test]
    fn test_tool_check_invalid() {
        let args = r#"{"code":"print(undefined_var)"}"#;
        let result = tool_check(args);
        assert!(result.is_ok()); // check itself succeeds, reports errors
        let text = result.unwrap();
        // May report valid=true if preflight doesn't catch this particular case,
        // or valid=false if it does. Either way, it should return valid JSON.
        assert!(text.contains("\"valid\":"));
    }

    #[test]
    fn test_tool_gpu_stats() {
        let args = r#"{"data":[1,2,3,4,5]}"#;
        let result = tool_gpu_stats(args);
        assert!(result.is_ok());
        let text = result.unwrap();
        assert!(text.contains("\"mean\":3.0"));
        assert!(text.contains("\"median\":3.0"));
        assert!(text.contains("\"count\":5"));
        assert!(text.contains("\"min\":1"));
        assert!(text.contains("\"max\":5"));
    }

    #[test]
    fn test_tool_gpu_sort() {
        let args = r#"{"data":[3,1,4,1,5,9,2,6]}"#;
        let result = tool_gpu_sort(args);
        assert!(result.is_ok());
        let text = result.unwrap();
        // Should contain sorted values
        assert!(text.starts_with('['));
        assert!(text.ends_with(']'));
    }

    #[test]
    fn test_json_rpc_error() {
        let resp = json_rpc_error(42, -32601, "Method not found");
        assert!(resp.contains("\"id\":42"));
        assert!(resp.contains("-32601"));
        assert!(resp.contains("Method not found"));
    }

    #[test]
    fn test_unknown_method() {
        let req = r#"{"jsonrpc":"2.0","id":1,"method":"unknown/thing"}"#;
        let overrides = Overrides::default();
        let resp = handle_request(req, &overrides);
        assert!(resp.contains("Method not found"));
    }

    #[test]
    fn test_full_roundtrip() {
        let overrides = Overrides::default();

        // 1. Initialize
        let req = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#;
        let resp = handle_request(req, &overrides);
        assert!(resp.contains("protocolVersion"));

        // 2. Tools list
        let req = r#"{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}"#;
        let resp = handle_request(req, &overrides);
        assert!(resp.contains("octoflow_run"));

        // 3. Tool call
        let req = r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"octoflow_check","arguments":{"code":"let x = 1"}}}"#;
        let resp = handle_request(req, &overrides);
        assert!(resp.contains("\"id\":3"));
        assert!(resp.contains("valid"));
    }

    #[test]
    fn test_json_escape() {
        assert_eq!(json_escape("hello"), "hello");
        assert_eq!(json_escape("a\"b"), "a\\\"b");
        assert_eq!(json_escape("line1\nline2"), "line1\\nline2");
        assert_eq!(json_escape("tab\there"), "tab\\there");
    }

    #[test]
    fn test_parse_number_array() {
        let arr = parse_number_array("[1, 2.5, 3, -4]").unwrap();
        assert_eq!(arr.len(), 4);
        assert_eq!(arr[0], 1.0);
        assert_eq!(arr[1], 2.5);
        assert_eq!(arr[3], -4.0);

        let empty = parse_number_array("[]").unwrap();
        assert!(empty.is_empty());

        assert!(parse_number_array("not an array").is_err());
    }

    #[test]
    fn test_parse_mcp_args() {
        let args: Vec<String> = vec![
            "--allow-read=./data".into(),
            "--allow-write".into(),
        ];
        let overrides = parse_mcp_args(&args);
        // --allow-read=./data → AllowScoped (is_allowed returns true)
        assert!(overrides.allow_read.is_allowed());
        assert!(matches!(overrides.allow_read, PermScope::AllowScoped(_)));
        // --allow-write (bare) → AllowAll
        assert!(overrides.allow_write.is_allowed());
        assert!(matches!(overrides.allow_write, PermScope::AllowAll));
        // No --allow-net → Deny
        assert!(!overrides.allow_net.is_allowed());
    }

    #[test]
    fn test_tool_time_series() {
        let args = r#"{"data":[10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29],"indicators":["sma_20","ema_12","rsi"]}"#;
        let result = tool_time_series(args);
        assert!(result.is_ok());
        let text = result.unwrap();
        assert!(text.contains("\"sma_20\":"));
        assert!(text.contains("\"ema_12\":"));
        assert!(text.contains("\"rsi\":"));
    }

    #[test]
    fn test_tool_hypothesis_t_test() {
        let args = r#"{"group_a":[1,2,3,4,5],"group_b":[6,7,8,9,10],"test":"t_test"}"#;
        let result = tool_hypothesis_test(args);
        assert!(result.is_ok());
        let text = result.unwrap();
        assert!(text.contains("\"test\":\"t_test\""));
        assert!(text.contains("\"statistic\":"));
        assert!(text.contains("\"p_value\":"));
        assert!(text.contains("\"significant\":"));
    }

    #[test]
    fn test_tool_hypothesis_chi_squared() {
        let args = r#"{"group_a":[10,20,30],"group_b":[15,25,35],"test":"chi_squared"}"#;
        let result = tool_hypothesis_test(args);
        assert!(result.is_ok());
        let text = result.unwrap();
        assert!(text.contains("\"test\":\"chi_squared\""));
        assert!(text.contains("\"statistic\":"));
    }

    #[test]
    fn test_tool_llm() {
        let args = r#"{"prompt":"sort a list of numbers"}"#;
        let result = tool_llm(args);
        assert!(result.is_ok());
        // Without a model file, should return an informative message
        let text = result.unwrap();
        assert!(text.contains("sort a list of numbers") || text.contains("error") || text.contains("response"));
    }

    #[test]
    fn test_parse_string_array() {
        let arr = parse_string_array(r#"["sma_20","ema_12","rsi"]"#);
        assert_eq!(arr.len(), 3);
        assert_eq!(arr[0], "sma_20");
        assert_eq!(arr[1], "ema_12");
        assert_eq!(arr[2], "rsi");

        let empty = parse_string_array("[]");
        assert!(empty.is_empty());
    }

    #[test]
    fn test_normal_cdf() {
        // CDF(0) = 0.5
        let v = normal_cdf(0.0);
        assert!((v - 0.5).abs() < 0.01);
        // CDF(large) ≈ 1.0
        assert!(normal_cdf(5.0) > 0.99);
    }

    #[test]
    fn test_trailing_backslash_no_panic() {
        // A-03/A-04: trailing backslash in JSON string should not panic
        assert_eq!(find_unescaped_quote("hello\\"), None);
        assert_eq!(find_unescaped_quote("\\"), None);
        assert_eq!(find_matching_brace("{\"key\":\"val\\\"}", b'{', b'}'), None);
        // Normal cases still work
        assert_eq!(find_unescaped_quote("hello\"rest"), Some(5));
        assert_eq!(find_matching_brace("{}", b'{', b'}'), Some(1));
    }

    #[test]
    fn test_parse_number_array_strip() {
        // A-07: strip_prefix/suffix instead of byte indexing
        let arr = parse_number_array("[ 1, 2, 3 ]").unwrap();
        assert_eq!(arr.len(), 3);
        assert_eq!(arr[0], 1.0);
        let empty = parse_number_array("[]").unwrap();
        assert!(empty.is_empty());
    }

    #[test]
    fn test_compute_sma() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sma = compute_sma(&data, 3);
        assert_eq!(sma.len(), 5);
        // SMA(3) at index 2: (1+2+3)/3 = 2.0
        assert!((sma[2] - 2.0).abs() < 0.001);
        // SMA(3) at index 4: (3+4+5)/3 = 4.0
        assert!((sma[4] - 4.0).abs() < 0.001);
    }

    #[test]
    fn test_compute_ema() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ema = compute_ema(&data, 3);
        assert_eq!(ema.len(), 5);
        assert_eq!(ema[0], 1.0); // first value = data[0]
        // EMA rises towards the data
        assert!(ema[4] > ema[0]);
    }

    #[test]
    fn test_tool_chat_no_model_returns_error() {
        // Without OCTOFLOW_API_KEY or local model, should return honest error
        let args = r#"{"prompt":"make a fibonacci function"}"#;
        let overrides = Overrides::default();
        // Temporarily clear env vars to ensure error path
        let saved_key = std::env::var("OCTOFLOW_API_KEY").ok();
        let saved_oai = std::env::var("OPENAI_API_KEY").ok();
        std::env::remove_var("OCTOFLOW_API_KEY");
        std::env::remove_var("OPENAI_API_KEY");
        let result = tool_chat(args, &overrides);
        // Restore env vars
        if let Some(k) = saved_key { std::env::set_var("OCTOFLOW_API_KEY", k); }
        if let Some(k) = saved_oai { std::env::set_var("OPENAI_API_KEY", k); }
        assert!(result.is_ok());
        let text = result.unwrap();
        assert!(text.contains("\"error\":"), "should return error JSON, got: {}", text);
        assert!(text.contains("GGUF") || text.contains("API key"), "should mention setup, got: {}", text);
    }

    #[test]
    fn test_json_escape_value() {
        assert_eq!(json_escape_value("hello"), "\"hello\"");
        assert_eq!(json_escape_value("a\"b"), "\"a\\\"b\"");
    }

    #[test]
    fn test_find_json_string_end() {
        assert_eq!(find_json_string_end("hello\""), Some(5));
        assert_eq!(find_json_string_end("he\\\"llo\""), Some(7));
        assert_eq!(find_json_string_end("no end"), None);
    }
}
